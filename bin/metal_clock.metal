// Rotate each rung in binary64, then round the four stored wires to float32.
#include <metal_stdlib>
using namespace metal;
#include "significand.metal"

struct Number {
    ulong m;  // Unsigned 53-bit significand, or zero.
    int e;    // Leading bit's exponent.
    bool negative;
};

// Finite float32 inputs and their products fit comfortably in the extended
// exponent range. Reserve one exponent for IEEE infinity and NaN.
constant int special = 2048;
inline Number not_a_number() { return {one52 + 1, special, false}; }
inline bool is_nan(Number x) { return x.e == special && x.m != one52; }

inline Number from_bits(uint bits) {
    uint m = bits & 0x7fffffu;
    int exp = int((bits >> 23) & 255);
    if (exp == 255) return {one52 + ulong(m != 0), special, bool(bits >> 31)};
    if (exp) return {ulong(m | 0x800000u) << 29, exp - 127, bool(bits >> 31)};
    if (!m) return {0, 0, bool(bits >> 31)};
    int top = 31 - int(clz(m));
    return {ulong(m) << (52 - top), top - 149, bool(bits >> 31)};
}

inline uint to_bits(Number n) {
    uint sign = uint(n.negative) << 31;
    if (n.e == special) return is_nan(n) ? 0x7fc00000u : sign | 0x7f800000u;
    if (!n.m) return sign;
    if (n.e > 127) return sign | 0x7f800000u;
    if (n.e < -126) {
        int shift = 29 + (-126 - n.e);
        return sign | (shift <= 53 ? uint(round_right(n.m, uint(shift))) : 0);
    }
    return sign | ((uint(n.e + 127) << 23) + uint(round_right(n.m, 29)) - 0x800000u);
}

inline Number scale(Number n, int exponent) { if (n.e != special) n.e += exponent; return n; }
inline Number neg(Number n) { n.negative = !n.negative; return n; }
inline Number mul(Number a, Number b) {
    if (a.e == special || b.e == special) {
        if (is_nan(a) || is_nan(b) || !a.m || !b.m) return not_a_number();
        return {one52, special, a.negative != b.negative};
    }
    int exp = a.e + b.e;
    ulong m = a.m && b.m ? multiply53(a.m, b.m, exp) : 0;
    return {m, exp, a.negative != b.negative};
}
inline Number divide(Number a, Number b) {
    if (a.e == special) return a;
    int exp = a.e - b.e;
    ulong m = a.m ? divide53(a.m, b.m, exp) : 0;
    return {m, exp, a.negative != b.negative};
}

// The low three bits carry guard, round and sticky through alignment.
inline Number add(Number a, Number b) {
    if (a.e == special || b.e == special) {
        if (is_nan(a) || is_nan(b) || (a.e == b.e && a.negative != b.negative)) return not_a_number();
        return a.e == special ? a : b;
    }
    if (!a.m) return b;
    if (!b.m) return a;
    if (a.e < b.e || (a.e == b.e && a.m < b.m)) {
        Number swap = a; a = b; b = swap;
    }
    int shift = a.e - b.e;
    ulong small = b.m << 3;
    if (shift >= 64) small = 1;
    else if (shift) small = (small >> shift) | ulong((small << (64 - shift)) != 0);
    ulong big = a.m << 3;
    ulong combined = a.negative == b.negative ? big + small : big - small;
    if (!combined) return {0, 0, false};
    int leading = (combined >> 32) ? 63 - int(clz(uint(combined >> 32))) : 31 - int(clz(uint(combined)));
    int exp = a.e + leading - 55;
    if (leading < 55) combined <<= (55 - leading);
    ulong rounded = round_right(combined, uint(max(leading - 52, 3)));
    rounded = normalize53(rounded, exp);
    return {rounded, exp, a.negative};
}

inline Number root(Number x) {
    if (!x.m) return x;
    // Integer square root of m << (52 or 53). Bring down two bits per
    // iteration. The remainder and the next trial divisor fit in one limb.
    uint shift = 52 + uint(x.e & 1);
    ulong high = x.m >> (64 - shift), low = x.m << shift;
    ulong q = 0, rem = 0;
    for (int pair = 104; pair >= 0; pair -= 2) {
        ulong digit = pair >= 64 ? high >> (pair - 64) : low >> pair;
        rem = (rem << 2) | (digit & 3);
        ulong divisor = (q << 2) | 1;
        q <<= 1;
        if (rem >= divisor) { rem -= divisor; ++q; }
    }
    int exp = (x.e - int(x.e & 1)) / 2;
    q = normalize53(q + ulong(rem > q), exp);
    return {q, exp, false};
}

constant Number one = {one52, 0, false};
inline Number centered(uint bits) { return add(scale(from_bits(bits), 1), neg(one)); }
inline uint stored(Number n) { return to_bits(scale(add(n, one), -1)); }

struct Rotation {
    Number ps, pc, ss, sc, s, c, length, y, z;
};

inline Rotation rotate(device const uint* present, device const uint* stamp) {
    Rotation r;
    r.ps = centered(present[0]); r.pc = centered(present[1]);
    r.ss = centered(stamp[0]); r.sc = centered(stamp[1]);
    r.s = add(mul(r.ps, r.sc), neg(mul(r.pc, r.ss)));
    r.c = add(mul(r.pc, r.sc), mul(r.ps, r.ss));
    r.length = root(add(mul(r.s, r.s), mul(r.c, r.c)));
    r.y = r.length.m ? divide(r.s, r.length) : r.s;
    r.z = r.length.m ? divide(r.c, r.length) : r.c;
    return r;
}

// These four binary64 constants come from the same encoder as the CPU read.
constant Number turn_sin[2] = {TURN_SIN};
constant Number turn_cos[2] = {TURN_COS};

inline bool has_stamp(device const uint* stamp) {
    return ((stamp[0] | stamp[1] | stamp[2] | stamp[3]) & 0x7fffffffu) != 0;
}
inline bool finite_pair(device const uint* pair) {
    return (pair[0] & 0x7fffffffu) < 0x7f800000u && (pair[1] & 0x7fffffffu) < 0x7f800000u;
}

kernel void clock_code(device uint* out, device const uint* present, device const uint* stamp,
                       device uint* saved, constant uint& size, uint i [[thread_position_in_grid]]) {
    uint k = 2 * i, rung = i & 1;
    if (size) {
        saved[k] = present[k]; saved[k+1] = present[k+1];
        saved[size+k] = stamp[k]; saved[size+k+1] = stamp[k+1];
    }
    if (!has_stamp(stamp + (k & ~3u))) { out[k] = out[k+1] = 0; return; }
    if (!finite_pair(stamp+k) || !finite_pair(present+k)) { out[k] = out[k+1] = 0x7fc00000u; return; }
    Rotation r = rotate(present+k, stamp+k);
    Number sine = turn_sin[rung], cosine = turn_cos[rung];
    out[k]   = stored(scale(add(add(r.y, mul(r.y, cosine)), mul(r.z, sine)), -1));
    out[k+1] = stored(scale(add(add(r.z, mul(r.z, cosine)), neg(mul(r.y, sine))), -1));
}

// Differentiate the original operations in binary64. A zero-length rotation
// has the same undefined sqrt derivative as the reference autograd graph.
kernel void clock_stamp_grad(device uint* out, device const uint* present, device const uint* stamp,
                             device const uint* grad, uint i [[thread_position_in_grid]]) {
    uint k = 2 * i, rung = i & 1;
    Rotation r = rotate(present+k, stamp+k);
    if (!r.length.m || !finite_pair(stamp+k) || !finite_pair(present+k)) {
        out[k] = out[k+1] = 0x7fc00000u; return;
    }
    if (!has_stamp(stamp + (k & ~3u))) { out[k] = out[k+1] = 0; return; }
    Number u = scale(from_bits(grad[k]), -2), v = scale(from_bits(grad[k+1]), -2);
    Number sine = turn_sin[rung], cosine = turn_cos[rung];
    Number dy = add(add(neg(mul(v, sine)), u), mul(u, cosine));
    Number dz = add(add(mul(v, cosine), v), mul(u, sine));
    Number dlength = neg(add(mul(dy, divide(r.y, r.length)), mul(dz, divide(r.z, r.length))));
    Number dsquare = divide(dlength, scale(r.length, 1));
    Number ds = add(divide(dy, r.length), scale(mul(dsquare, r.s), 1));
    Number dc = add(divide(dz, r.length), scale(mul(dsquare, r.c), 1));
    out[k]   = to_bits(scale(add(mul(dc, r.ps), neg(mul(ds, r.pc))), 1));
    out[k+1] = to_bits(scale(add(mul(dc, r.pc), mul(ds, r.ps)), 1));
}
