// Rotate each rung in binary64, then round the four stored wires to float32.
#include <metal_stdlib>
using namespace metal;
#include "significand.metal"

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
