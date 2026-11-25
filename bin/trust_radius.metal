// Reproduce float32(double(agreement) / double(views) * gain) on Metal.
// Each intermediate rounds to nearest, ties to even, as in ConceptTrust.
#include <metal_stdlib>
using namespace metal;

constant ulong one52 = 1ul << 52;
constant ulong one53 = 1ul << 53;

inline ulong round_right(ulong n, uint shift) {
    if (shift == 0) return n;
    ulong hi = n >> shift;
    ulong rest = n - (hi << shift);
    ulong half_way = 1ul << (shift - 1);
    return hi + ulong(rest > half_way || (rest == half_way && (hi & 1ul)));
}

inline ulong normalize53(ulong m, thread int& exponent) {
    if (m == one53) {
        ++exponent;
        m >>= 1;
    }
    return m;
}

// The remainder is below 2^53. Ten more quotient bits fit in one ulong.
inline ulong divide53(ulong numerator, ulong denominator, thread int& exponent) {
    if (numerator < denominator) {
        numerator <<= 1;
        --exponent;
    }
    ulong rest = numerator - denominator;
    ulong result = 1;
    for (uint bits = 0; bits < 52; bits += 10) {
        uint width = min(10u, 52u - bits);
        rest <<= width;
        ulong digit = rest / denominator;
        result = (result << width) + digit;
        rest -= digit * denominator;
    }
    rest <<= 1;
    result += ulong(rest > denominator || (rest == denominator && (result & 1ul)));
    return normalize53(result, exponent);
}

// A 53 x 53 bit product in base 2^26. Keep the discarded bits to round once.
inline ulong multiply53(ulong a, ulong b, thread int& exponent) {
    ulong mask = (1ul << 26) - 1;
    ulong a0 = a & mask, a1 = a >> 26, b0 = b & mask, b1 = b >> 26;
    ulong cross = a0 * b1 + a1 * b0;
    ulong low = a0 * b0 + ((cross & mask) << 26);
    ulong high = a1 * b1 + (cross >> 26) + (low >> 52);
    low &= one52 - 1;
    ulong half_way = one52 >> 1;
    if (high >= one53) {
        low += (high & 1ul) << 52;
        high >>= 1;
        half_way = one52;
        ++exponent;
    }
    high += ulong(low > half_way || (low == half_way && (high & 1ul)));
    return normalize53(high, exponent);
}

kernel void trust_radius(device uint* out, device const long* views,
                         device const uint* agreement, device const ulong* gains,
                         constant uint& last_gain, uint i [[thread_position_in_grid]]) {
    if (views[i] <= 0) { out[i] = 0; return; }
    uint a = agreement[i], fraction = a & 0x7fffffu, biased = (a >> 23) & 255u;
    if (biased == 255 || (a & 0x7fffffffu) == 0) { out[i] = a; return; }
    uint sign = a & 0x80000000u;
    int exponent = int(biased) - 127;
    ulong significand;
    if (biased == 0) {
        uint top = 31u - clz(fraction);
        exponent = int(top) - 149;
        significand = ulong(fraction) << (52u - top);
    } else {
        significand = ulong(fraction | 0x800000u) << 29;
    }
    ulong count = ulong(views[i]);
    uint upper = uint(count >> 32);
    int log_count = int(upper ? 63u - clz(upper) : 31u - clz(uint(count)));
    ulong divisor = log_count <= 52 ? count << (52 - log_count)
                                    : round_right(count, uint(log_count - 52));
    divisor = normalize53(divisor, log_count);
    exponent -= log_count;
    significand = divide53(significand, divisor, exponent);
    ulong gain = gains[min(count, ulong(last_gain))];
    if (gain != (1023ul << 52)) {
        exponent += int(gain >> 52) - 1023;
        significand = multiply53(significand, one52 | (gain & (one52 - 1)), exponent);
    }
    if (exponent < -126) {
        int shift = 29 + (-126 - exponent);
        out[i] = sign | (shift <= 53 ? uint(round_right(significand, uint(shift))) : 0u);
    } else {
        // A rounded carry into bit 23 also increments the biased exponent.
        uint rounded = uint(round_right(significand, 29));
        out[i] = sign | ((uint(exponent + 127) << 23) + rounded - 0x800000u);
    }
}
