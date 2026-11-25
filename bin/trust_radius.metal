// Reproduce float32(double(agreement) / double(views) * gain) on Metal.
// Each intermediate rounds to nearest, ties to even, as in ConceptTrust.
#include <metal_stdlib>
using namespace metal;

#include "significand.metal"

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
