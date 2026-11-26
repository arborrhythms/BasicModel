// Add the journal's binary64 increment before rounding to the stored activity.
#include <metal_stdlib>
using namespace metal;
#include "significand.metal"

kernel void activity_update(device uint* result, device const uint* current,
                            device const ulong* increments, uint i [[thread_position_in_grid]]) {
    ulong bits = increments[i];
    ulong fraction = bits & (one52 - 1);
    int exponent = int(bits >> 52);
    Number delta = {0, 0, false};
    if (exponent) {
        delta = {one52 | fraction, exponent - 1023, false};
    } else if (fraction) {
        int leading = 63 - int(clz(fraction));
        delta = {fraction << (52 - leading), leading - 1074, false};
    }
    Number total = add(from_bits(current[i]), delta);
    result[i] = is_nan(total) || (!total.negative && total.m && total.e >= 0)
              ? 0x3f800000u : to_bits(total);
}
