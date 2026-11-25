// One pair's binary64 co-occurrence and proximal update per thread.
#include <metal_stdlib>
using namespace metal;
#include "significand.metal"

inline Number unpack_stat(long bits) {
    if (!bits) return {0, 0, false};
    ulong magnitude = ulong(abs(bits));
    return {one52 | (magnitude & (one52 - 1)), int(magnitude >> 52) - 1023, bits < 0};
}

inline long pack_stat(Number n) {
    if (!n.m) return 0;
    long bits = long((ulong(n.e + 1023) << 52) | (n.m - one52));
    return n.negative ? -bits : bits;
}

kernel void pair_statistics(device long* out, device const long* old,
                            device const uint* observation, device const long* value,
                            device const long* gathered, device const long* steps,
                            device const long* penalty, uint i [[thread_position_in_grid]]) {
    for (uint j = 0; j < 3; ++j) {
        out[5*i+j] = pack_stat(add(unpack_stat(old[3*i+j]), from_bits(observation[3*i+j])));
    }
    long larger = max(out[5*i+1], out[5*i+2]);
    Number meet = unpack_stat(out[5*i]);
    Number ratio = larger ? divide(meet, unpack_stat(larger)) : meet;
    Number unit = {one52, 0, false};
    Number worth = add(scale(ratio, 1), neg(unit));
    if (worth.negative) worth = {0, 0, false};
    Number gradient = add(unpack_stat(value[i]), neg(worth));
    Number total = add(unpack_stat(gathered[i]), gradient);
    out[5*i+3] = pack_stat(total);
    ulong count = ulong(max(steps[i], 1l));
    int exponent = 63 - int(clz(count));
    ulong m = exponent <= 52 ? count << (52 - exponent) : round_right(count, uint(exponent - 52));
    m = normalize53(m, exponent);
    Number divisor = {m, exponent, false};
    Number mean = divide(total, divisor);
    Number magnitude = {mean.m, mean.e, false};
    Number size = add(magnitude, neg(unpack_stat(penalty[0])));
    out[5*i+4] = size.negative ? 0 : pack_stat(mean.negative ? size : neg(size));
}
