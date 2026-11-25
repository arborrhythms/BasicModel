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

