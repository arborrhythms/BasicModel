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
