"""Fixed sentence indexes and bipolar, open-world meaning coordinates.

The flat complement is [for_0 ... for_K-1 | against_0 ... against_K-1].
Its scale is evidence, never part of a form-direction normalization.
"""
import hashlib
from functools import lru_cache

import numpy as np
import torch


from Occurrence import sentence_key


def exchange(value):
    if value.shape[-1] % 2:
        raise ValueError('meaning requires an even number of pole coordinates')
    positive, negative = value.chunk(2, -1)
    return torch.cat((negative, positive), -1)


def compose(name, left, right=None):
    if name == 'not':
        return exchange(left)
    if name == 'sum':
        return (left + right) * .5
    positive, negative = left.chunk(2, -1)
    other_positive, other_negative = right.chunk(2, -1)
    if name in ('conjunction', 'intersection'):
        return torch.cat((torch.minimum(positive, other_positive),
                          torch.maximum(negative, other_negative)), -1)
    if name in ('disjunction', 'union'):
        return torch.cat((torch.maximum(positive, other_positive),
                          torch.minimum(negative, other_negative)), -1)
    raise ValueError('no declared bilattice rule for ' + name)


@lru_cache(maxsize=32768)
def identity_bits(content_key, pairs, ones):
    """Private generator: no Python, NumPy or Torch global RNG is consumed."""
    if not isinstance(content_key, bytes) or not 0 < ones <= pairs:
        raise ValueError('sentence identity needs a byte key and 0 < s <= K')
    seed = int.from_bytes(hashlib.sha256(content_key).digest()[:8], 'little')
    return tuple(map(int, np.random.default_rng(seed).permutation(pairs)[:ones]))


def identity_code(content_key, pairs, ones, *, like=None):
    result = torch.zeros(pairs) if like is None else like.new_zeros(pairs)
    result[list(identity_bits(content_key, pairs, ones))] = 1
    return result


def membership(pole, code):
    """Approximate code membership; the postings remain the exact authority."""
    return ((pole > 0) | (code == 0)).all(-1)


def certificate(meanings, codes, extents):
    """Exhaust every ordered word pair (including self) and every sentence.

    For order-zero positive evidence, group words by their coverage of a
    sentence's s identity bits. There are at most 2**s groups. Counting all
    pairs of groups is exactly equivalent to materializing V*V*S roots,
    including Bloom false positives; no vocabulary or row is sampled.
    """
    values = np.asarray(meanings, dtype=np.float32)
    codes = np.asarray(codes, dtype=np.float32)
    vocabulary, width = values.shape
    pairs = width // 2
    if width != 2 * codes.shape[1] or np.any(values < 0) or np.any(values[:, pairs:]):
        raise ValueError('the static certificate requires order-zero bipolar evidence')
    if len(extents) != vocabulary:
        raise ValueError('one posting extent is required per word')
    support = values[:, :pairs] > 0
    truths = np.zeros(len(codes), dtype=np.int64)
    for word, extent in enumerate(extents):
        for row in extent:
            truths[row] += 1
            if not support[word, codes[row] > 0].all():
                raise ValueError('a posted sentence is missing from its word code')
    errors = dict(conjunction=0, disjunction=0, not_against=0, and_not_against=0)
    for row, code in enumerate(codes):
        bits = np.flatnonzero(code)
        if not len(bits) or len(bits) > 12:
            raise ValueError('certificate requires 1..12 identity bits per sentence')
        masks = np.zeros(vocabulary, dtype=np.int32)
        for position, bit in enumerate(bits):
            masks |= support[:, bit].astype(np.int32) << position
        size = 1 << len(bits)
        counts = np.bincount(masks, minlength=size).astype(np.int64)
        supersets = counts.copy()
        for bit in range(len(bits)):
            step = 1 << bit
            block = supersets.reshape(-1, 2 * step)
            block[:, :step] += block[:, step:]
        full = size - 1
        present, exact = int(counts[full]), int(truths[row])
        errors['conjunction'] += present * present - exact * exact
        predicted_union = int(counts @ supersets[full ^ np.arange(size)])
        errors['disjunction'] += predicted_union - (vocabulary**2 - (vocabulary-exact)**2)
        errors['not_against'] += (present-exact) * vocabulary
        errors['and_not_against'] += (present-exact) * vocabulary
    checks = vocabulary * vocabulary * len(codes)
    return dict(words=vocabulary, sentences=len(codes), pairs=pairs,
        ordered_pairs=True, self_pairs=True, checks_per_case=checks,
        distinct_meanings=len({row.tobytes() for row in values}),
        errors={**errors, 'not_for': 0, 'and_not_for': 0},
        false_membership_rates={key: value/checks if checks else 0. for key, value in errors.items()},
        false_negatives=0, exact=not any(errors.values()),
        method='exhaustive identity-bit coverage histogram, all word pairs and rows')
