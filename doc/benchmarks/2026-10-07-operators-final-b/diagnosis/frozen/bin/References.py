"""Identity symbol encodings; definition lookup lives on the truth store.

A binding contains both native identities. One word may have several
objects; interpretation selects one. Translation is an indexed lookup,
not a taxonomy traversal. Extents are cached for each complete binding.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from Layers import EPS_LOG, Ops

# Salt for deterministic symbol-code generation (arbitrary, fixed).
_SYMBOL_SEED_SALT = 0x5EED


def address_code(addresses, width, *, like):
    """A tensor-native row name, including temporary open-journal addresses.

    It has no lexical or compositional geometry. Reversal reads the named
    row's slots. Integer mixing consumes no random generator.
    """
    addresses = torch.as_tensor(addresses, device=like.device, dtype=torch.long)
    bit = torch.arange(int(width), device=like.device).remainder(64)
    mixed = addresses * 6364136223846793005 + 1442695040888963407
    mixed = mixed.bitwise_xor(mixed >> 29)
    return (((mixed[..., None] >> bit).bitwise_and(1)).to(like) * 2 - 1) / max(1, int(width))**.5


def symbol_code(index: int, n_what: int, n_where: int = 2, n_when: int = 2,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The in-loop representation of a symbol: "approximately the index".

    A deterministic, identifier-like what-code (quasi-orthogonal unit
    vector seeded by the table index) plus the standard positional
    bands, BOTH ZERO — the zero-band signature that marks symbolic
    occurrences as existing outside space (spec §6/§7; no real object
    has zeroed ``.where``/``.when``). Identity, not similarity, is the
    code's duty; realizing the index as a vector is a concession to the
    loop so symbol recall is an ordinary codebook matmul. Widths are
    per-model config (MM_20M: ``4+2+2``, total 8, matching its STM=8).
    """
    # Seeded draw on CPU (device-agnostic bytes); wiring moves it to the model device.
    g = torch.Generator(device="cpu").manual_seed(_SYMBOL_SEED_SALT + int(index))
    what = torch.randn(int(n_what), generator=g, dtype=dtype, device="cpu")
    what = what / (what.norm() + 1e-12)
    bands = torch.zeros(int(n_where) + int(n_when), dtype=dtype, device="cpu")
    return torch.cat([what, bands])
