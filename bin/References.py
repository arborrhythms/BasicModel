"""The word/object binding table (MeronomySpec §6 rev 2026-06-11; Stage 6).

The table is the lexicon-shaped organ that links the two towers: rows
are FULL bindings ``(word: WS row id, object: PS row id)`` and nothing
else. Symbols — the carriers of the bindings — are zero-dimensional
atoms outside both towers' size orders (the atomicity argument: a truly
arbitrary word↔object association cannot be mereologically coupled),
persisted only as these rows; composition over symbols lives in the
towers, never here. The table is mereologically inert.

Design laws (spec §6, §10.8):

* **Full rows only.** A row exists iff both sides are bound. There are
  no half-bindings anywhere: an unbound word is just an WS tower code,
  a nameless concept just a PS tower code; bound-ness is never stored,
  only discovered by query (a lookup miss IS the "unknown word" state).
* **Word-keyed — the word is the reference for the object.**
  ``deref(word)`` is the indexed, cheap direction (the serial forward
  shift's mechanism). ``search(object probe)`` has NO index by design:
  an object stores nothing about its names; recall is a scan of the
  object side (tip-of-the-tongue is a failed object-side search).
* **Append-only, gate-licensed.** Only the interpret-as-word gate
  (search-then-mint, Stage 7) may create rows; naming follows
  demonstrated reuse, never first sight. Re-binding a bound word is an
  error.
* **Evaluate-before-cache.** A binding may cache its evaluated extent;
  ``𝟘``-valued extents are definable-but-empty (the ⊥ degeneracy,
  queryable), and ``𝟙``-saturated extents are flagged (the ⊤ hazard —
  a concept that gathers everything and distinguishes nothing).

Module name per plan §4 item 5 (rows ARE references).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from Layers import EPS_LOG, Ops

# Salt for deterministic symbol-code generation (arbitrary, fixed).
_SYMBOL_SEED_SALT = 0x5EED


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
    g = torch.Generator().manual_seed(_SYMBOL_SEED_SALT + int(index))
    what = torch.randn(int(n_what), generator=g, dtype=dtype)
    what = what / (what.norm() + 1e-12)
    bands = torch.zeros(int(n_where) + int(n_when), dtype=dtype)
    return torch.cat([what, bands])


class ReferenceTable:
    """Word-keyed, append-only store of full word/object bindings."""

    def __init__(self):
        # THE one mapping: word id -> object id. Deliberately the only
        # index in the class — the reverse direction must stay a search
        # (spec §6 "storage is one-way"; §10.8 API audit).
        self._by_word: Dict[int, int] = {}
        # Evaluate-before-cache: per-word cached extent + degeneracy
        # flags (⊥ definable-but-empty; ⊤ saturated).
        self._extent_cache: Dict[int, torch.Tensor] = {}
        self._empty_extent: Dict[int, bool] = {}
        self._saturated: Dict[int, bool] = {}

    # -- Binding (the gate's write) ------------------------------------

    def bind(self, word: int, obj: int, licensed: bool = False,
             object_row: Optional[torch.Tensor] = None,
             referent: Optional[torch.Tensor] = None,
             extent: Optional[torch.Tensor] = None
             ) -> Optional[torch.Tensor]:
        """Create one full binding. Returns the gauge-oriented object
        row when ``object_row`` (and a positive ``referent``) are
        supplied — the caller owns writing it back to the PS codebook;
        the table stores indices only.

        ``licensed`` must be True: only the interpret-as-word gate may
        name (search-then-mint, Stage 7). ``word``/``obj`` must both be
        present (full rows only) and ``word`` unbound (append-only).
        ``extent``, when given, is evaluated-before-cached with ⊥/⊤
        degeneracy detection.
        """
        if not licensed:
            raise RuntimeError(
                "ReferenceTable.bind: naming requires the gate license "
                "(search-then-mint; never first sight).")
        if word is None or obj is None:
            raise ValueError(
                "ReferenceTable.bind: full rows only — both word and "
                "object must be bound (there are no half-bindings; "
                "unbound words / nameless concepts are tower codes, "
                "not table states).")
        word = int(word)
        obj = int(obj)
        if word in self._by_word:
            raise ValueError(
                f"ReferenceTable.bind: word {word} is already bound "
                f"(append-only; re-binding is not a write).")
        self._by_word[word] = obj
        oriented = None
        if object_row is not None and referent is not None:
            # Mint-time gauge fixing (spec §3; Stage 5): +u toward
            # agreement with the positive referent. Lazy import keeps
            # this module Spaces-independent at import time.
            from Spaces import gauge_orient
            oriented = gauge_orient(object_row, referent)
        if extent is not None:
            ext = extent.detach().clone()
            self._extent_cache[word] = ext
            self._empty_extent[word] = bool((ext <= 2 * EPS_LOG).all())
            self._saturated[word] = bool((ext >= 1.0 - 2 * EPS_LOG).all())
        return oriented

    # -- deref: the indexed, cheap direction ----------------------------

    def deref(self, word: int) -> Optional[int]:
        """word → object id, or None on miss (the "unknown word" state
        is this query outcome, not a stored fact)."""
        return self._by_word.get(int(word))

    # -- ref: the unindexed direction (recall is search) ----------------

    def search(self, probe: torch.Tensor, rows: torch.Tensor,
               tol: float = 1e-6) -> List[Tuple[int, int]]:
        """Find the names of an object: scan bound rows for objects
        DOMINATING the probe (``probe ⊑ rows[obj]`` elementwise, the
        mint-time theorem's signature). ``rows`` is the PS codebook
        prototype — the table holds indices only and walks the bindings
        linearly: there is deliberately no object→word index.
        """
        out: List[Tuple[int, int]] = []
        for word in sorted(self._by_word):
            obj = self._by_word[word]
            if obj >= rows.shape[0]:
                continue
            if bool(Ops.partOf(probe, rows[obj] + tol)):
                out.append((word, obj))
        return out

    # -- Degeneracy queries (evaluate-before-cache) ---------------------

    def extent_of(self, word: int) -> Optional[torch.Tensor]:
        """The cached evaluated extent for a bound word, or None."""
        return self._extent_cache.get(int(word))

    def is_empty_extent(self, word: int) -> bool:
        """⊥: definable-but-empty — the binding exists, its constraints
        annihilate. Queryable, per spec §8."""
        return self._empty_extent.get(int(word), False)

    def is_saturated(self, word: int) -> bool:
        """⊤ hazard: the extent saturates toward 𝟙 — gathers everything,
        distinguishes nothing (spec §6 side note)."""
        return self._saturated.get(int(word), False)

    # -- Table shape -----------------------------------------------------

    def words(self) -> List[int]:
        """Bound words, sorted (the table is word-sorted)."""
        return sorted(self._by_word)

    def bound_words(self) -> List[int]:
        """Reference rows of the WS/intent tower: the bound word ids.

        Consumed by the §6d update law (GrammarOpsPass): references are
        shaped by the serial pass only.
        """
        return sorted(self._by_word)

    def bound_objects(self) -> List[int]:
        """Reference rows of the PS/extent tower: the bound object ids
        (deduplicated — synonyms bind one object many times)."""
        return sorted(set(self._by_word.values()))

    def __len__(self) -> int:
        return len(self._by_word)

    def __contains__(self, word: int) -> bool:
        return int(word) in self._by_word
