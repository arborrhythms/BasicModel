"""Stage 6 of doc/plans/MeronomyPlan.md: the word/object binding table.

MeronomySpec §6 (rev 2026-06-11) / §10.8 / §10.9: full rows only,
word-keyed (deref indexed; ref = unindexed object-side search),
append-only and gate-licensed; symbols are atomic (quasi-orthogonal
zero-banded codes, "approximately the index"); mint-time dominance
makes search work (`A ⊑ σ(A,B)`); ⊥ extents cached as
definable-but-empty and ⊤ saturation detectable.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from References import ReferenceTable, symbol_code
from Layers import PiLayer2, SigmaLayer2, Ops

D = 4


# ---------------------------------------------------------------------------
# Full rows only; append-only; gate-licensed.
# ---------------------------------------------------------------------------

def test_full_rows_only():
    t = ReferenceTable()
    with pytest.raises(ValueError):
        t.bind(word=None, obj=3, licensed=True)
    with pytest.raises(ValueError):
        t.bind(word=7, obj=None, licensed=True)
    t.bind(word=7, obj=3, licensed=True)
    assert t.deref(7) == 3
    assert len(t) == 1


def test_append_only():
    t = ReferenceTable()
    t.bind(word=1, obj=10, licensed=True)
    with pytest.raises(ValueError):
        t.bind(word=1, obj=11, licensed=True)   # re-binding refused
    t.bind(word=2, obj=10, licensed=True)       # synonyms are fine
    assert t.words() == [1, 2], "word-sorted table"


def test_gate_license_required():
    t = ReferenceTable()
    with pytest.raises(RuntimeError):
        t.bind(word=4, obj=2, licensed=False)
    assert len(t) == 0, "naming never happens without the gate"


def test_unknown_word_is_a_query_outcome():
    t = ReferenceTable()
    assert t.deref(99) is None
    assert 99 not in t


# ---------------------------------------------------------------------------
# deref indexed / ref unindexed — the API audit.
# ---------------------------------------------------------------------------

def test_no_reverse_index_anywhere():
    t = ReferenceTable()
    for w, o in ((1, 5), (2, 6), (3, 5)):
        t.bind(word=w, obj=o, licensed=True)
    # The ONLY id→id mapping on the instance is the word-keyed one;
    # an object stores nothing about its names.
    maps = {k: v for k, v in vars(t).items()
            if isinstance(v, dict) and v
            and all(isinstance(x, int) for x in v.values())}
    assert set(maps) == {"_by_word"}, (
        f"reverse/object-keyed index found: {set(maps) - {'_by_word'}}")
    assert set(maps["_by_word"].keys()) == {1, 2, 3}, "keys are words"
    # And no method offers an indexed object→word lookup.
    for name in ("word_of", "deref_object", "by_object", "names_of"):
        assert not hasattr(t, name)


def test_search_is_object_side_scan_by_dominance():
    t = ReferenceTable()
    rows = torch.tensor([
        [0.9, 0.8, 0.7, 0.9],    # obj 0: dominates the probe
        [0.2, 0.9, 0.9, 0.9],    # obj 1: fails on dim 0
        [0.9, 0.9, 0.9, 0.9],    # obj 2: dominates
    ])
    t.bind(word=10, obj=0, licensed=True)
    t.bind(word=11, obj=1, licensed=True)
    t.bind(word=12, obj=2, licensed=True)
    probe = torch.tensor([0.5, 0.5, 0.5, 0.5])
    hits = t.search(probe, rows)
    assert hits == [(10, 0), (12, 2)], (
        "ref returns the bindings whose objects dominate the probe")
    # Nameless region: nothing dominates -> empty result (a query
    # outcome -- tip-of-the-tongue is a failed object-side search).
    assert t.search(torch.ones(4) * 0.95, rows) == []


# ---------------------------------------------------------------------------
# Symbol codes: atomic, zero-banded, approximately the index.
# ---------------------------------------------------------------------------

def test_symbol_code_shape_and_zero_bands():
    c = symbol_code(0, n_what=4, n_where=2, n_when=2)
    assert c.shape == (8,), "MM_20M idiom: 4+2+2 = 8"
    assert (c[4:] == 0).all(), "zero-band signature: symbols are not in space"
    assert abs(c[:4].norm().item() - 1.0) < 1e-6


def test_symbol_code_deterministic_identity():
    a1 = symbol_code(3, n_what=8)
    a2 = symbol_code(3, n_what=8)
    b = symbol_code(4, n_what=8)
    assert torch.equal(a1, a2), "the code IS (approximately) the index"
    assert not torch.allclose(a1, b), "distinct indices, distinct codes"


def test_symbol_codes_are_mereologically_inert():
    # Quasi-orthogonal signed codes: pairwise incomparable under the
    # dominance order (seed-pinned configuration), so symbols carry no
    # size relations -- atoms, outside the meronomy.
    codes = [symbol_code(i, n_what=8)[:8] for i in range(12)]
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            assert not bool(Ops.partOf(codes[i], codes[j])), (
                f"symbol {i} ⊑ symbol {j}: codes must not be size-related")


# ---------------------------------------------------------------------------
# Mint-time dominance is what makes ref-search work (§10.9).
# ---------------------------------------------------------------------------

def test_minted_whole_is_searchable_by_its_parts():
    torch.manual_seed(1)
    sig = SigmaLayer2(2 * D, D, blocks=2)
    with torch.no_grad():
        for p in sig.parameters():
            p.uniform_(-5.0, -0.5)
    A = torch.rand(1, D) * 0.5 + 0.3
    B = torch.rand(1, D) * 0.5 + 0.3
    whole = sig.compose(A, B)[0]             # whole ≽ max(A, B) ≽ A
    rows = torch.stack([A[0] * 0.5, whole])  # obj 1 = the minted whole
    t = ReferenceTable()
    t.bind(word=42, obj=1, licensed=True)
    assert (42, 1) in t.search(A[0], rows), (
        "A ⊑ σ(A,B) at mint: the part finds its whole's name by search")
    pi = PiLayer2(2 * D, D, blocks=2)
    with torch.no_grad():
        for p in pi.parameters():
            p.uniform_(-5.0, -0.5)
    part = pi.compose(A, B)[0]               # part ≼ min(A, B) ≼ A
    assert bool(Ops.partOf(part, A[0])), "π(A,B) ⊑ A at mint"


# ---------------------------------------------------------------------------
# Gauge orientation at bind; evaluate-before-cache degeneracies.
# ---------------------------------------------------------------------------

def test_bind_gauge_orients_object_row():
    t = ReferenceTable()
    torch.manual_seed(2)
    ref = torch.rand(D)
    u = -ref / ref.norm()                    # anti-aligned representative
    oriented = t.bind(word=5, obj=0, licensed=True,
                      object_row=u, referent=ref)
    assert oriented is not None
    assert (oriented * ref).sum() >= 0, "+u oriented toward the referent"
    assert torch.allclose(oriented, -u), "the flip, not a rescale"


def test_bottom_extent_cached_definable_but_empty():
    t = ReferenceTable()
    t.bind(word=6, obj=1, licensed=True, extent=torch.zeros(D))
    assert t.is_empty_extent(6), "⊥: constraints annihilate"
    assert not t.is_saturated(6)
    assert torch.equal(t.extent_of(6), torch.zeros(D)), "queryable"


def test_top_saturation_detected():
    t = ReferenceTable()
    t.bind(word=7, obj=2, licensed=True, extent=torch.ones(D))
    assert t.is_saturated(7), (
        "⊤ hazard: gathers everything, distinguishes nothing")
    assert not t.is_empty_extent(7)
    t.bind(word=8, obj=3, licensed=True, extent=torch.rand(D) * 0.5 + 0.2)
    assert not t.is_saturated(8) and not t.is_empty_extent(8)
