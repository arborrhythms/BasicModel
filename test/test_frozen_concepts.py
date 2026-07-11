"""Frozen concepts (Alec 2026-07-11): freezing fixes a concept's
RELATIONAL STRUCTURE -- no FORMING of new connections, no FORGETTING of
existing ones, no WEIGHT drift on them. Content rows stay live. The
hard-coded 'reading' concept is the first frozen concept: desired, it
primes the word-isolating wholes (the staged span->slot->row chain).
cpu/eager.
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import functools
import sys

sys.path.insert(0, "bin")
import torch

from recon_bench import _build_model, _resolve_config


@functools.lru_cache(maxsize=None)
def _build(cfg):
    model, *_ = _build_model(_resolve_config(cfg))
    return model


def _fixture():
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=16, order=1)
    rA = _mint_row(cs, 1, 301)
    rB = _mint_row(cs, 1, 302)
    cs.add_concept_edge(rA, 3, weight=2.0)
    cs.add_concept_edge(rB, 5, weight=0.5)
    return cs, rA, rB


def test_frozen_no_forming():
    """No NEW connections form on a frozen concept's row."""
    cs, rA, _rB = _fixture()
    ly = cs._sparse_families(0)[1]
    cs.freeze_concept(301)
    nnz = ly.nnz
    assert cs.add_concept_edge(rA, 6) is None
    assert ly.nnz == nnz, "frozen definition must not grow"


def test_others_can_reference_frozen():
    """Other concepts may still BUILD ON a frozen concept (col allowed)."""
    cs, rA, rB = _fixture()
    ly = cs._sparse_families(0)[1]
    cs.freeze_concept(301)
    nnz = ly.nnz
    assert cs.add_concept_edge(rB, rA) is not None
    assert ly.nnz == nnz + 1, "references TO frozen concepts stay legal"


def test_frozen_weights_no_grad():
    """Backprop cannot move a frozen concept's edge values; unfrozen
    edges keep training."""
    cs, rA, rB = _fixture()
    ly = cs._sparse_families(0)[1]
    cs.freeze_concept(301)
    a_0 = torch.zeros(8, 1)
    a_0[3, 0] = 0.9
    a_0[5, 0] = 0.9
    what = torch.randn(16, 32)
    content, _a = cs.cs_forward_content(a_0, what)
    content.sum().backward()
    g = ly.values.grad
    assert g is not None
    idx_A = [i for (r, _c), i in ly._index.items() if r == rA]
    idx_B = [i for (r, _c), i in ly._index.items() if r == rB]
    assert idx_A and idx_B
    assert all(float(g[i]) == 0.0 for i in idx_A), "frozen weights: no drift"
    assert any(float(g[i]) != 0.0 for i in idx_B), "unfrozen weights train"


def test_frozen_no_forgetting():
    """retire_concept is a no-op on frozen concepts (no forgetting)."""
    cs, rA, _rB = _fixture()
    cs.freeze_concept(301)
    cs.retire_concept(301)
    assert cs._csw_row_of(301) == rA, "frozen concept must survive retire"


def test_mint_frozen_idempotent():
    """mint_frozen_concept: stable named handle, frozen, idempotent."""
    from test_cs_sparse_weights import _cs
    cs = _cs(nS=16, order=1)
    c1 = cs.mint_frozen_concept("reading")
    c2 = cs.mint_frozen_concept("reading")
    assert c1 == c2 and cs.is_frozen(c1)
    assert cs._csw_row_of(c1) is not None, "snap row reserved"


def test_set_reading_primes_the_concept():
    """set_reading desires the frozen 'reading' concept on the CS surface;
    set_reading(False) clears the sustained desire (decay fades the rest)."""
    m = _build("data/MM_sparse_concept.xml")
    cs0 = m.conceptualSpaces[0]
    m.relevance_on = True
    try:
        cid = m.set_reading(True)
        assert cid is not None and cs0.is_frozen(cid)
        row = cs0._csw_row_of(cid)
        b = cs0.priming_weights()
        assert b is not None and float(b[row]) > 1.0, "reading row desired"
        assert getattr(m, "_reading_desire", None) == 1.0
        m.set_reading(False)
        assert getattr(m, "_reading_desire", None) is None
    finally:
        m.relevance_on = False
        object.__setattr__(cs0, "_priming_boosts", None)
        object.__setattr__(m, "_reading_desire", None)


def test_reading_wiring_desires_staged_wholes():
    """While reading is desired, the assembler desires each batch's staged
    word-whole rows (the hard-coded concept->whole projection)."""
    m = _build("data/MM_sparse_concept.xml")
    ws0 = m.wholeSpaces[0]
    cs0 = m.conceptualSpaces[0]
    m.relevance_on = True
    try:
        m.set_reading(True)
        opt = m.getOptimizer(lr=0.01)
        object.__setattr__(ws0, "_priming_boosts", None)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
        b_read = ws0.priming_weights()
        assert b_read is not None, "staged rows must be primed"
        hot_read = float(b_read.max())
        m.set_reading(False)
        object.__setattr__(ws0, "_priming_boosts", None)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
        b_seen = ws0.priming_weights()
        assert b_seen is not None
        assert hot_read > float(b_seen.max()), (
            "desire must prime staged wholes beyond seen alone")
    finally:
        m.relevance_on = False
        m.set_reading(False)
        object.__setattr__(ws0, "_priming_boosts", None)
        object.__setattr__(cs0, "_priming_boosts", None)
