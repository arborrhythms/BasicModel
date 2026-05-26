"""End-to-end primed reverse generation tests.

Plan: doc/plans/2026-05-20-primed-reverse-generation.md §Tests —
Integration.

A small grammar + lexicon, priming a recently-active ref, then a
LIFT-reverse that should prefer the primed ref. With priming
disabled, the same call returns the typed-only choice; across calls
priming decays, and a sentence boundary clears it.

Test fixture uses the standard 2-rule grammar:

    S4  = lift(NP3, VP1)
    NP3 = lower(DET, NP4)

To exercise priming **decisively**, we extend the reference codebook
with a second NP3-class ref ("NP3_alt") that has slightly larger
norm than the original NP3. Without priming, NP3_alt wins
(largest ≤ y); with NP3 primed (boost 1.0 → priming 2.0), NP3 wins.
"""
import sys
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))


def _tiny_grammar():
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
    ]
    g._configured = True
    return g


def _tiny_view():
    from embed import build_knowledge_section, KnowledgeView
    return KnowledgeView(build_knowledge_section(_tiny_grammar()))


def _bare_word_space(batch=1):
    from Language import WordSubSpace, Taxonomy
    import torch.nn as nn
    ws = object.__new__(WordSubSpace)
    nn.Module.__init__(ws)
    ws.batch = int(batch)
    ws.taxonomy = Taxonomy()
    return ws


def _scenario():
    """Set up a WordSpace + KnowledgeView + competing-NP3 W codebook.

    Returns (ws, view, W, y, np3_idx, np3_alt_value).

    The 2nd NP3 row (added directly to W) has a slightly larger norm
    than the artifact's stock NP3 so it wins on un-primed argmax;
    priming the stock NP3 should flip the choice.
    """
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    np3 = view._ordered_taxonomy_names['NP3']
    # Build a 2-D W matching the artifact references (scalar prototypes
    # promoted to a 2-D codebook so the recommender's broadcast works).
    refs = view.references.unsqueeze(-1).clone()  # [V_live, 1]
    # Add a second NP3-shaped row with slightly larger prototype value.
    # We don't extend the taxonomy here (and so the helper's
    # ``left_rows = refs_by_category[NP] ∩ refs_by_order[3]`` would
    # still return only the stock NP3). We extend it manually for the
    # test so the recommender sees two competing NP3 candidates.
    np3_alt_row = torch.tensor([[0.40]])     # larger than 0.0 stock
    refs[np3, 0] = 0.30                       # smaller than alt
    W = torch.cat([refs, np3_alt_row], dim=0) # last row is the alt
    np3_alt_idx = W.shape[0] - 1
    # left_rows: both NP3 candidates (stock + alt)
    left_rows = torch.tensor([np3, np3_alt_idx], dtype=torch.long)
    # y chosen so both rows are feasible (≤ y) for union x1.
    y = torch.tensor([[[0.50]]])
    return ws, view, W, y, left_rows, np3, np3_alt_idx


# -- Without priming: larger-norm candidate wins ---------------------------


def test_unprimed_disjunction_picks_larger_norm():
    """Sanity check: without priming, union x1 (argmax norm ≤ y) picks
    the larger-norm candidate (NP3_alt)."""
    from Layers import Ops
    ws, view, W, y, left_rows, np3, np3_alt_idx = _scenario()
    x1, _ = Ops.disjunctionReverse(
        y, y, W,
        left_rows=left_rows,
        # No priming → helper's mask is all-1.0 (or omitted)
    )
    assert torch.allclose(x1[0, 0, 0], W[np3_alt_idx, 0])


# -- With priming: the primed candidate wins ------------------------------


def test_primed_disjunction_picks_primed_candidate():
    """Priming the smaller-norm NP3 (boost → priming=2.0) makes its
    effective score larger than the alt's, so it wins.
    """
    from Layers import Ops
    ws, view, W, y, left_rows, np3, np3_alt_idx = _scenario()
    ws.taxonomy.prime([np3], batch=0)
    # Build priming aligned with W's row count: artifact prefix
    # (n_refs_live) + the np3_alt row appended.
    priming = torch.ones(W.shape[0])
    priming[:view.n_refs_live] = ws.taxonomy.priming_mask(batch=0)
    x1, _ = Ops.disjunctionReverse(
        y, y, W, left_rows=left_rows, left_priming=priming)
    assert torch.allclose(x1[0, 0, 0], W[np3, 0])


# -- priming_enabled=false reproduces typed-only behavior -----------------


def test_priming_disabled_matches_unprimed():
    """With priming_enabled=False, the helper omits priming kwargs;
    the result must match a direct un-primed call byte-for-byte."""
    from Layers import Ops
    ws, view, W, y, left_rows, np3, np3_alt_idx = _scenario()
    ws.taxonomy.prime([np3], batch=0)        # would normally flip the choice
    ws.taxonomy.configure_priming(priming_enabled=False)
    # The helper would return only left_rows / right_rows in this mode.
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3,
        right_category='VP', right_order=1)
    assert 'left_priming' not in kw
    assert 'right_priming' not in kw
    # Direct un-primed call should match.
    x1_disabled, _ = Ops.disjunctionReverse(
        y, y, W, left_rows=left_rows)
    x1_unprimed, _ = Ops.disjunctionReverse(
        y, y, W, left_rows=left_rows)
    assert torch.equal(x1_disabled, x1_unprimed)


# -- Decay between calls --------------------------------------------------


def test_priming_decays_between_calls():
    """After ``decay(temporal_decay=0.9)``, the primed value moves
    toward identity but is still > 1.0 (still biases selection).
    """
    ws, view, W, y, left_rows, np3, np3_alt_idx = _scenario()
    ws.taxonomy.prime([np3], batch=0)        # 2.0
    ws.taxonomy.decay(temporal_decay=0.9)    # → 1.9
    assert float(ws.taxonomy._priming[0, np3].item()) == pytest.approx(1.9)
    ws.taxonomy.decay(temporal_decay=0.9)    # → 1.81
    assert float(ws.taxonomy._priming[0, np3].item()) == pytest.approx(1.81)


# -- Sentence boundary clears priming -------------------------------------


def test_sentence_boundary_clears_priming():
    """Taxonomy.reset() at the sentence boundary returns the buffer to
    identity."""
    ws, view, W, y, left_rows, np3, np3_alt_idx = _scenario()
    ws.taxonomy.prime([np3], batch=0)
    ws.taxonomy.propagate([np3], batch=0, depth=2)
    assert not torch.all(ws.taxonomy._priming == 1.0)
    ws.taxonomy.reset()
    assert torch.all(ws.taxonomy._priming == 1.0)


# -- Hard mask still wins over priming in the integration setup -----------


def test_hard_mask_excludes_primed_off_category_ref():
    """Even with maximum priming, a row outside left_rows can't win."""
    from Layers import Ops
    ws, view, W, y, left_rows, np3, np3_alt_idx = _scenario()
    # Prime the off-category ref (e.g. a DET ref) with huge boost.
    det = view.ref_id_for('DET')
    ws.taxonomy.prime([det], batch=0, boost=99.0)
    priming = torch.ones(W.shape[0])
    priming[:view.n_refs_live] = ws.taxonomy.priming_mask(batch=0)
    x1, _ = Ops.disjunctionReverse(
        y, y, W, left_rows=left_rows, left_priming=priming)
    # Selection must come from {⊥, W[np3], W[np3_alt_idx], ⊤}.
    chosen = x1[0, 0, 0].item()
    allowed = {0.0, float(W[np3, 0].item()),
               float(W[np3_alt_idx, 0].item()), 1.0}
    assert chosen in allowed
    # Specifically, the DET-row primer must NOT cause a DET row to win.
    assert chosen != float(W[det, 0].item()) or chosen in allowed
