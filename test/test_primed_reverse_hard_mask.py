"""Tests for Phase C wiring: WordSpace.attach_knowledge allocates the
Taxonomy priming buffer, and ``priming_kwargs_for_slots`` builds the
four recommender kwargs (left/right rows + left/right priming) from a
rule's typed slot info.

Plan: doc/plans/2026-05-20-primed-reverse-generation.md §Hard
admissibility mask + §Reverse operation flow.

Uses ``object.__new__`` to bypass WordSpace's heavy __init__ (which
needs PartSpace / ConceptualSpace / WholeSpace). We only
need attach_knowledge + the helper, which don't depend on the full
Space wiring.
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
    # Minimal scaffolding needed by attach_knowledge.
    ws.batch = int(batch)
    ws.taxonomy = Taxonomy()
    return ws


# -- attach_knowledge allocates priming ------------------------------------


def test_attach_allocates_priming_buffer():
    """After attach_knowledge, the taxonomy's priming buffer exists at
    [batch, V_ref_capacity] and is initialized to 1.0."""
    ws = _bare_word_space(batch=2)
    view = _tiny_view()
    ws.attach_knowledge(view)
    p = ws.taxonomy._priming
    assert p is not None
    assert p.shape[0] == 2
    assert p.shape[1] == int(view._parent.shape[0])
    assert torch.all(p == 1.0)


def test_attach_binds_view_for_propagation():
    """attach_knowledge wires the view onto the taxonomy so propagate
    has parent/children adjacency to walk."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    assert ws.taxonomy._priming_view is view


def test_reattach_preserves_primed_values():
    """Re-attaching the same view (or one with same shape) preserves
    in-progress priming."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    np3 = view._ordered_taxonomy_names['NP3']
    ws.taxonomy.prime([np3], batch=0)
    assert float(ws.taxonomy._priming[0, np3].item()) == 2.0
    ws.attach_knowledge(_tiny_view())
    # Same fixture shape; in-progress priming carries forward.
    assert float(ws.taxonomy._priming[0, np3].item()) == 2.0


def test_attach_without_view_is_noop_on_priming():
    """When called with None (or via the path that doesn't attach a
    view), the priming buffer stays unallocated."""
    ws = _bare_word_space()
    # _knowledge stays None — no view to bind.
    assert ws.taxonomy._priming is None


# -- priming_kwargs_for_slots -----------------------------------------------


def test_kwargs_returns_empty_without_knowledge():
    """No view attached → helper returns {} (graceful fallback to
    un-typed, un-primed selection at the caller)."""
    ws = _bare_word_space()
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3,
        right_category='VP', right_order=1)
    assert kw == {}


def test_kwargs_intersection_rows_for_binary_rule():
    """For S = LIFT(NP3, VP1), the helper produces:
       left_rows  = refs_by_category[NP] ∩ refs_by_order[3]  = [NP3]
       right_rows = refs_by_category[VP] ∩ refs_by_order[1]  = [VP1]
    """
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3,
        right_category='VP', right_order=1)
    np3 = view._ordered_taxonomy_names['NP3']
    vp1 = view._ordered_taxonomy_names['VP1']
    assert 'left_rows' in kw
    assert 'right_rows' in kw
    assert kw['left_rows'].tolist() == [np3]
    assert kw['right_rows'].tolist() == [vp1]


def test_kwargs_priming_present_when_buffer_allocated():
    """left_priming / right_priming are the live slice of the priming
    buffer; both slots get the same per-batch mask."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    # The <symbolicPriming> master switch (plan 2026-06-06-symbolic-heat-
    # retrieval) now defaults OFF, so attach_knowledge sets priming_enabled
    # False; opt in to exercise the priming-kwarg emission path.
    ws.taxonomy.configure_priming(priming_enabled=True)
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3,
        right_category='VP', right_order=1)
    assert 'left_priming' in kw
    assert 'right_priming' in kw
    assert kw['left_priming'].shape == (view.n_refs_live,)
    assert kw['right_priming'].shape == (view.n_refs_live,)
    # Default = identity 1.0 everywhere
    assert torch.all(kw['left_priming'] == 1.0)
    assert torch.all(kw['right_priming'] == 1.0)


def test_kwargs_priming_reflects_primed_state():
    """Priming a ref shows up in the helper's output."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    ws.taxonomy.configure_priming(priming_enabled=True)  # master switch (see above)
    np3 = view._ordered_taxonomy_names['NP3']
    ws.taxonomy.prime([np3], batch=0)
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3,
        right_category='VP', right_order=1,
        batch=0)
    assert float(kw['left_priming'][np3].item()) == 2.0


def test_kwargs_empty_intersection_returns_empty_long_tensor():
    """When the category × order intersection is empty (e.g. there's
    no NP at order 5), the recommender still gets an empty LongTensor
    rather than None — sentinels remain feasible."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=5,
        right_category='VP', right_order=1)
    assert kw['left_rows'].numel() == 0
    assert kw['left_rows'].dtype == torch.long


def test_kwargs_unary_omits_right():
    """A unary slot (right_category=None) omits right_* from the
    kwargs."""
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    ws.taxonomy.configure_priming(priming_enabled=True)  # master switch (see above)
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3)
    assert 'left_rows' in kw
    assert 'right_rows' not in kw
    assert 'left_priming' in kw
    assert 'right_priming' not in kw


def test_kwargs_per_batch_priming():
    """Priming on batch=0 doesn't leak into batch=1's kwargs."""
    ws = _bare_word_space(batch=2)
    view = _tiny_view()
    ws.attach_knowledge(view)
    ws.taxonomy.configure_priming(priming_enabled=True)  # master switch (see above)
    np3 = view._ordered_taxonomy_names['NP3']
    ws.taxonomy.prime([np3], batch=0)
    kw_b0 = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3, batch=0)
    kw_b1 = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3, batch=1)
    assert float(kw_b0['left_priming'][np3].item()) == 2.0
    assert float(kw_b1['left_priming'][np3].item()) == 1.0


# -- end-to-end: pass helper's kwargs straight to the recommender ---------


def test_helper_kwargs_drive_recommender_byte_for_byte():
    """A typical wiring: helper builds kwargs → Ops.disjunctionReverse
    consumes them. The result must respect both the hard mask (only
    NP3 / VP1 are in the candidate set) and priming (no-op default)."""
    from Layers import Ops
    ws = _bare_word_space()
    view = _tiny_view()
    ws.attach_knowledge(view)
    kw = ws.priming_kwargs_for_slots(
        left_category='NP', left_order=3,
        right_category='VP', right_order=1)
    # Use scalar-prototype-shaped W: 1-D references from the view.
    refs = view.references.clone().unsqueeze(-1)   # [V_live, 1]
    y = torch.tensor([[[0.5]]])
    x1, x2 = Ops.disjunctionReverse(y, y, refs, **kw)
    # Selection must be one of: ⊥(0.0), W[NP3], W[VP1], ⊤(1.0).
    np3 = view._ordered_taxonomy_names['NP3']
    vp1 = view._ordered_taxonomy_names['VP1']
    allowed_left = {0.0, float(refs[np3, 0].item()), 1.0}
    allowed_right = {0.0, float(refs[vp1, 0].item()), 1.0}
    assert float(x1.reshape(-1)[0].item()) in allowed_left
    assert float(x2.reshape(-1)[0].item()) in allowed_right
