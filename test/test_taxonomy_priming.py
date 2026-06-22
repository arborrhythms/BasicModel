"""Tests for the Taxonomy priming buffer + propagation methods.

Plan: doc/plans/2026-05-20-primed-reverse-generation.md §Part/whole
priming mask.

Priming uses boost-above-unity semantics:
  * default value = 1.0 (multiplicative identity, no priming)
  * freshly active ref → 2.0 (boost = 1.0)
  * one hop with hop_decay=0.5 → 1.5
  * two hops via shared parent → 1.25 (siblings)
  * temporal_decay between calls dissipates back toward 1.0
  * sentence boundary resets to 1.0 everywhere

The tiny test fixture builds a KnowledgeView from a 2-rule grammar:

    S4  = lift(NP3, VP1)
    NP3 = lower(DET, NP4)

producing a 9-ref taxonomy:

    0 = root
    1 = DET, 2 = NP, 3 = S, 4 = VP   (base category nodes under root)
    5 = S4 (under S), 6 = NP3 (under NP),
    7 = NP4 (under NP), 8 = VP1 (under VP)

so NP3 and NP4 are siblings via shared parent NP.
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
        g._parse_rule("S4", "lift(NP3, VP1)", space_role='SS'),
        g._parse_rule("NP3", "lower(DET, NP4)", space_role='SS'),
    ]
    g._configured = True
    return g


def _tiny_view():
    from embed import build_knowledge_section, KnowledgeView
    return KnowledgeView(build_knowledge_section(_tiny_grammar()))


def _allocated_taxonomy(view=None, batch_size=2):
    """A Taxonomy with priming allocated and a KnowledgeView attached."""
    from Language import Taxonomy
    if view is None:
        view = _tiny_view()
    tax = Taxonomy()
    tax.attach_view(view)
    # Allocate to the view's capacity (the underlying parent tensor's
    # length is V_ref_capacity in the artifact; live is n_refs_live).
    capacity = int(view._parent.shape[0])
    tax.allocate_priming(batch_size, capacity, view.n_refs_live)
    return tax, view


# -- Allocation + identity defaults -----------------------------------------


def test_priming_initial_shape_and_value():
    """After allocate_priming, the buffer is [B, V_ref_capacity] of 1.0."""
    tax, view = _allocated_taxonomy(batch_size=2)
    p = tax._priming
    assert p is not None
    assert p.shape[0] == 2
    assert p.shape[1] == int(view._parent.shape[0])
    assert torch.all(p == 1.0)


def test_priming_mask_returns_live_slice():
    """priming_mask() returns the [B, V_ref_live] live slice (no slack)."""
    tax, view = _allocated_taxonomy(batch_size=2)
    pm = tax.priming_mask()
    assert pm.shape == (2, view.n_refs_live)
    assert torch.all(pm == 1.0)


def test_priming_mask_per_batch():
    """priming_mask(batch=b) returns a 1-D [V_ref_live] view."""
    tax, view = _allocated_taxonomy(batch_size=2)
    pm = tax.priming_mask(batch=1)
    assert pm.shape == (view.n_refs_live,)


# -- prime() ----------------------------------------------------------------


def test_prime_sets_target_to_two():
    """prime([r], batch=0) sets _priming[0, r] = 2.0 (default boost=1.0)."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0)
    assert float(tax._priming[0, np3].item()) == 2.0


def test_prime_does_not_touch_other_entries():
    """Priming one ref leaves every other entry at 1.0."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0)
    other = torch.cat([tax._priming[0, :np3], tax._priming[0, np3 + 1:]])
    assert torch.all(other == 1.0)


def test_prime_is_max_not_replace():
    """A second prime with smaller boost does not lower the value."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0, boost=1.0)   # → 2.0
    tax.prime([np3], batch=0, boost=0.2)   # would be 1.2; max keeps 2.0
    assert float(tax._priming[0, np3].item()) == 2.0


def test_prime_is_max_lifts_existing():
    """A second prime with bigger boost lifts the value to the new max."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0, boost=0.3)   # → 1.3
    tax.prime([np3], batch=0, boost=1.0)   # → 2.0
    assert float(tax._priming[0, np3].item()) == pytest.approx(2.0)


def test_prime_isolates_batches():
    """prime on batch=0 does not touch batch=1."""
    tax, view = _allocated_taxonomy(batch_size=2)
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0)
    assert float(tax._priming[0, np3].item()) == 2.0
    assert float(tax._priming[1, np3].item()) == 1.0


def test_prime_drops_out_of_range():
    """Negative or oversized ref_ids are silently dropped."""
    tax, view = _allocated_taxonomy()
    cap = int(view._parent.shape[0])
    tax.prime([-1, cap, cap + 5], batch=0)
    assert torch.all(tax._priming == 1.0)


def test_prime_noop_when_unallocated():
    """prime() before allocate_priming is a no-op (no crash)."""
    from Language import Taxonomy
    tax = Taxonomy()
    tax.prime([0, 1, 2])
    assert tax._priming is None


# -- propagate() ------------------------------------------------------------


def test_propagate_depth1_lifts_parent():
    """Propagation hop 1 lifts immediate parent by hop_decay * boost."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    np = view.ref_id_for('NP')
    tax.prime([np3], batch=0)                          # NP3 = 2.0
    tax.propagate([np3], batch=0, depth=1, hop_decay=0.5)
    assert float(tax._priming[0, np].item()) == pytest.approx(1.5)
    # NP3 itself unchanged
    assert float(tax._priming[0, np3].item()) == pytest.approx(2.0)


def test_propagate_depth1_lifts_children():
    """Propagation hop 1 lifts immediate children of the seed."""
    tax, view = _allocated_taxonomy()
    np = view.ref_id_for('NP')
    np3 = view._ordered_taxonomy_names['NP3']
    np4 = view._ordered_taxonomy_names['NP4']
    tax.prime([np], batch=0)                           # NP = 2.0
    tax.propagate([np], batch=0, depth=1, hop_decay=0.5)
    assert float(tax._priming[0, np3].item()) == pytest.approx(1.5)
    assert float(tax._priming[0, np4].item()) == pytest.approx(1.5)


def test_propagate_depth1_does_not_lift_siblings():
    """At depth=1, siblings of the seed stay at 1.0."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    np4 = view._ordered_taxonomy_names['NP4']
    tax.prime([np3], batch=0)
    tax.propagate([np3], batch=0, depth=1, hop_decay=0.5)
    assert float(tax._priming[0, np4].item()) == 1.0


def test_propagate_depth2_lifts_siblings_through_parent():
    """At depth=2, siblings receive (1 + (1.5-1)*0.5) = 1.25 via shared parent."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    np4 = view._ordered_taxonomy_names['NP4']
    tax.prime([np3], batch=0)
    tax.propagate([np3], batch=0, depth=2, hop_decay=0.5)
    assert float(tax._priming[0, np4].item()) == pytest.approx(1.25)


def test_propagate_unrelated_refs_stay_identity():
    """Refs not reachable within depth hops remain at 1.0."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    det = view.ref_id_for('DET')
    vp1 = view._ordered_taxonomy_names['VP1']
    tax.prime([np3], batch=0)
    tax.propagate([np3], batch=0, depth=2, hop_decay=0.5)
    # DET is reachable only via root after 3 hops; VP1 is even further.
    assert float(tax._priming[0, det].item()) == 1.0
    assert float(tax._priming[0, vp1].item()) == 1.0


def test_propagate_zero_depth_is_noop():
    """propagate with depth=0 does not change anything."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    np = view.ref_id_for('NP')
    tax.prime([np3], batch=0)
    tax.propagate([np3], batch=0, depth=0, hop_decay=0.5)
    assert float(tax._priming[0, np].item()) == 1.0


def test_propagate_noop_without_view():
    """propagate before attach_view is a no-op."""
    from Language import Taxonomy
    tax = Taxonomy()
    tax.allocate_priming(1, 10, 10)
    tax.prime([3], batch=0)
    tax.propagate([3], batch=0, depth=2)
    # Only the seed should remain at 2.0; nothing else propagated.
    assert float(tax._priming[0, 3].item()) == 2.0
    others = torch.cat([tax._priming[0, :3], tax._priming[0, 4:]])
    assert torch.all(others == 1.0)


# -- decay() ----------------------------------------------------------------


def test_decay_pulls_primed_toward_identity():
    """decay(0.9) maps 2.0 → 1.9, 1.5 → 1.45, leaves 1.0 at 1.0."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    np = view.ref_id_for('NP')
    det = view.ref_id_for('DET')
    tax.prime([np3], batch=0)                          # 2.0
    tax.propagate([np3], batch=0, depth=1, hop_decay=0.5)  # NP=1.5
    tax.decay(temporal_decay=0.9)
    assert float(tax._priming[0, np3].item()) == pytest.approx(1.9)
    assert float(tax._priming[0, np].item()) == pytest.approx(1.45)
    assert float(tax._priming[0, det].item()) == 1.0


def test_decay_idempotent_at_identity():
    """decay on an all-1.0 buffer is a no-op."""
    tax, _ = _allocated_taxonomy()
    tax.decay(temporal_decay=0.9)
    assert torch.all(tax._priming == 1.0)


def test_decay_noop_when_unallocated():
    from Language import Taxonomy
    tax = Taxonomy()
    tax.decay()  # no crash


# -- reset() ----------------------------------------------------------------


def test_reset_returns_to_identity():
    """reset() puts every entry back to 1.0."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0)
    tax.propagate([np3], batch=0, depth=2)
    tax.reset()
    assert torch.all(tax._priming == 1.0)


def test_reset_noop_when_unallocated():
    from Language import Taxonomy
    tax = Taxonomy()
    tax.reset()  # no crash


# -- capacity-slack growth --------------------------------------------------


def test_allocate_priming_preserves_existing_primed_values():
    """Re-allocating with the same or larger size preserves prior values."""
    tax, view = _allocated_taxonomy()
    np3 = view._ordered_taxonomy_names['NP3']
    tax.prime([np3], batch=0)                          # 2.0
    cap = int(view._parent.shape[0])
    # Grow capacity; live grows too (simulate extend_artifact appending).
    tax.allocate_priming(2, cap + 4, view.n_refs_live + 2)
    assert tax._priming.shape == (2, cap + 4)
    # Existing primed value survived
    assert float(tax._priming[0, np3].item()) == 2.0
    # New columns initialize to 1.0
    assert torch.all(tax._priming[:, cap:] == 1.0)


def test_attach_view_updates_live_count():
    """attach_view records the view's n_refs_live."""
    tax, view = _allocated_taxonomy()
    assert tax.priming_live == view.n_refs_live
    tax.attach_view(None)  # detach
    # _priming_live is preserved at last attach's value (no view to query)
