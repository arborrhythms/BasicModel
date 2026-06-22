"""Tests for KnowledgeView — read-only facade over a loaded knowledge
section.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 — Loaders. The view is what Spaces consult at runtime to
answer "which refs belong to category X", "what's this ref's order",
etc., without each Space re-implementing the lookup logic.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _tiny_grammar():
    """Manually-built Grammar with two order-typed rules."""
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", space_role='SS'),
        g._parse_rule("NP3", "lower(DET, NP4)", space_role='SS'),
    ]
    g._configured = True
    return g


def _tiny_knowledge_section():
    """Build a knowledge section from the tiny grammar (no wv)."""
    from embed import build_knowledge_section
    return build_knowledge_section(_tiny_grammar())


def test_knowledge_view_n_refs_live():
    """``view.n_refs_live`` returns the live count (root + categories)."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    # root + base categories {S, NP, VP, DET} + ordered refs
    # {S4, NP3, VP1, NP4}
    assert view.n_refs_live == 9


def test_knowledge_view_references_slice_shape():
    """``view.references`` returns just the live rows (not the capacity-
    slack zeros)."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    assert view.references.shape[0] == view.n_refs_live


def test_knowledge_view_orders_slice_shape():
    """``view.orders`` likewise slices to live rows."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    assert view.orders.shape[0] == view.n_refs_live


def test_knowledge_view_refs_by_category():
    """``view.refs_by_category(name)`` returns the LongTensor of ref_ids
    for that category. Missing category returns empty tensor."""
    from embed import KnowledgeView
    import torch
    view = KnowledgeView(_tiny_knowledge_section())
    np_refs = view.refs_by_category('NP')
    assert isinstance(np_refs, torch.Tensor)
    assert np_refs.dtype == torch.long
    # Base NP plus ordered NP3 / NP4 refs.
    assert np_refs.shape[0] == 3
    # Missing category: empty
    none_refs = view.refs_by_category('NONEXISTENT')
    assert none_refs.shape[0] == 0


def test_knowledge_view_refs_by_order():
    """``view.refs_by_order(k)`` returns the LongTensor at order k.
    Bootstrap: base class nodes at order 0, explicit ordered refs at
    their grammatical order."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    refs_o0 = view.refs_by_order(0)
    assert refs_o0.shape[0] == 5
    assert view.refs_by_order(3).shape[0] == 1
    assert view.refs_by_order(4).shape[0] == 2
    # No order 7 yet
    assert view.refs_by_order(7).shape[0] == 0


def test_knowledge_view_rule_order_signatures():
    """``view.rule_order_signatures`` exposes the list of dicts."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    sigs = view.rule_order_signatures
    assert isinstance(sigs, list)
    assert len(sigs) == 2
    assert sigs[0]['op_name'] == 'lift'
    assert sigs[1]['op_name'] == 'lower'


def test_knowledge_view_taxonomy_names():
    """``view.taxonomy_names`` returns the name → ref_id dict."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    names = view.taxonomy_names
    for cat in ('S', 'NP', 'VP', 'DET'):
        assert cat in names
        assert isinstance(names[cat], int)


def test_knowledge_view_ref_id_for_name():
    """``view.ref_id_for(name)`` returns the ref_id of a named category;
    None for unknown names."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    np_id = view.ref_id_for('NP')
    assert np_id is not None
    assert np_id == view.taxonomy_names['NP']
    assert view.ref_id_for('UNKNOWN') is None


def test_knowledge_view_category_of_ref():
    """``view.category_of_ref(ref_id)`` returns the category name for a
    leaf ref by walking up to its first-named ancestor. For class
    nodes themselves, returns their own name."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    np_id = view.ref_id_for('NP')
    assert view.category_of_ref(np_id) == 'NP'
    # Root has no category
    assert view.category_of_ref(0) is None


def test_knowledge_view_order_of_ref():
    """``view.order_of_ref(ref_id)`` returns the conceptual order of
    a ref. Bootstrap: all class nodes at order 0."""
    from embed import KnowledgeView
    view = KnowledgeView(_tiny_knowledge_section())
    s_id = view.ref_id_for('S')
    assert view.order_of_ref(s_id) == 0


def test_knowledge_view_round_trip_from_load(tmp_path):
    """KnowledgeView works on a section loaded back from disk."""
    from embed import (save_artifact, load_artifact,
                       build_knowledge_section, KnowledgeView)
    ks = build_knowledge_section(_tiny_grammar())
    path = str(tmp_path / "rt.kv")
    save_artifact(path, knowledge=ks)
    loaded = load_artifact(path)['knowledge']
    view = KnowledgeView(loaded)
    assert view.n_refs_live == 9
    assert view.ref_id_for('NP') is not None


def test_load_knowledge_view_helper(tmp_path):
    """``load_knowledge_view(path)`` is a one-shot bridge: load artifact,
    extract knowledge section, wrap in KnowledgeView."""
    from embed import (save_artifact, build_knowledge_section,
                       load_knowledge_view)
    ks = build_knowledge_section(_tiny_grammar())
    path = str(tmp_path / "rt.kv")
    save_artifact(path, knowledge=ks)
    view = load_knowledge_view(path)
    assert view.n_refs_live == 9


def test_load_knowledge_view_missing_section_raises(tmp_path):
    """Artifact without a knowledge section: load_knowledge_view raises."""
    import torch
    from embed import save_artifact, load_knowledge_view
    path = str(tmp_path / "no_knowledge.kv")
    save_artifact(path, lexicon={
        'section_kind': 'lexicon',
        'vectors': torch.zeros(1, 4),
        'index_to_key': ['x'],
        'counts': None,
        'total_count': 0,
    })
    import pytest
    with pytest.raises(ValueError):
        load_knowledge_view(path)


# -- Typed admissibility --------------------------------------------
# `is_rule_admissible(sig, left_cat, left_order, right_cat=None,
# right_order=None) -> bool` decides whether a rule signature can fire
# against the given operand categories and orders. STM REDUCE masks
# inadmissible rules before the reducer softmax.


def _make_sig(lhs_category, lhs_order, rhs_categories, rhs_orders,
              op_name, order_delta):
    """Construct a serialized RuleOrderSignature dict shape (matches
    what ``grammar_signatures_to_serializable`` emits)."""
    return {
        'lhs_category':  lhs_category,
        'lhs_order':     lhs_order,
        'rhs_categories': list(rhs_categories),
        'rhs_orders':    list(rhs_orders),
        'op_name':       op_name,
        'order_delta':   order_delta,
    }


def test_admissible_binary_exact_match():
    """Rule ``S4 = lift(NP3, VP1)`` is admissible for operands
    (NP at order 3, VP at order 1)."""
    from embed import is_rule_admissible
    sig = _make_sig('S', 4, ('NP', 'VP'), (3, 1), 'lift', +1)
    assert is_rule_admissible(
        sig, left_cat='NP', left_order=3,
        right_cat='VP', right_order=1) is True


def test_inadmissible_binary_category_mismatch():
    """Left category mismatch → not admissible."""
    from embed import is_rule_admissible
    sig = _make_sig('S', 4, ('NP', 'VP'), (3, 1), 'lift', +1)
    assert is_rule_admissible(
        sig, left_cat='DET', left_order=3,
        right_cat='VP', right_order=1) is False


def test_inadmissible_binary_order_mismatch():
    """Left order mismatch → not admissible."""
    from embed import is_rule_admissible
    sig = _make_sig('S', 4, ('NP', 'VP'), (3, 1), 'lift', +1)
    assert is_rule_admissible(
        sig, left_cat='NP', left_order=2,  # rule wants 3
        right_cat='VP', right_order=1) is False


def test_inadmissible_right_category_mismatch():
    """Right category mismatch → not admissible."""
    from embed import is_rule_admissible
    sig = _make_sig('S', 4, ('NP', 'VP'), (3, 1), 'lift', +1)
    assert is_rule_admissible(
        sig, left_cat='NP', left_order=3,
        right_cat='AP', right_order=1) is False


def test_inadmissible_arity_mismatch():
    """Binary rule fed only a left operand → not admissible (arity wrong)."""
    from embed import is_rule_admissible
    sig = _make_sig('S', 4, ('NP', 'VP'), (3, 1), 'lift', +1)
    assert is_rule_admissible(
        sig, left_cat='NP', left_order=3) is False


def test_admissible_unary():
    """Unary rule ``S3 = not(S3)`` matches a single S-at-3 operand."""
    from embed import is_rule_admissible
    sig = _make_sig('S', 3, ('S',), (3,), 'not', 0)
    assert is_rule_admissible(
        sig, left_cat='S', left_order=3) is True
    # But binary call shape is wrong for a unary rule.
    assert is_rule_admissible(
        sig, left_cat='S', left_order=3,
        right_cat='S', right_order=3) is False


def test_admissible_lower_with_consistent_orders():
    """``NP3 = lower(DET, NP4)`` admissible for (DET@0, NP@4)."""
    from embed import is_rule_admissible
    sig = _make_sig('NP', 3, ('DET', 'NP'), (0, 4), 'lower', -1)
    assert is_rule_admissible(
        sig, left_cat='DET', left_order=0,
        right_cat='NP', right_order=4) is True


def test_inadmissible_lower_against_lower_order_noun():
    """``NP3 = lower(DET, NP4)`` cannot fire on (DET@0, NP@3) — the
    operand NP is order 3, not the rule's required 4."""
    from embed import is_rule_admissible
    sig = _make_sig('NP', 3, ('DET', 'NP'), (0, 4), 'lower', -1)
    assert is_rule_admissible(
        sig, left_cat='DET', left_order=0,
        right_cat='NP', right_order=3) is False


# -- Admissibility mask over a rule list ------------------------------
# STM REDUCE's mask-before-softmax: given the current stack-top state
# (left/right cat+order) and a list of rule signatures, produce a
# BoolTensor of length len(rules) where True = admissible.


def _three_rule_list():
    """Three serialized rule signatures: lift, lower, and an
    ordinary not-rule."""
    return [
        _make_sig('S', 4, ('NP', 'VP'), (3, 1), 'lift', +1),
        _make_sig('NP', 3, ('DET', 'NP'), (0, 4), 'lower', -1),
        _make_sig('S', 3, ('S',), (3,), 'not', 0),
    ]


def test_admissibility_mask_length():
    """Mask length == number of rules."""
    from embed import admissibility_mask
    rules = _three_rule_list()
    mask = admissibility_mask(
        rules, left_cat='NP', left_order=3,
        right_cat='VP', right_order=1)
    assert mask.shape[0] == len(rules)


def test_admissibility_mask_binary_picks_only_lift():
    """For binary input (NP3, VP1), only the lift rule is admissible."""
    from embed import admissibility_mask
    import torch
    rules = _three_rule_list()
    mask = admissibility_mask(
        rules, left_cat='NP', left_order=3,
        right_cat='VP', right_order=1)
    assert mask.dtype == torch.bool
    assert bool(mask[0].item()) is True
    assert bool(mask[1].item()) is False
    assert bool(mask[2].item()) is False


def test_admissibility_mask_unary_picks_only_unary_rule():
    """For unary input (S3,), only the not-rule is admissible."""
    from embed import admissibility_mask
    rules = _three_rule_list()
    mask = admissibility_mask(rules, left_cat='S', left_order=3)
    assert bool(mask[0].item()) is False
    assert bool(mask[1].item()) is False
    assert bool(mask[2].item()) is True


def test_admissibility_mask_all_false_when_nothing_fits():
    """No rule matches → all-False mask (legal — REDUCE handles this
    by stalling or backtracking, not by firing a wrong rule)."""
    from embed import admissibility_mask
    rules = _three_rule_list()
    mask = admissibility_mask(
        rules, left_cat='XYZ', left_order=99,
        right_cat='ABC', right_order=42)
    assert not mask.any()


# -- Logit masking for softmax ---------------------------------------
# REDUCE applies the admissibility mask to the rule logits before
# softmax: inadmissible rules get -inf, so their post-softmax weight
# is exactly 0. ``mask_logits(logits, mask)`` is the trivial helper
# that does this.


def test_mask_logits_zeros_inadmissible_after_softmax():
    """After masking + softmax, inadmissible rules have probability 0."""
    from embed import admissibility_mask, mask_logits
    import torch
    rules = _three_rule_list()
    mask = admissibility_mask(
        rules, left_cat='NP', left_order=3,
        right_cat='VP', right_order=1)
    # Synthetic logits — all equal so softmax would be uniform without mask
    logits = torch.zeros(len(rules))
    masked = mask_logits(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert float(probs[0].item()) > 0.99   # only admissible
    assert float(probs[1].item()) < 1e-6
    assert float(probs[2].item()) < 1e-6


def test_mask_logits_preserves_relative_admissible_weights():
    """When several rules are admissible, masking preserves their
    relative logits among the admissible set."""
    from embed import mask_logits
    import torch
    # Hypothetical 4-rule case: rules 0 and 2 admissible, 1 and 3 not.
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    mask = torch.tensor([True, False, True, False])
    masked = mask_logits(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    # Rules 0 and 2 should have probabilities proportional to
    # exp(2) and exp(0); rules 1 and 3 should be exactly 0.
    assert float(probs[1].item()) < 1e-6
    assert float(probs[3].item()) < 1e-6
    # Within admissible: exp(2) / (exp(2) + exp(0)) = e^2 / (e^2 + 1)
    import math
    expected_p0 = math.exp(2) / (math.exp(2) + math.exp(0))
    assert float(probs[0].item()) == pytest_approx(expected_p0)
    assert float(probs[2].item()) == pytest_approx(1 - expected_p0)


def pytest_approx(v, tol=1e-5):
    """Tiny helper since pytest.approx requires the pytest import in scope."""
    import pytest
    return pytest.approx(v, abs=tol)


def test_mask_logits_all_admissible_unchanged_after_softmax():
    """All-True mask → masked logits equal original logits."""
    from embed import mask_logits
    import torch
    logits = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.ones(3, dtype=torch.bool)
    masked = mask_logits(logits, mask)
    assert torch.allclose(masked, logits)


def test_mask_logits_all_inadmissible_returns_neg_inf():
    """All-False mask → every logit becomes -inf."""
    from embed import mask_logits
    import torch
    logits = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.zeros(3, dtype=torch.bool)
    masked = mask_logits(logits, mask)
    assert torch.isinf(masked).all()
    assert (masked < 0).all()
