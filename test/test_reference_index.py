"""September 28 retrieval uses grammar and the existing priming surface."""
import copy
from types import SimpleNamespace

import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from MemoryIndex import unfold_idea


def test_native_reference_is_not_a_substitute_for_unfolding_the_stored_value():
    store = TernaryTruthStore(4, capacity=8)
    seen = []
    def unfold(value, limit, **kwargs):
        seen.append(value.clone())
        return (7,), 1, True
    store.configure_leaf_index(code_row=lambda ref: 99, unfold=unfold)
    point = torch.tensor([1., 2., 3., 4.])
    meaning = ConceptualMeaning.from_description(point)
    from dataclasses import replace
    meaning = replace(meaning, role_refs=(('sym', 1), None, None))
    row = store.append_meaning(meaning)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], point, rtol=0, atol=0)
    assert store.rows_for_code(7) == (row,)
    assert store.rows_for_code(99) == ()


def test_only_the_inverted_index_is_owned_or_checkpointed():
    store = TernaryTruthStore(4, capacity=8)
    store.configure_leaf_index(unfold=lambda value, limit, **kwargs: ((int(value[0]),), 1, True))
    a = store.append_meaning(ConceptualMeaning.from_description(torch.tensor([2., 0., 0., 0.])))
    b = store.append_meaning(ConceptualMeaning.from_description(torch.tensor([3., 0., 0., 0.])))
    assert store.rows_for_code(2) == (a,) and store.rows_for_code(3) == (b,)
    for name in ('leaf_codes', 'leaf_offsets', '_leaf_used'):
        assert not hasattr(store, name)
        assert name not in store.state_dict()
    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    assert restored.rows_for_code(2) == (a,)
    assert restored.rows_for_code(3) == (b,)


def test_unfold_candidates_are_the_current_priming_surface():
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=(),
        reverse_inverses=lambda ops: (), generate_policy_logits=lambda point: point.new_zeros(1, 1))
    basis = torch.eye(3)
    weights = torch.tensor([1., 2., 1.])
    active = lambda: weights
    cold = unfold_idea(language, basis, basis[0], 4, activation=active)
    hot = unfold_idea(language, basis, basis[1], 4, activation=active)
    assert cold['codes'] == () and not cold['complete']
    assert hot['codes'] == (1,) and hot['complete']
    weights[0] = 3.
    assert unfold_idea(language, basis, basis[0], 4, activation=active)['codes'] == (0,)
