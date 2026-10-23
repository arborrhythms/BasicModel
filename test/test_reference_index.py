"""Retrieval postings follow a bounded Part read; generation stays separate."""
import copy
from types import SimpleNamespace

import torch

from Layers import LTM0
from Meaning import ConceptualMeaning
from MemoryIndex import unfold_idea


def test_named_compound_indexes_its_part_read_instead_of_its_whole_bank_seat():
    store = LTM0(4, capacity=8)
    seen = []
    def read(value, limit, *, meaning, role, **kwargs):
        seen.append(meaning.role_refs[role])
        return (7,), 1, True
    store.configure_leaf_index(code_row=lambda ref: 99, read=read, contextual=True)
    point = torch.tensor([1., 2., 3., 4.])
    meaning = ConceptualMeaning.from_description(point)
    from dataclasses import replace
    meaning = replace(meaning, role_refs=(('sym', 1), None, None))
    row = store.append_meaning(meaning)
    assert len(seen) == 1
    assert seen[0] == ('sym', 1)
    assert store.rows_for_code(7) == (row,)
    assert store.rows_for_code(99) == ()


def test_inverted_index_is_derived_after_checkpoint_restore():
    store = LTM0(4, capacity=8)
    store.configure_leaf_index(read=lambda value, limit, **kwargs: ((int(value[0]),), 1, True))
    a = store.append_meaning(ConceptualMeaning.from_description(torch.tensor([2., 0., 0., 0.])))
    b = store.append_meaning(ConceptualMeaning.from_description(torch.tensor([3., 0., 0., 0.])))
    assert store.rows_for_code(2) == (a,) and store.rows_for_code(3) == (b,)
    for name in ('leaf_codes', 'leaf_offsets', '_leaf_used'):
        assert not hasattr(store, name)
        assert name not in store.state_dict()
    restored = LTM0(4, capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    restored.configure_leaf_index(read=store._index_read)
    restored.reindex_meanings()
    assert restored.rows_for_code(2) == (a,)
    assert restored.rows_for_code(3) == (b,)


def test_unfold_candidates_are_the_current_priming_surface():
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=(), generate_policy_logits=lambda point: point.new_zeros(1, 1))
    basis = torch.eye(3)
    weights = torch.tensor([1., 2., 1.])
    active = lambda: weights
    cold = unfold_idea(language, basis, basis[0], 4, activation=active)
    hot = unfold_idea(language, basis, basis[1], 4, activation=active)
    assert cold['codes'] == () and not cold['complete']
    assert hot['codes'] == (1,) and hot['complete']
    weights[0] = 3.
    assert unfold_idea(language, basis, basis[0], 4, activation=active)['codes'] == (0,)
