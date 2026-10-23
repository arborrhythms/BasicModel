"""Storage-only 2b contracts; no parsing or learning claim."""
import copy

import pytest
import torch

from Layers import LTM0
from Meaning import ConceptualMeaning


def test_stored_question_is_refused_by_kind_without_a_mode_column():
    store = LTM0(4, capacity=4)
    question = ConceptualMeaning.from_description(torch.ones(4))
    from dataclasses import replace
    row = store.append_meaning(replace(question, mode='interrogative'), kind='question')
    state = copy.deepcopy(store.state_dict())
    assert 'grammatical_mode' not in state and not hasattr(store, 'MODES')
    restored = LTM0(4, capacity=4)
    restored.load_state_dict(state)
    restored.load_semantic_extras(store.semantic_extras())
    for owner in (store, restored):
        assert owner.row(row)['kind'] == 'question'
        assert owner.row(row)['evidence'] == (0., 0.)
        with pytest.raises(ValueError, match='question or estimate'):
            owner.accept_fact(row)
        ordinary = owner.append_meaning(question, sentence_index=1)
        owner.accept_fact(ordinary)
        assert owner.row(ordinary)['kind'] == 'fact'


def test_negative_trust_derived_pair_reverses_only_on_the_relation_edge():
    from dataclasses import replace
    from test_clause_storage import clause_store, part_clause
    store, _ = clause_store()
    clause = part_clause()
    clause = replace(clause, meaning=replace(clause.meaning, polarity=False))
    edge = store.write_clause(clause, trust=.7)
    assert len(store) == 1
    assert store.row(edge)['evidence'] == pytest.approx((0., .7))
    assert store.row(0)['evidence'] == pytest.approx((.7, 0.))
    assert not store.raw_meaning_of(0).polarity
    torch.testing.assert_close(store.slots[0, 0], clause.meaning.roles[2])


def test_pending_forgetting_pressure_survives_restore_below_high_water():
    from Forgetting import forget
    from test_forgetting import linked_store, config
    store, child, parent = linked_store(child_is_relation=True)
    cfg = config(trust_weight=1, utility_weight=0, luminosity_weight=0, age_weight=0)
    with pytest.warns(RuntimeWarning, match='low water'):
        report = forget(store, cfg, document_boundary=True)
    assert report['after'] == 3 and store.index_of_row(parent[2]) is None
    assert store.index_of_row(child[2]) is not None and bool(store.forget_pending)
    restored = LTM0(4, capacity=4)
    restored.load_state_dict(store.state_dict())
    restored.load_semantic_extras(store.semantic_extras())
    assert bool(restored.forget_pending)
    report = forget(restored, cfg, document_boundary=True)
    assert report['target_reached'] and not bool(restored.forget_pending)
    assert restored.index_of_row(child[2]) is None
