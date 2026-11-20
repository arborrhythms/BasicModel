"""Read-word object certainty and expectation-selected denotations."""
import math

import pytest
import torch
from nomination_fixtures import recurrent_word

pytestmark = pytest.mark.assessment_rung(1)


def test_88_read_word_fixes_its_objects_radius_without_freezing_its_content():
    from test_trust_round1 import learned
    from definition_fixtures import agreeing_views
    from ConceptTrust import observe, positive_length
    owner, store, obj, atoms = learned(1)
    word = recurrent_word(owner.interpret, [4], [], form='wug')
    agreeing_views(owner, word)
    row = owner.interpret.define(word, obj)
    assert positive_length(owner, obj) == 1.
    assert float(store.parts.truth_plus[row]) == 1.
    assert not owner.is_frozen(obj)
    observe(owner, obj, atoms[1])
    assert positive_length(owner, obj) == 1.
    bank = owner.similarity_codebook
    state = {key: value.clone() for key, value in bank.state_dict().items()}
    bank.load_state_dict(state)
    bank.trust.refresh(obj)
    assert positive_length(owner, obj) == 1.


@pytest.mark.parametrize('trust', (0., .75))
def test_88_stated_definition_keeps_the_callers_trust(trust):
    from test_trust_round1 import learned
    owner, store, obj, atoms = learned(1)
    word = recurrent_word(owner.interpret, [4], [], form='wug')
    from test_thought_model_fixture import trusted_premises
    trusted_premises(owner, ('sym', word), ('sym', obj))
    before = owner.similarity_codebook.trust.lengths.clone()
    row = owner.interpret.define(word, obj, trust=trust)
    assert float(store.parts.truth_plus[row]) == pytest.approx(trust)
    assert owner.interpret.forward(word) == (obj if trust >= .5 else None)
    assert float(store.parts.truth_plus[row]) == pytest.approx(trust)
    torch.testing.assert_close(owner.similarity_codebook.trust.lengths, before, rtol=0, atol=0)


def test_89_expectation_selects_among_orders_and_is_local_to_the_stream():
    from test_cs_sparse_weights import _cs
    from ConceptSpread import open_prediction
    from types import SimpleNamespace
    owner = _cs(nS=128, order=2, dim=16)
    word = recurrent_word(owner.interpret, [1], [2], form='label')
    low, high = owner.new_concept(order=0), owner.new_concept(order=1)
    for obj in (low, high):
        owner.interpret.define(word, obj)
    model = SimpleNamespace(_concept_owner=lambda: owner)
    owner._observed_knowing[0, owner._concept_row_of(low), 0] = 1.
    open_prediction(model, torch.tensor([True]))
    assert owner.interpret.denotations(word) == (low, high)
    assert owner.interpret.referent(word, batch=0) == high
    # A controlled spread (the transport, apart from the structure) gives
    # the observed concept positive expectation. The reader consumes only
    # what open_prediction writes, including its order limit.
    import ConceptSpread
    def controlled(owner_, knowing, **kwargs):
        return knowing.new_zeros(knowing.shape[0], owner_.nVectors, 2).index_copy(
            1, torch.arange(knowing.shape[1]), knowing.detach())
    ConceptSpread.expected, kept = controlled, ConceptSpread.expected
    try:
        open_prediction(model, torch.tensor([True]))
    finally:
        ConceptSpread.expected = kept
    assert owner._expected_knowing[0, owner._concept_row_of(low), 0] > 0
    assert owner.interpret.referent(word, batch=0) == low
    assert owner.interpret.referent(word, batch=1) == high
    assert owner.interpret.referent(word, batch=None) == high
    assert owner.interpret.forward(word, batch=0) == low
    assert owner.interpret.forward(word, order=1, batch=0) == high


def test_88_scalar_definition_reads_one_after_the_same_six_views(named_scalar_model):
    model, objects = named_scalar_model
    owner = model._concept_owner()
    store = owner._closed_clause_store()
    for name, obj in objects.items():
        word = owner.interpret.word_for_form(name)
        row = owner.definitions.row(word, obj)
        from WordTies import ties
        tie = ties(owner).get(word, obj)
        assert tie.views == 6 and tie.certainty == 1.
        assert float(store.parts.truth_plus[row]) == 1.


def test_88_derived_code_reads_do_not_rebuild_definition_index(named_scalar_model, monkeypatch):
    model, objects = named_scalar_model
    owner = model._concept_owner()
    bank, index, parts = owner.similarity_codebook, owner.definitions, owner._closed_clause_store().parts
    assert bank.mereology is not None and not bank.trust.tracked
    form, obj = next(iter(objects.items()))
    word = owner.interpret.word_for_form(form)
    expected = index.objects(word)
    row = index.row(word, obj)
    before = {name: getattr(parts, name).clone() for name in ('truth_plus', 'truth_minus', 'activity')}
    calls, rebuild = [], index.rebuild
    def record_rebuild():
        calls.append(1)
        return rebuild()
    monkeypatch.setattr(index, 'rebuild', record_rebuild)
    for _ in range(3):
        bank.lookup_rows([owner._concept_row_of(obj)])
        assert index.objects(word) == expected
    assert calls == []
    for name, value in before.items():
        torch.testing.assert_close(getattr(parts, name), value, rtol=0, atol=0)
    # A genuine evidence write still invalidates the same derived index.
    parts.add(obj, word, truth_minus=.2)
    assert index.objects(word) == expected
    assert calls == [1]
    assert float(parts.evidence(row)[1]) == pytest.approx(.2)
    bank.lookup_rows([owner._concept_row_of(obj)])
    assert index.objects(word) == expected
    assert calls == [1]


from test_item18_finish import named_scalar_model


def test_89_written_prediction_is_local_across_constructed_streams():
    from types import SimpleNamespace
    from test_priming_energy import _chain_cs
    from ConceptSpread import open_prediction
    owner, parent_row, _ = _chain_cs()
    low, high = (owner.concept_id_at_row(row) for row in (3, parent_row))
    word = recurrent_word(owner.interpret, [1], [2], form='label')
    from test_thought_model_fixture import trusted_premises
    trusted_premises(owner, ('sym', word), ('sym', low), ('sym', high))
    for obj in (low, high):
        owner.interpret.define(word, obj)
    model = SimpleNamespace(_concept_owner=lambda: owner)
    owner._observed_knowing.zero_()
    owner._observed_knowing[0, 3, 0] = 1.
    # Only stream zero presents this reading; stream one keeps its own state.
    present = torch.tensor([True, False])
    import ConceptSpread
    def controlled(owner_, knowing, **kwargs):
        return knowing.new_zeros(knowing.shape[0], owner_.nVectors, 2).index_copy(
            1, torch.arange(knowing.shape[1]), knowing.detach())
    ConceptSpread.expected, kept = controlled, ConceptSpread.expected
    try:
        open_prediction(model, present)
    finally:
        ConceptSpread.expected = kept
    assert owner._expected_knowing[0, 3, 0] > 0
    assert owner._expected_knowing[1, 3, 0] == 0
    assert owner.interpret.referent(word, batch=0) == low
    assert owner.interpret.referent(word, batch=1) == high
    from MemoryLink import Occurrence, member_order
    for stream, concept in enumerate((low, high)):
        assert member_order(owner, Occurrence((), concept=word), stream=stream) == owner._concept_source_order(concept)
