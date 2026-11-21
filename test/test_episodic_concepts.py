"""An episode is its original conceptual reading, independent of a new percept pass."""
import copy
import weakref
import pytest
import torch


@pytest.mark.parametrize('legacy', ('episodic_percepts', 'episodic_poles', 'episodic_where'))
def test_legacy_percept_entry_is_dropped_and_named(legacy):
    from Memory import LTM0
    store = LTM0(12, capacity=4, percept_width=12, episodic_concepts=2)
    state = copy.deepcopy(store.state_dict())
    state[legacy] = torch.ones(4, 2) if legacy == 'episodic_percepts' else torch.ones(4, 2, 2)
    with pytest.warns(UserWarning, match='dropped.*' + legacy) as warnings:
        store.load_state_dict(state)
    for name in ('episodic_percepts', 'episodic_poles', 'episodic_where'):
        assert not hasattr(store, name)
        assert name not in store.columns
        assert not any(name in key for key in store.state_dict())
    assert (store.episodic_concepts == -1).all()
    assert not store.episodic_values.any()
    assert len(warnings) == 1 and legacy in str(warnings[0].message)


def test_all_orders_keep_largest_signed_values_in_one_flat_entry():
    from EpisodicMemory import select
    from Understanding import ConceptualField
    values = torch.tensor([[[[.3, 0.], [.9, 0.]]],
                           [[[0., .8], [0., 0.]]],
                           [[[1., 1.], [1., 0.]]],
                           [[[1., 1.], [0., 0.]]]], requires_grad=True)
    field = ConceptualField(torch.tensor([30, 10, 20, 40]), values,
        torch.tensor([[[0, 2], [3, 5]]]), torch.tensor([[[1, 2], [1, 2]]]), None)
    ids, signed = select(field, 4, torch.tensor([True]), unkept=())
    assert ids.tolist() == [[20, 20, 40, 40]]
    torch.testing.assert_close(signed, torch.tensor([[1., -1., 1., -1.]]))
    assert not signed.requires_grad
    selected, _ = select(field, 2, torch.tensor([True]), unkept=())
    assert selected.tolist() == [[20, 20]]
    assert all(t.shape[1] == 0 for t in select(field, 0, torch.tensor([True]), unkept=()))
    # A concept that holds of every reading (the declared classes) is not
    # kept, and the next strongest takes its place.
    ids, signed = select(field, 4, torch.tensor([True]), unkept=(20,))
    assert ids.tolist() == [[40, 40, 30, 10]]
    torch.testing.assert_close(signed, torch.tensor([[1., -1., .9, -.8]]))


def test_recalled_flat_extension_is_retained_in_stm_without_running_the_grammar(tmp_path, monkeypatch):
    from test_item20_round7_time import native
    from reading_fixtures import force_absolute_reading
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from SentenceField import knowing
    from ThoughtChunks import held
    from ThoughtReferences import bindings
    from EpisodicMemory import entry_set
    model = native(tmp_path, monkeypatch)
    force_absolute_reading(model)
    # The episodic entry needs the sentence's words read; since the loop of
    # looks that is the chooser's draw, so they are read in source order.
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    try:
        # MTM nominates the forms on their concept_mint_count-th reading
        # (sigma-pi §21, rule 20); the episode is that reading's.
        from nomination_fixtures import readings_before_nomination
        readings_before_nomination(model, lambda: model.forward(model.inputSpace.prepInput(['alpha alpha beta'])))
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(['alpha alpha beta']))
        opened = model._sentence_open_fields[0]
        entries = (store.episodic_concepts[:len(store)] >= 0).any(-1)
        row = int(entries.nonzero()[0])
        assert bool(entries.any())
        assert 0 < int((store.episodic_concepts[row] >= 0).sum()) <= 4
        assert all(owner._row_order(owner._concept_row_of(cid)) >= 0
                   for cid in store.episodic_concepts[row].tolist() if cid >= 0)
        def forbidden(*args, **kwargs):
            raise AssertionError('direct episodic reconstruction invoked the grammar')
        monkeypatch.setattr(model, 'generate', forbidden)
        monkeypatch.setattr(model.languageSpace, 'reverse_binary_step', forbidden)
        before, percepts = knowing(model), owner._percept_field
        # The query's key is the top of STM's stack (sigma-pi §21, rule 60):
        # the kept idea, with its entry, is taken up as a thought's focus.
        from ThoughtChunks import put
        from Extension import Node
        from dataclasses import replace
        put(owner.stm, replace(store.meaning_of(row), extension=(Node(entry_set(store, row)),)), row=0, pin=True)
        query = ThoughtLTMCapability(store=weakref.ref(store), equal=lambda *args: 0., tau_id=.9, row=0)
        found = query.query(opened, max_records=store.capacity, work=QueryWorkBudget(2048))
        assert found['frames'] and found['value'] is None
        views = {bindings(value).get('_rule_row'): (when, value)
                 for when, value in held(owner.stm, 0)}
        for frame in found['frames']:
            when, value = views[frame['occurrence']]
            assert when == tuple(frame['time'].tolist())
            original = entry_set(store, frame['index'])
            actual = value.extension[0].field
            torch.testing.assert_close(actual.ids, original.ids, rtol=0, atol=0)
            torch.testing.assert_close(actual.values, original.values, rtol=0, atol=0)
        torch.testing.assert_close(knowing(model).evidence, before.evidence, rtol=0, atol=0)
        assert owner._percept_field is percepts
    finally:
        model.End()


def test_path_keep_selects_concepts_and_readings_together(tmp_path, monkeypatch):
    from test_item20_round7_time import native
    from Understanding import ConceptualField
    from SentenceField import knowing, publish
    from WalkTrials import ThoughtTrialState
    model = native(tmp_path, monkeypatch)
    try:
        from nomination_fixtures import readings_before_nomination
        readings_before_nomination(model, lambda: model.forward(model.inputSpace.prepInput(['alpha beta', 'beta alpha'])))
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(['alpha beta', 'beta alpha']))
        owner = model._concept_owner()
        ids = knowing(model).concept_ids.new_tensor(sorted(
            owner.interpret.word_for_form(word) for word in ('alpha', 'beta')))
        assert ids[0] != ids[1]
        first = torch.full((2, 2, 1, 2), .2, requires_grad=True)
        second = torch.full((1, 2, 3, 2), .7, requires_grad=True)
        predictions = [torch.tensor([[[.1]], [[.2]]], requires_grad=True),
                       torch.tensor([[[.3, .4, .5]], [[.6, .7, .8]]], requires_grad=True)]
        states = []
        for index, (cids, values) in enumerate(((ids, first), (ids[1:], second))):
            model._sentence_answer_predictions = predictions[index]
            where = torch.arange(values.shape[2]).repeat(2, 1)
            field = ConceptualField(cids, values, torch.stack((where, where + 1), -1),
                                    torch.ones(2, values.shape[2], 2, dtype=torch.long), None)
            publish(model, field)
            owner._cs_field_concept_ids = field.concept_ids
            owner._cs_field_where, owner._cs_field_when = field.where, field.when
            states.append(ThoughtTrialState(model))
        ThoughtTrialState.keep(states, torch.tensor([0, 1]))
        kept = knowing(model)
        torch.testing.assert_close(kept.evidence[:, 0, :1], first[:, 0])
        torch.testing.assert_close(kept.evidence[1:, 1], second[:, 1])
        assert not kept.evidence[0, 1].any() and not kept.evidence[:, 0, 1:].any()
        assert owner._cs_field_where.shape == (2, 3, 2)
        output = model._sentence_answer_predictions
        torch.testing.assert_close(output, torch.tensor([[[.1, 0., 0.]], [[.6, .7, .8]]]), rtol=0, atol=0)
        (kept.evidence.sum() + output.sum()).backward()
        assert first.grad[:, 0].all() and not first.grad[:, 1].any()
        assert second.grad[:, 1].all() and not second.grad[:, 0].any()
        assert predictions[0].grad[0].all() and not predictions[0].grad[1].any()
        assert predictions[1].grad[1].all() and not predictions[1].grad[0].any()
    finally:
        model.End()
