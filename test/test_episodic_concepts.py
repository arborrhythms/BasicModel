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
    with pytest.warns(UserWarning, match='dropped.*' + legacy):
        store.load_state_dict(state)
    for name in ('episodic_percepts', 'episodic_poles', 'episodic_where'):
        assert not hasattr(store, name)
        assert name not in store.row_columns
        assert not any(name in key for key in store.state_dict())
    assert (store.episodic_concepts == -1).all()
    assert not store.episodic_values.any()
    assert store.episode_migration['dropped'] == [legacy]


def test_basic_concept_readings_keep_largest_signed_values_in_reading_order():
    from EpisodicMemory import select
    from Understanding import ConceptualField
    values = torch.tensor([[[[.3, 0.], [.9, 0.]]],
                           [[[0., .8], [0., 0.]]],
                           [[[1., 1.], [1., 0.]]],
                           [[[1., 1.], [0., 0.]]]], requires_grad=True)
    field = ConceptualField(torch.tensor([30, 10, 20, 40]), values,
        torch.tensor([[[0, 2], [3, 5]]]), torch.tensor([[[1, 2], [1, 2]]]), None)
    ids, signed = select(field, 4, torch.tensor([True]), [0, 0, 1, 0])
    assert ids.tolist() == [[[10, 40, 40, 30]]]
    torch.testing.assert_close(signed, torch.tensor([[[-.8, 1., -1., .9]]]))
    assert not signed.requires_grad
    selected, _ = select(field, 2, torch.tensor([True]), [0, 0, 1, 0])
    assert selected.tolist() == [[[40, 40]]]
    assert all(t.shape[2] == 0 for t in select(field, 0, torch.tensor([True]), [0, 0, 1, 0]))


def test_recalled_concepts_decode_directly_and_withdrawn_ids_are_counted(tmp_path, monkeypatch):
    from test_item20_round7_time import native
    from reading_fixtures import force_absolute_reading
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from SentenceField import knowing
    from EpisodicMemory import reinstate
    model = native(tmp_path, monkeypatch)
    force_absolute_reading(model)
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    try:
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(['alpha alpha beta']))
        opened = model._sentence_open_fields[0]
        entries = (store.episodic_concepts[:len(store)] >= 0).any((-1, -2))
        row = int(entries.nonzero()[0])
        assert int(entries.sum()) == 1
        assert 0 < int((store.episodic_concepts[row] >= 0).sum()) <= 4 * store.episodic_orders
        assert all(owner._row_order(owner._csw_row_of(cid)) == order
                   for order, slice in enumerate(store.episodic_concepts[row].tolist()) for cid in slice if cid >= 0)
        before = model.reverse(model.conceptualSpace.subspace.carrier_like(), field=opened)
        def forbidden(*args, **kwargs):
            raise AssertionError('direct episodic reconstruction invoked the grammar')
        monkeypatch.setattr(model, 'generate', forbidden)
        monkeypatch.setattr(model.languageSpace, 'reverse_binary_step', forbidden)
        query = ThoughtLTMCapability(store=weakref.ref(store), equal=lambda *args: 0., tau_id=.9, row=0)
        found = query.query(opened, max_records=store.capacity,
                            work=QueryWorkBudget(2048))
        assert found['frames']
        recalled = knowing(model)
        def readings(field):
            ids = field.concept_ids[:, None] if field.concept_ids.ndim == 1 else field.concept_ids
            return {(int(ids[c, 0]), tuple(field.where[0, r].tolist())): field.evidence[c, 0, r]
                    for c in range(len(ids)) for r in range(field.evidence.shape[2])
                    if bool(field.evidence[c, 0, r].any())}
        original, actual = readings(opened), readings(recalled)
        assert original.keys() == actual.keys()
        for key in original:
            torch.testing.assert_close(original[key], actual[key], rtol=0, atol=0)
        after = model.reverse(model.conceptualSpace.subspace.carrier_like(), field=recalled)
        torch.testing.assert_close(before.materialize(), after.materialize(), rtol=0, atol=0)
        assert not owner._percept_field.values.any()
        # A missing surface keeps the reference with an empty extent; positions
        # come from the stored reading order, never from a saved input bracket.
        surface = owner.word_surface_for_row
        first = int(store.episodic_concepts[row, 0, 0])
        missing = int((store.episodic_concepts[row] == first).sum())
        monkeypatch.setattr(owner, 'word_surface_for_row',
            lambda slot: None if owner.concept_id_at_row(slot) == first else surface(slot))
        assert reinstate(store, [{'index': row}], 0)
        assert model._episode_counts['surfaces_missing'] == missing
        assert knowing(model).where[0, 0].tolist() == [0, 0]
        previous = model._episode_counts['ids_skipped']
        store.episodic_concepts[row, 0, 0] = 10**9
        assert reinstate(store, [{'index': row}], 0)
        assert model._episode_counts['ids_skipped'] == previous + 1
        assert 10**9 not in knowing(model).concept_ids.tolist()
    finally:
        model.End()


def test_path_keep_selects_concepts_and_readings_together(tmp_path, monkeypatch):
    from test_item20_round7_time import native
    from Understanding import ConceptualField
    from SentenceField import knowing, publish
    from WalkTrials import ThoughtTrialState
    model = native(tmp_path, monkeypatch)
    try:
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
