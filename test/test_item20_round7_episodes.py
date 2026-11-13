"""Detached concept readings, row lifecycle, and content-cued review."""
import copy
import weakref
from types import SimpleNamespace

import pytest
import torch

from EpisodicMemory import select
from Memory import LTM0
from Meaning import ConceptualMeaning
from Understanding import ConceptualField


def test_selection_keeps_signed_readings_in_order_and_zero_stores_nothing():
    # Repeated concepts and contradictory lanes are separate readings;
    # ranking uses magnitude, then retained references return to reading order.
    values = torch.tensor([[[[.5, .5], [.1, .1]]],
                           [[[.4, .6], [.7, .8]]],
                           [[[1., 0.], [0., 0.]]]], requires_grad=True)
    where = torch.tensor([[[0., 2.], [3., 7.]]])
    field = ConceptualField(torch.tensor([9, 4, 2]), values, where, where.long(), None)
    ids, signed = select(field, 4, torch.tensor([True]), [0, 0, 0])
    assert ids.tolist() == [[[2, 4, 4, 4]]]
    torch.testing.assert_close(signed, torch.tensor([[[1., -.6, .7, -.8]]]))
    assert not signed.requires_grad
    empty = select(field, 0, torch.tensor([True]), [0, 0, 0])
    assert all(value.shape[2] == 0 for value in empty)
    absent = select(field, 4, torch.tensor([False]), [0, 0, 0])
    assert (absent[0] == -1).all() and not absent[1].any()


def test_episode_columns_compact_with_rows_keep_first_entry_and_import_empty_legacy():
    from EpisodicMemory import write
    from clock_fixtures import install
    store = LTM0(12, capacity=4, percept_width=12, stream_capacity=2, episodic_concepts=2)
    owner = SimpleNamespace()
    model = install(SimpleNamespace(_episode_counts=dict(written=0)))
    model._open_sentence_slot = 0
    model._sentence_episodes = {0: (torch.tensor([[[2, -1]], [[7, -1]]]),
        torch.tensor([[[-.6, 0.]], [[.8, 0.]]]))}
    owner._model = model
    store._concept_owner = lambda: owner
    # Compaction only needs MTM's forgetting callback for an isolated store.
    owner.mtm = SimpleNamespace(forget=lambda _: None)
    meaning = ConceptualMeaning.from_description(torch.ones(12))
    for b in range(2):
        row = store.append_meaning(meaning, document_key=('episode', b), stream=b)
        write(store, row, tuple(value[b] for value in model._sentence_episodes[0]), new=True)
    first = tuple(getattr(store, name)[0].clone() for name in
                  ('episodic_concepts', 'episodic_values'))
    model._sentence_episodes[0][1].zero_()
    repeated = store.append_meaning(meaning, document_key=('episode', 0), stream=0)
    write(store, repeated, tuple(value[0] for value in model._sentence_episodes[0]), new=False)
    assert repeated == 0
    assert all(torch.equal(getattr(store, name)[0], value) for name, value in zip(
        ('episodic_concepts', 'episodic_values'), first))
    store.compact_rows([1])
    assert store.episodic_concepts[0].tolist() == [[7, -1]]
    assert (store.episodic_concepts[1:] == -1).all()
    assert not store.episodic_values[1:].any()
    saved = copy.deepcopy(store.state_dict())
    saved['truth_schema'].fill_(14)
    for name in ('episodic_concepts', 'episodic_values'):
        saved.pop(name)
    restored = LTM0(12, capacity=4, percept_width=12, stream_capacity=2, episodic_concepts=2)
    with pytest.warns(UserWarning, match='empty episodic'):
        restored.load_state_dict(saved)
    assert (restored.episodic_concepts == -1).all() and not restored.episodic_values.any()
    assert restored.episode_migration == dict(source=14, target=16, entries='empty', dropped=[])


def test_content_query_reinstates_own_stream_and_next_open_attention_replaces_it(tmp_path, monkeypatch):
    from test_item20_round7_time import native
    from reading_fixtures import force_absolute_reading
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from SentenceField import knowing
    model = native(tmp_path, monkeypatch)
    force_absolute_reading(model)
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    commit = model._commit_sentence
    def after_thought(*args, **kwargs):
        model._advance_when_time()
        return commit(*args, **kwargs)
    monkeypatch.setattr(model, '_commit_sentence', after_thought)
    try:
        with torch.no_grad():
            source = model.inputSpace.prepInput(['alpha', 'beta'])
            model.forward(source)
        occupied = (store.episodic_concepts[:len(store)] >= 0).any((-1, -2))
        indices = [int((occupied & store.stream[:len(store), b]).nonzero()[0]) for b in (0, 1)]
        assert indices[0] != indices[1]
        row = indices[0]
        assert int(model.when_encoding.decode_index(store.entry_time(row))) == model._sentence_ticks[0]
        wanted = store.episodic_concepts[row].flatten()
        expected = {int(cid): torch.stack((store.episodic_values[row].flatten()[k].clamp_min(0),
                    (-store.episodic_values[row].flatten()[k]).clamp_min(0)))
                    for k, cid in enumerate(wanted) if int(cid) >= 0}
        before = owner._percept_field
        old_other = {key: before.values[k, 1].sum(0) for k, key in enumerate(before.keys)}
        from WalkTrials import ThoughtTrialState
        trial = ThoughtTrialState(model)
        query = ThoughtLTMCapability(store=weakref.ref(store), equal=lambda *args: 0., tau_id=.9, row=0)
        found = query.query(model._sentence_open_fields[0],
            max_records=store.capacity, work=QueryWorkBudget(2048))
        assert found['frames']
        recalled = owner._percept_field
        assert recalled is not before
        assert not recalled.values[:, 0].any() and not recalled.events[:, 0].any()
        field = knowing(model)
        actual = {int(cid): field.evidence[k, 0].sum(0) for k, cid in enumerate(field.concept_ids)
                  if bool(field.evidence[k, 0].any())}
        assert actual.keys() == expected.keys()
        for concept, pair in expected.items():
            torch.testing.assert_close(actual[concept], pair, rtol=0, atol=0)
        for k, key in enumerate(recalled.keys):
            torch.testing.assert_close(recalled.values[k, 1].sum(0), old_other[key], rtol=0, atol=0)
        live = field.evidence[:, 0].any((0, 2))
        tick = model.when_encoding.decode_index(store.entry_time(row))
        assert (field.when[0, live, 0] == tick).all()
        assert model._episode_counts['reinstatements'] >= 1
        trial.restore(effects_only=True)
        torch.testing.assert_close(knowing(model).evidence, field.evidence)
        from EpisodicMemory import reinstate
        assert not reinstate(store, [], 0)
        assert owner._percept_field is recalled
        with torch.no_grad():
            source = model.inputSpace.prepInput(['beta', 'alpha'])
            model.forward(source)
        assert owner._percept_field is not recalled
        assert (owner._percept_field.when[..., 0] == model._sentence_ticks[0]).all()
    finally:
        model.End()
