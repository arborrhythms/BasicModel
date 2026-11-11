"""Detached percept witnesses, row lifecycle, and actual time-only review."""
import copy
import weakref
from types import SimpleNamespace

import pytest
import torch

from EpisodicMemory import select
from Memory import LTM0
from Meaning import ConceptualMeaning
from PerceptField import PerceptField


def test_selection_preserves_poles_locations_and_index_ties_and_zero_stores_nothing():
    # Key order deliberately differs from percept order. Opposing evidence
    # is strength too; it must not be subtracted from its positive pole.
    values = torch.tensor([[[[.5, .5], [.1, .1]]],
                           [[[.4, .6], [.7, .8]]],
                           [[[1., 0.], [0., 0.]]]], requires_grad=True)
    where = torch.tensor([[[0., 2.], [3., 7.]]])
    field = PerceptField((('bp', 9), ('bp', 4), ('bp', 2)), values,
        values[..., None, :], where[:, :1], where, where.long())
    ids, poles, spans = select(field, 4, torch.tensor([True]))
    assert ids.tolist() == [[4, 2, 9, -1]]
    assert poles.tolist()[0][0] == pytest.approx([.7, .8])
    assert spans.tolist()[0][0] == [3., 7.]
    assert not poles.requires_grad and not spans.requires_grad
    empty = select(field, 0, torch.tensor([True]))
    assert all(value.shape[1] == 0 for value in empty)
    absent = select(field, 4, torch.tensor([False]))
    assert (absent[0] == -1).all() and not absent[1].any()


def test_episode_columns_compact_with_rows_keep_first_entry_and_import_empty_legacy():
    from EpisodicMemory import write
    from clock_fixtures import install
    store = LTM0(12, capacity=4, percept_width=12, stream_capacity=2, episodic_percepts=2)
    owner = SimpleNamespace()
    model = install(SimpleNamespace(_episode_counts=dict(written=0)))
    model._open_sentence_slot = 0
    model._sentence_episodes = {0: (torch.tensor([[2, -1], [7, -1]]),
        torch.tensor([[[.2, .8], [0., 0.]], [[.9, .1], [0., 0.]]]),
        torch.tensor([[[1., 4.], [0., 0.]], [[6., 8.], [0., 0.]]]))}
    owner._model = model
    store._concept_owner = lambda: owner
    # Compaction only needs MTM's forgetting callback for an isolated store.
    owner.mtm = SimpleNamespace(forget=lambda _: None)
    meaning = ConceptualMeaning.from_description(torch.ones(12))
    for b in range(2):
        row = store.append_meaning(meaning, document_key=('episode', b), stream=b)
        write(store, row, tuple(value[b] for value in model._sentence_episodes[0]), new=True)
    first = tuple(getattr(store, name)[0].clone() for name in
                  ('episodic_percepts', 'episodic_poles', 'episodic_where'))
    model._sentence_episodes[0][1].zero_()
    repeated = store.append_meaning(meaning, document_key=('episode', 0), stream=0)
    write(store, repeated, tuple(value[0] for value in model._sentence_episodes[0]), new=False)
    assert repeated == 0
    assert all(torch.equal(getattr(store, name)[0], value) for name, value in zip(
        ('episodic_percepts', 'episodic_poles', 'episodic_where'), first))
    store.compact_rows([1])
    assert store.episodic_percepts[0].tolist() == [7, -1]
    assert (store.episodic_percepts[1:] == -1).all()
    assert not store.episodic_poles[1:].any()
    saved = copy.deepcopy(store.state_dict())
    saved['truth_schema'].fill_(14)
    for name in ('episodic_percepts', 'episodic_poles', 'episodic_where'):
        saved.pop(name)
    restored = LTM0(12, capacity=4, percept_width=12, stream_capacity=2, episodic_percepts=2)
    with pytest.warns(UserWarning, match='empty episodic'):
        restored.load_state_dict(saved)
    assert (restored.episodic_percepts == -1).all() and not restored.episodic_poles.any()
    assert restored.episode_migration == dict(source=14, target=15, entries='empty')


def test_time_only_query_reinstates_own_stream_and_next_open_attention_replaces_it(tmp_path, monkeypatch):
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
        occupied = (store.episodic_percepts[:len(store)] >= 0).any(-1)
        indices = [int((occupied & store.stream[:len(store), b]).nonzero()[0]) for b in (0, 1)]
        assert indices[0] != indices[1]
        row = indices[0]
        assert int(model.when_encoding.decode_index(store.entry_when(row))) == model._sentence_ticks[0]
        wanted = store.episodic_percepts[row]
        expected = {int(percept): store.episodic_poles[row, k].clone()
                    for k, percept in enumerate(wanted) if int(percept) >= 0}
        before = owner._percept_field
        old_other = {key: before.values[k, 1].sum(0) for k, key in enumerate(before.keys)}
        from WalkTrials import ThoughtTrialState
        trial = ThoughtTrialState(model)
        query = ThoughtLTMCapability(store=weakref.ref(store), equal=lambda *args: 0., tau_id=.9, row=0)
        found = query.query(when=store.entry_when(row), window=0,
            max_records=store.capacity, work=QueryWorkBudget(2048))
        assert found['frames']
        recalled = owner._percept_field
        assert recalled is not before
        actual = {key[1]: recalled.values[k, 0].sum(0) for k, key in enumerate(recalled.keys)
                  if bool(recalled.values[k, 0].any())}
        assert actual.keys() == expected.keys()
        for percept, pair in expected.items():
            torch.testing.assert_close(actual[percept], pair, rtol=0, atol=0)
        for k, key in enumerate(recalled.keys):
            torch.testing.assert_close(recalled.values[k, 1].sum(0),
                old_other.get(key, torch.zeros(2)), rtol=0, atol=0)
        live = recalled.values[:, 0].any((0, 2))
        assert torch.equal(recalled.when_band[0, live], store.entry_when(row).expand(int(live.sum()), -1))
        field = knowing(model)
        folded = owner.cs_read_percept_field(recalled)
        torch.testing.assert_close(field.evidence[:len(folded)], folded)
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
