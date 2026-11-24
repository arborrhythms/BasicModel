"""Detached concept readings, row lifecycle, and content-cued review."""
import copy
import weakref
from types import SimpleNamespace

import pytest
import torch

from EpisodicMemory import select
from Memory import LTM0
from architecture import WHERE_WIRES, WHEN_WIRES
from Meaning import ConceptualMeaning
from Understanding import ConceptualField


def test_selection_keeps_strongest_signed_lanes_and_zero_stores_nothing():
    # The operated set holds each concept's strongest reading on each lane.
    values = torch.tensor([[[[.5, .5], [.1, .1]]],
                           [[[.4, .6], [.7, .8]]],
                           [[[1., 0.], [0., 0.]]]], requires_grad=True)
    where = torch.tensor([[[0., 2.], [3., 7.]]])
    field = ConceptualField(torch.tensor([9, 4, 2]), values, where, where.long(), None)
    ids, signed = select(field, 4, torch.tensor([True]), unkept=())
    assert ids.tolist() == [[2, 4, 4, 9]]
    torch.testing.assert_close(signed, torch.tensor([[1., -.8, .7, .5]]))
    assert not signed.requires_grad
    empty = select(field, 0, torch.tensor([True]), unkept=())
    assert all(value.shape[1] == 0 for value in empty)
    absent = select(field, 4, torch.tensor([False]), unkept=())
    assert (absent[0] == -1).all() and not absent[1].any()


def test_episode_columns_compact_keep_prior_entries_and_load_empty_entries():
    from EpisodicMemory import write
    from clock_fixtures import install
    # Four wires of what and the percept band, all of it the percept.
    width = 4 + WHERE_WIRES + WHEN_WIRES
    store = LTM0(width, capacity=4, percept_width=width, episodic_concepts=2)
    owner = SimpleNamespace()
    model = install(SimpleNamespace(_episode_counts=dict(written=0)))
    model._open_sentence_slot = 0
    model._sentence_episodes = {0: (torch.tensor([[2, -1], [7, -1]]),
        torch.tensor([[-.6, 0.], [.8, 0.]]))}
    owner._model = model
    store._concept_owner = lambda: owner
    # Compaction only needs MTM's forgetting callback for an isolated store.
    from Memory import MediumTermMemory
    owner.mtm = MediumTermMemory()
    meaning = ConceptualMeaning.from_description(torch.ones(width))
    for b in range(2):
        row = store.append_meaning(meaning)
        write(store, row, tuple(value[b] for value in model._sentence_episodes[0]))
    first = tuple(getattr(store, name)[0].clone() for name in
                  ('episodic_concepts', 'episodic_values'))
    model._sentence_episodes[0][1].zero_()
    repeated = store.append_meaning(meaning)
    write(store, repeated, tuple(value[0] for value in model._sentence_episodes[0]))
    assert repeated == 2
    assert all(torch.equal(getattr(store, name)[0], value) for name, value in zip(
        ('episodic_concepts', 'episodic_values'), first))
    store.compact_rows([1])
    assert store.episodic_concepts[0].tolist() == [7, -1]
    assert (store.episodic_concepts[1:] == -1).all()
    assert not store.episodic_values[1:].any()
    saved = store.state_dict()
    saved['episodic_concepts'].fill_(-1)
    saved['episodic_values'].zero_()
    restored = LTM0(width, capacity=4, percept_width=width, episodic_concepts=2)
    restored.load_state_dict(saved)
    assert (restored.episodic_concepts == -1).all() and not restored.episodic_values.any()
    assert int(restored.truth_schema) == 19


def test_content_query_retains_entries_in_requesting_stm_and_preserves_perception(tmp_path, monkeypatch):
    from test_item20_round7_time import native
    from reading_fixtures import force_absolute_reading
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from SentenceField import knowing
    model = native(tmp_path, monkeypatch)
    force_absolute_reading(model)
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    committed = {}
    written_times = {}
    write = store.write_clause
    def recorded(clause, **kwargs):
        tick = model.present()
        row = write(clause, **kwargs)
        if isinstance(row, int) and kwargs.get('stream', -1) >= 0:
            committed[kwargs['stream']] = row
            written_times[kwargs['stream']] = tick
        return row
    monkeypatch.setattr(store, 'write_clause', recorded)
    commit = model._commit_sentence
    def after_thought(*args, **kwargs):
        model._advance_when_time()
        return commit(*args, **kwargs)
    monkeypatch.setattr(model, '_commit_sentence', after_thought)
    try:
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_forms
        nominated_forms(model, ['alpha', 'beta'])
        with torch.no_grad():
            source = model.inputSpace.prepInput(['alpha', 'beta'])
            model.forward(source)
        indices = [committed[b] for b in (0, 1)]
        assert all(bool((store.episodic_concepts[index] >= 0).any()) for index in indices)
        assert indices[0] != indices[1]
        row = indices[0]
        assert int(model.when_encoding.decode_index(store.entry_time(row))) == written_times[0]
        assert written_times[0] >= model._sentence_ticks[0]
        from ThoughtChunks import held
        from ThoughtReferences import bindings
        from EpisodicMemory import entry_set
        before, field = owner._percept_field, knowing(model)
        old_other = held(owner.stm, 1)
        from WalkTrials import ThoughtTrialState
        trial = ThoughtTrialState(model)
        # The query's key is the top of STM's stack (sigma-pi §21, rule 60):
        # the kept idea, with its entry, is taken up as a thought's focus.
        from ThoughtChunks import put
        from Extension import Node
        from dataclasses import replace
        put(owner.stm, replace(store.meaning_of(row), extension=(Node(entry_set(store, row)),)), row=0, pin=True)
        query = ThoughtLTMCapability(store=weakref.ref(store), equal=lambda *args: 0., tau_id=.9, row=0)
        found = query.query(model._last_sentence_understanding.understanding_field,
            max_records=store.capacity, work=QueryWorkBudget(2048))
        assert found['frames'] and found['value'] is None
        recalled = owner._percept_field
        assert recalled is before
        torch.testing.assert_close(knowing(model).evidence, field.evidence, rtol=0, atol=0)
        views = {bindings(value).get('_rule_row'): (when, value) for when, value in held(owner.stm, 0)}
        for frame in found['frames']:
            when, value = views[frame['occurrence']]
            assert when == tuple(frame['time'].tolist())
            original = entry_set(store, frame['index'])
            torch.testing.assert_close(value.extension[0].field.ids, original.ids, rtol=0, atol=0)
            torch.testing.assert_close(value.extension[0].field.values, original.values, rtol=0, atol=0)
        for (when, value), (old_when, old_value) in zip(held(owner.stm, 1), old_other, strict=True):
            assert when == old_when
            torch.testing.assert_close(value.roles, old_value.roles, rtol=0, atol=0)
        trial.restore(effects_only=True)
        assert {bindings(value).get('_rule_row') for _,value in held(owner.stm, 0)} == set(views)
        torch.testing.assert_close(knowing(model).evidence, field.evidence)
        with torch.no_grad():
            source = model.inputSpace.prepInput(['beta', 'alpha'])
            model.forward(source)
        assert owner._percept_field is not recalled
        assert (owner._percept_field.when[..., 0] == model._sentence_ticks[0]).all()
    finally:
        model.End()
