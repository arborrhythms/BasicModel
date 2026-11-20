"""Before the first event, MTM reads its references' current meanings."""
from Memory import STM0
from types import SimpleNamespace

import pytest
import torch


def _references():
    from ClauseRow import attach_clause_index
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from test_review13_subspaces import fixture, word
    owner, bank, _, _ = fixture(meaning_pairs=2)
    store = STM0(7, capacity=8, relation_capacity=8192)
    owner.symbolSpace = SimpleNamespace(ltm_store=store)
    attach_clause_index(SimpleNamespace(), store, owner)
    rows = [word(owner, [(source, 1.)]) for source in (4, 7)]
    concepts = [owner.concept_id_at_row(row) for row in rows]
    # §50: a lexical reading is an allocator-typed word, not a derived kind.
    for concept in concepts:
        owner._concept_allocator.symbol_types.setdefault(concept, set()).add('word')
    from definition_fixtures import agreeing_views
    agreeing_views(owner, *concepts)
    values = bank.lookup_rows(rows).detach()
    occurrences = [store.occurrence_of(store.append_meaning(
        ConceptualMeaning.from_description(value))) for index, value in enumerate(values)]
    # The word's first kept mention opens its token (a word read has its
    # blank object; MTM keeps no pending name).
    key = owner.mtm.open(occurrences[0], kind=None, when=0,
                         value=values[0], reading=concepts[0])
    return owner, bank, store, rows, concepts, occurrences, key


def _move(bank, rows):
    with torch.no_grad():
        bank.mereology.placement[rows] = torch.tensor([[.2, .8], [.7, .3]])
        bank.mereology.placement_witnesses[rows] = 1
        bank.mereology.placement_dirty[rows] = True


def test_bound_identity_joins_both_current_reference_meanings():
    owner, bank, store, rows, concepts, occurrences, key = _references()
    pool = owner.mtm
    pool.bind(key, occurrences[1], kind=None, when=1,
              value=bank.lookup_rows([rows[1]])[0], reading=concepts[1])
    old = pool.value_of(key).clone()
    _move(bank, rows)
    current = bank.lookup_rows(rows)
    start = bank.mereology.percept_event_width
    expected = current[:, start:].amax(0)
    assert not torch.equal(old[start:], expected)
    before = pool.state()
    actual = pool.value_of(key)
    torch.testing.assert_close(actual[start:], expected, rtol=0, atol=0)
    assert not actual.requires_grad
    # Reading the references writes neither a cached value nor a new state.
    assert not hasattr(pool.entries[key], 'value')
    assert pool.entries[key].occurrences == before['entries'][0]['occurrences']
    assert pool.state()['names'] == before['names']
    from PrependedSymbols import symbolize
    torch.testing.assert_close(symbolize(owner, store, key, pool.entries[key])[start:],
                               expected, rtol=0, atol=0)


def test_unlabelled_mention_reads_its_current_occurrence():
    owner, bank, store, rows, _, occurrences, key = _references()
    # Isolate the unlabelled occurrence from its collection's first word.
    # Use the fixture's existing declared word geometry, below this reading.
    _move(bank, rows)
    pool = owner.mtm
    pool.bind(key, occurrences[1], kind=None, when=1,
              value=store.point_of_row(occurrences[1][-1]), reading=None)
    start = bank.mereology.percept_event_width
    index = store.location(occurrences[1])
    store.slots[index, start:] = torch.tensor([.4, .9, .2, .1])
    expected = store.slots[index, start:]
    torch.testing.assert_close(pool.value_of(key)[start:], expected, rtol=0, atol=0)


def test_identity_with_an_event_reads_its_retained_idea():
    owner, bank, store, rows, concepts, occurrences, key = _references()
    pool = owner.mtm
    before = pool.value_of(key).clone()
    held = before.clone()
    start = bank.mereology.percept_event_width
    held[start:] = torch.tensor([.1, .3, .5, .7])
    event = store.occurrence_of(store.append_idea(held))
    pool.event(key, event, verb=concepts[1], role=0, when=1, before=before)
    _move(bank, rows)
    assert not torch.equal(bank.lookup_rows([rows[0]])[0, start:], held[start:])
    torch.testing.assert_close(pool.value_of(key), held, rtol=0, atol=0)


def test_first_event_keeps_the_computed_forward_operand_only():
    owner, bank, store, rows, concepts, occurrences, key = _references()
    _move(bank, rows)
    before = owner.mtm.value_of(key).clone()
    start = bank.mereology.percept_event_width
    event = store.occurrence_of(store.append_idea(before))
    owner.mtm.event(key, event, verb=concepts[1], role=0, when=1, before=before)
    bank.mereology.placement[rows[0]].zero_()
    bank.mereology.placement_dirty[rows[0]] = True
    torch.testing.assert_close(owner.mtm.entries[key].events[-1].before, before)
    torch.testing.assert_close(owner.mtm.value_of(key)[start:], before[start:], rtol=0, atol=0)


@pytest.mark.parametrize('missing', ['concept', 'occurrence', 'content'])
def test_missing_identity_reference_fails_loudly(monkeypatch, missing):
    owner, _, store, _, _, occurrences, key = _references()
    if missing == 'concept':
        monkeypatch.setattr(owner, '_concept_row_of', lambda identity: None)
    else:
        from Memory import Name
        owner.mtm.names = {occurrences[0]: Name(key, None)}
        if missing == 'occurrence':
            monkeypatch.setattr(store, 'location', lambda reference: None)
        else:
            monkeypatch.setattr(store, 'point_of_row', lambda reference: None)
    with pytest.raises(ValueError, match='unavailable memory member'):
        owner.mtm.value_of(key)


def test_event_on_alternative_reads_each_timelines_own_member():
    owner, bank, store, rows, concepts, occurrences, key = _references()
    pool = owner.mtm
    _move(bank, rows)
    expected = pool.value_of(key).clone()
    actual = store.occurrence_of(store.append_idea(expected))
    pool.event(key, actual, verb=concepts[1], role=0, when=1, before=expected)
    branch = pool.alternative(key, when=1)
    alternative = expected.clone()
    alternative[bank.mereology.percept_event_width:] = .9
    other = store.occurrence_of(store.append_idea(alternative))
    pool.event(key, other, verb=concepts[1], role=0, when=2, timeline=branch, before=expected)
    bank.mereology.placement[rows].zero_()
    bank.mereology.placement_dirty[rows] = True
    torch.testing.assert_close(pool.value_of(key), expected, rtol=0, atol=0)
    torch.testing.assert_close(pool.value_of(key, timeline=branch), alternative, rtol=0, atol=0)


def test_consolidation_candidate_keeps_its_live_support():
    from FamiliarTier import centroid
    from PrependedSymbols import symbolize
    owner, bank, store, rows, concepts, occurrences, _ = _references()
    pool = owner.mtm
    support = dict(zip(rows, (1., 2.)))
    key = pool.open_candidate(order=0, when=0, value=centroid(owner, support), support=support)
    _move(bank, rows)
    expected = centroid(owner, support).detach()
    torch.testing.assert_close(pool.value_of(key), expected, rtol=0, atol=0)
    assert not pool.value_of(key).requires_grad and not hasattr(pool, 'update')
    torch.testing.assert_close(symbolize(owner, store, key, pool.entries[key]), expected, rtol=0, atol=0)

