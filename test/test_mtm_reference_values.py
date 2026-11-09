"""Before the first event, MTM reads its references' current meanings."""
from types import SimpleNamespace

import pytest
import torch


def _references():
    from ClauseRow import attach_clause_index
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from test_review13_subspaces import fixture, word
    owner, bank, _, _ = fixture(meaning_pairs=2)
    store = LTM0(7, capacity=8)
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
        ConceptualMeaning.from_description(value), document_key='reference-values',
        sentence_index=index, order=0)) for index, value in enumerate(values)]
    key = owner.mtm.named(concepts[0], value=values[0])
    owner.mtm.bind(key, occurrences[0], kind=None, when=0,
                   value=values[0], reading=concepts[0])
    return owner, bank, store, rows, concepts, occurrences, key


def _move(bank, rows):
    with torch.no_grad():
        bank.mereology.placement[rows] = torch.tensor([[.2, .8], [.7, .3]])
        bank.mereology.placement_witnesses[rows] = 1
        bank.mereology.placement_valid[rows] = True
        bank.mereology.placement_dirty[rows] = True


def test_bound_identity_joins_both_current_reference_meanings():
    owner, bank, store, rows, concepts, occurrences, key = _references()
    pool = owner.mtm
    pool.bind(key, occurrences[1], kind=None, when=1,
              value=bank.lookup_rows([rows[1]])[0], reading=concepts[1])
    old = pool.entries[key].value.clone()
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
    torch.testing.assert_close(pool.entries[key].value, old, rtol=0, atol=0)
    assert pool.state()['names'] == before['names']
    from PrependedSymbols import symbolize
    torch.testing.assert_close(symbolize(owner, store, key, pool.entries[key])[start:],
                               expected, rtol=0, atol=0)


def test_unlabelled_mention_reads_its_current_occurrence():
    owner, bank, store, _, _, occurrences, key = _references()
    pool = owner.mtm
    pool.bind(key, occurrences[1], kind=None, when=1,
              value=store.point_of_row(occurrences[1][-1]), reading=None)
    start = bank.mereology.percept_event_width
    index = store.location(occurrences[1])
    store.slots[index, start:] = torch.tensor([.4, .9, .2, .1])
    expected = store.slots[index, start:]
    torch.testing.assert_close(pool.value_of(key)[start:], expected, rtol=0, atol=0)


def test_identity_with_an_event_keeps_its_stored_state():
    owner, bank, _, rows, concepts, occurrences, key = _references()
    pool = owner.mtm
    pool.event(key, occurrences[0], verb=concepts[1], role=0, when=1)
    held = pool.entries[key].value.clone()
    start = bank.mereology.percept_event_width
    held[start:] = torch.tensor([.1, .3, .5, .7])
    pool.update(key, held, when=1)
    _move(bank, rows)
    assert not torch.equal(bank.lookup_rows([rows[0]])[0, start:], held[start:])
    torch.testing.assert_close(pool.value_of(key), held, rtol=0, atol=0)


def test_first_event_stores_the_computed_start_state():
    owner, bank, _, rows, concepts, occurrences, key = _references()
    _move(bank, rows)
    start = bank.mereology.percept_event_width
    expected = bank.lookup_rows([rows[0]])[0, start:].detach().clone()
    owner.mtm.event(key, occurrences[0], verb=concepts[1], role=0, when=1)
    bank.mereology.placement[rows[0]].zero_()
    bank.mereology.placement_dirty[rows[0]] = True
    torch.testing.assert_close(owner.mtm.value_of(key)[start:], expected, rtol=0, atol=0)


@pytest.mark.parametrize('missing', ['concept', 'occurrence', 'content'])
def test_missing_identity_reference_fails_loudly(monkeypatch, missing):
    owner, _, store, _, _, occurrences, key = _references()
    if missing == 'concept':
        monkeypatch.setattr(owner, '_csw_row_of', lambda identity: None)
    else:
        from Memory import Name
        owner.mtm.names = {occurrences[0]: Name(key, None)}
        if missing == 'occurrence':
            monkeypatch.setattr(store, 'location', lambda reference: None)
        else:
            monkeypatch.setattr(store, 'point_of_row', lambda reference: None)
    with pytest.raises(ValueError, match='identity .*reference .*unavailable|identity .*unavailable occurrence reference|identity .*reference .*no content'):
        owner.mtm.value_of(key)


def test_event_on_alternative_preserves_each_timelines_state():
    owner, bank, _, rows, concepts, occurrences, key = _references()
    pool = owner.mtm
    branch = pool.alternative(key, when=1)
    _move(bank, rows)
    expected = pool.value_of(key).clone()
    pool.event(key, occurrences[0], verb=concepts[1], role=0, when=1, timeline=branch)
    alternative = expected.clone()
    alternative[bank.mereology.percept_event_width:] = .9
    pool.update(key, alternative, when=1, timeline=branch)
    bank.mereology.placement[rows].zero_()
    bank.mereology.placement_dirty[rows] = True
    torch.testing.assert_close(pool.value_of(key), expected, rtol=0, atol=0)
    torch.testing.assert_close(pool.value_of(key, timeline=branch), alternative, rtol=0, atol=0)


def test_consolidation_candidate_keeps_its_live_support_and_event_state():
    from FamiliarTier import centroid
    from PrependedSymbols import symbolize
    owner, bank, store, rows, concepts, occurrences, _ = _references()
    pool = owner.mtm
    support = dict(zip(rows, (1., 2.)))
    key = pool.open_candidate(order=0, when=0, value=centroid(owner, support), support=support)
    _move(bank, rows)
    expected = centroid(owner, support).detach()
    torch.testing.assert_close(pool.value_of(key), expected, rtol=0, atol=0)
    assert not pool.value_of(key).requires_grad
    pool.event(key, occurrences[0], verb=concepts[1], role=0, when=1)
    held = torch.full_like(expected, .6)
    pool.update(key, held, when=1)
    torch.testing.assert_close(symbolize(owner, store, key, pool.entries[key]), held, rtol=0, atol=0)


def test_second_sentence_composes_the_name_after_kept_close_moves_placement(monkeypatch):
    from pathlib import Path
    import Models
    import util
    from Language import LanguageSpace
    from test_mm_xor import _fresh_model
    from test_reverse_traversal import _select_completed_binary_path
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    _select_completed_binary_path(model)
    model.candidate_attention = None
    observed = []
    native_choose = LanguageSpace.choose_operation

    def choose(language, *args, **kwargs):
        result = native_choose(language, *args, **kwargs)
        selection = result[0]
        if bool((selection.applied & (selection.kind == 1)).any()):
            observed.append((selection.applied.detach().clone(), result[3].detach().clone()))
        return result

    monkeypatch.setattr(LanguageSpace, 'choose_operation', choose)
    try:
        raw, targets = next(iter(data.data_loader(split='train', num_streams=4)))
        batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(targets)
        model.train()
        with torch.no_grad():
            model.forward(batch[0])
        owner = model._concept_owner()
        bank, pool = owner.similarity_codebook, owner.mtm
        start = bank.mereology.percept_event_width
        record = model._last_sentence_understanding
        word_rows = torch.stack([record.word_rows[b, record.word_valid[b]]
                                 for b in range(len(record.root))])
        assert word_rows.shape == (4, 2)
        expected = bank.lookup_rows(word_rows).detach().clone()
        keys = [[pool.token_of(('name', owner.concept_id_at_row(int(row)))) for row in pair]
                for pair in word_rows]
        assert all(key is not None for pair in keys for key in pair), (keys, pool.names, word_rows)
        assert all(not any(branch.events for branch in pool.entries[key].timelines)
                   for pair in keys for key in pair)
        old = torch.stack([torch.stack([pool.entries[key].value for key in pair]) for pair in keys])
        assert not torch.equal(old[..., start:], expected[..., start:])
        assert bank.mereology.placement_witnesses[word_rows].gt(0).all()
        observed.clear()
        with torch.no_grad():
            model.forward(batch[0])
        assert observed
        seen = torch.zeros(4, dtype=torch.bool)
        for active, operands in observed:
            for stream in active.nonzero().flatten().tolist():
                seen[stream] = True
                actual = operands[stream, :, start:]
                wanted = expected[stream, :, start:]
                matches = (actual[:, None] == wanted[None]).all(-1)
                assert matches.any(0).all() and matches.any(1).all()
        assert seen.all()
    finally:
        model.End()
