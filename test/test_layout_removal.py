"""Item 3.0 step 1: semantic compression, tagged chains and percept time."""
import copy
from dataclasses import replace

import pytest
import torch

from ClauseRow import Clause, Composition
from LTM1 import LTM1, OCCURRENCE, SYMBOL, RELATION
from Layers import LTM0
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget
from RowTime import encoder
from part_fixtures import native_store, symbol


def tree(point, left, right, operation='lift'):
    return Composition(point, (left, right), operation=operation)


def field(term):
    return Clause(ConceptualMeaning.from_description(term.point), point=term.point, composition=term)


def test_resolved_sentence_compresses_to_one_idea_without_composing_edges():
    store, ids = native_store(width=16, points=torch.eye(16)[:3])
    leaves = [Composition(torch.eye(16)[i], reference=ref) for i, ref in enumerate(ids)]
    vp = tree(torch.eye(16)[3], leaves[1], leaves[2], 'verb')
    clause = field(tree(torch.eye(16)[4], leaves[0], vp))
    row = store.write_clause(clause)
    assert store.slots.shape == (32, 16)
    assert len(store) == 1 and len(store.parts) == 0
    assert not hasattr(store, 'role_mask') and not hasattr(store, 'when')
    assert 'role_refs' not in store._semantic_rows[int(store.address_keys[row])]
    assert dict(store.meaning_of(row).bindings)['_identity'] == ('sym', ids[0])
    torch.testing.assert_close(store.slots[row, :12], clause.point[:12])
    assert bool(store.entry_when(row).ne(0).any())


def pending_chain():
    store, (said,) = native_store(width=16, points=torch.eye(16)[3:4])
    rows = [store.append_meaning(ConceptualMeaning.from_description(torch.eye(16)[i]),
        kind='unverified', sentence_index=i, when=False) for i in range(3)]
    owner = store._concept_owner()
    owner.mtm.open(store.occurrence_of(rows[2]), kind=None, value=store.slots[rows[2]], when=0, pending=True)
    he, she, unknown = [Composition(store.slots[row].clone(), reference=int(store.address_keys[row])) for row in rows]
    word = Composition(torch.eye(16)[3], reference=said)
    inner = tree(torch.eye(16)[4], word, unknown, 'verb')
    middle = tree(torch.eye(16)[5], she, inner)
    outer = tree(torch.eye(16)[6], word, middle, 'verb')
    root = field(tree(torch.eye(16)[7], he, outer))
    written = {}
    result = store.write_clause(root, written_rows=written)
    return store, rows, said, root, written, result


def test_full_idea_inventory_can_rewitness_its_existing_sentence():
    store, ids = native_store(width=16, points=torch.eye(16)[:2])
    clause = field(tree(torch.eye(16)[2], *(
        Composition(torch.eye(16)[i], reference=cid) for i, cid in enumerate(ids))))
    row = store.write_clause(clause, document_key='same', sentence_index=0)
    for index in range(1, store.capacity):
        store.append_idea(torch.eye(16)[0], document_key='filler', sentence_index=index)
    before = store.entry_when(row).clone()
    assert store.write_clause(clause, document_key='same', sentence_index=0) == row
    assert len(store) == store.capacity and int(store.witness_count[row]) == 2
    torch.testing.assert_close(store.entry_when(row), before, rtol=0, atol=0)


@pytest.mark.parametrize('clause_write', (False, True))
def test_stream_overflow_rejects_before_any_durable_write(clause_write):
    store, ids = native_store(width=16, points=torch.eye(16)[:2])
    clause = field(tree(torch.eye(16)[2], *(
        Composition(torch.eye(16)[i], reference=cid) for i, cid in enumerate(ids))))
    before = copy.deepcopy(store.state_dict())
    extras = copy.deepcopy(store.semantic_extras())
    with pytest.raises(OverflowError, match='stream exceeds'):
        if clause_write:
            store.write_clause(clause, stream=store.stream.shape[1])
        else:
            store.append_meaning(clause.meaning, stream=store.stream.shape[1])
    for name, original in before.items():
        torch.testing.assert_close(store.state_dict()[name], original, rtol=0, atol=0)
    assert store.semantic_extras() == extras
    assert len(store) == len(store.parts) == 0


def test_description_chain_requires_a_live_relation_and_a_metered_read():
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkExhausted
    store, (left, right) = native_store(points=torch.eye(4)[:2])
    reference = store.append_relation(left, right)
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: 0., tau_id=.9)
    work = QueryWorkBudget(1)
    meaning, scanned = reader.resolve_description(reference, row=0, max_records=1, work=work)
    assert scanned == work.spent == 1
    assert meaning.role_refs[:2] == (('sym', left), ('sym', right))
    with pytest.raises(QueryWorkExhausted):
        reader.resolve_description(reference, row=0, max_records=1, work=work)
    with pytest.raises(ValueError, match='description chain'):
        reader.resolve_description(('sym', left), row=0, max_records=1, work=QueryWorkBudget(1))
    with pytest.raises(ValueError, match='description chain'):
        reader.resolve_description(reference, row=0, max_records=0, work=QueryWorkBudget(1))


def test_pending_sentence_is_a_tagged_chain_with_only_its_top_stamped():
    store, rows, said, clause, written, result = pending_chain()
    assert len(store) == 3 and len(store.parts) == 4
    assert written[id(clause)] == result
    assert written['root_occurrence'] == result
    node = result[1]
    for left in ((OCCURRENCE, int(store.address_keys[rows[0]])), (SYMBOL, said),
                 (OCCURRENCE, int(store.address_keys[rows[1]]))):
        first, second = store.parts.references(node)
        assert first == left and second[0] == RELATION
        assert store.parts.stratum(node) == 2
        node = second[1]
    assert store.parts.references(node) == ((SYMBOL, said), (OCCURRENCE, int(store.address_keys[rows[2]])))
    assert store.parts.rows_referencing(SYMBOL, said)
    assert not any(bool(store.entry_when(row).ne(0).any()) for row in rows)
    stamped = store.parts.percept[:, 4:].ne(0).any(-1).nonzero().flatten().tolist()
    assert stamped == [result[1]] and not hasattr(store.parts, 'when')
    before = copy.deepcopy(store.parts.state_dict())
    owner = store._concept_owner()
    token = store.occurrence_of(rows[2])
    owner.mtm.open(token, kind=None, value=store.slots[rows[2]], when=0, pending=True)
    owner.mtm.update(token, torch.eye(16)[8], when=1)
    for name, value in before.items():
        torch.testing.assert_close(store.parts.state_dict()[name], value, rtol=0, atol=0)


def test_reference_tags_disambiguate_equal_integer_values_and_protect_transitively():
    parts = LTM1(capacity=16)
    symbol_row = parts.add((SYMBOL, 4), (SYMBOL, 5))
    occurrence_row = parts.add((OCCURRENCE, 4), (SYMBOL, 5))
    outer = parts.add((RELATION, occurrence_row), (SYMBOL, 6))
    assert parts.find((SYMBOL, 4), (SYMBOL, 5)) == symbol_row
    assert parts.find((OCCURRENCE, 4), (SYMBOL, 5)) == occurrence_row
    assert parts.rows_referencing(OCCURRENCE, 4) == (occurrence_row,)
    assert parts.parts((SYMBOL, 5)) == ((SYMBOL, 4), (OCCURRENCE, 4))
    assert (OCCURRENCE, 4) in parts.protected_references((outer,))
    with pytest.raises(ValueError, match='dependency'):
        parts.retain([outer])
    assert parts.stratum(symbol_row) == 1 and parts.stratum(outer) == 2


def test_relation_readings_use_tagged_situations_and_symbol_types():
    parts = LTM1(capacity=16)
    temporal = parts.add((OCCURRENCE, 2), (OCCURRENCE, 3))
    spatial = parts.add((OCCURRENCE, 4), (OCCURRENCE, 5))
    kind = parts.add((SYMBOL, 4), (SYMBOL, 5))
    types = {(OCCURRENCE, i): ('situation',) for i in (2, 3)}
    types.update({(OCCURRENCE, i): ('spatial',) for i in (4, 5)})
    types.update({(SYMBOL, i): ('kind',) for i in (4, 5)})
    assert parts.readings(temporal, types.get) == ('conditional',)
    assert parts.readings(spatial, types.get) == ('part',)
    assert parts.readings(kind, types.get) == ('subsumption',)


def test_relation_query_cannot_use_its_own_closing_as_an_independent_witness():
    from Queries import ThoughtLTMCapability
    store, (left, right) = native_store(points=torch.eye(4)[:2])
    alloc = store._concept_owner()._concept_allocator
    alloc.singletons.add(right)
    relation = store.parts.add(left, right, truth_plus=1.)
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: 0., tau_id=.9)
    cue = replace(ConceptualMeaning.from_description(torch.eye(4)[0]),
        role_refs=(('sym', left), None, None),
        bindings={'_query_relation': 'equal', '_query_occurrence': ('sym', relation)})
    assert reader.best_match(cue, max_records=16, work=QueryWorkBudget(16))['frames'] == ()
    independent = replace(cue, bindings={'_query_relation': 'equal'})
    assert reader.best_match(independent, max_records=16, work=QueryWorkBudget(16))['witnesses'] == (('sym', relation),)


def test_time_only_recall_returns_all_nearest_sentences_under_the_work_meter():
    store, rows, _, _, _, chain = pending_chain()
    when = store.entry_when(chain).clone()
    idea = store.append_meaning(ConceptualMeaning.from_description(torch.eye(16)[9]),
        kind='observation', when=when, sentence_index=9)
    store.append_meaning(ConceptualMeaning.from_description(torch.eye(16)[10]),
        kind='observation', when=encoder().encode(50), sentence_index=10)
    work = QueryWorkBudget(64)
    found = store.recall_time(when, window=1, max_records=32, work=work)
    assert {row['occurrence'] for row in found['value']} == {chain, store.occurrence_of(idea)}
    assert not found['incomplete'] and work.spent == len(store)+len(store.parts)
    assert store.recall_identities(found['value'], work=work) == ()
    assert all(store._concept_owner().mtm.token_of(store.occurrence_of(row)) is not None for row in rows)
    limited = store.recall_time(when, window=1, max_records=32, work=QueryWorkBudget(1))
    assert limited['records_scanned'] == 1 and limited['incomplete'] == ('time_read_limit',)


def test_layout_checkpoint_is_versioned_and_never_resizes():
    store = LTM0(16, capacity=4)
    store.append_idea(torch.zeros(16))
    restored = LTM0(16, capacity=4)
    pointers = tuple(value.data_ptr() for value in restored.buffers())
    restored.load_state_dict(store.state_dict())
    restored.load_semantic_extras(store.semantic_extras())
    assert tuple(value.data_ptr() for value in restored.buffers()) == pointers
    state = copy.deepcopy(store.state_dict()); state['truth_schema'].fill_(10)
    with pytest.raises(ValueError, match='version'):
        restored.load_state_dict(state)
    with pytest.raises(ValueError, match='capacity'):
        LTM0(16, capacity=5).load_state_dict(store.state_dict())
    parts = LTM1(capacity=8); parts.add((OCCURRENCE, -20), (SYMBOL, 2))
    other = LTM1(capacity=8); other.load_state_dict(parts.state_dict())
    state = copy.deepcopy(parts.state_dict()); state['ref_kinds'][1, 0] = 0
    before = copy.deepcopy(other.state_dict())
    with pytest.raises(ValueError, match='tags'):
        other.load_state_dict(state)
    for name, value in before.items():
        torch.testing.assert_close(other.state_dict()[name], value)
    state = copy.deepcopy(parts.state_dict()); del state['ref_kinds']
    with pytest.raises(ValueError, match='version'):
        other.load_state_dict(state)


def test_reading_names_the_kept_journal_and_retired_readers_are_absent():
    import Understanding, MemoryIndex, ReferenceParticipants
    from MereologicalCodes import MereologicalCodes
    assert hasattr(Understanding, 'Reading')
    assert not hasattr(MemoryIndex, 'LeafCodeIndex')
    assert not hasattr(LTM0, 'leaf_terms')
    assert not hasattr(ReferenceParticipants, 'nonparticipants')
    assert not hasattr(MereologicalCodes, 'occurrence_terms')


def test_deep_checkpoint_chains_validate_without_a_python_recursion_limit():
    store = LTM1(capacity=2048)
    state = store.state_dict()
    # A descending chain makes the first visited row traverse every seat.
    state['ref_kinds'][1:, 0] = RELATION
    state['ref_kinds'][1:, 1] = SYMBOL
    state['refs'][1:, 0] = torch.arange(2, 2049)
    state['refs'][1:, 1] = 2
    state['ref_kinds'][-1, 0] = SYMBOL
    state['count'].fill_(2047)
    store.load_state_dict(state)
    assert len(store.protected_references((1,))) == 2048
    state['ref_kinds'][-1, 0] = RELATION
    state['refs'][-1, 0] = 1
    with pytest.raises(ValueError, match='cyclic'):
        store.load_state_dict(state)


@pytest.mark.parametrize('column,value', [('surprise', float('nan')), ('surprise', 2.),
                                         ('access_count', -1)])
def test_invalid_relation_checkpoint_rejected_before_mutation(column, value):
    store = LTM1(capacity=8)
    store.add(2, 3)
    state = copy.deepcopy(store.state_dict())
    state[column][1] = value
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match='invalid restored'):
        store.load_state_dict(state)
    for name, original in before.items():
        torch.testing.assert_close(store.state_dict()[name], original, rtol=0, atol=0)


def test_close_placement_updates_meaning_and_content_retrieval_finds_idea(monkeypatch):
    from types import SimpleNamespace
    from MereologicalCodes import MereologicalCodes
    from MeaningCodes import identity_code
    from Occurrence import sentence_key
    from test_review13_subspaces import fixture, word
    cs, cb, _, _ = fixture()
    a, b = word(cs, [(4, 1.)]), word(cs, [(7, 1.)])
    monkeypatch.setattr(cs, 'word_surface_for_row', lambda row: {a: b'p', b: b'q'}.get(row))
    derived = cb.mereology
    reading = SimpleNamespace(word_rows=torch.tensor([a, b, a]), concept_ids=None)
    before = derived.placement.clone()
    cs._sentence_seen_rows[0, :2] = torch.tensor([a,b])
    cs.prime_seen(torch.tensor([[a,b]]))
    derived.observe_close(reading)
    target = identity_code(b'p', derived.meaning_pairs,
                           derived.meaning_ones, like=before)
    target = torch.nn.functional.normalize(target, dim=0)
    assert (derived.placement[a] - target).norm() < (before[a] - target).norm()
    assert int(derived.placement_witnesses[a]) == int(derived.placement_witnesses[b]) == 1
    derived.begin_forward()
    value = cb.lookup_rows(a)
    assert value[derived.percept_event_width:].ne(0).any()
    store = LTM0(5, capacity=4, percept_width=0)
    row = store.append_idea(value, trust=1.)
    cue = ConceptualMeaning.from_description(value)
    result = store.cued_rows(cue, max_candidates=4, work=QueryWorkBudget(8))
    assert result['value'][0]['occurrence'] == store.occurrence_of(row)
    assert result['value'][0]['match'] == pytest.approx(1.)


def test_time_query_loads_the_requested_stream_and_identity_pool():
    from Layers import SymbolExpectation
    from Queries import ThoughtLTMCapability
    store, rows, _, _, _, chain = pending_chain()
    owner = store._concept_owner()
    for key in tuple(owner.mtm.entries):owner.mtm.forget(key)
    from Layers import ShortTermMemory
    owner.stm = ShortTermMemory(batch=2,capacity=8,concept_dim=16)
    discourse = SymbolExpectation(n_symbols=8, max_depth=8, n_dim=16, concept_dim=16)
    discourse._ltm_store = store
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: None,
        tau_id=.99, discourse=discourse, row=1)
    blank = torch.zeros(16); blank[store.when_start:store.when_start+4] = store.entry_when(chain)
    result = reader.best_match(ConceptualMeaning.from_description(blank),
                              max_records=32, work=QueryWorkBudget(64))
    assert result['witnesses'] == (chain,)
    assert [frame.row_id for frame in discourse.situation_references(1)] == [chain[1]]
    assert not discourse.situation_references(0)
    assert all(owner.mtm.token_of(store.occurrence_of(row)) is not None for row in rows)


def test_kept_thought_writes_identity_value_and_inference_without_rewriting_chain():
    from types import SimpleNamespace
    from ThoughtStream import write
    from ThoughtReferences import with_slots
    store, rows, _, _, _, chain = pending_chain()
    before = copy.deepcopy(store.parts.state_dict())
    identity = store.occurrence_of(rows[0])
    owner = store._concept_owner()
    owner.mtm.open(identity, kind=None, value=store.slots[rows[0]], when=0, pending=True)
    value = torch.eye(16)[8]
    meaning = ConceptualMeaning(torch.stack((store.slots[rows[0]], torch.zeros(16), value)),
        torch.ones(3, dtype=torch.bool), role_refs=(identity, ('sym', 42), store.occurrence_of(rows[1])),
        bindings={'_equality': True, '_producing_operation': 'query', '_bound_roles': (('referent', 2),)},
        sentence_kind='relation')
    meaning = with_slots(meaning, (), pair=(1., 0.))
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store))
    row = write(model, meaning, row=0)
    assert store.row(row)['kind'] == 'inference'
    torch.testing.assert_close(owner.mtm.value_of(identity), value)
    assert dict(store.raw_meaning_of(row).bindings)['_identity'] == identity
    from BindingAnswers import references
    assert references(store.meaning_of(row))[0][0] == meaning.role_refs[2]
    torch.testing.assert_close(references(store.meaning_of(row))[0][1], store.slots[row])
    for name, expected in before.items():
        torch.testing.assert_close(store.parts.state_dict()[name], expected, rtol=0, atol=0)
