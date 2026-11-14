"""Item 3.0 step 1: semantic compression, tagged chains and percept time."""
from Memory import STM0
from Occurrence import operand_name
import copy
from dataclasses import replace

import pytest
import torch

from ClauseRow import Clause, Composition
from Memory import LTM1, OCCURRENCE, SYMBOL, RELATION
from Memory import LTM0
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget
from RowTime import encoder
from part_fixtures import native_store, symbol


def tree(point, left, right, operation='lift'):
    return Composition(point, (left, right), operation=operation)


def field(term):
    return Clause(ConceptualMeaning.from_description(term.point), point=term.point, composition=term)


def test_resolved_sentence_compresses_to_one_idea_without_composing_edges():
    store, ids = native_store(width=16, points=torch.eye(16)[:3], percept_width=16)
    leaves = [Composition(torch.eye(16)[i], reference=ref) for i, ref in enumerate(ids)]
    vp = tree(torch.eye(16)[3], leaves[1], leaves[2], 'verb')
    clause = field(tree(torch.eye(16)[4], leaves[0], vp))
    row = store.write_clause(clause)
    assert store.slots.shape == (32, 16)
    assert len(store) == 1 and len(store.parts) == 0
    assert not hasattr(store, 'role_mask')
    torch.testing.assert_close(store.time[row], store.entry_time(row))
    assert not hasattr(store, '_semantic_rows')
    assert store.referent_of(row) == ('sym', ids[0])
    torch.testing.assert_close(store.slots[row, :12], clause.point[:12])
    assert bool(store.entry_time(row).ne(0).any())


def pending_chain():
    store, (said,) = native_store(width=16, points=torch.eye(16)[3:4], percept_width=16)
    rows = [store.append_meaning(ConceptualMeaning.from_description(torch.eye(16)[i]),
        ) for i in range(3)]
    owner = store._concept_owner()
    owner.mtm.open(store.occurrence_of(rows[2]), kind=None, value=store.slots[rows[2]], when=0, pending=True)
    he, she, unknown = [Composition(store.slots[row].clone(), reference=operand_name(store.row_name(row))) for row in rows]
    word = Composition(torch.eye(16)[3], reference=said)
    inner = tree(torch.eye(16)[4], word, unknown, 'part')
    middle = tree(torch.eye(16)[5], she, inner, 'part')
    outer = tree(torch.eye(16)[6], word, middle, 'part')
    root = field(tree(torch.eye(16)[7], he, outer, 'part'))
    written = {}
    result = store.write_clause(root, written_rows=written)
    return store, rows, said, root, written, result






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


def test_declared_nested_part_chain_preserves_typed_occurrences():
    store, rows, said, clause, written, result = pending_chain()
    assert len(store) == 3 and len(store.parts) == 4
    assert written[id(clause)] == result
    assert written['root_occurrence'] == result
    node = result[1]
    for left in ((OCCURRENCE, store.row_name(rows[0])), (SYMBOL, said),
                 (OCCURRENCE, store.row_name(rows[1]))):
        first, second = store.parts.references(node)
        assert first == left and second[0] == RELATION
        assert store.parts.stratum(node) == 2
        node = second[1]
    assert store.parts.references(node) == ((SYMBOL, said), (OCCURRENCE, store.row_name(rows[2])))
    assert store.parts.rows_referencing(SYMBOL, said)
    assert [int(encoder().decode_index(store.entry_time(row))) for row in rows] == [0, 1, 2]
    assert not hasattr(store.parts, 'percept') and not hasattr(store.parts, 'when')
    assert store.parts.contains(result[1])
    before = copy.deepcopy(store.parts.state_dict())
    owner = store._concept_owner()
    token = store.occurrence_of(rows[2])
    owner.mtm.open(token, kind=None, value=store.slots[rows[2]], when=0, pending=True)
    changed = store.append_idea(torch.eye(16)[8])
    owner.mtm.event(token, store.occurrence_of(changed), verb=said, role=0, when=1,
        before=store.slots[rows[2]])
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


def test_content_recall_reenters_retained_identities_under_the_work_meter():
    store, rows, _, _, _, chain = pending_chain()
    work = QueryWorkBudget(64)
    found = store.cued_rows(store.meaning_of(chain), references=(chain,), max_records=1, work=work)
    assert {row['occurrence'] for row in found['value']} == {chain}
    assert not found['incomplete'] and work.spent == 1
    assert store.recall_identities(found['value'], work=work) == ()
    assert all(store._concept_owner().mtm.token_of(store.occurrence_of(row)) is not None for row in rows)
    limited = store.cued_rows(store.meaning_of(chain), references=(chain,), max_records=1, work=QueryWorkBudget(0))
    assert not limited['value'] and limited['incomplete'] == ('work_budget',)


def test_layout_checkpoint_is_versioned_and_never_resizes():
    store = LTM0(16, capacity=4)
    store.append_idea(torch.zeros(16))
    restored = LTM0(16, capacity=4)
    pointers = tuple(value.data_ptr() for value in restored.buffers())
    restored.load_state_dict(store.state_dict())
    assert tuple(value.data_ptr() for value in restored.buffers()) == pointers
    state = copy.deepcopy(store.state_dict()); state['truth_schema'].fill_(10)
    with pytest.raises(ValueError, match='layout'):
        restored.load_state_dict(state)
    with pytest.raises(ValueError, match='capacity'):
        LTM0(16, capacity=5).load_state_dict(store.state_dict())
    parts = LTM1(capacity=8); parts.add((OCCURRENCE, 20), (SYMBOL, 2))
    other = LTM1(capacity=8); other.load_state_dict(parts.state_dict())
    state = copy.deepcopy(parts.state_dict()); state['ref_kinds'][1, 0] = 0
    before = copy.deepcopy(other.state_dict())
    with pytest.raises(ValueError, match='tags'):
        other.load_state_dict(state)
    for name, value in before.items():
        torch.testing.assert_close(other.state_dict()[name], value)
    state = copy.deepcopy(parts.state_dict()); del state['ref_kinds']
    with pytest.raises(ValueError, match='column'):
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


@pytest.mark.parametrize('column,value', [('activity', float('nan')), ('activity', 2.),
                                         ('activity', -1)])
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


def test_close_placement_updates_meaning_and_addressed_read_finds_idea(monkeypatch):
    from types import SimpleNamespace
    from MereologicalCodes import MereologicalCodes
    from MeaningCodes import identity_code
    from Occurrence import sentence_key
    from test_review13_subspaces import fixture, word
    cs, cb, _, _ = fixture()
    a, b = word(cs, [(4, 1.)]), word(cs, [(7, 1.)])
    from definition_fixtures import agreeing_views
    agreeing_views(cs, cs.concept_id_at_row(a), cs.concept_id_at_row(b))
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
    radius = 1 - __import__("math").exp(-1)
    direction = torch.nn.functional.normalize(derived.placement[a], dim=0)
    torch.testing.assert_close(value[3:4], direction * radius)
    assert not value[4:5].any()
    store = STM0(5, capacity=4, percept_width=0, relation_capacity=8192)
    row = store.append_idea(value, trust=1.)
    cue = ConceptualMeaning.from_description(value)
    assert not store.cued_rows(cue, max_candidates=4, work=QueryWorkBudget(8))['value']
    result = store.cued_rows(cue, references=(store.occurrence_of(row),),
                             max_candidates=4, work=QueryWorkBudget(8))
    assert result['value'][0]['occurrence'] == store.occurrence_of(row)
    assert result['value'][0]['match'] == 0.  # An addressed read alone supplies no field relevance.


def test_content_recall_loads_only_the_requested_stream_and_identity_pool():
    from Queries import ThoughtLTMCapability
    from Memory import ShortTermMemory, STMChunks
    store, rows, _, _, _, chain = pending_chain()
    owner = store._concept_owner()
    for key in tuple(owner.mtm.entries): owner.mtm.forget(key)
    owner.stm = ShortTermMemory(batch=2, capacity=8, concept_dim=16)
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: None, tau_id=.99, row=1)
    result = store.cued_rows(store.meaning_of(chain), references=(chain,),
                              max_records=1, work=QueryWorkBudget(64))
    assert tuple(row['occurrence'] for row in result['value']) == (chain,)
    assert not store.recall_chunks(result['value'], work=QueryWorkBudget(64), stream=1)
    assert owner.stm.state[STMChunks.REFS][1, 0, 0].tolist() == [STMChunks.RELATION, chain[1]]
    assert not owner.stm.state[STMChunks.READ][0].any()
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
    assert store.row(row)['trust'] == 1.
    # §50: a kept thought is one more reference until this identity has an event.
    torch.testing.assert_close(owner.mtm.value_of(identity),
        torch.maximum(store.point_of_row(identity), store.point_of_row(store.occurrence_of(row))))
    assert store.referent_of(row) == identity
    from BindingAnswers import references
    assert references(meaning)[0][0] == meaning.role_refs[2]
    torch.testing.assert_close(references(meaning)[0][1], value)
    for name, expected in before.items():
        torch.testing.assert_close(store.parts.state_dict()[name], expected, rtol=0, atol=0)
