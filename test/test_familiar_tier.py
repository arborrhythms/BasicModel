"""Shared class coverage, deferred occurrence members and retirement certificates."""
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
import torch
from torch.nn import functional as F
from test_review13_subspaces import fixture, word
from ConceptIndex import index_part_row
from LTM1 import SYMBOL, OCCURRENCE


def placed_fixture():
    from ClauseRow import attach_clause_index
    from Layers import LTM0
    cs, cb, ps, alloc = fixture(meaning_pairs=2)
    store = LTM0(cb.W.shape[-1], capacity=32)
    attach_clause_index(SimpleNamespace(), store, cs)
    a, b, base = word(cs, [(4,1.)]), word(cs, [(7,1.)]), word(cs, [(4,1.)])
    angles = torch.tensor([0.,10.])*math.pi/180
    cb.mereology.placement[[a,b]] = torch.stack((angles.cos(),angles.sin()),-1)
    cb.mereology.placement_witnesses[[a,b]] = 1
    return cs, cb, alloc, store, (a,b,base)


def test_symbol_and_promoted_occurrence_members_have_the_same_three_initializers():
    from Meaning import ConceptualMeaning
    cs, cb, alloc, store, (a,b,base) = placed_fixture()
    ids = [cs.concept_id_at_row(row) for row in (a,b,base)]
    first = index_part_row(cs, ids[0], ids[2])
    index_part_row(cs, ids[1], first)
    occurrence = store.append_meaning(ConceptualMeaning.from_description(cb.lookup_rows(b)),
        order=0, document_key='coverage', sentence_index=0)
    key = store.occurrence_of(occurrence)
    cs.mtm.open(key, kind=None, when=0, value=cb.lookup_rows(b))
    cs.mtm.promote(key, ids[1])
    other_base = word(cs, [(4,1.)])
    second = index_part_row(cs, ids[0], cs.concept_id_at_row(other_base))
    index_part_row(cs, (OCCURRENCE,key[-1]), second)
    left,right = cs._csw_row_of(first),cs._csw_row_of(second)
    assert dict(cs.concept_weights(left)) == dict(cs.concept_weights(right)) == {a:1.,b:1.}
    torch.testing.assert_close(cb.lookup_rows(left),cb.lookup_rows(right))
    torch.testing.assert_close(cb.W[left,:3],(cb.lookup_rows(a)[:3]+cb.lookup_rows(b)[:3])/2)
    torch.testing.assert_close(cb.mereology.placement[left],F.normalize(cb.mereology.placement[[a,b]].sum(0),dim=0))
    assert cb.mereology.placement_valid[[left,right]].all()


def test_transient_occurrence_is_deferred_and_covered_when_its_symbol_arrives():
    from Meaning import ConceptualMeaning
    cs, cb, alloc, store, (a,b,base) = placed_fixture()
    occurrence = store.append_meaning(ConceptualMeaning.from_description(cb.lookup_rows(b)),
        order=0, document_key='deferred', sentence_index=0)
    key = store.occurrence_of(occurrence)
    cs.mtm.open(key, kind=None, when=0, value=cb.lookup_rows(b))
    parent = index_part_row(cs, (OCCURRENCE,key[-1]), cs.concept_id_at_row(base))
    row = cs._csw_row_of(parent)
    assert not cs.concept_weights(row) and cb.mereology.placement_witnesses[row] == 0
    pointers = cb.W.data_ptr(),cb.mereology.placement.data_ptr()
    cs.mtm.promote(key, cs.concept_id_at_row(b))
    assert dict(cs.concept_weights(row)) == {b:1.}
    torch.testing.assert_close(cb.lookup_rows(row)[:3],cb.lookup_rows(b)[:3])
    torch.testing.assert_close(cb.mereology.placement[row],cb.mereology.placement[b])
    assert cb.mereology.placement_witnesses[row] == 1
    assert pointers == (cb.W.data_ptr(),cb.mereology.placement.data_ptr())


def test_defining_equality_promotes_a_pending_subject_without_an_order_or_choice():
    from test_orders_learned import equality_fixture, equal_close
    from Meaning import ConceptualMeaning
    cs, store, alloc = equality_fixture()
    subject = cs.interpret.lookup_word([1], [], form='new-name')
    label = cs.interpret.lookup_word([2], [], form='new-kind')
    value = cs.similarity_codebook.lookup_rows(cs._csw_row_of(subject))
    # First sentence's noun site retains an unnamed identity, with the native
    # occurrence's ordinary order; neither the token nor the word gets one.
    first = store.append_meaning(ConceptualMeaning.from_description(value),
        document_key='definition', sentence_index=0)
    key = store.occurrence_of(first)
    cs.mtm.open(key, kind=None, when=0, value=value, reading=subject)
    assert cs.definitions.objects(subject) == () and cs.mtm.entries[key].symbol is None
    equal_close(cs, store, key[-1], label)
    symbol = cs.mtm.entries[key].symbol
    assert symbol is not None and cs.definitions.objects(subject) == (symbol,)
    parents = [parent for parent in alloc.raised if (SYMBOL,symbol) in alloc.parts.parts((SYMBOL,parent))]
    assert len(parents) == 1
    row = cs._csw_row_of(parents[0])
    assert dict(cs.concept_weights(row))[cs._csw_row_of(symbol)] > 0
    assert cs.definitions.objects(label) == tuple(parents)


ROW_RETIREMENTS = ('provisional','managed','participation','context_seen','witness_strength',
                  '_provisional_orders','context_similarities')
OWNER_RETIREMENTS = ('_ensure_concept_pool','_assign_concept_parts','observe_concept_use',
                    '_write_concept_context','_concept_learning_active','promotion_observe','promotion_pass')
KNOB_RETIREMENTS = ('conceptPoolSize','conceptUseEWMA','conceptMintThreshold','conceptRecycleThreshold')


@pytest.mark.parametrize('name',ROW_RETIREMENTS)
def test_inventory_pool_retirement(name):
    from test_attention_promotion import _fixture
    cs,_ = _fixture()
    layer = cs._concept_allocator.layer()
    assert not hasattr(layer,name)
    saved = layer.parts_extras()
    assert name not in saved and name not in saved['buffers']
    assert 'where' not in saved and 'provisional_orders' not in saved


@pytest.mark.parametrize('name',OWNER_RETIREMENTS)
def test_owner_pool_retirement(name):
    from Spaces import ConceptualSpace
    assert not hasattr(ConceptualSpace,name)


@pytest.mark.parametrize('name',KNOB_RETIREMENTS)
def test_pool_configuration_retirement(name):
    root = Path(__file__).resolve().parents[1]
    assert name not in (root/'data/model.xml').read_text()
    assert name not in (root/'data/model.xsd').read_text()


def test_serial_close_observer_retired():
    root = Path(__file__).resolve().parents[1]
    assert not (root/'bin/SerialConcepts.py').exists()
    assert 'SerialConcepts' not in (root/'bin/Models.py').read_text()


def test_allocation_certificate_has_no_inventory_square_or_dense_candidate_table(tmp_path):
    import json, os
    from torch.utils._python_dispatch import TorchDispatchMode
    from test_attention_promotion import _fixture, _observe
    cs, rows = _fixture()
    outputs = []
    class AllocationRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            def read(value):
                if torch.is_tensor(value):
                    outputs.append(dict(operation=str(func), shape=list(value.shape),
                        dtype=str(value.dtype), layout=str(value.layout)))
                    assert not (value.layout == torch.strided and value.shape == (cs.nVectors,cs.nVectors))
                    assert not (value.layout == torch.strided and value.shape == (cs.mtm.capacity,cs.nVectors))
                elif isinstance(value,(list,tuple)):
                    for member in value: read(member)
            read(result)
            return result
    with AllocationRead():
        for _ in range(3):
            _observe(cs,{rows[0][1]:1.,rows[1][1]:1.,rows[2][1]:1.})
        cs.mtm.promote_candidates(cs)
        saved = cs.mtm.state()
        cs.mtm.restore(saved)
    assert all(isinstance(entry.support,dict) and len(entry.support)<=cs.mtm.field_width
               for entry in cs.mtm.entries.values())
    folder = Path(os.environ.get('ITEM27_RECEIPT',tmp_path))
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'allocation-read.json').write_text(json.dumps(dict(inventory=cs.nVectors,
        capacity=cs.mtm.capacity, support_bound=cs.mtm.field_width,
        candidate_support='bounded sparse dictionaries', inventory_square=False,
        dense_candidate_inventory_table=False, allocations=outputs),indent=2)+'\n')


def test_opposite_observations_in_the_horizon_also_refuse_promotion():
    from test_attention_promotion import _fixture, _close
    from FamiliarTier import observe_candidate
    cs,rows = _fixture()
    for pair in ((1.,0.),(0.,1.),(1.,0.)):
        key=observe_candidate(cs,1,{rows[0][1]:1., rows[1][1]:1.},pair=pair)
        _close(cs)
    assert cs.mtm.entries[key].count == 3
    assert not cs.mtm.promote_candidates(cs)


def test_same_field_readings_keep_both_lanes_without_counting_twice():
    from test_attention_promotion import _fixture, _close
    from FamiliarTier import observe_candidate
    cs, rows = _fixture()
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    first = observe_candidate(cs, 1, support, pair=(1.,0.))
    assert observe_candidate(cs, 1, support, pair=(0.,1.)) == first
    assert cs.mtm.entries[first].count == 1
    assert cs.mtm.entries[first].readings == ((0,1,1.,1.),)
    assert cs.mtm.entries[first].refinement['both']
    _close(cs)
    for _ in range(2):
        observe_candidate(cs, 1, support)
        _close(cs)
    assert not cs.mtm.promote_candidates(cs)


def test_candidate_count_tracks_observations_when_the_open_read_crosses_its_horizon():
    from MediumTermMemory import MediumTermMemory
    pool = MediumTermMemory(2, horizon=2, n_vectors=90)
    value, support = torch.ones(4), {1:1.}
    key = pool.open_candidate(order=1, when=0, value=value, support=support)
    pool.record_reading(key, 0, (1.,0.))
    for _ in range(3):
        pool.close(None, stream=0, references=(key,))
    assert pool.entries[key].count == 1
    pool.update_candidate(key, when=4, value=value, support=support)
    pool.record_reading(key, 0, (1.,0.))
    assert pool.entries[key].count == 1
    assert pool.entries[key].readings == ((0,4,1.,0.),)


def test_candidate_promotion_uses_weighted_centroid_placement_and_form():
    from FamiliarTier import observe_candidate
    from test_attention_promotion import _close
    cs, cb, alloc, store, (a,b,_) = placed_fixture()
    cs._promotion_enabled = True
    for _ in range(cs.concept_mint_count):
        key = observe_candidate(cs, 1, {a:2.,b:1.})
        _close(cs)
    pointers = cb.W.data_ptr(), cb.mereology.placement.data_ptr()
    expected_form = (cb.lookup_rows(a)[:3]*2+cb.lookup_rows(b)[:3])/3
    expected_place = F.normalize(cb.mereology.placement[a]*2+cb.mereology.placement[b],dim=0)
    symbol, = cs.mtm.promote_candidates(cs)
    row = cs._csw_row_of(symbol)
    assert cs.mtm.entries[key].symbol == symbol
    torch.testing.assert_close(cb.lookup_rows(row)[:3], expected_form)
    torch.testing.assert_close(cb.mereology.placement[row], expected_place)
    assert cb.mereology.placement_witnesses[row] == 2 and cb.mereology.placement_valid[row]
    assert pointers == (cb.W.data_ptr(),cb.mereology.placement.data_ptr())


def test_kept_close_restricts_promotion_to_active_streams_and_defers_coverage():
    from FamiliarTier import kept_close, observe_candidate
    from test_attention_promotion import _fixture, _close
    cs, rows = _fixture()
    for stream in (0,1):
        for _ in range(3):
            observe_candidate(cs, 1, {rows[2*stream][1]:1., rows[2*stream+1][1]:1.}, batch=stream)
            _close(cs,stream)
    model = SimpleNamespace(_concept_owner=lambda:cs)
    @kept_close
    def update(model, state, sid, active):
        assert cs._coverage_deferred
        return cs.mtm.promote_candidates(cs)
    assert update(model, None, 0, torch.tensor([False,False])) == []
    assert len(update(model, None, 0, torch.tensor([True,False]))) == 1
    pending = [entry for entry in cs.mtm.entries.values() if entry.symbol is None]
    assert len(pending) == 1 and set(pending[0].support) == {rows[2][1], rows[3][1]}
    assert not cs._coverage_deferred and cs._mtm_active_streams is None


def test_prepend_carries_candidates_with_live_gradient_and_rejects_recycled_transients():
    import STMChunks as chunks
    from FamiliarTier import centroid, observe_candidate
    from PrependedSymbols import SymbolEntries, stage, chunk_token, reconcile
    from test_attention_promotion import _fixture
    cs, rows = _fixture()
    key = observe_candidate(cs, 1, {rows[0][1]:2.,rows[1][1]:1.})
    token = cs.mtm.entries[key]
    store = cs._fixture_occurrences
    entries = SymbolEntries(1,1,cs.mtm.capacity,8)
    model = SimpleNamespace(_concept_owner=lambda:cs, inputSpace=SimpleNamespace(symbol_entries=entries),
        symbolSpace=SimpleNamespace(ltm_store=store), _open_sentence_slot=0)
    state = chunks.empty(1,8,8,device='cpu',dtype=torch.float32)
    pointers = {name:buffer.data_ptr() for name,buffer in entries.named_buffers()}
    state = stage(model,state,torch.tensor([True]))
    column = int((entries.concept_ids[0] == token.transient).nonzero()[0])
    seat = int((state[chunks.IDENTITY][0] == token.transient).nonzero()[0])
    assert entries.concept_orders[0,column] == 1
    assert chunk_token(cs.mtm,store,state,0,seat) == key
    torch.testing.assert_close(entries.lookup_atoms[0,column],centroid(cs,token.support))
    entries.lookup_atoms[0,column].sum().backward()
    assert cs.similarity_codebook.W.grad[rows[0][1]].abs().sum() > 0
    assert cs.similarity_codebook.W.grad[rows[1][1]].abs().sum() > 0
    cs.mtm.forget(key)
    next_key = cs.mtm.open_candidate(order=1,when=2,value=token.value,support=token.support)
    assert cs.mtm.entries[next_key].transient == token.transient
    restored = reconcile(state,cs.mtm,store)
    assert restored[chunks.IDENTITY][0,seat] == -1
    assert pointers == {name:buffer.data_ptr() for name,buffer in entries.named_buffers()}


def test_promoting_an_identity_does_not_reinitialize_unrelated_classes():
    from Meaning import ConceptualMeaning
    cs, cb, alloc, store, (a,b,base) = placed_fixture()
    parent = index_part_row(cs,cs.concept_id_at_row(a),cs.concept_id_at_row(base))
    row = cs._csw_row_of(parent)
    retained = cb.W[row].clone()
    with torch.no_grad():
        cb.W[row,:3].add_(.1)
    retained = cb.W[row].clone()
    occurrence = store.append_meaning(ConceptualMeaning.from_description(cb.lookup_rows(b)),
        document_key='unrelated',sentence_index=0)
    key = store.occurrence_of(occurrence)
    cs.mtm.open(key,kind=None,when=0,value=cb.lookup_rows(b))
    cs.mtm.promote(key,cs.concept_id_at_row(b))
    torch.testing.assert_close(cb.W[row],retained)


@pytest.mark.parametrize('promoted', (False, True))
def test_candidate_zero_survives_the_kept_bind_journal_without_an_occurrence_alias(monkeypatch, promoted):
    import STMChunks as chunks
    from test_clause_acceptance import SentenceFixture
    from chunk_fixtures import prepend, record
    from ClauseJournal import finish_clause
    from PrependedSymbols import chunk_token
    f = SentenceFixture(monkeypatch)
    symbol = f.noun('subject')
    row = f.cs._csw_row_of(symbol)
    pool = f.cs.mtm
    key = pool.open_candidate(order=1, when=0,
        value=f.cs.similarity_codebook.lookup_rows(row), support={row:1.})
    assert key == ('candidate', 0)
    if promoted:
        pool.promote(key, symbol)
    state = prepend(f, (), 1)
    entry, _, _ = record(f, ('bind','subject'), leaf_orders=[0])
    selected = int(entry.identity_actions[-1,4])-1
    assert entry.identity_origins[selected].tolist() == [5, 0, int(promoted)]
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, state)
    assert plan['items'][0]['selected'] == key
    prepared = resolver.prepare(clause, plan)
    assert plan['items'][0]['written_node'].refs[0] == (symbol if promoted else -1)
    original = pool.entries[key]
    written = {}
    closed = f.store.write_clause(prepared, written_rows=written,
        document_key='candidate-bind', sentence_index=0)
    resolver.commit(plan, closed, written, document='candidate-bind', position=0, training=False)
    assert pool.entries[key].count == original.count
    torch.testing.assert_close(pool.entries[key].value, original.value)
    assert pool.accesses[key][None] == pool.closes[None]
    committed = resolver.committed_chunks(state, plan, written, stream=0)
    identity = symbol if promoted else original.transient
    seat = int((committed[chunks.IDENTITY][0] == identity).nonzero()[0])
    assert chunk_token(pool, f.store, committed, 0, seat) == key
    assert committed[chunks.REFS][0,seat,1].tolist() == [chunks.IDENTITY_REF,0]


@pytest.mark.parametrize('promoted', (False, True))
def test_answer_judges_the_carried_candidate_bind_without_an_individual_menu(monkeypatch, promoted):
    """Lesson-13: candidate zero was selected but absent from the individual menu."""
    from test_clause_acceptance import SentenceFixture
    from chunk_fixtures import prepend, record
    from ClauseJournal import finish_clause
    from ThoughtReferences import open_slots, with_slots
    from BindingAnswers import matches
    f = SentenceFixture(monkeypatch)
    symbol = f.noun('subject')
    row = f.cs._csw_row_of(symbol)
    pool = f.cs.mtm
    key = pool.open_candidate(order=1, when=0,
        value=f.cs.similarity_codebook.lookup_rows(row), support={row:1.})
    if promoted:
        pool.promote(key, symbol)
    state = prepend(f, (), 1)
    entry, _, _ = record(f, ('bind', 'subject'), leaf_orders=[0])
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, state)
    item = plan['items'][0]
    assert item['selected'] == key and key not in plan['individuals']
    assert not item['candidates'] and item['record']['value'] is not None
    meaning = with_slots(clause.meaning, (('referent', 0),))
    count, size = pool.entries[key].count, len(f.store)
    accesses = {token: dict(uses) for token, uses in pool.accesses.items()}
    result = resolver.answer_meaning(plan, meaning)
    model = SimpleNamespace(_concept_owner=lambda:f.cs,
        symbolSpace=SimpleNamespace(ltm_store=f.store))
    assert matches(model, result, 'subject') == promoted
    if promoted:
        assert result.role_refs[0] == ('sym', symbol) and not open_slots(result)
        torch.testing.assert_close(result.roles[0], item['record']['value'])
    else:
        assert result is meaning and open_slots(result) == (('referent', 0),)
    assert (pool.entries[key].count, pool.accesses, len(f.store)) == (count, accesses, size)
