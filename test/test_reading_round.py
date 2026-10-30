"""Item 2.7 §29: durable reuse, lexical exclusion, epoch and reading contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from FamiliarTier import observe_candidate, witnessed_match
from test_attention_promotion import _fixture, _close


@pytest.mark.parametrize('translation', [-1e-23, 0., 1e-23])
def test_surface_generation_neutral_translation_has_finite_backward(translation):
    from Language import SurfaceLayer
    layer = SurfaceLayer(nInput=8, nOutput=8)
    with torch.no_grad():
        layer.marker_map.bias.fill_(translation)
    parent = torch.full((2, 8), .5, requires_grad=True)
    marker, content = layer.generate(parent)
    torch.testing.assert_close(layer.compose(marker, content), parent)
    (marker.square().mean() + content.square().mean()).backward()
    assert torch.isfinite(parent.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in layer.parameters())


@pytest.mark.parametrize('literal', [None, b'located'])
def test_existing_candidate_is_used_across_two_epochs_without_remint(literal):
    cs, rows = _fixture()
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    arguments = {} if literal is None else dict(literal=('ps',(4,7)), literal_key=literal)
    for _ in range(cs.concept_mint_count):
        key = observe_candidate(cs, 0, support, **arguments)
        _close(cs)
    cid, = cs.mtm.promote_candidates(cs)
    before = cs._concept_allocator.next_id
    for _ in range(2):
        cs.mtm.restore(None)
        for _ in range(3):
            assert observe_candidate(cs, 0, support, **arguments) is None
            _close(cs)
            assert cs.mtm.promote_candidates(cs) == []
        assert cs._concept_allocator.next_id == before and not cs.mtm.entries
    assert dict(cs._concept_allocator.store_of(cid).constituents(cid))['familiar_access'] == 6


def test_promotion_uses_a_row_that_became_witnessed_after_open():
    from FamiliarTier import centroid, cover_members, remember_candidate
    from order_fixtures import force_order
    cs, rows = _fixture()
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    key = observe_candidate(cs, 1, support)
    _close(cs)
    cid = cs.new_concept()
    force_order(cs, cid, 1)
    cover_members(cs, cid, support)
    remember_candidate(cs, cid, support)
    before = cs._concept_allocator.next_id
    assert cs.mtm.promote_candidates(cs) == [cid]
    assert cs.mtm.entries[key].symbol == cid and cs._concept_allocator.next_id == before


def test_support_reuse_requires_order_and_cosine_and_singleton_opens_nothing():
    from FamiliarTier import cover_members, remember_candidate
    from order_fixtures import force_order
    cs, rows = _fixture()
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    cid = cs.new_concept()
    force_order(cs, cid, 1)
    row = cover_members(cs, cid, support)
    remember_candidate(cs, cid, support)
    assert witnessed_match(cs, 1, support) == cid
    assert witnessed_match(cs, 0, support) is None
    with torch.no_grad():
        cs.similarity_codebook.W[row].copy_(torch.eye(8)[7])
    assert witnessed_match(cs, 1, support) is None
    before = cs.mtm.candidate_counter
    assert observe_candidate(cs, 1, {rows[0][1]: 1.}) is None
    assert cs.mtm.candidate_counter == before and not cs.mtm.entries


def test_literal_reuse_adds_coverage_by_maximum():
    cs, rows = _fixture()
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    for _ in range(3):
        observe_candidate(cs, 0, support, literal=('ps',(4,7)), literal_key=b'case')
        _close(cs)
    cid, = cs.mtm.promote_candidates(cs)
    row = cs._csw_row_of(cid)
    before = dict(cs.concept_weights(row))
    cs.mtm.restore(None)
    observe_candidate(cs, 0, {**support, rows[2][1]: 2.}, literal=('ps',(4,7)), literal_key=b'case')
    after = dict(cs.concept_weights(row))
    assert all(after[member] == weight for member,weight in before.items())
    assert after[rows[2][1]] == 2.


def test_lexical_letters_open_nothing_but_nonlexical_located_case_does():
    from FamiliarTier import observe_native_cases
    cs, rows = _fixture(pi=True)
    raw = {4:b'a', 7:b'b'}
    native = SimpleNamespace(bytes_for=raw.get)
    cs._model = SimpleNamespace(perceptualSpace=SimpleNamespace(percept_store=native, fuse_parts=lambda ids: ids,
        _forward_input=dict(native_indices=torch.tensor([[4,7]]), native_part_spans=torch.tensor([[[0,1],[1,2]]]))))
    cs._percept_field = SimpleNamespace(where=torch.tensor([[[0,2]]]))
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    observe_native_cases(cs, 0, support)
    assert len(cs.mtm.entries) == 1
    cs.mtm.restore(None)
    model = cs._model
    cs._model = None
    word = cs.interpret.lookup_word((4,7), (), form=b'ab')
    cs._model = model
    observe_native_cases(cs, 0, support)
    assert not cs.mtm.entries
    assert cs.interpret.admit_discovered() is None and not cs.definitions.objects(word)


def test_documents_retain_memory_and_mtm_until_epoch():
    from test_presentation_memory import model_for, write
    from test_document_covers import fixture
    from PresentationMemory import begin
    from Forgetting import model_boundary
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    first = write(store, atoms[0], 'first')
    owner.mtm.open(first, kind=None, when=0, value=atoms[0])
    begin(model)
    second = write(store, atoms[1], 'second')
    owner.mtm.open(second, kind=None, when=0, value=atoms[1])
    begin(model)
    assert all(store.location(ref) is not None and owner.mtm.token_of(ref) is not None for ref in (first,second))
    model_boundary(model, epoch_boundary=True)
    assert len(store) == 0 and not owner.mtm.entries


def test_reading_pending_name_survives_documents_then_defines_on_recurrence():
    from test_orders_learned import equality_fixture
    from test_presentation_memory import model_for
    from PresentationMemory import begin
    from Meaning import ConceptualMeaning
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word((4,7), (), form='label')
    ids = []
    for document in range(cs.concept_mint_count):
        begin(model_for(cs,store))
        identity, row, order = cs.interpret.reading(word)
        key = cs.mtm.token_of(identity)
        assert cs.mtm.entries[key].pending and not cs.definitions.objects(word)
        assert order == alloc.order_of(word) == 0
        ids.append(identity)
        value = cs.similarity_codebook.lookup_rows(row)
        index = store.append_meaning(ConceptualMeaning.from_description(value), order=order,
            document_key=('document',document), sentence_index=0)
        occurrence = store.occurrence_of(index)
        cs.mtm.bind(key, occurrence, kind=None, when=document, value=value, reading=word)
        cs.mtm.close(store, stream=0, references=(occurrence,))
        cs.mtm.promote_candidates(cs)
    assert len(set(ids)) == 1
    obj, = cs.definitions.objects(word)
    assert alloc.order_of(obj) == 1
    assert cs.interpret.reading(word) == (obj, cs._csw_row_of(obj), 1)
    from BindingAnswers import matches
    from dataclasses import replace
    answer = replace(ConceptualMeaning.from_description(value),
        role_refs=(('name', word), None, None), bindings={'_bound_roles': (('referent',0),)})
    model = SimpleNamespace(_concept_owner=lambda: cs, symbolSpace=SimpleNamespace(ltm_store=store))
    assert matches(model, answer, 'label')
    saved = copy.deepcopy(cs.mtm.state())
    cs.mtm.restore(saved)
    assert cs.mtm.state()['names'] == saved['names']


def test_pending_reading_checkpoint_before_first_kept_mention():
    cs, rows = _fixture()
    word = cs.interpret.lookup_word((4,7), (), form='label')
    before = cs.interpret.reading(word)
    saved = copy.deepcopy(cs.mtm.state())
    cs.mtm.restore(None)
    cs.mtm.restore(saved)
    assert cs.interpret.reading(word) == before


def test_epoch_keeps_a_promoted_names_witnessed_order_after_deleting_its_occurrence():
    from FamiliarTier import promote_identity
    from Forgetting import model_boundary
    from Meaning import ConceptualMeaning
    from test_orders_learned import equality_fixture
    from test_presentation_memory import model_for
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word((4, 7), (), form='label')
    identity, row, order = cs.interpret.reading(word)
    value = cs.similarity_codebook.lookup_rows(row)
    index = store.append_meaning(ConceptualMeaning.from_description(value),
        order=order, document_key='first', sentence_index=0)
    occurrence = store.occurrence_of(index)
    key = cs.mtm.token_of(identity)
    cs.mtm.bind(key, occurrence, kind=None, when=0, value=value, reading=word)
    obj = promote_identity(cs, key, word=word)
    before = cs.interpret.reading(word)
    assert before == (obj, cs._csw_row_of(obj), 1)
    for _ in range(2):
        model_boundary(model_for(cs, store), epoch_boundary=True)
        assert store.location(occurrence) is None and not cs.mtm.entries
        assert cs.interpret.reading(word) == before
        assert alloc.settle(obj) == 1
        assert ('part', ('occ', occurrence[-1])) not in alloc.store_of(obj).constituents(obj)


def pending_reading_journal(monkeypatch):
    from dataclasses import replace
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    f = SentenceFixture(monkeypatch)
    program = f.program(('equal', 'left', 'right'))
    words = [f.cs.interpret.lookup_word((81 + n,), (), form='pending-' + str(n))
             for n in range(2)]
    readings = [f.cs.interpret.reading(word) for word in words]
    ids, rows, orders = map(torch.tensor, zip(*readings))
    values = f.cs.similarity_codebook.lookup_rows(rows)
    program = record_reading(f.language, replace(program, rows=rows, word_rows=rows,
        word_ids=torch.tensor(words), concept_ids=ids, leaves=values, leaf_orders=orders,
        identity_leaves=torch.tensor([1, 2]), identity_ids=ids, identity_values=values,
        identity_origins=torch.stack((torch.full_like(ids, 6), torch.tensor(words), torch.zeros_like(ids)), 1),
        identity_actions=torch.zeros(len(program.actions), 6, dtype=torch.long)))
    return f, words, finish_clause(f.language, program, registry=f.registry)


def test_pending_reading_journal_keeps_order_zero_participants(monkeypatch):
    from ReferenceClose import constituents
    f, words, clause = pending_reading_journal(monkeypatch)
    sites = constituents(clause)
    assert len(sites) == 2
    assert all(site.order == 0 for site in sites)
    assert [site.reference_request['reading_identity'] for site in sites] == [1, 2]
    assert [site.reference_request['forward_identity'] for site in sites] == words


@pytest.mark.parametrize('durable_mention', [False, True])
def test_order_zero_bind_keeps_its_introduction_and_event(monkeypatch, durable_mention):
    from ClauseJournal import finish_clause
    from chunk_fixtures import record, context
    from Meaning import ConceptualMeaning
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry, _, handle = record(f, ('lift', ('bind', 'subject'), 'predicate'),
                             leaf_orders=[0, 0])
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, context(resolver, ()))
    introductions = [item for item in plan['items']
        if item['node'].identity['introduction'] and item['node'].identity['handle'] == handle]
    assert len(introductions) == 1 and introductions[0]['node'].order == 0
    word_occurrences = {}
    for index, value in enumerate(entry.leaves):
        row = f.store.append_meaning(ConceptualMeaning.from_description(value),
                                    order=0, document_key='words', sentence_index=index)
        word_occurrences[index] = f.store.occurrence_of(row)
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='order-zero')
    introduction = introductions[0]['written_node']
    if durable_mention:
        # The storage adapter may return the durable referent. The reading
        # close uses its already retained word occurrence as the mention.
        durable = ('sym', f.noun('subject'))
        written[id(introduction)] = durable
        written[('occurrence', id(introduction))] = durable
    resolver.commit(plan, row, written, document='order-zero', position=0, training=False,
                    word_occurrences=word_occurrences)
    occurrence = (word_occurrences[introductions[0]['leaves'][-1]] if durable_mention
                  else written[('occurrence', id(introduction))])
    token = f.cs.mtm.token_of(occurrence)
    assert token is not None and len(f.cs.mtm.entries[token].events) == 1


def test_reading_close_follows_a_named_alias_after_merge(monkeypatch):
    from chunk_fixtures import context
    f, words, clause = pending_reading_journal(monkeypatch)
    pool, resolver = f.cs.mtm, f.cs.components.reference_close
    alias, target = [('name', word) for word in words]
    pool.merge(alias, target, when=0)
    assert alias not in pool.entries and pool.token_of(alias) == target
    plan = resolver.preview(clause, context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, document_key='merged-name',
        sentence_index=0, written_rows=written)
    resolver.commit(plan, row, written, document='merged-name', position=0, training=False)
    assert pool.token_of(alias) == target
    assert pool.entries[target].pending and pool.entries[target].kind is None
    assert len(pool.entries[target].occurrences) == 2


@pytest.mark.parametrize('merged', [False, True])
def test_pending_reading_name_is_not_a_supplied_answer(merged, monkeypatch):
    from chunk_fixtures import context
    from ThoughtReferences import with_slots
    from test_modal_timelines import same
    f, words, clause = pending_reading_journal(monkeypatch)
    pool, resolver = f.cs.mtm, f.cs.components.reference_close
    if merged:
        pool.merge(('name', words[0]), ('name', words[1]), when=0)
    plan = resolver.preview(clause, context(resolver, ()))
    meaning = with_slots(clause.meaning, (('referent', 0), ('referent', 2)), pair=(0., 0.))
    before = copy.deepcopy(pool.state())
    assert resolver.answer_meaning(plan, meaning) is meaning
    same(pool.state(), before)


@pytest.mark.parametrize('pending', [False, True])
def test_reading_event_cases_accept_native_part_rows_and_filter_open_questions(monkeypatch, pending):
    from test_modal_timelines import fixture, recording, close, same
    from CaseSelection import CaseBank, stage_event_cases
    f, action, modifier = fixture(monkeypatch)
    _, frame, _, _ = close(f, recording(f, ('lift', ('lower_unary', 'subject'), 'move')),
        actions=(action, modifier), evidence=(.8, .2))
    token, entry = next(iter(f.cs.mtm.entries.items()))
    event = entry.events[0]
    verb = f.words['move'][1]
    target = (f.cs.interpret.lookup_word((81,), (), form='unresolved-event') if pending else verb)
    symbol = f.store.parts.add(f.words['subject'][1], target, truth_plus=.6, truth_minus=.4)
    occurrence = ('sym', symbol)
    assert f.store.location(occurrence) == occurrence
    assert f.store.row(occurrence)['kind'] == ('question' if pending else 'observation')
    f.cs.mtm.event(token, occurrence, verb=verb, role=0, when=2,
                   before=event.before, after=event.after)
    code = f.cs.similarity_codebook.W[f.cs._csw_row_of(verb)].clone()
    bank = CaseBank(torch.tensor([verb]), code[None, None], code.new_zeros(1, 1, 2),
        code.new_zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
    before = copy.deepcopy(f.cs.mtm.state())
    staged = stage_event_cases(bank, f.cs.components.reference_close, [(frame,)], limit=4)
    assert staged.codes.shape == (1, 4, 8)
    for actual, expected in zip(staged.codes[0, 1:], (event.before, code, event.after)):
        torch.testing.assert_close(actual, expected)
    expected = (.8, .2) if pending else (.6, .4)
    torch.testing.assert_close(staged.cases[0, 0, 1:], code.new_tensor([expected]).expand(3, -1))
    same(f.cs.mtm.state(), before)


def test_named_alias_follows_merge_restore_and_horizon_expiry():
    from MediumTermMemory import MediumTermMemory
    pool = MediumTermMemory(capacity=4, horizon=2, n_vectors=16)
    source = pool.named(3, value=torch.ones(4))
    target = pool.open(('ltm', 'test', 1), kind=None, when=0, value=torch.ones(4))
    pool.bind(source, ('ltm', 'test', 2), kind=None, when=0, reading=3)
    pool.merge(source, target, when=0)
    assert pool.token_of(('name', 3)) == target
    assert pool.named(3, value=torch.zeros(4)) == target
    saved = copy.deepcopy(pool.state())
    pool.restore(saved)
    assert pool.token_of(('name', 3)) == target
    pool.close(None, stream=0, references=(target,))
    for _ in range(pool.horizon + 1):
        pool.close(None, stream=0)
    assert not pool.entries and not pool.names and not pool.transient_used.any()


def test_forward_name_is_not_filled_by_its_own_defined_word():
    from dataclasses import replace
    from definition_fixtures import witness_object
    from Meaning import ConceptualMeaning
    from ThoughtReferences import open_slots
    cs, rows = _fixture()
    word = cs.interpret.lookup_word((4,7), (), form='label')
    witness_object(cs, word)
    value = cs.similarity_codebook.lookup_rows(cs._csw_row_of(word))
    meaning = replace(ConceptualMeaning.from_description(value),
        role_refs=(('sym',word),None,None), bindings={
            '_forward_references': ((0,word),), '_open_references': (('referent',0),)})
    store = cs._closed_clause_store()
    index = store.append_meaning(meaning, kind='question', order=0)
    result = store.read_reference_bindings(index, meaning)
    assert open_slots(result) == (('referent',0),)
    assert dict(result.bindings)['_forward_references'] == ((0,word),)


def test_kept_word_mention_retains_its_pending_referent():
    from FamiliarTier import commit_read_words
    from ThoughtReferences import open_slots
    cs, rows = _fixture()
    word = cs.interpret.lookup_word((4,7), (), form='label')
    identity, row, order = cs.interpret.reading(word)
    reading = SimpleNamespace(word_ids=torch.tensor([word]),
        leaves=cs.similarity_codebook.lookup_rows([row]), symbol_where=None)
    occurrence = commit_read_words(cs, reading, batch=0, document='first', position=0)[0]
    store = cs._closed_clause_store()
    index = store.location(occurrence)
    assert cs.mtm.entries[cs.mtm.token_of(occurrence)].pending
    assert store.KINDS[int(store.record_kind[index])] == 'question'
    assert float(store.c_plus[index]) == float(store.c_minus[index]) == 0.
    for meaning in (store.raw_meaning_of(index), store.meaning_of(index)):
        assert open_slots(meaning) == (('referent',0),)
        assert dict(meaning.bindings)['_forward_references'] == ((0,word),)


@pytest.mark.parametrize('promoted', [False, True])
def test_thought_part_persists_a_named_token_through_its_native_reference(monkeypatch, promoted):
    from dataclasses import replace
    from FamiliarTier import commit_read_words
    from ThoughtStream import write
    from ThoughtReferences import with_slots, open_slots
    from test_item6_2_thinking import world
    import Occurrence
    model, registry, store, (a, b, _) = world()
    cs = model.conceptualSpace
    word = cs.interpret.lookup_word((81,), (), form='pending-label')
    _, row, _ = cs.interpret.reading(word)
    reading = SimpleNamespace(word_ids=torch.tensor([word]),
        leaves=cs.similarity_codebook.lookup_rows([row]), symbol_where=None)
    occurrence = commit_read_words(cs, reading, batch=0, document='first', position=0)[0]
    key = cs.mtm.token_of(('name', word))
    if promoted:
        cs.mtm.promote(key, a[1])
    value = registry.form('isEqual', a, b)
    value = with_slots(replace(value, role_refs=(('name', word), value.role_refs[1], b)), (), pair=(0., 0.))
    monkeypatch.setattr(Occurrence, 'source_at', lambda *args: ('thought-document', 0))
    before = copy.deepcopy(cs.mtm.state())
    index = write(model, value, row=0)
    saved = store.raw_meaning_of(index)
    assert saved.role_refs[0] == (a if promoted else occurrence)
    assert bool(open_slots(saved)) is not promoted
    from test_modal_timelines import same
    same(cs.mtm.state(), before)


def test_thought_selection_of_pending_token_keeps_its_open_referent(monkeypatch):
    from dataclasses import replace
    from test_bind_thought import native, site
    from ThoughtReferences import open_slots
    f = native(monkeypatch)
    key = f.cs.mtm.named(7, value=f.atoms[0])
    meaning = replace(site(f, f.atoms[0]), bindings={
        '_forward_references': ((0,7),), '_open_references': (('referent',0),)})
    plan = dict(meaning=meaning, role=0, candidates=(key,), features=torch.zeros(1,8),
        cue=f.atoms[0], individuals={key:dict(value=f.atoms[0])}, records={}, base={}, forward_identity=7)
    monkeypatch.setattr(f.resolver, 'select', lambda *args, **kwargs: 0)
    result = f.resolver.finish_thought_bind(plan, logits=torch.ones(1))
    assert plan['candidates'] == (key,)
    assert result['incomplete'] == ('pending_antecedent',)
    assert open_slots(result['meaning']) == (('referent',0),)


def test_lesson13_has_two_phases_distinct_documents_and_supplied_math_records():
    from corpus_lessons import stage_documents
    root = Path(__file__).resolve().parents[1]
    lesson = json.loads((root/'data/corpus-lessons.json').read_text())
    assert lesson['version'] == 2 and lesson['epochs'] == 4
    assert [stage['name'] for stage in lesson['stages']] == ['ideas','relations']
    for stage in lesson['stages']:
        assert len(stage['documents']) == 4
        data = SimpleNamespace(text_answers={},source_addresses={})
        texts, targets = stage_documents(data, stage)
        assert sum(target is not None for target in targets) >= 8
        assert data.has_supervised_outputs
        assert len({row['document'] for row in data.source_addresses['train']}) == 4
        assert len({tuple(record['sentences'][0] for record in doc['records']) for doc in stage['documents']}) == 4
        assert not any(key in record for doc in stage['documents'] for record in doc['records']
                       for key in ('order','operation','part_of_speech'))
    registry = json.loads((root/'test/fixtures/phase-artifacts.json').read_text())
    assert registry['producers']['copular_identity']['phase'] == 'relations'


def test_native_read_sites_carry_one_token_across_three_documents(eager_reading):
    from data import TheData
    from Models import BaseModel
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    root = Path(__file__).resolve().parents[1]
    from util import init_config
    init_config(path=str(root/'data/MM_corpus_orders.xml'), defaults_path=str(root/'data/model.xml'))
    import Language
    Language.TheGrammar._configured = False
    TheData.processLM({split:dict(text=['.'] if split == 'train' else [],label=[])
                      for split in ('train','validation','test')})
    model = BaseModel.from_config(str(root/'data/MM_corpus_orders.xml'), data=TheData)[0]
    stage_documents(TheData, dict(documents=[dict(document=i,key=str(i),records=[
        dict(key=str(i),sentences=['label.'])]) for i in range(3)]))
    identities = []
    def observed(model, split, rows, result):
        cs = model._concept_owner()
        word = cs.interpret.word_for_form('label')
        assert word is not None
        key = cs.mtm.token_of(('name', word))
        assert key is not None
        entry = cs.mtm.entries[key]
        identities.append(key)
        assert entry.occurrences, 'the read word did not retain a kept mention'
        if len(identities) < cs.concept_mint_count:
            assert entry.pending and not cs.definitions.objects(word)
        else:
            assert entry.symbol in cs.definitions.objects(word)
            assert cs._concept_allocator.order_of(entry.symbol) == 1
    try:
        present(model, split='train', optimizer=model.getOptimizer(lr=.001), batch_size=1,after_batch=observed)
        assert len(identities) == 3 and len(set(identities)) == 1
        pool = model._concept_owner().mtm
        pool.restore(copy.deepcopy(pool.state()))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_defining_clause_promotes_a_named_reading_before_recurrence():
    from test_orders_learned import equality_fixture, equal_close
    from Meaning import ConceptualMeaning
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word((4,), (), form='subject')
    kind = cs.interpret.lookup_word((7,), (), form='kind')
    identity, row, order = cs.interpret.reading(word)
    token = cs.mtm.token_of(identity)
    value = cs.similarity_codebook.lookup_rows(row)
    index = store.append_meaning(ConceptualMeaning.from_description(value), order=order,
                                document_key='first', sentence_index=0)
    occurrence = store.occurrence_of(index)
    cs.mtm.bind(token, occurrence, kind=None, when=0, value=value, reading=word)
    cs.mtm.close(store, stream=0, references=(occurrence,))
    assert cs.mtm.promote_candidates(cs) == []
    equal_close(cs, store, occurrence[-1], kind, word=kind)
    obj, = cs.definitions.objects(word)
    assert cs.mtm.entries[token].symbol == obj and alloc.order_of(obj) == 1
    assert cs.interpret.reading(word)[0] == obj


def test_final_curriculum_has_four_ordered_phases_and_untrained_recombinations():
    import json
    from pathlib import Path
    from math_chain_corpus import ChainDocument,flatten
    source=json.loads((Path(__file__).resolve().parents[1]/'data/corpus-lessons.json').read_text())
    curriculum=source['final_round']
    assert curriculum['epochs']==4
    assert [stage['name'] for stage in curriculum['stages']]==['proper','mass','count-kinds','kind-verbs']
    for stage in curriculum['stages']:
        assert len(stage['documents'])==4 and stage['heldout']
        assert len({json.dumps(d['records'],sort_keys=True) for d in stage['documents']})==4
        for document in [*stage['documents'],*stage['heldout']]:
            for record in document['records']:
                item=ChainDocument(**dict(record,sentences=tuple(record['sentences'])))
                text,targets,addresses=flatten([item],supplied=True)
                assert len(text)==len(targets)==len(addresses)
                assert targets[item.question]==item.answer
                assert not {'order','operation','part_of_speech'} & record.keys()
    assert source['stages'][1]['produces']=='copular_identity'
