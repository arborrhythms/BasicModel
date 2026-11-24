"""Item 2.7 §29: durable reuse, lexical exclusion, epoch and reading contracts."""
from Memory import STM0
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from FamiliarTier import observe_candidate, witnessed_match
from test_attention_promotion import _fixture, _close
from nomination_fixtures import recurrent_word


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


def test_support_reuse_requires_order_and_parthood_and_singleton_opens_nothing():
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
    # The class chart is learned from context (§30.7); reuse compares its
    # retained witness centroid, not the class's own chart coordinates.
    assert witnessed_match(cs, 1, support) == cid
    remember_candidate(cs, cid, {rows[0][1]: 9., rows[1][1]: 1.})
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
    row = cs._concept_row_of(cid)
    before = dict(cs.concept_weights(row))
    cs.mtm.restore(None)
    observe_candidate(cs, 0, {**support, rows[2][1]: 2.}, literal=('ps',(4,7)), literal_key=b'case')
    after = dict(cs.concept_weights(row))
    # A candidate of order zero reads basic percepts in the order-zero
    # Sigma: a cover over members of its own order writes no entry of the
    # concept store (the merge by maximum is test_attention_promotion's,
    # at order one).
    assert cs._row_order(row) == 0 and before == after == {}


def test_lexical_letters_open_nothing_but_nonlexical_located_case_does():
    from FamiliarTier import observe_native_cases
    cs, rows = _fixture(pi=True)
    raw = {4:b'a', 7:b'b'}
    native = SimpleNamespace(bytes_for=raw.get)
    cs._model = SimpleNamespace(perceptualSpace=SimpleNamespace(percept_store=native, fuse_parts=lambda ids: ids,
        _forward_input=dict(native_indices=torch.tensor([[4,7]]), native_part_spans=torch.tensor([[[0,1],[1,2]]]))))
    from clock_fixtures import install
    install(cs._model, tick=1)
    cs._percept_field = SimpleNamespace(where=torch.tensor([[[0,2]]]))
    support = {rows[0][1]: 1., rows[1][1]: 1.}
    observe_native_cases(cs, 0, support)
    assert len(cs.mtm.entries) == 1
    cs.mtm.restore(None)
    model = cs._model
    cs._model = None
    word = recurrent_word(cs.interpret, (4,7), (), form=b'ab')
    cs._model = model
    observe_native_cases(cs, 0, support)
    assert not cs.mtm.entries
    # The located case is no object of the lexical word.
    assert not cs.definitions.objects(word)


def test_documents_and_epochs_retain_history_and_live_mtm():
    from test_presentation_memory import model_for, write
    from test_document_covers import fixture
    from Forgetting import model_boundary
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    model_boundary(model, epoch_boundary=True)
    first = write(store, atoms[0], 'first')
    owner.mtm.open(first, kind=None, when=0, value=atoms[0])
    second = write(store, atoms[1], 'second')
    owner.mtm.open(second, kind=None, when=0, value=atoms[1])
    assert all(store.location(ref) is not None and owner.mtm.token_of(ref) is not None for ref in (first,second))
    model_boundary(model, epoch_boundary=True)
    assert len(store) == 2 and all(owner.mtm.token_of(ref) is not None for ref in (first, second))


def test_pending_reading_checkpoint_before_first_kept_mention():
    cs, rows = _fixture()
    word = recurrent_word(cs.interpret, (4,7), (), form='label')
    before = cs.interpret.reading(word)
    saved = copy.deepcopy(cs.mtm.state())
    cs.mtm.restore(None)
    cs.mtm.restore(saved)
    assert cs.interpret.reading(word) == before


def pending_reading_journal(monkeypatch):
    from dataclasses import replace
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    f = SentenceFixture(monkeypatch)
    program = f.program(('part', 'left', 'right'))
    words = [recurrent_word(f.cs.interpret, (81 + n,), (), form='pending-' + str(n))
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


@pytest.mark.parametrize('durable_mention', [False, True])
def test_order_zero_bind_keeps_its_introduction_and_event(monkeypatch, durable_mention):
    from ClauseJournal import finish_clause
    from chunk_fixtures import record, context
    from Meaning import ConceptualMeaning
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry, _, handle = record(f, ('change', ('bind', 'subject'), 'predicate'),
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
                                     )
        word_occurrences[index] = f.store.occurrence_of(row)
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written)
    introduction = introductions[0]['written_node']
    if durable_mention:
        # §50/§52: a durable semantic return leaves the separately recorded
        # located mention intact; it does not substitute a word occurrence.
        durable = ('sym', f.noun('subject'))
        written[id(introduction)] = durable
        written[('occurrence', id(introduction))] = durable
    resolver.commit(plan, row, written, document='order-zero', position=0, training=False)
    occurrence = written[('mention', id(introduction))]
    assert occurrence not in word_occurrences.values()
    assert plan['mention_occurrences'][id(introduction)] == occurrence
    token = f.cs.mtm.token_of(occurrence)
    assert token is not None and len(f.cs.mtm.entries[token].events) == 1


@pytest.mark.parametrize('pending', [False, True])
def test_reading_event_cases_accept_native_part_rows_and_filter_open_questions(monkeypatch, pending):
    from test_modal_timelines import fixture, recording, close, same
    from CaseSelection import CaseBank, stage_event_cases
    f, action, modifier = fixture(monkeypatch)
    _, frame, _, _ = close(f, recording(f, ('change', ('lower_unary', 'subject'), 'move')),
        actions=(action, modifier), evidence=(.8, .2))
    token, entry = next(iter(f.cs.mtm.entries.items()))
    event = entry.events[0]
    verb = f.words['move'][1]
    target = (recurrent_word(f.cs.interpret, (81,), (), form='unresolved-event') if pending else verb)
    symbol = f.store.parts.add(f.words['subject'][1], target, truth_plus=.6, truth_minus=.4)
    occurrence = ('sym', symbol)
    assert f.store.location(occurrence) == occurrence
    f.cs.mtm.event(token, occurrence, verb=verb, role=0, when=2,
                   before=event.before)
    code = f.cs.similarity_codebook.W[f.cs._concept_row_of(verb)].clone()
    bank = CaseBank(torch.tensor([verb]), code[None, None], code.new_zeros(1, 1, 2),
        code.new_zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
    before = copy.deepcopy(f.cs.mtm.state())
    staged = stage_event_cases(bank, f.cs.components.reference_close, [(frame,)], limit=4)
    assert staged.codes.shape == (1, 4, f.cs.nDim)
    selected = event if pending else f.cs.mtm.entries[token].events[-1]
    for actual, expected in zip(staged.codes[0, 1:], (event.before, code, f.cs.mtm.event_value(selected))):
        torch.testing.assert_close(actual, expected)
    expected = (.8, .2) if pending else (.6, .4)
    torch.testing.assert_close(staged.cases[0, 0, 1:], code.new_tensor([expected]).expand(3, -1))
    same(f.cs.mtm.state(), before)


def test_forward_name_is_not_filled_by_its_own_defined_word():
    from dataclasses import replace
    from definition_fixtures import witness_object
    from Meaning import ConceptualMeaning
    from ThoughtReferences import open_slots
    cs, rows = _fixture()
    word = recurrent_word(cs.interpret, (4,7), (), form='label')
    witness_object(cs, word)
    value = cs.similarity_codebook.lookup_rows(cs._concept_row_of(word))
    meaning = replace(ConceptualMeaning.from_description(value),
        role_refs=(('sym',word),None,None), bindings={
            '_forward_references': ((0,word),), '_open_references': (('referent',0),)})
    store = cs._closed_clause_store()
    index = store.append_meaning(meaning, )
    result = store.read_reference_bindings(index, meaning)
    assert open_slots(result) == (('referent',0),)
    assert dict(result.bindings)['_forward_references'] == ((0,word),)


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


@pytest.fixture
def symbol_interpretation(monkeypatch):
    """Keep the symbol in a one-word clause with the existing instance menu."""
    from Language import OperationSelectionLayer
    configure = OperationSelectionLayer.configure_arguments
    def configured(layer, *args, **kwargs):
        configure(layer, *args, **kwargs)
    monkeypatch.setattr(OperationSelectionLayer, 'configure_arguments', configured)


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
