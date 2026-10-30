"""A training presentation forgets experience, not its learned concepts."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from DocumentCover import open_cover
from Layers import LTM0
from Meaning import ConceptualMeaning
from PresentationMemory import begin as begin_document
from Forgetting import model_boundary

def begin(model):
    return model_boundary(model, epoch_boundary=True)
from test_document_covers import fixture


def test_document_driver_clears_once_per_epoch_and_keeps_evaluation_memory(monkeypatch):
    from MathChainTraining import present
    addresses = [dict(document=book, sentence=sentence)
                 for book in ('first', 'second') for sentence in range(2)]
    data = SimpleNamespace(source_addresses={split: addresses for split in ('train', 'validation')})
    for split in ('train', 'validation'):
        setattr(data, split + '_input', ['first 0', 'first 1', 'second 0', 'second 1'])
        setattr(data, split + '_output', [0.] * 4)
    memory, starts, reads = [], [], []
    def boundary(model, *, epoch_boundary):
        assert epoch_boundary
        starts.append(tuple(memory))
        memory.clear()
    import Forgetting
    monkeypatch.setattr(Forgetting, 'model_boundary', boundary)
    documents = []
    def begin_presentation():
        documents.append(tuple(memory))
    def batch(**kwargs):
        reads.append((kwargs['split'], tuple(memory), tuple(kwargs['source_rows'])))
        memory.extend(kwargs['source_rows'])
        return None, None
    model = SimpleNamespace(inputSpace=SimpleNamespace(data=data, prepInput=lambda values: values),
        outputSpace=SimpleNamespace(prepOutput=lambda values: values),
        symbolSpace=SimpleNamespace(expectation=SimpleNamespace(reset=lambda: None)),
        _what_memory=lambda: None, train=lambda _: None, teacher=None,
        begin_presentation=begin_presentation, runBatch=batch,
        flush_word_buffers=lambda: None, dispatch_per_row_reset=lambda _: None,
        dispatch_soft_reset=lambda: None, post_tick_compact=lambda: None)
    for _ in range(2):
        report = present(model, split='train', optimizer=object(), batch_size=1)
        assert report == dict(sentences=4, batches=4)
        assert memory == [0, 1, 2, 3]
    assert starts == [(), (0, 1, 2, 3)]
    assert documents == [(), (0, 1)] * 2
    assert [before for _, before, _ in reads] == [(), (0,), (0, 1), (0, 1, 2)] * 2
    present(model, split='validation', batch_size=1)
    assert len(starts) == 2
    assert reads[8] == ('validation', (0, 1, 2, 3), (0,))


def model_for(owner, store):
    model = SimpleNamespace(conceptualSpace=owner, conceptualSpaces=[owner],
        symbolSpace=SimpleNamespace(ltm_store=store),
        _what_recall_sentence_history={0: ['old occurrence']})
    from Models import BasicModel
    model._clear_recall_history = lambda: BasicModel._clear_recall_history(model)
    return model


def write(store, point, document, origin=None, **kwargs):
    row = store.append_meaning(ConceptualMeaning.from_description(point),
        kind='observation', evidence=(1., 0.), document_key=document, **kwargs)
    if origin is not None:
        store.set_origin(row, origin)
    return store.occurrence_of(row)


@pytest.mark.parametrize('given', (False, True))
def test_boundary_clears_experience_and_occurrence_caches_preserving_learning(given):
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    experienced = write(store, atoms[0], 'experience')
    supplied = [] if not given else [
        write(store, atoms[n], ('given', n), origin)
        for n, origin in ((1, store.ORIGIN_USER), (2, store.ORIGIN_PROVISIONED))]
    supplied_rows = [store.row(store.location(ref)) for ref in supplied]
    owner.mtm.open(experienced, kind=None, value=atoms[0], when=0)
    from MemoryConsolidation import recurrence_groups
    recurrence_groups(store)
    assert store.recurrence_used.any()
    owner.components._innovations[experienced] = atoms[0]
    owner.components.verbs._pending.append({'witnesses': {experienced}, 'direction': atoms[0]})
    owner.components.verbs._witnesses[1] = {experienced}
    resolver._mentions = {experienced: atoms[0]}
    resolver.last_audit = ({'occurrence': experienced},)
    owner._concept_allocator.testimony_seen[1] = {experienced}
    model._document_covers = {0: ('old', 1, experienced)}
    model._thought_document_ends = (True,)
    model._last_understanding = {'row': experienced}
    model._sentence_understandings = {0: model._last_understanding}
    model._last_sentence_credit = {'bind_choices': ((experienced,),)}
    model._last_closing_thoughts = (model._last_understanding,)
    model._thought_ordinals = {('thought', 'experience', 0): 1}
    model._last_what_thinking = SimpleNamespace(slots=[experienced])
    model._recall_brackets = SimpleNamespace(row=experienced)
    for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence',
                 'forget_cutoff', 'forget_pending'):
        getattr(store, name).fill_(1)
    parameters = {name: value.detach().clone() for name, value in owner.named_parameters()}
    report = begin(model)
    assert report == dict(removed=1, retained=len(supplied), relations_removed=0)
    assert store.location(experienced) is None
    assert not owner.mtm.entries and not owner._concept_allocator.testimony_seen
    assert not owner.components._innovations and not owner.components.verbs._pending
    assert not owner.components.verbs._witnesses and not resolver.last_audit
    assert not getattr(resolver, '_mentions', {}) and not store.recurrence_used.any()
    assert store.recurrence_readings.eq(-1).all()
    assert model._document_covers == {} and model._thought_document_ends == ()
    assert model._last_understanding is None and model._sentence_understandings == {}
    assert model._last_sentence_credit is None and model._last_closing_thoughts == ()
    assert model._thought_ordinals == {}
    assert model._last_what_thinking is None and model._recall_brackets is None
    assert not model._what_recall_sentence_history
    assert all(not bool(getattr(store, name).any()) for name in (
        'forget_passes', 'forget_sentences', 'forget_last_sentence', 'forget_cutoff', 'forget_pending'))
    for ref, before in zip(supplied, supplied_rows):
        after = store.row(store.location(ref))
        assert after['origin'] == before['origin'] and after['evidence'] == before['evidence']
        torch.testing.assert_close(after['meaning'].roles, before['meaning'].roles, rtol=0, atol=0)
    for name, value in owner.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)


def test_given_knowledge_keeps_its_required_dependency_without_experience_authority():
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    dependency = write(store, atoms[0], 'dependency')
    given = replace(ConceptualMeaning.from_description(atoms[1]),
                    role_refs=(dependency, None, None))
    parent = store.append_meaning(given, kind='observation', evidence=(1., 0.), document_key='given')
    store.set_origin(parent, store.ORIGIN_PROVISIONED)
    begin(model_for(owner, store))
    assert len(store) == 2 and store.location(dependency) is not None
    assert store.raw_meaning_of(parent).role_refs[0] == dependency
    assert store.row(store.location(dependency))['evidence'] == (0., 0.)
    assert store.row(parent)['evidence'] == (1., 0.)


def test_boundary_removes_occurrence_relations_transitively_and_keeps_concept_relations():
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    from LTM1 import OCCURRENCE, SYMBOL, RELATION
    occurrence = write(store, atoms[0], 'dependency')
    left, right = owner.new_concept(), owner.new_concept()
    store.parts.add(left, right, symbol=400, truth_plus=1.)
    store.parts.add((OCCURRENCE, occurrence[2]), (SYMBOL, right), symbol=401, truth_plus=.5)
    store.parts.add((RELATION, 401), (SYMBOL, left), symbol=402, truth_plus=.5)
    before = {name: value.clone() for name, value in store.parts.row(400).items()}
    report = begin(model_for(owner, store))
    assert report['relations_removed'] == 2 and store.parts.indices().tolist() == [400]
    for name, value in store.parts.row(400).items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_cover_survives_empty_memory_checkpoint_and_reuses_symbol_without_rng():
    from ConceptForgetting import protected_concepts
    resolver, store, _ = fixture(percept=True)
    owner = resolver.components.owner
    first = open_cover(owner, store, stream=0, document='book')
    symbol = store.raw_meaning_of(store.location(first)).role_refs[0][1]
    when = store.entry_when(store.location(first)).clone()
    rng = store.cover_rng_state.clone()
    begin(model_for(owner, store))
    assert not len(store) and symbol in protected_concepts(owner, store)
    restored = LTM0(store.nDim, capacity=store.capacity, percept_width=store.percept_width)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    # A learned cover vector cannot change the address of that book.
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(symbol)].mul_(.9)
    second = open_cover(owner, restored, stream=2, document='book')
    assert first == second and len(restored) == 1
    assert restored.raw_meaning_of(0).role_refs[0] == ('sym', symbol)
    assert not torch.equal(restored.entry_when(0), when)
    assert restored.witness_count[0] == 1 and restored.stream_rows(2).tolist() == [0]
    torch.testing.assert_close(restored.cover_rng_state, rng, rtol=0, atol=0)
    open_cover(owner, restored, stream=2, document='other book')
    assert len(restored._cover_symbols) == 2 and not torch.equal(restored.cover_rng_state, rng)


@pytest.mark.parametrize('bad', ({'short': 1}, {'0'*64: -1}, {'0'*64: True},
                                {'0'*64: 1, '1'*64: 1}))
def test_invalid_cover_map_does_not_replace_a_valid_checkpoint(bad):
    resolver, store, _ = fixture()
    open_cover(resolver.components.owner, store, stream=0, document='book')
    before = copy.deepcopy(store.semantic_extras())
    malformed = dict(before, cover_symbols=bad)
    with pytest.raises(ValueError, match='cover concept checkpoint'):
        store.load_semantic_extras(malformed)
    assert store.semantic_extras() == before


def test_boundary_refuses_to_erase_a_live_sentence():
    resolver, store, atoms = fixture()
    model = model_for(resolver.components.owner, store)
    row = write(store, atoms[0], 'live')
    model._sentence_run_active = True
    with pytest.raises(RuntimeError, match='live sentence'):
        begin(model)
    assert store.location(row) is not None


def test_verb_group_boundary_forgets_witnesses_but_keeps_the_learned_basis():
    from MeaningKernels import MeaningAction
    resolver, store, atoms = fixture()
    model = model_for(resolver.components.owner, store)
    occurrence = write(store, atoms[0], 'event')
    action = MeaningAction(4)
    action.groups.used = 1
    action.groups.assignments[7] = 0
    action.groups.witnesses[8] = [occurrence]
    action.groups.reports.append(dict(occurrence=occurrence, verb=8))
    before = action.groups.vectors.detach().clone()
    model.languageSpace = SimpleNamespace(verb_action=lambda: action, modifier_action=lambda: None)
    begin(model)
    assert action.groups.witnesses == {} and action.groups.reports == []
    assert action.groups.used == 1 and action.groups.assignments == {7: 0}
    torch.testing.assert_close(action.groups.vectors, before, rtol=0, atol=0)
