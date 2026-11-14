"""Retained truth and learning survive an activity boundary (§17.14, 21)."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Memory import LTM0
from Meaning import ConceptualMeaning
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
    from clock_fixtures import install
    model = SimpleNamespace(conceptualSpace=owner, conceptualSpaces=[owner],
        symbolSpace=SimpleNamespace(ltm_store=store),
        _what_recall_sentence_history={0: ['old occurrence']})
    install(model)
    from Models import BasicModel
    model.present = lambda: BasicModel.present(model)
    model._clear_recall_history = lambda: BasicModel._clear_recall_history(model)
    return model


def write(store, point, document, *, trust=1., **kwargs):
    row = store.append_meaning(ConceptualMeaning.from_description(point), trust=trust, **kwargs)
    return store.occurrence_of(row)



@pytest.mark.parametrize('given', (False, True))
def test_boundary_preserves_retained_rows_and_learning(given):
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    begin(model)
    experienced = write(store, atoms[0], 'experience', trust=0.)
    supplied = [] if not given else [write(store, atoms[n], ('given', n)) for n in (1, 2)]
    supplied_rows = [store.row(store.location(ref)) for ref in supplied]
    owner.mtm.open(experienced, kind=None, value=atoms[0], when=0)
    parameters = {name: value.detach().clone() for name, value in owner.named_parameters()}
    begin(model)
    assert store.location(experienced) is not None
    for ref, before in zip(supplied, supplied_rows):
        after = store.row(store.location(ref))
        assert after['evidence'] == before['evidence']
        torch.testing.assert_close(after['meaning'].roles, before['meaning'].roles, rtol=0, atol=0)
    for name, value in owner.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)



def test_given_knowledge_keeps_its_required_dependency_without_experience_authority():
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    begin(model_for(owner, store))
    dependency = write(store, atoms[0], 'dependency', trust=0.)
    given = replace(ConceptualMeaning.from_description(atoms[1]), bindings={'_identity': dependency})
    parent = store.append_meaning(given, trust=1.)
    begin(model_for(owner, store))
    assert len(store) == 2 and store.location(dependency) is not None
    assert store.referent_of(parent) == dependency
    assert store.row(store.location(dependency))['evidence'] == (0., 0.)
    assert store.row(parent)['evidence'] == (1., 0.)









def test_boundary_preserves_a_live_sentence():
    resolver, store, atoms = fixture()
    model = model_for(resolver.components.owner, store)
    row = write(store, atoms[0], 'live')
    model._open_sentence_slot = 0
    assert begin(model) is None
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
    assert action.groups.used == 1 and action.groups.assignments == {7: 0}
    torch.testing.assert_close(action.groups.vectors, before, rtol=0, atol=0)


@pytest.mark.parametrize('negative', [0., .8])
def test_unasserted_part_does_not_raise_a_retained_identity_order(negative):
    from FamiliarTier import promote_identity
    from Memory import SYMBOL, RELATION
    from order_fixtures import force_order
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    alloc = owner._concept_allocator
    begin(model_for(owner, store))
    origin = write(store, atoms[0], 'identity-origin')
    key = owner.mtm.open(origin, kind=None, value=atoms[0], when=0)
    cid = promote_identity(owner, key)
    assert alloc.order_of(cid) == 1
    left = owner.new_concept()
    force_order(owner, left, 0)
    other_origin = write(store, atoms[1], 'other-identity-origin')
    other = owner.mtm.open(other_origin, kind=None, value=atoms[1], when=0)
    right = promote_identity(owner, other)
    relation = store.parts.add(left, right, truth_plus=1.)
    assert owner._concept_source_order(relation) == 1
    unknown = store.parts.add((RELATION, relation), (SYMBOL, cid), truth_minus=negative)
    assert alloc.settle(cid) == 1
    assert ('part', ('rel', relation)) not in alloc.records(cid)
    # Affirming this higher-order member is a real structural disagreement;
    # the native order guard must still reject it.
    store.parts.add((RELATION, relation), (SYMBOL, cid), truth_plus=.2)
    assert ('part', ('rel', relation)) in alloc.records(cid)
    with pytest.raises(ValueError, match='recorded concept order disagrees'):
        alloc.settle(cid)
