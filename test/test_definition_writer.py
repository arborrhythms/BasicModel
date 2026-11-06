"""Thinking §52 item 109: only explicitly named pairs are definitions."""
from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest
import torch

from ClauseRow import Clause, Composition
from Definitions import DefinitionIndex
from Meaning import ConceptualMeaning
from test_clause_acceptance import SentenceFixture


def _setup(monkeypatch):
    f = SentenceFixture(monkeypatch)
    first, second = f.noun('first'), f.noun('second')
    word = f.cs.interpret.lookup_word([81], [], form='later')
    genuine = f.cs.new_concept()
    f.cs.add_part(genuine, 81)
    f.cs._csw_concept_row(0, genuine)
    return f, word, first, second, genuine


def _chain(f, word, original, pair):
    left = Composition(f.registry._payload(('sym', original)), reference=original)
    right = Composition(torch.eye(8)[3], reference=word, word_id=word)
    point = torch.maximum(left.point, right.point)
    term = Composition(point, (left, right), operation='union', formal_name='union')
    clause = Clause(ConceptualMeaning.from_description(point), point=point, composition=term)
    row = f.store.write_clause(clause, evidence=pair)
    assert row[0] == 'sym'
    return row[1]


def _view(index):
    return {word: (index.objects(word),
                   tuple(index.row(word, obj) for obj in index.objects(word)),
                   index.description(word))
            for word in tuple(index.word_ids)}


@pytest.mark.parametrize('order', tuple(permutations(('first', 'second', 'name'))))
def test_incremental_index_equals_rebuild_in_every_write_order(monkeypatch, order):
    f, word, first, second, genuine = _setup(monkeypatch)
    named = False
    for operation in order:
        if operation == 'name':
            f.cs.interpret.define(word, genuine)
            named = True
        else:
            # Reorder the same row writes, independent of whether a later
            # grammar close can now compress its already-defined operands.
            f.store.parts.add(first if operation == 'first' else second, word,
                              truth_plus=1.)
        fresh = DefinitionIndex(f.store.parts, f.cs._concept_allocator)
        assert _view(f.cs.definitions) == _view(fresh)
        assert f.cs.definitions.objects(word) == ((genuine,) if named else ())


@pytest.mark.parametrize('pair', ((0., 0.), (0., 1.), (1., 0.)))
def test_late_descriptor_keeps_only_the_explicitly_named_object(monkeypatch, pair):
    f, word, first, _, genuine = _setup(monkeypatch)
    row = _chain(f, word, first, pair)
    saved = {name: value.clone() for name, value in f.store.parts.row(row).items()}
    assert f.cs.definitions.objects(word) == ()
    f.cs.interpret.define(word, genuine)
    assert f.cs.definitions.objects(word) == (genuine,)
    f.cs.definitions.rebuild()
    assert f.cs.definitions.objects(word) == (genuine,)
    assert f.cs.definitions.words(first) == (f.words['first'][0],)
    for name, value in saved.items():
        torch.testing.assert_close(f.store.parts.row(row)[name], value, rtol=0, atol=0)


def test_define_names_the_pair_before_the_store_and_index_read_it(monkeypatch):
    f, word, first, _, _ = _setup(monkeypatch)
    _chain(f, word, first, (0., 0.))
    original = f.store.parts.add
    calls = []
    def add(obj, name, **kwargs):
        if (obj, name) == (first, word):
            assert obj in f.cs._concept_allocator.named_identities.get(name, ())
            calls.append((obj, name))
        return original(obj, name, **kwargs)
    monkeypatch.setattr(f.store.parts, 'add', add)
    f.cs.interpret.define(word, first)
    assert calls == [(first, word)]
    assert f.cs.definitions.objects(word) == (first,)


@pytest.mark.parametrize('pair', ((0., 0.), (0., 1.), (1., 0.)))
def test_named_pair_permission_is_not_an_evidence_filter(monkeypatch, pair):
    f, word, _, _, genuine = _setup(monkeypatch)
    row = f.cs.interpret.define(word, genuine)
    f.store.parts.truth_plus[row], f.store.parts.truth_minus[row] = pair
    f.cs.definitions.rebuild()
    assert f.cs.definitions.objects(word) == (genuine,)
    f.store.parts.remove(genuine, word)
    assert f.cs.definitions.objects(word) == ()
    assert f.cs.interpret.remembered_objects(word) == (genuine,)


def _checkpoint_model(cs):
    from Models import BaseModel
    model = BaseModel()
    model.name = 'DefinitionWriterCheckpoint'
    model.spaces, model.conceptualSpaces, model.wholeSpaces = [], [cs], []
    return model


def test_old_definition_checkpoint_raises_before_any_allocator_mutation(monkeypatch):
    f, _, _, _, _ = _setup(monkeypatch)
    model = _checkpoint_model(f.cs)
    before = model._collect_structural_extras()
    saved = deepcopy(before['conceptual_spaces'][0]['allocator'])
    saved.pop('named_identities')
    saved['placement'] = {}
    saved['word_forms'] = {}
    alloc = f.cs._concept_allocator
    placement, forms = dict(alloc.placement), deepcopy(alloc.word_forms)
    rows = {name: value.clone() for name, value in alloc.parts.state_dict().items()}
    with pytest.raises(ValueError, match='named_identities'):
        model._restore_allocator_extras(f.cs, saved)
    assert alloc.placement == placement and alloc.word_forms == forms
    for name, value in rows.items():
        torch.testing.assert_close(alloc.parts.state_dict()[name], value, rtol=0, atol=0)


def test_old_definition_checkpoint_raises_before_materializing_an_allocator():
    from Models import BaseModel
    calls = []
    cs = SimpleNamespace(_concept_tables=lambda: calls.append('materialized'))
    with pytest.raises(ValueError, match='named_identities'):
        BaseModel()._restore_allocator_extras(cs, {'word_descriptions': {7: {'forms': ('later',)}}})
    assert calls == []


def test_checkpoint_load_does_not_backfill_names_from_generic_rows(monkeypatch):
    f, word, first, _, genuine = _setup(monkeypatch)
    _chain(f, word, first, (1., 0.))
    f.cs.interpret.define(word, genuine)
    model = _checkpoint_model(f.cs)
    saved = model._collect_structural_extras()
    names = deepcopy(f.cs._concept_allocator.named_identities)
    model._restore_structural_extras(saved)
    assert f.cs._concept_allocator.named_identities == names
    assert f.cs.definitions.objects(word) == (genuine,)
    assert f.store.parts.find(first, word) is not None
