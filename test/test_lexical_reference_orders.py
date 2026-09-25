"""Selected grammar roles resolve one form across persistent concept orders."""
from types import SimpleNamespace

import pytest
import torch

import Language
import Spaces
from Models import BasicModel
from test_cs_sparse_weights import _cs


def vocabulary():
    cs = _cs(nS=256, order=3)
    ids = []
    for order in range(3):
        cid = cs.new_concept()
        if ids:
            cs.add_part(cid, ('sym', ids[-1]))
        cs._csw_concept_row(order, cid)
        cs.bind_word_concept('cat', cid)
        ids.append(cid)
    row = cs._csw_row_of(ids[0])
    cs.remember_word_surface(row, b'cat', object_row=row, object_id=ids[0])
    return cs, ids, row


@pytest.mark.parametrize('context, order', [('event', 0), ('particular', 1),
    ('name', 1), ('pronoun', 1), ('kind', 2)])
def test_selected_grammar_context_selects_order(context, order):
    cs, ids, row = vocabulary()
    grammar = Language.Grammar()
    grammar.configure({'compose': {'rule': {
        '_': 'lower_O1 = lower.forward(lower_I1, lower_I2)',
        'reference': f'I2:{context}'}}})
    rules = [r for r in grammar.rules if r.method_name == 'lower']
    language = SimpleNamespace(_compose_binary_rules=rules, _compose_unary_rules=[])
    actions = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, -1]])
    got, orders = Language.LanguageSpace.resolve_lexical_references(
        language, cs, torch.tensor([-1, row]), torch.tensor([-1, ids[0]]), actions)
    assert got.tolist() == [-1, ids[order]]
    assert orders.tolist() == [-1, order]
    assert cs.word_concepts('cat') == tuple(ids)


def test_capture_owns_grammar_resolution_without_changing_reconstruction():
    cs, ids, row = vocabulary()
    rule = SimpleNamespace(reference_orders=(('I2', 1),))
    language = SimpleNamespace(_compose_binary_rules=[rule], _compose_unary_rules=[])
    language.resolve_lexical_references = lambda *args: Language.LanguageSpace.resolve_lexical_references(language, *args)
    host = SimpleNamespace(languageSpace=language, _concept_owner=lambda: cs)
    leaves = torch.randn(1, 2, 8)
    actions = torch.tensor([[[0, 0, 0], [0, 0, 1], [1, 0, -1]]])
    entry, = BasicModel._program_entries(host,
        (torch.tensor([[0, 1]]), actions, torch.zeros(1, 3, 8)),
        leaves, torch.tensor([[-1, row]]), torch.tensor([[-1, row]]),
        torch.ones(1, 2), torch.zeros(1, 3, 8), concept_ids=torch.tensor([[-1, ids[0]]]))
    assert entry.reference_ids.tolist() == [-1, ids[1]]
    assert entry.reference_orders.tolist() == [-1, 1]
    torch.testing.assert_close(entry.leaves, leaves[0])
    assert entry.concept_ids.tolist() == [-1, ids[0]]
    assert entry.detached().reference_ids.tolist() == [-1, ids[1]]


def test_form_sets_survive_checkpoint_and_missing_order_stays_unknown():
    from test_structural_checkpoint import _model_with
    cs, ids, row = vocabulary()
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs(nS=256, order=3)
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    assert restored.word_concepts('cat') == tuple(ids)
    assert restored.resolve_word_concept('cat', order=3) is None
    assert restored.resolve_word_concept('cat', order=1) == ids[1]


def test_resolved_reference_supplies_meaning_but_keeps_input_provenance(monkeypatch):
    from dataclasses import replace
    from test_selected_relation_meaning import _program_owner
    cs, grammar, registry, language, leaves, program, a, b = _program_owner(monkeypatch)
    original = program()
    resolved = replace(original, reference_ids=torch.tensor([b[1], b[1]]),
                       reference_orders=torch.tensor([1, 1]))
    meaning = language.program_meaning(resolved, registry)
    assert meaning.role_refs[0] == b
    torch.testing.assert_close(meaning.roles[0], registry._payload(b))
    torch.testing.assert_close(resolved.leaves, original.leaves)
    torch.testing.assert_close(resolved.concept_ids, original.concept_ids)


def test_ambiguous_particular_requires_a_carried_reference():
    cs, ids, row = vocabulary()
    other = cs.new_concept()
    cs.add_part(other, ('sym', ids[0]))
    cs._csw_concept_row(1, other)
    cs.bind_word_concept('cat', other)
    assert cs.resolve_word_concept('cat', order=1) is None
    assert cs.resolve_word_concept('cat', order=1, previous=other) == other
