"""An open reading yields independent clause fields and encoded coordinates."""
from dataclasses import replace

import torch
import pytest

from reading_fixtures import finish_reading
from Spaces import WhereEncoding, WhenStartDurationEncoding
from test_selected_nested_meaning import _nested


def test_nested_clause_retains_its_own_field_without_program_or_targets(monkeypatch):
    _, registry, language, entry, _ = _nested(monkeypatch)
    clause = finish_reading(language, entry,
        meaning=language.program_meaning(entry, registry), registry=registry)
    child, = clause.children
    assert not hasattr(child, 'derivation')
    torch.testing.assert_close(child.meaning.roles[0], entry.leaves[0])
    torch.testing.assert_close(child.meaning.roles[2], entry.leaves[1])
    assert child.refs[0] == int(entry.concept_ids[0])
    assert child.refs[2] == int(entry.concept_ids[1])


def test_closing_copies_encoded_field_bands_without_coordinatewise_pooling(monkeypatch):
    _, registry, language, entry, _ = _nested(monkeypatch)
    where = WhereEncoding(nWhere=4).set_capacity(1024)
    when = WhenStartDurationEncoding(n_when=4).set_capacity(128)
    stamps = where.encode(torch.tensor([3, 512, 923]))
    times = when.encode(torch.tensor([27, 27, 27]))
    entry = replace(entry, symbol_where=stamps, symbol_when=times)
    clause = finish_reading(language, entry,
        meaning=language.program_meaning(entry, registry), registry=registry)
    for value in (clause, *clause.children):
        assert int(where.decode_index(value.where)) == 3
        assert int(when.decode_index(value.when)) == 27
        torch.testing.assert_close(value.where, stamps[0])
        torch.testing.assert_close(value.when, times[0])


def test_selected_clause_tree_owns_its_local_references(monkeypatch):
    _, registry, language, entry, _ = _nested(monkeypatch)
    meaning = language.program_meaning(entry, registry)
    # The semantic projector's constituent inventory is a separate namespace.
    # It must not replace the selected clause tree's local child address.
    meaning = replace(meaning, role_refs=(meaning.role_refs[0], meaning.role_refs[1],
                                         ('constituent', 7)))
    clause = finish_reading(language, entry, meaning=meaning, registry=registry)
    assert clause.refs[2] == ('clause', 0)
    assert len(clause.children) == 1


def test_bare_unlocated_np_reuses_its_concept_identity(monkeypatch):
    from Layers import TernaryTruthStore
    _, registry, language, entry, _ = _nested(monkeypatch)
    entry = replace(entry, rows=entry.rows[:1], word_rows=entry.word_rows[:1],
                    activations=entry.activations[:1], leaves=entry.leaves[:1],
                    concept_ids=torch.tensor([1]), lexical_forms=(None,),
                    actions=torch.tensor([[0, -1, 0]]), targets=torch.tensor([-1]),
                    end_state=torch.cat((entry.leaves[:1], torch.zeros_like(entry.end_state[1:]))))
    clause = finish_reading(language, entry, registry=registry)
    store = TernaryTruthStore(entry.leaves.shape[-1], capacity=4)
    store.configure_clause_index(allocate=lambda _point: pytest.fail('eternal NP minted an identity'),
                                 concept_point=lambda cid: entry.leaves[0] if cid == 1 else None)
    row = store.write_clause(clause)
    assert int(store.row_ids[row]) == 1
    assert store.write_clause(clause) == row
    assert len(store) == 1


def test_three_slot_closing_registers_its_composite_operand(monkeypatch):
    from Meaning import ConceptualMeaning
    from Layers import TernaryTruthStore
    _, registry, language, entry, _ = _nested(monkeypatch)
    # A relative S may finish as a three-slot forest. Its left operand is a
    # completed inner relation and therefore has no fused-point reference.
    operation, _ = registry.operation_form('part')
    predicate = registry._reference((registry.descriptors['part'].domain, operation.semantic_id))
    leaves = torch.cat((entry.leaves[:2], registry._payload(predicate)[None], entry.leaves[:1]))
    actions = torch.cat((entry.actions[:3], torch.tensor([[0, -1, 2], [0, -1, 3]])))
    meaning = ConceptualMeaning(torch.stack((torch.zeros_like(leaves[0]), leaves[2], leaves[3])),
        torch.ones(3, dtype=torch.bool), role_refs=(('constituent', 8), predicate,
                                                  ('sym', int(entry.concept_ids[0]))))
    entry = replace(entry, rows=torch.arange(4), word_rows=torch.arange(4),
        activations=torch.ones(4), leaves=leaves, actions=actions,
        concept_ids=torch.tensor([*entry.concept_ids[:2].tolist(), predicate[1], int(entry.concept_ids[0])]),
        lexical_forms=(None,) * 4, end_state=meaning.roles)
    clause = finish_reading(language, entry, meaning=meaning, depth=3, registry=registry)
    assert clause.refs[0] == ('clause', 0)
    assert len(clause.children) == 1 and clause.children[0].relation == 'part'
    store = TernaryTruthStore(leaves.shape[-1], capacity=8)
    identities = iter(range(1000, 1008))
    store.configure_clause_index(allocate=lambda _: next(identities),
        concept_point=lambda cid: registry._payload(('sym', cid)))
    row = store.write_clause(clause)
    assert int(store.refs[row, 0]) == int(store.row_ids[0])
    assert not bool(store.slots[row, 0].any())
