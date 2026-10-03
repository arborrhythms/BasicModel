"""A relative S can refer to numerical operands with no existing concept ID.

Forced readings cover the unaligned embedding path used by MM_grammar;
they make no claim about a learned parsing or a selected random seed.
"""
from dataclasses import replace

import pytest
import torch

from reading_fixtures import finish_reading
from test_clause_acceptance import SentenceFixture


@pytest.mark.parametrize('unknown', [(0,), (1,), (0, 1)])
@pytest.mark.parametrize('nested', [False, True])
def test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent(
        monkeypatch, unknown, nested):
    f = SentenceFixture(monkeypatch)
    tree = ('part', 'cat', 'animal')
    if nested:
        tree = ('part', tree, 'living')
    entry = f.program(tree)
    ids = entry.concept_ids.clone()
    ids[list(unknown)] = -1
    entry = replace(entry, concept_ids=ids)
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause, trust=.7)
    relation = clause.children[0] if nested else clause
    assert len(relation.children) == len(unknown)
    for leaf, child in zip(unknown, relation.children):
        assert child.relation is None
        torch.testing.assert_close(child.point, entry.leaves[leaf])
    assert len(f.store) == len(unknown) + 1 + int(nested)
    assert (f.store.refs[f.store.relations()] > 0).all()
    assert f.store.trust[:row].tolist() == [0.] * row
    assert float(f.store.trust[row]) == pytest.approx(.7)
    for index in f.store.ideas().tolist():
        assert f.store.row(index)['kind'] == 'unverified'


def test_three_slot_relation_closes_unindexed_numerical_operands(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('part', 'cat', 'animal'))
    predicate = f.registry.clause_reference('part')
    leaves = torch.stack((entry.leaves[0], f.registry._payload(predicate), entry.leaves[1]))
    entry = replace(entry, leaves=leaves, rows=torch.full((3,), -1),
        word_rows=torch.full((3,), -1), word_ids=torch.full((3,), -1), activations=torch.ones(3),
        concept_ids=torch.tensor([-1, predicate[1], -1]),
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [0, -1, 2]]),
        symbol_where=None, symbol_when=None, lexical_forms=(None,) * 3)
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause, trust=.6)
    assert row == 2 and f.store.rel_type[:3].tolist() == [0, 0, 1]
    assert f.store.refs[row, [0, 2]].tolist() == f.store.row_ids[:2].tolist()
    torch.testing.assert_close(f.store.slots[:2, 0], leaves[[0, 2]])


@pytest.mark.usefixtures('eager_reading')
def test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows(monkeypatch):
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    model, _, _ = _fresh_model(str(Path(_PROJECT)/'data/MM_grammar.xml'))
    chooser = model.languageSpace._tree_layer(2).chooser
    part = next(i for i, r in enumerate(model.languageSpace._compose_binary_rules)
                if r.method_name == 'part')
    score_binary, score_unary = chooser.score_binary, chooser.score_unary
    def binary(*args, **kwargs):
        stop, scores = score_binary(*args, **kwargs)
        mask = torch.arange(scores.shape[-1], device=scores.device) != part
        return stop, scores.masked_fill(mask, -torch.inf) + 1e6
    def unary(*args, **kwargs):
        stop, scores = score_unary(*args, **kwargs)
        return stop, torch.full_like(scores, -torch.inf)
    monkeypatch.setattr(chooser, 'score_binary', binary)
    monkeypatch.setattr(chooser, 'score_unary', unary)
    try:
        inputs, _ = next(iter(model.inputSpace.data.data_loader(split='train', num_streams=4)))
        result = model.forward(model.inputSpace.prepInput(inputs))
        store = model.symbolSpace.ltm_store
        assert len(result) == 4
        relations = store.relations()
        clauses = relations[store.rel_type[relations] != store.REL_DEF]
        assert clauses.numel() > 0
        assert (store.refs[clauses] > 0).all()
        definitions = store.relations(store.REL_DEF)
        assert definitions.numel() > 0
        assert (store.refs[definitions][:, (0, 2)] > 0).all()
        assert model._word_symbol_concept_ids() is None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
