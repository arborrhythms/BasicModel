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
    words = entry.word_ids.clone()
    words[list(unknown)] = -1  # No fallback lexical identity for numerical operands.
    entry = replace(entry, concept_ids=ids, word_ids=words)
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause, trust=.7)
    relation = clause.children[0] if nested else clause
    assert len(relation.children) == len(unknown)
    for leaf, child in zip(unknown, relation.children):
        assert child.relation is None
        torch.testing.assert_close(child.point, entry.leaves[leaf])
    from LTM1 import OCCURRENCE
    from ChainRead import point
    assert len(f.store) == len(unknown)
    assert isinstance(row, tuple) and f.store.parts.contains(row[1])
    assert not f.store.trust[:len(unknown)].any()
    assert f.store.row(row)['evidence'] == pytest.approx((.7, 0.))
    operand_rows = f.store.ideas().tolist()
    for index, leaf in zip(operand_rows, unknown):
        assert f.store.row(index)['kind'] == 'unverified'
        torch.testing.assert_close(f.store.slots[index], entry.leaves[leaf])
    assert len(operand_rows) == len(unknown)
    # The sentence itself is a tagged chain, not a duplicate LTM0 payload.
    assert f.store.parts.truth_plus[row[1]] == pytest.approx(.7)


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
    from ChainRead import point
    from LTM1 import OCCURRENCE
    assert len(f.store) == 2 and isinstance(row, tuple)
    assert f.store.parts.ref_kinds[row[1]].tolist() == [OCCURRENCE, OCCURRENCE]
    for reference, expected in zip(f.store.parts.refs[row[1]].tolist(), leaves[[0, 2]]):
        torch.testing.assert_close(point(f.store, (OCCURRENCE, reference)), expected)
    torch.testing.assert_close(f.store.slots[:2], leaves[[0, 2]])


@pytest.mark.usefixtures('eager_reading')
def test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows(monkeypatch):
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    from configuration_fixtures import small_retained
    # This certificate closes four distinct binary derivations, whose
    # arguments now require native seats in addition to the four words.
    with small_retained('MM_grammar.xml', concept_rows=64) as path:
        model, _, _ = _fresh_model(path)
    constructed_vectors = model._concept_owner().nVectors
    chooser = model.languageSpace._tree_layer(2).chooser
    part = next(i for i, r in enumerate(model.languageSpace._compose_binary_rules)
                if r.method_name == 'part')
    score_binary, score_unary = chooser.score_binary, chooser.score_unary
    def binary(*args, **kwargs):
        stop, scores = score_binary(*args, **kwargs)
        mask = kwargs.get('op_indices', torch.arange(scores.shape[-1], device=scores.device)) != part
        return stop, scores.masked_fill(mask, -torch.inf) + 1e6
    def unary(*args, **kwargs):
        stop, scores = score_unary(*args, **kwargs)
        if kwargs.get('op_offset') == model._stm_reducer().r_reduce + model._stm_reducer().r_apply:
            return stop, scores
        indices = kwargs.get('op_indices')
        forced = torch.full_like(scores, -torch.inf)
        if indices is not None:
            forced = torch.where(indices >= model._stm_reducer().r_reduce + model._stm_reducer().r_apply,
                                 scores, forced)
        return stop, forced
    monkeypatch.setattr(chooser, 'score_binary', binary)
    monkeypatch.setattr(chooser, 'score_unary', unary)
    try:
        inputs, _ = next(iter(model.inputSpace.data.data_loader(split='train', num_streams=4)))
        result = model.forward(model.inputSpace.prepInput(inputs))
        store = model.symbolSpace.ltm_store
        assert len(result) == 4
        relations = store.relations()
        definitions = store.relations('definition')
        clauses = relations[~torch.isin(relations, definitions)]
        assert clauses.numel() > 0
        assert (store.parts.refs[clauses] > 0).all()
        # No native word rows were presented. Closing the numerical
        # arguments cannot fabricate lexical definitions for them.
        assert definitions.numel() == 0
        assert model._word_symbol_concept_ids() is None
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.usefixtures('eager_reading')
def test_mixing_relation_keeps_admitted_object_ids_without_duplicate_seats(monkeypatch):
    from pathlib import Path
    from test_mm_xor import _fresh_model, _PROJECT
    from PartClosing import PartClosing
    from reading_fixtures import capture_readings
    model, _, _ = _fresh_model(str(Path(_PROJECT)/'data/MM_grammar.xml'))
    constructed_vectors = model._concept_owner().nVectors
    chooser = model.languageSpace._tree_layer(2).chooser
    part = next(i for i, rule in enumerate(model.languageSpace._compose_binary_rules)
                if rule.method_name == 'part')
    score_binary, score_unary = chooser.score_binary, chooser.score_unary
    def binary(*args, **kwargs):
        stop, scores = score_binary(*args, **kwargs)
        ids = kwargs.get('op_indices', torch.arange(scores.shape[-1], device=scores.device))
        return stop, scores.masked_fill(ids != part, -torch.inf) + 1e6
    def unary(*args, **kwargs):
        stop, scores = score_unary(*args, **kwargs)
        if kwargs.get('op_offset') == model._stm_reducer().r_reduce + model._stm_reducer().r_apply:
            return stop, scores
        indices = kwargs.get('op_indices')
        forced = torch.full_like(scores, -torch.inf)
        if indices is not None:
            forced = torch.where(indices >= model._stm_reducer().r_reduce + model._stm_reducer().r_apply,
                                 scores, forced)
        return stop, forced
    monkeypatch.setattr(chooser, 'score_binary', binary)
    monkeypatch.setattr(chooser, 'score_unary', unary)
    preflight, plans = PartClosing.preflight, []
    def observe(closing):
        plans.append([record['reference'] for record in closing._planned.values()
                      if not record['children']])
        return preflight(closing)
    monkeypatch.setattr(PartClosing, 'preflight', observe)
    close = PartClosing.close
    seat_counts = []
    def observe_seats(closing, *args, **kwargs):
        before = sum(len(layer._tensor_rows) for layer in closing.alloc._layers.values())
        result = close(closing, *args, **kwargs)
        seat_counts.append((before, sum(len(layer._tensor_rows) for layer in closing.alloc._layers.values())))
        return result
    monkeypatch.setattr(PartClosing, 'close', observe_seats)
    try:
        inputs, _ = next(iter(model.inputSpace.data.data_loader(split='train', num_streams=4)))
        with capture_readings(model) as captured:
            model.forward(model.inputSpace.prepInput(inputs))
        assert model._concept_owner().nVectors == constructed_vectors
        assert model._word_symbol_concept_ids() is None
        source = model.inputSpace
        valid = source._ar_grammar_leaf_mask
        for b, program in enumerate(captured[0]):
            expected = source._ar_grammar_object_ids[b, valid[b]]
            torch.testing.assert_close(program.concept_ids, expected, rtol=0, atol=0)
            assert bool((expected > 0).all())
        from LTM1 import SYMBOL
        assert plans and all(ref is not None and ref[0] == SYMBOL for plan in plans for ref in plan)
        assert seat_counts and all(before == after for before, after in seat_counts)
        assert not model.symbolSpace.ltm_store.relations('definition').numel()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
