"""Round-4 Z / first AG probes: forced readings, no selected seed."""
import copy
from types import SimpleNamespace
import pytest
import torch

from reading_fixtures import finish_reading
from Spaces import _concept_alloc_of
from test_clause_acceptance import SentenceFixture


def inventory(cs):
    alloc = _concept_alloc_of(cs)
    return dict(next_id=alloc.next_id, placement=copy.deepcopy(alloc.placement),
                rows=copy.deepcopy(cs._csw_rows),
                keys=copy.deepcopy(alloc.relate_idx))


def test_closing_idea_is_narrow_and_the_operation_has_a_native_part(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', 'cat', 'runs'))
    before = inventory(f.cs)
    row = f.store.write_clause(clause)
    assert row >= 0
    assert inventory(f.cs)['next_id'] == before['next_id'] + 1
    assert f.store.parts.find(f.noun('cat'), f.noun('runs')) is not None
    assert int(f.store.row_ids[row]) not in _concept_alloc_of(f.cs).placement
    torch.testing.assert_close(f.store.point_of_row(int(f.store.row_ids[row])), clause.point)


def test_referenced_phrase_is_minted_as_a_part_argument(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cat', ('lift', 'chases', 'mouse')))
    row = f.store.write_clause(clause)
    reference = int(f.store.row(row)['refs'][1])
    assert f.cs._csw_row_of(reference) is not None
    assert f.store.parts.refs[reference].tolist() == [f.noun('chases'), f.noun('mouse')]
    torch.testing.assert_close(f.store._concept_point(reference), clause.children[0].point)
    assert len(f.store) == 1  # the narrowed constituent occurrence also survives


def test_closing_part_and_implies_never_ask_thought_authorization(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('implies', ('part', 'cat', 'animal'), ('lift', 'cat', 'runs')))
    def unavailable(*args, **kwargs):
        raise AssertionError('closing asked the thought registry for its relation identity')
    monkeypatch.setattr(f.registry, 'clause_reference', unavailable)
    before = dict(f.cs._csw_rows)
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause)
    assert f.store.row(row)['readings'] == ('conditional',)
    assert f.store.parts.find(f.noun('cat'), f.noun('animal')) is not None
    assert f.store.parts.refs.shape[-1] == 2
    assert all('predicate' not in f.cs._concept_allocator.symbol_types.get(cid, ())
               for cid in set(f.cs._csw_rows) - set(before))


def test_mm_grammar_full_inventory_refuses_the_whole_closing_atomically(monkeypatch):
    import Language
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    model, _, _ = _fresh_model(str(Path(_PROJECT) / 'data/MM_grammar.xml'))
    f = object.__new__(SentenceFixture)
    f.native_cube = False
    f.cs, f.grammar, f.registry = model._concept_owner(), Language.TheGrammar, model.grammatical_thoughts
    f.language = model.languageSpace
    f.binary, f.unary = f.language._compose_binary_rules, f.language._compose_unary_rules
    f.words, f.ops = {}, {}
    width = int(f.cs.outputShape[-1])
    def operation(rule):
        name = rule.method_name
        if name not in f.ops:
            cls = Language.GRAMMAR_LAYER_CLASSES[name]
            f.ops[name] = cls(width, width) if name in ('lift', 'verb', 'lower', 'surface', 'sum', 'implies') else cls()
        return f.ops[name]
    f.operation = operation
    f.store = model.symbolSpace.ltm_store
    from ClauseRow import attach_clause_index
    attach_clause_index(model, f.store, f.cs)
    try:
        entry = f.program(('disjunction', ('part', 'cat', 'animal'), 'sleeps'))
        layer = _concept_alloc_of(f.cs).layer(0)
        caps = f.cs._order_caps()
        # Occupy the remaining physical seats in the mechanism fixture;
        # configuration capacities are unchanged.
        for base, capacity in [(0, caps[0]), (sum(caps), f.cs.nVectors - sum(caps))] + [(sum(caps[:i]), caps[i]) for i in range(1, len(caps))]:
            layer._row_next[base] = capacity
        before = inventory(f.cs)
        clause = finish_reading(f.language, entry, registry=f.registry)
        rows, edges = len(f.store), len(f.store.parts)
        with pytest.raises(RuntimeError, match='concept row pool'):
            f.store.write_clause(clause)
        assert inventory(f.cs) == before
        assert len(f.store) == rows and len(f.store.parts) == edges
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_full_inventory_can_record_part_of_already_seated_arguments(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cat', 'animal'))
    alloc = _concept_alloc_of(f.cs)
    caps = f.cs._order_caps()
    for rung in range(len(caps)):
        alloc.layer(0)._row_next[sum(caps[:rung])] = caps[rung]
    alloc.layer(0)._row_next[sum(caps)] = f.cs.nVectors - sum(caps)
    before = inventory(f.cs)
    row = f.store.write_clause(clause)
    assert f.store.row(row)['refs'].tolist() == [f.noun('cat'), f.noun('animal')]
    after = inventory(f.cs)
    assert after['rows'] == before['rows']  # no new physical seat is needed
    assert after['next_id'] == before['next_id'] + 1  # the native edge handle
