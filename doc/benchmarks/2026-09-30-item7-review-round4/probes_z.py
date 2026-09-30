"""Round-4 Z / first AG probes: forced readings, no selected seed."""
import copy
from types import SimpleNamespace
import pytest
import torch

from reading_fixtures import finish_reading
from Spaces import _concept_alloc_of
from test_item7_acceptance import SentenceFixture


def inventory(cs):
    alloc = _concept_alloc_of(cs)
    return dict(next_id=alloc.next_id, placement=copy.deepcopy(alloc.placement),
                rows=copy.deepcopy(cs._csw_rows),
                keys=copy.deepcopy(alloc.relate_idx))


def test_closing_identity_is_its_ltm_occurrence_without_an_inventory_record(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', 'cat', 'runs'))
    before = inventory(f.cs)
    row = f.store.write_clause(clause)
    assert row >= 0
    assert inventory(f.cs) == before
    assert int(f.store.row_ids[row]) not in _concept_alloc_of(f.cs).placement
    torch.testing.assert_close(f.store.point_of_row(int(f.store.row_ids[row])), clause.point)


def test_referenced_phrase_is_read_from_ltm_without_an_inventory_row(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cat', ('verb', 'chases', 'mouse')))
    before = inventory(f.cs)
    row = f.store.write_clause(clause)
    assert row >= 0
    assert inventory(f.cs) == before
    reference = int(f.store.refs[row, 2])
    phrase = f.store.index_of_row(reference)
    assert phrase is not None and phrase != row
    assert f.store.rel_type[phrase] == f.store.REL_NONE
    torch.testing.assert_close(f.store.point_of_row(reference), f.store.slots[phrase, 0])


def test_closing_part_and_implies_never_ask_thought_authorization(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('implies', ('part', 'cat', 'animal'), ('lift', 'cat', 'runs')))
    def unavailable(*args, **kwargs):
        raise AssertionError('closing asked the thought registry for its relation identity')
    monkeypatch.setattr(f.registry, 'clause_reference', unavailable)
    before = dict(f.cs._csw_rows)
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause)
    assert f.store.rel_type[row] == f.store.REL_IMPLIES
    assert f.store.relations(f.store.REL_PARTOF).numel() == 1
    # A testified taxonomy may symbolize a parent when room exists; the
    # predicates themselves must not occupy a newly allocated native row.
    for index in f.store.relations().tolist():
        assert f.cs._csw_row_of(int(f.store.refs[index, 1])) is None


def test_mm_grammar_part_under_disjunction_closes_at_full_inventory(monkeypatch):
    import Language
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    model, _, _ = _fresh_model(str(Path(_PROJECT) / 'data/MM_grammar.xml'))
    f = object.__new__(SentenceFixture)
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
        row = f.store.write_clause(clause)
        assert row >= 0 and f.store.rel_type[row] == f.store.REL_OPERATOR
        assert inventory(f.cs) == before
        assert (f.store.refs[row] > 0).all()
        predicate = int(f.store.refs[row, 1])
        assert f.cs._csw_row_of(predicate) is None
        assert predicate not in _concept_alloc_of(f.cs).placement
        second = f.store.write_clause(clause)
        assert second == row
        assert inventory(f.cs) == before
    finally:
        model.End()
        model.symbolSpace.soft_reset()
