"""A thought operator's declared limits and payload contracts hold on reads."""
from Memory import STM0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from Memory import LTM0
from Meaning import ConceptualMeaning
from Queries import THOUGHT_EXECUTORS, ThoughtExecutorDescriptor
from test_cs_symbol_table import _cs
from test_query_vp_boundaries import _context, _signature


def test_explicit_addresses_bound_decoding_without_establishing_content_match(monkeypatch):
    store = STM0(8, percept_width=0, capacity=1024, relation_capacity=8192)
    meaning = ConceptualMeaning.from_description(torch.ones(8))
    for strength in (0.2, 0.8):
        store.append_meaning(meaning, trust=strength)
    rows = []
    original = store.row
    def read(index):
        rows.append(index)
        return original(index)
    monkeypatch.setattr(store, 'row', read)
    references = tuple(store.occurrence_of(i) for i in (1, 0))
    one = store.cued_rows(meaning, references=references, max_records=1)
    assert rows == [1] and one['records_scanned'] == 1
    assert len(one['value']) == 1 and not one['incomplete']
    rows.clear()
    both = store.cued_rows(meaning, references=references, max_records=2)
    assert rows == [1, 0] and both['records_scanned'] == 2
    assert both['value'][1]['trust'] == pytest.approx(.2)
    assert all(row['match'] == 0 for row in both['value'])
    # An address grants a read; without an entry it establishes no field match.
    context = replace(_context(_cs(), store=store, max_records=1),
        operand_references=tuple(('argin1', reference) for reference in references))
    assert _signature('query', 'argin1').invoke(context, meaning)['value'] == ()


def test_zero_budget_does_not_fetch_a_named_concept_payload(monkeypatch):
    from QueryWork import QueryWorkBudget
    cs=_cs()
    concept=cs.mint_frozen_concept('zero-budget-native-read')
    monkeypatch.setattr(cs.similarity_codebook,'active_prototypes',lambda:pytest.fail('zero budget read'))
    result=_signature('exist','argin1').invoke(_context(cs,work=QueryWorkBudget(0)),('sym',concept))
    assert result['value'] is None and result['support_true']==result['support_false']==0.
    assert 'work_budget' in result['incomplete']


def test_nonfinite_native_concept_atom_fails_instead_of_becoming_identity_evidence():
    cs = _cs()
    concept = cs.mint_frozen_concept('invalid-payload-native-read')
    with torch.no_grad():
        cs.similarity_codebook.getW()[cs._csw_row_of(concept)].fill_(float('nan'))
    context = _context(cs)
    with pytest.raises(FloatingPointError, match='finite'):
        _signature('equal', 'argin1', 'argin2').invoke(
            context, ('sym', concept), ('sym', concept))


def test_addressed_end_read_returns_complete_content_without_another_occurrence():
    store = STM0(8, percept_width=0, capacity=1024, relation_capacity=8192)
    meaning = ConceptualMeaning.from_description(torch.eye(8)[0])
    row = store.append_meaning(meaning, trust=.9)
    context = replace(_context(_cs(), store=store),
        operand_references=(('argin1', store.occurrence_of(row)),))
    found = _signature('isTrue', 'argin1').invoke(context, replace(meaning))
    torch.testing.assert_close(found['meaning'].roles, meaning.roles)
    assert found['occurrence'] == store.occurrence_of(row)
    assert found['support_true'] == pytest.approx(.9)
    assert found['support_false'] == 0
    assert len(store) == 1
    # The closed content question has no stored extension to read here.
    queried = _signature('query', 'argin1').invoke(context, replace(meaning))
    assert queried['value'] == () and queried['support_true'] == queried['support_false'] == 0


@pytest.mark.parametrize('declarations', [17, ['isPart(X,Y)', 'isPart(A,B)'], [None], ['isPart(X,,Y)']])
def test_retired_query_catalogue_is_rejected_without_mutating_grammar(declarations):
    grammar = Grammar()
    grammar.configure({'Symbolic': {'compose': {
        'rule': ['part_argout1 = part.forward(part_argin1, part_argin2)']}}})
    original_rules = tuple(grammar.rules)
    original_operations = grammar.thought_operations
    with pytest.raises(ValueError, match='Queries|retired'):
        grammar.configure({'Queries': {'query': declarations}})
    assert tuple(grammar.rules) == original_rules
    assert grammar.thought_operations == original_operations


def test_thought_descriptors_cannot_advertise_missing_executors_or_ambiguous_roles():
    part = THOUGHT_EXECUTORS['part']
    with pytest.raises(ValueError, match='executor'):
        replace(part, executor=None)
    with pytest.raises(ValueError, match='argument|unsupported'):
        ThoughtExecutorDescriptor(
            'part', part.domain, ('reference', 'unknown'), part.result_kind,
            part.read_scope, part.write_scope, part.evidence_kind, part.executor)
