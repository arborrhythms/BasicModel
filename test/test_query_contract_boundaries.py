"""A thought operator's declared limits and payload contracts hold on reads."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import THOUGHT_EXECUTORS, ThoughtExecutorDescriptor
from test_cs_symbol_table import _cs
from test_query_vp_boundaries import _context, _signature


def test_exist_executor_limits_native_fact_reads_and_reports_incompleteness(monkeypatch):
    store = TernaryTruthStore(8)
    store.configure_leaf_index(unfold=lambda idea, limit, **kw: ((7,), 1, True))
    meaning = ConceptualMeaning.from_description(torch.ones(8))
    for strength in (0.2, 0.8):
        store.append_meaning(meaning, trust=strength)
    rows = []
    original = store.row
    def read(index):
        rows.append(index)
        return original(index)
    monkeypatch.setattr(store, 'row', read)
    result = _signature('exist', 'I1').invoke(
        _context(_cs(), store=store, max_records=1), meaning)
    assert rows == [0]
    assert result['support_true'] == pytest.approx(0.2)
    assert result['records_scanned'] == 1
    assert 'candidate_limit' in result['incomplete']


def test_zero_quantize_budget_does_not_fetch_a_named_concept_payload(monkeypatch):
    cs = _cs()
    concept = cs.mint_frozen_concept('zero-budget-native-read')
    monkeypatch.setattr(cs.similarity_codebook, 'active_prototypes',
                        lambda: pytest.fail('zero query budget read the dictionary'))
    context = _context(cs, max_nodes=0)
    result = _signature('quantize', 'I1').invoke(context, ('sym', concept))
    assert result['value'] is None
    assert result['nodes_scanned'] == 0
    assert result['incomplete'] == ('capture_limit',)


def test_nonfinite_native_concept_atom_fails_instead_of_becoming_identity_evidence():
    cs = _cs()
    concept = cs.mint_frozen_concept('invalid-payload-native-read')
    with torch.no_grad():
        cs.similarity_codebook.getW()[cs._csw_row_of(concept)].fill_(float('nan'))
    context = _context(cs)
    with pytest.raises(FloatingPointError, match='finite'):
        _signature('equal', 'I1', 'I2').invoke(
            context, ('sym', concept), ('sym', concept))


def test_distinct_lookup_returns_complete_record_and_does_not_admit_observation():
    store = TernaryTruthStore(8)
    roles = torch.eye(8)[:3]
    meaning = ConceptualMeaning.from_description(roles)
    row = store.append_meaning(meaning, kind='observation', trust=0.9)
    context = _context(_cs(), store=store)
    # Only a prior what may bring this row into serial thinking.
    assert not _signature('lookup', 'I1', 'I2').invoke(context, roles[0], roles[2])['value']
    context = _context(context.conceptual_space._ThoughtConceptualCapability__space, store=store,
                       memory=SimpleNamespace(retrieved_frames=lambda **kw: (store.row(row),)))
    found = _signature('lookup', 'I1', 'I2').invoke(context, roles[0], roles[2])
    assert found['result_kind'] == 'set'
    assert found['evidence_kind'] == 'retrieval'
    assert len(found['value']) == 1
    torch.testing.assert_close(found['value'][0]['meaning'].roles, roles)
    assert found['value'][0]['occurrence'] == store.occurrence_of(row)
    assert _signature('exist', 'I1').invoke(context, meaning)['support_true'] == 0
    assert len(store) == 1


@pytest.mark.parametrize('declarations', [17, ['isPart(X,Y)', 'isPart(A,B)'], [None], ['isPart(X,,Y)']])
def test_retired_query_catalogue_is_rejected_without_mutating_grammar(declarations):
    grammar = Grammar()
    grammar.configure({'Symbolic': {'compose': {
        'rule': ['part_O1 = part.forward(part_I1, part_I2)']}}})
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
