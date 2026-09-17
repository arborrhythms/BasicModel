"""A query's declared limits and payload contracts hold on actual reads."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import BUILTIN_QUERIES, QueryContext
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def test_exist_executor_limits_native_fact_reads_and_reports_incompleteness(monkeypatch):
    store = TernaryTruthStore(4)
    meaning = ConceptualMeaning.from_description(torch.ones(4))
    for strength in (0.2, 0.8):
        store.append_meaning(meaning, trust=strength)
    rows = []
    original = store.row
    def read(index):
        rows.append(index)
        return original(index)
    monkeypatch.setattr(store, 'row', read)
    result = BUILTIN_QUERIES['isTrue'].invoke(
        QueryContext(TruthGroundedReasoner(store=store), max_records=1), meaning)
    assert rows == [0]
    assert result['support_true'] == pytest.approx(0.2)
    assert result['records_scanned'] == 1
    assert 'capture_limit' in result['incomplete']


def test_zero_quantize_budget_does_not_fetch_a_named_concept_payload(monkeypatch):
    cs = _cs()
    concept = cs.mint_frozen_concept('zero-budget-native-read')
    monkeypatch.setattr(cs.similarity_codebook, 'active_prototypes',
                        lambda: pytest.fail('zero query budget read the dictionary'))
    context = QueryContext(TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs)), max_nodes=0)
    result = BUILTIN_QUERIES['quantize'].invoke(context, ('sym', concept))
    assert result['value'] is None
    assert result['nodes_scanned'] == 0
    assert result['incomplete'] == ('capture_limit',)


def test_nonfinite_native_concept_atom_fails_instead_of_becoming_identity_evidence():
    cs = _cs()
    concept = cs.mint_frozen_concept('invalid-payload-native-read')
    with torch.no_grad():
        cs.similarity_codebook.getW()[cs._csw_row_of(concept)].fill_(float('nan'))
    context = QueryContext(TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs)))
    with pytest.raises(FloatingPointError, match='finite'):
        BUILTIN_QUERIES['isEqual'].invoke(context, ('sym', concept), ('sym', concept))


def test_distinct_lookup_returns_complete_record_and_does_not_admit_observation():
    store = TernaryTruthStore(4)
    roles = torch.eye(4)[:3]
    meaning = ConceptualMeaning.from_description(roles)
    row = store.append_meaning(meaning, kind='observation', trust=0.9)
    context = QueryContext(TruthGroundedReasoner(store=store))
    found = BUILTIN_QUERIES['query'].invoke(context, roles[0], roles[2])
    assert found['result_kind'] == 'set'
    assert found['evidence_kind'] == 'retrieval'
    assert len(found['value']) == 1
    torch.testing.assert_close(found['value'][0]['meaning'].roles, roles)
    assert found['value'][0]['occurrence'] == store.occurrence_of(row)
    assert BUILTIN_QUERIES['isTrue'].invoke(context, meaning)['support_true'] == 0
    assert len(store) == 1


@pytest.mark.parametrize('declarations', [17, ['isPart(X,Y)', 'isPart(A,B)'], [None], ['isPart(X,,Y)']])
def test_bad_query_reconfiguration_preserves_the_previous_grammar(declarations):
    grammar = Grammar()
    grammar.configure({'Symbolic': {'compose': {'rule': ['S = not.forward(S)']}},
                       'Queries': {'query': ['isTrue(X)']}})
    original_rules = tuple(grammar.rules)
    with pytest.raises(ValueError, match='query|signature'):
        grammar.configure({'Queries': {'query': declarations}})
    assert tuple(grammar.rules) == original_rules
    assert grammar.query_ops == ['isTrue(X)']


def test_query_contracts_cannot_advertise_nonexistent_executors_or_ambiguous_roles():
    part = BUILTIN_QUERIES['isPart']
    with pytest.raises(ValueError, match='executor'):
        replace(part, executor=None)
    with pytest.raises(ValueError, match='role'):
        replace(part, argument_roles=(0, 0))
