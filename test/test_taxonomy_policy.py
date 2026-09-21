"""Normal thought credit and evidence cannot regain world-vector shortcuts."""
import torch
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from reasoning import QuerySpec
from test_cs_symbol_table import _cs
from test_thought_model_fixture import model_for


def _setup():
    cs = _cs(nS=128)
    store = TernaryTruthStore(8, capacity=32)
    model = model_for(cs, store)
    refs = tuple(('sym', cs.synthesize_higher_order([('sym', cs.new_concept())]))
                 for _ in range(3))
    for ref in refs:
        cs._csw_concept_row(0, ref[1])
    return cs, refs, store, model


def test_numeric_world_evidence_cannot_establish_taxonomic_inclusion():
    cs, (a, b, _), store, model = _setup()
    store.append_meaning(ConceptualMeaning.from_description(torch.eye(8)[:3]), trust=1)
    result = model.reason_about(QuerySpec.from_surface('part', a, b))
    assert result.support_true == result.support_false == 0
    assert len(store) == 1


def test_unrelated_true_episode_cannot_establish_the_next_parent_relation():
    cs, (a, b, _), store, model = _setup()
    fact = model.grammatical_thoughts.form('part', a, b, mode='assertive')
    store.append_meaning(fact, trust=1)
    assert model.reason_about(QuerySpec.from_surface('exist', fact)).support_true == 1
    result = model.reason_about(QuerySpec.from_surface('part', a, b))
    assert result.support_true == 0


def test_nested_taxonomy_result_keeps_each_record_source_and_premise_ablation():
    cs, (a, b, c), store, model = _setup()
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    result = model.reason_about(QuerySpec.from_surface('part', a, c))
    assert result.support_true == 1
    path = result.result.evidence['path']
    assert {(edge.part, edge.whole) for edge in path} == {(a, b), (b, c)}
    assert len(store) == 0
    cs.retire_concept(b[1])
    assert model.reason_about(QuerySpec.from_surface('part', a, c)).support_true == 0
