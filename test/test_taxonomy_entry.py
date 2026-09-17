"""Public PartOf routes must use the conceptual taxonomy exclusively."""
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner, NeuralToolUser, TRUE, UNKNOWN
from thinking import ThinkingKernel
from test_cs_symbol_table import _cs


def _world():
    cs = _cs()
    refs = tuple(("sym", cs.new_concept()) for _ in range(3))
    cs._ltm_store = TernaryTruthStore(8, capacity=16)
    return cs, refs, TruthGroundedReasoner(SimpleNamespace(conceptualSpace=cs),
                                          store=cs._ltm_store)


def test_public_evaluate_follows_taxonomy_and_retains_record_provenance():
    cs, (a, b, c), reasoner = _world()
    relation = cs.reify_concept(a[1], b[1])
    cs.add_whole(b[1], c)
    result = reasoner.evaluate(QuerySpec.from_surface("isPart", a, c))
    assert result["posture"] == TRUE
    assert result["domain"] == "conceptual-taxonomy"
    assert result["support_true"] == 1 and result["support_false"] == 0
    assert {edge.owner for edge in result["path"]} == {("sym", relation), b}
    assert result["edges_expanded"] == 3
    assert len(cs._ltm_store) == 0
    cs.retire_concept(relation)
    assert reasoner.evaluate(QuerySpec.from_surface("isPart", a, c))["posture"] == UNKNOWN


def test_whole_interface_reverses_to_the_same_partof_query():
    cs, (a, b, _), reasoner = _world()
    cs.add_whole(a[1], b)
    forward = reasoner.evaluate(QuerySpec.from_surface("part", a, b))
    converse = reasoner.evaluate(QuerySpec.from_surface("isWhole", b, a))
    assert forward["path"] == converse["path"]
    assert converse["posture"] == TRUE
    assert reasoner.evaluate(QuerySpec.from_surface("whole", a, b))["posture"] == UNKNOWN


def test_public_queries_reject_an_unsupported_relation_domain():
    _cs, (a, b, _), reasoner = _world()
    with pytest.raises(ValueError, match="domain"):
        reasoner.evaluate(QuerySpec.from_surface("isPart", a, b, domain="world-facts"))


def test_neural_entry_does_not_verify_generated_vectors_as_taxonomy():
    cs, (a, b, _), reasoner = _world()
    cs.add_whole(a[1], b)
    class BadGenerator:
        def propose(self, *args, **kwargs):
            raise AssertionError("unnamed vector candidates cannot prove taxonomy")
    tool = NeuralToolUser(reasoner, generator=BadGenerator(), ga=object(),
                          spaces=[object()], materialize=True)
    result = tool.run(QuerySpec.from_surface("isPart", a, b))
    assert result.posture == TRUE
    assert result.chain[0]["domain"] == "conceptual-taxonomy"
    assert len(cs._ltm_store) == 0
    unknown = tool.run(QuerySpec.from_surface("isPart", torch.eye(8)[0], torch.ones(8)))
    assert unknown.posture == UNKNOWN and unknown.support_true == 0


def test_legacy_kernel_lookup_and_traversal_use_the_same_taxonomy_sources():
    cs, (a, b, _), reasoner = _world()
    cs.add_whole(a[1], b)
    kernel = ThinkingKernel(reasoner)
    result = kernel.run(QuerySpec.from_surface("isPart", a, b))
    assert result.value == "true"
    assert result.provenance
    rels = kernel.part(a, mode="taxonomy", direction="up")
    assert len(rels) == 1 and rels[0]["reference"] == b
    assert rels[0]["source"].owner == a
    assert kernel.part(b, mode="taxonomy", direction="down")[0]["reference"] == a
    with pytest.raises(ValueError, match="domain|supported"):
        kernel.part(a, mode="meronomy")


def test_kernel_does_not_certify_a_world_row_as_taxonomy():
    cs, _refs, reasoner = _world()
    a, b = torch.eye(8)[:2]
    cs._ltm_store.append_relation(a, torch.eye(8)[2], b,
                                 rel_type=cs._ltm_store.REL_PARTOF, trust=1)
    result = ThinkingKernel(reasoner).run(QuerySpec.from_surface("isPart", a, b))
    assert result.value in ("unknown", "bounded_unknown")
    assert result.interval.upper == 0


def test_budget_cutoff_and_missing_identity_are_unknown_with_diagnostics():
    cs, (a, b, c), reasoner = _world()
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    result = reasoner.evaluate(QuerySpec.from_surface("isPart", a, c), max_steps=1)
    assert result["posture"] == UNKNOWN
    assert "traversal_limit" in result["incomplete"]
    result = reasoner.evaluate(QuerySpec.from_surface("isPart", torch.zeros(8), torch.zeros(8)))
    assert result["posture"] == UNKNOWN
    assert "unbound_concept_reference" in result["incomplete"]


def test_actual_model_entry_and_checkpoint_keep_taxonomy_identity(tmp_path):
    from test_ltm_consolidation import _make_model, _STATEFUL_CONFIG
    source = _make_model(_STATEFUL_CONFIG)
    restored = None
    try:
        source.reasoning_iterations = 8
        cs = source.conceptualSpace
        a, b = (("sym", cs.new_concept()) for _ in range(2))
        relation = cs.reify_concept(a[1], b[1])
        spec = QuerySpec.from_surface("isPart", a, b)
        before = source.reason_about(spec, spaces=[])
        assert before.posture == TRUE
        path = str(tmp_path / "taxonomy.pt")
        source.save_weights(path)
        restored = _make_model(_STATEFUL_CONFIG)
        restored.reasoning_iterations = 8
        assert restored.load_weights(path, strict=True, require_match=True)
        after = restored.reason_about(spec, spaces=[])
        assert after.posture == TRUE
        assert after.chain[0]["path"] == before.chain[0]["path"]
        restored.conceptualSpace.retire_concept(relation)
        assert restored.reason_about(spec, spaces=[]).posture == UNKNOWN
    finally:
        source.End()
        if restored is not None:
            restored.End()
