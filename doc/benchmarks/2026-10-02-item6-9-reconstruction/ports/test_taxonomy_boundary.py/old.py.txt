"""Public typed PartOf evaluation preserves polarity and avoids unrelated reads."""
from types import SimpleNamespace

import pytest

from reasoning import QuerySpec, TruthGroundedReasoner
from test_thought_model_fixture import model_for
from test_cs_symbol_table import _cs


def _chain():
    cs = _cs()
    a, b, c = (("sym", cs.new_concept()) for _ in range(3))
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    for ref in (a, b, c):
        cs._csw_concept_row(0, ref[1])
    return (a, b, c), TruthGroundedReasoner(model_for(cs))


def test_negated_partof_keeps_the_supported_proposition_and_evidence():
    (a, _b, c), reasoner = _chain()
    q = QuerySpec.from_surface("isPart", a, c, polarity=False)
    result = reasoner.evaluate(q)
    assert result["posture"] == "FALSE"
    assert result["support_true"] == 0 and result["support_false"] == 1
    assert result["path"][0].part == a and result["path"][-1].whole == c
    assert reasoner.evaluate(QuerySpec.from_surface("isPart", c, a, polarity=False))["posture"] == "UNKNOWN"
    assert reasoner.model.reason_about(q).posture == "FALSE"


def test_normal_result_retains_missing_reference_diagnostics():
    (a, _b, _c), reasoner = _chain()
    with pytest.raises(ValueError, match="unavailable|allocated|concept"):
        reasoner.model.reason_about(QuerySpec.from_surface("isPart", a, ("sym", 999999)))


def test_model_taxonomy_entries_skip_global_vector_proposal_setup(monkeypatch, tmp_path):
    from test_ltm_consolidation import _make_model, _STATEFUL_CONFIG
    from test_thought_model_fixture import thought_config
    model = _make_model(thought_config(tmp_path))
    try:
        model.reasoning_iterations = model.thinking_budget = 128
        cs = model.conceptualSpace
        a, b = (("sym", cs.new_concept()) for _ in range(2))
        for ref in (a, b):
            cs._csw_concept_row(0, ref[1])
        cs.add_whole(a[1], b)
        def forbidden(*args, **kwargs):
            raise AssertionError("taxonomy query must not build/read the vector proposal route")
        monkeypatch.setattr(model, "_reasoning_spaces", forbidden)
        assert not hasattr(model, "_legacy_bridge_components")
        q = QuerySpec.from_surface("isPart", a, b)
        assert model.reason_about(q).posture == "TRUE"
        assert model.think_about(q).posture == "TRUE"
    finally:
        model.End()


def test_open_taxonomy_reports_a_neighbor_without_a_payload_without_allocating():
    cs = _cs()
    model = model_for(cs)
    a, b = (("sym", cs.new_concept()) for _ in range(2))
    cs._csw_concept_row(0, a[1])
    cs.add_whole(a[1], b)
    registry = model.grammatical_thoughts
    query = registry.form("part", a, open_roles=("I2",))
    from Queries import _existing_row
    with pytest.raises(ValueError, match="no allocated payload"):
        _existing_row(cs, b)
    result = model.reason_about(query).result
    assert result.result_kind == "set" and not result.value
    assert "unavailable_concept_payload" in result.incomplete
    assert result.evidence["unavailable_references"] == (b,)
    assert result.evidence["edges_expanded"] >= 1
    with pytest.raises(ValueError, match="no allocated payload"):
        _existing_row(cs, b)
