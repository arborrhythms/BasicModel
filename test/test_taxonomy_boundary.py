"""Public typed PartOf evaluation preserves polarity and avoids unrelated reads."""
from types import SimpleNamespace

import pytest

from reasoning import QuerySpec, TruthGroundedReasoner, NeuralToolUser
from thinking import ThinkingKernel
from test_cs_symbol_table import _cs


def _chain():
    cs = _cs()
    a, b, c = (("sym", cs.new_concept()) for _ in range(3))
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    return (a, b, c), TruthGroundedReasoner(SimpleNamespace(conceptualSpace=cs))


def test_negated_partof_keeps_the_supported_proposition_and_evidence():
    (a, _b, c), reasoner = _chain()
    q = QuerySpec.from_surface("isPart", a, c, polarity=False)
    result = reasoner.evaluate(q)
    assert result["posture"] == "FALSE"
    assert result["support_true"] == 0 and result["support_false"] == 1
    assert result["path"][0].part == a and result["path"][-1].whole == c
    assert reasoner.evaluate(QuerySpec.from_surface("isPart", c, a, polarity=False))["posture"] == "UNKNOWN"
    assert ThinkingKernel(reasoner).run(q).value == "false"


def test_neural_result_retains_missing_reference_diagnostics():
    (a, _b, _c), reasoner = _chain()
    result = NeuralToolUser(reasoner).run(QuerySpec.from_surface("isPart", a, ("sym", 999999)))
    assert result.posture == "UNKNOWN"
    assert "unavailable_reference" in result.evidence["incomplete"]


def test_model_taxonomy_entries_skip_global_vector_proposal_setup(monkeypatch):
    from test_ltm_consolidation import _make_model, _STATEFUL_CONFIG
    model = _make_model(_STATEFUL_CONFIG)
    try:
        model.reasoning_iterations = model.thinking_budget = 8
        cs = model.conceptualSpace
        a, b = (("sym", cs.new_concept()) for _ in range(2))
        cs.add_whole(a[1], b)
        def forbidden(*args, **kwargs):
            raise AssertionError("taxonomy query must not build/read the vector proposal route")
        monkeypatch.setattr(model, "_reasoning_spaces", forbidden)
        monkeypatch.setattr(model, "_reasoning_tooluser", forbidden)
        q = QuerySpec.from_surface("isPart", a, b)
        assert model.reason_about(q).posture == "TRUE"
        assert model.think_about(q).value == "true"
    finally:
        model.End()
