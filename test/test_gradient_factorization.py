"""Separate state paths, shared weights, and diagnostics without intervention."""
import json
import os

os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

from GradientDiagnostics import objective_agreement, record_opposition


@pytest.mark.parametrize("other, expected", [([1., 0.], 1.), ([-1., 0.], -1.), ([0., 1.], 0.)])
def test_operator_cosine_reads_weighted_objectives_without_mutating_gradients(other, expected):
    p = torch.nn.Parameter(torch.tensor([.2, .4]))
    p.grad = torch.tensor([7., 8.])
    before = p.detach().clone()
    answer = (p * torch.tensor(other)).sum()
    objectives = {"reconstruction": 2 * p[0], "output": 3 * answer, "expectation": p.sum()}
    report = objective_agreement(objectives, {"operator.CS.verb": [p, p]})["operator.CS.verb"]
    assert report["reconstruction_output_cosine"] == pytest.approx(expected)
    assert report["reconstruction_norm"] == pytest.approx(2.)
    assert report["output_norm"] == pytest.approx(3.)
    assert report["reconstruction_expectation_cosine"] == pytest.approx(2 ** -.5)
    torch.testing.assert_close(p, before)
    torch.testing.assert_close(p.grad, torch.tensor([7., 8.]))
    sum(objectives.values()).backward()
    torch.testing.assert_close(p.grad, torch.tensor([10., 9.]) + 3 * torch.tensor(other))


def test_sparse_codebook_cosine_uses_touched_rows(monkeypatch):
    table = torch.nn.Embedding(1_000_000, 2, sparse=True)
    a = table(torch.tensor([3, 42])).sum()
    b = -table(torch.tensor([42, 900_000])).sum()
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must not densify a codebook gradient")
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    report = objective_agreement({"reconstruction": a, "output": b}, {"codebook": [table.weight]})
    assert report["codebook"]["reconstruction_output_cosine"] == pytest.approx(-.5)
    assert report["codebook"]["reconstruction_expectation_cosine"] is None
    assert table.weight.grad is None


def test_missing_or_zero_gradients_are_not_reported_as_agreement():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    report = objective_agreement({"reconstruction": 0 * p.sum(), "output": p.sum()}, {"operator": [p]})
    assert report["operator"]["reconstruction_output_cosine"] is None
    assert report["operator"]["reconstruction_expectation_cosine"] is None


def test_persistent_opposition_is_named_without_changing_training():
    p = torch.nn.Parameter(torch.tensor([1.]))
    history = {}
    for step in range(3):
        report = objective_agreement({"reconstruction": p.sum(), "output": -p.sum()}, {"operator.CS.surface": [p]})
        record_opposition(report, history)
        assert report["operator.CS.surface"]["output_negative_streak"] == step + 1
    assert report["operator.CS.surface"]["persistent_opposition"] == ["output"]
    assert p.grad is None
    report = objective_agreement({"reconstruction": p.sum(), "output": p.sum()}, {"operator.CS.surface": [p]})
    record_opposition(report, history)
    assert not report["operator.CS.surface"]["persistent_opposition"]


def test_large_finite_gradients_and_mixed_sparse_dense_cosine():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    report = objective_agreement({"reconstruction": 2.e20 * p[0],
        "output": -3.e20 * p[0] + 4.e20 * p[1]}, {"operator": [p]})
    assert report["operator"]["reconstruction_output_cosine"] == pytest.approx(-.6)
    table = torch.nn.Embedding(10, 2, sparse=True)
    report = objective_agreement({"reconstruction": table(torch.tensor([2])).sum(),
        "output": table.weight.sum()}, {"codebook": [table.weight]})
    assert report["codebook"]["reconstruction_output_cosine"] == pytest.approx(10 ** -.5)


@pytest.mark.parametrize("name,value", [("reconstructionPriority", "true"),
    ("outputGradientRatio", "0.5"), ("reconstructionLossTolerance", "1e-8")])
def test_retired_projection_configuration_is_rejected(tmp_path, name, value):
    from test_meronomy_ladder import _build_ladder_variant
    with pytest.raises(ValueError, match=name):
        _build_ladder_variant(tmp_path, "retired", [
            ("<training>", f"<training><{name}>{value}</{name}>")])


def test_policy_observation_and_checked_truth_effect_are_detached():
    from dataclasses import replace
    from types import SimpleNamespace, MappingProxyType
    from Output import thought_answer_meanings
    from ThoughtFeatures import context_width
    from Queries import ThoughtResult
    from test_normal_thought_controller import _catalog_world
    model, registry, _, part, whole = _catalog_world()
    request = registry.form("part", part, whole)
    state = request.roles.detach().clone().requires_grad_()
    request = replace(request, roles=state)
    features = model._selected_thought_context(
        request, request, request, level=0, pressure=0.)
    assert features.numel() == context_width(state.shape[-1])
    assert not features.requires_grad
    checked = ThoughtResult("part", "taxonomy", "truth", "concept-taxonomy",
        request, MappingProxyType({"support_true": 1., "support_false": 0.}))
    answer = thought_answer_meanings(SimpleNamespace(meaning=request, result=checked))
    assert answer and not answer[0].roles.requires_grad


def test_normal_batch_logs_named_shared_operator_gradients(tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from test_compiled_word_chunk import _tiny_canonical_model
    curriculum = json.loads((Path(__file__).resolve().parents[1] / "data/grammar_wording.json").read_text())
    chosen = ("wheels belong to bicycles", "bicycles are equal to wheels")
    lessons = [next(row for row in curriculum["train"] if row["text"] == text) for text in chosen]
    torch.manual_seed(942)
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="8")
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.branch_diagnostics_every = 1
    model.inputSpace.data.grammar_lessons = {"train": lessons}
    model.inputSpace.data.has_supervised_outputs = False
    optimizer = model.getOptimizer(lr=.003)
    try:
        inputs = model.inputSpace.prepInput(list(chosen))
        result, _ = model.runBatch(train=True, batchNum=0, batchSize=2, split="train", optimizer=optimizer,
            source_rows=[0, 1], batch_override=(inputs, torch.empty(2, 0)))
        assert result is not None
        report = model._last_operator_gradients
        assert any(name.startswith("operator.") for name in report)
        assert any(entry["reconstruction_norm"] > 0 for entry in report.values())
        assert any(entry["output_norm"] > 0 for entry in report.values())
        assert isinstance(model.conceptualSpace.similarity_codebook.W, torch.nn.Parameter)
        assert any(name.startswith("codebook.") and entry["reconstruction_norm"] > 0
                   for name, entry in report.items())
        assert "[operator-gradients]" in capsys.readouterr().out
        active = {name: entry for name, entry in report.items()
                  if entry["reconstruction_output_cosine"] is not None}
        assert active, "measure actual overlap, not two disconnected gradient sets"
        print("MEASURED_OPERATOR_GRADIENTS", json.dumps(active, sort_keys=True))
    finally:
        model.inputSpace.data.grammar_lessons = {}
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
