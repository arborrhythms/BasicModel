"""Reviewer probes for integrated spec §10.3 and the §11 expectation rollout."""
import copy
from pathlib import Path
import re
import pytest
import torch
from test_sentence_expectation import layer, observe


def test_evaluation_reports_residual_without_training_or_weight_changes():
    model = layer()
    model.eval()
    before = copy.deepcopy(model.state_dict())
    with torch.enable_grad():
        observe(model, torch.ones(3, 4))
        observe(model, torch.full((3, 4), 2.))
    metrics = model.expectation_metrics()
    assert metrics["observations"] == 2
    assert metrics["cold_starts"] == 1
    assert metrics["predicted_targets"] == 1
    assert metrics["feature_mse"] > 0
    assert metrics["presence_bce"] > 0
    assert model.consume_inter_loss() is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    comparison = model.last_expectation_comparison(0)
    assert not comparison.estimate.roles.requires_grad
    assert not comparison.observed.requires_grad
    torch.testing.assert_close(comparison.residual,
                               comparison.observed - comparison.estimate.roles)


def test_fixed_prior_changing_target_changes_residual_not_estimate():
    a, b = layer(), layer()
    for model in (a, b):
        model.eval()
        observe(model, torch.ones(3, 4))
    observe(a, torch.zeros(3, 4))
    observe(b, torch.full((3, 4), 3.))
    ca, cb = a.last_expectation_comparison(0), b.last_expectation_comparison(0)
    torch.testing.assert_close(ca.estimate.roles, cb.estimate.roles)
    assert not torch.allclose(ca.residual, cb.residual)
    assert a.expectation_metrics()["predicted_targets"] == 1
    assert b.expectation_metrics()["predicted_targets"] == 1


def test_disable_reenable_drops_pending_and_requires_fresh_predecessor():
    model = layer()
    observe(model, torch.ones(3, 4))
    assert model.expect_next_meaning() is not None
    model.set_expectation_enabled(False)
    observe(model, torch.full((3, 4), 2.))
    observe(model, torch.full((3, 4), 3.))
    assert model.expect_next_meaning() is None
    assert model.consume_inter_loss() is None
    assert model.expectation_metrics()["predicted_targets"] == 0
    model.set_expectation_enabled(True)
    observe(model, torch.full((3, 4), 4.))
    assert model.consume_inter_loss() is None
    observe(model, torch.full((3, 4), 5.))
    assert model.consume_inter_loss() is not None
    assert model.expectation_metrics()["predicted_targets"] == 1


def test_all_padding_is_not_an_external_observation():
    model = layer()
    observe(model, torch.ones(3, 4))
    observe(model, torch.full((3, 4), 99.), mask=torch.zeros(3, dtype=torch.bool))
    assert model.expectation_metrics()["observations"] == 1
    assert model.consume_inter_loss() is None
    assert len(model._inter_context[0]) == 1


@pytest.mark.parametrize("setting,enabled", [(None, True), ("true", True), ("false", False)])
def test_omitted_enabled_and_disabled_model_settings(tmp_path, setting, enabled):
    from test_meronomy_ladder import _build_ladder_variant
    source = (Path(__file__).resolve().parents[1] / "data/MM_ladder.xml").read_text()
    replacement = "" if setting is None else f"<sentenceExpectation>{setting}</sentenceExpectation>"
    model = _build_ladder_variant(tmp_path, "expectation_default", [
        (re.search(r"<sentenceExpectation>.*?</sentenceExpectation>", source).group(), replacement),
        (re.search(r"<interLossWeight>.*?</interLossWeight>", source).group(), ""),
        (re.search(r"<armaScale>.*?</armaScale>", source).group(), ""),
    ])
    try:
        assert (model.symbolSpace.discourse is not None) == enabled
        assert model.inter_loss_weight == 0.1
        assert model.arma_scale == 0
        assert model.inter_contrastive_weight == 0
        if enabled:
            assert model.symbolSpace.discourse.expectation_scope == "structured"
    finally:
        model.End()
        torch._dynamo.reset()


def test_canonical_configuration_enables_only_the_selected_prediction_objective():
    import xml.etree.ElementTree as ET
    config = ET.parse(Path(__file__).resolve().parents[1] / "data/BasicModel.xml")
    training = config.getroot().find(".//architecture/training")
    assert training.findtext("sentenceExpectation") == "true"
    assert float(training.findtext("interLossWeight")) == .1
    assert float(training.findtext("armaScale")) == 0
    assert float(training.findtext("interContrastiveWeight")) == 0


def test_native_runtime_reports_pairs_without_accumulating_or_updating(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "runtime_expectation", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ("<interLossWeight>0.0</interLossWeight>", "<interLossWeight>0.1</interLossWeight>"),
    ])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    model.train()  # the runtime call must apply its own declared training gate
    discourse = model.symbolSpace.discourse
    before = [p.detach().clone() for p in discourse.parameters()]
    try:
        inputs = model.inputSpace.prepPackedInput([["1 plus 2", "3 plus 4"]])
        with torch.enable_grad():
            result, _ = model.runBatch(
                train=False, batchSize=1, split="runtime",
                batch_override=(inputs, torch.zeros(1, 1, 0)))
        assert result is not None
        assert discourse.consume_inter_loss() is None
        metrics = discourse.expectation_metrics()
        assert metrics["observations"] == 2
        assert metrics["predicted_targets"] == 1
        assert metrics["feature_mse"] >= 0
        assert all(not value.requires_grad for chain in discourse._inter_context
                   for _, value, _ in chain)
        for parameter, value in zip(discourse.parameters(), before):
            torch.testing.assert_close(parameter, value)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_native_future_and_other_row_changes_do_not_change_first_estimate(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    estimates = []
    for index, rows in enumerate((
        [["1 plus 2", "3 plus 4"], ["5 plus 6"]],
        [["1 plus 2", "7 plus 8"], ["9 plus 0"]],
    )):
        model = _build_ladder_variant(tmp_path, f"isolated_expectation_{index}", [
            ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
            ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
            ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
            ("<interLossWeight>0.0</interLossWeight>", "<interLossWeight>0.1</interLossWeight>"),
        ])
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
        model._install_unit_span_fn()
        model.set_sigma(0)
        try:
            inputs = model.inputSpace.prepPackedInput(rows)
            model.runBatch(train=False, batchSize=2, split="runtime",
                           batch_override=(inputs, torch.zeros(2, 1, 0)))
            comparison = model.symbolSpace.discourse.last_expectation_comparison(0)
            assert comparison is not None
            estimates.append(comparison.estimate.roles)
        finally:
            model.End()
            model.symbolSpace.soft_reset()
            torch._dynamo.reset()
    torch.testing.assert_close(estimates[0], estimates[1])


@pytest.mark.slow
def test_enable_after_disabled_construction_joins_optimizer_and_off_keeps_input_learning(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "toggle_expectation", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<interLossWeight>0.0</interLossWeight>", "<interLossWeight>0.1</interLossWeight>"),
    ])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    monkeypatch.setattr(model.inputSpace.data, "has_supervised_outputs", False)
    model.eval()
    optimizer = model.getOptimizer(lr=1e-5)
    owner = model._what_memory()
    assert model.symbolSpace.discourse is None
    try:
        for enabled in (True, False, True):
            model.set_sentence_expectation(enabled)
            discourse = model.symbolSpace.discourse
            parameters = list(discourse._inter_predictor.parameters())
            before = [p.detach().clone() for p in parameters]
            count = discourse.expectation_metrics()["predicted_targets"]
            inputs = model.inputSpace.prepPackedInput([["1 plus 2", "3 plus 4"]])
            result, _ = model.runBatch(train=True, batchSize=1, optimizer=optimizer,
                                      batch_override=(inputs, torch.zeros(1, 1, 0)))
            assert model._what_memory() is owner
            assert result.lossIn is not None
            changes = [not torch.equal(p, old) for p, old in zip(parameters, before)]
            assert any(changes) if enabled else not any(changes)
            assert discourse.expectation_metrics()["predicted_targets"] - count == int(enabled)
            owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert all(id(p) in owned for p in parameters)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_expectation_off_keeps_every_packed_observation_in_ltm(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "off_packed_memory", [
        ("<architecture>", "<architecture><ltmConsolidation>true</ltmConsolidation>"),
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
    ])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    assert model.symbolSpace.discourse is None
    store = model.symbolSpace.ltm_store
    try:
        before = len(store)
        inputs = model.inputSpace.prepPackedInput([["1 plus 2", "3 plus 4"]])
        model.runBatch(train=False, batchSize=1, split="runtime",
                       batch_override=(inputs, torch.zeros(1, 1, 0)))
        assert len(store) - before == 2
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_packed_ltm_ignores_masked_slots_even_with_retained_storage():
    from types import SimpleNamespace
    from Layers import TernaryTruthStore
    from Models import BasicModel
    discourse = layer(consolidated=True)
    store = discourse._ltm_store
    host = SimpleNamespace(
        symbolSpace=SimpleNamespace(discourse=discourse, ltm_store=store),
        conceptualSpace=SimpleNamespace(_ltm_consolidation=True),
        inputSpace=SimpleNamespace(
            _packed_sentence_slot_end_positions=torch.tensor([[1, 3]]),
            _packed_sentence_slot_mask=torch.tensor([[True, False]]),
            _packed_sentence_counts_host=(2,)),
        _packed_sentence_roots=torch.ones(1, 2, 4),
        _tensor_sentence_roots_live=torch.ones(1, 2, 12),
        _tensor_sentence_roots_depth=torch.tensor([[3, 3]]),
        _tensor_final_end_slots=torch.full((1, 3, 4), 99.),
        _tensor_final_end_depth=torch.tensor([3]))
    host._expectation_documents_for_slot = lambda t, batch: ["a"]
    BasicModel._drain_packed_stm_end_states(host)
    assert len(store) == 1
    assert discourse.expectation_metrics()["observations"] == 1


def test_enabling_expectation_preserves_the_callers_random_stream(tmp_path):
    from test_meronomy_ladder import _build_ladder_variant
    model = _build_ladder_variant(tmp_path, "expectation_rng", [])
    assert model.symbolSpace.discourse is None
    before = torch.get_rng_state().clone()
    try:
        model.set_sentence_expectation(True)
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        model.End()
        torch._dynamo.reset()
