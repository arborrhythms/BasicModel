"""The tied reconstruction learning contract, including evaluation and migration."""
import pytest
import torch


@pytest.mark.parametrize("training", [False, True])
def test_tied_runbatch_scores_the_same_owned_objective_once(tmp_path, monkeypatch, training):
    from test_meronomy_ladder import _build_ladder_variant
    model = _build_ladder_variant(tmp_path, "tied_objective", [
        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"
         "<reconstructInLoop>true</reconstructInLoop>")])
    # Production defaults to compiled-only HOP execution. Tied reconstruction
    # must also have completed sentence products during no-grad evaluation.
    model._tensor_peer_while_eager = False
    model._chart_compose_per_word = lambda: None
    model.loss.reconstruction_scale = 1.0
    model.reconstruction_priority = False
    model.leaf_distill_weight = 1.0
    previous_supervision = model.inputSpace.data.has_supervised_outputs
    model.inputSpace.data.has_supervised_outputs = False
    record = model.record_loss
    recorded = {}
    distillation_calls = []

    def capture(name, value, **kwargs):
        recorded[name] = value
        return record(name, value, **kwargs)

    def forbidden():
        raise AssertionError("tied mode entered the legacy per-word D3 objective")

    monkeypatch.setattr(model, "record_loss", capture)
    monkeypatch.setattr(model, "_d3_reconstruction_loss", forbidden)
    monkeypatch.setattr(model, "_leaf_distill_loss", lambda: distillation_calls.append(True))
    try:
        optimizer = model.getOptimizer(lr=0.) if training else None
        batch = (model.inputSpace.prepInput(["aa bb cc"]), torch.zeros(1, 1, 0))
        result, _ = model.runBatch(
            train=training, batchSize=1, split="train" if training else "validation",
            optimizer=optimizer, batch_override=batch)
        owned = model._last_understanding.input_reconstruction
        assert owned is not None
        torch.testing.assert_close(result.lossIn, owned.byte_cost.mean())
        torch.testing.assert_close(recorded["reconstruction"], owned.byte_cost.mean())
        reverse = model.primary_costs()["input_reconstruction_reverse"]
        assert reverse is None or float(reverse.detach()) == 0.
        assert distillation_calls == [], "tied mode must not train a separate root-to-leaf decoder"
    finally:
        model.inputSpace.data.has_supervised_outputs = previous_supervision
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_production_configuration_selects_tied_input_objective():
    from pathlib import Path
    import xml.etree.ElementTree as ET
    path = Path(__file__).resolve().parents[1] / "data" / "BasicModel.xml"
    training = ET.parse(path).getroot().find("architecture/training")
    assert training.findtext("teacherReconstruction") == "true"
    assert training.findtext("detachedReverse") == "false"
    assert training.findtext("reconstructInLoop") == "true"
    assert training.findtext("reconstructionPlacement") == "compiled"
    assert float(training.findtext("leafDistillWeight", "0")) == 0.


def test_student_checkpoint_migrates_shared_weights_and_adam_by_name(tmp_path):
    from test_meronomy_ladder import _build_ladder_variant
    legacy = _build_ladder_variant(tmp_path, "old_student", [
        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"
         "<detachedReverse>true</detachedReverse>")])
    fresh = None
    try:
        student = legacy.symbolSpace.reverse_chooser
        assert student is not None
        # Ordinary what()/training has materialized these answer modules
        # before saving. Their lazy construction is unrelated to retiring
        # the reconstruction student and would otherwise make strict load
        # fail because the fixture had never run an answer path.
        legacy._materialize_answer_path()
        parameters = dict(legacy.named_parameters())
        name = next(name for name in parameters if name.endswith("_sigma.layer.raw_L"))
        shared = parameters[name]
        optimizer = legacy.getOptimizer(lr=.001)
        legacy._optimizer = optimizer
        shared.grad = torch.full_like(shared, .125)
        student_parameter = next(student.parameters())
        student_parameter.grad = torch.full_like(student_parameter, .25)
        optimizer.step()
        expected_parameter = shared.detach().clone()
        expected_moment = optimizer.state[shared]["exp_avg"].clone()
        checkpoint = tmp_path / "student.ckpt"
        legacy.save_weights(str(checkpoint))
        saved = torch.load(checkpoint, weights_only=False)
        assert any("reverse_chooser." in key for key in saved["state_dict"])
        fresh = _build_ladder_variant(tmp_path, "new_tied", [
            ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"
             "<reconstructInLoop>true</reconstructInLoop>")])
        assert fresh.load_weights(str(checkpoint), strict=True, require_match=True)
        assert fresh.symbolSpace.reverse_chooser is None
        restored = dict(fresh.named_parameters())[name]
        torch.testing.assert_close(restored, expected_parameter, atol=0, rtol=0)
        new_optimizer = fresh.getOptimizer(lr=.001)
        torch.testing.assert_close(new_optimizer.state[restored]["exp_avg"], expected_moment)
        live = {id(p) for p in fresh.parameters()}
        assert all(id(p) in live for group in new_optimizer.param_groups for p in group["params"])
    finally:
        for model in (legacy, fresh):
            if model is not None:
                model.End()
                model.symbolSpace.soft_reset()
