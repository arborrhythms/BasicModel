"""Grammar-owned word learning: mechanism and production-wiring checks.

These are not a held-out language-quality or learned questioning-utility study.
In particular, the equal-output probe deliberately isolates information that
the ordinary grammar MLP must retain before it can learn converse wordings.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(_ROOT / "bin"))

import torch
from torch import nn
import pytest

from Language import BinaryStructuredReductionLayer, MLPTransformChooser


class _SymmetricCandidate(nn.Module):
    def forward(self, left, right):
        return left + right


@pytest.mark.parametrize("filename", ["complete.grammar", "ladder.grammar",
                                     "tied_reconstruction_output_benchmark.grammar"])
def test_natural_relation_words_have_no_bootstrap_operator_assignment(filename):
    import xml.etree.ElementTree as ET

    anchors = ET.parse(_ROOT / "data" / filename).getroot().find("Anchors")
    forms = {surface.strip().casefold(): node.tag
             for node in anchors for surface in (node.text or "").split(",")}
    assert forms["partof"] == "part" and forms["wholeof"] == "whole"
    assert forms["isequal"] == "equal"
    assert not {"has", "contains", "includes", "belongs", "equals", "is"}.intersection(forms)


@pytest.mark.parametrize("signal", ["concepts", "roles"])
def test_same_candidate_values_do_not_erase_operand_order(signal):
    torch.manual_seed(391)
    layer = BinaryStructuredReductionLayer(
        d_model=4, ops=[_SymmetricCandidate(), _SymmetricCandidate()],
        chooser="mlp", n_role_cats=2)
    chooser = layer.chooser
    pair = torch.tensor([[[0.9, -0.4, 0.2, 0.7], [-0.1, 0.8, -0.5, 0.3]]])
    x = torch.cat((pair, pair.flip(1)))
    cats = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    if signal == "roles":
        x = x.mean(1, keepdim=True).expand_as(x)
    else:
        cats = None
    candidate = x.sum(1)[:, None, None].expand(-1, 1, 2, -1)
    labels = torch.tensor([0, 1])
    optimizer = torch.optim.Adam(chooser.parameters(), lr=0.03)
    for _ in range(160):
        optimizer.zero_grad()
        _, scores = chooser.score_binary(x, candidate, None, None, cat_ctx=cats)
        loss = torch.nn.functional.cross_entropy(scores[:, 0], labels)
        loss.backward()
        optimizer.step()
    _, scores = chooser.score_binary(x, candidate, None, None, cat_ctx=cats)
    assert scores[:, 0].argmax(-1).tolist() == labels.tolist()
    assert float(torch.nn.functional.cross_entropy(scores[:, 0], labels).detach()) < 0.05


def test_legacy_grammar_weights_keep_predictions_and_new_order_weights_start_zero():
    torch.manual_seed(793)
    legacy = MLPTransformChooser(d_model=4, n_copy=1, n_op=2, n_role_cats=2)
    rng_after = torch.random.get_rng_state().clone()
    torch.manual_seed(793)
    current = MLPTransformChooser(
        d_model=4, n_copy=1, n_op=2, n_role_cats=2, ordered_binary=True)
    assert torch.equal(torch.random.get_rng_state(), rng_after)
    with torch.no_grad():
        current.operand_order.weight.fill_(9.)
    current.load_state_dict(legacy.state_dict(), strict=True)
    assert torch.count_nonzero(current.operand_order.weight) == 0
    assert torch.count_nonzero(current.copy_order.weight) == 0
    x, candidate, cats = torch.randn(2, 3, 4), torch.randn(2, 2, 2, 4), torch.randn(2, 3, 2)
    old = legacy.score_binary(x, candidate, None, None, cat_ctx=cats)
    new = current.score_binary(x, candidate, None, None, cat_ctx=cats)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    state = current.state_dict()
    del state["operand_order.weight"]
    with pytest.raises(RuntimeError, match="operand_order.weight"):
        current.load_state_dict(state, strict=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_order_projection_does_not_advance_the_default_device_rng():
    cpu_rng, mps_rng = torch.random.get_rng_state(), torch.mps.get_rng_state()
    try:
        with torch.device("mps"):
            torch.manual_seed(941)
            MLPTransformChooser(d_model=4, n_copy=1, n_op=2)
            expected = torch.mps.get_rng_state()
            torch.manual_seed(941)
            chooser = MLPTransformChooser(
                d_model=4, n_copy=1, n_op=2, ordered_binary=True)
            assert chooser.operand_order.weight.device.type == "mps"
            assert torch.equal(torch.mps.get_rng_state(), expected)
    finally:
        torch.random.set_rng_state(cpu_rng)
        torch.mps.set_rng_state(mps_rng)


def test_order_projection_compiles_and_restores_with_its_optimizer_moments():
    from checkpoint_migrations import build_optimizer_param_manifest, remap_optimizer_state_by_name

    kwargs = dict(d_model=4, n_copy=1, n_op=2, n_role_cats=2)
    legacy = MLPTransformChooser(**kwargs)
    old_optimizer = torch.optim.Adam(legacy.parameters(), lr=.001)
    sum(p.square().sum() for p in legacy.parameters()).backward()
    old_optimizer.step()
    current = MLPTransformChooser(**kwargs, ordered_binary=True)
    current.load_state_dict(legacy.state_dict(), strict=True)
    optimizer = torch.optim.Adam(current.parameters(), lr=.001)
    remapped = remap_optimizer_state_by_name(
        old_optimizer.state_dict(), build_optimizer_param_manifest(old_optimizer, legacy.named_parameters()),
        optimizer.state_dict(), build_optimizer_param_manifest(optimizer, current.named_parameters()))
    optimizer.load_state_dict(remapped.state)
    assert current.operand_order.weight not in optimizer.state
    for name, parameter in legacy.named_parameters():
        torch.testing.assert_close(
            old_optimizer.state[parameter]["exp_avg"],
            optimizer.state[dict(current.named_parameters())[name]]["exp_avg"], atol=0, rtol=0)
    x, candidate, cats = torch.randn(2, 3, 4), torch.randn(2, 2, 2, 4), torch.randn(2, 3, 2)
    def score(value):
        return current.score_binary(value, candidate, None, None, cat_ctx=cats)
    try:
        compiled = torch.compile(score, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), score(x))
        optimizer.zero_grad()
        sum(value.square().sum() for value in compiled(x)).backward()
        assert current.operand_order.weight.grad.abs().sum() > 0
        assert current.copy_order.weight.grad.abs().sum() > 0
        optimizer.step()
        restored = MLPTransformChooser(**kwargs, ordered_binary=True)
        restored.load_state_dict(current.state_dict(), strict=True)
        resumed = torch.optim.Adam(restored.parameters(), lr=.001)
        resumed.load_state_dict(optimizer.state_dict())
        torch.testing.assert_close(restored.score_binary(x, candidate, None, None, cat_ctx=cats), score(x))
        torch.testing.assert_close(resumed.state[restored.operand_order.weight]["exp_avg"],
                                   optimizer.state[current.operand_order.weight]["exp_avg"])
        torch.testing.assert_close(resumed.state[restored.copy_order.weight]["exp_avg"],
                                   optimizer.state[current.copy_order.weight]["exp_avg"])
    finally:
        torch._dynamo.reset()


def test_normal_text_reconstruction_updates_the_grammar_chooser(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model

    torch.manual_seed(613)
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="8,16")
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.conceptualSpace.intra_loss_weight = 0.0
    model.inter_loss_weight = 0.0
    assert model.output_in_loop and model.output_policy_weight > 0
    assert model.loss.reconstruction_scale == 1.0
    chooser = model.languageSpace._tree_layer(2).chooser
    before = {name: value.detach().clone() for name, value in chooser.named_parameters()}
    generator = model.languageSpace.generate_policy
    generate_before = [p.detach().clone() for p in generator.parameters()]
    optimizer = model.getOptimizer(lr=1e-3)
    words = ["a bicycle has a wheel", "a wheel belongs to a bicycle"]
    inputs = model.inputSpace.prepInput(words)
    data = model.inputSpace.data
    supervised_before = data.has_supervised_outputs
    data.has_supervised_outputs = False
    try:
        result, _ = model.runBatch(
            train=True, batchNum=0, batchSize=2, split="train", optimizer=optimizer,
            batch_override=(inputs, torch.empty(2, 0)))
        assert result is not None
        changed = {name for name, value in chooser.named_parameters()
                   if not torch.equal(before[name], value.detach())}
        assert "operand_order.weight" in changed
        assert any(name.startswith("mlp.") for name in changed)
        enlisted = [p for group in optimizer.param_groups for p in group["params"]]
        for parameter in chooser.parameters():
            assert sum(parameter is p for p in enlisted) == 1
        assert model.forward_grammar_weight == 0.0
        understanding = model._last_understanding
        assert len(understanding.answer_program) == len(words)
        assert all(program.leaves.shape[0] >= 5 for program in understanding.answer_program)
        assert understanding.input_reconstruction is not None
        assert bool(torch.isfinite(understanding.input_reconstruction.byte_cost).all())
        assert torch.is_tensor(model._output_policy_cost)
        assert any(step["operation"] == "generate:grammar"
                   for step in model._last_answer_construction.trace)
        assert all(torch.equal(old, new) for old, new in zip(generate_before, generator.parameters()))
    finally:
        data.has_supervised_outputs = supervised_before
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
