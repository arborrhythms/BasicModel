"""A generated surface needs a rectangular readout, not a square input map."""
from types import SimpleNamespace

import torch
import pytest

from Spaces import OutputSpace


def test_native_generated_percept_readout_has_bounded_parameter_storage():
    """Exercise the failed native dimensions without allocating a terabyte."""
    with torch.device("meta"):
        percepts = torch.zeros(2, 512, 1088)
        owner = SimpleNamespace(outputShape=(16, 1))
        result = OutputSpace.from_percepts(owner, percepts)
    assert result.shape == (2, 16, 1)
    n_input, n_output = 512 * 1088, 16
    size = sum(p.numel() for p in owner.percept_adapter.parameters())
    assert size <= 2 * n_input * n_output + 2 * n_output, (
        "an answer readout must scale with input x output, not input squared: "
        f"{size} parameters for {n_input} -> {n_output}")


@pytest.mark.parametrize("n_input,n_output", [(64, 16), (7, 7), (4, 12), (4, 0)])
def test_compact_readout_preserves_full_ldu_values_and_gradients(n_input, n_output):
    from Layers import InvertibleLinearLayer

    reference = InvertibleLinearLayer(n_input, n_output).to(device="cpu", dtype=torch.float64)
    owner = SimpleNamespace(outputShape=(n_output, 1))
    x = torch.randn(3, 1, n_input, device="cpu", dtype=torch.float64, requires_grad=True)
    OutputSpace.from_percepts(owner, x)
    compact = owner.percept_adapter
    rank = min(n_input, n_output)
    with torch.no_grad():
        reference.raw_L.normal_(0, .07)
        reference.raw_U.normal_(0, .07)
        reference.d.uniform_(.75, 1.25)
        reference.biasWeight.normal_(0, .1)
        compact.raw_L.copy_(reference.raw_L[:, :rank])
        compact.raw_U.copy_(reference.raw_U[:rank, :])
        compact.d.copy_(reference.d)
        compact.biasWeight.copy_(reference.biasWeight)
    expected = reference(x.reshape(3, n_input)).reshape(3, n_output, 1)
    actual = OutputSpace.from_percepts(owner, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    cotangent = torch.randn_like(expected)
    names = ("raw_L", "d", "raw_U", "biasWeight")
    expected_grad = torch.autograd.grad(
        (expected * cotangent).sum(), (x, *(getattr(reference, n) for n in names)))
    actual_grad = torch.autograd.grad(
        (actual * cotangent).sum(), (x, *(getattr(compact, n) for n in names)))
    slices = (expected_grad[0], expected_grad[1][:, :rank], expected_grad[2],
              expected_grad[3][:rank, :], expected_grad[4])
    for observed, wanted in zip(actual_grad, slices):
        torch.testing.assert_close(observed, wanted, atol=1e-12, rtol=1e-12)
    assert not torch.count_nonzero(expected_grad[1][:, rank:])
    assert not torch.count_nonzero(expected_grad[3][rank:, :])


@pytest.mark.parametrize("legacy_adapter", [False, True])
def test_model_checkpoint_preserves_readout_and_adam(tmp_path, legacy_adapter):
    from Layers import InvertibleLinearLayer, LDUReadout
    from What import What
    from test_output_path_supervised import _native_answer_model
    from test_output_walk import _capture_program_probe

    model = _native_answer_model(tmp_path, False)
    fresh = None
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1 plus 2", "3 plus 4"])
            model.reverseOutput(understanding, (What.supervised(0), What.supervised(1)))
        adapter = model.outputSpace.percept_adapter
        if legacy_adapter:
            # Preserve the legacy checkpoint layout, including parameters whose
            # gradients are zero in this forward-only use of the full inverse.
            old = InvertibleLinearLayer(adapter.nInput, adapter.nOutput).to("cpu")
            rank = min(adapter.nInput, adapter.nOutput)
            with torch.no_grad():
                old.raw_L[:, :rank].copy_(adapter.raw_L)
                old.raw_U[:rank, :].copy_(adapter.raw_U)
                old.d.copy_(adapter.d)
                old.biasWeight.copy_(adapter.biasWeight)
            model.outputSpace.percept_adapter = adapter = old
        optimizer = model.getOptimizer(lr=.001)
        model._optimizer = optimizer
        x = torch.linspace(-.2, .3, 2 * adapter.nInput, device="cpu").reshape(2, -1)
        optimizer.zero_grad()
        adapter(x).square().mean().backward()
        optimizer.step()
        assert all(p in optimizer.state for p in adapter.parameters())
        path = tmp_path / "readout.ckpt"
        model.save_weights(str(path))
        directory = tmp_path / "fresh"
        directory.mkdir()
        fresh = _native_answer_model(directory, False)
        assert fresh.load_weights(str(path), strict=True, require_match=True)
        restored = fresh.outputSpace.percept_adapter
        assert isinstance(restored, InvertibleLinearLayer if legacy_adapter else LDUReadout)
        resumed = fresh.getOptimizer(lr=.001)
        ids = [id(p) for group in resumed.param_groups for p in group["params"]]
        for name, parameter in adapter.named_parameters():
            loaded = dict(restored.named_parameters())[name]
            assert ids.count(id(loaded)) == 1
            torch.testing.assert_close(loaded, parameter, rtol=0, atol=0)
            for key, value in optimizer.state[parameter].items():
                torch.testing.assert_close(resumed.state[loaded][key], value, rtol=0, atol=0)
        torch.testing.assert_close(restored(x), adapter(x), rtol=0, atol=0)
        # A resumed step must continue the saved moments, not silently restart.
        for module, opt in ((adapter, optimizer), (restored, resumed)):
            opt.zero_grad()
            module(x).square().mean().backward()
            opt.step()
        for key, value in adapter.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    finally:
        for item in (model, fresh):
            if item is not None:
                item.End()
                item.symbolSpace.soft_reset()
        torch._dynamo.reset()
