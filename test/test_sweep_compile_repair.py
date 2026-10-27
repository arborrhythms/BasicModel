"""The sentence reconstruction brick uses the shared compiler policy."""
import torch
from torch import nn
import util
from Models import BasicModel


def test_reconstruction_brick_accepts_more_than_default_eight_shapes(monkeypatch):
    class Probe(BasicModel):
        def __init__(self):
            nn.Module.__init__(self)
        def _recon_placement(self):
            return 'compiled'
        def _reconstruct_sentences(self, value):
            return value.square().sum()

    torch._dynamo.reset()
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager')
    monkeypatch.setattr(torch._dynamo.config, 'recompile_limit', 8)
    model = Probe()
    compiled = model._compiled_reconstruct()
    assert model._compiled_reconstruct() is compiled
    # Nine distinct fixed shapes cross the default cache limit once. This
    # is a compile-capacity probe, not a model-size or learning-budget port.
    inputs = [torch.arange(width, dtype=torch.float32).requires_grad_()
              for width in range(1, 10)]
    try:
        for value in inputs:
            actual = compiled(value)
            torch.testing.assert_close(actual, value.square().sum(), rtol=0, atol=0)
            actual.backward()
            torch.testing.assert_close(value.grad, 2*value, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
