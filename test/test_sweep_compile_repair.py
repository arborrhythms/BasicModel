"""The sentence reconstruction brick uses the shared compiler policy."""
import torch
import pytest
from torch import nn
import util
from Models import BasicModel


def test_reconstruction_brick_rejects_second_width_in_one_phase(monkeypatch):
    class Probe(BasicModel):
        def __init__(self):
            nn.Module.__init__(self)
        def _recon_placement(self):
            return 'compiled'
        def _reconstruct_sentences(self, value):
            return value.square().sum()

    torch._dynamo.reset()
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager')
    # The config module serves its keys through __getattr__, so pytest's
    # monkeypatch (9.x) would undo a patch here by deleting the key for the
    # rest of the process; torch's own patch restores it.
    with torch._dynamo.config.patch(recompile_limit=8):
        model = Probe()
        compiled = model._compiled_reconstruct()
        assert model._compiled_reconstruct() is compiled
        # Reversed contract: a running phase admits its constructed width only.
        value = torch.arange(1, dtype=torch.float32).requires_grad_()
        try:
            actual = compiled(value)
            torch.testing.assert_close(actual, value.square().sum(), rtol=0, atol=0)
            actual.backward()
            torch.testing.assert_close(value.grad, 2*value, rtol=0, atol=0)
            with pytest.raises(RuntimeError, match='Unexpected model recompilation.*_reconstruct_sentences'):
                compiled(torch.arange(2, dtype=torch.float32).requires_grad_())
        finally:
            torch._dynamo.reset()
