"""Idea decoding uses the common generate walk and a completed field."""
from types import SimpleNamespace

import torch

from Models import BasicModel
from Spaces import SubSpace


def test_idea_decode_uses_the_field_and_a_fresh_carrier():
    seed = torch.tensor([[[1., 2., 3., 4.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    generated = torch.tensor([[[4., 3., 2., 1.], [1., 1., 1., 1.]]])
    carrier = SubSpace([1, 4], [1, 4], nInputDim=4, nOutputDim=4)
    carrier.set_event(seed[:, :1])
    before = carrier.materialize().clone()
    calls = []
    def generate(value, budget, stamped_events):
        calls.append((value, budget, stamped_events))
        return generated, torch.tensor([2]), torch.tensor([False]), torch.tensor(0.)
    owner = SimpleNamespace(
        conceptualSpace=SimpleNamespace(subspace=carrier),
        _sentence_end_state=lambda _: seed,
        _walk_operand=lambda value: value,
        _walk_budget=lambda: 2,
        _output_generate_walk=generate)
    result = BasicModel._run_idea_decode_generate(owner)
    assert result is not None and result is not carrier
    torch.testing.assert_close(result.materialize(), generated, rtol=0, atol=0)
    torch.testing.assert_close(carrier.materialize(), before, rtol=0, atol=0)
    assert len(calls) == 1 and calls[0][0] is seed
    assert calls[0][1:] == (2, False)


def test_idea_decode_without_a_completed_field_has_no_operand():
    owner = SimpleNamespace(_sentence_end_state=lambda _: None)
    assert BasicModel._run_idea_decode_generate(owner) is None
