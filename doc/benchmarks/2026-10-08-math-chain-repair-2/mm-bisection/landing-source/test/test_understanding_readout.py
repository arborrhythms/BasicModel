"""A numeric answer reads its concluded field; supplied trials train it."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('supplied', [False, True])
def test_answer_reads_understanding_with_gradient_cut(depth, supplied):
    from Models import BasicModel
    from Spaces import SubSpace

    class Readout:
        concept_ids = ()
        inputShape = (6, 4)
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.arange(1., 25.).reshape(1, 24))
        def __call__(self, sub):
            self.seen = sub.materialize()
            return self.seen.flatten(1) @ self.weight.T

    root = torch.arange(1., 9.).reshape(2, 4).requires_grad_()
    field = torch.arange(1., 25.).reshape(2, 3, 4).requires_grad_()
    slab = torch.full((2, 6, 4), -17., requires_grad=True)
    carrier = SubSpace(inputShape=(6, 4), outputShape=(6, 4), nInputDim=4, nOutputDim=4)
    carrier.set_event(slab)
    readout = Readout()
    end_depth = torch.full((2,), depth, dtype=torch.long)
    model = SimpleNamespace(word_brackets=True, outputSpace=readout,
        _stm_single_S=root, _stm_post_depth=end_depth,
        conceptualSpace=SimpleNamespace(stm=SimpleNamespace(_buffer=field)),
        _tensor_final_end_slots=field, _tensor_final_end_depth=end_depth)
    model._final_end_state = lambda r, d: BasicModel._final_end_state(model, r, d)
    answer = BasicModel._forward_head(model, carrier)
    expected = torch.zeros_like(slab)
    if depth == 1:
        expected[:, 0] = root.detach()
    else:
        expected[:, :3] = field.detach()
    torch.testing.assert_close(readout.seen, expected, rtol=0, atol=0)
    answer.sum().backward()
    assert readout.weight.grad is not None and readout.weight.grad.abs().sum() > 0
    assert slab.grad is None
    assert root.grad is None and field.grad is None


def test_named_concept_head_keeps_its_identity_carrier():
    from Models import BasicModel
    carrier = object()
    class Readout:
        concept_ids = (9,)
        def __call__(self, sub):
            assert sub is carrier
            return sub
    model = SimpleNamespace(word_brackets=False, outputSpace=Readout(), _combine_last_cs_sub=carrier)
    assert BasicModel._forward_head(model, object()) is carrier
