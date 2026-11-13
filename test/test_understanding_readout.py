"""A numeric answer reads its final field, with the shared output gradient."""
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize('readings', [1, 3])
def test_answer_reads_final_field_with_live_gradient(readings):
    from Models import BasicModel
    from Understanding import ConceptualField
    from Spaces import SubSpace
    class Readout:
        def __init__(self):
            self.field_readout = torch.nn.Linear(4, 1, bias=False)
            with torch.no_grad(): self.field_readout.weight.fill_(1.)
            self.subspace = SubSpace.from_tensor(torch.zeros(2, 1, 1),
                inputShape=(1, 1), outputShape=(1, 1), nInputDim=1, nOutputDim=1)
        def read_field(self, field, owner):
            self.seen = field
            return self.field_readout(field.evidence.amax(2).permute(1, 0, 2).flatten(1))[:, None]
    evidence = torch.rand(2, 2, readings, 2, requires_grad=True)
    field = ConceptualField(torch.tensor([9, 10]), evidence, None, None, None)
    irrelevant = torch.full((2, 6, 4), -17., requires_grad=True)
    readout = Readout()
    model = SimpleNamespace(outputSpace=readout, _concept_owner=lambda: None)
    answer = BasicModel._forward_head(model, irrelevant, field=field).materialize()
    assert readout.seen is field
    torch.testing.assert_close(answer.reshape(2), evidence.amax(2).sum((0, 2)))
    answer.sum().backward()
    assert readout.field_readout.weight.grad.abs().sum() > 0
    assert evidence.grad is not None and evidence.grad.abs().sum() > 0
    assert irrelevant.grad is None
