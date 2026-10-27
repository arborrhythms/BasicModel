"""Unknown verb groups keep identity and optimizer ownership without dense bases."""
import copy

import torch

import MeaningKernels
from MeaningKernels import VerbGroups, householder


def dense_reference(module, group):
    size = module.vectors.shape[-1]
    eye = torch.eye(size, dtype=module.vectors.dtype, device=module.vectors.device)
    vectors = module.vectors.index_select(0, group.clamp(0, module.capacity-1).reshape(-1))
    selected = householder(vectors.reshape(*group.shape, *module.vectors.shape[1:]))
    return torch.where((group >= 0)[..., None, None], selected, eye)


def test_native_unknown_group_shape_builds_no_discarded_basis(monkeypatch):
    # The retained native workload has B=28 and 896 meaning coordinates:
    # its two evidence lanes each have width 448, with one operand per row.
    groups = VerbGroups(448)
    group = torch.full((28, 1), -1, dtype=torch.long)
    calls = []
    original = MeaningKernels.householder
    def observed(vectors):
        calls.append(tuple(vectors.shape))
        return original(vectors)
    monkeypatch.setattr(MeaningKernels, 'householder', observed)
    result = groups(group)
    torch.testing.assert_close(result, torch.eye(448).expand(28, 1, 448, 448), rtol=0, atol=0)
    result.sum().backward()
    assert groups.vectors.grad is not None
    assert torch.count_nonzero(groups.vectors.grad) == 0
    assert calls == [], calls
    assert result.untyped_storage().nbytes() == 448*448*result.element_size()


def test_unknown_group_retains_dense_reference_gradients_and_optimizer_steps():
    groups = VerbGroups(4, capacity=2).double()
    with torch.no_grad():
        groups.vectors.add_(torch.arange(groups.vectors.numel(), dtype=torch.float64)
                            .reshape_as(groups.vectors)/100)
    reference = copy.deepcopy(groups)
    opt = torch.optim.AdamW(groups.parameters(), lr=.01)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=.01)
    coefficient = torch.arange(32, dtype=torch.float64).reshape(2, 4, 4)/32
    # A known/mixed update first leaves optimizer moments. Unknown rows must
    # then deliver zeros, rather than None, so the second update is identical.
    for group in (torch.tensor([0, -1]), torch.tensor([-1, -1])):
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        actual, expected = groups(group), dense_reference(reference, group)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (actual*coefficient).sum().backward()
        (expected*coefficient).sum().backward()
        assert groups.vectors.grad is not None and reference.vectors.grad is not None
        torch.testing.assert_close(groups.vectors.grad, reference.vectors.grad, rtol=0, atol=0)
        opt.step()
        ref_opt.step()
        torch.testing.assert_close(groups.vectors, reference.vectors, rtol=0, atol=0)
        for key, value in opt.state[groups.vectors].items():
            torch.testing.assert_close(value, ref_opt.state[reference.vectors][key], rtol=0, atol=0)
