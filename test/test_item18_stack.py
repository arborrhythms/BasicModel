"""Round B: the shared bounded stack, before its language consumers."""
from pathlib import Path
import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)


def test_79_stack_learns_the_four_disjoint_word_combinations():
    from SigmaPiStack import SigmaPiStack
    # Each word owns a coordinate; the omitted last word has no input evidence.
    present = torch.zeros(4, 6, 2)
    for row, pair in enumerate(((0, 1), (2, 3), (0, 3), (2, 1))):
        present[row, list(pair), 0] = 1
    target = present.clone()
    target[torch.arange(4), torch.tensor([4, 4, 5, 5]), 0] = 1
    x, y = present.flatten(1), target.flatten(1)
    design = torch.cat((x, torch.ones(4, 1)), -1).double()
    linear = torch.linalg.lstsq(design, y.double(), driver='gelsd').solution
    linear_read = (design @ linear).reshape_as(target)
    torch.testing.assert_close(linear_read[:, 4:, 0], torch.full((4, 2), .5, dtype=torch.float64))
    # The exact optimum ties. Do not treat floating-point noise in that tie
    # as a learned distinction between the two left-out words.
    linear_choice = (linear_read[:, 5, 0] > linear_read[:, 4, 0] + 1e-12).long()
    expected = torch.tensor([0, 0, 1, 1])
    assert int((linear_choice == expected).sum()) < 4
    stack = SigmaPiStack(12, 12, hidden=64, pairs=1, seed=0)
    assert torch.equal(stack(x), torch.zeros_like(y))
    optimizer = torch.optim.Adam(stack.parameters(), lr=.01)
    for _ in range(200):
        optimizer.zero_grad()
        loss = (stack(x) - y).square().mean()
        loss.backward()
        optimizer.step()
    read = stack(x).reshape_as(target)
    assert torch.equal(read[:, 4:, 0].argmax(-1), expected)
    assert (read[torch.arange(4), torch.tensor([4, 4, 5, 5]), 0] > .5).all()
    assert (read[torch.arange(4), torch.tensor([5, 5, 4, 4]), 0] < .5).all()
    print(dict(stack_correct=int((read[:, 4:, 0].argmax(-1) == expected).sum()),
               linear_correct=int((linear_choice == expected).sum()),
               presentations=200, output=read[:, 4:, 0].detach().tolist()))


def test_79_stages_are_the_shared_pi_and_sigma_layers():
    from Layers import PiLayer, SigmaLayer
    from SigmaPiStack import SigmaPiStack
    stack = SigmaPiStack(2, 2, hidden=2, pairs=1)
    assert isinstance(stack.stages[0], PiLayer)
    assert isinstance(stack.stages[1], SigmaLayer)
    pi_weight = stack.stages[0].layer.W
    sigma_weight = stack.stages[1].layer.W
    with torch.no_grad():
        pi_weight.copy_(torch.tensor([[.1, -.2], [0., .1]]).T)
        sigma_weight.copy_(torch.tensor([[.2, 0.], [-.1, .3]]).T)
    x = torch.tensor([[.3, .8], [0., 1.], [1., 0.]])
    chart = x.clamp(-1+1e-6, 1-1e-6).atanh()
    middle = (chart @ pi_weight).tanh()
    wanted = (middle @ sigma_weight).clamp(0, 1)
    torch.testing.assert_close(stack(x), wanted)
    stack(x).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in stack.parameters() if p.grad is not None)
    assert pi_weight.grad is not None and sigma_weight.grad is not None
    assert (stack(x) >= 0).all() and (stack(x) <= 1).all()


def test_79_stack_construction_leaves_the_global_draw_unchanged(tmp_path):
    from SigmaPiStack import SigmaPiStack
    before = torch.random.get_rng_state().clone()
    stack = SigmaPiStack(12, 12, hidden=64, pairs=2)
    assert len(stack.stages) == 4
    assert torch.equal(torch.random.get_rng_state(), before)
    from recon_bench import _build_model
    config = Path('data/MM_xor.xml').read_text()
    off = tmp_path/'stacks-off.xml'
    off.write_text(config.replace('<ConceptualSpace>', '<ConceptualSpace><expectHidden>0</expectHidden>'))
    state = torch.random.get_rng_state().clone()
    first, *_ = _build_model('data/MM_xor.xml')
    next_draw = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(state)
    second, *_ = _build_model(str(off))
    assert first._concept_owner().expect_stack is not None
    assert second._concept_owner().expect_stack is None
    assert torch.equal(torch.random.get_rng_state(), next_draw)
    first.End(); second.End()


def test_80_numeric_certificate_keeps_its_bar_with_the_stack_built():
    from numeric_xor import certificate
    model, prediction, error = certificate()
    stack = model._concept_owner().expect_stack
    assert stack is not None and stack.pairs == 1
    # The numeric field is observed without the retired intuition forecast.
    assert model._concept_owner()._observed_knowing[:4].shape == (4, model._concept_owner().nVectors, 2)
    assert error < 1e-6
    model.End()
