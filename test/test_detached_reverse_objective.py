"""Gradient-boundary tests for stable construction/reconstruction training."""

import torch

from Language import (
    BinaryStructuredReductionLayer,
    ReconstructionStack,
    ReverseConstructionChooser,
    UnaryStructuredLayer,
)


class _UnaryScale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return torch.tanh(x * self.scale)


class _BinaryMix(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, left, right):
        return torch.tanh(self.scale * left + (1.0 - self.scale) * right)


def _teacher_stack():
    stack = ReconstructionStack(batch=2, max_depth=16)
    leaves = torch.randn(2, 4, 7, device="cpu")
    part_ids = torch.tensor(
        [[[[1]], [[2]], [[3]], [[-1]]],
         [[[4]], [[5]], [[-1]], [[-1]]]],
        device="cpu").squeeze(-1)
    part_mask = part_ids >= 0
    stack.store_leaves(leaves)
    stack.store_word_parts(part_ids, part_mask)
    stack.prepare_choices(
        2, 14, device="cpu", unary_rule_ids=(1, 2),
        binary_rule_ids=(3, 4))
    stack.record_choice(
        0, torch.tensor([3, 4]), arity=2,
        mask=torch.tensor([True, True]))
    stack.record_choice(
        2, torch.tensor([1, 2]), arity=1,
        mask=torch.tensor([True, True]))
    return stack


def test_reverse_construction_loss_stops_at_sentence_idea():
    stack = _teacher_stack()
    chooser = ReverseConstructionChooser(
        idea_dim=9, n_rules=6, max_words=4, max_steps=14,
        leaf_dim=7, hidden=16).to("cpu")
    idea = torch.randn(2, 9, device="cpu", requires_grad=True)

    loss, terms = chooser.loss(idea, stack, surface_weight=1.0)
    assert loss is not None and torch.isfinite(loss)
    assert set(terms) == {"reverse_kind", "reverse_rule", "reverse_surface"}
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()

    assert idea.grad is None, "reverse supervision must stopgrad(S)"
    grads = [p.grad for p in chooser.parameters() if p.grad is not None]
    assert grads and any(bool(g.abs().sum() > 0) for g in grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def test_fixed_forward_loss_slab_preserves_chooser_gradient():
    stack = ReconstructionStack(batch=1, max_depth=8)
    stack.prepare_choices(
        1, 4, device="cpu", unary_rule_ids=(1, 2),
        binary_rule_ids=(3, 4))
    parameter = torch.nn.Parameter(torch.tensor(0.25, device="cpu"))
    stack.record_choice(
        1, torch.tensor([3]), arity=2, mask=torch.tensor([True]),
        local_structural_loss=(2.0 * parameter).reshape(1))
    loss = stack.forward_loss()
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert torch.allclose(parameter.grad, torch.tensor(2.0))


def test_unary_local_contrast_is_bounded_and_truncated():
    layer = UnaryStructuredLayer(
        d_model=4, ops=[_UnaryScale(0.5), _UnaryScale(1.5)],
        chooser="anchordot").to("cpu")
    layer.local_objective_enabled = True
    child = torch.randn(2, 1, 4, device="cpu", requires_grad=True)
    _hard, _soft, routing = layer(child)
    loss = routing["local_structural_loss"].mean()
    assert 0.0 <= float(loss.detach()) <= 1.0
    loss.backward()
    assert child.grad is None
    assert layer.apply_anchor.grad is not None
    assert torch.isfinite(layer.apply_anchor.grad).all()


def test_binary_local_contrast_is_bounded_and_truncated():
    layer = BinaryStructuredReductionLayer(
        d_model=4, ops=[_BinaryMix(0.25), _BinaryMix(0.75)],
        chooser="anchordot").to("cpu")
    layer.local_objective_enabled = True
    children = torch.randn(2, 2, 4, device="cpu", requires_grad=True)
    _hard, _soft, routing = layer(children)
    loss = routing["local_structural_loss"].mean()
    assert 0.0 <= float(loss.detach()) <= 1.0
    loss.backward()
    assert children.grad is None
    assert layer.reduce_anchor.grad is not None
    assert torch.isfinite(layer.reduce_anchor.grad).all()
