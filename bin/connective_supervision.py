"""Corpus-scale connective supervision (Phase R5).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
decision 8 + §5.5 + §8 R5. ``A AND B`` and ``A OR B`` are
surface-indiscriminable, so the role-collapsed grammar does NOT give them
distinct categories; they are discriminated by the slot-0 OPERATOR
SUPERPOSITION over {conjunction, disjunction}. This module supplies the
truth / consequence signal that makes that superposition load-bearing: a
differentiable operator-superposition (the gradient analogue of
``perceptual_analyzer.soft_operator_compose``) plus an MSE loss against the
observed consequence ``y``, so supervising on a corpus of (operands,
consequence) pairs drives the slot-0 distribution to the connective whose
truth table matches -- even when the operands (the surface) are identical
between the AND and OR corpora.
"""

import torch


def soft_connective_compose(dist, a, b, op_names, classes=None):
    """Tensor-weighted operator superposition over ``op_names`` -- the
    differentiable analogue of ``perceptual_analyzer.soft_operator_compose``.

    ``dist`` is a 1-D weight tensor aligned with ``op_names`` (gradient
    flows to it, unlike the float-coerced ``soft_operator_compose``). A
    one-hot ``dist`` reduces to that operator's hard compose, preserving the
    typed grammar as the limit.
    """
    if classes is None:
        from Language import GRAMMAR_LAYER_CLASSES as classes
    out = None
    for k, name in enumerate(op_names):
        y = classes[name]().compose(a, b)
        contrib = dist[k] * y
        out = contrib if out is None else out + contrib
    return out


def connective_truth_loss(logits, a, b, y, op_names, classes=None):
    """MSE between the slot-0 operator superposition's composed prediction
    and the observed consequence ``y``. ``logits`` are the (learnable)
    slot-0 operator logits; ``softmax(logits)`` is the superposition."""
    dist = torch.softmax(logits, dim=-1)
    pred = soft_connective_compose(dist, a, b, op_names, classes=classes)
    return ((pred - y) ** 2).mean()


def learn_connective_distribution(a, b, y,
                                  op_names=("conjunction", "disjunction"),
                                  steps=500, lr=0.1, seed=0, classes=None):
    """Fit the slot-0 operator superposition to a truth/consequence corpus.

    Minimizes :func:`connective_truth_loss` over ``steps`` Adam updates on
    the operator logits, then returns the learned distribution as
    ``{op_name: weight}``. The operands ``a`` / ``b`` are the surface (the
    same for AND and OR); ``y`` is the consequence that discriminates them.
    """
    torch.manual_seed(int(seed))
    logits = torch.zeros(len(op_names), requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = connective_truth_loss(logits, a, b, y, op_names, classes=classes)
        loss.backward()
        opt.step()
    dist = torch.softmax(logits.detach(), dim=-1)
    return {name: float(dist[k]) for k, name in enumerate(op_names)}
