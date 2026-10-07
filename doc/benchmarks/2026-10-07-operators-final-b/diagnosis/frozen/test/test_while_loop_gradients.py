"""Regression for the ``torch.while_loop`` autograd defect (torch 2.14 and
2.15 nightlies): per-trip checkpoints inherit the initial carry's
``requires_grad``, so a carry entering as plain zeros cuts the gradient
chain across trips (parameters used in the body are credited from the
last trip only; closures loaded mid-loop get nothing).  The model wraps
every loop's carries with ``Models._carries_with_grad``; these cases pin
the defect and the fix against a plain Python loop."""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(_ROOT / "bin"))

from Models import _carries_with_grad  # noqa: E402


def _python_loop(cond, body, carried):
    c = tuple(carried)
    while bool(cond(*c)):
        c = tuple(body(*c))
    return c


def _hop(cond, body, carried):
    return torch.while_loop(cond, body, _carries_with_grad(carried))


def _grads(build, loop):
    torch.manual_seed(0)
    leaves, fn = build()
    out = fn(loop)
    return [float(g.abs().sum()) for g in torch.autograd.grad(out.sum(), leaves)]


def _closure_in_body():
    w = torch.randn(4, 4, requires_grad=True)
    return [w], lambda loop: loop(
        lambda t, x: t < 3, lambda t, x: (t + 1, torch.tanh(x @ w).clone()),
        (torch.tensor(0), torch.ones(2, 4)))[1]


def _closure_loaded_mid_loop():
    p = torch.randn(2, 4, requires_grad=True)

    def fn(loop):
        def body(t, x):
            x = torch.where(t == 1, p, x)
            return t + 1, torch.tanh(x * 1.5).clone()
        return loop(lambda t, x: t < 4, body, (torch.tensor(0), torch.zeros(2, 4)))[1]
    return [p], fn


def _loaded_then_accumulated():
    p = torch.randn(2, 4, requires_grad=True)

    def fn(loop):
        def body(t, x, acc):
            x = torch.tanh(torch.where(t == 0, p, x) * 1.5)
            return t + 1, x.clone(), (acc + x.square().sum(-1)).clone()
        return loop(lambda t, x, acc: t < 3, body,
                    (torch.tensor(0), torch.zeros(2, 4), torch.zeros(2)))[2]
    return [p], fn


def test_while_loop_gradients_match_a_python_loop_with_grad_carries():
    for build in (_closure_in_body, _closure_loaded_mid_loop, _loaded_then_accumulated):
        ref = _grads(build, _python_loop)
        got = _grads(build, _hop)
        assert all(abs(a - b) < 1e-4 * max(1.0, abs(a)) for a, b in zip(ref, got)), (build.__name__, ref, got)


def test_bare_while_loop_still_shows_the_defect_or_is_fixed_upstream():
    """When this stops failing to differ, the upstream defect is fixed and
    the wrapper can retire; until then the wrapper is required."""
    ref = _grads(_closure_loaded_mid_loop, _python_loop)
    bare = _grads(_closure_loaded_mid_loop, torch.while_loop)
    if abs(ref[0] - bare[0]) < 1e-6:
        import warnings
        warnings.warn("torch.while_loop autograd now matches a Python loop: "
                      "Models._carries_with_grad can retire")


def test_no_grad_leaves_carries_untouched():
    with torch.no_grad():
        c = (torch.tensor(0), torch.zeros(2))
        out = _carries_with_grad(c)
    assert out[1] is c[1]


def test_release_loop_checkpoints_drops_the_node_payload():
    """After a loop's backward has run, ``_release_loop_checkpoints`` drops
    the stacked per-trip checkpoints its autograd node keeps (the node
    outlives a brick through the C++ graph); the count is at least one
    for a node that still exists."""
    from Models import _release_loop_checkpoints
    w = torch.randn(4, 4, requires_grad=True)
    out = _hop(lambda t, x: t < 3, lambda t, x: (t + 1, torch.tanh(x @ w).clone()),
               (torch.tensor(0), torch.ones(2, 4)))[1]
    out.sum().backward()
    node = out.grad_fn
    while node is not None and type(node).__name__ != "WhileLoopAutogradOpBackward":
        node = node.next_functions[0][0] if node.next_functions else None
    assert node is not None and getattr(node, "fw_outputs", None) is not None
    assert _release_loop_checkpoints((out,)) >= 1
    assert getattr(node, "fw_outputs", None) is None


def test_release_leaves_a_foreign_pending_graph_intact():
    """The release walks only the given roots' graphs: a loop whose
    backward has not run yet, held elsewhere, keeps its checkpoints."""
    from Models import _release_loop_checkpoints
    w = torch.randn(4, 4, requires_grad=True)
    def loop():
        return _hop(lambda t, x: t < 3, lambda t, x: (t + 1, torch.tanh(x @ w).clone()),
                    (torch.tensor(0), torch.ones(2, 4)))[1]
    done = loop(); done.sum().backward()
    pending = loop()                                        # backward not run yet
    released = _release_loop_checkpoints((done,))
    assert released >= 1
    pending.sum().backward()                                # still works
    assert w.grad is not None
