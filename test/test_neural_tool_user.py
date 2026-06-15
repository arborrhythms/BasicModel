"""NeuralToolUser hard-parse executor (two-pass, single-point divergence).

doc/plans/NeuralToolUser.md, user-refined design. Pass 1 runs the greedy
Viterbi tiling per level and saves the route; pass 2 replays to a random
divergence level L, forces a DIFFERENT op at one fired location there
(temperature 0..1 escalated until the draw differs), and re-decides after.
The chooser is trained by the single-point advantage between the two ops
at L (log-probs read off ONE live cross-product distribution). These tests
drive the executor with a deterministic fold stepper (it is not yet wired
into the live LanguageLayer.compose).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

from Language import NeuralToolUser, RouteStats


class _FoldStepper:
    """A minimal reduction stepper: copy (identity) or reduce adjacent
    pairs with one of R_reduce ops. Scores are fixed linear projections
    (requires_grad so we can check the chooser gradient)."""

    def __init__(self, R_reduce=2, D=3, seed=0, reduce_bias=5.0):
        g = torch.Generator().manual_seed(seed)
        self.R_copy = 1
        self.R_reduce = R_reduce
        self.D = D
        # Bias reduce scores above copy so the greedy tiling always folds
        # (folding is a property of the scores, not the executor; the bias
        # makes the fold deterministic for the test).
        self.reduce_bias = float(reduce_bias)
        self.copy_w = torch.randn(D, self.R_copy, generator=g, requires_grad=True)
        self.reduce_w = torch.randn(2 * D, self.R_reduce, generator=g,
                                    requires_grad=True)

    def score(self, x):
        B, N, D = x.shape
        copy_score = torch.einsum('bnd,dc->bnc', x, self.copy_w)
        if N >= 2:
            pairs = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)
            reduce_score = (torch.einsum('bpe,er->bpr', pairs, self.reduce_w)
                            + self.reduce_bias)
        else:
            reduce_score = x.new_zeros(B, 0, self.R_reduce)
        return copy_score, reduce_score

    def _reduce_op(self, l, r, op):
        return (l + r) / 2 if op == 0 else l - r

    def apply(self, x, copy_mask, reduce_mask):
        B, N, D = x.shape
        cm, rm = copy_mask[0], reduce_mask[0]
        out, t = [], 0
        while t < N:
            if t < N - 1 and rm.numel() > 0 and rm[t].sum() > 0:
                out.append(self._reduce_op(x[0, t], x[0, t + 1], int(rm[t].argmax())))
                t += 2
            else:
                out.append(x[0, t])
                t += 1
        return torch.stack(out, dim=0).unsqueeze(0)

    def done(self, x):
        return x.shape[1] <= 1


def test_greedy_folds_to_single_state_and_saves_route():
    stepper = _FoldStepper(seed=1)
    x = torch.randn(1, 6, stepper.D)
    ntu = NeuralToolUser(max_levels=16)
    final_x, stats = ntu.parse_greedy(x, stepper)
    assert final_x.shape[1] == 1                      # folded to one state
    assert stats.step_count >= 1
    assert len(stats.route) == stats.step_count
    assert torch.isfinite(stats.log_prob_sum).all()
    assert (stats.entropy_sum >= -1e-5).all()


def test_explore_diverges_at_one_op_and_differs():
    stepper = _FoldStepper(seed=2)
    x = torch.randn(1, 6, stepper.D)
    ntu = NeuralToolUser(max_levels=16)
    _, saved = ntu.parse_greedy(x, stepper)
    gen = torch.Generator().manual_seed(7)
    out = ntu.parse_explore(x, stepper, saved, generator=gen)
    assert out is not None
    _, stats_b, info = out
    # The divergence changed exactly one op at one fired location.
    assert info["old_op"] != info["new_op"]
    assert stats_b.diverge_level == info["L"]
    # The two competing log-probs come off the same live dist and differ.
    assert torch.isfinite(info["logp_a"]).all()
    assert torch.isfinite(info["logp_b"]).all()
    assert not torch.allclose(info["logp_a"], info["logp_b"])


def test_explore_logprob_is_differentiable_into_scores():
    stepper = _FoldStepper(seed=3)
    x = torch.randn(1, 6, stepper.D)
    ntu = NeuralToolUser(max_levels=16)
    _, saved = ntu.parse_greedy(x, stepper)
    gen = torch.Generator().manual_seed(11)
    out = ntu.parse_explore(x, stepper, saved, generator=gen)
    assert out is not None
    _, _, info = out
    # The single-point policy term must backprop into the chooser scores.
    (info["logp_b"] - info["logp_a"]).sum().backward()
    assert stepper.reduce_w.grad is not None
    assert torch.isfinite(stepper.reduce_w.grad).all()


def test_explore_is_deterministic_under_a_fixed_generator():
    stepper = _FoldStepper(seed=4)
    x = torch.randn(1, 7, stepper.D)
    ntu = NeuralToolUser(max_levels=16)
    _, saved = ntu.parse_greedy(x, stepper)
    o1 = ntu.parse_explore(x, stepper, saved,
                           generator=torch.Generator().manual_seed(99))
    o2 = ntu.parse_explore(x, stepper, saved,
                           generator=torch.Generator().manual_seed(99))
    assert o1[2]["L"] == o2[2]["L"]
    assert o1[2]["new_op"] == o2[2]["new_op"]


def test_no_divergence_when_single_op():
    # R_reduce=1 -> no fired location has >= 2 legal ops -> no divergence.
    stepper = _FoldStepper(R_reduce=1, seed=5)
    x = torch.randn(1, 6, stepper.D)
    ntu = NeuralToolUser(max_levels=16)
    _, saved = ntu.parse_greedy(x, stepper)
    out = ntu.parse_explore(x, stepper, saved,
                            generator=torch.Generator().manual_seed(3))
    assert out is None


def test_two_pass_loss_advantage_moves_logprob_the_right_way():
    # The central credit-assignment rule: minimizing the loss pushes a
    # below-baseline route's op log-prob UP. loss_choose is linear in the
    # log-probs with detached advantages, so d(loss)/d(logp_b) == lam*adv_b.
    stepper = _FoldStepper(seed=6)
    x = torch.randn(1, 6, stepper.D)
    ntu = NeuralToolUser(max_levels=16)

    # A task loss that distinguishes the two parses (sum of the folded state).
    def task_loss(final_x):
        return final_x.reshape(final_x.shape[0], -1).pow(2).mean(-1)

    loss, diag = ntu.two_pass_loss(
        x, stepper, task_loss, lam_choose=1.0,
        generator=torch.Generator().manual_seed(21))
    assert diag["diverged"]
    g_b = torch.autograd.grad(loss, diag["logp_b"], retain_graph=True)[0]
    # Gradient on logp_b equals lam_choose * adv_b (B=1). Sign of the
    # gradient is the sign of adv_b: a better (negative-adv) route gets a
    # negative gradient -> gradient descent raises its log-prob.
    assert torch.allclose(g_b, diag["adv_b"], atol=1e-5)


def test_two_pass_loss_trains_tools_when_no_divergence():
    stepper = _FoldStepper(R_reduce=1, seed=7)   # single op -> no divergence
    x = torch.randn(1, 6, stepper.D)
    ntu = NeuralToolUser(max_levels=16)

    def task_loss(final_x):
        return final_x.reshape(final_x.shape[0], -1).pow(2).mean(-1)

    loss, diag = ntu.two_pass_loss(
        x, stepper, task_loss, generator=torch.Generator().manual_seed(2))
    assert diag["diverged"] is False
    # Still differentiable into the tools (here the inputs / op path).
    assert torch.isfinite(loss)


def test_route_stats_accumulates():
    s = RouteStats()
    s.add(torch.tensor([1.0]), torch.tensor([0.5]))
    s.add(torch.tensor([2.0]), torch.tensor([0.25]))
    assert s.step_count == 2
    assert torch.allclose(s.log_prob_sum, torch.tensor([3.0]))
    assert torch.allclose(s.entropy_sum, torch.tensor([0.75]))
