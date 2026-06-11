"""Stage 2 of doc/plans/MeronomyPlan.md: PiLayer2 / SigmaLayer2.

The membership-domain fold kernels (MeronomySpec §4):

  * §10.1 order theorems for RANDOM ADMISSIBLE WEIGHTS: ``pi <= min``
    and ``sigma >= max`` elementwise (binary), ``pi(m) <= m`` /
    ``sigma(m) >= m`` (unary), monotone transport.
  * §10.2 roots: identities at near-identity init (to eps), absorbers
    as theorems for any admissible weights (``pi(A, 0) = 0``,
    ``sigma(A, 1) = 1`` to eps).
  * §10.4 De Morgan exactness with a single shared kernel object:
    ``sigma(m) = 1 - pi_kernel(1 - m)``; sigma owns no weights.
  * Inverse round-trips at the ``m -> 0`` corner under ``EPS_LOG``.
  * Binary folds stay recommender-reversible: the existing
    ``Ops.conjunctionReverse`` / ``Ops.disjunctionReverse`` machinery
    works on fold results from the new path.
  * Guard: no odds transforms anywhere in the new classes (the legacy
    ``(1+x)/(1-x)`` embedding is retired from the meronymic path).
"""
import inspect
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import (PiLayer2, SigmaLayer2, ContractiveInvertibleLinearLayer,
                    Ops, EPS_LOG)

SLACK = 1e-6   # fp slack on the exact order theorems
D = 4


def randomize_admissible(layer, seed, lo=-6.0, hi=1.0):
    """Random raw parameters = random ADMISSIBLE weights: the softplus
    law maps every raw value into the §4 constraint set, so any draw is
    a legal weight configuration."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(lo, hi)


def memberships(*shape, lo=0.05, hi=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(*shape) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# §10.1 order theorems -- random admissible weights, elementwise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_binary_pi_below_min(seed):
    pi = PiLayer2(2 * D, D, blocks=2)
    randomize_admissible(pi, seed)
    left = memberships(16, D, seed=seed + 100)
    right = memberships(16, D, seed=seed + 200)
    z = pi.compose(left, right)
    assert (z <= torch.minimum(left, right) + SLACK).all(), (
        "pi(left, right) <= min(left, right) violated")
    assert (z >= 0).all() and (z <= 1).all()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_binary_sigma_above_max(seed):
    sig = SigmaLayer2(2 * D, D, blocks=2)
    randomize_admissible(sig, seed)
    left = memberships(16, D, seed=seed + 100)
    right = memberships(16, D, seed=seed + 200)
    z = sig.compose(left, right)
    assert (z >= torch.maximum(left, right) - SLACK).all(), (
        "sigma(left, right) >= max(left, right) violated")
    assert (z >= 0).all() and (z <= 1).all()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_unary_contraction_and_extension(seed):
    pi = PiLayer2(D, D)
    sig = SigmaLayer2(D, D)
    randomize_admissible(pi, seed)
    randomize_admissible(sig, seed + 50)
    m = memberships(16, D, seed=seed + 100)
    assert (pi.forward(m) <= m + SLACK).all(), "pi(m) <= m violated"
    assert (sig.forward(m) >= m - SLACK).all(), "sigma(m) >= m violated"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_monotone_transport(seed):
    # W >= 0 plus increasing pointwise maps: m1 <= m2 => fold(m1) <= fold(m2).
    pi = PiLayer2(D, D)
    sig = SigmaLayer2(D, D)
    randomize_admissible(pi, seed)
    randomize_admissible(sig, seed + 50)
    torch.manual_seed(seed + 100)
    m1 = memberships(16, D)
    m2 = m1 + torch.rand(16, D) * (1.0 - m1)      # m2 >= m1, still <= 1
    assert (pi.forward(m1) <= pi.forward(m2) + SLACK).all()
    assert (sig.forward(m1) <= sig.forward(m2) + SLACK).all()


def test_binary_zero_operand_corner_still_ordered():
    # The EPS_LOG floor must not break the acceptance bound at m = 0.
    pi = PiLayer2(2 * D, D, blocks=2)
    randomize_admissible(pi, seed=7)
    left = torch.zeros(4, D)
    right = memberships(4, D, seed=8)
    z = pi.compose(left, right)
    assert (z <= torch.minimum(left, right) + SLACK).all()


# ---------------------------------------------------------------------------
# §10.2 roots: identities and absorbers.
# ---------------------------------------------------------------------------

def test_pi_absorber_is_a_theorem():
    # pi(A, 0) = 0 to eps for ANY admissible weights: the lifted block
    # diagonal (>= 1) alone forces z <= EPS_LOG; every other term only
    # deepens it.
    for seed in (1, 2, 3):
        pi = PiLayer2(2 * D, D, blocks=2)
        randomize_admissible(pi, seed)
        A = memberships(8, D, seed=seed + 10)
        z = pi.compose(A, torch.zeros(8, D))
        assert (z <= 2 * EPS_LOG).all(), "pi(A, 0) must annihilate"
        z = pi.compose(torch.zeros(8, D), A)
        assert (z <= 2 * EPS_LOG).all(), "pi(0, A) must annihilate"


def test_sigma_absorber_is_a_theorem():
    # sigma(A, 1) = 1 to eps for any admissible weights (mirror).
    for seed in (1, 2, 3):
        sig = SigmaLayer2(2 * D, D, blocks=2)
        randomize_admissible(sig, seed)
        A = memberships(8, D, seed=seed + 10)
        z = sig.compose(A, torch.ones(8, D))
        assert (z >= 1.0 - 2 * EPS_LOG).all(), "sigma(A, 1) must absorb"
        z = sig.compose(torch.ones(8, D), A)
        assert (z >= 1.0 - 2 * EPS_LOG).all(), "sigma(1, A) must absorb"


def test_pi_identity_at_init():
    # pi(A, 1) = A to eps at the near-identity init (raw = -5: d ~ 1,
    # offdiag ~ 0.007, b ~ -0.007). Identity is init-exact, not a
    # theorem for arbitrary weights -- the roots do double duty (§4).
    pi = PiLayer2(2 * D, D, blocks=2)
    A = memberships(8, D, lo=0.3, hi=0.9, seed=11)
    ones = torch.ones(8, D)
    assert torch.allclose(pi.compose(A, ones), A, atol=0.05)
    assert torch.allclose(pi.compose(ones, A), A, atol=0.05)


def test_sigma_identity_at_init():
    # sigma(A, 0) = A to eps at init (mirror through the involution).
    sig = SigmaLayer2(2 * D, D, blocks=2)
    A = memberships(8, D, lo=0.3, hi=0.9, seed=12)
    zeros = torch.zeros(8, D)
    assert torch.allclose(sig.compose(A, zeros), A, atol=0.05)
    assert torch.allclose(sig.compose(zeros, A), A, atol=0.05)


# ---------------------------------------------------------------------------
# §10.4 De Morgan exactness: one shared kernel object, two operators.
# ---------------------------------------------------------------------------

def test_de_morgan_exact_with_shared_kernel():
    pi = PiLayer2(2 * D, D, blocks=2)
    sig = SigmaLayer2(kernel=pi)              # SHARES the kernel object
    assert sig.kernel is pi
    randomize_admissible(pi, seed=21)         # randomizing pi moves sigma
    m = memberships(16, 2 * D, seed=22)
    assert torch.equal(sig.forward(m), 1.0 - pi.forward(1.0 - m))
    left = memberships(16, D, seed=23)
    right = memberships(16, D, seed=24)
    assert torch.equal(sig.compose(left, right),
                       1.0 - pi.compose(1.0 - left, 1.0 - right))


def test_sigma_owns_no_weights():
    # One kernel + one involution: sigma's parameters ARE the kernel's.
    pi = PiLayer2(D, D)
    sig = SigmaLayer2(kernel=pi)
    assert set(id(p) for p in sig.parameters()) == \
           set(id(p) for p in pi.parameters())
    # A standalone sigma owns exactly one kernel's worth of weights.
    solo = SigmaLayer2(D, D)
    assert isinstance(solo.kernel, PiLayer2)
    assert set(id(p) for p in solo.parameters()) == \
           set(id(p) for p in solo.kernel.parameters())


def test_sigma_kernel_kwargs_exclusive():
    pi = PiLayer2(D, D)
    with pytest.raises(ValueError):
        SigmaLayer2(D, D, kernel=pi, stable=True)
    with pytest.raises(ValueError):
        SigmaLayer2()                          # neither shape nor kernel


# ---------------------------------------------------------------------------
# Inverse round-trips, including the m -> 0 corner under EPS_LOG.
# ---------------------------------------------------------------------------

def test_unary_roundtrip_interior():
    pi = PiLayer2(D, D)
    randomize_admissible(pi, seed=31, lo=-5.0, hi=-0.5)
    m = memberships(16, D, lo=0.05, hi=0.95, seed=32)
    rec = pi.reverse(pi.forward(m))
    err = torch.norm(rec - m) / torch.norm(m)
    assert err < 1e-4, f"interior roundtrip error {err:.2e}"


def test_pi_roundtrip_at_zero_corner_under_eps_log():
    # m = 0 is floored to EPS_LOG by the forward; the exact inverse
    # recovers THE FLOOR (not 0, not garbage). The reverse's own floor
    # sits at dtype-tiny precisely so the fold result (far below
    # EPS_LOG) passes through the log un-truncated.
    pi = PiLayer2(D, D, stable=True)
    randomize_admissible(pi, seed=41, lo=-5.0, hi=-0.5)
    m = torch.tensor([[0.0, 1e-8, 0.5, 1.0]])
    expected = m.clamp(EPS_LOG, 1.0)           # [1e-6, 1e-6, 0.5, 1.0]
    z = pi.forward(m)
    assert (z > 0).all() and torch.isfinite(z).all()
    rec = pi.reverse(z)
    assert torch.allclose(rec, expected, rtol=1e-3, atol=1e-9), (
        f"corner roundtrip: {rec.tolist()} != {expected.tolist()}")


def test_sigma_roundtrip_at_zero_corner():
    # sigma's m -> 0 corner maps through the involution to the kernel's
    # BENIGN 1-corner (log 1 = 0), so the round-trip is exact there.
    # (The saturated m -> 1 corner is the involuted image of the
    # kernel's EPS_LOG floor: sigma values land within ~1e-7 of 1 and
    # the final ``1 - pi`` subtraction quantizes them away in fp32 --
    # the kernel stays exact; the involution loses the resolution.
    # Exactness at THAT corner is a float64 property, deliberately not
    # pinned; the plan's corner requirement is m -> 0 under EPS_LOG,
    # covered for the kernel above and for sigma here.)
    sig = SigmaLayer2(D, D, stable=True)
    randomize_admissible(sig, seed=42, lo=-5.0, hi=-0.5)
    m = torch.tensor([[0.0, 1e-8, 0.5, 0.9]])
    rec = sig.reverse(sig.forward(m))
    assert torch.allclose(rec, m, rtol=1e-3, atol=1e-5)


def test_reverse_is_total_on_degenerate_input():
    # z = 0 (below any fold image) must not NaN: the tiny-floor keeps
    # the solve finite; the output clamp keeps it in the chart.
    pi = PiLayer2(D, D)
    randomize_admissible(pi, seed=43, lo=-5.0, hi=-0.5)
    rec = pi.reverse(torch.zeros(2, D))
    assert torch.isfinite(rec).all()
    assert (rec >= 0).all() and (rec <= 1).all()


def test_gradients_flow_through_fold():
    pi = PiLayer2(2 * D, D, blocks=2)
    left = memberships(8, D, seed=51)
    right = memberships(8, D, seed=52)
    pi.compose(left, right).sum().backward()
    grads = [p.grad for p in pi.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# Binary folds stay recommender-reversible: the existing codebook-search
# inverse (Ops.conjunctionReverse / disjunctionReverse) works on results
# from the new path, because the §10.1 bounds are exactly its
# feasibility laws (intersection: operands dominate the result; union:
# operands are dominated by it).
# ---------------------------------------------------------------------------

def _codebook_with_distractors(A, B, z, dominate):
    """A, B plus 4 distractor rows made infeasible for the recommender:
    a zero coordinate (cannot dominate z) or a saturated one (cannot be
    dominated by z)."""
    distract = memberships(4, D, lo=0.55, hi=0.95, seed=61)
    if dominate:
        distract[:, 0] = 0.0     # z[0] > 0 -> cannot dominate z
    else:
        distract[:, 0] = 1.0     # z[0] < 1 -> cannot sit below z
    return torch.cat([A.unsqueeze(0), B.unsqueeze(0), distract], dim=0)


def test_binary_pi_recommender_reversible():
    pi = PiLayer2(2 * D, D, blocks=2)
    randomize_admissible(pi, seed=62, lo=-5.0, hi=-0.5)
    A = memberships(D, lo=0.55, hi=0.95, seed=63)
    B = memberships(D, lo=0.55, hi=0.95, seed=64)
    z = pi.compose(A, B)
    assert (z > 0).all() and (z < 1).all()
    W_cb = _codebook_with_distractors(A, B, z, dominate=True)
    x1, x2 = Ops.conjunctionReverse(z, None, W_cb, monotonic=True)
    # Feasibility law: recommended operands dominate the fold result.
    assert bool(Ops.partOf(z, x1)) and bool(Ops.partOf(z, x2))
    # The only dominating learned rows are the true operands; the
    # recommender must find them (x2 may fall back to the T sentinel).
    assert any(torch.allclose(x1, r) for r in (A, B)), (
        "x1 should recover a true operand")
    assert any(torch.allclose(x2, r)
               for r in (A, B, torch.ones(D))), "x2 outside codebook"


def test_binary_sigma_recommender_reversible():
    sig = SigmaLayer2(2 * D, D, blocks=2)
    randomize_admissible(sig, seed=65, lo=-5.0, hi=-0.5)
    A = memberships(D, lo=0.55, hi=0.95, seed=66)
    B = memberships(D, lo=0.55, hi=0.95, seed=67)
    z = sig.compose(A, B)
    assert (z > 0).all() and (z < 1).all()
    W_cb = _codebook_with_distractors(A, B, z, dominate=False)
    x1, x2 = Ops.disjunctionReverse(z, None, W_cb, monotonic=True)
    # Feasibility law (union): x1 is the largest part sitting below z.
    assert bool(Ops.partOf(x1, z))
    assert any(torch.allclose(x1, r) for r in (A, B)), (
        "x1 should recover a true operand")
    assert any(torch.allclose(x2, r)
               for r in (A, B, torch.zeros(D), torch.ones(D))), (
        "x2 outside augmented codebook")


# ---------------------------------------------------------------------------
# Structure guards.
# ---------------------------------------------------------------------------

def test_no_odds_transforms_on_the_new_path():
    # The legacy odds embedding (1+x)/(1-x) is retired from the
    # meronymic path: the new classes neither define nor inherit the
    # odds helpers, and their compute methods never invoke them.
    for cls in (PiLayer2, SigmaLayer2):
        inst = cls(D, D)
        for name in ("_to_mult", "_from_mult"):
            assert name not in cls.__dict__
            assert not hasattr(inst, name)
        for meth in ("forward", "reverse", "compose"):
            src = inspect.getsource(getattr(cls, meth))
            for token in ("_to_mult", "_from_mult", "atanh", "tanh"):
                assert token not in src, (
                    f"{cls.__name__}.{meth} touches the retired "
                    f"odds/atanh path ({token})")


def test_inner_layer_is_contractive():
    pi = PiLayer2(2 * D, D, blocks=2)
    assert isinstance(pi.layer, ContractiveInvertibleLinearLayer)
    assert pi.layer.blocks == 2
    # Composite-layer contract: the inner layer is registered so the
    # ergodic interface (set_sigma etc.) dispatches into it.
    assert pi.layer in pi.layers
    sig = SigmaLayer2(kernel=pi)
    assert sig.kernel in sig.layers


def test_compose_requires_binary_form():
    pi = PiLayer2(D, D)                        # unary: blocks=1
    with pytest.raises(RuntimeError):
        pi.compose(memberships(2, D, seed=71), memberships(2, D, seed=72))


def test_compose_is_forward_on_concat():
    pi = PiLayer2(2 * D, D, blocks=2)
    randomize_admissible(pi, seed=73)
    left = memberships(8, D, seed=74)
    right = memberships(8, D, seed=75)
    assert torch.equal(pi.compose(left, right),
                       pi.forward(torch.cat([left, right], dim=-1)))
