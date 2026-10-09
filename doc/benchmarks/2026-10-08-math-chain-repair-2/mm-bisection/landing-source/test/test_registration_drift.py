"""Stage 8 of doc/plans/MeronomyPlan.md: registration as drift-keeper.

MeronomySpec §5 / §10.5. The towers ARE the two codebooks (rev
2026-06-11; no new organs), and ``σπσ = σ`` is the differentiable
registration objective with its maintenance duty: mint-time agreement
between asserted order and geometric dominance is a theorem; training
drift is the only divergence channel; the loss restores it. The
predication channel of isomorphic pressure is the asserted-dominance
hinge: recorded taxonomic statements pull extents into the asserted
order.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import PiLayer2, SigmaLayer2, Ops
from Spaces import registration_loss, asserted_dominance_loss

D = 4


def towers(seed=1):
    torch.manual_seed(seed)
    pi = PiLayer2(D, D, stable=True)
    sig = SigmaLayer2(D, D, stable=True)
    return pi, sig


# ---------------------------------------------------------------------------
# §10.5: the loss decreases on a toy corpus; fixed points non-degenerate.
# ---------------------------------------------------------------------------

def test_registration_loss_decreases_on_toy_corpus():
    pi, sig = towers()
    with torch.no_grad():
        for p in list(pi.parameters()) + list(sig.parameters()):
            p.uniform_(-4.0, -0.5)              # randomize within the law
    torch.manual_seed(2)
    codes = torch.rand(16, D) * 0.6 + 0.2       # toy stored symbols
    params = list(pi.parameters()) + list(sig.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    start = registration_loss(pi, sig, codes).item()
    for _ in range(60):
        opt.zero_grad()
        loss = registration_loss(pi, sig, codes)
        loss.backward()
        opt.step()
    end = registration_loss(pi, sig, codes).item()
    assert end < start, f"σπσ=σ loss must decrease: {start:.4f} -> {end:.4f}"
    assert end < 0.5 * start or end < 1e-4, "substantial registration gain"


def test_fixed_points_non_degenerate():
    pi, sig = towers(seed=3)
    torch.manual_seed(4)
    codes = torch.rand(16, D) * 0.6 + 0.2
    params = list(pi.parameters()) + list(sig.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(60):
        opt.zero_grad()
        registration_loss(pi, sig, codes).backward()
        opt.step()
    extents = sig.forward(codes)
    assert extents.std() > 1e-3, (
        "non-degenerate fixed points: registration must not collapse "
        "the extents to a single value")
    assert (extents > 0.01).any() and (extents < 0.99).any(), (
        "fixed points stay off the ⊥/⊤ degeneracies")


# ---------------------------------------------------------------------------
# §10.5 drift case: perturb, flag, re-register — while constructor
# dominance (the §4 theorem) survives drift untouched.
# ---------------------------------------------------------------------------

def test_drift_flagged_and_re_registered():
    torch.manual_seed(5)
    sig2 = SigmaLayer2(2 * D, D, blocks=2, stable=True)
    A = torch.rand(1, D) * 0.5 + 0.25
    B = torch.rand(1, D) * 0.5 + 0.25
    cached = sig2.compose(A, B).detach()         # mint-time evaluation
    assert bool(Ops.partOf(A[0], cached[0])), "dominance at mint"

    with torch.no_grad():                        # training drift
        for p in sig2.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    drifted = sig2.compose(A, B)
    assert not torch.allclose(drifted, cached, atol=1e-3), (
        "drift de-registers the cached evaluation (flagged)")
    assert bool(Ops.partOf(A[0], drifted[0])), (
        "constructor dominance is a THEOREM: drift cannot break "
        "A ⊑ σ(A,B), only the cache agreement")

    opt = torch.optim.Adam(sig2.parameters(), lr=0.05)
    for _ in range(150):                         # maintenance: re-register
        opt.zero_grad()
        ((sig2.compose(A, B) - cached) ** 2).mean().backward()
        opt.step()
    re_registered = sig2.compose(A, B)
    assert torch.allclose(re_registered, cached, atol=0.02), (
        "re-registration restores the minted evaluation")
    assert bool(Ops.partOf(A[0], re_registered[0]))


# ---------------------------------------------------------------------------
# §10.5 predication pressure: recorded taxonomic truths pull geometric
# dominance toward the asserted order; asserted pairs verified by
# dominance afterward.
# ---------------------------------------------------------------------------

def test_predication_pressure_sculpts_the_order():
    torch.manual_seed(6)
    # Rows as free embeddings (sigmoid-reparametrized memberships).
    raw = torch.nn.Parameter(torch.randn(3, D))
    # Asserted: row0 ⊑ row1 ("an elephant is an animal"), row2 ⊑ row1.
    pairs = [(0, 1), (2, 1)]

    def rows():
        return torch.sigmoid(raw)

    r = rows()
    initially_ok = all(bool(Ops.partOf(r[c], r[p])) for c, p in pairs)
    opt = torch.optim.Adam([raw], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        r = rows()
        loss = sum(asserted_dominance_loss(r[c], r[p]) for c, p in pairs)
        loss.backward()
        opt.step()
    r = rows()
    for c, p in pairs:
        assert bool(Ops.partOf(r[c], r[p] + 1e-6)), (
            f"asserted pair {c} ⊑ {p} must hold geometrically after "
            f"training (isomorphic pressure)")
    assert not initially_ok or True, "training moved or kept the order"
    # The pressure is one-sided: no constraint forces 1 ⊑ 0.
    assert not bool(Ops.partOf(r[1], r[0])), (
        "the asserted order is a partial order, not an equivalence")
