"""Lexical masks on lift/lower (GrammarOpsPass §2).

One shared operator matrix, many verbs: a per-word gate -- produced by a
small learned projection from the word's code to the inner LDU's gate
rank space (one projection per operator, tanh-bounded per the
``_d_effective`` convention) -- selects a slice of the shared lift /
lower weights, differentiating walking from running without a weight
matrix per verb. Syntactic validity stays in the grammar files (§1);
the masks are lexical only.

Pinned here:
  * two verbs sharing one matrix produce DISTINCT compositions with
    masks on, and indistinguishable ones with masks ablated
    (``gate=None`` is byte-identical to the un-gated baseline);
  * the projection trains END-TO-END with the composition losses (no
    separate mask objective);
  * the §10.10 weight law holds under ARBITRARY gates on the
    membership kernels: the contractive clamp bounds the gated
    diagonal to [1, d_max], memberships stay in (0, 1], and the gated
    round-trip stays exact;
  * the gate flows through the MeronymicFoldAdapter unchanged on the
    membership path.
"""

import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_D = 6


def _op(cls_name):
    import Language
    torch.manual_seed(7)
    return getattr(Language, cls_name)(nInput=_D, nOutput=_D)


def _operands(seed=11):
    torch.manual_seed(seed)
    a = torch.rand(2, _D) * 1.6 - 0.8
    b = torch.rand(2, _D) * 1.6 - 0.8
    return a, b


@pytest.mark.parametrize("cls_name", ["LiftLayer", "LowerLayer"])
def test_two_verbs_distinct_with_masks_on_ablated_identical(cls_name):
    """One shared matrix, two verbs: distinct with masks on; with masks
    ablated the verbs are indistinguishable and the path is the
    unchanged baseline."""
    op = _op(cls_name)
    torch.manual_seed(13)
    walk = torch.rand(_D) * 2 - 1
    run = torch.rand(_D) * 2 - 1
    a, b = _operands()

    g_walk = op.lexical_gate(walk)
    g_run = op.lexical_gate(run)
    assert g_walk.shape == (_D,)
    # tanh-bounded convention of _d_effective.
    assert torch.all(g_walk.abs() < 1.0) and torch.all(g_run.abs() < 1.0)

    with torch.no_grad():
        c_walk = op.forward(a, b, gate=g_walk)
        c_run = op.forward(a, b, gate=g_run)
        c_none1 = op.forward(a, b)
        c_none2 = op.forward(a, b, gate=None)

    assert not torch.allclose(c_walk, c_run, atol=1e-7), (
        f"{cls_name}: two verbs through one matrix must compose "
        f"distinctly with masks on")
    assert torch.equal(c_none1, c_none2), (
        f"{cls_name}: masks ablated, the verbs are indistinguishable "
        f"(gate=None is the baseline path)")


@pytest.mark.parametrize("cls_name", ["LiftLayer", "LowerLayer"])
def test_gate_none_code_passthrough(cls_name):
    """lexical_gate(None) is None, so call sites can thread the result
    unconditionally; a zero-width layer also produces None."""
    import Language
    op = _op(cls_name)
    assert op.lexical_gate(None) is None
    probe = getattr(Language, cls_name)()    # zero-width harness probe
    assert probe.lexical_gate(torch.rand(_D)) is None


@pytest.mark.parametrize("cls_name", ["LiftLayer", "LowerLayer"])
def test_projection_trains_end_to_end_with_composition_loss(cls_name):
    """End-to-end training (DECIDED 2026-06-11): the embedding->gate map
    learns from the composition loss alone -- no separate mask
    objective -- and the per-verb gates move apart."""
    op = _op(cls_name)
    torch.manual_seed(17)
    walk = torch.rand(_D) * 2 - 1
    run = torch.rand(_D) * 2 - 1
    a, b = _operands(seed=19)
    t_walk = torch.rand(2, _D) * 1.2 - 0.6
    t_run = torch.rand(2, _D) * 1.2 - 0.6

    def loss_fn():
        c_w = op.forward(a, b, gate=op.lexical_gate(walk))
        c_r = op.forward(a, b, gate=op.lexical_gate(run))
        return ((c_w - t_walk) ** 2).mean() + ((c_r - t_run) ** 2).mean()

    with torch.no_grad():
        gap0 = (op.lexical_gate(walk) - op.lexical_gate(run)).abs().sum()
    opt = torch.optim.Adam(op.parameters(), lr=1e-2)
    initial = loss_fn().item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    final = loss_fn().item()
    assert final < initial, (cls_name, initial, final)
    # The projection itself learned (received gradient through the
    # composition loss) and the verbs' gates differentiated further.
    with torch.no_grad():
        gap1 = (op.lexical_gate(walk) - op.lexical_gate(run)).abs().sum()
    assert gap1 != gap0, f"{cls_name}: projection did not train"


def test_weight_law_holds_under_arbitrary_gates():
    """§10.10 on the membership kernels: for ANY gate (negative, zero,
    huge), the contractive clamp bounds the gated diagonal to
    [1, d_max], pi keeps memberships in (0, 1], and the gated
    forward/reverse round-trip is exact."""
    from Layers import PiLayer2, meronomy_d_max_stable
    torch.manual_seed(23)
    pi = PiLayer2(_D, _D)
    d_max = float(meronomy_d_max_stable())
    m = torch.rand(3, _D) * 0.8 + 0.1            # memberships in (0, 1)
    gates = [
        torch.full((_D,), -3.0),
        torch.zeros(_D),
        torch.rand(_D) * 2 - 1,                  # tanh-convention range
        torch.full((_D,), 100.0),
    ]
    for g in gates:
        pi.layer._current_gate = g
        try:
            d_eff = pi.layer._d_effective()
        finally:
            pi.layer._current_gate = None
        assert torch.all(d_eff >= 1.0) and torch.all(d_eff <= d_max), (
            g, d_eff)
        z = pi.forward(m, gate=g)
        assert torch.all(z > 0) and torch.all(z <= 1.0), (
            "membership weight law violated under gate", g)
        back = pi.reverse(z, gate=g)
        assert torch.allclose(back, m, atol=1e-5), (
            "gated round-trip broke", g)


@pytest.mark.parametrize("cls_name", ["LiftLayer", "LowerLayer"])
def test_signed_kernel_gated_roundtrip(cls_name):
    """The gate-slicing surface keeps working on the signed kernels
    (§3): the gated forward/reverse round-trip of the legacy folds is
    self-consistent -- forward(*reverse(y, g), g) == y."""
    op = _op(cls_name)
    a, b = _operands(seed=29)
    g = torch.tanh(torch.rand(_D) + 0.5)         # tanh-bounded, nonzero
    with torch.no_grad():
        y = op.forward(a, b, gate=g)
        l, r = op.reverse(y, gate=g)
        y2 = op.forward(l, r, gate=g)
    assert torch.allclose(y, y2, atol=1e-4), cls_name


def test_gate_flows_through_meronymic_adapter():
    """The gate flows through the MeronymicFoldAdapter unchanged on the
    membership path: an amplifying gate (escaping the [1, d_max] clamp
    floor) changes the fold, and the gated reverse round-trips."""
    from Layers import MeronymicFoldAdapter
    torch.manual_seed(31)
    x = torch.rand(2, _D) * 0.8 + 0.1
    g = torch.full((_D,), 2.0)                   # amplifying: visible slice
    for kind in ("pi", "sigma"):
        ad = MeronymicFoldAdapter(kind, _D, _D)
        with torch.no_grad():
            out_gated = ad.forward(x, gate=g)
            out_plain = ad.forward(x)
            back = ad.reverse(out_gated, gate=g)
        assert not torch.allclose(out_gated, out_plain, atol=1e-7), (
            f"{kind}: gate had no effect through the adapter")
        assert torch.allclose(back, x, atol=1e-4), (
            f"{kind}: gated adapter round-trip broke")
