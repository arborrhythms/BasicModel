# basicmodel/test/test_mask_dispatch.py
"""Tests for Mask parameter on S-tier derivations.

Plan reference: lazy-juggling-planet.md Phase A. Mask is a concept-axis
selector of shape ``[K]`` (outward-facing). Internally, when the operand
is a bivector storage tensor ``[..., 2K]``, mask is expanded via
``repeat_interleave(2)`` so paired poles stay co-masked. Invariants:
- ``mask=None`` is identity (bit-exact equality with pre-mask behavior).
- Non-selected concept indices zero both poles in the output.
- Under a constant mask, ``equalsForward`` is transitive on the masked
  subset (agreement on a fixed set of dims is transitive).
- ``project()``/``reverse_project()`` thread mask through to dispatched
  methods.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Models  # noqa: E402
import Spaces  # noqa: E402


def _make_layer(K=4):
    """Build a SymbolicSyntacticLayer + bivector SubSpace for K concepts.

    Storage dim is 2K (bivector paired-index encoding). Returns
    ``(layer, subspace, grammar)``.
    """
    g = Spaces.Grammar()
    g.configure({
        "S": ["true(S)", "false(S)", "non(S)", "conjunction(S, S)",
              "disjunction(S, S)", "equals(S, S)", "part(S, S)", "C"],
        "C": ["not(C)", "intersection(C, C)", "union(C, C)", "P"],
        "P": "I",
    })
    subspace = Models.SubSpace(inputShape=[2 * K, 2 * K], outputShape=[2 * K, 2 * K])
    basis = Models.Codebook()
    basis.create(2 * K, 2 * K, 2 * K, monotonic=True)
    subspace.what = basis
    layer = Spaces.SymbolicSyntacticLayer(
        nInput=2 * K, nOutput=2 * K,
        rules=g.symbolic(),
        transition_rule=g.symbolic_transition(),
        max_depth=3, hidden_dim=8, grammar=g,
    )
    layer.init_swap(2 * K, 2 * K)
    return layer, subspace, g


def _find_rule(g, name):
    for i, r in enumerate(g.rules):
        if r.method_name == name:
            return i
    raise KeyError(f"No rule with method_name={name!r}")


def test_mask_none_is_identity():
    """mask=None must not alter any operator's output."""
    layer, ss, _ = _make_layer(K=4)
    torch.manual_seed(0)
    a = torch.rand(1, 8)
    b = torch.rand(1, 8)
    for fn_name, binary in [
            ('intersectionForward', True),
            ('unionForward', True),
            ('equalsForward', True),
            ('partForward', True),
            ('notForward', False),
            ('trueForward', False),
            ('falseForward', False),
    ]:
        fn = getattr(layer, fn_name)
        if binary:
            ref = fn(a, b, ss)
            masked = fn(a, b, ss, mask=None)
        else:
            ref = fn(a, ss)
            masked = fn(a, ss, mask=None)
        assert torch.allclose(ref, masked), f"{fn_name} differs with mask=None"


def test_mask_zeros_nonselected_pairs():
    """Non-selected concepts in the mask zero both poles in the output."""
    layer, ss, _ = _make_layer(K=4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    a = torch.rand(1, 8) + 0.1  # ensure non-zero magnitudes
    b = torch.rand(1, 8) + 0.1

    for fn_name, binary in [
            ('intersectionForward', True),
            ('unionForward', True),
            ('equalsForward', True),
            ('partForward', True),
            ('notForward', False),
            ('trueForward', False),
            ('falseForward', False),
    ]:
        fn = getattr(layer, fn_name)
        if binary:
            out = fn(a, b, ss, mask=mask)
        else:
            out = fn(a, ss, mask=mask)
        # Concepts 1 and 3 are masked off -> storage indices 2,3 and 6,7 zero
        for k in [1, 3]:
            assert out[..., 2 * k].abs().max().item() == 0.0, \
                f"{fn_name}: storage[{2*k}] not zeroed"
            assert out[..., 2 * k + 1].abs().max().item() == 0.0, \
                f"{fn_name}: storage[{2*k+1}] not zeroed"


def test_equals_transitivity_under_mask():
    """Under a constant mask, equalsForward agrees transitively on masked dims."""
    layer, ss, _ = _make_layer(K=4)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])  # keep concepts 0, 1

    # On masked dims (storage 0..3), all three agree; off-mask they differ.
    a = torch.tensor([[0.7, 0.1, 0.8, 0.2, 0.9, 0.1, 0.3, 0.2]])
    b = torch.tensor([[0.7, 0.1, 0.8, 0.2, 0.2, 0.6, 0.9, 0.4]])
    c = torch.tensor([[0.7, 0.1, 0.8, 0.2, 0.5, 0.5, 0.1, 0.8]])

    ab = layer.equalsForward(a, b, ss, mask=mask)
    bc = layer.equalsForward(b, c, ss, mask=mask)
    ac = layer.equalsForward(a, c, ss, mask=mask)

    # On masked dims, ab and bc are perfect matches -> score=1; ac same.
    # Transitivity: if ab~b and bc~c on mask, then ac~c on mask.
    assert ab[..., :4].allclose(b[..., :4]), "ab masked dims should equal b"
    assert bc[..., :4].allclose(c[..., :4]), "bc masked dims should equal c"
    assert ac[..., :4].allclose(c[..., :4]), "ac masked dims should equal c"


def test_mask_threads_through_project():
    """SyntacticLayer.project forwards the mask kwarg to the dispatched method."""
    layer, ss, g = _make_layer(K=4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    rid = _find_rule(g, 'intersection')
    a = torch.rand(1, 8) + 0.1
    b = torch.rand(1, 8) + 0.1

    out = layer.project(g, rid, a, b, subspace=ss, mask=mask)
    # Masked concepts (1, 3) must be zeroed in the result.
    for k in [1, 3]:
        assert out[..., 2 * k].abs().max().item() == 0.0
        assert out[..., 2 * k + 1].abs().max().item() == 0.0


def test_mask_repeat_interleave_bivector_safety():
    """Masking concept k zeros both poles 2k and 2k+1 regardless of magnitudes."""
    layer, ss, _ = _make_layer(K=4)
    # Huge magnitudes in the non-selected concept pair to ensure the mask
    # actually zeros them (not just numerical noise).
    a = torch.tensor([[0.3, 0.2, 99.0, 77.0, 0.5, 0.1, 55.0, 88.0]])
    b = torch.tensor([[0.4, 0.1, 44.0, 33.0, 0.6, 0.2, 22.0, 11.0]])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = layer.intersectionForward(a, b, ss, mask=mask)
    assert out[..., 2].abs().item() == 0.0
    assert out[..., 3].abs().item() == 0.0
    assert out[..., 6].abs().item() == 0.0
    assert out[..., 7].abs().item() == 0.0
