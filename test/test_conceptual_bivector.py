"""Tests for ConceptualSpace bivector activation (Step 3 of the
lift / lower / bivector refactor).

Conceptual activations carry the tetralemma bivector ``[aP, aN]`` per
concept (B1, B5 of the spec).  The four corners encode the catuskoti:

    [1, 0] = TRUE       [0, 0] = NEITHER (no commitment)
    [0, 1] = FALSE      [1, 1] = BOTH    (contradiction)

These tests cover the bivector-distinguishability cases that the
signed-collapse shim ``ConceptualSpace.activation_signed()`` exposes
and the routing-correctness cases that fall out of the layout.

See:
- basicmodel/doc/plans/2026-04-25-step3-bivector-conceptual-handoff.md
- basicmodel/doc/specs/2026-04-24-lift-lower-bivector-design.md (B1-B7)
- basicmodel/doc/BuddhistParallels.md (catuskoti mapping)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

from Spaces import SubSpace, ActiveEncoding


def _make_subspace(N=4, D=8):
    """Build a bare SubSpace for direct activation/get-set testing."""
    ae = ActiveEncoding()
    return SubSpace(activeEncoding=ae,
                    inputShape=[N, D], outputShape=[N, D])


def _signed(act):
    """Compute aP - aN from a [B, N, 2] bivector."""
    assert act.ndim == 3 and act.shape[-1] == 2, act.shape
    return act[..., 0] - act[..., 1]


def _contradiction(act):
    """Compute aP * aN from a [B, N, 2] bivector (BOTH-ness)."""
    assert act.ndim == 3 and act.shape[-1] == 2, act.shape
    return act[..., 0] * act[..., 1]


def _ignorance(act):
    """Compute (1-aP) * (1-aN) from a [B, N, 2] bivector (NEITHER-ness)."""
    assert act.ndim == 3 and act.shape[-1] == 2, act.shape
    return (1.0 - act[..., 0]) * (1.0 - act[..., 1])


class TestBivectorRouting(unittest.TestCase):
    """Bivector poles route positive/negative evidence into the right axis."""

    def test_true_pole_routes_to_aP(self):
        """[1, 0] activations: signed = +1, contradiction = 0, ignorance = 0."""
        ss = _make_subspace()
        bivec = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # [B=1, N=2, 2]
        ss.set_activation(bivec)
        act = ss.get_activation()
        torch.testing.assert_close(_signed(act),
                                   torch.tensor([[1.0, 1.0]]))
        torch.testing.assert_close(_contradiction(act),
                                   torch.tensor([[0.0, 0.0]]))
        torch.testing.assert_close(_ignorance(act),
                                   torch.tensor([[0.0, 0.0]]))

    def test_false_pole_routes_to_aN(self):
        """[0, 1] activations: signed = -1, contradiction = 0, ignorance = 0."""
        ss = _make_subspace()
        bivec = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        ss.set_activation(bivec)
        act = ss.get_activation()
        torch.testing.assert_close(_signed(act),
                                   torch.tensor([[-1.0, -1.0]]))
        torch.testing.assert_close(_contradiction(act),
                                   torch.tensor([[0.0, 0.0]]))
        torch.testing.assert_close(_ignorance(act),
                                   torch.tensor([[0.0, 0.0]]))


class TestBivectorDistinguishability(unittest.TestCase):
    """[0, 0] (NEITHER) and [1, 1] (BOTH) collapse to signed = 0 but
    are distinguishable through contradiction / ignorance."""

    def test_neither_vs_both_share_signed_zero(self):
        """Both corners report signed = 0 -- the legacy scalar can't tell
        them apart, which is exactly why the bivector layout matters."""
        ss = _make_subspace()
        bivec = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])  # NEITHER, BOTH
        ss.set_activation(bivec)
        act = ss.get_activation()
        signed = _signed(act)
        torch.testing.assert_close(signed, torch.tensor([[0.0, 0.0]]))

    def test_neither_vs_both_separable_via_contradiction_and_ignorance(self):
        """contradiction = aP * aN: 0 for NEITHER, 1 for BOTH.
        ignorance = (1-aP) * (1-aN): 1 for NEITHER, 0 for BOTH.
        Either sibling scalar is sufficient to distinguish the corners."""
        ss = _make_subspace()
        bivec = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])  # NEITHER, BOTH
        ss.set_activation(bivec)
        act = ss.get_activation()
        torch.testing.assert_close(_contradiction(act),
                                   torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(_ignorance(act),
                                   torch.tensor([[1.0, 0.0]]))

    def test_all_four_corners_have_unique_contradiction_ignorance(self):
        """Each corner has a unique (signed, contradiction) signature."""
        ss = _make_subspace()
        corners = torch.tensor([[[1.0, 0.0],   # TRUE
                                 [0.0, 1.0],   # FALSE
                                 [0.0, 0.0],   # NEITHER
                                 [1.0, 1.0]]]) # BOTH
        ss.set_activation(corners)
        act = ss.get_activation()
        signed = _signed(act)
        contra = _contradiction(act)
        ignore = _ignorance(act)
        torch.testing.assert_close(signed,
                                   torch.tensor([[1.0, -1.0, 0.0, 0.0]]))
        torch.testing.assert_close(contra,
                                   torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        torch.testing.assert_close(ignore,
                                   torch.tensor([[0.0, 0.0, 1.0, 0.0]]))


class TestSignedCollapseShim(unittest.TestCase):
    """The pre-Step-3 single-scalar activation lifted via
    ``set_activation`` to ``[max(0, x), max(0, -x)]`` must round-trip
    through ``aP - aN`` to the original scalar (within sign).

    This is the rollback safety valve: every legacy caller that read a
    signed scalar continues to see the same number through the shim
    until it migrates to bivector-aware reads (Step 7).
    """

    def test_positive_scalar_round_trip_through_shim(self):
        """Pre-refactor positive scalar x lifts to [x, 0]; shim returns x."""
        ss = _make_subspace()
        scalar = torch.tensor([[0.5, 0.8, 0.1]])
        ss.set_activation(scalar)
        act = ss.get_activation()
        # set_activation lifts: positive scalars route to aP pole.
        torch.testing.assert_close(_signed(act), scalar)

    def test_negative_scalar_round_trip_through_shim(self):
        """Pre-refactor negative scalar x lifts to [0, |x|]; shim returns x."""
        ss = _make_subspace()
        scalar = torch.tensor([[-0.7, -0.2, -0.9]])
        ss.set_activation(scalar)
        act = ss.get_activation()
        torch.testing.assert_close(_signed(act), scalar)

    def test_mixed_sign_scalar_round_trip_through_shim(self):
        """Mixed sign scalars: sign and magnitude survive shim collapse."""
        ss = _make_subspace()
        scalar = torch.tensor([[-0.7, 0.4, 0.0, 0.9]])
        ss.set_activation(scalar)
        act = ss.get_activation()
        torch.testing.assert_close(_signed(act), scalar)


class TestConceptualSpaceShim(unittest.TestCase):
    """``ConceptualSpace.activation_signed()`` end-to-end: derive
    aP - aN from a populated subspace and confirm the math matches the
    direct bivector computation."""

    def _build_conceptual_subspace(self):
        from Spaces import ConceptualSpace
        # ConceptualSpace.__init__ requires TheXMLConfig — but for the
        # shim test we only need the bivector pass-through.  Construct a
        # minimal ConceptualSpace-like holder that exposes
        # ``self.subspace`` with a populated activation tensor.
        class _Holder:
            activation_signed = ConceptualSpace.activation_signed
            def __init__(self):
                self.subspace = _make_subspace()
        return _Holder()

    def test_shim_returns_aP_minus_aN(self):
        """activation_signed() returns [B, N] = aP - aN."""
        holder = self._build_conceptual_subspace()
        bivec = torch.tensor([[[1.0, 0.0],   # TRUE -> +1
                               [0.0, 1.0],   # FALSE -> -1
                               [0.0, 0.0],   # NEITHER -> 0
                               [1.0, 1.0]]]) # BOTH -> 0
        holder.subspace.set_activation(bivec)
        signed = holder.activation_signed()
        torch.testing.assert_close(signed,
                                   torch.tensor([[1.0, -1.0, 0.0, 0.0]]))

    def test_shim_returns_none_when_activation_unset(self):
        """activation_signed() returns None for an empty subspace."""
        holder = self._build_conceptual_subspace()
        # Force activation to None so the shim's None-guard fires.
        holder.subspace.activation.setW(None)
        self.assertIsNone(holder.activation_signed())

    def test_shim_legacy_scalar_round_trip(self):
        """Legacy scalar path: scalar -> set_activation lift -> shim."""
        holder = self._build_conceptual_subspace()
        scalar = torch.tensor([[-0.7, 0.4, 0.0, 0.9]])
        holder.subspace.set_activation(scalar)
        torch.testing.assert_close(holder.activation_signed(), scalar)


class TestTetralemmaPolicyConfig(unittest.TestCase):
    """``XMLConfig.tetralemma_policy(space)`` resolves the per-space
    policy with override-fallback inheritance from the global block
    (spec O4)."""

    def _make_config(self, xml_text):
        """Build an XMLConfig from an inline XML string."""
        import tempfile
        from util import XMLConfig
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml',
                                         delete=False) as f:
            f.write(xml_text)
            path = f.name
        return XMLConfig(path=path)

    def test_inherits_global_policy_when_override_disabled(self):
        cfg = self._make_config("""<?xml version="1.0"?>
<model>
  <architecture><type>mental</type></architecture>
  <TetralemmaPolicy>
    <allowExcludedMiddle>1</allowExcludedMiddle>
    <allowContradiction>0</allowContradiction>
    <neitherThreshold>0.1</neitherThreshold>
  </TetralemmaPolicy>
  <ConceptualSpace>
    <tetralemmaOverride enabled="false" />
  </ConceptualSpace>
</model>""")
        em, c, t = cfg.tetralemma_policy("ConceptualSpace")
        self.assertEqual(em, 1)
        self.assertEqual(c, 0)
        self.assertAlmostEqual(t, 0.1, places=5)

    def test_per_space_override_wins_when_enabled(self):
        cfg = self._make_config("""<?xml version="1.0"?>
<model>
  <architecture><type>mental</type></architecture>
  <TetralemmaPolicy>
    <allowExcludedMiddle>1</allowExcludedMiddle>
    <allowContradiction>0</allowContradiction>
    <neitherThreshold>0.1</neitherThreshold>
  </TetralemmaPolicy>
  <SymbolicSpace>
    <tetralemmaOverride enabled="true" />
    <TetralemmaPolicy>
      <allowExcludedMiddle>-1</allowExcludedMiddle>
      <allowContradiction>1</allowContradiction>
      <neitherThreshold>0.25</neitherThreshold>
    </TetralemmaPolicy>
  </SymbolicSpace>
</model>""")
        em, c, t = cfg.tetralemma_policy("SymbolicSpace")
        self.assertEqual(em, -1)
        self.assertEqual(c, 1)
        self.assertAlmostEqual(t, 0.25, places=5)

    def test_defaults_when_no_global_block(self):
        cfg = self._make_config("""<?xml version="1.0"?>
<model>
  <architecture><type>mental</type></architecture>
  <ConceptualSpace />
</model>""")
        em, c, t = cfg.tetralemma_policy("ConceptualSpace")
        # Defaults match the documented Kleene policy.
        self.assertEqual(em, 1)
        self.assertEqual(c, 0)
        self.assertAlmostEqual(t, 0.1, places=5)


if __name__ == '__main__':
    unittest.main()
