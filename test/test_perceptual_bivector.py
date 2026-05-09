"""PerceptualSpace bivector activation -- spec B3 acceptance.

Tests the Q2 promote / lower helpers and the round-trip property:
forward then reverse on a bitonic input recovers a signed-collapse
of the original (the per-feature pattern is lost by design -- see
PerceptualSpace._q2_lower_activation docstring).
"""
import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor_bivector.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Build a fresh MentalModel from MM_xor_bivector.xml with XOR data."""
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, cfg = Models.MentalModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m


class TestQ2Promotion(unittest.TestCase):
    """Q2: scalar bitonic -> bivector monotonic split."""

    @classmethod
    def setUpClass(cls):
        cls.model = _fresh_model()
        cls.percep = cls.model.perceptualSpace

    def test_bivector_output_flag_is_true(self):
        self.assertTrue(self.percep._bivector_output)

    def test_q2_promote_positive_event_yields_aP_only(self):
        # Per-slot positive event sums to +D_P; aP = +D_P, aN = 0.
        event = torch.ones(2, 8, 2)            # [B=2, N=8, D_P=2]
        bivec = self.percep._q2_promote_activation(event)
        self.assertEqual(bivec.shape, (2, 8, 2))
        torch.testing.assert_close(bivec[..., 0],
                                   2.0 * torch.ones(2, 8))
        torch.testing.assert_close(bivec[..., 1],
                                   torch.zeros(2, 8))

    def test_q2_promote_negative_event_yields_aN_only(self):
        event = -1.0 * torch.ones(2, 8, 2)
        bivec = self.percep._q2_promote_activation(event)
        torch.testing.assert_close(bivec[..., 0],
                                   torch.zeros(2, 8))
        torch.testing.assert_close(bivec[..., 1],
                                   2.0 * torch.ones(2, 8))

    def test_q2_promote_zero_event_yields_zero_bivec(self):
        # NEITHER corner of the tetralemma: aP = aN = 0.
        event = torch.zeros(2, 8, 2)
        bivec = self.percep._q2_promote_activation(event)
        torch.testing.assert_close(bivec, torch.zeros(2, 8, 2))

    def test_q2_promote_mixed_signs_per_slot(self):
        # Slot 0: positive, slot 1: negative. No "BOTH" emitted from a
        # single signed-sum scalar -- BOTH only arises when the signed
        # scalar is overwritten upstream with both poles set.
        event = torch.zeros(1, 2, 2)
        event[0, 0, :] = 0.5      # signed sum = 1.0 -> [1, 0]
        event[0, 1, :] = -0.25    # signed sum = -0.5 -> [0, 0.5]
        bivec = self.percep._q2_promote_activation(event)
        torch.testing.assert_close(bivec[0, 0],
                                   torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(bivec[0, 1],
                                   torch.tensor([0.0, 0.5]))


class TestQ2Lowering(unittest.TestCase):
    """Inverse Q2: bivector -> per-slot scalar broadcast across D_P."""

    @classmethod
    def setUpClass(cls):
        cls.model = _fresh_model()
        cls.percep = cls.model.perceptualSpace

    def test_lower_recovers_signed_scalar_broadcast(self):
        # Bivector [aP=0.7, aN=0.2] -> scalar 0.5 -> broadcast to D_P=4.
        bivec = torch.tensor([[[0.7, 0.2]]])      # [B=1, N=1, 2]
        event = self.percep._q2_lower_activation(bivec, content_dim=4)
        self.assertEqual(event.shape, (1, 1, 4))
        torch.testing.assert_close(event,
                                   0.5 * torch.ones(1, 1, 4))

    def test_lower_then_promote_collapses_per_feature_detail(self):
        # The promote-lower round-trip is lossy: per-feature variation
        # within a slot is replaced by the uniform broadcast of the
        # signed-sum scalar. Verify the scalar survives even though
        # the pattern doesn't.
        event = torch.tensor([[[0.3, -0.1, 0.5, 0.0]]])    # signed sum = 0.7
        bivec = self.percep._q2_promote_activation(event)
        recovered = self.percep._q2_lower_activation(bivec, content_dim=4)
        # Recovered is uniform 0.7 across all 4 features (vs. original
        # heterogeneous [0.3, -0.1, 0.5, 0.0]).
        torch.testing.assert_close(recovered,
                                   0.7 * torch.ones(1, 1, 4))


class TestPipelineRoundTrip(unittest.TestCase):
    """MM_xor_bivector end-to-end: input -> P -> C -> S runs without
    shape errors, and PerceptualSpace's activation tensor has the
    expected ``[B, N, 2]`` bivector shape."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def _run_forward(self):
        import warnings
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
            inp_items, out_items = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            with torch.no_grad():
                m.forward(inputTensor)

    def test_perceptual_activation_is_bivector_after_forward(self):
        self._run_forward()
        act = self.model.perceptualSpace.subspace.activation.getW()
        self.assertIsNotNone(act, "PerceptualSpace.activation is empty")
        self.assertEqual(act.dim(), 3,
                         f"Expected [B, N, 2], got shape {tuple(act.shape)}")
        self.assertEqual(act.shape[-1], 2,
                         f"Expected last dim=2 (bivector), got {act.shape[-1]}")

    def test_bivector_components_are_non_negative(self):
        self._run_forward()
        act = self.model.perceptualSpace.subspace.activation.getW()
        self.assertTrue((act >= 0).all(),
                        f"Bivector components must be in [0, 1]^2; "
                        f"min observed = {act.min().item()}")

    def test_loopback_widened_input_to_conceptual(self):
        # ConceptualSpace's PiLayer input width should be P.nOutputDim
        # (2) + S.nOutputDim (2) = 4 under the loopback widening.
        cs = self.model.conceptualSpace
        self.assertEqual(cs._right_half_dim, 2,
                         f"Expected loopback widen=2 (S.nOutputDim), "
                         f"got {cs._right_half_dim}")


if __name__ == "__main__":
    unittest.main()
