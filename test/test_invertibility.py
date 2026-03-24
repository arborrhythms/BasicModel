"""Invertibility tests for layers and Spaces.

Migrated from bin/scratch.py into the pytest harness.
"""
import os, sys, unittest
import torch
import torch.nn as nn
import torch.optim as optim

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from util import TheDevice
from Model import (
    LinearLayer, InvertibleLinearLayer, SigmaLayer,
    PiLayer, AttentionLayer, NormLayer, LiftingLayer,
)


def _reconstruction_error(x, x_rec, rel=False):
    """Return scalar reconstruction error."""
    if rel:
        denom = torch.norm(x).item()
        return (torch.norm(x - x_rec) / max(denom, 1e-12)).item()
    return torch.norm(x - x_rec).item()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Low-level layer invertibility
# ═══════════════════════════════════════════════════════════════════════════

class TestInvertibleLinearLayer(unittest.TestCase):
    """Roundtrip tests for InvertibleLinearLayer (LDU factorisation).

    Covers non-ergodic and ergodic paths, naive and non-naive dispatch,
    square / expand / contract shapes, batched inputs, and bias.
    """

    def _check(self, nIn, nOut, naive, tol, hasBias=True, batch=(2, 3)):
        torch.manual_seed(42)
        layer = InvertibleLinearLayer(nIn, nOut, naive=naive, hasBias=hasBias)
        layer.set_sigma(0)
        if nIn <= nOut:
            # Square / expand: left-invertible → reverse(forward(x)) ≈ x
            x = torch.randn(*batch, nIn).to(TheDevice)
            y = layer.forward(x)
            x_rec = layer.reverse(y)
            err = _reconstruction_error(x, x_rec)
            self.assertLess(err, tol,
                            f"{nIn}->{nOut} naive={naive} hasBias={hasBias}: err={err:.2e}")
        else:
            # Contract (nIn > nOut): right-invertible → forward(reverse(y)) ≈ y
            y = torch.randn(*batch, nOut).to(TheDevice)
            x_rev = layer.reverse(y)
            y_rec = layer.forward(x_rev)
            err = _reconstruction_error(y, y_rec)
            self.assertLess(err, tol,
                            f"{nIn}->{nOut} naive={naive} hasBias={hasBias}: err={err:.2e}")

    # non-ergodic, naive=True (dense W + pinv)
    def test_square_naive(self):    self._check(5, 5, True,  1e-3)
    def test_expand_naive(self):    self._check(5, 8, True,  1e-3)
    def test_contract_naive(self):  self._check(8, 5, True,  1e-3)
    def test_no_bias_naive(self):   self._check(5, 5, True,  1e-3, hasBias=False)

    # non-ergodic, naive=False (triangular solves)
    def test_square(self):          self._check(5, 5, False, 1e-3)
    def test_expand(self):          self._check(5, 8, False, 1e-3)
    def test_contract(self):        self._check(8, 5, False, 1e-3)
    def test_no_bias(self):         self._check(5, 5, False, 1e-3, hasBias=False)

    # 2-D batch (single batch dim)
    def test_2d_batch(self):        self._check(6, 6, False, 1e-3, batch=(4,))

    def test_W_Winverse_identity(self):
        """compute_W() @ compute_Winverse() should be identity."""
        torch.manual_seed(7)
        layer = InvertibleLinearLayer(5, 5, hasBias=False)
        layer.set_sigma(0)
        W     = layer.compute_W()
        W_inv = layer.compute_Winverse()
        I5    = torch.eye(5, device=W.device, dtype=W.dtype)
        err   = (W @ W_inv - I5).norm().item()
        self.assertLess(err, 1e-4, f"W@W_inv identity err={err:.2e}")

    def _check_ergodic(self, nIn, nOut, naive, stable, tol):
        torch.manual_seed(42)
        layer = InvertibleLinearLayer(nIn, nOut, naive=naive, ergodic=True, stable=stable)
        with torch.no_grad():
            layer.var.fill_(0.2)
            layer.bias.fill_(0.8)
        if nIn <= nOut:
            # Square / expand: left-invertible → reverse(forward(x)) ≈ x
            x = torch.randn(3, nIn).to(TheDevice)
            y = layer.forward(x)
            x_rec = layer.reverse(y)
            err = _reconstruction_error(x, x_rec, rel=True)
        else:
            # Contract (nIn > nOut): forward→reverse use the same ergodic factors
            # (forward resamples, reverse uses the stored buffers).  The map is
            # surjective so reverse(forward(x)) only recovers the row-space
            # component of x; the expected relative error is ≈ sqrt((nIn-nOut)/nIn).
            # Use a generous tolerance to verify the ergodic path runs correctly.
            x = torch.randn(3, nIn).to(TheDevice)
            y = layer.forward(x)
            x_rec = layer.reverse(y)
            err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, tol,
                        f"ergodic {nIn}->{nOut} naive={naive} stable={stable}: err={err:.2e}")

    # ergodic roundtrip: factor-level noise injection → exact inverse
    def test_ergodic_square_naive(self):        self._check_ergodic(5, 5, True,  False, 1e-3)
    def test_ergodic_square(self):              self._check_ergodic(5, 5, False, False, 1e-3)
    def test_ergodic_expand(self):              self._check_ergodic(5, 8, False, False, 1e-3)
    def test_ergodic_contract(self):            self._check_ergodic(8, 5, False, False, 0.9)
    def test_ergodic_stable_square(self):       self._check_ergodic(5, 5, False, True,  1e-3)
    def test_ergodic_stable_expand(self):       self._check_ergodic(5, 8, False, True,  1e-3)



class TestLiftingLayer(unittest.TestCase):
    def _check(self, nIn, nOut, tol):
        torch.manual_seed(42)
        layer = LiftingLayer(nIn, nOut)
        x = torch.randn(4, nIn).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, tol, f"{nIn}->{nOut}: err={err:.2e}")

    def test_square(self):    self._check(6, 6, 1e-3)
    def test_expand(self):    self._check(6, 10, 1e-3)
    def test_contract(self):  self._check(10, 6, 10.0)


class TestLinearLayerIdentity(unittest.TestCase):
    """LinearLayer with W=I and no bias should be exact identity."""
    def test_identity(self):
        torch.manual_seed(42)
        layer = LinearLayer(5, 5, hasBias=False, ergodic=True)
        x = torch.randn(3, 5).to(TheDevice)
        y = layer.forward(x)
        err = _reconstruction_error(x, y)
        self.assertLess(err, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ergodic layer invertibility
# ═══════════════════════════════════════════════════════════════════════════

class TestInvertibleSigmaLayer(unittest.TestCase):
    def _check(self, nIn, nOut, naive, tol):
        torch.manual_seed(42)
        layer = SigmaLayer(nIn, nOut, naive=naive, permuteInput=False, invertible=True)
        layer.set_sigma(0)
        x = torch.randn(2, 3, nIn).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, tol,
                        f"{nIn}->{nOut}, naive={naive}: err={err:.2e}")

    def test_expand_naive(self):     self._check(5, 7, True, 1e-3)
    def test_square_naive(self):     self._check(4, 4, True, 1e-3)
    def test_contract_naive(self):   self._check(8, 5, True, 10.0)
    def test_expand(self):           self._check(5, 7, False, 1e-3)
    def test_square(self):           self._check(4, 4, False, 1e-3)
    def test_contract(self):         self._check(8, 5, False, 10.0)

    def _check_permute(self, naive):
        nIn, nOut, seqLen = 5, 7, 3
        torch.manual_seed(42)
        layer = SigmaLayer(nIn, nOut, naive=naive, permuteInput=True, invertible=True)
        layer.set_sigma(0)
        x = torch.randn(2, nIn, seqLen).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        tol = 1e-2 if naive else 1e-3  # pinv less precise than SVD-factored inverse
        self.assertLess(err, tol,
                        f"perm=True, naive={naive}: err={err:.2e}")

    def test_permute_naive(self):  self._check_permute(True)
    def test_permute(self):        self._check_permute(False)


class TestInvertiblePiLayer3D(unittest.TestCase):
    def _check(self, naive, perm, bias):
        nIn, nOut = 4, 8
        torch.manual_seed(42)
        layer = PiLayer(nIn, nOut, naive=naive,
                        permuteInput=perm, hasBias=bias, invertible=True)
        layer.set_sigma(0)
        if perm:
            x = torch.randn(3, nIn, 5).to(TheDevice)
        else:
            x = torch.randn(3, 5, nIn).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.1,
                        f"naive={naive}, perm={perm}, bias={bias}: rel err={err:.2e}")

    def test_naive_perm_bias(self):       self._check(True, True, True)
    def test_naive_perm_nobias(self):     self._check(True, True, False)
    def test_naive_noperm_bias(self):     self._check(True, False, True)
    def test_naive_noperm_nobias(self):   self._check(True, False, False)
    def test_perm_bias(self):             self._check(False, True, True)
    def test_perm_nobias(self):           self._check(False, True, False)
    def test_noperm_bias(self):           self._check(False, False, True)
    def test_noperm_nobias(self):         self._check(False, False, False)


class TestInvertiblePiLayer2D(unittest.TestCase):
    def _check(self, naive, bias):
        nIn, nOut = 4, 8
        torch.manual_seed(42)
        layer = PiLayer(nIn, nOut, naive=naive,
                        permuteInput=False, hasBias=bias, invertible=True)
        layer.set_sigma(0)
        x = torch.randn(6, nIn).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.1,
                        f"naive={naive}, bias={bias}: rel err={err:.2e}")

    def test_naive_bias(self):     self._check(True, True)
    def test_naive_nobias(self):   self._check(True, False)
    def test_bias(self):           self._check(False, True)
    def test_nobias(self):         self._check(False, False)


class TestNonNaiveInvertiblePiLayer(unittest.TestCase):
    """Lock down non-naive InvertiblePiLayer behavior before refactoring.

    Non-naive mode uses OldInvertibleLinearLayer (SVD-factored weights).
    These tests verify forward/reverse roundtrip across 2D, 3D, bias,
    ergodic, and training scenarios.
    """

    def test_2d_roundtrip(self):
        """2D (batch, nInput) -> (batch, 2*nOutput) roundtrip, no noise."""
        torch.manual_seed(42)
        nIn, nOut = 4, 8
        layer = PiLayer(nIn, nOut, naive=False, hasBias=True, invertible=True)
        layer.train(False)
        x = torch.randn(6, nIn).to(TheDevice)
        with torch.no_grad():
            y = layer.forward(x)
            self.assertEqual(y.shape, (6, 2 * nOut))
            x_rec = layer.reverse(y)
            self.assertEqual(x_rec.shape, x.shape)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 1e-3, f"2D roundtrip rel err={err:.2e}")

    def test_2d_nobias(self):
        """2D roundtrip without bias term."""
        torch.manual_seed(42)
        nIn, nOut = 4, 8
        layer = PiLayer(nIn, nOut, naive=False, hasBias=False, invertible=True)
        layer.train(False)
        x = torch.randn(6, nIn).to(TheDevice)
        with torch.no_grad():
            y = layer.forward(x)
            x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 1e-3, f"2D no-bias rel err={err:.2e}")

    def test_3d_roundtrip(self):
        """3D (batch, seq, nInput) -> (batch, 2*seq, nOutput) roundtrip."""
        torch.manual_seed(42)
        nIn, nOut = 4, 8
        layer = PiLayer(nIn, nOut, naive=False, hasBias=True, invertible=True)
        layer.train(False)
        x = torch.randn(3, 5, nIn).to(TheDevice)
        with torch.no_grad():
            y = layer.forward(x)
            self.assertEqual(y.shape, (3, 10, nOut))
            x_rec = layer.reverse(y)
            self.assertEqual(x_rec.shape, x.shape)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 1e-3, f"3D roundtrip rel err={err:.2e}")

    def test_3d_permuted(self):
        """3D with permuteInput=True."""
        torch.manual_seed(42)
        nIn, nOut = 4, 8
        layer = PiLayer(nIn, nOut, naive=False, permuteInput=True, hasBias=True, invertible=True)
        layer.train(False)
        x = torch.randn(3, nIn, 5).to(TheDevice)
        with torch.no_grad():
            y = layer.forward(x)
            x_rec = layer.reverse(y)
            self.assertEqual(x_rec.shape, x.shape)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.1, f"3D permuted rel err={err:.2e}")

    def test_ergodic_roundtrip(self):
        """Non-naive with ergodic noise uses compute_Winverse."""
        torch.manual_seed(42)
        nIn, nOut = 4, 8
        layer = PiLayer(nIn, nOut, naive=False, ergodic=True, invertible=True)
        with torch.no_grad():
            layer.var.fill_(0.05)
            layer.bias.fill_(0.95)
        layer.train(True)
        x = torch.randn(8, nIn).to(TheDevice)
        with torch.no_grad():
            y = layer.forward(x)
            x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.3, f"Ergodic non-naive rel err={err:.2e}")

    def test_training_preserves_invertibility(self):
        """After gradient steps, non-naive roundtrip remains accurate."""
        torch.manual_seed(42)
        nIn, nOut = 4, 8
        layer = PiLayer(nIn, nOut, naive=False, hasBias=True, invertible=True)
        optimizer = optim.Adam(layer.parameters(), lr=0.01)
        x = torch.randn(8, nIn).to(TheDevice)
        # Run a few training steps with a dummy loss
        for _ in range(20):
            optimizer.zero_grad()
            y = layer.forward(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
        # Roundtrip after training
        layer.train(False)
        with torch.no_grad():
            y = layer.forward(x)
            x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 1e-2, f"Post-training rel err={err:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. NormLayer
# ═══════════════════════════════════════════════════════════════════════════

class TestNormLayerInvertibility(unittest.TestCase):
    def test_roundtrip(self):
        torch.manual_seed(42)
        layer = NormLayer(10, 12, pNorm=2)
        layer.lr = 0
        x = torch.randn(5, 10).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Paired layer training (non-invertible encoder/decoder pairs)
# ═══════════════════════════════════════════════════════════════════════════

class TestPairedSigmaTraining(unittest.TestCase):
    """Paired roundtrip with two InvertibleSigmaLayers (separate weights).

    Demonstrates case 3: reversible without weight-sharing.
    forward() on one layer, reverse() on the other.  The reverse path
    uses atanh then the SVD-based pseudoinverse of its linear layer.
    """
    def test_paired_roundtrip(self):
        torch.manual_seed(42)
        nIn, nOut = 6, 8
        sigma_fwd = SigmaLayer(nIn, nOut, invertible=True)
        sigma_rev = SigmaLayer(nIn, nOut, invertible=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(sigma_fwd.parameters()) + list(sigma_rev.parameters()), lr=0.01)
        x_data = torch.randn(8, 3, nIn).to(TheDevice)
        for _ in range(500):
            optimizer.zero_grad()
            y = sigma_fwd(x_data)
            x_rec = sigma_rev.reverse(y)
            loss = criterion(x_data, x_rec)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = sigma_fwd(x_data)
            x_rec = sigma_rev.reverse(y)
        err = _reconstruction_error(x_data, x_rec, rel=True)
        self.assertLess(err, 0.5, f"Paired Sigma rel err={err:.4f}")


class TestPairedPiTraining(unittest.TestCase):
    """Paired roundtrip with two InvertiblePiLayers (separate weights).

    Demonstrates case 3: reversible without weight-sharing.
    forward() on one layer, reverse() on the other.  The reverse path
    recovers x via gamma = 0.5*(log_z - log_y) = Wx, then x = pinv(W) @ gamma.
    Uses naive=True which computes torch.linalg.pinv at each step.
    """
    def test_paired_roundtrip(self):
        torch.manual_seed(42)
        nIn, nOut = 4, 6
        pi_fwd = PiLayer(nIn, nOut, naive=True, permuteInput=True, ergodic=False, invertible=True)
        pi_rev = PiLayer(nIn, nOut, naive=True, permuteInput=True, ergodic=False, invertible=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(pi_fwd.parameters()) + list(pi_rev.parameters()), lr=0.01)
        x_data = torch.randn(8, nIn, 5).to(TheDevice)
        for _ in range(500):
            optimizer.zero_grad()
            y = pi_fwd(x_data)
            x_rec = pi_rev.reverse(y)
            loss = criterion(x_data, x_rec)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = pi_fwd(x_data)
            x_rec = pi_rev.reverse(y)
        err = _reconstruction_error(x_data, x_rec, rel=True)
        self.assertLess(err, 0.5, f"Paired Pi rel err={err:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Space-level invertibility
# ═══════════════════════════════════════════════════════════════════════════

from BasicModel import (
    TheXMLConfig, PerceptualSpace, ConceptualSpace,
    SymbolicSpace, OutputSpace, SyntacticSpace,
)


def _setup_object_encoding(objSize=0, contentDim=6, outputDim=2, nObj=3,
                           reconstruct="FULL", flatten=True, ergodic=False,
                           passThrough=False, hasAttention=True,
                           invertible=False, hasNorm=False,
                           symbolPassThrough=False):
    """Configure TheXMLConfig for isolated Space tests."""
    nWhere = objSize // 2 if objSize > 0 else 0
    nWhen = objSize - nWhere if objSize > 0 else 0
    nObjects = nObj * 6  # 6 spaces, each with nObj vectors
    # Populate config (test equivalent of XML loading)
    TheXMLConfig._data.update({
        "architecture": {
            "reconstruct": reconstruct,
            "ergodic": ergodic,
            "naive": False, "processSymbols": False, "certainty": False,
            "objectSize": objSize, "nObjects": nObjects,
            "nWhere": nWhere, "nWhen": nWhen,
            "embeddingPath": None, "data": {}, "training": {},
        },
        "InputSpace":      {"nDim": contentDim, "nVectors": nObj, "nActive": nObj, "flatten": False, "quantized": False, "lexer": "word"},
        "PerceptualSpace": {"nDim": contentDim, "nVectors": nObj, "nActive": nObj, "flatten": flatten, "quantized": False, "passThrough": passThrough, "hasAttention": hasAttention, "invertible": invertible},
        "ConceptualSpace": {"nDim": contentDim, "nVectors": nObj, "nActive": nObj, "flatten": flatten, "quantized": False, "hasAttention": False, "hasNorm": hasNorm, "invertible": invertible},
        "SymbolicSpace":   {"nDim": contentDim, "nVectors": nObj, "nActive": nObj, "flatten": flatten, "passThrough": symbolPassThrough, "quantized": False},
        "SyntacticSpace":  {"nDim": contentDim, "nVectors": nObj, "nActive": nObj, "flatten": flatten, "quantized": False},
        "OutputSpace":     {"nDim": outputDim,  "nVectors": nObj, "nActive": nObj, "flatten": True,   "quantized": False, "invertible": False},
    })


class TestPerceptualSpacePassthrough(unittest.TestCase):
    """PerceptualSpace with passThrough=True is exact identity."""
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize, passThrough=True)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        pspace = PerceptualSpace(
            [nObj, contentDim], [nObj, contentDim], [nObj, contentDim],
        )
        pspace.eval()
        x = torch.randn(2, nObj, embDim).to(TheDevice)
        with torch.no_grad():
            y = pspace.forward(x)
            x_rec = pspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-6, f"objSize={objSize}: err={err:.2e}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestPerceptualSpaceReversePassTrained(unittest.TestCase):
    """PerceptualSpace with reversible=True, trained pair for roundtrip.

    InvertiblePiLayer doubles dim 1 (objects), so nActive_out = 2 * nActive_in.
    Uses flatten=False (3D mode) where the doubling maps cleanly to nActive.
    """
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize, passThrough=False, hasAttention=False,
                               ergodic=False)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        pspace = PerceptualSpace(
            [nObj, contentDim], [nObj, contentDim], [2*nObj, contentDim],
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(pspace.parameters(), lr=0.005)
        x_data = torch.randn(4, nObj, embDim).to(TheDevice)
        pspace.train()
        for _ in range(2000):
            optimizer.zero_grad()
            y = pspace.forward(x_data)
            x_rec = pspace.reverse(y)
            loss = criterion(x_data, x_rec)
            loss.backward()
            optimizer.step()
        pspace.eval()
        with torch.no_grad():
            y = pspace.forward(x_data)
            x_rec = pspace.reverse(y)
        err = _reconstruction_error(x_data, x_rec, rel=True)
        self.assertLess(err, 1.0, f"objSize={objSize}: rel err={err:.4f}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestConceptualSpaceInvertible(unittest.TestCase):
    """ConceptualSpace with invertible=True should roundtrip well."""
    def _check(self, objSize, flatten):
        _setup_object_encoding(objSize=objSize, invertible=True, hasNorm=False,
                               ergodic=False, flatten=flatten, hasAttention=False)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        cspace = ConceptualSpace(
            [nObj, contentDim], [nObj, contentDim], [nObj, contentDim],
        )
        cspace.eval()
        cspace.sigma.set_sigma(0)
        x = torch.randn(2, nObj, embDim).to(TheDevice)
        with torch.no_grad():
            y = cspace.forward(x)
            x_rec = cspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-2,
                        f"objSize={objSize}, flatten={flatten}: err={err:.2e}")

    def test_reshape_objsize0(self):      self._check(0, True)
    def test_reshape_objsize4(self):      self._check(4, True)
    def test_no_reshape_objsize0(self):   self._check(0, False)
    def test_no_reshape_objsize4(self):   self._check(4, False)


class TestConceptualSpacePairedSigma(unittest.TestCase):
    """ConceptualSpace with paired (non-invertible) sigma layers."""
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize, invertible=False, hasNorm=False,
                               ergodic=False, flatten=True, hasAttention=False)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        cspace = ConceptualSpace(
            [nObj, contentDim], [nObj, contentDim], [nObj, contentDim],
        )
        cspace.eval()
        x = torch.randn(2, nObj, embDim).to(TheDevice)
        with torch.no_grad():
            y = cspace.forward(x)
            x_rec = cspace.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 2.5, f"objSize={objSize}: rel err={err:.4f}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestConceptualSpaceHasNorm(unittest.TestCase):
    """ConceptualSpace with hasNorm=True.

    Norm factors (mean, std) are cached during forward and reattached
    during reverse, keeping the sigma layer square for exact invertibility.
    """
    def test_hasNorm_reshape(self):
        _setup_object_encoding(objSize=0, reconstruct="FULL", flatten=True,
                               invertible=True, hasNorm=True, ergodic=False,
                               hasAttention=False)
        nObj, contentDim = 3, 6
        torch.manual_seed(42)
        cspace = ConceptualSpace(
            [nObj, contentDim], [nObj, contentDim], [nObj, contentDim],
        )
        cspace.eval()
        cspace.sigma.set_sigma(0)
        x = torch.randn(2, nObj, contentDim).to(TheDevice)
        with torch.no_grad():
            y = cspace.forward(x)
            x_rec = cspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-2, f"err={err:.2e}")


class TestSymbolicSpacePassthrough(unittest.TestCase):
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize, reconstruct="FULL", flatten=True,
                               symbolPassThrough=True)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        sspace = SymbolicSpace(
            [nObj, contentDim], [nObj, contentDim], [nObj, contentDim],
        )
        sspace.eval()
        x = torch.randn(2, nObj, embDim).to(TheDevice)
        with torch.no_grad():
            y = sspace.forward(x)
            x_rec = sspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-6, f"objSize={objSize}: err={err:.2e}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestSyntacticSpace(unittest.TestCase):
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize, reconstruct="FULL")
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        synspace = SyntacticSpace(
            [nObj, contentDim], [nObj, contentDim], [nObj, contentDim],
        )
        synspace.eval()
        x = torch.randn(2, nObj, embDim).to(TheDevice)
        with torch.no_grad():
            y = synspace.forward(x)
            x_rec = synspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-6, f"objSize={objSize}: err={err:.2e}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestOutputSpaceReversePass(unittest.TestCase):
    """OutputSpace changes shape so roundtrip is lossy — just verify no crash."""
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize, outputDim=2, reconstruct="FULL",
                               flatten=True)
        nObj, contentDim, outputDim = 3, 6, 2
        embDim = contentDim + objSize
        torch.manual_seed(42)
        ospace = OutputSpace(
            [nObj, contentDim], [nObj, contentDim], [1, outputDim],
        )
        ospace.eval()
        x = torch.randn(2, nObj, embDim).to(TheDevice)
        with torch.no_grad():
            y = ospace.forward(x)
            x_rec = ospace.reverse(y)
        # Just verify shapes and no crash; roundtrip is lossy by design
        self.assertEqual(y.dim(), 3)

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestErgodicInvertibleLayers(unittest.TestCase):
    """Ergodic mode on invertible layers: noise injection during training."""

    def test_invertible_pi_ergodic_roundtrip(self):
        """InvertiblePiLayer with ergodic=True should still roundtrip after training.

        Directly sets moderate var; high var makes pinv imprecise
        because the effective weight matrix is heavily perturbed by noise.
        """
        torch.manual_seed(42)
        nIn, nOut = 4, 6
        layer = PiLayer(nIn, nOut, naive=True, ergodic=True, invertible=True)
        with torch.no_grad():
            layer.var.fill_(0.1)
            layer.bias.fill_(0.9)
        layer.train()
        x = torch.randn(8, nIn).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.15, f"Ergodic roundtrip error too large: {err}")

    def test_invertible_sigma_ergodic_roundtrip(self):
        """InvertibleSigmaLayer with ergodic=True should still roundtrip after training.

        tanh/atanh amplifies noise so tolerance is higher than for PiLayer.
        """
        torch.manual_seed(42)
        nIn, nOut = 6, 8
        layer = SigmaLayer(nIn, nOut, ergodic=True, invertible=True)
        with torch.no_grad():
            layer.var.fill_(0.05)
            layer.bias.fill_(0.95)
        layer.train()
        x = torch.randn(8, nIn).to(TheDevice)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.3, f"Ergodic roundtrip error too large: {err}")
