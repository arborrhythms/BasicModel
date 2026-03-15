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

from Model import (
    LinearLayer, InvertibleLinearLayer, InvertibleRotationLayer,
    InvertibleDiagonalLayer, SigmaLayer, InvertibleSigmaLayer,
    PiLayer, InvertiblePiLayer, AttentionLayer, NormLayer, LiftingLayer,
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

class TestInvertibleRotationLayer(unittest.TestCase):
    def _check(self, dim, naive):
        torch.manual_seed(42)
        layer = InvertibleRotationLayer(dim=dim, naive=naive)
        x = torch.randn(3, dim)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-5,
                        f"dim={dim}, naive={naive}: err={err:.2e}")

    def test_dim4_naive(self):   self._check(4, True)
    def test_dim8_naive(self):   self._check(8, True)
    def test_dim16_naive(self):  self._check(16, True)
    def test_dim4(self):         self._check(4, False)
    def test_dim8(self):         self._check(8, False)
    def test_dim16(self):        self._check(16, False)


class TestInvertibleDiagonalLayer(unittest.TestCase):
    def _check(self, nIn, nOut):
        torch.manual_seed(42)
        layer = InvertibleDiagonalLayer(nIn, nOut)
        x = torch.randn(5, nIn)
        if nIn > nOut:
            x[:, nOut:] = 0.0
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-5, f"{nIn}->{nOut}: err={err:.2e}")

    def test_square(self):       self._check(4, 4)
    def test_expand(self):       self._check(3, 6)
    def test_contract(self):     self._check(6, 3)


class TestInvertibleLinearLayer(unittest.TestCase):
    def _check(self, nIn, nOut, naive, tol):
        torch.manual_seed(42)
        layer = InvertibleLinearLayer(nIn, nOut, naive=naive, hasBias=True)
        x = torch.randn(2, 3, nIn)
        y = layer.forward(x, bias=1.0, temp=0.0)
        x_rec = layer.reverse(y, bias=1.0, temp=0.0)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, tol,
                        f"{nIn}->{nOut}, naive={naive}: err={err:.2e}")

    def test_square_naive(self):    self._check(5, 5, True, 1e-3)
    def test_expand_naive(self):    self._check(5, 8, True, 1e-3)
    def test_contract_naive(self):  self._check(8, 5, True, 10.0)
    def test_square(self):          self._check(5, 5, False, 1e-3)
    def test_expand(self):          self._check(5, 8, False, 1e-3)
    def test_contract(self):        self._check(8, 5, False, 10.0)


class TestLiftingLayer(unittest.TestCase):
    def _check(self, nIn, nOut, tol):
        torch.manual_seed(42)
        layer = LiftingLayer(nIn, nOut)
        x = torch.randn(4, nIn)
        y = layer.forward(x, bias=1.0, temp=0.0)
        x_rec = layer.reverse(y, bias=1.0, temp=0.0)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, tol, f"{nIn}->{nOut}: err={err:.2e}")

    def test_square(self):    self._check(6, 6, 1e-3)
    def test_expand(self):    self._check(6, 10, 1e-3)
    def test_contract(self):  self._check(10, 6, 10.0)


class TestLinearLayerIdentity(unittest.TestCase):
    """LinearLayer with W=I and no bias should be exact identity."""
    def test_identity(self):
        torch.manual_seed(42)
        layer = LinearLayer(5, 5, hasBias=False, W=torch.eye(5))
        x = torch.randn(3, 5)
        y = layer.forward(x, bias=1.0, temp=0.0)
        err = _reconstruction_error(x, y)
        self.assertLess(err, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ergodic layer invertibility
# ═══════════════════════════════════════════════════════════════════════════

class TestInvertibleSigmaLayer(unittest.TestCase):
    def _check(self, nIn, nOut, naive, tol):
        torch.manual_seed(42)
        layer = InvertibleSigmaLayer(nIn, nOut, naive=naive, permuteInput=False)
        layer.setAlpha(1e-9)
        x = torch.randn(2, 3, nIn)
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
        layer = InvertibleSigmaLayer(nIn, nOut, naive=naive, permuteInput=True)
        layer.setAlpha(1e-9)
        x = torch.randn(2, nIn, seqLen)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-3,
                        f"perm=True, naive={naive}: err={err:.2e}")

    def test_permute_naive(self):  self._check_permute(True)
    def test_permute(self):        self._check_permute(False)


class TestInvertiblePiLayer3D(unittest.TestCase):
    def _check(self, naive, perm, bias):
        nIn, nOut = 4, 8
        torch.manual_seed(42)
        layer = InvertiblePiLayer(nIn, nOut, naive=naive,
                                  permuteInput=perm, hasBias=bias)
        layer.setAlpha(1e-9)
        if perm:
            x = torch.randn(3, nIn, 5)
        else:
            x = torch.randn(3, 5, nIn)
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
        layer = InvertiblePiLayer(nIn, nOut, naive=naive,
                                  permuteInput=False, hasBias=bias)
        layer.setAlpha(1e-9)
        x = torch.randn(6, nIn)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 0.1,
                        f"naive={naive}, bias={bias}: rel err={err:.2e}")

    def test_naive_bias(self):     self._check(True, True)
    def test_naive_nobias(self):   self._check(True, False)
    def test_bias(self):           self._check(False, True)
    def test_nobias(self):         self._check(False, False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. NormLayer
# ═══════════════════════════════════════════════════════════════════════════

class TestNormLayerInvertibility(unittest.TestCase):
    def test_roundtrip(self):
        torch.manual_seed(42)
        layer = NormLayer(10, 12, pNorm=2)
        layer.lr = 0
        x = torch.randn(5, 10)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Paired layer training (non-invertible encoder/decoder pairs)
# ═══════════════════════════════════════════════════════════════════════════

class TestPairedSigmaTraining(unittest.TestCase):
    def test_paired_roundtrip(self):
        torch.manual_seed(42)
        nIn, nOut = 6, 8
        sigma_fwd = SigmaLayer(nIn, nOut, ergodic=False)
        sigma_rev = SigmaLayer(nOut, nIn, ergodic=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(sigma_fwd.parameters()) + list(sigma_rev.parameters()), lr=0.01)
        x_data = torch.randn(8, 3, nIn)
        for _ in range(500):
            optimizer.zero_grad()
            y = sigma_fwd(x_data)
            x_rec = sigma_rev(y)
            loss = criterion(x_data, x_rec)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = sigma_fwd(x_data)
            x_rec = sigma_rev(y)
        err = _reconstruction_error(x_data, x_rec, rel=True)
        self.assertLess(err, 0.5, f"Paired Sigma rel err={err:.4f}")


class TestPairedPiTraining(unittest.TestCase):
    """Paired PiLayer roundtrip — demonstrates the Pi layer gradient barrier.

    The PiLayer computes y = b * prod(1 + W*x), a multiplicative layer.
    When training a separate reverse PiLayer to invert the forward one,
    gradients flowing back through the product collapse toward zero
    (each partial derivative is a product of N-1 terms, all near 1,
    making the gradient signal ~1000x smaller than additive layers).

    This is the same gradient barrier that blocks reconstruction through
    PerceptualSpace.reverse() in the full model — the reverse Pi layer
    receives near-zero gradients and cannot learn the inverse mapping.

    Compare with TestPairedSigmaTraining which succeeds: SigmaLayer uses
    y = tanh(W*x + b), an additive layer whose gradients flow cleanly.
    """
    @unittest.expectedFailure
    def test_paired_roundtrip(self):
        torch.manual_seed(42)
        nIn, nOut = 4, 6
        pi_fwd = PiLayer(nIn, nOut, permuteInput=True, ergodic=False)
        pi_rev = PiLayer(nOut, nIn, permuteInput=True, ergodic=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(pi_fwd.parameters()) + list(pi_rev.parameters()), lr=0.01)
        x_data = torch.randn(8, nIn, 5)
        for _ in range(500):
            optimizer.zero_grad()
            y = pi_fwd(x_data)
            x_rec = pi_rev(y)
            loss = criterion(x_data, x_rec)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = pi_fwd(x_data)
            x_rec = pi_rev(y)
        err = _reconstruction_error(x_data, x_rec, rel=True)
        self.assertLess(err, 0.5, f"Paired Pi rel err={err:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Space-level invertibility
# ═══════════════════════════════════════════════════════════════════════════

from BasicModel import (
    TheObjectEncoding, PerceptualSpace, ConceptualSpace,
    SymbolicSpace, OutputSpace, SyntacticSpace,
)


def _setup_object_encoding(objSize=0, contentDim=6, outputDim=2):
    """Configure TheObjectEncoding for isolated Space tests."""
    TheObjectEncoding.objectSize = objSize
    TheObjectEncoding.nWhere = objSize // 2 if objSize > 0 else 0
    TheObjectEncoding.nWhen = objSize - (objSize // 2) if objSize > 0 else 0
    TheObjectEncoding.inputDim = contentDim
    TheObjectEncoding.perceptDim = contentDim
    TheObjectEncoding.conceptDim = contentDim
    TheObjectEncoding.symbolDim = contentDim
    TheObjectEncoding.outputDim = outputDim


class TestPerceptualSpacePassthrough(unittest.TestCase):
    """PerceptualSpace with passThrough=True is exact identity."""
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        pspace = PerceptualSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            quantized=False, reversePass=True,
            passThrough=True, reshape=False,
        )
        pspace.eval()
        x = torch.randn(2, nObj, embDim)
        with torch.no_grad():
            y = pspace.forward(x)
            x_rec = pspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-6, f"objSize={objSize}: err={err:.2e}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestPerceptualSpaceReshapeTrained(unittest.TestCase):
    """PerceptualSpace with reshape=True, trained pair for roundtrip."""
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        pspace = PerceptualSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            quantized=False, reversePass=True, nPrototypes=16,
            passThrough=False, reshape=True, ergodic=False,
            hasAttention=False,
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(pspace.parameters(), lr=0.005)
        x_data = torch.randn(4, nObj, embDim)
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
    def _check(self, objSize, reshape):
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        cspace = ConceptualSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            quantized=False, reversePass=True, nPrototypes=16,
            invertible=True, hasNorm=False, ergodic=False,
            reshape=reshape, hasAttention=False,
        )
        cspace.eval()
        cspace.sigma.setAlpha(1e-9)
        x = torch.randn(2, nObj, embDim)
        with torch.no_grad():
            y = cspace.forward(x)
            x_rec = cspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-2,
                        f"objSize={objSize}, reshape={reshape}: err={err:.2e}")

    def test_reshape_objsize0(self):      self._check(0, True)
    def test_reshape_objsize4(self):      self._check(4, True)
    def test_no_reshape_objsize0(self):   self._check(0, False)
    def test_no_reshape_objsize4(self):   self._check(4, False)


class TestConceptualSpacePairedSigma(unittest.TestCase):
    """ConceptualSpace with paired (non-invertible) sigma layers."""
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        cspace = ConceptualSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            quantized=False, reversePass=True, nPrototypes=16,
            invertible=False, hasNorm=False, ergodic=False,
            reshape=True, hasAttention=False,
        )
        cspace.eval()
        x = torch.randn(2, nObj, embDim)
        with torch.no_grad():
            y = cspace.forward(x)
            x_rec = cspace.reverse(y)
        err = _reconstruction_error(x, x_rec, rel=True)
        self.assertLess(err, 2.0, f"objSize={objSize}: rel err={err:.4f}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestConceptualSpaceHasNorm(unittest.TestCase):
    """ConceptualSpace with hasNorm=True.

    Norm factors (mean, std) are cached during forward and reattached
    during reverse, keeping the sigma layer square for exact invertibility.
    """
    def test_hasNorm_reshape(self):
        _setup_object_encoding(objSize=0)
        nObj, contentDim = 3, 6
        torch.manual_seed(42)
        cspace = ConceptualSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            quantized=False, reversePass=True, nPrototypes=16,
            invertible=True, hasNorm=True, ergodic=False,
            reshape=True, hasAttention=False,
        )
        cspace.eval()
        cspace.sigma.setAlpha(1e-9)
        x = torch.randn(2, nObj, contentDim)
        with torch.no_grad():
            y = cspace.forward(x)
            x_rec = cspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-2, f"err={err:.2e}")


class TestSymbolicSpacePassthrough(unittest.TestCase):
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        sspace = SymbolicSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            reversePass=True, passThrough=True, reshape=True,
        )
        sspace.eval()
        x = torch.randn(2, nObj, embDim)
        with torch.no_grad():
            y = sspace.forward(x)
            x_rec = sspace.reverse(y)
        err = _reconstruction_error(x, x_rec)
        self.assertLess(err, 1e-6, f"objSize={objSize}: err={err:.2e}")

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)


class TestSyntacticSpace(unittest.TestCase):
    def _check(self, objSize):
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim = 3, 6
        embDim = contentDim + objSize
        torch.manual_seed(42)
        synspace = SyntacticSpace(
            inputShape=[nObj, contentDim], outputShape=[nObj, contentDim],
            nVectors=nObj, nDim=contentDim,
            reversePass=True,
        )
        synspace.eval()
        x = torch.randn(2, nObj, embDim)
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
        _setup_object_encoding(objSize=objSize)
        nObj, contentDim, outputDim = 3, 6, 2
        embDim = contentDim + objSize
        torch.manual_seed(42)
        ospace = OutputSpace(
            inputShape=[nObj, contentDim], outputShape=[1, outputDim],
            nVectors=nObj, nDim=contentDim,
            reversePass=True,
        )
        ospace.eval()
        x = torch.randn(2, nObj, embDim)
        with torch.no_grad():
            y = ospace.forward(x)
            x_rec = ospace.reverse(y)
        # Just verify shapes and no crash; roundtrip is lossy by design
        self.assertEqual(y.dim(), 3)

    def test_objsize_0(self):  self._check(0)
    def test_objsize_4(self):  self._check(4)
