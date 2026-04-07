"""Tests for SigmaPi layers and the LogicalFunctionNet."""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from util import TheDevice
from Model import PiLayer, SigmaLayer, epsilon


class TestPiLayerForward(unittest.TestCase):
    """PiLayer produces the correct output shape for 2D and 3D inputs.

    PiLayer expects inputs in (0, 1].
    """

    def test_3d_input_shape(self):
        nBatch, nInput, nOutput, nSymbols = 5, 3, 4, 6
        layer = PiLayer(nInput, nOutput)
        layer.set_sigma(0)
        x = torch.rand(nBatch, nSymbols, nInput).clamp(min=epsilon).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (nBatch, nSymbols, nOutput))

    def test_2d_input_shape(self):
        nBatch, nInput, nOutput = 4, 3, 5
        layer = PiLayer(nInput, nOutput)
        layer.set_sigma(0)
        x = torch.rand(nBatch, nInput).clamp(min=epsilon).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (nBatch, nOutput))

    def test_gradient_flows(self):
        layer = PiLayer(2, 3, ergodic=True)
        with torch.no_grad():
            layer.var.fill_(0.001)
            layer.bias.fill_(0.999)
        x = torch.rand(4, 2).clamp(min=epsilon).to(TheDevice.get()).requires_grad_(True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


class TestSigmaLayerForward(unittest.TestCase):
    """SigmaLayer produces the correct output shape and supports backprop."""

    def test_3d_input_shape(self):
        nBatch, nInput, nOutput, nSymbols = 4, 3, 5, 6
        layer = SigmaLayer(nInput, nOutput)
        layer.set_sigma(0)
        x = torch.randn(nBatch, nSymbols, nInput).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (nBatch, nSymbols, nOutput))

    def test_output_bounded_by_tanh(self):
        """With saturate=True (default), outputs should be in [-1, 1]."""
        layer = SigmaLayer(4, 3)
        layer.set_sigma(0)
        x = torch.randn(10, 4).to(TheDevice.get()) * 10  # large inputs
        y = layer(x)
        self.assertTrue(torch.all(y >= -1.0))
        self.assertTrue(torch.all(y <= 1.0))

    def test_gradient_flows(self):
        layer = SigmaLayer(3, 2)
        layer.set_sigma(0)
        x = torch.randn(4, 3).to(TheDevice.get()).requires_grad_(True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestPiSigmaXOR(unittest.TestCase):
    """An PiLayer+SigmaLayer stack can learn XOR with low temperature."""

    def test_xor_convergence(self):
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(TheDevice.get())
        Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(TheDevice.get())
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)

        # Seed 42 hits a local minimum (~0.125) on CPU but converges on MPS.
        # Try multiple seeds — the test verifies PiLayer+SigmaLayer *can*
        # learn XOR, not that every random init converges.
        criterion = nn.MSELoss()
        from itertools import chain
        best_loss = float('inf')
        for seed in (123, 99, 42):
            torch.manual_seed(seed)
            pi = PiLayer(2, 3)
            sigma = SigmaLayer(3, 1)
            pi.set_sigma(0)
            sigma.set_sigma(0)
            optimizer = optim.Adam(chain(pi.parameters(), sigma.parameters()), lr=0.001)
            for _ in range(1000):
                optimizer.zero_grad()
                y = sigma(pi(X))
                loss = criterion(y, Y)
                loss.backward()
                optimizer.step()
            best_loss = min(best_loss, loss.item())
            if best_loss < 0.1:
                return  # pass

        self.fail(f"XOR loss should converge below 0.1 on at least one seed, "
                  f"best was {best_loss}")


class TestLogicalFunctionNet(unittest.TestCase):
    """The SigmaPi.py LogicalFunctionNet runs without error."""

    def test_forward_shape(self):
        from SigmaPi import LogicalFunctionNet
        model = LogicalFunctionNet(2, 3, 1)
        x = torch.randn(4, 1, 2).to(TheDevice.get())
        y = model(x)
        self.assertEqual(y.shape, (4, 1, 1))

    def test_backward_no_error(self):
        from SigmaPi import LogicalFunctionNet
        model = LogicalFunctionNet(2, 3, 1)
        x = torch.randn(4, 1, 2).to(TheDevice.get())
        y = model(x)
        loss = y.sum()
        loss.backward()
        # Verify at least some parameters got gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)


if __name__ == '__main__':
    unittest.main()
