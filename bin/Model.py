"""Core layer primitives used by BasicModel.

This module mixes conventional neural-network utilities with a set of
custom reversible, ergodic, and memory-style layers.  Most higher-level
model construction happens in ``BasicModel.py``; this file provides the
building blocks and the update rules they share.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from itertools import chain
import torch.optim as optim
import time

epsilon = 1e-7  # to avoid log(0)

def has_signal(value):
    """Return True when a tensor or scalar contains any non-zero signal."""
    if isinstance(value, torch.Tensor):
        return bool(torch.any(value != 0).item())
    return bool(value)

def sample_noise(reference, shape=None):
    """Sample noise that matches the device and dtype of a reference tensor."""
    if shape is None:
        shape = tuple(reference.shape)
    return torch.randn(shape, device=reference.device, dtype=reference.dtype)

class Message():
    """Tiny callable wrapper so legacy code can swap out message sinks later."""
    def __call__(self, txt, newline="\n"):
        print(txt, end=newline)
message = Message()

#region Layers
class Layer(nn.Module):
    """Base class for custom layers with optional symbol/object axis swapping."""
    def __init__(self, nInput, nOutput, permuteInput=False):
        super(Layer, self).__init__()
        self.nInput       = nInput
        self.nOutput      = nOutput
        self.permuteInput = permuteInput
        self.batch        = 0

    def freeze(self, learn=False):
        for param in self.parameters():
            param.requires_grad = not learn
    def permute(self, x):
        self.batch = x.shape[0]
        # Several layers treat the middle dimension as the symbol axis; this
        # flag lets them reuse the same implementation for either layout.
        if self.permuteInput:
            x = torch.permute(x, (0, 2, 1))
        return x
    def unpermute(self, y):
        if self.permuteInput:
            y = torch.permute(y, (0,2,1))
        return y
    def getParameters(self):
        params = [p for n, p in self.named_parameters()]
        return params
    def paramUpdate(self):
        pass

    def forward(self, x, bias=None, temp=None):
        batch = x.shape[0]
        assert x.shape[1] == self.nSymbols
        #assert x.shape[2] == TheObjectEncoding.objectSize
        return x
    def reverse(self, y, bias=None, temp=None):
        batch = y.shape[0]
        assert y.shape[1] == self.nOutput
        return y
class ErgodicLayer(Layer):
    """Layer base class that adapts its explore/exploit balance over training."""
    def __init__(self, nInput, nOutput, permuteInput=False):
        super().__init__(nInput, nOutput, permuteInput)
        # Alpha controls bias-variance tradeoff: bias = alpha, temp = 1 - alpha
        # Start at full exploration (alpha=0): bias=0, temp=1
        self.alpha       = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.bias        = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # Modified Adam: second moment only (gradient is zero-mean)
        self.register_buffer('v', torch.tensor(0.0))
        self.register_buffer('t_step', torch.tensor(0))            # step counter for bias correction
        self.register_buffer('certainty', torch.ones(nOutput))
        self.register_buffer('certainty_v', torch.zeros(nOutput))
        self.register_buffer('certainty_forward_ema', torch.zeros(nOutput))
        self.register_buffer('certainty_step', torch.tensor(0))
        self.register_buffer('certainty_forward_step', torch.tensor(0))
        self.global_temp = 1.0                                     # sensitivity knob
        self.beta = 0.999
        self.certainty_beta = 0.99
        self.certainty_forward_beta = 0.9
        self.certainty_scale = 1.0
        self.certainty_forward_weight = 0.5
        self.certainty_gradient_weight = 0.5
        self.dropoutRate = 0.0

    def getParameters(self):
        params = [p for n, p in self.named_parameters() if n != "temperature"]
        return params
    def setAlpha(self, alpha):
        with torch.no_grad():
            self.alpha.fill_(alpha)
            self.bias.fill_(alpha)
            self.temperature.fill_(1.0 - alpha)
    def setTemperature(self, temp=0.0):
        with torch.no_grad():
            self.global_temp = temp
            self.setAlpha(1.0 - temp)
    def local_tradeoff(self):
        # High-certainty outputs lean toward bias; uncertain outputs keep more
        # temperature so the layer continues to explore alternatives.
        certainty = self.certainty.to(device=self.bias.device, dtype=self.bias.dtype)
        local_bias = self.bias * certainty
        local_temp = self.temperature * torch.ones_like(certainty) + self.bias * (1.0 - certainty)
        return local_bias, local_temp.clamp(0.0, 1.0)
    @torch.no_grad()
    def reduce_certainty_signal(self, signal):
        if signal is None:
            return None
        signal = signal.detach()
        if signal.ndim == 0 or signal.shape[-1] != self.nOutput:
            return None
        signal = signal.abs()
        if signal.ndim > 1:
            signal = signal.mean(dim=tuple(range(signal.ndim - 1)))
        return torch.tanh(signal).clamp(0.0, 1.0)
    @torch.no_grad()
    def observe_forward_certainty(self, signal):
        signal = self.reduce_certainty_signal(signal)
        if signal is None:
            return
        self.certainty_forward_ema.mul_(self.certainty_forward_beta).add_((1 - self.certainty_forward_beta) * signal)
        self.certainty_forward_step += 1
    @torch.no_grad()
    def forward_certainty(self):
        if self.certainty_forward_step.item() == 0:
            return None
        forward = self.certainty_forward_ema / (1 - self.certainty_forward_beta ** self.certainty_forward_step.item())
        return forward.clamp(0.0, 1.0)
    @torch.no_grad()
    def certainty_gradient_energy(self):
        grad_energy = None
        for name, param in self.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            if name in {"alpha", "bias", "temperature"}:
                continue
            if "noise" in name.lower():
                continue
            if param.ndim == 0 or param.shape[-1] != self.nOutput:
                continue
            energy = param.grad.detach().pow(2)
            if energy.ndim > 1:
                energy = energy.mean(dim=tuple(range(energy.ndim - 1)))
            grad_energy = energy if grad_energy is None else grad_energy + energy
        return grad_energy
    @torch.no_grad()
    def gradient_certainty(self):
        grad_energy = self.certainty_gradient_energy()
        if grad_energy is None:
            return None
        self.certainty_v.mul_(self.certainty_beta).add_((1 - self.certainty_beta) * grad_energy)
        self.certainty_step += 1
        v_hat = self.certainty_v / (1 - self.certainty_beta ** self.certainty_step.item())
        certainty = 1.0 / (1.0 + self.certainty_scale * torch.sqrt(v_hat + epsilon))
        return certainty.clamp(0.0, 1.0)
    @torch.no_grad()
    def combine_certainty_sources(self, forward_certainty=None, gradient_certainty=None):
        combined = None
        total_weight = 0.0
        if forward_certainty is not None:
            combined = self.certainty_forward_weight * forward_certainty
            total_weight += self.certainty_forward_weight
        if gradient_certainty is not None:
            term = self.certainty_gradient_weight * gradient_certainty
            combined = term if combined is None else combined + term
            total_weight += self.certainty_gradient_weight
        if combined is None or total_weight == 0:
            return None
        return (combined / total_weight).clamp(0.0, 1.0)
    @torch.no_grad()
    def certainty_update(self):
        certainty = self.combine_certainty_sources(
            forward_certainty=self.forward_certainty(),
            gradient_certainty=self.gradient_certainty(),
        )
        if certainty is not None:
            # Certainty is tracked per output column so later passes can damp
            # exploration only where the layer has become reliable.
            self.certainty.copy_(certainty)
    @torch.no_grad()
    def alpha_update(self):
        if self.temperature.grad is None:
            return
        grad = self.temperature.grad.item()
        # Modified Adam: EMA of gradient energy (second moment only)
        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
        self.t_step += 1
        # Bias correction (same as Adam)
        v_hat = self.v / (1 - self.beta ** self.t_step)
        # Alpha = sensor output: high gradient energy → low alpha (explore)
        self.alpha.fill_(1.0 / (1.0 + self.global_temp * v_hat.sqrt()))
        # Derive bias and temp from alpha
        alpha = self.alpha.item()
        if random.random() < self.dropoutRate:
            # Occasional bias dropout forces another exploration step even when
            # the layer has converged toward exploitation.
            self.bias.fill_(0.0)
        else:
            self.bias.fill_(alpha)
        self.temperature.fill_(1.0 - alpha)
        self.temperature.grad.zero_()
    def paramUpdate(self):
        self.certainty_update()
        self.alpha_update()
    def global_temp_anneal(self, progress):
        """Anneal global_temp from 1.0→0.0 over training.
        progress: float 0..1 (fraction of training complete)."""
        self.global_temp = max(0.0, 1.0 - progress)

class LinearLayer(Layer):
    def __init__(self, nInput, nOutput, hasBias=True, W=None):
        super(LinearLayer, self).__init__(nInput, nOutput)
        self.hasBias = hasBias
        if W == None:
            W = torch.eye(self.nInput, self.nOutput)
        self.W      = nn.Parameter(W)
        self.register_buffer('noise', torch.randn(self.nInput, self.nOutput))
        self.bias   = nn.Parameter(torch.zeros(1,nOutput))
        self.register_buffer('biasNoise', torch.randn(1, nOutput))

    def resample_noise(self):
        self.noise = sample_noise(self.W)
        if self.hasBias:
            self.biasNoise = sample_noise(self.bias)

    def forward(self, x, bias=1.0, temp=0.0):
        if has_signal(temp):
            self.resample_noise()
        W = bias * self.W + temp * self.noise
        output = x @ W
        if self.hasBias:
            output += bias * self.bias + temp * self.biasNoise
        return output

    @staticmethod
    def test():
        nInput, nOutput = 3, 4
        W = torch.rand(nInput, nOutput)
        layer = LinearLayer(nInput=nInput, nOutput=nOutput, W = W)
        # note: x must be strictly positive for this to work
        input = torch.rand((1, nInput))
        output = layer(input)

        print(f"Input: {input}")
        print(f"After forward linear: {output}")
class ReversibleRotationLayer(Layer):
    def __init__(self, dim, naive=False, theta=None):
        super(ReversibleRotationLayer, self).__init__(dim, dim)
        self.dim = dim
        self.naive = naive
        if theta is None:
            theta = torch.randn(dim - 1) * 2 * torch.pi
        self.theta = nn.Parameter(theta)
        # For simplicity, we initialize noise deterministically.
        self.register_buffer('noise', torch.zeros(dim - 1, dtype=self.theta.dtype))

    def givens_rotation(self, i, j, theta):
        # Build a rotation matrix of size (dim x dim) that rotates in the (i,j) plane.
        R = torch.eye(self.dim, device=theta.device, dtype=theta.dtype)
        c, s = torch.cos(theta), torch.sin(theta)
        R[i, i] = c
        R[j, j] = c
        R[i, j] = -s
        R[j, i] = s
        return R
    def rotation_matrix(self, bias=1.0, temp=0.0):
        R_total = torch.eye(self.dim, device=self.theta.device, dtype=self.theta.dtype)
        for idx, angle in enumerate(self.theta):
            angle = bias*angle + temp * self.noise[idx]
            # Use consecutive indices. (Assumes len(theta)==dim-1.)
            R = self.givens_rotation(idx, idx + 1, angle)
            R_total = R @ R_total
        return R_total
    def forward(self, x, bias=1.0, temp=0.0):
        # For the non-naive implementation, apply each rotation sequentially.
        if self.naive:
            # Naive: build full rotation matrix and multiply on the right.
            R = self.rotation_matrix(bias, temp)
            return x @ R  # x (batch, dim) multiplied by R (dim, dim)
        else:
            # Direct approach applying each Givens rotation sequentially.
            x_rotated = x.clone()
            # Note: We assume rotations are applied in natural order.
            for idx, angle in enumerate(self.theta):
                angle = bias*angle + temp * self.noise[idx]
                c, s = torch.cos(angle), torch.sin(angle)
                xi = x_rotated[:, idx].clone()
                xj = x_rotated[:, idx + 1].clone()
                # Standard Givens rotation: [xi, xj] -> [c*xi - s*xj, s*xi + c*xj]
                x_rotated[:, idx] = c * xi - s * xj
                x_rotated[:, idx + 1] = s * xi + c * xj
            return x_rotated
    def forwardTranspose(self, x, bias=1.0, temp=0.0):
        # For the non-naive implementation, apply each rotation sequentially in transpose (inverse) form.
        if self.naive:
            # Naive: build full rotation matrix and multiply by its transpose
            R = self.rotation_matrix(bias, temp)
            return x @ R.T  # x (batch, dim) multiplied by R^T (dim, dim)
        else:
            # Sequentially apply inverse Givens rotations
            x_rotated = x.clone()
            # Apply in reverse order for exact transpose behavior
            for idx in reversed(range(len(self.theta))):
                angle = bias * self.theta[idx] + temp * self.noise[idx]
                c, s = torch.cos(angle), torch.sin(angle)
                # Inverse rotation: negate sine term
                xi = x_rotated[:, idx].clone()
                xj = x_rotated[:, idx + 1].clone()
                x_rotated[:, idx] = c * xi + s * xj
                x_rotated[:, idx + 1] = -s * xi + c * xj
            return x_rotated
    def reverse(self, x, bias=1.0, temp=0.0):
        # Since rotations are orthonormal, the inverse is given by the transpose.
        if self.naive:
            R = self.rotation_matrix(bias, temp).T
            return x @ R
        else:
            x_rotated = x.clone()
            # Reverse the sequence in the opposite order.
            for idx in reversed(range(len(self.theta))):
                angle = bias * self.theta[idx] + temp * self.noise[idx]
                c, s = torch.cos(angle), torch.sin(angle)
                xi = x_rotated[:, idx].clone()
                xj = x_rotated[:, idx + 1].clone()
                # Inverse rotation: [xi, xj] -> [c*xi + s*xj, -s*xi + c*xj]
                x_rotated[:, idx] = c * xi + s * xj
                x_rotated[:, idx + 1] = -s * xi + c * xj
            return x_rotated
    def reverseTranspose(self, x, bias=1.0, temp=0.0):
        # This is the inverse of forwardTranspose()
        if self.naive:
            R = self.rotation_matrix(bias, temp)
            return x @ R  # forwardTranspose naive used R.T, so its inverse uses R
        else:
            x_rotated = x.clone()
            # Apply in forward order (opposite of reverse()) for transpose-of-inverse
            for idx in range(len(self.theta)):
                angle = bias * self.theta[idx] + temp * self.noise[idx]
                c, s = torch.cos(angle), torch.sin(angle)
                # Undo the negated sine from forwardTranspose
                xi = x_rotated[:, idx].clone()
                xj = x_rotated[:, idx + 1].clone()
                x_rotated[:, idx] = c * xi - s * xj
                x_rotated[:, idx + 1] = s * xi + c * xj
            return x_rotated
    @staticmethod
    def test():
        dim    = 4
        theta  = torch.rand(dim - 1) * torch.pi / 2
        nlayer = ReversibleRotationLayer(dim=dim, naive=True, theta=theta)
        layer  = ReversibleRotationLayer(dim=dim, naive=False, theta=theta)
        x      = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

        # Test forward pass
        nrotated_x = nlayer.forward(x)
        rotated_x = layer.forward(x)
        print(f"Layer Weights: {layer.rotation_matrix()}")
        print(f"Original: {x}")
        print(f"Rotated Naive: {nrotated_x}")
        print(f"Rotated: {rotated_x}")
        print(f"Naive - Non-naive difference: {torch.norm(nrotated_x - rotated_x)}")

        # Test reverse pass
        ninverse_x = nlayer.reverse(nrotated_x)
        inverse_x = layer.reverse(rotated_x)
        print(f"Inverse Naive Rotation: {ninverse_x}")
        print(f"Inverse Rotation: {inverse_x}")
        print(f"Forward-Reverse check (naive): {torch.norm(x - ninverse_x)}")
        print(f"Forward-Reverse check (non-naive): {torch.norm(x - inverse_x)}")
class ReversibleDiagonalLayer(Layer):
    def __init__(self, nInput, nOutput):
        super(ReversibleDiagonalLayer, self).__init__(nInput, nOutput)
        self.nInput = nInput
        self.nOutput = nOutput
        self.rank = min(nInput, nOutput)
        # Store the nonzero singular values (for the effective rank).
        self.lamda = nn.Parameter(torch.ones(self.rank))
        self.register_buffer('noise', torch.zeros(self.rank))

    def stabilize(self):
        self.lamda.data = torch.minimum(self.lamda, torch.ones_like(self.lamda))
    def forward(self, x, bias=1.0, temp=0.0):
        # Compute the effective singular values.
        w = bias * self.lamda + temp * self.noise
        # Build the Sigma matrix S of shape (nOutput x nInput)
        S = torch.zeros(self.nOutput, self.nInput, device=x.device, dtype=x.dtype)
        for i in range(self.rank):
            S[i, i] = w[i]
        # Multiply input x (shape (..., nInput)) on the right by Sᵀ, so output is (..., nOutput)
        y = x @ S.T
        return y
    def reverse(self, y, bias=1.0, temp=0.0):
        w = bias * self.lamda + temp * self.noise
        # Inversion: We assume the effective mapping is invertible on the first 'rank' coordinates.
        if self.nInput <= self.nOutput:
            x = y[..., :self.nInput] / w
        else:
            x_known = y / w
            pad_shape = list(x_known.shape[:-1]) + [self.nInput - self.nOutput]
            x_pad = torch.zeros(pad_shape, device=y.device, dtype=y.dtype)
            x = torch.cat([x_known, x_pad], dim=-1)
        return x
    @staticmethod
    def test():
        """
        Runs several tests on the ReversibleDiagonalLayer:
          1. Square case: nInput == nOutput.
          2. Wide-output case: nInput < nOutput.
          3. Tall-input case: nInput > nOutput.
          4. Batch test.
        For the tall-input case and batch test (when nInput > nOutput), the tests force the extra input coordinates
        to zero so that the mapping is invertible.
        """
        print("Testing ReversibleDiagonalLayer...")

        # Test 1: Square case: nInput == nOutput.
        nInput, nOutput = 4, 4
        layer = ReversibleDiagonalLayer(nInput, nOutput)
        x = torch.rand((10, nInput))
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5), (
            f"Square test failed for nInput={nInput}, nOutput={nOutput}\n"
            f"x: {x}\nx_rec: {x_rec}"
        )
        print("Square test passed.")

        # Test 2: Wide output: nInput < nOutput.
        nInput, nOutput = 3, 5
        layer = ReversibleDiagonalLayer(nInput, nOutput)
        x = torch.rand((10, nInput))
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5), (
            f"Wide output test failed for nInput={nInput}, nOutput={nOutput}\n"
            f"x: {x}\nx_rec: {x_rec}"
        )
        print("Wide output test passed.")

        # Test 3: Tall input: nInput > nOutput.
        nInput, nOutput = 5, 3
        layer = ReversibleDiagonalLayer(nInput, nOutput)
        x = torch.rand((10, nInput))
        # For the mapping to be invertible, force the extra dimensions (nOutput: nInput) to zero.
        if nInput > nOutput:
            x[:, nOutput:] = 0.0
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5), (
            f"Tall input test failed for nInput={nInput}, nOutput={nOutput}\n"
            f"x: {x}\nx_rec: {x_rec}"
        )
        print("Tall input test passed.")

        # Test 4: Batch behavior.
        nInput, nOutput = 6, 4
        layer = ReversibleDiagonalLayer(nInput, nOutput)
        x = torch.rand((7, nInput))
        if nInput > nOutput:
            x[:, nOutput:] = 0.0
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5), (
            f"Batch behavior test failed for nInput={nInput}, nOutput={nOutput}\n"
            f"x: {x}\nx_rec: {x_rec}"
        )
        print("Batch behavior test passed.")

        print("All tests passed!")
class ReversibleLinearLayer(Layer):
    def __init__(self, nInput, nOutput, naive=False, hasBias=True, stable=False):
        super(ReversibleLinearLayer, self).__init__(nInput, nOutput)
        self.naive = naive
        self.hasBias = hasBias
        self.rank = min(nInput, nOutput)
        self.stable = stable

        # U is a square rotation matrix on the input side (nInput × nInput)
        self.U = ReversibleRotationLayer(dim=nInput, naive=self.naive)
        # V is a square rotation matrix on the output side (nOutput × nOutput)
        self.V = ReversibleRotationLayer(dim=nOutput, naive=self.naive)
        # Sigma now builds an (nInput × nOutput) diagonal matrix.
        self.Sigma = ReversibleDiagonalLayer(nInput, nOutput)

        if self.naive:
            self.register_buffer('noise', torch.randn(nInput, nOutput))
        if self.hasBias:
            self.bias = nn.Parameter(torch.zeros(1, nOutput))
            self.register_buffer('biasNoise', torch.randn(1, nOutput))

    def resample_naive_noise(self):
        self.noise = sample_noise(self.Sigma.lamda, shape=(self.nInput, self.nOutput))

    def resample_bias_noise(self):
        if self.hasBias:
            self.biasNoise = sample_noise(self.bias)

    def compute_W(self):
        # Compute full weight: W = U · Σ · Vᵀ.
        U_matrix = self.U.rotation_matrix()  # shape: (nInput, nInput)
        # Build Σ of shape (nInput, nOutput)
        Sigma_matrix = torch.zeros(self.nInput, self.nOutput, device=self.Sigma.lamda.device,
                                   dtype=self.Sigma.lamda.dtype)
        for i in range(self.rank):
            Sigma_matrix[i, i] = self.Sigma.lamda[i]
        V_matrix = self.V.rotation_matrix()  # shape: (nOutput, nOutput)
        # W: (nInput, nOutput)
        W = U_matrix @ Sigma_matrix @ V_matrix.T
        return W
    def compute_Winverse(self):
        # Compute pseudoinverse: W_inv = V · Σ⁻¹ · Uᵀ.
        U_matrix = self.U.rotation_matrix()  # shape: (nInput, nInput)
        Sigma_inv = torch.zeros(self.nOutput, self.nInput, device=self.Sigma.lamda.device,
                                dtype=self.Sigma.lamda.dtype)
        for i in range(self.rank):
            Sigma_inv[i, i] = 1.0 / self.Sigma.lamda[i]
        V_matrix = self.V.rotation_matrix()  # shape: (nOutput, nOutput)
        # W_inv: shape (nOutput, nInput)
        W_inv = V_matrix @ Sigma_inv @ U_matrix.T
        return W_inv
    def forward(self, x, bias=1.0, temp=0.0):
        if self.stable:
            self.Sigma.stabilize()
        if self.naive:
            self.resample_naive_noise()
            W = bias * self.compute_W() + temp * self.noise
            # Naive forward: y = x @ W
            y = x @ W
        else:
            # Non-naive branch: apply U, then Σ, then Vᵀ.
            xShape = list(x.shape)
            xShape[-1] = self.nOutput
            x = x.reshape(-1, self.nInput)
            x1 = self.U.forward(x, bias, temp)  # shape: (batch, nInput)
            x2 = self.Sigma.forward(x1, bias, temp)  # shape: (batch, nOutput)
            y = self.V.forward(x2, bias, temp)
            y = y.reshape(xShape)
        if self.hasBias:
            self.resample_bias_noise()
            y += bias * self.bias + temp * self.biasNoise
        return y
    def reverse(self, y, bias=1.0, temp=0.0):
        if self.hasBias:
            y -= bias * self.bias + temp * self.biasNoise
        if self.naive:
            W_inv = bias * self.compute_Winverse() + temp * self.noise.T
            # Naive reverse: x = y @ W_inv
            x = y @ W_inv
            self.resample_naive_noise()
        else:
            yShape = list(y.shape)
            yShape[-1] = self.nInput
            y = y.reshape(-1, self.nOutput)
            # Non-naive reverse: undo Vᵀ, then Σ, then U.
            x2 = self.V.reverse(y, bias, temp)
            x1 = self.Sigma.reverse(x2, bias, temp)  # shape: (batch, nInput)
            x = self.U.reverse(x1, bias, temp)  # shape: (batch, nInput)
            x = x.reshape(yShape)
        if self.hasBias:
            self.resample_bias_noise()
        return x

    @staticmethod
    def test():
        torch.manual_seed(42)
        nInput, nOutput = 7, 11
        layer = ReversibleLinearLayer(nInput=nInput, nOutput=nOutput, naive=True)
        gLayer = ReversibleLinearLayer(nInput=nInput, nOutput=nOutput, naive=False)

        # Disable noise completely for testing
        b = 1
        t = 0.000001
        # Create test input
        x = torch.rand((2, 5, nInput))
        print(f"Input shape: {x.shape}")

        # Test naive implementation
        start_time = time.time()
        y_naive = layer.forward(x, b, t)
        x_naive_restored = layer.reverse(y_naive, b, t)
        naive_error = torch.norm(x - x_naive_restored)
        print(f"Naive output shape: {y_naive.shape}")
        print(f"Naive reconstruction error: {naive_error}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        # Test givens implementation
        start_time = time.time()
        y_givens = gLayer.forward(x, b, t)
        x_givens_restored = gLayer.reverse(y_givens, b, t)
        givens_error = torch.norm(x - x_givens_restored)
        print(f"Givens output shape: {y_givens.shape}")
        print(f"Givens reconstruction error: {givens_error}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        # Compare outputs
        output_diff = torch.norm(y_naive - y_givens)
        print(f"Output difference: {output_diff}")

        # Now test with assertions
        assert naive_error < 0.001, f"Naive implementation failed with error {naive_error}"
        assert givens_error < 0.001, f"Givens implementation failed with error {givens_error}"
        print("All tests passed!")

# A layer to map between row and column operations of other layers
class LiftingLayer(ReversibleLinearLayer):
    def __init__(self, nInput, nOutput, init='orthogonal'):
        super(LiftingLayer, self).__init__(nInput, nOutput, naive=False, hasBias=False, stable=True)

class ColumnUsageTracker:
    def __init__(self, linearLayer, freezeThreshold=0.01, window=10):
        self.linear = linearLayer
        self.freezeThreshold = freezeThreshold
        self.window = window
        self.grad_history = []
        self.frozen_columns = torch.zeros(linearLayer.weight.shape[1], dtype=torch.bool)
        # Register backward hook to capture gradients
        self.linear.weight.register_hook(self._save_grad)
    def _save_grad(self, grad):
        # grad shape: (out_features, in_features)
        # Transpose to get per-column: shape (in_features, out_features)
        grad_columns = grad.t().detach().clone()
        self.grad_history.append(grad_columns)
        # Keep fixed window size
        if len(self.grad_history) > self.window:
            self.grad_history.pop(0)
    def freeze(self):
        if len(self.grad_history) < self.window:
            return  # not enough history yet
        # Stack history and compute norm per column
        stacked = torch.stack(self.grad_history, dim=0)  # (window, in_features, out_features)
        norms = stacked.norm(dim=-1).mean(dim=0)  # (in_features,)
        # Freeze columns with low average gradient norm
        to_freeze = norms < self.freezeThreshold
        self.frozen_columns |= to_freeze
        # Zero out gradients of frozen columns
        with torch.no_grad():
            self.linear.weight.grad[:, self.frozen_columns] = 0.0
    def freezeMask(self):
        return self.frozen_columns

# Create a differentiable map between spaces using quantized vectors in each domain.
class SoftMap(Layer):
    def __init__(self, nInput, nOutput, nInputCodes=None, nOutputCodes=None, soft=False, beta=10.0):
        super().__init__(nInput, nOutput)
        self.nInputCodes  = nInputCodes if nInputCodes is not None else 2*nInput
        self.nOutputCodes = nOutputCodes if nOutputCodes is not None else 2*nOutput
        self.soft         = soft
        self.beta         = beta
        self.codebookX = nn.Parameter(torch.randn(nInput, self.nInputCodes))   # (nInput, nInputCodes)
        self.codebookY = nn.Parameter(torch.randn(nOutput, self.nOutputCodes)) # (nOutput, nOutputCodes)
        self.penalty = nn.Parameter(torch.zeros(1))
        self.SVD = ReversibleLinearLayer(nInput, nOutput, naive=True)

    def quantize(self, v, codebook):
        dist = torch.cdist(v.reshape(-1, v.shape[-1]), codebook.T) ** 2  # (batch*seq, nCodes)
        if self.soft:
            weights = F.softmax(-self.beta * dist, dim=-1)  # (batch*seq, nCodes)
            quantized = weights @ codebook.T  # (batch*seq, dim)
        else:
            indices = torch.argmin(dist, dim=-1)  # (batch*seq,)
            quantized = codebook.T[indices]  # (batch*seq, dim)
        return quantized.view(*v.shape)

    def forward(self, x, t=0.0):
        qx = self.quantize(x, self.codebookX)
        y  = self.SVD.forward(qx, t)
        qy = self.quantize(y, self.codebookY)
        return qy, qx
        #rx = self.SVD.reverse(qy, 0)
        #self.penalty[:] = torch.norm(qx - rx)
        #return qy + self.penalty, qx

    def reverse(self, y, t=0.0):
        qy = self.quantize(y, self.codebookY)
        x  = self.SVD.reverse(qy, t) # Calling reverse before forward will not maintain the weights here ...
        qx = self.quantize(x, self.codebookX)
        #ry = self.SVD.forward(qx, 0)
        #penalty = torch.norm(qy - ry)
        return qx, qy

    def codebook_regularization(self):
        normX = F.normalize(self.codebookX, dim=0)
        normY = F.normalize(self.codebookY, dim=0)
        simX = torch.matmul(normX.T, normX)
        simY = torch.matmul(normY.T, normY)
        lossX = ((simX - torch.eye(simX.shape[0], device=simX.device))**2).sum()
        lossY = ((simY - torch.eye(simY.shape[0], device=simY.device))**2).sum()
        return lossX + lossY

    def normalize_codebooks(self):
        with torch.no_grad():
            self.codebookX.data = F.normalize(self.codebookX.data, dim=0)
            self.codebookY.data = F.normalize(self.codebookY.data, dim=0)

    @staticmethod
    def test():
        torch.manual_seed(0)
        nInput  = 4
        nOutput = 5
        mapper = SoftMap(nInput=nInput, nOutput=nOutput, beta=10.0)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mapper.parameters(), lr=0.001)
        epochs = 1000
        t = 0.0001
        for epoch in range(epochs):
            optimizer.zero_grad()
            xc = mapper.codebookX[:, 1]
            x = torch.rand([2, 3, nInput])
            # ---------- Training pass (soft) ----------
            qy, qx = mapper.forward(x)
            qx_rec, qy_rec = mapper.reverse(qy)
            loss = criterion(x, qx_rec) + criterion(qy, qy_rec) + 0.01 * mapper.codebook_regularization()
            #y = 2*x
            #loss = criterion(x, qx_rec)
            loss.backward()
            optimizer.step()
            mapper.normalize_codebooks()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, MSE: {loss.item():.6f}')
        # ---------- Evaluation (invertibility) using hard quantization ----------
        x[0, 0, :] = xc
        qy, qx = mapper.forward(x)
        qx_rec, qy_rec = mapper.reverse(qy)
        error = (x[0, 0, :] - qx_rec[0, 0, :]).norm(p=2).item()
        print("Reconstruction error (L2 norm):", error)
        assert error < 1e-3, f"Reconstruction error too high: {error}"

class SigmaLayer(ErgodicLayer):
    def __init__(self, nInput, nOutput, permuteInput=False, deterministic=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.layer       = LinearLayer(nInput, nOutput, hasBias=True)
        self.saturate    = True
        self.activation  = torch.zeros(1,nOutput,1)
        self.deterministic = deterministic

    def layer_tradeoff(self):
        return self.local_tradeoff()

    def forward(self, x):
        if self.deterministic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0      # pure learned weights, no noise
        else:
            bias, temp = self.layer_tradeoff()
        x = self.permute(x)
        y = self.layer.forward(x, bias, temp)   # (batch_size, output_dim)
        if self.saturate:
            self.activation = torch.tanh(y)
            y = self.activation.clone()
        self.observe_forward_certainty(y)
        y = self.unpermute(y)
        return y

    @staticmethod
    def test():
        nInput, nOutput = 3, 4
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput)

        x = torch.randn((2, 5, nInput))
        layer.setTemperature(0.001)

        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(layer.getParameters(), lr=0.01)  # Adam Optimizer
        optimizer.zero_grad()  # Clear gradients
        y = layer(x)
        #y_inv = layer.reverse(y)

        loss = criterion(y, y)  # Compute loss
        loss.backward()  # Backpropagation

        print(f"Original input: {x}")
        print(f"After linear: {y}")
        #print(f"Inverse operation result: {y_inv}")
class ReversibleSigmaLayer(SigmaLayer):
    def __init__(self, nInput, nOutput, naive=False, permuteInput=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.layer          = ReversibleLinearLayer(nInput, nOutput, naive=naive, hasBias=True)
    def layer_tradeoff(self):
        if self.layer.naive:
            return self.local_tradeoff()
        return self.bias, self.temperature
    def reverse(self, y):
        y  = self.permute(y)
        y = y.squeeze(0)
        if self.saturate:
            self.activation = torch.atanh(y) # this can be faster if we keep the tanh activation
            y = self.activation.clone()
        x  = self.layer.reverse(y, self.bias, self.temperature)  # (batch_size, output_dim)
        x = self.unpermute(x)
        return x

    @staticmethod
    def test():
        nInput, nOutput = 5, 7
        permute = False
        #naive = False
        layer   = ReversibleSigmaLayer(nInput=nInput, nOutput=nOutput, permuteInput=permute, naive=False)

        x = torch.randn((2, 5, nInput))
        layer.setTemperature(0.000000001)
        y = layer.forward(x)
        y_inv = layer.reverse(y)

        #print(f"Original input: {x}")
        #print(f"After reversible linear: {y}")
        #print(f"Inverse operation result: {y_inv}")
        assert(torch.norm(x-y_inv) < 0.00001)

        layer = ReversibleSigmaLayer(nInput=nInput, nOutput=nOutput, permuteInput=False, naive=True)
        x = torch.randn((4, 8, nInput))
        layer.setTemperature(0.00000001)
        y = layer.forward(x)
        assert y.shape == (4,8,nOutput), "Incorrect Size"
        y_inv = layer.reverse(y)
        assert(torch.norm(x-y_inv) < 0.00001)
class PiLayer(ErgodicLayer):
    def __init__(self, nInput, nOutput, permuteInput=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.weights          = nn.Parameter(torch.zeros(nInput, nOutput))
        self.register_buffer('noise', torch.randn(nInput, nOutput))
        self.biasWeight       = nn.Parameter(torch.zeros(1, 1, self.nOutput))  # Per-output-feature bias
        self.register_buffer('biasWeightNoise', torch.randn(1, self.nInput, self.nOutput))  # Per-output-feature bias

        self.saturate      = True
        self.hasBiasWeight = True
        self.useEpsilon    = True

    def resample_noise(self):
        self.noise = sample_noise(self.weights)
        self.biasWeightNoise = sample_noise(self.weights, shape=(1, self.nInput, self.nOutput))

    def forward(self, x):
        bias, temp = self.local_tradeoff()
        x = self.permute(x)
        # This method implements PI(1+wx)
        # Tried to implement as exp( SIGMA(log(e) * log(wx)) ) so that we could invert with standard methods,
        # but the matrix log(e) is not invertible. If we try simply PI(w+x), we get a weight matrix with negative values,
        # so we can't take it into the log domain.
        if has_signal(temp):
            self.resample_noise()
        w = bias * self.weights + temp * self.noise
        ndim = len(x.shape)
        assert self.nInput == x.shape[-1], "Incorrect shape in piLayer"
        # x: shape (N, J, K), w: shape (K, L)
        # Expand x to (N, J, K, 1) and w to (1, 1, K, L) so they can broadcast together:
        if ndim == 2:
            WX = x.unsqueeze(-1) * w.unsqueeze(0)
            if self.hasBiasWeight:
                WX += (bias * self.biasWeight + temp * self.biasWeightNoise)
            if self.saturate:
                term = 1 + torch.tanh(WX)  # shape (N, J, K)
            else:
                term = 1 + WX
            output = torch.prod(term, dim=1)  # result has shape (N, L)
        else:
            # x: shape (N, J, K), w: shape (K, L)
            # Expand x to (N, J, K, 1) and w to (1, 1, K, L) so they can broadcast together:
            x2 = x.unsqueeze(-1)
            w2 = w.unsqueeze(0)
            WX = x2 * w2
            if self.hasBiasWeight:
                WX += (bias*self.biasWeight.unsqueeze(1) + temp*self.biasWeightNoise.unsqueeze(1))
            if self.saturate:
                term = 1 + torch.tanh(WX)  # shape (N, J, K)
                if self.useEpsilon:
                    term += epsilon
            else:
                term  = 1 + WX
            # Compute the product along the J dimension:
            output   =  torch.prod(term, dim=2)  # result has shape (N, K, L)
        self.observe_forward_certainty(output)
        output = self.unpermute(output)
        return output

    @staticmethod
    def test():
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayer(nInput=nInput, nOutput=nOutput, permuteInput=True)

        # x must be positive
        x = torch.randn((nBatch, nInput, 6))
        layer.setTemperature(0.001)
        y = layer(x)
        assert(y.shape==(nBatch, nOutput, 6))

        print(f"Original input: {x}")
        print(f"After linear: {y}")
    @staticmethod
    def xorTest():
        X = torch.tensor([
            [0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        Y = torch.tensor([
            [0], [1], [1], [0]], dtype=torch.float32)
        X = X.unsqueeze(2)
        Y = Y.unsqueeze(2)
        nInput, nHidden, nOutput = (2,3,1)
        pi    = PiLayer(nInput, nHidden, permuteInput=True)      # Hidden layer using PiLayer
        sigma = SigmaLayer(nHidden, nOutput, permuteInput=True)  # Output layer using SigmaLayer

        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(chain(pi.parameters(), sigma.parameters()), lr=0.01)  # Adam Optimizer
        epochs = 1000
        sigma.setTemperature(0.0001)
        pi.setTemperature(0.0001)
        for epoch in range(epochs):
            optimizer.zero_grad()  # Clear gradients

            x1 = pi(X)  # Pass through PiLayer
            y = sigma(x1)  # Pass through SigmaLayer

            loss = criterion(y, Y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, MSE: {loss.item():.6f}')
class ReversiblePiLayer(ErgodicLayer):
# 	•	This computes:
# y_j = b_j \prod_i \left(1 - \tanh(w_{ji} x_i)\right)
# z_j = b_j \prod_i \left(1 + \tanh(w_{ji} x_i)\right)
# 	•	These match up with:
# \gamma_j = \frac{1}{2} \log\left( \frac{z_j}{y_j} \right) = \sum_i \tanh^{-1}(\tanh(w_{ji} x_i)) = \sum_i w_{ji} x_i = (W x)_j
# 	•	And thus:
# x \approx W^\dagger \gamma
#The full inversion process is:
#	1.	Define:
#s_j := \sum_i \tanh^{-1}(w_{ji} x_i) = W x
#	2.	In forward pass, use:
#y_j = b_j \cdot \prod_i (1 - \tanh(w_{ji} x_i)), \quad
#z_j = b_j \cdot \prod_i (1 + \tanh(w_{ji} x_i))
#	3.	In reverse pass, compute:
#\gamma_j = \frac{1}{2} \log\left( \frac{z_j}{y_j} \right) = \sum_i \tanh^{-1}(\tanh(w_{ji} x_i)) = \sum_i w_{ji} x_i
#\Rightarrow \gamma = W x
#	4.	Finally:
#x = W^\dagger \gamma

    def __init__(self, nInput, nOutput, naive=False, permuteInput=False, hasBias = True):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.naive   = naive
        self.hasBias = hasBias
        self.useEpsilon = True
        if not self.naive:
            assert 2*nInput == nOutput, "There must be twice as many outputs as inputs for invertibility when using the non-naive algorithm."
        #self.saturate = True # saturation is currently necessary for a smooth inversion
        # W: shape (out_features, in_features)
        if naive:
            self.W      = nn.Parameter(torch.randn(nInput, nOutput))
            self.register_buffer('noise', torch.randn(nInput, nOutput))
        else:
            self.layer  = ReversibleLinearLayer(nInput, nOutput, naive=True)
            self.register_buffer('noise', torch.randn(nOutput, nInput))
        self.biasWeight       = nn.Parameter(torch.zeros(1, 1, self.nOutput))  # Per-output-feature bias
        self.register_buffer('biasNoise', torch.randn(1, 1, self.nOutput))  # Per-output-feature bias

    def resample_noise(self):
        if self.naive:
            self.noise = sample_noise(self.W)
        else:
            self.noise = sample_noise(self.biasWeight, shape=(self.nOutput, self.nInput))
        self.biasNoise = sample_noise(self.biasWeight, shape=(1, 1, self.nOutput))

    def forward(self, x):
        bias, temp = self.bias, self.temperature

        """
        x: (batch_size, in_features)
        Output: yz: (batch_size, out_features, 2)
        y_j = b_j * prod_i (1 - tanh(w_{ji} * x_i))
        z_j = b_j * prod_i (1 + tanh(w_{ji} * x_i))
        """
        if temp != 0:
            self.resample_noise()
        x = self.permute(x)

        if not self.naive:
            W = self.layer.compute_W().T # XXX seems not to use bias, variance
            W  = (bias * W + temp * self.noise)
            WX = x.unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)
        else:
            W  = (bias * self.W + temp * self.noise).unsqueeze(0).unsqueeze(0)
            WX =  x.unsqueeze(-1) * W                 # (batch, out_features, in_features)
        if self.hasBias:
            WX = WX + (bias*self.biasWeight.unsqueeze(1) + temp*self.biasNoise.unsqueeze(1))

        # Compute tanh(w_{ji} * x_i)
        sWX = torch.tanh(WX)
        # Apply (1 - tanh(...)) and (1 + tanh(...))
        one_minus = 1 - sWX
        one_plus  = 1 + sWX
        if self.useEpsilon:
            one_minus += epsilon
            one_plus  += epsilon
        # Product over input dimension (i)
        y = torch.prod(one_minus, dim=2)
        z = torch.prod(one_plus, dim=2)
        #s = torch.concatenate((y, z), dim=1)  # (batch, out_features, 2)
        stacked = torch.stack((y,z), dim=1)
        interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)
        y = self.unpermute(interleaved)
        return y

    def reverse(self, yz):
        bias, temp = self.bias, self.temperature
        """
        Reverse pass: x ≈ W† * gamma, where gamma_j = 0.5 * log((1 + tanh(w x)) / (1 - tanh(w x))) = Wx
        """
        yz = self.permute(yz)
        n2 = round(yz.shape[1]/2)
        uninterleaved = torch.unflatten(yz, 1, (2,n2))
        y = uninterleaved[:,0,:,:].squeeze(1)
        z = uninterleaved[:,1,:,:].squeeze(1)
        gamma = 0.5 * torch.log(z / y)
        if self.hasBias:
            gamma = gamma - torch.sum(bias*self.biasWeight + temp*self.biasNoise, dim=1).unsqueeze(1) # (batch, out_features)
        if not self.naive:
            W_pinv = self.layer.compute_Winverse().T
            x = gamma @ W_pinv
        else:
            W_pinv = torch.linalg.pinv( (self.W + temp*self.noise) )  # (in_features, out_features)
            x = gamma @ W_pinv
        x = self.unpermute(x)
        if temp != 0:
            if self.naive:
                self.resample_noise()
            else:
                self.biasNoise = sample_noise(self.biasWeight, shape=(1, 1, self.nOutput))
        return x

    @staticmethod
    def test():
        nBatch    = 16
        nInput    = 3
        nOutput   = 2 * nInput
        nFeatures = 5

        layer = ReversiblePiLayer(nInput=nInput, nOutput=nOutput, naive=True, hasBias=True, permuteInput=True)
        x = torch.randn(nBatch, nInput, nFeatures)
        layer.setTemperature(0.00000001)
        yz = layer.forward(x)
        print("Forward output shape:", yz.shape)  # Should be (batch, out_features, 2)
        x_recon = layer.reverse(yz)
        print("Reconstructed x shape:", x_recon.shape)  # Should be (batch, in_features)

        error = torch.norm(x - x_recon) / torch.norm(x)
        print(f"Reconstruction relative error: {error.item():.6f}")
        assert error < 0.1, f"Reconstruction error too high: {error}"
        print("ReversiblePiLayer test passed.")

class VQLayer(Layer):
    nOutput = 0

    def __init__(self, dim, codebookSize, numQuantizers):
        super(VQLayer, self).__init__(dim, dim)
        self.vq = ResidualVQ(
            dim=dim,
            codebook_size=codebookSize,
            num_quantizers=numQuantizers,
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.0,  # the weight on the commitment loss
            use_cosine_sim=True,
            rotation_trick=True  # Set False to use the STE gradient estimator or True to use the rotation trick.
        )


    def distance(self, x, y):
        # This is a Euclidean distance
        return torch.sqrt(torch.sum((x - y) ** 2))

    def forward(self, x, t=0):
        batch = len(x)
        # percepts are Batch x nOutput x nFeatures
        x = x.reshape((-1, self.nInput))
        quantized, indices, commit_loss, all_codes = self.vq(x, return_all_codes=True)
        distances = all_codes.permute(1, 2, 0)
        # need to produce
        # Find the closest prototype for each input
        # closest_prototype_indices = torch.argmin(distances, dim=1)
        # Get the labels of the closest prototypes
        # predicted_labels = self.labels[closest_prototype_indices]
        return all_codes

    def reverse(self, y, t=0):
        raise ValueError("Value not computed")
        return x
class DecisionBoundaryLayer(Layer):
    def __init__(self, nInput, nOutput, learning_rate=0.01):
        super(DecisionBoundaryLayer, self).__init__(nInput, nOutput)
        self.learning_rate = learning_rate
        self.weight        = nn.Parameter(torch.zeros(nInput, nOutput))
        self.register_buffer('noise', torch.randn(nInput, nOutput))

    def forward(self, x, t=0):
        if t != 0:
            self.noise = sample_noise(self.weight)

        W = self.weight + t*self.noise
        dot_product = torch.matmul(x, W)
        decision = torch.where(dot_product >= 0, torch.tensor(1.0, device=x.device),
                               torch.tensor(-1.0, device=x.device))
        return decision

    def update(self, x, t=0):
        d1 = torch.norm(x - self.weight) ** 2
        d2 = torch.norm(x + self.weight) ** 2
        if d1 < d2:
            self.weight.data += self.learning_rate * (x.unsqueeze(1) - self.weight.data)
        else:
            self.weight.data -= self.learning_rate * (x.unsqueeze(1) + self.weight.data)

    @staticmethod
    def test():
        n_points = 100
        data = torch.randn(n_points, 2)
        data[:, 0] *= 1.5

        layer = DecisionBoundaryLayer(nInput=2, nOutput=1, learning_rate=0.01)
        for _ in range(1000):
            idx = torch.randint(0, n_points, (1,))
            x = data[idx].squeeze()
            layer.update(x)

        w = layer.weight.detach().cpu().numpy()
        w_neg = -w

        data_np = data.numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(data_np[:, 0], data_np[:, 1], label="Data", alpha=0.6)

        plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='r', linestyle='-',label='w')
        plt.quiver(0, 0, w_neg[0], w_neg[1], angles='xy', scale_units='xy', scale=1, color='b', linestyle='-',label='-w')

        a, b = w
        x_vals = np.linspace(np.min(data_np[:, 0]) - 1, np.max(data_np[:, 0]) + 1, 100)
        if np.abs(b) > 1e-5:
            y_vals = - (a / b) * x_vals
            plt.plot(x_vals, y_vals, color='g', linestyle='dashed', label='Hyperplane')
        else:
            plt.axvline(0, color='g', linestyle='dashed', label='Hyperplane')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Decision Boundary Learning')
        plt.legend()
        plt.grid(True)
        plt.xlim(np.min(data_np[:, 0]) - 2, np.max(data_np[:, 0]) + 2)
        plt.ylim(np.min(data_np[:, 1]) - 2, np.max(data_np[:, 1]) + 2)
        warnings.filterwarnings('ignore', message='.*line style')
        plt.show(block=False)
class NormLayer(Layer):
    pNorm      = 2
    removeBias = True
    takeExp    = False
    dim        = 1
    lr         = 0.001

    def __init__(self, nInput, nOutput, pNorm=2, exp=False):
        """
        Hand-coded implementation of Layer Normalization or Softmax-like operation.
        """
        super().__init__(nInput,nOutput)
        self.pNorm  = pNorm
        self.exp    = exp
        self.dim    = 1
        # predict the bias and variance within this layer
        self.W      = nn.Parameter(torch.zeros(2))
        #self.noise  = nn.Parameter(torch.zeros(2)) If is ErgodicLayer

    def forward(self, x):
        #bias, temp = self.bias, self.temperature
        shape = x.shape
        N = shape[self.dim]
        mean = x.mean(dim=self.dim, keepdim=True)
        if self.pNorm == 2:
            assert(self.exp == False)
            norm     = torch.sum((x - mean) ** 2, dim=self.dim, keepdim=True) / (N - 1)
            norm     = torch.sqrt(norm + epsilon)
            moments  = torch.concat([mean, norm], axis=self.dim)
            #error    = self.W - moments
            #self.W   = nn.Parameter((1-self.lr) * self.W + (self.lr) * error)
            moments -= self.W
            x_norm   = (x - moments[:, 0:1]) / moments[:, 1:2]
        else:
            assert(self.pNorm == 1)
            if self.exp:
                exp_x   = torch.exp(x)
                sum_exp = exp_x.sum(dim=self.dim, keepdim=True)
                x_norm  = exp_x / (sum_exp + epsilon)
                moments = sum_exp
            else:
                sum_x  = torch.sum(x, dim=self.dim, keepdim=True)
                x_norm = x / (sum_x + epsilon)
                moments = sum_x
            assert(False, "False until we add the W prediction in this code branch")
        # append mean and norm
        output = torch.concat((x_norm, moments), dim=-1)
        return output

    def reverse(self, y):
        """
        Reverse operation (only defined for pNorm=2 without softmax).
        """
        if self.pNorm != 2:
            raise NotImplementedError("Reverse only supported for pNorm=2.")
        x    = y[:,  0:-2]
        mean = y[:, -2:-1] + self.W[0:1]
        norm = y[:, -1:]   + self.W[1:2]
        return x * norm + mean

    @staticmethod
    def test():
        torch.manual_seed(42)

        print("== Testing NormLayer ==")
        # === Test 1: pNorm=2 (Standard LayerNorm) ===
        x = torch.randn(10, 20)
        manual_layer_norm = NormLayer(20, 22, pNorm=2)
        manual_layer_norm.lr = 0
        normalized = manual_layer_norm(x)

        # Shape check
        #assert normalized.shape == x.shape, "Output shape mismatch"
        # Mean check
        # assert torch.allclose(normalized.mean(dim=manual_layer_norm.dim), torch.zeros_like(normalized.mean(dim=manual_layer_norm.dim)), atol=1e-5), "Mean not close to 0"
        # Variance check
        # assert torch.allclose(normalized.var(dim=manual_layer_norm.dim, unbiased=True), torch.ones_like(normalized.var(dim=manual_layer_norm.dim)), atol=1e-5), "Variance not close to 1"

        # === Reverse test ===
        reconstructed = manual_layer_norm.reverse(normalized)
        assert torch.allclose(x, reconstructed, atol=1e-5), "Reverse reconstruction failed"

        print("✓ pNorm=2 forward + reverse passed.")

        # === Test 2: pNorm=1, exp=True (Softmax mode) ===
        x = torch.randn(10, 5)
        builtin_softmax = nn.Softmax(dim=1)
        manual_softmax = NormLayer(5, 7, pNorm=1, exp=True)

        builtin_output = builtin_softmax(x)
        manual_output = manual_softmax(x)[:,0:-1]

        # Shape check
        assert builtin_output.shape == manual_output.shape, "Softmax output shape mismatch"

        # Softmax values check
        assert torch.allclose(builtin_output, manual_output, atol=1e-6), "Softmax outputs differ"

        # Sum-to-1 check
        assert torch.allclose(manual_output.sum(dim=manual_softmax.dim), torch.ones(x.shape[0]), atol=1e-6), "Softmax output does not sum to 1"

        # Range check
        assert torch.all((manual_output >= 0) & (manual_output <= 1)), "Softmax output contains values outside [0, 1]"

        print("✓ Softmax mode passed.")

        # === Test 3: 3D input ===
        #x_3d = torch.randn(5, 10, 20)
        #manual_layer_norm_3d = NormLayer(2, pNorm=2)
        #normalized_3d = manual_layer_norm_3d(x_3d)
        # 3D shape check
        #assert normalized_3d.shape == x_3d.shape, "3D input normalization failed"
        #print("✓ 3D input passed.")
        #print("All NormLayer tests passed successfully!")
class AttentionLayer(Layer):
    def __init__(self, nInput, nOutput, nHidden=None, symmetric=False):
        super(AttentionLayer, self).__init__(nInput, nOutput)
        if not nHidden:
            self.nHidden = nOutput
        else:
            self.nHidden = nHidden
        self.mask = None
        self.beta = 10
        # self.dropout = nn.Dropout(p=0.1)
        self.symmetric  = symmetric
        self.reversible = False
        #self.target_norm  = target_norm
        #self.tol          = tol
        #self.maxIter      = maxIter

        if self.symmetric:
            self.A = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        else:
            self.Q = LinearLayer(self.nInput, self.nHidden)
            self.K = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        self.Out = LinearLayer(self.nHidden, self.nOutput)
    def create_mask(self, sentences):
        self.mask = torch.zeros(self.nOutput, self.objectSize, dtype=torch.bool)
        for i, s in enumerate(sentences):
            self.mask[i, len(s):] = True

    def forward(self, x, bias=1, temp=0):
        if self.symmetric:
            a2     = self.A(x, bias, temp)
            value  = x if self.nHidden == self.nInput else self.V(x, bias, temp)
            scores = torch.matmul(a2.transpose(-2, -1), a2) / (self.nInput ** 0.5)
        else:
            query  = self.Q(x, bias, temp)
            key    = self.K(x, bias, temp)
            value  = x if self.nHidden == self.nInput else self.V(x, bias, temp)
            scores = torch.matmul(query.transpose(-2, -1), key) / (self.nInput ** 0.5)

        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))

        # Hopfield networks repeat until convergence here
        #for _ in range(self.maxIter):
        #    norm = torch.norm(attention, p=2, dim=-1).mean()
        #    if torch.abs(norm - self.target_norm) < self.tol:
        #        break
        #    attention = F.softmax(self.beta * scores, dim=-1)

        if not self.reversible:
            attn = F.softmax(self.beta * scores, dim=-1)
        else:
            attn = scores

        # attn = self.dropout(attn)
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output, bias, temp)
        return output

    @staticmethod
    def test():
        """
        Static test for sanity check.
        """
        #torch.manual_seed(42)
        nInput = 6
        nOutput = 3
        layer = AttentionLayer(nInput=nInput, nOutput=nOutput, nHidden=7)

        x = torch.randn(4, 5, nInput)  # batch of 4
        y = layer.forward(x, bias=1, temp=0)

        #print("Input minus output:")
        #print(x-x_rec)
        #assert torch.norm(x-x_rec) < 1, "Norm too high"
#endregion

#region Activation Functions
class Activation:
    """
    ACTIVATION A set of functions that apply fuzzy logic operations to NN activations.
    """

    # Constants
    strictParts = True
    activationThreshold = 0.01

    @staticmethod
    def positive(x):
        return (x > 0) * x

    @staticmethod
    def negative(x):
        return (-(x < 0)) * x

    @staticmethod
    def neutral(x):
        return 1 - np.abs(x)

    @staticmethod
    def true():
        return 1

    @staticmethod
    def false():
        return -1

    @staticmethod
    def unknown():
        return 0

    # Determine if the concept is active
    @staticmethod
    def isActive(x):
        return np.abs(x) >= Activation.activationThreshold

    # Determine equality
    @staticmethod
    def isEqual(x1, x2):
        return np.all(x1 == x2)

    @staticmethod
    def isPart(x1, x2):
        tf = Activation.sign(x1) * Activation.sign(x2) * (np.minimum(np.abs(x1), np.abs(x2)) / np.abs(x1))
        # Remove NaN values (from dividing by zero)
        tf = tf[~np.isnan(tf)]  # equivalent to tf(isnan(tf)) = []

        if len(tf) == 0:
            tf = 1  # nothing is a part of everything
        else:
            tf = np.mean(tf)
            if Activation.strictParts and np.abs(tf) < 1:
                tf = 0
        return tf

    @staticmethod
    def isWhole(x1, x2):
        tf = Activation.sign(x1) * Activation.sign(x2) * np.abs(x1) / np.maximum(np.abs(x1), np.abs(x2))
        # Handle zero values
        tf[(x1 == 0) & (x2 != 0)] = 0  # zero values in x1 are not wholes
        # Remove values where both x1 and x2 are zero
        tf = tf[~((x1 == 0) & (x2 == 0))]  # equivalent to tf(x1==0 & x2==0) = []

        if len(tf) == 0:
            tf = 1  # nothing is a part of everything
        else:
            tf = np.mean(tf)
            if Activation.strictParts and np.abs(tf) < 1:
                tf = 0
        return tf

    @staticmethod
    def isReducer(x1, x2):
        # x1 reduces x2 if abs(x2-x1) < abs(x2)
        return np.sum(np.abs(x2 - x1)) < np.sum(np.abs(x2))

    @staticmethod
    def part(x1, x2):
        return (Activation.sign(x1) + Activation.sign(x2)) / 2 * np.minimum(np.abs(x1), np.abs(x2))

    @staticmethod
    def whole(x1, x2):
        y = (Activation.sign(x1) + Activation.sign(x2)) / 2 * np.maximum(np.abs(x1), np.abs(x2))
        y[Activation.sign(x1) != Activation.sign(x2)] = np.nan
        return y

    @staticmethod
    def minMag(x1, x2):
        if np.abs(x1) <= np.abs(x2):
            return x1
        else:
            return x2

    @staticmethod
    def maxMag(x1, x2):
        if np.abs(x1) >= np.abs(x2):
            return x1
        else:
            return x2

    @staticmethod
    def error(x1, x2):
        return np.linalg.norm(x1 - x2)

    # signum, where sgn(0) = 1
    @staticmethod
    def sign(v1):
        y = np.sign(v1.astype(float) if hasattr(v1, 'astype') else float(v1))
        y[y == 0] = 1
        return y

    @staticmethod
    def norm(x):
        return np.linalg.norm(x)

    # saturate to [-1..1]
    @staticmethod
    def saturate(x):
        if np.isnan(x):
            return 0
        else:
            return min(1, max(-1, x))

    # threshold to one of [-1,0,1]
    @staticmethod
    def threshold(x):
        if np.abs(x) < Activation.activationThreshold:
            return 0
        else:
            return x

    @staticmethod
    def complement(x):
        return Activation.sign(x) - x

    @staticmethod
    def negation(x):
        return -x

    # symbolic activation should be in the range -1..1
    @staticmethod
    def convertSensation(x):
        return 2 * x - 1

    @staticmethod
    def test():
        def neg(x):
            # y = 1-x;
            return -x

        def scale(x):
            # y = (x+1)/2;
            return x

        def combine(x1, x2):
            sgn = (np.sign(x1) + np.sign(x2)) / 2
            amp = np.multiply(np.abs(x1), np.abs(x2))
            # y = sgn.*amp;
            # y = x1.*x2;
            y = np.mean([x1, x2], axis=0)
            y = np.sign(x1) * np.sign(x2) * (np.minimum(np.abs(x1), np.abs(x2)) / np.abs(x1))
            return y

        x = np.arange(0, 2 * np.pi, 0.01)
        y1 = scale(np.cos(x - np.pi / 6))
        y2 = scale(np.sin(x))

        plt.clf()
        plt.bar(x, y1, color='red', alpha=0.5)
        #plt.hold(True)  # Note: plt.hold is deprecated in newer matplotlib versions
        plt.bar(x, y2, color='blue', alpha=0.5)
        plt.bar(x, combine(y1, y2), color='black', alpha=0.5)
        plt.show(block=False)
#endregion

#region Error Functions
class CertaintyWeightedMAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        abs_error = torch.abs(targets - predictions)
        certainty = torch.abs(predictions)
        loss = abs_error * (self.alpha * certainty + (1 - self.alpha))
        return torch.mean(loss)
class CertaintyWeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        """
        outputs: [batch_size, num_outputs]
        targets: [batch_size, num_outputs] (for regression or one-hot for classification)
        """
        # Certainty: magnitude of the prediction for each sample
        certainty = outputs.abs().sum(dim=1)  # [batch_size], or you could use .mean(dim=1)
        mse_loss = ((outputs - targets) ** 2).sum(dim=1)  # [batch_size]
        cw_mse_loss = mse_loss * certainty                 # [batch_size]
        hybrid_loss = self.alpha * cw_mse_loss.mean() + (1 - self.alpha) * mse_loss.mean()
        #hybrid_loss = self.alpha * cw_mse_loss + (1 - self.alpha) * mse_loss
        return hybrid_loss
class CertaintyWeightedCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # If targets are one-hot, convert to indices
        if targets.dim() == 2 and targets.size(1) == logits.size(1):
            targets = targets.argmax(dim=1)
        # Ensure targets are int64 and on the same device as logits
        targets = targets.to(dtype=torch.long, device=logits.device)
        batch_size = logits.size(0)
        batch_indices = torch.arange(batch_size, device=logits.device)

        log_probs = F.log_softmax(logits, dim=1)  # [batch, num_classes]
        ce_loss = -log_probs[batch_indices, targets]
        #norm_logits = torch.norm(logits, dim=1)
        #cwce_loss = -norm_logits * log_probs[batch_indices, targets]
        p_true = torch.exp(log_probs[batch_indices, targets])
        cwce_loss = -p_true * log_probs[batch_indices, targets]  # possibly erroneous
        hybrid_loss = self.alpha * cwce_loss + (1 - self.alpha) * ce_loss
        #norm = torch.norm(hybrid_loss, p=2)
        return hybrid_loss.mean()
#endregion

#region Various kinds of memory
# Base class for various kinds of (temporally-varying) Memory
class Mem:
    def __init__(self, sz=None):
        self.lr = 0.01
        self.nTrials = 0
        self.output = None
        self.reset(sz)

    def delta(self, *args):
        """Increment nTrials by one. Extra arguments are ignored."""
        self.nTrials += 1
    def get(self):
        """Return the current output array."""
        return self.output
    def set(self, in_val):
        """Set the output array to a new value."""
        self.output = in_val
    def reset(self, sz=None):
        """
        Reset the output to a zeros array and nTrials to 0.
        If sz is not provided, use the current output's shape (or (0, 0) if undefined).
        """
        if sz is None:
            if self.output is None:
                sz = (0, 0)
            else:
                sz = self.output.shape
        self.output = torch.zeros(sz)
        self.nTrials = 0
    def removeRC(self, r=None, c=None):
        """
        Remove a row and/or column from the output.
        The row (r) and column (c) indices are assumed to be given in MATLAB 1-indexed form.
        """
        if r is not None:
            self.output = np.delete(self.output, r - 1, axis=0)
        if c is not None:
            self.output = np.delete(self.output, c - 1, axis=1)
    def insertRC(self, r=None, c=None):
        """
        Insert a row and/or column of zeros into the output.
        The indices r and c are assumed to be 1-indexed.
        """
        if r is not None:
            new_row = np.zeros((1, self.output.shape[1]))
            self.output = np.insert(self.output, r - 1, new_row, axis=0)
        if c is not None:
            new_col = np.zeros((self.output.shape[0], 1))
            self.output = np.insert(self.output, c - 1, new_col, axis=1)
    def setRC(self, r, c, val):
        """
        Set the (r, c) element to val and its symmetric element (c, r) to val.
        Indices are assumed to be 1-indexed.
        """
        self.output[r - 1, c - 1] = val
        self.output[c - 1, r - 1] = val

    @staticmethod
    def test():
        """
        Test method that creates instances of several Mem-derived classes
        and calls their testImpulse method.
        """
        names = ['ExponentialMem', 'GammaMem', 'MeanMem']
        for name in names:
            # Create an instance using globals() (similar to MATLAB's feval)
            m = globals()[name]()
            Mem.testImpulse(m, name)

    @staticmethod
    def testOne(m):
        """
        Test one impulse by incrementing and plotting the output.
        """
        plt.figure(1)
        plt.clf()
        x = np.arange(1, 1001)
        y = np.zeros(1000)
        m.delta(1, 1)  # extra parameters are ignored
        for i in range(len(x)):
            y[i] = m.get()
            m.delta(1, 1)
        plt.plot(x, y)
        plt.title("Test One")
        plt.show(block=False)

    @staticmethod
    def testImpulse(m, name):
        """
        Test impulse response by incrementing and plotting the output.
        """
        plt.figure(1)
        x = np.arange(1, 1001)
        y = np.zeros(1000)
        m.delta(1.0)
        for i in range(len(x)):
            y[i] = m.get()
            m.delta(0)
        plt.plot(x, y)
        plt.title(f"Impulse Response: {name}")
        plt.show(block=False)
# ZOHMem subclass: Zero-Order Hold memory.
class ZOHMem(Mem):
    def __init__(self, sz=1):
        super().__init__(sz)

    def delta(self, in1, in2=None):
        # Call the base class delta and then set output to in1.
        super().delta()
        self.output = in1
# StateMem subclass: adds a 'state' property.
class StateMem(Mem):
    def __init__(self, sz=1):
        super().__init__(sz)
        self.state = None
        self.reset(sz)

    def reset(self, sz=None):
        """
        Reset output and state to zeros and nTrials to 0.
        """
        if sz is None:
            if self.output is None:
                sz = (0, 0)
            else:
                sz = self.output.shape
        super().reset(sz)
        self.state = torch.zeros(sz)

    def delta(self, *args):
        # Just call the base class delta.
        super().delta(*args)

    def removeRC(self, r=None, c=None):
        """
        Remove a row/column from both output and state.
        """
        super().removeRC(r, c)
        if r is not None:
            self.state = np.delete(self.state, r - 1, axis=0)
        if c is not None:
            self.state = np.delete(self.state, c - 1, axis=1)

    def insertRC(self, r=None, c=None):
        """
        Insert a row/column of zeros into both output and state.
        """
        super().insertRC(r, c)
        if r is not None:
            new_row = np.zeros((1, self.state.shape[1]))
            self.state = np.insert(self.state, r - 1, new_row, axis=0)
        if c is not None:
            new_col = np.zeros((self.state.shape[0], 1))
            self.state = np.insert(self.state, c - 1, new_col, axis=1)

    def setRC(self, r, c, val):
        """
        Set the (r,c) element in both output and state (and symmetrically).
        """
        super().setRC(r, c, val)
        self.state[r - 1, c - 1] = val
        self.state[c - 1, r - 1] = val
# RLSMem subclass: Recursive Least Squares memory.
class RLSMem(StateMem):
    def __init__(self, sz=1):
        super().__init__(sz)
        self.momLR = 0.2

    def delta(self, in1, in2=None):
        # Call the base class (Mem) delta.
        Mem.delta(self)
        # Compute error using the L2 norm.
        err = np.linalg.norm(self.output - in1, 2)
        self.output = self.output + err * in1
        # Optionally update state (commented out in original code):
        # self.state = self.state + self.momLR * err * in1
        # self.output = self.output + self.state
# ProbMem subclass: probabilistic memory update.
class ProbMem(Mem):
    def __init__(self, sz=1):
        super().__init__(sz)

    def delta(self, in1, in2):
        super().delta()
        # Iterate over the indices of in1 and in2.
        for r in range(len(in1)):
            for c in range(len(in2)):
                # Increase or decrease conditional probability based on Activation.
                if Activation.true(in2[c]) and Activation.true(in1[r]):
                    self.output[r, c] = ((self.nTrials - 1) / self.nTrials) * self.output[r, c] + (1 / self.nTrials) * 1
                elif Activation.true(in2[c]) and Activation.false(in1[r]):
                    self.output[r, c] = ((self.nTrials - 1) / self.nTrials) * self.output[r, c] + (1 / self.nTrials) * -1
# MeanMem subclass: computes a running mean.
class MeanMem(Mem):
    def __init__(self, sz=1):
        super().__init__(sz)

    def delta(self, in1, in2=None):
        super().delta()
        self.output = ((self.nTrials - 1) / self.nTrials) * self.output + (1 / self.nTrials) * in1
# GammaMem subclass: blends state with output using a second learning rate.
class GammaMem(StateMem):
    def __init__(self, sz=1, lr2=0.05):
        super().__init__(sz)
        self.lr2 = lr2

    def delta(self, in1, in2=None):
        # Call the StateMem delta method.
        StateMem.delta(self)
        self.state = (1 - self.lr) * self.state + self.lr * in1
        self.output = (1 - self.lr2) * self.output + self.lr2 * self.state

    @staticmethod
    def test():
        Mem.testOne(GammaMem())
# ExponentialMem subclass: exponential memory update.
class ExponentialMem(Mem):
    def __init__(self, sz=1, lr=None):
        super().__init__(sz)
        if lr is not None:
            self.lr = lr

    def delta(self, in1, in2=None):
        super().delta()
        self.output = (1 - self.lr) * self.output + self.lr * in1
    @staticmethod
    def test():
        # Create an instance of ExponentialMem and run a test.
        m_exp = ExponentialMem(sz=(5, 5), lr=0.1)
        # Simulate a delta update with an input (for example, a 5x5 array).
        x = np.ones((5, 5))
        m_exp.delta(x)
        x = x = np.zeros((5, 5))
        m_exp.delta(x)
        print("ExponentialMem output after two delta calls:")
        print(m_exp.get())
# CorrMem subclass: correlational memory update.
class CorrMem(Mem):
    def __init__(self, sz=1):
        super().__init__(sz)

    def delta(self, in1, in2):
        super().delta()
        for r in range(len(in1)):
            for c in range(len(in2)):
                val = in1[r] * in2[c]
                # Avoid division by zero.
                denom = np.sqrt(in1[r]**2 * in2[c]**2)
                if denom != 0:
                    val = Activation.saturate(val / denom)
                else:
                    val = 0
                amt = max(abs(in1[r]), abs(in2[c]))
                self.output[r, c] = ((self.nTrials - amt) / self.nTrials) * self.output[r, c] + (amt / self.nTrials) * val
#endregion

# Example usage:
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    LinearLayer.test()
    ReversibleRotationLayer.test()
    ReversibleDiagonalLayer.test()
    ReversibleLinearLayer.test()

    ReversiblePiLayer.test()
    SigmaLayer.test()
    ReversibleSigmaLayer.test()
    PiLayer.test()
    PiLayer.xorTest()
    ReversiblePiLayer.test()

    AttentionLayer.test()
    NormLayer.test()
    Mem.test()
    DecisionBoundaryLayer.test()
