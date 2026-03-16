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
from torchlogix.layers import FixedBinarization, GroupSum, LogicConv2d, LogicDense, OrPooling2d


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
        """Freeze all params (learn=False) or unfreeze them (learn=True)."""
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
        """Hook called after each optimizer step for custom parameter updates."""
        pass

    def forward(self, x, bias=None, temp=None):
        """Identity pass-through (subclasses override)."""
        batch = x.shape[0]
        assert x.shape[1] == self.nSymbols
        return x
    def reverse(self, y, bias=None, temp=None):
        """Identity pass-through (subclasses override)."""
        batch = y.shape[0]
        assert y.shape[1] == self.nOutput
        return y
class ErgodicLayer(Layer):
    """Layer base class that adapts its explore/exploit balance over training.

    The explore/exploit tradeoff is governed by a single scalar ``alpha``:
        - alpha = 1  (initial) => full exploration: bias=0, var=1
        - alpha = 0  (converged) => full exploitation: bias=1, var=0

    Subclasses mix learned weights (scaled by ``bias``) with random noise
    (scaled by ``var``) in their forward passes:
        effective_weight = bias * W + var * noise

    A per-output-column **certainty** tracker observes both forward
    activations and gradient energy to estimate how reliable each output
    dimension has become.  This certainty signal is available for
    downstream scheduling but does not yet feed back into alpha
    automatically (the commented-out ``alpha_update`` sketches that path).
    """
    def __init__(self, nInput, nOutput, permuteInput=False):
        super().__init__(nInput, nOutput, permuteInput)
        # --- Explore/exploit knobs (set externally via setAlpha) ----------
        # alpha in [0,1]: bias = 1-alpha, var = alpha.
        self.alpha       = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.bias        = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # var has requires_grad=True so gradient energy can be sensed
        self.var         = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # --- Buffers for the (disabled) per-layer Adam alpha update -------
        self.register_buffer('v', torch.tensor(0.0))             # EMA of squared gradient on var
        self.register_buffer('t_step', torch.tensor(0))          # step counter for bias correction

        # --- Per-output certainty tracking --------------------------------
        # certainty[j] in [0,1]: how reliable output column j is.
        self.register_buffer('certainty', torch.ones(nOutput))
        self.register_buffer('certainty_v', torch.zeros(nOutput))            # gradient energy EMA
        self.register_buffer('certainty_forward_ema', torch.zeros(nOutput))  # forward activation EMA
        self.register_buffer('certainty_step', torch.tensor(0))
        self.register_buffer('certainty_forward_step', torch.tensor(0))
        self.beta = 0.999                       # EMA decay for (disabled) alpha_update
        self.certainty_beta = 0.99              # EMA decay for gradient certainty
        self.certainty_forward_beta = 0.9       # EMA decay for forward certainty
        self.certainty_scale = 1.0              # scaling inside gradient certainty formula
        self.certainty_forward_weight = 0.5     # blending weight: forward vs gradient certainty
        self.certainty_gradient_weight = 0.5
        self.dropoutRate = 0.0

    def getParameters(self):
        """Return learnable params, excluding ``var`` (which is only sensed, not optimized)."""
        params = [p for n, p in self.named_parameters() if n != "var"]
        return params
    def setAlpha(self, alpha):
        """Set the explore/exploit balance.  alpha 1->0 = explore->exploit."""
        with torch.no_grad():
            self.alpha.fill_(alpha)
            self.bias.fill_(1.0 - alpha)
            self.var.fill_(alpha)
    # def local_tradeoff(self):
    #     # High-certainty outputs lean toward bias; uncertain outputs keep more
    #     # variance so the layer continues to explore alternatives.
    #     certainty = self.certainty.to(device=self.bias.device, dtype=self.bias.dtype)
    #     local_bias = self.bias * certainty
    #     local_var = self.var * torch.ones_like(certainty) + self.bias * (1.0 - certainty)
    #     return local_bias, local_var.clamp(0.0, 1.0)
    @torch.no_grad()
    def reduce_certainty_signal(self, signal):
        """Collapse a (possibly batched) signal to a per-output-column scalar in [0,1]."""
        if signal is None:
            return None
        signal = signal.detach()
        if signal.ndim == 0 or signal.shape[-1] != self.nOutput:
            return None
        signal = signal.abs()
        if signal.ndim > 1:
            # Average over all dimensions except the last (output) axis
            signal = signal.mean(dim=tuple(range(signal.ndim - 1)))
        return torch.tanh(signal).clamp(0.0, 1.0)
    @torch.no_grad()
    def observe_forward_certainty(self, signal):
        """Feed a forward-pass activation into the certainty EMA.  Called at end of forward()."""
        signal = self.reduce_certainty_signal(signal)
        if signal is None:
            return
        self.certainty_forward_ema.mul_(self.certainty_forward_beta).add_((1 - self.certainty_forward_beta) * signal)
        self.certainty_forward_step += 1
    @torch.no_grad()
    def forward_certainty(self):
        """Return bias-corrected forward certainty EMA, or None if no observations yet."""
        if self.certainty_forward_step.item() == 0:
            return None
        forward = self.certainty_forward_ema / (1 - self.certainty_forward_beta ** self.certainty_forward_step.item())
        return forward.clamp(0.0, 1.0)
    @torch.no_grad()
    def certainty_gradient_energy(self):
        """Sum squared-gradient energy across learnable params that share the output axis.

        Skips alpha/bias/var and noise params since those are not true learned weights.
        """
        grad_energy = None
        for name, param in self.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            if name in {"alpha", "bias", "var"}:
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
        """Compute per-output certainty from gradient energy: low energy => high certainty.

        Uses the Adam-style formula: certainty = 1 / (1 + scale * sqrt(v_hat)).
        """
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
        """Weighted average of forward and gradient certainty signals."""
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
    # alpha_update: locally tuning alpha per-layer from gradient energy
    # could substitute for global calls to setAlpha().
    # @torch.no_grad()
    # def alpha_update(self):
    #     if self.var.grad is None:
    #         return
    #     grad = self.var.grad.item()
    #     # Modified Adam: EMA of gradient energy (second moment only)
    #     self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
    #     self.t_step += 1
    #     # Bias correction (same as Adam)
    #     v_hat = self.v / (1 - self.beta ** self.t_step)
    #     # Alpha = sensor output: high gradient energy → low alpha (explore)
    #     self.alpha.fill_(1.0 / (1.0 + self.global_temp * v_hat.sqrt()))
    #     # Derive bias and var from alpha
    #     alpha = self.alpha.item()
    #     if random.random() < self.dropoutRate:
    #         # Occasional bias dropout forces another exploration step even when
    #         # the layer has converged toward exploitation.
    #         self.bias.fill_(0.0)
    #     else:
    #         self.bias.fill_(alpha)
    #     self.var.fill_(1.0 - alpha)
    #     self.var.grad.zero_()
    def paramUpdate(self):
        self.certainty_update()
        # self.alpha_update()

class LinearLayer(Layer):
    """Standard linear (affine) layer with ergodic noise injection.

    Forward: y = x @ (bias*W + temp*noise) [+ bias*b + temp*biasNoise]

    When bias=1, temp=0 (exploitation) this is an ordinary linear layer.
    Non-zero ``temp`` blends in freshly sampled Gaussian noise, enabling
    exploration.  Defaults to identity initialization.
    """
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
        """Draw fresh Gaussian noise matching W (and bias) shape/device."""
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
        input = torch.rand((1, nInput))
        output = layer(input)

        print(f"Input: {input}")
        print(f"After forward linear: {output}")
class InvertibleRotationLayer(Layer):
    """Learnable orthogonal rotation built from a chain of Givens rotations.

    Each of the (dim-1) angles rotates a consecutive pair of axes.
    The ``naive`` flag controls whether the full rotation matrix is
    materialized (True) or applied sequentially in-place (False, faster
    for large dim).  Because orthogonal matrices are their own
    pseudoinverse, ``reverse()`` just applies the transpose.
    """
    def __init__(self, dim, naive=False, theta=None):
        super(InvertibleRotationLayer, self).__init__(dim, dim)
        self.dim = dim
        self.naive = naive
        if theta is None:
            theta = torch.randn(dim - 1) * 2 * torch.pi
        self.theta = nn.Parameter(theta)
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
        nlayer = InvertibleRotationLayer(dim=dim, naive=True, theta=theta)
        layer  = InvertibleRotationLayer(dim=dim, naive=False, theta=theta)
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
class InvertibleDiagonalLayer(Layer):
    """Learnable diagonal (singular-value) matrix for use in a reversible SVD decomposition.

    Stores ``rank = min(nInput, nOutput)`` positive scalars (lamda) and
    builds a (nOutput x nInput) diagonal matrix S.  Forward: y = x @ S^T.
    Reverse inverts the diagonal on the shared rank dimensions and
    zero-pads any extra input dimensions.
    """
    def __init__(self, nInput, nOutput):
        super(InvertibleDiagonalLayer, self).__init__(nInput, nOutput)
        self.nInput = nInput
        self.nOutput = nOutput
        self.rank = min(nInput, nOutput)
        self.lamda = nn.Parameter(torch.ones(self.rank))
        self.register_buffer('noise', torch.zeros(self.rank))

    def stabilize(self):
        """Clamp singular values to at most 1.0 to prevent unbounded growth."""
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
        Runs several tests on the InvertibleDiagonalLayer:
          1. Square case: nInput == nOutput.
          2. Wide-output case: nInput < nOutput.
          3. Tall-input case: nInput > nOutput.
          4. Batch test.
        For the tall-input case and batch test (when nInput > nOutput), the tests force the extra input coordinates
        to zero so that the mapping is invertible.
        """
        print("Testing InvertibleDiagonalLayer...")

        # Test 1: Square case: nInput == nOutput.
        nInput, nOutput = 4, 4
        layer = InvertibleDiagonalLayer(nInput, nOutput)
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
        layer = InvertibleDiagonalLayer(nInput, nOutput)
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
        layer = InvertibleDiagonalLayer(nInput, nOutput)
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
        layer = InvertibleDiagonalLayer(nInput, nOutput)
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
class InvertibleLinearLayer(Layer):
    """Exactly-invertible linear layer factored as W = U * Sigma * V^T (thin SVD).

    U and V are orthogonal (InvertibleRotationLayer), Sigma is diagonal.
    Two modes:
      - naive=True:  materializes W and W^{-1} as dense matrices.
      - naive=False: applies U, Sigma, V sequentially (lower memory, exact inverse).

    When ``stable=True``, singular values are clamped to <=1 each step.
    """
    def __init__(self, nInput, nOutput, naive=False, hasBias=True, stable=False):
        super(InvertibleLinearLayer, self).__init__(nInput, nOutput)
        self.naive = naive
        self.hasBias = hasBias
        self.rank = min(nInput, nOutput)
        self.stable = stable

        self.U = InvertibleRotationLayer(dim=nInput, naive=self.naive)      # nInput x nInput
        self.V = InvertibleRotationLayer(dim=nOutput, naive=self.naive)     # nOutput x nOutput
        self.Sigma = InvertibleDiagonalLayer(nInput, nOutput)               # diagonal scaling

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

    def _rotation_matrices(self, bias=1.0, temp=0.0):
        """Compute and cache U, V rotation matrices for reuse within a forward+reverse pair."""
        U_matrix = self.U.rotation_matrix(bias, temp)
        V_matrix = self.V.rotation_matrix(bias, temp)
        self._cached_U = U_matrix
        self._cached_V = V_matrix
        return U_matrix, V_matrix

    def compute_W(self, bias=1.0, temp=0.0):
        """Materialize the full weight matrix W = U * Sigma * V^T.

        When bias/temp are provided, each SVD component incorporates its own
        noise: W_eff = U(bias,temp) @ Sigma(bias,temp) @ V(bias,temp)^T.
        Caches rotation matrices for reuse by compute_Winverse.
        """
        U_matrix, V_matrix = self._rotation_matrices(bias, temp)
        # Build Σ of shape (nInput, nOutput) with bias/temp applied
        w = bias * self.Sigma.lamda + temp * self.Sigma.noise
        Sigma_matrix = torch.zeros(self.nInput, self.nOutput, device=w.device, dtype=w.dtype)
        for i in range(self.rank):
            Sigma_matrix[i, i] = w[i]
        W = U_matrix @ Sigma_matrix @ V_matrix.T
        return W

    def compute_Winverse(self, bias=1.0, temp=0.0, noise=None):
        """Materialize the pseudoinverse W^+ = V * Sigma^{-1} * U^T.

        Reuses cached rotation matrices from compute_W when available.
        When bias/temp are provided (and noise is None), each SVD component
        incorporates its own internal noise — matching compute_W(bias, temp).

        When external noise is provided, rotates it into the SVD basis and
        inverts the augmented matrix exactly (for the naive/external-noise path).
        """
        # Reuse cached rotation matrices if available, otherwise recompute
        U_matrix = getattr(self, '_cached_U', None)
        V_matrix = getattr(self, '_cached_V', None)
        if U_matrix is None or V_matrix is None:
            U_matrix, V_matrix = self._rotation_matrices(bias, temp)
        if noise is not None and temp != 0:
            # External noise path: W_eff = W + temp*noise
            Sigma_matrix = torch.zeros(self.nInput, self.nOutput,
                                       device=self.Sigma.lamda.device,
                                       dtype=self.Sigma.lamda.dtype)
            for i in range(self.rank):
                Sigma_matrix[i, i] = self.Sigma.lamda[i]
            D = U_matrix.T @ noise @ V_matrix
            M = Sigma_matrix + temp * D
            Um, Sm, Vmh = torch.linalg.svd(M, full_matrices=False)
            Sm_inv = torch.where(Sm > 1e-7 * Sm.max(), 1.0 / Sm, torch.zeros_like(Sm))
            M_inv = (Vmh.mH * Sm_inv.unsqueeze(-2)) @ Um.mH
            W_inv = V_matrix @ M_inv @ U_matrix.T
        else:
            # Internal noise path: each component already includes bias/temp
            w = bias * self.Sigma.lamda + temp * self.Sigma.noise
            Sigma_inv = torch.zeros(self.nOutput, self.nInput,
                                    device=w.device, dtype=w.dtype)
            for i in range(self.rank):
                Sigma_inv[i, i] = 1.0 / w[i]
            W_inv = V_matrix @ Sigma_inv @ U_matrix.T
        # Clear cache after use
        self._cached_U = None
        self._cached_V = None
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
        layer = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, naive=True)
        gLayer = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, naive=False)

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

        # Test compute_W / compute_Winverse roundtrip
        layer2 = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, naive=False, hasBias=False)
        W = layer2.compute_W()
        W_inv = layer2.compute_Winverse()
        # W @ W_inv should be identity (nInput x nInput) for wide matrix
        identity_check = W @ W_inv  # (nInput, nInput)
        eye = torch.eye(nInput)
        identity_err = torch.norm(identity_check - eye)
        assert identity_err < 1e-4, f"compute_W/Winverse identity error: {identity_err}"
        print(f"compute_W/Winverse identity error: {identity_err:.2e}")

        # Test compute_W(bias, temp) with internal noise
        W_noisy = layer2.compute_W(bias=0.9, temp=0.1)
        W_inv_noisy = layer2.compute_Winverse(bias=0.9, temp=0.1)
        identity_noisy = W_noisy @ W_inv_noisy
        noisy_err = torch.norm(identity_noisy - eye)
        assert noisy_err < 1e-4, f"Noisy compute_W/Winverse identity error: {noisy_err}"
        print(f"Noisy compute_W/Winverse identity error: {noisy_err:.2e}")

        # Test rotation matrix caching: compute_W caches, compute_Winverse reuses
        layer2._cached_U = None
        layer2._cached_V = None
        W = layer2.compute_W(bias=0.8, temp=0.2)
        assert layer2._cached_U is not None, "compute_W should cache U"
        assert layer2._cached_V is not None, "compute_W should cache V"
        W_inv = layer2.compute_Winverse(bias=0.8, temp=0.2)
        assert layer2._cached_U is None, "compute_Winverse should clear cache"
        identity_cached = W @ W_inv
        cached_err = torch.norm(identity_cached - eye)
        assert cached_err < 1e-4, f"Cached roundtrip error: {cached_err}"
        print(f"Cached roundtrip error: {cached_err:.2e}")

        print("All tests passed!")

class LiftingLayer(InvertibleLinearLayer):
    """Bias-free, stable reversible linear layer for mapping between row/column spaces."""
    def __init__(self, nInput, nOutput, init='orthogonal'):
        super(LiftingLayer, self).__init__(nInput, nOutput, naive=False, hasBias=False, stable=True)

class ColumnUsageTracker:
    """Monitors gradient norms per weight column and freezes low-activity columns.

    Attaches a backward hook to a standard nn.Linear layer.  After
    ``window`` steps, columns whose average gradient norm falls below
    ``freezeThreshold`` are permanently frozen (gradients zeroed out).
    """
    def __init__(self, linearLayer, freezeThreshold=0.01, window=10):
        self.linear = linearLayer
        self.freezeThreshold = freezeThreshold
        self.window = window
        self.grad_history = []
        self.frozen_columns = torch.zeros(linearLayer.weight.shape[1], dtype=torch.bool)
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

class SigmaLayer(ErgodicLayer):
    """Additive (summation) layer: y = tanh(W @ x + b).

    When ``ergodic=True``, the layer participates in the explore/exploit
    schedule: during training, bias and temp are derived from setAlpha().
    When ``ergodic=False`` (default), the layer always uses deterministic
    weights (bias=1, temp=0), ignoring setAlpha entirely.
    At eval time, both modes use deterministic weights.
    """
    def __init__(self, nInput, nOutput, permuteInput=False, ergodic=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.layer       = LinearLayer(nInput, nOutput, hasBias=True)
        self.saturate    = True
        self.activation  = torch.zeros(1,nOutput,1)
        self.ergodic     = ergodic

    def forward(self, x):
        # ergodic=False or eval mode => deterministic pass (no noise)
        if not self.ergodic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0
        else:
            bias, temp = self.bias, self.var
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
        layer.setAlpha(0.001)

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
class InvertibleSigmaLayer(SigmaLayer):
    """SigmaLayer whose linear transform is exactly invertible (SVD-factored).

    Inherits the ergodic flag behavior from SigmaLayer.
    reverse() inverts tanh via atanh, then inverts the linear layer.
    """
    def __init__(self, nInput, nOutput, naive=False, permuteInput=False, ergodic=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput, ergodic=ergodic)
        self.layer          = InvertibleLinearLayer(nInput, nOutput, naive=naive, hasBias=True)
    def layer_tradeoff(self):
        return self.bias, self.var
    def reverse(self, y):
        if not self.ergodic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0
        else:
            bias, temp = self.bias, self.var
        y  = self.permute(y)
        y = y.squeeze(0)
        if self.saturate:
            self.activation = torch.atanh(y) # this can be faster if we keep the tanh activation
            y = self.activation.clone()
        x  = self.layer.reverse(y, bias, temp)  # (batch_size, output_dim)
        x = self.unpermute(x)
        return x

    @staticmethod
    def test():
        nInput, nOutput = 5, 7
        permute = False
        #naive = False
        layer   = InvertibleSigmaLayer(nInput=nInput, nOutput=nOutput, permuteInput=permute, naive=False)

        x = torch.randn((2, 5, nInput))
        layer.setAlpha(0.000000001)
        y = layer.forward(x)
        y_inv = layer.reverse(y)

        #print(f"Original input: {x}")
        #print(f"After reversible linear: {y}")
        #print(f"Inverse operation result: {y_inv}")
        assert(torch.norm(x-y_inv) < 0.00001)

        layer = InvertibleSigmaLayer(nInput=nInput, nOutput=nOutput, permuteInput=False, naive=True)
        x = torch.randn((4, 8, nInput))
        layer.setAlpha(0.00000001)
        y = layer.forward(x)
        assert y.shape == (4,8,nOutput), "Incorrect Size"
        y_inv = layer.reverse(y)
        assert(torch.norm(x-y_inv) < 0.00001)
class PiLayer(ErgodicLayer):
    """Multiplicative (product) layer: y_j = prod_i (1 + tanh(w_ji * x_i)).

    Whereas SigmaLayer sums weighted inputs, PiLayer takes their product,
    giving it a fundamentally different inductive bias (conjunction-like).
    The ``ergodic`` flag works identically to SigmaLayer: when False,
    the layer always uses deterministic weights regardless of setAlpha().
    """
    def __init__(self, nInput, nOutput, permuteInput=False, ergodic=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.ergodic = ergodic
        self.weights          = nn.Parameter(torch.zeros(nInput, nOutput))
        self.register_buffer('noise', torch.randn(nInput, nOutput))
        self.biasWeight       = nn.Parameter(torch.zeros(1, 1, self.nOutput))
        self.register_buffer('biasWeightNoise', torch.randn(1, self.nInput, self.nOutput))

        self.saturate      = True
        self.hasBiasWeight = True
        self.useEpsilon    = True   # add epsilon inside product terms to avoid exact zeros

    def resample_noise(self):
        self.noise = sample_noise(self.weights)
        self.biasWeightNoise = sample_noise(self.weights, shape=(1, self.nInput, self.nOutput))

    def forward(self, x):
        # ergodic=False or eval mode => deterministic pass (no noise)
        if not self.ergodic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0
        else:
            bias, temp = self.bias, self.var
        x = self.permute(x)
        # Implements y_j = prod_i (1 + tanh(w_ji * x_i + b_j)).
        # A log-domain formulation was attempted (exp(sum(log(...)))), but
        # the matrix log is not invertible with negative weights.
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
        layer.setAlpha(0.001)
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
        sigma.setAlpha(0.0001)
        pi.setAlpha(0.0001)
        for epoch in range(epochs):
            optimizer.zero_grad()  # Clear gradients

            x1 = pi(X)  # Pass through PiLayer
            y = sigma(x1)  # Pass through SigmaLayer

            loss = criterion(y, Y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, MSE: {loss.item():.6f}')
class InvertiblePiLayer(ErgodicLayer):
    """Invertible multiplicative layer that outputs paired log-products.

    Forward produces interleaved (log_y, log_z) pairs where:
        log_y_j = sum_i log(1 - tanh(w_ji * x_i))
        log_z_j = sum_i log(1 + tanh(w_ji * x_i))

    Computing in log-space (sum of logs instead of product) avoids numerical
    underflow when nInput is large.

    Reverse recovers x via gamma_j = 0.5 * (log_z_j - log_y_j) = (W @ x)_j,
    then x = W^+ @ gamma.  Because the output has both log_y and log_z
    channels, nOutput must equal 2*nInput when naive=False.

    The ``ergodic`` flag works identically to SigmaLayer/PiLayer.
    """
    def __init__(self, nInput, nOutput, naive=False, permuteInput=False, hasBias = True, ergodic=False):
        super().__init__(nInput, nOutput, permuteInput=permuteInput)
        self.ergodic = ergodic
        self.naive   = naive
        self.hasBias = hasBias
        self.useEpsilon = True
        if not self.naive:
            assert 2*nInput == nOutput, "Non-naive mode requires nOutput == 2*nInput for invertibility."
        if naive:
            self.W      = nn.Parameter(torch.randn(nInput, nOutput))
            self.register_buffer('noise', torch.randn(nInput, nOutput))
        else:
            # Non-naive: SVD-factored weights with internal noise at each
            # component (U, Sigma, V).  No external noise matrix needed.
            # hasBias=False because InvertiblePiLayer handles its own bias.
            self.layer  = InvertibleLinearLayer(nInput, nOutput, naive=False, hasBias=False)
        self.biasWeight       = nn.Parameter(torch.zeros(1, 1, self.nOutput))
        self.register_buffer('biasNoise', torch.randn(1, 1, self.nOutput))

    def resample_noise(self):
        if self.naive:
            self.noise = sample_noise(self.W)
        else:
            # Non-naive: resample noise at each SVD component (U, Sigma, V)
            self.layer.U.noise = sample_noise(self.layer.U.theta)
            self.layer.Sigma.noise = sample_noise(self.layer.Sigma.lamda)
            self.layer.V.noise = sample_noise(self.layer.V.theta)
        self.biasNoise = sample_noise(self.biasWeight, shape=(1, 1, self.nOutput))

    def forward(self, x):
        """Produce interleaved (y, z) product pairs from input x.

        Supports both 2D (batch, nInput) and 3D (batch, seq, nInput) inputs.
        """
        if not self.ergodic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0
        else:
            bias, temp = self.bias, self.var

        if temp != 0:
            self.resample_noise()
        self._input_ndim = x.ndim
        x = self.permute(x)

        if not self.naive:
            W = self.layer.compute_W(bias, temp)  # SVD components handle their own noise
        else:
            W  = (bias * self.W + temp * self.noise)

        ndim = x.ndim
        if ndim == 2:
            # 2D flattened: x is (batch, nInput), W is (nInput, nOutput)
            WX = x.unsqueeze(-1) * W.unsqueeze(0)       # (batch, nInput, nOutput)
            if self.hasBias:
                WX = WX + (bias*self.biasWeight.squeeze(0) + temp*self.biasNoise.squeeze(0))
            sWX = torch.tanh(WX)
            one_minus = 1 - sWX
            one_plus  = 1 + sWX
            if self.useEpsilon:
                one_minus += epsilon
                one_plus  += epsilon
            # Log-space summation avoids underflow when nInput is large.
            # Downstream layers see log-scale values; reverse uses subtraction
            # instead of log(z/y), preserving exact invertibility.
            log_y = torch.sum(torch.log(one_minus), dim=1)  # (batch, nOutput)
            log_z = torch.sum(torch.log(one_plus), dim=1)   # (batch, nOutput)
            stacked = torch.stack((log_y, log_z), dim=1)           # (batch, 2, nOutput)
            interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)  # (batch, 2*nOutput)
        else:
            # 3D: x is (batch, seq, nInput), W is (nInput, nOutput)
            WX = x.unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)  # (batch, seq, nInput, nOutput)
            if self.hasBias:
                WX = WX + (bias*self.biasWeight.unsqueeze(1) + temp*self.biasNoise.unsqueeze(1))
            sWX = torch.tanh(WX)
            one_minus = 1 - sWX
            one_plus  = 1 + sWX
            if self.useEpsilon:
                one_minus += epsilon
                one_plus  += epsilon
            # Log-space summation (see 2D comment above).
            log_y = torch.sum(torch.log(one_minus), dim=2)  # (batch, seq, nOutput)
            log_z = torch.sum(torch.log(one_plus), dim=2)   # (batch, seq, nOutput)
            stacked = torch.stack((log_y, log_z), dim=1)           # (batch, 2, seq, nOutput)
            interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)  # (batch, 2*seq, nOutput)

        result = self.unpermute(interleaved)
        return result

    def reverse(self, yz):
        """Recover x from interleaved (log_y, log_z): gamma = 0.5*(log_z - log_y) = Wx, then x = W^+ @ gamma.

        Forward outputs log-space values, so reverse uses subtraction instead
        of log(z/y), avoiding numerical issues with large nInput.

        Supports both 2D (batch, 2*nOutput) and 3D (batch, 2*seq, nOutput) inputs.
        """
        if not self.ergodic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0
        else:
            bias, temp = self.bias, self.var

        yz = self.permute(yz)
        ndim = yz.ndim

        if ndim == 2:
            # 2D flattened: yz is (batch, 2*nOutput) containing (log_y, log_z)
            n2 = yz.shape[1] // 2
            uninterleaved = torch.unflatten(yz, 1, (2, n2))  # (batch, 2, nOutput)
            log_y = uninterleaved[:, 0, :]  # (batch, nOutput)
            log_z = uninterleaved[:, 1, :]  # (batch, nOutput)
            gamma = 0.5 * (log_z - log_y)
            if self.hasBias:
                # Forward adds bias to each of nInput terms before summing,
                # so we must subtract nInput copies to recover x @ W.
                bias_corr = bias*self.biasWeight.squeeze(0) + temp*self.biasNoise.squeeze(0)
                gamma = gamma - self.nInput * torch.sum(bias_corr, dim=0)
        else:
            # 3D: yz is (batch, 2*seq, nOutput) containing (log_y, log_z)
            n2 = yz.shape[1] // 2
            uninterleaved = torch.unflatten(yz, 1, (2, n2))  # (batch, 2, seq, nOutput)
            log_y = uninterleaved[:, 0, :, :]  # (batch, seq, nOutput)
            log_z = uninterleaved[:, 1, :, :]  # (batch, seq, nOutput)
            gamma = 0.5 * (log_z - log_y)
            if self.hasBias:
                # Same nInput scaling as 2D case.
                gamma = gamma - self.nInput * torch.sum(bias*self.biasWeight + temp*self.biasNoise, dim=1).unsqueeze(1)

        if not self.naive:
            W_pinv = self.layer.compute_Winverse(bias, temp)  # SVD-based, internal noise
            x = gamma @ W_pinv
        else:
            W_pinv = torch.linalg.pinv( (self.W + temp*self.noise) )  # (nOutput, nInput)
            x = gamma @ W_pinv
        x = self.unpermute(x)

        if temp != 0:
            self.resample_noise()
        return x

    @staticmethod
    def test():
        nBatch    = 16
        nInput    = 3
        nOutput   = 2 * nInput
        nFeatures = 5

        layer = InvertiblePiLayer(nInput=nInput, nOutput=nOutput, naive=True, hasBias=True, permuteInput=True)
        x = torch.randn(nBatch, nInput, nFeatures)
        layer.setAlpha(0.00000001)
        yz = layer.forward(x)
        print("Forward output shape:", yz.shape)  # Should be (batch, out_features, 2)
        x_recon = layer.reverse(yz)
        print("Reconstructed x shape:", x_recon.shape)  # Should be (batch, in_features)

        error = torch.norm(x - x_recon) / torch.norm(x)
        print(f"Reconstruction relative error: {error.item():.6f}")
        assert error < 0.1, f"Reconstruction error too high: {error}"
        print("InvertiblePiLayer test passed.")

class VQLayer(Layer):
    """Vector-quantization layer backed by a residual VQ codebook.

    Flattens the input to (N, dim), quantizes each vector against a
    learned codebook using cosine similarity, and returns the codes from
    all quantizer stages.  The reverse pass is not implemented because
    codebook lookup is not uniquely invertible.
    """
    nOutput = 0

    def __init__(self, dim, codebookSize, numQuantizers):
        super(VQLayer, self).__init__(dim, dim)
        self.vq = ResidualVQ(
            dim=dim,
            codebook_size=codebookSize,
            num_quantizers=numQuantizers,
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=True,
            rotation_trick=True,  # rotation trick gradient estimator (vs STE)
        )

    def distance(self, x, y):
        """Euclidean distance between two tensors."""
        return torch.sqrt(torch.sum((x - y) ** 2))

    def forward(self, x, t=0):
        batch = len(x)
        x = x.reshape((-1, self.nInput))
        quantized, indices, commit_loss, all_codes = self.vq(x, return_all_codes=True)
        return all_codes

    def reverse(self, y, t=0):
        raise ValueError("VQLayer reverse is not defined; codebook lookup is not invertible.")
class DecisionBoundaryLayer(Layer):
    """Learns a hyperplane normal vector via online updates (not backprop).

    forward() returns +1/-1 on each side of the boundary.
    update() nudges the weight toward or away from an observation depending
    on which side it falls.
    """
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
        #bias, temp = self.bias, self.var
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
            raise NotImplementedError(
                "NormLayer pNorm=1 requires W prediction support in this branch."
            )
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
    """Scaled dot-product attention with optional symmetric (Hopfield-like) mode.

    In symmetric mode, a single projection A replaces Q and K, yielding
    scores = A^T @ A (positive semi-definite).  Standard QKV mode is used
    when ``symmetric=False``.
    """
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
    """MAE loss weighted by prediction magnitude (certainty).

    High-magnitude (confident) predictions are penalized more when wrong,
    blended with unweighted MAE via ``alpha``.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        abs_error = torch.abs(targets - predictions)
        certainty = torch.abs(predictions)
        loss = abs_error * (self.alpha * certainty + (1 - self.alpha))
        return torch.mean(loss)

class CertaintyWeightedMSELoss(nn.Module):
    """MSE loss weighted by prediction magnitude (certainty).

    Hybrid of certainty-weighted MSE and plain MSE, blended by ``alpha``.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        certainty = outputs.abs().sum(dim=1)                # per-sample confidence
        mse_loss = ((outputs - targets) ** 2).sum(dim=1)
        cw_mse_loss = mse_loss * certainty
        hybrid_loss = self.alpha * cw_mse_loss.mean() + (1 - self.alpha) * mse_loss.mean()
        return hybrid_loss

class CertaintyWeightedCrossEntropy(nn.Module):
    """Cross-entropy weighted by predicted probability of the true class.

    Hybrid of certainty-weighted CE and plain CE, blended by ``alpha``.
    """
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
class Mem:
    """Base class for temporal memory filters (exponential, gamma, mean, etc.).

    Subclasses implement ``delta()`` to update internal state from a new
    observation.  ``get()`` returns the current filtered output.  The
    ``removeRC``/``insertRC``/``setRC`` helpers support dynamic resizing
    of the output matrix (1-indexed for legacy compatibility).
    """
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

#region Logic

class LogicNet(nn.Module):
    def __init__(self, num_classes=10, groups_per_class=16, lut_rank=2):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if groups_per_class <= 0:
            raise ValueError("groups_per_class must be positive.")
        if lut_rank <= 0:
            raise ValueError("lut_rank must be positive.")
        flattened_dim = 32 * 7 * 7
        hidden_dim = max(256, (flattened_dim + lut_rank - 1) // lut_rank)
        min_grouped_logits = (hidden_dim + lut_rank - 1) // lut_rank
        grouped_logits = max(num_classes * groups_per_class, min_grouped_logits)
        if grouped_logits % num_classes != 0:
            grouped_logits += num_classes - (grouped_logits % num_classes)

        self.features = nn.Sequential(
            FixedBinarization(thresholds=[0.0]),
            LogicConv2d(
                in_dim=28,
                channels=1,
                num_kernels=16,
                tree_depth=3,
                receptive_field_size=5,
                padding=2
            ),
            OrPooling2d(kernel_size=2, stride=2, padding=0),

            LogicConv2d(
                in_dim=14,
                channels=16,
                num_kernels=32,
                tree_depth=3,
                receptive_field_size=3,
                padding=1
            ),
            OrPooling2d(kernel_size=2, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            LogicDense(in_dim=flattened_dim, out_dim=hidden_dim, lut_rank=lut_rank),
            LogicDense(in_dim=hidden_dim, out_dim=grouped_logits, lut_rank=lut_rank),
            GroupSum(k=num_classes, tau=8.0)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def test(batch_size=32, num_classes=10):
        model = LogicNet(num_classes=num_classes)
        model.eval()
        x = torch.rand(batch_size, 1, 28, 28)
        with torch.no_grad():
            output = model(x)
        expected_shape = (batch_size, num_classes)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"LogicNet produced {tuple(output.shape)}, expected {expected_shape}."
            )
        print(f"Output shape: {output.shape}")

#endregion


def test():
    LogicNet.test()
    torch.autograd.set_detect_anomaly(True)

    LinearLayer.test()
    InvertibleRotationLayer.test()
    InvertibleDiagonalLayer.test()
    InvertibleLinearLayer.test()

    InvertiblePiLayer.test()
    SigmaLayer.test()
    InvertibleSigmaLayer.test()
    PiLayer.test()
    PiLayer.xorTest()
    InvertiblePiLayer.test()

    AttentionLayer.test()
    NormLayer.test()
    Mem.test()
    DecisionBoundaryLayer.test()

def main():
    try:
        test()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

# Self-test: run the LogicNet smoke test when executed as a script.
if __name__ == "__main__":
    main()
