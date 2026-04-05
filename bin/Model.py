"""Core layer primitives used by BasicModel.

This module mixes conventional neural-network utilities with a set of
custom reversible, ergodic, and memory-style layers.  Most higher-level
model construction happens in ``BasicModel.py``; this file provides the
building blocks and the update rules they share.
"""
from __future__ import annotations  # allow X | Y union syntax on Python 3.9

import os
import warnings
import numpy as np
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from itertools import chain
import torch.optim as optim
import time
from typing import Optional, Tuple
from collections import namedtuple


epsilon = 1e-7  # to avoid log(0)

# Device used by all layers.
from util import TheXMLConfig, TheDevice

from util import TheMessage

#region Layers
class Layer(nn.Module):
    """Base class for custom layers with optional symbol/object axis swapping.

    Composite layers should register their child layers in ``self.layers``
    so that the ergodic interface (set_sigma, observe_sigma, etc.) and
    paramUpdate are automatically forwarded to them.
    """
    def __init__(self, nInput, nOutput):
        super(Layer, self).__init__()
        self.nInput       = nInput
        self.nOutput      = nOutput
        self.batch        = 0
        self.layers       = []  # child layers; populate in subclass __init__

    def freeze(self, learn=False):
        """Freeze all params (learn=False) or unfreeze them (learn=True)."""
        for param in self.parameters():
            param.requires_grad = not learn
    def getParameters(self):
        params = [p for n, p in self.named_parameters()]
        return params

    # --- Ergodic interface: dispatched to self.layers automatically ---
    def paramUpdate(self):
        for layer in self.layers:
            layer.paramUpdate()
    def set_sigma(self, sigma):
        for layer in self.layers:
            layer.set_sigma(sigma)
    def observe_sigma(self):
        for layer in self.layers:
            layer.observe_sigma()
    def sigma_to_ergodic(self):
        for layer in self.layers:
            layer.sigma_to_ergodic()

    def forward(self, x):
        """Identity pass-through (subclasses override)."""
        batch = x.shape[0]
        return x
    def reverse(self, y):
        """Identity pass-through (subclasses override)."""
        batch = y.shape[0]
        # For 3D tensors, nOutput matches the last dim (embedding), not dim 1 (sequence)
        if y.ndim == 3:
            assert y.shape[2] == self.nOutput
        else:
            assert y.shape[1] == self.nOutput
        return y
class ErgodicLayer(Layer):
    """Layer base class that adapts its explore/exploit balance over training.

    Tracks **sigma** — the running variance of the layer's gradient energy.
    Sigma drives scalar bias and var that broadcast over any weight shape:

        var  = sigma / (sigma + kappa)      exploration noise
        bias = 1 - var                      weight trust

    Low sigma (consistent gradient, found a minimum) → high bias, low var.
    High sigma (unstable gradient) → low bias, high var.

    Subclasses mix learned weights (scaled by ``bias``) with random noise
    (scaled by ``var``) in their forward passes:
        effective_weight = bias * W + var * noise

    External control via ``set_sigma(sigma)``:
        sigma=1: responsive exploration (low kappa)
        sigma=0: suppress exploration (high kappa, var ≈ 0)

    ``ergodic=False`` (default): sigma tracking is still wired but
    sigma_to_ergodic / paramUpdate are no-ops, keeping bias=1, var=0.
    """
    def __init__(self, nInput, nOutput, ergodic=False):
        super().__init__(nInput, nOutput)
        self.ergodic = ergodic

        # --- Scalar explore/exploit (broadcasts over any weight shape) -----
        # Start in pure-exploit mode; sigma_to_ergodic drives these after
        # the first gradient step.
        self.register_buffer('bias', torch.ones(1))
        self.register_buffer('var',  torch.zeros(1))

        # --- Scalar sigma: running gradient variance ----------------------
        self.register_buffer('sigma',      torch.zeros(1))
        self.register_buffer('sigma_mean', torch.zeros(1))
        self.register_buffer('sigma_step', torch.tensor(0))
        self.sigma_beta = 0.99
        self.sigma_kappa = 0.01

        self.dropoutRate = 0.0
        # Initialize in moderate exploration mode
        self.set_sigma(0.5)

    def getParameters(self):
        """Return learnable params (var/bias are buffers, not parameters)."""
        return list(self.parameters())

    def set_sigma(self, sigma):
        """Control exploration meta-parameters.

        sigma=1: encourage exploration (low kappa, responsive to gradient variance)
        sigma=0: suppress exploration (high kappa, var ≈ 0)
        """
        if sigma == 0:
            self.sigma_kappa = 1e6
        else:
            self.sigma_kappa = 0.01 / sigma
            self.sigma_beta = 0.99

    @torch.no_grad()
    def observe_sigma(self):
        """Track gradient variance via Welford's algorithm.

        Aggregates mean squared gradient energy across all weight parameters
        into a single scalar, then updates the running mean and variance.
        """
        grad_energy = None
        for name, param in self.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            if "noise" in name.lower():
                continue
            energy = param.grad.detach().pow(2).mean()
            grad_energy = energy if grad_energy is None else grad_energy + energy
        if grad_energy is None:
            return
        self.sigma_step += 1
        beta = self.sigma_beta
        delta = grad_energy - self.sigma_mean
        self.sigma_mean.add_((1 - beta) * delta)
        self.sigma.mul_(beta).add_((1 - beta) * delta * (grad_energy - self.sigma_mean))

    @torch.no_grad()
    def sigma_to_ergodic(self):
        """Update scalar bias and var from sigma (gradient variance)."""
        if not self.ergodic:
            return
        if self.sigma_step.item() == 0:
            return
        # Bias-corrected sigma estimate
        s = self.sigma / (1 - self.sigma_beta ** self.sigma_step.item())
        s = s.clamp(min=0)
        self.var.copy_((s / (s + self.sigma_kappa)).clamp(0, 0.95))
        self.bias.copy_((1.0 - self.var).clamp(min=0.05))

    def paramUpdate(self):
        if not self.ergodic:
            return
        self.observe_sigma()
        self.sigma_to_ergodic()

class LinearLayer(ErgodicLayer):
    """Standard linear (affine) layer.

    Forward (ergodic):     y = x @ (bias*W + var*noise) [+ bias*biasWeight + var*biasNoise]
    Forward (non-ergodic): y = x @ W [+ b]

    ``bias`` and ``var`` come from the ErgodicLayer explore/exploit schedule.
    When ergodic=False (default), noise buffers are not allocated and forward
    uses the learned weights directly.
    """
    def __init__(self, nInput, nOutput, hasBias=True, naive=False, stable=False, ergodic=False):
        super(LinearLayer, self).__init__(nInput, nOutput, ergodic=ergodic)
        self.stable  = stable
        self.hasBias = hasBias
        if ergodic:
            self.W = nn.Parameter(torch.eye(self.nInput, self.nOutput))
        else:
            self.W = nn.Parameter(torch.randn(self.nInput, self.nOutput))
        if self.hasBias:
            self.biasWeight = nn.Parameter(torch.zeros(1, nOutput))
        if ergodic:
            self.register_buffer('noise', torch.randn(self.nInput, self.nOutput))
            if self.hasBias:
                self.register_buffer('biasNoise', torch.randn(1, nOutput))
        else:
            self.set_sigma(0)

    def resample_noise(self):
        """Draw fresh Gaussian noise matching W shape/device. No-op if not ergodic."""
        if self.ergodic:
            self.noise = torch.randn(self.W.shape, device=TheDevice.get(), dtype=self.W.dtype)
            if self.hasBias:
                self.biasNoise = torch.randn(
                    self.biasWeight.shape,
                    device=TheDevice.get(),
                    dtype=self.biasWeight.dtype,
                )

    def compute_W_current(self):
        """Effective W for outer-product use (respects ergodic noise if active)."""
        if self.ergodic:
            return self.bias * self.W + self.var * self.noise
        return self.W

    def forward(self, x):
        if self.ergodic:
            self.resample_noise()
            W = self.bias * self.W + self.var * self.noise
        else:
            W = self.W
        output = x @ W
        return output
    def forwardBias(self, x):
        if self.hasBias:
            if self.ergodic:
                x = x + self.bias * self.biasWeight + self.var * self.biasNoise
            else:
                x = x + self.biasWeight
        return x

    def _effective_bias(self):
        """Bias for log-space use (unconstrained)."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            return self.bias * self.biasWeight + self.var * self.biasNoise
        return self.biasWeight

    @staticmethod
    def test():
        nInput, nOutput = 3, 4
        layer = LinearLayer(nInput=nInput, nOutput=nOutput)
        input = torch.rand((1, nInput), device=TheDevice.get())
        output = layer(input)

        print(f"Input: {input}")
        print(f"After forward linear: {output}")

class InvertibleLinearLayer(ErgodicLayer):
    """Exactly-invertible linear layer factored as W = L @ D_embed @ U.

    L is unit-lower-triangular [nInput, nInput], D is diagonal [rank],
    U is unit-upper-triangular [nOutput, nOutput].  D_embed zero-pads D
    into [nInput, nOutput] for rectangular cases.

    Non-ergodic inverse is exact via triangular solves: W⁻¹ = U⁻¹ D⁻¹ L⁻¹.

    Ergodic mode injects noise into each factor before extracting the
    triangular structure, preserving the LDU form so the exact inverse is
    always available — no approximation or SVD required:
        L_eff = I + strict_lower(raw_L + t * noise_raw_L)
        U_eff = I + strict_upper(raw_U + t * noise_raw_U)
        d_eff = b * d_effective + t * noise_d
        W_eff = L_eff @ D_eff_embed @ U_eff

    stable=True clamps d to [eps, 1] magnitude (sign preserved) via
    _d_effective(), keeping W well-conditioned and invertible.

    naive=False (default): sequential triangular solves, no W materialisation.
    naive=True: materialise W_eff as dense matrix; reverse uses pinv(W_eff).
    """
    def __init__(self, nInput, nOutput, naive=False, ergodic=False,
                 hasBias=True, stable=False):
        super().__init__(nInput, nOutput, ergodic=ergodic)
        self.naive   = naive
        self.hasBias = hasBias
        self.stable  = stable
        self.rank    = min(nInput, nOutput)

        # LDU learned parameters
        self.raw_L = nn.Parameter(torch.zeros(nInput, nInput))
        self.d     = nn.Parameter(torch.ones(self.rank))
        self.raw_U = nn.Parameter(torch.zeros(nOutput, nOutput))
        if hasBias:
            self.biasWeight = nn.Parameter(torch.zeros(1, nOutput))
        if ergodic:
            # Factor-level noise: perturb each LDU factor independently
            self.register_buffer('noise_raw_L', torch.randn(nInput, nInput))
            self.register_buffer('noise_raw_U', torch.randn(nOutput, nOutput))
            self.register_buffer('noise_d',     torch.ones(self.rank))
            if hasBias:
                self.register_buffer('biasNoise', torch.randn(1, nOutput))

    # --- Factor helpers ---
    def _L(self):
        """Unit-lower-triangular: strict lower of raw_L + I."""
        return torch.tril(self.raw_L, diagonal=-1) + torch.eye(
            self.nInput, device=self.raw_L.device, dtype=self.raw_L.dtype)

    def _U(self):
        """Unit-upper-triangular: strict upper of raw_U + I."""
        return torch.triu(self.raw_U, diagonal=1) + torch.eye(
            self.nOutput, device=self.raw_U.device, dtype=self.raw_U.dtype)

    def _d_effective(self):
        """Return d clamped to [eps, 1] magnitude with sign preserved when
        stable=True, else raw self.d.  Stability constraint lives here only."""
        if self.stable:
            return self.d.sign() * self.d.abs().clamp(epsilon, 1.0)
        return self.d

    def _D_embed(self):
        """Embed _d_effective() into [nInput, nOutput] rectangular diagonal."""
        d = self._d_effective()
        D = torch.zeros(self.nInput, self.nOutput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D[i, i] = d[i]
        return D

    # --- Noise resampling ---
    def resample_noise(self):
        if not self.ergodic:
            return
        # Mask to the entries that survive tril/triu, then unit-normalise so
        # that var is the exact Frobenius norm of the perturbation regardless
        # of matrix size.
        n_L = torch.tril(torch.randn_like(self.noise_raw_L), diagonal=-1)
        self.noise_raw_L = n_L / (n_L.norm() + 1e-8)

        n_U = torch.triu(torch.randn_like(self.noise_raw_U), diagonal=1)
        self.noise_raw_U = n_U / (n_U.norm() + 1e-8)

        n_d = torch.randn_like(self.noise_d)
        self.noise_d = n_d / (n_d.norm() + 1e-8)

        if self.hasBias:
            n_b = torch.randn_like(self.biasNoise)
            self.biasNoise = n_b / (n_b.norm() + 1e-8)

    # --- Effective ergodic factors ---
    def _L_eff(self):
        """L_eff = I + strict_lower(raw_L + var * noise_raw_L).
        noise_raw_L is pre-masked to strict lower triangular and unit-normalised."""
        raw = self.raw_L + self.var * self.noise_raw_L
        return (torch.tril(raw, diagonal=-1)
                + torch.eye(self.nInput, device=raw.device, dtype=raw.dtype))

    def _U_eff(self):
        """U_eff = I + strict_upper(raw_U + var * noise_raw_U).
        noise_raw_U is pre-masked to strict upper triangular and unit-normalised."""
        raw = self.raw_U + self.var * self.noise_raw_U
        return (torch.triu(raw, diagonal=1)
                + torch.eye(self.nOutput, device=raw.device, dtype=raw.dtype))

    def _d_eff(self):
        """d_eff = bias * d_effective + var * noise_d.
        noise_d is unit-normalised so var is the exact L2 norm of the perturbation.
        When stable=True, clamp magnitude to [eps, 1] so W_inv never blows up."""
        d = self.bias * self._d_effective() + self.var * self.noise_d
        if self.stable:
            d = d.sign() * d.abs().clamp(epsilon, 1.0)
        return d

    # --- W materialisation ---
    def compute_W(self):
        """W = L @ D_embed @ U.  Shape [nInput, nOutput]."""
        return self._L() @ self._D_embed() @ self._U()

    def compute_Winverse(self):
        """W⁻¹ = U⁻¹ @ D_embed⁻¹ @ L⁻¹.  Shape [nOutput, nInput].
        Uses _d_effective() so stable=True is automatically respected."""
        L = self._L()
        U = self._U()
        I_in  = torch.eye(self.nInput,  device=L.device, dtype=L.dtype)
        I_out = torch.eye(self.nOutput, device=U.device, dtype=U.dtype)
        L_inv = torch.linalg.solve_triangular(L, I_in,  upper=False, unitriangular=True)
        U_inv = torch.linalg.solve_triangular(U, I_out, upper=True,  unitriangular=True)
        d = self._d_effective()
        D_inv = torch.zeros(self.nOutput, self.nInput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D_inv[i, i] = 1.0 / d[i]
        return U_inv @ D_inv @ L_inv

    def compute_W_current(self):
        """Materialise W using current ergodic factors if ergodic, else clean W."""
        if not self.ergodic:
            return self.compute_W()
        L = self._L_eff()
        d = self._d_eff()
        U = self._U_eff()
        D = torch.zeros(self.nInput, self.nOutput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D[i, i] = d[i]
        return L @ D @ U

    def compute_Winverse_current(self):
        """Exact inverse of compute_W_current(). Shape [nOutput, nInput]."""
        if not self.ergodic:
            return self.compute_Winverse()
        L = self._L_eff()
        d = self._d_eff()
        U = self._U_eff()
        I_in  = torch.eye(self.nInput,  device=L.device, dtype=L.dtype)
        I_out = torch.eye(self.nOutput, device=U.device, dtype=U.dtype)
        L_inv = torch.linalg.solve_triangular(L, I_in,  upper=False, unitriangular=True)
        U_inv = torch.linalg.solve_triangular(U, I_out, upper=True,  unitriangular=True)
        D_inv = torch.zeros(self.nOutput, self.nInput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D_inv[i, i] = 1.0 / d[i]
        return U_inv @ D_inv @ L_inv

    # --- Parameterised sequential apply / solve ---
    def _apply_ldu(self, x, L, d, U):
        """Apply x @ L @ D_embed(d) @ U sequentially.  d is a [rank] vector.
        Supports arbitrary leading batch dimensions."""
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        x = x @ L
        if self.nInput <= self.nOutput:
            scaled = x * d
            if self.nOutput > self.nInput:
                pad = torch.zeros(x.shape[0], self.nOutput - self.nInput,
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([scaled, pad], dim=-1)
            else:
                x = scaled
        else:
            # rank == nOutput here (min(nInput,nOutput)), so no zero-pad needed
            x = x[..., :self.rank] * d
        x = x @ U
        out_shape = list(orig_shape); out_shape[-1] = self.nOutput
        return x.reshape(out_shape)

    def _solve_ldu(self, y, L, d, U):
        """Solve y @ (L D U) = x exactly via triangular solves.  d is [rank].
        Supports arbitrary leading batch dimensions."""
        orig_shape = y.shape
        y = y.reshape(-1, orig_shape[-1])
        y = torch.linalg.solve_triangular(U.T, y.T, upper=False, unitriangular=True).T
        if self.nInput <= self.nOutput:
            # rank == nInput here (min(nInput,nOutput)), so no zero-pad needed
            y = y[..., :self.rank] / d
        else:
            y = y / d
            pad = torch.zeros(y.shape[0], self.nInput - self.rank,
                              device=y.device, dtype=y.dtype)
            y = torch.cat([y, pad], dim=-1)
        y = torch.linalg.solve_triangular(L.T, y.T, upper=True, unitriangular=True).T
        out_shape = list(orig_shape); out_shape[-1] = self.nInput
        return y.reshape(out_shape)

    def _apply_forward(self, x):
        """Non-ergodic sequential forward using clean L, d, U."""
        return self._apply_ldu(x, self._L(), self._d_effective(), self._U())

    def _solve_reverse(self, y):
        """Non-ergodic sequential reverse using clean L, d_effective, U."""
        return self._solve_ldu(y, self._L(), self._d_effective(), self._U())

    # --- Forward / Reverse ---
    def forward(self, x):
        """Apply the LDU transform with optional ergodic noise injection.

        Ergodic (self.ergodic=True):
          Resamples noise at the start, then builds effective factors:
            L_eff = I + strict_lower(raw_L + t * noise_raw_L)
            U_eff = I + strict_upper(raw_U + t * noise_raw_U)
            d_eff = b * d_effective + t * noise_d
          naive=True: materialise W_eff = L_eff D_eff U_eff, dense matmul.
          naive=False: sequential triangular solves via _apply_ldu.
          Stored noise buffers remain unchanged after this call so that
          reverse() can reconstruct the identical factors exactly.

        Non-ergodic (self.ergodic=False):
          naive=True: materialise W = L D U, dense matmul.
          naive=False: sequential triangular solves.
        """
        if self.ergodic:
            self.resample_noise()
        if self.naive:
            W = self.compute_W_current()
            orig_shape = x.shape
            out_shape = list(orig_shape); out_shape[-1] = self.nOutput
            y = (x.reshape(-1, self.nInput) @ W).reshape(out_shape)
        else:
            if self.ergodic:
                y = self._apply_ldu(x, self._L_eff(), self._d_eff(), self._U_eff())
            else:
                y = self._apply_forward(x)
        y = self.forwardBias(y)
        return y
    def forwardBias(self, x):
        if self.hasBias:
            if self.ergodic:
                x = x + self.bias * self.biasWeight + self.var * self.biasNoise
            else:
                x = x + self.biasWeight
        return x
    def forwardBiasInterleaved(self, x):
        if self.hasBias:
            # [..., 2*S, nOut]: pairs along dim=-2, alternate +b/-b every row
            signs = x.new_ones(x.shape[-2], 1)
            signs[1::2, 0] = -1
            bWeight = signs * self.biasWeight
            if self.ergodic:
                bNoise = signs * self.biasNoise
                x = x + self.bias * bWeight + self.var * bNoise
            else:
                x = x + bWeight
        return x

    def reverseBias(self, y):
        if self.hasBias:
            if self.ergodic:
                y = y - (self.bias * self.biasWeight + self.var * self.biasNoise)
            else:
                y = y - self.biasWeight
        return y

    def reverseBiasInterleaved(self, y):
        if self.hasBias:
            # [..., 2*S, nOut]: pairs along dim=-2, alternate +b/-b every row
            signs = y.new_ones(y.shape[-2], 1)
            signs[1::2, 0] = -1
            bWeight = signs * self.biasWeight
            if self.ergodic:
                bNoise = signs * self.biasNoise
                y = y - (self.bias * bWeight + self.var * bNoise)
            else:
                y = y - bWeight
        return y
    def reverse(self, y):
        """Invert the LDU transform.

        Ergodic (self.ergodic=True):
          Reconstructs L_eff, d_eff, U_eff from the noise buffers set during
          the preceding forward() call — no new resampling is done until after
          the result is computed.
          naive=False: triangular solves (exact, no materialisation).
          naive=True: materialise W_eff, apply pinv.
          Resamples noise at end so the layer is ready for the next forward().

        Non-ergodic (self.ergodic=False):
          naive=True: materialise W_inv = U_inv D_inv L_inv, dense matmul.
          naive=False: sequential triangular solves.
        """
        y = self.reverseBias(y)
        if self.naive:
            W_inv = self.compute_Winverse_current()
            orig_shape = y.shape
            out_shape = list(orig_shape); out_shape[-1] = self.nInput
            result = (y.reshape(-1, self.nOutput) @ W_inv).reshape(out_shape)
        else:
            if self.ergodic:
                result = self._solve_ldu(y, self._L_eff(), self._d_eff(), self._U_eff())
            else:
                result = self._solve_reverse(y)
        if self.ergodic:
            self.resample_noise()
        return result

    @staticmethod
    def test():
        torch.manual_seed(42)
        device = TheDevice.get()
        nInput, nOutput = 7, 11

        # Non-ergodic roundtrip (expand)
        layer = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, hasBias=False)
        layer.set_sigma(0)
        x = torch.randn(5, nInput, device=device)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = torch.norm(x - x_rec) / torch.norm(x)
        assert err < 1e-4, f"roundtrip error: {err:.2e}"
        print(f"InvertibleLinearLayer roundtrip: err={err:.2e} OK")

        # With bias
        layer2 = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, hasBias=True)
        layer2.set_sigma(0)
        y2 = layer2.forward(x)
        x_rec2 = layer2.reverse(y2)
        err2 = torch.norm(x - x_rec2) / torch.norm(x)
        assert err2 < 1e-4, f"bias roundtrip error: {err2:.2e}"
        print(f"InvertibleLinearLayer bias roundtrip: err={err2:.2e} OK")

        # Square W @ W_inv = I
        layer3 = InvertibleLinearLayer(nInput=5, nOutput=5, hasBias=False)
        layer3.set_sigma(0)
        W = layer3.compute_W(); W_inv = layer3.compute_Winverse()
        id_err = torch.norm(W @ W_inv - torch.eye(5, device=W.device))
        assert id_err < 1e-4, f"W@W_inv identity err: {id_err:.2e}"
        print(f"InvertibleLinearLayer W@W_inv identity: err={id_err:.2e} OK")

        # Ergodic roundtrip (factor-level noise → exact inverse)
        elayer = InvertibleLinearLayer(nInput=5, nOutput=5, ergodic=True, stable=True)
        with torch.no_grad():
            elayer.var.fill_(0.2)
            elayer.bias.fill_(0.8)
        x3 = torch.randn(4, 5, device=device)
        y3 = elayer.forward(x3)
        x_rec3 = elayer.reverse(y3)
        err3 = torch.norm(x3 - x_rec3) / torch.norm(x3)
        assert err3 < 1e-4, f"ergodic roundtrip error: {err3:.2e}"
        print(f"InvertibleLinearLayer ergodic roundtrip: err={err3:.2e} OK")

        print("All InvertibleLinearLayer tests passed!")

class NonNegativeInvertibleLinearLayer(InvertibleLinearLayer):
    """InvertibleLinearLayer whose W = L @ D @ U is entry-wise non-negative.

    Applies softplus to off-diagonal entries of L and U, and to the
    diagonal d, so that all factors are non-negative.  Since the product
    of non-negative matrices is non-negative, W >= 0 by construction.

    The exact LDU inverse (triangular solves) is still available — the
    inverse of a non-negative W is not itself non-negative, but that's
    fine since only the forward W needs the constraint.

    Parameters are initialised so that softplus(raw) starts near the
    identity: raw_L, raw_U off-diagonals at -5 (softplus(-5) ~ 0.007),
    d at softplus_inverse(1) ~ 0.541.
    """

    def __init__(self, nInput, nOutput, naive=False, ergodic=False,
                 hasBias=True, stable=False):
        super().__init__(nInput, nOutput, naive=naive, ergodic=ergodic,
                         hasBias=hasBias, stable=stable)
        # Re-initialise so softplus(raw) gives near-identity W.
        with torch.no_grad():
            # softplus(-5) ~ 0.007 ≈ 0 for off-diagonals
            self.raw_L.fill_(-5.0)
            self.raw_U.fill_(-5.0)
            # softplus(0.5414) ~ 1.0 for diagonal
            self.d.fill_(math.log(math.e - 1))  # softplus_inverse(1.0)

    # --- Non-negative factor helpers ---
    def _L(self):
        """Unit-lower-triangular with non-negative off-diagonals."""
        off = torch.tril(nn.functional.softplus(self.raw_L), diagonal=-1)
        return off + torch.eye(self.nInput, device=self.raw_L.device, dtype=self.raw_L.dtype)

    def _U(self):
        """Unit-upper-triangular with non-negative off-diagonals."""
        off = torch.triu(nn.functional.softplus(self.raw_U), diagonal=1)
        return off + torch.eye(self.nOutput, device=self.raw_U.device, dtype=self.raw_U.dtype)

    def _d_effective(self):
        """Strictly positive diagonal via softplus."""
        d = nn.functional.softplus(self.d)
        if self.stable:
            d = d.clamp(epsilon, 1.0)
        return d

    # --- Ergodic overrides ---
    def _L_eff(self):
        """Ergodic L with non-negative off-diagonals."""
        raw = self.raw_L + self.var * self.noise_raw_L
        off = torch.tril(nn.functional.softplus(raw), diagonal=-1)
        return off + torch.eye(self.nInput, device=raw.device, dtype=raw.dtype)

    def _U_eff(self):
        """Ergodic U with non-negative off-diagonals."""
        raw = self.raw_U + self.var * self.noise_raw_U
        off = torch.triu(nn.functional.softplus(raw), diagonal=1)
        return off + torch.eye(self.nOutput, device=raw.device, dtype=raw.dtype)

    def _d_eff(self):
        """Ergodic d, strictly positive via softplus."""
        d_base = nn.functional.softplus(self.d)
        d = self.bias * d_base + self.var * self.noise_d
        d = nn.functional.softplus(d)  # ensure positive after noise
        if self.stable:
            d = d.clamp(epsilon, 1.0)
        return d


    def _effective_bias(self):
        """Bias for log-space use. Constrained to ≤ 0 so exp(wx+b) ∈ (0,1]."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            raw = self.bias * self.biasWeight + self.var * self.biasNoise
            return F.softplus(raw)                        # ≤ 0 even with noise
        return F.softplus(self.biasWeight)                # ≥ 0

    def forwardBias(self, x):
        if self.hasBias:
            x = x + self._effective_bias()                # _effective_bias ≤ 0, so adds ≥ 0
        return x

    def reverseBias(self, x):
        if self.hasBias:
            x = x - self._effective_bias()                # _effective_bias ≤ 0, so subtracts ≥ 0
        return x


class LiftingLayer(InvertibleLinearLayer):
    """Bias-free, stable reversible linear layer for mapping between row/column spaces."""
    def __init__(self, nInput, nOutput, init='orthogonal'):
        super().__init__(nInput, nOutput, naive=False, hasBias=False, stable=True)

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

class SigmaLayer(Layer):
    """Additive (summation) layer: y = tanh(W @ x + b).

    When ``invertible=True``, uses InvertibleLinearLayer so ``reverse()``
    is available via the exact LDU inverse.  When ``invertible=False``
    (default), uses a plain LinearLayer.

    The logit/sigmoid domain transforms that map between (0,1) and (-1,1)
    are applied by ConceptualSpace, not here — they encode the pipeline
    contract (perceptual→conceptual boundary), not a property of this layer.

    All ergodic machinery lives in the inner layer; SigmaLayer dispatches
    the ergodic interface (set_sigma, observe_sigma, etc.) there.
    """
    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, nonlinear=False):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.ergodic    = ergodic
        self.saturate   = True
        self.activation = torch.zeros(1, nOutput, 1)
        if invertible:
            self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def forward(self, x):
        y = self.layer.forward(x)
        if self.saturate:
            self.activation = torch.tanh(y)
            y = self.activation.clone()
        return y

    def reverse(self, y):
        """Invert tanh then apply W⁻¹. Requires invertible=True."""
        if self.saturate:
            y = y.clamp(min=-1 + epsilon, max=1 - epsilon)
            self.activation = torch.atanh(y)
            y = self.activation.clone()
        x = self.layer.reverse(y)
        return x

    @staticmethod
    def test():
        nInput, nOutput = 3, 4
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput)

        x = torch.randn((2, 5, nInput), device=TheDevice.get())
        layer.set_sigma(0.999)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(layer.getParameters(), lr=0.01)
        optimizer.zero_grad()
        y = layer(x)
        loss = criterion(y, y)
        loss.backward()
        print(f"Original input: {x}")
        print(f"After linear: {y}")

        nInput, nOutput = 5, 7
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=False, invertible=True)
        x = torch.randn((2, 5, nInput), device=TheDevice.get())
        layer.set_sigma(0.0)
        y = layer.forward(x)
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=True, invertible=True)
        x = torch.randn((4, 8, nInput), device=TheDevice.get())
        layer.set_sigma(0.0)
        y = layer.forward(x)
        assert y.shape == (4, 8, nOutput), "Incorrect Size"
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        print("SigmaLayer tests passed.")

class PiLayer(Layer):
    r"""Log-space multiplicative layer: [-1,1] → [-1,1].

    Inputs and outputs are in [-1, 1].  An affine shift maps to
    [ε, 1] for log-space, and the inverse maps back:
        to_log:   x' = (x + 1) / 2 * (1 - ε) + ε    [-1,1] → [ε, 1]
        from_log: x  = ((x' - ε) / (1 - ε)) * 2 - 1  [ε, 1] → [-1,1]

    Forward:
        y = from_log(exp(W @ log(to_log(x)) + b))   # b is multiplicative in data space

    Reverse (invertible=True):
        x = from_log(exp(W_inv @ (log(to_log(y)) − b)))

    No clamping is used — the affine mapping guarantees log-safe inputs.
    """
    # ε for the affine mapping [-1,1] → [ε, 1].
    # 1e-6 gives log(ε) ≈ -13.8, well within float32 range.
    _eps = 1e-6

    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.stable     = stable
        self.hasBias    = hasBias
        if invertible:
            self.layer = NonNegativeInvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                               naive=naive, ergodic=ergodic,
                                               stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=hasBias,
                                     naive=naive, ergodic=ergodic, stable=stable)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def resample_noise(self):
        self.layer.resample_noise()

    def _to_log_domain(self, x):
        """Map [-1, 1] → [ε, 1]."""
        return (x + 1) / 2 * (1 - self._eps) + self._eps

    def _from_log_domain(self, xp):
        """Map [ε, 1] → [-1, 1]."""
        return ((xp - self._eps) / (1 - self._eps)) * 2 - 1

    def forward(self, x):
        if self.layer.ergodic:
            self.resample_noise()
        W = self.layer.compute_W_current()                 # [nIn, nOut], non-negative by construction
        x = x.to(W.device)
        xp = self._to_log_domain(x)                      # [ε, 1]
        log_x = torch.log(xp)                            # [-13.8, 0]
        wx = log_x @ W                                   # [..., nOut], ≤ 0
        b  = self.layer._effective_bias()
        wx = wx - b                                       # bias in log-space (multiplicative)
        y = torch.exp(wx)                                 # (0, 1] when b ≤ 0
        result = self._from_log_domain(y)                 # [ε, 1] → [-1, 1]
        if not self.invertible:
            result = torch.tanh(result)                   # gate for unconstrained W
        return result

    def reverse(self, y):
        """Recover x from y.  Requires invertible=True.

        y' = to_log(y)                                    # [-1,1] → [ε, 1]
        x' = exp(W_inv @ (log(y') − b))                  # undo log-space bias
        x  = from_log(x')                                 # [ε, 1] → [-1, 1]

        Expects y in [-1, 1] (the forward output range).
        """
        W_inv = self.layer.compute_Winverse_current()       # [nOut, nIn], exact LDU inverse
        y = y.to(W_inv.device)
        yp = self._to_log_domain(y)                        # [-1, 1] → [ε, 1]
        log_y = torch.log(yp)
        b = self.layer._effective_bias()
        log_x = (log_y + b) @ W_inv                       # [..., nIn]
        xp = torch.exp(log_x)
        x = self._from_log_domain(xp)
        if self.layer.ergodic:
            self.resample_noise()
        return x

    @staticmethod
    def test():
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayer(nInput=nInput, nOutput=nOutput)
        device = next(layer.parameters()).device
        # Inputs in [-1, 1]
        x = torch.rand((nBatch, 6, nInput), device=device) * 2 - 1
        layer.set_sigma(0.999)
        y = layer(x)
        assert y.shape == (nBatch, 6, nOutput), f"shape mismatch: {y.shape}"
        assert torch.isfinite(y).all(), "PiLayer forward produced non-finite values"
        assert torch.all(y > 0), "PiLayer output must stay positive"
        print(f"PiLayer forward: input {x.shape} -> output {y.shape}")

        def check_roundtrip(desc, **kwargs):
            kw = dict(nInput=3, nOutput=6, invertible=True)
            kw.update(kwargs)
            layer = PiLayer(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            # Inputs in [-1, 1]
            inputs = [('3D [B,S,nIn]', torch.rand(16, 5, nI, device=device) * 2 - 1),
                      ('2D [B,nIn]',   torch.rand(16, nI,    device=device) * 2 - 1)]
            for tag, x in inputs:
                layer.set_sigma(0.0)
                y = layer.forward(x)
                x_recon = layer.reverse(y)
                error = torch.norm(x - x_recon) / torch.norm(x)
                assert error < 1e-4, f"{desc} {tag}: reconstruction error {error:.2e}"
            print(f"  {desc}: OK")

        def check_stability(desc, **kwargs):
            kw = dict(nInput=3, nOutput=6, invertible=True, stable=True)
            kw.update(kwargs)
            layer = PiLayer(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            dtype = next(layer.parameters()).dtype
            layer.set_sigma(0.0)

            # Test boundary values in [-1, 1]
            x_edge = torch.tensor(
                [[-1.0] * nI,
                 [0.0] * nI,
                 [1.0] * nI,
                 [-0.5, 0.0, 0.5][:nI] if nI <= 3 else [-0.5, 0.0, 0.5] + [0.25] * (nI - 3)],
                device=device,
                dtype=dtype,
            )

            y = layer.forward(x_edge)
            assert torch.isfinite(y).all(), f"{desc}: forward produced non-finite values"
            assert torch.all(y > 0), f"{desc}: forward produced non-positive values"

            x_recon = layer.reverse(y)
            assert torch.isfinite(x_recon).all(), f"{desc}: reverse produced non-finite values"
            print(f"  {desc}: OK")

        print("Invertible PiLayer roundtrip variations:")
        check_roundtrip("naive=T hasBias=T", naive=True,  hasBias=True)
        check_roundtrip("naive=T hasBias=F", naive=True,  hasBias=False)
        check_roundtrip("naive=F hasBias=T", naive=False, hasBias=True)
        check_roundtrip("naive=F hasBias=F", naive=False, hasBias=False)
        check_roundtrip("square nIn=nOut=6", naive=True,  hasBias=False, nInput=6, nOutput=6)
        check_roundtrip("ergodic naive=T hasBias=F", naive=True,  hasBias=False, ergodic=True)
        check_roundtrip("ergodic naive=T hasBias=T", naive=True,  hasBias=True,  ergodic=True)
        check_roundtrip("ergodic naive=F hasBias=T", naive=False, hasBias=True,  ergodic=True)

        print("Stable PiLayer boundary variations:")
        check_stability("stable naive=T hasBias=T", naive=True, hasBias=True)
        check_stability("stable naive=T hasBias=F", naive=True, hasBias=False)
        check_stability("stable naive=F hasBias=T", naive=False, hasBias=True)
        check_stability("stable naive=F hasBias=F", naive=False, hasBias=False)
        check_stability("stable square nIn=nOut=6", naive=True, hasBias=False, nInput=6, nOutput=6)
        print("PiLayer tests passed.")

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
            self.noise = torch.randn(
                self.weight.shape,
                device=TheDevice.get(),
                dtype=self.weight.dtype,
            )

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
        import matplotlib.pyplot as plt
        n_points = 100
        data = torch.randn(n_points, 2, device=TheDevice.get())
        data[:, 0] *= 1.5

        layer = DecisionBoundaryLayer(nInput=2, nOutput=1, learning_rate=0.01)
        for _ in range(1000):
            idx = torch.randint(0, n_points, (1,))
            x = data[idx].squeeze()
            layer.update(x)

        w = layer.weight.detach().cpu().numpy()
        w_neg = -w

        data_np = data.cpu().numpy()
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
class AttentionLayer(Layer):
    """Unified attention layer with three modes.

    type="symmetric"   — Hopfield-like: scores = A^T @ A (positive semi-definite).
                         Attends across feature channels.
    type="asymmetric"  — Channel attention: scores = Q^T @ K.
                         Attends across feature channels.
    type="transformer" — Standard multi-head attention over the object/token axis.
                         Q K^T / sqrt(d) with multi-head splitting.

    All modes require 3D input [batch, nObj, dim].
    """
    def __init__(self, nInput, nOutput, nHidden=None, type="asymmetric", nHeads=1):
        super(AttentionLayer, self).__init__(nInput, nOutput)
        self.nHidden = nOutput if not nHidden else nHidden
        self.type = type
        self.mask = None
        self.beta = 10
        self.reversible = False
        self.nHeads = nHeads

        if self.type == "symmetric":
            self.A = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        else:
            self.Q = LinearLayer(self.nInput, self.nHidden)
            self.K = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        self.Out = LinearLayer(self.nHidden, self.nOutput)

        # Transformer-specific: multi-head geometry
        if self.type == "transformer":
            if nHeads < 1:
                raise ValueError(f"nHeads must be >= 1, got {nHeads}")
            if self.nHidden % self.nHeads != 0:
                raise ValueError(
                    f"nHidden ({self.nHidden}) must be divisible by nHeads ({self.nHeads})")
            self.headDim = self.nHidden // self.nHeads
            self.scale = self.headDim ** -0.5

    def set_mask(self, mask: Optional[torch.Tensor]):
        """Set an optional attention mask (used by all types)."""
        self.mask = mask

    # --- Transformer helpers (multi-head) ---

    def _reshape_heads(self, x):
        batch, n_obj, _ = x.shape
        x = x.view(batch, n_obj, self.nHeads, self.headDim)
        return x.transpose(1, 2)

    def _normalize_mask(self, mask, batch, n_obj):
        if mask is None:
            return None
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.dim() == 2:
            if list(mask.shape) != [batch, n_obj]:
                raise ValueError(
                    f"2D mask must have shape {[batch, n_obj]}, got {list(mask.shape)}")
            mask = mask[:, None, None, :].expand(-1, self.nHeads, n_obj, -1)
        elif mask.dim() == 3:
            if list(mask.shape) != [batch, n_obj, n_obj]:
                raise ValueError(
                    f"3D mask must have shape {[batch, n_obj, n_obj]}, got {list(mask.shape)}")
            mask = mask[:, None, :, :].expand(-1, self.nHeads, -1, -1)
        elif mask.dim() == 4:
            if mask.shape[0] != batch or mask.shape[-2:] != (n_obj, n_obj):
                raise ValueError(
                    f"4D mask must end with {[n_obj, n_obj]} and batch {batch}, "
                    f"got {list(mask.shape)}")
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.nHeads, -1, -1)
            elif mask.shape[1] != self.nHeads:
                raise ValueError(
                    f"4D mask head dimension must be 1 or nHeads ({self.nHeads}), "
                    f"got {mask.shape[1]}")
        else:
            raise ValueError(f"Unsupported mask rank {mask.dim()}; expected 2, 3, or 4")
        return mask

    # --- Forward dispatch ---

    def forward(self, x):
        assert x.ndim == 3, f"AttentionLayer expects 3D input [B, N, D], got {list(x.shape)}"
        if self.type == "transformer":
            return self._forward_transformer(x)
        elif self.type == "symmetric":
            return self._forward_symmetric(x)
        else:
            return self._forward_asymmetric(x)

    def _forward_symmetric(self, x):
        a2     = self.A(x)
        value  = x if self.nHidden == self.nInput else self.V(x)
        scores = torch.matmul(a2.transpose(-2, -1), a2) / (self.nInput ** 0.5)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(self.beta * scores, dim=-1) if not self.reversible else scores
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    def _forward_asymmetric(self, x):
        query  = self.Q(x)
        key    = self.K(x)
        value  = x if self.nHidden == self.nInput else self.V(x)
        scores = torch.matmul(query.transpose(-2, -1), key) / (self.nInput ** 0.5)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(self.beta * scores, dim=-1) if not self.reversible else scores
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    def _forward_transformer(self, x):
        batch, n_obj, _ = x.shape
        query = self._reshape_heads(self.Q(x))
        key   = self._reshape_heads(self.K(x))
        value = self._reshape_heads(self.V(x))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = self._normalize_mask(self.mask, batch, n_obj)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch, n_obj, self.nHidden)
        output = self.Out(output)
        return output

    def reverse(self, y, bias=None, temp=None):
        """Attention is not analytically invertible; keep reverse as identity."""
        return super().reverse(y, bias=bias, temp=temp)

    @staticmethod
    def test():
        # Test all three types with 3D input
        for atype in ["symmetric", "asymmetric", "transformer"]:
            kwargs = {"nInput": 8, "nOutput": 8, "type": atype}
            if atype == "transformer":
                kwargs["nHeads"] = 2
            layer = AttentionLayer(**kwargs)
            x = torch.randn(4, 5, 8, device=TheDevice.get())
            y = layer(x)
            assert list(y.shape) == [4, 5, 8], f"type={atype}: expected [4,5,8], got {list(y.shape)}"
        # Test nInput != nOutput
        layer = AttentionLayer(nInput=6, nOutput=3, nHidden=7, type="asymmetric")
        x = torch.randn(4, 5, 6, device=TheDevice.get())
        y = layer(x)
        assert list(y.shape) == [4, 5, 3], f"asymmetric nIn!=nOut: expected [4,5,3], got {list(y.shape)}"
class AssociationLayer(Layer):
    """Cross-symbol associative memory for bidirectional pattern completion.

    Given a [B, N] activation vector over N symbol slots, computes an
    association matrix that measures how each symbol relates to every other.
    This implements the EQUALS rule: when symbol A is symbol B, they have a
    high cross-association score.

    type="symmetric"  — Hopfield-like: scores = A(x)^T @ A(x), positive
                        semi-definite.  Associations are symmetric (A≡B ↔ B≡A).
    type="hopfield"   — Modern Hopfield / softmax retrieval: projects x into
                        queries and keys, softmax attention across symbols,
                        returns the retrieved pattern.

    Input:  [B, N] symbol activations.
    Output: [B, N] associated activations (cross-symbol pattern completion).
    """

    def __init__(self, nInput, nOutput=None, nHidden=None,
                 type="symmetric", beta=10.0):
        nOutput = nOutput or nInput
        super().__init__(nInput, nOutput)
        self.nHidden = nHidden or nInput
        self.type = type
        self.beta = beta

        if self.type == "symmetric":
            # Hopfield energy: E = -x^T W x, stored patterns in W = A^T A
            self.A = LinearLayer(self.nInput, self.nHidden)
        elif self.type == "hopfield":
            # Modern Hopfield: separate query/key projections
            self.Q = LinearLayer(self.nInput, self.nHidden)
            self.K = LinearLayer(self.nInput, self.nHidden)
        else:
            raise ValueError(f"AssociationLayer type must be 'symmetric' or 'hopfield', got '{type}'")

        if self.nHidden != self.nOutput:
            self.Out = LinearLayer(self.nHidden, self.nOutput)
        else:
            self.Out = None

        self.layers = [self.A] if self.type == "symmetric" else [self.Q, self.K]
        if self.Out is not None:
            self.layers.append(self.Out)

    def forward(self, x):
        """Compute cross-symbol associations.

        Args:
            x: [B, N] symbol activation vector.

        Returns:
            [B, N] associated activation — pattern-completed via stored associations.
        """
        # Reshape to [B, N, 1] for matmul compatibility
        x3 = x.unsqueeze(-1)  # [B, N, 1]

        if self.type == "symmetric":
            # Project: [B, N, 1] → [B, N, H] via broadcasting
            # A operates on the last dim, so reshape to [B, N, nInput]
            # For 1-dim symbols, we tile to nInput width
            a = self.A.forward(x)               # [B, nHidden]
            # Association scores: outer product in hidden space
            scores = torch.bmm(
                a.unsqueeze(-1),                # [B, nHidden, 1]
                a.unsqueeze(-2)                 # [B, 1, nHidden]
            )                                   # [B, nHidden, nHidden]
            # Apply to original activation via projection
            attn = F.softmax(self.beta * scores, dim=-1)  # [B, H, H]
            # Map back: use A to project x to hidden, attend, project back
            h = a.unsqueeze(-1)                 # [B, H, 1]
            out = torch.bmm(attn, h).squeeze(-1)  # [B, H]

        elif self.type == "hopfield":
            q = self.Q.forward(x)               # [B, nHidden]
            k = self.K.forward(x)               # [B, nHidden]
            # Similarity in hidden space → association strength
            scores = q * k                      # [B, nHidden] element-wise
            out = F.softmax(self.beta * scores, dim=-1) * k  # [B, nHidden]

        if self.Out is not None:
            out = self.Out.forward(out)          # [B, nOutput]
        return out

    @staticmethod
    def test():
        for atype in ["symmetric", "hopfield"]:
            layer = AssociationLayer(nInput=8, type=atype)
            x = torch.randn(4, 8, device=TheDevice.get())
            y = layer(x)
            assert y.shape == (4, 8), f"type={atype}: expected (4,8), got {y.shape}"
        # nInput != nOutput
        layer = AssociationLayer(nInput=6, nOutput=4, nHidden=8, type="symmetric")
        x = torch.randn(2, 6, device=TheDevice.get())
        y = layer(x)
        assert y.shape == (2, 4), f"nIn!=nOut: expected (2,4), got {y.shape}"

class VerbLayer(Layer):
    """Codebook of verb weight matrices for conceptual composition.

    Each VERB in the codebook is a learned [D, D] weight matrix that
    transforms concept vectors bidirectionally (invertible linear map).
    VERB selection is soft: cosine similarity between a query embedding
    and learned codebook keys produces blending weights.

    Transitive (S → C VERB C):
        VP_eff @ C1 → attention added to C2 (forward)
        VP_eff^T @ C2 → attention added to C1 (backward)

    Intransitive (S → C VERB):
        VP_eff @ C1 → self-attention added to C1

    # Future: VP → VERB C prepositional phrases would modify the
    # VERB's attention matrix before application. The prepositional
    # object C would gate or bias the VP weight matrix, changing
    # the verb's semantics contextually. Deferred.
    """
    def __init__(self, nVerbs, nDim, ergodic=False):
        super().__init__(nDim, nDim)
        self.nVerbs = nVerbs
        self.nDim = nDim
        # Codebook keys for soft VERB selection [nVerbs, nDim]
        self.keys = nn.Parameter(torch.randn(nVerbs, nDim))
        # VERB weight matrices — stack of [nDim, nDim] (initialized near-identity)
        self.vp_weights = nn.ParameterList([
            nn.Parameter(torch.eye(nDim) + 0.01 * torch.randn(nDim, nDim))
            for _ in range(nVerbs)
        ])

    def _select_vp(self, query):
        """Soft-select a blended VERB matrix via cosine similarity.

        Args:
            query: [B, D] embedding vector for VERB selection.
        Returns:
            [B, D, D] blended VERB weight matrix.
        """
        sim = F.cosine_similarity(
            query.unsqueeze(1),           # [B, 1, D]
            self.keys.unsqueeze(0),       # [1, V, D]
            dim=-1
        )                                 # [B, V]
        weights = F.softmax(sim * 10.0, dim=-1)  # sharp selection
        vp_stack = torch.stack(list(self.vp_weights))  # [V, D, D]
        vp_eff = torch.einsum('bv, vij -> bij', weights, vp_stack)  # [B, D, D]
        return vp_eff

    def forward_transitive(self, C1, C2, vp_query):
        """Transitive VERB: C1 VERB C2.

        VP_eff @ C1 → attention added to C2 (forward)
        VP_eff^T @ C2 → attention added to C1 (backward)

        Args:
            C1: [B, N, D] subject concepts.
            C2: [B, N, D] object concepts.
            vp_query: [B, D] VERB embedding for codebook lookup.
        Returns:
            (C1', C2') updated concept tensors.
        """
        vp_eff = self._select_vp(vp_query)              # [B, D, D]
        fwd = torch.bmm(C1, vp_eff)                     # [B, N, D]
        bwd = torch.bmm(C2, vp_eff.transpose(1, 2))     # [B, N, D]
        return C1 + bwd, C2 + fwd

    def forward_reflexive(self, C1, vp_query):
        """Intransitive VERB: C1 VERB (self-application).

        Applies VERB as self-attention on C1, producing the post-VERB state.
        Replaces C1 in-place for this iteration.

        # NOTE: In a stateful model, this produces a state-of-affairs at T2
        # which is distinct from the T1 input. The temporal distinction is
        # significant for learning over sequences of events but is not
        # represented in the current single-timestamp encoding.

        Args:
            C1: [B, N, D] concept vectors.
            vp_query: [B, D] VERB embedding for codebook lookup.
        Returns:
            C1': [B, N, D] self-attended concept vectors.
        """
        vp_eff = self._select_vp(vp_query)              # [B, D, D]
        attended = torch.bmm(C1, vp_eff)                 # [B, N, D]
        return C1 + attended

    @staticmethod
    def test():
        device = TheDevice.get()
        B, N, D, V = 4, 8, 16, 6
        layer = VerbLayer(nVerbs=V, nDim=D)

        C1 = torch.randn(B, N, D, device=device)
        C2 = torch.randn(B, N, D, device=device)
        query = torch.randn(B, D, device=device)

        # Soft selection shape
        vp = layer._select_vp(query)
        assert vp.shape == (B, D, D), f"select shape: {vp.shape}"

        # Transitive
        C1_out, C2_out = layer.forward_transitive(C1, C2, query)
        assert C1_out.shape == (B, N, D), f"transitive C1: {C1_out.shape}"
        assert C2_out.shape == (B, N, D), f"transitive C2: {C2_out.shape}"

        # Reflexive
        C1_refl = layer.forward_reflexive(C1, query)
        assert C1_refl.shape == (B, N, D), f"reflexive: {C1_refl.shape}"

        # Gradient flow
        C1_grad = C1.clone().requires_grad_(True)
        q_grad = query.clone().requires_grad_(True)
        out = layer.forward_reflexive(C1_grad, q_grad)
        out.sum().backward()
        assert C1_grad.grad is not None, "no gradient on C1"
        assert q_grad.grad is not None, "no gradient on query"

        print("VerbLayer tests passed.")

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
        import matplotlib.pyplot as plt
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
class Loss(nn.Module):
    """Base class for loss computation.

    Subclasses override compute() to define how prediction and target
    tensors are compared. The default compute() is MSE.
    """
    def compute(self, pred, target):
        """Compute loss between pred and target. Override in subclasses."""
        return nn.functional.mse_loss(pred, target)
class ModelLoss(Loss):
    """Weighted reconstruction loss with separate scales for what/where/when."""

    def __init__(self, reverse_scale=0.5,
                 what_scale=0.7, where_scale=0.2, when_scale=0.1,
                 embedding_scale=0.1,
                 certainty=False, nOutput=2,
                 conceptualOrder=0, symbolicOrder=0,
                 nWhere=None, nWhen=None):
        super().__init__()
        self.reverse_scale = float(reverse_scale or 0.5)
        self.what_scale = float(what_scale or 0.7)
        self.where_scale = float(where_scale or 0.2)
        self.when_scale = float(when_scale or 0.1)
        self.embedding_scale = float(embedding_scale or 0.1)
        self.nWhere = nWhere if nWhere is not None else TheXMLConfig.get("architecture.nWhere")
        self.nWhen = nWhen if nWhen is not None else TheXMLConfig.get("architecture.nWhen")

        if certainty:
            self.output_criterion = CertaintyWeightedCrossEntropy()
        elif nOutput <= 2:
            self.output_criterion = nn.MSELoss()
        elif conceptualOrder > 0 or symbolicOrder > 0:
            self.output_criterion = nn.MSELoss()
        else:
            self.output_criterion = nn.CrossEntropyLoss()

    def output(self, pred, target):
        return self.output_criterion(pred, target)

    def compute(self, pred, target):
        embSize = pred.shape[-1]
        nWhere = self.nWhere
        nWhen = self.nWhen
        nWhat = embSize - nWhere - nWhen

        loss = pred.new_tensor(0.0)
        if nWhat > 0:
            loss = loss + self.what_scale * F.mse_loss(
                pred[..., :nWhat], target[..., :nWhat])
        if nWhere > 0:
            loss = loss + self.where_scale * F.mse_loss(
                pred[..., nWhat:nWhat + nWhere], target[..., nWhat:nWhat + nWhere])
        if nWhen > 0:
            loss = loss + self.when_scale * F.mse_loss(
                pred[..., nWhat + nWhere:], target[..., nWhat + nWhere:])
        return loss

    def forward(self, lossOut, lossIn=None, sbow=None):
        total = lossOut
        if lossIn is not None and not torch.isnan(lossIn):
            rr = self.reverse_scale
            total = (1 - rr) * lossOut + rr * lossIn
        if sbow is not None:
            total = total + self.embedding_scale * sbow
        return total

    def total(self, lossOut, lossIn=None, sbow=None):
        return self(lossOut, lossIn, sbow)
class CertaintyWeightedMAELoss(Loss):
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
class CertaintyWeightedMSELoss(Loss):
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
class CertaintyWeightedCrossEntropy(Loss):
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
        self.output = torch.zeros(sz, device=TheDevice.get())
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
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
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
        self.state = torch.zeros(sz, device=TheDevice.get())

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

# ── Method base class and subclasses ──────────────────────────────

class Method(Layer):
    """Base for grammar operations. Unary or binary.

    Each Method implements a reversible function on concept vectors.
    Binary methods take (left, right) → parent.
    Unary methods take (left) → parent.
    """
    def __init__(self, nDim, binary=False):
        super().__init__(nDim, nDim)
        self.binary = binary

    def forward(self, left, right=None):
        raise NotImplementedError

    def reverse(self, parent, left_hint=None, right_hint=None):
        raise NotImplementedError


class Union(Method):
    """Mereological union: max(left, right)."""
    def __init__(self, nDim):
        super().__init__(nDim, binary=True)

    def forward(self, left, right=None):
        return torch.max(left, right)

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


class Intersection(Method):
    """Mereological intersection: min(left, right)."""
    def __init__(self, nDim):
        super().__init__(nDim, binary=True)

    def forward(self, left, right=None):
        return torch.min(left, right)

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


class Equals(Method):
    """Equality as mutual parthood: equals(a,b) = part(a,b) ∧ part(b,a).

    Since part is min and min is commutative, the forward pass is
    equivalent to min(a, b).  The semantic distinction from Part
    is that equals asserts *both* directions of containment:
    C(A) <= C(B) AND C(B) <= C(A), therefore C(A) = C(B).
    Part asserts only one direction.
    """
    def __init__(self, nDim):
        super().__init__(nDim, binary=True)

    def forward(self, left, right=None):
        # part(a,b) = min(a,b), part(b,a) = min(b,a)
        # equals = part(a,b) ∧ part(b,a) = min(min(a,b), min(b,a)) = min(a,b)
        return torch.min(left, right)

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


class Part(Method):
    """Parthood: min(left, right).

    Symbol A is a part of symbol B to the extent that A's projection
    into conceptual space is contained in B's: C(A) <= C(B) for all
    elements.  When this holds, min(A, B) = A — the part passes
    through unchanged.  Where A exceeds B, min clips to B, enforcing
    containment.
    """
    def __init__(self, nDim):
        super().__init__(nDim, binary=True)

    def forward(self, left, right=None):
        return torch.min(left, right)

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


class Lift(Method):
    """Lift: binary lift_layer(left)*right or unary lift_layer(left)."""
    def __init__(self, nDim):
        super().__init__(nDim, binary=True)
        self.lift_layer = LinearLayer(nDim, nDim)
        self.layers = nn.ModuleList([self.lift_layer])

    def forward(self, left, right=None):
        lifted = self.lift_layer.forward(left)
        if right is not None:
            return lifted * right
        return lifted

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


class Lower(Method):
    """Lower: rank-reducing bottleneck (unary)."""
    def __init__(self, nDim, bottleneck=None):
        super().__init__(nDim, binary=False)
        bottleneck = bottleneck or max(1, nDim // 4)
        self.down = LinearLayer(nDim, bottleneck)
        self.up = LinearLayer(bottleneck, nDim)
        self.layers = nn.ModuleList([self.down, self.up])

    def forward(self, left, right=None):
        return self.up.forward(self.down.forward(left))

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent


class Not(Method):
    """Negation: -left (unary)."""
    def __init__(self, nDim):
        super().__init__(nDim, binary=False)

    def forward(self, left, right=None):
        return -left

    def reverse(self, parent, left_hint=None, right_hint=None):
        return -parent


class Non(Method):
    """Attenuated signal: sigmoid(alpha) * left (unary)."""
    def __init__(self, nDim):
        super().__init__(nDim, binary=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, left, right=None):
        return torch.sigmoid(self.alpha) * left

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent / (torch.sigmoid(self.alpha) + epsilon)


class Swap(Method):
    """Learnable soft permutation of arguments (binary).

    Given two arguments (left, right) and a learned marker, produces a
    soft permutation of all three via a 3x3 doubly-stochastic matrix
    (Sinkhorn normalisation of learnable logits).  The first two rows
    of the permutation matrix select the output ordering; the marker
    row is discarded.

    For the grammar rule ``swap(S, S)``, this rearranges the two S
    constituents with a learned probability, allowing the model to
    discover canonical argument order.
    """
    def __init__(self, nDim, sinkhorn_iters=5):
        super().__init__(nDim, binary=True)
        # Learnable marker vector — context for the permutation
        self.marker = nn.Parameter(torch.randn(nDim) * 0.01)
        # 3x3 logits for soft permutation (rows=output slots, cols=input slots)
        self.logits = nn.Parameter(torch.zeros(3, 3))
        self.sinkhorn_iters = sinkhorn_iters

    def _soft_perm(self):
        """Sinkhorn-normalise logits → doubly-stochastic 3x3 matrix."""
        M = self.logits
        for _ in range(self.sinkhorn_iters):
            M = M - M.logsumexp(dim=-1, keepdim=True)  # row normalise
            M = M - M.logsumexp(dim=-2, keepdim=True)  # col normalise
        return M.exp()

    def forward(self, left, right=None):
        P = self._soft_perm()                          # [3, 3]
        marker = self.marker.to(left.device)
        # Expand marker to match left/right shape
        if left.ndim == 3:
            # [B, N, D]
            m = marker.unsqueeze(0).unsqueeze(0).expand_as(left)
        elif left.ndim == 2:
            # [B, N]
            m = marker[:left.shape[-1]].unsqueeze(0).expand_as(left)
        else:
            m = marker
        if right is None:
            right = left
        # Stack inputs: [3, ...] then apply permutation
        stack = torch.stack([left, right, m], dim=0)    # [3, B, ...]
        # P[i, j] = weight of input j in output slot i
        # einsum over input index j: out[i, B, ...] = sum_j P[i,j] * stack[j, B, ...]
        out = torch.einsum('ij,j...->i...', P, stack)   # [3, B, ...]
        return out[0]                                    # first output slot

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


class IsTrue(Method):
    """Truth evaluation: signed maximum absolute value (unary).

    For each vector in the input, finds the element with the largest
    absolute value and returns that signed value, broadcast back to
    the original shape.  This collapses a distributed representation
    to its strongest signal — the "truth" of the expression.
    """
    def __init__(self, nDim):
        super().__init__(nDim, binary=False)

    def forward(self, left, right=None):
        # Find index of max |value| along last dimension
        abs_vals = left.abs()
        idx = abs_vals.argmax(dim=-1, keepdim=True)      # [..., 1]
        signed_max = left.gather(-1, idx)                 # [..., 1]
        return signed_max.expand_as(left)

    def reverse(self, parent, left_hint=None, right_hint=None):
        return left_hint if left_hint is not None else parent


class Chunk(Method):
    """Input accumulation: max(left, right) — chunks elements together."""
    def __init__(self, nDim):
        super().__init__(nDim, binary=True)

    def forward(self, left, right=None):
        if right is None:
            return left
        return torch.max(left, right)

    def reverse(self, parent, left_hint=None, right_hint=None):
        return parent, parent


# ── Grammar ──────────────────────────────────────────────────────

RuleDef = namedtuple('RuleDef', ['tier', 'canonical', 'arity', 'method_name'])

class Grammar(nn.Module):
    """Hierarchical 3-tier grammar (S, C, P) with rule methods and composition.

    Owns all rule execution (Method instances), rule prediction
    (SyntacticLayer instances), and shift/reduce composition logic.
    Spaces delegate syntax to TheGrammar rather than owning methods.

    Rules are parsed from functional notation in XML. Each rule maps to
    a Method subclass. During encoding, SyntacticLayer chooses which rule
    fires at each shift/reduce step. The <interpretation> lever biases
    union (surface) vs non-union (meaningful) rule choice.
    """

    # Default rule catalog — used when no XML grammar is configured.
    _DEFAULT_RULES = [
        RuleDef('START', 'START → true(S) EOF',   1, 'true'),
        RuleDef('S',     'S → swap(S, S)',         2, 'swap'),
        RuleDef('S',     'S → equals(S, S)',       2, 'equals'),
        RuleDef('S',     'S → part(S, S)',         2, 'part'),
        RuleDef('S',     'S → C',                  1, None),      # transition
        RuleDef('C',     'C → union(C, C)',        2, 'union'),
        RuleDef('C',     'C → intersection(C, C)', 2, 'intersection'),
        RuleDef('C',     'C → lower(C)',           1, 'lower'),
        RuleDef('C',     'C → lift(C, C)',         2, 'lift'),
        RuleDef('C',     'C → lift(C)',            1, 'lift'),
        RuleDef('C',     'C → not(C)',             1, 'not'),
        RuleDef('C',     'C → non(C)',             1, 'non'),
        RuleDef('C',     'C → P',                  1, None),      # transition
        RuleDef('P',     'P → chunk(I, P)',         2, 'chunk'),
        RuleDef('P',     'P → ε',                  0, None),      # terminal
    ]

    # Transition rule IDs (S→C = 4, C→P = 12)
    _TRANSITION_IDS = {4, 12}

    def __init__(self, lazy_init=True):
        super().__init__()
        self.rules = list(self._DEFAULT_RULES)
        self._configured = False
        self._lazy_init = lazy_init
        self.interpretation = 0.5
        # Placeholders — populated by init_layers()
        self.s_methods = None
        self.c_methods = None
        self.p_methods = None
        self.s_syntactic_layer = None
        self.c_syntactic_layer = None
        self.p_syntactic_layer = None
        self.verb_layer = None
        self._layers_initialized = False
        # S/R stacks (not persistent state — reset per derivation)
        self._s_stack = []
        self._s_where_stack = []
        self._s_words = []
        self._c_stack = []
        self._c_act_stack = []
        self._c_words = []
        self._p_words = []
        # Word encoders — registered by spaces after init_layers
        self._word_encoders = {}

    # ── Rule catalog ──────────────────────────────────────────────────

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx].canonical

    def arity(self, rule_id):
        return self.rules[rule_id].arity

    def method_name(self, rule_id):
        return self.rules[rule_id].method_name

    def tier(self, rule_id):
        return self.rules[rule_id].tier

    def binary_rules(self):
        return [i for i in range(len(self.rules)) if self.rules[i].arity == 2]

    # ── Configuration from XML ────────────────────────────────────────

    def configure(self, grammar_dict):
        """Configure from parsed XML grammar inside <architecture>.

        Parses functional notation: "equals(C, C)", "union(C, C)", "swap(S, S)",
        "I P", "ε", "true(S) EOF", "P", "C".
        """
        self.rules = []
        self._configured = True

        # Parse interpretation lever
        if isinstance(grammar_dict, dict):
            interp = grammar_dict.get('interpretation', 0.5)
            self.interpretation = float(interp)

        for lhs in ('START', 'S', 'C', 'P'):
            raw = grammar_dict.get(lhs, [])
            if isinstance(raw, str):
                raw = [raw]
            for rhs_text in raw:
                rhs = rhs_text.strip()
                rule_def = self._parse_rule(lhs, rhs)
                self.rules.append(rule_def)

    def _parse_rule(self, lhs, rhs):
        """Parse a single rule RHS into a RuleDef."""
        # Functional notation: "method(args)"
        if '(' in rhs:
            func_name = rhs[:rhs.index('(')]
            args_str = rhs[rhs.index('(') + 1:rhs.rindex(')')]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            suffix = rhs[rhs.rindex(')') + 1:].strip()  # e.g. "EOF"
            arity = len(args)
            canonical = f"{lhs} → {rhs}"
            return RuleDef(lhs, canonical, arity, func_name)

        # Terminal: "ε"
        if rhs == 'ε':
            return RuleDef(lhs, f"{lhs} → ε", 0, None)

        # Recursive: "I P"
        if rhs == 'I P':
            return RuleDef(lhs, f"{lhs} → I P", 1, None)

        # Transition: single nonterminal like "C" or "P"
        if rhs in ('S', 'C', 'P'):
            return RuleDef(lhs, f"{lhs} → {rhs}", 1, None)

        raise ValueError(f"Cannot parse grammar rule: {lhs} → {rhs}")

    def _ensure_configured(self):
        """Lazy init: read <grammar> from TheXMLConfig on first access.

        Checks <mentalModel><grammar> first, then <basicModel><grammar>,
        then legacy <architecture><grammar> for backward compatibility.
        """
        if self._configured or not self._lazy_init:
            return
        from util import TheXMLConfig
        cfg = None
        for path in ("mentalModel.grammar",
                      "architecture.grammar"):
            try:
                candidate = TheXMLConfig.get(path)
                if isinstance(candidate, dict):
                    cfg = candidate
                    break
            except (KeyError, AttributeError):
                continue
        if cfg is not None:
            self.configure(cfg)

    # ── Rule queries ──────────────────────────────────────────────────

    def symbolic(self):
        """Rule IDs for S-tier operations."""
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'S' and r.method_name is not None]

    def conceptual(self):
        """Rule IDs for C-tier operations (non-transition)."""
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'C' and r.method_name is not None]

    def perceptual(self):
        """Rule IDs for P-tier operations."""
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'P']

    # ── Transition helpers ────────────────────────────────────────────

    def symbolic_transition(self):
        """Return the S→C transition rule ID."""
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.tier == 'S' and r.method_name is None and r.arity == 1:
                return i
        return None

    def conceptual_transition(self):
        """Return the C→P transition rule ID."""
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.tier == 'C' and r.method_name is None and r.arity == 1:
                return i
        return None

    # ── Layer initialization ──────────────────────────────────────────

    def init_layers(self, concept_dim, symbol_dim,
                    n_concept_slots, n_symbol_slots, n_percept_slots):
        """Create per-tier methods and SyntacticLayers. Called once during
        model construction.

        Args:
            concept_dim:    ConceptualSpace embedding width (nDim).
            symbol_dim:     SymbolicSpace activation width (nSymbols).
            n_concept_slots: number of concept slots.
            n_symbol_slots:  number of symbol slots.
            n_percept_slots: number of percept slots.
        """
        if self._layers_initialized:
            return
        self._ensure_configured()

        # S-tier methods (dim = symbol_dim)
        self.s_methods = nn.ModuleDict({
            'swap':   Swap(symbol_dim),
            'equals': Equals(symbol_dim),
            'part':   Part(symbol_dim),
            'true':   IsTrue(symbol_dim),
        })

        # C-tier methods (dim = concept_dim)
        self.c_methods = nn.ModuleDict({
            'union':        Union(concept_dim),
            'intersection': Intersection(concept_dim),
            'lower':        Lower(concept_dim),
            'lift':         Lift(concept_dim),
            'not':          Not(concept_dim),
            'non':          Non(concept_dim),
        })

        # P-tier methods (chunking = input accumulation)
        self.p_methods = nn.ModuleDict({
            'chunk': Chunk(n_percept_slots),
        })

        # SyntacticLayers
        self.s_syntactic_layer = SyntacticLayer(
            nInput=n_symbol_slots, nOutput=n_symbol_slots,
            rules=self.symbolic(),
            transition_rule=self.symbolic_transition(),
            max_depth=max(n_symbol_slots - 1, 1),
            hidden_dim=min(256, max(64, n_symbol_slots * 4)),
            grammar=self,
        )
        self.c_syntactic_layer = SyntacticLayer(
            nInput=n_concept_slots, nOutput=n_concept_slots,
            rules=self.conceptual(),
            transition_rule=self.conceptual_transition(),
            max_depth=max(n_concept_slots - 1, 1),
            hidden_dim=min(256, max(64, n_concept_slots * 4)),
            grammar=self,
        )
        self.p_syntactic_layer = SyntacticLayer(
            nInput=n_percept_slots, nOutput=n_percept_slots,
            rules=self.perceptual(),
            transition_rule=None,
            max_depth=max(n_percept_slots - 1, 1),
            hidden_dim=min(256, max(64, n_percept_slots * 4)),
            grammar=self,
        )

        # Legacy verb layer
        self.verb_layer = VerbLayer(16, concept_dim)
        self._layers_initialized = True

    def register_word_encoder(self, tier, word_encoding):
        """Register a space's word encoder for word tuple generation."""
        self._word_encoders[tier] = word_encoding

    # ── Tier dispatch helpers ─────────────────────────────────────────

    def _tier_methods(self, tier):
        if tier == 'S':
            return self.s_methods
        elif tier == 'C':
            return self.c_methods
        elif tier == 'P':
            return self.p_methods
        return None

    def _tier_syntactic_layer(self, tier):
        if tier == 'S':
            return self.s_syntactic_layer
        elif tier == 'C':
            return self.c_syntactic_layer
        elif tier == 'P':
            return self.p_syntactic_layer
        return None

    # ── Rule execution ────────────────────────────────────────────────

    def project(self, tier, rule_id, left, right=None):
        """Execute a grammar rule on left (and right) operands.

        Dispatches to per-tier methods by method_name.
        Falls back to verb_layer for 'verb' method on C-tier (legacy).
        Transition rules pass through.
        """
        method_name = self.rules[rule_id].method_name
        methods = self._tier_methods(tier)
        if method_name and methods and method_name in methods:
            method = methods[method_name]
            if method.binary:
                if right is not None:
                    return method.forward(left, right)
                return left
            return method.forward(left)

        # Legacy: VERB on C-tier
        if tier == 'C' and method_name == 'verb' and self.verb_layer is not None:
            B, N, D = left.shape
            vp_query = left.mean(dim=1)
            if right is not None:
                C1_out, _ = self.verb_layer.forward_transitive(left, right, vp_query)
                return C1_out
            return self.verb_layer.forward_reflexive(left, vp_query)

        # Transition or unknown — pass through
        return left

    # ── S/R stacks ────────────────────────────────────────────────────

    def resetStack(self, tier):
        """Clear per-tier stacks for a new derivation."""
        if tier == 'S':
            self._s_stack = []
            self._s_where_stack = []
            self._s_words = []
        elif tier == 'C':
            self._c_stack = []
            self._c_act_stack = []
            self._c_words = []
        elif tier == 'P':
            self._p_words = []

    def _rewrite_rule_index(self):
        """Local index of the REWRITE/swap rule in S-tier rule set."""
        if self.s_syntactic_layer is None:
            return None
        for i, rid in enumerate(self.s_syntactic_layer.all_rules):
            if "swap" in self[rid].lower():
                return i
        return None

    def _encode_words(self, tier, activation, best_rule_id):
        """Generate word tuples from active positions."""
        words = []
        encoder = self._word_encoders.get(tier)
        if encoder is None:
            return words
        B = activation.shape[0]
        for b in range(B):
            active = (activation[b].abs() > 1e-6).nonzero(as_tuple=True)[0]
            for v in active:
                words.append(encoder.encode(b, v.item(), best_rule_id))
        return words

    def forward(self, tier, activation, vectors_or_where=None):
        """Unified shift/reduce for any tier.

        Args:
            tier: 'S', 'C', or 'P'
            activation: [B, N] activation vector
            vectors_or_where: [B, N, D] vectors (C-tier) or
                              [B, N, nWhere] where (S-tier), or None
        Returns:
            dict with transition, composed, words
        """
        sl = self._tier_syntactic_layer(tier)
        if sl is None:
            return {"transition": False, "composed": activation, "words": []}

        if tier == 'S':
            return self._write_symbolic(activation, vectors_or_where, sl)
        elif tier == 'C':
            return self._write_conceptual(activation, vectors_or_where, sl)
        elif tier == 'P':
            return self._write_perceptual(activation, vectors_or_where, sl)
        return {"transition": False, "composed": activation, "words": []}

    def _write_symbolic(self, symbol_act, where, sl):
        """S-tier shift/reduce with optional where-swap."""
        # SHIFT
        self._s_stack.append(symbol_act)
        if where is not None:
            self._s_where_stack.append(where)

        all_rules = sl.all_rules
        # PREDICT
        head = self._s_stack[-1]
        out = sl.forward(head)
        rule_probs = out["rule_probs"][:, 0, :]
        best_local = rule_probs.argmax(dim=-1)[0].item()
        best_rule_id = all_rules[best_local]

        # REDUCE (soft gate)
        B_act = symbol_act.shape[0]
        p_binary = torch.zeros(B_act, 1, device=rule_probs.device)
        for local_idx, rule_id in enumerate(all_rules):
            if self.arity(rule_id) == 2:
                p_binary = p_binary + rule_probs[:, local_idx:local_idx + 1]

        arity = self.arity(best_rule_id)
        has_where = where is not None and len(self._s_where_stack) > 0
        rewrite_idx = self._rewrite_rule_index() if has_where else None

        if arity == 2 and len(self._s_stack) >= 2:
            shift_result = self._s_stack[-1]
            right = self._s_stack.pop()
            left = self._s_stack.pop()

            results = []
            for local_idx, rule_id in enumerate(all_rules):
                a = self.arity(rule_id)
                if a == 2:
                    result = self.project('S', rule_id, left, right)
                elif a == 1:
                    result = self.project('S', rule_id, left)
                else:
                    result = self.project('S', rule_id, left)
                results.append(result)

            results = torch.stack(results, dim=1)
            reduce_result = (rule_probs.unsqueeze(-1) * results).sum(dim=1)
            composed = p_binary * reduce_result + (1.0 - p_binary) * shift_result
            self._s_stack.append(composed)

            # REWRITE where-swap
            if has_where and rewrite_idx is not None and len(self._s_where_stack) >= 2:
                shift_where = self._s_where_stack[-1]
                right_where = self._s_where_stack.pop()
                left_where = self._s_where_stack.pop()
                rewrite_prob = rule_probs[:, rewrite_idx]
                non_rewrite_prob = 1.0 - rewrite_prob
                reduce_where = (non_rewrite_prob[:, None, None] * left_where
                                + rewrite_prob[:, None, None] * right_where)
                composed_where = (p_binary[:, :, None] * reduce_where
                                  + (1.0 - p_binary[:, :, None]) * shift_where)
                self._s_where_stack.append(composed_where)

        elif arity == 1 and len(self._s_stack) >= 1:
            shift_result = self._s_stack[-1]
            operand = self._s_stack.pop()

            results = []
            for local_idx, rule_id in enumerate(all_rules):
                a = self.arity(rule_id)
                if a == 1:
                    result = self.project('S', rule_id, operand)
                else:
                    result = self.project('S', rule_id, operand)
                results.append(result)

            results = torch.stack(results, dim=1)
            reduce_result = (rule_probs.unsqueeze(-1) * results).sum(dim=1)
            p_unary = 1.0 - p_binary
            composed = p_unary * reduce_result + p_binary * shift_result
            self._s_stack.append(composed)

        # TRANSITION
        transition = self.rules[best_rule_id].method_name is None and self.arity(best_rule_id) == 1

        # WORDS
        words = self._encode_words('S', symbol_act, best_rule_id)
        self._s_words.extend(words)

        return {
            "transition": transition,
            "composed": self._s_stack[-1],
            "words": words,
        }

    def _write_conceptual(self, activation, vectors, sl):
        """C-tier shift/reduce on [B, N, D] vectors."""
        # SHIFT
        self._c_stack.append(vectors)
        self._c_act_stack.append(activation)

        all_rules = sl.all_rules
        # PREDICT
        head_act = self._c_act_stack[-1]
        out = sl.forward(head_act)
        rule_probs = out["rule_probs"][:, 0, :]
        best_local = rule_probs.argmax(dim=-1)[0].item()
        best_rule_id = all_rules[best_local]

        # REDUCE (soft gate)
        B_act = activation.shape[0]
        p_binary = torch.zeros(B_act, 1, device=rule_probs.device)
        for local_idx, rule_id in enumerate(all_rules):
            if self.arity(rule_id) == 2:
                p_binary = p_binary + rule_probs[:, local_idx:local_idx + 1]

        arity = self.arity(best_rule_id)

        if arity == 2 and len(self._c_stack) >= 2:
            shift_result = self._c_stack[-1]
            shift_act = self._c_act_stack[-1]
            right = self._c_stack.pop()
            left = self._c_stack.pop()
            right_act = self._c_act_stack.pop()
            left_act = self._c_act_stack.pop()

            results = []
            for local_idx, rule_id in enumerate(all_rules):
                a = self.arity(rule_id)
                if a == 2:
                    result = self.project('C', rule_id, left, right)
                elif a == 1:
                    result = self.project('C', rule_id, left)
                else:
                    result = self.project('C', rule_id, left)
                results.append(result)

            results = torch.stack(results, dim=1)
            probs = rule_probs.unsqueeze(-1).unsqueeze(-1)
            reduce_result = (probs * results).sum(dim=1)
            p_b = p_binary.unsqueeze(-1)
            composed = p_b * reduce_result + (1.0 - p_b) * shift_result
            self._c_stack.append(composed)

            reduce_act = reduce_result.norm(dim=-1)
            composed_act = p_binary * reduce_act + (1.0 - p_binary) * shift_act
            self._c_act_stack.append(composed_act)

        elif arity == 1 and len(self._c_stack) >= 1:
            shift_result = self._c_stack[-1]
            shift_act = self._c_act_stack[-1]
            operand = self._c_stack.pop()
            operand_act = self._c_act_stack.pop()

            results = []
            for local_idx, rule_id in enumerate(all_rules):
                a = self.arity(rule_id)
                if a == 1:
                    result = self.project('C', rule_id, operand)
                else:
                    result = self.project('C', rule_id, operand)
                results.append(result)

            results = torch.stack(results, dim=1)
            probs = rule_probs.unsqueeze(-1).unsqueeze(-1)
            reduce_result = (probs * results).sum(dim=1)
            p_unary = 1.0 - p_binary
            p_u = p_unary.unsqueeze(-1)
            composed = p_u * reduce_result + (1.0 - p_u) * shift_result
            self._c_stack.append(composed)

            reduce_act = reduce_result.norm(dim=-1)
            composed_act = p_unary * reduce_act + p_binary * shift_act
            self._c_act_stack.append(composed_act)

        # TRANSITION
        transition = self.rules[best_rule_id].method_name is None and self.arity(best_rule_id) == 1

        # WORDS
        words = self._encode_words('C', activation, best_rule_id)
        self._c_words.extend(words)

        return {
            "transition": transition,
            "composed": self._c_stack[-1],
            "words": words,
        }

    def _write_perceptual(self, activation, vectors, sl):
        """P-tier shift/reduce — terminal rule, accumulation via chunk."""
        # PREDICT
        out = sl.forward(activation)
        words = out.get("words", [])
        self._p_words.extend(words)

        # Apply chunk (accumulation) — same as union/max
        composed = vectors
        if vectors is not None:
            composed = self.project('P', 13, vectors)  # rule 13 = chunk

        return {
            "transition": False,
            "composed": composed,
            "words": words,
        }

    def reverse(self, tier, words, batch_size=1):
        """Reverse S/R: reconstruct activation from word tuples."""
        sl = self._tier_syntactic_layer(tier)
        if sl is None:
            return None
        nVectors = sl.nInput
        return sl.reverse(words, nVectors, batch_size)

    # ── Batch composition (non-S/R) ──────────────────────────────────

    def composeSyntax(self, tier, activation, vectors_or_where=None):
        """Predict rules and execute soft-weighted composition (batch mode).

        Args:
            tier: 'S' or 'C'
            activation: [B, N] activation for rule prediction
            vectors_or_where: [B, N, D] vectors (C-tier) or
                              [B, N, nWhere] where (S-tier), or None
        Returns:
            dict with composed, words, rule_probs, etc.
        """
        sl = self._tier_syntactic_layer(tier)
        if sl is None:
            out = {"composed": activation if vectors_or_where is None else vectors_or_where, "words": []}
            if tier == 'S' and vectors_or_where is not None:
                out["composed_where"] = vectors_or_where
            return out

        out = sl.forward(activation)
        rule_probs = out["rule_probs"]
        all_rules = sl.all_rules

        B, N = activation.shape
        active_positions = sl._active_positions(activation)
        max_leaves = max((len(p) for p in active_positions), default=0)

        if tier == 'C':
            return self._compose_conceptual(out, activation, vectors_or_where,
                                            rule_probs, all_rules, active_positions, max_leaves)
        elif tier == 'S':
            return self._compose_symbolic(out, activation, vectors_or_where,
                                          rule_probs, all_rules, active_positions, max_leaves)
        return out

    def _compose_conceptual(self, out, activation, vectors,
                            rule_probs, all_rules, active_positions, max_leaves):
        """Batch composition for C-tier on [B, N, D] vectors."""
        B, N = activation.shape
        if max_leaves == 0 or vectors is None:
            out["composed"] = vectors
            return out

        D = vectors.shape[-1] if vectors.ndim == 3 else 1

        masks = torch.zeros(B, max_leaves, N, device=vectors.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                masks[b, i, pos] = 1.0
        leaf_vecs = masks.unsqueeze(-1) * vectors.unsqueeze(1)

        composed = leaf_vecs[:, 0, :, :]
        sl = self.c_syntactic_layer
        max_depth = sl.max_depth

        for d in range(min(max_depth, max(max_leaves - 1, 1))):
            if d + 1 >= max_leaves:
                break
            left = composed
            right = leaf_vecs[:, d + 1, :, :]

            results = []
            for local_idx, rule_id in enumerate(all_rules):
                a = self.arity(rule_id)
                if a == 2:
                    result = self.project('C', rule_id, left, right)
                elif a == 1:
                    result = self.project('C', rule_id, left)
                else:
                    result = self.project('C', rule_id, left)
                results.append(result)

            results = torch.stack(results, dim=1)
            probs_d = rule_probs[:, d, :]
            probs_d = probs_d.unsqueeze(-1).unsqueeze(-1)
            composed = (probs_d * results).sum(dim=1)

        out["composed"] = composed
        return out

    def _compose_symbolic(self, out, activation, where,
                          rule_probs, all_rules, active_positions, max_leaves):
        """Batch composition for S-tier on [B, N] activations."""
        B, N = activation.shape
        if max_leaves == 0:
            out["composed"] = activation
            if where is not None:
                out["composed_where"] = where
            return out

        masks = torch.zeros(B, max_leaves, N, device=activation.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                masks[b, i, pos] = 1.0
        leaf_acts = masks * activation.unsqueeze(1)

        has_where = where is not None
        if has_where:
            leaf_wheres = masks.unsqueeze(-1) * where.unsqueeze(1)

        composed = leaf_acts[:, 0, :]
        if has_where:
            composed_where = leaf_wheres[:, 0, :, :]
        sl = self.s_syntactic_layer
        max_depth = sl.max_depth

        rewrite_idx = self._rewrite_rule_index() if has_where else None

        for d in range(min(max_depth, max(max_leaves - 1, 1))):
            if d + 1 >= max_leaves:
                break
            left = composed
            right = leaf_acts[:, d + 1, :]

            results = []
            for local_idx, rule_id in enumerate(all_rules):
                a = self.arity(rule_id)
                if a == 2:
                    result = self.project('S', rule_id, left, right)
                elif a == 1:
                    result = self.project('S', rule_id, left)
                else:
                    result = self.project('S', rule_id, left)
                results.append(result)

            results = torch.stack(results, dim=1)
            probs_d = rule_probs[:, d, :]
            composed = (probs_d.unsqueeze(-1) * results).sum(dim=1)

            if has_where and rewrite_idx is not None:
                left_where = composed_where
                right_where = leaf_wheres[:, d + 1, :, :]
                rewrite_prob = probs_d[:, rewrite_idx]
                non_rewrite_prob = 1.0 - rewrite_prob
                composed_where = (non_rewrite_prob[:, None, None] * left_where
                                  + rewrite_prob[:, None, None] * right_where)

        out["composed"] = composed
        if has_where:
            out["composed_where"] = composed_where
        return out

class SyntacticLayer(Layer):
    """Per-space rule prediction layer for the recursive grammar.

    Each instance handles a subset of the Grammar's rules (one cognitive
    space's rules).  Uses a weight-tied recursive architecture with depth
    embeddings.

    **This layer only predicts rules and generates word tuples.**  It does
    not execute operations on representations — that is done by the owning
    space's ``projectXxx()`` method, which knows the native representation
    type (activations, vectors, etc.).

    Args:
        nInput:    activation width (number of symbol/concept/percept slots).
        nOutput:   same as nInput.
        rules:     list of global Grammar rule IDs this layer handles
                   (e.g. [1,2,3,4,5] for the symbolic space).
        transition_rule: optional global rule ID for the transition rule
                   (e.g. 6 for S→C).  Included in prediction but signals
                   hand-off to the next space.
        max_depth: maximum derivation depth.
        hidden_dim: width of the shared derivation hidden state.
        grammar:   Grammar instance.
        tau:       Gumbel-softmax temperature.
    """

    def __init__(self, nInput, nOutput, rules, transition_rule=None,
                 max_depth=12, hidden_dim=256, grammar=None, tau=1.0):
        super().__init__(nInput, nOutput)
        # Store grammar as non-Module attribute to avoid circular nn.Module
        # reference (Grammar owns SyntacticLayers, SyntacticLayers reference
        # Grammar). Using object.__setattr__ bypasses nn.Module.__setattr__
        # which would register it as a submodule.
        object.__setattr__(self, 'grammar', grammar or Grammar())
        self.rules           = list(rules)
        self.transition_rule = transition_rule
        # Build the full set of rule IDs this layer predicts over
        self.all_rules = list(rules)
        if transition_rule is not None and transition_rule not in self.all_rules:
            self.all_rules.append(transition_rule)
        self.num_rules  = len(self.all_rules)
        # Map from local index → global rule ID
        self.rule_index = {rid: i for i, rid in enumerate(self.all_rules)}
        self.max_depth  = max_depth
        self.hidden_dim = hidden_dim
        self.tau        = tau

        # Rule prediction network (weight-tied across depths)
        self.input_proj       = LinearLayer(nInput, hidden_dim)
        self.derivation_layer = LinearLayer(hidden_dim, hidden_dim)
        self.rule_head        = LinearLayer(hidden_dim, self.num_rules)
        self.depth_embed      = nn.Embedding(max_depth, hidden_dim)
        self.activation_fn    = nn.GELU()

        # Register child layers for ergodic dispatch
        self.layers = [self.input_proj, self.derivation_layer, self.rule_head]

    # ── forward: predict rules ────────────────────────────────────

    def forward(self, x):
        """Predict rule distributions and build word tuples.

        Args:
            x: [B, N] activation vector from the space's subspace.

        Returns dict:
            rule_logits:     [B, max_depth, num_rules]  (local indices)
            rule_probs:      [B, max_depth, num_rules]
            predicted_rules: [B, max_depth]             (global rule IDs)
            words:           list of (batch, vector, rule) tuples
        """
        B, N = x.shape

        h = self.input_proj.forward(x)
        h = self.activation_fn(h)

        depth_ids = torch.arange(self.max_depth, device=x.device)
        depth_vecs = self.depth_embed(depth_ids)

        all_logits = []
        all_probs  = []

        for d in range(self.max_depth):
            h = h + depth_vecs[d]
            h = self.derivation_layer.forward(h)
            h = self.activation_fn(h)
            logits = self.rule_head.forward(h)

            if self.training:
                probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            else:
                probs = F.softmax(logits, dim=-1)

            all_logits.append(logits)
            all_probs.append(probs)

        rule_logits = torch.stack(all_logits, dim=1)
        rule_probs  = torch.stack(all_probs, dim=1)

        # Map local argmax to global rule IDs
        local_predicted = rule_logits.argmax(dim=-1)
        global_predicted = torch.tensor(
            [[self.all_rules[local_predicted[b, d].item()]
              for d in range(self.max_depth)]
             for b in range(B)],
            device=x.device, dtype=torch.long
        )

        active_positions = self._active_positions(x)
        words = self._generate_derivation(global_predicted, active_positions)

        return {
            "rule_logits":     rule_logits,
            "rule_probs":      rule_probs,
            "predicted_rules": global_predicted,
            "words":           words,
        }

    # ── helpers ────────────────────────────────────────────────────

    def _active_positions(self, x):
        """Extract per-batch lists of active (nonzero) positions."""
        B = x.shape[0]
        positions = []
        for b in range(B):
            active = torch.nonzero(x[b], as_tuple=False).squeeze(-1)
            positions.append(active.tolist())
        return positions

    def _generate_derivation(self, predicted_rules, active_positions):
        """Build word tuples from predicted rules and active positions."""
        B = predicted_rules.shape[0]
        all_words = []
        for b in range(B):
            rules     = predicted_rules[b].tolist()
            positions = active_positions[b]
            n = len(positions)
            if n == 0:
                continue
            if n == 1:
                terminal = self._find_terminal_rule()
                all_words.append((b, positions[0], terminal))
                continue
            pos_idx = 0
            for rule_id in rules:
                if pos_idx >= n - 1:
                    break
                arity = self.grammar.arity(rule_id)
                if arity != 2:
                    binary = [r for r in self.rules if self.grammar.arity(r) == 2]
                    rule_id = binary[0] if binary else rule_id
                all_words.append((b, positions[pos_idx], rule_id))
                pos_idx += 1
            terminal = self._find_terminal_rule()
            all_words.append((b, positions[-1], terminal))
        return all_words

    def _find_terminal_rule(self):
        """Find the terminal (arity 0) rule in this layer's rule set."""
        for r in self.all_rules:
            if self.grammar.arity(r) == 0:
                return r
        if self.transition_rule is not None:
            return self.transition_rule
        return self.all_rules[0]

    # ── reverse: deterministic tree-walk ──────────────────────────

    def reverse(self, words, nVectors, batch_size):
        """Decode derivation to recover the activation vector."""
        activation = torch.zeros(batch_size, nVectors, device=TheDevice.get())
        for b, v, r in words:
            activation[b, v] = 1.0
        return activation

    # ── utilities ─────────────────────────────────────────────────

    def set_tau(self, tau):
        """Anneal the Gumbel-softmax temperature."""
        self.tau = tau

class TruthLayer(Layer):
    """Truth store on SymbolicSpace: encoded truth statements scaled by DoT.

    Each truth statement is processed through the model pipeline to produce
    a symbolic activation ``[nSymbols]``.  The activation is then scaled by
    the DegreeOfTruth before storage:

        stored = activation * degree

    This means the stored vector carries the DoT intrinsically:
      - degree = +1 → full activation stored (attractor)
      - degree = -1 → negated activation stored (disperser)
      - degree =  0 → zero vector (inert, prunable)

    The ``field()`` method projects stored truths into ConceptualSpace
    via cosine similarity.  Because the degree is baked into the stored
    vectors, positive-DoT truths attract and negative-DoT truths repel
    without needing a separate degree buffer.

    Propositional structure is defined by the S-tier grammar:
      - ``part(S, S)`` — parthood / containment
      - ``equals(S, S)`` — identity / equivalence
    """

    def __init__(self, nDim: int, max_truths: int = 1024):
        super().__init__(nDim, nDim)
        self.nDim = nDim
        self.max_truths = max_truths

        # Storage buffer: activation * degree (DoT baked in)
        self.register_buffer(
            'truths',
            torch.zeros(max_truths, nDim),
        )
        self.register_buffer(
            'count',
            torch.tensor(0, dtype=torch.long),
        )

    # ── Record / Query ────────────────────────────────────────────────

    @torch.no_grad()
    def record(self, activation: torch.Tensor, degree: float) -> int:
        """Store a truth: activation scaled by its DegreeOfTruth.

        The stored vector is ``activation * degree``, so the DoT is
        encoded in both the magnitude and (for negative degrees) the
        direction of the stored representation.

        Args:
            activation: (nDim,) symbolic activation from the model pipeline.
            degree: scalar in [-1, 1].  +1 = certainly true, -1 = certainly
                    false, 0 = unknown/inert.

        Returns:
            Index of the stored entry.
        """
        if self.count >= self.max_truths:
            raise RuntimeError(
                f"Truth store full ({self.max_truths} entries). "
                "Increase max_truths or prune stale entries."
            )
        degree = max(-1.0, min(1.0, degree))
        idx = self.count.item()
        self.truths[idx] = activation.detach() * degree
        self.count += 1
        return idx

    def query(self, activation: torch.Tensor, threshold: float = 0.9
              ) -> Optional[Tuple[int, float]]:
        """Find the closest stored truth to ``activation``.

        Compares against the *direction* of stored truths (normalised).
        The sign of the cosine similarity tells you consonance (+) vs
        dissonance (−) with the stored truth.

        Args:
            activation: (nDim,) or (B, nDim) query vector.
            threshold: minimum absolute cosine similarity to count as a match.

        Returns:
            (index, similarity) of the best match, or None if no match
            exceeds the threshold.  similarity > 0 means consonant with
            a positive truth or dissonant with a negative truth.
        """
        n = self.count.item()
        if n == 0:
            return None

        stored = self.truths[:n]                                 # (n, D)
        q = activation.detach()
        if q.ndim == 1:
            q = q.unsqueeze(0)                                   # (1, D)

        q_norm = torch.nn.functional.normalize(q, dim=-1)
        s_norm = torch.nn.functional.normalize(stored, dim=-1)
        sims = (q_norm @ s_norm.T).squeeze(0)                   # (n,)

        best_abs, best_idx = sims.abs().max(dim=0)
        if best_abs.item() < threshold:
            return None
        idx = best_idx.item()
        return (idx, sims[idx].item())

    # ── Truth Field ───────────────────────────────────────────────────

    def field(self, concepts: torch.Tensor, eps: float = 1e-8
              ) -> torch.Tensor:
        """Project stored truths into a scalar truth field over concepts.

        Because the DoT is baked into the stored vectors, the field
        naturally produces attractors for positive truths and dispersers
        for negative truths:

            field(c) = (1/n) Σ_i  sim(c, truth_i)

        where ``truth_i = activation_i * degree_i``.

        Args:
            concepts: (B, N, D) concept vectors in ConceptualSpace.

        Returns:
            field: (B, N) scalar field in [-1, 1].
        """
        n = self.count.item()
        if n == 0:
            return torch.zeros(
                concepts.shape[0], concepts.shape[1],
                device=concepts.device, dtype=concepts.dtype,
            )

        stored = self.truths[:n]                                 # (n, D)

        c_norm = torch.nn.functional.normalize(
            concepts, dim=-1, eps=eps)                            # (B, N, D)
        # Don't normalize stored — the magnitude carries the DoT
        # Use dot product: stronger DoT → stronger field influence
        dots = torch.einsum('bnd,md->bnm', c_norm, stored)      # (B, N, n)
        truth_field = dots.sum(dim=-1) / (n + eps)               # (B, N)

        return truth_field.clamp(-1.0, 1.0)

    # ── Maintenance ───────────────────────────────────────────────────

    @torch.no_grad()
    def prune(self, min_norm: float = 1e-6):
        """Remove near-zero entries (truths with DoT ≈ 0).

        Compacts the store in-place.
        """
        n = self.count.item()
        if n == 0:
            return
        norms = self.truths[:n].norm(dim=-1)
        keep = norms > min_norm
        kept = self.truths[:n][keep]
        new_n = kept.shape[0]
        self.truths[:new_n] = kept
        self.truths[new_n:] = 0
        self.count.fill_(new_n)

    def __len__(self):
        return self.count.item()

    def __repr__(self):
        return (f"TruthLayer(nDim={self.nDim}, "
                f"truths={self.count.item()}/{self.max_truths})")

    # ── Test ──────────────────────────────────────────────────────────

    @staticmethod
    def test():
        D = 32
        tl = TruthLayer(D, max_truths=64)
        assert len(tl) == 0

        # Record truths — DoT is baked into the stored activation
        t1 = torch.randn(D)
        t2 = torch.randn(D)
        idx1 = tl.record(t1, degree=0.9)
        idx2 = tl.record(t2, degree=-0.7)
        assert len(tl) == 2

        # Stored vector = activation * degree
        assert torch.allclose(tl.truths[0], t1 * 0.9, atol=1e-6)
        assert torch.allclose(tl.truths[1], t2 * -0.7, atol=1e-6)

        # Query — exact match (high similarity)
        result = tl.query(t1, threshold=0.8)
        assert result is not None
        assert result[0] == 0
        assert result[1] > 0  # consonant with positive truth

        # Query — no match for random vector
        result = tl.query(torch.randn(D), threshold=0.99)
        assert result is None

        # Field projection
        concepts = torch.randn(2, 8, D)
        f = tl.field(concepts)
        assert f.shape == (2, 8)
        assert f.min() >= -1.0 and f.max() <= 1.0

        # Prune near-zero (DoT ≈ 0 produces near-zero stored vector)
        tl.record(torch.randn(D), degree=0.0)
        assert len(tl) == 3
        tl.prune(min_norm=1e-6)
        assert len(tl) == 2

#endregion


def test():
    torch.autograd.set_detect_anomaly(True)

    TruthLayer.test()
    LinearLayer.test()


    InvertibleLinearLayer.test()

    SigmaLayer.test()
    PiLayer.test()

    AttentionLayer.test()
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
