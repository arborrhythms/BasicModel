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
class NonNegativeLinearLayer(LinearLayer):
    """LinearLayer with entry-wise non-negative W via softplus.

    Applies softplus to the raw weight matrix so that all entries of the
    effective W are non-negative.  This preserves the monotonicity property
    of NonNegativeInvertibleLinearLayer without the LDU factorisation or
    invertibility machinery.

    Parameters are initialised so that softplus(raw) ≈ 1.0 (near-identity
    scaling), matching the NonNegativeInvertibleLinearLayer convention.
    """

    def __init__(self, nInput, nOutput, hasBias=True, naive=False, stable=False, ergodic=False):
        super().__init__(nInput, nOutput, hasBias=hasBias, naive=naive, stable=stable, ergodic=ergodic)
        with torch.no_grad():
            self.W.fill_(math.log(math.e - 1))  # softplus_inverse(1.0)

    def _get_W(self):
        """Effective weight matrix, non-negative by construction."""
        return nn.functional.softplus(self.W)

    def compute_W_current(self):
        if self.ergodic:
            return nn.functional.softplus(self.bias * self.W + self.var * self.noise)
        return self._get_W()

    def forward(self, x):
        if self.ergodic:
            self.resample_noise()
            W = nn.functional.softplus(self.bias * self.W + self.var * self.noise)
        else:
            W = self._get_W()
        return x @ W

    def _effective_bias(self):
        """Bias constrained to non-negative via softplus, matching NonNegativeInvertibleLinearLayer."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            raw = self.bias * self.biasWeight + self.var * self.biasNoise
            return F.softplus(raw)
        return F.softplus(self.biasWeight)

    def forwardBias(self, x):
        if self.hasBias:
            x = x + self._effective_bias()
        return x
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
    def _effective_bias(self):
        """Bias value for external use (e.g. PiLayer logit mode)."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            return self.bias * self.biasWeight + self.var * self.biasNoise
        return self.biasWeight

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


    # _effective_bias, forwardBias, reverseBias inherited from
    # InvertibleLinearLayer — no constraint needed with symmetric
    # log domain (−∞, +∞).

class MapppingLayer(InvertibleLinearLayer):
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
            self.layer = NonNegativeInvertibleLinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic)
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
    r"""Multiplicative boundary layer: [-1,1] → [-1,1].

    Both modes share the symmetric log-domain embedding (1+x)/(1-x):

        Forward:  z = _from_mult(exp(W @ log(_to_mult(x)) + b))
        Reverse:  x = _from_mult(exp(W⁻¹ @ (log(_to_mult(z)) - b)))

    Entry transform (1+x)/(1-x) = exp(2·atanh(x)):
        x = 0  →  1  →  log = 0   : absent = multiplicative identity
        x = +k and x = -k produce equal and opposite log-space contributions

    Exit transform (y-1)/(y+1) is the exact inverse.

    ``monotonic`` selects the weight constraint:
        monotonic=True:  W ≥ 0 (NonNegativeInvertibleLinearLayer) — ordering preserved
        monotonic=False: W unrestricted (InvertibleLinearLayer) — bitonic response
    """
    _eps = 1e-6

    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True, monotonic=False):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.stable     = stable
        self.hasBias    = hasBias
        self.monotonic  = monotonic
        if invertible:
            if monotonic:
                self.layer = NonNegativeInvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                                              naive=naive, ergodic=ergodic,
                                                              stable=stable)
            else:
                self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                                   naive=naive, ergodic=ergodic,
                                                   stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=hasBias)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def resample_noise(self):
        self.layer.resample_noise()

    # ── Symmetric domain transforms ──────────────────────────────────

    def _to_mult(self, x):
        """Map [-1, 1] → (0, ∞), identity at 0 → 1."""
        x = x.clamp(-1 + self._eps, 1 - self._eps)
        return (1 + x) / (1 - x)

    def _from_mult(self, y):
        """Map (0, ∞) → (-1, 1), identity at 1 → 0."""
        return (y - 1) / (y + 1)

    # ── forward / reverse ────────────────────────────────────────────

    def forward(self, x):
        if self.layer.ergodic:
            self.resample_noise()
        W = self.layer.compute_W_current()
        x = x.to(W.device)
        m = self._to_mult(x)                             # (0, ∞)
        l = torch.log(m)                                  # (-∞, +∞) = 2·atanh(x)
        wl = l @ W                                        # [..., nOut]
        b = self.layer._effective_bias()
        wl = wl + b                                       # unconstrained bias
        y = torch.exp(wl)                                 # (0, ∞)
        return self._from_mult(y)                         # → (-1, 1)

    def reverse(self, y):
        """Recover x from y.  Requires invertible=True."""
        W_inv = self.layer.compute_Winverse_current()
        y = y.to(W_inv.device)
        m = self._to_mult(y)                              # (0, ∞)
        l = torch.log(m)                                  # (-∞, +∞)
        b = self.layer._effective_bias()
        lx = (l - b) @ W_inv                              # [..., nIn]
        x_mult = torch.exp(lx)                            # (0, ∞)
        x = self._from_mult(x_mult)                       # → (-1, 1)
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
        assert torch.all(y >= -1) and torch.all(y <= 1), "PiLayer output must be in [-1, 1]"
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
            assert torch.all(y >= -1) and torch.all(y <= 1), f"{desc}: output outside [-1, 1]"

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

class LiftingLayer(Layer):
    """Codebook of verb weight matrices for conceptual composition (lift).

    Each verb in the codebook is a learned [D, D] weight matrix that
    transforms concept vectors bidirectionally (invertible linear map).
    Verb selection is soft: cosine similarity between a query embedding
    and learned codebook keys produces blending weights.

    Transitive — lift(C, C):
        VP_eff @ C1 → attention added to C2 (forward)
        VP_eff^T @ C2 → attention added to C1 (backward)

    Intransitive — lift(C):
        VP_eff @ C1 → self-attention added to C1
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

    def forward_transitive_svo(self, subject, verb, obj, symbolic_space):
        """Ternary LIFT: S V O.

        The object restricts the verb's lifting operation by intersecting
        verb symbols with object symbols in symbolic space, then mapping
        the restricted action back to conceptual space and applying it
        to the subject.

        PiLayer maps concept *activations* [B, nConcepts] → [B, nSymbols],
        so we first extract activations from the [B, N, D] concept tensors
        using the SubSpace norm/sqrt(D) convention, scaled to [-1, 1].

        Args:
            subject: [B, N, D] subject concepts (S).
            verb: [B, N, D] verb concepts (V).
            obj: [B, N, D] object concepts (O).
            symbolic_space: SymbolicSpace for concept<->symbol projection.
        Returns:
            [B, N, D] lifted subject concepts.
        """
        pi = symbolic_space.layer
        D = verb.shape[-1]

        # 1. Extract PiLayer-compatible activations from concept tensors
        #    Convention: norm/sqrt(D) ∈ [0,1] → 2x-1 ∈ [-1,1]
        verb_act = (2 * verb.norm(dim=-1) / math.sqrt(D) - 1).clamp(-0.999, 0.999)
        obj_act = (2 * obj.norm(dim=-1) / math.sqrt(D) - 1).clamp(-0.999, 0.999)

        # 2. Project activations to symbolic space via PiLayer
        verb_syms = pi.forward(verb_act)                  # [B, nSymbols]
        obj_syms = pi.forward(obj_act)                    # [B, nSymbols]

        # 3. Intersect: restrict verb by object (monotonic → min)
        restricted_syms = torch.min(verb_syms, obj_syms)  # [B, nSymbols]

        # 4. Map restricted symbols back to concept activation space
        restricted_acts = pi.reverse(restricted_syms)     # [B, nConcepts]
        restricted_w = (restricted_acts + 1) / 2          # → [0, 1] weights

        # 5. Weight verb content by restricted activations → query
        query = (verb * restricted_w.unsqueeze(-1)).mean(dim=1)  # [B, D]

        # 6. Select verb matrix using restricted concepts as query
        vp_eff = self._select_vp(query)                   # [B, D, D]

        # 5. Apply to subject
        fwd = torch.bmm(subject, vp_eff)                   # [B, N, D]
        return subject + fwd

    @staticmethod
    def test():
        device = TheDevice.get()
        B, N, D, V = 4, 8, 16, 6
        layer = LiftingLayer(nVerbs=V, nDim=D)

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

        # ── Ternary SVO ──────────────────────────────────────────
        # Mock symbolic space: PiLayer maps activations [B, N] → [B, nSym]
        class _MockSymSpace:
            pass
        mock_ss = _MockSymSpace()
        mock_ss.layer = PiLayer(N, N, monotonic=True, invertible=True)
        S = torch.randn(B, N, D, device=device)
        V = torch.randn(B, N, D, device=device)
        O = torch.randn(B, N, D, device=device)
        result = layer.forward_transitive_svo(S, V, O, mock_ss)
        assert result.shape == (B, N, D), f"SVO shape: {result.shape}"

        # Gradient flow through SVO
        S_g = S.clone().requires_grad_(True)
        V_g = V.clone().requires_grad_(True)
        O_g = O.clone().requires_grad_(True)
        out_svo = layer.forward_transitive_svo(S_g, V_g, O_g, mock_ss)
        out_svo.sum().backward()
        assert S_g.grad is not None, "no gradient on subject"
        assert V_g.grad is not None, "no gradient on verb"
        assert O_g.grad is not None, "no gradient on object"

        print("LiftingLayer tests passed.")
class LoweringLayer(Layer):
    """Rank-reducing bottleneck for conceptual composition (lower).

    Compresses a concept vector through a smaller dimension then expands
    back: [D] → [bottleneck] → [D]. For binary lower(C, C), the second
    argument gates the bottleneck representation to select a specific
    instance from the set represented by the first argument.
    """
    def __init__(self, nDim, bottleneck=None):
        if bottleneck is None:
            bottleneck = max(4, nDim // 4)
        super().__init__(nDim, nDim)
        self.nDim = nDim
        self.bottleneck = bottleneck
        self.down = LinearLayer(nDim, bottleneck)
        self.up = LinearLayer(bottleneck, nDim)

    def forward(self, left, right=None):
        """Lower a concept through the bottleneck.

        Args:
            left: [B, N, D] or [B, D] concept vectors (the set/type).
            right: [B, N, D] or [B, D] optional selector concept.
                   When provided, gates the bottleneck to select an instance.
        Returns:
            Lowered concept, same shape as left.
        """
        compressed = self.down.forward(left)        # [..., bottleneck]
        if right is not None:
            gate = torch.sigmoid(self.down.forward(right))  # [..., bottleneck]
            compressed = compressed * gate
        return self.up.forward(compressed)           # [..., D]

    @staticmethod
    def test():
        device = TheDevice.get()
        B, N, D = 4, 8, 16
        layer = LoweringLayer(nDim=D, bottleneck=4)

        left = torch.randn(B, N, D, device=device)
        right = torch.randn(B, N, D, device=device)

        # Unary
        out = layer.forward(left)
        assert out.shape == (B, N, D), f"unary shape: {out.shape}"

        # Binary (with selector)
        out2 = layer.forward(left, right)
        assert out2.shape == (B, N, D), f"binary shape: {out2.shape}"

        # Gradient flow
        left_g = left.clone().requires_grad_(True)
        right_g = right.clone().requires_grad_(True)
        out3 = layer.forward(left_g, right_g)
        out3.sum().backward()
        assert left_g.grad is not None, "no gradient on left"
        assert right_g.grad is not None, "no gradient on right"

        print("LoweringLayer tests passed.")

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

    # ── Luminosity ─────────────────────────────────────────────────────

    def luminosity(self, pi_layer=None) -> torch.Tensor:
        """Compute luminosity: ||min(truths)||.

        Takes the element-wise min across all stored truth activations
        (conjunction in conceptual space), then returns the L2 norm.

        High luminosity = coherent, consistent truth set (bright).
        Low luminosity  = contradictory or sparse truths (dim).

        Args:
            pi_layer: optional PiLayer to project from symbolic to
                      conceptual dim before computing the conjunction.

        Returns:
            Scalar luminosity value >= 0.
        """
        n = self.count.item()
        if n == 0:
            return torch.tensor(0.0, device=self.truths.device)

        stored = self.truths[:n]                          # (n, D)

        # Project to conceptual space if needed
        if pi_layer is not None:
            stored = pi_layer.reverse(stored)

        # Conjunction: element-wise min across all truths
        conjunction = stored.min(dim=0).values            # (D,)

        # Luminosity = norm of the positive part of the conjunction.
        # Negative dimensions represent darkness (conflicting truths)
        # and don't contribute to illumination.
        return torch.relu(conjunction).norm()

    # ── Universality (Golden Rule) ────────────────────────────────────

    def universality(self, subject, verb, obj, lifting_layer, symbolic_space):
        """Golden rule: measure luminosity change from K(X,Y) + K(Y,X).

        1. Compute luminosity_before (baseline truth brightness).
        2. Apply SVO: lift(S, V, O) → project → temporarily store.
        3. Apply OVS: lift(O, V, S) → project → temporarily store.
        4. Compute luminosity_after.
        5. Return luminosity_after - luminosity_before.

        Positive = action preserves/increases illumination (kind).
        Negative = action diminishes illumination (unkind).

        Args:
            subject: [B, N, D] subject concepts.
            verb: [B, N, D] verb concepts.
            obj: [B, N, D] object concepts.
            lifting_layer: LiftingLayer for verb application.
            symbolic_space: SymbolicSpace for projection.

        Returns:
            Scalar universality score.
        """
        pi = symbolic_space.layer
        luminosity_before = self.luminosity(pi)

        # K(X, Y): original action SVO
        result_svo = lifting_layer.forward_transitive_svo(
            subject, verb, obj, symbolic_space)

        # K(Y, X): dual action OVS
        result_ovs = lifting_layer.forward_transitive_svo(
            obj, verb, subject, symbolic_space)

        # Project results to symbolic space (extract activations first)
        D_dim = result_svo.shape[-1]
        svo_act = (2 * result_svo.norm(dim=-1) / math.sqrt(D_dim) - 1).clamp(-0.999, 0.999)
        ovs_act = (2 * result_ovs.norm(dim=-1) / math.sqrt(D_dim) - 1).clamp(-0.999, 0.999)
        svo_syms = pi.forward(svo_act)                   # [B, nSymbols]
        ovs_syms = pi.forward(ovs_act)                   # [B, nSymbols]

        # Temporarily extend truth store
        saved_count = self.count.item()
        self.record(svo_syms.mean(dim=0).detach(), degree=1.0)
        self.record(ovs_syms.mean(dim=0).detach(), degree=1.0)

        luminosity_after = self.luminosity(pi)

        # Restore truth store
        self.count.fill_(saved_count)
        self.truths[saved_count:] = 0

        return luminosity_after - luminosity_before

    # ── Implication Derivation ────────────────────────────────────────

    @torch.no_grad()
    def derive(self, grammar, threshold: float = 0.7,
               attenuation: float = 0.8) -> int:
        """Derive implied truths via pairwise mereological inference.

        For each pair of stored truths, checks if one is contained in
        the other (via Grammar ``_method_part``).  When the parthood
        score exceeds *threshold*, a new implied truth is recorded with
        attenuated DoT.

        Args:
            grammar: Grammar instance with ``_method_part`` method.
            threshold: minimum parthood score to derive an implication.
            attenuation: DoT scaling to prevent runaway chains.

        Returns:
            Number of new derived truths.
        """
        n = self.count.item()
        if n < 2:
            return 0

        stored = self.truths[:n]
        norms = stored.norm(dim=-1)
        derived = 0

        for i in range(n):
            if norms[i] < 1e-6:
                continue
            for j in range(n):
                if i == j or norms[j] < 1e-6:
                    continue
                if self.count >= self.max_truths:
                    return derived

                score = grammar._method_part(
                    stored[i].unsqueeze(0), stored[j].unsqueeze(0), 'S')
                if isinstance(score, torch.Tensor):
                    score = score.mean().item()
                if score > threshold:
                    # Direction of truth_j, degree attenuated by score
                    direction = F.normalize(stored[j].unsqueeze(0), dim=-1).squeeze(0)
                    # Sign of truth_i's DoT (positive or negative truth)
                    sign_i = 1.0 if stored[i].mean().item() >= 0 else -1.0
                    degree = attenuation * score * sign_i
                    self.record(direction, degree)
                    derived += 1

        return derived

    # ── Consistency Scoring ───────────────────────────────────────────

    @torch.no_grad()
    def consistency(self, sim_threshold: float = 0.7) -> torch.Tensor:
        """Detect logical contradictions within stored truths.

        Two truths pointing in similar directions but with opposite
        DoT signs represent a contradiction.

        Args:
            sim_threshold: minimum cosine similarity for two truths
                to be considered "same direction".

        Returns:
            Scalar in [0, 1] where 1 = fully consistent.
        """
        n = self.count.item()
        if n < 2:
            return torch.tensor(1.0, device=self.truths.device)

        stored = self.truths[:n]
        norms = stored.norm(dim=-1)

        valid = norms > 1e-6
        if valid.sum() < 2:
            return torch.tensor(1.0, device=self.truths.device)

        directions = F.normalize(stored[valid], dim=-1)

        # Pairwise cosine similarity
        sim_matrix = directions @ directions.T  # (m, m)
        m = directions.shape[0]

        # Conflict: two stored vectors that are anti-parallel.
        # With DoT baked in, same-content opposite-DoT truths point
        # in opposite directions (cosine sim ≈ -1).
        n_conflicts = 0
        n_pairs = 0
        for i in range(m):
            for j in range(i + 1, m):
                n_pairs += 1
                if sim_matrix[i, j] < -sim_threshold:
                    n_conflicts += 1

        if n_pairs == 0:
            return torch.tensor(1.0, device=self.truths.device)
        return torch.tensor(1.0 - n_conflicts / n_pairs,
                            device=self.truths.device)

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

        # ── Luminosity ────────────────────────────────────────────
        # Luminosity = ||min(truths)|| — conjunction then norm
        tl2 = TruthLayer(D, max_truths=64)
        # All-positive truth → luminosity is the norm of the min
        pos = torch.ones(D) * 0.5
        tl2.record(pos, degree=1.0)
        lum = tl2.luminosity()
        assert lum.item() > 0, f"single positive truth should have lum > 0, got {lum}"

        # Adding a contradictory truth (negative) should lower luminosity
        neg = torch.ones(D) * 0.5
        tl2.record(neg, degree=-1.0)
        lum2 = tl2.luminosity()
        assert lum2 < lum, (
            f"contradictory truth should lower luminosity: {lum2} >= {lum}")

        # Empty truth layer → luminosity = 0
        tl_empty = TruthLayer(D, max_truths=64)
        assert tl_empty.luminosity().item() == 0.0

        # ── Consistency ───────────────────────────────────────────
        tl3 = TruthLayer(D, max_truths=64)
        # Two consistent truths (same direction, same sign)
        v = torch.randn(D)
        tl3.record(v, degree=0.9)
        tl3.record(v * 1.1, degree=0.8)
        assert tl3.consistency().item() == 1.0, "same-sign truths should be consistent"

        # Add contradictory truth (same direction, opposite sign)
        tl3.record(v, degree=-0.9)
        c = tl3.consistency().item()
        assert c < 1.0, f"contradictory truth should lower consistency: {c}"

        # Empty / single truth → consistent
        tl4 = TruthLayer(D, max_truths=64)
        assert tl4.consistency().item() == 1.0
        tl4.record(torch.randn(D), degree=0.5)
        assert tl4.consistency().item() == 1.0

        # ── Universality ─────────────────────────────────────────
        # PiLayer maps activations [B, N] → [B, nSym].
        # TruthLayer nDim must match nSym (symbol dimension).
        B, N = 2, 4
        nSym = 6
        tl5 = TruthLayer(nSym, max_truths=64)
        # Store a "positive" moral axiom
        axiom = torch.rand(nSym) * 0.5 + 0.25  # positive vector
        tl5.record(axiom, degree=1.0)

        S = torch.randn(B, N, D)
        V = torch.randn(B, N, D)
        O = torch.randn(B, N, D)

        # Mock symbolic space: PiLayer maps [B, N] → [B, nSym]
        class _MockSS:
            pass
        mock_ss = _MockSS()
        mock_ss.layer = PiLayer(N, nSym, monotonic=True, invertible=True)
        lifting = LiftingLayer(nVerbs=8, nDim=D)

        u_score = tl5.universality(S, V, O, lifting, mock_ss)
        assert isinstance(u_score, torch.Tensor), "universality should return tensor"
        # Score should be finite
        assert torch.isfinite(u_score), f"universality NaN/Inf: {u_score}"
        # Truth store should be restored
        assert len(tl5) == 1, f"truth store not restored: {len(tl5)}"

        print("TruthLayer tests passed.")

class ChunkLayer(Layer):
    """Learned BPE-style codebook for perceptual chunking.

    Each entry stores a merge prototype (what the pair looks like) and a
    split prototype (the two constituents).  Forward scores adjacent pairs
    against the codebook; reverse looks up the entry to reconstruct.

    The entropic gate merges a pair only when the codebook match score
    exceeds a learned threshold — i.e. when encoding the pair as a single
    chunk saves more bits than it costs.

    Merging stops at word boundaries (whitespace characters) so that chunks
    never cross word edges.  In the byte stream the space character (0x20)
    and common whitespace bytes serve as boundary markers.
    """

    def __init__(self, nDim, nChunks=256):
        super().__init__(nDim, nDim)
        self.nDim = nDim
        self.nChunks = nChunks
        # Split prototypes: what each chunk decodes to  [nChunks, 2, nDim]
        self.split = nn.Parameter(torch.randn(nChunks, 2, nDim) * 0.02)
        # Merge prototypes: the single vector a chunk becomes  [nChunks, nDim]
        self.merge = nn.Parameter(torch.randn(nChunks, nDim) * 0.02)
        # Learned threshold — merge only when score exceeds this
        self.threshold = nn.Parameter(torch.zeros(1))

    def score_pair(self, v1, v2):
        """Score a single (v1, v2) pair against all codebook entries.

        Args:
            v1: [nDim]  — left element
            v2: [nDim]  — right element
        Returns:
            best_score: scalar — cosine similarity of best match
            best_id:    int    — codebook index of best match
        """
        pair = torch.stack([v1, v2], dim=0)              # [2, nDim]
        pair_flat = pair.reshape(1, -1)                   # [1, 2*nDim]
        split_flat = self.split.reshape(self.nChunks, -1) # [K, 2*nDim]
        sims = F.cosine_similarity(pair_flat, split_flat, dim=-1)  # [K]
        best_score, best_id = sims.max(dim=0)
        return best_score, best_id.item()

    def encode(self, v1, v2):
        """Encode a pair into a single chunk vector.

        Returns:
            merged: [nDim]     — the chunk vector
            chunk_id: int      — which codebook entry was used
        """
        best_score, best_id = self.score_pair(v1, v2)
        return self.merge[best_id], best_id

    def decode(self, chunk_id):
        """Decode a codebook entry back to two vectors.

        Args:
            chunk_id: int — codebook index
        Returns:
            v1: [nDim], v2: [nDim]
        """
        return self.split[chunk_id, 0], self.split[chunk_id, 1]

    def should_merge(self, v1, v2):
        """Entropic gate: merge only if score exceeds threshold."""
        best_score, best_id = self.score_pair(v1, v2)
        return best_score > self.threshold, best_id

    # ── Boundary detection ────────────────────────────────────────────

    BOUNDARY_BYTES = frozenset({0x00, 0x09, 0x0A, 0x0D, 0x20})

    def is_word_boundary(self, data, b, pos, subspace=None, byte_indices=None):
        """True if position is a word boundary.

        Byte mode: check byte_indices against BOUNDARY_BYTES (fast, exact).
        Word mode: cosine similarity against space embedding (learned, soft).
        """
        if byte_indices is not None:
            return byte_indices[b, pos].item() in self.BOUNDARY_BYTES
        if subspace is None:
            return False
        try:
            space_emb = subspace.vocabulary.get_space_embedding()
        except (AttributeError, RuntimeError):
            return False
        vec = data[b, pos]
        sim = F.cosine_similarity(vec.unsqueeze(0), space_emb.unsqueeze(0), dim=-1)
        return sim.item() > 0.9

    # ── Byte-mode hard merge + compaction ─────────────────────────────

    def hard_merge_spans(self, data, byte_indices):
        """Merge contiguous non-boundary byte slots via mean aggregation.

        Args:
            data: [B, N, D] byte-level vectors
            byte_indices: [B, N] long — byte values 0-255
        Returns:
            data: [B, N, D] — span starts hold mean vectors, rest zeroed
            span_meta: list[list[(start, end)]] per batch
        """
        B, N, D = data.shape
        data = data.clone()
        span_meta = []
        for b in range(B):
            spans = []
            i = 0
            while i < N:
                bval = byte_indices[b, i].item()
                if bval == 0:
                    break
                if bval in self.BOUNDARY_BYTES:
                    data[b, i] = 0.0
                    i += 1
                    continue
                start = i
                while i < N and byte_indices[b, i].item() not in self.BOUNDARY_BYTES:
                    i += 1
                end = i - 1
                data[b, start] = data[b, start:end + 1].mean(dim=0)
                if end > start:
                    data[b, start + 1:end + 1] = 0.0
                spans.append((start, end))
            span_meta.append(spans)
        return data, span_meta

    def compact(self, data, nWordSlots, span_meta, where_encoding=None):
        """Pack active span-start positions into dense [B, nWordSlots, D].

        After packing, overwrites the where-encoding dims ([-4, -3]) with
        sinusoidal encoding of each span's start byte offset, so downstream
        layers see word-level positional encoding instead of byte-level.

        Args:
            data: [B, N, D] sparse byte-level vectors (span starts populated)
            nWordSlots: target dense width
            span_meta: list[list[(start, end)]] per batch
            where_encoding: WhereEncoding instance (for sin/cos rewrite)
        Returns:
            dense: [B, nWordSlots, D]
            compact_map: list[list[(dense_idx, start, end)]] — for reverse
        """
        B, N, D = data.shape
        dense = torch.zeros(B, nWordSlots, D, device=data.device)
        compact_map = []
        for b in range(B):
            mapping = []
            for dense_idx, (start, end) in enumerate(span_meta[b]):
                if dense_idx >= nWordSlots:
                    break
                dense[b, dense_idx] = data[b, start]
                mapping.append((dense_idx, start, end))
            compact_map.append(mapping)

        # Overwrite where-encoding dims with span start byte offset
        if where_encoding is not None and where_encoding.nDim > 0:
            where_idx = [D + i for i in where_encoding.index]  # e.g. [-4,-3] → absolute
            for b in range(B):
                for dense_idx, start, _end in compact_map[b]:
                    pos_enc = where_encoding.encode(float(start))  # [2] sin/cos
                    dense[b, dense_idx, where_idx] = pos_enc

        return dense, compact_map

    def uncompact(self, dense, compact_map, nByteSlots):
        """Scatter dense word vectors back to byte positions (span copy)."""
        B, _, D = dense.shape
        data = torch.zeros(B, nByteSlots, D, device=dense.device)
        for b in range(B):
            for dense_idx, start, end in compact_map[b]:
                span_len = end - start + 1
                data[b, start:end + 1] = dense[b, dense_idx].unsqueeze(0).expand(span_len, -1)
        return data
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
