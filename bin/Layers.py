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

    def Start(self):
        """Per-sentence state reset. Cascades to child layers.

        Layers with per-call state (e.g. cached ButterflyStage diffs,
        accumulating regularizer buffers) override this to clear that
        state. The default walks self.layers.
        """
        for layer in self.layers:
            if hasattr(layer, 'Start'):
                layer.Start()

    def End(self):
        """Per-batch teardown. Counterpart to Start().

        Cascades End() to child layers so any per-call scratch state
        cleared in Start() is also released at batch completion. The
        default walks self.layers; subclasses with additional per-call
        caches override to drop them.
        """
        for layer in self.layers:
            if hasattr(layer, 'End'):
                layer.End()

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

    Tracks **sigma** -- the running variance of the layer's gradient energy.
    Sigma drives scalar bias and var that broadcast over any weight shape:

        var  = sigma / (sigma + kappa)      exploration noise
        bias = 1 - var                      weight trust

    Low sigma (consistent gradient, found a minimum) -> high bias, low var.
    High sigma (unstable gradient) -> low bias, high var.

    Subclasses mix learned weights (scaled by ``bias``) with random noise
    (scaled by ``var``) in their forward passes:
        effective_weight = bias * W + var * noise

    External control via ``set_sigma(sigma)``:
        sigma=1: responsive exploration (low kappa)
        sigma=0: suppress exploration (high kappa, var ~= 0)

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
        sigma=0: suppress exploration (high kappa, var ~= 0)
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
            self.W = nn.Parameter((2 * torch.rand(self.nInput, self.nOutput) - 1) / self.nInput)
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

    Parameters are initialised so that softplus(raw) ~= 1.0 (near-identity
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

    Non-ergodic inverse is exact via triangular solves: W^-^1 = U^-^1 D^-^1 L^-^1.

    Ergodic mode injects noise into each factor before extracting the
    triangular structure, preserving the LDU form so the exact inverse is
    always available -- no approximation or SVD required:
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
        """W^-^1 = U^-^1 @ D_embed^-^1 @ L^-^1.  Shape [nOutput, nInput].
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
          the preceding forward() call -- no new resampling is done until after
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

        # Ergodic roundtrip (factor-level noise -> exact inverse)
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

    The exact LDU inverse (triangular solves) is still available -- the
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
            # softplus(-5) ~ 0.007 ~= 0 for off-diagonals
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
    # InvertibleLinearLayer -- no constraint needed with symmetric
    # log domain (-inf, +inf).

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
    """Additive (summation) layer.

    With ``nonlinear=True`` (legacy behavior), ``forward`` returns
    ``tanh(W @ x + b)``. With ``nonlinear=False``, ``forward`` returns the
    raw linear result. ``reverse`` mirrors the same choice.

    When ``invertible=True``, uses an invertible linear layer so
    ``reverse()`` is available via the exact LDU inverse.  When
    ``invertible=False`` (default), uses a plain LinearLayer.

    Weight initialization (non-ergodic) is scaled by 1/nInput so that
    the output stays in [-1, 1] at init when input is in [-1, 1].

    All ergodic machinery lives in the inner layer; SigmaLayer dispatches
    the ergodic interface (set_sigma, observe_sigma, etc.) there.

    ``monotonic`` selects the weight constraint (only meaningful with
    ``invertible=True``):
        monotonic=True:  W >= 0 (NonNegativeInvertibleLinearLayer) -- ordering preserved
        monotonic=False: W unrestricted (InvertibleLinearLayer)    -- bitonic response
    """
    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, nonlinear=True, stable=False,
                 monotonic=False):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.ergodic    = ergodic
        self.nonlinear  = nonlinear
        self.stable     = stable
        self.monotonic  = monotonic
        self.activation = torch.zeros(1, nOutput, 1)
        if invertible:
            if monotonic:
                self.layer = NonNegativeInvertibleLinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic, stable=stable)
            else:
                self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic, stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def forward(self, x):
        if self.nonlinear:
            x = torch.atanh(x.clamp(-1 + epsilon, 1 - epsilon))
        y = self.layer.forward(x)
        if self.nonlinear:
            y = torch.tanh(y)
        self.activation = y
        return y

    def reverse(self, y):
        """Invert tanh then apply W^-1 then tanh. Requires invertible=True."""
        if self.nonlinear:
            y = torch.atanh(y.clamp(-1 + epsilon, 1 - epsilon))
        x = self.layer.reverse(y)
        if self.nonlinear:
            x = torch.tanh(x)
        self.activation = x
        return x

    @staticmethod
    def test():
        nInput, nOutput = 3, 4
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, nonlinear=True)

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
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=False,
                           invertible=True, nonlinear=True)
        x = torch.randn((2, 5, nInput), device=TheDevice.get())
        layer.set_sigma(0.0)
        y = layer.forward(x)
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=True,
                           invertible=True, nonlinear=True)
        x = torch.randn((4, 8, nInput), device=TheDevice.get())
        layer.set_sigma(0.0)
        y = layer.forward(x)
        assert y.shape == (4, 8, nOutput), "Incorrect Size"
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        print("SigmaLayer tests passed.")
class PiLayer(Layer):
    r"""Multiplicative boundary layer: [-1,1] -> [-1,1].

    Both modes share the symmetric log-domain embedding (1+x)/(1-x):

        Forward:  z = _from_mult(exp(W @ log(_to_mult(x)) + b))
        Reverse:  x = _from_mult(exp(W^-^1 @ (log(_to_mult(z)) - b)))

    Entry transform (1+x)/(1-x) = exp(2*atanh(x)):
        x = 0  ->  1  ->  log = 0   : absent = multiplicative identity
        x = +k and x = -k produce equal and opposite log-space contributions

    Exit transform (y-1)/(y+1) is the exact inverse.

    ``monotonic`` selects the weight constraint:
        monotonic=True:  W >= 0 (NonNegativeInvertibleLinearLayer) -- ordering preserved
        monotonic=False: W unrestricted (InvertibleLinearLayer) -- bitonic response
    """
    _eps = 1e-6

    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True,
                 monotonic=False, nonlinear=True):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.stable     = stable
        self.hasBias    = hasBias
        self.monotonic  = monotonic
        self.nonlinear  = nonlinear
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

    # -- Symmetric domain transforms ----------------------------------

    def _to_mult(self, x):
        """Map [-1, 1] -> (0, inf), identity at 0 -> 1."""
        if self.nonlinear:
            x = x.clamp(-1 + self._eps, 1 - self._eps)
        return (1 + x) / (1 - x)

    def _from_mult(self, y):
        """Map (0, inf) -> (-1, 1), identity at 1 -> 0."""
        return (y - 1) / (y + 1)

    # -- forward / reverse --------------------------------------------

    def forward(self, x):
        if self.layer.ergodic:
            self.resample_noise()
        W = self.layer.compute_W_current()
        x = x.to(W.device)
        m = self._to_mult(x)                             # (0, inf)
        l = torch.log(m)                                  # (-inf, +inf) = 2*atanh(x)
        wl = l @ W                                        # [..., nOut]
        b = self.layer._effective_bias()
        wl = wl + b                                       # unconstrained bias
        if self.nonlinear:
            return torch.tanh(wl / 2)                     # -> (-1, 1)
        return torch.exp(wl)                              # -> (0, inf)

    def reverse(self, y):
        """Recover x from y.  Requires invertible=True."""
        W_inv = self.layer.compute_Winverse_current()
        y = y.to(W_inv.device)
        if self.nonlinear:
            m = self._to_mult(y)                          # (0, inf)
            l = torch.log(m)                              # (-inf, +inf)
        else:
            l = torch.log(y)                              # exp-domain inverse
        b = self.layer._effective_bias()
        lx = (l - b) @ W_inv                              # [..., nIn]
        if self.nonlinear:
            x = torch.tanh(lx / 2)                        # -> (-1, 1)
        else:
            x = self._from_mult(torch.exp(lx))
        if self.layer.ergodic:
            self.resample_noise()
        return x

    @staticmethod
    def test():
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayer(nInput=nInput, nOutput=nOutput, nonlinear=True)
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
            kw = dict(nInput=3, nOutput=6, invertible=True, nonlinear=True)
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
            kw = dict(nInput=3, nOutput=6, invertible=True, stable=True,
                      nonlinear=True)
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

class ButterflyStage(Layer):
    """Butterfly wrapper: permute -> pack -> inner layer -> unpack -> merge.

    Wraps a SigmaLayer or PiLayer to implement one stage of a butterfly
    network with N-halving.  The inner layer sees pairs of adjacent
    vectors packed into ``[B, N/2, 2D]`` and produces ``[B, N/2, 2D]``.
    After unpacking back to ``[B, N, D]``, an average-merge halves N
    to ``[B, N/2, D]``, caching the difference for exact inversion.

    From the Space's perspective, ``ButterflyStage.forward(x)`` is a
    drop-in replacement for ``SigmaLayer.forward(x)`` -- the butterfly
    mechanics are an implementation detail of the layer.

    Args:
        inner: the SigmaLayer or PiLayer to wrap (operates on 2D dims).
        stage_idx: butterfly stage index (determines permutation pattern).
        initial_n: vector count at stage 0 (N halves each stage).
        is_last: if True, skip the merge (output stays [B, N_t, D]).
    """

    def __init__(self, inner, stage_idx, initial_n, is_last=False):
        pair_dim = inner.nInput
        assert pair_dim % 2 == 0, (
            f"ButterflyStage inner layer nInput must be even (got {pair_dim})")
        super().__init__(pair_dim // 2, pair_dim // 2)
        self.inner = inner
        self.stage_idx = stage_idx
        self.initial_n = initial_n
        self.is_last = is_last
        self._merge_diff = None
        self.layers = [inner]

    @property
    def n_current(self):
        """Vector count at this stage's input."""
        return self.initial_n // (2 ** self.stage_idx)

    @staticmethod
    def _permutation(n_vectors, stage, device=None):
        """Permutation that makes XOR-neighbors adjacent for a butterfly stage."""
        if n_vectors <= 1:
            return torch.arange(n_vectors, dtype=torch.long, device=device)
        span = int(math.log2(n_vectors))
        bit = stage % max(span, 1)
        stride = 1 << bit
        block = stride << 1
        order = []
        for start in range(0, n_vectors, block):
            for offset in range(stride):
                order.append(start + offset)
                order.append(start + offset + stride)
        return torch.tensor(order, dtype=torch.long, device=device)

    @staticmethod
    def _inverse_permutation(perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(len(perm), device=perm.device)
        return inv

    def forward(self, x):
        """[B, N_t, D] -> [B, N_t/2, D] (or [B, N_t, D] if is_last)."""
        B, N, D = x.shape
        # 1. Permute: reorder vectors so XOR-neighbors are adjacent
        perm = self._permutation(N, self.stage_idx, device=x.device)
        x_perm = x[:, perm, :]
        # 2. Pack pairs: [B, N, D] -> [B, N/2, 2D]
        pair_input = x_perm.reshape(B, N // 2, 2 * D)
        # 3. Inner layer: [B, N/2, 2D] -> [B, N/2, 2D]
        pair_output = self.inner.forward(pair_input)
        # 4. Unpack: [B, N/2, 2D] -> [B, N, D]
        inv_perm = self._inverse_permutation(perm)
        x_out = pair_output.reshape(B, N, D)[:, inv_perm, :]
        # 5. Merge: [B, N, D] -> [B, N/2, D] (skip on last stage)
        if self.is_last:
            return x_out
        left = x_out[:, 0::2, :]
        right = x_out[:, 1::2, :]
        self._merge_diff = left - right
        return (left + right) / 2

    def reverse(self, y):
        """[B, N_t/2, D] -> [B, N_t, D] (or [B, N_t, D] if is_last)."""
        # 1. Unmerge: [B, N/2, D] -> [B, N, D]
        if self.is_last:
            x_out = y
        else:
            diff = self._merge_diff
            assert diff is not None, "ButterflyStage.reverse called without prior forward"
            left = y + diff / 2
            right = y - diff / 2
            B, N_half, D = left.shape
            x_out = torch.zeros(B, N_half * 2, D, device=y.device, dtype=y.dtype)
            x_out[:, 0::2, :] = left
            x_out[:, 1::2, :] = right
            self._merge_diff = None
        B, N, D = x_out.shape
        # 2. Permute (same permutation as forward)
        perm = self._permutation(N, self.stage_idx, device=x_out.device)
        x_perm = x_out[:, perm, :]
        # 3. Pack: [B, N, D] -> [B, N/2, 2D]
        pair_output = x_perm.reshape(B, N // 2, 2 * D)
        # 4. Inner layer reverse: [B, N/2, 2D] -> [B, N/2, 2D]
        pair_input = self.inner.reverse(pair_output)
        # 5. Unpack + inverse permute: [B, N/2, 2D] -> [B, N, D]
        inv_perm = self._inverse_permutation(perm)
        x_in = pair_input.reshape(B, N, D)[:, inv_perm, :]
        return x_in

    @staticmethod
    def test():
        """Verify forward->reverse roundtrip and N-halving."""
        D = 6
        N = 16
        B = 4

        # Test with SigmaLayer inner
        sigma = SigmaLayer(2 * D, 2 * D, invertible=True, naive=True,
                           nonlinear=True)
        sigma.set_sigma(0.0)

        # Non-last stage: should halve N
        stage = ButterflyStage(sigma, stage_idx=0, initial_n=N, is_last=False)
        x = torch.randn(B, N, D)
        y = stage.forward(x)
        assert y.shape == (B, N // 2, D), f"Expected ({B}, {N//2}, {D}), got {y.shape}"
        x_recon = stage.reverse(y)
        assert x_recon.shape == (B, N, D), f"Reverse shape: {x_recon.shape}"
        error = (x - x_recon).norm() / x.norm()
        assert error < 1e-4, f"Roundtrip error: {error:.2e}"

        # Last stage: should keep N
        stage_last = ButterflyStage(
            SigmaLayer(2 * D, 2 * D, invertible=True, naive=True,
                       nonlinear=True),
            stage_idx=0, initial_n=N, is_last=True)
        stage_last.inner.set_sigma(0.0)
        y_last = stage_last.forward(x)
        assert y_last.shape == (B, N, D), f"Last stage shape: {y_last.shape}"
        x_recon_last = stage_last.reverse(y_last)
        error_last = (x - x_recon_last).norm() / x.norm()
        assert error_last < 1e-4, f"Last stage roundtrip error: {error_last:.2e}"

        # Multi-stage pipeline: N=16 -> 8 -> 4 -> 2
        stages = []
        n = N
        for i in range(3):
            s = ButterflyStage(
                SigmaLayer(2 * D, 2 * D, invertible=True, naive=True,
                           nonlinear=True),
                stage_idx=i, initial_n=N, is_last=(i == 2))
            s.inner.set_sigma(0.0)
            stages.append(s)

        x = torch.randn(B, N, D)
        state = x
        for s in stages:
            state = s.forward(state)
        # After 2 merges + 1 last: N=16->8->4 (last keeps 4)
        assert state.shape == (B, 4, D), f"Pipeline output: {state.shape}"

        for s in reversed(stages):
            state = s.reverse(state)
        assert state.shape == (B, N, D), f"Pipeline reverse: {state.shape}"
        error = (x - state).norm() / x.norm()
        assert error < 1e-4, f"Pipeline roundtrip error: {error:.2e}"

        print("ButterflyStage tests passed.")


class SortingLayer(Layer):
    """NeuralSort: differentiable O(1)-depth sorting (Grover et al. 2019).

    Learns a content-determined canonical ordering of vectors along dim 1.
    A direction vector ``w`` scores each vector: s_i = w^Tv_i + bias.
    Scores produce a soft permutation matrix via:

        P[i,j] = softmax((N+1-2i)*s / tau,  dim=j)

    Row i of P concentrates on the element with the i-th largest score.
    The sorted output is P @ V -- a single batched matmul, no loops.

    The scale of w controls effective temperature: small w -> soft
    (uniform P, near-identity), large w -> hard (crisp permutation).

    Forward caches the pre-sort tensor for exact restoration in reverse().
    """

    def __init__(self, symbol_dim, n_passes=None):
        super().__init__(symbol_dim, symbol_dim)
        self.symbol_dim = symbol_dim
        # n_passes accepted for config compat but unused by NeuralSort
        self.w = nn.Parameter(torch.randn(symbol_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        self._pre_sort = None

    def forward(self, act):
        """NeuralSort on [B, N, D] along dim 1 -> [B, N, D]."""
        self._pre_sort = act.clone()
        B, N, D = act.shape
        if N <= 1:
            return act
        # Score each vector: [B, N]
        scores = (act * self.w).sum(dim=-1) + self.bias
        # NeuralSort coefficients (1-indexed ranks)
        rank = torch.arange(1, N + 1, device=act.device, dtype=scores.dtype)
        coeff = (N + 1 - 2 * rank)                        # [N]
        # logits[b,i,j] = coeff[i] * scores[b,j]
        logits = coeff.unsqueeze(0).unsqueeze(-1) * scores.unsqueeze(1)
        P = torch.softmax(logits, dim=-1)                  # [B, N, N]
        return torch.bmm(P, act)                           # [B, N, D]

    def reverse(self, act):
        """Restore pre-sort tensor cached during forward."""
        if self._pre_sort is not None:
            return self._pre_sort
        return act

    @staticmethod
    def test():
        nBatch, nSeq, nDim = 4, 8, 16
        layer = SortingLayer(symbol_dim=nDim, n_passes=None)
        device = next(layer.parameters()).device

        x = torch.randn(nBatch, nSeq, nDim, device=device)
        y = layer.forward(x)
        assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
        assert torch.isfinite(y).all(), "forward produced non-finite values"

        x_restored = layer.reverse(y)
        assert x_restored.shape == x.shape, f"reverse shape mismatch"
        err = (x_restored - x).abs().max().item()
        assert err < 1e-6, f"reverse restoration error: {err}"

        # Gradient flow through w and bias
        x2 = torch.randn(nBatch, nSeq, nDim, device=device, requires_grad=True)
        y2 = layer.forward(x2)
        loss = y2.sum()
        loss.backward()
        assert layer.w.grad is not None, "no gradient on w"
        assert layer.bias.grad is not None, "no gradient on bias"
        assert layer.w.grad.abs().sum() > 0, "zero gradient on w"

        # N=1 edge case
        x1 = torch.randn(nBatch, 1, nDim, device=device)
        y1 = layer.forward(x1)
        assert torch.allclose(x1, y1), "N=1 should be identity"

        # Soft permutation matrix is row-stochastic
        x3 = torch.randn(2, 5, nDim, device=device)
        scores = (x3 * layer.w).sum(dim=-1) + layer.bias
        rank = torch.arange(1, 6, device=device, dtype=scores.dtype)
        coeff = (6 - 2 * rank)
        logits = coeff.unsqueeze(0).unsqueeze(-1) * scores.unsqueeze(1)
        P = torch.softmax(logits, dim=-1)
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "P rows must sum to 1"

        print("SortingLayer tests passed.")

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

    type="symmetric"   -- Hopfield-like: scores = A^T @ A (positive semi-definite).
                         Attends across feature channels.
    type="asymmetric"  -- Channel attention: scores = Q^T @ K.
                         Attends across feature channels.
    type="transformer" -- Standard multi-head attention over the object/token axis.
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
        # Cached for diagnostic readers (e.g. InterSentenceLayer uses
        # the last-position entropy as a confidence signal).  Detached
        # so holding it doesn't pin the graph.
        self.last_attn = attn.detach()
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

    type="symmetric"  -- Hopfield-like: scores = A(x)^T @ A(x), positive
                        semi-definite.  Associations are symmetric (A==B <-> B==A).
    type="hopfield"   -- Modern Hopfield / softmax retrieval: projects x into
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
            [B, N] associated activation -- pattern-completed via stored associations.
        """
        # Reshape to [B, N, 1] for matmul compatibility
        x3 = x.unsqueeze(-1)  # [B, N, 1]

        if self.type == "symmetric":
            # Project: [B, N, 1] -> [B, N, H] via broadcasting
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
            # Similarity in hidden space -> association strength
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

    Transitive -- lift(C, C):
        VP_eff @ C1 -> attention added to C2 (forward)
        VP_eff^T @ C2 -> attention added to C1 (backward)

    Intransitive -- lift(C):
        VP_eff @ C1 -> self-attention added to C1
    """
    def __init__(self, nVerbs, nDim, ergodic=False):
        super().__init__(nDim, nDim)
        self.nVerbs = nVerbs
        self.nDim = nDim
        # Codebook keys for soft VERB selection [nVerbs, nDim]
        self.keys = nn.Parameter(torch.randn(nVerbs, nDim))
        # VERB weight matrices -- stack of [nDim, nDim] (initialized near-identity)
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

        VP_eff @ C1 -> attention added to C2 (forward)
        VP_eff^T @ C2 -> attention added to C1 (backward)

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

        PiLayer maps on the nDim axis: [B, N, concept_dim] -> [B, N, symbol_dim].
        Full concept vectors pass through directly.

        Args:
            subject: [B, N, D] subject concepts (S).
            verb: [B, N, D] verb concepts (V).
            obj: [B, N, D] object concepts (O).
            symbolic_space: SymbolicSpace for concept<->symbol projection.
        Returns:
            [B, N, D] lifted subject concepts.
        """
        ss = symbolic_space

        # 1. Project concept vectors to symbol space via SymbolicSpace.forward()
        ss.subspace.set_event(verb)
        verb_syms = ss.forward(ss.subspace).materialize()     # [B, N, symbol_dim]
        ss.subspace.set_event(obj)
        obj_syms = ss.forward(ss.subspace).materialize()      # [B, N, symbol_dim]

        # 2. Intersect: restrict verb by object (monotonic -> min)
        restricted_syms = torch.min(verb_syms, obj_syms)      # [B, N, symbol_dim]

        # 3. Map restricted symbols back to concept space
        restricted = ss.layer.reverse(restricted_syms)        # [B, N, D]

        # 4. Weight verb by restricted concept norms -> query
        rw = restricted.norm(dim=-1, keepdim=True)            # [B, N, 1]
        rw = rw / (rw.max(dim=1, keepdim=True).values + 1e-6)
        query = (verb * rw).mean(dim=1)                       # [B, D]

        # 5. Select verb matrix and apply to subject
        vp_eff = self._select_vp(query)                       # [B, D, D]
        fwd = torch.bmm(subject, vp_eff)                      # [B, N, D]
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

        # -- Ternary SVO ------------------------------------------
        # Mock symbolic space: PiLayer maps on nDim axis [B, N, D] -> [B, N, D]
        class _MockSubspace:
            def __init__(self):
                self._event = None
                self.batch = 0
            def set_event(self, t, compute_activation=False):
                self._event = t
            def materialize(self):
                return self._event
        class _MockSymSpace:
            def __init__(self, pi):
                self.layer = pi
                self.subspace = _MockSubspace()
            def forward(self, vspace):
                act = vspace.materialize()
                act = self.layer.forward(act)
                vspace.set_event(act)
                return vspace
        mock_ss = _MockSymSpace(PiLayer(D, D, monotonic=True,
                                        invertible=True, nonlinear=True))
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
    back: [D] -> [bottleneck] -> [D]. For binary lower(C, C), the second
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

class SparsityRegularizer(Layer):
    """Soft-threshold L1 proximal operator.

    Shared by PerceptualSpace, ConceptualSpace, and SymbolicSpace so a
    single sparsity implementation is reused across tiers. Extracted from
    ``SymbolicSpace.l1_proximal``.

    Acts as identity when disabled or when ``l1_lambda <= 0``. Otherwise
    applies ``sign(x) * max(|x| - l1_lambda, 0)``, which zeros activations
    below the threshold and shrinks survivors.
    """

    def __init__(self, l1_lambda: float = 0.0, enabled: bool = True):
        # nInput/nOutput are unused -- this is a pointwise op -- but the
        # Layer base contract requires both. Pass zeros; dim-agnostic.
        super().__init__(0, 0)
        self.l1_lambda = float(l1_lambda)
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.l1_lambda <= 0.0:
            return x
        return torch.sign(x) * torch.clamp(
            torch.abs(x) - self.l1_lambda, min=0.0
        )


class SmoothingRegularizer(Layer):
    """Total-variation penalty on consecutive concepts of a symbol vector.

    Complements ``SparsityRegularizer`` by pressuring the symbol vector toward
    a piecewise-flat profile in addition to an L1 sparsity norm. Penalises
    ``|S[..., k+1] - S[..., k]|`` and returns a scalar.

    In bivector mode (even last dim) the operand is reshaped to
    ``[..., K, 2]`` and collapsed along the pole axis with ``amax`` before
    differencing. This respects the paired-index convention from the
    bivector encoding: indices ``2k`` and ``2k+1`` are poles of the same
    concept -- penalising their difference would fight the 4-valued
    (quaternary) truth encoding, so we measure discontinuity between
    *distinct* concepts only. See basicmodel/doc/BuddhistParallels.md
    for the tetralemma (catuskoti) mapping.

    Acts as identity (returns scalar zero) when disabled or ``lam <= 0``.
    """

    def __init__(self, lam: float = 0.0, enabled: bool = True):
        super().__init__(0, 0)
        self.lam = float(lam)
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.lam <= 0.0:
            return x.new_tensor(0.0) if torch.is_tensor(x) else torch.tensor(0.0)
        if x.shape[-1] >= 2 and x.shape[-1] % 2 == 0:
            pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
            collapsed = pair.amax(dim=-1)
        else:
            collapsed = x
        if collapsed.shape[-1] < 2:
            return x.new_tensor(0.0)
        diff = collapsed[..., 1:] - collapsed[..., :-1]
        return self.lam * diff.abs().mean()


class ImpenetrableLayer(Layer):
    """Mereological separation regularizer over a symbol codebook.

    Classifies each ordered pair (i, j) of codebook rows into one of five
    mereological relations using ``Basis.part`` (clipped-cosine scalar
    parthood, scalar=True) and penalises overlap between rows whose
    EMA usage frequencies disagree.

    Penalty: ``overlap_strength(i, j) * |trust(i) - trust(j)|``
    where ``overlap_strength = min(P[i,j], P[j,i]) * (1 - max(P[i,j], P[j,i])^k)``
    damps to zero as the pair approaches mutual identity (``equal``).

    Five relations (with thresholds τ and ε):
      disjoint:  P[i,j] < ε  and  P[j,i] < ε
      part_ij:   P[i,j] > τ  and  P[j,i] < ε
      part_ji:   P[i,j] < ε  and  P[j,i] > τ
      equal:     P[i,j] > τ  and  P[j,i] > τ
      overlap:   both partial (neither > τ nor < ε)

    Trust source: ``basis.vq.cluster_size`` EMA when a VectorQuantize is
    present; falls back to row norms when VQ is off (e.g., passThrough).

    A separate variance floor guards against row-collapse (all rows
    converging to a single point).

    Returns a scalar. When ``enabled`` is false or all weights are zero,
    the layer short-circuits to zero without touching the codebook.

    See basicmodel/doc/BuddhistParallels.md for the tetralemma (catuskoti)
    mapping of the 4-valued truth logic that the separated codebook carries.
    """

    def __init__(self, overlap_weight: float = 0.1,
                 variance_floor: float = 0.01,
                 enabled: bool = True,
                 full_part_threshold: float = 0.9,
                 disjoint_threshold: float = 0.1,
                 equal_suppression: float = 4.0):
        super().__init__(0, 0)
        self.overlap_weight = float(overlap_weight)
        self.variance_floor = float(variance_floor)
        self.enabled = bool(enabled)
        self.tau = float(full_part_threshold)
        self.eps = float(disjoint_threshold)
        self.equal_k = float(equal_suppression)
        # Diagnostic slots populated on each forward pass.
        self.last_overlap_loss = None
        self.last_variance = None
        self.last_relation_counts = None

    def _pairwise_parthood(self, codebook: torch.Tensor,
                           basis) -> torch.Tensor:
        """Compute P[i, j] = part(cb[i], cb[j], scalar=True) for all K*K pairs."""
        K = codebook.shape[0]
        cb_i = codebook.unsqueeze(1).expand(K, K, -1)
        cb_j = codebook.unsqueeze(0).expand(K, K, -1)
        return basis.part(cb_i, cb_j, monotonic=True, scalar=True)

    def _trust(self, codebook: torch.Tensor, basis) -> torch.Tensor:
        """Trust per codebook row.

        Prefer VQ cluster_size EMA usage when the underlying basis has a
        live VectorQuantize; fall back to normalised row norms otherwise.
        Returns a [K] tensor on ``codebook.device``.
        """
        vq = getattr(basis, "vq", None)
        if vq is not None and hasattr(vq, "cluster_size"):
            counts = vq.cluster_size
            if torch.is_tensor(counts) and counts.numel() == codebook.shape[0]:
                counts = counts.to(codebook.device).float()
                return counts / counts.sum().clamp(min=1.0)
        n = codebook.norm(dim=-1).float()
        return n / n.max().clamp(min=epsilon)

    def _classify(self, P: torch.Tensor) -> dict:
        """Classify each ordered off-diagonal (i, j) pair into one of five
        mereological relations. The diagonal is masked out so diagnostic
        counts sum to ``K * (K - 1)``."""
        high = P > self.tau
        low = P < self.eps
        high_T = high.transpose(0, 1)
        low_T = low.transpose(0, 1)
        eye = torch.eye(P.shape[0], device=P.device, dtype=torch.bool)
        off = ~eye
        return {
            "disjoint": (low & low_T) & off,
            "part_ij":  (high & low_T) & off,
            "part_ji":  (low & high_T) & off,
            "equal":    (high & high_T) & off,
            "overlap":  (~(high | low) & ~(high_T | low_T)) & off,
        }

    def forward(self, codebook: torch.Tensor, basis=None) -> torch.Tensor:
        zero = (codebook.new_tensor(0.0) if isinstance(codebook, torch.Tensor)
                else torch.tensor(0.0))
        self.last_overlap_loss = None
        self.last_variance = None
        self.last_relation_counts = None
        if not self.enabled:
            return zero
        if codebook is None or not isinstance(codebook, torch.Tensor):
            return zero
        if codebook.ndim != 2 or codebook.shape[0] < 2:
            return zero
        want_overlap = self.overlap_weight > 0.0 and basis is not None
        want_var = self.variance_floor > 0.0
        if not (want_overlap or want_var):
            return zero

        K = codebook.shape[0]
        total = zero

        if want_overlap:
            P = self._pairwise_parthood(codebook, basis)
            trust = self._trust(codebook, basis)
            trust_diff = (trust.unsqueeze(0) - trust.unsqueeze(1)).abs()
            P_T = P.transpose(0, 1)
            min_P = torch.minimum(P, P_T)
            max_P = torch.maximum(P, P_T)
            # Damp overlap as the pair approaches mutual identity so that
            # two rows meant to encode the same concept (``equal``) do
            # not contribute to the overlap penalty.
            damp = (1.0 - max_P.clamp(0.0, 1.0) ** self.equal_k).clamp(min=0.0)
            eye = torch.eye(K, device=codebook.device, dtype=torch.bool)
            keep = (~eye).float()
            denom = keep.sum().clamp(min=1.0)
            overlap_loss = (min_P * damp * trust_diff * keep).sum() / denom
            self.last_overlap_loss = overlap_loss.detach()
            total = total + self.overlap_weight * overlap_loss
            rels = self._classify(P)
            self.last_relation_counts = {
                k: int(v.sum().item()) for k, v in rels.items()
            }

        if want_var:
            std = codebook.std(dim=0, unbiased=False).mean()
            var_pen = torch.relu(codebook.new_tensor(self.variance_floor) - std)
            self.last_variance = var_pen.detach()
            total = total + var_pen

        return total


class TruthLayer(Layer):
    """Truth store on SymbolicSpace: encoded truth statements scaled by DoT.

    Each truth statement is processed through the model pipeline to produce
    a symbolic activation ``[nSymbols]``.  The activation is then scaled by
    the DegreeOfTruth before storage:

        stored = activation * degree

    This means the stored vector carries the DoT intrinsically:
      - degree = +1 -> full activation stored (attractor)
      - degree = -1 -> negated activation stored (disperser)
      - degree =  0 -> zero vector (inert, prunable)

    The ``field()`` method projects stored truths into ConceptualSpace
    via cosine similarity.  Because the degree is baked into the stored
    vectors, positive-DoT truths attract and negative-DoT truths repel
    without needing a separate degree buffer.

    Propositional structure is defined by the S-tier grammar:
      - ``part(S, S)`` -- parthood / containment
      - ``equals(S, S)`` -- identity / equivalence
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
        # Parallel metadata used by suggest_clarifications(). Indexed
        # alongside truths[:count]; missing entries fall back to a
        # generic "(truth #i)" reference.
        self._sources = []
        self._trusts = []

    # -- Record / Query ------------------------------------------------

    @torch.no_grad()
    def record(self, activation: torch.Tensor, degree: float,
               basis=None) -> int:
        """Store a truth: activation scaled by its DegreeOfTruth.

        Bivector path (``basis`` provided and ``basis.monotonic`` and
        ``activation.shape[-1]`` is even): indices 2k / 2k+1 encode the
        positive / negative poles of concept k. For ``degree >= 0``,
        store ``activation * degree`` (positive poles already hot). For
        ``degree < 0``, paired-index flip via ``basis.negation(...,
        monotonic=True)`` lands the mass on the negative poles, then
        scale by ``|degree|``. This preserves 4-valued (quaternary)
        truth semantics: asserting A and asserting not(A) are
        orthogonal, not cancelling. See
        basicmodel/doc/BuddhistParallels.md for the tetralemma mapping.

        Legacy path (no basis or odd last dim): the stored vector is
        ``activation * degree``, so the DoT is encoded in both the
        magnitude and (for negative degrees) the sign.

        Args:
            activation: (nDim,) symbolic activation from the model pipeline.
            degree: scalar in [-1, 1].  +1 = certainly true, -1 = certainly
                    false, 0 = unknown/inert.
            basis: optional Basis with negation(monotonic=True). When
                   provided and monotonic, enables bivector storage.

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
        vec = activation.detach()

        bivector_mode = (
            basis is not None
            and getattr(basis, 'monotonic', False)
            and vec.shape[-1] % 2 == 0
        )

        if bivector_mode and degree < 0:
            vec = basis.negation(vec, monotonic=True)
            stored_vec = vec * abs(degree)
        else:
            stored_vec = vec * degree
        self.truths[idx] = stored_vec
        self.count += 1

        # Legacy contradiction warning: anti-parallel cosine only applies
        # to bitonic storage. Under bivector, A and not(A) land on
        # orthogonal paired indices (cosine 0, not -1), so this branch
        # is skipped.
        if not bivector_mode and idx > 0:
            existing = self.truths[:idx]
            s_norm = F.normalize(stored_vec.unsqueeze(0), dim=-1)
            e_norm = F.normalize(existing, dim=-1)
            sims = (s_norm @ e_norm.T).squeeze(0)
            worst = sims.min()
            if worst.item() < -0.7:
                j = sims.argmin().item()
                warnings.warn(
                    f"TruthLayer: new truth [{idx}] contradicts existing "
                    f"truth [{j}] (cosine similarity {worst.item():.3f})",
                    stacklevel=2,
                )

        return idx

    @torch.no_grad()
    def should_store(self, activation: torch.Tensor,
                     min_magnitude: float = 0.3,
                     min_novelty: float = 0.5,
                     max_inconsistency: float = 0.3) -> float:
        """Continuous storage score in [0, 1] for truth gating.

        Returns a score that is multiplied by ``accumulateTruth`` to decide
        storage: ``store iff accumulateTruth * score > 0.5``.

        At accumulateTruth=1 (truth-set processing), most legitimate
        activations score >= 0.5 and pass -- preserving legacy behavior.
        At lower accumulateTruth, the bar rises proportionally.

        Three gates (each contributes a factor in [0, 1]):

        1. **Magnitude** -- activation norm vs *min_magnitude*.
        2. **Novelty** -- distance from nearest stored truth.
        3. **Consistency** -- absence of contradictions.
        """
        # Gate 1: magnitude -- smooth ramp from 0 at norm=0 to 1 at norm=min_magnitude
        norm = activation.norm().item()
        mag_score = min(1.0, norm / max(min_magnitude, 1e-8))

        # Gate 2: novelty -- 1.0 if no close match, drops toward 0 for duplicates
        n = self.count.item()
        sims = None
        if n == 0:
            nov_score = 1.0
        else:
            stored = self.truths[:n]
            a_norm = F.normalize(activation.unsqueeze(0), dim=-1)
            s_norm = F.normalize(stored, dim=-1)
            sims = (a_norm @ s_norm.T).squeeze(0)
            max_sim = sims.abs().max().item()
            # novelty = 1 when max_sim=0, drops to 0 when max_sim >= (1 - min_novelty)
            threshold = 1.0 - min_novelty
            nov_score = max(0.0, 1.0 - max_sim / max(threshold, 1e-8))

        # Gate 3: consistency -- 1.0 if no contradiction, 0 if strong anti-alignment
        if n == 0 or sims is None:
            con_score = 1.0
        else:
            min_sim = sims.min().item()
            if min_sim < -max_inconsistency:
                con_score = 0.0
            else:
                con_score = 1.0

        return mag_score * nov_score * con_score

    def query(self, activation: torch.Tensor, threshold: float = 0.9
              ) -> Optional[Tuple[int, float]]:
        """Find the closest stored truth to ``activation``.

        Compares against the *direction* of stored truths (normalised).
        The sign of the cosine similarity tells you consonance (+) vs
        dissonance (-) with the stored truth.

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

    # -- Truth Field ---------------------------------------------------

    def field(self, concepts: torch.Tensor, eps: float = 1e-8
              ) -> torch.Tensor:
        """Project stored truths into a scalar truth field over concepts.

        Because the DoT is baked into the stored vectors, the field
        naturally produces attractors for positive truths and dispersers
        for negative truths:

            field(c) = (1/n) Sigma_i  sim(c, truth_i)

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
        # Don't normalize stored -- the magnitude carries the DoT
        # Use dot product: stronger DoT -> stronger field influence
        dots = torch.einsum('bnd,md->bnm', c_norm, stored)      # (B, N, n)
        truth_field = dots.sum(dim=-1) / (n + eps)               # (B, N)

        return truth_field.clamp(-1.0, 1.0)

    # -- Luminosity -----------------------------------------------------

    def _positive_poles(self, v: torch.Tensor) -> torch.Tensor:
        """Extract positive poles from a **paired-index** storage vector.

        Assumes ``v`` is laid out as repeated pairs
        ``[pos_0, neg_0, pos_1, neg_1, ...]`` over ``v.shape[-1] == 2K``.
        Even indices are positive poles; odd indices are negative poles.

        .. warning::
            This layout is **not** the current SymbolicSpace codebook layout,
            where each row is ``[pos_pole, neg_pole, where..., when...]``
            with a single leading bivector plus trailing positional data.
            Before calling ``luminosity``/``darkness``/
            ``tetralemma_balance_penalty`` with a symbol activation from the
            new codebook, **slice the leading bivector** via
            ``v[..., :2]`` at the call site. See basicmodel/doc/Spaces.md
            "Codebook shape" note for the mapping.
        """
        return v[..., 0::2]

    def _negative_poles(self, v: torch.Tensor) -> torch.Tensor:
        """Extract negative poles from a paired-index storage vector.

        See :meth:`_positive_poles` for the layout caveat regarding the
        new leading-bivector SymbolicSpace codebook.
        """
        return v[..., 1::2]

    def luminosity(self, pi_layer=None) -> torch.Tensor:
        """Compute luminosity: ||relu(min(positive_poles(truths)))||.

        Bivector path (even last dim): luminosity is the L2 norm of the
        element-wise min across positive poles only. Contradictions land
        on paired negative-pole indices and cannot dim the positive
        conjunction under 4-valued (quaternary) truth semantics. See
        basicmodel/doc/BuddhistParallels.md for the tetralemma mapping.

        Legacy path (odd last dim): element-wise min across all truths,
        positive-part norm (unchanged).

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

        if stored.shape[-1] % 2 == 0:
            stored = self._positive_poles(stored)

        # Conjunction: element-wise min across all truths
        conjunction = stored.min(dim=0).values            # (D/2,) or (D,)

        # Luminosity = norm of the positive part of the conjunction.
        return torch.relu(conjunction).norm()

    def darkness(self, pi_layer=None) -> torch.Tensor:
        """Diagnostic: ||relu(min(negative_poles(truths)))||.

        Mirror of ``luminosity`` on the negative-pole half of bivector
        storage. Elevated darkness means many truths have co-active
        negative poles (i.e. a shared "not-X" conjunction). Returns 0
        for non-bivector storage (odd last dim).
        """
        n = self.count.item()
        if n == 0:
            return torch.tensor(0.0, device=self.truths.device)

        stored = self.truths[:n]
        if pi_layer is not None:
            stored = pi_layer.reverse(stored)

        if stored.shape[-1] % 2 != 0:
            return torch.tensor(0.0, device=self.truths.device)

        neg = self._negative_poles(stored)
        conjunction = neg.min(dim=0).values
        return torch.relu(conjunction).norm()

    def tetralemma_balance_penalty(self, bivector_activation: torch.Tensor,
                               allow_excluded_middle: int = 1,
                               allow_contradiction: int = 0,
                               neither_threshold: float = 0.1) -> torch.Tensor:
        """Penalize forbidden corners of the quaternary truth lattice (tetralemma).

        ``bivector_activation`` is expected to be a `[..., 2K]` tensor in
        **paired-index** layout, where each consecutive pair ``(t+, t-)``
        encodes one concept's tetralemma corner:

            T = (1, 0)   F = (0, 1)
            N = (0, 0)   B = (1, 1)

        .. warning::
            The current SymbolicSpace codebook layout is
            ``[pos_pole, neg_pole, where..., when...]`` — a **single**
            leading bivector plus trailing positional template data, not
            repeated pairs. If passing a symbol activation from the new
            codebook, slice the leading bivector first
            (``sym_act[..., :2]``) to avoid applying tetralemma corner
            policy to positional-template dims. See
            basicmodel/doc/Spaces.md "Codebook shape" and
            basicmodel/doc/BuddhistParallels.md.

        Flags control which corners are penalized:
            allow_excluded_middle == -1  =>  penalize N (force classical LEM)
            allow_excluded_middle ==  1  =>  permit N (Kleene default)
            allow_contradiction   ==  0  =>  penalize B (non-contradiction)
            allow_contradiction   ==  1  =>  permit B (paraconsistent / LP)

        ``neither_threshold`` is the activation level below which a concept
        is considered "dark" (no commitment). Scaled linearly so the penalty
        is well-behaved near the threshold.

        Returns a scalar tensor. Odd last dim -> returns 0 (non-bivector).
        """
        act = bivector_activation
        zero = act.new_tensor(0.0) if torch.is_tensor(act) else torch.tensor(0.0)
        if act is None or not torch.is_tensor(act):
            return zero
        if act.shape[-1] % 2 != 0:
            return zero

        pair = act.reshape(*act.shape[:-1], act.shape[-1] // 2, 2)
        t_pos = pair[..., 0]
        t_neg = pair[..., 1]

        total = zero
        if int(allow_excluded_middle) == -1:
            hottest = torch.maximum(t_pos, t_neg)
            total = total + torch.relu(
                act.new_tensor(float(neither_threshold)) - hottest
            ).mean()
        if int(allow_contradiction) == 0:
            total = total + (t_pos * t_neg).mean()
        return total

    # -- Mereological fusion -------------------------------------------

    def fusion(self, indices: torch.Tensor = None) -> torch.Tensor:
        """Mereological fusion (least upper bound) of stored truths.

        Returns the elementwise ``max`` across the stored truth set
        (or the rows selected by ``indices``) — the axis-aligned bounding
        hyperrectangle in bivector space. Every individual truth is
        componentwise dominated by the fusion: ``t_i <= fusion`` per dim.

        Under paired-index storage ``[2K]``, pairs ``(2k, 2k+1)`` encode
        concept k's ``(pos, neg)`` poles, so fusion tightens each pair to
        its per-truth maximum — the hyperrectangle's "top right" corner
        in the 2D ``(pos, neg)`` plane of every concept.

        Under the current SymbolicSpace codebook layout
        ``[pos_pole, neg_pole, where..., when...]``, fusion on raw stored
        rows also maxes the positional trailers; slice ``[..., :2]``
        at the call site for a pure bivector fusion.

        Returns a ``(D,)`` tensor in the same layout as stored truths, or
        a zero vector when no truths are stored.
        """
        n = int(self.count.item())
        if n == 0:
            return torch.zeros(self.truths.shape[-1], device=self.truths.device,
                               dtype=self.truths.dtype)
        stored = self.truths[:n] if indices is None else self.truths[indices]
        return stored.max(dim=0).values

    def truth_conjunction(self, basis, pi_layer=None):
        """Conjunction of all stored truths via bitonic intersection.

        Folds stored truths with ``Basis.conjunction()`` (sign-aware
        element-wise min), optionally projecting from symbolic to
        conceptual space first via ``pi_layer.reverse()``.

        Returns:
            (D,) conjunction vector, or None if no truths stored.
        """
        n = self.count.item()
        if n == 0:
            return None

        stored = self.truths[:n]                          # (n, symbol_dim)

        # Project to conceptual space if needed
        if pi_layer is not None:
            stored = pi_layer.reverse(stored)              # (n, concept_dim)

        # Fold via bitonic conjunction
        conj = stored[0]
        for i in range(1, n):
            conj = basis.conjunction(conj, stored[i])

        return conj

    # -- Universality (Golden Rule) ------------------------------------

    def universality(self, subject, verb, obj, lifting_layer, symbolic_space):
        """Golden rule: measure luminosity change from K(X,Y) + K(Y,X).

        1. Compute luminosity_before (baseline truth brightness).
        2. Apply SVO: lift(S, V, O) -> project -> temporarily store.
        3. Apply OVS: lift(O, V, S) -> project -> temporarily store.
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
        ss = symbolic_space
        luminosity_before = self.luminosity(ss.layer)

        # K(X, Y): original action SVO
        result_svo = lifting_layer.forward_transitive_svo(
            subject, verb, obj, ss)

        # K(Y, X): dual action OVS
        result_ovs = lifting_layer.forward_transitive_svo(
            obj, verb, subject, ss)

        # Project results to symbol space via SymbolicSpace.forward()
        ss.subspace.set_event(result_svo)
        svo_syms = ss.forward(ss.subspace).materialize()  # [B, N, symbol_dim]
        ss.subspace.set_event(result_ovs)
        ovs_syms = ss.forward(ss.subspace).materialize()  # [B, N, symbol_dim]

        # Temporarily extend truth store (average over batch and vectors)
        saved_count = self.count.item()
        basis = getattr(getattr(ss, 'subspace', None), 'basis', None)
        self.record(svo_syms.mean(dim=(0, 1)).detach(), degree=1.0, basis=basis)
        self.record(ovs_syms.mean(dim=(0, 1)).detach(), degree=1.0, basis=basis)

        luminosity_after = self.luminosity(ss.layer)

        # Restore truth store
        self.count.fill_(saved_count)
        self.truths[saved_count:] = 0

        return luminosity_after - luminosity_before

    # -- Implication Derivation ----------------------------------------

    @torch.no_grad()
    def derive(self, part_fn, threshold: float = 0.7,
               attenuation: float = 0.8, basis=None) -> int:
        """Derive implied truths via pairwise mereological inference.

        For each pair of stored truths, checks if one is contained in
        the other (via ``part_fn``).  When the parthood score exceeds
        *threshold*, a new implied truth is recorded with attenuated DoT.

        Args:
            part_fn: callable(left, right, subspace) -> parthood score tensor.
                     Typically ``syntacticLayer.partForward``.
            threshold: minimum parthood score to derive an implication.
            attenuation: DoT scaling to prevent runaway chains.
            basis: optional Basis forwarded to record() for bivector storage.

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

                score = part_fn(
                    stored[i].unsqueeze(0), stored[j].unsqueeze(0), None)
                if isinstance(score, torch.Tensor):
                    score = score.mean().item()
                if score > threshold:
                    # Direction of truth_j, degree attenuated by score
                    direction = F.normalize(stored[j].unsqueeze(0), dim=-1).squeeze(0)
                    # Sign of truth_i's DoT (positive or negative truth)
                    sign_i = 1.0 if stored[i].mean().item() >= 0 else -1.0
                    degree = attenuation * score * sign_i
                    self.record(direction, degree, basis=basis)
                    derived += 1

        return derived

    # -- Consistency Scoring -------------------------------------------

    @torch.no_grad()
    def consistency(self, sim_threshold: float = 0.7,
                    pair_threshold: float = 0.3,
                    basis=None,
                    part_threshold: float = 0.3,
                    return_report: bool = False):
        """Detect logical contradictions within stored truths.

        Default (``return_report=False``) returns a scalar score for
        back-compat. With ``return_report=True`` returns
        ``(score, contradictions)`` where ``contradictions`` is a list
        of ``(idx_i, idx_j, description)`` tuples describing cross-truth
        part-of relations with opposite sign polarity.

        Bivector path (even last dim):
        - Scalar (default): 1 - (fraction of (truth, concept) slots that
          are BOTH-hot within a single truth). Anti-parallel cosine is
          NOT a contradiction; under bivector encoding A and not(A)
          land on orthogonal paired indices.
        - Report path: ignores within-truth BOTH (valid catuṣkoṭi) and
          instead emits one entry per cross-truth pair (i, j) where
          ``max(basis.part(i, j), basis.part(j, i)) >= part_threshold``
          and the two truths have opposite sign polarity (one mostly
          positive-pole, the other mostly negative-pole). When
          ``basis`` is omitted, a structural proxy via positive-pole
          overlap ratio is used so the caller still gets a report.

        Legacy path (odd last dim): two truths pointing in opposite
        directions (anti-parallel cosine) represent a contradiction;
        the report path simply emits those anti-parallel pairs.

        Args:
            sim_threshold: legacy cosine threshold for anti-parallel
                truths to count as conflicting.
            pair_threshold: bivector threshold above which both poles
                of a concept are considered co-active (BOTH).
            basis: optional Basis with a ``part(a, b, monotonic=True)``
                method. Used only by the report path; omit for the
                scalar default.
            part_threshold: minimum part-of score to count (i, j) as a
                containment relation under the report path.
            return_report: when True, return ``(score, contradictions)``.

        Returns:
            Scalar in [0, 1] where 1 = fully consistent, OR a tuple
            ``(score, contradictions)`` when ``return_report=True``.
        """
        n = self.count.item()
        if n < 1:
            score = torch.tensor(1.0, device=self.truths.device)
            return (score, []) if return_report else score

        stored = self.truths[:n]
        bivector = stored.shape[-1] % 2 == 0

        if return_report:
            contradictions = self._detect_contradictions(
                stored, basis, part_threshold, sim_threshold
            )
            n_pairs = max(1, (n * (n - 1)) // 2)
            score = torch.tensor(
                float(max(0.0, 1.0 - len(contradictions) / n_pairs)),
                device=self.truths.device,
            )
            return score, contradictions

        # Scalar back-compat path.
        if bivector:
            pos = self._positive_poles(stored)
            neg = self._negative_poles(stored)
            both_hot = (pos > pair_threshold) & (neg > pair_threshold)
            total = both_hot.numel()
            if total == 0:
                return torch.tensor(1.0, device=self.truths.device)
            frac = both_hot.float().mean()
            return (1.0 - frac).clamp(0.0, 1.0).to(self.truths.device)

        # Legacy pairwise anti-parallel check.
        if n < 2:
            return torch.tensor(1.0, device=self.truths.device)

        norms = stored.norm(dim=-1)
        valid = norms > 1e-6
        if valid.sum() < 2:
            return torch.tensor(1.0, device=self.truths.device)

        directions = F.normalize(stored[valid], dim=-1)
        sim_matrix = directions @ directions.T
        m = directions.shape[0]

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

    def _detect_contradictions(self, stored, basis, part_threshold,
                               sim_threshold):
        """Return List[(i, j, description)] of cross-truth contradictions.

        Bivector: pair (i, j) is a contradiction when one is a
        part-of the other (``max(part(i, j), part(j, i)) >=
        part_threshold``) and the two have opposite sign polarity.
        Legacy (odd last dim): pair is a contradiction when cosine
        similarity is below ``-sim_threshold``.
        """
        n = stored.shape[0]
        contradictions = []
        if n < 2:
            return contradictions

        bivector = stored.shape[-1] % 2 == 0

        if bivector:
            for i in range(n):
                for j in range(i + 1, n):
                    p_ij = self._part_score(stored[i], stored[j], basis)
                    p_ji = self._part_score(stored[j], stored[i], basis)
                    score = max(p_ij, p_ji)
                    if score < part_threshold:
                        continue
                    if self._sign(stored[i]) == self._sign(stored[j]):
                        continue
                    contradictions.append(
                        (i, j,
                         f"part-of (score={score:.2f}) with opposite "
                         f"sign polarity")
                    )
            return contradictions

        # Legacy: anti-parallel pairs.
        norms = stored.norm(dim=-1)
        for i in range(n):
            if norms[i] < 1e-6:
                continue
            ni = F.normalize(stored[i].unsqueeze(0), dim=-1).squeeze(0)
            for j in range(i + 1, n):
                if norms[j] < 1e-6:
                    continue
                nj = F.normalize(stored[j].unsqueeze(0), dim=-1).squeeze(0)
                sim = float((ni * nj).sum().item())
                if sim < -sim_threshold:
                    contradictions.append(
                        (i, j, f"anti-parallel (cos={sim:.2f})")
                    )
        return contradictions

    def _part_score(self, a, b, basis):
        """Scalar part-of score for (a, b). Uses basis when provided,
        otherwise a positive-pole overlap proxy on bivector storage."""
        if basis is not None and hasattr(basis, "part"):
            score = basis.part(a, b, monotonic=True, scalar=True)
            if torch.is_tensor(score):
                return float(score.mean().item())
            return float(score)
        # Structural proxy: how much of b's positive-pole energy is
        # covered by a's positive-pole energy.
        if a.shape[-1] % 2 == 0:
            a_pos = self._positive_poles(a)
            b_pos = self._positive_poles(b)
            denom = float(b_pos.abs().sum().item()) + 1e-8
            return float(torch.minimum(a_pos, b_pos).sum().item()) / denom
        denom = float(b.abs().sum().item()) + 1e-8
        return float(torch.minimum(a, b).clamp_min(0).sum().item()) / denom

    def _sign(self, v):
        """Return +1 if v is predominantly positive-pole, -1 otherwise.

        Bivector: compare summed positive-pole mass to summed
        negative-pole mass. Legacy: use the sign of the mean activation.
        """
        if v.shape[-1] % 2 == 0:
            pos = self._positive_poles(v).abs().sum().item()
            neg = self._negative_poles(v).abs().sum().item()
            return 1 if pos >= neg else -1
        return 1 if v.mean().item() >= 0 else -1

    def suggest_clarifications(self, basis=None,
                               part_threshold: float = 0.3) -> list:
        """Generate one user-facing message per detected contradiction.

        The template is fixed (for test stability and translation
        ease): ::

            "'{source_i}' (trust={trust_i}) and '{source_j}' "
            "(trust={trust_j}) appear to contradict — please revise "
            "to enable more rational thought."

        Missing source falls back to ``"(truth #i)"`` and missing trust
        to ``"unknown"``.
        """
        _, contradictions = self.consistency(
            basis=basis, part_threshold=part_threshold, return_report=True
        )
        messages = []
        sources = getattr(self, "_sources", []) or []
        trusts = getattr(self, "_trusts", []) or []
        for i, j, _desc in contradictions:
            src_i = sources[i] if i < len(sources) and sources[i] else None
            src_j = sources[j] if j < len(sources) and sources[j] else None
            trust_i = trusts[i] if i < len(trusts) and trusts[i] is not None else None
            trust_j = trusts[j] if j < len(trusts) and trusts[j] is not None else None
            label_i = f"'{src_i}'" if src_i is not None else f"(truth #{i})"
            label_j = f"'{src_j}'" if src_j is not None else f"(truth #{j})"
            t_i = f"{trust_i:g}" if trust_i is not None else "unknown"
            t_j = f"{trust_j:g}" if trust_j is not None else "unknown"
            messages.append(
                f"{label_i} (trust={t_i}) and {label_j} (trust={t_j}) "
                f"appear to contradict — please revise to enable more "
                f"rational thought."
            )
        return messages

    # -- TruthLoss: Union Norm Reduction -----------------------------

    def falsity_penalty(self, symbol_states, basis):
        """Compute additive truth loss via union norm reduction.

        For each proposition in ``symbol_states``, measure how much the
        TruthSet union norm drops when the proposition is included.
        Contradiction cancels dimensions (via ``Basis.disjunction``'s
        bitonic same-sign logic), reducing the norm -> positive penalty.

        Agreeing propositions preserve or extend the union -> no penalty.
        Unknown propositions (zero dims) pass through -> no penalty.
        DoT weighting is implicit: high-DoT truths contribute more energy
        to the union, so contradicting them causes a larger norm drop.

        Both sides of the disjunction live in symbol space by
        construction: stored truths are recorded from
        ``SymbolicSpace.forwardEnd``, and ``symbol_states`` should be
        the post-pi activations cached during the Sigma-Pi loop (the
        model's ``self.symbol_states[-1]``).  Using symbols rather
        than pre-pi concepts keeps both operands in the basis's
        native space.

        Args:
            symbol_states: (B, N, D) symbolic activations from the
                forward pass (post-pi, post-l1_proximal -- the
                committed beliefs).
            basis: Basis instance with disjunction() method.

        Returns:
            Scalar penalty >= 0 (differentiable).
        """
        n = self.count.item()
        if n == 0:
            return symbol_states.new_tensor(0.0)

        stored = self.truths[:n]  # (n, D)

        # Fold stored truths into union vector via successive disjunction
        truth_union = stored[0]
        for i in range(1, n):
            truth_union = basis.disjunction(truth_union, stored[i])
        union_norm = truth_union.norm()

        # For each proposition, compute norm reduction
        B, N, D = symbol_states.shape
        propositions = symbol_states.reshape(-1, D)  # (B*N, D)

        penalties = []
        for p in range(propositions.shape[0]):
            extended = basis.disjunction(truth_union, propositions[p])
            reduction = union_norm - extended.norm()
            penalties.append(torch.relu(reduction))

        return torch.stack(penalties).mean()

    # -- Maintenance ---------------------------------------------------

    @torch.no_grad()
    def prune(self, min_norm: float = 1e-6):
        """Remove near-zero entries (truths with DoT ~= 0).

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

    @torch.no_grad()
    def orthogonalize(self, sim_threshold: float = 0.85):
        """Remove redundant truths using luminosity as the quality measure.

        For each pair with |cosine similarity| > sim_threshold, the truth
        contributing less to luminosity is removed.  Luminosity measures
        the norm of the positive conjunction across all truths -- dropping
        a redundant entry barely changes it, while dropping a unique entry
        reduces it.

        Args:
            sim_threshold: similarity above which two truths are considered
                near-duplicates.  Default 0.85.

        Returns:
            Number of truths removed.
        """
        n = self.count.item()
        if n < 2:
            return 0

        stored = self.truths[:n]
        norms = stored.norm(dim=-1)
        valid = norms > 1e-6
        if valid.sum() < 2:
            return 0

        directions = F.normalize(stored, dim=-1)
        sim_matrix = directions @ directions.T           # (n, n)

        # Find redundant pairs (|sim| > threshold, excluding diagonal)
        remove = set()
        for i in range(n):
            if i in remove or not valid[i]:
                continue
            for j in range(i + 1, n):
                if j in remove or not valid[j]:
                    continue
                if sim_matrix[i, j].abs().item() > sim_threshold:
                    # Drop whichever contributes less to luminosity
                    lum_without_i = self._luminosity_without(i)
                    lum_without_j = self._luminosity_without(j)
                    # Keep the one whose removal hurts luminosity more
                    if lum_without_i >= lum_without_j:
                        remove.add(i)
                    else:
                        remove.add(j)

        if not remove:
            return 0

        # Compact: keep non-removed entries
        keep_mask = torch.ones(n, dtype=torch.bool, device=stored.device)
        for idx in remove:
            keep_mask[idx] = False
        kept = stored[keep_mask]
        new_n = kept.shape[0]
        self.truths[:new_n] = kept
        self.truths[new_n:] = 0
        self.count.fill_(new_n)
        return len(remove)

    def _luminosity_without(self, exclude_idx: int) -> float:
        """Luminosity with one truth excluded (for orthogonalization)."""
        n = self.count.item()
        indices = [i for i in range(n) if i != exclude_idx]
        if not indices:
            return 0.0
        subset = self.truths[indices]
        if subset.shape[-1] % 2 == 0:
            subset = self._positive_poles(subset)
        conjunction = subset.min(dim=0).values
        return torch.relu(conjunction).norm().item()

    def __len__(self):
        return self.count.item()

    def __repr__(self):
        return (f"TruthLayer(nDim={self.nDim}, "
                f"truths={self.count.item()}/{self.max_truths})")

    # -- Test ----------------------------------------------------------

    @staticmethod
    def test():
        D = 32
        tl = TruthLayer(D, max_truths=64)
        assert len(tl) == 0

        # Record truths -- DoT is baked into the stored activation
        t1 = torch.randn(D)
        t2 = torch.randn(D)
        idx1 = tl.record(t1, degree=0.9)
        idx2 = tl.record(t2, degree=-0.7)
        assert len(tl) == 2

        # Stored vector = activation * degree
        assert torch.allclose(tl.truths[0], t1 * 0.9, atol=1e-6)
        assert torch.allclose(tl.truths[1], t2 * -0.7, atol=1e-6)

        # Query -- exact match (high similarity)
        result = tl.query(t1, threshold=0.8)
        assert result is not None
        assert result[0] == 0
        assert result[1] > 0  # consonant with positive truth

        # Query -- no match for random vector
        result = tl.query(torch.randn(D), threshold=0.99)
        assert result is None

        # Field projection
        concepts = torch.randn(2, 8, D)
        f = tl.field(concepts)
        assert f.shape == (2, 8)
        assert f.min() >= -1.0 and f.max() <= 1.0

        # Prune near-zero (DoT ~= 0 produces near-zero stored vector)
        tl.record(torch.randn(D), degree=0.0)
        assert len(tl) == 3
        tl.prune(min_norm=1e-6)
        assert len(tl) == 2

        # -- Luminosity --------------------------------------------
        # Luminosity = ||min(truths)|| -- conjunction then norm
        tl2 = TruthLayer(D, max_truths=64)
        # All-positive truth -> luminosity is the norm of the min
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

        # Empty truth layer -> luminosity = 0
        tl_empty = TruthLayer(D, max_truths=64)
        assert tl_empty.luminosity().item() == 0.0

        # -- Consistency -------------------------------------------
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

        # Empty / single truth -> consistent
        tl4 = TruthLayer(D, max_truths=64)
        assert tl4.consistency().item() == 1.0
        tl4.record(torch.randn(D), degree=0.5)
        assert tl4.consistency().item() == 1.0

        # -- Universality -----------------------------------------
        # PiLayer maps activations [B, N] -> [B, nSym].
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

        # Mock symbolic space: PiLayer maps [B, N] -> [B, nSym]
        class _MockSS:
            pass
        mock_ss = _MockSS()
        mock_ss.layer = PiLayer(N, nSym, monotonic=True, invertible=True,
                                nonlinear=True)
        lifting = LiftingLayer(nVerbs=8, nDim=D)

        u_score = tl5.universality(S, V, O, lifting, mock_ss)
        assert isinstance(u_score, torch.Tensor), "universality should return tensor"
        # Score should be finite
        assert torch.isfinite(u_score), f"universality NaN/Inf: {u_score}"
        # Truth store should be restored
        assert len(tl5) == 1, f"truth store not restored: {len(tl5)}"

        print("TruthLayer tests passed.")


class InterSentenceLayer(Layer):
    """Inter-sentence substrate: per-sentence ``[S | W]`` snapshots
    scored by a contrastive dual-force cosine loss.

    **What it is.** WordSpace clears at every sentence boundary -- its
    buffer is intentionally per-sentence. Inter-sentence coherence
    needs something one tier higher: a muxed view of both what the
    sentence committed (SymbolicSpace's final ``materialize()``) and
    how it was said (WordSpace's ``read()`` buffer). DiscourseSpace
    is the "buffer of snapshots across sentences, cleared at topic /
    discourse boundary" analog of WordSpace -- same fixed-buffer
    pattern one scale up.

    **Snapshot shape.** Each sentence produces an
    ``[n_sentence, n_dim]`` row where
    ``n_sentence = n_symbols + max_depth``. S fills rows
    ``[0 : n_symbols]``; W fills rows ``[n_symbols : n_sentence]``.
    Both inputs share the peer ``[what | where | when]`` column
    layout, so the concat along the N axis is a plain
    ``torch.cat([s, w], dim=0)`` with no per-source branching.

    **Loss.** Dual-force cosine similarity over the full flattened
    ``[n_sentence * n_dim]`` snapshot vector::

        loss(s_t) = (1 - cos(s_t, ctx_t))
                    + lambda_ * mean(cos(s_t, ctx_{t-i}) for i in 1..M)

    where ``ctx_t`` is the mean of the most recent ``context_window``
    snapshots and ``ctx_{t-i}`` are older centroids stored in a ring
    of length ``centroid_history``. Attractive pull toward the nearby
    context + repulsive push from older contexts makes
    representational collapse an unstable equilibrium at lambda_ ~= 1.01.
    Gradient flows only through the live ``s_tensor`` / ``w_tensor``
    arguments; all stored history is detached.

    Subclasses ``Layer`` (not ``Space``): DiscourseSpace has no
    SubSpace, no what/where/when basis slots, and no forward/reverse
    tensor-map contract. It lives inside ``WordSpace.layers`` so the
    Layer ergodic interface (``paramUpdate`` / ``set_sigma`` /
    ``observe_sigma`` / ``sigma_to_ergodic``) can reach any future
    learnable sub-modules via the same walk WordSpace uses for its
    SyntacticLayers. The contrastive loss itself has no learnable
    parameters.
    """

    name = "Discourse"

    def __init__(self, n_symbols, max_depth, n_dim,
                 context_window=12, centroid_history=3, lam=1.01,
                 concept_dim=None, batch=1):
        # n_sentence rows * n_dim cols is the shape of a single
        # snapshot; Layer's nInput / nOutput fields carry the
        # flattened count for any legacy consumers that read them.
        n_sentence = int(n_symbols) + int(max_depth)
        flat = n_sentence * int(n_dim)
        super().__init__(flat, flat)

        self.n_symbols = int(n_symbols)
        self.max_depth = int(max_depth)
        self.n_dim = int(n_dim)
        self.n_sentence = n_sentence
        self.snapshot_dim = flat
        # S-only flattened width: the prediction head operates on the
        # symbolic sub-block of the snapshot only.  The discourse
        # substrate (buffers, contrastive loss, snapshot history) still
        # stores the full [S | W] rows -- augmenting the predictor's
        # input with W was the redundant coupling that Task 5.3
        # removed.
        self.s_dim = self.n_symbols * self.n_dim
        self.context_window = int(context_window)
        self.centroid_history = int(centroid_history)
        self.lam = float(lam)
        # concept_dim is the target dim for the priming cast.  When
        # None, the predictor and cast are not built -- the layer
        # degrades to pure contrastive behavior (legacy path).
        self.concept_dim = int(concept_dim) if concept_dim is not None else None

        # Per-batch substrate. Microbatch refactor (Task 3): each row
        # owns an independent recent ring + prev-centroids ring +
        # counts; cross-row aggregation only happens at the loss
        # reduction.  ``ensure_batch(B)`` reallocates when B grows.
        self._batch = int(batch)

        # Recent buffer: last K = context_window snapshots per row.
        # Their per-row mean is the current attractive target.
        # Registered as a non-persistent buffer so ``.to(device)``
        # follows it without saving it in checkpoints -- the contents
        # are per-epoch transient state.
        self.register_buffer(
            "_recent",
            torch.zeros(self._batch, self.context_window,
                        self.n_sentence, self.n_dim),
            persistent=False)
        self.register_buffer(
            "_recent_count",
            torch.zeros(self._batch, dtype=torch.long),
            persistent=False)

        # Previous centroids: per row, M = centroid_history older
        # snapshot-window centroids the current sentence should be
        # repelled from.  A new centroid is folded in whenever the
        # recent buffer evicts its oldest entry.
        self.register_buffer(
            "_prev_centroids",
            torch.zeros(self._batch, self.centroid_history,
                        self.n_sentence, self.n_dim),
            persistent=False)
        self.register_buffer(
            "_prev_count",
            torch.zeros(self._batch, dtype=torch.long),
            persistent=False)

        # Learnable pieces for the AR sentence predictor.  Only built
        # when concept_dim is provided (i.e. the caller wants the
        # predictive/priming pathway).  Layer.layers carries them so
        # the ergodic walk reaches their parameters.
        self.layers = []
        self.predictor = None
        self.cast = None
        if self.concept_dim is not None:
            # Causal self-attention over [N, s_dim].  Output at the
            # final position is the predicted next S-block.  The
            # predictor consumes S only; W is the "how it was said"
            # view already captured by the per-sentence WordSpace
            # buffer, so feeding it here was redundant [S|W]
            # augmentation (Task 5.3 removed it).  Transformer mode
            # with a causal mask so every position also supplies a
            # predict-next training signal (only the last is read at
            # inference).
            #
            # s_dim can still be non-trivial (n_symbols * n_dim); a
            # bottleneck hidden dim keeps Q/K/V manageable.
            predictor_hidden = min(self.s_dim, 256)
            self.predictor = AttentionLayer(
                nInput=self.s_dim,
                nOutput=self.s_dim,
                nHidden=predictor_hidden,
                type="transformer",
                nHeads=1,
            )
            self.cast = LinearLayer(self.s_dim, self.concept_dim)
            self.layers.append(self.predictor)
            self.layers.append(self.cast)

    # -- per-batch resize ---------------------------------------------
    def ensure_batch(self, batch):
        """Resize per-row substrate to a new batch size.

        Reallocates the four ring buffers and their counts; per-row
        state is zeroed.  Stays a no-op when ``batch`` already
        matches.  Cascaded from ``WordSpace.ensure_batch`` at
        ``Models.Start`` so the body's per-(B*K) state is sized
        correctly under the microbatch refactor.
        """
        batch = int(batch)
        if batch == self._batch:
            return
        self._batch = batch
        device = self._recent.device
        dtype = self._recent.dtype
        self._recent = torch.zeros(
            batch, self.context_window, self.n_sentence, self.n_dim,
            dtype=dtype, device=device)
        self._recent_count = torch.zeros(
            batch, dtype=torch.long, device=device)
        self._prev_centroids = torch.zeros(
            batch, self.centroid_history, self.n_sentence, self.n_dim,
            dtype=dtype, device=device)
        self._prev_count = torch.zeros(
            batch, dtype=torch.long, device=device)

    # -- snapshot & history -------------------------------------------
    def _fit_rows(self, x, target_rows):
        """Pad-or-truncate ``x`` along its row axis to exactly
        ``target_rows``.  Accepts ``[rows, dim]`` (B=1 legacy) or
        ``[B, rows, dim]`` (microbatch); preserves the input rank in
        the output.

        Microbatch refactor (Task 3): the legacy ``x.mean(dim=0)``
        mean-pool is gone -- per-row state must stay distinct so the
        body's B*K rows do not corrupt each other.
        """
        is_2d = (x.ndim == 2)
        if is_2d:
            x = x.unsqueeze(0)
        B, rows, dim = x.shape
        if rows == target_rows:
            out = x
        elif rows > target_rows:
            out = x[:, :target_rows, :]
        else:
            pad = torch.zeros(
                B, target_rows - rows, dim,
                dtype=x.dtype, device=x.device)
            out = torch.cat([x, pad], dim=1)
        return out.squeeze(0) if is_2d else out

    def _fit_dim(self, x):
        """Pad-or-truncate ``x`` along its column axis to exactly
        ``self.n_dim``. Handles configs where S and W have mismatched
        n_dim (e.g. muxed vs. what-only)."""
        cur = x.shape[-1]
        if cur == self.n_dim:
            return x
        if cur > self.n_dim:
            return x[..., :self.n_dim]
        return F.pad(x, (0, self.n_dim - cur))

    def _assemble(self, s_tensor, w_tensor):
        """Assemble an ``[S | W]`` row from S + W tensors.

        Returns ``[n_sentence, n_dim]`` for 2D inputs (B=1 legacy
        path) and ``[B, n_sentence, n_dim]`` for 3D inputs.
        """
        s = self._fit_rows(s_tensor, self.n_symbols)
        w = self._fit_rows(w_tensor, self.max_depth)
        s = self._fit_dim(s)
        w = self._fit_dim(w)
        # Concat on the row axis: dim=0 for 2D, dim=-2 for 3D both work.
        return torch.cat([s, w], dim=-2)

    def _recent_centroid(self, b=None):
        """Mean of the recent-buffer entries currently in use, or
        ``None`` when the buffer is empty.  Returned tensor is
        detached (it comes from the stored non-persistent buffer).

        With ``b=None`` and ``self._batch == 1``, returns
        ``[n_sentence, n_dim]`` (legacy shape).  With explicit ``b``
        or batched layer, returns the per-row centroid for that row
        as ``[n_sentence, n_dim]``, or ``None`` if row b is empty.
        """
        if b is None:
            if self._batch != 1:
                raise ValueError(
                    "_recent_centroid(b=None) only legal for batch=1; "
                    "use _recent_centroid(b) for per-row layers")
            n = int(self._recent_count[0].item())
            if n == 0:
                return None
            return self._recent[0, :n].mean(dim=0)
        n = int(self._recent_count[b].item())
        if n == 0:
            return None
        return self._recent[b, :n].mean(dim=0)

    def snapshot(self, s_tensor, w_tensor, mask=None):
        """Commit a ``[S | W]`` snapshot to history.

        Ring semantics (per row): when the recent buffer is full, fold
        its current centroid into ``_prev_centroids`` (so the repulsive
        force has fresh fodder), then shift the recent ring left and
        append the new row at the end.  Stored tensors are detached.

        Inputs may be 2D ``[rows, dim]`` (B=1 legacy) or 3D
        ``[B, rows, dim]``.  ``mask`` is an optional ``[B] bool``
        selecting which rows ended a sentence this step (defaults to
        all True).  Mask is ignored for 2D inputs.

        Backward compat: if the layer is sized at B=1 and a 3D input
        with batch B>1 is given, the rows are mean-pooled into the
        single stored row.  Production sizes the layer to B*K via
        ``ensure_batch`` at ``Models.Start`` (Task 9), making the
        mean-pool path obsolete in the post-cutover world.
        """
        row = self._assemble(s_tensor, w_tensor).detach()
        if row.ndim == 2:
            # B=1 legacy: route through row 0.
            self._snapshot_row(0, row)
            return
        B = row.shape[0]
        if self._batch == 1 and B != 1:
            # Legacy fallback: collapse the input batch into the single
            # stored row.  Used pre-Task-9 when the model still hasn't
            # cascaded ensure_batch through Start().
            self._snapshot_row(0, row.mean(dim=0))
            return
        assert B == self._batch, (
            f"snapshot row-batch {B} != layer batch {self._batch}; "
            "call ensure_batch first")
        if mask is None:
            for b in range(B):
                self._snapshot_row(b, row[b])
        else:
            for b in range(B):
                if bool(mask[b]):
                    self._snapshot_row(b, row[b])

    def _snapshot_row(self, b, row):
        """Push one already-assembled row into row ``b``'s rings."""
        n = int(self._recent_count[b].item())
        if n < self.context_window:
            self._recent[b, n] = row
            self._recent_count[b] = n + 1
            return
        # Recent buffer is full: snapshot its centroid into the
        # prev_centroids ring before evicting the oldest row.
        ctx = self._recent[b].mean(dim=0).detach()
        m = int(self._prev_count[b].item())
        if m < self.centroid_history:
            self._prev_centroids[b, m] = ctx
            self._prev_count[b] = m + 1
        else:
            self._prev_centroids[b, :-1] = self._prev_centroids[b, 1:].clone()
            self._prev_centroids[b, -1] = ctx
        self._recent[b, :-1] = self._recent[b, 1:].clone()
        self._recent[b, -1] = row

    def reset(self):
        """Clear both rings on every row (epoch boundary)."""
        self._recent.zero_()
        self._recent_count.zero_()
        self._prev_centroids.zero_()
        self._prev_count.zero_()

    def __len__(self):
        """Count of recent entries.  For B=1 returns the single row's
        count (legacy); for B>1 returns the max across rows so a
        non-empty layer reports ``len > 0``.
        """
        if self._batch == 1:
            return int(self._recent_count[0].item())
        return int(self._recent_count.max().item())

    def latest_snapshot(self, b=None):
        """Return the most recently pushed snapshot.

        With ``b=None`` and ``self._batch == 1``, returns
        ``[n_sentence, n_dim]`` (legacy shape) or ``None`` if empty.
        With ``b`` provided, returns row ``b``'s latest snapshot or
        ``None`` if that row is empty.
        """
        if b is None:
            if self._batch != 1:
                raise ValueError(
                    "latest_snapshot(b=None) only legal for batch=1; "
                    "use latest_snapshot(b) for per-row layers")
            n = int(self._recent_count[0].item())
            if n == 0:
                return None
            return self._recent[0, n - 1]
        n = int(self._recent_count[b].item())
        if n == 0:
            return None
        return self._recent[b, n - 1]

    def split(self, snapshot):
        """Split a ``[n_sentence, n_dim]`` snapshot into its
        ``(S, W)`` parts.
        """
        s = snapshot[:self.n_symbols]
        w = snapshot[self.n_symbols:]
        return s, w

    # -- contrastive loss ---------------------------------------------
    def contrastive_loss(self, s_tensor, w_tensor):
        """Dual-force contrastive loss over the full flattened
        ``[n_sentence * n_dim]`` snapshot vector.

        - Attractive: ``1 - cos(current, recent_centroid)``
        - Repulsive:  ``lam * mean(cos(current, prev_centroid_i))``

        Inputs may be 2D ``[rows, dim]`` (B=1 legacy) or 3D
        ``[B, rows, dim]``.  The 3D path computes per-row losses and
        returns the mean over rows that have context (rows whose
        ``_recent_count`` is zero are skipped).  Returns ``None`` when
        no row has any context to contrast against.

        Gradient flows through the live ``s_tensor`` / ``w_tensor``
        arguments; all stored history is detached.
        """
        current = self._assemble(s_tensor, w_tensor)
        if current.ndim == 2:
            ctx = self._recent_centroid()
            if ctx is None:
                return None
            return self._contrastive_one_row(0, current, ctx)
        B = current.shape[0]
        if self._batch == 1 and B != 1:
            # Legacy fallback: pool the input batch and score against
            # the single stored row.  Used pre-Task-9.
            current_pooled = current.mean(dim=0)
            ctx = self._recent_centroid()
            if ctx is None:
                return None
            return self._contrastive_one_row(0, current_pooled, ctx)
        assert B == self._batch, (
            f"contrastive_loss row-batch {B} != layer batch "
            f"{self._batch}; call ensure_batch first")
        per_row = []
        for b in range(B):
            ctx_b = self._recent_centroid(b)
            if ctx_b is None:
                continue
            per_row.append(self._contrastive_one_row(b, current[b], ctx_b))
        if not per_row:
            return None
        stacked = torch.stack(per_row)
        return stacked.mean()

    def _contrastive_one_row(self, b, current, ctx):
        """Compute attractive + repulsive contrastive scalar for row b.

        ``current`` is ``[n_sentence, n_dim]`` and ``ctx`` is the
        already-fetched per-row centroid for row b.
        """
        current_flat = current.reshape(-1)
        ctx_flat = ctx.reshape(-1)
        attractive = 1.0 - F.cosine_similarity(
            current_flat.unsqueeze(0), ctx_flat.unsqueeze(0))
        m = int(self._prev_count[b].item())
        if m > 0:
            prev = self._prev_centroids[b, :m].reshape(m, -1)
            sims = F.cosine_similarity(
                current_flat.unsqueeze(0), prev, dim=-1)
            repulsive = sims.mean()
        else:
            repulsive = torch.tensor(
                0.0, device=current_flat.device, dtype=current_flat.dtype)
        return attractive.squeeze() + self.lam * repulsive

    # -- AR sentence prediction ---------------------------------------
    def _causal_mask(self, n, device):
        """Lower-triangular bool mask [1, n, n]: True = attend."""
        m = torch.ones(n, n, dtype=torch.bool, device=device).tril_()
        return m.unsqueeze(0)

    def predict(self, b=None):
        """Run the AR predictor over the recent buffer.

        With ``b=None``: behaves as before for B=1 layers (returns
        ``([s_dim], scalar)`` legacy shape).  For B>1 layers, returns
        per-row stacked tensors ``([B, s_dim], [B])`` -- rows whose
        recent buffer is empty get zero predictions/confidences.
        Returns ``(None, None)`` when no row has any context (or when
        the predictor isn't built).

        With explicit ``b``: returns ``([s_dim], scalar)`` for that row,
        or ``(None, None)`` if row b's recent buffer is empty.

        The predictor consumes S only -- the W block of each stored
        row is sliced off before attention so the head is not coupled
        to buffer-surface features (Task 5.3).  Confidence is derived
        from the last-position attention entropy (focused attention =
        high confidence).
        """
        if self.predictor is None:
            return None, None
        if b is not None:
            return self._predict_row(b)
        if self._batch == 1:
            return self._predict_row(0)
        # Batched: per-row stack so callers don't need a Python loop.
        per_pred = []
        per_conf = []
        any_valid = False
        device = self._recent.device
        dtype = self._recent.dtype
        zero_p = torch.zeros(self.s_dim, device=device, dtype=dtype)
        zero_c = torch.zeros((), device=device, dtype=dtype)
        for bi in range(self._batch):
            p, c = self._predict_row(bi)
            if p is None:
                per_pred.append(zero_p)
                per_conf.append(zero_c)
            else:
                per_pred.append(p)
                per_conf.append(c)
                any_valid = True
        if not any_valid:
            return None, None
        return torch.stack(per_pred), torch.stack(per_conf)

    def _predict_row(self, b):
        """Single-row predictor: returns ``([s_dim], scalar)`` or
        ``(None, None)`` for an empty row."""
        n = int(self._recent_count[b].item())
        if n == 0:
            return None, None
        # _recent[b] rows are [n_sentence, n_dim] with S in
        # [0:n_symbols].  Slice out the S rows and flatten to feed
        # the narrower predictor.
        s_rows = self._recent[b, :n, :self.n_symbols, :]
        seq = s_rows.reshape(n, self.s_dim).unsqueeze(0).detach().clone()
        self.predictor.set_mask(self._causal_mask(n, seq.device))
        try:
            out = self.predictor(seq)              # [1, n, s_dim]
        finally:
            # Clear the mask so the shared instance doesn't leak a
            # stale mask into any other caller.
            self.predictor.set_mask(None)
        predicted = out[0, -1]                      # [s_dim]
        attn = getattr(self.predictor, 'last_attn', None)
        if attn is None:
            confidence = torch.tensor(0.0, device=predicted.device)
        else:
            last_row = attn[0, 0, -1]               # [n]
            eps = 1e-12
            h = -(last_row * (last_row + eps).log()).sum()
            # Normalize by log(max(n, 2)) to avoid divide-by-zero at
            # n=1.  At n=1 the attention is a point mass so h=0 and
            # confidence is 1 regardless of the denominator; the
            # guard just keeps the arithmetic finite.
            denom = float(torch.log(torch.tensor(max(n, 2),
                dtype=last_row.dtype, device=last_row.device)))
            confidence = (1.0 - h / denom).clamp(0.0, 1.0)
        return predicted, confidence

    def predictive_loss(self, s_tensor, w_tensor, predicted):
        """Cosine-distance loss between the AR-predicted S-block and
        the actual S-slice of the snapshot assembled from the live
        ``(s_tensor, w_tensor)`` pair.

        Inputs may be 2D ``[rows, dim]`` (B=1 legacy, ``predicted``
        shape ``[s_dim]``) or 3D ``[B, rows, dim]`` (``predicted``
        shape ``[B, s_dim]``).  The 3D path returns the mean of the
        per-row cosine distances.

        The predictor consumes S only (Task 5.3), so the comparison
        is also S-only: the W rows of ``_assemble`` are still used by
        the contrastive loss, but this loss only scores the symbolic
        block that the head actually predicted.

        Returns ``None`` when ``predicted`` is ``None`` (first sentence
        cold-start, or predictor disabled).  Gradient flows through
        both ``predicted`` (via the predictor's parameters) and the
        live ``s_tensor`` arguments; ``w_tensor`` is still accepted
        for symmetry with the contrastive API but does not contribute
        to this loss.
        """
        if predicted is None:
            return None
        actual = self._assemble(s_tensor, w_tensor)
        if actual.ndim == 2:
            # Legacy B=1: [n_sentence, n_dim] -> [s_dim]
            actual_flat = actual[:self.n_symbols].reshape(-1)
            pred_flat = predicted.reshape(-1)
            sim = F.cosine_similarity(
                pred_flat.unsqueeze(0), actual_flat.unsqueeze(0))
            return (1.0 - sim).squeeze()
        B = actual.shape[0]
        if self._batch == 1 and B != 1:
            # Legacy fallback: pool the input batch into one row and
            # compare against the (single) stored prediction.
            actual_pooled = actual.mean(dim=0)
            actual_flat = actual_pooled[:self.n_symbols].reshape(-1)
            pred_flat = predicted.reshape(-1)
            sim = F.cosine_similarity(
                pred_flat.unsqueeze(0), actual_flat.unsqueeze(0))
            return (1.0 - sim).squeeze()
        # 3D batched: [B, n_sentence, n_dim] -> [B, s_dim]
        actual_flat = actual[:, :self.n_symbols, :].reshape(B, -1)
        if predicted.ndim == 1:
            # Single prediction broadcast across the batch.
            pred_flat = predicted.reshape(1, -1).expand(B, -1)
        else:
            pred_flat = predicted.reshape(B, -1)
        sim = F.cosine_similarity(pred_flat, actual_flat, dim=-1)  # [B]
        return (1.0 - sim).mean()

    def prime(self, predicted, confidence, scale):
        """Cast a predicted snapshot into concept space and gate it.

        Returns ``cast(predicted) * confidence * scale``.  Shape is
        ``[concept_dim]`` for 1D ``predicted`` (single row) and
        ``[B, concept_dim]`` for 2D ``predicted`` (batched
        per-row predictions).  Returns ``None`` when inputs are
        missing or the cast is not built.
        """
        if (self.cast is None
                or predicted is None
                or confidence is None):
            return None
        if predicted.ndim == 1:
            # cast expects a 2D [B, in]; wrap and unwrap the singleton.
            cast_out = self.cast(predicted.unsqueeze(0)).squeeze(0)
            return cast_out * confidence * float(scale)
        # Batched: cast operates on [B, s_dim] -> [B, concept_dim].
        cast_out = self.cast(predicted)
        # Confidence is [B]; broadcast over concept_dim.
        return cast_out * confidence.unsqueeze(-1) * float(scale)


class ChunkLayer(Layer):
    """Learned BPE-style codebook for perceptual chunking.

    Each entry stores a merge prototype (what the pair looks like) and a
    split prototype (the two constituents).  Forward scores adjacent pairs
    against the codebook; reverse looks up the entry to reconstruct.

    The entropic gate merges a pair only when the codebook match score
    exceeds a learned threshold -- i.e. when encoding the pair as a single
    chunk saves more bits than it costs.

    Merging stops at word boundaries (whitespace characters) so that chunks
    never cross word edges.  In the byte stream the space character (0x20)
    and common whitespace bytes serve as boundary markers.
    """

    def __init__(self, nDim, bpe=False,
                 n_vectors=1024, chunking_frequency=2):
        super().__init__(nDim, nDim)
        self.nDim = nDim
        # -- BPE state (active only when ``bpe`` is True) ------------------
        # Cold-start: empty merges table, vocab seeded with 256 single-byte
        # ids so byte_value == chunk_id for the 0..255 range.  This mirrors
        # the Embedding.create() placement of bytes at codebook indices
        # 0..255 (see Spaces.py -- ``byte_value == codebook_index``).
        self.bpe = bool(bpe)
        self.n_vectors = int(n_vectors)
        self.chunking_frequency = int(chunking_frequency)
        self.merges = []            # list[tuple[int, int]] in insertion order
        self.vocab = {}             # dict[tuple[int,...], int]
        self.id_to_bytes = {}       # dict[int, tuple[int,...]]
        for i in range(256):
            key = (i,)
            self.vocab[key] = i
            self.id_to_bytes[i] = key
        self._next_id = 256
        self._max_merge_len = 1

    # -- Boundary detection --------------------------------------------

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

    # -- BPE forward + training ----------------------------------------

    def forward(self, byte_indices):
        """Greedy longest-match BPE encoding of a batch of byte sequences.

        Active only in BPE mode (``self.bpe == True``); in legacy mode
        this is an identity pass-through so existing downstream callers
        keep their previous semantics.

        Args:
            byte_indices: ``[B, N]`` long tensor of byte values 0..255.
                A zero entry terminates the row (padding sentinel).
        Returns:
            When ``bpe`` is True:
                ``(chunks, spans)`` -- ``chunks`` is ``list[list[int]]`` of
                chunk ids per row; ``spans`` is ``list[list[(start, end,
                key)]]`` where ``key`` is the byte-tuple backing that
                chunk.  ``end`` is inclusive.
            When ``bpe`` is False:
                ``byte_indices`` unchanged.
        """
        if not self.bpe:
            return byte_indices
        if byte_indices.dim() != 2:
            raise ValueError(
                f"ChunkLayer.forward expects [B, N] byte indices, got "
                f"{tuple(byte_indices.shape)}")
        B, N = byte_indices.shape
        rows = byte_indices.tolist()
        all_chunks = []
        all_spans = []
        for b in range(B):
            row = rows[b]
            chunks = []
            spans = []
            i = 0
            while i < N:
                bval = row[i]
                if bval == 0:
                    break
                matched_key = None
                matched_len = 1
                upper = min(self._max_merge_len, N - i)
                for L in range(upper, 0, -1):
                    key = tuple(row[i:i + L])
                    if key in self.vocab:
                        matched_key = key
                        matched_len = L
                        break
                if matched_key is None:
                    matched_key = (bval,)
                    matched_len = 1
                chunk_id = self.vocab[matched_key]
                chunks.append(chunk_id)
                spans.append((i, i + matched_len - 1, matched_key))
                i += matched_len
            all_chunks.append(chunks)
            all_spans.append(spans)
        return all_chunks, all_spans

    def train_step(self, byte_indices, k_merges=1):
        """Learn up to ``k_merges`` new BPE merges from pair frequencies.

        Encodes the batch with the current merge table, counts adjacent
        ``(id_i, id_{i+1})`` pair frequencies across all rows, and
        promotes the top-k pairs whose count meets
        ``chunking_frequency`` into new vocab entries.  Returns the
        number of new merges added (0 when none qualify).

        Idempotent w.r.t. vocab size: stops once ``len(vocab)`` reaches
        ``n_vectors``.  No-op in legacy (``bpe=False``) mode.
        """
        if not self.bpe:
            return 0
        if len(self.vocab) >= self.n_vectors:
            return 0
        all_chunks, _ = self.forward(byte_indices)
        from collections import Counter
        counts = Counter()
        for chunks in all_chunks:
            for a, b in zip(chunks, chunks[1:]):
                counts[(a, b)] += 1
        if not counts:
            return 0
        added = 0
        for pair, freq in counts.most_common(k_merges):
            if freq < self.chunking_frequency:
                break
            if len(self.vocab) >= self.n_vectors:
                break
            left_bytes = self.id_to_bytes.get(pair[0])
            right_bytes = self.id_to_bytes.get(pair[1])
            if left_bytes is None or right_bytes is None:
                continue
            new_key = left_bytes + right_bytes
            if new_key in self.vocab:
                continue
            new_id = self._next_id
            self._next_id += 1
            self.vocab[new_key] = new_id
            self.id_to_bytes[new_id] = new_key
            self.merges.append(pair)
            if len(new_key) > self._max_merge_len:
                self._max_merge_len = len(new_key)
            added += 1
        return added

    # -- Byte-mode hard merge + compaction -----------------------------

    def hard_merge_spans(self, data, byte_indices):
        """Merge contiguous byte slots into spans via mean aggregation.

        Dispatches on ``self.bpe``:
          - ``bpe=False`` (legacy): whitespace-boundary spans via
            ``BOUNDARY_BYTES``.
          - ``bpe=True``: spans come from the learned BPE merge table
            (greedy longest-match via ``self.forward``).  When training,
            runs a single ``train_step`` first so the merge table grows
            from this batch's pair statistics.

        Args:
            data: ``[B, N, D]`` byte-level vectors.
            byte_indices: ``[B, N]`` long -- byte values 0..255.
        Returns:
            ``(data, span_meta)`` where ``data`` has span-start slots
            holding mean vectors (rest zeroed) and ``span_meta`` is
            ``list[list[(start, end, original_vectors)]]`` per batch row.
        """
        if self.bpe:
            if self.training:
                self.train_step(byte_indices)
            return self._hard_merge_spans_bpe(data, byte_indices)

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
                original = data[b, start:end + 1].clone()
                data[b, start] = data[b, start:end + 1].mean(dim=0)
                if end > start:
                    data[b, start + 1:end + 1] = 0.0
                spans.append((start, end, original))
            span_meta.append(spans)
        return data, span_meta

    def _hard_merge_spans_bpe(self, data, byte_indices):
        """BPE-mode span construction.

        Each BPE chunk produced by ``self.forward`` becomes one span;
        the span-start slot holds the mean of the span's byte vectors
        and the trailing slots are zeroed, matching the layout that
        ``compact`` / ``uncompact`` expect.  The original pre-merge
        byte vectors are stored in ``span_meta`` so reconstruction via
        ``uncompact`` is exact -- the merge table is used only for
        forward segmentation, never for decoding bytes.
        """
        B, N, D = data.shape
        data = data.clone()
        _, raw_spans = self.forward(byte_indices)
        span_meta = []
        for b in range(B):
            spans = []
            for start, end, _key in raw_spans[b]:
                if end >= N:
                    end = N - 1
                original = data[b, start:end + 1].clone()
                data[b, start] = data[b, start:end + 1].mean(dim=0)
                if end > start:
                    data[b, start + 1:end + 1] = 0.0
                spans.append((start, end, original))
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
            span_meta: list[list[(start, end)]] or
                list[list[(start, end, original_vectors)]] per batch
            where_encoding: WhereEncoding instance (for sin/cos rewrite)
        Returns:
            dense: [B, nWordSlots, D]
            compact_map: list[list[(dense_idx, start, end, original_vectors)]] -- for reverse
        """
        B, N, D = data.shape
        dense = torch.zeros(B, nWordSlots, D, device=data.device)
        compact_map = []
        for b in range(B):
            mapping = []
            for dense_idx, span in enumerate(span_meta[b]):
                if dense_idx >= nWordSlots:
                    break
                start, end = span[0], span[1]
                original = span[2] if len(span) > 2 else None
                dense[b, dense_idx] = data[b, start]
                mapping.append((dense_idx, start, end, original))
            compact_map.append(mapping)

        # Overwrite where-encoding dims with span start byte offset
        if where_encoding is not None and where_encoding.nDim > 0:
            where_idx = [D + i for i in where_encoding.index]  # e.g. [-4,-3] -> absolute
            for b in range(B):
                for dense_idx, start, _end, _original in compact_map[b]:
                    pos_enc = where_encoding.encode(float(start))  # [2] sin/cos
                    dense[b, dense_idx, where_idx] = pos_enc

        return dense, compact_map

    def uncompact(self, dense, compact_map, nByteSlots):
        """Scatter dense word vectors back to byte positions (span copy)."""
        B, _, D = dense.shape
        data = torch.zeros(B, nByteSlots, D, device=dense.device)
        for b in range(B):
            for item in compact_map[b]:
                dense_idx, start, end = item[:3]
                original = item[3] if len(item) > 3 else None
                span_len = end - start + 1
                if original is not None:
                    data[b, start:end + 1] = original.to(device=dense.device, dtype=dense.dtype)
                else:
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
                 conceptualOrder=0,
                 nWhere=None, nWhen=None):
        super().__init__()
        self.reverse_scale = float(reverse_scale or 0.5)
        self.what_scale = float(what_scale or 0.7)
        self.where_scale = float(where_scale or 0.2)
        self.when_scale = float(when_scale or 0.1)
        self.embedding_scale = float(embedding_scale or 0.1)
        # Resolve nWhere: prefer explicit arg, then architecture-level,
        # then fall back to InputSpace-level (per-space nWhere overrides arch).
        if nWhere is not None and nWhere > 0:
            self.nWhere = nWhere
        else:
            arch_nw = TheXMLConfig.get("architecture.nWhere")
            if arch_nw and arch_nw > 0:
                self.nWhere = arch_nw
            else:
                try:
                    self.nWhere = TheXMLConfig.space("InputSpace", "nWhere")
                except KeyError:
                    self.nWhere = 0
        if nWhen is not None and nWhen > 0:
            self.nWhen = nWhen
        else:
            arch_nn = TheXMLConfig.get("architecture.nWhen")
            if arch_nn and arch_nn > 0:
                self.nWhen = arch_nn
            else:
                try:
                    self.nWhen = TheXMLConfig.space("InputSpace", "nWhen")
                except KeyError:
                    self.nWhen = 0

        if certainty:
            self.output_criterion = CertaintyWeightedCrossEntropy()
        elif nOutput <= 2:
            self.output_criterion = nn.MSELoss()
        elif conceptualOrder > 0:
            self.output_criterion = nn.MSELoss()
        else:
            self.output_criterion = nn.CrossEntropyLoss()

    def output(self, pred, target):
        return self.output_criterion(pred, target)

    def compute(self, pred, target):
        """Per-slot MSE with what/where/when weighting (legacy).

        Used when nWhere == 0 (no positional encoding).  When nWhere > 0,
        the training loop calls ``compute_piecewise`` instead.
        """
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

    def compute_piecewise(self, pred, target):
        """Piecewise reconstruction loss via Chamfer distance.

        Instead of per-slot MSE(pred[v], target[v]), uses bidirectional
        nearest-neighbour matching:

        - **Accuracy**: each predicted token matches its nearest original.
        - **Coverage**: each original token is covered by some prediction.

        This handles token reordering (butterfly merge may swap vector
        positions) and eliminates error shadowing when tokens overlap in
        position space.  The what/where/when component weights are applied
        before computing L2 distances so the matching respects the
        relative importance of content vs position vs time.
        """
        embSize = pred.shape[-1]
        nWhere = self.nWhere
        nWhen = self.nWhen
        nWhat = embSize - nWhere - nWhen

        # Build per-dim weight vector: sqrt so that L2 distance^2 = weighted MSE
        w = pred.new_ones(embSize)
        if nWhat > 0:
            w[:nWhat] = self.what_scale
        if nWhere > 0:
            w[nWhat:nWhat + nWhere] = self.where_scale
        if nWhen > 0:
            w[nWhat + nWhere:] = self.when_scale
        w = w.sqrt()

        p = pred * w
        t = target * w

        # Ensure 3-D [B, N, D]
        if p.dim() == 2:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)

        # Pairwise squared L2 distances: [B, N_pred, N_target]
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a*b)   (MPS-safe, no cdist)
        p_sq = (p * p).sum(dim=-1, keepdim=True)           # [B, N, 1]
        t_sq = (t * t).sum(dim=-1, keepdim=True)           # [B, N, 1]
        dot = torch.bmm(p, t.transpose(1, 2))              # [B, N, N]
        dists_sq = (p_sq - 2 * dot + t_sq.transpose(1, 2)).clamp(min=0)

        # Accuracy: for each predicted token, squared distance to nearest original
        accuracy = dists_sq.min(dim=2).values.mean()
        # Coverage: for each original token, squared distance to nearest prediction
        coverage = dists_sq.min(dim=1).values.mean()

        return accuracy + coverage

    def forward(self, lossOut, lossIn=None, sbow=None):
        total = lossOut
        if lossIn is not None and torch.isfinite(lossIn).all():
            rr = self.reverse_scale
            total = (1 - rr) * lossOut + rr * lossIn
        if sbow is not None:
            total = total + self.embedding_scale * sbow
        return total

    def total(self, lossOut, lossIn=None, sbow=None):
        return self(lossOut, lossIn, sbow)

class Error:
    """Central registry for per-batch error/loss terms.

    ``Error`` is a bookkeeping client of ``Loss``: individual sites still
    compute their pred-vs-target comparisons via a ``Loss`` instance (or
    any other path that produces a scalar tensor), and then register the
    result here with a name, weight, originating space, and category.

    Why a registry? There are currently 12+ loss terms accumulated across
    four different call sites (``ModelLoss``, ``BasicModel.runBatch``,
    ``SymbolicSpace.accumulate_symbol_objective``, ``WordSpace.truth_modulated_loss``).
    Debugging convergence problems used to require grepping each site to
    answer "what fraction of today's gradient came from which term?".
    The registry makes that a one-call breakdown, and supports:

      * ``.total()``      -- weighted sum for backprop
      * ``.breakdown()``  -- per-term scalars for logging
      * ``.snapshot()`` + ``.covariance()`` -- running covariance across
                            batches, so you can detect terms that fight
                            each other (anti-correlation) or that carry
                            no signal (zero variance).
      * ``.disable(cat)`` / ``.enable(cat)`` -- one-line ablation by
                            category (``"reconstruction"``, ``"symbol"``,
                            ``"truth"``, ``"discourse"``, ``"embedding"``,
                            ``"prediction"``).

    ``Error`` never enforces specific math -- the caller decides how each
    term is computed and chooses its weight (usually from a config knob).
    The class just collects, sums, and reports.

    Usage pattern inside ``runBatch``:

        TheError.reset()
        TheError.compute("reconstruction", pred, target,
                          method="compute", weight=self.loss.reverse_scale,
                          space="InputSpace", category="reconstruction")
        TheError.add("symbol_residual", sym_term, weight=1.0,
                      space="SymbolicSpace", category="symbol")
        total = TheError.total()          # for backprop
        TheError.snapshot()                # record for covariance
    """

    _CATEGORIES = (
        "reconstruction", "prediction", "symbol",
        "truth", "discourse", "embedding", "other",
    )

    def __init__(self, loss: Loss = None, history_max: int = 1024):
        self._loss: Loss = loss
        self._terms: dict = {}   # name -> {weight, value, space, category, count}
        self._history: list = []  # each entry is {name: weighted_scalar}
        self._history_max = int(history_max)
        self._disabled: set = set()

    # ---- setup ---------------------------------------------------------

    def attach(self, loss: Loss):
        """Associate a ``Loss`` instance for ``compute()`` delegation.

        Optional: callers may use ``.add()`` directly and skip ``attach``.
        """
        self._loss = loss

    def reset(self):
        """Clear the per-batch term store.  Preserves history."""
        self._terms.clear()

    def clear(self):
        """Alias for ``reset``; used by per-subspace transit accumulators.

        When an ``Error`` instance is attached to a ``SubSpace`` as a
        pipeline-carried auxiliary-loss sink, ``runBatch`` calls
        ``clear()`` after harvesting its terms into ``TheError`` so the
        next forward pass starts fresh.
        """
        self._terms.clear()

    def terms(self):
        """Return the current terms as a list of 5-tuples.

        Each tuple is ``(name, tensor, weight, space, category)``.  Same-name
        contributions have been summed at ``add()`` time, so each unique
        name appears once.  Used by ``runBatch`` to fold per-subspace
        accumulators into the module-level ``TheError`` registry while
        preserving tensor references for autograd.
        """
        return [
            (name, rec["value"], rec["weight"], rec["space"], rec["category"])
            for name, rec in self._terms.items()
        ]

    def disable(self, category: str):
        """Exclude all terms of this category from ``.total()``."""
        if category not in self._CATEGORIES:
            warnings.warn(f"Error.disable: unknown category {category!r}; "
                          f"will still work but consider adding it to "
                          f"Error._CATEGORIES for consistency.")
        self._disabled.add(category)

    def enable(self, category: str):
        self._disabled.discard(category)

    @property
    def disabled_categories(self):
        return frozenset(self._disabled)

    # ---- accumulation --------------------------------------------------

    def add(self, name: str, value, *, weight: float = 1.0,
            space: str = None, category: str = "other"):
        """Register a pre-computed scalar term.

        Repeated ``name`` values are summed (useful when a term is
        contributed from multiple layers).  ``value`` may be ``None``
        (the call becomes a no-op) or a scalar tensor.
        """
        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        rec = self._terms.get(name)
        if rec is None:
            self._terms[name] = {
                "weight": float(weight),
                "value": value,
                "space": space,
                "category": category,
                "count": 1,
            }
        else:
            rec["value"] = rec["value"] + value
            rec["count"] += 1

    def compute(self, name: str, pred, target, *,
                weight: float = 1.0, method: str = "compute",
                space: str = None, category: str = "reconstruction"):
        """Compute a term via the attached ``Loss`` instance, then register it.

        ``method`` names a ``Loss`` method (``"compute"``,
        ``"compute_piecewise"``, ``"output"``, or a subclass extension).
        Returns the raw (unweighted) loss tensor for convenience -- the
        registry stores the unweighted value and applies ``weight`` at
        ``.total()`` time.
        """
        if self._loss is None:
            raise RuntimeError(
                "Error.compute() requires an attached Loss instance; "
                "call TheError.attach(model_loss) first."
            )
        fn = getattr(self._loss, method, None)
        if fn is None:
            raise AttributeError(
                f"Error.compute: Loss has no method {method!r}")
        value = fn(pred, target)
        self.add(name, value, weight=weight, space=space, category=category)
        return value

    # ---- aggregation / inspection --------------------------------------

    def total(self):
        """Return the weighted sum of all enabled terms (or ``None``)."""
        if not self._terms:
            return None
        total = None
        for rec in self._terms.values():
            if rec["category"] in self._disabled:
                continue
            contrib = rec["weight"] * rec["value"]
            total = contrib if total is None else total + contrib
        return total

    def breakdown(self):
        """Per-term snapshot keyed by name.

        Each entry is ``{weight, value, weighted, space, category, count}``
        where ``value`` and ``weighted`` are Python floats when the term
        is a scalar, or ``None`` for multi-element tensors.
        """
        out = {}
        for name, rec in self._terms.items():
            v_tensor = rec["value"]
            if isinstance(v_tensor, torch.Tensor) and v_tensor.numel() == 1:
                value_f = float(v_tensor.detach().item())
            else:
                value_f = None
            out[name] = {
                "weight": rec["weight"],
                "value": value_f,
                "weighted": rec["weight"] * value_f if value_f is not None else None,
                "space": rec["space"],
                "category": rec["category"],
                "count": rec["count"],
            }
        return out

    def snapshot(self):
        """Record the current weighted breakdown for covariance analysis.

        Only scalar terms are recorded; multi-element tensors are
        skipped (they contribute to ``.total()`` via backprop but not
        to the history).
        """
        snap = {}
        for name, rec in self._terms.items():
            v = rec["value"]
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                snap[name] = rec["weight"] * float(v.detach().item())
        if snap:
            self._history.append(snap)
            if len(self._history) > self._history_max:
                self._history.pop(0)

    def covariance(self, n_steps: int = None):
        """Running covariance of weighted term values across recent snapshots.

        Returns ``{"names": [...], "cov": tensor[T,T]}`` where T is the
        number of terms that appeared in the selected window.  Any term
        absent from a given snapshot is treated as zero for that step.
        """
        if not self._history:
            return {"names": [], "cov": torch.zeros(0, 0)}
        hist = self._history if n_steps is None else self._history[-n_steps:]
        names = sorted({k for s in hist for k in s.keys()})
        if not names:
            return {"names": [], "cov": torch.zeros(0, 0)}
        mat = torch.zeros(len(hist), len(names))
        for i, s in enumerate(hist):
            for j, n in enumerate(names):
                mat[i, j] = s.get(n, 0.0)
        centered = mat - mat.mean(dim=0, keepdim=True)
        denom = max(len(hist) - 1, 1)
        cov = (centered.transpose(0, 1) @ centered) / denom
        return {"names": names, "cov": cov}

    def format_breakdown(self) -> str:
        """Single-line summary of all terms, ordered by weighted magnitude."""
        bd = self.breakdown()
        rows = []
        for name, entry in bd.items():
            w = entry["weighted"]
            if w is None:
                continue
            rows.append((abs(w), name, entry["weight"], entry["value"], w,
                         entry["category"]))
        rows.sort(reverse=True)
        parts = [
            f"{name}[{cat}]={val:.4g}*{wt:.4g}={wval:.4g}"
            for _, name, wt, val, wval, cat in rows
        ]
        return " | ".join(parts) if parts else "<empty>"

# Module-level singleton.  Callers do ``from Layers import TheError`` and
# then ``TheError.reset() / .add(...) / .total()``.  A single ``Error``
# instance is kept at module scope so every space and every loss term
# registers into the same bookkeeping store per process.
TheError = Error()

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
