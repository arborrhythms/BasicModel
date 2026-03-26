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
import util as _util
_util.init_runtime_env()
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


epsilon = 1e-7  # to avoid log(0)

# Device used by all layers — delegates to util.
from util import TheXMLConfig

def _get_device():
    return _util.TheDevice

def init_device(device):
    """Update the device for all layers. Kept for backward compat."""
    _util.init_device(device)

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

from util import TheMessage

#region Layers
class _AutoDevice(type(nn.Module)):
    """Metaclass that moves nn.Modules to TheDevice after __init__."""
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        return instance.to(_get_device())

class Layer(nn.Module, metaclass=_AutoDevice):
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
            self.noise  = sample_noise(self.W)
            if self.hasBias:
                self.biasNoise = sample_noise(self.biasWeight)

    def compute_W_current(self):
        """Effective W for outer-product use (respects ergodic noise if active)."""
        if self.ergodic:
            return self.bias * self.W + self.var * self.noise
        return self.W

    def forward(self, x):
        if self.ergodic:
            self.resample_noise()
            output = x @ (self.bias * self.W + self.var * self.noise)
        else:
            output = x @ self.W
        return output
    def forwardBias(self, x):
        if self.hasBias:
            if self.ergodic:
                x = x + self.bias * self.biasWeight + self.var * self.biasNoise
            else:
                x = x + self.biasWeight
        return x

    @staticmethod
    def test():
        nInput, nOutput = 3, 4
        W = torch.rand(nInput, nOutput)
        layer = LinearLayer(nInput=nInput, nOutput=nOutput, W = W)
        input = torch.rand((1, nInput))
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
        """Non-ergodic sequential forward using clean L, d_effective, U."""
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
        device = _get_device()
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
        id_err = torch.norm(W @ W_inv - torch.eye(5))
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

    All ergodic machinery lives in the inner layer; SigmaLayer dispatches
    the ergodic interface (set_sigma, observe_sigma, etc.) there.
    """
    def __init__(self, nInput, nOutput, ergodic=False, naive=True, invertible=False):
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

        x = torch.randn((2, 5, nInput))
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
        x = torch.randn((2, 5, nInput))
        layer.set_sigma(0.0)
        y = layer.forward(x)
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=True, invertible=True)
        x = torch.randn((4, 8, nInput))
        layer.set_sigma(0.0)
        y = layer.forward(x)
        assert y.shape == (4, 8, nOutput), "Incorrect Size"
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        print("SigmaLayer tests passed.")

class PiLayer(Layer):
    """Multiplicative layer: y_j = prod_i (1 + tanh(w_ji * x_i)).

    Forward materializes W via the inner layer, computes the outer product
    x.unsqueeze(-1) * W.unsqueeze(0) to keep per-input factors separate,
    applies tanh element-wise, then products via exp(sum(log(...))).
    No bias is applied inside the outer product.

    When ``invertible=False`` (default):
        y_j = exp(sum_i log(1 + tanh(w_ji * x_i)))

    When ``invertible=True``: outputs interleaved (y, z) pairs where
        y_j = exp(sum_i log(1 + tanh(w_ji * x_i)))
        z_j = exp(sum_i log(1 - tanh(w_ji * x_i)))
    Reverse: gamma_j = 0.5*(log y_j - log z_j) = sum_i x_i*w_ji = (x@W)_j,
    then x = gamma @ W_inv using the materialized inverse.

    All ergodic machinery lives in the inner layer; PiLayer dispatches
    the ergodic interface there via self.layers.
    """
    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True):
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.saturate   = True
        self.stable     = stable
        self.hasBias    = hasBias
        if invertible:
            self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                               naive=naive, ergodic=ergodic, stable=stable)
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

    def _effective_bias(self):
        """Bias to add to WX, or None if hasBias=False."""
        if not self.hasBias:
            return 0
        if self.layer.ergodic:
            return self.layer.bias * self.layer.biasWeight + self.layer.var * self.layer.biasNoise
        return self.layer.biasWeight

    def forward(self, x):
        if self.layer.ergodic:
            self.resample_noise()
        W    = self.layer.compute_W_current()                         # [nIn, nOut]
        x    = x.to(W.device)
        # Outer product: [..., nIn, 1] * [nIn, nOut] -> [..., nIn, nOut]
        WX   = x.unsqueeze(-1) * W                                   # broadcasts over leading dims
        if self.saturate:
            t     = torch.tanh(WX)
            one_p = 1 + t
            if self.invertible:
                one_m = 1 - t
        else:
            one_p = 1 + WX
            if self.invertible:
                one_m = 1 - WX
        if self.stable:
            one_p = one_p.clamp(min=epsilon)
            if self.invertible:
                one_m = one_m.clamp(min=epsilon)
        y = torch.sum(torch.log(one_p), dim=-2)                      # sum over nIn -> [..., nOut]
        if self.invertible:
            z = torch.sum(torch.log(one_m), dim=-2)                   # [..., nOut]
            # Interleave (y, z) along the object axis:
            # [..., S, nOut] -> [..., S, 2, nOut] -> [..., 2*S, nOut]
            result = torch.stack((y, z), dim=-2).flatten(-3, -2)
        else:
            result = torch.exp(y)
        if self.invertible:
            result = self.layer.forwardBiasInterleaved(result)
        else:
            result = self.layer.forwardBias(result)
        return result

    def reverse(self, yz):
        """Recover x from interleaved (y, z) pairs. Requires invertible=True.

        gamma_j = 0.5*(log y_j - log z_j) = sum_i x_i*w_ji = (x@W)_j.
        x = gamma @ W_inv using the materialized inverse of current W.
        """
        if self.invertible:
            yz = self.layer.reverseBiasInterleaved(yz)
        else:
            yz = self.layer.reverseBias(yz)
        # De-interleave: [..., 2*S, nOut] -> [..., S, 2, nOut] -> y, z each [..., S, nOut]
        y, z = yz.unflatten(-2, (-1, 2)).unbind(-2)
        W_inv = self.layer.compute_Winverse_current()   # [nOut, nIn]
        gamma = 0.5 * (y - z)                           # [..., nOut]
        gamma = gamma.to(W_inv.device)
        x     = gamma @ W_inv
        if self.layer.ergodic:
            self.resample_noise()
        return x

    @staticmethod
    def test():
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayer(nInput=nInput, nOutput=nOutput)
        device = next(layer.parameters()).device
        x = torch.randn((nBatch, 6, nInput), device=device)
        layer.set_sigma(0.999)
        y = layer(x)
        assert y.shape == (nBatch, 6, nOutput)
        print(f"Original input: {x}")
        print(f"After PiLayer: {y}")

        def check_roundtrip(desc, **kwargs):
            kw = dict(nInput=3, nOutput=6, invertible=True)
            kw.update(kwargs)
            layer = PiLayer(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            inputs = [('3D [B,S,nIn]', torch.randn(16, 5, nI, device=device)),
                      ('2D [B,nIn]',   torch.randn(16, nI,    device=device))]
            for tag, x in inputs:
                layer.set_sigma(0.0)
                y = layer.forward(x)
                x_recon = layer.reverse(y)
                error = torch.norm(x - x_recon) / torch.norm(x)
                assert error < 1e-4, f"{desc} {tag}: reconstruction error {error:.2e}"
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
        print("PiLayer tests passed.")

    @staticmethod
    def xorTest():
        X = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).unsqueeze(2)
        Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).unsqueeze(2)
        nInput, nHidden, nOutput = 2, 3, 1
        pi    = PiLayer(nInput, nHidden)
        sigma = SigmaLayer(nHidden, nOutput)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(chain(pi.parameters(), sigma.parameters()), lr=0.01)
        sigma.set_sigma(0.9999); pi.set_sigma(0.9999)
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = criterion(sigma(pi(X)), Y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/1000, MSE: {loss.item():.6f}')

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
        import matplotlib.pyplot as plt
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

    def forward(self, x):
        if self.symmetric:
            a2     = self.A(x)
            value  = x if self.nHidden == self.nInput else self.V(x)
            scores = torch.matmul(a2.transpose(-2, -1), a2) / (self.nInput ** 0.5)
        else:
            query  = self.Q(x)
            key    = self.K(x)
            value  = x if self.nHidden == self.nInput else self.V(x)
            scores = torch.matmul(query.transpose(-2, -1), key) / (self.nInput ** 0.5)

        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))

        if not self.reversible:
            attn = F.softmax(self.beta * scores, dim=-1)
        else:
            attn = scores

        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    @staticmethod
    def test():
        nInput = 6
        nOutput = 3
        layer = AttentionLayer(nInput=nInput, nOutput=nOutput, nHidden=7)

        x = torch.randn(4, 5, nInput)  # batch of 4
        y = layer.forward(x)

        #print("Input minus output:")
        #print(x-x_rec)
        #assert torch.norm(x-x_rec) < 1, "Norm too high"
class TransformerAttentionLayer(Layer):
    """Standard multi-head attention over the object axis.

    Unlike ``AttentionLayer`` above, this implementation attends across
    objects/tokens rather than across feature channels.  For input
    ``x`` with shape ``[batch, nObj, dim]``:

    - ``Q = W_Q x`` produces queries: "what is each object looking for?"
    - ``K = W_K x`` produces keys: "what information does each object offer?"
    - ``V = W_V x`` produces values: "what content should flow if matched?"

    Each head compares every query object to every key object, producing an
    ``[nObj, nObj]`` attention map per head. The weighted values are then
    concatenated and projected back to ``nOutput``.

    The layer also accepts 2D input ``[batch, dim]`` and treats it as a
    single-object sequence for convenience.
    """

    def __init__(self, nInput, nOutput, nHeads=1, nHidden=None):
        super(TransformerAttentionLayer, self).__init__(nInput, nOutput)
        if nHeads < 1:
            raise ValueError(f"nHeads must be >= 1, got {nHeads}")
        self.nHeads = nHeads
        self.nHidden = nOutput if nHidden is None else nHidden
        if self.nHidden % self.nHeads != 0:
            raise ValueError(
                f"nHidden ({self.nHidden}) must be divisible by nHeads ({self.nHeads})")
        self.headDim = self.nHidden // self.nHeads
        self.scale = self.headDim ** -0.5
        self.mask = None

        self.Q = LinearLayer(self.nInput, self.nHidden)
        self.K = LinearLayer(self.nInput, self.nHidden)
        self.V = LinearLayer(self.nInput, self.nHidden)
        self.Out = LinearLayer(self.nHidden, self.nOutput)
        self.reversible = False

    def set_mask(self, mask: Optional[torch.Tensor]):
        """Set an optional attention mask.

        Supported mask shapes:
        - ``[batch, nObj]`` bool: True keeps a token, False masks it out.
        - ``[batch, nObj, nObj]`` bool: explicit per-query/per-key mask.
        - ``[batch, 1, nObj, nObj]`` or ``[batch, nHeads, nObj, nObj]`` bool:
          fully expanded mask.
        """
        self.mask = mask

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

    def forward(self, x):
        assert x.ndim == 3, f"TransformerAttentionLayer expects 3D input [B, N, D], got {list(x.shape)}"

        batch, n_obj, _ = x.shape

        query = self._reshape_heads(self.Q(x))
        key = self._reshape_heads(self.K(x))
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
        layer = TransformerAttentionLayer(nInput=8, nOutput=8, nHeads=2)
        x = torch.randn(4, 5, 8)
        y = layer(x)
        assert list(y.shape) == [4, 5, 8]
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
        if lossIn is not None:
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
        self.output = torch.zeros(sz, device=_get_device())
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
        self.state = torch.zeros(sz, device=_get_device())

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


# LogicNet disabled — torchlogix einsum ops incompatible with MPS
class LogicLayer(Layer):

    def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, D)
        Y: (B, M, D)
        Returns:
            D2: (B, N, M) with squared Euclidean distances
        """
        x2 = (X * X).sum(dim=-1, keepdim=True)                 # (B, N, 1)
        y2 = (Y * Y).sum(dim=-1).unsqueeze(1)                  # (B, 1, M)
        xy = torch.bmm(X, Y.transpose(1, 2))                   # (B, N, M)
        d2 = x2 + y2 - 2.0 * xy
        return d2.clamp_min(0.0)

    def _expand_sigma(
        sigma: Optional[torch.Tensor | float],
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns sigma as shape (B, N).
        Accepts:
        - None -> ones
        - scalar
        - tensor of shape (B, N)
        - tensor of shape (N,) (broadcasted across batch)
        """
        if sigma is None:
            return torch.ones(B, N, device=device, dtype=dtype)

        if isinstance(sigma, (float, int)):
            return torch.full((B, N), float(sigma), device=device, dtype=dtype)

        if sigma.ndim == 1:
            if sigma.shape[0] != N:
                raise ValueError(f"1D sigma must have shape ({N},), got {tuple(sigma.shape)}")
            return sigma.to(device=device, dtype=dtype).unsqueeze(0).expand(B, N)

        if sigma.ndim == 2:
            if sigma.shape != (B, N):
                raise ValueError(f"2D sigma must have shape ({B}, {N}), got {tuple(sigma.shape)}")
            return sigma.to(device=device, dtype=dtype)

        raise ValueError("sigma must be None, scalar, shape (N,), or shape (B,N)")

    def kernel_overlap(
        X: torch.Tensor,
        Y: torch.Tensor,
        sigma_x: Optional[torch.Tensor | float] = None,
        sigma_y: Optional[torch.Tensor | float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Pairwise fuzzy overlap kernel between two vector sets.

        X: (B, N, D)
        Y: (B, M, D)

        Returns:
            K: (B, N, M)

        Kernel:
            K_ij = exp( -||x_i - y_j||^2 / (2 * (sigma_x_i^2 + sigma_y_j^2)) )
        """
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")

        B, N, D = X.shape
        M = Y.shape[1]

        sx = _expand_sigma(sigma_x, B, N, X.device, X.dtype)   # (B, N)
        sy = _expand_sigma(sigma_y, B, M, Y.device, Y.dtype)   # (B, M)

        d2 = _pairwise_sq_dists(X, Y)                          # (B, N, M)
        denom = 2.0 * (sx.unsqueeze(2).square() + sy.unsqueeze(1).square()) + eps
        return torch.exp(-d2 / denom)

    def union(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Mereological union as set concatenation.

        X: (B, N, D)
        Y: (B, M, D)

        Returns:
            U: (B, N+M, D)
        """
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")
        return torch.cat([X, Y], dim=1)

    def intersection(
        X: torch.Tensor,
        Y: torch.Tensor,
        sigma_x: Optional[torch.Tensor | float] = None,
        sigma_y: Optional[torch.Tensor | float] = None,
        topk: Optional[int] = None,
        weight_threshold: Optional[float] = None,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuzzy intersection of two RBF-like vector sets.

        Returns a new vector set whose elements are precision-weighted pairwise
        merges of X and Y, with weights given by Gaussian overlap.

        X: (B, N, D)
        Y: (B, M, D)

        Returns:
            Z: (B, K, D)   merged intersection vectors
            W: (B, K)      corresponding intersection weights in [0,1]

        Notes:
        - Exact pairwise intersection would produce N*M vectors.
        - topk keeps only the strongest K pairwise intersections per batch.
        - weight_threshold filters weak overlaps.
        """
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")

        B, N, D = X.shape
        M = Y.shape[1]

        sx = _expand_sigma(sigma_x, B, N, X.device, X.dtype)   # (B, N)
        sy = _expand_sigma(sigma_y, B, M, Y.device, Y.dtype)   # (B, M)

        # Pairwise overlap weights
        Kxy = kernel_overlap(X, Y, sx, sy, eps=eps)            # (B, N, M)

        # Precision-weighted midpoint
        px = 1.0 / (sx.square().unsqueeze(2) + eps)            # (B, N, 1)
        py = 1.0 / (sy.square().unsqueeze(1) + eps)            # (B, 1, M)
        denom = px + py                                        # (B, N, M)

        Xp = X.unsqueeze(2)                                    # (B, N, 1, D)
        Yp = Y.unsqueeze(1)                                    # (B, 1, M, D)

        Z = (px.unsqueeze(-1) * Xp + py.unsqueeze(-1) * Yp) / denom.unsqueeze(-1)  # (B,N,M,D)
        W = Kxy                                                # (B,N,M)

        # Optional threshold
        if weight_threshold is not None:
            mask = W >= weight_threshold
            W = W * mask

        # Flatten pairwise results
        Z = Z.reshape(B, N * M, D)
        W = W.reshape(B, N * M)

        # Optional top-k pruning
        if topk is not None:
            k = min(topk, N * M)
            vals, idx = torch.topk(W, k=k, dim=1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, D)
            Z = torch.gather(Z, dim=1, index=gather_idx)
            W = vals

        return Z, W

    def part(
        X: torch.Tensor,
        Y: torch.Tensor,
        sigma_x: Optional[torch.Tensor | float] = None,
        sigma_y: Optional[torch.Tensor | float] = None,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        signed: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Fuzzy parthood scores:
            p(X -> Y), p(Y -> X)

        Uses max-coverage:
            p(X -> Y) = average_i max_j K(x_i, y_j)

        Inputs:
            X: (B, N, D)
            Y: (B, M, D)
            weights_x: optional (B, N)
            weights_y: optional (B, M)  # only used for symmetry / validation;
                                        # max-coverage itself weights the outer set

        Returns:
            P: (B, 2)
            [:,0] = parthood(X -> Y)
            [:,1] = parthood(Y -> X)

        If signed=True, maps [0,1] -> [-1,1] by 2p - 1.
        """
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must both have shape (B, N, D) and (B, M, D)")
        if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
            raise ValueError("Batch size and feature dimension must match")

        B, N, D = X.shape
        M = Y.shape[1]

        Kxy = kernel_overlap(X, Y, sigma_x=sigma_x, sigma_y=sigma_y, eps=eps)  # (B,N,M)
        Kyx = Kxy.transpose(1, 2)                                               # (B,M,N)

        cover_xy = Kxy.max(dim=2).values                                        # (B,N)
        cover_yx = Kyx.max(dim=2).values                                        # (B,M)

        if weights_x is None:
            p_xy = cover_xy.mean(dim=1)
        else:
            if weights_x.shape != (B, N):
                raise ValueError(f"weights_x must have shape ({B}, {N})")
            wx = weights_x.to(device=X.device, dtype=X.dtype)
            p_xy = (wx * cover_xy).sum(dim=1) / (wx.sum(dim=1) + eps)

        if weights_y is None:
            p_yx = cover_yx.mean(dim=1)
        else:
            if weights_y.shape != (B, M):
                raise ValueError(f"weights_y must have shape ({B}, {M})")
            wy = weights_y.to(device=Y.device, dtype=Y.dtype)
            p_yx = (wy * cover_yx).sum(dim=1) / (wy.sum(dim=1) + eps)

        out = torch.stack([p_xy, p_yx], dim=1)                                  # (B,2)

        if signed:
            out = 2.0 * out - 1.0

        return out

    def neg(X: torch.Tensor) -> torch.Tensor:
        """
        Affirming negation on the hypersphere:
            neg(x) = -x

        X: (B, N, D)
        Returns:
            (B, N, D)
        """
        return -X

    def non(X: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        """
        Non-affirming negation:
            non(x) = alpha * x, with alpha in [0, 1]

        alpha = 0.0 gives full withdrawal toward zero.
        alpha in (0,1) gives partial weakening.

        X: (B, N, D)
        Returns:
            (B, N, D)
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        return alpha * X

    def symbolize(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Symbolize a set of vectors by mean norm.

        X: (B, N, D)
        Returns:
            s: (B,) in [-1, 1], assuming ||x_i|| in [0, 1]
        """
        norms = torch.linalg.norm(X, dim=-1)          # (B, N)
        s = 2.0 * norms.mean(dim=1) - 1.0
        return s.clamp(-1.0, 1.0)


    def scalar_neg(a: torch.Tensor) -> torch.Tensor:
        """
        Affirming negation on [-1,1].
        """
        return -a


    def scalar_non(a: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        """
        Non-affirming negation: contraction toward zero.
        alpha in [0,1]. alpha=0 gives full neutralization.
        """
        return (alpha * a).clamp(-1.0, 1.0)


    def scalar_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Symbolic union on [-1,1].
        """
        return torch.maximum(a, b)


    def scalar_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Symbolic intersection on [-1,1].
        """
        return torch.minimum(a, b)


    def scalar_part(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Signed symbolic parthood/order score on [-1,1].

        Positive means a is contained by / does not exceed b.
        Zero means equal.
        Negative means a exceeds b.
        """
        return (b - a).clamp(-1.0, 1.0)
    
    @staticmethod
    def test():
        B, N, M, D = 4, 16, 20, 32
        X = torch.randn(B, N, D, device="gpu")
        Y = torch.randn(B, M, D, device="gpu")

        LL = LogicLayer()
        U = LL.union(X, Y)                    # (B, N+M, D)
        Z, W = LL.intersection(X, Y, topk=32) # Z: (B, 32, D), W: (B, 32)
        P = LL.part(X, Y)                     # (B, 2)
        X_neg = LL.neg(X)                     # (B, N, D)
        X_non = LL.non(X, alpha=0.2)          # (B, N, D)

#endregion


def test():
    torch.autograd.set_detect_anomaly(True)

    LogicLayer.test()
    LinearLayer.test()


    InvertibleLinearLayer.test()

    SigmaLayer.test()
    PiLayer.test()
    PiLayer.xorTest()

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
