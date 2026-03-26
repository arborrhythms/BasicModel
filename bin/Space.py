"""Top-level model assembly, data loading, and experiment reporting.

``BasicModel`` composes the custom layers from ``Model.py`` into a set of
spaces that move between raw inputs, percepts, concepts, symbols, syntax,
and outputs.  The same module also carries the project utilities used to
load datasets, resolve config paths, plot results, and save reports.
"""

import math, os, warnings
from collections import namedtuple
from contextlib import contextmanager, nullcontext
import numpy as np
warnings.filterwarnings(
    "ignore",
    message="Initializing zero-element tensors is a no-op",
    category=UserWarning,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
try:
    from torchviz import make_dot
except ImportError:
    make_dot = None
from sklearn.decomposition import PCA
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
import torch.optim as optim
from torch.profiler import profile as torch_profile, ProfilerActivity, schedule as profiler_schedule
from functools import partial
from datetime import datetime

import util
from util import TheDevice, TheMessage
util.init_runtime_env()
from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend
from embed import WordVectors, PretrainModel
from data import Data, TheData
from Model import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Model import VQLayer, NormLayer, LinearLayer, InvertibleLinearLayer, AttentionLayer
from Model import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon

class Encoding(nn.Module):
    """Abstract base class for per-slot encodings in the embedding vector.

    Each encoding occupies ``nDim`` contiguous slots at positions given by
    ``self.index`` (negative offsets from the end of the embedding).

    Subclasses implement:
        ``encode(value)``  → tensor [..., nDim]
        ``decode(encoded)`` → decoded value (tensor or scalar)
        ``forward(x)``     → stamped tensor (how values are assigned per-batch)
    """
    nDim = 0  # subclasses must set

    def __init__(self, index, maxVal):
        super().__init__()
        self.index = index
        self.maxVal = maxVal

    def encode(self, value):
        """Encode a value into nDim-wide representation. Subclass must override."""
        raise NotImplementedError

    def decode(self, encoded):
        """Decode nDim-wide representation back to a value. Subclass must override."""
        raise NotImplementedError

    def resolve(self, embSize):
        """Resolve negative index offsets to absolute indices for a given embedding size."""
        return np.add([embSize] * len(self.index), self.index)

    def reverse(self, y):
        """Extract encoding, decode to values, zero the slots.

        Returns (cleaned_tensor, decoded_values) where decoded_values is
        a float tensor of shape [batch, nVec].
        """
        if self.nDim == 0:
            return y, torch.zeros(y.shape[0], y.shape[1], device=TheDevice)
        embeddingSize = y.shape[-1]
        index = self.resolve(embeddingSize)
        if index[0] < 0 or index[0] >= embeddingSize:
            return y, torch.zeros(y.shape[0], y.shape[1], device=TheDevice)
        encoded = y[:, :, index].clone()  # [batch, nVec, nDim]
        y[:, :, index] = 0
        return y, self.decode(encoded)
class QuadratureEncoding(Encoding):
    """Sin/cos quadrature encoding (nDim=2).

    Encodes scalar values as (sin, cos) pairs at frequency ``div_term = 2π/maxVal``.
    Exactly invertible via ``atan2(sin, cos) / div_term``.

    Subclasses implement ``forward()`` to define how values are assigned
    (per-object for spatial, per-batch for temporal).
    """
    nDim = 2

    def __init__(self, index, maxVal):
        super().__init__(index, maxVal)
        self.div_term = 2 * math.pi / maxVal

    def encode(self, offsets):
        """Encode values to sin/cos pairs.

        Args:
            offsets: tensor [...] or scalar (int/float).

        Returns:
            Tensor [..., 2] with (sin, cos).
        """
        if not isinstance(offsets, torch.Tensor):
            offsets = torch.tensor(float(offsets), device=TheDevice)
        angle = offsets * self.div_term
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def decode(self, encoded):
        """Decode sin/cos encoded tensor back to offsets.

        Args:
            encoded: tensor [..., 2] with (sin, cos) or scalar pair.

        Returns:
            Decoded offsets (tensor or float).
        """
        if isinstance(encoded, torch.Tensor):
            sin_val = encoded[..., 0]
            cos_val = encoded[..., 1]
            angle = torch.atan2(sin_val, cos_val) % (2 * math.pi)
            return angle / self.div_term
        # scalar fallback
        sin_val, cos_val = encoded
        angle = math.atan2(float(sin_val), float(cos_val)) % (2 * math.pi)
        return angle / self.div_term

    def stamp(self, buf, batch_idx, pos_idx, offset):
        """Write encoded (sin, cos) pair at a specific buffer position.

        Uses scalar math for device-agnostic element-wise assignment.
        """
        idx = self.resolve(buf.shape[-1])
        if idx[0] >= 0 and idx[0] < buf.shape[-1]:
            angle = float(offset) * self.div_term
            buf[batch_idx, pos_idx, idx[0]] = math.sin(angle)
            buf[batch_idx, pos_idx, idx[1]] = math.cos(angle)
class ActiveEncoding(Encoding):
    """Per-slot scalar activation encoding (fuzzy sparsity).

    Occupies 1 dimension in the embedding vector (nDim=1).
    encode/decode are identity — the producing Space decides how
    to compute activation values (norm-based, sum-based, etc.).
    ActiveEncoding is the carrier, not the formula.
    """
    nDim = 1

    def __init__(self, maxVal=1.0):
        super().__init__([-5], maxVal)

    def encode(self, activation):
        """Identity encode: activation values pass through."""
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(float(activation))
        return activation.unsqueeze(-1) if activation.dim() == 0 else activation

    def decode(self, encoded):
        """Identity decode: return scalar activation."""
        if isinstance(encoded, torch.Tensor):
            return encoded.squeeze(-1)
        return encoded
class WhereEncoding(QuadratureEncoding):
    """Encode spatial position (nWhere) as sin/cos values in reserved embedding slots.

    Writes a (sin, cos) pair into the last few dimensions of each object vector,
    indexed by ``self.index`` (negative offsets from the end).  A monotonic
    counter ``self.p`` assigns each object a unique position within a dataset
    pass; it must be reset between epochs to avoid overflow.

    Used by SubSpace to stamp each object with a "where" tag.
    """
    index = [-4, -3]
    p = 0

    def __init__(self, maxP=0, nWhere=2):
        if nWhere > 0:
            print("Creating positional encoding ...")
            super().__init__([-4, -3], maxP)
        else:
            Encoding.__init__(self, [], 1)  # skip QuadratureEncoding div_term
            self.nDim = 0
        self.p = 0

    def forward(self, x):
        """Stamp sin/cos positional values into reserved embedding slots."""
        if self.nDim == 0:
            return x
        batch = x.shape[0]
        n     = x.shape[1] if len(x.shape) > 1 else 1
        embeddingSize = x.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        position = torch.arange(self.p, self.p+batch*n, dtype=torch.float32, device=TheDevice)
        pos = self.encode(position)  # [batch*n, 2]
        y = x.clone()
        y[:, :, index] = pos.reshape(batch, n, self.nDim)
        self.p += batch
        assert self.p < self.maxVal, "Overflow in object embedding"
        return y

    @staticmethod
    def test():
        pe = WhereEncoding(100)
        pe.p = 0
        x = torch.zeros([2, 4, 100], device=TheDevice)
        y = pe.forward(x)
        cleaned, offsets = pe.reverse(y)
        print(f"Positions decoded: {offsets}")
class WhenEncoding(QuadratureEncoding):
    """Encode temporal order (nWhen) as sin/cos values in reserved embedding slots.

    Uses the same quadrature encoding as PositionalEncoding: a (sin, cos) pair
    at a single frequency ``div_term = 2π/maxVal``.  This is exactly invertible
    via ``atan2(sin, cos) / div_term``.

    A global time counter ``self.t`` is incremented explicitly via
    ``increment(batch)`` at the end of each forward pass through the full model.

    Used by SubSpace to stamp each object with a "when" tag.
    """
    index = [-2, -1]
    t = 0

    def __init__(self, maxT=10000, nWhen=2):
        if nWhen > 0:
            super().__init__([-2, -1], maxT)
        else:
            Encoding.__init__(self, [], 1)  # skip QuadratureEncoding div_term
            self.nDim = 0
        self.t = 0

    def forward(self, x):
        """Stamp sin/cos temporal values into reserved embedding slots."""
        if self.nDim == 0:
            return x
        batch = x.shape[0]
        n = x.shape[1] if len(x.shape) > 1 else 1
        embeddingSize = x.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        time_vals = torch.arange(self.t, self.t + batch, dtype=torch.float32, device=TheDevice)
        time = self.encode(time_vals)  # [batch, 2]
        y = x.clone()
        y[:, :, index] = time.unsqueeze(1).expand(-1, n, -1)
        return y

    def increment(self, batch):
        """Advance the global time counter by `batch` steps (called per forward pass)."""
        self.t += batch

    @staticmethod
    def test():
        te = WhenEncoding(10000)
        te.t = 0
        x = torch.zeros([2, 4, 10], device=TheDevice)
        y = te.forward(x)
        cleaned, times = te.reverse(y)
        print(f"Times decoded: {times}")
class WhatEncoding(Encoding):
    """Handle the content-layout transform for a space's What factor.

    Unlike WhereEncoding and WhenEncoding, this encoding is not a fixed
    quadrature code.  It owns only the layout transform used by reshaped
    spaces: validating [batch, nObj, emb] tensors, flattening them to
    [batch, nObj*emb] before a layer pass, and restoring them afterwards.

    The encoding is parameter-free and identity by default.
    """

    nDim = 0  # WhatEncoding does not occupy fixed index slots

    def __init__(self, inputShape=None, outputShape=None):
        super().__init__([], 0)
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, objects, **kwargs):
        """Identity content pass-through."""
        return objects

    def forwardBegin(self, x, batch):
        """Validate or flatten input at the start of a forward pass."""
        input_size = self.inputShape[1]
        assert list(x.shape) == [batch, self.inputShape[0], input_size]
        return x

    def forwardEnd(self, x, batch):
        """Validate or unflatten output at the end of a forward pass."""
        output_size = self.outputShape[1]
        assert list(x.shape) == [batch, self.outputShape[0], output_size], \
            f"forwardEnd: got {list(x.shape)}, expected {[batch, self.outputShape[0], output_size]}"
        return x

    def reverseBegin(self, y, batch):
        """Validate or flatten output-side state at the start of reverse()."""
        output_size = self.outputShape[1]
        assert list(y.shape) == [batch, self.outputShape[0], output_size]
        return y

    def reverseEnd(self, y, batch):
        """Validate or unflatten input-side state at the end of reverse()."""
        input_size = self.inputShape[1]
        assert list(y.shape) == [batch, self.inputShape[0], input_size]
        return y

    def _flatten(self, x, batch, forward=True):
        """Collapse [batch, nObj, dim] -> [batch, nObj*dim] for a reshaped space."""
        if forward:
            size = self.inputShape[1] * self.inputShape[0]
        else:
            size = (self.outputShape[1]) * self.outputShape[0]
        return x.reshape(batch, size)

    def _unflatten(self, y, batch, forward=True):
        """Restore [batch, nObj*dim] -> [batch, nObj, dim] for a reshaped space."""
        if forward:
            per_obj = self.outputShape[1]
            return y.reshape(batch, self.outputShape[0], per_obj)
        else:
            per_obj = self.inputShape[1]
            return y.reshape(batch, self.inputShape[0], per_obj)
class ObjectEncoding(Encoding):
    """Handle the content-layout transform for a space's What factor.

    Unlike WhereEncoding and WhenEncoding, this encoding is not a fixed
    quadrature code.  It owns only the layout transform used by reshaped
    spaces: validating [batch, nObj, emb] tensors, flattening them to
    [batch, nObj*emb] before a layer pass, and restoring them afterwards.

    The encoding is parameter-free and identity by default.
    """

    nDim = 0  # WhatEncoding does not occupy fixed index slots

    def __init__(self, inputShape=None, outputShape=None, flatten=False):
        super().__init__([], 0)
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.flatten = flatten

    def forward(self, objects, **kwargs):
        """Identity content pass-through."""
        return objects

    # forwardBegin/End, reverseBegin/End removed — inlined into Space.forwardBegin/End

    def _unflatten(self, y, batch, forward=True):
        """Restore [batch, nObj*dim] -> [batch, nObj, dim] for a reshaped space."""
        if forward:
            return y.reshape(batch, self.outputShape[0], self.outputShape[1])
        else:
            return y.reshape(batch, self.inputShape[0], self.inputShape[1])

    def split_aux(self, y, nWhat):
        """Split a full object vector into content and aux fields.

        Reverse callers use this before snapping the content dimensions to a
        codebook row. The returned ``aux`` tail is treated as opaque here:
        it may contain where/when slots or any other non-what payload owned by
        the enclosing ``SubSpace``.
        """
        # If there is no aux tail, keep reverse callers on the same code path
        # by returning a cloned content tensor and ``None``.
        if y.shape[-1] <= nWhat:
            return y.clone(), None
        # ``nWhat`` is always the leading content block; everything after it is
        # preserved verbatim so the caller can restore it after content-only
        # reverse logic finishes.
        return y[:, :, :nWhat].clone(), y[:, :, nWhat:].clone()

    def restore_aux(self, content, aux=None):
        """Reattach the aux tail saved by ``split_aux()``.

        This is the last step in the caller-managed reverse flow:
        1. save aux with ``split_aux()``
        2. snap or decode content only
        3. restore the untouched aux tail here
        """
        if aux is None:
            return content
        return torch.cat([content, aux], dim=-1)

class Basis(nn.Module):
    """Shared runtime contract for SubSpace payloads."""

    def __init__(self):
        super().__init__()
        # W is NOT stored here — subclasses own the storage.
        # Tensor/Codebook use register_buffer; Embedding uses wv._vectors.
        self.activation = None
        self.activeSigma = None
        self.nInput = 0
        self.nVectors = 0
        self.nDim = 0
        self.objectSize = 0
        self.embeddingSize = 0
        self.passThrough = False
        self.signed = False
        self.ergodic = False
        self.sigma_kappa = 0.01

    def getW(self):
        """Return the current weight tensor. Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement getW()")

    def setW(self, value):
        """Set the weight tensor. Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement setW()")

    def create(self, nInput, nVectors, nDim, customVQ=True, signed=False,
               passThrough=False, objectSize=0):
        self.nInput = nInput
        self.nVectors = nVectors
        self.nDim = nDim or 0
        self.signed = signed
        self.passThrough = passThrough
        self.objectSize = objectSize
        self.embeddingSize = self.nDim + self.objectSize
        return self

    @property
    def size(self):
        w = self.getW()
        return 0 if w is None else w.shape[0]

    @property
    def width(self):
        w = self.getW()
        return 0 if w is None else w.shape[-1]

    @property
    def content_dim(self):
        return self.nDim

    @property
    def activeIndices(self):
        if self.activation is None:
            return None
        return self.activation != 0

    def clearActivation(self):
        self.activation = None
        self.activeSigma = None

    def forward(self, x):
        self.setW(x)
        return x

    def reverse(self, y, **kwargs):
        self.setW(y)
        return y

    def reverse_raw(self, y):
        return y

    def set_sigma(self, sigma):
        if sigma == 0:
            self.sigma_kappa = 1e6
        else:
            self.sigma_kappa = 0.01 / sigma

    def quantize(self, x):
        raise RuntimeError(f"{self.__class__.__name__} does not support quantize()")

    def _coerce_rows(self, value):
        if isinstance(value, (list, tuple)):
            value = torch.stack(list(value), dim=0)
        return value

    def replace(self, new_W):
        self.setW(self._coerce_rows(new_W))
        return self.getW()

    def insert(self, new_W):
        new_W = self._coerce_rows(new_W)
        w = self.getW()
        self.setW(new_W if w is None else torch.cat([w, new_W], dim=0))
        return self.getW()

    def remove(self, indices):
        w = self.getW()
        if w is None:
            return None
        mask = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
        mask[indices] = False
        self.setW(w[mask])
        return self.getW()

    def parameters_for_optimizer(self):
        w = self.getW()
        return [w] if isinstance(w, nn.Parameter) else []

    def _prototype_weight(self, weight=None, context="prototype lookup"):
        weight = self.getW() if weight is None else weight
        if weight is None or weight.ndim != 2:
            shape = None if weight is None else list(weight.shape)
            raise RuntimeError(
                f"{self.__class__.__name__}.{context} requires a 2-D prototype matrix, "
                f"but got shape {shape}.")
        return weight

    def _ensure_embedding_width(self, x):
        if x.shape[-1] == self.nDim and self.objectSize > 0:
            padding = torch.zeros(*x.shape[:-1], self.objectSize,
                                  device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=-1)
        return x

    def _snap_content(self, content, weight=None, nWhat=None):
        weight = self._prototype_weight(weight, context="reverse")
        nWhat = self.nDim if nWhat is None else nWhat
        snapped = content.clone()
        flat = snapped[:, :, :nWhat].reshape(-1, nWhat)
        nonzero = flat.abs().sum(dim=1) >= 1e-8
        if torch.any(nonzero):
            sims = F.cosine_similarity(
                flat[nonzero].unsqueeze(1),
                weight[:, :nWhat].unsqueeze(0),
                dim=2,
            )
            idx = sims.argmax(dim=1)
            flat[nonzero] = weight[idx, :nWhat]
            snapped[:, :, :nWhat] = flat.reshape(snapped.shape[0], snapped.shape[1], nWhat)
        return snapped

    def norm(self, x):
        return torch.norm(x, dim=-1)

    def normalize(self, x=None):
        target = self.getW() if x is None else x
        if target is None:
            raise RuntimeError(f"{self.__class__.__name__}.normalize() has no tensor to normalize.")
        if self.signed:
            normalized = F.normalize(target, p=2, dim=-1)
        else:
            normalized = torch.clamp(target, 0, 1)
            normalized = F.normalize(normalized, p=2, dim=-1)
        if x is None:
            self.setW(normalized)
            return self.getW()
        return normalized

    def negate(self, x):
        return 1 - x

    def distance(self, x, y):
        return (x.T @ y) / max(self.size, 1)

    def codebookDistance(self, x):
        weight = self._prototype_weight(context="codebookDistance")
        vec = weight[:, :self.nDim].to(TheDevice)
        return x @ vec.T / max(self.nDim, 1)

    def unsignedAngle(self, x, y, dim=-1):
        cos_sim = F.cosine_similarity(x, y, dim=dim)
        return 0.5 * (1 - cos_sim)

    def equal(self, x, y):
        return 1.0 - self.unsignedAngle(x, y)

    def part(self, x, y):
        return 1.0 - self.unsignedAngle(x, y)

    def whole(self, x, y):
        return 1.0 - self.unsignedAngle(y, x)

    def boundary(self, x, y):
        return torch.abs(self.part(x, y) - self.whole(x, y))

    def overlap(self, x, y):
        return torch.min(self.part(x, y), self.whole(x, y))

    def union(self, x, y):
        return torch.max(x, y)

    def intersection(self, x, y):
        return torch.min(x, y)

    def active_dense(self):
        w = self.getW()
        if self.activation is None or w is None:
            return None
        if self.activation.ndim == 1:
            return self.activation.unsqueeze(-1) * w
        return self.activation.unsqueeze(-1) * w.unsqueeze(0)

    @staticmethod
    def _pairwise_sq_dists(X, Y):
        x2 = (X * X).sum(dim=-1, keepdim=True)
        y2 = (Y * Y).sum(dim=-1).unsqueeze(1)
        xy = torch.bmm(X, Y.transpose(1, 2))
        return (x2 + y2 - 2.0 * xy).clamp_min(0.0)

    @staticmethod
    def _expand_sigma(sigma, B, N, device, dtype):
        if sigma is None:
            return torch.ones(B, N, device=device, dtype=dtype)
        if isinstance(sigma, (float, int)):
            return torch.full((B, N), float(sigma), device=device, dtype=dtype)
        if sigma.ndim == 1:
            return sigma.to(device=device, dtype=dtype).unsqueeze(0).expand(B, N)
        return sigma.to(device=device, dtype=dtype)

    @classmethod
    def kernel_overlap(cls, X, Y, sigma_x=None, sigma_y=None, eps=1e-8):
        B, N, _ = X.shape
        M = Y.shape[1]
        sx = cls._expand_sigma(sigma_x, B, N, X.device, X.dtype)
        sy = cls._expand_sigma(sigma_y, B, M, Y.device, Y.dtype)
        d2 = cls._pairwise_sq_dists(X, Y)
        denom = 2.0 * (sx.unsqueeze(2).square() + sy.unsqueeze(1).square()) + eps
        return torch.exp(-d2 / denom)

    @staticmethod
    def neg(X):
        return -X

    @staticmethod
    def non(X, alpha=0.0):
        return alpha * X

    @staticmethod
    def symbolize(X, eps=1e-8):
        norms = torch.linalg.norm(X, dim=-1)
        return (2.0 * norms.mean(dim=1) - 1.0).clamp(-1.0, 1.0)
class Tensor(Basis):
    """Dense tensor payload implementation used for ordinary SubSpace slots."""

    def __init__(self, nVectors=0, nDim=0, W=None):
        super().__init__()
        self.register_buffer('W', None)
        self.nVectors = nVectors
        self.nDim = nDim
        if W is not None:
            self.W = W

    def getW(self):
        return self.W

    def setW(self, value):
        self.W = value

    def forward(self, x):
        self.setW(x)
        return x

    def reverse(self, y, **kwargs):
        self.setW(y)
        return y
class Codebook(Basis):
    """Prototype basis with vector quantization and reverse snapping support."""

    def __init__(self):
        super().__init__()
        self.register_buffer('W', None)
        self.customVQ = True
        self.snapDistance = 0.1
        self.eta = 0.9
        self.alpha = 0.0
        self.codebookSize = 0
        self.vq = None

    def getW(self):
        return self.W

    def setW(self, value):
        self.W = value

    def getSize(self):
        return self.nVectors

    def create(self, nInput, nVectors, nDim, customVQ=True, signed=False,
               passThrough=False, objectSize=0):
        super().create(
            nInput,
            nVectors,
            nDim,
            customVQ=customVQ,
            signed=signed,
            passThrough=passThrough,
            objectSize=objectSize,
        )
        self.customVQ = customVQ
        self.alpha = 0.0
        if (not self.passThrough) and self.nVectors > 0 and self.getW() is None:
            self.addVectors(self.nVectors)
        return self

    def updateWeights(self, embed_sum, cluster_size):
        return torch.ones(self.vq.codebook_size, device=TheDevice)

    def addVectors(self, nVec=1, decay=0.9):
        """Allocate ``nVec`` prototype entries using the configured backend."""
        self.codebookSize = nVec
        if self.customVQ:
            self.vq = VectorQuantize(
                dim=self.embeddingSize,
                codebook_size=nVec,
                threshold_ema_dead_code=1,
                decay=decay,
                commitment_weight=1.0,
                rotation_trick=True,
            )
            self.setW(self.vq.codebook)
        else:
            W = torch.randn([nVec, self.embeddingSize], device=TheDevice)
            for i in range(nVec):
                W[i, :] = self.normalize(W[i, :]).squeeze(0)
            self.setW(W)
        return self.getW()

    def quantize(self, x):
        if self.passThrough:
            return x, None, torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x = self._ensure_embedding_width(x)
        if self.customVQ:
            quantized, indices, commit_loss = self.vq(
                x,
                ema_update_weight=self.updateWeights,
            )
            self.setW(self.vq.codebook)
            return quantized, indices, commit_loss
        weight = self._prototype_weight(context="quantize")
        flat = x.reshape(-1, x.shape[-1])
        dists = flat[:, :self.nDim] @ weight[:, :self.nDim].T / max(self.nDim, 1)
        indices = dists.argmax(dim=-1)
        quantized = weight[indices]
        quantized = quantized.reshape(*x.shape[:-1], weight.shape[-1])
        indices = indices.reshape(x.shape[:-1])
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return quantized, indices, loss

    def forward(self, input):
        if self.passThrough:
            self.setW(input)
            return input

        w = self.getW()
        if w is None or self.codebookSize == 0:
            self.addVectors(max(self.nVectors, input.shape[1]))
        x = self._ensure_embedding_width(input)
        batch = x.shape[0]
        act = torch.zeros([batch, self.codebookSize], device=TheDevice)
        if self.customVQ:
            flat = torch.reshape(x, [-1, self.embeddingSize])
            quantized, indices, _ = self.quantize(flat)
            batch = x.shape[0]
            n_tokens = flat.shape[0] // batch
            err = torch.norm(flat - quantized, dim=-1).reshape(batch, n_tokens)
            indices = indices.reshape(batch, n_tokens)
            # Use 3D views for token-level indexing (x may be 2D when reshape=True)
            x3d = x.reshape(batch, n_tokens, self.embeddingSize)
            input3d = input.reshape(batch, n_tokens, self.embeddingSize)
            quantized3d = quantized.reshape(batch, n_tokens, self.embeddingSize)
            k = min(self.nVectors, err.shape[1])
            _, indices_smallest = torch.topk(err, k=k, dim=1, largest=False)
            for i in range(indices_smallest.shape[0]):
                claimed = set()
                for j in range(indices_smallest.shape[1]):
                    row_idx = indices_smallest[i, j].item()
                    cb_idx = indices[i, row_idx].item()
                    if cb_idx in claimed:
                        continue
                    claimed.add(cb_idx)
                    x3d[i, row_idx, :] = quantized3d[i, row_idx, :]
                    if err[i, row_idx] <= self.snapDistance:
                        cosSim = self.unsignedAngle(
                            input3d[i, row_idx, :].clone(),
                            quantized3d[i, row_idx, :self.nDim].clone(),
                        )
                        act[i, cb_idx] = cosSim + self.alpha * random.random()
            x = x3d.reshape(x.shape)
        else:
            w = self.getW()
            dists = self.codebookDistance(input)
            for b in range(x.shape[0]):
                for v in range(x.shape[1]):
                    nearestDist, nearestIdx = torch.topk(dists[b, v, :], 1, dim=-1, largest=True)
                    err = nearestDist[0].item()
                    idx = nearestIdx[0].item()
                    if err <= self.snapDistance:
                        x[b, v, :] = w[idx, :]
                        act[b, idx] = nearestDist[0]
                    if self.training:
                        w[idx, :] = self.eta * w[idx, :] + (1 - self.eta) * x[b, v, :]
        self.activation = act
        self.activeSigma = None
        self.setW(x)
        return x

    def reverse(self, y, **kwargs):
        if self.passThrough:
            self.setW(y)
            return y
        if y.shape[-1] < self.nDim:
            raise RuntimeError(
                f"Codebook.reverse() expected at least {self.nDim} content dims, "
                f"got shape {list(y.shape)}.")
        content = y.clone() if y.shape[-1] == self.nDim else y[:, :, :self.nDim].clone()
        content = self._snap_content(content, weight=self.getW(), nWhat=self.nDim)
        self.setW(content)
        return content

    def replace(self, new_vectors):
        new_vectors = self._coerce_rows(new_vectors)
        if self.customVQ and self.vq is not None:
            self.vq.codebook = new_vectors
        self.setW(new_vectors)
        w = self.getW()
        self.codebookSize = 0 if w is None else w.shape[0]
        return w

    def insert(self, new_vectors):
        new_vectors = self._coerce_rows(new_vectors)
        new_vectors = self.normalize(new_vectors)
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.unsqueeze(0)
        current = self.getW()
        if current is None:
            self.replace(new_vectors)
        else:
            self.replace(torch.cat([current, new_vectors], dim=0))
        return self.getW()

    def remove(self, indices):
        w = self.getW()
        if w is None:
            return None
        mask = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
        mask[indices] = False
        self.replace(w[mask])
        return self.getW()

    def learn(self, x, target_idx, lr=0.01):
        x = F.normalize(x, p=2, dim=-1)
        w = self.getW()
        selected_vectors = w[target_idx]
        delta = lr * (x - selected_vectors)
        w[target_idx] += delta
        self.normalize()

    @staticmethod
    def conceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        A_norm = A / A.norm()
        B_norm = B / B.norm()
        cross_prod = torch.linalg.cross(A_norm, B_norm)
        orthogonal_vector = cross_prod / cross_prod.norm()
        distance = orthogonal_vector.norm()
        return torch.clamp(distance, 0, 1)

    @staticmethod
    def perceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        A, B = A.clamp(0, 1), B.clamp(0, 1)
        ratio = torch.minimum(A / (B + epsilon), torch.ones_like(A))
        return torch.prod(ratio).item()
class Embedding(Basis):
    """Text-backed Basis using a differentiable nn.Embedding with online CBOW/SBOW training.

    Loads pretrained weights from a WordVectors artifact (e.g. sentence.pt)
    and wraps them in an nn.Embedding so gradients can flow through the
    forward pass.  ``train_step(words, method)`` runs one sentence-level
    embedding update: CBOW (padded context) or SBOW (leave-one-out centroid).

    This is a direct ``Basis`` implementation: ``forward()`` performs the text
    encoding step, and ``reverse()`` performs the decode step.

    Tokens are handled as ``(text, start)`` pairs from the lexer.

    Public API is intentionally small:
    ``create()``, ``forward()``, ``reverse()``, ``train_step()``,
    ``set_sigma()``, ``predict()``,
    ``embed_token()``, reconstruction helpers, and read-only properties.
    """

    def __init__(self):
        super().__init__()
        self.pretrain = None       # PretrainModel, created in create()
        self.wv = None             # WordVectors (nn.Module with nn.Parameter)
        self.doc_spans = []
        self.optimize_embedding = False  # set by ModelFactory from <trainEmbedding>
        object.__setattr__(self, '_model', None)  # back-ref to BasicModel, avoids nn.Module submodule registration
        self.doc_sources = []

    def getW(self):
        """Return live embedding vectors (always current, even after insert/remove).

        Returns the nn.Parameter directly (not .data) so gradients flow through.
        """
        if self.wv is not None and self.wv._vectors is not None:
            return self.wv._vectors
        return None

    def setW(self, value):
        """Embedding W is managed by wv._vectors — setW is a no-op."""
        pass

    @staticmethod
    def _to_text(buf):
        """Convert a byte list, tensor, or string to a plain string."""
        if isinstance(buf, str):
            return buf
        if isinstance(buf, torch.Tensor):
            buf = buf.squeeze().tolist()
        return "".join(chr(int(c) & 0xFF) for c in buf).rstrip("\x00")

    def _token_stream(self, text):
        from parse import quick_parser
        return quick_parser(self._to_text(text))

    def _token_to_index(self, text):
        if text in self.pretrain.key_to_index:
            return self.pretrain.key_to_index[text]
        ch = text[0] if text else ' '
        return self.pretrain.key_to_index.get(ch, 0)

    def tokenize(self, data):
        """Tokenize a batch (strings or byte tensors) into token text lists."""
        return [[text for text, _ in self._token_stream(item)] for item in data]

    def create(self, nInput=None, nVectors=None, nDim=None, passThrough=True,
               wv=None, embedding_path=None, source=None, learning_rate=0.001,
               min_frequency=0.0, neg_samples=64,
               objectSize=0):
        """Initialise from WordVectors or load from embedding_path.

        Accepts the same positional signature as ``Basis.create()`` so it can
        be used as the ``what`` basis for ``InputSpace``.

        If *wv* is provided, use it directly.  Otherwise load from
        *embedding_path* if given (.pt or .txt).  When no file is found or
        path is None, starts with an empty vocabulary — new words are
        added dynamically during forward passes via ``insert()``.

        If *source* is provided, build per-document lexer token streams and
        ensure every token text has an embedding row.
        """
        if wv is None:
            wv = self._load_embeddings(embedding_path=embedding_path, nDim=nDim)
        if wv is None:
            # No matching embeddings on disk — start with a single placeholder.
            # Real words are added dynamically during forward passes via insert().
            dim = nDim or 20
            print(f"Starting with dynamic {dim}-dim embedding (words added at runtime)")
            placeholder = torch.randn(1, dim, device=TheDevice)
            placeholder = F.normalize(placeholder, p=2, dim=1)
            wv = WordVectors(placeholder, ["<pad>"])
        self.wv = wv
        vocab_size = len(wv)
        vector_size = wv.vector_size

        # Verify dimensionality match: loaded vectors must agree with config nDim.
        # A mismatch (e.g. 100-dim sentence.pt loaded for nDim=10 config) causes
        # silent NaN during training as downstream layers expect different shapes.
        if nDim is not None and vector_size != nDim:
            raise ValueError(
                f"Embedding dimension mismatch: loaded vectors are {vector_size}-dim "
                f"but config requires nDim={nDim}. Check embeddings file or XML config.")

        super().create(nInput or max(1, vocab_size), nVectors or max(1, vocab_size),
                       vector_size, passThrough=passThrough,
                       objectSize=objectSize)
        # W is managed by wv._vectors; getW() returns live data

        self.pretrain = PretrainModel(wv, learning_rate=learning_rate, neg_samples=neg_samples)
        self.wv = wv  # register as submodule for .to(device) and state_dict
        self.min_frequency = float(min_frequency)
        self._pending_counts: dict = {}

        # Add \x00 as the EOS/padding sentinel.
        # A real character with a learned embedding — used to fill padding
        # positions in _forward_lex so the model has a concrete reconstruction
        # target for positions beyond the real tokens.
        self.insert("\x00")

        # Bootstrap codebook with printable ASCII (1–127) so any OOV word
        # can be spelled out character-by-character.
        for cp in range(1, 127):
            ch = chr(cp)
            if ch not in self.pretrain.key_to_index:
                self.insert(ch)

        # Pre-tokenize source documents for later inspection and optional reuse.
        if source:
            self.doc_spans = []
            self.doc_sources = []
            n_docs = len(source)
            for i, doc in enumerate(source):
                if i % 500 == 0:
                    print(f"  Building span table: {i}/{n_docs} docs")
                doc_bytes = doc.encode('utf-8')
                doc_tensor = torch.frombuffer(doc_bytes, dtype=torch.uint8).clone()
                self.doc_sources.append(doc_tensor)
                doc_tokens = self._token_stream(doc)
                self.doc_spans.append(doc_tokens)
                for token_text, _ in doc_tokens:
                    if token_text not in self.wv:
                        self.insert(token_text)

    def _rebuild_optimizer(self):
        self.pretrain.optimizer = torch.optim.Adam(
            [self.wv._vectors],
            lr=self.pretrain.optimizer.param_groups[0]['lr'],
        )
        # W is managed by wv._vectors; getW() returns live data

    def replace(self, new_W):
        new_W = self._coerce_rows(new_W).to(TheDevice)
        with torch.no_grad():
            self.wv._vectors = nn.Parameter(new_W, requires_grad=True)
        # W is managed by wv._vectors; getW() returns live data
        self.wv._normed = None
        if self.pretrain is not None:
            self._rebuild_optimizer()
        return self.getW()

    def insert(self, word, vector=None, initial_count=0):
        """Add a word to the vocabulary and codebook.

        Extends the WordVectors, PretrainModel vocabulary, and nn.Embedding so
        that subsequent forward passes can look up this word.

        Args:
            word: string to add.
            vector: optional (dim,) or (1, dim) tensor. If None, a random
                    normalized vector is generated.
            initial_count: frequency count for the new word.

        Returns:
            The embedding vector for the new word.
        """
        dim = self.wv.vector_size
        if vector is not None:
            new_vec = vector.to(TheDevice)
            if new_vec.dim() == 1:
                new_vec = new_vec.unsqueeze(0)
        else:
            new_vec = torch.randn(1, dim, device=TheDevice)
        new_vec = F.normalize(new_vec, p=2, dim=1)

        # Extend WordVectors parameter
        with torch.no_grad():
            new_data = torch.cat([self.wv._vectors.data, new_vec.to(self.wv._vectors.device)], dim=0)
            self.wv._vectors = nn.Parameter(new_data, requires_grad=True)
        self.wv.counts = np.append(self.wv.counts, np.int64(initial_count))
        self._pending_counts.pop(word, None)
        self.wv._normed = None  # invalidate cache
        idx = len(self.wv.index_to_key)
        self.wv.index_to_key.append(word)
        self.wv.key_to_index[word] = idx

        # PretrainModel shares wv's mappings
        self.pretrain.index_to_key = self.wv.index_to_key
        self.pretrain.key_to_index = self.wv.key_to_index

        self._rebuild_optimizer()
        # W is managed by wv._vectors; getW() returns live data

        return new_vec.squeeze(0)  # (1, dim) -> (dim,); preserves 1-dim case

    def remove(self, words_or_indices):
        """Remove words by name or index from vocabulary and codebook.

        Args:
            words_or_indices: list of word strings or integer indices to remove.

        Updates nn.Embedding, WordVectors, PretrainModel mappings, special token
        indices, and the optimizer in one shot.
        """
        if not words_or_indices:
            return
        # Resolve to indices
        if isinstance(words_or_indices[0], str):
            indices = [self.pretrain.key_to_index[w] for w in words_or_indices
                       if w in self.pretrain.key_to_index]
        else:
            indices = list(words_or_indices)
        if not indices:
            return

        # Shrink WordVectors (parameter + vocab)
        self.wv.remove(indices)

        # PretrainModel shares wv's mappings
        self.pretrain.index_to_key = self.wv.index_to_key
        self.pretrain.key_to_index = self.wv.key_to_index

        self._rebuild_optimizer()
        # W is managed by wv._vectors; getW() returns live data

    @staticmethod
    def _load_embeddings(embedding_path=None, nDim=None):
        """Load embeddings from a specific path, or return None for dynamic vocab.

        Detects file type by extension:
        - .txt: word2vec text format (``<vocab_size> <vector_size>\\n<word> <floats>...``)
        - .pt:  torch-saved WordVectors dict

        When *nDim* is given, validates that loaded vector dimension matches.
        Returns None if no path given or file doesn't exist.
        """
        if embedding_path is None:
            return None

        if not os.path.exists(embedding_path):
            return None

        if embedding_path.endswith(".txt"):
            wv = WordVectors.load_word2vec_format(embedding_path)
        else:
            wv = WordVectors.load(embedding_path)

        if nDim is not None and wv.vector_size != nDim:
            return None

        print(f"Loading embeddings from {embedding_path}...")
        return wv

    @staticmethod
    def print_info(path):
        """Print a human-readable summary of a .kv embedding artifact.

        Does not require a model to be loaded.  Useful for diagnosing
        mismatches between a saved artifact and a changed XML config.
        """
        if not os.path.exists(path):
            print(f"Embedding file not found: {path}")
            return
        wv = WordVectors.load(path)
        vocab_size = len(wv)
        vector_size = wv.vector_size
        total_count = int(wv.total_count)
        print(f"Embedding file : {path}")
        print(f"  Vocab size   : {vocab_size:,}")
        print(f"  Vector size  : {vector_size}")
        print(f"  Total tokens : {total_count:,}")
        if vocab_size > 0 and wv.counts is not None and len(wv.counts) > 0:
            top_n = min(5, vocab_size)
            top_idx = np.argsort(wv.counts)[::-1][:top_n]
            top_words = [(wv.index_to_key[i], int(wv.counts[i])) for i in top_idx]
            print(f"  Top words    : {top_words}")

    # --- Public properties (encapsulate internals) --------------------
    @property
    def codebook_weight(self):
        """nn.Embedding weight tensor (read-only reference)."""
        return self.getW()

    @property
    def vocab_keys(self):
        """Ordered list of words in the codebook."""
        return self.wv.index_to_key

    @property
    def embedding_dim(self):
        """Content dimensionality (nWhat)."""
        return self.getW().shape[1]

    def embedding_parameters(self):
        """Return embedding parameters for optimizer inclusion."""
        return [self.wv._vectors]

    def parameters_for_optimizer(self):
        return self.embedding_parameters()

    def _ergodic_var(self, token_idx, device, dtype):
        """Map CBOW's per-word sigma to a noise scale in [0, 1]."""
        if (not self.ergodic) or (not self.training) or self.pretrain is None:
            return None
        sigma = getattr(self.pretrain, 'sigma', None)
        sigma_step = getattr(self.pretrain, 'sigma_step', 0)
        if (
            sigma is None or sigma_step <= 0 or token_idx is None
            or token_idx >= sigma.shape[0]
        ):
            return None
        null_idx = self.wv.key_to_index.get("\x00")
        if token_idx == null_idx:
            return None
        beta = float(getattr(self.pretrain, 'sigma_beta', 0.99))
        denom = 1.0 - (beta ** sigma_step)
        if denom <= 0:
            return None
        s = sigma[token_idx].detach().to(device=device, dtype=dtype) / denom
        s = s.clamp(min=0)
        return (s / (s + self.sigma_kappa)).clamp(0, 1)

    def _encode_vector(self, vec, token_idx=None):
        """Pad and normalize one embedding vector, with optional ergodic noise."""
        if vec.dim() != 1:
            vec = vec.squeeze()
        vec = vec.clone()
        var = self._ergodic_var(token_idx, TheDevice, vec.dtype)
        if var is not None and torch.any(var > 0):
            vec = vec + var * torch.randn_like(vec)
        vec = F.pad(vec, (0, self.objectSize))
        return F.normalize(vec, p=2, dim=0)

    def _nearest_idx(self, vec, codebook=None):
        if codebook is None:
            codebook = self.getW().detach()
        vec = vec.to(TheDevice)
        sims = F.cosine_similarity(vec.unsqueeze(0), codebook, dim=1)
        return sims.argmax().item()

    def forward(self, input, return_meta=False):
        """Tokenize via Lex, look up embedding vectors from codebook.

        Input: (batch, max_len) byte tensor from Data.
        Output: (batch, nInput, embeddingSize) padded embedding tensor.

        When
        ``ergodic`` is enabled during training, the returned content vectors
        receive additive noise scaled by per-word CBOW sigma.
        """
        if input.dim() == 3:
            input = input.squeeze(1)  # [batch, 1, len] -> [batch, len]
        if input.dim() == 1:
            input = input.unsqueeze(0)  # [len] -> [1, len]
        batch = input.shape[0]
        codebook = self.getW()
        batch_tokens = []
        span_counts = []
        final_offsets = []

        # Phase 1: Tokenize all batch items and collect OOV words
        all_streams = []
        oov_words = []
        oov_seen = set()
        for b in range(batch):
            stream = self._token_stream(input[b])
            all_streams.append(stream)
            for token_text, _ in stream:
                if (token_text not in self.pretrain.key_to_index
                        and token_text not in oov_seen):
                    oov_words.append(token_text)
                    oov_seen.add(token_text)

        # Phase 2: Batch-insert OOV words into codebook
        if oov_words:
            for word in oov_words:
                self.insert(word)
            codebook = self.getW()  # refresh after insert
            if self.optimize_embedding:
                model = getattr(self, '_model', None)
                if model is not None:
                    model.rebuild_optimizer()

        # Phase 3: Build result tensor (all tokens now in codebook)
        result = torch.zeros([batch, self.nInput, self.embeddingSize], device=TheDevice)
        for b in range(batch):
            stream = all_streams[b]
            n_tokens = min(len(stream), self.nInput)
            tokens = stream[:n_tokens]
            batch_tokens.append(tokens)
            span_counts.append(n_tokens)
            if tokens:
                last_text, last_start = tokens[-1]
                final_offsets.append(
                    last_start + len(last_text.encode('utf-8')))
            else:
                final_offsets.append(0)
            for i, (token_text, _) in enumerate(tokens):
                cb_idx = self._token_to_index(token_text)
                result[b, i, :] = self._encode_vector(
                    codebook[cb_idx], token_idx=cb_idx)
            # Fill remaining positions with NULL embedding (input buffer is null-terminated)
            null_idx = self.wv.key_to_index.get("\x00")
            if null_idx is not None:
                null_vec = self._encode_vector(codebook[null_idx], token_idx=null_idx)
                for i in range(n_tokens, self.nInput):
                    result[b, i, :] = null_vec
        if not return_meta:
            return result
        return result, {
            'tokens': batch_tokens,
            'span_counts': span_counts,
            'final_offsets': final_offsets,
        }

    def train_step(self, words, method='CBOW'):
        """One sentence-level embedding gradient step.  Returns loss or None.

        method: 'CBOW' — predict each word from padded leave-one-out context
                'SBOW' — predict each word from leave-one-out centroid (faster)
        """
        if method == 'SBOW':
            return self.pretrain.sbow_step(words)
        return self.pretrain.train_step(words)

    def sbow_loss(self, words):
        """Return SBOW loss tensor for joint optimization (no backward/step).

        The caller adds this to the model loss before a single backward pass.
        Returns a scalar loss tensor, or None if insufficient words.
        """
        return self.pretrain.sbow_loss(words)

    def save_embeddings(self, path):
        """Save current embedding vectors and vocabulary to a .pt file."""
        self.wv.save(path)

    def reverse_raw(self, y):
        """Return the raw reverse-path vector with all subspaces intact.

        Does NOT strip encoding overhead — the full vector (nWhat + nWhere + nWhen)
        is preserved for MSE loss computation.

        Returns: raw tensor [batch, nVec, embSize] with positions intact.
        """
        return y

    def _decode_offset(self, positions, batch_idx, vector_idx, subspace=None):
        where_encoding = None if subspace is None else subspace.whereEncoding
        if where_encoding is None:
            return None
        sin_val = positions[batch_idx, vector_idx, 0].item()
        cos_val = positions[batch_idx, vector_idx, 1].item()
        if abs(sin_val) < 1e-8 and abs(cos_val) < 1e-8:
            return None
        return round(where_encoding.decode(torch.tensor([sin_val, cos_val])).item())

    def _render_tokens(self, batch_tokens, buf_size=None):
        has_positions = any(offset is not None for _, offset in batch_tokens)
        if not has_positions:
            text = []
            for word, _ in batch_tokens:
                if word != "":
                    text.append(word)
                if word == "\x00":
                    break
            return "".join(text)

        terminator = None
        max_end = 0
        for word, offset in batch_tokens:
            if offset is None or offset < 0:
                continue
            if word == "\x00":
                terminator = offset if terminator is None else min(
                    terminator, offset)
                continue
            if word == "":
                continue
            max_end = max(max_end, offset + len(word.encode('utf-8')))

        limit = (terminator + 1) if terminator is not None else max_end
        if buf_size is not None:
            limit = buf_size if limit <= 0 else min(limit, buf_size)
        if limit <= 0:
            return ""

        buf = [' '] * limit
        if terminator is not None and 0 <= terminator < limit:
            buf[terminator] = "\x00"
        for word, offset in batch_tokens:
            if word in ("", "\x00") or offset is None:
                continue
            if offset < 0 or offset >= limit:
                continue
            for ci, ch in enumerate(word):
                pos = offset + ci
                if pos >= limit:
                    break
                buf[pos] = ch
        return "".join(buf)

    @staticmethod
    def _decoded_tokens(decoded):
        if decoded is None:
            raise RuntimeError("decode metadata is required")
        if isinstance(decoded, dict):
            return decoded.get('tokens', [])
        return decoded

    def _get_codebook(self):
        """Return the live codebook parameter (Embedding uses WordVectors)."""
        return self.getW()

    def reverse(self, y, return_meta=False, subspace=None):
        """Snap content vectors to the nearest embedding rows.

        The caller owns aux split/restore. When ``return_meta=True``, metadata is
        decoded from the snapped content tensor only; callers that need restored
        position/time metadata should call ``decode_reverse_meta()`` after
        reattaching aux fields.
        """
        if self.passThrough:
            content = y
        else:
            if y.shape[-1] < self.embedding_dim:
                raise RuntimeError(
                    f"Embedding.reverse() expected at least {self.embedding_dim} "
                    f"content dims, got shape {list(y.shape)}.")
            content = y.clone() if y.shape[-1] == self.embedding_dim else y[:, :, :self.embedding_dim].clone()
            content = self._snap_content(content, weight=self.getW(), nWhat=self.embedding_dim)
        if return_meta:
            return content, self.decode_reverse_meta(content, subspace=subspace)
        return content

    def decode_reverse_meta(self, vectors, subspace=None):
        """Recover lexical metadata from restored reverse-path vectors."""
        embSize = vectors.shape[-1]
        positions = None
        times = None
        where_encoding = None if subspace is None else subspace.whereEncoding
        when_encoding = None if subspace is None else subspace.whenEncoding
        if where_encoding is not None and where_encoding.nDim > 0:
            where_idx = where_encoding.resolve(embSize)
            if where_idx[0] >= 0 and where_idx[0] < embSize:
                positions = vectors[:, :, where_idx]
        if when_encoding is not None and when_encoding.nDim > 0:
            when_idx = when_encoding.resolve(embSize)
            if when_idx[0] >= 0 and when_idx[0] < embSize:
                times = vectors[:, :, when_idx]

        batch = vectors.shape[0]
        nVec = vectors.shape[1]
        codebook = self.getW()
        words_list = self.wv.index_to_key
        nWhat = codebook.shape[1]

        recovered_words = [[] for _ in range(batch)]
        recovered_offsets = [[] for _ in range(batch)]
        recovered_tokens = [[] for _ in range(batch)]
        for b in range(batch):
            for v in range(nVec):
                offset = self._decode_offset(positions, b, v, subspace=subspace) if positions is not None else None
                vec = vectors[b, v, :nWhat]
                # Content is already snapped; _nearest_idx confirms the word label
                idx = self._nearest_idx(vec, codebook=codebook)
                word = words_list[idx]
                recovered_words[b].append(word)
                recovered_offsets[b].append(offset)
                recovered_tokens[b].append((word, offset))
        meta = {
            'tokens': recovered_tokens,
            'words': recovered_words,
            'positions': positions,
            'times': times,
            'offsets': recovered_offsets,
        }
        return meta

    def reconstruct_data(self, decoded, text=False):
        """Render recovered words from explicit reverse() metadata.

        Args:
            decoded: reverse() metadata dict or token stream
            text: If True, return list of joined strings. If False, return
                  list of word lists (one per batch element).

        Returns:
            List of word lists or joined strings, one per batch element.
            Empty words (from zero-padded vectors) are stripped.
        """
        batch_tokens_list = self._decoded_tokens(decoded)
        if text:
            return list(self.reconstruct_to_buffer(decoded))
        result = []
        for batch_tokens in batch_tokens_list:
            words = []
            for word, _ in batch_tokens:
                if word == "\x00":
                    break
                if word != "":
                    words.append(word)
            result.append(words)
        return result

    def get_recovered_word(self, decoded, batch_idx, position):
        """Return a single recovered word from explicit reverse() metadata."""
        rt = self._decoded_tokens(decoded)
        if batch_idx < len(rt) and position < len(rt[batch_idx]):
            return rt[batch_idx][position][0]
        return None

    def reconstruct_to_buffer(self, decoded, buf_size=None):
        """Place recovered words at nWhere byte offsets in a character buffer.

        Args:
            decoded: reverse() metadata dict or token stream
            buf_size: buffer length in characters. Default: input byte-tensor
                      length (self.nInput * 4 as rough upper bound, or 256).

        Returns:
            List of strings, one per batch element.
        """
        batch_tokens_list = self._decoded_tokens(decoded)
        results = []
        for batch_tokens in batch_tokens_list:
            results.append(self._render_tokens(batch_tokens, buf_size=buf_size))
        return results

    def predict(self, vector):
        """Decode output vector(s) via ``reverse()``.

        Args:
            vector: [batch, 1, embeddingSize] or [batch, embeddingSize]
        Returns:
            List of predicted words (one per batch element).
        """
        if vector.dim() == 2:
            vector = vector.unsqueeze(1)
        content = self.reverse(vector)
        meta = self.decode_reverse_meta(content)
        return [words[0] if words else "" for words in meta['words']]

    def embed_token(self, word):
        """Look up a single word in the codebook, return its content vector.

        Returns:
            Tensor of shape [nWhat] (content dims only).
        """
        if word in self.pretrain.key_to_index:
            idx = self.pretrain.key_to_index[word]
        else:
            ch = word[0] if word else ' '
            idx = self.pretrain.key_to_index.get(ch, 0)
        return self.getW()[idx].detach()

    def get_space_embedding(self):
        """Return the codebook embedding for the space character ' '."""
        return self.embed_token(' ')

    def get_mask_embedding(self):
        """Return a zero vector the same size as a codebook entry."""
        return torch.zeros(self.getW().shape[1], device=TheDevice)

class SubSpace(nn.Module):
    """Per-space runtime state container.

    Holds the factored representation (what, where, when, activation) along
    with the encoding objects that describe each factor.  Spaces populate
    a SubSpace during forward() and will eventually pass it as output.

    ``materialize()`` bridges back to the current dense-tensor API so
    existing Model.py math continues to work unchanged.

    In v1 this is a plain Python object (not nn.Module) because it holds
    references rather than learnable parameters.
    """

    def __init__(self, inputShape, outputShape,
                 flatten=False,
                 objectEncoding=None, activeEncoding=None, whatEncoding=None,
                 whereEncoding=None, whenEncoding=None,
                 object=None, what=None, where=None, when=None, activation=None):
        super().__init__()
        self.inputShape = inputShape    # [nActive, nDim]
        self.outputShape = outputShape  # [nActive, nDim]

        self.activeEncoding = activeEncoding
        self.whereEncoding = whereEncoding if whereEncoding is not None else WhereEncoding(0, 0)
        self.whenEncoding = whenEncoding if whenEncoding is not None else WhenEncoding(0, 0)
        self.whatEncoding = whatEncoding if whatEncoding is not None else WhatEncoding(inputShape, outputShape)
        self.objectEncoding = objectEncoding if objectEncoding is not None else ObjectEncoding(
            inputShape, outputShape, flatten=flatten)

        self.flatten = flatten
        self.objectSize = self.whereEncoding.nDim + self.whenEncoding.nDim

        self.object = self._coerce_basis(object, role="object")
        self.what   = self._coerce_basis(what, role="what")
        self.where  = self._coerce_basis(where, role="where")
        self.when   = self._coerce_basis(when, role="when")
        self.activation = self._coerce_basis(activation, role="activation")
        self.batch = 0
        payload = self.materialize()
        if isinstance(payload, torch.Tensor) and payload.ndim > 0:
            self.batch = payload.shape[0]

    def _coerce_basis(self, value, role="what"):
        if value is None or isinstance(value, Basis):
            return value
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"SubSpace {role} must be a Basis, Tensor, or None")
        basis = Tensor()
        if role == "object":
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.outputShape[1],
                passThrough=True,
                objectSize=self.objectSize,
            )
        elif role == "what":
            basis.create(
                self.outputShape[0],
                self.outputShape[0],
                self.outputShape[1],
                passThrough=True,
                objectSize=self.objectSize,
            )
        elif role == "where":
            basis.create(self.outputShape[0], self.outputShape[0], self.whereEncoding.nDim, passThrough=True)
        elif role == "when":
            basis.create(self.outputShape[0], self.outputShape[0], self.whenEncoding.nDim, passThrough=True)
        else:
            last_dim = value.shape[-1] if value.ndim > 1 else 1
            n_vectors = value.shape[-1] if value.ndim == 1 else value.shape[-2]
            basis.create(n_vectors, n_vectors, last_dim, passThrough=True)
        basis.setW(value)
        return basis

    def get_vectors(self):
        return self.object

    def set_vectors(self, vectors):
        """Store the current dense vectors (forward output) for materialize().

        This is separate from the object's codebook (getW/setW) — for Embedding,
        getW() returns the codebook while set_vectors stores the forward result.
        """
        if self.object is None:
            self.object = Tensor()
        self._vectors = vectors
        self.object.setW(vectors)
        self.set_activation_vectors(vectors)

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------

    def getEncodingSize(self, nDim):
        """Full vector width: nDim + objectSize."""
        return nDim + self.objectSize

    def getEncodedInputSize(self):
        """Return flattened input size (inputShape already includes objectSize)."""
        we = self.objectEncoding
        size = we.inputShape[1]
        if we.flatten:
            size *= we.inputShape[0]
        return size

    def getEncodedOutputSize(self):
        """Return flattened output size (outputShape already includes objectSize)."""
        we = self.objectEncoding
        size = we.outputShape[1]
        if we.flatten:
            size *= we.outputShape[0]
        return size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        """Shape of the current dense object tensor (or None if unset)."""
        payload = self.materialize()
        return None if payload is None else payload.shape

    # ------------------------------------------------------------------
    # Reshape / validate helpers (delegates to objectEncoding)
    # ------------------------------------------------------------------

    # forwardBegin/End, reverseBegin/End removed — inlined into Space.forwardBegin/End

    def flatten(self, x, forward=True):
        """Collapse [batch, nObj, dim] -> [batch, nObj*dim]."""
        return self.objectEncoding._flatten(x, self.batch, forward=forward)

    def unflatten(self, y, forward=True):
        """Restore [batch, nObj*dim] -> [batch, nObj, dim]."""
        return self.objectEncoding._unflatten(y, self.batch, forward=forward)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_tensor(cls, tensor, *, flatten=False,
                    inputShape, outputShape, **kw):
        """Wrap an existing dense tensor into a SubSpace."""
        return cls(flatten=flatten,
                   inputShape=inputShape, outputShape=outputShape,
                   object=tensor, **kw)

    @classmethod
    def from_components(cls, *, object=None, what=None, activation=None,
                        where=None, when=None, **kw):
        """Build a SubSpace with explicit factor values."""
        return cls(object=object, activation=activation, where=where, when=when, **kw)

    # ------------------------------------------------------------------
    # Activation management
    # ------------------------------------------------------------------

    def set_activation_vectors(self, y):
        """Compute and store subspace activation from dense vectors [B, N, D]."""
        assert y.ndim == 3, "Must be dim==3"
        activation = torch.norm(y, dim=-1)  # [B, N]
        self.set_activation(activation)

    def set_activation(self, activation_tensor):
        """Store scalar subspace activation for each object vector.

        Args:
            activation_tensor: [batch, nVectors] or [batch, nVectors, 1]
                scalar activation values per vector.
        """
        if activation_tensor.ndim == 3:
            assert activation_tensor.shape[-1] == 1, \
                f"activation last dim must be 1, got {activation_tensor.shape}"
            activation_tensor = activation_tensor.squeeze(-1)
        assert activation_tensor.ndim == 2, \
            f"activation must be [batch, nVectors], got {activation_tensor.shape}"
        self._activation = activation_tensor

    def get_activation(self):
        """Return stored activation [batch, nVectors] or None."""
        return getattr(self, '_activation', None)

    def materialize(self, k=None):
        """Return dense tensor of the top-k most active object vectors.

        When activations have been set via set_activation(), selects the k
        vectors with highest activation and returns them as a dense tensor.

        Args:
            k: number of vectors to return. If None, returns all vectors.

        Returns:
            Tensor [batch, k, dim] of the k highest-activation vectors.
            Stores selection indices in self._topk_indices for downstream use.
        """
        # _vectors holds the dense forward result (set by set_vectors).
        # Falls back to object.getW() (the codebook) if no forward has run.
        x = getattr(self, '_vectors', None)
        if x is None:
            if self.object is None:
                return None
            x = self.object.getW()
        if x is None:
            return None

        activation = self.get_activation()
        if k is None or activation is None:
            return x

        nSpace = x.shape[1]
        if k >= nSpace:
            return x

        # Select top-k by activation
        _, indices = torch.topk(activation, k, dim=1)  # [batch, k]
        self._topk_indices = indices

        # Gather the corresponding vectors: [batch, k, dim]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        selected = torch.gather(x, 1, indices_expanded)
        return selected

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, objects, what=False, where=True, when=True, pad=False):
        """Stamp positional/temporal encodings onto object vectors."""
        if pad:
            objects = self.pad(objects)
        if what:
            objects = self.whatEncoding(objects)
        if where:
            objects = self.whereEncoding(objects)
        if when:
            objects = self.whenEncoding(objects)
        return objects

    def decode(self, objects):
        """Extract positional/temporal components. Returns (objects, space, time)."""
        objects, space = self.whereEncoding.reverse(objects)
        objects, time = self.whenEncoding.reverse(objects)
        return objects, space, time

    def pad(self, objects, where=True, when=True):
        """Pad vectors with zeros for where/when encoding slots."""
        size = 0
        size += self.whereEncoding.nDim if where else 0
        size += self.whenEncoding.nDim if when else 0
        return F.pad(objects, (0, size))

    def slice(self, object, where=True, when=True):
        """Slice off where/when encoding dims from the end of a vector."""
        size = 0
        size += self.whereEncoding.nDim if where else 0
        size += self.whenEncoding.nDim if when else 0
        if size == 0:
            return object
        return object[0:-size]
    
class Space(nn.Module):
    """Base class for all spaces in the processing pipeline.

    The model is organized as a chain of spaces, each transforming object
    vectors from one representation to the next:

        InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace

    When ``reversible=True``, the chain also runs in reverse (OutputSpace
    back to InputSpace), enabling reconstruction of the input from the latent
    representation.

    Key parameters:
      - ``inputShape``/``outputShape``: each is [nObjects, nDim] describing the
        count and content-dimensionality of vectors entering/leaving this space.
        nDim is read from ``TheXMLConfig`` (per-space config section).
      - ``nVectors``: codebook size, also from config.  May differ from
        nActive (the active count); the factory validates ``nVectors >= nActive``.
      - ``reshape``: when True, flattens [batch, nObj, dim] -> [batch, nObj*dim]
        before passing through layers, then unflattens after.  Required when the
        input and output object counts differ (since layers operate on the last dim).
      - ``processSymbols``: when True, reduces full embedding vectors to scalar
        activations (norms) for the symbolic representation.
      - ``quantized``: when True, input vectors are quantized against the
        current ``Codebook`` basis after the main layer transformation.


    ``set_sigma(sigma)`` propagates exploration meta-parameters (1=explore, 0=suppress)
    from BasicModel down to all layers and Basis slots in this space.
    """
    name         = ""
    activation   = None
    config_section = None  # set by subclasses

    def __init__(self, inputShape, spaceShape, outputShape, customVQ=True):
        super(Space, self).__init__()
        section = self.config_section
        self.inputShape   = inputShape   # [nInput,   nInputDim]
        self.spaceShape   = spaceShape   # [nVectors, nDim]  — codebook / internal basis
        self.outputShape  = outputShape  # [nOutput,  nOutputDim]
        self.nVectors     = spaceShape[0]  # codebook size
        self.nDim         = spaceShape[1]  # content dimensionality of the codebook vectors
        self.flatten      = TheXMLConfig.space(section, "flatten")
        self.reversible   = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self.quantized    = TheXMLConfig.space(section, "quantized")
        try:
            _nWhere = TheXMLConfig.space(section, "nWhere")
        except KeyError:
            _nWhere = 0
        try:
            _nWhen = TheXMLConfig.space(section, "nWhen")
        except KeyError:
            _nWhen = 0
        self.objectSize = _nWhere + _nWhen
        self.customVQ  = customVQ
        # inputShape/outputShape already include objectSize in dim (set by factory).
        objectEncoding = ObjectEncoding(inputShape, outputShape, flatten=self.flatten)
        whatEncoding   = WhatEncoding(inputShape, outputShape)
        whereEncoding  = WhereEncoding(TheXMLConfig.get("architecture.nObjects"), _nWhere)
        whenEncoding   = WhenEncoding(10000, _nWhen)
        self.subspace  = SubSpace(
            flatten=self.flatten,
            inputShape=inputShape,
            outputShape=outputShape,
            objectEncoding=objectEncoding,
            whatEncoding=whatEncoding,
            whereEncoding=whereEncoding,
            whenEncoding=whenEncoding,
            object=self._build_object_basis(),
            what=self._build_what_basis(),
            where=self._build_where_basis(),
            when=self._build_when_basis(),
            activation=self._build_activation_basis(),
        )
        self.embeddingSize = self.subspace.getEncodingSize(self.nDim)
        self.params = []   # parameters for the optimizer (excludes temperature params)
        self.layers = nn.ModuleList()   # layer instances for paramUpdate() delegation
        self._register_requirements()

    def _build_object_basis(self):
        basis = Codebook()
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
            customVQ=self.customVQ,
            passThrough=not self.quantized,
            objectSize=self.objectSize,
        )
        basis.ergodic = getattr(self, "ergodic", False)
        return basis

    def _build_what_basis(self):
        return None

    def _build_where_basis(self):
        return None

    def _build_when_basis(self):
        return None

    def _build_activation_basis(self):
        return None

    def _register_requirements(self):
        """Register base-class config requirements."""
        section_name = self.config_section
        nV = self.nVectors
        nA = self.outputShape[0]
        TheXMLConfig.require(
            lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv >= _na,
            f"{section_name}: nVectors ({nV}) must be >= nActive ({nA})"
        )
        if not self.quantized:
            TheXMLConfig.require(
                lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv == _na,
                f"{section_name}: non-quantized space requires nVectors ({nV}) == nActive ({nA})"
            )

    def get_vectors(self):
        """Convenience accessor — delegates to subspace."""
        return self.subspace.get_vectors()

    def forwardBegin(self, vspace):
        """Materialize input and optionally flatten for space-specific processing."""
        x = vspace.materialize()
        self.subspace.batch = x.shape[0]
        if self.subspace.flatten:
            x = x.reshape(x.shape[0], 1, -1)  # [B, N, D] -> [B, 1, N*D]
        return x

    def forwardEnd(self, x):
        """Optionally unflatten output, store in subspace, return SubSpace."""
        if self.subspace.flatten:
            self._pre_unflatten_shape = x.shape  # save for reverseBegin
            x = self.subspace.objectEncoding._unflatten(x, self.subspace.batch, forward=True)
        self.subspace.set_vectors(x)
        return self.subspace

    def reverseBegin(self, vspace):
        """Materialize output-side tensor for space-specific reverse processing.

        For flatten mode, restores the shape that existed before forwardEnd's
        unflatten, so the reverse layer sees the same tensor layout as during
        forward (e.g. invertible PiLayer's interleaved objects are preserved).
        """
        y = vspace.materialize()
        self.subspace.batch = y.shape[0]
        if self.subspace.flatten:
            pre = getattr(self, '_pre_unflatten_shape', None)
            if pre is not None:
                y = y.reshape(pre)
            else:
                y = y.reshape(y.shape[0], 1, -1)
        return y

    def reverseEnd(self, y):
        """Optionally unflatten output, store in subspace, return SubSpace."""
        if self.subspace.flatten:
            y = self.subspace.objectEncoding._unflatten(y, self.subspace.batch, forward=False)
        self.subspace.set_vectors(y)
        return self.subspace

    # _2d/_3d removed — all layers now operate on [..., D] natively.

    def lookup(self, x):
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.zeros([1,1, TheXMLConfig.space("ConceptualSpace", "nDim")], device=TheDevice), x[:,:,1:]], dim=2)
        output, index, _ = self.subspace.get_vectors().quantize(x)
        #output[:,:,0:conceptDim] = output[:,:,0:conceptDim] * activation  # multiply the codebook vector by the activation
        return output
    def dereference(self, symbols):
        # we get [ batch x nConcepts x symbolEmbedding ],
        # and must compute [ batch x nConcepts x conceptEmbedding ]
        batch = symbols.shape[0]
        nActive = self.outputShape[0]
        assert list(symbols.shape) == [batch, nActive, TheXMLConfig.space("SymbolicSpace", "nDim") + self.objectSize], "Incorrect input size for dereference"
        objects = torch.zeros(batch, nActive, self.embeddingSize, device=TheDevice)
        for b in range(batch):
            for s in range(nActive):
                x = self.lookup(symbols[b,s,:])
                objects[b,s,:] = x
        assert list(objects.shape) == [batch, nActive, self.embeddingSize], "Incorrect output size for dereference"
        return objects

    def stats(self, x):
        #codebookUse = self.subspace.get_vectors().codebookUse
        #TheMessage(f"{self.name} Codebook activation: { np.sum(self.subspace.get_vectors().codebookAct.get()) }")
        return
    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all layers and Basis slots."""
        for l in self.layers:
            if hasattr(l, 'set_sigma'):
                l.set_sigma(sigma)
        for basis in (self.subspace.what, self.subspace.where, self.subspace.when, self.subspace.activation):
            if basis is not None and hasattr(basis, 'set_sigma'):
                basis.set_sigma(sigma)
    def getParameters(self):
        return self.params
    def paramUpdate(self):
        for l in self.layers:
            l.paramUpdate()
class InputSpace(Space):
    """Receives the source buffer from Data() and encodes it as vectors.

    For text: delegates tokenization to Lex, which produces a span table
    (start, end, type) over the source buffer.  Each span is encoded as a
    vector with nWhat (token content via the active ``Basis`` / ``Codebook``)
    and nWhere
    (positional encoding from the span's start offset).  A whole sentence
    is sent at once as a batch of [nWhat + nWhere] vectors.

    For numeric data: the tensor path is unchanged — no span table, no Lex,
    and objectEncoding contributes nothing when nWhat/nWhere are absent.

    reverse() reconstructs the source buffer from the latent state by
    decoding nWhat back to token IDs and using nWhere for positioning.

    Future: parse.py integration
    ----------------------------
    parse.py can produce constituent span tables (NP, VP, PP, etc.) over
    the same source buffer that Lex tokenizes.  If a convention is found
    for representing syntactic constituent information in per-space encoding
    (e.g. a new nSyntax dimension, or extending symbolDim to carry the
    constituent type), then InputSpace could encode both word-level and
    constituent-level spans.  This would require:
      1. A shared grammar available to both InputSpace and OutputSpace.
      2. An objectEncoding dimension convention for constituent types
         agreed upon by both sides.
      3. OutputSpace would use a generative grammar (inverse of parse.py's
         analytical grammar) to expand constituent-tagged symbols into
         structured token sequences in the destination buffer.
    Until such a convention is established, parse.py is not used here.
    """
    name = "Inputs"
    config_section = "InputSpace"

    def _build_object_basis(self):
        if self.model_type == "embedding":
            basis = Embedding()
            basis.ergodic = self.ergodic
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.nDim,
                embedding_path=self.embedding_path,
                source=self.embedding_source,
                min_frequency=self.min_frequency,
                neg_samples=self.neg_samples,
                objectSize=self.objectSize,
            )
            return basis

        if self.model_type == "vq":
            basis = Codebook()
            basis.create(
                self.inputShape[0],
                self.nVectors,
                self.nDim,
                customVQ=self.customVQ,
                passThrough=False,
                objectSize=self.objectSize,
            )
            return basis

        if self.model_type in ("passthrough", "simple"):
            basis = Tensor()
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.nDim,
                passThrough=True,
                objectSize=self.objectSize,
            )
            return basis

        raise RuntimeError("Unexpected model_type")

    def __init__(self, inputShape, spaceShape, outputShape, model_type="simple"):

        section = self.config_section
        ergodic = TheXMLConfig.get("architecture.ergodic")
        lexer = TheXMLConfig.space(section, "lexer")
        min_frequency = float(TheXMLConfig.data_param("minFrequency", 0.0))
        neg_samples = int(TheXMLConfig.training("negSamples", 64))
        embedding_path = TheXMLConfig.get("architecture.embeddingPath", None) or None
        data = TheData
        self.data = data
        self.model_type = model_type
        self.lexer = lexer  # "word", "sentence", or "grammar" — selects .cfg file
        self.ergodic = ergodic
        self.min_frequency = float(min_frequency)
        self.neg_samples = neg_samples
        self.embedding_path = embedding_path
        self.embedding_source = data.train_input if data.train_input else None
        super().__init__(inputShape, spaceShape, outputShape)
        # InputSpace never flattens — it operates on raw [batch, nObj, dim] tensors.
        # Flatten is applied by downstream spaces (Perceptual, Conceptual, etc.).
        self.flatten = False
        self.subspace.flatten = False
        self.subspace.objectEncoding.flatten = False
        lexical_basis = self.subspace.object
        if isinstance(lexical_basis, Embedding):
            self.doc_spans = lexical_basis.doc_spans
            self.doc_sources = lexical_basis.doc_sources
            if data.train_input and self.subspace.whereEncoding.nDim > 0:
                maxP = max(self.subspace.whereEncoding.maxVal, data.inputLength)
                self.subspace.whereEncoding.maxVal = maxP
                self.subspace.whereEncoding.div_term = 2 * math.pi / maxP
        else:
            self.doc_spans = []
            self.doc_sources = []

        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = False
        fullSize  = outputShape[0]*outputShape[1]
        self.lift = LiftingLayer(fullSize, fullSize)
        self.params = self.lift.getParameters()
        self.layers = nn.ModuleList([self.lift])
    # Data client interface
    def getTrainData(self):
        return self.data.train_input, self.data.train_output
    def getTestData(self):
        return self.data.test_input, self.data.test_output
    def prepInput(self, inputBatch):
        if isinstance(inputBatch, list):
            tensors = [self.data.stringTensor(s) if isinstance(s, str) else s
                       for s in inputBatch]
            return torch.stack(tensors, dim=0).unsqueeze(1).to(TheDevice)
        return inputBatch  # already [B, D, 1] and on device after toDevice()
    def shuffle(self):
        self.data.shuffle()
    # The world presenting itself
    def forward(self, input, mask=None):
        # ARIR cache bypass: if _cached_embedding is set, use it directly
        # instead of re-lexing / re-embedding.
        cached = getattr(self, '_cached_embedding', None)
        if cached is not None:
            self._cached_embedding = None  # consume once
            self.input = cached
            self._forward_input = None
            self.subspace.set_vectors(self.input)
            return self.subspace

        # Reset positional counter for each forward pass
        self.subspace.whereEncoding.p = 0

        batch = input.shape[0]
        object_basis = self.subspace.get_vectors()
        lexical_basis = self.subspace.object
        if not isinstance(lexical_basis, Embedding):
            assert list(input.shape) == [batch, self.inputShape[0], self.inputShape[1]]
            self.input = object_basis.forward(input)
            self._forward_input = None
            if self.objectSize > 0:
                self.input = self.subspace.encode(self.input)
            object_basis.setW(self.input)
        else:
            self.input, meta = lexical_basis.forward(input, return_meta=True)
            self._forward_input = meta
            if self.objectSize > 0:
                if self.subspace.whereEncoding.nDim > 0:
                    for b, batch_tokens in enumerate(meta['tokens']):
                        for i, (_, start) in enumerate(batch_tokens):
                            self.subspace.whereEncoding.stamp(
                                self.input, b, i, start)
                        final_offset = meta['final_offsets'][b]
                        for i in range(len(batch_tokens), self.outputShape[0]):
                            pad_offset = final_offset + (i - len(batch_tokens))
                            self.subspace.whereEncoding.stamp(
                                self.input, b, i, pad_offset)
                if self.subspace.whenEncoding.nDim > 0:
                        self.input = self.subspace.encode(
                            self.input,
                            where=False,
                            when=True,
                        )
            lexical_basis.setW(self.input)
        object_basis.setW(self.input)

        output = self.subspace.getEncodedOutputSize()
        assert list(self.input.shape) == [batch, self.outputShape[0], output]

        self.subspace.set_vectors(self.input)
        return self.subspace
    
    def expand_masked(self, embedded, sentence_text, maskedPrediction='MLM'):
        """Expand one sentence's embedding into N masked copies.

        Modes:
            MLM: Zero content at position i, preserve position encoding.
            ARLM: Zero content at position i, truncate all future positions (j > i).
            ARUS: Same as ARLM (output-side behavior differs in OutputSpace).
            RARLM: Zero content at position (N-1-i), truncate all previous positions (j < pos).

        Args:
            embedded: [1, nVectors, embeddingSize] output of forward()
            sentence_text: original sentence string (used for word count)
            maskedPrediction: prediction mode ('MLM', 'ARLM', 'ARUS', or 'RARLM')

        Returns:
            (masked_batch, mask_positions):
                masked_batch: [N, nVectors, embeddingSize]
                mask_positions: list[int] of length N
        """
        words = sentence_text.split()
        N = min(len(words), self.outputShape[0])  # cap at nVectors

        # Repeat the embedded sentence N times
        masked = embedded.expand(N, -1, -1).detach().clone()  # [N, nVec, embSize]

        # Determine which dims are content (to zero) vs position (to preserve)
        embSize = embedded.shape[-1]
        content_mask = torch.ones(embSize, dtype=torch.bool, device=TheDevice)
        # Preserve nWhere dims (indices [-4, -3] from end of embedding)
        if self.subspace.whereEncoding.nDim > 0:
            where_idx = np.add([embSize, embSize], self.subspace.whereEncoding.index)
            for wi in where_idx:
                if 0 <= wi < embSize:
                    content_mask[wi] = False
        # Preserve nWhen dims (indices [-2, -1] from end of embedding)
        if self.subspace.whenEncoding.nDim > 0:
            when_idx = np.add([embSize, embSize], self.subspace.whenEncoding.index)
            for wi in when_idx:
                if 0 <= wi < embSize:
                    content_mask[wi] = False

        # Determine mask position for each copy
        for i in range(N):
            pos = (N - 1 - i) if maskedPrediction == 'RARLM' else i
            masked[i, pos, content_mask] = 0.0

        # ARLM/ARUS: truncate all future positions (j > i)
        if maskedPrediction in ('ARLM', 'ARUS'):
            for i in range(N):
                if i + 1 < masked.shape[1]:
                    masked[i, i + 1:, :] = 0.0

        # RARLM: truncate all previous positions (j < pos)
        if maskedPrediction == 'RARLM':
            for i in range(N):
                pos = N - 1 - i
                if pos > 0:
                    masked[i, :pos, :] = 0.0

        if maskedPrediction == 'RARLM':
            return masked, list(range(N - 1, -1, -1))
        return masked, list(range(N))

    def reverse(self, vspace):
        y = self.reverseBegin(vspace)
        # Store full vector (all subspaces) for MSE loss BEFORE partitioning
        object_basis = self.subspace.get_vectors()
        content_basis = self.subspace.what if isinstance(self.subspace.what, Embedding) else object_basis
        raw = object_basis.reverse_raw(y)
        self.reconstructed = raw.detach()
        nWhat = content_basis.content_dim
        object_encoding = self.subspace.objectEncoding
        if object_encoding is not None:
            content, aux = object_encoding.split_aux(y, nWhat)
        else:
            if y.shape[-1] <= nWhat:
                content, aux = y.clone(), None
            else:
                content, aux = y[:, :, :nWhat].clone(), y[:, :, nWhat:].clone()
        # Partition into nWhat/nWhere/nWhen for display
        content = content_basis.reverse(content)
        if object_encoding is not None:
            self.input = object_encoding.restore_aux(content, aux)
        elif aux is not None:
            self.input = torch.cat([content, aux], dim=-1)
        else:
            self.input = content
        content_basis.setW(self.input)
        object_basis.setW(self.input)
        if isinstance(content_basis, Embedding):
            self._recovered_input = content_basis.decode_reverse_meta(
                self.input, subspace=self.subspace)
        else:
            self._recovered_input = None
        vspace = self.reverseEnd(self.input)
        return vspace

    def reconstruct_data(self, text=False):
        """Render the last recovered text state stored on InputSpace."""
        if getattr(self, '_recovered_input', None) is None:
            raise RuntimeError("reconstruct_data() called before reverse()")
        return self.subspace.get_vectors().reconstruct_data(self._recovered_input, text=text)

    def reconstruct_to_buffer(self, buf_size=None):
        """Render the last recovered text buffer stored on InputSpace."""
        if getattr(self, '_recovered_input', None) is None:
            raise RuntimeError("reconstruct_to_buffer() called before reverse()")
        return self.subspace.get_vectors().reconstruct_to_buffer(
            self._recovered_input, buf_size=buf_size)

    def get_forward_meta(self):
        """Return the last forward-pass lexical metadata for text input."""
        return getattr(self, '_forward_input', None)

    def get_recovered_word(self, batch_idx, position):
        """Return one recovered token from the last InputSpace.reverse()."""
        if getattr(self, '_recovered_input', None) is None:
            return None
        return self.subspace.get_vectors().get_recovered_word(
            self._recovered_input, batch_idx, position)

    # ------------------------------------------------------------------
    # Training policy — InputSpace decides WHEN, Embedding does HOW
    # ------------------------------------------------------------------

    def train_embeddings(self, words, method='CBOW'):
        """Run one CBOW/SBOW gradient step if words are available."""
        emb = self.subspace.get_vectors()
        if isinstance(emb, Embedding) and words:
            return emb.train_step(words, method=method)
        return None

    def sbow_loss(self, words):
        """Return SBOW loss tensor for joint optimization (no backward/step)."""
        emb = self.subspace.get_vectors()
        if isinstance(emb, Embedding) and words:
            return emb.sbow_loss(words)
        return None

    def _snapshot_embeddings(self):
        """Return the current WordVectors (no-op, vectors are always live)."""
        emb = self.subspace.get_vectors()
        if isinstance(emb, Embedding):
            return emb.wv
        return None

    def set_embedding_sigma(self, sigma):
        """Control exploration noise on the embedding."""
        emb = self.subspace.get_vectors()
        if hasattr(emb, 'set_sigma'):
            emb.set_sigma(sigma)

    def getBatch(self, batchNum, batchSize=10, split="train"):
        """Return next batch of (input, output) data and the next batchNum.

        For standard mode: slices train_input/train_output by batchSize.
        For masked prediction: takes sentence batchNum, embeds it,
            expands into N masked copies (one per word), computes N targets.
            Batch size is dynamic (= words in sentence).

        Args:
            batchNum: current batch index
            batchSize: number of examples per batch (standard mode only)
            split: "train", "test", or "validation"

        Returns:
            ((inputBatch, outputBatch), nextBatchNum)
            Returns (None, batchNum) when data is exhausted.
        """
        # Select data for the requested split
        if split == "train" or split == "runtime":
            inputData = self.data.train_input
            outputData = self.data.train_output
        elif split == "test":
            inputData = self.data.test_input
            outputData = self.data.test_output
        elif split == "validation":
            inputData = self.data.validation_input
            outputData = self.data.validation_output
        else:
            raise ValueError(f"Unknown split: {split}")

        # ── ARIR state machine ──────────────────────────────────────
        if split == "runtime" and getattr(self.data, '_runtime_mode', None) == 'ARIR':
            return self._getBatch_arir(inputData, batchNum)

        # Use standard (non-masked) path when: no masked prediction configured,
        # or runtime split with no sentences staged (inference via runBatch).
        # Raw strings for masked prediction — all splits store strings directly
        if (isinstance(inputData, list) and inputData
                and isinstance(inputData[0], str)):
            sentences = inputData
        else:
            sentences = None
        use_masked = (hasattr(self.data, 'masked_prediction')
                      and self.data.masked_prediction != 'NONE'
                      and sentences is not None)
        if not use_masked:
            # Standard mode: fixed-size batch slicing
            i = batchNum * batchSize
            if i >= len(inputData):
                return None, batchNum
            inputBatch = inputData[i:i + batchSize]
            inputTensor = self.prepInput(inputBatch)
            if outputData is not None:
                outputBatch = outputData[i:i + batchSize]
                outputTensor = self.outputSpace.prepOutput(outputBatch)
            else:
                outputTensor = None
            self._unmasked_embedding = None
            self._mask_positions = None
            return (inputTensor, outputTensor), batchNum + 1
        else:
            # Masked prediction: one sentence -> N masked examples.
            # Embed once, use that embedding for both targets and masked input.
            if batchNum >= len(sentences):
                return None, batchNum
            sentence = sentences[batchNum]
            inputTensor = self.prepInput(inputData[batchNum:batchNum + 1])

            # Embed once — retain gradient graph for the masked input path.
            # Targets are detached (they're labels, not part of the forward graph).
            embedded = self.forward(inputTensor).materialize()  # [1, nVec, embSize]

            # Compute targets from detached embedding (labels, no gradient needed)
            targets = self.outputSpace.expand_masked(
                embedded.detach(), sentence, self.data.masked_prediction)

            # Build masked copies from the live embedding (retains gradient graph)
            masked_batch, mask_positions = self.expand_masked(
                embedded, sentence, self.data.masked_prediction)

            # Cache unmasked embedding for reconstruction loss target
            self._unmasked_embedding = embedded.detach()  # [1, nVec, embSize]
            self._mask_positions = mask_positions           # list[int], len=N

            # Hand masked embedding to forward() via cache — no re-embedding,
            # but gradient flows back through masked_batch → embedded → embedding weights
            self._cached_embedding = masked_batch

            return (inputTensor, targets), batchNum + 1

    def get_reconstruction_target(self):
        """Return (target, mask) for reconstruction loss.

        target: [batch, nVec, embSize] — unmasked post-encoding embedding
        mask:   [batch, nVec] bool — True at masked positions to compute loss on.
                None when maskedPrediction=NONE (use whole buffer).
        """
        unmasked = getattr(self, '_unmasked_embedding', None)
        positions = getattr(self, '_mask_positions', None)
        if unmasked is None or positions is None:
            return None, None
        N = len(positions)
        nVec = unmasked.shape[1]
        target = unmasked.expand(N, -1, -1)
        mask = torch.zeros(N, nVec, dtype=torch.bool, device=TheDevice)
        for i, pos in enumerate(positions):
            mask[i, pos] = True
        return target, mask

    def predict(self, vector):
        """Delegates to Embedding.predict()."""
        return self.subspace.get_vectors().predict(vector)

    # ------------------------------------------------------------------
    # ARIR helpers
    # ------------------------------------------------------------------

    def embed_token(self, word):
        """Delegates to Embedding.embed_token()."""
        return self.subspace.get_vectors().embed_token(word)

    def get_space_embedding(self):
        """Delegates to Embedding.get_space_embedding()."""
        return self.subspace.get_vectors().get_space_embedding()

    def get_mask_embedding(self):
        """Delegates to Embedding.get_mask_embedding()."""
        return self.subspace.get_vectors().get_mask_embedding()

    # ── ARIR state machine ──────────────────────────────────────────

    def _getBatch_arir(self, inputData, batchNum):
        """ARIR state machine for getBatch(): embed seed, then iteratively
        place [MASK] and read back reconstructed latent vectors.

        First call (cursor is None):
            Embed seed text, fill future with spaces, place [MASK] at seed_len.

        Subsequent calls (cursor is not None):
            Read reconstruction from previous reverse pass, decode word,
            write reconstructed latent at previous cursor, advance, place [MASK].

        Returns None when cursor reaches nVec, EOF detected, or max_chars exceeded.
        """
        nVec = self.outputShape[0]
        embSize = self.subspace.get_vectors().embeddingSize
        nWhat = self.subspace.get_vectors().embedding_dim

        if self._arir_cursor is None:
            # ── First call: embed seed, prepare buffer ──────────────
            inputTensor = self.prepInput(inputData)
            embedded = self.forward(inputTensor)  # [1, nVec, embSize]
            self._arir_embedded = embedded.detach().clone()

            # Read span count from the lex pass
            meta = getattr(self, '_forward_input', None) or {}
            counts = meta.get('span_counts', [])
            seed_len = counts[0] if counts else 1

            # Read byte offset from the lex pass
            offsets = meta.get('final_offsets', [])
            self._arir_byte_offset = offsets[0] if offsets else 0

            # Fill future positions (seed_len .. nVec-1) with NULL embeddings
            null_emb = self.subspace.get_vectors().embed_token("\x00")
            for k in range(seed_len, nVec):
                self._arir_embedded[0, k, :nWhat] = null_emb[:nWhat]
                est_offset = self._arir_byte_offset + (k - seed_len)
                self._arir_stamp_where(self._arir_embedded, k, est_offset)

            # Set cursor and place [MASK] at seed_len
            self._arir_cursor = seed_len
            self._arir_embedded[0, self._arir_cursor, :nWhat] = 0.0
            self._arir_stamp_where(
                self._arir_embedded, self._arir_cursor, self._arir_byte_offset
            )

            # Inject via the cached-embedding bypass in forward()
            self._cached_embedding = self._arir_embedded.clone()

            # Return a dummy input tensor (forward() will use _cached_embedding)
            dummy_input = inputTensor
            return (dummy_input, None), batchNum + 1

        else:
            # ── Subsequent calls: read reconstruction, advance cursor ──
            prev_cursor = self._arir_cursor

            # Read decoded word from previous reverse pass
            word = self.get_recovered_word(0, prev_cursor)

            # EOF check
            if word is None or word == '' or word == '\x00':
                self._arir_reset()
                return None, batchNum

            # Record the token
            self._arir_tokens.append(word)
            self._arir_total_chars += len(word)

            # Max chars check
            if self._arir_total_chars >= self._arir_max_chars:
                self._arir_reset()
                return None, batchNum

            # Write reconstructed latent vector at current cursor
            # (no codebook lookup -- stay in latent space)
            recon = self.reconstructed  # [1, nVec, embSize] from reverse()
            self._arir_embedded[0, prev_cursor, :nWhat] = recon[0, prev_cursor, :nWhat]
            self._arir_stamp_where(
                self._arir_embedded, prev_cursor, self._arir_byte_offset
            )

            # Advance byte offset for positional encoding
            self._arir_byte_offset += len(word.encode('utf-8'))

            # Write reconstructed vectors beyond cursor as steering signal
            for k in range(prev_cursor + 1, nVec):
                self._arir_embedded[0, k, :nWhat] = recon[0, k, :nWhat]
                # Keep existing positional encoding at those positions

            # Advance cursor
            self._arir_cursor = prev_cursor + 1

            # Buffer full check
            if self._arir_cursor >= nVec:
                self._arir_reset()
                return None, batchNum

            # Place [MASK] at new cursor
            self._arir_embedded[0, self._arir_cursor, :nWhat] = 0.0
            self._arir_stamp_where(
                self._arir_embedded, self._arir_cursor, self._arir_byte_offset
            )

            # Inject via cached-embedding bypass
            self._cached_embedding = self._arir_embedded.clone()

            # Return dummy input (forward() will use _cached_embedding)
            dummy_input = torch.zeros(1, device=TheDevice)
            return (dummy_input, None), batchNum + 1

    def _arir_reset(self):
        """Reset all ARIR state attributes."""
        self._arir_cursor = None
        self._arir_embedded = None
        self._arir_byte_offset = 0
        self._arir_tokens = []
        self._arir_max_chars = 256
        self._arir_total_chars = 0

    def get_predicted_tokens(self):
        """Return the list of tokens predicted during ARIR inference."""
        return getattr(self, '_arir_tokens', [])

    def _arir_stamp_where(self, buf, pos_idx, byte_off):
        """Stamp positional encoding at a buffer position via subspace.whereEncoding."""
        if self.subspace.whereEncoding.nDim > 0:
            self.subspace.whereEncoding.stamp(buf, 0, pos_idx, byte_off)
class PerceptualSpace(Space):
    """Transforms raw input vectors into percepts via a PiLayer.

    In the forward data flow: InputSpace -> **PerceptualSpace** -> ConceptualSpace.
    Uses a PiLayer (permutation-equivariant layer) to map input embeddings to
    perceptual embeddings, optionally followed by self-attention and VQ
    codebook quantization.

    When ``reversible=True``, uses InvertiblePiLayer whose forward
    interleaves (log_y, log_z) pairs, doubling the output.  In 3D mode
    the doubling is along the sequence axis; in 2D (reshape) mode,
    the layer's nOutput is halved so the 2x produces the correct
    unflatten width.  With ``invertible=True``, a single layer serves
    both directions (shared weights).  Without invertibility, two
    InvertiblePiLayers with separate weights are used: forward() on
    one, reverse() on the other.  **Note:** the non-invertible reverse
    path currently involves a matrix pseudoinverse (``pinv``) which may
    be numerically unstable; this is not a recommended code path.

    ``passThrough=True`` makes this a no-op (identity), useful when the input
    is already in the desired perceptual form.
    """
    name = "Percepts"
    config_section = "PerceptualSpace"

    def __init__(self, inputShape, spaceShape, outputShape):

        section = self.config_section
        passThrough = TheXMLConfig.space(section, "passThrough")
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        naive = TheXMLConfig.get("architecture.naive")
        super().__init__(inputShape, spaceShape, outputShape)
        self.passThrough = passThrough
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        self.invertible = invertible
        if passThrough:
            return
        input = self.subspace.getEncodedInputSize()
        output = self.subspace.getEncodedOutputSize()
        unflatOutput = output
        # InvertiblePiLayer doubles its output; halve the layer's nOutput
        # so the 2x interleaving produces the correct unflatten width.
        if self.flatten and self.reversible:
            output = output // 2
        self.attention = AttentionLayer(unflatOutput, unflatOutput)
        if self.reversible:
            if invertible:
                self.pi  = PiLayer(input, output, naive=naive, ergodic=ergodic, invertible=True)
                self.forwardPi, self.reversePi = self.pi.forward, self.pi.reverse
                self.params = self.pi.getParameters()
                self.layers = nn.ModuleList([self.pi])
            else:
                self.pi1 = PiLayer(input, output, naive=naive, ergodic=ergodic, invertible=True)
                self.pi2 = PiLayer(input, output, naive=naive, ergodic=ergodic, invertible=True)
                self.forwardPi, self.reversePi = self.pi1.forward, self.pi2.reverse
                self.params = self.pi1.getParameters() + self.pi2.getParameters()
                self.layers = nn.ModuleList([self.pi1, self.pi2])
        else:
            self.pi        = PiLayer(input, output, naive=naive, ergodic=ergodic)
            self.forwardPi = self.pi.forward
            self.params = self.pi.getParameters()
            self.layers = nn.ModuleList([self.pi])
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)

    def _register_requirements(self):
        """Register PerceptualSpace-specific config requirements."""
        # passThrough spaces are identity mappings — shape constraints don't apply.
        passThrough = TheXMLConfig.space(self.config_section, "passThrough")
        if passThrough:
            return

        nV = self.nVectors
        nA = self.outputShape[0]   # nOutput
        nI = self.inputShape[0]    # nInput
        nI_dim = self.inputShape[1]
        nA_dim = self.outputShape[1]

        invertible = TheXMLConfig.space(self.config_section, "invertible")
        if invertible:
            if self.reversible:
                if self.flatten:
                    # 4*nInput*inputDim == nOutput*outputDim
                    TheXMLConfig.require(
                        lambda cfg, _ni=nI, _nid=nI_dim, _na=nA, _nad=nA_dim:
                            4 * _ni * _nid == _na * _nad,
                        f"PerceptualSpace: invertible+flatten requires 4*nInput*inputDim == nOutput*outputDim "
                        f"(got 4*{nI}*{nI_dim}={4*nI*nI_dim}, {nA}*{nA_dim}={nA*nA_dim})"
                    )
                else:
                    # nOutput == 2 * nInput
                    TheXMLConfig.require(
                        lambda cfg, _ni=nI, _na=nA: _na == 2 * _ni,
                        f"PerceptualSpace: invertible without flatten requires nOutput ({nA}) == 2*nInput ({nI})"
                    )
            # When invertible=True: skip nVectors checks (InvertiblePiLayer manages sizing)
        else:
            # Standard checks
            TheXMLConfig.require(
                lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv >= _na,
                f"PerceptualSpace: nVectors ({nV}) must be >= nOutput ({nA})"
            )
            if not self.quantized:
                TheXMLConfig.require(
                    lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv == _na,
                    f"PerceptualSpace: non-quantized requires nVectors ({nV}) == nOutput ({nA})"
                )

    def distance(self, x, y):
        return torch.prod( [1-x, 1-y] )
    def certainty(self, x):
        pass
    def forward(self, vspace):
        """Perception: map input vectors to percepts via PiLayer + optional attention + VQ."""
        if self.passThrough:
            return vspace
        x = self.forwardBegin(vspace)
        if self.hasAttention:
            x = self.attention.forward(x)
        x = self.forwardPi(x)
        if self.quantized:
            x = self.subspace.get_vectors().forward(x)
        vspace = self.forwardEnd(x)
        return vspace

    def reverse(self, vspace):
        """Manifesting: reconstruct input vectors from percepts via reverse PiLayer."""
        if self.passThrough:
            return vspace
        y = self.reverseBegin(vspace)
        if self.reversible:
            y = self.reversePi(y)
        vspace = self.reverseEnd(y)
        return vspace

    @staticmethod
    def test():
        pass
class ConceptualSpace(Space):
    """Transforms percepts into concepts via a SigmaLayer (summation layer).

    In the forward data flow: PerceptualSpace -> **ConceptualSpace** -> SymbolicSpace.
    Uses a SigmaLayer to combine perceptual features into conceptual
    representations.  The SigmaLayer computes weighted sums (inner products)
    rather than the permutation-equivariant operations of PiLayer.

    Supports optional NormLayer preprocessing, self-attention, and VQ codebook
    quantization.

    When ``invertible=True``, uses a InvertibleSigmaLayer whose inverse is
    exact.  When ``reversible=True`` without invertibility, a separate
    SigmaLayer is trained for the reverse direction.
    """
    name = "Concepts"
    config_section = "ConceptualSpace"

    def __init__(self, inputShape, spaceShape, outputShape):
        section = self.config_section
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        hasNorm = TheXMLConfig.space(section, "hasNorm")
        naive = TheXMLConfig.get("architecture.naive")
        super().__init__(inputShape, spaceShape, outputShape)
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        input = self.subspace.getEncodedInputSize()
        output = self.subspace.getEncodedOutputSize()
        self.hasNorm = hasNorm
        self.attention = AttentionLayer(output, output)
        if hasNorm:
            self.norm = NormLayer(input, input + 2)
            # Don't expand input dim for invertible path — norm factors
            # are cached during forward and reapplied during reverse,
            # so the sigma layer stays square (input == output).
            if not invertible:
                input += 2
        if self.reversible:
            if invertible:
                self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic, invertible=True)
                self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                self.params = self.sigma.getParameters()
                self.layers = nn.ModuleList([self.sigma])
            else:
                self.sigma1 = SigmaLayer(input, output, naive=naive, ergodic=ergodic, invertible=True)
                self.sigma2 = SigmaLayer(input, output, naive=naive, ergodic=ergodic, invertible=True)
                self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.reverse
                self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
                self.layers = nn.ModuleList([self.sigma1, self.sigma2])
        else:
            self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic)
            self.forwardSigma = self.sigma.forward
            self.params = self.sigma.getParameters()
            self.layers = nn.ModuleList([self.sigma])
    def distance(self, x, y):
        # This is a dot-product distance that assumes the X are normalized.
        # However, if the X are not normalized, the magnitudes may be taken as a degree of certainty or knowing.
        # In which case, how do they grow from ignorance to certainty?
        # They would do so naturally if the input vectors are normalized.
        # It would also be possible to use a tunable transfer function.
        return x.T @ y
    def certainty(self, x):
        return x.T @ x
    def forward(self, vspace):
        """Knowing: map percepts to concepts via SigmaLayer + optional attention + VQ."""
        x = self.forwardBegin(vspace)
        y = self.forwardSigma(x)
        if self.hasAttention:
            y = self.attention.forward(y)
        if self.quantized:
            y = self.subspace.get_vectors().forward(y)
        vspace = self.forwardEnd(y)
        return vspace

    def reverse(self, vspace):
        """Visualizing: reconstruct percepts from concepts via reverse SigmaLayer."""
        y = self.reverseBegin(vspace)
        if self.processSymbols:
            y = self.dereference(y)
        y = self.reverseSigma(y)
        self.concepts = y
        vspace = self.reverseEnd(y)
        return vspace

    @staticmethod
    def test():
        pass
class SymbolicSpace(Space):
    """Converts continuous concept vectors into discrete symbols.

    In the forward data flow: ConceptualSpace -> **SymbolicSpace** -> OutputSpace.
    Applies optional discretization (thresholding or top-k serial activation) to
    produce symbolic representations from concept embeddings.

    When ``processSymbols=True``, computes scalar activations (vector norms) from
    concept vectors.  In reverse, ``dereference()`` looks up the corresponding
    concept vector from the ConceptualSpace's codebook.

    ``passThrough=True`` skips all symbolic processing, passing concept vectors
    through unchanged (useful when the conceptual representation is sufficient).
    """
    name = "Symbols"
    config_section = "SymbolicSpace"
    threshold        = 0       # discretization threshold (0 = disabled)
    serialActivation = False   # if True, only the top-1 activation is kept per batch
    symbols          = None

    def __init__(self, inputShape, spaceShape, outputShape, conceptualSpace=None):

        section = self.config_section
        passThrough = TheXMLConfig.space(section, "passThrough")
        super().__init__(inputShape, spaceShape, outputShape, customVQ=True)
        self.conceptualSpace = conceptualSpace
        self.passThrough = passThrough
        #self.mapping     = TODO(inputShape[1], nDim, soft=False)
    def distance(self, x, y):
        return x == y
    def certainty(self, x):
        return x.T @ x
    def discretize(self, symbols):
        batch = symbols.shape[0]
        if self.serialActivation:
            for b in range(0,batch):
                top, indices = torch.topk(symbols[b,:], k=1)
                symbols[b,:] = 0
                symbols[b,indices] = top[indices]
        elif self.threshold:
            symbols[symbols > self.threshold] =  1
            symbols[symbols < self.threshold] = -1
        return symbols

    def forward(self, vspace):
        """Naming: convert concept vectors to discrete symbols."""
        x = self.forwardBegin(vspace)
        if not self.passThrough:
            x = self.discretize(x)
        if self.quantized:
            x = self.subspace.get_vectors().forward(x)
        vspace = self.forwardEnd(x)
        return vspace

    def reverse(self, vspace):
        """Interpretation: map symbols back to concept vectors (via codebook dereference)."""
        y = self.reverseBegin(vspace)
        if not self.passThrough:
            if self.processSymbols:
                y = self.conceptualSpace.dereference(y)
        self.symbols = y
        vspace = self.reverseEnd(y)
        return vspace

    @staticmethod
    def test():
        pass
class SyntacticSpace(Space):
    """Placeholder for syntactic processing between symbols and words.

    Currently a passthrough that reshapes symbols without transformation.
    Used when ``symbolicOrder >= 1`` to add an extra processing stage
    between the symbolic and output spaces.  Future work would integrate
    generative grammar operations here.
    """
    name  = "Syntactic"
    config_section = "SyntacticSpace"
    words = None

    def __init__(self, inputShape, spaceShape, outputShape, conceptualSpace=None):
        super().__init__(inputShape, spaceShape, outputShape, customVQ=False)
        self.processSymbols = True  # class invariant
        assert(inputShape[0] == outputShape[0]) # 1:1 mapping
        self.conceptualSpace = conceptualSpace
        #self.mapping     = TODO(inputShape[1], nDim, soft=False)
    def distance(self, x, y):
        return x == y
    def certainty(self, x):
        return x.T @ x
    def forward(self, vspace):
        x = self.forwardBegin(vspace)
        # Identity — no transform
        vspace = self.forwardEnd(x)
        return vspace

    def reverse(self, vspace):
        """Syntax reverse: passthrough."""
        y = self.reverseBegin(vspace)
        self.symbols = y
        vspace = self.reverseEnd(y)
        return vspace

    @staticmethod
    def test():
        pass
class OutputSpace(Space):
    """Maps symbolic vectors to task targets (classification logits, regression values).

    In the forward data flow: SymbolicSpace -> **OutputSpace** -> loss.
    Uses a LinearLayer to project the (flattened) symbolic representation down
    to the target dimensionality.  Always uses reshape=True since the number of
    input objects (symbols) typically differs from the number of outputs.

    ``text_mode``: when enabled via ``set_text_mode()``, supports reconstructing
    text from symbolic vectors by snapping to the nearest codebook entry and
    recovering byte-offset positions.
    """
    name = "Outputs"
    config_section = "OutputSpace"
    text_mode = False

    def _register_requirements(self):
        """OutputSpace always reshapes; nVectors and nActive are independently specified."""
        pass

    def _build_object_basis(self):
        initial_vectors = getattr(self, "_initial_vectors", None)
        if initial_vectors is not None:
            if isinstance(initial_vectors, Basis):
                return initial_vectors
            basis = Tensor()
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.nDim,
                passThrough=True,
                objectSize=self.objectSize,
            )
            basis.setW(initial_vectors)
            return basis
        basis = Tensor()
        basis.create(
            self.inputShape[0],
            self.outputShape[0],
            self.nDim,
            passThrough=True,
            objectSize=self.objectSize,
        )
        return basis

    def __init__(self, inputShape, spaceShape, outputShape, masked_prediction=False, vectors=None):
        section = self.config_section
        invertible = TheXMLConfig.space(section, "invertible")
        self.masked_prediction = masked_prediction
        object.__setattr__(self, "_initial_vectors", vectors)
        super().__init__(inputShape, spaceShape, outputShape)
        self.data = TheData
        self.text_mode = isinstance(self.subspace.get_vectors(), Embedding)
        input = self.subspace.getEncodedInputSize()
        output = self.subspace.getEncodedOutputSize()
        if self.reversible:
            if invertible:
                self.linear1 = InvertibleLinearLayer(input, output)
                self.forwardLinear, self.reverseLinear = self.linear1.forward, self.linear1.reverse
                self.layers = nn.ModuleList([self.forwardLinear])
            else:
                self.linear1 = LinearLayer(input, output)
                self.linear2 = LinearLayer(output, input)
                self.forwardLinear, self.reverseLinear = self.linear1.forward, self.linear2.forward
                self.layers = nn.ModuleList([self.linear1, self.linear2])
        else:
            self.forwardLinear = LinearLayer(input, output)
            self.layers = nn.ModuleList([self.forwardLinear])
        self.params = list(self.parameters())
    def getTestOutput(self):
        if not self.data.test_output:
            return None
        out = self.data.test_output
        if isinstance(out, list):
            out = torch.stack(out)
        return out.squeeze(-1) if out.ndim == 3 else out
    def prepOutput(self, outputBatch):
        if isinstance(outputBatch, list):
            return torch.stack(outputBatch, dim=0).unsqueeze(1).to(TheDevice)
        return outputBatch  # already [B, D, 1] and on device after toDevice()
    def forward(self, vspace):
        """Acting: project flattened symbols to task output via LinearLayer."""
        x = self.forwardBegin(vspace)
        output = self.forwardLinear(x)
        if self.quantized:
            output = self.subspace.get_vectors().forward(output)
        vspace = self.forwardEnd(output)
        return vspace

    def reverse(self, vspace):
        """Being acted upon: map output back to symbolic space via reverse LinearLayer."""
        y = self.reverseBegin(vspace)
        y = self.reverseLinear(y)
        vspace = self.reverseEnd(y)
        return vspace

    def expand_masked(self, embedded, sentence_text, maskedPrediction='MLM'):
        """Extract target embedding vectors from the embedded sentence.

        Each target is the original embedded word vector at the masked position,
        paralleling InputSpace.expand_masked() which creates masked copies.

        Modes:
            MLM/ARLM: Target[i] = embedded word i.
            ARUS: Zero vectors (loss suppressed in runEpoch).
            RARLM: Targets in reverse order (matching reverse mask positions).

        Args:
            embedded: [1, nVectors, embeddingSize] from InputSpace.forward()
            sentence_text: original sentence string (for word count)
            maskedPrediction: prediction mode ('MLM', 'ARLM', 'ARUS', or 'RARLM')

        Returns:
            targets: [N, 1, embeddingSize] tensor of target word embeddings
        """
        words = sentence_text.split()
        N = min(len(words), embedded.shape[1])
        embSize = embedded.shape[-1]
        if N == 0:
            return torch.zeros(0, 1, embSize, device=TheDevice)
        # Extract the first N full word vectors (nWhat + nWhere + nWhen)
        targets = embedded[0, :N, :].clone()  # [N, embSize]
        if maskedPrediction == 'ARUS':
            targets = torch.zeros_like(targets)
        elif maskedPrediction == 'RARLM':
            targets = targets.flip(0)
        return targets.unsqueeze(1)  # [N, 1, embSize]
    # --- Text reconstruction from symbolic vectors ---
    def set_text_mode(self, input_space):
        """Share InputSpace's Basis so OutputSpace can reconstruct text.

        Convenience method for tests. Production code passes vectors= to __init__.
        """
        vs = input_space.subspace.get_vectors()
        self.subspace.object = vs
        self.text_mode = isinstance(vs, Embedding)

    def _reverse_text_vectors(self, vectors):
        emb = self.subspace.object
        nWhat = emb.content_dim
        object_encoding = self.subspace.objectEncoding
        if object_encoding is not None:
            content, aux = object_encoding.split_aux(vectors, nWhat)
        else:
            if vectors.shape[-1] <= nWhat:
                content, aux = vectors.clone(), None
            else:
                content, aux = vectors[:, :, :nWhat].clone(), vectors[:, :, nWhat:].clone()
        content = emb.reverse(content)
        if object_encoding is not None:
            restored = object_encoding.restore_aux(content, aux)
        elif aux is not None:
            restored = torch.cat([content, aux], dim=-1)
        else:
            restored = content
        return restored, emb.decode_reverse_meta(restored, subspace=self.subspace)

    def reconstruct_tokens(self, vectors):
        """Return positioned tokens decoded from symbolic vectors.

        Delegates to the Basis / Embedding reverse() path.
        """
        if not self.text_mode:
            raise RuntimeError("reconstruct_tokens() requires text_mode.")
        _, meta = self._reverse_text_vectors(vectors)
        return meta['tokens']

    def reconstruct_data(self, vectors):
        """Reconstruct words and positions from symbolic vectors.

        Delegates to the Basis / Embedding reverse() which handles
        Encoding stripping, codebook snapping, and reassembly.

        Args:
            vectors: Tensor of shape [batch, nVec, embeddingSize] containing
                     symbolic vectors with nWhat content and nWhere positioning.

        Returns:
            (recovered_words, recovered_offsets) where:
              - recovered_words is a list of lists of strings, one per batch element
              - recovered_offsets is a list of lists of floats (byte offsets),
                one per batch element.  None entries mean nWhere was absent (zero).
        """
        if not self.text_mode:
            raise RuntimeError("reconstruct_data() called but text_mode is not enabled. "
                               "Call set_text_mode(input_space) first.")
        recovered_tokens = self.reconstruct_tokens(vectors)
        return (
            [[word for word, _ in batch] for batch in recovered_tokens],
            [[offset for _, offset in batch] for batch in recovered_tokens],
        )

    def reconstruct_buffer(self, vectors, buf_size=256):
        """Reconstruct a destination buffer string from symbolic vectors.

        Uses nWhere byte offsets (when present) to place words at specific
        positions; falls back to consecutive placement when nWhere is absent.

        Args:
            vectors: Tensor of shape [batch, nVec, embeddingSize].
            buf_size: Size of the destination buffer in bytes.

        Returns:
            List of strings, one per batch element.
        """
        _, meta = self._reverse_text_vectors(vectors)
        return self.subspace.get_vectors().reconstruct_to_buffer(
            meta, buf_size=buf_size)

    def clearBatchResults(self):
        """Clear accumulated batch results. Called at start of each runEpoch."""
        self._batch_results = []

    def putBatch(self, result):
        """Collect output from a completed batch (symmetric with getBatch).

        Results are cleared at the start of each runEpoch() via clearBatchResults().

        Args:
            result: BatchResult namedtuple from runBatch().
        """
        if not hasattr(self, '_batch_results'):
            self._batch_results = []
        self._batch_results.append(result)
