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
from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend
from embed import WordVectors, PretrainModel
from data import Data, TheData
from Model import Layer, PiLayer, SigmaLayer # Import custom layers from Model.py
from Model import VQLayer, LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, LiftingLayer, LoweringLayer, ChunkLayer
from Model import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from collections import namedtuple as _namedtuple

class Encoding(nn.Module):
    """Abstract base class for per-slot encodings in the embedding vector.

    Each encoding occupies ``nDim`` contiguous slots at positions given by
    ``self.index`` (negative offsets from the end of the embedding).

    Subclasses implement:
        ``encode(value)``  → tensor [..., nDim]
        ``decode(encoded)`` → decoded value (tensor or scalar)
        ``forward(x)``     → stamped tensor (how values are assigned per-batch)
    """
    TARGETS = ("activation", "what", "where", "when", "event", "all")
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
            return y, torch.zeros(y.shape[0], y.shape[1], device=TheDevice.get())
        embeddingSize = y.shape[-1]
        index = self.resolve(embeddingSize)
        if index[0] < 0 or index[0] >= embeddingSize:
            return y, torch.zeros(y.shape[0], y.shape[1], device=TheDevice.get())
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
            offsets = torch.tensor(float(offsets), device=TheDevice.get())
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

    Index is computed dynamically from nWhere and nWhen so the slots
    always align with the muxed layout ``[what, where, when]``:
      index = [-(nWhere+nWhen), ..., -(nWhen+1)]

    A monotonic counter ``self.p`` assigns each object a unique position
    within a dataset pass; it must be reset between epochs to avoid overflow.
    """
    p = 0

    def __init__(self, maxP=0, nWhere=2, nWhen=0):
        if nWhere > 0:
            # Dynamic: where sits at [-(nWhere+nWhen), ..., -(nWhen+1)]
            index = [-(nWhere + nWhen) + i for i in range(nWhere)]
            super().__init__(index, maxP)
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
        index = np.add([embeddingSize] * len(self.index), self.index)
        position = torch.arange(self.p, self.p+batch*n, dtype=torch.float32, device=TheDevice.get())
        pos = self.encode(position)  # [batch*n, 2]
        y = x.clone()
        y[:, :, index] = pos.reshape(batch, n, self.nDim)
        self.p += batch
        assert self.p < self.maxVal, "Overflow in object embedding"
        return y

    @staticmethod
    def test():
        pe = WhereEncoding(100, nWhere=2, nWhen=0)
        pe.p = 0
        x = torch.zeros([2, 4, 12], device=TheDevice.get())
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
        time_vals = torch.arange(self.t, self.t + batch, dtype=torch.float32, device=TheDevice.get())
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
        x = torch.zeros([2, 4, 10], device=TheDevice.get())
        y = te.forward(x)
        cleaned, times = te.reverse(y)
        print(f"Times decoded: {times}")
class WordEncoding(Encoding):
    """Word encoding: each word is a (batch, vector, rule, order) 4-tuple.

    Words are stored as a Python list of tuples, not muxed into the
    event tensor.  Each tuple indexes into [B, N] activation space
    and specifies a grammar rule from TheGrammar plus the epistemic
    level (conceptual order) at which the rule was applied.
    """
    nDim = 4  # (batch, vector, rule, order)

    def __init__(self, nBatch=0, nActive=0):
        super().__init__([], maxVal=0)  # no index slots — not muxed
        self.nBatch = nBatch
        self.nActive = nActive

    # Word tuple layout:
    #   [0] batch, [1] vector (position), [2] order (depth),
    #   [3] rule,  [4] leaf1, [5] leaf2, [6] leaf3
    # leaf slots hold codebook indices (-1 = unused).
    BATCH = 0
    VECTOR = 1
    ORDER = 2
    RULE = 3
    LEAF1 = 4
    LEAF2 = 5
    LEAF3 = 6

    def encode(self, batch, vector, rule, order=0,
               leaf1=-1, leaf2=-1, leaf3=-1):
        """Validate and return a 7-tuple word."""
        assert 0 <= batch, f"batch {batch} must be >= 0"
        assert 0 <= vector, f"vector {vector} must be >= 0"
        assert 0 <= rule < len(TheGrammar), f"rule {rule} out of range [0, {len(TheGrammar)})"
        return (batch, vector, order, rule, leaf1, leaf2, leaf3)

    def decode(self, word):
        """Unpack a word tuple into (batch, vector, rule)."""
        return word[self.BATCH], word[self.VECTOR], word[self.RULE]
class WhatEncoding(Encoding):
    """Handle the content-layout transform for a space's What factor.

    Unlike WhereEncoding and WhenEncoding, this encoding is not a fixed
    quadrature code.  It stores input/output shapes and provides
    shape-validation stubs (forwardBegin/End, reverseBegin/End).

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

class EventEncoding(Encoding):
    """Handle the content-layout transform for a space's Event factor.

    Unlike WhereEncoding and WhenEncoding, this encoding is not a fixed
    quadrature code.  It stores the input/output shapes for auxiliary
    operations (split_aux, restore_aux) but boundary reshaping is now
    handled by Space.forwardBegin/End via nInputDim/nOutputDim.

    The encoding is parameter-free and identity by default.
    """

    nDim = 0  # EventEncoding does not occupy fixed index slots

    def __init__(self, inputShape=None, outputShape=None):
        super().__init__([], 0)
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, objects, **kwargs):
        """Identity content pass-through."""
        return objects

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
        self.passThrough = False
        self.monotonic = True
        self.ergodic = False
        self.sigma_kappa = 0.01

    def getW(self):
        """Return the current weight tensor. Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement getW()")

    def setW(self, value):
        """Set the weight tensor. Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement setW()")

    def create(self, nInput, nVectors, nDim, customVQ=True, monotonic=True,
               passThrough=False):
        self.nInput = nInput
        self.nVectors = nVectors
        self.nDim = nDim or 0
        self.monotonic = monotonic
        self.passThrough = passThrough
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
        if not self.monotonic:
            normalized = F.normalize(target, p=2, dim=-1)
        else:
            normalized = torch.clamp(target, 0, 1)
            normalized = F.normalize(normalized, p=2, dim=-1)
        if x is None:
            self.setW(normalized)
            return self.getW()
        return normalized

    # -- Logic operations ---------------------------------------------------
    # Domain and range: [-1, +1].
    #
    # monotonic=False (bitonic): sign = direction, magnitude = confidence.
    #   +1 = true, 0 = unknown, -1 = false.
    #   Conjunction/disjunction respect sign agreement.
    #
    # monotonic=True (unsigned): linear scale from -1 to +1.
    #   Plain min/max. not() extracts negative magnitudes as positive (structural negation).

    def conjunction(self, x, y, monotonic=False):
        """Conjunction (intersection). Domain/range [-1, 1]."""
        if monotonic:
            return torch.min(x, y)
        same_sign = (x * y > 0).float()
        min_mag = torch.min(torch.abs(x), torch.abs(y))
        return same_sign * torch.sign(x) * min_mag

    def disjunction(self, x, y, monotonic=False):
        """Disjunction (union). Domain/range [-1, 1]."""
        if monotonic:
            return torch.max(x, y)
        same_sign = (x * y > 0).float()
        max_mag = torch.max(torch.abs(x), torch.abs(y))
        core = same_sign * torch.sign(x) * max_mag
        x_zero = (x == 0).float()
        y_zero = (y == 0).float()
        return core + x_zero * y + y_zero * x

    def negation(self, x, monotonic=False):
        """Negation. Bitonic: sign flip. Monotonic: extract negative magnitudes as positive.
        Domain [-1, 1]. Range: bitonic [-1, 1], monotonic [0, 1]."""
        if monotonic:
            return torch.relu(-x)
        return -x

    def non(self, x, monotonic=False, threshold=None):
        """Non-affirming negation. Bitonic: → 0. Monotonic: learnable threshold.
        Domain [-1, 1]. Range [0, 1] (monotonic) or {0} (bitonic)."""
        if monotonic and threshold is not None:
            return torch.relu(x - threshold)
        return torch.zeros_like(x)

    # -- Inverse logic operations -----------------------------------------------

    def negation_inverse(self, x, monotonic=False):
        """Inverse of negation. Bitonic: exact (self-inverse). Monotonic: lossy stub.
        Domain [-1, 1]."""
        if monotonic:
            return x  # relu(-x) is lossy; best-effort identity
        return -x

    def conjunction_inverse(self, result, y, monotonic=False):
        """Inverse of conjunction via codebook search.

        Find the codebook vector x such that conjunction(x, cb_j) ≈ result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        return self._binary_op_inverse(result, self.conjunction, monotonic)

    def disjunction_inverse(self, result, y, monotonic=False):
        """Inverse of disjunction via codebook search.

        Find the codebook vector x such that disjunction(x, cb_j) ≈ result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        return self._binary_op_inverse(result, self.disjunction, monotonic)

    def _binary_op_inverse(self, result, op, monotonic):
        """Search codebook for pair (cb[i], cb[j]) whose op(cb[i], cb[j]) ≈ result.

        Returns cb[i] (the left operand) for each position in result.
        result shape: (..., D).  Codebook shape: (K, D).
        """
        try:
            cb = self.getW()  # (K, D)
        except (NotImplementedError, AttributeError):
            warnings.warn("_binary_op_inverse: no codebook available", stacklevel=3)
            return result

        if cb is None or cb.shape[0] == 0:
            return result

        K, D = cb.shape
        flat = result.reshape(-1, D)  # (N, D)
        N = flat.shape[0]

        # Precompute op(cb[i], cb[j]) for all pairs → (K, K, D)
        cb_i = cb.unsqueeze(1).expand(K, K, D)  # (K, K, D)
        cb_j = cb.unsqueeze(0).expand(K, K, D)  # (K, K, D)
        composed = op(cb_i, cb_j, monotonic=monotonic)  # (K, K, D)
        composed_flat = composed.reshape(K * K, D)  # (K*K, D)

        # Find closest composed pair for each position
        # Use chunked computation to limit memory: process N positions in chunks
        chunk_size = max(1, min(N, 2048 // K))
        best_i = torch.empty(N, dtype=torch.long, device=result.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # (end-start, 1, D) - (1, K*K, D) → (end-start, K*K)
            diffs = (flat[start:end].unsqueeze(1) - composed_flat.unsqueeze(0))
            dists = diffs.pow(2).sum(dim=-1)  # (chunk, K*K)
            pair_idx = dists.argmin(dim=-1)   # (chunk,)
            best_i[start:end] = pair_idx // K  # left operand index

        return cb[best_i].reshape(result.shape)

    def pos(self, x):
        """Positive projection (ReLU). Domain [-1, 1], range [0, 1]."""
        return torch.relu(x)

    def distance(self, x, y, monotonic=False, dim=-1):
        """Distance in [0, 1]. Bitonic: angular. Monotonic: volume-weighted L2.

        Monotonic distance weights each element by max(|x|, |y|) so that
        matching zeros contribute nothing — zero-volume elements have no
        bearing on parthood.
        """
        if monotonic:
            w = torch.max(x.abs(), y.abs())
            total_weight = w.sum(dim=dim).clamp(min=epsilon)
            return (w * (x - y) ** 2).sum(dim=dim) / total_weight
        return (1 - F.cosine_similarity(x, y, dim=dim)) / 2

    def codebookDistance(self, x):
        weight = self._prototype_weight(context="codebookDistance")
        vec = weight[:, :self.nDim].to(TheDevice.get())
        return x @ vec.T / max(self.nDim, 1)

    # -- Mereological operations --------------------------------------------
    # part(x, y): degree to which x is part of y (x ⊆ y).
    # Defined as: x == intersection(x,y) AND y == union(x,y).
    # All return values in [0, 1].

    def part(self, x, y, monotonic=False):
        """Parthood in [0, 1]. 1 = x is part of y."""
        m = monotonic
        intersection = self.conjunction(x, y, monotonic=m)
        union = self.disjunction(x, y, monotonic=m)
        cond1 = 1 - self.distance(x, intersection, monotonic=True)
        cond2 = 1 - self.distance(y, union, monotonic=True)
        return self.conjunction(cond1, cond2, monotonic=m)

    def whole(self, x, y, monotonic=False):
        """Wholeness: degree to which x contains y."""
        return self.part(y, x, monotonic=monotonic)

    def equal(self, x, y, monotonic=False):
        """Equality: mutual parthood. 1 = mereologically identical."""
        m = monotonic
        return self.conjunction(self.part(x, y, monotonic=m),
                                self.part(y, x, monotonic=m), monotonic=m)

    def overlap(self, x, y, monotonic=False):
        """Overlap in [0, 1]: degree to which x and y share parts."""
        max_norm = torch.max(self.norm(x), self.norm(y)).clamp(min=epsilon)
        return self.norm(self.conjunction(x, y, monotonic=monotonic)) / max_norm

    def boundary(self, x, y, monotonic=False):
        """Boundary: asymmetry of containment between x and y."""
        m = monotonic
        return torch.abs(self.part(x, y, monotonic=m) - self.whole(x, y, monotonic=m))

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
    def symbolize(X, eps=1e-8):
        norms = torch.linalg.norm(X, dim=-1)
        return (2.0 * norms.mean(dim=1) - 1.0).clamp(-1.0, 1.0)
class Tensor(Basis):
    """Dense tensor payload implementation used for ordinary SubSpace slots."""

    def __init__(self, nVectors=0, nDim=0, W=None):
        super().__init__()
        self.W = None
        self.nVectors = nVectors
        self.nDim = nDim
        if W is not None:
            self.W = W
        else:
            self.W = torch.zeros(nVectors, nDim)

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
        self.W = None
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

    def create(self, nInput, nVectors, nDim, customVQ=True, monotonic=True,
               passThrough=False):
        super().create(
            nInput,
            nVectors,
            nDim,
            customVQ=customVQ,
            monotonic=monotonic,
            passThrough=passThrough,
        )
        self.customVQ = customVQ
        self.alpha = 0.0
        if (not self.passThrough) and self.nVectors > 0 and self.getW() is None:
            self.addVectors(self.nVectors)
        return self

    def updateWeights(self, embed_sum, cluster_size):
        return torch.ones(self.vq.codebook_size, device=TheDevice.get())

    def addVectors(self, nVec=1, decay=0.9):
        """Allocate ``nVec`` prototype entries using the configured backend."""
        self.codebookSize = nVec
        if self.customVQ:
            self.vq = VectorQuantize(
                dim=self.nDim,
                codebook_size=nVec,
                threshold_ema_dead_code=1,
                decay=decay,
                commitment_weight=1.0,
                rotation_trick=True,
            )
            self.setW(self.vq.codebook)
        else:
            W = torch.randn([nVec, self.nDim], device=TheDevice.get())
            for i in range(nVec):
                W[i, :] = self.normalize(W[i, :]).squeeze(0)
            self.setW(W)
        return self.getW()

    def quantize(self, x):
        if self.passThrough:
            return x, None, torch.tensor(0.0, device=x.device, dtype=x.dtype)
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
        _vspace = None
        if isinstance(input, SubSpace):
            _vspace = input
            input = _vspace.materialize()

        if self.passThrough:
            self.setW(input)
            if _vspace is not None:
                _vspace.set_event(input)
                return _vspace
            return input

        w = self.getW()
        if w is None or self.codebookSize == 0:
            self.addVectors(max(self.nVectors, input.shape[1]))
        x = input
        batch = x.shape[0]
        act = torch.zeros([batch, self.codebookSize], device=TheDevice.get())
        if self.customVQ:
            flat = torch.reshape(x, [-1, self.nDim])
            quantized, indices, _ = self.quantize(flat)
            batch = x.shape[0]
            n_tokens = flat.shape[0] // batch
            err = torch.norm(flat - quantized, dim=-1).reshape(batch, n_tokens)
            indices = indices.reshape(batch, n_tokens)
            # Use 3D views for token-level indexing (x may be 2D when reshape=True)
            x3d = x.reshape(batch, n_tokens, self.nDim)
            input3d = input.reshape(batch, n_tokens, self.nDim)
            quantized3d = quantized.reshape(batch, n_tokens, self.nDim)
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
                        cosSim = self.distance(
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
        if _vspace is not None:
            _vspace.set_event(x, compute_activation=False)
            _vspace.set_activation(act)
            return _vspace
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
        """Return live embedding vectors, L2-normalized so elements are in [-1, 1].

        Returns the nn.Parameter directly (not .data) so gradients flow through.
        """
        if self.wv is not None and self.wv._vectors is not None:
            return F.normalize(self.wv._vectors, p=2, dim=-1)
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
        if getattr(self, 'byte_mode', False):
            return self._byte_stream(text)
        from parse import quick_parser
        return quick_parser(self._to_text(text))

    def _byte_stream(self, text):
        """Tokenize as raw bytes: one (chr(byte), byte_offset) per byte."""
        raw = self._to_text(text).encode('utf-8')
        return [(chr(b), i) for i, b in enumerate(raw)]

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
               min_frequency=0.0, neg_samples=64, byte_mode=False):
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
            placeholder = torch.randn(1, dim, device=TheDevice.get())
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
                       vector_size, passThrough=passThrough)
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

        # Bootstrap codebook with ASCII (and full 0-255 in byte mode).
        self.byte_mode = byte_mode
        upper = 256 if byte_mode else 127
        for cp in range(1, upper):
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
                doc_tensor = torch.frombuffer(bytearray(doc_bytes), dtype=torch.uint8).clone()
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
        new_W = self._coerce_rows(new_W).to(TheDevice.get())
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
            new_vec = vector.to(TheDevice.get())
            if new_vec.dim() == 1:
                new_vec = new_vec.unsqueeze(0)
        else:
            new_vec = torch.randn(1, dim, device=TheDevice.get())
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
        var = self._ergodic_var(token_idx, TheDevice.get(), vec.dtype)
        if var is not None and torch.any(var > 0):
            vec = vec + var * torch.randn_like(vec)
        return F.normalize(vec, p=2, dim=0)

    def _nearest_idx(self, vec, codebook=None):
        if codebook is None:
            codebook = self.getW().detach()
        vec = vec.to(TheDevice.get())
        sims = F.cosine_similarity(vec.unsqueeze(0), codebook, dim=1)
        return sims.argmax().item()

    def forward(self, input, return_meta=False):
        """Tokenize via Lex, look up embedding vectors from codebook.

        Input: (batch, max_len) byte tensor from Data.
        Output: (batch, nInput, nDim) embedding tensor (pure content, no positional padding).

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

        # Phase 2: Batch-insert OOV words into codebook (skip in byte mode)
        if oov_words and not getattr(self, 'byte_mode', False):
            for word in oov_words:
                self.insert(word)
            codebook = self.getW()  # refresh after insert
            if self.optimize_embedding:
                model = getattr(self, '_model', None)
                if model is not None:
                    model.rebuild_optimizer()

        # Phase 3: Build result tensor (all tokens now in codebook)
        result = torch.zeros([batch, self.nInput, self.nDim], device=TheDevice.get())
        batch_indices = torch.zeros([batch, self.nInput], dtype=torch.long,
                                     device=TheDevice.get())
        null_idx = self.wv.key_to_index.get("\x00", 0)
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
                batch_indices[b, i] = cb_idx
                result[b, i, :] = self._encode_vector(
                    codebook[cb_idx], token_idx=cb_idx)
            # Fill remaining positions with NULL embedding (input buffer is null-terminated)
            if null_idx is not None:
                null_vec = self._encode_vector(codebook[null_idx], token_idx=null_idx)
                for i in range(n_tokens, self.nInput):
                    batch_indices[b, i] = null_idx
                    result[b, i, :] = null_vec
        if not return_meta:
            return result
        return result, {
            'tokens': batch_tokens,
            'indices': batch_indices,
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

    def save_embeddings(self, path, truth_data=None):
        """Save current embedding vectors and vocabulary to a .pt file."""
        self.wv.save(path, truth_data=truth_data)

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
        if math.isnan(sin_val) or math.isnan(cos_val):
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
        # Ensure 3D: [B, N, D]
        if vectors.ndim == 2:
            vectors = vectors.unsqueeze(0)
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
        return torch.zeros(self.getW().shape[1], device=TheDevice.get())

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
                 nInputDim=0, nOutputDim=0,
                 objectEncoding=None, activeEncoding=None, whatEncoding=None,
                 whereEncoding=None, whenEncoding=None, wordEncoding=None,
                 object=None, what=None, where=None, when=None, activation=None,
                 word=None):
        super().__init__()
        self.inputShape = inputShape    # [nActive, nDim]
        self.outputShape = outputShape  # [nActive, nDim]

        self.activeEncoding = activeEncoding if activeEncoding is not None else ActiveEncoding()
        self.objectEncoding = objectEncoding if objectEncoding is not None else EventEncoding(
            inputShape, outputShape)
        self.whatEncoding   = whatEncoding if whatEncoding is not None else WhatEncoding(inputShape, outputShape)
        self.whereEncoding  = whereEncoding if whereEncoding is not None else WhereEncoding(0, 0)
        self.whenEncoding   = whenEncoding if whenEncoding is not None else WhenEncoding(0, 0)
        self.wordEncoding   = wordEncoding if wordEncoding is not None else WordEncoding()

        # Resolved nInputDim/nOutputDim (0 → constructor dim, -1 → skip, >0 → explicit)
        self._nInputDim = inputShape[1] if nInputDim == 0 else nInputDim
        self._nOutputDim = outputShape[1] if nOutputDim == 0 else nOutputDim
        self.nWhere = self.whereEncoding.nDim
        self.nWhen = self.whenEncoding.nDim
        # nWhat: content width from outputShape (full dim minus where/when)
        self.nWhat = outputShape[1] - self.nWhere - self.nWhen
        self.muxedSize = self.nWhat + self.nWhere + self.nWhen

        self.activation = self._coerce_basis(activation, role="activation")
        self.event = self._coerce_basis(object, role="event")
        self.what   = self._coerce_basis(what, role="what")
        self.where  = self._coerce_basis(where, role="where")
        self.when   = self._coerce_basis(when, role="when")
        self.word   = word if word is not None else []  # list of (batch, vector, rule) tuples
        self._demuxed = False
        # active: [B, N, M] per-modality indices into the Basis slots.
        # M = number of modalities (what, where, when).
        # active[b, n, m] = index into modality m's Basis for position n.
        # activation: [B, N] strength gate — materialize() = event * activation.
        self._active = None  # [B, N, M] index tensor
        self.batch = 0
        payload = self.materialize()
        if isinstance(payload, torch.Tensor) and payload.ndim > 0:
            self.batch = payload.shape[0]

    @property
    def basis(self):
        """Return the primary content Basis (what) for this SubSpace."""
        return self.what

    @property
    def is_demuxed(self):
        """True when what/where/when are stored independently (not muxed into event)."""
        return self._demuxed

    def set_muxed(self, event_tensor):
        """Store muxed event tensor directly. Clears demuxed modalities.

        Args:
            event_tensor: [B, N, D] where D = nWhat + nWhere + nWhen
        """
        self.event.setW(event_tensor)
        self._demuxed = False

    def set_demuxed(self, what_tensor, where_tensor=None, when_tensor=None):
        """Store modality tensors independently. Computes active flags.

        Also computes activation from what-content norms.
        Clears event cache; call materialize() to rebuild it.

        Args:
            what_tensor: [B, N, nWhat] content vectors
            where_tensor: [B, N, nWhere] positional encoding (or None)
            when_tensor: [B, N, nWhen] temporal encoding (or None)
        """
        self.what.setW(what_tensor)
        if where_tensor is not None:
            self.where.setW(where_tensor)
        if when_tensor is not None:
            self.when.setW(when_tensor)
        self.event.setW(None)  # clear muxed cache
        self._demuxed = True
        self._compute_active(what_tensor, where_tensor, when_tensor)

    def _compute_active(self, what_tensor, where_tensor=None, when_tensor=None):
        """Compute modal presence flags from modality tensors.

        active[b, n, m] = 1 if modality m is nonzero at position n, else 0.
        Also computes event activation from what-content norms.

        Args:
            what_tensor: [B, N, nWhat]
            where_tensor: [B, N, nWhere] or None
            when_tensor: [B, N, nWhen] or None
        """
        B, N = what_tensor.shape[0], what_tensor.shape[1]
        flags = []
        # what is always present as a modality
        flags.append((what_tensor.norm(dim=-1) > 1e-8).float())  # [B, N]
        if where_tensor is not None and self.nWhere > 0:
            flags.append((where_tensor.norm(dim=-1) > 1e-8).float())
        if when_tensor is not None and self.nWhen > 0:
            flags.append((when_tensor.norm(dim=-1) > 1e-8).float())
        self._active = torch.stack(flags, dim=-1)  # [B, N, M]
        # Event activation: what-content norm scaled to [-1, 1]
        d = what_tensor.shape[-1]
        act = torch.norm(what_tensor, dim=-1) / math.sqrt(d)  # [B, N] in [0, 1]
        act = 2 * act - 1  # scale to [-1, 1]
        self.set_activation(act)

    def _coerce_basis(self, value, role):
        if isinstance(value, Basis):
            return value
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"SubSpace {role} must be a Basis, Tensor, or None")
        basis = Tensor()
        if role == "activation":
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.activeEncoding.nDim,
                passThrough=True,
            )
        elif role == "event":
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.muxedSize,
                passThrough=True,
            )
        elif role == "what":
            basis.create(
                self.outputShape[0],
                self.outputShape[0],
                self.nWhat,
                passThrough=True,
            )
        elif role == "where":
            basis.create(self.outputShape[0], self.outputShape[0], self.nWhere, passThrough=True)
        elif role == "when":
            basis.create(self.outputShape[0], self.outputShape[0], self.nWhen, passThrough=True)
        else:
            last_dim = value.shape[-1] if value.ndim > 1 else 1
            n_vectors = value.shape[-1] if value.ndim == 1 else value.shape[-2]
            basis.create(n_vectors, n_vectors, last_dim, passThrough=True)
        basis.setW(value)
        return basis

    def get_vectors(self):
        """Return the event Basis object.

        To get the dense tensor, use materialize() instead.
        """
        return self.event

    @property
    def vocabulary(self):
        """Return the content Basis (Embedding/Codebook) for codebook operations."""
        return self.what

    def set_event(self, event_tensor, compute_activation=False):
        """Store a muxed event tensor [B, N, D] where D = nWhat + nWhere + nWhen.

        Sets activation to all-ones by default — activation should be set
        explicitly (via normalize, attention, etc.) rather than auto-derived.

        Args:
            event_tensor: [B, N, D] muxed event vector.
            compute_activation: if True, recompute activation from event norms.
        """
        if self.event is None:
            self.event = Tensor()
        self.set_muxed(event_tensor)
        if compute_activation:
            self.set_activation_from_event()
        else:
            B, N = event_tensor.shape[0], event_tensor.shape[1]
            self.set_activation(torch.ones(B, N, device=event_tensor.device))

    def set_what(self, what_tensor):
        """Store what-content vectors [B, N, nWhat].

        Invalidates cached event so materialize() re-concatenates.
        """
        self.what.setW(what_tensor)
        self.event.setW(None)
        self._demuxed = True

    def set_where(self, where_tensor):
        """Store positional encoding vectors [B, N, nWhere].

        Invalidates cached event so materialize() re-concatenates.
        """
        self.where.setW(where_tensor)
        self.event.setW(None)
        self._demuxed = True

    def set_when(self, when_tensor):
        """Store temporal encoding vectors [B, N, nWhen].

        Invalidates cached event so materialize() re-concatenates.
        """
        self.when.setW(when_tensor)
        self.event.setW(None)
        self._demuxed = True

    def set_forward_content(self, what_indices, where_indices=None, when_indices=None,
                            activation=None):
        """Set per-modality indices and activation for the forward pass.

        Args:
            what_indices: [B, N] indices into .what Basis
            where_indices: [B, N] indices into .where Basis (or None)
            when_indices: [B, N] indices into .when Basis (or None)
            activation: [B, N] strength gate (or None to compute from indices)
        """
        B, N = what_indices.shape[0], what_indices.shape[1]
        parts = [what_indices.unsqueeze(-1)]  # [B, N, 1]
        if where_indices is not None:
            parts.append(where_indices.unsqueeze(-1))
        if when_indices is not None:
            parts.append(when_indices.unsqueeze(-1))
        self._active = torch.cat(parts, dim=-1)  # [B, N, M]
        self.event.setW(None)  # clear cached event
        self._demuxed = True
        if activation is not None:
            self.set_activation(activation)
        else:
            # Default: all indexed positions are fully active
            act = torch.ones(B, N, device=what_indices.device)
            self.set_activation(act)

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------

    def getEncodingSize(self, nDim):
        """Full muxed vector width: nWhat + nWhere + nWhen."""
        return self.muxedSize

    def getEncodedInputSize(self):
        """Return effective input dim after nInputDim reshape."""
        if self._nInputDim == -1:
            return self.objectEncoding.inputShape[1]
        return self._nInputDim

    def getEncodedOutputSize(self):
        """Return the layer output dim (before forwardEnd reshape).

        When forwardBegin doesn't change vector count (nInputDim == inputShape[1]),
        the layer simply maps per-vector from nInputDim → nOutputDim.

        When forwardBegin DOES change vector count (e.g. flatten), the layer
        output dim is computed so that forwardEnd can reshape to [B, oS[0], nOutputDim]:
          layer_out = oS[0] * nOutputDim * nInputDim / (iS[0] * iS[1])
        """
        if self._nInputDim == -1 or self._nOutputDim == -1:
            return self.objectEncoding.outputShape[1]
        iS = self.objectEncoding.inputShape
        if self._nInputDim == iS[1]:
            return self._nOutputDim
        oS = self.objectEncoding.outputShape
        return oS[0] * self._nOutputDim * self._nInputDim // (iS[0] * iS[1])

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

    # forwardBegin/End, reverseBegin/End — handled by Space via nInputDim/nOutputDim reshape.

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_tensor(cls, tensor, *,
                    inputShape, outputShape, **kw):
        """Wrap an existing dense tensor into a SubSpace."""
        return cls(
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

    def set_activation_from_event(self):
        """Compute activation and active flags from muxed event vectors.

        Activation is the L2 norm of the event divided by sqrt(D), scaled
        to [-1, 1].  Active flags are derived by checking each modality
        slice for nonzero content.
        """
        y = self.event.getW()
        assert y is not None and y.ndim == 3, "Must be dim==3"
        d = y.shape[-1]
        activation = torch.norm(y, dim=-1) / math.sqrt(d)  # [B, N] in [0, 1]
        activation = 2 * activation - 1                     # scale to [-1, 1]
        self.set_activation(activation)
        # Derive active flags from modality slices
        flags = []
        what_slice = y[:, :, :self.nWhat]
        flags.append((what_slice.norm(dim=-1) > 1e-8).float())
        if self.nWhere > 0:
            where_slice = y[:, :, self.nWhat:self.nWhat + self.nWhere]
            flags.append((where_slice.norm(dim=-1) > 1e-8).float())
        if self.nWhen > 0:
            when_slice = y[:, :, self.nWhat + self.nWhere:]
            flags.append((when_slice.norm(dim=-1) > 1e-8).float())
        self._active = torch.stack(flags, dim=-1)  # [B, N, M]

    #def set_activations(self, whatA, whereA, whenA): 
    #    # to store a cross-product activation
    #    return
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
        self.activation.setW(activation_tensor)

    def get_activation(self):
        """Return stored activation [batch, nVectors] or None."""
        if self.activation is None:
            return None
        return self.activation.getW()

    def get_active(self):
        """Return modal presence flags [B, N, M] or None."""
        return self._active

    def set_active(self, active_tensor):
        """Store modal presence flags.

        Args:
            active_tensor: [B, N, M] where M = number of modalities.
                Binary (or soft) flags indicating which modalities are
                populated at each position.
        """
        assert active_tensor.ndim == 3, \
            f"active must be [B, N, M], got {active_tensor.shape}"
        self._active = active_tensor

    def effective_activation(self):
        """Return effective activation: activation * product of modal flags.

        Returns:
            [B, N] tensor. A position is active only if all its modalities
            are present AND its event activation is positive.  Returns
            plain activation if no modal flags are set.
        """
        act = self.get_activation()
        if act is None:
            return None
        if self._active is not None:
            modal_gate = self._active.prod(dim=-1)  # [B, N]
            return act * modal_gate
        return act

    def dematerialize(self):
        """Split event → modalities, recover active flags.

        Reads .event, splits into what/where/when by encoding widths,
        computes active flags from which slices are nonzero, and
        sets activation from event norms.

        Returns:
            dict with 'what', 'where', 'when' tensors (where present).
        """
        event = self.event.getW()
        if event is None:
            return None
        what_tensor = event[:, :, :self.nWhat]
        self.what.setW(what_tensor)

        result = {'what': what_tensor}
        where_tensor = None
        when_tensor = None

        if self.nWhere > 0:
            where_tensor = event[:, :, self.nWhat:self.nWhat + self.nWhere]
            self.where.setW(where_tensor)
            result['where'] = where_tensor
        if self.nWhen > 0:
            when_tensor = event[:, :, self.nWhat + self.nWhere:]
            self.when.setW(when_tensor)
            result['when'] = when_tensor

        self._demuxed = True
        self._compute_active(what_tensor, where_tensor, when_tensor)
        return result

    def set_symbols(self, symbols_tensor):
        """Store symbolic presence [0,1] by mapping to conceptual activation [-1,1].

        Symbols are percepts: each dimension is the presence (1) or absence (0)
        of a symbol.  Internally stored as activation via x = 2*y - 1.

        Args:
            symbols_tensor: [batch, nSymbols] values in [0, 1].
        """
        self.set_activation(symbols_tensor)

    def get_symbols(self):
        """Return symbolic presence [0,1] mapped from activation [-1,1].

        Returns [batch, nSymbols] in [0,1] via y = (x+1)/2, or None.
        """
        act = self.get_activation()
        if act is None:
            return None
        return act

    # ------------------------------------------------------------------
    # Word management
    # ------------------------------------------------------------------

    def set_words(self, words):
        """Store word list. Each entry is a (batch, vector, rule) tuple."""
        self.word = list(words)

    def get_words(self):
        """Return the word list."""
        return self.word

    def add_word(self, batch, vector, rule, order=0,
                 leaf1=-1, leaf2=-1, leaf3=-1):
        """Append a validated word tuple."""
        self.word.append(self.wordEncoding.encode(
            batch, vector, rule, order, leaf1, leaf2, leaf3))

    # ── Stack-scanning helpers ────────────────────────────────────────

    def active_positions(self, b, data=None):
        """Return sorted list of active (non-zero) position indices for batch element b."""
        if data is None:
            data = self.materialize()
        if data.ndim == 3:
            act = data[b].norm(dim=-1)
        else:
            act = data[b]
        nz = act.nonzero(as_tuple=False)
        if nz.numel() == 0:
            return []
        return nz.squeeze(-1).tolist()

    def top_of_stack(self, data=None):
        """Find the last active (non-zero) position per batch element.

        Returns:
            list of int — top-of-stack position for each batch element,
            or -1 if the entire row is zero (empty stack).
        """
        if data is None:
            data = self.materialize()
        tops = []
        for b in range(data.shape[0]):
            active = self.active_positions(b, data)
            tops.append(active[-1] if active else -1)
        return tops

    def top_two_of_stack(self, data=None):
        """Find the last two active positions per batch element.

        Returns:
            list of (pos1, pos2) tuples — pos1 is second-to-top, pos2 is top.
            Either may be -1 if fewer than two active positions exist.
        """
        if data is None:
            data = self.materialize()
        result = []
        for b in range(data.shape[0]):
            active = self.active_positions(b, data)
            if len(active) >= 2:
                result.append((active[-2], active[-1]))
            else:
                result.append((-1, -1))
        return result

    def _lookup_modality(self, basis, indices):
        """Look up vectors from a Basis using index tensor [B, N].

        For Embedding/Codebook: index into codebook rows.
        For WhereEncoding: compute sin/cos from offset indices.
        """
        codebook = basis.getW()
        if codebook is None:
            return None
        if codebook.ndim == 2:
            # [V, D] codebook — index with [B, N] → [B, N, D]
            return codebook[indices.long()]
        # Already [B, N, D] (e.g. from set_what on a Tensor basis)
        return codebook

    def materialize(self, k=None, mode="active"):
        """Build event from active indices, return event * activation.

        Index-based path (set_forward_content): active [B, N, M] holds indices
        into each modality's Basis. Looks up vectors from .what, .where, .when.

        Legacy path (set_demuxed/set_what): reads vectors from Basis slots directly.

        Muxed path: returns event directly.

        Args:
            k: number of vectors to return. If None, returns all vectors.
            mode: "active" (default) returns event * activation;
                  "activation" returns the effective activation [batch, nVectors].

        Returns:
            Tensor [batch, k, dim] (mode="active"),
            or Tensor [batch, nVectors] (mode="activation").
        """
        if mode == "activation":
            eff = self.effective_activation()
            if eff is not None:
                return eff
            activation = self.get_activation()
            if activation is not None:
                return activation
            self.set_activation_from_event()
            return self.get_activation()

        x = self.event.getW()

        if x is None and self._demuxed:
            if self._active is not None and self._active.ndim == 3:
                # Index-based path: active [B, N, M] holds per-modality indices
                parts = []
                m = 0
                if self.what is not None:
                    what_w = self.what.getW()
                    if what_w is not None:
                        if what_w.ndim == 2:
                            # Codebook [V, D] — index lookup
                            parts.append(what_w[self._active[:, :, m].long()])
                        else:
                            parts.append(what_w)
                        m += 1
                if self.nWhere > 0 and m < self._active.shape[-1]:
                    where_indices = self._active[:, :, m]  # byte offsets
                    # Raw mux: store offsets directly via WhereEncoding.encode()
                    where_vecs = self.whereEncoding.encode(where_indices)
                    parts.append(where_vecs)
                    m += 1
                if self.nWhen > 0 and m < self._active.shape[-1]:
                    when_indices = self._active[:, :, m]
                    when_vecs = self.whenEncoding.encode(when_indices)
                    parts.append(when_vecs)
                if parts:
                    x = torch.cat(parts, dim=-1)
                    self.event.setW(x)
            else:
                # Legacy path: read vectors from Basis slots
                parts = []
                what_w = self.what.getW()
                if what_w is not None and what_w.shape[-1] > 0:
                    parts.append(what_w)
                where_w = self.where.getW()
                if where_w is not None and where_w.shape[-1] > 0:
                    parts.append(where_w)
                when_w = self.when.getW()
                if when_w is not None and when_w.shape[-1] > 0:
                    parts.append(when_w)
                if parts:
                    x = torch.cat(parts, dim=-1)
                    self.event.setW(x)

        if x is None:
            return None

        # Apply activation gate: materialize() = event * activation
        act = self.get_activation()
        if act is not None:
            x = x * act.unsqueeze(-1)

        # top-k selection if requested
        if k is not None and k < x.shape[-2]:
            score = act if act is not None else x.norm(dim=-1)
            _, indices = torch.topk(score, k, dim=-1)
            self._topk_indices = indices
            x = torch.gather(x, -2, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        return x

    # ------------------------------------------------------------------
    # Encoding target selection
    # ------------------------------------------------------------------

    def select(self, target="event"):
        """Return the tensor for the named encoding target.

        Args:
            target: one of Encoding.TARGETS — "activation", "what", "where",
                    "when", "event", or "all".

        Returns:
            Tensor for the requested encoding, or a dict of {target: tensor}
            when target="all" (skipping empty encodings).
        """
        assert target in Encoding.TARGETS, f"Unknown target {target!r}, expected one of {Encoding.TARGETS}"

        if target == "activation":
            return self.materialize(mode="activation")

        if target == "all":
            result = {}
            for t in ("activation", "what", "where", "when", "event"):
                v = self.select(t)
                if v is not None and v.numel() > 0:
                    result[t] = v
            return result

        if target == "event":
            return self.materialize()

        # Demuxed mode: read from independent factor tensors
        if self._demuxed:
            if target == "what":
                return self.what.getW()
            elif target == "where":
                return self.where.getW()
            elif target == "when":
                return self.when.getW()

        # Muxed mode: slice from event tensor
        event = self.materialize()
        if event is None:
            return None
        if target == "what":
            return event[:, :, :self.nWhat]
        elif target == "where":
            if self.nWhere == 0:
                return None
            return event[:, :, self.nWhat:self.nWhat + self.nWhere]
        elif target == "when":
            if self.nWhen == 0:
                return None
            return event[:, :, self.nWhat + self.nWhere:]
        return None

    def put(self, target, tensor):
        """Write a tensor back to the named encoding target.

        Args:
            target: one of "activation", "what", "where", "when", "event".
            tensor: the tensor to store.
        """
        assert target in Encoding.TARGETS and target != "all", \
            f"Cannot put to target {target!r}"

        if target == "activation":
            self.set_activation(tensor)
            return

        if target == "event":
            self.set_event(tensor)
            return

        # Demuxed mode: write to independent factor tensors
        if self._demuxed:
            if target == "what":
                self.set_what(tensor)
            elif target == "where":
                self.set_where(tensor)
            elif target == "when":
                self.set_when(tensor)
            return

        # Muxed mode: clone event then splice in the new slice
        event = self.materialize()
        if event is None:
            return
        event = event.clone()
        if target == "what":
            event[:, :, :self.nWhat] = tensor
        elif target == "where" and self.nWhere > 0:
            event[:, :, self.nWhat:self.nWhat + self.nWhere] = tensor
        elif target == "when" and self.nWhen > 0:
            event[:, :, self.nWhat + self.nWhere:] = tensor
        self.set_event(event, compute_activation=False)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, kind, target="activation", normalize=False, strict=False):
        """Normalize or check range of an encoding of this subspace.

        When normalize=True, transforms the tensor in-place to the
        correct range for the space kind.  When normalize=False,
        checks whether the tensor is already in range.

        When strict=True (nonlinear path), raises ValueError on
        out-of-range input — the sigmoid/logit pair guarantees the
        range contract so a violation is a real bug.  When strict=False,
        emits a warning — allows exploring the unconstrained path.

        Geometry-aware:
          - "percepts" on vectors → sigmoid + clamp [0,1] (hypercube)
          - "percepts" on activation → sigmoid [0,1]
          - "concepts" on vectors → L2 unit-norm (hypersphere)
          - "concepts" on activation → tanh [-1,1]
          - "symbols" → STE round {0,1} (activation only)
          - "input" on vectors → min-max scale to [0,1]

        Args:
            kind: "percepts", "concepts", "symbols", or "input".
            target: encoding target — "activation", "what", "where",
                    "when", "event", or "all".
            normalize: if True, apply the normalization in-place.
                If False, check range only.
            strict: if True, raise ValueError on violation.
                If False, emit a warning.
        """
        x = self.select(target)
        if x is None or x.numel() == 0:
            return
        if target == "event":
            assert not self._demuxed, (
                f"normalize(target='event') requires muxed state, "
                f"but subspace is demuxed. Use target='what'/'where'/'when' "
                f"individually, or call set_event() first."
            )
        if not normalize:
            is_vector = target in ("what", "where", "when", "event")
            xd = x.detach()
            if not is_vector:
                # Activation: scalar range [-1, 1]
                lo, hi = -1, 1
                xmin, xmax = xd.min().item(), xd.max().item()
                if xmin - lo < -1e-2 or xmax - hi > 1e-2:
                    msg = (f"Range violation: kind={kind!r}, target={target!r} "
                           f"range [{xmin:.6f}, {xmax:.6f}] outside [{lo}, {hi}].")
                    if strict:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg)
            elif kind == "concepts":
                # Concepts: elements in [-1, 1] (tanh)
                xmin, xmax = xd.min().item(), xd.max().item()
                if xmin < -1 - 1e-2 or xmax > 1 + 1e-2:
                    msg = (f"Range violation: kind={kind!r}, target={target!r} "
                           f"range [{xmin:.6f}, {xmax:.6f}] outside [-1, 1].")
                    if strict:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg)
            elif kind == "percepts":
                # Percepts: elements in [-1, 1] (tanh)
                xmin, xmax = xd.min().item(), xd.max().item()
                if xmin < -1 - 1e-2 or xmax > 1 + 1e-2:
                    msg = (f"Range violation: kind={kind!r}, target={target!r} "
                           f"range [{xmin:.6f}, {xmax:.6f}] outside [-1, 1].")
                    if strict:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg)
            else:
                # symbols [0,1], input [-1,1]
                lo, hi = {"symbols": (0, 1), "input": (-1, 1)}.get(kind, (None, None))
                if lo is not None:
                    xmin, xmax = xd.min().item(), xd.max().item()
                    if xmin - lo < -1e-2 or xmax - hi > 1e-2:
                        msg = (f"Range violation: kind={kind!r}, target={target!r} "
                               f"range [{xmin:.6f}, {xmax:.6f}] outside [{lo}, {hi}].")
                        if strict:
                            raise ValueError(msg)
                        else:
                            warnings.warn(msg)
            return
        normalized = self._apply_normalization(kind, x, target=target)
        self.put(target, normalized)

    def _apply_normalization(self, kind, x, target="activation"):
        """Apply normalization function to tensor x.

        The combination of kind and target determines the geometry:
          - Perceptual vectors and activations use tanh [-1,1].
          - Conceptual vectors and activations use tanh [-1,1].
          - Activations use scalar transfer functions (tanh/STE).
        """
        is_vector = target in ("what", "where", "when", "event")
        if kind == "percepts":
            return torch.tanh(x)
        elif kind == "concepts":
            return torch.tanh(x)
        elif kind == "symbols":
            soft = torch.sigmoid(x)
            hard = torch.round(soft)
            return hard - soft.detach() + soft  # straight-through estimator
        elif kind == "input":
            if is_vector:
                from data import TheData
                return TheData.normalize(x, which="input")  # [0,1] via global min-max
            return x
        else:
            raise ValueError(f"Unknown normalization kind: {kind!r}")

    def denormalize(self, kind, target="activation"):
        """Reverse the normalization applied by normalize().

        Only meaningful for kinds with invertible transforms:
          - "input" on vectors → scale from [0,1] back to [input_min, input_max]
          - "output" on vectors → scale from [output_min, output_max] to [-1,1]

        Invertible transforms:
          - "percepts" → logit (inverse sigmoid): [0,1] → ℝ
          - "concepts" → atanh (inverse tanh): [-1,1] → ℝ
          - "input" on vectors → scale from [0,1] back to [input_min, input_max]
          - "output" on vectors → scale from [output_min, output_max] to [-1,1]
          - "symbols" (STE round) is not invertible and is skipped.
        """
        x = self.select(target)
        if x is None or x.numel() == 0:
            return
        is_vector = target in ("what", "where", "when", "event")
        from data import TheData
        if kind == "percepts":
            # logit: inverse of sigmoid
            x = x.clamp(min=epsilon, max=1 - epsilon)
            self.put(target, torch.log(x / (1 - x)))
        elif kind == "concepts":
            if is_vector:
                raise RuntimeError("Cannot denormalize")
            else:
                # atanh: inverse of tanh
                x = x.clamp(min=-1 + epsilon, max=1 - epsilon)
                self.put(target, torch.atanh(x))
        elif kind == "input" and is_vector:
            self.put(target, TheData.denormalize(x, which="input"))
        elif kind == "output" and is_vector:
            self.put(target, TheData.normalize(x, which="output"))

    # ------------------------------------------------------------------
    # Luminosity
    # ------------------------------------------------------------------

    def luminosity(self, x=None, target="what", reduce="batch"):
        """Measure contrast (signal energy) as MSE against zero.

        For activations in [-1, 1], luminosity ranges [0, 1].
        A zero vector has luminosity 0 (nothing); a fully saturated
        vector has luminosity ~1 (everything).

        Args:
            x: tensor to measure. If None, uses self.select(target).
            target: encoding target to measure.
            reduce: "batch" → mean over objects → [B] (default),
                    "vector" → per-object → [B, N].

        Returns:
            Tensor of luminosity values.
        """
        if x is None:
            x = self.select(target)
        if x is None or x.numel() == 0:
            return torch.tensor(0.0, device=TheDevice.get())
        mse = (x ** 2).mean(dim=-1)
        if reduce == "batch" and mse.ndim > 1:
            return mse.mean(dim=-1)  # [B]
        return mse  # [B, N] or scalar

    def luminosity_match(self, x1, x2, target="what", reduce="batch"):
        """Measure truth: the magnitude of agreement between two signals.

        Agreement at each dimension is the smaller magnitude when signs
        match (both confirm at least that much).  When signs disagree,
        agreement is zero — the representations contradict.

        The MSE of the agreement vector gives the luminosity of truth
        between the two signals.

        Args:
            x1, x2: tensors to compare (same shape).
            target: encoding target (for documentation; tensors are
                    passed explicitly).
            reduce: "batch" → [B], "vector" → [B, N].

        Returns:
            Tensor of truth-luminosity values.
        """
        agreement = torch.where(
            x1 * x2 >= 0,
            torch.min(x1.abs(), x2.abs()),
            torch.zeros_like(x1),
        )
        mse = (agreement ** 2).mean(dim=-1)
        if reduce == "batch" and mse.ndim > 1:
            return mse.mean(dim=-1)
        return mse

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
class Grammar:
    """Hierarchical 3-tier grammar (S, C, P) rule catalog.

    Owns the rule definitions parsed from XML config.  All learnable
    parameters and rule execution live on the tier-specific SyntacticLayer
    subclasses (SymbolicSyntacticLayer, ConceptualSyntacticLayer,
    PerceptualSyntacticLayer).
    """

    RuleDef = _namedtuple('RuleDef', ['tier', 'canonical', 'arity', 'method_name'])

    def __init__(self):
        self.rules = []
        self._configured = False
        self.interpretation = 0.5

    # ── Rule catalog ──────────────────────────────────────────────────

    def __len__(self):
        self._ensure_configured()
        return len(self.rules)

    def __getitem__(self, idx):
        self._ensure_configured()
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
        self.rules = []
        self._configured = True
        for lhs in ('START', 'S', 'C', 'P'):
            raw = grammar_dict.get(lhs, [])
            if isinstance(raw, str):
                raw = [raw]
            for rhs_text in raw:
                rhs = rhs_text.strip()
                rule_def = self._parse_rule(lhs, rhs)
                self.rules.append(rule_def)

    def _parse_rule(self, lhs, rhs):
        if '(' in rhs:
            func_name = rhs[:rhs.index('(')]
            args_str = rhs[rhs.index('(') + 1:rhs.rindex(')')]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            arity = len(args)
            canonical = f"{lhs} → {rhs}"
            return self.RuleDef(lhs, canonical, arity, func_name)
        if rhs == 'ε':
            return self.RuleDef(lhs, f"{lhs} → ε", 0, None)
        if rhs == 'I P':
            return self.RuleDef(lhs, f"{lhs} → chunk(I, P)", 2, 'chunk')
        if rhs in ('S', 'C', 'P'):
            return self.RuleDef(lhs, f"{lhs} → {rhs}", 1, None)
        if rhs == 'I':
            return self.RuleDef(lhs, f"{lhs} → I", 0, None)
        raise ValueError(f"Cannot parse grammar rule: {lhs} → {rhs}")

    _NOOP_GRAMMAR = {'START': 'S', 'S': 'C', 'C': ['not(C)', 'P'], 'P': 'I'}

    def _ensure_configured(self):
        if self._configured:
            return
        from util import TheXMLConfig
        cfg = None
        for path in ("mentalModel.grammar", "architecture.language.grammar", "architecture.grammar"):
            try:
                candidate = TheXMLConfig.get(path)
                if isinstance(candidate, dict):
                    cfg = candidate
                    break
            except (KeyError, AttributeError):
                continue
        if cfg is None:
            cfg = self._NOOP_GRAMMAR
        self.configure(cfg)
        # interpretation: check language section first, then legacy mentalModel
        for ipath in ("architecture.language.interpretation", "mentalModel.interpretation"):
            try:
                interp = TheXMLConfig.get(ipath)
                self.interpretation = float(interp)
                break
            except (KeyError, AttributeError):
                continue

    # ── Rule queries ──────────────────────────────────────────────────

    def symbolic(self):
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'S']

    def conceptual(self):
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'C']

    def perceptual(self):
        self._ensure_configured()
        return [i for i, r in enumerate(self.rules) if r.tier == 'P']

    def symbolic_transition(self):
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.tier == 'S' and r.method_name is None and r.arity == 1:
                return i
        return None

    def conceptual_transition(self):
        self._ensure_configured()
        for i, r in enumerate(self.rules):
            if r.tier == 'C' and r.method_name is None and r.arity == 1:
                return i
        return None

    @property
    def s_methods(self):
        """Set of method names available on the S (symbolic) tier."""
        return {r.method_name for r in self.rules if r.tier == 'S' and r.method_name}

    @property
    def c_methods(self):
        """Set of method names available on the C (conceptual) tier."""
        return {r.method_name for r in self.rules if r.tier == 'C' and r.method_name}

    @property
    def p_methods(self):
        """Set of method names available on the P (perceptual) tier."""
        return {r.method_name for r in self.rules if r.tier == 'P' and r.method_name}

    def _c_rule_ids(self):
        """Return dict of method_name → rule_id for C-tier operational rules."""
        result = {}
        for i, r in enumerate(self.rules):
            if r.tier == 'C' and r.method_name is not None:
                result[r.method_name] = i
        return result

    # _conceptual_forward, _symbolic_forward, forward, reverse — moved to
    # specialized SyntacticLayer subclasses (ConceptualSyntacticLayer,
    # SymbolicSyntacticLayer, PerceptualSyntacticLayer).  Grammar retains
    # only rule catalog, project(), and *Forward/*Reverse operations.

    # composeSyntax, _compose_conceptual, _compose_symbolic — removed.
    # Soft superposition is now inlined in _conceptual_forward and _symbolic_forward.

    # ── C-tier operations live on SyntacticLayer / ConceptualSyntacticLayer
    # as *Forward / *Reverse method pairs.  See _RULE_METHODS dispatch.

TheGrammar = Grammar()
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

    # Transition bias scale: (1 - interpretation) * TRANSITION_SCALE is added
    # to the transition rule's logit. The transition rule (S→C or C→P) acts
    # as NOP — "stop deriving this tier, pass through."
    # Low interpretation → transition dominates → no reductions (episodic).
    # High interpretation → grammar rules fire → composition (semantic).
    TRANSITION_SCALE = 10.0

    def __init__(self, nInput, nOutput, rules, transition_rule=None,
                 max_depth=12, hidden_dim=256, grammar=None, tau=1.0):
        super().__init__(nInput, nOutput)
        # Store grammar as non-Module attribute to avoid circular nn.Module
        # reference (Grammar owns SyntacticLayers, SyntacticLayers reference
        # Grammar). Using object.__setattr__ bypasses nn.Module.__setattr__
        # which would register it as a submodule.
        if grammar is None:
            grammar = Grammar()
        object.__setattr__(self, 'grammar', grammar)
        self.rules           = list(rules)
        self.transition_rule = transition_rule
        # Build the full set of rule IDs this layer predicts over
        self.all_rules = list(rules)
        if transition_rule is not None and transition_rule not in self.all_rules:
            self.all_rules.append(transition_rule)
        self.num_rules  = len(self.all_rules)
        # Map from local index → global rule ID
        self.rule_index = {rid: i for i, rid in enumerate(self.all_rules)}
        # Local index of the transition rule (for interpretation bias)
        self.transition_index = (self.rule_index.get(transition_rule)
                                 if transition_rule is not None else None)
        self.max_depth  = max_depth
        self.hidden_dim = hidden_dim
        self.tau        = tau

        # Rule prediction network (weight-tied across depths)
        self.input_proj       = LinearLayer(nInput, hidden_dim)
        self.derivation_layer = LinearLayer(hidden_dim, hidden_dim)
        self.rule_head        = LinearLayer(hidden_dim, self.num_rules)
        self.depth_embed      = nn.Embedding(max_depth, hidden_dim)
        self.activation_fn    = nn.GELU()

        # Xavier initialization so logits start in a numerically stable range.
        # LinearLayer defaults to torch.randn which gives std=1.0; for large
        # dims this produces huge activations that saturate softmax/gumbel.
        for layer in [self.input_proj, self.derivation_layer, self.rule_head]:
            nn.init.xavier_normal_(layer.W)
        nn.init.normal_(self.depth_embed.weight, std=0.02)

        # Register child layers for ergodic dispatch
        self.layers = [self.input_proj, self.derivation_layer, self.rule_head]

    # ── Basis-delegated rule execution ────────────────────────────

    def _basis(self, subspace):
        """Return the Basis from a SubSpace (or None)."""
        return subspace.basis if subspace is not None else None

    def _mono(self, subspace):
        """True if this subspace uses monotonic logic."""
        b = self._basis(subspace)
        return b is None or b.monotonic

    # ── Forward/Reverse dispatch ────────────────────────────────────
    #
    # C-tier ops (invertible): not, intersection, union, lift, lower
    # S-tier ops (lossy, no inverse): equals, part, true, non, swap
    # P-tier ops (invertible): chunk
    #
    # _RULE_METHODS maps rule name → (forwardName, reverseName|None, binary)

    _RULE_METHODS = {
        'union':        ('unionForward',        'unionReverse',        True),
        'intersection': ('intersectionForward', 'intersectionReverse', True),
        'not':          ('notForward',          'notReverse',          False),
        'equals':       ('equalsForward',       None,                  True),
        'part':         ('partForward',         None,                  True),
        'chunk':        ('chunkForward',        'chunkReverse',        True),
        'true':         ('trueForward',         None,                  False),
        'non':          ('nonForward',          None,                  False),
    }

    def project(self, grammar, rule_id, left, right=None, third=None, subspace=None):
        """Execute a grammar rule forward. Subclasses override for parametric rules."""
        method_name = grammar.rules[rule_id].method_name
        if method_name is None:
            return left  # transition — pass through

        if method_name in self._RULE_METHODS:
            fn_name, _, binary = self._RULE_METHODS[method_name]
            fn = getattr(self, fn_name)
            if binary:
                if right is not None:
                    return fn(left, right, subspace)
                return left
            return fn(left, subspace)

        return left

    def reverse_project(self, grammar, rule_id, result, right=None, subspace=None):
        """Execute a grammar rule inverse. Returns best-effort recovery of left operand."""
        method_name = grammar.rules[rule_id].method_name
        if method_name is None:
            return result

        if method_name in self._RULE_METHODS:
            _, rev_name, binary = self._RULE_METHODS[method_name]
            if rev_name is None:
                return result  # lossy op — no inverse
            fn = getattr(self, rev_name)
            if binary:
                return fn(result, right, subspace)
            return fn(result, subspace)

        return result

    # ── C-tier: invertible operations ─────────────────────────────

    def notForward(self, left, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.negation(left, monotonic=self._mono(subspace))
        return -left

    def notReverse(self, result, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.negation_inverse(result, monotonic=self._mono(subspace))
        return -result

    def intersectionForward(self, left, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.conjunction(left, right, monotonic=self._mono(subspace))
        return torch.min(left, right)

    def intersectionReverse(self, result, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.conjunction_inverse(result, right, monotonic=self._mono(subspace))
        return result

    def unionForward(self, left, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.disjunction(left, right, monotonic=self._mono(subspace))
        return torch.max(left, right)

    def unionReverse(self, result, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.disjunction_inverse(result, right, monotonic=self._mono(subspace))
        return result

    # ── P-tier: chunk (invertible) ────────────────────────────────

    def chunkForward(self, left, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.disjunction(left, right, monotonic=True)
        if right is None:
            return left
        return torch.max(left, right)

    def chunkReverse(self, result, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.disjunction_inverse(result, right, monotonic=True)
        return result

    # ── S-tier: lossy operations (no inverse) ─────────────────────

    def equalsForward(self, left, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            score = b.equal(left, right, monotonic=self._mono(subspace))
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            return score * right
        return torch.min(left, right)

    def partForward(self, left, right, subspace):
        b = self._basis(subspace)
        if b is not None:
            score = b.part(left, right, monotonic=self._mono(subspace))
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            return score * right
        return torch.min(left, right)

    def trueForward(self, left, subspace):
        b = self._basis(subspace)
        if b is not None:
            return b.pos(left)
        return torch.relu(left)

    def nonForward(self, left, subspace):
        b = self._basis(subspace)
        if b is not None:
            m = self._mono(subspace)
            threshold = getattr(self, 'non_threshold', None)
            t = torch.sigmoid(threshold) if threshold is not None else None
            return b.non(left, monotonic=m, threshold=t)
        return torch.zeros_like(left)

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

        # Transition bias: (1 - interpretation) * scale on the transition
        # rule logit. The transition rule (S→C or C→P) is the NOP — "stop
        # deriving, pass through." Low interpretation biases toward it.
        interp = self.grammar.interpretation if self.grammar is not None else 0.5
        transition_bias = (1.0 - interp) * self.TRANSITION_SCALE

        for d in range(self.max_depth):
            h = h + depth_vecs[d]
            h = self.derivation_layer.forward(h)
            h = self.activation_fn(h)
            logits = self.rule_head.forward(h)  # [B, num_rules]

            # Bias the transition rule logit. Detach the bias so it
            # doesn't flow gradients — interpretation is a hyperparameter,
            # the grammar shouldn't learn to predict NOP.
            if self.transition_index is not None and transition_bias > 0:
                logits = logits.clone()
                logits[:, self.transition_index] = (
                    logits[:, self.transition_index].detach() + transition_bias
                )

            if self.training:
                probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            else:
                probs = F.softmax(logits, dim=-1)

            all_logits.append(logits)
            all_probs.append(probs)

        rule_logits = torch.stack(all_logits, dim=1)
        rule_probs  = torch.stack(all_probs, dim=1)

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
class PerceptualSyntacticLayer(SyntacticLayer):
    """P-tier SyntacticLayer: BPE-style chunk merging.

    Owns the chunk codebook and iterative merge loop.  Repeatedly merges
    the top two active positions while the codebook score exceeds the
    entropic threshold.  Stops at word boundaries (whitespace).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_layer = None  # created lazily on first P-tier data

    def _chunk_rule_id(self):
        """Return the rule_id for the P-tier chunk rule, or None."""
        for r in self.all_rules:
            if self.grammar.rules[r].method_name == 'chunk':
                return r
        return None

    def _ensure_chunk_layer(self, nDim):
        """Lazily create the chunk codebook when we first see P-tier data."""
        if self.chunk_layer is None:
            self.chunk_layer = ChunkLayer(nDim).to(
                next(self.parameters()).device if list(self.parameters()) else 'cpu')

    def compose(self, data, subspace, grammar):
        """Apply P-tier chunk merging.

        In byte mode (subspace._byte_indices set): hard-merge at whitespace
        boundaries first, then run learned BPE, then compact to dense word slots.

        Args:
            data: [B, N, D] percept tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for subspace access
        Returns:
            data with chunk merges applied (and compacted in byte mode)
        """
        subspace.word = []
        chunk_rid = self._chunk_rule_id()
        if chunk_rid is None or data.ndim != 3:
            return data

        self._ensure_chunk_layer(data.shape[-1])
        cb = self.chunk_layer

        # Byte mode: hard-merge at whitespace boundaries
        byte_indices = getattr(subspace, '_byte_indices', None)
        if byte_indices is not None:
            data, span_meta = cb.hard_merge_spans(data, byte_indices)
            subspace._byte_span_meta = span_meta

        # Learned BPE loop — boundary check delegates to ChunkLayer
        while True:
            any_merged = False
            pairs = subspace.top_two_of_stack(data)
            for b, (pos1, pos2) in enumerate(pairs):
                if pos1 < 0 or pos2 < 0:
                    continue
                if cb.is_word_boundary(data, b, pos2,
                                       subspace=subspace, byte_indices=byte_indices):
                    continue
                v1, v2 = data[b, pos1], data[b, pos2]
                should, chunk_id = cb.should_merge(v1, v2)
                if not should:
                    continue
                merged, _ = cb.encode(v1, v2)
                data = data.clone()
                data[b, pos1] = merged
                data[b, pos2] = 0.0
                subspace.word.append((b, pos1, pos2, chunk_rid))
                any_merged = True
            if not any_merged:
                break

        # Byte mode: compact sparse → dense word slots
        if byte_indices is not None:
            nWordSlots = getattr(subspace, '_nWordSlots', data.shape[1])
            where_enc = getattr(subspace, 'whereEncoding', None)
            data, compact_map = cb.compact(data, nWordSlots, span_meta, where_enc)
            subspace._compact_map = compact_map

        return data

    def decompose(self, data, subspace, grammar):
        """Reverse P-tier chunk merges using recorded 4-tuple words.

        In byte mode: un-compacts first, then undoes BPE merges.

        Args:
            data: [B, N, D] tensor (compacted in byte mode)
            subspace: SubSpace with recorded words
            grammar: Grammar instance (unused, kept for interface consistency)
        Returns:
            data with chunk merges undone
        """
        # Byte mode: un-compact dense word slots back to sparse byte positions
        compact_map = getattr(subspace, '_compact_map', None)
        if compact_map is not None and self.chunk_layer is not None:
            byte_indices = getattr(subspace, '_byte_indices', None)
            nByteSlots = byte_indices.shape[1] if byte_indices is not None else data.shape[1]
            data = self.chunk_layer.uncompact(data, compact_map, nByteSlots)

        # Undo BPE merges
        words = subspace.get_words()
        for word in reversed(words):
            if len(word) != 4:
                continue
            b, pos1, pos2, rule_id = word
            if self.chunk_layer is None:
                continue
            merged = data[b, pos1]
            best_sim = -1.0
            best_k = 0
            for k in range(self.chunk_layer.nChunks):
                sim = F.cosine_similarity(
                    merged.unsqueeze(0),
                    self.chunk_layer.merge[k].unsqueeze(0), dim=-1)
                if sim.item() > best_sim:
                    best_sim = sim.item()
                    best_k = k
            v1, v2 = self.chunk_layer.decode(best_k)
            data = data.clone()
            data[b, pos1] = v1
            data[b, pos2] = v2
        return data
class ConceptualSyntacticLayer(SyntacticLayer):
    """C-tier SyntacticLayer: deterministic not + soft-weighted composition.

    Rule application order:
      1. not(C) — flips negative concepts to positive (mean < 0).
      2. Soft superposition — remaining rules weighted by predicted probs.

    Owns lift/lower layers.
    """

    _RULE_METHODS = {
        **SyntacticLayer._RULE_METHODS,
        'lift':  ('liftForward',  'liftReverse',  True),
        'lower': ('lowerForward', 'lowerReverse', True),
    }

    def init_conceptual_params(self, concept_dim):
        """Initialize C-tier learnable parameters. Called by Space.init_syntactic_layer."""
        self.lifting_layer = LiftingLayer(16, concept_dim)
        self.lowering_layer = LoweringLayer(concept_dim)
        self._symbolic_space = None  # set by BasicModel after init

    # ── C-tier projected ops: lift/lower via PiLayer ────────────────

    def _cs_layer(self):
        """PiLayer for concept→symbol projection (ss.layer)."""
        if self._symbolic_space is not None:
            return getattr(self._symbolic_space, 'layer', None)
        return None

    def liftForward(self, left, right, subspace):
        """Projected conjunction in symbolic space: s_a * s_b."""
        cs = self._cs_layer()
        if cs is not None:
            s_a = cs.forward(left)
            s_b = cs.forward(right)
            return s_a * s_b
        return left * right

    def liftReverse(self, result, right, subspace):
        """Recover first operand: s_a = result / s_b, then PiLayer.reverse."""
        cs = self._cs_layer()
        if cs is not None:
            s_b = cs.forward(right)
            s_a = result / (s_b + epsilon)
            return cs.reverse(s_a)
        return result / (right + epsilon)

    def lowerForward(self, left, right, subspace):
        """Projected disjunction in symbolic space: s_a + s_b."""
        cs = self._cs_layer()
        if cs is not None:
            s_a = cs.forward(left)
            s_b = cs.forward(right)
            return s_a + s_b
        return left + right

    def lowerReverse(self, result, right, subspace):
        """Recover first operand: s_a = result - s_b, then PiLayer.reverse."""
        cs = self._cs_layer()
        if cs is not None:
            s_b = cs.forward(right)
            s_a = result - s_b
            return cs.reverse(s_a)
        return result - right

    def project(self, grammar, rule_id, left, right=None, third=None, subspace=None):
        """Execute a rule. Lift/lower are in _RULE_METHODS via super()."""
        return super().project(grammar, rule_id, left, right, third, subspace=subspace)

    def reverse_project(self, grammar, rule_id, result, right=None, subspace=None):
        """Inverse dispatch — delegates to super()."""
        return super().reverse_project(grammar, rule_id, result, right, subspace=subspace)

    def compose(self, data, subspace, grammar, target_count=None):
        """Apply C-tier composition.

        Args:
            data: [B, N, D] concept tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for rule execution
            target_count: If set, use pairwise reduction to this token count
                          (hierarchical mode). None uses cascading accumulator.
        Returns:
            (composed_data, svo_or_None) — svo is set if ternary lift fired
        """
        subspace.word = []
        self.last_svo = None   # reset per-compose
        self.last_rule_probs = None  # per-depth composable rule probs
        self.last_composable_rules = None  # global rule IDs for columns
        c_rules = grammar._c_rule_ids()
        not_rid = c_rules.get('not')

        # Snapshot codebook indices before any modifications (for decompose)
        basis = getattr(subspace, 'basis', None)
        cb = basis.getW() if basis is not None else None
        if cb is not None and data.shape[-1] == cb.shape[-1]:
            B0, N0, D0 = data.shape
            self._leaf_cb_indices = (
                data.detach().reshape(-1, D0) @ cb.T
            ).argmax(dim=-1).reshape(B0, N0)
        else:
            self._leaf_cb_indices = None

        # Phase 1: deterministic not at top-of-stack
        tops = subspace.top_of_stack(data)
        for b, pos in enumerate(tops):
            if pos < 0:
                continue
            vec = data[b, pos]

            # ── not: negate via Basis.negation (bitonic: -x, self-inverse)
            if not_rid is not None:
                if vec.mean() < 0:
                    data = data.clone()
                    data[b, pos] = self.notForward(vec.unsqueeze(0).unsqueeze(0),
                                                    subspace).squeeze(0).squeeze(0)
                    subspace.add_word(b, pos, not_rid)

        # Dispatch: hierarchical pairwise reduction or cascading accumulator
        if target_count is not None:
            return self._compose_to_target(data, subspace, grammar, target_count,
                                           c_rules, not_rid)

        # Phase 2: soft-weighted composition via SyntacticLayer
        B, N, D = data.shape

        # Guard: skip soft superposition if data dims don't match SyntacticLayer
        expected_n = self.input_proj.nInput
        if N != expected_n:
            return data, self.last_svo

        # Derive [B, N] activation for SyntacticLayer
        activation = torch.norm(data, dim=-1) / math.sqrt(D)

        # Get rule probabilities from SyntacticLayer
        out = super().forward(activation)
        rule_probs = out['rule_probs']  # [B, max_depth, num_rules]

        # Identify composable rules (exclude not — already applied in Phase 1)
        exclude = {'not'}
        composable_local = []
        composable_global = []
        for local_idx, global_id in enumerate(self.all_rules):
            if grammar.rules[global_id].method_name not in exclude:
                composable_local.append(local_idx)
                composable_global.append(global_id)

        if not composable_global:
            return data, self.last_svo

        # Need at least one binary+ rule for cascading to combine anything;
        # unary rules just return left, consuming leaves without merging.
        has_binary = any(grammar.arity(gid) >= 2 for gid in composable_global)
        if not has_binary:
            return data, self.last_svo

        # Build per-batch active positions
        active_positions = [subspace.active_positions(b, data) for b in range(B)]
        max_leaves = max((len(p) for p in active_positions), default=0)
        if max_leaves == 0:
            return data, self.last_svo

        # Record terminal words for each leaf (transition rule + codebook index)
        cb_indices = self._leaf_cb_indices
        t_rid = self.transition_rule if self.transition_rule is not None else composable_global[0]
        if cb_indices is not None:
            for b in range(B):
                for i, pos in enumerate(active_positions[b]):
                    if i < max_leaves:
                        subspace.add_word(b, pos, t_rid, order=-1,
                                          leaf1=cb_indices[b, pos].item())

        # Extract leaf vectors via masks
        masks = torch.zeros(B, max_leaves, N, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                if i < max_leaves:
                    masks[b, i, pos] = 1.0
        leaf_vecs = masks.unsqueeze(-1) * data.unsqueeze(1)  # [B, L, N, D]

        composed = leaf_vecs[:, 0, :, :]  # start with first leaf
        self.last_composable_rules = composable_global
        depth_probs = []  # collect per-depth renormalized probs

        d = 0
        leaf_idx = 1  # next leaf to consume
        while d < self.max_depth and leaf_idx < max_leaves:
            left = composed
            right = leaf_vecs[:, leaf_idx, :, :]

            # Check if a ternary rule can fire (needs one more leaf)
            has_third = leaf_idx + 1 < max_leaves

            results = []
            for global_id in composable_global:
                a = grammar.arity(global_id)
                if a == 3 and has_third:
                    third = leaf_vecs[:, leaf_idx + 1, :, :]
                    result = self.project(grammar, global_id, left, right, third, subspace=subspace)
                elif a == 2:
                    result = self.project(grammar, global_id, left, right, subspace=subspace)
                else:
                    result = self.project(grammar, global_id, left, subspace=subspace)
                results.append(result)

            results = torch.stack(results, dim=1)  # [B, n_composable, N, D]

            # Extract and renormalize probabilities for composable subset
            probs_d = rule_probs[:, d, :][:, composable_local]  # [B, n_composable]
            probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-8)
            depth_probs.append(probs_d.detach())                # [B, n_composable]

            # Hard selection in eval mode (exact for decompose); soft mixture in training
            best = probs_d.argmax(dim=-1)  # [B]
            if self.training:
                probs_d = probs_d.unsqueeze(-1).unsqueeze(-1)   # [B, n_composable, 1, 1]
                composed = (probs_d * results).sum(dim=1)       # [B, N, D]
            else:
                # Select argmax rule output per batch element
                idx = best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
                idx = idx.expand(-1, 1, results.shape[2], results.shape[3])
                composed = results.gather(1, idx).squeeze(1)    # [B, N, D]

            # Record argmax rule as word
            best_global = composable_global[best[0].item()]
            for b in range(B):
                if d < len(active_positions[b]):
                    subspace.add_word(b, active_positions[b][min(d, len(active_positions[b]) - 1)],
                                      composable_global[best[b].item()])

            # Advance: ternary rules consume 2 leaves, others consume 1
            best_arity = grammar.arity(best_global)
            leaf_idx += (2 if best_arity == 3 and has_third else 1)
            d += 1

        if depth_probs:
            self.last_rule_probs = torch.stack(depth_probs, dim=1)  # [B, depths, n_composable]
        return composed, self.last_svo

    def _compose_to_target(self, data, subspace, grammar, target_count,
                           c_rules, not_rid):
        """Reduce active tokens to target_count via independent pairwise grammar reductions.

        Used by the hierarchical forward loop. Each round pairs adjacent active
        positions, applies soft-weighted grammar rules, and zeros consumed tokens
        until the active count reaches target_count.
        """
        B, N, D = data.shape

        # Identify composable rules (exclude not — already applied in Phase 1)
        exclude = {'not'}
        composable_local = []
        composable_global = []
        for local_idx, global_id in enumerate(self.all_rules):
            if grammar.rules[global_id].method_name not in exclude:
                composable_local.append(local_idx)
                composable_global.append(global_id)

        if not composable_global:
            return data, self.last_svo

        # Get rule probabilities from SyntacticLayer
        activation = torch.norm(data, dim=-1) / math.sqrt(D)
        expected_n = self.input_proj.nInput
        if N == expected_n:
            out = super().forward(activation)
            rule_probs = out['rule_probs']  # [B, max_depth, num_rules]
        else:
            # Dims don't match SyntacticLayer — use uniform probs
            rule_probs = torch.ones(B, self.max_depth, len(self.all_rules),
                                    device=data.device) / len(self.all_rules)

        # Build per-batch active positions
        active = [subspace.active_positions(b, data) for b in range(B)]

        # Record terminal words for each active leaf (codebook indices from pre-Phase1)
        cb_indices = self._leaf_cb_indices
        t_rid = self.transition_rule if self.transition_rule is not None else composable_global[0]
        if cb_indices is not None:
            for b in range(B):
                for pos in active[b]:
                    subspace.add_word(b, pos, t_rid, order=-1,
                                      leaf1=cb_indices[b, pos].item())

        d = 0

        while d < self.max_depth:
            max_active = max(len(a) for a in active)
            if max_active <= target_count:
                break

            new_data = data.clone()
            for b in range(B):
                positions = active[b]
                new_positions = []
                i = 0
                while i < len(positions) - 1 and (len(positions) - i + len(new_positions)) > target_count:
                    left_pos, right_pos = positions[i], positions[i + 1]
                    left = data[b:b+1, left_pos:left_pos+1, :]
                    right = data[b:b+1, right_pos:right_pos+1, :]

                    results = []
                    for gid in composable_global:
                        a = grammar.arity(gid)
                        if a >= 2:
                            r = self.project(grammar, gid, left, right, subspace=subspace)
                        else:
                            r = self.project(grammar, gid, left, subspace=subspace)
                        results.append(r)
                    results = torch.stack(results, dim=1)  # [1, n_composable, 1, D]

                    probs_d = rule_probs[b:b+1, min(d, rule_probs.shape[1]-1), :]
                    probs_d = probs_d[:, composable_local]
                    probs_d = probs_d / (probs_d.sum(dim=-1, keepdim=True) + 1e-8)

                    best_local = probs_d.argmax(dim=-1)[0].item()
                    if self.training:
                        composed = (probs_d.unsqueeze(-1).unsqueeze(-1) * results).sum(dim=1)
                    else:
                        composed = results[:, best_local]

                    new_data[b, left_pos] = composed[0, 0]
                    new_data[b, right_pos] = 0.0  # zero out consumed

                    best_rid = composable_global[best_local]
                    subspace.add_word(b, left_pos, best_rid, order=d)
                    new_positions.append(left_pos)
                    i += 2

                while i < len(positions):
                    new_positions.append(positions[i])
                    i += 1
                active[b] = new_positions

            data = new_data
            d += 1

        # Gather remaining active positions into dense [B, target_count, D]
        max_remaining = max(len(a) for a in active)
        result = torch.zeros(B, max_remaining, D, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active[b]):
                if i < max_remaining:
                    result[b, i] = data[b, pos]

        return result, self.last_svo

    def decompose(self, data, subspace, grammar):
        """Reconstruct pre-compose tensor from symbolic word record.

        Terminal words (order == -1) carry codebook indices of the original
        leaf vectors.  Reconstruction looks up each leaf from the codebook
        and places it at its recorded position, producing the exact
        pre-compose tensor without any cached tensors.

        Args:
            data: tensor (same shape as compose output, used for shape/device)
            subspace: SubSpace with recorded words
            grammar: Grammar instance (unused, kept for API compat)
        Returns:
            [B, N, D] tensor with leaf vectors at their original positions
        """
        words = subspace.get_words()
        basis = getattr(subspace, 'basis', None)
        cb = basis.getW() if basis is not None else None
        if cb is None:
            return data  # no codebook — fall back to identity

        result = torch.zeros_like(data)
        for word in words:
            if word[WordEncoding.ORDER] != -1:
                continue  # skip rule words — only terminals carry leaves
            b = word[WordEncoding.BATCH]
            pos = word[WordEncoding.VECTOR]
            cb_idx = word[WordEncoding.LEAF1]
            if cb_idx >= 0:
                result[b, pos] = cb[cb_idx]
        return result
class SymbolicSyntacticLayer(SyntacticLayer):
    """S-tier SyntacticLayer: soft-weighted composition on 2D activations.

    All S-tier rules (true, non, swap, equals, part, transition) are applied
    fractionally using learned rule probabilities.

    Owns swap parameters (Sinkhorn-normalised soft permutation).
    """

    def init_swap(self, symbol_dim, n_symbol_slots):
        """Initialize swap and non parameters. Called by Space.init_syntactic_layer."""
        swap_size = max(symbol_dim, n_symbol_slots, 1)
        self.swap_marker = nn.Parameter(torch.randn(swap_size) * 0.01)
        self.swap_logits = nn.Parameter(torch.zeros(3, 3))
        self._swap_sinkhorn_iters = 5
        self.non_threshold = nn.Parameter(torch.tensor(0.0))

    def _swap_soft_perm(self):
        M = self.swap_logits
        for _ in range(self._swap_sinkhorn_iters):
            M = M - M.logsumexp(dim=-1, keepdim=True)
            M = M - M.logsumexp(dim=-2, keepdim=True)
        return M.exp()

    def swapForward(self, left, right, subspace=None):
        """Soft permutation via Sinkhorn-normalised logits."""
        P = self._swap_soft_perm()
        marker = self.swap_marker.to(left.device)
        if left.ndim == 3:
            m = marker.unsqueeze(0).unsqueeze(0).expand_as(left)
        elif left.ndim == 2:
            m = marker[:left.shape[-1]].unsqueeze(0).expand_as(left)
        else:
            m = marker
        if right is None:
            right = left
        stack = torch.stack([left, right, m], dim=0)
        out = torch.einsum('ij,j...->i...', P, stack)
        return out[0]

    _RULE_METHODS = {
        **SyntacticLayer._RULE_METHODS,
        'swap': ('swapForward', None, True),
    }

    def project(self, grammar, rule_id, left, right=None, subspace=None):
        """Execute a rule via _RULE_METHODS dispatch."""
        return super().project(grammar, rule_id, left, right, subspace=subspace)

    def compose(self, data, subspace, grammar):
        """Apply S-tier soft-weighted composition.

        Args:
            data: [B, N] or [B, N, D] symbol activation tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for rule execution
        Returns:
            composed symbol activations, same shape as input
        """
        subspace.word = []
        if data.ndim == 3:
            # 3D vector mode: extract norms for grammar, scale vectors by result
            norms = data.norm(dim=-1)                    # [B, N]
            composed_norms = self.compose(norms, subspace, grammar)  # [B, N]
            scale = composed_norms / (norms + 1e-8)      # [B, N]
            return data * scale.unsqueeze(-1)             # [B, N, D]

        B, N = data.shape

        # Guard: skip soft superposition if data dims don't match SyntacticLayer
        expected_n = self.input_proj.nInput
        if N != expected_n:
            return data

        # Get rule probabilities from SyntacticLayer
        out = super().forward(data)
        rule_probs = out['rule_probs']  # [B, max_depth, num_rules]
        all_rules = self.all_rules

        # Build per-batch active positions
        active_positions = [subspace.active_positions(b, data) for b in range(B)]
        max_leaves = max((len(p) for p in active_positions), default=0)
        if max_leaves == 0:
            return data

        # Extract leaf activations via masks
        masks = torch.zeros(B, max_leaves, N, device=data.device)
        for b in range(B):
            for i, pos in enumerate(active_positions[b]):
                if i < max_leaves:
                    masks[b, i, pos] = 1.0
        leaf_acts = masks * data.unsqueeze(1)  # [B, L, N]

        composed = leaf_acts[:, 0, :]  # start with first leaf

        for d in range(min(self.max_depth, max(max_leaves - 1, 1))):
            if d + 1 >= max_leaves:
                break
            left = composed
            right = leaf_acts[:, d + 1, :]

            results = []
            for rule_id in all_rules:
                a = grammar.arity(rule_id)
                if a == 2:
                    result = self.project(grammar, rule_id, left, right, subspace=subspace)
                else:
                    result = self.project(grammar, rule_id, left, subspace=subspace)
                results.append(result)

            results = torch.stack(results, dim=1)  # [B, num_rules, N]
            probs_d = rule_probs[:, d, :]           # [B, num_rules]
            composed = (probs_d.unsqueeze(-1) * results).sum(dim=1)  # [B, N]

            # Record argmax rule as word
            best = probs_d.argmax(dim=-1)  # [B]
            for b in range(B):
                if d < len(active_positions[b]):
                    subspace.add_word(b, active_positions[b][d], all_rules[best[b].item()])

        return composed

    def decompose(self, data, subspace, grammar):
        """Reverse S-tier operations using recorded word tuples.

        Args:
            data: [B, N] or [B, N, D] tensor (same shape as compose output)
            subspace: SubSpace with recorded words
            grammar: Grammar instance for rule info
        Returns:
            data with grammar operations undone (best-effort)
        """
        words = subspace.get_words()
        for word in reversed(words):
            if len(word) < 3:
                continue
            b = word[WordEncoding.BATCH]
            pos = word[WordEncoding.VECTOR]
            rule_id = word[WordEncoding.RULE]
            rule = grammar.rules[rule_id]
            if rule.method_name in ('non', 'union', 'intersection',
                                    'lift', 'lower'):
                pass  # Non-invertible
            elif rule.method_name is not None:
                pass  # Not cleanly invertible
        return data

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
      - ``nInputDim``/``nOutputDim``: configurable boundary reshape.  0 (default)
        resolves to the constructor's dim; -1 flattens (nInput * dim); a positive
        value reshapes via ``x.reshape(B, -1, nInputDim)``.
      - ``processSymbols``: when True, reduces full embedding vectors to scalar
        activations (norms) for the symbolic representation.
      - ``codebook``: when True, input vectors are quantized against the
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
        # Resolve nInputDim/nOutputDim:
        #   0  → inherit from constructor dim (inputShape[1] / outputShape[1])
        #  -1  → flatten: nInput * dim (reshape [N, D] → [1, N*D])
        #  >0  → explicit value
        try:
            raw = TheXMLConfig.space(section, "nInputDim")
        except KeyError:
            raw = 0
        if raw == -1:
            self.nInputDim = inputShape[0] * inputShape[1]
        else:
            self.nInputDim = inputShape[1] if raw == 0 else raw
        try:
            raw = TheXMLConfig.space(section, "nOutputDim")
        except KeyError:
            raw = 0
        if raw == -1:
            self.nOutputDim = outputShape[1]
        else:
            self.nOutputDim = outputShape[1] if raw == 0 else raw

        self.reversible   = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self.codebook     = TheXMLConfig.space(section, "codebook")
        _nWhere = TheXMLConfig.space(section, "nWhere")
        _nWhen = TheXMLConfig.space(section, "nWhen")
        self.nWhere = _nWhere
        self.nWhen = _nWhen
        self.nWhat = self.nDim
        self.muxedSize = self.nWhat + self.nWhere + self.nWhen
        self.customVQ  = customVQ
        # inputShape/outputShape already include muxed width in dim (set by factory).
        objectEncoding = EventEncoding(inputShape, outputShape)
        whatEncoding   = WhatEncoding(inputShape, outputShape)
        whereEncoding  = WhereEncoding(TheXMLConfig.get("architecture.nObjects"), _nWhere, _nWhen)
        whenEncoding   = WhenEncoding(10000, _nWhen)
        self.subspace  = SubSpace(
            nInputDim=self.nInputDim,
            nOutputDim=self.nOutputDim,
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
        self.muxedSize = self.subspace.getEncodingSize(self.nDim)

        self.syntacticLayer = None  # populated by init_syntactic_layer() if grammar is used
        self.params = []   # parameters for the optimizer (excludes temperature params)
        self.layers = nn.ModuleList()   # layer instances for paramUpdate() delegation
        self._register_requirements()

    def _build_object_basis(self):
        basis = Codebook()
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.muxedSize,  # Codebook processes full event vectors
            customVQ=self.customVQ,
            passThrough=not self.codebook,
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
        if not self.codebook:
            TheXMLConfig.require(
                lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv == _na,
                f"{section_name}: non-codebook space requires nVectors ({nV}) == nActive ({nA})"
            )

    def get_vectors(self):
        """Convenience accessor — delegates to subspace."""
        return self.subspace.get_vectors()

    @property
    def vocabulary(self):
        """Return the content Basis (Embedding/Codebook) for codebook operations."""
        return self.subspace.vocabulary

    def forwardBegin(self, vspace, returnVectors=False):
        """Prepare input for space-specific processing.

        Args:
            vspace: input SubSpace.
            returnVectors: if True, materialize to a dense tensor and
                reshape to [B, -1, nInputDim]. If False, pass the SubSpace
                through without materializing (for spaces that operate on
                activation vectors rather than event tensors).
        """
        if not returnVectors:
            self.subspace.batch = vspace.batch
            return vspace
        x = vspace.materialize()
        self.subspace.batch = x.shape[0]
        if self.nInputDim != -1:
            self._pre_reshape_input = (x.shape[1], x.shape[2])
            x = x.reshape(x.shape[0], -1, self.nInputDim)
        else:
            self._pre_reshape_input = None
        return x

    def forwardEnd(self, x, returnVectors=False, compute_activation=False):
        """Finalize output after space-specific processing.

        Args:
            x: dense tensor (returnVectors=True) or SubSpace (returnVectors=False).
            returnVectors: if True, reshape to [B, -1, nOutputDim] and store
                tensor into subspace.  If False, SubSpace passes through with
                reshape applied if nOutputDim != -1.
            compute_activation: if True (default), derive activation from vectors.
                Set to False for spaces (e.g. PerceptualSpace) where activation
                is not meaningful.
        """
        if not returnVectors:
            # Ensure [B, N, D] invariant even for SubSpace pass-through
            if self.nOutputDim != -1:
                vectors = x.materialize()
                if vectors is not None and vectors.ndim == 3 and vectors.shape[-1] != self.nOutputDim:
                    self._pre_reshape_output = (vectors.shape[1], vectors.shape[2])
                    vectors = vectors.reshape(vectors.shape[0], -1, self.nOutputDim)
                    x.set_event(vectors, compute_activation=compute_activation)
                else:
                    self._pre_reshape_output = None
            else:
                self._pre_reshape_output = None
            return x
        if self.nOutputDim != -1:
            self._pre_reshape_output = (x.shape[1], x.shape[2])
            x = x.reshape(x.shape[0], -1, self.nOutputDim)
        else:
            self._pre_reshape_output = None
        self.subspace.set_event(x, compute_activation=compute_activation)
        return self.subspace

    def reverseBegin(self, vspace, returnVectors=False):
        """Prepare input for space-specific reverse processing.

        Undoes the forwardEnd output reshape so the layer sees the same
        shape it produced during forward.

        Args:
            vspace: input SubSpace.
            returnVectors: if True, materialize to a dense tensor and
                undo the output reshape. If False, pass the SubSpace
                through without materializing.
        """
        if not returnVectors:
            self.subspace.batch = vspace.batch
            return vspace
        y = vspace.materialize()
        self.subspace.batch = y.shape[0]
        pre = getattr(self, '_pre_reshape_output', None)
        if pre is not None:
            y = y.reshape(y.shape[0], pre[0], pre[1])
        elif self.nOutputDim != -1:
            # Fallback when reverse is called without a prior forward:
            # reshape from [B, ?, nOutputDim] to [B, -1, layer_out_dim]
            layer_out = self.subspace.getEncodedOutputSize()
            if y.shape[-1] != layer_out:
                y = y.reshape(y.shape[0], -1, layer_out)
        return y

    def reverseEnd(self, y, returnVectors=False):
        """Finalize output after space-specific reverse processing.

        Undoes the forwardBegin input reshape so downstream sees the
        original input shape.

        Args:
            y: dense tensor (returnVectors=True) or SubSpace (returnVectors=False).
            returnVectors: if True, undo input reshape and store tensor
                into subspace.  If False, pass SubSpace through unchanged.
        """
        if not returnVectors:
            return y
        pre = getattr(self, '_pre_reshape_input', None)
        if pre is not None:
            y = y.reshape(y.shape[0], pre[0], pre[1])
        elif self.nInputDim != -1:
            # Fallback: reshape from [B, ?, nInputDim] to [B, -1, inputShape[1]]
            if y.shape[-1] != self.inputShape[1]:
                y = y.reshape(y.shape[0], -1, self.inputShape[1])
        self.subspace.set_event(y)
        return self.subspace

    # _2d/_3d removed — all layers now operate on [..., D] natively.

    def lookup(self, x):
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.zeros([1,1, TheXMLConfig.space("ConceptualSpace", "nDim")], device=TheDevice.get()), x[:,:,1:]], dim=2)
        output, index, _ = self.subspace.get_vectors().quantize(x)
        #output[:,:,0:conceptDim] = output[:,:,0:conceptDim] * activation  # multiply the codebook vector by the activation
        return output
    def dereference(self, symbols):
        # we get [ batch x nConcepts x symbolEmbedding ],
        # and must compute [ batch x nConcepts x conceptEmbedding ]
        batch = symbols.shape[0]
        nActive = self.outputShape[0]
        assert list(symbols.shape) == [batch, nActive, TheXMLConfig.space("SymbolicSpace", "nDim") + self.muxedSize - self.nWhat], "Incorrect input size for dereference"
        objects = torch.zeros(batch, nActive, self.muxedSize, device=TheDevice.get())
        for b in range(batch):
            for s in range(nActive):
                x = self.lookup(symbols[b,s,:])
                objects[b,s,:] = x
        assert list(objects.shape) == [batch, nActive, self.muxedSize], "Incorrect output size for dereference"
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

    def init_syntactic_layer(self, n_slots, grammar):
        """Override in subclasses that use grammar. Default: no-op."""
        self.syntacticLayer = None

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
        """InputSpace .event is a writable Tensor (receives muxed forward results)."""
        return None

    def _build_what_basis(self):
        """InputSpace .what holds the vocabulary (Embedding/Codebook/Tensor)."""
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
                byte_mode=getattr(self, 'byte_mode', False),
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
            )
            return basis

        if self.model_type in ("passthrough", "simple"):
            basis = Tensor()
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.nDim,
                passThrough=True,
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
        try:
            self.demuxed = TheXMLConfig.space(section, "demuxed")
        except KeyError:
            self.demuxed = False
        self.lexer = lexer  # "word", "sentence", "grammar", or "byte"
        self.byte_mode = (lexer == "byte")
        self.ergodic = ergodic
        self.min_frequency = float(min_frequency)
        self.neg_samples = neg_samples
        self.embedding_path = embedding_path
        self.embedding_source = data.train_input if data.train_input else None
        super().__init__(inputShape, spaceShape, outputShape)
        # InputSpace operates on raw [B, N, D] tensors directly (no forwardBegin/End).
        # Override any flatten-derived nInputDim/nOutputDim to skip reshape.
        self.nInputDim = -1
        self.nOutputDim = -1
        self.subspace._nInputDim = -1
        self.subspace._nOutputDim = -1
        lexical_basis = self.subspace.what
        if isinstance(lexical_basis, Embedding):
            self.doc_spans = lexical_basis.doc_spans
            self.doc_sources = lexical_basis.doc_sources
            if data.train_input and self.subspace.whereEncoding.nDim > 0:
                # Compute maxP from actual max byte offset in data,
                # not from data.inputLength (buffer size), which can be
                # far too large for short text and wastes encoding resolution.
                if (isinstance(data.train_input, list) and data.train_input
                        and isinstance(data.train_input[0], str)):
                    actual_max = max(len(s.encode('utf-8'))
                                     for s in data.train_input)
                    # 2x margin for validation/test data that may be longer
                    maxP = max(self.subspace.whereEncoding.maxVal,
                               actual_max * 2)
                else:
                    maxP = max(self.subspace.whereEncoding.maxVal,
                               data.inputLength)
                self.subspace.whereEncoding.maxVal = maxP
                self.subspace.whereEncoding.div_term = 2 * math.pi / maxP
        else:
            self.doc_spans = []
            self.doc_sources = []

        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = False
        fullSize  = outputShape[0]*outputShape[1]
        self.lift = MapppingLayer(fullSize, fullSize)
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
            return torch.stack(tensors, dim=0).unsqueeze(1).to(TheDevice.get())
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
            self.subspace.set_event(self.input)
            return self.subspace

        # Reset positional counter for each forward pass
        self.subspace.whereEncoding.p = 0

        batch = input.shape[0]
        nObj = self.outputShape[0]
        vocab = self.subspace.what
        dev = TheDevice.get()

        if not isinstance(vocab, Embedding):
            # Non-text path: input is already a tensor, no codebook indices
            assert list(input.shape) == [batch, self.inputShape[0], self.inputShape[1]]
            what = vocab.forward(input)
            self._forward_input = None
            self.subspace.set_what(what)
            if self.nWhere > 0:
                positions = torch.arange(nObj, dtype=torch.float32, device=dev).unsqueeze(0).expand(batch, -1)
                self.subspace.set_where(self.subspace.whereEncoding.encode(positions))
            if self.nWhen > 0:
                timesteps = torch.arange(nObj, dtype=torch.float32, device=dev).unsqueeze(0).expand(batch, -1)
                self.subspace.set_when(self.subspace.whenEncoding.encode(timesteps))
            self.input = self.subspace.materialize()
        else:
            # Text path: get token indices and byte offsets
            what, meta = vocab.forward(input, return_meta=True)
            self._forward_input = meta

            # what_indices: [B, N] codebook indices
            what_indices = meta['indices']  # [B, N] long tensor

            # where_indices: [B, N] byte offsets
            if self.nWhere > 0:
                where_indices = torch.zeros(batch, nObj, dtype=torch.long, device=dev)
                for b, batch_tokens in enumerate(meta['tokens']):
                    for i, (_, start) in enumerate(batch_tokens):
                        where_indices[b, i] = start
                    final_offset = meta['final_offsets'][b]
                    for i in range(len(batch_tokens), nObj):
                        where_indices[b, i] = final_offset + (i - len(batch_tokens))
            else:
                where_indices = None

            # when_indices: [B, N] timestep indices (sequential for now)
            if self.nWhen > 0:
                when_indices = torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1)
            else:
                when_indices = None

            # Set per-modality indices — materialize() looks up vectors from Basis
            self.subspace.set_forward_content(what_indices, where_indices, when_indices)
            self.input = self.subspace.materialize()

        # Scale what-content to [0,1] via data min-max (or assert if out of range).
        self.subspace.normalize("input", target="what", normalize=True)
        self.input = self.subspace.materialize()

        return self.subspace

    def expand_masked(self, embedded, sentence_text, maskedPrediction='MLM', n_words=None):
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
            n_words: explicit word count (byte mode — from span_meta after compaction)

        Returns:
            (masked_batch, mask_positions):
                masked_batch: [N, nVectors, embeddingSize]
                mask_positions: list[int] of length N
        """
        if n_words is not None:
            N = min(n_words, embedded.shape[1])
        else:
            words = sentence_text.split()
            N = min(len(words), self.outputShape[0])  # cap at nVectors

        # Repeat the embedded sentence N times
        masked = embedded.expand(N, -1, -1).detach().clone()  # [N, nVec, embSize]

        # Determine which dims are content (to zero) vs position (to preserve)
        embSize = embedded.shape[-1]
        content_mask = torch.ones(embSize, dtype=torch.bool, device=TheDevice.get())
        # Preserve nWhere dims (dynamic indices from whereEncoding)
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
        y = self.reverseBegin(vspace, returnVectors=True)
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
        self.subspace.set_event(self.input)

        # Word recovery — content is already denormalized, codebook is L2-normalized [-1,1]
        if isinstance(content_basis, Embedding):
            self._recovered_input = content_basis.decode_reverse_meta(
                self.input, subspace=self.subspace)
        else:
            self._recovered_input = None

        self.subspace.normalize("input", target="what", normalize=True)
        return self.subspace

    def reconstruct_data(self, text=False):
        """Render the last recovered text state stored on InputSpace."""
        if getattr(self, '_recovered_input', None) is None:
            raise RuntimeError("reconstruct_data() called before reverse()")
        return self.subspace.vocabulary.reconstruct_data(self._recovered_input, text=text)

    def reconstruct_to_buffer(self, buf_size=None):
        """Render the last recovered text buffer stored on InputSpace."""
        if getattr(self, '_recovered_input', None) is None:
            raise RuntimeError("reconstruct_to_buffer() called before reverse()")
        return self.subspace.vocabulary.reconstruct_to_buffer(
            self._recovered_input, buf_size=buf_size)

    def get_forward_meta(self):
        """Return the last forward-pass lexical metadata for text input."""
        return getattr(self, '_forward_input', None)

    def get_recovered_word(self, batch_idx, position):
        """Return one recovered token from the last InputSpace.reverse()."""
        if getattr(self, '_recovered_input', None) is None:
            return None
        return self.subspace.vocabulary.get_recovered_word(
            self._recovered_input, batch_idx, position)

    # ------------------------------------------------------------------
    # Training policy — InputSpace decides WHEN, Embedding does HOW
    # ------------------------------------------------------------------

    def train_embeddings(self, words, method='CBOW'):
        """Run one CBOW/SBOW gradient step if words are available."""
        emb = self.subspace.vocabulary
        if isinstance(emb, Embedding) and words:
            return emb.train_step(words, method=method)
        return None

    def sbow_loss(self, words):
        """Return SBOW loss tensor for joint optimization (no backward/step)."""
        emb = self.subspace.vocabulary
        if isinstance(emb, Embedding) and words:
            return emb.sbow_loss(words)
        return None

    def _snapshot_embeddings(self):
        """Return the current WordVectors (no-op, vectors are always live)."""
        emb = self.subspace.vocabulary
        if isinstance(emb, Embedding):
            return emb.wv
        return None

    def set_embedding_sigma(self, sigma):
        """Control exploration noise on the embedding."""
        emb = self.subspace.vocabulary
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
        mask = torch.zeros(N, nVec, dtype=torch.bool, device=TheDevice.get())
        for i, pos in enumerate(positions):
            mask[i, pos] = True
        return target, mask

    def predict(self, vector):
        """Delegates to Embedding.predict()."""
        return self.subspace.vocabulary.predict(vector)

    # ------------------------------------------------------------------
    # ARIR helpers
    # ------------------------------------------------------------------

    def embed_token(self, word):
        """Delegates to Embedding.embed_token()."""
        return self.subspace.vocabulary.embed_token(word)

    def get_space_embedding(self):
        """Delegates to Embedding.get_space_embedding()."""
        return self.subspace.vocabulary.get_space_embedding()

    def get_mask_embedding(self):
        """Delegates to Embedding.get_mask_embedding()."""
        return self.subspace.vocabulary.get_mask_embedding()

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
        embSize = self.muxedSize
        nWhat = self.subspace.vocabulary.embedding_dim

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
            null_emb = self.subspace.vocabulary.embed_token("\x00")
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
            dummy_input = torch.zeros(1, device=TheDevice.get())
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
    Uses a PiLayer (log-space multiplicative layer) to map input embeddings to
    perceptual embeddings, optionally followed by self-attention and VQ
    codebook quantization.

    When ``reversible=True`` and ``invertible=True``, a single layer
    serves both directions (shared weights).  Without invertibility, two
    PiLayers with separate weights are used: forward() on
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
        self.attention = AttentionLayer(input, input, type="transformer")
        self.subspace._nWordSlots = outputShape[0]
        self.params = []
        self.layers = nn.ModuleList()

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
        if not invertible:
            # Standard checks
            TheXMLConfig.require(
                lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv >= _na,
                f"PerceptualSpace: nVectors ({nV}) must be >= nOutput ({nA})"
            )
            if not self.codebook:
                TheXMLConfig.require(
                    lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv == _na,
                    f"PerceptualSpace: non-codebook requires nVectors ({nV}) == nOutput ({nA})"
                )

    def everything(self, target="what"):
        """The universal whole — vertex (1,1,...,1) of the perceptual hypercube.

        Valid only in perceptual space where vectors are sigmoid-normalized
        to [0,1]^d and mereological operations (min/max) apply.

        Args:
            target: encoding target — "what", "event", or "activation".
        """
        dim = {"what": self.nDim, "event": self.muxedSize,
               "activation": self.outputShape[0]}[target]
        return torch.ones(dim, device=TheDevice.get())

    def nothing(self, target="what"):
        """The empty set — origin (0,0,...,0) of the perceptual hypercube.

        Valid only in perceptual space where vectors are sigmoid-normalized
        to [0,1]^d and mereological operations (min/max) apply.

        Args:
            target: encoding target — "what", "event", or "activation".
        """
        dim = {"what": self.nDim, "event": self.muxedSize,
               "activation": self.outputShape[0]}[target]
        return torch.zeros(dim, device=TheDevice.get())

    def distance(self, x, y):
        return torch.prod( [1-x, 1-y] )
    def certainty(self, x):
        pass
    def forward(self, vspace):
        """Perception: map input vectors to percepts via attention + VQ + chunking."""
        if self.passThrough:
            return vspace
        # Pass byte values from input for boundary detection in compose()
        if getattr(vspace, '_demuxed', False) and vspace._active is not None:
            self.subspace._byte_indices = vspace._active[:, :, 0].long()
        x = self.forwardBegin(vspace, returnVectors=True)
        if self.hasAttention:
            x = self.attention.forward(x)
        if self.codebook:
            x = self.subspace.get_vectors().forward(x)
        if self.syntacticLayer is not None:
            x = self.syntacticLayer.compose(x, self.subspace, TheGrammar)
        vspace = self.forwardEnd(x, returnVectors=True)
        vspace.normalize("percepts", target="event", normalize=True)
        return vspace

    def init_syntactic_layer(self, n_slots, grammar):
        """Create the P-tier SyntacticLayer."""
        self.syntacticLayer = PerceptualSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.perceptual(),
            transition_rule=None,
            max_depth=max(n_slots - 1, 1),
            hidden_dim=min(256, max(64, n_slots * 4)),
            grammar=grammar,
        )
        self.layers.append(self.syntacticLayer)

    def reverse(self, vspace):
        """Manifesting: reconstruct input vectors from percepts."""
        if self.passThrough:
            return vspace
        y = self.reverseBegin(vspace, returnVectors=True)
        if self.syntacticLayer is not None:
            y = self.syntacticLayer.decompose(y, self.subspace, TheGrammar)
        vspace = self.reverseEnd(y, returnVectors=True)
        vspace.normalize("input", target="what", normalize=True)
        return vspace

    @staticmethod
    def test():
        pass
class ModalSpace(Space):
    """Composite space routing what/where/when through independent PerceptualSpaces.

    Default: what branch is processed (PiLayer), where/when branches are passthrough.
    When nWhere=nWhen=0, degenerates to a single PerceptualSpace on the full embedding.

    Per-branch passthrough flags are read from <ModalSpace> config:
        whatPassThrough  (default False)
        wherePassThrough (default True)
        whenPassThrough  (default True)
    """
    name = "Percepts"
    config_section = "ModalSpace"

    def __init__(self, inputShape, spaceShape, outputShape):
        section = self.config_section
        super().__init__(inputShape, spaceShape, outputShape)

        # Per-branch passthrough defaults
        try:
            whatPT = TheXMLConfig.space(section, "whatPassThrough")
        except KeyError:
            whatPT = False
        try:
            wherePT = TheXMLConfig.space(section, "wherePassThrough")
        except KeyError:
            wherePT = True
        try:
            whenPT = TheXMLConfig.space(section, "whenPassThrough")
        except KeyError:
            whenPT = True

        # Derive branch shapes (symmetric — subtract off the modality you don't need)
        whatDim = self.muxedSize - self.nWhere - self.nWhen
        whatInputShape = [inputShape[0], whatDim]
        whatOutputShape = [outputShape[0], whatDim]
        whatSpaceShape = [spaceShape[0], spaceShape[1]]

        # Build what branch — override passThrough in config temporarily
        saved_pt = TheXMLConfig._data.get("PerceptualSpace", {}).get("passThrough")
        TheXMLConfig._data.setdefault("PerceptualSpace", {})["passThrough"] = whatPT
        self.whatSpace = PerceptualSpace(whatInputShape, whatSpaceShape, whatOutputShape)
        TheXMLConfig._data["PerceptualSpace"]["passThrough"] = saved_pt

        # Build where branch (if nWhere > 0)
        if self.nWhere > 0:
            whereShape = [inputShape[0], self.nWhere]
            whereSpaceShape = [spaceShape[0], self.nWhere]
            saved_pt = TheXMLConfig._data.get("PerceptualSpace", {}).get("passThrough")
            TheXMLConfig._data["PerceptualSpace"]["passThrough"] = wherePT
            self.whereSpace = PerceptualSpace(whereShape, whereSpaceShape, whereShape)
            TheXMLConfig._data["PerceptualSpace"]["passThrough"] = saved_pt
        else:
            self.whereSpace = None

        # Build when branch (if nWhen > 0)
        if self.nWhen > 0:
            whenShape = [inputShape[0], self.nWhen]
            whenSpaceShape = [spaceShape[0], self.nWhen]
            saved_pt = TheXMLConfig._data.get("PerceptualSpace", {}).get("passThrough")
            TheXMLConfig._data["PerceptualSpace"]["passThrough"] = whenPT
            self.whenSpace = PerceptualSpace(whenShape, whenSpaceShape, whenShape)
            TheXMLConfig._data["PerceptualSpace"]["passThrough"] = saved_pt
        else:
            self.whenSpace = None

        # Collect parameters and layers from all branches
        self.params = list(self.whatSpace.getParameters())
        self.layers = nn.ModuleList([self.whatSpace])
        if self.whereSpace is not None:
            self.params.extend(self.whereSpace.getParameters())
            self.layers.append(self.whereSpace)
        if self.whenSpace is not None:
            self.params.extend(self.whenSpace.getParameters())
            self.layers.append(self.whenSpace)

    def _register_requirements(self):
        """ModalSpace manages its own branch requirements."""
        pass

    def forward(self, vspace):
        """Route each modality through its branch PerceptualSpace."""
        if vspace.is_demuxed:
            what_in = vspace.what.getW()
            where_in = vspace.where.getW() if vspace.where is not None else None
            when_in = vspace.when.getW() if vspace.when is not None else None
        else:
            # Fallback: split muxed event into branches
            event = vspace.materialize()
            what_in = event[..., :self.nWhat]
            where_in = event[..., self.nWhat:self.nWhat + self.nWhere] if self.nWhere > 0 else None
            when_in = event[..., self.nWhat + self.nWhere:] if self.nWhen > 0 else None

        # Route what through whatSpace
        what_sub = SubSpace(inputShape=[what_in.shape[1], what_in.shape[2]],
                           outputShape=[what_in.shape[1], what_in.shape[2]])
        what_sub.set_what(what_in)
        what_out = self.whatSpace.forward(what_sub).materialize()

        # Route where through whereSpace (passthrough if no whereSpace)
        where_out = where_in
        if self.whereSpace is not None and where_in is not None:
            where_sub = SubSpace(inputShape=[where_in.shape[1], where_in.shape[2]],
                                outputShape=[where_in.shape[1], where_in.shape[2]])
            where_sub.set_where(where_in)
            where_out = self.whereSpace.forward(where_sub).materialize()

        # Route when through whenSpace (passthrough if no whenSpace)
        when_out = when_in
        if self.whenSpace is not None and when_in is not None:
            when_sub = SubSpace(inputShape=[when_in.shape[1], when_in.shape[2]],
                               outputShape=[when_in.shape[1], when_in.shape[2]])
            when_sub.set_when(when_in)
            when_out = self.whenSpace.forward(when_sub).materialize()

        # Build output demuxed SubSpace
        out = SubSpace(inputShape=self.outputShape, outputShape=self.outputShape,
                      whereEncoding=self.subspace.whereEncoding,
                      whenEncoding=self.subspace.whenEncoding)
        out.set_demuxed(what_out, where_out, when_out)
        return out

    def reverse(self, vspace):
        """Split event into modalities, reverse each branch, rebuild."""
        event = vspace.materialize()
        what_in = event[..., :self.nWhat]
        where_in = event[..., self.nWhat:self.nWhat + self.nWhere] if self.nWhere > 0 else None
        when_in = event[..., self.nWhat + self.nWhere:] if self.nWhen > 0 else None

        # Reverse what
        what_sub = SubSpace(inputShape=[what_in.shape[1], what_in.shape[2]],
                           outputShape=[what_in.shape[1], what_in.shape[2]])
        what_sub.set_what(what_in)
        what_rev = self.whatSpace.reverse(what_sub).materialize()

        # Reverse where
        where_rev = where_in
        if self.whereSpace is not None and where_in is not None:
            where_sub = SubSpace(inputShape=[where_in.shape[1], where_in.shape[2]],
                                outputShape=[where_in.shape[1], where_in.shape[2]])
            where_sub.set_where(where_in)
            where_rev = self.whereSpace.reverse(where_sub).materialize()

        # Reverse when
        when_rev = when_in
        if self.whenSpace is not None and when_in is not None:
            when_sub = SubSpace(inputShape=[when_in.shape[1], when_in.shape[2]],
                               outputShape=[when_in.shape[1], when_in.shape[2]])
            when_sub.set_when(when_in)
            when_rev = self.whenSpace.reverse(when_sub).materialize()

        # Rebuild demuxed SubSpace
        out = SubSpace(inputShape=self.inputShape, outputShape=self.inputShape,
                      whereEncoding=self.subspace.whereEncoding,
                      whenEncoding=self.subspace.whenEncoding)
        out.set_demuxed(what_rev, where_rev, when_rev)
        return out

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all branch spaces."""
        self.whatSpace.set_sigma(sigma)
        if self.whereSpace is not None:
            self.whereSpace.set_sigma(sigma)
        if self.whenSpace is not None:
            self.whenSpace.set_sigma(sigma)

    def getParameters(self):
        return self.params

    def paramUpdate(self):
        self.whatSpace.paramUpdate()
        if self.whereSpace is not None:
            self.whereSpace.paramUpdate()
        if self.whenSpace is not None:
            self.whenSpace.paramUpdate()
class ConceptualSpace(Space):
    """Transforms percepts into concepts via a SigmaLayer (summation layer).

    In the forward data flow: PerceptualSpace -> **ConceptualSpace** -> SymbolicSpace.
    Uses a SigmaLayer to combine perceptual features into conceptual
    representations.  The SigmaLayer computes weighted sums (inner products)
    rather than the permutation-equivariant operations of PiLayer.

    Supports optional self-attention and VQ codebook quantization.

    When ``invertible=True``, uses a InvertibleSigmaLayer whose inverse is
    exact.  When ``reversible=True`` without invertibility, a separate
    SigmaLayer is trained for the reverse direction.
    """
    name = "Concepts"
    config_section = "ConceptualSpace"

    def __init__(self, inputShape, spaceShape, outputShape, level_shapes=None):
        section = self.config_section
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        naive = TheXMLConfig.get("architecture.naive")
        super().__init__(inputShape, spaceShape, outputShape)
        self.nonlinear = nonlinear
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        input = self.subspace.getEncodedInputSize()
        output = self.subspace.getEncodedOutputSize()
        if hasAttention:
            self.attention = AttentionLayer(output, output, type="transformer")

        # ── Hierarchical mode: per-level Sigma layers ────────────────
        # Average-merge keeps norms bounded, so tanh saturation is unnecessary
        # and harmful: cascaded atanh in reverse clamps values outside (-1,1),
        # destroying sample variance.  Per-level sigmas use saturate=False.
        if level_shapes is not None and len(level_shapes) >= 1:
            self._hierarchical = True
            self._level_shapes = level_shapes
            self.sigmas = nn.ModuleList()
            for t, (n_t, d_t) in enumerate(level_shapes):
                sig = SigmaLayer(d_t, d_t, naive=naive, ergodic=ergodic,
                                 invertible=invertible)
                sig.saturate = False
                self.sigmas.append(sig)
            # Dim projections for syntactic mode (grammar keeps D, need projection)
            # One per level: level 0 projects from percept_dim, others from prior level
            self.dim_projections = nn.ModuleList()
            percept_dim = inputShape[1]  # pre-merge percept dim
            for t, (n_t, d_t) in enumerate(level_shapes):
                d_in = percept_dim if t == 0 else level_shapes[t - 1][1]
                self.dim_projections.append(nn.Linear(d_in, d_t))
            self.layers = nn.ModuleList(
                list(self.sigmas) + list(self.dim_projections))
            self.params = []
            for s in self.sigmas:
                self.params.extend(s.getParameters())
            # Set forwardSigma to level-0 for backward compat callers
            self.forwardSigma = self.sigmas[0].forward
        else:
            # ── Original single-layer mode ────────────────────────────
            self._hierarchical = False
            self._level_shapes = None
            if self.reversible:
                if invertible:
                    self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                            invertible=True, nonlinear=nonlinear)
                    self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                    self.params = self.sigma.getParameters()
                    self.layers = nn.ModuleList([self.sigma])
                else:
                    self.sigma1 = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                             invertible=True, nonlinear=nonlinear)
                    self.sigma2 = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                             invertible=True, nonlinear=nonlinear)
                    self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.reverse
                    self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
                    self.layers = nn.ModuleList([self.sigma1, self.sigma2])
            else:
                self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                        nonlinear=nonlinear)
                self.forwardSigma = self.sigma.forward
                self.params = self.sigma.getParameters()
                self.layers = nn.ModuleList([self.sigma])
        # Grammar methods and SyntacticLayers are now on TheGrammar.
        # Spaces delegate to TheGrammar.project('C', ...) and
        # TheGrammar.composeSyntax('C', ...).
        self._interpretation = TheGrammar.interpretation

    def __getitem__(self, t):
        """Index into conceptual order levels.

        Non-hierarchical: returns self (shared sigma for all t).
        Hierarchical: returns a _LevelView that routes through sigmas[t].
        """
        if not self._hierarchical:
            return self
        return self._CSLevelView(self, t)

    class _CSLevelView:
        """Proxy routing .forward()/.reverse() through a per-level sigma."""
        def __init__(self, parent, t):
            self._parent = parent
            self._sigma = parent.sigmas[t]
            self.subspace = parent.subspace

        def forward(self, vspace):
            x = vspace.materialize()
            y = self._sigma.forward(x)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

        def reverse(self, vspace):
            x = vspace.materialize()
            y = self._sigma.reverse(x)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

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
        """Knowing: map percepts to concepts via SigmaLayer + optional attention + VQ.

        When nonlinear=True, applies atanh() before SigmaLayer so that the
        reverse tanh (needed to bound output to (-1,1)) has an exact
        mathematical inverse.
        """
        x = self.forwardBegin(vspace, returnVectors=True)
        if self.nonlinear:
            # atanh only on nWhat dims — sin/cos where/when values near ±1
            # would explode through atanh, so leave them untransformed.
            nW = self.subspace.nWhat
            x_what = torch.atanh(x[:, :, :nW] * (1 - 1e-6))
            x = torch.cat([x_what, x[:, :, nW:]], dim=-1)
        y = self.forwardSigma(x)
        if self.hasAttention:
            y = self.attention.forward(y)
        if self.codebook:
            y = self.subspace.get_vectors().forward(y)
        if self.syntacticLayer is not None:
            y, self._last_svo = self.syntacticLayer.compose(y, self.subspace, TheGrammar)
        vspace = self.forwardEnd(y, returnVectors=True)
        vspace.normalize("concepts", target="what")       # range check
        vspace.normalize("concepts", target="activation")  # range check
        return vspace

    def init_syntactic_layer(self, n_slots, grammar, concept_dim=0):
        """Create the C-tier SyntacticLayer."""
        self.syntacticLayer = ConceptualSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.conceptual(),
            transition_rule=grammar.conceptual_transition(),
            max_depth=max(n_slots - 1, 1),
            hidden_dim=min(256, max(64, n_slots * 4)),
            grammar=grammar,
        )
        self.syntacticLayer.init_conceptual_params(concept_dim)
        self.layers.append(self.syntacticLayer)

    @property
    def last_svo(self):
        """Return SVO tuple from last ternary lift, or None."""
        return getattr(self, '_last_svo', None)

    def reverse(self, vspace):
        """Visualizing: reconstruct percepts from concepts via reverse SigmaLayer.

        When nonlinear=True, applies tanh() after reverse SigmaLayer to
        guarantee output in (-1,1) for PiLayer.
        """
        y = self.reverseBegin(vspace, returnVectors=True)
        if self.syntacticLayer is not None:
            y = self.syntacticLayer.decompose(y, self.subspace, TheGrammar)
        if self.processSymbols:
            y = self.dereference(y)
        y = self.reverseSigma(y)
        if self.nonlinear:
            # tanh only on nWhat dims — mirror the forward atanh split.
            nW = self.subspace.nWhat
            y_what = torch.tanh(y[:, :, :nW])
            y = torch.cat([y_what, y[:, :, nW:]], dim=-1)
        self.concepts = y
        vspace = self.reverseEnd(y, returnVectors=True)
        vspace.normalize("percepts", target="what")  # range check
        return vspace

    @staticmethod
    def test():
        pass
class SymbolicSpace(Space):
    """Codebook-backed symbol stack with swap operations.

    In the forward data flow: ConceptualSpace -> **SymbolicSpace** -> OutputSpace.
    The symbol stack (StackSpace) holds entries produced by ConceptualSpace's
    shift/reduce loop. Each entry has what (codebook index), where (position),
    and when (derivation order).

    S-tier operations (swap) operate on whereEncodings of node children.
    The START-level true() evaluates the full stack activation → scalar.
    """
    name = "Symbols"
    config_section = "SymbolicSpace"

    def __init__(self, inputShape, spaceShape, outputShape, conceptualSpace=None,
                 level_shapes=None):

        section = self.config_section
        passThrough = TheXMLConfig.space(section, "passThrough")
        super().__init__(inputShape, spaceShape, outputShape, customVQ=True)
        self.conceptualSpace = conceptualSpace
        self.passThrough = passThrough
        # PiLayer maps on the nDim axis: concept_dim+obj → symbol_dim+obj.
        # nVectors passes through unchanged via batched matmul.
        nConceptDim = inputShape[1]     # concept_dim + obj (where+when)
        nSymbolDim = outputShape[1]     # symbol_dim + obj (where+when)
        nSymbols = spaceShape[0]

        if level_shapes is not None and len(level_shapes) >= 1:
            self._hierarchical = True
            self._level_shapes = level_shapes
            self.pi_layers = nn.ModuleList()
            for t, (n_t, d_t) in enumerate(level_shapes):
                self.pi_layers.append(
                    PiLayer(d_t, nSymbolDim, invertible=True, monotonic=True))
            self.layer = self.pi_layers[0]  # default for non-ramsified codepaths
        else:
            self._hierarchical = False
            self.pi_layers = None
        self.layer = PiLayer(nConceptDim, nSymbolDim, invertible=True, monotonic=True) if not self._hierarchical else self.pi_layers[0]

        # Truth accumulation: accumulateTruth is 0..1 (DoT for recorded symbols).
        # Default 0 (off). Server sets to 1 when processing the TruthSet,
        # then resets to 0.  TruthLayer is always created so the server can
        # toggle recording at runtime.
        try:
            self.accumulateTruth = float(TheXMLConfig.space(section, "accumulateTruth"))
        except (KeyError, TypeError, ValueError):
            self.accumulateTruth = 0.0
        from Model import TruthLayer
        self.truth = TruthLayer(nSymbolDim)

        # Truth storage criterion thresholds (used by should_store())
        def _truth_cfg(key, default):
            try:
                return float(TheXMLConfig.space(section, key))
            except (KeyError, TypeError, ValueError):
                return default
        self._truth_min_magnitude = _truth_cfg("truthMinMagnitude", 0.3)
        self._truth_min_novelty = _truth_cfg("truthMinNovelty", 0.5)
        self._truth_max_inconsistency = _truth_cfg("truthMaxInconsistency", 0.3)

        # Odd-even sorting network: learns a canonical ordering of symbols.
        try:
            sort_enabled = TheXMLConfig.space(section, "sortNetwork")
        except (KeyError, TypeError, ValueError):
            sort_enabled = False
        self.sortNetwork = None
        if sort_enabled:
            try:
                sort_passes_cfg = int(TheXMLConfig.space(section, "sortPasses"))
            except (KeyError, TypeError, ValueError):
                sort_passes_cfg = 0
            from Model import SortingLayer
            n_passes = sort_passes_cfg if sort_passes_cfg > 0 else None
            self.sortNetwork = SortingLayer(nSymbolDim, n_passes=n_passes)

        pi_list = list(self.pi_layers) if self._hierarchical else [self.layer]
        self.layers = nn.ModuleList(
            pi_list + ([self.sortNetwork] if self.sortNetwork else [])
        )

        # Assign fixed where encodings to symbol positions.
        nPercepts = inputShape[0]
        if self.nWhere > 0:
            positions = torch.arange(nPercepts, nPercepts + nSymbols, dtype=torch.float32)
            self._symbol_where = self.subspace.whereEncoding.encode(positions)
        else:
            self._symbol_where = None

    def _build_object_basis(self):
        """Event is a writable Tensor — codebook lives on .what."""
        return None

    def _build_what_basis(self):
        """Symbol codebook on .what, monotonic (negation meaningless)."""
        basis = Codebook()
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
            customVQ=self.customVQ,
            passThrough=not self.codebook,
            monotonic=True,
        )
        return basis

    def __getitem__(self, t):
        """Index into conceptual order levels.

        Non-hierarchical: returns self (shared PiLayer for all t).
        Hierarchical: returns a _LevelView that routes through pi_layers[t].
        """
        if not self._hierarchical:
            return self
        return self._SSLevelView(self, t)

    class _SSLevelView:
        """Proxy routing .forward()/.reverse() through a per-level PiLayer."""
        def __init__(self, parent, t):
            self._parent = parent
            self._pi = parent.pi_layers[t]
            self.subspace = parent.subspace

        def forward(self, vspace):
            x = vspace.materialize()
            y = self._pi.forward(x)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

        def reverse(self, vspace):
            x = vspace.materialize()
            y = self._pi.reverse(x)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

    @property
    def vocabulary(self):
        return self.subspace.what

    def forward(self, vspace):
        """Map concept vectors to symbol vectors via PiLayer (Π).

        PiLayer maps on the nDim axis: [B, nVectors, concept_dim] →
        [B, nVectors, symbol_dim].  nVectors passes through unchanged.
        With a single-concept subspace [B, 1, D], produces [B, 1, symbol_dim].

        1. Materialize full concept vectors from input subspace.
        2. Map through PiLayer (log-space multiplicative, monotonic).
        3. Grammar derivation (if syntax=True): shift/reduce over S-tier.
        4. Apply swap on whereEncodings of binary node children.
        5. Store as symbol vectors in subspace event.
        """
        if self.passThrough:
            return vspace
        vspace = self.forwardBegin(vspace)
        act = vspace.materialize()                        # [B, N, concept_dim]
        act = self.layer.forward(act)                     # [B, N, symbol_dim]

        if self.accumulateTruth > 0:
            for i in range(act.shape[0]):
                for j in range(act.shape[1]):
                    vec = act[i, j]
                    score = self.truth.should_store(
                        vec,
                        min_magnitude=self._truth_min_magnitude,
                        min_novelty=self._truth_min_novelty,
                        max_inconsistency=self._truth_max_inconsistency)
                    if self.accumulateTruth * score > 0.5:
                        self.truth.record(vec, degree=self.accumulateTruth)

        if self._symbol_where is not None:
            B = act.shape[0]
            nAct = act.shape[1]
            # Only apply where if vector count matches stored encodings
            if nAct == self._symbol_where.shape[0]:
                where = self._symbol_where.unsqueeze(0).expand(B, -1, -1)
                where = where.to(act.device)
                self.subspace.where.setW(where)

        if self.sortNetwork is not None:
            act = self.sortNetwork.forward(act)

        if self.syntacticLayer is not None:
            act = self.syntacticLayer.compose(act, self.subspace, TheGrammar)

        if self.codebook:
            self.subspace.set_event(act)
            self.subspace.what.forward(self.subspace)
            vspace = self.forwardEnd(self.subspace)
        else:
            self.subspace.set_event(act)

        return vspace

    def init_syntactic_layer(self, n_slots, grammar, symbol_dim=0):
        """Create the S-tier SyntacticLayer."""
        self.syntacticLayer = SymbolicSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.symbolic(),
            transition_rule=grammar.symbolic_transition(),
            max_depth=max(n_slots - 1, 1),
            hidden_dim=min(256, max(64, n_slots * 4)),
            grammar=grammar,
        )
        self.syntacticLayer.init_swap(symbol_dim, n_slots)
        self.layers.append(self.syntacticLayer)

    def reverse(self, vspace):
        """Map symbol vectors back to concept vectors via PiLayer.reverse (Π⁻¹).

        Reverse maps on nDim axis: [B, N, symbol_dim] → [B, N, concept_dim].
        """
        if self.passThrough:
            return vspace
        vspace = self.reverseBegin(vspace)
        act = vspace.materialize()                        # [B, N, symbol_dim]
        if self.syntacticLayer is not None:
            act = self.syntacticLayer.decompose(act, self.subspace, TheGrammar)
        if self.sortNetwork is not None:
            act = self.sortNetwork.reverse(act)
        act = self.layer.reverse(act)                     # [B, N, concept_dim]
        if self.codebook:
            self.subspace.set_event(act)
            result = self.reverseEnd(self.subspace)
        else:
            self.subspace.set_event(act)
            result = self.subspace
        result.normalize("concepts", target="what", normalize=True)
        return result

    def evaluate_truth(self, vspace):
        """START-level: evaluate truth of the full stack → scalar."""
        act = vspace.materialize(mode="activation")
        return self.syntacticLayer.trueForward(act, self.subspace)

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
        # OutputSpace always uses its own Tensor basis for forward results.
        # The Embedding reference (if any) is stored separately for text_mode reverse.
        initial_vectors = getattr(self, "_initial_vectors", None)
        if isinstance(initial_vectors, Basis):
            self._vocabulary = initial_vectors  # keep for text_mode reverse
        basis = Tensor()
        basis.create(
            self.inputShape[0],
            self.outputShape[0],
            self.muxedSize,  # full event width
            passThrough=True,
        )
        return basis

    def __init__(self, inputShape, spaceShape, outputShape, masked_prediction=False, vectors=None):
        section = self.config_section
        invertible = TheXMLConfig.space(section, "invertible")
        self.masked_prediction = masked_prediction
        object.__setattr__(self, "_initial_vectors", vectors)
        self.nonlinear_output = TheXMLConfig.space(section, "nonlinear")
        super().__init__(inputShape, spaceShape, outputShape)
        self.data = TheData
        self._vocabulary = getattr(self, '_vocabulary', None)
        self.text_mode = isinstance(self._vocabulary, Embedding)

        if self.nonlinear_output:
            # PiLayer activation-mode path for ramsified symbol output
            nIn = inputShape[0]
            nOut = outputShape[0]
            self._piLayer = PiLayer(nIn, nOut, invertible=True, monotonic=True)
            self.layers = nn.ModuleList([self._piLayer])
        else:
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
            return torch.stack(outputBatch, dim=0).unsqueeze(1).to(TheDevice.get())
        return outputBatch  # already [B, D, 1] and on device after toDevice()
    def forward(self, vspace):
        """Acting: project symbols to task output."""
        if self.nonlinear_output:
            # Activation-mode: PiLayer on symbol activations [B, nSymbols] → [B, nOutput]
            act = vspace.materialize(mode="activation")
            output = self._piLayer.forward(act)
            from data import TheData
            output = TheData.denormalize(output, which="output")
            self.subspace.set_activation(output)
            return self.subspace

        # Default vector-mode: LinearLayer on flattened vectors
        x = self.forwardBegin(vspace, returnVectors=True)
        output = self.forwardLinear(x)
        from data import TheData
        output = TheData.denormalize(output, which="output")
        if self.codebook:
            output = self.subspace.get_vectors().forward(output)
        vspace = self.forwardEnd(output, returnVectors=True)
        return vspace

    def reverse(self, vspace):
        """Being acted upon: map output back to symbolic space."""
        if self.nonlinear_output:
            # Activation-mode: PiLayer reverse [B, nOutput] → [B, nSymbols]
            act = vspace.materialize(mode="activation")
            from data import TheData
            act_norm = TheData.normalize(act, which="output")
            symbol_act = self._piLayer.reverse(act_norm)
            self.subspace.set_activation(symbol_act)
            return self.subspace

        # Default vector-mode
        y = self.reverseBegin(vspace, returnVectors=True)
        self.subspace.set_event(y)
        self.subspace.denormalize("output", target="what")
        y = self.subspace.materialize()
        y = self.reverseLinear(y)
        vspace = self.reverseEnd(y, returnVectors=True)
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
            return torch.zeros(0, 1, embSize, device=TheDevice.get())
        # Extract the first N full word vectors (nWhat + nWhere + nWhen)
        targets = embedded[0, :N, :].clone()  # [N, embSize]
        if maskedPrediction == 'ARUS':
            targets = torch.zeros_like(targets)
        elif maskedPrediction == 'RARLM':
            targets = targets.flip(0)
        return targets.unsqueeze(1)  # [N, 1, embSize]
    # --- Text reconstruction from symbolic vectors ---
    def set_text_mode(self, input_space):
        """Share InputSpace's Embedding so OutputSpace can reconstruct text.

        Convenience method for tests. Production code passes vectors= to __init__.
        """
        vs = input_space.subspace.vocabulary
        self._vocabulary = vs
        self.text_mode = isinstance(vs, Embedding)

    def _reverse_text_vectors(self, vectors):
        emb = self._vocabulary
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
        return self._vocabulary.reconstruct_to_buffer(
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
