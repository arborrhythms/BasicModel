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
# Minimal in-repo VectorQuantize -- covers the subset of the
# vector_quantize_pytorch API Codebook uses (forward, .codebook, EMA hook
# stub). The external package was removed once the commitment loss, STE
# path, and rotation trick were owned by Codebook directly.
_vq_F = F
class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        commitment_weight=1.0,
        use_cosine_sim=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.use_cosine_sim = use_cosine_sim
        self.codebook = torch.randn(codebook_size, dim)

    @property
    def codebook(self):
        return self._parameters["_codebook"]

    @codebook.setter
    def codebook(self, value):
        param = value if isinstance(value, nn.Parameter) else nn.Parameter(value.detach().clone())
        if "_codebook" in self._parameters:
            self._parameters["_codebook"] = param
        else:
            self.register_parameter("_codebook", param)
        self.codebook_size = param.shape[0]

    def forward(self, x, return_all_codes=False, **kwargs):
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        codebook = self.codebook
        if self.use_cosine_sim:
            flat_cmp = _vq_F.normalize(flat, dim=-1)
            codebook_cmp = _vq_F.normalize(codebook, dim=-1)
            indices = (flat_cmp @ codebook_cmp.T).argmax(dim=-1)
        else:
            indices = torch.cdist(flat, codebook).argmin(dim=-1)
        quantized_raw = codebook[indices].reshape(original_shape)
        commit_loss = self.commitment_weight * _vq_F.mse_loss(
            x, quantized_raw.detach()
        )
        quantized = x + (quantized_raw - x).detach()
        indices = indices.reshape(original_shape[:-1])
        if return_all_codes:
            return quantized, indices, commit_loss, quantized.unsqueeze(0)
        return quantized, indices, commit_loss
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
from Layers import Layer, PiLayer, SigmaLayer, ButterflyStage  # Import custom layers from Model.py
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, LiftingLayer, LoweringLayer, ChunkLayer
from Layers import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon
from Layers import SortingLayer, TruthLayer, InterSentenceLayer, SparsityRegularizer, SmoothingRegularizer, ImpenetrableLayer
from parse import quick_parser
from collections import namedtuple as _namedtuple


def topk_by_magnitude_per_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all but the top-k entries by |x| along the last dim, per row.

    Shape: x is (..., W). Returns tensor of same shape where each row has
    at most k nonzero entries (the k largest by absolute value).
    """
    if k <= 0:
        return torch.zeros_like(x)
    W = x.shape[-1]
    if k >= W:
        return x
    _, idx = torch.topk(x.abs(), k=k, dim=-1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return torch.where(mask, x, torch.zeros_like(x))


class Encoding(nn.Module):
    """Abstract base class for per-slot encodings in the embedding vector.

    Each encoding occupies ``nDim`` contiguous slots at positions given by
    ``self.index`` (negative offsets from the end of the embedding).

    Subclasses implement:
        ``encode(value)``  -> tensor [..., nDim]
        ``decode(encoded)`` -> decoded value (tensor or scalar)
        ``forward(x)``     -> stamped tensor (how values are assigned per-batch)
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

    Encodes scalar values as (sin, cos) pairs at frequency ``div_term = 2pi/maxVal``.
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
    """Per-slot 4-valued (quaternary) truth activation.

    Carries a bivector [aP, aN] per position encoding the four corners of
    the tetralemma (catuskoti): TRUE=[1,0], FALSE=[0,1], BOTH=[1,1],
    NEITHER=[0,0]. See basicmodel/doc/BuddhistParallels.md for the
    Nagarjuna-style semantics.

    encode/decode are identity -- the producing Space decides how to
    compute activation values. ActiveEncoding is the carrier.
    """
    nDim = 2

    def __init__(self, maxVal=1.0):
        super().__init__([-5], maxVal)

    def encode(self, activation):
        """Identity encode: activation values pass through."""
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(float(activation))
        return activation.unsqueeze(-1) if activation.dim() == 0 else activation

    def decode(self, encoded):
        """Identity decode: return activation tensor."""
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
    at a single frequency ``div_term = 2pi/maxVal``.  This is exactly invertible
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
        super().__init__([], maxVal=0)  # no index slots -- not muxed
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
        # W is NOT stored here -- subclasses own the storage.
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
        """Negation.
        Bitonic: sign flip (-x).
        Monotonic, last-dim == 2: swap (pos, neg) -- the tetralemma flip
          on a demuxed `.what` bivector [aP, aN] -> [aN, aP].
        Monotonic, last-dim == 2K (legacy paired-index storage): pair flip.
        Domain [-1, 1]. Range: bitonic [-1, 1], monotonic [0, 1]."""
        if not monotonic:
            return -x
        n = x.shape[-1]
        if n == 2:
            return x.flip(dims=(-1,))
        if n % 2 == 0:
            pair = x.reshape(*x.shape[:-1], n // 2, 2)
            flipped = pair.flip(dims=(-1,))
            return flipped.reshape(*x.shape)
        raise ValueError(
            f"Basis.negation(monotonic=True) requires even last dim; "
            f"got shape {tuple(x.shape)}"
        )

    def non(self, x, monotonic=False, threshold=None):
        """Non-affirming negation. Bitonic: -> 0. Monotonic: learnable threshold.
        Domain [-1, 1]. Range [0, 1] (monotonic) or {0} (bitonic)."""
        if monotonic and threshold is not None:
            return torch.relu(x - threshold)
        return torch.zeros_like(x)

    # -- Inverse logic operations -----------------------------------------------

    def negation_inverse(self, x, monotonic=False):
        """Inverse of negation. Self-inverse in both modes.
        Bitonic: sign flip. Monotonic: paired-index flip."""
        return self.negation(x, monotonic=monotonic)

    def conjunction_inverse(self, result, y, monotonic=False):
        """Inverse of conjunction via codebook search.

        Find the codebook vector x such that conjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        return self._binary_op_inverse(result, self.conjunction, monotonic)

    def disjunction_inverse(self, result, y, monotonic=False):
        """Inverse of disjunction via codebook search.

        Find the codebook vector x such that disjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        return self._binary_op_inverse(result, self.disjunction, monotonic)

    def _binary_op_inverse(self, result, op, monotonic):
        """Search codebook for pair (cb[i], cb[j]) whose op(cb[i], cb[j]) ~= result.

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

        # Precompute op(cb[i], cb[j]) for all pairs -> (K, K, D)
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
            # (end-start, 1, D) - (1, K*K, D) -> (end-start, K*K)
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
        matching zeros contribute nothing -- zero-volume elements have no
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
    # Parthood is the fundamental mereological operation. Every member of
    # the suite (whole, equal, overlap, underlap, boundary) has both a
    # vector form (default) that operates over the concept space and a
    # scalar form (scalar=True) that reduces to [0, 1] / bool.
    #
    # Vector forms (scalar=False):
    #     part(x, y)     = x * (y / ||y||)                     elementwise
    #     whole(x, y)    = (1 - x) * (y / ||y||)               elementwise
    #     equal(x, y)    = part(x, y) * part(y, x)             elementwise
    #     overlap(x, y)  = min(part(x, y), part(y, x))         elementwise
    #     underlap(x, y) = min(whole(x, y), whole(y, x))       elementwise
    #     boundary(x, y) = |part(x, y) - part(y, x)|           elementwise
    #
    # Scalar forms (scalar=True): clipped-cosine parthood and the
    # region-indicator relations derived from it. See each method.

    def part(self, x, y, monotonic=False, scalar=False):
        """Part of x under y.

        Vector form (default, scalar=False):
            part(x, y) = x * (y / ||y||)   -- elementwise projection of
            x into y's unit direction. Returns a tensor shaped like x.

        Scalar form (scalar=True): clipped cosine projection in [0, 1].
            part(x, y) = max(0, x.y) / (||x|| * ||y||)
        Satisfies Boole's contrapositive: part(x, y) = part(-y, -x).
        Empty-set conventions:
            part(empty, y) = 1, part(x, empty) = 0, part(empty, empty) = 1.
        """
        ny_raw = self.norm(y)
        ny = ny_raw.clamp(min=epsilon)
        if not scalar:
            return x * (y / ny.unsqueeze(-1))
        nx = self.norm(x)
        dot = (x * y).sum(dim=-1)
        clipped = torch.clamp(dot, min=0.0)
        denom = (nx * ny).clamp(min=epsilon)
        score = (clipped / denom).clamp(0.0, 1.0)
        empty_x = nx < epsilon
        empty_y = ny_raw < epsilon
        ones = torch.ones_like(score)
        zeros = torch.zeros_like(score)
        return torch.where(empty_x, ones, torch.where(empty_y, zeros, score))

    def whole(self, x, y, monotonic=False, scalar=False):
        """Whole(x, y): complement of x in y's unit direction.

        Vector form (default):
            whole(x, y) = (1 - x) * (y / ||y||)
        Scalar form: degree to which x contains y, i.e. part(y, x, scalar=True).
        """
        if scalar:
            return self.part(y, x, monotonic=monotonic, scalar=True)
        ny = self.norm(y).clamp(min=epsilon).unsqueeze(-1)
        return (1.0 - x) * (y / ny)

    def equal(self, x, y, monotonic=False, scalar=False):
        """Mutual parthood.

        Vector form:  equal(x, y) = part(x, y) * part(y, x)   (elementwise)
        Scalar form:  equal(x, y) = part(x, y, scalar=True) * part(y, x, scalar=True)
            partitions [0, 1] into three regions:
                equal == 0       -> underlap (disjoint)
                0 < equal < 1    -> overlap  (strictly partial)
                equal == 1       -> identity (perfect mutual parthood)
        """
        p_xy = self.part(x, y, monotonic=monotonic, scalar=scalar)
        p_yx = self.part(y, x, monotonic=monotonic, scalar=scalar)
        return p_xy * p_yx

    def overlap(self, x, y, monotonic=False, scalar=False):
        """Overlap.

        Vector form:  elementwise min of part(x, y) and part(y, x) -- the
            shared-parthood coordinates.
        Scalar form:  boolean region indicator 0 < equal(..., scalar=True) < 1.
        """
        if scalar:
            e = self.equal(x, y, monotonic=monotonic, scalar=True)
            return (e > 0) & (e < 1)
        return torch.minimum(
            self.part(x, y, monotonic=monotonic),
            self.part(y, x, monotonic=monotonic),
        )

    def underlap(self, x, y, monotonic=False, scalar=False):
        """Underlap.

        Vector form:  elementwise min of whole(x, y) and whole(y, x) -- the
            shared-wholeness coordinates (mutual complement).
        Scalar form:  boolean region indicator equal(..., scalar=True) == 0.
        """
        if scalar:
            e = self.equal(x, y, monotonic=monotonic, scalar=True)
            return e == 0
        return torch.minimum(
            self.whole(x, y, monotonic=monotonic),
            self.whole(y, x, monotonic=monotonic),
        )

    def boundary(self, x, y, monotonic=False, scalar=False):
        """Boundary: directional asymmetry of parthood.

        Vector form:  |part(x, y) - part(y, x)|   (elementwise)
        Scalar form:  |part(x, y, scalar=True) - part(y, x, scalar=True)|
            Zero under clipped-cosine parthood (cosine is symmetric).
        """
        m = monotonic
        return torch.abs(
            self.part(x, y, monotonic=m, scalar=scalar)
            - self.part(y, x, monotonic=m, scalar=scalar)
        )

    def copart(self, x, y, monotonic=False, scalar=False):
        """Copart of x under y: the part of y not accounted for by x.

        Vector form (default):
            copart(x, y) = y - x
        Scalar form (scalar=True): complement of parthood in [0, 1].
            copart(x, y) = 1 - part(x, y, scalar=True)
        """
        if scalar:
            return (1.0 - self.part(x, y, monotonic=monotonic, scalar=True)).clamp(0.0, 1.0)
        return y - x

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
        # nn.Module.__setattr__ refuses to replace a registered Parameter
        # with a plain tensor (torch >=2.11). W flip-flops between the two
        # across lifecycle calls; clear the slot first so either is legal.
        if "W" in self._parameters:
            del self._parameters["W"]
        self.W = value

    def forward(self, x):
        self.setW(x)
        return x

    def reverse(self, y, **kwargs):
        self.setW(y)
        return y


class Codebook(Basis):
    """Prototype basis with vector quantization and reverse snapping support."""

    class _RotationTrickFn(torch.autograd.Function):
        """Rotation-trick gradient estimator for VQ codebooks.

        Forward returns ``q``. Backward rotates the upstream gradient from
        ``q``'s direction back to ``e``'s direction (per row, via the
        Householder reflection that maps ``hat(q) -> hat(e)``) and scales
        by ``||q|| / ||e||`` so magnitude information is preserved. Only
        ``e`` receives gradient; ``q`` does not (the codebook is trained
        through the separate commitment loss and EMA updates).
        """

        @staticmethod
        def forward(ctx, e, q):
            ctx.save_for_backward(e.detach(), q.detach())
            return q

        @staticmethod
        def backward(ctx, grad_q):
            e, q = ctx.saved_tensors
            eps = 1e-8
            flat_shape = (-1, e.shape[-1])
            e_flat = e.reshape(flat_shape)
            q_flat = q.reshape(flat_shape)
            g_flat = grad_q.reshape(flat_shape)

            e_norm = e_flat.norm(dim=-1, keepdim=True).clamp(min=eps)
            q_norm = q_flat.norm(dim=-1, keepdim=True).clamp(min=eps)
            e_hat = e_flat / e_norm
            q_hat = q_flat / q_norm

            # Householder reflection with v = q_hat - e_hat maps q_hat -> e_hat.
            v = q_hat - e_hat
            v_dot_v = (v * v).sum(dim=-1, keepdim=True).clamp(min=eps)
            v_dot_g = (v * g_flat).sum(dim=-1, keepdim=True)
            reflected = g_flat - 2.0 * v_dot_g / v_dot_v * v
            scaled = reflected * (q_norm / e_norm)
            grad_e = scaled.reshape(grad_q.shape)
            return grad_e, None

    def __init__(self):
        super().__init__()
        self.W = None
        # Transient per-forward activation payload. Held separately from
        # self.W so the codebook Parameter (self.W once ``addVectors`` has
        # run) is never overwritten -- activations must not end up in
        # state_dict.
        self._active_payload = None
        self.customVQ = True
        self.snapDistance = 0.1
        self.eta = 0.9
        self.alpha = 0.0
        self.codebookSize = 0
        self.vq = None
        # Latest commitment loss from the most recent forward/quantize pass.
        # Consumer sites (SymbolicSpace._symbol_objective_terms,
        # ConceptualSpace.forward, etc.) read and clear this between steps.
        self.last_commit_loss = None

    def getW(self):
        # While an activation payload is cached, callers see it; otherwise
        # they see the codebook Parameter.
        if self._active_payload is not None:
            return self._active_payload
        return self.W

    def setW(self, value):
        """Assign W without ever clobbering the codebook Parameter.

        The ``event`` slot on a SubSpace is dual-purpose: at build time it
        receives the VQ codebook (``nn.Parameter`` via ``addVectors``) and
        at forward time it receives transient muxed activations. If we
        blindly let activations replace the Parameter, they leak into
        ``state_dict()`` and the checkpoint ends up recording a
        batch-shaped tensor instead of the codebook. Route non-Parameter
        writes to ``_active_payload`` so the codebook stays put.

        ``value=None`` clears only the transient activation; the codebook
        Parameter (if any) is preserved.
        """
        if value is None:
            self._active_payload = None
            return
        if isinstance(value, nn.Parameter):
            # Register (or re-register) the codebook Parameter. Clear any
            # stale activation so subsequent getW() returns the codebook.
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = value
            self._active_payload = None
            return
        # Plain tensor -- an activation. If a codebook is already held,
        # cache the activation without disturbing the Parameter. If there
        # is no codebook yet (e.g. non-customVQ init passes a plain
        # tensor), fall back to the legacy slot so callers still see it.
        if "W" in self._parameters:
            self._active_payload = value
            return
        self.W = value
        self._active_payload = None

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
            # Initialize codebook entries in [-1, 1] so downstream range
            # checks on the concepts/symbols 'what' field pass on the
            # first forward pass. The prior external package did its own
            # initialization; the in-repo fallback uses raw torch.randn.
            with torch.no_grad():
                init = self.vq.codebook.detach()
                init = init / init.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                self.vq.codebook = init
            self.setW(self.vq.codebook)
        else:
            W = torch.randn([nVec, self.nDim], device=TheDevice.get())
            for i in range(nVec):
                W[i, :] = self.normalize(W[i, :]).squeeze(0)
            self.setW(W)
        return self.getW()

    def quantize(self, x):
        if self.passThrough:
            zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            self.last_commit_loss = zero
            return x, None, zero
        if self.customVQ:
            quantized, indices, commit_loss = self.vq(
                x,
                ema_update_weight=self.updateWeights,
            )
            self.setW(self.vq.codebook)
            self.last_commit_loss = commit_loss
            return quantized, indices, commit_loss
        weight = self._prototype_weight(context="quantize")
        flat = x.reshape(-1, x.shape[-1])
        dists = flat[:, :self.nDim] @ weight[:, :self.nDim].T / max(self.nDim, 1)
        indices = dists.argmax(dim=-1)
        quantized = weight[indices]
        quantized = quantized.reshape(*x.shape[:-1], weight.shape[-1])
        indices = indices.reshape(x.shape[:-1])
        loss = self.commit_loss(x, quantized)
        self.last_commit_loss = loss
        return quantized, indices, loss

    @staticmethod
    def apply_gradient_estimator(e, q, mode="snap"):
        """Combine encoder output ``e`` and quantized output ``q`` into a
        forward signal carrying a chosen gradient estimator.

        Modes:
          ``snap``      -- forward ``q``, zero gradient back to ``e``.
          ``ste``       -- forward ``q``, backward identity to ``e`` (the
                           straight-through estimator).
          ``rotation``  -- forward ``q``, backward rotates the upstream
                           gradient from ``q``'s direction back to ``e``'s
                           direction and scales by ``||q|| / ||e||``.

        Always returns a tensor shaped like ``q``.
        """
        if mode == "snap":
            return q.detach()
        if mode == "ste":
            return e + (q - e).detach()
        if mode == "rotation":
            return Codebook._RotationTrickFn.apply(e, q)
        raise ValueError(
            f"Codebook.apply_gradient_estimator: unknown mode {mode!r}; "
            "expected one of 'snap', 'ste', 'rotation'.")

    def commit_loss(self, e, q):
        """Commitment loss (the encoder side of the VQ-VAE objective).

        Returns ``commitment_weight * MSE(e, sg[q])`` where the quantized
        codes are detached so the gradient flows only into ``e``. If the
        codebook is in ``passThrough`` mode or hasn't been initialized,
        returns a zero scalar on ``e``'s device/dtype.
        """
        beta = 1.0
        if self.customVQ and self.vq is not None:
            beta = float(getattr(self.vq, "commitment_weight", 1.0))
        if self.passThrough or e.numel() == 0 or q.numel() == 0:
            return e.new_tensor(0.0)
        n = min(e.shape[-1], q.shape[-1])
        if n <= 0:
            return e.new_tensor(0.0)
        return beta * F.mse_loss(e[..., :n], q[..., :n].detach())

    # -- Per-entry freezing (Task 3: ColumnUsageTracker analogue) ------
    # When an entry's sigma (running stdev of gradient norm across a window)
    # falls below ``freeze_threshold``, the entry has converged and we
    # zero its gradient from here on. Distinct from ColumnUsageTracker:
    # operates on codebook rows and uses sigma (ergodic measure) rather than
    # raw grad-norm mean.
    def _ensure_freezing_buffers(self, n):
        if getattr(self, "_grad_norm_history", None) is None:
            self._grad_norm_history = []
        # Pin freezing state to CPU so device swaps on the codebook weight
        # don't split the state across devices.
        if getattr(self, "usage_sigma", None) is None or self.usage_sigma.shape[0] != n:
            self.usage_sigma = torch.zeros(n, device="cpu")
        if getattr(self, "frozen_entries", None) is None or self.frozen_entries.shape[0] != n:
            self.frozen_entries = torch.zeros(n, dtype=torch.bool, device="cpu")

    def _record_codebook_grad(self, grad):
        # grad shape: (nVectors, nDim)
        per_entry = grad.detach().norm(dim=-1).cpu()
        n = per_entry.shape[0]
        self._ensure_freezing_buffers(n)
        self._grad_norm_history.append(per_entry)
        if len(self._grad_norm_history) > int(getattr(self, "freeze_window", 10)):
            self._grad_norm_history.pop(0)
        if len(self._grad_norm_history) >= 2:
            stacked = torch.stack(self._grad_norm_history, dim=0)
            # sigma = std over the window; "consistent grad-norm" => small sigma.
            self.usage_sigma = stacked.std(dim=0, unbiased=False)
        # Zero the gradient rows of already-frozen entries so the
        # optimizer can't drift them.
        if self.frozen_entries.any():
            mask = self.frozen_entries.to(device=grad.device)
            grad = grad.clone()
            grad[mask] = 0.0
        return grad

    def attach_freeze_hook(self, threshold=0.01, window=10):
        """Register a backward hook on the codebook weight that records
        per-entry gradient norms and honors the frozen mask. Idempotent.
        """
        self.freeze_threshold = float(threshold)
        self.freeze_window = int(window)
        w = self.getW()
        if w is None or not isinstance(w, nn.Parameter):
            return False
        if getattr(self, "_freeze_hook_handle", None) is not None:
            return True
        self._ensure_freezing_buffers(w.shape[0])
        self._freeze_hook_handle = w.register_hook(self._record_codebook_grad)
        return True

    def freeze_well_learned(self, threshold=None):
        """Mark entries whose sigma has dropped below ``threshold`` as frozen.

        Returns the number of newly-frozen entries. Call after each
        optimizer step (or periodically) so the hook can zero their
        gradients on subsequent backward passes.
        """
        if threshold is None:
            threshold = float(getattr(self, "freeze_threshold", 0.01))
        if getattr(self, "usage_sigma", None) is None:
            return 0
        to_freeze = (self.usage_sigma < threshold)
        already = self.frozen_entries
        new_mask = to_freeze & (~already)
        self.frozen_entries = already | to_freeze
        return int(new_mask.sum().item())

    def forward(self, input, topK: int = 0):
        """Codebook forward. When ``topK > 0`` and less than the codebook
        size, ``self.activation`` is pruned to the top-K strongest entries
        per batch row -- realizing the wide-codebook narrow-output pattern
        where nVectors >> nOutput. ``topK=0`` preserves legacy behavior.
        """
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
        if topK and 0 < topK < act.shape[-1]:
            act = topk_by_magnitude_per_batch(act, k=topK)
        self.activation = act
        self.activeSigma = None
        self.setW(x)
        if _vspace is not None:
            _vspace.set_event(x, compute_activation=False)
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
        """Embedding W is managed by wv._vectors -- setW is a no-op."""
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
        path is None, starts with an empty vocabulary -- new words are
        added dynamically during forward passes via ``insert()``.

        If *source* is provided, build per-document lexer token streams and
        ensure every token text has an embedding row.
        """
        if wv is None:
            wv = self._load_embeddings(embedding_path=embedding_path, nDim=nDim)
        if wv is None:
            # No matching embeddings on disk -- start with \x00 at index 0.
            # This is both the EOS/padding sentinel AND the first real
            # byte, which makes byte_value == codebook_index throughout
            # the full 0..255 range.  ChunkLayer.BOUNDARY_BYTES assumes
            # this alignment (see code review).  Real words are added
            # dynamically during forward passes via insert().
            dim = nDim or 20
            print(f"Starting with dynamic {dim}-dim embedding (words added at runtime)")
            placeholder = torch.randn(1, dim, device=TheDevice.get())
            placeholder = F.normalize(placeholder, p=2, dim=1)
            wv = WordVectors(placeholder, ["\x00"])
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

        # Bootstrap codebook with ASCII (and full 0-255 in byte mode).
        # \x00 is already at index 0 from the placeholder row above; fill
        # chr(1)..chr(upper-1) at indices 1..upper-1 so byte_value aligns
        # with codebook_index.  ChunkLayer.BOUNDARY_BYTES (space, tab, LF,
        # CR, NUL) depends on this alignment to detect word boundaries.
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

        method: 'CBOW' -- predict each word from padded leave-one-out context
                'SBOW' -- predict each word from leave-one-out centroid (faster)
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

        Does NOT strip encoding overhead -- the full vector (nWhat + nWhere + nWhen)
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

        # Resolved nInputDim/nOutputDim (0 -> constructor dim, -1 -> skip, >0 -> explicit)
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
        # activation: [B, N] strength gate -- materialize() = event * activation.
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
        """Compute modal presence flags and 4-valued (quaternary) activation.

        active[b, n, m] = 1 if modality m is nonzero at position n, else 0.
        Activation is the tetralemma bivector [aP, aN]:
            aP = ||relu(what)|| / sqrt(nWhat)    (positive content)
            aN = ||relu(-what)|| / sqrt(nWhat)   (negative content)
        See basicmodel/doc/BuddhistParallels.md for the catuskoti mapping.

        Args:
            what_tensor: [B, N, nWhat]
            where_tensor: [B, N, nWhere] or None
            when_tensor: [B, N, nWhen] or None
        """
        flags = []
        # what is always present as a modality
        flags.append((what_tensor.norm(dim=-1) > 1e-8).float())  # [B, N]
        if where_tensor is not None and self.nWhere > 0:
            flags.append((where_tensor.norm(dim=-1) > 1e-8).float())
        if when_tensor is not None and self.nWhen > 0:
            flags.append((when_tensor.norm(dim=-1) > 1e-8).float())
        self._active = torch.stack(flags, dim=-1)  # [B, N, M]
        d = max(what_tensor.shape[-1], 1)
        pos = torch.relu(what_tensor).norm(dim=-1) / math.sqrt(d)
        neg = torch.relu(-what_tensor).norm(dim=-1) / math.sqrt(d)
        act = torch.stack([pos.clamp(0.0, 1.0), neg.clamp(0.0, 1.0)], dim=-1)
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

        Sets activation to all-ones by default -- activation should be set
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

    def demux(self, muxed):
        """Split a muxed [B, N, D] tensor along the canonical [what|where|when]
        column layout and store each block in the corresponding modality slot.

        Used at the C -> S tier boundary (Rule #2): C stays muxed as exploratory
        soup; S-tier commitment requires axis-separated what/where/when blocks
        so the grammar can make axis-restricted commitments (and so verbs can
        operate as prepositions when fed axis-restricted arguments).

        Args:
            muxed: [B, N, D] tensor where D = nWhat + nWhere + nWhen.

        The column split uses this SubSpace's configured widths; any block of
        zero width is skipped (so a config with nWhere=0, nWhen=0 still works).
        """
        nWhat = self.nWhat
        nWhere = self.nWhere
        if nWhat > 0:
            self.set_what(muxed[..., :nWhat])
        if nWhere > 0:
            self.set_where(muxed[..., nWhat:nWhat + nWhere])
        if self.nWhen > 0:
            self.set_when(muxed[..., nWhat + nWhere:])

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
        the layer simply maps per-vector from nInputDim -> nOutputDim.

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

    # forwardBegin/End, reverseBegin/End -- handled by Space via nInputDim/nOutputDim reshape.

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
        """Compute 4-valued activation and active flags from muxed event vectors.

        Activation is the tetralemma bivector [aP, aN] derived from the
        what-slice:
            aP = ||relu(what)|| / sqrt(nWhat)
            aN = ||relu(-what)|| / sqrt(nWhat)
        Active flags are derived by checking each modality slice for
        nonzero content.
        """
        y = self.event.getW()
        assert y is not None and y.ndim == 3, "Must be dim==3"
        what_slice = y[:, :, :self.nWhat]
        d = max(self.nWhat, 1)
        pos = torch.relu(what_slice).norm(dim=-1) / math.sqrt(d)
        neg = torch.relu(-what_slice).norm(dim=-1) / math.sqrt(d)
        act = torch.stack([pos.clamp(0.0, 1.0), neg.clamp(0.0, 1.0)], dim=-1)
        self.set_activation(act)
        # Derive active flags from modality slices
        flags = []
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
        """Store 4-valued (quaternary) activation for each object vector.

        Accepts either the full tetralemma bivector [B, N, 2] = [aP, aN]
        or a legacy scalar [B, N] which is lifted to the bivector:
            x > 0  -> [x, 0]   (positive truth)
            x < 0  -> [0, -x]  (negative truth)
            x == 0 -> [0, 0]   (NEITHER)

        Args:
            activation_tensor: [B, N, 2] bivector or [B, N] legacy scalar.
        """
        nd = self.activeEncoding.nDim
        if activation_tensor.ndim == 2:
            if nd == 2:
                pos = torch.relu(activation_tensor)
                neg = torch.relu(-activation_tensor)
                activation_tensor = torch.stack([pos, neg], dim=-1)
            # nd == 1 path: leave as [B, N]
        elif activation_tensor.ndim == 3:
            assert activation_tensor.shape[-1] == nd, (
                f"activation last dim must be {nd}, got {activation_tensor.shape}"
            )
        else:
            raise AssertionError(
                f"activation must be [B, N] or [B, N, {nd}], got {activation_tensor.shape}"
            )
        self.activation.setW(activation_tensor)

    def get_activation(self):
        """Return stored activation [B, N, nDim] or None."""
        if self.activation is None:
            return None
        return self.activation.getW()

    def activation_presence(self):
        """Scalar presence gate reduced from the activation tensor.

        For the 4-valued bivector [aP, aN], presence = max(aP, aN) --
        either pole being lit means the position carries information
        (TRUE, FALSE, or BOTH all count as present; only NEITHER=[0,0]
        gates the position off).
        Returns [B, N] or None.
        """
        act = self.get_activation()
        if act is None:
            return None
        if act.ndim == 3:
            return act.max(dim=-1).values
        return act

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
            [B, N] scalar tensor. A position is active only if all its
            modalities are present AND its event activation is positive.
            Scalar form comes from activation_presence() which reduces the
            4-valued bivector via max(aP, aN). See BuddhistParallels.md
            for the tetralemma mapping.
        """
        pres = self.activation_presence()
        if pres is None:
            return None
        if self._active is not None:
            modal_gate = self._active.prod(dim=-1)  # [B, N]
            return pres * modal_gate
        return pres

    def dematerialize(self):
        """Split event -> modalities, recover active flags.

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

    # -- Stack-scanning helpers ----------------------------------------

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
            list of int -- top-of-stack position for each batch element,
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
            list of (pos1, pos2) tuples -- pos1 is second-to-top, pos2 is top.
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
            # [V, D] codebook -- index with [B, N] -> [B, N, D]
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
            pres = self.activation_presence()
            if pres is not None:
                return pres
            self.set_activation_from_event()
            return self.activation_presence()

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
                            # Codebook [V, D] -- index lookup
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

        # Apply activation gate: materialize() = event * activation_presence.
        # The 4-valued bivector is reduced to a scalar presence (max of
        # poles) for event gating; see BuddhistParallels.md for the
        # tetralemma mapping.
        pres = self.activation_presence()
        if pres is not None:
            x = x * pres.unsqueeze(-1)

        # top-k selection if requested
        if k is not None and k < x.shape[-2]:
            score = pres if pres is not None else x.norm(dim=-1)
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
            target: one of Encoding.TARGETS -- "activation", "what", "where",
                    "when", "event", or "all".

        Returns:
            Tensor for the requested encoding, or a dict of {target: tensor}
            when target="all" (skipping empty encodings).
        """
        assert target in Encoding.TARGETS, f"Unknown target {target!r}, expected one of {Encoding.TARGETS}"

        if target == "activation":
            # Return the stored bivector as-is (4-valued quaternary truth).
            # For a gated scalar gate, callers should use effective_activation()
            # or materialize(mode="activation").
            act = self.get_activation()
            if act is not None:
                return act
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

    def normalize(self, kind, target="activation", normalize=False,
                  reverse=False):
        """Assert range or apply normalization to an encoding of this subspace.

        When normalize=False (default): checks that values are finite and
        in the expected range for the space kind.  In ergodic mode,
        violations emit a warning; otherwise they raise AssertionError.

        When normalize=True: applies the normalizing transform in-place
        (e.g. tanh for percepts/concepts, STE for symbols).

        When reverse=True with normalize=True: applies the inverse transform.

        Range contracts:
          - "percepts", "concepts": elements in [-1, 1]
          - "symbols": elements in [0, 1]
          - "input": elements in [-1, 1]

        Args:
            kind: "percepts", "concepts", "symbols", or "input".
            target: encoding target -- "activation", "what", "where",
                    "when", "event", or "all".
            normalize: if True, apply the normalization in-place.
                If False, check range only.
            reverse: if True, apply the inverse normalizing transform.
                Requires normalize=True.
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
        if reverse and not normalize:
            raise ValueError("reverse=True requires normalize=True")

        # -- Range check (normalize=False) ----------------------------
        if not normalize:
            strict = not TheXMLConfig.get("architecture.ergodic")
            xd = x.detach()
            # Non-finite check
            finite_ok = torch.isfinite(xd).all()
            if not finite_ok:
                msg = (f"Non-finite values in kind={kind!r}, target={target!r}: "
                       f"{int((~torch.isfinite(xd)).sum().item())}/{xd.numel()} "
                       f"entries are nan/inf.")
                if strict:
                    assert False, msg
                else:
                    warnings.warn(msg)
                return
            # Range check -- vectors are always [-1, 1]; symbol activations are [0, 1]
            is_vector = target in ("what", "where", "when", "event")
            if kind == "symbols" and not is_vector:
                lo, hi = 0, 1
            else:
                lo, hi = -1, 1
            xmin, xmax = xd.min().item(), xd.max().item()
            if xmin < lo - 1e-2 or xmax > hi + 1e-2:
                msg = (f"Range violation: kind={kind!r}, target={target!r} "
                       f"range [{xmin:.6f}, {xmax:.6f}] outside [{lo}, {hi}].")
                if strict:
                    assert False, msg
                else:
                    warnings.warn(msg)
            return

        # -- Apply normalization (normalize=True) ---------------------
        if reverse:
            normalized = self._apply_reverse_normalization(kind, x, target=target)
        else:
            normalized = self._apply_normalization(kind, x, target=target)
        self.put(target, normalized)

    def _apply_normalization(self, kind, x, target="activation"):
        """Apply normalization function to tensor x.

        The combination of kind and target determines the geometry:
          - Perceptual vectors and activations use clamped tanh -> [-(1-eps), 1-eps].
          - Conceptual vectors and activations use clamped tanh -> [-(1-eps), 1-eps].
          - Activations use scalar transfer functions (tanh/STE).

        Forward and reverse are made symmetric by clamping both sides
        into the closed interval ``[-(1-eps), 1-eps]``.  Any bijection
        between R and a closed sub-interval of (-1, 1) must fail at the
        boundary (``atanh`` diverges at +-1), so we accept a tiny
        roundtrip error in the saturated tail in exchange for the same
        bounded domain on both paths -- this lets reverse absorb
        codebook/STE outputs that can sit exactly at +-1 without
        special-casing.
        """
        is_vector = target in ("what", "where", "when", "event")
        if kind == "percepts":
            return torch.tanh(x).clamp(min=-1.0 + epsilon, max=1.0 - epsilon)
        elif kind == "concepts":
            return torch.tanh(x).clamp(min=-1.0 + epsilon, max=1.0 - epsilon)
        elif kind == "symbols":
            soft = torch.sigmoid(x)
            hard = torch.round(soft)
            return hard - soft.detach() + soft  # straight-through estimator
        elif kind == "input":
            if is_vector:
                return TheData.normalize(x, which="input")  # [-1,1] via global min-max
            return x
        else:
            raise ValueError(f"Unknown normalization kind: {kind!r}")

    def _apply_reverse_normalization(self, kind, x, target="activation"):
        """Apply the (approximate) inverse of ``_apply_normalization``.

        Forward is ``y = clamp(tanh(x), -(1-eps), 1-eps)``; reverse is
        ``x = atanh(clamp(y, -(1-eps), 1-eps))``.  The clamp is
        symmetric so that codebook/STE values sitting exactly at +-1
        round-trip through ``atanh`` without diverging.  For non-
        saturated ``x``, the roundtrip is exact to ``atanh`` precision.
        """
        is_vector = target in ("what", "where", "when", "event")
        if kind in ("percepts", "concepts"):
            z = x.clamp(min=-1.0 + epsilon, max=1.0 - epsilon)
            return torch.atanh(z)
        elif kind == "input" and is_vector:
            return TheData.denormalize(x, which="input")
        elif kind == "symbols":
            raise RuntimeError("Cannot reverse-normalize symbols")
        else:
            raise ValueError(f"Unknown reversible normalization kind: {kind!r}")

    def denormalize(self, kind, target="activation"):
        """Reverse the normalization applied by normalize().

        Only meaningful for kinds with invertible transforms:
          - "input" on vectors -> scale from [-1,1] back to [input_min, input_max]
          - "output" on vectors -> scale from [output_min, output_max] to [-1,1]

        Invertible transforms:
          - "percepts" -> atanh (inverse tanh): [-1,1] -> R
          - "concepts" -> atanh (inverse tanh): [-1,1] -> R
          - "input" on vectors -> scale from [-1,1] back to [input_min, input_max]
          - "output" on vectors -> scale from [output_min, output_max] to [-1,1]
          - "symbols" (STE round) is not invertible and is skipped.
        """
        x = self.select(target)
        if x is None or x.numel() == 0:
            return
        is_vector = target in ("what", "where", "when", "event")
        if kind in ("percepts", "concepts", "input"):
            self.normalize(kind, target=target, normalize=True, reverse=True)
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
            reduce: "batch" -> mean over objects -> [B] (default),
                    "vector" -> per-object -> [B, N].

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
        agreement is zero -- the representations contradict.

        The MSE of the agreement vector gives the luminosity of truth
        between the two signals.

        Args:
            x1, x2: tensors to compare (same shape).
            target: encoding target (for documentation; tensors are
                    passed explicitly).
            reduce: "batch" -> [B], "vector" -> [B, N].

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
class WordSubSpace(SubSpace):
    """Fixed-size word-stream stack -- a SubSpace subclass with stack discipline.

    Stores a sequence of derivation-step rows in the inherited
    ``[what | where | when]`` Basis-backed column blocks of shape
    ``[batch, max_depth, <modality_dim>]``. Each rule application emits
    a ``(1 + max_arity)``-row block in prefix order:

        row 0:             rule-identity codebook vector
        rows 1..max_arity: leaf-identity codebook vectors

    Unused leaf slots produce zero-valued rows (empty-slot sentinel --
    consumers detect empties via ``row_what.norm() == 0``). Rows beyond
    top-of-stack are plain zeros. Rule-identity vectors come from a
    back-referenced ``SymbolicSpace.lookup_rule(rule_id)`` -- the one
    unified rule codebook -- so ``push()`` resolves identity to a dense
    vector at write time and ``read()`` returns a ready-to-concat
    ``[batch, max_depth, nDim]`` muxed tensor.

    **Why subclass ``SubSpace``**: the word stream's rows share the
    peer ``[what | where | when]`` column layout with
    ``PerceptualSubSpace`` / ``ConceptualSubSpace`` / ``SymbolicSubSpace``.
    Re-implementing ``set_what`` / ``set_where`` / ``set_when`` on top of
    a private buffer duplicated API surface while bypassing the ``Basis``
    storage contract other SubSpaces use. Inheriting from SubSpace (a)
    makes the peer-space claim structural (not duck-typed), (b) lets
    ``ConceptualSpace`` read all three inputs via one uniform
    ``materialize()`` call, and (c) puts all modality storage through
    one ``Basis`` abstraction.

    **Additive stack discipline**: ``push()``, ``clear()``, the
    per-batch ``_top`` pointer, and the ``_blocks`` parse-tree ledger
    are not part of the SubSpace contract -- they sit on top. A plain
    SubSpace is a snapshot; a WordSubSpace is a stack that happens to
    live inside the same Basis slots a snapshot would use.

    Parse-tree reconstruction walks the stack via ``get_blocks(b)``,
    which returns the original rule_id + leaves for each pushed block.
    """

    def __init__(self, nDim, nWhat, nWhere, nWhen,
                 max_depth, max_arity=3, batch=1):
        # Build encodings sized to our column widths. SubSpace reads
        # `whereEncoding.nDim` / `whenEncoding.nDim` to compute
        # `self.nWhere` / `self.nWhen`, and derives `self.nWhat` as
        # `outputShape[1] - nWhere - nWhen`.
        _muxed = int(nWhat) + int(nWhere) + int(nWhen)
        assert _muxed <= int(nDim), (
            f"WordSubSpace columns exceed nDim: "
            f"nWhat={nWhat} + nWhere={nWhere} + nWhen={nWhen} > nDim={nDim}")

        shape = [int(max_depth), _muxed]
        where_enc = WhereEncoding(
            max(1, int(max_depth) * max(1, int(batch))),
            int(nWhere), int(nWhen))
        when_enc = WhenEncoding(10000, int(nWhen))

        super().__init__(
            inputShape=shape,
            outputShape=shape,
            whereEncoding=where_enc,
            whenEncoding=when_enc,
        )

        # SubSpace.__init__ has now created Tensor()-backed Basis
        # objects for .what/.where/.when/.event/.activation and computed
        # self.nWhat / self.nWhere / self.nWhen / self.muxedSize from
        # outputShape + encodings.
        assert self.nWhat == int(nWhat) and self.nWhere == int(nWhere) \
            and self.nWhen == int(nWhen), (
            f"WordSubSpace column widths disagree with SubSpace post-init: "
            f"SubSpace derived nWhat={self.nWhat} nWhere={self.nWhere} "
            f"nWhen={self.nWhen}, expected nWhat={nWhat} nWhere={nWhere} "
            f"nWhen={nWhen}"
        )

        # Preserve the peer-compatible total `nDim` attribute. SubSpace
        # does not set `self.nDim`; peer spaces expose it, and
        # ``WordSpace.__init__`` reads it when sizing the word buffer, so
        # we keep it.
        self.nDim = int(nDim)
        self.max_depth = int(max_depth)
        self.max_arity = int(max_arity)
        self.block_size = 1 + self.max_arity  # rule row + leaf rows

        # Allocate actual [batch, max_depth, <modality>] storage into
        # the inherited Basis slots. This replaces the old standalone
        # `self._buffer` -- storage flows through Basis.setW() now.
        self._allocate_storage(int(batch))

        # Stack-discipline metadata (not part of SubSpace contract).
        # Uses register_buffer so the top pointer follows `.to(device)`.
        self.register_buffer(
            "_top", torch.zeros(self.batch, dtype=torch.long),
            persistent=False)
        self._blocks = [[] for _ in range(self.batch)]

        # Back-reference to the host providing lookup_rule(rule_id).
        # Stored via `object.__setattr__` so the host (typically a
        # `SymbolicSpace` nn.Module) is NOT registered as an nn.Module
        # child -- that would create a parent/child cycle
        # (`symbolicSpace <-> wordSpace.subspace`) and make
        # `model.to(device)` recurse forever. Set by
        # `attach_codebook_host()`; WordSpace wires this at construction.
        object.__setattr__(self, 'rule_codebook_host', None)

    # -- storage allocation via inherited Basis slots -----------------
    def _allocate_storage(self, batch):
        """Allocate zero tensors of shape ``[batch, max_depth, <modality>]``
        into the inherited ``.what``/``.where``/``.when`` Basis slots.

        Called from ``__init__`` and ``clear()`` / ``ensure_batch()``.
        Also primes ``.event``/``.activation`` so a ``materialize()`` call
        directly after allocation returns an all-zero tensor of the
        expected muxed shape.

        **Device**: tensors are allocated without an explicit ``device=``
        argument so ``torch.set_default_device(TheDevice.get())``
        (installed in ``util.py``) puts them on the process's canonical
        device. Forcing CPU here would leave Basis storage stranded when
        the model later moves to MPS / CUDA, breaking ``push()``'s
        in-place writes against ``host.lookup_rule()`` vectors that are
        already on the target device.
        """
        self.batch = int(batch)
        # Populate the demuxed modality slots through the Basis contract.
        if self.nWhat > 0:
            self.what.setW(
                torch.zeros(self.batch, self.max_depth, self.nWhat))
        if self.nWhere > 0:
            self.where.setW(
                torch.zeros(self.batch, self.max_depth, self.nWhere))
        if self.nWhen > 0:
            self.when.setW(
                torch.zeros(self.batch, self.max_depth, self.nWhen))
        # Prime the muxed event cache and activation gate so
        # `materialize()` returns a properly-shaped tensor even before
        # the first `push()`.
        muxed = torch.zeros(self.batch, self.max_depth, self.muxedSize)
        self.event.setW(muxed)
        self._demuxed = True
        # All-ones activation -- WordSubSpace has no gating beyond the
        # zero rows themselves, and multiplying by ones is cheap.
        self.set_activation(
            torch.ones(self.batch, self.max_depth))

    # -- device-propagation override ----------------------------------
    def _apply(self, fn, recurse=True):
        """Propagate tensor-moving operations through inherited Basis
        storage.

        ``SubSpace``'s ``.what``/``.where``/``.when``/``.event`` /
        ``.activation`` slots are backed by a ``Tensor`` Basis whose
        internal ``W`` attribute is a plain Python reference, NOT a
        registered buffer. As a result, vanilla ``nn.Module._apply``
        never touches it, so ``model.to(device)`` on a SubSpace silently
        leaves the Basis W tensors on their original device. Regular
        SubSpace usage avoids this by always re-setting ``.W`` from a
        fresh device-correct tensor in every forward pass -- but
        ``WordSubSpace`` allocates its whole buffer eagerly in
        ``__init__`` and mutates it in place, so we need the Basis
        tensors to actually follow the module to the target device.

        This override walks each Basis slot and applies ``fn`` to its
        ``W``, then calls the parent implementation to handle
        registered buffers / parameters / submodules in the usual way.
        """
        for basis_attr in ('what', 'where', 'when', 'event', 'activation'):
            basis = getattr(self, basis_attr, None)
            if basis is None:
                continue
            w = basis.getW() if hasattr(basis, 'getW') else None
            if isinstance(w, torch.Tensor):
                basis.setW(fn(w))
        return super()._apply(fn, recurse=recurse)

    # -- codebook + batch lifecycle -----------------------------------
    def attach_codebook_host(self, host):
        """Register an object exposing ``lookup_rule(rule_id)``.

        The returned vector is embedded into the row's ``.what`` block
        at push time, matching ``SymbolicSpace``'s codebook-on-``.what``
        convention so peer spaces consume a uniform dense stream.

        Stored via ``object.__setattr__`` -- see the ``__init__``
        back-reference comment for the nn.Module cycle rationale.
        """
        object.__setattr__(self, 'rule_codebook_host', host)

    def ensure_batch(self, batch):
        """Resize the buffer to ``batch`` (zeros the contents)."""
        if int(batch) == self.batch:
            return
        self._allocate_storage(int(batch))
        self._top = torch.zeros(
            self.batch, dtype=torch.long, device=self._top.device)
        self._blocks = [[] for _ in range(self.batch)]

    def clear(self):
        """Reset buffer to all-zero and rewind top-of-stack (per-sentence)."""
        self._allocate_storage(self.batch)
        if self._top is not None:
            self._top.zero_()
        self._blocks = [[] for _ in range(self.batch)]

    # -- push / read -------------------------------------------------
    def _lookup(self, rule_id):
        """Return a ``[nWhat]`` dense vector for a rule_id, or ``None``."""
        if self.rule_codebook_host is None or rule_id is None:
            return None
        if rule_id < 0:
            return None
        host = self.rule_codebook_host
        vec = None
        if hasattr(host, 'lookup_rule'):
            vec = host.lookup_rule(rule_id)
        if vec is None:
            return None
        # Trim/pad to nWhat.
        if vec.shape[-1] >= self.nWhat:
            return vec[:self.nWhat]
        pad = torch.zeros(self.nWhat - vec.shape[-1], device=vec.device)
        return torch.cat([vec, pad], dim=-1)

    def push(self, b, rule_id, leaves):
        """Append a ``(1 + max_arity)``-row block to batch row ``b``.

        Row 0 holds the rule-identity vector in its ``.what`` block;
        rows 1..max_arity hold leaf-identity vectors in prefix (operand)
        order. Unused leaf slots produce zero rows. When peer
        ``nWhere > 0``, each row of the block gets a monotonic
        derivation-step index written into its ``.where`` block.
        Overflow beyond ``max_depth`` silently drops the push.

        Writes go through the inherited ``Basis.setW()`` /
        ``Basis.getW()`` contract -- ``self.what.getW()`` returns a
        mutable reference to the ``[batch, max_depth, nWhat]`` tensor
        allocated by ``_allocate_storage``, and ``push()`` mutates that
        in place. After mutation we invalidate the cached ``.event``
        tensor so the next ``materialize()`` rebuilds the muxed view.

        Args:
            b: batch index.
            rule_id: 0-based grammar rule id. Must be a valid index into
                the unified symbolic codebook's rule sub-range.
            leaves: iterable of rule_ids for the operand slots; entries
                with value ``None`` or ``< 0`` are treated as empty.
                Iterable is zero-padded / truncated to exactly
                ``max_arity``.
        """
        if self.rule_codebook_host is None:
            return
        if b < 0 or b >= self.batch:
            return
        top = int(self._top[b].item())
        if top + self.block_size > self.max_depth:
            return  # buffer full -- silently drop

        # Normalize leaves to a fixed-length tuple.
        leaves_list = list(leaves) if leaves is not None else []
        while len(leaves_list) < self.max_arity:
            leaves_list.append(-1)
        leaves_list = [int(x) if x is not None else -1
                       for x in leaves_list[:self.max_arity]]

        step_idx = len(self._blocks[b])  # monotonic derivation step
        self._blocks[b].append({
            'start': top,
            'rule_id': int(rule_id),
            'leaves': tuple(leaves_list),
        })

        what_W = self.what.getW() if self.nWhat > 0 else None
        where_W = self.where.getW() if self.nWhere > 0 else None

        # Row 0: rule identity vector in .what block.
        rule_vec = self._lookup(int(rule_id))
        if what_W is not None and rule_vec is not None:
            what_W[b, top] = rule_vec.to(what_W.device)
        if where_W is not None:
            where_W[b, top] = float(step_idx)

        # Rows 1..max_arity: leaf identity vectors in .what block.
        for k in range(self.max_arity):
            leaf_id = leaves_list[k]
            row = top + 1 + k
            if leaf_id < 0:
                continue  # empty-slot sentinel (zero row)
            leaf_vec = self._lookup(leaf_id)
            if what_W is not None and leaf_vec is not None:
                what_W[b, row] = leaf_vec.to(what_W.device)
            if where_W is not None:
                where_W[b, row] = float(step_idx)

        self._top[b] = top + self.block_size

        # Invalidate the cached muxed event -- next materialize()
        # rebuilds it from the mutated .what/.where/.when Basis slots.
        if self.event is not None:
            self.event.setW(None)

    def read(self):
        """Return the muxed ``[batch, max_depth, nDim]`` stack tensor.

        Delegates to ``SubSpace.materialize()`` which concatenates the
        demuxed ``.what``/``.where``/``.when`` Basis blocks into one
        muxed event tensor (and caches it on ``.event`` until the next
        ``push()`` invalidates the cache).
        """
        return self.materialize()

    def get_blocks(self, b):
        """Return the parse-tree ledger for batch row ``b``.

        Each entry is a dict with keys ``start`` (row index in the
        buffer), ``rule_id`` (the pushed rule id), and ``leaves``
        (tuple of ``max_arity`` operand rule ids; ``-1`` indicates an
        empty slot). Used by parse-tree reconstruction and
        invertibility tests.
        """
        if 0 <= b < len(self._blocks):
            return list(self._blocks[b])
        return []

    def top_of_stack(self, b=None):
        """Return top-of-stack row index per batch (or for one batch)."""
        if self._top is None:
            return 0 if b is not None else []
        if b is not None:
            return int(self._top[b].item())
        return self._top.tolist()


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
        self.spaceShape   = spaceShape   # [nVectors, nDim]  -- codebook / internal basis
        self.outputShape  = outputShape  # [nOutput,  nOutputDim]
        self.nVectors     = spaceShape[0]  # codebook size
        self.nDim         = spaceShape[1]  # content dimensionality of the codebook vectors
        # Resolve nInputDim/nOutputDim:
        #   0  -> inherit from constructor dim (inputShape[1] / outputShape[1])
        #  -1  -> flatten: nInput * dim (reshape [N, D] -> [1, N*D])
        #  >0  -> explicit value
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

        # wordSpace is still held as a non-Module pointer so the few
        # call sites that reach across to ``wordSpace.truth_layer``
        # (SymbolicSpace) keep working; composition dispatch is no
        # longer done here -- home spaces take ``wordSpace`` as a
        # per-call parameter and call ``wordSpace.forwardPercepts`` /
        # ``.forwardConcepts`` / ``.forwardSymbols`` (and the reverse
        # variants) explicitly.
        self.wordSpace = None
        self.params = []   # parameters for the optimizer (excludes temperature params)
        self.layers = nn.ModuleList()   # layer instances for paramUpdate() delegation
        self._register_requirements()

    def attach_wordSpace(self, wordSpace):
        """Wire the shared WordSpace as a non-Module routing pointer.

        The wordSpace reference is stored via ``object.__setattr__`` so
        the WordSpace nn.Module is NOT registered as a child of this
        Space -- that would create a ``space -> wordSpace -> space`` cycle
        (WordSpace already owns the SyntacticLayer and its codebook
        host is the SymbolicSpace) and make ``model.to(device)``
        recurse forever. The wordSpace is owned at the model level
        instead, with each Space holding only this non-Module pointer.
        Layer attachment is done directly via
        ``wordSpace.attach_layer(kind, layer)`` by the WordSpace
        factory methods, not by this helper.
        """
        object.__setattr__(self, 'wordSpace', wordSpace)

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
        """Register base-class config requirements.

        Codebook spaces sample with replacement, so ``nVectors`` can be
        smaller, equal, or larger than ``nActive`` -- any positive value
        is valid.  Non-codebook spaces still require ``nVectors == nActive``
        because they use a direct one-to-one vector store.
        """
        section_name = self.config_section
        nV = self.nVectors
        nA = self.outputShape[0]
        if not self.codebook:
            TheXMLConfig.require(
                lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv == _na,
                f"{section_name}: non-codebook space requires nVectors ({nV}) == nActive ({nA})"
            )

    def get_vectors(self):
        """Convenience accessor -- delegates to subspace."""
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

    # _2d/_3d removed -- all layers now operate on [..., D] natively.

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
        """Propagate exploration meta-parameters to all layers and Basis slots.

        ``self.subspace`` may be ``None`` (DiscourseSpace) or a non-SubSpace
        buffer without Basis objects (WordSpace's ``WordSubSpace``). In those
        cases we only walk ``self.layers`` -- no basis slots exist to update.
        """
        for l in self.layers:
            if hasattr(l, 'set_sigma'):
                l.set_sigma(sigma)
        sub = getattr(self, 'subspace', None)
        if sub is None:
            return
        if not all(hasattr(sub, attr) for attr in ('what', 'where', 'when', 'activation')):
            return
        for basis in (sub.what, sub.where, sub.when, sub.activation):
            if basis is not None and hasattr(basis, 'set_sigma'):
                basis.set_sigma(sigma)

    def getParameters(self):
        return self.params
    def paramUpdate(self):
        for l in self.layers:
            if hasattr(l, 'paramUpdate'):
                l.paramUpdate()
class InputSpace(Space):
    """Receives the source buffer from Data() and encodes it as vectors.

    For text: delegates tokenization to Lex, which produces a span table
    (start, end, type) over the source buffer.  Each span is encoded as a
    vector with nWhat (token content via the active ``Basis`` / ``Codebook``)
    and nWhere
    (positional encoding from the span's start offset).  A whole sentence
    is sent at once as a batch of [nWhat + nWhere] vectors.

    For numeric data: the tensor path is unchanged -- no span table, no Lex,
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

    def _register_requirements(self):
        """InputSpace vocabularies are inherently sampling-with-replacement.

        The ``<codebook>`` XML flag in InputSpace controls downstream
        quantization behaviour, *not* whether the vocabulary allows a
        vocab-size (``nVectors``) independent from the buffer size
        (``nActive`` / ``nOutput``).  A byte-mode InputSpace with
        ``nVectors=256`` and ``nOutput=32`` is perfectly valid: every
        one of the 32 output positions draws (with replacement) from
        the 256-entry vocabulary.  So we impose no cross-constraint
        here.
        """
        pass

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
        # Byte lexer inputs are discrete indices (0-255) looked up via
        # Embedding -- a global linear lift over the flattened buffer is
        # unnecessary and prohibitively expensive for large nOutput.
        # Only create the MappingLayer for non-byte (continuous) inputs.
        if self.byte_mode:
            self.lift = None
            self.params = []
            self.layers = nn.ModuleList()
        else:
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

    def prep_sentence_batch(self, sentences):
        """Turn a tuple/list of B sentence strings into a [B, nVec, 1] tensor.

        Thin wrapper over ``prepInput`` kept separate from ``getBatch`` so the
        streaming path does not touch the legacy cursor-based code.
        """
        return self.prepInput(list(sentences))
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

            # Set per-modality indices -- materialize() looks up vectors from Basis
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
            n_words: explicit word count (byte mode -- from span_meta after compaction)

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

    def expand_masked_batched(self, embedded, sentences, maskedPrediction,
                              pos):
        """Apply per-position masking to a B-wide embedded batch.

        ``embedded`` has shape [B, nVectors, embeddingSize]. For each row we
        mask position ``pos`` (or ``N-1-pos`` for RARLM) using the same
        content/positional dim discrimination as ``expand_masked``, and
        return the masked batch, the per-row targets at that position, and
        the mask index used for each row.

        Rows whose sentence has fewer than ``pos+1`` words are passed
        through unchanged and their target row is left zero; the caller
        is responsible for skipping those rows via a loss mask.

        Args:
            embedded: [B, nVec, embSize] output of forward()
            sentences: list[str] of length B, parallel to embedded
            maskedPrediction: 'MLM' / 'ARLM' / 'ARUS' / 'RARLM'
            pos: which word index to mask in each row

        Returns:
            (masked, targets, mask_positions):
                masked:         [B, nVec, embSize] gradient-connected to embedded
                targets:        [B, embSize] detached target vectors
                mask_positions: list[int] of length B -- resolved mask index
                                per row (-1 for rows that were passed through)
        """
        B, nVec, embSize = embedded.shape
        dev = embedded.device
        masked = embedded.clone()

        content_mask = torch.ones(embSize, dtype=torch.bool, device=dev)
        if self.subspace.whereEncoding.nDim > 0:
            where_idx = np.add([embSize] * len(self.subspace.whereEncoding.index),
                               self.subspace.whereEncoding.index)
            for wi in where_idx:
                if 0 <= wi < embSize:
                    content_mask[wi] = False
        if self.subspace.whenEncoding.nDim > 0:
            when_idx = np.add([embSize] * len(self.subspace.whenEncoding.index),
                              self.subspace.whenEncoding.index)
            for wi in when_idx:
                if 0 <= wi < embSize:
                    content_mask[wi] = False

        targets = torch.zeros(B, embSize, device=dev)
        mask_positions = [-1] * B
        for b in range(B):
            words = sentences[b].split()
            N = min(len(words), nVec)
            if N == 0 or pos >= N:
                continue
            p = (N - 1 - pos) if maskedPrediction == 'RARLM' else pos
            targets[b] = embedded[b, p].detach()
            masked[b, p, content_mask] = 0.0
            if maskedPrediction in ('ARLM', 'ARUS') and p + 1 < nVec:
                masked[b, p + 1:, :] = 0.0
            if maskedPrediction == 'RARLM' and p > 0:
                masked[b, :p, :] = 0.0
            mask_positions[b] = p

        return masked, targets, mask_positions

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

        # Word recovery -- content is already denormalized, codebook is L2-normalized [-1,1]
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
    # Training policy -- InputSpace decides WHEN, Embedding does HOW
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

        # -- ARIR state machine --------------------------------------
        if split == "runtime" and getattr(self.data, '_runtime_mode', None) == 'ARIR':
            return self._getBatch_arir(inputData, batchNum)

        # Use standard (non-masked) path when: no masked prediction configured,
        # or runtime split with no sentences staged (inference via runBatch).
        # Raw strings for masked prediction -- all splits store strings directly
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

            # Embed once -- retain gradient graph for the masked input path.
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

            # Hand masked embedding to forward() via cache -- no re-embedding,
            # but gradient flows back through masked_batch -> embedded -> embedding weights
            self._cached_embedding = masked_batch

            return (inputTensor, targets), batchNum + 1

    def get_reconstruction_target(self):
        """Return (target, mask) for reconstruction loss.

        target: [batch, nVec, embSize] -- unmasked post-encoding embedding
        mask:   [batch, nVec] bool -- True at masked positions to compute loss on.
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
            # pos < 0 signals "skip this row" in the streaming path
            # (sentence too short at the current mask position).
            if pos >= 0:
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

    # -- ARIR state machine ------------------------------------------

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
            # -- First call: embed seed, prepare buffer --------------
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
            # -- Subsequent calls: read reconstruction, advance cursor --
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
        try:
            self.chunking_mode = str(
                TheXMLConfig.space(section, "chunking") or "lexicon"
            )
        except KeyError:
            self.chunking_mode = "lexicon"
        if passThrough:
            return
        input = self.subspace.getEncodedInputSize()
        self.attention = AttentionLayer(input, input, type="transformer")
        self.subspace._nWordSlots = outputShape[0]
        self.params = []
        self.layers = nn.ModuleList()

    def _register_requirements(self):
        """Register PerceptualSpace-specific config requirements."""
        # passThrough spaces are identity mappings -- shape constraints don't apply.
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
            # Codebook sampling is with replacement, so nVectors is
            # independent of nOutput.  Non-codebook still needs a
            # direct one-to-one mapping.
            if not self.codebook:
                TheXMLConfig.require(
                    lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv == _na,
                    f"PerceptualSpace: non-codebook requires nVectors ({nV}) == nOutput ({nA})"
                )

    def everything(self, target="what"):
        """The universal whole -- vertex (1,1,...,1) of the perceptual hypercube.

        Valid only in perceptual space where vectors are sigmoid-normalized
        to [0,1]^d and mereological operations (min/max) apply.

        Args:
            target: encoding target -- "what", "event", or "activation".
        """
        dim = {"what": self.nDim, "event": self.muxedSize,
               "activation": self.outputShape[0]}[target]
        return torch.ones(dim, device=TheDevice.get())

    def nothing(self, target="what"):
        """The empty set -- origin (0,0,...,0) of the perceptual hypercube.

        Valid only in perceptual space where vectors are sigmoid-normalized
        to [0,1]^d and mereological operations (min/max) apply.

        Args:
            target: encoding target -- "what", "event", or "activation".
        """
        dim = {"what": self.nDim, "event": self.muxedSize,
               "activation": self.outputShape[0]}[target]
        return torch.zeros(dim, device=TheDevice.get())

    def distance(self, x, y):
        return torch.prod( [1-x, 1-y] )
    def certainty(self, x):
        pass
    @staticmethod
    def chunk_static(stream: bytes, mode: str) -> list:
        """Three-way chunking switch: raw | bpe | lexicon.

        - raw: split into single bytes.
        - lexicon: split on whitespace (word-level).
        - bpe: cold-start BPE (byte-level fallback when no trained merges).
        """
        if mode == "raw":
            return [bytes([b]) for b in stream]
        if mode == "lexicon":
            return stream.split()
        if mode == "bpe":
            # Cold-start BPE: no merges table available here -> fall back to
            # single bytes. The real BPE path runs through ChunkLayer (see
            # basicmodel/bin/Layers.py) once merges have been learned.
            return [bytes([b]) for b in stream]
        raise ValueError(
            f"chunking mode must be raw|bpe|lexicon, got {mode!r}"
        )

    def forward(self, vspace, wordSpace=None, quantize=True):
        """Perception: map input vectors to percepts via attention + VQ + chunking."""
        if self.passThrough:
            return vspace
        # Pass byte values from input for boundary detection in compose()
        if getattr(vspace, '_demuxed', False) and vspace._active is not None:
            self.subspace._byte_indices = vspace._active[:, :, 0].long()
        x = self.forwardBegin(vspace, returnVectors=True)
        if self.hasAttention:
            x = self.attention.forward(x)
        if self.codebook and quantize:
            x = self.subspace.get_vectors().forward(x)
        # Shared sparsity regularizer on the percept activations. No-op when
        # l1_lambda defaults to 0; attribute-only so configs opt in.
        if not hasattr(self, "_sparsity"):
            self._sparsity = SparsityRegularizer(
                l1_lambda=float(getattr(self, "l1_lambda", 0.0) or 0.0),
                enabled=bool(getattr(self, "codebook", False)),
            )
        x = self._sparsity(x)
        if wordSpace is not None:
            x = wordSpace.forwardPercepts(x, self.subspace)
        vspace = self.forwardEnd(x, returnVectors=True)
        vspace.normalize("percepts", target="event", normalize=True)
        return vspace

    def reverse(self, vspace, wordSpace=None):
        """Manifesting: reconstruct input vectors from percepts."""
        if self.passThrough:
            return vspace
        if self.invertible:
            vspace.normalize("percepts", target="event",
                             normalize=True, reverse=True)
        y = self.reverseBegin(vspace, returnVectors=True)
        if wordSpace is not None:
            y = wordSpace.reversePercepts(y, self.subspace)
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

        # Derive branch shapes (symmetric -- subtract off the modality you don't need)
        whatDim = self.muxedSize - self.nWhere - self.nWhen
        whatInputShape = [inputShape[0], whatDim]
        whatOutputShape = [outputShape[0], whatDim]
        whatSpaceShape = [spaceShape[0], spaceShape[1]]

        # Build what branch -- override passThrough in config temporarily
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

    def __init__(self, inputShape, spaceShape, outputShape, level_shapes=None,
                 butterfly_config=None):
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
        # last_svo is a @property backed by _last_svo (set during forward)
        input = self.subspace.getEncodedInputSize()
        output = self.subspace.getEncodedOutputSize()
        if hasAttention:
            self.attention = AttentionLayer(output, output, type="transformer")

        # -- Hierarchical mode: per-level Sigma layers ----------------
        # Average-merge keeps norms bounded, so tanh saturation is unnecessary
        # and harmful: cascaded atanh in reverse clamps values outside (-1,1),
        # destroying sample variance.  Per-level sigmas use saturate=False.
        if level_shapes is not None and len(level_shapes) >= 1:
            self._hierarchical = True
            self._level_shapes = level_shapes
            self.sigmas = nn.ModuleList()
            if butterfly_config is not None:
                # Butterfly variant: SigmaLayer is 2D*2D, wrapped in a
                # ButterflyStage that permutes, packs, sigma-applies,
                # unpacks and merges (halves N).
                pair_dim = 2 * butterfly_config["state_dim"]
                initial_n = butterfly_config["state_vectors"]
                n_stages = butterfly_config["conceptual_order"]
                for t in range(n_stages):
                    sig = SigmaLayer(
                        pair_dim, pair_dim,
                        naive=butterfly_config["naive"],
                        ergodic=butterfly_config["ergodic"],
                        invertible=True,
                    )
                    sig.saturate = False
                    stage = ButterflyStage(
                        sig, stage_idx=t, initial_n=initial_n,
                        is_last=(t == n_stages - 1))
                    self.sigmas.append(stage)
            else:
                for t, (n_t, d_t) in enumerate(level_shapes):
                    sig = SigmaLayer(d_t, d_t, naive=naive, ergodic=ergodic,
                                     invertible=invertible)
                    sig.saturate = False
                    self.sigmas.append(sig)
            # Dim projections only meaningful for the grammar path (non-butterfly).
            # One per level: level 0 projects from percept_dim, others from prior level.
            self.dim_projections = nn.ModuleList()
            if butterfly_config is None:
                percept_dim = inputShape[1]  # pre-merge percept dim
                for t, (n_t, d_t) in enumerate(level_shapes):
                    d_in = percept_dim if t == 0 else level_shapes[t - 1][1]
                    self.dim_projections.append(nn.Linear(d_in, d_t))
            self.layers = nn.ModuleList(
                list(self.sigmas) + list(self.dim_projections))
            self.params = []
            for s in self.sigmas:
                # ButterflyStage forwards .getParameters() to the inner sigma
                # via .parameters(); fall back if the stage doesn't expose it.
                if hasattr(s, "getParameters"):
                    self.params.extend(s.getParameters())
                else:
                    self.params.extend(list(s.parameters()))
            # Set forwardSigma/reverseSigma to level-0 for backward compat callers
            self.forwardSigma = self.sigmas[0].forward
            self.reverseSigma = self.sigmas[0].reverse
        else:
            # -- Original single-layer mode ----------------------------
            self._hierarchical = False
            self._level_shapes = None
            if self.reversible:
                if invertible:
                    self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                            invertible=True, nonlinear=nonlinear,
                                            stable=True)
                    self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                    self.params = self.sigma.getParameters()
                    self.layers = nn.ModuleList([self.sigma])
                else:
                    self.sigma1 = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                             invertible=True, nonlinear=nonlinear,
                                             stable=True)
                    self.sigma2 = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                             invertible=True, nonlinear=nonlinear,
                                             stable=True)
                    self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.reverse
                    self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
                    self.layers = nn.ModuleList([self.sigma1, self.sigma2])
            else:
                self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic,
                                        nonlinear=nonlinear, stable=True)
                self.forwardSigma = self.sigma.forward
                self.params = self.sigma.getParameters()
                self.layers = nn.ModuleList([self.sigma])
        # Grammar methods and SyntacticLayers are now on TheGrammar.
        # Spaces delegate to TheGrammar.project('C', ...) and
        # TheGrammar.composeSyntax('C', ...).
        self._interpretation = TheGrammar.interpretation
        self._last_svo = None

    def __getitem__(self, t):
        """Index into conceptual order levels.

        Non-hierarchical: returns self (shared sigma for all t).
        Hierarchical: returns a _LevelView that routes through sigmas[t].
        """
        if not self._hierarchical:
            return self
        return self._CSLevelView(self, t)

    class _CSLevelView:
        """Proxy routing .forward()/.reverse() through a per-level sigma.

        Accepts ``wordSpace`` for signature parity with the non-hierarchical
        path, but ignores it -- hierarchical sigmas don't invoke compose.
        """
        def __init__(self, parent, t):
            self._parent = parent
            self._t = t
            if parent._hierarchical:
                self._sigma = parent.sigmas[t]
            else:
                self._sigma = getattr(parent, 'sigma', None)
            self.subspace = parent.subspace

        def forward(self, vspace, wordSpace=None, target_count=None):
            x = vspace.materialize()
            y = self._sigma.forward(x)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

        def reverse(self, vspace, wordSpace=None):
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

    def forward(self, vspace, wordSpace=None, target_count=None):
        """Knowing: map percepts to concepts via SigmaLayer + optional attention + VQ.

        When nonlinear=True (stable=True on SigmaLayer), the weight matrix
        is L1-column-normalized so the output stays in [-1, 1] without
        requiring tanh saturation.

        ``target_count`` is forwarded to ``wordSpace.forwardConcepts``
        so the C-tier compose can use pairwise slot-mixing reduction
        (``_compose_to_target``) instead of the default cascading
        accumulator -- the latter cannot move information across the
        slot axis.  Pass ``target_count=nOutputSymbols`` from the
        non-butterfly MentalModel loop so compose reduces to exactly
        the slot count OutputSpace will read.
        """
        x = self.forwardBegin(vspace, returnVectors=True)
        y = self.forwardSigma(x)
        if self.hasAttention:
            y = self.attention.forward(y)
        if self.codebook:
            # Wide-codebook top-K: when nVectors > nOutput, route through the
            # content Codebook with topK=nOutput so the per-codebook-entry
            # activation is pruned to the nOutput strongest survivors.
            if (isinstance(self.subspace.what, Codebook)
                    and self.nVectors > self.outputShape[0]):
                y = self.subspace.what.forward(y, topK=self.outputShape[0])
            else:
                y = self.subspace.get_vectors().forward(y)
        # Shared sparsity regularizer on the concept activations. No-op when
        # l1_lambda defaults to 0; attribute-only so configs opt in.
        if not hasattr(self, "_sparsity"):
            self._sparsity = SparsityRegularizer(
                l1_lambda=float(getattr(self, "l1_lambda", 0.0) or 0.0),
                enabled=bool(getattr(self, "codebook", False)),
            )
        y = self._sparsity(y)
        if wordSpace is not None:
            y, self._last_svo = wordSpace.forwardConcepts(
                y, self.subspace, target_count=target_count)
        else:
            self._last_svo = None
        vspace = self.forwardEnd(y, returnVectors=True)
        vspace.normalize("concepts", target="what")       # range check
        vspace.normalize("concepts", target="where")      # range check
        vspace.normalize("concepts", target="activation")  # range check
        return vspace

    @property
    def last_svo(self):
        """Return SVO tuple from last ternary lift, or None."""
        return self._last_svo

    def reverse(self, vspace, wordSpace=None):
        """Visualizing: reconstruct percepts from concepts via reverse SigmaLayer."""
        y = self.reverseBegin(vspace, returnVectors=True)
        if wordSpace is not None:
            y = wordSpace.reverseConcepts(y, self.subspace)
        if self.processSymbols:
            y = self.dereference(y)
        y = self.reverseSigma(y)
        self.concepts = y
        vspace = self.reverseEnd(y, returnVectors=True)
        vspace.normalize("percepts", target="what")   # range check
        vspace.normalize("percepts", target="where")  # range check
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
    The START-level true() evaluates the full stack activation -> scalar.
    """
    name = "Symbols"
    config_section = "SymbolicSpace"

    def empty_state(self, batch=1):
        """Return a zero tensor shaped like this space's symbolic state.

        Used by the unified merged outer loop to seed ``ss`` before the
        first concept emission. ``batch`` sizes the leading axis; the
        rest comes from ``outputShape``.
        """
        nOutput = int(self.outputShape[0])
        nDim = int(self.outputShape[1])
        return torch.zeros(int(batch), nOutput, nDim, device=TheDevice.get())

    def __init__(self, inputShape, spaceShape, outputShape, conceptualSpace=None,
                 level_shapes=None, butterfly_config=None):

        section = self.config_section
        passThrough = TheXMLConfig.space(section, "passThrough")
        super().__init__(inputShape, spaceShape, outputShape, customVQ=True)
        self.conceptualSpace = conceptualSpace
        self.passThrough = passThrough
        # Symbols carry 4-valued (quaternary) truth in .what via a 2-dim
        # bivector [pos_pole, neg_pole]. Override the inherited content
        # width accordingly so the codebook row = 2 + nWhere + nWhen.
        # See basicmodel/doc/BuddhistParallels.md for the tetralemma.
        self.subspace.nWhat = 2
        self.subspace.muxedSize = 2 + self.subspace.nWhere + self.subspace.nWhen
        # PiLayer maps on the nDim axis: concept_dim+obj -> symbol_dim+obj.
        # nVectors passes through unchanged via batched matmul.
        nConceptDim = inputShape[1]     # concept_dim + obj (where+when)
        nSymbolDim = outputShape[1]     # symbol_dim + obj (where+when)
        nSymbols = spaceShape[0]

        if level_shapes is not None and len(level_shapes) >= 1:
            self._hierarchical = True
            self._level_shapes = level_shapes
            self.pi_layers = nn.ModuleList()
            if butterfly_config is not None:
                pair_dim = 2 * butterfly_config["state_dim"]
                initial_n = butterfly_config["state_vectors"]
                n_stages = butterfly_config["conceptual_order"]
                self._butterfly_symbol_width = butterfly_config["symbol_width"]
                self._butterfly_symbol_factor = butterfly_config["symbol_factor"]
                for t in range(n_stages):
                    pi = PiLayer(pair_dim, pair_dim, invertible=True, monotonic=True)
                    # Pi operates on sigma's merged output (N already halved).
                    stage = ButterflyStage(
                        pi, stage_idx=t, initial_n=initial_n // 2,
                        is_last=True)  # no merge -- sigma already merged
                    self.pi_layers.append(stage)
            else:
                self._butterfly_symbol_width = None
                self._butterfly_symbol_factor = None
                for t, (n_t, d_t) in enumerate(level_shapes):
                    self.pi_layers.append(
                        PiLayer(d_t, nSymbolDim, invertible=True, monotonic=True))
            self.layer = self.pi_layers[0]  # default for non-hierarchical callers
        else:
            self._hierarchical = False
            self.pi_layers = None
            self._butterfly_symbol_width = None
            self._butterfly_symbol_factor = None
        if not self._hierarchical:
            self.layer = PiLayer(nConceptDim, nSymbolDim, invertible=True, monotonic=True)

        # Symbol objective: residual accuracy is primary; L1 remains a
        # secondary compactness pressure on latent activations.
        def _symbol_cfg(key, default):
            try:
                return float(TheXMLConfig.space(section, key))
            except (KeyError, TypeError, ValueError):
                return default

        self.l1_lambda = _symbol_cfg("l1Lambda", 0.0)
        self.discontinuity_lambda = _symbol_cfg("discontinuityLambda", 0.0)
        self.symbol_residual_scale = _symbol_cfg("symbolResidualScale", 1.0)
        self.output_symbol_residual_scale = _symbol_cfg(
            "outputSymbolResidualScale", 0.0)
        self.commitment_beta = _symbol_cfg("commitmentBeta", 0.25)
        try:
            use_vqvae_raw = TheXMLConfig.space(section, "useVQVAE")
        except (KeyError, TypeError, ValueError):
            use_vqvae_raw = False
        self.use_vqvae = bool(use_vqvae_raw)
        try:
            raw_mode = TheXMLConfig.space(section, "gradientMode")
        except (KeyError, TypeError, ValueError):
            raw_mode = None
        if raw_mode is None:
            self.gradient_mode = "ste"
        else:
            mode_str = str(raw_mode).strip().lower()
            if mode_str not in ("snap", "ste", "rotation"):
                raise ValueError(
                    f"SymbolicSpace gradientMode={raw_mode!r} is invalid; "
                    "expected one of 'snap', 'ste', 'rotation'.")
            self.gradient_mode = mode_str
        self.decorrelation_weight = _symbol_cfg("decorrelationWeight", 0.0)
        self.spectral_flatness_weight = _symbol_cfg("spectralFlatnessWeight", 0.0)
        # ImpenetrableLayer: mereological separation regularizer on the
        # codebook. Defaults to 0 (no-op) so existing configs are unchanged.
        # ``impenetrableOverlap`` replaces the legacy antisymmetry+transitivity
        # pair with a single ``overlap * |trust-diff|`` penalty driven by
        # clipped-cosine parthood. See Layers.ImpenetrableLayer.
        self.impenetrable_overlap = _symbol_cfg("impenetrableOverlap", 0.0)
        self.impenetrable_variance = _symbol_cfg("impenetrableVariance", 0.0)
        self.reset_symbol_objective()

        # Truth accumulation: accumulateTruth is 0..1 (DoT for recorded symbols).
        # Default 0 (off). Server sets to 1 when processing the TruthSet,
        # then resets to 0. The TruthLayer lives on ``WordSpace``; callers
        # reach it via ``self.wordSpace.truth_layer`` (see forward() below
        # and the truth-using paths in BasicModel).
        try:
            self.accumulateTruth = float(TheXMLConfig.space(section, "accumulateTruth"))
        except (KeyError, TypeError, ValueError):
            self.accumulateTruth = 0.0

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

        self.params = list(self.parameters())

        # Rule codebook -- learned vectors for grammar rule identities.
        # Index 0 is reserved as the empty-slot sentinel (zero vector,
        # non-training via padding_idx); indices 1..nRules map to
        # grammar rule_ids 0..nRules-1.
        # See plan: Architectural addition -- WordSpace / Unified codebook.
        self.rule_codebook = None
        self.nRuleEntries = 0

    def _build_object_basis(self):
        """Event is a writable Tensor -- codebook lives on .what."""
        return None

    def _build_what_basis(self):
        """Symbol codebook on .what, monotonic. One row per symbol.

        Row width = 2 + nWhere + nWhen. The leading 2 dims carry the
        bivector [pos_pole, neg_pole] encoding the 4-valued truth of the
        symbol (tetralemma: TRUE=[1,0], FALSE=[0,1], BOTH=[1,1],
        NEITHER=[0,0]). Trailing nWhere+nWhen dims carry per-symbol
        positional/temporal template info. Negation operates on the
        leading 2 dims only (see BuddhistParallels.md for the catuskoti
        mapping).
        """
        basis = Codebook()
        basis.create(
            self.inputShape[0],
            self.nVectors,
            2 + self.nWhere + self.nWhen,
            customVQ=self.customVQ,
            passThrough=not self.codebook,
            monotonic=True,
        )
        return basis

    @classmethod
    def _build_sparsity_regularizer(cls, l1_lambda, codebook_enabled):
        return SparsityRegularizer(
            l1_lambda=float(l1_lambda or 0.0),
            enabled=bool(codebook_enabled),
        )

    def l1_proximal(self, x):
        """Soft-threshold activations used as a sparsity bias.

        Delegates to the shared SparsityRegularizer. Kept as a thin
        wrapper for backward compatibility with call sites in this file
        and in Models.py.
        """
        if not hasattr(self, "_sparsity"):
            self._sparsity = self._build_sparsity_regularizer(
                self.l1_lambda, self.codebook
            )
        return self._sparsity(x)

    def smoothing_penalty(self, x):
        """Total-variation penalty along the concept axis of symbol activations.

        Bivector-aware via pair-max collapse; 0 when discontinuityLambda=0
        or when disabled. See Layers.SmoothingRegularizer.
        """
        if not hasattr(self, "_smoothing"):
            self._smoothing = SmoothingRegularizer(
                lam=float(self.discontinuity_lambda or 0.0),
                enabled=bool(self.discontinuity_lambda and self.discontinuity_lambda > 0.0),
            )
        return self._smoothing(x)

    def _build_impenetrable_layer(self):
        return ImpenetrableLayer(
            overlap_weight=float(self.impenetrable_overlap or 0.0),
            variance_floor=float(self.impenetrable_variance or 0.0),
            enabled=True,
        )

    def impenetrable_loss(self):
        """Return the ImpenetrableLayer regularizer over the current codebook.

        Returns a scalar tensor. If no codebook has been built yet, or all
        weights are zero, returns a zero scalar on the current device.
        """
        if not hasattr(self, "_impenetrable"):
            self._impenetrable = self._build_impenetrable_layer()
        basis = getattr(self.subspace, "basis", None)
        if basis is None:
            return torch.zeros((), device=TheDevice.get())
        W = basis.getW() if hasattr(basis, "getW") else None
        if W is None or not isinstance(W, torch.Tensor):
            return torch.zeros((), device=TheDevice.get())
        return self._impenetrable(W, basis)

    def reset_symbol_objective(self):
        self._symbol_objective_terms = {}
        self._symbol_objective_count = 0

    def _decorrelation_loss(self, residual):
        flat = residual.reshape(-1, residual.shape[-1])
        if flat.shape[0] < 2 or flat.shape[-1] < 2:
            return residual.new_tensor(0.0)
        flat = flat - flat.mean(dim=0, keepdim=True)
        std = flat.std(dim=0, unbiased=False, keepdim=True).clamp_min(epsilon)
        flat = flat / std
        corr = flat.transpose(0, 1) @ flat / max(flat.shape[0], 1)
        eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
        return ((corr * (1 - eye)) ** 2).mean()

    def _spectral_flatness_loss(self, residual):
        if residual.shape[-1] < 2:
            return residual.new_tensor(0.0)
        power = torch.fft.rfft(residual, dim=-1).abs().square() + epsilon
        log_power = torch.log(power)
        return (log_power - log_power.mean(dim=-1, keepdim=True)).square().mean()

    @staticmethod
    def _vq_chunk_budget():
        """Memory budget (bytes) for VQ distance matrix chunks.

        With d content dims and K codebook entries, each row of the
        distance matrix costs K * 4 bytes.  The budget controls how
        many rows are processed per matmul.  Larger budgets mean fewer
        sequential chunks -- critical for ARLM batches where N can
        reach hundreds of thousands.
        """
        device = str(TheDevice.get())
        if 'cuda' in device:
            try:
                props = torch.cuda.get_device_properties(device)
                return max(256 << 20, props.total_mem // 4)
            except Exception:
                pass
        # MPS (Apple Silicon) or ROCm without CUDA properties:
        # use 4 GiB -- safe on any >=16 GB unified/GPU memory system.
        if 'mps' in device or 'cuda' in device:
            return 4 << 30
        # CPU fallback
        return 2 << 30

    def _nearest_symbol_target(self, predicted):
        """Nearest codebook symbols as detached residual targets.

        The core operation is a batched L2-nearest-neighbour lookup:
        for each of N flattened symbol rows (d content dims), find the
        closest of K codebook entries.  This decomposes into the matmul
        ``[N, d] @ [d, K]`` plus per-row and per-codebook-entry norms.

        With d = 4 the matmul is trivially fast; the bottleneck is the
        [N, K] output matrix.  We chunk over N to keep that matrix
        within the memory budget from ``_vq_chunk_budget()``.
        """
        if not self.codebook:
            return None
        basis = getattr(self.subspace, "what", None)
        if basis is None or getattr(basis, "passThrough", False):
            return None
        weight = basis.getW() if hasattr(basis, "getW") else None
        if weight is None or weight.numel() == 0:
            return None
        flat = predicted.reshape(-1, predicted.shape[-1])
        weight = weight.detach().to(device=flat.device, dtype=flat.dtype)
        weight = weight.reshape(-1, weight.shape[-1])
        n = min(flat.shape[-1], weight.shape[-1])
        if n <= 0:
            return None
        flat_content = flat[:, :n]
        weight_content = weight[:, :n]
        weight_sq = (weight_content * weight_content).sum(dim=-1).unsqueeze(0)  # [1, K]

        K = weight_content.shape[0]
        budget = self._vq_chunk_budget()
        max_rows = max(1, budget // (K * 4))
        N = flat_content.shape[0]

        if N <= max_rows:
            # Single matmul: [N, d] @ [d, K] -> [N, K]
            flat_sq = (flat_content * flat_content).sum(dim=-1, keepdim=True)
            dists = flat_sq - 2 * (flat_content @ weight_content.T) + weight_sq
            indices = dists.argmin(dim=-1)
        else:
            # Chunked: each chunk is [chunk, d] @ [d, K]
            indices = torch.empty(N, dtype=torch.long, device=flat.device)
            for start in range(0, N, max_rows):
                end = min(start + max_rows, N)
                chunk = flat_content[start:end]
                chunk_sq = (chunk * chunk).sum(dim=-1, keepdim=True)
                chunk_dists = chunk_sq - 2 * (chunk @ weight_content.T) + weight_sq
                indices[start:end] = chunk_dists.argmin(dim=-1)

        target = flat.detach().clone()
        target[:, :n] = weight_content[indices]
        return target.reshape_as(predicted)

    def accumulate_symbol_objective(self, predicted, target=None,
                                    use_codebook_target=False,
                                    residual_scale=None):
        """Accumulate residual-first symbol objective terms for one Pi pass."""
        terms = {}
        residual = None
        if target is None and use_codebook_target:
            target = self._nearest_symbol_target(predicted)
        scale = self.symbol_residual_scale if residual_scale is None else residual_scale
        if target is not None and scale > 0.0:
            target = target.detach()
            residual = predicted - target
            terms["symbol_residual"] = (
                scale * F.mse_loss(predicted, target)
            )
        # L1 only makes sense when symbols are discretized through a
        # codebook; on continuous symbols it would just shrink magnitudes
        # without promoting compactness of the codebook selection itself.
        if self.codebook and self.l1_lambda > 0.0:
            terms["symbol_l1"] = self.l1_lambda * predicted.abs().mean()
        if self.discontinuity_lambda and self.discontinuity_lambda > 0.0:
            terms["symbol_smoothing"] = self.smoothing_penalty(predicted)
        if residual is not None and self.decorrelation_weight > 0.0:
            terms["symbol_decorrelation"] = (
                self.decorrelation_weight * self._decorrelation_loss(residual)
            )
        if residual is not None and self.spectral_flatness_weight > 0.0:
            terms["symbol_spectral_flatness"] = (
                self.spectral_flatness_weight * self._spectral_flatness_loss(residual)
            )
        if not terms:
            return
        for key, value in terms.items():
            self._symbol_objective_terms[key] = (
                self._symbol_objective_terms.get(key, value.new_tensor(0.0)) + value
            )
        self._symbol_objective_count += 1

    def symbol_objective_terms(self):
        if not self._symbol_objective_terms:
            averaged = {}
        else:
            count = max(self._symbol_objective_count, 1)
            averaged = {
                key: value / count
                for key, value in self._symbol_objective_terms.items()
            }
        # ImpenetrableLayer operates on the codebook itself, not on per-
        # prediction residuals, so it is computed once per term-collection
        # rather than accumulated per Pi pass.
        if (self.impenetrable_overlap > 0.0
                or self.impenetrable_variance > 0.0):
            imp = self.impenetrable_loss()
            if imp is not None and torch.is_tensor(imp) and imp.requires_grad:
                averaged["symbol_impenetrable"] = imp
            elif imp is not None and torch.is_tensor(imp) and imp.abs().item() > 0.0:
                averaged["symbol_impenetrable"] = imp
        return averaged

    def symbol_objective_loss(self):
        terms = self.symbol_objective_terms()
        if not terms:
            return None
        return sum(terms.values())

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
            self._t = t
            if parent._hierarchical:
                self._pi = parent.pi_layers[t]
            else:
                self._pi = parent.layer
            self.subspace = parent.subspace

        def forward(self, vspace, wordSpace=None, quantize=True, is_last=False):
            x = vspace.materialize()
            y = self._pi.forward(x)
            # Butterfly path: ``_pi`` is a ButterflyStage.  Codebook-aware
            # losses are expensive on full symbol grids (millions of rows
            # under ARLM), so they run only on the terminal stage.
            # Grammar / plain-hierarchical path: per-level PiLayer.
            # Intermediate stages contribute training signal via
            # ``accumulate_symbol_objective`` on every step, matching the
            # pre-refactor contract; without it the non-terminal Pi
            # layers receive no loss gradient and the model plateaus.
            is_butterfly = isinstance(self._pi, ButterflyStage)
            if is_butterfly:
                if is_last:
                    nearest_target = self._parent._nearest_symbol_target(y)
                    if (self._parent.use_vqvae and self._parent.codebook
                            and self._parent.commitment_beta > 0.0
                            and nearest_target is not None):
                        target_detached = nearest_target.detach().to(
                            device=y.device, dtype=y.dtype)
                        n = min(y.shape[-1], target_detached.shape[-1],
                                self._parent.nDim)
                        if n > 0:
                            commit = self._parent.commitment_beta * F.mse_loss(
                                y[..., :n], target_detached[..., :n])
                            prev = self._parent._symbol_objective_terms.get(
                                "symbol_commitment", y.new_tensor(0.0))
                            self._parent._symbol_objective_terms["symbol_commitment"] = prev + commit
                    self._parent.accumulate_symbol_objective(y, target=nearest_target)
            else:
                if quantize:
                    y = self._parent.l1_proximal(y)
                self._parent.accumulate_symbol_objective(
                    y, use_codebook_target=quantize)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

        def reverse(self, vspace, wordSpace=None):
            x = vspace.materialize()
            y = self._pi.reverse(x)
            self._parent.subspace.set_event(y)
            return self._parent.subspace

    @property
    def vocabulary(self):
        return self.subspace.what

    def forward(self, vspace, wordSpace=None, quantize=True, is_last=False):
        """Map concept vectors to symbol vectors via PiLayer (Pi).

        PiLayer maps on the nDim axis: [B, nVectors, concept_dim] ->
        [B, nVectors, symbol_dim].  nVectors passes through unchanged.
        With a single-concept subspace [B, 1, D], produces [B, 1, symbol_dim].

        1. Materialize full concept vectors from input subspace.
        2. Map through PiLayer (log-space multiplicative, monotonic).
        3. Grammar derivation (if syntax=True): shift/reduce over S-tier.
        4. Apply swap on whereEncodings of binary node children.
        5. Optionally quantize into the symbol codebook.
        6. Store as symbol vectors in subspace event.
        """
        if self.passThrough:
            return vspace
        vspace = self.forwardBegin(vspace)
        act = vspace.materialize()                        # [B, N, concept_dim]
        act = self.layer.forward(act)                     # [B, N, symbol_dim]
        if quantize:
            act = self.l1_proximal(act)                   # sparsity bias only

        if self.accumulateTruth > 0 and wordSpace is not None:
            truth_layer = getattr(wordSpace, 'truth_layer', None)
            if truth_layer is not None:
                basis = getattr(self.subspace, 'basis', None)
                for i in range(act.shape[0]):
                    for j in range(act.shape[1]):
                        vec = act[i, j]
                        score = truth_layer.should_store(
                            vec,
                            min_magnitude=self._truth_min_magnitude,
                            min_novelty=self._truth_min_novelty,
                            max_inconsistency=self._truth_max_inconsistency)
                        if self.accumulateTruth * score > 0.5:
                            truth_layer.record(vec, degree=self.accumulateTruth,
                                               basis=basis)

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

        if wordSpace is not None:
            # Rule #2: C -> S axis commitment. The demux side effect
            # (populating the subspace's what/where/when modality slots
            # from the muxed [B, N, D] tensor) happens inside
            # forwardSymbols() so the axis-separated state is live
            # before any slot selector runs.
            act = wordSpace.forwardSymbols(act, self.subspace)

        # VQ-VAE modes (gated on ``self.use_vqvae``):
        #   reversible model  -> continuous forward + commitment loss only.
        #       The reverse path requires exact invertibility
        #       (``pi.reverse(pi.forward(x)) == x``).  A hard-quantized
        #       forward would break that, so we keep ``act`` continuous
        #       and rely on commitment loss to pull the encoder toward
        #       the nearest codebook entry.  The commitment loss alone
        #       suffices: the encoder already receives gradient through
        #       the continuous forward, so no STE is needed.
        #   non-reversible model -> STE forward.  Downstream sees the
        #       hard codebook pick, backward flows as identity through
        #       the quantization bottleneck back into the encoder.
        #   non-VQVAE + codebook + caller opted into quantize ->
        #       hard-quantize forward, no gradient estimator (commitment
        #       loss is not accumulated either; the codebook is updated
        #       via its manual EMA path in Codebook.forward).
        use_vqvae_reversible = self.use_vqvae and self.reversible and self.codebook
        use_vqvae_nonreversible = (self.use_vqvae and not self.reversible
                                   and self.codebook)
        hard_quantize = (not self.use_vqvae) and self.codebook and quantize
        if use_vqvae_reversible:
            # Continuous forward -- event carries z_e so reverse is exact.
            z_e = act
            predicted = z_e.clone()
            self.subspace.set_event(z_e)
            vspace = self.forwardEnd(self.subspace)
            target = self._nearest_symbol_target(z_e)
            if target is not None and self.commitment_beta > 0.0:
                target_detached = target.detach().to(
                    device=z_e.device, dtype=z_e.dtype)
                n = min(z_e.shape[-1], target_detached.shape[-1], self.nDim)
                if n > 0:
                    commit = self.commitment_beta * F.mse_loss(
                        z_e[..., :n], target_detached[..., :n])
                    prev = self._symbol_objective_terms.get(
                        "symbol_commitment", commit.new_tensor(0.0))
                    self._symbol_objective_terms["symbol_commitment"] = prev + commit
            self.accumulate_symbol_objective(predicted, target)
        elif use_vqvae_nonreversible:
            # Non-reversible VQVAE: forward = hard quantized, backward uses
            # the configured gradient estimator (snap/ste/rotation).
            z_e = act
            predicted = z_e.clone()
            self.subspace.set_event(z_e)
            self.subspace.what.forward(self.subspace)
            vspace = self.forwardEnd(self.subspace)
            quantized = vspace.materialize()
            z_q = Codebook.apply_gradient_estimator(
                z_e, quantized, mode=self.gradient_mode)
            self.subspace.set_event(z_q)
            vspace = self.forwardEnd(self.subspace)
            quantized_detached = quantized.detach()
            n = min(z_e.shape[-1], quantized_detached.shape[-1], self.nDim)
            if self.commitment_beta > 0.0 and n > 0:
                commit = self.commitment_beta * F.mse_loss(
                    z_e[..., :n], quantized_detached[..., :n])
                prev = self._symbol_objective_terms.get(
                    "symbol_commitment", commit.new_tensor(0.0))
                self._symbol_objective_terms["symbol_commitment"] = prev + commit
            # Also pick up the codebook-internal commit loss (VQ's own
            # encoder-commitment term). Previously discarded; now threaded
            # into the objective so it actually drives learning.
            cb_commit = getattr(self.subspace.what, "last_commit_loss", None)
            if cb_commit is not None and torch.is_tensor(cb_commit) and cb_commit.requires_grad:
                prev = self._symbol_objective_terms.get(
                    "codebook_commit", cb_commit.new_tensor(0.0))
                self._symbol_objective_terms["codebook_commit"] = prev + cb_commit
            self.accumulate_symbol_objective(predicted, quantized_detached)
        elif hard_quantize:
            # Non-VQVAE hard-quantize: run the codebook forward to snap
            # to the nearest entry and use that as the symbol target.
            predicted = act.clone()
            self.subspace.set_event(act)
            self.subspace.what.forward(self.subspace)
            vspace = self.forwardEnd(self.subspace)
            target = vspace.materialize()
            self.accumulate_symbol_objective(predicted, target)
        else:
            self.accumulate_symbol_objective(act)
            self.subspace.set_event(act)
            vspace = self.forwardEnd(self.subspace)

        vspace.normalize("symbols", target="what")   # range check
        vspace.normalize("symbols", target="where")  # range check
        return vspace

    def init_rule_codebook(self, grammar):
        """Allocate the unified rule codebook for WordSubSpace lookups.

        Sized to the grammar's S-tier rule count. Each row is a learnable
        vector the same width as the symbol codebook (self.nDim). Index 0
        is an empty-slot sentinel (padding_idx prevents gradient). Indices
        1..nRules map to rule_ids 0..nRules-1. Used by WordSubSpace.push()
        to embed rule identities into the word-stream buffer.

        Idempotent: calling twice is a no-op after the first allocation.
        Called by ``WordSpace._build_symbolic_layer`` as part of the
        unified grammar-infrastructure construction path.
        """
        if self.rule_codebook is not None:
            return
        try:
            nRules = int(len(grammar.symbolic()))
        except Exception:
            nRules = 0
        self.nRuleEntries = nRules
        if nRules > 0 and self.nDim > 0:
            self.rule_codebook = nn.Embedding(nRules + 1, self.nDim, padding_idx=0)
            nn.init.normal_(self.rule_codebook.weight, std=0.01)
            with torch.no_grad():
                self.rule_codebook.weight[0].zero_()
            self.layers.append(self.rule_codebook)
            self.params = list(self.parameters())

    def lookup_rule(self, rule_id):
        """Return the learnable codebook vector for a grammar rule_id.

        rule_id is a 0-based index into grammar.symbolic(); the codebook
        stores it at position rule_id+1 (index 0 is the empty-slot
        sentinel). Returns a [nDim] tensor, or None if the codebook has
        not been allocated yet.
        """
        if self.rule_codebook is None:
            return None
        if rule_id is None or rule_id < 0 or rule_id >= self.nRuleEntries:
            # Return the zero sentinel at index 0.
            idx = torch.zeros(1, dtype=torch.long,
                              device=self.rule_codebook.weight.device)
        else:
            idx = torch.tensor([rule_id + 1], dtype=torch.long,
                               device=self.rule_codebook.weight.device)
        return self.rule_codebook(idx).squeeze(0)

    def reverse(self, vspace, wordSpace=None):
        """Map symbol vectors back to concept vectors via PiLayer.reverse (Pi^-^1).

        Reverse maps on nDim axis: [B, N, symbol_dim] -> [B, N, concept_dim].
        """
        if self.passThrough:
            return vspace
        vspace = self.reverseBegin(vspace)
        act = vspace.materialize()                        # [B, N, symbol_dim]
        if wordSpace is not None:
            act = wordSpace.reverseSymbols(act, self.subspace)
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
        result.normalize("concepts", target="where")  # range check
        return result

    def evaluate_truth(self, vspace, wordSpace=None):
        """START-level: evaluate truth of the full stack -> scalar.

        Reads ``trueForward`` from the S-tier SyntacticLayer owned by
        WordSpace. Returns a passthrough when no WordSpace is wired
        (e.g. unit tests constructing a SymbolicSpace in isolation).
        """
        act = vspace.materialize(mode="activation")
        if wordSpace is None:
            return act
        layer = getattr(wordSpace, 'symbolicSyntacticLayer', None)
        if layer is None:
            return act
        return layer.trueForward(act, self.subspace)

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
            # PiLayer activation-mode path for butterfly symbol output
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
            # Activation-mode: PiLayer on symbol activations [B, nSymbols] -> [B, nOutput]
            act = vspace.materialize(mode="activation")
            output = self._piLayer.forward(act)
            output = TheData.denormalize(output, which="output")
            self.subspace.set_activation(output)
            return self.subspace

        # Default vector-mode: LinearLayer on flattened vectors
        x = self.forwardBegin(vspace, returnVectors=True)
        output = self.forwardLinear(x)
        output = TheData.denormalize(output, which="output")
        if self.codebook:
            output = self.subspace.get_vectors().forward(output)
        vspace = self.forwardEnd(output, returnVectors=True)
        return vspace

    def reverse(self, vspace):
        """Being acted upon: map output back to symbolic space."""
        if self.nonlinear_output:
            # Activation-mode: PiLayer reverse [B, nOutput] -> [B, nSymbols]
            act = vspace.materialize(mode="activation")
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
        self.thought_free = False

    # -- Rule catalog --------------------------------------------------

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

    # -- Configuration from XML ----------------------------------------

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
            canonical = f"{lhs} -> {rhs}"
            return self.RuleDef(lhs, canonical, arity, func_name)
        if rhs == 'epsilon':
            return self.RuleDef(lhs, f"{lhs} -> epsilon", 0, None)
        if rhs == 'I P':
            return self.RuleDef(lhs, f"{lhs} -> chunk(I, P)", 2, 'chunk')
        if rhs in ('S', 'C', 'P'):
            return self.RuleDef(lhs, f"{lhs} -> {rhs}", 1, None)
        if rhs == 'I':
            return self.RuleDef(lhs, f"{lhs} -> I", 0, None)
        raise ValueError(f"Cannot parse grammar rule: {lhs} -> {rhs}")

    _NOOP_GRAMMAR = {'START': 'S', 'S': 'C', 'C': ['not(C)', 'P'], 'P': 'I'}

    def _ensure_configured(self):
        if self._configured:
            return
        cfg = None
        try:
            candidate = TheXMLConfig.get("WordSpace.language.grammar")
            if isinstance(candidate, dict):
                cfg = candidate
        except (KeyError, AttributeError):
            pass
        if cfg is None:
            cfg = self._NOOP_GRAMMAR
        self.configure(cfg)
        try:
            interp = TheXMLConfig.get("WordSpace.language.interpretation")
            self.interpretation = float(interp)
        except (KeyError, AttributeError, TypeError, ValueError):
            pass

    # -- Rule queries --------------------------------------------------

    def symbolic(self):
        self._ensure_configured()
        if self.thought_free:
            # Shamatha speech: restrict to S -> C only (one derivation rule)
            t = self.symbolic_transition()
            return [t] if t is not None else []
        return [i for i, r in enumerate(self.rules) if r.tier == 'S']

    def conceptual(self):
        self._ensure_configured()
        rules = [i for i, r in enumerate(self.rules) if r.tier == 'C']
        if self.thought_free:
            rules = [i for i in rules if self.rules[i].method_name != 'not']
        return rules

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
        """Set of method names available on the C (conceptual) tier.

        Under shamatha (``thought_free=True``) the ``not`` method is omitted
        so negation only manifests at/after SymbolicSpace.
        """
        return {r.method_name for r in self.rules
                if r.tier == 'C' and r.method_name
                and not (self.thought_free and r.method_name == 'not')}

    @property
    def p_methods(self):
        """Set of method names available on the P (perceptual) tier."""
        return {r.method_name for r in self.rules if r.tier == 'P' and r.method_name}

    def _c_rule_ids(self):
        """Return dict of method_name -> rule_id for C-tier operational rules."""
        result = {}
        for i, r in enumerate(self.rules):
            if r.tier == 'C' and r.method_name is not None:
                result[r.method_name] = i
        return result

    # _conceptual_forward, _symbolic_forward, forward, reverse -- moved to
    # specialized SyntacticLayer subclasses (ConceptualSyntacticLayer,
    # SymbolicSyntacticLayer, PerceptualSyntacticLayer).  Grammar retains
    # only rule catalog, project(), and *Forward/*Reverse operations.

    # composeSyntax, _compose_conceptual, _compose_symbolic -- removed.
    # Soft superposition is now inlined in _conceptual_forward and _symbolic_forward.

    # -- C-tier operations live on SyntacticLayer / ConceptualSyntacticLayer
    # as *Forward / *Reverse method pairs.  See _RULE_METHODS dispatch.
TheGrammar = Grammar()

class SyntacticLayer(Layer):
    """Per-space rule prediction layer for the recursive grammar.

    Each instance handles a subset of the Grammar's rules (one cognitive
    space's rules).  Uses a weight-tied recursive architecture with depth
    embeddings.

    **This layer only predicts rules and generates word tuples.**  It does
    not execute operations on representations -- that is done by the owning
    space's ``projectXxx()`` method, which knows the native representation
    type (activations, vectors, etc.).

    Args:
        nInput:    activation width (number of symbol/concept/percept slots).
        nOutput:   same as nInput.
        rules:     list of global Grammar rule IDs this layer handles
                   (e.g. [1,2,3,4,5] for the symbolic space).
        transition_rule: optional global rule ID for the transition rule
                   (e.g. 6 for S->C).  Included in prediction but signals
                   hand-off to the next space.
        max_depth: maximum derivation depth.
        hidden_dim: width of the shared derivation hidden state.
        grammar:   Grammar instance.
        tau:       Gumbel-softmax temperature.
    """

    # Transition bias scale: (1 - interpretation) * TRANSITION_SCALE is added
    # to the transition rule's logit. The transition rule (S->C or C->P) acts
    # as NOP -- "stop deriving this tier, pass through."
    # Low interpretation -> transition dominates -> no reductions (episodic).
    # High interpretation -> grammar rules fire -> composition (semantic).
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
        # Map from local index -> global rule ID
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

    # -- Basis-delegated rule execution ----------------------------

    def _basis(self, subspace):
        """Return the Basis from a SubSpace (or None)."""
        return subspace.basis if subspace is not None else None

    def _mono(self, subspace):
        """True if this subspace uses monotonic logic."""
        b = self._basis(subspace)
        return b is None or b.monotonic

    @staticmethod
    def _expand_mask(mask, feature_dim):
        """Expand a concept-axis mask ``[K]`` to a storage mask ``[feature_dim]``.

        When ``feature_dim == 2 * K`` the input is interpreted as bivector
        storage and the mask is repeated so paired poles ``(2k, 2k+1)`` stay
        co-masked. When ``feature_dim == K`` the mask is used as-is.
        Returns a tensor on the caller's device/dtype.
        """
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)
        K = mask.shape[-1]
        if feature_dim == 2 * K:
            return mask.repeat_interleave(2)
        if feature_dim == K:
            return mask
        # Fallback: if neither matches, broadcast / truncate conservatively.
        if feature_dim < K:
            return mask[:feature_dim]
        return torch.cat([mask, mask.new_zeros(feature_dim - K)], dim=-1)

    def _apply_mask(self, out, mask, subspace=None):
        """Apply a mask either along the feature axis (default) or the
        position axis (when ``subspace`` is provided and ``mask`` aligns
        with ``out.shape[-2]``).

        Feature-axis path: element-wise multiply along the last dim.
        Position-axis path: zero the corresponding rows on
        ``subspace._active`` so ``SubSpace.materialize()`` gating propagates
        the mask downstream. Returns ``out`` unchanged in this case.
        No-op when ``mask is None``.
        """
        if mask is None or not torch.is_tensor(out):
            return out
        if (subspace is not None
                and torch.is_tensor(mask)
                and out.ndim >= 2
                and mask.shape[-1] == out.shape[-2]
                and getattr(subspace, "_active", None) is not None):
            active = subspace._active
            # mask aligns with the N (position) axis of _active = [..., N, M].
            # Append a singleton trailing dim for M; broadcasting handles
            # any leading batch dims automatically.
            m = mask.to(device=active.device, dtype=active.dtype).unsqueeze(-1)
            subspace._active = active * m
            return out
        m = self._expand_mask(mask, out.shape[-1])
        if m is None:
            return out
        return out * m.to(device=out.device, dtype=out.dtype)

    # -- Forward/Reverse dispatch ------------------------------------
    #
    # C-tier ops (invertible): not, intersection, union, lift, lower
    # S-tier ops (lossy, no inverse): equals, part, true, non, swap
    # P-tier ops (invertible): chunk
    #
    # _RULE_METHODS maps rule name -> (forwardName, reverseName|None, binary)

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

    def project(self, grammar, rule_id, left, right=None, third=None, subspace=None,
                mask=None):
        """Execute a grammar rule forward. Subclasses override for parametric rules.

        ``mask`` (optional concept-axis Mask of shape ``[K]``) is forwarded
        to the dispatched operator. ``None`` preserves legacy behavior.
        """
        method_name = grammar.rules[rule_id].method_name
        if method_name is None:
            return left  # transition -- pass through

        if method_name in self._RULE_METHODS:
            fn_name, _, binary = self._RULE_METHODS[method_name]
            fn = getattr(self, fn_name)
            if binary:
                if right is not None:
                    return fn(left, right, subspace, mask=mask)
                return left
            return fn(left, subspace, mask=mask)

        return left

    def reverse_project(self, grammar, rule_id, result, right=None, subspace=None,
                        mask=None):
        """Execute a grammar rule inverse. Returns best-effort recovery of left operand."""
        method_name = grammar.rules[rule_id].method_name
        if method_name is None:
            return result

        if method_name in self._RULE_METHODS:
            _, rev_name, binary = self._RULE_METHODS[method_name]
            if rev_name is None:
                return result  # lossy op -- no inverse
            fn = getattr(self, rev_name)
            if binary:
                return fn(result, right, subspace, mask=mask)
            return fn(result, subspace, mask=mask)

        return result

    # -- C-tier: invertible operations -----------------------------

    def notForward(self, left, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.negation(left, monotonic=self._mono(subspace))
        else:
            out = -left
        return self._apply_mask(out, mask, subspace=subspace)

    def notReverse(self, result, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.negation_inverse(result, monotonic=self._mono(subspace))
        else:
            out = -result
        return self._apply_mask(out, mask, subspace=subspace)

    def intersectionForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.conjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.min(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def intersectionReverse(self, result, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.conjunction_inverse(result, right, monotonic=self._mono(subspace))
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    def unionForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.max(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def unionReverse(self, result, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction_inverse(result, right, monotonic=self._mono(subspace))
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    # -- P-tier: chunk (invertible) --------------------------------

    def chunkForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction(left, right, monotonic=True)
        elif right is None:
            out = left
        else:
            out = torch.max(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def chunkReverse(self, result, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction_inverse(result, right, monotonic=True)
        else:
            out = result
        return self._apply_mask(out, mask, subspace=subspace)

    # -- S-tier: lossy operations (no inverse) ---------------------

    def equalsForward(self, left, right, subspace, mask=None):
        """S -> equals(S, S): agreement score via concept-level mutual parthood.

        When called from the S-tier with a wired SymbolicSpace back-reference,
        reverse-project both operands from S to C via the owning
        SymbolicSpace's PiLayer, then delegate to the C-tier Basis.equal
        (mutual parthood, scalar=True) on the bitonic concept subspace.
        Otherwise fall back to the local subspace basis or elementwise min.

        Under a mask, agreement is computed only on the selected dims.
        """
        if mask is not None:
            m = self._expand_mask(mask, left.shape[-1])
            m = m.to(device=left.device, dtype=left.dtype)
            denom = m.sum().clamp(min=1.0)
            agree = 1.0 - ((left - right).abs() * m).sum(dim=-1) / denom
            agree = agree.clamp(0.0, 1.0)
            while agree.ndim < right.ndim:
                agree = agree.unsqueeze(-1)
            return self._apply_mask(agree * right, mask, subspace=subspace)

        sym_space = getattr(self, "_symbolic_space", None)
        concept_space = getattr(sym_space, "conceptualSpace", None) if sym_space else None
        concept_basis = None
        if concept_space is not None:
            c_sub = getattr(concept_space, "subspace", None)
            concept_basis = getattr(c_sub, "basis", None) if c_sub else None
        pi = getattr(sym_space, "layer", None) if sym_space else None

        # Reverse-project only when the operand actually looks like a per-symbol
        # vector (last dim == PiLayer's nOutput). Activations over symbol
        # indices [B, K] fall through to the local subspace basis.
        pi_output_dim = getattr(pi, "nOutput", None) if pi is not None else None
        if (concept_basis is not None
                and pi is not None
                and hasattr(pi, "reverse")
                and pi_output_dim is not None
                and left.shape[-1] == pi_output_dim
                and right.shape[-1] == pi_output_dim):
            left_c = pi.reverse(left)
            right_c = pi.reverse(right)
            score = concept_basis.equal(left_c, right_c, monotonic=False, scalar=True)
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            return score * right

        b = self._basis(subspace)
        if b is not None:
            score = b.equal(left, right, monotonic=self._mono(subspace), scalar=True)
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            return score * right
        return torch.min(left, right)

    def partForward(self, left, right, subspace, mask=None):
        b = self._basis(subspace)
        if b is not None:
            score = b.part(left, right, monotonic=self._mono(subspace), scalar=True)
            while score.ndim < right.ndim:
                score = score.unsqueeze(-1)
            out = score * right
        else:
            out = torch.min(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    # -- S-tier trinity: true / false / non as partition of unity --
    # For x  in  [-1, 1]:  true(x) + false(x) + non(x) = 1
    #   true(x)  = max(0, x)     "I commit: yes"
    #   false(x) = max(0, -x)    "I commit: no"
    #   non(x)   = 1 - |x|       "I commit: indeterminate"
    # Inputs are clamped to [-1, 1] defensively so the partition holds
    # regardless of upstream producer conventions.

    def trueForward(self, left, subspace, mask=None):
        left = torch.clamp(left, -1.0, 1.0)
        b = self._basis(subspace)
        if b is not None:
            out = b.pos(left)
        else:
            out = torch.relu(left)
        return self._apply_mask(out, mask, subspace=subspace)

    def falseForward(self, left, subspace, mask=None):
        """Positive rectification of the negation. The 'no' commitment.

        Partitions with trueForward/nonForward: true + false + non = 1.
        """
        left = torch.clamp(left, -1.0, 1.0)
        b = self._basis(subspace)
        if b is not None:
            out = b.pos(-left)
        else:
            out = torch.relu(-left)
        return self._apply_mask(out, mask, subspace=subspace)

    def nonForward(self, left, subspace, mask=None):
        """Triangular residual: 1 - |x|. The 'indeterminate' commitment.

        Completes the S-tier trinity partition of unity. Replaces the
        earlier sigmoid/zero response which was incompatible with
        true + false + non = 1.
        """
        left = torch.clamp(left, -1.0, 1.0)
        out = 1.0 - left.abs()
        return self._apply_mask(out, mask, subspace=subspace)

    def conjunctionForward(self, left, right, subspace, mask=None):
        """S-tier sentence-level AND. Hadamard conjunction on bitonic activations.

        Distinct from C-tier intersection which composes concepts; this
        composes propositions. Delegates to Basis.conjunction when available
        (which respects sign agreement); falls back to torch.minimum.
        """
        b = self._basis(subspace)
        if b is not None:
            out = b.conjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.minimum(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    def disjunctionForward(self, left, right, subspace, mask=None):
        """S-tier sentence-level OR. Hadamard disjunction on bitonic activations.

        Distinct from C-tier union which composes concepts; this composes
        propositions. Delegates to Basis.disjunction when available
        (which respects sign agreement); falls back to torch.maximum.
        """
        b = self._basis(subspace)
        if b is not None:
            out = b.disjunction(left, right, monotonic=self._mono(subspace))
        else:
            out = torch.maximum(left, right)
        return self._apply_mask(out, mask, subspace=subspace)

    # -- Rule #2: S-tier slot selectors (what / where / when) -----
    # Parameter-free axis projections. Each zeros non-selected column
    # blocks while preserving shape. The C -> S boundary demux has
    # already put the content in the canonical [what|where|when]
    # layout (see SubSpace.demux); these selectors just mask the
    # non-selected blocks when the activation tensor is vector-shaped.
    #
    # When compose() passes [B, N] scalar norms (non-vector mode) the
    # block structure isn't accessible, so selectors degenerate to
    # identity -- the grammar's axis semantics still hold because the
    # selected vs non-selected dimensions are carried by the
    # subspace's modality tensors rather than the [B, N] activation.

    def _split_widths(self, subspace):
        if subspace is None:
            return None, None, None
        nWhat = getattr(subspace, 'nWhat', None)
        nWhere = getattr(subspace, 'nWhere', 0)
        nWhen = getattr(subspace, 'nWhen', 0)
        return nWhat, nWhere, nWhen

    def whatForward(self, left, subspace, mask=None):
        """Axis selector: keep what-block, zero where/when-blocks."""
        if left.ndim < 3:
            return self._apply_mask(left, mask, subspace=subspace)  # scalar mode -- no columns
        nWhat, nWhere, nWhen = self._split_widths(subspace)
        if nWhat is None or (nWhere == 0 and nWhen == 0):
            return self._apply_mask(left, mask, subspace=subspace)
        out = torch.zeros_like(left)
        out[..., :nWhat] = left[..., :nWhat]
        return self._apply_mask(out, mask, subspace=subspace)

    def whereForward(self, left, subspace, mask=None):
        """Axis selector: keep where-block, zero what/when-blocks."""
        if left.ndim < 3:
            return self._apply_mask(left, mask, subspace=subspace)
        nWhat, nWhere, nWhen = self._split_widths(subspace)
        if nWhat is None or nWhere == 0:
            return self._apply_mask(torch.zeros_like(left), mask, subspace=subspace)
        out = torch.zeros_like(left)
        out[..., nWhat:nWhat + nWhere] = left[..., nWhat:nWhat + nWhere]
        return self._apply_mask(out, mask, subspace=subspace)

    def whenForward(self, left, subspace, mask=None):
        """Axis selector: keep when-block, zero what/where-blocks."""
        if left.ndim < 3:
            return self._apply_mask(left, mask, subspace=subspace)
        nWhat, nWhere, nWhen = self._split_widths(subspace)
        if nWhat is None or nWhen == 0:
            return self._apply_mask(torch.zeros_like(left), mask, subspace=subspace)
        out = torch.zeros_like(left)
        out[..., nWhat + nWhere:] = left[..., nWhat + nWhere:]
        return self._apply_mask(out, mask, subspace=subspace)

    # -- forward: predict rules ------------------------------------

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
        # rule logit. The transition rule (S->C or C->P) is the NOP -- "stop
        # deriving, pass through." Low interpretation biases toward it.
        interp = self.grammar.interpretation if self.grammar is not None else 0.5
        transition_bias = (1.0 - interp) * self.TRANSITION_SCALE

        for d in range(self.max_depth):
            h = h + depth_vecs[d]
            h = self.derivation_layer.forward(h)
            h = self.activation_fn(h)
            logits = self.rule_head.forward(h)  # [B, num_rules]

            # Bias the transition rule logit. Detach the bias so it
            # doesn't flow gradients -- interpretation is a hyperparameter,
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

    # -- helpers ----------------------------------------------------

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

    # -- reverse: deterministic tree-walk --------------------------

    def reverse(self, words, nVectors, batch_size):
        """Decode derivation to recover the activation vector."""
        activation = torch.zeros(batch_size, nVectors, device=TheDevice.get())
        for b, v, r in words:
            activation[b, v] = 1.0
        return activation

    # -- utilities -------------------------------------------------

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
        """Lazily create the chunk codebook when we first see P-tier data.

        BPE mode and the minimum pair frequency are read from
        ``PerceptualSpace`` config.  The BPE target vocabulary size is
        derived from the codebook's ``nVectors`` (falling back to
        ``chunkTargetVocabSize`` for legacy configs, then 1024).

        Legacy defaults (``chunkBPE=false``) keep the whitespace-boundary
        behavior unchanged; ``chunkBPE=true`` switches the layer into
        greedy longest-match BPE with a learned merge table that grows
        during ``train_step``.
        """
        if self.chunk_layer is None:
            def _pcfg(key, default):
                try:
                    return TheXMLConfig.space("PerceptualSpace", key)
                except (KeyError, TypeError, ValueError):
                    return default
            bpe = bool(_pcfg("chunkBPE", False))
            # Derive target vocab from codebook nVectors; fall back to
            # legacy chunkTargetVocabSize, then 1024.
            try:
                target_vocab = int(_pcfg("nVectors", 0))
            except (TypeError, ValueError):
                target_vocab = 0
            if target_vocab <= 0:
                try:
                    target_vocab = int(_pcfg("chunkTargetVocabSize", 1024))
                except (TypeError, ValueError):
                    target_vocab = 1024
            try:
                min_pair_freq = int(_pcfg("chunkMinPairFrequency", 2))
            except (TypeError, ValueError):
                min_pair_freq = 2
            self.chunk_layer = ChunkLayer(
                nDim,
                bpe=bpe,
                target_vocab_size=target_vocab,
                min_pair_frequency=min_pair_freq,
            ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')

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

        # Learned BPE loop -- boundary check delegates to ChunkLayer
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

        # Byte mode: compact sparse -> dense word slots
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
      1. not(C) -- flips negative concepts to positive (mean < 0).
      2. Soft superposition -- remaining rules weighted by predicted probs.

    Owns lift/lower layers.
    """

    _RULE_METHODS = {
        **SyntacticLayer._RULE_METHODS,
        'lift':  ('liftForward',  'liftReverse',  True),
        'lower': ('lowerForward', 'lowerReverse', True),
    }

    def init_conceptual_params(self, concept_dim):
        """Initialize C-tier learnable parameters."""
        self.lifting_layer = LiftingLayer(16, concept_dim)
        self.lowering_layer = LoweringLayer(concept_dim)
        self._symbolic_space = None  # set by BasicModel after init

    # -- C-tier projected ops: lift/lower via PiLayer ----------------

    def _cs_layer(self):
        """PiLayer for concept->symbol projection (ss.layer)."""
        if self._symbolic_space is not None:
            return getattr(self._symbolic_space, 'layer', None)
        return None

    def liftForward(self, left, right, subspace, mask=None):
        """Projected conjunction through symbolic space, back to concept space."""
        cs = self._cs_layer()
        if cs is not None:
            s_a = cs.forward(left)
            s_b = cs.forward(right)
            out = cs.reverse(s_a * s_b)
        else:
            out = left * right
        return self._apply_mask(out, mask, subspace=subspace)

    def liftReverse(self, result, right, subspace, mask=None):
        """Recover first operand: s_a = result / s_b, then PiLayer.reverse."""
        cs = self._cs_layer()
        if cs is not None:
            s_res = cs.forward(result)
            s_b = cs.forward(right)
            s_a = s_res / (s_b + epsilon)
            out = cs.reverse(s_a)
        else:
            out = result / (right + epsilon)
        return self._apply_mask(out, mask, subspace=subspace)

    def lowerForward(self, left, right, subspace, mask=None):
        """Projected disjunction through symbolic space, back to concept space.

        Rescale the sum by 1/2 so it stays in ``(-1, 1)`` -- the operand
        domain that ``PiLayer.reverse`` requires.  Both ``s_a`` and
        ``s_b`` are tanh outputs in ``(-1, 1)``; their unscaled sum lies
        in ``(-2, 2)`` which hits the hard clamp inside ``_to_mult`` and
        silently saturates the backward pass, yielding a badly
        conditioned gradient that leaks into every optimizer step via
        the soft rule mixture.  Training instability on deep chains
        (pairwise compose over conceptualOrder iterations) traces back
        to this saturation.  Dividing by 2 is information-preserving
        (the reverse scales by 2) and keeps the operand strictly inside
        the PiLayer domain.
        """
        cs = self._cs_layer()
        if cs is not None:
            s_a = cs.forward(left)
            s_b = cs.forward(right)
            out = cs.reverse((s_a + s_b) / 2)
        else:
            out = (left + right) / 2
        return self._apply_mask(out, mask, subspace=subspace)

    def lowerReverse(self, result, right, subspace, mask=None):
        """Recover first operand from the rescaled lower forward.

        Given ``result = cs.reverse((s_a + s_b) / 2)`` we have
        ``s_res = cs.forward(result) = (s_a + s_b) / 2``, so
        ``s_a = 2 * s_res - s_b``.  For a valid forward pair both
        ``s_a`` and ``s_b`` lie in ``(-1, 1)`` by construction, so
        ``2 * s_res - s_b`` also lies in ``(-1, 1)`` and the PiLayer
        reverse does not clamp.
        """
        cs = self._cs_layer()
        if cs is not None:
            s_res = cs.forward(result)
            s_b = cs.forward(right)
            s_a = 2 * s_res - s_b
            out = cs.reverse(s_a)
        else:
            out = 2 * result - right
        return self._apply_mask(out, mask, subspace=subspace)

    def project(self, grammar, rule_id, left, right=None, third=None, subspace=None,
                mask=None):
        """Execute a rule. Lift/lower are in _RULE_METHODS via super()."""
        return super().project(grammar, rule_id, left, right, third,
                               subspace=subspace, mask=mask)

    def reverse_project(self, grammar, rule_id, result, right=None, subspace=None,
                        mask=None):
        """Inverse dispatch -- delegates to super()."""
        return super().reverse_project(grammar, rule_id, result, right,
                                       subspace=subspace, mask=mask)

    def compose(self, data, subspace, grammar, target_count=None):
        """Apply C-tier composition.

        Args:
            data: [B, N, D] concept tensor
            subspace: SubSpace for word recording
            grammar: Grammar instance for rule execution
            target_count: If set, use pairwise reduction to this token count
                          (hierarchical mode). None uses cascading accumulator.
        Returns:
            (composed_data, svo_or_None) -- svo is set if ternary lift fired
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

            # -- not: negate via Basis.negation (bitonic: -x, self-inverse)
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

        # Identify composable rules (exclude not -- already applied in Phase 1)
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

        Only binary-or-greater rules participate in the pairwise reduce.
        A unary rule by definition cannot merge two operands -- including
        one would make its per-pair output ignore ``right``, and if its
        probability dominates the soft mixture (which happens at init
        for any fresh SyntacticLayer with a biased softmax prior) the
        reduce degenerates to ``composed ~= left`` and ``right``'s
        content is silently discarded.  ``not`` is also excluded since
        it's already applied in Phase 1 of ``compose``.
        """
        B, N, D = data.shape

        # Identify composable rules (exclude not -- already applied in
        # Phase 1 -- AND any rule whose arity is < 2, which can't merge
        # a pair at all; see docstring).
        exclude = {'not'}
        composable_local = []
        composable_global = []
        for local_idx, global_id in enumerate(self.all_rules):
            if grammar.rules[global_id].method_name in exclude:
                continue
            if grammar.arity(global_id) < 2:
                continue
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
            # Dims don't match SyntacticLayer -- use uniform probs
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

        # Return a tensor with the ORIGINAL N shape so downstream
        # (including the reverse path through ``conceptualSpace.reverse``
        # and ``perceptualSpace.reverse``) keeps the slot-axis width it
        # was built for.  Surviving active positions hold the reduced
        # content; consumed positions are left zero.  Compact this down
        # to ``[B, target_count, D]`` at the caller if needed.
        result = torch.zeros(B, N, D, device=data.device)
        for b in range(B):
            for pos in active[b]:
                result[b, pos] = data[b, pos]

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
            return data  # no codebook -- fall back to identity

        result = torch.zeros_like(data)
        for word in words:
            if word[WordEncoding.ORDER] != -1:
                continue  # skip rule words -- only terminals carry leaves
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
        """Initialize swap and non parameters."""
        swap_size = max(symbol_dim, n_symbol_slots, 1)
        self.swap_marker = nn.Parameter(torch.randn(swap_size) * 0.01)
        self.swap_logits = nn.Parameter(torch.zeros(3, 3))
        self._swap_sinkhorn_iters = 5
        self.non_threshold = nn.Parameter(torch.tensor(0.0))
        # Set by WordSpace._build_symbolic_layer so equalsForward can
        # reverse-project S operands back to C and delegate to Basis.equal.
        self._symbolic_space = None

    def _swap_soft_perm(self):
        M = self.swap_logits
        for _ in range(self._swap_sinkhorn_iters):
            M = M - M.logsumexp(dim=-1, keepdim=True)
            M = M - M.logsumexp(dim=-2, keepdim=True)
        return M.exp()

    def swapForward(self, left, right, subspace=None, mask=None):
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
        return self._apply_mask(out[0], mask, subspace=subspace)

    _RULE_METHODS = {
        **SyntacticLayer._RULE_METHODS,
        'swap':        ('swapForward',        None, True),
        # Rule #1: trinity + coordination (S-tier only)
        'false':       ('falseForward',       None, False),
        'conjunction': ('conjunctionForward', None, True),
        'disjunction': ('disjunctionForward', None, True),
        # Rule #2: symbol demux slot selectors (S-tier only)
        'what':        ('whatForward',        None, False),
        'where':       ('whereForward',       None, False),
        'when':        ('whenForward',        None, False),
        # Rule #3: query (contradiction marker at accumulation point)
        'query':       ('queryForward',       None, True),
    }

    # Rule #3: Norm-drop threshold. If a new rule-application result
    # would reduce the accumulator's norm below this fraction of its
    # current value, the accumulation point interprets it as symbolic
    # contradiction and emits a query word + preserves the existing
    # accumulator instead of absorbing the cancelling contribution.
    # Tuning note: start at 0.1 (90% reduction) per plan; too tight
    # emits spurious queries on legitimate near-cancellations, too
    # loose lets real contradictions collapse silently.
    _QUERY_NORM_DROP_RATIO = 0.1

    def queryForward(self, left, right, subspace=None, mask=None):
        """Query: return the preserved accumulator operand.

        The query marker is pushed onto WordSubSpace at the
        accumulation point (see `compose()`), not by this forward.
        When the parse tree is re-evaluated downstream, `queryForward`
        returns the first operand -- the accumulator state that was
        preserved when the cancelling contribution arrived. The second
        operand (the dropped symbol) exists only in the parse-tree
        record and is unused here.
        """
        return self._apply_mask(left, mask, subspace=subspace)

    def project(self, grammar, rule_id, left, right=None, subspace=None, mask=None):
        """Execute a rule via _RULE_METHODS dispatch."""
        return super().project(grammar, rule_id, left, right,
                               subspace=subspace, mask=mask)

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

        # Rule #3 state: track the rule applied at the previous step
        # per batch row, so a query push at the norm-drop site has a
        # referent for "what was the preserved accumulator's rule".
        # -1 = no prior rule (accumulator is still a raw leaf).
        last_rule_per_batch = [-1 for _ in range(B)]
        query_rid = None
        for _idx, _gid in enumerate(all_rules):
            if grammar.rules[_gid].method_name == 'query':
                query_rid = _gid
                break

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

            # -- Rule #3: norm-drop detection at the accumulation point --
            # Any symbolic contradiction (A  and  not A, true(A)  and  false(A),
            # axis-restricted variants) manifests as a significant drop
            # in the accumulator norm when the candidate is mixed in.
            # We detect the symptom here, push a `query` marker onto the
            # word-stream buffer, and preserve the prior accumulator.
            # See plan: "Rule #3 -- Query at S-tier".
            candidate = (probs_d.unsqueeze(-1) * results).sum(dim=1)  # [B, N]
            prev_norm = left.norm(dim=-1)         # [B]
            cand_norm = candidate.norm(dim=-1)    # [B]
            drop_threshold = self._QUERY_NORM_DROP_RATIO * prev_norm
            # Only fire when there was a real accumulator to cancel
            # (prev_norm > 1e-6), and the candidate norm is below the
            # drop threshold.
            query_mask = (prev_norm > 1e-6) & (cand_norm < drop_threshold)
            if query_mask.any():
                # For batches where query fires: preserve the old accumulator.
                # For batches where it does not: use the candidate mixture.
                mask = query_mask.unsqueeze(-1).expand_as(candidate)  # [B, N]
                composed = torch.where(mask, left, candidate)
                # Push query marker onto the word-stream buffer for
                # each batch row that tripped the check. The leaves
                # record the rule identities of the preserved side
                # (left_rule_id) and the incoming rule that would have
                # caused the cancellation (right_rule_id).
                best_for_push = probs_d.argmax(dim=-1)  # [B]
                word_sub = getattr(self, 'word_subspace', None)
                for b in range(B):
                    if not bool(query_mask[b].item()):
                        continue
                    left_rid = last_rule_per_batch[b]
                    right_rid = int(best_for_push[b].item())
                    right_gid = all_rules[right_rid] if right_rid < len(all_rules) else -1
                    if query_rid is not None and word_sub is not None:
                        word_sub.push(b, query_rid,
                                      leaves=(left_rid, right_gid, -1))
                    # Preserve the prior rule identity for this batch
                    # row -- the accumulator did not advance.
            else:
                composed = candidate

            # Record argmax rule as word
            best = probs_d.argmax(dim=-1)  # [B]
            for b in range(B):
                if d < len(active_positions[b]):
                    subspace.add_word(b, active_positions[b][d], all_rules[best[b].item()])
                    # Track last advancing rule per batch for future
                    # query-push referents -- only update for rows
                    # that did not trip the query mask this step.
                    if not bool(query_mask[b].item()):
                        last_rule_per_batch[b] = all_rules[best[b].item()]

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
class WordSpace(Space):
    """Service space that owns the word-stream buffer, the SyntacticLayers,
    the truth store, and the inter-sentence discourse substrate.

    Runtime-parallel to PerceptualSpace / ConceptualSpace / SymbolicSpace
    but functionally a buffer + composition dispatcher. WordSpace owns
    the SyntacticLayers directly; home spaces receive ``wordSpace`` as
    a per-call parameter on ``forward(vspace, wordSpace=...)`` /
    ``reverse(vspace, wordSpace=...)`` and reach the layers via the
    explicit per-tier methods ``forwardPercepts`` / ``forwardConcepts``
    / ``forwardSymbols`` (and the matching ``reverse*`` variants). The
    layers push their word records into ``self.subspace`` (a
    ``WordSubSpace``) via a back-reference set at construction time, so
    ConceptualSpace can read a muxed view of machine state that
    includes percepts, symbols, and words.

    One unified constructor builds everything: WordSubSpace, all three
    SyntacticLayers, TruthLayer, and (conditionally) DiscourseSpace.
    XML config drives the truth-store capacity and discourse-prediction
    gating.

    Per-sentence lifecycle: BasicModel calls ``clear_sentence()`` at
    sentence boundaries to rewind the buffer.

    Subclasses ``Space`` for the universal training contract
    (``getParameters`` / ``paramUpdate`` / ``set_sigma``), but
    bypasses ``Space.__init__`` because there is no factory-style
    input/output/codebook shape tuple -- the subspace is a
    ``WordSubSpace`` built from the symbolic peer's column layout
    and all children are registered directly into ``self.layers`` /
    ``self.params`` so the inherited training-contract walks still
    work.
    """

    name = "Words"
    config_section = "WordSpace"

    def __init__(self, perceptualSpace, conceptualSpace, symbolicSpace,
                 nPercepts, nConcepts, nSymbols,
                 concept_dim, symbol_dim):
        # Bypass Space.__init__ -- WordSpace doesn't fit the factory
        # style. Call nn.Module directly and populate the Space-contract
        # fields by hand.
        nn.Module.__init__(self)

        # 1. Grammar must be configured before any SyntacticLayer
        # construction can resolve rule sets / transition rules.
        TheGrammar._configured = False
        TheGrammar._ensure_configured()
        grammar = TheGrammar

        # 2. Size WordSubSpace from SymbolicSpace's subspace column
        # layout so downstream consumers of wordSpace.read() concat
        # cleanly with peer tensors.
        sub = symbolicSpace.subspace
        nWhere = int(getattr(sub, 'nWhere', 0) or 0)
        nWhen  = int(getattr(sub, 'nWhen',  0) or 0)
        nWhat  = int(getattr(sub, 'nWhat',  0) or 0)
        muxed  = int(getattr(sub, 'muxedSize', nWhat + nWhere + nWhen)
                     or (nWhat + nWhere + nWhen))
        self.subspace = WordSubSpace(
            nDim=muxed, nWhat=nWhat, nWhere=nWhere, nWhen=nWhen,
            max_depth=256, max_arity=3, batch=1,
        )

        # 3. Space-contract fields.
        self.layers = nn.ModuleList()
        self.params = []
        self.wordSpace = None                        # no parent wordSpace
        self.nDim = muxed
        self.nWhat = nWhat
        self.nWhere = nWhere
        self.nWhen = nWhen
        self.muxedSize = muxed
        self.inputShape  = [0, muxed]
        self.outputShape = [0, muxed]
        self.spaceShape  = [0, muxed]

        # 4. Layer slots (filled below).
        self.perceptualSyntacticLayer = None
        self.conceptualSyntacticLayer = None
        self.symbolicSyntacticLayer = None

        # 5. Build the three SyntacticLayers, each of which back-wires
        # the home space's ``wordSpace`` routing pointer.
        if perceptualSpace is not None:
            self._build_perceptual_layer(perceptualSpace, nPercepts, grammar)
        if conceptualSpace is not None:
            self._build_conceptual_layer(
                conceptualSpace, nConcepts, grammar, concept_dim)
        if symbolicSpace is not None:
            self._build_symbolic_layer(
                symbolicSpace, nSymbols, grammar, symbol_dim)

        # 6. TruthLayer -- shared truth store for symbolic activations.
        # Lives on WordSpace so SymbolicSpace doesn't have to carry it
        # alongside its already heavy pi/sort/codebook machinery.
        try:
            max_truths = int(TheXMLConfig.get("WordSpace.truthMaxEntries"))
        except (KeyError, TypeError, ValueError):
            max_truths = 1024
        self.truth_layer = TruthLayer(symbol_dim, max_truths=max_truths)
        if self.truth_layer not in self.layers:
            self.layers.append(self.truth_layer)
        for p in self.truth_layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

        # 7. DiscourseSpace -- optional inter-sentence substrate.
        # Gated on <architecture><training><sentencePrediction>; tasks
        # without inter-sentence structure (XOR, MNIST) leave it off.
        # The contrastive loss has no learnable parameters; the three
        # training keys that survive are ``sentenceContextWindow``
        # (recent buffer depth used for the attractive centroid),
        # ``sentenceCentroidHistory`` (older centroids used for the
        # repulsive force), and ``sentenceLambda`` (repulsive scale).
        self.discourse = None
        if bool(TheXMLConfig.training("sentencePrediction", False)):
            try:
                n_sym_rows = int(symbolicSpace.outputShape[0])
            except (AttributeError, IndexError, TypeError):
                n_sym_rows = int(getattr(symbolicSpace, 'nVectors', 0) or 0)
            if n_sym_rows > 0 and muxed > 0:
                context_window = int(TheXMLConfig.training(
                    "sentenceContextWindow", 12) or 12)
                centroid_history = int(TheXMLConfig.training(
                    "sentenceCentroidHistory", 3) or 3)
                sentence_lambda = float(TheXMLConfig.training(
                    "sentenceLambda", 1.01) or 1.01)
                self.discourse = InterSentenceLayer(
                    n_symbols=n_sym_rows,
                    max_depth=int(getattr(
                        self.subspace, 'max_depth', 256) or 256),
                    n_dim=muxed,
                    context_window=context_window,
                    centroid_history=centroid_history,
                    lam=sentence_lambda,
                    concept_dim=int(concept_dim),
                )
                self.layers.append(self.discourse)
                for p in self.discourse.parameters():
                    if all(p is not q for q in self.params):
                        self.params.append(p)

    # -- truth-modulated loss -----------------------------------------
    def truth_modulated_loss(self, total_loss, symbolic_space,
                             symbol_acts=None, universality_score=None,
                             luminosity_weight=0.1, universality_weight=0.1,
                             truth_loss_weight=0.0,
                             allow_excluded_middle=1,
                             allow_contradiction=0,
                             balance_weight=0.1):
        """Apply the WordSpace-owned TruthLayer modulation to a loss.

        The transform has two parts:

        1. **Multiplicative modulation** -- penalize irrational and
           unkind propositions by scaling ``total_loss`` by
           ``(1 + lum_w * (1 - lum_norm) + u_w * (1 - u_norm))``,
           where ``lum_norm = luminosity(symbolic_space.layer).clamp(0, 1)``
           and ``u_norm = universality_score.clamp(-1, 1)`` (or 0
           when the caller has no universality score cached yet).

        2. **Additive falsity penalty** -- when
           ``truth_loss_weight > 0`` and the caller provides
           committed symbol activations, add
           ``truth_loss_weight * falsity_penalty(symbol_acts, basis)``
           using ``symbolic_space.subspace.basis``.  ``symbol_acts``
           should be the last entry of the model's ``symbol_states``
           cache -- the post-pi activations from the final Sigma-Pi
           iteration.  Both operands of the
           disjunction then live in symbol space by construction
           (stored truths were also recorded from symbol-space
           activations in ``SymbolicSpace.forwardEnd``).

        Returns ``total_loss`` unchanged when the TruthLayer is
        absent or empty (bootstrap case with no truths recorded
        yet).  The caller is responsible for only invoking this in
        train mode -- the method itself has no ``train`` flag.

        All inputs that reach outside WordSpace (``symbolic_space``,
        ``symbol_acts``, ``universality_score``, the three weights)
        are passed explicitly so WordSpace never needs a back-
        reference to the model.
        """
        if self.truth_layer is None or len(self.truth_layer) == 0:
            return total_loss

        lum = self.truth_layer.luminosity(symbolic_space.layer)
        lum_norm = lum.clamp(0, 1)
        if universality_score is not None:
            u_norm = universality_score.clamp(-1, 1)
        else:
            u_norm = torch.tensor(0.0, device=total_loss.device)

        total_loss = total_loss * (1 + luminosity_weight * (1 - lum_norm)
                                     + universality_weight * (1 - u_norm))

        if truth_loss_weight > 0 and symbol_acts is not None:
            basis = getattr(
                getattr(symbolic_space, 'subspace', None), 'basis', None)
            if basis is not None:
                truth_penalty = self.truth_layer.falsity_penalty(
                    symbol_acts, basis)
                total_loss = total_loss + truth_loss_weight * truth_penalty

        # Quaternary-corner balance penalty: discourages forbidden corners
        # (N, B) on committed symbol activations. Runs whenever the knobs
        # select a non-permissive corner and symbol_acts are provided.
        # Under the current SymbolicSpace layout each row is
        # [pos_pole, neg_pole, where..., when...] -- slice the leading
        # bivector before passing to the paired-index penalty so that
        # positional-template dims don't spuriously register as N/B.
        # See basicmodel/doc/BuddhistParallels.md and doc/Spaces.md.
        wants_balance = (int(allow_excluded_middle) == -1
                         or int(allow_contradiction) == 0)
        if (balance_weight > 0 and wants_balance
                and symbol_acts is not None
                and torch.is_tensor(symbol_acts)
                and symbol_acts.shape[-1] >= 2):
            bivector = symbol_acts[..., :2]
            balance = self.truth_layer.tetralemma_balance_penalty(
                bivector,
                allow_excluded_middle=int(allow_excluded_middle),
                allow_contradiction=int(allow_contradiction))
            total_loss = total_loss + balance_weight * balance

        return total_loss

    # -- wiring -------------------------------------------------------
    def attach_codebook_host(self, host):
        """Wire the unified rule codebook provider to WordSubSpace.

        `host` must expose `lookup_rule(rule_id)` returning a dense
        vector for the identity at that rule_id. Typically the
        `SymbolicSpace` instance, whose `rule_codebook` is the single
        source of truth for rule identity vectors.
        """
        self.subspace.attach_codebook_host(host)

    def attach_layer(self, kind, layer):
        """Register a pre-built SyntacticLayer as this WordSpace's
        ``<kind>SyntacticLayer``.

        Sets ``layer.word_subspace`` as a back-reference so compose()
        can push onto the shared buffer, appends the layer to
        ``self.layers`` for ``Space.paramUpdate`` delegation, and
        merges its parameters into ``self.params`` for the curated
        ``Space.getParameters`` walk.
        """
        if layer is None:
            return
        attr = f'{kind}SyntacticLayer'
        if not hasattr(self, attr):
            raise ValueError(
                f"WordSpace: unknown syntactic kind {kind!r}; "
                f"expected one of 'perceptual', 'conceptual', 'symbolic'")
        setattr(self, attr, layer)
        layer.word_subspace = self.subspace
        if layer not in self.layers:
            self.layers.append(layer)
        for p in layer.parameters():
            if all(p is not q for q in self.params):
                self.params.append(p)

    # -- private factory helpers: build + wire SyntacticLayers --------
    def _resolve_hidden_dim(self, n_slots):
        try:
            configured = int(TheXMLConfig.get("WordSpace.syntacticHiddenDim"))
            if configured > 0:
                return configured
        except (KeyError, TypeError, ValueError):
            pass
        return min(256, max(64, n_slots * 4))

    def _build_perceptual_layer(self, space, n_slots, grammar):
        layer = PerceptualSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.perceptual(),
            transition_rule=None,
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
        )
        self.attach_layer('perceptual', layer)
        space.attach_wordSpace(self)
        return layer

    def _build_conceptual_layer(self, space, n_slots, grammar, concept_dim):
        layer = ConceptualSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.conceptual(),
            transition_rule=grammar.conceptual_transition(),
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
        )
        layer.init_conceptual_params(concept_dim)
        self.attach_layer('conceptual', layer)
        space.attach_wordSpace(self)
        return layer

    def _build_symbolic_layer(self, space, n_slots, grammar, symbol_dim):
        layer = SymbolicSyntacticLayer(
            nInput=n_slots, nOutput=n_slots,
            rules=grammar.symbolic(),
            transition_rule=grammar.symbolic_transition(),
            max_depth=max(n_slots - 1, 1),
            hidden_dim=self._resolve_hidden_dim(n_slots),
            grammar=grammar,
        )
        layer.init_swap(symbol_dim, n_slots)
        layer._symbolic_space = space
        space.init_rule_codebook(grammar)
        self.attach_codebook_host(space)
        self.attach_layer('symbolic', layer)
        space.attach_wordSpace(self)
        return layer

    # -- per-tier composition methods ---------------------------------
    def forwardPercepts(self, data, subspace):
        """P-tier compose. Side effect: word-emitting pushes onto the
        buffer. Returns the composed activation.
        """
        layer = getattr(self, 'perceptualSyntacticLayer', None)
        if layer is None:
            return data
        return layer.compose(data, subspace, TheGrammar)

    def forwardConcepts(self, data, subspace, target_count=None):
        """C-tier compose. ``ConceptualSyntacticLayer.compose`` may
        return ``(data, svo)`` when a ternary lift fires; we preserve
        that tuple contract so callers (ConceptualSpace.forward) can
        stash the SVO on themselves for the ``last_svo`` property.

        ``target_count`` routes into the pairwise reduction path in
        ``ConceptualSyntacticLayer._compose_to_target``.  Pairwise
        reduction slices each slot to ``[1, 1, D]`` before invoking
        the grammar's binary rules, which degenerates the per-slot
        PiLayer to a pure ``D->D`` map and lets the two operands'
        content actually merge into one slot.  The cascading default
        (``target_count=None``) keeps full ``[B, N, D]`` shapes and so
        cannot move information across the slot axis -- fine for
        sparse-representation use but useless whenever the two
        operands live in different slots.
        """
        layer = getattr(self, 'conceptualSyntacticLayer', None)
        if layer is None:
            return data, None
        result = layer.compose(data, subspace, TheGrammar, target_count=target_count)
        if isinstance(result, tuple):
            return result
        return result, None

    def forwardSymbols(self, data, subspace):
        """S-tier compose. Includes the Rule #2 demux side effect: the
        muxed [B, N, D] symbol tensor gets split into what/where/when
        modality slots before compose runs, so slot selectors see
        axis-separated state.
        """
        layer = getattr(self, 'symbolicSyntacticLayer', None)
        if layer is None:
            return data
        if data.ndim == 3 and data.shape[-1] == subspace.muxedSize:
            subspace.demux(data)
        return layer.compose(data, subspace, TheGrammar)

    def reversePercepts(self, data, subspace):
        layer = getattr(self, 'perceptualSyntacticLayer', None)
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    def reverseConcepts(self, data, subspace):
        layer = getattr(self, 'conceptualSyntacticLayer', None)
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    def reverseSymbols(self, data, subspace):
        layer = getattr(self, 'symbolicSyntacticLayer', None)
        if layer is None:
            return data
        return layer.decompose(data, subspace, TheGrammar)

    # -- buffer access + lifecycle ------------------------------------
    def read(self):
        """Return the fixed-width stack tensor for ConceptualSpace to
        concat with percepts and symbols.
        """
        return self.subspace.read()

    def clear_sentence(self):
        """Reset the stack at sentence boundaries."""
        self.subspace.clear()

    def get_blocks(self, b=0):
        """Return the parse-tree ledger for batch row `b`."""
        return self.subspace.get_blocks(b)

    def ensure_batch(self, batch):
        """Resize the underlying buffer to match a new batch size."""
        self.subspace.ensure_batch(batch)
