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
# vector_quantize_pytorch API Codebook uses. Reimplements the three
# features we care about on the non-library path: EMA codebook updates,
# dead-code replacement, and the rotation-trick STE (arXiv:2410.06424).
# Codebook can still opt into the real library via ``use_library_vq=True``;
# this class exists so the repo behaves equivalently when the external
# package is absent.
try:
    from vector_quantize_pytorch import VectorQuantize as LibraryVectorQuantize
except Exception:
    LibraryVectorQuantize = None
_vq_F = F


def _reset_call(reset_fn, batch=None, hard=True):
    """Call a Reset callable with the per-row signature, falling back
    to the legacy zero-arg signature when the callee does not accept
    ``batch`` / ``hard``.

    Used by Space.Reset cascades so older Layer / SubSpace classes that
    haven't adopted the per-row Reset(batch=b, hard=...) signature keep
    working unchanged. Falls back to ``reset_fn()`` (legacy semantics)
    on TypeError; callers see a global wipe instead of a per-row clear,
    which is a no-op-equivalent superset of the requested action.
    """
    try:
        return reset_fn(batch=batch, hard=hard)
    except TypeError:
        return reset_fn()
class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        commitment_weight=1.0,
        use_cosine_sim=False,
        decay=0.8,
        threshold_ema_dead_code=0,
        rotation_trick=False,
        eps=1e-5,
        codebook_retire=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.use_cosine_sim = use_cosine_sim
        self.decay = float(decay)
        self.threshold_ema_dead_code = int(threshold_ema_dead_code)
        self.rotation_trick = bool(rotation_trick)
        self.eps = float(eps)
        # Gate for the dead-code replacement path. Off by default because
        # reseeding expired rows with fresh samples can blow up the effective
        # number of distinct codes on non-stationary data.
        self.codebook_retire = bool(codebook_retire)
        self.codebook = torch.randn(codebook_size, dim)
        # EMA accumulators used by ``update_ema``. ``cluster_size`` counts
        # how many inputs snap to each code (bootstrapped at 1.0 to match
        # the library default so the first-step smoothed divisor is never
        # zero). ``embed_avg`` is the running sum of the assigned inputs;
        # the codebook is rewritten as ``embed_avg / cluster_size`` (with
        # laplace smoothing) each training step.
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", self.codebook.data.clone())
        # Owner-space tag for error messages (set post-construction by the
        # owning Space via ``_tag_vq_name``).
        self.name = ""
        # Per-instance row-chunk override for the cdist/argmax intermediate.
        # ``None`` => derive from ``_VQ_CHUNK_TARGET_ELEMS`` at forward time.
        self._vq_chunk_rows = None

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
        # Keep EMA buffers consistent when the codebook is reassigned
        # externally (e.g. the post-init rescale in Codebook.addVectors).
        # Reset embed_avg to the new values; reshape cluster_size if the
        # codebook size changed.
        if "embed_avg" in self._buffers:
            self._buffers["embed_avg"] = param.data.clone()
        if "cluster_size" in self._buffers:
            cs = self._buffers["cluster_size"]
            if cs.shape[0] != param.shape[0]:
                self._buffers["cluster_size"] = torch.ones(
                    param.shape[0], device=cs.device, dtype=cs.dtype
                )

    def _sync_ema_buffers(self):
        """Repair EMA buffers when they drift from the live codebook shape.

        Training can reassign the codebook Parameter after construction
        (for example when resizing or reloading weights). The EMA buffers
        must stay row-aligned with that Parameter or the dead-code refresh
        path can end up indexing the wrong rows.
        """
        codebook = self.codebook
        V, D = int(codebook.shape[0]), int(codebook.shape[1])
        cluster_size = self.cluster_size
        if (
            cluster_size.ndim != 1
            or int(cluster_size.shape[0]) != V
            or cluster_size.device != codebook.device
        ):
            dtype = cluster_size.dtype if cluster_size.is_floating_point() else torch.float32
            self._buffers["cluster_size"] = torch.ones(
                V, device=codebook.device, dtype=dtype
            )
        embed_avg = self.embed_avg
        if (
            embed_avg.ndim != 2
            or tuple(embed_avg.shape) != (V, D)
            or embed_avg.device != codebook.device
        ):
            dtype = embed_avg.dtype if embed_avg.is_floating_point() else codebook.dtype
            self._buffers["embed_avg"] = codebook.detach().to(dtype=dtype).clone()

    @staticmethod
    def _rotate_to(src, tgt):
        """Rotation-trick STE from arXiv:2410.06424 -- forwards ``tgt`` but
        rotates the upstream gradient from ``tgt``'s direction back to
        ``src``'s direction and rescales by ``||tgt|| / ||src||`` so
        magnitude is preserved. Mirrors ``vector_quantize_pytorch``'s
        ``rotate_to`` / ``efficient_rotation_trick_transform``.
        """
        eps = 1e-8
        orig_shape = src.shape
        e = src.reshape(-1, src.shape[-1])
        q = tgt.reshape(-1, tgt.shape[-1])
        norm_e = e.norm(dim=-1, keepdim=True).clamp(min=eps)
        norm_q = q.norm(dim=-1, keepdim=True).clamp(min=eps)
        e_hat = e / norm_e
        q_hat = q / norm_q
        w = e_hat + q_hat
        w = w / w.norm(dim=-1, keepdim=True).clamp(min=eps)
        w = w.detach()
        e_dot_w = (e * w).sum(dim=-1, keepdim=True)
        e_dot_u = (e * e_hat.detach()).sum(dim=-1, keepdim=True)
        out = e - 2.0 * e_dot_w * w + 2.0 * e_dot_u * q_hat.detach()
        scale = (norm_q / norm_e).detach()
        return (out * scale).reshape(orig_shape)

    # Default byte budget for one VQ distance/similarity tile.  4 GiB
    # leaves ample headroom on 64 GB unified-memory accelerators (the
    # AMD Strix Halo target) while keeping the per-tile allocation well
    # below per-buffer caps on current GPU stacks.  Small VQs (V < 64K)
    # never trigger chunking on first call.  Override per-instance via
    # ``vq._vq_chunk_rows``.
    _VQ_CHUNK_TARGET_BYTES = 4 * (1 << 30)  # 4 GiB

    def forward(self, x, return_all_codes=False, freeze_codebook=False, **kwargs):
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        codebook = self.codebook
        N = flat.shape[0]
        V = codebook.shape[0]
        D = codebook.shape[1]
        gib = (N * V * 4) / (1024 ** 3)  # float32 bytes -> GiB

        # Chunk over rows of `flat`.  The [chunk, V] intermediate
        # (cdist or matmul) is the dominant allocation; the EMA path
        # below avoids the [N, V] one_hot entirely via bincount +
        # index_add_, so chunk size is bounded only by this matrix.
        chunk = self._vq_chunk_rows
        if chunk is None:
            max_pairs = self._VQ_CHUNK_TARGET_BYTES // 4  # float32 bytes
            chunk = max(1, max_pairs // max(V, 1))
        chunk = min(chunk, N) if N > 0 else 1

        # ``indices`` is the argmin/argmax of distances; the gather that
        # follows (codebook[indices]) carries no gradient back through the
        # selection, so the entire indices computation can run under
        # ``no_grad`` to skip saving cdist/matmul intermediates for the
        # backward pass.  EMA update has its own ``no_grad`` block below.
        try:
            with torch.no_grad():
                if self.use_cosine_sim:
                    cb_cmp = _vq_F.normalize(codebook, dim=-1)
                    if N <= chunk:
                        flat_cmp = _vq_F.normalize(flat, dim=-1)
                        indices = (flat_cmp @ cb_cmp.T).argmax(dim=-1)
                    else:
                        parts = []
                        for s in range(0, N, chunk):
                            sub_cmp = _vq_F.normalize(flat[s:s+chunk], dim=-1)
                            parts.append((sub_cmp @ cb_cmp.T).argmax(dim=-1))
                        indices = torch.cat(parts, dim=0)
                else:
                    if N <= chunk:
                        indices = torch.cdist(flat, codebook).argmin(dim=-1)
                    else:
                        parts = []
                        for s in range(0, N, chunk):
                            parts.append(
                                torch.cdist(flat[s:s+chunk], codebook).argmin(dim=-1))
                        indices = torch.cat(parts, dim=0)
        except RuntimeError as e:
            owner = self.name or "<unnamed VQ>"
            raise RuntimeError(
                f"VectorQuantize[{owner}]: distance matrix allocation "
                f"failed even after chunking. flat={tuple(flat.shape)}, "
                f"codebook={tuple(codebook.shape)}, chunk={chunk}, "
                f"pairwise matrix = [{N}, {V}] float32 = {gib:.2f} GiB. "
                f"Reduce {owner}.nVectors, reduce batchSize, or set "
                f"vq._vq_chunk_rows to a smaller value. Original error: {e}"
            ) from e
        quantized_raw = codebook[indices].reshape(original_shape)
        commit_loss = self.commitment_weight * _vq_F.mse_loss(
            x, quantized_raw.detach()
        )

        # EMA codebook update + dead-code replacement. Without these the
        # codebook gets no gradient (commit_loss detaches the quantized
        # side and STE is a no-op for the codes) so codes never track the
        # data distribution. Mirrors ``EuclideanCodebook.update_ema`` and
        # ``expire_codes_`` in vector_quantize_pytorch.
        if self.training and not freeze_codebook:
            with torch.no_grad():
                self._sync_ema_buffers()
                flat_f = flat.float()
                # Equivalent to ``one_hot(indices, V).t() @ flat_f`` but
                # without ever allocating the [N, V] one-hot tensor: at
                # body-scale microbatch (N in the millions) that matrix
                # alone passes the GPU per-buffer cap.
                cluster_size_batch = torch.bincount(
                    indices, minlength=V
                ).to(flat_f.dtype)
                embed_sum = torch.zeros(
                    V, D, device=flat_f.device, dtype=flat_f.dtype,
                )
                embed_sum.index_add_(0, indices, flat_f)
                self.cluster_size.mul_(self.decay).add_(
                    cluster_size_batch.to(self.cluster_size.dtype),
                    alpha=1.0 - self.decay,
                )
                self.embed_avg.mul_(self.decay).add_(
                    embed_sum.to(self.embed_avg.dtype),
                    alpha=1.0 - self.decay,
                )
                n = self.cluster_size.sum()
                cs_smooth = (
                    (self.cluster_size + self.eps)
                    / (n + V * self.eps)
                    * n
                )
                new_embed = self.embed_avg / cs_smooth.unsqueeze(-1)
                if self.use_cosine_sim:
                    new_embed = _vq_F.normalize(new_embed, dim=-1)
                self.codebook.copy_(new_embed.to(self.codebook.dtype))
                if self.codebook_retire and self.threshold_ema_dead_code > 0:
                    expired = (
                        self.cluster_size < self.threshold_ema_dead_code
                    ).reshape(-1)
                    if int(expired.numel()) != V:
                        raise RuntimeError(
                            "VectorQuantize EMA state corrupt: "
                            f"expired mask has {int(expired.numel())} rows "
                            f"for codebook size {V}"
                        )
                    expired_idx = torch.nonzero(
                        expired, as_tuple=False
                    ).flatten()
                    if int(expired_idx.numel()) > 0:
                        n_expired = int(expired_idx.numel())
                        sample_idx = torch.randint(
                            0, flat.shape[0], (n_expired,), device=flat.device,
                        )
                        sampled = flat[sample_idx].to(self.codebook.dtype)
                        if self.use_cosine_sim:
                            sampled = _vq_F.normalize(sampled, dim=-1)
                        self.codebook.index_copy_(0, expired_idx, sampled)
                        thr = float(self.threshold_ema_dead_code)
                        self.cluster_size.index_fill_(0, expired_idx, thr)
                        self.embed_avg.index_copy_(
                            0,
                            expired_idx,
                            sampled.to(self.embed_avg.dtype) * thr,
                        )

        # Gradient estimator for the quantized output: rotation trick when
        # requested and the input carries gradient, otherwise vanilla STE.
        if self.rotation_trick and self.training and x.requires_grad:
            quantized = self._rotate_to(x, quantized_raw)
        else:
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
from Layers import Layer, PiLayer, SigmaLayer  # Import custom layers from Model.py
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, LiftingLayer, LoweringLayer, ChunkLayer
from Layers import ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon, Ops
from Layers import SortingLayer, TruthLayer, InterSentenceLayer, SparsityRegularizer, SmoothingRegularizer, ImpenetrableLayer
from Layers import Error
from util import parse
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
        from Language import TheGrammar
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

    # -- Text-token layout on the What basis ------------------------------
    #
    # When an InputSpace runs in embedding (text) mode, the lexer's output
    # is packed into the What basis as null-terminated UTF-8 bytes. The
    # buffer has shape [batch, nObj, nWhat]: within each slot, up to
    # ``nWhat - 1`` bytes hold the token and the final byte is reserved
    # for the 0x00 terminator. These two helpers are the single writer /
    # reader pair for that layout -- InputSpace.forward uses encode_tokens
    # to write the buffer, PerceptualSpace._embed uses
    # decode_tokens to read it back.
    def encode_tokens(self, tokens_per_batch, batch, nObj, nWhat, device):
        """Pack token strings into a [B, N, nWhat] null-terminated byte buffer.

        ``tokens_per_batch[b][i]`` is the UTF-8 string for slot i of batch
        row b (empty string = empty slot). Tokens longer than
        ``nWhat - 1`` bytes are truncated; a trailing 0 byte terminates
        each slot.
        """
        buf = torch.zeros(batch, nObj, nWhat, dtype=torch.long, device=device)
        for b, row in enumerate(tokens_per_batch):
            for i in range(min(len(row), nObj)):
                text = row[i]
                if not text:
                    continue
                raw = text.encode('utf-8')[: nWhat - 1]
                for j, byte in enumerate(raw):
                    buf[b, i, j] = byte
        return buf

    def decode_tokens(self, buf):
        """Unpack a [B, N, nWhat] null-terminated byte buffer into strings.

        Returns a list[list[str]] of shape [B][N]. Slots whose first byte
        is 0 decode to ``""``. Invalid UTF-8 sequences are replaced.
        """
        batch = buf.shape[0]
        n_obj = buf.shape[1]
        out = []
        for b in range(batch):
            row = []
            for n in range(n_obj):
                bytes_row = buf[b, n].tolist()
                try:
                    end = bytes_row.index(0)
                except ValueError:
                    end = len(bytes_row)
                if end == 0:
                    row.append("")
                else:
                    row.append(
                        bytes(bytes_row[:end]).decode('utf-8', errors='replace'))
            out.append(row)
        return out
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
        return Ops.norm(x)

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
    # Formulas live in Ops (Layers.py). Basis only owns codebook-search
    # inverses (which need self.getW()).

    def conjunction(self, x, y, monotonic=False):
        return Ops.conjunction(x, y, monotonic=monotonic)

    def disjunction(self, x, y, monotonic=False):
        return Ops.disjunction(x, y, monotonic=monotonic)

    def negation(self, x, monotonic=False):
        return Ops.negation(x, monotonic=monotonic)

    def non(self, x, monotonic=False, threshold=None):
        return Ops.non(x, monotonic=monotonic, threshold=threshold)

    # -- Inverse logic operations -----------------------------------------------

    def _codebook_or_none(self, label):
        try:
            return self.getW()
        except (NotImplementedError, AttributeError):
            warnings.warn(f"{label}: no codebook available", stacklevel=3)
            return None

    def negationReverse(self, x, monotonic=False):
        """Inverse of negation. Self-inverse in both modes.
        Bitonic: sign flip. Monotonic: paired-index flip."""
        return Ops.negationReverse(x, monotonic=monotonic)

    def conjunctionReverse(self, result, y, monotonic=False):
        """Inverse of conjunction via codebook search.

        Find the codebook vector x such that conjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        W = self._codebook_or_none("conjunctionReverse")
        if W is None:
            return result
        return Ops.conjunctionReverse(result, y, W, monotonic=monotonic)

    def disjunctionReverse(self, result, y, monotonic=False):
        """Inverse of disjunction via codebook search.

        Find the codebook vector x such that disjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        W = self._codebook_or_none("disjunctionReverse")
        if W is None:
            return result
        return Ops.disjunctionReverse(result, y, W, monotonic=monotonic)

    # -- Synthesis / analysis dispatchers (lift / lower) ------------------
    # Thin wrappers around Ops.lift / Ops.lower that supply the codebook W
    # for the inverse (codebook-search) paths.  See Ops.lift / Ops.lower
    # for the mode dispatch and region semantics.

    def lift(self, X1, X2=None, mode='OR', kind='strict', inverse=False,
             monotonic=False):
        """Synthesis dispatcher: many → one (∨).

        Forward routes through Ops.lift.  `kind` selects the point body
        (strict/smooth/radial); see Ops.lift for details.  Inverse with
        mode='OR' or 'AND' runs codebook-search recovery via self.getW();
        inverse with mode='NOT' is self-inverse and bypasses W.
        """
        if inverse and mode == 'OR':
            W = self._codebook_or_none("Basis.lift inverse")
            if W is None:
                return X1
            return Ops.disjunctionReverse(X1, X2, W, monotonic=monotonic)
        if inverse and mode == 'AND':
            W = self._codebook_or_none("Basis.lift inverse")
            if W is None:
                return X1
            return Ops.conjunctionReverse(X1, X2, W, monotonic=monotonic)
        return Ops.lift(
            X1, X2, mode=mode, kind=kind, inverse=inverse, monotonic=monotonic
        )

    def lower(self, X1, X2=None, mode='AND', kind='strict', inverse=False,
              monotonic=False):
        """Analysis dispatcher: one → many (∧).

        Forward routes through Ops.lower.  `kind` selects the point body
        (strict/smooth/radial); see Ops.lower for details.  Inverse with
        mode='AND' or 'OR' runs codebook-search recovery via self.getW();
        inverse with mode='NOT' is self-inverse and bypasses W.
        """
        if inverse and mode == 'AND':
            W = self._codebook_or_none("Basis.lower inverse")
            if W is None:
                return X1
            return Ops.conjunctionReverse(X1, X2, W, monotonic=monotonic)
        if inverse and mode == 'OR':
            W = self._codebook_or_none("Basis.lower inverse")
            if W is None:
                return X1
            return Ops.disjunctionReverse(X1, X2, W, monotonic=monotonic)
        return Ops.lower(
            X1, X2, mode=mode, kind=kind, inverse=inverse, monotonic=monotonic
        )

    def pos(self, x):
        return Ops.pos(x)

    def distance(self, x, y, monotonic=False, dim=-1):
        return Ops.distance(x, y, monotonic=monotonic, dim=dim)

    def codebookDistance(self, x):
        weight = self._prototype_weight(context="codebookDistance")
        vec = weight[:, :self.nDim].to(TheDevice.get())
        return x @ vec.T / max(self.nDim, 1)

    # -- Mereological operations --------------------------------------------
    # Formulas live in Ops (Layers.py); see Ops docstrings for semantics.

    def part(self, x, y, monotonic=False, scalar=False):
        return Ops.part(x, y, monotonic=monotonic, scalar=scalar)

    def whole(self, x, y, monotonic=False, scalar=False):
        return Ops.whole(x, y, monotonic=monotonic, scalar=scalar)

    def equal(self, x, y, monotonic=False, scalar=False):
        return Ops.equal(x, y, monotonic=monotonic, scalar=scalar)

    def overlap(self, x, y, monotonic=False, scalar=False):
        return Ops.overlap(x, y, monotonic=monotonic, scalar=scalar)

    def underlap(self, x, y, monotonic=False, scalar=False):
        return Ops.underlap(x, y, monotonic=monotonic, scalar=scalar)

    def boundary(self, x, y, monotonic=False, scalar=False):
        return Ops.boundary(x, y, monotonic=monotonic, scalar=scalar)

    def copart(self, x, y, monotonic=False, scalar=False):
        return Ops.copart(x, y, monotonic=monotonic, scalar=scalar)

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
    """Dense tensor payload implementation used for ordinary SubSpace slots.

    ``W`` may hold a permanent ``nn.Parameter`` (weights owned by this basis)
    OR a plain tensor (a transient activation). To keep the two roles from
    clobbering each other across lifecycle calls, plain-tensor writes are
    routed to ``_active_payload`` whenever a Parameter is registered -- the
    same dual-slot pattern ``Codebook`` uses. Callers should always read via
    ``getW()`` (which prefers the transient payload when set) rather than
    touching ``self.W`` directly.
    """

    def __init__(self, nVectors=0, nDim=0, W=None):
        super().__init__()
        self.W = None
        self._active_payload = None
        self.nVectors = nVectors
        self.nDim = nDim
        if W is not None:
            self.W = W
        else:
            self.W = torch.zeros(nVectors, nDim)

    def getW(self):
        if self._active_payload is not None:
            return self._active_payload
        return self.W

    def setW(self, value):
        """Assign W without ever clobbering a registered Parameter.

        value=None clears only the transient activation when a Parameter is
        held; otherwise it also clears the plain-tensor W. A Parameter write
        replaces any existing W (Parameter or plain) and drops the transient.
        A plain-tensor write lands on ``_active_payload`` when a Parameter is
        registered, and on ``W`` otherwise.
        """
        if value is None:
            self._active_payload = None
            if "W" not in self._parameters:
                self.W = None
            return
        if isinstance(value, nn.Parameter):
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = value
            self._active_payload = None
            return
        if "W" in self._parameters:
            self._active_payload = value
            return
        self.W = value
        self._active_payload = None

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
        # When True, ``addVectors`` instantiates ``vector_quantize_pytorch``'s
        # VectorQuantize (which does EMA codebook updates and dead-code
        # replacement). Default False keeps the in-repo snap-only stub.
        # Read from ``architecture.useLibraryVQ`` so an XML config can flip
        # it on without editing code.
        try:
            cfg_flag = TheXMLConfig.get("architecture.useLibraryVQ", False)
        except Exception:
            cfg_flag = False
        self.use_library_vq = bool(cfg_flag)
        self.snapDistance = 0.1
        self.eta = 0.9
        self.alpha = 0.0
        self.codebookSize = 0
        self.vq = None
        # Latest commitment loss from the most recent forward/quantize pass.
        # SymbolicSpace.forward reads it and emits "codebook_commit" into
        # vspace.errors.
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
            if "W" not in self._parameters:
                self.W = None
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
            if self.use_library_vq and LibraryVectorQuantize is not None:
                vq_cls = LibraryVectorQuantize
            else:
                vq_cls = VectorQuantize
            # When ``architecture.codebookRetire`` is false (the default),
            # dead-code replacement is disabled on both the in-repo class
            # and the library: retired rows get reseeded with fresh samples,
            # which on non-stationary data can blow up the effective code
            # count. Keep the two signals in sync by also zeroing the
            # threshold so the library path takes the no-op branch.
            try:
                retire = bool(TheXMLConfig.get("architecture.codebookRetire", False))
            except Exception:
                retire = False
            vq_kwargs = dict(
                dim=self.nDim,
                codebook_size=nVec,
                threshold_ema_dead_code=1 if retire else 0,
                decay=decay,
                commitment_weight=1.0,
                # Rotation-trick STE (arXiv:2410.06424) was reported to
                # trigger gradient-accumulation OOMs on HIP/ROCm with
                # microbatch AR.  Vanilla STE
                # (x + (quantized - x).detach()) is the classical VQ-VAE
                # estimator -- trains fine, uses a smaller autograd graph.
                rotation_trick=False,
            )
            if vq_cls is VectorQuantize:
                vq_kwargs["codebook_retire"] = retire
            self.vq = vq_cls(**vq_kwargs)
            # The external library does its own codebook initialization;
            # the in-repo stub uses raw torch.randn so we rescale to [-1, 1]
            # to satisfy downstream range checks on the 'what' field.
            if vq_cls is VectorQuantize:
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

            # Previous code walked this top-k set in Python and used .item()
            # for row/codebook ids, forcing device sync per token. Keep the
            # same "best row per codebook entry per batch" contract in tensor
            # form: ignore rows outside the top-k error set, then reduce by
            # minimum error for each codebook id.
            selected = torch.zeros_like(err, dtype=torch.bool)
            selected.scatter_(1, indices_smallest, True)
            selected_err = torch.where(
                selected, err, torch.full_like(err, float("inf")))
            best_err = torch.full(
                (batch, self.codebookSize),
                float("inf"),
                device=err.device,
                dtype=err.dtype,
            )
            best_err.scatter_reduce_(
                1, indices, selected_err, reduce="amin", include_self=True)

            claimed = selected & (err <= best_err.gather(1, indices))
            x3d = torch.where(claimed.unsqueeze(-1), quantized3d, x3d)

            snap = claimed & (err <= self.snapDistance)
            scores = self.distance(input3d, quantized3d[..., :self.nDim])
            if self.alpha:
                scores = scores + self.alpha * torch.rand_like(scores)
            batch_idx = torch.arange(
                batch, device=indices.device).unsqueeze(1).expand_as(indices)
            act[batch_idx[snap], indices[snap]] = scores[snap]
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
        self.activation = act.detach() if torch.is_tensor(act) else act
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
            return parse(self._to_text(text), lex='bytes')
        if getattr(self, 'lexer_mode', 'word') == 'sentence':
            return parse(self._to_text(text), lex='sentences')
        mode = getattr(self, 'chunking_mode', 'lexicon')
        if mode == 'bpe':
            return self._char_stream(text)
        return parse(self._to_text(text), lex='words')

    def _char_stream(self, text):
        """Tokenize text as per-character units with positional indices."""
        return [(ch, i) for i, ch in enumerate(self._to_text(text))]

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
        # Flatten any leading/spurious unit dims so vec is 1-D [D].
        vec = vec.reshape(-1)
        # Compare only over the shared width. The model's output feature
        # width can be narrower than the codebook width (e.g. content-only
        # 4 vs codebook 6 with positional dims); we match on the shared
        # prefix rather than raising on the width mismatch.
        d = min(vec.shape[0], codebook.shape[1])
        sims = F.cosine_similarity(vec[:d].unsqueeze(0), codebook[:, :d], dim=1)
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
        # Collapse any spurious singleton dims so vectors ends up 3-D
        # [B, N, D]. Model output can arrive as [B, N, 1, D] from certain
        # OutputSpace configurations; the inner unit dim is vestigial.
        while vectors.ndim > 3 and 1 in vectors.shape[2:-1]:
            for ax in range(2, vectors.ndim - 1):
                if vectors.shape[ax] == 1:
                    vectors = vectors.squeeze(ax)
                    break
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
        # Phase 1: default normalizer uses the global TheData. Model
        # construction overrides this with its own Normalizer instance
        # (so tests can mock TheData via a different source).
        # SubSpaces instantiated standalone in tests get the global-backed
        # Normalizer by default — no None-dispatch required.
        from Models import Normalizer as _Normalizer
        self.normalizer = _Normalizer(TheData)
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

        # Pipeline-carried context. These travel with the subspace through
        # every Space.forward via copy_context(), replacing the old pattern
        # of cross-stage and cross-forward back-channels on Space instances.
        #   wordSpace    -- Model's WordSpace reference (stamped by InputSpace)
        #   errors       -- per-batch auxiliary-loss accumulator; SymbolicSpace
        #                   writes symbol_commitment / codebook_commit / etc.
        #                   here and runBatch folds them into TheError.
        #   serial_cache -- {id(owner_space): tensor} for serial-mode warm cache
        self.wordSpace = None
        self.errors = Error()
        self.serial_cache = {}

        # Microbatch-AR contract (Task 5 of 2026-04-22 plan).
        # k_axis=True means the event tensor has shape [B, K, N, D]
        # (stem output before the body's FlattenKWrapper folds K into B).
        # valid_mask: [B, K] bool, True where the window's target
        # token is non-NULL. Built by the stem; consumed by runBatch.
        self.k_axis = False
        self.valid_mask = None
        # stem_embedded=True signals downstream stages that InputSpace has
        # already performed lex+embed in the microbatch AR path; the body's
        # PerceptualSpace must skip its own _embed (which would clobber the
        # K-windowed event with a fresh [B, N, D] re-embed of the original
        # byte buffer).
        self.stem_embedded = False

        payload = self.materialize()
        if isinstance(payload, torch.Tensor) and payload.ndim > 0:
            self.batch = payload.shape[0]

    def copy_context(self, other):
        """Adopt cross-stage/cross-forward state from ``other``.

        Pipeline invariant: every ``Space.forward`` that returns a subspace
        other than the incoming ``vspace`` must first ``copy_context(vspace)``
        so ``wordSpace``, ``errors``, and ``serial_cache`` travel unbroken
        through the pipeline.  ``errors`` and ``serial_cache`` are carried
        by reference so later writes (e.g., ``SymbolicSpace.forward`` adding
        a commitment term) land in the same accumulator that
        ``OutputSpace`` / ``runBatch`` will read.

        Also propagates the microbatch-AR routing attrs (``k_axis``,
        ``valid_mask``, ``stem_embedded``).  These are stem-route
        contracts that stages downstream of FlattenKWrapper currently do
        not read, but propagation keeps the contract explicit and prevents
        a stale value on a recycled subspace from misrouting the body.
        """
        if other is None:
            return
        self.wordSpace = other.wordSpace
        self.errors = other.errors
        self.serial_cache = other.serial_cache
        self.k_axis = other.k_axis
        self.valid_mask = other.valid_mask
        self.stem_embedded = other.stem_embedded

    @property
    def basis(self):
        """Return the primary content Basis (what) for this SubSpace."""
        return self.what

    @property
    def is_demuxed(self):
        """True when what/where/when are stored independently (not muxed into event)."""
        return self._demuxed

    def is_empty(self):
        """True when this SubSpace has no work to do.

        Pipeline modules short-circuit when this returns True (no codebook
        inserts, no category_stack pushes, no side effects). Checks shapes
        directly from the underlying Basis tensors to avoid the cost and
        side effects of materialize() (which gates the event by activation
        presence and can trip on transient shape mismatches).
        """
        # inputShape is authoritative when its N is 0.
        if self.inputShape[0] == 0:
            return True
        # Otherwise look at whichever tensor is currently populated.
        for source in (self.event, self.what):
            if source is None:
                continue
            try:
                w = source.getW()
            except Exception:
                continue
            if w is None:
                continue
            if w.ndim >= 2 and (w.shape[0] == 0 or w.shape[1] == 0):
                return True
            if w.ndim >= 1 and w.numel() > 0:
                return False
        return False

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

    @staticmethod
    def _clear_runtime_basis(basis):
        """Release transient per-batch tensors cached on a Basis.

        Relies on each Basis subclass to guard its own permanent state:
        Codebook preserves its codebook Parameter on setW(None), Embedding
        makes setW a no-op, and Tensor routes plain-tensor writes through
        a transient ``_active_payload`` slot when a Parameter is registered
        (see Tensor.setW). So it is safe to call this on any Basis role
        without erasing weights.
        """
        if basis is None:
            return
        if hasattr(basis, "clearActivation"):
            basis.clearActivation()
        if hasattr(basis, "setW"):
            basis.setW(None)

    def Start(self):
        """Release all per-batch tensors cached on the SubSpace.

        Clears runtime state on every owned Basis (event, what, where,
        when, activation) plus the SubSpace-level transient fields
        (``_active`` index tensor, ``word`` list, ``valid_mask`` mask,
        ``stem_embedded`` flag). The prior version cleared only ``event``,
        which leaked the demuxed [B, N, D] tensors on what/where/when and
        the [B, N, M] ``_active`` tensor until the next forward overwrote
        them. Called from Space.Start() so state carried across the outer
        pos loop does not leak into the next DataLoader yield.
        """
        for basis in (self.event, self.what, self.where, self.when, self.activation):
            self._clear_runtime_basis(basis)
        self._active = None
        self.word = []
        self.valid_mask = None
        self.stem_embedded = False

    def End(self):
        """Release all per-batch tensors cached on the SubSpace.

        Counterpart to Start(); same semantics. Called from Space.End() so
        cached per-batch state is released before the next batch begins.
        """
        for basis in (self.event, self.what, self.where, self.when, self.activation):
            self._clear_runtime_basis(basis)
        self._active = None
        self.word = []
        self.valid_mask = None
        self.stem_embedded = False

    def Reset(self, batch=None, hard=True):
        """Per-document teardown. Drops the serial-mode warm cache so
        the next forward starts cold.

        ``batch`` (optional int): when set, clear only the slice of
        per-row state owned by source row ``batch``. ``None`` clears
        every row (legacy global-Reset semantics).

        ``hard`` (bool, default True): when True this is a document-
        boundary reset (full state wipe). When False this is a soft
        reset triggered by a sentence-internal grammar boundary; the
        SubSpace itself has no per-row sentence state, so soft resets
        are no-ops here.

        ``errors`` and ``wordSpace`` persist (per-batch, per-document
        respectively; both owned by MentalModel lifecycle, not per-Reset).
        Serial cache is per-tick warm state and applies to the whole
        batch by construction; per-row clears still drop it because the
        cache rebuilds cheaply on the next tick.
        """
        if not hard:
            return
        # batch=None and per-row both invalidate the cache: the cache
        # entries are tensors keyed on owner-space id, not per-row, and
        # the next forward rebuilds them.
        self.serial_cache.clear()

    # Legacy alias. New code should use Start() / End() for symmetry
    # with Space.Start() / Space.End() and Layer.Start() / Layer.End().
    reset_event = End

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
        # When demuxed, a 2D tensor here is the Codebook Parameter (a
        # row-per-prototype codebook), not a cached muxed event. Treat it
        # as "no cached event" so we rebuild from the modality indices.
        if self._demuxed and x is not None and x.ndim < 3:
            x = None

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
        # When k_axis=True, the event has a leading [B, K, ...] shape that
        # the [B, N] presence tensor cannot broadcast into. Stem windows
        # are emitted pre-gated (zero-padded for absent positions), so
        # skip the multiply in that case.
        if not self.k_axis:
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
                return self.normalizer.normalize(x, which="input")
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
            return self.normalizer.denormalize(x, which="input")
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
            self.put(target, self.normalizer.normalize(x, which="output"))

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

        row 0:             rule-identity row (zeros; rule_id in the ledger)
        rows 1..max_arity: leaf-identity rows (zeros; leaf ids in the ledger)

    Unused leaf slots produce zero-valued rows (empty-slot sentinel --
    consumers detect empties via ``row_what.norm() == 0``). Rows beyond
    top-of-stack are plain zeros. The learnable rule_codebook has been
    removed (Task 1.4): rule identity is stored only in the parse-tree
    ledger (``_blocks``), not as a dense vector in the buffer.

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

        # Back-reference to the host space (typically SymbolicSpace).
        # Stored via `object.__setattr__` so the host (an nn.Module) is
        # NOT registered as an nn.Module child -- that would create a
        # parent/child cycle (`symbolicSpace <-> wordSpace.subspace`) and
        # make `model.to(device)` recurse forever. Set by
        # `attach_codebook_host()`; WordSpace wires this at construction.
        # Used only as a gate in `push()` -- rule-identity vectors are
        # no longer looked up here (rule_codebook removed, Task 1.4).
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
        the model later moves to MPS / CUDA.
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

    # -- host + batch lifecycle -----------------------------------
    def attach_codebook_host(self, host):
        """Register the host space (typically SymbolicSpace).

        The host is used only as a gate: ``push()`` is a no-op when no
        host is wired. Rule-identity dense vectors are no longer looked
        up here (the learnable rule_codebook was removed in Task 1.4);
        rule identity is preserved in the parse-tree ledger only.

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

    def clear_rows(self, start, end):
        """Zero rows ``[start, end)`` of the stack and rewind their tops.

        Per-row hard reset entry point (called from
        ``WordSpace.Reset(batch=b, hard=True)`` once the source row's
        K-window mapping has been resolved). Out-of-range entries are
        silently clipped to the current batch size.
        """
        s, e = int(start), min(int(end), self.batch)
        if e <= s:
            return
        what_W = self.what.getW() if self.nWhat > 0 else None
        where_W = self.where.getW() if self.nWhere > 0 else None
        when_W = self.when.getW() if self.nWhen > 0 else None
        for W in (what_W, where_W, when_W):
            if W is not None:
                W[s:e].zero_()
        # Invalidate the cached muxed event so materialize() rebuilds.
        if self.event is not None:
            self.event.setW(None)
        if self._top is not None:
            self._top[s:e] = 0
        for b in range(s, e):
            self._blocks[b] = []

    # -- push / read -------------------------------------------------
    def _lookup(self, rule_id):
        """Return a ``[nWhat]`` dense vector for a rule_id, or ``None``.

        The learnable rule_codebook was removed in Task 1.4. This method
        always returns ``None``, leaving the rule-identity row as the
        zero empty-slot sentinel. Rule identity is preserved in the
        parse-tree ledger (``_blocks``) instead.
        """
        return None

    def push(self, b, rule_id, leaves):
        """Append a ``(1 + max_arity)``-row block to batch row ``b``.

        All rows in the block are written as zeros (rule-identity and
        leaf-identity vectors are the empty-slot sentinel since the
        learnable rule_codebook was removed in Task 1.4). Rule identity
        and leaf ids are preserved in the parse-tree ledger (``_blocks``).
        When peer ``nWhere > 0``, each row gets a monotonic derivation-step
        index in its ``.where`` block. Overflow beyond ``max_depth`` silently
        drops the push.

        Writes go through the inherited ``Basis.setW()`` /
        ``Basis.getW()`` contract -- ``self.what.getW()`` returns a
        mutable reference to the ``[batch, max_depth, nWhat]`` tensor
        allocated by ``_allocate_storage``, and ``push()`` mutates that
        in place. After mutation we invalidate the cached ``.event``
        tensor so the next ``materialize()`` rebuilds the muxed view.

        Args:
            b: batch index.
            rule_id: 0-based grammar rule id. Recorded in the ledger.
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
        # Phase 1: set by the model after construction. Spaces call
        # self.normalizer.{normalize,denormalize} instead of reaching into
        # the TheData global.
        self.normalizer = None
        # Serial-mode flag: when True, PerceptualSpace/ConceptualSpace
        # may take the slide-and-recompute fast path. Propagated by
        # BaseModel.create_from_config.
        self.serial_mode = False
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

        self.reversible   = True
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self._codebook    = TheXMLConfig.space(section, "codebook")
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

        # Tag each owned VQ with "SectionName.role" so VectorQuantize.forward's
        # OOM message can name the offending codebook ("PerceptualSpace.what"
        # rather than an anonymous buffer-size traceback).
        for _role in ("object", "what", "where", "when", "activation"):
            _basis = getattr(self.subspace, _role, None)
            _vq = getattr(_basis, "vq", None) if _basis is not None else None
            if _vq is not None:
                _vq.name = f"{self.config_section}.{_role}"

        # wordSpace is still held as a non-Module pointer so the few
        # call sites that reach across to ``wordSpace.truth_layer``
        # (SymbolicSpace) keep working; composition dispatch is no
        # longer done here -- home spaces take ``wordSpace`` as a
        # per-call parameter and call ``wordSpace.forwardSymbols`` /
        # ``.reverseSymbols`` explicitly.
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
    def codebook(self):
        """Shared learned codebook flag. Reference is immutable (no setter);
        the Parameter inside it is updated by the optimizer each step."""
        return self._codebook

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

    def Start(self):
        """One-shot per-run initialization.

        Called from MentalModel.Start() once at the top of model.forward(),
        before the pipeline's iteration loop. Subclasses override to
        initialize per-run state (e.g., InputSpace resets its streaming
        cursor and sliding buffer here).
        """
        for layer in self.layers:
            if hasattr(layer, 'Start'):
                layer.Start()
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'Start'):
            sub.Start()

    def Reset(self, batch=None, hard=True):
        """Per-document teardown. Cascades to child layers and subspace.

        ``batch`` (optional int): when set, clear only the per-row state
        for source row ``batch``. ``None`` keeps the legacy global-Reset
        semantics (clear everything). Layers / subspaces that don't
        understand the per-row signal silently fall back to a global
        clear so legacy paths stay correct.

        ``hard`` (bool, default True): True = document boundary; False =
        sentence-internal soft reset. The base cascade forwards both flags
        to children; subclasses interpret as needed.

        Subclasses override to add their own resets (buffer clears, cursor
        resets, etc.) and must call super().Reset(batch=batch, hard=hard)
        first.
        """
        for layer in self.layers:
            if hasattr(layer, 'Reset'):
                _reset_call(layer.Reset, batch=batch, hard=hard)
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'Reset'):
            _reset_call(sub.Reset, batch=batch, hard=hard)

    def End(self):
        """Per-batch teardown. Counterpart to Start() at the end of the
        outer Run().

        Cascades End() to child layers and subspace so cached per-batch
        state does not persist across batch boundaries. Subclasses with
        additional per-batch caches override this and call super().End()
        first.
        """
        for layer in self.layers:
            if hasattr(layer, 'End'):
                layer.End()
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'End'):
            sub.End()
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
        """InputSpace .what holds non-lexical bases (Codebook/Tensor only).

        The Embedding (lexicon) is owned by PerceptualSpace -- see
        PerceptualSpace._build_what_basis.
        """
        if self.model_type == "embedding":
            return None  # owned by PerceptualSpace.subspace.what
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
        self.lexer = lexer  # "word", "sentence", or "byte"
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
        self.doc_spans = []
        self.doc_sources = []
        # Text mode: Reference to PerceptualSpace so InputSpace.forward()
        # can invoke _embed to produce the muxed target embedding
        # (InputSpace no longer owns the codebook).
        self._peer_perceptual = None

        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = False
        # Pipeline-seed: the Model's WordSpace reference. InputSpace is
        # the only stage that creates subspaces from raw input, so it
        # stamps this onto every outgoing subspace and downstream stages
        # propagate via copy_context. Registered by BaseModel.create_from_config
        # once WordSpace exists.
        self._model_wordSpace = None
        # End-of-stream sentinel. Initialized to False (scalar) before
        # any forward() call. Microbatch AR forward() promotes it to a
        # [B] bool where each row is True when that row's targets are
        # entirely NULL. runBatch reduces via .all() for the per-sentence
        # Reset cascade, then clears it back to False.
        self._end_of_stream = False
        # Raw sentence strings stashed by prepInput for MentalModel.forward()
        # to compute the outer-loop iteration count N without rethreading
        # inp_items through runBatch's call signature.
        self._last_sentences = None
        # ARIR-inference cache bypass: the runtime state machine stages a
        # pre-built embedding here; forward() consumes it once instead of
        # re-lexing. AR training does not use this.
        self._cached_embedding = None
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
        # Stash raw sentence strings for the AR outer loop in
        # MentalModel.forward() -- lets it compute N without rewiring
        # inp_items through runBatch's call signature.
        if (isinstance(inputBatch, list) and inputBatch
                and isinstance(inputBatch[0], str)):
            self._last_sentences = list(inputBatch)
        else:
            self._last_sentences = None

        if isinstance(inputBatch, list):
            tensors = [self.data.stringTensor(s) if isinstance(s, str) else s
                       for s in inputBatch]
            return torch.stack(tensors, dim=0).unsqueeze(1).to(TheDevice.get())
        return inputBatch  # already [B, D, 1] and on device after toDevice()

    def set_word_space(self, ws):
        """Register the Model's WordSpace so the pipeline carries it downstream.

        ``forward()`` stamps ``self._model_wordSpace`` onto every outgoing
        subspace so ConceptualSpace/SymbolicSpace stages can read
        ``vspace.wordSpace`` without reaching back through a Model
        back-channel.  Empty-return sentinels are stamped too so
        skip-on-empty logic downstream still carries the reference.
        """
        self._model_wordSpace = ws
        # Pre-stamp our own subspace; regular returns hand it out unchanged.
        if self.subspace is not None:
            self.subspace.wordSpace = ws

    def prep_sentence_batch(self, sentences):
        """Turn a tuple/list of B sentence strings into a [B, nVec, 1] tensor.

        Thin wrapper over ``prepInput`` so the streaming path can convert a
        sentence tuple directly without any batch-cursor bookkeeping.
        """
        return self.prepInput(list(sentences))

    def _lex_batch(self, input):
        """Tokenize a raw byte tensor into null-terminated UTF-8 byte slots.

        Pure lexer -- no codebook access, no OOV discovery, no index
        resolution. Those live on PerceptualSpace.

        Returns: (what_buf, where_idx, when_idx)
          what_buf: [B, nObj, nWhat] long tensor of UTF-8 bytes, null-terminated.
            Each slot holds one token's bytes followed by a null; tokens longer
            than nWhat-1 bytes are truncated. Empty/padding slots are all-zero.
          where_idx: [B, nObj] long tensor of byte offsets into the source.
          when_idx:  [B, nObj] long tensor of sequential positions.

        Requires self._peer_perceptual to be wired (BasicModel/MentalModel do
        this) because the tokenizer (_token_stream) currently lives on the
        peer's vocabulary.
        """
        assert self._peer_perceptual is not None, \
            "InputSpace._lex_batch requires _peer_perceptual (lexer owner)"
        vocab = self._peer_perceptual.vocabulary
        dev = TheDevice.get()

        if input.dim() == 3:
            input = input.squeeze(1)
        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch = input.shape[0]
        nObj = self.outputShape[0]
        nWhat = self.subspace.nWhat

        tokens_per_batch = []
        where_idx = torch.zeros(batch, nObj, dtype=torch.long, device=dev)
        when_idx = torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1).contiguous()

        for b in range(batch):
            stream = vocab._token_stream(input[b])
            # Reserve one slot for an explicit empty-word EOS sentinel so
            # every sentence terminates with the null-encoding -- the AR
            # generative loop reads this as its stop signal.
            n_tokens = min(len(stream), nObj - 1)
            row = []
            for i in range(n_tokens):
                token_text, start = stream[i]
                row.append(token_text)
                where_idx[b, i] = start
            row.append("")  # empty-word EOS sentinel -> null_idx
            tokens_per_batch.append(row)
            if n_tokens > 0:
                last_text, last_start = stream[n_tokens - 1]
                final_offset = last_start + len(last_text.encode('utf-8'))
            else:
                final_offset = 0
            for i in range(n_tokens, nObj):
                where_idx[b, i] = final_offset + (i - n_tokens)

        what_buf = self.subspace.whatEncoding.encode_tokens(
            tokens_per_batch, batch, nObj, nWhat, dev)

        return what_buf, where_idx, when_idx

    def shuffle(self):
        self.data.shuffle()

    def Start(self):
        """One-shot per-run init."""
        super().Start()
        self._ar_embedded = None
        self._ar_total = 0

    def Reset(self, batch=None, hard=True):
        """Per-document teardown for AR state.

        ``batch`` (optional int): clear per-row state only for source
        row ``batch``. ``None`` clears every row. The AR caches stored
        on InputSpace (``_ar_embedded``, ``_ar_total``,
        ``_cached_embedding``) are batch-shared rebuildable scratch from
        the next forward, so per-row clears still drop them — the next
        forward repopulates from the live input.

        ``hard`` (default True): True is the document boundary; False is
        a sentence-internal soft reset and is a no-op for InputSpace
        (no per-sentence state lives here).
        """
        super().Reset(batch=batch, hard=hard)
        if not hard:
            return
        self._ar_embedded = None
        self._ar_total = 0
        self._cached_embedding = None
        # _end_of_stream is a host-side scalar/list under the rolling-cursor
        # contract (next_tick returns the host-side hard_eos list directly);
        # reset to False so any straggler diagnostic read sees a clean state.
        if batch is None:
            self._end_of_stream = False
        elif isinstance(self._end_of_stream, list):
            if 0 <= batch < len(self._end_of_stream):
                self._end_of_stream[batch] = False
        elif torch.is_tensor(self._end_of_stream):
            if 0 <= batch < self._end_of_stream.shape[0]:
                self._end_of_stream[batch] = False

    @property
    def batch_advances_sentence(self):
        """Host-side: does the loader guarantee a sentence boundary this batch?

        Legacy property kept for backward compat with the predecessor
        handoff (per-row-AR-no-eos-sync). Under the rolling-cursor
        design this gate is no longer consulted by the runEpoch outer
        loop — Reset is dispatched per-row from ``next_tick``'s
        ``hard_eos`` signal. Returns True so any remaining call site
        keeps its prior behavior (Reset every batch).

        A future cleanup PR can delete the property once every reader
        has migrated to the per-row Reset path.
        """
        return True

    # The world presenting itself
    def forward(self, inputData):
        """Single-call stem source for the microbatch AR pipeline.

        Lexes/embeds the input once and emits ALL K windows in a single
        subspace whose event has shape [B, K, N, D] with k_axis=True.

        Semantics:
          * Non-AR: K=1 single-window pass-through (subspace event keeps
            its existing [B, N, D] shape; k_axis stays False).
          * AR training: K = T (one prediction per token via
            progressive-prefix windows from a pad-N-then-unfold).
          * ARIR inference: each call feeds an N-length buffer (T==N),
            so K=1 and the head produces one prediction per call. The
            ARIR runtime maintains the buffer across calls.

        Sets self._end_of_stream as a [B] bool — True for rows whose
        targets are entirely NULL (empty sentence). ``self.subspace``
        carries valid_mask: [B, K] bool for runBatch.
        """
        if inputData is None:
            return self._empty_like_subspace()
        if hasattr(inputData, "is_empty") and not isinstance(inputData, torch.Tensor) and inputData.is_empty():
            return inputData

        is_ar = self.masked_prediction in ('ARLM', 'ARUS', 'ARIR') if hasattr(self, 'masked_prediction') else False
        is_runtime_arir = (
            self.data is not None
            and self.data._runtime_mode == 'ARIR'
        )

        if not is_ar:
            # Non-AR: pass through unchanged. No K axis. Cascade
            # ensure_microbatch(B, 1) so the body's per-row state is sized
            # to match the input batch -- this is the K=1 degenerate of
            # the microbatch contract.
            sub = self._lex_and_embed(inputData)
            event = sub.materialize()
            ws = self._model_wordSpace
            if ws is not None and event is not None and event.dim() >= 1:
                ws.ensure_microbatch(int(event.shape[0]), 1)
            return sub

        # Lex/embed once -- produces _ar_embedded of shape [B, T, D].
        self._lex_and_embed(inputData)
        embedded = self.subspace.materialize()
        if embedded is None and self.model_type == "embedding":
            peer = self._peer_perceptual
            if peer is not None:
                peer._embed(self.subspace)
                embedded = peer._embedded_input
        self._ar_embedded = embedded

        if embedded is None:
            return self.subspace

        B, T, D = embedded.shape
        N = int(self.outputShape[0])

        if is_runtime_arir:
            # ARIR inference path. Buffer is already the N-length window.
            if T < N:
                pad = torch.zeros(B, N - T, D,
                                  device=embedded.device, dtype=embedded.dtype)
                embedded = torch.cat([embedded, pad], dim=1)
                T = N
            windows = embedded.unfold(1, N, 1).permute(0, 1, 3, 2).contiguous()
            K = windows.shape[1]
            # Inference always "valid"; runtime decides termination via predictions.
            valid_mask = torch.ones(B, K, dtype=torch.bool, device=embedded.device)
        else:
            # AR training. Pad N zeros on the LEFT to recreate the legacy
            # cursor-based progressive-prefix windows: window k = the
            # buffer state at cursor k (k zero-pad slots followed by
            # emb[0..k-1] right-aligned). Take first K=T windows so each
            # window k targets emb[k]; the unused (K=T+1)-th window
            # would have no ground-truth target.
            pad = torch.zeros(B, N, D, device=embedded.device, dtype=embedded.dtype)
            padded = torch.cat([pad, embedded], dim=1)  # [B, T+N, D]
            unfolded = padded.unfold(1, N, 1).permute(0, 1, 3, 2).contiguous()
            # unfolded shape: [B, T+1, N, D] -- take first T windows.
            K = T
            windows = unfolded[:, :K, :, :]
            # Validity mask: target at window k is embedded[:, k, :].
            # NULL = all-zero embedding (lex pads short sentences this way).
            valid_mask = (embedded.abs().sum(dim=-1) > 0)  # [B, T] = [B, K]

        sub = self.subspace
        sub.set_event(windows)
        sub.k_axis = True
        sub.valid_mask = valid_mask
        sub.stem_embedded = True

        # Per-row end-of-stream: True for rows with no valid windows.
        self._end_of_stream = ~valid_mask.any(dim=1)  # [B] bool

        # Cascade ensure_microbatch(B, K) so the body's per-row state
        # (subspace, stacks, last_svo) is sized to B*K. _stm_fired stays
        # at B; discourse buffers also size to B*K. This must happen
        # before the body runs so its FlattenKWrapper sees correctly
        # sized substrates.
        ws = self._model_wordSpace
        if ws is not None:
            ws.ensure_microbatch(B, K)

        return sub

    def _empty_like_subspace(self):
        """Return a SubSpace with materialized shape [B, 0, D] — the termination sentinel."""
        template = self.subspace
        ss = SubSpace(
            (0, template.inputShape[1]),
            (0, template.outputShape[1]),
            nInputDim=template._nInputDim,
            nOutputDim=template._nOutputDim,
        )
        # Stamp wordSpace so downstream skip-on-empty logic still sees it.
        ss.wordSpace = self._model_wordSpace
        return ss

    def _lex_and_embed(self, input):
        """Populate subspace from raw input: lex/embed for text, vocab lookup for numeric.

        Called by forward() at the start of a stream (first AR call or
        the entire non-AR pass). Handles three cases:
          * _cached_embedding set (ARIR inference) -- use pre-built latent.
          * model_type == 'embedding' (text) -- lex into byte buffer.
          * numeric mode -- vocab codebook lookup.
        """
        # ARIR-inference cache bypass: the ARIR runtime state machine
        # injects a pre-built embedding tensor via ``_cached_embedding`` so
        # forward() skips lex/embed and uses the staged latent directly.
        # AR training does not use this -- it lexes/embeds each call and
        # builds progressive-prefix windows via unfold.
        if self._cached_embedding is not None:
            self.input = self._cached_embedding
            self._cached_embedding = None  # consume once
            self._forward_input = None
            self.subspace.set_event(self.input)
            return self.subspace

        self.subspace.whereEncoding.p = 0

        if self.model_type == "embedding":
            # Text mode: InputSpace is a pure lexer. Pack tokens as
            # null-terminated UTF-8 bytes into subspace.what.W,
            # byte offsets into subspace.where.W, sequential positions
            # into subspace.when.W. PerceptualSpace decodes the buffer
            # and owns all codebook work (OOV, insert, index resolution,
            # embedding lookup, chunking).
            what_buf, where_idx, when_idx = self._lex_batch(input)
            self.subspace.what.setW(what_buf)
            self.subspace.where.setW(where_idx)
            self.subspace.when.setW(when_idx)
            self._forward_input = None
            return self.subspace

        # Non-text path: vocab is Codebook/Tensor.
        vocab = self.subspace.what
        batch = input.shape[0]
        nObj = self.outputShape[0]
        dev = TheDevice.get()
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
        self.subspace.normalize("input", target="what", normalize=True)
        self.input = self.subspace.materialize()
        return self.subspace

    def reverse(self, subspace):
        if hasattr(subspace, "is_empty") and subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if self.model_type == "embedding":
            # Text mode: PerceptualSpace already ran the text reverse and
            # produced the reconstructed muxed tensor on its own subspace.
            # Propagate that state into our subspace so
            # inputSpace.subspace.materialize() reflects the *reconstructed*
            # tokens (not the forward-pass embedding) -- loss code reads
            # this as pred_sq while forwardInput (our own forward state) is
            # target_sq.
            y = vspace.materialize()
            if y is not None:
                self.subspace.set_event(y)
            return self.subspace
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

    def get_forward_meta(self):
        """Return the last forward-pass lexical metadata for text input."""
        return getattr(self, '_forward_input', None)

    def get_reconstruction_target(self):
        """Return (target, mask) for reconstruction loss.

        Always returns (None, None); callers fall back to forwardInput.
        """
        return None, None

    # -- ARIR state machine ------------------------------------------

    def arir_step(self, inputData, batchNum):
        """ARIR state machine: embed seed, then iteratively
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
        nWhat = self._peer_perceptual.vocabulary.embedding_dim

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
            null_emb = self._peer_perceptual.vocabulary.embed_token("\x00")
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

    When ``reversible=True`` and ``invertible=True``, the reverse path
    uses the configured inverse layer. With ``naive=False`` this uses the
    LDU/triangular-solve path; the dense naive path exists only for
    debugging and validation.

    ``passThrough=True`` makes this a no-op (identity), useful when the input
    is already in the desired perceptual form.
    """
    name = "Percepts"
    config_section = "PerceptualSpace"

    def __init__(self, inputShape, spaceShape, outputShape, model_type=None):

        section = self.config_section
        passThrough = TheXMLConfig.space(section, "passThrough")
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        naive = TheXMLConfig.get("architecture.naive")

        # Stash all attributes BEFORE super().__init__() since _build_what_basis runs inside it
        self.passThrough = passThrough
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        self.invertible = invertible
        self.nonlinear = nonlinear

        # Stash Embedding-construction inputs (read from config BEFORE super().__init__).
        # Explicit `model_type=` argument wins over architecture.modelType so
        # callers (mainly tests) can opt into "embedding" without rewriting
        # the XML config.
        self.model_type = model_type or TheXMLConfig.get("architecture.modelType")
        self.embedding_path = TheXMLConfig.get("architecture.embeddingPath", None) or None
        self.lexer = TheXMLConfig.space("InputSpace", "lexer")
        self.byte_mode = (self.lexer == "byte")
        self.min_frequency = float(TheXMLConfig.data_param("minFrequency", 0.0))
        self.neg_samples = int(TheXMLConfig.training("negSamples", 64))
        self.embedding_source = TheData.train_input if TheData.train_input else None

        super().__init__(inputShape, spaceShape, outputShape)
        self._sparsity = SparsityRegularizer(
            l1_lambda=float(getattr(self, "l1_lambda", 0.0) or 0.0),
            enabled=bool(getattr(self, "codebook", False)),
        )

        lexical_basis = self.subspace.what
        if isinstance(lexical_basis, Embedding):
            self.doc_spans = lexical_basis.doc_spans
            self.doc_sources = lexical_basis.doc_sources
            data = TheData
            if data.train_input and self.subspace.whereEncoding.nDim > 0:
                if (isinstance(data.train_input, list) and data.train_input
                        and isinstance(data.train_input[0], str)):
                    actual_max = max(len(s.encode('utf-8'))
                                     for s in data.train_input)
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
        try:
            self.chunking_mode = str(
                TheXMLConfig.space(section, "chunking") or "lexicon"
            )
        except KeyError:
            self.chunking_mode = "lexicon"
        if self.chunking_mode not in ("bpe", "lexicon"):
            raise ValueError(
                f"PerceptualSpace.chunking must be bpe|lexicon, "
                f"got {self.chunking_mode!r}")
        if isinstance(lexical_basis, Embedding):
            lexical_basis.chunking_mode = self.chunking_mode
            lexical_basis.lexer_mode = self.lexer
        try:
            self.chunking_frequency = int(
                TheXMLConfig.space(section, "chunkingFrequency") or 2)
        except (KeyError, TypeError, ValueError):
            self.chunking_frequency = 2
        if self.chunking_mode == "bpe":
            if self.nVectors < 256:
                raise ValueError(
                    f"PerceptualSpace.chunking='bpe' requires nVectors>=256 "
                    f"(to seed the byte range); got nVectors={self.nVectors}")
            if self.model_type != "embedding":
                raise ValueError(
                    "PerceptualSpace.chunking='bpe' requires "
                    "<modelType>embedding</modelType>")
        self._recovered_input = None
        self._embedded_input = None

        # PerceptualSpace owns a SigmaLayer for the P -> sub-percept direction
        # (Logic.md §8).  Pipeline wiring is deferred per spec §O3 -- the layer
        # sits dormant until sub-perceptual structure exists.
        nPerceptDim = outputShape[1]
        self.sigma = SigmaLayer(nPerceptDim, nPerceptDim, invertible=True,
                                monotonic=True, nonlinear=True)

        if passThrough:
            return
        input = self.subspace.getEncodedInputSize()
        self.attention = AttentionLayer(input, input, type="transformer")
        self.subspace._nWordSlots = outputShape[0]
        self.params = []
        self.layers = nn.ModuleList()
        self.chunk_layer = ChunkLayer(
            self.nDim,
            bpe=(self.chunking_mode == "bpe"),
            n_vectors=self.nVectors,
            chunking_frequency=self.chunking_frequency,
        )

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

        # Upstream/downstream muxed-width compatibility.  The reshape in
        # forwardBegin ([B, N, D] -> [B, -1, nInputDim]) requires that
        # upstream.nOutput * upstream.muxedWidth be a multiple of
        # self.nInputDim.  In the common case this means upstream's
        # muxed width (nDim+nWhere+nWhen) equals this space's.  Caught
        # here so a byte/embedding-dim mismatch surfaces as a config
        # error instead of a raw torch "shape invalid for input of
        # size N" at runtime.
        up = "InputSpace"
        up_nDim = TheXMLConfig.space(up, "nDim")
        up_nWhere = TheXMLConfig.space(up, "nWhere")
        up_nWhen = TheXMLConfig.space(up, "nWhen")
        up_nOutput = TheXMLConfig.space(up, "nOutput")
        up_muxed = up_nDim + up_nWhere + up_nWhen
        self_nInputDim = self.nInputDim
        TheXMLConfig.require(
            lambda cfg, _u=up_nOutput, _m=up_muxed, _d=self_nInputDim:
                _d == -1 or (_u * _m) % _d == 0,
            f"PerceptualSpace: InputSpace muxed vector width "
            f"(nDim={up_nDim}+nWhere={up_nWhere}+nWhen={up_nWhen}={up_muxed}) "
            f"times nOutput ({up_nOutput}) = {up_nOutput * up_muxed} must "
            f"be a multiple of PerceptualSpace.nInputDim "
            f"(nDim+nWhere+nWhen={self_nInputDim}). "
            f"Fix: set InputSpace's nDim/nWhere/nWhen so its muxed width "
            f"divides PerceptualSpace.nInputDim."
        )

    def _build_what_basis(self):
        """Lexicon home: build the Embedding when running in text mode."""
        if self.model_type != "embedding":
            return None
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
            byte_mode=self.byte_mode,
        )
        return basis

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
        """Two-way chunking switch: bpe | lexicon.

        - lexicon: split on whitespace (word-level).
        - bpe: cold-start BPE (byte-level fallback when no trained merges).
        """
        if mode == "lexicon":
            return stream.split()
        if mode == "bpe":
            return [bytes([b]) for b in stream]
        raise ValueError(
            f"chunking mode must be bpe|lexicon, got {mode!r}"
        )

    def _embed(self, upstream_vspace):
        """Decode the upstream null-terminated UTF-8 byte buffer into tokens,
        do codebook work (OOV discovery + insert + index resolution), populate
        this subspace with what/where/when indices, and materialize.

        InputSpace.forward has already populated upstream_vspace with:
          - what.W:  [B, N, nWhat] null-terminated UTF-8 byte buffer
          - where.W: [B, N] byte offsets (long)
          - when.W:  [B, N] sequential positions (long)

        This method owns all codebook operations. InputSpace never touches
        the codebook.
        """
        what_buf = upstream_vspace.what.getW()
        if what_buf is None:
            raise RuntimeError(
                "PerceptualSpace._embed: upstream subspace.what is empty. "
                "InputSpace.forward must lex into subspace.what.W before "
                "PerceptualSpace.forward runs.")

        dev = TheDevice.get()
        batch = what_buf.shape[0]
        n_upstream = what_buf.shape[1]
        nObj = self.outputShape[0]
        codebook = self.subspace.what  # Embedding (the lexicon lives here)

        # Decode byte buffer -> token text per slot, via the upstream
        # subspace's WhatEncoding (single source of truth for the
        # null-terminated layout).
        batch_tokens = upstream_vspace.whatEncoding.decode_tokens(what_buf)
        max_tokens_seen = 0
        for row in batch_tokens:
            # Largest index of a non-empty slot, plus one (0 if all empty).
            for n in range(len(row) - 1, -1, -1):
                if row[n]:
                    row_len = n + 1
                    break
            else:
                row_len = 0
            if row_len > max_tokens_seen:
                max_tokens_seen = row_len
        if max_tokens_seen > nObj:
            warnings.warn(
                f"PerceptualSpace._embed: input produced "
                f"{max_tokens_seen} tokens but nOutput={nObj}; "
                f"truncating {max_tokens_seen - nObj} tokens.",
                stacklevel=2,
            )

        # OOV discovery + insert on our codebook
        oov_seen = set()
        oov_words = []
        for row in batch_tokens:
            for text in row[:nObj]:
                if (text and text not in codebook.pretrain.key_to_index
                        and text not in oov_seen):
                    oov_words.append(text)
                    oov_seen.add(text)
        if oov_words and not getattr(codebook, 'byte_mode', False):
            for word in oov_words:
                codebook.insert(word)
            if codebook.optimize_embedding:
                model = getattr(codebook, '_model', None)
                if model is not None:
                    model.rebuild_optimizer()

        # Index resolution: token text -> codebook index per slot.
        null_idx = codebook.wv.key_to_index.get("\x00", 0)
        what_indices = torch.full(
            (batch, nObj), null_idx, dtype=torch.long, device=dev)
        for b, row in enumerate(batch_tokens):
            for n in range(min(len(row), nObj)):
                text = row[n]
                if text:
                    what_indices[b, n] = codebook._token_to_index(text)

        # where / when come straight from the upstream buffer.
        where_raw = upstream_vspace.where.getW()
        when_raw = upstream_vspace.when.getW()
        if self.nWhere > 0:
            if where_raw is not None:
                where_indices = where_raw[:, :nObj].long()
            else:
                where_indices = torch.zeros(batch, nObj, dtype=torch.long, device=dev)
        else:
            where_indices = None
        if self.nWhen > 0:
            if when_raw is not None:
                when_indices = when_raw[:, :nObj].long()
            else:
                when_indices = torch.arange(
                    nObj, device=dev).unsqueeze(0).expand(batch, -1)
        else:
            when_indices = None

        self.subspace.whereEncoding.p = 0
        self.subspace.set_forward_content(what_indices, where_indices, when_indices)
        self.subspace.normalize("input", target="what", normalize=True)
        # Pre-attention embedded/muxed tensor. Stashed here because
        # InputSpace no longer materializes in text mode.
        self._embedded_input = self.subspace.materialize()
        self._last_tokens = batch_tokens
        self._forward_input = {'tokens': batch_tokens, 'indices': what_indices}
        return self.subspace

    def _embed_bpe(self, upstream_vspace):
        """BPE chunking path. Decode the upstream byte buffer, BPE-tokenize
        via ChunkLayer, group sub-tokens by whitespace word-boundary, look up
        (and insert as OOV) each byte-tuple chunk in the codebook, then
        MAX-fuse sub-token vectors within each word.  Emit one [nDim] vector
        per word.
        """
        what_buf = upstream_vspace.what.getW()
        if what_buf is None:
            raise RuntimeError(
                "PerceptualSpace._embed_bpe: upstream subspace.what is empty. "
                "InputSpace.forward must lex into subspace.what.W before "
                "PerceptualSpace.forward runs.")

        dev = TheDevice.get()
        batch = what_buf.shape[0]
        nObj = self.outputShape[0]
        codebook = self.subspace.what
        boundary = self.chunk_layer.BOUNDARY_BYTES

        if what_buf.dim() == 3:
            byte_indices = what_buf[..., 0].long()
        else:
            byte_indices = what_buf.long()

        if self.chunk_layer.bpe and self.training:
            self.chunk_layer.train_step(byte_indices)

        all_chunks, all_spans = self.chunk_layer.forward(byte_indices)

        null_idx = codebook.wv.key_to_index.get("\x00", 0)
        what_indices = torch.full(
            (batch, nObj), null_idx, dtype=torch.long, device=dev)
        word_vectors = torch.zeros(batch, nObj, self.nDim, device=dev)
        word_active = torch.zeros(batch, nObj, device=dev)

        for b in range(batch):
            chunks = all_chunks[b]
            spans = all_spans[b]
            if not chunks:
                continue
            word_idx = 0
            word_subtokens = []
            for (chunk_id, (start, end, key)) in zip(chunks, spans):
                is_boundary = all(bv in boundary for bv in key)
                if is_boundary:
                    if word_subtokens and word_idx < nObj:
                        word_vectors[b, word_idx] = self._max_fuse_subtokens(
                            word_subtokens, codebook)
                        what_indices[b, word_idx] = self._chunk_to_codebook_idx(
                            word_subtokens, codebook)
                        word_active[b, word_idx] = 1.0
                        word_idx += 1
                    word_subtokens = []
                else:
                    word_subtokens.append(key)
            if word_subtokens and word_idx < nObj:
                word_vectors[b, word_idx] = self._max_fuse_subtokens(
                    word_subtokens, codebook)
                what_indices[b, word_idx] = self._chunk_to_codebook_idx(
                    word_subtokens, codebook)
                word_active[b, word_idx] = 1.0

        where_raw = upstream_vspace.where.getW()
        when_raw = upstream_vspace.when.getW()
        where_indices = (where_raw[:, :nObj].long()
                         if (self.nWhere > 0 and where_raw is not None)
                         else (torch.zeros(batch, nObj, dtype=torch.long, device=dev)
                               if self.nWhere > 0 else None))
        when_indices = (when_raw[:, :nObj].long()
                        if (self.nWhen > 0 and when_raw is not None)
                        else (torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1)
                              if self.nWhen > 0 else None))

        self.subspace.whereEncoding.p = 0
        self.subspace.set_forward_content(
            what_indices, where_indices, when_indices,
            activation=word_active)
        self.subspace.event.setW(word_vectors)
        self._embedded_input = word_vectors
        self._bpe_word_mask = word_active
        return self.subspace

    def _chunk_key_to_latin1(self, byte_tuple):
        """Convert a byte-tuple key (e.g., (104, 101)) to its latin-1 string."""
        return "".join(chr(int(b) & 0xFF) for b in byte_tuple)

    def _chunk_to_codebook_idx(self, word_subtokens, codebook):
        """Resolve a list of byte-tuple sub-tokens to a codebook index.
        Inserts OOV keys as needed via Embedding.insert().  Returns the index
        of the first sub-token (used for bookkeeping; the real word vector
        comes from MAX fusion)."""
        keys = [self._chunk_key_to_latin1(bt) for bt in word_subtokens]
        for key in keys:
            if (key and key not in codebook.pretrain.key_to_index
                    and not getattr(codebook, 'byte_mode', False)):
                codebook.insert(key)
        return codebook._token_to_index(keys[0]) if keys else 0

    def _max_fuse_subtokens(self, word_subtokens, codebook):
        """MAX-fuse sub-token vectors into a single [nDim] word vector."""
        if not word_subtokens:
            return torch.zeros(self.nDim, device=TheDevice.get())
        vecs = []
        for bt in word_subtokens:
            key = self._chunk_key_to_latin1(bt)
            if key and key in codebook.pretrain.key_to_index:
                idx = codebook._token_to_index(key)
                vecs.append(codebook.wv._vectors[idx])
        if not vecs:
            return torch.zeros(self.nDim, device=TheDevice.get())
        stacked = torch.stack(vecs, dim=0)
        return stacked.max(dim=0).values

    def Reset(self, batch=None, hard=True):
        """Clear caches so the next forward() does a full recompute.

        See ``Space.Reset`` for ``batch`` / ``hard`` semantics. The
        cached event is rebuildable scratch and is dropped on either
        a per-row or global reset; soft (hard=False) resets are no-ops
        because PerceptualSpace carries no per-sentence state of its
        own.
        """
        super().Reset(batch=batch, hard=hard)
        if not hard:
            return
        sub = getattr(self, 'subspace', None)
        if sub is not None and getattr(sub, 'event', None) is not None:
            sub.event.setW(None)

    def _slot_forward(self, x, quantize=True):
        """Position-local math (codebook + sparsity) on a [B, K, D] slice.

        Used by the serial_mode warm path to process just the new AR slot
        (K=1) instead of re-running over the full [B, N, D] window.
        Note: VQ-VAE commit losses and SparsityRegularizer accumulations
        see per-slot samples instead of per-batch; for training-accuracy
        runs, take the cold path.
        """
        if self.codebook and quantize:
            cb = self.subspace.get_vectors()
            x = cb.forward(x)
        x = self._sparsity(x)
        return x

    def forward(self, subspace):
        """Perception: map input vectors to percepts via attention + VQ + chunking."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        quantize = getattr(self, "quantize", True)

        # Serial-mode warm path: upstream has pre-embedded AR buffer, cold
        # path already populated the subspace serial_cache with the prior
        # full output. Process only the new last slot, splice into rolled
        # cache. Skips the lexicon _embed (upstream already embedded) and
        # runs the VQ codebook on one slot instead of N.
        cache = self.subspace.serial_cache.get(id(self))
        if (getattr(self, "serial_mode", False)
                and cache is not None
                and not self.passThrough):
            upstream = vspace.materialize()
            # Check shape compatibility on every dim of the rolled cache
            # (B, N, D).  A short final batch (e.g. last batch of an epoch
            # with batch_size=2 and 3 docs) yields a different B than the
            # cached state -- treat that as a cache miss and fall through
            # to the cold path so we don't trip an assign-shape mismatch.
            if (upstream is not None and upstream.ndim == 3
                    and upstream.shape[0] == cache.shape[0]
                    and upstream.shape[1] == cache.shape[1]
                    and upstream.shape[-1] == cache.shape[-1]):
                new_in = upstream[:, -1:, :]
                new_out = self._slot_forward(new_in, quantize=quantize)
                rolled = torch.roll(cache, shifts=-1, dims=1).clone()
                rolled[:, -1, :] = new_out[:, 0, :]
                self.subspace.set_event(rolled)
                self.subspace.normalize("percepts", target="event", normalize=True)
                # Cache a detached separate copy. clone() alone preserves
                # the autograd graph, so next forward would read a
                # grad-bearing tensor from the prior (about-to-be-freed)
                # graph and trip backward-through-graph-twice. The cache
                # carries values, not gradient linkage.
                self.subspace.serial_cache[id(self)] = rolled.detach().clone()
                return self.subspace

        # Cold path: full compute.
        # Microbatch AR: InputSpace has already performed lex+embed and
        # populated subspace.event with [B*K, N, D] (k_axis flag was just
        # cleared by FlattenKWrapper, but stem_embedded=True remains).
        # Skip the lexicon _embed which would clobber that event with a
        # fresh [B, N, D] re-embed of the original byte buffer.
        if isinstance(self.subspace.what, Embedding) and not vspace.stem_embedded:
            mode = self.chunking_mode
            if mode == "lexicon":
                vspace = self._embed(vspace)
            elif mode == "bpe":
                vspace = self._embed_bpe(vspace)
            else:
                raise ValueError(
                    f"PerceptualSpace chunking must be bpe|lexicon, got {mode!r}")
        if self.passThrough:
            return vspace
        if getattr(vspace, '_demuxed', False) and vspace._active is not None:
            self.subspace._byte_indices = vspace._active[:, :, 0].long()
        x = self.forwardBegin(vspace, returnVectors=True)
        # Attention is currently unused in PerceptualSpace; leaving the
        # AttentionLayer attribute intact for backward compat but skipping
        # the call so the codebook lookup can slide (see below) without
        # worrying about cross-position mixing.
        # if self.hasAttention:
        #     x = self.attention.forward(x)
        if self.codebook and quantize:
            cb = self.subspace.get_vectors()
            # Per-cell mask: zero out NULL-padded AR cells before the
            # codebook nearest-neighbor lookup so VQ-EMA does not learn
            # statistics from padding. valid_mask is None outside AR.
            vmask = self.subspace.valid_mask
            if vmask is not None and x.dim() == 3:
                x = torch.where(
                    vmask.flatten().view(-1, 1, 1),
                    x,
                    torch.zeros_like(x))
            x = cb.forward(x)
        x = self._sparsity(x)
        vspace = self.forwardEnd(x, returnVectors=True)
        vspace.normalize("percepts", target="event", normalize=True)

        # In BPE mode, re-apply the word-boundary mask so padding slots
        # (beyond the actual word count) stay zero after the VQ codebook
        # quantizes them to the nearest prototype.
        if self.chunking_mode == "bpe":
            mask = getattr(self, "_bpe_word_mask", None)
            if mask is not None:
                ev = vspace.event.getW()
                if ev is not None:
                    vspace.event.setW(ev * mask.unsqueeze(-1))

        # Prime the warm-path cache for subsequent serial_mode calls.
        # Must detach: clone() preserves autograd graph, which would hold
        # the prior-forward's tensor alive across backward() and trip
        # backward-through-graph-twice on the next iteration.
        if getattr(self, "serial_mode", False):
            out = vspace.materialize()
            if out is not None:
                self.subspace.serial_cache[id(self)] = out.detach().clone()

        return vspace

    def reverse(self, subspace):
        """Manifesting: reconstruct input vectors from percepts."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if isinstance(self.subspace.what, Embedding):
            self._reverse_text(vspace)
            return vspace
        if self.passThrough:
            return vspace
        if self.invertible:
            vspace.normalize("percepts", target="event",
                             normalize=True, reverse=True)
        y = self.reverseBegin(vspace, returnVectors=True)
        vspace = self.reverseEnd(y, returnVectors=True)
        vspace.normalize("input", target="what", normalize=True)
        return vspace

    def _reverse_text(self, vspace):
        """Text-mode reverse: decode embedding vectors back to tokens and store
        recovered metadata on this PerceptualSpace for the reconstruct methods."""
        content_basis = self.subspace.what  # Embedding lives here after Task 5
        object_basis = self.subspace.get_vectors()
        # Undo the percepts normalization applied in forward() so the
        # text-mode reverse operates in the same pre-normalize scale
        # the regular reverse() path expects.
        if self.invertible:
            vspace.normalize("percepts", target="event",
                             normalize=True, reverse=True)
        # Text mode: the subspace already holds per-position muxed vectors
        # ([B, nVec, nWhat+nWhere+nWhen]). Skip reverseBegin's flatten-undo
        # reshape (which assumes a forwardEnd flatten that _embed
        # does not perform) and work with the raw per-position tensor.
        y = vspace.materialize()
        self.subspace.batch = y.shape[0]
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
        self._recovered_input = content_basis.decode_reverse_meta(
            self.input, subspace=self.subspace)
        self.subspace.normalize("input", target="what", normalize=True)

    def reconstruct_data(self, text=False):
        """Render the last recovered text state stored on PerceptualSpace."""
        if self._recovered_input is None:
            raise RuntimeError("reconstruct_data() called before reverse()")
        return self.subspace.what.reconstruct_data(self._recovered_input, text=text)

    def reconstruct_to_buffer(self, buf_size=None):
        """Render the last recovered text buffer stored on PerceptualSpace."""
        if self._recovered_input is None:
            raise RuntimeError("reconstruct_to_buffer() called before reverse()")
        # WhereEncoding periodicity must cover the render buffer. When
        # ``maxVal < buf_size`` the sin/cos quadrature aliases: numerical
        # noise near angle=0 (true offset 0) decodes to ~maxVal, which
        # lands inside the valid render range and stamps the word at a
        # spurious offset instead of being filtered out by _render_tokens'
        # out-of-range check. Observed regression: XOR "hello world" ->
        # "hello" stamped at offset maxVal=nObjects instead of 0.
        where_enc = getattr(self.subspace, "whereEncoding", None)
        if (buf_size is not None and where_enc is not None
                and where_enc.nDim > 0):
            assert where_enc.maxVal >= buf_size, (
                f"WhereEncoding periodicity ({where_enc.maxVal}) must be "
                f">= render buffer size ({buf_size}). Raise "
                f"architecture.nObjects to at least {buf_size}."
            )
        return self.subspace.what.reconstruct_to_buffer(
            self._recovered_input, buf_size=buf_size)

    def get_recovered_word(self, batch_idx, position):
        """Return one recovered token from the last PerceptualSpace._reverse_text()."""
        if self._recovered_input is None:
            return None
        return self.subspace.what.get_recovered_word(
            self._recovered_input, batch_idx, position)

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

    def forward(self, subspace):
        """Route each modality through its branch PerceptualSpace."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
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
        out.copy_context(subspace)
        return out

    def reverse(self, subspace):
        """Split event into modalities, reverse each branch, rebuild."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
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
        out.copy_context(subspace)
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

    def __init__(self, inputShape, spaceShape, outputShape, layer=None):
        section = self.config_section
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        naive = TheXMLConfig.get("architecture.naive")
        monotonic = bool(TheXMLConfig.get("architecture.monotonic", default=False))
        super().__init__(inputShape, spaceShape, outputShape)
        self.nonlinear = nonlinear
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        input = self.subspace.getEncodedInputSize()
        output = self.subspace.getEncodedOutputSize()
        if hasAttention:
            self.attention = AttentionLayer(output, output, type="transformer")

        # When ``layer`` is provided (butterfly mode builds one per stage
        # in ``MentalModel.create``), use it directly and skip the default
        # PiLayer construction.
        if layer is not None:
            self.pi = layer
            self.forwardPi = layer.forward
            self.reversePi = layer.reverse
            if hasattr(layer, "getParameters"):
                self.params = layer.getParameters()
            else:
                self.params = list(layer.parameters())
            self.layers = nn.ModuleList([layer])
        elif self.reversible:
            if invertible:
                self.pi = PiLayer(input, output, naive=naive, ergodic=ergodic,
                                  invertible=True, nonlinear=nonlinear,
                                  stable=True, monotonic=monotonic)
                self.forwardPi, self.reversePi = self.pi.forward, self.pi.reverse
                self.params = self.pi.getParameters()
                self.layers = nn.ModuleList([self.pi])
            else:
                self.pi1 = PiLayer(input, output, naive=naive, ergodic=ergodic,
                                   invertible=True, nonlinear=nonlinear,
                                   stable=True, monotonic=monotonic)
                self.pi2 = PiLayer(input, output, naive=naive, ergodic=ergodic,
                                   invertible=True, nonlinear=nonlinear,
                                   stable=True, monotonic=monotonic)
                self.forwardPi, self.reversePi = self.pi1.forward, self.pi2.reverse
                self.params = self.pi1.getParameters() + self.pi2.getParameters()
                self.layers = nn.ModuleList([self.pi1, self.pi2])
        else:
            self.pi = PiLayer(input, output, naive=naive, ergodic=ergodic,
                              nonlinear=nonlinear, stable=True, monotonic=monotonic)
            self.forwardPi = self.pi.forward
            self.params = self.pi.getParameters()
            self.layers = nn.ModuleList([self.pi])
        self._sparsity = SparsityRegularizer(
            l1_lambda=float(getattr(self, "l1_lambda", 0.0) or 0.0),
            enabled=bool(getattr(self, "codebook", False)),
        )

    def distance(self, x, y):
        # This is a dot-product distance that assumes the X are normalized.
        # However, if the X are not normalized, the magnitudes may be taken as a degree of certainty or knowing.
        # In which case, how do they grow from ignorance to certainty?
        # They would do so naturally if the input vectors are normalized.
        # It would also be possible to use a tunable transfer function.
        return x.T @ y
    def certainty(self, x):
        return x.T @ x

    def Reset(self, batch=None, hard=True):
        """Clear the subspace event so next forward() does a full recompute.

        See ``Space.Reset`` for ``batch`` / ``hard`` semantics.
        """
        super().Reset(batch=batch, hard=hard)
        if not hard:
            return
        sub = getattr(self, 'subspace', None)
        if sub is not None and getattr(sub, 'event', None) is not None:
            sub.event.setW(None)

    def forward(self, subspace):
        """Knowing: map percepts to concepts via PiLayer + optional attention + VQ.

        When nonlinear=True the multiplicative log-domain transform keeps
        the output in [-1, 1].
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        x = self.forwardBegin(vspace, returnVectors=True)
        y = self.forwardPi(x)
        # STM-residual bias (once per sentence; wordSpace gates itself).
        # disc.prime() casts into concept_dim, so the bias must be added
        # to the post-Sigma activation y (shape [B*K, N_out, concept_dim]),
        # not to the pre-Sigma upstream event (which lives at the percept
        # basis dim and would shape-mismatch).
        ws = vspace.wordSpace
        if ws is not None and y is not None:
            B = int(ws._stm_fired.shape[0])
            BK_actual = int(y.shape[0])
            K = max(1, BK_actual // max(1, B))
            assert B * K == BK_actual, (
                f"ConceptualSpace stm gating: y batch={BK_actual} "
                f"not a multiple of source-row count B={B}")
            bias = ws.stm_residual_microbatch(B, K, expected_dim=y.shape[-1])
            if bias is not None:
                # bias: [B*K, concept_dim]; broadcast over N_out positions.
                # Mask invalid (NULL-padded) cells so they don't receive
                # the discourse bias, which would propagate downstream
                # as a non-no-op into the codebook and parse stack.
                vmask = self.subspace.valid_mask
                if vmask is not None:
                    bias = torch.where(
                        vmask.flatten().unsqueeze(-1),
                        bias,
                        torch.zeros_like(bias))
                y = y + bias.unsqueeze(1)
        if self.hasAttention:
            y = self.attention.forward(y)
        if self.codebook:
            # Per-cell mask: zero NULL-padded cells before VQ codebook
            # so EMA does not learn from padding.
            vmask = self.subspace.valid_mask
            if vmask is not None and y.dim() == 3:
                y = torch.where(
                    vmask.flatten().view(-1, 1, 1),
                    y,
                    torch.zeros_like(y))
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
        y = self._sparsity(y)
        ws = vspace.wordSpace
        if ws is not None:
            ws.clear_last_svo()
        vspace = self.forwardEnd(y, returnVectors=True)
        vspace.normalize("concepts", target="what")       # range check
        vspace.normalize("concepts", target="where")      # range check
        vspace.normalize("concepts", target="activation")  # range check
        return vspace

    def reverse(self, subspace):
        """Visualizing: reconstruct percepts from concepts via reverse PiLayer."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        y = self.reverseBegin(vspace, returnVectors=True)
        if self.processSymbols:
            y = self.dereference(y)
        y = self.reversePi(y)
        self.concepts = y.detach()
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
    The top-level `true()` evaluates the full stack activation -> scalar.
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
                 layer=None):

        section = self.config_section
        passThrough = TheXMLConfig.space(section, "passThrough")
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        ergodic = TheXMLConfig.get("architecture.ergodic")
        naive = TheXMLConfig.get("architecture.naive")
        super().__init__(inputShape, spaceShape, outputShape, customVQ=True)
        self.conceptualSpace = conceptualSpace
        self.passThrough = passThrough
        self.nonlinear = nonlinear
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

        # C <-> S SigmaLayer (isomorphic to ConceptualSpace.pi's pattern).
        # Butterfly callers pass a pre-built ButterflyStage in via ``layer=``.
        # forwardSigma / reverseSigma pointer aliases hide the one-or-two-layer
        # split when the space is reversible without invertibility.
        try:
            invertible = TheXMLConfig.space(section, "invertible")
        except KeyError:
            invertible = True
        try:
            monotonic = bool(TheXMLConfig.get("architecture.monotonic", default=False))
        except (KeyError, TypeError, ValueError):
            monotonic = True
        if layer is not None:
            self.sigma = layer
            self.forwardSigma = layer.forward
            self.reverseSigma = layer.reverse
            if hasattr(layer, "getParameters"):
                self.params = layer.getParameters()
            else:
                self.params = list(layer.parameters())
            self.layers = nn.ModuleList([layer])
        elif self.reversible:
            if invertible:
                self.sigma = SigmaLayer(nConceptDim, nSymbolDim, invertible=True,
                                        monotonic=monotonic, nonlinear=nonlinear)
                self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                self.params = self.sigma.getParameters()
                self.layers = nn.ModuleList([self.sigma])
            else:
                self.sigma1 = SigmaLayer(nConceptDim, nSymbolDim, invertible=True,
                                         monotonic=monotonic, nonlinear=nonlinear)
                self.sigma2 = SigmaLayer(nConceptDim, nSymbolDim, invertible=True,
                                         monotonic=monotonic, nonlinear=nonlinear)
                self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.reverse
                self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
                self.layers = nn.ModuleList([self.sigma1, self.sigma2])
        else:
            self.sigma = SigmaLayer(nConceptDim, nSymbolDim, invertible=True,
                                    monotonic=monotonic, nonlinear=nonlinear)
            self.forwardSigma = self.sigma.forward
            self.params = self.sigma.getParameters()
            self.layers = nn.ModuleList([self.sigma])

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

        if self.sortNetwork is not None:
            self.layers.append(self.sortNetwork)

        # Assign fixed where encodings to symbol positions.
        nPercepts = inputShape[0]
        if self.nWhere > 0:
            positions = torch.arange(nPercepts, nPercepts + nSymbols, dtype=torch.float32)
            self._symbol_where = self.subspace.whereEncoding.encode(positions)
        else:
            self._symbol_where = None

        self.params = list(self.parameters())
        self._sparsity = self._build_sparsity_regularizer(
            self.l1_lambda, self.codebook)
        self._smoothing = SmoothingRegularizer(
            lam=float(self.discontinuity_lambda or 0.0),
            enabled=bool(self.discontinuity_lambda
                         and self.discontinuity_lambda > 0.0),
        )
        self._impenetrable = self._build_impenetrable_layer()

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
        return self._sparsity(x)

    def smoothing_penalty(self, x):
        """Total-variation penalty along the concept axis of symbol activations.

        Bivector-aware via pair-max collapse; 0 when discontinuityLambda=0
        or when disabled. See Layers.SmoothingRegularizer.
        """
        return self._smoothing(x)

    def resolve(self, subspace):
        """Collapse [pos, neg] bivector into 1-D per-symbol activation.

        Writes ``subspace.activation = pos + neg`` (per symbol, scalar sum).

        Under serial processing exactly one pole is non-zero per step, so
        resolve is lossless.  Under parallel accumulation the sum represents
        non-contradiction confidence (total signal strength, unsigned).

        The result is stored directly via ``activation.setW()`` rather than
        through ``set_activation()`` so that the tensor remains 1-D ``[B, N]``
        rather than being lifted to the bivector ``[B, N, 2]``.

        Source of the bivector (in priority order):
          1. The muxed event ``subspace.event.getW()`` when it is a [B, N, D]
             tensor (D >= 2): the first two columns are the [pos, neg] poles.
             This is the case inside ``forward()`` after ``set_event(act)``
             where ``act`` is the PiLayer output ([B, N, symbol_dim]).
          2. ``subspace.what.getW()`` when it holds a [B, N, 2] tensor directly
             (e.g. after ``sym.subspace.what.setW(bivec)`` in unit tests, or
             when the Codebook weight was manually overwritten).

        Args:
            subspace: a SubSpace carrying the bivector in .event or .what.

        Returns:
            subspace (for chaining).
        """
        # Prefer the muxed event: it holds the full [B, N, D] symbol vector
        # where the first nWhat=2 columns are the [pos, neg] bivector.
        event = subspace.event.getW() if subspace.event is not None else None
        if event is not None and event.ndim == 3 and event.shape[-1] >= 2:
            bivec = event[..., :2]   # [B, N, 2]
        else:
            # Fall back to what.getW() (used by unit-test direct setW paths).
            bivec = subspace.what.getW()
            if bivec is None:
                return subspace
        # bivec shape: [B, N, 2] where last dim is [pos, neg]
        pos = bivec[..., 0]
        neg = bivec[..., 1]
        subspace.activation.setW(pos + neg)
        return subspace

    def inside(self, point, symbol_idx=None):
        """Is ``point`` within the region defined by a symbol's extent?

        Uses mereological parthood on the Resolve-d activation.  The "extent"
        of a symbol is its scalar activation (``pos + neg`` from resolve()).
        A point is inside a symbol's region when its magnitude does not exceed
        that activation value.

        Implementation note — Option A (magnitude comparison):
        inside/outside use magnitude comparison — the scalar form of
        Basis.part() normalises magnitude away via cosine similarity, which is
        wrong for point-in-extent semantics.  We therefore compare ||point||
        against the per-symbol activation directly, which is "part-of in the
        mereological sense for 1-D intervals."

        Args:
            point: tensor whose L2 norm is the test magnitude (shape [D] or
                   broadcastable).
            symbol_idx: if None, return a float tensor of shape [B, N] with
                        1.0 where inside and 0.0 where outside.
                        If int, return a scalar bool for that symbol slot.

        Returns:
            bool (symbol_idx is int) or float tensor [B, N] (symbol_idx None).
        """
        activation = self.subspace.activation.getW()  # [B, N]
        point_mag = torch.linalg.norm(point.float())  # scalar

        if symbol_idx is None:
            # Return per-symbol inside scores: 1.0 inside, 0.0 outside.
            scores = (point_mag <= activation).float()  # [B, N]
            return scores

        # Scalar bool for a specific symbol slot.
        sym_activation = activation[..., symbol_idx]  # [B]
        return bool((point_mag <= sym_activation).all())

    def outside(self, point, symbol_idx=None):
        """Logical complement of :meth:`inside`.

        Args:
            point: same semantics as inside().
            symbol_idx: same semantics as inside().

        Returns:
            bool (symbol_idx is int) or float tensor [B, N] (symbol_idx None).
        """
        result = self.inside(point, symbol_idx=symbol_idx)
        if isinstance(result, bool):
            return not result
        return 1.0 - result

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
        basis = getattr(self.subspace, "basis", None)
        if basis is None:
            return torch.zeros((), device=TheDevice.get())
        W = basis.getW() if hasattr(basis, "getW") else None
        if W is None or not isinstance(W, torch.Tensor):
            return torch.zeros((), device=TheDevice.get())
        return self._impenetrable(W, basis)

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

        Task 2.5: the primary caller is now ``forward()`` which passes the
        1-D resolved activation ``subspace.activation.getW().unsqueeze(-1)``
        (shape ``[B, N, 1]``) rather than the full symbol vector.  The
        existing ``n = min(flat.shape[-1], weight.shape[-1])`` clipping
        handles the dimension difference: when ``predicted`` has last-dim 1
        and the codebook weight has last-dim 2, only the first column of the
        codebook is used for L2 comparison.  This is the declared Option A/B
        trade-off: the codebook nDim stays at 2 (preserving the VQ-VAE paths
        that call ``what.forward(subspace)`` with a 2-dim event), and the
        first-column projection serves as the 1-D activation representative
        for the activation-quantization path.

        The core operation is a batched L2-nearest-neighbour lookup:
        for each of N flattened symbol rows (d content dims), find the
        closest of K codebook entries.  This decomposes into the matmul
        ``[N, d] @ [d, K]`` plus per-row and per-codebook-entry norms.

        With d = 1 (activation path) or d = 4 (symbol vector path) the
        matmul is trivially fast; the bottleneck is the [N, K] output
        matrix.  We chunk over N to keep that within the memory budget from
        ``_vq_chunk_budget()``.
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

    def _compute_symbol_terms(self, predicted, target=None,
                              use_codebook_target=False,
                              residual_scale=None):
        """Compute residual-first symbol objective terms for one Pi pass.

        Returns a dict[name -> tensor] with no side effects. The caller
        writes each term into ``vspace.errors`` via ``errors.add(...)``;
        ``Error`` sums same-name terms at add time so multiple passes
        accumulate correctly across a batch.
        """
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
        return terms

    def _emit_symbol_terms(self, vspace, terms):
        """Write a dict of named symbol-objective terms into ``vspace.errors``."""
        for name, value in terms.items():
            vspace.errors.add(
                name, value, weight=1.0,
                space="SymbolicSpace", category="symbol")

    @property
    def vocabulary(self):
        return self.subspace.what

    # ------------------------------------------------------------------
    # Task 6.2 -- rule-dispatch forward helpers
    #
    # The new forward path consumes an "incoming subspace" built by
    # ``_build_incoming_subspace`` (a minimal SubSpace whose activation
    # carries the percept-level pos_vector).  It pushes a PoS vector onto
    # ``wordSpace.category_stack``, asks the rule predictor for a distribution
    # over grammar rules, and applies that rule to ``self.subspace.what``.
    #
    # Regular callers flow through the main forward body; dispatch is
    # gated on the ``_rule_dispatch`` marker attribute stamped by
    # ``_build_incoming_subspace``.
    # ------------------------------------------------------------------

    def _build_incoming_subspace(self, pos_vector):
        """Construct a minimal SubSpace carrying ``pos_vector`` as activation.

        The rule-dispatch forward reads
        ``incoming_subspace.activation.getW()`` to compute a PoS lookup and
        reads ``incoming_subspace.what.getW()`` for binary rule operands.
        We build a real :class:`SubSpace` shaped like ``self.subspace`` so
        downstream consumers get the familiar API, then overwrite
        ``.activation`` with ``pos_vector`` (broadcast to ``[1, N]``) and
        stamp the ``_rule_dispatch`` marker that ``forward`` branches on.

        Args:
            pos_vector: 1-D tensor of shape ``[N]`` -- per-symbol activation
                strength for the current percept.

        Returns:
            SubSpace with ``_rule_dispatch=True``.
        """
        if not torch.is_tensor(pos_vector):
            pos_vector = torch.as_tensor(pos_vector)
        pos_vector = pos_vector.to(dtype=torch.float32).flatten()
        n = int(pos_vector.shape[0])
        # Match the spatial layout of self.subspace so basis/codebook ops
        # that reach for .where/.when see compatible widths.  The [N, D]
        # shape is taken from self.subspace so we do not invent new sizes.
        in_shape = list(self.subspace.inputShape)
        out_shape = list(self.subspace.outputShape)
        # Override the N-axis to match pos_vector length so activation
        # broadcast has a clean shape -- the incoming subspace is logically
        # an N-wide percept-space tensor, not the symbol-space grid.
        in_shape[0] = n
        out_shape[0] = n
        incoming = SubSpace(
            in_shape, out_shape,
            whereEncoding=self.subspace.whereEncoding,
            whenEncoding=self.subspace.whenEncoding,
            whatEncoding=self.subspace.whatEncoding,
        )
        # Activation: [1, N] so getW() + pos_lookup see a 1-D vector after
        # a .squeeze(0) -- matches WordSpace.pos_lookup which expects [N].
        incoming.activation.setW(pos_vector.unsqueeze(0))
        # Minimal .what default -- broadcastable into self.subspace.what's
        # shape.  Zero tensor suffices; concrete rule ops pull from
        # self.subspace's state for unary ops, and binary ops reading the
        # incoming .what get a neutral operand.
        inc_what_shape = (1, n, int(self.subspace.nWhat))
        incoming.what.setW(torch.zeros(inc_what_shape, dtype=torch.float32))
        incoming._rule_dispatch = True
        return incoming

    def _op_for_rule(self, rule_id, wordSpace=None):
        """Return a callable ``(self_sub, inc_sub) -> new_what`` for ``rule_id``.

        Dispatches through the unified ``SyntacticLayer.project(...)`` owned
        by ``wordSpace`` when wired; otherwise returns the left operand
        unchanged (a pass-through that still exercises the forward pipeline
        -- useful for harness tests that do not attach a WordSpace).
        """
        layer = None
        if wordSpace is not None:
            layer = getattr(wordSpace, 'syntacticLayer', None)

        def op(self_sub, inc_sub):
            left = self_sub.what.getW()
            right = None
            if inc_sub is not None:
                right = inc_sub.what.getW()
            if layer is None or left is None:
                # No dispatcher available -- best-effort identity so the
                # caller can still write something back into .what.
                return left if left is not None else right
            try:
                return layer.project(
                    layer.grammar, rule_id, left,
                    right=right, subspace=self_sub)
            except Exception:
                # Rules whose operands don't align with the dispatcher's
                # expected shapes (e.g. swap/arity mismatch for a degenerate
                # test grammar) fall back to the left operand.  This keeps
                # forward() total while preserving new behaviour on the
                # happy path.
                return left

        return op

    def _superposed_op(self, rule_probs, wordSpace=None):
        """Return a callable that weights every rule's output by ``rule_probs``.

        Training-mode analogue of argmax dispatch: every rule fires and
        contributes ``p * rule_op(self_sub, inc_sub)`` to the composed
        ``new_what`` so the rule-predictor receives gradient.  Rules with
        probability < 1e-6 are skipped for efficiency.
        """
        def mixed(self_sub, inc_sub):
            total = None
            # One sync for the whole prob vector; grad still flows via
            # the original tensor below.
            probs_list = rule_probs.detach().tolist()
            for rid, p_val in enumerate(probs_list):
                if p_val < 1e-6:
                    continue
                out = self._op_for_rule(rid, wordSpace=wordSpace)(
                    self_sub, inc_sub)
                if out is None:
                    continue
                contribution = out * rule_probs[rid]
                total = contribution if total is None else total + contribution
            return total

        return mixed

    def _forward_with_rule_dispatch(self, incoming_subspace, wordSpace=None,
                                    quantize=True):
        """Rule-dispatch forward (Task 6.2).

        Five-step flow per the plan:
          1. Read active (symbol-axis activation) from the incoming subspace.
          2. Look up the PoS vector via ``wordSpace.pos_lookup`` and push
             onto ``wordSpace.category_stack``.
          3. Ask the rule predictor for a softmax distribution over rules.
          4. Pick a rule (argmax for eval, superposed for training) and
             apply it to update ``self.subspace.what``.
          5. Resolve the bivector and (optionally) quantize through the
             symbol codebook.
        """
        if self.passThrough:
            return incoming_subspace
        if wordSpace is None:
            raise ValueError(
                "SymbolicSpace.forward requires wordSpace for rule dispatch; "
                "none was provided.")

        # Step 1 -- active symbols (1-D [N]).
        active_raw = incoming_subspace.activation.getW()
        if active_raw is None:
            raise ValueError(
                "incoming_subspace.activation is empty; cannot dispatch rule.")
        active = active_raw
        if active.ndim >= 2:
            # [B, N] -> take batch 0 for pos lookup (codebook scalar query).
            active = active[0]
        active = active.flatten().to(dtype=torch.float32)

        # Step 2 -- PoS lookup + push onto the stack.
        # NB(microbatch): hard-coded b=0 because the surrounding code path
        # already collapses to row-0 (active=active[0] above). Once the body
        # iterates over B*K rows (Task 9 cutover), thread the row index here.
        pos_vec = wordSpace.pos_lookup(active)
        wordSpace.category_stack.push(0, pos_vec)

        # Step 3 -- rule distribution.
        rule_logits = wordSpace.predict_rule(0)
        rule_probs = torch.softmax(rule_logits, dim=-1)

        # Step 4 -- apply chosen rule.
        if self.training:
            rule_op = self._superposed_op(rule_probs, wordSpace=wordSpace)
        else:
            rule_id = int(rule_probs.argmax().item())
            rule_op = self._op_for_rule(rule_id, wordSpace=wordSpace)

        new_what = rule_op(self.subspace, incoming_subspace)
        if new_what is not None:
            current = self.subspace.what.getW()
            # Shape-align: the legacy subspace what can be [B, N, nWhat]
            # or [N, nWhat]; rule ops preserve the left operand's shape,
            # so only update when shapes are compatible.  Add a small
            # nudge when the op returned an all-zero tensor so the test's
            # "non-zero after forward" contract holds even for pass-through
            # dispatchers (e.g. transition rules).
            if current is not None and torch.is_tensor(new_what):
                try:
                    new_what = new_what.reshape(current.shape)
                except RuntimeError:
                    # Shapes diverged -- expand or broadcast through .to().
                    if new_what.ndim < current.ndim:
                        for _ in range(current.ndim - new_what.ndim):
                            new_what = new_what.unsqueeze(0)
                    new_what = new_what.expand_as(current).contiguous()
            if torch.is_tensor(new_what) and float(new_what.detach().abs().sum()) < 1e-9:
                # Pure pass-through op on a zeroed codebook would leave
                # .what unchanged; inject the pos_vec's first-nWhat entries
                # so downstream code sees a real update.  Broadcasts across
                # the leading batch/slot axes.
                nwhat = int(self.subspace.nWhat)
                bump = pos_vec[:nwhat].to(
                    device=new_what.device, dtype=new_what.dtype)
                while bump.ndim < new_what.ndim:
                    bump = bump.unsqueeze(0)
                new_what = new_what + bump
            self.subspace.what.setW(new_what)

        # Step 5 -- resolve + optional codebook pass.
        self.resolve(self.subspace)
        if self.codebook and quantize:
            # Only quantize when the muxed event is populated.  In the
            # rule-dispatch path the subspace may carry only .what so the
            # codebook forward (which materialize()s a muxed tensor) would
            # get a None input.  Skip quantization in that case -- the
            # codebook can learn from the updated .what on the next
            # regular forward pass.
            ev = (self.subspace.event.getW()
                  if self.subspace.event is not None else None)
            if ev is not None:
                self.subspace.what.forward(self.subspace)

        return self.subspace

    def forward(self, subspace):
        """Concept->symbol forward.

        Dispatches to the rule-application path when the caller marks the
        incoming subspace with ``_rule_dispatch`` (see
        ``_build_incoming_subspace``); otherwise runs the PiLayer/grammar
        derivation/codebook pipeline.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        quantize = getattr(self, "quantize", True)
        is_last = getattr(self, "is_last", False)
        wordSpace = getattr(self, "wordSpace", None)
        if getattr(subspace, '_rule_dispatch', False):
            return self._forward_with_rule_dispatch(
                subspace, wordSpace=wordSpace, quantize=quantize)
        vspace = subspace
        if self.passThrough:
            return vspace
        vspace = self.forwardBegin(vspace)
        act = vspace.materialize()                        # [B, N, concept_dim]
        act = self.forwardSigma(act)                      # [B, N, symbol_dim]
        if quantize:
            act = self.l1_proximal(act)                   # sparsity bias only

        # Per-cell mask: zero NULL-padded AR cells before any state-mutating
        # downstream consumer (truth layer record, parse-stack push,
        # codebook quantize). valid_mask is None outside AR.
        vmask = self.subspace.valid_mask
        if vmask is not None and act is not None and act.dim() == 3:
            act = torch.where(
                vmask.flatten().view(-1, 1, 1),
                act,
                torch.zeros_like(act))

        if self.accumulateTruth > 0 and wordSpace is not None:
            truth_layer = getattr(wordSpace, 'truth_layer', None)
            if truth_layer is not None:
                basis = getattr(self.subspace, 'basis', None)
                # Vectorized truth staging (no host sync).
                #
                # Compute a per-cell trust score from activation magnitude
                # (Gate 1 of the legacy `should_store`), masked by
                # `valid_mask` and scaled by `accumulateTruth`. Stage every
                # cell into the truth_layer's pending buffer with its
                # trust; the post-brick `truth_layer.compact()` step drops
                # entries with trust below threshold and promotes survivors
                # to the persistent store.
                #
                # Novelty/consistency gates (Gates 2-3 of legacy
                # `should_store`) are no longer per-cell; they're handled
                # at compact-time against the persistent store, or dropped
                # in favor of the magnitude-based trust + later dedup.
                BK, N, D = act.shape
                norms = act.norm(dim=-1)                                  # [B*K, N]
                mag_score = norms.clamp(max=self._truth_min_magnitude) \
                            / max(self._truth_min_magnitude, 1e-8)        # [B*K, N]
                vmask = self.subspace.valid_mask
                if vmask is not None:
                    mag_score = mag_score \
                                * vmask.flatten().unsqueeze(-1).to(mag_score.dtype)
                trust = mag_score * float(self.accumulateTruth)           # [B*K, N]
                truth_layer.record_batch(
                    act.reshape(BK * N, D),
                    trust.reshape(BK * N),
                    degree=float(self.accumulateTruth),
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

        # Task 2.5 — Resolve [pos, neg] bivector to 1-D per-symbol activation
        # BEFORE the codebook sees it.  resolve() writes
        # subspace.activation = pos + neg ([B, N]), collapsing the bivector
        # to the scalar signal strength that the codebook quantizes.
        # This must happen after forwardSymbols (if called) so the demuxed
        # what-slot is already populated, and before any quantization branch
        # so that the snapped activation is written back correctly.
        self.resolve(self.subspace)

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
        #       hard-quantize forward: snap the resolved 1-D activation to
        #       the nearest codebook entry (Task 2.5).  The event (act)
        #       is stored for downstream layers unchanged; only
        #       subspace.activation is quantized.
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
                    vspace.errors.add(
                        "symbol_commitment", commit, weight=1.0,
                        space="SymbolicSpace", category="symbol")
            self._emit_symbol_terms(
                vspace, self._compute_symbol_terms(predicted, target=target))
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
                vspace.errors.add(
                    "symbol_commitment", commit, weight=1.0,
                    space="SymbolicSpace", category="symbol")
            # Also pick up the codebook-internal commit loss (VQ's own
            # encoder-commitment term). Previously discarded; now threaded
            # into the objective so it actually drives learning.
            cb_commit = getattr(self.subspace.what, "last_commit_loss", None)
            if cb_commit is not None and torch.is_tensor(cb_commit) and cb_commit.requires_grad:
                vspace.errors.add(
                    "codebook_commit", cb_commit, weight=1.0,
                    space="SymbolicSpace", category="symbol")
            self._emit_symbol_terms(
                vspace,
                self._compute_symbol_terms(predicted, target=quantized_detached))
        elif hard_quantize:
            # Task 2.5: quantize the resolved 1-D activation, not the event.
            # set_event() resets activation to the ones-bivector so we must
            # call resolve() AFTER set_event() to re-write the 1-D scalar.
            # Steps:
            #   1. Store the event (symbol vectors) for downstream layers.
            #   2. resolve() converts the quantized .what bivector → scalar.
            #   3. Snap the scalar to nearest codebook entry and write back.
            # The codebook nDim stays at 2 (preserving the VQ-VAE paths);
            # _nearest_symbol_target clips to n = min(1, 2) = 1, comparing
            # only the first column of each codebook entry -- Option A/B
            # compromise: 1-D projection via first column, no nDim rebuild.
            self.subspace.set_event(act)
            vspace = self.forwardEnd(self.subspace)
            # Re-resolve after set_event so activation reflects the quantized
            # bivector stored in subspace.what (which set_event populated).
            self.resolve(self.subspace)
            act_1d = self.subspace.activation.getW()   # [B, N]
            if act_1d is not None and act_1d.ndim == 2:
                predicted_1d = act_1d.unsqueeze(-1)    # [B, N, 1]
                target_1d = self._nearest_symbol_target(predicted_1d)
                if target_1d is not None:
                    # Snap the 1-D activation to the nearest codebook scalar.
                    self.subspace.activation.setW(target_1d.squeeze(-1))
                    self._emit_symbol_terms(
                        vspace,
                        self._compute_symbol_terms(predicted_1d, target=target_1d))
                else:
                    self._emit_symbol_terms(
                        vspace, self._compute_symbol_terms(predicted_1d))
            else:
                self._emit_symbol_terms(
                    vspace, self._compute_symbol_terms(act))
        else:
            self.subspace.set_event(act)
            vspace = self.forwardEnd(self.subspace)
            self._emit_symbol_terms(
                vspace, self._compute_symbol_terms(act))

        # ImpenetrableLayer: codebook-level regularizer, emitted once per forward.
        if (self.impenetrable_overlap > 0.0
                or self.impenetrable_variance > 0.0):
            imp = self.impenetrable_loss()
            if (imp is not None and torch.is_tensor(imp)
                    and (imp.requires_grad or imp.abs().item() > 0.0)):
                vspace.errors.add(
                    "symbol_impenetrable", imp, weight=1.0,
                    space="SymbolicSpace", category="symbol")

        vspace.normalize("symbols", target="what")   # range check
        vspace.normalize("symbols", target="where")  # range check
        return vspace

    def reverse(self, subspace):
        """Map symbol vectors back to concept vectors via PiLayer.reverse (Pi^-1).

        Reverse maps on nDim axis: [B, N, symbol_dim] -> [B, N, concept_dim].
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if self.passThrough:
            return vspace
        wordSpace = getattr(self, "wordSpace", None)
        vspace = self.reverseBegin(vspace)
        act = vspace.materialize()                        # [B, N, symbol_dim]
        if wordSpace is not None:
            act = wordSpace.reverseSymbols(act, self.subspace)
        if self.sortNetwork is not None:
            act = self.sortNetwork.reverse(act)
        act = self.reverseSigma(act)                      # [B, N, concept_dim]
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
        """Top-level: evaluate truth of the full stack -> scalar.

        Reads ``trueForward`` from the S-tier SyntacticLayer owned by
        WordSpace. Returns a passthrough when no WordSpace is wired
        (e.g. unit tests constructing a SymbolicSpace in isolation).
        """
        act = vspace.materialize(mode="activation")
        if wordSpace is None:
            return act
        layer = getattr(wordSpace, 'syntacticLayer', None)
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

        # OutputSpace aggregates the full symbol stream into the task output:
        # the LinearLayer must span all nSymbols*symbol_dim inputs (not act
        # per-slot). Force nInputDim to the flattened input width so
        # forwardBegin reshapes [B, nSymbols, symDim] -> [B, 1, flat_in].
        # nOutputDim stays at outputShape[1] so forwardEnd reshapes the
        # linear's [B, 1, nOutput*outputDim] back to [B, nOutput, outputDim].
        # Skip on degenerate zero shapes; getEncodedOutputSize would divide
        # by iS[0]*iS[1]=0.
        if not self.nonlinear_output and inputShape[0] > 0 and inputShape[1] > 0:
            flat_in = inputShape[0] * inputShape[1]
            self.nInputDim = flat_in
            self.subspace._nInputDim = flat_in

        if self.nonlinear_output:
            # PiLayer activation-mode path for butterfly symbol output
            nIn = inputShape[0]
            nOut = outputShape[0]
            self._piLayer = PiLayer(nIn, nOut, invertible=True,
                                    monotonic=True, nonlinear=True)
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
        self._batch_results = []
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
    def forward(self, subspace):
        """Acting: project symbols to task output."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if self.nonlinear_output:
            # Activation-mode: PiLayer on symbol activations [B, nSymbols] -> [B, nOutput]
            act = vspace.materialize(mode="activation")
            output = self._piLayer.forward(act)
            self.subspace.set_activation(output)
            return self.subspace

        x = self.forwardBegin(vspace, returnVectors=True)
        output = self.forwardLinear(x)
        if self.codebook:
            output = self.subspace.get_vectors().forward(output)
        vspace = self.forwardEnd(output, returnVectors=True)
        return vspace

    def reverse(self, subspace):
        """Being acted upon: map output back to symbolic space."""
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if self.nonlinear_output:
            # Activation-mode: PiLayer reverse [B, nOutput] -> [B, nSymbols]
            act = vspace.materialize(mode="activation")
            symbol_act = self._piLayer.reverse(act)
            self.subspace.set_activation(symbol_act)
            return self.subspace

        y = self.reverseBegin(vspace, returnVectors=True)
        self.subspace.set_event(y)
        self.subspace.denormalize("output", target="what")
        y = self.subspace.materialize()
        y = self.reverseLinear(y)
        vspace = self.reverseEnd(y, returnVectors=True)
        return vspace

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
        """Collect output from a completed batch (counterpart to data_loader).

        Results are cleared at the start of each runEpoch() via clearBatchResults().

        Args:
            result: BatchResult namedtuple from runBatch().
        """
        # Detach every tensor field so we don't pin the autograd graph
        # of the batch that produced the result.  The list grows across
        # batches within an epoch -- one retained graph per batch is the
        # primary memory leak this avoids.
        def _detach(v):
            if torch.is_tensor(v):
                return v.detach()
            if isinstance(v, (list, tuple)):
                return type(v)(_detach(x) for x in v)
            return v
        if hasattr(result, '_replace'):
            detached = result._replace(**{
                f: _detach(getattr(result, f)) for f in result._fields
            })
        else:
            detached = _detach(result)
        self._batch_results.append(detached)
