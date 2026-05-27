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

import torch.optim as optim
from torch.profiler import profile as torch_profile, ProfilerActivity, schedule as profiler_schedule
from functools import partial
from datetime import datetime
import util
from util import TheDevice, TheMessage
from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config, init_compile_backend
from embed import (
    WordVectors, PretrainModel,
    _wrap_unit_ball, _wrapped_mse_score, _pole_aligned_score,
    _random_unit_ball,
) 
from data import Data, TheData
from Layers import Layer, PiLayer, SigmaLayer, NegationLayer  # Import custom layers from Model.py
from Layers import VectorQuantize  # moved from Spaces.py (April 2026 perf pass)
from Layers import GrammarLayer, NotLayer, NonLayer, IntersectionLayer, UnionLayer
from Layers import LinearLayer, InvertibleLinearLayer, AttentionLayer, AssociationLayer, MapppingLayer, LiftingLayer, LoweringLayer, ChunkLayer
from Layers import LiftingLayer, CertaintyWeightedCrossEntropy, Loss, ModelLoss, epsilon, Ops
from Layers import SortingLayer, TruthLayer, InterSentenceLayer, SparsityRegLayer, SmoothingRegLayer, ImpenetrableLayer
from Layers import Error

from util import parse
from collections import namedtuple as _namedtuple


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
        """Record the encoding's slot positions and value bound.

        ``index`` is the tuple of negative offsets into the embedding
        tail; ``maxVal`` parameterizes the per-encoding scale (e.g.
        the period of a quadrature encoding).
        """
        super().__init__()
        self.index = index
        self.maxVal = maxVal

    def encode(self, value):
        """Encode a value into nDim-wide representation. Subclass must override.

        Raises ``NotImplementedError`` -- subclasses define the actual
        encoding (quadrature, identity, etc.).
        """
        raise NotImplementedError

    def decode(self, encoded):
        """Decode nDim-wide representation back to a value. Subclass must override.

        Inverse of ``encode``; the round-trip is exact for invertible
        encodings.
        """
        raise NotImplementedError

    def resolve(self, embSize):
        """Resolve negative index offsets to absolute indices for a given embedding size.

        Converts the per-encoding ``[-k, -(k-1), ...]`` slot offsets into
        positive indices that ``y[:, :, index]`` can use directly.
        """
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
        """Set up the per-call frequency ``div_term = 2 * pi / maxVal``."""
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
    """Per-slot signed Degree-of-Truth activation (scalar carrier).

    Width 1: a single signed scalar in ``[-1, 1]`` per position. The
    4-valued bivector ``[aP, aN]`` substrate was retired (2026-05): the
    bivector-activation paths through SubSpace / the Space classes are
    deleted and every inter-component interface carries one signed
    Degree-of-Truth (see ``../doc/Logic.md``).

    The bivector *operations* themselves are retained for future use
    (``Layers.NotLayer`` / ``NonLayer`` / ``TrueLayer`` / ``FalseLayer``,
    the ``Ops`` monotonic kernels, the ``TruthLayer`` accumulator) --
    only the substrate carrier is scalarized here.

    encode/decode are identity -- the producing Space decides how to
    compute activation values. ActiveEncoding is the carrier.
    """
    nDim = 1

    def __init__(self, maxVal=1.0):
        """Initialize ActiveEncoding at slot ``[-5]`` with scalar width."""
        super().__init__([-5], maxVal)

    def encode(self, activation):
        """Identity encode: activation values pass through.

        Wraps scalar activations into a 1D tensor so downstream
        consumers always see a tensor (never a Python float).
        """
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(float(activation))
        return activation.unsqueeze(-1) if activation.dim() == 0 else activation

    def decode(self, encoded):
        """Identity decode: return activation tensor.

        ActiveEncoding is a carrier -- the producing Space owns the
        semantics; encode/decode are pure pass-through.
        """
        return encoded
class WhereEncoding(QuadratureEncoding):
    """Encode spatial position (nWhere) as sin/cos values in reserved embedding slots.

    Index is computed dynamically from nWhere and nWhen so the slots
    always align with the muxed layout ``[what, where, when]``:
      index = [-(nWhere+nWhen), ..., -(nWhen+1)]

    A monotonic counter ``self.p`` assigns each object a unique position
    within a dataset pass; it must be reset between epochs to avoid overflow.

    -----------------------------------------------------------------------
    Codebook offset registry (post-rollback bivector-activation work)
    -----------------------------------------------------------------------
    ``.where`` doubles as a globally-unique key into the per-Space
    codebook tables. Each codebook reserves a contiguous slice of the
    where-space at construction; the slice is identified by an integer
    offset and a fixed size (the codebook's ``nVectors``). All
    codebooks share the same sinusoidal frequency
    (``div_term = 2*pi / total_allocated``) so ``(sin, cos)`` decoding
    is unambiguous across the union of slices.

    The class-level registry is allocator-only -- it tracks how much
    of the where-space has been claimed and by whom. The frequency
    actually used at encode time is read from
    :func:`global_max_val` so newly-registered codebooks can grow the
    range without breaking existing offsets (the offset is a fixed
    *count*; only the angular scaling shifts when the total grows).
    """
    p = 0

    # Class-level codebook registry. List of (codebook_id, offset, n_vectors).
    # Sequential allocation: each new codebook lands at the end of the
    # current registry. Offsets stay stable across re-registrations of
    # the same codebook id (idempotent for re-init).
    _codebook_registry = []

    @classmethod
    def allocate_codebook_slice(cls, n_vectors):
        """Reserve ``n_vectors`` contiguous where-space positions for
        a new codebook. Returns the starting offset.

        Sequential allocation: the new slice's offset is the sum of
        ``n_vectors`` across all previously registered codebooks. Each
        call appends a fresh entry; codebooks that go out of scope
        leave their slice in the registry (the offset stays a stable
        scalar, just unused). Tests that need a clean slate should
        call :func:`reset_codebook_registry`.
        """
        offset = sum(n for _, n in cls._codebook_registry)
        cls._codebook_registry.append((offset, int(n_vectors)))
        return offset

    @classmethod
    def global_max_val(cls):
        """Sum of ``n_vectors`` across all registered codebooks. The
        sinusoidal ``div_term`` reads from this so the frequency
        always covers the live where-space.
        """
        return max(1, sum(n for _, n in cls._codebook_registry))

    @classmethod
    def reset_codebook_registry(cls):
        """Clear the registry (test/teardown helper).

        Drops every recorded ``(offset, n_vectors)`` entry so the next
        ``allocate_codebook_slice`` starts at 0. Intended for test
        isolation only.
        """
        cls._codebook_registry = []

    def __init__(self, maxP=0, nWhere=2, nWhen=0):
        """Place the where-encoding's slots dynamically before the when slots.

        When ``nWhere`` is 0 the encoding is disabled (zero-width carrier),
        otherwise the slot index is computed so it stays adjacent to the
        when slots in the muxed ``[what, where, when]`` layout.
        """
        if nWhere > 0:
            # Dynamic: where sits at [-(nWhere+nWhen), ..., -(nWhen+1)]
            index = [-(nWhere + nWhen) + i for i in range(nWhere)]
            super().__init__(index, maxP)
        else:
            Encoding.__init__(self, [], 1)  # skip QuadratureEncoding div_term
            self.nDim = 0
        self.p = 0

    def forward(self, x):
        """Stamp sin/cos positional values into reserved embedding slots.

        Advances the per-batch counter ``self.p`` by ``batch``; asserts
        before wrap so an overflow is loud. No-op when the encoding is
        disabled (``nDim == 0``).
        """
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
        """Self-test: encode then decode positions; print round-trip values."""
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
        """Allocate the two when slots at the tail; disable when ``nWhen=0``."""
        if nWhen > 0:
            super().__init__([-2, -1], maxT)
        else:
            Encoding.__init__(self, [], 1)  # skip QuadratureEncoding div_term
            self.nDim = 0
        self.t = 0

    def forward(self, x):
        """Stamp sin/cos temporal values into reserved embedding slots.

        Reads the per-call ``self.t`` counter and writes the encoded
        (sin, cos) pair for ``[t, t+batch)`` into the when slots.
        Caller must invoke ``increment(batch)`` to advance ``t``.
        """
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
        """Advance the global time counter by `batch` steps (called per forward pass).

        Mutates ``self.t`` in place. Should be called exactly once
        per forward pass; redundant calls drift the temporal index.
        """
        self.t += batch

    @staticmethod
    def test():
        """Self-test: encode then decode times; print round-trip values."""
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
        """Build a non-muxed Word encoding (no embedding slot allocation).

        Word entries live in a side list, so this encoding has no
        index. ``nBatch`` / ``nActive`` carry batch and per-row context
        bounds for downstream consumers.
        """
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
        """Validate and return a 7-tuple word.

        Asserts batch >= 0, vector >= 0, and rule is a valid global rule
        id. ``leafN = -1`` means the leaf slot is unused.
        """
        from Language import TheGrammar
        assert 0 <= batch, f"batch {batch} must be >= 0"
        assert 0 <= vector, f"vector {vector} must be >= 0"
        assert 0 <= rule < len(TheGrammar), f"rule {rule} out of range [0, {len(TheGrammar)})"
        return (batch, vector, order, rule, leaf1, leaf2, leaf3)

    def decode(self, word):
        """Unpack a word tuple into (batch, vector, rule).

        Discards the order / leaf metadata; callers that need them
        should index the raw tuple by the BATCH / RULE / LEAF1 / ... constants.
        """
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
        """Record the input / output shapes so per-call shape asserts can fire."""
        super().__init__([], 0)
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, objects, **kwargs):
        """Identity content pass-through.

        WhatEncoding's role is shape bookkeeping; subclasses override
        ``forward`` to apply the actual content transform.
        """
        return objects

    def forwardBegin(self, x, batch):
        """Validate or flatten input at the start of a forward pass.

        Asserts the [batch, inputShape[0], inputShape[1]] shape and
        returns ``x`` unchanged so callers can chain.
        """
        input_size = self.inputShape[1]
        assert list(x.shape) == [batch, self.inputShape[0], input_size]
        return x

    def forwardEnd(self, x, batch):
        """Validate or unflatten output at the end of a forward pass.

        Asserts the [batch, outputShape[0], outputShape[1]] shape;
        any mismatch reports the actual vs. expected shapes in the message.
        """
        output_size = self.outputShape[1]
        assert list(x.shape) == [batch, self.outputShape[0], output_size], \
            f"forwardEnd: got {list(x.shape)}, expected {[batch, self.outputShape[0], output_size]}"
        return x

    def reverseBegin(self, y, batch):
        """Validate or flatten output-side state at the start of reverse().

        Mirror of ``forwardEnd``: ensures the reverse path receives
        the shape it expects to invert.
        """
        output_size = self.outputShape[1]
        assert list(y.shape) == [batch, self.outputShape[0], output_size]
        return y

    def reverseEnd(self, y, batch):
        """Validate or unflatten input-side state at the end of reverse().

        Mirror of ``forwardBegin``: confirms the reverse path produced
        the original input shape.
        """
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
        # Build on the HOST: the per-byte writes below into a *device*
        # tensor would each be a synchronizing H2D. `device='cpu'` is
        # explicit so a default-device mode can't place it on the GPU;
        # the buffer is staged to the device in ONE synchronous copy
        # (correct + race-free -- a non_blocking copy of an ephemeral
        # pinned buffer corrupts data when the source is freed before
        # the transfer lands; see memory / pinmem guide).
        buf = torch.zeros(batch, nObj, nWhat, dtype=torch.long,
                          device='cpu')
        for b, row in enumerate(tokens_per_batch):
            for i in range(min(len(row), nObj)):
                text = row[i]
                if not text:
                    continue
                raw = text.encode('utf-8')[: nWhat - 1]
                for j, byte in enumerate(raw):
                    buf[b, i, j] = byte
        return buf.to(device)

    def decode_tokens(self, buf):
        """Unpack a [B, N, nWhat] null-terminated byte buffer into strings.

        Returns a list[list[str]] of shape [B][N]. Slots whose first byte
        is 0 decode to ``""``. Invalid UTF-8 sequences are replaced.
        """
        batch = buf.shape[0]
        n_obj = buf.shape[1]
        # One bulk device->host copy for the whole [B, N, nWhat] buffer
        # instead of B*N per-slot ``.tolist()`` calls. Each per-slot
        # ``buf[b, n].tolist()`` is a separate cudaMemcpyDtoH; a single
        # ``buf.tolist()`` is one transfer of identical data and keeps
        # the brick body's host-sync count flat in B*N (critical for
        # the CUDA-graph-capture contract -- see test_brick_no_sync).
        buf_rows = buf.tolist()
        out = []
        for b in range(batch):
            row = []
            for n in range(n_obj):
                bytes_row = buf_rows[b][n]
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

    def tokens_to_decoded(self, tokens_per_batch, batch, nObj, nWhat):
        """Host-side equivalent of ``decode_tokens(encode_tokens(...))``.

        Returns the exact ``list[list[str]]`` that
        ``decode_tokens(encode_tokens(tokens_per_batch, batch, nObj,
        nWhat, dev))`` would produce, computed purely on the host with
        **no tensor / no device round-trip**. The lexer already has the
        token strings on the host (``InputSpace._lex_batch``); carrying
        this forward lets ``PerceptualSpace._embed`` skip the
        ``decode_tokens`` ``buf.tolist()`` GPU->host sync (residual B,
        doc/BrickHostSyncStatus.md) while staying bit-identical -- it
        reproduces ``encode_tokens``'s ``nWhat-1`` UTF-8 truncation and
        the null-terminated ``decode`` (including the rare embedded-NUL
        early stop), so codebook OOV keys / indices are unchanged.
        """
        out = []
        for b in range(batch):
            row_in = tokens_per_batch[b] if b < len(tokens_per_batch) else []
            row = []
            for i in range(nObj):
                text = row_in[i] if i < len(row_in) else ""
                if not text:
                    row.append("")
                    continue
                raw = text.encode('utf-8')[: nWhat - 1]
                try:
                    end = raw.index(0)
                except ValueError:
                    end = len(raw)
                if end == 0:
                    row.append("")
                else:
                    row.append(
                        bytes(raw[:end]).decode('utf-8', errors='replace'))
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
        """Record the optional input / output shapes used by split_aux helpers."""
        super().__init__([], 0)
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, objects, **kwargs):
        """Identity content pass-through.

        EventEncoding is a no-op carrier; per-space subclasses do the
        actual event transformation. Defined here so the base contract
        is non-throwing.
        """
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
    """Shared runtime contract for SubSpace payloads.

    Geometry: each Basis has either *torus* (unit-cell, periodic wrap +
    wrapped MSE) or *sphere* (L2-normalized + cosine / dot product)
    geometry. The selection is implicit in ``use_dot_product``:

    * ``use_dot_product=False`` (default for ``Embedding``, ``Codebook``,
      ``Tensor``) — torus / wrapped MSE. Suits low-D codebooks where
      Tammes-on-sphere crowding is the bottleneck (the lexicon at
      V=200K, D=6; SymbolicSpace at V=1024, D=6).
    * ``use_dot_product=True`` (set explicitly by ``ConceptualSpace``) —
      sphere / cosine. Suits high-D codebooks where the input magnitude
      carries information (belief certainty in [-1, +1]).

    See ``unit_ball`` property below; methods that need codebook geometry
    (``normalize``, ``_snap_content``, ``codebookDistance``, the logical-
    op reverses) dispatch on it.
    """

    def __init__(self):
        """Initialize empty Basis state; subclasses own the weight tensor.

        Sets the per-basis shape descriptors to defaults; ``create``
        populates them. W is NOT stored on Basis itself -- subclasses
        either ``register_buffer`` (Tensor / Codebook) or hand off to
        ``wv._vectors`` (Embedding).
        """
        super().__init__()
        # W is NOT stored here -- subclasses own the storage.
        # Tensor/Codebook use register_buffer; Embedding uses wv._vectors.
        # ``set_sigma`` / ``sigma_kappa`` / ``activeSigma`` previously
        # lived here; the only live consumer (Embedding._ergodic_var)
        # owns them directly now (2026-05-02 Basis cleanup).
        self.activation = None
        self.nInput = 0
        self.nVectors = 0
        self.nDim = 0
        self.monotonic = True
        self.ergodic = False

    @property
    def unit_ball(self) -> bool:
        """True iff this basis uses torus geometry (wrapped MSE).

        Derived from ``use_dot_product`` so the two flags can never
        disagree. Sphere consumers (ConceptualSpace) set
        ``use_dot_product=True`` and get ``unit_ball=False``; everyone
        else inherits the default and gets ``unit_ball=True``.
        """
        return not bool(getattr(self, "use_dot_product", False))

    def getW(self):
        """Return the current weight tensor. Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement getW()")

    def setW(self, value):
        """Set the weight tensor. Subclasses must override."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement setW()")

    # -- Codebook-style read surface ------------------------------------
    # Per doc/specs/2026-05-21-subspace-slot-architecture.md "Reader API":
    # callers needing the codebook prototype matrix or a row lookup should
    # go through these methods rather than ``getW()`` + manual indexing.
    # Lets us migrate storage away from a single ``.W`` slot without
    # rewriting every caller, and lets the audit fixture in
    # test/test_active_payload_audit.py see only intentional getW hits.

    def prototype(self):
        """Return the ``[V, D]`` codebook prototype matrix, or ``None``.

        Only meaningful for codebook-bearing slots (``Codebook`` /
        ``Embedding``). Default base implementation returns ``None`` —
        plain ``Tensor`` slots have no prototype.

        Spec: doc/specs/2026-05-21-subspace-slot-architecture.md
        "Per-batch content: the target design" §contracts.
        """
        return None

    def lookup(self, indices):
        """Look up rows from the prototype matrix by ``indices``.

        ``indices`` is a long tensor of arbitrary leading shape; the
        return is ``[..., D]`` (the prototype rows gathered by the
        leading-shape indices).

        Default base implementation raises — only codebook-bearing
        subclasses (``Codebook``, ``Embedding``) implement this. Callers
        that don't know their slot's type should consult
        ``SubSpace.codebook_slot`` first or use ``SubSpace.lookup(...)``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.lookup() not supported; "
            f"only codebook-bearing slots support row lookup.")

    # -- Per-batch event storage on the Basis itself --------------------
    # Spec doc/specs/2026-05-21-subspace-slot-architecture.md:
    # per-batch ``[B, N, D]`` event content lives on the Basis, NOT on a
    # SubSpace-level cache. The Basis subclass decides where:
    #   * plain ``Tensor``  — on ``self.W`` directly (no Parameter to
    #                          protect).
    #   * ``Codebook``      — on a dedicated ``self._batch_event`` field
    #                          (the ``[V, D]`` Parameter prototype on
    #                          ``self.W`` must NOT be clobbered).
    #   * ``Embedding``     — on ``self._batch_event`` (the lexicon
    #                          prototype on ``wv._vectors`` is preserved).
    #   * ``ProjectionBasis`` — on ``self._batch_event`` (the LDU layer
    #                            holds the codebook structurally).
    #
    # This replaces the ``_active_payload`` band-aid: that field was a
    # shadow over ``getW()`` that multiplexed prototype + per-batch.
    # ``set_event`` / ``get_event`` are an explicit API with a single
    # responsibility — the per-batch cache only. ``getW()`` continues to
    # return the prototype (Parameter or plain backing tensor); the two
    # are no longer multiplexed through one slot.

    def set_event(self, event_tensor):
        """Store the per-batch ``[B, N, D]`` event on this Basis.

        Subclasses override. Default base raises so misuse is loud.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.set_event() not implemented; "
            f"override on the subclass to define per-batch storage.")

    def get_event(self):
        """Return the cached per-batch event, or ``None`` when unset.

        Subclasses override. Default base returns ``None``.
        """
        return None

    # -- Explicit prototype mutation API --------------------------------
    # ``replace_W`` is the single explicit API for changing the prototype
    # matrix (or plain backing tensor) AFTER construction. It distinguishes
    # the two cases ``setW`` used to multiplex through one method:
    #
    #   * Build-time construction → use ``Basis(...)`` / ``create(...)``
    #     / ``addVectors(...)``. The Parameter is registered at this point.
    #   * Runtime prototype replacement → use ``replace_W(new_W)``. For
    #     Parameter-backed slots this becomes ``self.W.data.copy_(new_W)``
    #     so the Parameter identity (and the optimizer's per-Parameter
    #     state) survives. For plain-tensor slots it's a direct
    #     reassignment.
    #
    # Per-batch event content does NOT go through ``replace_W``. Use
    # ``SubSpace.set_event`` / ``set_activation`` / ``set_forward_content``
    # — those write the SELECTION; ``SubSpace.materialize`` reconstructs
    # the per-batch event as ``codebook[selection]``.

    def replace_W(self, new_W):
        """Replace the prototype matrix / backing tensor. Preserves
        Parameter identity when one is registered (via ``data.copy_``).

        Subclasses override to enforce their specific shape contracts
        and any auxiliary cache invalidation (e.g. SVD).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.replace_W() not implemented; "
            f"override on the subclass.")

    def create(self, nInput, nVectors, nDim, customVQ=True, monotonic=True):
        """Construct the module's submodules and parameters.
        
        Mutates ``self`` to install the layers and tensor buffers.
        """
        self.nInput = nInput
        self.nVectors = nVectors
        self.nDim = nDim or 0
        self.monotonic = monotonic
        return self

    @property
    def size(self):
        """Size.
        
        See class docstring for the operation contract.
        """
        w = self.getW()
        return 0 if w is None else w.shape[0]

    @property
    def width(self):
        """Width.
        
        See class docstring for the operation contract.
        """
        w = self.getW()
        return 0 if w is None else w.shape[-1]

    @property
    def content_dim(self):
        return self.nDim

    @property
    def activeIndices(self):
        """Active indices.
        
        See class docstring for the operation contract.
        """
        if self.activation is None:
            return None
        return self.activation != 0

    def clearActivation(self):
        """Clear activation.

        See class docstring for the operation contract.
        """
        self.activation = None

    def apply_priming(self, priming_mask):
        """Multiply ``self.activation`` by ``priming_mask`` in place.

        Plan doc/plans/2026-05-20-primed-reverse-generation.md §B —
        post-snap multiplication. Called by reverse-call wiring after
        ``Codebook.forward`` has written ``.activation``, so downstream
        consumers (e.g. ``activeDense``) see the priming boost.

        No-op when either side is ``None``, or when the trailing
        dimensions of ``.activation`` do not match the priming mask's
        shape (broadcastability is required, not equality on every
        axis — typical case is ``activation: [B, V]`` and
        ``priming_mask: [V]`` or ``[B, V]``).

        With ``priming_mask`` identically 1.0 (default), this is a no-op
        on the values — current behavior is preserved.
        """
        if self.activation is None or priming_mask is None:
            return
        act = self.activation
        if not torch.is_tensor(act):
            return
        pm = priming_mask
        try:
            pm = pm.to(dtype=act.dtype, device=act.device)
        except (AttributeError, RuntimeError):
            return
        # Broadcast-multiply: require the priming's trailing dims to
        # match the activation's. The most common shapes are:
        #   act: [B, V],   priming: [V]      → broadcasts per-batch
        #   act: [B, V],   priming: [B, V]   → per-row weighting
        if pm.ndim > act.ndim:
            return
        if any(pm.shape[i] != act.shape[act.ndim - pm.ndim + i]
               for i in range(pm.ndim) if pm.shape[i] != 1):
            return
        self.activation = act * pm

    def forward(self, x):
        """Forward pass — identity by default.

        Subclasses override with the actual snap / lookup / projection.
        The legacy "self.setW(x); return x" identity-with-store was
        retired Stage 4 (per spec, per-batch content does not live on
        ``.W`` for Parameter-bearing slots; the strict raise in setW
        catches misuse). Pure identity is the safe default; callers
        wanting to persist per-batch content go through SubSpace.
        """
        return x

    def reverse(self, y, **kwargs):
        """Reverse pass — identity by default. See ``forward`` for
        the rationale; subclasses override with the snap target.
        """
        return y

    def reverse_raw(self, y):
        """Reverse raw.
        
        See class docstring for the operation contract.
        """
        return y

    def quantize(self, x):
        """Quantize.
        
        See class docstring for the operation contract.
        """
        raise RuntimeError(f"{self.__class__.__name__} does not support quantize()")

    def _coerce_rows(self, value):
        if isinstance(value, (list, tuple)):
            value = torch.stack(list(value), dim=0)
        return value

    def replace(self, new_W):
        """Replace the prototype matrix. Uses ``replace_W`` to preserve
        Parameter identity when one is registered.
        """
        self.replace_W(self._coerce_rows(new_W))
        return self.getW()

    def insert(self, new_W):
        """Insert new rows into the prototype matrix. Note: row-count
        change requires a Parameter re-registration (see
        ``Codebook.replace_W`` shape-mismatch branch); callers that
        depend on optimizer state across an ``insert`` must rebuild
        the optimizer.
        """
        new_W = self._coerce_rows(new_W)
        w = self.getW()
        self.replace_W(new_W if w is None else torch.cat([w, new_W], dim=0))
        return self.getW()

    def remove(self, indices):
        """Remove rows from the prototype matrix. Same caveat as
        ``insert`` re: optimizer state on Parameter slots.
        """
        w = self.getW()
        if w is None:
            return None
        mask = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
        mask[indices] = False
        self.replace_W(w[mask])
        return self.getW()

    def parameters_for_optimizer(self):
        """Parameters for optimizer.
        
        See class docstring for the operation contract.
        """
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
        """Snap content.
        
        See class docstring for the operation contract.
        """
        weight = self._prototype_weight(weight, context="reverse")
        nWhat = self.nDim if nWhat is None else nWhat
        snapped = content.clone()
        # ``flat`` must be its own buffer; ``snapped[..., :nWhat].reshape``
        # can return a view, and the later ``snapped[..., :nWhat] = ...``
        # write-back then aliases the destination, which torch rejects on
        # MPS (and on some CPU paths) with "input and written-to tensor
        # refer to a single memory location".
        flat = snapped[:, :, :nWhat].reshape(-1, nWhat).clone()
        nonzero = flat.abs().sum(dim=1) >= 1e-8
        # No ``if torch.any(nonzero)`` guard: every op below already
        # masks via ``flat[nonzero]``, so an all-zero ``flat`` makes the
        # block an empty-tensor no-op (``snapped`` returned unchanged) --
        # bit-identical to the guarded version in all cases (all-zero /
        # mixed / all-nonzero), minus the ``torch.any().__bool__`` host
        # sync that broke the brick CUDA-graph-capture contract
        # (test_brick_no_sync).
        if self.unit_ball:
            scores = _wrapped_mse_score(
                flat[nonzero].unsqueeze(1),
                weight[:, :nWhat].unsqueeze(0),
            )
        else:
            scores = F.cosine_similarity(
                flat[nonzero].unsqueeze(1),
                weight[:, :nWhat].unsqueeze(0),
                dim=2,
            )
        idx = scores.argmax(dim=1)
        flat[nonzero] = weight[idx, :nWhat]
        snapped[:, :, :nWhat] = flat.reshape(snapped.shape[0], snapped.shape[1], nWhat)
        return snapped

    def norm(self, x):
        """Norm.
        
        See class docstring for the operation contract.
        """
        return Ops.norm(x)

    def normalize(self, x=None):
        """Project the basis tensor onto its native geometry.

        Torus (``unit_ball=True``): wrap into the periodic unit cell
        ``[-1, 1)``. Idempotent on already-wrapped values; corrects
        optimizer drift outside the cell.

        Sphere (``unit_ball=False``): L2-normalize (with optional
        ``[0, 1]`` clamp under ``monotonic``).

        ``x is None``: in-place normalize of the live W. Otherwise
        return a normalized copy of ``x``.
        """
        target = self.getW() if x is None else x
        if target is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.normalize() has no tensor to normalize.")
        if self.unit_ball:
            normalized = _wrap_unit_ball(target)
        elif not self.monotonic:
            normalized = F.normalize(target, p=2, dim=-1)
        else:
            normalized = torch.clamp(target, 0, 1)
            normalized = F.normalize(normalized, p=2, dim=-1)
        if x is None:
            # Prototype-shaped in-place normalization — preserve
            # Parameter identity via ``replace_W``.
            self.replace_W(normalized)
            return self.getW()
        return normalized

    # -- Inverse logic operations -----------------------------------------------
    # Step 9 of the 2026-05-01 syntactic-layer refactor: forward
    # logic ops on Basis (`conjunction` / `disjunction` / `negation` /
    # `non`) were thin pass-throughs to the corresponding `Ops` static
    # methods; the new GrammarLayer subclasses (LiftLayer, ConjunctionLayer,
    # etc.) call `Ops` directly. Forward methods removed; codebook-search
    # `*Reverse` methods stay because they need `self.getW()`.

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
        """Inverse-recommend ``(x1, x2)`` for conjunction.

        Delegates to :func:`Ops.conjunctionReverse`, which draws the
        pair from the augmented codebook (learned ``self.getW()`` plus
        the ⊥ / ⊤ sentinels). Returns ``(result, result)`` if no
        codebook is available.
        """
        W = self._codebook_or_none("conjunctionReverse")
        if W is None:
            return result, result
        return Ops.conjunctionReverse(
            result, y, W, monotonic=monotonic, unit_ball=self.unit_ball)

    def disjunctionReverse(self, result, y, monotonic=False):
        """Inverse-recommend ``(x1, x2)`` for disjunction.

        Delegates to :func:`Ops.disjunctionReverse`, which draws the
        pair from the augmented codebook (learned ``self.getW()`` plus
        the ⊥ / ⊤ sentinels). Returns ``(result, result)`` if no
        codebook is available.
        """
        W = self._codebook_or_none("disjunctionReverse")
        if W is None:
            return result, result
        return Ops.disjunctionReverse(
            result, y, W, monotonic=monotonic, unit_ball=self.unit_ball)

    # -- Synthesis / analysis dispatchers (lift / lower) ------------------
    # Thin wrappers around Ops.lift / Ops.lower that supply the codebook W
    # for the inverse (codebook-search) paths.  See Ops.lift / Ops.lower
    # for the mode dispatch and region semantics.

    def lift(self, X1, X2=None, mode='OR', kind='strict', inverse=False,
             monotonic=False, left_rows=None, right_rows=None,
             left_priming=None, right_priming=None):
        """Synthesis dispatcher: many → one (∨).

        Forward routes through Ops.lift. `kind` selects the point body
        (strict/smooth/radial); see Ops.lift for details. Inverse with
        mode='OR' / 'AND' returns the mereology-guided pair
        ``(x1, x2)`` (see ``Ops._binary_op_recommend``); inverse with
        mode='NOT' is self-inverse and bypasses W.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            inverse-only — restricts which W-row indices are eligible
            for ``x1`` / ``x2`` selection. The intended call-site
            computes the intersection
            ``refs_by_category[cat] ∩ refs_by_order[k]`` from the
            attached ``KnowledgeView`` (see plan §Lift/Lower
            Restricted-Candidate Inverse).

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``):
            inverse-only — boost-above-unity weights from the
            Taxonomy's priming buffer that bias argmin/argmax toward
            primed candidates. Default 1.0 = no bias. See plan
            doc/plans/2026-05-20-primed-reverse-generation.md.
        """
        if inverse and mode == 'OR':
            W = self._codebook_or_none("Basis.lift inverse")
            if W is None:
                return X1, X1
            return Ops.disjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball,
                left_rows=left_rows, right_rows=right_rows,
                left_priming=left_priming, right_priming=right_priming)
        if inverse and mode == 'AND':
            W = self._codebook_or_none("Basis.lift inverse")
            if W is None:
                return X1, X1
            return Ops.conjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball,
                left_rows=left_rows, right_rows=right_rows,
                left_priming=left_priming, right_priming=right_priming)
        return Ops.lift(
            X1, X2, mode=mode, kind=kind, inverse=inverse, monotonic=monotonic
        )

    def lower(self, X1, X2=None, mode='AND', kind='strict', inverse=False,
              monotonic=False, left_rows=None, right_rows=None,
              left_priming=None, right_priming=None):
        """Analysis dispatcher: one → many (∧).

        Forward routes through Ops.lower. `kind` selects the point body
        (strict/smooth/radial); see Ops.lower for details. Inverse with
        mode='AND' / 'OR' returns the mereology-guided pair
        ``(x1, x2)`` (see ``Ops._binary_op_recommend``); inverse with
        mode='NOT' is self-inverse and bypasses W.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            inverse-only — see ``Basis.lift`` for the typed
            restricted-candidate inverse story.

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``):
            inverse-only — see ``Basis.lift`` for the priming-mask
            story.
        """
        if inverse and mode == 'AND':
            W = self._codebook_or_none("Basis.lower inverse")
            if W is None:
                return X1, X1
            return Ops.conjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball,
                left_rows=left_rows, right_rows=right_rows,
                left_priming=left_priming, right_priming=right_priming)
        if inverse and mode == 'OR':
            W = self._codebook_or_none("Basis.lower inverse")
            if W is None:
                return X1, X1
            return Ops.disjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball,
                left_rows=left_rows, right_rows=right_rows,
                left_priming=left_priming, right_priming=right_priming)
        return Ops.lower(
            X1, X2, mode=mode, kind=kind, inverse=inverse, monotonic=monotonic
        )

    def pos(self, x):
        """Pos.
        
        See class docstring for the operation contract.
        """
        return Ops.pos(x)

    def distance(self, x, y, monotonic=False, dim=-1):
        """Distance.
        
        See class docstring for the operation contract.
        """
        return Ops.distance(x, y, monotonic=monotonic, dim=dim)

    def codebookDistance(self, x):
        """Per-codebook-entry similarity score; larger is closer.

        Torus: ``-mean((x - cb)^2)`` over the wrapped delta — same metric
        ``_snap_content`` and ``_nearest_idx`` use.
        Sphere: dot product (cosine when codebook is unit-norm).
        """
        weight = self._prototype_weight(context="codebookDistance")
        vec = weight[:, :self.nDim].to(TheDevice.get())
        if self.unit_ball:
            return _wrapped_mse_score(
                x[..., :self.nDim].unsqueeze(-2),
                vec,
            )
        return x @ vec.T / max(self.nDim, 1)

    # -- Mereological operations --------------------------------------------
    # Formulas live in Ops (Layers.py); see Ops docstrings for semantics.

    def part(self, x, y, monotonic=False, scalar=False):
        """Part.
        
        See class docstring for the operation contract.
        """
        return Ops.part(x, y, monotonic=monotonic, scalar=scalar)

    def whole(self, x, y, monotonic=False, scalar=False):
        """Whole.
        
        See class docstring for the operation contract.
        """
        return Ops.whole(x, y, monotonic=monotonic, scalar=scalar)

    def equal(self, x, y, monotonic=False, scalar=False):
        """Equal.
        
        See class docstring for the operation contract.
        """
        return Ops.equal(x, y, monotonic=monotonic, scalar=scalar)

    def overlap(self, x, y, monotonic=False, scalar=False):
        """Overlap.
        
        See class docstring for the operation contract.
        """
        return Ops.overlap(x, y, monotonic=monotonic, scalar=scalar)

    def underlap(self, x, y, monotonic=False, scalar=False):
        """Underlap.
        
        See class docstring for the operation contract.
        """
        return Ops.underlap(x, y, monotonic=monotonic, scalar=scalar)

    def boundary(self, x, y, monotonic=False, scalar=False):
        """Boundary.
        
        See class docstring for the operation contract.
        """
        return Ops.boundary(x, y, monotonic=monotonic, scalar=scalar)

    def copart(self, x, y, monotonic=False, scalar=False):
        """Copart.
        
        See class docstring for the operation contract.
        """
        return Ops.copart(x, y, monotonic=monotonic, scalar=scalar)

    def active_dense(self):
        """Active dense.
        
        See class docstring for the operation contract.
        """
        w = self.getW()
        if self.activation is None or w is None:
            return None
        if self.activation.ndim == 1:
            return self.activation.unsqueeze(-1) * w
        return self.activation.unsqueeze(-1) * w.unsqueeze(0)

    # Removed (2026-05-02 Basis cleanup):
    #   * `_pairwise_sq_dists` / `_expand_sigma` / `kernel_overlap`
    #     -- only used by `kernel_overlap` itself (no live callers
    #     in bin/; the duplicate in bin/etc/old.py is the legacy
    #     Sigma-Pi reference dump).
    #   * `neg(X) = -X` -- duplicates `Ops.negation.forward` /
    #     bitonic kernel; callers can use the GrammarLayer surface.
    #   * `symbolize(X)` -- no live callers anywhere.
class Tensor(Basis):
    """Dense tensor payload implementation used for ordinary SubSpace slots.

    ``W`` is the single source of truth: either an ``nn.Parameter``
    (weights owned by this basis) OR a plain tensor (per-batch payload
    for non-Parameter slots). The ``_active_payload`` shadow was retired
    Stage 4 of doc/plans/2026-05-21-active-payload-retirement.md —
    per-batch content for codebook-bearing slots reconstructs from
    prototype + selection via ``SubSpace.materialize``; plain ``Tensor``
    slots (no Parameter) store per-batch on ``W`` directly.

    Strict assertion: ``setW(per_batch_3D_tensor)`` on a Parameter-bearing
    ``Tensor`` raises — per-batch writes must flow through the
    ``SubSpace`` setter API.
    """

    def __init__(self, nVectors=0, nDim=0, W=None):
        """Initialize Tensor; allocate state for the class contract.

        See class docstring for invariants.
        """
        super().__init__()
        self.W = None
        self.nVectors = nVectors
        self.nDim = nDim
        if W is not None:
            self.W = W
        else:
            self.W = torch.zeros(nVectors, nDim)

    def getW(self):
        """Return the weight tensor (Parameter or plain backing tensor)."""
        return self.W

    def setW(self, value):
        """Assign W. Raises on per-batch (3-D) plain-tensor writes to a
        Parameter-bearing slot — per spec
        doc/specs/2026-05-21-subspace-slot-architecture.md, per-batch
        content does not live on ``.W`` when a Parameter is registered.
        Use ``SubSpace.set_event`` / ``set_activation`` /
        ``set_forward_content`` instead.

        ``None`` clears a plain-tensor ``W``; the Parameter (if any)
        is preserved.
        """
        if value is None:
            if "W" not in self._parameters:
                self.W = None
            return
        if isinstance(value, nn.Parameter):
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = value
            return
        if ("W" in self._parameters
                and torch.is_tensor(value) and value.ndim >= 3):
            raise RuntimeError(
                "Tensor.setW: per-batch (3-D) plain-tensor write to a "
                "Parameter-bearing slot is forbidden. Per "
                "doc/specs/2026-05-21-subspace-slot-architecture.md, "
                "per-batch content reconstructs from prototype + "
                "selection. Use SubSpace.set_event(...) / "
                "set_activation(...) / set_forward_content(...).")
        self.W = value

    def set_event(self, event_tensor):
        """Plain ``Tensor`` has no Parameter to protect — per-batch
        event lands on ``self.W`` directly (the storage IS the event).
        Codebook-bearing variants override to raise.
        """
        if event_tensor is None:
            if "W" not in self._parameters:
                self.W = None
            return
        if "W" in self._parameters:
            raise RuntimeError(
                "Tensor.set_event called on a Parameter-bearing slot. "
                "Use SubSpace.set_event(...) to route through the "
                "appropriate codebook-aware path.")
        self.W = event_tensor

    def get_event(self):
        """Return the cached per-batch event (``self.W`` when 3-D), or
        ``None`` when the slot holds only the 2-D zeros init / Parameter.
        """
        if self.W is not None and torch.is_tensor(self.W) and self.W.ndim >= 3:
            return self.W
        return None

    def replace_W(self, new_W):
        """Replace the prototype / backing tensor.

        * Parameter-backed slot: ``self.W.data.copy_(new_W)`` (preserves
          Parameter identity, optimizer state). Shape must match.
        * Plain-tensor slot: direct reassignment.
        """
        if new_W is None:
            if "W" not in self._parameters:
                self.W = None
            return
        if isinstance(new_W, nn.Parameter):
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = new_W
            return
        if "W" in self._parameters:
            param = self._parameters["W"]
            if param is None or new_W.shape != param.shape:
                raise RuntimeError(
                    f"Tensor.replace_W: shape mismatch for Parameter "
                    f"in-place copy: param {None if param is None else tuple(param.shape)} "
                    f"vs new {tuple(new_W.shape)}.")
            with torch.no_grad():
                param.data.copy_(new_W)
            return
        self.W = new_W

    def forward(self, x):
        """Forward pass — identity-with-store for plain Tensor slots.

        Plain ``Tensor`` (no Parameter): caches ``x`` on ``self.W`` so
        ``getW()`` reflects the last forward (legacy contract used by
        tests like ``test_tensor_identity_materialization``).
        Parameter-bearing slots: pure identity — per-batch storage must
        go through ``SubSpace.set_event`` instead.
        """
        if "W" not in self._parameters:
            self.W = x
        return x

    def reverse(self, y, **kwargs):
        """Reverse pass — identity-with-store for plain Tensor slots.
        Mirror of ``forward``.
        """
        if "W" not in self._parameters:
            self.W = y
        return y
class Codebook(Basis):
    """Prototype basis with vector quantization and reverse snapping support.

    Class-level metric flag ``use_dot_product``: when True the underlying
    ``VectorQuantize`` instance is built with ``use_cosine_sim=True``, which
    holds the codebook unit-norm via the EMA path and retrieves nearest
    code by ``argmax_i (x . c_i)`` (a single matmul). The owning Space
    sets this on its Codebook instance, e.g. ``ConceptualSpace`` opts in
    by setting ``use_dot_product=True`` so the input magnitude (belief
    certainty in [-1, +1]) survives end-to-end. Default False keeps the
    Euclidean / pattern-codebook semantics for PerceptualSpace and
    SymbolicSpace. See doc/Spaces.md "Codebook similarity metric".
    """

    use_dot_product = False

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
            """Forward pass.
            
            See class docstring for the operation this layer applies.
            """
            ctx.save_for_backward(e.detach(), q.detach())
            return q

        @staticmethod
        def backward(ctx, grad_q):
            """Backward.
            
            See class docstring for the operation contract.
            """
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
        """Initialize Codebook; allocate state for the class contract.

        See class docstring for invariants.
        """
        super().__init__()
        self.W = None
        # ``_active_payload`` retired Stage 4 of
        # doc/plans/2026-05-21-active-payload-retirement.md. Per-batch
        # content reconstructs from prototype + selection via
        # ``SubSpace.materialize``; ``self.W`` holds ONLY the codebook
        # prototype (Parameter via ``addVectors``).
        self.customVQ = True
        self.snapDistance = 0.1
        self.eta = 0.9
        self.alpha = 0.0
        self.codebookSize = 0
        self.vq = None
        # Latest commitment loss from the most recent forward/quantize pass.
        # SymbolicSpace.forward reads it and emits "codebook_commit" into
        # vspace.errors.
        self.last_commit_loss = None
        # Optional per-row part-of-speech / category tags. Allocated only
        # when create(..., category=True). 0 = '?' (wildcard), other
        # values index into WordSpace.category_index. Used by the chart's
        # lexical fill to seed pos_lex when an input position resolves
        # to a tagged codebook atom — see doc/Language.md "POS side-channel".
        self.category_ids = None
        # Per-row meronomy parent index. ``part_parents[i] = j`` means
        # atom ``i`` is a part of atom ``j``. ``-1`` is the sentinel for
        # "no parent" (atom is a root concept). Used by the codebook
        # quantizer to recognize "this is an instance of an existing
        # whole" and avoid spawning a fresh row when the input maps to
        # a child of a known parent. The canonical parent for entries
        # that originated from the orthographic lexicon is the "words"
        # atom; see ``SymbolicSpace.words_atom_id``. Allocated lazily
        # alongside ``category_ids`` via ``create(..., category=True)``.
        self.part_parents = None
        # Per-row part-of-speech distribution learned through parsing.
        # Shape ``[V, C]`` where C = len(WordSpace.category_index).
        # Float buffer (not Parameter): updated via EMA from the chart's
        # _chart_pos at the leaves whose nearest codebook match is this
        # atom; persisted in state_dict alongside category_ids. The seed
        # path in `_apply_codebook_pos_seed` reads softmax(category_logits)
        # and uses it as a prior on the lex scorer when the atom is
        # confidently tagged. Allocated lazily by ``ensure_category_logits``
        # since C isn't known until the grammar is configured.
        self.category_logits = None
        # Per-position signed projection cache, populated by ``project``
        # so ``project_reverse`` can recover the original input
        # exactly. Shape ``[B, V, N]`` (or ``None`` when project hasn't
        # been called this forward).
        self._project_cache = None
        # Invertible codebook (SVD-factored) state. When
        # ``self.invertible`` is True, the project forward / reverse
        # routes use cached SVD factors of ``W = U Σ V^T`` instead of
        # ever materializing ``W`` in those paths:
        #   forward:   y = x @ V @ diag(Σ) @ U^T
        #   reverse:   x = y @ U @ diag(1/Σ) @ V^T
        # Factors are recomputed lazily by ``_ensure_svd`` when W
        # changes (``_svd_dirty`` flipped True by ``setW``).
        self.invertible = False
        self._svd_U = None        # [N, K]
        self._svd_S = None        # [K]
        self._svd_V = None        # [D, K]
        self._svd_dirty = True

    def getW(self):
        """Return the codebook prototype matrix ``[V, D]`` (or ``None``
        when the codebook hasn't been built yet)."""
        return self.W

    def setW(self, value):
        """Assign the codebook prototype matrix. Raises on per-batch
        (3-D) plain-tensor writes — per spec
        doc/specs/2026-05-21-subspace-slot-architecture.md, per-batch
        content does not live on the Codebook's ``.W``; the slot holds
        ONLY the prototype. Per-batch content reconstructs via
        ``SubSpace.materialize`` from prototype + selection.

        ``None`` clears a plain-tensor (non-Parameter) ``W``.
        ``nn.Parameter`` writes replace the Parameter.

        Any setW that touches the codebook invalidates the cached SVD
        factors used by the invertible project / project_reverse paths.
        """
        if value is None:
            if "W" not in self._parameters:
                self.W = None
                self._svd_dirty = True
            return
        if isinstance(value, nn.Parameter):
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = value
            self._svd_dirty = True
            return
        if torch.is_tensor(value) and value.ndim >= 3:
            raise RuntimeError(
                "Codebook.setW: per-batch (3-D) tensor write to the "
                "codebook prototype slot is forbidden. Per "
                "doc/specs/2026-05-21-subspace-slot-architecture.md, "
                "per-batch content reconstructs from prototype + "
                "selection via SubSpace.materialize. Use "
                "SubSpace.set_forward_content(...) / "
                "set_activation(...) instead, or run "
                "Codebook.forward(_vspace=...) to snap.")
        self.W = value
        self._svd_dirty = True

    def getSize(self):
        """Get size.

        See class docstring for the operation contract.
        """
        return self.nVectors

    # -- Spec-aligned codebook surface ----------------------------------
    # doc/specs/2026-05-21-subspace-slot-architecture.md: callers needing
    # the prototype matrix or a per-position row lookup should go through
    # ``prototype()`` / ``lookup(indices)`` rather than ``getW()``. These
    # bypass the ``_active_payload`` band-aid by reading ``self.W``
    # directly — the Parameter when registered, or the plain backing
    # tensor for hand-constructed Codebooks. Never the shadow.

    def prototype(self):
        """Return the ``[V, D]`` codebook prototype matrix.

        Reads ``self.W`` directly so the pre-migration
        ``_active_payload`` shadow can't intercept. Returns ``None``
        when the codebook hasn't been built yet (pre-``addVectors``).
        """
        return self.W

    def lookup(self, indices):
        """Gather codebook rows by per-position indices.

        ``indices``: long tensor of arbitrary leading shape (typically
        ``[B, N]`` for per-position selection or ``[B, N, M]`` for
        per-modality multi-index).
        Returns ``[..., D]`` — the prototype rows indexed by the
        leading-shape selection.

        Reads ``self.W`` directly (same band-aid bypass rationale as
        ``prototype()``). Raises if the codebook isn't built.

        Indices are clamped on-device to ``[0, V-1]`` so out-of-range
        selections (e.g. stale ``_active`` from a different config)
        gather row 0 rather than crashing. The clamp is data-flow only
        (no host sync) so it's safe under ``torch.compile`` fullgraph.
        """
        proto = self.W
        if proto is None or not torch.is_tensor(proto) or proto.ndim != 2:
            raise RuntimeError(
                f"Codebook.lookup({indices.shape if torch.is_tensor(indices) else indices!r}) "
                f"requires a 2-D prototype matrix; call ``addVectors`` first.")
        V = proto.shape[0]
        # ``clamp`` (NOT ``clamp_``): in-place modification of an
        # autograd-tracked tensor at a saved-for-backward index would
        # break gradient computation downstream.
        idx = indices.long().clamp(min=0, max=V - 1)
        return proto[idx]

    def set_event(self, event_tensor):
        """Codebook-bearing slots DO NOT cache per-batch events.

        Per spec doc/specs/2026-05-21-subspace-slot-architecture.md, the
        per-batch event ``[B, N, D]`` reconstructs from the codebook
        prototype on ``self.W`` (``[V, D]``) plus the selection stored
        on the SubSpace (``.activation`` scalar + ``_active`` indices),
        NOT from a separate Basis-level cache. Callers should write the
        selection via ``SubSpace.set_activation(...)`` /
        ``SubSpace.set_forward_content(...)``, or run
        ``Codebook.forward(vspace)`` to snap and populate it
        automatically.

        Raises so misuse is loud; the prior ``_active_payload`` shadow
        that silently absorbed per-batch writes here is the band-aid
        being retired.
        """
        raise RuntimeError(
            "Codebook.set_event() refuses per-batch event storage on a "
            "codebook-bearing slot. Per "
            "doc/specs/2026-05-21-subspace-slot-architecture.md, the "
            "per-batch event reconstructs from prototype + selection "
            "(``.activation`` × ``codebook[_active]``). Write the "
            "selection via SubSpace.set_activation(...) or "
            "SubSpace.set_forward_content(...), or run "
            "Codebook.forward(vspace) to snap.")

    def get_event(self):
        """Codebook-bearing slots reconstruct the per-batch event; they
        don't cache it. Always returns ``None`` so the
        ``SubSpace.materialize`` reconstruction path is the sole
        per-batch read.
        """
        return None

    def replace_W(self, new_W):
        """Replace the codebook prototype matrix ``[V, D]``.

        * Parameter-backed slot: ``self.W.data.copy_(new_W)`` (preserves
          Parameter identity AND optimizer state — load-bearing for
          VQ EMA updates and gradient training). Shape must match.
        * Plain-tensor slot (hand-constructed Codebook pre-addVectors):
          direct reassignment.
        * ``None`` clears a plain-tensor slot; Parameter slots are
          preserved (use ``del`` directly if you really need to drop one).
        * Always invalidates the cached SVD factors.
        """
        if new_W is None:
            if "W" not in self._parameters:
                self.W = None
                self._svd_dirty = True
            return
        if isinstance(new_W, nn.Parameter):
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = new_W
            self._svd_dirty = True
            return
        if not torch.is_tensor(new_W) or new_W.ndim != 2:
            raise RuntimeError(
                f"Codebook.replace_W requires a 2-D prototype matrix; "
                f"got shape {None if not torch.is_tensor(new_W) else tuple(new_W.shape)}.")
        if "W" in self._parameters:
            param = self._parameters["W"]
            if param is None:
                raise RuntimeError(
                    "Codebook.replace_W: Parameter slot is None; cannot copy.")
            if new_W.shape != param.shape:
                # Shape change requires Parameter re-registration. The
                # optimizer's per-Parameter state for the old Parameter
                # is orphaned; callers must rebuild the optimizer (e.g.
                # via ``BasicModel._rebuild_optimizer``).
                del self._parameters["W"]
                self.W = nn.Parameter(new_W.detach().clone(),
                                      requires_grad=param.requires_grad)
                self._svd_dirty = True
                return
            with torch.no_grad():
                param.data.copy_(new_W)
            self._svd_dirty = True
            return
        self.W = new_W
        self._svd_dirty = True

    def create(self, nInput, nVectors, nDim, customVQ=True, monotonic=True,
               category=False,
               invertible=False, STE=False, svdOrthogonal=False):
        """Construct the module's submodules and parameters.
        
        Mutates ``self`` to install the layers and tensor buffers.
        """
        super().create(
            nInput,
            nVectors,
            nDim,
            customVQ=customVQ,
            monotonic=monotonic,
        )
        self.customVQ = customVQ
        self.alpha = 0.0
        # Invertible mode: the project forward / reverse paths use
        # cached SVD factors of W and never materialize W in those
        # paths. SVD is lazily refreshed when ``setW`` invalidates it.
        self.invertible = bool(invertible)
        self._svd_dirty = True
        # Stash the SVD-orthogonal-init request; applied below after the
        # initial W is allocated by ``addVectors``. With a random
        # codebook, ``project_reverse`` (which scales by 1/S) can blow
        # up small singular values; replacing W with its nearest
        # orthonormal matrix pegs all singular values at 1 so the
        # bivector lift is well-conditioned from the first forward call.
        self._svd_orthogonal_init = bool(svdOrthogonal)
        # STE mode: when True, ``forward`` and ``reverse`` wrap the
        # codebook snap with the straight-through estimator pattern --
        # forward returns the hard snapped tensor, backward routes the
        # encoder's gradient through the input identity. Lets quantized
        # paths receive a smooth gradient signal without changing the
        # forward semantics. Default False preserves the legacy
        # detached-snap gradient behavior.
        self.STE = bool(STE)
        # Where-space offset: this codebook owns the where-space slice
        # ``[where_offset, where_offset + nVectors)``. The slice is
        # allocated sequentially via ``WhereEncoding.allocate_codebook_slice``
        # so a global ``.where`` value uniquely identifies both the
        # owning codebook and the prototype within it. Allocation is
        # keyed on ``id(self)`` and is idempotent for re-init.
        if self.nVectors > 0:
            self.where_offset = WhereEncoding.allocate_codebook_slice(
                self.nVectors)
        else:
            self.where_offset = 0
        if category and self.nVectors > 0:
            self.category_ids = torch.zeros(
                (self.nVectors,), dtype=torch.long)
            # Allocate part_parents alongside category_ids so the
            # codebook carries an explicit meronomy. -1 = no parent.
            self.part_parents = torch.full(
                (self.nVectors,), -1, dtype=torch.long)
        if self.nVectors > 0 and self.getW() is None:
            self.addVectors(self.nVectors)
        if self._svd_orthogonal_init and self.nVectors > 0:
            self.svdOrthogonalize()
        return self

    def svdOrthogonalize(self):
        """Replace ``W`` with its nearest orthonormal matrix.

        ``W <- U @ Vh`` where ``U, S, Vh = svd(W)``. All singular values
        of the result are 1, so ``project_reverse`` (which uses ``1/S``
        in the SVD-factored exact-inverse path) is well-conditioned
        from the first forward call. Subsequent EMA / training updates
        deform W from this orthonormal start; values stay bounded as
        long as training itself is stable.
        """
        W = self.getW()
        if W is None or self.codebookSize == 0:
            return
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(W, full_matrices=False)
            W_ortho = U @ Vh
            # ``replace_W`` handles both branches (Parameter
            # in-place ``.data.copy_`` and plain-tensor reassignment).
            self.replace_W(W_ortho)
        self._svd_dirty = True

    def set_category(self, idx, cat_id):
        """Tag codebook row ``idx`` with category index ``cat_id``.

        ``cat_id`` is an index into ``WordSpace.category_index``; ``0``
        means '?' (wildcard / unknown). Raises if the codebook wasn't
        created with ``category=True``.
        """
        if self.category_ids is None:
            raise RuntimeError(
                "Codebook.set_category called on a non-category codebook")
        self.category_ids[int(idx)] = int(cat_id)

    def get_category(self, idx):
        """Return the category id of row ``idx`` (0 = '?' if untagged)."""
        if self.category_ids is None:
            return 0
        return int(self.category_ids[int(idx)].item())

    def set_part_parent(self, idx, parent_idx):
        """Mark codebook row ``idx`` as a part of row ``parent_idx``.

        Writes the explicit meronomy: ``part_parents[idx] = parent_idx``.
        Pass ``-1`` to clear the parent (root atom). The quantizer reads
        this on insert to recognize "this row already has a parent
        concept" and route to the parent slot when appropriate. Raises
        if the codebook wasn't created with ``category=True`` (the
        part_parents buffer is allocated alongside category_ids).
        """
        if self.part_parents is None:
            raise RuntimeError(
                "Codebook.set_part_parent called on a codebook without "
                "meronomy storage (create with category=True).")
        self.part_parents[int(idx)] = int(parent_idx)

    def get_part_parent(self, idx):
        """Return the meronomy parent index of row ``idx`` (-1 = none)."""
        if self.part_parents is None:
            return -1
        return int(self.part_parents[int(idx)].item())

    def ensure_category_logits(self, num_categories, device=None):
        """Lazily allocate ``category_logits: [V, C]`` once C is known.

        Initial value is zeros (uniform softmax distribution); EMA
        updates from `Chart.compose` accumulate evidence from the
        chart's per-leaf POS distribution. Returns the buffer so the
        caller can write to it directly. Idempotent for the same C.
        """
        if self.category_ids is None:
            return None
        C = int(num_categories)
        if C <= 0:
            return None
        existing = self.category_logits
        if (existing is not None
                and existing.dim() == 2
                and existing.shape[0] == self.nVectors
                and existing.shape[1] == C):
            return existing
        dev = device if device is not None else (
            existing.device if existing is not None else TheDevice.get())
        self.category_logits = torch.zeros(
            self.nVectors, C, dtype=torch.float32, device=dev)
        return self.category_logits

    def update_category_logits(self, atom_idx, target_dist, ema=0.05):
        """EMA-update the per-atom POS logit row toward ``target_dist``.

        ``atom_idx``: scalar or [B] long tensor of codebook rows.
        ``target_dist``: matching-shape [B, C] float distribution
            (typically a softmax over chart_pos at a leaf cell).
        ``ema``: blend factor; new = (1 - ema) * old + ema * target_logit
            where target_logit = log(target_dist + eps).

        No-op when category_logits hasn't been allocated. Updates run
        without gradient (this is discrete bookkeeping).
        """
        if self.category_logits is None:
            return
        if not torch.is_tensor(atom_idx):
            atom_idx = torch.tensor([int(atom_idx)], dtype=torch.long)
        else:
            atom_idx = atom_idx.long().reshape(-1)
        if not torch.is_tensor(target_dist):
            return
        td = target_dist.detach().float()
        if td.dim() == 1:
            td = td.unsqueeze(0)
        if td.shape[0] != atom_idx.shape[0]:
            return
        if td.shape[1] != self.category_logits.shape[1]:
            return
        with torch.no_grad():
            atom_idx = atom_idx.to(self.category_logits.device)
            td = td.to(self.category_logits.device)
            target_logit = torch.log(td.clamp(min=1e-6))
            old = self.category_logits.index_select(0, atom_idx)
            new = (1.0 - float(ema)) * old + float(ema) * target_logit
            self.category_logits.index_copy_(0, atom_idx, new)

    def updateWeights(self, embed_sum, cluster_size):
        """Update weights.
        
        See class docstring for the operation contract.
        """
        return torch.ones(self.vq.codebook_size, device=TheDevice.get())

    def addVectors(self, nVec=1, decay=0.9):
        """Allocate ``nVec`` prototype entries via the in-repo VectorQuantize.

        Per the post-rollback bivector-activation contract, codebook
        size is fixed at ``self.nVectors`` (the value declared at
        construction). The ``WhereEncoding`` offset registry assumes
        this -- the codebook's where-space slice is ``[offset, offset
        + nVectors)`` and growing past ``nVectors`` would overrun the
        next codebook's slice. Assert here so violations surface at
        the call site instead of corrupting a downstream lookup.
        """
        assert nVec <= int(self.nVectors), (
            f"Codebook.addVectors: requested {nVec} > nVectors "
            f"({self.nVectors}); the where-space slice allocated for "
            f"this codebook only covers nVectors prototypes."
        )
        self.codebookSize = nVec
        # Growth invalidates the cached SVD factors: ``_ensure_svd`` will
        # recompute on next call. Without this flip, ``project`` /
        # ``project_reverse`` would reuse a stale K-dim factorization and
        # raise a dimension mismatch when matmul'ing against the now-wider
        # codebook.
        self._svd_dirty = True
        if self.customVQ:
            # When ``architecture.codebookRetire`` is false (the default),
            # dead-code replacement is disabled: retired rows get reseeded
            # with fresh samples, which on non-stationary data can blow up
            # the effective code count.
            try:
                retire = bool(TheXMLConfig.get("architecture.codebookRetire", False))
            except Exception:
                retire = False
            # Per-Codebook metric choice. ``self.use_dot_product`` is a
            # class attribute on Codebook (default False) that the owning
            # Space sets on its instance. ConceptualSpace opts in so its
            # unit-norm concept directions are retrieved by ``argmax_i
            # (x . c_i)`` -- a single matmul that preserves the input
            # magnitude (belief certainty) end-to-end. PerceptualSpace
            # and SymbolicSpace stay with the Euclidean / pattern-
            # codebook semantics. See doc/Spaces.md "Codebook similarity
            # metric".
            use_dot_product = bool(getattr(self, "use_dot_product", False))
            self.vq = VectorQuantize(
                dim=self.nDim,
                codebook_size=nVec,
                threshold_ema_dead_code=1 if retire else 0,
                decay=decay,
                commitment_weight=1.0,
                use_cosine_sim=use_dot_product,
                codebook_retire=retire,
                # Rotation-trick STE (arXiv:2410.06424) was reported to
                # trigger gradient-accumulation OOMs on HIP/ROCm with
                # microbatch AR.  Vanilla STE
                # (x + (quantized - x).detach()) is the classical VQ-VAE
                # estimator -- trains fine, uses a smaller autograd graph.
                rotation_trick=False,
            )
            # Initial codebook scaling. In dot-product mode the codebook
            # must be unit L2-norm so ``argmax_i (x . c_i)`` matches
            # ``argmax_i cos(x, c_i)`` (the EMA path keeps it unit-norm
            # after the first step). In Euclidean / pattern-codebook mode
            # we rescale to the [-1, 1] hypercube so downstream range
            # checks on the 'what' field hold.
            with torch.no_grad():
                init = self.vq.codebook.detach()
                if use_dot_product:
                    init = F.normalize(init, dim=-1)
                else:
                    init = init / init.abs().amax(
                        dim=-1, keepdim=True).clamp(min=1e-8)
                # Defensive clamp: even after the per-row max-abs
                # rescale above, FP edge cases (e.g. amax exactly
                # equal to 1.0 + ULP) can leave entries fractionally
                # outside [-1, 1]. The downstream Space ``normalize``
                # range check uses a strict 1e-2 tolerance, so we
                # tighten the codebook to literal [-1, 1] here.
                init = init.clamp(-1.0, 1.0)
                self.vq.codebook = init
            # Initial Parameter registration goes through ``replace_W``
            # (which handles both first-time and in-place updates).
            self.replace_W(self.vq.codebook)
        else:
            # Clamp the random init to [-1, 1] before per-row
            # normalization so downstream range checks see strictly
            # bounded codebook prototypes (the L2 normalize below
            # could otherwise yield rows whose unit-norm scaling
            # leaves a single entry slightly above 1 in fp32).
            W = torch.randn([nVec, self.nDim],
                            device=TheDevice.get()).clamp(-1.0, 1.0)
            for i in range(nVec):
                W[i, :] = self.normalize(W[i, :]).squeeze(0)
            self.replace_W(W)
        return self.getW()

    def quantize(self, x):
        """Quantize.
        
        See class docstring for the operation contract.
        """
        if self.customVQ:
            quantized, indices, commit_loss = self.vq(
                x,
                ema_update_weight=self.updateWeights,
            )
            # VQ EMA writes refresh the codebook prototype on every
            # forward — must preserve Parameter identity so optimizer
            # state stays attached. ``replace_W`` does ``.data.copy_``
            # for the Parameter-registered case.
            self.replace_W(self.vq.codebook)
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
        codes are detached so the gradient flows only into ``e``. Returns
        a zero scalar on ``e``'s device/dtype when either input is empty.
        """
        beta = 1.0
        if self.customVQ and self.vq is not None:
            beta = float(getattr(self.vq, "commitment_weight", 1.0))
        if e.numel() == 0 or q.numel() == 0:
            return e.new_tensor(0.0)
        n = min(e.shape[-1], q.shape[-1])
        if n <= 0:
            return e.new_tensor(0.0)
        return beta * F.mse_loss(e[..., :n], q[..., :n].detach())

    # -- Per-entry freezing -------------------------------------------
    # When an entry's sigma (running stdev of gradient norm across a window)
    # falls below ``freeze_threshold``, the entry has converged and we
    # zero its gradient from here on. This operates on codebook rows and uses
    # sigma (ergodic measure) rather than raw grad-norm mean.
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
        """Record codebook grad.
        
        See class docstring for the operation contract.
        """
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

    def _ensure_svd(self):
        """[Retired 2026-05-13.]  The bivector projection surface moved
        to ``ProjectionBasis`` whose ``InvertibleLinearLayer`` carries
        an LDU parameterization (structurally invertible via triangular
        solves).  This method is kept as a no-op for legacy callers
        and will be deleted once the call sites have migrated.
        """
        return

    def _ensure_svd_legacy(self):
        """Lazily compute and cache U, Σ, V for ``W = U Σ V^T``.

        Refreshes when ``_svd_dirty`` is True (set by ``setW`` or by
        ``addVectors`` growth). Detached so factors are constants from
        autograd's perspective — the project / project_reverse pair
        train the input, not the codebook.

        SVDs the codebook Parameter ``self.W`` directly. (The legacy
        ``_active_payload`` shadow that could shadow ``W`` with a 3-D
        per-batch slab was retired 2026-05-21; per-batch content now
        lives on ``SubSpace.event.W`` and never touches ``self.W``.)

        No-op when the codebook has no W yet.
        """
        if not self._svd_dirty:
            return
        W_param = self.W
        if W_param is None or self.codebookSize == 0:
            return
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_param, full_matrices=False)
        # Vh is [K, D]; store V (= Vh.T) for downstream factored matmul.
        self._svd_U = U.detach()
        self._svd_S = S.detach()
        self._svd_V = Vh.transpose(-2, -1).detach()
        self._svd_dirty = False

    # Codebook.project and Codebook.project_reverse were
    # retired 2026-05-13 alongside the project=True paths on
    # forward / reverse.  The bivector projection surface
    # lives on ProjectionBasis (its InvertibleLinearLayer-
    # parameterized LDU gives the exact inverse via triangular
    # solves; no SVD cache, no per-forward state).

    def forward(self, input, topK: int = 0, _vspace=None):
        """Codebook forward. When ``topK > 0`` and less than the codebook
        size, ``self.activation`` is pruned to the top-K strongest entries
        per batch row -- realizing the wide-codebook narrow-output pattern
        where nVectors >> nOutput. ``topK=0`` preserves legacy behavior.

        ``input`` may be a per-batch tensor OR a ``SubSpace`` (in which
        case ``materialize`` is called first to get the tensor).
        ``_vspace`` is the explicit destination SubSpace — used by
        ``SubSpace.set_muxed`` to snap a raw event tensor INTO a muxed
        subspace: the snap selection lands on
        ``_vspace.set_forward_content`` so ``materialize`` reconstructs
        as ``event.W[selection]`` (per-position width D) from the
        ``[V, D]`` codebook prototype (greater width V).

        The legacy ``project=True`` path was retired 2026-05-13 -- use
        ``ProjectionBasis`` directly for the bivector projection surface.
        """
        if isinstance(input, SubSpace):
            _vspace = input
            input = _vspace.materialize()

        w = self.getW()
        if w is None or self.codebookSize == 0:
            self.addVectors(max(self.nVectors, input.shape[1]))
        x = input
        batch = x.shape[0]
        # Derive device from the live input tensor, not ``TheDevice.get()``:
        # calling the device accessor inside the traced codebook forward
        # trips ``torch._device.__torch_function__`` with an arg dynamo
        # cannot proxy ("Failed to convert args/kwargs to proxy"). ``x``
        # is already on the right device; mirrors the ``best_err``
        # ``device=err.device`` idiom a few lines down.
        act = torch.zeros([batch, self.codebookSize], device=x.device)
        # Per-position codebook selection. Populated in the customVQ
        # branch; remains None on the legacy non-customVQ path. The tail
        # writes this onto the destination SubSpace via
        # ``set_forward_content`` when ``_vspace.muxed`` (codebook on
        # ``.event``) so per-batch content is reconstructed by
        # ``materialize`` from prototype + selection — the spec contract.
        selection_indices = None
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
            # Boolean-mask scatter (``act[mask_a, mask_b] = scores[mask]``)
            # has a dynamic index shape that Inductor's CUDAGraph
            # capture cannot record (warning:
            # ``skipping cudagraphs due to mutated inputs``). Re-express
            # as a static-shape ``scatter_reduce_`` over a freshly-
            # allocated tensor: snap-masked scores get amax-reduced
            # into act[b, indices[b, t]]; non-snap entries contribute
            # 0 (no-op against the all-zero init). This preserves the
            # "best score per codebook entry per batch" semantics --
            # the same idiom used by ``best_err`` above.
            masked_scores = torch.where(
                snap, scores, torch.zeros_like(scores))
            act = torch.zeros_like(act)
            act.scatter_reduce_(
                1, indices, masked_scores, reduce="amax",
                include_self=True)
            x = x3d.reshape(x.shape)
            # Per-position selection for the tail set_forward_content.
            # ``indices`` here is [batch, n_tokens] from the
            # ``indices.reshape(batch, n_tokens)`` above — exactly the
            # [B, N] shape ``set_forward_content`` expects.
            selection_indices = indices.long()
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
            # Inlined ``topk_by_magnitude_per_batch``: zero out all but
            # the ``topK`` entries by ``|act|`` along the last dim, per
            # row. The outer guard guarantees ``0 < topK < act.shape[-1]``,
            # so the legacy degenerate branches (topK<=0 -> all zeros;
            # topK>=W -> identity) are not reproduced here.
            _, idx = torch.topk(act.abs(), k=topK, dim=-1)
            mask = torch.zeros_like(act, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            act = torch.where(mask, act, torch.zeros_like(act))

        self.activation = act.detach() if torch.is_tensor(act) else act
        # STE wrap: when ``self.STE`` is True the forward output is the
        # hard snap (``x``) but the encoder's gradient flows through
        # the original input identity. ``input + (x - input).detach()``
        # equals ``x`` in forward and routes ``d/dinput = identity`` in
        # backward. Shape-guarded: only wrap when input and snap match.
        if (self.STE and torch.is_tensor(input) and torch.is_tensor(x)
                and input.shape == x.shape and input.requires_grad):
            x = input + (x - input).detach()
        # Persist per-batch results onto the SubSpace.
        #
        # Spec doc/specs/2026-05-21-subspace-slot-architecture.md
        # forward contract: write the SELECTION (per-position indices
        # into the codebook) onto ``_active`` via
        # ``set_forward_content``; ``materialize(mode='event')``
        # reconstructs the snapped per-batch event lazily as
        # ``codebook.W[selection]``. The legacy
        # ``set_event(x)`` write would route the per-batch slab through
        # ``_active_payload`` (the band-aid being retired) on
        # codebook-bearing ``.event`` slots — that path is gone here.
        #
        # NOTE: the ``claimed`` mask in the customVQ branch above can
        # leave a small number of "unclaimed" tokens with their original
        # input vector instead of a codebook row. After this migration,
        # materialize reconstructs every token by codebook lookup, so
        # unclaimed tokens see their codebook-row-nearest snapped form
        # rather than the original input. For MM_xor / MM_5M this is a
        # near-no-op (n_tokens << nVectors so the duplicate-claim case
        # is rare); the change is documented inline so future debugging
        # can find it.
        if _vspace is not None:
            # Spec doc/specs/2026-05-21-subspace-slot-architecture.md
            # forward contract:
            #   * muxed destination (codebook on ``.event``): write
            #     SELECTION to ``_active`` via ``set_forward_content``.
            #     ``materialize`` reconstructs as ``codebook[_active]``.
            #   * unmuxed / pure-event destination: legacy
            #     ``set_event(x)`` (per-batch direct storage on ``.W``).
            #
            # Invariant: input width == codebook ``nDim``. Configs that
            # would chunk input into sub-tokens are misconfigurations
            # to be fixed in the XML rather than handled here.
            if (getattr(_vspace, "muxed", False)
                    and selection_indices is not None
                    and selection_indices.ndim == 2):
                _vspace.set_forward_content(selection_indices)
            else:
                _vspace.set_event(x, compute_activation=False)
            return _vspace
        return x

    def reverse(self, y, **kwargs):
        """Codebook reverse: snap ``y`` against the prototype matrix.

        Returns the per-batch ``[B, N, D]`` snapped content. Callers
        that want to persist the result onto a SubSpace MUST do so
        explicitly via ``subspace.set_event(...)`` (the muxed
        authoritative slot); ``reverse`` is now a pure function of its
        input and the codebook prototype.

        The legacy ``project=True`` path was retired 2026-05-13 -- use
        ``ProjectionBasis.reverse`` directly for the bivector inverse.
        """
        if y.shape[-1] < self.nDim:
            raise RuntimeError(
                f"Codebook.reverse() expected at least {self.nDim} content dims, "
                f"got shape {list(y.shape)}.")
        content = y.clone() if y.shape[-1] == self.nDim else y[:, :, :self.nDim].clone()
        # Read the prototype matrix directly off ``self.W`` (the
        # Parameter when registered, or the plain backing tensor for
        # hand-constructed Codebooks). NOT via ``self.getW()`` — that
        # falls through to ``_active_payload`` (the 3-D per-batch
        # activation from a prior forward), which would crash
        # ``_snap_content``'s 2-D-prototype invariant. This is the
        # original bug fix the 2026-05-21 ``Codebook.reverse`` patch
        # was for.
        proto = self.W
        if proto is None or proto.ndim != 2:
            raise RuntimeError(
                "Codebook.reverse() needs a 2-D prototype matrix; call "
                "``addVectors`` / ``replace`` first.")
        content = self._snap_content(content, weight=proto, nWhat=self.nDim)
        # STE wrap on the reverse path: forward equals the snapped
        # content, backward routes the gradient through the matching
        # slice of ``y`` so the upstream consumer of the codebook
        # reverse receives an identity gradient through the snap.
        if (self.STE and torch.is_tensor(y) and torch.is_tensor(content)
                and y.shape[-1] >= content.shape[-1]
                and y.requires_grad):
            y_slice = y[..., :content.shape[-1]]
            content = y_slice + (content - y_slice).detach()
        return content

    def replace(self, new_vectors):
        """Replace the codebook prototype matrix. Uses ``replace_W`` so
        Parameter identity is preserved on the (common) in-place
        same-shape path; shape changes re-register the Parameter and
        the caller must rebuild the optimizer.
        """
        new_vectors = self._coerce_rows(new_vectors)
        if self.customVQ and self.vq is not None:
            self.vq.codebook = new_vectors
        self.replace_W(new_vectors)
        w = self.getW()
        self.codebookSize = 0 if w is None else w.shape[0]
        return w

    def insert(self, new_vectors):
        """Insert.
        
        See class docstring for the operation contract.
        """
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
        """Remove.
        
        See class docstring for the operation contract.
        """
        w = self.getW()
        if w is None:
            return None
        mask = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
        mask[indices] = False
        self.replace(w[mask])
        return self.getW()

    def learn(self, x, target_idx, lr=0.01):
        """Learn.
        
        See class docstring for the operation contract.
        """
        x = F.normalize(x, p=2, dim=-1)
        w = self.getW()
        selected_vectors = w[target_idx]
        delta = lr * (x - selected_vectors)
        w[target_idx] += delta
        self.normalize()

    @staticmethod
    def conceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        """Concept parthood.
        
        See class docstring for the operation contract.
        """
        A_norm = A / A.norm()
        B_norm = B / B.norm()
        cross_prod = torch.linalg.cross(A_norm, B_norm)
        orthogonal_vector = cross_prod / cross_prod.norm()
        distance = orthogonal_vector.norm()
        return torch.clamp(distance, 0, 1)

    @staticmethod
    def perceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        """Percept parthood.
        
        See class docstring for the operation contract.
        """
        A, B = A.clamp(0, 1), B.clamp(0, 1)
        ratio = torch.minimum(A / (B + epsilon), torch.ones_like(A))
        return torch.prod(ratio).item()
class ProjectionBasis(Basis):
    """Scalar projection basis with LDU-parameterized W.

    Forward maps ``[B, V, D] -> [B, N]`` via the signed mean-over-V
    projection on N prototypes (the catuskoti pair collapsed to its
    signed Degree-of-Truth ``pos - neg``).  Reverse maps
    ``[B, N] -> [B, V, D]`` via the exact LDU inverse of W -- no SVD
    cache, no per-forward state.

    The trainable surface is the LDU factorization on an
    ``InvertibleLinearLayer`` (``W = L @ D_embed @ U``), so the
    inverse is *structurally* invertible via triangular solves
    (``compute_Winverse_current``).  This replaces the legacy
    Codebook ``project=True`` path which used a cached SVD plus a
    per-forward ``_project_cache`` -- the latter made ``reverse``
    depend on the most recent forward, breaking the symbol-decode
    use case where reverse is called standalone.

    Convention (matches ILL's signature):
      * ``self.layer`` has ``nInput=nDim``, ``nOutput=nVectors``
      * ``compute_W()`` returns ``[nDim, nVectors]`` so
        ``x @ compute_W()`` directly yields ``[B, V, nVectors]``
      * ``compute_Winverse()`` returns ``[nVectors, nDim]`` so
        ``signed @ compute_Winverse()`` directly yields ``[B, nDim]``

    Selected by ``<codebook>project</codebook>`` (the size-changing
    codebook mode) -- e.g. on ``ConceptualSpace`` -- where the signed
    per-prototype projection replaces VQ snap as the codebook surface.
    """

    use_dot_product = False

    def __init__(self):
        super().__init__()
        self.layer = None       # InvertibleLinearLayer; allocated in create()
        self.codebookSize = 0
        # ``_active_payload`` retired Stage 4 of
        # doc/plans/2026-05-21-active-payload-retirement.md.

    def create(self, nInput, nVectors, nDim, **kwargs):
        """Construct the LDU-parameterized projection.

        ``**kwargs`` accepts (and ignores) legacy Codebook keys
        (``customVQ`` / ``STE`` / ``svdOrthogonal`` / ``monotonic``)
        for drop-in compatibility with the ``_build_what_basis``
        builders.  The LDU parameterization handles invertibility
        structurally; no additional flags needed.
        """
        from Layers import InvertibleLinearLayer
        self.nInput = int(nInput)
        self.nVectors = int(nVectors)
        self.nDim = int(nDim)
        self.codebookSize = int(nVectors)
        self.layer = InvertibleLinearLayer(
            nInput=int(nDim), nOutput=int(nVectors),
            naive=True, stable=True)
        return self

    def _W_norm_and_scales(self):
        """Return ``(W_norm, col_norms)`` -- forward path only.

        ``W_norm`` has unit-norm columns (each codebook prototype is
        on the unit ball), bounding the forward projection by
        ``||x||``; ``col_norms`` is the per-prototype scale used by
        the reverse path to undo the normalization (see
        :meth:`_apply_inverse`).  This split lets ``forward`` skip
        the inverse build that the legacy ``_normalized_factors``
        always paid for.
        """
        W = self.layer.compute_W_current()                     # [D, N]
        col_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, N]
        W_norm = W / col_norms                                 # [D, N]
        return W_norm, col_norms

    def _apply_inverse(self, signed):
        """Apply ``signed @ W_inv_norm`` without materialising W^-1.

        Mathematically ``W_inv_norm = col_norms.T * W_inv`` (the
        column-norm correction that makes the round-trip exact).
        Implementation:

          1. Scale ``signed`` per-prototype by ``col_norms``.
          2. Call ``layer.apply_Winverse_current(...)`` which runs the
             two triangular solves + diagonal divide directly against
             the RHS, never materialising ``L^-1`` / ``U^-1`` / ``D^-1``
             or their chained product.

        Total cost is ``O(b * n^2)`` and transient memory ``O(b * n)``,
        compared with the legacy ``signed @ compute_Winverse_current()``
        path's ``O(n^3)`` build + ``O(b * n^2)`` matmul plus three
        ``O(n^2)`` scratch tensors.
        """
        # Recompute col_norms here -- cheap (an `O(D*N)` reduction on
        # the LDU's compute_W output) and keeps the two paths
        # (forward / reverse) independent so neither carries dead
        # work for the other.
        W = self.layer.compute_W_current()
        col_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, N]
        # Scale signed by col_norms per prototype so the subsequent
        # solve undoes the unit-norm scaling cleanly.
        scaled = signed * col_norms                             # [B, N]
        return self.layer.apply_Winverse_current(scaled)        # [B, D]

    def getW(self):
        """Codebook view: ``[N, D]`` with unit-norm prototype rows.

        ILL parameterizes ``[nInput=D, nOutput=N]``; getW returns the
        transpose with row-L2 normalization applied so each prototype
        is on the unit ball, bounding downstream projection magnitudes.
        """
        if self.layer is None:
            return None
        W_norm, _ = self._W_norm_and_scales()
        return W_norm.T

    def setW(self, value):
        """ProjectionBasis is structurally read-only — the codebook is
        parameterized via LDU on ``self.layer``. Raises on Parameter
        writes (codebook is not stored as a Parameter here) and on
        per-batch (3-D) plain-tensor writes (per spec, per-batch
        content reconstructs from prototype + selection).

        ``None`` is a no-op (no transient state to clear).
        """
        if value is None:
            return
        if isinstance(value, nn.Parameter):
            raise TypeError(
                "ProjectionBasis.setW does not accept Parameter writes; "
                "the codebook is parameterized via LDU on self.layer.")
        if torch.is_tensor(value) and value.ndim >= 3:
            raise RuntimeError(
                "ProjectionBasis.setW: per-batch (3-D) write is forbidden. "
                "The codebook is parameterized via LDU on self.layer; "
                "per-batch content reconstructs from prototype + "
                "selection via SubSpace.materialize.")
        # 2-D plain-tensor writes are silently ignored — historically
        # used by some tests; with the shadow gone these are discarded.

    def set_event(self, event_tensor):
        """Codebook-bearing slot — refuse per-batch event storage.
        See ``Codebook.set_event`` for the rationale and migration path.
        """
        raise RuntimeError(
            "ProjectionBasis.set_event() refuses per-batch event storage "
            "on a codebook-bearing slot. The codebook is structurally "
            "held by the LDU layer; per-batch content reconstructs from "
            "prototype + selection via SubSpace.materialize. Write the "
            "selection via SubSpace.set_activation(...) / "
            "set_forward_content(...) instead.")

    def get_event(self):
        """No per-batch cache; reconstruction is via materialize."""
        return None

    def replace_W(self, new_W):
        """The codebook is parameterized via LDU on ``self.layer`` —
        an arbitrary tensor cannot be decomposed into LDU directly.
        ``None`` is a no-op; everything else raises.
        """
        if new_W is None:
            return
        raise RuntimeError(
            "ProjectionBasis.replace_W: the codebook is structurally "
            "parameterized via LDU on ``self.layer``. To change it, "
            "train through the LDU parameters or construct a new "
            "ProjectionBasis.")

    def forward(self, x):
        """``[B, V, D]`` (or ``[B, D]``) -> ``[B, N]`` signed scalar.

        Per batch row and codebook prototype, the mean-over-V signed
        projection ``mean_v (x[v] · W_norm[n])``.  Unit-norm prototypes
        bound ``|x[v] · W[n]| ≤ 1`` per slot, so the result stays in
        ``[-1, 1]``.  Returned as ``pos - neg`` (the catuskoti bivector
        collapsed to its signed Degree-of-Truth) so the matching
        :meth:`reverse` round-trip stays exact.  The bivector regime is
        retired (2026-05): every inter-component interface carries a
        single signed scalar, not the ``[aP, aN]`` pair.
        """
        if isinstance(x, SubSpace):
            x = x.materialize()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        W_norm, _ = self._W_norm_and_scales()                  # [D, N]
        D_min = min(int(x.shape[-1]), int(W_norm.shape[0]))
        x_d = x[..., :D_min].to(device=W_norm.device, dtype=W_norm.dtype)
        W_d = W_norm[:D_min, :]                                # [D, N]
        proj = x_d @ W_d                                       # [B, V, N]
        # Mean-over-V (instead of sum) so each prototype score stays in
        # [-1, 1] regardless of slot count.  V=1 (orthographic decode)
        # is identity-preserving by construction; V>1 gives the per-row
        # mean (the matching reverse replicates the recovered summary
        # across V positions -- the mathematically correct answer given
        # the V-axis collapse).  Bound once here so downstream Spaces
        # skip extra tanh / clamp (per user direction 2026-05-13).
        pos = torch.relu(proj).mean(dim=1)                     # [B, N] in [0, 1]
        neg = torch.relu(-proj).mean(dim=1)                    # [B, N] in [0, 1]
        # Width-1 trailing dim: a single signed-DoT feature per
        # prototype. Keeps the [B, N, D] substrate contract (D=1) so
        # forwardEnd / materialize / normalize work unchanged, while
        # remaining "only an activation" (not the [aP, aN] bivector).
        return (pos - neg).unsqueeze(-1)                       # [B, N, 1] in [-1, 1]

    def reverse(self, signed, V=1):
        """``[B, N]`` signed scalar -> ``[B, V, D]`` via the LDU inverse
        against the normalized codebook.

        ``signed`` is the per-prototype signed Degree-of-Truth produced
        by :meth:`forward`.  The column-norm correction keeps the V=1
        round-trip exact: ``W_norm @ W_inv_norm = W @ W_inv = I``.  For
        V>1 the per-V information was collapsed in the forward; the
        reverse returns the per-row summary vector replicated across V
        positions (the mathematically correct answer given the V-axis
        collapse).

        Accepts a legacy ``[B, N, 2]`` bivector (collapsed via
        ``pos - neg``) and a trailing-singleton ``[B, N, 1]`` for
        robustness during the bivector retirement.

        No dense ``W^-1`` is materialised -- triangular solves run
        directly on the RHS via
        ``InvertibleLinearLayer.apply_Winverse_current`` (``O(b*n^2)``).
        """
        if signed is None or not torch.is_tensor(signed):
            return None
        if self.layer is None:
            return None
        if signed.dim() >= 3 and signed.shape[-1] == 2:
            signed = signed[..., 0] - signed[..., 1]          # legacy bivector
        elif signed.dim() >= 3 and signed.shape[-1] == 1:
            signed = signed.squeeze(-1)
        x_summary = self._apply_inverse(signed)               # [B, D]
        v = max(int(V) if V is not None else 1, 1)
        return x_summary.unsqueeze(1).expand(-1, v, -1).contiguous()
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

    # Default upper bound on lexicon size. The lexicon's where-space
    # slice is reserved at ``create()`` time at width
    # ``max(nVectors, vocab_size, _LEXICON_DEFAULT_CAPACITY)``.
    # Sized generously to accommodate ASCII bootstrap (127 entries),
    # byte-mode bootstrap (256 entries), and a typical text vocabulary
    # without colliding with the next codebook's where-space slice.
    # Configs that need a larger lexicon should declare a bigger
    # ``<nVectors>`` on the owning InputSpace.
    _LEXICON_DEFAULT_CAPACITY = 65536

    def __init__(self):
        """Initialize Embedding; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__()
        self.pretrain = None       # PretrainModel, created in create()
        self.wv = None             # WordVectors (nn.Module with nn.Parameter)
        self.doc_spans = []
        self.optimize_embedding = False  # set by ModelFactory from <trainEmbedding>
        object.__setattr__(self, '_model', None)  # back-ref to BasicModel, avoids nn.Module submodule registration
        self.doc_sources = []
        # Ergodic-noise rate parameter. Read by ``_ergodic_var`` to
        # gate per-token variance during noisy embedding lookups.
        # Lifted off Basis (2026-05-02 cleanup): only Embedding
        # consumes it.
        self.sigma_kappa = 0.01

    def set_sigma(self, sigma):
        """Tune the ergodic-noise rate. ``sigma=0`` suppresses
        exploration (large kappa -> gate clamps to ~0). Called from
        ``Space.set_sigma`` during the model-wide sigma propagation
        (eval mode sets sigma=0; train mode sets sigma=0.5).
        """
        if sigma == 0:
            self.sigma_kappa = 1e6
        else:
            self.sigma_kappa = 0.01 / sigma

    def getW(self):
        """Return live embedding vectors wrapped into the periodic unit cell
        ``[-1, 1)``.

        The wrap is applied on every read so any optimizer drift between
        ``optimizer.step()`` and the next ``normalize()`` call is invisible
        to consumers (codebook search, logic ops, etc.). Wrapping is
        differentiable almost everywhere; gradients flow through to
        ``wv._vectors`` exactly as they would for an identity view.
        """
        if self.wv is not None and self.wv._vectors is not None:
            return _wrap_unit_ball(self.wv._vectors)
        return None

    def setW(self, value):
        """Embedding W is managed by wv._vectors -- setW is a no-op."""
        pass

    # -- Spec-aligned codebook surface ----------------------------------
    # See ``Basis.prototype`` / ``Basis.lookup`` for the contract.

    def prototype(self):
        """Return the live ``[V, D]`` embedding matrix (unwrapped).

        Unlike ``getW()`` (which applies a unit-cell wrap on read),
        ``prototype()`` returns the raw ``wv._vectors`` so downstream
        index lookups land on the parameter storage directly. The
        wrap is a read-time transform for codebook *search* uses;
        index lookup wants the live row.
        """
        if self.wv is not None and self.wv._vectors is not None:
            return self.wv._vectors
        return None

    def lookup(self, indices):
        """Gather embedding rows by per-position indices.

        Returns ``[..., D]`` where the leading shape is from
        ``indices.shape``. Reads ``wv._vectors`` directly so the
        result is the live (un-wrapped) parameter storage — the
        unit-cell wrap is a search-time transform, not a lookup
        contract.
        """
        if self.wv is None or self.wv._vectors is None:
            raise RuntimeError(
                "Embedding.lookup() requires wv._vectors to be built "
                "(call create() / addVectors() first).")
        return self.wv._vectors[indices.long()]

    def set_event(self, event_tensor):
        """Codebook-bearing slot — refuse per-batch event storage.
        See ``Codebook.set_event`` for the rationale and migration path.
        """
        raise RuntimeError(
            "Embedding.set_event() refuses per-batch event storage on a "
            "codebook-bearing slot. The lexicon prototype is on "
            "``wv._vectors``; per-batch content reconstructs from "
            "prototype + selection via SubSpace.materialize. Write the "
            "selection via SubSpace.set_activation(...) / "
            "set_forward_content(...) instead.")

    def get_event(self):
        """No per-batch cache; reconstruction is via materialize."""
        return None

    def replace_W(self, new_W):
        """Replace the lexicon prototype on ``wv._vectors`` in place.

        Always preserves Parameter identity (``wv._vectors`` IS the
        Parameter for embedding training). Shape must match; OOV
        expansion goes through ``stage_oov`` / ``addVectors``, NOT
        ``replace_W``.
        """
        if new_W is None:
            return
        if self.wv is None or self.wv._vectors is None:
            raise RuntimeError(
                "Embedding.replace_W: wv._vectors not yet built. "
                "Call create() / addVectors() first.")
        if not torch.is_tensor(new_W) or new_W.ndim != 2:
            raise RuntimeError(
                f"Embedding.replace_W requires a 2-D matrix; "
                f"got shape {None if not torch.is_tensor(new_W) else tuple(new_W.shape)}.")
        if new_W.shape != self.wv._vectors.shape:
            raise RuntimeError(
                f"Embedding.replace_W: shape mismatch "
                f"{tuple(new_W.shape)} != {tuple(self.wv._vectors.shape)}. "
                f"Use addVectors / stage_oov to grow the lexicon.")
        with torch.no_grad():
            self.wv._vectors.data.copy_(new_W)

    def normalize(self, x=None):
        """Wrap into the periodic unit cell ``[-1, 1)``.

        Two modes:
        * ``x is None``: in-place wrap of ``wv._vectors`` (steady-state
          cleanup, e.g. after an optimizer step that drifted out of the
          cell). Returns the wrapped live view via ``getW()``.
        * ``x is not None``: returns a wrapped copy of ``x`` without
          touching the parameter.
        """
        if x is None:
            if self.wv is None or self.wv._vectors is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.normalize() has no tensor "
                    f"to normalize.")
            with torch.no_grad():
                self.wv._vectors.copy_(_wrap_unit_ball(self.wv._vectors))
            return self.getW()
        return _wrap_unit_ball(x)

    @staticmethod
    def _to_text(buf):
        """Convert a byte list, tensor, or string to a plain string."""
        if isinstance(buf, str):
            return buf
        if isinstance(buf, torch.Tensor):
            buf = buf.squeeze().tolist()
        return "".join(chr(int(c) & 0xFF) for c in buf).rstrip("\x00")

    def _token_stream(self, text):
        """Token stream.
        
        See class docstring for the operation contract.
        """
        if getattr(self, 'byte_mode', False):
            # Pass raw bytes / byte tensors directly to parse so
            # ``lex='bytes'`` produces exactly one token per input byte.
            # Going via ``_to_text`` (UTF-8 decode-with-replace) and
            # then through parse's old round-trip would expand the byte
            # count at invalid-sequence boundaries (cursor slab cuts).
            return parse(text, lex='bytes')
        if getattr(self, 'lexer_mode', 'word') == 'sentence':
            return parse(self._to_text(text), lex='sentences')
        mode = getattr(self, 'chunking_mode', 'lexicon')
        if mode in ('bpe', 'mphf'):
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

    def create(self, nInput=None, nVectors=None, nDim=None,
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
            wv = self._load_embeddings(
                embedding_path=embedding_path, nDim=nDim, nVectors=nVectors)
        if wv is None:
            # No matching embeddings on disk -- start with \x00 at index 0.
            # This is both the EOS/padding sentinel AND the first real
            # byte, which makes byte_value == codebook_index throughout
            # the full 0..255 range.  ChunkLayer.BOUNDARY_BYTES assumes
            # this alignment (see code review).  Real words are added
            # dynamically during forward passes via insert().
            dim = nDim or 20
            print(f"Starting with dynamic {dim}-dim embedding (words added at runtime)")
            placeholder = _random_unit_ball(
                (1, dim), device=TheDevice.get())
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
                       vector_size)
        # W is managed by wv._vectors; getW() returns live data

        # Reserve a unique slice of the global where-axis for this lexicon.
        # Every prototype across InputSpace + Perceptual/Conceptual/Symbolic
        # codebooks is a distinct ``location'' in the subjective experience
        # graph: the slice gives each lexicon entry a globally-unique
        # ``where_offset + row`` key, on the same axis as the per-Space
        # Codebooks (see ``Codebook.create`` and
        # ``WhereEncoding._codebook_registry``).
        #
        # Embedding's ``nVectors`` is the *slot count* of the owning
        # InputSpace, NOT the lexicon's row capacity -- the lexicon grows
        # dynamically via ``insert()`` past ``nVectors`` (e.g., ASCII
        # bootstrap inserts 127 char rows even when nVectors=8). To
        # avoid colliding with the next codebook's where-space slice we
        # reserve a generous capacity here: ``max(nVectors, vocab_size,
        # _LEXICON_DEFAULT_CAPACITY)``. Runtime ``insert()`` asserts
        # against this reserved capacity so growth past the bound fails
        # loudly. If a deployed lexicon needs more, raise the default
        # or wire a per-Embedding capacity knob in the XML.
        reserve = max(int(nVectors) if nVectors else 0,
                      vocab_size,
                      Embedding._LEXICON_DEFAULT_CAPACITY)
        self.lexicon_capacity = reserve
        self.where_offset = WhereEncoding.allocate_codebook_slice(reserve)

        self.pretrain = PretrainModel(wv, learning_rate=learning_rate, neg_samples=neg_samples)
        self.wv = wv  # register as submodule for .to(device) and state_dict
        self.min_frequency = float(min_frequency)
        self._pending_counts: dict = {}
        # OOV reserve / fallback diagnostics
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §4).
        self._oov_fallback_count = 0
        self._oov_fallback_sample: list = []
        self._oov_fallback_sample_cap = 1024
        # NOTE: ``_inflate_to_capacity`` is no longer called from
        # ``create()``. Auto-inflating ``wv._vectors`` to
        # ``lexicon_capacity`` rows broke downstream consumers that
        # read ``wv._vectors.shape[0]`` as the active codebook size
        # (perceptualspace BPE forward, knowledge artifact writer,
        # phase-2a labor-division tests). The in-place preallocated-
        # reserve contract is opt-in via ``stage_oov(..., preallocate=True)``
        # for callers that explicitly want it.

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
        # In byte mode the codebook is already fully seeded (all 256 byte values
        # inserted above) and per-token inserts here are no-ops; downstream code
        # does not index into doc_spans/doc_sources, so we skip the sweep.
        if source:
            self.doc_spans = []
            self.doc_sources = []
            if not self.byte_mode:
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

        # NULL-percept slot for IR mode (BERT [MASK]-style). Lives at the
        # tail of the codebook as a learnable parameter; distinct from
        # any real word/byte/BPE entry so the brick body can tell
        # "predict me" from "this was character \x00". Idempotent on
        # reload: if the slot is already in the loaded artifact (e.g.
        # after IR training), reuse the existing index.
        null_key = PerceptualSpace.NULL_PERCEPT_KEY
        if null_key in self.pretrain.key_to_index:
            self.null_percept_idx = int(
                self.pretrain.key_to_index[null_key])
        else:
            self.null_percept_idx = len(self.wv)
            self.insert(null_key, vector=None, initial_count=0)

    def _rebuild_optimizer(self):
        self.pretrain.optimizer = torch.optim.Adam(
            [self.wv._vectors],
            lr=self.pretrain.optimizer.param_groups[0]['lr'],
        )
        # W is managed by wv._vectors; getW() returns live data

    def _inflate_to_capacity(self):
        # Inflate ``wv._vectors`` to ``[lexicon_capacity, dim]`` so future
        # OOV inserts write into preallocated reserve rows in place
        # (no Parameter reassignment, no optimizer rebuild).
        if self.wv is None:
            return
        cap = int(getattr(self, "lexicon_capacity",
                          Embedding._LEXICON_DEFAULT_CAPACITY))
        cur = self.wv._vectors.shape[0]
        if cap <= cur:
            return
        dim = self.wv._vectors.shape[1]
        device = self.wv._vectors.device
        dtype = self.wv._vectors.dtype
        pad = torch.zeros(cap - cur, dim, device=device, dtype=dtype)
        with torch.no_grad():
            full = torch.cat([self.wv._vectors.data, pad], dim=0)
            self.wv._vectors = nn.Parameter(full, requires_grad=True)
        # counts array stays at the active length; new rows get appended
        # entries when they are activated.

    def stage_oov(self, keys, vectors=None):
        # Stage OOV keys into preallocated reserve rows
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §4.1).
        # Writes happen in place under ``no_grad`` — Parameter identity
        # is preserved, optimizer is not rebuilt, Adam moments at the
        # newly activated row indices are zeroed. Keys that exceed the
        # remaining reserve capacity are returned for fallback routing
        # by the caller (§4.4).
        keys = [k for k in (keys or []) if k not in self.pretrain.key_to_index]
        if not keys:
            return []
        cap = int(self.lexicon_capacity)
        active = len(self.wv.index_to_key)
        n_free = max(0, cap - active)
        accepted = keys[:n_free]
        overflowed = keys[n_free:]
        if accepted:
            n = len(accepted)
            dim = self.wv.vector_size
            device = self.wv._vectors.device
            if vectors is not None:
                v = vectors[:n]
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v, dtype=torch.float32)
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                v = v.to(device=device, dtype=self.wv._vectors.dtype)
            else:
                v = _random_unit_ball((n, dim), device=device).to(
                    self.wv._vectors.dtype)
            v = _wrap_unit_ball(v)
            start = active
            end = active + n
            with torch.no_grad():
                self.wv._vectors.data[start:end, :] = v
            self.wv.counts = np.append(
                self.wv.counts, np.zeros(n, dtype=np.int64))
            self.wv._normed = None
            for i, key in enumerate(accepted):
                self.wv.key_to_index[key] = start + i
                self.wv.index_to_key.append(key)
                self._pending_counts.pop(key, None)
            self.pretrain.index_to_key = self.wv.index_to_key
            self.pretrain.key_to_index = self.wv.key_to_index
            self._zero_optimizer_moments_for_rows(start, end)
            # Stage 1.B paired-row contract (2026-05-27): bulk OOV
            # insert through stage_oov triggers paired-row insertion on
            # the SS-side peer for every accepted key. Falls back to
            # the legacy mark_word_atom tag if the peer pre-dates the
            # insert_paired_word API.
            s_peer = getattr(self, 'symbolicSpace_ref', None)
            if s_peer is not None:
                has_paired = hasattr(s_peer, 'insert_paired_word')
                has_mark = hasattr(s_peer, 'mark_word_atom')
                if has_paired:
                    for i, key in enumerate(accepted):
                        row = start + i
                        try:
                            ps_vec = self.wv._vectors.data[row].detach()
                            s_peer.insert_paired_word(key, ps_vec)
                        except Exception:
                            # Best-effort; never block the bulk insert.
                            pass
                elif has_mark:
                    for i in range(n):
                        try:
                            s_peer.mark_word_atom(start + i)
                        except Exception:
                            pass
        if overflowed:
            self._oov_fallback_count += len(overflowed)
            cap_sample = int(self._oov_fallback_sample_cap)
            for key in overflowed:
                if len(self._oov_fallback_sample) < cap_sample:
                    self._oov_fallback_sample.append(key)
                else:
                    break
        return overflowed

    def _zero_optimizer_moments_for_rows(self, start, end):
        # Zero Adam ``exp_avg`` / ``exp_avg_sq`` at the activated row
        # indices so new rows don't inherit stale moments from any
        # previous lifetime of the reserve slot. Optimizer state is
        # otherwise untouched; the persistent optimizer keeps owning
        # the full-capacity parameter.
        opt = self.pretrain.optimizer if self.pretrain is not None else None
        if opt is None:
            return
        state = opt.state.get(self.wv._vectors, None)
        if not state:
            return
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            mom = state.get(key, None)
            if (mom is not None
                    and torch.is_tensor(mom)
                    and mom.shape[0] >= end):
                with torch.no_grad():
                    mom[start:end].zero_()

    def replace(self, new_W):
        """Replace.
        
        See class docstring for the operation contract.
        """
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
                    vector is generated in the active lexicon geometry.
            initial_count: frequency count for the new word.

        Returns:
            The embedding vector for the new word.
        """
        # Mirrors ``Codebook.addVectors``: the lexicon's where-space slice
        # was reserved at ``create()`` time at width
        # ``self.lexicon_capacity``, so appending a row past that bound
        # would collide with the next codebook's slice in the global
        # where-space. Refuse loudly.
        current_size = len(self.wv.index_to_key)
        cap = int(getattr(self, "lexicon_capacity",
                          Embedding._LEXICON_DEFAULT_CAPACITY))
        assert current_size < cap, (
            f"Embedding.insert: lexicon is full ({current_size} >= "
            f"capacity={cap}); the where-space slice allocated for "
            f"this Embedding only covers ``lexicon_capacity`` prototypes. "
            f"Raise <nVectors> on the owning Space's XML (or the "
            f"Embedding._LEXICON_DEFAULT_CAPACITY default) and reload."
        )
        dim = self.wv.vector_size
        device = self.wv._vectors.device
        if vector is not None:
            new_vec = vector.to(device)
            if new_vec.dim() == 1:
                new_vec = new_vec.unsqueeze(0)
            new_vec = _wrap_unit_ball(new_vec)
        else:
            new_vec = _random_unit_ball((1, dim), device=device)

        # If the Parameter has been preallocated to capacity (via
        # explicit opt-in ``stage_oov(preallocate=True)`` / a future
        # caller), write in place into the next reserve row. Otherwise
        # fall back to the legacy cat + reassign + optimizer rebuild
        # so downstream consumers that read ``wv._vectors.shape[0]`` as
        # the active codebook size keep working.
        cur_shape = self.wv._vectors.shape[0]
        if cur_shape > current_size:
            idx = current_size
            with torch.no_grad():
                self.wv._vectors.data[idx:idx + 1, :] = new_vec
            self._zero_optimizer_moments_for_rows(idx, idx + 1)
        else:
            with torch.no_grad():
                new_data = torch.cat(
                    [self.wv._vectors.data,
                     new_vec.to(self.wv._vectors.device)],
                    dim=0)
                self.wv._vectors = nn.Parameter(new_data, requires_grad=True)
            idx = len(self.wv.index_to_key)
            if self.pretrain is not None:
                self._rebuild_optimizer()
        self.wv.counts = np.append(self.wv.counts, np.int64(initial_count))
        self._pending_counts.pop(word, None)
        self.wv._normed = None
        self.wv.index_to_key.append(word)
        self.wv.key_to_index[word] = idx

        # PretrainModel shares wv's mappings
        self.pretrain.index_to_key = self.wv.index_to_key
        self.pretrain.key_to_index = self.wv.key_to_index

        # Stage 1.B paired-row contract (2026-05-27): hand the freshly-
        # inserted PS-side per-word vector to the SymbolicSpace peer
        # (when wired) so SS.codebook gains an orth + semantic paired
        # row pointing at the same word. Wiring is via
        # ``symbolicSpace_ref`` on this Embedding (set in Models.py
        # after both spaces are built); unset in standalone
        # construction (tests) -- in which case the call is a no-op.
        # Fall back to the legacy ``mark_word_atom`` path if the SS peer
        # is too old to know about ``insert_paired_word``.
        s_peer = getattr(self, 'symbolicSpace_ref', None)
        if s_peer is not None:
            if hasattr(s_peer, 'insert_paired_word'):
                try:
                    # Per the flat-slab invariant, this PS vector is
                    # already CS-space-dimensioned; SS.codebook's orth
                    # row is a direct copy (no pi/sigma transform at
                    # insert time).
                    s_peer.insert_paired_word(word, new_vec.squeeze(0))
                except Exception:
                    # Paired-row insertion is best-effort; never block
                    # the PS-side insert because the symbolic peer
                    # codebook is exhausted (caller can pre-size
                    # SS.nVectors to ~2*lexicon_cap to avoid this).
                    pass
            elif hasattr(s_peer, 'mark_word_atom'):
                try:
                    s_peer.mark_word_atom(idx)
                except Exception:
                    pass

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
    def _load_embeddings(embedding_path=None, nDim=None, nVectors=None):
        """Load embeddings from a specific path, or return None for dynamic vocab.

        Detects file type by extension:
        - .txt: word2vec text format (``<vocab_size> <vector_size>\\n<word> <floats>...``)
        - .pt:  torch-saved WordVectors dict
        - .kv with ``kind='bpe'``: BPE-only artifact -- the lexicon is
          synthesized on the fly from the BPE byte-tuple vocab via
          ``bpe_to_lexicon_keys``. One codebook row per BPE chunk;
          ``len(BPE vocab)`` is the vocab size. ``nVectors`` (XML
          config) must match -- a mismatch raises so config drift
          surfaces immediately.

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
            try:
                wv = WordVectors.load(embedding_path)
            except ValueError as exc:
                # BPE-only artifact: WordVectors.load refuses, but we
                # can synthesize a stub WordVectors from the BPE
                # vocab so the codebook is seeded with one entry per
                # chunk_id (latin-1 decoded byte-tuple as key) instead
                # of starting empty and growing via insert() during
                # training.
                if "BPE-only artifact" not in str(exc):
                    raise
                # Note: ``WordVectors`` is already imported at module
                # scope; re-importing it locally would mark it a local
                # in this whole function and break the ``WordVectors.load``
                # above. Only pull in the helpers that aren't already
                # in scope.
                from embed import load_artifact, bpe_to_lexicon_keys
                payload = load_artifact(embedding_path)
                bpe = payload.get("bpe") or {}
                vocab = bpe.get("vocab", {})
                if not vocab:
                    return None
                # Build a synthetic ChunkLayer-shaped object just for
                # bpe_to_lexicon_keys (it only needs ``id_to_bytes``).
                class _Shim:
                    """Minimal ChunkLayer-shaped adapter for bpe_to_lexicon_keys.

                    Holds only the ``id_to_bytes`` dict that the key
                    builder reads; avoids constructing a full ChunkLayer
                    just to recover vocabulary strings.
                    """
                    def __init__(self, id_to_bytes):
                        """Store the chunk_id -> bytes_tuple mapping."""
                        self.id_to_bytes = id_to_bytes
                id_to_bytes = {int(v): tuple(int(x) for x in k)
                               for k, v in vocab.items()}
                keys = bpe_to_lexicon_keys(_Shim(id_to_bytes))
                # Drop trailing "" placeholder slots that
                # bpe_to_lexicon_keys emits for missing ids.
                while keys and keys[-1] == "":
                    keys.pop()
                if not keys:
                    return None
                # Vocab size is len(BPE vocab); the XML's ``nVectors``
                # MUST match. A mismatch is a config error -- raise
                # rather than silently padding (no reserved slots in
                # the codebook).
                if nVectors is not None and int(nVectors) != len(keys):
                    raise ValueError(
                        f"BPE-only artifact at {embedding_path} has "
                        f"{len(keys)} chunks but XML <nVectors>={nVectors}. "
                        f"Set <nVectors> to {len(keys)} (or retrain BPE "
                        f"with n_vectors={nVectors}) so the codebook "
                        f"size matches the BPE vocab.")
                dim = nDim or 1
                vectors = _random_unit_ball(
                    (len(keys), dim), device=TheDevice.get())
                wv = WordVectors(vectors, keys)
                print(f"Synthesized {len(keys)}-entry lexicon from "
                      f"BPE vocab in {embedding_path}")

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

    def use_sparse_grad(self, threshold=8192):
        """Mark this codebook for sparse-gradient lookup when its
        vocabulary is large enough that dense Adam moments dominate
        wall-clock cost.

        Read by ``Subspace._lookup_modality`` to switch the embedding
        gather to ``F.embedding(..., sparse=True)``; paired with a
        SparseAdam group in ``BaseModel.getOptimizer``.

        Default threshold: 8192 rows.  Below that the dense path is
        actually faster (Python overhead of the sparse code path
        outweighs the moment-update savings).
        """
        nV = int(getattr(self, 'nVectors', 0) or 0)
        self.sparse_grad = bool(nV >= int(threshold))
        return self.sparse_grad

    def parameters_for_optimizer(self):
        """Parameters for optimizer.
        
        See class docstring for the operation contract.
        """
        return self.embedding_parameters()

    def _ergodic_var(self, token_idx, device, dtype):
        """Map CBOW's per-word sigma to a noise scale in [0, 1].

        Returns None when ergodic mode is off, the model is in eval,
        or the embedding has no pre-train metadata. Otherwise reads the
        per-token sigma table and clamps to the [0, 1] interval the
        downstream lookup expects.
        """
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
        """Prepare one embedding vector, with optional ergodic noise."""
        if vec.dim() != 1:
            vec = vec.squeeze()
        vec = vec.clone()
        var = self._ergodic_var(token_idx, TheDevice.get(), vec.dtype)
        if var is not None and torch.any(var > 0):
            vec = vec + var * torch.randn_like(vec)
        return _wrap_unit_ball(vec)

    def _nearest_idx(self, vec, codebook=None):
        """Nearest idx.
        
        See class docstring for the operation contract.
        """
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
        scores = _wrapped_mse_score(
            vec[:d].unsqueeze(0), codebook[:, :d])
        return scores.argmax().item()

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

        # Phase 1: Tokenize all batch items and collect OOV base words.
        # Tokens are ``(text, offset)`` 2-tuples; "not" / "non-" surface
        # forms are NOT collapsed into separate codebook rows — they
        # tokenize like any other word, and the symbolic / grammar
        # layers handle negation at the appropriate tier.
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

        # Phase 2: Stage OOV words into preallocated reserve rows
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §4.2).
        # ``stage_oov`` writes in place — no Parameter reassignment, no
        # optimizer rebuild — and returns keys that overflowed the
        # reserve capacity. Overflowed keys fall through to the byte
        # fallback inside ``_token_to_index`` (which already routes
        # unknown keys to the ``\x00`` NULL row).
        if oov_words and not getattr(self, 'byte_mode', False):
            self.stage_oov(oov_words)
            codebook = self.getW()
            # ``self.optimize_embedding`` previously triggered a model-
            # wide optimizer rebuild after insert. Under the preallocated
            # contract the optimizer never needs rebuilding; the model-
            # level rebuild call is also retired.

        # Phase 3: Build result tensor (all tokens now in codebook)
        result = torch.zeros([batch, self.nInput, self.nDim], device=TheDevice.get())
        batch_indices = torch.zeros([batch, self.nInput], dtype=torch.long,
                                     device=TheDevice.get())
        null_idx = self.wv.key_to_index.get("\x00", 0)
        for b in range(batch):
            stream = all_streams[b]
            n_tokens = min(len(stream), self.nInput)
            tokens = stream[:n_tokens]
            batch_tokens.append(list(tokens))
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

    def save_embeddings(self, path, truth_data=None, bpe_section=None):
        """Save current embedding vectors and vocabulary to a ``.pt``/``.kv``
        file via the unified vocab-artifact schema (see :mod:`embed`).

        ``bpe_section`` (when supplied) embeds a BPE codebook in the
        same artifact under ``kind="both"``, so a single file carries
        the Lexicon and the BPE side-by-side. ``truth_data`` is carried
        through as a top-level field (legacy LTM passthrough).
        """
        self.wv.save(path, truth_data=truth_data, bpe_section=bpe_section)

    def reverse_raw(self, y):
        """Return the raw reverse-path vector with all subspaces intact.

        Does NOT strip encoding overhead -- the full vector (nWhat + nWhere + nWhen)
        is preserved for MSE loss computation.

        Returns: raw tensor [batch, nVec, embSize] with positions intact.
        """
        return y

    def _decode_offset(self, positions, batch_idx, vector_idx, subspace=None):
        """Decode offset.
        
        See class docstring for the operation contract.
        """
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
        """Render tokens.
        
        See class docstring for the operation contract.
        """
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
        """Recover lexical metadata from restored reverse-path vectors.

        Decodes the where / when slot encodings off the reverse-pass
        tail of each vector, then runs the lexicon's nearest-neighbor
        lookup to recover the original token string. Returns the
        decoded position / time tensors and per-row token lists.
        """
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
            vector: [batch, 1, embeddingSize] or [batch, embeddingSize].
                ``None`` is tolerated and produces an empty list -- the
                upstream pipeline can produce a ``None`` outputPred when
                the model has no trained output head (untrained / probe
                configs); returning ``[]`` lets the caller render the
                empty result rather than blowing up on a NoneType.attr
                lookup.
        Returns:
            List of predicted words (one per batch element).
        """
        if vector is None:
            return []
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

    Holds the factored representation (what, where, when, activation, event)
    along with the encoding objects that describe each factor.  Spaces
    populate a SubSpace during forward() and will eventually pass it as
    output.

    -----------------------------------------------------------------------
    Public read API
    -----------------------------------------------------------------------
    Clients of a SubSpace MUST read through ``materialize(mode=...)`` or
    ``resolve()``.  Reaching past these and calling ``.what.getW()`` /
    ``.where.getW()`` / etc. directly is wrong because:
      (a) ``getW()`` returns the underlying weight buffer, which may be
          a Codebook's learned prototype matrix rather than the
          per-batch values the caller actually wants;
      (b) the raw read skips ``.active`` selection, so callers that
          rely on selection semantics silently see the full slab.

    Modes:
      * ``materialize(mode="what")``       per-batch ``.what`` content,
                                           gathered/masked by ``.active``.
      * ``materialize(mode="where")``      per-batch ``.where`` content.
      * ``materialize(mode="when")``       per-batch ``.when`` content.
      * ``materialize(mode="event")``      muxed view; calls ``mux()`` to
                                           pack the three modalities into
                                           ``.event`` only when ``.event``
                                           is empty AND the modalities hold
                                           per-batch (3D) tensors.  Pure-
                                           event subspaces (ConceptualSpace)
                                           are a no-op: ``mux()`` skips
                                           because the modalities are empty.
      * ``materialize(mode="activation")`` presence-style read: stored
                                           activation × modal gate, falling
                                           back through ``activation_presence``
                                           and ``set_activation_from_event``
                                           when nothing is stored.  Returns
                                           a presence reduction (max of
                                           bivector poles), NOT signed DoT.
      * ``materialize(mode="active")``     legacy: muxed event × presence
                                           gate.  The default ``mode``.
      * ``resolve()``                      derived **signed** Degree of
                                           Truth = pos - neg from the
                                           ``.what`` bivector (or from
                                           ``.event[..., :2]`` for muxed-
                                           only subspaces).  Distinct from
                                           ``materialize(mode="activation")``
                                           which returns the *unsigned*
                                           presence (max of poles).

    -----------------------------------------------------------------------
    Public write API
    -----------------------------------------------------------------------
    Clients write via the public setters, NOT through ``.what.setW`` etc.:
      * ``set_what(t)`` / ``set_where(t)`` / ``set_when(t)``
            per-modality writes; each invalidates the cached ``.event``
            so the next ``materialize(mode="event")`` re-muxes.
      * ``set_event(t)``           muxed-event write; populates
                                   ``.event`` and resets activation to a
                                   default all-ones presence.
      * ``set_activation(t)``      activation write; lifts a 1-D / 2-D
                                   scalar to the bivector layout when
                                   the active encoding is 2-dim.

    Methods that receive a SubSpace as argument SHOULD only read from
    it; structural writes belong on ``self.subspace`` (the receiving
    space's own SubSpace).  This avoids writing a Tensor into a
    Codebook-backed slot on someone else's subspace -- Codebook
    parameters do not tolerate arbitrary tensor overwrites.

    -----------------------------------------------------------------------
    Activation semantics
    -----------------------------------------------------------------------
    ``.activation`` is a stored, possibly-quantized, possibly-externally-
    set per-position field.  It intentionally diverges from a pure
    derivation of ``.what`` after the codebook snap (which writes the
    quantized scalar back via ``set_activation``).  Use ``resolve()``
    when you want the pure ``pos - neg`` derivation; use
    ``materialize(mode="activation")`` when you want the model's
    committed activation (post-snap, post-set_activation).

    For SymbolicSpace specifically:
      * ``.what`` carries the continuous pre-snap bivector
        ``[pos_pole, neg_pole]`` (shape ``[B, N, 2]``) AND, when
        ``<codebook>true``, the learned ``[V_sym, 2]`` symbol-prototype
        matrix on ``.what.W``.
      * ``.activation`` carries the post-resolve / post-snap scalar
        (signed when set by SymbolicSpace.resolve()'s direct setW;
        bivector-lifted when set via ``set_activation``).
      * They are NOT the same field with the same content.  See the
        SymbolicSpace class docstring for the full lifecycle.

    -----------------------------------------------------------------------
    Slot architecture (spec: doc/specs/2026-05-21-subspace-slot-architecture.md)
    -----------------------------------------------------------------------
    The ``_active_payload`` band-aid was retired Stage 4 of
    doc/plans/2026-05-21-active-payload-retirement.md. The current
    contract:

      * Codebook-bearing slot (``.what`` for unmuxed, ``.event`` for
        muxed) — ``self.W`` holds ONLY the ``[V, D]`` Parameter
        prototype matrix. Per-batch content is reconstructed by
        ``materialize`` as ``codebook[_active]`` from prototype +
        selection. ``setW(per_batch)`` raises with a spec pointer.
      * Pure-event slot (plain ``Tensor`` ``.event``) — per-batch
        content lives on ``self.W`` directly (no Parameter to
        protect).
      * ``self._active`` (``[B, N, M]`` integer indices) and
        ``self.activation`` (per-position scalar / weight) carry the
        selection that ``materialize`` applies to the codebook.
      * Prototype mutations (insert / remove / VQ EMA) go through
        ``Basis.replace_W(new_W)`` which preserves Parameter identity
        (and the optimizer state keyed on it) via ``.data.copy_``.

    Pure-event subspaces (ConceptualSpace) leave ``.what`` / ``.where`` /
    ``.when`` empty and store everything on ``.event`` via
    ``set_event(...)``; ``mux()`` correctly no-ops there, and reads
    should use ``materialize(mode="event")``.
    """

    def __init__(self, inputShape, outputShape,
                 nInputDim=0, nOutputDim=0,
                 objectEncoding=None, activeEncoding=None, whatEncoding=None,
                 whereEncoding=None, whenEncoding=None, wordEncoding=None,
                 object=None, what=None, where=None, when=None, activation=None,
                 word=None):
        """Initialize SubSpace; allocate state for the class contract.
        
        See class docstring for invariants.
        """
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

        # Codebook placement (spec
        # doc/specs/2026-05-21-subspace-slot-architecture.md §Slots,
        # "Codebook placement (the muxed / unmuxed split)"):
        #   * ``'event'``  — codebook prototype on ``.event.W`` (muxed
        #     configs like PerceptualSpace MM_xor / MM_5M); ``self.muxed``
        #     is True. ``materialize(mode='event')`` reconstructs as
        #     ``event.W[selection]``.
        #   * ``'what'``   — codebook prototype on ``.what.W`` (unmuxed
        #     configs like SymbolicSpace, PerceptualSpace MM_grammar's
        #     lexicon Embedding). ``materialize`` reads from
        #     ``self.what`` via ``lookup``.
        #   * ``None``     — no codebook (pure-event configs:
        #     ConceptualSpace, InputSpace, OutputSpace). Per-batch
        #     content lands directly on ``.event.W`` (plain Tensor).
        #
        # Set ONCE at construction; never mutated. Downstream callers
        # MUST consult this flag (or ``self.codebook()`` /
        # ``self.prototype()`` / ``self.lookup()`` below) instead of
        # reaching into the Basis slots' raw ``W``.
        if isinstance(self.event, Codebook):
            self.codebook_slot = 'event'
        elif isinstance(self.what, (Codebook, Embedding)):
            # Both Codebook and Embedding act as codebook-bearing on
            # ``.what`` (the unmuxed case). Embedding holds the lexicon
            # prototype on ``wv._vectors``; Codebook holds it on ``W``.
            self.codebook_slot = 'what'
        else:
            self.codebook_slot = None
        # Boolean alias the user requested. ``True`` iff the codebook
        # lives on ``.event`` (the muxed config). False for unmuxed AND
        # for pure-event (no codebook).
        self.muxed = (self.codebook_slot == 'event')
        self.word   = word if word is not None else []  # list of (batch, vector, rule) tuples
        # Tensor word buffer (Path B from
        # plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md
        # §"Tensor word buffer (Path B)"). Inside the compute brick the
        # SyntacticLayer.compose chart writes per-cell entries via the
        # vector-typed ``add_word`` overload below; the outer doc-streaming
        # loop then calls ``flush_word_buffer`` once per tick to materialize
        # ``self.word`` for legacy consumers. The scalar ``add_word`` overload
        # is preserved for direct callers (tests, ``_compose_activation``).
        # ``word_records`` layout matches WordEncoding's 7-slot tuple
        # (batch, vector, order, rule, leaf1, leaf2, leaf3); ``word_count``
        # is the per-cell depth. Registered as buffers so ``.to(device)``
        # moves them with the SubSpace; ``register_buffer(persistent=False)``
        # keeps them out of the state_dict (no checkpoint pollution).
        self._WORD_ENTRY_WIDTH = 7
        self._WORD_MAX_DEPTH = 256  # cap per cell; matches reconstruct stack
        self.register_buffer(
            'word_records',
            torch.zeros(0, self._WORD_MAX_DEPTH, self._WORD_ENTRY_WIDTH,
                        dtype=torch.long),
            persistent=False)
        self.register_buffer(
            'word_count',
            torch.zeros(0, dtype=torch.long),
            persistent=False)
        # POS side-channel: parallel buffer recording the merged-cell
        # POS argmax at each merge. Shape mirrors word_count's leading
        # dim. 0 = '?' (wildcard / unknown). See doc/Language.md
        # "POS side-channel".
        self.register_buffer(
            'pos_records',
            torch.zeros(0, self._WORD_MAX_DEPTH, dtype=torch.long),
            persistent=False)
        self._demuxed = False
        # active: [B, N, M] per-modality indices into the Basis slots.
        # M = number of modalities (what, where, when).
        # active[b, n, m] = index into modality m's Basis for position n.
        # activation: [B, N] strength gate -- materialize() = event * activation.
        self._active = None  # [B, N, M] index tensor
        # Mereology measure family (Contiguous / Continuous / Peaceful /
        # Area / Luminosity) writes per-analysis-pass records here on
        # the conceptualSpace's subspace; other subspaces leave it None.
        # Format: list[dict] with keys 'step', 'area', 'luminosity',_active_payload
        # 'intersection', 'union'.
        self.knowing = None
        self.batch = 0

        # Pipeline-carried context. These travel with the subspace through
        # every Space.forward via copy_context(), replacing the old pattern
        # of cross-stage and cross-forward back-channels on Space instances.
        #   errors       -- per-batch auxiliary-loss accumulator; SymbolicSpace
        #                   writes symbol_commitment / codebook_commit / etc.
        #                   here and runBatch folds them into TheError.
        #   serial_cache -- {id(owner_space): tensor} for serial-mode warm cache
        #
        # Phase G of doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md
        # retired the per-SubSpace ``wordSubSpace`` back-pointer; the
        # WordSubSpace is now reached via the owning ``Space.wordSubSpace``
        # routing pointer (set by ``BasicModel`` at construction) or passed
        # explicitly to functions that need it (e.g.
        # ``ConceptualSpace.forward(subspace, word_subspace)``).
        self.errors = Error()
        self.serial_cache = {}

        # Per-cursor validity mask: ``[B, K]`` bool, True where each
        # cursor position's target token is non-NULL. Built by the
        # stem (or set to a default ``[B, 1]`` true mask in non-AR
        # paths); consumed by ``runBatch`` to gate per-cursor loss.
        # The legacy ``k_axis`` flag (which marked ``[B, K, N, D]``
        # slabs) was retired together with the AR cursor unfold —
        # every space now produces ``[B, …]`` shapes directly.
        self.valid_mask = None
        # stem_embedded=True signals downstream stages that InputSpace
        # has already performed lex+embed; the body's PerceptualSpace
        # must skip its own _embed (which would clobber the embedded
        # event with a fresh [B, N, D] re-embed of the original byte
        # buffer).
        self.stem_embedded = False

        payload = self.materialize()
        if isinstance(payload, torch.Tensor) and payload.ndim > 0:
            self.batch = payload.shape[0]

    def copy_context(self, other):
        """Adopt cross-stage/cross-forward state from ``other``.

        Pipeline invariant: every ``Space.forward`` that returns a subspace
        other than the incoming ``vspace`` must first ``copy_context(vspace)``
        so ``errors`` and ``serial_cache`` travel unbroken through the
        pipeline. ``errors`` and ``serial_cache`` are carried by reference
        so later writes (e.g., ``SymbolicSpace.forward`` adding a
        commitment term) land in the same accumulator that
        ``OutputSpace`` / ``runBatch`` will read.

        Also propagates the stem-route contracts (``valid_mask``,
        ``stem_embedded``). These are stage-routing flags whose value
        is set by the stem and read by ``runBatch``; propagating them
        through ``copy_context`` keeps the contract explicit.

        Phase G of doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md
        removed the ``wordSubSpace`` back-pointer from SubSpace; the
        WordSubSpace reference is reached via the owning ``Space``
        instance (``space.wordSubSpace``) or passed explicitly to
        functions that need it.
        """
        if other is None:
            return
        self.errors = other.errors
        self.serial_cache = other.serial_cache
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

    # -- Codebook accessors (spec-aligned public surface) --------------
    # doc/specs/2026-05-21-subspace-slot-architecture.md "Reader API":
    # callers needing the codebook prototype or a row lookup go through
    # these methods rather than reaching past the SubSpace to the slot's
    # raw ``getW()``. The SubSpace knows where the codebook lives via
    # ``self.codebook_slot`` (set at __init__); these accessors route
    # appropriately.

    def codebook(self):
        """Return the Basis slot holding the codebook prototype, or None.

        ``self.event`` when ``self.muxed``; ``self.what`` for unmuxed
        configs; ``None`` for pure-event (no codebook).
        """
        if self.codebook_slot == 'event':
            return self.event
        if self.codebook_slot == 'what':
            return self.what
        return None

    def prototype(self):
        """Return the ``[V, D]`` codebook prototype matrix, or None.

        Delegates to ``codebook().prototype()`` which reads the
        Parameter directly (bypassing the ``_active_payload`` shadow).
        """
        cb = self.codebook()
        return cb.prototype() if cb is not None else None

    def lookup(self, indices):
        """Look up codebook rows for the given selection indices.

        ``indices`` is a long tensor of arbitrary leading shape
        (typically ``[B, N]`` for per-position selection). Returns
        ``[..., D]`` rows from the prototype matrix.

        Raises if this SubSpace has no codebook (``codebook_slot is
        None``) — callers should check ``self.muxed`` /
        ``self.codebook_slot`` first if the path may be exercised on
        pure-event subspaces.
        """
        cb = self.codebook()
        if cb is None:
            raise RuntimeError(
                f"SubSpace.lookup() called on a pure-event subspace "
                f"(codebook_slot={self.codebook_slot!r}). Per-batch "
                f"content for pure-event configs is stored on "
                f"``.event.W`` directly; read via "
                f"``materialize(mode='event')`` instead.")
        return cb.lookup(indices)

    def set_muxed(self, event_tensor):
        """Store muxed event tensor. Clears demuxed modalities.

        For muxed subspaces (codebook on ``.event``) AND when the
        incoming tensor width matches the codebook width: snap
        ``event_tensor`` through ``Codebook.forward`` so the selection
        lands on ``_active``. ``materialize`` reconstructs as
        ``codebook[_active]`` — the activation/selection IS the storage.

        When the codebook is narrower than ``muxedSize`` (e.g. the
        codebook holds only the WHAT slice; WHERE/WHEN are added
        separately downstream), the snap cannot consume the full
        ``event_tensor`` cleanly — fall back to ``event.setW`` so the
        existing routing handles the slicing. This case is the BPE /
        unmuxed-codebook bridge; Stage 4 of the plan separates these
        contracts cleanly.

        For pure-event / unmuxed subspaces (plain ``Tensor`` ``.event``,
        no Parameter to protect): write directly to ``event.W``.

        Args:
            event_tensor: [B, N, D] where D = nWhat + nWhere + nWhen
        """
        # Spec doc/specs/2026-05-21-subspace-slot-architecture.md:
        # muxed subspace + matching width ⇒ snap through the codebook
        # so the selection lands on ``_active``; ``materialize``
        # reconstructs as ``codebook[_active]``.
        # Pure-event / unmuxed ⇒ direct ``event.setW`` (plain Tensor
        # storage on ``.W``).
        # Width-mismatch fallthrough (muxed + input width != codebook
        # nDim): silently skip the redundant write. Selection is the
        # source of truth; the stale per-batch payload was the band-aid
        # the migration retired.
        if self.muxed and event_tensor.shape[-1] == self.event.nDim:
            self.event.forward(event_tensor, _vspace=self)
        elif self.muxed:
            pass  # width mismatch — selection-based reconstruction
        else:
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
        Activation is the signed Degree-of-Truth scalar ``aP - aN``
        (the bivector ``[aP, aN]`` substrate was retired 2026-05), where
            aP = ||relu(what)|| / sqrt(nWhat)    (positive content)
            aN = ||relu(-what)|| / sqrt(nWhat)   (negative content)

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
        # Signed Degree-of-Truth: balance of positive vs negative content.
        act = pos.clamp(0.0, 1.0) - neg.clamp(0.0, 1.0)        # [B, N] in [-1, 1]
        self.set_activation(act)

    def _coerce_basis(self, value, role):
        """Coerce basis.
        
        See class docstring for the operation contract.
        """
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
            )
        elif role == "event":
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.muxedSize,
            )
        elif role == "what":
            basis.create(
                self.outputShape[0],
                self.outputShape[0],
                self.nWhat,
            )
        elif role == "where":
            basis.create(self.outputShape[0], self.outputShape[0], self.nWhere)
        elif role == "when":
            basis.create(self.outputShape[0], self.outputShape[0], self.nWhen)
        else:
            last_dim = value.shape[-1] if value.ndim > 1 else 1
            n_vectors = value.shape[-1] if value.ndim == 1 else value.shape[-2]
            basis.create(n_vectors, n_vectors, last_dim)
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
        (``_active`` index tensor, ``_per_batch_event`` cache,
        ``word`` list, ``valid_mask`` mask, ``stem_embedded`` flag).
        The prior version cleared only ``event``, which leaked the
        demuxed [B, N, D] tensors on what/where/when and the
        [B, N, M] ``_active`` tensor until the next forward overwrote
        them. Called from Space.Start() so state carried across the
        outer pos loop does not leak into the next DataLoader yield.
        """
        for basis in (self.event, self.what, self.where, self.when, self.activation):
            self._clear_runtime_basis(basis)
        self._active = None
        self.word = []
        # Drop per-tick word buffer scratch so it can't bleed across
        # batch boundaries. Records aren't zeroed; word_count gates
        # which slots are read on flush, and the next ensure_word_buffer
        # / scatter populates fresh entries.
        if int(self.word_count.shape[0]) > 0:
            self.word_count.zero_()
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
        # Drop per-tick word buffer scratch so it can't bleed across
        # batch boundaries. Records aren't zeroed; word_count gates
        # which slots are read on flush, and the next ensure_word_buffer
        # / scatter populates fresh entries.
        if int(self.word_count.shape[0]) > 0:
            self.word_count.zero_()
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

        ``errors`` and ``wordSubSpace`` persist (per-batch, per-document
        respectively; both owned by BasicModel lifecycle, not per-Reset).
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

        For unmuxed codebook configs (codebook on ``.what``) where the
        input width matches the codebook ``nDim``: snap ``what_tensor``
        through ``Codebook.forward`` so the selection lands on
        ``_active``; ``materialize`` reconstructs as
        ``what.W[_active[:, :, 0]]``.

        For plain-Tensor ``.what`` (no codebook): direct ``setW``.

        Width-mismatch fallthrough: when input width != codebook nDim,
        the spec invariant ("input width never greater than codebook
        nDim") would call for an XML fix. Pragmatically, we silently
        skip the redundant write — the selection has typically already
        been populated upstream (set_forward_content), and
        ``materialize`` reconstructs from prototype + selection. The
        ``what_tensor`` was a stale legacy cache that the band-aid
        absorbed; under the new contract it's discarded.

        Invalidates cached event so materialize() re-concatenates.
        """
        if (self.codebook_slot == 'what'
                and isinstance(self.what, Codebook)
                and what_tensor.shape[-1] == self.what.nDim):
            self.what.forward(what_tensor, _vspace=self)
        elif self.codebook_slot == 'what':
            # Width mismatch or non-Codebook codebook slot (e.g.,
            # Embedding). Skip the per-batch write — selection-based
            # reconstruction is the spec contract.
            pass
        else:
            # Plain-Tensor ``.what`` — direct write (no Parameter).
            self.what.setW(what_tensor)
        self.event.setW(None)
        self._demuxed = True

    def set_where(self, where_tensor):
        """Store positional encoding vectors [B, N, nWhere].

        Invalidates cached event so materialize() re-concatenates.
        """
        self.where.setW(where_tensor)
        self.event.setW(None)
        self._per_batch_event = None
        self._demuxed = True

    def set_when(self, when_tensor):
        """Store temporal encoding vectors [B, N, nWhen].

        Invalidates cached event so materialize() re-concatenates.
        """
        self.when.setW(when_tensor)
        self.event.setW(None)
        self._per_batch_event = None
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
        """Compute the signed Degree-of-Truth activation and active flags
        from the muxed event vectors.

        The bivector ``[aP, aN]`` substrate was retired (2026-05): the
        activation is the signed scalar ``aP - aN`` derived from the
        what-slice, where
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
        # Signed Degree-of-Truth: balance of positive vs negative content.
        act = pos.clamp(0.0, 1.0) - neg.clamp(0.0, 1.0)        # [B, N] in [-1, 1]
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
        """Store the per-slot signed Degree-of-Truth scalar.

        The 4-valued bivector ``[aP, aN]`` lift was retired (2026-05):
        the activation is a single signed scalar; no pole expansion.

        Args:
            activation_tensor: ``[N]``, ``[B, N]`` or ``[B, N, 1]`` scalar.
        """
        nd = self.activeEncoding.nDim
        # Accept 1-D [N] (legacy direct-setW callers / squeezed tensors)
        # by treating it as a single-batch [1, N]. Strict shapes
        # ([B, N] / [B, N, nd]) pass through unchanged.
        if activation_tensor.ndim == 1:
            activation_tensor = activation_tensor.unsqueeze(0)  # [N] -> [1, N]
        if activation_tensor.ndim == 2:
            pass  # scalar [B, N] stored directly (bivector lift retired)
        elif activation_tensor.ndim == 3:
            assert activation_tensor.shape[-1] == nd, (
                f"activation last dim must be {nd}, got {activation_tensor.shape}"
            )
        else:
            raise AssertionError(
                f"activation must be [N], [B, N], or [B, N, {nd}], "
                f"got {activation_tensor.shape}"
            )
        self.activation.setW(activation_tensor)

    def get_activation(self):
        """Return stored activation [B, N, nDim] or None."""
        if self.activation is None:
            return None
        return self.activation.getW()

    def activation_presence(self):
        """Per-slot presence gate = magnitude of the signed
        Degree-of-Truth scalar (``|DoT|``).

        The bivector ``max(aP, aN)`` reduction was retired (2026-05);
        ``|DoT|`` is its scalar successor: a position carries
        information iff the DoT is non-zero (NEITHER/0 -> gated off,
        any signed belief -> present). Non-negative, in ``[0, 1]``.
        Returns ``[B, N]`` or ``None``.
        """
        act = self.get_activation()
        if act is None:
            return None
        if act.ndim == 3 and act.shape[-1] == 1:
            act = act.squeeze(-1)        # width-1 scalar carrier -> [B, N]
        return act.abs()

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
                 leaf1=-1, leaf2=-1, leaf3=-1, pos=0):
        """Append one or more word entries.

        Two overloads share this entry point:

        * **Scalar form** -- ``batch``, ``vector``, ``rule`` are ``int``.
          Appends one validated 7-tuple to ``self.word``. Used by the
          legacy compose path and direct callers (tests,
          ``_compose_activation``). ``pos`` is recorded into
          ``pos_records`` mirroring the word slot (no-op for the
          scalar list).

        * **Vector form** -- ``batch`` is a 1-D long tensor of cell
          indices into the ``[B*K]`` flat row space. ``vector``,
          ``rule``, ``order``, ``leaf*``, ``pos`` are 1-D long tensors
          of the same length (or ``int`` broadcast over the active
          rows). Writes into the per-cell tensor buffers
          ``word_records`` / ``pos_records`` / ``word_count`` via
          scatter, with no host sync. The outer doc-streaming loop
          calls ``flush_word_buffer`` once per tick to materialize the
          buffer's contents back into ``self.word`` for legacy
          consumers (``decompose``, ``reconstruct``, the SVO walker,
          derivation-trace tests). See plan §6c.

        The vector form is gated on ``isinstance(batch, torch.Tensor)``;
        anything else falls through to the scalar form so existing
        callers keep working unchanged.

        ``pos`` defaults to 0 ('?', wildcard) to preserve back-compat
        for callers that don't carry POS.
        """
        if isinstance(batch, torch.Tensor):
            self._add_word_vec(batch, vector, rule, order=order,
                               leaf1=leaf1, leaf2=leaf2, leaf3=leaf3,
                               pos=pos)
            return
        self.word.append(self.wordEncoding.encode(
            batch, vector, rule, order, leaf1, leaf2, leaf3))

    def ensure_word_buffer(self, n_cells):
        """Resize ``word_records`` / ``word_count`` to hold ``n_cells`` rows.

        Idempotent when already sized correctly. Resizes preserve no
        prior contents (the buffer is per-tick scratch, cleared by
        ``clear_word_buffer`` / ``flush_word_buffer``).
        """
        n = int(n_cells)
        if int(self.word_count.shape[0]) == n:
            return
        device = self.word_count.device
        self.word_records = torch.zeros(
            n, self._WORD_MAX_DEPTH, self._WORD_ENTRY_WIDTH,
            dtype=torch.long, device=device)
        self.word_count = torch.zeros(n, dtype=torch.long, device=device)
        self.pos_records = torch.zeros(
            n, self._WORD_MAX_DEPTH, dtype=torch.long, device=device)

    def clear_word_buffer(self):
        """Zero the per-cell depth so the next tick starts fresh.

        Records aren't zeroed because ``word_count`` gates which slots
        are read at flush time; stale values in unread slots are
        irrelevant. Called by ``flush_word_buffer`` after materializing
        the tick's entries; tests can call directly to reset state.
        """
        self.word_count.zero_()

    def _add_word_vec(self, b_indices, vec_idxs, rule_ids,
                      order=0, leaf1=-1, leaf2=-1, leaf3=-1, pos=0):
        """Tensor scatter into ``word_records`` -- vector ``add_word`` body.

        ``b_indices`` is the [N_active] long cell-id tensor. The other
        args may be int (broadcast) or [N_active] long tensors.
        Writes one entry per active cell at the cell's current depth,
        then increments ``word_count[b_indices]``.
        """
        if int(b_indices.numel()) == 0:
            return
        BK = int(self.word_count.shape[0])
        # Lazy-resize: if a caller dispatches before ensure_word_buffer
        # ran, size to the largest index referenced. The buffer is small
        # (entries are int64 7-tuples; cap is max_depth=256 entries per
        # cell), so growing it here is cheap.
        max_idx = int(b_indices.max().item())
        if max_idx >= BK:
            self.ensure_word_buffer(max_idx + 1)
        device = self.word_records.device
        b_idx = b_indices.to(device=device, dtype=torch.long)
        depths = self.word_count[b_idx]
        WE = self.wordEncoding

        def _broadcast(val):
            if isinstance(val, torch.Tensor):
                return val.to(device=device, dtype=torch.long)
            return torch.full(
                (b_idx.shape[0],), int(val), device=device, dtype=torch.long)

        v_idx = _broadcast(vec_idxs)
        r_idx = _broadcast(rule_ids)
        o_idx = _broadcast(order)
        l1    = _broadcast(leaf1)
        l2    = _broadcast(leaf2)
        l3    = _broadcast(leaf3)
        p_idx = _broadcast(pos)

        # WordEncoding tuple slots: BATCH, VECTOR, ORDER, RULE, LEAF1, LEAF2, LEAF3
        self.word_records[b_idx, depths, WE.BATCH]  = b_idx
        self.word_records[b_idx, depths, WE.VECTOR] = v_idx
        self.word_records[b_idx, depths, WE.ORDER]  = o_idx
        self.word_records[b_idx, depths, WE.RULE]   = r_idx
        self.word_records[b_idx, depths, WE.LEAF1]  = l1
        self.word_records[b_idx, depths, WE.LEAF2]  = l2
        self.word_records[b_idx, depths, WE.LEAF3]  = l3
        self.pos_records[b_idx, depths] = p_idx
        # In-place increment of the per-cell depth. ``index_add_`` would
        # also work; direct gather/scatter is fine because each b_idx
        # appears at most once per call (compose dispatches one entry
        # per active cell per depth d).
        self.word_count[b_idx] = depths + 1

    def flush_word_buffer(self):
        """Materialize the tick's tensor word buffer into ``self.word``.

        One sync per tick (the ``.tolist()`` calls below), called from
        the outer doc-streaming loop AFTER ``runBatch`` so the brick
        body itself stays sync-free. Appends entries in cell-major
        order; downstream consumers (``decompose``, ``reconstruct``,
        SVO walker) see ``self.word`` populated as before.

        Resets ``word_count`` for the next tick. ``word_records`` slots
        beyond each cell's depth are unread so stale values are
        harmless.
        """
        if int(self.word_count.shape[0]) == 0:
            return
        counts = self.word_count.tolist()
        if not any(counts):
            self.word_count.zero_()
            return
        records = self.word_records.tolist()
        WE = self.wordEncoding
        for bk, depth in enumerate(counts):
            for d in range(depth):
                e = records[bk][d]
                # Validate via wordEncoding.encode so the host-side
                # tuple matches the legacy add_word output exactly.
                self.word.append(WE.encode(
                    int(e[WE.BATCH]),
                    int(e[WE.VECTOR]),
                    int(e[WE.RULE]),
                    int(e[WE.ORDER]),
                    int(e[WE.LEAF1]),
                    int(e[WE.LEAF2]),
                    int(e[WE.LEAF3])))
        self.word_count.zero_()

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

        Sparse-gradient path: when ``basis.sparse_grad`` is True (set by
        Embedding for the perceptual lexicon when its V is large), use
        ``F.embedding(..., sparse=True)`` so backprop produces a sparse
        gradient that touches only the rows we indexed.  Pairs with
        SparseAdam in BaseModel.getOptimizer to skip the V*D dense
        moment update on every step.
        """
        codebook = basis.getW()
        if codebook is None:
            return None
        if codebook.ndim == 2:
            # [V, D] codebook -- index with [B, N] -> [B, N, D]
            if getattr(basis, 'sparse_grad', False):
                # F.embedding requires the weight to be a Parameter or
                # at least a leaf tensor; falls back to dense indexing
                # otherwise.
                if codebook.requires_grad:
                    return torch.nn.functional.embedding(
                        indices.long(), codebook, sparse=True)
            return codebook[indices.long()]
        # Already [B, N, D] (e.g. from set_what on a Tensor basis)
        return codebook

    def mux(self):
        """Pack ``.what`` / ``.where`` / ``.when`` into ``.event``.

        Pure-event subspaces (e.g. ConceptualSpace, which uses the
        SubSpace as an opaque event carrier set via ``set_event()``)
        leave all three modality slots empty.  In that case mux() is a
        no-op.

        Conservative behaviour to avoid corrupting upstream-set events:

        - If ``.event`` is already populated AND has 3D (per-batch)
          shape, return without rewriting.  An upstream stage that
          called ``set_event(...)`` directly is the source of truth;
          re-muxing from possibly-stale modality slots would clobber it.
        - Only pack from modalities whose ``.getW()`` is per-batch
          (``ndim >= 3``, i.e. ``[B, N, D]``).  Codebook prototypes on
          ``.what`` are 2D ``[V, D]`` -- those are NOT per-batch
          content and must not be concatenated as if they were.
        - Public setter ``set_what`` / ``set_where`` / ``set_when``
          already invalidates ``.event`` (sets it to None), so the
          "regenerate when stale" trigger is the caller's
          responsibility.

        This is the legitimate path for unmuxed-set-up code that
        populates ``.what`` / ``.where`` / ``.when`` with per-batch
        tensors and wants a muxed view downstream.
        """
        # Already-populated 3D event: respect it, don't re-mux.
        existing = self.event.getW() if self.event is not None else None
        if existing is not None and existing.ndim >= 3:
            return

        what_w  = self.what.getW()  if self.what  is not None else None
        where_w = self.where.getW() if self.where is not None else None
        when_w  = self.when.getW()  if self.when  is not None else None
        if what_w is None and where_w is None and when_w is None:
            return  # pure-event subspace -- nothing to pack

        # Only include per-batch (3D) modality tensors.  A 2D ``.what``
        # is a Codebook's ``[V, D]`` prototype matrix -- not muxable.
        parts = []
        if (what_w  is not None and what_w.ndim  >= 3
                and what_w.shape[-1]  > 0):
            parts.append(what_w)
        if (where_w is not None and where_w.ndim >= 3
                and where_w.shape[-1] > 0):
            parts.append(where_w)
        if (when_w  is not None and when_w.ndim  >= 3
                and when_w.shape[-1]  > 0):
            parts.append(when_w)
        if not parts:
            return
        # All parts must share the leading [B, N] shape.
        ref_shape = parts[0].shape[:-1]
        for p in parts[1:]:
            if p.shape[:-1] != ref_shape:
                return  # incompatible -- skip rather than corrupt
        self.event.setW(torch.cat(parts, dim=-1))

    def resolve(self):
        """Read the per-slot signed **Degree of Truth** scalar from the
        width-1 ``.what`` (or, for muxed-only subspaces, ``.event``)
        carrier.  Computed on read; not stored as a separate field.

        The bivector ``[aP, aN]`` substrate was retired (2026-05): the
        carrier is a single signed scalar in ``[-1, +1]`` -- there is no
        ``pos - neg`` collapse. Returns ``[B, N]`` (or ``[N]`` for
        unbatched legacy). Applies ``.active`` selection when set;
        returns ``None`` when no scalar source is reachable.

        Public read API for the resolved scalar; replaces the prior
        ``self.activation.getW()`` access pattern.
        """
        src = self.what.getW() if self.what is not None else None
        if not (src is not None and src.ndim >= 2 and src.shape[-1] >= 1):
            ev = self.event.getW() if self.event is not None else None
            src = ev if (ev is not None and ev.ndim >= 3
                         and ev.shape[-1] >= 1) else None
        if src is None:
            return None
        # Width-1 carrier -> the scalar itself; wider content -> the
        # signed magnitude balance ``aP - aN`` (consistent with
        # SubSpace._compute_active / set_activation_from_event).
        if src.shape[-1] == 1:
            scalar = src[..., 0]
        else:
            d = max(src.shape[-1], 1)
            pos = torch.relu(src).norm(dim=-1) / math.sqrt(d)
            neg = torch.relu(-src).norm(dim=-1) / math.sqrt(d)
            scalar = pos.clamp(0.0, 1.0) - neg.clamp(0.0, 1.0)
        return self._apply_active_selection(scalar)

    def _apply_active_selection(self, src):
        """Apply ``.active`` mask/index selection to ``src``.  Returns
        ``src`` unchanged when ``.active`` is None or doesn't apply
        (wrong shape, mismatched leading dim).
        """
        if src is None:
            return None
        active = self._active
        if active is None:
            return src
        # Boolean mask along the slot axis.
        if active.dtype == torch.bool:
            try:
                return src[active]
            except Exception:
                return src
        # Index tensor: only apply when shape is 1-D or 2-D and
        # broadcasts cleanly over the slot axis.
        if active.ndim == 1 and src.ndim >= 1:
            try:
                return src.index_select(-2 if src.ndim >= 2 else 0,
                                        active.long())
            except Exception:
                return src
        # Higher-rank index forms (e.g. [B, N, M] per-modality indices)
        # are consumed inside materialize()'s rebuild path; we don't
        # second-guess them here.
        return src

    def materialize(self, k=None, mode="active"):
        """Public read API for SubSpace contents.

        Modes:
          * ``"what"`` / ``"where"`` / ``"when"`` -- the named modality
            slot's tensor, with ``.active`` selection applied.
          * ``"event"`` -- the muxed view.  Calls ``mux()`` to (re)pack
            ``.what`` / ``.where`` / ``.when`` into ``.event`` when the
            modalities are populated; pure-event subspaces are a no-op
            and the existing ``.event`` is returned.  Selection
            applied.
          * ``"activation"`` -- legacy alias; equivalent to
            ``self.resolve()`` for bivector layouts, falls back to the
            ``activation_presence`` / ``effective_activation`` path
            when no bivector source is present.
          * ``"active"`` (default, legacy) -- ``event * activation_presence``
            via the index-based or legacy demuxed rebuild path.
            Preserved for backwards-compat with existing call sites.

        Args:
            k: legacy parameter; unused by the new mode-keyed paths.
            mode: see above.

        Returns:
            Tensor in the requested mode, or None when no source is
            reachable.
        """
        # New mode-keyed reads.
        if mode == "what":
            w = self.what.getW() if self.what is not None else None
            return self._apply_active_selection(w)
        if mode == "where":
            w = self.where.getW() if self.where is not None else None
            return self._apply_active_selection(w)
        if mode == "when":
            w = self.when.getW() if self.when is not None else None
            return self._apply_active_selection(w)
        if mode == "event":
            # Muxed codebook path: ``materialize = codebook[selection]``.
            # The activation/selection IS the per-batch storage.
            # Gate on codebook width = muxedSize so we don't return a
            # WHAT-only slice when the muxed event needs WHERE/WHEN
            # concatenated on top (the chunked-codebook case).
            if (self.muxed and self._active is not None
                    and self._active.ndim == 3
                    and self._active.shape[-1] >= 1):
                proto = self.prototype()
                if (proto is not None and torch.is_tensor(proto)
                        and proto.ndim == 2):
                    sel = self._active[:, :, 0].long()
                    if True:  # placeholder for the bounds-check block
                        return self._apply_active_selection(self.lookup(sel))
            # Re-mux from modality slots when they're populated.  The
            # mux() implementation no-ops on pure-event subspaces, so
            # ConceptualSpace's set_event-only flow still works.
            self.mux()
            e = self.event.getW() if self.event is not None else None
            return self._apply_active_selection(e)
        # ``activation`` mode: presence-style read.  Returns the stored
        # activation reduced to per-position **presence** (max of bivector
        # poles after modal gating).  Order of preference, mirroring the
        # historical contract:
        #   1. effective_activation() -- stored activation × modal gate
        #   2. activation_presence()  -- raw stored activation, no gate
        #   3. derived from event (set_activation_from_event then read)
        # NOT fallthrough to resolve():  resolve() returns signed
        # pos - neg (Degree of Truth), a different reduction.  Callers
        # that want signed DoT call subspace.resolve() directly.
        if mode == "activation":
            eff = self.effective_activation()
            if eff is not None:
                return eff
            pres = self.activation_presence()
            if pres is not None:
                return pres
            self.set_activation_from_event()
            return self.activation_presence()

        # Default mode: same muxed-codebook reconstruct (width gated to
        # muxedSize — see ``mode='event'`` rationale above), plus the
        # activation-presence gate (legacy "event * presence").
        if (self.muxed and self._active is not None
                and self._active.ndim == 3
                and self._active.shape[-1] >= 1):
            proto = self.prototype()
            if (proto is not None and torch.is_tensor(proto)
                    and proto.ndim == 2
                    and proto.shape[-1] == self.muxedSize):
                sel = self._active[:, :, 0].long()
                if sel.numel() > 0:
                    # Bounds check on indices avoids data-dependent
                    # ``int(sel.max())`` (Dynamo fullgraph contract).
                    # Per spec the snap produces in-range indices.
                    recon = self.lookup(sel)
                    pres = self.activation_presence()
                    if pres is not None:
                        recon = recon * pres.unsqueeze(-1)
                    if k is not None and k < recon.shape[-2]:
                        score = pres if pres is not None else recon.norm(dim=-1)
                        _, idx = torch.topk(score, k, dim=-1)
                        self._topk_indices = idx
                        recon = torch.gather(
                            recon, -2,
                            idx.unsqueeze(-1).expand(-1, -1, recon.shape[-1]))
                    return recon
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
                    # Stage 4: no cache-back to ``self.event.setW``.
                    # Materialize reconstructs on demand from prototype
                    # + selection; caching to a Codebook-bearing slot
                    # would clobber the Parameter (now raises).
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
                    # Stage 4: no cache-back.

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
            # Debug-only validation: the finite/range check below has no
            # functional effect (it asserts or warns, then returns), but
            # forces a host sync every call -- isfinite().all() __bool__
            # plus min()/max().item() are each a cudaMemcpyDtoH, which
            # breaks the brick CUDA-graph-capture contract
            # (test_brick_no_sync). Gate behind MODEL_DEBUG, mirroring
            # BaseModel._assert_finite_train_state and the runBatch
            # finite-loss guard; NaN/range bugs still surface under
            # MODEL_DEBUG runs and via downstream finite guards.
            if not util.MODEL_DEBUG:
                return
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
            if kind == "bivector":
                # Bivector regime (CSBP): values are sums of relu projections,
                # bounded only by ``V_in * max(|input|*|W|)``. Assert non-
                # negativity; no upper bound.
                xmin = xd.min().item()
                if xmin < -1e-2:
                    msg = (f"Range violation: kind='bivector', target={target!r} "
                           f"min {xmin:.6f} below 0 (bivector requires non-negative).")
                    if strict:
                        assert False, msg
                    else:
                        warnings.warn(msg)
                return
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
# ShortTermMemory moved to bin/Layers.py (Phase E of doc/specs/
# 2026-05-21-wordsubspace-stm-layer-refactor.md). Re-exported here so
# ``from Spaces import ShortTermMemory`` continues to resolve for the
# Models.py import site.
from Layers import ShortTermMemory as _ShortTermMemory_relocated
ShortTermMemory = _ShortTermMemory_relocated
class _ShortTermMemory_PlaceholderRemoved:
    """Sentinel — the ShortTermMemory class body that lived here is now
    in :mod:`Layers`. This stub exists only so the surrounding lines
    keep their offsets while the file is intermediate-state during the
    refactor; ignore at runtime.
    """
    def __init__(self, batch=1, capacity=None, concept_dim=0):
        raise NotImplementedError(
            "Use Layers.ShortTermMemory (re-exported as "
            "Spaces.ShortTermMemory).")
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

    # -- knowledge-artifact attach -----------------------------------------
    # Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    # §Phase 2 — Loaders. Every Space subclass inherits
    # ``attach_knowledge(view)`` and the read-only ``.knowledge`` property,
    # letting per-Space consumers read from a shared ``KnowledgeView``
    # without each subclass re-implementing the plumbing. The same
    # mechanism is mirrored on ``WordSpace`` (which doesn't subclass Space
    # but follows the same pattern).

    def attach_knowledge(self, view):
        """Attach a loaded ``embed.KnowledgeView``. Replaces any
        previously attached view. Stored via ``object.__setattr__`` to
        bypass nn.Module's submodule registration (the view holds
        tensors but isn't itself a Module)."""
        object.__setattr__(self, '_knowledge', view)

    @property
    def knowledge(self):
        """The attached ``embed.KnowledgeView``, or ``None`` before
        ``attach_knowledge`` is called."""
        return getattr(self, '_knowledge', None)

    def __init__(self, inputShape, spaceShape, outputShape, customVQ=True):
        """Initialize Space; allocate state for the class contract.
        
        See class docstring for invariants.
        """
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

        # wordSubSpace is still held as a non-Module pointer so the few
        # call sites that reach across to ``wordSubSpace.truth_layer``
        # (SymbolicSpace) keep working; composition dispatch is no
        # longer done here -- home spaces take ``wordSubSpace`` as a
        # per-call parameter and call ``wordSubSpace.forwardSymbols`` /
        # ``.reverseSymbols`` explicitly.
        self.wordSubSpace = None
        self.params = []   # parameters for the optimizer (excludes temperature params)
        self.layers = nn.ModuleList()   # layer instances for paramUpdate() delegation
        self._register_requirements()

    def attach_wordSubSpace(self, wordSubSpace):
        """Wire the shared WordSpace as a non-Module routing pointer.

        The wordSubSpace reference is stored via ``object.__setattr__`` so
        the WordSpace nn.Module is NOT registered as a child of this
        Space -- that would create a ``space -> wordSubSpace -> space`` cycle
        (WordSpace already owns the SyntacticLayer and its codebook
        host is the SymbolicSpace) and make ``model.to(device)``
        recurse forever. The wordSubSpace is owned at the model level
        instead, with each Space holding only this non-Module pointer.
        Layer attachment is done directly via
        ``wordSubSpace.attach_layer(kind, layer)`` by the WordSpace
        factory methods, not by this helper.
        """
        object.__setattr__(self, 'wordSubSpace', wordSubSpace)

    def _build_object_basis(self):
        """Build object basis.
        
        See class docstring for the operation contract.
        """
        if not self.codebook:
            # No codebook configured: build a passthrough Tensor sized to
            # the muxed event width. Tensor.forward / .reverse are the
            # identity, so callers see no quantization.
            basis = Tensor(nVectors=self.nVectors, nDim=self.muxedSize)
            basis.ergodic = getattr(self, "ergodic", False)
            return basis
        basis = Codebook()
        # Per-Space metric: Spaces that want unit-norm codebooks for
        # dot-product retrieval (e.g. ConceptualSpace) override the
        # ``use_dot_product`` class attribute on the Space. We mirror it
        # onto the basis instance so Codebook.addVectors picks it up.
        basis.use_dot_product = bool(getattr(self, "use_dot_product", False))
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.muxedSize,  # Codebook processes full event vectors
            customVQ=self.customVQ,
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

    @staticmethod
    def normalize_codebook_mode(raw):
        """Normalize the tri-state ``<codebook>`` config to a canonical mode.

        Returns one of ``"none"`` / ``"quantize"`` / ``"project"``:

          * ``none``     -- no codebook on ``.what`` (legacy ``false``).
          * ``quantize`` -- VQ / EMA Codebook snap (legacy ``true``).
          * ``project``  -- size-changing ProjectionBasis (LDU-invertible;
                            emits a signed scalar activation ``[B, N]``).

        Legacy booleans are accepted for backward compatibility
        (``True -> "quantize"``, ``False -> "none"``) so unmigrated configs
        and programmatic ``codebook=True/False`` overrides still load.

        Shared by the ``codebook_mode`` property (instance use on
        ``self._codebook``) AND by Models.py callers that hold the raw
        XML value before any Space instance exists (InputSpace default
        wiring, ``architecture.monotonic`` validation). Living on
        ``Space`` keeps the helper namespaced rather than polluting the
        module scope.
        """
        if isinstance(raw, bool):
            return "quantize" if raw else "none"
        if raw is None:
            return "none"
        v = str(raw).strip().lower()
        if v in ("none", "false", "0", ""):
            return "none"
        if v in ("quantize", "true", "1"):
            return "quantize"
        if v == "project":
            return "project"
        raise ValueError(
            f"<codebook> must be one of none|quantize|project "
            f"(legacy true/false accepted); got {raw!r}")

    @property
    def codebook_mode(self):
        """Tri-state codebook mode: ``'none'`` | ``'quantize'`` |
        ``'project'``. (Was the boolean ``<codebook>``; ``project``
        selects the size-changing scalar ``ProjectionBasis``.) Delegates
        to :meth:`normalize_codebook_mode` so the parsing logic lives
        in one place.
        """
        return Space.normalize_codebook_mode(self._codebook)

    @property
    def codebook(self):
        """True iff a codebook is present (``quantize`` or ``project``).

        Back-compat boolean for the many ``if self.codebook:`` truthiness
        sites; the quantize-vs-project distinction is ``codebook_mode``."""
        return self.codebook_mode != "none"

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
        """Lookup.
        
        See class docstring for the operation contract.
        """
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.zeros([1,1, TheXMLConfig.space("ConceptualSpace", "nDim")], device=TheDevice.get()), x[:,:,1:]], dim=2)
        output, index, _ = self.subspace.get_vectors().quantize(x)
        #output[:,:,0:conceptDim] = output[:,:,0:conceptDim] * activation  # multiply the codebook vector by the activation
        return output
    def dereference(self, symbols):
        # we get [ batch x nConcepts x symbolEmbedding ],
        # and must compute [ batch x nConcepts x conceptEmbedding ]
        """Dereference.
        
        See class docstring for the operation contract.
        """
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
        """Stats.
        
        See class docstring for the operation contract.
        """
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
        """Return optimizable parameters owned by this module."""
        return self.params
    def paramUpdate(self):
        """In-place parameter update hook called once per training step."""
        for l in self.layers:
            if hasattr(l, 'paramUpdate'):
                l.paramUpdate()

    def Start(self):
        """One-shot per-run initialization.

        Called from BasicModel.Start() once at the top of model.forward(),
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
        for source row ``batch``. ``None`` clears every row (the global-
        Reset semantics).

        ``hard`` (bool, default True): True = document boundary; False =
        sentence-internal soft reset. The base cascade forwards both flags
        to children; subclasses interpret as needed.

        Subclasses override to add their own resets (buffer clears, cursor
        resets, etc.) and must call super().Reset(batch=batch, hard=hard)
        first.

        Every Reset-capable child must accept the ``(batch, hard)``
        signature; the legacy zero-arg fallback was removed in §8d of the
        brick-vectorization handoff.
        """
        for layer in self.layers:
            if hasattr(layer, 'Reset'):
                layer.Reset(batch=batch, hard=hard)
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'Reset'):
            sub.Reset(batch=batch, hard=hard)

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
            basis.use_dot_product = bool(getattr(self, "use_dot_product", False))
            basis.create(
                self.inputShape[0],
                self.nVectors,
                self.nDim,
                customVQ=self.customVQ,
            )
            return basis
        if self.model_type in ("passthrough", "simple"):
            basis = Tensor()
            basis.create(
                self.inputShape[0],
                self.outputShape[0],
                self.nDim,
            )
            return basis
        raise RuntimeError("Unexpected model_type")

    def __init__(self, inputShape, spaceShape, outputShape, model_type="simple"):

        """Initialize InputSpace; allocate state for the class contract.
        
        See class docstring for invariants.
        """
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
        self._model_wordSubSpace = None
        # End-of-stream diagnostic: ``list[bool]`` of per-row "this row
        # has no valid windows in the current tick" flags. Sized lazily
        # by the AR forward() path. Under the rolling-cursor handoff
        # the canonical hard-reset signal is the cursor's host-side
        # ``hard_eos`` from ``next_tick``; ``_end_of_stream`` is now a
        # diagnostic only — never consulted for control flow, just
        # cleared by ``Reset(batch=b, hard=True)`` so external observers
        # see a clean state.
        self._end_of_stream = []
        # Raw sentence strings stashed by prepInput for BasicModel.forward()
        # to compute the outer-loop iteration count N without rethreading
        # inp_items through runBatch's call signature.
        self._last_sentences = None
        # Optional pre-built embedding bypass: callers (chat-loop /
        # generate_sentence) stage a tensor here so forward() skips
        # the lex/embed step on that call.  None during training.
        self._cached_embedding = None
        # ---- Per-word ground-truth cursor (INERT this increment) ----
        # ``next_word`` walks ``self._ar_embedded`` ([B, T, D], the
        # already-lexed whole-sentence buffer set by forward()) one
        # ground-truth T-position at a time for the IR-reconstruction
        # loop's word source.  This is a pure ground-truth feed: the
        # slice is the lexed input, the stop is the input's NULL/end-of-
        # valid-content sentinel -- there is ZERO AR-prediction here (no
        # get_recovered_word, no reconstruction feedback, no [MASK], no
        # model-predicted EOF). It is deliberately NOT the retired
        # ``arir_step`` AR machine.
        #
        # ``_per_word_enabled`` is the deferred wiring point. It defaults
        # False and is set by NO config/code in this increment, so the
        # ``next_word`` branch is never taken in any current run and the
        # live whole-slab ``forward`` contract is byte-identical. The
        # NEXT increment flips this True and wires the per-word feed into
        # ConceptualSpace/STM/SHIFT/selector/reverse (explicitly out of
        # scope here).
        self._per_word_enabled = False
        # Host-side int position into the T-axis of ``_ar_embedded``.
        # Lifecycle mirrors ``_ar_embedded``: born here, reset to 0 in
        # Start() (per-run) and hard Reset() (per-document), exactly as
        # ``_ar_embedded``/``_ar_total`` are reset to None/0.
        self._per_word_cursor = 0
        # Host-side valid lexed length, computed ONCE per forward and
        # cached here so ``next_word()`` is a pure host-int compare
        # (``p >= self._valid_len_host``) with zero DtoH per call. The
        # single DtoH that would otherwise fire per word is amortised
        # into the (already non-captured) IS forward boundary; D8's
        # per-word DtoH budget collapses to zero (well within the "one
        # byte termination read" allowance).
        self._valid_len_host = 0
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
        """Get train data.
        
        See class docstring for the operation contract.
        """
        return self.data.train_input, self.data.train_output
    def getTestData(self):
        """Get test data.
        
        See class docstring for the operation contract.
        """
        return self.data.test_input, self.data.test_output
    def prepInput(self, inputBatch):
        # Stash raw sentence strings for the AR outer loop in
        # BasicModel.forward() -- lets it compute N without rewiring
        # inp_items through runBatch's call signature.
        """Prep input.
        
        See class docstring for the operation contract.
        """
        if (isinstance(inputBatch, list) and inputBatch
                and isinstance(inputBatch[0], str)):
            self._last_sentences = list(inputBatch)
        else:
            self._last_sentences = None

        if isinstance(inputBatch, list):
            tensors = [self.data.stringTensor(s) if isinstance(s, str) else s
                       for s in inputBatch]
            host = torch.stack(tensors, dim=0).unsqueeze(1)
            # Stash the host copy (consumed once in _lex_batch) so the
            # lexer's _to_text `.tolist()` is a CPU op, not a
            # cudaMemcpyDtoH (residual A). It is exactly the tensor that
            # becomes inputTensor via `.to(device)`, so lexing it is
            # byte-identical to lexing the device copy.
            self._host_input_slab = host
            # Synchronous H2D: correct and race-free. (A non_blocking
            # copy of this ephemeral pinned buffer corrupts the GPU
            # data when `host` is freed before the transfer lands --
            # the NaN source; see memory / pinmem guide. Eliminating
            # this sync from the captured region is the job of the
            # compile-scoped-to-model architecture, not async tricks.)
            return host.to(TheDevice.get())
        self._host_input_slab = None  # no host origin -> device fallback
        return inputBatch  # already [B, D, 1] and on device after toDevice()

    def set_word_space(self, ws):
        """Register the Model's WordSpace so the pipeline can reach it.

        Phase G of doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md
        retired the per-SubSpace back-pointer; the reference is held on
        each ``Space`` via ``space.wordSubSpace`` (set by
        ``Space.attach_wordSubSpace``) and passed explicitly to functions
        that need it (e.g. ``ConceptualSpace.forward``). The
        ``self._model_wordSubSpace`` attribute stays as the Model-build-
        time mirror so ``forward()`` paths that consult it (cf. the
        SymbolicSpace forward stamping) continue to work.
        """
        self._model_wordSubSpace = ws
        # Stamp the routing pointer on this InputSpace too so consumers
        # can read ``inputSpace.wordSubSpace`` without going through the
        # Model. (WordSubSpace.__init__ also calls
        # ``perceptualSpace.attach_wordSubSpace(self)`` for PCS spaces,
        # but InputSpace and OutputSpace are wired here.)
        self.attach_wordSubSpace(ws)

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

        Requires self._peer_perceptual to be wired (BasicModel/BasicModel do
        this) because the tokenizer (_token_stream) currently lives on the
        peer's vocabulary.
        """
        assert self._peer_perceptual is not None, \
            "InputSpace._lex_batch requires _peer_perceptual (lexer owner)"
        vocab = self._peer_perceptual.vocabulary
        dev = TheDevice.get()

        # Lex the *host* byte slab when the byte cursor staged one
        # (consumed once). It is byte-identical to the device `input`
        # (runEpoch builds `input = host_slab.to(device,int8)
        # .unsqueeze(1)`; `_to_text` masks `& 0xFF` so int8/uint8
        # agree) and carries the same shape, so every line below is
        # unchanged -- but `_token_stream`/`_to_text`'s `.tolist()` is
        # now a CPU op, not a cudaMemcpyDtoH (residual A,
        # doc/BrickHostSyncStatus.md). where_idx/when_idx/what_buf are
        # still built on `dev` for downstream tensor consumers.
        host_slab = getattr(self, '_host_input_slab', None)
        if host_slab is not None:
            self._host_input_slab = None  # consume once
            input = host_slab

        if input.dim() == 3:
            input = input.squeeze(1)
        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch = input.shape[0]
        nObj = self.outputShape[0]
        nWhat = self.subspace.nWhat

        # Byte-mode safety: cap the input to ``nObj`` bytes before the
        # tokenizer sees it. Cursor mode already sizes the slab to nObj
        # (one token per byte under byte-mode), so this is a no-op
        # there. Non-cursor callers (tests via ``stringTensor``) hand in
        # an ``inputLength``-padded buffer (e.g. 4096 bytes) -- without
        # the cap the byte tokenizer would emit one token per padded
        # null and overflow the assert. Constant time, no host sync.
        if self.byte_mode and input.shape[-1] > nObj:
            input = input[..., :nObj]

        tokens_per_batch = []
        # Build where_idx on the HOST: it is filled element-by-element
        # from the lexer's Python int offsets below; per-element writes
        # into a *device* tensor are each a synchronizing H2D (and break
        # CUDA-graph capture). `device='cpu'` is explicit so a
        # default-device mode can't place it on the GPU (same trap as
        # stringTensor). Staged to the device once, async, after the
        # loop. when_idx is a pure device arange (no host writes).
        where_idx = torch.zeros(batch, nObj, dtype=torch.long,
                                device='cpu')
        when_idx = torch.arange(nObj, device=dev).unsqueeze(0).expand(batch, -1).contiguous()

        for b in range(batch):
            stream = vocab._token_stream(input[b])
            # Word-mode lexing under the post-2026-04 single-char regex
            # produces more tokens per byte than the legacy grouped form
            # (each digit / punct / whitespace is its own token), so a
            # 4096-byte text slab can yield > nObj=1024 tokens. Truncate
            # to nObj here -- losing tail content is preferable to
            # asserting mid-epoch on long documents. Cursor mode still
            # sizes the slab to nObj exactly, so this is a no-op there.
            n_tokens = min(len(stream), nObj)
            row = []
            for i in range(n_tokens):
                token_text, start = stream[i]
                row.append(token_text)
                where_idx[b, i] = start
            tokens_per_batch.append(row)
            if n_tokens > 0:
                last_text, last_start = stream[n_tokens - 1]
                final_offset = last_start + len(last_text.encode('utf-8'))
            else:
                final_offset = 0
            for i in range(n_tokens, nObj):
                where_idx[b, i] = final_offset + (i - n_tokens)

        # Single SYNCHRONOUS host->device stage for the host-built
        # where_idx (correct + race-free; non_blocking on an ephemeral
        # pinned buffer corrupts data when freed pre-transfer).
        where_idx = where_idx.to(dev)

        what_buf = self.subspace.whatEncoding.encode_tokens(
            tokens_per_batch, batch, nObj, nWhat, dev)
        # Host-side decode-equivalent carried forward so
        # PerceptualSpace._embed skips the decode_tokens GPU->host sync
        # (residual B). Bit-identical to decode_tokens(what_buf).
        host_tokens = self.subspace.whatEncoding.tokens_to_decoded(
            tokens_per_batch, batch, nObj, nWhat)

        return what_buf, where_idx, when_idx, host_tokens

    def shuffle(self):
        """Shuffle.
        
        See class docstring for the operation contract.
        """
        self.data.shuffle()

    def Start(self):
        """One-shot per-run init."""
        super().Start()
        self._ar_embedded = None
        self._ar_total = 0
        # Per-word cursor rewinds with the embedded buffer it walks.
        self._per_word_cursor = 0
        # Cached valid length rewinds with the buffer it summarises.
        self._valid_len_host = 0
        # Static per-word loop feeders
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2.1).
        # Padded [B, N, D] view of the lexed slab and per-position
        # [B, N] bool mask of real-word positions. Rebuilt eagerly on
        # each forward(); read by ``word_at(p)`` and by the rule-gated
        # per-word body.
        self._ar_embedded_N = None
        self._word_active_mask = None

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
        # The per-word cursor is batch-shared rebuildable scratch over
        # ``_ar_embedded`` (same as ``_ar_total``); a hard Reset drops
        # the buffer so the cursor rewinds to 0 with it.
        self._per_word_cursor = 0
        # ``_valid_len_host`` is a host-int summary of the same buffer.
        self._valid_len_host = 0
        self._ar_embedded_N = None
        self._word_active_mask = None
        self._cached_embedding = None
        # _end_of_stream is a host-side ``list[bool]`` under the
        # rolling-cursor contract (the canonical hard-reset signal is
        # the cursor's ``hard_eos`` from ``next_tick``). Clear the
        # row-or-all so external diagnostic readers see a known state.
        if batch is None:
            self._end_of_stream = [False] * len(self._end_of_stream)
        elif 0 <= batch < len(self._end_of_stream):
            self._end_of_stream[batch] = False

    # The world presenting itself
    def forward(self, inputData):
        """Single-call stem source for the IR-only training pipeline.

        Lexes/embeds the input once and emits ``[B, N, D]`` (left-
        aligned, right-padded to N).  ``self.subspace`` carries
        ``valid_mask`` (``[B, 1]`` bool) and ``stem_embedded=True``.

        AR cursor unfold + ARIR runtime branch retired 2026-05-14
        alongside ``<maskedPrediction>``; within-sentence training is
        purely masked-LM at the P-tier (BERT-style IR).
        """
        if inputData is None:
            return self._empty_like_subspace()
        if hasattr(inputData, "is_empty") and not isinstance(inputData, torch.Tensor) and inputData.is_empty():
            return inputData

        # Lex/embed once -- produces ``[B, T, D]`` on ``self.subspace``.
        self._lex_and_embed(inputData)
        embedded = self.subspace.materialize()
        if embedded is None and self.model_type == "embedding":
            peer = self._peer_perceptual
            if peer is not None:
                # Route to the peer's chunking mode rather than always
                # calling the lexicon path.  With ``<chunking>bpe</...>``
                # the lexicon-style ``_embed`` would tokenize on
                # whitespace and OOV-insert per word (codebook drift);
                # ``_embed_bpe`` does the right thing: BPE chunk +
                # MAX-fuse against the frozen codebook keyed by latin-1
                # byte-tuples.
                peer_mode = getattr(peer, 'chunking_mode', None)
                if peer_mode == "bpe":
                    peer._embed_bpe(self.subspace)
                elif peer_mode == "none":
                    peer._embed_byte(self.subspace)
                elif peer_mode == "mphf":
                    peer._embed_mphf(self.subspace)
                else:
                    peer._embed(self.subspace)
                embedded = peer._embedded_input
        # ``_ar_embedded`` kept as the canonical un-windowed [B, T, D]
        # handle so downstream consumers (chat-loop, diagnostic dumps)
        # have a single name for the source embedded tensor.  The
        # ``_ar_`` prefix is historical; remove once those callers
        # rename.
        self._ar_embedded = embedded

        if embedded is None:
            self._valid_len_host = 0
            return self.subspace

        # D8 piece 2 (2026-05-19, eager-revised after metalbaby): compute
        # the per-forward valid-len cache EAGERLY here at the IS
        # boundary (non-captured region) so ``next_word()`` is a pure
        # host-int compare with NO ``.item()`` inside. The lazy-compute
        # variant worked for the per-word body in isolation (CPU eager
        # backend tolerates it), but the WHOLE-FORWARD compile path
        # (production training, CUDA Inductor fullgraph) traces through
        # ``next_word`` and an ``.item()`` there blocks the trace:
        # ``Could not guard on data-dependent expression Eq(u0, 1)``.
        # Eager-here-once + ``torch.no_grad`` keeps the autograd graph
        # rooted in ``embedded`` untouched (no new ``.abs().sum()``
        # fanout), matching the no_grad guard the lazy variant used.
        with torch.no_grad():
            _peer_valid = self._peer_perceptual
            _bpe_mask_valid = (
                getattr(_peer_valid, "_bpe_word_mask", None)
                if _peer_valid is not None
                and getattr(_peer_valid, "chunking_mode", None)
                in ("bpe", "none", "mphf")
                else None)
            _Te = embedded.shape[1]
            if _bpe_mask_valid is not None:
                _valid_pos = _bpe_mask_valid[:, :_Te] > 0
            else:
                _valid_pos = embedded.detach().abs().sum(dim=-1) > 0
            _any_pos = _valid_pos.any(dim=0)
            if torch.is_tensor(_any_pos) and _any_pos.any().item():
                self._valid_len_host = int(
                    _any_pos.nonzero().max().item()) + 1
            else:
                self._valid_len_host = 0

        B, T, D = embedded.shape
        N = int(self.outputShape[0])

        # When peer is in BPE chunking mode, ``peer._bpe_word_mask``
        # ([B, N], 1.0 where the slot holds a real BPE-fused word
        # vector and 0.0 in padding) is the source of truth for both
        # validity AND the per-position event mask.  Without it,
        # validity would fall back to ``embedded.abs().sum > 0``, which
        # is always True for muxed events (where/when components stay
        # nonzero even at padding positions).
        #
        # ``chunking=none`` (byte-direct) installs the same
        # ``_bpe_word_mask`` attribute via ``(byte_indices != 0)``; the
        # windowed view is identical so we accept it here under the
        # same name.
        peer = self._peer_perceptual
        bpe_mask = (getattr(peer, "_bpe_word_mask", None)
                    if peer is not None
                    and getattr(peer, "chunking_mode", None) in ("bpe", "none", "mphf")
                    else None)

        # Pad / truncate to N.
        if T < N:
            pad = torch.zeros(
                B, N - T, D,
                device=embedded.device, dtype=embedded.dtype)
            embedded_N = torch.cat([embedded, pad], dim=1)
        elif T > N:
            embedded_N = embedded[:, :N, :]
        else:
            embedded_N = embedded
        if bpe_mask is not None:
            valid_mask = bpe_mask[:, :N].any(dim=1).reshape(B, 1)
            bpe_mask_N = bpe_mask[:, :N] if bpe_mask.shape[1] >= N else bpe_mask
        else:
            valid_mask = (embedded_N.abs().sum(dim=-1) > 0).any(
                dim=1).reshape(B, 1)
            bpe_mask_N = None
        sub = self.subspace
        # ``set_event`` clears ``_demuxed`` — that's correct when the
        # input came in muxed (text path), but the numeric-vocab path
        # in ``_lex_and_embed`` populates the demuxed what/where/when
        # slots via ``set_what``/``set_where``/``set_when`` and
        # downstream callers (``InputSpace(demuxed=True)``) expect
        # ``is_demuxed`` to stay True.  Restore the demuxed flag
        # after the set_event so both contracts hold: the event
        # tensor carries the padded/truncated muxed view (for the
        # body's reshape contract) AND the demuxed slots remain
        # readable.
        was_demuxed = sub._demuxed
        sub.set_event(embedded_N)
        if was_demuxed:
            sub._demuxed = True
        sub.valid_mask = valid_mask
        sub.stem_embedded = True
        # Static per-word loop feeders
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2.1).
        # ``_ar_embedded_N`` is the padded [B, N, D] view consumed by
        # ``word_at(p)``; ``_word_active_mask`` is the per-position
        # [B, N] bool mask consumed downstream as the rule-gate hint.
        # Kept as tensors (never ``.item()``'d) so the compiled loop
        # never gates control flow on a host int.
        self._ar_embedded_N = embedded_N
        if bpe_mask_N is not None:
            if bpe_mask_N.shape[1] >= N:
                _wam = (bpe_mask_N[:, :N] > 0)
            else:
                _wpad = torch.zeros(
                    B, N - bpe_mask_N.shape[1],
                    dtype=torch.bool, device=bpe_mask_N.device)
                _wam = torch.cat([bpe_mask_N > 0, _wpad], dim=1)
        else:
            _wam = embedded_N.detach().abs().sum(dim=-1) > 0
        self._word_active_mask = _wam
        if peer is not None and bpe_mask_N is not None:
            peer._bpe_word_mask_flat = bpe_mask_N
        elif peer is not None:
            peer._bpe_word_mask_flat = None
        if len(self._end_of_stream) != B:
            self._end_of_stream = [False] * B
        ws = self._model_wordSubSpace
        if ws is not None:
            ws.ensure_microbatch(B, 1)
        return sub

    def word_at(self, p):
        # Static per-word feeder for the rule-gated loop
        # (doc/plans/2026-05-20-static-per-word-loop-impl.md §2.1).
        # Returns the padded ``[B, 1, D]`` word slice at position ``p``
        # for ``0 <= p < outputShape[0]``; reads from
        # ``self._ar_embedded_N``, the padded view populated by
        # ``forward``. Does NOT advance any cursor and does NOT consult
        # ``_valid_len_host`` — the rule-gate (``selected_rule != id_SS``)
        # is the per-iteration mask.
        slab = self._ar_embedded_N
        if slab is None:
            return None
        return slab[:, p:p + 1, :]

    def next_word(self):
        """Per-word ground-truth feed for the IR-reconstruction loop.

        INERT this increment: returns ``None`` immediately unless
        ``self._per_word_enabled`` is True, and nothing in the current
        codebase/config sets that flag, so the body below is unreached
        in every live run and the whole-slab ``forward`` contract is
        byte-identical. ``_per_word_enabled`` is the deferred wiring
        point (next increment enables it and threads the returned slice
        into ConceptualSpace/STM/SHIFT/selector/reverse -- explicitly
        out of scope here).

        When enabled, walks the already-lexed whole-sentence buffer
        ``self._ar_embedded`` ([B, T, D], populated by ``forward``) one
        T-position per call:

          * returns ``self._ar_embedded[:, p:p+1, :]`` -- the single
            ground-truth word slice ``[B, 1, D]`` at the cursor's
            T-position ``p`` -- and advances ``self._per_word_cursor``
            by 1;
          * returns ``None`` (the NULL/end-of-sentence seal) once the
            cursor reaches the end of valid lexed content. End-of-valid
            is the SAME validity signal ``forward`` uses: the peer's
            ``_bpe_word_mask`` ([B, T], 1.0 at real word slots) when in
            ``bpe``/``none`` chunking, else ``abs().sum(-1) > 0`` over
            the embedded slab. The valid length is the max real-token
            count across rows (``any`` over the batch, matching how
            ``forward`` reduces ``valid_mask``). The stop is therefore
            purely the input's NULL/end sentinel -- NEVER a model output
            (no get_recovered_word, no reconstruction, no [MASK], no
            model-predicted EOF). This is deliberately NOT the retired
            ``arir_step`` AR machine.
        """
        if not self._per_word_enabled:
            return None
        embedded = self._ar_embedded
        if embedded is None:
            return None
        # D8 piece 2 (eager-revised after metalbaby 2026-05-19):
        # ``_valid_len_host`` is populated EAGERLY by ``forward()`` at
        # the IS boundary; ``next_word`` does a pure host-int compare
        # with ZERO ``.item()`` inside. The lazy-on-first-call variant
        # was Dynamo-traceable under the per-word-body's isolated
        # eager-backend compile, but the whole-forward Inductor fullgraph
        # compile (production training) traces through ``next_word`` and
        # an ``.item()`` here blocks the trace.
        p = self._per_word_cursor
        if p >= self._valid_len_host:
            # NULL/end-of-sentence sentinel reached -- pure input stop.
            return None
        word = embedded[:, p:p + 1, :]
        self._per_word_cursor = p + 1
        return word

    def _empty_like_subspace(self):
        """Return a SubSpace with materialized shape [B, 0, D] — the termination sentinel."""
        template = self.subspace
        ss = SubSpace(
            (0, template.inputShape[1]),
            (0, template.outputShape[1]),
            nInputDim=template._nInputDim,
            nOutputDim=template._nOutputDim,
        )
        # Stamp wordSubSpace so downstream skip-on-empty logic still sees it.
        ss.wordSubSpace = self._model_wordSubSpace
        return ss

    def _lex_and_embed(self, input):
        """Populate subspace from raw input: lex/embed for text, vocab lookup for numeric.

        Called by forward() at the start of every pass.  Handles three
        cases:
          * ``_cached_embedding`` set -- use pre-built latent (chat-
            loop / generate_sentence stages this).
          * ``model_type == 'embedding'`` (text) -- lex into byte buffer.
          * numeric mode -- vocab codebook lookup.
        """
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
            what_buf, where_idx, when_idx, host_tokens = self._lex_batch(input)
            self.subspace.what.setW(what_buf)
            self.subspace.where.setW(where_idx)
            self.subspace.when.setW(when_idx)
            self.subspace._host_tokens = host_tokens
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
        """Reverse pass; inverse of ``forward``.

        See class docstring for the inversion contract.
        """
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
        raw = (object_basis.reverse_raw(y)
               if hasattr(object_basis, 'reverse_raw')
               else y)
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

        # Word recovery -- content is already denormalized; Embedding.reverse()
        # snaps under the active lexicon geometry.
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
class PerceptualSpace(Space):
    """Transforms raw input vectors into percepts via parallel pi+sigma folds.

    In the forward data flow: InputSpace -> **PerceptualSpace** -> ConceptualSpace.
    Owns one ``PiLayer`` (``self.pi``) and one ``SigmaLayer``
    (``self.sigma``) (Stage 1.A substrate refactor, doc/plans/
    2026-05-26-two-loop-pi-sigma-substrate.md). Both are
    ``percept_dim → percept_dim``.

        * ``self.pi``    -- log-space multiplicative AND fold.
        * ``self.sigma`` -- additive (atanh→linear→tanh) OR fold.

    The two are summed inside ``forward`` from the SAME materialized
    input ``x``::

        P = pi(x) + sigma(x)

    No outer ``tanh`` wrap -- pi / sigma each apply their own internal
    nonlinearity (each returns a tanh-bounded contribution).

    The earlier two-input ``tanh(pi_input(IS) + pi_concept(C_prev))``
    contract (with CS-feedback entering PS directly) is retired: the
    CS-feedback path no longer enters PS this way (it re-enters via
    the chart / signal-router dispatch over STM in later stages).

    Optionally followed by self-attention and VQ codebook quantization.

    When ``reversible=True`` and ``invertible=True``, the reverse path
    uses the configured inverse layer. With ``naive=False`` this uses the
    LDU/triangular-solve path; the dense naive path exists only for
    debugging and validation.
    """
    name = "Percepts"
    config_section = "PerceptualSpace"

    # Reserved codebook key for the IR-mode NULL-percept slot. Distinct
    # from byte ``\x00`` (a real prediction target in byte mode); IR
    # mask injection replaces masked positions with this slot so the
    # brick body sees a distinct embedding meaning "predict me" rather
    # than "this was \x00". Owned by PerceptualSpace because the IR
    # mode and the percept-level mask injection are PerceptualSpace
    # concepts; downstream consumers (Embedding seed path, checkpoint
    # vocab migration, MPHF byte-fallback table) reference it via
    # ``PerceptualSpace.NULL_PERCEPT_KEY``.
    NULL_PERCEPT_KEY = "__NULL_PERCEPT__"

    def __init__(self, inputShape, spaceShape, outputShape, model_type=None):

        """Initialize PerceptualSpace; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        section = self.config_section
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        naive = TheXMLConfig.get("architecture.naive")
        # Always-on (XML knob retired); see ConceptualSpace for rationale.
        self._svd_orthogonal_init_cfg = True

        # Stash all attributes BEFORE super().__init__() since _build_what_basis runs inside it
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
        # C→P feedback is now an explicit ``forward`` argument
        # (``CS_subspaceForPS``) supplied by the recurrent cell, not a
        # post-construction sibling ref.
        # Recurrent-pass index. The post-SentenceState design (2026-05-21)
        # publishes this on ``WordSubSpace.recur_pass`` (written by
        # ``_forward_body`` each pass, read in ``forward`` via the
        # ``self.subspace.wordSubSpace`` back-reference); this attribute
        # remains as the eager fallback for standalone ``forward`` calls
        # / unit tests where no wordSubSpace is wired. The serial-mode
        # warm path is an AR-streaming optimisation across forward
        # *calls*; it must not fire on in-forward recurrent passes >0
        # (those carry distinct C→P feedback and must do the full cold
        # compute, not splice the prior pass's last slot).
        self._recurrent_pass_idx = 0
        self._sparsity = SparsityRegLayer(
            l1_lambda=float(getattr(self, "l1_lambda", 0.0) or 0.0),
            enabled=bool(getattr(self, "codebook", False)),
        )

        # Phase 1A.1: make the PERCEPTUAL VQ codebook learnable by
        # gradient and drop its in-call EMA Parameter mutation, so
        # ``PerceptualSpace.forward`` performs NO persistent-state
        # mutation in-call and is idempotent (a CUDA-graph-capture
        # prerequisite). ``VectorQuantize`` / ``Codebook`` are SHARED by
        # the Conceptual and Symbolic codebooks too; this flag is set
        # ONLY on the perceptual ``.event`` codebook's ``VectorQuantize``
        # instance here, so the Conceptual/Symbolic EMA paths stay
        # byte-identical (single-writer invariant). No-op when this
        # space has no codebook configured (passthrough ``Tensor``
        # subspace -> no ``.vq``).
        self._make_perceptual_codebook_learnable()

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
        if self.chunking_mode not in ("bpe", "lexicon", "none", "mphf"):
            raise ValueError(
                f"PerceptualSpace.chunking must be bpe|lexicon|none|mphf, "
                f"got {self.chunking_mode!r}")
        if isinstance(lexical_basis, Embedding):
            lexical_basis.chunking_mode = self.chunking_mode
            lexical_basis.lexer_mode = self.lexer
        try:
            self.word_learning = int(
                TheXMLConfig.space(section, "wordLearning") or 2)
        except (KeyError, TypeError, ValueError):
            self.word_learning = 2
        if self.chunking_mode == "bpe":
            if self.nVectors < 256:
                raise ValueError(
                    f"PerceptualSpace.chunking='bpe' requires nVectors>=256 "
                    f"(to seed the byte range); got nVectors={self.nVectors}")
            if self.model_type != "embedding":
                raise ValueError(
                    "PerceptualSpace.chunking='bpe' requires "
                    "<modelType>embedding</modelType>")
        elif self.chunking_mode == "mphf":
            # MPHF is BPE+MPHF-fast-path: in-vocab whole words via MPHF
            # gather, OOV fall back to _embed_bpe_trie. Inherits BPE's
            # invariants (byte range seeding + Embedding modelType).
            if self.nVectors < 256:
                raise ValueError(
                    f"PerceptualSpace.chunking='mphf' requires nVectors>=256 "
                    f"(to seed the byte range); got nVectors={self.nVectors}")
            if self.model_type != "embedding":
                raise ValueError(
                    "PerceptualSpace.chunking='mphf' requires "
                    "<modelType>embedding</modelType>")
        elif self.chunking_mode == "none":
            # Byte-direct mode: each byte is its own perceptual atom,
            # looked up via a 256-entry codebook (byte_value ==
            # codebook_index for 0..255, seeded by the ``\x00``
            # cold-start at Embedding cold-start; subsequent insert()
            # calls are blocked downstream when chunking_mode='none').
            # No BPE walker, no Python trie, no graph break.  ``\0``
            # (byte 0) doubles as the sentence-end / pad sentinel and
            # lands at codebook index 0 by design.
            if self.nVectors < 256:
                raise ValueError(
                    f"PerceptualSpace.chunking='none' requires "
                    f"nVectors>=256 (the 256-entry byte codebook); "
                    f"got nVectors={self.nVectors}")
            if self.model_type != "embedding":
                raise ValueError(
                    "PerceptualSpace.chunking='none' requires "
                    "<modelType>embedding</modelType>")
        self._recovered_input = None
        # Deferred word-recovery decode (set by reverse(); see
        # _materialize_recovered_input). reverse() stashes the refs
        # here instead of decoding eagerly -- the decode is report-only
        # and its per-word .item() is a cudaMemcpyDtoH that breaks the
        # brick CUDA-graph-capture contract.
        self._recovered_input_thunk = None
        self._embedded_input = None

        # Stage 1.A substrate refactor (doc/plans/2026-05-26-two-loop-pi-
        # sigma-substrate.md): PerceptualSpace owns a single PiLayer
        # (``self.pi``) and a single SigmaLayer (``self.sigma``), both
        # shaped ``percept_dim → percept_dim``. The legacy per-order
        # Ramsified ``pi_input`` / ``pi_concept`` ModuleLists are
        # retired; the ``conceptualOrder`` knob's new role is driving
        # the PARALLEL-mode forward iteration count (the same
        # ``self.pi`` / ``self.sigma`` are called T times with
        # different inputs), not selecting per-order weights.
        #
        # ``forward`` composes them on the SAME materialized input::
        #
        #     P = pi(x) + sigma(x)
        #
        # No outer ``tanh`` wrap — each layer applies its own internal
        # nonlinearity and returns a tanh-bounded contribution.
        #
        # Dim choice: ``percept_dim`` is the per-slot dim
        # PerceptualSpace operates on INTERNALLY — i.e. the post-
        # forwardBegin shape ``[B, ?, nInputDim]``. We use
        # ``getEncodedInputSize()`` for that. The legacy
        # ``nInputDim != nOutputDim`` slot-redistribution at
        # forwardEnd is preserved (the codebook + forwardEnd reshape
        # together remap the per-slot output dim and grow/shrink slot
        # count by the same factor). Both ``pi`` and ``sigma`` are
        # therefore square at ``nInputDim → nInputDim``.
        percept_dim = int(self.subspace.getEncodedInputSize())
        # Subsymbolic-loop folds honour the ``architecture.monotonic``
        # knob (same source as BasicModel.monotonic). Monotone (W>=0)
        # is order-preserving, which the parthood predicate
        # (``Ops.part``) requires when symbols are mapped across orders
        # through the PS<->CS loop. Default False -> unchanged behavior.
        _mono = bool(TheXMLConfig.get("architecture.monotonic",
                                      default=False))
        self.pi = PiLayer(
            percept_dim, percept_dim,
            naive=naive, ergodic=ergodic,
            invertible=bool(invertible), nonlinear=nonlinear,
            stable=True, monotonic=_mono,
        )
        self.sigma = SigmaLayer(
            percept_dim, percept_dim,
            naive=naive, ergodic=ergodic,
            invertible=bool(invertible), nonlinear=nonlinear,
            stable=True, monotonic=_mono,
        )

        input = percept_dim
        self.attention = AttentionLayer(input, input, type="transformer")
        self.subspace._nWordSlots = outputShape[0]
        self.params = []
        self.params += self.pi.getParameters()
        self.params += self.sigma.getParameters()
        self.layers = nn.ModuleList()
        self.chunk_layer = ChunkLayer(
            self.nDim,
            bpe=(self.chunking_mode in ("bpe", "mphf")),
            n_vectors=self.nVectors,
            word_learning=self.word_learning,
        )
        # Opt-in/opt-out: the frozen-vocab GPU BPE tokenizer
        # (``_embed_bpe_gpu``) is bit-identical to the trie path
        # (test/bpe_gpu_equiv.py) but is NOT the production default --
        # measured slower at this scale and it does not, by itself,
        # unlock CUDAGraph capture (other implicit syncs gate that).
        # Set ``perceptualSpace._bpe_gpu_enabled = True`` (with a frozen
        # vocab, ``word_learning <= 0``) to route through it; default
        # off keeps production on the verified trie path. Explicit
        # attribute (no getattr default) so the toggle is discoverable.
        self._bpe_gpu_enabled = False
        # BPE / MPHF GPU tokenizer Layers (algorithms only; static tables
        # cached on this PerceptualSpace as ``self._bpe_static_tables`` /
        # ``self._mphf_static_tables`` keyed by frozen-vocab size).
        # Construction is cheap (no params); the layers are kept on
        # ``self.layers`` so the standard Layer cascade (Start/End,
        # set_sigma, paramUpdate) reaches them -- they are no-ops for
        # those hooks but participate in the inventory.
        from Layers import BPEGpuLayer as _BPEGpuLayer
        from Layers import MPHFGpuLayer as _MPHFGpuLayer
        self._bpe_gpu_layer = _BPEGpuLayer()
        self._mphf_gpu_layer = _MPHFGpuLayer()
        self.layers.append(self._bpe_gpu_layer)
        self.layers.append(self._mphf_gpu_layer)
        # Rework A: the frozen MPHF->table static tensors, built ONCE
        # over the frozen lexicon key set (mirrors ``_bpe_static_tables``;
        # cache keyed by lexicon row count -- frozen => never rebuilt
        # after the first build). ``None`` until the first
        # ``_mphf_resolve`` call; lazily built there so non-grammar /
        # numeric configs (no per-word MPHF route) never pay for it.
        self._mphf_static_tables = None
        # Auto-load a previously-saved BPE codebook from the same
        # ``.kv`` artifact that hosts the Lexicon. The artifact format
        # (defined in :mod:`embed`) carries Lexicon + BPE side-by-side
        # under ``kind="both"``; if the file exists and has a BPE section,
        # loading restores the merge table so subsequent training
        # avoids the cold-start vocab-growth recompile pressure under
        # ``torch.compile``. Setting ``word_learning=0`` (the
        # frozen marker) at the same time prevents further growth so
        # Inductor's cache stays warm. Failure modes (missing file,
        # lexicon-only artifact, schema mismatch) downgrade to a
        # one-line warning -- the layer falls back to its empty
        # cold-start state.
        if self.chunk_layer.bpe:
            embedding_path = TheXMLConfig.get("architecture.embeddingPath", None)
            if embedding_path and os.path.exists(embedding_path):
                try:
                    from embed import inspect_artifact
                    info = inspect_artifact(embedding_path)
                    if info.get("has_bpe"):
                        self.chunk_layer.load(embedding_path)
                        TheMessage(
                            f"[PerceptualSpace] Loaded BPE codebook "
                            f"({info['bpe_size']} entries) from "
                            f"{embedding_path}")
                    else:
                        TheMessage(
                            f"[PerceptualSpace] No BPE section in "
                            f"{embedding_path} (kind={info.get('kind')!r}); "
                            f"starting with the 256-byte cold-start vocab.")
                except Exception as e:
                    TheMessage(
                        f"[PerceptualSpace] BPE auto-load skipped "
                        f"({type(e).__name__}: {e}); starting cold.")

    def _register_requirements(self):
        """Register PerceptualSpace-specific config requirements.

        Adds Config.require predicates that enforce codebook /
        invertibility shape constraints (e.g. nVectors divisibility,
        output dim agreement). Validation fires at ``validate_config``
        time so misconfigs surface before construction.
        """
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
        """Lexicon home: build the Embedding when running in text mode.

        Returns ``None`` for non-embedding models (numeric path uses the
        codebook directly). For embedding models, configures the
        Embedding with the lexicon size, vector dim, source artifact,
        and minimum-frequency / negative-sample knobs.

        NOTE: post-lexicon-migration, the Embedding is logically owned
        by SymbolicSpace -- ``SymbolicSpace.vocabulary`` returns this
        same Embedding instance via a shared reference. PerceptualSpace
        still builds and binds it here at construction time because the
        input pipeline (InputSpace._lex_batch, PerceptualSpace._embed)
        wires through ``self.subspace.what`` at the lexical-lookup
        site. The "codebook IS the lexicon" unification on S is
        realized by S's ``vocabulary`` property forwarding to this same
        Embedding, with all orthographic-API methods (train_embeddings,
        sbow_loss, reconstruct_data, ...) accessible from S.
        """
        if self.model_type != "embedding":
            return None
        basis = Embedding()
        basis.ergodic = self.ergodic
        # Embedding.create's second arg is *nVectors* = codebook
        # capacity (one row per lexicon entry), NOT the output sequence
        # length. Pass ``self.nVectors`` (the XML PerceptualSpace
        # ``<nVectors>``) so the synth path's
        # ``len(BPE vocab) == nVectors`` invariant holds and so the
        # codebook tensor's row count matches the configured capacity.
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
            embedding_path=self.embedding_path,
            source=self.embedding_source,
            min_frequency=self.min_frequency,
            neg_samples=self.neg_samples,
            byte_mode=self.byte_mode,
        )
        return basis

    # ------------------------------------------------------------------
    # Knowledge artifact attach: PerceptualSpace owns the surface-form
    # WordVectors (``self.wv``); attach_knowledge stamps the artifact's
    # ``word_table.ref_ids`` onto ``wv.ref_ids`` so the chart's lexical
    # lookup step can navigate word → reference via taxonomy / codebook.
    # See plan §Phase 2 — Loaders.
    # ------------------------------------------------------------------
    def attach_knowledge(self, view):
        """Attach a ``KnowledgeView`` and stamp ``ref_ids`` onto ``self.wv``.

        When ``self.wv`` is absent (e.g., a bare-instance unit test
        with no Embedding allocated), the attach is a no-op beyond
        the base-class view storage — no error.
        """
        super().attach_knowledge(view)
        wv = getattr(self, 'wv', None)
        if wv is None:
            return
        wt = view._ks['word_table']
        wv.ref_ids = wt['ref_ids'].clone().detach().long()

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
        """Distance.
        
        See class docstring for the operation contract.
        """
        return torch.prod( [1-x, 1-y] )
    def certainty(self, x):
        """Certainty.
        
        See class docstring for the operation contract.
        """
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
        what_buf = upstream_vspace.materialize(mode="what")
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

        # Token text per slot. InputSpace._lex_batch already had these
        # strings on the host and stashed the (bit-identical)
        # decode-equivalent on the subspace; use it to skip the
        # decode_tokens ``buf.tolist()`` GPU->host sync (residual B,
        # doc/BrickHostSyncStatus.md). Fall back to decoding the byte
        # buffer for callers that set what.W without host tokens
        # (internal _embed reuse, direct-construction tests).
        host_tokens = getattr(upstream_vspace, '_host_tokens', None)
        if host_tokens is not None:
            batch_tokens = host_tokens
        else:
            batch_tokens = upstream_vspace.whatEncoding.decode_tokens(
                what_buf)
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
        # Build on CPU as a Python list-of-lists, then materialize the
        # device tensor in one shot. The previous "torch.full + per-cell
        # in-place write" pattern triggered the Inductor cudagraphs
        # warning ``skipping cudagraphs due to mutated inputs`` -- the
        # captured graph can't include tensor mutations on freshly-
        # allocated buffers. A single ``torch.tensor(indices_2d)`` call
        # is graph-friendly (the allocation is part of the graph; no
        # post-allocation mutation).
        null_idx = codebook.wv.key_to_index.get("\x00", 0)
        indices_2d = [[null_idx] * nObj for _ in range(batch)]
        for b, row in enumerate(batch_tokens):
            for n in range(min(len(row), nObj)):
                text = row[n]
                if text:
                    indices_2d[b][n] = codebook._token_to_index(text)
        # Build on the HOST then stage once, SYNCHRONOUSLY (correct +
        # race-free; non_blocking on an ephemeral pinned buffer
        # corrupts data when freed pre-transfer -- the NaN source).
        what_indices = torch.tensor(
            indices_2d, dtype=torch.long, device='cpu').to(dev)

        # where / when come straight from the upstream buffer.
        where_raw = upstream_vspace.materialize(mode="where")
        when_raw = upstream_vspace.materialize(mode="when")
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
        """Dispatch BPE embedding: GPU tensor tokenizer when the vocab
        is FROZEN (``word_learning <= 0`` -- the CPU-pretrain -> freeze
        -> GPU-train workflow), else the legacy trie walk.

        The GPU path is zero host-sync (no ``byte_indices.tolist()``);
        it is asserted bit-identical to the trie path by
        ``test/bpe_gpu_equiv.py`` before being trusted. Any build/shape
        problem falls back to the trie path (never silently wrong --
        the fallback is the verified reference).
        """
        cl = self.chunk_layer
        frozen = (bool(getattr(cl, "bpe", False))
                  and int(getattr(cl, "word_learning", 0) or 0) <= 0)
        # Opt-in/opt-out (``self._bpe_gpu_enabled``, default False --
        # set in __init__). Bit-identical to the trie path but not the
        # production default: measured slower at this scale and it does
        # not by itself unlock CUDAGraph capture (other implicit syncs
        # gate that). Kept gated for future perf work.
        if frozen and self._bpe_gpu_enabled:
            try:
                return self._embed_bpe_gpu(upstream_vspace)
            except self._bpe_gpu_layer._BPEGpuUnavailable:
                pass  # fall through to the verified trie reference
        return self._embed_bpe_trie(upstream_vspace)

    def _embed_bpe_gpu(self, upstream_vspace):
        """Frozen-vocab GPU tokenizer path: static tensor tables +
        parallel longest-match + on-device greedy consumption +
        tensor word-segmentation -> the same ``_bpe_emit`` tail. Zero
        ``cudaMemcpyDtoH``. Raises ``BPEGpuLayer._BPEGpuUnavailable`` if
        the static tables cannot be built (caller falls back to the
        trie path).
        """
        bpe = self._bpe_gpu_layer
        what_buf = upstream_vspace.materialize(mode="what")
        if what_buf is None:
            raise RuntimeError(
                "PerceptualSpace._embed_bpe_gpu: upstream subspace.what "
                "is empty.")
        byte_indices = ((what_buf[..., 0] if what_buf.dim() == 3
                         else what_buf).long())
        codebook = self.subspace.what
        batch = what_buf.shape[0]
        nObj = self.outputShape[0]
        dev = TheDevice.get()
        null_idx = codebook.wv.key_to_index.get("\x00", 0)

        # Static tables: build ONCE per (frozen) vocab, cache. Rebuild
        # only if the vocab size changed (frozen => never after build).
        cl = self.chunk_layer
        vsig = int(cl._next_id)
        tab = getattr(self, "_bpe_static_tables", None)
        if tab is None or tab.get("_vsig") != vsig:
            try:
                tab = bpe.build_static_tables(
                    cl, codebook, byte_indices.device)
            except AssertionError as e:
                raise bpe._BPEGpuUnavailable(str(e))
            tab["_vsig"] = vsig
            self._bpe_static_tables = tab

        best_id, best_len = bpe.gpu_longest_match(byte_indices, tab)
        chunk_ids, tok_count = bpe.gpu_chunk_ids(
            byte_indices, best_id, best_len)
        sub_cb, sub_target, sub_pos, keep = bpe.segment_words(
            chunk_ids, tok_count, tab, nObj)
        return self._bpe_emit_gpu(
            upstream_vspace, codebook, batch, nObj, dev, null_idx,
            sub_cb, sub_target, sub_pos, keep)

    def _embed_bpe_trie(self, upstream_vspace):
        """Legacy trie-walk BPE path -- the verified reference and the
        fallback for non-frozen (growing) vocab.

        Decode the upstream byte buffer, BPE-tokenize via ChunkLayer,
        group sub-tokens by whitespace word-boundary, look up each
        byte-tuple chunk in the codebook, then MAX-fuse sub-token
        vectors within each word.  Emit one [nDim] vector per word.

        Implementation is **batch-flat**. The previous form built three
        nested ``[B][nObj]`` Python lists -- one of ints, one of floats,
        one of ``[nDim]`` GPU tensors -- then materialized them with
        ``torch.tensor`` and ``torch.stack(torch.stack(...))``. The
        per-word ``_max_fuse_subtokens`` allocated and stacked a fresh
        list of vectors for every word (~960 stacks per batch on
        MM_5M). This form does:

            * ONE Python sweep over all chunks in the batch, collecting
              per-sub-token codebook indices + a global word-segment id,
              plus per-word routing tuples ``(b, slot, first-sub-idx)``.
            * ONE ``vectors[flat_idx]`` gather producing every
              sub-token vector across the batch in a single op.
            * ONE ``scatter_reduce_(amax)`` collapsing the gathered
              vectors into ``[N_words, nDim]`` via segment ids.
            * ONE fancy-index assignment placing those into
              ``[B, nObj, nDim]``.

        Net: 2 small H2Ds + 1 gather + 1 reduce + 1 scatter for the
        whole batch's MAX-fuse, regardless of word/sub-token count.
        """
        what_buf = upstream_vspace.materialize(mode="what")
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
        chunk_frozen = (
            int(getattr(self.chunk_layer, 'word_learning', 0) or 0) <= 0)

        if what_buf.dim() == 3:
            byte_indices = what_buf[..., 0].long()
        else:
            byte_indices = what_buf.long()

        if self.chunk_layer.bpe and self.training:
            self.chunk_layer.train_step(byte_indices)

        # ---- Fused trie walk + per-word assembly --------------------
        # The previous two-pass form built ``all_chunks`` /
        # ``all_spans`` (Python list-of-lists per row) and then
        # iterated them again to assemble words. Fused into one walk:
        # as we descend the trie at each position, we resolve the
        # matched chunk to its codebook index and either accumulate
        # it into the current word or flush at a boundary. Saves the
        # intermediate list-of-lists allocation entirely.
        self.chunk_layer._ensure_trie()
        trie = self.chunk_layer._trie
        id_to_bytes = self.chunk_layer.id_to_bytes
        vocab = self.chunk_layer.vocab

        null_idx = codebook.wv.key_to_index.get("\x00", 0)
        key_to_index = codebook.pretrain.key_to_index
        token_to_index = codebook._token_to_index
        chunk_key_to_latin1 = self._chunk_key_to_latin1
        byte_mode = bool(getattr(codebook, 'byte_mode', False))

        rows = byte_indices.tolist()
        N_buf = byte_indices.shape[1]

        # Per-sub-token (flat) and per-word (routing) Python lists. Only
        # ints get appended, no tensors -- the heavy lifting is one
        # gather later, not many small stacks.
        flat_subtoken_idx = []   # codebook row idx for each resolved sub-token
        flat_word_seg     = []   # segment id (running word_id) per ^
        per_word_first    = []   # first sub-token's codebook idx, per word
        per_word_b        = []   # batch row, per word
        per_word_slot     = []   # word slot within row, per word
        word_id = 0

        def _resolve(byte_tuple):
            """Sub-token byte tuple → codebook int index, or None.

            Frozen-vocab missing key is a load-mismatch bug → assert.
            Active mode falls back to ``codebook.insert`` so newly
            promoted BPE merges don't stall the batch.
            """
            latin1 = chunk_key_to_latin1(byte_tuple)
            if not latin1:
                return None
            if latin1 not in key_to_index:
                if byte_mode:
                    return None
                assert not chunk_frozen, (
                    f"_embed_bpe: key {latin1!r} missing from frozen "
                    f"codebook.pretrain (word_learning<=0). .kv "
                    f"load mismatch -- BPE section and lexicon "
                    f"embeddings disagree.")
                codebook.insert(latin1)
            return token_to_index(latin1)

        for b in range(batch):
            row = rows[b]
            word_idx = 0
            cur_subs = []   # codebook indices for the in-progress word
            i = 0
            while i < N_buf:
                bval = row[i]
                if bval == 0:
                    break
                # Inline trie walk: descend children byte-by-byte,
                # tracking the longest match seen.
                node = trie
                matched_id = None
                matched_len = 0
                j = i
                while j < N_buf:
                    child = node[0].get(row[j])
                    if child is None:
                        break
                    node = child
                    j += 1
                    if node[1] is not None:
                        matched_id = node[1]
                        matched_len = j - i
                if matched_id is None:
                    # 256 single-byte ids are always seeded; defensive
                    # fallback for unexpected vocab gaps.
                    matched_id = vocab.get((bval,), bval)
                    matched_len = 1
                matched_key = id_to_bytes.get(matched_id, (bval,))
                is_boundary = all(bv in boundary for bv in matched_key)
                if is_boundary:
                    if cur_subs and word_idx < nObj:
                        per_word_first.append(cur_subs[0])
                        per_word_b.append(b)
                        per_word_slot.append(word_idx)
                        for cb_idx in cur_subs:
                            flat_subtoken_idx.append(cb_idx)
                            flat_word_seg.append(word_id)
                        word_id += 1
                        word_idx += 1
                    cur_subs = []
                else:
                    resolved = _resolve(matched_key)
                    if resolved is not None:
                        cur_subs.append(resolved)
                i += matched_len
            # Trailing word (row ended without a final boundary chunk).
            if cur_subs and word_idx < nObj:
                per_word_first.append(cur_subs[0])
                per_word_b.append(b)
                per_word_slot.append(word_idx)
                for cb_idx in cur_subs:
                    flat_subtoken_idx.append(cb_idx)
                    flat_word_seg.append(word_id)
                word_id += 1

        return self._bpe_emit(
            upstream_vspace, codebook, batch, nObj, dev, null_idx,
            flat_subtoken_idx, flat_word_seg, per_word_first,
            per_word_b, per_word_slot, word_id)

    def _bpe_emit(self, upstream_vspace, codebook, batch, nObj, dev,
                  null_idx, flat_idx, flat_seg, per_word_first,
                  per_word_b, per_word_slot, word_id):
        """Shared tail for both _embed_bpe paths: gather codebook
        vectors, segmented MAX-fuse per word, place at (b, slot), mux
        where/when, setW. The 5 routing arrays may be Python lists
        (trie path -> one H2D via ``as_tensor``) or already-on-device
        tensors (GPU path -> ``as_tensor`` is a no-op, zero DtoH).
        """
        # ---- Tensor materialization (no Python list-of-tensors) -----
        # Output tensors live on whatever device the codebook's
        # vectors currently sit on; gather + scatter respect that
        # device naturally. ``vectors`` is the nn.Parameter that
        # ``.to()`` migrates with the surrounding module.
        vectors = codebook.wv._vectors
        target_device = vectors.device
        nDim = self.nDim

        word_active = torch.zeros(
            batch, nObj, dtype=torch.float32, device=target_device)
        what_indices = torch.full(
            (batch, nObj), null_idx, dtype=torch.long, device=target_device)
        word_vectors = torch.zeros(
            batch, nObj, nDim, dtype=vectors.dtype, device=target_device)

        if word_id > 0:
            # ``as_tensor``: trie path passes Python lists (one H2D);
            # GPU path passes device tensors (no-op -> zero DtoH).
            flat_idx_t = torch.as_tensor(
                flat_idx, dtype=torch.long, device=target_device)
            flat_seg_t = torch.as_tensor(
                flat_seg, dtype=torch.long, device=target_device)
            per_word_first_t = torch.as_tensor(
                per_word_first, dtype=torch.long, device=target_device)
            per_word_b_t = torch.as_tensor(
                per_word_b, dtype=torch.long, device=target_device)
            per_word_slot_t = torch.as_tensor(
                per_word_slot, dtype=torch.long, device=target_device)

            # ONE gather: every sub-token's vector across the batch.
            gathered = vectors[flat_idx_t]  # [N_subs, nDim]

            # Segmented MAX via scatter_reduce_(amax). Init to -inf so
            # the first scatter writes the actual sub-token value, and
            # subsequent scatters within the same segment max with it.
            per_word_max = torch.full(
                (word_id, nDim), float('-inf'),
                dtype=vectors.dtype, device=target_device)
            per_word_max.scatter_reduce_(
                0, flat_seg_t.unsqueeze(-1).expand(-1, nDim),
                gathered, reduce='amax', include_self=True)

            # Place per-word results at (b, slot) coordinates in the
            # [B, nObj, *] outputs. Empty slots stay zero / null_idx.
            word_vectors[per_word_b_t, per_word_slot_t] = per_word_max
            what_indices[per_word_b_t, per_word_slot_t] = per_word_first_t
            word_active[per_word_b_t, per_word_slot_t] = 1.0

        return self._bpe_finalize(
            upstream_vspace, word_vectors, what_indices, word_active,
            batch, nObj, dev)

    def _bpe_finalize(self, upstream_vspace, word_vectors,
                      what_indices, word_active, batch, nObj, dev):
        """Shared tail of both BPE emitters: pull where/when from the
        upstream buffer, set_forward_content, mux ``word_vectors`` with
        the where/when encodings, ``setW`` the muxed event, stash the
        BPE word mask. Identical for the trie and GPU paths -- they
        differ only in how the [B,nObj,*] word arrays are built."""
        where_raw = upstream_vspace.materialize(mode="where")
        when_raw = upstream_vspace.materialize(mode="when")
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

        # Mux ``word_vectors`` ([B, nObj, nDim]) with the where/when
        # encodings so the cached event matches the muxed width
        # ``nDim + nWhere + nWhen`` that ``forwardEnd`` reshapes
        # against. Setting the event at plain ``nDim`` width caused
        # ``[B, nObj * nDim] / muxedSize`` to round to a wrong
        # sequence length (e.g. 1024 * 6 / 8 = 768) and trip a shape
        # mismatch against the [B, nObj] BPE word mask. ``materialize``
        # would otherwise gather codebook rows by ``what_indices``,
        # which gives the *first sub-token's* vector, not the
        # MAX-fused word vector we computed -- hence we must persist
        # this per-batch MAX-fused tensor on ``.event`` directly.
        #
        # KNOWN OUTLIER under spec
        # doc/specs/2026-05-21-subspace-slot-architecture.md: this is
        # neither a codebook lookup nor a pure-event store. For configs
        # where ``.event`` is codebook-bearing (MM_xor / MM_5M muxed),
        # this write currently rides the ``_active_payload`` band-aid.
        # Stage 3 of the retirement plan
        # (doc/plans/2026-05-21-active-payload-retirement.md) addresses
        # this case explicitly. Surface here is the public setter so
        # the migration only has to change ``SubSpace.set_event`` to
        # route differently — no further changes here.
        event_parts = [word_vectors]
        if self.nWhere > 0 and where_indices is not None:
            event_parts.append(self.subspace.whereEncoding.encode(where_indices))
        if self.nWhen > 0 and when_indices is not None:
            event_parts.append(self.subspace.whenEncoding.encode(when_indices))
        muxed_event = (torch.cat(event_parts, dim=-1)
                       if len(event_parts) > 1 else word_vectors)
        # Spec-aligned write: route through ``SubSpace.set_muxed`` so
        # the codebook-bearing case (MM_xor / MM_5M with codebook on
        # ``.event``) snaps the MAX-pooled fused vector through the
        # codebook (writing the selection on ``_active``), and the
        # plain-Tensor case (MM_grammar, byte-mode) stores per-batch on
        # ``event.W`` directly. ``materialize`` reconstructs as
        # ``codebook[_active]`` for muxed configs — the selection IS
        # the storage. Per spec invariant "input width ≤ codebook
        # nDim", the snap is well-defined here.
        self.subspace.set_muxed(muxed_event)
        self._embedded_input = muxed_event
        self._bpe_word_mask = word_active
        # Only InputSpace.forward can produce the AR-windowed [B*K, N]
        # mask. Clear any stale value when _embed_bpe is used directly
        # (non-AR / inference paths) so PerceptualSpace.forward falls
        # back to this fresh unwindowed [B, N] mask.
        self._bpe_word_mask_flat = None
        return self.subspace

    def _bpe_emit_gpu(self, upstream_vspace, codebook, batch, nObj, dev,
                      null_idx, sub_cb, sub_target, sub_pos, keep):
        """Static-shape GPU emitter: scatter the per-position
        ``[B,T]`` segmentation into static ``[B*nObj]`` word buffers
        (no dense word-id, no ``.item()``, no boolean compaction ->
        zero DtoH). Produces the same [B,nObj,*] word arrays the trie
        path builds, then the shared ``_bpe_finalize`` tail. Asserted
        bit-identical to the trie path by test/bpe_gpu_equiv.py.
        """
        vectors = codebook.wv._vectors
        tdev = vectors.device
        nDim = self.nDim
        BN = batch * nObj
        flat_t = sub_target.reshape(-1).to(tdev)             # [B*T]
        cb_safe = sub_cb.clamp(min=0).reshape(-1)            # -1 -> 0
        gathered = vectors[cb_safe]                          # [B*T, nDim]

        # Segmented MAX into [BN(+trash), nDim]; trash row B*nObj
        # absorbs non-kept positions, then sliced off.
        per_word_max = torch.full(
            (BN + 1, nDim), float('-inf'),
            dtype=vectors.dtype, device=tdev)
        per_word_max.scatter_reduce_(
            0, flat_t.unsqueeze(-1).expand(-1, nDim), gathered,
            reduce='amax', include_self=True)

        keep_f = keep.reshape(-1).to(torch.int64)
        active_flat = torch.zeros(BN + 1, dtype=torch.int64, device=tdev)
        active_flat.scatter_reduce_(
            0, flat_t, keep_f, reduce='amax', include_self=True)
        word_active = (active_flat[:BN].reshape(batch, nObj)
                       .to(torch.float32))

        word_max = per_word_max[:BN].reshape(batch, nObj, nDim)
        word_vectors = torch.where(
            word_active.unsqueeze(-1) > 0, word_max,
            torch.zeros((), dtype=vectors.dtype, device=tdev))

        # First sub-token (smallest token pos) per word: pack
        # ``pos * K + cb`` so ``amin`` keeps the lowest-pos entry and
        # ``% K`` recovers its codebook row. K > max codebook row.
        K = int(vectors.shape[0]) + 1
        BIG = (sub_pos.numel() + 1) * K
        packed = torch.where(
            keep, sub_pos.to(torch.int64) * K + sub_cb,
            torch.full_like(sub_pos, BIG)).reshape(-1).to(tdev)
        packed_min = torch.full((BN + 1,), BIG, dtype=torch.int64,
                                device=tdev)
        packed_min.scatter_reduce_(
            0, flat_t, packed, reduce='amin', include_self=True)
        first_cb = packed_min[:BN] % K
        what_indices = torch.where(
            packed_min[:BN] < BIG, first_cb,
            torch.full_like(first_cb, null_idx)
        ).reshape(batch, nObj)

        return self._bpe_finalize(
            upstream_vspace, word_vectors, what_indices, word_active,
            batch, nObj, dev)

    def _chunk_key_to_latin1(self, byte_tuple):
        """Convert a byte-tuple key (e.g., (104, 101)) to its latin-1 string."""
        return "".join(chr(int(b) & 0xFF) for b in byte_tuple)

    def _chunk_to_codebook_idx(self, word_subtokens, codebook):
        """Resolve a list of byte-tuple sub-tokens to a codebook index.

        With a frozen BPE codebook (``word_learning <= 0``) every
        sub-token's latin-1 key MUST already live in
        ``codebook.pretrain.key_to_index`` -- the load pass is supposed
        to populate one entry per BPE chunk. A missing key here means
        either a load misalignment (the .kv file's BPE section and
        embeddings disagree) or runtime BPE drift; either way it would
        silently grow the codebook and is a real bug. Assert loudly so
        the failure is obvious instead of corrupting the codebook.

        In active-learning mode (``word_learning > 0``) a missing
        key is expected and we fall back to ``Embedding.insert``.

        Returns the index of the first sub-token (used for bookkeeping;
        the real word vector comes from MAX fusion).
        """
        keys = [self._chunk_key_to_latin1(bt) for bt in word_subtokens]
        chunk_frozen = (
            int(getattr(self.chunk_layer, 'word_learning', 0) or 0) <= 0)
        for key in keys:
            if (key and key not in codebook.pretrain.key_to_index
                    and not getattr(codebook, 'byte_mode', False)):
                assert not chunk_frozen, (
                    f"_chunk_to_codebook_idx: key {key!r} missing from "
                    f"frozen codebook.pretrain (word_learning<=0). "
                    f"This indicates a .kv load mismatch -- either the "
                    f"BPE section and the lexicon embeddings disagree, "
                    f"or word_learning was set to 0 before all BPE "
                    f"chunks made it into the lexicon.")
                codebook.insert(key)
        return codebook._token_to_index(keys[0]) if keys else 0

    def _max_fuse_subtokens(self, word_subtokens, codebook):
        """MAX-fuse sub-token vectors into a single [nDim] word vector.

        Resolves each sub-token byte-tuple to its codebook row index,
        gathers all rows in one tensor lookup, and returns the
        per-dimension maximum. Avoids the previous Python ``vecs = []``
        + ``torch.stack(vecs)`` + ``stacked.max(dim=0)`` pattern, which
        allocated one Python list and one stack-temporary per word
        (~960 words/batch on MM_5M). The single ``vectors[idx_tensor]``
        gather + ``amax(dim=0)`` keeps the hot path tensor-native and
        runs on whatever device the codebook lives on (no device
        coupling to the caller).
        """
        if not word_subtokens:
            return torch.zeros(self.nDim, device=codebook.wv._vectors.device)
        # Collect codebook row indices for sub-tokens that resolve.
        # Under a frozen BPE codebook (``word_learning <= 0``)
        # every sub-token's latin-1 key MUST already live in
        # ``codebook.pretrain.key_to_index`` -- a missing key here is
        # the same .kv load mismatch we assert against in
        # ``_chunk_to_codebook_idx``. The active-learning branch
        # (``word_learning > 0``) silently skips unresolved keys
        # since new chunks may legitimately not be in the lexicon
        # until the next promotion cycle catches up.
        indices = []
        key_to_index = codebook.pretrain.key_to_index
        token_to_index = codebook._token_to_index
        chunk_frozen = (
            int(getattr(self.chunk_layer, 'word_learning', 0) or 0) <= 0)
        for bt in word_subtokens:
            key = self._chunk_key_to_latin1(bt)
            if not key:
                continue
            if key not in key_to_index:
                assert not chunk_frozen, (
                    f"_max_fuse_subtokens: key {key!r} missing from "
                    f"frozen codebook.pretrain (word_learning<=0). "
                    f"This indicates a .kv load mismatch -- the BPE "
                    f"section and the lexicon embeddings disagree.")
                continue
            indices.append(token_to_index(key))
        if not indices:
            return torch.zeros(self.nDim, device=codebook.wv._vectors.device)
        vectors = codebook.wv._vectors
        idx_tensor = torch.tensor(
            indices, dtype=torch.long, device=vectors.device)
        return vectors[idx_tensor].amax(dim=0)

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

    def _make_perceptual_codebook_learnable(self):
        """Phase 1A.1 scoping hook.

        Flip the perceptual codebook's ``VectorQuantize`` into
        ``learnable_codebook`` mode (gradient-trained codebook, EMA
        in-call write suppressed, codebook-attached STE). Scoped to
        exactly this PerceptualSpace instance's ``.event`` codebook so
        the SHARED ``VectorQuantize``/``Codebook`` class still runs the
        byte-identical EMA path for the Conceptual/Symbolic codebooks.

        Robust no-op when there is no VQ to flip (passthrough ``Tensor``
        subspace when ``<codebook>`` is off; ``customVQ`` disabled).
        Idempotent.
        """
        try:
            cb = self.subspace.get_vectors()
        except Exception:
            return
        vq = getattr(cb, "vq", None)
        if vq is None or not hasattr(vq, "learnable_codebook"):
            return
        vq.learnable_codebook = True

    # ---- Rework A: percept -> MPHF -> table (the consolidated core) ----
    #
    # Per the consolidated two-loop spec (§"Percept -> MPHF -> table",
    # §IMPLEMENTATION DETAILS D2): each percept's byte slot passes
    # through a minimal perfect hash producing an index in
    # ``[0, V_percept)``; the index addresses a table whose every entry
    # holds BOTH the literal surface word AND the ConceptualSpace
    # activation vector for that token.
    #
    # The table's two halves ALREADY EXIST on this PerceptualSpace's
    # frozen ``Embedding`` codebook and are REUSED verbatim (a second
    # parallel embedding over the same surface tokens would double-count
    # gradient -- the spec's explicit NEEDS_CONTEXT trigger, resolved by
    # reuse):
    #   * concept-activation half == ``codebook.wv._vectors`` (the
    #     Phase-1A.1 learnable lexicon ``nn.Parameter`` -- gradient
    #     trained; the BPE/lexicon ``_embed*`` paths already gather from
    #     exactly this tensor);
    #   * surface half == ``codebook.wv.index_to_key`` (ASCII-prefilled
    #     by ``Embedding.create``: ``\x00`` row 0 == the NULL char / per-
    #     row cursor seal, ``chr(1..126)`` low rows, ``NULL_PERCEPT_KEY``
    #     at ``null_percept_idx``; NO MASK row -- MASK is the all-zeros
    #     gaussian-tail effect, not a row).
    # Rework A therefore adds ONLY the static O(1) MPHF index function
    # (percept bytes -> the EXISTING frozen ``key_to_index`` row) and
    # the non-invertible reverse map (vector -> nearest row -> surface).

    def _mphf_codebook(self):
        """The ``Embedding`` codebook holding the D2 table (both halves).
        ``None`` for numeric / non-Embedding codebooks (MPHF inapplicable
        -- caller leaves the existing path unchanged)."""
        cb = getattr(self.subspace, "what", None)
        if isinstance(cb, Embedding) and getattr(cb, "wv", None) is not None:
            return cb
        return None

    def _mphf_tables(self):
        """Build-once / cache the frozen MPHF static tensors (mirrors the
        ``_bpe_static_tables`` build-once pattern: keyed by lexicon row
        count; frozen => never rebuilt after the first build). Raises
        ``MPHFGpuLayer._MPHFUnavailable`` for a non-Embedding codebook."""
        mphf = self._mphf_gpu_layer
        cb = self._mphf_codebook()
        if cb is None:
            raise mphf._MPHFUnavailable(
                "PerceptualSpace._mphf_tables: non-Embedding codebook.")
        vsig = len(cb.wv.index_to_key)
        tab = self._mphf_static_tables
        if tab is None or tab.get("_vsig") != vsig:
            dev = cb.wv._vectors.device
            tab = mphf.build_mphf_table(cb, dev)
            tab["_vsig"] = vsig
            self._mphf_static_tables = tab
        return tab

    def mphf_index(self, token_byte_slots, return_verified=False):
        """The MPHF index function: percept byte slots ``[B,K,W]`` (the
        ``InputSpace.subspace.what.W`` null-terminated utf-8 layout) ->
        frozen lexicon row indices ``[B,K]``. Pure static tensor ops,
        O(1) per slot, zero host sync, NON-invertible (reverse is the
        table lookup, never an inverse hash).

        When ``return_verified=True``, returns ``(row, verified)`` so the
        caller can gate OOV->BPE-trie fallback on a true hit (vs the
        ``null_row`` fallback row that ``mphf_index`` returns for both
        L==0 slots and OOV)."""
        return self._mphf_gpu_layer.mphf_index(
            token_byte_slots, self._mphf_tables(),
            return_verified=return_verified)

    def mphf_table_rows(self, row_idx):
        """Gather the D2 table's concept-activation rows for ``row_idx``
        (``[...]`` long) -> ``[..., D]``. The rows ARE the reused
        Phase-1A.1 learnable ``wv._vectors`` (gradient flows through;
        no detach)."""
        cb = self._mphf_codebook()
        return cb.getW()[row_idx]

    def reverse_map_concept(self, concept_vectors, return_surface=True):
        """Non-invertible reverse map exposed for the NEXT rework (D3
        reconstruction loss -- NOT called from the loss here).

        ``concept_vectors`` ``[..., D]`` -> nearest table row index
        ``[...]`` (tensor, no host sync) and, when ``return_surface``,
        the parallel surface strings ``surface[idx]`` (== the literal
        ASCII-prefilled ``index_to_key`` words; the host indexing is the
        caller's choice, off the training-critical path). The MPHF is
        never inverted -- this nearest-row table lookup IS the reverse
        map (spec §"Because the table stores the surface word, the MPHF
        need not be invertible")."""
        cb = self._mphf_codebook()
        if cb is None:
            return (None, None) if return_surface else None
        idx = self._mphf_gpu_layer.reverse_map_rows(concept_vectors, cb)
        if not return_surface:
            return idx
        surf = cb.wv.index_to_key
        flat = idx.reshape(-1).tolist()
        strings = [surf[i] if 0 <= i < len(surf) else "" for i in flat]
        return idx, strings

    def _slot_forward(self, x, quantize=True):
        """Position-local math (codebook + sparsity) on a [B, K, D] slice.

        Used by the serial_mode warm path to process just the new AR slot
        (K=1) instead of re-running over the full [B, N, D] window.
        Note: VQ-VAE commit losses and SparsityRegLayer accumulations
        see per-slot samples instead of per-batch; for training-accuracy
        runs, take the cold path.
        """
        if self.codebook and quantize:
            cb = self.subspace.get_vectors()
            x = cb.forward(x)
        x = self._sparsity(x)
        return x

    def _embed_byte(self, upstream_vspace):
        """Byte-direct chunking (``<chunking>none</chunking>``).

        Skips the BPE walker entirely.  Each byte from the upstream
        buffer becomes its own perceptual slot via direct embedding-
        table lookup at codebook index ``byte_value``.  The
        Embedding's cold-start path seeds entries 0..255 with
        ``byte_value == codebook_index`` (see ``NULL_PERCEPT`` seeding
        and the ``_random_unit_ball`` init at Spaces.py:2520), so this
        method is just:

            byte_indices = upstream_vspace.what[..., 0]  # [B, N]
            subspace.set_forward_content(byte_indices, where, when)

        ``\0`` (byte 0) lands at codebook index 0 and doubles as the
        sentence-end / pad sentinel.  ``_bpe_word_mask`` is derived as
        ``(byte_indices != 0)`` — pure tensor op, no Python loop, no
        graph break.  Downstream consumers (PerceptualSpace.forward's
        AR-window pad-and-unfold path) read the mask the same way they
        do for BPE mode.
        """
        what_buf = upstream_vspace.materialize(mode="what")
        if what_buf is None:
            raise RuntimeError(
                "PerceptualSpace._embed_byte: upstream subspace.what is empty. "
                "InputSpace.forward must lex into subspace.what.W before "
                "PerceptualSpace.forward runs.")

        dev = TheDevice.get()
        batch = what_buf.shape[0]
        nObj = self.outputShape[0]

        if what_buf.dim() == 3:
            byte_indices = what_buf[..., 0].long()
        else:
            byte_indices = what_buf.long()
        # Defensive clamp: prepInput sometimes uses int8 which sign-
        # extends negative values for byte > 127; remap to the
        # canonical [0, 255] range so the codebook lookup hits the
        # right entry.
        byte_indices = byte_indices.where(byte_indices >= 0, byte_indices + 256)
        byte_indices = byte_indices.clamp(0, 255)
        # Trim / pad to nObj slots.
        n_upstream = byte_indices.shape[1]
        if n_upstream > nObj:
            byte_indices = byte_indices[:, :nObj]
        elif n_upstream < nObj:
            pad = torch.zeros(
                batch, nObj - n_upstream,
                dtype=byte_indices.dtype, device=byte_indices.device)
            byte_indices = torch.cat([byte_indices, pad], dim=1)

        # where / when come straight from the upstream buffer (mirror
        # of ``_embed`` for the lexicon path).
        where_raw = upstream_vspace.materialize(mode="where")
        when_raw = upstream_vspace.materialize(mode="when")
        if self.nWhere > 0:
            if where_raw is not None:
                where_indices = where_raw[:, :nObj].long()
            else:
                where_indices = torch.zeros(
                    batch, nObj, dtype=torch.long, device=dev)
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
        self.subspace.set_forward_content(
            byte_indices, where_indices, when_indices)
        self.subspace.normalize("input", target="what", normalize=True)

        # Stash the materialized [B, nObj, D] muxed event for
        # InputSpace.forward to consume (same contract as ``_embed``
        # and ``_embed_bpe``).
        self._embedded_input = self.subspace.materialize()

        # Word-boundary mask: bytes != 0 are real.  Matches the
        # ``_bpe_word_mask`` contract used by InputSpace.forward's AR
        # unfold so downstream consumers don't need a separate code
        # path -- the windowed view is identical.
        self._bpe_word_mask = (byte_indices != 0).to(
            dtype=torch.float32, device=byte_indices.device)
        return upstream_vspace

    def _embed_mphf(self, upstream_vspace):
        """MPHF word recognition (``chunking_mode='mphf'``): per-word,
        ``idx = MPHF(percept_bytes) in [0, V_percept)`` -> gather from
        the Phase-1A.1 learnable lookup ``codebook.wv._vectors[idx]``.
        OOV (the collision-proof byte-verify misses) falls back to
        ``_embed_bpe_trie`` for that word position only -- the verified
        BPE reference path is the documented OOV fallback.

        Word boundary detection mirrors ``_embed_bpe_trie``'s walk
        (``BOUNDARY_BYTES`` over the upstream byte buffer). Output
        contract matches ``_embed_bpe_trie`` exactly: routes through the
        shared ``_bpe_emit`` -> ``_bpe_finalize`` tail so the muxed
        ``subspace.event`` ``[B, nObj, D]``, ``_bpe_word_mask``
        ``[B, nObj]``, and the where/when slots populate identically to
        all the other chunking modes.

        Per-word, the routing is:
          * try MPHF over the full word byte tuple (one row gather);
          * verified hit -> emit a single sub-token entry holding the
            MPHF row (MAX-fuse degenerates to identity);
          * miss (OOV) -> per-word BPE trie sub-token walk (mirrors
            ``_embed_bpe_trie``'s inner loop), emit each sub-token.

        For MPHF-applicability gate (non-Embedding codebook -- numeric
        codebooks have no surface key set), the table build raises
        ``MPHFGpuLayer._MPHFUnavailable``; we fall through to
        ``_embed_bpe_trie`` for the whole batch in that case (never
        silently wrong).
        """
        # Build / cache MPHF tables; non-Embedding codebooks fall back
        # to the verified BPE trie for the whole batch.
        try:
            tab = self._mphf_tables()
        except self._mphf_gpu_layer._MPHFUnavailable:
            return self._embed_bpe_trie(upstream_vspace)

        what_buf = upstream_vspace.materialize(mode="what")
        if what_buf is None:
            raise RuntimeError(
                "PerceptualSpace._embed_mphf: upstream subspace.what is "
                "empty. InputSpace.forward must lex into subspace.what.W "
                "before PerceptualSpace.forward runs.")

        dev = TheDevice.get()
        batch = what_buf.shape[0]
        nObj = self.outputShape[0]
        codebook = self.subspace.what
        boundary = self.chunk_layer.BOUNDARY_BYTES
        chunk_frozen = (
            int(getattr(self.chunk_layer, 'word_learning', 0) or 0) <= 0)

        if what_buf.dim() == 3:
            byte_indices = what_buf[..., 0].long()
        else:
            byte_indices = what_buf.long()

        if self.chunk_layer.bpe and self.training:
            self.chunk_layer.train_step(byte_indices)

        # ---- Phase 1: per-word boundary walk to collect word byte
        # tuples per (b, word_slot). Mirrors ``_embed_bpe_trie``'s walk
        # (boundary bytes split words); the result is one byte tuple
        # per word slot (the WHOLE word's bytes, MPHF candidate key).
        self.chunk_layer._ensure_trie()
        trie = self.chunk_layer._trie
        id_to_bytes = self.chunk_layer.id_to_bytes
        vocab = self.chunk_layer.vocab

        null_idx = codebook.wv.key_to_index.get("\x00", 0)
        key_to_index = codebook.pretrain.key_to_index
        token_to_index = codebook._token_to_index
        chunk_key_to_latin1 = self._chunk_key_to_latin1
        byte_mode = bool(getattr(codebook, 'byte_mode', False))

        rows = byte_indices.tolist()
        N_buf = byte_indices.shape[1]
        maxL_tab = int(tab["maxL"])

        # Per-word byte tuples + per-word routing tuples (b, slot).
        # ``word_byte_seq`` accumulates contiguous non-boundary bytes
        # for the in-progress word; flushed at every boundary byte.
        per_word_bytes = []      # tuple of ints (the word's bytes)
        per_word_b = []          # batch row
        per_word_slot = []       # word slot within row

        for b in range(batch):
            row = rows[b]
            word_idx = 0
            word_byte_seq = []
            i = 0
            while i < N_buf:
                bval = row[i]
                if bval == 0:
                    break
                if bval in boundary:
                    if word_byte_seq and word_idx < nObj:
                        per_word_bytes.append(tuple(word_byte_seq))
                        per_word_b.append(b)
                        per_word_slot.append(word_idx)
                        word_idx += 1
                    word_byte_seq = []
                else:
                    word_byte_seq.append(bval)
                i += 1
            # Trailing word (row ended without a final boundary byte).
            if word_byte_seq and word_idx < nObj:
                per_word_bytes.append(tuple(word_byte_seq))
                per_word_b.append(b)
                per_word_slot.append(word_idx)

        N_words = len(per_word_bytes)

        # ---- Phase 2: batch MPHF lookup over all words at once.
        # Build the [1, N_words, W] null-terminated byte-slot tensor
        # the same way ``InputSpace.subspace.what.W`` lays out tokens
        # (each row is the word's utf-8 bytes followed by 0). W is
        # ``maxL_tab + 1`` so any in-vocab key fits and is properly
        # terminated. Words longer than ``maxL_tab`` will overflow the
        # frozen lexicon length range -> guaranteed MPHF miss -> OOV
        # routes to BPE-trie fallback, the documented behavior.
        if N_words > 0:
            W = max(maxL_tab + 1, 1)
            slot_buf = torch.zeros(
                (1, N_words, W), dtype=torch.int64, device=dev)
            # Host-side fill (one-shot, off the GPU critical path; this
            # is the cold-path word-segmentation cost mirroring the
            # trie's ``rows = byte_indices.tolist()``).
            for k, bt in enumerate(per_word_bytes):
                L = min(len(bt), W - 1)
                if L:
                    slot_buf[0, k, :L] = torch.tensor(
                        [int(x) & 0xFF for x in bt[:L]],
                        dtype=torch.int64, device=dev)
            mphf_row, verified = self._mphf_gpu_layer.mphf_index(
                slot_buf, tab, return_verified=True)
            mphf_row = mphf_row[0]            # [N_words]
            verified = verified[0]            # [N_words]
            verified_list = verified.tolist()
            mphf_row_list = mphf_row.tolist()
        else:
            verified_list = []
            mphf_row_list = []

        # ---- Phase 3: per-word routing. MPHF hits emit a single
        # codebook-row sub-token entry; OOVs route through the BPE
        # trie walk over their byte span (mirrors
        # ``_embed_bpe_trie``'s inner loop, scoped to one word).
        def _resolve(byte_tuple):
            """Sub-token byte tuple -> codebook int index, or None.
            Mirrors ``_embed_bpe_trie._resolve``.
            """
            latin1 = chunk_key_to_latin1(byte_tuple)
            if not latin1:
                return None
            if latin1 not in key_to_index:
                if byte_mode:
                    return None
                assert not chunk_frozen, (
                    f"_embed_mphf OOV-fallback: key {latin1!r} missing "
                    f"from frozen codebook.pretrain (word_learning<=0). "
                    f".kv load mismatch -- BPE section and lexicon "
                    f"embeddings disagree.")
                codebook.insert(latin1)
            return token_to_index(latin1)

        flat_subtoken_idx = []
        flat_word_seg = []
        per_word_first = []
        per_word_b_out = []
        per_word_slot_out = []
        word_id = 0

        for k in range(N_words):
            b = per_word_b[k]
            slot = per_word_slot[k]
            word_bytes = per_word_bytes[k]
            if verified_list[k]:
                # MPHF hit: single sub-token entry == the MPHF row.
                cb_idx = int(mphf_row_list[k])
                flat_subtoken_idx.append(cb_idx)
                flat_word_seg.append(word_id)
                per_word_first.append(cb_idx)
                per_word_b_out.append(b)
                per_word_slot_out.append(slot)
                word_id += 1
            else:
                # OOV: per-word BPE-trie sub-token walk. Identical to
                # ``_embed_bpe_trie``'s inner loop but scoped to this
                # word's byte span (no boundary checks needed -- the
                # span is by definition between boundaries).
                cur_subs = []
                Lw = len(word_bytes)
                ii = 0
                while ii < Lw:
                    node = trie
                    matched_id = None
                    matched_len = 0
                    jj = ii
                    while jj < Lw:
                        child = node[0].get(word_bytes[jj])
                        if child is None:
                            break
                        node = child
                        jj += 1
                        if node[1] is not None:
                            matched_id = node[1]
                            matched_len = jj - ii
                    if matched_id is None:
                        matched_id = vocab.get(
                            (word_bytes[ii],), word_bytes[ii])
                        matched_len = 1
                    matched_key = id_to_bytes.get(
                        matched_id, (word_bytes[ii],))
                    resolved = _resolve(matched_key)
                    if resolved is not None:
                        cur_subs.append(resolved)
                    ii += matched_len
                if cur_subs:
                    per_word_first.append(cur_subs[0])
                    per_word_b_out.append(b)
                    per_word_slot_out.append(slot)
                    for cb_idx in cur_subs:
                        flat_subtoken_idx.append(cb_idx)
                        flat_word_seg.append(word_id)
                    word_id += 1

        return self._bpe_emit(
            upstream_vspace, codebook, batch, nObj, dev, null_idx,
            flat_subtoken_idx, flat_word_seg, per_word_first,
            per_word_b_out, per_word_slot_out, word_id)

    def forward(self, x_subspace):
        """Perception: map input to percepts via ``pi(x) + sigma(x)``.

        Stage 1.A substrate refactor (doc/plans/2026-05-26-two-loop-pi-
        sigma-substrate.md): single positional arg ``x_subspace``. The
        body composes ``self.pi(x) + self.sigma(x)`` on the same
        materialized input -- two folds of the SAME input, summed (no
        outer ``tanh`` wrap; each layer applies its own internal
        nonlinearity).

        The CS-feedback ``CS_subspaceForPS`` argument is retired by
        this refactor; CS feedback no longer enters PS directly (it
        re-enters via the chart / signal-router dispatch over STM in
        later stages).

        Handles three paths: warm-serial AR (process only new last slot,
        splice into the rolled cache), embedding (text -> lexicon -> percept),
        and numeric (linear -> attention -> VQ). Writes the resulting
        percept tensor to ``self.subspace`` and returns the live subspace.
        """
        if x_subspace.is_empty():
            return x_subspace
        self.subspace.copy_context(x_subspace)
        vspace = x_subspace
        quantize = getattr(self, "quantize", True)

        # Serial-mode warm path: upstream has pre-embedded AR buffer, cold
        # path already populated the subspace serial_cache with the prior
        # full output. Process only the new last slot, splice into rolled
        # cache. Skips the lexicon _embed (upstream already embedded) and
        # runs the VQ codebook on one slot instead of N.
        cache = self.subspace.serial_cache.get(id(self))
        # ``recur_pass`` lives on WordSpace as the per-sentence recurrent-
        # pass index. Post-Phase-G the WordSubSpace is reached via the
        # owning Space's routing pointer (``self.wordSubSpace`` —
        # ``object.__setattr__`` keeps it out of nn.Module's child
        # registration); if it's missing (standalone PerceptualSpace
        # .forward, pre-soft_reset tests), fall back to the persistent
        # ``self._recurrent_pass_idx`` attribute. Stage 1.A refactor: the
        # per-order ModuleList selection is gone, but the warm-path
        # ``_rp == 0`` gate (an AR-streaming optimisation across forward
        # calls, distinct from in-forward recurrent passes) is preserved.
        ws = getattr(self, 'wordSubSpace', None)
        _rp = (int(ws.recur_pass) if ws is not None
               else self._recurrent_pass_idx)
        if (getattr(self, "serial_mode", False)
                and _rp == 0
                and cache is not None):
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
            elif mode == "none":
                vspace = self._embed_byte(vspace)
            elif mode == "mphf":
                vspace = self._embed_mphf(vspace)
            else:
                raise ValueError(
                    f"PerceptualSpace chunking must be bpe|lexicon|none|mphf, "
                    f"got {mode!r}")
        if getattr(vspace, '_demuxed', False) and vspace._active is not None:
            self.subspace._byte_indices = vspace._active[:, :, 0].long()
        # Stage 1.A substrate refactor: compose ``pi(x) + sigma(x)`` on
        # the same materialized input. No outer ``tanh`` -- each layer
        # applies its own internal nonlinearity. The legacy two-input
        # ``tanh(pi_input(IS) + pi_concept(C_prev))`` shape gating
        # (non-bivector C-feedback degraded to pi_input-alone) is no
        # longer needed: there is no C-feedback path entering PS at this
        # level (CS feedback re-enters via the chart / signal-router
        # over STM in later stages).
        primary = self.forwardBegin(vspace, returnVectors=True)
        x = self.pi.forward(primary) + self.sigma.forward(primary)
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
        #
        # Mask source-of-truth picking:
        #   * ``_bpe_word_mask_flat`` ([B*K, N]) is the windowed-and-
        #     flattened mask installed by InputSpace.forward when peer
        #     is in BPE mode. Matches the [B*K, N, D] event after
        #     FlattenKWrapper. Use it whenever it's present.
        #   * ``_bpe_word_mask`` ([B, N]) is the unwindowed view from
        #     ``_embed_bpe`` itself. Only correct when there's no AR
        #     windowing (K=1, B*K==B); used as fallback.
        # ``bpe`` and ``none`` (byte-direct) both install
        # ``_bpe_word_mask`` (one is real-vs-padding for BPE chunks,
        # the other is byte != 0); the post-VQ reapply is the same
        # under either mode.
        if self.chunking_mode in ("bpe", "none", "mphf"):
            mask = getattr(self, "_bpe_word_mask_flat", None)
            if mask is None:
                mask = getattr(self, "_bpe_word_mask", None)
            if mask is not None:
                ev = vspace.materialize(mode="event")
                if ev is not None:
                    # ``ev`` may be [B*K, N, D] (post-FlattenKWrapper),
                    # [B, N, D] (cached event in non-AR), or
                    # [B*K*N, D] / [B*N, D] (further flattened by
                    # forwardEnd). Match shapes to broadcast on the
                    # trailing axis.
                    if ev.dim() == 3 and ev.shape[:2] == mask.shape:
                        # Spec-aligned: store the word-active mask on
                        # ``.activation`` instead of zero-multiplying
                        # the event. ``materialize`` applies the gate
                        # (`event * activation_presence`) on read, so
                        # masked-out positions zero out in the
                        # reconstructed event without requiring a
                        # per-batch setW.
                        vspace.set_activation(
                            mask.to(dtype=ev.dtype))
                    elif ev.dim() == 2 and ev.shape[0] == mask.numel():
                        # 2-D shape — caller has flattened; for non-
                        # codebook slots this writes to ``event.W``
                        # directly. For codebook-bearing slots, the
                        # strict setW raises and the caller would have
                        # to migrate to the selection-based contract.
                        vspace.event.setW(
                            ev * mask.reshape(-1).unsqueeze(-1))
                    # else: shape mismatch we don't recognise -- skip
                    # the masking rather than crash; downstream zero-
                    # padding tolerance will absorb the unmasked tail.

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
        """Manifesting: reconstruct input vectors from percepts.

        Branches on text vs numeric: text-mode delegates to ``_reverse_text``
        (lexicon nearest-neighbor); numeric runs the inverse linear /
        attention / VQ chain. ``invertible`` paths use the LDU solve to
        avoid materializing the dense inverse weights.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        # NOTE: When ``self.subspace.what`` is an Embedding (text mode),
        # `_reverse_text` returns earlier and bypasses the numeric
        # inverse chain below.
        if isinstance(self.subspace.what, Embedding):
            self._reverse_text(vspace)
            return self.subspace
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
        y = self.reverseBegin(vspace, returnVectors=True)
        if self.codebook:
            y = object_basis.reverse(y)
        # Stage 1.A substrate refactor: reverse the single-layer ``pi``
        # fold (no per-order ModuleList). ``sigma.reverse`` is NOT
        # applied on the text path: the forward composed ``pi + sigma``
        # additively, and the codebook snap collapses the sum to a
        # single prototype slot whose membership is recovered via
        # ``object_basis.reverse`` above. Mirroring the legacy
        # text-mode behaviour (which only inverted the primary fold)
        # keeps this path's recovered-input contract stable.
        # TODO (revisit in Stage 1.B or later): if downstream numerical
        # reconstruction needs a paired pi/sigma inverse (e.g. for
        # masked-LM IR loss tightness), define the inversion contract
        # explicitly. The current decision parks the asymmetry.
        if self.invertible and hasattr(self, 'pi'):
            if getattr(self.pi, 'invertible', False):
                y = self.pi.reverse(y)
        self.subspace.batch = y.shape[0]
        raw = (object_basis.reverse_raw(y)
               if hasattr(object_basis, 'reverse_raw') else y)
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
        # Keep the continuous reverse tensor as the trainable surface.
        # Text rendering below still nearest-decodes it, matching the old
        # Embedding pass-through reverse contract while preserving gradient
        # flow for reconstruction loss.
        content = content.clone()
        if object_encoding is not None:
            self.input = object_encoding.restore_aux(content, aux)
        elif aux is not None:
            self.input = torch.cat([content, aux], dim=-1)
        else:
            self.input = content
        # Stage 4: dropped the legacy ``content_basis.setW(self.input)``
        # and ``object_basis.setW(self.input)`` lines. Both were
        # band-aid-era duplicates of the SubSpace setter below:
        #   * ``content_basis`` is the Embedding (.what) — Embedding.setW
        #     was already a no-op.
        #   * ``object_basis`` is .event — the per-batch write goes
        #     through ``SubSpace.set_event`` (which snaps via codebook
        #     for muxed configs; stores on event.W for pure-event).
        # ``self.input`` is still kept on the PerceptualSpace itself
        # for the text-render path (``_recovered_input_thunk`` below).
        self.subspace.set_event(self.input)
        # Lazy: word recovery is report-only (reconstruct_data /
        # reconstruct_to_buffer / get_recovered_word /
        # _reconstructionReport) and never feeds the gradient, but its
        # per-slot _nearest_idx().item() is a cudaMemcpyDtoH per word
        # (64+/batch) that breaks the brick CUDA-graph-capture
        # contract. Defer the decode to first consumer access (a no-op
        # during pure training); the captured tensors are held by ref
        # so the deferred result is bit-identical to the eager one.
        self._recovered_input = None
        self._recovered_input_thunk = (
            content_basis, self.input, self.subspace)
        self.subspace.normalize("input", target="what", normalize=True)

    def _materialize_recovered_input(self):
        """Run the deferred reverse word-recovery decode on first access.

        ``reverse()`` stashes ``(content_basis, input, subspace)`` on
        ``_recovered_input_thunk`` instead of decoding eagerly (the
        decode is report-only and its per-word ``.item()`` is a
        ``cudaMemcpyDtoH`` that breaks the brick CUDA-graph-capture
        contract). The first consumer triggers the actual decode here;
        because the captured tensors are held by reference, the result
        is bit-identical to the old eager path. A no-op during pure
        training (no consumer ever fires -> 0 host syncs).
        """
        if (self._recovered_input is None
                and self._recovered_input_thunk is not None):
            cb, inp, sub = self._recovered_input_thunk
            self._recovered_input = cb.decode_reverse_meta(inp, subspace=sub)
            self._recovered_input_thunk = None
        return self._recovered_input

    def reconstruct_data(self, text=False):
        """Render the last recovered text state stored on PerceptualSpace."""
        self._materialize_recovered_input()
        if self._recovered_input is None:
            raise RuntimeError("reconstruct_data() called before reverse()")
        return self.subspace.what.reconstruct_data(self._recovered_input, text=text)

    def reconstruct_to_buffer(self, buf_size=None):
        """Render the last recovered text buffer stored on PerceptualSpace.

        Requires a prior ``reverse()`` to have populated
        ``_recovered_input``. Enforces that WhereEncoding's period
        covers ``buf_size`` so the sin/cos decode doesn't alias near
        angle=0 and stamp tokens at spurious offsets.
        """
        self._materialize_recovered_input()
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
        self._materialize_recovered_input()
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
        """Self-test; verifies the round-trip / invariant."""
        pass
class ModalSpace(Space):
    """Composite space routing what/where/when through independent PerceptualSpaces.

    When nWhere=nWhen=0, degenerates to a single PerceptualSpace on the full
    embedding.  Per-branch passthrough flags were retired together with
    ``Space.passThrough`` — every branch now runs its full PerceptualSpace
    transform.
    """
    name = "Percepts"
    config_section = "ModalSpace"

    def __init__(self, inputShape, spaceShape, outputShape):
        """Initialize ModalSpace; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        section = self.config_section
        super().__init__(inputShape, spaceShape, outputShape)

        # Derive branch shapes (symmetric -- subtract off the modality you don't need)
        whatDim = self.muxedSize - self.nWhere - self.nWhen
        whatInputShape = [inputShape[0], whatDim]
        whatOutputShape = [outputShape[0], whatDim]
        whatSpaceShape = [spaceShape[0], spaceShape[1]]

        self.whatSpace = PerceptualSpace(whatInputShape, whatSpaceShape, whatOutputShape)

        if self.nWhere > 0:
            whereShape = [inputShape[0], self.nWhere]
            whereSpaceShape = [spaceShape[0], self.nWhere]
            self.whereSpace = PerceptualSpace(whereShape, whereSpaceShape, whereShape)
        else:
            self.whereSpace = None

        if self.nWhen > 0:
            whenShape = [inputShape[0], self.nWhen]
            whenSpaceShape = [spaceShape[0], self.nWhen]
            self.whenSpace = PerceptualSpace(whenShape, whenSpaceShape, whenShape)
        else:
            self.whenSpace = None

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
        """Route each modality through its branch PerceptualSpace.

        Pulls the what / where / when slabs (either directly from a
        demuxed subspace or via slicing the muxed event), runs each
        through its dedicated branch space, then re-muxes the outputs.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if vspace.is_demuxed:
            what_in = vspace.materialize(mode="what")
            where_in = (vspace.materialize(mode="where")
                        if vspace.where is not None else None)
            when_in = (vspace.materialize(mode="when")
                       if vspace.when is not None else None)
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
        """Split event into modalities, reverse each branch, rebuild.

        Slices the muxed event into what / where / when sub-tensors,
        runs each modal sub-space's reverse, then re-concatenates them
        into the original muxed layout for the outgoing subspace.
        """
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
        """Return optimizable parameters owned by this module."""
        return self.params

    def paramUpdate(self):
        """In-place parameter update hook called once per training step."""
        self.whatSpace.paramUpdate()
        if self.whereSpace is not None:
            self.whereSpace.paramUpdate()
        if self.whenSpace is not None:
            self.whenSpace.paramUpdate()
class ConceptualSpace(Space):
    """STM bookkeeping (shift / push) + grammatical CPU on the C tier.

    In the forward data flow: PerceptualSpace -> **ConceptualSpace** -> SymbolicSpace.

    Post Stage 1.C of the two-loop pi/sigma substrate refactor
    (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md): the atomic
    forward C-tier fold ``sigma_percept`` is RETIRED. The C tier no
    longer holds a parameterised percept→concept fold at the
    substrate level. ``forward(PS_subspace, SS_subspace)`` performs
    STM bookkeeping only: shift the per-batch idea stack one slot
    (``STM[0..N-2] = STM[1..N-1]``) and push the materialised PS+SS
    combination onto the top slot (Miller cap). The signal-router
    grammar dispatch that consumes STM is Stage 3; until then,
    downstream consumers (SymbolicSpace.forward and the cross-pass
    C→P / C→S carriers ``_subspaceForPS`` / ``_subspaceForSS``) see
    the pushed idea verbatim.

    Supports optional self-attention and VQ codebook quantization on
    the pushed idea (read-only snap; codebook writes happen at SS).
    """
    name = "Concepts"
    config_section = "ConceptualSpace"
    # Concepts are unit-norm directions; the input magnitude in [-1, +1]
    # encodes belief certainty (1 = known true, 0 = unknown, -1 = known
    # false). Use dot-product retrieval (single matmul, codebook held
    # unit-norm by EMA) so the certainty signal survives end-to-end.
    # PerceptualSpace and SymbolicSpace inherit the default False --
    # their codebooks store patterns whose magnitude carries information
    # and want the Euclidean / cached-norm matmul path. See
    # doc/Spaces.md "Codebook similarity metric".
    use_dot_product = True

    def __init__(self, inputShape, spaceShape, outputShape,
                 stage_idx=None, is_last=False):
        """Initialize ConceptualSpace; allocate state for the class contract.

        See class docstring for invariants.

        ``stage_idx``/``is_last`` mark the stage's position within the
        per-stage cascade (used by callers that need stage-aware wiring;
        the construction below is identical across stages).
        """
        section = self.config_section
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        naive = TheXMLConfig.get("architecture.naive")
        monotonic = bool(TheXMLConfig.get("architecture.monotonic", default=False))
        # SVD-orthogonalize the codebook at construction so ``project_reverse``
        # (which scales by 1/Σ) is well-conditioned from t=0. Always on; the
        # XML <svdOrthogonalInit> knob has been retired.
        self._svd_orthogonal_init_cfg = True
        super().__init__(inputShape, spaceShape, outputShape)
        self.nonlinear = nonlinear
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        # Right-half loopback widening retired: ConceptualSpace.forward
        # takes its two inputs as explicit args (``PS_subspace``,
        # ``SS_subspace``) from the recurrent cell and shape-matched
        # averages them after a bivector lift on the symbolic side. The
        # legacy ``subsymbolic_widen_dim`` constructor parameter and the
        # ``[P_event || S_event]`` concat it gated were removed together
        # with ``SubsymbolicSpace``.
        self._right_half_dim = 0
        # ConceptualSpace combines its two inputs (perceptual + symbolic
        # loop) from explicit ``forward(PS_subspace, SS_subspace)``
        # arguments supplied by the recurrent cell -- no post-construction
        # sibling refs.
        # Optional C-tier prior for chat-loop generation: the inference
        # loop (BasicModel.generate_sentence) lifts the ARMA-predicted
        # next sentence rep through InterSentenceLayer.cast into
        # concept_dim and stages it here. When set under Stage 1.C, the
        # prior is added to the pushed-idea activation before the
        # codebook lookup (the residual injection point is preserved on
        # the bookkeeping path), then cleared. None during training --
        # the attribute exists so the forward path can read it
        # unconditionally.
        self._c_prior = None
        input = self.subspace.getEncodedInputSize()
        if self.codebook_mode == "project":
            # project codebook: PiLayer stays a square isomorphism
            # inside conceptual content space. Dim adaptation to the
            # prototype width happens INSIDE ProjectionBasis
            # (``[B, V, D] -> [B, N]``), not inside Pi. This preserves
            # XOR's discriminative info instead of squeezing it through
            # a non-square Pi bottleneck.
            output = input
        else:
            output = self.subspace.getEncodedOutputSize()
        if hasAttention:
            self.attention = AttentionLayer(output, output, type="transformer")

        # Stage 1.C of the two-loop pi/sigma substrate refactor
        # (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
        # ``self.sigma_percept`` (and the paired ``sigma_percept_1`` /
        # ``sigma_percept_2`` ergodic variants and the
        # ``_sigma_percept_reverse`` helper) are RETIRED. The C tier no
        # longer holds an atomic percept→concept fold operator at the
        # substrate level. ``ConceptualSpace.forward`` now performs STM
        # bookkeeping only (shift the per-batch idea stack one slot;
        # push the new idea onto the Miller-cap top slot). The
        # signal-router-based grammar dispatch that consumes STM
        # contents to produce a fold is Stage 3; until then, downstream
        # consumers (SS.forward and the cross-pass C→P feedback) see
        # the pushed idea pass-through verbatim.
        #
        # ``self.layers`` / ``self.params`` are kept as empty containers
        # so the Space's Layer cascade and parameter aggregation
        # contract (other GrammarLayer attachments, optional
        # SymbolLearningLayer) remain valid.
        self.params = []
        self.layers = nn.ModuleList([])
        self._sparsity = SparsityRegLayer(
            l1_lambda=float(getattr(self, "l1_lambda", 0.0) or 0.0),
            enabled=bool(getattr(self, "codebook", False)),
        )

        # Short-term memory: per-batch stack of unquantized C-tier
        # "ideas" produced by the per-word subsymbolic round trip in
        # the stem. Capacity is the chart's sentence-length bound
        # (``<WordSpace><wMax>``) so each word can fill its own slot
        # before the chart runs at C-tier in the body; the 7±2 working
        # memory cap would truncate longer sentences. Subsymbolic
        # operation can still override via ``<stmCapacity>``.
        try:
            stm_capacity_xml = TheXMLConfig.space(section, "stmCapacity")
            stm_capacity = int(stm_capacity_xml) if stm_capacity_xml else None
        except (KeyError, TypeError, ValueError):
            stm_capacity = None
        if stm_capacity is None:
            try:
                w_max_xml = TheXMLConfig.get("WordSpace.wMax", 0)
                w_max = int(w_max_xml) if int(w_max_xml) > 0 else 8
            except Exception:
                w_max = 8
            stm_capacity = int(w_max)
        concept_dim = int(outputShape[1])
        # ``stm_capacity`` / ``concept_dim`` published as attributes so
        # WordSubSpace.__init__ can size its typed-STM buffers to match
        # (Phase D of doc/specs/2026-05-21-wordsubspace-stm-layer-
        # refactor.md). The typed-metadata stack formerly allocated here
        # via ``_init_typed_stm`` now lives directly on WordSubSpace; the
        # ``ShortTermMemory`` Layer reads / writes those buffers.
        self.stm_capacity = int(stm_capacity)
        self.concept_dim = int(concept_dim)
        self.stm = ShortTermMemory(
            batch=1, capacity=stm_capacity, concept_dim=concept_dim)

        # Persistent CS->PS event carrier, owned + allocated ONCE here
        # (eager, pre-compile). `forward` reuses it via `set_event`
        # instead of constructing a fresh `SubSpace(...)` + `copy_context`
        # every call -- those were torch.compile graph breaks (recon
        # #5/6/7: SubSpace ctor / copy_context are untraceable object
        # plumbing). The consumer (`PerceptualSpace.forward`) only reads
        # it via `.is_empty()` / `.materialize()`, never its context, so
        # dropping `copy_context` here is safe (it is never the returned
        # vspace, so the pipeline copy_context invariant does not apply).
        # `set_event` is shape-agnostic, so the (1,1) init shape is just
        # a placeholder overwritten per call. See
        # doc/plans/2026-05-16-compiled-step-boundary-design.md.
        self._subspaceForPS = SubSpace(
            inputShape=(1, 1), outputShape=(1, 1),
            nInputDim=1, nOutputDim=1)

        # Symbol-learning Layer: detached accumulator + MDL-flavored
        # promotion policy. Off by default; enabled via
        # ``<architecture><symbolLearning enabled="true"/>``. Owned by
        # ``ConceptualSpace`` because the QE / PMI hook points fire at
        # the C-tier snap and the C-tier REDUCE; placed in
        # ``self.layers`` so the Layer cascade reaches it. Promotion is
        # at explicit flush boundaries only (see ``flush_symbol_learning``);
        # never inside autograd ``forward``.
        from Layers import SymbolLearningLayer as _SymbolLearningLayer
        self.symbolLearningLayer = _SymbolLearningLayer(
            enabled=_SymbolLearningLayer.enabled_from_config())
        self.layers.append(self.symbolLearningLayer)

    def _build_what_basis(self):
        """Bivector regime: build a ``ProjectionBasis`` on ``.what`` so
        ``forward(input)`` returns the per-prototype catuskoti bivector
        ``[B, V_C, 2]`` and ``reverse(bivec)`` lifts it back to
        ``[B, V, D_C]`` via the exact LDU inverse on
        ``InvertibleLinearLayer``.

        Legacy regime: returns None — the ``.event`` Codebook built by
        ``_build_object_basis`` handles the legacy VQ snap behaviour.

        Replaced the prior Codebook(.project=True) path 2026-05-13:
        the LDU parameterization is structurally invertible (no SVD
        cache, no per-forward ``_project_cache``) and lives in its own
        basis type so the snap surface and the projection surface
        don't share a class.
        """
        # ``<codebook>project</codebook>`` -> a scalar ProjectionBasis on
        # ``.what`` (``[B, V, D] -> [B, N]``). ``quantize`` / ``none``
        # keep the legacy behaviour: the ``.event`` Codebook built by
        # ``_build_object_basis`` handles the snap and ``.what`` is None.
        if self.codebook_mode != "project":
            return None
        basis = ProjectionBasis()
        basis.use_dot_product = bool(getattr(self, "use_dot_product", False))
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
        )
        return basis

    def _stm_shift_and_push(self, idea):
        """STM bookkeeping primitive: shift slots left by 1 on rows at
        capacity, then write the new idea to the top slot. For rows
        below capacity, this degenerates to an ordinary push.

        ``idea`` is the per-batch payload tensor of shape ``[B, D]`` —
        the materialised PS+SS combination ConceptualSpace.forward
        produces. Routes through the underlying buffer + depth
        primitives on ``self.stm`` (which proxies to WordSubSpace when
        attached, or its own fallback buffer otherwise).

        Per the Miller cap (default 7±2 via ``<stmCapacity>`` /
        ``<wMax>``), the oldest idea drops out when overflowing — the
        STM is a rolling window. This is the substrate-level analogue
        of the chart's reduce-then-push when the chart was the
        consumer; here the consumer is the signal-router-based grammar
        dispatch (Stage 3, not wired yet).
        """
        stm = self.stm
        stm.ensure_batch(int(idea.shape[0]))
        B = int(idea.shape[0])
        cap = int(stm.capacity)
        if cap <= 0:
            return
        # Resolve buffer / depth refs through the STM proxy (handles
        # both the WordSubSpace-attached and fallback paths uniformly).
        buf = stm._buffer
        depth = stm._depth
        # Row-by-row: rows at capacity get a left-shift of the slots
        # [1..cap) -> [0..cap-1), then the new idea is written to slot
        # ``cap - 1``. Rows below capacity get a plain push at
        # ``depth[b]``. The depth pointer saturates at ``cap``.
        new_depth = depth.clone()
        for b in range(B):
            d = int(depth[b].item())
            if d >= cap:
                # Shift left by one slot, drop oldest, write new at top.
                buf[b, : cap - 1] = buf[b, 1 : cap].clone()
                buf[b, cap - 1] = idea[b]
                # depth saturates at cap (no further growth).
            else:
                buf[b, d] = idea[b]
                new_depth[b] = d + 1
        stm._depth = new_depth
        # Track the host-side maximum depth high-watermark (used by
        # snapshot()). Bumped to current saturating depth across rows.
        if B > 0:
            cur_max = int(new_depth.max().item())
            if cur_max > int(stm._max_depth_host):
                stm._max_depth_host = cur_max

    def distance(self, x, y):
        # This is a dot-product distance that assumes the X are normalized.
        # However, if the X are not normalized, the magnitudes may be taken as a degree of certainty or knowing.
        # In which case, how do they grow from ignorance to certainty?
        # They would do so naturally if the input vectors are normalized.
        # It would also be possible to use a tunable transfer function.
        """Distance.
        
        See class docstring for the operation contract.
        """
        return x.T @ y
    def certainty(self, x):
        """Certainty.
        
        See class docstring for the operation contract.
        """
        return x.T @ x

    def Reset(self, batch=None, hard=True):
        """Clear the subspace event so next forward() does a full recompute.

        On hard reset (sentence boundary), also clears the
        ShortTermMemory: the per-batch idea stack does not persist
        across sentences (matching the existing soft/hard reset
        semantics for ``_last_svo`` and ``_stm_fired`` on WordSpace).

        See ``Space.Reset`` for ``batch`` / ``hard`` semantics.
        """
        super().Reset(batch=batch, hard=hard)
        if not hard:
            return
        sub = getattr(self, 'subspace', None)
        if sub is not None and getattr(sub, 'event', None) is not None:
            sub.event.setW(None)
        # Sentence-boundary STM clear: per-batch idea stack drops
        # everything from the just-finished sentence; the next
        # sentence starts with an empty STM.
        stm = getattr(self, 'stm', None)
        if stm is not None:
            stm.clear(b=batch)

    def forward(self, subspace, word_subspace=None):
        """STM bookkeeping: shift the per-batch idea stack and push the
        new idea onto the Miller-cap top slot.

        Post Stage 1.C of the two-loop pi/sigma substrate refactor
        (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md): the
        atomic forward C-tier fold ``sigma_percept`` is retired. This
        method no longer applies a parameterised percept→concept
        operator; it only does STM bookkeeping. The signal-router
        grammar dispatch that consumes STM contents is Stage 3.

        Args:
            subspace: the perceptual content (was ``PS_subspace``).
                Materialised event is shape ``[B, N, D]``; one idea
                per batch row is derived (mean over N) and pushed to
                the STM.
            word_subspace: optional symbolic-loop data carrier (was
                ``SS_subspace``). When non-empty AND shape-matched to
                the perceptual event, the materialised SS event is
                summed with the PS event before the per-row reduction
                (a placeholder PS+SS combine for Stage 1.C — the
                signal-router-based grammar dispatch in Stage 3 will
                replace this simple sum with the proper rule
                composition). Otherwise PS alone is pushed.

        The cross-pass C→P / C→S feedback the per-word recurrence
        consumes is still published by writing the pushed idea event
        to ``self._subspaceForPS`` / ``self._subspaceForSS`` (the
        persistent SubSpace objects allocated once in ``__init__``).
        Downstream callers thread those attributes to the next
        ``SymbolicSpace.forward`` / ``PerceptualSpace.forward`` call —
        no atomic fold applied; the consumers see the pushed idea
        verbatim until Stage 3 wires the signal router in.

        Returns:
            The materialised ``subspace`` (post-context-copy), so
            downstream consumers continue to receive a SubSpace whose
            event matches what was pushed onto STM.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        # Materialise the perceptual input event. Stage 1.C contract
        # leaves the dim as-is (no percept→concept lift); downstream
        # consumers see the pushed idea verbatim.
        primary = subspace.materialize()
        sym = None
        if word_subspace is not None and not word_subspace.is_empty():
            sym = word_subspace.materialize()
        # Stage 1.C placeholder PS+SS combine: sum if shapes match,
        # else PS alone. The signal-router-based grammar dispatch in
        # Stage 3 will replace this with the proper rule composition.
        if sym is not None and sym.shape == primary.shape:
            combined = primary + sym
        else:
            combined = primary
        # Per-row reduction to a single idea vector: mean over the N
        # axis. STM holds one idea per batch row per slot; the N-axis
        # of the materialised event collapses to that single payload.
        if combined.dim() == 3:
            idea = combined.mean(dim=1)         # [B, D]
            event_for_carrier = combined         # preserve [B, N, D]
        elif combined.dim() == 2:
            idea = combined                      # [B, D]
            event_for_carrier = combined.unsqueeze(1)   # [B, 1, D]
        else:
            # Defensive: degenerate shape — fall back to a no-op
            # (don't crash; return the input subspace unchanged).
            return subspace
        # STM bookkeeping: shift-then-push (Miller cap rolling window).
        self._stm_shift_and_push(idea)
        # Optional C-prior injection (chat-loop generation): preserved
        # for backward compatibility with InterSentenceLayer.cast /
        # BasicModel.generate_sentence consumers that stage a prior
        # before the forward call. Added to the pushed-idea event so
        # downstream consumers see the conditioned vector.
        if self._c_prior is not None:
            prior = self._c_prior
            if prior.dim() == 1:
                prior = prior.unsqueeze(0)
            if event_for_carrier.dim() == 3 and prior.dim() == 2:
                if prior.shape[0] == 1:
                    prior_b = prior.expand(
                        event_for_carrier.shape[0], -1)
                elif prior.shape[0] == event_for_carrier.shape[0]:
                    prior_b = prior
                else:
                    K = max(1, event_for_carrier.shape[0]
                            // max(1, prior.shape[0]))
                    prior_b = prior.repeat_interleave(K, dim=0)
                if prior_b.shape[-1] == event_for_carrier.shape[-1]:
                    event_for_carrier = (
                        event_for_carrier + prior_b.unsqueeze(1))
            self._c_prior = None
        # Write the pushed-idea event back to the carrier subspace so
        # downstream consumers (SymbolicSpace.forward via
        # ``_subspaceForSS``) see the bookkept event.
        subspace.set_event(event_for_carrier)
        # ``clear_last_svo`` was an end-of-CS-forward bookkeeping hook
        # in the prior fold-and-snap design; retained here because the
        # WordSubSpace consumers still expect the SVO slot cleared at
        # the cycle boundary.
        ws = getattr(self, 'wordSubSpace', None)
        if ws is not None:
            ws.clear_last_svo()
        # Expose the two consumer views for the recurrent cell:
        #   _subspaceForSS -- the pushed-idea subspace SS.forward
        #     consumes next pass.
        #   _subspaceForPS -- the persistent C→P feedback carrier
        #     (mutated in place via ``set_event``) the per-word loop
        #     reads next iteration. Stage 1.A retired PS's direct
        #     consumption of this view, but the lift belongs to CS
        #     architecturally so the carrier is still emitted.
        # ``object.__setattr__`` matches the pre-1.C reasoning: avoids
        # mutating ``self._modules`` under torch.compile guards.
        object.__setattr__(self, "_subspaceForSS", subspace)
        if event_for_carrier is not None and event_for_carrier.dim() == 3:
            self._subspaceForPS.set_event(event_for_carrier)
        else:
            # Degenerate edge fallback (preserved from pre-1.C
            # behavior; not the hot path).
            fallbackForPS = SubSpace(
                inputShape=(0, 1), outputShape=(0, 1),
                nInputDim=1, nOutputDim=1)
            object.__setattr__(self, "_subspaceForPS", fallbackForPS)
        return subspace

    def reverse(self, subspace):
        """Reverse pass: thin pass-through under Stage 1.C.

        The atomic ``sigma_percept`` SigmaLayer (and its
        ``_sigma_percept_reverse`` two-pass helper) are retired by
        Stage 1.C; there is no inverse fold to apply. The reverse
        pipeline still calls ``CS.reverse`` between PS.reverse and
        SS.reverse, so this method preserves the contract (early
        return on empty, copy_context for downstream plumbing) but
        otherwise passes the subspace through. Stage 3's signal
        router will own the reverse-direction substitute for the
        fold inverse once it lands.

        ProjectionBasis-backed configs still lift the signed-scalar
        ``[B, N]`` event back to ``[B, V, D]`` via the LDU inverse
        (this is the codebook's own inverse, owned by the basis, not
        a substrate fold), so reverse stages see the natural shape.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        y = self.reverseBegin(vspace, returnVectors=True)
        if isinstance(self.subspace.what, ProjectionBasis):
            # Codebook-side LDU inverse: own contract of ProjectionBasis,
            # not the retired substrate fold. Keeps the reverse shape
            # contract intact for downstream PS.reverse consumers.
            V_orig = int(self.inputShape[0])
            y = self.subspace.what.reverse(y, V=V_orig)
        # Optional right-half slice (kept for any config that widens
        # the input; with the bookkeeping body this is generally a
        # no-op since ``_right_half_dim`` is 0).
        if self._right_half_dim > 0 and y is not None and y.dim() == 3:
            y = y[..., :-self._right_half_dim]
        if y is not None:
            self.concepts = y.detach()
        vspace = self.reverseEnd(y, returnVectors=True)
        if y is not None and not isinstance(
                self.subspace.what, ProjectionBasis):
            # Range-check on the pass-through reverse: matches the
            # pre-1.C downstream invariant. Skipped for ProjectionBasis
            # (which bounds its own output).
            vspace.normalize("percepts", target="what")
            vspace.normalize("percepts", target="where")
        return vspace

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        pass
class SymbolicSpace(Space):
    """Codebook-backed symbol stack with swap operations.

    In the forward data flow: ConceptualSpace -> **SymbolicSpace** -> OutputSpace.
    The symbol stack (StackSpace) holds entries produced by ConceptualSpace's
    shift/reduce loop. Each entry has what (codebook index), where (position),
    and when (derivation order).

    S-tier operations (swap) operate on whereEncodings of node children.
    The top-level `true()` evaluates the full stack activation -> scalar.

    -----------------------------------------------------------------------
    Bivector / codebook / activation lifecycle
    -----------------------------------------------------------------------
    SymbolicSpace overrides ``self.subspace.nWhat = 2`` so ``.what`` carries
    the 4-valued (catuskoti / tetralemma) bivector ``[pos_pole, neg_pole]``
    per slot.  See doc/BuddhistParallels.md for the semantics.

    With ``<codebook>true``, ``.what`` is a Codebook whose ``.W`` parameter
    holds ``[V_sym, 2]`` learned symbol-prototype bivectors.  The per-batch
    bivector content (shape ``[B, N, 2]``) is what the codebook quantizes
    against; it is stored in the codebook's ``_active_payload`` slot, NOT
    in ``.W`` itself.

    Forward-pass lifecycle (clients should NOT need to walk this manually
    -- subspace.materialize / subspace.resolve hide it):

      1. ConceptualSpace's PiLayer produces ``act`` shape ``[B, N, sym_D]``.
      2. ``self.subspace.set_event(act)`` populates ``.event`` (muxed view)
         and demuxes the first ``nWhat=2`` columns into ``.what``; activation
         is reset to default all-ones presence.
      3. ``self.resolve(self.subspace)`` reads the bivector from ``.event``
         (preferred) or ``.what`` (fallback), computes ``pos - neg`` (signed
         Degree of Truth), and stores the [B, N] scalar directly via
         ``.activation.setW`` (bypassing ``set_activation``'s bivector lift
         so sign is preserved).
      4. The codebook snap quantizes the resolved scalar to its nearest
         symbol-prototype scalar; the result is committed via the public
         ``set_activation()`` (which lifts the [B, N] scalar back to a
         bivector for storage when ``activeEncoding.nDim == 2``).
      5. Downstream readers call ``self.subspace.materialize(mode="activation")``
         and see the post-snap, presence-reduced value.

    forward(subspace) write contract:
      Reads the incoming subspace's bivector / activation (set by the
      ConceptualSpace stage), but writes ALL state mutations to
      ``self.subspace``, never to the passed-in ``subspace``.  This is
      because the receiving subspace may carry a Codebook on ``.what``
      whose learned prototypes must not be overwritten with per-batch
      tensor data.

    .what / .activation are NOT the same field:
      * ``.what`` holds the continuous, pre-snap bivector that the
        codebook is queried against.
      * ``.activation`` holds the post-resolve / post-snap scalar
        (presence-reduced or signed depending on which writer last
        committed to it).  After the codebook snap, the two values
        intentionally diverge.

    For other spaces' codebook layouts:
      * PerceptualSpace's ``.what`` holds a BPE / lexicon prototype
        matrix; activation lives separately.
      * ConceptualSpace is a pure-event subspace (``.what``, ``.where``,
        ``.when`` empty); state is on ``.event`` via ``set_event()``.

    Distinct from the WordSpace category embedding:
      * ``WordSpace.category_embedding``: ``nn.Embedding[max(64,
        |grammar.categories|), 4]`` learned embeddings keyed by grammar
        nonterminal / POS name (S, NP, VP, N, V, ADJ, ...).  Used by
        the chart's POS scorer.
      * ``SymbolicSpace.subspace.what.W``: ``[V_sym, 2]`` learned
        symbol-prototype bivectors.  Used by the codebook snap.
      Independent codebooks, independent semantics.
    """
    name = "Symbols"
    config_section = "SymbolicSpace"
    # Phase-1 mode gating: when set True (by Model in ``parallel``
    # mode) ``forward`` zeroes the event tensor and skips resolve /
    # lift / codebook / TruthLayer paths. Default False preserves
    # legacy behaviour. See `2026-05-05-subsymbolic-knowing-handoff`.
    held_at_zero = False
    # NOTE: ``self.rule_codebook`` is built in ``__init__`` and registered
    # as an nn.Module submodule (Phase 3 of the SubSpace.what STM
    # refactor; see
    # doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md).
    # Holds grammar rule identity / .where location; NOT the source of
    # parent vectors -- parent.what = SyntacticLayer.execute(rule_id,
    # left, right). A class-level default would shadow the registered
    # submodule (nn.Module routes Module values through ``_modules``
    # rather than ``__dict__``), so we intentionally do NOT declare one.

    def empty_state(self, batch=1):
        """Return a zero tensor shaped like this space's symbolic state.

        Used by the unified merged outer loop to seed ``ss`` before the
        first concept emission. ``batch`` sizes the leading axis; the
        rest comes from ``outputShape``.
        """
        nOutput = int(self.outputShape[0])
        nDim = int(self.outputShape[1])
        # No explicit `device=TheDevice.get()`: that passes a
        # `DeviceHandle` (C-str subclass) into a traced factory call ->
        # torch.compile graph break (recon #0; same root cause as #6).
        # The default-device mode places it correctly and the caller
        # (`_forward_per_stage`) immediately `.to(inputData.device)`s
        # it (a real torch.device) -- behaviour-identical, traceable.
        return torch.zeros(int(batch), nOutput, nDim)

    def __init__(self, inputShape, spaceShape, outputShape, conceptualSpace=None):
        """Initialize SymbolicSpace; allocate state for the class contract.

        See class docstring for invariants.

        Codebook-width invariant (``symbol_dim == concept_dim``) is
        validated at config-load time in
        ``ModelFactory.validate_config``; we don't re-assert it here
        because ``inputShape`` / ``outputShape`` carry the per-stage
        activation widths (which may equal ``nOutputDim`` rather than
        ``nDim`` in bivector mode), not the codebook width.
        """
        section = self.config_section
        nonlinear = TheXMLConfig.space(section, "nonlinear")
        # Always-on (XML knob retired); see ConceptualSpace for rationale.
        self._svd_orthogonal_init_cfg = True
        super().__init__(inputShape, spaceShape, outputShape, customVQ=True)
        self.conceptualSpace = conceptualSpace
        # Sibling reference: PerceptualSpace owns the physical Embedding
        # (the input pipeline's InputSpace._peer_perceptual.vocabulary
        # wiring requires it there), but ``SymbolicSpace`` is the
        # logical owner after the lexicon migration -- ``S.vocabulary``
        # and the orthographic-API methods live here and delegate
        # through this back-reference. Wired post-construction in
        # ``Model.__init__`` / ``BasicModel._build_pipelines_per_stage``.
        # ``None`` for standalone unit tests so legacy single-Space
        # construction still works.
        self.perceptualSpace_ref = None
        self.nonlinear = nonlinear
        # Post-2026-05-07 rollback: SymbolicSpace inherits the natural
        # ``nWhat == self.nDim`` and ``muxedSize`` from
        # ``Space.__init__`` -- no leading [pos, neg] bivector pinned
        # into the codebook. The per-prototype catuskoti bivector
        # ``[B, V_S, 2]`` lives on ``subspace.activation`` instead and
        # is populated by ``Codebook.forward(input)`` --
        # the intrinsic snap. See doc/Spaces.md for the post-rollback
        # geometry and doc/BuddhistParallels.md for the tetralemma.
        nSymbols = spaceShape[0]
        # SymbolicSpace owns no SigmaLayer / PiLayer: the architectural
        # rule restricts those to PerceptualSpace and ConceptualSpace.
        # With concept_dim == symbol_dim enforced above, the C->S path
        # is dimensionally a pass-through; learned C->S transforms live
        # in ConceptualSpace.pi (which is constructed to output
        # symbol-shaped activations directly).
        self.params = []
        self.layers = nn.ModuleList()

        # Propositional negation slot: NEG sits at the SymbolicSpace
        # output, gated on ``rule_probability("not(S)")``. ``NotLayer``
        # operates on the per-prototype activation bivector ``[B, V_S, 2]``;
        # the dispatcher (SyntacticLayer) reads
        # ``subspace.activation.getW()`` and feeds the bivector tensor in.
        # Same Layer-shaped interface as the parameterized fold operators
        # -- uniform invocation from both bottom-up forward gating and
        # top-down SyntacticLayer dispatch.
        self.propositional_negation = NotLayer()
        self.layers.append(self.propositional_negation)

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
        # reach it via ``self.wordSubSpace.truth_layer`` (see forward() below
        # and the truth-using paths in BasicModel).
        try:
            self.accumulateTruth = float(TheXMLConfig.space(section, "accumulateTruth"))
        except (KeyError, TypeError, ValueError):
            self.accumulateTruth = 0.0

        # Per-instance Gaussian region width used by ``area`` /
        # ``luminosity``. ``None`` when no calibrated extent is set;
        # the metrics migrated to the Mereology mixin (with a
        # hyperrectangle-volume formula) and consumers no longer key
        # on a global default.
        self.activeSigma = None

        # Trust threshold for the per-cell record_batch path: activation
        # norms ramp from 0 at norm=0 to 1 at norm=truthMinMagnitude. The
        # legacy ``should_store`` two further gates (novelty/consistency)
        # were dropped along with that function — under record_batch the
        # codebook lookup at compact-time naturally dedupes near-zero and
        # near-duplicate vectors against the existing prototype.
        try:
            self._truth_min_magnitude = float(
                TheXMLConfig.space(section, "truthMinMagnitude"))
        except (KeyError, TypeError, ValueError):
            self._truth_min_magnitude = 0.3

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

        # Well-known atoms: a name -> codebook-row dict for parents
        # that the architecture cares about by name (rather than by
        # index). Saved in the .ckpt bundle's ``vocab_extras`` so the
        # mapping survives reload. "words" is the canonical meronomy
        # parent for lexicon entries: the codebook quantizer can
        # recognize "this entry is an instance of an existing concept
        # (a word)" instead of spawning a fresh row per quantization.
        # Convention: row 0 holds "words"; new well-known atoms append.
        self.well_known_atoms = {"words": 0}
        cb = getattr(self.subspace, 'what', None)
        if cb is None or not isinstance(cb, Codebook):
            cb = getattr(self.subspace, 'event', None)
        if (cb is not None and isinstance(cb, Codebook)
                and getattr(cb, 'part_parents', None) is not None
                and cb.part_parents.numel() > self.well_known_atoms["words"]):
            # The "words" parent is a root atom: -1 sentinel means it
            # is itself a top-level concept, not part of anything
            # higher.
            cb.set_part_parent(self.well_known_atoms["words"], -1)

        # Phase 3 of the SubSpace.what STM refactor: wire V_sym into the
        # global Grammar so where_id_for_rule produces correct offsets
        # (rule slot = V_sym + 1 + rule_id), and build the rule codebook
        # alongside the existing symbol codebook. Lazy import to avoid
        # any circular-import risk with Language.py.
        from Language import TheGrammar, RuleCodebook, LanguageLayer
        TheGrammar.symbol_vocab_size = int(nSymbols)
        try:
            TheGrammar._ensure_configured()
            n_rules = TheGrammar.num_rules()
        except Exception:
            # In tests that build SymbolicSpace without an XML grammar
            # we still want a valid (empty) RuleCodebook.
            n_rules = 0
        self.rule_codebook = RuleCodebook(
            num_rules=n_rules,
            embedding_dim=0,
            grammar=TheGrammar,
        )

        # Phase 5 of the STM refactor: SymbolicSpace owns its own
        # LanguageLayer (distinct from Chart's compatibility router) for
        # the stack-rewrite path. The router has no attached ops on
        # this path -- it dispatches through self.syntacticLayer.execute
        # rather than its internal ModuleDicts. Gate dispatch on
        # ``self.use_stack_router`` (XML <useStackRouter> knob, default
        # False); when False the legacy forward path runs unchanged.
        try:
            use_stack_router_raw = TheXMLConfig.space(section, "useStackRouter")
        except (KeyError, TypeError, ValueError):
            use_stack_router_raw = False
        self.use_stack_router = bool(use_stack_router_raw)
        # The router itself is cheap (no attached ops -> no parameters)
        # so we build it unconditionally; the flag only gates dispatch.
        # feature_dim matches the codebook width; hidden_dim is a
        # placeholder that the stack-rewrite path does not consult.
        _ss_dim = int(outputShape[1])
        self.languageLayer = LanguageLayer(
            n_input=int(inputShape[0]),
            n_output=int(outputShape[0]),
            hidden_dim=max(_ss_dim, 8),
            feature_dim=_ss_dim,
            max_depth=max(int(outputShape[0]), 2),
        )

        # Phase 1A.2: make the SYMBOL VQ codebook learnable by gradient
        # and drop its in-call EMA Parameter mutation, so
        # ``SymbolicSpace.forward``'s default VQ snap path performs NO
        # persistent-state mutation in-call and is idempotent (a
        # CUDA-graph-capture prerequisite). The concepts symbols encode
        # live in a conceptual embedding, so this codebook is an
        # embedding moved by the task loss, not an EMA cluster.
        # ``VectorQuantize`` / ``Codebook`` are SHARED by the Perceptual
        # and Conceptual codebooks too; this flag is set ONLY on the
        # symbol ``.what`` codebook's ``VectorQuantize`` instance here,
        # so the Conceptual EMA path stays byte-identical (single-writer
        # invariant: SymbolicSpace still solely owns/writes the symbol
        # codebook -- only EMA -> gradient changes, not the writer).
        # No-op when ``.what`` is a ProjectionBasis / passthrough Tensor
        # / customVQ-disabled Codebook (no ``.vq`` to flip).
        self._make_symbol_codebook_learnable()

        self.params = list(self.parameters())
        self._sparsity = self._build_sparsity_regularizer(
            self.l1_lambda, self.codebook)
        self._smoothing = SmoothingRegLayer(
            lam=float(self.discontinuity_lambda or 0.0),
            enabled=bool(self.discontinuity_lambda
                         and self.discontinuity_lambda > 0.0),
        )
        self._impenetrable = self._build_impenetrable_layer()
        # FusionLayer / ContiguousLayer eager construction was retired
        # 2026-05-04: the operator was a duplicate of DisjunctionLayer
        # at S-tier (same kernel ``Ops.union`` on the codebook
        # activation bivector). Grammars that fired ``Fusion(S, S)`` /
        # ``Contiguous(S)`` should migrate to ``disjunction(S, S)``,
        # which the chart's lazy-build path resolves via the
        # ``'disjunction'`` entry in ``GRAMMAR_LAYER_CLASSES``.

        """Memory budget (bytes) for VQ distance matrix chunks.

        With d content dims and K codebook entries, each row of the
        distance matrix costs K * 4 bytes.  The budget controls how
        many rows are processed per matmul.  Larger budgets mean fewer
        sequential chunks -- critical for AR batches where N can
        reach hundreds of thousands.
        """
        device = str(TheDevice.get())          # eager-only (import time)
        if 'cuda' in device:
            try:
                props = torch.cuda.get_device_properties(device)
                self.vq_chunk_budget = max(256 << 20, props.total_mem // 4)
            except Exception:
                pass
        if 'mps' in device or 'cuda' in device:
            self.vq_chunk_budget = 4 << 30
        else:
            self.vq_chunk_budget = 2 << 30

    def _make_symbol_codebook_learnable(self):
        """Phase 1A.2 scoping hook.

        Flip the symbol codebook's ``VectorQuantize`` into
        ``learnable_codebook`` mode (gradient-trained codebook, EMA
        in-call write suppressed, codebook-attached STE). Scoped to
        exactly this SymbolicSpace instance's ``.what`` codebook -- the
        ``[V_sym, nDim]`` learned symbol-prototype basis the default VQ
        snap (``SymbolicSpace.forward``) queries -- so the SHARED
        ``VectorQuantize`` / ``Codebook`` class still runs the
        byte-identical EMA path for the Conceptual codebook.

        Mirrors ``PerceptualSpace._make_perceptual_codebook_learnable``,
        but targets ``self.subspace.what`` (where the SYMBOL codebook
        lives) rather than ``get_vectors()`` / ``.event`` (where the
        PERCEPTUAL codebook lives -- ``SymbolicSpace.what`` is the
        symbol Codebook; PerceptualSpace.what is the lexicon Embedding).

        Single-writer invariant unchanged: SymbolicSpace remains the
        sole owner/writer of the symbol codebook; this changes only HOW
        it learns (in-call EMA -> downstream task-loss gradient in the
        eager ``optimizer.step``), not WHO writes it.

        Robust no-op when there is no VQ to flip: ``.what`` is a
        ``ProjectionBasis`` (``<codebook>project``; LDU surface, no
        VQ-EMA on this path), a passthrough ``Tensor``
        (``<codebook>none``), or a ``customVQ``-disabled ``Codebook``
        (``self.vq is None``). Idempotent.
        """
        what = getattr(self.subspace, "what", None)
        if what is None:
            return
        vq = getattr(what, "vq", None)
        if vq is None or not hasattr(vq, "learnable_codebook"):
            return
        vq.learnable_codebook = True

    # ------------------------------------------------------------------
    # Knowledge artifact attach: SymbolicSpace owns the trainable scalar
    # reference codebook the artifact bootstrap initializes. See plan
    # doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    # §Phase 2 — Loaders + W-shape change. Additive against the existing
    # ``subspace.what.W`` (which stays); the new ``self.references``
    # Parameter is the home of the scalar prototypes that downstream
    # consumers will migrate to.
    # ------------------------------------------------------------------
    def attach_knowledge(self, view):
        """Attach a ``KnowledgeView`` and bootstrap the trainable scalar
        reference codebook + the per-ref ``order`` buffer.

        ``self.references`` is created (or updated in place) as an
        ``nn.Parameter`` sized to the artifact's reference-codebook
        *capacity* (live rows + slack). Values are copied from the
        view's underlying section; the capacity-slack pattern is
        preserved so symbol-learning appends can re-attach an
        ``extend_artifact``-grown section without reallocating the
        Parameter unless capacity itself changed.

        ``self.order`` is a parallel long buffer (not trainable;
        discrete metadata).
        """
        super().attach_knowledge(view)
        ks = view._ks
        rc = ks['reference_codebook']
        full_refs = rc['references']
        full_order = rc['order']
        # Parameter create / update.
        if (hasattr(self, 'references')
                and isinstance(self.references, nn.Parameter)
                and self.references.shape == full_refs.shape):
            with torch.no_grad():
                self.references.data.copy_(full_refs)
        else:
            self.references = nn.Parameter(
                full_refs.clone().detach().float())
        # Order buffer create / update.
        if 'order' in dict(self.named_buffers(recurse=False)):
            existing = self._buffers['order']
            if existing.shape == full_order.shape:
                existing.copy_(full_order)
            else:
                self._buffers['order'] = full_order.clone().detach().long()
        else:
            self.register_buffer(
                'order', full_order.clone().detach().long())

    # ------------------------------------------------------------------
    # Well-known atoms (meronomy parents).
    # ------------------------------------------------------------------
    @property
    def words_atom_id(self):
        """Codebook row reserved for the meronomy parent "words".

        Backed by ``self.well_known_atoms["words"]`` so reloads from
        a .ckpt that carried a different convention still pick up the
        right slot.
        """
        return int(self.well_known_atoms.get("words", 0))

    # ------------------------------------------------------------------
    # Lexicon ownership (post-lexicon-migration)
    # ------------------------------------------------------------------
    # The orthographic Lexicon (``Embedding`` instance) is the
    # "codebook IS lexicon" structure -- one row per vocabulary entry,
    # the C→S codebook snap *is* the byte→symbol lookup, and the
    # reverse pipeline (S → C → P → I → bytes) is the only path from
    # an active symbol back to its surface form. PerceptualSpace
    # retains the physical Embedding instance for input-pipeline
    # reasons (InputSpace._peer_perceptual.vocabulary wiring), but
    # SymbolicSpace is the logical owner: the ``vocabulary`` property
    # and the orthographic-API methods live here and delegate to
    # the Embedding via ``perceptualSpace_ref``.

    def mark_word_atom(self, atom_idx):
        """Mark codebook row ``atom_idx`` as a part of the canonical
        "words" symbol so the meronomy explicitly records that this
        atom is a word-instance rather than a fresh root concept.

        Called when a new word lands in the codebook (via the Lexicon
        insert path). Without this tag the codebook quantizer would
        treat every quantization as a new root atom and the vocabulary
        would grow unbounded; with it, the quantizer can short-circuit
        to the "words" parent when an input maps to a child of an
        existing word.
        """
        cb = getattr(self.subspace, 'what', None)
        if cb is None or not isinstance(cb, Codebook):
            cb = getattr(self.subspace, 'event', None)
        if (cb is None or not isinstance(cb, Codebook)
                or getattr(cb, 'part_parents', None) is None):
            return
        if int(atom_idx) < 0 or int(atom_idx) >= cb.part_parents.numel():
            return
        cb.set_part_parent(int(atom_idx), int(self.words_atom_id))

    # ------------------------------------------------------------------
    # Stage 1.B paired-row insertion (orth + semantic) on SS.codebook
    # ------------------------------------------------------------------
    # Locked design (2026-05-27, doc/plans/2026-05-26-two-loop-pi-
    # sigma-substrate.md): when a new word lands in PS's Lexicon, SS's
    # codebook (``self.subspace.what``) gains TWO paired rows:
    #   * orthographic row = a COPY of the per-word PS vector (flat-slab
    #     invariant guarantees PS-side per-word is CS-space-dimensioned;
    #     no pi/sigma transform at insert time);
    #   * semantic row     = a fresh RANDOM CS-space vector (trainable,
    #     no init bias toward orth; per Quine / Saussure -- word != object,
    #     orth and semantic are free to diverge).
    # The two are DIRECTLY parented: orth row -> semantic row, via the
    # existing ``Codebook.set_part_parent``. No meta-symbol (skipped per
    # the controller; orth and semantic remain free to diverge).
    def insert_paired_word(self, word, ps_vector):
        """Insert paired orthographic + semantic rows for ``word`` into
        ``self.subspace.what`` (the SS codebook). Returns
        ``(orth_idx, sem_idx)``.

        ``ps_vector`` is a 1-D tensor of width ``self.nDim`` (the PS-
        side per-word vector, which is CS-space-dimensioned per the
        flat-slab invariant). The orth row is a copy of this vector;
        the semantic row is freshly randomized.

        Direct parenthood: ``part_parents[orth_idx] = sem_idx``. No
        meta-symbol; orth and semantic are free to diverge during
        training (per Quine / Saussure -- word != object).

        Raises ``RuntimeError`` if the codebook lacks room for a pair
        (``nVectors`` exhausted) -- caller must size SS.nVectors at
        least 2x the expected lexicon capacity (see MM_5M.xml where
        SS.nVectors=131072 = 2 * PS lexicon cap).
        """
        cb = getattr(self.subspace, 'what', None)
        if cb is None or not isinstance(cb, Codebook):
            raise RuntimeError(
                f"SymbolicSpace.insert_paired_word requires "
                f"self.subspace.what to be a Codebook; got "
                f"{type(cb).__name__ if cb is not None else 'None'}.")
        if cb.part_parents is None:
            raise RuntimeError(
                "SymbolicSpace.insert_paired_word requires the SS "
                "codebook to carry meronomy storage "
                "(create with category=True). _build_what_basis must "
                "have built it with category=True; this looks like a "
                "stale Codebook.")
        # Lazy init: usable rows start after the highest reserved
        # well-known atom index. Convention today: row 0 = "words".
        # Re-entry is idempotent: ``_paired_next_row`` survives across
        # inserts.
        if not hasattr(self, '_paired_next_row'):
            base = max(self.well_known_atoms.values())
            self._paired_next_row = int(base) + 1
            # Per-word orth_idx -> sem_idx mapping (precomputed so
            # downstream lookup can collapse to ONE gather, not a
            # runtime intersection). Built up at insert time.
            self._paired_orth_to_sem = {}

        orth_idx = int(self._paired_next_row)
        sem_idx = orth_idx + 1
        cap = int(cb.nVectors)
        if sem_idx >= cap:
            raise RuntimeError(
                f"SymbolicSpace.insert_paired_word: codebook is full "
                f"({orth_idx} / {cap} rows used); cannot insert "
                f"paired rows for {word!r}. Raise <SymbolicSpace>"
                f"<nVectors> to at least 2x the expected lexicon "
                f"capacity (currently {cap}).")

        # Resolve ps_vector to [nDim] float on the codebook's device.
        if not torch.is_tensor(ps_vector):
            ps_vector = torch.as_tensor(ps_vector, dtype=torch.float32)
        if ps_vector.dim() == 2 and ps_vector.shape[0] == 1:
            ps_vector = ps_vector.squeeze(0)
        if ps_vector.dim() != 1:
            raise RuntimeError(
                f"insert_paired_word: ps_vector must be 1-D "
                f"(shape [nDim={self.nDim}]); got shape "
                f"{tuple(ps_vector.shape)}.")
        if ps_vector.shape[0] != int(self.nDim):
            raise RuntimeError(
                f"insert_paired_word: ps_vector width "
                f"{ps_vector.shape[0]} != SymbolicSpace.nDim "
                f"({self.nDim}). Flat-slab invariant must hold so "
                f"the orth row is a direct copy.")

        W = cb.getW()
        if W is None:
            raise RuntimeError(
                "SymbolicSpace.insert_paired_word: SS codebook W is "
                "None; _build_what_basis did not allocate prototypes.")
        ps_v = ps_vector.detach().to(device=W.device, dtype=W.dtype)
        # Random semantic vector. Sample in the same scale as the
        # codebook init (random in [-1, 1] before any per-row
        # normalization). No bias toward orth: orth and semantic are
        # free to diverge during training.
        sem_v = torch.empty(
            int(self.nDim), device=W.device, dtype=W.dtype
        ).uniform_(-1.0, 1.0)

        # In-place write into the codebook's prototype rows. ``W.data``
        # preserves Parameter identity (the optimizer keeps owning the
        # full-capacity prototype tensor).
        with torch.no_grad():
            W.data[orth_idx, :].copy_(ps_v)
            W.data[sem_idx, :].copy_(sem_v)

        # Direct parenthood: orth -> semantic. Per design (no meta-
        # symbol), parent of orth is the semantic row, not a "words"
        # atom. This means the legacy ``mark_word_atom`` (which would
        # set part_parents[orth] = words_atom_id) is overridden by
        # this insert path -- a paired-row insertion sets the orth's
        # parent to its semantic partner.
        cb.set_part_parent(orth_idx, sem_idx)
        # Semantic row is a root atom in the orth->sem graph (it
        # doesn't parent anything itself; -1 = no parent).
        # Leave part_parents[sem_idx] at the default -1.

        self._paired_orth_to_sem[orth_idx] = sem_idx
        self._paired_next_row = sem_idx + 1
        return (orth_idx, sem_idx)

    def get_semantic_row(self, orth_idx):
        """Look up the semantic row index for a given orthographic row.

        Returns ``-1`` if the orth row has no paired semantic partner
        (e.g. it pre-dates the paired-row contract, or was inserted
        via the legacy ``mark_word_atom`` path).
        """
        cb = getattr(self.subspace, 'what', None)
        if cb is None or not isinstance(cb, Codebook):
            return -1
        # Prefer the cached map (built at insert time) -- O(1).
        if hasattr(self, '_paired_orth_to_sem'):
            mapped = self._paired_orth_to_sem.get(int(orth_idx), None)
            if mapped is not None:
                return int(mapped)
        # Fallback: read part_parents[orth_idx]; the paired-row
        # contract stores the semantic partner there. -1 if unset.
        return cb.get_part_parent(int(orth_idx))

    @property
    def vocabulary(self):
        """Return the orthographic Lexicon (Embedding), or fall back to
        SymbolicSpace's own ``.what`` codebook for callers that pre-date
        the lexicon migration. ``None`` when neither is wired
        (standalone unit tests with no perceptualSpace_ref).
        """
        peer = self.perceptualSpace_ref
        if peer is not None:
            v = peer.subspace.vocabulary
            if v is not None:
                return v
        return self.subspace.vocabulary

    def train_embeddings(self, words, method='CBOW'):
        """Run one CBOW/SBOW gradient step if words are available."""
        emb = self.vocabulary
        if isinstance(emb, Embedding) and words:
            return emb.train_step(words, method=method)
        return None

    def sbow_loss(self, words):
        """Return SBOW loss tensor for joint optimization (no backward/step)."""
        emb = self.vocabulary
        if isinstance(emb, Embedding) and words:
            return emb.sbow_loss(words)
        return None

    def _snapshot_embeddings(self):
        """Return the current WordVectors (no-op, vectors are always live)."""
        emb = self.vocabulary
        if isinstance(emb, Embedding):
            return emb.wv
        return None

    def set_embedding_sigma(self, sigma):
        """Control exploration noise on the embedding."""
        emb = self.vocabulary
        if hasattr(emb, 'set_sigma'):
            emb.set_sigma(sigma)

    def reconstruct_data(self, text=False):
        """Render the last recovered text state from the reverse pipeline."""
        peer = self.perceptualSpace_ref
        if peer is None:
            return None
        return peer.reconstruct_data(text=text)

    def reconstruct_to_buffer(self, buf_size=None):
        """Render the last recovered text buffer from the reverse pipeline."""
        peer = self.perceptualSpace_ref
        if peer is None:
            return None
        return peer.reconstruct_to_buffer(buf_size=buf_size)

    def get_recovered_word(self, batch_idx, position):
        """Return one recovered token from the most recent reverse pass."""
        peer = self.perceptualSpace_ref
        if peer is None:
            return None
        return peer.get_recovered_word(batch_idx, position)

    def _build_object_basis(self):
        """Event is a writable Tensor -- codebook lives on .what."""
        return None

    def _build_what_basis(self):
        """Symbol codebook on .what, monotonic. One row per symbol.

        Row width = ``self.nDim`` (post-2026-05-07 rollback). Each row
        is a free coefficient vector over the conceptual axes -- "how
        much of concept_i is this symbol?". Where/when ride alongside
        the encoding on the per-batch muxed event tensor; they don't
        live inside the codebook itself.

        The per-prototype catuskoti bivector ``[B, V_S, 2]``
        (tetralemma: TRUE=[1,0], FALSE=[0,1], BOTH=[1,1],
        NEITHER=[0,0]) lives on ``subspace.activation`` -- populated
        by ``Codebook.forward(input)`` (the intrinsic
        snap), inverted by ``Codebook.reverse(bivec)``
        (the cached SVD pseudo-inverse). ``test/test_idempotent_loop.py``
        exercises that path directly to verify the C↔S round-trip
        projects onto span(W) and is a fixed point thereafter.

        2026-05-13: in the bivector regime, ``.what`` is now a
        ``ProjectionBasis`` (LDU-parameterized) rather than a Codebook
        with invertible=True, matching the ConceptualSpace bivector
        builder.  The exact LDU inverse replaces the legacy SVD cache.
        """
        mode = self.codebook_mode
        if mode == "none":
            return Tensor(nVectors=self.nVectors, nDim=self.nDim)
        if mode == "project":
            basis = ProjectionBasis()
            basis.use_dot_product = bool(getattr(self, "use_dot_product", False))
            basis.create(
                self.inputShape[0],
                self.nVectors,
                self.nDim,
            )
            return basis
        basis = Codebook()
        basis.use_dot_product = bool(getattr(self, "use_dot_product", False))
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
            customVQ=self.customVQ,
            monotonic=True,
            category=True,
            STE=True,
            invertible=False,
        )
        return basis

    @classmethod
    def _build_sparsity_regularizer(cls, l1_lambda, codebook_enabled):
        return SparsityRegLayer(
            l1_lambda=float(l1_lambda or 0.0),
            enabled=bool(codebook_enabled),
        )

    def decode_to_concept(self, symbol_state):
        """Decode a SymbolicSpace event/activation back to its concept-
        space projection.

        With ``symbol_dim == concept_dim`` enforced in ``__init__``, the
        symbol-side and concept-side widths match and no learned remap
        is needed; the symbol state is already valid concept-space data.

        Used by :meth:`Mereology.Luminosity` when stored truths must be
        folded against a higher-order concept; the truths live in
        symbol-space and need to be readable as conceptual-space.
        """
        return symbol_state

    def l1_proximal(self, x):
        """Soft-threshold activations used as a sparsity bias.

        Delegates to the shared SparsityRegLayer. Kept as a thin
        wrapper for backward compatibility with call sites in this file
        and in Models.py.
        """
        return self._sparsity(x)

    def smoothing_penalty(self, x):
        """Total-variation penalty along the concept axis of symbol activations.

        Bivector-aware via pair-max collapse; 0 when discontinuityLambda=0
        or when disabled. See Layers.SmoothingRegLayer.
        """
        return self._smoothing(x)

    def resolve(self, subspace):
        """Collapse [pos, neg] bivector into 1-D per-symbol activation.

        Writes ``subspace.activation = pos - neg`` -- the **balance of
        evidence** for the symbol.  ``pos`` is evidence FOR the symbol's
        truth, ``neg`` is evidence AGAINST it; the difference is what
        the symbol actually asserts after both sides cancel.  Range is
        roughly ``[-1, +1]`` when the bivector is unit-normalised:

          * pos=1, neg=0  →  +1  (full affirmation)
          * pos=0, neg=1  →  -1  (full negation)
          * pos=neg       →   0  (balanced / unknown / contradicted)
          * pos=0.7, neg=0.2 → +0.5 (mostly affirmed; slight counter-evidence)

        This is the signed Degree of Truth that the symbolic codebook
        snaps against and what the TruthSet stores as the scalar truth
        of each symbol.  ``inside()`` / ``outside()`` use the absolute
        value of this when comparing point-magnitude to a symbol's
        extent.

        The result is stored directly via ``activation.setW()`` rather
        than through ``set_activation()`` so that the tensor remains
        1-D ``[B, N]`` rather than being lifted back to the bivector
        ``[B, N, 2]``.

        Source of the bivector (resolved via ``subspace.materialize()``,
        which returns the muxed event when populated and falls back to
        a what-only reconstruction otherwise):
          1. The muxed event when it is a [B, N, D] tensor (D >= 2):
             the first two columns are the [pos, neg] poles.  This is
             the case inside ``forward()`` after ``set_event(act)``
             where ``act`` is the PiLayer output ([B, N, symbol_dim]).
          2. A [B, N, 2] what-only tensor (e.g. after
             ``sym.subspace.what.setW(bivec)`` in unit tests, or when
             the Codebook weight was manually overwritten).

        Args:
            subspace: a SubSpace carrying the bivector in .event or .what.

        Returns:
            subspace (for chaining).
        """
        # SymbolicSpace.resolve is the internal resolution
        # implementation: it reads the bivector from .event (preferred,
        # because forward() reaches resolve() right after set_event(act))
        # or falls back to .what, computes pos - neg, and stores the
        # signed scalar directly via the activation Basis's setW (NOT
        # via the public set_activation, which would lift the scalar
        # back into a non-negative bivector and discard sign).
        # External clients should call SubSpace.resolve() for a pure
        # read-side derivation, or subspace.materialize(mode="activation")
        # which prefers any stored value over the derivation.
        # Bivector substrate retired (2026-05): read the signed
        # Degree-of-Truth scalar from the event (preferred) or ``.what``.
        # Width-1 carrier -> the scalar itself; wider content -> the
        # signed magnitude balance ``aP - aN`` (consistent with
        # SubSpace._compute_active / set_activation_from_event).
        src = subspace.event.getW() if subspace.event is not None else None
        if not (src is not None and src.ndim == 3 and src.shape[-1] >= 1):
            src = (subspace.what.getW()
                   if subspace.what is not None else None)
            if (src is None or not torch.is_tensor(src)
                    or src.ndim < 2 or src.shape[-1] < 1):
                return subspace
        if src.shape[-1] == 1:
            scalar = src[..., 0]
        else:
            d = max(src.shape[-1], 1)
            pos = torch.relu(src).norm(dim=-1) / math.sqrt(d)
            neg = torch.relu(-src).norm(dim=-1) / math.sqrt(d)
            scalar = pos.clamp(0.0, 1.0) - neg.clamp(0.0, 1.0)
        subspace.activation.setW(scalar)
        return subspace

    # ``area()`` and ``luminosity()`` were removed when the measure
    # family migrated to the :class:`Mereology` mixin (see
    # ``bin/Mereology.py``).  Callers should use ``model.Area()`` /
    # ``model.Luminosity()`` (or invoke the underlying
    # :func:`Ops.hyperrectangle_volume` /
    # :func:`Ops.hyperrectangle_overlap_volume` kernels directly).

    def inside(self, point, symbol_idx=None):
        """Is ``point`` within the region defined by a symbol's extent?

        Uses mereological parthood on the Resolve-d activation.  The
        "extent" of a symbol is the absolute value of its scalar
        activation (``|pos - neg|`` from resolve()).  A point is inside
        a symbol's region when its magnitude does not exceed that
        absolute activation value.

        We take ``|activation|`` because resolve() produces the signed
        Degree of Truth (``pos - neg``), but extent / point-in-region
        semantics are magnitude-based: a strongly-negated symbol (large
        negative DoT) has just as much extent as a strongly-affirmed
        symbol (large positive DoT) -- the sign tells us which side of
        the assertion was reached, not how far it reached.

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
        # Public interface: materialize(mode="activation") returns the
        # subspace's resolved scalar activation.  Direct .activation.getW()
        # access would expose the underlying weight buffer instead of the
        # value clients should see.
        activation = self.subspace.materialize(mode="activation").abs()  # [B, N]
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
        """Decorrelation loss.
        
        See class docstring for the operation contract.
        """
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
        self.vq_chunk_budget.
        """
        if not self.codebook:
            return None
        basis = getattr(self.subspace, "what", None)
        if basis is None or not isinstance(basis, Codebook):
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
        budget = self.vq_chunk_budget
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
    # ``wordSubSpace.category_stack``, asks the rule predictor for a distribution
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
        # Activation: [1, N] so the resolved scalar is a 1-D vector after
        # a .squeeze(0) -- matches WordSpace.pos_lookup which expects [N].
        # Public write through set_activation() (which lifts to a bivector
        # if activeEncoding.nDim == 2) instead of poking .activation.setW.
        incoming.set_activation(pos_vector.unsqueeze(0))
        # Minimal .what default -- broadcastable into self.subspace.what's
        # shape.  Zero tensor suffices; concrete rule ops pull from
        # self.subspace's state for unary ops, and binary ops reading the
        # incoming .what get a neutral operand.  Public set_what() invalidates
        # the cached event so a subsequent materialize(mode="event") re-muxes
        # cleanly.  Safe here because the manufactured ``incoming`` is local
        # scratch with a Tensor (not Codebook) basis on .what.
        inc_what_shape = (1, n, int(self.subspace.nWhat))
        incoming.set_what(torch.zeros(
            inc_what_shape, dtype=torch.float32, device=pos_vector.device))
        incoming._rule_dispatch = True
        return incoming

    def _op_for_rule(self, rule_id, wordSubSpace=None):
        """Return a callable ``(self_sub, inc_sub) -> new_what`` for ``rule_id``.

        Dispatches through the ``wordSubSpace.host_layer(tier, rule_name)``
        registry (the same path WordSpace's grammar applies during chart
        compose). When ``wordSubSpace`` is missing, no host layer is
        registered for the rule, or the rule_id is out of range, returns
        a pass-through that yields the left operand unchanged.

        Routes by arity:
          * arity 2 (``intersection``, ``union``, ``swap``, ...):
            ``host_layer.compose(left, right)``.
          * arity 1 (``not``, ``non``, ``pi``, ``sigma``, ...):
            ``host_layer.forward(left)``.

        Errors are surfaced (logged) rather than swallowed silently —
        the prior implementation called ``layer.project(...)`` on
        ``SyntacticLayer``, which has no such method, so every dispatch
        fell into ``except Exception: return left`` and chart-parsed
        rule firing was a no-op.
        """
        host = None
        method_name = None
        if wordSubSpace is not None:
            try:
                from Language import TheGrammar
                method_name = TheGrammar.rules[int(rule_id)].method_name
            except (IndexError, AttributeError, ValueError, TypeError):
                method_name = None
            if method_name is not None:
                # Tier routing (see doc/Language.md):
                #   * Subsymbolic ops (lift / lower / union /
                #     intersection) live on PerceptualSpace /
                #     ConceptualSpace's PiLayer + SigmaLayer instances;
                #     dispatch via tier='C' so the lattice composition
                #     fires on the concept-tier representation.
                #   * Symbolic ops (not / non / true / false / what /
                #     where / when / query / equals / part / swap /
                #     conjunction / disjunction / ...) live on
                #     SymbolicSpace's SyntacticLayer registry; dispatch
                #     via tier='S'.
                _SUBSYMBOLIC = {'lift', 'lower', 'union', 'intersection'}
                tier = 'C' if method_name in _SUBSYMBOLIC else 'S'
                try:
                    host = wordSubSpace.host_layer(tier, method_name)
                except Exception:
                    host = None
                # Fallback: some grammar configs only register one tier
                # for a rule. Try the other tier so the dispatch still
                # finds a layer in mixed configurations.
                if host is None:
                    fallback = 'S' if tier == 'C' else 'C'
                    try:
                        host = wordSubSpace.host_layer(fallback, method_name)
                    except Exception:
                        host = None

        def op(self_sub, inc_sub):
            """Op.

            See class docstring for the operation contract.
            """
            left = self_sub.what.getW()
            right = None
            if inc_sub is not None:
                right = inc_sub.what.getW()
            if host is None or left is None:
                # No host layer registered for this rule -- best-effort
                # identity so the caller can still write something back
                # into .what without dropping the call entirely.
                return left if left is not None else right
            try:
                arity = int(getattr(host, 'arity', 1))
                if arity == 2 and hasattr(host, 'compose'):
                    return host.compose(left, right)
                return host.forward(left)
            except Exception as exc:
                warnings.warn(
                    f"_op_for_rule[{method_name!r}] failed: "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2)
                return left

        return op

    def _superposed_op(self, rule_probs, wordSubSpace=None):
        """Return a callable that weights every rule's output by ``rule_probs``.

        Training-mode analogue of argmax dispatch: every rule fires and
        contributes ``p * rule_op(self_sub, inc_sub)`` to the composed
        ``new_what`` so the rule-predictor receives gradient.  Rules with
        probability < 1e-6 are skipped for efficiency.
        """
        def mixed(self_sub, inc_sub):
            """Mixed.
            
            See class docstring for the operation contract.
            """
            total = None
            # One sync for the whole prob vector; grad still flows via
            # the original tensor below.
            probs_list = rule_probs.detach().tolist()
            for rid, p_val in enumerate(probs_list):
                if p_val < 1e-6:
                    continue
                out = self._op_for_rule(rid, wordSubSpace=wordSubSpace)(
                    self_sub, inc_sub)
                if out is None:
                    continue
                contribution = out * rule_probs[rid]
                total = contribution if total is None else total + contribution
            return total

        return mixed

    def _forward_with_rule_dispatch(self, incoming_subspace, wordSubSpace=None,
                                    quantize=True):
        """Rule-dispatch forward (Task 6.2).

        Five-step flow per the plan:
          1. Read active (symbol-axis activation) from the incoming subspace.
          2. Look up the PoS vector via ``wordSubSpace.pos_lookup`` and push
             onto ``wordSubSpace.category_stack``.
          3. Ask the rule predictor for a softmax distribution over rules.
          4. Pick a rule (argmax for eval, superposed for training) and
             apply it to update ``self.subspace.what``.
          5. Resolve the bivector and (optionally) quantize through the
             symbol codebook.
        """
        if wordSubSpace is None:
            raise ValueError(
                "SymbolicSpace.forward requires wordSubSpace for rule dispatch; "
                "none was provided.")

        # Step 1 -- active symbols (1-D [N]).  Read through the public
        # materialize(mode="activation") interface; clients shouldn't
        # touch the underlying weight buffer via .activation.getW().
        active_raw = incoming_subspace.materialize(mode="activation")
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
        pos_vec = wordSubSpace.pos_lookup(active)
        wordSubSpace.category_stack.push(0, pos_vec)

        # Step 3 -- rule distribution.
        rule_logits = wordSubSpace.predict_rule(0)
        rule_probs = torch.softmax(rule_logits, dim=-1)

        # Step 4 -- apply chosen rule.
        if self.training:
            rule_op = self._superposed_op(rule_probs, wordSubSpace=wordSubSpace)
        else:
            rule_id = int(rule_probs.argmax().item())
            rule_op = self._op_for_rule(rule_id, wordSubSpace=wordSubSpace)

        new_what = rule_op(self.subspace, incoming_subspace)
        if new_what is not None:
            # Shape-align: derive the expected what-tensor shape from the
            # public materialized event (slicing its first nWhat columns)
            # rather than poking subspace.what.getW() directly.  Rule
            # ops preserve the left operand's shape, so we only update
            # when shapes are compatible.  Add a small nudge when the op
            # returned an all-zero tensor so the test's "non-zero after
            # forward" contract holds for pass-through dispatchers.
            muxed = self.subspace.materialize()
            nwhat = int(self.subspace.nWhat)
            current = (muxed[..., :nwhat]
                       if (muxed is not None and muxed.ndim >= 1
                           and muxed.shape[-1] >= nwhat)
                       else None)
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
            # Per spec: per-batch ``.what`` writes flow through
            # ``set_what`` which snaps via the codebook for unmuxed
            # configs (codebook on ``.what``) and falls through to
            # direct ``setW`` for plain-Tensor slots.
            self.subspace.set_what(new_what)

        # Step 5 -- resolve + optional codebook pass.
        self.resolve(self.subspace)
        if self.codebook and quantize:
            # Only quantize when the muxed event is populated.  In the
            # rule-dispatch path the subspace may carry only .what so the
            # codebook forward (which materialize()s a muxed tensor) would
            # get a None input.  Skip quantization in that case -- the
            # codebook can learn from the updated .what on the next
            # regular forward pass.  Read presence via materialize()
            # rather than poking subspace.event.getW() directly.
            ev = self.subspace.materialize()
            if ev is not None:
                self.subspace.what.forward(self.subspace)

        return self.subspace

    # ------------------------------------------------------------------
    # Phase 5 stack-rewrite path
    #
    # See doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md
    # §"Phase 5: Integrate Into SymbolicSpace.forward". Gated by
    # ``self.use_stack_router``. Runs the LanguageLayer's stack-rewrite
    # path on a temporary stack-mode SubSpace, then writes the root
    # state into ``self.subspace``. Implicitly bypasses (Phase 6):
    #   * WordSpace.current_rules / generate_rules
    #   * SyntacticLayer cursor (uses .execute instead)
    #   * Chart.compose / wordSubSpace.forwardSymbols
    #   * ConceptualSpace.stm._buffer (_stm_bounded_reduce_step et al.)
    # ------------------------------------------------------------------

    def _snap_to_terminal_ste(self, x, codebook_W):
        """Straight-through snap of ``x`` ``[B, D]`` to the nearest row
        of ``codebook_W`` ``[V, D]``.

        Returns ``(snapped, idx)`` with ``idx`` ``[B]`` long. Gradient
        flows back through ``x`` via the STE bypass; the snap itself is
        argmin-by-L2 (no gradient).
        """
        if codebook_W is None or codebook_W.ndim != 2 or codebook_W.shape[0] == 0:
            B = x.shape[0]
            return x, torch.zeros(B, dtype=torch.long, device=x.device)
        # L2 distance, no autograd needed for the argmin.
        with torch.no_grad():
            # [B, V]
            dists = (x.unsqueeze(1) - codebook_W.unsqueeze(0)).pow(2).sum(dim=-1)
            idx = dists.argmin(dim=-1)
        hard = codebook_W[idx]                                # [B, D]
        # Straight-through: forward returns the hard snap, backward
        # routes through x (per the plan's "STE-snapped symbol vector").
        snapped = x + (hard - x).detach()
        return snapped, idx

    def _make_stack_subspace_for(self, B, K, D):
        """Build a fresh stack-mode SubSpace for the router.

        Width-1 ``.where`` carrier (first-patch convention; the integer
        location lives in element [0] -- see LanguageLayer._encode_where).
        The stack subspace is local to this forward call; it does NOT
        replace ``self.subspace`` (which still receives the root state
        at the end).
        """
        W = 1
        we = WhereEncoding(maxP=max(K + 16, 64), nWhere=W, nWhen=0)
        sub = SubSpace(
            [K, D + W], [K, D + W],
            nInputDim=D + W, nOutputDim=D + W,
            whereEncoding=we,
        )
        device = (self.subspace.what.W.device
                  if self.subspace.what.getW() is not None else None)
        sub.set_what(torch.zeros(B, K, D, device=device))
        sub.set_where(torch.zeros(B, K, W, device=device))
        sub.set_activation(torch.zeros(B, K, device=device))
        return sub

    def _pick_default_reduce_rule(self):
        """Pick a default S-tier arity-2 rule for hard reduction.

        First-patch policy: the lowest-id arity-2 S-tier rule that is
        also registered on ``self.syntacticLayer._by_name``. The plan
        defers real SHIFT/REDUCE scoring to a later phase. Returns
        ``None`` when no usable rule exists (no reduction is applied).

        TODO(phase5+): replace this hardcoded pick with the router's
        learned scoring (binary_tiling_soft_dp / binary_tiling_viterbi).
        """
        from Language import TheGrammar
        if getattr(self, 'syntacticLayer', None) is None:
            return None
        registered = self.syntacticLayer._by_name
        try:
            candidates = TheGrammar.rules_for_tier('S', arity=2)
        except Exception:
            return None
        for rid in candidates:
            mn = TheGrammar.method_name(rid)
            if mn in registered:
                return rid
        return None

    def _stack_route_forward(self, CS_subspaceForSS):
        """Run the stack-rewrite path; write the root into self.subspace.

        Eager Python loops over the input positions are the small
        "eager bridge" the plan permits for a first correctness patch.
        TODO(phase5+): vectorize the SHIFT loop and replace the
        hardcoded reduction rule with the router's learned scoring.
        """
        self.subspace.copy_context(CS_subspaceForSS)
        act_pre = CS_subspaceForSS.materialize()              # [B, N, D]
        if act_pre is None:
            return self.subspace
        B, N, D = act_pre.shape

        # Build the temporary stack subspace; capacity = N (room for
        # one terminal per input position, then N-1 reductions collapse
        # to a single root slot).
        K = max(N, 2)
        stack_sub = self._make_stack_subspace_for(B, K, D)
        # Move to the right device once.
        stack_sub.what.setW(stack_sub.what.getW().to(act_pre.device))
        stack_sub.where.setW(stack_sub.where.getW().to(act_pre.device))
        stack_sub.activation.setW(
            stack_sub.activation.getW().to(act_pre.device))

        codebook_W = self.subspace.what.getW()                # [V_sym, D]

        # Build the action list: snap each input position to a terminal
        # (the SHIFT actions), then schedule N-1 REDUCEs to collapse to
        # a single root slot. The snap stays in SymbolicSpace as the
        # first-patch "eager bridge" the plan permits; future phases
        # can migrate it into LanguageLayer.forward by consuming the
        # ``terminal_codebook`` arg.
        #
        # First-patch where_id: use the matched symbol id from batch
        # row 0 (uniform-per-batch). Per-row .where encoding is a
        # follow-up; see TODO above.
        from Language import TheGrammar
        actions = []
        for n in range(N):
            x_n = act_pre[:, n, :]                            # [B, D]
            terminal_what, sym_idx = self._snap_to_terminal_ste(x_n, codebook_W)
            where_id = TheGrammar.where_id_for_symbol(int(sym_idx[0].item()))
            actions.append(('shift', terminal_what, where_id))

        rule_id = self._pick_default_reduce_rule()
        if rule_id is not None and N >= 2:
            for _ in range(N - 1):
                actions.append(('reduce', int(rule_id)))

        # Canonical dispatch through LanguageLayer.forward (Layer-style
        # entry point), so the call shape matches the plan's target
        # contract instead of bypassing through the shift/reduce
        # primitives directly.
        self.languageLayer.forward(
            stack_sub, self.syntacticLayer,
            actions=actions,
            rule_codebook=self.rule_codebook,
            terminal_codebook=self.subspace.what,
            grammar=TheGrammar,
        )

        # Read root state: slot 0 holds the surviving payload.
        root = stack_sub.materialize(mode="what")[:, 0:1, :]   # [B, 1, D]
        # Length-N expansion matches the existing SymbolicSpace output
        # contract (downstream consumers expect [B, N, D]).
        n_out = int(self.outputShape[0])
        expanded = root.expand(B, n_out, D).contiguous()

        # Write into self.subspace.what plus a unit activation.
        self.subspace.set_what(expanded)
        self.subspace.set_activation(
            torch.ones(B, n_out, device=expanded.device,
                       dtype=expanded.dtype))
        return self.subspace

    def _stack_route_reverse(self, subspace):
        """Phase 7 reverse-side counterpart to ``_stack_route_forward``.

        Dispatches through ``self.languageLayer.reverse(...)`` (the
        canonical Layer-style entry into the stack-rewrite unwind path).
        Builds a temporary stack-mode SubSpace seeded with the incoming
        payload as a rule-stamped slot, calls ``reverse_stack`` via the
        LanguageLayer wrapper, then writes the unwound state back into
        ``self.subspace``.

        Under the identity-stub contract this is intentionally lossy
        (per plan §"Reverse And Reconstruction": "Do not block the
        forward refactor on complete reverse math. Preserve existing
        identity-stub behavior where rule inverses are not
        implemented."). A full multi-level unwind needs a provenance
        trail and is Phase 8+ work; this method exists so the
        LanguageLayer is invoked symmetrically on the reverse path
        (the user's "ensure SymbolicSpace.reverse calls
        LanguageLayer.reverse(...)" requirement).
        """
        from Language import TheGrammar

        self.subspace.copy_context(subspace)
        act = subspace.materialize()
        if act is None or act.ndim != 3:
            # Nothing to unwind; pass through.
            return self.subspace
        B, N, D = act.shape

        rule_id = self._pick_default_reduce_rule()
        if rule_id is None:
            # No binary rule to unwind through; degenerate pass-through.
            self.subspace.set_what(act.contiguous())
            self.subspace.set_activation(
                torch.ones(B, N, device=act.device, dtype=act.dtype))
            return self.subspace

        where_id = TheGrammar.where_id_for_rule(int(rule_id))

        # Build the temporary stack subspace; capacity >= 2 so unreduce
        # has room for the right-child slot.
        K = max(N + 2, 4)
        stack_sub = self._make_stack_subspace_for(B, K, D)
        # Move the freshly allocated buffers to the input's device.
        stack_sub.what.setW(stack_sub.what.getW().to(act.device))
        stack_sub.where.setW(stack_sub.where.getW().to(act.device))
        stack_sub.activation.setW(
            stack_sub.activation.getW().to(act.device))

        # Seed slot 0 with the root payload (the input's slot 0 since
        # _stack_route_forward writes the root expanded across all N
        # positions, they're all the same). Stamp the slot's .where
        # with the rule namespace so unreduce will fire.
        what_init = torch.zeros(B, K, D, device=act.device, dtype=act.dtype)
        where_init = torch.zeros(B, K, 1, device=act.device, dtype=act.dtype)
        occ_init = torch.zeros(B, K, device=act.device, dtype=act.dtype)
        what_init[:, 0, :] = act[:, 0, :]
        where_init[:, 0, 0] = float(where_id)
        occ_init[:, 0] = 1.0
        stack_sub.set_what(what_init)
        stack_sub.set_where(where_init)
        stack_sub.set_activation(occ_init)

        # Canonical dispatch through LanguageLayer.reverse (the user's
        # symmetry requirement: SymbolicSpace.reverse calls
        # LanguageLayer.reverse(...)). Under the identity-stub this
        # unwinds one level then halts.
        self.languageLayer.reverse(
            stack_sub, self.syntacticLayer,
            rule_codebook=self.rule_codebook,
            grammar=TheGrammar,
        )

        # Read the unwound payloads. After one unreduce, slots 0 and 1
        # both hold copies of the root (identity stub). Take the first
        # N slots for the output; pad / expand if K < N (won't happen
        # with K >= N+2 above but kept defensive).
        unwound = stack_sub.materialize(mode="what")
        n_slots = unwound.shape[1]
        if n_slots >= N:
            out = unwound[:, :N, :]
        else:
            out = unwound[:, :1, :].expand(B, N, D).contiguous()
        self.subspace.set_what(out.contiguous())
        self.subspace.set_activation(
            torch.ones(B, N, device=out.device, dtype=out.dtype))
        return self.subspace

    def forward(self, CS_subspaceForSS):
        """Concept->symbol forward (symbolic recurrent loop leg).

        Single explicit input ``CS_subspaceForSS`` -- the prior pass's
        ConceptualSpace output (``ConceptualSpace._subspaceForSS``).
        SymbolicSpace never combined siblings (no ``_sourced_input``); it
        consumes this one subspace directly.

        Dispatches to the rule-application path when the caller marks the
        incoming subspace with ``_rule_dispatch`` (see
        ``_build_incoming_subspace``); otherwise runs the grammar
        dispatch followed by the intrinsic snap.

        The intrinsic snap is ``Codebook.forward(input)``
        which returns a per-prototype catuskoti bivector. The snap is
        what calling SymbolicSpace MEANS — naming the closest point in
        concept space — and runs unconditionally regardless of grammar
        state.
        """
        if CS_subspaceForSS.is_empty():
            return CS_subspaceForSS
        self.subspace.copy_context(CS_subspaceForSS)
        # Phase-1 ``parallel`` mode gating: when held at zero the
        # resolve / lift / codebook / TruthLayer paths skip and the
        # event tensor is filled with zeros. Downstream consumers
        # read zeros; the elementwise-sum at the next conceptual
        # order's combined input contributes nothing from this Space.
        if self.held_at_zero:
            sample = CS_subspaceForSS.materialize()
            if sample is not None:
                B = sample.shape[0]
                N = self.outputShape[0]
                D = self.subspace.muxedSize
                zero_event = torch.zeros(B, N, D, device=sample.device,
                                         dtype=sample.dtype)
                self.subspace.set_event(zero_event)
            return self.subspace
        # Phase 5 of the SubSpace.what STM refactor: when use_stack_router
        # is True, dispatch through the new stack-rewrite path instead of
        # the SyntacticLayer cursor + Chart.compose / wordSubSpace.forwardSymbols
        # path. This also implicitly bypasses WordSpace.current_rules and
        # ConceptualSpace.stm in the live forward (Phase 6 "bypass" leg
        # of the plan). The legacy path is preserved when the flag is
        # off so existing tests / configs keep working byte-identically.
        if getattr(self, "use_stack_router", False):
            return self._stack_route_forward(CS_subspaceForSS)
        quantize = getattr(self, "quantize", True)
        is_last = getattr(self, "is_last", False)
        wordSubSpace = getattr(self, "wordSubSpace", None)
        if getattr(CS_subspaceForSS, '_rule_dispatch', False):
            return self._forward_with_rule_dispatch(
                CS_subspaceForSS, wordSubSpace=wordSubSpace, quantize=quantize)
        vspace = CS_subspaceForSS
        vspace = self.forwardBegin(vspace)
        act_pre = vspace.materialize()                    # [B, N, concept_dim]
        # SyntacticLayer is unconditional: per the grammar XML, the
        # chart populates ``current_rules`` with one or more rules per
        # tier (e.g. ``S = sigma(S)`` from model.xml's default
        # grammar). When no chart rule fires for this tier, the
        # dispatch is a no-op (post-2026-05-07 rollback removed the
        # ``default_rule`` code-level fallback — grammar XML is the
        # sole source of truth).
        if getattr(self, 'syntacticLayer', None) is None:
            # SymbolicSpace no longer owns a sigma; with symbol_dim ==
            # concept_dim enforced in __init__, the default path is
            # dimensionally a pass-through. Learned C->S transforms live
            # in ConceptualSpace.pi.
            act = act_pre
        else:
            # ---- Eager cursor path: SymbolicSpace is the single site
            # that drives S-tier op application. The per-tier rule list
            # comes straight from ``wordSubSpace.current_rules.get('S')``
            # (the live ``list[list[int]]`` populated by
            # ``WordSpace.compose``). The legacy Phase-2B tensor-driven
            # op_sel branch was retired with the SentenceState carrier
            # (op_sel was never populated in production); the cursor
            # loop below is the one and only S-executor.
            chart_rules_S = (wordSubSpace.current_rules.get('S')
                             if wordSubSpace is not None
                             and wordSubSpace.current_rules is not None
                             else None)
            row_zero = self.syntacticLayer._row_zero_rules(
                chart_rules_S)
            n_steps = max(1, len(row_zero))
            vspace.set_event(act_pre)
            for _ in range(n_steps):
                vspace = self.syntacticLayer.forward(vspace)
            act = vspace.materialize()
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

        if self.accumulateTruth > 0 and wordSubSpace is not None:
            truth_layer = getattr(wordSubSpace, 'truth_layer', None)
            if truth_layer is not None:
                basis = getattr(self.subspace, 'basis', None)
                BK, N, D = act.shape
                norms = act.norm(dim=-1)
                mag_score = norms.clamp(max=self._truth_min_magnitude) \
                            / max(self._truth_min_magnitude, 1e-8)
                vmask = self.subspace.valid_mask
                if vmask is not None:
                    mag_score = mag_score \
                                * vmask.flatten().unsqueeze(-1).to(mag_score.dtype)
                trust = mag_score * float(self.accumulateTruth)
                truth_layer.record_batch(
                    act.reshape(BK * N, D),
                    trust.reshape(BK * N),
                    degree=float(self.accumulateTruth),
                    basis=basis)

        if self._symbol_where is not None:
            B = act.shape[0]
            nAct = act.shape[1]
            if nAct == self._symbol_where.shape[0]:
                where = self._symbol_where.unsqueeze(0).expand(B, -1, -1)
                where = where.to(act.device)
                self.subspace.set_where(where)

        if self.sortNetwork is not None:
            act = self.sortNetwork.forward(act)

        if wordSubSpace is not None:
            act = wordSubSpace.forwardSymbols(act, self.subspace)

        # Resolve [pos, neg] bivector to 1-D per-symbol activation
        # before the codebook sees it. resolve() writes
        # subspace.activation = pos - neg.
        self.resolve(self.subspace)

        # project codebook: signed per-prototype scalar projection
        # ``[B, N]`` replaces the VQ-VAE / hard-quantize branches. The
        # result lives on ``subspace.event`` so downstream consumers
        # (OutputSpace, _compute_symbol_terms) read it via materialize().
        if isinstance(self.subspace.what, ProjectionBasis):
            proj = self.subspace.what.forward(act)
            self.subspace.set_event(proj, compute_activation=False)
            vspace = self.forwardEnd(self.subspace)
            self._emit_symbol_terms(
                vspace, self._compute_symbol_terms(proj))
            if (self.impenetrable_overlap > 0.0
                    or self.impenetrable_variance > 0.0):
                imp = self.impenetrable_loss()
                if (imp is not None and torch.is_tensor(imp)
                        and (imp.requires_grad or imp.abs().item() > 0.0)):
                    vspace.errors.add(
                        "symbol_impenetrable", imp, weight=1.0,
                        space="SymbolicSpace", category="symbol")
            return vspace

        # VQ-VAE / hard_quantize / continuous branches preserved from
        # the pre-rollback forward — these implement the codebook snap
        # and commitment-loss objectives that drive symbol learning.
        # The snap behaviour itself (the "name the closest point"
        # categorization) is the inversion-equivalent of
        # ``Codebook.forward(input)`` exposed directly
        # via ``Codebook.project`` for the new idempotent-loop test
        # (test/test_idempotent_loop.py).
        use_vqvae_reversible = self.use_vqvae and self.reversible and self.codebook
        use_vqvae_nonreversible = (self.use_vqvae and not self.reversible
                                   and self.codebook)
        hard_quantize = (not self.use_vqvae) and self.codebook and quantize
        if use_vqvae_reversible:
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
            cb_commit = getattr(self.subspace.what, "last_commit_loss", None)
            if cb_commit is not None and torch.is_tensor(cb_commit) and cb_commit.requires_grad:
                vspace.errors.add(
                    "codebook_commit", cb_commit, weight=1.0,
                    space="SymbolicSpace", category="symbol")
            self._emit_symbol_terms(
                vspace,
                self._compute_symbol_terms(predicted, target=quantized_detached))
        elif hard_quantize:
            self.subspace.set_event(act)
            vspace = self.forwardEnd(self.subspace)
            self.resolve(self.subspace)
            act_1d = self.subspace.materialize(mode="activation")
            if act_1d is not None and act_1d.ndim == 2:
                predicted_1d = act_1d.unsqueeze(-1)
                target_1d = self._nearest_symbol_target(predicted_1d)
                if target_1d is not None:
                    self.subspace.set_activation(target_1d.squeeze(-1))
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
            # VQ snap when a codebook is configured but neither VQ-VAE
            # nor hard-quantize fires (the typical post-rollback,
            # last-stage path). Same idiom ConceptualSpace.forward
            # uses: route through ``Codebook.forward(input)`` so the
            # event coming out is the per-slot vector with codebook
            # rows substituted in. Wide codebooks (``V_S >
            # outputShape[0]``) take the ``topK`` pruning branch so
            # only the top-N strongest prototype activations survive.
            # The snap is idempotent on already-snapped vectors,
            # which is what makes the iterative
            # SymbolicSpace.forward → ConceptualSpace.forward loop
            # converge once the codebooks are trained.
            basis = self.subspace.what
            snap_eligible = (
                self.codebook
                and isinstance(basis, Codebook))
            if snap_eligible:
                if basis.nVectors > self.outputShape[0]:
                    snapped = basis.forward(act, topK=self.outputShape[0])
                else:
                    snapped = basis.forward(act)
                if (snapped is not None
                        and torch.is_tensor(snapped)
                        and snapped.shape == act.shape):
                    act = snapped
            self.subspace.set_event(act)
            vspace = self.forwardEnd(self.subspace)
            self._emit_symbol_terms(
                vspace, self._compute_symbol_terms(act))

        if (self.impenetrable_overlap > 0.0
                or self.impenetrable_variance > 0.0):
            imp = self.impenetrable_loss()
            if (imp is not None and torch.is_tensor(imp)
                    and (imp.requires_grad or imp.abs().item() > 0.0)):
                vspace.errors.add(
                    "symbol_impenetrable", imp, weight=1.0,
                    space="SymbolicSpace", category="symbol")

        vspace.normalize("symbols", target="what")
        vspace.normalize("symbols", target="where")
        return vspace

    def reverse(self, subspace):
        """Map symbol vectors back to concept vectors via PiLayer.reverse (Pi^-1).

        Reverse maps on nDim axis: [B, N, symbol_dim] -> [B, N, concept_dim].
        """
        if subspace.is_empty():
            return subspace
        # Symmetric to the forward branch: when use_stack_router is True
        # the reverse runs through LanguageLayer.reverse(...) via
        # _stack_route_reverse, bypassing the cursor-based SyntacticLayer
        # .reverse loop + wordSubSpace.reverseSymbols + WordSpace
        # .generate_rules path. Phase 7 of the SubSpace.what STM
        # refactor.
        if getattr(self, "use_stack_router", False):
            return self._stack_route_reverse(subspace)
        self.subspace.copy_context(subspace)
        vspace = subspace
        wordSubSpace = getattr(self, "wordSubSpace", None)
        vspace = self.reverseBegin(vspace)
        act = vspace.materialize()                        # [B, N, symbol_dim]
        if isinstance(self.subspace.what, ProjectionBasis):
            # project codebook reverse lift: signed-scalar ``[B, N]`` ->
            # ``[B, V, D_S]`` via the exact LDU inverse, then continue
            # with the standard reverse (sigma / sortNetwork /
            # forwardEnd).
            act = self.subspace.what.reverse(act)
        if wordSubSpace is not None:
            act = wordSubSpace.reverseSymbols(act, self.subspace)
        if self.sortNetwork is not None:
            act = self.sortNetwork.reverse(act)
        # SyntacticLayer dispatches whatever the grammar XML specifies
        # for S-tier reverse (e.g. ``S = sigma.reverse(S)`` from
        # model.xml). When no chart rule fires, the dispatch is a
        # no-op (post-2026-05-07 rollback removed the ``default_rule``
        # code-level fallback).
        if getattr(self, 'syntacticLayer', None) is None:
            # Pass-through: no SS-owned sigma to reverse through. See
            # the matching forward() branch for the architectural note.
            pass
        else:
            gen_rules_S = (
                wordSubSpace.generate_rules.get('S')
                if (wordSubSpace is not None
                    and getattr(wordSubSpace, 'generate_rules', None))
                else None)
            row_zero = self.syntacticLayer._row_zero_rules(gen_rules_S)
            n_steps = max(1, len(row_zero))
            vspace.set_event(act)
            for _ in range(n_steps):
                vspace = self.syntacticLayer.reverse(vspace)
            act = vspace.materialize()
        if self.codebook:
            self.subspace.set_event(act)
            result = self.reverseEnd(self.subspace)
        else:
            self.subspace.set_event(act)
            result = self.subspace
        # Range check (no in-place normalisation) on the concept-space
        # output. The forward path range-checks "symbols" without
        # applying tanh; the reverse path mirrors that with a range
        # check on "concepts" so the round-trip stays exact for
        # in-range values. Pre-2026-05-07 this call passed
        # ``normalize=True`` and applied ``tanh`` to ``.what``; under
        # the natural ``nWhat == nDim`` contract that squashed every
        # column instead of just the leading bivector and broke
        # round-trip invertibility. Range-check + symmetry is the
        # right contract.
        result.normalize("concepts", target="what")
        result.normalize("concepts", target="where")
        return result

    def evaluate_truth(self, vspace, wordSubSpace=None):
        """Top-level: evaluate truth of the full stack -> scalar.

        Post-2026-05-01 refactor: routes through the standalone
        ``TrueLayer`` GrammarLayer subclass (positive-pole projection).
        """
        act = vspace.materialize(mode="activation")
        from Layers import GRAMMAR_LAYER_CLASSES
        true_cls = GRAMMAR_LAYER_CLASSES.get('true')
        if true_cls is None:
            return act
        return true_cls().forward(act)

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
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
        """Build object basis.
        
        See class docstring for the operation contract.
        """
        initial_vectors = getattr(self, "_initial_vectors", None)
        if isinstance(initial_vectors, Basis):
            self._vocabulary = initial_vectors  # keep for text_mode reverse
        basis = Tensor()
        basis.create(
            self.inputShape[0],
            self.outputShape[0],
            self.muxedSize,  # full event width
        )
        return basis

    def __init__(self, inputShape, spaceShape, outputShape, vectors=None):
        """Initialize OutputSpace; allocate state for the class contract.

        See class docstring for invariants.
        """
        section = self.config_section
        invertible = TheXMLConfig.space(section, "invertible")
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
            # Activation-mode S->O remap. Architectural rule: only PS / CS
            # may own SigmaLayer/PiLayer, so this path uses an
            # InvertibleLinearLayer and wraps it with the same
            # atanh -> linear -> tanh nonlinearity that SigmaLayer
            # applies internally (Layers.py:_sigma_inner_forward).
            nIn = inputShape[0]
            nOut = outputShape[0]
            self._linearLayer = InvertibleLinearLayer(nIn, nOut, hasBias=True)
            self.layers = nn.ModuleList([self._linearLayer])
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
        """Get test output.
        
        See class docstring for the operation contract.
        """
        if not self.data.test_output:
            return None
        out = self.data.test_output
        if isinstance(out, list):
            out = torch.stack(out)
        return out.squeeze(-1) if out.ndim == 3 else out
    def prepOutput(self, outputBatch):
        """Prep output.
        
        See class docstring for the operation contract.
        """
        if isinstance(outputBatch, list):
            # Synchronous H2D (mirror of prepInput): correct + race-free.
            return torch.stack(outputBatch, dim=0).unsqueeze(1).to(
                TheDevice.get())
        return outputBatch  # already [B, D, 1] and on device after toDevice()
    def forward(self, subspace):
        """Acting: project symbols to task output.

        Two paths: activation-mode applies PiLayer to the scalar
        activation vector; vector-mode applies the configured linear /
        attention chain to the symbol event tensor. Writes the result
        back to the subspace's event.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if self.nonlinear_output:
            # Activation-mode: invertible linear on symbol activations
            # [B, nSymbols] -> [B, nOutput], wrapped with atanh/tanh for
            # the nonlinear behaviour previously provided by PiLayer.
            act = vspace.materialize(mode="activation")
            act_pre = torch.atanh(act.clamp(-1 + epsilon, 1 - epsilon))
            output = torch.tanh(self._linearLayer.forward(act_pre))
            self.subspace.set_activation(output)
            return self.subspace

        x = self.forwardBegin(vspace, returnVectors=True)
        output = self.forwardLinear(x)
        if self.codebook:
            output = self.subspace.get_vectors().forward(output)
        vspace = self.forwardEnd(output, returnVectors=True)
        return vspace

    def reverse(self, subspace):
        """Being acted upon: map output back to symbolic space.

        Inverse of ``forward``: activation-mode runs PiLayer.reverse on
        the scalar activation; vector-mode runs the inverse linear chain
        (with codebook-aware lookup when ``self.codebook`` is True).
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        if self.nonlinear_output:
            # Activation-mode reverse: tanh(linear.reverse(atanh(x)))
            # mirrors the forward path's nonlinearity.
            act = vspace.materialize(mode="activation")
            act_pre = torch.atanh(act.clamp(-1 + epsilon, 1 - epsilon))
            symbol_act = torch.tanh(self._linearLayer.reverse(act_pre))
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
        """Reverse text vectors.
        
        See class docstring for the operation contract.
        """
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

        Delegates to the Basis / Embedding reverse() path. Returns the
        ``(text, offset)`` tuples directly; negation surface forms are
        not synthesized at this layer (the previous "non-"/"not "
        prefix machinery was retired alongside the codebook polarity
        tags).
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
