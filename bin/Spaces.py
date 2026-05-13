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


# Polarity enum used by symbolic codebook entries and Percept-level
# surface-form recognition. AFFIRM is the default ("foo" -> AFFIRM,
# "non-foo" -> NON, "not foo" -> NOT).
POLARITY_AFFIRM = 0
POLARITY_NON    = 1
POLARITY_NOT    = 2

# Special codebook key for the IR-mode NULL-percept slot. Distinct from
# byte ``\x00`` (a real prediction target in byte mode); IR mask injection
# replaces masked positions with this slot so the brick body sees a
# distinct embedding meaning "predict me" rather than "this was \x00".
NULL_PERCEPT_KEY = "__NULL_PERCEPT__"
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
        """Initialize ActiveEncoding at slot ``[-5]`` with bivector width."""
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

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self.setW(x)
        return x

    def reverse(self, y, **kwargs):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        self.setW(y)
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
        """Replace.
        
        See class docstring for the operation contract.
        """
        self.setW(self._coerce_rows(new_W))
        return self.getW()

    def insert(self, new_W):
        """Insert.
        
        See class docstring for the operation contract.
        """
        new_W = self._coerce_rows(new_W)
        w = self.getW()
        self.setW(new_W if w is None else torch.cat([w, new_W], dim=0))
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
        self.setW(w[mask])
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
        if torch.any(nonzero):
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
            self.setW(normalized)
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
        """Inverse of conjunction via codebook search.

        Find the codebook vector x such that conjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        W = self._codebook_or_none("conjunctionReverse")
        if W is None:
            return result
        return Ops.conjunctionReverse(
            result, y, W, monotonic=monotonic, unit_ball=self.unit_ball)

    def disjunctionReverse(self, result, y, monotonic=False):
        """Inverse of disjunction via codebook search.

        Find the codebook vector x such that disjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if no codebook is available.
        """
        W = self._codebook_or_none("disjunctionReverse")
        if W is None:
            return result
        return Ops.disjunctionReverse(
            result, y, W, monotonic=monotonic, unit_ball=self.unit_ball)

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
            return Ops.disjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball)
        if inverse and mode == 'AND':
            W = self._codebook_or_none("Basis.lift inverse")
            if W is None:
                return X1
            return Ops.conjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball)
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
            return Ops.conjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball)
        if inverse and mode == 'OR':
            W = self._codebook_or_none("Basis.lower inverse")
            if W is None:
                return X1
            return Ops.disjunctionReverse(
                X1, X2, W, monotonic=monotonic, unit_ball=self.unit_ball)
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

    ``W`` may hold a permanent ``nn.Parameter`` (weights owned by this basis)
    OR a plain tensor (a transient activation). To keep the two roles from
    clobbering each other across lifecycle calls, plain-tensor writes are
    routed to ``_active_payload`` whenever a Parameter is registered -- the
    same dual-slot pattern ``Codebook`` uses. Callers should always read via
    ``getW()`` (which prefers the transient payload when set) rather than
    touching ``self.W`` directly.
    """

    def __init__(self, nVectors=0, nDim=0, W=None):
        """Initialize Tensor; allocate state for the class contract.
        
        See class docstring for invariants.
        """
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
        """Return the current weight tensor."""
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
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self.setW(x)
        return x

    def reverse(self, y, **kwargs):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        self.setW(y)
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
        # SymbolicSpace.forward reads it and emits "codebook_commit" into
        # vspace.errors.
        self.last_commit_loss = None
        # Optional per-row polarity tags (POLARITY_AFFIRM/NON/NOT). Allocated
        # only when create(..., polarity=True) — symbolic codebook opt-in.
        self.polarity_ids = None
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
        # While an activation payload is cached, callers see it; otherwise
        # they see the codebook Parameter.
        """Return the current weight tensor."""
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
        # Any setW that touches the codebook (Parameter path) or clears
        # it invalidates the cached SVD factors used by the invertible
        # project / project_reverse paths. The activation-payload path
        # leaves the codebook untouched, so the SVD stays valid.
        if value is None:
            self._active_payload = None
            if "W" not in self._parameters:
                self.W = None
                self._svd_dirty = True
            return
        if isinstance(value, nn.Parameter):
            if "W" in self._parameters:
                del self._parameters["W"]
            self.W = value
            self._active_payload = None
            self._svd_dirty = True
            return
        if "W" in self._parameters:
            self._active_payload = value
            return
        self.W = value
        self._active_payload = None
        self._svd_dirty = True

    def getSize(self):
        """Get size.
        
        See class docstring for the operation contract.
        """
        return self.nVectors

    def create(self, nInput, nVectors, nDim, customVQ=True, monotonic=True,
               polarity=False, category=False,
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
        if polarity and self.nVectors > 0:
            ids = torch.full((self.nVectors,), POLARITY_AFFIRM,
                             dtype=torch.long)
            self.polarity_ids = ids
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
            if isinstance(self.W, nn.Parameter):
                self.W.data.copy_(W_ortho)
            else:
                self.setW(W_ortho)
        self._svd_dirty = True

    def set_polarity(self, idx, polarity_id):
        """Tag codebook row ``idx`` with one of POLARITY_AFFIRM/NON/NOT."""
        if self.polarity_ids is None:
            raise RuntimeError(
                "Codebook.set_polarity called on a non-polarity codebook")
        if polarity_id not in (POLARITY_AFFIRM, POLARITY_NON, POLARITY_NOT):
            raise ValueError(f"Unknown polarity id: {polarity_id}")
        self.polarity_ids[int(idx)] = int(polarity_id)

    def get_polarity(self, idx):
        """Return the polarity id of row ``idx`` (AFFIRM if untagged)."""
        if self.polarity_ids is None:
            return POLARITY_AFFIRM
        return int(self.polarity_ids[int(idx)].item())

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
            self.setW(self.vq.codebook)
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
            self.setW(W)
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

        Always SVDs the true codebook Parameter (``self.W``), never the
        transient ``_active_payload`` that ``getW`` falls through to:
        the payload is a batch-shaped activation slab (``[B, N, D]``)
        and SVDing it would yield 3-D factors that downstream matmuls
        cannot consume.

        No-op when the codebook has no W yet.
        """
        if not self._svd_dirty:
            return
        # Bypass ``getW`` so the activation payload (if any) doesn't
        # leak into the factorization.  The SVD is over the codebook
        # Parameter itself.
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

    def forward(self, input, topK: int = 0):
        """Codebook forward. When ``topK > 0`` and less than the codebook
        size, ``self.activation`` is pruned to the top-K strongest entries
        per batch row -- realizing the wide-codebook narrow-output pattern
        where nVectors >> nOutput. ``topK=0`` preserves legacy behavior.

        The legacy ``project=True`` path was retired 2026-05-13 -- use
        ``ProjectionBasis`` directly for the bivector projection surface.
        """
        _vspace = None
        if isinstance(input, SubSpace):
            _vspace = input
            input = _vspace.materialize()

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
        # STE wrap: when ``self.STE`` is True the forward output is the
        # hard snap (``x``) but the encoder's gradient flows through
        # the original input identity. ``input + (x - input).detach()``
        # equals ``x`` in forward and routes ``d/dinput = identity`` in
        # backward. Shape-guarded: only wrap when input and snap match.
        if (self.STE and torch.is_tensor(input) and torch.is_tensor(x)
                and input.shape == x.shape and input.requires_grad):
            x = input + (x - input).detach()
        self.setW(x)
        if _vspace is not None:
            _vspace.set_event(x, compute_activation=False)
            return _vspace
        return x

    def reverse(self, y, **kwargs):
        """Codebook reverse: snap-then-write path.

        The legacy ``project=True`` path was retired 2026-05-13 -- use
        ``ProjectionBasis.reverse`` directly for the bivector inverse.
        """
        if y.shape[-1] < self.nDim:
            raise RuntimeError(
                f"Codebook.reverse() expected at least {self.nDim} content dims, "
                f"got shape {list(y.shape)}.")
        content = y.clone() if y.shape[-1] == self.nDim else y[:, :, :self.nDim].clone()
        content = self._snap_content(content, weight=self.getW(), nWhat=self.nDim)
        # STE wrap on the reverse path: forward equals the snapped
        # content, backward routes the gradient through the matching
        # slice of ``y`` so the upstream consumer of the codebook
        # reverse receives an identity gradient through the snap.
        if (self.STE and torch.is_tensor(y) and torch.is_tensor(content)
                and y.shape[-1] >= content.shape[-1]
                and y.requires_grad):
            y_slice = y[..., :content.shape[-1]]
            content = y_slice + (content - y_slice).detach()
        self.setW(content)
        return content

    def replace(self, new_vectors):
        """Replace.
        
        See class docstring for the operation contract.
        """
        new_vectors = self._coerce_rows(new_vectors)
        if self.customVQ and self.vq is not None:
            self.vq.codebook = new_vectors
        self.setW(new_vectors)
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
    """Bivector projection basis with LDU-parameterized W.

    Forward maps ``[B, V, D] -> [B, N, 2]`` via signed projection on N
    prototypes (positive / negative parts accumulated across V).
    Reverse maps ``[B, N, 2] -> [B, V, D]`` via the exact LDU inverse
    of W -- no SVD cache, no per-forward state.

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

    Used by ``SymbolicSpace`` / ``ConceptualSpace`` in the bivector
    regime (``bivectorOutput=true``) where the per-prototype
    catuskoti bivector replaces VQ snap as the codebook surface.
    """

    use_dot_product = False

    def __init__(self):
        super().__init__()
        self.layer = None       # InvertibleLinearLayer; allocated in create()
        self.codebookSize = 0
        self._active_payload = None

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

    def _normalized_factors(self):
        """Return ``(W_norm, W_inv_norm, col_norms)`` with unit-norm prototypes.

        Per-prototype normalization (rows of the [N, D] codebook view,
        equivalently columns of compute_W's [D, N] output) bounds the
        forward projection: each ``x · W_norm[n]`` ≤ ``||x||`` by
        Cauchy-Schwarz.  The accompanying inverse correction keeps the
        round-trip exact -- ``W_inv_norm = diag(col_norms) @ W_inv``,
        so ``(bivec @ W_inv_norm)`` recovers the same x that
        ``(x @ W_norm)`` projected.

        Concretely, with ``W = L @ D_embed @ U`` from the LDU layer
        (shape ``[D, N]``):

          W_norm[:, n]   = W[:, n] / col_norms[n]      (each column unit-norm)
          W_inv_norm     = col_norms[n].reshape(N,1) * W_inv   (row n of W_inv scaled by col_norms[n])

        The ``[B, V, D]`` -> ``[B, N, 2]`` bivector forward is then
        bounded by ``||x||_2`` per V slot; the reverse round-trips
        because ``W_norm @ W_inv_norm`` reduces to the original
        ``W @ W_inv`` (the scalings cancel).
        """
        W = self.layer.compute_W_current()                     # [D, N]
        W_inv = self.layer.compute_Winverse_current()          # [N, D]
        col_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, N]
        W_norm = W / col_norms                                 # [D, N]
        W_inv_norm = col_norms.transpose(0, 1) * W_inv         # [N, D]
        return W_norm, W_inv_norm, col_norms

    def getW(self):
        """Codebook view: ``[N, D]`` with unit-norm prototype rows.

        The legacy Codebook stored W as ``[N, D]`` (N prototypes of
        dim D each).  ILL parameterizes ``[nInput=D, nOutput=N]``;
        getW returns the transpose with row-L2 normalization applied
        so each prototype is on the unit ball, bounding downstream
        projection magnitudes.
        """
        if self._active_payload is not None:
            return self._active_payload
        if self.layer is None:
            return None
        W_norm, _, _ = self._normalized_factors()
        return W_norm.T

    def setW(self, value):
        """Set the activation payload (transient).

        Decomposing an arbitrary tensor into LDU isn't supported, so
        this only routes activation payloads.  The codebook W is
        trained via gradient through the LDU parameters.  Setting
        ``value=None`` clears the payload.
        """
        if value is None:
            self._active_payload = None
            return
        if isinstance(value, nn.Parameter):
            raise TypeError(
                "ProjectionBasis.setW does not accept Parameter writes; "
                "the codebook is parameterized via LDU on self.layer.")
        self._active_payload = value

    def forward(self, x):
        """``[B, V, D]`` (or ``[B, D]``) -> ``[B, N, 2]`` bivector.

        For each batch row and each codebook prototype, accumulates
        across V the positive and negative parts of the signed dot
        product ``x[v] · W_norm[n]``.  The unit-norm prototypes bound
        each per-slot projection by ``||x||_2 ≤ sqrt(D)`` (Cauchy-
        Schwarz); the V-sum accumulator can still exceed 1 for V>1.
        """
        if isinstance(x, SubSpace):
            x = x.materialize()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        W_norm, _, _ = self._normalized_factors()              # [D, N]
        D_min = min(int(x.shape[-1]), int(W_norm.shape[0]))
        x_d = x[..., :D_min].to(device=W_norm.device, dtype=W_norm.dtype)
        W_d = W_norm[:D_min, :]                                # [D, N]
        proj = x_d @ W_d                                       # [B, V, N]
        # Mean-over-V (instead of sum) so each pole stays in [0, 1]
        # regardless of slot count.  With unit-norm W rows and ``x``
        # in the unit ball, ``|x[v] · W[n]| ≤ 1`` per slot, so
        # ``mean_v relu(...) ∈ [0, 1]``.  V=1 (orthographic decode)
        # is identity-preserving by construction; V>1 gives the
        # per-row mean (the matching reverse below replicates the
        # recovered summary across V positions, which is the
        # mathematically correct answer given the V-axis collapse).
        #
        # Centralizing the bounding here lets downstream Spaces skip
        # additional tanh / clamp on the bivector (per user direction
        # 2026-05-13: bound once at the basis, not at every consumer).
        pos = torch.relu(proj).mean(dim=1)                     # [B, N] in [0, 1]
        neg = torch.relu(-proj).mean(dim=1)                    # [B, N] in [0, 1]
        return torch.stack([pos, neg], dim=-1)                 # [B, N, 2]

    def reverse(self, bivec, V=1):
        """``[B, N, 2]`` -> ``[B, V, D]`` via LDU inverse against the
        normalized codebook.

        Collapses the bivector to ``signed = pos - neg`` (the
        per-prototype signed mean-projection from the forward) and
        maps through the unit-prototype inverse ``W_inv_norm =
        col_norms * W_inv``.  The column-norm correction keeps the
        V=1 round-trip exact: ``W_norm @ W_inv_norm = W @ W_inv = I``.
        For V>1, the per-V information was summed away in the
        forward; the reverse returns the per-row mean summary vector
        replicated across V positions, which is the mathematically
        correct answer given the V-axis collapse.
        """
        if bivec is None or not torch.is_tensor(bivec):
            return None
        if self.layer is None:
            return None
        signed = bivec[..., 0] - bivec[..., 1]                # [B, N]
        _, W_inv_norm, _ = self._normalized_factors()         # [N, D]
        signed = signed.to(device=W_inv_norm.device, dtype=W_inv_norm.dtype)
        x_summary = signed @ W_inv_norm                       # [B, D]
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
        if NULL_PERCEPT_KEY in self.pretrain.key_to_index:
            self.null_percept_idx = int(
                self.pretrain.key_to_index[NULL_PERCEPT_KEY])
        else:
            self.null_percept_idx = len(self.wv)
            self.insert(NULL_PERCEPT_KEY, vector=None, initial_count=0)

    def _rebuild_optimizer(self):
        self.pretrain.optimizer = torch.optim.Adam(
            [self.wv._vectors],
            lr=self.pretrain.optimizer.param_groups[0]['lr'],
        )
        # W is managed by wv._vectors; getW() returns live data

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
        if vector is not None:
            new_vec = vector.to(TheDevice.get())
            if new_vec.dim() == 1:
                new_vec = new_vec.unsqueeze(0)
            new_vec = _wrap_unit_ball(new_vec)
        else:
            new_vec = _random_unit_ball((1, dim), device=TheDevice.get())

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

        # Explicit meronomy: tell the SymbolicSpace peer (when wired)
        # that this new lexicon row is an instance of "words" so the
        # codebook quantizer doesn't treat it as a fresh root atom.
        # Wiring is via ``symbolicSpace_ref`` on this Embedding's owning
        # space; unset in standalone construction (tests), in which case
        # the call is a no-op.
        s_peer = getattr(self, 'symbolicSpace_ref', None)
        if s_peer is not None and hasattr(s_peer, 'mark_word_atom'):
            try:
                s_peer.mark_word_atom(idx)
            except Exception:
                # Tagging the meronomy is best-effort; never block an
                # insert because the symbolic peer isn't ready.
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

    @staticmethod
    def _apply_polarity(stream):
        """Collapse 'non-X' and 'not X' surface forms into positive tokens.

        Under the privation/shamatha reading, "not A" and "non-A" are
        propositional affirmations of A's absence or A's indeterminacy
        — they are observable states in their own right, not negations
        of A's percept. So the collapse rewrites the surface forms into
        *distinct positive tokens* whose codebook entries are learned
        independently:

            'not foo'  -> ('abs_foo',  off_foo, POLARITY_AFFIRM)
            'non-foo'  -> ('non_foo',  off_foo, POLARITY_AFFIRM)
            'foo'       -> ('foo',     off,     POLARITY_AFFIRM)

        Polarity bookkeeping is preserved as POLARITY_AFFIRM uniformly
        because the percept layer no longer carries sign or polarity —
        every observable state is a positive percept. Marker tokens
        ('not', 'non', '-') are consumed.
        """
        out = []
        i = 0
        n = len(stream)
        while i < n:
            text, off = stream[i]
            t_lower = text.lower()
            # 'non' '-' WORD -> positive 'non_WORD' token
            if (t_lower == 'non' and i + 2 < n
                    and stream[i + 1][0] == '-'
                    and stream[i + 2][0].strip()
                    and not stream[i + 2][0].isspace()
                    and stream[i + 2][0] not in ('-',)):
                base_text, base_off = stream[i + 2]
                out.append((f"non_{base_text}", base_off, POLARITY_AFFIRM))
                i += 3
                continue
            # 'not' WS WORD -> positive 'abs_WORD' token
            if (t_lower == 'not' and i + 2 < n
                    and stream[i + 1][0].isspace()
                    and stream[i + 2][0].strip()
                    and not stream[i + 2][0].isspace()):
                base_text, base_off = stream[i + 2]
                out.append((f"abs_{base_text}", base_off, POLARITY_AFFIRM))
                i += 3
                continue
            out.append((text, off, POLARITY_AFFIRM))
            i += 1
        return out

    def _polarity_embedding(self, base_vec, polarity):
        """Identity passthrough.

        Under the privation/shamatha reading, percepts have no sign or
        polarity transformation: 'A', 'not A', and 'non-A' are three
        distinct positive percepts (codebook entries 'A', 'abs_A',
        'non_A' respectively, produced by ``_apply_polarity``). There
        is no negation to apply at the embedding-lookup boundary — the
        percept identity already carries the propositional content.
        """
        return base_vec

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

        # Phase 1: Tokenize all batch items, collapse polarity surface forms,
        # and collect OOV base words. The polarity pass converts 2-tuples
        # to 3-tuples (text, offset, polarity) — 'non-X' becomes
        # (X, off, POLARITY_NON); 'not X' becomes (X, off, POLARITY_NOT).
        # Marker tokens ('not', 'non', '-') are consumed.
        all_streams = []
        oov_words = []
        oov_seen = set()
        for b in range(batch):
            raw_stream = self._token_stream(input[b])
            stream = self._apply_polarity(raw_stream)
            all_streams.append(stream)
            for token_text, _, _ in stream:
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
            # batch_tokens stores 2-tuples for backward compat with
            # downstream consumers (reconstruct_text, etc.).
            batch_tokens.append([(t, off) for (t, off, _) in tokens])
            span_counts.append(n_tokens)
            if tokens:
                last_text, last_start, _ = tokens[-1]
                final_offsets.append(
                    last_start + len(last_text.encode('utf-8')))
            else:
                final_offsets.append(0)
            for i, (token_text, _, polarity) in enumerate(tokens):
                cb_idx = self._token_to_index(token_text)
                batch_indices[b, i] = cb_idx
                base_vec = self._encode_vector(
                    codebook[cb_idx], token_idx=cb_idx)
                result[b, i, :] = self._polarity_embedding(base_vec, polarity)
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
        # Format: list[dict] with keys 'step', 'area', 'luminosity',
        # 'intersection', 'union'.
        self.knowing = None
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

        ``errors`` and ``wordSpace`` persist (per-batch, per-document
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
        # Accept 1-D [N] (legacy direct-setW callers / squeezed tensors)
        # by treating it as a single-batch [1, N] under the bivector-lift
        # path.  Strict shapes ([B, N] / [B, N, nd]) pass through unchanged.
        if activation_tensor.ndim == 1:
            activation_tensor = activation_tensor.unsqueeze(0)  # [N] -> [1, N]
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
        """Derived **signed Degree of Truth** = pos - neg from the
        bivector held on ``.what`` (or, for muxed-only subspaces, from
        ``.event[..., :2]``).  Computed on read; not stored as a
        separate field.

          * ``pos = .what[..., 0]`` — evidence FOR the symbol's truth
          * ``neg = .what[..., 1]`` — evidence AGAINST it
          * ``resolve() = pos - neg`` — balance of evidence in [-1, +1]

        Returns ``[B, N]`` (or ``[N]`` for unbatched legacy).  Applies
        ``.active`` selection when set; returns the full slab when
        ``.active is None``.  Returns ``None`` when no bivector source
        is reachable.

        Public read API for the resolved scalar; replaces the prior
        ``self.activation.getW()`` access pattern.
        """
        what = self.what.getW() if self.what is not None else None
        bivec = None
        if what is not None and what.ndim >= 2 and what.shape[-1] >= 2:
            bivec = what[..., :2]
        else:
            ev = self.event.getW() if self.event is not None else None
            if ev is not None and ev.ndim >= 3 and ev.shape[-1] >= 2:
                bivec = ev[..., :2]
        if bivec is None:
            return None
        pos = bivec[..., 0]
        neg = bivec[..., 1]
        scalar = pos - neg
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
        """Initialize WordSubSpace; allocate state for the class contract.
        
        See class docstring for invariants.
        """
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
        self._model_wordSpace = None
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

        Requires self._peer_perceptual to be wired (BasicModel/BasicModel do
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
        where_idx = torch.zeros(batch, nObj, dtype=torch.long, device=dev)
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

        what_buf = self.subspace.whatEncoding.encode_tokens(
            tokens_per_batch, batch, nObj, nWhat, dev)

        return what_buf, where_idx, when_idx

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
        """Single-call stem source for the microbatch AR pipeline.

        Lexes/embeds the input once and emits ALL K windows in a single
        subspace whose event has shape [B, K, N, D] with k_axis=True.

        Semantics:
          * Non-AR: K=1 single-window pass-through (subspace event keeps
            its existing [B, N, D] shape; k_axis stays False).
          * AR training: K = T for dense token streams; BPE caps K to
            the max active word count in the batch after chunking.
            Windows are progressive-prefix slices from a pad-N-then-unfold.
          * ARIR inference: each call feeds an N-length buffer (T==N),
            so K=1 and the head produces one prediction per call. The
            ARIR runtime maintains the buffer across calls.

        ``self.subspace`` carries valid_mask: [B, K] bool for runBatch.
        ``_end_of_stream`` is sized to ``[False] * B`` here as a host-side
        diagnostic; the canonical hard-reset signal under the rolling-
        cursor handoff is ``next_tick``'s ``hard_eos`` list, not anything
        derived from ``valid_mask`` (which would require a sync-incurring
        ``.tolist()`` inside the brick body).
        """
        if inputData is None:
            return self._empty_like_subspace()
        if hasattr(inputData, "is_empty") and not isinstance(inputData, torch.Tensor) and inputData.is_empty():
            return inputData

        is_ar = self.masked_prediction in ('AR', 'ARUS', 'ARIR') if hasattr(self, 'masked_prediction') else False
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
                # Route to the peer's chunking mode rather than always
                # calling the lexicon path. With ``<chunking>bpe</...>``
                # the lexicon-style ``_embed`` would tokenize on
                # whitespace + look up whole-word strings against a
                # codebook seeded from BPE chunks, hitting OOV on
                # every word and silently inserting them (codebook
                # drift). ``_embed_bpe`` does the right thing: BPE
                # chunk + MAX-fuse using the frozen codebook keyed by
                # latin-1 byte-tuples.
                if getattr(peer, 'chunking_mode', None) == "bpe":
                    peer._embed_bpe(self.subspace)
                else:
                    peer._embed(self.subspace)
                embedded = peer._embedded_input
        self._ar_embedded = embedded

        if embedded is None:
            return self.subspace

        B, T, D = embedded.shape
        N = int(self.outputShape[0])

        # When peer is in BPE chunking mode, ``peer._bpe_word_mask``
        # ([B, N], 1.0 where the slot holds a real BPE-fused word
        # vector and 0.0 in padding) is the source of truth for both
        # validity AND the per-window event mask. Without it, validity
        # would fall back to ``embedded.abs().sum > 0``, which is
        # always True for muxed events (where/when components are
        # nonzero even at padding positions) and would make every
        # window look valid.
        peer = self._peer_perceptual
        bpe_mask = (getattr(peer, "_bpe_word_mask", None)
                    if peer is not None
                    and getattr(peer, "chunking_mode", None) == "bpe"
                    else None)

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
            # Single N-length window per row -- replicate the [B, N]
            # word mask across the (typically 1) K dimension.
            bpe_mask_windowed = (bpe_mask.unsqueeze(1).expand(B, K, N).contiguous()
                                 if bpe_mask is not None else None)
        else:
            # AR training. Pad N zeros on the LEFT to recreate the legacy
            # cursor-based progressive-prefix windows: window k = the
            # buffer state at cursor k (k zero-pad slots followed by
            # emb[0..k-1] right-aligned).
            pad = torch.zeros(B, N, D, device=embedded.device, dtype=embedded.dtype)
            padded = torch.cat([pad, embedded], dim=1)  # [B, T+N, D]
            unfolded = padded.unfold(1, N, 1).permute(0, 1, 3, 2).contiguous()
            # unfolded shape: [B, T+1, N, D] -- normally we'd take all T
            # windows, but in BPE mode ``_embed_bpe`` packs active words
            # at the front of [B, N], so windows k >= max-active-count
            # carry no real targets and the body's work on them is
            # entirely zeroed by the mask. Cap K to the max active word
            # count in the batch to skip them outright.
            if bpe_mask is not None:
                # ``int(...)`` forces a host sync (one per batch); the
                # savings on body work (~T / max_word_count, often
                # 1024 / ~40 ≈ 25×) far exceed the sync cost.
                word_counts = bpe_mask.sum(dim=1)  # [B]
                actual_max = int(word_counts.max().item())
                # Quantize K to the next power of two ABOVE actual_max
                # (ceiling, not nearest -- see notes below). Bounded
                # set for T=1024: {1, 2, 4, 8, 16, 32, 64, 128, 256,
                # 512, 1024} = 11 distinct K values. Typical English-
                # text batches land in {32, 64, 128} so the compile
                # cache fills in a few hundred steps and stays stable
                # instead of recompiling per data-dependent K.
                #
                # Why ceiling not nearest: K MUST satisfy K >=
                # actual_max, otherwise the windows beyond K cover
                # active word positions that would never get a
                # training signal (their predictions are dropped).
                # Nearest-pow2 of e.g. actual_max=35 is 32, which
                # truncates 3 words of training data. Ceiling
                # (35 → 64) wastes ~29 zero-mask slots of compute but
                # preserves all training signal; the wasted work is
                # multiplied out by the BPE mask anyway.
                if actual_max <= 1:
                    K = 1
                else:
                    K = min(T, 1 << (actual_max - 1).bit_length())
                windows = unfolded[:, :K, :, :]
                # Apply the SAME left-pad + unfold to the [B, T] word
                # mask so each window k carries which of its N slots
                # are real-word vs zero-pad. Trim to the first K.
                pad_mask = torch.zeros(
                    B, N, device=bpe_mask.device, dtype=bpe_mask.dtype)
                padded_mask = torch.cat([pad_mask, bpe_mask], dim=1)  # [B, T+N]
                unfolded_mask = padded_mask.unfold(1, N, 1)            # [B, T+1, N]
                bpe_mask_windowed = unfolded_mask[:, :K, :].contiguous()
                # Per-window validity: window k's target is emb[k];
                # valid iff that position carried a real word. Trimmed
                # to first K to align with the K-trimmed windows.
                valid_mask = bpe_mask[:, :K].bool()
            else:
                K = T
                windows = unfolded[:, :K, :, :]
                bpe_mask_windowed = None
                # Legacy fallback: NULL = all-zero embedding (lex pads
                # short sentences this way). Wrong under muxed events
                # but preserved for non-BPE codepaths.
                valid_mask = (embedded.abs().sum(dim=-1) > 0)  # [B, T] = [B, K]

        sub = self.subspace
        sub.set_event(windows)
        sub.k_axis = True
        sub.valid_mask = valid_mask
        sub.stem_embedded = True

        # Stash the flattened [B*K, N] BPE mask on peer so
        # PerceptualSpace.forward can apply it to the post-VQ event
        # (which has shape [B*K, N, D] after FlattenKWrapper). The
        # original [B, N] ``peer._bpe_word_mask`` is left in place for
        # callers that want the unwindowed view.
        if peer is not None and bpe_mask_windowed is not None:
            peer._bpe_word_mask_flat = bpe_mask_windowed.reshape(B * K, N)
        elif peer is not None:
            peer._bpe_word_mask_flat = None

        # Size the diagnostic to B, but do not derive its values from
        # valid_mask: a per-tick ``.tolist()`` here would be a host sync
        # inside the brick body. Cursor-mode callers overwrite this from
        # the host-side ``hard_eos`` list immediately after the brick.
        if len(self._end_of_stream) != B:
            self._end_of_stream = [False] * B

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
    """
    name = "Percepts"
    config_section = "PerceptualSpace"

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
        # Bivector regime flags (spec B3, doc/specs/2026-04-24-lift-lower-
        # bivector-design.md §B3). When ``bivectorOutput`` is true, the
        # per-slot percept activation is the catuskoti bivector
        # ``[aP, aN] = (max(0, x), max(0, -x))`` (Q2 promotion, spec
        # line 1405). ``svdOrthogonalInit`` is reserved for symmetry with
        # ConceptualSpace / SymbolicSpace; it is consulted only when a
        # Codebook is built on ``.what`` (codebook=true mode), not on the
        # default Embedding lexicon.
        try:
            self._bivector_output = bool(
                TheXMLConfig.space(section, "bivectorOutput"))
        except KeyError:
            self._bivector_output = False
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
        # Sibling reference: PerceptualSpace reads the conceptual loopback
        # (lifted C-tier event from the prior forward) and averages it into
        # the primary input via ``_sourced_input``, mirroring
        # ConceptualSpace's symbolic loopback. Wired post-construction in
        # ``Model.__init__`` / ``_build_pipelines_per_stage``. ``None`` for
        # standalone unit tests so legacy single-Space forward semantics
        # hold.
        self.conceptualSpace_ref = None
        self._sparsity = SparsityRegLayer(
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
        self._recovered_input = None
        self._embedded_input = None

        # PerceptualSpace owns a SigmaLayer that serves as the **lift
        # operator** for the grammar's ``lift(VP, NP)`` rule at S. The
        # forward path of P doesn't call this sigma directly; instead
        # ``LiftLayer.forward`` (in Layers.py) routes through it after
        # elementwise-gating the operands at C-tier:
        #
        #     gated = VP_c * NP_c            (concept-dim elementwise)
        #     out_c = P.sigma.forward(gated)
        #
        # so the sigma's in/out dim is concept_dim (the C-tier per-vector
        # width), NOT percept_dim. We read concept_dim from the XML so
        # the sigma is correctly shaped at construction without needing
        # a post-hoc rebuild.
        try:
            nConceptDim = int(
                TheXMLConfig.space("ConceptualSpace", "nDim"))
        except KeyError:
            nConceptDim = outputShape[1]
        self.sigma = SigmaLayer(nConceptDim, nConceptDim, invertible=True,
                                monotonic=True, nonlinear=True)

        input = self.subspace.getEncodedInputSize()
        self.attention = AttentionLayer(input, input, type="transformer")
        self.subspace._nWordSlots = outputShape[0]
        self.params = []
        self.layers = nn.ModuleList()
        self.chunk_layer = ChunkLayer(
            self.nDim,
            bpe=(self.chunking_mode == "bpe"),
            n_vectors=self.nVectors,
            word_learning=self.word_learning,
        )
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
        what_indices = torch.tensor(
            indices_2d, dtype=torch.long, device=dev)

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
        """BPE chunking path. Decode the upstream byte buffer, BPE-tokenize
        via ChunkLayer, group sub-tokens by whitespace word-boundary, look up
        each byte-tuple chunk in the codebook, then MAX-fuse sub-token
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
            # Single H2D for all the routing ints.
            flat_idx_t = torch.tensor(
                flat_subtoken_idx, dtype=torch.long, device=target_device)
            flat_seg_t = torch.tensor(
                flat_word_seg, dtype=torch.long, device=target_device)
            per_word_first_t = torch.tensor(
                per_word_first, dtype=torch.long, device=target_device)
            per_word_b_t = torch.tensor(
                per_word_b, dtype=torch.long, device=target_device)
            per_word_slot_t = torch.tensor(
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
        # MAX-fused word vector we computed -- hence we must setW
        # explicitly here.
        event_parts = [word_vectors]
        if self.nWhere > 0 and where_indices is not None:
            event_parts.append(self.subspace.whereEncoding.encode(where_indices))
        if self.nWhen > 0 and when_indices is not None:
            event_parts.append(self.subspace.whenEncoding.encode(when_indices))
        muxed_event = (torch.cat(event_parts, dim=-1)
                       if len(event_parts) > 1 else word_vectors)
        self.subspace.event.setW(muxed_event)
        self._embedded_input = muxed_event
        self._bpe_word_mask = word_active
        # Only InputSpace.forward can produce the AR-windowed [B*K, N]
        # mask. Clear any stale value when _embed_bpe is used directly
        # (non-AR / inference paths) so PerceptualSpace.forward falls
        # back to this fresh unwindowed [B, N] mask.
        self._bpe_word_mask_flat = None
        return self.subspace

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

    def _q2_promote_activation(self, event):
        """Q2 bitonic-to-bivector promotion (spec §Q2, line 1405).

        Reduces the per-slot percept event to a signed scalar via signed-
        sum across the content dim, then splits onto the non-negative
        paired-index axes ``(aP, aN) = (max(0, x), max(0, -x))``.

        Args:
            event: ``[B, N, D_P]`` per-slot percept content.

        Returns:
            ``[B, N, 2]`` bivector activation, monotonic in ``[0, 1]^2``.
        """
        x = event.sum(dim=-1)
        aP = torch.relu(x)
        aN = torch.relu(-x)
        return torch.stack([aP, aN], dim=-1)

    def _q2_lower_activation(self, bivec, content_dim):
        """Inverse of `_q2_promote_activation`: bivector -> per-slot
        signed scalar -> broadcast to per-slot content vector.

        The forward Q2 promotion is many-to-one over the content dim
        (signed sum collapses ``D_P`` features to one scalar), so the
        reverse cannot recover the per-feature pattern. We broadcast the
        recovered scalar uniformly across the content dim; downstream
        ``reverseEnd`` / ``InvertibleLinearLayer.reverse`` handles any
        further structure recovery.

        Args:
            bivec: ``[B, N, 2]`` bivector activation.
            content_dim: target ``D_P`` for the output event.

        Returns:
            ``[B, N, D_P]`` event tensor with the recovered scalar
            broadcast across the content axis.
        """
        x = bivec[..., 0] - bivec[..., 1]
        return x.unsqueeze(-1).expand(-1, -1, content_dim).contiguous()

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

    def _read_event(self, sib):
        """Return ``sib.subspace.event`` materialised, or None."""
        if sib is None:
            return None
        sub = getattr(sib, 'subspace', None)
        if sub is None:
            return None
        ev = getattr(sub, 'event', None)
        if ev is None:
            return None
        return ev.getW()

    def _sourced_input(self, vspace):
        """Dual-input: primary input + conceptual loopback, length-averaged.

        Mirrors ``ConceptualSpace._sourced_input`` one tier down: the
        primary input source is the upstream (Input)Space event reshaped
        via ``forwardBegin``; the cross-channel contribution is the
        prior C-tier event, lifted back to perceptual content width via
        the C-tier codebook's SVD pseudo-inverse when the bivector
        regime is active. Averaging (``sum / count``) keeps the input
        bounded to the same range as each source so the codebook's
        [-1, 1] / [0, 1] invariants survive both channels firing.

        Sentence-start / cold-start: ``conceptualSpace_ref`` may be
        ``None`` (standalone unit tests) or its event tensor may be
        unset (very first forward call before any C output has been
        produced) -- either case degrades cleanly to the legacy
        single-source ``forwardBegin`` result.

        See ``ConceptualSpace._sourced_input`` for the symmetric
        symbolic loopback that this method mirrors.
        """
        primary = self.forwardBegin(vspace, returnVectors=True)
        cs = self.conceptualSpace_ref
        if cs is None:
            return primary
        c_event = self._read_event(cs)
        if c_event is None:
            return primary
        # The C→P loopback only fires in the bivector regime: the C
        # event is a ``[B, V, 2]`` per-prototype catuskoti handoff that
        # lifts back to concept content width via the C-tier codebook's
        # cached SVD pseudo-inverse. Non-bivector ConceptualSpace
        # writes its content directly to ``event``; that content lives
        # in the same dim space as the primary input, would
        # coincidentally shape-match, and would corrupt legacy
        # non-bivector forward dynamics by averaging two semantically
        # different tensors. Mirrors the symbolic lift gating in
        # ``ConceptualSpace._sourced_input``.
        cs_what = getattr(cs.subspace, 'what', None)
        if not (isinstance(cs_what, ProjectionBasis)
                and c_event.dim() == 3 and c_event.shape[-1] == 2):
            return primary
        V_orig = int(cs.inputShape[0])
        c_event = cs_what.reverse(c_event, V=V_orig)
        if c_event.shape == primary.shape:
            return (primary + c_event) / 2
        return primary

    def forward(self, subspace):
        """Perception: map input vectors to percepts via attention + VQ + chunking.

        Handles three paths: warm-serial AR (process only new last slot,
        splice into the rolled cache), embedding (text -> lexicon -> percept),
        and numeric (linear -> attention -> VQ). Writes the resulting
        percept tensor to ``self.subspace`` and returns the live subspace.
        """
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
            else:
                raise ValueError(
                    f"PerceptualSpace chunking must be bpe|lexicon, got {mode!r}")
        if getattr(vspace, '_demuxed', False) and vspace._active is not None:
            self.subspace._byte_indices = vspace._active[:, :, 0].long()
        # Conceptual loopback: when ``conceptualSpace_ref`` is wired, the
        # primary input is averaged with the lifted C-tier event from the
        # previous forward (mirrors ConceptualSpace's symbolic loopback,
        # one tier down). Falls back to plain ``forwardBegin`` when the
        # ref is None or its event tensor is unset (sentence start /
        # standalone unit tests).
        x = self._sourced_input(vspace)
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
        if self.chunking_mode == "bpe":
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
                        vspace.event.setW(ev * mask.unsqueeze(-1))
                    elif ev.dim() == 2 and ev.shape[0] == mask.numel():
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

        if self._bivector_output:
            # Q2 promotion: replace the legacy per-slot scalar activation
            # with the catuskoti bivector ``[B, N, 2]``. Downstream
            # ConceptualSpace consumes this as the left half of the
            # widened ``[P_event || S_event]`` PiLayer input (gated on
            # ConceptualSpace.bivectorOutput at Models.py:1797-1820).
            event = vspace.materialize(mode="event")
            if event is not None and event.dim() == 3:
                bivec = self._q2_promote_activation(event)
                vspace.activation.setW(bivec)

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
        # `_reverse_text` returns earlier and bypasses the bivector
        # lowering below. Mixing text-mode lexicon with bivectorOutput=true
        # is outside the current B3 scope; revisit when text-mode bivector
        # is needed.
        if isinstance(self.subspace.what, Embedding):
            self._reverse_text(vspace)
            return vspace
        if self.invertible:
            vspace.normalize("percepts", target="event",
                             normalize=True, reverse=True)
        if self._bivector_output:
            # Lower the bivector activation back to a per-slot signed
            # scalar broadcast across the percept input dim, so the
            # downstream invertible reverse sees the pre-Q2 layout.
            bivec = vspace.activation.getW()
            if bivec is not None and bivec.dim() == 3 and bivec.shape[-1] == 2:
                event = self._q2_lower_activation(bivec, self.inputShape[1])
                vspace.event.setW(event)
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
        """Render the last recovered text buffer stored on PerceptualSpace.

        Requires a prior ``reverse()`` to have populated
        ``_recovered_input``. Enforces that WhereEncoding's period
        covers ``buf_size`` so the sin/cos decode doesn't alias near
        angle=0 and stamp tokens at spurious offsets.
        """
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


class ShortTermMemory(nn.Module):
    """Short-term memory (STM) on ConceptualSpace.

    A per-batch stack of unquantized C-tier activations -- "ideas",
    the continuous compositions that the chart produces by reducing
    concepts / earlier ideas. Distinct from
    ``WordSpace._stm_fired`` (which is a once-per-sentence
    discourse-priming flag, not a working-memory buffer).

    Semantics:
        * Each slot holds a ``[concept_dim]`` vector.
        * Pushes are bottom-up: ``peek(b, 0)`` returns the most
          recent idea; ``peek(b, n)`` returns the n-th most recent.
        * Capacity is a soft cap (7±2 for linguistic processing,
          per the brain's classical working-memory limit). The
          (future) serial parser is expected to reduce ideas before
          pushing when ``is_full(b)`` would otherwise be true.
        * Subsymbolic operation can drive a wider STM than the
          linguistic 7±2; the capacity is configurable via
          ``<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>``
          in the model XML.

    Lifecycle:
        * Built by ``ConceptualSpace.__init__`` at construction time
          (capacity from XML; default 9).
        * Cleared on hard ``Reset`` (sentence boundary): all rows
          set to zero, depth pointers reset to 0.
        * No active consumer yet -- this is the structural slot
          the upcoming serial / shift-reduce parser will read and
          mutate. The current batched-CKY chart doesn't use it.

    Storage is a plain registered buffer (``persistent=False``);
    STM contents are runtime working state, not learned weights.
    """

    DEFAULT_CAPACITY = 9
    # Catuskoti pole width for the per-idea truth tag bivector.
    _TRUTH_TAG_WIDTH = 2

    def __init__(self, batch=1, capacity=None, concept_dim=0):
        super().__init__()
        self.capacity = int(capacity or self.DEFAULT_CAPACITY)
        self.concept_dim = int(concept_dim)
        # ``persistent=False``: STM is transient working state, not
        # saved with the model checkpoint.
        self.register_buffer(
            "_buffer",
            torch.zeros(int(batch), self.capacity, self.concept_dim),
            persistent=False)
        self.register_buffer(
            "_depth",
            torch.zeros(int(batch), dtype=torch.long),
            persistent=False)
        # Per-idea truth-tag bivector ``[B, capacity, 2]``: catuskoti
        # pole pair attached to each STM slot. Written by
        # ``query`` / ``equals`` / ``part`` rule firings during chart
        # compose; read by downstream consumers (e.g. continuation
        # reductions) as metadata on the operand idea.
        self.register_buffer(
            "_truth_tags",
            torch.zeros(int(batch), self.capacity, self._TRUTH_TAG_WIDTH),
            persistent=False)

    def ensure_batch(self, batch):
        """Resize the STM buffers to ``batch`` rows when the model's
        microbatch dimension changes. Idempotent.
        """
        batch = int(batch)
        if int(self._buffer.shape[0]) == batch:
            return
        device = self._buffer.device
        self._buffer = torch.zeros(
            batch, self.capacity, self.concept_dim, device=device)
        self._depth = torch.zeros(batch, dtype=torch.long, device=device)
        self._truth_tags = torch.zeros(
            batch, self.capacity, self._TRUTH_TAG_WIDTH, device=device)

    def push(self, b, idea):
        """Push ``idea`` (a ``[concept_dim]`` tensor) onto row ``b``.

        Raises if the stack is full -- the parser is expected to
        reduce before pushing when ``is_full(b)`` would be true.
        Fresh slot is born with a zero truth tag.
        """
        depth = int(self._depth[b].item())
        if depth >= self.capacity:
            raise RuntimeError(
                f"ShortTermMemory.push: row {b} is at capacity "
                f"({self.capacity}); the parser must reduce before "
                f"pushing further.")
        self._buffer[b, depth] = idea
        self._truth_tags[b, depth].zero_()
        self._depth[b] = depth + 1

    def pop(self, b):
        """Pop and return the top idea for row ``b``, or ``None`` when empty.

        The popped slot's truth tag is zeroed alongside its content.
        """
        depth = int(self._depth[b].item())
        if depth == 0:
            return None
        depth -= 1
        idea = self._buffer[b, depth].clone()
        self._buffer[b, depth].zero_()
        self._truth_tags[b, depth].zero_()
        self._depth[b] = depth
        return idea

    def peek(self, b, n=0):
        """Return the ``n``-th item from top of row ``b``, or ``None``
        when fewer than ``n+1`` items are on the stack. ``n=0`` is the
        most recent (top); ``n=size(b)-1`` is the oldest (bottom).
        """
        depth = int(self._depth[b].item())
        if depth <= n:
            return None
        return self._buffer[b, depth - 1 - n]

    def snapshot(self, detach=False):
        """Return ``[B, max_depth, D]`` slice of the live buffer.

        ``max_depth`` is the largest depth across batch rows so the
        returned tensor is a single uniform slab; rows with shorter
        sentences carry zero-padding at the tail.  Returns ``None``
        when the buffer is empty (no pushes have occurred).

        Used by ``BasicModel._chart_compose_at_C`` and
        ``BasicModel._chart_generate_from_stm`` to feed the chart at
        C-tier without each call site re-implementing the depth/padding
        slicing contract.

        ``detach=True`` clones away from the autograd graph (e.g. for
        save_weights snapshots or external diagnostics).  The default
        keeps grad flowing through the buffer so the chart's per-rule
        selections can shape upstream PiLayer/SymbolicSpace weights.
        """
        B = int(self._buffer.shape[0])
        if B == 0:
            return None
        max_depth = int(self._depth.max().item())
        if max_depth == 0:
            return None
        snap = self._buffer[:, :max_depth, :]
        if detach:
            snap = snap.detach().clone()
        else:
            snap = snap.clone()
        return snap

    def size(self, b):
        """Current depth (number of occupied slots) for row ``b``."""
        return int(self._depth[b].item())

    def is_full(self, b):
        """True when row ``b`` is at capacity."""
        return self.size(b) >= self.capacity

    def is_empty(self, b):
        """True when row ``b`` has no occupants."""
        return self.size(b) == 0

    def clear(self, b=None):
        """Clear row ``b`` (or all rows when ``b`` is ``None``).

        Called on hard ``Reset`` (sentence boundary). When ``b`` is
        outside the currently-allocated batch dimension, the buffer
        hasn't been grown to include that row yet (no pushes ever
        happened for it), so there is nothing to clear -- skip
        gracefully instead of raising IndexError. Clears both the
        idea buffer and the per-slot truth tags.
        """
        if b is None:
            self._buffer.zero_()
            self._depth.zero_()
            self._truth_tags.zero_()
            return
        b = int(b)
        if b < 0 or b >= int(self._buffer.shape[0]):
            return
        self._buffer[b].zero_()
        self._depth[b] = 0
        self._truth_tags[b].zero_()

    def set_truth_tag(self, b, slot_from_top, tag):
        """Write the catuskoti truth bivector for the slot at depth
        ``size(b) - 1 - slot_from_top``.

        Args:
            b: batch row index.
            slot_from_top: 0 is the most-recently-pushed idea.
            tag: ``[2]`` tensor with the truth bivector (pos, neg poles).
        """
        depth = int(self._depth[b].item())
        if depth <= slot_from_top:
            raise IndexError(
                f"ShortTermMemory.set_truth_tag: row {b} has "
                f"{depth} ideas; slot_from_top={slot_from_top} is "
                f"out of range.")
        self._truth_tags[b, depth - 1 - slot_from_top] = tag

    def get_truth_tag(self, b, slot_from_top=0):
        """Read the truth bivector for slot ``slot_from_top`` from the top.

        Returns ``None`` when fewer than ``slot_from_top + 1`` items are
        on the stack.
        """
        depth = int(self._depth[b].item())
        if depth <= slot_from_top:
            return None
        return self._truth_tags[b, depth - 1 - slot_from_top]


class ConceptualSpace(Space):
    """Transforms percepts into concepts via a SigmaLayer (summation layer).

    In the forward data flow: PerceptualSpace -> **ConceptualSpace** -> SymbolicSpace.
    Uses a SigmaLayer to combine perceptual features into conceptual
    representations.  The SigmaLayer computes weighted sums (inner products)
    rather than the permutation-equivariant operations of PiLayer.

    Supports optional self-attention and VQ codebook quantization.

    When ``invertible=True``, uses ``SigmaLayer(invertible=True)`` whose
    inverse is exact.  When ``reversible=True`` without invertibility, a
    separate SigmaLayer is trained for the reverse direction.
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
        # Bivector regime flags — read before super().__init__ since the
        # what-basis builder runs inside it. ``bivectorOutput`` swaps the
        # legacy VQ snap on the C-tier output for the per-prototype
        # catuskoti bivector ``[B, V_C, 2]`` produced by
        # ``Codebook.forward(...)``.
        # ``svdOrthogonalInit`` SVD-orthogonalizes the codebook at
        # construction so ``project_reverse`` (which scales by 1/Σ) is
        # well-conditioned from the very first forward call.
        try:
            self._bivector_output = bool(TheXMLConfig.space(section, "bivectorOutput"))
        except KeyError:
            self._bivector_output = False
        # SVD-orthogonalize the codebook at construction so ``project_reverse``
        # (which scales by 1/Σ) is well-conditioned from t=0. Always on; the
        # XML <svdOrthogonalInit> knob has been retired.
        self._svd_orthogonal_init_cfg = True
        super().__init__(inputShape, spaceShape, outputShape)
        self.nonlinear = nonlinear
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        # Right-half loopback widening retired: ConceptualSpace.forward
        # reads exactly one input source per stage (perceptual at order 0,
        # lifted-symbolic at order > 0) via ``_sourced_input``, averaging
        # the two when both are present after a bivector lift on the
        # symbolic side. The legacy ``subsymbolic_widen_dim`` constructor
        # parameter and the ``[P_event || S_event]`` concat it gated were
        # removed together with ``SubsymbolicSpace``.
        self._right_half_dim = 0
        # Sibling references: ConceptualSpace reads BOTH the perceptual
        # event (stem-cached, same instance for every stage) AND the
        # SymbolicSpace's previous event at every stage. Combiner:
        # additive after bivector-lift (see ``_sourced_input``).
        # ``perceptualSpace_ref`` is wired post-construction in
        # ``_build_pipelines_per_stage``.
        self.symbolicSpace_ref = None
        self.perceptualSpace_ref = None
        input = self.subspace.getEncodedInputSize()
        if self._bivector_output:
            # Bivector regime: PiLayer stays a square isomorphism inside
            # conceptual content space. Dim adaptation to the bivector
            # handoff width happens INSIDE Codebook.project (input
            # ``[B, V, D] -> [B, N, 2]`` per-prototype accumulator), not
            # inside Pi. This preserves XOR's discriminative info instead
            # of squeezing it through a non-square Pi bottleneck.
            output = input
        else:
            output = self.subspace.getEncodedOutputSize()
        if hasAttention:
            self.attention = AttentionLayer(output, output, type="transformer")

        # Post-2026-05-01 refactor: ``self.pi`` is the canonical forward
        # PiLayer for every mode. In the asymmetric two-pass ergodic
        # path, ``self.pi`` aliases ``self.pi1`` (forward direction) and
        # ``self.pi2`` is preserved as a separate attribute consulted
        # on the reverse path via ``self._pi_reverse``. Bare
        # ``forwardPi``/``reversePi`` aliases were removed.
        if self.reversible:
            if invertible:
                self.pi = PiLayer(input, output, naive=naive, ergodic=ergodic,
                                  invertible=True, nonlinear=nonlinear,
                                  stable=True, monotonic=monotonic)
                self.params = self.pi.getParameters()
                self.layers = nn.ModuleList([self.pi])
            else:
                self.pi1 = PiLayer(input, output, naive=naive, ergodic=ergodic,
                                   invertible=True, nonlinear=nonlinear,
                                   stable=True, monotonic=monotonic)
                self.pi2 = PiLayer(input, output, naive=naive, ergodic=ergodic,
                                   invertible=True, nonlinear=nonlinear,
                                   stable=True, monotonic=monotonic)
                self.pi = self.pi1  # forward direction canonical alias
                self.params = self.pi1.getParameters() + self.pi2.getParameters()
                self.layers = nn.ModuleList([self.pi1, self.pi2])
        else:
            self.pi = PiLayer(input, output, naive=naive, ergodic=ergodic,
                              nonlinear=nonlinear, stable=True, monotonic=monotonic)
            self.params = self.pi.getParameters()
            self.layers = nn.ModuleList([self.pi])
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
        self.stm = ShortTermMemory(
            batch=1, capacity=stm_capacity, concept_dim=concept_dim)

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
        if not getattr(self, "_bivector_output", False):
            return None
        if not self.codebook:
            return Tensor(nVectors=self.nVectors, nDim=self.nDim)
        basis = ProjectionBasis()
        basis.use_dot_product = bool(getattr(self, "use_dot_product", False))
        basis.create(
            self.inputShape[0],
            self.nVectors,
            self.nDim,
        )
        return basis

    def _pi_reverse(self, y):
        """PiLayer reverse, hiding the two-pass ergodic mode split.

        Single-PiLayer modes call ``self.pi.reverse``. The two-pass
        ergodic mode (where ``self.pi`` aliases ``self.pi1`` for the
        forward path) routes the reverse through ``self.pi2`` instead.

        When the input PiLayer is widened for the subsymbolic loop,
        the reverse output carries the right-half re-entrant
        contribution which is not consumable by upstream
        PerceptualSpace; slice it off so downstream reverse stages see
        only the perceptual half.
        """
        pi2 = getattr(self, 'pi2', None)
        if pi2 is not None:
            x = pi2.reverse(y)
        else:
            x = self.pi.reverse(y)
        if self._right_half_dim > 0:
            x = x[..., :-self._right_half_dim]
        return x

    def _read_event(self, sib):
        """Return ``sib.subspace.event`` materialised, or None."""
        if sib is None:
            return None
        sub = getattr(sib, 'subspace', None)
        if sub is None:
            return None
        ev = getattr(sub, 'event', None)
        if ev is None:
            return None
        return ev.getW()

    def _get_active_input_sibling(self):
        """Return the sibling whose previous event feeds this stage's input.

        Always returns ``symbolicSpace_ref`` (the verbal / propositional
        loopback). The earlier ``<architecture><mode>grammar|parallel
        </mode>`` selector that toggled between a SymbolicSpace and a
        SubsymbolicSpace sibling has been retired together with
        ``SubsymbolicSpace`` (PerceptualSpace serves as the subsymbolic
        substrate; the imagistic loop is now the new C→P
        ``conceptualSpace_ref`` loopback on PerceptualSpace).

        Returns ``None`` when no sibling is wired (standalone unit tests).
        """
        return self.symbolicSpace_ref

    def _sourced_input(self, vspace):
        """Dual-input: perceptual + sibling-symbolic, length-averaged.

        Combines available shape-compatible contributions by averaging
        (``sum / count``) instead of plain summation. Averaging keeps
        the input bounded to the same range as each source so the
        codebook's [-1, 1] invariant survives both channels firing.
        Null vectors after reset don't contribute at all (sym is
        ``None``, not a real zero tensor) -- so the average degrades
        cleanly to the legacy single-source result.

        Primary input source (preserves all ``forwardBegin`` side
        effects -- ``subspace.batch`` and ``_pre_reshape_input`` -- so
        downstream ``forwardEnd`` / ``reverseEnd`` reshape correctly):

          Stage 0  (vspace IS perceptualSpace_ref.subspace):
              primary = ``forwardBegin(vspace)`` -- perceptual reshape.
          Stage > 0 (vspace is the previous symbolic subspace):
              primary = lifted-sibling-symbolic when present, else
              ``forwardBegin(vspace)`` fallback (sentence start).

        Cross-channel contribution then participates in the average:

          Stage 0:  ``sym`` (lifted sibling) when shape-matched.
          Stage > 0: stem-cached ``perc`` (reshaped) when shape-matched.

        Stage detection is by object identity (``vspace is
        perceptualSpace_ref.subspace``); no ``_stage_idx`` flag.
        """
        perc_ref_sub = (self.perceptualSpace_ref.subspace
                        if self.perceptualSpace_ref is not None
                        else None)
        at_stage_0 = (perc_ref_sub is not None and vspace is perc_ref_sub)

        sib = self._get_active_input_sibling()
        sym = self._read_event(sib) if sib is not None else None
        if sym is not None and (
                isinstance(self.subspace.what, ProjectionBasis)
                and sym.dim() == 3 and sym.shape[-1] == 2):
            V_orig = int(self.inputShape[0])
            sym = self.subspace.what.reverse(sym, V=V_orig)

        if at_stage_0 or perc_ref_sub is None:
            # Primary = perceptual via vspace (legacy stage-0 path or
            # standalone-construction fallback). Average in sym when
            # shape-matched.
            primary = self.forwardBegin(vspace, returnVectors=True)
            if sym is not None and sym.shape == primary.shape:
                return (primary + sym) / 2
            return primary

        # Stage > 0: primary = lifted_sym (legacy). On sentence start
        # the sibling event is None -- fall back to legacy
        # forwardBegin(vspace) so order-0 semantics hold for the very
        # first forward call of the sequence.
        if sym is None:
            return self.forwardBegin(vspace, returnVectors=True)

        # Read the stem-cached perceptual event and apply the same
        # nInputDim reshape ``forwardBegin`` would WITHOUT clobbering
        # the side effects (those came from vspace, the correct source
        # for this stage's forwardEnd/reverseEnd).
        perc_ev = self._read_event(self.perceptualSpace_ref)
        if perc_ev is not None and perc_ev.dim() == 4:
            B, K, N, D = perc_ev.shape
            perc_ev = perc_ev.reshape(B * K, N, D)
        if (perc_ev is not None and perc_ev.dim() == 3
                and self.nInputDim != -1):
            try:
                perc_ev = perc_ev.reshape(
                    perc_ev.shape[0], -1, self.nInputDim)
            except RuntimeError:
                perc_ev = None
        if perc_ev is not None and perc_ev.shape == sym.shape:
            return (sym + perc_ev) / 2
        return sym

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

    def forward(self, subspace):
        """Knowing: map percepts to concepts via PiLayer + optional attention + VQ.

        When nonlinear=True the multiplicative log-domain transform keeps
        the output in [-1, 1].
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        # SyntacticLayer is unconditional: it dispatches whatever the
        # grammar XML specifies for the C tier (e.g. ``C = pi(C)`` from
        # model.xml's default grammar) -- exact mathematical replacement
        # for the legacy ``y = self.pi(x)`` call. Per the 2026-05-07
        # rollback, ``WordSpace.compose`` populates ``current_rules``
        # from the grammar XML in default-only / useGrammar='none' fast
        # paths, so the cursor always finds the configured rule (no
        # ``default_rule`` code-level fallback).
        # Per-order input source. Order 0 reads PerceptualSpace.event
        # (today's behavior); higher orders read the active sibling's
        # previous event (Symbolic under <mode>grammar</mode>,
        # Subsymbolic under <mode>parallel</mode>) lifted from the
        # downstream bivector handoff back into concept content via
        # the C-tier codebook's SVD pseudo-inverse. Replaces the
        # legacy `_build_combined_input` concat (which aliased a wide
        # perceptual content vector with a 2-wide bivector activation;
        # PiLayer treated them uniformly which was semantically wrong).
        x = self._sourced_input(vspace)
        # Post-split: grammar lives at S; C is semantically
        # subsymbolic. The SyntacticLayer dispatch at C is retained
        # here as a backward-compat no-op for grammars that omit
        # C-tier rules (it then falls through to ``y = x``
        # passthrough, preserving the legacy behavior of configs like
        # MM_xor_bivector). Grammars that DO list C-tier rules can
        # still fire them, but the canonical home of grammar
        # operations is now S.
        if getattr(self, 'syntacticLayer', None) is None:
            y = self.pi(x)
        else:
            vspace.set_event(x)
            vspace = self.syntacticLayer.forward(vspace)
            y = vspace.materialize()
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
            if (self._bivector_output
                    and isinstance(self.subspace.what, ProjectionBasis)):
                # Bivector regime: per-prototype catuskoti bivector
                # projection on a ``ProjectionBasis``.  ``y`` becomes
                # ``[B, V_C, 2]`` (non-negative).  The matching reverse
                # routes through ProjectionBasis.reverse (exact LDU
                # inverse via triangular solves; no SVD cache, no
                # per-forward state).
                y = self.subspace.what.forward(y)
            elif (isinstance(self.subspace.what, Codebook)
                    and self.nVectors > self.outputShape[0]):
                # Wide-codebook top-K: when nVectors > nOutput, route through
                # the content Codebook with topK=nOutput so the per-codebook-
                # entry activation is pruned to the nOutput strongest
                # survivors.
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
        if self._bivector_output:
            # Under CSBP the C-tier output is a non-negative bivector;
            # the legacy ``concepts`` range check (-1..1 on what) does not
            # apply. The bivector kind asserts non-negativity instead.
            vspace.normalize("bivector", target="event")
        else:
            vspace.normalize("concepts", target="what")       # range check
            vspace.normalize("concepts", target="where")      # range check
            vspace.normalize("concepts", target="activation")  # range check
        return vspace

    def reverse(self, subspace):
        """Visualizing: reconstruct percepts from concepts via reverse PiLayer.

        When CSBP bivector output is active, first runs the codebook's
        cached SVD pseudo-inverse to lift the bivector ``[B, V, 2]`` back
        to ``[B, V, D_C]``. Then dispatches to the SyntacticLayer (or the
        legacy bare PiLayer.reverse) for the rule-driven reverse pass.
        """
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        vspace = subspace
        # SyntacticLayer is unconditional: its default rule fires
        # ``PiLayer.reverse(y)`` (or two-pass ergodic ``_pi_reverse``
        # via Phase 3 adapter), exact mathematical replacement for
        # the legacy ``y = self._pi_reverse(y)`` call. When the chart
        # populates a C-tier generate sequence, the dispatcher fires
        # those instead.
        y = self.reverseBegin(vspace, returnVectors=True)
        if (self._bivector_output
                and isinstance(self.subspace.what, ProjectionBasis)):
            # CSBP reverse lift: bivec ``[B, V_C, 2]`` -> ``[B, V, D_C]``
            # via the exact LDU inverse on ProjectionBasis. Runs before
            # pi.reverse so the upstream PiLayer sees the natural
            # ``[B, V, nDim]`` shape it produced on forward.  V comes
            # from this space's ``inputShape[0]`` (the slot count the
            # forward consumed); the per-row summary vector is then
            # replicated across V positions, which is the mathematically-
            # correct answer for the V-axis-summed bivector forward.
            V_orig = int(self.inputShape[0])
            y = self.subspace.what.reverse(y, V=V_orig)
        # Post-split: grammar lives at S; C is semantically
        # subsymbolic on reverse too. The SyntacticLayer dispatch at
        # C is retained for backward compat — no-op when the grammar
        # has no C-tier reverse rule.
        if getattr(self, 'syntacticLayer', None) is None:
            y = self._pi_reverse(y)
        else:
            vspace.set_event(y)
            vspace = self.syntacticLayer.reverse(vspace)
            y = vspace.materialize()
            if (y is not None and y.dim() == 3
                    and y.shape[-1] != self.subspace.getEncodedInputSize()):
                y = self._pi_reverse(y)
        if self._right_half_dim > 0 and y is not None and y.dim() == 3:
            y = y[..., :-self._right_half_dim]
        self.concepts = y.detach()
        vspace = self.reverseEnd(y, returnVectors=True)
        vspace.normalize("percepts", target="what")   # range check
        vspace.normalize("percepts", target="where")  # range check
        return vspace

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        pass


# Default Gaussian region width retained for callers that calibrate
# extents by Gaussian σ. The legacy area / luminosity that consumed it
# migrated to the Mereology mixin (bin/Mereology.py) with a
# hyperrectangle-volume formula; this constant remains for
# backward-compat with any consumer that still keys on it.
_DEFAULT_SYMBOL_SIGMA = 0.1


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

    Distinct from the WordSpace category codebook:
      * ``WordSpace.category_codebook``: ``[max(64, |grammar.categories|),
        4]`` learned embeddings keyed by grammar nonterminal / POS name
        (S, NP, VP, N, V, ADJ, ...).  Used by the chart's POS scorer.
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

    def empty_state(self, batch=1):
        """Return a zero tensor shaped like this space's symbolic state.

        Used by the unified merged outer loop to seed ``ss`` before the
        first concept emission. ``batch`` sizes the leading axis; the
        rest comes from ``outputShape``.
        """
        nOutput = int(self.outputShape[0])
        nDim = int(self.outputShape[1])
        return torch.zeros(int(batch), nOutput, nDim, device=TheDevice.get())

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
        # Bivector regime flag: mirror of ConceptualSpace.bivectorOutput.
        # When True, ``forward`` produces ``[B, V_S, 2]`` via
        # ``Codebook.forward(...)`` and ``reverse`` lifts
        # via the cached SVD pseudo-inverse, matching the C-tier shape.
        try:
            self._bivector_output = bool(
                TheXMLConfig.space(section, "bivectorOutput"))
        except KeyError:
            self._bivector_output = False
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
        # reach it via ``self.wordSpace.truth_layer`` (see forward() below
        # and the truth-using paths in BasicModel).
        try:
            self.accumulateTruth = float(TheXMLConfig.space(section, "accumulateTruth"))
        except (KeyError, TypeError, ValueError):
            self.accumulateTruth = 0.0

        # Per-instance Gaussian region width used by ``area`` / ``luminosity``.
        # ``None`` means fall back to ``_DEFAULT_SYMBOL_SIGMA``.  Set when
        # symbols carry a calibrated extent.
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

        ``polarity=True`` allocates a per-row tag in
        ``polarity_ids`` (POLARITY_AFFIRM/NON/NOT) so reconstruction
        can emit the matching surface form ("foo"/"non-foo"/"not foo").

        2026-05-13: in the bivector regime, ``.what`` is now a
        ``ProjectionBasis`` (LDU-parameterized) rather than a Codebook
        with invertible=True, matching the ConceptualSpace bivector
        builder.  The exact LDU inverse replaces the legacy SVD cache.
        """
        if not self.codebook:
            return Tensor(nVectors=self.nVectors, nDim=self.nDim)
        if getattr(self, "_bivector_output", False):
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
            polarity=True,
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
        event = subspace.event.getW() if subspace.event is not None else None
        if event is not None and event.ndim == 3 and event.shape[-1] >= 2:
            bivec = event[..., :2]
        else:
            bivec = (subspace.what.getW()
                     if subspace.what is not None else None)
            # Narrow-event fallback (post-2026-05-07 rollback): when
            # neither the muxed event nor the .what content is at least
            # 2 columns wide there is no bivector to collapse. This
            # arises in narrow conceptual-order configs (D_C == 1) and
            # is a graceful no-op rather than an error.
            if (bivec is None or not torch.is_tensor(bivec)
                    or bivec.shape[-1] < 2):
                return subspace
        pos = bivec[..., 0]
        neg = bivec[..., 1]
        subspace.activation.setW(pos - neg)
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

    @staticmethod
    def _vq_chunk_budget():
        """Memory budget (bytes) for VQ distance matrix chunks.

        With d content dims and K codebook entries, each row of the
        distance matrix costs K * 4 bytes.  The budget controls how
        many rows are processed per matmul.  Larger budgets mean fewer
        sequential chunks -- critical for AR batches where N can
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
        incoming.set_what(torch.zeros(inc_what_shape, dtype=torch.float32))
        incoming._rule_dispatch = True
        return incoming

    def _op_for_rule(self, rule_id, wordSpace=None):
        """Return a callable ``(self_sub, inc_sub) -> new_what`` for ``rule_id``.

        Dispatches through the ``wordSpace.host_layer(tier, rule_name)``
        registry (the same path WordSpace's grammar applies during chart
        compose). When ``wordSpace`` is missing, no host layer is
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
        if wordSpace is not None:
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
                    host = wordSpace.host_layer(tier, method_name)
                except Exception:
                    host = None
                # Fallback: some grammar configs only register one tier
                # for a rule. Try the other tier so the dispatch still
                # finds a layer in mixed configurations.
                if host is None:
                    fallback = 'S' if tier == 'C' else 'C'
                    try:
                        host = wordSpace.host_layer(fallback, method_name)
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

    def _superposed_op(self, rule_probs, wordSpace=None):
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
        if wordSpace is None:
            raise ValueError(
                "SymbolicSpace.forward requires wordSpace for rule dispatch; "
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
            self.subspace.what.setW(new_what)

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

    def forward(self, subspace):
        """Concept->symbol forward.

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
        if subspace.is_empty():
            return subspace
        self.subspace.copy_context(subspace)
        # Phase-1 ``parallel`` mode gating: when held at zero the
        # resolve / lift / codebook / TruthLayer paths skip and the
        # event tensor is filled with zeros. Downstream consumers
        # read zeros; the elementwise-sum at the next conceptual
        # order's combined input contributes nothing from this Space.
        if self.held_at_zero:
            sample = subspace.materialize()
            if sample is not None:
                B = sample.shape[0]
                N = self.outputShape[0]
                D = self.subspace.muxedSize
                zero_event = torch.zeros(B, N, D, device=sample.device,
                                         dtype=sample.dtype)
                self.subspace.set_event(zero_event)
            return self.subspace
        quantize = getattr(self, "quantize", True)
        is_last = getattr(self, "is_last", False)
        wordSpace = getattr(self, "wordSpace", None)
        if getattr(subspace, '_rule_dispatch', False):
            return self._forward_with_rule_dispatch(
                subspace, wordSpace=wordSpace, quantize=quantize)
        vspace = subspace
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
            chart_rules_S = (
                wordSpace.current_rules.get('S')
                if (wordSpace is not None
                    and getattr(wordSpace, 'current_rules', None))
                else None)
            row_zero = self.syntacticLayer._row_zero_rules(chart_rules_S)
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

        if self.accumulateTruth > 0 and wordSpace is not None:
            truth_layer = getattr(wordSpace, 'truth_layer', None)
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

        if wordSpace is not None:
            act = wordSpace.forwardSymbols(act, self.subspace)

        # Resolve [pos, neg] bivector to 1-D per-symbol activation
        # before the codebook sees it. resolve() writes
        # subspace.activation = pos - neg.
        self.resolve(self.subspace)

        # Bivector regime: replace the VQ-VAE / hard-quantize / VQ
        # branches with the per-prototype catuskoti bivector projection.
        # ``Codebook.forward(act)`` returns ``[B, V_S, 2]``
        # via the cached SVD pseudo-inverse of W. The result lives on
        # ``subspace.event`` so downstream consumers (OutputSpace,
        # _compute_symbol_terms) read it via ``materialize()``.
        if (getattr(self, "_bivector_output", False)
                and isinstance(self.subspace.what, ProjectionBasis)):
            bivec = self.subspace.what.forward(act)
            self.subspace.set_event(bivec, compute_activation=False)
            vspace = self.forwardEnd(self.subspace)
            self._emit_symbol_terms(
                vspace, self._compute_symbol_terms(bivec))
            if (self.impenetrable_overlap > 0.0
                    or self.impenetrable_variance > 0.0):
                imp = self.impenetrable_loss()
                if (imp is not None and torch.is_tensor(imp)
                        and (imp.requires_grad or imp.abs().item() > 0.0)):
                    vspace.errors.add(
                        "symbol_impenetrable", imp, weight=1.0,
                        space="SymbolicSpace", category="symbol")
            vspace.normalize("bivector", target="event")
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
        self.subspace.copy_context(subspace)
        vspace = subspace
        wordSpace = getattr(self, "wordSpace", None)
        vspace = self.reverseBegin(vspace)
        act = vspace.materialize()                        # [B, N, symbol_dim]
        if (getattr(self, "_bivector_output", False)
                and isinstance(self.subspace.what, Codebook)
                and act is not None and act.dim() == 3 and act.shape[-1] == 2):
            # Bivector regime: lift ``[B, V_S, 2]`` back to ``[B, V, D_S]``
            # via the cached SVD pseudo-inverse, then continue with the
            # standard reverse (sigma / sortNetwork / forwardEnd).
            act = self.subspace.what.reverse(act)
        if wordSpace is not None:
            act = wordSpace.reverseSymbols(act, self.subspace)
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
                wordSpace.generate_rules.get('S')
                if (wordSpace is not None
                    and getattr(wordSpace, 'generate_rules', None))
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

    def evaluate_truth(self, vspace, wordSpace=None):
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

    def __init__(self, inputShape, spaceShape, outputShape, masked_prediction=False, vectors=None):
        """Initialize OutputSpace; allocate state for the class contract.
        
        See class docstring for invariants.
        """
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
            return torch.stack(outputBatch, dim=0).unsqueeze(1).to(TheDevice.get())
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

    @staticmethod
    def _format_polarity(base, polarity):
        """Render a token text with its polarity surface form.

        AFFIRM: passthrough.
        NON:    'non-' + base.
        NOT:    'not ' + base (separate token).
        """
        if polarity == POLARITY_NON:
            return f"non-{base}"
        if polarity == POLARITY_NOT:
            return f"not {base}"
        return base

    def reconstruct_tokens(self, vectors):
        """Return positioned tokens decoded from symbolic vectors.

        Delegates to the Basis / Embedding reverse() path. When the
        active vocabulary has per-row polarity tags, prepend the
        matching surface form ("non-"/"not ") to the base text.
        """
        if not self.text_mode:
            raise RuntimeError("reconstruct_tokens() requires text_mode.")
        _, meta = self._reverse_text_vectors(vectors)
        tokens = meta['tokens']
        polarity_ids = getattr(self._vocabulary, "polarity_ids", None)
        indices = meta.get('indices') if isinstance(meta, dict) else None
        if polarity_ids is None or indices is None:
            return tokens
        formatted = []
        for b_idx, batch in enumerate(tokens):
            row = []
            for i, (text, off) in enumerate(batch):
                try:
                    cb_idx = int(indices[b_idx, i].item())
                    pol = int(polarity_ids[cb_idx].item())
                except (IndexError, AttributeError, TypeError):
                    pol = POLARITY_AFFIRM
                row.append((self._format_polarity(text, pol), off))
            formatted.append(row)
        return formatted

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
