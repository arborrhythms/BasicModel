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
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
import torch.optim as optim
from functools import partial
from datetime import datetime

import util
util.init_runtime_env()
TheDevice = util.TheDevice
TheMessage = util.TheMessage

from visualize import Report, TheReport
from util import ProjectPaths, compile, TheXMLConfig, init_config
from embed import WordVectors, PretrainModel
from data import Data, TheData
from Model import Layer, PiLayer, SigmaLayer, InvertibleSigmaLayer, InvertiblePiLayer # Import custom layers from Model.py
from Model import VQLayer, NormLayer, LinearLayer, InvertibleLinearLayer, AttentionLayer
from Model import GammaMem, ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, Loss, epsilon



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

    def __init__(self, inputShape=None, outputShape=None, reshape=False, objectSize=0):
        super().__init__([], 0)
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.reshape = reshape
        self.objectSize = objectSize
        self.raw_output = False

    def forward(self, objects, **kwargs):
        """Identity content pass-through."""
        return objects

    def forwardBegin(self, x, batch):
        """Validate or flatten input at the start of a forward pass."""
        if self.reshape:
            return self.flatten(x, batch, forward=True)
        input_size = self.inputShape[1] + self.objectSize
        assert list(x.shape) == [batch, self.inputShape[0], input_size]
        return x

    def forwardEnd(self, x, batch):
        """Validate or unflatten output at the end of a forward pass."""
        if self.reshape:
            return self.unflatten(x, batch, forward=True)
        output_size = self.outputShape[1] + (0 if self.raw_output else self.objectSize)
        assert list(x.shape) == [batch, self.outputShape[0], output_size], \
            f"forwardEnd: got {list(x.shape)}, expected {[batch, self.outputShape[0], output_size]}"
        return x

    def reverseBegin(self, y, batch):
        """Validate or flatten output-side state at the start of reverse()."""
        if self.reshape:
            return self.flatten(y, batch, forward=False)
        output_size = self.outputShape[1] + (0 if self.raw_output else self.objectSize)
        assert list(y.shape) == [batch, self.outputShape[0], output_size]
        return y

    def reverseEnd(self, y, batch):
        """Validate or unflatten input-side state at the end of reverse()."""
        if self.reshape:
            return self.unflatten(y, batch, forward=False)
        input_size = self.inputShape[1] + self.objectSize
        assert list(y.shape) == [batch, self.inputShape[0], input_size]
        return y

    def flatten(self, x, batch, forward=True):
        """Collapse [batch, nObj, dim] -> [batch, nObj*dim] for a reshaped space."""
        if forward:
            size = (self.inputShape[1] + self.objectSize) * self.inputShape[0]
        else:
            size = (self.outputShape[1] + (0 if self.raw_output else self.objectSize)) * self.outputShape[0]
        return x.reshape(batch, size)

    def unflatten(self, y, batch, forward=True):
        """Restore [batch, nObj*dim] -> [batch, nObj, dim] for a reshaped space."""
        if forward:
            per_obj = self.outputShape[1] + (0 if self.raw_output else self.objectSize)
            return y.reshape(batch, self.outputShape[0], per_obj)
        else:
            per_obj = self.inputShape[1] + self.objectSize
            return y.reshape(batch, self.inputShape[0], per_obj)

class ModelLoss(Loss):
    """Weighted reconstruction loss with separate scales for what/where/when.

    Holds output and reconstruction criteria, combines component losses
    via forward().  total() delegates to forward() through nn.Module.__call__.

    Attributes:
        reverse_scale:    weight of reconstruction vs output loss.
        what_scale:       weight for content dimensions in reconstruction.
        where_scale:      weight for positional encoding dimensions.
        when_scale:       weight for temporal encoding dimensions.
        embedding_scale:  weight of embedding (SBOW) loss in JOINT mode.
        output_criterion: loss function for forward-pass prediction.
    """
    def __init__(self, reverse_scale=0.5,
                 what_scale=0.7, where_scale=0.2, when_scale=0.1,
                 embedding_scale=0.1,
                 certainty=False, nOutput=2,
                 conceptualOrder=0, symbolicOrder=0,
                 nWhere=None, nWhen=None):
        super().__init__()
        self.reverse_scale   = float(reverse_scale or 0.5)
        self.what_scale      = float(what_scale or 0.7)
        self.where_scale     = float(where_scale or 0.2)
        self.when_scale      = float(when_scale or 0.1)
        self.embedding_scale = float(embedding_scale or 0.1)
        self.nWhere = nWhere if nWhere is not None else TheXMLConfig.get("architecture.nWhere")
        self.nWhen  = nWhen if nWhen is not None else TheXMLConfig.get("architecture.nWhen")

        # Select output criterion based on model config
        if certainty:
            self.output_criterion = CertaintyWeightedCrossEntropy()
        elif nOutput <= 2:
            self.output_criterion = nn.MSELoss()
        elif conceptualOrder > 0 or symbolicOrder > 0:
            self.output_criterion = nn.MSELoss()
        else:
            self.output_criterion = nn.CrossEntropyLoss()

    def output(self, pred, target):
        """Compute output loss (prediction vs target)."""
        return self.output_criterion(pred, target)

    def compute(self, pred, target):
        """Compute scaled MSE across what/where/when dimension groups.

        Args:
            pred:   predicted tensor, last dim is embedding.
            target: target tensor, same shape as pred.

        Returns:
            Scalar loss tensor.
        """
        embSize = pred.shape[-1]
        nWhere = self.nWhere
        nWhen  = self.nWhen
        nWhat  = embSize - nWhere - nWhen

        loss = torch.tensor(0.0, device=TheDevice)
        if nWhat > 0:
            loss = loss + self.what_scale * nn.functional.mse_loss(
                pred[..., :nWhat], target[..., :nWhat])
        if nWhere > 0:
            loss = loss + self.where_scale * nn.functional.mse_loss(
                pred[..., nWhat:nWhat+nWhere], target[..., nWhat:nWhat+nWhere])
        if nWhen > 0:
            loss = loss + self.when_scale * nn.functional.mse_loss(
                pred[..., nWhat+nWhere:], target[..., nWhat+nWhere:])
        return loss

    def forward(self, lossOut, lossIn=None, sbow=None):
        """Combine output, reconstruction, and embedding losses.

        Args:
            lossOut: output prediction loss.
            lossIn:  reconstruction loss (None if not reversible).
            sbow:    embedding co-occurrence loss (None if not JOINT).

        Returns:
            Combined total loss.
        """
        total = lossOut
        if lossIn is not None:
            rr = self.reverse_scale
            total = (1 - rr) * lossOut + rr * lossIn
        if sbow is not None:
            total = total + self.embedding_scale * sbow
        return total

    def total(self, lossOut, lossIn=None, sbow=None):
        """Combine component losses. Delegates to forward() via nn.Module.__call__."""
        return self(lossOut, lossIn, sbow)

class VectorSet(nn.Module):
    """Codebook of prototype vectors with vector-quantization (VQ) support.

    Each Space owns a VectorSet that maps continuous input vectors to their
    nearest codebook entries.  Two VQ backends are supported:

    * **customVQ=True** (default): uses ``vector_quantize_pytorch.VectorQuantize``
      with EMA codebook updates and the rotation trick for gradients.
    * **customVQ=False**: a simpler manual VQ loop with explicit LVQ-style
      prototype attraction (controlled by ``eta``).

    Key concepts:
      - ``snapDistance``: distance threshold below which an input snaps to its
        nearest prototype (codebook entry).
      - ``frozen``: list of codebook indices that are locked and no longer
        updated.  Once a codebook entry's activation (tracked by ``codebookAct``)
        exceeds ``freezingTemp``, it gets frozen.
      - ``passThrough=True``: disables quantization entirely; forward() is identity.
      - ``alpha``: jitter factor injected during exploration (high early in
        training, zero at convergence).  Driven by per-neuron sigma via set_sigma().

    The mereological methods (part, whole, overlap, etc.) operate on normalized
    vectors and are used for reasoning about concept parthood relationships.
    """
    nInput           = 0   # number of input vectors per batch element
    nVectors         = 0   # number of active vectors selected via topk (nActive)
    nDim             = 0   # content dimensionality (before encoding overhead)
    embeddingSize    = 0   # nDim + objectSize = full vector width
    vectors          = nn.Parameter(torch.randn([nVectors, embeddingSize]))
    snapDistance     = 0.1 # distance threshold for snapping to prototypes
    eta              = 0.9 # EMA decay for manual VQ updates (customVQ=False)
    codebookAct      = GammaMem(nVectors)  # tracks per-entry activation history
    frozen           = []  # indices of locked codebook entries
    returnOnlyFrozen = False
    freezingTemp     = 0.25  # activation threshold above which entries freeze
    passThrough      = False

    def __init__(self):
        super().__init__()
        self.ergodic = False
        self.sigma_kappa = 0.01

    def getSize(self):
        return self.nVectors

    def create(self, nInput, nVectors, nDim, customVQ=True, signed=False, passThrough=False,
               objectSize=0, where_encoding=None, when_encoding=None):
        """Initialize codebook dimensions.  Call ``addVectors()`` after to allocate entries.

        nVectors here is the active count (topk selection size), not the codebook
        size.  The codebook size is set separately by ``addVectors(nVec=...)``.
        """
        self.nInput      = nInput
        self.nVectors    = nVectors
        self.nDim        = nDim
        self.customVQ    = customVQ
        self.signed      = signed    # True: vectors may have negative components
        self.passThrough = passThrough
        self.alpha       = 0         # exploration jitter (set by set_sigma)
        self.objectSize  = objectSize
        self.where_encoding = where_encoding
        self.when_encoding  = when_encoding
        if nDim != None:
            self.embeddingSize = nDim + self.objectSize
        if passThrough:
            return
    def updateWeights(self, embed_sum, cluster_size):
        # Zero out gradients for frozen indices
        weights = torch.ones(self.vq.codebook_size, device=TheDevice)
        if len(self.frozen) > 0:
            weights[self.frozen] = 0
            if not self.customVQ:
                self.vectors.grad[self.frozen,:] = 0
        return weights
    def freeze(self, activations = None):
        if activations is not None:
            # set the activations that are already frozen
            #unfrozenAct = [a if a not in self.frozen else 0 for a in activations ]
            unfrozenAct = [activations[a].detach().cpu().numpy() if a not in self.frozen else 0 for a in range(self.codebookSize)]
            #unfrozenAct = activations[unfrozen]
            #if len(unfrozen) > 0:
            indexMax = np.argmax(unfrozenAct)
            if unfrozenAct[indexMax] > self.freezingTemp:
                if not indexMax in self.frozen:
                    self.frozen.append(indexMax)
                    TheMessage(f"Frozen activation at {indexMax}")
    def addVectors(self, nVec=1, decay = 0.9):
        """Allocate ``nVec`` codebook entries using the configured VQ backend."""
        self.codebookSize = nVec
        self.codebookAct  = GammaMem(nVec)
        self.frozen       = []
        if self.customVQ:
            self.vq = VectorQuantize(
                dim = self.embeddingSize,
                codebook_size = nVec,
                threshold_ema_dead_code = 1,
                #num_quantizers=1,         # Return the N nearest quantized vectors
                decay = decay,              # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.0,   # the weight on the commitment loss
                #sample_codebook_temp=0.0, #
                #use_cosine_sim=True,
                #learnable_codebook=True,
                #ema_update=False,
                rotation_trick = True      # Set False to use the STE gradient estimator or True to use the rotation trick.
            )
        else:
            # self.vq = VQLayer(
            #    nDim          = nDim,
            #    codebookSize  = nVectors,
            #    numQuantizers = nVectors)
            vec = torch.randn([nVec, self.embeddingSize], device=TheDevice)
            for i in range(0, nVec):
                v = vec[i, :].unsqueeze(0).unsqueeze(0)
                if self.where_encoding is not None:
                    v = self.where_encoding(v)
                if self.when_encoding is not None:
                    v = self.when_encoding(v)
                vec[i, :] = F.normalize(v, p=2, dim=1)
            self.vectors = vec[:, :]
    def forward(self, input):
        """Quantize input vectors against the codebook.

        Input shape: [batch, nInput, nDim] (or embeddingSize).
        Returns: [batch, nInput, embeddingSize] with vectors snapped to
        their nearest codebook entry when within ``snapDistance``.
        Also updates ``codebookAct`` activation tracking and freezing state.
        """
        if self.passThrough:
            return self._passthroughForward(input)
        x     = input
        batch = input.shape[0]
        act   = torch.zeros([batch, self.codebookSize], device=TheDevice)
        if self.customVQ:
            if x.shape[-1] == self.nDim:
                x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], self.objectSize], device=TheDevice)], dim=2)
            y   = torch.reshape(x, [-1, self.embeddingSize])

            quantized, indices, commit_loss = self.vq(y, ema_update_weight=self.updateWeights)

            err = torch.norm(y-quantized, dim=1)
            err = torch.reshape(err, x.shape[0:2])
            indices = indices.reshape(x.shape[0:2])
            quantized = torch.reshape(quantized, x.shape)
            # pick the nVector symbols with the smallest reconstruction error
            # Get the top nVectors smallest reconstruction errors
            values_smallest, indices_smallest = torch.topk(err, k=self.nVectors, dim=1, largest=False)
            for i in range(0, indices_smallest.shape[0]):
                claimed = set()  # track which codebook entries are taken in this batch
                for j in range(0, indices_smallest.shape[1]):
                    cb_idx = indices[i, j].item()
                    if cb_idx in claimed:
                        # Codebook entry already used — keep original vector
                        # so distinct inputs stay distinct downstream.
                        continue
                    claimed.add(cb_idx)
                    if err[i,j] <= self.snapDistance:
                        cosSim = self.unsignedAngle(x[i, j, :].clone(), quantized[i, indices_smallest[i, j], :].clone())
                        x[i, j, :] = quantized[i, indices_smallest[i, j], :]
                        # Is this filling the entire activation matrix properly (on the LHS)?
                        assert torch.all(indices_smallest < self.codebookSize), "activation dimension is not correct."
                        act[i, indices_smallest[i, j]] = cosSim + self.alpha * random.random()
                    else:
                        #TheMessage("codebook miss")
                        x[i, j, :] = quantized[i, indices_smallest[i, j], :]
                        #x[i, j, :] = torch.zeros( [1, 1, self.embeddingSize])
        else:
            dists = self.codebookDistance(x)
            x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], self.objectSize], device=TheDevice)], dim=2)
            # Project the set of input vectors onto the basis vectors (the vector set).
            # Then compute the column norm of the basis, which result in activations (neuron power).
            # The top of those activations become the "Conscious" set of the current space.
            for b in range(0, x.shape[0]):
                for v in range(0, x.shape[1]):
                    nearestDist, nearestIdx = torch.topk(dists[b,v,:], 1, dim =-1, largest=True)
                    err = nearestDist[0]
                    if err <= self.snapDistance:
                        #TheMessage("Using prototype vector")
                        x[b,v,:] = self.vectors[nearestIdx,:]
                        act[b, nearestIdx[0]] = nearestDist[0]
                    #else:
                    #    x[b,v,:] = x[b,v,:]
                    # Update the codebook
                    # Train the closest vector even if it is not used.
                    if self.training:
                        self.vectors[nearestIdx, :] = self.eta * (self.vectors[nearestIdx, :]) + (1-self.eta) *  x[b,v,:]
                        #self.vectors[nearestIdx, :] = F.normalize(self.vectors[nearestIdx, :], p=2, dim=1)
        for b in range(0, x.shape[0]):
            self.codebookAct.delta(act[b,:])
            if self.returnOnlyFrozen:
                unfrozen = [a for a in range(self.codebookSize) if a not in self.frozen]
                act[b, unfrozen] = 0
            self.freeze(activations = self.codebookAct.get())
        return x
    def _passthroughForward(self, x):
        """PassThrough forward: identity transform, skipping quantization."""
        return x
    def set_sigma(self, sigma):
        """Control exploration meta-parameters for VectorSet subclasses."""
        if sigma == 0:
            self.sigma_kappa = 1e6
        else:
            self.sigma_kappa = 0.01 / sigma
    def reverse_raw(self, y):
        """Return raw reverse vector with all subspaces intact (identity for VectorSet)."""
        return y

    def _get_codebook(self):
        """Return the live codebook parameter for reverse-path snapping."""
        return self.vq.codebook

    def reverse(self, y):
        """Reverse: strip positional encoding, snap content to codebook, reassemble.

        Returns full-length vectors with snapped codebook content and
        original positions/times restored, so gradients flow through
        the live codebook parameter.
        """
        if self.passThrough:
            return y
        content = y.clone()
        positions = None
        times = None
        if self.where_encoding is not None:
            content, positions = self.where_encoding.reverse(content)
        if self.when_encoding is not None:
            content, times = self.when_encoding.reverse(content)
        batch = content.shape[0]
        nVec = content.shape[1]
        codebook = self._get_codebook()  # live parameter
        nWhat = self.nDim

        # Snap each vector to nearest codebook entry
        for b in range(batch):
            for v in range(nVec):
                vec = content[b, v, :nWhat]
                if vec.abs().sum().item() < 1e-8:
                    continue  # zero-padding sentinel
                sims = F.cosine_similarity(vec.unsqueeze(0), codebook[:, :nWhat], dim=1)
                idx = sims.argmax().item()
                content[b, v, :nWhat] = codebook[idx, :nWhat]

        # Reassemble: re-encode decoded positions/times back to sin/cos
        embSize = content.shape[-1]
        if self.where_encoding is not None and self.where_encoding.nDim > 0:
            where_idx = self.where_encoding.resolve(embSize)
            if where_idx[0] >= 0 and where_idx[0] < embSize:
                content[:, :, where_idx] = self.where_encoding.encode(positions)
        if self.when_encoding is not None and self.when_encoding.nDim > 0:
            when_idx = self.when_encoding.resolve(embSize)
            if when_idx[0] >= 0 and when_idx[0] < embSize:
                content[:, :, when_idx] = self.when_encoding.encode(times)
        return content

    # The following routine needs also to check if the inner product is positive,
    # otherwise the intersection of the hyperplanes outside of the unit circle
    # may indicate that the two regions are disjoint rather than parts of one another.
    # Therefore, there are three possible results:
    # part(a,b), part(b,a), or disjoint(a,b)
    def conceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        # Normalize vectors A and B
        A_norm = A / A.norm()
        B_norm = B / B.norm()
        # Find orthogonal vector to both A and B (cross product for 3D, generalized for nD)
        cross_prod = torch.linalg.cross(A_norm, B_norm)
        orthogonal_vector = cross_prod / cross_prod.norm()
        # Calculate distance of intersection hyperplane from origin
        distance = orthogonal_vector.norm()
        # Normalize distance to get a measure between 0 and 1
        measure = torch.clamp(distance, 0, 1)
        return measure

        # Example usage
        A = torch.tensor([1.0, 2.0, 3.0])
        B = torch.tensor([3.0, 2.0, 1.0])
        measure = hyperplaneParthood(A, B)

    # The following should be replaced with a mereological framework
    # that operates on the voronoi cells of the LVQ as atoms.
    def perceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        """
        Computes the directional parthood ratio: how much of A is contained within B.

        Both A and B must be in [0, 1]^n and define axis-aligned hyperrectangles with origin.

        Parameters:
            A (torch.Tensor): vector defining hyperrectangle A (outer corner)
            B (torch.Tensor): vector defining hyperrectangle B (outer corner)
            eps (float): small value to avoid division by zero

        Returns:
            float: parthood ratio in [0, 1]
        """
        A, B = A.clamp(0, 1), B.clamp(0, 1)
        ratio = torch.minimum(A / (B + epsilon), torch.ones_like(A))
        return torch.prod(ratio).item()

       # Example usage
        A = torch.tensor([1.0, 2.0, 3.0])
        B = torch.tensor([3.0, 2.0, 1.0])
        measure = hyperrectParthood(A, B)

    def learn(self, x, target_idx, lr=0.01):
        """
        Simple LVQ prototype update
        x: (batch, nDim) input vectors
        target_idx: (batch,) indices of the "correct" prototype
        lr: learning rate
        """
        x = F.normalize(x, p=2, dim=-1)
        selected_vectors = self.vectors[target_idx]  # (batch, nDim )

        # LVQ update: move prototypes toward or away from inputs
        # We will implement "attraction" only for now (classic LVQ1)
        delta = lr * (x - selected_vectors)
        self.vectors.data[target_idx] += delta

        # Optional normalization to keep vectors on the sphere
        self.normalize()
    # --- Vector Insertion ---
    def replace(self, new_vectors):
        #assert(self.nVectors == self.vectors.shape[0])
        if self.customVQ:
            self.vq.codebook = torch.stack(new_vectors, dim=0)
        else:
            #vec = torch.randn([nVec, self.embeddingSize])
            #for i in range(0, nVec):
            #    vec[i, :] = F.normalize(TheObjectEncoding(vec[i, :].unsqueeze(0).unsqueeze(0)), p=2, dim=1)
            self.vectors = new_vectors
    def insert(self, new_vectors):
        """
        Insert one or more new vectors
        new_vectors: (nNew, nDim  )
        """
        new_vectors = F.normalize(new_vectors, p=2, dim =-1)
        for i in range(0, new_vectors.shape[0]):
            v = new_vectors[i, :].unsqueeze(0).unsqueeze(0)
            if self.where_encoding is not None:
                v = self.where_encoding(v)
            if self.when_encoding is not None:
                v = self.when_encoding(v)
            new_vectors[i, :] = v.squeeze(0).squeeze(0)
        self.vectors = torch.cat([self.vectors, new_vectors], dim=0)
        self.nVectors = self.vectors.shape[0]
    # --- Vector Removal ---
    def remove(self, indices):
        """
        Remove vectors by index
        indices: list or tensor of indices to remove
        """
        mask = torch.ones(self.vectors.shape[0], dtype=torch.bool, device=TheDevice)
        mask[indices] = False
        self.vectors = self.vectors[mask]
        self.nVectors = self.vectors.shape[0]
    # --- Fuzzy / Mereology Methods ---
    def norm(self, x):
        return torch.norm(x, dim=-1)
    def normalize(self, x=None):
        if x is None:
            with torch.no_grad():
                if self.signed:
                    self.vectors.data = F.normalize(self.vectors.data, p=2, dim =-1)
                else:
                    self.vectors.data = torch.maximum(torch.minimum(self.vectors.data, 1), 0)
                    self.vectors.data = F.normalize(self.vectors.data, p=2, dim =-1)
        else:
            if self.signed:
                x = F.normalize(x, p=2, dim=-1)
            else:
                x = torch.maximum(torch.minimum(x, torch.tensor(1.0, device=TheDevice)), torch.tensor(0.0, device=TheDevice))
                x = F.normalize(x, p=2, dim=-1)
            return x
    def negate(self, x):
        return 1 - x
    def distance(self, x, y):
        N = self.codebookSize
        dist = (x.T @ y) / N
        return dist
    def codebookDistance(self, x):
        vec = self.vectors[:, 0:self.nDim] if self.objectSize == 0 else self.vectors[:, 0:-self.objectSize]
        vec = vec.to(TheDevice)
        dist = x @ vec.T / self.nDim
        return dist
    def unsignedAngle(self, x, y, dim=-1):
        #xShape = x.shape
        #yShape = y.shape
        #x = torch.reshape(x, (-1, xShape[-1]))
        #y = torch.reshape(y, (-1, yShape[-1]))
        cos_sim = F.cosine_similarity(x, y, dim=-1)
        #dot = torch.sum(x * y, dim=dim)
        #norm_x = torch.norm(x, p=2, dim=dim)
        #norm_y = torch.norm(y, p=2, dim=dim)
        #cos_sim =  dot / (norm_x * norm_y + epsilon)
        # scale 0-1
        return 0.5 * (1-cos_sim) # scale [0-1]
    def equal(self, x, y):
        return 1.0 - self.angle(x, y)
    def part(self, x, y):
        return 1.0 - self.angle(x, y)
    def whole(self, x, y):
        return 1.0 - self.angle(y, x)
    def boundary(self, x, y):
        return torch.abs(self.part(x, y) - self.whole(x, y))
    def overlap(self, x, y):
        return torch.min(self.part(x, y), self.whole(x, y))
    def union(self, x, y):
        return torch.max(x, y)
    def intersection(self, x, y):
        return torch.min(x, y)
class Embedding(VectorSet):
    """VectorSet backed by a differentiable nn.Embedding with online CBOW/SBOW training.

    Loads pretrained weights from a WordVectors artifact (e.g. sentence.pt)
    and wraps them in an nn.Embedding so gradients can flow through the
    forward pass.  ``train_step(words, method)`` runs one sentence-level
    embedding update: CBOW (padded context) or SBOW (leave-one-out centroid).

    This is a direct ``VectorSet`` override: ``forward()`` performs the text
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
        self.optimize_embedding = False  # set by BasicModelFactory from <trainEmbedding>
        object.__setattr__(self, '_model', None)  # back-ref to BasicModel, avoids nn.Module submodule registration
        self.doc_sources = []

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
               objectSize=0, where_encoding=None, when_encoding=None):
        """Initialise from WordVectors or load from embedding_path.

        Accepts the same positional signature as VectorSet.create() so it
        can be used as a drop-in vectorSet member of InputSpace.

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
                       objectSize=objectSize, where_encoding=where_encoding,
                       when_encoding=when_encoding)

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
            new_data = torch.cat([self.wv._vectors.data, new_vec], dim=0)
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

    # --- Public properties (encapsulate internals) --------------------
    @property
    def codebook_weight(self):
        """nn.Embedding weight tensor (read-only reference)."""
        return self.wv._vectors

    @property
    def vocab_keys(self):
        """Ordered list of words in the codebook."""
        return self.wv.index_to_key

    @property
    def embedding_dim(self):
        """Content dimensionality (nWhat)."""
        return self.wv._vectors.shape[1]

    def embedding_parameters(self):
        """Return embedding parameters for optimizer inclusion."""
        return [self.wv._vectors]

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
            codebook = self.wv._vectors.detach() # XXX
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
        codebook = self.wv._vectors
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
            codebook = self.wv._vectors  # refresh after insert
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

    def _decode_offset(self, positions, batch_idx, vector_idx):
        if self.where_encoding is None:
            return None
        sin_val = positions[batch_idx, vector_idx, 0].item()
        cos_val = positions[batch_idx, vector_idx, 1].item()
        if abs(sin_val) < 1e-8 and abs(cos_val) < 1e-8:
            return None
        return round(self.where_encoding.decode(torch.tensor([sin_val, cos_val])).item())

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
        return self.wv._vectors

    def reverse(self, y, return_meta=False):
        """Snap to codebook via super(), then build word/offset metadata.

        Returns: reassembled tensor [batch, nVec, embSize] with snapped
        codebook content and original positions/times restored.
        When return_meta=True, also returns metadata for text rendering.
        """
        # Base class does encoding reverse + snap + reassemble
        content = super().reverse(y)

        # Extract positions from the reassembled tensor for metadata
        embSize = content.shape[-1]
        positions = None
        times = None
        if self.where_encoding is not None and self.where_encoding.nDim > 0:
            where_idx = self.where_encoding.resolve(embSize)
            if where_idx[0] >= 0 and where_idx[0] < embSize:
                positions = content[:, :, where_idx]
        if self.when_encoding is not None and self.when_encoding.nDim > 0:
            when_idx = self.when_encoding.resolve(embSize)
            if when_idx[0] >= 0 and when_idx[0] < embSize:
                times = content[:, :, when_idx]

        batch = content.shape[0]
        nVec = content.shape[1]
        codebook = self.wv._vectors
        words_list = self.wv.index_to_key
        nWhat = codebook.shape[1]

        recovered_words = [[] for _ in range(batch)]
        recovered_offsets = [[] for _ in range(batch)]
        recovered_tokens = [[] for _ in range(batch)]
        for b in range(batch):
            for v in range(nVec):
                offset = self._decode_offset(positions, b, v) if positions is not None else None
                vec = content[b, v, :nWhat]
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
        if return_meta:
            return content, meta
        return content

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
        _, meta = self.reverse(vector, return_meta=True)
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
        return self.wv._vectors[idx].detach().clone()

    def get_space_embedding(self):
        """Return the codebook embedding for the space character ' '."""
        return self.embed_token(' ')

    def get_mask_embedding(self):
        """Return a zero vector the same size as a codebook entry."""
        return torch.zeros(self.wv._vectors.shape[1], device=TheDevice)

class SubSpace:
    """Per-space runtime state container.

    Holds the factored representation (what, where, when, activation) along
    with the encoding objects that describe each factor.  Spaces populate
    a SubSpace during forward() and will eventually pass it as output.

    ``materialize()`` bridges back to the current dense-tensor API so
    existing Model.py math continues to work unchanged.

    In v1 this is a plain Python object (not nn.Module) because it holds
    references rather than learnable parameters.
    """

    __slots__ = (
        'activation', 'what', 'where', 'when',
        'activeEncoding', 'whatEncoding', 'whereEncoding', 'whenEncoding',
        'reshape', 'objectSize',
        'inputShape', 'outputShape',
        'batch',
    )

    def __init__(self, inputShape, outputShape,
                 reshape=False,
                 activeEncoding=None, whatEncoding=None,
                 whereEncoding=None, whenEncoding=None):
        self.activation = None
        self.what = None
        self.where = None
        self.when = None
        self.activeEncoding = activeEncoding
        self.whereEncoding = whereEncoding if whereEncoding is not None else WhereEncoding(0, 0)
        self.whenEncoding = whenEncoding if whenEncoding is not None else WhenEncoding(0, 0)
        self.whatEncoding = whatEncoding
        self.reshape = reshape
        self.objectSize = self.whereEncoding.nDim + self.whenEncoding.nDim
        self.inputShape = inputShape    # [nActive, nDim]
        self.outputShape = outputShape  # [nActive, nDim]
        self.batch = 0

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------

    def getEncodingSize(self, nDim):
        """Full vector width: nDim + objectSize."""
        return nDim + self.objectSize

    def getEmbeddedIO(self):
        """Return (input_emb_size, output_emb_size).

        Accounts for reshape, objectSize, and raw_output.
        """
        we = self.whatEncoding
        input_size = we.inputShape[1] + we.objectSize
        output_size = we.outputShape[1] + (0 if we.raw_output else we.objectSize)
        if we.reshape:
            input_size *= we.inputShape[0]
            output_size *= we.outputShape[0]
        return input_size, output_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        """Shape of the what tensor (or None if unset)."""
        if self.what is None:
            return None
        if isinstance(self.what, torch.Tensor):
            return self.what.shape
        return None

    # ------------------------------------------------------------------
    # Reshape / validate helpers (delegates to whatEncoding)
    # ------------------------------------------------------------------

    def forwardBegin(self, x):
        """Validate/reshape input at the start of the forward pass."""
        self.batch = x.shape[0]
        return self.whatEncoding.forwardBegin(x, self.batch)

    def forwardEnd(self, x):
        """Validate/unflatten output at the end of the forward pass."""
        return self.whatEncoding.forwardEnd(x, self.batch)

    def reverseBegin(self, y):
        """Validate/reshape at the start of the reverse pass (output-side)."""
        self.batch = y.shape[0]
        return self.whatEncoding.reverseBegin(y, self.batch)

    def reverseEnd(self, y):
        """Validate/unflatten at the end of the reverse pass (input-side)."""
        return self.whatEncoding.reverseEnd(y, self.batch)

    def flatten(self, x, forward=True):
        """Collapse [batch, nObj, dim] -> [batch, nObj*dim]."""
        return self.whatEncoding.flatten(x, self.batch, forward=forward)

    def unflatten(self, y, forward=True):
        """Restore [batch, nObj*dim] -> [batch, nObj, dim]."""
        return self.whatEncoding.unflatten(y, self.batch, forward=forward)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_tensor(cls, tensor, *, reshape=False,
                    inputShape, outputShape, **kw):
        """Wrap an existing dense tensor into a SubSpace."""
        ss = cls(reshape=reshape,
                 inputShape=inputShape, outputShape=outputShape, **kw)
        ss.what = tensor
        ss.batch = tensor.shape[0]
        return ss

    @classmethod
    def from_components(cls, *, what=None, activation=None,
                        where=None, when=None, **kw):
        """Build a SubSpace with explicit factor values."""
        ss = cls(**kw)
        ss.what = what
        ss.activation = activation
        ss.where = where
        ss.when = when
        return ss

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(self):
        """Return the dense tensor expected by current model code.

        - If ``what`` is a plain tensor, return it directly.
        - If ``what`` is a VectorSet and ``activation`` is set, use
          activation to select elements (quantized path).
        - v1: activation is NOT multiplied into what — kept explicit.
        - Returns None when ``what`` is unset.
        """
        if self.what is None:
            return None
        if isinstance(self.what, torch.Tensor):
            return self.what
        # VectorSet path — return quantized output directly.
        # In v1 the VectorSet's forward() already selects the top-k;
        # SubSpace just holds the reference.  Full sparse selection
        # via activation is deferred to a later phase.
        if hasattr(self.what, 'vectors') and callable(getattr(self.what, 'forward', None)):
            # Return the codebook vectors as the dense stand-in.
            # Callers that need VQ output go through the VectorSet directly.
            return None
        return None

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
      - ``quantized``: when True, input vectors are quantized against the codebook
        (VectorSet) after the main layer transformation.

    ``getEmbeddedIO()`` returns (input_dim, output_dim) for this space's layers.
    When reshape=False these are the per-object embedding sizes; when reshape=True
    they are multiplied by the respective object counts.  OutputSpace overrides
    this to use raw target dimensions (no encoding overhead on output).

    ``set_sigma(sigma)`` propagates exploration meta-parameters (1=explore, 0=suppress)
    from BasicModel down to all layers and VectorSets in this space.
    """
    name         = ""
    activation   = None
    config_section = None  # set by subclasses

    def __init__(self, inputShape, outputShape, nVectors, customVQ=True):
        super(Space, self).__init__()
        section = self.config_section
        self.inputShape   = inputShape   # [nObjects, nDim] for input
        self.outputShape  = outputShape  # [nObjects, nDim] for output
        self.nVectors     = nVectors     # codebook size (total vectors in the space)
        self.nDim         = outputShape[1]  # content dimensionality (derived from outputShape)
        self.reshape      = TheXMLConfig.get("architecture.reshape")
        self.reversible   = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
        self.processSymbols = TheXMLConfig.get("architecture.processSymbols")
        self.quantized    = TheXMLConfig.space(section, "quantized")
        # self.objectSize needs to be removed . right now it is used by WhatEncoding to take account of the other encodings
        _nWhere = TheXMLConfig.get("architecture.nWhere")
        _nWhen  = TheXMLConfig.get("architecture.nWhen")
        self.objectSize = _nWhere + _nWhen
        self.subspace     = SubSpace(reshape=self.reshape,
                                     inputShape=inputShape, outputShape=outputShape,
                                     whatEncoding=WhatEncoding(inputShape, outputShape, reshape=self.reshape, objectSize=self.objectSize),
                                     whereEncoding=WhereEncoding(TheXMLConfig.get("architecture.nObjects"), _nWhere),
                                     whenEncoding=WhenEncoding(10000, _nWhen))
        self.embeddingSize = self.subspace.getEncodingSize(self.nDim)
        self.vectorSet    = nn.ModuleList()  # holds this space's VectorSet (accessed via self.vectors())
        self.customVQ     = customVQ
        self.params = []   # parameters for the optimizer (excludes temperature params)
        self.layers = nn.ModuleList()   # layer instances for paramUpdate() delegation
        self._register_requirements()

    def _register_requirements(self):
        """Register base-class config requirements."""
        section_name = self.config_section
        nV = self.nVectors
        nA = self.outputShape[0]
        TheXMLConfig.require(
            lambda cfg, _nv=nV, _na=nA: _nv == 0 or _nv >= _na,
            f"{section_name}: nVectors ({nV}) must be >= nActive ({nA})"
        )

    def getEmbeddedIO(self):
        """Return (input_dim, output_dim) for reshape/validation and layer construction.

        Delegates to ``self.subspace.getEmbeddedIO()`` which uses local
        objectSize from this subspace's WhatEncoding.
        """
        return self.subspace.getEmbeddedIO()
    def lookup(self, x):
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.zeros([1,1, TheXMLConfig.space("ConceptualSpace", "nDim")], device=TheDevice), x[:,:,1:]], dim=2)
        output, index, _ = self.vectors().vq(x)
        #output[:,:,0:conceptDim] = output[:,:,0:conceptDim] * activation  # multiply the codebook vector by the activation
        return output
    def dereference(self, symbols):
        # we get [ batch x nConcepts x symbolEmbedding ],
        # and must compute [ batch x nConcepts x conceptEmbedding ]
        batch = symbols.shape[0]
        nActive = self.outputShape[0]
        assert list(symbols.shape) == [batch, nActive, TheXMLConfig.space("SymbolicSpace", "nDim") + self.objectSize], "Incorrect input size for dereference"
        input,_ = self.getEmbeddedIO()
        objects = torch.zeros(batch, nActive, self.embeddingSize, device=TheDevice)
        for b in range(batch):
            for s in range(nActive):
                x = self.lookup(symbols[b,s,:])
                objects[b,s,:] = x
        assert list(objects.shape) == [batch, nActive, self.embeddingSize], "Incorrect output size for dereference"
        return objects

    def stats(self, x):
        #codebookUse = self.vectors().codebookUse
        #TheMessage(f"{self.name} Codebook activation: { np.sum(self.vectors().codebookAct.get()) }")
        return
    def vectors(self):
        """Accessor for this space's VectorSet (first element of the ModuleList)."""
        return self.vectorSet[0]
    def createVectorSet(self, quantized=True):
        vs = VectorSet()
        vs.create(self.inputShape[0], self.outputShape[0], self.nDim, passThrough=not quantized,
                    objectSize=self.objectSize,
                    where_encoding=self.subspace.whereEncoding,
                    when_encoding=self.subspace.whenEncoding)
        self.vectorSet.append(vs)
    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all layers and VectorSets."""
        for l in self.layers:
            if hasattr(l, 'set_sigma'):
                l.set_sigma(sigma)
        for vs in self.vectorSet:
            if hasattr(vs, 'set_sigma'):
                vs.set_sigma(sigma)
    def getParameters(self):
        return self.params
    def paramUpdate(self):
        for l in self.layers:
            l.paramUpdate()

class InputSpace(Space):
    """Receives the source buffer from Data() and encodes it as vectors.

    For text: delegates tokenization to Lex, which produces a span table
    (start, end, type) over the source buffer.  Each span is encoded as a
    vector with nWhat (token content via VectorSet codebook) and nWhere
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
    def __init__(self, nActiveInput, nActiveOutput=None, model_type="simple"):
        
        section = self.config_section
        nActiveOutput = nActiveOutput if nActiveOutput is not None else TheXMLConfig.space(section, "nActive")
        _inputDim = TheXMLConfig.space(section, "nDim")
        _nVectors = TheXMLConfig.space(section, "nVectors")
        ergodic = TheXMLConfig.get("architecture.ergodic")
        lexer = TheXMLConfig.space(section, "lexer")
        min_frequency = float(TheXMLConfig.data_param("minFrequency", 0.0))
        neg_samples = int(TheXMLConfig.training("negSamples", 64))
        embedding_path = TheXMLConfig.get("architecture.embeddingPath", None) or None
        # inputShape uses the data's native dimension (e.g. 784 for MNIST);
        # outputShape uses nDim from config (set from XML nDim).
        data = TheData
        dataDim     = data.getInputSize() if data.train_input else _inputDim
        inputShape  = [nActiveInput, dataDim]
        outputShape = [nActiveOutput, _inputDim]
        super().__init__(inputShape, outputShape, _nVectors)
        # InputSpace never reshapes — it operates on raw [batch, nObj, dim] tensors.
        # Reshape is applied by downstream spaces (Perceptual, Conceptual, etc.).
        self.reshape = False
        self.subspace.whatEncoding.reshape = False
        self.data = data
        self.model_type = model_type
        self.lexer = lexer  # "word", "sentence", or "grammar" — selects .cfg file
        self.ergodic = ergodic
        self.min_frequency = float(min_frequency)
        if model_type == "embedding":
            source = data.train_input if data.train_input else None
            vs = Embedding()
            vs.ergodic = ergodic
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim, embedding_path=embedding_path, source=source, min_frequency=min_frequency, neg_samples=neg_samples,
                       objectSize=self.objectSize, where_encoding=self.subspace.whereEncoding, when_encoding=self.subspace.whenEncoding)
            self.nDim = vs.nDim
            self.outputShape = [self.outputShape[0], _inputDim]
            self.subspace.outputShape = self.outputShape
            self.subspace.whatEncoding.outputShape = self.outputShape
            self.vectorSet.append(vs)
            self.doc_spans = vs.doc_spans
            self.doc_sources = vs.doc_sources
            if data.train_input and self.subspace.whereEncoding.nDim > 0:
                maxP = max(self.subspace.whereEncoding.maxVal, data.inputLength)
                self.subspace.whereEncoding.maxVal = maxP
                self.subspace.whereEncoding.div_term = 2 * math.pi / maxP
        elif model_type == "passthrough":
            vs = VectorSet()
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim, passThrough=True,
                      objectSize=self.objectSize, where_encoding=self.subspace.whereEncoding,
                      when_encoding=self.subspace.whenEncoding)
            self.vectorSet.append(vs)
        elif model_type == "vq":
            vs = VectorSet()
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim,
                      objectSize=self.objectSize, where_encoding=self.subspace.whereEncoding,
                      when_encoding=self.subspace.whenEncoding)
            self.vectorSet.append(vs)
        else:  # "simple"
            self.createVectorSet(quantized=self.quantized)
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = False
        self._output_perm = None
        self._output_perm_inv = None
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
            return self.input

        # Reset positional counter for each forward pass
        self.subspace.whereEncoding.p = 0

        batch = input.shape[0]
        emb = self.vectors()
        if not isinstance(emb, Embedding):
            assert list(input.shape) == [batch, self.inputShape[0], self.inputShape[1]]
            self.input = emb.forward(input)
            self._forward_input = None
            if self.objectSize > 0:
                self.input = self.subspace.encode(self.input)
        else:
            self.input, meta = emb.forward(input, return_meta=True)
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

        _, output = self.getEmbeddedIO()
        assert list(self.input.shape) == [batch, self.outputShape[0], output]

        # Permute flat dimensions so downstream Givens chains mix all tokens
        if self._output_perm is not None:
            B = self.input.shape[0]
            flat = self.input.reshape(B, -1)
            flat = flat[:, self._output_perm]
            self.input = flat.reshape(B, self._output_perm_nActive, self._output_perm_embDim)

        return self.input

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
        masked = embedded.expand(N, -1, -1).clone()  # [N, nVec, embSize]

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

    def reverse(self, y):
        # Undo the flat permutation before reversing
        if self._output_perm_inv is not None:
            B = y.shape[0]
            flat = y.reshape(B, -1)
            flat = flat[:, self._output_perm_inv]
            y = flat.reshape(B, self._output_perm_nActive, self._output_perm_embDim)
        y = self.subspace.reverseBegin(y)
        # Store full vector (all subspaces) for MSE loss BEFORE partitioning
        raw = self.vectors().reverse_raw(y)
        self.reconstructed = raw.detach()
        # Partition into nWhat/nWhere/nWhen for display
        if isinstance(self.vectors(), Embedding):
            self.input, self._recovered_input = self.vectors().reverse(
                y, return_meta=True)
        else:
            self.input = self.vectors().reverse(y)
            self._recovered_input = None
        return self.input

    def reconstruct_data(self, text=False):
        """Render the last recovered text state stored on InputSpace."""
        if getattr(self, '_recovered_input', None) is None:
            raise RuntimeError("reconstruct_data() called before reverse()")
        return self.vectors().reconstruct_data(self._recovered_input, text=text)

    def reconstruct_to_buffer(self, buf_size=None):
        """Render the last recovered text buffer stored on InputSpace."""
        if getattr(self, '_recovered_input', None) is None:
            raise RuntimeError("reconstruct_to_buffer() called before reverse()")
        return self.vectors().reconstruct_to_buffer(
            self._recovered_input, buf_size=buf_size)

    def get_forward_meta(self):
        """Return the last forward-pass lexical metadata for text input."""
        return getattr(self, '_forward_input', None)

    def get_recovered_word(self, batch_idx, position):
        """Return one recovered token from the last InputSpace.reverse()."""
        if getattr(self, '_recovered_input', None) is None:
            return None
        return self.vectors().get_recovered_word(
            self._recovered_input, batch_idx, position)

    # ------------------------------------------------------------------
    # Training policy — InputSpace decides WHEN, Embedding does HOW
    # ------------------------------------------------------------------

    def train_embeddings(self, words, method='CBOW'):
        """Run one CBOW/SBOW gradient step if words are available."""
        emb = self.vectors()
        if isinstance(emb, Embedding) and words:
            return emb.train_step(words, method=method)
        return None

    def sbow_loss(self, words):
        """Return SBOW loss tensor for joint optimization (no backward/step)."""
        emb = self.vectors()
        if isinstance(emb, Embedding) and words:
            return emb.sbow_loss(words)
        return None

    def _snapshot_embeddings(self):
        """Return the current WordVectors (no-op, vectors are always live)."""
        emb = self.vectors()
        if isinstance(emb, Embedding):
            return emb.wv
        return None

    def set_embedding_sigma(self, sigma):
        """Control exploration noise on the embedding."""
        emb = self.vectors()
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
            embedded = self.forward(inputTensor)  # [1, nVec, embSize]

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
        return self.vectors().predict(vector)

    # ------------------------------------------------------------------
    # ARIR helpers
    # ------------------------------------------------------------------

    def embed_token(self, word):
        """Delegates to Embedding.embed_token()."""
        return self.vectors().embed_token(word)

    def get_space_embedding(self):
        """Delegates to Embedding.get_space_embedding()."""
        return self.vectors().get_space_embedding()

    def get_mask_embedding(self):
        """Delegates to Embedding.get_mask_embedding()."""
        return self.vectors().get_mask_embedding()

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
        embSize = self.vectors().embeddingSize
        nWhat = self.vectors().embedding_dim

        if self._arir_cursor is None:
            # ── First call: embed seed, prepare buffer ──────────────
            inputTensor = self.prepInput(inputData)
            embedded = self.forward(inputTensor)  # [1, nVec, embSize]
            self._arir_embedded = embedded.clone().detach()

            # Read span count from the lex pass
            meta = getattr(self, '_forward_input', None) or {}
            counts = meta.get('span_counts', [])
            seed_len = counts[0] if counts else 1

            # Read byte offset from the lex pass
            offsets = meta.get('final_offsets', [])
            self._arir_byte_offset = offsets[0] if offsets else 0

            # Fill future positions (seed_len .. nVec-1) with NULL embeddings
            null_emb = self.vectors().embed_token("\x00")
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

    def __init__(self, nActiveInput, nActiveOutput=None, inputDim=None):

        section = self.config_section
        nActiveOutput = nActiveOutput if nActiveOutput is not None else TheXMLConfig.space(section, "nActive")
        passThrough = TheXMLConfig.space(section, "passThrough")
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        naive = TheXMLConfig.get("architecture.naive")
        _input_dim = inputDim if inputDim is not None else TheXMLConfig.space("InputSpace", "nDim")
        _percept_dim = TheXMLConfig.space(section, "nDim")
        _nVectors = TheXMLConfig.space(section, "nVectors")
        inputShape  = [nActiveInput, _input_dim]
        outputShape = [nActiveOutput, _percept_dim]
        super().__init__(inputShape, outputShape, _nVectors)
        self.passThrough = passThrough
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        self.invertible = invertible
        if passThrough:
            return
        input, output = self.getEmbeddedIO()
        unflatOutput = output
        # InvertiblePiLayer doubles its output; halve the layer's nOutput
        # so the 2x interleaving produces the correct unflatten width.
        if self.reshape and self.reversible:
            output = output // 2
        self.attention = AttentionLayer(unflatOutput, unflatOutput)
        if invertible and not naive:
            if self.reshape:
                if 2 * input != output:
                    raise ValueError(
                        f"invertible=True with reshape requires nOutput == 2*nInput, "
                        f"but got nInput={input}, nOutput={output}.")
            else:
                if input != output:
                    raise ValueError(
                        f"invertible=True without reshape requires nInput == nOutput, "
                        f"but got nInput={input}, nOutput={output}.")
        if self.reversible:
            if invertible:
                self.pi  = InvertiblePiLayer(input, output, naive=naive, ergodic=ergodic)
                self.forwardPi, self.reversePi = self.pi.forward, self.pi.reverse
                self.params = self.pi.getParameters()
                self.layers = nn.ModuleList([self.pi])
                # InvertiblePiLayer doubles sequence in 3D or flat dim in 2D
                if not self.reshape:
                    self.outputShape = [2 * self.inputShape[0], self.outputShape[1]]
                    self.subspace.outputShape = self.outputShape
                    self.subspace.whatEncoding.outputShape = self.outputShape
            else:
                self.pi1 = InvertiblePiLayer(input, output, naive=naive, ergodic=ergodic)
                self.pi2 = InvertiblePiLayer(input, output, naive=naive, ergodic=ergodic)
                self.forwardPi, self.reversePi = self.pi1.forward, self.pi2.reverse
                self.params = self.pi1.getParameters() + self.pi2.getParameters()
                self.layers = nn.ModuleList([self.pi1, self.pi2])
        else:
            self.pi        = PiLayer(input, output, naive=naive, ergodic=ergodic)
            self.forwardPi = self.pi.forward
            self.params = self.pi.getParameters()
            self.layers = nn.ModuleList([self.pi])
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.createVectorSet(quantized=self.quantized)
    def distance(self, x, y):
        return torch.prod( [1-x, 1-y] )
    def certainty(self, x):
        pass
    def forward(self, x):
        """Perception: map input vectors to percepts via PiLayer + optional attention + VQ."""
        if self.passThrough:
            return x
        x = self.subspace.forwardBegin(x)
        x = self.forwardPi(x)
        if self.hasAttention:
            x = self.attention.forward(x)
        if self.quantized:
            x  = self.vectors().forward(x)
        if self.processSymbols:
            # Collapse content dims to scalar activation, keep positional encoding
            encoding = x[:,:,-self.objectSize:]
            x = torch.norm( x[:,:,0:-self.objectSize], dim=2 ) / (2*self.outputShape[0])
            x = x.unsqueeze(-1)
            x = torch.concatenate((x, encoding), dim=2)
        self.percepts = self.subspace.forwardEnd(x)
        return self.percepts
    def reverse(self, y):
        """Manifesting: reconstruct input vectors from percepts via reverse PiLayer."""
        if self.passThrough:
            return y
        if self.reversible:
            y = self.subspace.reverseBegin(y)
            y = self.reversePi(y)
            y = self.subspace.reverseEnd(y)
        return y
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

    def __init__(self, nActiveInput, nActiveOutput=None):
        section = self.config_section
        nActiveOutput = nActiveOutput if nActiveOutput is not None else TheXMLConfig.space(section, "nActive")
        ergodic = TheXMLConfig.get("architecture.ergodic")
        hasAttention = TheXMLConfig.space(section, "hasAttention")
        invertible = TheXMLConfig.space(section, "invertible")
        hasNorm = TheXMLConfig.space(section, "hasNorm")
        naive = TheXMLConfig.get("architecture.naive")
        _percept_dim = TheXMLConfig.space("PerceptualSpace", "nDim")
        _concept_dim = TheXMLConfig.space(section, "nDim")
        _nVectors = TheXMLConfig.space(section, "nVectors")
        inputShape  = [nActiveInput, _percept_dim]
        outputShape = [nActiveOutput, _concept_dim]
        super().__init__(inputShape, outputShape, _nVectors)
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        input, output = self.getEmbeddedIO()
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
                self.sigma = InvertibleSigmaLayer(input, output, naive=naive, ergodic=ergodic)
                self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                self.params = self.sigma.getParameters()
                self.layers = nn.ModuleList([self.sigma])
            else:
                self.sigma1 = InvertibleSigmaLayer(input, output, naive=naive, ergodic=ergodic)
                self.sigma2 = InvertibleSigmaLayer(input, output, naive=naive, ergodic=ergodic)
                # self.sigma1 = SigmaLayer(input, output, ergodic=ergodic)
                # self.sigma2 = SigmaLayer(output, input, ergodic=ergodic)
                self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.reverse
                self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
                self.layers = nn.ModuleList([self.sigma1, self.sigma2])
        else:
            self.sigma = SigmaLayer(input, output, naive=naive, ergodic=ergodic)
            self.forwardSigma = self.sigma.forward
            self.params = self.sigma.getParameters()
            self.layers = nn.ModuleList([self.sigma])
        self.createVectorSet(quantized=self.quantized)
    def distance(self, x, y):
        # This is a dot-product distance that assumes the X are normalized.
        # However, if the X are not normalized, the magnitudes may be taken as a degree of certainty or knowing.
        # In which case, how do they grow from ignorance to certainty?
        # They would do so naturally if the input vectors are normalized.
        # It would also be possible to use a tunable transfer function.
        return x.T @ y
    def certainty(self, x):
        return x.T @ x
    def forward(self, x):
        """Knowing: map percepts to concepts via SigmaLayer + optional attention + VQ."""
        x = self.subspace.forwardBegin(x)
        if self.hasNorm:
            x = self.norm.forward(x)
            if hasattr(self, 'sigma') and isinstance(self.sigma, InvertibleSigmaLayer):
                # Cache norm factors for reverse, pass only normalized content
                self._norm_factors = x[..., -2:]
                x = x[..., :-2]
        y = self.forwardSigma(x)
        if self.hasAttention:
            y = self.attention.forward(y)
        # Quantize against codebook: snap dynamic vectors to static prototypes
        if self.quantized:
            y = self.vectors().forward(y)
        if self.processSymbols:
            # Collapse content dims to scalar activation, keep positional encoding
            encoding = y[:,:,-self.objectSize:]
            y = torch.sum(y[:,:,0:-self.objectSize], dim=2) / (2*self.outputShape[0])
            y = y.unsqueeze(-1)
            y = torch.concatenate((y, encoding), dim=2)
        self.concepts = self.subspace.forwardEnd(y)
        return self.concepts
    def reverse(self, y):
        """Visualizing: reconstruct percepts from concepts via reverse SigmaLayer."""
        self.concepts = self.subspace.reverseBegin(y)
        # we are receiving symbols, and we turn them into concepts.
        if self.processSymbols:
            self.concepts = self.dereference(self.concepts)
        if self.hasAttention:
            self.concepts = self.attention.reverse(self.concepts)
        # reverseSigma was built for the full embedding dim (content + objectEncoding),
        # so pass the complete tensor — do NOT strip objectEncoding first.        
        # if self.reshape:
        #    # Flattened path — no object encoding to strip
        #    self.concepts = self.reverseSigma(self.concepts)
        #else:
        #    # preserve the codebook's positional encoding
        #    encoding = self.concepts[:, :, -TheObjectEncoding.objectSize:]
        #    if TheObjectEncoding.objectSize > 0:
        #        self.concepts = self.reverseSigma(self.concepts[:,:,0:-TheObjectEncoding.objectSize])
        #    else:
        #        self.concepts = self.reverseSigma(self.concepts)
        self.concepts = self.reverseSigma(self.concepts)
        if self.hasNorm:
            if hasattr(self, '_norm_factors'):
                # Reattach cached norm factors for un-normalization
                self.concepts = torch.cat([self.concepts, self._norm_factors], dim=-1)
            self.concepts = self.norm.reverse(self.concepts)
        #self.concepts = torch.concatenate((self.concepts, encoding), dim=2)
        self.concepts = self.subspace.reverseEnd(self.concepts)
        return self.concepts
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

    def __init__(self, nActiveInput, nActiveOutput=None, conceptualSpace=None):

        section = self.config_section
        nActiveOutput = nActiveOutput if nActiveOutput is not None else TheXMLConfig.space(section, "nActive")
        passThrough = TheXMLConfig.space(section, "passThrough")
        _concept_dim = TheXMLConfig.space("ConceptualSpace", "nDim")
        _symbol_dim = TheXMLConfig.space(section, "nDim")
        if passThrough:
            _symbol_dim = _concept_dim  # passthrough flows at concept dim
        _nVectors = TheXMLConfig.space(section, "nVectors")
        inputShape  = [nActiveInput, _concept_dim]
        outputShape = [nActiveOutput, _symbol_dim]
        super().__init__(inputShape, outputShape, _nVectors, customVQ=True)
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
    def computeActivation(self, x):
        # we get [ batch x nConcepts x conceptEmbedding ],
        # and must compute [ batch x nConcepts x symbolEmbedding ]
        activations = torch.norm( x[:,:,0:self.outputShape[1]] , dim=2)
        activations = activations.unsqueeze(2)
        activations = torch.concatenate((activations, x[:,:,self.inputShape[1]:]), dim=2)
        return activations

    def forward(self, x):
        """Naming: convert concept vectors to discrete symbols."""
        self.symbols = self.subspace.forwardBegin(x)
        if not self.passThrough:
            if self.processSymbols:
                self.symbols = self.computeActivation(self.symbols)
            self.symbols = self.discretize(self.symbols)
        self.symbols = self.subspace.forwardEnd(self.symbols)
        if self.quantized:
            self.symbols  = self.vectors().forward(self.symbols)
        return self.symbols
    def reverse(self, y):
        """Interpretation: map symbols back to concept vectors (via codebook dereference)."""
        self.symbols = self.subspace.reverseBegin(y)
        if not self.passThrough:
            if self.processSymbols:
                self.symbols = self.conceptualSpace.dereference(self.symbols)
        self.symbols = self.subspace.reverseEnd(self.symbols)
        return self.symbols

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

    def __init__(self, nActiveInput, nActiveOutput=None, conceptualSpace=None):
        section = self.config_section
        nActiveOutput = nActiveOutput if nActiveOutput is not None else TheXMLConfig.space(section, "nActive")
        _symbol_dim = TheXMLConfig.space("SymbolicSpace", "nDim")
        _word_dim = TheXMLConfig.space(section, "nDim")
        _nVectors = TheXMLConfig.space(section, "nVectors")
        inputShape  = [nActiveInput, _symbol_dim]
        outputShape = [nActiveOutput, _word_dim]
        super().__init__(inputShape, outputShape, _nVectors, customVQ=False)
        self.processSymbols = True  # class invariant
        assert(inputShape[0] == outputShape[0]) # 1:1 mapping
        self.conceptualSpace = conceptualSpace
        #self.mapping     = TODO(inputShape[1], nDim, soft=False)
    def distance(self, x, y):
        return x == y
    def certainty(self, x):
        return x.T @ x
    def computeActivation(self, x):
        # we get [ batch x nConcepts x conceptEmbedding ],
        # and must compute [ batch x nConcepts x symbolEmbedding ]
        if x.size(-1) != TheXMLConfig.space("SymbolicSpace", "nDim"):
            activations = torch.norm( x[:,:,0:self.outputShape[1]] , dim=2)
            activations = activations.unsqueeze(2)
            activations = torch.concatenate((activations, x[:,:,self.inputShape[1]:]), dim=2)
        else:
            activations = x
        return activations
    # Naming
    def forward(self, x):
        self.symbols = self.subspace.forwardBegin(x)
        #self.symbols = self.computeActivation(self.symbols)
        self.symbols = self.subspace.forwardEnd(self.symbols)
        return self.symbols
    # Interpretation
    def reverse(self, y):
        self.symbols = self.subspace.reverseBegin(y)
        self.symbols = self.subspace.reverseEnd(self.symbols)
        return self.symbols

    @staticmethod
    def test():
        pass
class OutputSpace(Space):
    """Maps symbolic vectors to task targets (classification logits, regression values).

    In the forward data flow: SymbolicSpace -> **OutputSpace** -> loss.
    Uses a LinearLayer to project the (flattened) symbolic representation down
    to the target dimensionality.  Always uses reshape=True since the number of
    input objects (symbols) typically differs from the number of outputs.

    Overrides ``getEmbeddedIO()`` so the output side uses raw target dimensions
    (no encoding overhead), since targets are not embedded.

    ``text_mode``: when enabled via ``set_text_mode()``, supports reconstructing
    text from symbolic vectors by snapping to the nearest codebook entry and
    recovering byte-offset positions.
    """
    name = "Outputs"
    config_section = "OutputSpace"
    text_mode = False
    def __init__(self, nActiveInput, nActiveOutput=None, masked_prediction=False, vectors=None):
        section = self.config_section
        nActiveOutput = nActiveOutput if nActiveOutput is not None else TheXMLConfig.space(section, "nActive")
        _symbol_dim = TheXMLConfig.space("SymbolicSpace", "nDim")
        _output_dim = TheXMLConfig.space(section, "nDim")
        _nVectors = TheXMLConfig.space(section, "nVectors")
        inputShape  = [nActiveInput, _symbol_dim]
        outputShape = [nActiveOutput, _output_dim]
        self.masked_prediction = masked_prediction
        super().__init__(inputShape, outputShape, _nVectors)
        # OutputSpace always reshapes: input/output object counts typically differ,
        # so the layer must operate on flattened [batch, nObj*dim] tensors.
        self.reshape = True
        self.subspace.whatEncoding.reshape = True
        # Output targets are not embedded — raw_output skips objectSize on the output side
        if not masked_prediction:
            self.subspace.whatEncoding.raw_output = True
        self.data = TheData
        if vectors is not None:
            self.text_mode = isinstance(vectors, Embedding)
            self.vectorSet.append(vectors)
        else:
            self.text_mode = False
        input, output = self.getEmbeddedIO()
        if self.reversible:
            self.linear1 = LinearLayer(input, output)
            self.linear2 = LinearLayer(output, input)
            self.forwardLinear, self.reverseLinear = self.linear1.forward, self.linear2.forward
            #self.linear = InvertibleLinearLayer(input, output)
            #self.forwardLinear, self.reverseLinear = self.linear.forward, self.linear.reverse
        else:
            self.forwardLinear = LinearLayer(input, output)
        self.params = list(self.parameters())
        self.layers = nn.ModuleList([self.forwardLinear] if not self.reversible else [self.linear1, self.linear2])
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
    def forward(self, x):
        """Acting: project flattened symbols to task output via LinearLayer."""
        y = self.subspace.forwardBegin(x)
        output = self.forwardLinear(y)
        output = self.subspace.forwardEnd(output)
        if self.quantized:
            self.output  = self.vectors().output(self.percepts)
        self.predicted = output.detach()
        return output
    def reverse(self, y):
        """Being acted upon: map output back to symbolic space via reverse LinearLayer."""
        y = self.subspace.reverseBegin(y)
        y = self.reverseLinear(y)
        output = self.subspace.reverseEnd(y)
        return output
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
        """Share InputSpace's VectorSet so OutputSpace can reconstruct text.

        Convenience method for tests. Production code passes vectors= to __init__.
        """
        vs = input_space.vectors()
        self.text_mode = True
        if len(self.vectorSet) == 0:
            self.vectorSet.append(vs)
        else:
            self.vectorSet[0] = vs

    def reconstruct_tokens(self, vectors):
        """Return positioned tokens decoded from symbolic vectors.

        Delegates to the VectorSet/Embedding reverse() path.
        """
        if not self.text_mode:
            raise RuntimeError("reconstruct_tokens() requires text_mode.")
        _, meta = self.vectors().reverse(vectors, return_meta=True)
        return meta['tokens']

    def reconstruct_data(self, vectors):
        """Reconstruct words and positions from symbolic vectors.

        Delegates to the VectorSet/Embedding reverse() which handles
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
        _, meta = self.vectors().reverse(vectors, return_meta=True)
        return self.vectors().reconstruct_to_buffer(
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

class BaseModel(nn.Module):
    """Shared training, plotting, and persistence infrastructure for all models."""
    name           = "BaseModel"
    spaces         = []
    reversible    = False
    plot           = False
    _optimizer     = None

    @staticmethod
    def load_config(config_path=None):
        """Load model settings from an XML config file.

        Delegates to XMLConfig._parse_xml().  Returns a dict of dicts;
        missing fields are filled by create_from_config() using model.xml.
        """
        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "model.xml")
        from util import XMLConfig
        return XMLConfig._parse_xml(config_path)

    @staticmethod
    def from_config(config_path=None, model_type=None, data=None):
        """Factory: create the right model type from XML config."""
        model = BasicModel()
        cfg = model.create_from_config(config_path, model_type=model_type, data=data)
        return model, cfg

    def create(self, **kwargs):
        """Override in subclasses to build model architecture."""
        pass

    def getOptimizer(self, lr=0.01):
        """Build an Adam optimizer over all space parameters.

        Uses getParameters() from each Space (the universal training contract),
        which excludes temperature params managed by alpha_update.
        Falls back to standard PyTorch parameters() when not in ergodic mode.

        When trainEmbedding is NONE or ARLM, embedding parameters are excluded
        from the optimizer.
        """
        if getattr(self, 'ergodic', True):
            params = []
            for s in self.spaces:
                params.extend(s.getParameters())
        else:
            params = list(self.parameters())
        # Exclude embedding params when trainEmbedding is NONE or ARLM
        if not getattr(self, 'optimize_embedding', False):
            exclude = set()
            if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.vectors(), Embedding):
                for p in self.inputSpace.vectors().embedding_parameters():
                    exclude.add(p.data_ptr())
            if exclude:
                params = [p for p in params if p.data_ptr() not in exclude]
        return optim.Adam(params, lr=lr)

    def rebuild_optimizer(self):
        """Rebuild the main optimizer after codebook expansion."""
        if self._optimizer is None:
            return
        lr = self._optimizer.param_groups[0]['lr']
        self._optimizer = self.getOptimizer(lr=lr)

    def run(self, numTrials=1, numEpochs=1, batchSize=10, lr=0.001):
        """Run multiple independent trials, recreating the model each time.

        Each trial calls create_from_config() to rebuild from scratch so
        results are statistically independent.  If the model was already
        configured by the caller (e.g. manually built models without
        _config_path), trial 0 skips recreation and uses the model as-is.
        """
        acc = np.zeros([numTrials, numEpochs])
        has_config = hasattr(self, '_config_path') and self._config_path is not None
        already_configured = len(list(self.parameters())) > 0
        print(f"\n\n==== {self.name} ====")
        for trial in range(numTrials):
            print(f"\nTrial [{trial + 1}/{numTrials}]")
            if has_config and (trial > 0 or not already_configured):
                self.create_from_config(self._config_path, data=self._config_data)
            acc[trial, :] = self.runTrial(numEpochs=numEpochs, batchSize=batchSize, lr=lr)
        np.savetxt(ProjectPaths.output_path(f"{self.name}.csv"), np.array(acc), delimiter=",")
        return acc

    def paramUpdate(self):
        """Delegate ergodic in-place parameter updates to all spaces."""
        for s in self.spaces:
            s.paramUpdate()

    def set_sigma(self, sigma):
        """Propagate exploration meta-parameters to all spaces."""
        for s in self.spaces:
            s.set_sigma(sigma)

    def _get_embedding(self):
        """Return the Embedding instance if this model uses one, else None."""
        if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.vectors(), Embedding):
            return self.inputSpace.vectors()
        return None

    def save_weights(self, path=None):
        """Persist model weights (excluding embeddings) to disk.

        Embedding weights live in a separate artifact (the .kv/.pt file
        specified by <embeddingPath> in the XML config).  The three files
        — XML config, embedding artifact, weights checkpoint — partition
        the model's behaviour and are managed independently.
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Filter out embedding parameters — they belong to the .kv artifact
        state = {k: v for k, v in self.state_dict().items()
                 if "wv._vectors" not in k}
        torch.save({"state_dict": state}, path)
        print(f"[{self.name}] Weights saved to {path}")

    def save_embeddings(self, path=None):
        """Snapshot current nn.Embedding weights and save the .pt artifact."""
        if path is None:
            path = getattr(self, 'embedding_path', None)
        if path is None:
            return
        emb = self._get_embedding()
        if emb is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        emb.save_embeddings(path)
        print(f"[{self.name}] Embeddings saved to {path}")

    def load_embeddings(self, path=None):
        """Load embedding weights and vocab from a .pt artifact."""
        if path is None:
            path = getattr(self, 'embedding_path', None)
        if path is None:
            return False
        if not os.path.exists(path):
            return False
        emb = self._get_embedding()
        if emb is None:
            return False
        wv = WordVectors.load(path)
        self._restore_vocab(emb, list(wv.index_to_key),
                            counts=wv.counts.tolist(),
                            total_count=int(wv.total_count))
        # Copy loaded weights into the live parameter
        with torch.no_grad():
            emb.wv._vectors.data.copy_(wv._vectors.to(emb.wv._vectors.device))
        print(f"[{self.name}] Embeddings loaded from {path}")
        return True

    def load_weights(self, path=None, strict=False):
        """Load model weights from disk (excluding embeddings).

        Embedding weights are loaded separately from the .kv artifact
        specified by <embeddingPath>.  This method only restores layer
        weights, attention parameters, etc.

        Supports both new format {"state_dict": ...} and legacy format
        (bare state_dict).
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        if not os.path.exists(path):
            print(f"[{self.name}] No checkpoint at {path}, starting fresh")
            return False
        saved = torch.load(path, map_location=TheDevice, weights_only=False)

        if isinstance(saved, dict) and "state_dict" in saved:
            state = saved["state_dict"]
        else:
            state = saved

        try:
            self.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            print(f"[{self.name}] Warning: cannot load {path}: {e}")
            return False

        print(f"[{self.name}] Weights loaded from {path}")
        return True

    def _restore_vocab(self, emb, saved_vocab,
                       counts=None, total_count=0, pending_counts=None):
        """Resize Embedding to match saved vocabulary exactly."""
        dim = emb.wv._vectors.shape[1]
        vocab_size = len(saved_vocab)

        # Rebuild word mappings (shared between wv and pretrain)
        emb.wv.index_to_key = list(saved_vocab)
        emb.wv.key_to_index = {w: i for i, w in enumerate(saved_vocab)}
        emb.pretrain.index_to_key = emb.wv.index_to_key
        emb.pretrain.key_to_index = emb.wv.key_to_index
        emb.wv._vectors = nn.Parameter(
            torch.zeros(vocab_size, dim, device=TheDevice), requires_grad=True)
        emb.wv.counts = (np.asarray(counts, dtype=np.int64) if counts is not None
                         else np.zeros(vocab_size, dtype=np.int64))
        emb.wv.total_count = np.int64(total_count)
        emb._pending_counts = dict(pending_counts) if pending_counts else {}
        emb.wv._normed = None

    def mnistReport(self):
        """Run test epoch, compute per-digit accuracy, and plot."""
        if hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE':
            return torch.zeros(1)  # no classification report for masked prediction
        self.set_sigma(0)  # suppress exploration for evaluation
        _, _, y_pred, last_x_pred = self.runEpoch(split="test")
        if y_pred.dim() == 1 or y_pred.shape[-1] == 1:
            predicted = (y_pred.squeeze() > 0.5).long()
            actual = (self.outputSpace.getTestOutput().squeeze() > 0.5).long()
        else:
            _, predicted = torch.max(y_pred, 1)
            _, actual = torch.max(self.outputSpace.getTestOutput(), 1)

        nClasses = int(actual.max().item()) + 1
        if self.certainty:
            # forwardLinear may be a bound method (reversible=True) or a
            # LinearLayer (reversible=False).  Get the layer either way.
            fwd_layer = (self.outputSpace.linear1
                         if hasattr(self.outputSpace, 'linear1')
                         else self.outputSpace.forwardLinear)
            norms = torch.linalg.norm(fwd_layer.W, dim=0)
            rCorrect = torch.zeros_like(norms)
        else:
            rCorrect = torch.zeros((nClasses))
        for i in range(nClasses):
            total    = (actual == i).sum().item()
            correct  = (actual==i) & (predicted==actual)
            nCorrect = correct.sum().item()
            rCorrect[i] = nCorrect / total if total > 0 else 0.0
            print(f"Correctly predicted {i}: {rCorrect[i]}")
            if self.certainty:
                print(f"Weight norm: {norms[i]}")

        if self.certainty:
            input_matrix = torch.stack((rCorrect, norms))
            correlation_matrix = torch.corrcoef(input_matrix)
            correlation_value = correlation_matrix[0, 1]
            print(f"Pearson Correlation: {correlation_value}")
            TheReport.plotAccuracyAndCertainty(self.name, rCorrect, self.reversible, last_x_pred, TheData.test_output)
        else:
            TheReport.plotAccuracy(self.name, rCorrect)
        return rCorrect

    def _get_sentences(self, split):
        """Return raw sentence strings for a data split.

        All splits store raw strings directly in their input lists.
        Runtime maps to train_input (staged by runtime_batch).
        """
        data = self.inputSpace.data
        if split == "train" or split == "runtime":
            result = data.train_input
        elif split == "test":
            result = data.test_input
        elif split == "validation":
            result = data.validation_input
        else:
            return None
        if result and isinstance(result[0], str):
            return result
        return None

    @staticmethod
    def _bytes_to_text(tensor):
        """Decode a byte tensor (or padded int8 tensor) to a string."""
        if isinstance(tensor, str):
            return tensor
        if tensor.dim() > 1:
            tensor = tensor.squeeze()
        chars = [chr(int(b) & 0xFF) for b in tensor.tolist()]
        return "".join(chars).rstrip("\x00")

    def _reconstructionReport(self):
        """Run a test pass with reverse and report input vs reconstructed text."""
        if hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE':
            return  # masked prediction has variable batch sizes; skip reconstruction report
        self.set_sigma(0)  # suppress exploration for evaluation
        test_input, test_output = self.inputSpace.getTestData()
        _, _, allOut, _ = self.runEpoch(batchSize=len(test_input), split="test")

        rows = []
        # Use reconstruct_data() for lex-based models (embedding vectors, not bytes)
        use_lex_recon = (self.inputSpace.model_type == "embedding" and
                         self.inputSpace.get_recovered_word(0, 0) is not None)
        if use_lex_recon:
            recon_text_list = self.inputSpace.reconstruct_data(text=True)
        for i in range(len(test_input)):
            original = self._bytes_to_text(test_input[i])
            if use_lex_recon:
                recon = recon_text_list[i]
            elif hasattr(self.inputSpace, 'reconstructed'):
                recon = self._bytes_to_text(self.inputSpace.reconstructed[i])
            else:
                recon = "(no reconstruction)"
            # Strip \x00 padding from both sides before comparing words
            orig_words = original.replace("\x00", " ").split()
            recon_words = recon.replace("\x00", " ").split()
            match = orig_words == recon_words
            css = "match" if match else "mismatch"
            label = test_output[i]
            if isinstance(label, torch.Tensor):
                label = label.squeeze().tolist()
            pred_val = allOut[i]
            if pred_val.numel() == 1:
                pred_str = f'{pred_val.item():.4f}'
            else:
                pred_str = f'[{pred_val.shape}]'
            rows.append([
                f'{original}',
                f'<span class="{css}">{recon}</span>',
                f'{label}',
                pred_str,
                f'<span class="{css}">{"Yes" if match else "No"}</span>',
            ])
            print(f"  Input: {original:30s} -> Reconstructed: {recon:30s} Predicted: {pred_str} {'OK' if match else 'MISMATCH'}")

        TheReport.add_table(
            "Input vs Reconstructed",
            ["Input", "Reconstructed", "Label", "Predicted", "Match"],
            rows)

        # Buffer reconstruction via nWhere byte offsets (non-differentiable display)
        recovered_meta = getattr(self.inputSpace, '_recovered_input', None)
        if use_lex_recon and recovered_meta is not None:
            buf_size = max(len(test_input[0].tolist()) if isinstance(test_input[0], torch.Tensor) else 64, 64)
            buffer_strings = self.inputSpace.reconstruct_to_buffer(buf_size=buf_size)
            buf_rows = []
            total_chars = 0
            matching_chars = 0
            for i in range(len(test_input)):
                original = self._bytes_to_text(test_input[i])
                buf_recon = buffer_strings[i] if i < len(buffer_strings) else ""
                # Character-level accuracy
                orig_stripped = original.rstrip('\x00')
                n = max(len(orig_stripped), len(buf_recon))
                chars_match = sum(
                    a == b for a, b in zip(orig_stripped.ljust(n), buf_recon.ljust(n)))
                total_chars += n
                matching_chars += chars_match
                acc = chars_match / max(n, 1) * 100
                css = "match" if acc > 90 else "mismatch"
                buf_rows.append([
                    f'{orig_stripped}',
                    f'{buf_recon}',
                    f'<span class="{css}">{acc:.0f}%</span>',
                ])
                print(f"  Buffer: {orig_stripped:30s} -> {buf_recon:30s} ({acc:.0f}% char accuracy)")
            overall_acc = matching_chars / max(total_chars, 1) * 100
            buf_rows.append(["<strong>Overall</strong>", "", f"<strong>{overall_acc:.1f}%</strong>"])
            TheReport.add_table(
                "Buffer Reconstruction (nWhere placement)",
                ["Original", "Buffer", "Char Accuracy"],
                buf_rows)

            # Push reconstructed data to TheData
            self.inputSpace.data.reconstructed_input = buffer_strings

        # Push reconstructed output predictions to TheData
        if allOut is not None:
            self.inputSpace.data.reconstructed_output = [
                allOut[i].detach().cpu() for i in range(allOut.shape[0])]
class BasicModel(BaseModel):
    """Core model: assembles Spaces into a forward and (optionally) reverse pipeline.

    The forward pass flows:
        InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace

    The reverse pass mirrors it:
        OutputSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace

    Higher-order processing (conceptualOrder, symbolicOrder) inserts additional
    Percept/Concept/Symbol cycles between the first SymbolicSpace and OutputSpace,
    concatenating their symbol outputs before the final projection.

    ``create()`` builds the full space hierarchy.  ``create_from_config()`` is the
    XML-driven factory that reads architecture and training parameters from config,
    then delegates to ``create()``.
    """
    name = "BasicModel"

    def _resolve_artifact_path(self, relpath):
        """Resolve a relative artifact path against the XML config file's directory.

        If relpath is absolute, return as-is.  Otherwise join with the
        directory containing the XML config file.
        """
        if os.path.isabs(relpath):
            return relpath
        config_dir = os.path.dirname(self._config_path)
        return os.path.join(config_dir, relpath)

    def create_from_config(self, config_path=None, model_type=None, data=None):
        """Create the model using settings from an XML config file.

        Loads defaults from model.xml, overlays model-specific config,
        then creates the model and optionally loads saved weights.
        """
        # Store for runTrials() re-creation
        self._config_path = config_path
        self._config_data = data

        # Load defaults from model.xml, overlay model-specific config
        defaults_path = os.path.join(ProjectPaths.DATA_DIR, "model.xml")
        init_config(path=config_path, defaults_path=defaults_path)
        cfg = TheXMLConfig.data

        BasicModelFactory.validate_config(cfg)

        arch = cfg["architecture"]
        _t = TheXMLConfig.training
        _d = TheXMLConfig.data_param

        # Caller overrides XML; XML overrides defaults
        if model_type is None:
            model_type = arch["modelType"]
        # Resolve embedding_path and store back in config for InputSpace
        embedding_path = TheXMLConfig.get("architecture.embeddingPath", None) or None
        if embedding_path is not None:
            embedding_path = self._resolve_artifact_path(embedding_path)
            TheXMLConfig._data["architecture"]["embeddingPath"] = embedding_path

        # Positional/temporal encoding: architecture-level numeric values
        _s = TheXMLConfig.space
        _nWhere = TheXMLConfig.get("architecture.nWhere")
        _nWhen  = TheXMLConfig.get("architecture.nWhen")
        _objectSize = _nWhere + _nWhen
        TheXMLConfig._data.setdefault("architecture", {})["objectSize"] = _objectSize

        # Codebook sizes (from model.xml defaults, overridden by model-specific XML)
        _nObjects = (_s("InputSpace", "nVectors") + _s("PerceptualSpace", "nVectors") +
                     _s("ConceptualSpace", "nVectors") + _s("SymbolicSpace", "nVectors") +
                     _s("SyntacticSpace", "nVectors") + _s("OutputSpace", "nVectors"))
        TheXMLConfig._data.setdefault("architecture", {})["nObjects"] = _nObjects

        self.create(
            nInput=_s("InputSpace", "nActive"),
            nPercepts=_s("PerceptualSpace", "nActive"),
            nConcepts=_s("ConceptualSpace", "nActive"),
            nSymbols=_s("SymbolicSpace", "nActive"),
            nWords=_s("SyntacticSpace", "nActive"),
            nOutput=_s("OutputSpace", "nActive"),
            conceptualOrder=arch["conceptualOrder"],
            symbolicOrder=arch["symbolicOrder"],
            model_type=model_type, data=data, embedding_path=embedding_path,
            reverse_scale=_t("reverseScale"),
            what_scale=_t("whatScale"),
            where_scale=_t("whereScale"),
            when_scale=_t("whenScale"),
            masked_prediction=arch["maskedPrediction"].upper(),
            reconstruct=arch["reconstruct"],
        )
        # trainEmbedding:
        #   NONE  = frozen embeddings, frozen model
        #   CBOW  = embedding CBOW updates only (predict from padded context)
        #   SBOW  = embedding SBOW updates only (predict from leave-one-out centroid, faster)
        #   ARLM  = network layers only, embeddings frozen
        #   BOTH  = SBOW embedding updates + network layers (two optimizers)
        #   JOINT = single loss: model_loss + embeddingScale * sbow_loss, one optimizer
        if "trainEmbedding" in arch and not isinstance(arch["trainEmbedding"], dict):
            te = arch["trainEmbedding"]
        elif "trainEmbeddings" in arch and not isinstance(arch["trainEmbeddings"], dict):
            te = arch["trainEmbeddings"]
        else:
            te = _t("trainEmbedding")
        if te is True:
            te = "BOTH"
        elif te is False or te is None:
            te = "NONE"
        self.train_embedding = te.upper()
        # Embeddings participate in the optimizer unless frozen
        # Valid trainEmbedding values: NONE, CBOW, SBOW, BACKPROP, BOTH, JOINT
        # (ARLM/ARUS/RARLM are maskedPrediction modes, not trainEmbedding modes)
        self.optimize_embedding = self.train_embedding not in ("NONE", "CBOW", "SBOW")
        if self.optimize_embedding and isinstance(self.inputSpace.vectors(), Embedding):
            emb_params = self.inputSpace.vectors().embedding_parameters()
            self.inputSpace.params = self.inputSpace.params + emb_params
        # embeddingScale: weight of embedding loss in JOINT mode
        self.loss.embedding_scale = float(_t("embeddingScale") or 0.1)
        # Propagate flag to Embedding so forward()/reverse() can detach
        if isinstance(self.inputSpace.vectors(), Embedding):
            self.inputSpace.vectors().optimize_embedding = self.optimize_embedding
            object.__setattr__(self.inputSpace.vectors(), '_model', self)
        # Auto-load weights if configured
        if _t("autoload"):
            wpath = TheXMLConfig.get("architecture.weightsPath")
            wpath = self._resolve_artifact_path(wpath)
            self.load_weights(wpath)
        # Inference config
        self.max_response_length = arch["maxResponseLength"]
        return cfg

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               conceptualOrder=1, symbolicOrder=1,
               model_type="simple", data=None, embedding_path=None,
               reverse_scale=0.5, what_scale=0.7, where_scale=0.2, when_scale=0.1,
               masked_prediction='NONE', reconstruct='NONE'):
        """Build the full space hierarchy from architecture parameters.

        Config-derivable flags (reshape, ergodic, quantized, etc.) are read
        from TheXMLConfig by each Space constructor.  Only runtime/pipeline
        params are passed here.

        Args:
            nInput/nPercepts/nConcepts/nSymbols/nOutput: object counts per space.
            nWords: object count for the SyntacticSpace (used when symbolicOrder >= 1).
            conceptualOrder: number of extra Percept->Concept->Symbol cycles.
            symbolicOrder: number of extra Syntax->Symbol cycles.
            model_type: "simple", "embedding", "passthrough", or "vq".
        """
        self.spaces = []  # reset — prevent stale accumulation from prior create() calls
        TheXMLConfig._requirements.clear()  # clear stale requirements from prior create()/tests
        # Read config-derivable flags
        self.reconstruct     = reconstruct.lower()
        self.reversible      = str(TheXMLConfig.get("architecture.reconstruct")).upper() != "NONE"
        self.reshape          = TheXMLConfig.get("architecture.reshape")
        self.ergodic          = TheXMLConfig.get("architecture.ergodic")
        self.processSymbols   = TheXMLConfig.get("architecture.processSymbols")
        self.certainty        = TheXMLConfig.get("architecture.certainty")
        self.lexer            = TheXMLConfig.space("InputSpace", "lexer")
        self.quantized        = TheXMLConfig.space("InputSpace", "quantized")
        self.perceptQuantized = TheXMLConfig.space("PerceptualSpace", "quantized")
        self.conceptQuantized = TheXMLConfig.space("ConceptualSpace", "quantized")
        self.perceptPassThrough = TheXMLConfig.space("PerceptualSpace", "passThrough")
        self.symbolPassThrough  = TheXMLConfig.space("SymbolicSpace", "passThrough")
        self.invertible       = TheXMLConfig.space("PerceptualSpace", "invertible")
        self.hasNorm          = TheXMLConfig.space("ConceptualSpace", "hasNorm")
        self.perceptHasAttention = TheXMLConfig.space("PerceptualSpace", "hasAttention")
        self.conceptHasAttention = TheXMLConfig.space("ConceptualSpace", "hasAttention")
        self.perceptPrototypes  = TheXMLConfig.space("PerceptualSpace", "nVectors")
        self.conceptPrototypes  = TheXMLConfig.space("ConceptualSpace", "nVectors")
        self.min_frequency    = float(TheXMLConfig.data_param("minFrequency", 0.0))
        self.neg_samples      = int(TheXMLConfig.training("negSamples", 64))
        # Runtime params
        self.nInput           = nInput
        self.nOutput          = nOutput
        self.nPercepts        = nPercepts
        self.nConcepts        = nConcepts
        self.nSymbols         = nSymbols
        TheXMLConfig.require(
            lambda cfg, _ns=nSymbols, _no=nOutput: _ns >= _no,
            f"nSymbols ({nSymbols}) must be >= nOutput ({nOutput}): "
            f"the symbolic bottleneck must have at least as many symbols as outputs"
        )
        self.nOutputSymbols   = nOutput
        self.nReconSymbols    = max(0, nSymbols - nOutput)
        self.recon_symbols    = None
        self.nWords           = nWords
        self.data             = data
        self.model_type       = model_type
        self.embedding_path   = embedding_path
        self.conceptualOrder  = conceptualOrder
        self.symbolicOrder    = symbolicOrder
        self.loss = ModelLoss(reverse_scale=reverse_scale,
                         what_scale=what_scale,
                         where_scale=where_scale,
                         when_scale=when_scale,
                         certainty=self.certainty,
                         nOutput=nOutput,
                         conceptualOrder=conceptualOrder,
                         symbolicOrder=symbolicOrder,
                         nWhere=TheXMLConfig.get("architecture.nWhere"),
                         nWhen=TheXMLConfig.get("architecture.nWhen"))
        self.masked_prediction = masked_prediction
        if data is not None and hasattr(data, 'masked_prediction') and data.masked_prediction != 'NONE':
            data.masked_prediction = masked_prediction
        # nOutputSymbols tracks total symbol count fed to OutputSpace.
        # Starts with only the output-destined symbols (not reconstruction symbols).
        # It grows as higher-order cycles (conceptualOrder, symbolicOrder) append symbols.
        nOutputSymbols = self.nOutputSymbols
        self.inputSpace      = InputSpace(self.nInput,
                                           model_type=model_type)
        # Convert masked-word string labels to embedding vectors now that
        # the Embedding vocabulary is available.
        if data is not None and hasattr(data, '_lm_labels') and data._lm_labels is not None:
            embedding = self.inputSpace.vectorSet[0] if self.inputSpace.vectorSet else None
            if embedding is not None and hasattr(embedding, 'pretrain'):
                data.prepare_lm_targets(embedding)
                # Move new targets to device
                data.toDevice()
        self.perceptualSpace = PerceptualSpace(self.inputSpace.outputShape[0])
        self.conceptualSpace = ConceptualSpace(self.perceptualSpace.outputShape[0])
        self.symbolicSpace   = SymbolicSpace(self.conceptualSpace.outputShape[0],
                                              conceptualSpace=self.conceptualSpace)
        self.spaces.extend([self.inputSpace, self.perceptualSpace, self.conceptualSpace, self.symbolicSpace])

        if self.conceptualOrder == 2:
            self.perceptualSpace2 = PerceptualSpace(self.conceptualSpace.outputShape[0],
                                                    inputDim=TheXMLConfig.space("SymbolicSpace", "nDim"))
            self.conceptualSpace2 = ConceptualSpace(self.perceptualSpace2.outputShape[0])
            self.symbolicSpace2   = SymbolicSpace(self.conceptualSpace2.outputShape[0],
                                                  conceptualSpace=self.conceptualSpace2)
            nOutputSymbols += (self.conceptualOrder - 1) * self.nSymbols
            self.spaces.extend([self.perceptualSpace2, self.conceptualSpace2, self.symbolicSpace2])

        if self.symbolicOrder == 2:
            # SyntacticSpace3 receives the full symbol tensor (nSymbols objects)
            self.syntacticSpace3 = SyntacticSpace(self.nSymbols, self.nSymbols)
            self.symbolicSpace3  = SymbolicSpace(self.syntacticSpace3.outputShape[0])
            nOutputSymbols += (self.symbolicOrder - 1) * self.nSymbols
            self.spaces.extend([self.syntacticSpace3, self.symbolicSpace3])

        self.nTotalOutputSymbols = nOutputSymbols
        self.outputSpace     = OutputSpace(nOutputSymbols,
                                           nActiveOutput=nOutput,
                                           masked_prediction=(masked_prediction != 'NONE'),
                                           vectors=self.inputSpace.vectors())
        self.spaces.extend([self.outputSpace])
        self.inputSpace.outputSpace = self.outputSpace

        # The output dimensionality of the input layer must be equal to the output dimensionality of the perceptual layer, since the conceptual layer operates on both.
        #assert self.inputSpace.outputShape[1] == self.perceptualSpace2.outputShape[1] # inputDim == perceptDim
        # The input dimensionality of the symbolic layer must be equal to the input dimensionality of the perceptual layer, since they both operate on the output of the conceptual layer.
        #assert self.symbolicSpace.inputShape[1] == self.perceptualSpace2.inputShape[1] == self.conceptualSpace.outputShape[1]#  conceptDim = conceptDim
        # The output shape of the symbolic space is equal to the input shape of the output space
        #assert self.symbolicSpace.outputShape[1] == self.outputSpace.inputShape[1] # these are in conceptual space, or symbolic space if symbols emit objectSize symbols (processSymbols == True)

        self.to(TheDevice)
        TheXMLConfig.validate()

    def Start(self, inputData):
        """Forward pass through the core pipeline: Input -> Percept -> Concept -> Symbol."""
        input = self.inputSpace(inputData)
        percepts = self.perceptualSpace(input)
        concepts = self.conceptualSpace(percepts)
        symbols = self.symbolicSpace(concepts)
        if self.plot:
            TheReport.plotActivations(figure=1, concepts=concepts)
        return input, concepts, symbols
    def StartReverse(self, symbols):
        """Reverse pass: Symbol -> Concept -> Percept -> Input (reconstruction)."""
        concepts = self.symbolicSpace.reverse(symbols)
        percepts = self.conceptualSpace.reverse(concepts)
        input = self.perceptualSpace.reverse(percepts)
        inputData  = self.inputSpace.reverse(input)
        return inputData, input
    def SubsymbolicThought(self, data):
        """Extra Percept->Concept->Symbol cycle (conceptualOrder >= 1)."""
        percepts = self.perceptualSpace2(data)
        concepts = self.conceptualSpace2(percepts)
        symbols  = self.symbolicSpace2(concepts)
        if self.plot:
            TheReport.plotActivations(figure=1, percepts=percepts, concepts=concepts)
        return concepts, symbols
    def SubsymbolicThoughtReverse(self, concepts, symbols):
        """Reverse of SubsymbolicThought."""
        concepts = self.symbolicSpace2.reverse(symbols)
        percepts = self.conceptualSpace2.reverse(concepts)
        return percepts
    def SymbolicThought(self, data):
        """Extra Syntax->Symbol cycle (symbolicOrder >= 1)."""
        words   = self.syntacticSpace3(data)
        symbols = self.symbolicSpace3(words)
        if self.plot:
            TheReport.plotActivations(figure=1, symbols=symbols)
        return symbols, words
    def SymbolicThoughtReverse(self, symbols, words):
        """Reverse of SymbolicThought."""
        symbols = self.syntacticSpace3.reverse(words)
        data    = self.symbolicSpace3.reverse(symbols)
        return data
    def Finish(self, symbols):
        """Project concatenated symbols to task output via OutputSpace."""
        outputData = self.outputSpace(symbols)
        if self.plot:
            TheReport.plotActivations(figure=1, symbols=symbols)
        return outputData
    def FinishReverse(self, outputData):
        """Reconstruct the symbol tensor from output for the reverse pass.

        reconstruct="symbols" (default): use cached forward symbols only.
        reconstruct="output": use outputSpace.reverse(outputData) only.
        reconstruct="both": reversed output + cached recon_symbols.
        """
        mode = getattr(self, 'reconstruct', 'symbols')
        if mode == 'output':
            return self.outputSpace.reverse(outputData)
        elif mode == 'both':
            output_symbols = self.outputSpace.reverse(outputData)
        else:  # 'symbols'
            output_symbols = self.output_symbols
        if self.recon_symbols is not None and self.nReconSymbols > 0:
            return torch.cat([output_symbols, self.recon_symbols], dim=1)
        return output_symbols

    def infer(self, text, max_length=None, mode=None):
        """Autoregressive inference via the standard batch pipeline.

        Two modes:

        ``ARLM`` (append-and-rerun): stages seed text, runs forward,
        decodes the output token, appends it to the input via
        ``pushInput()``, and repeats.  Each iteration re-lexes and
        re-embeds the full (growing) input.

        ``ARIR`` (autoregressive input reconstruction, default): TODO —
        reconstructs a degraded input in-place, reusing the lexing and
        codebook lookup from the initial forward pass.  See design plan
        in ``docs/plans/``.

        Stops when: EOF is predicted, ``max_length`` characters have
        been produced, or the InputSpace output buffer is full.

        Args:
            text: input string (seed text)
            max_length: max characters to generate
            mode: 'ARLM' for traditional append-and-rerun,
                  'ARIR' for input reconstruction (default).
                  Also accepts traditional=True/False for backwards compat
                  via keyword: ``infer(text, traditional=True)`` is
                  equivalent to ``infer(text, mode='ARLM')``.

        Returns:
            list of predicted tokens (words or characters)
        """
        if mode is None:
            mode = getattr(self, 'masked_prediction', 'ARIR')
        mode = mode.upper()

        if mode == 'ARLM':
            return self._infer_traditional(text, max_length)
        elif mode == 'ARIR':
            return self._infer_arir(text, max_length)
        else:
            raise ValueError(f"infer: unknown mode '{mode}'. Use 'ARLM' or 'ARIR'.")

    def _infer_traditional(self, text, max_length=None):
        """Traditional append-and-rerun autoregressive inference.

        Each step: forward pass → decode output → pushInput(token) → repeat.
        Re-lexes and re-embeds the full input every iteration.
        """
        if max_length is None:
            max_length = getattr(self, 'max_response_length', 256)

        self.eval()
        self.set_sigma(0)
        nOutput = self.inputSpace.outputShape[0]
        tokens = []
        total_chars = 0

        with torch.no_grad(), TheData.runtime_batch([text]):
            while True:
                result, _ = self.runBatch(
                    train=False, batchNum=0, batchSize=1, split="runtime",
                )
                if result is None:
                    break

                # Decode the output prediction to a token
                decoded = self.inputSpace.predict(result.outputPred)
                word = decoded[0]

                # EOF check (consistent with ARIR path)
                if word is None or word == '' or word == '\x00':
                    break

                tokens.append(word)
                total_chars += len(word)

                # Stop if max characters produced
                if total_chars >= max_length:
                    break

                # Stop if output buffer is full
                if len(tokens) >= nOutput:
                    break

                TheData.pushInput(word)

        return tokens

    def _infer_arir(self, text, max_length=None):
        """ARIR: autoregressive input reconstruction inference.

        Pushes data onto TheData and calls runEpoch().  All ARIR logic
        (embedding, [MASK] placement, reconstruction copy, cursor advance)
        lives in InputSpace._getBatch_arir() as a state machine.

        Falls back to ARLM if the model is not reversible.
        """
        if not self.reversible:
            import warnings
            warnings.warn(
                "ARIR requires reversible=True; falling back to ARLM.",
                RuntimeWarning, stacklevel=2,
            )
            return self._infer_traditional(text, max_length)

        max_length = max_length or getattr(self, 'max_response_length', 256)
        self.eval()
        self.set_sigma(0)

        with torch.no_grad(), TheData.runtime_batch([text], [[0]], mode='ARIR'):
            self.inputSpace._arir_reset()
            self.inputSpace._arir_max_chars = max_length
            self.runEpoch(batchSize=1, split="runtime")

        return self.inputSpace.get_predicted_tokens()

    def forward(self, inputData):
        """Full forward pass: core pipeline + higher-order cycles + output projection.

        Returns (output_prediction, perceptual_state).
        Symbols from each processing stage are concatenated before OutputSpace.
        """
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice)
        input, concepts, symbols = self.Start(inputData)
        # Higher-order subsymbolic cycles (conceptualOrder extra passes)
        for n in range(1,self.conceptualOrder):
            NA, symbols1 = self.SubsymbolicThought(concepts)
            symbols = torch.cat((symbols, symbols1), dim=1)
        # Higher-order symbolic cycles (symbolicOrder extra passes)
        for n in range(1,self.symbolicOrder):
            NA, symbols2 = self.SymbolicThought(symbols)
            symbols = torch.cat((symbols, symbols2), dim=1)
        # Split AFTER higher-order cycles: output symbols for prediction,
        # recon symbols for reconstruction
        if self.nReconSymbols > 0:
            self.output_symbols = symbols[:, :self.nTotalOutputSymbols, :]
            self.recon_symbols = symbols[:, self.nTotalOutputSymbols:, :]
        else:
            self.output_symbols = symbols
            self.recon_symbols = None
        outputData = self.Finish(self.output_symbols)
        batch = input.shape[0]
        self.inputSpace.subspace.whenEncoding.increment(batch)
        return input, symbols, outputData
    def reverse(self, symbols, outputData):
        """Full reverse pass: unwind higher-order cycles then core reconstruction.

        Slices the concatenated symbol tensor to route each chunk to its
        corresponding reverse stage, in reverse order of the forward pass.
        """
        symbols = self.FinishReverse(outputData)
        nSym = round(self.nSymbols)
        symbolIndex = 0
        for n in range(1, self.symbolicOrder):
            symbols1 = symbols[:, symbolIndex*nSym:(symbolIndex+1)*nSym]
            symbolIndex += 1
            symbols = self.SymbolicThoughtReverse(symbols, symbols1)
        for n in range(1, self.conceptualOrder):
            symbols1 = symbols[:, symbolIndex*nSym:(symbolIndex+1)*nSym]
            symbolIndex += 1
            symbols = self.SubsymbolicThoughtReverse(symbols, symbols1)
        # Final chunk goes to the core reverse pipeline
        symbols = symbols[:, symbolIndex * nSym:(symbolIndex + 1) * nSym]
        inputData, input = self.StartReverse(symbols)
        return inputData, input

    def runTrial(self, numEpochs=1, batchSize=10, lr=0.01):
        """Main training loop: train for numEpochs, evaluate on test set each epoch.

        Alpha (exploration temperature) anneals from 1.0 (full exploration)
        to 0.0 (full exploitation) over the first 5% of training.  This is
        propagated to all Spaces and their layers/VectorSets via set_sigma().

        A single persistent optimizer is used across all epochs so Adam's
        momentum and variance estimates accumulate properly.

        Returns a list of per-epoch test accuracies.
        """
        trainLosses       = [[],[]]  # [output_losses, reconstruction_losses]
        validationLosses  = [[],[]]
        testLosses        = [[],[]]
        self.plot         = False
        accuracy          = []
        self._optimizer   = self.getOptimizer(lr=lr)

        # Enable sigma-driven self-annealing for ergodic layers
        self.set_sigma(1.0)

        # Baseline evaluation before any training
        self.set_sigma(0)
        outErr, inErr, allOut, lastIn = self.runEpoch(batchSize=batchSize, split="test")
        self.set_sigma(1.0)
        testLosses[0].append(outErr)
        testLosses[1].append(inErr)
        print(f"Baseline Test Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")

        for epoch in range(numEpochs):
            print(f"Epoch [{epoch + 1}/{numEpochs}]")

            outErr, inErr, allOut, lastIn = self.runEpoch(optimizer=self._optimizer, batchSize=batchSize, split="train")
            trainLosses[0].append(outErr)
            trainLosses[1].append(inErr)
            print(f"Train Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")

            self.set_sigma(0)  # suppress exploration during eval
            outErr, inErr, allOut, lastIn = self.runEpoch(batchSize=batchSize, split="test")
            self.set_sigma(1.0)  # re-enable for next training epoch
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)

            if hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE':
                # Masked prediction: report loss only (no classification accuracy)
                accuracy += [0.0]
                print(f"Test Loss: output={outErr:.4f}, reconstruction={inErr:.4f}")
            elif allOut.dim() == 1:
                predicted = (allOut > 0.5).long()
                actual = (self.outputSpace.getTestOutput().squeeze() > 0.5).long()
                total   = predicted.size(0)
                correct = (predicted == actual).sum().item()
                accuracy += [correct / total]
                print(f"Test Accuracy: {100 * correct / total:.2f}%")
            else:
                _, predicted = torch.max(allOut, 1)
                _, actual = torch.max(self.outputSpace.getTestOutput(), 1)
                total   = predicted.size(0)
                correct = (predicted == actual).sum().item()
                accuracy += [correct / total]
                print(f"Test Accuracy: {100 * correct / total:.2f}%")

            self.inputSpace.shuffle()

        print(f"Final Stats:")
        TheReport.plotLoss(self.name, trainLosses, validationLosses, testLosses)
        self.rCorrect = self.mnistReport()

        # Reconstruction report: run final test pass and show input vs reconstructed
        if self.reversible and self.inputSpace.model_type == "embedding":
            self._reconstructionReport()

        self.trainLosses = trainLosses
        self.testLosses  = testLosses
        return accuracy
    
    BatchResult = namedtuple('BatchResult', [
        'outputPred', 'symbols', 'lossOut', 'lossIn', 'inputPred', 'forwardInput',
    ])

    def runBatch(self, train=True, batchNum=0, batchSize=10, split="train",
                 optimizer=None):
        """Run a single batch: forward pass, loss, and (if training) backward + step.

        Args:
            train: whether to compute gradients and update parameters.
            batchNum: opaque cursor returned by getBatch for the next batch.
            batchSize: number of examples per batch.
            split: "train", "test", or "validation".
            optimizer: pre-built optimizer (required when train=True).

        Returns:
            (BatchResult, nextBatchNum) on success, or (None, batchNum) when
            the dataset is exhausted.
        """
        sentenceIdx = batchNum  # sentence index before getBatch increments
        batch, batchNum = self.inputSpace.getBatch(batchNum, batchSize, split)
        if batch is None:
            return None, batchNum

        inputTensor, outputTensor = batch
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'
        inference_only = not train and split == "runtime"
        arir_mode = (split == "runtime"
                     and getattr(self.inputSpace.data, '_runtime_mode', None) == 'ARIR')

        if train:
            optimizer.zero_grad()

        # Forward pass (masking, if any, is applied inside InputSpace.forward())
        forwardInput, symbols, outputDataPred = self.forward(inputTensor)

        if arir_mode:
            # ARIR inference: no output loss, but always run reverse pass
            # so that reconstructed vectors and _recovered_words are available
            # for the next getBatch() call.
            inputPred = None
            if self.reversible:
                _, inputPred = self.reverse(symbols, outputDataPred)
            return self.BatchResult(
                outputPred=outputDataPred, symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=inputPred, forwardInput=forwardInput,
            ), batchNum

        if inference_only:
            # Inference path: forward only, no loss, no reverse.
            return self.BatchResult(
                outputPred=outputDataPred, symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=None, forwardInput=forwardInput,
            ), batchNum

        if outputTensor is None:
            raise RuntimeError(
                f"runBatch: missing output targets for split='{split}'. "
                "For inference use split='runtime', or stage runtime_batch(..., outputs=...) "
                "if targets are required."
            )

        outputPred = outputDataPred.squeeze()
        output     = outputTensor.squeeze()
        lossOut    = self.loss.output(outputPred, output)

        # ARUS: suppress output loss (unsupervised — no target signal)
        if hasattr(self, 'masked_prediction') and self.masked_prediction == 'ARUS':
            lossOut = torch.tensor(0.0, device=TheDevice)

        use_recon = self.reversible and self.loss.reverse_scale > 0
        if use_recon:
            inputDataPred, inputPred = self.reverse(symbols, outputDataPred)
            pred_sq = inputDataPred
            masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'

            # Use pre-masked, post-encoding target when available
            recon_target, recon_mask = self.inputSpace.get_reconstruction_target()
            if recon_target is not None:
                target_sq = recon_target.squeeze()
            else:
                target_sq = forwardInput.squeeze()

            if masked_pred and recon_mask is not None and pred_sq.dim() >= 2:
                # Masked prediction: compute loss only at masked positions
                mask = recon_mask
                if pred_sq.dim() == 3:
                    mask = mask.unsqueeze(-1).expand_as(pred_sq)
                lossIn = self.loss.compute(pred_sq[mask], target_sq[mask])
            else:
                lossIn = self.loss.compute(pred_sq, target_sq)
        else:
            inputDataPred = None
            lossIn = None

        # JOINT mode: compute SBOW embedding loss
        sbow = None
        if train:
            te = getattr(self, 'train_embedding', 'NONE')
            if te == 'JOINT':
                emb = self.inputSpace.vectors()
                if isinstance(emb, Embedding):
                    sentences = self._get_sentences(split)
                    if sentences and sentenceIdx < len(sentences):
                        sentence = sentences[sentenceIdx]
                        from parse import quick_parser
                        words = [t for t, _ in quick_parser(sentence)]
                        sbow = self.inputSpace.sbow_loss(words)

        totalLoss = self.loss.total(lossOut, lossIn, sbow)

        if train:
            totalLoss.backward()
            if self.ergodic:
                self.paramUpdate()
            optimizer.step()

        result = self.BatchResult(
            outputPred=outputDataPred,
            symbols=symbols,
            lossOut=lossOut,
            lossIn=lossIn,
            inputPred=inputDataPred,
            forwardInput=forwardInput,
        )
        return result, batchNum

    def runEpoch(self, optimizer=None, batchSize=10, split="train"):
        """Run one epoch over the dataset (training if optimizer given, eval if None).

        Uses getBatch() stream interface for flexible batch iteration.
        Delegates per-batch work to ``runBatch()``.

        In inference mode (split="runtime", no optimizer): skips loss
        construction, output accumulation, progress printing, and CBOW
        updates.  Returns immediately after the getBatch/runBatch loop.

        Args:
            optimizer: pre-built Adam optimizer (persistent across epochs).
                       Pass None for evaluation mode.
            batchSize: number of examples per batch (standard mode only)
            split: "train", "test", or "validation"

        Returns (output_loss, reconstruction_loss, all_predictions, last_reconstruction).
        For inference mode, returns (0, 0, [], []).
        """
        training = optimizer is not None
        inference = split == "runtime" and not training
        self.train(training)
        self.outputSpace.clearBatchResults()
        ctx = torch.no_grad() if not training else nullcontext()

        # Inference fast path: skip loss construction and accumulation
        if inference:
            with ctx:
                batchNum = 0
                while True:
                    result, batchNum = self.runBatch(
                        train=False, batchNum=batchNum, batchSize=batchSize,
                        split=split,
                    )
                    if result is None:
                        break
                    self.outputSpace.putBatch(result)
            return 0, 0, [], []

        # Training / evaluation path
        allOutput = []
        allInput = []
        outErr = 0
        inErr = 0
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'

        with ctx:
            batchNum = 0
            batchIdx = 0
            while True:
                result, batchNum = self.runBatch(
                    train=training, batchNum=batchNum, batchSize=batchSize,
                    split=split, optimizer=optimizer,
                )
                if result is None:
                    break

                self.outputSpace.putBatch(result)

                if training and batchIdx % 100 == 0:
                    print(f"  batch {batchIdx}", end="\r", flush=True)

                # Embedding training (post-batch, needs batchIdx for sentence lookup)
                if training:
                    te = getattr(self, 'train_embedding', 'NONE')
                    if masked_pred and te in ('CBOW', 'SBOW', 'BOTH'):
                        emb = self.inputSpace.vectors()
                        if isinstance(emb, Embedding):
                            sentences = self._get_sentences(split)
                            if sentences and batchIdx < len(sentences):
                                sentence = sentences[batchIdx]
                                from parse import quick_parser
                                words = [t for t, _ in quick_parser(sentence)]
                                # CBOW uses padded context; SBOW and BOTH use the faster centroid method
                                method = 'CBOW' if te == 'CBOW' else 'SBOW'
                                self.inputSpace.train_embeddings(words, method=method)

                outErr = result.lossOut.item()
                inErr = result.lossIn.item() if result.lossIn is not None else 0

                outputDataPred = result.outputPred.clone().detach().squeeze()
                if batchIdx == 0:
                    allOutput = outputDataPred
                else:
                    allOutput = torch.concat((allOutput, outputDataPred), dim=0)

                if self.reversible and result.inputPred is not None:
                    allInput = result.inputPred.clone().detach().squeeze()

                batchIdx += 1

        return outErr, inErr, allOutput, allInput

    def classificationReport(self, min=0, max=1):
        test_input, test_output = self.inputSpace.getTestData()
        _, _, y_pred, x_pred = self.runTest(test_input, test_output)
        y_actual = self.outputSpace.getTestOutput()
        y_pred_sat = np.maximum(min, np.minimum(max, np.round(np.array(y_pred)).squeeze()))
        performance = classification_report(
            y_actual, y_pred_sat,
            target_names=["Negative Review", "Positive Review"]
        )
        print(performance)
class MentalModel(BaseModel):
    name = "MentalModel"

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nOutput,
               reversible=True, quantized=False,
               perceptQuantized=None, conceptQuantized=None,
               model_type="simple", data=None, embedding_path=None,
               lexer="word", reverse_scale=0.5,
               what_scale=0.7, where_scale=0.2, when_scale=0.1,
               reshape=False, naive=False,
               perceptHasAttention=True, conceptHasAttention=False,
               masked_prediction='NONE', min_frequency=0.0,
               neg_samples=64):

        self.spaces = []
        self.reversible = reversible
        self.nInput = nInput
        self.nPercepts = nPercepts
        self.nConcepts = nConcepts
        self.nSymbols = nSymbols
        self.nOutput = nOutput
        self.data = data
        self.model_type = model_type
        self.embedding_path = embedding_path
        self.lexer = lexer
        self.reshape = reshape
        self.quantized = quantized
        self.perceptQuantized = perceptQuantized if perceptQuantized is not None else quantized
        self.conceptQuantized = conceptQuantized if conceptQuantized is not None else quantized
        self.masked_prediction = masked_prediction

        self.loss = ModelLoss(
            reverse_scale=reverse_scale,
            what_scale=what_scale,
            where_scale=where_scale,
            when_scale=when_scale,
            nOutput=nOutput,
            conceptualOrder=1,
            symbolicOrder=1,
        )

        self.inputSpace = InputSpace(
            nInput,
            model_type=model_type,
        )

        # Branch 1: Input -> Percepts
        self.perceptualSpace = PerceptualSpace(
            self.inputSpace.outputShape[0],
        )

        # Branch 2: Input -> Concepts
        # Assumes inputDim == perceptDim with current ConceptualSpace contract.
        self.conceptualSpace = ConceptualSpace(
            self.inputSpace.outputShape[0],
        )

        # Join: [Percepts, Concepts] -> Symbols
        # Assumes perceptDim == conceptDim so concat along object axis is valid.
        self.symbolicSpace = SymbolicSpace(
            nPercepts + nConcepts,
            conceptualSpace=self.conceptualSpace,
        )

        self.outputSpace = OutputSpace(
            nSymbols,
            masked_prediction=(masked_prediction != 'NONE'),
            vectors=self.inputSpace.vectors(),
        )

        self.spaces.extend([
            self.inputSpace,
            self.perceptualSpace,
            self.conceptualSpace,
            self.symbolicSpace,
            self.outputSpace,
        ])

        self.to(TheDevice)

    def Start(self, inputData):
        input_state = self.inputSpace(inputData)
        percepts = self.perceptualSpace(input_state)
        concepts = self.conceptualSpace(input_state)
        merged = torch.cat([percepts, concepts], dim=1)
        symbols = self.symbolicSpace(merged)
        return input_state, percepts, concepts, symbols

    def Finish(self, symbols):
        return self.outputSpace(symbols)

    def forward(self, inputData):
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice)
        input_state, percepts, concepts, symbols = self.Start(inputData)
        outputData = self.Finish(symbols)
        return input_state, percepts, concepts, symbols, outputData

    def reverse(self, symbols, outputData):
        symbols = self.outputSpace.reverse(outputData)
        merged = self.symbolicSpace.reverse(symbols)

        percepts = merged[:, :self.nPercepts, :]
        concepts = merged[:, self.nPercepts:self.nPercepts + self.nConcepts, :]

        input_from_percepts = self.perceptualSpace.reverse(percepts)
        input_from_concepts = self.conceptualSpace.reverse(concepts)

        # Hypothetical merge rule for the two reconstructed input streams.
        input_latent = 0.5 * (input_from_percepts + input_from_concepts)
        input_data = self.inputSpace.reverse(input_latent)

        return input_data, input_latent
TheBasicModel = BasicModel()

class BasicModelFactory:
    """Create, train, and evaluate models from an XML config file.

    Dispatches to the right model class based on <architecture> flags:
      - modelType=embedding   → BasicModel (embedding/language model path)
      - modelType=passthrough → BasicModel (passthrough path)
      - modelType=vq         → BasicModel (vector-quantized path)
      - Otherwise             → SimpleModel parameterized by:
            ergodic, certainty, quantized, normed, reverse, invert
    """

    @staticmethod
    def model_name(ergodic, certainty, quantized, normed=False, reverse=False, invert=False):
        """Generate a human-readable model name from its flags."""
        if not ergodic and not certainty and not quantized:
            return "SimpleModel"
        parts = []
        if ergodic:
            parts.append("Ergodic")
        if certainty:
            parts.append("Certainty")
        if quantized:
            parts.append("Quantized")
        if normed:
            parts.append("Normed")
        if invert:
            parts.append("Invertible")
        elif reverse:
            parts.append("Reversible")
        return " + ".join(parts) if parts else "SimpleModel"

    @staticmethod
    def get_space_param(cfg, space_name, key):
        """Look up key in space section, fall back to architecture section.

        Resolution order: cfg[space_name][key] -> cfg["architecture"][key]
        All parameters must be in model.xml; raises KeyError if missing.
        """
        space = cfg.get(space_name, {})
        if key in space:
            return space[key]
        arch = cfg.get("architecture", {})
        if key in arch:
            return arch[key]
        raise KeyError(f"Required parameter '{key}' not found in <{space_name}> or <architecture>")

    @staticmethod
    def validate_config(cfg):
        """Check merged config for known inconsistencies and raise on error.

        Called after defaults have been merged so all keys are present.
        Uses get_space_param() to read from space-scoped sections.
        """
        gsp = BasicModelFactory.get_space_param
        arch = cfg.get("architecture", {})
        errors = []

        reshape = arch.get("reshape", False)

        # Attention is incompatible with reshape (attention expects 3D, reshape flattens to 2D)
        if reshape and gsp(cfg, "PerceptualSpace", "hasAttention"):
            errors.append(
                "PerceptualSpace hasAttention=True is incompatible with reshape=True. "
                "Set <hasAttention>false</hasAttention> in <PerceptualSpace>.")
        if reshape and gsp(cfg, "ConceptualSpace", "hasAttention"):
            errors.append(
                "ConceptualSpace hasAttention=True is incompatible with reshape=True. "
                "Set <hasAttention>false</hasAttention> in <ConceptualSpace>.")

        # SymbolicSpace passThrough requires symbolDim == conceptDim
        sym_pt = gsp(cfg, "SymbolicSpace", "passThrough")
        if sym_pt:
            symDim = gsp(cfg, "SymbolicSpace", "nDim")
            conDim = gsp(cfg, "ConceptualSpace", "nDim")
            if symDim != 0 and symDim != conDim:
                errors.append(
                    f"SymbolicSpace passThrough=True requires symbolDim == conceptDim "
                    f"(got symbolDim={symDim}, conceptDim={conDim}). "
                    f"Set <nDim>{conDim}</nDim> in <SymbolicSpace> or use <nDim>0</nDim>.")

        # When invertible=True, the InvertiblePiLayer doubles the sequence,
        # so ConceptualSpace.nVectors must be 2 * PerceptualSpace.nVectors.
        reversible = arch.get("reconstruct", "NONE").upper() != "NONE"
        percept_inv = gsp(cfg, "PerceptualSpace", "invertible")
        percept_pt = gsp(cfg, "PerceptualSpace", "passThrough")
        if percept_inv:
            p_nvec = gsp(cfg, "PerceptualSpace", "nVectors")
            c_nvec = gsp(cfg, "ConceptualSpace", "nVectors")
            if c_nvec != 2 * p_nvec:
                errors.append(
                    f"PerceptualSpace invertible=True doubles sequence length, "
                    f"so ConceptualSpace.nVectors must be 2*PerceptualSpace.nVectors "
                    f"(got PerceptualSpace.nVectors={p_nvec}, ConceptualSpace.nVectors={c_nvec}). "
                    f"Set <nVectors>{2*p_nvec}</nVectors> in <ConceptualSpace>.")

        # Warn about reversible + not invertible: uses pinv which may be numerically unstable
        if reversible and not percept_inv and not percept_pt:
            warnings.warn(
                "PerceptualSpace: reversible=True with invertible=False uses two "
                "InvertiblePiLayers with separate weights. The reverse path involves "
                "a matrix pseudoinverse (pinv) which may be numerically unstable. "
                "Consider setting <invertible>true</invertible> for shared-weight "
                "inversion, or be aware of potential SVD convergence failures.",
                stacklevel=2)

        if errors:
            raise ValueError(
                "XML config inconsistencies:\n  - " + "\n  - ".join(errors))

    @staticmethod
    def resolve_xml(path):
        """Resolve an XML config path relative to the project directory."""
        if os.path.isabs(path):
            return path
        # Try relative to project root first (handles "data/simple.xml")
        candidate = os.path.join(ProjectPaths.PROJECT_DIR, path)
        if os.path.exists(candidate):
            return candidate
        # Try inside data/ (handles bare "simple.xml")
        candidate = os.path.join(ProjectPaths.PROJECT_DIR, "data", path)
        if os.path.exists(candidate):
            return candidate
        return path

    @staticmethod
    def run(config_path):
        """Main entry point — create, train, and evaluate a model from XML config."""
        # Pre-read config for dataset loading (needed before create_from_config)
        defaults_path = os.path.join(ProjectPaths.DATA_DIR, "model.xml")
        init_config(path=config_path, defaults_path=defaults_path)
        cfg = TheXMLConfig.data
        arch = cfg.get("architecture", {})
        dat = arch.get("data", {})
        trn = arch.get("training", {})

        dataset = os.environ.get("BASIC_DATASET", dat.get("dataset"))
        # Environment overrides for num_shards/max_docs (set by train.py)
        num_shards = int(os.environ.get("BASIC_NUM_SHARDS", dat.get("numShards", 1)))
        max_docs = int(os.environ.get("BASIC_MAX_DOCS", dat.get("maxDocs", 10000)))
        TheData.load(dataset,
                     num_shards=num_shards,
                     max_docs=max_docs,
                     shard_dir=dat.get("shardDir"),
                     dat=dat)

        m = BasicModel()
        # Store config refs so runTrials can call create_from_config per trial
        m._config_path = config_path
        m._config_data = TheData
        TheMessage(f"Device: {TheDevice}")

        m = compile(m)

        def _t(key, default=None):
            return trn.get(key, default)

        def _d(key, default=None):
            return dat.get(key, default)

        m.run(_t("numTrials", 1),
                    _t("numEpochs", 3),
                    _t("batchSize", 10),
                    lr=_t("learningRate", 0.01))

        report_kwargs = {}
        cmin = _d("classificationMin")
        cmax = _d("classificationMax")
        if cmin is not None:
            report_kwargs["min"] = cmin
        if cmax is not None:
            report_kwargs["max"] = cmax
        if report_kwargs:
            m.classificationReport(**report_kwargs)

        if _t("autosave", False):
            wpath = TheXMLConfig.get("architecture.weightsPath", "weights.ckpt")
            wpath = m._resolve_artifact_path(wpath)
            m.save_weights(wpath)
            m.save_embeddings()

        return [(m.name, m.rCorrect, m)]

def test():
    """Smoke test: verify encodings and run the XOR config end-to-end."""
    WhereEncoding.test()
    WhenEncoding.test()
    BasicModelFactory.run(os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml"))


# --- CLI entry point ---
# Usage: python BasicModel.py [config.xml]
#        python BasicModel.py --compare config1.xml config2.xml
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog="BasicModel.py",
        description=(
            "Train and evaluate a BasicModel from an XML config file.\n\n"
            "Examples:\n"
            "  python BasicModel.py data/xor.xml\n"
            "  python BasicModel.py data/XOR_spaces.xml\n"
            "  python BasicModel.py --compare data/xor.xml data/XOR_exact.xml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        metavar="CONFIG",
        help=(
            "Path to the XML config file (relative to data/ or absolute). "
            "Defaults to data/xor.xml when omitted."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG1", "CONFIG2"),
        help=(
            "Run two configs side by side and plot per-digit accuracy, "
            "combined accuracy, and combined loss comparisons."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate figures and HTML report at the end of the run.",
    )
    args = parser.parse_args()

    TheReport.enabled = args.report

    if args.compare:
        # Compare mode: run two XML configs and plot per-digit accuracy side by side
        xml1 = BasicModelFactory.resolve_xml(args.compare[0])
        xml2 = BasicModelFactory.resolve_xml(args.compare[1])
        TheReport.add_xml(xml1)
        TheReport.add_xml(xml2)
        results = BasicModelFactory.run(xml1) + BasicModelFactory.run(xml2)
        if len(results) >= 2:
            TheReport.plotComparison([(name, rc) for name, rc, _ in results])
            TheReport.plotCombinedAccuracy([(name, rc) for name, rc, _ in results])
            TheReport.plotCombinedLoss([m for _, _, m in results])
    else:
        # Single run mode
        xml = BasicModelFactory.resolve_xml(args.config) if args.config else os.path.join(ProjectPaths.PROJECT_DIR, "data", "xor.xml")
        TheReport.add_xml(xml)
        results = BasicModelFactory.run(xml)

    TheReport.write_html()
