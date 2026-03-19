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
import util
util.init_runtime_env()
try:
    from torchviz import make_dot
except ImportError:
    make_dot = None
from matplotlib import pyplot as plt
from datasets import load_dataset
from embed import WordVectors, CBOWModel
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from Model import Layer, PiLayer, SigmaLayer, InvertibleSigmaLayer, InvertiblePiLayer # Import custom layers from Model.py
from lex import Lex
from Model import VQLayer, NormLayer, LinearLayer, InvertibleLinearLayer, AttentionLayer
from Model import GammaMem, ColumnUsageTracker, LiftingLayer, CertaintyWeightedCrossEntropy, epsilon
import torch.optim as optim
from functools import partial

# Device selection: BASICMODEL_DEVICE env var > cuda > mps > cpu.
# vector_quantize_pytorch einsum fails on MPS — set BASICMODEL_DEVICE=cpu
# for VQ-enabled configs.
from util import ProjectPaths, compile

TheDevice = util.TheDevice

from datetime import datetime
from visualize import Report, TheReport

class PositionalEncoding(nn.Module):
    """Encode spatial position (nWhere) as sin/cos values in reserved embedding slots.

    Writes a (sin, cos) pair into the last few dimensions of each object vector,
    indexed by ``self.index`` (negative offsets from the end).  A monotonic
    counter ``self.p`` assigns each object a unique position within a dataset
    pass; it must be reset between epochs to avoid overflow.

    Used by ObjectEncoding to stamp each object with a "where" tag.
    """
    nDim   = 2
    index  = [-4, -3]      # which embedding dimensions to write (negative = from end)
    p      = 0
    maxP   = 0
    period = [65521, 65537]

    def __init__(self, maxP=0):
        print("Creating positional encoding ...")
        super(PositionalEncoding, self).__init__()
        self.p    = 0
        self.maxP = maxP
        self.div_term = 2*math.pi / maxP
    def forward(self, x):
        """Stamp sin/cos positional values into reserved embedding slots."""
        batch = x.shape[0]
        n     = x.shape[1] if len(x.shape) > 1 else 1
        embeddingSize = x.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        # Write sin/cos position into the last reserved dimensions
        position = torch.arange(self.p, self.p+batch*n, dtype=torch.float32, device=x.device) * self.div_term
        p1 = torch.sin(position * self.div_term).unsqueeze(0).unsqueeze(0)
        p2 = torch.cos(position * self.div_term).unsqueeze(0).unsqueeze(0)
        pos = torch.concatenate((p1, p2), dim=2)
        y = x.clone()
        y[:, :, index] = pos.reshape(batch, n, self.nDim)
        self.p += batch
        assert self.p < self.maxP, "Overflow in object embedding"
        return y
    def stamp(self, buf, batch_idx, pos_idx, byte_offset):
        """Stamp sin/cos positional encoding at a specific buffer position.

        Uses the same encoding as forward(): sin/cos of (offset * div_term^2).

        Args:
            buf: [batch, nVec, embSize] tensor to write into
            batch_idx: which batch element
            pos_idx: which vector position within the element
            byte_offset: the byte offset to encode
        """
        embSize = buf.shape[-1]
        idx = np.add([embSize, embSize], self.index)
        if idx[0] >= 0 and idx[0] < embSize:
            p = byte_offset * self.div_term
            buf[batch_idx, pos_idx, idx[0]] = math.sin(p * self.div_term)
            buf[batch_idx, pos_idx, idx[1]] = math.cos(p * self.div_term)

    def reverse(self, y):
        """Extract and zero-out positional encoding; return (cleaned, positions)."""
        embeddingSize = y.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        # Guard: if embedding is too small for positional slots, return zeros
        if index[0] < 0 or index[0] >= embeddingSize:
            return y, torch.zeros(y.shape[0], y.shape[1], len(self.index), device=y.device)
        pos = y[:,:, index]
        y[:, :, index] = 0
        return y, pos
    @staticmethod
    def test():
        x=  torch.zeros([2,4,100])
        pe= PositionalEncoding(100)
        y = pe.forward(x)
        z = pe.reverse(y)
        print(z)
class TemporalEncoding(nn.Module):
    """Encode temporal order (nWhen) as cos values in reserved embedding slots.

    Similar to PositionalEncoding but tracks a global time counter ``self.t``
    that is incremented explicitly via ``increment(batch)`` at the end of each
    forward pass through the full model.  The two periods produce slowly-varying
    signals that distinguish objects seen at different times.

    Used by ObjectEncoding to stamp each object with a "when" tag.
    """
    nDim= 2
    index  = [-2, -1]      # which embedding dimensions to write (negative = from end)
    period = [1193, 2000147]
    t      = 0

    def __init__(self, maxT=0):
        super().__init__()
        self.t    = 0
        self.maxT = maxT
    def forward(self, x):
        """Stamp cos-based temporal values into reserved embedding slots."""
        batch = x.shape[0]
        n = x.shape[1] if len(x.shape) > 1 else 1
        embeddingSize = x.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        # Cosine encoding scaled to [0, 1] via 0.5*(1+cos(...))
        t1 = ( 0.5*(1+torch.cos(math.pi + 2*math.pi * torch.tensor(range(self.t, self.t+batch), device=x.device)/self.period[0] )) ).unsqueeze(0).unsqueeze(0)
        t2 = ( 0.5*(1+torch.cos(math.pi + 2*math.pi * torch.tensor(range(self.t, self.t+batch), device=x.device)/self.period[0] )) ).unsqueeze(0).unsqueeze(0)
        time = torch.concatenate((t1, t2), dim=2)
        y = x.clone()
        y[:, :, index] = time.reshape(batch, 1, self.nDim)
        return y

    def increment(self, batch):
        """Advance the global time counter by `batch` steps (called per forward pass)."""
        self.t += batch

    def reverse(self, y):
        """Extract and zero-out temporal encoding; return (cleaned, times)."""
        batch = y.shape[0]
        embeddingSize = y.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        # Guard: if embedding is too small for temporal slots, return zeros
        if index[0] < 0 or index[0] >= embeddingSize:
            return y, torch.zeros(y.shape[0], y.shape[1], len(self.index), device=y.device)
        t =  y[:, :, index]
        y[:, :, index] = 0
        return y, t
    @staticmethod
    def test():
        x=  torch.zeros([2,4,10])
        te= TemporalEncoding(4)
        y = te.forward(x)
        z = te.reverse(y)
        print(z)
class ObjectEncoding(nn.Module):
    """Augments each object vector with positional (nWhere) and temporal (nWhen) tags.

    Every vector in the pipeline has the layout:
        [nWhat content dims | nWhere (2 dims) | nWhen (2 dims)]

    ``nWhat`` varies per subspace (inputDim, perceptDim, conceptDim, wordDim, etc.)
    while nWhere and nWhen are fixed overhead.  ``objectSize = nWhere + nWhen`` is
    the total overhead appended to the content portion.

    ``getEncodingSize(nDim)`` returns ``nDim + objectSize`` — the full vector width
    for a given content dimensionality.

    ``computeNObjects()`` sets up spatial and temporal encodings:
        nObjects = nInput + nPercepts + nConcepts + nSymbols + nWords + nOutput
    Each object receives a unique spatial encoding via ``PositionalEncoding(nObjects)``.
    Temporal encoding supports up to t=10000 unique time steps via
    ``TemporalEncoding(10000)``.

    A single global instance ``TheObjectEncoding`` is used throughout the model.
    """
    # nWhat: varies by subspace — set via setInputDim/setPerceptDim/etc.
    nWhere       = PositionalEncoding.nDim
    nWhen        = TemporalEncoding.nDim

    inputDim     = 0
    perceptDim   = 0
    conceptDim   = 0
    symbolDim    = 0
    wordDim      = 0
    outputDim    = 0

    nInput    = 0  # codebook size for InputSpace
    nPercepts = 0  # codebook size for PerceptualSpace
    nConcepts = 0  # codebook size for ConceptualSpace
    nSymbols  = 0  # codebook size for SymbolicSpace
    nWords    = 0  # codebook size for SyntacticSpace
    nOutput   = 0  # codebook size for OutputSpace

    objectSize = nWhere + nWhen  # total encoding overhead per vector
    nObjects   = 0               # 0 = uninitialized sentinel
    what       = lambda x : True

    def __init__(self):
        super().__init__()
        # where/when must be instance attrs (not class attrs) so that
        # nn.Module.__setattr__ can register them as submodules later
        # without being shadowed by a class-level None.
        self.where = None
        self.when  = None

    def setDimensions(self, inputDim, perceptDim, conceptDim, symbolDim, outputDim):
        assert inputDim == perceptDim, "The input and percept dimensions do not match" # they are both input to concepts
        TheObjectEncoding.setInputDim(inputDim)
        TheObjectEncoding.setPerceptDim(perceptDim)
        TheObjectEncoding.setConceptDim(conceptDim)
        TheObjectEncoding.setSymbolDim(symbolDim)
        TheObjectEncoding.setOutputDim(outputDim)
    def setInputDim(self, nDim):
        self.inputDim = nDim
    def setPerceptDim(self, nDim):
        self.perceptDim = nDim
    def setConceptDim(self, nDim):
        self.conceptDim = nDim
    def setSymbolDim(self, nDim):
        self.symbolDim = nDim
    def setWordDim(self, nDim):
        self.wordDim = nDim
    def setOutputDim(self, nDim):
        self.outputDim = nDim

    def computeNObjects(self):
        """Compute nObjects and create positional/temporal encodings. Called once.

        nObjects = sum of all codebook sizes across spaces. Each object gets a
        unique spatial encoding; temporal encoding supports up to t=10000.
        """
        assert self.nObjects == 0, "computeNObjects must only be called once"
        self.nObjects = (self.nInput + self.nPercepts + self.nConcepts + self.nSymbols + self.nWords + self.nOutput)
        self.where = PositionalEncoding(self.nObjects)
        self.when = TemporalEncoding(10000)

    def getObjectEncodingSize(self, nDim):
        return nDim + self.objectSize
    def getInputEncodingSize(self):
        return self.getObjectEncodingSize(self.inputDim)
    def getPerceptEncodingSize(self):
        return self.getObjectEncodingSize(self.perceptDim)
    def getConceptEncodingSize(self):
        return self.getObjectEncodingSize(self.conceptDim)
    def getSymbolEncodingSize(self):
        return self.getObjectEncodingSize(self.symbolDim)
    def getWordEncodingSize(self):
        return self.getObjectEncodingSize(self.wordDim)
    def getOutputEncodingSize(self):
        return self.outputDim # the output is not embedded

    def pad(self, objects, where=True, when=True):
        size = 0
        size += self.nWhere if where else 0
        size += self.nWhen if when else 0
        objects = F.pad(objects, (0, size))
        return objects
    def slice(self, object, where=True, when=True):
        size = 0
        size += self.nWhere if where else 0
        size += self.nWhen if when else 0
        if size == 0:
            return object
        return object[0:-size]

    def forward(self, objects, what=False, where=True, when=True, pad=False):
        assert self.nObjects != 0, "nObjects was not set"
        if self.nObjects == 1: # no positional encoding if there is only one object
            return objects
        if pad:
            objects = self.pad(objects)
        if what:
            objects = self.what(objects)
        if where:
            objects = self.where(objects)
        if when:
            objects = self.when(objects)
        return objects
    def reverse(self, objects):
        assert self.nObjects != 0, "nObjects was not set"
        objects, space = self.where.reverse(objects)
        objects, time  = self.when.reverse(objects)
        return objects, space, time
    #@staticmethod
    #def removeEncoding(x):
    #    e = x.shape[-1]
    #    e -= TheObjectEncoding.objectSize
    #    if len(x.shape) == 2:
    #       x = x[:, 0:e]
    #   else:
    #        x = x[:, :, 0:e]
    #    return x
TheObjectEncoding = ObjectEncoding()

class Data():
    """Dataset container: loads, preprocesses, and serves train/validation/test splits.

    Supports MNIST (numeric), XOR (toy text), and Rotten Tomatoes (real text).
    Text datasets go through ``processLM()`` which tokenizes via ``stringTensor()``
    (ASCII byte encoding, zero-padded to ``inputLength``) and builds an immutable
    source buffer for span-table integration.

    After ``load()``, tensors are moved to ``TheDevice`` and pre-shaped to
    [N, D, 1] so the training loop avoids per-batch unsqueezes.

    A single global instance ``TheData`` is used by ``BasicModelFactory.run()``.
    """
    train_input       = []
    train_output      = []
    validation_input  = []
    validation_output = []
    test_input        = []
    test_output       = []

    _runtime_input    = None
    _runtime_output   = None

    inputLength       = 4096   # max byte length for text inputs (zero-padded)
    combinedTokens    = []

    def __init__(self):
        self.train_input       = []
        self.train_output      = []
        self.validation_input  = []
        self.validation_output = []
        self.test_input        = []
        self.test_output       = []
        self.combinedTokens    = []
        self.train_texts       = None
        self.test_texts        = None

    @property
    def nInput(self):
        """Number of input features, derived from train_input shape."""
        if isinstance(self.train_input, torch.Tensor):
            return self.train_input.shape[1]
        return len(self.train_input)  # list of strings for LM

    @property
    def inputDim(self):
        """Dimensionality per input feature, derived from train_input shape."""
        if isinstance(self.train_input, torch.Tensor) and self.train_input.ndim >= 3:
            return self.train_input.shape[2]
        return 1

    @property
    def nOutput(self):
        """Number of output features, derived from train_output shape."""
        if isinstance(self.train_output, torch.Tensor):
            return self.train_output.shape[1]
        if isinstance(self.train_output, list) and len(self.train_output) > 0:
            return len(self.train_output[0])
        return 0

    @property
    def outputDim(self):
        """Dimensionality per output feature, derived from train_output shape."""
        if isinstance(self.train_output, torch.Tensor) and self.train_output.ndim >= 3:
            return self.train_output.shape[2]
        return 1

    def load(self, dataset, num_shards=1, max_docs=10000, shard_dir=None, dat=None):
        if dataset == "mnist":
            self.loadMNist()
        if dataset == "xor":
            self.loadXOR()
        if dataset == "tomatoes":
            self.loadTomatoes()
        if dataset == "text":
            self.loadShards(num_shards, max_docs, shard_dir)
        if dataset == "inline":
            self.loadInline(dat or {})
        self.toDevice()
    def toDevice(self):
        """Move all data tensors to TheDevice and pre-shape for training.

        Adds the trailing dimension (unsqueeze) that prepInput/prepOutput
        previously applied per-batch, so the hot loop can skip both the
        unsqueeze and the redundant .to() call.
        """
        for attr in ("train_input", "train_output",
                      "test_input", "test_output",
                      "validation_input", "validation_output"):
            v = getattr(self, attr)
            if isinstance(v, torch.Tensor):
                v = v.to(TheDevice)
                if v.ndim == 2:          # [N, D] → [N, D, 1]
                    v = v.unsqueeze(2)
                setattr(self, attr, v)
            elif isinstance(v, list):
                setattr(self, attr, [
                    t.to(TheDevice) if isinstance(t, torch.Tensor) else t
                    for t in v
                ])

    @contextmanager
    def runtime_batch(self, inputs, outputs=None, sentences=None, mode=None):
        """Stage transient inference data as a 'runtime' split.

        Args:
            inputs: list of input tensors/strings for the runtime split.
            outputs: optional list of output targets.
            sentences: optional list of sentence strings for masked prediction.
            mode: optional inference mode ('ARIR', 'ARLM', or None).
        """
        self._runtime_input = inputs
        self._runtime_output = outputs
        self._runtime_mode = mode
        if sentences is not None:
            if not hasattr(self, '_lm_sentences'):
                self._lm_sentences = {}
            self._lm_sentences["runtime"] = sentences
        try:
            yield
        finally:
            self._runtime_input = None
            self._runtime_output = None
            self._runtime_mode = None
            if hasattr(self, '_lm_sentences'):
                self._lm_sentences.pop("runtime", None)

    def pushInput(self, token):
        """Append a token to the runtime input buffer for the next runBatch() call.

        For text-based ARLM inference the runtime input is a list containing
        a single tensor (byte-encoded string).  Decodes the existing bytes,
        appends the token, and re-encodes.
        """
        assert self._runtime_input is not None, "pushInput requires an active runtime_batch"
        if not isinstance(self._runtime_input, list) or len(self._runtime_input) == 0:
            raise TypeError(f"pushInput: unsupported runtime_input type {type(self._runtime_input)}")
        elem = self._runtime_input[0]
        if isinstance(elem, str):
            self._runtime_input[0] = elem + token
        elif isinstance(elem, torch.Tensor):
            # Decode existing byte tensor to text, append token, re-encode
            chars = [chr(int(b) & 0xFF) for b in elem.tolist()]
            existing = "".join(chars).rstrip("\x00")
            self._runtime_input[0] = self.stringTensor(existing + token)
        else:
            raise TypeError(f"pushInput: unsupported element type {type(elem)}")

    def pushOutput(self, token):
        """Append a target to the runtime output buffer (training only).

        For text-based data, appends the token to the first output sentence.
        Not used during inference but provided for completeness.
        """
        assert self._runtime_output is not None, "pushOutput requires an active runtime_batch"
        if isinstance(self._runtime_output, list) and len(self._runtime_output) > 0:
            self._runtime_output[0] = self._runtime_output[0] + " " + token
        else:
            raise TypeError(f"pushOutput: unsupported runtime_output type {type(self._runtime_output)}")

    def shuffle(self):
        rand_indx = torch.randperm(len(self.train_output))
        if isinstance(self.train_input, list):
            self.train_input  = [self.train_input[i] for i in rand_indx]
            self.train_output = [self.train_output[i] for i in rand_indx]
        else:
            self.train_input = self.train_input[rand_indx]
            self.train_output = self.train_output[rand_indx]
        if getattr(self, 'masked_prediction', 'NONE') != 'NONE':
            self._lm_sentences["train"] = [self._lm_sentences["train"][i] for i in rand_indx]
    def loadMNist(self):
        df = pd.read_csv(os.path.join(ProjectPaths.DATA_DIR, 'mnist_train.csv'))
        train = df.values
        df = pd.read_csv(os.path.join(ProjectPaths.DATA_DIR, 'mnist_test.csv'))
        test = df.values
        self.train_input  = torch.tensor(train[:, 1:]/255.0, dtype=torch.float)
        mnistMean = torch.mean(self.train_input)
        self.train_input = self.train_input - mnistMean
        mnistSTD = torch.std(self.train_input)
        self.train_input = self.train_input / mnistSTD
        self.train_output = torch.zeros((train.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(train[:, 0]):
            self.train_output[i][ndx:ndx+1] = 1.0
        self.test_input  = torch.tensor(test[:, 1:]/255.0, dtype=torch.float)
        self.test_input  = (self.test_input - mnistMean) / mnistSTD
        self.test_output = torch.zeros((test.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(test[:, 0]):
            self.test_output[i][ndx:ndx+1] = 1.0
        self.validation_input  = torch.tensor(test[:, 1:]/255.0, dtype=torch.float)
        self.validation_output = torch.zeros((test.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(test[:, 0]):
            self.validation_output[i][ndx:ndx+1] = 1.0
        self.inputLength = 28 * 28
    def loadXOR(self):
        data = {
            "train": {
                "text": ["hello world", "hello there", "loving world", "loving there" ], # nPercepts = 3
                "label": [[0], [1], [1], [0]]
                #"label": [[0, 1], [1, 0], [1, 0], [0, 1]]
            },
            "validation": {
                "text": ["hello world", "hello there", "loving world", "loving there" ], # nPercepts = 3
                "label": [[0], [1], [1], [0]]
                #"label": [[0, 1], [1, 0], [1, 0], [0, 1]]
            },
            "test": {
                "text": ["hello world", "hello there", "loving world", "loving there" ], # nPercepts = 3
                "label": [[0], [1], [1], [0]]
                #"label": [[0, 1], [1, 0], [1, 0], [0, 1]]
            }
        }
        self.train_input      = data["train"]["text"]
        self.train_output      = data["train"]["label"]
        self.validation_input = data["validation"]["text"]
        self.validation_output = data["validation"]["label"]
        self.test_input       = data["test"]["text"]
        self.test_output       = data["test"]["label"]
        self.processLM(data)
    def loadInline(self, dat):
        """Load dataset from inline XML ``<input use="...">`` / ``<output use="...">`` elements.

        ``dat`` is the parsed ``<data>`` dict.  ``<input>`` and ``<output>``
        children with a ``use`` attribute carry pipe-separated sentence strings
        and numeric labels respectively::

            <input use="train">zero xor zero|zero xor one|one xor zero|one xor one</input>
            <output use="train">0|1|1|0</output>

        Missing ``use="validation"`` falls back to the test split.
        """
        def _items(key):
            v = dat.get(key, [])
            return v if isinstance(v, list) else [v]

        def _get_split(items, use_val):
            for item in items:
                if isinstance(item, dict) and item.get("use") == use_val:
                    return str(item.get("_", ""))
            return ""

        def _parse_pipe(text):
            return [s.strip() for s in text.split("|") if s.strip()]

        def _parse_labels(raw):
            result = []
            for v in raw:
                try:
                    result.append([float(v)])
                except ValueError:
                    result.append([0.0])
            return result

        inputs  = _items("input")
        outputs = _items("output")

        train_texts  = _parse_pipe(_get_split(inputs,  "train"))
        test_texts   = _parse_pipe(_get_split(inputs,  "test"))
        val_texts    = _parse_pipe(_get_split(inputs,  "validation")) or test_texts

        train_labels = _parse_labels(_parse_pipe(_get_split(outputs, "train")))
        test_labels  = _parse_labels(_parse_pipe(_get_split(outputs, "test")))
        val_labels   = _parse_labels(_parse_pipe(_get_split(outputs, "validation"))) or test_labels

        data = {
            "train":      {"text": train_texts, "label": train_labels},
            "validation": {"text": val_texts,   "label": val_labels},
            "test":       {"text": test_texts,  "label": test_labels},
        }
        self.processLM(data)
    def loadTomatoes(self):
        cache_file = os.path.join(ProjectPaths.DATA_DIR, "rottenTomatoes.data")

        # Load or cache the pre-trained Word2Vec model
        if os.path.exists(cache_file):
            print("Loading cached data...")
            data = torch.load(cache_file, weights_only=False)
        else:
            print("Downloading data...")
            data = load_dataset("rotten_tomatoes")
            torch.save(data, cache_file)
        self.processLM(data)
    def loadShards(self, num_shards, max_docs, shard_dir):
        """Load training text from FineWeb-EDU parquet shards.

        Uses the same shard infrastructure as embed.py so the model trains
        on the same corpus that produced the word embeddings.
        """
        from embed import get_shard_paths, iter_documents

        if shard_dir is None:
            shard_dir = os.path.join(ProjectPaths.DATA_DIR, "fineweb")
        # Resolve relative paths against project root
        if not os.path.isabs(shard_dir):
            shard_dir = os.path.join(ProjectPaths.PROJECT_DIR, shard_dir)

        print(f"Loading text: {num_shards} shard(s), max {max_docs} docs "
              f"from {shard_dir}")
        shard_paths = get_shard_paths(shard_dir, num_shards=num_shards)
        if not shard_paths:
            raise RuntimeError(f"No shards found in {shard_dir}. "
                               "Run 'make basic_data' first.")

        docs = list(iter_documents(shard_paths, max_docs=max_docs))
        if not docs:
            raise RuntimeError("No documents found in shards.")

        # Split 80/10/10 into train/validation/test
        random.shuffle(docs)
        n = len(docs)
        n_val = max(1, n // 10)
        n_test = max(1, n // 10)
        n_train = n - n_val - n_test

        train_texts = docs[:n_train]
        val_texts = docs[n_train:n_train + n_val]
        test_texts = docs[n_train + n_val:]

        data = {
            "train":      {"text": train_texts, "label": []},
            "validation": {"text": val_texts,   "label": []},
            "test":       {"text": test_texts,  "label": []},
        }

        print(f"Loaded {n_train} train, {n_val} val, {n_test} test documents")
        self.processLM(data)
    def processLM(self, data, permute=True):
        train_tokens      = data["train"]["text"]
        train_labels      = data["train"]["label"]
        validation_tokens = data["validation"]["text"]
        validation_labels = data["validation"]["label"]
        test_tokens       = data["test"]["text"]
        test_labels       = data["test"]["label"]

        self.combinedTokens = train_tokens + validation_tokens + test_tokens
        self.combinedTokens = list(set(self.combinedTokens))

        # Store raw text lists for per-document vocab building
        self.train_texts = train_tokens
        self.test_texts = test_tokens

        self.train_input  = [self.stringTensor(t) for t in train_tokens]
        self.validation_input  = [self.stringTensor(t) for t in validation_tokens]
        self.test_input  = [self.stringTensor(t) for t in test_tokens]

        # For masked LM, labels are target word strings — store for later
        # conversion to embedding vectors by prepare_lm_targets().
        # For non-LM tasks, labels are numeric lists.
        if not train_labels:
            # Masked prediction mode: targets computed at runtime
            self._lm_sentences = {
                "train": list(train_tokens),
                "validation": list(validation_tokens),
                "test": list(test_tokens),
            }
            self.masked_prediction = 'MLM'
            # Sentinel outputs for OutputSpace sizing — shape must match embedding dim
            # (actual targets computed at runtime by expand_masked)
            self.train_output = [torch.zeros(1) for _ in train_tokens]
            self.validation_output = [torch.zeros(1) for _ in validation_tokens]
            self.test_output = [torch.zeros(1) for _ in test_tokens]
            self._lm_labels = None
        elif isinstance(train_labels[0], str):
            self._lm_labels = {
                "train": list(train_labels),
                "validation": list(validation_labels),
                "test": list(test_labels),
            }
            # Placeholder until embeddings are available
            self.train_output = [torch.zeros(1) for _ in train_labels]
            self.validation_output = [torch.zeros(1) for _ in validation_labels]
            self.test_output = [torch.zeros(1) for _ in test_labels]
        else:
            self._lm_labels = None
            self.train_output = [torch.tensor(l, dtype=torch.float) for l in train_labels]
            self.validation_output = [torch.tensor(l, dtype=torch.float) for l in validation_labels]
            self.test_output = [torch.tensor(l, dtype=torch.float) for l in test_labels]

        if permute:
            rand_indx = torch.randperm(len(self.train_output))
            self.train_input  = [self.train_input[i] for i in rand_indx]
            self.train_output = [self.train_output[i] for i in rand_indx]
            if self._lm_labels is not None:
                self._lm_labels["train"] = [self._lm_labels["train"][i] for i in rand_indx]
            if getattr(self, 'masked_prediction', 'NONE') != 'NONE':
                self._lm_sentences["train"] = [self._lm_sentences["train"][i] for i in rand_indx]

    def tokenize(self, TheLanguageModel):
        self.train_input      = TheLanguageModel.tokenize(self.train_input)
        self.validation_input = TheLanguageModel.tokenize(self.validation_input)
        self.test_input       = TheLanguageModel.tokenize(self.test_input)
    def data(self):
        data = {
            "train": {
                "text": self.train_input,
                "label": self.train_output
            },
            "validation": {
                "text":self.validation_input,
                "label": self.validation_output
            },
            "test": {
                "text": self.test_input,
                "label": self.test_output
            }
        }
    def getEmbeddingSize(self):
        return self.getInputSize(), self.getOutputSize()
    def getInputSize(self):
        inShape = len(self.train_input)
        inputEmbeddingSize  = self.train_input[0].shape[0]
        return inputEmbeddingSize
    def getOutputSize(self):
        outShape = len(self.train_output)
        outputEmbeddingSize = self.train_output[0].shape[0]
        return outputEmbeddingSize
    def prepare_lm_targets(self, embedding):
        """Convert word-string labels to embedding vectors for masked LM training.

        Called after InputSpace creates the Embedding, so word vectors are available.
        Each target word is looked up in the CBOW vocabulary; unknown words get
        a zero vector (the model should learn to predict known words).

        Args:
            embedding: An Embedding instance with cbow.key_to_index and _emb weights.
        """
        if self._lm_labels is None:
            return
        weights = embedding._emb.weight  # (vocab_size, vec_size)
        vec_size = weights.shape[1]
        vocab_size = weights.shape[0]

        def words_to_embeddings(word_list):
            targets = []
            for word in word_list:
                w = word.lower()
                if w in embedding.cbow.key_to_index:
                    idx = embedding.cbow.key_to_index[w]
                    one_hot = torch.zeros(vocab_size, device=weights.device)
                    one_hot[idx] = 1.0
                    v = one_hot @ weights  # differentiable
                    v = F.normalize(v, p=2, dim=0)
                else:
                    v = torch.zeros(vec_size, device=weights.device)
                targets.append(v)
            return targets

        self.train_output = words_to_embeddings(self._lm_labels["train"])
        self.validation_output = words_to_embeddings(self._lm_labels["validation"])
        self.test_output = words_to_embeddings(self._lm_labels["test"])
        print(f"LM targets: {len(self.train_output)} train, "
              f"{len(self.validation_output)} val, "
              f"{len(self.test_output)} test "
              f"(embedding dim={vec_size})")
        self._lm_labels = None  # free memory

    def stringTensor(self, string):
        # Encode to ASCII, replacing non-ASCII chars (smart quotes, accents, etc.)
        ascii_values = list(string.encode('ascii', errors='replace'))[:self.inputLength]
        tensor = torch.tensor(ascii_values, dtype=torch.int8)
        if tensor.size(0) < self.inputLength:
            tensor = F.pad(tensor, (0, self.inputLength - tensor.size(0)), 'constant', 0)
        return tensor
TheData = Data()
class Message():
    def __call__(self, txt, newline="\n"):
        print(txt, end=newline)
message = Message()

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
    nDim             = 0   # content dimensionality (before ObjectEncoding overhead)
    embeddingSize    = 0   # nDim + objectSize = full vector width
    vectors          = nn.Parameter(torch.randn([nVectors, embeddingSize]))
    snapDistance     = 0.1 # distance threshold for snapping to prototypes
    eta              = 0.9 # EMA decay for manual VQ updates (customVQ=False)
    codebookAct      = GammaMem(nVectors)  # tracks per-entry activation history
    frozen           = []  # indices of locked codebook entries
    returnOnlyFrozen = False
    freezingTemp     = 0.25  # activation threshold above which entries freeze
    passThrough      = False

    def getSize(self):
        return self.nVectors

    def create(self, nInput, nVectors, nDim, customVQ=True, signed=False, passThrough=False):
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
        if nDim != None:
            self.embeddingSize = TheObjectEncoding.getObjectEncodingSize(nDim)
        if passThrough:
            return
    def updateWeights(self, embed_sum, cluster_size):
        # Zero out gradients for frozen indices
        weights = torch.ones(self.vq.codebook_size, device=self.vq.codebook.device)
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
                    message(f"Frozen activation at {indexMax}")
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
                vec[i, :] = F.normalize(TheObjectEncoding(vec[i, :].unsqueeze(0).unsqueeze(0)), p=2, dim=1)
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
        act   = torch.zeros([batch, self.codebookSize], device=input.device)
        if self.customVQ:
            if x.shape[-1] == self.nDim:
                x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize], device=x.device)], dim=2)
            y   = torch.reshape(x, [-1, self.embeddingSize])

            # VQ einsum fails on MPS — round-trip VQ module and data through CPU
            _dev = y.device
            self.vq.cpu()
            quantized, indices, commit_loss = self.vq(y.cpu(), ema_update_weight=self.updateWeights)
            self.vq.to(_dev)
            quantized, indices = quantized.to(_dev), indices.to(_dev)

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
                        #message("codebook miss")
                        x[i, j, :] = quantized[i, indices_smallest[i, j], :]
                        #x[i, j, :] = torch.zeros( [1, 1, self.embeddingSize])
        else:
            dists = self.codebookDistance(x)
            x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize], device=x.device)], dim=2)
            # Project the set of input vectors onto the basis vectors (the vector set).
            # Then compute the column norm of the basis, which result in activations (neuron power).
            # The top of those activations become the "Conscious" set of the current space.
            for b in range(0, x.shape[0]):
                for v in range(0, x.shape[1]):
                    nearestDist, nearestIdx = torch.topk(dists[b,v,:], 1, dim =-1, largest=True)
                    err = nearestDist[0]
                    if err <= self.snapDistance:
                        #message("Using prototype vector")
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
    def reverse(self, y):
        if self.passThrough:
            return y
        return y  # existing VectorSet has no explicit reverse beyond identity

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
            new_vectors[i, :] = TheObjectEncoding(new_vectors[i, :])
        self.vectors = torch.cat([self.vectors, new_vectors], dim=0)
        self.nVectors = self.vectors.shape[0]
    # --- Vector Removal ---
    def remove(self, indices):
        """
        Remove vectors by index
        indices: list or tensor of indices to remove
        """
        mask = torch.ones(self.vectors.shape[0], dtype=torch.bool, device=self.vectors.device)
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
                x = torch.maximum(torch.minimum(x, torch.tensor(1.0, device=x.device)), torch.tensor(0.0, device=x.device))
                x = F.normalize(x, p=2, dim=-1)
            return x
    def negate(self, x):
        return 1 - x
    def distance(self, x, y):
        N = self.codebookSize
        dist = (x.T @ y) / N
        return dist
    def codebookDistance(self, x):
        vec = self.vectors[:, 0:self.nDim] if TheObjectEncoding.objectSize == 0 else self.vectors[:, 0:-TheObjectEncoding.objectSize]
        vec = vec.to(x.device)
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
    """VectorSet backed by a differentiable nn.Embedding with online CBOW training.

    Loads pretrained weights from a WordVectors artifact (e.g. sentence.pt)
    and wraps them in an nn.Embedding so gradients can flow through the
    forward pass.  ``train_step(words)`` runs one sentence-level CBOW
    update using the same objective that produced the pretrained weights.
    """

    def __init__(self):
        super().__init__()
        self._lex = Lex()
        self.cbow = None       # CBOWModel, created in create()
        self._emb = None       # nn.Embedding, shared with cbow
        self.wv = None         # WordVectors snapshot for reverse()
        self.ergodic = False   # when True, inject per-word noise from SBOW sigma
        self.sigma_kappa = 0.01

    def tokenize(self, data):
        """Tokenize byte-tensor batch into token lists via Lex.

        All token categories from ``lex_buffer()`` are included — WORD, SPACE,
        SEPARATOR, PUNCT.  With the current Lex rules:
        - Digit runs are categorised as WORD (not NUMBER)
        - SPACE is emitted only between words within a sentence; whitespace
          after a SEPARATOR (inter-sentence) is silently discarded

        Side effect: stores ``self._byte_offsets`` as a list of lists of
        byte offsets (one per token per batch element), for nWhere encoding.
        """
        tokenized = []
        self._byte_offsets = []
        for b in range(len(data)):
            sentence = "".join(chr(int(i) & 0xFF) for i in data[b].tolist())
            sentence = sentence.rstrip("\x00")
            tokens = self._lex.lex_buffer(sentence, 0)
            tokenized.append([tok['text'] for tok in tokens])
            self._byte_offsets.append([tok['start'] for tok in tokens])
        return tokenized

    def create(self, nInput=None, nVectors=None, nDim=None, passThrough=True,
               wv=None, embedding_path=None, source=None, learning_rate=0.001,
               min_frequency=0.0, neg_samples=64):
        """Initialise from WordVectors or load from embedding_path.

        Accepts the same positional signature as VectorSet.create() so it
        can be used as a drop-in vectorSet member of InputSpace.

        If *wv* is provided, use it directly.  Otherwise load from
        *embedding_path* if given (.pt or .txt).  When no file is found or
        path is None, starts with an empty vocabulary — new words are
        added dynamically during forward passes via ``_add_word()``.

        If *source* is provided (a raw text string), build a Lex span table
        and a token_id → embedding index mapping for grammatical tokenization.
        Sets ``self.doc_spans``, ``self.doc_sources``, and ``self.lex_to_emb``.
        """
        if wv is None:
            wv = self._load_embeddings(embedding_path=embedding_path, nDim=nDim)
        if wv is None:
            # No matching embeddings on disk — start with a single placeholder.
            # Real words are added dynamically during forward passes via _add_word().
            dim = nDim or 20
            print(f"Starting with dynamic {dim}-dim embedding (words added at runtime)")
            placeholder = torch.randn(1, dim)
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

        self.cbow = CBOWModel(wv, learning_rate=learning_rate, neg_samples=neg_samples)
        self._emb = self.cbow.embeddings   # shared nn.Embedding
        self.min_frequency = float(min_frequency)
        self._pending_counts: dict = {}

        # Add [MASK] as zero-vector codebook entry (internal-only token)
        self._add_word("[MASK]")
        with torch.no_grad():
            self.mask_token_idx = self.cbow.key_to_index["[MASK]"]
            self._emb.weight[self.mask_token_idx] = 0.0

        # Add \x00 as the EOS/padding sentinel.
        # Gets a regular random embedding — distinct from [MASK]'s all-zeros.
        # Used to fill padding positions in _forward_lex so the model has a
        # concrete reconstruction target for positions beyond the real tokens.
        self._add_word("\x00")
        self.null_token_idx = self.cbow.key_to_index["\x00"]

        # Bootstrap codebook with printable ASCII (32–126) so any OOV word
        # can be spelled out character-by-character without zero vectors.
        for cp in range(32, 127):
            ch = chr(cp)
            if ch not in self.cbow.key_to_index:
                self._add_word(ch)

        # Span-table integration for grammatical tokenizer
        if source is not None:
            # Process each document individually instead of one giant concatenation
            self.doc_spans = []
            self.doc_sources = []
            for doc in source:
                doc_tensor = torch.tensor(list(doc.encode('utf-8')), dtype=torch.uint8)
                self._lex.build_vocab(doc_tensor)
                self.doc_spans.append(self._lex.encode(doc_tensor))
                self.doc_sources.append(doc_tensor)
            self.lex_to_emb = {}
            for word, token_id in self._lex.vocab.items():
                if word not in self.wv:
                    self._add_word(word)
                self.lex_to_emb[token_id] = self.wv.key_to_index[word]

    def _observe(self, word):
        """Observe a word token; promote to vocab once it meets min_frequency.

        Returns the embedding vector if the word is (or just became) in vocab,
        or None if it is still below the frequency threshold (OOV).
        """
        self.wv.total_count += 1
        if word in self.cbow.key_to_index:
            idx = self.cbow.key_to_index[word]
            self.wv.counts[idx] += 1
            return self._emb.weight[idx].detach()
        count = self._pending_counts.get(word, 0) + 1
        self._pending_counts[word] = count
        if self.min_frequency <= 0 or (
                self.wv.total_count > 0 and
                count / self.wv.total_count >= self.min_frequency):
            return self._add_word(word, initial_count=count)
        return None

    def _add_word(self, word, initial_count=0):
        """Unconditionally add a word with a random normalized embedding.

        Extends the WordVectors, CBOWModel vocabulary, and nn.Embedding so
        that subsequent forward passes can look up this word.  Returns the
        new embedding vector on the same device as existing weights.
        Call ``_observe()`` instead for frequency-gated runtime additions.
        """
        dim = self.wv.vector_size
        device = self._emb.weight.device
        new_vec = torch.randn(1, dim, device=device)
        new_vec = F.normalize(new_vec, p=2, dim=1)

        # Extend WordVectors (stays on CPU for neighbor queries)
        new_vec_np = new_vec.cpu().numpy()
        self.wv._vectors = np.concatenate([self.wv._vectors, new_vec_np], axis=0)
        self.wv.counts = np.append(self.wv.counts, np.int64(initial_count))
        self._pending_counts.pop(word, None)
        self.wv._normed = None  # invalidate cache
        idx = len(self.wv.index_to_key)
        self.wv.index_to_key.append(word)
        self.wv.key_to_index[word] = idx

        # Extend CBOWModel
        self.cbow.index_to_key.append(word)
        self.cbow.key_to_index[word] = idx
        old_emb = self.cbow.embeddings
        new_emb = nn.Embedding(idx + 1, dim, device=device)
        with torch.no_grad():
            if old_emb.weight.shape[0] > 0:
                new_emb.weight[:idx] = old_emb.weight
            new_emb.weight[idx] = new_vec.squeeze()
        self.cbow.embeddings = new_emb
        self._emb = new_emb

        # Rebuild optimizer with new embedding parameters
        self.cbow.optimizer = torch.optim.Adam(
            self.cbow.embeddings.parameters(),
            lr=self.cbow.optimizer.param_groups[0]['lr'],
        )

        return new_vec.squeeze(0)  # (1, dim) -> (dim,); preserves 1-dim case

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

    def tokenizeList(self, data):
        """Build vocabulary from a list of strings via Lex."""
        tokenized = []
        for sentence in data:
            tokens = self._lex.lex_buffer(sentence, 0)
            for tok in tokens:
                #if tok['category'] == 'WORD':
                tokenized.append(tok['text'])
        vocab = list(set(tokenized))
        # Include padding sentinel — it has a codebook entry and participates
        # in reconstruction targets for positions beyond the real tokens.
        if "\x00" not in vocab:
            vocab.append("\x00")
        return vocab

    def untokenize(self, tokenized):
        """Convert batch of token lists back to byte tensors."""
        data = []
        for b in range(len(tokenized)):
            sentence = ""
            for w in range(self.nInput):
                sentence += tokenized[b][w]
                if w < self.nInput - 1:
                    sentence += " "
            data.append(TheData.stringTensor(sentence))
        data = torch.stack(data)
        return data.unsqueeze(1)

    def getDictionary(self):
        """Return {word: encoded_vector} for every word in the vocabulary."""
        dictionary = {}
        for key in self.wv.index_to_key:
            word = torch.reshape(self.wv[key].detach().clone(), [1, 1, self.nDim])
            dictionary[key] = TheObjectEncoding.forward(word, pad=True).squeeze()
        return dictionary

    def getVectors(self):
        return self.wv.get_normed_vectors()

    def getSize(self):
        return len(self.wv)

    # --- Public properties (encapsulate internals) --------------------
    @property
    def codebook_weight(self):
        """nn.Embedding weight tensor (read-only reference)."""
        return self._emb.weight

    @property
    def vocab_keys(self):
        """Ordered list of words in the codebook."""
        return self.wv.index_to_key

    @property
    def embedding_dim(self):
        """Content dimensionality (nWhat)."""
        return self._emb.weight.shape[1]

    def embedding_parameters(self):
        """Return nn.Embedding parameters for optimizer inclusion."""
        return list(self._emb.parameters())

    # --- Lex delegation ---------------------------------------------------
    def encode(self, source_tensor):
        """Delegate to Lex.encode() — returns span table [num_spans, 3]."""
        return self._lex.encode(source_tensor)

    def lex_buffer(self, buf, start=0):
        """Delegate to Lex.lex_buffer() — returns list of token dicts."""
        return self._lex.lex_buffer(buf, start)

    def id_to_word(self, token_id):
        """Look up word string from Lex token ID. Returns '' for unknown IDs."""
        return self._lex.id_to_word.get(token_id, "")

    def forward(self, input):
        """Tokenize via Lex, look up embedding vectors from codebook.

        Input: (batch, max_len) byte tensor from Data.
        Output: (batch, nInput, embeddingSize) padded embedding tensor.

        Side effect: stores self._last_tokens (list of word lists per batch)
        and self._last_span_counts for encode_input metadata.
        """
        if input.dim() == 3:
            input = input.squeeze(1)  # [batch, 1, len] -> [batch, len]
        if input.dim() == 1:
            input = input.unsqueeze(0)  # [len] -> [1, len]
        batch = input.shape[0]
        codebook = self._emb.weight.detach().to(input.device)
        pad_size = TheObjectEncoding.objectSize

        self._last_tokens = []
        self._last_span_counts = []

        result = torch.zeros([batch, self.nInput, self.embeddingSize], device=input.device)
        for b in range(batch):
            raw_bytes = input[b].squeeze().tolist()
            text = "".join(chr(int(c) & 0xFF) for c in raw_bytes).rstrip("\x00")
            source_tensor = torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8)
            spans = self.encode(source_tensor)
            n_spans = min(spans.shape[0], self.nInput)
            self._last_span_counts.append(n_spans)
            b_tokens = []
            for i in range(n_spans):
                token_id = spans[i, 2].item()
                b_tokens.append(self.id_to_word(token_id))
                cb_idx = None
                if token_id in self.lex_to_emb:
                    cb_idx = self.lex_to_emb[token_id]
                else:
                    word = b_tokens[-1]
                    if word and word[0] in self.cbow.key_to_index:
                        cb_idx = self.cbow.key_to_index[word[0]]
                if cb_idx is not None:
                    vec = codebook[cb_idx]
                    vec = F.pad(vec, (0, pad_size))
                    vec = F.normalize(vec, p=2, dim=0)
                    result[b, i, :] = vec
            self._last_tokens.append(b_tokens)
            # Fill remaining positions with NULL (\x00) embedding
            null_idx = getattr(self, 'null_token_idx', None)
            if null_idx is not None:
                null_vec = codebook[null_idx]
                null_vec = F.pad(null_vec, (0, pad_size))
                null_vec = F.normalize(null_vec, p=2, dim=0)
                for i in range(n_spans, self.nInput):
                    result[b, i, :] = null_vec
        return result

    def encode_input(self, input_tensor, return_meta=False):
        """Tokenize + embed in one call.

        Args:
            input_tensor: [batch, max_len] byte tensor
            return_meta: if True, also return token metadata

        Returns:
            embedded: [batch, nInput, embeddingSize] tensor
            meta (only if return_meta): dict with 'tokens' and 'span_counts'
        """
        embedded = self.forward(input_tensor)
        if not return_meta:
            return embedded
        meta = {
            'tokens': self._last_tokens,
            'span_counts': self._last_span_counts,
        }
        return embedded, meta

    def decode_output(self, vectors):
        """Snap vectors to codebook and return words/positions.

        Composes reverse() + reconstruct_text() into a single call.

        Args:
            vectors: [batch, nVec, embSize] tensor

        Returns:
            dict with 'words' (list of list of str),
                       'positions' (recovered positions tensor),
                       'times' (recovered times tensor)
        """
        self.reverse(vectors)
        return {
            'words': self.reconstruct_text(join=False),
            'positions': self._recovered_positions,
            'times': self._recovered_times,
        }

    def set_sigma(self, sigma):
        """Control exploration for embedding lookups."""
        if sigma == 0:
            self.sigma_kappa = 1e6
        else:
            self.sigma_kappa = 0.01 / sigma

    def train_step(self, words, method='SBOW'):
        """One sentence-level embedding gradient step.  Returns loss or None.

        method: 'CBOW' — predict each word from leave-one-out context (padded)
                'SBOW' — predict each word from leave-one-out centroid (vectorised)
        """
        if method == 'SBOW':
            return self.cbow.sbow_step(words)
        return self.cbow.train_step(words)

    def snapshot(self):
        """Update the internal WordVectors from current nn.Embedding weights."""
        self.wv = self.cbow.to_word_vectors()
        return self.wv

    def reverse(self, y):
        """Snap each vector to nearest codebook entry by cosine similarity.

        Strips ObjectEncoding first, then compares nWhat content against
        the codebook. Stores recovered words/positions for reconstruct_text().

        Returns: content tensor [batch, nVec, embSize] with encoding zeroed.
        """
        content, positions, times = TheObjectEncoding.reverse(y.clone())
        batch = content.shape[0]
        nVec = content.shape[1]
        codebook = self._emb.weight.detach().to(y.device)
        words_list = self.wv.index_to_key
        nWhat = codebook.shape[1]
        recovered_words = [[] for _ in range(batch)]
        for b in range(batch):
            for v in range(nVec):
                vec = content[b, v, :nWhat]
                sims = F.cosine_similarity(vec.unsqueeze(0), codebook, dim=1)
                idx = sims.argmax().item()
                recovered_words[b].append(words_list[idx])
        self._recovered_words = recovered_words
        self._recovered_positions = positions
        self._recovered_times = times
        return content

    def reconstruct_text(self, join=False):
        """Return recovered words from the last reverse() call.

        Args:
            join: If True, return list of joined strings. If False, return
                  list of word lists (one per batch element).

        Returns:
            List of word lists or joined strings, one per batch element.
            Empty words (from zero-padded vectors) are stripped.
        """
        if not hasattr(self, '_recovered_words'):
            raise RuntimeError("reconstruct_text() called before reverse()")
        result = []
        for b_words in self._recovered_words:
            words = [w for w in b_words if w]
            if join:
                result.append(" ".join(words))
            else:
                result.append(words)
        return result

    def get_recovered_word(self, batch_idx, position):
        """Return a single recovered word from the last reverse() pass."""
        rw = getattr(self, '_recovered_words', None)
        if rw and batch_idx < len(rw) and position < len(rw[batch_idx]):
            return rw[batch_idx][position]
        return None

    def predict(self, vector):
        """Decode output vector(s) to nearest codebook word(s).

        Args:
            vector: [batch, 1, embeddingSize] or [batch, embeddingSize]
        Returns:
            List of predicted words (one per batch element).
        """
        if vector.dim() == 3:
            vector = vector.squeeze(1)
        codebook = self._emb.weight.detach().to(vector.device)
        words_list = self.wv.index_to_key
        nWhat = codebook.shape[1]
        predictions = []
        for b in range(vector.shape[0]):
            vec = vector[b, :nWhat]
            sims = F.cosine_similarity(vec.unsqueeze(0), codebook, dim=1)
            idx = sims.argmax().item()
            predictions.append(words_list[idx])
        return predictions

    def embed_token(self, word):
        """Look up a single word in the codebook, return its content vector.

        Returns:
            Tensor of shape [nWhat] (content dims only).
        """
        if word in self.cbow.key_to_index:
            idx = self.cbow.key_to_index[word]
        else:
            ch = word[0] if word else ' '
            idx = self.cbow.key_to_index.get(ch, 0)
        return self._emb.weight[idx].detach().clone()

    def get_space_embedding(self):
        """Return the codebook embedding for the space character ' '."""
        return self.embed_token(' ')

    def get_mask_embedding(self):
        """Return the [MASK] zero vector from the codebook."""
        return self._emb.weight[self.mask_token_idx].detach().clone()


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
        nDim is read from ``TheObjectEncoding`` (e.g. ``inputDim``, ``perceptDim``),
        not passed as a separate constructor argument.
      - ``nVectors``: codebook size, also read from ``TheObjectEncoding``
        (e.g. ``nPercepts``).  May differ from nActive (the active count);
        the factory validates ``nVectors >= nActive``.
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
    this to use raw target dimensions (no ObjectEncoding overhead on output).

    ``set_sigma(sigma)`` propagates exploration meta-parameters (1=explore, 0=suppress)
    from BasicModel down to all layers and VectorSets in this space.
    """
    name         = ""
    activation   = None
    processSymbols = False

    def __init__(self, inputShape, outputShape, nVectors, quantized=False, customVQ=True, reversible=False, processSymbols=False, reshape=False):
        super(Space, self).__init__()
        self.inputShape   = inputShape   # [nObjects, nDim] for input
        self.outputShape  = outputShape  # [nObjects, nDim] for output
        self.nVectors     = nVectors     # codebook size (total vectors in the space)
        self.nDim         = outputShape[1]  # content dimensionality (derived from outputShape)
        self.embeddingSize = TheObjectEncoding.getObjectEncodingSize(self.nDim)
        self.batch        = 0
        self.vectorSet    = nn.ModuleList()  # holds this space's VectorSet (accessed via self.vectors())
        self.quantized        = quantized
        self.customVQ     = customVQ
        self.reversible = reversible
        self.processSymbols = processSymbols
        self.reshape      = reshape
        self.params = []   # parameters for the optimizer (excludes temperature params)
        self.layers = nn.ModuleList()   # layer instances for paramUpdate() delegation

    def getEmbeddedIO(self):
        """Return (input_dim, output_dim) for reshape/validation.

        Without reshape: returns per-object embedding sizes (nDim + objectSize).
        With reshape: multiplies by object counts, giving the flattened vector
        width for unflatten after the layer pass.

        See also ``getLayerIO()`` which returns the widths the layer itself
        should be constructed with (may differ when the layer doubles output).
        """
        input  = TheObjectEncoding.getObjectEncodingSize(self.inputShape[1])
        output = TheObjectEncoding.getObjectEncodingSize(self.outputShape[1])
        if self.reshape:
            input  *= self.inputShape[0]
            output *= self.outputShape[0]
        return input, output
    def getLayerIO(self):
        """Return (input_dim, output_dim) for constructing this space's layer.

        By default, same as ``getEmbeddedIO()``.  Subclasses override when the
        layer's raw output width differs from the unflatten width (e.g.
        InvertiblePiLayer doubles its output).
        """
        return self.getEmbeddedIO()
    def lookup(self, x):
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.zeros([1,1, TheObjectEncoding.conceptDim], device=x.device), x[:,:,1:]], dim=2)
        output, index, _ = self.vectors().vq(x)
        #output[:,:,0:TheObjectEncoding.conceptDim] = output[:,:,0:TheObjectEncoding.conceptDim] * activation  # multiply the codebook vector by the activation
        return output
    def dereference(self, symbols):
        # we get [ batch x nConcepts x symbolEmbedding ],
        # and must compute [ batch x nConcepts x conceptEmbedding ]
        nActive = self.outputShape[0]
        assert list(symbols.shape) == [self.batch, nActive, TheObjectEncoding.getSymbolEncodingSize()], "Incorrect input size for dereference"
        input,_ = self.getEmbeddedIO()
        objects = torch.zeros(self.batch, nActive, self.embeddingSize, device=symbols.device)
        for b in range(self.batch):
            for s in range(nActive):
                x = self.lookup(symbols[b,s,:])
                objects[b,s,:] = x
        assert list(objects.shape) == [self.batch, nActive, self.embeddingSize], "Incorrect output size for dereference"
        return objects

    def stats(self, x):
        #codebookUse = self.vectors().codebookUse
        #message(f"{self.name} Codebook activation: { np.sum(self.vectors().codebookAct.get()) }")
        return
    def vectors(self):
        """Accessor for this space's VectorSet (first element of the ModuleList)."""
        return self.vectorSet[0]
    def createVectorSet(self, quantized=True):
        if quantized:
            self.vectorSet.append(VectorSet())
            # nActive (outputShape[0]) sets topk selection count;
            # self.nVectors (codebook size from TheObjectEncoding) sets codebook allocation.
            self.vectors().create(self.inputShape[0], self.outputShape[0], self.nDim, self.customVQ)
            self.vectors().addVectors(nVec=self.nVectors)
        else:
            vs = VectorSet()
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim, passThrough=True)
            self.vectorSet.append(vs)
    def forwardBegin(self, x):
        """Validate/reshape input at the start of the forward pass.
        When reshape=True, flattens [batch, nObj, dim] -> [batch, nObj*dim]."""
        self.batch = x.shape[0]
        if self.reshape:
            x = self.flatten(x, True)
        else:
            input, _ = self.getEmbeddedIO()
            assert list(x.shape) == [self.batch, self.inputShape[0], input]
        return x
    def forwardEnd(self, x):
        """Validate/unflatten output at the end of the forward pass.
        When reshape=True, unflattens [batch, nObj*dim] -> [batch, nObj, dim]."""
        if self.reshape:
            x = self.unflatten(x, True)
        else:
            _, output = self.getEmbeddedIO()
            assert list(x.shape)==[self.batch, self.outputShape[0], output], f"{self.__class__.__name__} forwardEnd: got {list(x.shape)}, expected {[self.batch, self.outputShape[0], output]}"
        return x
    def reverseBegin(self, y):
        """Validate/reshape at the start of the reverse pass (output-side)."""
        self.batch = y.shape[0]
        if self.reshape:
            y = self.flatten(y, False)
        else:
            _, output = self.getEmbeddedIO()
            assert list(y.shape) == [self.batch, self.outputShape[0], output]
        return y
    def reverseEnd(self, y):
        """Validate/unflatten at the end of the reverse pass (input-side)."""
        if self.reshape:
            y = self.unflatten(y, False)
        else:
            input, _ = self.getEmbeddedIO()
            assert list(y.shape) == [self.batch, self.inputShape[0], input]
        return y
    def flatten(self, x, forward=True):
        """Collapse [batch, nObj, dim] -> [batch, nObj*dim] for layer input."""
        input, output = self.getEmbeddedIO()
        if forward:
            x = x.reshape(self.batch, input)
        else:
            x = x.reshape(self.batch, output)
        return x
    def unflatten(self, y, forward=True):
        """Restore [batch, nObj*dim] -> [batch, nObj, dim] after layer output."""
        input, output = self.getEmbeddedIO()
        if forward:
            y = y.reshape(self.batch, self.outputShape[0], output // self.outputShape[0])
        else:
            y = y.reshape(self.batch, self.inputShape[0], input // self.inputShape[0])
        return y
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
    for representing syntactic constituent information in ObjectEncoding
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
    def __init__(self, nActiveInput, nActiveOutput, model_type="simple",
                 tokenizedInput=False, quantized=True, embedding_path=None, data=None,
                 lexer="word", ergodic=False, min_frequency=0.0,
                 neg_samples=64):
        # inputShape uses the data's native dimension (e.g. 784 for MNIST);
        # outputShape uses TheObjectEncoding.inputDim (set from XML nDim).
        dataDim     = data.getInputSize() if data is not None else TheObjectEncoding.inputDim
        inputShape  = [nActiveInput, dataDim]
        outputShape = [nActiveOutput, TheObjectEncoding.inputDim]
        nVectors    = TheObjectEncoding.nInput
        super(InputSpace, self).__init__(inputShape, outputShape, nVectors, quantized=quantized)
        self.data = data
        self.model_type = model_type
        self.lexer = lexer  # "word", "sentence", or "grammar" — selects .cfg file
        self.ergodic = ergodic
        self.min_frequency = float(min_frequency)
        if model_type == "embedding":
            source = data.train_texts
            vs = Embedding()
            vs.ergodic = ergodic
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim, embedding_path=embedding_path, source=source, min_frequency=min_frequency, neg_samples=neg_samples)
            self.nDim = vs.nDim
            # LM mode: the embedding determines content dims for all spaces
            # that process content vectors (input, percept, concept).
            # Symbol/output dims are set from XML (symbolDim=1, outputDim from data).
            TheObjectEncoding.setInputDim(vs.nDim)
            TheObjectEncoding.setPerceptDim(vs.nDim)
            TheObjectEncoding.setConceptDim(vs.nDim)
            self.outputShape = [self.outputShape[0], TheObjectEncoding.inputDim]
            self.vectorSet.append(vs)
            self.doc_spans = vs.doc_spans
            self.doc_sources = vs.doc_sources
        elif model_type == "passthrough":
            vs = VectorSet()
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim, passThrough=True)
            self.vectorSet.append(vs)
        elif model_type == "vq":
            vs = VectorSet()
            vs.create(self.inputShape[0], self.outputShape[0], self.nDim)
            self.vectorSet.append(vs)
        else:  # "simple"
            self.createVectorSet(quantized=self.quantized)
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = tokenizedInput
        # When positional encoding is present, apply a fixed random permutation
        # to the flattened output (nActive × embDim) so that dimensions from
        # different tokens are adjacent.  This lets the consecutive Givens chain
        # in downstream InvertiblePiLayer mix all token positions effectively.
        if False: # TheObjectEncoding.nWhere > 0:
            embDim = inputDim if inputDim is not None else TheObjectEncoding.inputDim
            flat_size = outputShape[0] * embDim
            perm = torch.randperm(flat_size)
            self.register_buffer('_output_perm', perm)
            self.register_buffer('_output_perm_inv', torch.argsort(perm))
            self._output_perm_nActive = outputShape[0]
            self._output_perm_embDim = embDim
        else:
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
            return torch.stack(inputBatch, dim=0).unsqueeze(1).to(TheDevice)
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
            self.batch = cached.shape[0]
            self.input = cached
            return self.input

        self.batch = input.shape[0]
        if not isinstance(self.vectors(), Embedding):
            assert list(input.shape) == [self.batch, self.inputShape[0], self.inputShape[1]]
        self.input = self.vectors().forward(input)

        # Object encoding: nWhere + nWhen stamped into reserved embedding slots
        if TheObjectEncoding.objectSize > 0:
            self.input = TheObjectEncoding.forward(self.input)

        _, output = self.getEmbeddedIO()
        assert list(self.input.shape) == [self.batch, self.outputShape[0], output]

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
        content_mask = torch.ones(embSize, dtype=torch.bool, device=embedded.device)
        # Preserve nWhere dims (indices [-4, -3] from end of embedding)
        if TheObjectEncoding.nWhere > 0:
            where_idx = np.add([embSize, embSize], PositionalEncoding.index)
            for wi in where_idx:
                if 0 <= wi < embSize:
                    content_mask[wi] = False
        # Preserve nWhen dims (indices [-2, -1] from end of embedding)
        if TheObjectEncoding.nWhen > 0:
            when_idx = np.add([embSize, embSize], TemporalEncoding.index)
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
        y = self.reverseBegin(y)
        self.input = self.vectors().reverse(y)
        self.reconstructed = self.input.detach()
        return self.input

    def reconstruct_text(self, join=False):
        """Delegates to Embedding.reconstruct_text()."""
        return self.vectors().reconstruct_text(join=join)

    # ------------------------------------------------------------------
    # Training policy — InputSpace decides WHEN, Embedding does HOW
    # ------------------------------------------------------------------

    def train_embeddings(self, words):
        """Run one CBOW/SBOW gradient step if words are available."""
        emb = self.vectors()
        if isinstance(emb, Embedding) and words:
            return emb.train_step(words)
        return None

    def _snapshot_embeddings(self):
        """Update WordVectors snapshot from current nn.Embedding weights."""
        emb = self.vectors()
        if isinstance(emb, Embedding):
            return emb.snapshot()
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
        if split == "train":
            inputData = self.data.train_input
            outputData = self.data.train_output
        elif split == "test":
            inputData = self.data.test_input
            outputData = self.data.test_output
        elif split == "validation":
            inputData = self.data.validation_input
            outputData = self.data.validation_output
        elif split == "runtime":
            inputData = self.data._runtime_input
            outputData = self.data._runtime_output
        else:
            raise ValueError(f"Unknown split: {split}")

        # ── ARIR state machine ──────────────────────────────────────
        if split == "runtime" and getattr(self.data, '_runtime_mode', None) == 'ARIR':
            return self._getBatch_arir(inputData, batchNum)

        # Use standard (non-masked) path when: no masked prediction configured,
        # or runtime split with no sentences staged (inference via runBatch).
        use_masked = (hasattr(self.data, 'masked_prediction')
                      and self.data.masked_prediction != 'NONE'
                      and split in getattr(self.data, '_lm_sentences', {}))
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
            return (inputTensor, outputTensor), batchNum + 1
        else:
            # Masked prediction: one sentence -> N masked examples.
            # Embed once, use that embedding for both targets and masked input.
            sentences = self.data._lm_sentences[split]
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
            masked_batch, _ = self.expand_masked(
                embedded, sentence, self.data.masked_prediction)

            # Hand masked embedding to forward() via cache — no re-embedding,
            # but gradient flows back through masked_batch → embedded → embedding weights
            self._cached_embedding = masked_batch

            return (inputTensor, targets), batchNum + 1

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
            counts = getattr(self, '_lex_span_counts', None)
            seed_len = counts[0] if counts and len(counts) > 0 else 1

            # Read byte offset from the lex pass
            offsets = getattr(self, '_lex_final_byte_offsets', None)
            self._arir_byte_offset = offsets[0] if offsets and len(offsets) > 0 else 0

            # Fill future positions (seed_len .. nVec-1) with space embeddings
            space_emb = self.get_space_embedding()
            for k in range(seed_len, nVec):
                self._arir_embedded[0, k, :nWhat] = space_emb[:nWhat]
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
            word = self.vectors().get_recovered_word(0, prev_cursor)

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
        """Stamp positional encoding at a buffer position via TheObjectEncoding.where."""
        if TheObjectEncoding.nWhere > 0:
            TheObjectEncoding.where.stamp(buf, 0, pos_idx, byte_off)

class PerceptualSpace(Space):
    """Transforms raw input vectors into percepts via a PiLayer.

    In the forward data flow: InputSpace -> **PerceptualSpace** -> ConceptualSpace.
    Uses a PiLayer (permutation-equivariant layer) to map input embeddings to
    perceptual embeddings, optionally followed by self-attention and VQ
    codebook quantization.

    When ``reversible=True``, uses InvertiblePiLayer whose forward
    interleaves (log_y, log_z) pairs, doubling the output.  In 3D mode
    the doubling is along the sequence axis; in 2D (reshape) mode,
    ``getLayerIO()`` halves nOutput so the 2x produces the correct
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

    def getLayerIO(self):
        """Layer I/O widths, accounting for InvertiblePiLayer's 2x doubling.

        When reshape=True and reversible=True, the layer's nOutput is halved
        so that InvertiblePiLayer's 2x interleaving produces exactly the
        unflatten width that getEmbeddedIO() returns.
        In 3D mode (reshape=False), the doubling is along the sequence axis
        and doesn't affect the per-object embedding width, so no adjustment.
        """
        input, output = self.getEmbeddedIO()
        if self.reshape and self.reversible:
            output = output // 2
        return input, output

    def __init__(self, nActiveInput, nActiveOutput, quantized=True, reversible=False, processSymbols=False, passThrough=False, reshape=False, ergodic=False, hasAttention=True, invertible=False, inputDim=None):
        inputShape  = [nActiveInput, inputDim if inputDim is not None else TheObjectEncoding.inputDim]
        outputShape = [nActiveOutput, TheObjectEncoding.perceptDim]
        nVectors    = TheObjectEncoding.nPercepts
        super(PerceptualSpace, self).__init__(inputShape, outputShape, nVectors, quantized=quantized, reversible=reversible, processSymbols=processSymbols, reshape=reshape)
        self.passThrough = passThrough
        self.ergodic = ergodic
        self.hasAttention = hasAttention
        self.invertible = invertible
        if passThrough:
            return
        input, output = self.getLayerIO()
        _, unflatOutput = self.getEmbeddedIO()
        self.attention = AttentionLayer(unflatOutput, unflatOutput)
        if invertible:
            use_naive = False  # invertible always uses SVD-based exact inverse
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
        else:
            use_naive = (2 * input != output)  # naive when dims don't match 2x
        if self.reversible:
            if invertible:
                self.pi  = InvertiblePiLayer(input, output, naive=use_naive, ergodic=ergodic)
                self.forwardPi, self.reversePi = self.pi.forward, self.pi.reverse
                self.params = self.pi.getParameters()
                self.layers = nn.ModuleList([self.pi])
                # InvertiblePiLayer doubles sequence in 3D or flat dim in 2D
                if not self.reshape:
                    self.outputShape = [2 * self.inputShape[0], self.outputShape[1]]
            else:
                self.pi1 = InvertiblePiLayer(input, output, naive=use_naive, ergodic=ergodic)
                self.pi2 = InvertiblePiLayer(input, output, naive=use_naive, ergodic=ergodic)
                self.forwardPi, self.reversePi = self.pi1.forward, self.pi2.reverse
                self.params = self.pi1.getParameters() + self.pi2.getParameters()
                self.layers = nn.ModuleList([self.pi1, self.pi2])
        else:
            self.pi        = PiLayer(input, output, ergodic=ergodic)
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
            self.batch = x.shape[0]
            return x
        x = self.forwardBegin(x)
        x = self.forwardPi(x)
        if self.hasAttention:
            x = self.attention.forward(x)
        if self.quantized:
            x  = self.vectors().forward(x)
        if self.processSymbols:
            # Collapse content dims to scalar activation, keep positional encoding
            encoding = x[:,:,-TheObjectEncoding.objectSize:]
            x = torch.norm( x[:,:,0:-TheObjectEncoding.objectSize], dim=2 ) / (2*self.outputShape[0])
            x = x.unsqueeze(-1)
            x = torch.concatenate((x, encoding), dim=2)
        self.percepts = self.forwardEnd(x)
        return self.percepts
    def reverse(self, y):
        """Manifesting: reconstruct input vectors from percepts via reverse PiLayer."""
        if self.passThrough:
            return y
        if self.reversible:
            y = self.reverseBegin(y)
            y = self.reversePi(y)
            y = self.reverseEnd(y)
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

    def __init__(self, nActiveInput, nActiveOutput, quantized=True, reversible=False, processSymbols=False, invertible=False, hasNorm=False, ergodic=False, reshape=False, hasAttention=False):
        inputShape  = [nActiveInput, TheObjectEncoding.perceptDim]
        outputShape = [nActiveOutput, TheObjectEncoding.conceptDim]
        nVectors    = TheObjectEncoding.nConcepts
        super(ConceptualSpace, self).__init__(inputShape, outputShape, nVectors, quantized=quantized, reversible=reversible, processSymbols=processSymbols, reshape=reshape)
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
        if reversible:
            if invertible:
                self.sigma = InvertibleSigmaLayer(input, output, ergodic=ergodic)
                self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                self.params = self.sigma.getParameters()
                self.layers = nn.ModuleList([self.sigma])
            else:
                self.sigma1 = InvertibleSigmaLayer(input, output, ergodic=ergodic)
                self.sigma2 = InvertibleSigmaLayer(input, output, ergodic=ergodic)
                # self.sigma1 = SigmaLayer(input, output, ergodic=ergodic)
                # self.sigma2 = SigmaLayer(output, input, ergodic=ergodic)
                self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.reverse
                self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
                self.layers = nn.ModuleList([self.sigma1, self.sigma2])
        else:
            self.sigma = SigmaLayer(input, output, ergodic=ergodic)
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
        x = self.forwardBegin(x)
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
            encoding = y[:,:,-TheObjectEncoding.objectSize:]
            y = torch.sum(y[:,:,0:-TheObjectEncoding.objectSize], dim=2) / (2*self.outputShape[0])
            y = y.unsqueeze(-1)
            y = torch.concatenate((y, encoding), dim=2)
        self.concepts = self.forwardEnd(y)
        return self.concepts
    def reverse(self, y):
        """Visualizing: reconstruct percepts from concepts via reverse SigmaLayer."""
        self.concepts = self.reverseBegin(y)
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
        self.concepts = self.reverseEnd(self.concepts)
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
    threshold        = 0       # discretization threshold (0 = disabled)
    serialActivation = False   # if True, only the top-1 activation is kept per batch
    symbols          = None

    def __init__(self, nActiveInput, nActiveOutput, reversible=False, conceptualSpace=None, processSymbols=False, passThrough=False, reshape=False):
        inputShape  = [nActiveInput, TheObjectEncoding.conceptDim]
        outputShape = [nActiveOutput, TheObjectEncoding.symbolDim]
        nVectors    = TheObjectEncoding.nSymbols
        super(SymbolicSpace, self).__init__( inputShape, outputShape, nVectors, customVQ=True, reversible=reversible, processSymbols=processSymbols, reshape=reshape)
        assert(inputShape[0] == outputShape[0]) # 1:1 mapping
        self.conceptualSpace = conceptualSpace
        self.passThrough = passThrough
        #self.mapping     = TODO(inputShape[1], nDim, soft=False)
        #self.createVectorSet()  # TODO: enable when symbolic codebook is ready
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
        self.symbols = self.forwardBegin(x)
        if not self.passThrough:
            if self.processSymbols:
                self.symbols = self.computeActivation(self.symbols)
            self.symbols = self.discretize(self.symbols)
        self.symbols = self.forwardEnd(self.symbols)
        if self.quantized:
            self.symbols  = self.vectors().forward(self.symbols)
        return self.symbols
    def reverse(self, y):
        """Interpretation: map symbols back to concept vectors (via codebook dereference)."""
        self.symbols = self.reverseBegin(y)
        if not self.passThrough:
            if self.processSymbols:
                self.symbols = self.conceptualSpace.dereference(self.symbols)
        self.symbols = self.reverseEnd(self.symbols)
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
    words = None

    def __init__(self, nActiveInput, nActiveOutput, reversible=False, conceptualSpace=None):
        inputShape  = [nActiveInput, TheObjectEncoding.symbolDim]
        outputShape = [nActiveOutput, TheObjectEncoding.wordDim]
        nVectors    = TheObjectEncoding.nWords
        super(SyntacticSpace, self).__init__( inputShape, outputShape, nVectors, customVQ=False, reversible=reversible, processSymbols=True)
        assert(inputShape[0] == outputShape[0]) # 1:1 mapping
        self.conceptualSpace = conceptualSpace
        #self.mapping     = TODO(inputShape[1], nDim, soft=False)
        #self.createVectorSet()
    def distance(self, x, y):
        return x == y
    def certainty(self, x):
        return x.T @ x
    def computeActivation(self, x):
        # we get [ batch x nConcepts x conceptEmbedding ],
        # and must compute [ batch x nConcepts x symbolEmbedding ]
        if x.size(-1) != TheObjectEncoding.symbolDim:
            activations = torch.norm( x[:,:,0:self.outputShape[1]] , dim=2)
            activations = activations.unsqueeze(2)
            activations = torch.concatenate((activations, x[:,:,self.inputShape[1]:]), dim=2)
        else:
            activations = x
        return activations
    # Naming
    def forward(self, x):
        self.symbols = self.forwardBegin(x)
        #self.symbols = self.computeActivation(self.symbols)
        self.symbols = self.forwardEnd(self.symbols)
        return self.symbols
    # Interpretation
    def reverse(self, y):
        self.symbols = self.reverseBegin(y)
        self.symbols = self.reverseEnd(self.symbols)
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
    (no ObjectEncoding overhead), since targets are not embedded.

    ``text_mode``: when enabled via ``set_text_mode()``, supports reconstructing
    text from symbolic vectors by snapping to the nearest codebook entry and
    recovering byte-offset positions.
    """
    name = "Outputs"
    text_mode = False
    def getEmbeddedIO(self):
        """Override: output uses raw target dims unless masked prediction needs encoding."""
        input = TheObjectEncoding.getObjectEncodingSize(self.inputShape[1])
        output = TheObjectEncoding.getObjectEncodingSize(self.outputShape[1]) if self.masked_prediction else self.outputShape[1]
        if self.reshape:
            input  *= self.inputShape[0]
            output *= self.outputShape[0]
        return input, output
    def __init__(self, nActiveInput, nActiveOutput, reversible=False, data=None, masked_prediction=False):
        symDim = TheObjectEncoding.symbolDim
        inputShape  = [nActiveInput, symDim]
        outputShape = [nActiveOutput, TheObjectEncoding.outputDim]
        nVectors    = TheObjectEncoding.nOutput
        self.masked_prediction = masked_prediction
        super(OutputSpace, self).__init__(inputShape, outputShape, nVectors, reshape=True)
        self.data = data
        self.text_mode = False
        input, output = self.getEmbeddedIO()
        if reversible:
            self.linear1 = LinearLayer(input, output)
            self.linear2 = LinearLayer(output, input)
            self.forwardLinear, self.reverseLinear = self.linear1.forward, self.linear2.forward
            #self.linear = InvertibleLinearLayer(input, output)
            #self.forwardLinear, self.reverseLinear = self.linear.forward, self.linear.reverse
        else:
            self.forwardLinear = LinearLayer(input, output)
        self.params = list(self.parameters())
        self.layers = nn.ModuleList([self.forwardLinear] if not reversible else [self.linear1, self.linear2])
    def getTestOutput(self):
        if self.data is None:
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
        y = super().forwardBegin(x)
        output = self.forwardLinear(y)
        output = self.forwardEnd(output)
        if self.quantized:
            self.output  = self.vectors().output(self.percepts)
        self.predicted = output.detach()
        return output
    def reverse(self, y):
        """Being acted upon: map output back to symbolic space via reverse LinearLayer."""
        y = self.reverseBegin(y)
        y = self.reverseLinear(y)
        output = self.reverseEnd(y)
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
            return torch.zeros(0, 1, embSize, device=embedded.device)
        # Extract the first N full word vectors (nWhat + nWhere + nWhen)
        targets = embedded[0, :N, :].clone()  # [N, embSize]
        if maskedPrediction == 'ARUS':
            targets = torch.zeros_like(targets)
        elif maskedPrediction == 'RARLM':
            targets = targets.flip(0)
        return targets.unsqueeze(1)  # [N, 1, embSize]
    # --- Text reconstruction from symbolic vectors ---
    def set_text_mode(self, input_space):
        """Enable text reconstruction by storing references from InputSpace.

        Args:
            input_space: An InputSpace instance that has lex, codebook, and
                         words attributes from the 'embedding' model_type path.
        """
        if input_space.model_type != "embedding":
            return
        self.text_mode = True
        vs = input_space.vectors()
        if hasattr(vs, 'codebook_weight'):
            # Embedding-backed path
            self._codebook = vs.codebook_weight.detach()
            self._words_list = vs.vocab_keys
        else:
            # Legacy VQ-backed path
            self._codebook = vs.vq.codebook
            self._words_list = vs.words
        self._embedding_size = vs.embeddingSize

    def reconstruct_text(self, vectors):
        """Reconstruct words and positions from symbolic vectors.

        Extracts nWhat and nWhere from vectors via ObjectEncoding.reverse(),
        snaps nWhat to nearest codebook entry by cosine similarity, and
        recovers byte-offset positions from nWhere.

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
            raise RuntimeError("reconstruct_text() called but text_mode is not enabled. "
                               "Call set_text_mode(input_space) first.")
        # 1. Strip positional/temporal encoding via ObjectEncoding.reverse()
        content, positions, times = TheObjectEncoding.reverse(vectors.clone())
        batch = content.shape[0]
        nVec = content.shape[1]
        codebook = self._codebook.to(content.device)
        words_list = self._words_list
        nWhat = self._embedding_size - TheObjectEncoding.objectSize
        cb_what = codebook[:, :nWhat]  # [codebookSize, nWhat]
        div_term = TheObjectEncoding.where.div_term

        # 2. For each vector, find nearest codebook entry by cosine sim on nWhat
        recovered_words = [[] for _ in range(batch)]
        recovered_offsets = [[] for _ in range(batch)]
        for b in range(batch):
            for v in range(nVec):
                vec = content[b, v, :nWhat]
                # Zero vectors are padding / EOS sentinels — emit \x00 so that
                # reconstruct_buffer (consecutive mode) can stop there cleanly.
                if vec.abs().sum().item() < 1e-8:
                    recovered_words[b].append("\x00")
                    recovered_offsets[b].append(None)
                    continue
                sims = F.cosine_similarity(vec.unsqueeze(0), cb_what, dim=1)
                idx = sims.argmax().item()
                recovered_words[b].append(words_list[idx])
                # 3. Decode nWhere: positions holds raw [sin, cos] values
                sin_val = positions[b, v, 0].item()
                cos_val = positions[b, v, 1].item()
                if abs(sin_val) < 1e-8 and abs(cos_val) < 1e-8:
                    recovered_offsets[b].append(None)  # no position -> consecutive
                else:
                    angle = math.atan2(sin_val, cos_val)
                    if angle < 0:
                        angle += 2 * math.pi
                    offset = angle / (div_term * div_term)
                    recovered_offsets[b].append(round(offset))
        return recovered_words, recovered_offsets

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
        words_list, offsets_list = self.reconstruct_text(vectors)
        results = []
        for b in range(len(words_list)):
            words = words_list[b]
            offsets = offsets_list[b]
            has_positions = any(o is not None for o in offsets)
            if has_positions:
                # Positioned write: place each word at its byte offset.
                # Null-initialized so spaces are only present when predicted.
                buf = bytearray(buf_size)
                for word, offset in zip(words, offsets):
                    if offset is None:
                        continue
                    encoded = word.encode('utf-8')
                    end = min(offset + len(encoded), buf_size)
                    buf[offset:end] = encoded[:end - offset]
                results.append(buf.rstrip(b'\x00').decode('utf-8', errors='replace'))
            else:
                # Consecutive: concatenate tokens directly — space is a predicted token.
                # Stop at the first \x00 (EOS/padding sentinel); trailing padding
                # slots that decode to \x00 are not included in the output.
                clean = []
                for w in words:
                    if w == "\x00":
                        break
                    clean.append(w)
                results.append("".join(clean))
        return results

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

    @staticmethod
    def load_config(config_path=None):
        """Load model settings from an XML config file.

        Parses sections recursively — e.g. ``<architecture><training><numEpochs>``
        becomes ``cfg["architecture"]["training"]["numEpochs"]``.  Leaf elements
        (no children) are auto-cast to bool/int/float/str.  Elements with
        children become nested dicts.

        Returns a dict of dicts; missing fields are filled by create_from_config()
        using defaults.xml.
        """
        import xml.etree.ElementTree as ET

        def _parse_element(elem):
            """Recursively parse an XML element into a dict or scalar.

            Leaf nodes with XML attributes return a dict with ``"_"`` holding
            the auto-cast text value and one key per attribute — e.g.
            ``<input use="train">hello|world</input>`` becomes
            ``{"_": "hello|world", "use": "train"}``.

            When multiple sibling elements share the same tag name (e.g. two
            ``<input>`` elements), they are aggregated into a list so no entry
            is silently overwritten.
            """
            children = list(elem)
            if not children:
                # Leaf node — auto-cast value
                text = elem.text.strip() if elem.text else ""
                if text.lower() in ("true", "false"):
                    val = text.lower() == "true"
                else:
                    try:
                        val = int(text)
                    except ValueError:
                        try:
                            val = float(text)
                        except ValueError:
                            val = text
                # Wrap in dict when XML attributes are present
                if elem.attrib:
                    return {"_": val, **elem.attrib}
                return val
            # Has children — recurse into a dict; aggregate repeated tags into lists
            d = {}
            for child in children:
                parsed = _parse_element(child)
                if child.tag in d:
                    existing = d[child.tag]
                    if isinstance(existing, list):
                        existing.append(parsed)
                    else:
                        d[child.tag] = [existing, parsed]
                else:
                    d[child.tag] = parsed
            return d

        if config_path is None:
            config_path = os.path.join(ProjectPaths.PROJECT_DIR, "model.xml")
        if not os.path.exists(config_path):
            return {}
        tree = ET.parse(config_path)
        root = tree.getroot()
        cfg = {}
        for section in root:
            cfg[section.tag] = _parse_element(section)
        return cfg

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

        When trainEmbeddings is False, embedding parameters are excluded
        from the optimizer regardless of path.
        """
        if getattr(self, 'ergodic', True):
            params = []
            for s in self.spaces:
                params.extend(s.getParameters())
        else:
            params = list(self.parameters())
        # Exclude embedding params unless network gradients are enabled (ARLM or BOTH)
        te = getattr(self, 'train_embeddings', 'NONE')
        if te not in ('ARLM', 'BOTH'):
            exclude = set()
            if hasattr(self, 'inputSpace') and isinstance(self.inputSpace.vectors(), Embedding):
                for p in self.inputSpace.vectors().embedding_parameters():
                    exclude.add(p.data_ptr())
            if exclude:
                params = [p for p in params if p.data_ptr() not in exclude]
        return optim.Adam(params, lr=lr)

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
        """Persist model weights, vocab, and ergodic state to disk."""
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {"state_dict": self.state_dict()}
        emb = self._get_embedding()
        if emb is not None:
            save_dict["vocab"] = list(emb.vocab_keys)
            save_dict["counts"] = emb.wv.counts
            save_dict["total_count"] = int(emb.wv.total_count)
            save_dict["pending_counts"] = dict(emb._pending_counts)
        torch.save(save_dict, path)
        print(f"[{self.name}] Weights saved to {path}")

    def load_weights(self, path=None, strict=False):
        """Load model weights from disk.

        If the saved file includes a vocab list, restores the Embedding
        vocabulary first so tensor shapes match exactly.  Falls back to
        legacy format (bare state_dict) for older weight files.
        """
        if path is None:
            path = os.path.join(ProjectPaths.OUTPUT_DIR, "weights.ckpt")
        if not os.path.exists(path):
            print(f"[{self.name}] No checkpoint at {path}, starting fresh")
            return False
        saved = torch.load(path, map_location=TheDevice, weights_only=False)

        # Support both new format {"state_dict": ..., "vocab": ...}
        # and legacy format (bare state_dict)
        if isinstance(saved, dict) and "state_dict" in saved:
            state = saved["state_dict"]
            saved_vocab = saved.get("vocab")
            saved_counts = saved.get("counts")
            saved_total = saved.get("total_count", 0)
            saved_pending = saved.get("pending_counts", {})
        else:
            state = saved
            saved_vocab = None
            saved_counts = None
            saved_total = 0
            saved_pending = {}

        # Restore vocab on the Embedding so shapes match
        emb = self._get_embedding()
        if emb is not None and saved_vocab is not None:
            self._restore_vocab(emb, saved_vocab,
                                counts=saved_counts, total_count=saved_total,
                                pending_counts=saved_pending)

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
        dim = emb._emb.weight.shape[1]
        device = emb._emb.weight.device
        vocab_size = len(saved_vocab)

        # Rebuild word mappings
        emb.cbow.index_to_key = list(saved_vocab)
        emb.cbow.key_to_index = {w: i for i, w in enumerate(saved_vocab)}
        emb.wv.index_to_key = list(saved_vocab)
        emb.wv.key_to_index = {w: i for i, w in enumerate(saved_vocab)}
        emb.wv._vectors = np.zeros((vocab_size, dim), dtype=np.float32)
        emb.wv.counts = (np.asarray(counts, dtype=np.int64) if counts is not None
                         else np.zeros(vocab_size, dtype=np.int64))
        emb.wv.total_count = np.int64(total_count)
        emb._pending_counts = dict(pending_counts) if pending_counts else {}
        emb.wv._normed = None

        # Resize nn.Embedding to match
        emb.cbow.embeddings = nn.Embedding(vocab_size, dim, device=device)
        emb._emb = emb.cbow.embeddings

        # Update mask token index
        if "[MASK]" in emb.cbow.key_to_index:
            emb.mask_token_idx = emb.cbow.key_to_index["[MASK]"]

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

    @staticmethod
    def _bytes_to_text(tensor):
        """Decode a byte tensor (or padded int8 tensor) to a string."""
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
        # Use reconstruct_text() for lex-based models (embedding vectors, not bytes)
        use_lex_recon = (self.inputSpace.model_type == "embedding" and
                         self.inputSpace.vectors().get_recovered_word(0, 0) is not None)
        if use_lex_recon:
            recon_text_list = self.inputSpace.reconstruct_text(join=True)
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

        Loads defaults from defaults.xml, overlays model-specific config,
        then creates the model and optionally loads saved weights.
        """
        # Store for runTrials() re-creation
        self._config_path = config_path
        self._config_data = data

        # Load defaults, then overlay model-specific config
        defaults_path = os.path.join(ProjectPaths.DATA_DIR, "defaults.xml")
        defaults = self.load_config(defaults_path)
        cfg = self.load_config(config_path)

        def _deep_merge(base, overlay):
            """Recursively merge overlay into base (overlay wins on conflicts)."""
            merged = dict(base)
            for k, v in overlay.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = _deep_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        for section in defaults:
            if section not in cfg:
                cfg[section] = defaults[section]
            else:
                cfg[section] = _deep_merge(defaults[section], cfg[section])

        BasicModelFactory.validate_config(cfg)

        arch = cfg["architecture"]
        trn = arch.get("training", {})
        dat = arch.get("data", {})

        def _t(key, default=None):
            """Look up a training param from arch.training."""
            return trn.get(key, default)

        def _d(key, default=None):
            """Look up a data param from arch.data."""
            return dat.get(key, default)

        # Caller overrides XML; XML overrides defaults
        if model_type is None:
            model_type = arch.get("modelType", "simple")
        # embedding_path: set in config via <embeddingPath>; absence means dynamic vocab
        embedding_path = _t("embeddingPath", None)
        if embedding_path is not None:
            embedding_path = self._resolve_artifact_path(embedding_path)

        # ObjectEncoding setup — positional/temporal encoding config from InputSpace
        gsp = BasicModelFactory.get_space_param
        TheObjectEncoding.nWhere = gsp(cfg, "InputSpace", "nWhere")
        TheObjectEncoding.nWhen = gsp(cfg, "InputSpace", "nWhen")
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

        # Codebook sizes (nVectors from XML — must be in defaults.xml or model XML)
        TheObjectEncoding.nInput    = gsp(cfg, "InputSpace", "nVectors")
        TheObjectEncoding.nPercepts = gsp(cfg, "PerceptualSpace", "nVectors")
        TheObjectEncoding.nConcepts = gsp(cfg, "ConceptualSpace", "nVectors")
        TheObjectEncoding.nSymbols  = gsp(cfg, "SymbolicSpace", "nVectors")
        TheObjectEncoding.nWords    = gsp(cfg, "SyntacticSpace", "nVectors")
        TheObjectEncoding.nOutput   = gsp(cfg, "OutputSpace", "nVectors")

        # Validate: codebook size must be >= active count for every space
        for space_name, n_attr in [
            ("InputSpace",      "nInput"),
            ("PerceptualSpace", "nPercepts"),
            ("ConceptualSpace", "nConcepts"),
            ("SymbolicSpace",   "nSymbols"),
            ("SyntacticSpace",  "nWords"),
            ("OutputSpace",     "nOutput"),
        ]:
            nVectors = getattr(TheObjectEncoding, n_attr)
            nActive = gsp(cfg, space_name, "nActive")
            if nVectors > 0:
                assert nVectors >= nActive, \
                    f"{space_name}: nVectors ({nVectors}) must be >= nActive ({nActive})"

        TheObjectEncoding.nObjects = 0  # reset for re-creation
        TheObjectEncoding.computeNObjects()

        # Content dimensions from XML — set on TheObjectEncoding, not on Space constructors
        TheObjectEncoding.setInputDim(gsp(cfg, "InputSpace", "nDim"))
        TheObjectEncoding.setPerceptDim(gsp(cfg, "PerceptualSpace", "nDim"))
        TheObjectEncoding.setConceptDim(gsp(cfg, "ConceptualSpace", "nDim"))
        TheObjectEncoding.setSymbolDim(gsp(cfg, "SymbolicSpace", "nDim"))
        TheObjectEncoding.setWordDim(gsp(cfg, "SyntacticSpace", "nDim"))
        outDim = gsp(cfg, "OutputSpace", "nDim")
        TheObjectEncoding.setOutputDim(outDim)

        self.create(
            nInput=gsp(cfg, "InputSpace", "nActive"),
            nPercepts=gsp(cfg, "PerceptualSpace", "nActive"),
            nConcepts=gsp(cfg, "ConceptualSpace", "nActive"),
            nSymbols=gsp(cfg, "SymbolicSpace", "nActive"),
            nWords=gsp(cfg, "SyntacticSpace", "nActive"),
            nOutput=gsp(cfg, "OutputSpace", "nActive"),
            reversible=arch.get("reconstruct", "NONE").upper() != "NONE",
            perceptPassThrough=gsp(cfg, "PerceptualSpace", "passThrough"),
            symbolPassThrough=gsp(cfg, "SymbolicSpace", "passThrough"),
            perceptPrototypes=gsp(cfg, "PerceptualSpace", "nVectors"),
            conceptPrototypes=gsp(cfg, "ConceptualSpace", "nVectors"),
            ergodic=arch["ergodic"],
            certainty=arch["certainty"],
            quantized=gsp(cfg, "InputSpace", "quantized"),
            perceptQuantized=gsp(cfg, "PerceptualSpace", "quantized"),
            conceptQuantized=gsp(cfg, "ConceptualSpace", "quantized"),
            invertible=gsp(cfg, "PerceptualSpace", "invertible"),
            hasNorm=gsp(cfg, "ConceptualSpace", "hasNorm"),
            conceptualOrder=arch["conceptualOrder"],
            symbolicOrder=arch["symbolicOrder"],
            processSymbols=arch["processSymbols"],
            reshape=arch["reshape"],
            perceptHasAttention=gsp(cfg, "PerceptualSpace", "hasAttention"),
            conceptHasAttention=gsp(cfg, "ConceptualSpace", "hasAttention"),
            model_type=model_type, data=data, embedding_path=embedding_path,
            lexer=gsp(cfg, "InputSpace", "lexer"),
            recon_ratio=_t("reconRatio", 0.5),
            masked_prediction=arch.get("maskedPrediction", "NONE").upper(),
            min_frequency=_d("minFrequency", 0.0),
            neg_samples=_t("negSamples", 64),
            reconstruct=arch.get("reconstruct", "none"),
        )
        # train (legacy: trainEmbeddings): NONE=frozen, CBOW=SBOW only, ARLM=network only, BOTH=SBOW+network
        # Check flat arch for either name first (explicit override), then nested defaults
        if "train" in arch and not isinstance(arch["train"], dict):
            te = arch["train"]
        elif "trainEmbeddings" in arch and not isinstance(arch["trainEmbeddings"], dict):
            te = arch["trainEmbeddings"]
        else:
            te = _t("train", "NONE")
        if te is True:
            te = "BOTH"
        elif te is False:
            te = "NONE"
        self.train_embeddings = te.upper()
        if self.train_embeddings in ("ARLM", "BOTH") and isinstance(self.inputSpace.vectors(), Embedding):
            emb_params = self.inputSpace.vectors().embedding_parameters()
            self.inputSpace.params = self.inputSpace.params + emb_params
        # Auto-load weights if configured
        wcfg = cfg.get("weights", {})
        if _t("autoload", wcfg.get("autoload", True)):
            wpath = _t("weightsPath", wcfg.get("path", "weights.ckpt"))
            wpath = self._resolve_artifact_path(wpath)
            self.load_weights(wpath)
        # Inference config
        self.max_response_length = arch.get("maxResponseLength", 64)
        return cfg

    def create(self, nInput, nPercepts, nConcepts, nSymbols, nWords=16, nOutput=32,
               reversible=True, perceptPassThrough=False, symbolPassThrough=False,
               perceptPrototypes=0, conceptPrototypes=0,
               ergodic=False, certainty=False, quantized=False,
               perceptQuantized=None, conceptQuantized=None,
               invertible=False, hasNorm=False,
               conceptualOrder=1, symbolicOrder=1, processSymbols=False,
               reshape=False,
               perceptHasAttention=True, conceptHasAttention=False,
               model_type="simple", data=None, embedding_path=None,
               lexer="word", recon_ratio=0.5,
               masked_prediction='NONE', min_frequency=0.0,
               neg_samples=64, reconstruct='NONE'):
        """Build the full space hierarchy from architecture parameters.

        Args:
            nInput/nPercepts/nConcepts/nSymbols/nOutput: object counts per space.
            nWords: object count for the SyntacticSpace (used when symbolicOrder >= 1).
            reversible: enable the reverse (reconstruction) pipeline.
            perceptPassThrough: make PerceptualSpace an identity.
            symbolPassThrough: make SymbolicSpace an identity (passes conceptDim through).
            ergodic: use ergodic (temperature-based) parameter updates in layers.
            certainty: use CertaintyWeightedCrossEntropy loss.
            quantized: enable VQ codebook quantization in spaces.
            invertible: use InvertiblePiLayer/InvertibleSigmaLayer (exact inverse)
                in PerceptualSpace and ConceptualSpace.
            reshape: flatten [batch, nObj, dim] before layers (required when
                     input/output object counts differ).
            conceptualOrder: number of extra Percept->Concept->Symbol cycles.
            symbolicOrder: number of extra Syntax->Symbol cycles.
            model_type: "simple", "embedding", "passthrough", or "vq".
        """
        self.spaces = []  # reset — prevent stale accumulation from prior create() calls
        self.lexer            = lexer  # "word", "sentence", or "grammar" — selects .cfg file
        self.reversible      = reversible
        self.reconstruct     = reconstruct.lower()
        self.nInput           = nInput
        self.nOutput          = nOutput
        self.nPercepts        = nPercepts
        self.nConcepts        = nConcepts
        self.nSymbols         = nSymbols
        assert nSymbols >= nOutput, (
            f"nSymbols ({nSymbols}) must be >= nOutput ({nOutput}). "
            f"The symbolic bottleneck must have at least as many symbols as outputs."
        )
        self.nOutputSymbols   = nOutput
        self.nReconSymbols    = max(0, nSymbols - nOutput)
        self.recon_symbols    = None
        self.nWords           = nWords
        self.data             = data
        self.model_type       = model_type
        self.embedding_path   = embedding_path
        self.perceptPassThrough = perceptPassThrough
        self.symbolPassThrough  = symbolPassThrough
        self.perceptPrototypes  = perceptPrototypes
        self.conceptPrototypes  = conceptPrototypes
        self.ergodic          = ergodic
        self.certainty        = certainty
        self.min_frequency    = float(min_frequency)
        self.neg_samples      = int(neg_samples)
        self.quantized        = quantized
        self.perceptQuantized = perceptQuantized if perceptQuantized is not None else quantized
        self.conceptQuantized = conceptQuantized if conceptQuantized is not None else quantized
        self.invertible       = invertible
        self.hasNorm          = hasNorm
        self.conceptualOrder  = conceptualOrder
        self.symbolicOrder    = symbolicOrder
        self.processSymbols   = processSymbols
        self.reshape          = reshape
        self.perceptHasAttention = perceptHasAttention
        self.conceptHasAttention = conceptHasAttention
        self.recon_ratio      = recon_ratio
        self.masked_prediction = masked_prediction
        if data is not None and hasattr(data, 'masked_prediction') and data.masked_prediction != 'NONE':
            data.masked_prediction = masked_prediction
        # nOutputSymbols tracks total symbol count fed to OutputSpace.
        # Starts with only the output-destined symbols (not reconstruction symbols).
        # It grows as higher-order cycles (conceptualOrder, symbolicOrder) append symbols.
        nOutputSymbols = self.nOutputSymbols
        self.inputSpace      = InputSpace(self.nInput, self.nInput,
                                           model_type=model_type, data=data,
                                           embedding_path=embedding_path,
                                           quantized=self.quantized,
                                           lexer=self.lexer,
                                           ergodic=self.ergodic,
                                           min_frequency=self.min_frequency,
                                           neg_samples=self.neg_samples)
        # Convert masked-word string labels to embedding vectors now that
        # the Embedding vocabulary is available.
        if data is not None and hasattr(data, '_lm_labels') and data._lm_labels is not None:
            embedding = self.inputSpace.vectorSet[0] if self.inputSpace.vectorSet else None
            if embedding is not None and hasattr(embedding, 'cbow'):
                data.prepare_lm_targets(embedding)
                # Move new targets to device
                data.toDevice()
        self.perceptualSpace = PerceptualSpace(self.inputSpace.outputShape[0], self.nPercepts,
                                               reversible=reversible,
                                               quantized=self.perceptQuantized,
                                               passThrough=perceptPassThrough,
                                               reshape=reshape,
                                               ergodic=self.ergodic,
                                               hasAttention=self.perceptHasAttention,
                                               invertible=self.invertible)
        self.conceptualSpace = ConceptualSpace(self.perceptualSpace.outputShape[0], self.nConcepts,
                                               reversible=reversible,
                                               invertible=self.invertible,
                                               hasNorm=self.hasNorm,
                                               ergodic=self.ergodic,
                                               quantized=self.conceptQuantized,
                                               reshape=reshape,
                                               hasAttention=self.conceptHasAttention)
        if symbolPassThrough:
            # passThrough means data flows at conceptDim — set symbolDim accordingly
            TheObjectEncoding.setSymbolDim(TheObjectEncoding.conceptDim)

        self.symbolicSpace   = SymbolicSpace(self.conceptualSpace.outputShape[0], self.nSymbols,
                                              reversible=reversible,
                                              conceptualSpace=self.conceptualSpace,
                                              processSymbols=self.processSymbols,
                                              passThrough=symbolPassThrough,
                                              reshape=reshape)
        self.spaces.extend([self.inputSpace, self.perceptualSpace, self.conceptualSpace, self.symbolicSpace])

        if self.conceptualOrder == 2:
            self.perceptualSpace2 = PerceptualSpace(self.conceptualSpace.outputShape[0],self.nPercepts,
                                                    reversible = reversible,
                                                    inputDim = TheObjectEncoding.symbolDim)
            self.conceptualSpace2 = ConceptualSpace(self.perceptualSpace2.outputShape[0], self.nConcepts,
                                                    reversible = reversible)
            self.symbolicSpace2   = SymbolicSpace(self.conceptualSpace2.outputShape[0], self.nSymbols,
                                                reversible = reversible,
                                                conceptualSpace = self.conceptualSpace2,
                                                processSymbols = self.processSymbols)
            nOutputSymbols += (self.conceptualOrder - 1) * self.nSymbols
            self.spaces.extend([self.perceptualSpace2, self.conceptualSpace2, self.symbolicSpace2])

        if self.symbolicOrder == 2:
            # SyntacticSpace3 receives the full symbol tensor (nSymbols objects)
            self.syntacticSpace3 = SyntacticSpace(self.nSymbols, self.nSymbols,
                                                reversible = reversible)
            self.symbolicSpace3  = SymbolicSpace(self.syntacticSpace3.outputShape[0], self.nSymbols,
                                                reversible = reversible,
                                                reshape = reshape)
            nOutputSymbols += (self.symbolicOrder - 1) * self.nSymbols
            self.spaces.extend([self.syntacticSpace3, self.symbolicSpace3])
            
        self.nTotalOutputSymbols = nOutputSymbols
        self.outputSpace     = OutputSpace(nOutputSymbols, self.nOutput,
                                           reversible=reversible, data=data,
                                           masked_prediction=(masked_prediction != 'NONE'))
        self.spaces.extend([self.outputSpace])
        self.inputSpace.outputSpace = self.outputSpace

        # The output dimensionality of the input layer must be equal to the output dimensionality of the perceptual layer, since the conceptual layer operates on both.
        #assert self.inputSpace.outputShape[1] == self.perceptualSpace2.outputShape[1] # inputDim == perceptDim
        # The input dimensionality of the symbolic layer must be equal to the input dimensionality of the perceptual layer, since they both operate on the output of the conceptual layer.
        #assert self.symbolicSpace.inputShape[1] == self.perceptualSpace2.inputShape[1] == self.conceptualSpace.outputShape[1]#  conceptDim = conceptDim
        # The output shape of the symbolic space is equal to the input shape of the output space
        #assert self.symbolicSpace.outputShape[1] == self.outputSpace.inputShape[1] # these are in conceptual space, or symbolic space if symbols emit objectSize symbols (processSymbols == True)

        self.to(TheDevice)

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

        # Stage sentence text so getBatch() enters the masked prediction path
        # when the model uses masked prediction (ARLM/MLM). This ensures
        # inference sees the same masking pattern as training.
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'
        sentences = [text] if masked_pred else None
        with torch.no_grad(), TheData.runtime_batch(
            [TheData.stringTensor(text)], sentences=sentences
        ):
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

        with torch.no_grad(), TheData.runtime_batch(
            [TheData.stringTensor(text)], [[0]], mode='ARIR'
        ):
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
        TheObjectEncoding.when.increment(batch)
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
        optimizer         = self.getOptimizer(lr=lr)

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

            outErr, inErr, allOut, lastIn = self.runEpoch(optimizer=optimizer, batchSize=batchSize, split="train")
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
    
    def _getLossFn(self):
        """Return (outputLossFn, inputLossFn) based on model config.

        outputLossFn: used for the forward pass (prediction vs target).
        inputLossFn:  used for the reverse pass (reconstruction loss), always MSE.
        """
        if self.certainty:
            return CertaintyWeightedCrossEntropy(), nn.MSELoss()
        elif self.nOutput <= 2:
            # Binary classification or scalar regression — MSE is appropriate
            return nn.MSELoss(), nn.MSELoss()
        elif self.conceptualOrder > 0 or self.symbolicOrder > 0:
            return nn.MSELoss(), nn.MSELoss()
        else:
            return nn.CrossEntropyLoss(), nn.MSELoss()

    BatchResult = namedtuple('BatchResult', [
        'outputPred', 'symbols', 'lossOut', 'lossIn', 'inputPred', 'forwardInput',
    ])

    def runBatch(self, train=True, batchNum=0, batchSize=10, split="train",
                 optimizer=None, criterionOutput=None, criterionInput=None):
        """Run a single batch: forward pass, loss, and (if training) backward + step.

        Args:
            train: whether to compute gradients and update parameters.
            batchNum: opaque cursor returned by getBatch for the next batch.
            batchSize: number of examples per batch.
            split: "train", "test", or "validation".
            optimizer: pre-built optimizer (required when train=True).
            criterionOutput: loss function for the output prediction.
            criterionInput: loss function for the input reconstruction.

        Returns:
            (BatchResult, nextBatchNum) on success, or (None, batchNum) when
            the dataset is exhausted.
        """
        batch, batchNum = self.inputSpace.getBatch(batchNum, batchSize, split)
        if batch is None:
            return None, batchNum

        inputTensor, outputTensor = batch
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'
        inference_only = not train and criterionOutput is None
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

        outputPred = outputDataPred.squeeze()
        output     = outputTensor.squeeze()
        lossOut    = criterionOutput(outputPred, output)

        # ARUS: suppress output loss (unsupervised — no target signal)
        if hasattr(self, 'masked_prediction') and self.masked_prediction == 'ARUS':
            lossOut = torch.tensor(0.0, device=outputPred.device)

        if self.reversible:
            inputDataPred, inputPred = self.reverse(symbols, outputDataPred)
            lossIn = criterionInput(inputPred, forwardInput.squeeze())
            rr = self.recon_ratio
            totalLoss = (1 - rr) * lossOut + rr * lossIn
        else:
            inputDataPred = None
            lossIn = torch.tensor(0.0, device=outputPred.device)
            totalLoss = lossOut

        if train:
            totalLoss.backward()
            if self.ergodic:
                self.paramUpdate()
            optimizer.step()

            # Re-zero the [MASK] embedding after each optimizer step
            if masked_pred and hasattr(self.inputSpace.vectors(), 'mask_token_idx'):
                with torch.no_grad():
                    emb = self.inputSpace.vectors()
                    emb.codebook_weight[emb.mask_token_idx] = 0.0

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
        criterionOutput, criterionInput = self._getLossFn()
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
                    criterionOutput=criterionOutput,
                    criterionInput=criterionInput,
                )
                if result is None:
                    break

                self.outputSpace.putBatch(result)

                if training and batchIdx % 100 == 0:
                    print(f"  batch {batchIdx}", end="\r", flush=True)

                # CBOW training (post-batch, needs batchIdx for sentence lookup)
                if training:
                    te = getattr(self, 'train_embeddings', 'NONE')
                    if masked_pred and te in ('CBOW', 'BOTH'):
                        emb = self.inputSpace.vectors()
                        if isinstance(emb, Embedding):
                            sentences = self.inputSpace.data._lm_sentences[split]
                            if batchIdx < len(sentences):
                                sentence = sentences[batchIdx]
                                tokens = emb.lex_buffer(sentence, 0)
                                words = [t['text'] for t in tokens
                                         if t['category'] == 'WORD']
                                self.inputSpace.train_embeddings(words)

                outErr = result.lossOut.item()
                inErr = result.lossIn.item() if self.reversible else 0

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
        All parameters must be in defaults.xml or model XML; raises KeyError if missing.
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
        cfg = BaseModel.load_config(config_path)
        arch = cfg.get("architecture", {})
        dat = arch.get("data", {})
        trn = arch.get("training", {})

        dataset = dat.get("dataset")
        TheData.load(dataset,
                     num_shards=dat.get("numShards", 1),
                     max_docs=dat.get("maxDocs", 10000),
                     shard_dir=dat.get("shardDir"),
                     dat=dat)

        m = BasicModel()
        # Store config refs so runTrials can call create_from_config per trial
        m._config_path = config_path
        m._config_data = TheData
        message(f"Device: {TheDevice}")

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
            wpath = _t("weightsPath", cfg.get("weights", {}).get("path", "weights.ckpt"))
            wpath = m._resolve_artifact_path(wpath)
            m.save_weights(wpath)

        return [(m.name, m.rCorrect, m)]

def test():
    """Smoke test: verify encodings and run the XOR config end-to-end."""
    PositionalEncoding.test()
    TemporalEncoding.test()
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
    args = parser.parse_args()

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
