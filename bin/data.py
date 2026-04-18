"""Dataset loading, FineWeb-EDU streaming, and data utilities.

Data class loads, preprocesses, and serves train/validation/test splits for
MNIST (numeric), XOR (toy text), Rotten Tomatoes (real text), inline XML,
and FineWeb-EDU shards.

FineWeb-EDU streaming downloads and reads parquet shards from HuggingFace,
shared by both embed.py (embedding training) and BasicModel.py (model training).
"""

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

import util
from util import ProjectPaths

TheDevice = util.TheDevice


# ---------------------------------------------------------------------------
# FineWeb-EDU streaming
# ---------------------------------------------------------------------------

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822

def _shard_filename(index):
    return f"shard_{index:05d}.parquet"

def download_shard(index, data_dir):
    """Download a single shard if not already present. Returns filepath or None."""
    filename = _shard_filename(index)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath
    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return filepath
        except (requests.RequestException, IOError) as e:
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts: {e}")
                return None
    return None

def get_shard_paths(data_dir, num_shards=1, random_select=False):
    """Ensure shards are downloaded and return their file paths."""
    os.makedirs(data_dir, exist_ok=True)
    if random_select and num_shards <= MAX_SHARD:
        import random as _rng
        indices = sorted(_rng.sample(range(MAX_SHARD + 1), num_shards))
        print(f"  Random shard indices: {indices}")
    else:
        indices = list(range(min(num_shards, MAX_SHARD + 1)))
    paths = []
    for i in indices:
        path = download_shard(i, data_dir)
        if path:
            paths.append(path)
    return paths

def iter_documents(shard_paths, max_docs=None):
    """Yield text documents from parquet shard files."""
    count = 0
    for filepath in shard_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            for text in texts:
                yield text
                count += 1
                if max_docs is not None and count >= max_docs:
                    return


class SentenceStreamDataset(IterableDataset):
    """Yields B-wide batches from B contiguous streams of the dataset.

    The ordered ``inputs`` sequence is split into ``num_streams`` contiguous
    slabs of equal length ``L = len(inputs) // num_streams`` (any tail
    shorter than ``num_streams`` is dropped so every step is rectangular).
    At step ``t`` the dataset yields a batch of ``num_streams`` items, where
    position ``b`` is ``inputs[b * L + t]`` -- so batch row ``b`` sees a
    coherent document-order stream across all steps.

    ``inputs`` may be a list (of strings or per-item tensors) or a pre-
    stacked tensor of shape ``[N, ...]``. ``outputs`` is a parallel
    structure; pass ``None`` for unsupervised / inference paths. When
    supplied, each step yields ``(input_batch, output_batch)``; otherwise
    just ``input_batch``.

    Designed to be wrapped in ``DataLoader(ds, batch_size=None, ...)`` so
    the dataset self-batches and the DataLoader only provides async
    prefetch.
    """

    def __init__(self, inputs, num_streams, outputs=None):
        n = (inputs.shape[0] if isinstance(inputs, torch.Tensor)
             else len(inputs))
        if n == 0:
            raise ValueError("SentenceStreamDataset: inputs is empty")
        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")
        if num_streams > n:
            raise ValueError(
                f"num_streams={num_streams} exceeds available items={n}"
            )
        self.inputs = inputs
        self.outputs = outputs
        self.num_streams = num_streams
        self.stream_length = n // num_streams

    def __len__(self):
        return self.stream_length

    @staticmethod
    def _slice(data, indices):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data[indices]
        return [data[i] for i in indices]

    def __iter__(self):
        L = self.stream_length
        B = self.num_streams
        for t in range(L):
            indices = [b * L + t for b in range(B)]
            inp = self._slice(self.inputs, indices)
            if self.outputs is None:
                yield inp
            else:
                yield inp, self._slice(self.outputs, indices)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

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
        self.reconstructed_input  = None  # filled after reverse pass (buffer strings)
        self.reconstructed_output = None  # filled after output reverse
        # Global min/max for data scaling (computed by _compute_ranges)
        self.input_min  = None
        self.input_max  = None
        self.output_min = None
        self.output_max = None

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
        self._compute_ranges()
        self.toDevice()
    def toDevice(self):
        """Move pre-stacked data tensors to TheDevice and pre-shape them.

        Applies the trailing-dim unsqueeze that ``prepInput`` / ``prepOutput``
        used to apply per-batch so the hot loop avoids both the unsqueeze
        and a redundant ``.to()`` call.

        List-of-tensor splits (e.g. masked-LM placeholder outputs, per-
        sentence targets) are deliberately left on CPU: the streaming
        ``DataLoader`` pickles slices of these lists across worker
        processes, and live CUDA tensors can't cross that boundary.
        ``prepInput`` / ``prepOutput`` still call ``.to(TheDevice.get())``
        after stacking in the main process, so the per-batch transfer
        cost stays O(batch) rather than O(dataset).
        """
        for attr in ("train_input", "train_output",
                      "test_input", "test_output",
                      "validation_input", "validation_output"):
            v = getattr(self, attr)
            if isinstance(v, torch.Tensor):
                v = v.to(TheDevice.get())
                if v.ndim == 2:          # [N, D] -> [N, D, 1]
                    v = v.unsqueeze(2)
                setattr(self, attr, v)
            elif isinstance(v, list):
                # List-of-tensors must live on CPU: the streaming
                # DataLoader pickles slices of these lists across worker
                # processes, and live accelerator tensors (CUDA / MPS)
                # cannot be reduced for cross-process sharing. The
                # default torch device may be an accelerator in this
                # environment, so tensors created via plain
                # ``torch.zeros(...)`` in ``processLM`` may already be
                # off-CPU -- move them back here, in place, to preserve
                # list identity for any external references.
                for i, t in enumerate(v):
                    if isinstance(t, torch.Tensor) and t.device.type != "cpu":
                        v[i] = t.cpu()

    def _compute_ranges(self):
        """Compute global scalar min/max for input and output data.

        Called at the end of each load*() method, before toDevice().
        These values are used by InputSpace (scale to [0,1]) and
        OutputSpace (rescale from [-1,1] to original range).
        """
        if isinstance(self.train_input, torch.Tensor):
            self.input_min = self.train_input.min().item()
            self.input_max = self.train_input.max().item()
        else:
            # Text data: embedded, L2-normalized -> elements in [-1, 1]
            self.input_min = -1.0
            self.input_max = 1.0
        if isinstance(self.train_output, torch.Tensor):
            self.output_min = self.train_output.min().item()
            self.output_max = self.train_output.max().item()
        elif isinstance(self.train_output, list) and len(self.train_output) > 0:
            if isinstance(self.train_output[0], torch.Tensor):
                stacked = torch.stack(self.train_output)
                self.output_min = stacked.min().item()
                self.output_max = stacked.max().item()

    def normalize(self, x, which="input"):
        """Normalize x into the canonical range for the given role.

        Args:
            x: tensor to normalize.
            which: "input" scales [input_min, input_max] -> [-1, 1].
                   "output" scales [output_min, output_max] -> [-1, 1].
        """
        if which == "input":
            if self.input_min is None or self.input_max is None or self.input_max == self.input_min:
                return x
            return (x - self.input_min) / (self.input_max - self.input_min) * 2 - 1
        else:
            if self.output_min is None or self.output_max is None or self.output_max == self.output_min:
                return x
            return (x - self.output_min) / (self.output_max - self.output_min) * 2 - 1

    def denormalize(self, x, which="input"):
        """Reverse normalize(): map from canonical range back to data range.

        Args:
            x: tensor to denormalize.
            which: "input" scales [-1, 1] -> [input_min, input_max].
                   "output" scales [-1, 1] -> [output_min, output_max].
        """
        if which == "input":
            if self.input_min is None or self.input_max is None or self.input_max == self.input_min:
                return x
            return (x + 1) / 2 * (self.input_max - self.input_min) + self.input_min
        else:
            if self.output_min is None or self.output_max is None or self.output_max == self.output_min:
                return x
            return (x + 1) / 2 * (self.output_max - self.output_min) + self.output_min

    @contextmanager
    def runtime_batch(self, inputs, outputs=None, mode=None):
        """Stage transient inference data into train_input/train_output.

        Saves and restores the previous contents so training data is
        not corrupted (though in practice, inference and training never
        overlap).

        Args:
            inputs: list of strings for the runtime batch.
            outputs: optional list of output targets.
            mode: optional inference mode ('ARIR', 'ARLM', or None).
        """
        saved_input = self.train_input
        saved_output = self.train_output
        self.train_input = inputs
        self.train_output = outputs
        self._runtime_mode = mode
        # Validate tensor inputs are within the expected range
        if isinstance(inputs, torch.Tensor) and self.input_min is not None:
            xmin = inputs.min().item()
            xmax = inputs.max().item()
            if xmin < self.input_min - 1e-2 or xmax > self.input_max + 1e-2:
                import warnings
                warnings.warn(
                    f"Inference input range [{xmin:.4f}, {xmax:.4f}] "
                    f"outside training range [{self.input_min}, {self.input_max}]")
        try:
            yield
        finally:
            self.train_input = saved_input
            self.train_output = saved_output
            self._runtime_mode = None

    def pushInput(self, token):
        """Append a token to train_input[0] during inference.

        For text-based ARLM inference, train_input is a list containing
        a single string.  Appends the predicted token for autoregressive
        generation.
        """
        if not isinstance(self.train_input, list) or len(self.train_input) == 0:
            raise TypeError(f"pushInput: unsupported train_input type {type(self.train_input)}")
        elem = self.train_input[0]
        if isinstance(elem, str):
            self.train_input[0] = elem + token
        elif isinstance(elem, torch.Tensor):
            chars = [chr(int(b) & 0xFF) for b in elem.tolist()]
            existing = "".join(chars).rstrip("\x00")
            self.train_input[0] = self.stringTensor(existing + token)
        else:
            raise TypeError(f"pushInput: unsupported element type {type(elem)}")

    def pushOutput(self, token):
        """Append a target to train_output[0] during inference.

        For text-based data, appends the token to the first output sentence.
        Not used during inference but provided for completeness.
        """
        if isinstance(self.train_output, list) and len(self.train_output) > 0:
            self.train_output[0] = self.train_output[0] + " " + token
        else:
            raise TypeError(f"pushOutput: unsupported train_output type {type(self.train_output)}")

    def shuffle(self):
        rand_indx = torch.randperm(len(self.train_output))
        if isinstance(self.train_input, list):
            self.train_input  = [self.train_input[i] for i in rand_indx]
            self.train_output = [self.train_output[i] for i in rand_indx]
        else:
            self.train_input = self.train_input[rand_indx]
            self.train_output = self.train_output[rand_indx]
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
        on the same corpus that produced the word embeddings. Documents
        are kept in canonical shard order so ``SentenceStreamDataset`` can
        produce contiguous streams.
        """
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
    def processLM(self, data):
        train_tokens      = data["train"]["text"]
        train_labels      = data["train"]["label"]
        validation_tokens = data["validation"]["text"]
        validation_labels = data["validation"]["label"]
        test_tokens       = data["test"]["text"]
        test_labels       = data["test"]["label"]

        self.combinedTokens = train_tokens + validation_tokens + test_tokens
        self.combinedTokens = list(set(self.combinedTokens))

        # Store raw strings -- tensorized lazily in prepInput()
        self.train_input  = list(train_tokens)
        self.validation_input  = list(validation_tokens)
        self.test_input  = list(test_tokens)

        # For masked LM, labels are target word strings -- store for later
        # conversion to embedding vectors by prepare_lm_targets().
        # For non-LM tasks, labels are numeric lists.
        if not train_labels:
            # Masked prediction mode: targets computed at runtime
            # Raw sentences are already in train_input/validation_input/test_input
            self.masked_prediction = 'MLM'
            # Sentinel outputs for OutputSpace sizing -- shape must match embedding dim
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

    def tokenize(self, TheLanguageModel):
        self.train_input      = TheLanguageModel.tokenize(self.train_input)
        self.validation_input = TheLanguageModel.tokenize(self.validation_input)
        self.test_input       = TheLanguageModel.tokenize(self.test_input)

    def data_loader(self, split, num_streams, num_workers=0,
                    prefetch_factor=None, pin_memory=False):
        """Return a DataLoader over the given split as B contiguous streams.

        ``split`` selects one of ``train`` / ``validation`` / ``test``.
        ``num_streams`` is the effective batch size. It is capped at the
        number of available items so callers never need to worry about
        tiny eval splits (``batchSize=10`` on a 4-item XOR test set yields
        one batch of 4 rows). ``num_workers`` and ``prefetch_factor`` are
        forwarded to ``torch.utils.data.DataLoader``. The dataset self-
        batches (yields ``(inputs, outputs)`` per step), so
        ``batch_size=None`` on the DataLoader.
        """
        inputs = getattr(self, f"{split}_input")
        outputs = getattr(self, f"{split}_output", None)
        n = (inputs.shape[0] if isinstance(inputs, torch.Tensor)
             else len(inputs))
        if n == 0:
            raise RuntimeError(
                f"data_loader: {split}_input is empty -- "
                "load() before building a loader"
            )
        streams = max(1, min(num_streams, n))
        ds = SentenceStreamDataset(inputs, num_streams=streams,
                                   outputs=outputs)

        # Live CUDA tensors can't cross DataLoader worker process
        # boundaries (torch.multiprocessing reductions fail with
        # "Attempted to send CUDA tensor received from another process").
        # ``toDevice()`` moves both stacked tensor splits and list-of-
        # tensor splits onto CUDA, so we have to check both shapes. If
        # anything the worker would yield is CUDA-resident, collapse to
        # in-process batching and skip pin_memory (which only applies to
        # host tensors).
        def _has_cuda(x):
            if x is None:
                return False
            if isinstance(x, torch.Tensor):
                return x.is_cuda
            if isinstance(x, (list, tuple)) and len(x) > 0:
                head = x[0]
                return isinstance(head, torch.Tensor) and head.is_cuda
            return False

        on_cuda = _has_cuda(inputs) or _has_cuda(outputs)
        if on_cuda:
            if num_workers > 0:
                print(
                    f"[data_loader] {split} data is on CUDA; forcing "
                    f"num_workers=0 (requested {num_workers}) to avoid "
                    "cross-process CUDA tensor sharing."
                )
                num_workers = 0
            pin_memory = False

        kwargs = {"batch_size": None, "num_workers": num_workers,
                  "pin_memory": pin_memory}
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(ds, **kwargs)
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
        if self.train_input and isinstance(self.train_input[0], str):
            return self.inputLength
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
            embedding: An Embedding instance with pretrain.key_to_index and _emb weights.
        """
        if self._lm_labels is None:
            return
        weights = embedding.wv._vectors  # (vocab_size, vec_size)
        vec_size = weights.shape[1]
        vocab_size = weights.shape[0]

        def words_to_embeddings(word_list):
            targets = []
            for word in word_list:
                w = word.lower()
                if w in embedding.pretrain.key_to_index:
                    idx = embedding.pretrain.key_to_index[w]
                    # These tensors are labels, not a differentiable target path.
                    # Snapshot the current embedding row so later optimizer steps do
                    # not silently move the targets.
                    v = F.normalize(weights[idx].detach().clone(), p=2, dim=0)
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
            # Pad with NULL (0) -- input buffer is variable-length, null-terminated
            tensor = F.pad(tensor, (0, self.inputLength - tensor.size(0)), 'constant', 0)
        return tensor

TheData = Data()
