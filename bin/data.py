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
        on the same corpus that produced the word embeddings.
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

        # Store raw strings — tensorized lazily in prepInput()
        self.train_input  = list(train_tokens)
        self.validation_input  = list(validation_tokens)
        self.test_input  = list(test_tokens)

        # For masked LM, labels are target word strings — store for later
        # conversion to embedding vectors by prepare_lm_targets().
        # For non-LM tasks, labels are numeric lists.
        if not train_labels:
            # Masked prediction mode: targets computed at runtime
            # Raw sentences are already in train_input/validation_input/test_input
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
            # Pad with NULL (0) — input buffer is variable-length, null-terminated
            tensor = F.pad(tensor, (0, self.inputLength - tensor.size(0)), 'constant', 0)
        return tensor

TheData = Data()
