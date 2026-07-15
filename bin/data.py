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
    """Download a single shard if not already present. Returns filepath or None.

    Retries up to 5 times with exponential backoff. Uses a ``.tmp``
    file rename to avoid leaving a partial parquet on disk on failure.
    Returns ``None`` after all retries fail (logs to stdout).
    """
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
    """Ensure shards are downloaded and return their file paths.

    ``random_select`` picks ``num_shards`` distinct indices from the
    valid range for variety across runs; otherwise indices are
    contiguous from 0. Creates ``data_dir`` if missing. Skips any
    index whose download fails.
    """
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
    """Yield text documents from parquet shard files.

    Reads each shard row-group at a time to avoid materializing the
    entire shard in memory. Stops after ``max_docs`` documents when
    set; otherwise yields everything.
    """
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

    Two cursor modes share the same ``next_tick`` interface:

    * **Byte cursor** (``slab_bytes`` set): per-row byte cursor that
      walks each document one ``slab_bytes``-byte slab at a time. A
      document longer than one slab is consumed across multiple ticks
      for the same row; documents are concatenated end-to-end within a
      row's stream. ``hard_eos[b]`` marks ticks that complete a
      document (host-side bool, no GPU sync). Used for AR text byte
      training (``runEpoch`` cursor branch).

    * **Trial cursor** (``slab_bytes`` is ``None``): each tick yields
      one batch of ``num_streams`` trials (one per row); the cursor's
      step counter advances by 1. ``hard_eos = [True] * B`` every tick
      because each trial completes immediately. Used for non-AR /
      numeric / non-byte data (MNIST, XOR with labels, tomatoes), per
      the brick-vectorization handoff §8e ("data cursor aligns with
      the trial" for non-AR paths).

    The legacy ``__iter__`` path is preserved for callers that hold a
    ``DataLoader`` directly (existing tests).
    """

    def __init__(self, inputs, num_streams, outputs=None, slab_bytes=None):
        """Split ``inputs`` into ``num_streams`` contiguous slabs.

        ``slab_bytes`` selects byte-cursor mode (UTF-8 walk over text
        rows) when set; ``None`` selects trial-cursor mode (one tick =
        one batch of trials). Tail items past ``num_streams *
        stream_length`` are dropped so every step is rectangular.
        """
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
        # Cursor state. Two regimes share ``next_tick``:
        #   * Byte cursor (slab_bytes set): per-row (doc_idx, offset)
        #     walks UTF-8 byte streams. Used by AR text byte training.
        #   * Trial cursor (slab_bytes=None): one tick = one batch of
        #     trials. ``_trial_step`` is the timestep cursor; reaches
        #     ``stream_length`` when the epoch is done.
        self.slab_bytes = slab_bytes
        if slab_bytes is not None:
            if slab_bytes < 1:
                raise ValueError(f"slab_bytes must be >= 1, got {slab_bytes}")
            self.doc_idx = [b * self.stream_length for b in range(num_streams)]
            self.offset = [0] * num_streams
            self._encoded_cache = {}  # doc_idx -> bytes, populated lazily
        else:
            self._trial_step = 0

    def __len__(self):
        """Number of timesteps in one epoch (per-stream length)."""
        return self.stream_length

    @staticmethod
    def _slice(data, indices):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data[indices]
        return [data[i] for i in indices]

    def __iter__(self):
        """Back-compat trial iterator: yield one batch per timestep.

        Iterates the trial-cursor logic sharded across DataLoader
        workers so ``num_workers>0`` doesn't replay the same batches.
        Canonical access is ``next_tick``; ``__iter__`` exists for tests
        that hold a DataLoader directly.
        """
        # Back-compat path for callers that hold the ``DataLoader``
        # directly (existing tests, e.g. test_stream_smoke). The
        # canonical dispatch under the brick-vectorization handoff
        # §8e is ``next_tick`` (cursor universal); the runEpoch outer
        # loop never iterates the DataLoader. ``__iter__`` here mirrors
        # the trial-cursor's per-tick logic, sharded across workers.
        L = self.stream_length
        B = self.num_streams
        # Shard the timestep range across DataLoader workers so num_workers>0
        # doesn't replay the same batches num_workers times. Each worker gets
        # every Nth step starting at its worker_id.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, stride = 0, 1
        else:
            start, stride = worker_info.id, worker_info.num_workers
        for t in range(start, L, stride):
            indices = [b * L + t for b in range(B)]
            inp = self._slice(self.inputs, indices)
            if self.outputs is None:
                yield inp
            else:
                yield inp, self._slice(self.outputs, indices)

    # ------------------------------------------------------------------
    # Rolling-cursor mode
    # ------------------------------------------------------------------

    def _doc_bytes(self, doc_idx):
        """Return UTF-8 bytes for ``self.inputs[doc_idx]``, cached.

        Strings are encoded once and cached so repeated tick reads of the
        same long doc avoid re-encoding. Cache is small in practice
        because each row holds at most one open doc at a time.
        """
        cached = self._encoded_cache.get(doc_idx)
        if cached is not None:
            return cached
        item = self.inputs[doc_idx]
        if isinstance(item, str):
            data = item.encode('utf-8')
        elif isinstance(item, (bytes, bytearray)):
            data = bytes(item)
        elif isinstance(item, torch.Tensor):
            # Treat 1D byte tensors as raw bytes (e.g. pre-tokenized inputs).
            data = bytes(item.to(torch.uint8).tolist())
        else:
            raise TypeError(
                f"SentenceStreamDataset cursor: unsupported input type "
                f"{type(item).__name__}; expected str/bytes/Tensor"
            )
        self._encoded_cache[doc_idx] = data
        return data

    def all_done(self):
        """True iff the cursor has consumed all assigned data.

        Byte-cursor mode: every row has crossed its assigned doc window.
        Trial-cursor mode: the trial step has reached the stream length.
        """
        if self.slab_bytes is None:
            return self._trial_step >= self.stream_length
        return all(
            self.doc_idx[b] >= (b + 1) * self.stream_length
            for b in range(self.num_streams)
        )

    def next_tick(self):
        """Read one tick of input across all rows.

        Returns ``(input, output, hard_eos)`` where:

        * **Byte-cursor mode** (``slab_bytes`` set): ``input`` is a
          ``[num_streams, slab_bytes]`` ``uint8`` CPU tensor (bytes past
          the per-row advance are zero -- existing NULL-pad contract);
          ``output`` is ``None`` (AR is self-supervised on the input
          bytes); ``hard_eos[b]`` is True iff the slab consumed the rest
          of row b's current document. Concatenating the per-row
          populated prefixes across consecutive ticks for a single row
          reproduces the original document bytes exactly.

        * **Trial-cursor mode** (``slab_bytes`` is ``None``): ``input``
          and ``output`` are the row b's trial at the current step --
          one item per row, so a list of length ``num_streams`` (or a
          stacked tensor when the underlying inputs are tensors).
          ``hard_eos = [True] * num_streams`` every tick because each
          trial is its own document.

        The brick-vectorization handoff §8e unified non-AR data flow
        (numeric / non-byte text) through this trial-cursor mode, so
        the outer doc-streaming loop in ``runEpoch`` can use one
        ``next_tick`` interface regardless of the data path.
        """
        if self.slab_bytes is None:
            return self._trial_next_tick()
        return self._byte_next_tick()

    def _byte_next_tick(self):
        """Read one byte-cursor tick of ``slab_bytes`` bytes per row.

        Returns ``(slab[B, slab_bytes] uint8, None, hard_eos[B])``.
        Rows that finished their assigned window emit pure NULL pads.
        Rows that crossed a doc boundary set hard_eos True and advance
        to the next doc; mutates per-row ``doc_idx`` / ``offset`` state.
        """
        slab = torch.zeros(
            self.num_streams, self.slab_bytes, dtype=torch.uint8)
        hard_eos = [False] * self.num_streams
        for b in range(self.num_streams):
            window_end = (b + 1) * self.stream_length
            if self.doc_idx[b] >= window_end:
                # Row exhausted its assigned window; emit pure NULLs.
                continue
            doc = self._doc_bytes(self.doc_idx[b])
            remaining = len(doc) - self.offset[b]
            advance = min(self.slab_bytes, remaining)
            if advance > 0:
                # Copy the slab range into a writable bytearray so
                # frombuffer doesn't pin the immutable bytes object.
                buf = bytearray(doc[self.offset[b]:self.offset[b] + advance])
                slab[b, :advance] = torch.frombuffer(buf, dtype=torch.uint8)
            self.offset[b] += advance
            if self.offset[b] >= len(doc):
                hard_eos[b] = True
                # Drop the cached encoded bytes for the doc we just finished.
                self._encoded_cache.pop(self.doc_idx[b], None)
                self.doc_idx[b] += 1
                self.offset[b] = 0
        # Byte mode: AR is self-supervised, no separate output.
        return slab, None, hard_eos

    def _trial_next_tick(self):
        """Read one trial-cursor tick (one trial per row).

        Returns ``(input, output, hard_eos=[True]*B)`` -- every trial is
        atomic so hard_eos fires every tick. Past-end ticks emit empty
        batches with all-True hard_eos so callers can drain cleanly.
        Advances ``self._trial_step``.
        """
        if self._trial_step >= self.stream_length:
            # Defensive: callers should check all_done() first; emit an
            # empty batch with all-True hard_eos so the outer loop's
            # post-tick housekeeping cleans up cleanly.
            empty_inp = self._slice(self.inputs, [])
            empty_out = (self._slice(self.outputs, [])
                         if self.outputs is not None else None)
            return empty_inp, empty_out, [True] * self.num_streams
        L = self.stream_length
        indices = [b * L + self._trial_step for b in range(self.num_streams)]
        inp = self._slice(self.inputs, indices)
        out = (self._slice(self.outputs, indices)
               if self.outputs is not None else None)
        self._trial_step += 1
        # Each trial is its own atomic unit -> hard_eos True every row.
        return inp, out, [True] * self.num_streams

    def reset_cursor(self):
        """Rewind the cursor for a fresh epoch.

        Byte-cursor mode: every row's (doc_idx, offset) returns to the
        start of its assigned window; the byte cache clears.

        Trial-cursor mode: the step counter returns to 0.
        """
        if self.slab_bytes is None:
            self._trial_step = 0
            return
        self.doc_idx = [
            b * self.stream_length for b in range(self.num_streams)]
        self.offset = [0] * self.num_streams
        self._encoded_cache.clear()

    def progress(self):
        """Return cursor progress as a fraction in ``[0.0, 1.0]``.

        Byte-cursor mode: average doc-fraction across all streams.
        Each stream owns ``stream_length`` docs starting at its window
        offset; ``doc_idx[b] - b * stream_length`` is the count of
        docs row b has fully consumed. We add a fractional contribution
        from any in-progress doc using ``offset[b] / len(doc)`` so the
        report ticks smoothly within a long doc rather than jumping at
        each doc boundary.

        Trial-cursor mode: ``_trial_step / stream_length`` (each tick
        is one trial per row).

        ``runEpoch`` calls this once per tick to display
        ``batch = N (..., X.X%)``. Cheap: the in-progress doc's bytes
        are already cached by the cursor's last read.
        """
        if self.slab_bytes is None:
            L = self.stream_length
            if L <= 0:
                return 1.0
            return min(1.0, max(0.0, self._trial_step / L))
        total_docs = self.num_streams * self.stream_length
        if total_docs <= 0:
            return 1.0
        consumed = 0.0
        for b in range(self.num_streams):
            row_start = b * self.stream_length
            consumed += (self.doc_idx[b] - row_start)
            # Fractional credit for the in-progress doc
            window_end = (b + 1) * self.stream_length
            if self.doc_idx[b] < window_end and self.offset[b] > 0:
                cur_doc = self._encoded_cache.get(self.doc_idx[b])
                if cur_doc is not None and len(cur_doc) > 0:
                    consumed += min(1.0, self.offset[b] / len(cur_doc))
        return min(1.0, max(0.0, consumed / total_docs))


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
        """Initialize empty train / validation / test splits.

        Populates per-split input / output lists, the combined-tokens
        vocabulary slot, reconstruction buffers, and the global min/max
        scaling values (filled later by ``_compute_ranges``).
        """
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
        # True  => measured presence features -> [0,1] percept hypercube.
        # False => signed embedding vectors (text) -> keep [-1,1].
        # Set per-dataset by _compute_ranges (doc/Spaces.md#percept-live-path).
        self.input_presence = True
        # Self-supervised corpora still carry placeholder OutputSpace tensors
        # for batching, but those placeholders are not supervised targets.
        self.has_supervised_outputs = False
        self.source_manifest = None
        # ``_runtime_mode`` retired 2026-05-14 alongside ARIR; runtime
        # callers no longer pass a mode token.  Field kept on the class
        # at None so legacy ``getattr(data, '_runtime_mode', None)``
        # reads stay well-defined; remove next release.
        self._runtime_mode = None

    @property
    def nInput(self):
        """Number of input features, derived from train_input shape.

        Tensor inputs: dim 1; list inputs (LM): list length.
        """
        if isinstance(self.train_input, torch.Tensor):
            return self.train_input.shape[1]
        return len(self.train_input)  # list of strings for LM

    @property
    def inputDim(self):
        """Dimensionality per input feature, derived from train_input shape.

        Returns ``train_input.shape[2]`` for 3D tensor data, else 1.
        """
        if isinstance(self.train_input, torch.Tensor) and self.train_input.ndim >= 3:
            return self.train_input.shape[2]
        return 1

    @property
    def nOutput(self):
        """Number of output features, derived from train_output shape.

        Tensor outputs: dim 1; list-of-tensor outputs: length of the
        first item; empty list: 0.
        """
        if isinstance(self.train_output, torch.Tensor):
            return self.train_output.shape[1]
        if isinstance(self.train_output, list) and len(self.train_output) > 0:
            return len(self.train_output[0])
        return 0

    @property
    def outputDim(self):
        """Dimensionality per output feature, derived from train_output shape.

        Returns ``train_output.shape[2]`` for 3D tensor data, else 1.
        """
        if isinstance(self.train_output, torch.Tensor) and self.train_output.ndim >= 3:
            return self.train_output.shape[2]
        return 1

    def load(self, dataset, num_shards=1, max_docs=10000, shard_dir=None,
             dat=None, random_shards=False, max_tokens=None):
        """Dispatch to the per-dataset loader, then compute ranges + move to device.

        ``dataset`` selects ``mnist`` / ``xor`` / ``tomatoes`` / ``text``
        (FineWeb-EDU shards) / ``inline`` (XML payload). Mutates the
        train / validation / test attributes and the min/max scaling
        values; finally moves tensors to ``TheDevice``.
        """
        self.has_supervised_outputs = False
        self.source_manifest = {"dataset": str(dataset)}
        if dataset == "mnist":
            self.loadMNist()
        if dataset == "xor":
            self.loadXOR()
        if dataset == "phrases":
            self.loadPhrases()
        if dataset == "substitution":
            self.loadSubstitution()
        if dataset == "queries":
            self.loadQueries()
        if dataset == "sequences":
            self.loadSequences()
        if dataset == "tomatoes":
            self.loadTomatoes()
        if dataset == "text":
            self.loadShards(num_shards, max_docs, shard_dir,
                            random_shards=random_shards,
                            max_tokens=max_tokens)
        if dataset == "inline":
            self.loadInline(dat or {})
        if dataset == "mnist":
            self.has_supervised_outputs = True
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
            self.input_presence = True   # measured features -> [0,1] presence
        else:
            # Text data: embedded, L2-normalized -> elements in [-1, 1]. These
            # are SIGNED embedding vectors (concept-ish), not one-sided
            # presence, so they keep the signed [-1,1] canonical range until
            # the percept lexicon itself moves to [0,1]
            # (doc/Spaces.md#percept-live-path -- the half-done move breaks the
            # invertible embedding reconstruction chain, e.g. XOR_exact).
            self.input_min = -1.0
            self.input_max = 1.0
            self.input_presence = False
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
            which: "input" scales [input_min, input_max] -> [0, 1] when
                   ``input_presence`` (measured presence on the positive unit
                   hypercube: 0 = absent / nothing, 1 = present / everything;
                   one-sided, the percept antipode is the complement 1-x -- see
                   doc/Spaces.md#percept-live-path). Restores the [0,1] target the
                   _compute_ranges docstring always declared. Signed text
                   embeddings (``input_presence`` False) keep the [-1,1] map
                   (the prior ``* 2 - 1``) until the lexicon moves to [0,1].
                   "output" scales [output_min, output_max] -> [-1, 1].
        """
        if which == "input":
            if self.input_min is None or self.input_max is None or self.input_max == self.input_min:
                return x
            u = (x - self.input_min) / (self.input_max - self.input_min)   # -> [0,1]
            return u if getattr(self, "input_presence", True) else u * 2 - 1
        else:
            if self.output_min is None or self.output_max is None or self.output_max == self.output_min:
                return x
            return (x - self.output_min) / (self.output_max - self.output_min) * 2 - 1

    def denormalize(self, x, which="input"):
        """Reverse normalize(): map from canonical range back to data range.

        Args:
            x: tensor to denormalize.
            which: "input" inverts the input normalize above -- [0,1] ->
                   [input_min, input_max] for presence data, [-1,1] -> range
                   for signed embeddings.
                   "output" scales [-1, 1] -> [output_min, output_max].
        """
        if which == "input":
            if self.input_min is None or self.input_max is None or self.input_max == self.input_min:
                return x
            u = x if getattr(self, "input_presence", True) else (x + 1) / 2
            return u * (self.input_max - self.input_min) + self.input_min
        else:
            if self.output_min is None or self.output_max is None or self.output_max == self.output_min:
                return x
            return (x + 1) / 2 * (self.output_max - self.output_min) + self.output_min

    @contextmanager
    def runtime_batch(self, inputs, outputs=None, mode=None):
        """Stage transient inference data into ``train_input`` / ``train_output``.

        Saves and restores the previous contents so training data is
        not corrupted (training and inference never overlap in
        practice).  ``mode`` is accepted for back-compat and ignored;
        the legacy ARIR-mode signalling retired 2026-05-14.
        """
        del mode  # retained for signature back-compat only
        saved_input = self.train_input
        saved_output = self.train_output
        self.train_input = inputs
        self.train_output = outputs
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

    def pushInput(self, token):
        """Append a token to train_input[0] during inference.

        For text-based AR inference, train_input is a list containing
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
        """Permute training inputs and outputs in unison.

        Uses a single random permutation so input / output pairs stay
        aligned. Works for both list and tensor data; validation /
        test splits are untouched.
        """
        rand_indx = torch.randperm(len(self.train_output))
        if isinstance(self.train_input, list):
            self.train_input  = [self.train_input[i] for i in rand_indx]
            self.train_output = [self.train_output[i] for i in rand_indx]
        else:
            self.train_input = self.train_input[rand_indx]
            self.train_output = self.train_output[rand_indx]
    def loadMNist(self):
        """Read MNIST train/test CSVs into per-pixel float tensors.

        Normalizes pixels by mean and std of the train split, builds
        one-hot label tensors, and mirrors the test split into the
        validation slot. Sets ``inputLength = 28 * 28``.
        """
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
        """Load a 4-sentence XOR toy dataset and hand off to ``processLM``.

        All three splits share the same four sentences and ``[0,1,1,0]``
        XOR labels. Used as the smallest end-to-end smoke test for the
        text pipeline.
        """
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
    def loadPhrases(self):
        """Tiny DET+noun / ADJ+noun phrase set for the idea-decoder round-trip
        (doc/old/2026-06-20-idea-decoder.md). Two-word phrases with shared
        vocabulary (DET the/a, ADJ black/red, nouns cat/dog/world) so the
        codebook learns the words and the grammar learns the DET/ADJ roles --
        the constituency the default xor 'sentences' (2 independent XOR
        features) lack. Label = DET-phrase 0 / ADJ-phrase 1 (incidental; the
        reconstruction loss is what drives the decode round-trip)."""
        phrases = ["the cat", "the dog", "the world", "a cat", "a dog",
                   "black cat", "black dog", "red cat"]
        labels = [[0], [0], [0], [0], [0], [1], [1], [1]]
        data = {
            "train":      {"text": phrases, "label": labels},
            "validation": {"text": phrases, "label": labels},
            "test":       {"text": phrases, "label": labels},
        }
        self.train_input       = data["train"]["text"]
        self.train_output      = data["train"]["label"]
        self.validation_input  = data["validation"]["text"]
        self.validation_output = data["validation"]["label"]
        self.test_input        = data["test"]["text"]
        self.test_output       = data["test"]["label"]
        self.processLM(data)

    def loadSubstitution(self):
        """Substitutability grid for the conceptual SBOW (word2vec) demonstration
        (doc/Mereology.md#mereological-algorithm / the conceptual-similarity
        layer): a FULL noun×verb
        grid, so every noun shares all verb-contexts and every verb shares all
        noun-contexts -- the distributional structure SBOW situates. After
        training, substitutable concepts should CO-LOCATE in similarity_codebook:
        nouns {cat,dog,bird} cluster, verbs {runs,eats,sleeps} cluster, and
        within-class cosine > cross-class cosine. Labels are incidental (the
        conceptual SBOW loss drives co-location, not the labels)."""
        nouns = ["cat", "dog", "bird"]
        verbs = ["runs", "eats", "sleeps"]
        texts = ["%s %s" % (n, v) for n in nouns for v in verbs]
        labels = [[i % 2] for i in range(len(texts))]
        data = {
            "train":      {"text": texts, "label": labels},
            "validation": {"text": texts, "label": labels},
            "test":       {"text": texts, "label": labels},
        }
        self.train_input       = data["train"]["text"]
        self.train_output      = data["train"]["label"]
        self.validation_input  = data["validation"]["text"]
        self.validation_output = data["validation"]["label"]
        self.test_input        = data["test"]["text"]
        self.test_output       = data["test"]["label"]
        self.processLM(data)

    def loadQueries(self):
        """Tiny yes/no parthood QA set for reasoning Phase C/E validation. Each
        item is an interrogative; label = 1 if the parthood holds under the
        MM_query_reasoning truthSet (socrates ⊑ human ⊑ mortal), else 0. The
        answer-policy loss (Phase C) is self-supervised from the truthSet store,
        not these labels -- this set drives the codebook to learn the words and
        is the serve/infer (Phase E) validation corpus."""
        texts = [
            "is socrates part of human",    # direct  -> true
            "is socrates part of mortal",   # 2-hop   -> true
            "is human part of mortal",      # direct  -> true
            "is mortal part of socrates",   # reversed -> false
            "is human part of socrates",    # reversed -> false
            "is socrates part of stone",    # no chain -> false
        ]
        labels = [[1], [1], [1], [0], [0], [0]]
        data = {
            "train":      {"text": texts, "label": labels},
            "validation": {"text": texts, "label": labels},
            "test":       {"text": texts, "label": labels},
        }
        self.train_input       = data["train"]["text"]
        self.train_output      = data["train"]["label"]
        self.validation_input  = data["validation"]["text"]
        self.validation_output = data["validation"]["label"]
        self.test_input        = data["test"]["text"]
        self.test_output       = data["test"]["label"]
        self.processLM(data)

    def loadSequences(self):
        """Multi-sentence DOCUMENTS for inter-sentence prediction (the L_inter
        MSE + InfoNCE contrastive next-idea terms). Each row is ONE LONG document
        of '. '-delimited sentences over a shared vocab. Two preconditions for
        the discourse end-state chain to span sentences (so the next-idea loss
        fires): (1) the BYTE cursor (this config sets <lexer>byte</lexer>) keeps
        the document in one stream -- a trial cursor resets the chain every tick;
        (2) each document must EXCEED the byte slab width (InputSpace nIdeas, ~1024)
        so it is walked over MULTIPLE ticks -- one end-state per tick, the chain
        accumulating to >=2 before the document-boundary reset. A short document
        collapses to a single tick / single end-state and never trains the
        predictor. So each base pattern is repeated to ~1600 bytes."""
        bases = ["the cat sat. the dog ran. ",
                 "the dog sat. the cat ran. ",
                 "the cat ran. the dog sat. ",
                 "the dog ran. the cat sat. "]
        docs = [(b * 64).strip() for b in bases]   # ~1600 bytes each (> slab)
        labels = [[0], [0], [0], [0]]
        data = {
            "train":      {"text": docs, "label": labels},
            "validation": {"text": docs, "label": labels},
            "test":       {"text": docs, "label": labels},
        }
        self.train_input       = data["train"]["text"]
        self.train_output      = data["train"]["label"]
        self.validation_input  = data["validation"]["text"]
        self.validation_output = data["validation"]["label"]
        self.test_input        = data["test"]["text"]
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

        When the ``<data>`` block has no ``<input>`` children at all
        (``<dataset>inline</dataset>`` standalone), a deterministic
        random sentence corpus is synthesized so probe / smoke configs
        like ``data/idempotent.xml`` can drive the pipeline end-to-end
        without authoring a payload. The random vocabulary is just
        ``tok0 ... tok7``; labels are ``0`` / ``1`` alternating.
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

        if not (train_texts or test_texts or val_texts):
            # Standalone <dataset>inline</dataset> with no <input> /
            # <output> children. Synthesize a deterministic random
            # corpus so probe configs (idempotent.xml etc.) can drive
            # the full forward / reverse pipeline.
            train_texts, train_labels = self._synthetic_inline_split(seed=0,  n=16)
            val_texts,   val_labels   = self._synthetic_inline_split(seed=1,  n=4)
            test_texts,  test_labels  = self._synthetic_inline_split(seed=2,  n=4)

        data = {
            "train":      {"text": train_texts, "label": train_labels},
            "validation": {"text": val_texts,   "label": val_labels},
            "test":       {"text": test_texts,  "label": test_labels},
        }
        self.processLM(data)

    @staticmethod
    def _synthetic_inline_split(seed, n, vocab_size=8, sentence_len=4):
        """Deterministic random sentences + alternating 0/1 labels.

        Used as the auto-fallback when ``<dataset>inline</dataset>``
        appears without ``<input>`` / ``<output>`` children.
        """
        import random
        rng = random.Random(seed)
        vocab = [f"tok{i}" for i in range(vocab_size)]
        texts = []
        labels = []
        for i in range(n):
            sentence = " ".join(rng.choice(vocab) for _ in range(sentence_len))
            texts.append(sentence)
            labels.append([float(i % 2)])
        return texts, labels
    def loadTomatoes(self):
        """Load the rotten_tomatoes HuggingFace dataset (cached on disk).

        Downloads on first call, then caches as a single ``.data``
        pickle for fast reload. Forwards the splits to ``processLM``.
        """
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
    def loadShards(self, num_shards, max_docs, shard_dir,
                   random_shards=False, max_tokens=None):
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
        shard_paths = get_shard_paths(
            shard_dir, num_shards=num_shards,
            random_select=bool(random_shards))
        if not shard_paths:
            raise RuntimeError(f"No shards found in {shard_dir}. "
                               "Run 'make basic_data' first.")
        self.source_manifest = {
            "dataset": "text",
            "shards": [os.path.basename(p) for p in shard_paths],
            "max_docs": int(max_docs) if max_docs is not None else None,
            "max_tokens": (int(max_tokens)
                           if max_tokens is not None else None),
            "random_shards": bool(random_shards),
            "split": "document_mod10_8_1_1",
        }

        # Split by DOCUMENT before sentence expansion.  The old sentence-level
        # slice leaked boundary documents across train/eval. FineWeb shards are
        # already shuffled, so a stable 8/1/1 block assignment preserves order
        # and keeps every document in exactly one split. Consume the parquet
        # iterator once rather than first materializing a second ``docs`` list.
        # Sentence rows remain resident because SentenceStreamDataset requires
        # random access for its contiguous streams.
        # Each item in the resulting list is one sentence -- the unit the
        # SentenceStreamDataset feeds per row in trial-cursor mode, and
        # the unit the word-lexer tokenizes downstream. Sentence-sized
        # rows fit comfortably in the InputSpace slab (typically 5-30
        # word tokens vs. ~3000 for a full document under the per-char
        # whitespace/punct lexer).
        from util import parse
        split_sentences = {"train": [], "validation": [], "test": []}
        docs_seen = 0
        for doc_idx, doc in enumerate(
                iter_documents(shard_paths, max_docs=max_docs)):
            residue = doc_idx % 10
            split = ("train" if residue < 8 else
                     "validation" if residue == 8 else "test")
            for sent_text, _ in parse(doc, lex='sentences'):
                if sent_text.strip():
                    if max_tokens is not None:
                        words = sent_text.split()
                        sent_text = " ".join(
                            words[:max(1, int(max_tokens))])
                    split_sentences[split].append(sent_text)
            docs_seen += 1

        if docs_seen == 0:
            raise RuntimeError("No documents found in shards.")
        n = sum(len(v) for v in split_sentences.values())
        if n == 0:
            raise RuntimeError("No sentences found after splitting documents.")

        train_texts = split_sentences["train"]
        val_texts = split_sentences["validation"]
        test_texts = split_sentences["test"]

        data = {
            "train":      {"text": train_texts, "label": []},
            "validation": {"text": val_texts,   "label": []},
            "test":       {"text": test_texts,  "label": []},
        }

        print(f"Loaded {docs_seen} docs -> {n} sentences "
              f"({len(train_texts)} train, {len(val_texts)} val, "
              f"{len(test_texts)} test)")
        self.processLM(data)
    def processLM(self, data):
        """Stash text splits as lists; tensorize labels eagerly when numeric.

        Inputs are kept as raw strings (tensorized lazily in
        ``prepInput``). Labels with string values become an LM target
        deferral (``_lm_labels``), numeric labels become per-row float
        tensors. Missing labels produce zero-tensor sentinels sized
        for OutputSpace.
        """
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
            # No labels provided: sentinel outputs for OutputSpace sizing.
            self.train_output = [torch.zeros(1) for _ in train_tokens]
            self.validation_output = [torch.zeros(1) for _ in validation_tokens]
            self.test_output = [torch.zeros(1) for _ in test_tokens]
            self._lm_labels = None
            self.has_supervised_outputs = False
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
            self.has_supervised_outputs = False
        else:
            self._lm_labels = None
            self.has_supervised_outputs = True
            # ``torch.as_tensor`` is tensor-aware: for tensor inputs it
            # avoids the "copy-construct from a tensor" deprecation
            # warning that ``torch.tensor`` triggers; for list inputs it
            # builds a new tensor like ``torch.tensor`` does.
            self.train_output = [torch.as_tensor(l, dtype=torch.float) for l in train_labels]
            self.validation_output = [torch.as_tensor(l, dtype=torch.float) for l in validation_labels]
            self.test_output = [torch.as_tensor(l, dtype=torch.float) for l in test_labels]

    def tokenize(self, TheLanguageModel):
        """Replace each split's text with ``TheLanguageModel.tokenize`` output.

        Used when the consumer wants per-document token id lists rather
        than raw strings (e.g., when the embedding pipeline does the
        tokenization upstream of the model).
        """
        self.train_input      = TheLanguageModel.tokenize(self.train_input)
        self.validation_input = TheLanguageModel.tokenize(self.validation_input)
        self.test_input       = TheLanguageModel.tokenize(self.test_input)

    def data_loader(self, split, num_streams, num_workers=0,
                    prefetch_factor=None, pin_memory=False,
                    slab_bytes=None):
        """Return a DataLoader over the given split as B contiguous streams.

        ``split`` selects one of ``train`` / ``validation`` / ``test``.
        ``num_streams`` is the effective batch size. It is capped at the
        number of available items so callers never need to worry about
        tiny eval splits (``batchSize=10`` on a 4-item XOR test set yields
        one batch of 4 rows). ``num_workers`` and ``prefetch_factor`` are
        forwarded to ``torch.utils.data.DataLoader``. The dataset self-
        batches (yields ``(inputs, outputs)`` per step), so
        ``batch_size=None`` on the DataLoader.

        ``slab_bytes`` (optional): when set, enables the rolling-cursor
        path on the returned dataset (see ``SentenceStreamDataset.next_tick``).
        Callers that drive the cursor directly (``ds.next_tick()``) bypass
        the DataLoader-yielded batches; ``num_workers`` / ``prefetch_factor``
        only affect the legacy ``__iter__`` path.
        """
        # ``runtime`` is the transient inference split that ``runtime_batch``
        # stages into ``train_input`` / ``train_output`` (there is no separate
        # ``runtime_*`` attribute); map it to those so a runtime/inference
        # epoch (``runEpoch(split="runtime")``) drives the staged data.
        eff_split = "train" if split == "runtime" else split
        inputs = getattr(self, f"{eff_split}_input")
        outputs = getattr(self, f"{eff_split}_output", None)
        n = (inputs.shape[0] if isinstance(inputs, torch.Tensor)
             else len(inputs))
        if n == 0:
            raise RuntimeError(
                f"data_loader: {eff_split}_input is empty -- "
                "load() before building a loader"
            )
        streams = max(1, min(num_streams, n))
        ds = SentenceStreamDataset(inputs, num_streams=streams,
                                   outputs=outputs,
                                   slab_bytes=slab_bytes)

        kwargs = {"batch_size": None, "num_workers": num_workers,
                  "pin_memory": pin_memory}
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(ds, **kwargs)
    def data(self):
        """Return the train / validation / test splits as a nested dict.

        Mirrors the input layout that ``processLM`` consumes. Useful for
        debugging or for handing the structured view to another loader.
        """
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
        """Return ``(input_size, output_size)`` for embedding layer sizing."""
        return self.getInputSize(), self.getOutputSize()
    def getInputSize(self):
        """Return per-input vector length (string -> inputLength; else dim 0).

        Strings use the fixed ``self.inputLength`` byte buffer; tensor
        rows report their leading dim. Assumes the split is non-empty.
        """
        if self.train_input and isinstance(self.train_input[0], str):
            return self.inputLength
        inputEmbeddingSize  = self.train_input[0].shape[0]
        return inputEmbeddingSize
    def getOutputSize(self):
        """Return per-output vector length (first item's leading dim).

        Assumes ``self.train_output`` is a non-empty list of tensors.
        """
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
            """Look up each word in the CBOW vocabulary, return L2-normalized vectors.

            Unknown words emit a zero vector. The lookup is detached and
            cloned so later optimizer steps cannot move the targets out
            from under the loss.
            """
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
        """ASCII-encode ``string`` into a fixed-length ``int8`` tensor.

        Non-ASCII chars (smart quotes, accents) are replaced. The
        result is truncated or NULL-padded to ``self.inputLength`` so
        downstream callers can stack across rows.
        """
        # Encode to ASCII, replacing non-ASCII chars (smart quotes, accents, etc.)
        ascii_values = list(string.encode('ascii', errors='replace'))[:self.inputLength]
        # Force CPU: this is a host string->bytes encode. A default-device
        # mode (e.g. MPS) would otherwise place it on-device, so the
        # lexer's host-token carry (residual A) would still round-trip
        # through the GPU. Callers move to the device explicitly when
        # they need it (prepInput stacks then `.to(device)`).
        tensor = torch.tensor(ascii_values, dtype=torch.int8, device='cpu')
        if tensor.size(0) < self.inputLength:
            # Pad with NULL (0) -- input buffer is variable-length, null-terminated
            tensor = F.pad(tensor, (0, self.inputLength - tensor.size(0)), 'constant', 0)
        return tensor

TheData = Data()
