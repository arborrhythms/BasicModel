"""Word embedding pipeline: FineWeb-EDU -> parse -> CBOW -> BasicModel.kv

Training phase that produces a static word embedding artifact. InputSpace
loads this artifact at startup -- it does not train.

Pipeline stages:
  1. Stream text documents from FineWeb-EDU parquet shards
  2. parse(text, lex='sentences') then parse(sent, lex='words'):
     split into sentences, then word tokens
  3. Build (target, context) training examples per sentence
  4. Train CBOW embeddings: predict target from mean of context vectors
  5. Save as WordVectors artifact (.kv, gensim-compatible KeyedVectors)

Usage:
    python bin/embed.py --output output/BasicModel.kv \
        --num-shards 1 --max-docs 10000 --vector-size 100 --epochs 10

This module also owns the unified ``.kv``/``.pt`` *vocab-artifact*
schema used by both ``WordVectors`` (the Lexicon path) and
``ChunkLayer`` (the BPE path). See ``save_artifact`` / ``load_artifact``
below; they let a single artifact carry a Lexicon, a BPE codebook, or
both side-by-side, distinguishable by the top-level ``kind`` field
(and per-section ``section_kind`` markers).
"""

import os
import sys
import argparse
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict, Iterable, List, Tuple, Optional

from util import atomic_torch_save, parse
import util
from Layers import Lexicon


# ---------------------------------------------------------------------------
# Unified vocab-artifact schema
#
# Both ``WordVectors`` (word strings -> learned embedding vectors) and
# ``ChunkLayer`` (byte-tuple merges -> integer chunk ids) need to save /
# load a vocabulary so training can resume without rediscovering it from
# scratch. This module gives both paths one schema:
#
#     {
#         "format_version": 1,
#         "kind": "lexicon" | "bpe" | "both",
#         "lexicon": { ... },     # WordVectors section (when present)
#         "bpe":     { ... },     # ChunkLayer section  (when present)
#         "truth_data":   {... }, # optional LTM snapshot (legacy field)
#         "metadata":     {... }, # creation timestamp, source corpus, etc.
#     }
#
# Each section also carries a ``section_kind`` marker so a consumer that
# was handed a section dict directly can still distinguish it.
#
# Backward compatibility: files saved by older ``WordVectors.save`` (no
# ``format_version`` key) are recognised by ``load_artifact`` and lifted
# into the unified shape transparently with ``kind="lexicon"``.
# ---------------------------------------------------------------------------

FORMAT_VERSION = 1
KIND_LEXICON = "lexicon"
KIND_BPE = "bpe"
KIND_BOTH = "both"
_VALID_KINDS = (KIND_LEXICON, KIND_BPE, KIND_BOTH)


def _wrap_unit_ball(x: torch.Tensor) -> torch.Tensor:
    """Wrap coordinates into the periodic unit cell [-1, 1).

    Idempotent on already-wrapped inputs; on legacy sphere vectors (norm
    ~= 1, components in [-1, 1]) it is a no-op. Optimizer-drifted values
    outside the cell get reprojected to their wrapped equivalent.
    """
    return torch.remainder(x + 1.0, 2.0) - 1.0


def _wrapped_mse_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torus similarity: ``-mean(wrapped_delta^2)``. Larger is closer.

    Thin wrapper around :py:meth:`Layers.Lexicon.similarity` for paths
    that compare differently-sized tensors (the ``min(...)`` width
    truncation isn't on Lexicon's static API). New call sites should
    use ``Lexicon.similarity`` directly when ``a`` and ``b`` have
    matching last-dim widths.
    """
    d = min(a.shape[-1], b.shape[-1])
    delta = _wrap_unit_ball(a[..., :d] - b[..., :d])
    return -delta.square().mean(dim=-1)


def _random_unit_ball(shape, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Uniform random coordinates in the periodic unit cell [-1, 1)."""
    return torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)


def save_artifact(
    path: str,
    *,
    lexicon: Optional[Dict[str, Any]] = None,
    bpe: Optional[Dict[str, Any]] = None,
    truth_data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a vocabulary artifact in the unified schema.

    At least one of ``lexicon`` or ``bpe`` must be provided; ``kind`` is
    inferred from which sections are present.
    """
    if lexicon is None and bpe is None:
        raise ValueError(
            "save_artifact: at least one of lexicon / bpe must be provided")
    if lexicon is not None and bpe is not None:
        kind = KIND_BOTH
    elif lexicon is not None:
        kind = KIND_LEXICON
    else:
        kind = KIND_BPE
    md = dict(metadata) if metadata else {}
    md.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "metadata": md,
    }
    if lexicon is not None:
        payload["lexicon"] = lexicon
    if bpe is not None:
        payload["bpe"] = bpe
    if truth_data is not None:
        payload["truth_data"] = truth_data
    atomic_torch_save(payload, path)


def load_artifact(path: str) -> Dict[str, Any]:
    """Load a vocabulary artifact and return the parsed payload.

    Old-format ``WordVectors`` files (no ``format_version`` key) are
    transparently lifted into the unified schema with ``kind="lexicon"``.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"load_artifact: {path!r} did not contain a dict")
    if "format_version" in raw:
        return raw
    lexicon_section = {
        k: raw[k]
        for k in ("vectors", "index_to_key", "counts", "total_count")
        if k in raw
    }
    if not lexicon_section:
        raise ValueError(
            f"load_artifact: {path!r} is neither a unified-schema "
            f"artifact (no 'format_version') nor a legacy WordVectors "
            f"payload (no 'vectors' / 'index_to_key' keys)")
    lexicon_section.setdefault("section_kind", KIND_LEXICON)
    upgraded: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": KIND_LEXICON,
        "metadata": {"upgraded_from_legacy": True},
        "lexicon": lexicon_section,
    }
    if "truth_data" in raw and raw["truth_data"] is not None:
        upgraded["truth_data"] = raw["truth_data"]
    return upgraded


def lexicon_section_from_word_vectors(wv) -> Dict[str, Any]:
    """Build the ``lexicon`` section from a ``WordVectors`` instance.

    Vectors are wrapped into the periodic unit cell ``[-1, 1)`` before
    saving, so artifacts on disk are always in the canonical domain even
    if the live parameter has drifted between optimizer steps.
    """
    raw = wv._vectors.detach()
    vectors = _wrap_unit_ball(raw).cpu()
    counts = getattr(wv, "counts", None)
    if counts is not None:
        counts = np.asarray(counts, dtype=np.int64)
    return {
        "section_kind": KIND_LEXICON,
        "vectors": vectors,
        "index_to_key": list(wv.index_to_key),
        "counts": counts,
        "total_count": int(getattr(wv, "total_count", 0)),
    }


def bpe_section_from_chunk_layer(chunk_layer) -> Dict[str, Any]:
    """Build the ``bpe`` section from a ``ChunkLayer`` instance.

    Stores the merge table in insertion order, plus the byte seed and
    config knobs needed to resume training in the same regime
    (``n_vectors``, ``chunking_frequency``). The section carries
    ``section_kind="bpe"``.
    """
    return {
        "section_kind": KIND_BPE,
        "merges": list(chunk_layer.merges),
        "vocab": {tuple(k): int(v) for k, v in chunk_layer.vocab.items()},
        "id_to_bytes": {
            int(k): tuple(v) for k, v in chunk_layer.id_to_bytes.items()},
        "next_id": int(chunk_layer._next_id),
        "max_merge_len": int(chunk_layer._max_merge_len),
        "n_vectors": int(chunk_layer.n_vectors),
        "chunking_frequency": int(chunk_layer.chunking_frequency),
    }


def inspect_artifact(path: str) -> Dict[str, Any]:
    """Quick peek at an artifact: ``kind``, metadata, section sizes.

    No Module instantiation, no NaN repair, no embedding allocation.
    Useful before deciding which loader to invoke (``WordVectors.load``
    vs ``ChunkLayer.load``) on an unknown file.
    """
    payload = load_artifact(path)
    lex = payload.get("lexicon") or {}
    bpe = payload.get("bpe") or {}
    return {
        "format_version": payload.get("format_version", FORMAT_VERSION),
        "kind": payload.get("kind", "unknown"),
        "metadata": payload.get("metadata", {}),
        "has_lexicon": bool(lex),
        "has_bpe": bool(bpe),
        "lexicon_size": len(lex.get("index_to_key", [])) if lex else 0,
        "bpe_size": len(bpe.get("vocab", {})) if bpe else 0,
        "has_truth_data": payload.get("truth_data") is not None,
    }


def bpe_to_lexicon_keys(chunk_layer) -> List[str]:
    """Map a ChunkLayer's byte-tuple merge entries to string keys.

    Each chunk id maps to its byte-tuple decoded as Latin-1 (1 byte =
    1 codepoint, lossless round-trip via ``str.encode('latin-1')``).
    Suitable as a WordVectors ``index_to_key`` for cold-starting a
    Lexicon from a trained BPE codebook. Vectors are NOT generated --
    the caller materializes per-key embeddings.
    """
    keys: List[str] = []
    n = (max(chunk_layer.id_to_bytes.keys()) + 1
         if chunk_layer.id_to_bytes else 0)
    for i in range(n):
        bt = chunk_layer.id_to_bytes.get(i)
        if bt is None:
            keys.append("")
        else:
            keys.append(bytes(bt).decode("latin-1"))
    return keys


def lexicon_to_bpe_seed(words: Iterable[str],
                        n_vectors: int = 4096) -> Dict[str, Any]:
    """Best-effort BPE seed from a Lexicon vocab list.

    Each word is registered as a single multi-byte chunk -- the merge
    order that produced it is *not* recovered (a Lexicon doesn't record
    that). Useful for skipping BPE cold-start: subsequent training
    sees Lexicon words as ready-made chunks.

    Returns a ``bpe`` section dict directly consumable by
    ``ChunkLayer.load``.
    """
    vocab: Dict[tuple, int] = {(i,): i for i in range(256)}
    id_to_bytes: Dict[int, tuple] = {i: (i,) for i in range(256)}
    next_id = 256
    max_merge_len = 1
    merges: List[tuple] = []
    for w in words:
        if not isinstance(w, str) or not w:
            continue
        if next_id >= n_vectors:
            break
        bt = tuple(w.encode("utf-8"))
        if bt in vocab:
            continue
        if len(bt) >= 2:
            merges.append((bt[0], next_id - 1 if next_id > 256 else bt[1]))
        vocab[bt] = next_id
        id_to_bytes[next_id] = bt
        max_merge_len = max(max_merge_len, len(bt))
        next_id += 1
    return {
        "section_kind": KIND_BPE,
        "merges": merges,
        "vocab": {tuple(k): int(v) for k, v in vocab.items()},
        "id_to_bytes": {int(k): tuple(v) for k, v in id_to_bytes.items()},
        "next_id": next_id,
        "max_merge_len": max_merge_len,
        "n_vectors": int(n_vectors),
        "chunking_frequency": 2,
    }


# ---------------------------------------------------------------------------
# Word vector store
# ---------------------------------------------------------------------------

class WordVectors(nn.Module):
    """Stores word embeddings as a trainable nn.Parameter with word <-> index mappings.

    API-compatible with gensim's KeyedVectors: ``wv.vectors``, ``wv.vector_size``,
    ``wv[word]``, ``wv.key_to_index``, ``wv.index_to_key``, ``wv.most_similar()``.

    Being an nn.Module, the parameter moves to device via ``.to(device)``
    and participates in ``state_dict()`` / ``parameters()``.
    """

    def __init__(self, vectors, index_to_key: List[str],
                 counts=None, total_count: int = 0):
        """Create from a (vocab_size, vector_size) array/tensor and word list.

        Coordinates are wrapped into the periodic unit cell ``[-1, 1)`` on
        construction so callers passing arbitrary tensors (legacy sphere
        artifacts, fresh randoms, etc.) all land in the canonical domain.
        """
        super().__init__()
        assert len(index_to_key) == vectors.shape[0]
        if not isinstance(vectors, torch.Tensor):
            vectors = torch.as_tensor(vectors, dtype=torch.float32)
        else:
            vectors = vectors.detach().float()
        vectors = _wrap_unit_ball(vectors)
        self._vectors = nn.Parameter(vectors, requires_grad=True)
        self.index_to_key = list(index_to_key)
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        n = len(self.index_to_key)
        if counts is not None:
            self.counts = np.asarray(counts, dtype=np.int64)
        else:
            self.counts = np.zeros(n, dtype=np.int64)
        self.total_count = np.int64(total_count)
        self._normed: Optional[torch.Tensor] = None

    @property
    def vectors(self) -> nn.Parameter:
        """Trainable embedding matrix (vocab_size, vector_size)."""
        return self._vectors

    @property
    def vector_size(self) -> int:
        """Dimensionality of each vector."""
        return self._vectors.shape[1]

    # -- Factory methods --

    @classmethod
    def from_vocab(cls, words: List[str], vector_size: int = 20) -> "WordVectors":
        """Build random embeddings for a vocabulary list."""
        unique = list(dict.fromkeys(words))  # preserve order, deduplicate
        vecs = _random_unit_ball((len(unique), vector_size))
        return cls(vecs, unique)

    @classmethod
    def load_word2vec_format(cls, path: str) -> "WordVectors":
        """Load vectors from a word2vec text-format file.

        First line: ``<vocab_size> <vector_size>``
        Subsequent lines: ``<word> <float> <float> ...``
        """
        words: List[str] = []
        vecs: List[List[float]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().split()
            vocab_size, vector_size = int(header[0]), int(header[1])
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != vector_size + 1:
                    continue  # skip malformed lines
                words.append(parts[0])
                vecs.append([float(x) for x in parts[1:]])
        return cls(torch.tensor(vecs, dtype=torch.float32), words)

    def remove(self, indices):
        """Remove entries by index, returning the pruned count.

        Shrinks vectors, counts, and vocabulary mappings in-place.
        """
        if not indices:
            return 0
        removed_set = set(indices)
        mask = torch.ones(self._vectors.shape[0], dtype=torch.bool)
        for i in removed_set:
            mask[i] = False
        self._vectors = nn.Parameter(self._vectors.data[mask], requires_grad=True)
        self.counts = np.delete(self.counts, list(indices))
        new_keys = [w for i, w in enumerate(self.index_to_key) if i not in removed_set]
        self.index_to_key = new_keys
        self.key_to_index = {w: i for i, w in enumerate(new_keys)}
        self._normed = None
        return len(removed_set)

    # -- Persistence --

    def save(self, path: str, truth_data: dict = None,
             bpe_section: dict = None, metadata: dict = None) -> None:
        """Save vectors, vocabulary, and word frequencies via the unified
        vocab-artifact format (see ``save_artifact`` in this module).

        Vectors are wrapped into the periodic unit cell ``[-1, 1)`` before
        saving (see ``lexicon_section_from_word_vectors``).

        Args:
            path: destination file path.
            truth_data: optional dict with ``"truths"`` tensor and
                ``"count"`` int, persisting LTM alongside the embedding
                artifact.
            bpe_section: optional pre-built BPE section (typically from
                ``bpe_section_from_chunk_layer``). When provided the
                saved file carries both the Lexicon and the BPE
                codebook side-by-side (kind="both") so a single
                ``.kv`` artifact can serve either path.
            metadata: optional free-form metadata dict carried through
                into the artifact's ``metadata`` field.
        """
        save_artifact(
            path,
            lexicon=lexicon_section_from_word_vectors(self),
            bpe=bpe_section,
            truth_data=truth_data,
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str) -> "WordVectors":
        """Load from a ``.pt``/``.kv`` file.

        Accepts both the unified vocab-artifact schema (kind in
        {lexicon, both}) and legacy ``WordVectors``-only payloads (no
        ``format_version``); ``load_artifact`` lifts the latter into
        the unified shape transparently.
        """
        payload = load_artifact(path)
        if payload.get("kind") == KIND_BPE:
            raise ValueError(
                f"WordVectors.load: {path!r} is a BPE-only artifact "
                f"(kind=bpe); no lexicon section to load. Use "
                f"ChunkLayer.load instead, or convert via "
                f"bpe_to_lexicon_keys.")
        section = payload.get("lexicon")
        if section is None:
            raise ValueError(
                f"WordVectors.load: {path!r} has no lexicon section "
                f"(kind={payload.get('kind')!r})")
        vectors = section["vectors"]
        if isinstance(vectors, np.ndarray):
            vectors = torch.as_tensor(vectors, dtype=torch.float32)
        else:
            vectors = vectors.float()
        # Replace NaN vectors with fresh uniform draws in the unit cell.
        nan_mask = torch.isnan(vectors).any(dim=1)
        n_nan = int(nan_mask.sum().item())
        if n_nan > 0:
            dim = vectors.shape[1]
            replacement = _random_unit_ball(
                (n_nan, dim), device=vectors.device, dtype=vectors.dtype)
            vectors[nan_mask] = replacement
            import warnings
            warnings.warn(f"Replaced {n_nan} NaN embedding vectors in {path}")
        # Legacy sphere artifacts have norm ~= 1 so each component is
        # already in [-1, 1]; ``WordVectors.__init__`` re-wraps to make
        # the domain explicit and idempotent.
        counts = section.get("counts")
        total_count = section.get("total_count", 0)
        wv = cls(vectors, section["index_to_key"],
                 counts=counts, total_count=total_count)
        wv.truth_data = payload.get("truth_data", None)
        # Preserve the artifact's metadata dict so callers can recover
        # training-time settings (e.g. learning_rate, sigma) and use
        # them as defaults on resume.
        wv.metadata = dict(payload.get("metadata") or {})
        return wv

    # -- Vector access --

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, word: str) -> torch.Tensor:
        """Return the raw (unnormalised) vector for *word*."""
        return self._vectors[self.key_to_index[word]]

    def get_normed_vectors(self) -> torch.Tensor:
        """Return raw vectors. Kept for API symmetry; values already in
        ``[-1, 1)`` after every step / save / load."""
        return self._vectors.detach()

    # -- Similarity --

    def most_similar(self, positive,
                     topn: int = 1) -> List[Tuple[str, float]]:
        """Find the *topn* words closest to *positive* under wrapped MSE."""
        if not isinstance(positive, torch.Tensor):
            positive = torch.as_tensor(positive, dtype=torch.float32)
        vectors = self.get_normed_vectors()
        positive = positive.float().flatten().to(vectors.device)
        sims = _wrapped_mse_score(vectors, positive)
        if topn < len(sims):
            _, top_idx = sims.topk(topn)
        else:
            _, top_idx = sims.sort(descending=True)
        return [(self.index_to_key[i], float(sims[i].detach())) for i in top_idx]

    def similarity(self, word1: str, word2: str) -> float:
        """Return wrapped-MSE similarity (larger is closer)."""
        v1 = self[word1].detach()
        v2 = self[word2].detach()
        return float(_wrapped_mse_score(v1, v2))

    # -- Exploration / CLI helpers --

    def neighbors(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Return nearest neighbors of *word*, excluding itself."""
        if word not in self:
            return []
        results = self.most_similar(self[word], topn=topn + 1)
        return [(w, s) for w, s in results if w != word][:topn]

    def random_word(self) -> str:
        """Return a random word from the vocabulary."""
        import random
        return random.choice(self.index_to_key)

    def explore(self, words: List[str], topn: int = 10) -> None:
        """Interactive exploration: print neighbors and/or similarity.

        - No words: random word + neighbors
        - One word: neighbors of that word
        - Two words: similarity + neighbors for each
        - Three+ words: neighbors for each
        """
        if len(words) == 0:
            word = self.random_word()
            print(f"Random word: '{word}'\n")
            words = [word]

        if len(words) == 2 and words[0] in self and words[1] in self:
            sim = self.similarity(words[0], words[1])
            print(f"similarity('{words[0]}', '{words[1]}') = {sim:.4f}\n")

        for word in words:
            if word not in self:
                print(f"'{word}' not in vocabulary ({len(self)} words)")
                lower = word.lower()
                if lower in self:
                    print(f"  (did you mean '{lower}'?)")
                print()
                continue
            results = self.neighbors(word, topn=topn)
            print(f"Nearest neighbors of '{word}':")
            for w, sim in results:
                print(f"  {w:20s}  {sim:.4f}")
            print()


# FineWeb-EDU streaming -- canonical implementation lives in data.py
from data import download_shard, get_shard_paths, iter_documents  # noqa: F401


# ---------------------------------------------------------------------------
# Embedding pretrainer (streaming)
# ---------------------------------------------------------------------------

class PretrainModel:
    """Stateful embedding pretrainer: optimizer + negative sampling.

    Supports both bulk training (``train``) and incremental single-sentence
    updates (``train_step``).  Operates directly on the WordVectors'
    nn.Parameter -- no separate nn.Embedding.
    """

    def __init__(self, wv: WordVectors, learning_rate=0.01, neg_samples=64):
        """Initialise from an existing WordVectors.

        Uses negative sampling instead of a full-softmax linear head,
        keeping MPS/GPU memory proportional to ``neg_samples`` rather
        than ``vocab_size``.
        """
        self.wv = wv
        self.index_to_key = wv.index_to_key
        self.key_to_index = wv.key_to_index
        self.neg_samples = neg_samples

        self.optimizer = optim.Adam(
            [wv._vectors],
            lr=learning_rate,
        )

        # Per-word gradient variance (sigma) -- lazy-initialized in observe_sigma
        self.sigma = None
        self.sigma_mean = None
        self.sigma_step = 0
        self.sigma_beta = 0.99

    @torch.no_grad()
    def observe_sigma(self, word_indices):
        """Track per-word gradient variance via Welford's algorithm.

        Called after backward() and before optimizer.step() so that
        _vectors.grad is available.  Works regardless of which
        training method (CBOW, SBOW, etc.) produced the gradients.
        """
        grad = self.wv._vectors.grad
        if grad is None:
            return
        device = grad.device
        vocab_size = self.wv._vectors.shape[0]
        if self.sigma is None:
            self.sigma = torch.zeros(vocab_size, device=device)
            self.sigma_mean = torch.zeros(vocab_size, device=device)
        elif self.sigma.shape[0] < vocab_size:
            old = self.sigma.shape[0]
            self.sigma = nn.functional.pad(self.sigma, (0, vocab_size - old))
            self.sigma_mean = nn.functional.pad(self.sigma_mean, (0, vocab_size - old))
        self.sigma_step += 1
        beta = self.sigma_beta
        for wi in word_indices:
            g = grad[wi].pow(2).mean()
            delta = g - self.sigma_mean[wi]
            self.sigma_mean[wi] += (1 - beta) * delta
            self.sigma[wi] = beta * self.sigma[wi] + (1 - beta) * delta * (g - self.sigma_mean[wi])

    # -- single-sentence update ------------------------------------------------

    def _neg_sampling_loss(self, queries, target_idx):
        """Negative sampling loss: -log sigma(q*w^+) - Sigma log sigma(-q*w_k^-).

        Args:
            queries: [N, dim] query vectors (context centroids).
            target_idx: [N] indices of positive (target) words.
        Returns:
            Scalar loss.
        """
        device = queries.device
        vocab_size = self.wv._vectors.shape[0]
        K = min(self.neg_samples, vocab_size - 1)

        pos_vecs = self.wv._vectors[target_idx]                      # [N, dim]
        pos_scores = _wrapped_mse_score(queries, pos_vecs)           # [N]

        neg_idx = torch.randint(0, vocab_size, (queries.shape[0], K),
                                device=device)                        # [N, K]
        neg_vecs = self.wv._vectors[neg_idx]                         # [N, K, dim]
        neg_scores = _wrapped_mse_score(
            queries.unsqueeze(1), neg_vecs)                          # [N, K]

        return -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()

    def _post_step(self):
        with torch.no_grad():
            self.wv._vectors.copy_(_wrap_unit_ball(self.wv._vectors))
        self.wv._normed = None

    # -- single-sentence update ------------------------------------------------

    def train_step(self, words):
        """Run one CBOW gradient step on a sentence via negative sampling.

        Returns the mean loss for the sentence, or ``None`` if no usable
        examples were found.
        """
        targets = []
        contexts = []
        for i, w in enumerate(words):
            if w not in self.key_to_index:
                continue
            ctx = [words[j] for j in range(len(words))
                   if j != i and words[j] in self.key_to_index]
            if not ctx:
                continue
            targets.append(self.key_to_index[w])
            contexts.append([self.key_to_index[c] for c in ctx])

        if not targets:
            return None

        device = self.wv._vectors.device

        max_ctx = max(len(c) for c in contexts)
        n = len(targets)
        ctx_padded = torch.zeros(n, max_ctx, dtype=torch.long, device=device)
        ctx_mask = torch.zeros(n, max_ctx, device=device)
        for i, c in enumerate(contexts):
            ctx_padded[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)
            ctx_mask[i, :len(c)] = 1.0
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)

        ctx_embeds = self.wv._vectors[ctx_padded]
        masked = ctx_embeds * ctx_mask.unsqueeze(-1)
        ctx_mean = masked.sum(dim=1) / ctx_mask.sum(dim=1, keepdim=True)

        loss = self._neg_sampling_loss(ctx_mean, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.observe_sigma(targets)
        self.optimizer.step()
        self._post_step()

        return loss.item()

    # -- SBOW: sentence bag-of-words update ------------------------------------

    def sbow_step(self, words):
        """Sentence BOW via negative sampling: predict every word from its
        leave-one-out centroid.  Returns mean loss, or None if < 2 usable words.
        """
        word_indices = [self.key_to_index[w] for w in words
                        if w in self.key_to_index]
        if len(word_indices) < 2:
            return None

        device = self.wv._vectors.device
        idx = torch.tensor(word_indices, dtype=torch.long, device=device)
        N = len(word_indices)

        vecs = self.wv._vectors[idx]                     # [N, dim]
        total = vecs.sum(dim=0)                           # [dim]
        centroids = (total.unsqueeze(0) - vecs) / (N - 1) # [N, dim]

        loss = self._neg_sampling_loss(centroids, idx)

        self.optimizer.zero_grad()
        loss.backward()
        self.observe_sigma(word_indices)
        self.optimizer.step()
        self._post_step()

        return loss.item()

    def sbow_loss(self, words):
        """Return SBOW loss as a differentiable tensor (no backward, no step).

        For joint optimization: the caller accumulates this loss with the
        model loss and calls backward() once on the combined scalar.

        Returns a scalar loss tensor, or None if < 2 usable words.
        """
        word_indices = [self.key_to_index[w] for w in words
                        if w in self.key_to_index]
        if len(word_indices) < 2:
            return None

        device = self.wv._vectors.device
        idx = torch.tensor(word_indices, dtype=torch.long, device=device)
        N = len(word_indices)

        vecs = self.wv._vectors[idx]                     # [N, dim]
        total = vecs.sum(dim=0)                           # [dim]
        centroids = (total.unsqueeze(0) - vecs) / (N - 1) # [N, dim]

        return self._neg_sampling_loss(centroids, idx)

    # -- export ----------------------------------------------------------------



class _CBOWModule(nn.Module):
    """CBOW forward pass as an nn.Module for torch.compile."""

    def __init__(self, vocab_size, vector_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vector_size)
        self.linear = nn.Linear(vector_size, vocab_size)

    def forward(self, ctx_padded, ctx_mask):
        ctx_embeds = self.embeddings(ctx_padded)
        masked = ctx_embeds * ctx_mask.unsqueeze(-1)
        ctx_mean = masked.sum(dim=1) / ctx_mask.sum(dim=1, keepdim=True)
        return self.linear(ctx_mean)


class _SBOWEmbedding(nn.Module):
    """SBOW embedding: just an embedding table, no linear head.

    Uses ``Lexicon`` so weights live on the flat n-torus ``[-1, 1)^D``;
    init is uniform on the torus and post-step re-projection is
    available via ``embeddings.normalize()``.
    """

    def __init__(self, vocab_size, vector_size):
        super().__init__()
        from Layers import Lexicon
        self.embeddings = Lexicon(vocab_size, vector_size, torus=True)


class StreamingSBOWTrainer:
    """Two-pass SBOW trainer: build vocab first, then stream-train per sentence.

    Pass 1: Stream documents to count words and build vocabulary.
    Pass 2: Stream documents again, train SBOW per sentence -- each word
            predicted from its leave-one-out centroid via full softmax.
    The model is allocated once at its final size.  Uses SGD to avoid
    the 2x memory overhead of Adam's momentum buffers.
    """

    def __init__(self, vector_size=100, min_count=5, learning_rate=0.001,
                 sigma=5.0, normalize=True):
        self.vector_size = vector_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        # Inner Gaussian / random-out-group SBOW knobs.
        # ``sigma`` is the position-distance scale of the inner kernel
        # (Mikolov-style local window). Sentences shorter than ``2σ``
        # produce a near-flat inner kernel and are skipped.
        # ``normalize=True`` re-wraps weights into the periodic unit
        # cell after every optimizer step (Lexicon.normalize()).
        self.sigma = float(sigma)
        self.normalize = bool(normalize)

        self.word_counts = Counter()
        self.word_to_idx = {}
        self.idx_to_word = []

        self.device = util.auto_device()
        print(f"Training on {self.device}")

        self.model = None
        self.optimizer = None
        self.neg_samples = 64

        self.n_examples = 0
        self._total_loss = 0.0
        self._loss_count = 0

    @property
    def vocab_size(self):
        return len(self.idx_to_word)

    @property
    def avg_loss(self):
        return self._total_loss / self._loss_count if self._loss_count else 0.0

    # -- Pass 1: vocabulary building ------------------------------------------

    def scan_document(self, text):
        """Parse one document and count words (no training).

        Hot path -- avoids the sentence split (boundaries are unused
        here) and skips ``parse``'s per-token byte-offset work; uses
        ``Counter.update`` (C-implemented batch increment) instead of
        a Python increment loop.
        """
        from util import _PARSE_WORD_RE
        words = (w for w in _PARSE_WORD_RE.findall(text) if not w.isspace())
        self.word_counts.update(words)

    def build_vocab(self):
        """Promote words that meet min_count and allocate the model.

        Reserved slots are always populated first, before any corpus
        words, regardless of corpus frequency:
          - index 0: NULL_PERCEPT (IR-mode mask sentinel; consumed by
            ``BasicModel.create_ir_mask`` to mark predict-here positions)
          - indices 1..256: single-byte characters ``chr(0)..chr(255)``
            (byte-level fallback / OOV decomposition)

        These guarantee that IR mode and byte-level fallback always have
        known-good slots even on tiny corpora. Corpus words (those with
        count >= ``min_count``) take indices 257+. Words that happen to
        coincide with reserved entries (e.g., a literal space token in
        the corpus) reuse the reserved slot rather than getting a
        duplicate.
        """
        from Spaces import NULL_PERCEPT_KEY

        reserved = [NULL_PERCEPT_KEY] + [chr(b) for b in range(256)]
        for word in reserved:
            self.word_to_idx[word] = len(self.idx_to_word)
            self.idx_to_word.append(word)

        for word, count in self.word_counts.items():
            if count >= self.min_count and word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.idx_to_word)
                self.idx_to_word.append(word)

        self.model = _SBOWEmbedding(
            self.vocab_size, self.vector_size).to(self.device)
        self.model = util.compile(self.model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.neg_samples = 64
        n_corpus = self.vocab_size - len(reserved)
        print(f"  Vocab: {len(reserved)} reserved (NULL + 256 bytes) + "
              f"{n_corpus} corpus = {self.vocab_size} entries "
              f"(min_count={self.min_count})")

    def load_existing(self, wv):
        """Restore vocab + weights from a previously-saved WordVectors.

        Replaces both passes' setup so a subsequent training run picks
        up the same word indices, frequency counts, and vector values
        from where the prior run stopped. Caller has already verified
        ``wv.vector_size == self.vector_size``.

        If the loaded artifact is missing reserved entries (NULL_PERCEPT
        and the 256 single-byte characters -- old kv files saved before
        these were guaranteed), they're appended at the tail so existing
        corpus-word indices remain valid. New kvs always have them at
        indices 0..256 by ``build_vocab``'s contract.
        """
        from Spaces import NULL_PERCEPT_KEY
        self.idx_to_word = list(wv.index_to_key)
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
        # Restore frequency counts so any future build_vocab() call (e.g.,
        # mixed-mode resume + new-vocab pass) sees the same min_count
        # gating as the original training run.
        if wv.counts is not None:
            for word, ct in zip(self.idx_to_word, wv.counts.tolist()):
                self.word_counts[word] = int(ct)

        # Backfill any missing reserved entries at the tail. New kvs
        # have these at the head (build_vocab); legacy kvs may not.
        reserved = [NULL_PERCEPT_KEY] + [chr(b) for b in range(256)]
        appended = []
        for word in reserved:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.idx_to_word)
                self.idx_to_word.append(word)
                appended.append(word)

        self.model = _SBOWEmbedding(
            self.vocab_size, self.vector_size).to(self.device)
        with torch.no_grad():
            saved = wv._vectors.to(self.device)
            n_saved = saved.shape[0]
            target = self.model.embeddings.weight
            if saved.shape == target.shape:
                target.copy_(saved)
            elif n_saved < target.shape[0] and saved.shape[1] == target.shape[1]:
                # Saved file is shorter (didn't include reserved tail).
                # Copy what's there; leave the appended reserved rows at
                # their fresh Lexicon torus-uniform init.
                target[:n_saved].copy_(saved)
            else:
                raise ValueError(
                    f"Loaded vectors {tuple(saved.shape)} do not match "
                    f"model {tuple(target.shape)}")
        self.model = util.compile(self.model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.neg_samples = 64
        if appended:
            print(f"  load_existing: appended {len(appended)} missing "
                  f"reserved entries at indices "
                  f"{self.vocab_size - len(appended)}..{self.vocab_size - 1}")

    # -- Pass 2: streaming SBOW training --------------------------------------

    def process_document(self, text):
        """Parse one document, train SBOW on each sentence immediately."""
        for sent_text, _ in parse(text, lex='sentences'):
            words = [w for w, _ in parse(sent_text, lex='words')
                     if not w.isspace()]
            self._train_sentence(words)

    def _train_sentence(self, words):
        """Pode/antipode SBOW with Gaussian in-group and random out-group.

        For each in-sentence position:
        - **Pode** = inner-Gaussian-weighted mean of nearby in-sentence
          words (excluding self). Concentrates on local context.
        - **Antipode** = unique geometric antipode of pode on the torus
          (``Lexicon.antipode``: per-axis +1 mod 2). The maximum-distance
          point from pode in any dimension.
        - **In-group attraction**: every in-sentence word is pulled
          toward its pode -- distributional collapse driven by corpus
          co-occurrence.
        - **Out-group attraction**: K random codebook samples are pulled
          toward the antipode -- isotropic counter-pressure that
          prevents codebook collapse without imposing directional bias
          (random podes across sentences imply uniform random antipodes).

        Both forces are pure attractions, equal power per term.
        Structured in-group pull collapses meaningfully; random out-group
        averages to no preferred direction over many sentences. K is
        spectrally matched to the inner kernel's effective N so the
        random sample's mean has comparable variance to the in-group
        sample's mean.

        Sentences shorter than ``ceil(2*sigma)`` words yield a
        degenerate near-flat inner kernel; skip them.
        """
        word_indices = [self.word_to_idx[w] for w in words
                        if w in self.word_to_idx]
        N = len(word_indices)
        sigma = self.sigma
        if N < max(2, int(2 * sigma)):
            return

        idx = torch.tensor(word_indices, dtype=torch.long, device=self.device)
        vecs = self.model.embeddings(idx)                              # [N, D]
        dev = vecs.device
        dtype = vecs.dtype

        # Inner Gaussian kernel: peaks at d=0, leave-one-out diagonal,
        # row-normalized. Outer kernel is not needed -- the antipode
        # comes from the unique torus per-axis flip of pode.
        pos = torch.arange(N, device=dev, dtype=dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()             # [N, N]
        inner = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
        inner.fill_diagonal_(0.0)
        inner = inner / inner.sum(dim=1, keepdim=True).clamp(min=1e-8)

        pode     = inner @ vecs                                        # [N, D]
        antipode = Lexicon.antipode(pode)                              # [N, D]

        # K spectrally matched to inner kernel's effective N so the
        # random out-group's mean has comparable variance to the
        # in-group's mean. Single .item() per sentence -- one CPU sync.
        n_eff_inner = (1.0 / (inner ** 2).sum(dim=1).clamp(min=1e-8)).mean()
        K = max(1, int(n_eff_inner.round().item()))

        V = self.vocab_size
        out_idx  = torch.randint(0, V, (N, K), device=dev)             # [N, K]
        out_vecs = self.model.embeddings(out_idx)                       # [N, K, D]

        # In-group attraction: each in-sentence word pulled toward its pode.
        in_scores = self.model.embeddings.similarity(pode, vecs)       # [N]
        # Out-group attraction: each random sample pulled toward antipode.
        out_scores = self.model.embeddings.similarity(
            antipode.unsqueeze(1), out_vecs)                            # [N, K]

        # Equal-power pure-attraction: both terms are -logsigmoid(score),
        # minimized when their similarity is high (target close to anchor).
        # No push terms.
        loss = (-F.logsigmoid(in_scores).mean()
                - F.logsigmoid(out_scores).mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.normalize:
            self.model.embeddings.normalize()

        self.n_examples += N
        self._total_loss += loss.item() * N
        self._loss_count += N

    def finish(self):
        """Return trained WordVectors (wrapped into the unit cell)."""
        n = self.vocab_size
        if n == 0:
            return WordVectors.from_vocab([], vector_size=self.vector_size)
        with torch.no_grad():
            vectors = _wrap_unit_ball(self.model.embeddings.weight)
            vectors = vectors.detach().cpu().numpy()
        return WordVectors(vectors, list(self.idx_to_word))


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def project_codebook(wv: "WordVectors", target_dim: int) -> "WordVectors":
    """PCA-project a codebook from its native dim down to ``target_dim``.

    Used by the latent-training pipeline: SBOW trains at a higher
    ``latent_vector_size`` for cleaner dynamics (more room on the torus
    for distinct clusters), then PCA extracts the top ``target_dim``
    principal components so the saved artifact fits the model's
    expected ``<nDim>``.

    PCA operates on chart coordinates. This is fine when the codebook
    is locally clustered -- the regime SBOW reaches with sufficient
    capacity. If clusters span the wrap boundary on some axis, the
    naive PCA on that axis will misread a few-percent of the variance,
    but the principal directions are dominated by content variance, not
    boundary effects.

    Output is rescaled per-axis to ``[-1, 1)`` and re-wrapped via
    ``Lexicon.wrap`` so the saved artifact is in canonical torus form.
    Returns a new :class:`WordVectors` at ``target_dim``; the source is
    unchanged.
    """
    src = wv._vectors.detach().cpu().numpy()
    src_dim = src.shape[1]
    if src_dim == target_dim:
        return wv
    if src_dim < target_dim:
        raise ValueError(
            f"project_codebook: source dim {src_dim} < target {target_dim}; "
            f"cannot project up.")

    from sklearn.decomposition import PCA
    print(f"Phase 1.5: PCA-projecting codebook from D={src_dim} to "
          f"D={target_dim}...", flush=True)
    pca = PCA(n_components=target_dim, whiten=False)
    projected = pca.fit_transform(src)
    var_kept = float(pca.explained_variance_ratio_.sum())
    print(f"  Variance retained: {var_kept:.3f} "
          f"({var_kept * 100:.1f}% of D={src_dim} signal)")

    # Rescale per-axis into [-1, 1), then wrap to canonicalize.
    projected_t = torch.from_numpy(projected).float()
    mn = projected_t.min(dim=0).values
    mx = projected_t.max(dim=0).values
    scale = (mx - mn).clamp(min=1e-8)
    scaled = 2.0 * (projected_t - mn) / scale - 1.0
    scaled = Lexicon.wrap(scaled)

    return WordVectors(scaled.numpy(), list(wv.index_to_key))


DEFAULT_LEARNING_RATE = 0.01


def build_embeddings(shard_paths, output_path, max_docs=10000,
                     vector_size=100, epochs=10, min_count=5,
                     batch_size=256, sigma=5.0, normalize=True,
                     latent_vector_size=None, learning_rate=None):
    """Train SBOW embeddings, optionally at a higher latent dim with PCA.

    When ``latent_vector_size > vector_size``, SBOW trains in the higher
    latent dim where the codebook has more geometric room (1M codes on
    a flat n-torus: density at D=64 is ~1M / 2^64 codes per cell, which
    is essentially unconstrained, vs ~15.6K codes per cell at D=6).
    After training, PCA projects to ``vector_size`` for the final
    artifact. The latent codebook is also saved alongside the projected
    output (sibling path with ``.latent_d{N}.kv``) so subsequent runs
    can resume training in latent space.

    Args:
        latent_vector_size: SBOW training dim. ``None`` means
            ``max(64, vector_size)`` -- training defaults to D=64 for
            small target dims and exactly ``vector_size`` for larger
            ones. Set explicitly to ``vector_size`` to disable
            latent-then-project.
    """
    if latent_vector_size is None:
        latent_vector_size = max(64, vector_size)
    if latent_vector_size < vector_size:
        raise ValueError(
            f"latent_vector_size ({latent_vector_size}) must be >= "
            f"vector_size ({vector_size})")
    use_pca = latent_vector_size != vector_size

    # When projecting, the SBOW trainer's intermediate codebook is saved
    # to a sibling path so resume picks up the unprojected weights. The
    # user-facing output_path always carries the final projected artifact.
    if use_pca:
        if output_path.endswith(".kv"):
            latent_path = output_path[:-3] + f".latent_d{latent_vector_size}.kv"
        else:
            latent_path = output_path + f".latent_d{latent_vector_size}"
    else:
        latent_path = output_path

    # Peek at the existing latent artifact (if any) so we can recover the
    # learning_rate stamped at the previous run's save. This lets the
    # user anneal across runs without restating the rate every invocation:
    # explicit ``--learning-rate`` overrides; otherwise the saved value
    # carries forward; otherwise the package default takes over.
    pre_load_wv = None
    if os.path.exists(latent_path):
        try:
            pre_load_wv = WordVectors.load(latent_path)
        except Exception as exc:
            print(f"Could not load existing embeddings at {latent_path} "
                  f"({type(exc).__name__}: {exc}); starting from scratch.")

    if learning_rate is None:
        if pre_load_wv is not None:
            saved_lr = (getattr(pre_load_wv, "metadata", {}) or {}).get(
                "learning_rate")
            if saved_lr is not None:
                learning_rate = float(saved_lr)
                print(f"Using saved learning_rate={learning_rate} "
                      f"from {latent_path} metadata")
        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATE

    trainer = StreamingSBOWTrainer(
        vector_size=latent_vector_size,
        min_count=min_count,
        learning_rate=learning_rate,
        sigma=sigma,
        normalize=normalize,
    )

    # Resume path: if the latent artifact loaded successfully and matches
    # the configured latent_vector_size, replay its weights into the
    # trainer. Otherwise (dim mismatch or load failure), start fresh.
    resumed = False
    if pre_load_wv is not None:
        if pre_load_wv.vector_size != latent_vector_size:
            print(f"Existing embeddings at {latent_path} have "
                  f"vector_size={pre_load_wv.vector_size} but trainer is at "
                  f"latent_vector_size={latent_vector_size}; "
                  f"starting from scratch.")
        else:
            trainer.load_existing(pre_load_wv)
            resumed = True
            print(f"Resuming from {latent_path}: "
                  f"{trainer.vocab_size} words, "
                  f"{latent_vector_size}-dim vectors, "
                  f"learning_rate={learning_rate}. "
                  f"Skipping Pass 1 (vocab already built).")

    if not resumed:
        # Pass 1: build vocabulary
        print(f"Pass 1: scanning vocabulary from {len(shard_paths)} shard(s), "
              f"max_docs={max_docs}...")
        count = 0
        for doc in iter_documents(shard_paths, max_docs=max_docs):
            trainer.scan_document(doc)
            count += 1
            if count % 100 == 0:
                print(f"  Scanned {count} docs, "
                      f"unique words={len(trainer.word_counts)}", flush=True)
        trainer.build_vocab()

    # Pass 2: stream-train SBOW at the latent dim.
    print(f"Pass 2: training {latent_vector_size}-dim SBOW, "
          f"epochs={epochs}{' (will project to D=' + str(vector_size) + ' after)' if use_pca else ''}...")
    for epoch in range(epochs):
        count = 0
        trainer._total_loss = 0.0
        trainer._loss_count = 0
        for doc in iter_documents(shard_paths, max_docs=max_docs):
            trainer.process_document(doc)
            count += 1
            if count % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: {count} docs, "
                      f"examples={trainer.n_examples}, "
                      f"loss={trainer.avg_loss:.4f}", flush=True)
        print(f"Epoch {epoch+1}/{epochs} complete, loss={trainer.avg_loss:.4f}",
              flush=True)

    wv = trainer.finish()

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Stamp training-time metadata so subsequent resumes pick up the
    # same learning_rate / sigma without restating them on the CLI.
    save_metadata = {
        "learning_rate": float(learning_rate),
        "sigma": float(sigma),
        "latent_vector_size": int(latent_vector_size),
        "vector_size": int(vector_size),
        "epochs_run": int(epochs),
        "n_examples": int(trainer.n_examples),
    }

    if use_pca:
        # Save the latent codebook for future resume / analysis.
        wv.save(latent_path, metadata=dict(save_metadata))
        print(f"Saved latent codebook to {latent_path} "
              f"({len(wv)} words, D={latent_vector_size}, "
              f"learning_rate={learning_rate})")
        # Project to the user-facing target dim.
        wv = project_codebook(wv, vector_size)

    wv.save(output_path, metadata=dict(save_metadata))
    print(f"Saved embeddings to {output_path} ({len(wv)} words, "
          f"D={vector_size})")
    return wv


# ---------------------------------------------------------------------------
# BPE pipeline: discover merges, then distribute vectors over the codebook
#
# Producing a "BPE embedding" is two steps. Phase A learns the merge
# table (the codebook -- a vocabulary of byte-tuples). Phase B learns
# vectors that distribute those codebook slots in semantic space, by
# running SBOW on documents re-tokenized through the frozen merge
# table. The two halves are owned by different modules (``ChunkLayer``
# holds the merges; the embedding matrix lives downstream), so freezing
# one while training the other is just ``chunking_frequency=0``.
#
# The result is a single ``.kv`` artifact with ``kind="both"`` -- BPE
# section carries the merge table, Lexicon section carries the
# chunk-keyed vector matrix.
# ---------------------------------------------------------------------------

def _doc_to_byte_row(text: str, max_seq: int) -> List[int]:
    """Encode text as UTF-8 bytes and pad/truncate to ``max_seq`` length.

    The padding sentinel is 0, which ``ChunkLayer.forward`` interprets
    as end-of-row -- so trailing zeros are safe.
    """
    bts = text.encode("utf-8", errors="ignore")[:max_seq]
    row = list(bts)
    if len(row) < max_seq:
        row.extend([0] * (max_seq - len(row)))
    return row


def discover_bpe(shard_paths, output_path, *,
                 max_docs=10000, n_vectors=4096,
                 chunking_frequency=2, batch_size=32, max_seq=2048,
                 k_merges=4):
    """Phase A: discover BPE merges from a corpus.

    Streams documents, encodes each as UTF-8 bytes, batches into
    ``[B, max_seq]`` tensors, and feeds them through
    ``ChunkLayer.train_step`` until either the document budget is
    exhausted or the codebook reaches ``n_vectors`` entries. Saves the
    resulting merge table as a ``kind=bpe`` artifact.

    No vectors are learned in this phase -- the codebook is purely a
    vocabulary. Phase B (``embed_bpe``) handles vector distribution.
    """
    from Layers import ChunkLayer

    # ChunkLayer holds the codebook only; the nDim arg is unused here
    # (the embedding matrix is allocated in Phase B).
    cl = ChunkLayer(nDim=1, bpe=True,
                    n_vectors=n_vectors,
                    chunking_frequency=chunking_frequency)
    print(f"Phase A: BPE merge discovery from {len(shard_paths)} shard(s), "
          f"max_docs={max_docs}, target n_vectors={n_vectors}, "
          f"chunking_frequency={chunking_frequency}",
          flush=True)
    import time as _time
    t_start = _time.time()
    t_last = t_start

    batch_rows: List[List[int]] = []
    docs_seen = 0
    last_vocab = cl._next_id
    last_print_docs = 0
    PRINT_EVERY = 50  # docs between progress lines
    for doc in iter_documents(shard_paths, max_docs=max_docs):
        row = _doc_to_byte_row(doc, max_seq)
        if not any(row):
            continue
        batch_rows.append(row)
        docs_seen += 1
        if len(batch_rows) >= batch_size:
            tensor = torch.tensor(batch_rows, dtype=torch.long)
            cl.train_step(tensor, k_merges=k_merges)
            batch_rows = []
            if cl._next_id >= n_vectors:
                elapsed = _time.time() - t_start
                print(f"  vocab full ({cl._next_id}/{n_vectors}) at "
                      f"{docs_seen} docs after {elapsed:.1f}s; stopping.",
                      flush=True)
                break
        if docs_seen - last_print_docs >= PRINT_EVERY:
            now = _time.time()
            elapsed = now - t_start
            interval = now - t_last
            d_vocab = cl._next_id - last_vocab
            docs_per_sec = (docs_seen - last_print_docs) / max(interval, 1e-3)
            pct = (docs_seen / max_docs * 100.0) if max_docs else 0.0
            print(f"  [Phase A] {docs_seen}/{max_docs} docs ({pct:.1f}%), "
                  f"vocab={cl._next_id}/{n_vectors} (+{d_vocab}), "
                  f"N={cl._total_pairs}, "
                  f"{docs_per_sec:.1f} docs/s, "
                  f"elapsed={elapsed:.1f}s",
                  flush=True)
            last_print_docs = docs_seen
            last_vocab = cl._next_id
            t_last = now

    # Flush any remaining partial batch.
    if batch_rows and cl._next_id < n_vectors:
        tensor = torch.tensor(batch_rows, dtype=torch.long)
        cl.train_step(tensor, k_merges=k_merges)

    print(f"Phase A done: {docs_seen} docs scanned, "
          f"vocab={cl._next_id}, merges={len(cl.merges)}")

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    cl.save(output_path,
            metadata={"phase": "bpe-discover",
                      "docs_seen": docs_seen,
                      "n_vectors_target": n_vectors})
    print(f"Saved BPE codebook to {output_path} (kind=bpe)")
    return cl


class StreamingChunkSBOWTrainer:
    """SBOW trainer over BPE chunk-id sequences (Phase B).

    Vocabulary is fixed by the loaded ``ChunkLayer`` codebook -- there
    is no Pass 1 vocab scan; the embedding matrix is allocated at
    ``chunk_layer.n_vectors`` immediately and stream-trained per
    sentence. Each document is tokenized through ``chunk_layer.forward``
    on demand.

    ``chunk_layer.chunking_frequency`` is forced to 0 on entry so the
    codebook is held constant throughout this phase -- only the
    vector matrix shifts.
    """

    def __init__(self, chunk_layer, *,
                 vector_size: int = 100,
                 learning_rate: float = 0.001,
                 max_seq: int = 2048,
                 neg_samples: int = 64):
        self.chunk_layer = chunk_layer
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.max_seq = max_seq

        # Freeze the codebook for the duration of Phase B.
        chunk_layer.chunking_frequency = 0

        self.vocab_size = int(chunk_layer.n_vectors)
        # Human-readable keys (latin-1 decode of each chunk's byte tuple)
        # for the saved Lexicon section. Indices align with chunk ids.
        keys = bpe_to_lexicon_keys(chunk_layer)
        while len(keys) < self.vocab_size:
            keys.append("")
        self.idx_to_key = keys

        self.device = util.auto_device()
        print(f"Phase B trainer on {self.device}, "
              f"vocab={self.vocab_size} (chunks)")

        self.model = _SBOWEmbedding(
            self.vocab_size, self.vector_size).to(self.device)
        self.model = util.compile(self.model)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.learning_rate)
        self.neg_samples = int(neg_samples)

        self.n_examples = 0
        self._total_loss = 0.0
        self._loss_count = 0

    @property
    def avg_loss(self):
        return self._total_loss / self._loss_count if self._loss_count else 0.0

    def _tokenize(self, text: str) -> List[int]:
        """Encode text as bytes and run through ChunkLayer to get chunk ids."""
        row = _doc_to_byte_row(text, self.max_seq)
        tensor = torch.tensor([row], dtype=torch.long)
        chunks_per_row, _ = self.chunk_layer.forward(tensor)
        return chunks_per_row[0]

    def process_document(self, text: str) -> None:
        """Train SBOW per sentence over BPE-tokenized text."""
        for sent_text, _ in parse(text, lex='sentences'):
            chunk_ids = self._tokenize(sent_text)
            if len(chunk_ids) >= 2:
                self._train_sentence(chunk_ids)

    def _train_sentence(self, chunk_ids: List[int]) -> None:
        """SBOW via negative sampling: predict each chunk from leave-one-out
        centroid of its sentence-mates. Mirrors StreamingSBOWTrainer's
        method, but operates on chunk ids that are already integer
        indices into the embedding matrix (no string lookup)."""
        idx = torch.tensor(chunk_ids, dtype=torch.long, device=self.device)
        N = idx.shape[0]
        if N < 2:
            return

        vecs = self.model.embeddings(idx)                       # [N, dim]
        total = vecs.sum(dim=0)                                 # [dim]
        centroids = (total.unsqueeze(0) - vecs) / (N - 1)       # [N, dim]

        K = min(self.neg_samples, self.vocab_size - 1)
        pos_vecs = self.model.embeddings(idx)                   # [N, dim]
        pos_scores = self.model.embeddings.similarity(centroids, pos_vecs)    # [N]

        neg_idx = torch.randint(0, self.vocab_size, (N, K),
                                device=self.device)             # [N, K]
        neg_vecs = self.model.embeddings(neg_idx)               # [N, K, dim]
        neg_scores = self.model.embeddings.similarity(
            centroids.unsqueeze(1), neg_vecs)                   # [N, K]

        loss = (-F.logsigmoid(pos_scores).mean()
                - F.logsigmoid(-neg_scores).mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.embeddings.normalize()

        self.n_examples += int(N)
        self._total_loss += float(loss.item()) * int(N)
        self._loss_count += int(N)

    def finish(self) -> "WordVectors":
        """Return a chunk-keyed WordVectors (wrapped into the unit cell)."""
        with torch.no_grad():
            vectors = _wrap_unit_ball(self.model.embeddings.weight).detach().cpu()
        return WordVectors(vectors, list(self.idx_to_key))


def embed_bpe(shard_paths, artifact_path, *,
              output_path=None, max_docs=10000,
              vector_size=100, epochs=10, max_seq=2048,
              learning_rate=0.001):
    """Phase B: train SBOW vectors over a frozen BPE codebook.

    Loads the codebook from ``artifact_path`` (must be ``kind`` in
    ``{"bpe", "both"}``), forces ``chunking_frequency=0``, and trains
    a (n_vectors x vector_size) matrix via SBOW on chunk-id sequences.
    Writes the result to ``output_path`` (defaults to overwriting
    ``artifact_path``) as ``kind=both`` -- merge table + vectors.
    """
    from Layers import ChunkLayer
    output_path = output_path or artifact_path

    cl = ChunkLayer(nDim=1, bpe=True).load(artifact_path)
    print(f"Phase B: loaded BPE codebook ({cl._next_id} chunks of "
          f"{cl.n_vectors} slots) from {artifact_path}; "
          f"freezing codebook for vector training.")

    trainer = StreamingChunkSBOWTrainer(cl,
                                        vector_size=vector_size,
                                        learning_rate=learning_rate,
                                        max_seq=max_seq)

    # Resume path: if output_path already has Lexicon vectors of the
    # right shape (kind=both artifact from a prior Phase B run), copy
    # them into the embedding matrix and continue training from there
    # instead of restarting from a fresh random init each invocation.
    if output_path and os.path.exists(output_path):
        try:
            existing = WordVectors.load(output_path)
        except Exception as exc:
            print(f"Phase B: could not load existing vectors at "
                  f"{output_path} ({type(exc).__name__}: {exc}); "
                  f"starting from random init.")
        else:
            expected_shape = (trainer.vocab_size, vector_size)
            if (existing.vector_size == vector_size
                    and len(existing) == trainer.vocab_size):
                with torch.no_grad():
                    saved = existing._vectors.to(trainer.device)
                    target = trainer.model.embeddings.weight
                    if saved.shape == target.shape:
                        target.copy_(saved)
                        print(f"Phase B: resumed from {output_path} "
                              f"({len(existing)} chunks, "
                              f"{vector_size}-dim).")
                    else:
                        print(f"Phase B: existing vectors at "
                              f"{output_path} have shape "
                              f"{tuple(saved.shape)}, expected "
                              f"{expected_shape}; starting fresh.")
            else:
                print(f"Phase B: existing vectors at {output_path} "
                      f"don't match (have {len(existing)} x "
                      f"{existing.vector_size}, expected "
                      f"{trainer.vocab_size} x {vector_size}); "
                      f"starting fresh.")

    print(f"Phase B: training {vector_size}-dim SBOW over BPE chunks, "
          f"epochs={epochs}",
          flush=True)
    import time as _time
    t_start = _time.time()
    PRINT_EVERY = 50
    for epoch in range(epochs):
        count = 0
        trainer._total_loss = 0.0
        trainer._loss_count = 0
        last_print = 0
        t_last = _time.time()
        for doc in iter_documents(shard_paths, max_docs=max_docs):
            trainer.process_document(doc)
            count += 1
            if count - last_print >= PRINT_EVERY:
                now = _time.time()
                elapsed = now - t_start
                interval = now - t_last
                docs_per_sec = (count - last_print) / max(interval, 1e-3)
                pct = (count / max_docs * 100.0) if max_docs else 0.0
                print(f"  [Phase B] Epoch {epoch+1}/{epochs}, "
                      f"{count}/{max_docs} docs ({pct:.1f}%), "
                      f"examples={trainer.n_examples}, "
                      f"loss={trainer.avg_loss:.4f}, "
                      f"{docs_per_sec:.1f} docs/s, "
                      f"elapsed={elapsed:.1f}s",
                      flush=True)
                last_print = count
                t_last = now
        print(f"Epoch {epoch+1}/{epochs} done, "
              f"loss={trainer.avg_loss:.4f}, "
              f"elapsed={_time.time() - t_start:.1f}s",
              flush=True)

    wv = trainer.finish()
    bpe_section = bpe_section_from_chunk_layer(cl)

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    wv.save(output_path,
            bpe_section=bpe_section,
            metadata={"phase": "bpe-embed",
                      "vector_size": int(vector_size),
                      "epochs": int(epochs)})
    print(f"Saved BPE+vectors to {output_path} (kind=both, "
          f"{len(wv)} chunks, {vector_size} dims)")
    return wv


def train_bpe(shard_paths, output_path, *,
              max_docs=10000, n_vectors=4096,
              chunking_frequency=2, vector_size=100, epochs=10,
              batch_size=32, max_seq=2048, learning_rate=0.001):
    """Convenience: Phase A then Phase B in one invocation."""
    discover_bpe(shard_paths, output_path,
                 max_docs=max_docs, n_vectors=n_vectors,
                 chunking_frequency=chunking_frequency,
                 batch_size=batch_size, max_seq=max_seq)
    return embed_bpe(shard_paths, output_path,
                     max_docs=max_docs,
                     vector_size=vector_size, epochs=epochs,
                     max_seq=max_seq, learning_rate=learning_rate)


def prune_embeddings(path, min_frequency, output=None):
    """Remove words from a .kv codebook whose frequency is below min_frequency.

    Words are kept when ``count / total_count >= min_frequency``.
    If ``total_count`` is zero (no frequency data), no pruning is performed.

    Args:
        path: path to the .kv file to prune
        min_frequency: minimum frequency ratio for retention (e.g. 0.00001)
        output: save path; defaults to overwriting the input file
    """
    wv = WordVectors.load(path)
    total = int(wv.total_count)

    if total == 0:
        print(f"No frequency data in {path} (total_count=0) -- nothing to prune.")
        return

    prune_indices = [i for i, word in enumerate(wv.index_to_key)
                     if wv.counts[i] / total < min_frequency]
    old_size = len(wv)
    n_pruned = wv.remove(prune_indices)
    print(f"Vocabulary: {old_size} -> {len(wv)}  ({n_pruned} pruned below {min_frequency})")

    out_path = output or path
    wv.save(out_path)
    print(f"Saved to {out_path}")


def _find_embeddings(path=None):
    """Locate and load a .pt embedding file from standard paths."""
    candidates = [path] if path else [
        "output/BasicModel.kv",
        "output/embeddings/sentence.pt",
        "data/sentence.pt",
        "sentence.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return WordVectors.load(p)
    print(f"No embedding file found. Searched: {candidates}", file=sys.stderr)
    print("Run 'make basic_train' to train embeddings first.", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Word embeddings: train or explore")
    sub = parser.add_subparsers(dest="command")

    # --- train subcommand ---
    train_p = sub.add_parser("train", help="Train CBOW embeddings from FineWeb-EDU")
    train_p.add_argument('--output', default='output/BasicModel.kv')
    train_p.add_argument('--data-dir', default=None)
    train_p.add_argument('--num-shards', type=int, default=1)
    train_p.add_argument('--max-docs', type=int, default=10000)
    train_p.add_argument('--vector-size', type=int, default=100)
    train_p.add_argument('--latent-vector-size', type=int, default=None,
                         help='SBOW training dim. Defaults to '
                              'max(64, vector-size). When greater than '
                              'vector-size, SBOW trains at this higher dim '
                              'for cleaner cluster dynamics on the flat '
                              'n-torus, then PCA projects the codebook '
                              'down to vector-size for the saved artifact. '
                              'Set explicitly to vector-size to disable '
                              'latent-then-project.')
    train_p.add_argument('--learning-rate', '--lr', type=float, default=None,
                         dest='learning_rate',
                         help='SGD learning rate for SBOW training. '
                              'When omitted, falls back to the value '
                              'stamped in the existing .kv metadata '
                              '(if resuming) or 0.01 (fresh run). For '
                              'manual annealing, pass a smaller rate on '
                              'each successive resume; it overrides the '
                              'saved value and gets persisted for the '
                              'next run.')
    train_p.add_argument('--epochs', type=int, default=10)
    train_p.add_argument('--min-count', type=int, default=5)
    train_p.add_argument('--batch-size', type=int, default=256)
    train_p.add_argument('--sigma', type=float, default=5.0,
                         help='Inner Gaussian width (position units). '
                              'The outer kernel is its complement. '
                              '5.0 matches Mikolov skip-gram window. '
                              'Larger values blend more topic into the pode.')
    train_p.add_argument('--no-normalize', dest='normalize',
                         action='store_false',
                         help='Skip the per-step torus wrap. Default '
                              'wraps embeddings into [-1, 1)^D after '
                              'every optimizer step (canonical for the '
                              'flat n-torus codebook).')
    train_p.set_defaults(normalize=True)
    train_p.add_argument('--random-shards', action='store_true',
                         help='Randomly select which shards to download')
    train_p.add_argument('--compile', default=None,
                         metavar='BACKEND',
                         help='Compilation backend: none, inductor, eager, aot_eager. '
                              'Overrides MODEL_COMPILE env var.')

    # --- BPE subcommands -------------------------------------------------
    # Phase A: discover merge table from a corpus
    bpe_disc_p = sub.add_parser(
        "bpe-discover",
        help="Phase A: discover BPE merges from a corpus (no vectors).")
    bpe_disc_p.add_argument('--output', default='output/BasicModel.kv')
    bpe_disc_p.add_argument('--data-dir', default=None)
    bpe_disc_p.add_argument('--num-shards', type=int, default=1)
    bpe_disc_p.add_argument('--max-docs', type=int, default=10000)
    bpe_disc_p.add_argument('--n-vectors', type=int, default=4096)
    bpe_disc_p.add_argument('--chunking-frequency', type=int, default=2)
    bpe_disc_p.add_argument('--batch-size', type=int, default=32)
    bpe_disc_p.add_argument('--max-seq', type=int, default=2048)
    bpe_disc_p.add_argument('--k-merges', type=int, default=4,
                             help='Promotions per train_step batch.')
    bpe_disc_p.add_argument('--random-shards', action='store_true')

    # Phase B: train vectors over a frozen BPE codebook
    bpe_emb_p = sub.add_parser(
        "bpe-embed",
        help="Phase B: train SBOW vectors over a frozen BPE codebook.")
    bpe_emb_p.add_argument('--input', required=True,
                            help='kind=bpe or kind=both artifact to load.')
    bpe_emb_p.add_argument('--output', default=None,
                            help='Output path (defaults to --input, overwriting it).')
    bpe_emb_p.add_argument('--data-dir', default=None)
    bpe_emb_p.add_argument('--num-shards', type=int, default=1)
    bpe_emb_p.add_argument('--max-docs', type=int, default=10000)
    bpe_emb_p.add_argument('--vector-size', type=int, default=100)
    bpe_emb_p.add_argument('--epochs', type=int, default=10)
    bpe_emb_p.add_argument('--max-seq', type=int, default=2048)
    bpe_emb_p.add_argument('--learning-rate', type=float, default=0.001)
    bpe_emb_p.add_argument('--random-shards', action='store_true')

    # Convenience: A then B in one go
    bpe_train_p = sub.add_parser(
        "bpe-train",
        help="Convenience: Phase A (discover) then Phase B (embed).")
    bpe_train_p.add_argument('--output', default='output/BasicModel.kv')
    bpe_train_p.add_argument('--data-dir', default=None)
    bpe_train_p.add_argument('--num-shards', type=int, default=1)
    bpe_train_p.add_argument('--max-docs', type=int, default=10000)
    bpe_train_p.add_argument('--n-vectors', type=int, default=4096)
    bpe_train_p.add_argument('--chunking-frequency', type=int, default=2)
    bpe_train_p.add_argument('--vector-size', type=int, default=100)
    bpe_train_p.add_argument('--epochs', type=int, default=10)
    bpe_train_p.add_argument('--batch-size', type=int, default=32)
    bpe_train_p.add_argument('--max-seq', type=int, default=2048)
    bpe_train_p.add_argument('--learning-rate', type=float, default=0.001)
    bpe_train_p.add_argument('--random-shards', action='store_true')

    # --- prune subcommand ---
    prune_p = sub.add_parser("prune", help="Remove low-frequency words from a .kv codebook")
    prune_p.add_argument("filename", help="Path to the .kv file to prune")
    prune_p.add_argument("--min-frequency", type=float, required=True,
                         help="Minimum frequency ratio for retention (e.g. 0.00001). "
                              "Matches <minFrequency> in the XML config.")
    prune_p.add_argument("--output", "-o", default=None,
                         help="Output path (default: overwrite input file)")

    # --- explore subcommand ---
    explore_p = sub.add_parser("explore", help="Explore trained embeddings")
    explore_p.add_argument("words", nargs="*", help="Word(s) to look up")
    explore_p.add_argument("--path", "-p", default=None, help="Path to .pt file")
    explore_p.add_argument("--topn", "-n", type=int, default=10)
    explore_p.add_argument("--vocab", action="store_true", help="Print vocabulary stats")

    args = parser.parse_args()

    if args.command == "prune":
        prune_embeddings(args.filename, args.min_frequency, output=args.output)

    elif args.command == "explore":
        import random
        wv = _find_embeddings(args.path)
        if args.vocab:
            print(f"Vocabulary: {len(wv)} words, {wv.vector_size}-dim vectors")
            print(f"Sample: {', '.join(random.sample(wv.index_to_key, min(20, len(wv))))}")
        else:
            wv.explore(args.words, topn=args.topn)

    elif args.command == "train":
        if args.compile is not None:
            util.init_compile_backend(args.compile)

        data_dir = args.data_dir
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data" / "fineweb")

        shard_paths = get_shard_paths(data_dir, num_shards=args.num_shards,
                                      random_select=args.random_shards)
        if not shard_paths:
            print("No shards available.")
            sys.exit(1)

        build_embeddings(
            shard_paths=shard_paths,
            output_path=args.output,
            max_docs=args.max_docs,
            vector_size=args.vector_size,
            epochs=args.epochs,
            min_count=args.min_count,
            batch_size=args.batch_size,
            sigma=args.sigma,
            normalize=args.normalize,
            latent_vector_size=args.latent_vector_size,
            learning_rate=args.learning_rate,
        )

    elif args.command in ("bpe-discover", "bpe-embed", "bpe-train"):
        data_dir = args.data_dir
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data" / "fineweb")
        shard_paths = get_shard_paths(data_dir, num_shards=args.num_shards,
                                      random_select=args.random_shards)
        if not shard_paths:
            print("No shards available.")
            sys.exit(1)

        if args.command == "bpe-discover":
            discover_bpe(
                shard_paths=shard_paths,
                output_path=args.output,
                max_docs=args.max_docs,
                n_vectors=args.n_vectors,
                chunking_frequency=args.chunking_frequency,
                batch_size=args.batch_size,
                max_seq=args.max_seq,
                k_merges=args.k_merges,
            )
        elif args.command == "bpe-embed":
            embed_bpe(
                shard_paths=shard_paths,
                artifact_path=args.input,
                output_path=args.output,
                max_docs=args.max_docs,
                vector_size=args.vector_size,
                epochs=args.epochs,
                max_seq=args.max_seq,
                learning_rate=args.learning_rate,
            )
        else:  # bpe-train
            train_bpe(
                shard_paths=shard_paths,
                output_path=args.output,
                max_docs=args.max_docs,
                n_vectors=args.n_vectors,
                chunking_frequency=args.chunking_frequency,
                vector_size=args.vector_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_seq=args.max_seq,
                learning_rate=args.learning_rate,
            )

    else:
        parser.print_help()
