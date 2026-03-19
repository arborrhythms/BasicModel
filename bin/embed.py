"""Word embedding pipeline: FineWeb-EDU → lex → parse → CBOW → BasicModel.kv

Training phase that produces a static word embedding artifact. InputSpace
loads this artifact at startup — it does not train.

Pipeline stages:
  1. Stream text documents from FineWeb-EDU parquet shards
  2. Lex + parse_buffer: tokenize words, group into sentences
  3. Build (target, context) training examples per sentence
  4. Train CBOW embeddings: predict target from mean of context vectors
  5. Save as WordVectors artifact (.kv, gensim-compatible KeyedVectors)

Usage:
    python bin/embed.py --output output/BasicModel.kv \
        --num-shards 1 --max-docs 10000 --vector-size 100 --epochs 10
"""

import os
import sys
import time
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Optional

from parse import parse_buffer, set_sentence_cfg


def _get_device():
    """Select best available device, respecting BASICMODEL_DEVICE env var."""
    from util import resolve_device
    env = os.environ.get("BASICMODEL_DEVICE", "").strip().lower()
    if env:
        return resolve_device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_inductor_paths():
    """Replace iCloud paths (with spaces) with /bits symlink in inductor commands."""
    try:
        from torch._inductor import cpp_builder
        _orig = cpp_builder._run_compile_cmd
        _bits = "/bits"
        if os.path.islink(_bits):
            _target = os.readlink(_bits)
            def _patched(cmd_line, cwd):
                return _orig(cmd_line.replace(_target, _bits), cwd)
            cpp_builder._run_compile_cmd = _patched
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Word vector store
# ---------------------------------------------------------------------------

class WordVectors:
    """Stores word embeddings as a numpy array with word <-> index mappings.

    API-compatible with gensim's KeyedVectors: ``wv.vectors``, ``wv.vector_size``,
    ``wv[word]``, ``wv.key_to_index``, ``wv.index_to_key``, ``wv.most_similar()``.
    """

    def __init__(self, vectors, index_to_key: List[str],
                 counts=None, total_count: int = 0):
        """Create from a (vocab_size, vector_size) array/tensor and word list."""
        assert len(index_to_key) == vectors.shape[0]
        if isinstance(vectors, torch.Tensor):
            self._vectors: np.ndarray = vectors.detach().cpu().float().numpy()
        else:
            self._vectors = np.asarray(vectors, dtype=np.float32)
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
    def vectors(self) -> np.ndarray:
        """Raw embedding matrix (vocab_size, vector_size) — gensim compat."""
        return self._vectors

    @property
    def vector_size(self) -> int:
        """Dimensionality of each vector — gensim compat."""
        return self._vectors.shape[1]

    # -- Factory methods --

    @classmethod
    def from_vocab(cls, words: List[str], vector_size: int = 20) -> "WordVectors":
        """Build random unit-normalised embeddings for a vocabulary list."""
        unique = list(dict.fromkeys(words))  # preserve order, deduplicate
        vecs = torch.randn(len(unique), vector_size)
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return cls(vecs.numpy(), unique)

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
        return cls(np.array(vecs, dtype=np.float32), words)

    def remove(self, indices):
        """Remove entries by index, returning the pruned count.

        Shrinks vectors, counts, and vocabulary mappings in-place.
        """
        if not indices:
            return 0
        removed_set = set(indices)
        new_keys = [w for i, w in enumerate(self.index_to_key) if i not in removed_set]
        self._vectors = np.delete(self._vectors, indices, axis=0)
        self.counts = np.delete(self.counts, indices)
        self.index_to_key = new_keys
        self.key_to_index = {w: i for i, w in enumerate(new_keys)}
        self._normed = None
        return len(removed_set)

    # -- Persistence --

    def save(self, path: str) -> None:
        """Save vectors, vocabulary, and word frequencies to a .pt file."""
        torch.save({
            "vectors": self._vectors,
            "index_to_key": self.index_to_key,
            "counts": self.counts,
            "total_count": int(self.total_count),
        }, path)

    @classmethod
    def load(cls, path: str) -> "WordVectors":
        """Load from a .pt file saved by ``save()``."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        vectors = data["vectors"]
        # Migrate old files that stored torch tensors
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.float().numpy()
        counts = data.get("counts")
        total_count = data.get("total_count", 0)
        return cls(vectors, data["index_to_key"],
                   counts=counts, total_count=total_count)

    # -- Vector access --

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the raw (unnormalised) vector for *word* — gensim compat."""
        return self._vectors[self.key_to_index[word]]

    def get_normed_vectors(self) -> torch.Tensor:
        """Return all vectors L2-normalised on TheDevice (cached)."""
        if self._normed is None:
            from util import TheDevice
            t = torch.as_tensor(self._vectors).to(TheDevice)
            self._normed = torch.nn.functional.normalize(t, p=2, dim=1)
        return self._normed

    # -- Similarity --

    def most_similar(self, positive,
                     topn: int = 1) -> List[Tuple[str, float]]:
        """Find the *topn* words closest to *positive* by cosine similarity."""
        if not isinstance(positive, torch.Tensor):
            positive = torch.as_tensor(positive, dtype=torch.float32)
        normed = self.get_normed_vectors()
        positive = positive.float().flatten().to(normed.device)
        norm = positive.norm()
        if norm > 0:
            positive = positive / norm
        sims = normed @ positive
        if topn < len(sims):
            _, top_idx = sims.topk(topn)
        else:
            _, top_idx = sims.sort(descending=True)
        return [(self.index_to_key[i], float(sims[i])) for i in top_idx]

    def similarity(self, word1: str, word2: str) -> float:
        """Return cosine similarity between two words."""
        from util import TheDevice
        v1 = torch.as_tensor(self[word1], dtype=torch.float32).to(TheDevice)
        v2 = torch.as_tensor(self[word2], dtype=torch.float32).to(TheDevice)
        return float(torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)))

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

def get_shard_paths(data_dir, num_shards=1):
    """Ensure shards are downloaded and return their file paths."""
    os.makedirs(data_dir, exist_ok=True)
    paths = []
    for i in range(min(num_shards, MAX_SHARD + 1)):
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
# CBOW embedding trainer (streaming)
# ---------------------------------------------------------------------------

class CBOWModel:
    """Stateful CBOW model: embedding + linear head + optimizer.

    Supports both bulk training (``train``) and incremental single-sentence
    updates (``train_step``).  The same ``nn.Embedding`` and optimizer are
    reused across calls so that gradient state is preserved.
    """

    def __init__(self, wv: WordVectors, learning_rate=0.01, neg_samples=64):
        """Initialise from an existing WordVectors.

        Uses negative sampling instead of a full-softmax linear head,
        keeping MPS/GPU memory proportional to ``neg_samples`` rather
        than ``vocab_size``.
        """
        self.index_to_key = list(wv.index_to_key)
        self.key_to_index = dict(wv.key_to_index)
        vocab_size = len(self.index_to_key)
        vector_size = wv.vector_size
        self.neg_samples = neg_samples

        self.embeddings = nn.Embedding(vocab_size, vector_size)
        with torch.no_grad():
            self.embeddings.weight.copy_(torch.as_tensor(wv._vectors, dtype=torch.float32))

        self.optimizer = optim.Adam(
            self.embeddings.parameters(),
            lr=learning_rate,
        )

        # Per-word gradient variance (sigma) — lazy-initialized in observe_sigma
        self.sigma = None
        self.sigma_mean = None
        self.sigma_step = 0
        self.sigma_beta = 0.99

    @torch.no_grad()
    def observe_sigma(self, word_indices):
        """Track per-word gradient variance via Welford's algorithm.

        Called after backward() and before optimizer.step() so that
        embedding.weight.grad is available.  Works regardless of which
        training method (CBOW, SBOW, etc.) produced the gradients.
        """
        grad = self.embeddings.weight.grad
        if grad is None:
            return
        device = grad.device
        vocab_size = self.embeddings.weight.shape[0]
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
        """Negative sampling loss: −log σ(q·w⁺) − Σ log σ(−q·wₖ⁻).

        Args:
            queries: [N, dim] query vectors (context centroids).
            target_idx: [N] indices of positive (target) words.
        Returns:
            Scalar loss.
        """
        device = queries.device
        vocab_size = self.embeddings.weight.shape[0]
        K = min(self.neg_samples, vocab_size - 1)

        pos_vecs = self.embeddings(target_idx)                        # [N, dim]
        pos_scores = (queries * pos_vecs).sum(dim=1)                  # [N]

        neg_idx = torch.randint(0, vocab_size, (queries.shape[0], K),
                                device=device)                        # [N, K]
        neg_vecs = self.embeddings(neg_idx)                           # [N, K, dim]
        neg_scores = torch.bmm(neg_vecs,
                               queries.unsqueeze(2)).squeeze(2)       # [N, K]

        return -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()

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

        device = self.embeddings.weight.device

        max_ctx = max(len(c) for c in contexts)
        n = len(targets)
        ctx_padded = torch.zeros(n, max_ctx, dtype=torch.long, device=device)
        ctx_mask = torch.zeros(n, max_ctx, device=device)
        for i, c in enumerate(contexts):
            ctx_padded[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)
            ctx_mask[i, :len(c)] = 1.0
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)

        ctx_embeds = self.embeddings(ctx_padded)
        masked = ctx_embeds * ctx_mask.unsqueeze(-1)
        ctx_mean = masked.sum(dim=1) / ctx_mask.sum(dim=1, keepdim=True)

        loss = self._neg_sampling_loss(ctx_mean, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.observe_sigma(targets)
        self.optimizer.step()

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

        device = self.embeddings.weight.device
        idx = torch.tensor(word_indices, dtype=torch.long, device=device)
        N = len(word_indices)

        vecs = self.embeddings(idx)                       # [N, dim]
        total = vecs.sum(dim=0)                           # [dim]
        centroids = (total.unsqueeze(0) - vecs) / (N - 1) # [N, dim]

        loss = self._neg_sampling_loss(centroids, idx)

        self.optimizer.zero_grad()
        loss.backward()
        self.observe_sigma(word_indices)
        self.optimizer.step()

        return loss.item()

    # -- export ----------------------------------------------------------------

    def to_word_vectors(self):
        """Snapshot current embedding weights as a new WordVectors."""
        with torch.no_grad():
            vectors = self.embeddings.weight.detach().cpu().numpy()
        return WordVectors(vectors, self.index_to_key)


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
    """SBOW embedding: just an embedding table, no linear head."""

    def __init__(self, vocab_size, vector_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vector_size)
        # Initialise on the unit sphere
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight)
            self.embeddings.weight.copy_(
                torch.nn.functional.normalize(self.embeddings.weight, p=2, dim=1)
            )


class StreamingSBOWTrainer:
    """Two-pass SBOW trainer: build vocab first, then stream-train per sentence.

    Pass 1: Stream documents to count words and build vocabulary.
    Pass 2: Stream documents again, train SBOW per sentence — each word
            predicted from its leave-one-out centroid via full softmax.
    The model is allocated once at its final size.  Uses SGD to avoid
    the 2x memory overhead of Adam's momentum buffers.
    """

    def __init__(self, vector_size=100, min_count=5, learning_rate=0.001):
        self.vector_size = vector_size
        self.min_count = min_count
        self.learning_rate = learning_rate

        self.word_counts = Counter()
        self.word_to_idx = {}
        self.idx_to_word = []

        self.device = _get_device()
        print(f"Training on {self.device}")

        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

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
        """Parse one document and count words (no training)."""
        pos = 0
        while pos < len(text):
            result, next_pos = parse_buffer(text, pos)
            if next_pos == pos:
                break
            pos = next_pos
            for sent in result['sentences']:
                for t in sent['tokens']:
                    self.word_counts[t['text']] += 1

    def build_vocab(self):
        """Promote words that meet min_count and allocate the model."""
        for word, count in self.word_counts.items():
            if count >= self.min_count:
                self.word_to_idx[word] = len(self.idx_to_word)
                self.idx_to_word.append(word)
        self.model = _CBOWModule(self.vocab_size, self.vector_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(f"  Vocab: {len(self.word_counts)} unique words -> "
              f"{self.vocab_size} after min_count={self.min_count}")

    # -- Pass 2: streaming SBOW training --------------------------------------

    def process_document(self, text):
        """Parse one document, train SBOW on each sentence immediately."""
        pos = 0
        while pos < len(text):
            result, next_pos = parse_buffer(text, pos)
            if next_pos == pos:
                break
            pos = next_pos

            for sent in result['sentences']:
                words = [t['text'] for t in sent['tokens']]
                self._train_sentence(words)

    def _train_sentence(self, words):
        """SBOW: predict each word from leave-one-out centroid via full softmax."""
        word_indices = [self.word_to_idx[w] for w in words
                        if w in self.word_to_idx]
        if len(word_indices) < 2:
            return

        idx = torch.tensor(word_indices, dtype=torch.long, device=self.device)
        N = len(word_indices)

        vecs = self.model.embeddings(idx)                       # [N, dim]
        total = vecs.sum(dim=0)                                 # [dim]
        centroids = (total.unsqueeze(0) - vecs) / (N - 1)       # [N, dim]

        logits = self.model.linear(centroids)                   # [N, vocab]
        loss = self.loss_fn(logits, idx)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_examples += N
        self._total_loss += loss.item() * N
        self._loss_count += N

    def finish(self):
        """Return trained WordVectors."""
        n = self.vocab_size
        if n == 0:
            return WordVectors.from_vocab([], vector_size=self.vector_size)
        with torch.no_grad():
            vectors = self.model.embeddings.weight.detach().cpu().numpy()
        return WordVectors(vectors, list(self.idx_to_word))


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def build_embeddings(shard_paths, output_path, max_docs=10000,
                     vector_size=100, epochs=10, min_count=5,
                     batch_size=256):
    trainer = StreamingSBOWTrainer(
        vector_size=vector_size,
        min_count=min_count,
        learning_rate=0.001,
    )

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

    # Pass 2: stream-train SBOW (full softmax, per-sentence)
    print(f"Pass 2: training {vector_size}-dim SBOW, epochs={epochs}...")
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
    wv.save(output_path)
    print(f"Saved embeddings to {output_path} ({len(wv)} words, {vector_size} dims)")
    return wv


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
        print(f"No frequency data in {path} (total_count=0) — nothing to prune.")
        return

    prune_indices = [i for i, word in enumerate(wv.index_to_key)
                     if wv.counts[i] / total < min_frequency]
    old_size = len(wv)
    n_pruned = wv.remove(prune_indices)
    print(f"Vocabulary: {old_size} → {len(wv)}  ({n_pruned} pruned below {min_frequency})")

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
    train_p.add_argument('--config', default='data/sentence.cfg',
                         help='Sentence grammar config used by parse_buffer')
    train_p.add_argument('--output', default='output/BasicModel.kv')
    train_p.add_argument('--data-dir', default=None)
    train_p.add_argument('--num-shards', type=int, default=1)
    train_p.add_argument('--max-docs', type=int, default=10000)
    train_p.add_argument('--vector-size', type=int, default=100)
    train_p.add_argument('--epochs', type=int, default=10)
    train_p.add_argument('--min-count', type=int, default=5)
    train_p.add_argument('--batch-size', type=int, default=256)

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
        data_dir = args.data_dir
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data" / "fineweb")

        print(f"Config: {args.config}")
        set_sentence_cfg(args.config)

        shard_paths = get_shard_paths(data_dir, num_shards=args.num_shards)
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
        )

    else:
        parser.print_help()
