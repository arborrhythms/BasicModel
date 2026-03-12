"""Lightweight word-vector store using plain NumPy and PyTorch.

Drop-in replacement for the gensim Word2Vec / KeyedVectors API surface
used by LanguageModel in BasicModel.py.  Supports:

  - Building random embeddings from a vocabulary list
  - Loading word2vec-format text files (e.g. enwiki_20180420_100d.txt)
  - Save / load via torch.save / torch.load
  - Vector lookup by word, membership test, normalized vectors
  - Nearest-neighbour (most_similar) search by cosine distance
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


class WordVectors:
    """Stores word embeddings as a NumPy matrix with word ↔ index mappings."""

    def __init__(self, vectors: np.ndarray, index_to_key: List[str]):
        """Create from a (vocab_size, vector_size) matrix and word list."""
        assert len(index_to_key) == vectors.shape[0]
        self._vectors = vectors.astype(np.float32)
        self.index_to_key = list(index_to_key)
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        self._normed: Optional[np.ndarray] = None

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def from_vocab(cls, words: List[str], vector_size: int = 20) -> "WordVectors":
        """Build random unit-normalised embeddings for a vocabulary list."""
        unique = list(dict.fromkeys(words))  # preserve order, deduplicate
        vecs = np.random.randn(len(unique), vector_size).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms
        return cls(vecs, unique)

    @classmethod
    def load_word2vec_format(cls, path: str) -> "WordVectors":
        """Load vectors from a word2vec text-format file.

        First line: ``<vocab_size> <vector_size>``
        Subsequent lines: ``<word> <float> <float> ...``
        """
        words: List[str] = []
        vecs: List[np.ndarray] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().split()
            vocab_size, vector_size = int(header[0]), int(header[1])
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != vector_size + 1:
                    continue  # skip malformed lines
                words.append(parts[0])
                vecs.append(np.array(parts[1:], dtype=np.float32))
        return cls(np.stack(vecs), words)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save vectors and vocabulary to a .pt file."""
        torch.save({
            "vectors": self._vectors,
            "index_to_key": self.index_to_key,
        }, path)

    @classmethod
    def load(cls, path: str) -> "WordVectors":
        """Load from a .pt file saved by ``save()``."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(np.asarray(data["vectors"], dtype=np.float32),
                   data["index_to_key"])

    # ── Vector access ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the raw (unnormalised) vector for *word*."""
        return self._vectors[self.key_to_index[word]]

    def get_normed_vectors(self) -> np.ndarray:
        """Return all vectors L2-normalised (cached)."""
        if self._normed is None:
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._normed = self._vectors / norms
        return self._normed

    # ── Similarity ───────────────────────────────────────────────────

    def most_similar(self, positive: np.ndarray,
                     topn: int = 1) -> List[Tuple[str, float]]:
        """Find the *topn* words closest to *positive* by cosine similarity."""
        positive = np.asarray(positive, dtype=np.float32).ravel()
        norm = np.linalg.norm(positive)
        if norm > 0:
            positive = positive / norm
        normed = self.get_normed_vectors()
        sims = normed @ positive
        # argpartition is faster than full sort for large vocabs
        if topn < len(sims):
            top_idx = np.argpartition(-sims, topn)[:topn]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        else:
            top_idx = np.argsort(-sims)[:topn]
        return [(self.index_to_key[i], float(sims[i])) for i in top_idx]
