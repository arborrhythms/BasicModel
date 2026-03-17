"""Word embedding pipeline: FineWeb-EDU → lex → parse → CBOW → sentence.pt

Training phase that produces a static word embedding artifact. InputSpace
loads this artifact at startup — it does not train.

Pipeline stages:
  1. Stream text documents from FineWeb-EDU parquet shards
  2. Lex + parse_buffer: tokenize words, group into sentences
  3. Build (target, context) training examples per sentence
  4. Train CBOW embeddings: predict target from mean of context vectors
  5. Save as WordVectors artifact (.pt)

Usage:
    python bin/embed.py --output output/embeddings/sentence.pt \
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
import torch.optim as optim
from typing import List, Tuple, Optional

from parse import parse_buffer, set_sentence_cfg


# ---------------------------------------------------------------------------
# Word vector store
# ---------------------------------------------------------------------------

class WordVectors:
    """Stores word embeddings as a torch tensor with word <-> index mappings.

    Drop-in replacement for the gensim Word2Vec / KeyedVectors API surface
    used by Embedding in BasicModel.py.
    """

    def __init__(self, vectors, index_to_key: List[str]):
        """Create from a (vocab_size, vector_size) tensor/array and word list."""
        assert len(index_to_key) == vectors.shape[0]
        if isinstance(vectors, torch.Tensor):
            self._vectors = vectors.float()
        else:
            self._vectors = torch.as_tensor(vectors, dtype=torch.float32)
        self.index_to_key = list(index_to_key)
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        self._normed: Optional[torch.Tensor] = None

    # -- Factory methods --

    @classmethod
    def from_vocab(cls, words: List[str], vector_size: int = 20) -> "WordVectors":
        """Build random unit-normalised embeddings for a vocabulary list."""
        unique = list(dict.fromkeys(words))  # preserve order, deduplicate
        vecs = torch.randn(len(unique), vector_size)
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
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

    # -- Persistence --

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
        return cls(data["vectors"], data["index_to_key"])

    # -- Vector access --

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, word: str) -> torch.Tensor:
        """Return the raw (unnormalised) vector for *word*."""
        return self._vectors[self.key_to_index[word]]

    def get_normed_vectors(self) -> torch.Tensor:
        """Return all vectors L2-normalised (cached)."""
        if self._normed is None:
            self._normed = torch.nn.functional.normalize(self._vectors, p=2, dim=1)
        return self._normed

    # -- Similarity --

    def most_similar(self, positive,
                     topn: int = 1) -> List[Tuple[str, float]]:
        """Find the *topn* words closest to *positive* by cosine similarity."""
        if not isinstance(positive, torch.Tensor):
            positive = torch.as_tensor(positive, dtype=torch.float32)
        positive = positive.float().flatten()
        norm = positive.norm()
        if norm > 0:
            positive = positive / norm
        normed = self.get_normed_vectors()
        sims = normed @ positive
        if topn < len(sims):
            _, top_idx = sims.topk(topn)
        else:
            _, top_idx = sims.sort(descending=True)
        return [(self.index_to_key[i], float(sims[i])) for i in top_idx]

    def similarity(self, word1: str, word2: str) -> float:
        """Return cosine similarity between two words."""
        v1 = self[word1].float()
        v2 = self[word2].float()
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
# Corpus builder
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """Accumulates vocabulary and word-in-sentence examples from streamed text."""

    def __init__(self):
        self.vocab = set()
        self.vocab_counts = Counter()
        self.examples = []  # list of {'target': str, 'context': [str]}

    def process_document(self, text):
        """Lex + parse a document, accumulate vocabulary + training examples."""
        pos = 0
        while pos < len(text):
            result, next_pos = parse_buffer(text, pos)
            if next_pos == pos:
                break  # no progress -- remaining text has no sentence separator
            pos = next_pos

            for sent in result['sentences']:
                # sent['tokens'] are the WORD tokens in this sentence
                words = [t['text'] for t in sent['tokens']]

                for w in words:
                    self.vocab.add(w)
                    self.vocab_counts[w] += 1

                # Build word-in-sentence examples
                for i, w in enumerate(words):
                    context = [words[j] for j in range(len(words)) if j != i]
                    self.examples.append({
                        'target': w,
                        'context': context,
                    })

    def get_vocab_list(self, min_count=1):
        """Return vocabulary sorted by frequency, filtered by min_count."""
        return [
            word for word, count in self.vocab_counts.most_common()
            if count >= min_count
        ]


# ---------------------------------------------------------------------------
# CBOW embedding trainer
# ---------------------------------------------------------------------------

class CBOWModel:
    """Stateful CBOW model: embedding + linear head + optimizer.

    Supports both bulk training (``train``) and incremental single-sentence
    updates (``train_step``).  The same ``nn.Embedding`` and optimizer are
    reused across calls so that gradient state is preserved.
    """

    def __init__(self, wv: WordVectors, learning_rate=0.01):
        """Initialise from an existing WordVectors (e.g. loaded from sentence.pt).

        The embedding weights are copied from *wv*; the linear head is
        freshly initialised.
        """
        self.index_to_key = list(wv.index_to_key)
        self.key_to_index = dict(wv.key_to_index)
        vocab_size = len(self.index_to_key)
        vector_size = wv._vectors.shape[1]

        self.embeddings = nn.Embedding(vocab_size, vector_size)
        with torch.no_grad():
            self.embeddings.weight.copy_(wv._vectors)

        self.linear = nn.Linear(vector_size, vocab_size)
        self.optimizer = optim.Adam(
            list(self.embeddings.parameters()) + list(self.linear.parameters()),
            lr=learning_rate,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # -- single-sentence update ------------------------------------------------

    def train_step(self, words):
        """Run one CBOW gradient step on a sentence (list of word strings).

        For each word in *words* whose neighbours are in the vocabulary,
        predict the target from the mean of its context embeddings and
        back-propagate.  Returns the mean loss for the sentence, or
        ``None`` if no usable examples were found.
        """
        # Build (target_idx, [context_idxs]) pairs for words in vocab
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

        # Pad contexts to uniform length
        max_ctx = max(len(c) for c in contexts)
        n = len(targets)
        ctx_padded = torch.zeros(n, max_ctx, dtype=torch.long)
        ctx_mask = torch.zeros(n, max_ctx)
        for i, c in enumerate(contexts):
            ctx_padded[i, :len(c)] = torch.tensor(c, dtype=torch.long)
            ctx_mask[i, :len(c)] = 1.0
        target_tensor = torch.tensor(targets, dtype=torch.long)

        # Forward
        ctx_embeds = self.embeddings(ctx_padded)
        masked = ctx_embeds * ctx_mask.unsqueeze(-1)
        ctx_mean = masked.sum(dim=1) / ctx_mask.sum(dim=1, keepdim=True)
        logits = self.linear(ctx_mean)
        loss = self.loss_fn(logits, target_tensor)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # -- export ----------------------------------------------------------------

    def to_word_vectors(self):
        """Snapshot current embedding weights as a new WordVectors."""
        with torch.no_grad():
            vectors = self.embeddings.weight.detach().clone()
        return WordVectors(vectors, self.index_to_key)


class EmbeddingTrainer:

    def __init__(self, vector_size=100, min_count=1, learning_rate=0.01,
                 batch_size=256):
        self.vector_size = vector_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, corpus_builder, epochs=10):
        vocab = corpus_builder.get_vocab_list(min_count=self.min_count)
        if not vocab:
            return WordVectors.from_vocab([], vector_size=self.vector_size)

        word_to_idx = {w: i for i, w in enumerate(vocab)}
        vocab_size = len(vocab)

        # Pre-compute targets and variable-length context index lists
        targets = []
        contexts = []
        for ex in corpus_builder.examples:
            target = ex['target']
            context = [w for w in ex['context'] if w in word_to_idx]
            if target in word_to_idx and len(context) > 0:
                targets.append(word_to_idx[target])
                contexts.append([word_to_idx[w] for w in context])

        if not targets:
            return WordVectors.from_vocab(vocab, vector_size=self.vector_size)

        n_examples = len(targets)
        target_tensor = torch.tensor(targets, dtype=torch.long)

        # Pad context lists to uniform length for batched lookup
        max_ctx = max(len(c) for c in contexts)
        ctx_padded = torch.zeros(n_examples, max_ctx, dtype=torch.long)
        ctx_mask = torch.zeros(n_examples, max_ctx, dtype=torch.float32)
        for i, c in enumerate(contexts):
            ctx_padded[i, :len(c)] = torch.tensor(c, dtype=torch.long)
            ctx_mask[i, :len(c)] = 1.0

        embeddings = nn.Embedding(vocab_size, self.vector_size)
        linear = nn.Linear(self.vector_size, vocab_size)
        optimizer = optim.Adam(
            list(embeddings.parameters()) + list(linear.parameters()),
            lr=self.learning_rate,
        )
        loss_fn = nn.CrossEntropyLoss()

        n_batches = (n_examples + self.batch_size - 1) // self.batch_size
        for epoch in range(epochs):
            total_loss = 0.0
            perm = torch.randperm(n_examples)
            for batch_i, start in enumerate(range(0, n_examples, self.batch_size)):
                idx = perm[start:start + self.batch_size]
                batch_ctx = ctx_padded[idx]       # (B, max_ctx)
                batch_mask = ctx_mask[idx]         # (B, max_ctx)
                batch_target = target_tensor[idx]  # (B,)

                ctx_embeds = embeddings(batch_ctx)  # (B, max_ctx, vec_size)
                # Masked mean: sum embeddings for real context, divide by count
                masked = ctx_embeds * batch_mask.unsqueeze(-1)
                ctx_mean = masked.sum(dim=1) / batch_mask.sum(dim=1, keepdim=True)

                logits = linear(ctx_mean)  # (B, vocab_size)
                loss = loss_fn(logits, batch_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)

                # Progress within epoch
                if n_batches > 10 and (batch_i + 1) % max(1, n_batches // 10) == 0:
                    pct = 100 * (batch_i + 1) / n_batches
                    avg = total_loss / (start + len(idx))
                    print(f"  Epoch {epoch+1}/{epochs} [{pct:3.0f}%] loss: {avg:.4f}",
                          flush=True)

            print(f"Epoch {epoch+1}/{epochs} complete, loss: {total_loss/n_examples:.4f}",
                  flush=True)

        with torch.no_grad():
            vectors = embeddings.weight.detach().clone()
        return WordVectors(vectors, vocab)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def build_embeddings(shard_paths, output_path, max_docs=10000,
                     vector_size=100, epochs=10, min_count=5,
                     batch_size=256):
    print(f"Building corpus from {len(shard_paths)} shard(s), max_docs={max_docs}...")
    cb = CorpusBuilder()
    count = 0
    for doc in iter_documents(shard_paths, max_docs=max_docs):
        cb.process_document(doc)
        count += 1
        if count % 100 == 0:
            print(f"  Processed {count} docs, vocab={len(cb.vocab)}, "
                  f"examples={len(cb.examples)}")

    print(f"Corpus complete: {count} docs, {len(cb.vocab)} unique words, "
          f"{len(cb.examples)} training examples")

    vocab = cb.get_vocab_list(min_count=min_count)
    print(f"Vocabulary after min_count={min_count} filter: {len(vocab)} words")

    print(f"Training {vector_size}-dim embeddings for {epochs} epochs "
          f"(batch_size={batch_size})...")
    trainer = EmbeddingTrainer(
        vector_size=vector_size,
        min_count=min_count,
        learning_rate=0.01,
        batch_size=batch_size,
    )
    wv = trainer.train(cb, epochs=epochs)

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    wv.save(output_path)
    print(f"Saved embeddings to {output_path} ({len(wv)} words, {vector_size} dims)")
    return wv


def _find_embeddings(path=None):
    """Locate and load a .pt embedding file from standard paths."""
    candidates = [path] if path else [
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
    train_p.add_argument('--output', default='output/embeddings/sentence.pt')
    train_p.add_argument('--data-dir', default=None)
    train_p.add_argument('--num-shards', type=int, default=1)
    train_p.add_argument('--max-docs', type=int, default=10000)
    train_p.add_argument('--vector-size', type=int, default=100)
    train_p.add_argument('--epochs', type=int, default=10)
    train_p.add_argument('--min-count', type=int, default=5)
    train_p.add_argument('--batch-size', type=int, default=256)

    # --- explore subcommand ---
    explore_p = sub.add_parser("explore", help="Explore trained embeddings")
    explore_p.add_argument("words", nargs="*", help="Word(s) to look up")
    explore_p.add_argument("--path", "-p", default=None, help="Path to .pt file")
    explore_p.add_argument("--topn", "-n", type=int, default=10)
    explore_p.add_argument("--vocab", action="store_true", help="Print vocabulary stats")

    args = parser.parse_args()

    if args.command == "explore":
        import random
        wv = _find_embeddings(args.path)
        if args.vocab:
            print(f"Vocabulary: {len(wv)} words, {wv._vectors.shape[1]}-dim vectors")
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
