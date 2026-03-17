# Training

## Overview

BasicModel training has two phases: **embedding pretraining** and **network training**.
Embedding pretraining builds word vectors from a large corpus. Network training uses
those vectors as the input representation and learns to predict and reconstruct
sentences.

The two phases can overlap: the `trainEmbeddings` config controls whether and how
embeddings continue to evolve during network training.

---

## Phase 1: Embedding Pretraining (`make train` / `embed.py train`)

Produces a static `sentence.pt` artifact containing word vectors.

### Pipeline

```
FineWeb-EDU parquet shards
    → Pass 1: stream documents, lex + parse, count words, build vocabulary
    → Pass 2: stream documents again, train SBOW per sentence, discard examples
    → Save WordVectors to sentence.pt
```

### SBOW (Sentence Bag of Words)

SBOW is a sentence-level variant of CBOW. Given a sentence of N words
with embedding vectors and a linear output matrix with bias:

Leave-one-out centroid for word i:

$$c_i = \frac{1}{N-1} \sum_{j \neq i} v_j$$

Logits (unnormalised scores for every word in vocabulary V):

$$z_i = W_o \, c_i + b_o$$

Softmax probability that centroid predicts word i:

$$P(w_i \mid c_i) = \frac{\exp(z_{i,w_i})}{\sum_{k=1}^{V} \exp(z_{i,k})}$$

Loss (cross-entropy, averaged over all N sentence positions):

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid c_i)$$

The gradient of the loss with respect to embedding v_i has two terms:

- **Attractive** (from position i where w_i is the target): pulls v_i
  toward directions that increase its probability.
- **Repulsive** (from positions j ≠ i where v_i is part of the centroid):
  pushes v_i away from centroids whose targets are other words.

The softmax partition function ensures that all V words compete: any word k
too close to a centroid it doesn't belong to increases the denominator and is
pushed away with force proportional to its softmax probability.
This adaptive repulsion distributes vectors across the embedding space.

This gives N positive and N × (V-1) repulsive updates per sentence. Every
word in the sentence is a positive example (unlike CBOW which picks one target).

```python
vecs = embedding(idx)                         # [N, dim]
total = vecs.sum(dim=0)                       # [dim]
centroids = (total - vecs) / (N - 1)          # [N, dim]  leave-one-out
logits = linear(centroids)                    # [N, vocab] full softmax
loss = cross_entropy(logits, idx)
```

### Two-Pass Architecture

The trainer uses two passes over the data to avoid dynamic model resizing:

- **Pass 1 (vocabulary):** Stream all documents, count every word via lex + parse.
  Words meeting `min_count` are promoted to the trainable vocabulary. The model
  (embedding + linear head) is allocated once at the final vocabulary size.

- **Pass 2 (training):** Stream documents again. For each sentence, run one SBOW
  step and immediately discard the examples. No examples are accumulated in memory.
  Multiple epochs re-stream the data.

### Configuration (Makefile variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `BASIC_SHARDS` | 1 | Number of FineWeb-EDU parquet shards |
| `BASIC_MAX_DOCS` | 10000 | Maximum documents to process |
| `BASIC_VEC_SIZE` | 100 | Embedding dimensionality |
| `BASIC_EPOCHS` | 10 | Training epochs |
| `BASIC_MIN_COUNT` | 10 | Minimum word count for vocabulary inclusion |
| `BASIC_BATCH_SIZE` | 8 | Batch size (unused by SBOW; per-sentence training) |

### Memory Considerations

The dominant memory cost is the linear output head: `vocab_size × vector_size` parameters
plus optimizer state. With 200K vocabulary and 100 dimensions:

- Embedding: 200K × 100 × 4 bytes = ~80MB
- Linear head: 100 × 200K × 4 bytes = ~80MB
- Adam optimizer: 2× momentum buffers = ~320MB total

The full softmax computes `(N, vocab_size)` logits per sentence. For large vocabularies
this can be expensive. Training runs on CPU by default (`BASICMODEL_DEVICE=cpu`) to
avoid MPS/GPU memory limits.

---

## Phase 2: Network Training (`BasicModel.py`)

The network learns to predict and reconstruct sentences using the pretrained embeddings
as its input representation.

### Masked Prediction Modes

The `maskedPrediction` config controls the prediction objective:

| Mode | Masking | Future truncation | Target | Description |
|------|---------|-------------------|--------|-------------|
| `NONE` | None | No | Dataset labels | Standard supervised |
| `MLM` | Zero word i | No | Embedding of word i | Bidirectional (BERT-style) |
| `ARLM` | Zero word i | Yes (j > i zeroed) | Embedding of word i | Autoregressive (GPT-style) |
| `ARUS` | Same as ARLM | Yes | Zero vector | Unsupervised (loss suppressed) |
| `RARLM` | Zero word i | Reverse (j < pos zeroed) | Embedding of word i | Right-to-left autoregressive |

### trainEmbeddings: EM-style Embedding Updates

The `trainEmbeddings` config controls whether and how embeddings evolve during
network training. This implements an EM-like alternation:

- **E-step (network):** Forward pass + masked prediction + backward + optimizer step.
  The network learns to predict/reconstruct with the current embedding space.

- **M-step (SBOW):** After each network step, run one SBOW update on the same
  sentence. The embedding vectors move to better capture sentence-level co-occurrence.

| Value | Network gradients → embeddings | SBOW step | Description |
|-------|-------------------------------|-----------|-------------|
| `NONE` | No | No | Embeddings frozen at pretrained values |
| `CBOW` | No | Yes | EM separation: only SBOW updates embeddings |
| `ARLM` | Yes | No | Only network backprop updates embeddings |
| `BOTH` | Yes | Yes | Both SBOW and network gradients update embeddings |

**`CBOW` is the recommended mode** for EM-style training. It maintains clean separation
between the two objectives: the network adapts to the embedding space (E-step), and
SBOW reshapes the space based on distributional co-occurrence (M-step). Neither
objective's gradients interfere with the other.

**`BOTH` risks gradient interference.** The network's reconstruction loss pulls embeddings
toward being maximally distinguishable (spreading apart), while SBOW pulls co-occurring
words together. These forces can conflict, potentially destabilizing training.

**`ARLM` lets the network reshape embeddings freely** without the distributional
constraint of SBOW. Useful when the prediction objective alone provides sufficient
signal for good embeddings.

### Why MSE over Embeddings (not Cross-Entropy over Vocabulary)

The network's output loss uses MSE against the target word's embedding vector, not
cross-entropy over vocabulary indices. This is an architectural advantage:

- A prediction near "cat" when the target is "kitten" incurs less loss than a prediction
  near "democracy" — the embedding space captures semantic similarity.
- Output dimensionality is the embedding dimension (~100) rather than the vocabulary
  size (~200K), reducing parameter count.

### Training Loop

Each batch (one sentence in masked prediction mode):

1. Embed sentence, expand to N masked copies (one per word position)
2. Forward through network: InputSpace → Percept → Concept → Symbol → Output
3. Compute output loss (MSE against target word embeddings)
4. If reversible: reverse pass reconstructs input, compute reconstruction loss
5. Combined loss = `(1 - recon_ratio) × output_loss + recon_ratio × recon_loss`
6. Backprop + optimizer step (excludes embedding params unless `ARLM` or `BOTH`)
7. If `trainEmbeddings` is `CBOW` or `BOTH`: run SBOW step on same sentence
8. Re-zero the [MASK] embedding

---

## SBOW vs CBOW

| Property | CBOW | SBOW |
|----------|------|------|
| Targets per sentence | 1 (pick one word) | N (every word) |
| Context | Other N-1 words | Leave-one-out centroid of N-1 |
| Positive updates | 1 per step | N per step |
| Repulsive force | vocab-1 implicit via softmax | N × (vocab-1) via softmax |
| Signal density | Low (1 gradient per sentence) | High (N gradients per sentence) |

SBOW provides richer signal per sentence because every word acts as both context
(for other words) and target (predicted from others). The full softmax over the
vocabulary at each position ensures vectors distribute across the hypersphere:
words too close to centroids they don't belong to are pushed away with force
proportional to their proximity.

---

## Embedding Space Geometry

The full softmax provides a distributional guarantee that random negative sampling
cannot match. For predicting word w_i from centroid c_i, the loss is:

$$\mathcal{L}_i = -\log \frac{\exp(z_{i,w_i})}{\sum_{j=1}^{V} \exp(z_{i,j})}$$

where the logits are:

$$z_i = W_o \, c_i + b_o$$

The partition function (denominator) sums over every word in the vocabulary. The
gradient with respect to the logit for any non-target word k ≠ w_i is:

$$\frac{\partial \mathcal{L}_i}{\partial z_{i,k}} = P(k \mid c_i)$$

Words with high softmax probability — those too close to the centroid — receive
the strongest repulsive gradient. Words already far away have near-zero probability
and are left alone. This adaptive repulsion naturally spreads vectors across the
hypersphere.

At equilibrium, no word can move closer to any centroid without increasing the loss.
This is the key advantage over approaches like direct push/pull with random negatives,
where most sampled negatives are already far away on a high-dimensional hypersphere
and contribute little useful gradient.
