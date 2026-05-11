# Training

## Overview

Two phases: **embedding pretraining** and **network training**. Embedding
pretraining builds word vectors from a large corpus. Network training uses
those vectors as input representation and learns to predict and reconstruct
sentences.

The two phases can overlap: `<trainEmbedding>` controls whether and how
embeddings continue to evolve during network training.

---

## Phase 1: Embedding Pretraining (`make train` / `embed.py train`)

Produces a static embedding artifact (e.g. `BasicModel.kv`). Output path from
`<embeddingPath>`.

### Pipeline

```
FineWeb-EDU parquet shards
    -> Pass 1: stream documents, lex + parse, count words, build vocabulary
    -> Pass 2: stream documents again, train SBOW per sentence, discard examples
    -> Save WordVectors to sentence.pt
```

### SBOW (Sentence Bag of Words)

Sentence-level variant of CBOW. Given a sentence of $N$ words:

Leave-one-out centroid for word $i$:

$$c_i = \frac{1}{N-1} \sum_{j \neq i} v_j$$

Logits (unnormalised, for every word in vocabulary $V$):

$$z_i = W_o \, c_i + b_o$$

Softmax probability:

$$P(w_i \mid c_i) = \frac{\exp(z_{i,w_i})}{\sum_{k=1}^{V} \exp(z_{i,k})}$$

Loss (cross-entropy, averaged over all $N$ positions):

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid c_i)$$

Gradient of the loss has two terms:

- **Attractive** (from position $i$ where $w_i$ is the target): pulls $v_i$
  toward directions that increase its probability.
- **Repulsive** (from positions $j \neq i$ where $v_i$ is part of the
  centroid): pushes $v_i$ away from centroids whose targets are other words.

The softmax partition function ensures all $V$ words compete: any word too
close to a centroid it doesn't belong to increases the denominator and is
pushed away with force proportional to its softmax probability. This
adaptive repulsion distributes vectors across the embedding space.

```python
vecs = embedding(idx)                         # [N, dim]
total = vecs.sum(dim=0)                       # [dim]
centroids = (total - vecs) / (N - 1)          # [N, dim]  leave-one-out
logits = linear(centroids)                    # [N, vocab] full softmax
loss = cross_entropy(logits, idx)
```

### CBOW (Continuous Bag of Words)

CBOW predicts a single target word from its context. Context mean (with
padding for variable-length contexts):

$$\bar{c}_i = \frac{1}{|C_i|} \sum_{j \in C_i} v_j$$

Loss applies the same negative-sampling objective per-word to the padded
context mean rather than the leave-one-out centroid:

$$\mathcal{L}_{\text{CBOW}} = -\log \sigma({\bar{c}_i}^\top v_{w_i}) - \sum_{k=1}^{K} \log \sigma(-{\bar{c}_i}^\top v_{w_k^-})$$

where $K$ is the number of negative samples and $w_k^-$ are sampled words.

### Two-Pass Architecture

- **Pass 1 (vocabulary):** Stream all documents, count every word. Words
  meeting `min_count` are promoted; model (embedding + linear head) allocated
  once at final vocab size.
- **Pass 2 (training):** Stream documents again. Per sentence, run one SBOW
  step and discard examples. Multiple epochs re-stream the data.

### Configuration (Makefile variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `BASIC_SHARDS` | 1 | FineWeb-EDU parquet shards |
| `BASIC_MAX_DOCS` | 10000 | Max documents to process |
| `BASIC_VEC_SIZE` | 100 | Embedding dimensionality |
| `BASIC_EPOCHS` | 10 | Training epochs |
| `BASIC_MIN_COUNT` | 10 | Min word count for vocabulary inclusion |
| `BASIC_BATCH_SIZE` | 8 | Batch size (unused by SBOW; per-sentence) |

### Memory Considerations

Dominant cost is the linear output head: `vocab_size * vector_size` parameters
plus optimizer state. With 200K vocab and 100 dims:

- Embedding: 200K × 100 × 4 bytes = ~80MB
- Linear head: 100 × 200K × 4 bytes = ~80MB
- Adam optimizer: ~320MB total

Full softmax computes `(N, vocab_size)` logits per sentence. Training runs on
CPU by default (`BASICMODEL_DEVICE=cpu`) to avoid MPS/GPU memory limits.

---

## Phase 2: Network Training (`BasicModel.py`)

The network learns to predict and reconstruct sentences using pretrained
embeddings.

### Masked Prediction Modes

`maskedPrediction` controls the prediction objective:

| Mode | Input Masking | Truncation | Target | Description |
|------|---------------|------------|--------|-------------|
| `NONE` | None | No | Dataset labels | Standard supervised |
| `IR` | Zero word i | No | Embedding of word i | Bidirectional input reconstruction |
| `AR` | Zero word i | Yes (j > i zeroed) | Embedding of word i | Autoregressive (GPT-style) |
| `ARUS` | Same as AR | Yes | Zero vector | Unsupervised (loss suppressed) |
| `ARIR` | Per-pos | Yes | Embedding + reconstruction | Autoregressive iterative reconstruction |

### `<trainEmbedding>`: Embedding Update Mode

Controls whether and how embeddings evolve during network training. When
enabled (CBOW, SBOW, or BOTH), implements EM-like alternation:

- **E-step (network):** Forward + masked prediction + backward + optimizer.
- **M-step (CBOW/SBOW):** Run one embedding update on the same sentence.

| Value | Embedding | Model layers | Description |
|-------|-----------|-------------|-------------|
| `NONE` | Frozen | Trained | Only model layers train |
| `CBOW` | CBOW (padded context, own optimizer) | Trained | Model layers train separately |
| `SBOW` | SBOW (centroid, own optimizer) | Trained | Faster variant |
| `BACKPROP` | Backprop only | Trained | Codebook trained purely via model loss |
| `BOTH` | SBOW post-batch | Trained | Two optimizers |
| `JOINT` | Single backward | Trained | Single optimizer: combined model + SBOW loss |

### Gradient Flow Through Codebook

`<trainEmbedding>` determines whether the codebook weight is **detached**
during forward/reverse and whether it appears in the main optimizer:

| `trainEmbedding` | `detach()` | In main optimizer | Method |
|------------------|------------|-------------------|--------|
| `NONE` | Yes | No | Frozen |
| `CBOW` | Yes | No | CBOW (own optimizer) |
| `SBOW` | Yes | No | SBOW (own optimizer) |
| `BACKPROP` | No | Yes | Model loss only |
| `BOTH` | No | Yes | SBOW + backprop |
| `JOINT` | No | Yes | Single combined loss |

- **`NONE`** is recommended for constrained hardware (Apple MPS <16GB).
- **`CBOW`/`SBOW`** maintains clean EM separation.
- **`BACKPROP`** is the simplest trainable mode. Embedding shaped entirely
  by the task; best for small vocabularies.
- **`BOTH` risks gradient interference** — reconstruction pulls embeddings
  apart (distinguishability); SBOW pulls co-occurring words together. These
  forces can conflict because they use separate optimizers.

### `JOINT`: Single Combined Loss

Avoids EM alternation by computing one combined loss before one backward:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$$

where $\lambda$ is `trainEmbeddingRatio` (default 0.1).

SBOW loss uses the same negative-sampling objective:

$$\mathcal{L}_{\text{SBOW}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \sigma(c_i^\top v_{w_i}) + \sum_{k=1}^{K} \log \sigma(-c_i^\top v_{w_k^-}) \right]$$

**Advantages.** One optimizer, one momentum buffer; gradient coherence; no
post-batch step.

**Trade-off.** Both objectives share LR and Adam state. If loss surfaces have
very different curvature, $\lambda$ may need tuning.

### Why MSE over Embeddings (not Cross-Entropy)

- "cat" prediction when target is "kitten" incurs less loss than "democracy"
  — embedding space captures semantic similarity.
- Output dim is the embedding dim (~100) vs. vocab size (~200K).

### Training Loop

Data flows through `SentenceStreamDataset` wrapped in `DataLoader`. The
ordered training list is split into `B = batchSize` contiguous slabs of
length `L = len(split) // B`; at step `t`, row `b` is item `b * L + t`.
Temporal context is coherent across steps. No per-epoch global shuffle.

For each B-wide batch in AR modes:

1. `BasicModel.Start(inputData)` cascades reset through every Space.
2. `InputSpace.forward()` lexes/embeds once into `[B, T, D]`, then builds all
   K progressive-prefix windows via `tensor.unfold(1, N, 1)` — left-padding
   N zeros so window `k` is `k` zero-pad slots followed by `emb[0..k-1]`
   right-aligned. The subspace carries `event: [B, K, N, D]` with
   `k_axis=True` and a `[B, K]` validity mask. Body consumes K as microbatch
   (flattened to `B*K`); head produces all K predictions in one pass.
3. For `ARIR` only, a single terminal `reverse(symbols)` produces a
   `[B, N, D]` reconstruction.
4. `runBatch` computes loss once via `TheError.add`:
   - `AR`: output prediction only (masked by validity).
   - `ARUS`: no output term; no reconstruction.
   - `ARIR`: output prediction + reconstruction, weighted by `reverseScale`.
5. One `backward()` + `optimizer.step()` per DataLoader yield.
6. Embedding training (`CBOW`/`SBOW`/`BOTH`) runs once per batch.

Non-AR modes (`maskedPrediction=NONE`) do a single forward/backward/step per
B-wide batch, skipping the pos loop.

Mode contract:

| Mode | Per-pos output | Terminal reverse | Reconstruction loss |
|------|----------------|------------------|--------------------|
| `AR` | Yes | No (ignores `<reconstruct>`) | Not trained |
| `ARUS` | No | No | Not trained |
| `ARIR` | Yes | Yes | Over `[B, N, D]` |

`<maskedPrediction>ARIR</maskedPrediction>` requires `<reconstruct>` not
`NONE`.

---

## SBOW vs CBOW

| Property | CBOW | SBOW | Masked Prediction (Phase 2) |
|----------|------|------|----------------------------|
| Targets per sentence | 1 (pick one word) | N (every word) | N (one per masked pos) |
| Context | Other N-1 words | LOO centroid of N-1 | All unmasked (+ future truncation for AR) |
| Positive updates | 1 per step | N per step | N per step |
| Repulsive force | vocab-1 implicit via softmax | N × (vocab-1) | Implicit via MSE |
| Signal density | Low | High | High |
| Loss | Cross-entropy over vocab | Cross-entropy over vocab | MSE over embeddings |
| Updates embeddings | Yes (own optimizer) | Yes (own optimizer) | Only for `BACKPROP`/`BOTH`/`JOINT` |
| Used in (`<trainEmbedding>`) | `CBOW` | `SBOW`, `BOTH`, `JOINT` | `BACKPROP`, `BOTH`, `JOINT` |

SBOW provides richer signal — every word acts as both context (for others)
and target (predicted from others). The full softmax pushes vectors to
distribute across the hypersphere.

---

## Three-File Architecture

| Artifact | Example | Contents | Updated by |
|----------|---------|----------|-----------|
| XML config | `data/BasicModel.xml` | Architecture, hyperparameters, paths | Hand-edited |
| Embedding | `data/BasicModel.kv` | Word vectors ($V \times d$) | Phase 1 |
| Weights | `data/BasicModel.ckpt` | Model layer parameters | Phase 2 |

Checkpoint excludes embedding parameters (`wv._vectors`). Enables:
- Retrain embeddings without touching model weights (`--force-embeddings`)
- Retrain model with frozen codebook (`<trainEmbedding>NONE`)
- Swap codebooks between models sharing the same architecture

---

## Embedding Space Geometry

The full softmax provides a distributional guarantee that random negative
sampling cannot match. For predicting $w_i$ from centroid $c_i$:

$$\mathcal{L}_i = -\log \frac{\exp(z_{i,w_i})}{\sum_{j=1}^{V} \exp(z_{i,j})}$$

The gradient with respect to the logit for any non-target word $k \neq w_i$:

$$\frac{\partial \mathcal{L}_i}{\partial z_{i,k}} = P(k \mid c_i)$$

Words with high softmax probability (too close to the centroid) receive the
strongest repulsive gradient; words already far away contribute almost
nothing. This naturally spreads vectors across the hypersphere.

At equilibrium, no word can move closer to any centroid without increasing
the loss. Random negatives, by contrast, mostly sample words already far away
on a high-dimensional hypersphere and contribute little useful gradient.

---

## Profiling

`--profile` wraps Phase 2 in `cProfile`:

```bash
make train_micro_remote  # add --profile via train.py
# Or:
python bin/train.py --model data/BasicModel.xml --profile --max-docs 500
```

Output: a `.prof` file in `output/profiles/` plus a top-30 summary to stdout.
View with `snakeviz`:

```bash
pip install snakeviz
snakeviz output/profiles/train_20260319_111441.prof
```

For live profiling, `py-spy` can attach by PID (requires root on macOS):

```bash
pip install py-spy
sudo py-spy record -o profile.svg --pid <PID> --duration 30
```
