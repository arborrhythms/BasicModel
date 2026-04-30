# Training

## Overview

BasicModel training has two phases: **embedding pretraining** and **network training**.
Embedding pretraining builds word vectors from a large corpus. Network training uses
those vectors as the input representation and learns to predict and reconstruct
sentences.

The two phases can overlap: the `<trainEmbedding>` config controls whether
and how embeddings continue to evolve during network training.

---

## Phase 1: Embedding Pretraining (`make train` / `embed.py train`)

Produces a static embedding artifact (e.g. `BasicModel.kv`) containing word vectors.
The output path is read from `<embeddingPath>` in the XML config.

### Pipeline

```
FineWeb-EDU parquet shards
    -> Pass 1: stream documents, lex + parse, count words, build vocabulary
    -> Pass 2: stream documents again, train SBOW per sentence, discard examples
    -> Save WordVectors to sentence.pt
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
- **Repulsive** (from positions $j \neq i$ where $v_i$ is part of the centroid):
  pushes v_i away from centroids whose targets are other words.

The softmax partition function ensures that all V words compete: any word k
too close to a centroid it doesn't belong to increases the denominator and is
pushed away with force proportional to its softmax probability.
This adaptive repulsion distributes vectors across the embedding space.

This gives N positive and N $\times$ (V-1) repulsive updates per sentence. Every
word in the sentence is a positive example (unlike CBOW which picks one target).

```python
vecs = embedding(idx)                         # [N, dim]
total = vecs.sum(dim=0)                       # [dim]
centroids = (total - vecs) / (N - 1)          # [N, dim]  leave-one-out
logits = linear(centroids)                    # [N, vocab] full softmax
loss = cross_entropy(logits, idx)
```

### CBOW (Continuous Bag of Words)

CBOW predicts a single target word from its context. Given a sentence of $N$ words,
for target word $w_i$ with context $C_i = \{w_j : j \neq i\}$:

Context mean (with padding for variable-length contexts):

$$\bar{c}_i = \frac{1}{|C_i|} \sum_{j \in C_i} v_j$$

The loss is the same negative-sampling objective as SBOW but applied per-word
to the padded context mean rather than the leave-one-out centroid:

$$\mathcal{L}_{\text{CBOW}} = -\log \sigma({\bar{c}_i}^\top v_{w_i}) - \sum_{k=1}^{K} \log \sigma(-{\bar{c}_i}^\top v_{w_k^-})$$

where $K$ is the number of negative samples and $w_k^-$ are randomly sampled words.

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

The dominant memory cost is the linear output head: `vocab_size * vector_size` parameters
plus optimizer state. With 200K vocabulary and 100 dimensions:

- Embedding: 200K $\times$ 100 $\times$ 4 bytes = ~80MB
- Linear head: 100 $\times$ 200K $\times$ 4 bytes = ~80MB
- Adam optimizer: $2 \times$ momentum buffers = ~320MB total

The full softmax computes `(N, vocab_size)` logits per sentence. For large vocabularies
this can be expensive. Training runs on CPU by default (`BASICMODEL_DEVICE=cpu`) to
avoid MPS/GPU memory limits.

---

## Phase 2: Network Training (`BasicModel.py`)

The network learns to predict and reconstruct sentences using the pretrained embeddings
as its input representation.

### Masked Prediction Modes

The `maskedPrediction` config controls the prediction objective:

| Mode | Input Masking | Truncation | Target | Description |
|------|---------------|------------|--------|-------------|
| `NONE` | None | No | Dataset labels | Standard supervised |
| `IR` | Zero word i | No | Embedding of word i | Bidirectional input reconstruction (non-AR sibling of `ARIR`) |
| `AR` | Zero word i | Yes (j > i zeroed) | Embedding of word i | Autoregressive (GPT-style) |
| `ARUS` | Same as AR | Yes | Zero vector | Unsupervised (loss suppressed) |
| `ARIR` | Per-pos | Yes | Embedding + reconstruction | Autoregressive iterative reconstruction |

### `<trainEmbedding>`: Embedding Update Mode

The `trainEmbedding` config controls whether and how embeddings evolve during
network training.  When embedding updates are enabled (CBOW, SBOW, or BOTH),
this implements an EM-like alternation:

- **E-step (network):** Forward pass + masked prediction + backward + optimizer step.
  The network learns to predict/reconstruct with the current embedding space.

- **M-step (CBOW/SBOW):** After each network step, run one embedding update on the
  same sentence. The embedding vectors move to better capture co-occurrence.

| Value | Embedding method | Model layers | Description |
|-------|-----------------|-------------|-------------|
| `NONE` | Frozen | Trained | Embeddings frozen; only model layers train |
| `CBOW` | CBOW (padded context) | Trained | CBOW reshapes embeddings via own optimizer; model layers train separately |
| `SBOW` | SBOW (centroid) | Trained | Faster variant: predict word from leave-one-out centroid |
| `BACKPROP` | Backprop only | Trained | Codebook trained purely via model loss gradients; no SBOW/CBOW |
| `BOTH` | SBOW post-batch | Trained | Two optimizers: SBOW updates embeddings, Adam updates model layers |
| `JOINT` | Single backward | Trained | Single optimizer: combined model + SBOW loss |

### Gradient Flow Through Codebook

The `<trainEmbedding>` mode determines whether the codebook weight tensor is
**detached** during forward/reverse passes and whether it appears in the main
optimizer's parameter list.

| `trainEmbedding` | `detach()` in forward/reverse | In main optimizer | Embedding method |
|------------------|------------------------------|-------------------|-----------------|
| `NONE` | Yes -- codebook is a frozen lookup | No | Frozen |
| `CBOW` | Yes -- codebook is a frozen lookup | No | CBOW (own optimizer) |
| `SBOW` | Yes -- codebook is a frozen lookup | No | SBOW (own optimizer) |
| `BACKPROP` | No -- gradients flow through codebook | Yes | Model loss only |
| `BOTH` | No -- gradients flow through codebook | Yes | SBOW + backprop |
| `JOINT` | No -- gradients flow through codebook | Yes | Single combined loss |

**`NONE` is the recommended mode for constrained hardware** (e.g. Apple MPS
with <16GB). The codebook from Phase 1 is frozen, all memory and compute go
to the model layers.

**`CBOW`/`SBOW`** maintains clean EM separation: CBOW/SBOW reshapes the
embedding space using its own optimizer. No gradients flow from the model
through the codebook.

**`BACKPROP`** is the simplest trainable mode. The codebook participates in the
main optimizer and receives gradients from the model's output and reconstruction
losses, but no embedding-specific loss (SBOW/CBOW) is applied. The embedding
space is shaped entirely by the task. Best for small vocabularies or when the
pretrained embedding geometry should adapt freely to the downstream objective.

**`BOTH` risks gradient interference.** The network's reconstruction loss pulls
embeddings toward being maximally distinguishable (spreading apart), while
SBOW pulls co-occurring words together. These forces can conflict because
they use separate optimizers with separate momentum buffers.

### `JOINT`: Single Combined Loss

JOINT mode avoids the EM alternation of `BOTH` by computing a single combined
loss before one backward pass:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$$

where $\mathcal{L}_{\text{model}}$ is the standard output loss (plus reconstruction
loss if `reversible=true`), and $\lambda$ is the `trainEmbeddingRatio` hyperparameter (default 0.1).

The SBOW loss for the current sentence is computed using the same negative-sampling
objective as Phase 1:

$$\mathcal{L}_{\text{SBOW}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \sigma(c_i^\top v_{w_i}) + \sum_{k=1}^{K} \log \sigma(-c_i^\top v_{w_k^-}) \right]$$

where $c_i$ is the leave-one-out centroid, $v_{w_i}$ is the target word's embedding,
and $w_k^-$ are negative samples drawn uniformly from the vocabulary.

**Advantages over `BOTH`:**
- **One optimizer, one momentum buffer.** Adam sees the full gradient landscape
  from both objectives simultaneously, avoiding the oscillation risk of two
  independent optimizers pulling embeddings in different directions.
- **Gradient coherence.** The model's MSE loss and the SBOW co-occurrence loss
  are balanced by $\lambda$ before any parameter update, so the step direction
  reflects the intended trade-off.
- **Simpler implementation.** No post-batch embedding step; everything flows
  through `totalLoss.backward()`.

**Trade-off:** Both objectives share the same learning rate and Adam state.
If the two loss surfaces have very different curvature, `trainEmbeddingRatio` may need
tuning. The `BOTH` mode's separate optimizers can use different learning rates.

### Why MSE over Embeddings (not Cross-Entropy over Vocabulary)

The network's output loss uses MSE against the target word's embedding vector, not
cross-entropy over vocabulary indices. This is an architectural advantage:

- A prediction near "cat" when the target is "kitten" incurs less loss than a prediction
  near "democracy" -- the embedding space captures semantic similarity.
- Output dimensionality is the embedding dimension (~100) rather than the vocabulary
  size (~200K), reducing parameter count.

### Training Loop

Data flows through a `SentenceStreamDataset` wrapped in a PyTorch `DataLoader`.
The ordered training list is split into `B = batchSize` contiguous slabs of
length `L = len(split) // B`; at step `t` the loader yields a B-wide batch
where row `b` is item `b * L + t`. Each batch row therefore carries its own
document-order stream, so temporal context is coherent across steps. No
per-epoch global shuffle is applied. `numWorkers > 0` enables async prefetch.

For each B-wide batch in AR modes (`AR` / `ARUS` / `ARIR`):

1. `MentalModel.Start(inputData)` cascades reset through every Space and
   Layer (clearing per-sentence scratch).
2. `InputSpace.forward()` lexes/embeds the input once into `[B, T, D]`,
   then builds all K progressive-prefix windows in a single
   `tensor.unfold(1, N, 1)` -- left-padding N zeros so window `k` is
   `k` zero-pad slots followed by `emb[0..k-1]` right-aligned. The
   emitted subspace carries `event: [B, K, N, D]` with `k_axis=True`
   and a `[B, K]` validity mask (NULL = all-zero target embedding).
   The body consumes the K axis as microbatch (flattened to `B*K`) and
   the head produces all K predictions in one pass.

3. For `ARIR` only, a single terminal `reverse(symbols)` after the
   body produces a `[B, N, D]` reconstruction.

4. `runBatch` computes the loss once via `TheError.add`:
   - `AR`: output prediction only (K per-row predictions, masked by validity).
   - `ARUS`: no output term (suppressed); no reconstruction.
   - `ARIR`: output prediction + reconstruction, weighted by `reverseScale`.

5. One `backward()` + `optimizer.step()` per DataLoader yield. The
   single-unfold microbatch path replaces the earlier N-pass cursor
   loop, collapsing the N forward/reverse calls into one.

6. Embedding training (`trainEmbedding` in `CBOW`/`SBOW`/`BOTH`) runs
   once per batch after the forward/backward cycle.

Non-AR modes (`maskedPrediction=NONE`) do a single forward/backward/step
per B-wide batch through the same `runBatch` code path, skipping the
outer pos loop.

Mode contract:

| Mode   | Per-pos output | Terminal reverse | Reconstruction loss |
|--------|----------------|--------------------------------|-----------------|
| `AR` | Yes            | No (ignores `<reconstruct>`)   | Not trained     |
| `ARUS` | No             | No (ignores `<reconstruct>`)   | Not trained     |
| `ARIR` | Yes            | Yes                            | Over `[B, N, D]` |

`<maskedPrediction>ARIR</maskedPrediction>` requires
`<reconstruct>` to be something other than `NONE` at validation.

---

## SBOW vs CBOW

| Property | CBOW | SBOW | Masked Prediction (Phase 2) |
|----------|------|------|----------------------------|
| Targets per sentence | 1 (pick one word) | N (every word) | N (one per masked position) |
| Context | Other N-1 words | Leave-one-out centroid of N-1 | All unmasked words (+ future truncation for AR/AR) |
| Positive updates | 1 per step | N per step | N per step |
| Repulsive force | vocab-1 implicit via softmax | N $\times$ (vocab-1) via softmax | Implicit via MSE in embedding space |
| Signal density | Low (1 gradient per sentence) | High (N gradients per sentence) | High (N gradients per sentence) |
| Loss function | Cross-entropy over vocabulary | Cross-entropy over vocabulary | MSE over embedding vectors |
| Updates embeddings directly | Yes (own optimizer) | Yes (own optimizer) | Only for `BACKPROP`/`BOTH`/`JOINT` |
| Used in (`<trainEmbedding>`) | `CBOW` | `SBOW`, `BOTH`, `JOINT` | `BACKPROP`, `BOTH`, `JOINT` |

SBOW provides richer signal per sentence because every word acts as both context
(for other words) and target (predicted from others). The full softmax over the
vocabulary at each position ensures vectors distribute across the hypersphere:
words too close to centroids they don't belong to are pushed away with force
proportional to their proximity.

---

## Three-File Architecture

Model behaviour is partitioned across three independently managed artifacts:

| Artifact | Example | Contents | Updated by |
|----------|---------|----------|-----------|
| XML config | `data/BasicModel.xml` | Architecture, hyperparameters, file paths | Hand-edited |
| Embedding | `data/BasicModel.kv` | Word vectors ($V \times d$ codebook matrix) | Phase 1: `embed.py` |
| Weights | `data/BasicModel.ckpt` | Model layer parameters (attention, Pi/Sigma weights) | Phase 2: `BasicModel.py` |

The checkpoint explicitly excludes embedding parameters (`wv._vectors`).
This separation enables independent iteration:

- Retrain embeddings without touching model weights (`--force-embeddings`)
- Retrain model with frozen codebook (`<trainEmbedding>NONE</trainEmbedding>`)
- Swap codebooks between models sharing the same architecture

---

## Embedding Space Geometry

The full softmax provides a distributional guarantee that random negative sampling
cannot match. For predicting word w_i from centroid c_i, the loss is:

$$\mathcal{L}_i = -\log \frac{\exp(z_{i,w_i})}{\sum_{j=1}^{V} \exp(z_{i,j})}$$

where the logits are:

$$z_i = W_o \, c_i + b_o$$

The partition function (denominator) sums over every word in the vocabulary. The
gradient with respect to the logit for any non-target word $k \neq w_i$ is:

$$\frac{\partial \mathcal{L}_i}{\partial z_{i,k}} = P(k \mid c_i)$$

Words with high softmax probability -- those too close to the centroid -- receive
the strongest repulsive gradient. Words already far away have near-zero probability
and are left alone. This adaptive repulsion naturally spreads vectors across the
hypersphere.

At equilibrium, no word can move closer to any centroid without increasing the loss.
This is the key advantage over approaches like direct push/pull with random negatives,
where most sampled negatives are already far away on a high-dimensional hypersphere
and contribute little useful gradient.

---

## Profiling

The `--profile` flag wraps Phase 2 in `cProfile`:

```bash
make train_micro_remote  # add --profile via train.py
# Or directly:
python bin/train.py --model data/BasicModel.xml --profile --max-docs 500
```

Output: a `.prof` file in `output/profiles/` plus a top-30 summary printed to stdout.
View interactively with `snakeviz`:

```bash
pip install snakeviz
snakeviz output/profiles/train_20260319_111441.prof
```

For live profiling of a running process, `py-spy` can attach by PID (requires root on macOS):

```bash
pip install py-spy
sudo py-spy record -o profile.svg --pid <PID> --duration 30
```
