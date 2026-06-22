# Training

> **2026-05-29 deltas:**
>
> - **Embedding unit-ball normalization.** The training loop calls
>   `Lexicon.normalize()` (unit-ball projection on the per-row
>   vectors) right after `optimizer.step()` in
>   `bin/Models.py::runBatch`. Keeps embedding vectors from drifting
>   off the unit ball under `JOINT` / `BACKPROP` modes.
> - **Seeded retries for MM_xor tests.** `test_learns_xor_signal`
>   and `test_convergence` in `test/test_mm_xor.py` use
>   `for seed in (42, 123, 7): torch.manual_seed(seed); …` (the
>   previously-dead `seed` loop variable in `test_convergence` is now
>   actually consumed). Pass-if-any-attempt-converges semantics.
> - **Reconstruction is unconditional from concepts.** The `<reconstruct>`
>   element (and `reconstructEnum`) was RETIRED (A1, 2026-06-09); there is
>   no knob to set or override. Whenever reconstruction fires it is always
>   concepts-seeded from the terminal `ConceptualSpace` STM snapshot.
> - **Supervised output loss restored.** `bin/Models.py::runBatch`
>   computes MSE on the supervised output in addition to the
>   reconstruction loss (`reconstructionScale`-weighted).
> - See [doc/old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

## Overview

Two phases: **embedding pretraining** and **network training**. Embedding
pretraining builds word vectors from a large corpus. Network training uses
those vectors as input representation and learns to predict and reconstruct
sentences.

The two phases can overlap: `<trainEmbedding>` controls whether and how
embeddings continue to evolve during network training.

---

## The staged objective curriculum

Network training climbs five objectives, from substrate-building reconstruction
to deliberate question answering. The order is not arbitrary — it is the
architecture's **two learning rules** in their natural developmental order:

- **EMA / occurrence** moves the codebooks: concepts form by being *seen*, with
  no gradient and no goal. This is *prediction as substrate* — unconscious,
  statistical, **System 1**. You cannot ask "why did the empire fall?" until
  *empire* and *fall* are codebook rows, and those are EMA-built by exposure.
- **Gradient on a task error** moves the attention readouts and the relation
  store: the model learns *where to look* and *what relates to what* by being
  *tested* — directed, credit-assigned, **System 2**.

So "bootstrap with prediction, then answer questions" is just the EMA
substrate-builder running first and the gradient deliberate-layer on top (the
§6c sentence protocol's *prime subsymbolically from what you see, then pump
symbolically*). Reconstruction (stages 1–2) is the **gate**: a question can only
be *answered* once an idea can be turned back into words, so the grammar-operation
inverses those stages need are the load-bearing prerequisite for the whole
curriculum.

| stage | objective | trains | dominant rule | status |
|---|---|---|---|---|
| 1 | reconstruct, **keep syntax**, fill the missing **words** | the lexical/leaf inverse + the codebook (word↔slot) | EMA (System 1) | recon loss + codebook decode exist; masking + leaf fill to build. **Needs no new grammar inverses** (the kept parse tree drives the existing reverse). |
| 2 | reconstruct with **no syntax / no words** (from the idea alone) | the full generative inverse (deliverable C) + abstraction | mixed | reverse path exists but is parse-tree-*dependent* → must be **decoupled** to drive from the primed symbolic space (attention) instead of `generate_rules`. |
| 3 | predict the **next sentence** | the inter-sentence (discourse AR) predictor | EMA → gradient | exists (`<prediction>interSentence`). |
| 4 | answer a question by **reasoning** (no search) | relations, modus ponens (`consequents()`), the catuṣkoṭi/trust | gradient (System 2) | the truth stores + `consequents` exist; the QA framing + consistency loss to build. |
| 5 | answer a question by **LTM search** | global attention (the typed `.where`) + the soft-read fed back + the explorer | gradient (System 2) | global attention (B) built/dark; the consumer (feed the read back, train by the answer) + book paging are deferred. |

Plain next-token/next-sentence prediction *under-trains this architecture's*
retrieval and relation machinery (the local window usually suffices, so the
separate global attention gets no reward); that is why prediction is the
**substrate** here, not the goal — stages 4–5 train the deliberate layer, and the
stochastic exploration finally earns its keep in stage 5 (search has only a
distal reward, so it must explore to break symmetry).

> Full design (with per-stage machinery, losses, and prerequisites):
> `doc/old/training-stages.md`. The grammar-operation inverses stages 1–2 require
> are enumerated in `doc/old/2026-06-19-grammar-inverses-handoff.md`.

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

- Embedding: 200K $\times$ 100 $\times$ 4 bytes = ~80MB
- Linear head: 100 $\times$ 200K $\times$ 4 bytes = ~80MB
- Adam optimizer: ~320MB total

Full softmax computes `(N, vocab_size)` logits per sentence. Training runs on
CPU by default (`BASICMODEL_DEVICE=cpu`) to avoid MPS/GPU memory limits.

---

## Phase 2: Network Training (`BasicModel.py`)

The network learns to predict and reconstruct sentences using pretrained
embeddings.

### Within-sentence training objective (IR-only)

Within-sentence training is **always IR** (masked-LM at the subsymbolic (PS)).
`create_ir_mask` replaces a `mask_rate` fraction of WHAT positions
with `NULL_PERCEPT` and snapshots the pre-mask event on
`_ir_pre_mask_input`; `runBatch` computes
`MSE(perceptualSpace at masked positions, _ir_pre_mask_input at
masked positions)` --- no head loss, no reverse pipeline.  The legacy
`<maskedPrediction>` knob and the AR / ARUS / ARIR modes were
retired 2026-05-14; sentence-level AR moved to
`InterSentenceLayer` (see `doc/Architecture.md` Section "Sentence-level AR
(`InterSentenceLayer`)").

| Knob | Effect |
|------|--------|
| `maskRate` | Bernoulli mask probability at the subsymbolic (PS) (BERT default 0.15) |
| `reconstructionScale` | Blend weight between output and reconstruction loss; `total = (1 - r)*output + r*recon`.  Legacy `<reverseScale>` parsed with deprecation warning. |
| `<reconstruct>` | RETIRED (A1, 2026-06-09; `reconstructEnum` removed).  There is no longer a target space-role knob: reconstruction is unconditionally concepts-seeded from the terminal `ConceptualSpace` STM snapshot, weighted by `reconstructionScale`. |

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
- **`BOTH` risks gradient interference** --- reconstruction pulls embeddings
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
  --- embedding space captures semantic similarity.
- Output dim is the embedding dim (~100) vs. vocab size (~200K).

### Training Loop

Data flows through `SentenceStreamDataset` wrapped in `DataLoader`. The
ordered training list is split into `B = batchSize` contiguous slabs of
length `L = len(split) // B`; at step `t`, row `b` is item `b * L + t`.
Temporal context is coherent across steps. No per-epoch global shuffle.

For each B-wide batch:

1. `BasicModel.Start(inputData)` cascades reset through every Space.
2. `InputSpace.forward()` lexes/embeds once into `[B, N, D]`
   (left-aligned, right-padded to N). No K axis, no cursor unfold.
3. `create_ir_mask` replaces a `mask_rate` fraction of WHAT positions
   with `NULL_PERCEPT`; pre-mask event stashed on
   `_ir_pre_mask_input` and the position mask on
   `_ir_mask_positions`.
4. `_forward_body` runs T stages on B rows.
5. `_forward_head` produces `[B, N, predDim]` (a side channel --- IR
   loss is computed at the subsymbolic (PS), not at the head).
6. `runBatch` computes loss via `TheError.add`:
   - `reconstruction` (subsymbolic (PS)): MSE between post-body
     `perceptualSpace` and `_ir_pre_mask_input` at masked positions.
   - `reconstruction_reverse` (concepts-seeded, unconditional):
     the reverse pass is always seeded from the terminal
     `ConceptualSpace` STM snapshot (the `<reconstruct>` enum was
     retired), weighted by `reconstructionScale`.
   - `arma` (sentence-level): `InterSentenceLayer.observe(s_t)` MSE
     between the ARMA(p, q) prediction and the current sentence rep,
     weighted by `armaScale`.
7. One `backward()` + `optimizer.step()` per DataLoader yield.
8. Embedding training (`CBOW`/`SBOW`/`BOTH`) runs once per batch.

---

## SBOW vs CBOW

| Property | CBOW | SBOW | Masked Prediction (Phase 2) |
|----------|------|------|----------------------------|
| Targets per sentence | 1 (pick one word) | N (every word) | N (one per masked pos) |
| Context | Other N-1 words | LOO centroid of N-1 | All unmasked (+ future truncation for AR) |
| Positive updates | 1 per step | N per step | N per step |
| Repulsive force | vocab-1 implicit via softmax | N $\times$ (vocab-1) | Implicit via MSE |
| Signal density | Low | High | High |
| Loss | Cross-entropy over vocab | Cross-entropy over vocab | MSE over embeddings |
| Updates embeddings | Yes (own optimizer) | Yes (own optimizer) | Only for `BACKPROP`/`BOTH`/`JOINT` |
| Used in (`<trainEmbedding>`) | `CBOW` | `SBOW`, `BOTH`, `JOINT` | `BACKPROP`, `BOTH`, `JOINT` |

SBOW provides richer signal --- every word acts as both context (for others)
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
