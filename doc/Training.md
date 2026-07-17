# Training

> For the current ownership of the training lifecycle and the proposed split
> between launch resolution, corpus cursors, host orchestration, compiled tensor
> steps, objectives, and checkpoints, see
> [Runtime Architecture and Componentization](Componentization.md).

> **2026-05-29 deltas:**
>
> - **Embedding unit-cell wrap.** The training loop calls `normalize()`
>   on the `Embedding` basis (`perceptualSpace.subspace.what`) right
>   after `optimizer.step()` in `bin/Models.py::runBatch`. That
>   `normalize()` is a periodic unit-cell **wrap** into $[-1, 1)$ via
>   `embed._wrap_unit_ball` (torus geometry), *not* an L2 ball
>   projection (`Layers.Lexicon.normalize` does ball-project, but is
>   not on this path). Keeps embedding vectors from drifting out of
>   the unit cell under `JOINT` / `BACKPROP` modes.
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
>   computes MSE on labeled datasets. Unlabeled corpora such as FineWeb
>   train only through reconstruction/prediction objectives.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Training keeps the LLM-like objectives of prediction and reconstruction, but
does not make next-token likelihood the whole story. Early objectives shape the
embedding and codebook substrate; later objectives train the Formal Concept
Analysis-like concept order, the DisCoCat-like grammar composition path, and the
truth/reasoning machinery that tests composed meanings. This is why the staged
curriculum treats prediction as substrate-building and question answering as a
separate deliberate objective.

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
| 1 | reconstruct, **keep syntax**, fill the missing **words** | the lexical/leaf inverse + the codebook (word$\leftrightarrow$slot) | EMA (System 1) | recon loss + codebook decode exist; masking + leaf fill to build. **Needs no new grammar inverses** (the kept parse tree drives the existing reverse). |
| 2 | reconstruct with **no syntax / no words** (from the idea alone) | the full generative inverse (deliverable C) + abstraction | mixed | reverse path exists but is parse-tree-*dependent* $\to$ must be **decoupled** to drive from the primed symbolic space (attention) instead of `generate_rules`. |
| 3 | predict the **next sentence** | the inter-sentence (discourse AR) predictor | EMA $\to$ gradient | exists (`<prediction>interSentence`). |
| 4 | answer a question by **reasoning** (no search) | relations, modus ponens (`consequents()`), the catuṣkoṭi/trust | gradient (System 2) | the truth stores + `consequents` exist; the QA framing + consistency loss to build. |
| 5 | answer a question by **LTM search** | global attention (the typed `.where`) + the soft-read fed back + the explorer | gradient (System 2) | global attention (B) built/dark; the consumer (feed the read back, train by the answer) + book paging are deferred. |

Plain next-token/next-sentence prediction *under-trains this architecture's*
retrieval and relation machinery (the local window usually suffices, so the
separate global attention gets no reward); that is why prediction is the
**substrate** here, not the goal — stages 4–5 train the deliberate layer, and the
stochastic exploration finally earns its keep in stage 5 (search has only a
distal reward, so it must explore to break symmetry).

---

## Phase 1: Embedding Pretraining (`make train` / `embed.py train`)

Produces a static embedding artifact (e.g. `BasicModel.kv`). Output path from
`<embeddingPath>`. Under `make train` (`bin/train.py`), Phase 1 runs **only**
when the artifact is absent or `--force-embeddings` is passed (byte-lexer
configs skip it entirely); Phase 2 always runs.

### Pipeline

```
FineWeb-EDU parquet shards
    -> Pass 1: stream documents, lex + parse, count words, build vocabulary
    -> Pass 2: stream documents again, train SBOW per sentence, discard examples
    -> Save WordVectors to sentence.pt
```

### SBOW (Sentence Bag of Words)

The current streaming SBOW trainer uses a `Lexicon` embedding table only: no
vocabulary projection head and no full softmax. Given a sentence of $N$ words,
it builds a Gaussian-weighted leave-one-out in-group center (`pode`) for each
word, samples a same-power random out-group, and trains two attractive terms:

- the word vector is pulled toward its in-sentence pode;
- random out-group vectors are pulled toward the pode's torus antipode.

Implementation: `StreamingSBOWTrainer._train_sentence()` (private) in
`bin/embed.py`. `sim(...)` below is `Lexicon.similarity`, the wrapped-MSE
torus similarity in $[-1, 1]$ --- not a dot product.

```python
vecs = embeddings(idx)              # [N, D]
pode = inner_kernel @ vecs           # [N, D], diagonal excluded
antipode = Lexicon.antipode(pode)
loss = -logsigmoid(sim(pode, vecs)).mean()
loss -= logsigmoid(sim(antipode, out_vecs)).mean()
```

### CBOW (Continuous Bag of Words)

CBOW predicts each target word from its context; every in-vocab word in the
sentence takes a turn as the target ($N$ examples per sentence), with a
leave-one-out padded context mean:

$$\bar{c}_i = \frac{1}{|C_i|} \sum_{j \in C_i} v_j$$

Loss applies the same negative-sampling objective per-word to the padded
context mean rather than the leave-one-out centroid:

$$\mathcal{L}_{\text{CBOW}} = -\log \sigma\big(s(\bar{c}_i, v_{w_i})\big) - \frac{1}{K}\sum_{k=1}^{K} \log \sigma\big(-s(\bar{c}_i, v_{w_k^-})\big)$$

where $s(a, b)$ is the wrapped-MSE torus similarity (`_wrapped_mse_score`,
$1 - 2\,\mathrm{mean}(\delta_{\text{wrap}}^2)$ in $[-1, 1]$ --- not a dot
product), $K = 64$ is the negative-sample count, and $w_k^-$ are uniformly
sampled words. No full-softmax vocabulary head is involved. Implementation:
`PretrainModel.train_step` $\to$ `_neg_sampling_loss` in `bin/embed.py`.

### Two-Pass Architecture

- **Pass 1 (vocabulary):** Stream all documents, count every word. Words
  meeting `min_count` are promoted; the model (a `Lexicon` embedding table
  only --- no vocab-projection head) is allocated once at final vocab size.
- **Pass 2 (training):** Stream documents again. Per sentence, run one SBOW
  step and discard examples. Multiple epochs re-stream the data.

### Configuration (train.py / environment overrides)

| Variable | Default | Description |
|----------|---------|-------------|
| `BASIC_DATASET` | XML default | Dataset/data source selector |
| `BASIC_MAX_DOCS` | XML default | Max documents to process |
| `BASIC_NUM_SHARDS` | XML default | FineWeb-EDU shard count |
| `BASIC_NUM_EPOCHS` | XML default | Phase 2 epoch override |
| `BASIC_BATCH_SIZE` | XML default | Phase 2 batch-size override (`--batch-size`) |
| `BASIC_MAX_TOKENS` | XML default | Token budget |
| `BASIC_MAX_BATCHES` | XML default | Phase 2 batch cap |
| `BASIC_RANDOM_SHARDS` | `0` | `1` = randomize FineWeb shard order (`--random-shards`) |
| `BASIC_CHECKPOINT_EVERY_BATCHES` | XML `checkpointEveryBatches` (0) | Mid-epoch periodic checkpoint cadence in batches; 0 disables |
| `BASIC_RUN_TEST` | unset | Enable test passes; optional value caps test batches |

### Memory Considerations

For SBOW, dominant cost is the `Lexicon` table plus optimizer state. With 200K
vocab and 100 dims:

- Embedding: 200K $\times$ 100 $\times$ 4 bytes = ~80MB
- Optimizer state depends on the chosen optimizer and device

Streaming SBOW trains per sentence and normalizes the Lexicon after each step.

---

## Phase 2: Network Training (`Models.py`)

The network learns to predict and reconstruct sentences using pretrained
embeddings.

### Within-sentence training objective (IR-only)

Within-sentence training is **always IR** (masked-LM at the subsymbolic (PS)).
`create_ir_mask` replaces a `mask_rate` fraction of WHAT positions
with `NULL_PERCEPT` and snapshots the pre-mask event on
`_ir_pre_mask_input`. On the whole-slab / non-grammar path
(`_per_word_enabled=False`) `runBatch` computes the dense masked-LM
`MSE(perceptualSpace at masked positions, _ir_pre_mask_input at
masked positions)` via `compute_masked`. On the per-word grammar path the
`reconstruction` slot is instead the D3 reverse($S$) reconstruction ---
there `lossIn` IS the reverse pipeline. The supervised output-head loss is
also back (2026-05-28): the `output` channel is scored at weight 1.0
whenever labels exist (unlabeled corpora degrade it to zero). Two carve-outs
qualify "reconstruction is always concepts-seeded": at train time the D3
path **dedupes** the separate `reconstruction_reverse` term (`lossIn`
already carries the reverse objective, so the concepts-seeded reverse is
skipped to avoid double counting), and at serial EVAL the decode consumes
the Method-1 stored-leaves replay (`_reverse_method1_leaves`) rather than
decoding from the concept snapshot. The legacy
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

`<trainEmbedding>` determines whether the embedding parameters appear in the
**main optimizer** --- that is the whole freezing mechanism. There is no
mode-conditional `detach()` of the codebook weight anywhere:
`optimize_embedding = train_embedding not in ("NONE", "CBOW", "SBOW")`, and
`getOptimizer` filters embedding params out of the main param list when it is
False. Under NONE/CBOW/SBOW gradients may still flow through the lookup, but
no main-optimizer step ever moves the rows:

| `trainEmbedding` | In main optimizer | Embedding rows moved by |
|------------------|-------------------|-------------------------|
| `NONE` | No | Nothing (frozen) |
| `CBOW` | No | CBOW pretrainer (own optimizer) |
| `SBOW` | No | SBOW pretrainer (own optimizer) |
| `BACKPROP` | Yes | Model loss only |
| `BOTH` | Yes | Model loss + post-batch SBOW step |
| `JOINT` | Yes | Single combined loss |

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

where $\lambda$ is `<embeddingScale>` (default 0.1).

SBOW loss uses the same negative-sampling objective, with $s(a, b)$ the
wrapped-MSE torus similarity (`_wrapped_mse_score`) rather than a dot
product:

$$\mathcal{L}_{\text{SBOW}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \sigma\big(s(c_i, v_{w_i})\big) + \frac{1}{K}\sum_{k=1}^{K} \log \sigma\big(-s(c_i, v_{w_k^-})\big) \right]$$

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

1. `_start_spaces_for_forward()` calls `Start()` on every Space
   (`runBatch` pre-invokes it before the compiled step; `forward()`
   self-invokes when not externally started).
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
   - `output` (supervised head): MSE between the aligned head
     prediction and the labels, weight 1.0 when labels exist
     (zero-weighted otherwise).
   - `reconstruction` (subsymbolic (PS)): MSE between post-body
     `perceptualSpace` and `_ir_pre_mask_input` at masked positions
     (on the per-word grammar path this slot is the D3 reverse($S$)
     reconstruction instead).
   - `reconstruction_reverse` (concepts-seeded): the reverse pass
     seeded from the terminal `ConceptualSpace` STM snapshot (the
     `<reconstruct>` enum was retired), weighted by
     `reconstructionScale`; skipped at train time when D3 already
     carries the reverse objective (dedupe).
   - `embedding_sbow` (`JOINT` / byte-lexer perceptual SBOW), weighted
     by `<embeddingScale>`.
   - `arma` (sentence-level): `InterSentenceLayer.observe(s_t)` MSE
     between the ARMA(p, q) prediction and the current sentence rep,
     weighted by `armaScale`.
   - `intra` / `inter` / `inter_contrastive`: the intra-sentence
     predict-then-perceive term, the inter-sentence end-state term,
     and the InfoNCE next-idea contrastive term.
   - Optional, default-off (weight 0.0 unless configured):
     `conceptual_sbow`, `definition_sparsity`, `answer`, `thinking`,
     `predict_next`, `leaf_distill`, `gate_l1` --- plus any auxiliary
     terms the pipeline Spaces wrote to their shared `Error` instance.
7. One `backward()` + `optimizer.step()` per DataLoader yield.
8. Embedding training (`CBOW`/`SBOW`/`BOTH`) runs once per batch.

---

## SBOW vs CBOW

| Property | CBOW | SBOW | Masked Prediction (Phase 2) |
|----------|------|------|----------------------------|
| Targets per sentence | N (every in-vocab word) | N (every word) | N (one per masked pos) |
| Context | Padded LOO mean of other in-vocab words | LOO centroid of N-1 | All unmasked |
| Positive updates | N per step | N per step | N per step |
| Repulsive force | K=64 random negatives pushed from the context mean | random out-group to torus antipode | Implicit via MSE |
| Signal density | High | High | High |
| Loss | two `-logsigmoid` negative-sampling terms (wrapped-MSE scores) | two `-logsigmoid(sim(...))` terms | MSE over embeddings |
| Updates embeddings | Yes (own optimizer) | Yes (own optimizer) | Only for `BACKPROP`/`BOTH`/`JOINT` |
| Used in (`<trainEmbedding>`) | `CBOW` | `SBOW`, `BOTH`, `JOINT` | `BACKPROP`, `BOTH`, `JOINT` |

Both embedding trainers now make every in-vocab word a target under negative
sampling; neither uses a full-softmax projection head. The remaining
difference is geometric: CBOW queries a uniform padded context mean and
pushes $K = 64$ random negatives away from it, while the streaming SBOW
builds a Gaussian-weighted leave-one-out pode and pulls random out-group
vectors toward its torus antipode.

---

## Integrated checkpoint architecture

| Artifact | Example | Contents | Updated by |
|----------|---------|----------|-----------|
| XML config | `data/MM_20M_fineweb.xml` | Architecture, objectives, corpus and checkpoint path | Hand-edited |
| Checkpoint | `output/MM_20M_fineweb.ckpt` | Model and embedding state, vocabulary/BPE extras, optimizer, counters, RNG and corpus manifest | Phase 2 |

The integrated checkpoint is resumable. Mid-epoch resume requires the same
corpus manifest and batch size; mismatches fail loudly instead of replaying a
different cursor. `--force-embeddings` remains a deliberate Phase-1 rebuild,
not a separate model artifact contract.

---

## Embedding Space Geometry

`Lexicon` defaults to projective unit-ball lookup in the model, while the
streaming SBOW trainer still constructs its training table with `ball=False`
for the legacy torus pode/antipode objective. See [Lexicon.md](Lexicon.md) for
the current geometry and the migration note.

---

## Profiling

`--profile` wraps Phase 2 in `cProfile`:

```bash
python bin/train.py --model data/BasicModel.xml --profile --max-docs 500
# make train_micro is the capped micro run (no --profile flag of its own;
# invoke bin/train.py directly to add it). make bench_remote profiles the
# recon bench on ArborStudio via torch.profiler, not cProfile.
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
