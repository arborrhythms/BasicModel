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
masked positions)` via `compute_masked`. On the per-word grammar path with
`detachedReverse=true`, the `reconstruction` slot instead trains the
idea-only reverse construction chooser from $\operatorname{stopgrad}(S)$.
The old D3 reverse-pipeline loss remains available only to legacy configs and
compatibility evaluation. The supervised output-head loss is also back
(2026-05-28): the `output` channel is scored at weight 1.0
whenever labels exist (unlabeled corpora degrade it to zero). Two carve-outs
qualify "reconstruction is always concepts-seeded": at train time either the
detached reverse chooser or legacy D3 path **dedupes** the separate
`reconstruction_reverse` term (`lossIn` already carries the reverse
objective, so the concepts-seeded reverse is skipped to avoid double
counting), and at serial EVAL the decode consumes
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
| `reconstructionPriority` | Remove opposing output-gradient components on optimizer-owned PartSpace, WholeSpace, and ConceptualSpace parameters, then cap their norms relative to reconstruction. BasicModel enables this; legacy configs default off. Uses separate autograd traversals and one optimizer step. |
| `outputGradientRatio` | Maximum protected output/reconstruction gradient-norm ratio, per parameter tensor; default `0.5`, required `0 <= ratio < 1`. A missing/zero reconstruction gradient gives output no budget there. Independent SymbolSpace/OutputSpace heads retain ordinary credit. |
| `detachedReverse` | On the serial grammar training path, replace D3 trace replay with the static idea-only reverse chooser. Its input is `stopgrad(S)` and its targets live in `SymbolSubSpace.reconstruction_stack`. |
| `leafDistillWeight` | With `detachedReverse`, weight the chooser's bounded exact-leaf surface term. Without it, retain the legacy standalone root-to-leaf distillation head. |
| `forwardGrammarWeight` | Weight the bounded, one-fold structural contrast recorded for committed unary/binary grammar choices. `0` disables this branch. |
| `<reconstruct>` | RETIRED (A1, 2026-06-09; `reconstructEnum` removed).  There is no longer a target space-role knob: reconstruction is unconditionally concepts-seeded from the terminal `ConceptualSpace` STM snapshot, weighted by `reconstructionScale`. |

### Construction trace and derivation pressure

The forward construction teacher now has one owner:
`SymbolSubSpace.reconstruction_stack`. Before a compiled sentence body runs,
the eager boundary stores detached exact leaves, lossless radix word
spellings, and any available durable word-concept IDs. The body then records
committed unary/binary grammar choices in a fixed
`[B, 3*W + stmCapacity - 1]` rule-ID/arity/mask slab. There is no model-owned
parallel leaf or reduction cache, and no Python trace append in the captured
loop.

That trace supervises the *reverse chooser*; it never becomes a chooser input.
`ReverseConstructionChooser` is statically registered under `SymbolSubSpace`
before optimizer creation and predicts the active arity, global rule ID, and
exact percept leaf at every fixed slot. Its live gradient boundary is:

$$
\mathcal L_{\mathrm{reverse}}
= \operatorname{CE}\!\left(
q_\phi(\text{word parts},\text{rules}\mid
\operatorname{stopgrad}(S)),\;T_{\mathrm{stack}}
\right).
$$

The implementation detaches $S$ inside the chooser and detaches every teacher
artifact when it enters `ReconstructionStack`. Reverse therefore learns the
observed construction from the completed idea without sending a derivative
through any of the $W$ recurrent forward folds. The remaining trace-replay
reverse is a compatibility/evaluation path; canonical `BasicModel.xml`
training does not call it.

For the *forward* chooser, whole-sentence degraded reconstruction is useful as
a small auxiliary but is a poor primary signal: a jointly trained decoder can
collude with the encoder, punish legitimate paraphrases, and recreate the same
long recurrent Jacobian that produced the W=256 instability. The implemented
pressure is local to each candidate fold. For every type-valid unary or binary
rule, the layer re-scores detached children and detached candidate results
through the same chooser parameters. It uses the bounded proxy

$$
E_r = 0.2\,d(c_r,\operatorname{clip}_{[-1,1]}(c_r))
    + 0.8\,d(\tanh(c_r),\tanh(x),\tanh(y)),
\qquad
\mathcal L_{\mathrm{forward},t}
= \tfrac12\left[\sum_r p_\theta(r\mid x,y)E_r
+ \operatorname{rank}(r^*,r^- )\right].
$$

The first distance is an idempotent signed-carrier closure and the second is a
child-incidence preservation proxy (one child for unary, both for binary).
This is deliberately **not** claimed to be the data-derived FCA double-prime
closure; the latter remains a stronger future replacement once a bounded
codebook closure query is available. The expected energy plus pairwise rank
term lies in $[0,1]$. It is copied into a fixed `ReconstructionStack` loss slab
and averaged only over committed folds. Because the evidence tensors are
detached, credit assignment ends at that chooser call and cannot reopen the
sentence recurrence. The forward chooser is not trained by cross-entropy
against its own argmax trace, which would be self-confirming evidence.

`BasicModel.xml` currently leaves `forwardGrammarWeight=0`. A July 2026 MPS
production probe showed that its extra candidate re-score is substantial when
paid at every live fold. The objective and its compiler-safe fixed loss slab
remain available and tested, but production activation waits for a dedicated
quality/throughput measurement (or a compact-evidence re-score at the eager
sentence boundary). This cost decision is independent of `detachedReverse`,
which remains enabled as the numerical-stability gate.

The literature supports this split rather than an exact global inverse:

- Original DisCoCat lifts grammatical reductions to semantic morphisms that
  compose constituent meanings into a whole; it does not require those
  generally information-losing maps to be invertible
  ([Coecke, Sadrzadeh, and Clark, 2010](https://arxiv.org/abs/1003.4394)).
- Functorial language models obtain a probability distribution over word
  sequences from a grammar-to-meaning monoidal functor, suggesting grammatical
  likelihood as a generative objective
  ([Toumi and Koziell-Pipe, 2021](https://arxiv.org/abs/2103.14411)). A recent
  DisCoCat tensor encoder also found that contrastive training over explicit
  derivations improves sensitivity to word order and predicate roles
  ([Lo et al., 2025](https://aclanthology.org/2025.starsem-1.25/)).
- FCA defines concepts as fixed points of the Galois connection, and formal
  concepts are sufficient optimal factors for reconstructing Boolean
  object--attribute incidence matrices
  ([Denniston, Melton, and Rodabaugh, 2013](https://arxiv.org/abs/1309.5134),
  [Belohlavek and Vychodil, 2010](https://doi.org/10.1016/j.jcss.2009.05.002)).
  For graded activations, fuzzy Galois connections extend the same construction
  to complete residuated lattices
  ([Belohlavek, 1999](https://belohlavek.inf.upol.cz/publications/Bel_Fgc.pdf)).

The bounded exact-leaf term in the detached reverse chooser is the retained
degraded surface objective. It updates only that reverse student; it neither
updates $S$ nor unfolds a grammar recurrence. Its role is an anti-collapse
check, while the local carrier/incidence contrast is the direct derivation
signal.

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

When `reconstructionPriority` is enabled and a supervised output loss is
active, the optimizer seam separates the primary branch gradients before the
single update. It removes output's opposing component on P/W/concept
parameters and caps the remainder by `outputGradientRatio`; SBOW remains an
auxiliary contribution. This is a first-order gradient constraint, not a
monotonic-loss guarantee under Adam or momentum. Unlabeled batches keep the
ordinary single backward. A detached reconstruction supplies no protected
output budget; the policy does not reconnect its graph.

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

The live aligned P/W-to-concept handoff uses plain Sigma:
`a_c = tanh(b_c + sum_i W_ci * evidence_i)`. Inputs are up to eight identified
part/whole-code references at a location, not fold RMS scalars; there is no
source-count mean or gate. BasicModel opts into weak incoming-connection L1
with `ConceptualSpace/conceptReadoutL1=0.01` (absent/zero disables it). Each
concept's weights/bias are gathered from `IndexedSigmaConceptsFromPercepts`
at sentence staging, then consumed by
the eager or compiled word loop. Sparse gradients use the compact row-local
optimizer and participate in the same reconstruction-priority parameter set.
Within the single optimizer step, a diagonal-metric L1 prox soft-thresholds
only admitted input connections. Its threshold is
`lr * (lambda / distinct_observed_concepts) / (sqrt(v_hat) + eps)`;
biases and unused slots are excluded. `concept_readout_l1` is a separate
detached reporting cost, not another backpropagated penalty. Output-gradient
projection stays unchanged; L1 itself is an intentional sparsity/accuracy
tradeoff, not a promise of monotonic reconstruction improvement. Policies are
batch-local, never applied in evaluation, and cleared at the next `zero_grad`
after a skipped update.
If the readout has no learning gradient (`grad=None`), it is left untouched:
the current detached reverse boundary can still disconnect it from accuracy
feedback, and sparsity alone must not collapse the definition. Coupled
reconstruction updates and detached no-update behavior are tested separately.
There is no second optimization step. Full processed native fold state remains
separate from this scalar readout and is released at batch/sentence teardown.
Checkpoint coefficient tensors and the conceptual vocabulary's reference
manifest must be restored together. See the unified spec, section 5.5, for
the remaining collective-decoding and benchmark acceptance gates.

Data flows through `SentenceStreamDataset` wrapped in `DataLoader`. The
ordered training list is split into `B = batchSize` contiguous slabs of
length `L = len(split) // B`; at step `t`, row `b` is item `b * L + t`.
Temporal context is coherent across steps. No per-epoch global shuffle.

When `<packSentences>true</packSentences>`, the real `runEpoch` optimizer
loop advances each contiguous row until the next complete sentence would
exceed `serialWordCapacity`, the raw byte slab, or the fixed sentence-root
FIFO. Thus one B-wide optimizer brick can contain a different number of
sentences in every row while preserving each row's corpus chronology. A
sentence never crosses a brick. Under the canonical `<analysis>meronomy
</analysis>` the brick budget and the word-to-sentence layout are counted
in the analysis ladder's units (whitespace and digit units included; the
joining space belongs to the sentence it follows), the same cut the stem
stages, so the packed layout and the staged unit mask agree by
construction (`WholeSpace.unit_spans_of_bytes`,
`InputSpace.sentence_unit_count`). The packer runs on the prefetch
thread, so the unit tiler reads a host copy of the boundary predicates,
primed when the predicates are built or updated and refreshed by every
main-thread tiling (never the accelerator parameters; a worker thread
without the copy fails loud); a boundary update between packing and
staging is caught by the stem's alignment check. Explicit word-to-sentence IDs drive the
row-local soft resets inside CSLang; the final sentence in each row remains
live through loss, backward, discourse, and the contextual concept update,
then resets at the eager brick boundary. The log reports complete sentences
per second at epoch or wall-clock-cap exit.

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
7. One `optimizer.step()` per DataLoader yield. Normally one `backward()`;
   active `reconstructionPriority` uses separate branch traversals before
   projection and the same single update.
8. Embedding training (`CBOW`/`SBOW`/`BOTH`) runs once per batch.

---

## What questions and the two primary costs

Since the What spec cutover (basicmodel 195b129 / 077c18d), every
training batch is a batch of `WhatQuestion`s (`bin/What.py`), not a bare
input/output tensor pair. `runBatch` derives them from the cursor rows
(`_questions_for_batch`: supervised when the dataset has outputs, present
otherwise, future on `predict` trials, inference at runtime) or from the
`<whatCurriculum>` schedule, and scores two independently weighted,
independently normalized primary costs recorded before `backward()`:

| Cost (`primary_costs()`) | Compares | Trains |
|---|---|---|
| `input_reconstruction` (+ `input_reconstruction_reverse`, `reverseReconstruct()`'s own cost) | the reconstructed input with the presented (or clean) input | the bottom-up understanding and its input-associated inverse |
| `answer_construction` | the response constructed by `reverseOutput()` with `Data.what(question)` (`_what_answer_target`, rows without an available answer masked out, never substituted) | the resolve step, the question conditioner, the dedicated synthesis layers, the output adapter, and the shared understanding under reconstruction priority |

The desired answer is resolved only after the model response is fixed and
enters loss preparation only. `reconstructionPriority` differentiates the
two branches separately at one parameter version, removes the opposing
output component on protected perceptual / conceptual parameters, caps the
remainder at `outputGradientRatio` times the reconstruction norm, and takes
one optimizer step. `what_report()` gives per-family means of both costs,
thinking statistics (episodes, mean iterations, forced-closure rate), the
hard-choice policy credit (`forwardGrammarWeight`) separately from the
continuous answer credit, and sentences/s. Full contract: the
[What spec, Section 9](specs/2026-07-27-teaching-modes-and-next-iteration.md#9-training-and-loss);
training *through* multi-iteration thinking episodes is the
[mathematical thinking specification](specs/2026-09-09-mathematical-thinking.md):
with `<whatThinkingIterations>` above one, `runBatch` drives
`Model.think()` (one `forward()`, iterated resolve steps, LIFO closure),
scores the root answers after parity, adds the `WhatStepChooser` policy
term (`<whatThinkingPolicyWeight>`), takes the one optimizer step, and
then ends the episode (`<whatThinkingDetach>`), all reported under
`what_report()["thinking"]` and `["policy"]["thinking"]`.

---


## Reconstruction objectives and the declared migration (2026-09-12)

What trains today on the serial per-word path:

- With `<detachedReverse>true</detachedReverse>` (the production
  `BasicModel.xml`, with `<teacherReconstruction>`), `lossIn` is the
  detached idea-only student `ReverseConstructionChooser`
  (`_detached_reverse_construction_loss`): its own parameters (an idea
  projection, a per-step slot embedding, a kind head and a rule head)
  predict the ReconstructionStack's *detached* arity, rule and leaf targets
  from a *detached* root idea. No gradient reaches the forward through it.
  `reverseReconstruct` is deduplicated out of the training step and serves
  evaluation (the trace-driven un-fold and the stage-walk reverse).
- Without it, `lossIn` is the D3 per-word reconstruction objective
  (`_d3_reconstruction_loss`, `reverse(S)` from the root scored against
  the unmasked input), or the masked-event loss where neither applies.

Declared migration (the
[compiled reverse-loops plan](plans/2026-09-12-compiled-reverse-loops.md)):
`<reconstructInLoop>` selects one bounded compiled traversal of the
completed sentence's retained derivation with the compose path's tied
inverse transforms (the existing `invertible=True` forward/reverse
pairing). Slice 1 (2026-09-12) is in: after the final seal, a
`torch.while_loop` replays the recorded choices backward from the root
(`_reconstruct_sentence_traversal`; the reverse steps
`LanguageSpace.reverse_binary_step` / `reverse_unary_step` through the
gate-free `generate_functional` inverses and the exact residual for
`chunk`/`sum` against the word's dictionary row), recovers each word's idea
at its position, and scores it against the word's retained reference (the
leaf the forward pushed: the word's symbol row's dictionary atom, its
object concept's where it has one, scaled by the word's signed activation;
built after the loop from the staged atoms and the loop's activation slab,
not carried, since the loop's autograd stacks every carry once per step;
until 2026-09-14 the reference was the unscaled word atom while the
forward folded the object atom, so the residual reverses were measured
against the wrong leaf); that cost is the idea-level diagnostic, and a per-row truncation flag reports a
derivation that did not account for a word. Slice 2 (2026-09-12) adds the
byte-level fidelity that is `lossIn`: each recovered idea is softly assigned
(`softmax(|cos| / 0.1)`: a symbol's value is its signed activation and its
identity its row, so the snap ignores the sign) to the sentence's own word rows (the retained
constituent references, bounded by the word width), the assignment's
expected bytes are scored by cross-entropy against the word's own bytes over
its byte window (the percept store's byte atoms), and the gradient flows
through the assignment into the recovered ideas and the tied inverses. A concept
row is an identity code, not a fold of the word's byte atoms (concept
rows are relational identities; the byte fold lives in the perceptual
ladder), so the tied inverse of the concept lookup is the snap to a row
and candidate surfaces belong to WORD rows. Admission retains each WORD's
UTF-8 bytes in its ConceptualSpace owner's `_row_surfaces`; an OBJECT owns
only an index back to its concept identity, then translates through the
current `word_concept_of_object()` association to the WORD row. Changing an
interpretation therefore does not copy or overwrite WORD bytes
(`bin/Spaces.py:19949`). The brick's bounded dictionary snapshot contains
only its staged WORD/OBJECT rows (`_ar_concept_lookup_rows`) and resolves
their candidate bytes from this store (`_ar_bank_bytes`); input part IDs
provide the byte-window shape and scoring targets, never candidate bytes
(`bin/Models.py:10791`).

Per-stage WORD stores and OBJECT row indices persist in `vocab_extras`
under `concept_word_surfaces`, including stage 0 when it owns the shared
identities (`bin/Models.py:4244`, `bin/Models.py:5483`). The existing
structural extras retain the current OBJECT-to-WORD association. Strict
load also materialises a saved lazy chunk prior before the key audit,
so no preparatory input pass is required (`bin/Models.py:4871`).

A missing WORD surface removes that candidate. A missing snapshot never
falls back to the presented words' bytes. A null candidate of similarity
zero and uniform bytes remains available: with scoreable targets and no
known candidates, the byte cost is `log(256)`, not a manufactured perfect
reconstruction (`bin/Models.py:11267`). Rows absent from the snapshot do
not enter the score. The
score is taken at the pop step, inside the traversal, and the loop carries
only the running sums (idea cost, byte cost, word count) besides the
reverse stack: the loop's autograd stacks every carried tensor once per
step, so a `[B, W, D]` slab of recovered ideas in the carry did not fit the
accelerator at the production width; the slab is kept only when
`_recon_keep_ideas` is set (the tests' diagnostic), and the expected byte
distribution is accumulated by scatter, never as one-hot tensors.
The reconstruction runs in two bounded passes that share the forward's
word index (`_reconstruct_sentences`): pass A, one `torch.while_loop` trip
per sentence slot up to the row's highest sentence id (a tensor bound; a
host sentence count specialised the graph once per distinct count, a
recompile of about 45 s on every brick), un-seals each sentence from its
end state: the top three STM slots and the depth at the sentence's end
(a relative sentence keeps its depth-3 end state, the three LTM slots; an
absolute one its single root), carried per sentence by the word loop for
intermediate ends and read from the sealed buffer for the row's last
sentence (the live root the loop stored at
the sentence's intermediate end; `S` after the final seal for the row's
last sentence) through the recorded seal binaries into its pre-seal stack;
pass B is one `torch.while_loop` over the word index, latest word first,
that loads a sentence's pre-seal stack at the word that ends it, undoes
the word's recorded unary, post-binary and pre-binary folds, and pops and
scores the word. Operand identity (contract 1): the trace records each
binary fold's operand concept rows (left = STM slot 1, right = slot 0,
read before the reduce moves the stack, including every intermediate
packed sentence seal; `record_choice` on the eager path, two bank slabs
on the compiled one), and an undo routes the
residual reverse to the operand that is a word of the sentence, on
whichever side the fold put it, with that word's retained reference; a
compound operand (a composite folded earlier) takes the residual. A fold
recorded without rows keeps the positional fallback: the seals fold
newest-first, so the k-th seal undone, last first, returns word `lo + k`
on the left, and a per-word fold's known operand is the pushed word on
the right. Where the recorded op is declared lossy (the set ops,
`part`, `whole`) or a balanced split (`lift`, `lower`), the recovered
ideas are not the words: the fidelity of those derivations is what the
byte cost measures and trains, not an exactness the reverses could
promise. Under the tied contract no legacy reconstruction runs: the
per-word reverse-from-S objective and the training call to
`reverseReconstruct` are skipped. Costs accumulate per sentence slot (`[B, slots]`,
reported) and the row cost is the mean over the sentences present. The
earlier form, one traversal per sentence over a schedule the width of the
row, cost a sentence count times the loop steps of the forward (about
150 s per brick at the production width); the two-pass form costs one
loop step per word. Measured on `data/BasicModel.xml` with
`<reconstructInLoop>` (B = 4, 400 documents, MPS, the eager compile
backend), the reconstruction adds about 0.7 s to the 8.4 s forward and
2 s to the 20 s backward of a batch. `BASICMODEL_RECON_PLACEMENT`
(`graph`, the default; `compiled`, its own compiled call after the
forward; `eager`) selects where the traversal runs, for the performance
protocol.

The answer-materialisation boundary (2026-09-15).
`Understanding.answer_program` owns each row's final sentence program:
its symbol rows, signed activations, compact conceptual leaves, identified
compose actions, reconstruction targets and three-slot end state. Each
`AnswerProgram` clones its tensors while retaining the current forward's
gradients (`bin/Understanding.py:23`). The capture publishes explicit
compiled outputs before reading them; packed sentence slots and the final
per-row programs share the same captured records
(`bin/Models.py:7864`, `bin/Models.py:11637`). Repeated thinking over the
same execution reuses that understanding (`bin/Models.py:9662`).

Resolution selects the understanding's program for identity/reasoning, or
the recalled sentence's frozen program for recall, and captures the
question's target-free context on the derivation (`bin/Models.py:8017`,
`bin/Output.py:68`). Discourse observation retains detached copies of those
captured sentence products, including separate packed slots
(`bin/Models.py:8612`). Subsequent staging and memory advances therefore
cannot replace a held answer's operands.

`_materialize_answer_idea` replays only the derivation's owned leaves and
compose actions through the shared forward grammar operators. It applies
one question conditioner at the conceptual width, once, to the root slot.
A missing program is explicitly unresolved and has no current-input
substitute; a future prediction without a concept-level predictor has no
program (`bin/Models.py:11377`). The replayed compose actions recover the
identified idea; the output walk chooses its own generate derivation.
The program's reverse targets remain reconstruction metadata.

Historical conditioner widths remain in the checkpoint registry, with the
singular alias retained for compatibility. Strict loading restores all
saved widths and optimizer adoption includes each once. Answer generation
actively calls only the conceptual-width module. On the compatibility
synthesis path the conditioner receives the concepts produced by the shared
inverse, before the dedicated conceptual answer operator transforms them
(`bin/Models.py:9057`, `bin/Spaces.py:22879`). This keeps question context
inside that operator's supervised learning path.

The idea enters the output walk as three opaque concept slots, newest at
slot 0. `_walk_operand` reverses the live slots onto the bounded traversal
stack; `<generate>` chooses how to unfold them, and the emitted concepts
are realised through the tied reverse body and perceptual chain
(`bin/Models.py:11493`, `bin/Models.py:9048`). The dense symbolic-state
dependency during resolution remains item 7 of the ownership plan.

The output walk's generate policy (2026-09-15, contract 5).
`LanguageSpace.generate_policy` is a linear chooser created only under
`<outputInLoop>`. Its parameters belong to SymbolSpace's explicit optimizer
list. A training `reverseOutput` samples its own actions from the chooser;
it does not follow the input's identified compose derivation. The loop
returns the sequence negative log probability of those choices. It draws
one fixed-shape random slab before entering the loop and detaches the
policy's top-slot features, so policy credit trains the chooser while the
ordinary realised-answer error trains the selected numerical transforms.

`runBatch` resolves supplied desired answers after generation, then applies
answer-error credit only to available `What.supervised` rows. For row `b`,
let `E_b` be the realised output's error against that supplied answer and
`C_b` the sequence negative log probability. The credit is
`mean((-E_b - baseline) * C_b)` over eligible rows, with detached errors and
the previous exponential moving return baseline (initially zero; updated
with decay 0.9). `outputPolicyWeight` multiplies this term in the actual
training total, and `output_policy` reports it. Zero weight, missing
supervision, present/input targets and prediction-only trials supply no
policy gradient. The previous batch's cost is cleared at entry even when
the next batch skips output generation. This follows the answer-error
credit pattern used by the thinking chooser; no gold parse is assumed.

Evaluation uses the chooser's highest-scoring action. The rule inventory
comes from the grammar's `<generate>` section, with the number of LHS
outputs determining binary/unary expansion; it can differ from `<compose>`
and can contain rules absent there (`bin/Language.py:14159`). The numerical
inverse kernels remain shared with reconstruction, while the catalogs,
policies and traversal state are independent (`bin/Models.py:11783`). An
explicit output-owned generate stamp can replay an output rule; an input
compose stamp cannot. The low-level `targets` argument is retained only as
an ignored compatibility argument. There is no `outputTeacherForcing` knob.
Reconstruction continues to follow the input's identified compose parse;
that parse is not assumed correct and provides no output imitation credit.

Checkpoints record stable generate-action keys. Loading maps chooser rows
and Adam moments by those keys, preserving stop and giving new actions
fresh weights and zero moments. Older checkpoints without keys map from
the legacy compose catalog (`bin/Language.py:14214`,
`bin/checkpoint_migrations.py:799`).
A completed constituent is popped into the emitted sequence and the walk
continues with its pending constituents. A row completes when no live slot
remains; exhausting the budget with pending slots reports truncation.
Emitted words are returned left to right.

The eager trace replay (`_reverse_reduce_unfold`) and the exact leaves
teacher (`_reverse_method1_leaves`) are retired (2026-09-13): in
evaluation `reverseReconstruct` with `<reconstructFromIdea>` takes the
sentence's end state and unwinds the recorded derivation through the
same traversal (`_recovered_word_ideas`), then realises the recovered
per-word ideas through the reverse chain.

The grammar ops treat the concept event as opaque (Language, "Concept
events are opaque to the grammar ops"): the lift/lower inner layers are
sized to the muxed concept width, so checkpoints written before
2026-09-13 re-initialise those layers on load.

Loop gradients (2026-09-13). The tying gate found that the reconstruction
cost reached the references but not the fold weights, and the cause is in
`torch.while_loop`'s autograd (torch 2.14 and 2.15 nightlies): the
per-trip checkpoints inherit the initial carry's `requires_grad`, so a
carry that enters a loop as plain zeros returns a zero gradient from every
trip. The chain across trips is cut, parameters used in the body are
credited from the last trip only, and closures loaded mid-loop get
nothing. The forward word loop was affected too: on the ladder fixture 26
of 42 parameters received no gradient through the loop and 13 more a
different one. Every loop's carries now pass through
`Models._carries_with_grad`, which adds a zero scalar leaf that requires
grad (a per-device anchor created eagerly, since a tensor factory with
`requires_grad=True` cannot be traced inside a compiled region); with it
the loop's gradients equal a plain Python loop's on every parameter.
`test/test_while_loop_gradients.py` pins the defect and the fix. The
same node also keeps its per-trip checkpoints alive after the brick
(it outlives the brick through the C++ graph), so
`Models._release_loop_checkpoints` drops them at brick entry, after the
optimizer step, walking only the previous brick's own graphs (from its
published tensors), so a pending graph held elsewhere keeps its loops; the STM's live state and the trace's loss slab, both
updated in place, are detached there too (Benchmarks, "Open"). The traversal is part of the tensor word
pipeline's sentence state (the compiled path and its eager `while_loop`
form); the legacy static scheduler produces no such state, and `lossIn`
there falls back to the D3 objective. A checkpoint carrying the detached
student's parameters loads under the tied contract with those keys dropped
(reported), and a tied checkpoint under `<detachedReverse>` rebuilds the
student fresh. Slice 3 (2026-09-12): with `<outputInLoop>`, `reverseOutput`
runs the resolved answer's generate walk as a second, separately compiled
`torch.while_loop` (`_output_generate_walk`): the top slot's `.where`
stamp is decoded in tensor form, a rule stamp applies that rule's tied
inverse (binary opens the slot above; unary in place), children are
stamped empty as the eager `unreduce` does, and the walk ends when no row's
top is a rule or the budget (`N - 1`) is spent (reported truncation). The
walk owns its state and termination and reads nothing of the input
reconstruction; the spaces' reverse chain and the output decode follow it
unchanged. It
is a change of learning contract, so before `<detachedReverse>` retires:

- objective: sentence reconstruction fidelity (byte cross-entropy over each
  word's window) plus the diagnostic idea-level cost, reported separately
  from linear-inverse accuracy;
- parameter ownership: the traversal owns no parameters; it moves the tied
  transforms and the forward's fold parameters; the student's parameters
  leave the state dict;
- gradient boundaries: the cost stops at the recorded discrete choices
  (credited through the existing policy credit) and at constituent
  references (indices); the two knobs are mutually exclusive;
- checkpoint migration: loading a checkpoint that carries the student's
  parameters under the tied contract ignores them; the reverse direction
  (tied checkpoint under the detached contract) rebuilds the student fresh.

## Epoch report: throughput and word units

`runEpoch` closes a packed epoch with one line, e.g.
`Packed training throughput: 412 complete sentences in 61.2s = 6.73
sentences/s (16 optimizer bricks, B=8, W=64; word units 99.6% of 5,120
words)`.  The per-batch `batch = k (Δ=…s)` lines give the steady-state
rate once compilation has warmed up.  `word units` is the assurance that
the model operates over words now that the analysis tiling is learned
rather than fixed (`<boundaryTypes>`, doc/Mereology.md): the fraction of
whitespace-delimited words of the epoch's presentations that were staged
as exactly one unit (`BaseModel.word_unit_fraction()`; counters reset per
epoch).  Under the canonical boundary types plain text reads 100%; a
numeral under `<digitWholes>` is deliberately several digit units, so it
counts as a word that is not one unit; the atomic cold start
(`<boundaryTypes>none</boundaryTypes>`) reads 0% until the boundary
learner has acquired the whitespace boundary (test/test_meronomy_ladder.py).

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

## Category utility and chunk admission (2026-09-10)

At the training path's sentence boundary `ConceptualSpace` commits the
presentation's utility observations (one per resolved unit: the concept
with its constituent rows and property rows as features; a concept counts
at most once per presentation) and the chunk proposals made by the reduce
step; a proposal that has recurred `admissionCount` times is admitted as a
concept over its member concepts. `category_utility` (Corter & Gluck 1992,
Laplace-smoothed, withheld below `utilityMinCount`) is the estimator of
the basic level (doc/plans/2026-09-10-meronomy-fold-ladder.md, contract 4).

An admitted phrase also gets a row of its own in the concept dictionary,
initialised from the additive composition of its members; when the
reduce pass chunks a pair that matches an admitted phrase, the folded
parent's STM reference is set to that row, so later reads and the
answer path use the phrase's row and the losses train it. The utility
counts, the phrase hits, admissions, rows and gains, and the acquired
predicates of split property rows ride the structural extras of the
checkpoint (doc/plans/2026-09-10-meronomy-fold-ladder.md, contracts 6-7).
