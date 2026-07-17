# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`).
Default values are loaded from `data/model.xml` and overridden by
model-specific configs. The schema is in `data/model.xsd`.

This document is the human-facing reference for common knobs and migrations.
`data/model.xsd` is authoritative for the complete accepted element set.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Several XML knobs expose the architecture's relation to LLMs, Formal Concept
Analysis, and DisCoCat. `dataType=embedding`, sentence prediction, and
reconstruction cover the LLM-like prediction/generation path.
`subsymbolicOrder`, `symbolicOrder`, `mereologyRaise`, `monotonic`, and the
codebook settings control the FCA-like concept order and part/whole lattice.
`serial`, `learning`, `transformChooser`, `categoryCodebook`, and grammar-layer
settings control the DisCoCat-like typed composition path.

Overlay merge order:
`data/model.xml`  $\to$  `<config>.xml`  $\to$  runtime overrides.

Note on naming: the schema and the XML configs use `nInput` / `nOutput`
(sequence-length sentinels), `nDim` (per-vector dim), `nVectors`
(codebook size), and `nInputDim` / `nOutputDim` (raw shape hints for the
LiftingLayer). Older docs sometimes call `nOutput` / `nInput` `nActive`;
treat that as the same thing as `nOutput`.

> 2026-05-28: `<nWhere>` / `<nWhen>` are retired from per-file configs.
> The band is architectural (`canonical_shape`, `bin/architecture.py`):
> (nWhere=4, nWhen=4) on every interior space (2026-07-09 multi-rung
> pass), (0, 0) on OutputSpace. The
> `.where` field is the canonical positional identifier; see
> [doc/old/2026-05-28-where-keyed-taxonomy.md](old/2026-05-28-where-keyed-taxonomy.md).
>
> 2026-06-16: `.where` is an **endpoint-sum bracket** `[start,
> end]` (angle = span center, magnitude = extent), the invertible
> `EndpointSumWhere` form. An instant (`start==end`) is byte-identical to the
> prior single-quadrature point. **Superseded 2026-07-09**: the
> endpoint-sum bracket is retired from the muxed band — `.where` is now
> the same 4-dim 2-rung quadrature ladder as `.when`, over the byte
> START only (LF period = `<wherePeriod>`, HF = `wherePeriod /
> <whereRungRatio>`); the analyzer-side `EndpointSumWhere` keeps the
> bracket form.
>
> 2026-07-04 (encoding pass): `.when` is the **4-dim 2-rung start ladder**
> (`WhenStartDurationEncoding`; onset only — duration left the band; the
> exact long-int clock addresses LTM, the Option-C hybrid). New
> `<architecture>` knobs: `<wherePeriod>` (default 8192 input bytes;
> `.where` period, decoupled from nObjects, warn-once raise-to-fit),
> `<whenPeriod>` (default $10^6$ ticks), `<whenRungRatio>` (default 32).
> Tense is the onset-vs-`now` relation; see [doc/Spaces.md](Spaces.md).

---

## `<architecture>`

Core model settings. Training and data parameters are in nested
sub-elements `<training>` and `<data>` (see below).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data/dataType` | string | `"numeric"` | The data space-role (was the retired architecture-level `modelType`), set under `<data>`: `embedding` (LM / chat with sequence processing — PartSpace owns the byte/word lexicon) or `numeric` (dense slab, e.g. MNIST pixels). The old `simple`+`passthrough` collapsed to `numeric`; `vq` was dropped (0 configs). |
| `reconstruct` | — | — | RETIRED (A1, 2026-06-09): the `reconstructEnum` / `<reconstruct>` element no longer exists. The ConceptualCombine now unconditionally integrates all three streams (PS + SS + CS), so reconstruction is always concepts-seeded from the conceptual space-role — there is no longer a selectable mode (`none` / `symbols` / `both` are gone). |
| `subsymbolicOrder` | int | `1` | Iteration depth of the P$\to$C$\to$S pipeline. `order > 1` partitions the symbolic codebook geometrically; the partition width is set by `conceptualWidth`. |
| `wherePeriod` | int | `8192` | The `.where` quadrature period in input BYTES (2026-07-04 encoding pass; decoupled from the retired implicit $\Sigma$ nVectors period). The build seam raise-to-fits for longer inputs, warning once — never silent aliasing. |
| `whenPeriod` | int | `1000000` | The `.when` v2 LF horizon in clock ticks (the LTM event-addressing range). The band is the SIMILARITY channel; absolute addressing rides the exact long-int clock (`when_time`). |
| `whenRungRatio` | int | `32` | LF/HF period ratio of the `.when` 2-rung start ladder (safe branch bound $\approx 35$ at the dense-support floor). |
| `processSymbols` | bool | `false` | Apply extra symbolic processing after Sigma. |
| `monotonic` | bool | `false` | Constrain invertible Sigma / Pi to $W \ge 0$ so the lift / lower chain is order-preserving on the parthood cone. |
| `ergodic` | bool | `false` | Ergodic exploration: eligible layers use `W_eff = bias * W + var * noise`; `bias`/`var` are gradient-energy buffers, not Adam parameters. See [Ergodic.md](Ergodic.md). |
| `naive` | bool | `false` | Materialise `W_eff` densely in `InvertibleLinearLayer`. Slower; debugging only. `false` uses sequential L / D / U triangular solves. |
| `serial` | bool | derived | Forward-dispatch mode. `true` = serial / grammatical (per-word `[B, 1, D]` body); `false` = parallel (whole-slab `[B, N, D]` body). If omitted, legacy configs derive it from `symbolicOrder > 0`; new configs should set it explicitly. |
| `symbolicOrder` | int | `0` | The symbolic ITERATION BUDGET $K$ of the conceptual wave (v3 iterated loop, 2026-07-03; the pump stays purely subsymbolic — `cs_symbolic_phase` snaps the settled field to the snap block, then the wave $a^{i+1} = \tanh(W[a^i \mid 1] + s)$ runs $K$ steps over the single untyped square `ConceptualAttentionLayer` at the cutover). NOT forced ramsification: $K$ is the maximum possible conceptual order — the order reached if a novel concept were introduced at every iteration; a depth-$d$ vine completes at iteration $d$. `0` = no symbolic phase (byte-identical legacy path); `serial` stays an independent knob. Default derives from the grammar for legacy configs (1 when `useGrammar != "none"`, else 0). See [Architecture.md sec A](Architecture.md) and [STM.md](STM.md#2-serial-sequencing). |
| `subsymbolicNoop` | string | (empty) | Pass indices whose per-pass stack layer is the IDENTITY pass-through, e.g. `"0,2"` — the slot still occupies its pass so the pump always runs to `subsymbolicOrder`. The P4 sigma/pi stacks (DISTINCT per-pass subsymbolic layers — PS `sigmas[t]` / WS `pis[t]`, `t` in `0..subsymbolicOrder-1`; depth IS mereological order, the stack is the subsymbolic reasoning engine) are now CANONICAL and always built: pass 0 is the base `sigma`/`pi`; construction is RNG-neutral (save/restore), so an order-1 config (the single reused fold) is byte-identical. The old `subsymbolicStack` boolean toggle is retired. |
| `sparseReplace` | — | — | RETIRED (2026-07-02 P3 two-phase forward): phase separation makes non-replacement structural — the symbolic phase's outputs feed the SS leg, the head-side losses, and the concept table, never substituting the subsymbolic advance. Parsed only to emit a `DeprecationWarning`. |
| `routerWireSerial` | string | `"both"` | Per-word router-fire gating on the serial path: `per-word` (fire per word, boundary off), `boundary` (fire only at the sentence boundary), `both` (default — both fire), `off` (neither). The per-word fire populates `symbolSpace.current_rules` for SS dispatch. See [STM.md Section 7](STM.md#7-per-word-router-firing). |
| `learning` | bool | `false` | Two-pass soft-superposition training for the grammar chooser. When true, each TRAINING batch runs twice as two trials: pass A at superposition temperature 0 (sharp, recorded) and pass B at `exploreTemperature` (flatter exploration, trimmed from the batch error). The chooser is in the gradient path directly. Default off $\to$ one ordinary forward (byte-identical). See [Language.md $\to$ Soft-superposition route](Language.md). |
| `exploreTemperature` | decimal | `0.5` | Superposition temperature $t \in [0,1]$ for pass B of `learning`. `0` = the chooser's own (sharp) softmax, `1` = uniform (flat). Route scores are scaled by `1 - t`. |
| `transformChooser` | string | `"anchordot"` | Placement scorer for the structured grammar layers. `anchordot` = stateless cosine-to-anchor (byte-identical default, no new params); `mlp` = learned `MLPTransformChooser` (owns tool-embedding + MLP params $\to$ deliberate fresh-basin cutover). See [Language.md](Language.md). |
| `reconstructFromIdea` | bool | `false` | Reverse from the idea rather than replaying the forward parse. When true, `reverse()` clears the grammar/routing traces left by comprehension, then asks `SymbolSpace.generate()` to infer a fresh reverse rule path from the supplied idea snapshot. Default off keeps cached-derivation reverse. |
| `categoryCodebook` | bool | `true` | MetaSymbol participation-category codebook for the role-collapsed grammar: a small role-space `VectorQuantize` initialized with one prototype per labelled grammar role (`op_I1`, `op_I2`, `op_O1`, ...). Unsettled MetaSymbols accumulate bounded temporary role evidence; once mass/confidence/margin/stability thresholds are met, the MetaSymbol commits to one category id and its pending row is discarded. Structured grammar layers use the role context for all `transformChooser` modes: as an input feature for `mlp`, and as a labelled-role score prior for anchordot/default routing. See [Language.md $\to$ Participation Categories](Language.md). |
| `adverbEigEdit` | bool | `false` | Legacy/direct `LiftLayer` helper flag for the adverb sparse eigenvalue edit. The live `adverb` grammar operator force-builds the same zero-init projection and calls `LiftLayer.apply_adverb`, so ordinary grammar use does not depend on this flag. When enabled for plain `LiftLayer`, an adverb modifies a composed VP by `a2 = atanh(vp) + p_vp * delta_adv`, masked by the VP's own eigen-signature. Default off keeps plain `LiftLayer` byte-identical. |
| `mereologyRaise` | bool | `false` | Mereological order-raising: perception's autobind hook builds a meronymic lattice over the two towers and raises a higher-order PART when a whole accumulates more than `K_many` parts (abstraction order tracked via the ramsification table; provenance in `part_chain`). Enables the table on the PartSpace + terminal WholeSpace codebooks at build; `subsymbolicOrder` sets the max order. Default off $\to$ no table, no raising (byte-identical). See [doc/old/mereological-order-raising.md](old/mereological-order-raising.md). |
| `embeddingPath` | string | (empty) | gensim `KeyedVectors` path. Empty disables embedding load. |
| `weightsPath` | string | (empty) | Model weights checkpoint path. Empty falls back to `output/<name>.ckpt`. |
| `maxResponseLength` | int | `4096` | Inference token budget (characters / bytes / tokens). Caps output alongside `InputSpace.nOutput`. |
| `amp` | string | `"off"` | AMP autocast mode: `off`, `fp16`, `bf16`. |
| `truthBiasScale` | decimal | `0.1` | Strength of luminosity-based bias on concept input during forward. |
| `LuminosityWeight` | decimal | `0.1` | Multiplicative loss penalty for incoherent truths. |
| `UniversalityWeight` | decimal | `0.1` | Multiplicative loss penalty for unkind propositions. |
| `allowExcludedMiddle` | int | `1` | Catuskoti policy: `1` permits NEITHER, `-1` enforces classical LEM. |
| `allowContradiction` | int | `0` | Catuskoti policy: `0` forbids BOTH (non-contradiction), `1` permits paraconsistent BOTH. |
| `gateL1Lambda` | float | `0.0` | L1 sparsity penalty on lift / lower gates. `0` disables. |
| `loadBalanceWeight` | float | `0.0` | Sparse-MoE load-balance loss weight. Currently **inert / no-op**: the knob is read (`Models.py`) but has no consumer — the chart's sparse-MoE load-balance bookkeeping was retired with the chart itself, and the knob stands by for any future signal-router rule load-balancing. (`chartTopK` is itself retired.) |
| `l1Lambda` | decimal | `0.0` | Architecture-wide L1 penalty hook. Most configs leave this at `0` and use `WholeSpace.l1Lambda` instead. |
| `discontinuityLambda` | decimal | `0.0` | Architecture-wide discontinuity penalty hook (legacy). |
| `conceptualWidth` | string | `"tapered"` | Symbol-partition geometry for `subsymbolicOrder > 1`. `tapered` = geometrically narrowing slices; `uniform` = equal-width slices. |
| `maxActivePerLayer` | int | `8` | Maximum active nodes per layer for reverse / mereological walks. |
| `codebookRetire` | bool | `false` | Retire codebook path where supported. Mostly a migration hook. |
| `symbolLearning.enabled` | bool | `false` | Enables runtime symbol-learning hooks. Disabled by default. |
| `mereologicalTreeSize` | int | -- | Retired compatibility knob. Values are accepted but ignored; the codebook is the meronymic structure. |
| `writeSyntax` | bool | `false` | Dump POS-labelled syntax tree at end of every `MentalModel.forward()`. |
| `syntaxOutPath` | string | -- | Output path for `writeSyntax`. |

Luminosity and universality are **multiplicative** --- they scale the total loss by

$$
(1 + \mathit{LuminosityWeight} \cdot (1 - \text{luminosity}) + \mathit{UniversalityWeight} \cdot (1 - \text{universality})).
$$

`TruthLoss` (in `<training>`, below) is **additive** --- it directly
penalises propositions that contradict the TruthSet. Both coexist. See
[Ethics.md](../../doc/Ethics.md) Section Universality and
[Reasoning.md](Reasoning.md) Section TruthLoss for details.

---

### `<architecture><data>`

Data loading and filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | string | `"xor"` | Dataset key: `xor`, `mnist`, `tomatoes`, `text`, `inline`. Omit for inference-only models. |
| `numShards` | int | `1` | Shard count (streaming text datasets). |
| `maxDocs` | int | `10000` | Per-shard document cap. |
| `shardDir` | string | (empty) | Text shard directory (e.g. `data/fineweb`). Only used when `dataset=text`. |
| `minFrequency` | float | `0.0` | Vocabulary admission threshold (fraction of corpus). Words below it are held in a pending buffer until they accumulate enough occurrences. `0.0` admits everything. |
| `classificationMin` | float | -- | **Unparsed**: declared in `data/model.xml` / the schema, read by nothing in `bin/`. |
| `classificationMax` | float | -- | **Unparsed**: declared in `data/model.xml` / the schema, read by nothing in `bin/`. |

Inline-mode configs additionally accept `<input use="train|test|validation">...|...|...</input>`
and matching `<output>` elements directly inside `<data>`.

---

### `<architecture><training>`

Training loop and I/O.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numTrials` | int | `1` | Independent training runs per invocation. |
| `numEpochs` | int | `3` | Epochs per trial. |
| `batchSize` | int | `64` | Contiguous streams through the dataset. Each batch row `b` receives the next item from stream `b`, so temporal context is coherent across steps. Capped at split length, so small eval sets yield one rectangular batch. |
| `numWorkers` | int | `0` | DataLoader prefetch workers. `0` = synchronous in-process batch assembly. |
| `learningRate` | float | `0.001` | Adam learning rate. |
| `reconstructionScale` | float | `0.5` | Weight of reconstruction loss vs prediction loss in $[0, 1]$: $\mathcal{L}_{\text{total}} = (1-r)\,\mathcal{L}_{\text{output}} + r\,\mathcal{L}_{\text{recon}}$.  Legacy name `reverseScale` is still parsed with a one-shot deprecation warning. |
| `whatScale` | float | `0.7` | Loss weight on the `.what` (content) channel. |
| `whereScale` | float | `0.2` | Loss weight on the `.where` (positional) channel. |
| `whenScale` | float | `0.1` | Loss weight on the `.when` (temporal) channel. |
| `negSamples` | int | `64` | Negative samples per positive for CBOW / SBOW. Memory cost: $O(\text{batch} \cdot \text{negSamples} \cdot \text{dim})$ vs $O(\text{batch} \cdot \text{vocab})$ for full softmax. |
| `TruthLoss` | float | `0.0` | Additive penalty for propositions that contradict the TruthSet (union-norm reduction via `Basis.disjunction()`). `0.0` disables. |
| `maskRate` | float | `0.15` | Within-sentence IR Bernoulli mask rate. BERT default 0.15. |
| `trainEmbedding` | string | `"NONE"` | Embedding update mode: `NONE`, `CBOW`, `SBOW`, `BACKPROP`, `BOTH`, `JOINT`. See table below. |
| `embeddingScale` | float | `0.25` | JOINT-mode embedding loss weight $\lambda$: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{emb}}$. Ignored by other modes. (Older docs called this `trainEmbeddingRatio`.) |
| `autoload` | bool | `true` | Load weights from `weightsPath` at construction. Stale incompatible checkpoints fail fast. |
| `autosave` | bool | `false` | Save weights after training completes. |
| `checkpointEveryBatches` | int | `0` | Save mid-training checkpoint every N optimizer steps. `0` disables periodic checkpoints. |
| `profile` | bool | `false` | cProfile the training loop. |
| `certainty` | bool | `false` | Per-neuron certainty tracking in ergodic layers. Allows individual neurons to transition exploration $\to$ exploitation at different rates. |
| `sentencePrediction` | bool | `false` | Enables `InterSentenceLayer`, an ARMA(p, q) predictor over sentence representations. |
| `armaScale` | float | `0.1` | Loss weight for the ARMA sentence-prediction MSE. |
| `sentencePrimingScale` | float | `0.05` | AR prediction cast into `concept_dim` and added as a bias to `concept_input` before the sigma-pi loop. Scaled by `confidence * sentencePrimingScale`. |
| `intraLossWeight` | float | `0.1` | Loss weight on the in-STM next-idea term $\mathcal{L}_\text{intra} = \mathrm{MSE}(\hat{c}_t, c_t)$ from `IntraSentenceLayer` (owned by ConceptualSpace), added to the IR-loss path. `0` disables. See [STM.md Section 6](STM.md#6-intrasentencelayer). |
| `interLossWeight` | float | `0.1` | Loss weight on the inter-sentence next-end-state term $\mathcal{L}_\text{inter} = \mathrm{MSE}(\hat{p}, p)$ from `InterSentenceLayer`'s inter-level predictor (roots over the LTM chain). Mirrors `intraLossWeight`. `0` disables. See [STM.md Section 11](STM.md#11-inter-sentence-prediction). |

#### `<trainEmbedding>` --- Embedding Update Modes

| Value | Embedding method | Model layers | Gradients through codebook |
|-------|------------------|--------------|---------------------------|
| `NONE` | Frozen | Trained | No |
| `CBOW` | CBOW (padded context) | Trained | No |
| `SBOW` | SBOW (leave-one-out centroid) | Trained | No |
| `BACKPROP` | Backprop only | Trained | Yes |
| `BOTH` | SBOW post-batch | Trained | Yes (two optimizers) |
| `JOINT` | Single backward | Trained | Yes ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$) |

---

### `<architecture><server>`

HTTP server settings (used by `serve.py`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Bind address. |
| `port` | int | `8001` | Port. Override per model to run several in parallel. |
| `timeout` | int | `120` | **Unparsed**: declared in `data/model.xml`, read by nothing — `serve.py` reads only `host` and `port` from this block. |

---

## BasicModel / MentalModel Quick Reference

The two configurations that train the full P$\to$C$\to$S pipeline:

```xml
<architecture>
  <truthBiasScale>0.1</truthBiasScale>
  <LuminosityWeight>0.1</LuminosityWeight>
  <UniversalityWeight>0.1</UniversalityWeight>
  <subsymbolicOrder>1</subsymbolicOrder>
  <monotonic>false</monotonic>
</architecture>

<training>
  <TruthLoss>0.0</TruthLoss>
</training>
```

The grammar section of `MentalModel.xml` declares `lift` as an SS
binary rule:

```xml
<S>lift(S, S)</S>      <!-- lift is binary; SVO is assembled by
                            the chart-compose path S -> S VO,
                            VO -> V O, not a ternary lift -->
```

Transitive SVO is produced compositionally via typed grammar rules such
as `S -> S VO` / `VO -> V O`, or via STM `ParseState` extraction for
a subject lift over a transitive verb-phrase derivation. Grammar mode is
derived from the loaded grammar.

---

## Space sections

Every space block uses the same sentinel convention:

- `nInput = 0` --- InputSpace: derive from `TheData`; other spaces: derive from the previous space's `outputShape[0]`.
- `nOutput = 0` --- same as `nInput` for this space (passthrough count).
- `nVectors = 0` --- same as `nOutput` (used for non-codebook spaces).

Per-Space `<nWhere>` / `<nWhen>` declarations were retired on
2026-05-28 (see top-of-file note); the per-Space defaults live in
`data/model.xml` and don't need re-declaration in per-experiment configs.

### `<InputSpace>`

Lifts raw data into the model's internal representation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Sequence length: max tokens per input. For XOR: 2. For text models, OOV words are spelled out as individual characters, so a single word may consume multiple slots --- a short sentence with many OOV words can exceed this limit and trigger a truncation warning suggesting a lower `<minFrequency>`. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$. Default 9 = 1 content + 4 `.where` + 4 `.when`. |
| `nInputDim` / `nOutputDim` | int | `0` | Raw shape hints. `0` defers to the loader's native dim. |
| `nVectors` | int | sentinel | Codebook size. Defaults to `nOutput`. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. Legacy `true`/`false` are accepted as `quantize`/`none`. |

> **`<lexer>` moved to `<WholeSpace>`** (Phase 4b of the
> analysis/synthesis dual-input plan, rev. 2026-06-09): lexing is analytic
> CUTTING, owned by the analysis side. An InputSpace-side `<lexer>` now
> fails schema validation and the runtime reader. InputSpace emits the
> dual view — `forward(x) -> (percepts_in, concepts_in)`: the atom view
> (content `[B, N, 1]`) for the perceptual branch and the unity view
> (`[B, 1, N]`) for the symbolic branch.

`inputShape` uses the data's native dim (e.g. 784 for MNIST,
`inputLength` for text); `outputShape` uses `nDim` from
`TheObjectEncoding`. `LiftingLayer` bridges the two.

### `<PartSpace>`

Transforms lifted input into perceptual features via its synthesis fold —
a `SigmaLayer` (additive/union; the Pi/Sigma swap of the analysis/synthesis
plan, rev. 2026-06-09: PS is bottom-up SYNTHESIS over atoms; the
multiplicative Pi fold moved to WholeSpace as top-down analysis).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Number of active perceptual feature vectors. Should be $\geq$ `InputSpace.nOutput`. For XOR: 4. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$ (default 9 = 1 + 4 + 4). |
| `nVectors` | int | sentinel | Codebook size. When `> nOutput`, enables vector quantization with top-k selection of `nOutput` from `nVectors`. |
| `invertible` | bool | `false` | `true`: the PartSpace `SigmaLayer` uses the invertible LDU path for forward/reverse. |
| `hasAttention` | bool | `false` | DEPRECATED and INERT: the legacy boolean no longer constructs a QKV attention pass; it is kept only as a backward-compat alias. Superseded by `<attention>` (off / primer / second-order / low-rank). |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `synthesis` | string | `"lexicon"` | Bottom-up union strategy (renamed from `<chunking>`, which is rejected loudly): `lexicon`, `bpe`, `mphf`, `byte` (canonical; `none` is its internal alias), `radix`, `meronomy`. `radix` routes through `RadixLayer`; `meronomy` aliases the radix path with meronomy word-group bookkeeping. `analyse` was removed from PartSpace; top-down cuts live on WholeSpace `<analysis>`. |
| `butterfly` | bool | — | DEPRECATED per-space alias for the architecture-level `<sigmaPi>` enum (`true` $\to$ `butterfly`, `false` $\to$ `last`). The code default for `<sigmaPi>` is `butterfly`, so the FFT-style element-pair cascade on the PS fold (cross-element mixing on the flattened `[B, N*D]` view) is **ON by default**; configs set `false` to opt out. Required for `MM_xor.xml` convergence. See [doc/old/2026-05-26-two-loop-pi-sigma-substrate.md](old/2026-05-26-two-loop-pi-sigma-substrate.md). |
| `wordLearning` | int | `2` | Active lexicon-growth mode. `0` = frozen codebook; `>=1` = on first sight of a new word, insert into the lexicon and tag it on the meronomy. Several trained configs override this to `0` or `1`. |

Sigma layer math: $y_j = \tanh(W x + b)$ (the additive/union fold). The
PS `<codebook>` element was retired (the percept prototypes live on the
`.what` basis; radix's PerceptStore IS the surface codebook). See
[Architecture.md](Architecture.md) and [Philosophy.md](Philosophy.md) for
the analysis/synthesis orientation.

### `<ConceptualSpace>`

Transforms perceptual features into abstract concepts via Sigma layers
(additive / linear).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Active concept vectors. For XOR: 3. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$ (default 9 = 1 + 4 + 4). |
| `nVectors` | int | sentinel | Codebook size. |
| `invertible` | bool | `false` | `true`: `SigmaLayer(invertible=True)` (exact inversion via `atanh` + $W^{-1}$); `false`: separate `sigma1` / `sigma2`. |
| `hasAttention` | bool | `false` | DEPRECATED and INERT: no longer constructs an attention pass in conceptual processing; kept only as a backward-compat alias. Superseded by `<attention>` (off / primer / second-order / low-rank). |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `butterfly` | bool | — | **Silently ignored**: ConceptualSpace never reads a per-space `<butterfly>` — only PartSpace and WholeSpace carry the per-space read (the deprecated alias for architecture-level `<sigmaPi>`). A `<ConceptualSpace><butterfly>` element in a config has no effect. |
| `stmCapacity` | int | `8` | STM ring depth (within Miller's $7 \pm 2$ band). Per-batch buffer `[B, stmCapacity, nDim]` + depth pointers `[B]`. |

Sigma layer math: $y_j = \tanh(W x + b)$. See [Architecture.md](Architecture.md).
`InvertibleSigmaLayer` has been merged into `SigmaLayer(invertible=True)`.

> `ConceptualSpace` is now a bookkeeping carrier for STM/event state. The
> former per-stage `sigma_in` / `sigma_cs` layers are retired and no longer
> constructed on the live path; symbolic generalization lives outside CS.

### `<WholeSpace>`

Discrete symbolic representation --- the information bottleneck between
perception and output, and (post the analysis/synthesis plan, rev.
2026-06-09) the home of top-down ANALYSIS: SS owns the Pi fold
(multiplicative/intersection), the `<analysis>` division knob, and the
`<lexer>` intake knob. See [Language.md](Language.md) and
[Philosophy.md](Philosophy.md) for the design.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Total symbols. When reconstruction symbols are enabled, `nOutputSymbols = OutputSpace.nOutput` are fed to output; the rest carry reconstruction information. |
| `nDim` | int | `0` | `0` = inherit `ConceptualSpace.nDim` (`symbol_dim` must equal `concept_dim` under the ownership rule). A concrete positive value is allowed but must match `ConceptualSpace.nDim`. |
| `nVectors` | int | sentinel | Codebook size. When `codebook=quantize`, equals the number of symbol prototypes. |
| `codebook` | mode | `quantize` | `none`, `quantize`, or `project`. Required for the full symbolic pipeline; when a config omits the element the code hard-defaults WholeSpace to `quantize` (`Spaces.py`). **Pending removal** (backlog #13: SS `quantize` becomes hardwired) — deliberately deferred until the reverse decodes pre-snap $z$ (backlog #16 / the asymmetric recon gather), because `XOR_exact.xml` relies on `none` for its byte-exact chain. |
| `analysis` | string | `"byte"` | Top-down division strategy over the unity view: `byte`, `word`, `raw`, `sentence`, `grammatical`, or `meronomy`. The old `analyse` spelling is not accepted by the schema. |
| `lexer` | string | — | DEPRECATED spelling: the reader prefers the newer `<analyzer>` spelling, falling back to `<lexer>`; **both are deprecated in favor of the unified `<analysis>` knob** (its lexer-valued options `raw`/`byte`/`word`/`sentence` set the intake; `grammatical`/`meronomy` leave the lexer at `raw`). Intake granularity InputSpace executes; moved here from `<InputSpace>` (Phase 4b: lexing is analytic cutting — an InputSpace-side `<lexer>` is rejected loudly). |
| `nonlinear` | bool | `true` | Tanh-bound the activation. |
| `sortNetwork` | bool | `false` | Bitonic-sort regularizer on the codebook ordering. |
| `sortPasses` | int | `0` | Sort-network depth. `0` disables. |
| `l1Lambda` | decimal | `0.01` | L1 sparsity penalty on the symbolic activation. |
| `symbolResidualScale` | decimal | `1.0` | Symbol-residual injection scale (legacy mereology hook). |
| `outputSymbolResidualScale` | decimal | `0.0` | Output-side symbol-residual injection scale. |
| `commitmentBeta` | float | `0.25` | Space-level VQ-VAE commitment-loss $\beta$. **Asymmetric dispatch (2026-06-10)**: on the PARALLEL leg this is now INERT — the dispatch rewrite retired the space-level commitment, the VQ-internal cb_commit, and the nearest-target `symbol_residual` pull (the recon gather + STE carry the two asymmetric legs; the codebook is gradient-trained, `vq.ema_update = False`, `vq.commitment_weight = 0`). The SERIAL/grammar leg keeps the legacy coupling (this $\beta$ included) behind an explicit gate until its own asymmetric recon leg lands (D-territory). |
| `semanticArrangement` | float | `0` | Task 5 (C-13): post-sentence semantic arrangement over SS heat — pode (activated-rows centroid attraction) + antipode (rest-of-codebook repulsion), gradient on the active rows only. `0` = OFF (the default); the semantic payoff is validated under D, not XOR. |
| `decorrelationWeight` | decimal | `0.0` | `ImpenetrableLayer` decorrelation regularizer on codebook rows. |
| `spectralFlatnessWeight` | decimal | `0.0` | `ImpenetrableLayer` spectral-flatness regularizer. |
| `truthCriterion` | unitInterval | `1.0` | **Single continuous truth bar** governing BOTH (a) WholeSpace truth **recording** --- a per-cell activation is recorded into the TruthLayer iff its clamped magnitude $\ge$ `truthCriterion` (fires during training and `store_truths` gold ingestion alike), and (b) learned relative-sentence **acceptance** --- a relation `predicate(idea1, idea2)` enters the SS codebook iff its learn-score $\ge$ `truthCriterion` (learn-score $= \text{children\_in\_codebook} \times \text{is\_truth\_obvious} \times \text{resolves\_contradiction}$). At `1` nothing is recorded/learned; at `0` everything is. Read onto both ConceptualSpace and WholeSpace; per-space override of the `<architecture>` value. Replaces the retired binary `<accumulateTruth>` / `<truthMinMagnitude>` switches. See [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states). |
| `trust` | unitInterval | `1.0` | **Architecture-level only** — read as `architecture.trust` (`Models.py`); a `<WholeSpace><trust>` element is not read. Model-level multiplier for incoming assertions/testimony. Runtime `store_truths` entries and static `<truthSet>` rows use `effective_trust = trust * incoming_trust` (clamped to `[-1, 1]`) before they enter the TruthLayer/LTM; persisted STM descriptions also use it, with absolute rows storing event trust and relative rows storing `trust * (t - f)`. The existing `truthCriterion` and TruthSet factors still decide whether testimony is strong enough to support learned relations. |

WholeSpace owns one square invertible
`PiLayer` at `self.pi` — the top-down analysis (product/intersection)
fold — bridging the C $\leftrightarrow$ S boundary in both directions
via its exact inverse (the Pi/Sigma swap, rev. 2026-06-09; SS owned a
SigmaLayer at `self.sigma` before the corrected orientation). Consumers
use `model.wholeSpace.pi` directly; the grammar rule binding registers
it under both the `pi` rule name and the legacy `sigma` alias.

### `<OutputSpace>`

Maps symbols to final predictions via linear layers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Output values. For XOR: 1. |
| `nDim` | int | `1` | Per-output dim. |
| `nVectors` | int | sentinel | Codebook size. Defaults to `nOutput`. |
| `invertible` | bool | `false` | Rarely set on OutputSpace. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `nonlinear` | bool | `false` | OutputSpace is linear by default; rescales via `Data.denormalize`. |
| *(readout — retired 2026-06-19)* | — | — | The OutputSpace regression head (backlog #11): with `nVectors == 1` (or no codebook) it is an unquantised linear `nInputDim -> nOutputDim` plus a learned scalar intercept. The former `<readout>` enum (`identity` \| `sigmoid`) was retired — the head is always linear+bias (a binary $\{0,1\}$ target like XOR is regressed, not squashed; see `OutputSpace.lrScale` + `subsymbolicOrder=3`). Ignored by quantised heads. |

`LinearLayer` with `(bias, temp)` support for ergodic mode.

---

## `<SymbolSpace>`

Grammar infrastructure (SyntacticLayers, TruthLayer). Lives outside
`<architecture>` as a top-level sibling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `syntacticHiddenDim` | int | `64` | SyntacticLayer MLP hidden size. |
| `truthMaxEntries` | int | `1024` | TruthLayer SS-codebook capacity. |
| `ltmCapacity` | int | `1024` | Long-term-memory (LTM) chain capacity on `InterSentenceLayer`: the per-row bounded `deque` of STM end-states (the AR sequence feeding inter-sentence prediction) is capped at this many entries. Separate from `truthMaxEntries`. See [STM.md Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states). |
| `parserBackend` | — | — | RETIRED (Stage 3 cleanup, 2026-05-27): the chart backend is gone and the signal router (`LanguageLayer`) is now the sole, fixed parser. The element is rejected — leaving it in an XML config raises a loud `ValueError` at load time (`Language._assert_retired_chart_knobs_absent`). |
| `routerKind` | — | — | RETIRED (Stage 3 cleanup, 2026-05-27): there is no longer a selectable router; the signal router is fixed. Rejected with a loud `ValueError` if present in a config. |
| `chartCollapse` | string | `"root"` | **Unparsed**: declared in `data/model.xml` / the schema, read by nothing in `bin/` (the writeback contract it described rode out with the chart). |
| `chartTau` | — | — | RETIRED (Stage 3 cleanup, 2026-05-27) alongside the chart. Rejected with a loud `ValueError` if present in a config. |
| `wMax` | — | — | RETIRED: the legacy STM-capacity alias is no longer read. STM depth comes from `<stmCapacity>` (or the `DEFAULT_CAPACITY = 8` fallback). A config that still sets `<SymbolSpace><wMax>` now raises a loud `ValueError`. |
| `chartTopK` | — | — | RETIRED (Stage 3 cleanup, 2026-05-27) alongside the chart. Rejected with a loud `ValueError` if present in a config. |
| `chartNoiseEps` | — | — | RETIRED (Stage 3 cleanup, 2026-05-27) alongside the chart. Rejected with a loud `ValueError` if present in a config. |
| `iterationsPerWord` | int | `1` | **Unparsed**: declared in `data/model.xml` / the schema, read by nothing in `bin/`. |
| `downwardGeneration` | bool | `false` | Emit a codebook head via `SymbolSpace.reconstruct()` and stash on `self._predicted_head`. |
| `language.interpretation` | float | `0.5` | Soft-interpolation weight between forward and generation directions. |

The `<language><grammar>` block contains space-role-scoped rules in
`<compose>` and `<generate>` sections, each holding `<percepts>` /
`<concepts>` / `<symbols>` sub-blocks of `<rule>` elements. Per the
2026-05-07 rollback, the grammar XML is the sole source of truth for
which rule fires at each space-role (no code-level `default_rule` fallback).
Legacy bare-space-role elements like `<P>P = sigma(P)</P>` directly under
`<grammar>` are still accepted.

---

## Additional parsed knobs (2026-07 audit)

Knobs that are parsed and live in `bin/` but were previously undocumented
here. Every default below was verified against its read-site; `data/model.xsd`
carries the full schema commentary. Site names are module + owning
symbol (line anchors drift).

### `<architecture>` level

| Knob | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `meronomy` (attr `dMaxStable`) | `Layers.py` (`meronomy_enabled` / `meronomy_d_max_stable`) | code `off`; `data/model.xml` ships `on` with `dMaxStable="4.0"` $\Rightarrow$ effectively **ON** | Binds the meronymic slots (PS.sigma / SS.pi) to the membership kernels; `dMaxStable` clamps the contractive diagonal. |
| `sigmaPi` | `Models.py` (BaseModel init) + `Spaces.py` (PS/WS fold builders) | `butterfly` | Fold span enum `last` \| `butterfly` \| `full`; the per-space `<butterfly>` boolean is its deprecated alias (PS/WS only). |
| `whereRungRatio` | `Spaces.py` (`WhereEncoding` construction) | `32` | LF/HF period ratio of the `.where` 2-rung ladder; HF period = `wherePeriod / ratio`. |
| `syntacticOrder` | `Models.py` (BaseModel init) | `0` | Parse-tree depth cap for the serial reduce sweep; `0` = unbounded. Inert in parallel mode. |
| `sentenceProtocol` | `Models.py` (BaseModel init) | = `serial` | Whole-sentence gist prelude (parallel `subsymbolicOrder` pumps, intent-only commit) before the serial per-word loop. |
| `truthSet` (`<truth>` rows: text, `trust` / `kind` attrs) | `Models.py` (`provision_ltm`) | (none) | Config-provisioned trusted truths run through the real forward and appended to the consolidated LTM at load; row trust $\times$ `architecture.trust`. Read only when `<ltmConsolidation>` is on — otherwise ignored. |
| `thinkingBudget` | `Models.py` (BaseModel init) | `0` | Thinking-kernel op budget per top-level `think()` frame; `0`/absent = off, positive attaches the kernel's certified result to `answer_query`. |
| `reasoningIterations` | `Models.py` (BaseModel init) | `1` | Truth-grounded reasoning chain depth; a query routes to the recurrent tool-use loop at this depth. `0` = old generative-infer behavior. |
| `queryReasoning` | `Models.py` (BaseModel init) | `false` | DEPRECATED alias: `true` $\Rightarrow$ `reasoningIterations = 10`. Read only when `<reasoningIterations>` is absent. |
| `ltmConsolidation` | `Models.py`, `Language.py` (SymbolSubSpace) | `false` | Unifies the discourse LTM chain + RelativeTruthStore into one persisted `TernaryTruthStore` (`ltm_store`). |
| `stateless` | `Models.py`, `Language.py` | `true` | Request-scoped user TruthSet rows: on state-dict load the consolidated LTM is revived without `ORIGIN_USER` rows. `false` = stateful deployment. |
| `globalAttention` | `Models.py` (BaseModel init) | `false` | Typed addressable attention over input window / STM / LTM / codebook; the soft-read is parked on `_global_attention_obs`. |
| `globalAttentionConsume` | `Models.py` (BaseModel init) | `false` | Feeds the parked soft-read back into the head (`Finish`) as a zero-init gated residual. Requires `<globalAttention>`. |
| `readingAttention` | `Models.py` (BaseModel init) | `false` | Learned `.where` scope producer at each $t>0$ subsymbolic pass (feeds the `<mereologyRaise>` handoff). |
| `relevance` | `Models.py` (BaseModel init) | `false` | Relevance-integration gate (Architecture.md sec C). |
| `primingDecay` | `Models.py` (BaseModel init) | `0.9` | Priming-energy decay per prime event. |
| `primingSpread` | `Models.py` (BaseModel init) | `0.25` | Fraction of a connected row's standing energy diffused to neighbors per prime event (live by default; `0` = pure decay+bump). |
| `stmReduceTau` | `Models.py` (BaseModel init) | `0.5` | Mid-reading opportunistic STM reduce: fold the top-2 where the reducer DP's reduce marginal exceeds $\tau$. |
| `radialStmReduce` | `Models.py` (`_create_per_stage`) | `false` | STM idea folds use the radial (signed-safe) radmin/radmax kernels. |
| `symbolicPriming` | `Language.py` (`attach_knowledge` $\to$ `configure_priming`) | `false` | Taxonomy forward heat production (symbolic-heat retrieval). |
| `symbolTower` | `Models.py` (BaseModel init) | `false` | 3-stream CS bind (PS + WS + SS): SymbolSpace becomes the symbol tower with its own codebook stream. |
| `serialObjectMeta` | `Models.py` (BaseModel init) | `false` | Serial-only: per-word hard mask to the active word's span; STM push + eager mint fire once per word. |
| `conceptIndexRead` | `Models.py` (BaseModel init) | `false` | Serial per-word idea reads through the index to the concept's `similarity_codebook` row instead of the computed percept-binding event. |
| `ideaDecode` | `Models.py` (BaseModel init) | `false` | Chart-free reverse: generate the surface from the idea alone (no `generate_rules` rebuild). |
| `verbSpectrum` | `Language.py` (`LiftLayer` init) | `false` | Verb eig-spectrum operator on the composed VP (VP parameterization). |
| `prediction` | `Models.py` (BaseModel init) | `"none"` | `predictionEnum`: `none` \| `interSentence`; stored as the exact enum string for dispatch. |
| `predictionTrialRatio` | `Models.py` (ModelLoss wiring) | `0.0` | Fraction of training batches run as pure next-idea prediction (recon terms zeroed). |
| `overlapWhereTiling` | `Models.py` (BaseModel init) | `false` | Experimental overlapping PS/WS `.where` lattice; requires `<mereologyRaise>true</mereologyRaise>` (loud `ValueError` otherwise). |
| `continuityNorm` | `Mereology.py` (contemplative mixin) | `'inf'` | Norm for the continuity measure (`'l2'` available). |
| `continuityRatioTarget` | `Mereology.py` | `1.0` | Target ratio for the continuity measure. |
| `continuitySharpness` | `Mereology.py` | `1.0` | Sharpness of the continuity squash. |
| `TetralemmaPolicy` block (+ per-space `tetralemmaOverride`) | `util.py` (`tetralemma_policy`) | `allowExcludedMiddle=1`, `allowContradiction=0`, `neitherThreshold=0.1` | Shared catuskoti policy under `<architecture>`; a space sets `<tetralemmaOverride enabled="true">` + its own `<TetralemmaPolicy>` to override. Legacy top-level `<TetralemmaPolicy>` still read as fallback. |

### `<architecture><training>` level

| Knob | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `seed` | `Models.py` (run entry; env `BASIC_SEED` overrides) | unset | RNG pin (torch/python/numpy) for reproducible single-CLI runs; the `XOR_exact` gates rely on it. |
| `answerLossWeight` | `Models.py` (BaseModel init) | `0.0` | Reasoner answer loss weight. |
| `predictNextLossWeight` | `Models.py` (BaseModel init) | `0.0` | Next-idea blend loss weight (`reason_predict_next` / `NextIdeaScorer`). |
| `thinkingLossWeight` | `Models.py` (BaseModel init) | `0.0` | Thinking-kernel loss weight. |
| `leafDistillWeight` | `Models.py` (BaseModel init) | `0.0` | Leaf-distillation loss weight. |
| `interContrastiveWeight` | `Models.py` (ModelLoss), `Language.py` (discourse layer) | `0.0` | InfoNCE next-idea contrastive term weight; `0` = MSE-only. |
| `interContrastiveTemp` | same | `0.1` | InfoNCE temperature. |
| `conceptualSimilarityScale` | `Models.py` (ModelLoss wiring) | `0.0` | SBOW training weight for the concept `similarity_codebook`. |

### Per-space

| Knob (space) | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `attention` (PartSpace / ConceptualSpace / WholeSpace) | `Spaces.py` (each space init) | `off` | Symbolic-retrieval attention mode enum: `off` \| `primer` \| `second-order` \| `low-rank`. Replaces the inert `hasAttention` booleans. |
| `attentionPromotion` (ConceptualSpace) | `Spaces.py` (CS init) | `false` | Attention-to-relation promotion gate; the promotion bar is the same `truthCriterion` law as sentence learning. |
| `wordStore` (PartSpace) | `Spaces.py`, `Language.py` | `false` | The percept store's WORD collection becomes the reverse recommenders' candidate basis. |
| `chunkPromotionThreshold` (PartSpace) | `Spaces.py` (PS init) | `4` | Radix chunk promotion: recurrence count bar on the PerceptStore gate. |
| `chunkPromotionMinLength` (PartSpace) | `Spaces.py` (PS init) | `2` | Radix chunk promotion: minimum chunk length. |
| `demuxed` (InputSpace) | `Models.py` (`_make_perceptual_space`) | `false` | Routes what/where/when through the demuxed `ModalSpace` composite instead of the muxed PartSpace. |
| `lrScale` (OutputSpace) | `Models.py` (`getOptimizer`) | `1.0` | Puts OutputSpace params in their own Adam group at `learningRate * lrScale`. |
| `definitionFreeSize` (ConceptualSpace) | `Spaces.py` (CS init) | `2` | Symbols a concept's definition may use for free (genus + differentia); the rank-ordered soft-L0 penalizes only surplus ranks. |
| `lbgThreshold` (WholeSpace) | `Spaces.py` (meta codebook) | `0.5` | LBG auto-granularity: variance threshold for splitting a codebook row. |
| `lbgMinCount` (WholeSpace) | same | `8` | LBG: minimum hit count before a row may split. |
| `lbgEpsilon` (WholeSpace) | same | `0.1` | LBG: split displacement magnitude. |
| `divideWithinWhole` (WholeSpace) | `Spaces.py` (WS init + tiling site) | `true` | Divides unattested type-runs into attested concepts by greedy longest-match tiling; explicit `false` keeps the undivided span. |
| `gradientMode` (WholeSpace) | `Spaces.py` (WS init) | `"ste"` | Codebook gradient coupling: `snap` \| `ste` \| `rotation` (invalid values raise). |
| `useStackRouter` (WholeSpace) | `Spaces.py` (WS init) | `false` | Gates WS's stack-rewrite LanguageLayer dispatch; `false` keeps the legacy forward path. |
| `initScale` (any space) | `Spaces.py` (`_read_init_scale`) | unset | Per-row seed magnitude for the space's codebooks; small values ($\approx 0.02$) keep the sigma/pi folds in their linear regime. Unset/0 = legacy unit prefill. |
| `impenetrableOverlap` (WholeSpace) | `Spaces.py` (WS init) | `0.0` | ImpenetrableLayer separation penalty on codebook rows: overlap scaled by the absolute trust difference (replaces the legacy antisymmetry+transitivity pair). |
| `impenetrableVariance` (WholeSpace) | `Spaces.py` (WS init) | `0.0` | ImpenetrableLayer variance regularizer. |
| `commitmentDecay` (PS/CS/WS) | `Models.py` (VQ override loop) | unset ($\to$ VQ default `0.9`) | VQ-VAE EMA decay override; higher = slower, more stable assignment. |
| `codebookEmaDeadThreshold` (PS/CS/WS) | same | unset ($\to$ VQ default: `1` if `codebookRetire` else `0`) | Dead-code retire: EMA `cluster_size` below this replaces the slot with a fresh row. |
| `codebookGrowthEpsilon` (PS/CS/WS) | same | `0.0` | Stashed as `vq.growth_epsilon`; `runBatch` consults it for codebook growth. |

### `<SymbolSpace>` level

| Knob | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `armaP` | `Language.py` (InterSentenceLayer build) | `5` | ARMA AR order for inter-sentence prediction. |
| `armaQ` | same | `2` | ARMA MA order. |
| `armaHiddenDim` | same | unset (auto) | ARMA predictor hidden width. |
| `signal.temperature` | `Language.py` (signal-router build) | `1.0` | Signal-router softmax temperature. |
| `language.useGrammar` | `Language.py` (grammar load) | — | DEPRECATED: still read, but only to warn loudly and force `default.grammar`. |
| `language.start` | `Language.py` (grammar load) | `"S"` | Accepted completed-derivation shapes (start patterns); named starts tag `relative_truth` / `absolute_truth` sets. |

**Config-only orphan**: `useSubspaceActivation` appears in the schema and
in test fixtures (`test_basicmodel.py`) but is parsed nowhere in `bin/` —
setting it in a config has no effect.

---

## `InvertibleLinearLayer` (programmatic)

The LDU-factorised invertible linear primitive used internally by
`SigmaLayer(invertible=True)` and `PiLayer(invertible=True)`. Not set
via XML directly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `naive` | bool | `false` | `false`: apply L, D, U sequentially via triangular solves (no $W$ materialisation). `true`: materialise dense $W_{\text{eff}}$ and the dense LDU inverse for the reverse pass. Slower; debug-only. |
| `stable` | bool | `false` | Clamps each diagonal entry $d_i$ to magnitude $[\epsilon, 1]$ with sign preserved before the ergodic blend. Keeps $W_{\text{eff}}$ bounded away from singularity. |
| `ergodic` | bool | `false` | Factor-level noise injection. Registers `noise_raw_L`, `noise_raw_U`, `noise_d` buffers; resamples at the start of each `forward()` and end of each `reverse()`. |

**Class renames and removals.**

| Old name | New name / status |
|----------|------------------|
| `LULayer` | Renamed to `InvertibleLinearLayer` |
| `InvertibleLinearLayer` (SVD-based) | Removed; replaced by LDU factorisation |
| `InvertibleSigmaLayer` | Merged into `SigmaLayer(invertible=True)` |
| `InvertiblePiLayer` | Merged into `PiLayer(invertible=True)` |

See [Architecture.md](Architecture.md) for the LDU factorisation and
factor-level noise injection.

---

## Example: XOR Configuration

```xml
<model>
  <architecture>
    <data><dataType>embedding</dataType></data>

    <training>
      <numEpochs>400</numEpochs>
      <learningRate>0.01</learningRate>
      <autoload>false</autoload>
      <trainEmbedding>JOINT</trainEmbedding>
      <embeddingScale>0.05</embeddingScale>
    </training>
  </architecture>

  <InputSpace>
    <nVectors>8</nVectors>
    <nDim>10</nDim>
    <nOutput>8</nOutput>
    <nOutputDim>10</nOutputDim>
  </InputSpace>

  <PartSpace>
    <nInput>8</nInput>
    <nVectors>1000</nVectors>
    <nDim>10</nDim>
    <nOutput>8</nOutput>
    <hasAttention>false</hasAttention>
    <invertible>true</invertible>
    <synthesis>lexicon</synthesis>
  </PartSpace>

  <ConceptualSpace>
    <nInput>8</nInput>
    <nVectors>8</nVectors>
    <nDim>10</nDim>
    <nOutput>8</nOutput>
    <invertible>true</invertible>
    <codebook>quantize</codebook>
  </ConceptualSpace>

  <WholeSpace>
    <nInput>8</nInput>
    <nVectors>8</nVectors>
    <nDim>2</nDim>
    <nOutput>8</nOutput>
    <codebook>quantize</codebook>
  </WholeSpace>

  <OutputSpace>
    <nOutput>1</nOutput>
    <nVectors>1</nVectors>
  </OutputSpace>
</model>
```

Everything not listed inherits from `data/model.xml`.

---

## Example: Embedding / Chatbot Configuration

```xml
<model>
  <architecture>
    <data>
      <dataType>embedding</dataType>
      <shardDir>data/fineweb</shardDir>
      <minFrequency>0.00001</minFrequency>
    </data>
    <weightsPath>BasicModel.ckpt</weightsPath>
    <embeddingPath>BasicModel.kv</embeddingPath>
    <training>
      <numEpochs>2</numEpochs>
      <batchSize>128</batchSize>
      <reconstructionScale>0.1</reconstructionScale>
      <trainEmbedding>BACKPROP</trainEmbedding>
      <autosave>true</autosave>
      <checkpointEveryBatches>500</checkpointEveryBatches>
      <sentencePrediction>true</sentencePrediction>
      <armaScale>0.1</armaScale>
    </training>
  </architecture>

  <!-- space definitions inherit from model.xml -->
</model>
```

Key points:

- `<data><dataType>embedding</dataType>` activates the embedding-based input pipeline alongside the neural model.
- `trainEmbedding=BACKPROP` trains model layers with gradients flowing through the codebook.
- XML config defines the architecture. `weightsPath` stores the neural model checkpoint.
- `minFrequency` gates vocabulary admission; words below it are buffered until they exceed the threshold.
