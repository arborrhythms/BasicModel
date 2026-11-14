# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`).
Default values are loaded from `data/model.xml` and overridden by
model-specific configs. The schema is in `data/model.xsd`.

This document is the human-facing reference for common knobs and migrations.
`data/model.xsd` is authoritative for the complete accepted element set.

**Sizes (Alec, 2026-11-04).** Every constant set by hand that is a size (a
capacity, horizon, budget, cap, cadence or width) is a power of two, for example
`PartSpace.basicPerceptCapacity` $2^{18}$. A width made by adding such constants
stays a sum: an event is its content plus the fixed eight-coordinate band (four
`.where`, four `.when`), and the basic-percept row is $64 + 32 + 256$
(`identityPairDim`, `identityLengthDim`, `propertyDim`). Seeds, versions,
booleans, 0 and 1, rates, weights and ratios are not sizes; nor are counts that
the structure fixes (the eight canonical property rows plus the word row,
MNIST's 784 inputs and 10 classes) or values such as ports and angles. Some free
sizes are not yet powers of two, among them `maxDocs` 10000, `verbGroups` 24,
`placementNegatives` 5 and `identityOnes` 3.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Several XML knobs expose the architecture's relation to LLMs, Formal Concept
Analysis, and DisCoCat. `dataType=embedding`, sentence prediction, and
reconstruction cover the LLM-like prediction/generation path.
`attentionBudget`, `bindingDepth`, `conceptLayers`, `mereologyRaise`, `monotonic`, and the
codebook settings control the FCA-like concept order and part/whole lattice.
`transformChooser` and the grammar-layer settings control the DisCoCat-like
typed composition path.

Overlay merge order:
`data/model.xml`  $\to$  `<config>.xml`  $\to$  runtime overrides.

Note on naming: the schema and the XML configs use `nInput` / `nOutput`
(sequence-length sentinels), `nDim` (per-vector dim), `nVectors`
(codebook size), and `nInputDim` / `nOutputDim` (raw shape hints for the
LiftingLayer). Older docs sometimes call `nOutput` / `nInput` `nActive`;
treat that as the same thing as `nOutput`.

> 2026-05-28: `<nWhere>` / `<nWhen>` are retired from per-file configs.
> The band widths are architectural (`canonical_shape` in `bin/architecture.py`).
> Input, part, whole and symbol events use two quadrature pairs for `.where`
> and two for `.when`; conceptual codes and final outputs have no local band.
> A complete model shares **one** spatial ladder and **one** temporal ladder.
> The spatial long period is derived from the total input/percept/symbol
> registry capacity; the temporal long period covers the LTM allocation.
> Short periods preserve one-location resolution. Integer spatial addresses
> are decoded from the band, not saved beside it. Everything observed in one
> field shares its time; the exact clock side-band stays. Span endpoints are
> separate extent metadata. The old `wherePeriod`,
> `whenPeriod` and `whenRungRatio` XML knobs are retired.

---

## `<architecture>`

Core model settings. Training and data parameters are in nested
sub-elements `<training>` and `<data>` (see below).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data/dataType` | string | `"numeric"` | The data space-role, set under `<data>`: `embedding` (language modelling and chat with sequence processing; PartSpace owns the byte and word lexicon) or `numeric` (a dense slab, e.g. MNIST pixels). |
| `conceptBinding` | string | `"mixing"`; BasicModel `aligned` | PS/WS concept-formation mode. `aligned` preserves location and fuses every non-raw cumulative fold from both towers, without allocating `ConceptualCombine`; exact ordered paths and the actual concept order ride on the concept. `mixing` keeps the learned `ConceptualCombine` matrix as the parallel-path alternative. |
| `processSymbols` | bool | `false` | Apply extra symbolic processing after Sigma. |
| `monotonic` | bool | `false` | Constrain invertible Sigma / Pi to $W \ge 0$ so the lift / lower chain is order-preserving on the parthood cone. |
| `ergodic` | bool | `false` | Ergodic exploration: eligible layers use `W_eff = bias * W + var * noise`; `bias`/`var` are gradient-energy buffers, not Adam parameters. See [Ergodic.md](Ergodic.md). |
| `naive` | bool | `false` | Materialise `W_eff` densely in `InvertibleLinearLayer`. Slower; debugging only. `false` uses sequential L / D / U triangular solves. |
| `attentionBudget` | nonnegative int | `32` | One allowance for input bracket operations, symbol recall and thought attention. Grammatical word composition does not spend it. A positive value also sizes the padded bracket table; zero disables the walk and thought work while retaining ordinary word composition for the VP-only control. |
| `thoughtSearchExhaustion` | positive int | `1` | Consecutive completed region queries with no candidate before conclude is legal despite free variables or missing evidence. At that closing, source-supported forward referents are provisionally minted; regions without source evidence stay question rows. A query cut short by the budget is not a completed empty search. |
| `bindingDepth` | nonnegative int | `1` | Number of constructed percept/concept binding layers, with live recurrence between them; model capacity, not an attention work allowance. Exactly one WholeSpace is shared by these stages; its property inventory and where slice do not multiply with this depth. |
| `conceptLayers` | positive int | `1` | Concept inventory depth including order zero. It does not run repeated symbolic promotion passes. `data/model.xml` declares 1; where no file declares it, two read sites (the concept order limit and the serial word-major path) fall back to 2 and two others to 1. |
| `composeTemperature` | finite float $\ge 0$ | `0` | Sampling temperature of the hard choices in training. Zero selects the greatest logit, with the structural rule on exact ties. Positive values sample `softmax(logits / temperature)` during training. Evaluation always uses argmax. Straight-through credit always uses the untempered model softmax; this parameter never scales its gradients. |
| `reducePressure` | finite float $\ge 0$ | `1.0` | Initial value of the compose chooser's learned load gain. Each binary logit gets `reducePressure * (d / a + max(0, d - a) / r)`, zero on an empty stack: `d` is the STM depth, `a` the allowance (`K - 1` during online rounds; at the closing, one for an absolute sentence and three for a relative one) and `r` the rounds remaining in the phase, the current one included. The divisors are clamped to at least one for numerical safety, while deadline masking uses the actual allowance and count of remaining rounds: when `r <= max(0, d - a)` only binary operations are eligible. Every binary operator gets the same bonus, and it enters the model's softmax and straight-through credit before the sampling temperature. See [Language](Language.md#one-operation-per-round). |
| `conceptualPi` | bool | `false` | Open or update located native conjunction candidates in MTM; their order-0 definitions precede higher-order sigma folds. Pi edges above order 0 are rejected. |
| `attentionPromotion` | bool | `false` | Enable familiar candidates in MTM. The open sentence field observes them; the kept close promotes them. Explicit clause admission is independent of this switch. |
| `conceptMintCount` | positive int | `3` | Recurrences within `mtmHorizon` required to promote a familiar candidate or identity; a candidate whose horizon reading is both refines and waits. A kept defining clause also promotes its pending subject. |
| `conceptUseFloor` | decimal | `0.5` | Field activation must exceed this floor to count as an MTM access. |
| `conceptMatchCos` | decimal | `0.8` | Minimum cosine for a same-order MTM candidate match, using conceptual-width centroids. Located conjunctions retain the complete native literal identity. |
| `conceptPartFloor` | decimal | `0.001` | Drop support below this magnitude when a candidate is promoted; retained definition edges merge by maximum. |
| `routerWireSerial` | string | `"both"` | Per-word router-fire gating on the serial path: `per-word` (fire per word, boundary off), `boundary` (fire only at the sentence boundary), `both` (default — both fire), `off` (neither). The per-word fire populates `symbolSpace.current_rules` for SS dispatch. See [STM.md Section 7](STM.md#7-per-word-router-firing). |
| `transformChooser` | string | `"anchordot"` | Placement scorer for the structured grammar layers. `anchordot` = stateless cosine-to-anchor (byte-identical default, no new params); `mlp` = learned `MLPTransformChooser` (owns tool-embedding + MLP params $\to$ deliberate fresh-basin cutover). See [Language.md](Language.md). |
| `transformChooserHidden` | int | `0` | Hidden width of each grammar MLP when `transformChooser=mlp`; `0` derives `max(8, grammar feature width)`. Positive values override it. Canonical BasicModel currently derives 1024. Inert for `anchordot`. |
| `transformChooserDepth` | int | `1` | Number of hidden Linear/GELU blocks in each grammar MLP, followed by one scalar Linear head; must be positive. Not the number of grammatical reductions or thought iterations. |
| `whatThinkingHidden`, `whatThinkingDepth` | int | `16`, `1` | Unread in effect: both are parsed and validated, but the selected-thought chooser they sized is no longer built (thought is scored by `OperationSelectionLayer.thought_logits`). |
| `intuitionCapacity` | positive int | `4096` | Slots in each preallocated sparse Pi and Sigma bank owned by output. Pi addresses are rule/literal references and Sigma addresses are rules. Unit initial exponents give the exact max-min DNF. Exhaustion raises; storage never grows. |
| `separateChooser` | bool | `false` | False retains live compose features and projects the answer gradient against reconstruction per optimizer unit before one shared step. True is the optional per-owner backward: each cost trains its declared owner, independently of its R/A side. Thought and generate features remain detached. |
| `reconstructionJitter` | real in `[0,1]` | `0` | Draw one strength `s ~ U[1-j,1]` per sentence and apply it to reconstruction alone, shared by all of the sentence's paths. Zero draws no random number. Output, expectation, attention and chooser credit retain scale 1.0. |
| `explorePaths` | nonnegative int | `1` | Number of distinct one-departure complete paths beside the greedy path in training. Departures draw over attention, compose, candidate, bind, reconstruction, thought and output; each stream keeps its strictly cheapest path on R + A, ties to the greedy path, and a departure reuses the prefix before it. Evaluation runs the greedy path alone. See [Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11). |
| `omissionTarget` | float in `[0,1]` | `0.9` | Desired long-term reconstruction accuracy over visible and omitted input. `1` disables omission and its random draws; XOR proofs declare this override. |
| `omissionLimit` | float in `[0,1]` | `0.5` | Maximum independent word and within-kept-word percept-slot omission rates. One word stays whole. Omission never runs in evaluation. |
| `omissionHorizon` | positive int | `1024` | Sentence horizon for the visible/omitted accuracy averages at both omission sizes. Both rates and all four averages travel in checkpoints. |
| `generateBudget` | positive int | `32` | Architecture capacity of each free output and sentence-decoder generation walk. Fixed at model construction; its output, action, residual and feature arrays are allocated to this trip count before the walk. Exhaustion reports pending work and retains its per-lane frontier residual. Recorded input derivations keep their full structural trace capacity independently. |
| `wordSurfaceCapacity` | positive int | `64` | Maximum raw bytes in one word, fixed at construction independently of sentence input length and word count. Targets have this width; candidate byte tables reserve one additional EOS position. Bytes remain `uint8`. A longer surface raises before staging; no clipping or resizing. Native walk banks carry terminal presence and primitive support without copying the raw byte slab. |
| `ltmWindow` | positive int | `1024` | Maximum LTM0 row window for the per-order episodic product. Relevance intersects the question’s nonempty orders. Only the best `max_records` and explicitly referenced rows are materialized; the subsequent role match does not scan the window or relation table. Fixed role-match scratch is allocated at construction. |
| `reconstructFromIdea` | bool | `false` | Legacy evaluation selection of recovered word ideas instead of the single-slot seed. Tied mode always returns its owned completed reconstruction, independently of this flag, and never enters the free generate chart ([Models.py](../bin/Models.py)). |
| `adverbEigEdit` | bool | `false` | Legacy/direct `LiftLayer` helper flag for the adverb sparse eigenvalue edit. The live `adverb` grammar operator force-builds the same zero-init projection and calls `LiftLayer.apply_adverb`, so ordinary grammar use does not depend on this flag. When enabled for plain `LiftLayer`, an adverb modifies a composed VP by `a2 = atanh(vp) + p_vp * delta_adv`, masked by the VP's own eigen-signature. Default off keeps plain `LiftLayer` byte-identical. |
| `mereologyRaise` | bool | `false` | Enable native meronomy binding and the conceptual refine/raise decision. Beyond order-1 symbolization, a raise requires a both reading with discontiguous retained support after stalled learning; counts never trigger it. |
| `mereologyRefinePatience` | positive int | `3` | Completed optimizer updates without a strict improvement in the worst local `min(c⁺, c⁻)` reading before discontiguous support may raise an order. Pure/unknown evidence and improvement reset patience; repeated reads do not advance it. |
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
| `loadBalanceWeight` | float | `0.0` | Unread in effect: the value is parsed and nothing consumes it. |
| `l1Lambda` | decimal | `0.0` | Unread; declared in `data/model.xml`. Configurations set `WholeSpace.l1Lambda`, which is unread too. |
| `discontinuityLambda` | decimal | `0.0` | Unread; declared in `data/model.xml`. |
| `maxActivePerLayer` | int | `8` | Maximum active nodes per layer for reverse / mereological walks. |
| `codebookRetire` | bool | `false` | Retire codebook path where supported. Mostly a migration hook. |
| `symbolLearning.enabled` | bool | `false` | Enables runtime symbol-learning hooks. Disabled by default. |
| `mereologicalTreeSize` | int | -- | Unread; the schema accepts it and one configuration still declares it. |
| `writeSyntax` | bool | `false` | Dump POS-labelled syntax tree at end of every `MentalModel.forward()`. |
| `syntaxOutPath` | string | `output/syntax.xml` | Output path for `writeSyntax`. |
| `trust` | unitInterval | `1.0` | Model-level multiplier for incoming provenance: the source trust of a configured truth is its row trust times this value. Runtime `store_truths` and static `<truthSet>` inputs supply the outer assertion through the shared clause closing. Positive and negative evidence remain separate; embedded registration grants no assertion authority, and sentence content never sets its own trust. |

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
| `conceptLessons` | string | (empty) | Optional JSON primitive memberships and single-occurrence name lessons, resolved relative to `data/`. Runs before ordinary supervised training; unlabelled training inputs witness located conjunctions and output sigma weights start at zero. Used by `XOR_exact.xml`. |
| `dataset` | string | `"xor"` | Dataset key read by `Data.load` in `bin/data.py`: `xor`, `phrases`, `grammar`, `substitution`, `queries`, `sequences`, `mnist`, `text`, `curriculum`, `inline`, `math`, `math_chain`. Curriculum reads verified shards in stage order with whole-document stream partitions. Omit for inference-only models. |
| `numShards` | int | `1` | Shard count (streaming text datasets). |
| `maxDocs` | int | `10000` | Per-shard document cap for text. For curriculum, selects a whole-document prefix across the manifest, before sentence admission filters and across all three splits. |
| `maxSentenceWords` | int | *(unbounded)* | Whole-sentence admission ceiling for text shards. Longer splitter records are excluded intact, never clipped. Set this no higher than `serialWordCapacity` when packed training is enabled. |
| `maxSentenceUnits` | int | *(unbounded)* | Curriculum's conservative whole-sentence limit: pinned maximal letter words plus every remaining byte, including the packing separator. This remains safe when learned boundary cuts change. Exclusions and source-address gaps are recorded. |
| `corpusManifest` | string | *(required for curriculum)* | Version-1 corpus manifest, relative to the project unless absolute. Pins shard hashes and row counts; every row supplies its split, stage, document and sentence address. Curriculum requires complete-sentence packing and rejects `max_tokens` clipping. |
| `grammarLessons` | string | `grammar_wording.json` | Annotated text curriculum for `dataset=grammar`, resolved relative to `data/` unless absolute. Train/validation/test text enters the ordinary loader; only train annotations supply post-forward grammar loss. Use unpacked sentences. See [SelectedMeaning](SelectedMeaning.md). |
| `shardDir` | string | (empty) | Text shard directory (e.g. `data/fineweb`). Only used when `dataset=text`. |
| `minFrequency` | float | `0.0` | Vocabulary admission threshold (fraction of corpus). Words below it are held in a pending buffer until they accumulate enough occurrences. `0.0` admits everything. |
| `contextWhole` | string | (none) | Stable name of the corpus, book or conversation whole that scopes the Teacher's objective addresses. It is not written into the model's `.where` and `.when` bands. |
| `classificationMin`, `classificationMax` | float | -- | Unread; declared in `data/model.xml`. |

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
| `packSentences` | bool | `false` | For self-supervised word-loop training, pack consecutive complete sentences independently in each contiguous batch row up to `serialWordCapacity`. Row-local sentence boundaries are explicit and reset softly; sentences are never split, clipped, transposed, or reordered. Under the canonical `<analysis>meronomy</analysis>` the brick budget and the row layout count the ladder's units (whitespace and digit units included), not whitespace words. Environment override: `BASIC_PACK_SENTENCES`. |
| `numWorkers` | int | `0` | DataLoader prefetch workers. `0` = synchronous in-process batch assembly. |
| `learningRate` | float | `0.001` | Adam learning rate. |
| `reconstructionScale` | float | `0.5` | Weight of reconstruction loss vs prediction loss in $[0, 1]$: $\mathcal{L}_{\text{total}} = (1-r)\,\mathcal{L}_{\text{output}} + r\,\mathcal{L}_{\text{recon}}$.  |
| `conceptualContextLearningRate` | float | `0.0` | Unread except to reject a nonzero value at construction; configurations still declare `0`. |
| `conceptualContextSituationWeight`, `conceptualContextSituationAnchors`, `conceptualContextExpectationWeight`, `conceptualContextNegatives` | -- | -- | Unread; still declared in `data/model.xml` or in configurations. |
| `reconstructInLoop` | bool | -- | Unread; four configurations still declare it. Word-major sentence training always runs the one tied reconstruction traversal. |
| `reconstructionBasisLimit` | positive int | `16` | Maximum candidate prototypes per side for approximate reconstruction through the selected compose operator: at most `K*K` pairs. Independent of word, STM and field capacity. The invocation owns detached values and masks; missing candidates or inverses report incompleteness ([Language.py](../bin/Language.py)). It also bounds the lexical shortlist obtained once per generate walk from the admitted training forms; structural candidates remain separate. |
| `reconstructionPlacement` | `graph` / `compiled` / `eager` | `graph`; BasicModel `compiled` | Run the traversal inside the forward graph, as a separate fullgraph call, or eagerly. The separate call promotes the `eager` backend to `aot_eager` to cache backward too. The `auto` policy resolves to its first configured backend before PyTorch lookup. `BASICMODEL_RECON_PLACEMENT` remains a diagnostic override. Placement changes execution, not the tied objective or 21-value forward state ([Models.py](../bin/Models.py)). |
| `outputInLoop` | bool | `false`; BasicModel `true` | `reverseOutput` unfolds owned answer concepts with its bounded generate walk. The chooser uses the declared `<generate>` rules and stop: training samples actions; evaluation selects the highest-scoring action. It ignores input compose traces, teacher targets and reconstruction witnesses. Unavailable requested operations remain pending and report bounded truncation ([Models.py](../bin/Models.py)). |
| `outputPolicyWeight` | float | -- | Unread; three configurations still declare it. |
| `grammarLessonWeight` | float | `1.0` | Weight of supplied grammar annotations: compose-choice cross entropy plus separately annotated generate-choice/child-value loss. Added to the actual `runBatch` total after student output; no term without annotations, on evaluation splits, or on prediction-only trials. `0` disables it. Shared parameters obey reconstruction priority. |
| `whatScale` | float | `0.7` | Loss weight on the `.what` (content) channel. |
| `whereScale` | float | `0.2` | Loss weight on the `.where` (positional) channel. |
| `whenScale` | float | `0.1` | Loss weight on the `.when` (temporal) channel in general event comparisons. Per-word reconstruction excludes the field's shared timestamp. |
| `negSamples` | int | `64` | Negative samples per positive for CBOW / SBOW. Memory cost: $O(\text{batch} \cdot \text{negSamples} \cdot \text{dim})$ vs $O(\text{batch} \cdot \text{vocab})$ for full softmax. |
| `TruthLoss` | float | `0.0` | Additive penalty for opposite-evidence overlap with the TruthSet: affirmed coordinates overlap denials, and denied coordinates overlap affirmations. `0.0` disables. |
| `maskRate` | float | `0.15` | Within-sentence IR Bernoulli mask rate. BERT default 0.15. |
| `trainEmbedding` | string | `"NONE"` | Embedding update mode: `NONE`, `CBOW`, `SBOW`, `BACKPROP`, `BOTH`, `JOINT`. See table below. |
| `embeddingScale` | float | `0.25` | JOINT-mode embedding loss weight $\lambda$: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{emb}}$. Ignored by other modes. |
| `autoload` | bool | `true` | Load weights from `weightsPath` at construction. Stale incompatible checkpoints fail fast. |
| `autosave` | bool | `false` | Save weights after training completes. |
| `checkpointEveryBatches` | int | `0` | Save mid-training checkpoint every N optimizer steps. `0` disables periodic checkpoints. |
| `intervalLogEveryBatches` | int | `0` | Log captured losses, retention and inventory every N training batches; run the fixed probe panels in a separate evaluation copy. `0` disables. Each JSON record includes collection time and probe failures. |
| `intervalLogFile` | string | `output/intervals.jsonl` | Append-only interval JSON records. The reconstruction sample includes its native reconstructed event and byte cost. |
| `profile` | bool | `false` | cProfile the training loop. |
| `certainty` | bool | `false` | Per-neuron certainty tracking in ergodic layers. Allows individual neurons to transition exploration $\to$ exploitation at different rates. |
| `wordExpectation` | bool | `true` | Order-zero reads of the shared two-lane native-symbol activation predictor. |
| `wordExpectationLossWeight` | nonnegative float | `0.1` | Local activation MSE weight at accepted word brackets. |
| `sentenceExpectation` | bool | `true` | Enables symbol/closing/document expectation and residuals. Explicit `false` bypasses this cycle while understanding and the single What memory remain available. |
| `expectationPrime` | bool | `true` | Enables the selection prime in compose and candidate attention independently of attenuation. Off only in the declared parity configuration (`test/fixtures/item4-parity.json`); prediction, residuals and retention stay enabled. |
| `perceptExpectationGain` | float in `[0,1]` | `0` | Scales the detached signed percept residual that candidate attention reads as salience. The forward content is the exact attended observation; this knob does not attenuate it. |
| `symbolExpectationGain`, `rowExpectationGain`, `documentExpectationGain` | float in `[0,1]` | `0` each | Attenuation strength $g$ of expectation at the symbol, row and document levels: what was predicted is dimmed toward the floor $1 - g$, and 0 leaves content whole. The row and document gains compose at the closing. See [Expectation](ExpectationRetention.md). |
| `perceptExpectationRate` | float in `(0,1]` | `0.05` | Moment EMA rate after the first external observation; padding, evaluation and speculative trials do not update it. It sets no optimizer rate: the percept expectation's parameters move in the sentence's one optimizer step. |
| `expectationOmissionThreshold` | float in `[0,1]` | `0.5` | Minimum for-lane omission confidence for the conceived absence thought. |
| `expectationEpsilon` | positive finite float | `1e-6` | Denominator floor for normalized percept and symbol residuals. |
| `expectationSaturation` | enum | `clip` | `clip` uses `min(1, ratio)`; `smooth` uses `magnitude/(denominator+magnitude)`. The entire gain is detached. The signed percept residual is never saturated. |
| `explorationUniformFraction` | float in `(0,1]` | `0.5` | Uniform share $\varphi$ of a departure's alternative draw, so every eligible non-greedy action keeps positive probability. |
| `explorationTemperature` | positive finite float | `1` | Temperature of the policy component of the alternative proposal. Credit uses the actual mixture probability. |
| `sentenceExpectationLossWeight` | float | `0.1` | Weight on shared symbol/closing/document activation MSE. All source context and targets are detached. E trains the predictor independently of the paired R+A judge; zero disables this objective. See [ExpectationRetention](ExpectationRetention.md). |
| `independenceWeight` | nonnegative float | `0.1` | Weight of the LTM-population sparse-coding objective, owned by expectation. Applies to grammars declaring individual references. |
| `independencePriorScale` | positive float | `0.05` | Laplace activation prior and automatic-relevance pressure on unit direction columns; also scales column incoherence. |
| `independenceMintThreshold` | positive float | `0.2` | Unexplained event-coordinate residual norm required for column admission after four distinct witnesses. Admission remains a kept-closing effect. |
| `actionRateRatio` | float in `(0,1]` | `1` | Ratio $\kappa$ of the action learning rates inside reconstruction's descent: the lift action uses $\kappa$ times the configured rate and the adverb action $\kappa^2$ times it; codes keep the configured rate. See [GradientFlow](GradientFlow.md). |

Which objective trains which parameters is in [GradientFlow](GradientFlow.md).

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
| `timeout` | int | `120` | Unread; declared in `data/model.xml`. `serve.py` reads only `host` and `port`. |

---

### LTM forgetting (`<architecture>`)

The forgetting pass runs after backward at
closed document boundaries in training, or after stateful inference. It never
runs inside an open sentence or stateless request. `SymbolSpace.ltmCapacity`
remains the hard limit; a document that fills the store before its boundary
still fails atomic admission. The following settings are validated at model
construction. The low-water fraction must be below high water; all weights
are finite and nonnegative. A pass deletes unprotected rows lowest value first
until occupancy falls to `forgetLowWater`. A row's value is
$V = w_t T + w_u U + w_l L - w_a A$, each term in $[0, 1]$ so the weights below
are comparable, with $T$, $U$, $L$ and $A$ the trust, surprise, luminosity and
age terms their rows describe; ties go to the older row and then the stable
address, so the same state always deletes the same rows. Protected origins,
live thought and expectation roots, rows that an `LTM1` row references and
everything they reach are kept, and under pressure a row loses its source text
before it goes. See [STM](STM.md#10-ltm-as-the-chain-of-stm-end-states),
"Forgetting".

| Parameter | Type | Default | Description |
|---|---|---|---|
| `forgetting` | bool | `true` | Enable LTM retention and boundary concept reclamation. Capacity warnings remain visible. |
| `forgetHighWater` | float | `0.9` | Occupancy fraction that triggers a document-boundary pass. |
| `forgetLowWater` | float | `0.7` | Target occupancy fraction; protected roots and gradual detail loss can prevent reaching it. |
| `forgetMinInterval` | int | `256` | Minimum completed sentence presentations between passes. |
| `forgetAtEpoch` | bool | `false` | Also consider retention at a completed training epoch, subject to the interval. |
| `forgetTrustWeight` | float | `1.0` | Weight of the row's trust, the larger of its two evidence lanes, $\max(t^+, t^-)$. |
| `forgetUtilityWeight` | float | `1.0` | Weight of the row's stored surprise in $[0, 1]$; a row whose surprise is unknown counts as 1. |
| `forgetLuminosityWeight` | float | `1.0` | Weight of the luminosity term. For an occurrence row it is 0 when a live row of the same order and scope, within `forgetSimilarity`, carries the contrary pole with more trust, and 1 otherwise (with `forgetCoverage`, the row's leave-one-out coverage instead); for a part or taxonomy row it is $\log(1 + \text{accesses})$. |
| `forgetAgeWeight` | float | `1.0` | Weight of the bounded age penalty $1 - e^{-a/h}$, for occurrence rows and for part and taxonomy rows alike. |
| `forgetHalfLife` | float | `4096` | Positive scale $h$ of the age penalty. The age $a$ is the model time since the row's `.when` stamp; a row without a stamp of its own takes the youngest age of a stamped row or part that refers to it. The name is not a literal half-life. |
| `forgetSimilarity` | float | `0.9` | Cosine threshold in `[0,1]` for opposing idea rows in the same order/scope. |
| `forgetCoverage` | bool | `false` | Use the more expensive leave-one-out catuṣkoṭi coverage contribution. |
| `forgetProtectOrigins` | string | `provisioned user` | Space-separated origins that cannot be changed; accepts `conversation`, `provisioned`, `user`. Empty protects no origin. Live references remain protected. |

## BasicModel / MentalModel Quick Reference

`data/BasicModel.xml` and `data/MentalModel.xml` train the full
P$\to$C$\to$S pipeline. Both load `complete.grammar`, in which `lift` is a
binary rule (`lift_argout1 = lift.forward(lift_argin1, lift_argin2)`):
transitive SVO is assembled compositionally, not by a ternary lift. Grammar
mode is derived from the loaded grammar.

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

> **`<lexer>` moved to `<WholeSpace>`** (2026-06-09): lexing is analytic
> CUTTING, owned by the analysis side. An InputSpace-side `<lexer>` now
> fails schema validation and the runtime reader. InputSpace emits the
> dual view — `forward(x) -> (percepts_in, concepts_in)`: the atom view
> (content `[B, N, 1]`) for the perceptual branch and the unity view
> (`[B, 1, N]`) for the symbolic branch.

`inputShape` uses the data's native dim (e.g. 784 for MNIST,
`inputLength` for text); `outputShape` uses `nDim` from
`TheObjectEncoding`. `LiftingLayer` bridges the two.

### `<PartSpace>`

Synthesizes recurrent ordered groups from existing parts. A group's
distributed code is the max of its constituent codes; the ordered
containment read retains the distinction between anagrams. Native
PartSpace has no `SigmaLayer` or `PiLayer`.

`nVectors` is a fixed reserve of part rows. Rows store admitted parts, so the
sentence count alone does not determine the number needed. If the reserve
fills, training stops with a capacity error: the table cannot grow while an
optimizer owns it. BasicModel reserves 32,768 rows, a small-run setting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Number of active perceptual feature vectors. Should be $\geq$ `InputSpace.nOutput`. For XOR: 4. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$ (default 9 = 1 + 4 + 4). |
| `nVectors` | int | sentinel | Fixed physical percept-codebook capacity. Logical admission writes unused rows without changing tensor shape or optimizer ownership. Exhaustion raises an error naming this knob. Serial attention may select `nOutput` rows; parallel attention reads all active percepts. |
| `propertyDim` | positive int | `256` | Property coordinates appended to the identity pair and length coordinates at text-model construction. With `propertyOnes=1` and width 256, each byte has its own coordinate: type codes are their weighted byte extension without hash collisions. Numeric layouts are unchanged. |
| `propertyOnes` | positive int | `1` | Bits per byte in the property subspace; must not exceed `propertyDim`. The 256/1 default uses byte coordinates directly; other configured layouts use `WordIdentity.bit_stream`. |
| `invertible` | bool | `false` | Space construction flag; native reconstruction uses retained ordered perceptual views and canonical part ids, without an inverse fold. |
| `hasAttention` | bool | `false` | Inert: the boolean constructs no attention pass and is kept as an alias. Use `<attention>` (`off`, `primer`, `second-order`, `low-rank`). |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `synthesis` | string | `"meronomy"` | Bottom-up synthesis strategy. `meronomy` is the only accepted value, and any other value raises. Top-down cuts live on WholeSpace `<analysis>`. |
| `butterfly` | bool | -- | Unread; configurations still declare it. The fold span is the architecture-level `sigmaPi`. |
| `wordLearning` | int | `2` | Active lexicon-growth mode. `0` = frozen codebook; `>=1` = on first sight of a new word, insert into the lexicon and tag it on the meronomy. Several trained configs override this to `0` or `1`. |
| `basicPerceptCapacity` | positive int | `262144` | Fixed number of basic-percept rows, allocated at construction; a full table raises. Lesson configurations may declare a smaller capacity. |
| `identityByParts` | bool | `true` | Text forms are an identity pair plus length bits; see the width rule below. |
| `identityPairDim` | positive int | `64` | Width $D$ of a form's identity-pair code. |
| `identityLengthDim` | int | `32` | One-hot over word lengths of 1 to 32 bytes; must be 32. |
| `identityOnes` | positive int | `3` | Set bits $s$ in each identity-pair code, $0 < s \le D$. |

In an embedding configuration with `identityByParts` (the default),
`WordIdentity.resolve_identity_layout` rewrites the declared widths before any
space is built. Every input, percept and whole event is `identityPairDim` + 32
length bits + `propertyDim` + the eight band coordinates, $64 + 32 + 256 + 8 =
360$ by default, and the conceptual width is that plus the declared difference
between the ConceptualSpace and PartSpace `nDim`. The declared `nDim` values of
such a configuration therefore matter only through that difference; numeric
configurations, and those with `identityByParts` false, keep their declared
widths.

PartSpace has no `<codebook>` element: the percept prototypes live on the
`.what` basis. See [Architecture.md](Architecture.md) and
[Philosophy.md](Philosophy.md) for the analysis/synthesis orientation.

### `<ConceptualSpace>`

Order-0 concepts read their signed weights over native feature memberships.
PartSpace supplies located parts; WholeSpace supplies pervading properties.
The concept unions alternative percept conjunctions within an occurrence;
readout unions occurrences inside the subject's extent. `conceptUseFloor` is
the separate discovery gate over measured use.

Holds the conceptual dictionary, location-aligned PS/WS binding state, and
short-term memory. Located pi is performed in the order-0 field, and sigma
unions symbols at higher orders. Grammatical lift/lower own their chart maps.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Active concept vectors. For XOR: 3. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$ (default 9 = 1 + 4 + 4). |
| `nVectors` | int | sentinel | Physical conceptual-codebook capacity. ConceptualSpace owns the concept inventory and taxonomy; word–object DEF relations live in the common LTM store. Aligned property-basis stages share one table; order is metadata, not a row partition. This value is independent of `WholeSpace.nVectors`, PartSpace's physical capacity, live `nOutput`, SymbolSpace references, and STM capacity. |
| `invertible` | bool | `false` | Space construction flag. Aligned native binding retains both perceptual views for reconstruction; ConceptualSpace owns no unary chart fold. |
| `hasAttention` | bool | `false` | Inert: the boolean constructs no attention pass and is kept as an alias. Use `<attention>`. |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `butterfly` | bool | -- | Unread; configurations still declare it. |
| `stmCapacity` | int | `8` | STM ring depth (within Miller's $7 \pm 2$ band). Per-batch buffer `[B, stmCapacity, nDim]` + depth pointers `[B]`. Thought content starts at four rows per focus chunk and may grow to its exact size, bounded by four times the declared sentence leaves plus operator seats; this changes no focus depth. |
| `mtmCapacity` | positive int | `64` | Shared capacity of `ConceptualSpace.mtm` for identities and familiar candidates. Identities retain working values, mention readings and event histories; candidates retain bounded sparse support, a centroid and two-lane readings. Oldest access is evicted at capacity without changing LTM rows. Candidate content reads through the live member codes. |
| `mtmHorizon` | positive int | `16` | Maximum intervening kept closes in a stream since the last reference. Identities and candidates share this access clock and horizon. The end of a document leaves MTM unchanged (`end_document` ends only the document's cover), and the epoch boundary returns it to its state at the epoch's start (`PresentationMemory.epoch`). Recurrence for `conceptMintCount` counts retained observations within the horizon, independently of access count. Pending definition operands hold their seats. Among the others, eviction takes an entry untouched this close first, then the least recent access, then the earliest entry time. If all seats are held, admission raises "MTM capacity ... exhausted by pending definition operands". |
| `mtmPrepend` | positive int | `64` | Leading symbol entries reserved at construction for MTM identities and candidates together. |
| `verbGroups` | positive int | `24` | Maximum learned intransitive bases on paired meaning configurations. A third distinct failed-fit occurrence joins the best fitting group or opens one; capacity is reported. |
| `verbBasisReflections` | positive even int | `4` | Householder reflections per object/group basis. Identical initial pairs give the identity. |
| `verbGroupFit` | nonnegative float | `0.1` | Maximum mean squared chart residual for admitting a verb to an existing basis. |
| `latticeMargin` | decimal | `0` | Unread; declared in `data/model.xml`. |
| `meaningWidth` | nonnegative int | computed | Width of the meaning band of a concept's code; unset, it is the code width less the percept event width. `data/model.xml` ships 128. At 0 a paired chart supplies only form: membership records remain possible, but there are no class directions or conceptual views, and hierarchical conditioning needs a meaning band. |
| `placementRate` | nonnegative float | `0.1` | Step size of the one meet-ratio subgradient step that places a concept's meaning at each kept close. |
| `placementNegatives` | nonnegative int | `5` | Placed rows with zero seen excess drawn as negatives by a generator seeded from the sentence's key; fewer eligible rows give fewer negatives, never repeats. |
| `placementNegativeWeight` | nonnegative float | `1.0` | Weight of the negative term, the brake on collapse. |
| `recurrenceAngle` | float | `15.0` | Degrees: an idea, its time coordinates removed, joins a reading seat whose representative lies strictly within this angle. Must lie in $(0, 180]$. |
| `recurrenceCapacity` | positive int | `max(1, nVectors)` | Fixed number of reading seats; exhaustion raises. |
| `raiseEvidence` | unitInterval | `0.0` | A retained row qualifies as a witness when its positive evidence exceeds this value. |
| `raiseCount` | int $\ge 2$ | `3` | Number of pairwise distinct witnesses of the same order required to create the shared side's class. |
| `readerBlockNormalization` | bool | `true` | Scale the form block and the meaning block of a value read from the sentence's primed symbol bank, each by that block's largest norm over the frozen bank, so the read stays affine. |
| `conceptReadoutL1` | nonnegative float | `0.0` | L1 penalty on the incoming connections of the concepts-from-percepts Sigma; 0 disables. |

`ConceptualSpace` owns definitions, evidence and STM/event state. On aligned
BasicModel, native PS and WS views bind by location and the actual concept
order is explicit metadata; symbolic generalization remains a separate phase.

#### Trust and gates

| Parameter | Reader | Default | Meaning |
|---|---|---|---|
| `trustViews` | `ConceptTrust`, `PartsOfSpeech.operation` | `3` | Finite positive view scale in $\ell = s(1 - e^{-n/\mathrm{trustViews}})$: $n$ counts kept views and $s$ is the accumulated request coverage divided by that count. Views allocate no memory rows. Empty stored content agrees with a view by one, and its first nonempty view becomes the reference. The radius applies only to the for lane. An operand's trust comes from a seated, tracked concept row; an untracked operand contributes one, and the whole norm of an untracked chart remains its extent. The same rule gives a formal argument its class-view radius, and an operation's trust is the minimum input separability times $1 - e^{-\mathit{views}/\mathrm{trustViews}}$. Identity mechanics and fixed relations contribute one. |
| `definitionTrust` | `DefinitionIndex.add` | `0.5` | Finite threshold in $[0,1]$ for a named row's effective positive pole to admit its word-to-object binding; equality is admitted. The threshold gates the binding, never the word's identity: below it a named word resolves by its form. Naming writes the source evidence even below the threshold, so learning can change admission without another write. |
| `gates.closedTrust` | `TrustGates.cap`, `LTM1.refresh_effective` | `0.1` | Finite value in $[0,1]$, contributed to the authority minimum by every covering closed gate; stored evidence is unchanged. |
| `gates.threshold` | `TrustGates.advance` | `0.75` | Finite value in $[0,1]$: the minimum definiteness of all observed kept ancestors before an operator opens. No observed ancestor leaves a nonexempt operator closed. |

`<gates>` is a child of `<ConceptualSpace>` beside `<trustViews>` and
`<definitionTrust>`, with child elements rather than attributes so that
configuration serialization preserves its shape:

```xml
<gates>
  <closedTrust>0.1</closedTrust>
  <threshold>0.75</threshold>
</gates>
```

Every declared operator has its own monotone gate. Fixed operators and
declared clause roots start open; any other operator opens only after the
arguments of its observed kept ancestors reach `threshold`. The defaults and
the lesson configuration use the same values. The retired `interpret` and
`clause` groups fail at construction, and so does their checkpoint state.
Runtime validation rejects nonfinite or out-of-range values. Gate settings are
construction data; openings, dependencies and opening times are checkpointed
model state. No operation-count scale is added. The
learned-argument inventory reserves one more native concept seat per declared
formal argument at construction, beside the configured inventory capacity.

### `<WholeSpace>`

Each property owns 256 learned primitive memberships alongside its
distributed code.
A-priori byte examples teach the initial properties; runtime analysis reads
those parameters. Constant-signature runs keep their input brackets.
Word staging retains byte counts, and the compiled read evaluates the live
memberships. Native conceptual definitions read PartSpace containment and WholeSpace
pervasion from those memberships. The distributed codes serve similarity and
tied reconstruction, never conceptual presence. See [the membership design](Architecture.md#one-attention).

The whole-percept/property side of perception and the home of top-down
analysis. WholeSpace owns primitive property memberships, the
`<analysis>` division knob, and the deprecated intake spellings. It is
upstream of ConceptualSpace and SymbolSpace and therefore owns no concept,
META, taxonomy, or symbol-prototype inventory. See
[Philosophy.md](Philosophy.md) for the design.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Number of simultaneously live whole-percept locations. This is a field width, not a property-inventory or symbol count. |
| `nDim` | int | `0` | Width of one whole-percept/property vector. `0` inherits the preceding compatible event width. |
| `nVectors` | int | sentinel | Fixed physical whole-percept/property capacity. BasicModel starts with eight ASCII-class properties; further admission uses unused rows inside the declared nVectors and is independent of both `ConceptualSpace.nVectors` and live `nOutput`. |
| `codebook` | mode | `quantize` | `none`, `quantize`, or `project` for the property-percept basis. It never selects a concept or symbol codebook. |
| `analysis` | string | `"meronomy"` | Top-down division over the unity view. `meronomy` is the only value a WholeSpace accepts: the older cuts (`byte`, `raw`, `sentence`, `word`, `grammatical`) are retired, and any other value raises. The old `analyse` spelling is not accepted by the schema. |
| `analyzer`, `lexer` | string | -- | Deprecated spellings of the intake cut: the reader prefers `<analyzer>` and falls back to `<lexer>`, and `<analysis>` supersedes both. An InputSpace-side `<lexer>` is rejected. |
| `nonlinear` | bool | `true` | Tanh-bound the activation. |
| `sortNetwork`, `sortPasses`, `l1Lambda`, `symbolResidualScale`, `outputSymbolResidualScale`, `commitmentBeta`, `decorrelationWeight`, `spectralFlatnessWeight` | -- | -- | Unread; declared in `data/model.xml`. |

WholeSpace owns primitive property memberships and the native analysis
policy. It allows primitives by max and computes pervasion by min over
observed positions. It has no `PiLayer` or inverse fold. Conceptual pi
is restricted to the order-0 attentive field; grammatical operators own
their parameters independently of either perceptual tower.

### `<OutputSpace>`

Maps symbols to final predictions, or reads named conceptual positive poles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Output values. For XOR: 1. |
| `nDim` | int | `1` | Per-output dim. |
| `nVectors` | int | sentinel | Codebook size. Defaults to `nOutput`. |
| `invertible` | bool | `false` | Rarely set on OutputSpace. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `nonlinear` | bool | `false` | OutputSpace is linear by default; rescales via `Data.denormalize`. |

With `nVectors == 1` (or no codebook) the head is an unquantised linear map
`nInputDim -> nOutputDim` plus a learned scalar intercept; named concept
outputs read native evidence instead. `LinearLayer` has `(bias, temp)` support
for ergodic mode.

---

## `<SymbolSpace>`

Grammar infrastructure (SyntacticLayers, TruthLayer). Lives outside
`<architecture>` as a top-level sibling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `syntacticHiddenDim` | int | `64` | SyntacticLayer MLP hidden size. |
| `truthMaxEntries` | int | `1024` | TruthLayer SS-codebook capacity. |
| `ltmCapacity` | int | `1024` | Capacity of the shared LTM occurrence store and bounded inter-sentence history. The `.when` ladder uses `RowTime.HORIZON`, independently of this reservation. An explicit value takes precedence over `truthMaxEntries` for LTM; `truthMaxEntries` sizes the truth activation view. The store raises when full. MM_math_chain reserves 131,072 rows for its seventeen graded presentations; exhaustion still raises. See [STM.md Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states). |
| `episodicConcepts` | nonnegative int | `4` | Maximum signed lane readings kept at each declared conceptual order from the sentence’s open field, beside its LTM0 idea. The strongest readings are stored in reading and concept-ID order as detached row-by-order-by-slot `episodic_concepts` and `episodic_values`; contradictory lanes occupy separate slots. `0` stores none. Content recall intersects lane-wise products at the question’s occupied orders, ranks positive relevance with newer rows first among equals, and reinstates each selected concept at its own order. Missing surfaces and stale IDs are counted. Relation-only sentences can use an `episode` anchor within `ltmCapacity`. Old entries without an order axis load at order zero with a warning; retired percept, pole and place columns are dropped with a warning. |
| `relationCapacity` | positive int | `8192` | Fixed `SymbolSpace` shared-ID `LTM1` reservation, wired through both `ConceptAllocator` and `LTM0`. It must cover the highest handle, not only occupied membership records. Construction and strict checkpoint restore check this shape; exhaustion raises before allocation. |
| `equalityWitnessCapacity` | positive int | `8192` | Fixed `SymbolSpace` first-equality-witness row reservation in `ConceptAllocator.singleton_values`; independent of relation storage. Snapshots use the actual conceptual width. No live resizing. |
| `ltmStreamCapacity` | positive int | `max(1, batchSize)` | One LTM membership column per concurrent batch stream; a larger runtime batch must set it at construction. Not in the schema. |
| `chartCollapse`, `iterationsPerWord` | -- | -- | Unread; declared in `data/model.xml`. |
| `downwardGeneration` | bool | `false` | Emit a codebook head via `SymbolSpace.reconstruct()` and stash on `self._predicted_head`. |
| `language.interpretation` | float | `0.5` | Soft-interpolation weight between forward and generation directions. |

The `<language><grammar>` block contains space-role-scoped rules in
`<compose>` and `<generate>` sections, each holding `<percepts>` /
`<concepts>` / `<symbols>` sub-blocks of `<rule>` elements. The grammar XML
is the sole source of truth for which rule fires at each space-role (no
code-level `default_rule` fallback).
Legacy bare-space-role elements like `<P>P = sigma(P)</P>` directly under
`<grammar>` are still accepted.

---

## Additional knobs

Further knobs the code reads, with the module and owning symbol of each read
(line numbers drift). `data/model.xsd` carries the full schema commentary. A
knob marked "not in the schema" cannot be declared in a validated
configuration and is set only by a runtime override.

### `<architecture>` level

| Knob | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `meronomy` (attr `dMaxStable`) | `Layers.py` (`meronomy_enabled` / `meronomy_d_max_stable`) | code `off`; `data/model.xml` ships `on` with `dMaxStable="4.0"` $\Rightarrow$ effectively **ON** | Binds the meronymic slots (PS.sigma / WS.pi) directly to `[0,1]` membership kernels; `dMaxStable` bounds the complete learned fold's inverse gain. |
| `sigmaPi` | `Models.py` (BaseModel init) + `Spaces.py` (PS/WS fold builders) | `butterfly` | Fold span enum `last` \| `butterfly` \| `full`. The per-space `<butterfly>` boolean is no longer read. |
| `syntacticOrder` | `Models.py` (BaseModel init) | `0` | Parse-tree depth cap for the serial reduce sweep; `0` = unbounded. Inert in parallel mode. |
| `sentenceProtocol` | `Models.py` (BaseModel init) | follows the word-bracket mode | Accepted for compatibility: the open bracket is unconditional, and this flag schedules no perception or second forward pass. |
| `truthSet` (`<truth>` rows: text, `trust` attrs) | `Models.py` (`provision_ltm`) | (none) | Configured English truths run through the sentence driver; grammar determines their kind and external provenance supplies trust; row trust $\times$ `architecture.trust`. Admitted by the shared grammatical clause closing with external provenance. |
| `answerSynthesis` | `Models.py` (`reverseOutput`, `generate`, `reverse`) | `false`; `data/BasicModel.xml` ships `true` | `true` resolves an owned conceptual answer, runs the generate grammar, writes terminal occurrences at output brackets, and calls the common order-zero reverse. The resulting continuous percepts are scored only against supplied answers; decoded bytes are the surface report. `false` retains the numeric forward head. |
| `synthesisBindings` | `Models.py` (`_select_perceptual_bindings`) | `0`; BasicModel `4` | Number of most-salient perceptual context slots a derivation may NAME as bindings for answer synthesis; they ride in the derivation trace. `0` keeps perceptual context out of the answer. |
| `whatCurriculum` | `Models.py` (`_curriculum_questions`) | `none`; BasicModel `full` | `none` \| `present` \| `temporal` \| `full`: a Bresenham fraction (`whatCurriculumRatio`) of training batches asks past/future (`temporal`) or also inference (`full`) questions instead of the default family. `none` is byte-identical. |
| `whatCurriculumDistance` | `Models.py` (`_curriculum_questions`) | `1` | Presentation offset of the past/future curriculum questions. |
| `whatCurriculumRatio` | `Models.py` (`_curriculum_questions`) | `0.25` | Fraction of training batches that are curriculum trials. |
| `whatThinkingDetach` | `Models.py` (`_what_memory` applies it to the slot owner) | `slot` | Episode credit boundary: `slot` detaches every interaction value at append (established); `episode` keeps values appended between `begin_what_episode` / `end_what_episode` live until the episode ends. |
| `whatThinkingIterations` | `Models.py` (BaseModel init) | `1` | Configuration metadata and an optional serving-summary gate; `think()` performs one presentation and the normal controller shares `attentionBudget`. |
| `whatThinkingPressure` | `Models.py` (`_thinking_pressure`) | `linear` | Retained numerical schedule utility (`linear`, `quadratic`, `step`). The normal controller derives pressure from shared work; this setting does not change its execution. |
| `reasoningIterations` | `Models.py` (BaseModel init; public reasoning APIs) | `1` | Shared work allowance for explicit `reason_about` / `answer_query`; `0` disables those APIs. Normal `resolveAnswer` shares `attentionBudget`. A selected completed meaning grants execution; raw prompt words do not. |
| `queryReasoning` | `Models.py` (BaseModel init) | `false` | DEPRECATED alias: `true` $\Rightarrow$ `reasoningIterations = 10`. Read only when `<reasoningIterations>` is absent. |
| `ltmConsolidation` | `Models.py` (BaseModel init) | `false` | Read when selecting the serial word-major path. |
| `whereRungRatio` | `Models.py` (BaseModel init) | `32` | Ratio that assigns spatial periods to the rungs of the `.where` ladder. Not in the schema. |
| `stateless` | `Models.py`, `Language.py` | `true` | Request-scoped user TruthSet rows: on state-dict load the consolidated LTM is revived without `ORIGIN_USER` rows. `false` = stateful deployment. |
| `primingDecay` | `Models.py` (BaseModel init) | `0.9` | Decay of the observed seen ledger per prime event; placement reads this ledger. Concept spread comes from ActivationPredictor. |
| `primingBridgeCapacity` | `ConceptualSpace` | `0` | Fixed reservation in sparse 32-destination blocks for projection to the percept and property towers together. Zero preserves the complete bipartite reservation. A positive value caps that reservation; a rebuild that exceeds it raises before writing the cache or target surfaces. No resize or dropped connection. The corpus candidate reserves 8,388,608 blocks per stage. |
| `serialWordCapacity` | `Models.py` (BaseModel init) | `256`, declared in `data/model.xml` | Sentence word bound for every model and the word count used to size its pinned STM banks. Inherited from the defaults file when omitted locally; construction raises if neither file declares it. Independent of PS/WS/CS field width and recent STM depth. Longer sentences raise instead of clipping or resizing. The realised output slab allows `2W` words, and the tensor loop executes only through its final live column. |
| `serialResidualPartCapacity` | `Models.py` / `InputSpace` | `16` | Fixed compiler-visible integer radix/subword-ID extent for one cold spelling. It is not a dense percept-vector axis; the masked PS fold reduces those IDs into PS's configured live field. BasicModel uses 64. |
| `serialWordBuckets` | `Models.py` / `InputSpace` | one bucket equal to `serialWordCapacity` | Comma-separated staging capacities. The largest value must equal `serialWordCapacity`; an overlong sentence fails rather than clipping. BasicModel uses only `256`, because its compiled `torch.while_loop` has a runtime trip count and therefore needs neither padding execution nor separate 16/32/64/128 graphs. Multiple buckets remain a compatibility/performance option for the static scheduler. |
| `radialStmReduce` | `Models.py` (`_create_per_stage`) | `false` | STM idea folds use the radial (signed-safe) radmin/radmax kernels. |
| `symbolicPriming` | `Language.py` (`attach_knowledge` $\to$ `configure_priming`) | `false` | Taxonomy forward heat production (symbolic-heat retrieval). |
| `symbolTower` | `Models.py` (BaseModel init) | `false` | 3-stream CS bind (PS + WS + SS): SymbolSpace becomes the symbol tower with its own codebook stream. |
| `serialObjectMeta` | `Models.py` (BaseModel init) | `false` | Serial word-loop mereology path: InputSpace presents every residual constituent of word $w$, PartSpace sigma-synthesizes it inside iteration $w$, all configured PS/WS folds bind by location in CS, and exactly one actual-order word concept commits to STM. Legacy non-word-major inputs retain the hard same-word-span fallback. |
| `conceptIndexRead` | `Models.py` (BaseModel init) | `false` | Serial per-word idea reads through the index to the concept's `similarity_codebook` row instead of the computed percept-binding event. |
| `verbSpectrum` | `Language.py` (`LiftLayer` init; forced on by `VerbLayer`) | `false` | Verb eig-spectrum operator on the composed VP, the verb's mechanism (2026-10-19). Not in the schema. |
| `predictionTrialRatio` | `Models.py` (ModelLoss wiring) | `0.0` | Fraction of training batches run as pure next-idea prediction (recon terms zeroed). |
| `continuityNorm` | `Mereology.py` (contemplative mixin) | `'inf'` | Norm for the continuity measure (`'l2'` available). Not in the schema, like the two knobs below. |
| `continuityRatioTarget` | `Mereology.py` | `1.0` | Target ratio for the continuity measure. |
| `continuitySharpness` | `Mereology.py` | `1.0` | Sharpness of the continuity squash. |
| `TetralemmaPolicy` block (+ per-space `tetralemmaOverride`) | `util.py` (`tetralemma_policy`) | `allowExcludedMiddle=1`, `allowContradiction=0`, `neitherThreshold=0.1` | Shared catuskoti policy under `<architecture>`; a space sets `<tetralemmaOverride enabled="true">` + its own `<TetralemmaPolicy>` to override. Legacy top-level `<TetralemmaPolicy>` still read as fallback. |
| `attentionFloor` | `ModelCandidateAttention.py` | `0.1` | Weight on the residual of every unread field item in the attention cost, $\mathit{floor} \cdot (1 - \mathit{admission}) \cdot \mathit{residual}$; finite and strictly between 0 and 1. |
| `hetTolerance` | `ModelCandidateAttention.py` | `1.0` | Tolerance $\tau \in [0, 1]$ for heterogeneity on the two evidence lanes of the candidate support read into attention: $(c^+, c^-) \leftarrow (c^+, c^-) - (1 - \tau)\min(c^+, c^-)$. At $\tau = 1$ nothing changes; at $\tau = 0$ the shared overlap is removed, so *both* collapses to its net lane while *neither* survives, leaving three corners (for, against, unknown). No signed scalar appears at any setting. |
| `device` | `Models.py` (run entry) | unset | Runtime device (`cpu`, `gpu`, `cuda`, `mps`), used only when `BASICMODEL_DEVICE` is unset. |
| `trainEmbeddings` | `Models.py` (BaseModel init) | unset | Architecture-level boolean read ahead of `<training><trainEmbedding>`: `true` selects `BOTH`, `false` selects `NONE`. |

### `<architecture><training>` level

| Knob | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `seed` | `Models.py` (run entry; env `BASIC_SEED` overrides) | unset | RNG pin (torch, Python and NumPy) applied before construction, to reproduce a single CLI run. It reproduces a run and is never chosen so that a gate passes ([Testing](Testing.md)); numeric XOR certificates and measurements do not pin it. |
| `branchDiagnosticsEvery` | `Models.py` (BaseModel init; `branch_gradient_diagnostics`) | `0` | Every N training batches, log state-branch norms plus weighted reconstruction/output/expectation norms and cosines per shared operator (codebooks excluded). Checkpoint the negative-cosine streak; unused samples preserve it. Diagnostic failures warn and never stop training. `0` = off; BasicModel uses `100`; the supplied-answer tied benchmark uses `1`. See [GradientFlow](GradientFlow.md). |
| `leafDistillWeight` | `Models.py` (BaseModel init; training step) | `0.0` | Weight of the root-to-leaf distillation term added to reconstruction, in which the exact-leaf teacher supervises a from-root decoder. It applies only where the tied reconstruction does not run; `0` adds no term. |
| `equalityLossWeight` | `Models.py` (BaseModel init) | `0.0` | Positive values compute the equality-learning cost at each sentence and report it as `expectation.equal` with this weight. The value is detached and trains nothing. |
| `definitionSparsityScale` | `Models.py` (ModelLoss wiring) | `0.0` | Weight of the rank-ordered soft-L0 penalty that keeps concept definitions compact, on the in-edges beyond `definitionFreeSize`; `0` disables it. |
| `teacherReconstruction` | `Models.py` (Teacher wiring) | `false` | The Teacher owns complete-present-input reconstruction: clean targets, objective corpus addresses and loss reporting. It never overwrites the model's `.where` and `.when`. |
| `conceptualSimilarityScale` | `Models.py` (ModelLoss wiring) | `0.0` | Legacy autograd SBOW weight for non-serial experimental configurations. |

### `<architecture><data>` math dataset

| Knob | Default | Purpose |
|------|---------|---------|
| `mathStage` | `1` | `0` = direct arithmetic (`a op b` in, its value out, split by unseen pairs); `1` = dependency chains; `2` = simultaneous linear constraints. |
| `mathOperators` | `add` | Stage-0 operators, a comma list of `add` / `sub` / `mul`, or `succ` (the successor "n plus one", every fact trained). |
| `mathRange` | `64` | `R`: answers are one-hot over `[0, R)`; `OutputSpace.nOutput` must equal it. |
| `mathDepths` / `mathTestDepths` | `1,2,3` / `4,5,6` | Trained / held-out chain depths (stage 1); comma lists (the XML loader evaluates `1-3`). |
| `mathDistractors` | `2` | Maximum distractor premises. |
| `mathSeed` / `mathProblems` | `0` / `256` | Generator seed and problem count (use thousands for stage 0). |

`bin/exact.py` generates the `math` dataset on the data side only
(`Data.loadMath`). Every problem has a unique solution, every equation is
consistent, and every constant and intermediate value lies in $[0, R)$.
Variable names and constants are fresh for each problem; distractor premises
are well-formed equations over variables off the query's dependency chain, and
the premise order is shuffled. Splits partition problems by `structure`, the
shape of the dependency graph with names and constants erased, so a test
problem never shares a structure with a training problem; the held-out depths
(`mathTestDepths`) occur only in test. Stage 0 holds out unseen operand pairs
instead, and `succ` trains every fact. The generator records each problem's
depth, structure and solver order for scoring; none of it enters a
presentation.

### Per-space

| Knob (space) | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `attention` (PartSpace / ConceptualSpace / WholeSpace) | `Spaces.py` (each space init) | `off` | Symbolic-retrieval attention mode enum: `off` \| `primer` \| `second-order` \| `low-rank`. Replaces the inert `hasAttention` booleans. |
| `attentionPromotion` (ConceptualSpace) | `Spaces.py` (CS init) | `false` | Enable the row pool; its context/use parameters and `conceptualPi` also accept per-space overrides. |
| `chunkPromotionThreshold` (PartSpace) | `Spaces.py` (PS init) | `4` | Radix chunk promotion: recurrence count bar on the PerceptStore gate. |
| `chunkPromotionMinLength` (PartSpace) | `Spaces.py` (PS init) | `2` | Radix chunk promotion: minimum chunk length. |
| `demuxed` (InputSpace) | `Models.py` (`_make_perceptual_space`) | `false` | Routes what/where/when through the demuxed `ModalSpace` composite instead of the muxed PartSpace. Not in the schema. |
| `lrScale` (OutputSpace) | `Models.py` (`getOptimizer`) | `1.0` | Puts OutputSpace params in their own Adam group at `learningRate * lrScale`. |
| `definitionFreeSize` (ConceptualSpace) | `Spaces.py` (CS init) | `2` | Symbols a concept's definition may use for free (genus + differentia); the rank-ordered soft-L0 penalizes only surplus ranks. |
| `lbgThreshold` (WholeSpace) | `Spaces.py` (CS init) | `0.5` | Read only as the default of `admissionRadius`, `sqrt(lbgThreshold)`. Not in the schema. |
| `divideWithinWhole` (WholeSpace) | `Spaces.py` (WS init + tiling site) | `true` | Divides unattested type-runs into attested concepts by greedy longest-match tiling; explicit `false` keeps the undivided span. |
| `digitWholes` (WholeSpace) | `Spaces.py` (WS init + `stage_analysis_spans`) | `false` | Every digit byte is a whole by itself at the analysis cut (`12` -> `1`, `2`, each with its own `.where`), so the multi-digit numeral is a whole over digit parts rather than one type-run. Default off, byte-identical. |
| `boundaryTypes` (WholeSpace) | `Spaces.py` (WS init; `_build_boundary_predicates`) | `canonical` | Initialisation of the learned boundary weights over the atomic wholes (one column per byte value) and the property rows: `canonical` turns on the begins / ends of the whitespace, punctuation and pad rows and, under `digitWholes`, the digit row's atom level; `none` starts every weight off (the cold start is the atomic tiling). The weights are parameters (`begins_weight`, `ends_weight`, `atom_level`, logits thresholded at 0.5 in the cut). |
| `whitespaceUnits` (WholeSpace) | `Spaces.py` (WS init; `_predicate_masks`) | `false` | Present whitespace runs as units whose grammatical operation is `null` (the identity), instead of discarding them as a boundary-only class. |
| `boundaryLearningRate` (WholeSpace) | `Spaces.py` (`_update_boundary_predicates`) | `0` | Step of the boundary learner's score update at the presentation boundary: a property row's logit moves by the difference between its candidate tiling's score (recurrence of the wholes it yields minus the density penalty) and the current tiling's. `0` freezes the predicates. |
| `boundaryDensityWeight` (WholeSpace) | same | `0.05` | Weight of the wholes-per-presentation penalty in the boundary learner's score (the working-memory pressure). |
| `boundaryUpdateEvery` (WholeSpace) | same | `32` | Presentations accumulated before one boundary score update (recurrence is measured over that window). |
| `boundaryTypeWeight` (WholeSpace) | same | `0.25` | Weight of the distinct-wholes-per-occurrence penalty in the boundary learner's score (the long-term-memory pressure: a tiling that memorises sentences pays here). |
| `utilitySmoothing` (ConceptualSpace) | `Spaces.py` (CS init; `category_utility`) | `1.0` | Laplace smoothing of the category-utility counts. |
| `utilityMinCount` (ConceptualSpace) | same | `4` | Minimum presentations of a concept before its utility is trusted. |
| `admissionCount` (ConceptualSpace) | same | `2` | Recurrences of a chunked phrase before it is admitted as a concept of its own. |
| `admissionRadius` (ConceptualSpace) | `Spaces.py` (CS init) | `sqrt(lbgThreshold)` | Nearest-row identity radius for admission; defined from the LBG variance threshold (the standard deviation along the split axis) so one geometry setting governs identity and division; overridable. |
| `utilityPriorRate` (ConceptualSpace) | same | `1.0` | Step of the score update that moves the `chunk` structural prior by the admitted phrase's utility gain; 0 freezes the prior. |
| `initScale` (any space) | `Spaces.py` (`_read_init_scale`) | unset | Per-row seed magnitude for the space's codebooks; small values ($\approx 0.02$) keep the sigma/pi folds in their linear regime. Unset/0 = legacy unit prefill. |
| `impenetrableOverlap`, `impenetrableVariance` (WholeSpace) | -- | `0.0` | Unread; one configuration still declares them. |
| `commitmentDecay` (PS/CS/WS) | `Models.py` (VQ override loop) | unset ($\to$ VQ default `0.9`) | VQ-VAE EMA decay override; higher = slower, more stable assignment. |
| `codebookEmaDeadThreshold` (PS/CS/WS) | same | unset ($\to$ VQ default: `1` if `codebookRetire` else `0`) | Dead-code retire: EMA `cluster_size` below this replaces the slot with a fresh row. |
| `codebookGrowthEpsilon` (PS/CS/WS) | same | `0.0` | Stashed as `vq.growth_epsilon`; `runBatch` consults it for codebook growth. |
| `activeVectors` (ConceptualSpace) | `Models.py`, `Layers.py` | `nVectors` | Initially selectable prefix of the physically preallocated shared conceptual codebook. Prefix growth updates a fixed-shape mask in place, so reserve activation cannot replace the optimizer-owned Parameter. WholeSpace properties use their own small inventory and lifecycle. |
| `retrievalAlpha`, `retrievalBeta` (ConceptualSpace / WholeSpace) | `Language.py` (heat retrieval) | `1.0`, `0.5` | Weights of content similarity and of log heat in heat-biased candidate scoring; read per space role (ConceptualSpace, or WholeSpace for the symbolic role). |
| `retrievalGamma`, `retrievalDelta` (ConceptualSpace / WholeSpace) | same | `0.0` each | First-order and second-order carrier weights; forced to 0 in `primer` mode. |
| `retrievalTopKContent`, `retrievalTopKHeat`, `retrievalOuterTopK` (ConceptualSpace / WholeSpace) | same | `64`, `64`, `32` | Sizes of the content and heat candidate unions, and the active rows for the outer-heat builders. |

### `<SymbolSpace>` level

| Knob | Where read | Default | Purpose |
|------|-----------|---------|---------|
| `language.start` | `Language.py` (grammar load) | `"S"` | Accepted completed-derivation shapes (start patterns); named starts tag `relative_truth` / `absolute_truth` sets. |

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

See [Architecture.md](Architecture.md) for the LDU factorisation and
factor-level noise injection.

---

## Example: XOR Configuration

Abridged from `data/MM_xor.xml`:

```xml
<model>
  <architecture>
    <omissionTarget>1</omissionTarget>
    <bindingDepth>3</bindingDepth>
    <data><dataType>embedding</dataType></data>
    <conceptLayers>1</conceptLayers>

    <training>
      <numEpochs>400</numEpochs>
      <learningRate>0.005</learningRate>
      <reconstructionScale>0.1</reconstructionScale>
      <trainEmbedding>JOINT</trainEmbedding>
      <embeddingScale>0.05</embeddingScale>
      <autoload>false</autoload>
    </training>
  </architecture>

  <InputSpace>
    <nVectors>8</nVectors>
    <nDim>14</nDim>
    <nOutput>8</nOutput>
    <nOutputDim>14</nOutputDim>
  </InputSpace>

  <PartSpace>
    <!-- numeric XOR tensors have no letter identity -->
    <identityByParts>false</identityByParts>
    <nInput>8</nInput>
    <nVectors>200</nVectors>
    <basicPerceptCapacity>200</basicPerceptCapacity>
    <nDim>14</nDim>
    <nOutput>8</nOutput>
    <invertible>true</invertible>
    <synthesis>meronomy</synthesis>
  </PartSpace>

  <ConceptualSpace>
    <nInput>8</nInput>
    <nVectors>8</nVectors>
    <nDim>14</nDim>
    <nOutput>8</nOutput>
    <invertible>true</invertible>
  </ConceptualSpace>

  <WholeSpace>
    <analysis>meronomy</analysis>
    <nInput>8</nInput>
    <nVectors>64</nVectors>
    <nDim>14</nDim>
    <nOutput>8</nOutput>
  </WholeSpace>

  <OutputSpace>
    <nInput>8</nInput>
    <nVectors>1</nVectors>
    <nOutput>1</nOutput>
  </OutputSpace>

  <SymbolSpace>
    <ltmCapacity>1024</ltmCapacity>
    <language>
      <start>S</start>
      <grammar>xor.grammar</grammar>
    </language>
  </SymbolSpace>
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
      <sentenceExpectation>true</sentenceExpectation>
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

## Bounds that are not XML parameters

Some limits are arguments of calls, with defaults in code, and no
configuration element sets them.

- **Taxonomy queries.** The reader scans at most 256 concepts and 1024 native
  reference records; traversal takes at most 8 links and 1024 edge
  examinations, and public predicate evaluation uses `beam * max_steps` as its
  edge limit. A limit reached, or an unavailable reference, reports incomplete
  evidence ([TaxonomyQueries](TaxonomyQueries.md)).
- **Thought calls.** `ThoughtGrammarContext` validates a row address and the
  same local limits: 256 concepts, 1024 records, 8 path steps and 1024 edge
  examinations. They are call limits, not an episode allowance; the controller
  supplies its one episode meter ([QueryContracts](QueryContracts.md)).
- **Thought history.** `begin_thought_episode` takes an explicit `work_budget`
  (API default 32). `attentionBudget` supplies it for the normal controller,
  which forwards one meter ([ThoughtHistory](ThoughtHistory.md)).
- **Query phases** and the evidence metadata of the clause store introduce no
  configured, learned or checkpointed parameter
  ([QueryPhases](QueryPhases.md), [ExistenceEvidence](ExistenceEvidence.md)).

## Environment variables

Some parameters have an environment override: `BASIC_SEED` (`seed`),
`BASIC_PACK_SENTENCES` (`packSentences`), `BASICMODEL_RECON_PLACEMENT`
(`reconstructionPlacement`), `MODEL_AMP` (`amp`) and `BASICMODEL_DEVICE`
(`device`). [Installation](Installation.md#environment-variables) lists the
run-control variables. The code also reads `BASIC_AUTOLOAD`, `BASIC_AUTOSAVE`,
`BASIC_WEIGHTS_PATH`, `BASIC_XML`, `BASIC_MAX_SENTENCE_WORDS`,
`BASIC_FULLGRAPH`, `BASIC_PROFILE_DIAG`, `BASIC_PROFILE_LEAK`,
`BASIC_PROFILE_TENSORS`, `BASICMODEL_API_TOKEN`, `BASICMODEL_RATE_LIMIT`,
`XVQ_PROBE`, `XCHUNK_PROBE`, `EXPORT_MLX_BACKEND` and the `BASICMODEL_MPS_*`
names.

## Retired elements

An old configuration may carry elements that no longer exist. The loader
rejects these by name:

- `<architecture>`: `subsymbolicNoop`, `conceptualWidth`, `subsymbolicOrder`,
  `symbolicOrder`, `subsymbolicLoop`, `serial`, `modeSchedule`,
  `readingAttention`, `globalAttention`, `globalAttentionConsume`,
  `selectedThoughtBudget` and `thinkingBudget` (use the typed bracket and
  `attentionBudget`), and `whatThinkingPrimitives`.
- `<training>`: `interLossWeight`, `armaScale`, `interContrastiveWeight` and
  `selectedThoughtPolicyWeight` (per-level expectation and the paired grammar
  credit replace them); `reconstructionPriority`, `outputGradientRatio`,
  `reconstructionLossTolerance` and `forwardGrammarWeight` (objectives use
  separate state paths); and any nonzero `answerLossWeight`,
  `predictNextLossWeight` or `conceptualContextLearningRate`.
- Spaces: `WholeSpace.propertyBasis` (every WholeSpace is a property basis),
  an InputSpace-side `<lexer>` (it belongs to WholeSpace), `PartSpace.chunking`
  (renamed `synthesis`), and any `synthesis` or `analysis` value other than
  `meronomy`.
- `<SymbolSpace>`: `parserBackend`, `routerKind`, `chartTau`, `chartTopK`,
  `chartNoiseEps` and `wMax` (the signal router is the one parser, and STM
  depth is `stmCapacity`).

The schema does not declare `reconstruct`, `truthCriterion`, `readout`,
`categoryCodebook`, `detachedReverse`, `ideaDecode`, `overlapWhereTiling`,
`wordStore`, `gradientMode`, `useStackRouter`, `lbgMinCount`, `lbgEpsilon`,
`intraLossWeight`, `reconstructionReferenceWeight`, `verbTeacherWeight`,
`modalTeacherWeight`, `architecture.learning`, `exploreTemperature`,
`stmReduceTau`,
`wherePeriod`, `whenPeriod`, `whenRungRatio` or `language.useGrammar`, so a
configuration that declares one fails schema validation. The schema still
accepts, and the code ignores, `sparseReplace` (with a `DeprecationWarning`),
`semanticArrangement`, `useSubspaceActivation`, `thinkingLossWeight` and
`whatThinkingPolicyWeight`.

## Item 1.8 parameter amendment (2026-11-12)

`OutputSpace.conceptIds` is retired and rejected. The numeric reader has a fixed pair of weights per constructed concept seat and a bias. `dataType=numeric` plus a field grammar selects fixed numeric axes; `nInput` declares their count. The existing `raiseCount` and `attentionPromotion` govern recurrent intersections, and `trustViews` governs scalar definition evidence. No scalar scale, box-learning rate, or per-value inventory parameter is introduced. Scalar input is an optional `scalar` beside `text` in `prepInput` records; omitted and zero have distinct input representations but neither asserts a perceptual negation.

`XOR_exact.xml` retains 64 epochs and sets its existing output `lrScale` to 10 for the fixed logarithmic input scale. `MM_xor.xml`, `MM_xor_fixture.xml` and `XOR_exact.xml` have two numeric dimensions. Their former symbolic shapes survive in `MM_text.xml`, `MM_text_fixture.xml` and `XOR_symbols.xml` for verbal and mechanism fixtures. Grammar declarations decide examination and vector walks independently of inventory sizes. The default's tier-local `pi` and `sigma` remain substrate declarations and keep its parallel body.
