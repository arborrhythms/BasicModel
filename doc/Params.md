# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`).
Default values are loaded from `data/model.xml` and overridden by
model-specific configs. The schema is in `data/model.xsd`.

This document is the human-facing reference for common knobs and migrations.
`data/model.xsd` is authoritative for the complete accepted element set.

**Sizes.** Every constant set by hand that is a size (a
capacity, horizon, budget, cap, cadence or width) is a power of two, for example
`PartSpace.basicPerceptCapacity` $2^{18}$. A width made by adding such constants
stays a sum: an event is its content plus the fixed eight-coordinate band (four
`.where`, four `.when`), and a percept's code is $64 + 64$, a parts half and a
wholes half (`identityPairDim`, `propertyDim`). Seeds, versions,
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
The grammar-layer settings control the DisCoCat-like
typed composition path.

Overlay merge order:
`data/model.xml`  $\to$  `<config>.xml`  $\to$  runtime overrides.

Note on naming: the schema and the XML configs use `nInput` / `nOutput`
(sequence-length sentinels), `nDim` (per-vector dim), `nVectors`
(codebook size), and `nInputDim` / `nOutputDim` (raw shape hints for the
LiftingLayer). Older docs sometimes call `nOutput` / `nInput` `nActive`;
treat that as the same thing as `nOutput`.

> `<nWhere>` / `<nWhen>` are not set in per-file configs.
> The band widths are architectural (`canonical_shape` in `bin/architecture.py`).
> Input, part, whole and symbol events use two quadrature pairs for `.where`
> and two for `.when`; conceptual codes and final outputs have no local band.
> A complete model shares **one** spatial ladder and **one** temporal ladder.
> The spatial long period is derived from the total input/percept/symbol
> registry capacity; the temporal long period covers the LTM allocation.
> Short periods preserve one-location resolution. Integer spatial addresses
> are decoded from the band, not saved beside it. Each input modality (text,
> the numeric input of a numeric file, and the scalar beside the words) has a
> range of its own in the registry, so the range of an address gives its
> modality; a place on two axes, such as a pixel, is one row-major address in
> its modality's range, decoded by the declared shape. Everything observed in one
> field shares its time; the exact clock side-band stays. Span endpoints are
> separate extent metadata. There are no `wherePeriod`,
> `whenPeriod` or `whenRungRatio` XML keys.

---

## `<architecture>`

Core model settings. Training and data parameters are in nested
sub-elements `<training>` and `<data>` (see below).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data/dataType` | string | `"numeric"` | The data space-role, set under `<data>`: `embedding` (language modelling and chat with sequence processing; PartSpace owns the byte and word lexicon) or `numeric` (a dense slab, e.g. MNIST pixels). |
| `conceptBinding` | string | `"mixing"`; BasicModel `aligned` | PS/WS concept-formation mode. `aligned` preserves location and fuses every non-raw cumulative fold from both towers, without allocating `ConceptualCombine`; exact ordered paths ride on the concept, and its order is read from the block its row lies in. `mixing` keeps the learned `ConceptualCombine` matrix as the parallel-path alternative. |
| `processSymbols` | bool | `false` | Apply extra symbolic processing after Sigma. |
| `monotonic` | bool | `false` | Constrain invertible Sigma / Pi to $W \ge 0$ so the lift / lower chain is order-preserving on the parthood cone. |
| `ergodic` | bool | `false` | Ergodic exploration: eligible layers use `W_eff = bias * W + var * noise`; `bias`/`var` are gradient-energy buffers, not Adam parameters. See [Ergodic.md](Ergodic.md). |
| `naive` | bool | `false` | Materialise `W_eff` densely in `InvertibleLinearLayer`. Slower; debugging only. `false` uses sequential L / D / U triangular solves. |
| `attentionBudget` | nonnegative int | `32` | One allowance for the looks of a sentence, symbol recall and thought attention. Each call of `attend` spends one unit. Grammatical word composition does not spend it. Zero disables the looks and thought work while retaining ordinary word composition for the VP-only control. |
| `thoughtSearchExhaustion` | positive int | `1` | Consecutive completed region queries with no candidate before conclude is legal despite free variables or missing evidence. At that closing, source-supported forward referents are provisionally minted; regions without source evidence stay question rows. A query cut short by the budget is not a completed empty search. |
| `bindingDepth` | nonnegative int | `1` | Number of constructed percept/concept binding layers, with live recurrence between them; model capacity, not an attention work allowance. Exactly one WholeSpace is shared by these stages; its property inventory and where slice do not multiply with this depth. |
| `conceptLayers` | positive int | `1` | Concept inventory depth including order zero. It does not run repeated symbolic promotion passes. `data/model.xml` declares 1; where no file declares it, two read sites (the concept order limit and the serial word-major path) fall back to 2 and two others to 1. |
| `composeTemperature` | finite float $\ge 0$ | `0` | Sampling temperature of the hard choices in training. Zero selects the greatest logit, with the structural rule on exact ties. Positive values sample `softmax(logits / temperature)` during training. Evaluation always uses argmax. Straight-through credit always uses the untempered model softmax; this parameter never scales its gradients. |
| `reductionCost` | finite float $\ge 0$ | `1.0` | Scale of the cost of working-memory load left during compose (`WholePaths.load_cost`). Each composed program is replayed on its stack depth, and at every recorded step the depth held, as a fraction of the allowance, is charged; the sum is linear in what is left and in the steps it is left for. No learned number multiplies a count: the chooser is instead shown how much of its budget is left as a sinusoidal code among its inputs (`Language.budget_code`), so that it can learn to reduce before the cost falls ("it should be a cost, as long as the network is exposed to a temporal signal that will allow it to learn that cost"). Deadline masking is unchanged: when the rounds left do not exceed the excess depth, only binary operations are eligible. See [Language](Language.md#one-operation-per-round). |
| `attentionPromotion` | bool | `false` | Enable familiar candidates in MTM. The open sentence field observes them; the kept close promotes them. Explicit clause admission is independent of this switch. |
| `conceptMintCount` | positive int | `3` | Readings within `mtmHorizon` after which MTM nominates a row of a Pi: a basic percept at order zero, a symbol at every order above. Each nominated row brings one concept, and nothing becomes a row on its first reading. A reading is one sentence of one stream, so a form that stands twice in a sentence is counted once, and every reader of a form adds to the one count MTM keeps under it. A candidate whose horizon reading is both refines and waits. A kept defining clause also promotes its pending subject. |
| `conceptUseFloor` | decimal | `0.5` | Field activation must exceed this floor to count as an MTM access; an MTM entry is counted as used when the lesser of its two coverages with the field's used rows exceeds it. |
| `conceptMatchPart` | decimal in (0, 1] | `0.7` | Least share of size by which two same-order collections are the same when their member sets or proportions are not exactly equal: each must hold at least this share of the other, both ways (the lesser of `ConceptualMatch.match`'s two coverages of their centroids). Read by the MTM candidate match (a tie at the greatest is a draw), by the match of a witnessed concept with equal members, and by the remembered context that raises an order. Located conjunctions retain the complete native literal identity. |
| `conceptPartFloor` | decimal | `0.001` | Drop support below this magnitude when a candidate is promoted; retained definition edges merge by maximum. |
| `routerWireSerial` | string | `"both"` | Per-word router-fire gating on the serial path: `per-word` (fire per word, boundary off), `boundary` (fire only at the sentence boundary), `both` (default — both fire), `off` (neither). The per-word fire populates `symbolSpace.current_rules` for SS dispatch. See [STM.md Section 7](STM.md#7-per-word-router-firing). |
| `transformChooserHidden` | int | `0` | Hidden width of each grammar MLP; `0` derives `max(8, grammar feature width)`. Positive values override it. Canonical BasicModel currently derives 1024. |
| `transformChooserDepth` | int | `1` | Number of hidden Linear/GELU blocks in each grammar MLP, followed by one scalar Linear head; must be positive. Not the number of grammatical reductions or thought iterations. |
| `whatThinkingHidden`, `whatThinkingDepth` | int | `16`, `1` | Unread in effect: both are parsed and validated, but nothing they size is built (thought is scored by `OperationSelectionLayer.thought_logits`). |
| `intuitionCapacity` | positive int | `4096` | Slots in each preallocated sparse Pi and Sigma bank of the conceptual space's `IntuitionLayer`, which applies stored implications when thought calls `infer`. Pi addresses are rule/literal references and Sigma addresses are rules. Unit initial exponents give the exact max-min DNF. Exhaustion raises; storage never grows. `intuit` has no stack of its own: it deposits `expect`'s field and closes it upward. |
| `separateChooser` | bool | `false` | False retains live compose features and projects the answer gradient against reconstruction per optimizer unit before one shared step. True is the optional per-owner backward: each cost trains its declared owner, independently of its R/A side. Thought and generate features remain detached. |
| `reconstructionJitter` | real in `[0,1]` | `0` | Draw one strength `s ~ U[1-j,1]` per sentence and apply it to reconstruction alone, shared by all of the sentence's paths. Zero draws no random number. Output, expectation, attention and chooser credit retain scale 1.0. |
| `explorePaths` | nonnegative int | `1` | Number of distinct one-departure complete paths beside the greedy path in training. Departures draw over attention, compose, bind, thought and output; each stream keeps its strictly cheapest path on R + A, ties to the greedy path, and a departure reuses the prefix before it. Evaluation runs the greedy path alone. See [Architecture](Architecture.md#whole-paths-and-concept-episodes). |
| `omissionTarget` | float in `[0,1]` | `0.9` | Desired long-term reconstruction accuracy over visible and omitted input. `1` disables omission and its random draws; XOR proofs declare this override. |
| `omissionLimit` | float in `[0,1]` | `0.5` | Maximum independent word and within-kept-word percept-slot omission rates. One word stays whole. Omission never runs in evaluation. |
| `omissionHorizon` | positive int | `1024` | Sentence horizon for the visible/omitted accuracy averages at both omission sizes. Both rates and all four averages travel in checkpoints. |
| `generateBudget` | positive int | `32` | Architecture capacity of each free output and sentence-decoder generation walk. Fixed at model construction; its output, action, residual and feature arrays are allocated to this trip count before the walk. Exhaustion reports pending work and retains its per-lane frontier residual. Recorded input derivations keep their full structural trace capacity independently. |
| `wordSurfaceCapacity` | positive int | `64` | Maximum raw bytes in one word, fixed at construction independently of sentence input length and word count. Targets have this width; candidate byte tables reserve one additional EOS position. Bytes remain `uint8`. A longer surface raises before staging; no clipping or resizing. Native walk banks carry terminal presence and primitive support without copying the raw byte slab. |
| `ltmWindow` | positive int | `1024` | Number of the latest LTM0 rows a recall reads. Each operand's set of the question is matched against the entries of the rows in the window, order by order and never reduced across orders or rows (`EpisodicMemory.matches`), and the question's operators combine the results as they combine sets. Only the best `max_records` rows are materialized; explicitly addressed rows are read within the same limit. |
| `reconstructFromIdea` | bool | `false` | Legacy evaluation selection of recovered word ideas instead of the single-slot seed. Tied mode always returns its owned completed reconstruction, independently of this flag, and never enters the free generate chart ([Models.py](../bin/Models.py)). |
| `modifierEigEdit` | bool | `false` | Direct `LiftLayer` helper flag for the modifier's sparse eigenvalue edit. The live `modifier` grammar operator force-builds the same zero-init projection, so ordinary grammar use does not depend on this flag. When enabled for plain `LiftLayer`, a modifier modifies a composed VP by `a2 = atanh(vp) + p_vp * delta_mod`, masked by the VP's own eigen-signature. Default off keeps plain `LiftLayer` byte-identical. |
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
| `trust` | unitInterval | `1.0` | The model's trust in what it reads in training: a sentence read in training is written at this value times its words' certainty. A truth supplied through runtime `store_truths` or a static `<truthSet>` keeps its own signed trust, at face value, on its outer clause only; a conversation's sentence is written at zero. Sentence content never sets its own trust. |

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
| `intervalLogEveryBatches` | int | `0` | Every N training batches, log the reconstruction and expectation costs, LTM occupancy with the last forgetting pass, the truth view's luminosity (`truth_luminosity`) and the order-zero inventory, and run the fixed probe panels in a separate evaluation copy. Each completed forgetting pass is logged too. `0` disables. Each JSON record includes collection time and probe failures. |
| `intervalLogFile` | string | `output/intervals.jsonl` | Append-only interval JSON records. The reconstruction sample includes its native reconstructed event and byte cost. |
| `profile` | bool | `false` | cProfile the training loop. |
| `certainty` | bool | `false` | Per-neuron certainty tracking in ergodic layers. Allows individual neurons to transition exploration $\to$ exploitation at different rates. |
| `wordExpectation` | bool | `true` | Order-zero reads of the shared two-lane native-symbol activation predictor. |
| `wordExpectationLossWeight` | nonnegative float | `0.1` | Local activation MSE weight at accepted word brackets. |
| `sentenceExpectation` | bool | `true` | Enables the field read during the read of a sentence (`SymbolField`) and its residuals. It makes no estimate of a sentence. Explicit `false` bypasses that read while understanding and the single What memory remain available. |
| `expectationPrime` | bool | `true` | Enables the selection prime in compose and in `attend` independently of attenuation. Off only in the declared parity configuration (`test/fixtures/item4-parity.json`); prediction, residuals and retention stay enabled. |
| `perceptExpectationGain` | float in `[0,1]` | `0` | Scales the detached signed percept residual `attend` reads through its context projection. The forward content is the exact attended observation; this knob does not attenuate it. |
| `symbolExpectationGain` | float in `[0,1]` | `0` | Attenuation strength $g$ of the word and field reads: what was predicted is dimmed toward the floor $1 - g$, and 0 leaves content whole. See [Expectation](ExpectationRetention.md). |
| `perceptExpectationRate` | float in `(0,1]` | `0.05` | Moment EMA rate after the first external observation; padding, evaluation and speculative trials do not update it. It sets no optimizer rate: the percept expectation's parameters move in the sentence's one optimizer step. |
| `expectationOmissionThreshold` | float in `[0,1]` | `0.5` | Minimum for-lane omission confidence for the conceived absence thought. |
| `expectationEpsilon` | positive finite float | `1e-6` | Denominator floor for normalized percept and symbol residuals. |
| `expectationSaturation` | enum | `clip` | `clip` uses `min(1, ratio)`; `smooth` uses `magnitude/(denominator+magnitude)`. The entire gain is detached. The signed percept residual is never saturated. |
| `explorationUniformFraction` | float in `(0,1]` | `0.5` | Uniform share $\varphi$ of a departure's alternative draw, so every eligible non-greedy action keeps positive probability. |
| `explorationTemperature` | positive finite float | `1` | Temperature of the policy component of the alternative proposal. Credit uses the actual mixture probability. |
| `symbolFieldLossWeight` | nonnegative float | `0.1` | Weight on the field read's activation MSE (`expectation.symbols`). All source context and targets are detached. E trains the predictor independently of the paired R+A judge; zero books no term. See [ExpectationRetention](ExpectationRetention.md). |
| `independenceWeight` | nonnegative float | `0.1` | Weight of the LTM-population sparse-coding objective, owned by expectation. Applies to grammars declaring individual references. |
| `independencePriorScale` | positive float | `0.05` | Laplace activation prior and automatic-relevance pressure on unit direction columns; also scales column incoherence. |
| `independenceMintThreshold` | positive float | `0.2` | Unexplained event-coordinate residual norm required for column admission after four distinct witnesses. Admission remains a kept-closing effect. |
| `actionRateRatio` | float in `(0,1]` | `1` | Ratio $\kappa$ of the action learning rates inside reconstruction's descent: the verb's action (`change`) uses $\kappa$ times the configured rate and the modifier's action $\kappa^2$ times it; codes keep the configured rate. See [GradientFlow](GradientFlow.md). |

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
still fails atomic admission, and a serving write under pressure runs the same
pass first. The following settings are validated at model construction. The
low-water fraction must be below high water, and the four activity settings
lie in $[0, 1]$.

Forgetting reads one number, a row's **activity** in $[0, 1]$, in both tables
of LTM: an `LTM0` row (an idea) and an `LTM1` row (a relation pair). A write
sets it from the sentence's surprise, or to `activityDefault` where the
surprise is unknown; each access raises it by `activityStep`, up to one; each
pass multiplies it by `activityDecay`. A pass that finds a table at
`forgetHighWater` of its capacity drops its least active rows until occupancy
falls to `forgetLowWater`; among equals the older row goes first (an `LTM0`
row is named by its time), so the same state always drops the same rows. A row
at activity one is never lowered or dropped, and neither is a row that an
`LTM1` row references, a row that another row names as its referent, or a live
thought or expectation root.
An `LTM0` row whose activity reaches `activityPromotion` is promoted to a
concept: the minted concept becomes the row's referent and the row's activity
becomes one. See [STM](STM.md#10-ltm-as-the-chain-of-stm-end-states),
"Forgetting".

| Parameter | Type | Default | Description |
|---|---|---|---|
| `forgetting` | bool | `true` | Enable the activity pass over `LTM0` and `LTM1` and boundary concept reclamation. Capacity warnings remain visible. |
| `activityDefault` | unitInterval | `0.5` | Activity of a row written with no known surprise; a row written with a known surprise takes the surprise. |
| `activityStep` | unitInterval | `0.1` | Rise of a row's activity at each access, capped at one. |
| `activityDecay` | unitInterval | `0.9` | Factor applied to each row's activity at each pass; activity one is not lowered. |
| `activityPromotion` | unitInterval | `0.9` | Activity at which an `LTM0` row is promoted to a concept. |
| `forgetHighWater` | float | `0.9` | Occupancy fraction of a table at which a pass drops rows. |
| `forgetLowWater` | float | `0.7` | Occupancy fraction a pass drops to; rows that cannot be dropped can prevent reaching it, with a warning. |
| `forgetMinInterval` | int | `256` | Minimum completed sentence presentations between unpressured passes. High-water pressure in either LTM table overrides it at a closed boundary. |
| `forgetAtEpoch` | bool | `false` | Also run a pass at a completed training epoch, subject to the interval; below high water it only decays. |

## BasicModel / MentalModel Quick Reference

`data/BasicModel.xml` and `data/MentalModel.xml` train the full
P$\to$C$\to$S pipeline. Both load `complete.grammar`, in which `change` is a
binary rule (`change(NP, VP)` applies the verb phrase's action to its
subject) and `verb(VP, NP)` gives the verb phrase its basis from its object:
transitive SVO is assembled compositionally, not by a ternary rule. `lift` and
`lower` take one operand each. Grammar mode is derived from the loaded
grammar.

---

## Space sections

Every space block uses the same sentinel convention:

- `nInput = 0` --- InputSpace: derive from `TheData`; other spaces: derive from the previous space's `outputShape[0]`.
- `nOutput = 0` --- same as `nInput` for this space (passthrough count).
- `nVectors = 0` --- same as `nOutput` (used for non-codebook spaces).

Per-Space `<nWhere>` / `<nWhen>` are not declared (see top-of-file note); the per-Space defaults live in
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

> **`<lexer>` moved to `<WholeSpace>`**: lexing is analytic
> CUTTING, owned by the analysis side. An InputSpace-side `<lexer>` now
> fails schema validation and the runtime reader. InputSpace emits the
> dual view — `forward(x) -> (percepts_in, concepts_in)`: the atom view
> (content `[B, N, 1]`) for the perceptual branch and the unity view
> (`[B, 1, N]`) for the symbolic branch.

`inputShape` uses the data's native dim (e.g. 784 for MNIST,
`inputLength` for text); `outputShape` uses `nDim` from
`TheObjectEncoding`. `LiftingLayer` bridges the two.

### `<PartSpace>`

Holds the parts of forms. A part is a run of adjacent characters, the space
among them; a form's code is the set of its parts, so one form lies inside
another exactly when its code lies inside the other's. A longer part is added
only where two forms would otherwise have the same parts, or where it is
taught ([Mereology](Mereology.md)). Native PartSpace has no `SigmaLayer` or
`PiLayer`: the first Pi, whose rows are the basic percepts, is described in
[Architecture](Architecture.md#sigma-and-pi-layers).

`nVectors` is a fixed reserve of part rows. Rows store admitted parts, so the
sentence count alone does not determine the number needed. If the reserve
fills, training stops with a capacity error: the table cannot grow while an
optimizer owns it. BasicModel reserves 32,768 rows, a small-run setting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Number of active perceptual feature vectors. Should be $\geq$ `InputSpace.nOutput`. For XOR: 4. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$ (default 9 = 1 + 4 + 4). |
| `nVectors` | int | sentinel | Fixed physical percept-codebook capacity. Logical admission writes unused rows without changing tensor shape or optimizer ownership. Exhaustion raises an error naming this knob. Serial attention may select `nOutput` rows; parallel attention reads all active percepts. |
| `propertyDim` | positive int | `64` | Width of the wholes half of a percept's code, after the parts half, at text-model construction. The part catalog is drawn a second time in these coordinates; a whole's code is the OR of the codes of the catalog parts that belong to it, so the class of all characters is all ones. Numeric layouts are unchanged. |
| `propertyOnes` | positive int | `3` | Coordinates each catalog part takes in the wholes half; must not exceed `propertyDim`. Every byte is a part of the catalog, so the OR of all of them covers the half. |
| `invertible` | bool | `false` | Space construction flag; native reconstruction uses retained ordered perceptual views and canonical part ids, without an inverse fold. |
| `hasAttention` | bool | `false` | Inert: the boolean constructs no attention pass and is kept as an alias. Use `<attention>` (`off`, `primer`, `second-order`, `low-rank`). |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `synthesis` | string | `"meronomy"` | Bottom-up synthesis strategy. `meronomy` is the only accepted value, and any other value raises. Top-down cuts live on WholeSpace `<analysis>`. |
| `butterfly` | bool | -- | Unread; configurations still declare it. The fold span is the architecture-level `sigmaPi`. |
| `wordLearning` | int | `2` | Active lexicon-growth mode. `0` = frozen codebook; `>=1` = on first sight of a new word, insert into the lexicon and tag it on the meronomy. Several trained configs override this to `0` or `1`. |
| `basicPerceptCapacity` | positive int | `262144` | Fixed number of basic-percept rows, allocated at construction; a full table raises. Lesson configurations may declare a smaller capacity. |
| `identityByParts` | bool | `true` | A text form's code is the set of its parts, a parts half beside a wholes half; see the width rule below. |
| `identityPairDim` | positive int | `64` | Width $D$ of the parts half of a percept's code. The parts are spread over all $D$ coordinates. |
| `identityOnes` | positive int | `3` | Coordinates $s$ each part takes in the parts half, $0 < s \le D$; a form's half is the join of its parts'. |

A form's code holds no length; a configuration that declares
`identityLengthDim` raises.

In an embedding configuration with `identityByParts` (the default),
`WordIdentity.resolve_identity_layout` rewrites the declared widths before any
space is built. Every input, percept and whole event is `identityPairDim` +
`propertyDim` + the eight band coordinates, $64 + 64 + 8 = 136$ by default, and the conceptual width is that plus the declared difference
between the ConceptualSpace and PartSpace `nDim`. The declared `nDim` values of
such a configuration therefore matter only through that difference; numeric
configurations, and those with `identityByParts` false, keep their declared
widths.

PartSpace has no `<codebook>` element: the percept prototypes live on the
`.what` basis. See [Architecture.md](Architecture.md) and
[Philosophy.md](Philosophy.md) for the analysis/synthesis orientation.

### `<ConceptualSpace>`

Order-zero concepts are a sparse Sigma over the signed values the Not gives
for a reading's basic percept rows, one concept for each basic percept; above
order zero a Pi makes symbols over concepts, the Not joins their lanes, and a
Sigma makes the concepts, each order with
half the units of the one below ([Architecture](Architecture.md#sigma-and-pi-layers)).
`conceptUseFloor` is the separate discovery gate over measured use.

Holds the conceptual dictionary, the order-zero Sigma, and short-term memory.
Grammatical lift/lower own their chart maps.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Active concept vectors. For XOR: 3. |
| `nDim` | int | `9` | Per-vector muxed EVENT width; content $\mathrm{nWhat} = \mathrm{nDim} - \mathrm{nWhere} - \mathrm{nWhen}$ (default 9 = 1 + 4 + 4). |
| `nVectors` | int | sentinel | Physical conceptual-codebook capacity. ConceptualSpace owns the concept inventory and taxonomy; word–object DEF relations live in the common LTM store. Aligned property-basis stages share one table. Rows lie in their order's block (`order_slice`), so a concept's order is read from where its row lies and is stored nowhere beside the vector; a full block raises a capacity error naming its order. This value is independent of `WholeSpace.nVectors`, PartSpace's physical capacity, live `nOutput`, SymbolSpace references, and STM capacity. |
| `invertible` | bool | `false` | Space construction flag. Aligned native binding retains both perceptual views for reconstruction; ConceptualSpace owns no unary chart fold. |
| `hasAttention` | bool | `false` | Inert: the boolean constructs no attention pass and is kept as an alias. Use `<attention>`. |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `butterfly` | bool | -- | Unread; configurations still declare it. |
| `stmCapacity` | int | `8` | STM ring depth (within Miller's $7 \pm 2$ band). Per-batch buffer `[B, stmCapacity, nDim]` + depth pointers `[B]`. Thought content starts at four rows per focus chunk and may grow to its exact size, bounded by four times the declared sentence leaves plus operator seats; this changes no focus depth. |
| `mtmCapacity` | positive int | `64` | Shared capacity of `ConceptualSpace.mtm` for identities and familiar candidates. Identities retain working values, mention readings and event histories; candidates retain bounded sparse support, a centroid and two-lane readings. Oldest access is evicted at capacity without changing LTM rows. Candidate content reads through the live member codes. |
| `mtmHorizon` | positive int | `16` | Maximum intervening kept closes in a stream since the last reference. Identities and candidates share this access clock and horizon. MTM is not cleared at the end of a document or at an epoch boundary. Recurrence for `conceptMintCount` counts retained observations within the horizon, independently of access count. Pending definition operands hold their seats. Among the others, eviction takes an entry untouched this close first, then the least recent access, then the earliest entry time. If all seats are held, admission raises "MTM capacity ... exhausted by pending definition operands". |
| `mtmPrepend` | positive int | `64` | Leading symbol entries reserved at construction for MTM identities and candidates together. |
| `verbGroups` | positive int | `24` | Maximum learned intransitive bases on paired meaning configurations. A third distinct failed-fit occurrence joins the best fitting group or opens one; capacity is reported. |
| `verbBasisReflections` | positive even int | `4` | Householder reflections per object/group basis. Identical initial pairs give the identity. |
| `verbGroupFit` | nonnegative float | `0.1` | Maximum mean squared chart residual for admitting a verb to an existing basis. |
| `latticeMargin` | decimal | `0` | Unread; declared in `data/model.xml`. |
| `meaningWidth` | nonnegative int | computed | Width of the meaning band of a concept's code; unset, it is the code width less the percept event width. `data/model.xml` ships 128. At 0 a paired chart supplies only form: membership records remain possible, but there are no class directions or conceptual views, and hierarchical conditioning needs a meaning band. |
| `placementRate` | nonnegative float | `0.1` | Step size of the one meet-ratio subgradient step that places a concept's meaning at each kept close. The context of placement is the field at every order, a present token's types among it, read before any weighting by likeness of places; a place moves relative to rows of its own order only, and a class is placed by its context like any concept. A row above the basic percept carries no perceptual form. |
| `placementNegatives` | nonnegative int | `5` | Placed rows with zero seen excess drawn as negatives by a generator seeded from the sentence's key; fewer eligible rows give fewer negatives, never repeats. |
| `placementNegativeWeight` | nonnegative float | `1.0` | Weight of the negative term, the brake on collapse. |
| `raiseEvidence` | unitInterval | `0.0` | A retained row qualifies as a witness when its positive evidence exceeds this value. |
| `raiseCount` | int $\ge 2$ | `3` | Number of pairwise distinct witnesses of the same order required to create the shared side's class. |
| `readerBlockNormalization` | bool | `true` | Scale the form block and the meaning block of a value read from the sentence's primed symbol bank, each by that block's largest norm over the frozen bank, so the read stays affine. |
| `intuitHidden` | int | `64` | Width of the one hidden layer of `intuit`'s own map, a Pi, Not and Sigma stack from the concepts of order zero to themselves that stands outside the architecture (`ConceptualSpace.intuit_map`). It is sized once, at construction. Each row of its first Pi starts on two concepts: a row started on every concept does not learn on exact concepts. |
| `intuitLayers` | int | `1` | Pi and Sigma pairs of `intuit`'s map. |
| `fragmentBias` | float in $[0, 1]$ | `0` | Probability that, at a look whose word begins a run of two or three unread known words that the kept reading made one constituent, the grammar takes that run as one leaf read at one time through the idea stack (`FragmentRead`), with no composition over its words. At `0` nothing of the read runs: no read, no teaching and no take. Above `0` the read is also taught, by the composed value of the constituent (`expectation.fragment`) and by the class it filled (`expectation.fragment_class`). No form, row or symbol is made for a run. |
| `wordTieCapacity` | positive int | `256` | Room for the ties of words to objects kept in MTM (`WordTies`). A tie holds its own certainty; the weakest tie is forgotten when room is needed. Read by the source and not yet declared in the schema. |
| `candidateCapacity` | positive int | `64` | Room of each medium-term matrix of candidate entries beside a Pi or a Sigma (`CandidateGrowth`). Read by the source and not yet declared in the schema. |
| `candidatePenalty` | float $\ge 0$ | `0.2` | The penalty a candidate entry must earn its weight against (`CandidateMatrix`); measured as the middle of the strengths (.1 to .3) at which the trials make exactly the right entries. Read by the source and not yet declared in the schema. |
| `structureReadThreshold` | float $\ge 0$ | `0.001` | The weight at or above which thought reads an entry of a Sigma or a Pi as present (`StructureRead`: `parts`, `wholes`). Read by the source and not yet declared in the schema. |
| `ideaHidden` | positive int | `64` | Width of the hidden layer of the read to a point, from the observed order-zero field, both lanes, to one point of a code's width. Built only on the conceptual space that owns the concepts, and only where sentences are comprehended. |
| `ideaLayers` | positive int | `1` | Pi and Sigma pairs of the read to a point. |
| `markerKernels` | int | `8` | Learned kernels of the one-operand `marker` operator. Each kernel is one candidate of the operator, chosen by the chooser from the sentence's context; none is named. |
| `markerHidden` | int | `64` | Hidden width of each `marker` kernel's stack, from the operand's form block to its meaning block. |
| `markerLayers` | int | `2` | Pi and Sigma pairs of each `marker` kernel's stack. |

`intuitHidden`, `intuitLayers`, `ideaHidden`, `ideaLayers`,
`markerKernels`, `markerHidden` and `markerLayers` are declared in
`data/model.xsd`, each with the default above, so a configuration that omits
them is unchanged. The three maps are one class, a stacked sigma-pi
(`SigmaPiStack`), a row of fused pairs (`SigmaPiLayer`), each the structure
of every order: a Pi of unsigned rows on the two lanes (`PiLayer3`, a product
of the inputs' memberships with weights at zero or above), then a Sigma
(`SigmaLayer3`) whose entries are for or Not and which gives one signed
number in $[-1, 1]$ for each output. The next pair reads that number as its
two lanes, its positive part for and its negative part against. The layer
counts count pairs. The last
sigma's weights start at zero, so a new stack gives exactly zero; and its
first weights come from a private
generator, so building one changes no other random draw. `expect` publishes
its field once, at the open pass, and is taught by omission: where units of the sentence were
hidden, the field with them present is its detached teacher, and a sentence
with nothing hidden gives it no cost (`expectation.concepts`). `intuit` uses
`expect`'s field and has no stack. The read to a point is taught by the root
idea the kept reading leaves (`expectation.idea`). A `marker` kernel passes
the form block unchanged and shifts the meaning in the chart; a new kernel is
the identity.

`ConceptualSpace` owns definitions, evidence and STM/event state. On aligned
BasicModel, native PS and WS views bind by location and the actual concept
order is read from the block its row lies in; symbolic generalization remains
a separate phase.

#### Trust

| Parameter | Reader | Default | Meaning |
|---|---|---|---|
| `trustViews` | `ConceptTrust`, `PartsOfSpeech.operation` | `3` | Finite positive view scale in $\ell = s(1 - e^{-n/\mathrm{trustViews}})$: $n$ counts kept views and $s$ is the accumulated request coverage divided by that count. Views allocate no memory rows. Empty stored content agrees with a view by one, and its first nonempty view becomes the reference. The radius applies only to the for lane. An operand's trust comes from a seated, tracked concept row; an untracked operand contributes one, and the whole norm of an untracked chart remains its extent. A relation's effective pair is its stored pair times the least confidence of its endpoints; an operator contributes no factor. An object reached through a read word is at radius one, and the rise by views does not apply to it. The same rule gives a formal argument its class-view radius, and each formal argument of an operation has a definiteness, its separability times $1 - e^{-\mathit{views}/\mathrm{trustViews}}$, which the lessons read as a measurement; it opens nothing and scales no row's trust. |
| `definitionTrust` | `DefinitionIndex.add` | `0.5` | Finite threshold in $[0,1]$ for a named row's effective positive pole to admit its word-to-object binding; equality is admitted. The threshold gates the binding, never the word's identity: below it a named word resolves by its form. Naming writes the source evidence even below the threshold, so learning can change admission without another write. |

No setting gates an operator. Every operator a grammar file declares is open
from the first sentence, and no row's trust is scaled by an operator's
concept; the lessons are the stages at which a test suite moves on
([Testing](Testing.md#rules-for-tests-and-measurements)). Runtime validation
rejects nonfinite or out-of-range values. The learned-argument inventory
reserves one more native concept seat per declared formal argument at
construction, beside the configured inventory capacity.

### `<WholeSpace>`

Each whole is one row of the table of wholes, defined by what belongs to
it: its learned memberships over the 256 byte primitives
(`PrimitiveProperties.members`), or other rows of the table, whose AND it is
(`PrimitiveProperties.define_by_rows`). A row's code is the OR of the codes
of the catalog parts that belong to it, in the wholes half of the percept's
code; the class of all characters is all ones, and a class inside another
lies at or below it. A-priori byte examples teach the initial properties;
runtime analysis reads those parameters. Constant-signature runs keep their
input brackets. A whole holds of a form where it holds over the form's run,
and a basic percept reads it there ([Architecture](Architecture.md#sigma-and-pi-layers)).

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
| `analysis` | string | `"meronomy"` | Top-down division over the unity view. `meronomy` is the only value a WholeSpace accepts: any other value raises (`byte`, `raw`, `sentence`, `word` and `grammatical` among them), and the spelling `analyse` is not accepted by the schema. |
| `analyzer`, `lexer` | string | -- | Deprecated spellings of the intake cut: the reader prefers `<analyzer>` and falls back to `<lexer>`, and `<analysis>` supersedes both. An InputSpace-side `<lexer>` is rejected. |
| `nonlinear` | bool | `true` | Tanh-bound the activation. |
| `sortNetwork`, `sortPasses`, `l1Lambda`, `symbolResidualScale`, `outputSymbolResidualScale`, `commitmentBeta`, `decorrelationWeight`, `spectralFlatnessWeight` | -- | -- | Unread; declared in `data/model.xml`. |

WholeSpace owns primitive property memberships and the native analysis
policy. It allows primitives by max and computes pervasion by min over
observed positions. It has no `PiLayer` or inverse fold; grammatical
operators own their parameters independently of either perceptual tower.

### `<OutputSpace>`

Maps symbols to final predictions, or reads the field.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Output values. For XOR: 1. |
| `nDim` | int | `1` | Per-output dim. |
| `nVectors` | int | sentinel | Codebook size. Defaults to `nOutput`. |
| `invertible` | bool | `false` | Rarely set on OutputSpace. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `nonlinear` | bool | `false` | OutputSpace is linear by default; rescales via `Data.denormalize`. |

Where the grammar reads the field, the head is one linear layer over it
(`OutputSpace.read_field`): a fixed pair of weights for each conceptual row,
one for each evidence lane, and a bias. `OutputSpace.conceptIds` is rejected:
the output reads every concept row and both lanes. Otherwise, with
`nVectors == 1` (or no codebook) the head is an unquantised linear map
`nInputDim -> nOutputDim` plus a learned scalar intercept. `LinearLayer` has
`(bias, temp)` support for ergodic mode.

---

## `<SymbolSpace>`

Grammar infrastructure (SyntacticLayers, TruthLayer). Lives outside
`<architecture>` as a top-level sibling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `syntacticHiddenDim` | int | `64` | SyntacticLayer MLP hidden size. |
| `truthMaxEntries` | int | `1024` | Capacity of the truth view: `TruthLayer.sync_from_ltm` takes the latest `truthMaxEntries` LTM0 rows whose trust is not zero, each represented at its trust. A row is a truth by its nonzero trust alone. |
| `ltmCapacity` | int | `1024` | Capacity of the shared LTM occurrence store and bounded inter-sentence history. The `.when` ladder uses `RowTime.HORIZON`, independently of this reservation. An explicit value takes precedence over `truthMaxEntries` for LTM; `truthMaxEntries` sizes the truth activation view. The store raises when full. MM_math_chain reserves 131,072 rows for its seventeen graded presentations; exhaustion still raises. See [STM.md Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states). |
| `episodicConcepts` | nonnegative int | `4` | Number of an LTM0 row's slots beside its idea: concepts of the sentence's field of understanding, over all orders, each with its value, in one flat list with no order axis (`episodic_concepts`, `episodic_values`). A value's sign is its lane, and contradictory lanes occupy separate slots. `0` stores none. Recall matches a question against the slots order by order, never reducing across orders, ranks positive relevance with newer rows first among equals, and pushes the best row onto the STM stack. |
| `relationCapacity` | positive int | `8192` | Fixed `SymbolSpace` shared-ID `LTM1` reservation, wired through both `ConceptAllocator` and `LTM0`. It must cover the highest handle, not only occupied membership records. Construction and strict checkpoint restore check this shape; exhaustion raises before allocation. |
| `equalityWitnessCapacity` | positive int | `8192` | Fixed `SymbolSpace` first-equality-witness row reservation in `ConceptAllocator.singleton_values`; independent of relation storage. Snapshots use the actual conceptual width. No live resizing. |
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
| `sigmaPi` | `Models.py` (BaseModel init) + `Spaces.py` (PS/WS fold builders) | `butterfly` | Fold span enum `last` \| `butterfly` \| `full`. The per-space `<butterfly>` boolean is not read. |
| `syntacticOrder` | `Models.py` (BaseModel init) | `0` | Parse-tree depth cap for the serial reduce sweep; `0` = unbounded. Inert in parallel mode. |
| `sentenceProtocol` | `Models.py` (BaseModel init) | follows the word-bracket mode | Accepted for compatibility: the open run, which reads the whole sentence at once, is unconditional, and this flag schedules no perception or second forward pass. |
| `truthSet` (`<truth>` rows: text, `trust` attrs) | `Models.py` (`provision_ltm`) | (none) | Configured truths run through the sentence driver and the shared grammatical clause closing. Each `trust` attribute, in $[-1, 1]$, is written at face value on the truth's outer clause, independently of `architecture.trust`. |
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
| `stateless` | `Models.py`, `Language.py` | `true` | Serving lifecycle. `true` runs no forgetting pass after a conversation's request; `false` (a stateful deployment) runs the activity pass after each. Restoring a checkpoint keeps every retained row and its trust: a row has no origin by which it could be withdrawn. |
| `primingDecay` | `Models.py` (BaseModel init) | `0.9` | Decay of the observed seen ledger per prime event; placement reads this ledger. Concept spread comes from `expect` (`ConceptSpread`). |
| `primingBridgeCapacity` | `ConceptualSpace` | `0` | Fixed reservation in sparse 32-destination blocks for projection to the percept and property towers together. Zero preserves the complete bipartite reservation. A positive value caps that reservation; a rebuild that exceeds it raises before writing the cache or target surfaces. No resize or dropped connection. The corpus candidate reserves 8,388,608 blocks per stage. |
| `serialWordCapacity` | `Models.py` (BaseModel init) | `256`, declared in `data/model.xml` | Sentence word bound for every model and the word count used to size its pinned STM banks. Inherited from the defaults file when omitted locally; construction raises if neither file declares it. Independent of PS/WS/CS field width and recent STM depth. Longer sentences raise instead of clipping or resizing. The realised output slab allows `2W` words, and the tensor loop executes only through its final live column. |
| `serialResidualPartCapacity` | `Models.py` / `InputSpace` | `16` | Fixed compiler-visible integer radix/subword-ID extent for one cold spelling. It is not a dense percept-vector axis; the masked PS fold reduces those IDs into PS's configured live field. BasicModel uses 64. |
| `serialWordBuckets` | `Models.py` / `InputSpace` | one bucket equal to `serialWordCapacity` | Comma-separated staging capacities. The largest value must equal `serialWordCapacity`; an overlong sentence fails rather than clipping. BasicModel uses only `256`, because its compiled `torch.while_loop` has a runtime trip count and therefore needs neither padding execution nor separate 16/32/64/128 graphs. Multiple buckets remain a compatibility/performance option for the static scheduler. |
| `radialStmReduce` | `Models.py` (`_create_per_stage`) | `false` | STM idea folds use the radial (signed-safe) radmin/radmax kernels. |
| `serialObjectMeta` | `Models.py` (BaseModel init) | `false` | Serial word-loop mereology path: InputSpace presents every residual constituent of word $w$, PartSpace sigma-synthesizes it inside iteration $w$, all configured PS/WS folds bind by location in CS, and exactly one actual-order word concept commits to STM. Legacy non-word-major inputs retain the hard same-word-span fallback. |
| `conceptIndexRead` | `Models.py` (BaseModel init) | `false` | Serial per-word idea reads through the index to the concept's `similarity_codebook` row instead of the computed percept-binding event. |
| `verbSpectrum` | `Language.py` (`LiftLayer` init; forced on by `VerbLayer`) | `false` | Verb eig-spectrum operator on the composed VP, the verb's mechanism. Not in the schema. |
| `predictionTrialRatio` | `Models.py` (ModelLoss wiring) | `0.0` | Fraction of training batches run as pure next-idea prediction (recon terms zeroed). |
| `continuityNorm` | `Mereology.py` (contemplative mixin) | `'inf'` | Norm for the continuity measure (`'l2'` available). Not in the schema, like the two knobs below. |
| `continuityRatioTarget` | `Mereology.py` | `1.0` | Target ratio for the continuity measure. |
| `continuitySharpness` | `Mereology.py` | `1.0` | Sharpness of the continuity squash. |
| `TetralemmaPolicy` block (+ per-space `tetralemmaOverride`) | `util.py` (`tetralemma_policy`) | `allowExcludedMiddle=1`, `allowContradiction=0`, `neitherThreshold=0.1` | Shared catuskoti policy under `<architecture>`; a space sets `<tetralemmaOverride enabled="true">` + its own `<TetralemmaPolicy>` to override. Legacy top-level `<TetralemmaPolicy>` still read as fallback. |
| `interpretBias` | `OperationSelectionLayer.interpret_at_look` | `1` | Probability of interpreting the most active word concept of a look's returned set directly. The learned grammar decides the remaining offers. Accepts 0 through 1 in training and evaluation. |
| `hetTolerance` | `AttentionContext.install`, `SentenceField.arrive` | `1.0` | The fraction of overlapping evidence retained on both lanes of an arrival. Zero removes the shared overlap, and one preserves both lanes. |
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
| `attentionPromotion` (ConceptualSpace) | `Spaces.py` (CS init) | `false` | Enable the row pool; its context/use parameters also accept per-space overrides. |
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
    <data>
      <dataType>numeric</dataType>
      <dataset>xor</dataset>
    </data>
    <conceptLayers>1</conceptLayers>

    <training>
      <numEpochs>400</numEpochs>
      <learningRate>0.005</learningRate>
      <reconstructionScale>0.1</reconstructionScale>
      <trainEmbedding>JOINT</trainEmbedding>
      <embeddingScale>0.05</embeddingScale>
      <autoload>false</autoload>
    </training>
    <conceptBinding>aligned</conceptBinding>
    <attentionPromotion>true</attentionPromotion>
  </architecture>

  <InputSpace>
    <nVectors>2</nVectors>
    <nDim>9</nDim>
    <nOutput>2</nOutput>
    <nOutputDim>9</nOutputDim>
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
      <grammar>field.grammar</grammar>
    </language>
  </SymbolSpace>
</model>
```

Everything not listed inherits from `data/model.xml`. The two numeric
dimensions are read through `NumericPercepts`, and `field.grammar` declares
only the field operators `intersection`, `union` and `not`.

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
  `selectedThoughtBudget` and `thinkingBudget` (use the looks and
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
- `<OutputSpace>`: `conceptIds` (the output reads every concept row and both
  lanes).

The schema does not declare `conceptualPi` (a basic percept is a row of the
first Pi, which every model has), `symbolTower` (symbols are split from
concepts and have no codebook of their own), `attentionFloor` (attention dims
and does not cut), `identityLengthDim`, `conceptReadoutL1` (no readout
follows the order-zero Sigma), `reconstruct`, `truthCriterion`, `readout`,
`categoryCodebook`, `detachedReverse`, `ideaDecode`, `overlapWhereTiling`,
`wordStore`, `gradientMode`, `useStackRouter`, `lbgMinCount`, `lbgEpsilon`,
`intraLossWeight`, `reconstructionReferenceWeight`, `verbTeacherWeight`,
`modalTeacherWeight`, `architecture.learning`, `exploreTemperature`,
`stmReduceTau`, `rowExpectationGain`, `documentExpectationGain` and
`sentenceExpectationLossWeight` (nothing estimates a sentence),
`expectationMap`, `expectHidden` and `expectLayers` (expectation has no map
beside the structure; the stack is `intuit`'s, under `intuitHidden` and
`intuitLayers`), `transformChooser` (the chooser of acts is the network
alone), `reducePressure` (the load is a cost, `reductionCost`),
`wherePeriod`, `whenPeriod`, `whenRungRatio` or `language.useGrammar`, so a
configuration that declares one fails schema validation. Nor does it declare
the weighted forgetting settings `forgetTrustWeight`, `forgetUtilityWeight`,
`forgetLuminosityWeight`, `forgetAgeWeight`, `forgetHalfLife`,
`forgetSimilarity`, `forgetCoverage` and `forgetProtectOrigins` (activity
orders a pass, and a row has no origin); `ConceptualSpace.recurrenceAngle` and
`recurrenceCapacity` (a sentence read again is a new row);
`ConceptualSpace.gates` with its `closedTrust` and `threshold` (no operator is
gated); `adverbEigEdit` (renamed `modifierEigEdit`); or `symbolicPriming` and the
heat-retrieval knobs `retrievalTopKContent`, `retrievalTopKHeat`,
`retrievalOuterTopK`, `retrievalAlpha`, `retrievalBeta`, `retrievalGamma`,
`retrievalDelta` and `retrievalUseOuterProduct`. The schema still
accepts, and the code ignores, `sparseReplace` (with a `DeprecationWarning`),
`semanticArrangement`, `useSubspaceActivation`, `thinkingLossWeight` and
`whatThinkingPolicyWeight`.

## Numeric and scalar input

`dataType=numeric` with a grammar of field operators reads its input through
`NumericPercepts`, the one reader for any number of dimensions; the input's
shape gives the number. Each dimension is a place in the numeric input's own
`.where` range. A reading whose felt value is above zero is a value percept at
its place. Its code is its place in one sorted table of values, over 64 bins
of the felt value: the parts half holds the bins at or below the value, the
wholes half is all ones with the bins above it cut away, and the against lane
holds what the wholes half cuts away, so a value lies between its parts and
its wholes as a form does ([Mereology](Mereology.md)). There is one basic
percept for each place and value met; each place has a table of its own, on
coordinates of its own, and a collection of values is read by its bounds, 0
or 1, not through the Sigma. Zero is no presence. Reverse
returns each dimension's number from the level its concept holds. The output
reads the field (`OutputSpace.read_field`). The existing `raiseCount` and
`attentionPromotion` govern recurrent intersections, and `trustViews` governs
scalar definition evidence. No scalar scale, box-learning rate or per-value
inventory parameter exists. Scalar input is an optional `scalar` beside `text`
in `prepInput` records; omitted and zero have distinct input representations,
but neither asserts a perceptual negation.

`MM_xor.xml`, `MM_xor_fixture.xml` and `XOR_exact.xml` have two numeric
dimensions; `XOR_exact.xml` runs 64 epochs and sets its output `lrScale` to 10
for the fixed logarithmic input scale. `MM_text.xml`, `MM_text_fixture.xml` and
`XOR_symbols.xml` keep symbolic shapes for verbal and mechanism fixtures.
`data/MM_image.xml` is numeric, with field operators only, a 28 by 28 input on
two place axes, and reads two-level images from
`image_corpus.load`. Grammar declarations decide examination and vector walks
independently of inventory sizes. The default's tier-local `pi` and `sigma`
remain substrate declarations and keep its parallel body.
