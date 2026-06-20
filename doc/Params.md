# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`).
Default values are loaded from `data/model.xml` and overridden by
model-specific configs. The schema is in `data/model.xsd`.

Overlay merge order:
`data/model.xml`  $\to$  `<config>.xml`  $\to$  runtime overrides.

Note on naming: the schema and the XML configs use `nInput` / `nOutput`
(sequence-length sentinels), `nDim` (per-vector dim), `nVectors`
(codebook size), and `nInputDim` / `nOutputDim` (raw shape hints for the
LiftingLayer). Older docs sometimes call `nOutput` / `nInput` `nActive`;
treat that as the same thing as `nOutput`.

> 2026-05-28: `<nWhere>` / `<nWhen>` are retired from per-file configs.
> `data/model.xml` carries per-Space defaults (`<nWhere>2</nWhere>` on
> InputSpace / PartSpace / WholeSpace, `0` elsewhere). The
> `.where` field is the canonical positional identifier; see
> [doc/old/2026-05-28-where-keyed-taxonomy.md](old/2026-05-28-where-keyed-taxonomy.md).
>
> 2026-06-16: `.where` and `.when` are now **endpoint-sum brackets** `[start,
> end]` (angle = span center, magnitude = extent), the invertible
> `EndpointSumWhere` form. An instant (`start==end`) is byte-identical to the
> prior single-quadrature point, so `whereScale` / `whenScale` are unchanged.
> `.when` tense is the interval-vs-`now` relation (the magnitude-D tense scheme
> is retired); see [doc/Spaces.md](Spaces.md).

---

## `<architecture>`

Core model settings. Training and data parameters are in nested
sub-elements `<training>` and `<data>` (see below).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data/dataType` | string | `"numeric"` | The data tier (was the retired architecture-level `modelType`), set under `<data>`: `embedding` (LM / chat with sequence processing — PartSpace owns the byte/word lexicon) or `numeric` (dense slab, e.g. MNIST pixels). The old `simple`+`passthrough` collapsed to `numeric`; `vq` was dropped (0 configs). |
| `reconstruct` | string | `"concepts"` | Reverse-pass mode: `none` disables; `symbols` reconstructs from cached output symbols; `concepts` reconstructs at the conceptual tier (default); `both` combines both reconstruction losses. (Default flipped from `symbols` to `concepts` on 2026-05-29; per-experiment XMLs no longer need to override.) |
| `subsymbolicOrder` | int | `1` | Iteration depth of the P$\to$C$\to$S pipeline. `order > 1` partitions the symbolic codebook geometrically; the partition width is set by `conceptualWidth`. |
| `processSymbols` | bool | `false` | Apply extra symbolic processing after Sigma. |
| `monotonic` | bool | `false` | Constrain invertible Sigma / Pi to $W \ge 0$ so the lift / lower chain is order-preserving on the parthood cone. |
| `ergodic` | bool | `false` | Ergodic exploration: every Layer uses `W_eff = bias * W + temp * noise`. See [Ergodic.md](Ergodic.md). |
| `naive` | bool | `false` | Materialise `W_eff` densely in `InvertibleLinearLayer`. Slower; debugging only. `false` uses sequential L / D / U triangular solves. |
| `symbolicOrder` | int | `0` | Forward-dispatch depth (replaced the `conceptualMode` enum 2026-06-13). `0` = parallel (whole-slab `[B, N, D]` forward; required for the butterfly cascade); `>= 1` = serial (per-word `[B, 1, D]`, loops the modules). Values > 1 plumbed but behave as 1. Default derives from the grammar (1 when `useGrammar != "none"`, else 0); class-level default on `BaseModel` is `0`. See [STM.md](STM.md#2-serial-sequencing). |
| `routerWireSerial` | string | `"both"` | Per-word router-fire gating on the serial path: `per-word` (fire per word, boundary off), `boundary` (fire only at the sentence boundary), `both` (default — both fire), `off` (neither). The per-word fire populates `symbolicSpace.current_rules` for SS dispatch. See [STM.md Section 7](STM.md#7-per-word-router-firing). |
| `learning` | bool | `false` | Two-pass soft-superposition training for the grammar chooser. When true, each TRAINING batch runs twice as two trials: pass A at superposition temperature 0 (sharp, recorded) and pass B at `exploreTemperature` (flatter exploration, trimmed from the batch error). The chooser is in the gradient path directly. Independent of `neuralToolUser`. Default off → one ordinary forward (byte-identical). See [Language.md → Soft-superposition route](Language.md). |
| `exploreTemperature` | decimal | `0.5` | Superposition temperature $t \in [0,1]$ for pass B of `learning`. `0` = the chooser's own (sharp) softmax, `1` = uniform (flat). Route scores are scaled by `1 - t`. |
| `transformChooser` | string | `"anchordot"` | Placement scorer for the structured grammar layers. `anchordot` = stateless cosine-to-anchor (byte-identical default, no new params); `mlp` = learned `MLPTransformChooser` (owns tool-embedding + MLP params → deliberate fresh-basin cutover). See [NeuralToolUser.md → MLP TransformChooser](old/NeuralToolUser.md). |
| `symbolicComposition` | bool | `false` | Cognitive op (2): at `subsymbolicOrder > 0`, re-feed the prior pass's symbolic carrier (`cs._subspaceForWS`) to PartSpace so Sigma composes higher-order symbols. Default off is byte-identical. See [Architecture.md](Architecture.md). |
| `neuralToolUser` | bool | `false` | Legacy hard-parse executor for the grammar's binary reduce stage (`parse_greedy` route + cross-product distributions). **Superseded by the soft-superposition route (`learning`); off the live path.** See [NeuralToolUser.md](old/NeuralToolUser.md). |
| `categoryCodebook` | bool | `false` | MetaSymbol participation-category codebook: a small `VectorQuantize` over the live MetaSymbol vectors, learned in perception (E/M in the autobind hook), keying each word's syntactic category to its frequency of participation across operator roles. Default off → no codebook allocated (byte-identical). See [Language.md → Participation Categories](Language.md). |
| `chooserCategoryContext` | bool | `false` | Phase 2 of `categoryCodebook`: thread the per-slot centroid role vector into `MLPTransformChooser` as syntactic-category context (widens its first `Linear`). Meaningful only with `transformChooser=mlp` + `categoryCodebook=true`. Default off → chooser ignores category (byte-identical). |
| `verbEigEdit` | bool | `false` | Verb sparse eigenvalue edit on `lift(NP, VP)`: the verb edits the NP through a sparse residual `x₂ = x₁ + p_class ⊙ δ_v`, masked by the NP's own eigen-signature (no learned mask), zero-init so untrained = the sigma fold. Default off → byte-identical. See [doc/old/semantic_verb_np_mask_eigenvalue_proposal.md](old/semantic_verb_np_mask_eigenvalue_proposal.md). |
| `mereologyRaise` | bool | `false` | Mereological order-raising: perception's autobind hook builds a meronymic lattice over the two towers and raises a higher-order PART when a whole accumulates more than `K_many` parts (abstraction order tracked via the ramsification table; provenance in `part_chain`). Enables the table on the PartSpace + terminal WholeSpace codebooks at build; `subsymbolicOrder` sets the max order. Default off → no table, no raising (byte-identical). See [doc/old/mereological-order-raising.md](old/mereological-order-raising.md). |
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
| `loadBalanceWeight` | float | `0.0` | Sparse-MoE load-balance loss weight. Active only when `SymbolicSpace.chartTopK > 0`. |
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
| `classificationMin` | float | -- | Lower bound for classification-accuracy reporting. |
| `classificationMax` | float | -- | Upper bound for classification-accuracy reporting. |

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
| `timeout` | int | `120` | Per-request timeout (s). |

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

The grammar section of `MentalModel.xml` declares `lift` as an S-tier
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
| `nDim` | int | `1` | Per-vector content dim. |
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
| `nDim` | int | `1` | Per-vector content dim. |
| `nVectors` | int | sentinel | Codebook size. When `> nOutput`, enables vector quantization with top-k selection of `nOutput` from `nVectors`. |
| `invertible` | bool | `false` | `true`: one `PiLayer(invertible=True)` handles both directions; `false`: separate `pi1` (forward) and `pi2` (reverse). |
| `hasAttention` | bool | `true` | Attention reweighting of percepts before the Pi fold. |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `synthesis` | string | `"lexicon"` | Bottom-up union strategy (renamed from `<chunking>`, Phase 4a — the legacy spelling is rejected loudly): `lexicon`, `bpe`, `mphf`, `byte` (canonical; `none` is its legacy alias), `radix`. `radix` routes the input lookup through `RadixLayer` (radix trie + inverse table + learned codebook + byte fallback; promotion `threshold=4, min_length=2` by default). `analyse` was REMOVED (Phase 4b): the meronymic analyzer is top-down ANALYSIS and lives on WholeSpace as `<analysis>analyse`. |
| `butterfly` | bool | `false` | FFT-style element-pair butterfly cascade on the PS fold (`self.sigma` post-swap) for cross-element mixing on the flattened `[B, N*D]` view. Required for `MM_xor.xml` convergence. See [doc/old/2026-05-26-two-loop-pi-sigma-substrate.md](old/2026-05-26-two-loop-pi-sigma-substrate.md). |
| `wordLearning` | int | `0` | Active lexicon-growth mode. `0` = frozen codebook (training default). `>=1` = on first sight of a new word, insert into the lexicon and tag as a part of "words" on the meronomy. Only `embed.py` sets this to `1` at lexicon-build time. (Was `chunkingFrequency` pre-2026-05-12.) |

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
| `nDim` | int | `1` | Per-vector content dim. |
| `nVectors` | int | sentinel | Codebook size. |
| `invertible` | bool | `false` | `true`: `SigmaLayer(invertible=True)` (exact inversion via `atanh` + $W^{-1}$); `false`: separate `sigma1` / `sigma2`. |
| `hasAttention` | bool | `false` | Attention in conceptual processing. Violates position locality required by `serial_mode`. |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. |
| `butterfly` | bool | `false` | Butterfly cascade on CS's PiLayer (same machinery as PartSpace). Required by `MM_xor.xml`. |
| `stmCapacity` | int | `8` | STM ring depth (within Miller's $7 \pm 2$ band; also the `wMax` fallback). Per-batch buffer `[B, stmCapacity, nDim]` + depth pointers `[B]`. |

Sigma layer math: $y_j = \tanh(W x + b)$. See [Architecture.md](Architecture.md).
`InvertibleSigmaLayer` has been merged into `SigmaLayer(invertible=True)`.

> 2026-05-29 (clean-stack STM): `ConceptualSpace.forward` no longer
> applies `sigma_in` / `sigma_cs` on the forward path. The Stage-10
> additive composition `sigma_in(combined) + sigma_cs(prev)` is
> replaced with per-stage tier attribution (`folded = primary` at
> stage 0, `folded = sym` at k > 0). The sigma layers remain
> allocated and are still invoked on the reverse path, but their
> parameters get no gradient on forward — they're dead weight under
> the current experiment. See
> [doc/old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

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
| `nDim` | int | `1` | Per-vector content dim. |
| `nVectors` | int | sentinel | Codebook size. When `codebook=quantize`, equals the number of symbol prototypes. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. Required for the full symbolic pipeline. **Pending removal** (backlog #13: SS `quantize` becomes hardwired) — deliberately deferred until the reverse decodes pre-snap $z$ (backlog #16 / the asymmetric recon gather), because `XOR_exact.xml` relies on `none` for its byte-exact chain. |
| `analysis` | string | `"byte"` | Top-down division strategy over the unity view (Phase 4b): `byte` = uniform contiguous regions (stage-0 region-mean pooling); `word` = whitespace-cut parts (part $k$ → symbol slot $k$, per-part coarse means); `analyse` = the meronymic analyzer front end (space-lexer cut today; learned-merge integration follows with the deeper SS analyzer work). |
| `lexer` | string | — | Intake granularity InputSpace executes (`word`, `sentence`, `byte`, `raw`). Moved here from `<InputSpace>` (Phase 4b: lexing is analytic cutting). |
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
| *(readout — retired 2026-06-19)* | — | — | The OutputSpace regression head (backlog #11): with `nVectors == 1` (or no codebook) it is an unquantised linear `nInputDim → nOutputDim` plus a learned scalar intercept. The former `<readout>` enum (`identity` \| `sigmoid`) was retired — the head is always linear+bias (a binary $\{0,1\}$ target like XOR is regressed, not squashed; see `OutputSpace.lrScale` + `subsymbolicOrder=3`). Ignored by quantised heads. |

`LinearLayer` with `(bias, temp)` support for ergodic mode.

---

## `<SymbolicSpace>`

Grammar infrastructure (SyntacticLayers, TruthLayer). Lives outside
`<architecture>` as a top-level sibling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `syntacticHiddenDim` | int | `64` | SyntacticLayer MLP hidden size. |
| `truthMaxEntries` | int | `1024` | TruthLayer SS-codebook capacity. |
| `ltmCapacity` | int | `1024` | Long-term-memory (LTM) chain capacity on `InterSentenceLayer`: the per-row bounded `deque` of STM end-states (the AR sequence feeding inter-sentence prediction) is capped at this many entries. Separate from `truthMaxEntries`. See [STM.md Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states). |
| `parserBackend` | string | `"chart"` | Parser backend: `chart`, `stm`, or `parallel`. STM requires an attached knowledge artifact. |
| `routerKind` | string | `"chart"` | Chart-backend router: `chart` or `signal`. |
| `chartCollapse` | string | `"root"` | Soft-superposition chart output: `root` (single root sentence node, other positions zero) or `broadcast` (replicate root across all positions). |
| `chartTau` | float | `1.0` | Temperature for the chart rule-score softmax. |
| `wMax` | int | `8` | Max span width considered by the parser chart path and default STM capacity fallback. |
| `chartTopK` | int | `0` | Sparse-MoE rule gating: keep top-K rules per (cell, split). `0` disables. |
| `chartNoiseEps` | float | `0.0` | Gaussian noise epsilon for `chartTopK` gating. |
| `iterationsPerWord` | int | `1` | P->C iterations per word in the per-word stem path. |
| `downwardGeneration` | bool | `false` | Emit a codebook head via `SymbolicSpace.reconstruct()` and stash on `self._predicted_head`. |
| `language.interpretation` | float | `0.5` | Soft-interpolation weight between forward and generation directions. |

The `<language><grammar>` block contains tier-scoped rules in
`<compose>` and `<generate>` sections, each holding `<percepts>` /
`<concepts>` / `<symbols>` sub-blocks of `<rule>` elements. Per the
2026-05-07 rollback, the grammar XML is the sole source of truth for
which rule fires at each tier (no code-level `default_rule` fallback).
Legacy bare-tier elements like `<P>P = sigma(P)</P>` directly under
`<grammar>` are still accepted.

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
    <codebook>quantize</codebook>
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
    <data><dataType>embedding</dataType></data>
    <weightsPath>BasicModel.ckpt</weightsPath>
    <embeddingPath>BasicModel.kv</embeddingPath>

    <data>
      <shardDir>data/fineweb</shardDir>
      <minFrequency>0.00001</minFrequency>
    </data>

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
