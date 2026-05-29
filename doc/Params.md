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
> InputSpace / PerceptualSpace / SymbolicSpace, `0` elsewhere). The
> `.where` field is now the canonical positional identifier (quadrature
> sinusoidal); see [doc/plans/2026-05-28-where-keyed-taxonomy.md](plans/2026-05-28-where-keyed-taxonomy.md).

---

## `<architecture>`

Core model settings. Training and data parameters are in nested
sub-elements `<training>` and `<data>` (see below).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelType` | string | `"simple"` | `simple` (feedforward), `embedding` (LM / chat with sequence processing), `passthrough` (identity bench), `vq` (codebook-only). |
| `reconstruct` | string | `"concepts"` | Reverse-pass mode: `none` disables; `symbols` reconstructs from cached output symbols; `concepts` reconstructs at the conceptual tier (default); `both` combines both reconstruction losses. (Default flipped from `symbols` to `concepts` on 2026-05-29; per-experiment XMLs no longer need to override.) |
| `conceptualOrder` | int | `1` | Iteration depth of the P$\to$C$\to$S pipeline. `order > 1` partitions the symbolic codebook geometrically; the partition width is set by `conceptualWidth`. |
| `processSymbols` | bool | `false` | Apply extra symbolic processing after Sigma. |
| `monotonic` | bool | `false` | Constrain invertible Sigma / Pi to $W \ge 0$ so the lift / lower chain is order-preserving on the parthood cone. |
| `ergodic` | bool | `false` | Ergodic exploration: every Layer uses `W_eff = bias * W + temp * noise`. See [Ergodic.md](Ergodic.md). |
| `naive` | bool | `false` | Materialise `W_eff` densely in `InvertibleLinearLayer`. Slower; debugging only. `false` uses sequential L / D / U triangular solves. |
| `conceptualMode` | string | `"parallel"` | `parallel` (whole-slab `[B, N, D]` forward; required for the butterfly cascade), `serial` (per-word `[B, 1, D]`). The class-level default on `BaseModel` is `parallel` (2026-05-29). |
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
| `loadBalanceWeight` | float | `0.0` | Sparse-MoE load-balance loss weight. Active only when `WordSpace.chartTopK > 0`. |
| `l1Lambda` | decimal | `0.0` | Architecture-wide L1 penalty hook. Most configs leave this at `0` and use `SymbolicSpace.l1Lambda` instead. |
| `discontinuityLambda` | decimal | `0.0` | Architecture-wide discontinuity penalty hook (legacy). |
| `conceptualWidth` | string | `"tapered"` | Symbol-partition geometry for `conceptualOrder > 1`. `tapered` = geometrically narrowing slices; `uniform` = equal-width slices. |
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
  <conceptualOrder>1</conceptualOrder>
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
`S = lift(NP, VP)` over `VP = intersection(V, O)`. Grammar mode is
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
| `lexer` | string | `"word"` | Tokenization mode: `word`, `sentence`, `byte`. |

`inputShape` uses the data's native dim (e.g. 784 for MNIST,
`inputLength` for text); `outputShape` uses `nDim` from
`TheObjectEncoding`. `LiftingLayer` bridges the two.

### `<PerceptualSpace>`

Transforms lifted input into perceptual features via Pi layers
(multiplicative interactions).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Number of active perceptual feature vectors. Should be $\geq$ `InputSpace.nOutput`. For XOR: 4. |
| `nDim` | int | `1` | Per-vector content dim. |
| `nVectors` | int | sentinel | Codebook size. When `> nOutput`, enables vector quantization with top-k selection of `nOutput` from `nVectors`. |
| `invertible` | bool | `false` | `true`: one `PiLayer(invertible=True)` handles both directions; `false`: separate `pi1` (forward) and `pi2` (reverse). |
| `hasAttention` | bool | `true` | Attention reweighting of percepts before the Pi fold. |
| `nonlinear` | bool | `true` | Tanh-bound output to $[-1, 1]$. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. Legacy `true`/`false` are accepted. Set to `none` to make PS a pure input-encoder (the butterfly cascade carries the trainable transform); set to `quantize` for VQ-EMA snapping. `MM_xor.xml` uses `none` (the snap kills gradient flow into the butterfly weights). |
| `chunking` | string | `"lexicon"` | Multi-token chunking strategy: `lexicon`, `bpe`, `radix`. `radix` routes the input lookup through `RadixLayer` (radix trie + inverse table + learned codebook + byte fallback; promotion `threshold=4, min_length=2` by default). `RadixLayer` is in `bin/Layers.py` (relocated 2026-05-29 from `bin/PerceptStore.py`). |
| `butterfly` | bool | `false` | FFT-style element-pair butterfly cascade on `self.pi` (and `self.sigma` historically) for cross-element mixing on the flattened `[B, N*D]` view. Required for `MM_xor.xml` convergence. See [doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md](plans/2026-05-26-two-loop-pi-sigma-substrate.md). |
| `wordLearning` | int | `0` | Active lexicon-growth mode. `0` = frozen codebook (training default). `>=1` = on first sight of a new word, insert into the lexicon and tag as a part of "words" on the meronomy. Only `embed.py` sets this to `1` at lexicon-build time. (Was `chunkingFrequency` pre-2026-05-12.) |

Pi layer math: $y_j = b_j \prod_i (1 + W_{ji} x_i)$. See [Architecture.md](Architecture.md).
`InvertiblePiLayer` has been merged into `PiLayer(invertible=True)`.

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
| `butterfly` | bool | `false` | Butterfly cascade on CS's PiLayer (same machinery as PerceptualSpace). Required by `MM_xor.xml`. |
| `stmCapacity` | int | `7` | STM ring depth (Miller, 7±2). Per-batch buffer `[B, stmCapacity, nDim]` + depth pointers `[B]`. |

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
> [doc/plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

### `<SymbolicSpace>`

Discrete symbolic representation --- the information bottleneck between
perception and output. See [Language.md](Language.md) for the design.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nOutput` | int | sentinel | Total symbols. When reconstruction symbols are enabled, `nOutputSymbols = OutputSpace.nOutput` are fed to output; the rest carry reconstruction information. |
| `nDim` | int | `1` | Per-vector content dim. |
| `nVectors` | int | sentinel | Codebook size. When `codebook=quantize`, equals the number of symbol prototypes. |
| `codebook` | mode | `none` | `none`, `quantize`, or `project`. Required for the full symbolic pipeline. |
| `nonlinear` | bool | `true` | Tanh-bound the activation. |
| `sortNetwork` | bool | `false` | Bitonic-sort regularizer on the codebook ordering. |
| `sortPasses` | int | `0` | Sort-network depth. `0` disables. |
| `l1Lambda` | decimal | `0.01` | L1 sparsity penalty on the symbolic activation. |
| `symbolResidualScale` | decimal | `1.0` | Symbol-residual injection scale (legacy mereology hook). |
| `outputSymbolResidualScale` | decimal | `0.0` | Output-side symbol-residual injection scale. |
| `commitmentBeta` | float | `0.25` | VQ-VAE commitment-loss $\beta$. Active whenever `<codebook>quantize</codebook>` is set (2026-05-29: the prior `<useVQVAE>` toggle was retired; codebook training is implicit). |
| `decorrelationWeight` | decimal | `0.0` | `ImpenetrableLayer` decorrelation regularizer on codebook rows. |
| `spectralFlatnessWeight` | decimal | `0.0` | `ImpenetrableLayer` spectral-flatness regularizer. |
| `truthMinMagnitude` | decimal | `0.3` | Minimum magnitude before a SymbolicSpace activation contributes to TruthLayer commit. |

SymbolicSpace owns one
`SigmaLayer(nConcepts, nSymbols, invertible=True, monotonic=monotonic)`
at `self.sigma` bridging the C $\leftrightarrow$ S boundary in both directions via its
own self-inverse (`forwardSigma` / `reverseSigma` pointer aliases hide
the one-or-two-layer split). The legacy `SymbolicSpace.layer` PiLayer
attribute is gone; consumers use `model.symbolicSpace.sigma` directly.

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

`LinearLayer` with `(bias, temp)` support for ergodic mode.

---

## `<WordSpace>`

Grammar infrastructure (SyntacticLayers, TruthLayer). Lives outside
`<architecture>` as a top-level sibling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `syntacticHiddenDim` | int | `64` | SyntacticLayer MLP hidden size. |
| `truthMaxEntries` | int | `1024` | TruthLayer LTM capacity. |
| `parserBackend` | string | `"chart"` | Parser backend: `chart`, `stm`, or `parallel`. STM requires an attached knowledge artifact. |
| `routerKind` | string | `"chart"` | Chart-backend router: `chart` or `signal`. |
| `chartCollapse` | string | `"root"` | Soft-superposition chart output: `root` (single root sentence node, other positions zero) or `broadcast` (replicate root across all positions). |
| `chartTau` | float | `1.0` | Temperature for the chart rule-score softmax. |
| `wMax` | int | `8` | Max span width considered by the parser chart path and default STM capacity fallback. |
| `chartTopK` | int | `0` | Sparse-MoE rule gating: keep top-K rules per (cell, split). `0` disables. |
| `chartNoiseEps` | float | `0.0` | Gaussian noise epsilon for `chartTopK` gating. |
| `iterationsPerWord` | int | `1` | P->C iterations per word in the per-word stem path. |
| `downwardGeneration` | bool | `false` | Emit a codebook head via `WordSpace.reconstruct()` and stash on `self._predicted_head`. |
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
    <modelType>embedding</modelType>

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

  <PerceptualSpace>
    <nInput>8</nInput>
    <nVectors>1000</nVectors>
    <nDim>10</nDim>
    <nOutput>8</nOutput>
    <hasAttention>false</hasAttention>
    <invertible>true</invertible>
    <codebook>quantize</codebook>
  </PerceptualSpace>

  <ConceptualSpace>
    <nInput>8</nInput>
    <nVectors>8</nVectors>
    <nDim>10</nDim>
    <nOutput>8</nOutput>
    <invertible>true</invertible>
    <codebook>quantize</codebook>
  </ConceptualSpace>

  <SymbolicSpace>
    <nInput>8</nInput>
    <nVectors>8</nVectors>
    <nDim>2</nDim>
    <nOutput>8</nOutput>
    <codebook>quantize</codebook>
  </SymbolicSpace>

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
    <modelType>embedding</modelType>
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

- `modelType=embedding` activates the embedding-based input pipeline alongside the neural model.
- `trainEmbedding=BACKPROP` trains model layers with gradients flowing through the codebook.
- XML config defines the architecture. `weightsPath` stores the neural model checkpoint.
- `minFrequency` gates vocabulary admission; words below it are buffered until they exceed the threshold.
