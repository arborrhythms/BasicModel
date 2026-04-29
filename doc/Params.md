# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`). Default values are loaded from `data/model.xml` and overridden by model-specific configs. The schema is defined in `data/model.xsd`.

---

## `<architecture>`

Core model settings. Training and data parameters are in nested sub-elements `<training>` and `<data>` (see below).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelType` | string | `"simple"` | Model type: `"simple"` (feedforward), `"lm"` (language model with sequence processing), `"embedding"` (CBOW/SBOW embedding trainer). |
| `ergodic` | bool | `false` | Enable ergodic exploration mode. Layers use `W_eff = bias * W + temp * noise` where bias/temp are derived from alpha. See Architecture.md for the gradient energy sensor. |
| `certainty` | bool | `false` | Enable per-neuron certainty tracking in ergodic layers. Allows individual neurons to transition from exploration to exploitation at different rates. |
| `reshape` | bool | `false` | Reshape input data for sequence processing. Required for `modelType=lm`. |
| `conceptualOrder` | int | `1` | Order of conceptual processing. Controls how many higher-order conceptual transforms are applied. |
| `processSymbols` | bool | `false` | Enable additional symbolic processing steps. |
| `maskedPrediction` | string | `"NONE"` | Masked prediction mode: `NONE`, `MLM`, `ARLM`, `ARUS`, `RARLM`. See [Training.md](Training.md). |
| `reconstruct` | string | `"NONE"` | Controls the reverse (reconstruction) pass. `"NONE"` disables reconstruction entirely. `"symbols"` reconstructs from cached output symbols (most common). `"output"` runs `outputSpace.reverse()` only. `"both"` combines reversed output with reconstruction symbols. Any non-`NONE` value enables the full bidirectional training pipeline. |
| `maxResponseLength` | int | `64` | Maximum number of characters/tokens to generate during inference. Measured in characters for uniformity with `InputSpace.nActive`. Both this value and `InputSpace.nActive` cap generation: truncation occurs when the expanded token sequence (OOV words spelled out as characters) exceeds `nActive`. |

---

### `<architecture><data>`

Data loading and filtering settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | string | `"xor"` | Dataset to load. Determines input/output data via `TheData.load()`. Common values: `"xor"`, `"mnist"`, `"text"`. |
| `minFrequency` | float | `0.0` | Minimum word frequency ratio for vocabulary admission. Words below this threshold are held in a pending buffer until they accumulate enough occurrences. `0.0` admits all words immediately. |
| `shardDir` | string | -- | Directory containing text shards for streaming datasets (e.g. `"data/fineweb"`). |
| `numShards` | int | `1` | Number of shards to load from `shardDir`. |
| `maxDocs` | int | `10000` | Maximum number of documents to load per shard. |
| `classificationMin` | float | -- | Minimum threshold for classification accuracy reporting. |
| `classificationMax` | float | -- | Maximum threshold for classification accuracy reporting. |

---

### `<architecture><training>`

Training loop and I/O settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numTrials` | int | `1` | Number of independent training runs. |
| `numEpochs` | int | `3` | Number of training epochs per trial. |
| `batchSize` | int | `10` | Number of contiguous streams through the dataset. Each batch row `b` receives the next item from stream `b`, so temporal context is coherent across steps. Capped at split length, so small eval sets yield one rectangular batch. |
| `numWorkers` | int | `0` | DataLoader prefetch workers. `0` means synchronous in-process batch assembly. |
| `learningRate` | float | `0.001` | Learning rate for the Adam optimizer. |
| `reconRatio` | float | `0.5` | Weight of reconstruction loss in combined loss: `total = (1-r)*output + r*recon`. |
| `trainEmbedding` | string | `"NONE"` | Embedding update mode. Controls both the embedding training method and whether gradients flow through the codebook. See table below and [Training.md](Training.md). |
| `trainEmbeddingRatio` | float | `0.1` | Weight $\lambda$ of embedding loss in JOINT mode: $\mathcal{L} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{emb}}$. Ignored by other modes. |
| `weightsPath` | string | `"output/BasicModel.ckpt"` | File path for saving/loading model weights checkpoint. Filename conventionally matches the XML config (e.g. `BasicModel.xml` $\rightarrow$ `output/BasicModel.ckpt`). |
| `embeddingPath` | string | -- | File path for the word vector store (`.kv` extension, gensim-compatible `KeyedVectors`). When absent, embedding training is skipped. |
| `autoload` | bool | `true` | Automatically load weights from `weightsPath` on model creation. Set to `false` for fresh training. |
| `autosave` | bool | `false` | Automatically save weights after training completes. |
| `checkpointEveryBatches` | int | `0` | Save in-progress weights and embeddings every N optimizer steps. `0` disables periodic checkpoints. Training exceptions also write an `.emergency.ckpt` when `autosave` or periodic checkpoints are enabled. |
| `negSamples` | int | `64` | Number of negative samples per positive example for CBOW/SBOW training. Controls memory usage: `O(batch * negSamples * dim)` vs `O(batch * vocab)` for full softmax. |

#### `<trainEmbedding>` -- Embedding Update Modes

| Value | Embedding method | Model layers | Gradients through codebook | Description |
|-------|-----------------|-------------|---------------------------|-------------|
| `NONE` | Frozen | Trained | No | Embeddings frozen; only model layers train |
| `CBOW` | CBOW (padded context) | Trained | No | CBOW reshapes embeddings via own optimizer; model layers train separately |
| `SBOW` | SBOW (centroid) | Trained | No | Faster variant: predict each word from leave-one-out centroid |
| `BACKPROP` | Backprop only | Trained | Yes | Codebook trained purely via model loss gradients; no SBOW/CBOW |
| `BOTH` | SBOW post-batch | Trained | Yes | Two optimizers: SBOW updates embeddings, Adam updates model layers |
| `JOINT` | Single backward | Trained | Yes | Single optimizer: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$ |

#### `<trainEmbeddingRatio>` -- Embedding Loss Weight (JOINT mode)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainEmbeddingRatio` | float | `0.1` | Weight $\lambda$ of the embedding co-occurrence loss in JOINT mode. Higher values bias training toward embedding quality; lower values prioritize model prediction. |

---

## MentalModel Configuration

The basicModel's `MentalModel.xml` defines the neural architecture.
Three parameters control the TruthLayer integration (truth bias during
inference and loss modification during training). All are optional;
omitting them defaults to 0.1.

| Parameter            | Type    | Default | Location         | Description                                                        |
|----------------------|---------|---------|------------------|--------------------------------------------------------------------|
| `truthBiasScale`     | decimal | `0.1`   | `<architecture>` | Strength of luminosity-based bias on concept input during forward. |
| `LuminosityWeight`   | decimal | `0.1`   | `<architecture>` | Loss penalty weight for low luminosity (incoherent truths).        |
| `UniversalityWeight` | decimal | `0.1`   | `<architecture>` | Loss penalty weight for low universality (unkind propositions).    |
| `TruthLoss`          | decimal | `0.0`   | `<training>`     | Additive loss penalty for propositions that contradict the TruthSet. Uses union norm reduction via `Basis.disjunction()`. 0.0 = disabled. |
| `conceptualOrder`    | int     | `1`     | `<architecture>` | Number of Percept$\rightarrow$Concept$\rightarrow$Symbol iterations. Higher orders use a geometrically partitioned symbolic space. |
| `useButterflies`     | boolean | `false` | `<architecture>` | Enable pairwise sigma/pi mixing via butterfly-mode Pi/Sigma layers (N-halving per conceptual order). Mutually exclusive with `useGrammar`. |
| `monotonic`          | boolean | `false` | `<architecture>` | When true, invertible SigmaLayers use W>=0 (NonNegativeInvertibleLinearLayer) preserving ordering; false uses unconstrained InvertibleLinearLayer (bitonic response). |
| `useGrammar`         | string | `"none"` | `<WordSpace>`    | Grammar mode. Current parser accepts `none`, `all`, and legacy `thoughtFree`; target Shamatha Speech work adds/aliases `shamathaSpeech` for the narrow DNF object grammar. Full grammar mode is mutually exclusive with `useButterflies`. |

```xml
<architecture>
  <truthBiasScale>0.1</truthBiasScale>
  <LuminosityWeight>0.1</LuminosityWeight>
  <UniversalityWeight>0.1</UniversalityWeight>
  <conceptualOrder>1</conceptualOrder>
  <useButterflies>false</useButterflies>
  <monotonic>false</monotonic>
</architecture>

<training>
  <TruthLoss>0.0</TruthLoss>
</training>
```

Shamatha Speech should not be confused with serial mode.  Serial mode is
the rolling cursor / next-token runtime path.  Shamatha Speech is a
grammar policy: it may compose over all active percepts, but every
logical merge must preserve connected `where()` support and continuous
`when()` support so the resulting DNF denotes one object.

Luminosity and universality are **multiplicative** -- they scale the total loss by
`(1 + LuminosityWeight*(1-luminosity) + UniversalityWeight*(1-universality))`.
TruthLoss is **additive** -- it directly penalizes specific propositions that
contradict stored truths, measured by union norm reduction. Both coexist.
See [Ethics.md](../../doc/Ethics.md) Section Universality and
[Reasoning.md](./Reasoning.md) Section TruthLoss for details.

The grammar section of `MentalModel.xml` declares `lift` as an S-tier
binary rule:

```xml
<S>lift(S, S)</S>      <!-- lift is binary; SVO is assembled by
                            the chart-compose path S -> S VO,
                            VO -> V O, not a ternary lift -->
```

Transitive SVO is produced compositionally via the typed chart rules
(`S -> S VO` / `VO -> V O`) when `<chartCompose>true</chartCompose>`;
see [Language.md](./Language.md) and
[Ethics.md](../../doc/Ethics.md) Section Universality for the
architectural details.

---

## `<InputSpace>`

The entry point that lifts raw data into the model's internal representation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Sequence length: maximum number of tokens per input. For XOR: 2 (two binary inputs). For text models, OOV words are spelled out as individual characters, so a single word may consume multiple slots -- a short sentence with many OOV words can exceed this limit and trigger a truncation warning suggesting a lower `<minFrequency>`. |
| `nDim` | int | `1` | Dimensionality of each input vector. Set on TheObjectEncoding via XML; not passed to the Space constructor. |
| `nVectors` | int | = `nActive` | Codebook size (total vectors in the space). Defaults to `nActive` for InputSpace. |
| `nWhere` | int | `0` | Number of spatial/positional dimensions appended to each vector. When > 0, enables PositionalEncoding on all objects throughout the model. |
| `nWhen` | int | `0` | Number of temporal dimensions appended to each vector. When > 0, enables TemporalEncoding on all objects throughout the model. |
| `codebook` | bool | `false` | Whether input values use codebook vector quantization. |
| `lexer` | string | `"word"` | Tokenization mode for text inputs. Common values in the current configs are `"word"` and `"sentence"`. |

**Note:** InputSpace's `inputShape` uses the data's native dimension (e.g. 784 for MNIST, `inputLength` for text), while `outputShape` uses `nDim` from TheObjectEncoding. The LiftingLayer bridges the two.

**Layers:** Contains a `LiftingLayer` that projects raw input into the model's working dimensionality.

---

## `<PerceptualSpace>`

Transforms lifted input into perceptual features via Pi layers (multiplicative interactions).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Number of active perceptual feature vectors (sparse subset selected in forward pass). Should be >= InputSpace nActive. For XOR: 4. |
| `nDim` | int | `1` | Dimensionality of each perceptual vector. Set on TheObjectEncoding; not passed to the Space constructor. |
| `nVectors` | int | `0` | Codebook size (total vectors in the space). When > 0, enables vector quantization with topk selection of `nActive` from `nVectors`. |
| `invertible` | bool | `false` | Use a single invertible Pi layer instead of separate forward/reverse layers. When `true`, one `PiLayer(invertible=True)` handles both directions via `InvertibleLinearLayer` internally. When `false`, separate `pi1` (forward) and `pi2` (reverse) layers are created. |
| `hasAttention` | bool | `true` | Enable attention mechanism in this space. |
| `passThrough` | bool | `false` | Skip perceptual processing entirely; pass input through unchanged. |

**Layers:** Pi layers -- multiplicative: `y_j = b_j * prod_i(1 + W_ji * x_i)`. See Architecture.md.

**Note:** `InvertiblePiLayer` has been merged into `PiLayer(invertible=True)`; the standalone class is removed.

---

## `<ConceptualSpace>`

Transforms perceptual features into abstract concepts via Sigma layers (additive/linear).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Number of active concept vectors (sparse subset). For XOR: 3. |
| `nDim` | int | `1` | Dimensionality of each concept vector. Set on TheObjectEncoding; not passed to the Space constructor. |
| `nVectors` | int | `0` | Codebook size (total vectors in the space). |
| `invertible` | bool | `false` | Use single invertible Sigma layer vs. separate forward/reverse layers (`sigma1`/`sigma2`). When `true`, routes to `SigmaLayer(invertible=True)` which uses `InvertibleLinearLayer` internally for exact inversion via atanh + $W^{-1}$. |
| `hasAttention` | bool | `false` | Enable attention in conceptual processing. |
| `hasNorm` | bool | `false` | Enable layer normalization in this space. |

**Layers:** Sigma layers -- additive: `y_j = tanh(W x + b)`. See Architecture.md.

**Note:** `InvertibleSigmaLayer` has been merged into `SigmaLayer(invertible=True)`; the standalone class is removed.

---

## `<SymbolicSpace>`

Discrete symbolic representation -- the information bottleneck between perception and output.
Maps concept activations to a one-hot encoding over a codebook of symbol prototypes.
See [Language.md](Language.md) for the full design.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Number of active symbols. When reconstruction symbols are enabled, `nOutputSymbols = OutputSpace.nActive` are fed to output, and the rest carry reconstruction information. |
| `nDim` | int | `1` | Dimensionality of each symbol (typically 1 -- symbols are scalar activations). |
| `nVectors` | int | = `nActive` | Codebook size. When `codebook=true`, equals the number of symbol prototypes. |
| `passThrough` | bool | `false` | Pass concepts through as symbols unchanged. Typically `true` for simple models. |
| `codebook` | bool | `false` | Enable codebook quantization. When `true`, the forward path produces a one-hot activation over codebook entries. Required for the full symbolic pipeline. |

**Layers:** SymbolicSpace owns one
`SigmaLayer(nConcepts, nSymbols, invertible=True, monotonic=monotonic)`
at `self.sigma` that bridges the C$\leftrightarrow$S boundary in both directions via
its own self-inverse (`forwardSigma` / `reverseSigma` pointer aliases
hide the one-or-two-layer split).  The legacy `SymbolicSpace.layer`
PiLayer attribute is gone; consumers use `model.symbolicSpace.sigma`
directly. Codebook provides dense vectors when codebook=true.

---

## `<SyntacticSpace>`

Generates binary derivation trees (deep structure) from active symbols using a
Chomsky Normal Form grammar. See [Language.md](Language.md) for the grammar and
open implementation questions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | `16` | Number of active word vectors. |
| `nDim` | int | `1` | Dimensionality of each word vector. |
| `nVectors` | int | = `nActive` | Codebook size (total vectors in the space). |

**Layers:** No trainable parameters currently. Rule selection is random; derivation
is stored as word tuples `(batch, vector, rule)` on the output subspace.

---

## `<OutputSpace>`

Maps symbols to final predictions via linear layers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Number of active output values. For XOR: 1 (single binary output). |
| `nDim` | int | `1` | Dimensionality of each output. Set on TheObjectEncoding; not passed to the Space constructor. |
| `nVectors` | int | = `nActive` | Codebook size (total vectors in the space). Defaults to `nActive` for OutputSpace. |

**Layers:** Linear layers with `(bias, temp)` support for ergodic mode.

---

## `InvertibleLinearLayer`

The LDU-factorised invertible linear primitive used internally by
`SigmaLayer(invertible=True)` and `PiLayer(invertible=True)`. These parameters are not
set via XML directly; they are configured programmatically when constructing the layer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `naive` | bool | `False` | `False`: apply L, D, U sequentially via triangular solves -- no W materialisation, backprop through each factor separately. `True`: materialise `W_eff` as a dense matrix and materialise the dense LDU inverse for the reverse pass. The `naive=True` path is slower and uses more memory; it exists for debugging and validation. |
| `stable` | bool | `False` | Clamps each diagonal entry `d_i` to magnitude `[eps, 1]` with sign preserved in `_d_effective()` before the ergodic blend. Keeps `W_eff` bounded away from singularity. This is the only stability constraint on d; no additional clamp is applied to `d_eff`. |
| `ergodic` | bool | `False` | Enables factor-level noise injection. When `True`, registers noise buffers `noise_raw_L`, `noise_raw_U`, `noise_d` and zero-initialises the learned parameters. Noise is resampled at the start of each `forward()` and at the end of each `reverse()`. |

**Class renames and removals.**

| Old name | New name / status |
|----------|------------------|
| `LULayer` | Renamed to `InvertibleLinearLayer` |
| `InvertibleLinearLayer` (SVD-based) | Removed; replaced by LDU factorisation |
| `InvertibleSigmaLayer` | Merged into `SigmaLayer(invertible=True)` |
| `InvertiblePiLayer` | Merged into `PiLayer(invertible=True)` |

See [Architecture.md](Architecture.md) for the full mathematical treatment of the LDU
factorisation and factor-level noise injection.

---

## `<architecture><server>`

HTTP server settings (used by `serve.py`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Server bind address. |
| `port` | int | `8001` | Server port. |
| `timeout` | int | `120` | Request timeout in seconds. |

---

## Example: XOR Configuration

```xml
<model>
  <architecture>
    <reconstruct>symbols</reconstruct>
    <reshape>true</reshape>
    <modelType>lm</modelType>
    <ergodic>false</ergodic>

    <data>
      <dataset>xor</dataset>
    </data>

    <training>
      <numEpochs>1000</numEpochs>
      <learningRate>0.01</learningRate>
      <autoload>false</autoload>
      <autosave>false</autosave>
    </training>
  </architecture>

  <InputSpace>
    <nActive>2</nActive>
    <nDim>1</nDim>
  </InputSpace>

  <PerceptualSpace>
    <nActive>4</nActive>
    <nDim>1</nDim>
    <nVectors>16</nVectors>
    <invertible>false</invertible>
    <hasAttention>false</hasAttention>
  </PerceptualSpace>

  <ConceptualSpace>
    <nActive>3</nActive>
    <nDim>1</nDim>
    <nVectors>16</nVectors>
    <invertible>false</invertible>
  </ConceptualSpace>

  <SymbolicSpace>
    <nActive>3</nActive>
    <passThrough>true</passThrough>
  </SymbolicSpace>

  <OutputSpace>
    <nActive>1</nActive>
    <nDim>1</nDim>
  </OutputSpace>
</model>
```

In this configuration:
- 2 binary inputs (`nActive=2`) are lifted, transformed through 4 perceptual and 3 conceptual active features
- PerceptualSpace and ConceptualSpace each have a codebook of 16 vectors (`nVectors=16`), selecting `nActive` via topk
- 3 symbols are produced: 1 for output prediction, 2 for reconstruction
- The reverse pass reconstructs the original 2 inputs from all 3 symbols
- Ergodic mode is off; training uses standard Adam with combined forward+reverse loss

---

## Example: Embedding / Chatbot Configuration

```xml
<model>
  <architecture>
    <reconstruct>symbols</reconstruct>
    <modelType>embedding</modelType>
    <maskedPrediction>ARLM</maskedPrediction>

    <data>
      <dataset>text</dataset>
      <shardDir>data/fineweb</shardDir>
      <numShards>1</numShards>
      <maxDocs>10000</maxDocs>
      <minFrequency>0.00001</minFrequency>
    </data>

    <training>
      <numEpochs>1</numEpochs>
      <batchSize>1</batchSize>
      <learningRate>0.001</learningRate>
      <trainEmbedding>BACKPROP</trainEmbedding>
      <weightsPath>BasicModel.ckpt</weightsPath>
      <embeddingPath>BasicModel.kv</embeddingPath>
      <autoload>true</autoload>
      <autosave>true</autosave>
      <negSamples>64</negSamples>
    </training>
  </architecture>

  <InputSpace>
    <nActive>128</nActive>
    <nDim>100</nDim>
    <nWhere>2</nWhere>
    <nWhen>2</nWhen>
    <lexer>sentence</lexer>
    <codebook>true</codebook>
  </InputSpace>

  <!-- ... space definitions ... -->
</model>
```

Key points:
- `modelType=embedding` activates the embedding-based input pipeline alongside the neural model
- `trainEmbedding=BACKPROP` trains model layers with gradients flowing through the codebook; no separate SBOW/CBOW step
- The three files partition model behaviour: **XML config** (architecture), **`.kv` embedding** (codebook), **`.ckpt` weights** (model layers)
- `weightsPath` stores the neural model checkpoint; `embeddingPath` stores the word vectors
- `minFrequency` gates vocabulary admission: words are buffered until their frequency ratio exceeds this threshold
