# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`). Default values are loaded from `data/defaults.xml` and overridden by model-specific configs.

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
| `symbolicOrder` | int | `1` | Order of symbolic processing. Controls how many higher-order symbolic transforms are applied. |
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
| `shardDir` | string | — | Directory containing text shards for streaming datasets (e.g. `"data/fineweb"`). |
| `numShards` | int | `1` | Number of shards to load from `shardDir`. |
| `maxDocs` | int | `10000` | Maximum number of documents to load per shard. |
| `classificationMin` | float | — | Minimum threshold for classification accuracy reporting. |
| `classificationMax` | float | — | Maximum threshold for classification accuracy reporting. |

---

### `<architecture><training>`

Training loop and I/O settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numTrials` | int | `1` | Number of independent training runs. |
| `numEpochs` | int | `3` | Number of training epochs per trial. |
| `batchSize` | int | `10` | Mini-batch size for training. |
| `learningRate` | float | `0.001` | Learning rate for the Adam optimizer. |
| `reconRatio` | float | `0.5` | Weight of reconstruction loss in combined loss: `total = (1-r)*output + r*recon`. |
| `train` | string | `"NONE"` | Embedding update mode. See table below and [Training.md](Training.md). |
| `OptimizeEmbedding` | bool | *(see below)* | Whether embedding weights participate in the main optimizer graph (backprop through codebook lookups). When `false`, codebook vectors are detached during forward/reverse — no gradients flow through them. Default: `true` for `ARLM`/`BOTH`/`JOINT`, `false` otherwise. |
| `sbowRatio` | float | `0.1` | Weight $\lambda$ of SBOW loss in JOINT mode: $\mathcal{L} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$. Ignored by other modes. |
| `weightsPath` | string | `"output/BasicModel.ckpt"` | File path for saving/loading model weights checkpoint. Filename conventionally matches the XML config (e.g. `BasicModel.xml` → `output/BasicModel.ckpt`). |
| `embeddingPath` | string | — | File path for the word vector store (`.kv` extension, gensim-compatible `KeyedVectors`). When absent, embedding training is skipped. |
| `autoload` | bool | `true` | Automatically load weights from `weightsPath` on model creation. Set to `false` for fresh training. |
| `autosave` | bool | `false` | Automatically save weights after training completes. |
| `negSamples` | int | `64` | Number of negative samples per positive example for CBOW/SBOW training. Controls memory usage: `O(batch × negSamples × dim)` vs `O(batch × vocab)` for full softmax. |

#### `<train>` — Embedding Update Modes

| Value | Embedding method | Model layers | Description |
|-------|-----------------|-------------|-------------|
| `NONE` | Frozen | Frozen | Inference only; no parameters updated |
| `CBOW` | CBOW (padded context) | Frozen | True CBOW: predict each word from leave-one-out context with padding |
| `SBOW` | SBOW (centroid) | Frozen | Faster variant: predict each word from leave-one-out centroid |
| `ARLM` | Frozen | Backprop | Train model layers only; codebook is fixed |
| `BOTH` | SBOW post-batch | Backprop | Two optimizers: SBOW updates embeddings, Adam updates model layers |
| `JOINT` | Single backward | Backprop | Single optimizer: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}} + \lambda \cdot \mathcal{L}_{\text{SBOW}}$ |

#### `<sbowRatio>` — SBOW Loss Weight (JOINT mode)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sbowRatio` | float | `0.1` | Weight $\lambda$ of the SBOW co-occurrence loss in JOINT mode. Higher values bias training toward embedding quality; lower values prioritize model prediction. |

#### `<OptimizeEmbedding>` — Gradient Flow Through Codebook

Controls whether the codebook `nn.Embedding` weight is **detached** during forward/reverse passes and whether it is included in the main optimizer's parameter list.

| `OptimizeEmbedding` | `detach()` in forward/reverse | In main optimizer | Use case |
|---------------------|------------------------------|-------------------|----------|
| `true` | No — gradients flow through codebook | Yes | ARLM/BOTH/JOINT: model backprop fine-tunes embeddings |
| `false` | Yes — codebook is a frozen lookup table | No | SBOW/CBOW: embedding updates are isolated to their own optimizer |

**`<train>` and `<OptimizeEmbedding>` are independent axes.** Common combinations:

| `<train>` | `<OptimizeEmbedding>` | Effect |
|-----------|-----------------------|--------|
| `ARLM` | `false` | Model trains, codebook completely frozen (no gradient flow) |
| `ARLM` | `true` (default) | Model trains, backprop also fine-tunes codebook |
| `SBOW` | `false` (default) | Only SBOW updates codebook via its own optimizer |
| `BOTH` | `true` (default) | SBOW + model backprop both update codebook (two optimizers) |
| `BOTH` | `false` | SBOW updates codebook via its own optimizer; model trains but codebook detached from backprop |
| `JOINT` | `true` (default) | Single combined loss; one optimizer updates everything |
| `JOINT` | `false` | Model loss doesn't flow through codebook, but SBOW loss still does (unusual) |

---

## `<InputSpace>`

The entry point that lifts raw data into the model's internal representation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Sequence length: maximum number of tokens per input. For XOR: 2 (two binary inputs). For text models, OOV words are spelled out as individual characters, so a single word may consume multiple slots — a short sentence with many OOV words can exceed this limit and trigger a truncation warning suggesting a lower `<minFrequency>`. |
| `nDim` | int | `1` | Dimensionality of each input vector. Set on TheObjectEncoding via XML; not passed to the Space constructor. |
| `nVectors` | int | = `nActive` | Codebook size (total vectors in the space). Defaults to `nActive` for InputSpace. |
| `nWhere` | int | `0` | Number of spatial/positional dimensions appended to each vector. When > 0, enables PositionalEncoding on all objects throughout the model. |
| `nWhen` | int | `0` | Number of temporal dimensions appended to each vector. When > 0, enables TemporalEncoding on all objects throughout the model. |
| `quantized` | bool | `false` | Whether input values are quantized/discrete. |
| `tokenizer` | string | `"traditional"` | Tokenization method for text inputs. |

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
| `invertible` | bool | `false` | Use a single invertible Pi layer instead of separate forward/reverse layers. When `true`, one layer handles both directions. When `false`, separate `pi1` (forward) and `pi2` (reverse) layers are created. |
| `hasAttention` | bool | `true` | Enable attention mechanism in this space. |
| `passThrough` | bool | `false` | Skip perceptual processing entirely; pass input through unchanged. |

**Layers:** Pi layers — multiplicative: `y_j = b_j * prod_i(1 + W_ji * x_i)`. See Architecture.md.

---

## `<ConceptualSpace>`

Transforms perceptual features into abstract concepts via Sigma layers (additive/linear).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Number of active concept vectors (sparse subset). For XOR: 3. |
| `nDim` | int | `1` | Dimensionality of each concept vector. Set on TheObjectEncoding; not passed to the Space constructor. |
| `nVectors` | int | `0` | Codebook size (total vectors in the space). |
| `invertible` | bool | `false` | Use single invertible Sigma layer vs. separate forward/reverse layers (`sigma1`/`sigma2`). |
| `hasAttention` | bool | `false` | Enable attention in conceptual processing. |
| `hasNorm` | bool | `false` | Enable layer normalization in this space. |

**Layers:** Sigma layers — additive: `y_j = b_j + sum_i(W_ji * x_i)`. See Architecture.md.

---

## `<SymbolicSpace>`

Discrete symbolic representation — the information bottleneck between perception and output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | *required* | Number of active symbols. This is the full count; when reconstruction symbols are enabled, `nOutputSymbols = OutputSpace.nActive` are fed to output, and the rest carry reconstruction information. For XOR: 3 (1 output + 2 reconstruction). |
| `nDim` | int | `1` | Dimensionality of each symbol (typically 0 — symbols are zero-dimensional). Set on TheObjectEncoding; not passed to the Space constructor. |
| `nVectors` | int | = `nActive` | Codebook size (total vectors in the space). |
| `passThrough` | bool | `false` | Pass concepts through as symbols unchanged (no learned transformation). Typically `true` for simple models. |

**Layers:** None when `passThrough=true`. The bottleneck effect comes from the dimensionality constraint, not learned weights.

---

## `<SyntacticSpace>`

Syntactic processing stage between symbols and output. Used when `symbolicOrder >= 1` to add an extra transformation layer. Future work will integrate generative grammar operations here.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nActive` | int | `16` | Number of active word vectors (sparse subset selected in forward pass). |
| `nDim` | int | `1` | Dimensionality of each word vector (symbolic layer, so typically low-dimensional). Set on TheObjectEncoding as `wordDim`. |
| `nVectors` | int | = `nActive` | Codebook size (total vectors in the space). Stored on TheObjectEncoding as `nWords`. |

**Layers:** Currently a passthrough that reshapes symbols without transformation.

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
    <reversible>true</reversible>
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
    <reversible>true</reversible>
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
      <train>ARLM</train>
      <OptimizeEmbedding>false</OptimizeEmbedding>
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
    <quantized>true</quantized>
  </InputSpace>

  <!-- ... space definitions ... -->
</model>
```

Key points:
- `modelType=embedding` activates the embedding-based input pipeline alongside the neural model
- `train=ARLM` trains only the network layers; `OptimizeEmbedding=false` detaches the codebook so no gradients flow through it
- The three files partition model behaviour: **XML config** (architecture), **`.kv` embedding** (codebook), **`.ckpt` weights** (model layers)
- `weightsPath` stores the neural model checkpoint; `embeddingPath` stores the word vectors
- `minFrequency` gates vocabulary admission: words are buffered until their frequency ratio exceeds this threshold
