# XML Configuration Parameters

Model configuration is specified in XML files (e.g. `data/XOR_exact.xml`). Default values are loaded from `data/defaults.xml` and overridden by model-specific configs.

---

## `<architecture>`

Core model settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelType` | string | `"simple"` | Model type. `"lm"` for language model (adjusts dimensions for sequence processing), `"simple"` for basic feedforward. |
| `dataset` | string | `"xor"` | Dataset to load. Determines input/output data via `TheData.load()`. |
| `pretrained` | bool | `false` | Whether to use a pretrained model (e.g. BERT embeddings). |
| `reversePass` | bool | `false` | Enable the reverse (reconstruction) pass. When true, the model trains both forward prediction and backward reconstruction. |
| `ergodic` | bool | `false` | Enable ergodic exploration mode. Layers use `W_eff = bias * W + temp * noise` where bias/temp are derived from alpha. See Architecture.md for the gradient energy sensor. |
| `certainty` | bool | `false` | Enable per-neuron certainty tracking in ergodic layers. Allows individual neurons to transition from exploration to exploitation at different rates. |
| `reshape` | bool | `false` | Reshape input data for sequence processing. Required for `modelType=lm`. |
| `numEpochs` | int | `3` | Number of training epochs. |
| `numTrials` | int | `1` | Number of independent training runs. |
| `batchSize` | int | `10` | Mini-batch size for training. |
| `learningRate` | float | `0.01` | Learning rate for the Adam optimizer. |
| `stoppingCriterion` | float | — | Early stopping threshold. Training halts when output loss falls below this value. |
| `detectAnomaly` | bool | `false` | Enable PyTorch anomaly detection. Useful for debugging NaN gradients but slows training. |
| `conceptualOrder` | int | `0` | Order of conceptual processing. Controls how many higher-order conceptual transforms are applied. |
| `symbolicOrder` | int | `0` | Order of symbolic processing. Controls how many higher-order symbolic transforms are applied. |
| `processSymbols` | bool | `false` | Enable additional symbolic processing steps. |
| `nWhere` | int | `0` | Number of spatial/positional dimensions. Used for spatially-structured data. |
| `nWhen` | int | `0` | Number of temporal dimensions. Used for temporally-structured data. |
| `objectSize` | int | `0` | Size of object representations. Used with reshape for structured input encoding. |
| `nWords` | int | `16` | Vocabulary size for tokenized inputs. |
| `weightsPath` | string | `"output/weights.pt"` | File path for saving/loading model weights. |
| `autoload` | bool | `true` | Automatically load weights from `weightsPath` on model creation. Set to `false` for fresh training. |
| `autosave` | bool | `false` | Automatically save weights after training completes. |
| `classificationMin` | float | — | Minimum threshold for classification accuracy reporting. |
| `classificationMax` | float | — | Maximum threshold for classification accuracy reporting. |

---

## `<InputSpace>`

The entry point that lifts raw data into the model's internal representation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nVectors` | int | *required* | Number of input vectors (sequence length or input dimensionality). For XOR: 2 (two binary inputs). |
| `dim` | int | `1` | Dimensionality of each input vector. |
| `quantized` | bool | `false` | Whether input values are quantized/discrete. |
| `tokenizer` | string | `"traditional"` | Tokenization method for text inputs. |

**Layers:** Contains a `LiftingLayer` that projects raw input into the model's working dimensionality.

---

## `<PerceptualSpace>`

Transforms lifted input into perceptual features via Pi layers (multiplicative interactions).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nVectors` | int | *required* | Number of perceptual feature vectors. Should be >= InputSpace nVectors. For XOR: 4. |
| `dim` | int | `1` | Dimensionality of each perceptual vector. |
| `prototypes` | int | `0` | Number of prototype vectors for prototype-based computation. When > 0, enables prototype matching. |
| `invertible` | bool | `false` | Use a single invertible Pi layer instead of separate forward/reverse layers. When `true`, one layer handles both directions. When `false`, separate `pi1` (forward) and `pi2` (reverse) layers are created. |
| `hasAttention` | bool | `true` | Enable attention mechanism in this space. |
| `passThrough` | bool | `false` | Skip perceptual processing entirely; pass input through unchanged. |

**Layers:** Pi layers — multiplicative: `y_j = b_j * prod_i(1 + W_ji * x_i)`. See Architecture.md.

---

## `<ConceptualSpace>`

Transforms perceptual features into abstract concepts via Sigma layers (additive/linear).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nVectors` | int | *required* | Number of concept vectors. For XOR: 3. |
| `dim` | int | `1` | Dimensionality of each concept vector. |
| `prototypes` | int | `0` | Number of prototype vectors. |
| `invertible` | bool | `false` | Use single invertible Sigma layer vs. separate forward/reverse layers (`sigma1`/`sigma2`). |
| `hasAttention` | bool | `false` | Enable attention in conceptual processing. |
| `hasNorm` | bool | `false` | Enable layer normalization in this space. |

**Layers:** Sigma layers — additive: `y_j = b_j + sum_i(W_ji * x_i)`. See Architecture.md.

---

## `<SymbolicSpace>`

Discrete symbolic representation — the information bottleneck between perception and output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nVectors` | int | *required* | Total number of symbols. This is the full count; when reconstruction symbols are enabled, `nOutputSymbols = OutputSpace.nVectors` are fed to output, and the rest carry reconstruction information. For XOR: 3 (1 output + 2 reconstruction). |
| `dim` | int | `1` | Dimensionality of each symbol. |
| `passThrough` | bool | `false` | Pass concepts through as symbols unchanged (no learned transformation). Typically `true` for simple models. |

**Layers:** None when `passThrough=true`. The bottleneck effect comes from the dimensionality constraint, not learned weights.

---

## `<OutputSpace>`

Maps symbols to final predictions via linear layers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nVectors` | int | *required* | Number of output values. For XOR: 1 (single binary output). |
| `dim` | int | `1` | Dimensionality of each output. |

**Layers:** Linear layers with `(bias, temp)` support for ergodic mode.

---

## `<weights>` (Legacy)

Legacy section for weight management. Parameters have been migrated to `<architecture>` but this section is still read as fallback.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `autoload` | bool | `true` | Fallback for `architecture/autoload`. |
| `path` | string | `"output/weights.pt"` | Fallback for `architecture/weightsPath`. |

---

## `<server>`

HTTP server settings (used by `serve.py`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Server bind address. |
| `port` | int | `8001` | Server port. |

---

## Example: XOR Configuration

```xml
<model>
  <architecture>
    <reversePass>true</reversePass>
    <reshape>true</reshape>
    <dataset>xor</dataset>
    <modelType>lm</modelType>
    <numEpochs>1000</numEpochs>
    <learningRate>0.01</learningRate>
    <ergodic>false</ergodic>
    <autoload>false</autoload>
    <autosave>false</autosave>
  </architecture>

  <InputSpace>
    <nVectors>2</nVectors>
    <dim>1</dim>
  </InputSpace>

  <PerceptualSpace>
    <nVectors>4</nVectors>
    <dim>1</dim>
    <prototypes>16</prototypes>
    <invertible>false</invertible>
    <hasAttention>false</hasAttention>
  </PerceptualSpace>

  <ConceptualSpace>
    <nVectors>3</nVectors>
    <dim>1</dim>
    <prototypes>16</prototypes>
    <invertible>false</invertible>
  </ConceptualSpace>

  <SymbolicSpace>
    <nVectors>3</nVectors>
    <passThrough>true</passThrough>
  </SymbolicSpace>

  <OutputSpace>
    <nVectors>1</nVectors>
    <dim>1</dim>
  </OutputSpace>
</model>
```

In this configuration:
- 2 binary inputs are lifted, transformed through 4 perceptual and 3 conceptual features
- 3 symbols are produced: 1 for output prediction, 2 for reconstruction
- The reverse pass reconstructs the original 2 inputs from all 3 symbols
- Ergodic mode is off; training uses standard Adam with combined forward+reverse loss
