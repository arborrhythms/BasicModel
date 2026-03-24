# Spaces

## Overview

BasicModel is organized as a pipeline of six **spaces**, each performing a distinct
representational transformation. Data flows forward from raw input to task output;
the reverse pass reconstructs the original input from the symbolic representation.

```
Forward:  InputSpace → PerceptualSpace → ConceptualSpace → SymbolicSpace → SyntacticSpace → OutputSpace
Reverse:  OutputSpace → SyntacticSpace → SymbolicSpace → ConceptualSpace → PerceptualSpace → InputSpace
```

---

## Base Class: Space

All spaces inherit from a common `Space` base that manages the shared infrastructure:

**Shape management.** Each space tracks `inputShape` and `outputShape` as `[nObjects, nDim]`
tuples, where `nObjects` is the active count and `nDim` is the per-object dimensionality.
Dimensions are not passed as constructor arguments; each subclass reads them from
`TheObjectEncoding` (e.g. `TheObjectEncoding.perceptDim` for PerceptualSpace).

**Codebook / VQ quantization.** When `nVectors > nActive`, the space maintains a
codebook of `nVectors` candidate vectors and selects the `nActive` most activated ones
via top-k selection. This is the vector-quantization bottleneck.

**Reshape flag.** When the input and output object counts differ, the reshape flag
flattens the `[B, nObj, nDim]` tensor to `[B, nObj * nDim]` before passing it to the
next space, then restores the object structure on the way back.

**Attention.** Each space may optionally include an attention mechanism
(`hasAttention=true`) that reweights objects before the main transformation.

**set_sigma propagation.** When ergodic mode is active, `set_sigma()` is forwarded from
the top-level model down through every space and into every child layer, so the
per-neuron exploration level is kept consistent across the entire pipeline.

---

## InputSpace

**Role.** Receives the raw source buffer from `Data()` and lifts it into the model's
internal working dimensionality.

**Text mode — forward.** Delegates tokenization to `Lex`, which produces a span table
of `(start, end, type)` entries — one span per token. Each span is converted to a
vector with two components:

- `nWhat` dimensions — token content, encoded via `Basis` / `Codebook` (the word
  embedding lookup).
- `nWhere` dimensions — positional information derived from the character offset.

The result is a `[nActive, nWhat + nWhere]` tensor representing the tokenized sentence.

**Text mode — reverse.** Reconstructs the source string from the latent state by
inverting the span encoding: each vector is mapped back to its nearest codebook entry
(word embedding), then spans are reassembled into characters using the stored offset
table.

**Numeric mode.** Tensor data is passed through unchanged; the LiftingLayer projects
the native input dimension (e.g. 784 for MNIST) to the model's working dimensionality
`nDim`.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Sequence length: maximum tokens per input |
| `nDim` | Output dimensionality per vector (set on TheObjectEncoding) |
| `nWhere` | Positional dimensions appended to each token vector |
| `nWhen` | Temporal dimensions appended to each token vector |
| `lexer` | Tokenization mode: `"word"` or `"sentence"` |
| `quantized` | Whether input values are discrete |

**Layer.** `LiftingLayer` — bridges native input dimension to `nDim`.

**Invertibility.** InputSpace is always non-invertible in the strict sense; the reverse
path is a separate reconstruction procedure using the span table, not a matrix inverse.

---

## PerceptualSpace

**Role.** Transforms raw input vectors into perceptual representations via multiplicative
interactions. Models prototype-based feature detection: each percept is a product of
local input features.

**Forward operation.**
```
y_j = b_j * prod_i(1 + W_ji * x_i)
```
Multiplicative structure allows the layer to detect conjunctions of features.

**Reverse operation (invertible=True).** A single `PiLayer(invertible=True)` is shared
for both directions. The forward pass outputs interleaved `(log_y, log_z)` pairs, where:
```
y_j = b_j * prod_i(1 - tanh(W_ji * x_i))
z_j = b_j * prod_i(1 + tanh(W_ji * x_i))
```
The reverse pass recovers `gamma = W x` from `gamma_j = 0.5 * log(z_j / y_j)`, then
solves for `x = W^{-1} * gamma` using InvertibleLinearLayer.

**Reverse operation (invertible=False, reversible=True).** Two separate `PiLayer`
instances — `pi1` for forward, `pi2` for reverse — each with independent weights. The
reverse layer uses a pseudoinverse to approximately invert the forward mapping.

**passThrough=True.** Identity: input passes through unchanged, no Pi computation.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of active perceptual vectors |
| `nVectors` | Codebook size; enables VQ when > nActive |
| `invertible` | True: shared invertible layer; False: separate pi1/pi2 |
| `passThrough` | Skip perceptual processing entirely |
| `hasAttention` | Enable attention reweighting |

**Layer.** `PiLayer` (one or two instances depending on `invertible`).

**Invertibility.** `invertible=True`: shared layer, exact inverse. `invertible=False`:
separate layers, approximate pseudoinverse in reverse.

---

## ConceptualSpace

**Role.** Transforms perceptual vectors into abstract concepts via additive linear
operations. Models conceptual hyperplanes that partition perceptual space.

**Forward operation.**
```
y_j = tanh(W x + b)
```
The tanh nonlinearity squashes activations to `[-1, +1]`, giving each concept a
graded membership value.

**Reverse operation (invertible=True).** A single `SigmaLayer(invertible=True)` shared
for both directions. Exact inverse via:
```
pre_tanh = atanh(y)
x = W^{-1} * (pre_tanh - b)
```
The matrix inverse `W^{-1}` is computed via `InvertibleLinearLayer` (LDU
factorisation), so no SVD is required and the inverse is exact.

**Reverse operation (invertible=False, reversible=True).** Two separate `SigmaLayer`
instances — `sigma1` for forward, `sigma2` for reverse.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of active concept vectors |
| `nVectors` | Codebook size |
| `invertible` | True: shared invertible layer; False: separate sigma1/sigma2 |
| `hasAttention` | Enable attention |
| `hasNorm` | Enable layer normalization |

**Layer.** `SigmaLayer` (one or two instances depending on `invertible`).

**Invertibility.** `invertible=True`: exact inverse via atanh + `W^{-1}`. `invertible=False`:
separate layers with independent weights.

---

## SymbolicSpace

**Role.** Converts continuous concept vectors into discrete symbols. This is the
information bottleneck of the pipeline: rich perceptual–conceptual representations
are compressed into a small set of symbols that can be matched, compared, and
reasoned over.

**Forward operation (processSymbols=True).** Computes scalar activations from concept
vectors as vector norms. Each concept vector is measured, and the resulting scalar is
the "activation level" of that symbol. The `threshold` flag controls hard
discretization: activations below threshold are zeroed out. The `serialActivation`
flag controls whether symbols fire one at a time (serial) or in parallel.

**Reverse operation (processSymbols=True).** Dereferences concept vectors from the
`ConceptualSpace` codebook: each symbol index is looked up in the codebook and the
corresponding concept vector is retrieved, reconstructing a continuous representation
for further reverse processing.

**passThrough=True.** Concept vectors pass through unchanged with no norm computation
or discretization. Typically used for simple feedforward models where symbolic
compression is not needed.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Total number of symbols (output + reconstruction symbols) |
| `nVectors` | Codebook size (defaults to nActive) |
| `passThrough` | Skip symbolic processing entirely |
| `threshold` | Enable hard activation threshold for discretization |
| `serialActivation` | Fire symbols one at a time rather than in parallel |
| `processSymbols` | Enable norm-based symbol activation |

**Layer.** None when `passThrough=True`. When `processSymbols=True`, the bottleneck
effect comes from the dimensionality constraint rather than a learned weight matrix.

**Invertibility.** Not invertible in the weight-matrix sense. Reverse dereferences the
codebook by symbol index.

---

## SyntacticSpace

**Role.** Placeholder for future generative grammar operations. Currently performs a
reshape of the symbol tensor without any learned transformation. Sits between
SymbolicSpace and OutputSpace in the pipeline.

**Active when.** Used when `symbolicOrder >= 1`.

**Forward operation.** Reshapes the symbol tensor from `[B, nSymbols, symbolDim]` to
the word-vector layout `[B, nWords, wordDim]`. No nonlinearity or weight matrix is
applied.

**Reverse operation.** Inverse reshape.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of active word vectors (`nWords` on TheObjectEncoding) |
| `nDim` | Word vector dimensionality (`wordDim` on TheObjectEncoding) |
| `nVectors` | Codebook size (defaults to nActive) |

**Layer.** Passthrough reshape; no trainable parameters.

**Invertibility.** Trivially invertible (reshape is lossless).

---

## OutputSpace

**Role.** Maps symbolic (or syntactic) vectors to task targets via a linear projection.
This is the final prediction stage.

**Forward operation.** Linear projection from symbol dimensionality to output
dimensionality:
```
y = W_out * x + b_out
```
Always uses `reshape=True` so the `[B, nSymbols, symbolDim]` tensor is flattened before
projection.

**Reverse operation.** Projects output predictions back to symbol space via the
pseudoinverse of `W_out`. In text mode, snaps each output vector to the nearest entry
in the codebook (nearest-neighbour lookup) to recover a discrete token, then assembles
tokens into a string.

**getEmbeddedIO() override.** OutputSpace overrides `getEmbeddedIO()` to return the
raw target dimensions rather than the encoded dimensions. This ensures the loss is
computed in the output vocabulary space, not the embedding space, regardless of any
encoding overhead.

**Text mode generation.** In autoregressive language model mode, OutputSpace iterates
token-by-token: at each step the predicted token vector is snapped to its nearest
codebook entry, that entry is fed back as input, and generation continues until
`maxResponseLength` is reached or an end-of-sequence token is produced.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of output values (e.g. 1 for XOR, vocab size for LM) |
| `nDim` | Output vector dimensionality |
| `nVectors` | Codebook size (defaults to nActive) |

**Layer.** `LinearLayer` with `(bias, temp)` support for ergodic mode. Always
`reshape=True`.

**Invertibility.** Reverse uses pseudoinverse; not exactly invertible in general.
