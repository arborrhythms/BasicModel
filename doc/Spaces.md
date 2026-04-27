# Spaces

## Overview

BasicModel is organized as a pipeline of six **spaces**, each performing a distinct
representational transformation. Data flows forward from raw input to task output;
the reverse pass reconstructs the original input from the symbolic representation.

```
Forward:  InputSpace $\rightarrow$ PerceptualSpace $\rightarrow$ ConceptualSpace $\rightarrow$ SymbolicSpace $\rightarrow$ SyntacticSpace $\rightarrow$ OutputSpace
Reverse:  OutputSpace $\rightarrow$ SyntacticSpace $\rightarrow$ SymbolicSpace $\rightarrow$ ConceptualSpace $\rightarrow$ PerceptualSpace $\rightarrow$ InputSpace
```

![WikiOracle Space Hierarchy](diagrams/vector_spaces.svg)

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

**Reset cascade — hard vs. soft.** Every space exposes
`Reset(batch=None, hard=True)`:

| Call form | Scope | Use |
|-----------|-------|-----|
| `space.Reset()` | All rows | Whole-state wipe (e.g. epoch boundary, manual reload) |
| `space.Reset(batch=b, hard=True)` | Row `b` only | Document boundary, full per-row state wipe |
| `wordSpace.soft_reset(batch=b)` | Row `b`, sentence-scoped state only | Grammar `<start>` reduction completes |

Hard-reset clears: parse stack, `_last_svo`, `_stm_fired`, codebook commit
accumulator, discourse history, `serial_cache`, `_ar_embedded`,
`_end_of_stream` for the affected row(s).

Soft-reset clears: parse stack, `_last_svo`, re-arms `_stm_fired`. **Does
not** touch discourse history or codebook EMA — those are
document-scoped, not sentence-scoped.

Reset is dispatched from the outer doc-streaming loop in `runEpoch`, never
from inside `runBatch` (which is a pure compute brick — see
[Architecture.md](Architecture.md) "Pipeline as a unit, two-tier reset").

---

## Normalization and Ranges

Every space enforces a canonical range on its vectors and activations.
Each space always normalizes its output to the correct range.

| Space | Data Contract | Geometry |
|-------|--------------|----------|
| InputSpace | Data scaled -1..1 for scalars or vector norms | Signed unit interval `[-1,1]` |
| PerceptualSpace | Modal/demuxed (what/where/when encoding). Signed unit-magnitude scalars or vectors centered at the origin. No negation operator, but signed for symmetry | Signed hypercube `[-1,1]^d` |
| ConceptualSpace | Combined/muxed (event encoding). Signed unit-magnitude scalars or vectors (tanh-bounded). Event norm stored on `subspace.activation` | `[-1,1]` per element (tanh) |
| SymbolicSpace | Symbols are percepts. Concept-to-symbol mapping. One symbol encoded at a time (most highly active). Each symbol receives where/when from PerceptualSpace | `[0,1]` presence |
| SyntacticSpace | Words are concepts encoding grammatical rules. Stored as word tuples with production rules | Word tuples `(batch, vector, rule)` |
| OutputSpace | Rescaled from activation range to original data range | Data range |

**Data scaling.** `Data` computes global scalar `input_min`/`input_max` and
`output_min`/`output_max` from the training data at load time. InputSpace uses
`Data.normalize(x, "input")` to scale non-embedding inputs to `[-1, 1]`, and
`Data.denormalize(x, "input")` in reverse to restore the original range.
OutputSpace uses `Data.denormalize(x, "output")` to map from `[-1, 1]`
(symbolic activation range) to the original output range, and
`Data.normalize(x, "output")` in reverse.

**Symbols as percepts.** Symbols represent the presence or absence of a named
entity and live in `[0, 1]`. Since conceptual activations range `[-1, 1]`,
the mapping is `symbol = (activation + 1) / 2`. The `SubSpace.get_symbols()`
and `set_symbols()` methods perform this conversion, and SymbolicSpace /
SyntacticSpace use them exclusively instead of `get_activation` /
`set_activation`.

**Demuxed mode.** When `InputSpace.demuxed=true`, what/where/when components
are stored independently in the SubSpace rather than concatenated into a single
event tensor. This keeps the codebook pure (positional data never contaminates
content vectors). ModalSpace routes each component through independent
PerceptualSpaces. Downstream spaces see an identical muxed tensor via
`materialize()`.

---

## InputSpace

**Role.** Receives the raw source buffer from `Data()` and lifts it into the model's
internal working dimensionality.

**Text mode -- forward.** Delegates tokenization to `Lex`, which produces a span table
of `(start, end, type)` entries -- one span per token. Each span is converted to a
vector with two components:

- `nWhat` dimensions -- token content, encoded via `Basis` / `Codebook` (the word
  embedding lookup).
- `nWhere` dimensions -- positional information derived from the character offset.

The result is a `[nActive, nWhat + nWhere]` tensor representing the tokenized sentence.

**Text mode -- reverse.** Reconstructs the source string from the latent state by
inverting the span encoding: each vector is mapped back to its nearest codebook entry
(word embedding), then spans are reassembled into characters using the stored offset
table.

**Numeric mode.** Tensor data is passed through unchanged; the LiftingLayer projects
the native input dimension (e.g. 784 for MNIST) to the model's working dimensionality
`nDim`. Non-embedding inputs are then scaled to `[-1, 1]` using the global data min/max,
and the reverse path restores the original range.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Sequence length: maximum tokens per input |
| `nDim` | Output dimensionality per vector (set on TheObjectEncoding) |
| `nWhere` | Positional dimensions appended to each token vector |
| `nWhen` | Temporal dimensions appended to each token vector |
| `lexer` | Tokenization mode: `"word"` or `"sentence"` |
| `codebook` | Whether input values are discrete |
| `demuxed` | Store what/where/when independently (default: `false`) |

**Layer.** `LiftingLayer` -- bridges native input dimension to `nDim`.

**Invertibility.** InputSpace is always non-invertible in the strict sense; the reverse
path is a separate reconstruction procedure using the span table, not a matrix inverse.

**Document streaming and `valid_mask`.** Documents longer than `nOutput` bytes are not
truncated. `TheData` maintains a per-row cursor `(doc_idx[b], offset[b])` and
`next_tick()` returns a `[B, nOutput]` slab containing the next ≤`nOutput` bytes from
each row's current document, plus a host-side `[B] list[bool]` `hard_eos` flag set on
ticks where a row's cursor exhausts the current document. A short fill at document end
NULL-pads the slab tail; the stem's `valid_mask: [B, K] bool` ([Spaces.py:4622]) flips
False for the padded positions, and downstream state-mutation propagation (codebook
EMA, parse-stack push, truth-layer record) skips them. Concatenating per-tick slabs
across a row's document run reproduces the original bytes byte-exact.

The legacy `_end_of_stream` tensor remains for diagnostic readback only — runBatch no
longer consumes it for control flow (the in-loop `.all().item()` gate was removed by
the predecessor handoff and the surrounding Reset block was relocated to the outer
loop by the rolling-cursor handoff).

---

## PerceptualSpace

**Role.** Transforms raw input vectors into perceptual representations via multiplicative
interactions. Models prototype-based feature detection: each percept is a product of
local input features.

**Forward operation (log-space linear).**
```
s_i = log((1 + x_i) / (1 - x_i))          -- atanh domain transform
z_j = W @ s + b                             -- linear in log-multiplicative space
y_j = (exp(z_j) - 1) / (exp(z_j) + 1)     -- tanh back to [-1, 1]
```
The atanh/tanh domain transform gives multiplicative semantics: addition in
log-space corresponds to multiplication of the original features.

**Reverse operation (invertible=True).** A single `PiLayer(invertible=True)` is shared
for both directions. The reverse path inverts each step: `_to_mult(y)`, log,
`W^{-1}(z - b)`, exp, `_from_mult`. The matrix inverse uses InvertibleLinearLayer (LDU
factorisation) for exact inversion.

**Reverse operation (invertible=False, reversible=True).** Two separate `PiLayer`
instances -- `pi1` for forward, `pi2` for reverse -- each with independent weights.

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

**Range.** Vectors live in the signed hypercube `[-1, 1]^d` (tanh-bounded). The space
is centered at the origin for geometric symmetry, though it has no negation operator --
percepts represent feature magnitudes with sign indicating direction. Activation is
`[-1, 1]`. Tanh is applied to enforce the range.

**Invertibility.** `invertible=True`: shared layer, exact inverse. `invertible=False`:
separate layers, approximate pseudoinverse in reverse.

---

## ConceptualSpace

**Role.** Transforms perceptual vectors into abstract concepts via additive linear
operations. Models conceptual hyperplanes that partition perceptual space.

**Owned layer.**

| Layer                | Direction        | Math                                | Notes                                              |
|----------------------|------------------|-------------------------------------|----------------------------------------------------|
| `self.pi` (`PiLayer`) | P $\leftrightarrow$ C            | log-domain multiplicative, monotonic | `forwardPi` (P → C) and `reversePi` (C → P) pointer aliases |

ConceptualSpace owns one PiLayer that handles both directions of the
P$\leftrightarrow$C boundary via its own self-inverse.  When the space is reversible
without an invertible layer, two PiLayers (`pi1`, `pi2`) are
constructed and `forwardPi` / `reversePi` route to them; otherwise a
single `pi` serves both.

**Binary forward (Step 5).**  `self.pi.forward` accepts a
`binary=True` flag that hard-selects the top-2 input operands by
$|x_i|$ (via `Ops.top2_select_ste`) and zeros the rest before the
log-domain fold.  Zero is the multiplicative identity for Pi's AND so
unselected operands drop cleanly out of the pool.  Backward is
straight-through: every input dim retains a learning signal so the
layer learns to make all candidates sensible, even when they aren't
currently in the top-2.  Used by grammar dispatch where a binary rule
combines the two most-active constituents.

**Forward operation (P → C).** PiLayer's log-domain product fold:
```
m   = (1 + x) / (1 - x)               # (-1,1) -> (0, inf)
y   = tanh(W * log(m) / 2 + b)        # back to (-1, 1)
```
The `monotonic` flag constrains $W \ge 0$ so the AND fold preserves
ordering; `nonlinear=True` keeps the output in `[-1, 1]`.

**Reverse operation (invertible=True).** Exact log-domain inverse via
`InvertibleLinearLayer` (or `NonNegativeInvertibleLinearLayer` when
`monotonic=True`):
```
l   = log((1 + y) / (1 - y))
x   = tanh(W^{-1} * (l - b) / 2)
```

**Reverse operation (invertible=False, reversible=True).** Two
separate `PiLayer` instances -- `pi1` for forward, `pi2` for reverse.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of active concept vectors |
| `nVectors` | Codebook size |
| `invertible` | True: shared invertible layer; False: separate sigma1/sigma2 |
| `hasAttention` | Enable attention |
| `hasNorm` | Enable layer normalization |

**Layer.** `PiLayer` (one or two instances depending on `invertible`).

**Range.** Vectors are tanh-bounded: each element is in `[-1, 1]` (applied by SigmaLayer).
The boundary between PerceptualSpace and ConceptualSpace uses
`atanh` (forward) / `tanh` (reverse) as exact inverses. Tanh is applied to enforce
the element-wise range.

**Activation carrier.** `ActiveEncoding.nDim = 2`: activation is a 2-dim
bivector `[aP, aN]` per position, encoding the four corners of the
tetralemma (*catuskoti*):

| State            | `[aP, aN]` |
|------------------|------------|
| TRUE (*asti*)    | `[1, 0]`   |
| FALSE (*nasti*)  | `[0, 1]`   |
| BOTH (*ubhaya*)  | `[1, 1]`   |
| NEITHER (*anubhaya*) | `[0, 0]` |

BOTH encodes first-class inconsistency (same position affirmed and
negated by independent sources/frames); NEITHER encodes unknown
(neither affirmed nor negated).  Operations obey De Morgan under
pole-swap negation $\neg[aP, aN] = [aN, aP]$:

- Conjunction: `[min(aP, bP), max(aN, bN)]`
- Disjunction: `[max(aP, bP), min(aN, bN)]`

See [BuddhistParallels.md](BuddhistParallels.md).

**Invertibility.** `invertible=True`: exact inverse via atanh + `W^{-1}`. `invertible=False`:
separate layers with independent weights.

**MASK on `SubSpace._active`.** `SubSpace` tracks two orthogonal
per-position tensors: `activation` (the 4-valued truth bivector above)
and `_active` shaped `[B, N, M]` where $M$ is the number of modality
presence flags (what / where / when).  Grammar-rule masking is shape-
disambiguated by `_apply_mask`:

| Mask shape   | Effect                                                       |
|--------------|--------------------------------------------------------------|
| Aligns with `out.shape[-1]` (feature axis) | Element-wise multiply on the output tensor. |
| Aligns with `out.shape[-2]` (position axis) | Zero the masked rows of `subspace._active`; `materialize()` then gates those positions downstream. |

This makes MASK a first-class filter on the presence flags rather than
an arithmetic multiplication, so masked positions propagate their
"absent" status through the pipeline's active-materialization path.

---

## SymbolicSpace

**Role.** Converts continuous concept activations into a discrete set of active
symbols. This is the information bottleneck of the pipeline: rich
perceptual-conceptual representations are compressed into a sparse presence
pattern over a codebook of symbol prototypes.

**Owned layer.**

| Layer                       | Direction | Math                              | Notes                                                 |
|-----------------------------|-----------|-----------------------------------|-------------------------------------------------------|
| `self.sigma` (`SigmaLayer`) | C $\leftrightarrow$ S     | atanh-domain additive, monotonic  | `forwardSigma` (C → S) and `reverseSigma` (S → C) pointer aliases |

SymbolicSpace owns one SigmaLayer that handles both directions of the
C$\leftrightarrow$S boundary via its own self-inverse, isomorphic to `ConceptualSpace.pi`.
When the space is reversible without an invertible layer, two SigmaLayers
(`sigma1`, `sigma2`) are constructed and `forwardSigma` / `reverseSigma`
route to them; otherwise a single `sigma` serves both.  The legacy
`self.layer` PiLayer attribute is gone -- consumers use
`model.symbolicSpace.sigma` directly.

`self.sigma.forward` accepts the Step-5 `binary=True` flag (see
ConceptualSpace section).

**Symbols are percepts.** Each symbol represents the presence (`1`) or absence
(`0`) of a named entity. Since conceptual activations range `[-1, 1]`, the
mapping between the two domains is `symbol = (activation + 1) / 2`.
`SubSpace.get_symbols()` and `set_symbols()` perform this conversion;
SymbolicSpace uses them exclusively instead of raw `get_activation` /
`set_activation`.

See [Language.md](Language.md) for the full language system design.

**Forward operation (codebook=True).**

1. Extract concept activation `[B, nConcepts]` from the input subspace.
2. Map through `SigmaLayer(nConcepts, nSymbols, invertible=True, monotonic=True)` to `[B, nSymbols]`.
3. Store as symbolic presence `[0, 1]` via `set_symbols()`.
4. Reshape to `[B, nSymbols, 1]` (each symbol is 1-dim) and pass through the
   codebook, which quantizes and produces a one-hot activation over codebook
   entries weighted by similarity.

The output is a one-hot encoding over the codebook. The codebook provides dense
vectors for downstream spaces that require `[B, N, D]` tensors.

**Forward operation (codebook=False).** The SigmaLayer maps the activation
to symbolic presence via `set_symbols()`; vectors pass through from the input
subspace unchanged.

**Reverse operation.** Reads symbolic presence via `get_symbols()`, then the
PiLayer's exact inverse maps `[B, nSymbols]` back to `[B, nConcepts]`,
recovering the concept activation.

**passThrough=True.** Concept vectors and activation pass through unchanged.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Total number of symbols (output + reconstruction symbols) |
| `nVectors` | Codebook size (= nSymbols when codebook=true) |
| `passThrough` | Skip symbolic processing entirely |
| `codebook` | Enable codebook quantization (required for one-hot output) |

**Range.** Symbols are percepts: each symbol represents the presence (`1`) or absence
(`0`) of a named entity and lives in `[0, 1]`. One symbol is encoded at a time (the
most highly active; negative products never activate). Each symbol receives where/when
encoding from PerceptualSpace, making symbols uniform with percepts. Internally stored
as activation in `[-1, 1]` via the `(x+1)/2` mapping.

**Codebook shape.** The symbol codebook has one row per symbol, full muxed
width: `SymbolicSpace.subspace.what.getW().shape == (nVectors, 2 + nWhere + nWhen)`
with `nWhat = 2`.  The leading 2 dims of each row carry the bivector
`[pos_pole, neg_pole]` encoding the 4-valued (quaternary) truth of the
symbol via the tetralemma / *catuskoti*: TRUE=[1,0], FALSE=[0,1],
BOTH=[1,1], NEITHER=[0,0].  Trailing `nWhere + nWhen` dims carry
per-symbol positional/temporal template info.  `Basis.negation` swaps
`(pos, neg)` on the leading 2 dims only.  See
[BuddhistParallels.md](BuddhistParallels.md) for the catuskoti mapping.

**Layer.** `PiLayer(nConcepts, nSymbols, invertible=True, monotonic=True)` -- maps
between activation spaces via monotonic multiplicative transform. The `monotonic=True`
constraint ensures weights $W \geq 0$, preserving ordering. Exact inverse via the
internal `InvertibleLinearLayer` (LDU factorisation).

**Hierarchical mode.** When `<useButterflies>true</useButterflies>` or
`<useGrammar>true</useGrammar>`, SymbolicSpace stores an `nn.ModuleList` of
per-level PiLayers in `self.pi_layers` (ButterflyStage-wrapped when `useButterflies`
is active).  The symbol dimension is geometrically partitioned so each order
writes only to its designated slice.  See [Reasoning.md](Reasoning.md)
Section Architecture Quadrants.

**Invertibility.** Exactly invertible via the PiLayer's reverse path.

---

## SyntacticSpace

**Role.** Generates a binary derivation tree (deep structure) from the set of
active symbols produced by SymbolicSpace. Words are concepts encoding grammatical
rules. The derivation is a Chomsky Normal Form (CNF) grammar stored as word tuples
`(batch, vector, rule)` on the output subspace.

See [Language.md](Language.md) for the grammar, word encoding, and open questions
about differentiable tree structure.

**Forward operation.**

1. Read symbolic presence via `get_symbols()` (nonzero entries are present symbols).
2. For N present symbols per batch, generate a CNF derivation: N-1 binary rules
   (randomly selected) + 1 terminal (S $\rightarrow$ W).
3. Store the derivation as word tuples on the output subspace. Vectors and
   symbolic presence pass through unchanged via `set_symbols()`.

**Reverse operation.**

1. Walk the word list: every `(batch, vector, rule)` entry marks that position
   as present.
2. Reconstruct the symbolic presence vector deterministically from the derivation
   via `set_symbols()`.

The round-trip `forward $\rightarrow$ (delete symbols) $\rightarrow$ reverse` recovers the original
present positions exactly.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of active word vectors |
| `nDim` | Word vector dimensionality |
| `nVectors` | Codebook size (defaults to nActive) |

**Layer.** No trainable parameters (rule selection is currently random).

**Invertibility.** Reverse deterministically recovers activation from the
derivation tree.

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

**Range.** The forward pass rescales output from `[-1, 1]`
(symbolic activation range) to the original data range via `Data.denormalize()`.
The reverse pass applies `Data.normalize(x, "output")` to map back to `[-1, 1]`
before the reverse linear projection.

**Invertibility.** Reverse uses pseudoinverse; not exactly invertible in general.
