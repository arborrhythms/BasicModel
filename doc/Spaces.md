# Spaces

## Overview

BasicModel is organized as a pipeline of six **spaces**, each performing a distinct
representational transformation. Data flows forward from raw input to task output;
the reverse pass reconstructs the original input from the symbolic representation.

```
Forward:  InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> SyntacticSpace -> OutputSpace
Reverse:  OutputSpace -> SyntacticSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace
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
`Reset(batch=None, hard=True)`. The `(batch, hard)` signature is now
**required** — the legacy zero-arg fallback in `_reset_call` was
removed by the brick-vectorization handoff (§8d).

| Call form | Scope | Use |
|-----------|-------|-----|
| `space.Reset()` | All rows | Whole-state wipe (e.g. epoch boundary, manual reload) |
| `space.Reset(batch=b, hard=True)` | Row `b` only | Document boundary, full per-row state wipe |
| `wordSpace.soft_reset(batch=b)` | Row `b`, sentence-scoped state only | Grammar `<start>` reduction completes |

Hard-reset clears: parse stack, `_last_svo`, `_stm_fired`, codebook commit
accumulator, discourse history, `serial_cache`, `_ar_embedded`,
`_end_of_stream` for the affected row(s).

Soft-reset clears the per-sentence working buffers: the parse stack
rows owned by the source row, `_last_svo[b]`, the category and
reconstruction stacks for those rows, and re-arms `_stm_fired[b]` so
the next sentence's discourse-prediction bias fires once. **Does not**
touch discourse history (`InterSentenceLayer` ring buffer) or codebook
EMA — those are document-scoped and form the inter-sentence prior.

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

## Codebook Similarity Metric

`Codebook` (in `bin/Spaces.py`) wraps `VectorQuantize` (in `bin/Layers.py`,
moved out of Spaces.py during the April 2026 perf pass). The choice of
similarity metric for nearest-code retrieval is not one-size-fits-all
across spaces — it follows the geometry of what each codebook stores.

### Per-space metric

| Space | Codebook geometry | What is stored | Metric | Retrieval |
|-------|------------------|---------------|--------|-----------|
| PerceptualSpace | `[-1, +1]^d` hypercube | Feature *patterns* — magnitude carries intensity, no codebook-level negation | Euclidean L2 | `argmax_i (x · cᵢ − ½·‖cᵢ‖²)` via matmul + cached-norm subtract |
| SymbolicSpace | `[-1, +1]^d` hypercube (paired bivector slots for tetralemma corners) | Symbol *patterns* — same as percepts; negation lives in the bivector layout, not in vector arithmetic | Euclidean L2 | `argmax_i (x · cᵢ − ½·‖cᵢ‖²)` via matmul + cached-norm subtract |
| ConceptualSpace | Unit L2-norm directions (concepts are *named directions* in belief space) | Concept *directions*; the *input* magnitude in `[-1, +1]` encodes belief certainty (`+1` known true, `0` unknown, `−1` known false; intermediate magnitudes carry partial belief with sign) | Dot product | `argmax_i (x · cᵢ)` via a single matmul |

### Why Euclidean for Perceptual and Symbolic

These codebooks store *what something looks like* (a pattern of feature
intensities). A vector `0.5·v` carries half as much "of feature v" as
`1.0·v`, and the right notion of "different code" is *coordinate-wise
distance*, not direction. Cosine would conflate `0.5·v` with `1.0·v`
because they share a direction — that loses real information. There is
no codebook-level negation operator on these spaces (negation in
SymbolicSpace is encoded structurally in paired-index bivector slots,
not by negating the vector), so the natural metric is Euclidean L2.

The retrieval is implemented as a matmul + cached-norm subtract, not
`torch.cdist`. Expanding the squared distance:

```
‖x − cᵢ‖² = ‖x‖² + ‖cᵢ‖² − 2·(x · cᵢ)
```

`‖x‖²` is a positive constant across i and drops from the argmin. So:

```
argmin_i ‖x − cᵢ‖² = argmax_i (x · cᵢ − ½·‖cᵢ‖²)
```

`VectorQuantize` keeps `‖cᵢ‖²` in a `[V]` buffer (`_b_norms_sq`,
refreshed in the codebook setter and at the end of each EMA update),
and each forward does:

```python
indices = (flat @ codebook.T - 0.5 * b_norms_sq).argmax(dim=-1)
```

That's one matmul (`[N, D] · [D, V]`) plus one broadcast subtract
plus one argmax. Same FLOPs as `torch.cdist`'s internal mm-trick path,
but skips the `sqrt`, the per-row `‖x‖²` add, and the cdist autograd
plumbing — and gives Inductor a smaller graph to compile.

The naive expansion `((codebook − x)**2).sum(-1)` would be **slower**
because it broadcasts to a full `[N, V, D]` intermediate before reducing
(GBs of memory at PerceptualSpace's `V = 8192`). Both `cdist`'s
mm-trick and the cached-norm matmul above avoid that.

### Why dot product (not Euclidean, not cosine) for Conceptual

ConceptualSpace concepts are *named directions* in belief space. The
codebook entry `cᵢ` is a unit vector pointing toward concept i. An
input `x` projected onto `cᵢ` via the dot product gives the *signed
strength of belief that x affirms concept i*:

- `x · cᵢ = +1` — input fully affirms concept i
- `x · cᵢ =  0` — input is orthogonal (no information about i)
- `x · cᵢ = −1` — input fully denies concept i (strong negative)
- intermediate values — partial belief, with sign preserved

Nearest concept (most-affirmed) is `argmax_i (x · cᵢ)`. Two consequences:

1. **The codebook must be unit L2-norm.** The `EMA` path in
   `VectorQuantize` (the `use_cosine_sim=True` branch) renormalizes the
   running codebook to unit norm after each update, so this invariant
   holds end-to-end. Init-time normalization happens in
   `Codebook.addVectors`.
2. **The input must NOT be normalized.** The input magnitude *is the
   certainty signal* and must survive into the dot product so downstream
   layers see the right strength. Cosine similarity would divide it
   out — wrong for this space.

For *ranking* the concepts (which is what `argmax` cares about),
`x · cᵢ` and `cos(x, cᵢ) = (x · cᵢ) / ‖x‖` are monotone-equivalent
because `‖x‖ ≥ 0` is a positive constant across i and cancels out. So
omitting the input-side normalization preserves the certainty signal
*and* costs less — `O(N·D + V·D)` for the matmul (no per-input
normalize sweep).

This is why a single matmul suffices for ConceptualSpace retrieval:

```python
# codebook is unit L2-norm (maintained by EMA)
indices = (flat @ codebook.T).argmax(dim=-1)
```

### Configuring the metric

The metric is a class attribute on `Codebook` (`use_dot_product`,
default `False`). Each `Space` exposes a parallel class attribute of
the same name; `Space._build_object_basis` (and the other Codebook-
construction sites) mirrors `self.use_dot_product` onto the basis
instance before calling `basis.create()`. To opt a Space into
dot-product retrieval, set `use_dot_product = True` as a class
attribute on the Space subclass — `ConceptualSpace` does this; no XML
knob is required. `Codebook.addVectors` reads `self.use_dot_product`
and propagates it as `VectorQuantize(use_cosine_sim=…)`.

The flag name `use_cosine_sim` on the underlying `VectorQuantize` is
historical (it originally meant "normalize both sides and use
cosine"); after the April 2026 perf pass the input-side normalization
is gone for exactly the reason above, so the operator is now a pure
dot product and the flag's effective meaning is "codebook-is-unit-
norm; rank by dot product".

The default codebook initialization (`Codebook.addVectors`) likewise
branches: dot-product mode L2-normalizes the random init so the EMA
path starts from a valid unit-norm codebook; pattern mode rescales to
the `[-1, +1]` hypercube.

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
`next_tick()` returns `(input, output, hard_eos)` where `input` is a
`[B, nOutput]` slab containing the next `nOutput` or fewer bytes from each row's current
document and `hard_eos` is a host-side `[B] list[bool]` flag set on ticks where a
row's cursor exhausts the current document. A short fill at document end
NULL-pads the slab tail; the stem's `valid_mask: [B, K] bool` flips False for the
padded positions, and downstream state-mutation propagation (codebook EMA,
parse-stack push, truth-layer record) skips them. Concatenating per-tick slabs
across a row's document run reproduces the original bytes byte-exact. The
`_lex_batch` truncation that silently dropped overlong inputs was replaced with
an `assert n_tokens < nObj` in §8g of the brick-vectorization handoff; the cursor
is responsible for sizing the slab to fit the lex's `nObj - 1` content slots.

**Cursor universal — trial mode for non-AR data.** The brick-vectorization
handoff §8e unified the data path: `next_tick()` is the single dispatch
interface for both AR text byte (the rolling-cursor case) and non-AR data
(numeric, non-byte text). In trial mode (`slab_bytes` not set), each tick
yields one batch of trials with `hard_eos = [True] * B` -- the data cursor
aligns with the trial, mirroring the legacy DataLoader contract. The runEpoch
outer loop drives `ds.next_tick()` directly for both modes; the surrounding
DataLoader exists only so existing tests can grab `loader.dataset`.

`_end_of_stream` is a host-side `list[bool]` diagnostic only — never
consulted for control flow — under the brick-vectorization handoff
§8c. The canonical hard-reset signal is the cursor's `hard_eos` list
from `next_tick()`. The scalar/tensor variants of `_end_of_stream`
were removed; the diagnostic list is sized lazily by the AR forward
path and cleared per-row by `Reset(batch=b, hard=True)`. The legacy
`InputSpace.batch_advances_sentence` stub property was deleted in §8a.

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
`<useGrammar>all</useGrammar>`, MentalModel stores per-stage
ConceptualSpace/SymbolicSpace instances in `self.conceptualSpaces` and
`self.symbolicSpaces`. Butterfly mode passes butterfly-aware Pi/Sigma
layers into those spaces. The symbol dimension is geometrically
partitioned so each order writes only to its designated slice.  The
planned `shamathaSpeech` mode is a narrow DNF-object policy rather than
the full grammar hierarchy.  See [Reasoning.md](Reasoning.md) Section
Architecture Modes.

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

The round-trip `forward -> (delete symbols) -> reverse` recovers the original
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
