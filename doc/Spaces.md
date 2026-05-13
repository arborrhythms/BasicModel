# Spaces

## Overview

BasicModel is a pipeline of five **spaces** plus a grammar host
(`WordSpace`), each performing a distinct representational
transformation. Data flows forward from raw input to task output; the
reverse pass reconstructs the original input from the symbolic
representation. Two feedback loops connect the pipeline back upward:
`S → C` (per-stage symbolic loopback at order ≥ 1) and `C → P`
(cross-forward subsymbolic loopback). The legacy `SubsymbolicSpace` and
`SyntacticSpace` classes have been retired — the subsymbolic role is
filled by `PerceptualSpace` plus the new C→P feedback, and the grammar
runs from `WordSpace`'s `SyntacticLayer` attached to `SymbolicSpace`
(the canonical grammar host / "calculator").

```
Forward:  InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace
Reverse:  OutputSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace

Feedback:
    SymbolicSpace ──→ ConceptualSpace   (S→C, per stage when bivectorOutput active)
    ConceptualSpace ──→ PerceptualSpace (C→P, cross-forward; bivector-gated)
```

![WikiOracle Space Hierarchy](diagrams/vector_spaces.svg)

---

## Base Class: Space

All spaces inherit from `Space`, which manages:

- **Shape management.** `inputShape` / `outputShape` as `[nObjects, nDim]`.
  Subclasses read dimensions from `TheObjectEncoding`.
- **Codebook / VQ quantization.** When `nVectors > nActive`, a codebook holds
  candidate vectors; top-k selection gives the bottleneck.
- **Reshape flag.** When in/out object counts differ, the `[B, nObj, nDim]`
  tensor is flattened before the next space and restored on the way back.
- **Attention.** Optional `hasAttention=true` reweights objects.
- **`set_sigma` propagation.** Ergodic-mode noise level cascades from the
  top-level model down through every child layer.

### Reset cascade — hard vs. soft

Every space exposes `Reset(batch=None, hard=True)`. The signature is required
(legacy zero-arg fallback removed).

| Call form | Scope | Use |
|-----------|-------|-----|
| `space.Reset()` | All rows | Whole-state wipe (epoch boundary) |
| `space.Reset(batch=b, hard=True)` | Row `b` only | Document boundary |
| `wordSpace.soft_reset(batch=b)` | Row `b` sentence-scoped state | Grammar `<start>` reduction |

Hard-reset clears: parse stack, `_last_svo`, `_stm_fired`, codebook commit
accumulator, discourse history, `serial_cache`, `_ar_embedded`,
`_end_of_stream` for the affected rows.

Soft-reset clears per-sentence working buffers (parse stack rows, `_last_svo[b]`,
category and reconstruction stacks) and re-arms `_stm_fired[b]`. Does **not**
touch discourse history (`InterSentenceLayer` ring buffer) or codebook EMA —
those are document-scoped.

Reset is dispatched from `runEpoch`, never from inside `runBatch` (the pure
compute brick — see [Architecture.md](Architecture.md)).

---

## Normalization and Ranges

| Space | Data Contract | Geometry |
|-------|--------------|----------|
| InputSpace | Data scaled -1..1 for scalars or vector norms | Signed unit interval `[-1,1]` |
| PerceptualSpace | Modal/demuxed (what/where/when encoding). Signed unit-magnitude scalars or vectors. No negation operator | Signed hypercube `[-1,1]^d` |
| ConceptualSpace | Combined/muxed (event encoding). Signed unit-magnitude (tanh-bounded). Event norm on `subspace.activation` | `[-1,1]` per element (tanh) |
| SymbolicSpace | Symbols are percepts. Concept-to-symbol mapping. One symbol encoded at a time | `[0,1]` presence |
| SyntacticSpace | Words are concepts encoding grammar rules. Word tuples + production rules | `(batch, vector, rule)` |
| OutputSpace | Rescaled from activation range to original data range | Data range |

**Data scaling.** `Data` computes global `input_min`/`input_max` and
`output_min`/`output_max` at load time. InputSpace uses `Data.normalize(x,
"input")` to scale to `[-1, 1]`; OutputSpace uses `Data.denormalize(x,
"output")` to restore the original output range.

**Symbols as percepts.** Symbols live in `[0, 1]`; since conceptual activations
range `[-1, 1]`, the mapping is `symbol = (activation + 1) / 2`.
`SubSpace.get_symbols()` / `set_symbols()` perform the conversion.

**Demuxed mode.** When `InputSpace.demuxed=true`, what/where/when components
are stored independently in the SubSpace rather than concatenated. ModalSpace
routes each component through independent PerceptualSpaces; downstream spaces
see an identical muxed tensor via `materialize()`.

---

## Codebook Similarity Metric

`Codebook` wraps `VectorQuantize`. Similarity metric per space:

| Space | Codebook geometry | Stored | Metric | Retrieval |
|-------|------------------|--------|--------|-----------|
| PerceptualSpace | $[-1, +1]^d$ hypercube | Feature *patterns* | Euclidean L2 | $\arg\max_i (x \cdot c_i - \tfrac{1}{2}\|c_i\|^2)$ |
| SymbolicSpace | $[-1, +1]^d$ hypercube (bivector slots for tetralemma) | Symbol *patterns* | Euclidean L2 | $\arg\max_i (x \cdot c_i - \tfrac{1}{2}\|c_i\|^2)$ |
| ConceptualSpace | Unit L2-norm directions (named directions) | Concept *directions*; input magnitude in $[-1, +1]$ encodes belief certainty | Dot product | $\arg\max_i (x \cdot c_i)$ |

### Euclidean (Perceptual / Symbolic)

These codebooks store *what something looks like* — `0.5·v` carries half as
much "of feature v" as `1.0·v`, so the right notion is coordinate-wise
distance. Retrieval expands $\|x - c_i\|^2$:

$$
\|x - c_i\|^2 = \|x\|^2 + \|c_i\|^2 - 2\,(x \cdot c_i)
$$

$\|x\|^2$ is constant across $i$ and drops from argmin:

$$
\arg\min_i \|x - c_i\|^2 = \arg\max_i (x \cdot c_i - \tfrac{1}{2}\,\|c_i\|^2)
$$

`VectorQuantize` keeps $\|c_i\|^2$ in `_b_norms_sq`:

```python
indices = (flat @ codebook.T - 0.5 * b_norms_sq).argmax(dim=-1)
```

One matmul + one broadcast subtract + one argmax. Skips the `sqrt`, the per-row
`||x||²` add, and the cdist autograd plumbing.

### Dot product (Conceptual)

ConceptualSpace concepts are *named directions* in belief space. $x \cdot
c_i$ gives the *signed strength of belief that $x$ affirms concept $i$*:

- $+1$ fully affirms; $0$ orthogonal; $-1$ fully denies

Two consequences:

1. **Codebook must be unit L2-norm.** EMA renormalizes after each update.
2. **Input must NOT be normalized.** The magnitude *is* the certainty signal.
   Cosine similarity would divide it out.

For *ranking*, $x \cdot c_i$ and $\cos(x, c_i) = (x \cdot c_i) / \|x\|$ are
monotone-equivalent (positive constant cancels). Omitting input normalization
preserves certainty and costs less:

```python
# codebook is unit L2-norm (maintained by EMA)
indices = (flat @ codebook.T).argmax(dim=-1)
```

### Configuring the metric

`use_dot_product` is a class attribute on `Codebook` (default `False`). Set it
on a Space subclass to opt in — `ConceptualSpace` does this. The underlying
`VectorQuantize.use_cosine_sim` flag is historical; after the April 2026 perf
pass, input-side normalization is gone, so the effective meaning is "codebook
unit-norm; rank by dot product".

---

## Codebook Uniqueness Contract

Every codebook entry must be **unique under both `WhereEncoding` and
`WhatEncoding`**:

- **`.where` — globally unique positional key.** Enforced structurally via
  the class-level **codebook offset registry** on `WhereEncoding`
  ([`Spaces.py:223-276`](../bin/Spaces.py)). Each codebook calls
  `allocate_codebook_slice(n_vectors)` and gets a contiguous integer offset;
  all codebooks share `div_term = 2π / total_allocated`. Each codebook's
  entries live in disjoint `.where` slices.
- **`.what` — distinct prototype content.** Identical `.what` collapses to the
  same parthood identity (`equal(A, A) = 1`) — a redundant pair the network
  can't distinguish.

Current enforcement:

| Source | Mechanism | Status |
|---|---|---|
| SymbolicSpace codebook | `ImpenetrableLayer` overlap penalty + variance floor; five-relations classifier pushes pairs toward **disjoint** | Active by default |
| ConceptualSpace codebook | `ImpenetrableLayer` available; not yet wired by default | Opt-in |
| PerceptualSpace Lexicon | Cosine-margin pode/antipode SBOW training | Active for trained Lexicons |
| InputSpace vocabulary | Shares PerceptualSpace's Lexicon | Inherited (text); manual (raw) |

`.where` uniqueness is **structural** (enforced at construction); `.what`
uniqueness is **learned** (encouraged by `ImpenetrableLayer` + antipodal
quotient). Together they guarantee the parthood lattice is well-formed.

---

## Lexicon (Projective Unit Ball)

The **Lexicon** ([`bin/Layers.py`](../bin/Layers.py)) backs PerceptualSpace
word embeddings and SymbolicSpace symbol prototypes. Each row is a vector
$w_i$ in the **projective unit ball** — the closed ball $B^D = \{x : \|x\|_2
\le 1\}$ with the **negation identification** $w \sim -w$ realizing real
projective space $\mathbb{RP}^D$.

**Terminology pin** — three notions sometimes conflated:

- **Pode** of $(a, b)$: midpoint $(a + b)/2$; SBOW positive-pair attractor.
- **Wrapped pode**: midpoint via the $\pm$-quotient, $(a - b)/2$; the
  midpoint through *negation* of $b$.
- **Antipode** of a single point $p$: furthest point. On the flat torus
  unique ($\mathrm{wrap}(p + 1)$); on $\mathbb{RP}^D$ **not unique** — the
  maximum-distance set is the orthogonal hyperplane.

So $-w$ is the **negation** of $w$, *not* the antipode.

### Distance and lookup

For $a, b \in B^D$ the projective squared distance is

$$
d_{\mathbb{RP}}^2(a, b) = \min(\|a-b\|_2^2,\; \|a+b\|_2^2)
= \|a\|_2^2 + \|b\|_2^2 - 2\,|\langle a, b\rangle|.
$$

With $\operatorname{pode}(a, b) = (a + b)/2$ and $\operatorname{wpode}(a, b)
= (a - b)/2$, $d_{\mathbb{RP}}(a, b) = 2 \cdot \min(\|a -
\operatorname{pode}\|,\ \|a - \operatorname{wpode}\|)$. The lookup picks
whichever rep of $b$ ($b$ or $-b$) is closer to $a$.

Sorting by smallest $d_{\mathbb{RP}}^2$ = sorting by largest
$\operatorname{score}(x, w_i) = |\langle x, w_i\rangle| - \tfrac{1}{2}\|w_i\|_2^2$.

Implementation: cache `W_norm2 = W.square().sum(-1)` once per optimizer
step; top-k is `(x @ W.T).abs() - 0.5 * W_norm2` followed by `torch.topk` —
dense matmul + abs + broadcast subtract. No $V \cdot D$ outer-product.

The `Lexicon` API:

```python
lexicon = Lexicon(V, D)
lexicon.project_unit_ball_()         # after optimizer.step()
W_index, W_norm2 = lexicon.lookup_index()

# Projective (RP^D) — antipode-aware, default.
idx, dist_sq, scores = Lexicon.topk_rp(x, W_index, W_norm2, k=32)

# Plain L2 — for sites where w and -w are distinct.
idx, dist_sq, scores = Lexicon.topk_l2(x, W_index, W_norm2, k=32)

# Pairwise primitives:
Lexicon.rp_distance_sq(a, b)
Lexicon.rp_similarity(a, b)
Lexicon.rp_pode(a, b)
Lexicon.rp_wrapped_pode(a, b)
Lexicon.rp_closer_rep(a, b)          # sign(<a, b>) * b
```

For $V \gtrsim 10^5$, use `topk_rp_chunked` to bound peak score-tensor size.

### SBOW training: pode (attractor) and antipode (repulsion target)

- **Pode (attractor).** Positive-pair updates pull $a$ and $b$ toward
  $\operatorname{pode}(a, b)$; the gradient picks the closer of $b$ and $-b$
  for shorter-arc attraction.
- **Antipode (balancing repulsion target).** Negative-pair updates push the
  row toward the furthest point. On $\mathbb{RP}^D$ this is a $(D-1)$-sphere,
  so SBOW samples a representative orthogonal direction.

Negative-sampling gradient has two regimes by $\mathrm{sign}\langle a,
b\rangle$: positive case is standard contrastive repulsion along $(a - b)$;
negative case pushes $a$ away from $-b$ along $(a + b)$.

After every optimizer step the trainer calls `lexicon.normalize()`, clipping
$\|w_i\| \le 1$. `W_norm2` should be refreshed when weights change.

Torus primitives (`Lexicon.wrap`, `Lexicon.delta`, etc.) and the
`torus=True` constructor flag remain as **legacy** static methods (the
earlier Lexicon used the flat torus $T^D = [-1, 1)^D$ with wrapped MSE). New
code must use the `rp_*` primitives.

---

## InputSpace

**Role.** Receives the raw source buffer and lifts it into the model's
internal working dimensionality.

**Text mode forward.** Delegates tokenization to `Lex`, producing a span table
of `(start, end, type)`. Each span $\to$ a vector with two components:

- `nWhat` dims — token content, encoded via `Basis` / `Codebook` (the word
  embedding lookup).
- `nWhere` dims — positional information from the character offset.

Result: `[nActive, nWhat + nWhere]` tensor.

**Text mode reverse.** Inverts the span encoding: each vector $\to$ nearest
codebook entry, then spans $\to$ characters via the stored offset table.

**Numeric mode.** Tensor data passes through unchanged; `LiftingLayer` projects
native input dim (e.g. 784 for MNIST) to `nDim`. Non-embedding inputs are
scaled to `[-1, 1]` via the global data min/max.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Sequence length |
| `nDim` | Output dim per vector |
| `nWhere` | Positional dims |
| `nWhen` | Temporal dims |
| `lexer` | Tokenization mode: `"word"` or `"sentence"` |
| `codebook` | Whether input values are discrete |
| `demuxed` | Store what/where/when independently |

**Invertibility.** Always non-invertible; reverse is a separate reconstruction
using the span table.

### Document streaming and `valid_mask`

Documents longer than `nOutput` bytes are not truncated. `TheData` maintains a
per-row cursor `(doc_idx[b], offset[b])` and `next_tick()` returns
`(input, output, hard_eos)` where `input` is a `[B, nOutput]` slab containing
the next ≤`nOutput` bytes from each row's current document. `hard_eos[b]` is
a host-side bool set when row `b`'s cursor exhausts the current document. A
short fill at document end NULL-pads the slab tail; `valid_mask: [B, K] bool`
flips False for padded positions, and state-mutation propagation skips them.

**Cursor universal — trial mode for non-AR data.** `next_tick()` is the single
dispatch for both AR text byte (rolling cursor) and non-AR data (numeric).
In trial mode (`slab_bytes` not set), each tick yields one batch of trials
with `hard_eos = [True] * B`. The runEpoch outer loop drives `ds.next_tick()`
directly for both modes; the DataLoader exists only so existing tests can
grab `loader.dataset`.

`_end_of_stream` is a host-side `list[bool]` diagnostic only; the canonical
hard-reset signal is the cursor's `hard_eos`.

---

## PerceptualSpace

**Role.** Transforms raw input vectors into perceptual representations via
multiplicative interactions. Models prototype-based feature detection.

**Forward operation (log-space linear).**

```
s_i = log((1 + x_i) / (1 - x_i))          # atanh domain transform
z_j = W @ s + b                            # linear in log-multiplicative space
y_j = (exp(z_j) - 1) / (exp(z_j) + 1)     # tanh back to [-1, 1]
```

Addition in log-space corresponds to multiplication of the original features.

**Reverse (invertible=True).** A single `PiLayer(invertible=True)` is shared
for both directions: `_to_mult(y)`, log, `W^{-1}(z - b)`, exp, `_from_mult`.
Matrix inverse via `InvertibleLinearLayer` (LDU).

**Reverse (invertible=False, reversible=True).** Two `PiLayer` instances —
`pi1` for forward, `pi2` for reverse — with independent weights.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Number of active perceptual vectors |
| `nVectors` | Codebook size; enables VQ when > nActive |
| `invertible` | True: shared invertible layer; False: separate pi1/pi2 |
| `codebook` | False -> `.what` is a passthrough `Tensor`; True -> `Codebook` |
| `hasAttention` | Enable attention reweighting |
| `bivectorOutput` | Applies Q2 promotion $(a_P, a_N) = (\max(0, x), \max(0, -x))$ to the per-slot percept event; writes a $[B, N, 2]$ catuskoti bivector to `subspace.activation` |

**Range.** Vectors live in `[-1, 1]^d` (tanh-bounded). No negation operator —
percepts represent feature magnitudes with sign indicating direction.

Under `bivectorOutput=true`, per-slot scalar activation is replaced by a
non-negative paired-index pair $[aP, aN] \in [0, 1]^2$. The signed-sum
`x = sum_d event[..., d]` is split via `aP = max(0, x); aN = max(0, -x)`.
Reverse recovers `aP - aN` and broadcasts uniformly across the input dim
(per-feature detail within a slot is intentionally lossy).

---

## ConceptualSpace

**Role.** Transforms perceptual vectors into abstract concepts via additive
linear operations. Models conceptual hyperplanes that partition perceptual
space.

**Geometry contrast with the Lexicon.** ConceptualSpace's codebook stores
**named directions** (unit-norm $c_i \in S^{D-1}$); input magnitude in
$[-1, +1]$ encodes belief certainty with sign. No antipodal identification —
sign matters.

**Owned layer.**

| Layer | Direction | Math | Notes |
|-------|-----------|------|-------|
| `self.pi` (`PiLayer`) | P $\leftrightarrow$ C | log-domain multiplicative, monotonic | `forwardPi`/`reversePi` aliases |

One PiLayer handles both directions via self-inverse. Two PiLayers
(`pi1`, `pi2`) are constructed when reversible without invertibility.

**Binary forward.** `pi.forward(x, binary=True)` hard-selects the top-2 input
operands by $|x_i|$ via `Ops.top2_select_ste` and zeros the rest before the
log-domain fold. Zero is the multiplicative identity for Pi's AND, so unselected
operands drop out. Backward is straight-through: every input dim retains a
learning signal.

**Forward (P $\to$ C).**

```
m   = (1 + x) / (1 - x)               # (-1,1) -> (0, inf)
y   = tanh(W * log(m) / 2 + b)        # back to (-1, 1)
```

`monotonic` constrains $W \ge 0$; `nonlinear=True` keeps output in `[-1, 1]`.

**Reverse (invertible=True).**

```
l   = log((1 + y) / (1 - y))
x   = tanh(W^{-1} * (l - b) / 2)
```

**Activation carrier.** `ActiveEncoding.nDim = 2`: activation is a 2-dim
bivector `[aP, aN]` per position, encoding tetralemma (*catuskoti*) corners:

| State | `[aP, aN]` |
|-------|------------|
| TRUE (*asti*) | `[1, 0]` |
| FALSE (*nasti*) | `[0, 1]` |
| BOTH (*ubhaya*) | `[1, 1]` |
| NEITHER (*anubhaya*) | `[0, 0]` |

BOTH encodes first-class inconsistency; NEITHER encodes unknown. Operations
obey De Morgan under pole-swap negation $\neg[aP, aN] = [aN, aP]$:

- Conjunction: `[min(aP, bP), max(aN, bN)]`
- Disjunction: `[max(aP, bP), min(aN, bN)]`

See [BuddhistParallels.md](BuddhistParallels.md).

**MASK on `SubSpace._active`.** Two orthogonal per-position tensors:
`activation` (4-valued bivector) and `_active: [B, N, M]` (modality presence
flags). `_apply_mask` is shape-disambiguated:

| Mask shape | Effect |
|------------|--------|
| Aligns with `out.shape[-1]` (feature axis) | Element-wise multiply on output |
| Aligns with `out.shape[-2]` (position axis) | Zero masked rows of `_active`; `materialize()` gates downstream |

### ShortTermMemory

ConceptualSpace also owns `self.stm` — a `ShortTermMemory` instance, a
per-batch stack of unquantized C-tier "ideas" (continuous compositions
of concepts produced by reduce operations). This is distinct from the
sentence-scoped `_stm_fired` flag on `WordSpace` (which is a discourse-
priming single-shot signal, not a working-memory buffer).

| Property | Default | Configurable via |
|---|---|---|
| Capacity | Auto-sized to `<WordSpace><wMax>` (sentence length bound), fallback 8 | `<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>` (explicit override) |
| Storage | `[batch, capacity, concept_dim]` buffer + `[batch]` depth pointers | `persistent=False` (working state, not saved) |
| Cleared on | Hard `Reset` (sentence boundary) | Soft reset leaves it intact |

API: `push(b, idea)`, `pop(b)`, `peek(b, n=0)`, `snapshot(detach=False)`,
`size(b)`, `is_full(b)`, `is_empty(b)`, `clear(b=None)`,
`ensure_batch(batch)`.

The per-word stem inside `BasicModel._forward_stem_per_word` pushes one
post-quantized idea per word; the body's `_chart_compose_at_C` consumes
the buffer via `snapshot()` at every stage. The reverse mirror,
`_chart_generate_from_stm`, fires at the symmetric C-tier point inside
the reverse pipeline. The 7±2 cap is enforced by `wMax`-driven capacity
plus per-row depth pointers.

### Lift/Lower factorization

ConceptualSpace's `self.pi` plays double duty: it's the bare C-forward
operator (no grammar dispatch), AND the substrate that `LowerLayer`
routes through. Similarly, the activated `PerceptualSpace.sigma` is the
substrate that `LiftLayer` routes through (it's also nominally the
sub-percept aggregator, currently a single-role surface).

Both grammar layers factor as:

```
LiftLayer.forward(VP_bivec, NP_bivec):
    VP_c, NP_c = S.codebook.reverse(VP_bivec, NP_bivec, project=True)
    gated_NP   = VP_c * NP_c                       # elementwise mask at C-tier
    out_c      = P.sigma.forward(gated_NP)         # substrate sigma (lift)
    return       S.codebook.forward(out_c, project=True)

LowerLayer.forward(VP_bivec, NP_bivec):
    VP_c, NP_c = S.codebook.reverse(VP_bivec, NP_bivec, project=True)
    gated_NP   = VP_c * NP_c                       # same gate
    out_c      = C.pi.forward(gated_NP)            # substrate pi (lower)
    return       S.codebook.forward(out_c, project=True)
```

The shared `L · U` per-layer LDU basis is reused across every VP. The
per-call gate `VP_c * NP_c` (elementwise multiplicative on the C-tier
prototype × dim grid) is what makes different VPs produce different
transformations — "VP is the mask." No `raw_gate` learnable parameter;
the gating signal comes from VP's codebook content. Different
adjective / verb codebook activations give different outputs from the
same shared matrix.

Sigma vs Pi asymmetry maps directly to lift vs lower:
- Lift uses sigma (additive log-domain expansion) — naturally
  "lifting features onto concepts."
- Lower uses pi (multiplicative log-domain contraction) — naturally
  "lowering concepts into specific percept-realizations."

See [Layers.md](Layers.md#liftlayer--lowerlayer) for the GrammarLayer
specifics.

---

## SymbolicSpace

**Role.** Converts continuous concept activations into a discrete set of
active symbols. The information bottleneck.

**Owned layer.**

| Layer | Direction | Math | Notes |
|-------|-----------|------|-------|
| `self.sigma` (`SigmaLayer`) | C $\leftrightarrow$ S | atanh-domain additive, monotonic | `forwardSigma`/`reverseSigma` aliases |

Symmetric to `ConceptualSpace.pi`.

**Symbols are percepts.** Each symbol represents presence (`1`) or absence
(`0`) of a named entity. Mapping: `symbol = (activation + 1) / 2`.

See [Language.md](Language.md) for the language system design.

**Forward (codebook=True).**

1. Extract concept activation `[B, nConcepts]`.
2. Map through `SigmaLayer(nConcepts, nSymbols, invertible=True, monotonic=True)`.
3. Store as symbolic presence `[0, 1]` via `set_symbols()`.
4. Reshape to `[B, nSymbols, 1]`; codebook produces a one-hot activation
   weighted by similarity.

**Reverse.** Reads symbolic presence via `get_symbols()`, then the PiLayer's
exact inverse maps `[B, nSymbols]` back to `[B, nConcepts]`.

**Key parameters.**

| Parameter | Description |
|-----------|-------------|
| `nActive` | Total symbols (output + reconstruction) |
| `nVectors` | Codebook size (= nSymbols when codebook=true) |
| `codebook` | Enable codebook quantization |
| `bivectorOutput` | Returns per-prototype catuskoti bivector $[B, V_S, 2]$ via `Codebook.forward(..., project=True)`; reverse lifts via cached SVD pseudo-inverse |

**Codebook shape.** One row per symbol at the natural `nDim` width:
`subspace.what.getW().shape == (nVectors, nDim)`. Each row is a free
coefficient vector over conceptual axes. Codebook is *not* unit-norm; symbol
prototypes are free patterns, retrieved via Euclidean L2.

The per-prototype catuskoti bivector `[B, V_S, 2]` lives on
`subspace.activation`, NOT in the codebook. Populated by
`Codebook.forward(input, project=True)` — the **intrinsic snap**:

```
pos[b, n] = sum_v relu(input[b, v] · W[n])
neg[b, n] = sum_v relu(-input[b, v] · W[n])
```

The matching decode `Codebook.reverse(bivec, project=True)` is the cached SVD
pseudo-inverse: C $\to$ S $\to$ C round-trip projects the input onto span(W)
and is a fixed point thereafter (verified by `test/test_idempotent_loop.py`).

The C $\leftrightarrow$ S boundary as **categorization**: calling
`SymbolicSpace.forward` IS the act of *naming* — projecting concept
activation onto the named codebook lattice. Clean prototype match $\to$
TRUE-corner activation; noisy match $\to$ degraded TRUE pole; off-lattice
$\to$ NEITHER. The dialectic loop is snap (synthesis) $\to$ grammar (logic)
$\to$ decode (analysis) $\to$ next pass.

See [BuddhistParallels.md](BuddhistParallels.md), [Logic.md](Logic.md), and
[Mereology.md](Mereology.md).

**Hierarchical mode.** When `<useButterflies>true</useButterflies>` or
`<useGrammar>all</useGrammar>`, BasicModel stores per-stage
ConceptualSpace/SymbolicSpace instances. Symbol dimension is geometrically
partitioned per order. See [Reasoning.md](Reasoning.md).

---

## Monotonicity of the lift / lower chain

Under `<bivectorOutput>true</bivectorOutput>` on P, C, and S, the P $\to$ C
$\to$ S chain is an **order-preserving map on a positive cone**.

Three pieces:

1. **Activations live on the positive cone** `[0, 1]^{2K}` (paired-index
   bivector). PerceptualSpace's Q2 promotion is the bitonic-to-bivector entry
   point. The componentwise partial order $\leq$ *is* the parthood order:
   $s_1 \leq s_2$ componentwise $\Leftrightarrow$ `s1` is part of `s2`.

2. **The Pi / Sigma maps are restricted to $W \geq 0$** entry-wise
   (`monotonic=True` selects `NonNegativeInvertibleLinearLayer` or
   `NonNegativeLinearLayer`):

$$
a \leq b \text{ componentwise} \Longrightarrow Wa \leq Wb \text{ componentwise}
$$

3. **Therefore Pi / Sigma preserve parthood pole-by-pole.**

The bivector layout keeps the contradiction corner `[1, 1]` distinct from the
ignorance corner `[0, 0]` under positive matmul — a single bitonic axis would
let $aP - aN$ cancel under summation.

The `ImpenetrableLayer` regularizer maintains an antichain of same-rank
prototypes, complementing the structural `.where`-uniqueness from the
codebook offset registry. See [Logic.md §Parthood as Projection](Logic.md)
and [BuddhistParallels.md](BuddhistParallels.md).

---

## SyntacticSpace — retired

The standalone `SyntacticSpace` class has been retired. Grammar /
chart / derivation-tree machinery now lives on `WordSpace`, which
attaches a `SyntacticLayer` to `SymbolicSpace` (the canonical grammar
host). The CNF binary-derivation behavior previously documented here
is preserved by `WordSpace.compose` + the chart at S; words are still
concepts encoding grammatical rules, and the derivation is still
stored as word tuples — just on `WordSpace` rather than on a separate
Space.

See [Language.md](Language.md) for the grammar and the chart's per-tier
rule dispatch.

---

## OutputSpace

**Role.** Maps symbolic (or syntactic) vectors to task targets via linear
projection.

**Forward.** `y = W_out * x + b_out`. Always `reshape=True` — the
`[B, nSymbols, symbolDim]` tensor is flattened before projection.

**Reverse.** Pseudoinverse of `W_out`. Text mode snaps each output vector to
the nearest codebook entry (nearest-neighbour lookup).

**`getEmbeddedIO()` override.** Returns raw target dimensions rather than
encoded dimensions, so loss is computed in the output vocabulary space.

**Text mode generation.** Autoregressive: each step's predicted token vector
is snapped to its nearest codebook entry and fed back as input until
`maxResponseLength` or EOS.

**Key parameters.** `nActive`, `nDim`, `nVectors`.

**Layer.** `LinearLayer` with `(bias, temp)` for ergodic mode.

**Range.** Forward rescales `[-1, 1]` to the original data range via
`Data.denormalize()`. Reverse applies `Data.normalize(x, "output")`.

**Invertibility.** Pseudoinverse; not exactly invertible in general.
