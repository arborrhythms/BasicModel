# Reasoning System

The reasoning system provides four methods on `BaseModel` for truth-aware
inference, a bidirectional reasoning loop, and a grammar learning mode.
These build on the TruthLayer infrastructure described in
[Logic.md](Logic.md) and the grammar composition
described in [Language.md](Language.md).

## Partitioned Symbolic Space

The symbolic dimension is statically partitioned across conceptual orders
using geometric decay. Each order writes only to its designated slice of
`symbolSum`, while reading the full vector as feedback.

```
order 0:  [0,      D//2)       <- 1/2 of symbol_dim
order 1:  [D//2,   3D//4)      <- 1/4
order 2:  [3D//4,  7D//8)      <- 1/8
...
last order: remainder of D
```

This makes the symbolic space **self-describing**: the position of an
activation reveals its conceptual order. Truth methods use
`_activation_order()` to determine a query's order by finding the partition
with the highest energy.

Partition boundaries are precomputed once at model creation via
`MentalModel._order_partitions(symbol_dim, conceptual_order)`.

## Reasoning Methods

### `isConsistent() -> dict`

Analyzes the TruthSet for internal consistency by folding all stored truths
into a single summary vector via successive `Basis.disjunction()` calls.
In bitonic mode, conflicting +/- assertions on the same dimension cancel
to zero, reducing the score.

Returns `{'consistent': bool, 'score': float, 'sites': tensor, 'union_vector': tensor}`.

### `ground(activation, threshold=0.6) -> dict`

Finds the minimal subset of the TruthSet that entails a query activation.
Uses `_activation_order()` to filter truths by compatible partition before
comparison. Falls back to `TruthLayer.derive()` for indirect derivation
when direct grounding is insufficient.

Returns `{'grounded': bool, 'basis': [indices], 'trace': [...], 'confidence': float}`.

### `isTrue(activation) -> float`

Grounds a proposition and returns a scalar Degree of Truth in [-1, 1].
Positive = true, negative = false, zero = unknown. Delegates to `ground()`.

### `extrapolate(grammar, seed_indices, max_new, attenuation) -> dict`

Generalizes `TruthLayer.derive()` to all two-argument grammar methods
(union, intersection, equals, part). For each pair of stored truths,
applies every eligible method and accepts results that preserve or increase
luminosity; rejects those that decrease it.

Accepted truths are recorded at `attenuation * min(DoT_i, DoT_j)`.

Returns `{'added': [indices], 'rejected': [(i, j, rule, delta_lum), ...]}`.

## TruthLoss

An additive loss penalty for false propositions, configured via
`<TruthLoss>` in model.xml (default 0.0 = disabled).

TruthLoss measures the **union norm reduction** when a proposition is
included in the TruthSet union via `Basis.disjunction()`:

```
truth_union = disjunction(all stored truths)
extended    = disjunction(truth_union, new_proposition)
penalty     = max(0, ||truth_union|| - ||extended||)
```

| Case | Effect |
|------|--------|
| Agreeing proposition | Preserves or extends union dimensions -> no penalty |
| Unknown proposition (zero dims) | Passes through -> no penalty |
| Contradicting proposition | Cancels conflicting dimensions -> positive penalty |

DoT weighting is implicit: stored truth vectors carry DoT in their
magnitude, so contradicting a high-DoT truth causes a larger norm drop.

TruthLoss is **additive** and coexists with the existing **multiplicative**
luminosity modulation (`totalLoss * (1 + lum_weight * (1 - luminosity))`).

## Bidirectional Reasoning Loop

`MentalModel.reason(givens, target, direction, max_steps)` implements
iterative reasoning in two directions:

**Forward** (givens -> conclusion): Encode givens into the TruthSet,
extrapolate new truths each step, check `isTrue(target)` until the DoT
exceeds threshold or `max_steps` is reached.

**Reverse** (target -> grounding): Encode target, call `ground()` to find
a minimal basis, extrapolate if insufficient.

Luminosity non-decrease is the validity certificate at each step.

## Grammar Learning

`MentalModel.grammar_learning_step()` learns grammar weights from a
symbolic reconstruction objective:

1. Forward pass produces `symbolSum`
2. Reverse pass reconstructs input
3. Re-encode reconstruction to `symbolSum_hat`
4. Loss = `||symbolSum_hat - symbolSum||^2` (symbolic level, not conceptual)
5. Optional luminosity validity penalty for rules that decrease luminosity

Grammar weights emerge from gradient descent. Paraphrase-invariance holds
because semantically similar sentences snap to nearby codebook entries.

## Ramsified vs Non-Ramsified Architecture

The `<ramsified>` flag selects between two fundamentally different
Sigma-Pi loop implementations.  Both share the same six-space pipeline;
they differ in how the loop body constructs its input, indexes its
layers, and feeds information between iterations.

### Non-Ramsified (default)

The standard path treats the Sigma-Pi loop as a **flat, monolithic
cycle**.  At each conceptual order `t`:

1. **Input construction.**  Percepts and symbol feedback are
   **concatenated** along the vector dimension:
   `concept_input = cat([percepts, sym_feedback], dim=1)`,
   producing a `[B, nPercepts+nSymbols, D]` tensor.  Every order sees
   the full percept set plus the full symbol set.

2. **Sigma** (ConceptualSpace): A single shared layer transforms the
   combined input.

3. **Pi** (SymbolicSpace): A single shared PiLayer projects concepts →
   symbols.  All orders write to the **entire** symbol dimension.

4. **Feedback**: Symbol activation norms are broadcast back to the
   symbol portion of the input for the next iteration.

5. **Reverse**: `ConceptualSpace.reverse` → peel off the symbol
   portion → `SymbolicSpace.reverse`, repeated per order.

The key property: **all conceptual orders share one undifferentiated
symbolic space**.  There is no way to tell, from a symbol vector alone,
which order produced it.

### Ramsified

The ramsified path makes the architecture **hierarchically partitioned**
and **self-describing** via two mechanisms.

**Mechanism 1 — Butterfly merge/unmerge.**  Instead of concatenating
percepts with symbols, the ramsified path progressively compresses the
percept sequence:

```
percepts:  [B, N, D]
order 0:   [B, N/2, D]    ← average adjacent pairs
order 1:   [B, N/4, D]    ← average again
order k:   [B, N/2^(k+1), D]
```

Each merge averages adjacent vector pairs and **caches the difference**
(`left - right`) in `self._merge_diffs` so the reverse pass can
reconstruct the originals exactly.  This is the "butterfly" pattern —
analogous to increasing receptive fields in visual cortex (V1→V2→V4→IT).

Symbol feedback is additive (`x = x + sym_feedback`) rather than
concatenated, and is also halved to match the current level's vector
count.

**Mechanism 2 — Geometric partition of symbol_dim.**  The symbol
dimension `D` is statically sliced by order (see §Partitioned Symbolic
Space above).  Each order writes **only to its slice** via per-level
indexed Pi layers (`self.symbolicSpace[t]`, `self.conceptualSpace[t]`).
ConceptualSpace and SymbolicSpace store `nn.ModuleList`s of per-level
layers when `level_shapes` is provided.

**Forward loop** (`BasicModel.py` `forward()`):

```
for t in range(conceptualOrder):
    x = butterfly_merge(x)           # halve vector count
    x = x + sym_feedback             # additive feedback
    concepts = conceptualSpace[t](x) # per-level sigma
    symbols  = symbolicSpace[t](concepts)  # per-level pi
    sym_feedback = symbols.norm(...)  # for next iteration
```

**Reverse loop** (`BasicModel.py` `reverse()`):

```
x = symbolicSpace[last].reverse(sym_vec)
for t in reversed(range(conceptualOrder)):
    x = conceptualSpace[t].reverse(x)
    x = x - cached_feedback[t]       # undo additive feedback
    x = butterfly_unmerge(x)         # restore vector count
```

The butterfly unmerge uses the cached `_merge_diffs` to recover both
original vectors from each averaged pair — the inverse is exact.

### Why It Matters

Non-ramsified is simpler and sufficient for `conceptualOrder=1`.
Ramsified is required when you want **partition-aware reasoning** —
truth grounding, consistency checks, and extrapolation that respect
which conceptual order a proposition belongs to.  The self-describing
property means the model's symbolic space carries structural metadata
in its geometry, not in external bookkeeping.

## Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `<TruthLoss>` | `<training>` in model.xml | 0.0 | Weight for additive truth loss penalty |
| `<conceptualOrder>` | `<architecture>` | 1 | Number of Percept->Concept->Symbol iterations |
| `<ramsified>` | `<architecture>` | false | Enable per-partition serial Pi layers with butterfly merge/unmerge |
| `truthMinMagnitude` | `<SymbolicSpace>` | 0.3 | Minimum activation norm for truth storage |
| `truthMinNovelty` | `<SymbolicSpace>` | 0.5 | Minimum novelty for truth storage |
| `truthMaxInconsistency` | `<SymbolicSpace>` | 0.3 | Maximum inconsistency for truth storage |

## Contemplative Awareness Methods

Four stub methods on `BaseModel` characterize stages of contemplative
awareness as spatial/computational properties of the model state.  Each
raises `NotImplementedError` — they define the target characterization,
not an implementation.

| Method | Stage | Characterization |
|--------|-------|-----------------|
| `Contiguous()` | One-Pointedness (Shamatha / FA) | Current state occupies a single connected, convex region in PerceptualSpace and a contiguous span in SymbolicSpace. |
| `Continuous()` | Simplicity (Continuity / OA) | Concept states flow continuously without discrete jumps; the Jacobian of the forward map is bounded. |
| `Peaceful()` | One Taste (Emotional Symmetry) | TruthLayer luminosity is uniformly high across all stored propositions; no truth is privileged. |
| `Done()` | Buddhahood (Non-Meditation / Resonance) | The model is a fixed point of its own forward–reverse cycle; reconstruction loss is zero. |

When `thought_free` mode is active, the grammar already enforces the
one-pointedness that `Contiguous()` characterizes (see
[Language.md](Language.md) §Thought-Free Mode).

## Testing

Unit tests in `basicmodel/test/test_reasoning.py` cover all methods without
requiring a trained model. English-level tests (syllogisms, contrapositives,
semantic equivalence) are `@pytest.mark.xfail` until word identity is
learned through training.
