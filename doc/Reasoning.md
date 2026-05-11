# Reasoning System

Four methods on `BaseModel` for truth-aware inference, a bidirectional
reasoning loop, and a grammar learning mode. Builds on the TruthLayer
infrastructure ([Logic.md](Logic.md)) and grammar composition
([Language.md](Language.md)).

## Partitioned Symbolic Space

The symbolic dimension is statically partitioned across conceptual orders
using geometric decay. Each order writes only to its slice of `symbolSum`,
while reading the full vector as feedback.

```
order 0:  [0,      D//2)       <- 1/2 of symbol_dim
order 1:  [D//2,   3D//4)      <- 1/4
order 2:  [3D//4,  7D//8)      <- 1/8
...
last order: remainder of D
```

Makes the symbolic space **self-describing**: position reveals conceptual
order. Truth methods use `_activation_order()` to determine a query's order
by finding the partition with the highest energy. Partition boundaries are
precomputed once at model creation via `BasicModel._order_partitions`.

## Reasoning Methods

### `isConsistent() -> dict`

Analyzes the TruthSet for internal consistency by folding all stored truths
into a single summary via successive `Basis.disjunction()`. In bitonic mode,
conflicting +/- assertions on the same dimension cancel to zero. Returns
`{'consistent': bool, 'score': float, 'sites': tensor, 'union_vector': tensor}`.

### `ground(activation, threshold=0.6) -> dict`

Finds the minimal subset of the TruthSet entailing a query activation. Uses
`_activation_order()` to filter truths by partition. Falls back to
`TruthLayer.derive()` for indirect derivation. Returns
`{'grounded': bool, 'basis': [indices], 'trace': [...], 'confidence': float}`.

### `isTrue(activation) -> float`

Grounds a proposition and returns a scalar Degree of Truth in [-1, 1].
Positive = true, negative = false, zero = unknown. Delegates to `ground()`.

### `extrapolate(grammar, seed_indices, max_new, attenuation) -> dict`

Generalizes `TruthLayer.derive()` to all two-argument grammar methods (union,
intersection, equals, part). For each pair of stored truths, applies every
eligible method and accepts results that preserve or increase luminosity.
Accepted truths recorded at `attenuation * min(DoT_i, DoT_j)`. Returns
`{'added': [indices], 'rejected': [(i, j, rule, delta_lum), ...]}`.

## TruthLoss

Additive loss penalty for false propositions, via `<TruthLoss>` in model.xml
(default 0.0 = disabled).

Measures the **union norm reduction** when a proposition is included in the
TruthSet union via `Basis.disjunction()`:

```
truth_union = disjunction(all stored truths)
extended    = disjunction(truth_union, new_proposition)
penalty     = max(0, ||truth_union|| - ||extended||)
```

| Case | Effect |
|------|--------|
| Agreeing proposition | Preserves/extends union dims → no penalty |
| Unknown proposition (zero dims) | Passes through → no penalty |
| Contradicting proposition | Cancels conflicting dims → positive penalty |

DoT weighting is implicit: stored vectors carry DoT in magnitude, so
contradicting a high-DoT truth causes a larger norm drop.

TruthLoss is **additive** and coexists with **multiplicative** luminosity
modulation (`totalLoss * (1 + lum_weight * (1 - luminosity))`).

## Bidirectional Reasoning Loop

`BasicModel.reason(givens, target, direction, max_steps)`:

- **Forward** (givens → conclusion): Encode givens into TruthSet, extrapolate
  new truths each step, check `isTrue(target)` until DoT exceeds threshold
  or `max_steps` is reached.
- **Reverse** (target → grounding): Encode target, call `ground()` to find
  minimal basis, extrapolate if insufficient.

Luminosity non-decrease is the validity certificate.

## Grammar Learning

`BasicModel.grammar_learning_step()` learns grammar weights from a symbolic
reconstruction objective:

1. Forward pass produces `symbolSum`.
2. Reverse pass reconstructs input.
3. Re-encode reconstruction to `symbolSum_hat`.
4. Loss = `||symbolSum_hat - symbolSum||^2` (symbolic, not conceptual).
5. Optional luminosity validity penalty for rules decreasing luminosity.

Paraphrase-invariance holds because semantically similar sentences snap to
nearby codebook entries.

## Architecture Modes: useButterflies × useGrammar

Two knobs select among Sigma-Pi architectures. Full grammar mode and
butterflies are **mutually exclusive** — butterfly permutations fight
constituency structure; rejected at load time. Shamatha Speech is a target
narrow-grammar mode wired as a DNF-object policy with contiguity checks.

| | **useGrammar="none"** | **useGrammar="shamathaSpeech" target** | **useGrammar="all"** |
|---|---|---|---|
| **useButterflies=false** | Flat shared sigma (default) | DNF object grammar + contiguity | Grammar-directed composition |
| **useButterflies=true** | Pairwise butterfly mixing | TBD shape/policy | excluded |

### Flat (both false)

A single shared PiLayer (P↔C) and SigmaLayer (C↔S) on a concatenated
`[percepts, symbols]` tensor at every conceptual order. All orders share one
undifferentiated symbolic space.

Per iteration `t`:

1. **Input.** `concept_input = cat([percepts, sym_feedback], dim=1)`.
2. **Pi** (`ConceptualSpace.pi`): percepts → concepts via `forwardPi`.
3. **Sigma** (`SymbolicSpace.sigma`): concepts → symbols via `forwardSigma`.
4. **Feedback**: Symbol activation norms broadcast back to the symbol portion.
5. **Reverse**: `reverseSigma` → peel symbol portion → `reversePi`, per order.

Key: all conceptual orders share one undifferentiated symbolic space; no way
to tell from a symbol vector alone which order produced it.

### Butterfly (useButterflies=true, useGrammar="none")

Butterfly-mode layers permute inputs, pack adjacent pairs, apply the layer,
unpack, and merge — halving `N` at each conceptual order while keeping `D`
constant. Merge is internal; the reverse path inverts each stage exactly.
Requires `<reconstruct>symbols</reconstruct>`.

Analogous to V1→V2→V4→IT in visual cortex; pairwise mixing lets information
flow across the slot axis — suitable for tasks like XOR where information at
different slots must collide.

### Grammar-directed (useButterflies=false, useGrammar="all")

Progressive-bottleneck path with external pair-average merge (`_butterfly_merge`
caching `left - right` diffs in `_merge_diffs`), per-level indexed
Sigma/Pi (`conceptualSpace[t]` / `symbolicSpace[t]`), and cached symbol
feedback. Symbol dimension geometrically partitioned per order — gives
**partition-aware reasoning**: truth grounding, consistency, and
extrapolation respect conceptual order.

Forward:

```
for t in range(conceptualOrder):
    x = butterfly_merge(x)               # halve vector count (external)
    x = x + sym_feedback                 # additive feedback
    concepts = conceptualSpace[t](x)     # per-level sigma
    symbols  = symbolicSpace[t](concepts) # per-level pi
    sym_feedback = symbols.norm(...)
```

Reverse:

```
x = symbolicSpace[last].reverse(sym_vec)
for t in reversed(range(conceptualOrder)):
    x = conceptualSpace[t].reverse(x)
    x = x - cached_feedback[t]           # undo additive feedback
    x = butterfly_unmerge(x)             # restore vector count
```

Butterfly unmerge uses cached `_merge_diffs` to recover both originals from
each averaged pair — inverse is exact.

## Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `<TruthLoss>` | `<training>` | 0.0 | Additive truth-loss weight |
| `<conceptualOrder>` | `<architecture>` | 1 | Percept→Concept→Symbol iterations |
| `<useButterflies>` | `<architecture>` | false | Pairwise butterfly mixing with N-halving |
| `<useGrammar>` | `<WordSpace>` | false | Grammar-directed composition |
| `truthMinMagnitude` | `<SymbolicSpace>` | 0.3 | Activation-norm cap driving per-cell trust score in `TruthLayer.record_batch`. Codebook NN lookup at compact time dedupes near-zero/near-duplicate vectors. |

## Contemplative Awareness Methods

Four stubs on `BaseModel` characterizing stages of contemplative awareness
as spatial/computational properties. Each raises `NotImplementedError` —
they define the target characterization, not implementation.

| Method | Stage | Characterization |
|--------|-------|------------------|
| `Contiguous()` | One-Pointedness (Shamatha / FA) | Single connected, convex region in PerceptualSpace; contiguous span in SymbolicSpace |
| `Continuous()` | Simplicity (Continuity / OA) | Concept states flow continuously; Jacobian of forward map is bounded |
| `Peaceful()` | One Taste (Emotional Symmetry) | TruthLayer luminosity uniformly high across stored propositions |
| `Done()` | Buddhahood (Non-Meditation) | Model is a fixed point of forward-reverse; reconstruction loss zero |

Shamatha Speech is the target grammar mode for `Contiguous()`: complete DNF
object grammar plus spatiotemporal contiguity. Every `conjunction` /
`disjunction` over object parts must keep `where()` support connected and
`when()` support continuous. Differs from serial mode — may reduce over all
active percepts at once; rejects scattered object fields, not multi-percept
fields. See
[Language.md](Language.md) §Shamatha Speech Mode and
[plans/2026-04-28-shamatha-speech-contiguity-handoff.md](plans/2026-04-28-shamatha-speech-contiguity-handoff.md).

## Testing

Unit tests in `basicmodel/test/test_reasoning.py` cover all methods without
requiring a trained model. English-level tests (syllogisms, contrapositives,
semantic equivalence) are `@pytest.mark.xfail` until word identity is
learned through training.
