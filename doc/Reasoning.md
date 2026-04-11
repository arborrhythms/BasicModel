# Reasoning System

The reasoning system provides four methods on `BaseModel` for truth-aware
inference, a bidirectional reasoning loop, and a grammar learning mode.
These build on the TruthLayer infrastructure described in
[../../doc/Truth.md](../../doc/Truth.md) and the grammar composition
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

## Ramsified Reverse

The ramsified reverse pass unwinds partition slices from highest order to
lowest, reversing through `SymbolicSpace` and `ConceptualSpace` per
partition, then combines percept estimates across orders.

## Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `<TruthLoss>` | `<training>` in model.xml | 0.0 | Weight for additive truth loss penalty |
| `<conceptualOrder>` | `<architecture>` | 1 | Number of Percept->Concept->Symbol iterations |
| `<ramsified>` | `<architecture>` | false | Enable serial per-concept Pi with L1 sparsity |
| `truthMinMagnitude` | `<SymbolicSpace>` | 0.3 | Minimum activation norm for truth storage |
| `truthMinNovelty` | `<SymbolicSpace>` | 0.5 | Minimum novelty for truth storage |
| `truthMaxInconsistency` | `<SymbolicSpace>` | 0.3 | Maximum inconsistency for truth storage |

## Testing

Unit tests in `basicmodel/test/test_reasoning.py` cover all methods without
requiring a trained model. English-level tests (syllogisms, contrapositives,
semantic equivalence) are `@pytest.mark.xfail` until word identity is
learned through training.
