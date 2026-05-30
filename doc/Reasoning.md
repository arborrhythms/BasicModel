# Reasoning System

> **2026-05-29 delta:** the chart / signal-router reverse path now
> passes the tier-local Basis (`subspace.what`) to binary GrammarLayer
> reverses as `basis=tier_basis`. The mereology-guided recommender
> (`Ops._binary_op_recommend`) walks the Codebook's `W` rows to find
> operand pairs $(x_1, x_2)$ such that $\mathrm{op}(x_1, x_2) \approx \mathit{parent}$. Under
> `<codebook>none</codebook>` on SymbolicSpace the recommender has no
> rows to walk and falls back to the lossy `(parent, parent)`
> pseudo-inverse — degrading the reasoning loop's structural recovery
> on multi-stage chart parses. See
> [doc/plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

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
| Agreeing proposition | Preserves/extends union dims $\to$ no penalty |
| Unknown proposition (zero dims) | Passes through $\to$ no penalty |
| Contradicting proposition | Cancels conflicting dims $\to$ positive penalty |

DoT weighting is implicit: stored vectors carry DoT in magnitude, so
contradicting a high-DoT truth causes a larger norm drop.

TruthLoss is **additive** and coexists with **multiplicative** luminosity
modulation (`totalLoss * (1 + lum_weight * (1 - luminosity))`).

## Bidirectional Reasoning Loop

`BasicModel.reason(givens, target, direction, max_steps)`:

- **Forward** (givens $\to$ conclusion): Encode givens into TruthSet, extrapolate
  new truths each step, check `isTrue(target)` until DoT exceeds threshold
  or `max_steps` is reached.
- **Reverse** (target $\to$ grounding): Encode target, call `ground()` to find
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

## Parser And Conceptual Order

Grammar mode is derived from the loaded grammar block. Default-only unary
`pi` / `sigma` rules take the fast path; non-default operator rules enable
grammar-directed parser dispatch.

`conceptualOrder` controls the number of P$\to$C$\to$S stages and the
symbol partition geometry. Higher-order symbols write to later partitions,
so truth grounding, consistency, and extrapolation can respect conceptual
order.

The parser backend is selected by `WordSpace.parserBackend`:

| Backend | Use |
|---------|-----|
| `chart` | Default compatibility path. |
| `stm` | Shift/reduce over typed STM with admissibility masks from grammar signatures. |
| `parallel` | Migration bridge: initializes STM while chart remains authoritative. |

Explicit ordered grammar is preferred:

```
S4 = lift(NP3, VP1)
S5 = lift(NP4, MP1)
```

Here all NPs share base category `NP`; the suffix gives the conceptual
order. Lift and lower are the only syntactic operations that change
argument/return order.

## Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `<TruthLoss>` | `<training>` | 0.0 | Additive truth-loss weight |
| `<conceptualOrder>` | `<architecture>` | 1 | Percept$\to$Concept$\to$Symbol iterations |
| `<parserBackend>` | `<WordSpace>` | chart | Parser backend: `chart`, `stm`, or `parallel` |
| `truthMinMagnitude` | `<SymbolicSpace>` | 0.3 | Gold-`<truth>` recording **gate** (repurposed). Armed at `1` by `store_truths`; the default `0.3` leaves recording off during training. Took over the gating role of the retired `<accumulateTruth>` knob. See [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states). |
| `truthCriterion` | `<architecture>` / `<SymbolicSpace>` | 0.3 | Learn-score acceptance gate for learned relative-sentence relations (accept iff learn-score $\ge$ `truthCriterion`). See [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states). |
| `intraLossWeight` | `<training>` | 0.1 | In-STM next-idea loss $\mathcal{L}_\text{intra}$ weight (`IntraSentenceLayer`). See [STM.md Section 6](STM.md#6-intrasentencelayer). |
| `interLossWeight` | `<training>` | 0.1 | Inter-sentence next-end-state loss $\mathcal{L}_\text{inter}$ weight. See [STM.md Section 11](STM.md#11-inter-sentence-prediction). |
| `routerWireSerial` | `<architecture>` | both | Per-word router-fire gating on the serial path (`per-word` / `boundary` / `both` / `off`). See [STM.md Section 7](STM.md#7-per-word-router-firing). |
| `ltmCapacity` | `<WordSpace>` | 1024 | LTM chain capacity (`InterSentenceLayer` deque of STM end-states). See [STM.md Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states). |

The relative-vs-absolute end-state machinery, the content-aware
learn-score gate, and the tetralemma trust 4-tuple carried on accepted
relative META edges are documented in
[STM.md Section 9](STM.md#9-relative-vs-absolute-end-states).

## Contemplative Awareness Methods

Four stubs on `BaseModel` characterizing stages of contemplative awareness
as spatial/computational properties. Each raises `NotImplementedError` ---
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
`when()` support continuous. Differs from serial mode --- may reduce over all
active percepts at once; rejects scattered object fields, not multi-percept
fields. See
[Language.md](Language.md) Section Shamatha Speech Mode and
[plans/2026-04-28-shamatha-speech-contiguity-handoff.md](plans/2026-04-28-shamatha-speech-contiguity-handoff.md).

## Testing

Unit tests in `basicmodel/test/test_reasoning.py` cover all methods without
requiring a trained model. English-level tests (syllogisms, contrapositives,
semantic equivalence) are `@pytest.mark.xfail` until word identity is
learned through training.
