# Ops vs. Conceptual Operations — Comparison Report

This report compares `Layers.py:Ops` (the static namespace from which
`Spaces.py:Basis` delegates all logic, mereology, and metric methods)
against the six conceptual operations specified in
[Logic.md](Logic.md) Section 8 *Regions, Witness Sets, and the Slab
Lattice* — part / whole, intersection / union, conjunction /
disjunction.

The comparison is documentation-only.  No code changes were made.

**Note (2026-04-24 update).**  The framing of this report has evolved
through several rounds with the user.  The current consensus —
documented in [Logic.md](Logic.md) §8 *Regions, Witness Sets, and the
Slab Lattice* — is the **Cantorian lift / lower** axis: a set is "a
many thought of as a one," so lift (C → S, many → one) is *synthesis*
= union (∨) and lower (S → C, one → many) is *analysis* = intersection
(∧).  At the layer scale, **`SigmaLayer` is the synthesis primitive
(the lift)** and **`PiLayer` is the analysis primitive (the lower)**.
The convexity asymmetry of Logic.md §8 attaches to the level-crossing
direction: **lift is lossy (convex hull over-approximation), lower is
exact**.  This report's *Recommendations* (§7) reflect that framing.

**Operating context.** `Ops` is **agnostic of the producing
subspace**.  It operates on tensors with values in `[-1, 1]`:

- a **single vector** is a **point**;
- a **pair of vectors `(ℓ, u)`** is a **region** (the lower and upper
  envelopes of an axis-aligned slab system).

Whatever upstream stage produced the values has already chosen the
basis (the witness set).  Most commonly that basis is the
ConceptualSpace codebook — percepts and symbols project onto it
before flowing through the rest of the model — but `Ops` doesn't
know or care.  In particular `Ops` is exercised on:

- post-projection `subspace.what` of SymbolicSpace (the bivector
  `.what`, where the upstream C → S projection has aligned witness
  directions to the standard basis); and
- ConceptualSpace activation vectors directly, before any C → S
  demux, when conceptual-space operations are needed.

Same primitives in both cases.

**Target signature.** Each operation should be overloaded to a single
form:

```
Y = f(X1, X2, inverse=False)
```

with `X1`, `X2` either points or regions, and a point treated as a
degenerate region containing the origin `[0]` — the box
`(min(0, x), max(0, x))` elementwise.  The four combinations
(point×point, point×region, region×point, region×region) collapse to
one case — region against region — with the point cases falling out
as the degenerate specialization.  `inverse=False` picks the forward
direction; `inverse=True` picks the configured inverse (converse,
region subtraction, De Morgan, codebook-search witness recovery, or
self-inverse, depending on op).

---

## 1. Inventory of `Ops` methods

`Ops` is a static namespace (`class Ops:` at
[`bin/Layers.py:4060`](../bin/Layers.py)).  Methods grouped by role:

### Tetralemma constants
| Method | Returns |
|---|---|
| `true()` | `1` |
| `false()` | `-1` |
| `unknown()` | `0` |

### Scalar / elementwise primitives
| Method | One-line |
|---|---|
| `positive(x)` | `relu(x)` — non-negative half. |
| `negative(x)` | `−relu(−x)` — non-positive half. |
| `neutral(x)` | `1 − |x|` — distance from a saturated truth. |
| `sign(v)` | `signum` with `sgn(0) = 1` (not 0). |
| `saturate(x)` | `clamp(−1, 1)`, NaN → 0. |
| `threshold(x, τ)` | Zero out `|x| < τ`. |
| `complement(x)` | `sign(x) − x`. |
| `convertSensation(x)` | Affine map `2x − 1` ([0, 1] → [−1, 1]). |
| `minMag / maxMag` | Pick the operand with smaller / larger `|·|`. |
| `error(x, y)` | `‖x − y‖₂`. |
| `isActive(x, τ)` | `|x| ≥ τ`. |
| `isEqual(x, y)` | `torch.equal`. |
| `isReducer(x, y)` | `‖y − x‖₁ < ‖y‖₁` — does `y → x` reduce L1 norm. |

### Tensor primitives
| Method | One-line |
|---|---|
| `pos(x)` | `relu(x)`. |
| `norm(x)` | Last-dim L2 norm. |

### Binary fuzzy logic
| Method | One-line |
|---|---|
| `conjunction(x, y, monotonic)` | Monotonic: `min(x, y)`.  Bitonic: same-sign-min-magnitude (= **RadMin** of [Logic.md](Logic.md) §7). |
| `disjunction(x, y, monotonic)` | Monotonic: `max(x, y)`.  Bitonic: same-sign-max-magnitude with zero-transparent additive correction (= **RadMax**). |
| `negation(x, monotonic)` | Bitonic: `−x`.  Monotonic, even last dim: paired-index pair flip on bivector layout. |
| `non(x, monotonic, threshold)` | Bitonic: `1 − |clamp(x, −1, 1)|` (triangular residual; `true + false + non = 1`).  Monotonic: `relu(x − threshold)` or `0`. |

### Inverse logic operations
| Method | One-line |
|---|---|
| `negationReverse(x, monotonic)` | Self-inverse — delegates to `negation`. |
| `conjunctionReverse(result, y, W, monotonic)` | Codebook-search inverse: find left operand `cb_i` such that `conjunction(cb_i, cb_j) ≈ result` for some `cb_j ∈ W`. |
| `disjunctionReverse(result, y, W, monotonic)` | Same pattern for disjunction. |
| `_binary_op_inverse_impl(...)` | Shared search implementation: forms all `K × K` codebook pairs, picks the L2-closest composed match. |

### In-space algebra (lift / lower)
| Method | One-line |
|---|---|
| `lift(left, right)` | Elementwise product. |
| `liftReverse(result, right)` | `result / (right + ε)`. |
| `lower(left, right)` | Arithmetic mean. |
| `lowerReverse(result, right)` | `2 · result − right`. |

### Axis selectors
| Method | One-line |
|---|---|
| `what(x, nWhat, nWhere, nWhen)` | Keep the leading `nWhat` block; zero the rest. |
| `where(x, ...)` | Keep `[nWhat : nWhat + nWhere]`; zero the rest. |
| `when(x, ...)` | Keep `[nWhat + nWhere :]`; zero the rest. |

### Metric
| Method | One-line |
|---|---|
| `distance(x, y, monotonic, dim)` | Bitonic: angular `(1 − cos) / 2`.  Monotonic: volume-weighted L2 (weights `max(|x|, |y|)` so zero-volume coords don't count). |

### Mereology — the parthood suite
| Method | One-line |
|---|---|
| `part(x, y, monotonic, scalar)` | **Vector form** (default): `x · (y / ‖y‖)` elementwise — projection of `x` into `y`'s unit direction.  **Scalar form**: clipped cosine `max(0, x·y) / (‖x‖ ‖y‖)` ∈ [0, 1] with empty-operand contract `(empty, y) → 1`, `(x, empty) → 0`. |
| `whole(x, y, ...)` | Vector: `(1 − x) · (y / ‖y‖)`.  Scalar: `part(y, x)` — converse. |
| `equal(x, y, ...)` | `part(x, y) · part(y, x)` — mutual parthood. |
| `overlap(x, y, ...)` | Vector: `min(part(x,y), part(y,x))`.  Scalar: `0 < equal < 1` (boolean indicator). |
| `underlap(x, y, ...)` | Vector: `min(whole(x,y), whole(y,x))`.  Scalar: `equal == 0`. |
| `boundary(x, y, ...)` | `|part(x, y) − part(y, x)|` — directional asymmetry; zero under symmetric clipped cosine. |
| `copart(x, y, ...)` | Vector: `y − x`.  Scalar: `1 − part(x, y)` clamped. |

`Basis` (in `Spaces.py`) thinly wraps each of these for the runtime
contract used by SubSpace payloads — formulas live in `Ops`.

---

## 2. Mapping table — six conceptual operations to `Ops`

| # | Conceptual operation | `Ops` method(s) | Match / Diverge | Notes |
|---|---|---|---|---|
| 1 | **Part** (per-Whole projection) | `part(scalar=False)` (vector projection) and `part(scalar=True)` (clipped cosine) | **Match (per-Whole agreement form)**, partial on the two-scalar contract | Implements `agreement = max(0, p_∥)`.  Disagreement is recoverable via `part(-x, y, scalar=True)` or via `whole/copart/boundary`, but not returned in a single call.  Per-Whole projection is what the model uses for point-level parthood; witness-set dominance is **not** implemented as a separate `Ops` method (see §3 below). |
| 2 | **Whole** (converse of Part) | `whole(scalar=True) ≡ part(y, x)`; vector form `(1 − x) · (y / ‖y‖)` | **Match** for converse; vector form is a complement-style transform, not a strict converse | Scalar `whole = part(y, x)` is exactly the converse.  Vector `whole = (1 − x) · ŷ` is a different object — interpretable as "projection of the complement of `x` along `y`," used in `overlap`/`underlap` composition. |
| 3 | **Intersection** (`R₁ ⊓ R₂`) | `conjunction(monotonic=True)` = `min(x, y)`; envelope-pair arithmetic implicit in `TruthLayer.luminosity` (uses `min` over stored truths) | **Match on a single envelope side** | Conceptual meet computes both `(max ℓ, min u)` jointly.  `Ops.conjunction` is `min(x, y)` on activation vectors — which corresponds to the `min u` half of the meet (or to point-meet of degenerate slabs).  The full envelope-pair operation is not in `Ops`; its components live in `TruthLayer` (`luminosity` for `min`-based ℓ, `fusion` for `max`-based u). |
| 4 | **Union** (`R₁ ⊔ R₂`) | `disjunction(monotonic=True)` = `max(x, y)`; `TruthLayer.fusion` is the `max`-over-truths upper envelope | **Match on a single envelope side, accepting over-approximation** | Same envelope-pair caveat as intersection.  Single-region monotonic `max` is the convex-hull / slab-system join — over-approximates the set-union by including bridge points.  The conceptual doc flags this as correct for śamatha single-pointedness. |
| 5 | **Conjunction** (`P ∧ Q`) | Routed through `conjunction` | **Match** | Conjunction grounds cleanly in geometric meet (intersection of convex regions is convex).  Symbolic `P ∧ Q` is computed by the same `conjunction` op as geometric intersection — no level-crossing loss. |
| 6 | **Disjunction** (`P ∨ Q`) | Routed through `disjunction` | **Diverge** from "faithful set-union", **Match** to deliberate single-region over-approximation | The implementation stays in single-region monotonic form (`max`).  No multi-region / DNF representation exists.  This is a known design choice (śamatha convexity), not a bug. |

### Per-Whole vs witness-set dominance: which is in use?

The implementation uses the **per-Whole projection** form for point-level parthood:
`part(scalar=True) = clipped cosine`.  Properties:

- Returns *agreement* in [0, 1]; disagreement available via `part(-x, y)`.
- Symmetric under the clipped-cosine kernel
  (`part(A, B) = part(B, A)` because `cos` is symmetric and norms are
  sign-invariant).  Asymmetric subsumption is recovered relationally
  via figure / ground (`part(A, B)` vs `part(A, ¬B)`); see
  [Mereology.md](Mereology.md).
- **Not transitive** as a hard threshold relation (cones don't nest).
- Boole's contrapositive holds exactly:
  `part(A, B) = part(−B, −A)`.

Witness-set dominance (`P · V_i ≤ W · V_i ∀ V_i ∈ V`) is also
present, but factored differently.  The witness set is the
ConceptualSpace codebook; projection onto that basis happens *upstream*
of `Ops` (in the C → S boundary), so by the time `Ops` runs on
`subspace.what`, the dominance check would be elementwise dominance on
the post-projection bivector activations.  `Ops` does not currently
expose that as a named method, but the building blocks are present:

- `ImpenetrableLayer._pairwise_parthood` ([`bin/Layers.py:2035`](../bin/Layers.py)) builds the
  full `K × K` parthood matrix `P[i, j] = part(cb_i, cb_j, scalar=True)`
  over the codebook.  Pair-level relations (disjoint / part / equal /
  overlap) are then classified per the five-relations table in
  [Mereology.md](Mereology.md).  This is the per-Whole form lifted to
  the whole codebook.
- `TruthLayer.fusion` and `TruthLayer.luminosity` compute the upper
  and lower envelopes over stored prototypes — i.e. the
  hyperrectangle bounds in the post-projection bivector `.what`.
  Region containment (envelope dominance) can be checked componentwise
  from these envelopes but does not have a named method.

So the region-parthood form lives in scattered helpers, not in `Ops`.
The unified-signature redesign (Recommendation 4 below) is the natural
home for it: `part(R1, R2)` over two regions = envelope dominance.

---

## 3. Inverse table — six conceptual inverses to `Ops`

| # | Conceptual inverse | `Ops` inverse (if any) | Match / Diverge | Notes |
|---|---|---|---|---|
| 1 | **Part⁻¹** = Whole (converse).  Functional inverse `P` from `W` is intrinsically lossy. | `whole(scalar=True) = part(y, x)` — the converse. | **Match** for the converse direction. | No functional inverse is attempted (correctly — `P_⊥` is unrecoverable from `W` and `agreement(P, W)`). |
| 2 | **Whole⁻¹** = Part. | `part(scalar=True)`. | **Match.** | Same lossy direction. |
| 3 | **Intersection⁻¹**: region subtraction `R₁ \ R₂` — non-unique, generally non-convex. | None as a region operation.  Closest analogue: `Ops.copart(x, y, scalar=False) = y − x` (point-level). | **Gap (intentional).** | Conceptual model says: no functional inverse.  `copart` is documented as "the part of `y` not accounted for by `x`" — point-level analogue, used for relational analysis, not for region recovery.  Adding a region subtraction would exit the slab formalism. |
| 4 | **Union⁻¹**: same — non-unique, non-convex. | None.  No analogue. | **Gap (intentional).** | Same reasons.  No code currently asks for a union inverse. |
| 5 | **Conjunction⁻¹**: De Morgan via negation: `P ∧ Q ≡ ¬(¬P ∨ ¬Q)`. | `conjunctionReverse(result, y, W)` — codebook search. | **Diverge in kind.** | The implementation is *not* a De Morgan / negation inverse.  It is a witness-recovery search: "find a codebook vector `cb_i` such that `conjunction(cb_i, cb_j) ≈ result` for some `cb_j`."  This is a *lossy pseudo-inverse* that recovers *a* solution `R₂` (or a witness) given the result, not the conceptual inverse via complement.  Useful for invertibility of a forward pass, not for deriving symbolic complements. |
| 6 | **Disjunction⁻¹**: De Morgan via negation. | `disjunctionReverse(result, y, W)` — codebook search. | **Diverge in kind.** | Same pattern.  Codebook-search pseudo-inverse, not a De Morgan inverse. |
|   | **Negation⁻¹** (unary): self-inverse. | `negationReverse = negation`. | **Match.** | Bitonic: sign flip, involutive.  Monotonic: paired-index flip on bivector, also involutive. |

### Lossy-vs-functional distinction in `Ops`

The conceptual doc's category breakdown is honored:

- **Lossy inverses that recover *a* solution**: `conjunctionReverse`,
  `disjunctionReverse`.  Both perform codebook search.
- **Functional inverses (do not exist for binary meet/join)**:
  correctly absent.
- **Unary inverses (negation/complement)**: `negationReverse`
  (involutive).  No region-level complement is implemented — the
  conceptual doc itself flags this as undefined for slab systems.
- **Analytic inverses for in-space algebra**: `liftReverse`
  (division), `lowerReverse` (`2·result − right`).  Not part of the
  six conceptual operations but worth noting — the only true
  functional inverses in `Ops`.

---

## 4. Gaps

Operations or inverses the conceptual model expects that are not in `Ops`.

| # | Gap | Severity | Comment |
|---|---|---|---|
| G1 | **No region-level operations on `(ℓ, u)` pairs.**  Conceptual intersection/union are envelope-pair operations: `(max ℓ, min u)` and `(min ℓ, max u)`.  `Ops` has only single-vector min/max. | Medium | The pair operation is implicit across `TruthLayer.luminosity` (lower) and `TruthLayer.fusion` (upper).  An explicit `Ops.regionMeet(ℓ₁, u₁, ℓ₂, u₂)` / `regionJoin(...)` would make region arithmetic a first-class object instead of two scattered slot operations.  Worth adding if region-level reasoning becomes load-bearing. |
| G2 | **No region-form parthood as a named method.**  Witness-set dominance lifted to regions is `R₁ ⊆ R₂ ↔ ℓ₂ ≤ ℓ₁ ∧ u₁ ≤ u₂` componentwise. | Medium | The witness set is the ConceptualSpace codebook — projection happens upstream of `Ops`.  After projection, region dominance is just elementwise envelope comparison, but it has no named method.  This becomes a first-class operation under the unified `f(X1, X2, inverse=False)` signature when `X1`, `X2` are regions. |
| G3 | **No two-scalar parthood return.**  Conceptual `(agreement, disagreement)` is a pair; `Ops.part(scalar=True)` returns one (the clipped agreement). | Low | Disagreement is recoverable via `part(-x, y)`.  But callers that want a faithful per-Whole frame would benefit from a single call returning both (e.g. `partFrame(x, y) → (agreement, disagreement, residual_norm)`).  Currently no caller asks for it. |
| G4 | **No multi-region / DNF representation for disjunction.**  Conceptual model says faithful `P ∨ Q` requires a set of regions. | Medium (architectural) | Implementation stays in single-region monotonic `max`, accepting the convex-hull over-approximation.  Aligns with śamatha single-pointedness; deferring is a deliberate design choice.  Would touch much more than `Ops` to add. |
| G5 | **No region complement / negation.**  Conceptual model flags this as an open question (slab-system complement is non-convex). | Low | `Ops.negation(monotonic=True)` works on the bivector pair (pole flip), not on a region.  Open by design. |
| G6 | **No graded transitive parthood (`deg(A ≼ C) ≥ T(deg(A ≼ B), deg(B ≼ C))`).** | Low | The conceptual doc proposes this as one of two transitivity recoveries.  Not present in `Ops` — `TruthLayer.derive` is the closest analogue (pairwise inference with attenuated DoT) and operates at the truth-store level, not as a logic primitive. |
| G7 | **No quantile / σ-multiple envelopes.**  Conceptual outlier mitigation. | Low | Hard `min` / `max` everywhere.  Worth adding only if outliers become a measured pain point in the codebook or truth set. |

---

## 5. Divergences (implementation differs from conceptual specification)

`Ops` methods that compute something different from the conceptual
model.  Each divergence comes with a recommendation: doc update if the
implementation is the deliberate design, code change if the
implementation is incidental.

| # | `Ops` method | Conceptual expectation | What it actually does | Recommendation |
|---|---|---|---|---|
| D1 | `Ops.conjunction(monotonic=True)` | "Intersection of regions" — envelope pair `(max ℓ, min u)`. | Pointwise `min(x, y)` — single-envelope side or point-meet of degenerate slabs. | **Doc update.**  Logic.md §8 already notes this; the explicit framing — that single-vector `min`/`max` is the per-slot envelope arithmetic, with the full `(ℓ, u)` pair operation living in `TruthLayer` — should remain documented.  No code change. |
| D2 | `Ops.disjunction(monotonic=True)` | Same pair caveat plus convex-hull over-approximation note. | Pointwise `max(x, y)`. | **Doc update only.**  This is the deliberate śamatha-convexity choice. |
| D3 | `Ops.conjunctionReverse` / `disjunctionReverse` | De Morgan inverse via negation. | Codebook-search witness recovery. | **Doc update.**  Rename or annotate as "witness recovery" / "codebook pseudo-inverse," not "inverse."  The De Morgan path through `negation` is also available and should be cross-referenced.  Optional code: a thin `conjunctionDeMorgan(p, q) = negation(disjunction(negation(p), negation(q)))` for callers that want the symbolic inverse explicitly. |
| D4 | `Ops.part(scalar=True)` returns one scalar | Per-Whole frame returns *two* scalars (agreement, disagreement). | Returns clipped cosine = agreement only.  Disagreement requires `part(-x, y)`. | **Doc update + optional code addition.**  Document that disagreement is `part(-x, y)`.  Optionally add `partFrame(x, y) → (agreement, disagreement)` for callers that want the full frame in one call.  No urgent need. |
| D5 | `Ops.part` symmetric under clipped cosine | Conceptual model treats parthood as directional. | `part(A, B) = part(B, A)` because `cos` is symmetric and norms are sign-invariant. | **Doc update.**  Logic.md and Mereology.md already note this — asymmetric subsumption is recovered relationally via figure/ground.  Already aligned. |
| D6 | `Ops.boundary` is identically zero | Conceptual asymmetry-of-containment scalar. | Zero under clipped cosine because `part` is symmetric. | **Doc update.**  Already noted in code docstring and Mereology.md.  Method retained for bases with asymmetric `part` (e.g. codebook-search based) — no code change. |
| D7 | `Ops.copart(x, y) = y − x` (vector form) | Region subtraction (non-convex, exits slab formalism). | Vector difference — point-level "the part of `y` not in `x`." | **Doc update.**  This is *not* region subtraction; it's a point-level relational helper.  Logic.md should clarify the scope when describing intersection inverses. |
| D8 | `Ops.distance(monotonic=True)` is volume-weighted L2 | Conceptual model is silent on metric in monotonic mode. | Weights each coordinate by `max(|x|, |y|)` so zero-volume coords contribute nothing. | **Doc update.**  This is an additional, principled choice not in the conceptual doc — worth adding to Logic.md §8 (or a new metric section) as the canonical monotonic distance. |
| D9 | `Ops.non` triangular residual | Conceptual model has no separate "non-affirming negation." | Bitonic: `1 − |clamp(x)|` — completes the trinity-of-truth partition `true + false + non = 1`.  Monotonic: `relu(x − threshold)` or `0`. | **Doc update only.**  Already documented in Logic.md §3 / §7.  Worth a one-line cross-reference in §8 noting that the conceptual six operations are extended by `non` for tetralemma support, with no conflict. |

No divergences rise to the level of a code bug.  Every implementation
choice traces back to a documented design commitment (śamatha
convexity, paired-index bivector, codebook-search inverse,
clipped-cosine parthood, etc.) or an open design question still being
worked out.

---

## 6. Specific watch-list checks

Per the handoff document's "Specific things to watch for":

### Witness set representation
`Ops` is agnostic of the witness set: it consumes tensor values in
`[-1, 1]` and treats coordinates, not directions.  The basis is fixed
by whichever upstream stage produced the values — most often the
ConceptualSpace codebook (when post-C → S projection puts activations
in the bivector `.what` of SymbolicSpace), or the conceptual codebook
itself (when operating directly on ConceptualSpace activations).
Either way, no `V` parameter is needed on `Ops` methods because the
basis has already been chosen upstream.  A single vector is a point;
a pair `(ℓ, u)` is a region.  Now documented in Logic.md §8 *Witness
sets*.

### Two-scalar parthood
`Ops.part(scalar=True)` returns a single graded `[0, 1]` value
(agreement only).  No t-norm beyond clipped cosine; no sigmoid.
Disagreement is recoverable as `part(-x, y, scalar=True)`.  See
divergence D4.

### Origin-as-uncertainty
Honored throughout:

- `Ops.part`'s `max(0, ·)` clip maps negative dot product to zero — no
  evidence rather than negative evidence.
- Tetralemma constants `unknown() = 0`.
- `Ops.non` (bitonic) returns 0 only at saturated truth/falsity, and
  positive at the origin — partition-of-unity sums to 1.
- TruthLayer's bivector layout encodes `(0, 0)` as NEITHER (no
  commitment) and `(1, 1)` as BOTH (contradiction) — origin sits in
  the NEITHER corner of the tetralemma, distinguishing "no evidence"
  from "negative evidence."

### Convexity / single-pointedness
Single-region representation throughout.  Disjunction stays at the
geometric layer (`max`) — the over-approximation is intentional under
śamatha single-pointedness.  No multi-region / DNF escape exists.
This is gap G4 and divergence D2 — flagged as deliberate.

### CWCE interaction
`CertaintyWeightedCrossEntropy` ([`bin/Layers.py:4934`](../bin/Layers.py)) penalizes
overconfident predictions.  `Ops.part` returns a graded scalar in
`[0, 1]` (not a boolean) and respects origin-as-uncertainty — CWCE
can consume it sensibly.  No pre-thresholding to booleans.  Confirmed
no conflict.

### Differentiability
Hard `min`/`max` throughout `Ops`.  Sub-differentiable, but produces
sparse gradients — only the extremal codebook row receives gradient
in any min/max-bottlenecked path.  No softmin/softmax/log-sum-exp
smoothing.  This is shared infrastructure with the differentiable
sorting layer (odd-even bubble sort with swap probabilities) —
flagged in Logic.md §8 *Differentiability*.

---

## 7. Recommendations

Prioritized.  Doc and code, with rationale.

### High priority — doc only

1. **Logic.md §8 (landed and revised through 2026-04-24)** — region /
   slab formalism, witness sets, two definitions of parthood,
   convexity asymmetry, **Cantorian lift / lower axis with Sigma =
   synthesis (∨) and Pi = analysis (∧)**, ownership flip, **bivector
   lower into ConceptualSpace** for contradiction preservation,
   refined unified signature with `mode` and `inverse`.  Done.
2. **Cross-reference from Mereology.md** to Logic.md §8 for the
   region-level lift of parthood and the lift / lower framing.
3. **Annotate `conjunctionReverse` / `disjunctionReverse` docstrings**
   to say "codebook-search witness recovery" rather than "inverse,"
   and reference De Morgan as the symbolic inverse via `negation` +
   `disjunction` / `conjunction`.

### High priority — code redesign (now planned)

The full multi-step plan lives in
[plans/2026-04-24-lift-lower-bivector-refactor.md](plans/2026-04-24-lift-lower-bivector-refactor.md).
The headline items it covers:

4. **`Ops.lift` / `Ops.lower` as the canonical level-crossing
   primitives** under the unified signature
   `Y = f(X1, X2=None, mode='AND'|'OR'|'NOT', inverse=False)`.
   `lift` defaults to `mode='OR'` (Cantorian synthesis); `lower`
   defaults to `mode='AND'` (Cantorian analysis).  Region inputs
   `(ℓ, u)` dispatch to envelope arithmetic; point inputs auto-promote
   to degenerate regions containing the origin.

5. **Name-inversion flag for the *current* `Ops.lift` / `Ops.lower`.**
   The current bodies are correct but the labels are swapped relative
   to the Cantorian polarity:
   - Current `Ops.lift = ⊙` (elementwise product, [Layers.py:4309](../bin/Layers.py)) is
     analysis-shaped — it is the pointwise **lower** with `mode='AND'`.
   - Current `Ops.lower = mean` ([Layers.py:4319](../bin/Layers.py)) is
     synthesis-shaped — it is the pointwise **lift** with `mode='OR'`
     (smoothed / averaging variant).
   The plan renames; deprecation aliases keep the old labels live for
   one release.

6. **PiLayer / SigmaLayer ownership flip.**  Synthesis is owned by
   the receiving "one" layer (narrow end of the projection); analysis
   is owned by the producing "many" layer (wide end).
   - PerceptualSpace owns its PiLayer (analysis down to features).
   - ConceptualSpace owns a SigmaLayer (synthesis from P) and a
     PiLayer (analysis down to S).
   - SymbolicSpace owns its SigmaLayer (synthesis from C).
   This inverts the current arrangement at [Spaces.py:5945](../bin/Spaces.py)
   where SymbolicSpace constructs the C → S `PiLayer`.

7. **Bivector activation in ConceptualSpace.**
   `ConceptualSpace.subspace.activation` becomes `[N, 2]` — N
   concepts each carrying a `(pos, neg)` pair, mirroring
   SymbolicSpace's `subspace.what`.  Lower preserves both poles;
   contradiction `[1, 1]` and ignorance `[0, 0]` stay distinguishable
   instead of both collapsing to `0` under signed-sum collapse.
   `tetralemma_balance_penalty` extends from a symbol-only to a
   per-concept regularizer.  Two derivable scalars when needed:
   `signed = aP − aN` and `contradiction = aP · aN`.

   This subsumes earlier separate items:
   - region meet / join over `(ℓ, u)` pairs (was G1),
   - region-form parthood (was G2),
   - point-meets-region operations,
   - explicit forward / inverse selection.

### Medium priority — code, only if a caller asks

8. **`Ops.partFrame(x, y) → (agreement, disagreement)`** — single
   call returning both scalars of the per-Whole frame (D4 / G3).
   Trivial wrapper around two `part` calls.  May fold into the
   unified-signature redesign as the point×point case of `part`.

### High priority — code redesign (continued)

12. **Grammar-driven dispatch.**  Each production in
    [`data/grammar.cfg`](../data/grammar.cfg) carries a
    `(mode, direction)` annotation; the parser's apply-rule step
    dispatches `combine(X1, X2, mode, inverse)` with mode and
    direction supplied by the rule and the operand regions designated
    by the RHS states' **category vectors**.  Without this wiring,
    the syntactic-category region of the codebook (ADJ, N, V, NP, VP,
    ...) gets no structured pressure and the soft superposition stays
    unstructured.  Plan covers it in
    [plans/2026-04-24-lift-lower-bivector-refactor.md](plans/2026-04-24-lift-lower-bivector-refactor.md)
    Step 6.

    Key observations grounded in `grammar.cfg`:

    - **Only ~4 S productions are pure logical operators**:
      `S -> S AND S` (AND, lower), `S -> S OR S` (OR, lift),
      `S -> NOT S` (NOT, lift), and `VP -> NOT VP` (NOT, lift).
    - **Most rules are modification (`AND` at `lower`)**: every
      `X -> modifier X` rule (NP -> AP NP, VP -> ADV VP, AP -> ADJ
      AP, etc.) is a meet on the dimensions the modifier touches.
      The canonical ADJ ∩ N example generalizes here.
    - **NP coordination is `OR` at `lift`** for both surface forms
      ("apples and oranges" / "apples or oranges") — entity-set
      union as Cantorian synthesis.  The propositional AND/OR
      distinction reasserts at the predicate level when the
      coordinated NP is consumed.
    - **Mereological / relational binds extend the trinity**:
      `EQUAL` (copula identification, `S -> NP DEF NP`), `PART`
      (predicative attribution, `S -> NP DEF AP`), `HAS`
      (possession), and `BIND` (generic predicate-argument
      composition for `S -> NP VP`, `VP -> V NP`, `PP -> P NP`,
      etc.).  These route to the existing mereology suite from
      Logic.md §3 / Mereology.md.
    - **Terminal projections** (`NP -> N`, `VP -> V`, etc.) are
      identity lifts that stamp the LHS category vector onto the
      RHS activation — trivial 1-of-1 pooling that types the state.

### Low priority — defer until needed

13. **Quantile / σ-multiple envelopes** (G7).  Defer until outlier
    sensitivity bites in measured behavior.
14. **Multi-region / DNF disjunction** (G4).  Largely subsumed by the
    Sigma-as-synthesis framing: each Sigma-pooled symbol is already
    a `c_a ∨ c_b` proposition in disjunctive form (after hard-pair
    extraction).  Multi-region DNF emerges as a *set* of
    Sigma-symbols.  Tracked under the lift / lower plan rather than
    as a separate gap.
15. **Region complement** (G5).  Open conceptual question; defer.

### Process
10. The handoff document lives at
    [`doc/plans/files/claude-code-handoff.md`](plans/files/claude-code-handoff.md)
    and the conceptual source at
    [`doc/plans/files/parthood-and-region-operations.md`](plans/files/parthood-and-region-operations.md).
    With the conceptual content now in Logic.md §8, the source can be
    archived or removed at the user's preference.  Keeping it under
    `plans/` preserves the handoff history; removing it avoids drift
    if Logic.md is edited.
