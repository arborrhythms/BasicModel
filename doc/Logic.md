# Logic

## Overview

This document defines the logic system at three levels:

1. **Subsymbolic (vector / field level)** --- geometry in ConceptualSpace
2. **Symbolic (scalar level in [-1,1])** --- order + polarity in WholeSpace
3. **Rationality** --- propositional truth store built on both

Executable implementations of the subsymbolic and symbolic operators are the
`GrammarLayer` subclasses in `bin/Language.py` and the `Ops.*` kernels in
`bin/Layers.py` they dispatch to (see [Language.md](Language.md)).

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The logic layer states what a BasicModel sentence meaning can be trusted to do
after it has been composed. In an LLM this kind of inferential behavior is often
implicit in next-token priors. Here it is explicit: Formal Concept Analysis gives
the order-theoretic reading of concept support and intent, DisCoCat gives the
typed composition route that builds sentence meanings, and the truth operators
test or accumulate the resulting propositions.

> **Terminology (2026-06-21).**
> One noun per-space: **percept** = a PartSpace/WholeSpace thing (dimensionally
> embedded, EXTENSIONAL; *part* = atom / bottom-up $\sigma$, *whole* = property/region
> / top-down $\pi$ are its two subtypes); **concept** = a ConceptualSpace relation
> tying one part-percept $\leftrightarrow$ one whole-percept (the Concept codebook,
> diachronic); **symbol** = a SymbolSpace thing (0-D, INTENSIONAL, references a
> concept; the symbol codebook). The Section 2 scalar emission is the *symbol*
> that migrates to SymbolSpace, while WholeSpace itself holds whole-percepts.
> Code identifiers below are unchanged (a separate code pass renames them).

---

## 1. Subsymbolic Layer

Objects: vector sets $(B, N, D)$ interpreted as RBF / luminosity fields.

### Operators

- **Union**: `Ops.union` --- the saturating RadMax lattice fold (dual to
  `intersection`), NOT concatenation. The additive combine that briefly held
  the `union` name was renamed `chunk` (2026-07-05) and moved to
  `<PartSpace>` as a structural op (`ChunkLayer`: `left + right`,
  residual-bearing).
- **Intersection**: Co-supported regions (RadMin lattice fold; RBF merge).
- **Negation (affirming)**: `neg(x) = -x`. Antipodal opposition on hypersphere.
- **Non (non-affirming)**: `Ops.non` (the `_non_kernel`; `NonLayer` is the
  grammar-surface form, which complements the leading bivector poles) ---
  bitonic returns the triangular residual $1 - |\mathrm{clamp}(x, -1, 1)|$
  (completing the partition of unity
  $\mathrm{true} + \mathrm{false} + \mathrm{non} = 1$); monotonic is
  $\mathrm{relu}(x - \mathrm{threshold})$ when a threshold is supplied,
  else zero. (`Basis.non` was removed in the 2026-05-01 Step-9 cleanup;
  only the `*Reverse` methods remain on `Basis`.)

> **Meronomy reconciliation (2026-06-11; MeronomySpec §3 wins).** The
> sign flip `−x` is licensed only at the REFERENCE space-role — downstairs
> the sign is occupied (form content) or nonexistent (one-sided
> percepts); only at the reference space-role is it vacant of denotational
> duty and free to carry polarity (`a = −1` on A's row = present,
> certain denial; never negative mass in ground space). `non()`'s
> withdrawal reading is the prasajya move — lowering `|a|` toward 0
> without crossing — gated, never spontaneous. RadMin/RadMax below
> remain signed-scalar kernels at their own space-role; the odds embedding
> is retired from the meronymic path (legacy `PiLayer` keeps it for
> non-meronymic consumers).
- **Parthood** (fundamental): `Basis.part()` / `Ops.part`. The default
  (vector) form is the elementwise projection of $x$ into $y$'s unit
  direction:

$$
\operatorname{part}(x, y) = x \cdot \frac{y}{\lVert y \rVert}
$$

The `scalar=True` form is the clipped cosine projection:

$$
\operatorname{part}(x, y) = \frac{\max(0,\; x \cdot y)}{\lVert x \rVert \cdot \lVert y \rVert}
$$

(The `monotonic` flag is accepted for signature parity but unused by the
part kernel.) The scalar form satisfies Boole's contrapositive trivially
(dot product and norms sign-invariant under joint negation). The full suite
(`whole`, `equal`, `overlap`, `underlap`, `boundary`) composes through
`part`.

---

## 2. Symbolization

> **Historical (design note only).** The norm-projection formula below has
> no implementation: `Basis.symbolize` was removed in the 2026-05-01
> Step-9 cleanup (no live callers), and live symbol emission is the
> WholeSpace codebook snap (the quantize/lookup that names a
> whole-percept's prototype row). The formula is kept as the original
> design reading of "symbolization = norm projection".

For $X \in (B, N, D)$:

$$
s(X) = 2 \cdot \mathrm{mean}(\lVert x_i \rVert) - 1 \quad \in [-1, 1]
$$

Interpretation: $+1$ = strong presence, $0$ = neutral, $-1$ = absence.

---

## 3. Symbolic Layer (Scalars in [-1,1])

`Basis` supports **monotonic** (plain min/max over positive-cone values,
including user-supplied truth-set bivectors) and **bitonic** (sign-aware).
Monotonic forms here; bitonic forms (RadMin, RadMax) in Section 7.
See [Spaces.md "Monotonicity of the lift / lower
chain"](Spaces.md#monotonicity-of-the-lift--lower-chain).

Let $a, b \in [-1,1]$.

| Operator | Bitonic | Monotonic |
|----------|---------|-----------|
| Negation (affirming) | $\operatorname{neg}(a) = -a$ | --- |
| Non (non-affirming) | $\operatorname{non}(a) = 1 - |a|$ (triangular residual) | $\operatorname{non}(a) = \operatorname{relu}(a - \tau)$ when $\tau$ is supplied, else $0$ ($\tau$ is a plain argument, not learned) |
| Union | soft RadMax (LSE-smoothed; Section 7) | $a \cup b = \max(a, b)$ |
| Intersection | soft RadMin (LSE-smoothed; Section 7) | $a \cap b = \min(a, b)$ |

### Parthood as Projection

Parthood is the **fundamental mereological operation**. The scalar
(`scalar=True`) form, for $A, B \in \mathbb{R}^D$:

$$
\operatorname{part}(A, B) = \frac{\max(0,\; A \cdot B)}{\lVert A \rVert \cdot \lVert B \rVert}
\quad \in [0, 1]
$$

(The default vector form is the elementwise projection
$A \cdot (B / \lVert B \rVert)$; see Section 1.) Boole's contrapositive
$\operatorname{part}(A, B) = \operatorname{part}(-B, -A)$ holds trivially.

The full suite composes through `part`:

| Method | Formula |
|--------|---------|
| `whole(A, B)` | `part(B, A)` |
| `equal(A, B)` | $\operatorname{part}(A, B) \cdot \operatorname{part}(B, A)$ |
| `overlap(A, B)` | $0 < \operatorname{equal}(A, B) < 1$ |
| `underlap(A, B)` | $\operatorname{equal}(A, B) = 0$ |
| `boundary(A, B)` | $|\operatorname{part}(A, B) - \operatorname{part}(B, A)|$ (zero under clipped cosine) |

Under clipped cosine, `part` is symmetric, so `equal` reduces to
$\operatorname{part}^2$. Asymmetric classical subsumption is recovered
**relationally** via figure/ground: compare $\operatorname{part}(A, B)$
against $\operatorname{part}(A, \neg B)$. See
[Mereology.md](Mereology.md).

For ternary scalar values (monotonic projection):

| part(a,b) | **+1** | **0** | **-1** |
|-----------|--------|-------|--------|
| **+1**    | 1      | 0     | 0      |
| **0**     | 1      | 1     | 1      |
| **-1**    | 0      | 0     | 1      |

Empty-set conventions (asymmetric): zero is vacuously part of everything
($\operatorname{part}(\emptyset, y) = 1$), but nothing non-empty is part of
the empty set ($\operatorname{part}(x, \emptyset) = 0$;
$\operatorname{part}(\emptyset, \emptyset) = 1$).

---

## 4. Key Insight

- Subsymbolic = geometry
- Symbolic = order + polarity
- Symbolization = norm projection

Cleanly separates representation (vectors) from logic (scalars).

---

## 5. Rationality

Rationality is the propositional logic layer, built on the grammar rules
`part(S, S)` (CS space-role, `PartLayer`) and `isEqual(S, S)` (SS
space-role, `IsEqualLayer`).

### Truth Statements

A **truth statement** is any assertion processed through the full pipeline.
The `TruthLayer` --- owned by the symbol tower, reached as
`self.symbolSpace.truth_layer` --- stores activations **scaled by**
DegreeOfTruth:

$$
\text{stored} = \text{activation} \times \text{degree}
$$

| Degree | Stored Vector | Effect |
|--------|--------------|--------|
| +1 | full activation | attractor |
| 0 < d < +1 | scaled activation | weak attractor |
| 0 | zero vector (inert) | prunable |
| -1 < d < 0 | scaled, negated | weak disperser |
| -1 | negated activation | disperser |

### TruthLayer (symbolSpace.truth_layer)

`TruthLayer` is instantiated in `SymbolSubSpace.__init__` (`SymbolSpace` is
a transparent forwarding container over the held `SymbolSubSpace`
coordinator, so `symbolSpace.truth_layer` reaches it). `WholeSpace.forward`
records activations into the TruthLayer governed by the single continuous
`<truthCriterion>` bar (0 $=$ record every activation, 1 $=$ record none): a
per-cell activation is recorded when its clamped magnitude clears
`truthCriterion`. The same knob also gates learned-concept acceptance in
`ConceptualSpace` (relations admitted into the Concept codebook), so one
continuous knob governs all truth learning. The
binary `<accumulateTruth>` / `<truthMinMagnitude>` switches are **retired**.
Recording now fires wherever `WholeSpace.forward` runs — both normal
training and the `store_truths` gold-ingestion epoch (which drops
`truthCriterion` to `0` to capture every provided gold truth, then restores
it). See [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states) and
[Params.md](Params.md):

- **`record(activation, degree)`** --- store `activation * degree`.
- **`query(activation)`** --- find the closest stored truth; returns
  `(index, cosine similarity)` of the best match (or `None` below the
  threshold). The similarity is signed --- consonance (+) vs. dissonance
  (-) --- not the stored degree.
- **`field(concepts)`** --- project all stored truths into ConceptualSpace as a
  scalar field over concept vectors.

**Batch recording (`record_batch` / `compact`).** `record_batch(activations,
trust, degree)` stages a batch into a bounded pending buffer with per-entry
trust, making no per-cell storage decision inside the compute brick;
`compact(min_trust)` --- called outside the brick, after
forward/backward/step --- drops entries below the trust threshold and
promotes the survivors to the persistent `truths` buffer (its one host sync
per tick stays out of CUDA-graph capture).

**Admission governance.** `conflict_profile(extra=None)` measures the
per-dimension contested mass $\min(T_k, F_k)$ over the absolute store,
optionally with candidate row(s) added (measured, never stored);
`conflict_mass` is its per-dimension max. `admissible(candidate, threshold)`
is the commit-point gate: a candidate belief is admissible iff admitting it
would keep the corpus's conflict mass at or below the threshold.
`preemption_signal(threshold, hysteresis)` is the preattention trigger ---
it fires when the conflict mass exceeds the threshold and stays fired until
the mass falls below $\mathrm{threshold} - \mathrm{hysteresis}$ (chatter
guard); it returns `(mass, fired)`.

**Paraconsistent assessment (`assess`).** `assess()` is the terminal
paraconsistent read of the accumulator: `support` / `conflict` /
`ignorance` in $[0, 1]$, recovered from the affirming/denying poles of the
stored Degrees of Truth ($\mathrm{support} = a_P (1 - a_N)$;
$\mathrm{conflict} = a_P R_N + a_N R_P$;
$\mathrm{ignorance} = (1 - \max(a_P, a_N)) (1 - \max(R_P, R_N))$, where
$a$ are mean and $R$ max pole masses). Keeping *conflict* (the set splits
on a proposition) distinct from *ignorance* (the set is silent on it) is
exactly the degeneracy a scalar $a_P - a_N$ collapse loses.

> **Integration with the meronomy.**
> The absolute truth set (propositions, e.g. `cat <= [animal() & orange() &
> object()]`) and the `RelativeTruthStore` (relations between two ideas, e.g.
> `[cats] <= [furry]`) are **the same two structures** the corpus callosum needs for
> the concept-relation meronomy: a two-code part$\leftrightarrow$whole LUT (absolute) plus relations
> over concept indices (relative) — both live in the Concept codebook. These
> **are integrated**, behind `<ltmConsolidation>`: the unified
> `TernaryTruthStore` is the canonical store — the part$\leftrightarrow$whole LUT *is* the
> absolute table; the concept-taxonomy relations *are* the relative table —
> and `TruthLayer` becomes a compatibility read model over it
> (`attach_ltm` binds the store; `sync_from_ltm` rematerializes the view).
> Taxonomy relations are also learned **explicitly from trusted language**
> ("cats are furry" $\to$ the relation `[cats] <= [furry]`, admitted when
> trusted).

### Truth Field

$$
f(c) = \frac{1}{n} \sum_{i=1}^{n} c \cdot t_i
$$

where $t_i = \text{activation}_i \times \text{degree}_i$. Two kinds of regions:

- **Attractors** ($f \to +1$): concept vectors near +DoT truths.
- **Dispersers** ($f \to -1$): concept vectors near -DoT truths.

### Consonance and Dissonance

1. **Internal consonance** --- stored truths should be mutually consistent.
2. **External consonance** --- incoming statements are evaluated against the
   truth field.

### Propositional Structure

Parthood and equality are grammar operations over activations:

- **`part(S, S)`** (CS space-role) --- `PartLayer.forward` passes the
  encompassing parent `right` through unchanged; the parthood relationship
  between the operands is carried by the codebook geometry (the codebook IS
  the meronymic tree), not by any output scaling.
- **`isEqual(S, S)`** (SS space-role) --- `IsEqualLayer.forward` is
  `torch.maximum(left, right)`, the lattice join on the bivector cone;
  asserted equality shows up as codebook co-location (mutual parthood)
  learned through training.

Together they define a partial order over symbolic activations.

### Truth Accumulation Pipeline

Truth entries arrive as `(text, DoT)` pairs. `store_truths()` stages all texts
on TheData and runs a standard inference epoch via `runEpoch(split="runtime")`.
During forward, `WholeSpace.forward()` records each raw symbolic activation
(degree 1.0). After the epoch, each activation is scaled by its DoT:

$$
t_i = \text{activation}_i \times \text{DoT}_i
$$

Two-phase design (encode all, then scale) ensures all truths are encoded in
the same pipeline context before DoT modulation.

---

## 6. Consistency and Verification

### Internal Consistency (within a TruthSet)

The design reading is a leave-one-out check: for each $t_j$, evaluate the
field produced by the rest of the set at $c_j$:

$$
d_j = f_{\setminus j}(c_j) = \frac{1}{n-1} \sum_{i \neq j} c_j \cdot t_i
$$

- $d_j > 0$: consonant; $d_j \approx 0$: independent; $d_j < 0$: dissonant.

A disperser ($\text{DoT} < 0$) near an attractor ($\text{DoT} > 0$) naturally
produces negative field values, flagging the contradiction.

The $d_j$ field check as written is **not implemented**. The live machinery
is `TruthLayer.consistency()` --- cross-truth contradiction detection
returning a scalar score (and, with `return_report=True`, the list of
contradicting index pairs) --- and `TruthLayer._luminosity_without(j)`, the
leave-one-out luminosity used as the orthogonalization criterion.

### Verification (incoming statement against stored truth)

The design reading: process the statement, then query the field:

$$
v = f(c_a) = \frac{1}{n} \sum_{i=1}^{n} c_a \cdot t_i
$$

| $v$ | Interpretation |
|-----|---------------|
| $v \to +1$ | strong support |
| $v \approx 0$ | no opinion |
| $v \to -1$ | strong contradiction |

There is no `verify()` method; the nearest live surfaces are
`TruthLayer.query` (nearest stored truth), `TruthLayer.assess` (the
paraconsistent support / conflict / ignorance read, Section 5), and
`TruthGroundedReasoner.evaluate` (posture over a `QuerySpec`;
[Reasoning.md](Reasoning.md)). For point lookups,
`symbolSpace.truth_layer.query()` returns `(index, cosine similarity)` of
the single closest truth --- the signed similarity, not the stored degree.

### Logical Entailment (symbolic closure)

Subsymbolic field checks detect *geometric* consistency but not *logical*
entailments. Example: `part(A, B) = +1` and `part(B, C) = +1` entail
`part(A, C)`, but $A$ and $C$ could be geometrically distant. The symbolic
closure layer:

1. **Extract propositions** by decomposing stored SS activations into
   relation + two operands.
2. **Compute closure** via transitivity:
   - $\text{part}(A, B) \wedge \text{part}(B, C) \Rightarrow \text{part}(A, C)$
   - $\text{equals}(A, B) \wedge \text{equals}(B, C) \Rightarrow \text{equals}(A, C)$
   - $\text{equals}(A, B) \wedge \text{part}(A, C) \Rightarrow \text{part}(B, C)$

   Each entailed proposition inherits DoT = min of the premise DoTs.
3. **Inject entailments** as new symbolic activations recorded in the
   TruthLayer.

This closure **already exists**: `TruthGroundedReasoner.is_part`
(`bin/reasoning.py`) walks beam-limited MIN-trust chains --- a chain is only
as true as its weakest hop --- and `materialize` writes a verified chain's
conclusion back as a direct lemma (a `REL_PARTOF` row carrying the
MIN-composed trust), so a later identical query is a direct hit.
`TruthLayer.derive` performs the pairwise transitivity step and
`extrapolate` generalizes it to all two-argument methods, each derived
truth recorded with attenuated DoT inherited from its premises. The
geometric field alone remains sufficient for soft consistency checks and
nearest-truth lookups.

## 7. Radial Operators for Hypersphere Ternary Logic

![Ternary Logic Operators](diagrams/ternary_logic.svg)

RadMin, RadMax, NOT, and NON operate over signed magnitude space `[-1, +1]`.
Sign = direction of assertion; magnitude = strength; zero = **unknown**.
$+1$ = True; $0$ = Unknown (no assertion); $-1$ = False. Magnitude ordering:
$x \subset y$ iff $|x| < |y|$.

> **Soft defaults (2026-05-29).** The DEFAULT bitonic kernels behind
> `Ops.union` / `Ops.intersection` are the LSE-smoothed soft forms
> (`kind='soft'`: `Ops._soft_radmax` / `Ops._soft_radmin`), so both
> operands receive softmax-weighted gradient per cell (hard RadMax routed
> gradient only to the winning operand, zeroing the loser's `.grad`). The
> hard bodies survive as `kind='radial'`; the table below gives the hard
> (radial) semantics, which the soft forms approach as $\tau \to 0$.

| Op | Behavior | Ternary truth table |
|----|----------|---------------------|
| **RadMin** (AND) | Sign disagree $\to$ 0; else $\mathrm{sign}(x) \cdot \min(|x|, |y|)$. Zero absorbing. | `[+1,0,0; 0,0,0; 0,0,-1]` |
| **RadMax** (OR) | Sign disagree $\to$ 0; else $\mathrm{sign}(x) \cdot \max(|x|, |y|)$. Zero transparent ($\mathrm{RadMax}(x,0) = x$). | `[+1,+1,0; +1,0,-1; 0,-1,-1]` |
| **NOT** | $-x$. Involutive; preserves magnitude; $\mathrm{NOT}(0) = 0$. | input $\to$ -input |
| **NON** | $1 - |x|$ (triangular residual). Full indeterminacy at $x = 0$, zero at $x = \pm 1$; completes the partition of unity $\mathrm{true} + \mathrm{false} + \mathrm{non} = 1$. Withdraws assertion (vs. NOT which inverts). | input $\to 1 - |\mathrm{input}|$ |

Both RadMin and RadMax are commutative and collapse to zero on sign
disagreement.

---

## Luminosity

![Luminosity: Truth Coherence Measure](diagrams/luminosity.svg)

> **Meronomy reconciliation (2026-06-11; MeronomySpec §3 rev b wins).**
> There are TWO measures, split as follows. The **truth-set
> luminosity** — what `Mereology.Luminosity(truth_layer=…)` /
> `TruthLayer.luminosity` computes — is now the catuṣkoṭi coverage
> measure over the stored codes: per conceptual dimension the signed
> references split into true/false pole coverage `(T_k, F_k)`
> (elementwise max of `relu(±row)`), and
>
> ```
> luminosity = mean_k[(T_k − F_k) − min(T_k, F_k)]   in [−1, 1]
> ```
>
> — total area weighted by sign, minus the regions where the sign
> differs (contradictory evidence). It is computed **on the codes**
> (no decode pullback) and is order-independent. The meet/GLB form
> below survives only as the **orthogonalization criterion**
> (`TruthLayer._luminosity_without`) and `darkness()`'s mirror — it is
> no longer the truth-set measure. (The diagram shows the meet form.)

The meet-form coherence scalar (orthogonalization criterion):

```
||relu(min(truths))||
```

Element-wise min computes the conjunction (point where all truths agree); relu
removes negative dimensions; L2 norm gives brightness.

- **High luminosity**: truths are coherent and mutually reinforcing.
- **Low luminosity**: truths are sparse, contradictory, or incoherent.

Two roles in the model:

1. **Top-down bias** (NOT implemented): the design has concept input scaled
   by luminosity each conceptual iteration
   (`concept_input * (1 + truthBiasScale * luminosity)`). The
   `<truthBiasScale>` knob is parsed (`architecture.truthBiasScale` $\to$
   `truth_bias_scale`) but never consumed --- a dead knob today.
2. **Loss modification** (live): low luminosity increases training loss via
   `SymbolSubSpace.truth_modulated_loss`, the multiplicative factor
   $\mathrm{totalLoss} \cdot (1 + w_{lum}(1 - \mathrm{lum}) + w_{univ}(1 - u))$
   (see [Reasoning.md](Reasoning.md) "TruthLoss").

### Fusion

Fusion is the **mereological least upper bound** of the stored truth set:

```
fusion = max_i truths[i]
```

In bivector space (paired-index `[p0, n0, p1, n1, ...]`), fusion names the
top-right corner of the smallest hyperrectangle dominating every stored truth
componentwise. Geometric dual of luminosity:

- **Luminosity** = `||relu(min(truths))||` --- GLB / meet, scalar coherence.
- **Fusion** = `max(truths)` --- LUB / join, vector coverage.

Trust (DoT) is already baked into stored truths, so fusion is trust-weighted
automatically.

**Layout caveat.** TruthLayer's paired-index slicing (`[..., 0::2]` for
positive poles, `[..., 1::2]` for negative) assumes a *repeated* bivector
layout. The WholeSpace codebook uses a *leading* bivector plus positional
trailers: `[pos, neg, where..., when...]`. Callers feeding WholeSpace
activations into `luminosity()` must slice the leading 2 dims first
(`acts[..., :2]`); see the `SymbolSubSpace.truth_modulated_loss` call-site
in `bin/Language.py`.

### Supporting measures

- **Consistency** (`isConsistent()`): folds truths into a union via successive
  `Ops.disjunction` (the thin `Basis.disjunction` wrapper was removed in the
  2026-05-01 Step-9 cleanup). In bitonic mode, conflicting +/- assertions
  cancel.
- **Grounding** (`ground()`): finds the minimal subset entailing a query.
- **TruthLoss**: additive penalty for contradictions, measured by union norm
  reduction. See [Reasoning.md](Reasoning.md).
- **Derive**: pairwise mereological inference via `part()`; generalized by
  `extrapolate()` to all two-argument methods.

---

## Semantic Summary

| Operator | Type | Role | Behavior on Contradiction |
|----------|------|------|---------------------------|
| RadMin | Binary | Conjunction (AND) | Collapses to zero |
| RadMax | Binary | Disjunction (OR) | Collapses to zero |
| NOT | Unary | Sign-flipping negation | N/A |
| NON | Unary | Non-affirming negation | N/A |

---

## 8. Cross-references

The level-crossing operations (`lift`, `lower`, `intersection`, `union`),
`Pi`/`Sigma` layer ownership, witness-set / slab-lattice geometry, and `Ops.*`
namespace inventory have moved out of this document.

- **`Ops.*` reference**: [Language.md](Language.md).
- **PiLayer / SigmaLayer ownership** (`forwardPi`/`reversePi`/`forwardSigma`/
  `reverseSigma` aliases): [Spaces.md](Spaces.md).
- **Mereological suite** (`part`, `whole`, `equal`, `overlap`, `underlap`,
  `boundary`, `copart`): [Mereology.md](Mereology.md).
- **Tetralemma bivector layout** (TRUE/FALSE/BOTH/NEITHER corners):
  [Spaces.md](Spaces.md) and [Philosophy.md](Philosophy.md).
