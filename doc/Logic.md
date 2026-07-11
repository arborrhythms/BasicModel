# Logic

## Overview

This document defines the logic system at three levels:

1. **Subsymbolic (vector / field level)** --- geometry in ConceptualSpace
2. **Symbolic (scalar level in [-1,1])** --- order + polarity in WholeSpace
3. **Rationality** --- propositional truth store built on both

Executable implementations of the subsymbolic and symbolic operators are the
`Method` subclasses in Language.md.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The logic layer states what a BasicModel sentence meaning can be trusted to do
after it has been composed. In an LLM this kind of inferential behavior is often
implicit in next-token priors. Here it is explicit: Formal Concept Analysis gives
the order-theoretic reading of concept support and intent, DisCoCat gives the
typed composition route that builds sentence meanings, and the truth operators
test or accumulate the resulting propositions.

> **Terminology (2026-06-21; see
> [doc/old/2026-06-21-terminology-percepts-concepts-symbols.md](old/2026-06-21-terminology-percepts-concepts-symbols.md)).**
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

- **Union**: `union(A, B) = concat(A, B)`. Combine sets.
- **Intersection**: Co-supported regions (RBF product / merge).
- **Negation (affirming)**: `neg(x) = -x`. Antipodal opposition on hypersphere.
- **Non (non-affirming)**: `Basis.non()` --- bitonic returns zero (complete
  withdrawal); monotonic is `relu(x - threshold)`.

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
- **Parthood** (fundamental): `Basis.part()` --- clipped cosine projection:

$$
\operatorname{part}(x, y) = \frac{\max(0,\; x \cdot y)}{\lVert x \rVert \cdot \lVert y \rVert}
$$

Satisfies Boole's contrapositive trivially (dot product and norms sign-invariant
under joint negation). The full suite (`whole`, `equal`, `overlap`, `underlap`,
`boundary`) composes through `part`.

---

## 2. Symbolization

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
| Non (non-affirming) | $\operatorname{non}(a) = 0$ | $\operatorname{non}(a) = \operatorname{relu}(a - \tau)$ (learnable $\tau$) |
| Union | RadMax (Section 7) | $a \cup b = \max(a, b)$ |
| Intersection | RadMin (Section 7) | $a \cap b = \min(a, b)$ |

### Parthood as Projection

Parthood is the **fundamental mereological operation**. For $A, B \in
\mathbb{R}^D$:

$$
\operatorname{part}(A, B) = \frac{\max(0,\; A \cdot B)}{\lVert A \rVert \cdot \lVert B \rVert}
\quad \in [0, 1]
$$

Boole's contrapositive $\operatorname{part}(A, B) = \operatorname{part}(-B, -A)$
holds trivially.

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
| **+1**    | 1      | 1     | 0      |
| **0**     | 1      | 1     | 1      |
| **-1**    | 0      | 1     | 1      |

Zero is vacuously part of everything (empty-set convention).

---

## 4. Key Insight

- Subsymbolic = geometry
- Symbolic = order + polarity
- Symbolization = norm projection

Cleanly separates representation (vectors) from logic (scalars).

---

## 5. Rationality

Rationality is the propositional logic layer, built on SS grammar rules
`part(S, S)` and `isEqual(S, S)`.

### Truth Statements

A **truth statement** is any assertion processed through the full pipeline.
The `TruthLayer` --- owned by `SymbolSpace`, reached as
`self.wordSpace.truth_layer` --- stores activations **scaled by** DegreeOfTruth:

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

### TruthLayer (SymbolSpace.truth_layer)

`TruthLayer` is instantiated by `SymbolSpace.__init__`. `WholeSpace.forward`
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
- **`query(activation)`** --- find the closest stored truth by cosine similarity.
- **`field(concepts)`** --- project all stored truths into ConceptualSpace as a
  scalar field over concept vectors.

> **Integration with the meronomy ([doc/old/mereological-order-raising.md](old/mereological-order-raising.md)).**
> The absolute truth set (propositions, e.g. `cat <= [animal() & orange() &
> object()]`) and the `RelativeTruthStore` (relations between two ideas, e.g.
> `[cats] <= [furry]`) are **the same two structures** the corpus callosum needs for
> the concept-relation meronomy: a two-code part$\leftrightarrow$whole LUT (absolute) plus relations
> over concept indices (relative) — both live in the Concept codebook. These are to
> be **integrated** — the part$\leftrightarrow$whole LUT *is* the absolute table; the concept-taxonomy
> relations *are* the relative table. Taxonomy relations are also learned
> **explicitly from trusted language** ("cats are furry" $\to$ the relation
> `[cats] <= [furry]`, admitted when trusted).

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

Both parthood and equality are SS operations over symbolic activations:

- **`part(S, S)`** --- containment by clipped-cosine projection. Grammar
  applies it as `score * B`, scaling the whole by parthood degree.
- **`isEqual(S, S)`** --- identity as mutual parthood; delegates to `Basis.equal`.

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

Leave-one-out check: for each $t_j$, evaluate the field produced by the rest
of the set at $c_j$:

$$
d_j = f_{\setminus j}(c_j) = \frac{1}{n-1} \sum_{i \neq j} c_j \cdot t_i
$$

- $d_j > 0$: consonant; $d_j \approx 0$: independent; $d_j < 0$: dissonant.

A disperser ($\text{DoT} < 0$) near an attractor ($\text{DoT} > 0$) naturally
produces negative field values, flagging the contradiction.

### Verification (incoming statement against stored truth)

`verify(statement)` processes the statement, then queries the field:

$$
v = f(c_a) = \frac{1}{n} \sum_{i=1}^{n} c_a \cdot t_i
$$

| $v$ | Interpretation |
|-----|---------------|
| $v \to +1$ | strong support |
| $v \approx 0$ | no opinion |
| $v \to -1$ | strong contradiction |

`wordSpace.truth_layer.query()` returns the single closest truth and its
degree for point lookups.

### Logical Entailment (symbolic closure)

Subsymbolic field checks detect *geometric* consistency but not *logical*
entailments. Example: `part(A, B) = +1` and `part(B, C) = +1` entail
`part(A, C)`, but $A$ and $C$ could be geometrically distant. A symbolic
closure layer would:

1. **Extract propositions** by decomposing stored SS activations into
   relation + two operands.
2. **Compute closure** via transitivity:
   - $\text{part}(A, B) \wedge \text{part}(B, C) \Rightarrow \text{part}(A, C)$
   - $\text{equals}(A, B) \wedge \text{equals}(B, C) \Rightarrow \text{equals}(A, C)$
   - $\text{equals}(A, B) \wedge \text{part}(A, C) \Rightarrow \text{part}(B, C)$

   Each entailed proposition inherits DoT = min of the premise DoTs.
3. **Inject entailments** as new symbolic activations recorded in the
   TruthLayer.

Worth implementing for reasoning rather than retrieval. The geometric field
alone is sufficient for soft consistency checks and nearest-truth lookups.

## 7. Radial Operators for Hypersphere Ternary Logic

![Ternary Logic Operators](diagrams/ternary_logic.svg)

RadMin, RadMax, NOT, and NON operate over signed magnitude space `[-1, +1]`.
Sign = direction of assertion; magnitude = strength; zero = **unknown**.
$+1$ = True; $0$ = Unknown (no assertion); $-1$ = False. Magnitude ordering:
$x \subset y$ iff $|x| < |y|$.

| Op | Behavior | Ternary truth table |
|----|----------|---------------------|
| **RadMin** (AND) | Sign disagree $\to$ 0; else $\mathrm{sign}(x) \cdot \min(|x|, |y|)$. Zero absorbing. | `[+1,0,0; 0,0,0; 0,0,-1]` |
| **RadMax** (OR) | Sign disagree $\to$ 0; else $\mathrm{sign}(x) \cdot \max(|x|, |y|)$. Zero transparent ($\mathrm{RadMax}(x,0) = x$). | `[+1,+1,0; +1,0,-1; 0,-1,-1]` |
| **NOT** | $-x$. Involutive; preserves magnitude; $\mathrm{NOT}(0) = 0$. | input $\to$ -input |
| **NON** | $0$ for all x. Absorbing; idempotent; withdraws assertion (vs. NOT which inverts). | input $\to$ 0 |

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

1. **Top-down bias**: concept input scaled by luminosity each conceptual
   iteration: `concept_input * (1 + truthBiasScale * luminosity)`.
2. **Loss modification**: low luminosity increases training loss.

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
(`acts[..., :2]`); see `truth_loss` call-site in `bin/Spaces.py`.

### Supporting measures

- **Consistency** (`isConsistent()`): folds truths into a union via successive
  `Basis.disjunction()`. In bitonic mode, conflicting +/- assertions cancel.
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
