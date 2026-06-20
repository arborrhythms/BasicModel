# Parthood, Region Operations, and Symbolic Connectives

A two-level architecture for representing conceptual structure in vector
spaces, with a geometric layer (prototypes, regions, witness sets) and a
symbolic layer (predicates and Boolean connectives). Six operations are
defined across the two levels: **part/whole**, **intersection/union**,
and **conjunction/disjunction**.

## 0. Setup

### Two levels

- **Geometric layer.** Particulars are vectors in a high-dimensional
  space. Categories are *regions* defined by collections of prototype
  vectors. Single-pointedness (śamatha) is encoded as a *convexity*
  constraint on regions: a category is a single bounded region, not a
  union of disjoint pieces.
- **Symbolic layer.** Predicates (concept terms) operate one epistemic
  level higher. Boolean connectives — conjunction, disjunction, negation
  — combine predicates and inherit their semantics from the geometric
  grounding of each predicate's extension.

### Origin as uncertainty

The origin **0** represents epistemic neutrality, not the absence of a
vector. Positive and negative evidence accumulate on a single signed
measure along each direction. A vector orthogonal to a reference axis
projects to **0** on that axis, which means *no evidence either way*
about the property that axis represents — semantic silence, not
disagreement.

### Witness sets

A *witness set* `V = {V₁, …, V_n}` is a finite collection of directions
in the space, treated as the semantically meaningful axes against which
parthood and region membership are evaluated. The witness set plays the
role of a basis but may be non-orthogonal, overcomplete, or scoped to a
semantic region. Choice of `V` is a substantive commitment: it
determines which dimensions the relations care about and which are
treated as W-irrelevant.

---

## 1. Part / Whole

### Definition (per-Whole frame)

For a candidate Whole `W` and candidate Part `P`, decompose `P` in `W`'s
frame:

```
P = p_∥ · Ŵ  +  P_⊥
```

Two scalars summarize the parthood relation:

```
agreement(P, W)    = max(0,  p_∥)
disagreement(P, W) = max(0, −p_∥)
```

Both are non-negative, both bottom out at the origin, and `P_⊥` is set
aside as W-irrelevant (uncertainty rather than evidence).

### Definition (witness-set / elementwise)

Given a witness set `V`, define:

```
P ≼_V W   iff   P · V_i  ≤  W · V_i   for all V_i ∈ V
```

This is elementwise dominance in the (possibly non-orthogonal,
overcomplete) frame given by `V`. It is a genuine partial order:
reflexive, antisymmetric, transitive.

### Properties

| Property        | Per-Whole projection | Witness-set dominance |
|-----------------|----------------------|------------------------|
| Reflexive       | ✓                    | ✓                      |
| Antisymmetric   | ✗ (collinear collapse) | ✓                    |
| Transitive      | ✗ (cones don't nest) | ✓                      |
| Basis-free      | ✓                    | ✗ (V is the basis)    |
| Two scalars     | ✓ (agreement/disagreement) | derived from envelope deltas |

### Transitivity

Crisp transitivity fails for projection-thresholded parthood because
angle composition is super-additive in the worst case:
`cos(α + β) < cos(α) · cos(β)` is not guaranteed, and chains of
"mostly part-of" relations can degrade arbitrarily.

Transitivity is recovered in two ways:

1. **Witness-set dominance** is structurally transitive (the universal
   quantifier over `V` carries through composition). This *is*
   elementwise dominance in the basis given by `V`, and shares its
   commitments.
2. **Graded parthood** via t-norm composition:
   `deg(A ≼ C) ≥ T(deg(A ≼ B), deg(B ≼ C))` for some t-norm `T`. The
   product t-norm corresponds to cosine composition under small-angle
   approximation.

### Inverse

The natural inverse of "P is a part of W" is the converse relation:
"W has P as a part" (the Whole-of relation). This is a notational
inversion; it does not recover `P` from `W` (that recovery is lossy).

In the graded setting, no functional inverse exists in general:
agreement/disagreement scalars do not uniquely determine `P` from `W`,
since the orthogonal residual `P_⊥` is unconstrained by the parthood
relation.

---

## 2. Intersection / Union (Geometric)

### Region representation

A region `R` is a collection of prototype vectors, summarized by its
*envelopes* over the witness set `V`:

```
ℓ_R(V_i) = min over prototypes p ∈ R of  p · V_i
u_R(V_i) = max over prototypes p ∈ R of  p · V_i
```

The region's extension is the slab system:

```
R = { x : ℓ_R(V_i) ≤ x · V_i ≤ u_R(V_i) for all V_i ∈ V }
```

This is the convex hull (in the dual sense — the tightest enclosing
slab system) of the prototype set. When prototypes straddle the origin,
`ℓ_R(V_i) ≤ 0 ≤ u_R(V_i)`, encoding both positive and negative
evidence relative to `V_i`.

### Intersection (meet)

```
ℓ_{R₁ ⊓ R₂}(V_i) = max(ℓ_{R₁}(V_i), ℓ_{R₂}(V_i))
u_{R₁ ⊓ R₂}(V_i) = min(u_{R₁}(V_i), u_{R₂}(V_i))
```

**Tighter envelope on each side, on each witness.** Exact within the
slab formalism (intersection of convex regions is convex). Empty when
`ℓ > u` on any witness.

### Union (join)

```
ℓ_{R₁ ⊔ R₂}(V_i) = min(ℓ_{R₁}(V_i), ℓ_{R₂}(V_i))
u_{R₁ ⊔ R₂}(V_i) = max(u_{R₁}(V_i), u_{R₂}(V_i))
```

**Looser envelope on each side, on each witness.** This is the convex
hull of the set-union, *not* the set-union itself. It is an
over-approximation that includes "bridge" points between the regions.
For a single-pointed (śamatha) representation this is correct: adding
prototypes to a category should expand the region to enclose them.

### Lattice properties

The slab systems over a fixed `V` form a distributive lattice under
`⊓` and `⊔`. Region-parthood `R₁ ⊆ R₂` is envelope dominance:

```
R₁ ⊆_V R₂   iff   ℓ_{R₂}(V_i) ≤ ℓ_{R₁}(V_i)  and  u_{R₁}(V_i) ≤ u_{R₂}(V_i)
                  for all V_i ∈ V
```

Region-parthood is a transitive partial order, even though point-level
parthood by projection alone is not.

### Inverses

- **Intersection inverse.** Given `R₁ ⊓ R₂` and `R₁`, recovery of `R₂`
  is *not* unique: many regions yield the same intersection with `R₁`.
  No functional inverse. The closest constructive notion is *relative
  complement* / region subtraction `R₁ \ R₂`, which is itself
  generally non-convex and exits the slab formalism.
- **Union inverse.** Given `R₁ ⊔ R₂` and `R₁`, recovery of `R₂` is
  also non-unique. The set-difference is non-convex.

Both intersection and union are lossy as binary operations and admit
only partial inverses.

### Outlier sensitivity

Envelopes are determined by extremal prototypes. Standard mitigations:

- **Quantile envelopes.** Replace `min`/`max` with low/high quantiles
  (e.g., 5th/95th percentile projections).
- **Statistical envelopes.** Slab width as `k · σ` of projection
  distribution.

Both preserve lattice structure approximately but trade exact
containment for outlier robustness.

---

## 3. Conjunction / Disjunction (Symbolic)

### Definition

Predicates `P, Q` have extensions `R_P, R_Q ⊆` space.

```
ext(P ∧ Q) = R_P ∩ R_Q     (set intersection, exact)
ext(P ∨ Q) = R_P ∪ R_Q     (set union, generally non-convex)
ext(¬P)    = complement(R_P)  relative to a domain
```

### Conjunction grounds cleanly in geometric meet

`R_P ∩ R_Q` is convex (intersection of convex regions is convex). The
geometric meet `R_P ⊓_V R_Q` over the slab formalism *equals* the
set-theoretic intersection exactly. Conjunction at the symbolic layer
corresponds to intersection at the geometric layer with no information
loss across the level boundary.

### Disjunction does not ground in geometric join

`R_P ∪ R_Q` is *not* convex in general. The geometric join
`R_P ⊔_V R_Q` is the convex hull of the union — a strict
over-approximation that includes points satisfying neither `P` nor `Q`.

```
R_P ∪ R_Q  ⊊  R_P ⊔_V R_Q   (in general)
```

Symbolic disjunction therefore has no faithful representation within
the single-region slab formalism. It is represented as a *set of
regions* `{R_P, R_Q}` with membership defined disjunctively. This is
*disjunctive normal form over regions* — a multi-region object that
lives essentially at the symbolic layer.

### The asymmetry is structural

Convex sets are closed under intersection but not under union, in any
dimension above 1. The same asymmetry appears in topology
(intersections of closed sets are closed; arbitrary unions need not
be), in measure theory (σ-algebras require explicit closure under
union), and in lattice theory generally.

This aligns with apoha / exclusion-based accounts in Indo-Tibetan
epistemology: conjunctive concepts remain within a single
exclusion-determined region, while disjunctive concepts require a
genuinely higher-order construction over predicates.

### Inverses

- **Conjunction inverse.** Negation via De Morgan:
  `P ∧ Q ≡ ¬(¬P ∨ ¬Q)`. The unary inverse on the symbolic layer is
  *negation*, which corresponds to *complement* on the geometric layer
  (relative to a domain) — itself generally non-convex.
- **Disjunction inverse.** Negation via De Morgan:
  `P ∨ Q ≡ ¬(¬P ∧ ¬Q)`. Same unary inverse.

Negation does not preserve convexity. The symbolic layer's Boolean
algebra is closed under all Boolean operations; the geometric layer's
slab lattice is closed under meet and join only. **Crossing the level
boundary is faithful for conjunction, lossy for disjunction, and
breaks convexity for negation.**

---

## 4. Summary Table

| Op | Symbol | Geometric counterpart | Exact? | Inverse |
|---|---|---|---|---|
| Part | `P ≼ W` | projection scalars / witness dominance | depends on definition | Whole-of (converse); lossy as functional inverse |
| Whole | `W ≽ P` | converse of Part | — | Part-of |
| Intersection | `R₁ ⊓ R₂` | tighter envelope (meet) | ✓ exact | none (lossy); region subtraction is non-convex |
| Union | `R₁ ⊔ R₂` | looser envelope (join) | ✗ over-approximates set-union | none (lossy) |
| Conjunction | `P ∧ Q` | geometric meet | ✓ exact | negation (De Morgan) |
| Disjunction | `P ∨ Q` | *not* geometric join; multi-region object | — (no single-region representation) | negation (De Morgan) |

---

## 5. Open Design Questions

1. **Witness set scoping.** Fixed global `V` vs. per-Whole `V_W`.
   Per-Whole witness sets break transitivity along chains unless
   `V_B ⊆ V_C` whenever `B ≼ C`. Is monotone witness growth a
   reasonable design constraint?
2. **Magnitude semantics.** Does parthood-agreement scale with `||P||`
   (containment-style), or only with direction (alignment-style)?
   Depends on whether vector magnitude encodes confidence/salience or
   semantic content in the representation.
3. **Outlier handling.** Quantile vs. σ-multiple envelopes; which
   preserves lattice structure best in practice?
4. **Negation / complement.** Has not been worked out at the geometric
   layer. Complement of a slab system is non-convex; some bounded
   "domain of relevance" is needed to make complement well-defined.
5. **Implication.** Not yet defined. Material implication
   `P → Q ≡ ¬P ∨ Q` would inherit disjunction's multi-region structure.
   A geometric counterpart (e.g., region containment as a graded truth
   value) may be more natural.
