# Mereology

Single-page reference for the mereological grammar: parthood as the
fundamental operation, the five mereological relations, and the
`ImpenetrableLayer` regularizer that enforces them on the symbol
codebook.

---

## Parthood as Clipped Cosine Projection

Parthood is the single fundamental operation of the grammar.  Every
other mereological relation is defined in terms of it.

For two concepts $A, B \in \mathbb{R}^D$:

$$
\mathrm{part}(A, B) = \frac{\max(0,\ A \cdot B)}{|A|\,|B|}
$$

This clipped cosine projection lies in $[0, 1]$ and satisfies Boole's
contrapositive trivially:

$$
\mathrm{part}(A, B) = \mathrm{part}(-B, -A)
$$

because $(-B) \cdot (-A) = A \cdot B$ and norms are sign-invariant.

**Empty-operand contract.** When $|A|$ or $|B|$ is near zero, `part`
returns 1.0 — the empty set is part of everything, matching the legacy
semantics.

---

## The Full Suite

Every member of the suite is expressible through `part` (plus the
existing bitonic `conjunction` / `disjunction` on `Basis` as helpers):

| Method              | Formula                                      | Interpretation |
|---------------------|----------------------------------------------|----------------|
| `part(A, B)`        | $\max(0, A \cdot B) / (|A|\,|B|)$            | Fundamental. A is part of B. |
| `whole(A, B)`       | $\mathrm{part}(B, A)$                        | A contains B. |
| `equal(A, B)`       | $\mathrm{part}(A, B) \cdot \mathrm{part}(B, A)$ | Mutual parthood, $[0, 1]$. |
| `overlap(A, B)`     | $0 < \mathrm{equal}(A, B) < 1$               | Strictly-partial mutual parthood (boolean indicator). |
| `underlap(A, B)`    | $\mathrm{equal}(A, B) = 0$                   | No mutual parthood (boolean indicator). |
| `boundary(A, B)`    | $|\mathrm{part}(A, B) - \mathrm{part}(B, A)|$ | Asymmetry of containment (zero under clipped cosine; kept for bases with asymmetric `part`). |

---

## Region Partition

`equal(A, B)` partitions the mutual-parthood scalar into three disjoint
regions:

| Region    | Condition                     |
|-----------|-------------------------------|
| Underlap  | $\mathrm{equal} = 0$          |
| Overlap   | $0 < \mathrm{equal} < 1$      |
| Identity  | $\mathrm{equal} = 1$          |

Under clipped cosine, `part` is symmetric, so the three regions collapse
to a single scalar classifier.  For a future `Basis` with asymmetric
`part` (e.g. codebook-search based), the general five-relation table
applies — see the classification below.

**Five-relation table** (general case, for thresholds $\tau \approx 1$
and $\epsilon \approx 0$):

| Relation        | Condition                                  |
|-----------------|--------------------------------------------|
| `disjoint(a,b)` | $P[a,b] < \epsilon$ and $P[b,a] < \epsilon$ |
| `part(a,b)`     | $P[a,b] > \tau$ and $P[b,a] < \epsilon$     |
| `part(b,a)`     | $P[a,b] < \epsilon$ and $P[b,a] > \tau$     |
| `equal(a,b)`    | $P[a,b] > \tau$ and $P[b,a] > \tau$         |
| `overlap(a,b)`  | Both partial (neither $> \tau$ nor $< \epsilon$) |

**Asymmetric subsumption.** Since `part` is symmetric under clipped
cosine, classical asymmetric subsumption ("A is a subset of B, but not
vice versa") is *not* encoded in the raw magnitude of `part`.  It is
recovered **relationally** via figure / ground: compare
$\mathrm{part}(A, B)$ against $\mathrm{part}(A, \neg B)$.  This is
intentional — it is what makes Boole's contrapositive hold exactly.

---

## Mereological Fusion

Fusion of a truth set is the **least upper bound** (LUB / join) over
stored truth vectors:

$$
\mathrm{fusion} = \max_i\, \mathrm{truths}[i]
$$

(elementwise max).  In bivector space, the fusion vector names the
top-right corner of the axis-aligned bounding hyperrectangle
dominating every stored truth.  Fusion is the geometric dual of
luminosity: LUB (join) vs GLB (meet).

Degree of Truth is already baked into each stored truth
(`stored[i] = activation_i * degree_i` in `TruthLayer.record`), so
fusion is trust-weighted automatically.

See [Logic.md](Logic.md) section **Fusion** for the full discussion
and the leading-bivector / paired-index layout caveat.

---

## ImpenetrableLayer: Five-Relations Regularizer

`ImpenetrableLayer` is a regularizer over the SymbolicSpace symbol
codebook.  It classifies each ordered pair of codebook rows $(i, j)$
into one of the five mereological relations above (using
`Basis.part` on the rows), and penalizes partial overlap when paired
with a trust mismatch.  This is the learned half of the [Codebook
Uniqueness Contract](Spaces.md#codebook-uniqueness-contract):
`.what` content must distinguish every prototype so the parthood
lattice stays well-formed (the structural `.where`-uniqueness is
enforced separately by the codebook offset registry).

### Penalty

$$
\mathcal{L} = \mathrm{overlap\_weight} \cdot
   \frac{\sum_{i \neq j} \mathrm{overlap}(i, j)
                        \cdot |\mathrm{trust}(i) - \mathrm{trust}(j)|}
        {K(K-1)}
   + \mathrm{variance\_floor\ term}
$$

where the overlap strength is damped to zero as the pair approaches
`equal`:

$$
\mathrm{overlap\_strength}(i, j) =
   \min(P[i,j],\ P[j,i]) \cdot
   \left( 1 - \max(P[i,j],\ P[j,i])^k \right)
$$

with $k = \mathrm{equal\_suppression}$ (default 4.0).  This makes the
penalty zero for `disjoint`, `equal`, and `part` (strict) relations,
and active only for the `overlap` region.

A separate `variance_floor` term guards against row collapse by
penalizing when the per-dim standard deviation of codebook rows falls
below a floor.

### Trust

Trust is per-row usage frequency, derived from the Codebook's
vector-quantization EMA counts:

$$
\mathrm{trust}(i) = \frac{\mathrm{cluster\_size}[i]}
                         {\sum_j \mathrm{cluster\_size}[j]}
$$

When VQ is absent (e.g. the basis is a `Tensor` rather than a
`Codebook`), trust falls back to $\|cb[i]\| / \max_j \|cb[j]\|$.

### Diagnostics

After `forward()`:

- `last_overlap_loss` — scalar overlap-penalty contribution.
- `last_variance` — scalar variance-floor contribution.
- `last_relation_counts` — dict with counts of each relation
  (`disjoint`, `part_ij`, `part_ji`, `equal`, `overlap`) summing to
  $K(K-1)$.

### Configuration

XML knobs (under the SymbolicSpace config section):

| Knob                    | Default | Meaning |
|-------------------------|---------|---------|
| `overlapWeight`         | 0.1     | Weight of the overlap $\times$ trust-diff penalty |
| `varianceFloor`         | 0.01    | Minimum per-dim std; below this triggers the variance penalty |
| `fullPartThreshold`     | 0.9     | $\tau$: part score above this is "full part" |
| `disjointThreshold`     | 0.1     | $\epsilon$: part score below this is "disjoint" |
| `equalSuppression`      | 4.0     | $k$: damping exponent near the equal corner |

---

## Why This Design

- **Parthood as projection.**  One formula, Boole-contrapositive
  exact, continuous in $[0, 1]$, and the full suite composes on it.
  The old composite formula `conjunction(1 - dist(x, x \cap y), 1 -
  dist(y, x \cup y))` mixed set-valued operands with a distance — the
  projection form is simpler and operates directly on the bivector
  SymbolicSubSpace.

- **Overlap penalty (not antisymmetry).**  The legacy
  `ImpenetrableLayer` penalized mutual parthood $P[i,j] \cdot P[j,i]$
  directly, which pushes *every* overlapping pair apart, including
  pairs that should be equal (synonyms, tied codebook slots).  The
  five-relations design leaves `equal` pairs alone and only
  penalizes the genuinely ambiguous middle region, gated by trust
  mismatch so that two high-trust near-synonyms with matched usage
  aren't forced apart either.

- **Trust via VQ EMA.**  Codebook usage frequency is already tracked
  for VQ commit loss; reusing it as trust avoids a second bookkeeping
  path.

- **Parthood is preserved by Pi / Sigma.**  Under
  `<bivectorOutput>true</bivectorOutput>` on PerceptualSpace,
  ConceptualSpace, and SymbolicSpace, every activation in the chain
  lives on the non-negative paired-index cone $[0, 1]^{2K}$ and the
  Pi / Sigma layers are restricted to entry-wise $W \geq 0$ via
  `NonNegativeInvertibleLinearLayer` (or `NonNegativeLinearLayer`).
  Positive matrices are exactly the monotone operators on a positive
  cone: $a \leq b$ componentwise $\Rightarrow Wa \leq Wb$
  componentwise.  Componentwise $\leq$ on the cone *is* the parthood
  partial order, so each lift / lower in the chain preserves
  parthood pole-by-pole — a whole always contains its parts after
  Pi / Sigma.  The bivector layout keeps the contradiction corner
  $[1, 1]$ distinct from the ignorance corner $[0, 0]$ under the
  positive matmul, which a single bitonic axis would collapse.  See
  [Spaces.md "Monotonicity of the lift / lower
  chain"](Spaces.md#monotonicity-of-the-lift--lower-chain).
