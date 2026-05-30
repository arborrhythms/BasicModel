# Mereology

Single-page reference for the mereological grammar: parthood as the
fundamental operation, the five mereological relations, and the
`ImpenetrableLayer` regularizer.

> **2026-05-29 delta — binary GrammarLayer reverses take Basis, not W.**
> `Ops._binary_op_recommend(result, W, op_name, …)` (the
> mereology-guided recommender that walks `W` rows to find an
> operand pair such that $\mathrm{op}(x_1, x_2) \approx \mathit{result}$) is still keyed on
> the raw `W` tensor at the low-level kernel signature. But the
> public `UnionLayer.reverse(parent, basis=None)` /
> `IntersectionLayer.reverse(parent, basis=None)` now accept a
> Codebook / Basis object (typically `SymbolicSpace.subspace.what`)
> and extract `W = basis.getW()` internally before dispatching to
> `Ops.disjunctionReverse` / `Ops.conjunctionReverse`. The chart
> reverse / signal-router dispatch (`bin/Language.py::unreduce()`)
> passes `basis=tier_basis` at the call site; no back-ref is stored
> on the layer. See
> [doc/plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

> **Codebook IS the meronymic structure.** The standalone
> `MereologicalTree` sidecar that formerly stored explicit parent /
> equality links was retired in favour of pure-geometric parthood
> on the `SymbolicSpace` bivector codebook. The grammar layers
> `PartLayer`, `IsEqualLayer`, `EqualLayer`, and `QueryLayer` operate directly on
> codebook bivector activations via clipped cosine projection --- no
> separate adjacency table, no `<architecture><mereologicalTreeSize>`
> XML knob (silently ignored if present). Asserted meronymic
> relations are learned by training pulling the codebook geometry
> into the right configuration.

---

## Parthood as Clipped Cosine Projection

Parthood is the single fundamental operation. Every other mereological
relation is defined in terms of it.

For two concepts $A, B \in \mathbb{R}^D$:

$$
\mathrm{part}(A, B) = \frac{\max(0,\ A \cdot B)}{|A|\,|B|}
$$

The clipped cosine projection lies in $[0, 1]$ and satisfies Boole's
contrapositive:

$$
\mathrm{part}(A, B) = \mathrm{part}(-B, -A)
$$

because $(-B) \cdot (-A) = A \cdot B$ and norms are sign-invariant.

**Empty-operand contract.** When $|A|$ or $|B|$ is near zero, `part` returns
1.0 --- the empty set is part of everything.

---

## The Full Suite

Every member of the suite is expressible through `part`:

| Method | Formula | Interpretation |
|--------|---------|----------------|
| `part(A, B)` | $\max(0, A \cdot B) / (|A|\,|B|)$ | Fundamental. A is part of B. |
| `whole(A, B)` | $\mathrm{part}(B, A)$ | A contains B. |
| `equal(A, B)` | $\mathrm{part}(A, B) \cdot \mathrm{part}(B, A)$ | Mutual parthood, $[0, 1]$. |
| `overlap(A, B)` | $0 < \mathrm{equal}(A, B) < 1$ | Strictly-partial mutual parthood (indicator). |
| `underlap(A, B)` | $\mathrm{equal}(A, B) = 0$ | No mutual parthood (indicator). |
| `boundary(A, B)` | $|\mathrm{part}(A, B) - \mathrm{part}(B, A)|$ | Asymmetry of containment (zero under clipped cosine; for bases with asymmetric `part`). |

---

## Region Partition

`equal(A, B)` partitions into three disjoint regions:

| Region | Condition |
|--------|-----------|
| Underlap | $\mathrm{equal} = 0$ |
| Overlap | $0 < \mathrm{equal} < 1$ |
| Identity | $\mathrm{equal} = 1$ |

Under clipped cosine, `part` is symmetric, so the three regions collapse to
a single scalar classifier. For a future `Basis` with asymmetric `part`, the
general five-relation table applies:

**Five-relation table** (general case, $\tau \approx 1$, $\epsilon \approx 0$):

| Relation | Condition |
|----------|-----------|
| `disjoint(a,b)` | $P[a,b] < \epsilon$ and $P[b,a] < \epsilon$ |
| `part(a,b)` | $P[a,b] > \tau$ and $P[b,a] < \epsilon$ |
| `part(b,a)` | $P[a,b] < \epsilon$ and $P[b,a] > \tau$ |
| `equal(a,b)` | $P[a,b] > \tau$ and $P[b,a] > \tau$ |
| `overlap(a,b)` | Both partial |

**Asymmetric subsumption.** Since `part` is symmetric under clipped cosine,
classical asymmetric subsumption is not encoded in raw magnitude. It is
recovered **relationally** via figure / ground: compare $\mathrm{part}(A, B)$
against $\mathrm{part}(A, \neg B)$. This is what makes Boole's contrapositive
hold exactly.

---

## Mereological Fusion

Fusion of a truth set is the **least upper bound** (LUB / join) over stored
truth vectors:

$$
\mathrm{fusion} = \max_i\, \mathrm{truths}[i]
$$

(elementwise max). In bivector space, the fusion vector names the top-right
corner of the axis-aligned bounding hyperrectangle dominating every stored
truth. Fusion is the geometric dual of luminosity: LUB (join) vs GLB (meet).

DoT is already baked into each stored truth (`stored[i] = activation_i *
degree_i` in `TruthLayer.record`), so fusion is trust-weighted automatically.

See [Logic.md](Logic.md) section **Fusion** for full discussion and the
leading-bivector / paired-index layout caveat.

---

## ImpenetrableLayer: Five-Relations Regularizer

`ImpenetrableLayer` regularizes the SymbolicSpace symbol codebook. It
classifies each ordered pair of codebook rows $(i, j)$ into one of the five
mereological relations (using `Basis.part`), and penalizes partial overlap
when paired with a trust mismatch. The learned half of the [Codebook
Uniqueness Contract](Spaces.md#codebook-uniqueness-contract).

### Penalty

$$
\mathcal{L} = \mathrm{overlap\_weight} \cdot
   \frac{\sum_{i \neq j} \mathrm{overlap}(i, j)
                        \cdot |\mathrm{trust}(i) - \mathrm{trust}(j)|}
        {K(K-1)}
   + \mathrm{variance\_floor\ term}
$$

Overlap strength is damped to zero as the pair approaches `equal`:

$$
\mathrm{overlap\_strength}(i, j) =
   \min(P[i,j],\ P[j,i]) \cdot
   \left( 1 - \max(P[i,j],\ P[j,i])^k \right)
$$

with $k = \mathrm{equal\_suppression}$ (default 4.0). Penalty is zero for
`disjoint`, `equal`, and strict `part`, active only for `overlap`.

A separate `variance_floor` term guards against row collapse.

### Trust

Per-row usage frequency, derived from VQ EMA counts:

$$
\mathrm{trust}(i) = \frac{\mathrm{cluster\_size}[i]}{\sum_j \mathrm{cluster\_size}[j]}
$$

When VQ is absent, trust falls back to $\|cb[i]\| / \max_j \|cb[j]\|$.

> **Relative relations and tetralemma trust.** A relative sentence in
> the `part` / `isEqual` predicate family is preserved at depth 3 as
> `[predicate, idea1, idea2]` and may be learned into the codebook as a
> ternary META edge (predicate as parent, the two ideas as children) via
> `SymbolicSpace.insert_relation`. An accepted relation carries a
> **tetralemma trust 4-tuple** $(t, f, b, n)$ (TRUE / FALSE / BOTH /
> NEITHER, summing to $1$) from the TruthSet posture, gated by the
> content-aware learn-score against `<truthCriterion>`. See
> [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states).

### Diagnostics

After `forward()`: `last_overlap_loss`, `last_variance`,
`last_relation_counts` (dict summing to $K(K-1)$).

### Configuration

XML knobs (under SymbolicSpace):

| Knob | Default | Meaning |
|------|---------|---------|
| `overlapWeight` | 0.1 | Weight of overlap $\times$ trust-diff penalty |
| `varianceFloor` | 0.01 | Minimum per-dim std |
| `fullPartThreshold` | 0.9 | $\tau$: part score above this is "full part" |
| `disjointThreshold` | 0.1 | $\epsilon$: part score below this is "disjoint" |
| `equalSuppression` | 4.0 | $k$: damping exponent near the equal corner |

---

## Why This Design

- **Parthood as projection.** One formula, Boole-contrapositive exact,
  continuous in $[0, 1]$. The old composite formula
  $\operatorname{conjunction}(1 - \operatorname{dist}(x, x \cap y),
  1 - \operatorname{dist}(y, x \cup y))$ mixed set-valued operands
  with a distance.

- **Overlap penalty (not antisymmetry).** Legacy `ImpenetrableLayer`
  penalized mutual parthood $P[i,j] \cdot P[j,i]$ directly, pushing
  *every* overlapping pair apart including synonyms. The five-relations
  design leaves `equal` pairs alone and only penalizes the ambiguous middle
  region, gated by trust mismatch.

- **Trust via VQ EMA.** Already tracked for VQ commit loss.

- **Parthood is preserved by Pi / Sigma.** Under `monotonic=true`, Pi/Sigma
  are restricted to $W \geq 0$. User-supplied truth-set bivectors live on
  the non-negative paired `[pos, neg]` cone, where componentwise $\leq$ is
  the parthood partial order. Each lift / lower preserves parthood
  pole-by-pole for that truth surface. The bivector layout keeps
  contradiction `[1, 1]` distinct from ignorance `[0, 0]` under positive
  matmul. See
  [Spaces.md "Monotonicity of the lift / lower chain"](Spaces.md#monotonicity-of-the-lift--lower-chain).
