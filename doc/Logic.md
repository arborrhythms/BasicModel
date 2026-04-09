# Logic

## Overview

This document defines the logic system at two levels:

1. **Subsymbolic (vector / field level)** — geometry in ConceptualSpace
2. **Symbolic (scalar level in [-1,1])** — order + polarity in SymbolicSpace
3. **Rationality** — propositional truth store built on top of both

The executable implementations of the subsymbolic and symbolic operators are
the `Method` subclasses documented in Language.md § Methods.

---

## 1. Subsymbolic Layer

Objects:
- Vector sets: (B, N, D)
- Interpreted as RBF / luminosity fields

### Operators

- **Union**:
  Combine sets
  ```
  union(A, B) = concat(A, B)
  ```

- **Intersection**:
  Co-supported regions (RBF product / merge)

- **Negation (affirming)**:
  ```
  neg(x) = -x
  ```
  Antipodal opposition on hypersphere

- **Non (non-affirming negation)**:
  $$
  \operatorname{non}(x) = \alpha x, \quad \alpha \in [0,1)
  $$
  Contraction toward zero (withdrawal of assertion)

- **Parthood**:
  Fuzzy max-coverage:
  $$
  P(A \to B) \in [-1,1]
  $$

---

## 2. Symbolization

Map vectors → scalar truth strength

For $X \in (B, N, D)$:

$$
s(X) = 2 \cdot \mathrm{mean}(\lVert x_i \rVert) - 1
$$

Range: [-1, 1]

Interpretation:
- +1 → strong presence
-  0 → neutral
- -1 → absence

---

## 3. Symbolic Layer (Scalars in [-1,1])

Let $a, b \in [-1,1]$.

### Negation (affirming)
$$
\operatorname{neg}(a) = -a
$$

### Non (non-affirming)
$$
\operatorname{non}(a) = \alpha a
$$

### Union
$$
a \cup b = \max(a, b)
$$

### Intersection
$$
a \cap b = \min(a, b)
$$

### Parthood (order relation)
$$
\operatorname{part}(a, b) = \operatorname{clamp}(b - a, -1, 1)
$$

---

## 4. Key Insight

- Subsymbolic layer = geometry
- Symbolic layer = order + polarity
- Symbolization = norm projection

This cleanly separates:
- representation (vectors)
- logic (scalars)

---

## 5. Rationality

Rationality is the propositional logic layer.  It is built on the S-tier
grammar rules `part(S, S)` and `equals(S, S)`, which are the two relations
that compose propositions about the world.

### Truth Statements

A **truth statement** is any assertion that the model has meaningfully
processed through the full pipeline (InputSpace → ... → SymbolicSpace),
producing a symbolic activation vector.  The `TruthLayer` on SymbolicSpace
stores these activations **scaled by** the DegreeOfTruth:

$$
\text{stored} = \text{activation} \times \text{degree}
$$

The DegreeOfTruth in $[-1, 1]$ is baked into the stored vector:

| Degree | Stored Vector | Effect |
|--------|--------------|--------|
| +1 | full activation | attractor |
| 0 < d < +1 | scaled activation | weak attractor |
| 0 | zero vector (inert) | prunable |
| -1 < d < 0 | scaled, negated | weak disperser |
| -1 | negated activation | disperser |

### TruthLayer (SymbolicSpace.truth)

`TruthLayer` in `Model.py` is instantiated on `SymbolicSpace` when
`<accumulateTruth>` is set to a value > 0 (the degree of truth, 0..1):

- **record(activation, degree)** — store `activation * degree`.
- **query(activation)** — find the closest stored truth by cosine similarity.
- **field(concepts)** — project all stored truths into ConceptualSpace as a
  scalar field over concept vectors.

### Truth Field

When stored truths are projected into ConceptualSpace via `field()`, they
form a scalar field $f: \mathbb{R}^D \to [-1, 1]$ over concept vectors:

$$
f(c) = \frac{1}{n} \sum_{i=1}^{n} c \cdot t_i
$$

where $t_i = \text{activation}_i \times \text{degree}_i$ is a stored truth
(DoT already baked in) and $c$ is a unit-normalised concept vector.

This field has two kinds of regions:

- **Attractors** ($f \to +1$): concept vectors near truths with high positive
  degree.  These represent regions of conceptual space where stored knowledge
  says "this is true."
- **Dispersers** ($f \to -1$): concept vectors near truths with high negative
  degree.  These represent regions where stored knowledge says "this is false."

### Consonance and Dissonance

Stored truths should satisfy two consistency conditions:

1. **Internal consonance** — truths should be mutually consistent.  If
   $\text{part}(A, B)$ has degree +1, and $\text{part}(B, C)$ has degree +1,
   then $\text{part}(A, C)$ should not have degree -1.

2. **External consonance** — incoming statements processed through the pipeline
   should be evaluated against the truth field.  High similarity to a +1 truth
   is consonance; high similarity to a -1 truth is dissonance.

### Propositional Structure

The S-tier grammar provides the two propositional relations:

- **part(S, S)** — containment.  "A is part of B."  Symbol A is a part of
  symbol B to the extent that A's projection into conceptual space is
  contained in B's: $C(A) \le C(B)$ element-wise for all elements.
  Implemented as `min(A, B)`: when $C(A) \le C(B)$ holds, the part passes
  through unchanged; where A exceeds B, min clips to B, enforcing
  containment.  Asymmetric: part(A, B) does not imply part(B, A).

- **equals(S, S)** — identity as mutual parthood.
  $\text{equals}(A, B) = \text{part}(A, B) \wedge \text{part}(B, A)$.
  This asserts both $C(A) \le C(B)$ and $C(B) \le C(A)$, therefore
  $C(A) = C(B)$.  Implemented as `min(A, B)` (same as part, since min
  is commutative); the semantic distinction is that equals asserts both
  directions of containment.

Together, `equals` and `part` define a partial order over symbolic activations.
The truth store captures this order as a database of grounded propositions.

### Truth Accumulation Pipeline

Truth entries arrive as `(text, DoT)` pairs from the client's TruthSet.
`store_truths()` stages all texts on TheData and runs a standard inference
epoch via `runEpoch(split="runtime")`.  During forward processing,
`SymbolicSpace.forward()` records each raw symbolic activation into the
TruthLayer (degree 1.0).  After the epoch completes, each stored
activation is scaled by its DegreeOfTruth:

$$
t_i = \text{activation}_i \times \text{DoT}_i
$$

This two-phase design (encode all, then scale) ensures all truths are
encoded in the same pipeline context before DoT modulation is applied.

---

## 6. Consistency and Verification

### Internal Consistency (within a TruthSet)

A TruthSet should be internally consonant: its stored truths should not
contradict each other.  The truth field provides a natural test.

Given $n$ stored truths $\{t_i\}$, project them into ConceptualSpace via
`field()`.  For each truth $t_j$, evaluate the field produced by all
*other* truths $\{t_i : i \neq j\}$ at the concept vector underlying
$t_j$.  If the field value is strongly negative, truth $j$ is dissonant
with the rest of the set:

$$
d_j = f_{\setminus j}(c_j) = \frac{1}{n-1} \sum_{i \neq j} c_j \cdot t_i
$$

where $c_j$ is the unit-normalised concept vector for truth $j$.

- $d_j > 0$: truth $j$ is consonant with the set (supported by neighbours)
- $d_j \approx 0$: truth $j$ is independent (orthogonal — neither supported
  nor contradicted)
- $d_j < 0$: truth $j$ is dissonant (contradicted by neighbours)

This is a leave-one-out consistency check.  It does not require symbolic
logic — it falls out of the geometry of the stored vectors and the DoT
already baked into them.  A disperser ($\text{DoT} < 0$) near an
attractor ($\text{DoT} > 0$) will naturally produce negative field values,
flagging the contradiction.

### Verification (incoming statement against stored truth)

`verify(statement)` processes an incoming statement through the pipeline
to obtain its symbolic activation $a$, then queries the truth field:

$$
v = f(c_a) = \frac{1}{n} \sum_{i=1}^{n} c_a \cdot t_i
$$

where $c_a$ is the concept vector underlying $a$.

| $v$ | Interpretation |
|-----|---------------|
| $v \to +1$ | strong support — statement aligns with stored attractors |
| $v \approx 0$ | no opinion — statement is orthogonal to stored knowledge |
| $v \to -1$ | strong contradiction — statement aligns with dispersers |

This gives a scalar degree of verification in $[-1, 1]$ without requiring
the incoming statement to exactly match any stored truth.  Cosine
similarity in SymbolicSpace (`query()`) can also find the single closest
truth and return its degree, which is useful for point lookups.

### Logical Entailment (augmenting geometry with symbolic closure)

The subsymbolic field checks above detect *geometric* consistency — truths
that point in contradictory directions in the embedding space.  But they
do not enforce *logical* entailments.  Consider:

- Stored: $\text{part}(A, B)$ with DoT +1
- Stored: $\text{part}(B, C)$ with DoT +1
- The transitive closure $\text{part}(A, C)$ is *entailed* but not stored.

If a statement $\neg\text{part}(A, C)$ arrives, the field check may not
flag it — $A$ and $C$ could be geometrically distant even though the
chain of parthood connects them.  Geometry captures similarity, not
inference chains.

A symbolic closure layer would address this by operating on the
propositional structure of stored truths:

1. **Extract propositions.** Each stored truth whose activation was
   produced by an S-tier grammar rule (`equals(S,S)` or `part(S,S)`)
   carries relational structure: a relation and two operands.  These
   can be read off the symbolic activation by decomposing it into
   the relation slot and the two argument slots.

2. **Compute closure.** Apply transitivity rules on the extracted
   propositions:
   - $\text{part}(A, B) \wedge \text{part}(B, C) \Rightarrow \text{part}(A, C)$
   - $\text{equals}(A, B) \wedge \text{equals}(B, C) \Rightarrow \text{equals}(A, C)$
   - $\text{equals}(A, B) \wedge \text{part}(A, C) \Rightarrow \text{part}(B, C)$

   Each entailed proposition inherits a DoT from its premises — the
   minimum (intersection) of the premise DoTs, following the symbolic
   intersection rule $a \cap b = \min(a, b)$.

3. **Inject entailments.** Entailed propositions that are not already
   stored can be synthesised as new symbolic activations (by composing
   the relation and operand embeddings) and recorded in the TruthLayer
   with their derived DoT.

This is worth implementing when the truth store is used for reasoning
(verification, planning) rather than just retrieval.  The geometric
field alone is sufficient for soft consistency checks and nearest-truth
lookups.  The symbolic closure makes the store *deductively closed* —
turning it from a database of assertions into something closer to a
knowledge base with inference.

## 7.

# Radial Operators for Hypersphere Ternary Logic

## Overview

RadMin, RadMax, NOT, and NON are four logical operators defined over a signed magnitude space where truth values range from -1 to +1. The semantic space is a hypersphere with zero at the origin representing **unknown**, +1 representing **true**, and -1 representing **false**.

The operators are designed to respect the geometric structure of this space: sign represents direction of assertion, magnitude represents strength of assertion, and zero represents the absence of assertion.

## Truth Values

| Value | Meaning |
|-------|---------|
| +1    | True (full positive assertion) |
| 0     | Unknown (no assertion) |
| -1    | False (full negative assertion) |

Intermediate values (e.g., +0.5, -0.3) represent partial assertions with varying degrees of confidence.

## Magnitude Ordering

The ordering relation $\subset$ ("is a part of") is defined by magnitude:

> $x \subset y$ iff $|x| < |y|$

This means "closer to zero" is "less than" in the radial sense. Zero is the weakest value; ±1 are the strongest.

---

## Operators

### RadMin (Radial Minimum — Conjunction / AND)

RadMin is a binary operator representing conjunction. It collapses toward zero on sign disagreement and takes the minimum magnitude on sign agreement.

**Definition:**

- If $\mathrm{sign}(x) \ne \mathrm{sign}(y)$: $\mathrm{RadMin}(x, y) = 0$
- If $\mathrm{sign}(x) = \mathrm{sign}(y)$: $\mathrm{RadMin}(x, y) = \mathrm{sign}(x) \cdot \min(|x|, |y|)$

**Truth Table (Ternary):**

|        | **+1** | **0** | **-1** |
|--------|--------|-------|--------|
| **+1** | +1     | 0     | 0      |
| **0**  | 0      | 0     | 0      |
| **-1** | 0      | 0     | -1     |

**Properties:**
- Commutative: RadMin(x, y) = RadMin(y, x)
- Zero is absorbing: RadMin(x, 0) = 0 for all x
- Anti-diagonal symmetric
- Sign disagreement produces zero (contradiction → unknown)

---

### RadMax (Radial Maximum — Disjunction / OR)

RadMax is a binary operator representing disjunction. It collapses toward zero on sign disagreement and takes the maximum magnitude on sign agreement.

**Definition:**

- If $\mathrm{sign}(x) \ne \mathrm{sign}(y)$: $\mathrm{RadMax}(x, y) = 0$
- If $\mathrm{sign}(x) = \mathrm{sign}(y)$: $\mathrm{RadMax}(x, y) = \mathrm{sign}(x) \cdot \max(|x|, |y|)$
- RadMax(x, 0) = x (zero is transparent / neutral for OR)

**Truth Table (Ternary):**

|        | **+1** | **0** | **-1** |
|--------|--------|-------|--------|
| **+1** | +1     | +1    | 0      |
| **0**  | +1     | 0     | -1     |
| **-1** | 0      | -1    | -1     |

**Properties:**
- Commutative: RadMax(x, y) = RadMax(y, x)
- Zero is transparent: RadMax(x, 0) = x for all x
- Anti-diagonal symmetric
- Sign disagreement produces zero (contradiction → unknown)

---

### NOT (Sign-Flipping Negation)

NOT is a unary operator that inverts the sign of the assertion while preserving magnitude.

**Definition:**

> NOT(x) = -x

**Truth Table (Ternary):**

| Input | Output |
|-------|--------|
| +1    | -1     |
| 0     | 0      |
| -1    | +1     |

**Properties:**
- Involutive: NOT(NOT(x)) = x
- Preserves magnitude: |NOT(x)| = |x|
- Zero is a fixed point: NOT(0) = 0

---

### NON (Non-Affirming Negation)

NON is a unary operator that drives any assertion toward zero. It represents the withdrawal of assertion rather than the inversion of assertion.

**Definition:**

> NON(x) = 0 for all x

**Truth Table (Ternary):**

| Input | Output |
|-------|--------|
| +1    | 0      |
| 0     | 0      |
| -1    | 0      |

**Properties:**
- Absorbing: NON(x) = 0 for all x
- Idempotent: NON(NON(x)) = NON(x) = 0
- Semantically distinct from NOT: NON does not assert the opposite; it withdraws assertion entirely

---

## Semantic Summary

| Operator | Type   | Role                          | Behavior on Contradiction |
|----------|--------|-------------------------------|---------------------------|
| RadMin   | Binary | Conjunction (AND)             | Collapses to zero         |
| RadMax   | Binary | Disjunction (OR)              | Collapses to zero         |
| NOT      | Unary  | Sign-flipping negation        | N/A                       |
| NON      | Unary  | Non-affirming negation        | N/A                       |

## Open Questions

- **Functional completeness:** Do RadMin, RadMax, NOT, and NON form a functionally complete set over the ternary radial space?
- **Associativity:** Does RadMin(RadMin(a, b), c) = RadMin(a, RadMin(b, c)) hold for all combinations?
- **Duality:** RadMin and RadMax are not classical duals via NOT due to differing treatment of zero (absorbing vs. transparent). What is the formal relationship between them?
- **Extension to fuzzy domain:** In the continuous fuzzy extension (values in [-1, +1]), what is the behavior of NON? Does it drive toward zero continuously or collapse discretely?
