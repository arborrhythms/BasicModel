# Logic.md

## Overview

This document defines a two-layer logic system:

1. **Subsymbolic (vector / field level)**
2. **Symbolic (scalar level in [-1,1])**

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

## 4. Interpretation

| Operator | Meaning |
|----------|--------|
| neg | oppositional negation |
| non | withdrawal / neutrality |
| union | strongest affirmation |
| intersection | shared commitment |
| part | signed containment |

---

## 5. Key Insight

- Subsymbolic layer = geometry
- Symbolic layer = order + polarity
- Symbolization = norm projection

This cleanly separates:
- representation (vectors)
- logic (scalars)
