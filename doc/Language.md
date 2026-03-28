# Language

## Overview

The language system is distributed across three cognitive spaces, each with its
own grammar rules and projection operations.  A **SyntacticLayer** at each space
predicts rule distributions via Gumbel-softmax, then the space's **projection
method** executes all candidate rules in soft superposition, weighted by those
probabilities.

```
PerceptualSpace  →  ConceptualSpace  →  SymbolicSpace  →  OutputSpace
  rule 11 (P→W)     rules 7-9 (C)        rules 1-5 (S)
  terminal           mereological         logical
  [B, N, D] vectors  [B, N, D] vectors    [B, N] activations
```

The grammar is hierarchical: symbolic rules (S-level) compose first, then
transition via $S \to C$ to conceptual rules, then via $C \to P$ to the
perceptual terminal.

---

## Grammar

`TheGrammar` is a singleton `Grammar` instance with 12 rules partitioned
across three cognitive levels:

### Symbolic Rules (SymbolicSpace)

| Rule | Production | Arity | Operation |
|------|-----------|-------|-----------|
| 0 | S | 0 | start symbol |
| 1 | $S \to S\ \text{EQUALS}\ S$ | 2 | `AssociationLayer` (Hopfield cross-symbol memory) |
| 2 | $S \to S\ \text{AND}\ S$ | 2 | $\min(l, r)$ (Godel t-norm) |
| 3 | $S \to S\ \text{OR}\ S$ | 2 | $\max(l, r)$ (Godel t-conorm) |
| 4 | $S \to \text{NOT}\ S$ | 1 | $1 - x$ (complement) |
| 5 | $S \to \text{NON}\ S$ | 1 | $\sigma(\alpha) \cdot x$ (learnable dampening) |
| 6 | $S \to C$ | 0 | transition to conceptual level |

### Conceptual Rules (ConceptualSpace)

| Rule | Production | Arity | Operation |
|------|-----------|-------|-----------|
| 7 | $C \to C\ \text{PART}\ C$ | 2 | $\min(l, r)$ on vectors |
| 8 | $C \to C\ \text{UNION}\ C$ | 2 | $\max(l, r)$ on vectors |
| 9 | $C \to C\ \text{INTERSECTION}\ C$ | 2 | $\min(l, r)$ on vectors |
| 10 | $C \to P$ | 0 | transition to perceptual level |

### Perceptual Rules (PerceptualSpace)

| Rule | Production | Arity | Operation |
|------|-----------|-------|-----------|
| 11 | $P \to W$ | 0 | terminal: word embedding recovery |

### Space-Level Accessors

- `Grammar.symbolic()` $\to$ `[1, 2, 3, 4, 5]`
- `Grammar.conceptual()` $\to$ `[7, 8, 9]`
- `Grammar.perceptual()` $\to$ `[11]`

### Transition Rules

Rules 6 ($S \to C$) and 10 ($C \to P$) are **transitions** between grammar
levels.  They have arity 0 at their own level (the RHS nonterminal is a
different symbol than the LHS).  In the XML parse tree output, transitions
are transparent --- they do not emit tags.

---

## Word Encoding

Each word is a 3-tuple `(batch, vector, rule)`:

- **batch** --- index into the batch dimension $[0, B)$
- **vector** --- index into the activation vector $[0, N)$
- **rule** --- grammar rule ID from `TheGrammar` $[0, 12)$

Words are stored as a Python list of tuples on the SubSpace.  The list is a
**derivation tree** in pre-order: the first entry is the root, and binary rules
expand left-first.

---

## SyntacticLayer (Rule Prediction)

### Architecture: Recursive Hybrid with Depth Embedding

Each space has its own `SyntacticLayer` instance that predicts rule distributions
over **only that space's rules**.  The architecture is weight-tied (recursive):
a single shared derivation layer and rule head are applied at each depth with a
learned depth embedding.

$$
\text{activation}\ [B, N]
\xrightarrow{\text{input\_proj}}
h \in \mathbb{R}^{B \times d_{\text{hidden}}}
$$

For each depth $d = 0, 1, \ldots, D$, using **shared weights**:

$$
h \leftarrow h + \text{depth\_embed}[d], \quad
h \leftarrow \text{GELU}(\text{derivation\_layer}(h)), \quad
\text{logits}_d \leftarrow \text{rule\_head}(h)
$$

The `rule_head` output dimension equals the number of rules in that space
(e.g.\ 6 for symbolic, 4 for conceptual, 1 for perceptual).

### Differentiable Rule Selection

- **Training:** `gumbel_softmax(logits, tau)` --- soft one-hot, gradients flow
  through all candidate rules proportional to their probability.
- **Eval:** `softmax` + `argmax` --- discrete rule selection.
- **Annealing:** Temperature $\tau$ starts at 1.0 (diffuse) and decreases toward
  0.1 (near-discrete) over training.

---

## Per-Space Projection Operations

### SymbolicSpace: `projectSymbols(rule_id, left, right)`

Operates on **$[B, N]$ scalar activations** (cross-symbol attention/association).

| Rule | Implementation |
|------|---------------|
| EQUALS | `AssociationLayer(type="symmetric")` --- Hopfield-like bidirectional associative memory.  Learns which symbols retrieve which other symbols.  Output modulated by right's activation. |
| AND | $\min(l, r)$ --- Godel t-norm (fuzzy conjunction) |
| OR | $\max(l, r)$ --- Godel t-conorm (fuzzy disjunction) |
| NOT | $1 - x$ --- standard complement |
| NON | $\sigma(\alpha) \cdot x$ --- learnable dampening ($\alpha$ is a parameter) |

### ConceptualSpace: `projectConcepts(rule_id, left, right)`

Operates on **$[B, N, D]$ embedded vectors** (mereological composition in
continuous concept space).

| Rule | Implementation |
|------|---------------|
| PART | $\min(l, r)$ --- parthood: the overlap region |
| UNION | $\max(l, r)$ --- mereological union: combined extent |
| INTERSECTION | $\min(l, r)$ --- mereological intersection: shared region |

### PerceptualSpace: `projectPercepts(rule_id, vspace)`

Operates on **SubSpace** (recovering word embeddings via the reverse PiLayer).

| Rule | Implementation |
|------|---------------|
| W | Reverse PiLayer projection: maps percepts back toward input space |

---

## Soft Superposition: `composeSyntax()`

Each space provides a `composeSyntax()` method that combines rule prediction
with projection execution.  At each derivation depth:

1. Compute **all** rule operations on the current left and right children.
2. Weight each result by the Gumbel-softmax probability for that rule.
3. Sum to produce the parent representation.

$$
\text{parent} = \sum_{r \in \text{rules}} p_r^{(d)} \cdot \text{project}(r,\ \text{left},\ \text{right})
$$

During training, all rules contribute proportional to their soft probability.
During eval, the argmax rule dominates.  Over training, reconstruction loss
sharpens the probabilities: rules that consistently support successful
reconstruction accumulate weight, while weaker alternatives diminish.

**Deep structure emerges as a learned artifact of reconstruction.**

---

## Parse Tree Output

The model emits an XML parse tree via `parse.derivation_to_xml()`.  Word tuples
from all three spaces are collected (with global Grammar rule IDs) and
reconstructed into a tree by recursive descent over the pre-order word list.

Transition rules ($S \to C$, $C \to P$) are transparent --- they do not emit XML
tags.  Terminal rules ($P \to W$) emit `<token>` leaves.  All other rules emit
their named tag (`<conjunction>`, `<equals>`, `<not>`, `<union>`, etc.).

Example for "dogs AND NOT cats":

```xml
<conjunction>
  <token word="dogs"/>
  <not>
    <token word="cats"/>
  </not>
</conjunction>
```

---

## AssociationLayer (EQUALS Implementation)

The `AssociationLayer` is a cross-symbol associative memory used by the EQUALS
rule.  Two modes are available:

- **`type="symmetric"`** --- Hopfield-like: learns projection $A$, computes
  association scores $A^T A$, softmax-retrieves the associated pattern.
  Associations are symmetric ($A \equiv B \Leftrightarrow B \equiv A$).
- **`type="hopfield"`** --- Modern Hopfield: separate query/key projections,
  softmax-gated retrieval.

Input and output are both $[B, N]$ activation vectors.  The layer is learnable
and its parameters are trained end-to-end via the reconstruction loss.

---

## SymbolicSpace

### Forward Path

1. Extract concept activation $[B, n_{\text{Concepts}}]$.
2. Map through `InvertibleLinearLayer` to $[B, n_{\text{Symbols}}]$.
3. Codebook quantization (when enabled) produces one-hot activation + vectors.

### Reverse Path

1. Extract symbol activation $[B, n_{\text{Symbols}}]$.
2. Exact inverse of the invertible layer recovers $[B, n_{\text{Concepts}}]$.

### Key Properties

- Symbols are **zero-dimensional** --- pure activation scalars, not vectors.
- The invertible layer allows $n_{\text{Concepts}} \neq n_{\text{Symbols}}$.
- `composeSyntax()` runs the symbolic SyntacticLayer and executes the soft
  superposition of EQUALS/AND/OR/NOT/NON projections.
