# Language

## Overview

The language system is distributed across three cognitive spaces, each with its
own grammar rules and projection operations.  A **SyntacticLayer** at each space
predicts rule distributions via Gumbel-softmax, then the space's **projection
method** executes all candidate rules in soft superposition, weighted by those
probabilities.

```
PerceptualSpace  →  ConceptualSpace  →  SymbolicSpace  →  OutputSpace
  rules 13–14 (P)    rules 5–12 (C)      rules 1–3 (S)
  structural          mereological        logical
  [B, N, D] vectors   [B, N, D] vectors   [B, N] activations
```

The grammar is hierarchical: symbolic rules (S-level) compose first, then
transition via $S \to C$ to conceptual rules, then via $C \to P$ to the
perceptual terminal.

---

## Grammar

`TheGrammar` is a singleton `Grammar` instance with 15 rules (IDs 0–14)
partitioned across three cognitive levels plus one start rule:

### START Rule

| Rule | Production | Arity | Method |
|------|-----------|-------|--------|
| 0 | START → true(S) EOF | 1 | `trueMethod` |

### Symbolic Rules (SymbolicSpace)

| Rule | Production | Arity | Method |
|------|-----------|-------|--------|
| 1 | $S \to \text{swap}(S,\ S)$ | 2 | `swapMethod` |
| 2 | $S \to \text{equals}(S,\ S)$ | 2 | `equalsMethod` |
| 3 | $S \to \text{part}(S,\ S)$ | 2 | `partMethod` |
| 4 | $S \to C$ | 1 | — (transition) |

### Conceptual Rules (ConceptualSpace)

| Rule | Production | Arity | Method |
|------|-----------|-------|--------|
| 5 | $C \to \text{union}(C,\ C)$ | 2 | `unionMethod` |
| 6 | $C \to \text{intersection}(C,\ C)$ | 2 | `intersectionMethod` |
| 7 | $C \to \text{lower}(C)$ | 1 | `lowerMethod` |
| 8 | $C \to \text{lift}(C,\ C)$ | 2 | `liftMethod` |
| 9 | $C \to \text{lift}(C)$ | 1 | `liftMethod` |
| 10 | $C \to \text{not}(C)$ | 1 | `notMethod` |
| 11 | $C \to \text{non}(C)$ | 1 | `nonMethod` |
| 12 | $C \to P$ | 1 | — (transition) |

### Perceptual Rules (PerceptualSpace)

| Rule | Production | Arity | Method |
|------|-----------|-------|--------|
| 13 | $P \to I\ P$ | 1 | — (recursive) |
| 14 | $P \to \varepsilon$ | 0 | — (terminal) |

### Space-Level Accessors

- `Grammar.symbolic()` → `[1, 2, 3]`
- `Grammar.conceptual()` → `[5, 6, 7, 8, 9, 10, 11]`
- `Grammar.perceptual()` → `[13, 14]`

### Transition Rules

Rules 4 ($S \to C$) and 12 ($C \to P$) are **transitions** between grammar
levels.  They have arity 1 at their own level (the RHS nonterminal is a
different symbol than the LHS).  In the XML parse tree output, transitions
are transparent — they do not emit tags.

### Grammar Configuration

`Grammar.configure()` loads an XML `<grammar>` section (inside `<mentalModel>`)
and restricts the active rule set by left-hand side:

- `<S>...</S>` entries become the active symbolic rule set
- `<C>...</C>` entries become the active conceptual rule set
- `<P>...</P>` entries become the active perceptual rule set

Rules are specified in functional notation: `equals(S, S)`, `union(C, C)`,
`swap(S, S)`, `true(S) EOF`, `I P`, `ε`, `P`, `C`.

---

## Methods (Rule Semantics)

There are 10 `Method` subclasses.  They are the executable side of the
functional grammar defined in `MentalModel.xml`.  The grammar parser turns
entries like `union(C, C)` or `true(S) EOF` into `RuleDef`s with a
`method_name` and `arity` in `Model.py`.  Then `SyntacticLayer` predicts which
rule fires, and the owning space executes it: C-tier conceptual rules dispatch
through `self.methods` in `ConceptualSpace.projectConcepts()`, while S-tier
symbolic rules are handled separately in `projectSymbols()`.

By grammar tier, the current XML is:

- **START:** `true(S) EOF`
- **S:** `swap(S, S)`, `equals(S, S)`, `part(S, S)`, then transition $S \to C$
- **C:** `union`, `intersection`, `lower`, `lift` (binary and unary), `not`,
  `non`, then transition $C \to P$
- **P:** structural only: `I P` and `ε`

### Plain-English Summary

- **trueMethod** — used by START → true(S) EOF.  Collapses the final symbolic
  activation into a scalar truth score by taking mean squared activation.
  "How strongly true/present is the finished statement?"

- **swapMethod** — used by S → swap(S, S).  Swaps or softly mixes the
  children's *where* encodings, not their identities.  "Same two things, but
  exchange their positions/roles."

- **equalsMethod** — used by S → equals(S, S).  On vectors it computes cosine
  similarity and uses that as a gate on the right child; on activations it
  applies a learned association and multiplies by the right child.  "Pass the
  right thing upward only to the degree the left matches it."

- **partMethod** — used by S → part(S, S).  Elementwise min(left, right).
  "Keep only the overlapping part."

- **unionMethod** — used by C → union(C, C).  Elementwise max(left, right).
  "Merge both concepts and keep whatever either one contains."

- **intersectionMethod** — used by C → intersection(C, C).  Elementwise
  min(left, right).  "Keep only what both concepts share."

- **lowerMethod** — used by C → lower(C).  Compresses through a bottleneck and
  expands back out.  "Make the concept simpler/coarser, then reconstruct it."

- **liftMethod** — used by C → lift(C, C) and C → lift(C).  Applies a learned
  linear transform to the left child; in the binary form, multiplies that by the
  right child.  "Reinterpret or elevate the left concept, optionally constrained
  by the right one."

- **notMethod** — used by C → not(C).  Negates the vector.  "Invert the
  concept."

- **nonMethod** — used by C → non(C).  Scales the vector by a learned sigmoid
  factor.  "Soft negation or attenuation, not full logical inversion."

### Implementation Notes

The C-tier methods are the cleanest fully-wired path: `projectConcepts()`
dispatches by `method_name` directly.  The S-tier is mid-migration: the grammar
now says swap/equals/part, but `projectSymbols()` still branches on legacy
names like EQUALS, AND, OR, NOT, NON, and REWRITE.  The intended grammar
meaning is clear, but some symbolic Method classes are not as directly used as
the conceptual ones.  Also, unary `lift(C)` is specified in the grammar, but
because `liftMethod` is marked binary and unary dispatch skips binary methods,
the unary form likely behaves as pass-through today.

---

## Two-Layer Logic

The logic system operates at two levels:

### Subsymbolic Layer (Vectors)

Objects are vector sets $(B, N, D)$ interpreted as RBF / luminosity fields.

| Operator | Implementation | Meaning |
|----------|---------------|---------|
| union | $\max(l, r)$ | strongest affirmation |
| intersection | $\min(l, r)$ | shared commitment |
| negation (not) | $-x$ | antipodal opposition on hypersphere |
| non | $\alpha x,\ \alpha \in [0,1)$ | contraction toward zero (withdrawal of assertion) |
| parthood | fuzzy max-coverage $\in [-1,1]$ | signed containment |

### Symbolization

Map vectors → scalar truth strength:

$$
s(X) = 2 \cdot \mathrm{mean}(\lVert x_i \rVert) - 1 \quad \in [-1, 1]
$$

Interpretation: +1 → strong presence, 0 → neutral, −1 → absence.

### Symbolic Layer (Scalars in [-1,1])

Let $a, b \in [-1,1]$.

| Operator | Formula | Meaning |
|----------|---------|---------|
| neg | $-a$ | oppositional negation |
| non | $\alpha a$ | withdrawal / neutrality |
| union | $\max(a, b)$ | strongest affirmation |
| intersection | $\min(a, b)$ | shared commitment |
| part | $\operatorname{clamp}(b - a, -1, 1)$ | signed containment |

The key insight: subsymbolic = geometry, symbolic = order + polarity,
symbolization = norm projection.

---

## Word Encoding

Each word is a 3-tuple `(batch, vector, rule)`:

- **batch** — index into the batch dimension $[0, B)$
- **vector** — index into the activation vector $[0, N)$
- **rule** — grammar rule ID from `TheGrammar` $[0, 15)$

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
(e.g.\ 4 for symbolic, 8 for conceptual, 2 for perceptual).

### Differentiable Rule Selection

- **Training:** `gumbel_softmax(logits, tau)` — soft one-hot, gradients flow
  through all candidate rules proportional to their probability.
- **Eval:** `softmax` + `argmax` — discrete rule selection.
- **Annealing:** Temperature $\tau$ starts at 1.0 (diffuse) and decreases toward
  0.1 (near-discrete) over training.

The actual rule semantics are owned by the spaces:

- `PerceptualSpace.projectPercepts()`
- `ConceptualSpace.projectConcepts()`
- `SymbolicSpace.projectSymbols()` / `SyntacticSpace.projectSymbols()`

---

## Per-Space Projection Operations

### SymbolicSpace: `projectSymbols(rule_id, left, right)`

Operates on **$[B, N]$ scalar activations** (cross-symbol attention/association).

| Rule | Implementation |
|------|---------------|
| EQUALS | `AssociationLayer(type="symmetric")` — Hopfield-like bidirectional associative memory.  Learns which symbols retrieve which other symbols.  Output modulated by right's activation. |
| AND | $\min(l, r)$ — Godel t-norm (fuzzy conjunction) |
| OR | $\max(l, r)$ — Godel t-conorm (fuzzy disjunction) |
| NOT | $1 - x$ — standard complement |
| NON | $\sigma(\alpha) \cdot x$ — learnable dampening ($\alpha$ is a parameter) |

### ConceptualSpace: `projectConcepts(rule_id, left, right)`

Operates on **$[B, N, D]$ embedded vectors** (mereological composition in
continuous concept space).

| Rule | Implementation |
|------|---------------|
| PART | $\min(l, r)$ — parthood: the overlap region |
| UNION | $\max(l, r)$ — mereological union: combined extent |
| INTERSECTION | $\min(l, r)$ — mereological intersection: shared region |

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

## `write*` and `read*` Methods

Syntax-capable spaces have incremental shift/reduce methods alongside the
batch `composeSyntax()` path.

### `writePercepts` / `readPercepts`

These live in `PerceptualSpace`.

`writePercepts(activation, vectors)`:
- shifts one percept activation/vector
- calls the perceptual `syntactic_layer` to generate percept-level word tuples
- applies the terminal rule $P \to W$
- uses the reverse `PiLayer` path to recover an input-level word embedding

This is a degenerate terminal case rather than a rich reduce stack.

`readPercepts(words, batch_size)`:
- rebuilds a percept activation mask from word tuples

### `writeConcepts` / `readConcepts`

These live in `ConceptualSpace`.

`writeConcepts(activation, vectors)`:
- shifts one concept activation and one concept vector set onto a stack
- predicts a rule on the stack head
- computes a soft reduce using `projectConcepts()`
- records word tuples
- marks whether a transition rule $C \to P$ fired

Conceptual reductions operate on vectors, not scalar activations.

`readConcepts(words, batch_size)`:
- rebuilds concept activation from word tuples

### `writeSymbols` / `readSymbols`

These live in `SyntacticSpace`.

`writeSymbols(symbol_act, where=None)`:
- shifts one symbol activation onto the stack
- predicts a symbolic rule
- computes a soft reduce using `projectSymbols()`
- optionally carries and rewrites `where` encodings
- records word tuples
- marks whether a transition rule $S \to C$ fired

`readSymbols(words, batch_size)`:
- reconstructs symbol activation from word tuples

---

## Pipeline Versus Syntax

The main model pipeline is a chain of geometric projections between spaces.
These are not shift/reduce operations.

### BasicModel

Forward: `InputSpace → PerceptualSpace → ConceptualSpace → SymbolicSpace → OutputSpace`

Reverse: `OutputSpace → SymbolicSpace → ConceptualSpace → PerceptualSpace → InputSpace`

BasicModel does not use syntax.

### MentalModel

The iterative `MentalModel` loop is:

1. `InputSpace → PerceptualSpace`
2. Repeat `conceptualOrder` times:
   - concatenate `[percepts, previous_symbols]`
   - `ConceptualSpace.forward(...)`
   - `SymbolicSpace.forward(concepts)`
3. `OutputSpace.forward(final_concepts)`

Reverse is the corresponding output-to-input unwinding.

The key distinction is:
- `PerceptualSpace.forward()` accumulates input into percepts
- `ConceptualSpace.forward()` accumulates percepts (and in MentalModel, also
  previous symbols) into concepts
- `SymbolicSpace.forward()` projects concept activation into symbol activation
- Syntax is a separate derivation mechanism layered on top of those states

---

## Parse Tree Output

The model emits an XML parse tree via `parse.derivation_to_xml()`.  Word tuples
from all three spaces are collected (with global Grammar rule IDs) and
reconstructed into a tree by recursive descent over the pre-order word list.

Transition rules ($S \to C$, $C \to P$) are transparent — they do not emit XML
tags.  Terminal rules ($P \to \varepsilon$) emit `<token>` leaves.  All other
rules emit their named tag (`<conjunction>`, `<equals>`, `<not>`, `<union>`,
etc.).

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

- **`type="symmetric"`** — Hopfield-like: learns projection $A$, computes
  association scores $A^T A$, softmax-retrieves the associated pattern.
  Associations are symmetric ($A \equiv B \Leftrightarrow B \equiv A$).
- **`type="hopfield"`** — Modern Hopfield: separate query/key projections,
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

- Symbols are **zero-dimensional** — pure activation scalars, not vectors.
- The invertible layer allows $n_{\text{Concepts}} \neq n_{\text{Symbols}}$.
- `composeSyntax()` runs the symbolic SyntacticLayer and executes the soft
  superposition of swap/equals/part projections.

---

## How Syntax Is Trained

### The Intended Differentiable Mechanism

The syntax code is written to be differentiable:

- `SyntacticLayer.forward()` produces soft `rule_probs`
- `writeConcepts()` and `writeSymbols()` use soft reduce/shift interpolation
- `composeSyntax()` also uses soft superposition over all candidate rules

A downstream loss can backpropagate into rule probabilities, rule-prediction
weights, depth embeddings, and rule-execution layers such as `equals_layer`.

### The Actual Losses in `runBatch()`

`BasicModel.runBatch()` computes three losses:

1. output loss
2. optional reconstruction loss
3. optional embedding loss

There is no explicit syntax loss, parse-tree loss, or rule-label supervision.

### Current Integration Status

Under `MentalModel.xml`:

- syntax is enabled and syntax modules are instantiated
- syntax rules are predicted, syntax states and word tuples are produced
- but `MentalModel.forward()` goes through `SigmaLayer` rather than invoking
  the syntax pipeline directly
- syntax is active as computation and analysis, but not strongly coupled to
  the task loss

The syntax system is best understood as a real, parameterized, differentiable
grammar module — configured by XML rule subsets, implemented with shift/reduce
helpers and batch composition helpers — but not yet placed on a strong loss
path in the stock `MentalModel.xml` loop.
