# Conceptual Introspection — Design Summary

## Overview

This document describes the design for adding a class of **introspective
grammar operations** to BasicModel: local functions that the network can
apply to its own mental content at each step of the conceptual order loop.
These operations expose properties like area, luminosity, and direct
parthood as first-class grammar rules whose outputs feed back into
subsequent computation as activations — making them trainable under the
primary prediction loss rather than requiring separate objectives or
explicit reasoning passes.

---

## Motivation

### The credit-assignment gap in explicit derivation

Grammar operations like `query()`, `equals()`, and `part()` currently have
symbolic derivations (TruthLayer lookup, mereological tree traversal).
These derivations run outside the gradient path: the network encodes the
question and receives the answer, but the connection between question
quality and answer usefulness is severed at the non-differentiable lookup.
The network cannot learn, via backprop, which questions to ask.

### The cognitive science constraint on parthood

Human mereological processing is hierarchical and sequential: determining
whether A is a part of B takes longer when A and B are further apart in
the conceptual hierarchy (Smith & Minda, 1998; Murphy, 2002). This
implies the mind does not compute the generic transitive `part(A, B)`
in a single step. Instead it applies `directPartOf(A, X)` layer by layer
until it reaches B (or concludes it cannot). The transitive closure emerges
from iteration, not from a single operation.

This means the grammar should expose `directPartOf()` and leave the
transitive closure to the network's own iterative process — rather than
exposing the fully general `part()` which has no plausible single-step
cognitive implementation.

---

## Design

### 1. Introspective operations as grammar rules

The following operations are added to the grammar alongside the existing
`conjunction`, `disjunction`, `lift`, `lower`, etc.:

| Rule | Input | Output | Notes |
|------|-------|--------|-------|
| `area(S)` | One symbol activation | Scalar ∈ [0, 1] | Gaussian region size: σ² normalised |
| `luminosity(S, S)` | Two symbol activations | Scalar ∈ [−1, 1] | `area − overlapArea × \|t_A − t_B\|` |
| `directPartOf(S, S)` | Child, parent activations | Scalar ∈ [0, 1] | Kernel overlap at the current conceptual order |

These are registered in `_RULE_METHODS` (SyntacticLayer dispatch table)
alongside the existing grammar rules.  The SyntacticLayer selects among
them via its normal Gumbel-softmax routing — the network learns which
operations to apply where, rather than having them hard-coded.

`directPartOf` is a one-step kernel overlap between the child's Gaussian
region and the parent's Gaussian region at the current conceptual order
level.  It answers: "is the child immediately contained in the parent at
this level of granularity?"

`part(A, B)` (generic, transitive) is **not** exposed as a grammar rule.
The transitive closure is computed by the network across conceptual order
iterations: each iteration applies `directPartOf` and the accumulated
result across orders constitutes the full parthood relation.

### 2. Sidecar channels in the conceptual order loop

At each conceptual order level `t`, the SigmaLayer/PiLayer produces the
main activation `h_t` of shape `[B, N, D]`.  Alongside it, the
introspective rules compute scalar or vector annotations:

```
[area_t, luminosity_t, directPartOf_t]  ← introspective sidecar
```

These are concatenated to `h_t` as additional channels and passed as
input to the next level's SigmaLayer.  The loop structure is:

```
for t in range(conceptual_order):
    h_t = sigma_t(h_{t-1})               # main conceptual activation
    q_t = question_head(h_t)             # question encoding (learned projection)
    a_t = introspect(q_t, h_t)           # answer: area, luminosity, directPartOf
    h_t = concat(h_t, a_t)              # sidecar injection
```

The introspective computations (`area`, `luminosity`, `directPartOf`) are
differentiable functions of `h_t` — they have no parameters of their own
but carry gradient back to the SigmaLayer/PiLayer weights that produced
`h_t`.  The network learns to produce activations whose introspective
properties are useful for downstream prediction.

The `question_head` is a small learned projection (one `nn.Linear` per
introspective type) that maps `h_t` to the query space appropriate for
each operation.  This is the only learnable component added.

### 3. Transitive closure of directPartOf via iteration

Because `directPartOf` is applied at every conceptual order level, and
because the sidecar result is fed forward to the next level's input, the
network accumulates parthood evidence across levels naturally:

- Level 0: Is `word` a direct part of `phrase`?
- Level 1: Is `phrase` a direct part of `clause`?
- Level 2: Is `clause` a direct part of `sentence`?

The accumulated sidecar signal across levels is the network's implicit
computation of the transitive closure.  No explicit `part(A, B)` rule is
needed — it emerges from the depth of the conceptual order loop and the
gradient that shapes each `directPartOf` to be useful in context.

This matches the cognitive science result: deeper hierarchical distance
requires more computation (more loop iterations), exactly as the human
reaction-time data shows.

### 4. External queries via straight-through estimation

For queries directed at external memory (TruthLayer, Mereological Tree)
where the lookup `f(q) → a` is non-differentiable:

```python
a_external = f(q.detach())               # hard lookup, no gradient
next_input = a_external + (q - q.detach())  # STE: forward=a, backward=q
```

Forward pass: the next iteration sees the real answer `a_external`.  
Backward pass: gradient of the downstream loss flows directly through `q`
(the question encoding), bypassing the non-differentiable lookup.

This means the network learns to form questions whose position in activation
space is associated with useful downstream outcomes, without requiring
gradient through the lookup itself.  The signal is coarse — it cannot
distinguish "bad question" from "question unanswerable by memory" — but
it is the correct signal for question-formation learning and is the same
technique used in VQ-VAE codebook training.

External queries use this STE path; internal introspective operations
(area, luminosity, directPartOf) use exact gradient.

---

## Grammar integration

### Registration

Add to `data/grammar.cfg` (or `grammar_shamatha.cfg` for that mode):

```
S -> area(S)
S -> luminosity(S, S)
S -> directPartOf(S, S)
```

Add corresponding `areaForward`, `luminosityForward`, `directPartOfForward`
methods to `SyntacticLayer._RULE_METHODS`.

### Inverse / reverse productions

`area` and `luminosity` are lossy (scalar outputs from vector inputs) — no
inverse is registered.  They are query-only operations whose outputs are
consumed as sidecar annotations, not as compositional inputs to further
grammar rules.

`directPartOf` is also lossy in the same sense.  Its inverse would require
recovering a child from a parent and an overlap score, which is
underdetermined.  Register as lossy; mark explicitly in the grammar table.

### Sidecar vs. full grammar composition

Introspective results are injected as sidecar channels, not as grammar
productions that replace the main activation.  This preserves the existing
grammar semantics: the compositional structure of the sentence is unchanged;
the introspective signals are additional context available to each
subsequent level.

If a full production is needed (e.g., emitting a luminosity value as a
symbol in the derivation), the scalar output can be lifted via `lift()` to
the symbolic level — but this is a separate use case, not the default.

---

## Implementation steps

1. **Implement `area()`, `luminosity()`, `directPartOf()`** as static
   methods (no `nn.Module` needed — these are fixed computations).
   Add to `Layers.py` near `TruthLayer`, alongside the `_gaussian_kernel_overlap`
   helper from the resolve/luminosity plan.

2. **Add `question_head` projections** — one `nn.Linear` per introspective
   type, living on `ConceptualSpace` or `SyntacticLayer`.  Input dim:
   conceptual activation dim.  Output dim: query dim (matches the kernel
   overlap input shape).

3. **Modify the conceptual order loop** (`ConceptualSpace.forward()` or
   `MentalModel.forward()`) to compute sidecar annotations at each level
   and concatenate them before passing to the next sigma stage.

4. **Register grammar rules** in `data/grammar.cfg` and add `*Forward`
   methods to `SyntacticLayer`.

5. **Add STE wrapper** for external queries — a small utility:
   ```python
   def ste_answer(q, f):
       """Straight-through estimator for non-differentiable lookup f."""
       a = f(q.detach())
       return a + (q - q.detach())
   ```

6. **Tests:**
   - `area()` returns values in [0, 1].
   - `luminosity()` returns values in [−1, 1].
   - `directPartOf()` returns values in [0, 1]; higher for overlapping
     activations, lower for orthogonal ones.
   - Gradient flows through sidecar into sigma weights (backprop test).
   - STE: forward sees `a`, backward sees gradient w.r.t. `q`.
   - Transitive parthood: three-level hierarchy resolves correctly across
     three conceptual order iterations.

---

## Relationship to existing plans

- **resolve/luminosity handoff** (`2026-05-04-resolve-luminosity-handoff.md`):
  implements `luminosity()` on `TruthLayer` and fixes `resolve()`.  The
  `luminosity()` method defined there is the building block for the grammar
  rule here.

- **Shamatha Speech** (`2026-04-28-shamatha-speech-contiguity-handoff.md`):
  `area()` and `directPartOf()` integrate naturally with contiguity
  checking (`where_connected`, `when_continuous`) — the same Gaussian
  kernel σ underlies all three.

- **AR Sentence Prediction** (todo.md): the sidecar introspective channels
  contribute to the STM gamma trace across sentences, extending the scope
  of what the LTM/STM predictors can condition on.
