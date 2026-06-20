# Handoff: Compare Layers.py:Ops() to Conceptual Definitions

## Context

Alec and I (web-Claude, in conversation) have worked out conceptual
definitions for six operations spanning a two-level architecture
(geometric layer of vector regions; symbolic layer of predicates):

1. **Part** / **Whole**
2. **Intersection** / **Union**
3. **Conjunction** / **Disjunction**

The conceptual definitions are captured in
`parthood-and-region-operations.md` (delivered alongside this handoff).

I do not have access to the `basicmodel/` repository. Claude Code does.
This document specifies what to do.

---

## Tasks

### Task 1 — Place the conceptual definitions document

Move/copy `parthood-and-region-operations.md` into the appropriate
location in the basicmodel docs directory. Likely target:

```
basicmodel/doc/parthood-and-region-operations.md
```

If `basicmodel/doc/` already has a naming convention (numeric prefixes,
date prefixes, kebab-case vs snake_case), match it. Adjust the
filename accordingly but preserve the content.

### Task 2 — Read and analyze `Layers.py:Ops()`

Locate the `Ops` class (or namespace / module) in
`basicmodel/Layers.py` and inventory every method/operation it
defines. For each, capture:

- Method name and signature
- Docstring (if any)
- The mathematical operation actually performed (read the body — do
  not trust the name alone)
- Whether it consumes/produces vectors, regions (slab systems with
  `ℓ`/`u` envelopes), or predicates

### Task 3 — Map `Ops()` methods to the six conceptual operations

For each conceptual operation in `parthood-and-region-operations.md`,
identify:

- Which `Ops()` method (if any) implements it
- Which of the two definitions of parthood is in use (per-Whole frame
  projection vs. witness-set elementwise dominance), if applicable
- Whether the implementation matches the conceptual definition or
  diverges (and how)
- Whether the implementation preserves the documented properties
  (e.g., does intersection actually compute `(max ℓ, min u)`? does
  union compute `(min ℓ, max u)`? does conjunction route through
  intersection? does disjunction stay in single-region form
  incorrectly, or correctly escape to a multi-region representation?)

For each conceptual operation **not** present in `Ops()`, note the
gap.

### Task 4 — Inverses

For each conceptual operation, the doc identifies a "possible inverse"
(converse for part/whole, region subtraction for intersection/union,
De Morgan / negation for conjunction/disjunction). For each:

- Is an inverse method implemented in `Ops()`?
- If so, does it match the conceptual inverse, or is it something
  else (e.g., a pseudo-inverse, an additive inverse, a reflection)?
- If not, should it be added? Flag as a gap if the inverse is
  load-bearing for downstream code.

Pay particular attention to whether `Ops()` distinguishes between:
- Lossy inverses that recover *a* solution (some `R₂` such that
  `R₁ ⊓ R₂` matches the target) vs.
- Functional inverses (which don't exist in general for binary
  meet/join)
- Unary inverses like negation/complement (which exit the convex
  slab formalism)

### Task 5 — Produce a comparison report

Write the report to:

```
basicmodel/doc/ops-comparison.md
```

(again, match local naming conventions). The report should have:

1. **Inventory.** Every `Ops()` method, one-line summary each.
2. **Mapping table.** Six rows (the six operations), columns:
   conceptual definition, `Ops()` implementation, match/diverge,
   notes.
3. **Inverse table.** Six rows, columns: conceptual inverse,
   `Ops()` inverse (if any), match/diverge, notes.
4. **Gaps.** Operations or inverses missing from `Ops()` that the
   conceptual model expects.
5. **Divergences.** `Ops()` methods that compute something different
   from what the conceptual model specifies. For each, propose
   either:
   - An update to the conceptual doc (if the implementation reflects
     a deliberate design choice), or
   - A code change to `Ops()` (if the implementation is incidental
     or buggy relative to the design intent).
6. **Recommendations.** Prioritized list of changes (code, docs, or
   both) with rationale.

Do not modify `Layers.py` in this pass. The output of this task is
documentation only.

---

## Specific things to watch for

### Witness set representation

The conceptual model assumes a witness set `V`. How does `Ops()`
represent this? Possibilities:

- Implicit (the standard basis is used; no `V` parameter)
- Explicit parameter (operations take `V` as argument)
- Stored as state on a region/layer object
- Conflated with the prototype set itself

If the implementation uses the standard basis implicitly, this is
*equivalent to elementwise dominance in the standard basis* — flag
this as a design commitment to be made explicit in the doc, not
a divergence.

### Two-scalar parthood

The conceptual model uses two scalars (agreement, disagreement) per
parthood evaluation. Does `Ops()` return:

- A single scalar (and which one)?
- Two scalars?
- A boolean?
- A graded `[0, 1]` value (and via what t-norm or sigmoid)?

### Origin-as-uncertainty

The conceptual model treats the origin as epistemic neutrality. Does
`Ops()` respect this — e.g., does it preserve sign of projection,
distinguish "no evidence" from "negative evidence," handle zero
specially?

### Convexity / single-pointedness

The geometric layer is committed to convex regions. Does `Ops()`
enforce this? Where does disjunction live — does it stay at the
geometric layer (incorrect over-approximation) or escape to a
symbolic / multi-region layer (correct)?

### CWCE interaction

Alec's `CertaintyWeightedCrossEntropy` loss penalizes overconfident
predictions. If `Ops()` participates in the forward pass that feeds
CWCE, the agreement/disagreement scalars should be in a form that
CWCE can consume sensibly (not pre-thresholded to booleans, for
instance). Worth checking but not blocking.

### Differentiability

`Ops()` likely needs to be differentiable for use in the Basic Model
of Cognition's training. `min`/`max` over witnesses are
sub-differentiable but produce sparse gradients. Check whether
`Ops()` uses smoothed versions (softmin/softmax, log-sum-exp), and
if so, document the smoothing as part of the implementation choice.
This may interact with the differentiable sorting (swap-probability
odd-even bubble sort) work — flag any shared infrastructure.

---

## Deliverable summary

Two files in `basicmodel/doc/`:

1. `parthood-and-region-operations.md` — conceptual definitions
   (provided; place verbatim or with naming-convention adjustment).
2. `ops-comparison.md` — comparison report (to be authored by Claude
   Code per Task 5).

No code changes in this pass.
