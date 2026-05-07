# Introspective Subsymbolic — Combined Design Spec

## Overview

This document combines two prior specs and one set of forward-looking
requirements into a single coherent design. It is broader in scope
than either prior spec on its own; not all of it ships in one
checkin. Phasing is explicit (§5).

**Source specs:**

- `2026-05-04-conceptual-introspection-handoff.md` — adds `area`,
  `luminosity`, and a containment operation (renamed; see §1) as
  introspective grammar operations whose outputs feed back into
  the conceptual-order loop and are trainable under the primary
  prediction loss.
- `2026-05-05-subsymbolic-knowing-handoff.md` — adds
  `SubsymbolicSpace`: a parallel re-entrant Space whose only
  operations are `SigmaLayer` then `PiLayer`, whose event tensor
  sums with `SymbolicSpace.event` at the next conceptual order's
  combined input, and which embodies the imagistic / felt-sense
  half of dual-coding.

**Forward-looking requirements (this spec's framing):**

- Sentences are sometimes composites of ideas. Questions relate
  two ideas (`is(subject, predicate)?`, `part(x, y)?`); the IS of
  definition is `equals(x, y)`.
- Definitions, when sentence confidence is high, write explicit
  parthood / equality onto `WordSpace`'s **Mereonomy** (renamed
  from `MereologicalTree`).
- Memory of previous sentences requires prediction relating one
  sentence to the next; the model stores sentences explicitly and
  trains a sentence-to-sentence predictor.
- Truth statements arrive as conceptual bivectors (ideas) and / or
  meronymic relationships and are processed by both the
  conceptual / subsymbolic stack and the Mereonomy.
- Query quality is scored by the **change in `luminosity()`
  versus the Truth** that the query induces.

The 05-05 doc closes with a one-paragraph teaser: "introspection
*is* the Subsymbolic loop's content." This spec realises that
paragraph and extends the surrounding architecture so that the
introspective measures defined here are usable downstream by the
question-handling, definition-binding, sentence-prediction, and
query-scoring machinery sketched above.

---

## 1. Renamings

### 1.1 `directPartOf` → `subset` (geometric)

The 05-04 plan called the kernel-overlap containment operation
`directPartOf`, framed via mereological parthood and
transitive-closure-via-iteration. This combined spec renames it
`subset`. **Sets are intransitive parts** — set membership /
containment does not compose hierarchically the way mereological
parthood does (a member of a member is not a member). What the
kernel-overlap measure actually computes — "how much of region A
sits inside region B at this granularity" — is the set-style
containment relation, not parthood.

`subset` is the *geometric* operation — continuous, parameter-free,
realised as a chart-driven composition of `union` /
`intersection` / complement on bivectors (Phase A 'C'-tier
dispatch — see §2.2).

`part` is reserved for the *symbolic* parthood relation — discrete,
stored on the Mereonomy, written on definitions and looked up on
questions (§3.3, §3.4). The two coexist; they are not aliases.

### 1.2 `MereologicalTree` → `Mereonomy`

`WordSpace.MereologicalTree` is renamed to `WordSpace.Mereonomy`
throughout. A mereonomy is precisely the data structure in question
— a hierarchical relation graph capturing parthood and equality —
and the renamed class will host the new `equals` and `part` symbolic
relations (§3.3).

The rename touches every reference (class definition, imports,
attribute accesses, tests, docs). Phase 1c lands the rename
together with the symbolic relation API; the rename and the API
addition are atomic — no transitional alias.

### 1.3 `luminosity` is one operation with two uses

`luminosity` is defined once and used in two places:

- As a **per-slot 'C'-tier `GrammarLayer`** (`LuminosityLayer`,
  §2.3) writing into the slot's `LUMINOSITY_IDX` reserved
  channel during Phase A on the bivector activation.
- As a **query quality metric**, scoring the change in
  `TruthLayer.luminosity` the query induces against the stored
  Truth (§3.5).

Both usages call the same arithmetic; the second usage is the
first usage measured before-and-after a candidate query.

---

## 2. Phase 1b — tier-mapped grammar dispatch within `SymbolicSpace`

### 2.0 Architectural simplification: `SubsymbolicSpace` dissolves

The user's framing: **`SubsymbolicSpace` as a separate Space is
not necessary**. The "subsymbolic" content is precisely what
`SymbolicSpace` carries *before* its codebook quantification —
the bivector activation in `subspace.activation` pre-snap. The
"symbolic" content is what `SymbolicSpace` carries *after* the
codebook snap and any post-codebook grammar dispatch.

There are not two parallel re-entrant Spaces; there is one
Space (`SymbolicSpace`) running through phased processing, and
"thinking" / "knowing" name the two phases of its event:

- **`.thinking`** — `SymbolicSpace`'s pre-codebook bivector
  state, after **'C' tier** (Conceptual) grammar dispatch on the
  bivector activation. The imagistic / felt-sense aspect of
  Paivio's dual coding.
- **`.knowing`** — `SymbolicSpace`'s post-codebook state, after
  **'S' tier** (Symbolic) grammar dispatch. The verbal /
  propositional aspect of dual coding.

Their summation `thinking + knowing` happens **automatically**
because both are aspects of the same `SymbolicSpace.event`,
written by the same conceptual-order loop's round-trip stack of
operations.

This simplification supersedes the prior plan's separate
`SubsymbolicSpace` re-entrant loop (the 05-05 handoff's Phase
1a). The existing Phase 1a class can be retired or repurposed in
Phase 1c (§5).

### 2.1 Tier mapping

The existing `GrammarLayer.tier` attribute already supports the
mapping:

| Tier | Meaning                                  | Where it dispatches in `SymbolicSpace.forward`     |
|------|------------------------------------------|-----------------------------------------------------|
| 'P'  | Percepts                                 | Upstream (`PerceptualSpace`); not relevant here.    |
| 'C'  | Concepts (pre-codebook bivectors)        | **Pre-codebook phase** (the thinking phase).        |
| 'L'  | Logical (bivector-typed; chart-bindable) | At C or S depending on chart binding (§2.4).        |
| 'S'  | Symbols (post-codebook codebook content) | **Post-codebook phase** (the knowing phase).        |

The chart's per-space `SyntacticLayer` already gates dispatch by
`tier`. The new wiring routes the 'C' tier dispatch to fire on
the bivector activation before the codebook snap, and 'S' tier
dispatch to fire after — two phases of the same forward.

### 2.2 No new dispatch logic — chart already handles it

The existing chart / `SyntacticLayer` infrastructure already
handles tier-aware dispatch: chart cells have tier bindings;
`GrammarLayer`s with matching `tier` fire at matching cells.
**No new `dispatch(tier=...)` method is introduced.** Phase 1b
just registers the new layers in `GRAMMAR_LAYER_CLASSES` and
adds grammar productions; the chart fires them as part of its
normal CKY pass, alongside the existing 'S'-tier ops and the
existing 'L'-tier `IntersectionLayer` / `UnionLayer`.

The existing `IntersectionLayer` and `UnionLayer` (tier 'L',
`bin/Layers.py:2146`, `:2193`) already bind to whatever tier
the chart cell carries — their docstring confirms it: *"the
operands' upstream tier (C vs S codebook activation) is
determined by the chart binding, not by this layer."* When the
chart binds them at C-tier cells (pre-codebook bivectors), they
fire on bivector activations; when bound at S-tier cells
(post-codebook), they fire on codebook-snapped content. **Same
op, both tiers, no per-tier code change.**

The codebook snap happens at its existing point in
`SymbolicSpace.forward` — Phase 1b doesn't change where or how
the snap occurs. What changes is **what flows through the
snap**: the augmented `slot_dim` (raw + introspective trailing
channels) travels through, with the codebook scope restricted
to the raw channels (§2.5) so the introspective channels — and
any C-tier ops' results that the chart wrote into them — survive
unchanged. The S-tier ops in subsequent chart cells then read
the codebook-snapped content, with the trailing introspective
channels available alongside.

The `.thinking` / `.knowing` framing of §2.0 is therefore a
description of the chart's own behaviour, not a new pipeline:
chart cells bound at 'C' carry thinking; cells bound at 'S'
carry knowing; the final derivation's event tensor integrates
both. No explicit phase boundaries in code. **`SymbolicSpace`
gets new `GrammarLayer`s in its grammar config; nothing else
changes about how its forward runs.**

### 2.3 Unary introspection on bivectors: `area` and `luminosity`

`area` and `luminosity` are added as new **'C'-tier unary**
GrammarLayers that fire in Phase A on the bivector activation.
Both are parameter-free differentiable functions; both write
their scalar outputs into reserved trailing channels of the
slot's bivector representation.

```python
class AreaLayer(GrammarLayer):
    """``C -> area(C)`` -- per-slot Gaussian region area, σ² normalised."""
    rule_name = "area"
    arity = 1
    tier  = 'C'
    reads_activation = True
    ...

class LuminosityLayer(GrammarLayer):
    """``C -> luminosity(C)`` -- per-slot signed clarity measure.

    Slot's own area minus its self-contradiction penalty (the
    overlap of the slot's pos and neg poles weighted by |DoT|).
    Zero on a perfectly committed slot; negative on internally
    contradictory slots. Range: [-1, 1].
    """
    rule_name = "luminosity"
    arity = 1
    tier  = 'C'
    reads_activation = True
    ...
```

Both call into `_gaussian_kernel_overlap` (added by the
prerequisite `2026-05-04-resolve-luminosity-handoff.md`).

**Channel layout.** `area` and `luminosity` write into the last
two channels of every slot's bivector representation:

```
slot_dim = slot_dim_raw + 2          # +2 for area, luminosity

slot encoding:
  [ ... raw bivector content ... | area | luminosity ]
```

Channel indices live as class constants on `Space`
(`AREA_IDX = -2`, `LUMINOSITY_IDX = -1`). Pre-codebook content
fills `area` and `luminosity`; the codebook snap may zero them
or carry them through depending on whether the codebook is
defined over the full slot_dim or just the raw content (§2.5).

### 2.4 `SwapLayer` (existing) and `CopyLayer` (new)

Both fire at 'S' tier in Phase C, alongside the existing
`equals`, `part`, `lift`, `lower`, etc.

`SwapLayer` already exists at `bin/Layers.py:3103` (parameter-
free `GrammarLayer`, `tier='S'`, `forward(left, right) →
right`). No code change.

`CopyLayer` is added in this checkin as a new `GrammarLayer`
adjacent to `SwapLayer`:

```python
class CopyLayer(GrammarLayer):
    """``S -> copy(S, S)`` -- copy the left argument to the parent slot.

    Dual of SwapLayer: forward returns ``left``. Lossy; reverse
    is the symmetric ``(parent, parent)`` pseudo-inverse.
    Parameter-free.
    """
    rule_name = "copy"
    arity = 2
    invertible = False
    lossy = True
    tier  = 'S'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, left, right):
        return left

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)
```

Register `'copy': CopyLayer` in `GRAMMAR_LAYER_CLASSES` and add
`S -> copy(S, S)` to `data/grammar.cfg` (and
`grammar_shamatha.cfg` if applicable).

**Training signal.** Both `swap` and `copy` are parameter-free.
The gradient signal that trains them flows through
`Grammar.rule_probability('swap')` /
`rule_probability('copy')` (global per-rule probabilities) and
the chart's CKY pair-selection state — the existing mechanism,
already differentiable under the standard prediction loss.

`forward(left, right) → left` is the proposed default for
`CopyLayer` as the canonical dual of swap. Alternative bodies
(mix, marker metadata) are open.

### 2.5 Codebook scope across the introspective channels

Open question for implementation: whether `SymbolicSpace`'s
codebook is defined over `slot_dim_raw` only (introspective
channels pass through unchanged) or over the full
`slot_dim_raw + 2` (introspective channels get codebook-snapped
along with everything else).

Recommended: **codebook over `slot_dim_raw` only.** Rationale:
- `area` and `luminosity` are continuous scalars; codebook
  quantising them collapses degree-of-truth nuance.
- The introspective channels carry information that is more
  useful as continuous values into Phase C and onward (e.g.
  `sentenceConfidence()` integrates `luminosity` over orders).
- Symbolic ops in Phase C that need the introspective values can
  read them directly from the post-codebook event tensor's
  trailing channels.

The implementation surface is the codebook lookup machinery in
`Spaces.py` (around `_snap_content` / `codebookDistance`); the
existing `aux` tail-passthrough convention at `Spaces.py:472`
is the natural place for the introspective channels to ride.

### 2.6 Mode gating, sentence boundary

The 05-05 handoff's `<architecture><mode>grammar|parallel</mode>`
mode-gating mechanism dissolves alongside `SubsymbolicSpace`:
there is one re-entrant loop (the symbolic loop, with its
phased dispatch), and gating is a per-phase question.

For Phase 1b: `mode = grammar` continues today's behaviour
unchanged (Phase A is a no-op when no 'C'-tier rules are
registered; the existing 'S'-tier-only flow runs). The "parallel"
mode is reinterpreted as "Phase A's 'C'-tier dispatch is
non-empty"; explicit XML knob deferred to Phase 1c.

Sentence boundary: `SymbolicSpace.event` clears at sentence
start (existing 05-05 behaviour); both thinking and knowing
aspects clear together because they are aspects of the same
event.

---

## 3. Future phases — composite-sentence handling

These sections describe the destination architecture so that the
Phase 1b design above is built compatibly. **Implementation work
beyond Phase 1b is out of scope for this checkin** but motivates
the design choices made in §2.

### 3.1 Sentences as composites of ideas — stack semantics on `nVectors`

A sentence is one or more ideas, connected by binary relating
operations. Composite handling treats
`ConceptualSpace.nVectors` (canonically 7 ± 2, Miller's
working-memory number) as a **stack**: atomic ideas push;
relating operations consume the top two slots and collapse to a
result.

Each binary operation reads (A) the previous idea (top-1) and
(B) the incoming idea (top). The **top two slots are privileged**:

- **Lift / lower** (grammar, 'S' tier) operate on top-of-stack:
  `lift` turns the activation into a symbolic codebook entry;
  `lower` turns a symbol back into an activation.
- **Union / intersection** (lattice, 'L' tier bound at 'C' in
  Phase A) on top and top-1 fire via the chart's standard
  dispatch — the same way 'S' tier binary ops fire on
  post-codebook content. The chart binds the operands; the
  Phase A `'C'` dispatch (§2.2) executes them on the bivector
  activation, over all chart positions.
- **Higher-level operations** — `subset(top, top-1)`,
  `equals(top, top-1)`, `part(top, top-1)` — are **staged
  compositions** built from the chart's iterated firing of
  union / intersection / complement across conceptual orders;
  they are not Phase 1b built-in scalars.
- **Symbolic relations** — `equals(top, top-1)` and `part(top,
  top-1)` — additionally write to the Mereonomy on
  high-confidence definitions / assertions (§3.3) when fired in
  Phase C ('S' tier dispatch).

Atomic-idea types include conceptual bivectors (`SymbolicSpace`
content pre-codebook). Binary relating operations include:

- `is(subject, predicate)` — predicate-question or
  predicate-assertion (the copula).
- `part(x, y)` — parthood-question or parthood-assertion.
- `equals(x, y)` — definition / identity.

**Today the conceptual-space codebook constrains nVector content,
suppressing this stack potential.** Phase 1b is built around the
stack interpretation without removing the codebook (the codebook
handling stays as today; the stack semantics ride alongside).
The chart's standard CKY pair-selection logic supplies the
top-of-stack binding for each binary op — the per-cell binding
already encodes "which two operands compose at this position".

Stack accounting (explicit pointer state, push/pop transitions
across conceptual orders) is a Phase 1d concern; for Phase 1b /
1c, positional convention (slot 0 = top) is sufficient.

### 3.2 Question vs definition vs assertion

- **Question** — terminates with a question marker; the model
  treats the relating-op as a query.
  - Predicate questions (`is(subject, predicate)?`) primarily
    read from `TruthLayer`; if `TruthLayer`'s analysis classifies
    the query's referent as a relative truth (§3.3), the lookup
    routes through the Mereonomy instead.
  - `part(x, y)?` and `equals(x, y)?` read directly from the
    Mereonomy.
  - STE wraps any non-differentiable lookup so gradient flows
    through the question-formation path.
- **Definition** — `X is Y` or `X equals Y`. Write candidate to
  the Mereonomy under `sentenceConfidence()` gating (§3.4).
- **Assertion** — predicate truth statement. Write candidate to
  `TruthLayer` first; `TruthLayer` analyses it for ultimate vs.
  relative truth and, when confidence is sufficient, promotes the
  classification result to the Mereonomy with appropriate
  per-symbol DoTs (§3.3).

The conceptual-order loop produces, alongside its standard event
output, a per-sentence type tag (`question | definition |
assertion`). Phase 1c lands the tag emission, the read paths, and
TruthLayer's ultimate-vs-relative analysis; Phase 1d the
sentence-buffer / sentence-prediction / query-scoring layer.

### 3.3 Mereonomy hosts both ultimate and relative truths, with per-symbol DoT

`Mereonomy` (renamed from `MereologicalTree`) is the persistent
truth store. **Every Mereonomy entry carries a Degree of Truth
(DoT) for each referenced symbol**, so the same graph stores two
kinds of truth without separate machinery:

- **Ultimate truth** — leaf entry: a symbol with its DoT,
  mapping to an object in the world. No parthood relation.
  - Stored: `(symbol_x, DoT_x)`
  - Semantics: "x is the case to degree DoT_x", with x mapping to
    a concrete object.
- **Relative truth** — interior entry: two or more symbols in a
  parthood (or equality) relation, each carrying its own DoT.
  - Stored: `part(x, y)` with `(symbol_x, DoT_x)` and
    `(symbol_y, DoT_y)`; or `equals(x, y)` with the same.
  - Semantics: "x is a part of y, with x being the case to
    degree DoT_x and y being the case to degree DoT_y" — the
    relation is qualified by both endpoints' truth values.

`TruthLayer`'s role becomes intake and analysis: incoming
linguistic truths are classified into ultimate vs. relative and,
when sentence confidence is sufficient, promoted to the Mereonomy
as the entry of matching kind. `TruthLayer` retains its working
truth field for transient truths that have not (yet) been
promoted; the Mereonomy is the long-term storage.

| Linguistic input matches                       | Mereonomy entry kind       |
|------------------------------------------------|----------------------------|
| single object / predicate                      | ultimate `(x, DoT_x)`      |
| relation between existing symbols              | relative `part / equals`   |
| below confidence threshold                     | not written; held in TruthLayer only |

`equals(x, y)` and `part(x, y)` write APIs (Phase 1c):

- `equals(x, y)` — bidirectional identity. Both symbols
  reconciled to a shared DoT per a merge rule (placeholder:
  weighted average; revisable).
- `part(x, y)` — directed parthood with distinct per-endpoint
  DoTs. Mereonomy's transitive closure makes `part(x, z)`
  recoverable for any `z` reachable via the parthood DAG; the DoT
  of a transitively-derived `part(x, z)` is computed from the
  endpoint DoTs along the path (placeholder: min along path;
  revisable).

Reads happen on questions via the STE wrapper. Lookups return
the relation **plus the per-symbol DoTs**, so question answers
carry truth qualification rather than bare relation membership.

### 3.4 Sentence-confidence gating — `sentenceConfidence()` method

Definitions and parthood assertions write to the Mereonomy only
when sentence-level confidence exceeds a threshold. The score is
encapsulated in a method so the formula can be revisited without
touching the call sites in Mereonomy / TruthLayer write paths:

```python
def sentenceConfidence(self, sentence_state) -> float:
    """Per-sentence confidence score in [0, 1].

    Combines the sentence's prediction loss with the change in
    per-slot luminosity (the `LUMINOSITY_IDX` channel populated
    by `LuminosityLayer` in Phase A) across the sentence's
    conceptual orders. Exact arithmetic deferred — placeholder
    pending empirical calibration.
    """
    ...
```

The threshold (`<architecture><mereonomyWriteThreshold>`,
placeholder default `0.7`) and the formula inside the method are
both revisitable. Below threshold, writes are skipped; the
Mereonomy stays consistent under uncertain input.

This connects the Phase 1b `LUMINOSITY_IDX` channel directly
to a downstream gating decision: the per-order luminosity values
produced by `LuminosityLayer` accumulate across a sentence and
feed the confidence method on sentence completion.

### 3.5 Query scoring via Δ luminosity vs Truth

A user query is scored on the change in `luminosity(TruthLayer)`
that answering it would produce, relative to the prior. High
scores mark queries that are informative — they shift the truth
field's coherence — versus queries that are redundant
(`Δ luminosity ≈ 0`) or harmful (`Δ luminosity < 0`,
contradiction-introducing).

```
score(query) = luminosity(TruthLayer | answer(query)) − luminosity(TruthLayer)
```

The arithmetic is the existing `TruthLayer.luminosity` (per the
05-04 resolve / luminosity handoff) measured before and after a
hypothetical answer. The Phase 1b `LuminosityLayer` (§2.3) and
`TruthLayer.luminosity` share the formula and the σ convention;
one fires per-slot in Phase A on the bivector activation, the
other operates on `TruthLayer`'s stored truth field as a metric.

### 3.6 Sentence-level memory and prediction

Sentences are stored explicitly in a sentence buffer, distinct
from the per-token / per-order working memory:

- `WordSpace.sentenceBuffer` — bounded ring buffer of recent
  sentence representations (final-order conceptual activation +
  sentence type tag + confidence score).
- `WordSpace.sentencePredictor` — a small predictor (analogous
  to the existing percept / symbol predictors) trained on
  next-sentence prediction loss. Conditioned on the buffer; emits
  a predicted next-sentence representation.

This is a third memory tier alongside STM (`concept_states`) and
LTM (`TruthLayer` / Mereonomy). Phase 1d adds the buffer and
predictor; the prediction loss adds to the optimizer's loss
sum without disturbing existing token-level objectives.

---

## 4. How Phase 1b feeds the future phases

Phase 1b is the load-bearing piece. The choices in §2 are made
*because* the future phases need them:

| Phase 1b choice                              | Phase 1c / 1d use                                                    |
|----------------------------------------------|-----------------------------------------------------------------------|
| `luminosity` unary 'C'-tier op (Phase A)     | `sentenceConfidence()` (§3.4); Mereonomy write gating                 |
| `luminosity` formula shared with `TruthLayer`| Query scoring `Δ luminosity vs Truth` (§3.5)                          |
| `area` unary 'C'-tier op (Phase A)           | Component of confidence; salience measure                             |
| 'C'-tier dispatch on bivectors (Phase A)     | `subset`, predicate matching, etc. fire as chart-driven compositions |
| 'S'-tier dispatch post-codebook (Phase C)    | `swap`, `copy`, `equals`, `part`, etc. fire on codebook content       |
| `CopyLayer` registered in grammar            | Phase 1c / 1d question routing and definition writes                  |
| `.thinking` / `.knowing` framing             | Single `SymbolicSpace.event` carries both aspects; sum is automatic   |
| STE wrapper                                  | Question-time Mereonomy / TruthLayer lookups (§3.2, §3.3)             |
| Reserved channel constants on `Space`        | Future DoT field plugs into Mereonomy entries (§3.3)                  |

Every measure that downstream gating / scoring will need is
either landed in Phase 1b (area / luminosity as Phase-A 'C'-tier
ops) or expressible via Phase 1b's chart dispatch (higher-level
operations as compositions of union / intersection / complement).
Phase 1c / 1d build on the dispatch layer rather than revisiting
it.

---

## 5. Phasing

- **Phase 1a** (landed in 05-05): `SubsymbolicSpace` was added
  as a separate Space with Sigma·Pi composition, mode gating,
  sentence-boundary reset, combined-input plumbing. **This is
  superseded by the §2.0 simplification:** `SubsymbolicSpace`
  is no longer a separate Space. The Phase 1a code is retired
  / repurposed as part of Phase 1c (its Sigma·Pi composition
  remains useful as part of `SymbolicSpace`'s pre-codebook
  pathway, but the standalone class dissolves).
- **Phase 1b** (this checkin):
  - Augment slot embedding by `nIntrospect = 2` on
    `SymbolicSpace` (`area`, `luminosity`).
  - Add `AreaLayer` and `LuminosityLayer` as new 'C'-tier
    `GrammarLayer` subclasses in `bin/Layers.py`; register
    `'area'` and `'luminosity'` in `GRAMMAR_LAYER_CLASSES`;
    add `C -> area(C)` and `C -> luminosity(C)` productions to
    grammar configs.
  - Add `CopyLayer` as a new 'S'-tier `GrammarLayer` adjacent
    to the existing `SwapLayer` (`bin/Layers.py:3103`); register
    `'copy': CopyLayer`; add `S -> copy(S, S)` to grammar configs.
  - Wire `SymbolicSpace.forward` to dispatch the chart
    Phase-A → codebook snap → Phase-C sequence (§2.2). The
    per-space `SyntacticLayer` does the tier-gated dispatch
    using existing infrastructure.
  - Wire `_gaussian_kernel_overlap` (prerequisite handoff) into
    the new `AreaLayer` and `LuminosityLayer`.
  - Decide codebook scope (raw-only vs raw+introspective);
    recommended raw-only (§2.5).
  - Add `ste_answer` utility in `Layers.py`.
  - Tests for area / luminosity ranges, copy forward / reverse,
    Phase A vs Phase C dispatch correctness, codebook-passthrough
    of introspective channels, backward compatibility.
- **Phase 1c** (follow-up checkin):
  - **Retire / repurpose the standalone `SubsymbolicSpace`**
    class (Phase 1a's contribution). Its Sigma·Pi composition
    can be folded into `SymbolicSpace`'s pre-codebook pathway
    if useful, or removed entirely. The `<architecture><mode>`
    knob's "parallel" branch is replaced by Phase A's 'C'-tier
    dispatch firing.
  - Rename `MereologicalTree` → `Mereonomy` (atomic; touches every
    reference).
  - Extend Mereonomy entries to carry per-symbol DoTs (§3.3):
    ultimate-truth leaves and relative-truth interior entries
    encode in the same graph.
  - Add `Mereonomy.equals(x, y)` and `Mereonomy.part(x, y)` write
    APIs (with per-symbol DoTs).
  - Add `Mereonomy.lookup_equals(x)` and `Mereonomy.lookup_part(x)`
    read APIs (return relation + per-symbol DoTs; used via STE on
    questions).
  - Implement `sentenceConfidence()` method (§3.4); formula
    revisitable.
  - `TruthLayer` analysis pass: classify incoming linguistic
    truths into ultimate vs. relative and route promotions to the
    Mereonomy.
- **Phase 1d** (follow-up checkin):
  - Sentence type tag emission on the conceptual-order loop.
  - Question-type recognition: route question sentences to
    Mereonomy / TruthLayer reads.
  - `WordSpace.sentenceBuffer` and `WordSpace.sentencePredictor`.
  - Query scoring (`Δ luminosity vs Truth`) as a user-facing
    metric.
- **Phase 2** (deferred, jointly with 05-05's Phase 2): both
  loops active simultaneously; manasikāra gating; papañca
  regularisation.

The user's framing — "doing all the work in a single checkin is
probably too ambitious" — anchors this phasing. Phase 1b is the
checkin in scope; later phases are scoped only enough to ensure
Phase 1b's design accommodates them.

---

## 6. Implementation steps (Phase 1b detail)

### Step 1. Land prerequisite

Step 2 of `2026-05-04-resolve-luminosity-handoff.md` —
`_gaussian_kernel_overlap` helper near `TruthLayer` in `Layers.py`.

### Step 2. Add channel-index constants on `Space`

In `bin/Spaces.py`, on the `Space` base class:

```python
nIntrospect    = 2
AREA_IDX       = -2
LUMINOSITY_IDX = -1
```

Negative indexing keeps the layout robust to future raw-content
extensions.

### Step 3. Augment `SymbolicSpace`'s slot embedding

XML knob `<architecture><nIntrospect>` (default `2`). The Model
factory passes the augmented per-slot dimension. Legacy
compatibility: `nIntrospect = 0` recovers pre-spec behaviour
exactly.

### Step 4. Add `AreaLayer` and `LuminosityLayer` as 'C'-tier `GrammarLayer`s

In `bin/Layers.py`, alongside the existing 'L'-tier
`IntersectionLayer` (~line 2146) and `UnionLayer` (~line 2193):

```python
_DEFAULT_SUBSYMBOLIC_SIGMA = 0.1

class AreaLayer(GrammarLayer):
    """``C -> area(C)`` -- per-slot Gaussian region area."""
    rule_name = "area"
    arity = 1
    invertible = False
    lossy = True
    tier  = 'C'
    reads_activation = True

    def __init__(self, sigma=None):
        super().__init__(0, 0)
        self.sigma = sigma if sigma is not None else _DEFAULT_SUBSYMBOLIC_SIGMA

    def forward(self, x):
        # x: bivector activation [B, V, 2] or augmented [B, V, 2 + ...]
        # Returns the same shape with AREA_IDX channel populated.
        ...

    def reverse(self, parent):
        return parent             # unary; pseudo-inverse identity

    def compose(self, x):
        return self.forward(x)

    def generate(self, parent):
        return self.reverse(parent)


class LuminosityLayer(GrammarLayer):
    """``C -> luminosity(C)`` -- per-slot signed clarity measure."""
    rule_name = "luminosity"
    arity = 1
    invertible = False
    lossy = True
    tier  = 'C'
    reads_activation = True

    def __init__(self, sigma=None):
        super().__init__(0, 0)
        self.sigma = sigma if sigma is not None else _DEFAULT_SUBSYMBOLIC_SIGMA

    def forward(self, x):
        # area minus self-contradiction penalty
        ...

    def reverse(self, parent):
        return parent

    def compose(self, x):
        return self.forward(x)

    def generate(self, parent):
        return self.reverse(parent)
```

Both write into the slot's `AREA_IDX` / `LUMINOSITY_IDX`
trailing channels and call `_gaussian_kernel_overlap` for the
arithmetic. Register `'area'` and `'luminosity'` in
`GRAMMAR_LAYER_CLASSES`. Add `C -> area(C)` and
`C -> luminosity(C)` to `data/grammar.cfg` and
`data/grammar_shamatha.cfg`.

### Step 5. Add `CopyLayer` as an 'S'-tier `GrammarLayer`

`SwapLayer` already exists at `bin/Layers.py:3103` and is
registered in `GRAMMAR_LAYER_CLASSES['swap']`. No change needed
there.

Add `CopyLayer` adjacent to `SwapLayer`, following the same
pattern (parameter-free, lossy, `arity=2`, `tier='S'`,
`forward(left, right) → left` per §2.4). Add
`'copy': CopyLayer` to `GRAMMAR_LAYER_CLASSES` and
`S -> copy(S, S)` to the grammar configs.

Training signal for both `swap` and `copy` flows through the
existing `rule_probability` + chart pair-selection mechanism.

### Step 6. (No `SymbolicSpace.forward` change required)

The chart's existing CKY pass already dispatches `GrammarLayer`s
by tier; the new layers (`AreaLayer`, `LuminosityLayer`,
`CopyLayer`) integrate without code changes to `forward`.

Confirm by audit:
- The chart binds cells with tier annotations matching the
  grammar productions added in Steps 4 / 5.
- Existing `IntersectionLayer` / `UnionLayer` already bind at
  whatever tier the chart cell carries; the new productions
  put 'C'-tier cells in the chart that the chart already knows
  how to fire.
- The codebook snap (`resolve` and friends) happens at its
  existing place in `SymbolicSpace.forward`. The augmented
  `slot_dim` flows through it; the codebook scope decision in
  Step 7 governs whether trailing channels survive.

If the audit surfaces an actual gap (e.g. the chart's tier
annotation doesn't propagate the new C-tier productions
correctly, or the SyntacticLayer's per-tier dispatch hook is
absent for C), file the gap as the *real* implementation work
in this step. Do **not** introduce a parallel `dispatch(tier=)`
API; instead extend the existing chart / SyntacticLayer
mechanism in place.

### Step 7. Codebook scope

Decide on the codebook's scope across the augmented slot_dim:
recommended **codebook over `slot_dim_raw` only**, with the
trailing introspective channels passing through unchanged via
the existing `aux` tail-passthrough convention
(`Spaces.py:472`). Concretely:

- `Basis.codebookDistance` (and friends) operate on the leading
  `slot_dim_raw` channels only.
- `Basis._snap_content` writes the snapped value into the
  leading channels; trailing introspective channels are
  preserved from the input.

### Step 8. Validate config

Extend the existing config validator (`Models.py` ~lines 2540 and
~4980 — `BasicModel` and `MentalModel`) to confirm `nIntrospect`
on `SymbolicSpace` matches what the new grammar productions
expect.

### Step 9. STE utility

`def ste_answer(q, f)` in `Layers.py` near `TruthLayer`.

### Step 10. Tests

**Unit-level (unary introspection grammar layers).**

- `test_area_layer_range` — `AreaLayer.forward` writes values in
  `[0, 1]` to `AREA_IDX` over random bivector activations; other
  channels untouched.
- `test_luminosity_layer_range` — `LuminosityLayer.forward`
  writes values in `[-1, 1]` to `LUMINOSITY_IDX`; equals area on
  a self-consistent slot; negative on internally contradictory
  slots.
- `test_introspective_layers_have_grads` — backward through
  either layer's output reaches the upstream weights that
  produced the bivector activation.

**Unit-level (grammar dispatch via the existing chart).**

- `test_C_tier_fires_pre_codebook` — when the grammar registers
  `area` / `luminosity` and the chart binds them at C-tier
  cells, they fire on the bivector activation (pre-codebook).
  Verified by checking that `AREA_IDX` / `LUMINOSITY_IDX` carry
  computed values after `SymbolicSpace.forward`.
- `test_S_tier_fires_post_codebook` — `swap` / `copy` /
  existing 'S'-tier ops fire on the codebook-snapped content
  (post-codebook). Verified by setting up a derivation that
  routes through both tiers.
- `test_codebook_passthrough_introspective` — the codebook snap
  preserves the `AREA_IDX` and `LUMINOSITY_IDX` channels
  unchanged (raw-only codebook scope per Step 7).
- `test_copy_layer_forward_returns_left` — `CopyLayer.forward(a,
  b) == a`.
- `test_copy_layer_grammar_registration` — `'copy'` is in
  `GRAMMAR_LAYER_CLASSES`; instantiation via
  `GRAMMAR_LAYER_CLASSES['copy']()` returns a `CopyLayer`
  instance.
- `test_area_layer_grammar_registration` — `'area'` and
  `'luminosity'` are in `GRAMMAR_LAYER_CLASSES`; instantiation
  returns the matching `GrammarLayer` instances.

**Integration.**

- `test_thinking_knowing_split` — `SymbolicSpace.event` after
  Phase A but before Phase B has the introspective channels
  populated and the bivector content reflects 'C'-tier ops; after
  Phase C has the codebook-snapped content reflecting 'S'-tier
  ops.
- `test_combined_input_carries_event` — at order `t+1`,
  `ConceptualSpace`'s combined input reflects the previous
  order's full `SymbolicSpace.event` (thinking + knowing
  integrated).

**STE.**

- `test_ste_forward_real_answer`.
- `test_ste_backward_identity`.

**Backward compatibility.**

- `test_baseline_unchanged_with_nIntrospect_zero` —
  `<nIntrospect>0</nIntrospect>` recovers pre-spec output exactly.
- `test_no_C_tier_rules_no_op_phase_A` — when no 'C'-tier rules
  are registered (legacy grammar), Phase A is a no-op and the
  flow matches today's `SymbolicSpace.forward`.

**Unit-level (grammar-layer routing).**

- `test_copy_layer_forward_returns_left` — `CopyLayer.forward(a,
  b) == a` for arbitrary tensors.
- `test_copy_layer_reverse` — `CopyLayer.reverse(parent) ==
  (parent, parent)`; matches the lossy `(parent, parent)`
  pseudo-inverse pattern shared with `SwapLayer`.
- `test_copy_layer_grammar_dispatch` — `'copy'` is in
  `GRAMMAR_LAYER_CLASSES`; instantiation via
  `GRAMMAR_LAYER_CLASSES['copy']()` produces a parameter-free
  `CopyLayer` instance.
- `test_swap_and_copy_dual` — `SwapLayer.forward(a, b) == b`
  and `CopyLayer.forward(a, b) == a` confirms the dual relation.
- `test_rule_probability_gradient` — backward through a chart
  derivation that fires `swap` and `copy` produces non-zero
  gradient on `Grammar.rule_probability['swap']` and
  `Grammar.rule_probability['copy']`.

**Integration (event tensor).**

- `test_event_layout` — values land in the right indices
  (`AREA_IDX`, `LUMINOSITY_IDX`) after `SymbolicSpace.forward`;
  raw channels reflect the chart's normal output.
- `test_round_trip_raw_channels` — forward → reverse round-trips
  the raw-content channels of `slot_dim_raw` to `atol = 1e-5`.
  Introspective channels are forward-only / lossy by design.

---

## 7. Critical files

**Phase 1b:**

- `bin/Spaces.py`
  - `Space` base class — `nIntrospect`, `AREA_IDX`,
    `LUMINOSITY_IDX` constants.
  - `SymbolicSpace` — codebook scope restricted to
    `slot_dim_raw` so introspective trailing channels survive
    the snap. `forward` body itself unchanged unless the audit
    in Step 6 surfaces a real gap.
- `bin/Layers.py`
  - `_gaussian_kernel_overlap` (prerequisite handoff).
  - `ste_answer` utility.
  - **New `AreaLayer(GrammarLayer)`** ('C'-tier, unary,
    parameter-free) — registered in `GRAMMAR_LAYER_CLASSES['area']`.
  - **New `LuminosityLayer(GrammarLayer)`** ('C'-tier, unary,
    parameter-free) — registered as `'luminosity'`.
  - **New `CopyLayer(GrammarLayer)`** ('S'-tier, binary,
    parameter-free) — adjacent to existing `SwapLayer` (~line
    3103); registered as `'copy'` (~line 3239).
- `bin/Models.py` — config validator (~lines 2540 and ~4980 in
  `BasicModel` / `MentalModel`); the slot-embedding extension
  flows through existing shape plumbing.
- `data/grammar.cfg` and `data/grammar_shamatha.cfg` — add
  `C -> area(C)`, `C -> luminosity(C)`, and `S -> copy(S, S)`
  productions.
- `test/test_symbolic_space.py` (or new
  `test/test_phased_dispatch.py`) — extended with Phase A / C
  dispatch, codebook passthrough, area / luminosity / copy unit
  tests.

**Phase 1c (sketched, not in this checkin):**

- `bin/Spaces.py` — retire / repurpose the standalone
  `SubsymbolicSpace` class (Phase 1a code) per the §2.0
  simplification; rename `MereologicalTree` → `Mereonomy`; add
  `equals` / `part` write and lookup APIs; `sentenceConfidence()`
  method.
- `bin/Layers.py` — `TruthLayer` analysis pass for
  ultimate-vs-relative truth classification.

**Phase 1d (sketched):**

- `bin/Spaces.py` (`WordSpace`) — `sentenceBuffer`,
  `sentencePredictor`; sentence-type tag emission.
- `bin/Models.py` — wire sentence-prediction loss into the
  optimizer's loss sum.
- A new query-scoring entry point (file TBD per the runtime
  query interface) — `Δ luminosity vs Truth` measurement.

---

## 8. Decision log

Resolved by this combined spec:

1. **`subset` over `directPartOf` (naming) — RESOLVED.**
   Sets are intransitive parts; kernel overlap measures
   containment, not parthood.
2. **`MereologicalTree` → `Mereonomy` — RESOLVED.** Lands in
   Phase 1c.
3. **`SubsymbolicSpace` as a separate Space dissolves —
   RESOLVED.** The "subsymbolic" content is precisely
   `SymbolicSpace`'s pre-codebook bivector activation; the
   "symbolic" content is its post-codebook state. One Space,
   phased dispatch.
4. **Tier-mapped grammar dispatch within `SymbolicSpace` —
   RESOLVED.** Phase A: 'C'-tier (and 'L'-bound-at-C) on the
   bivector activation (the thinking phase). Phase B: codebook
   snap. Phase C: 'S'-tier on the post-codebook content (the
   knowing phase). Existing per-space `SyntacticLayer`
   tier-gates the dispatch.
5. **`area` and `luminosity` are unary 'C'-tier `GrammarLayer`s
   — RESOLVED.** Both fire in Phase A; both write to reserved
   trailing channels (`AREA_IDX`, `LUMINOSITY_IDX`); both
   parameter-free.
6. **`subset` is not a Phase 1b reserved channel — RESOLVED.**
   `subset` is a higher-level operation derivable from
   chart-driven Sigma·Pi compositions across conceptual orders.
   The rename `directPartOf → subset` (§1.1) is the naming for
   the operation at higher levels.
7. **Union and intersection act over all locations — RESOLVED.**
   The chart's standard CKY pair-selection logic dispatches
   `union` / `intersection` (existing 'L'-tier
   `IntersectionLayer` / `UnionLayer`) at any chart binding in
   Phase A — same pattern as 'S'-tier ops use today. **The
   prior "compositional staging on 4 nVector slots" design is
   superseded** by this full chart-driven dispatch.
8. **Geometric `subset` and symbolic stored `part` are distinct
   — RESOLVED.** Continuous staged composition vs. stored
   Mereonomy entry; they coexist.
9. **`luminosity` is one operation with two uses — RESOLVED.**
   Per-slot reserved channel (Phase 1b) and query-quality
   metric (Phase 1d) share the formula.
10. **`SwapLayer` (existing) + new `CopyLayer` — RESOLVED.**
    Both 'S'-tier `GrammarLayer`s, parameter-free; gradient
    signal flows through `rule_probability` + chart
    pair-selection (the existing mechanism). No `nn.Module`
    routing layers added.
11. **Codebook scope — RESOLVED (recommended).** Codebook
    operates on `slot_dim_raw` only; introspective channels
    pass through unchanged via the existing `aux` tail-
    passthrough convention.
12. **No sidecar, no question heads — RESOLVED.** Reserved
    trailing channels for unary ops; chart-driven dispatch for
    compositional ops. All new GrammarLayers parameter-free.
13. **Stack semantics on `ConceptualSpace.nVectors` (7 ± 2) —
    RESOLVED.** Atomic ideas push, relating ops collapse via
    chart-driven binary dispatch. Phase 1b builds for it;
    explicit pointer state is a Phase 1d concern. Today's
    codebook handling on ConceptualSpace stays as today; the
    stack interpretation rides alongside.
14. **Mereonomy hosts both ultimate and relative truths via
    per-symbol DoT — RESOLVED.** A leaf entry is an ultimate
    truth `(symbol, DoT)`; an interior entry is a relative truth
    `part(x, y)` or `equals(x, y)` with per-endpoint DoTs.
    `TruthLayer` analyses incoming truths and routes promotions
    accordingly. Lands in Phase 1c.
15. **`sentenceConfidence()` is a method, formula deferred —
    RESOLVED.** Encapsulation lets the formula be revisited
    without touching call sites.
16. **Phase boundary — RESOLVED.** Phase 1b: tier-mapped grammar
    dispatch in `SymbolicSpace` + new 'C'-tier `AreaLayer` /
    `LuminosityLayer` + new 'S'-tier `CopyLayer`. Phase 1c:
    retire standalone `SubsymbolicSpace` + Mereonomy rename +
    symbolic relations with DoT + `sentenceConfidence()` +
    TruthLayer ultimate-vs-relative analysis. Phase 1d: stack
    pointer state + sentence buffer + sentence prediction +
    query scoring.
17. **σ source — RESOLVED for Phase 1b.**
    `_DEFAULT_SUBSYMBOLIC_SIGMA = 0.1`.

Deferred:

18. **`sentenceConfidence()` formula and threshold default.**
    Method shell lands in Phase 1c; arithmetic and threshold
    calibrated empirically.
19. **DoT merge rule for `equals(x, y)`** — placeholder weighted
    average; revisable.
20. **Transitive DoT propagation along `part` paths** —
    placeholder min-along-path; revisable.
21. **Ring buffer size for `sentenceBuffer`.** Set under
    Phase 1d.
22. **Cross-loop introspection lift** — whether bivector-side
    introspective channels should ever survive across the
    codebook snap to influence `'S'`-tier ops. Default: yes,
    via the codebook-passthrough recommendation (§2.5). Whether
    to expose them as 'S'-tier readable inputs is a Phase 1c
    detail.
23. **Per-slot or per-pair learned σ** — deferred until the
    fixed-σ design is empirically characterised.
24. **Stack-pointer state and push/pop transitions across
    conceptual orders.** Phase 1b uses positional convention
    (slot 0 = top); explicit state is a Phase 1d design.
25. **Codebook role on `ConceptualSpace`** — today's behaviour
    is preserved; whether the codebook should be reinterpreted /
    relaxed to fully expose the stack is a follow-up question.
26. **`CopyLayer.forward` exact arithmetic** — proposed default
    `forward(left, right) → left` as the canonical dual of swap;
    alternatives (mix, marker metadata) open.

---

## 9. Departures from the originals

### From 2026-05-04 (Conceptual Introspection)

- **Renamed** `directPartOf` → `subset` (§1.1).
- **Reframed** `area` and `luminosity` as unary per-slot
  measures (the original spec had `luminosity` and
  `directPartOf` binary).
- **Reframed** introspective ops as `GrammarLayer`s in the
  existing tier system: `area` and `luminosity` are 'C'-tier
  unary; `subset` / `equals` / `part` are higher-level
  operations expressible as chart-driven compositions or stored
  Mereonomy entries.
- **Dropped** `_RULE_METHODS` (the per-`SyntacticLayer`
  table). The new layers register in
  `GRAMMAR_LAYER_CLASSES` like every other GrammarLayer.
- **Dropped** sidecar concat-and-feed-forward (replaced by
  reserved trailing channels written by 'C'-tier grammar layers).
- **Dropped** question heads (no learnable parameters added).
- **Preserved** STE wrapper.
- **Preserved** the motivation (introspective ops trainable via
  prediction loss).

### From 2026-05-05 (Subsymbolic Knowing)

- **Superseded** the standalone `SubsymbolicSpace` Space:
  `SubsymbolicSpace` is no longer a separate Space. Its
  responsibilities (the bivector "thinking" content) are
  recognised as `SymbolicSpace`'s pre-codebook state. The
  Phase 1a code retires in Phase 1c.
- **Superseded** the dual re-entrant loop with summation: a
  single `SymbolicSpace` event tensor carries both thinking and
  knowing aspects through phased dispatch; their summation is
  automatic because they are aspects of the same event.
- **Superseded** the `<architecture><mode>grammar|parallel</mode>`
  toggle: phased dispatch makes 'C'-tier ("parallel"-like) and
  'S'-tier ("grammar") complementary, not exclusive. Mode gating
  collapses into "are 'C'-tier rules registered" (Phase 1c
  detail).
- **Preserved** "no parameters beyond Sigma·Pi" claim — all new
  grammar layers (`AreaLayer`, `LuminosityLayer`, `CopyLayer`)
  are parameter-free.
- **Preserved** sentence boundary, combined-input shape (now
  trivially: `perceptual_event || symbolic_event`), one-order
  delay.

### Net new (this combined spec)

- **`SubsymbolicSpace` dissolves** (§2.0): `SubsymbolicSpace`
  is recognised as `SymbolicSpace`'s pre-codebook bivector
  state. One Space, phased dispatch.
- **Tier-mapped phased dispatch within `SymbolicSpace`** (§2.2):
  Phase A 'C'-tier on bivectors (thinking) → codebook snap →
  Phase C 'S'-tier on codebook content (knowing). Existing
  per-space `SyntacticLayer` tier-gates the dispatch.
- **`.thinking` and `.knowing` framing** for the two phases of
  `SymbolicSpace.event`; their summation is automatic via the
  shared event tensor.
- **`AreaLayer` and `LuminosityLayer`** (§2.3) as new 'C'-tier
  unary `GrammarLayer`s, parameter-free, writing to reserved
  trailing channels of the slot embedding.
- **`CopyLayer`** (§2.4) as a new 'S'-tier `GrammarLayer`
  alongside the existing `SwapLayer`; both parameter-free with
  training via the existing `rule_probability` mechanism.
- **Union / intersection over all locations** via chart
  dispatch (existing `IntersectionLayer` / `UnionLayer`
  bind at C in Phase A) — supersedes the prior "compositional
  staging on 4 slots" design.
- **Codebook scope** restricted to `slot_dim_raw` (§2.5);
  introspective channels pass through unchanged.
- Composite-sentence framing with **stack semantics on
  `ConceptualSpace.nVectors` (7 ± 2)** (§3.1).
- `Mereonomy` rename and symbolic `equals` / `part` (§1.2, §3.3,
  Phase 1c).
- **Mereonomy hosts both ultimate and relative truths via
  per-symbol DoT** (§3.3); TruthLayer analyses incoming
  linguistic truths and routes promotions accordingly.
- `sentenceConfidence()` as a method (§3.4); formula deferred.
- Sentence buffer and sentence-to-sentence prediction (§3.6,
  Phase 1d).
- Query scoring via `Δ luminosity` vs Truth (§3.5, Phase 1d).

---

## 10. Verification (Phase 1b)

```
# Unit
basicmodel/.venv/bin/python -m pytest test/test_phased_dispatch.py -v

# Integration: phased dispatch populates introspective channels via Phase A,
# preserves them through codebook snap, and runs Phase C on the snapped content.
basicmodel/.venv/bin/python -c "
from Models import BasicModel
m = BasicModel(...)            # nIntrospect = 2, area/luminosity/copy registered in grammar
m.forward(small_fixture_batch)
ev = m.symbolicSpace.subspace.event

# Unary introspective channels populated by Phase A
assert (ev[..., -2:] != 0).any()
assert (0.0 <= ev[..., -2]).all() and (ev[..., -2] <= 1.0).all()      # area
assert (-1.0 <= ev[..., -1]).all() and (ev[..., -1] <= 1.0).all()     # luminosity
"

# Regression: with no 'C'-tier rules registered, behaviour matches today
make test-baseline
```

---

## 11. Relationship to existing plans

- **Conceptual Introspection handoff (2026-05-04)** — superseded
  in operation naming (`subset`), location (`GrammarLayer`s
  registered in the existing tier-dispatched
  `GRAMMAR_LAYER_CLASSES`, not new `_RULE_METHODS` entries),
  feed-forward mechanism (reserved trailing channels, not sidecar
  concat), parameter budget (none, not five `nn.Linear` heads).
  Motivation and arithmetic preserved.
- **Subsymbolic Knowing handoff (2026-05-05)** — **Phase 1a is
  superseded by §2.0**: `SubsymbolicSpace` as a separate Space
  is dissolved in favour of phased dispatch within
  `SymbolicSpace`. The Phase 1a `SubsymbolicSpace` class retires
  in Phase 1c. Sentence boundary and combined-input shape carry
  forward; the dual re-entrant loop simplifies to one Space's
  phased event.
- **Resolve / Luminosity handoff (2026-05-04)** — Phase 1b
  prerequisite. `_gaussian_kernel_overlap` and the `area = σ²`
  convention are reused by the new `AreaLayer` /
  `LuminosityLayer`.
- **Memory architecture (`project_memory_architecture`)** —
  extended in Phase 1d. STM remains `concept_states`; LTM remains
  `TruthLayer`; **Mereonomy joins LTM as a separate stored
  relation graph**; **`sentenceBuffer` is a new mid-tier sentence
  memory**.
