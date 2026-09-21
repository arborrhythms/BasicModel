# Thought-operator contracts

Status: one normal controller and subsystem-scoped thought effects. The
loader/registry catalogue is grammar-owned. Learned questioning utility and
compound generativity remain unproven. See [AccessibleMind](AccessibleMind.md).

Thought operators are the checked, boundary-executed faces of the one sentence
grammar. They were formerly called checked queries. This document describes the
production interface; the historical reasoner classes remain quarantined until
their removal receives review.

## One source of operator identity

`Grammar.configure` rejects the legacy capitalized `<Queries>` spelling at
every grammar nesting level and every rule-level `query` attribute. It parses
the model's lower-case `<thought>` declarations and builds the immutable,
declaration-ordered `Grammar.thought_operations` tuple from **only those
selected forms**, joining any role-labelled `<compose>`/`<generate>` faces
that share their identity.
Each `ThoughtOperationSpec` supplies canonical semantic ID, `I1 … In` operand
roles, `O1` result role, selected structural forms, role permutation, and
any matching forward/reverse rule IDs (empty for a thought-only face). The
executor table supplies only
non-grammatical capability facts: domain, argument kinds, subsystem write
target, enumerated read/write scopes, evidence kind, and the callable.
`QuerySignature` variants inherit those same `Subsystem` members from the
canonical descriptor and validate the thought permission row. Free-form scope
strings are rejected; aliases cannot introduce a second access vocabulary.

A structural family omitted from `<thought>` stays pure grammar even if an
executor exists. An executor with no selected thought form is unavailable. A
selected form without a checked executor fails registry installation. The
catalogue owns no embedding table, learned parameter, semantic store, or
checkpoint schema.

| Canonical thought operator | Bound input kind | Checked result |
| --- | --- | --- |
| `exist` | complete description | LTM fact evidence |
| `part` | full-width concepts or higher-order references | vector residual at order zero; symbolic inclusion or an open-role taxonomy set above it |
| `isPart` | typed conceptual references | taxonomy evidence under that exact model spelling |
| `equal` | full-width concepts | identity evidence |
| `lookup` | full-width concepts | members of already retrieved frames |
| `quantize` | full-width concept | existing conceptual-code result |
| `arma` | complete description | `[3, D]` expectation end state |
| `what` | complete interrogative description | one cued LTM frame in serial context, or a same-controller subgoal |

`part` and `isPart` are exact model-level identities. `part` distinguishes
order-zero geometry from higher-order taxonomy; `isPart` explicitly selects
taxonomy. Neither spelling implicitly enables the other.
A model ordinarily declares one spelling in whichever of compose, thought,
and generate it needs.

`true` is reserved for the deferred two-truths sealed-clause representation;
it has no production executor until that representation exists.

Canonical `part` means `part(I1=part, I2=whole)`. Leaving `I1` open returns
parts of the bound whole; leaving `I2` open returns wholes containing the bound
part. `whole` is a grammar-spelled converse form with permutation `(I2, I1)`,
not a second relation, VP, or executor alias.

## Common grammar-face signature

Every public grammar face receives an immutable owner-selected base context:

```text
GrammarContext(word_stream, conceptual_space, primed_symbols)
```

The only public face signatures are:

```text
compose(operands: tuple[Concept, ...], *, context: StructuralGrammarContext)
    -> Concept
generate(result: Concept, *, context: StructuralGrammarContext)
    -> tuple[Concept, ...]
execute(request: ConceptualMeaning, *, context: ThoughtGrammarContext)
    -> ThoughtResult
```

`StructuralGrammarContext` is exactly the base context plus phase
`compose`/`generate`. Compose receives the current input-owned stream; generate
receives only its output-owned emitted prefix and no priming. It has neither LTM, taxonomy, a
controller, a model/reasoner, nor a writable priming buffer. The dispatcher
adapts old tensor-kernel `compose`/`generate` and unary `forward`/`reverse`
call shapes behind this contract, preserving gradients through full-width
operands, outputs, and registered structural parameters.

`ThoughtGrammarContext` is a distinct boundary context. It includes the same
frozen completed stream, conceptual-space capability, and primed-symbol
snapshot, plus descriptor-scoped read-only LTM/taxonomy views, one shared
`QueryWorkBudget`, a narrow `what` continuation, and the completed-row boundary
permit. An executor never receives a generic model or reasoner. An undeclared
reader is absent from its capability view.

Compose/generate snapshots preserve live structural values without aliases.
Thought context and effects are detached boundary data. Policy learning uses
explicit credit; no answer-state gradient crosses back through a reader.

Typed truth, residual, set, code, prediction and subgoal effects all have owned
answer adapters. A retrieved set remains complete meanings; a residual or code
is a conceptual value; a prediction retains all roles and presence logits.
`reverseOutput` never rereads LTM or repeats a thought to realize an answer.
The given-conclusion cut is documented in [GradientFlow](GradientFlow.md).

## Formation and dispatch

`GrammaticalThoughtRegistry.install` binds one existing frozen native VP concept
per declared `(domain, semantic_id)`. It neither allocates during formation nor
maintains a second VP table. `form` builds canonical `[NP1, VP, NP2]` meanings
with role masks, polarity, bindings, scope, and typed provenance. It is pure:
it does not execute, write memory, or mint a concept.

For an unreduced lexical `[NP1, VP, NP2]` program, the eager forward resolves
the segmented word through the `LanguageSpace` instance's copied grammar
anchor table when its WORD row is admitted or reused. InputSpace retains that
form keyed by the WORD row for the current staging. Capture gathers the
resolved form by the retained WORD row and freezes it alongside the leaf,
without reading text or looking up an anchor again. This distinguishes a
converse such as `whole` from `part` even though
both intentionally share one canonical native VP and checked executor; it
also makes every grammar-declared anchor spelling for that form equivalent.
Recovery validates that form against the model's selected `<thought>`
catalogue and applies its declared role permutation. It stores neither the raw
surface, a row, nor a native address as a semantic feature. A legacy capture
without form provenance may recover a single-form family, but declines an
ambiguous shared-VP family rather than guessing declaration order.
The staging map is cleared with the word rows at the next forward, Start, or
hard reset; a soft sentence reset preserves it. Owned programs and their
detached recall copies retain the captured form independently. Unresolved
rows never acquire provenance from a matching spelling alone.

When the controller selects a later catalogue action, its new VP and legal role
assignment come from that action's grammar contract; mode, polarity, bindings,
and scope are copied from the selected source meaning. A refinement therefore
cannot silently turn `not`/`non` into a positive question or discard its
semantic context.

Setup preflights all missing VPs as one group against both the allocator and
the order-zero snap capacity. If a deliberately small model cannot reserve the
whole group, it retains its structural grammar and only already checkpointed
VPs remain executable; the others are explicitly unavailable. It never mints a
declaration-order prefix and never allocates lazily at a boundary.

`registry.execute(request, context)` is the production thought call. It admits
the completed row once before any VP, occurrence, or native-payload read; then
validates the grammar-owned VP, role occupancy, full conceptual width, and
descriptor contract. Direct `ThoughtSignature.invoke` makes the same admission
for isolated checked calls. Invalid requests do not reach a reader. A negated
truth result swaps its known support channels; absent evidence remains unknown.

A description-valued operand names an existing durable `ltm` occurrence or an
ordinary live `thought` occurrence. Its illumination summary is not a substitute
for the full role-labelled record. Resolution preserves the complete meaning,
including VP, NP2, scope, bindings, and provenance; it never substitutes NP1,
rebinding, a row number, or a surface token.

At owned compose-program recovery, one direct signed leaf can form only a
descriptor-declared `concept` operand. A direct leaf or its arbitrary concept
ID cannot manufacture an `ltm`/`thought` occurrence for a `description` or
reference operand; recovery declines until that owner exists.

## Phase, work, and lifecycle

Structural composition, reconstruction, and generation cannot execute a thought
operator. Only the completed-row phase gate may call the registry. The normal
controller forms candidates without reader calls, selects one catalog action or
`conclude`, and records the actual checked `ThoughtResult` as a detached typed
snapshot in the existing thought-history owner. Only executed `thought`,
causal `return`, and final `finish` transitions can carry that snapshot;
summary support remains controller evidence, not a substitute for a typed
result. The v3 history sidecar tags a nested `ConceptualMeaning` result value
so replay preserves its type, metadata, and hard boundary; v1/v2 sidecars
remain readable. Nested `what` reuses the same episode, level discipline,
continuation, and meter.

The one `QueryWorkBudget` covers VP/reference/payload preparation, description
resolution, fact and taxonomy reads, codebook scans, prediction context, and
nested callbacks. Local limits only narrow that meter. Exhaustion reports
`work_budget` incompleteness; it does not turn unknown into false or permit an
uncharged retry. See [shared work accounting](QueryWork.md) and [thought
history](ThoughtHistory.md).

## Compatibility quarantine

`QueryContext`, `QuerySignature`, and `GrammaticalQueryRegistry` remain isolated
historical reader contracts pending review. The frame kernel and `NeuralToolUser`
class are deleted; public reasoning APIs enter the one normal controller. A production `BasicModel` installs only
`GrammaticalThoughtRegistry`, and its boundary lookup cannot find the legacy
registry. Compatibility aliases cannot define grammar identity, become policy
features, or grant a thought executor authority.

`resolveAnswer()` also never calls the legacy `answer_query()` from a
`WhatQuestion.prompt`, even when `reasoningIterations` is positive. An
unselected program retains its owned identity/temporal seed rather than letting
raw surface text select a hidden legacy tool or append a second reasoner trace.
`answer_query()` is an explicit serving API over the same completed meaning
and normal controller. The older controller has been removed.

## Validation

Catalogue, VP, program-recovery and normal-controller probes cover the
contexts, subsystem permissions, canonical/converse/open-role forms, boundary
admission, signed leaves, detached effects and shared work. Index, retrieval,
checkpoint and generativity measurements are in
[AccessibleMind](AccessibleMind.md) and [Testing](Testing.md).
