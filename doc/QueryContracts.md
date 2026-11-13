# Thought-operator contracts

Thought operators are the checked, boundary-executed faces of the one sentence
grammar (formerly called checked queries). One normal controller runs them, and
their catalogue is owned by the grammar. This document describes their
interface; [AccessibleMind](AccessibleMind.md) describes their effects on the
subsystems.

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
| `query` | complete pattern with open references | one best matching LTM row, its pair and occurrence |
| `ask` | complete row | a filling, or a nested question on the same meter |
| `isTrue` / `exist` | ended proposition / conceptual content | the row's pair and trust / conceptual presence and its pair |
| `isPart` / `part` | references / full-width codes | symbolic parthood witnesses / containment content and a pair |
| `isEqual` / `equal` | references / full-width codes | LTM1 equality evidence (membership in a singleton) / code identity content and a pair |
| `infer` / `implies` | two addressed chunks, a premise and a recalled rule / full-width codes | the instantiated consequent with the path-minimum pair and its ordered witnesses, or a subgoal question chunk at `(0, 0)` / region containment and a pair |
| `bind` | a selected reference site (meaning) | the site filled, with its pair, evidence `reference` |
| `intuit` | meaning (the question's content) | one order-zero Sigma-over-Pi field in the episode's addressed STM chunk, with the fired rules' reference plans as `infer` candidates and knowing deposits; evidence kind `intuition`, provisional until `infer` selects a firing |
| `not` | pair, meaning, or conceptual content | exchanged poles with content retained |
| `gain` | conceptual magnitude | content and pair; sets the next sentence's expectation gain |

Every operation leaves a pair or content, never a scalar. `whole` remains the
converse spelling of conceptual `part`, with permutation `(I2, I1)`.
`isPart` is the symbolic face at every order. Results become serial slots with
their pair, witnesses and producing operation; subsequent requests bind those
slots without an implicit LTM search.

Retired operator names raise with their replacements (2026-10-07): thought
`what` with `ask`; LTM `what` and `lookup` with `query`; grammar `chunk` with
`synthesize` (inverse `analyze`); `true` with `isTrue`. `quantize`, `arma` and
`expect` are removed from thought: `<sentenceExpectation>` and `gain` govern
the global predictor.

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

The rule fixes the operand tuple's arity and its canonical order `I1 ... In`.
A binary composition takes the older constituent as `I1` and the newer as
`I2`, the order of STM and of the derivation; a converse spelling such as
`whole` changes only the declared permutation, never this order.

`StructuralGrammarContext` is exactly the base context plus phase
`compose`/`generate`. Compose receives the current input-owned stream; generate
receives only its output-owned emitted prefix and no priming. It has neither
LTM, taxonomy, a controller, a model/reasoner, nor a writable priming buffer.
The dispatcher adapts old tensor-kernel `compose`/`generate` and unary
`forward`/`reverse` call shapes behind this contract, preserving gradients
through full-width operands, outputs, and registered structural parameters.

`ThoughtGrammarContext` is a distinct boundary context. It includes the same
frozen completed stream, conceptual-space capability, and primed-symbol
snapshot, plus descriptor-scoped read-only LTM/taxonomy views, one shared
`QueryWorkBudget`, a narrow `ask` continuation, and the completed-row boundary
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
assignment come from that action's grammar contract; polarity, bindings and
scope are copied from the selected source meaning. A refinement therefore
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

Operand admission distinguishes a reference from a vector. A reference operand
is admitted when its handle is live: placed by the allocator and not retired
(`Queries._existing_symbol`); a structural symbol such as an unseated Part
handle qualifies. Only a consumer that reads the operand's vector also requires
a payload row (`_existing_row`). Invalid or retired handles are refused loudly;
admission never allocates a seat or invents content. A query never cites the
row or relation written by its own close as an independent witness of its
answer: both the relation and the content branch exclude the query's own
occurrence (`_query_occurrence`).

A description-valued operand names an existing durable `ltm` occurrence, an
ordinary live `thought` occurrence, or a complete already-held constituent.
Holding that constituent does not write a speculative LTM row. Its illumination
summary is not a substitute for the full role-labelled record. Resolution
preserves the complete meaning, including VP, NP2, scope, bindings, and
provenance; it never substitutes NP1, rebinding, a row number, or a surface
token.

At owned compose-program recovery, one direct signed leaf can form only a
descriptor-declared `concept` operand. A direct leaf or its arbitrary concept
ID cannot manufacture an `ltm`/`thought` occurrence for a `description` or
reference operand; recovery declines until that owner exists.

## Phase, work, and lifecycle

Every completed field is offered one root episode at its close
(`ThoughtClosing.close`). For a field with no open slot, the chooser's first
menu holds the catalogue candidates and `None`; choosing `None` is not
thinking: it spends no work, writes no `begin` or `conclude` record, and returns
the field's bound result unchanged. Choosing an operation opens the episode, and
its `begin` record carries `opened_by_chooser`. A field with an open referent or
relation has no `None` in its first menu while work remains, so it is always
thought about. Whether to think about a closed sentence is thereby learned from
the parse's form, like every other choice. Grammatical mode is neither stored
nor read: an asked clause is unasserted because its kept head applied `ask`
([ExistenceEvidence](ExistenceEvidence.md#meaning-ownership-and-storage)).
Where the data supervises output, a sentence with no supplied answer is taught
silence (`WhatAnswer.silence`). `(0,0)` is ignorance, never an extra operand
slot; question marks and wh-words reach the chooser only through the parse. At
the committed host boundary, completed rows alone receive permission; compose,
reconstruction and generate cannot call readers.

One `attentionBudget` pays for selection, preparation, reads, operations and
nested questions. `selectedThoughtBudget` and `thinkingBudget` fail at load.
`conclude` is unavailable while a referent or relation is open and work
remains, unless `thoughtSearchExhaustion` complete queries have found no
candidate, or every operation on the menu is the self-inverse `not`: negating a
held description cannot supply a missing reference, so such a menu also offers
an honest conclusion rather than spending the meter on repeated negation. An
empty menu finishes the episode with the reference open and no conclusion
(`no_legal_operation`). Search exhaustion provisionally mints source-supported
forward referents; questions remain open. Their row-and-role column addresses
are ordinary bind candidates, and a later copula can fill the addressed
variable. The evidence pair never gates the answer cost of a filled referent.

A retrieved row answers only through a constrained role: a query whose every
role is open returns no witness, so no held or primed row can answer a bare free
name; choosing an antecedent is `bind`'s work. A sentence's enclosing
occurrences are excluded from its own nested query. A relation whose operand is
itself still open marks that operand unresolved (`_unresolved_operands`); it can
never serve as a completed answer, although an equality's other endpoint still
can. Re-reading an equality backwards so that a free region would be answered by
its own fixed side is refused as no progress (`ThoughtBinding.integrate`); an
explicitly reflexive witness remains valid.

Only the kept walk publishes conclusions: occurrence-addressed LTM rows of
kind `inference` (`unverified` for a `(0, 0)` pair, `question` for open
slots), with both poles and provenance references. The document and
turn, ordinal within that turn, and content determine their addresses. The
closing retains unresolved questions; thought operations have no generic LTM
append capability. `descend` and `return` remain chronological trace records,
not a separate planner stack.

The grammar's operation scorer, the one compose uses, makes every thought
choice. Thinking is part of each whole path
([Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11)):
a departure at a thought decision is credited by the proposal-corrected score
function $p(a)\,\Delta C$ on its complete path's change in reconstruction plus
supplied-answer cost (R + A), and exact ties contribute no policy term. Work
bounds an episode but is not a cost in the comparison. Expectation trains only
its own predictor. The presented answer is the binding; there is no separate
comparison reader.

Before a field with no open slot is offered its episode, the close may conceive
one absence: an expectation image whose omission of a predicted symbol exceeds
`expectationOmissionThreshold` stores `not X`, an absence inference with that
confidence against (`ThoughtClosing.absence`).
See [thought history](ThoughtHistory.md) and [accessible mind](AccessibleMind.md).

## Compatibility quarantine

`QueryContext`, `QuerySignature` and its `BUILTIN_QUERIES` table are isolated
standalone reader contracts. A production `BasicModel` installs only
`GrammaticalThoughtRegistry`, and public reasoning APIs enter the one normal
controller. Compatibility aliases cannot define grammar identity, become policy
features, or grant a thought executor authority.

`resolveAnswer()` never calls `answer_query()` from a `WhatQuestion.prompt`,
even when `reasoningIterations` is positive. An unselected program retains its
owned identity/temporal seed rather than letting raw surface text select a
hidden tool or append a second reasoner trace. `answer_query()` is an explicit
serving API over the same completed meaning and normal controller.
