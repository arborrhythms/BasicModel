# Thought operators across compose, thought, and generate

Status: corrected design contract and implemented grammar-loader/registry
foundation (Alec directive 2026-09-19). Normal-controller, learned-policy and
end-to-end generation integration remain pending. This file retains its
original path so the item-0 link remains stable. It replaces the earlier,
incorrect premise that a structural
`<compose>`/`<generate>` declaration by itself authorizes post-composition
thought.

## 1. Principle

One model grammar has three peer sections, in model-file order:

```xml
<compose>...</compose>
<thought>...</thought>
<generate>...</generate>
```

They use one vocabulary of canonical operator identities, but they do not
have one availability list. An operator can have a structural compose face, a
structural generate face, or a checked thought face. Where the same named operator is
declared in more than one section, its arity, canonical operand roles and
result role must agree. It is then one operator used in different phases, not
an alias or a second relation.

`<thought>` is the authoritative, per-model allow-list for operations the
post-composition controller may execute. A structural declaration alone never
grants that permission. Thus a model can compose or generate `not` while
deliberately omitting it from thought; it can understand or say negation
without offering negation as a reasoning action. A thought-only declaration
is also valid: it has the same role-labelled grammar form and its own checked
context. Whenever an operator occurs in more than one section, its identity
and role contract must agree rather than becoming a tool-local alias.

The legacy capitalized `<Queries>` spelling is not another catalogue and stays
a configuration error. `query="false"` remains retired: omission from
`<thought>`, rather than a Boolean attribute on a structural rule, is the
model's explicit thought policy.

## 2. Grammar form

The sections use the same role-labelled `<rule>` form. The directional suffix
identifies the face, not a Python method to call directly:

```xml
<compose>
  <!-- structural faces used while understanding -->
  <rule>part_O1 = part.forward(part_I1, part_I2)</rule>
  <rule>not_O1 = not.forward(not_I1)</rule>
</compose>

<thought>
  <!-- the explicit boundary allow-list; `not` is intentionally absent -->
  <rule>part_O1 = part.thought(part_I1, part_I2)</rule>
  <rule>exist_O1 = exist.thought(exist_I1)</rule>
  <rule>arma_O1 = arma.thought(arma_I1)</rule>
</thought>

<generate>
  <!-- structural faces used while expressing an answer -->
  <rule>part_I1, part_I2 = part.reverse(part_O1)</rule>
  <rule>not_I1 = not.reverse(not_O1)</rule>
</generate>
```

`part.thought(...)` is a declarative role signature. At runtime a thought
face is invoked through the public `execute(request, context)` contract below,
not by passing those XML operands into a tensor layer. A model that calls its
canonical relation `isPart` writes `isPart` consistently in every face it
uses; the loader does not silently turn `part` and `isPart` into an
authoritative alias pair. Existing grammar family/permutation metadata may
still express a converse structural form (`whole` for canonical `part`) and
the thought declaration selects the individual form(s) that are legal at the
boundary.

The loader preserves the lexical order of `<thought>` declarations. That order
is stable metadata for the model/checkpoint but never a learned numeric input.
It does not infer a thought list from the order or contents of the other two
sections.

## 3. Shared identity and independent contexts

Every public grammar face receives the same owned, read-only base context:

```text
GrammarContext(
    word_stream,          # owner-selected ordered words/leaves
    conceptual_space,     # full-width conceptual values and operations
    primed_symbols,       # captured recently-primed-symbol snapshot
)
```

The stream is row-owned: compose sees the current input stream, generate sees
only its output-owned emitted prefix, and thought sees the frozen completed
program/meaning. `conceptual_space` and `primed_symbols` are narrow
capability views, never a model/reasoner reference. Typed references are
metadata; rows, IDs, addresses and surfaces are never learned features.

The public calls are intentionally similar in shape while preserving their
different phase authority:

| grammar | public call | result | additional authority |
|---|---|---|---|
| compose | `compose(operands: tuple[Concept, ...], *, context: StructuralGrammarContext)` | one full-width `Concept` | none |
| thought | `execute(request: ConceptualMeaning, *, context: ThoughtGrammarContext)` | one typed `ThoughtResult` | descriptor-scoped LTM/taxonomy reads, shared work meter, boundary permit, narrow continuation |
| generate | `generate(result: Concept, *, context: StructuralGrammarContext)` | declared full-width operand tuple | none |

`StructuralGrammarContext` has phase `compose` or `generate`. It cannot
execute a thought operation, allocate a concept, write memory, recover live
priming, access LTM/taxonomy, or schedule a subgoal. `ThoughtGrammarContext`
extends the same immutable base only at an open completed-row boundary with
read-only, descriptor-scoped LTM and taxonomy capabilities; the one
non-renewable `QueryWorkBudget`; a narrow `what` continuation; and the phase
permit. The controller remains the sole result recorder and permitted writer.

Context objects, LTM/taxonomy reads, priming snapshots, work accounting and
typed thought results are detached ownership data. Compose/generate retain
their configured gradients through full-width tensor operands, outputs and
registered operator parameters. Thought-policy and residual learning receive
explicit credit; they do not acquire a hidden gradient through a reader.

## 4. Catalogue and executor contract

After grammar normalization, the loader builds immutable
`Grammar.thought_operations` from **only** the declarations in `<thought>`,
joining any same-named structural faces. Each `ThoughtOperationSpec` records
canonical semantic identity, operand/result roles, selected thought forms,
and any matching structural rule IDs. A thought-only form has empty
structural ID tuples. Structural families not selected by `<thought>` remain
ordinary grammar rules and do not enter this tuple.

`bin/Queries.py` supplies the non-grammatical executor descriptor keyed by
that canonical identity: domain, argument kinds, result kind, read/write
scope, evidence kind and callable. It never supplies membership, arity, roles,
permutation or aliases. Configuration fails closed when:

- matching faces disagree on canonical identity, roles or result role;
- a thought declaration has malformed roles or family/permutation metadata;
- a selected form lacks a checked executor or its executor's argument kinds
  disagree with the declared arity;
- a family/permutation declaration is malformed; or
- the old `<Queries>` spelling or a retired `query` attribute appears.

A structural family with no executor is valid grammar. If a model lists it in
`<thought>`, registry installation fails rather than exposing a non-callable
action. An executor table entry absent from `<thought>` is unavailable, even
if the same operator occurs in compose/generate. This is how each model sets
its legal thought action set without changing a global tool menu.

The controller sees the ordered selected thought forms with executors plus
`conclude`. It may choose a valid operation that did not occur in the current
parse, but it may never manufacture a form absent from that model's
`<thought>` list. `arma` remains a typed prediction result carrying the
owned `[3, D]` expectation end state; it is not a fabricated root or a change
to the 21-value compiled tuple. `true` remains absent until the deferred
two-truths sealed-clause representation exists.

## 5. Required implementation order

1. Add red loader/registry probes proving that structural membership does not
   imply thought permission; `<thought>` selects a subset; a common operator
   (for example `part`/`isPart`) has one role contract in compose and thought;
   and mismatches fail before runtime readers are reached.
2. Parse `<thought>` in inline and external grammar sources, preserving its
   declared order and validating `.thought` role signatures. Keep `<Queries>`
   rejected rather than providing a parallel compatibility path.
3. Rebuild `Grammar.thought_operations` from thought declarations, joining
   any matching compose/generate faces. Track selected forms rather than
   exposing every structural converse automatically.
4. Change `GrammaticalThoughtRegistry`, candidate formation and the normal
   controller to consume only that explicit catalogue. Preserve the existing
   hard phase gate, shared work accounting, signed operands, native VP
   identity, checkpoint isolation and typed results.
5. Migrate `complete.grammar`, `default.grammar`, `ladder.grammar`, inline
   model grammar and checked fixtures together. Place `<thought>` physically
   between `<compose>` and `<generate>` and declare only the operations each
   model is meant to execute.
6. Update the documentation named below, capture focused affected receipts,
   then run one source-matched full default suite before the BasicModel and
   WikiOracle commits.

## 6. Evidence and documentation

The reviewer probes must cover explicit allow-list behavior, role/family
agreement, per-model omission, compose/generate/thought capability separation,
no parser-time executor effects, signed/arbitrary-symbol preservation,
MPS-safe normal-controller integration, work accounting, checkpoint replay,
and the typed `arma` handoff. A full passing receipt is required after source
changes; focused mechanism receipts alone do not complete item 0.

Update `QueryContracts.md`, `Language.md`, `Params.md`, relevant grammar
commentary, `GradientFlow.md`, `ThoughtHistory.md`, `Testing.md`, and
`todo.md`. Protected historical documents remain untouched without separate
review.
