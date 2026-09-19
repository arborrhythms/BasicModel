# Thought-operator unification plan

Status: corrected 2026-09-19; the grammar-loader/registry catalogue foundation
is implemented, while the normal-controller and end-to-end item-0 work remain
pending. The September 18 compose-derived catalogue is archival below, not an
implemented or desired result. The authoritative contract is
[thought operators across compose, thought, and generate](../specs/2026-09-18-thought-operations-in-compose.md).

## 2026-09-19 authoritative correction

The model grammar has three peer sections in this physical order:

```xml
<compose>...</compose>
<thought>...</thought>
<generate>...</generate>
```

They share canonical operator identities and exact role contracts, but not an
implicit availability list. Compose and generate own structural use.
`<thought>` is the explicit per-model allow-list for post-composition
execution. Thus one parthood operator can occur in both compose and thought
with one identity but different legal signatures and contexts; an operator
such as `not` can occur structurally while being intentionally unavailable to
thought. A structural declaration never silently grants thought permission.

The thought declaration uses the same role-labelled rule form:

```xml
<compose>
  <rule>part_O1 = part.forward(part_I1, part_I2)</rule>
</compose>
<thought>
  <rule>part_O1 = part.thought(part_I1, part_I2)</rule>
</thought>
<generate>
  <rule>part_I1, part_I2 = part.reverse(part_O1)</rule>
</generate>
```

`.thought(...)` declares the role schema; runtime dispatch remains
`execute(request: ConceptualMeaning, *, context: ThoughtGrammarContext)`.
Compose/generate receive the owned word stream, conceptual-space capability
and primed-symbol snapshot through `StructuralGrammarContext`. Thought gets
the same frozen base context plus descriptor-scoped LTM/taxonomy readers, the
shared work meter, boundary permit and narrow continuation. The structural
faces remain pure; all readers/results and work accounting remain detached
from structural gradients.

A thought declaration may stand alone. Whenever the same operator also has
a compose/generate face, they must agree on semantic identity, ordered I
roles and O1. Existing family/permutation metadata can describe a converse
form, but only an explicitly declared thought form is controller-legal. A
Python executor descriptor never creates membership. The capitalized legacy
`<Queries>` spelling and `query`
attributes remain rejected; they must not become a second catalogue.

### Corrected implementation order

1. Add red loader/registry probes: structural membership alone is not enough;
   `<thought>` selects a subset; omission yields no action; matching/mismatched
   role contracts succeed/fail before readers; inline/external grammar agree;
   and a converse is legal only when explicitly listed.
2. Parse `.thought` rules separately from structural runtime rules, preserving
   declaration order and structural rule IDs.
3. Build `Grammar.thought_operations` by joining only the selected thought
   forms to structural families. Do not derive it from every compose/generate
   rule or expose unselected forms.
4. Rebase registry installation, signature lookup and controller candidates on
   that explicit catalogue while preserving signed operands, opaque concepts,
   native VP identity, phase gating, work accounting and typed results.
5. Add `<thought>` to production/external/inline grammar sources between
   compose and generate, selecting only operators each model authorizes.
6. Complete the normal controller and typed `arma` handoff only after this
   catalog is green; do not begin two-truths or forgetting first.
7. Update the named architecture/gradient docs, run affected bounded tests,
   then one frozen-snapshot full default suite before both repository pushes.

### Corrected completion criteria

Item 0 is complete only with a fresh full receipt proving that membership
comes from `<thought>`, matching faces agree, structural-only/table-only
operators stay unavailable, context authority stays separated, selected
forms/open roles/converses are exact, and the existing ownership, gradient,
checkpoint, optimizer and compiled-tuple invariants remain intact.

## Superseded September 18 draft (archival)

The remainder preserves the earlier draft's useful background and testing
ideas, but every statement that derives thought membership from compose or
generate, rejects a distinct thought section, or calls item 0 implemented is
superseded by the correction above.

### Original proposed outcome

One operator family declared in the model's structural grammar is the source
of both legal uses of that operator:

1. Its structural forward and/or reverse face may run while the sentence
   grammar is composing, reconstructing, or realizing a generated sentence.
2. If the same declared operator has a checked executor, the completed
   sentence's boundary controller may choose it **after** composition.

The second use is not inferred from a rule that happened to fire in the input
parse. At model configuration time, every executable operator declared in
`<compose>`, `<generate>`, or both enters the boundary action catalog. Matching
forward/reverse declarations merge into one spec and must agree on semantic id,
arity and canonical role order. A completed idea, its retained occurrences and
the attended context supply the operands when the controller later selects one
of those actions. Thus a configured `part`, `exist`, `lookup`, `quantize`,
`arma`, or `what` operation is available for thought even if that operator was
not used by the immediately preceding sentence.

Inline `<SymbolSpace><language><grammar>…<compose>…` configurations and an
external `*.grammar` named by the same model-file element produce the same
catalog. There is no separate thought grammar and no effectful parser path.

## Uniform grammar-face signature

The unified catalog does not mean every operator receives the whole model.
Every face receives the same immutable, owner-selected base context:

```text
GrammarContext(word_stream, conceptual_space, primed_symbols)
```

`word_stream` is row-owned: compose sees the current input stream, generate
sees only its output-owned emitted prefix, and thought sees the held completed
program. `conceptual_space` is a full-width read/operation capability rather
than an allocator or model reference. `primed_symbols` is a captured
row-local view and grants no route to taxonomy. References may be typed
metadata, but no row, ID, address, or surface may enter a learned feature.

The only public grammar-face calls are:

```text
compose(operands: tuple[Concept, ...], *, context: StructuralGrammarContext)
    -> Concept
generate(result: Concept, *, context: StructuralGrammarContext)
    -> tuple[Concept, ...]
execute(request: ConceptualMeaning, *, context: ThoughtGrammarContext)
    -> ThoughtResult
```

The rule defines the tuple arity and canonical `I1 … In` order; ordinary
binary composition remains older-left/newer-right. `StructuralGrammarContext`
contains precisely the base context and a structural phase. It never contains
LTM, taxonomy, a generic model/reasoner, a controller, or a writable priming
buffer, so compose/generate stay pure and cannot execute a thought operation.
`Grammar.thought_operations` is an immutable, first-declaration-ordered tuple
of `ThoughtOperationSpec` values; a spec exposes its semantic ID, canonical
operand/result roles, and forward/reverse structural rule IDs. It is the sole
configuration-level catalog from which lookup maps are derived.

`ThoughtGrammarContext` is a distinct context type, not a structural context
with optional readers. It extends the base context only under the checked
completed-row boundary with read-only, descriptor-scoped `ltm` and `taxonomy`
views, the single `QueryWorkBudget`, and a narrow subgoal continuation for
`what`, plus the one-row phase-gate `boundary` permit. The descriptor's
`read_scope` fails closed for unavailable methods;
the controller remains the only writer and result recorder. Structural
full-width tensors and registered structural parameters retain their configured
gradient paths. Memory/taxonomy reads, priming snapshots, work accounting and
typed thought results are detached hard-boundary data; thought-policy and
residual learning receive explicit credit instead.

## Non-negotiable boundaries

- A structural `forward` or `reverse` never invokes an executor. The existing
  sentence/query phase mask remains the sole permission to call a hard face.
- The compose/generate declaration family, not an alias list or an integer
  concept ID, is the source of semantic identity, arity, canonical role order,
  open-role forms, converse permutation and structural operator identity.
- The executor declaration supplies only non-grammatical information: domain,
  argument kinds, result kind, read/write scope, evidence kind and callable.
  A table entry without a matching `<thought>` declaration is unavailable.
- Concepts remain opaque and symbol/concept rows remain aligned. A boundary
  candidate carries full-width meaning values and typed references; it never
  uses a row, concept ID, native handle or surface token as a learned feature.
- The action trace remains structural metadata. It must preserve the actual
  signed leaves and older-left/newer-right orientation; it cannot be repurposed
  as a second semantic or query store.
- `arma` is a typed boundary result, not a fake concept root: its executable
  result is the owned `[3, D]` expectation end state and is handed explicitly
  to answer resolution. It does not change the 21-value compiled tuple.

## Semantic decisions to lock before coding

### Interrogative mode

Retiring `query="true"` requires a positive representation for question mode.
The proposed canonical form is a unary structural `what` wrapper around an
already formed clause/operation frame:

```text
part(A, B)             # assertive grammatical meaning
what(part(A, B))       # the same meaning in interrogative mode
```

`what` is not a two-NP relation and does not make the inner relation true. Its
structural face preserves the inner canonical roles, bindings, scope and
polarity while setting interrogative mode. Its checked face schedules a
complete internal question through the existing boundary controller. The first
reviewer probe must establish this exact distinction; no code may retain a
hidden interpretation of a removed `query` attribute.

`true` remains a unary operation over a sealed clause reference. Its executor
is typed to that clause and returns evidence, not a new relation row. The
two-truths representation is still deferred; item 0 supplies the catalog and
rejection behavior until a sealed clause is available.

### Families, open roles and converse forms

Each compose/generate declaration family produces a `ThoughtOperationSpec` with a stable
semantic id, operand roles, result role and structural rule key. A grammar-owned
family/permutation declaration makes `whole` the converse surface/operator
form of canonical `part`; it is not an executor alias. The exact XML spelling
for that grammar-owned family declaration is decided by a red loader probe
before changing production grammars. It must be available in both inline and
external grammar syntax, survive grammar copying, and participate in
reconstruction/generation meaning keys.

Open-role variants are derived from the spec and an explicit operand-occupancy
mask at formation time. They do not mint `parts`/`wholes` methods, native VP
rows or learned operation heads. Canonical `part(I1=part, I2=whole)` with
`I1` open returns parts of its bound whole; with `I2` open it returns wholes
containing its bound part. Both use one canonical `part` identity with
different result/role masks.

### Boundary results

`ThoughtSignature.invoke` returns a typed result record containing the source
spec, complete request meaning, support/incompleteness, evidence provenance and
one of: truth, native reference set, code, subgoal, or prediction end state.
Only the prediction variant may carry `[3, D]`; answer resolution consumes it
through an explicit expectation adapter. Other result kinds cannot silently
become facts, grammar roots or answer targets.

## Implementation order

### 1. Establish red catalog probes

Add a focused `test_thought_operation_catalog.py` before changing runtime
code. It covers the twelve cases in the item-0 draft plus these integration
guards:

- inline model grammar and external `.grammar` expose the same specs;
- compose-only, generate-only and paired declarations expose one consistent
  boundary spec, while conflicting paired arities/roles fail at configuration;
- a configured executable operation is available post-composition even if the
  current answer program did not use it;
- `part(A,B)` remains assertive while `what(part(A,B))` is interrogative;
- signed live operands, masks, scope and bindings survive formation; IDs and
  addresses do not become values or chooser features;
- `arma` cannot enter the compiled sentence result as a fabricated root;
- unrecognized old aliases and a `<Queries>` block fail loudly.
- compose/generate are handed only their owner-selected base context, while a
  thought executor receives that same base plus only descriptor-authorized,
  bounded LTM/taxonomy capabilities and the one shared meter.

Capture the red bounded-run receipt before implementation. Do not alter source
while its worker is running.

### 2. Install the uniform face dispatcher and derive one immutable catalog

Extend `RuleDef`/`Grammar.configure` so the model's `<compose>` and
`<generate>` sections yield immutable merged `ThoughtOperationSpec` values
after all inline/external grammar normalization but before grammar runtime
layers are wired. A spec may have a forward face, a reverse face, or both;
paired faces must agree. The catalog must:

- construct an immutable `StructuralGrammarContext` at each compose/generate
  call from the owning stream, conceptual-space capability and primed-symbol
  snapshot, adapting legacy layer internals behind that one dispatcher;
- reject `<Queries>` and `query` attributes;
- retain only operator facts derivable from grammar, including family and
  permutation metadata;
- preserve the existing structural rule ids and local compose snapshots used
  by reconstruction;
- expose a deterministic ordered list for the controller and a stable semantic
  key for output/checkpoint migration; and
- leave a rule with no executor as structural-only.

This is configuration metadata, not a new parameter, memory owner or compiled
output. Assert the existing grammar copies never share mutable catalog state.

### 3. Join checked thought executors to that catalog

Replace checked-query declarations with executor descriptors keyed by canonical
semantic id. `GrammaticalThoughtRegistry.install` receives the immutable operation
catalog and rejects arity/type/family mismatches. It creates one native VP
concept per `(domain, semantic_id)`, not one per alias or open form.

Rebuild `form`, validation and dispatch from `ThoughtOperationSpec` plus
occupancy. The registry must reject a structure-only operation, an executor
missing from the grammar, malformed open forms, a foreign width/reference, or
an attempt to call inside a sentence phase before reading any native payload.
It builds `ThoughtGrammarContext` from the held owner stream, base capability
views, descriptor-scoped LTM/taxonomy views, continuation, boundary permit and the existing
single work meter; it never passes a generic reasoner/model. Keep that meter
flowing through preparation and execution.

### 4. Migrate grammar files and structural faces

Update `complete.grammar`, `default.grammar`, `ladder.grammar`, model-inline
grammar, and intentionally checked fixtures together. A two-faced operator may
be declared in `<compose>`, `<generate>`, or both; paired declarations merge
to one identity and a generate-only declaration is still available to the
post-composition controller. Remove the `<Queries>` blocks and retired
attributes. Verify every old alias has a single canonical structural
replacement and that production grammar comments no longer describe parser
effects.

Pure composition, tied reconstruction and output generation must remain
effect-free. The existing phase-permission/fullgraph suite is a regression gate
here, not an optional later test.

### 5. Rebase the normal controller onto catalog actions

Only after the catalog is green, replace the in-progress fixed
`query`/`finish` controller with candidates of the form:

```text
(ThoughtOperationSpec, canonical operand assignment, open-role mask, conclude)
```

The chooser scores complete root/active/candidate meanings plus bounded
attended context and actual evidence; it never scores raw IDs or arbitrary
catalog positions as semantic values. Candidate formation is side-effect-free.
The selected action alone debits the one shared episode meter, forms the
canonical request, executes under boundary permission, records its actual
result and makes that result available to the next choice. Nested `what` keeps
the same episode, meter, level discipline and causal record sources.

The controller must be able to select an executable configured operation that
did not occur in the previous parse. The completed parse supplies an initial
root/context, not a restricted tool menu.

### 6. Add typed prediction handoff and lifecycle coverage

Wire `arma` results to the existing expectation/answer path through an explicit
result adapter. Prove that changing another row, arriving content or a future
target cannot influence pre-observation action formation. Exercise eval
teardown, training-through-one-optimizer-step teardown, checkpoint restore,
mixed rows, cutoff drain, nested return and recurrent controller choices.

### 7. Retire replaced paths only after their replacements prove coverage

After all callers and tests use the catalog, remove `<Queries>`, declaration
parsing, aliases, addressee/Testimony routing and stale controller dispatch.
Before removing an unused reasoning method such as `NeuralToolUser.run_legacy_world`,
inventory callers and obtain the required user review. Removal is a separate,
reviewed cleanup within item 0; no compatibility fallback may silently preserve
the old catalog.

### 8. Documentation, validation and delivery

Update `QueryContracts.md`, `Language.md`, `Params.md`, grammar comments,
`GradientFlow.md`, `ThoughtHistory.md`, `Testing.md` and `todo.md` to describe
the actual catalog and hard/continuous gradient boundaries. Leave protected
user documents unchanged unless separately reviewed.

For each implementation slice: failing reviewer probe, fix, affected bounded
tests, then one full default suite using the bounded runner while the tested
snapshot is frozen. Commit and push BasicModel with the required trailer, then
commit/push the WikiOracle submodule bump. Rebuild either venv via its Makefile
only after the successful suite and both pushes, as directed.

## Completion criteria

Item 0 is complete only when the draft's twelve tests, the new cross-source
and mode/typed-result probes, the unchanged phase/fullgraph tests, affected
grammar/reconstruction/output tests, and the full bounded default suite are
green. The result must show:

- one compose/generate operator family serving structural and
  post-composition use;
- no parser-time executor effects;
- no independently authoritative aliases/tool grammar;
- exact canonical family/open-role/converse behavior;
- safe `arma` result ownership; and
- no regression of arbitrary-symbol, signed-leaf, checkpoint, optimizer or
  compiled-tuple contracts.

This closes catalog unification only. `true` remains deliberately absent until
the deferred two-truths sealed-clause representation supplies its structural
meaning. Learned controller utility, residual credit, generation-catalog
ownership, expectation gates, two-truths and forgetting remain their separately
ordered work.
