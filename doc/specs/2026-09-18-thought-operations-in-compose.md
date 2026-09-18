# Thought operations are compose rules

Status: implemented catalog contract (Alec directive 2026-09-18). Depends on
the sentence/query phase permission candidate and precedes the remaining
normal thought-controller and learning work (todo item 1). Supersedes the
`<Queries>` block of the grammar. The implementation sequence is recorded in
[the thought-operator unification plan](../plans/2026-09-18-thought-operator-unification.md).

## 1. Principle

One grammar. A thinking operation is exposed by listing its structural face in
the grammar's `<compose>` section, `<generate>` section, or both. Matching
forward/reverse declarations describe one operator identity and must agree on
its role contract. There is no second catalogue: the `<Queries>` block, the
`is`-prefixed aliases and the `query="false"` attribute are retired.

Every declared operator has a **structural face** (a `forward` face builds an
idea during sentence composition; a `reverse` face unbuilds it during
reconstruction or generation) and may have an **executable face** (an executor
the thought controller runs at a completed boundary). The thinking grammar has
no relations of its own: it has the sentence grammar's operators, executed
rather than built. Listing an operator in either structural section makes it
available to thought; an operator with no registered executor is
structure-only. Generating a rule never invokes its executor.

The `exist` rule of `complete.grammar` already has this shape: a structural
no-op forward and an LTM-evidence executor. This spec makes that the rule
rather than the exception.

## 2. Grammar form

```xml
<compose>
  <!-- structure-only operators (no executor registered) -->
  <rule>union_O1 = union.forward(union_I1, union_I2)</rule>
  <rule>verb_O1 = verb.forward(verb_I1, verb_I2)</rule>

  <!-- two-faced operators: built in a sentence, executable at a boundary -->
  <rule>part_O1 = part.forward(part_I1, part_I2)</rule>
  <rule>equal_O1 = equal.forward(equal_I1, equal_I2)</rule>
  <rule>exist_O1 = exist.forward(exist_I1)</rule>
  <!-- `true` is reserved until the deferred two-truths sealed-clause form -->
  <rule>what_O1 = what.forward(what_I1)</rule>
  <rule>lookup_O1 = lookup.forward(lookup_I1, lookup_I2)</rule>
  <rule>quantize_O1 = quantize.forward(quantize_I1)</rule>
  <rule>arma_O1 = arma.forward(arma_I1)</rule>
</compose>
<generate>
  <!-- a matching reverse face, or a generate-only structural declaration -->
</generate>
```

No attribute distinguishes the two kinds of rule. The distinction is derived
at configuration time from the executor table (§5).

## 3. What the rule supplies

From the compose/generate rule family alone the loader derives, per operator:

| derived | from |
|---|---|
| `semantic_id` | the method name (`part`, `equal`, `arma`, …) |
| arity and argument roles | `op_I1 … op_In` |
| result role | `op_O1` |
| open-role variants | any subset of input roles left unbound: canonical `part(I1=part, I2=whole)` with `I1` open returns the parts of its bound whole (today's `parts(X)`); with `I2` open it returns the wholes containing its bound part (today's `wholes(Y)`); a unary operator has one open form |
| the converse interface | the declared argument permutation (`whole` = `part` with `(I1, I2)` swapped) is a grammar-level alias of the same identity, not a second relation |

Mode is not an operator. Interrogative versus assertive belongs to the
completed idea (plan §1). `true` is reserved for the future truth-mode
operator over a sealed clause (`NP → REF(S)`, two-truths spec §3); it is not a
production executor until that representation exists. `what` is the current
open-role interrogative subgoal. Neither is a relation between two NPs.

## 4. Common operation call contract

"Thought operator" is the common name for the checked, boundary-executed
face formerly called a query or thinking operation. It is not a second
grammar. Every public grammar face receives the same owned, read-only base
context:

```text
GrammarContext(
    word_stream,          # the owner-selected ordered word/leaf stream
    conceptual_space,     # full-width conceptual operations and values
    primed_symbols,       # row-local snapshot of recently primed symbols
)
```

`word_stream` is not a loose reference to live staging. Compose receives the
current sentence's owned stream; generate receives the output-owned emitted
prefix, never an input teacher or parse trace; and thought receives the
completed idea's frozen program/stream. `conceptual_space` and
`primed_symbols` are capability views, not the model: the former exposes
full-width conceptual values and grammar support without allocation or
mutation, while the latter exposes a captured priming snapshot without a route
back into taxonomy traversal. Typed references may be carried for lookup, but
IDs, rows, addresses, and surfaces are dispatch metadata and never learned
numeric features.

The rule declaration fixes arity and canonical `I1 … In` order; Python
signatures do not rediscover it from aliases. These are the only public
grammar-face contracts; layer-specific tensor kernels and executor descriptor
adapters stay behind their respective dispatchers:

| face | common call | result |
|---|---|---|
| compose | `compose(operands: tuple[Concept, ...], *, context: StructuralGrammarContext)` | one full-width composed `Concept` |
| generate | `generate(result: Concept, *, context: StructuralGrammarContext)` | the declared `I1 … In` full-width operand tuple |
| thought | `execute(request: ConceptualMeaning, *, context: ThoughtGrammarContext)` | one typed `ThoughtResult` |

`Grammar.thought_operations` is the ordered immutable tuple of the merged
`ThoughtOperationSpec` values, in first structural-declaration order. Each
spec exposes `semantic_id`, canonical `operand_roles`, `result_role`, and its
forward/reverse rule IDs; callers derive a lookup map from that tuple rather
than installing another mutable catalogue.

`StructuralGrammarContext` is exactly the base context and has phase
`compose` or `generate`. Structural faces are pure with respect to LTM,
taxonomy, priming state, occurrence stores, and controller state: they may
read the supplied snapshots but cannot run an executor, allocate a concept,
write memory, or schedule a subgoal. Existing layer-specific
`compose`/`generate` and unary `forward`/`reverse` call shapes may be adapted
behind the dispatcher during migration, but this is the only grammar-face
contract exposed to new code.

`ThoughtGrammarContext` is a distinct context type, not a structural context
with optional readers. It extends the same base context only at an open,
completed-row boundary:

```text
ThoughtGrammarContext(
    word_stream, conceptual_space, primed_symbols,
    ltm, taxonomy, work, continuation, boundary,
)
```

`ltm` and `taxonomy` are read-only, bounded capability views. The executor
descriptor's declared `read_scope` determines which methods each view admits;
an undeclared capability fails closed rather than letting an operator recover
the model or reasoner. `work` is the one shared episode meter and every
preparation/read/traversal charges it. `continuation` is the narrow controller
capability needed by `what`; it is not a general controller or memory writer.
`boundary` is the one-row permission check supplied by the phase gate; it
opens no model capability and fails closed outside a completed answer row.
The controller alone records a selected result and performs any permitted
write.

The context objects themselves are non-differentiable ownership metadata.
Compose/generate retain the configured gradient path through their full-width
tensor operands, outputs, and registered operator parameters. LTM/taxonomy
reads, priming snapshots, work accounting, and typed thought results are
detached hard-boundary data; learned thought utility receives its separate
policy/residual credit rather than a hidden gradient through a reader.

## 5. What the executor table supplies

`bin/Queries.py` keeps one table keyed by canonical operator name
(`THOUGHT_EXECUTORS`). Each entry carries only what the grammar
cannot: `domain`, `argument_kinds` (`reference` / `concept` /
`description`), `result_kind` (`truth` / `set` / `code` / `prediction` /
`subgoal`), `read_scope`, `write_scope`, `evidence_kind`, and the executor.
It carries no arity, no roles and no aliases; those come from the rule.

Configuration joins the two by name. Mismatch is an error, not a fallback:

* an executor whose declared argument-kind count differs from the rule's
  arity fails `install()`;
* a declared structural rule whose method name matches no executor is
  structure-only, silently;
* an executor whose name matches no compose or generate rule is **not exposed** — the
  grammar decides what thought may do.

The production implementation uses `ThoughtSignature` /
`ThoughtGrammarContext` at this boundary; there is no parallel production
query and thought interface. `GrammaticalThoughtRegistry` mints one frozen named concept per
`(domain, semantic_id)` as now; both faces reference it. The `compose_faces`
field disappears because the face *is* the rule.

## 6. Production retirements and compatibility quarantine

The following are removed from the production grammar loader and boundary
dispatcher, not merely deprioritized:

* the `<Queries>` block and `checked_query_declarations`;
* the alias names `isTrue`, `isEqual`, `queryEqual`, `isPart`, `PartOf`,
  `queryPart`, `isWhole`, `parts`, `wholes`, `query`; the operator names are
  `true`, `equal`, `part`, `whole` (permutation alias), `exist`, `lookup`,
  `quantize`, `arma`, `what`;
* the `query="false"` rule attribute and `_parse_bool_attr(query_raw)`;

The historical query/reasoning classes, addressee table, `Testimony`, and
`NeuralToolUser.run_legacy_world` remain isolated compatibility code until
Alec reviews their removal, per the repository preservation rule. No
production model installs `GrammaticalQueryRegistry`, no production boundary
lookup can find it, and none of those compatibility names supplies grammar
identity, policy features, or thought capability authority.

Interrogative mode is a structural `what` wrapper on a completed idea, not a
rule attribute.

## 7. Interaction with the phase gate

Unchanged. Structural faces run inside the sentence mask; executable faces
run only through `ThoughtSignature.invoke` / `registry.execute` under an open
`_query_boundary_scope`. Listing an operator in either structural section never lets its
executor run during composition, reconstruction or generation.

## 8. Interaction with the thought controller (todo item 1)

The controller's action set is the set of declared compose/generate operators
that have an executor, plus "conclude". It chooses a `semantic_id`, an operand assignment
from the frame's bindings and attended context, and which roles stay open.
`arma` is the synthesis face: the idea it emits is the `[3, D]` predicted end
state (not the root), handed to `_resolve_answer → reverseOutput`. Support
and closure remain owned by the hard faces and the runtime budget.

## 9. Tests

1. A grammar with `part` in `<compose>`, `<generate>`, or both and no
   `<Queries>` block installs one registry identity exposing `part` with two
   closed forms and two open forms; inconsistent paired role contracts fail.
2. A `<Queries>` block is a configuration error.
3. A structural rule with no executor (`union`) is structure-only: it is absent
   from the controller's action set and `registry.execute` rejects it.
4. An executor with no structural rule is not exposed even though the table
   holds it.
5. Arity mismatch between rule and executor fails `install()`.
6. `whole` resolves to `part`'s identity with the declared permutation; the
   named concept count is one.
7. Open-role dispatch: `part` with `I1` open returns the set today's
   `parts(X)` returns, byte-identical on the fixture.
8. `what` is a unary operator on an interrogative idea and does not mint a
   relation row. `true` is absent from the production executor catalogue until
   the deferred two-truths sealed-clause representation is implemented.
9. `arma` declared in the structural grammar composes structurally as a no-op and executes
   as the predictor, returning `[3, D]`.
10. The full phase-permission suite passes unchanged against the new
    catalogue (the 23 cases of `test_query_phase_permissions.py` and the
    fullgraph probe).
11. `complete.grammar`, `default.grammar` and `ladder.grammar` load; the
    MM_grammar XOR bar and the MM_query_reasoning validation are unchanged.
12. Every compose/generate call receives only its owned word stream,
    conceptual-space view and captured primed-symbol view; it cannot obtain
    LTM, taxonomy, a controller, or an executor. A thought call receives the
    same owned base views plus only its descriptor-authorized, bounded LTM and
    taxonomy methods and the shared work meter.

## 10. Documentation

`QueryContracts.md` (replace the `<Queries>` section with §§2–4),
`Language.md` (grammar form), `Params.md` (no new knobs; note the removed
attribute), the grammar files' own commentary, `GradientFlow.md`, `Testing.md`,
and `todo.md`. `Reasoning.md` remains a protected historical document until
its separately reviewed cleanup.
