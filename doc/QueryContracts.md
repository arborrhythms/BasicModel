# Checked boundary-query contracts

Implementation reference, verified September 16. This is the contract/executor and shared-VP
foundation; the selected linguistic derivation adapter and ordinary levelled
controller remain separate integration work.

## Declarations and relation identities

Grammar loading validates each `<Queries>` declaration before changing the
configured grammar. Unknown executors, wrong arities, malformed signatures,
duplicates and deferred tense queries fail explicitly. The real file loader
preserves query and anchor lists while expanding order alternatives only in
rule bodies. Both `complete.grammar` and production `ladder.grammar` declare
`what(Q)` alongside the distinct `query(X,Y)` lookup.
[Configuration](../bin/Language.py#L1027),
[normalization](../bin/Language.py#L514),
[signature validation](../bin/Queries.py#L528).

Each frozen signature declares its relation/domain identity, canonical argument
roles and types, occupied/open/absent/result roles, result kind, read/write
scope, evidence kind, grammatical compose faces and executable function. Each
grammar owns its catalog, preserving the established copy path. Definitions
add no semantic vector store, learned parameters or checkpoint tensor schema.
[Contract](../bin/Queries.py#L58).

| Interface | Domain / read owner | Canonical inputs | Result / evidence |
|---|---|---|---|
| `isTrue(X)` / `exist(X)` | Accepted LTM facts | Complete description in NP1 argument | Both support degrees and occurrence provenance |
| `isPart(X,Y)` / `part` / `PartOf` | Conceptual taxonomy | Native NP1 and NP2 references | Structural proof or unknown/incomplete |
| `isWhole(X,Y)` / `whole` | Same part relation/domain | Surface arguments map to NP2, NP1 | Same canonical evidence |
| `parts(X)` | Conceptual taxonomy | Bound NP2; open NP1 | Native references and source records |
| `wholes(X)` | Conceptual taxonomy | Bound NP1; open NP2 | Native references and source records |
| `isEqual(X,Y)` / `equal` | Conceptual payload identity | Equal-width NP1 and NP2 values/references | Existing graded identity score |
| `query(X,Y)` | LTM records | Two conceptual operands | Complete retrieved records, retaining their evidence kind |
| `quantize(X)` | Existing conceptual codebook | Conceptual value/reference | An allocated conceptual atom and native handle |
| `arma(X)` | Row-local preceding observation context | Complete trajectory description | Full role/presence estimate |
| `what(Q)` | The active episode's controller | Full conceptual question | Scheduled conceptual subgoal |

`part`, `whole`, `parts` and `wholes` share one semantic relation identity;
the interface's role mapping determines direction or the open operand. This
metadata links the intended grammatical and executor faces. It does not by
itself populate the completed sentence's middle VP; the explicit binding
adapter below constructs and dispatches that typed meaning.
[Definitions](../bin/Queries.py#L305).

## One native VP, grammatical mode and canonical roles

`GrammaticalQueryRegistry.install` explicitly binds one named native concept
per relation/domain at setup. It reuses ConceptualSpace's allocator, aligned
payload rows and checkpointed named-concept registry. Alias forms, including
`part`/`isPart`, and converse `whole`/`isWhole` share that reference and payload.
The adapter adds no parameter, parallel semantic memory or embedding table.
Repeated setup and integrated checkpoint restore preserve the same handles.
[Binding](../bin/Queries.py#L398),
[existing named owner](../bin/Spaces.py#L18963),
[checkpoint owner](../bin/Models.py#L4409).

`form` builds canonical `[NP1, VP, NP2]` payloads, presence masks, native
references, grammatical mode, polarity, bindings and scope. Converse forms
permute operands into this order. Open/absent roles have no fabricated concept.
Formation neither executes an operation nor mints a concept or LTM occurrence.
An assertion and its corresponding question differ in mode, with the same VP.
[Formation](../bin/Queries.py#L431).

`execute` derives the checked interface from the middle VP and occupied/open
roles. Only an interrogative meaning with a declared execution interface and
the correct conceptual width can execute. Compose-only rules may represent a
relation without licensing execution. Foreign/unavailable native references
fail explicitly. The result retains the entire query proposition, scope and
separate evidence; negation swaps known support channels and leaves missing
evidence unknown.
[Dispatch](../bin/Queries.py#L472),
[execution](../bin/Queries.py#L496).

A description-valued argument references an existing LTM occurrence. The
outer NP retains that stable reference and a sum of its occupied full-width
roles divided by the square root of their count for illumination. That summary
is not its structural record: execution resolves the complete description,
including VP, NP2, bindings and scope. Missing namespaces, missing metadata or
an exhausted local read limit fail without substituting NP1 or another row.
Direct uncommitted descriptions cannot be silently stored during candidate
formation. The eventual boundary controller must own their prior commitment.
[Occurrence resolution](../bin/Queries.py#L348).

## Selected execution and evidence

`QuerySignature.invoke` validates arity, types and domain before executing.
Concept IDs remain typed addresses. Conceptual vectors must be finite and
full-width; native payloads are read from allocated, active ConceptualSpace
rows without creating an allocator or assigning a row. SymbolSpace snapping
is not part of these executors. Results retain their domain, semantic identity,
result kind and evidence kind.
When a ConceptualSpace owner is available, the signature rejects vectors
and complete descriptions of a different width before any executor runs.
[Invocation](../bin/Queries.py#L99),
[native lookup](../bin/Queries.py#L129),
[conceptual quantization](../bin/Queries.py#L239).

Exist preserves all occupied roles, scope, bindings and referenced constituents
through the existing complete-description matcher. Its optional record limit
bounds the inspected prefix, including non-fact eligibility reads. It reports
the actual scan count and incomplete capture separately from missing required
metadata. A limit does not create negative evidence or increase support.
The compatibility reasoner retains its unlimited default; checked execution
passes its declared local bound.
[Bounded lookup](../bin/reasoning.py#L156).

The distinct `query(X,Y)` retrieval returns full records, including their VP,
occurrence, evidence kind and context. Finding an observation or estimate does
not admit it as a fact. Taxonomy queries keep native proof provenance and cannot
materialize world assertions. Missing or truncated neighbor reads preserve
incomplete diagnostics even when the result set is empty.
[LTM retrieval](../bin/Queries.py#L209),
[taxonomy neighbors](../bin/Queries.py#L187).

`arma` reads a complete `MeaningExpectation`, including distinct roles and
presence logits. `expect_next_meaning(record=False)` preserves the pending
comparison for the next external observation; it neither observes another
sentence nor turns the estimate into evidence. The supplied description
identifies the current trajectory context; the predictor reads its established
row-local preceding observations.
[Prediction executor](../bin/Queries.py#L284),
[non-staging read](../bin/Layers.py#L10064).

`what(Q)` requires interrogative meaning and passes the complete question to the supplied active-controller
continuation and fails explicitly when that continuation is unavailable.
It does not call `Data.what`, provide oracle subgoals or hide a second solver.
The ordinary controller still has to supply this continuation and record the
selected action in its shared episode history.
[Subgoal executor](../bin/Queries.py#L297).

## Bounds and phase integration

The call context declares row and local node/record/path/edge bounds. Invalid
negative or noninteger bounds fail. A zero conceptual-read budget cannot fetch
a native payload. These limits do not constitute a new episode allowance;
the future ordinary controller must intersect them with its remaining shared
budget and charge actual work. A selected subgoal uses that same controller.
[Context](../bin/Queries.py#L19).

This interface does not yet supply the model's per-row sentence-phase guard.
The normal resolver's raw-text legacy route must still be replaced by dispatch
through this registry from the completed grammatical VP, with queries masked throughout
compose, reconstruction and generate. Merely having checked callables does not
establish those execution or causal-answer gates.

## Native referents in captured programs

`AnswerProgram.concept_ids` keeps the native concept identities of its leaves
beside dictionary `rows` and surface `word_rows`. Capture selects an OBJECT ID
only where forward selected the OBJECT row; otherwise it uses the WORD ID.
A missing OBJECT identity remains unknown instead of falling back to a different
WORD referent. Fractional metadata cannot be truncated into a valid address.
Older programs default to unknown IDs (`-1`), never inferred row identities.
[ID selection](../bin/Models.py#L11110),
[capture](../bin/Models.py#L12072),
[owned value](../bin/Understanding.py#L25).

The addresses survive later staging and detached recall copies. They do not
enter numerical chooser features or change the 21-value compiled result.
This supplies grounding for the later selected-derivation semantic adapter;
it does not itself reconstruct a missing grammatical VP or nested clause.

## Gradients and persistence

Contracts, addresses, hard matching and discrete evidence selection have no
ordinary gradient. Selected conceptual payload clones and full predictor
outputs retain their existing continuous paths. Durable records/targets remain
detached by their existing owners; the current occurrence reader therefore
returns a detached description. Live episode occurrences will need the same
owner's live read path before controller integration can claim episode credit.
No new loss or optimizer-owned parameter is
added. Query policy credit, the combined downstream budget and one optimizer
step remain governed by [GradientFlow](GradientFlow.md).

Signature definitions are rebuilt from checked grammar declarations. Native
addresses stay in the existing program value and allocator ownership; no
parallel planner, semantic memory or VP embedding table is introduced.

## Validation and remaining work

The 53 new reviewer cases cover declarations, effects, loader/copy behavior,
native leaf identities, shared VP/checkpoint identity and width/occurrence bounds.
The initial focused run passed 57 tests; the 36-file affected run passed
353 tests with five skips in 366.32 seconds, including output, reconstruction
and compiled word paths. After the final direct-call width guard, 18 affected
files passed 138 tests with one skip in 13.31 seconds. Each managed run exited
zero. The full suite completed: **4371 passed, 51 skipped, 7 xfailed, 184 warnings, 4 subtests passed in 7222.95s (2:00:22)**. All 579
runtime/test/configuration files remained frozen.

[Initial focused](benchmarks/2026-09-16-checked-query-data/focused-green.log),
[affected](benchmarks/2026-09-16-checked-query-data/affected-green.log),
[final affected](benchmarks/2026-09-16-checked-query-data/final-affected-green.log),
[declaration red](benchmarks/2026-09-16-checked-query-data/declarations-red.log),
[VP red](benchmarks/2026-09-16-checked-query-data/grammatical-vp-red.log),
[width red](benchmarks/2026-09-16-checked-query-data/owner-width-red.log),
[full green](benchmarks/2026-09-16-checked-query-data/full-green.log),
[frozen source manifest](benchmarks/2026-09-16-checked-query-data/full-validation-manifest.json).

Full-suite command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```

Remaining integration includes linguistic recognition, surface/sense
permutations, canonical selected-parse
meaning through all three observation writers, causal answer resolution,
ordinary levelled history and budget, nested retention, prior-view isolation,
residual policy credit, catalog separation and measured learned utility.
The separate two-truths specification remains deferred to the next session.

## Live thought occurrences

An occurrence argument may designate a durable `ltm` occurrence or a live
row-local `thought` occurrence. The latter resolves only through the existing
`WhatInteractionMemory` owner and retains its episode gradient until the
explicit credit boundary. Resolving it does not execute a query, alter a level,
or promote a question or estimate into evidence. See [ordinary thought
history](ThoughtHistory.md).
