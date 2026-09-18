# Conceptual-taxonomy PartOf evidence

PartOf reads ConceptualSpace's existing conceptual reference records. A native
`("sym", concept_id)` handle is an address; it is never interpreted as a
dictionary row or a numeric feature. The derived invocation view contains no
authoritative memory and creates no allocator, concept, fact or learned
parameter. Raw percept/whole codes, vector containment and LTM world-relation
rows cannot establish this relation.
[Reference validation](../bin/Taxonomy.py#L11),
[bounded capture](../bin/Taxonomy.py#L117).

## Evidence and domains

A recorded `part` reference points from the constituent to its owner; a
`whole` reference points from the owner to its containing concept. Both direct
definitions and reified relations are visible. For a reified relation concept
R with part A and whole B, the evidence is the two native links A → R → B;
the endpoint sets need not have been modified. Each proof keeps its record
owner, role and endpoints. Retired/unavailable references cannot supply a
link. The snapshot does not change when its source is subsequently edited.
[Capture](../bin/Taxonomy.py#L117),
[proof search](../bin/Taxonomy.py#L69).

Structural inclusion has support 1 when a path exists; repeated links do not
increase it. Sparse learned weights are not interpreted as world-fact
confidence. Missing paths remain unknown. A supported path refutes the
negated inclusion question; a missing path does not prove that negation.
LTM existence degrees and conflicting world facts remain the separate
[Exist evidence domain](ExistenceEvidence.md).
[PartOf evaluation](../bin/reasoning.py#L586).

The public legacy `QuerySpec` aliases `PartOf`, `part`, `queryPart` and
`isPart` select conceptual-taxonomy evidence. `whole` / `isWhole` reverse
their supplied operands into that canonical direction. Unsupported requested
domains fail explicitly. A legacy vector-only operand has no grounded
concept reference and returns unknown with an `unbound_concept_reference`
diagnostic; it is not silently snapped to a codebook row.
[Interface](../bin/reasoning.py#L59),
[evidence adapter](../bin/reasoning.py#L376).

`parts` and `wholes` return typed neighboring references and their native
record sources. The legacy kernel's `part(..., mode="taxonomy")` uses this
reader. Its former `meronomy` mode is rejected as unsupported: a perceptual
mereonomy adapter must declare its own domain and source before it can run.
[Neighbors](../bin/reasoning.py#L397),
[kernel adapter](../bin/thinking.py#L305).

## Public routes and provenance

`TruthGroundedReasoner.evaluate`, `NeuralToolUser.run`, `BasicModel.reason_about`
and the PartOf path through `BasicModel.think_about` use this evidence source.
The model entries do not initialize or read the old global vector-proposal
route. The returned reasoning result preserves the complete evidence dict,
including unknown/incomplete diagnostics, rather than only its posture.
[Reasoner](../bin/reasoning.py#L586),
[tool entry](../bin/reasoning.py#L762),
[model entry](../bin/Models.py#L22259),
[kernel entry](../bin/Models.py#L22277).

The existing frame kernel may still follow a taxonomy neighbor into a child
question. A child contributes to its parent only through a checked native
taxonomy hop, with the same requested goal and polarity. An unrelated true
child or numeric testimony cannot certify the parent's taxonomic inclusion.
The final result retains the hop sources and the child's evidence. An
incomplete-read diagnostic remains visible but contributes no invented zero
evidence to an otherwise supported interval.
[Selected step](../bin/thinking.py#L219),
[aggregation](../bin/thinking.py#L398).

Taxonomy queries are read-only. Neither a successful proof nor the legacy
`materialize` flag grants permission to append a world-fact lemma. Legacy
incomplete relation testimony remains unverified and cannot satisfy the
taxonomy reader or full-description Exist.
[Public PartOf helper](../bin/reasoning.py#L429),
[write boundary](../bin/thinking.py#L484),
[testimony](../bin/thinking.py#L355).

## Bounds, persistence and gradients

The reader defaults to at most 256 scanned concepts and 1024 native reference
records; direct `part_of` traversal allows at most 8 links and 1024 examined
edges. `evaluate` further bounds edge examinations by `beam × max_steps`.
Every fetched record, including an ignored raw-domain reference, is counted.
Limits and unavailable references yield explicit incomplete diagnostics.
Node scans, record reads and edge expansions are distinct counters. When a
selected QueryWorkBudget is supplied, each of those actions debits the same
meter before its read; capture reserves a bounded share for proof traversal.
Local limits still only tighten that allowance. Standalone taxonomy audits can
omit the meter and retain their explicit bounds.
[Capture limits](../bin/Taxonomy.py#L117),
[traversal limits](../bin/Taxonomy.py#L69),
[query limits](../bin/reasoning.py#L586). See
[shared query work](QueryWork.md).

Concept identities and ordered reference records already belong to the
existing structural checkpoint sidecar. The view itself is not serialized;
a restored model captures it again from that owner. The model save/load probe
checks identical proof sources after strict restore and loss of support when
the restored reified relation is retired. No checkpoint schema or learned
parameter was added for this reader.
[Structural ownership](../bin/Models.py#L4409).

Native reference traversal and its evidence are hard reads, with no gradient
through the discrete lookup/path choice. IDs do not enter numerical chooser
features. The old operation head's behavior-cloning examples now come from
bounded native taxonomy paths; this checks the existing training mechanism,
not learned question utility. The architecture-wide gradient contract remains
in [GradientFlow](GradientFlow.md).
[Trace curriculum](../bin/thinking.py#L621),
[operation-head training](../bin/thinking.py#L610).

## Compatibility and unfinished work

Old geometric/world-row helper methods have explicit `legacy_...` names;
`NeuralToolUser.run_legacy_world` retains the historical vector-proposal
experiment. Its tests are labelled as legacy. Those helpers are not used by
public PartOf dispatch. The older optional prediction/answer policy objectives
still use their explicitly named legacy evidence machinery and are not the
default sentence-expectation controller.
[Legacy proposal entry](../bin/reasoning.py#L772),
[legacy prediction experiment](../bin/reasoning.py#L864),
[legacy answer objective](../bin/reasoning.py#L972).

[Query contracts](QueryContracts.md) supplies the explicit shared-VP adapter
above this evidence reader. This change does not implement linguistic relation/sense composition, the causal
answer adapter, ordinary levelled thought history, nested semantic traversal,
anticipation isolation, residual policy credit or learned utility. The frame
kernel remains a legacy controller. The full integrated specification remains
unfinished, and the separate two-truths design remains deferred as requested.
[Implementation order](plans/2026-09-15-next-sentence-as-the-production-objective.md#10-consolidated-implementation-and-verification-order).

## Validation

The repository's **37 new probes pass in 3.33 s**. The sixteen affected
files pass **277 tests, with one skip, two expected failures and seven
warnings, in 173.39 s**. The full suite passes with exit status zero: **4317 passed, 51 skipped, 7 xfailed, 184 warnings, 4 subtests passed in 5604.66s (1:33:24)**. All 569
runtime/test/configuration files remained frozen.
[Focused green](benchmarks/2026-09-16-taxonomy-query-data/focused-green.log),
[affected green](benchmarks/2026-09-16-taxonomy-query-data/affected-green.log),
[full green](benchmarks/2026-09-16-taxonomy-query-data/full-green.log),
[frozen source manifest](benchmarks/2026-09-16-taxonomy-query-data/full-validation-manifest.json).

Full-suite command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```


The existing public-routing probes failed before the fix. Further failures
caught native read accounting, public entry points, operation-head curriculum,
unrelated child testimony, public-helper bypasses and zero-threshold unknown
classification. Each received a regression test.
[Routing red](benchmarks/2026-09-16-taxonomy-query-data/routing-red.log),
[view red](benchmarks/2026-09-16-taxonomy-query-data/view-red.log),
[work counter red](benchmarks/2026-09-16-taxonomy-query-data/work-counter-red.log),
[entry red](benchmarks/2026-09-16-taxonomy-query-data/entry-red.log),
[policy red](benchmarks/2026-09-16-taxonomy-query-data/policy-red.log),
[boundary red](benchmarks/2026-09-16-taxonomy-query-data/boundary-red.log),
[public helper red](benchmarks/2026-09-16-taxonomy-query-data/public-helper-red.log),
[unknown red](benchmarks/2026-09-16-taxonomy-query-data/unknown-red.log).
