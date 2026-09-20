# Conceptual-taxonomy PartOf evidence

PartOf reads ConceptualSpace's existing conceptual reference records. A native
`("sym", concept_id)` handle is an address; it is never interpreted as a
dictionary row or a numeric feature. The derived invocation view contains no
authoritative memory and creates no allocator, concept, fact or learned
parameter. Raw percept/whole codes, vector containment and LTM world-relation
rows cannot establish this relation.
[Reference validation](../bin/Taxonomy.py),
[bounded capture](../bin/Taxonomy.py).

## Evidence and domains

A recorded `part` reference points from the constituent to its owner; a
`whole` reference points from the owner to its containing concept. Both direct
definitions and reified relations are visible. For a reified relation concept
R with part A and whole B, the evidence is the two native links A → R → B;
the endpoint sets need not have been modified. Each proof keeps its record
owner, role and endpoints. Retired/unavailable references cannot supply a
link. The snapshot does not change when its source is subsequently edited.
[Capture](../bin/Taxonomy.py),
[proof search](../bin/Taxonomy.py).

Structural inclusion has support 1 when a path exists; repeated links do not
increase it. Sparse learned weights are not interpreted as world-fact
confidence. Missing paths remain unknown. A supported path refutes the
negated inclusion question; a missing path does not prove that negation.
LTM existence degrees and conflicting world facts remain the separate
[Exist evidence domain](ExistenceEvidence.md).
[PartOf evaluation](../bin/reasoning.py).

The public legacy `QuerySpec` aliases `PartOf`, `part`, `queryPart` and
`isPart` select conceptual-taxonomy evidence. `whole` / `isWhole` reverse
their supplied operands into that canonical direction. Unsupported requested
domains fail explicitly. A legacy vector-only operand has no grounded
concept reference and returns unknown with an `unbound_concept_reference`
diagnostic; it is not silently snapped to a codebook row.
[Interface](../bin/reasoning.py),
[evidence adapter](../bin/reasoning.py).

Grammar-open `part` forms return typed neighboring references and their native
record sources. They share the canonical operation identity and checked reader.

## Public routes and provenance

`TruthGroundedReasoner.evaluate` is a checked evidence adapter.
`BasicModel.reason_about`, `think_about`, `answer_query` and normal answer
resolution enter `run_selected_thought`, the one boundary controller. Typed
results preserve full evidence and incomplete diagnostics.

A selected `what(Q)` opens a child on the ordinary history owner. Its checked
return is recorded as a causal source of its parent's result. An unrelated
true child cannot establish a different taxonomy edge. Reader traversal may
prove several native edges within one operation; that is not evidence of
policy-selected decomposition. The [review probes](../test/test_thought_review.py)
separately exercise a chooser-selected descent and its policy credit.

Taxonomy queries are read-only. Successful proofs cannot append world-fact
lemmas or change conceptual definitions. Unsupported/world-row evidence cannot
certify taxonomy inclusion. The frame kernel, neural-tool facade, addressees,
legacy geometric readers and bridge-policy loss have been deleted.
See [checked registry](../bin/Queries.py), [evidence readers](../bin/reasoning.py)
and [the controller](SelectedMeaning.md).

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
[Capture limits](../bin/Taxonomy.py),
[traversal limits](../bin/Taxonomy.py),
[query limits](../bin/reasoning.py). See
[shared query work](QueryWork.md).

Concept identities and ordered reference records already belong to the
existing structural checkpoint sidecar. The view itself is not serialized;
a restored model captures it again from that owner. The model save/load probe
checks identical proof sources after strict restore and loss of support when
the restored reified relation is retired. No checkpoint schema or learned
parameter was added for this reader.
[Structural ownership](../bin/Models.py).

Native traversal and evidence are hard reads, with no derivative through the
lookup/path choice. Native IDs do not enter numerical chooser features. The
one thought policy receives supplied-answer quality and actual shared work;
verifier traces do not manufacture training labels. Learned utility remains
unproven pending the held-out, matched-compute multi-seed study.
See [GradientFlow](GradientFlow.md).

## Remaining work

Natural wording must be learned by compose/generate. The previous standalone
codec is removed and its learning gate reopened. Expectation/residual credit,
generation ownership and causal utility remain explicit todo items. The separate
two-truths design remains deferred. See [SelectedMeaning](SelectedMeaning.md).

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
