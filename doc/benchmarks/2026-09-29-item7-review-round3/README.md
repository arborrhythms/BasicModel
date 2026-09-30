# Item 7, review round 3 — stopped for review, candidate red

No changes are committed. This receipt follows spec sections 16–18. Earlier
receipts remain historical records. HEAD is `1678ee1fb79c474ffaee5725fd035716de5f1913`.

The one [source-matched full sweep](full-sweep/summary.json) completes all
**5,050 cases: 4,714 passed, 13 failed, 322 skipped, and one expected failure**.
All **170 item 7 cases pass**. The final XOR table restores the fourteen
required cases, including both XOR_exact CLI gates; both XOR_grammar gates
remain red. The explicit graph-release gate, one reconstruction memory
attempt, two MM_grammar attempts and the native predictor-context observation
also remain red, separately from the default sweep.

Six of the thirteen full-sweep failures passed in round 2. Fourteen of that
receipt's sixteen regressions now pass; FineWeb checkpoint restoration and
attended-field identity still fail, with their current causes recorded.
The [failure ledger](full-sweep/failure-ledger.json) and
[diagnoses](full-sweep/diagnoses.json) retain every failure. This is an
incomplete candidate for Claude's review, not an acceptance claim. Nothing
has been committed, staged, pushed or re-baselined.

Protocol declared before this pass's measurements:

* Measure the complete XOR table on HEAD and the incoming candidate first,
  then after Stage A, after Stage B, and on the final tested source.
* Every repair has a failing probe saved before its implementation. No
  configuration, protected assertion, threshold or passing seed is selected.
* Each gate uses the existing 8 GiB process-tree guard. A stopped case is run
  once without the memory guard as a separately labelled diagnostic. A
  diagnostic does not turn a failed gate green.
* Reconstruction uses seeds **0, 1, 2, 3, 4, 5, 6, 7**, all eight on both
  trees, with definitions enabled on the candidate. All results are retained.
* The final MM_grammar measurement uses **ten fresh, unseeded runs per tree**,
  each for the full **900 epochs**, reporting the ending error and all four
  predictions. It is separate from the existing early-stop gate.
* Stage A retains words' parts and wholes and changes their read. Stage B
  replaces META with identity-based definition rows. Only after both stages
  come the optimizer, graph-release, reconstruction and final sweep steps.

`run_xor.py` includes both XOR_exact command-line gates and all three slow
MM cases as explicit gates, alongside grounded XOR, the curriculum, and both
XOR_grammar gates. `xor_observer.py` only records returned CLI output and
loss inputs; it changes no prediction, objective, training budget or gate.

The XOR tables below retain each stage's result. XOR_grammar is measured
and remains outside item 7's acceptance.

## Incoming XOR baseline

Both trees were measured before runtime changes, with no chosen seed. Raw
receipts and source manifests are in `baseline-head/` and
`baseline-candidate/`; their `summary.json` files retain every assertion and
`measurements.jsonl` retains CLI output and the slow MM tests' predictions.

| Proof | HEAD | Incoming candidate |
|---|---|---|
| Grounded XOR, six exact cases | 6 pass | 6 fail, zero discovered cases |
| XOR_exact curriculum, three cases | 3 pass | 3 fail, eleven cases instead of four |
| XOR_exact CLI, output MSE < .05 | crashes, `int(None)` in alternative admission | fails, MSE 0.24809641 |
| XOR_exact CLI, inputs reconstruct | same crash | passes, 4/4 |
| MM_xor, convergence < .20 | passes, stopping MSE 0.17414780 | passes, stopping MSE 0.18838423 |
| MM_xor, signal < .26 | passes, stopping MSE 0.25025421 | passes, stopping MSE 0.25000277 |
| MM_grammar, signal < .20 | passes, stopping MSE 0.19989115 | passes, stopping MSE 0.19969021 |
| XOR_grammar, class accuracy | crashes, symbol codebook capacity | fails, class 0 accuracy 0.0 |
| XOR_grammar, reconstruction | same capacity crash | fails, 0/4 |

These MM measurements are the tests' stopping errors, not full-budget errors.
The separate ten-run comparison below trains for all 900 epochs.

## Stage A failing probes

All 14 probes in `probes_stage_a.py` fail before the repairs
(`stage-a-before/group-00`): canonical fusion, repeat recognition despite
property drift, atomic capacity refusal, word presence at its own bracket
(a minimal case and the three configurations), sparse property evaluation,
and six exact grounded XOR cases with the word boundary added. The original
grounded test's assertions are untouched; only the new boundary variant
omits the unrelated-percept assertion as §17.7 specifies.

The category gate stops at 8.34 GiB and the 64-word trace at 8.29 GiB. Each
was repeated once without the memory guard, separately from its gate:
category still fails (`codebook never enabled`), peak 10.98 GiB; the trace
passes, peak 13.56 GiB. Those diagnostics are retained in
`stage-a-before/unguarded-01` and `unguarded-02` and confer no passing gate.

## Stage A outcome

The fourteen pre-fix acceptance probes now pass. Category codebook: 3 pass,
peak 0.90 GiB. The 64-word trace passes at 0.94 GiB. Compile-cache recovery,
word router, attended-field reuse, concept memberships and percept-definition
files pass in full. The aligned word-publication case exposed a taxonomy
check that rejected Stage A's shared word/object identity; the failing probe
is retained in `stage-a-regressions/group-05`, and the affected file then
passes 4/4 in `stage-a-taxonomy`. Stage B gives the two concepts distinct IDs.
The optimizer layout remains for §16.6.

| XOR proof | Stage A |
|---|---|
| Grounded, six exact cases | 6 pass |
| Curriculum, three cases | 3 pass |
| XOR_exact CLI crisp output | pass; MSE **0.0**, predictions **0 1 1 0** |
| XOR_exact CLI reconstruction | pass; **4/4**, MSE 0.0 |
| MM_xor convergence | pass; stopping MSE 0.18661237 |
| MM_xor signal | pass; stopping MSE 0.24999011 |
| MM_grammar signal | pass; stopping MSE 0.19847000 |
| XOR_grammar class accuracy | fail; class 0 accuracy 0.0 |
| XOR_grammar reconstruction | fail; 0/4 |

Raw measurements, predictions and source manifests: `stage-a-xor/`.
XOR_grammar remains visible and is not an item 7 condition. The XOR table
was measured before the isolated taxonomy-check repair; that repair's
source manifest and rerun are recorded separately, not claimed as the same
snapshot. The final full receipt below uses one source snapshot.

## Stage B probes and repairs

All eleven `probes_stage_b.py` cases failed before Stage B. A genuine HEAD
META fixture was generated by `make_legacy_definition_fixture.py` against the
unchanged HEAD archive and retained as `legacy-definition.pt`.

The permanent `test/test_item7_definitions.py` now passes 11/11, including
new-word accounting (two identities, one inventory row, one store row),
bidirectional indexed lookup, ambiguity and synonyms, store-capacity
atomicity, fixed `.when` with refreshed recency, forgetting, all three
unchanged configurations, code-independent identity, and HEAD migration.
The external probe draft had three fixture errors discovered on this pass:
it assumed the allocator was eagerly initialized; it represented every
non-radix word by the same literal `[7]`; and it assumed the optional discourse
head was already constructed. The permanent tests explicitly initialize the
allocator, supply each word's own byte literals, and call the existing head
constructor. The original draft and all failed outputs remain unchanged.

`stage-b-first/` retains the serial capture failure: it addressed a word by
its former inventory seat, which now belongs to its object. The reader now
uses the definition index. `stage-b-probes/` passes that complete file (4/4)
and the eleven definition cases. Grounded XOR remains 6/6 in
`stage-b-first/group-01`. The complete Stage B XOR table is recorded below.

Definition operand vectors are null, permitted by §17.2; `refs[0]` and
`refs[2]` are the authoritative native identities. The fixed DEF predicate
has an identity code. A definition's own recency-row address uses a tagged
store occurrence (`2^62 + occurrence_id`), so addressing the record does not
consume a third conceptual identity. No reader compares operand vectors.

Stage B's additional failing receipts are retained under `stage-b-integrity-before`,
`stage-b-word-definition-before`, `stage-b-readers-before`,
`stage-b-pending-before`, `stage-b-index-owners-before`,
`stage-b-object-resolution-before`, `stage-b-lexical-before`, and
`stage-b-structural-pending-before`. They caught copied-index ownership,
reservation overbooking, a missing word before field discovery, mutation of a
word's definition during identity pruning, stale decoding after forgetting,
lost pending admissions on checkpoint, taxonomy form propagation, META
re-registration during migration, readers of retired caches, and word/object
ambiguity at order zero. The first pending-checkpoint draft omitted the
store's tensor state; the corrected probe loads tensors before the semantic
sidecar and fails on the missing pending reservation. Both runs are retained.

The permanent test 33 now asserts all four word concepts exist before case
discovery. The earlier Stage A variant injected the boundary but failed to
assert this prerequisite; its six green outcomes alone did not prove that
requirement. `stage-b-reader-repairs` and `stage-b-completion` run the stronger
six cases successfully. Their captured output also records the unrelated
percept-event counts, as §17.7 requires.

Each test port retains its old and new body in `test-ports.json`,
`reader-test-ports.json`, and `index-test-ports.json`. The first file records
intermediate edits as well as the final ports; `final-test-ports.json`
consolidates their final bodies for review. No numerical gate was relaxed. The original
25-file port probe ran slow cases too: four were stopped by the guard and
repeated once without it. All four diagnostics passed, at respectively
9.97 GiB (`test_reading_promotes_the_batch_words`), 16.18 GiB
(`test_ws_routes_universe_on_parallel_path`), 20.46 GiB
(`test_unconditional_integration_end_to_end`) and 9.78 GiB
(`test_sparse_concept_forward_smoke`). These remain resource failures of
that probe run, not passing gates. Subsequent ordinary affected-file runs
use the suite's normal slow skips; the requested slow gates run explicitly.

The decoder's independent row-spelling caches were removed: generation and
lexicalization read forms through the definition index. A word's native
identity has no second object payload row and is excluded from object
selection. Pending field admissions are checkpointed with their reserved
store row; their DEF row is written when the field admits the object.

The `stage-b-index-final` run included one mistyped 64-word selector, recorded
as a collection error. `stage-b-completion` uses the actual 64-word selector;
its gate passes at 0.94 GiB, and the three category EM cases pass at 0.90 GiB.

## Stage B outcome and XOR table

`stage-b-completion` passes all 57 cases: definition and integrity acceptance,
word admission (including all six strengthened test-33 cases), lexical-order
resolution, category EM, the 64-word trace, and the serial/parallel interpret
file. The definition index owns word/object lookup; old spelling caches are gone.
All six test-33 variants recorded **0** unrelated percept events (not asserted).

| XOR proof | Stage B |
|---|---|
| Grounded, six exact cases | 6 pass |
| Curriculum, three cases | 3 pass |
| XOR_exact CLI crisp output | pass; MSE **0.0**, predictions **0 1 1 0** |
| XOR_exact CLI reconstruction | pass; **4/4**, MSE 0.0 |
| MM_xor convergence | pass; stopping MSE 0.19748278 |
| MM_xor signal | pass; stopping MSE 0.25004476 |
| MM_grammar signal | pass; stopping MSE 0.19947769 |
| XOR_grammar class accuracy | fail; class 0 accuracy 0.0 |
| XOR_grammar reconstruction | fail; 0/4 |

Raw receipts and all four MM predictions are in `stage-b-xor/summary.json`.
This table and `stage-b-completion` have the same source snapshot. Outcomes
match Stage A; the independent unseeded MM stopping errors differ. The
configuration audit records no changed XML, XSD or grammar file since the
round-3 baseline and no seed-setting call in an item-7 test.


## Optimizer layout (§16.6 / finding T)

The unchanged checkpoint-resume probe fails in `optimizer-before`, with dense
parameter groups **91 / 1** and a row-local group of **3**. The first repair
covered the space-level adopter; `optimizer-after` still fails because
`SparseLayer._replace_values` creates the parameter before that adopter runs.
Both paths now put a new definition weight in the existing first dense group,
where `getOptimizer` puts the same parameter when constructing a restored model.
No learning rate, existing moment, test assertion or configuration changes.

`optimizer-completion` passes **50 tests**, with one existing FineWeb skip,
across checkpoint resume, dynamic part capacity, row-local optimization and
sparse layers. Its `layout.json` records **92 / 3** on both the saved and
restored models, and 16 saved optimizer states restored by name.

## Graph release diagnosis (§16.7 / finding S)

The unchanged two-epoch gate passes at HEAD at **5.91 GiB**. The candidate
stops at **8.22 GiB** under the same **8 GiB** guard. Its one unguarded
repeat fails at **16.76 GiB**, before reaching the graph-release assertions.
See `graph-release-head/` and `graph-release-candidate/`; the unguarded
result is a diagnostic, never a passing gate.

The exception is a **setup capacity / clause predicate mismatch**, not an
inventory filled by sentence endings. `graph-setup-diagnosis.json` records
this before any sentence: zero inventory rows, zero LTM rows, next identity
1, four snap seats, and eight unavailable thought families. The registry
reserves native thought VPs all-or-nothing. `ClauseJournal.recover` calls
`clause_reference('part')`, which calls the executable `operation_spec` and
raises that setup-time capacity error. The structural clause recovery path
therefore depends on a thought capability the small model never acquired.
The first observer draft assumed the allocator was eagerly initialized; the
corrected observer reads its lazy initial state without initializing it
(`graph-setup-completion`: pass, 2.92 GiB). No runtime fix or configuration
expansion is claimed for S; this is the diagnosis requested by §16.7.

An address for each newly completed absolute S is intended by §§2, 3.3 and
3.5: each S writes its ended state, and a later S may refer to that state.
The next state gets another row and can point to the earlier row; identity
is imputed by the predictor. Repeated relations instead use the explicit
relation deduplication rule. Re-reading an unchanged word definition does
not create another word, object or DEF row. These are separate counts.
The native identity budget's coupling to inventory capacity remains a
streaming limitation; it did not cause this setup-time failure.

`final-test-ports.json` consolidates every recorded port to its earliest
captured old body and its actual final body, including renamed and retired
META/ReferenceTable tests. Intermediate edits remain in the three earlier
ledgers. It now contains 97 unique entries: 69 ports, 22 retirements and six
renames, with no protected numerical gate relaxed. The predeclared
reconstruction dispatcher initially rejected thread
signal handlers before any worker started; `reconstruction-launch-error.txt`
records that infrastructure error. The same measurement now runs in at most
three independent guarded supervisor processes.

The first reconstruction observer also assumed HEAD had the candidate's
common LTM store. It raised after HEAD's first validation phase; the partial
runs were stopped before changing that observer, and **all eight seeds on
both trees** restarted with the store-absent case handled. There was no
runtime or protocol change. The interrupted receipts remain in the two
`*-reconstruction-observer-error/` directories, with the reason in
`reconstruction-observer-error.json`; they are not used as measurements.

The actual repeated-reading observation (`sentence-identity-diagnosis.json`)
keeps **12 physical concept rows and 5 definitions** on both presentations;
next identity advances **23 → 25**, store rows **10 → 12**, and idea rows
**2 → 4**. The same four inputs do not cause a blanket four new identities:
this unseeded reading also selected deduplicated relations. Its numerical
probe passes; the grouped receipt retains the separate observer setup error.

The supplemental unseeded MM_xor / MM_grammar validation readings complete,
with respectively **4/4** and **4/36** store rows being definitions. Neither
workload makes a new sentence estimate, so a definition fraction among actual
predictor context reads is **undefined (0 reads)**, not a measured zero effect.
MM_ladder explicitly sets `sentenceExpectation=false` and likewise supplies
no such reads during reconstruction. Two supplementary packed-input attempts
on the unchanged MM_grammar_wording configuration reject their surface/boundary
layout before reading; both failures remain in `packed-definition-context/`
and `packed-definition-context-v2/`. They supply no context-share measurement.
No predictor-context impact is inferred from these empty denominators.

A later observer uses the existing runtime packed-input protocol on unchanged
`MM_grammar_wording.xml`, with three word-separated sentences and no validation
question attached to the runtime input. It passes at **2.13 GiB**; see
`native-definition-context/` and `native-definition-context.json`. The store
holds **4 definitions out of 9 rows**. Its two new estimates read one and two
prior observation rows respectively: **0 definitions out of 3 predictor rows
(0%)**. This is a nonempty measurement of this workload. The structured
predictor's default bounded context contains its earlier external observations;
DEF rows remain in the common recency view and available for retrieval. No
general absence of an expectation effect is inferred from this one reading.

The [acceptance map](acceptance-map.md) links every test 22–34 to its permanent
probe or explicit gate. The current Architecture, Language, Lexicon, STM,
Spaces, Params and Training paragraphs now describe DEF rows, inventory
replacement and indexed lookup.
`documentation-contract-before.json` records the stale META/word-row text;
`documentation-contract-after.json` confirms its replacement and, at that
point, byte-identical preservation of Claude's September 28 architectural
sections. The later `current-architectural-doc-sections.json` records the
sections reread during the closing audit: Architecture still matches that
capture; Language includes subsequent September 29 amendments and has a new
hash. Those current sections are retained as read. The STM and
remaining-contract audit files record the additional documentation ports.
The subsequent
documentation-link run passes all **121** then-collected cases, including
`todo.md`; the final sweep also covers the completed receipt documents.

## Eight-seed reconstruction before the identity-owner repair (§16.8 / finding J)

All sixteen predeclared runs complete under the unchanged 8 GiB guard.
After-training mean error is **0.11123813** at HEAD and **0.11675237** in the
candidate. HEAD spans **0.10134130–0.12380376**; the candidate spans
**0.10001503–0.14672459**. The candidate mean lies inside HEAD's eight-seed
range; its seed-5 result lies above that range. Nothing is re-baselined and
no seed is removed. The initial concept-atom prefix fingerprints match for
each corresponding seed; this is not a claim that all parameters match.

Peak worker memory is **6.96 GiB** at HEAD and **7.72 GiB** in the candidate.
The protocol remains two warmup plus five measured training batches and
four validation batches before and after, batch size two. This is a short
native measurement, not a million-sentence campaign. The original per-seed
phases, errors, timings and process outcomes remain under
`head-reconstruction/` and `candidate-reconstruction/`.
The [sixteen-result table](reconstruction-table.md) reports before and after
values for every seed; [the raw comparison](reconstruction-comparison.json)
also retains timings, process outcomes and definition counts.

## XOR table before the identity-owner repair (§16.9)

These gates use the source of the preceding sixteen reconstruction runs and
the first full-900-epoch comparison. The subsequent explicit acceptance files
exposed the identity-owner defect described below, before the full sweep
started. This source's MM_grammar gate is **red**, despite passing the earlier
Stage A and B runs. Its outcome is retained when measuring the repaired source.

| XOR proof | Final candidate |
|---|---|
| Grounded, six exact cases | **6 pass** |
| Curriculum, three cases | **3 pass** |
| XOR_exact CLI crisp output | **pass**; MSE **0.0**, predictions **0 1 1 0** |
| XOR_exact CLI reconstruction | **pass**; **4/4**, MSE 0.0 |
| MM_xor convergence | **pass**; stopping MSE **0.18749104** at epoch 21 |
| MM_xor signal | **pass**; stopping MSE **0.25000519** at epoch 1 |
| MM_grammar signal | **fail**; best MSE **0.20906077**, ending MSE **0.24089603**, all 900 epochs; threshold stays **0.20** |
| XOR_grammar class accuracy | **fail**; class 0 accuracy **0.0** |
| XOR_grammar reconstruction | **fail**; **0/4** |

[Raw final observations](final-xor/summary.json) retain all four MM predictions.
The failing MM_grammar predictions are **.4907833, .5091786, .5091696, .4908112**.
The MM signal gate's loose threshold still permits a near-constant answer;
no stronger learning claim is made for that pass. XOR_grammar's failed
reconstructions include `hello world → "  there"`. Its two failures remain
outside item 7's acceptance condition, as decided; the MM_grammar failure
does not receive that exemption. The historical MM **.21757** failure and
depth-3 campaign **[1, 1, 1, 1]** remain red in the earlier receipts.

## MM_grammar before the identity-owner repair (§16.9 / finding U)

All twenty fresh, unseeded runs complete all **900 epochs** under the unchanged
8 GiB guard. These are the final training-forward errors, at the same observation
point as the protected gate, without its early stop. The separate evaluation
after update 900 is also recorded. All four predictions for every run are in
the [twenty-run table](mm-grammar-table.md), with exact values, targets, process
outcomes and store counts in [the raw comparison](mm-grammar-comparison.json).

| Tree | Runs | Median ending MSE | Mean | Ending below .05 | Range |
|---|---:|---:|---:|---:|---|
| HEAD | 10 | 0.03512730 | 0.13424505 | 5/10 | 2.76068e-7–0.59691989 |
| Candidate | 10 | 0.00125679 | 0.05816619 | 7/10 | 8.09353e-13–0.27024609 |

Peak process-tree memory is **7.64 GiB** at HEAD and **0.56 GiB** in the
candidate. No run is omitted and no guard repeat is needed. The candidate has
lower sample mean and median here; these ten independent runs per tree do not
establish a general learning improvement. They do not replace the failed
MM_grammar gate in the preceding XOR table.


## Closing fixture ports and the shared identity owner

The first explicit selection accounts for **197** cases: **189 passed,
6 failed, 1 skipped**, and the previously measured graph-release memory stop.
The six failures are the three XOR-table failures above and three item 7
fixtures. Their exact outputs remain in `explicit-final/`; no full sweep had
started when they were found.

The fixture ports are recorded in `closing-test-ports.json`: provisioning
must select its three asserted rows amid the new DEF rows; a program expanded
to three leaves must expand its word-identity column too; and an asserted
relation's three native references are distinct from a definition's two
symbol operands and fixed DEF predicate. The related consolidation file was
then probed in full before any port (`provenance-related-before/`). Trust
values, numerical bars and the original resubmission assertion are retained.

That probe exposed a runtime defect as well. In a multi-stage unshared
reading, forced interpretation used stage zero's allocator while the grammar
and clause closing used the terminal allocator. Their unrelated integer IDs
collided: DEF operands **6 and 7** accidentally named the two old user
sentence rows during retention. `provenance-context.json` records the false
links. Its observational case passes the authority-withdrawal checks, while
the imported consolidation cases retain their failures. The dedicated two
identity-owner probes both fail in `definition-owner-before/`.

`_concept_owner()` now returns the installed grammar registry's space, so
words, objects and ended clauses share its identity allocator. The constructor's
pre-registry aligned owner remains stage zero. Both permanent owner probes
pass, and `test_resubmit_replaces_user_rows_only` passes with its original
body unchanged. The first count port assumed six DEF rows in this fixture;
the correct owner's existing capacity admits four. Those two failed draft
expectations remain in `definition-owner-after/`. The final provenance tests
assert three supplied truths and account separately for every DEF row,
without expanding any inventory or changing configuration.

`definition-owner-completion/` reruns all eight affected files: **102 passed,
3 slow skips**. The complete item 7 selection then passes **170/170** in
`item7-after-owner/` before freezing this repaired source.
The consolidated port ledger now has **97** entries, including the eight
closing ports. Four trailing blank lines found by `git diff --check` are
removed with AST equality recorded in `formatting-repair.json`. The reading
staging docstring now describes the full unary rather than its retired
admission-only behavior.

The earlier candidate measurements are retained above as measurements of that
earlier source. All eight candidate reconstruction seeds and ten fresh
900-epoch MM_grammar attempts were measured again, with the unchanged HEAD
controls retained. Their final results follow below. The final XOR table,
explicit gates and one full sweep use this repaired source. No baseline is
selected or revised.

The repaired source's graph-release gate remains **red**, stopped at
**8.30 GiB**. Its one unguarded diagnostic fails at **17.01 GiB** with the
same unavailable `part` predicate in clause recovery, before the graph-release
assertions. The unchanged HEAD control passes at **5.91 GiB**. See
`graph-release-candidate-after-owner/`; these latest observations supplement
the earlier diagnosis without changing the 8 GiB guard.

The final documentation audit also found two Philosophy passages and
FutureWork §5 still describing the retired META proposal as the current
word/object mechanism. Their failing probes and corrected checks are in
`philosophy-definitions-before.json` / `-after.json` and
`future-definitions-before.json` / `-after.json`. Those paragraphs now follow
§17; FutureWork retains its old inbound anchor. Architecture also names the
shared grammar-registry identity owner. These prose corrections do not change
the frozen runtime or measurement drivers.

## Final-source reconstruction (§16.8)

The [final sixteen-attempt table](reconstruction-table-final.md) accounts for
all declared seeds on HEAD and on the repaired source. HEAD completes all
eight under the guard, with mean **0.11123813**, range
**0.10134130–0.12380376**, and peak **6.96 GiB**. Seven candidate attempts
complete; their mean is **0.11602650**. Candidate seed 2 stops during training
at **8.01 GiB** under the unchanged **8 GiB** guard. Its original result
remains **red**.

The required single unguarded seed-2 diagnostic completes at **7.22 GiB**,
with after-training error **0.12183350**. The diagnostic result is separate
from the guarded table. Including its value only for the distribution
comparison gives all eight candidate errors: mean **0.11675237**, median
**0.11700905**, range **0.10001503–0.14672459**. The mean is inside HEAD's
observed range; seed 5 remains above it. All eight numerical values reproduce
the earlier candidate measurement; the new guarded memory failure is also
retained. **Nothing is re-baselined.**

The completed candidate states hold **11 DEF rows** each and **41–104 total
store rows**, including the diagnostic's 11 of 51. This reconstruction
configuration produces no situation estimates, so its context-read denominator
is empty. The separate native predictor-context observation follows below.
[The raw comparison](reconstruction-comparison-final.json) retains each
attempt's phases, counts, process result and the diagnostic's complete data.
The measurement drivers match the retained HEAD controls byte for byte.

## Final-source predictor context: partial observation and reader failure

`native-definition-context-after-owner/` is **red**, at **2.19 GiB**. Before
failing, it records two new estimates reading **0 DEF rows out of 3 prior
rows**, with **4 definitions out of 7 total store rows**. Those are partial
observations, not a completed packed reading. The raw observation is
`native-definition-context-after-owner.json`.

The failure is `query concept has no allocated payload row`, reached from
`program_meaning` when a reference differs from its original leaf. The
adapter requests that reference through the registry's concept-inventory
reader. The isolated diagnostic in `ltm-program-reference-completion/`
confirms a gap in that path: the lexical order lookup returns an earlier
clause (native ID 12 in this probe), its point exists in LTM, and it correctly
has no concept-inventory row; program recovery then raises that same error.
The first diagnostic draft had not attached the existing clause index and
fails at setup in `ltm-program-reference-diagnosis/`; it is retained.
`ltm-program-reference-diagnosis.json` records the corrected mechanism probe.
The diagnostic confirms the gap and does not repair the reader or turn the
failed native observation green. It is a finding for this review; the frozen
source and scheduled full sweep remain unchanged.

## Final-source XOR table (§15.1 / §16.9)

All fourteen cases that are conditions of item 7 pass on the repaired
source. XOR_grammar's two gates remain red and remain outside that condition.
No seed, threshold or configuration is changed for these runs.

| XOR proof | Repaired source |
|---|---|
| Grounded, six exact cases | **6 pass** |
| Curriculum, three cases | **3 pass** |
| XOR_exact CLI crisp output | **pass**; MSE **0.0**, predictions **0 1 1 0** |
| XOR_exact CLI reconstruction | **pass**; **4/4**, MSE 0.0 |
| MM_xor convergence | **pass**; stopping MSE **0.19995962** at epoch 20 |
| MM_xor signal | **pass**; stopping MSE **0.25001639** at epoch 1 |
| MM_grammar signal | **pass**; stopping MSE **0.19939213** at epoch 84 |
| XOR_grammar class accuracy | **fail**; class 0 accuracy **0.0** |
| XOR_grammar reconstruction | **fail**; **0/4** |

[The final raw table](final-xor-after-owner/summary.json) includes all four
MM predictions and the complete failed assertions. The MM signal result
is near constant, as its unchanged .26 threshold permits. MM_grammar's
early-stop result is separate from the ten full-budget attempts below.
The preceding source's MM_grammar failure, the historical **.21757** failure,
and the depth-3 campaign **[1, 1, 1, 1]** remain visible in their receipts.

## Final-source MM_grammar, ten attempts per tree (§16.9)

All ten unseeded attempts per tree remain in the
[final trial table](mm-grammar-table-final.md) and
[raw comparison](mm-grammar-comparison-final.json). HEAD completes **10/10**;
the candidate completes **8/10**. Candidate runs **2 and 5 fail** with
`native conceptual row capacity exhausted before clause admission`, through
`ClauseTaxonomyPlan.validate_rows`. Both fail before the driver's first
50-epoch report, so their exact last epoch and 900-epoch ending errors are
unavailable. They are not replaced or retried.

| Tree | Completed / attempted | Median ending MSE, completed runs | Mean, completed runs | Completed runs ending below .05 | Completed range |
|---|---:|---:|---:|---:|---|
| HEAD | 10/10 | 0.03512730 | 0.13424505 | 5/10 | 2.76068e-7–0.59691989 |
| Candidate | 8/10 | 0.00349623 | 0.02950105 | 7/8 | 4.37896e-6–0.21557093 |

Every completed run reaches all 900 epochs. Statistics for completed runs
do not establish an improvement while two candidate attempts are missing
their ending measurement. Peak process-tree memory is **7.64 GiB** at HEAD
and **0.68 GiB** in the candidate; neither tree needs a memory diagnostic for
these attempts. The completed candidate states contain **4 DEF rows** each,
in **1,020–1,024 total store rows**. Full predictions, targets, post-update
evaluations and process outcomes are retained for every completed run.

## Final explicit gates

The [explicit summary](explicit-after-owner/summary.json) accounts for every
one of **199** selected cases exactly once: **195 passed, 2 failed, 1 skipped,
and 1 memory stop**. All **170 item 7 cases pass**. The two failed assertions
are XOR_grammar's recorded accuracy and reconstruction gates. The memory stop
is the two-epoch graph-release gate; its one unguarded diagnostic remains a
separate failure, as described above.

The complete-forward fullgraph check, both packed-reverse checks and all four
strict parity checks pass. Parity peaks at **7.81 GiB**.
The category codebook passes all three cases at **0.90 GiB**, and the 64-word
trace passes at **0.93 GiB**. No numerical assertion or guard was relaxed.
The thinking-kernel depth-3 case skips for its existing missing
`BASICMODEL_FINEWEB_CHECKPOINT` prerequisite (at least 1,000,000 completed
training sentences). That skip supplies no passing campaign result; the
historical **[1, 1, 1, 1]** failure remains red. No million-sentence training
campaign is started by this pass.

The final measurements, XOR table, explicit gates and full sweep share the
same **704-file** runtime/test/configuration snapshot, with SHA-256 digest
`c573224ffd71917af1932208e298170fd2dd9bde206bf47a391275029095a70a`.
The manifests also verify the supporting inputs. Documentation revisions are
recorded separately. The final documentation-link check passes **126/126**
after completing this receipt and the todo entry; its result is retained
under `documentation-final/`. `git diff --check` also passes.


## Full-sweep findings for review

The [failure diagnoses](full-sweep/diagnoses.json) distinguish the observed
failure from a source-based explanation or an untested probable repair.
The frozen source has not been changed to address findings from this sweep.
These findings keep the candidate unaccepted even though the focused item 7
acceptance selection passes.

* **Checkpoint restoration:** the synthesized-answer, old-clock and FineWeb
  checkpoint gates all reach a knowing code owner whose allocator has not
  been restored. Allocators and cross-space knowing fields are reconstructed
  in the same per-space loop. The owner change needs its restoration
  dependencies resolved. The FineWeb gate now stops before the optimizer
  assertions; the isolated 92/3 layout measurement does not make that full
  checkpoint gate pass.
* **Lexical references:** the owned-answer test's initial equality assertions
  pass, then its order-1 word-object lookup returns `None`. The replacement
  object keeps the word row's native order, while a runtime referent reader
  still filters out order zero. The separate native predictor observation
  also exposes the LTM-clause versus concept-inventory reader mismatch
  described above.
* **Remaining fixture and identity accounting:** two tests count all store
  rows as asserted sentence rows, including new DEF rows. Their later
  assertions remain untested. The attended-field test also fails when the
  identity returned for `lynx` is absent from the published carrier; whether
  that requires only an object-row port or a runtime repair is not proven by
  this sweep. Stage A's earlier pass is retained, not claimed for Stage B.
* **Closing:** the partition-isolation forward raises on a nested relation
  without three resolved references. The identity-candidate space check
  still reaches a completed field with neither one nor three slots.
* **Earlier red checks remain:** the part-width graph count is two instead
  of one; the detached-reverse fixture receives a detached root before its
  student assertions; the normal-policy training path indexes batch member
  one from stale one-row metadata; provisioning retains three episode slots
  where its hard-reset assertion requires none.

No protected assertion is weakened to accommodate these findings. The
complete selector-level comparison with round 2 is in the final sweep
summary; test ports and retirements remain in the separate port ledger.

## Full-sweep receipt and review stop

The single sweep completes **5,050/5,050 unique cases**, with no missing or
duplicate result: **4,714 passed, 13 failed, 322 skipped, one expected
failure**, exit **1**. It takes **9,386.17 seconds** across three bounded
workers. Peak worker memory is **7.24 GiB**, below the unchanged 8 GiB
guard; peak aggregate memory is **16.39 GiB**. There are no sweep memory
stops, timeout continuations or compile-cache retries. The slow explicit
gates and their resource failures are recorded separately above.

On matching selector names, **15 failed → passed**, **3 memory-stopped →
passed**, and **6 passed → failed** compared with round 2. Seven failures
remain failed. All **170 item 7 cases pass** in this sweep as well as in
the explicit selection. The sweep has 51 new selector names and 27 removed
names; the [port ledger](final-test-ports.json) records the ports, renames
and retirements, so the changed test inventory is not hidden in the totals.

Of the sixteen round-2 regressions, fourteen now pass. The two remaining
ones are `test_checkpoint_restores_optimizer_counters_and_rng`, now stopped
by allocator restoration before optimizer assertions, and
`test_more_than_eight_words_reuse_the_attended_field`, now stopped by the
word/object identity mismatch. Their Stage A and optimizer-specific passes
do not substitute for these failed final results.

The complete sweep, final measurements and explicit gates match the
704-file source manifest and digest above. Runtime, tests, configuration and
supporting inputs have not changed during or after the sweep. Prose updates
are recorded separately. The remaining failures are handed to Claude for
review under §16.9; no further code changes or commits are made in this pass.
