# Item 7 review changes — complete sweep, red candidate

Nothing in this candidate is committed. The ordered work list is
[§12 of the spec](../../specs/2026-09-16-two-truths-ideas-and-relations.md#12-hand-off-to-codex-claude-2026-09-28-what-to-change-in-the-item-7-candidate).
The review changes and September 28 trust/evidence/order schema amendment
are integrated. The fixed measurements and component diagnosis are complete.
The [current gates and full sweep](#current-gates-and-full-sweep) use the
same 694-file source: **4,675 passed, 30 failed, 322 skipped and one expected
failure**, with complete, unique coverage of all 5,028 selected cases.
All 126 item 7 cases pass. The broader sweep exposes remaining fixture and
runtime failures, including 23 previously passing selectors. Their
[complete ledger](closing-sweep/failure-ledger.json) is part of this review;
the candidate remains unaccepted and uncommitted. Work stops for Claude's
review as §12 requests.

The reissued measurements match every preceding numerical result. The
reconstruction and strict parity failures remain visible, with no new
baseline, threshold or passing-seed selection.

Claude's September 28 attention/domain-of-discourse documentation is
preserved. It does not amend this item 7 implementation. The deferred
`NonLayer` and `ConjunctionLayer` defects remain untouched.

## Run history

The [first full-sweep attempt](final/interruption.json) exposed further
property/dictionary fixtures that need porting and was stopped after 2,915
seconds to finish those repairs. Its source remained unchanged and its partial
receipt is retained. The test-only repairs below are now integrated and the
full affected meronomy file passes (34 passed, four existing slow-test skips).
The gates have been reissued. A [second sweep attempt](final-complete/interruption.json)
was stopped after 146 seconds when a caller audit found three more fixtures
still expecting trust to manufacture evidence. Their isolated failing probes
and ports are recorded below. The caller audit and all affected-file repairs
are complete and integrated. The [intermediate gate run](review-gates/summary.json)
recorded an MM runtime failure before its loss measurement: a nested relation
reaches the writer without three references. The [third sweep attempt](review-sweep/interruption.json)
was stopped to diagnose that failure. The preceding MM passes remain recorded;
that attempt is red. A [deterministic repair](mm-admission/repair.json) now passes
all eight new probes and the 47-case affected run. The current reconstruction
measurements, explicit gates and full sweep follow this repair.

## 1. Mechanical vocabulary change

The [preceding full receipt](../2026-09-27-item7/README.md) matches all 687
validated source files before the rename, digest
`e0b217e0340c4dc11e65c1d3c51649af77b8242f7cbac324feaaf39ef65a9f41`.
The separate rename sweep validates digest
`3d455cfcdbc9b1d86535614afea92b08e102ae4fd0b398f3ce93dc75341e8110`.

| Outcome | Before | After |
|---|---:|---:|
| Passed | 4,611 | 4,611 |
| Failed | 84 | 84 |
| Skipped | 320 | 320 |
| Expected failure | 1 | 1 |
| Total | 5,016 | 5,016 |

Every individual case has the same outcome after applying only the recorded
identifier and filename mapping. Coverage is complete and unique. There was
no seed override or pass-seeking rerun. Elapsed time was 5,733.36 seconds,
with three workers, an 8 GiB limit per worker and a 24 GiB aggregate limit.
Peak measured usage was 6.23 GiB per worker and 14.68 GiB aggregate. No
compiler-cache retries occurred.

- [Mapping and source hashes](rename/mapping.json)
- [Mechanical patch](rename/mechanical.patch)
- [Python AST audit](rename/python-mechanical-audit.json)
- [Protected contracts](rename/protected-contracts.json)
- [Full result](rename/full/result.json)
- [Per-case comparison](rename/comparison.json)
- [Exact archived source for the later bisect](rename/source-archive.json)

The initial comparison script mistakenly renamed two historical benchmark
paths inside parametrized documentation test IDs. Its
[initial comparison](rename/comparison-before-path-normalization-fix.json)
is preserved. The comparison now uses the source rename's same historical
path protection; no case outcome was normalized or changed.

## Documentation-link repair

The rename script restored numbered placeholders by prefix, so placeholder
1 also matched 10, 11 and later numbers. Restoration now consumes the whole
placeholder. The earlier interrupted attempt and its five-document repair
remain in the rename directory. Claude subsequently identified twelve
damaged links in `todo.md`, including two damaged anchors; those twelve
targets were restored from the saved originals before any design changes.

The [link audit](rename/link-repair-audit.json) compares 1,218 existing
Markdown targets across all 94 touched files with the saved originals and
the intended filename changes. It also checks 24 numbered placeholders.
All 76 relative links in `todo.md` resolve. The running sweep's validated
source remained unchanged during this repair; `todo.md` was outside its
link-test collection.

After the rename sweep finished, the documentation-link test was extended
to collect `todo.md`. Its [rerun](rename/documentation-with-todo.log) passed
117 cases, before this progress receipt was added.

## 2–6. Design changes and focused checks

The row stores its actual one-slot or three-slot end state and never a
reading program. The temporary numerical journal captures operands before
fusion for expectation training and cached references; it is discarded at
the sentence boundary. Readback uses generation. The three initial
[failing probes](end-state-before.log) and the separate
[pre-fusion probe](pre-fusion-before.log) pass in the
[11-case storage check](end-state-storage.log).

The retrieval index unfolds occupied row values using the existing semantic
priming activation. It stores global inverted postings, with no per-row word
or leaf list. The [initial probes](unfold-index-before.log) and their
[14-case rerun](unfold-index.log) are retained.
Raw forward, evaluation and training use the same sentence driver. Inference
runs one exploit path with no optimizer. Its [initial probes](sentence-boundary-before.log)
and [17-case rerun](sentence-boundary.log) cover that boundary.

Every WholeSpace is now a property basis. The retired XML element is rejected,
and dictionary-form checkpoints warn and drop WholeSpace word and META state.
A migration probe also exposed loss of the a-priori negative examples; loading
missing primitive-property state now preserves both membership and observation
state. The [20-case check](property-only.log) passes. By §11.4 decision,
**61 of 74 shipped configurations change behavior**, including XOR_grammar,
whose file is unchanged. This is not a gate-specific adjustment.

Given truths accept plain English and external trust without a kind
annotation. The shipped texts use the three specified rewordings.
The [focused run](plain-truth.log) has three passes and two explicit skips:
the depth-3 and reasoning provisioning measurements require the existing
mature-checkpoint fixture. No million-sentence training run was started.
The depth-3 assertion remains intact and its historical red measurement remains
visible below; a missing prerequisite is not a pass.

## 7. Fixture integration

The [affected-file run](affected/full/result.json) completed all 1,300 cases:
1,160 passed, 55 failed and 85 skipped. Source stayed fixed throughout.
The [isolated changes](reference-integration/integration.json) were integrated
only after verifying every hash against that run. Formatting has a separate
[AST-equivalence audit](reference-integration/format-audit.json).

Native references now resolve before their numerical grammar operation. The
open reading records the actual selected operands and references; it does not
replace an operand identity after computing its value. Reference candidates
are limited to the existing predictor situation and earlier live constituents.
Known grammar-requested interpretations are read without allocating during
a trial. The winning boundary admits new identities, and the numerical
journal is discarded after both paths train. The
[compiled packed/live-reference check](reference-integration/item7-reference-real-order.log)
passed all nine cases.

Fixture ports use the new concept/category owner and completed fields.
Caller-owned probes capture a reading while it is open; they do not keep a
program in a completed row. The runtime fixes also preserve role observations
before discarding the journal, give each supplied grammar lesson its own
context batch, and prevent the old padding correction from erasing the exact
STM host depth published by the tensor driver.

The [integration run](integration/result.json) completed 386 cases: 382 passed,
two failed and two skipped. Both failures had fixture-specific corrections:
the category-context test no longer provisions arbitrary untrained relations,
and the discard test now supplies and checks both new temporary reference
columns. The [isolated long-fixture check](isolated-fixtures/result.json)
passed the normal grammar lesson, both supervised-generation gradient cases,
and both development/production answer-ownership cases. Its controller fixture
needed an allocated native payload row; its five old idea-decode fixtures
stopped at the property-inventory capacity requirement, already corrected in
the integrated source. The [final fixture rerun](final-fixtures/result.json) passed all 140 cases:
three category fixtures, nineteen controller fixtures and 118 documentation
link cases. All five old idea-decode cases had already passed in the
[preceding rerun](fixture-completion/result.json).

[Protected assertions](protected-contracts.json) are unchanged, including the
original depth-3 assertion. `XOR_grammar.xml` is byte-identical to HEAD;
`MM_grammar.xml` only loses the retired property element as §11.4 requires.
There is no pass-seeking seed choice or threshold change.

A final audit found a remaining [relative-catalog fixture](relative-catalog.json)
whose exact expected set predated `implies`. Its equality predicate remains
exact; the expected catalog now includes the operator required by item 7,
just as the thought catalog includes `true`. The [failing probe](relative-catalog-before.log)
and [integrated rerun](relative-catalog-integrated.log) record one failure
becoming five passes. No protected assertion or runtime implementation changed.

The subsequent full sweep found five failures in the old meronomy analysis
doubles, which supplied only an analysis mode and no primitive properties.
A related isolated probe found eleven more stale expectations in
`test_type_run_spans.py`, including the retired frozen type dictionary and
removed helper APIs. The [33-case failing probe](sweep-fixtures/before/result.json)
has 17 passes and 16 failures. The [isolated rerun](sweep-fixtures/after-r3/result.json)
passes all 33 after supplying real taught primitive memberships and porting
the obsolete type-dictionary checks to the property owner. Every meronomy
assertion is unchanged; the old type-dictionary assertions now check the
current property contract, including trainable membership ownership and a
checkpoint round-trip. None is a protected gate. The [original patch and audit](sweep-fixtures/pending.json)
and [integration record](sweep-fixtures/integration.json) retain the exact
changes, including the later EOF whitespace cleanup.

That audit found 43 further cases requiring WholeSpace's retired word/operator
dictionary, its SHIFT gate, or losses/readers of those rows. The
[case-by-case disposition](sweep-fixtures/retired-cases.json) records their
removal under §11.4 and lists the active chooser, concept-binding, priming and
occupancy coverage; no compatibility method or skip is introduced. These are
explicit test retirements, not passing measurements. Protected gates are
unchanged. The 55-case [deleted-API probe](sweep-fixtures/deleted-api-before/result.json)
had 45 failures and ten passes. Four failures initially stopped at a grammar
fixture omitted from the isolated copy; the
[corrected-input probe](sweep-fixtures/grammar-api-before/result.json) confirms
all four reach the deleted dictionary APIs once that unchanged input is present.

The semantic-category fixture is ported to explicit consequence/signature
inputs with every original assertion intact; its
[rerun](sweep-fixtures/category-after-input-restored/result.json) passes both
cases. Native identity-counter checks now exercise ConceptAllocator and the
structural checkpoint: the [four failing old-owner cases](sweep-fixtures/allocator-before/result.json)
become [four passes](sweep-fixtures/allocator-after/result.json), retaining the
exact positive/monotonic/resume values. The
[retained-file run](sweep-fixtures/retained-after/result.json) has 283 passes,
three skips and the single missing-input failure corrected above. The
[full meronomy file](sweep-fixtures/meronomy-file-after/result.json) accounts
for all 38 selected cases: 34 passed and four existing slow-test skips, in
958.32 seconds with 5.22 GiB peak usage. The
[integrated documentation check](sweep-fixtures/integrated-doc-links.log)
passes all 118 cases. These integrated fixture changes
touch tests only; runtime source is unchanged.

## September 28 row-schema amendment

Scalar source trust remains independent of the ended clause's `(c⁺, c⁻)`.
Appending without identification evidence records neither, even if a source
supplies nonzero trust. Updating or withdrawing trust preserves both poles;
checkpoint migration cannot reconstruct evidence from trust. The flat
luminosity view expands both poles without relabelling them as source trust.
Forgetting's value term remains specified in terms of scalar `|trust|`;
item 7 does not implement the later forgetting algorithm.

The row also stores its order, retained through loading, compaction, completed
fields and recency. Retrieval unfolding carries the stamp into bounded sigma
inverse descent. It indexes named abstractions as well as recovered witnesses;
there is still no per-row word list, leaf list or activation snapshot. A legacy
row without a stamp warns and remains unknown (`-1`), rather than inventing an
order. The serial reference API accepts both input poles explicitly; a
scalar-only serial input can supply only its expressed pole. Neither path
uses source authority to manufacture identification.

The [initial probes](schema-amendment/item7-schema-before.log) failed before
the correction. The [integration patch](schema-amendment/integration.patch)
and [file/fixture audit](schema-amendment/integration.json) record the amendment.
The new schema probes pass, including a native sentence carrying both poles
into its row, checkpoint restoration, compaction, authority withdrawal and
abstract unfolding. The first affected run had 261 passes and 14 failures;
the failures exposed stale fixtures and missing index order handling. Its
[source-matched result](schema-amendment/affected-before/result.json) is kept.
The 45-case provenance rerun passed. The integrated
[affected-file run](schema-amendment/affected/result.json) completed 411 cases:
410 passed and one failed. The remaining lexical-reference fixture supplied
the old three-part open-reading tuple; its isolated nine-case rerun passes
with the original assertions intact. It was integrated after every frozen
measurement completed. The [final fixture/link run](schema-amendment/fixture-final/result.json)
passes all 127 cases. Fixtures now supply identification evidence separately
from trust.

A late [caller audit](sweep-fixtures/evidence-caller-audit.json) found three
files omitted from that affected selection. Their
[25-case probe](sweep-fixtures/evidence-fixtures-before/result.json) has
20 passes and five failures: they still expected supplied trust to manufacture
the pair. The [fixture patch](sweep-fixtures/evidence-fixtures.json) supplies
both poles explicitly for repeated claims. Registration now holds the pair
fixed while varying trust and checks both columns. The original registration,
row-identity and contradictory-evidence expectations remain; the withdrawn
trust-to-pair expectation is replaced. An unused legacy collapse callback in
the storage fixture is now an assertion trap. The
[48-case affected rerun](sweep-fixtures/evidence-fixtures-after/result.json)
passes, and these four fixture files are integrated with matching hashes.
The second sweep attempt is retained as an
interrupted receipt, with its source unchanged.

The [remaining 28 caller files](sweep-fixtures/evidence-callers/result.json)
completed all 267 cases: 259 passed, seven failed and one skipped. Five files
still lacked the numerical index owner or explicit initial evidence. Their
[fixture ports](sweep-fixtures/reader-fixtures.json) retain all **150 original
assertions** and the [final affected rerun](sweep-fixtures/reader-fixtures-final/result.json)
passes all 55 cases. Retrieval uses forced terminal generation through the
production numeric index, actual priming and native order stamps; this is
mechanism coverage, not learned parsing. The two-unit work-budget fixture
reads an already indexed and explicitly primed candidate set with the same
operation and record charges. It supplies no generation work in that query.
Relation fixtures now supply their initial evidence independently of trust.
Three unused retired-gate helpers and stale rebaking comments are removed;
no case is retired in this repair. These six files are integrated with hashes
matching the affected run. Runtime executable source remains unchanged.

The deferred `NonLayer` and `ConjunctionLayer` class bodies are
[AST-identical to HEAD](schema-amendment/deferred-operators-unchanged.json).
Their two defects remain assigned to the later item.

## Unindexed relative operands

The latest MM gate exposed an admission defect before measuring its loss.
The unaligned embedding path has numerical NP fields but no native word
addresses. Selecting `part` therefore supplied `-1` operands to the strict
relation writer. An unselected diagnostic happened to select only absolute
operations; its result is retained, and it was not counted as an MM gate pass.
The [forced-reading probes](sweep-fixtures/unindexed-before/result.json)
reproduce eight failures, including the same stack in the real MM forward.

The [repair](mm-admission/runtime.patch) closes an unindexed operand as an
unasserted one-slot S before its relative parent refers to it. It uses the
existing numerical field and shared row allocator, without inventing a
concept identity or snapping the vector. Indexed operands are unchanged.
The writer still requires all three references, and the source trust remains
on the outer field. All eight new cases and the full
[47-case affected run](sweep-fixtures/unindexed-after/result.json) pass.
This is a runtime change; the earlier measurement-source equivalence proof
ends before it. The fixed measurements are reissued below.

## Reissued local predictor measurements (§7.15 and §7.21)

The [fixed local protocol](measure_identity.py) retains seed 42, identical
initial weights, fixed corpus order and 240 updates per condition. Its only
port replaces completed-program replay with numerical frames supplied while
the fixture reading is open. These are explicitly forced mechanism readings,
not evidence of learned English parsing or identity selection.

| Condition | Role MSE before | Role MSE after |
|---|---:|---:|
| References off | 0.1356097888 | 0.0002154751 |
| References on | 0.1829218827 | 0.0000932359 |

With references on, the wrong held identity raises predicate-content MSE
from 0.0000629995 to 0.2507680357. The mixed-kind measurement (four ideas,
two relations) changes BCE from 0.6931471825 to 0.0022676998 and accuracy
from 2/3 to 1. These measurements are observations, not tuned gates.
The [complete result and source manifest](identity.json),
[driver migration](identity-driver-migration.json) and
[bounded process record](identity-process.json) are retained.

## Fixed full-model measurements and component diagnosis

The [reissued serial measurement](closing-measurements/serial-baseline.json) retains
the reviewed seed 42, four validation batches, seven training batches (two
warmup, five measured), and four validation batches afterwards.

| Phase | Reconstruction mean |
|---|---:|
| Before training | 0.12243161350488663 |
| Warmed training | 0.11974661648273469 |
| After training | 0.12279699929058552 |

The validation regression remains; the baseline has not been changed.
The [component protocol](bisect/protocol.json) records four independent
counterfactuals in fresh processes. The diagnostic source lives only under
this receipt, with each exact override archived alongside its result.
The native `interpret` counterfactual improves from 0.11755186505615711
before training to 0.10100938938558102 afterwards. Restoring the old context
updater leaves every measured reconstruction and answer value unchanged;
the situation, expectation and expectation-loss weights are all zero. The
identity-gradient removal and journal-gradient detachment also leave every
numeric batch result unchanged. The [four-way comparison](bisect/comparison.json)
records the exact scopes; the baseline has no reference-annotated grammar rules.

The [association-only diagnostic](interpret-diagnostic/comparison.json)
narrows the regression to `InterpretLayer.forward` calling `bind_meta`.
The sigma association admits a weighted META row before the object. This
changes which initialized dictionary atom the object receives: the first
object (identity 5) uses row 5 instead of row 4 in this workload. Restoring
only the association construction reproduces the old initial and training
measurements and improves to 0.10119657963514328 after training. The smaller
gap from the full `interpret` counterfactual belongs to the remaining
`interpret` changes and is not separately attributed. No grammar rule in this baseline requests an explicit
reference order. The association change is part of the required concept
index; no old runtime branch is retained, no seed is changed, and the
baseline remains unchanged.

The diagnostic uses Python recurrence bodies to permit observation. Both its
current and HEAD controls reproduce **every numeric batch result** of their
native counterparts exactly. The three override sources, first-object
admissions, all phases and process records are retained beside the comparison.

Packed and single byte-cost means remain 0.38411848741816357 and
0.3841184933844488. Their parameters, dictionaries, roots, references and
recovered ideas are identical. The fourth sentence costs
0.0005294966977089643 versus 0.0005295205628499389, a strict parity failure.
The [byte scorer diagnostic](byte-diagnostic/comparison.json) reproduces both
complete native reports exactly and identifies the first difference at
float32 softmax: identical logits occupy different padded columns. For the
last `1`, the largest assignment difference is 1.1920928955078125e-7.
Shifting only the candidate columns by six makes logits, assignments, byte
probabilities and cost exact. This is padding-position-dependent softmax
roundoff, not a different recovered idea or derivation. The tolerance and
scoring implementation are unchanged; the strict result stays red.

The [current comparison](closing-measurements/comparison.json) follows the
unindexed-operand closing repair. Every numeric batch result and both complete
parity reports match the preceding measurements exactly; timing and source
metadata are excluded from that comparison. The
[exact source audit](closing-measurements/review-source-equivalence.json)
links these reruns to all 694 current validated files. The
[source archive](closing-measurements/source-archive.json) also includes both
supporting fixture inputs. No new baseline or tolerance is introduced.

The preceding measurements predated the last fixture integration. The
[first source comparison](measurements/final-source-equivalence.json) lists
the only four differences: the lexical fixture's current input tuple,
an EOF blank line, the relative catalog's expected operator, and a luminosity docstring correction. Runtime executable
AST, all data/configuration files and every other source file are unchanged.
The [current source comparison](measurements/review-source-equivalence.json)
also covers the subsequent property/dictionary fixture repairs and corrects
the stale `TruthLayer` docstring about the LTM view's independent poles.
That comparison predates the unindexed-operand closing repair; it remains
valid for the preceding fixture-only changes. The new runtime source has
its own measurement receipt and exact comparison. The reissued gates and
current sweep use the same exact final source.

## Explicit gates

The [first explicit attempt](explicit-before-catalog-and-isolation/result.json)
passed the compiled-forward case and both packed-reconstruction cases, then
the two-epoch word-store case exceeded its 8 GiB memory guard (8.34 GiB
sampled peak). The bounded runner correctly stopped that group, so the
learning gates had not yet run. The final [gate driver](run_explicit.py)
isolates each gate file in a fresh bounded group and continues the others
after a resource failure. The worker limit remains 8 GiB, the overall time
limit remains 5,400 seconds, and no test, dataset or seed is changed by this
dispatch adjustment. A resource-limited case is reported as such, never as
a pass or ordinary test completion.

The [preceding gate receipt](explicit-final/result.json) and
[outcome summary](explicit-final/summary.json) contain eight selected cases:
four pass, two fail, one skips for its prerequisite, and one is resource
limited (seven ordinary pytest completions).

| Gate | Preceding result |
|---|---|
| Complete forward: one compiled graph across runtime lengths | Pass |
| Packed pre-fold operands and per-sentence reconstruction | Both pass |
| Two-epoch word-store graph release | Memory failure, 8.42 GiB sampled against 8 GiB limit |
| XOR_grammar class accuracy | 0.0 against at least 0.5; fail |
| XOR_grammar recovery | 0/4 against at least 50%; fail |
| MM grammar signal | Pass: best MSE below 0.20 within at most 900 updates |
| Depth-3 campaign | Mature checkpoint unavailable; skipped, historical `[1, 1, 1, 1]` stays red |

The unchanged successful MM test does not emit the exact best loss or stopping
epoch. The previous [completed gate receipt](explicit/summary.json) has the
same case outcomes, with 8.64 GiB sampled at the memory failure. This reissue
follows the final fixture integration so the gates and full sweep share exact
source; it does not select a seed or a favorable value. Each MM pass is one
unselected initialization, not a general learned-utility claim. All protected
assertions and thresholds remain unchanged.

Those preceding gates used the [693-file source archive](explicit-final/source-archive.json),
digest `b920eff6e0ef15935ccf6d69744301f6697b53915b67f2afdf616f62add6e80c`.
The subsequent [gate attempt](review-gates/summary.json) recorded three passes,
three failures, one prerequisite skip and one memory-limited case (8.07 GiB).
Its third failure was the MM admission error repaired above; it is retained
alongside the earlier MM passes.

The current measurements use the [694-file archive](closing-measurements/source-archive.json),
digest `64a48544242ffcf014f819e6ddcc5927f11c5f6edc9d8f3422c29a7659c73bc3`.
The [current audit](review-source-audit.json) verifies all protected files and
both deferred operator classes. The new gates and full sweep use that exact
validated source and the two supporting fixture inputs.

## Current gates and full sweep

The [current explicit receipt](closing-gates/summary.json) selects eight cases:
four pass, two fail, one skips for its prerequisite and one exceeds its
memory guard. Source matches the reissued measurements exactly.

| Gate | Current result |
|---|---|
| Complete forward across runtime lengths | Pass |
| Packed pre-fold operands and per-sentence reconstruction | Both pass |
| Two-epoch graph release | Resource failure: 8.47 GiB sampled against the 8 GiB guard |
| XOR_grammar class accuracy | 0.0 against ≥0.5; fail |
| XOR_grammar input recovery | 0/4 against ≥50%; fail |
| MM grammar signal | Pass: best MSE <0.20 within the original 900-update budget |
| Depth-3 campaign | Prerequisite skip; historical `[1, 1, 1, 1]` remains red |

The unchanged MM gate does not print its exact best loss or stopping epoch.
This is one unselected initialization on the repaired source, not a retry to
obtain a passing seed. The deterministic admission probes establish the
runtime fix independently of this learning result.

The [full sweep](closing-sweep/full/result.json) and
[audited summary](closing-sweep/summary.json) account for every selected case
exactly once:

| Outcome | Cases |
|---|---:|
| Passed | 4,675 |
| Failed | 30 |
| Skipped | 322 |
| Expected failure | 1 |
| Total | 5,028 |

Elapsed time was **8,715.64 seconds**. The run kept three workers, an 8 GiB
per-worker guard and a 24 GiB aggregate guard; measured peaks were 7.24 GiB
and 16.31 GiB respectively. No resource limit was exceeded, and no
compiler-cache retry occurred. Runtime, configuration and test files stayed
fixed and match both the measurements and explicit gates, including both supporting fixture
inputs. The source audit also confirms every protected contract and both
deferred operator classes remain unchanged.

All **126 `test_item7_*` cases** pass, as do all 118 documentation-link cases
in the full sweep. Those mechanism results do not erase the broader
integration failures. The [failure ledger](closing-sweep/failure-ledger.json)
contains every failing selector, its current trace and its preceding report.

| Remaining failures | Cases | Observed result |
|---|---:|---|
| Retired WholeSpace API expectations | 15 | Fixtures still require syntactic/lexicon layers, analysis state, references, order or the old regularizer |
| Compilation | 3 | Recompilation limit of eight; two graphs instead of one; missing gradient anchor |
| Gradient assertions | 2 | No positive shared-operator output norm; detached root has no gradient function |
| Batch and clause admission | 2 | Staged batch index outside a one-row tensor; completed field has neither one nor three slots |
| Answer and MM output | 2 | Synthesis outputs stay zero; MM continuous-symbol distance is zero |
| Provisioning counts and reset | 3 | Two truth-count checks remain zero; What episode retains three slots |
| Action, depth and history assertions | 3 | Ended action record differs; shared operation leaves depth two; projected concept already has a row |

Of the 30 failures, **23 selectors previously passed, six previously failed,
and one ported selector has no same-name predecessor**. A previously failing
case need not have the same cause now; both traces are retained. Seventy-six
same-name failures become passes, while two become the explicitly authorized
mature-checkpoint prerequisite skips. The raw selector inventories contain
98 additions and 86 removals, including renames/replacements and the separately
documented 43 retired WholeSpace cases. These counts are not additional passes.

The unchanged MM learning gate passes, but the full sweep's separate
`test_forward_keeps_continuous_symbols` measures **0.0 against >1e-6** and
fails. Its result remains alongside the learning measurement. No failed case
was edited, waived or rerun to obtain a favorable outcome after this sweep.

The sweep records a new failure in
`test_native_interleave_supplies_context_then_reads_the_same_sentences[True]`:
`stage_cs_lang` exceeds Dynamo's existing recompilation limit of eight. The
[worker log](closing-sweep/full/worker-030.log) reports the last guard as
`current_stm[0]` requiring a different `requires_grad` value. The evaluation
variant preceding it in the same worker passes. The cause beyond that trace
is not yet established; no compiler limit, assertion or seed was changed,
and the failed outcome is retained.

`test_normal_batch_logs_named_shared_operator_gradients` also fails: none
of the named shared operators has positive `output_norm` in its report
([worker log](closing-sweep/full/worker-060.log)). This case passed in the
mechanical-rename receipt, so it is a changed outcome. Its existing seed
942 and all assertions are unchanged. The test is not waived by todo's
deferred report-assertion follow-up, and its current failure remains red.

## Measurements that remain visible

The [rejected candidate's receipt](../2026-09-27-item7/README.md) is retained
as the historical comparison. Serial reconstruction was
0.1224316135 before, 0.1191720113 during and 0.1237179693 after training.
Packed/single byte costs were 0.3841184874 / 0.3841184934, with a strict
parity failure of approximately 2.4e-8 on one sentence. No baseline has
been changed.

XOR_grammar class accuracy was 0.0 against at least 0.5, and recovery was
0/4 against at least 50%. The MM gate failed before reaching its current
measurement; the earlier measurement was 0.21757 after 900 updates against
less than 0.20. The earlier depth-3 campaign remained red at `[1, 1, 1, 1]`.
The current explicit outcomes and all 30 full-sweep failures are above.
All three interrupted sweeps remain partial receipts. The preceding MM
admission failure and its deterministic repair are recorded above. The
fixed measurements have been reissued with every numerical result unchanged.
The completed source-matched sweep is red, and the candidate stops here for
review without a commit or submodule bump.

Final documentation-link rerun: **118 passed**
([log](review-doc-links.log)).
