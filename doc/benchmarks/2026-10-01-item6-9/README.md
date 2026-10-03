# Item 6.9: six repairs and measurement-only variants

This follows Alec and Claude's October 1 instructions: repair part 2, measure
part 3, then stop for review. Part 4 has not started. Nothing is committed.
The candidate remains at plan step 5; no probe variant is incorporated into
its source. The environment is unchanged; rebuilding it belongs to part 4.

## Receipt contract

The accepted item-7 receipt is the baseline: **44/49** in the XOR table,
**12/15** exact round trips, and MM_grammar median ending MSE **.1066**.
This receipt runs only the candidate and one attempt of each named XOR
case, including one MM_20M_xor exact round trip. There is no HEAD run.
The three requested XOR_grammar variants each get ten unseeded 400-epoch
runs; the final MM_grammar receipt gets ten unseeded 900-update runs.
The worker and aggregate guards remain 8 GiB and 24 GiB. No configuration,
seed, threshold, production grammar or optimizer setting is changed.

[Starting source and contract](starting-record.json),
[environment before measurement](environment-before.txt).

## Part 2: repairs

The [six saved failures](saved-six-failures/failures.json) and their raw
logs come from the source-matched September 30 full sweep. The candidate
matched that sweep's frozen source before either repair. The four staging
failures were also reproduced in a [fresh probe](staging-red/run/result.json).
No test body or assertion is changed in this follow-up.

The compiled numeric head now takes its root and final occupied slots from
the body's explicit sentence-state return. It no longer depends on the
eager publication that runs after the compiled call. The answer's gradient
cut remains. [Complete old/new bodies and patch](compiled-head/bodies.json).

The mixing binding now stages three grammar-only tensors: the admitted
object's row, order and full code. Its word-symbol rows, IDs, lexical roles,
lookup bank and published evidence retain their previous ownership. The
grammar uses the object code times activation, and the retained grammar
leaf slab uses that same code. The aligned binding retains its existing
native word-symbol staging. [Complete old/new bodies and patch](grammar-staging/bodies.json).

The faulty write put object rows in `_ar_word_concept_rows` while also
publishing WORD and OBJECT IDs. That made `_word_symbol_concept_ids()`
nonempty in the mixing binding. An unlocated single-word reading then met
`ClauseJournal.finish_clause`'s eternal-concept condition (known reference,
no symbol occurrence time). `ClauseRow.write_clause` consequently reused
that object reference as the sentence row ID, producing the definition
alias. It also made
`commit_word_reference_slab` consider those leaves valid symbols: its
default scalar activation becomes positive evidence, which the closing
journal then carries into the stored truth. Keeping the grammar references
separate removes both unauthorized effects; no pole rule or evidence
assertion is relaxed.

All [12 checks](repaired-six/run/result.json) pass: the six reported
failures, the three full-object-leaf cases, and the three numeric/named-head
and gradient-boundary cases. Both real full-graph cases pass.

The [candidate XOR table](repaired-xor/candidate/table.md) is **33/35**:
both XOR_grammar gates remain red; the single exact round trip passes.
All slow proofs are named in the table. Removing fourteen repeated exact
attempts accounts for the table's smaller denominator; one success is not
a replacement estimate for the accepted historical 12/15 success rate.
The table completed in about 146 seconds. The SPNN smoke case is still
listed here; its requested removal belongs to the separate part-4 change.

The table's grammar diagnostics are calculated from the saved four answers.
The inherited summarizer initially displayed the old embedding accuracy
placeholder; its [saved probe and reporting repair](xor-summary-failure/explanation.json)
replace that label with the actual class count and MSE. Raw observations,
test outcomes and assertions are preserved, and no measurement was rerun.

## Part 3: isolated probe patches

[Claude's original deriv69.py](deriv69.py) is copied byte for byte, with
its source path and hash in the starting record. The observer executes
that probe's `margin`, `simulate`, `enumerate_full` and `align` functions.
It makes no additional model forward pass or clock tick.

The saved patches are [a](probe-patches/a.patch),
[b](probe-patches/b.patch), and [c](probe-patches/c.patch). Full patched
source and complete method bodies accompany them. A fresh worker installs
only the changed methods in its own Python process, using the saved source
for inspection. The candidate's files remain unchanged. Every worker
checks the candidate modules' hashes when installing the patch. The
supervisor checks all 717 candidate source files and the measurement
harness before and after workers run.

- **(a)** Add supplied-answer MSE, in the output's original units, to the
  trial's comparison cost with the entire added term detached. The existing
  expectation/reconstruction gradients and batch-end answer training remain.
- **(b)** As (a), with greedy plus four independently sampled deviations of
  greedy. All five are costed before any update. Each trains once, extending
  the existing per-trial training rule from two to five optimizer steps.
  The cheapest trial wins separately in every row; ties keep the earliest.
  Its observation and prediction record follow that same winning index.
  The extra optimizer steps are part of this probe and must be considered
  when interpreting it as a search-breadth measurement.
- **(c)** Add the same supplied-answer MSE with its graph intact. Only the
  trial's answer read bypasses the understanding cut; its loss can reach the
  object codes and chooser through that trial's own perception pullback.
  Both trial losses train, followed by the existing batch-end answer update.

The [small mechanism checks](probe-selftest/) pass for all three patches:
detached versus uncut answer error, identical parameter versions throughout
the comparison, separate backwards at the original values, and row-local
selection including an intermediate explore winner in (b).

Every full run records all four saved final answers and targets, MSE,
correct-class count, final derivations, and Claude's margin. The margin is
the maximum absolute non-bias coefficient of his pseudoinverse affine
solution for targets [-1, +1, +1, -1], with its exactness flag; it is not a
measured trained head weight. The final margin uses all four selected roots
together, since the same answer map must satisfy all four sentences.
The best available common derivation is enumerated over the captured
leaves at the first epoch, last epoch before its updates, and final
evaluation after training. Trial logs retain every grammar operation choice
and reconstruction deviations. Pair logs assert that every trial was
costed under identical parameter versions before the first update. Probe
(c) also records answer-only parameter gradient norms at the first epoch.

The initial observer attempt exposed a limitation in Claude's hook:
`OperationSelectionLayer.forward` receives only the two newest occupied
slots. Treating that window as the complete stack fails when a third leaf
has been pushed before the older pair reduces. The
[saved failure](observer-window-failure/measurements/a-01/driver.log) stopped
(a) at epoch 148; the supervisor interrupted (b) and (c) at epochs 105 and
168. All three incomplete attempts are preserved. None had a final
measurement, and no completed statistical result was discarded or retried.
The [small failing reproduction](observer-window-failure/depth-three-red.log)
and [passing reproduction](observer-window-failure/depth-three-green.log)
retain the same reconstruction assertions. The corrected observer samples
the full STM at `LanguageSpace.choose_operation`, translates its newest-first
positions to chronological order, and feeds the unchanged simulation and
margin functions. [Old/new observer bodies](observer-window-failure/bodies.json)
and [the adapter](probe_observation.py) record the repair. Candidate source,
variant patches and the settled bar were unchanged.

All thirty variant runs completed. At the decided bar (all four correct,
MSE below .05), the results are:

| Variant | Meets bar | Four correct | Median MSE | MSE range |
|---|---:|---:|---:|---:|
| (a), detached answer comparison | 0/10 | 1/10 | .2382143314 | .2091779086–.3753944102 |
| (b), detached with four explores | 0/10 | 0/10 | .2939883538 | .2520753577–.3954252060 |
| (c), answer gradient into codes/chooser | 9/10 | 9/10 | .0010992477 | .0000024158–.2873804912 |

The [per-run report](variant-results.md) gives every answer, final derivation,
margin, and best available derivation at the first and last epoch, with raw
records linked. Final derivations are readable under Claude's weight ≤5
criterion in 1/10, 1/10 and 10/10 runs respectively. That criterion alone
does not imply the learned answer passes: (c)'s first run has margin
4.02665, yet finishes at 3/4 correct with MSE .2873804912. It is retained.
Answer-only gradient samples in all ten (c) runs reach both the object
codes and the chooser. Neither detached variant meets the bar in any run;
the gradient variant still falls short of ten successes. None is landed.

The final MM receipt initially inherited `MODEL_COMPILE=none` from the
variant jobs instead of the established MM setting, `eager`. The three
active attempts were stopped at their last saved checkpoints of 800, 500
and 300 updates; none reached 900. Their logs, partial results, process
records, [failing dispatch probe](mm-environment-failure/red.log),
[passing probe](mm-environment-failure/green.log), and complete old/new
supervisor scripts are retained in the
[environment incident](mm-environment-failure/explanation.json).
The dispatcher now sets `eager` for MM and records that environment in
each worker directory. No variant result, candidate source, fixture or
statistical failure was changed or discarded. Across both harness issues,
six incomplete attempts are preserved in addition to the thirty completed
variant runs and ten completed final MM runs.

The MM measurement helper is byte-identical to
[the accepted item-7 helper](../2026-09-30-item7-review-round5/measure_mm_grammar.py);
its `eager` environment matches
[the accepted dispatcher](../2026-09-30-item7-review-round5/run_final_mm_grammar.py).
All [ten final MM runs](final-mm-table.md) complete 900 updates. Median
ending MSE is **6.575284761e-11**, versus the accepted **.1066**. The
range is **2.220446049e-16–.25**; runs 5 and 10 finish at .25 and
.1054604203. These are independent unseeded measurements, and the whole
table is retained.

The predeclared campaign is in [the plan](measurements/plan.json), with
[completed progress](measurements/progress.json) and a
[machine-readable summary](measurement-summary.json).

## Verification and review stop

The [final audit](verification.json) matches all 717 candidate source files
across the repaired tests, XOR table and forty completed measurements.
Only `bin/Models.py` changed in this follow-up. The test bodies and
configurations match the starting candidate. All repair body ledgers match
their before/after source, the four protected historical records retain
their hashes, and [pip freeze after measurement](environment-after.txt)
matches the saved environment before measurement exactly.

There were no resource stops in the completed measurements; aggregate
peak memory was **1.897 GiB**, below the unchanged 24 GiB aggregate and
8 GiB worker limits. The six incomplete attempts are accounted for above.
The [documentation check](final-doc-links/run/result.json) covers the six
new or updated receipt/task documents. No full sweep was repeated: the
next full sweep belongs to part 4, after review.

Stop here for Alec and Claude's review. The candidate stays at step 5,
with the answer's gradient cut. The probe variants remain unapplied,
part 4 is unstarted, and nothing is staged or committed.
