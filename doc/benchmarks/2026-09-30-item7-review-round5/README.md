# Item 7, review round 5 — reviewed runtime and accepted fixture ports

The [accepted landing](landing/README.md) records the three test ports and
passing checks authorized by spec §25 and Alec on September 30. The runtime
is unchanged from the sweep below; no new full sweep is required. This page
retains the original review receipt and its red measurements.

The work began under [spec §22](../../specs/2026-09-16-two-truths-ideas-and-relations.md#22-hand-off-to-codex-claude-2026-09-30-what-to-change-after-review-round-5),
from basicmodel HEAD `1678ee1f`.

The one [source-matched full sweep](full-sweep/summary.json) is complete:
**4,793 passed, 6 failed, 322 skipped, one expected failure**. Every one of
the 5,122 selected cases completed once. All three round-4 failures and all
thirteen round-3 failures now pass. The six new failures, all passing in
round 4, were the fixture assumptions described in [the review findings](full-sweep/findings.md);
the accepted landing resolves them with their original evidence assertions intact.
The sweep took **89.6 minutes**, against the previous run's 154.7 minutes.
There were no resource stops. This is the original red sweep, preserved separately
from the accepted test-port checks.

The [incoming source](incoming-source.json) matches round 4's final tested
runtime exactly. Its three failing assertions are preserved in the
[saved probe](saved-probe.json). The fresh [before run](ak-before/result.json)
reproduces those three failures and all six new mechanism failures: part and
implies disagree in both their identity and predicate point, the registry
allocates inventory rows for them, and a full inventory prevents a part
question.

The [AK repair](ak-repair.patch) makes closing and thought use the same
predicate identity and point. The registry reserves no inventory row for
part or implies. Predicate-kind recognition has one path. The three original
failing assertions remain unchanged. The [first affected-file run](ak-after-before-ports/result.json)
has 78 passes and three obsolete inventory-ownership expectations. After the
[three recorded ports](test-ports.json), with each complete old and new body
retained, the [focused rerun](ak-after-ports/result.json) passes all 54 cases.
These include the three original regressions, six new checks and both ported
files. The selected-relation and item-9b files themselves are byte-identical
to the incoming candidate.

The initial candidate is frozen in [frozen-source.json](frozen-source.json), with
712 source files, and [frozen-inputs.json](frozen-inputs.json). The
[source audit](source-audit.json) records exactly three runtime files, two
ported test files and one new test file changed this round. Both repository
HEADs and indexes are unchanged.

The [validation plan](validation-plan.json) includes every item 7 case, the
thought and reasoning files, every named XOR proof including the slow cases
and fifteen exact round trips, ten full MM_grammar runs, the graph-release
gate, and one final source-matched full sweep. The guard remains 8 GiB.
Any memory stop receives one unguarded diagnostic, kept separate from the
gate. No seed, threshold or model configuration changes.
The [initial coordinator](run_validation.py) started at 18:16 UTC on September 30
and stopped before the sweep when the additional item-7 edge below failed.
Every initial job has finished; thought/reasoning records 315 passes and five
existing skips. The initial attempts and [process ownership](active-validation.json)
remain intact. The separately recorded final campaign supersedes this initial
candidate for the current review.

The [initial item-7 run](item7/result.json) has 197 passes and one failure:
`test_three_slot_relation_closes_unindexed_numerical_operands`. A first
three-slot predicate occurrence still tries to read its row-free identity
through the inventory. Its assertion remains unchanged. Four further
[failing probes](three-slot-before/result.json) reproduce the missing first
occurrence and the unwanted dependence on thought authorization for part
and implies.

The [initial XOR table](xor-comparison.md) names all 49 attempts: 47 pass,
the two XOR_grammar gates fail, and all fifteen exact round trips pass.
Peak memory is 4.66 GiB, with no guard stops. All ten initial MM_grammar runs
finish under the guard; median ending MSE is .0578681566 against HEAD's
.0696157217 ([all runs](mm-grammar-table.md)). These are measurements of the
initial source, retained independently of the final campaign.

After the initial jobs finished, the [repair continuation](finish_three_slot_repair.py)
made the three-slot closing carry the canonical predicate point as the composed
path already does. The failed test is unchanged; all four added probes and the
affected selection pass: [66/66](three-slot-after/result.json). The final source
is [final-source.json](final-source.json), with supporting inputs in
[final-inputs.json](final-inputs.json). All [202 item-7 cases pass](final-item7/result.json)
in 308 seconds. The final [source audit](final-source-audit.json) confirms the
unchanged protected assertions, configurations, HEADs and indexes.

**September 30 runtime correction.** Alec rejected a 3–4 hour test workflow.
The queued serial coordinator was stopped and its follow-up paused; its existing
XOR/MM measurements were allowed to finish without repetition. The [timing audit](validation-runtime.md)
finds a 155-minute previous full sweep with only three workers and 644 fresh
batches; 54 cases consume 95% of execution time. The replacement
[continuation](resume_scheduled_validation.py) adopts those existing measurements,
runs the required affected files through one bounded collection, and retains
the one full sweep. Up to ten workers are admitted according to measured memory
and time. The 8 GiB individual and 24 GiB overall limits, threads, deadlines,
test selection and assertions remain unchanged. The historical replay projected
86 minutes; the completed run measured 89.6 minutes. The observed duration is
42.1% below the prior sweep, with source/selection differences recorded in the
timing audit rather than treated as a controlled performance benchmark.
The expensive native training fixtures remain a separate performance issue.

The [scheduling probes](scheduling-probes/result.json) and a
[real worker integration](scheduling-integration/run/result.json) pass.
[Active final ownership](active-final-validation.json) and
[the continuation log](scheduled-validation-driver.log) identify progress.
The [new harness manifest](scheduling-harness-source.json) preserves the earlier
manifest and records the scheduling revision. A saved
[summary-source probe](summary-source-before.json) also caught a report helper
reading the initial XOR receipt for the final table; the
[corrected check](summary-source-after.json) requires the final receipt.
No model or test source changed for the scheduling correction. The follow-up
is paused at the review hand-off.

**Explicit results, 19:53 UTC.** Every required explicit attempt has completed
on the final source ([coverage audit](explicit-coverage-audit.json),
[full explicit summary](final-explicit-summary.json)):

| Selection | Final-source result |
|---|---|
| Item 7 | 202 passed |
| Thought and reasoning | 315 passed, 5 existing skips; 1,840 seconds |
| Graph release | Passed; peak 7.631 GiB under the unchanged 8 GiB guard |
| Named XOR proofs and repetitions | 44 passed, 5 failed; no resource stops |
| MM_grammar, ten full runs | 10/10 completed; median ending MSE .1066178977 |

The [final XOR table](final-xor-comparison.md) records both XOR_exact CLI gates
passing. XOR_grammar remains at zero accuracy and 0/4 reconstruction. Exact
MM_20M_xor round trips pass 12/15: trials 9, 10 and 11 have exact-match rates
.75, .75 and .25. HEAD's retained result is 14/15; the initial AK source's
15/15 is retained separately above. No trial is retried to obtain a pass.
The [final MM_grammar table](final-mm-grammar-table.md) includes all ten runs
per tree, with HEAD's median .0696157217 and the before-AK candidate's .1342967600.
The five thought/reasoning skips are unchanged: three optional slow arithmetic
cases and two cases requiring the unavailable mature FineWeb checkpoint.

The one full sweep ran from **19:51 to 21:20 UTC**. It collected
5,122 cases: every one of round 4's 5,100 cases, ten new predicate-identity
cases and twelve additional documentation-link cases. The
[coverage audit](full-sweep/coverage-audit.json) confirms no missing, duplicate
or unreported selected cases, unchanged skips and the same expected failure.
The flags test's four unittest subtests are counted under its one selected node.
Peak memory was 5.263 GiB per worker and 19.169 GiB in aggregate. There were no
guard stops, unguarded diagnostics or compiler-cache retries. The runtime and
harness manifests still match; both repository HEADs and indexes are unchanged.

**Six fixture failures at the review stop.** Three parameterizations of
`test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees` and
`test_unrelated_true_episode_cannot_establish_the_next_parent_relation`
stop in `test/index_fixtures.py::terminal_model_index`, before their evidence
assertions. The fixture calls `_existing_row` for every non-empty relation
reference, including the canonical `part` predicate that AK deliberately
stores without an inventory row. `test_normal_what_effect_enters_recency_and_detached_knowing`
has the same assumption in its local `terms` fixture. The sixth failure,
`test_renamed_native_vocabulary_preserves_checked_relation_answers`, requires
all three references to differ between renamed vocabularies, including the
now shared predicate; its subsequent setup also assumes a predicate inventory
row. All six passed in round 4. The [failure analysis](full-sweep/failure-analysis.json)
preserves every traceback and unchanged helper/test body. No repair or port
had been applied when this sweep was handed off; §25 subsequently authorized
the [three test-only ports](landing/test-ports.json). Their protected evidence, ownership and
oracle-poisoning assertions remain unchanged.

HEAD's measurements stand under §21 because its commit has not moved;
the [reuse record](baseline-reuse.json) identifies the exact receipts. The
previous [XOR table](../2026-09-30-item7-review-round4/xor-final-comparison.md)
has both candidate XOR_exact CLI gates passing, both XOR_grammar gates red,
and 14/15 exact round trips on each tree. The previous
[MM_grammar table](../2026-09-30-item7-review-round4/final-mm-grammar-table.md)
records all twenty full runs. The [eight-seed reconstruction campaign](../2026-09-30-item7-review-round4/final-reconstruction-table.md)
remains evidence for round 4: its original deadline stops and guarded
completions are preserved, not rerun or relabelled as this final source.
No reconstruction re-baseline is made. The prior
[depth-three campaign stays red](../2026-09-27-item7-5-landing/README.md).
Item 6.9, NonLayer and ConjunctionLayer remain outside this repair.

The [bounded documentation-link check](documentation-links/group-00/result.json)
covers the final prose, including todo.md. Final audit details and check counts
are in the [hand-off record](review-handoff.json). Nanochat is untouched by this
work. The original stop for Claude's review is superseded by the
[accepted landing](landing/README.md) and Alec's commit/push authorization.
