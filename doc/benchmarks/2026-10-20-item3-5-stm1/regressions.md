# Independent regressions during the read-contract review

These are historical mechanism diagnostics from native-42–45, not
learning claims. **Current disposition:** the Part-derived bootstrap
repairs the saved `.1` replay and the unchanged random certificate;
both saved premise orders pass in the completed whole §14 file;
IntraSentenceLayer is retired and the whole compiled file completes.
See [current gates](status.md), [bootstrap](bootstrap.md) and the later
[closing-read causes](closing-read-causes.md). The missing type source
in bind plan §6.9 and all four decomposition failures remain required
step-1b work. The sections below preserve the earlier failures and
their contemporaneous diagnosis.

## 1. Why do the paired reconstruction costs differ?

`native-42` reports R values of `0.1` versus `0` in three paired rows.
Its random initialization was not retained. The first instrumented attempt,
`native-43`, also fails; its exact pre-test RNG states are saved in
`regression-diagnostic-43/test_both_training_trials_share_bootstrap_snapshot-initial-rng.pt`.
There was no seed selection or changed assertion. The certificate stopped
on its existing assertion during its second prescribed batch.

The named costs establish that this is **content loss**, not failed byte
decoding, missing output words, or field reconstruction. Both trials emit
two words, both have zero byte error, and both use the same target words.
At the failing batch, one word has meaning mass `0.816497`; the other has
zero. The conjunction's positive lane loses the lone code against zero;
the disjunction retains it.

| Batch row | Greedy decoder / root meaning mass / content error | Explore decoder / root meaning mass / content error |
|---:|---|---|
| 0 | conjunction / 0 / 1 | disjunction / 0.816497 / 0 |
| 1 | conjunction / 0 / 1 | disjunction / 0.816497 / 0 |
| 2 | disjunction / 0.816497 / 0 | conjunction / 0 / 1 |
| 3 | conjunction / 0 / 1 | disjunction / 0.816497 / 0 |

The reconstruction weight is `0.1`, giving R of `0.1` where the relative
content error is one. This is consistent with the incomplete bootstrap
membership already observed in `storage-41`: the Part graph retains both
operands, but the existing LTM1-only read does not credit both words.
That bootstrap read is the second pending architecture question. No loss
term has been removed or relaxed, and `recover_lanes` has not been made to
invent a code erased by conjunction. Fixing the read must still be followed
by replay at this initialization and the unchanged random certificate.

Evidence: [named costs and full values](regression-diagnostic-43/test_both_training_trials_share_bootstrap_snapshot.jsonl),
[failed certificate](native-43/report.html). The diagnostic plugin records
inputs and costs without changing them.

## 2. Why does the forgetting fixture overflow?

| Observation | Result |
|---|---|
| Old physical row capacity | 16 |
| First closing | One open occurrence |
| Subsequent closings | Estimate, observation, then conceived absence |
| Extra-row caller | `ThoughtClosing.close → absence → ThoughtStream.write` |
| Effect of a zero episode budget | Absence still has its ordinary retained row |
| Eight-sentence footprint | Up to 24 rows before the document boundary |
| Fixture port | Capacity 24; absolute high/low thresholds remain 8 and 6 |
| Production forgetting policy | Unchanged |
| Verification | Passes in `native-45`, including the checkpoint and later read/write assertions |

The port retains the original eight presented sentences, interval of 256,
six-row retention assertion, checkpoint restoration and subsequent
read/write checks. It makes room for the additional kind of retained row;
it does not silently suppress conceived absences.

`native-44` reached the retention checks but then failed because the
temporary zero-episode-budget diagnostic setting also disabled the field
expected by the last assertion. That setting has been removed: the final
fixture uses its normal episode budget. The failed attempt remains in
the run table.

Evidence: [append callers](regression-diagnostic-43/test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads.jsonl).

## 3. Do the compiled fixture and profiler use the current mechanism?

| Repair | `native-43` result |
|---|---|
| Remove the retired `symbolicOrder` fixture override | Pass |
| Assert the single symbolic handoff, independently of maximum concept order | Pass |
| Profile that same single handoff and production `Interpret.forward` over the full native object code | Pass |
| Whole compiled-word file | 16 completed: 13 passed, 3 skipped |
| Addressed copular fixture | Pass; reads the new conclusion and asserts the source remains open |

The compiled comparison completed within its existing timeout. Its
worker took 266.9 seconds, including compilation; this is not a timeout
reported as a pass. The entire scoped run completed 19 cases:
14 passed, 2 failed, 3 skipped. The two failures are questions 1 and 2
above. No results from different source snapshots are combined into
a landing result.

## 4. What remained at native-45?

The c/e fixture now inspects the address actually returned by the kept
close and **passes in `native-44`**. Its original empty-search, mint,
question-storage, operation, forcing and keep assertions remain, and it
also checks that the source occurrence retains its open reference.
The forward-order pending-premise case passes. The reverse-order case
still fails its no-unresolved-forward-references assertion; its
[read diagnostics](regression-diagnostic-44/test_forced_ordinary_pending_premise_in_both_orders[True].jsonl)
retain the unresolved rows and Part graph. This remains an open mechanism
failure.

`native-44` completed all 36 selected cases: 34 passed, 2 failed
(reverse-order premise and the temporary zero-budget fixture setting).
`native-45` completed the repaired forgetting case: 1 passed. Neither
scoped run is a full landing sweep.

The two architecture questions, both decomposition lengths in both gain
arms, the paired reconstruction invariant, the remaining native
regressions, whole §14 and one full landing sweep remain required. No
commit, push, main advancement or Step 2 work has occurred.

Those statements describe native-45. The current receipt supersedes
the open status of bootstrap, paired reconstruction and reverse premise
order; it does not erase their failed runs. Publication is still pending
the expression read, both decomposition lengths in both arms and the
full landing sweep.
