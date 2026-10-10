# Part-1 follow-ups H and I (2026-10-10)

H gates the post-read coverage charge by the existing `eligible` mask.
A missing candidate bank still marks the sentence incomplete and contributes
zero reconstruction cost. The unchanged failing packed-sentence test passed
after the fix; the scope and stored-generation group passed 24 tests.

The frozen source has 765 files and differs from landed part 1 only in
`bin/Models.py` ([manifest](source.json), [archive metadata](source-metadata.json)).
Part 2's new corpus and measurement harness are outside this source snapshot.
The [full sweep](full/summary.json) completed 5,745/5,745 tests: **5,451
passed, 292 skipped, one failed, one XFAIL**, in 598.62 seconds. Source matched
all 765 frozen files; peak aggregate memory was 8.61 GiB against the 28 GiB
aggregate / 8 GiB worker bounds. The only failure is the persisting
`test_forced_ordinary_answer_fills_committed_question_without_its_own_episode`
(`five`). There are **no new failing nodes against accepted 6.1**; the
XPASS→XFAIL overlap outcome remains adverse and is explained below. The other
three accepted-baseline failures and part 1's b fluctuation pass this run;
their historical findings remain open ([comparison](full/baseline-comparison.json)).

Five extra skips came from the isolated tree lacking the existing
`output/embeddings/sentence.pt` runtime asset. After exposing that same asset,
the [five embedding checks all passed](embedding-coverage/summary.json) on the
identical source. The fixture hash is [recorded](embedding-coverage/fixture.json).
These are supplemental coverage, not a rewritten full-sweep count. The first
supplement collected before the corrected asset link was in place and retained
five skips; the reported supplement ran with the link verified. No production
source changed, and no full sweep was rerun to choose a better initialization.

## I: two causes behind the same 0.000 overlap

The [paired bisection](bisect/report.json) uses one measurement initialization,
seed 20261010. All 354 initial state tensors and the forward root match bit
for bit across five source versions. No seed was selected to pass the assertion.
The forward artifact is frozen for each inverse ablation; target word rows
are used only to grade emitted leaves.

| Source | Emitted words | Walk complete | Per-position top-3 overlap |
|---|---:|---|---:|
| Accepted 6.1 | 2 | yes | 1.000 |
| First item-6 candidate | 0 | no | 0.000 |
| Before terminal mask | 0 | no | 0.000 |
| First terminal mask | 0 | no | 0.000 |
| Landed part 1 | 2, reversed | yes | 0.000 |

The first candidate's expanded search bank lets the live whole win as a
self-pair. The later progress check rejects that pair, hiding the valid lexical
split. Removing live constituents restores two words and 1.000 overlap on the
same artifact. Restoring only 6.1's eligibility instead permits a single STOP
on the whole and gives .500; restoring its walk and eligibility gives 1.000.
The terminal mask is later than the regression and does not cause its onset.

Part 1's pre-ranking progress filter removes the self-pair obstruction. The
remaining failure is orientation: the live bank supplies a near-copy of word
row 0 at row 64. Pair `(1, 64)` has relative squared recomposition residual
`3.7408028504524755e-15`, versus `6.9542497642080016e-15` for `(0, 1)`.
Both are within the existing exact-fit threshold `9.094947017729282e-13`.
`_bounded_binary_reconstruction` nevertheless takes the raw minimum, overriding
learned decomposition scores; the symmetric pair's bank-order tie then emits
the words in reverse. Removing live constituents again gives 1.000. Thus the
same reported zero changed from an empty inverse to the correct multiset in
the wrong order. This is the cause of I, not a repaired order-learning result.

No English order, fallback pair order, relaxed tolerance, passing seed, or
changed assertion was added. The historical .800 criterion and non-strict
xfail remain. The exact-fit override and the learned orientation lesson stay
open in part 2, along with review findings J–L and the four carried findings.

Reproduce with [the probe](bisect/probe.py), using source trees extracted from
accepted 6.1's `landing/source.tar.gz`, item 6's `development/initial-source`,
`pre-terminal-source`, `terminal-filter-source`, and this receipt's archive:

```
BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin:test .venv/bin/python doc/benchmarks/2026-10-10-item6/followup/bisect/probe.py SOURCE_TREE RESULT_JSON BASELINE_TREE
```

Each stage retains its complete JSON, saved tensors and ablation results in
`bisect/`. The source archives remain in their original receipts. The initial
full-suite attempt in the isolated tree was stopped after discovering that
its 5,384-test collection lacked the documentation runtime assets. It is an
incomplete setup attempt, not the follow-up receipt; the corrected collection
selects the same 5,745 tests as part 1.
