# Student-state stage 4e pass — preparation, no learning claim

The accepted intersection/pressure preparation is landed on BasicModel main
at `b09c6c3d4`, with WikiOracle bumped to `c8c2cda`. This candidate adds native
student-state bracket targets and an explicit paired-update budget. The
16-presentation pilot exposed a target-indexing defect; that defect is fixed,
but the corrected learner has **no valid pilot result and no full rerun**.

## Reading-order lesson before 4e

The existing two-word and four-word lessons ran from the accepted stages 1–2
checkpoint: seed **20261010**, learning rate **.003**, **64 B4 batches each**
(**512 training presentations**), and **16 evaluation presentations**. Both
final free checks ran after both lessons. Runtime: **116.61 minutes**.
The existing CE gate remains **≤ .01 nats/read on every row**.

| Lesson | First training CE, row range | Final training CE, row range | Final free source order | Complete stage-3 gate |
|---|---:|---:|---:|---|
| Two words | .346574 | .032011–.032286 | 4/4 | **Fail: CE** |
| Four words | .193684–.198295 | .016490–.016762 | 4/4 | **Fail: CE** |

Each lesson's initial free check was also 4/4; the four-word initial check
followed two-word training. CE fell, with no before/after gain on these free
checks and no held-out constituent result.
[Gate numbers](order-gate.json), [protocol](order-protocol.json),
[all batch CEs](order-progress.json).

Codex initially added an extra stop condition on either order-CE miss. Before
the pilot, that execution condition was corrected to match Alec's request:
report the full order gate, then run the requested pilot once the reader reads
freely in source order. **The stage-3 CE gates remain failed**, including in
pilot metadata; neither threshold nor certificate was relaxed.

## Targets on the student's states

The native chooser's actual windows, masks, references, context and pressure
arguments are captured, including waits. Provenance follows the student's
actual actions, including wrong folds and a paired trial's reused prefix.
The annotation supplies a fold target only when the top operands form its
constituent; otherwise it supplies STOP. It never installs a teacher parse.

The pilot revealed that a grammar rule id is **not** an index into the native
probability vector: reference candidates expand each operator into multiple
actions. The original four fold probes indexed unrelated masked actions.
The corrected CE sums the probability of **all native candidates for the
annotated operator**, leaving reference choice unsupervised by brackets.
STOP remains the last action. No action mask, logit term or kept program is
changed. Unreachable targets remain explicitly reported, not replaced by a
fallback fold. Legacy operator-only lessons retain their contract.

| Contract | Evidence |
|---|---|
| Wrong folds stay in the student's state; waits have a chooser gradient | Fast regression passes |
| Captured probabilities match the native chooser | Native four-row regression passes |
| Rule target maps to every matching expanded-menu action | New regression passes, with gradient to both reference variants |
| Case-free native corpus folds have nonzero target probability after the repair | Native regression passes |

The native integration is in the weekly slow tier under `doc/Testing.md` and
was explicitly run. [Focused results](focused.json): **6 passed after repair**.
No new claim is made about compiled annotated teaching or the carried compiled
retention finding.

## Fixed optimizer budget

The scoped experiment adapter suppresses optional batch-end optimizer steps.
Ordinary training's optimizer policy is unchanged.

| Run | Training presentations | Paired updates planned / applied | Extra step attempts / applied |
|---|---:|---:|---:|
| Stage 3 | 512 | 256 / 256 | 1 / 0 |
| Pilot | 16 | 8 / 8 | 0 / 0 |

## Pilot and conditional rerun

The pilot restored the order-trained checkpoint, with the same seed and first
16 shuffled corpus presentations. Its matched probe contains all actual states
from the first paired batch, observed before its first update. Runtime: **3.40
minutes**. These numbers belong to the **pre-repair pilot**:

| Target | Student states | Before CE | After CE | Interpretation |
|---|---:|---:|---:|---|
| Wait | 145 | 5.775928 | 2.423406 | Fell under the miswired training objective |
| Fold | 4 | 69.077553 | 69.077553 | **Invalid measurement: wrong action indices** |
| Full 144-presentation / 40-held-out rerun | — | — | — | **Not run: no valid passing pilot** |

The first eight kept pilot readings followed source order; the complete pilot
has **13/16** in source order. The three departures are retained in
[pilot validity](pilot-validity.json), with no causal attribution from the
miswired pilot. [Raw pilot result](pilot-results.json),
[student-state probes](pilot-student-states.json),
[lesson audits](pilot-lesson-audits.json), [kept programs](pilot-kept-programs.json).
No extra learning run was added after the target repair. The saved
`output/stage4e-student-order-20261017/order.ckpt` can supply the next corrected
16-presentation pilot; **do not repeat the order warm-up merely for this fix**.
No identity run or 5.5 clock/chain/surface change was made.

## One default sweep and its limit

The single default sweep at `dc58b4ed1` (production `49ea2c8c3`) completed
**5,800/5,800** cases in **483.89 seconds**: **5,463 passed, 334 skipped,
one XFAIL, two failures**. It preceded the pilot's target-index repair.
The repair has focused coverage above; **the final source has not had another
full sweep**. This preserves the one-sweep budget and does not claim a sweep
of untested final code.

| Finding | Accepted intersection sweep | This sweep |
|---|---|---|
| Exact NOT/OR reconstruction, including `R == [0.]*4` | Pass | Pass, unchanged |
| Paired bootstrap zero-cost certificate | Pass | Carried counterexample recurs: approximately .0517 |
| c, ordinary committed-question answer `five` | Pass | Carried `five` mismatch recurs |
| Native student-state integration | Teacher-state predecessor passed | Rewritten contract in slow tier; explicitly passed separately |

Both failures are **changed outcomes relative to the immediately preceding
green sweep**, not newly discovered contracts. The paired .0517327
counterexample already has matched accepted-main/candidate evidence in the
[October 15 receipt](../2026-10-15-item6-orders-paired/README.md); c remains the
carried structure finding. No assertion was relaxed and no sweep was retried.
[Full comparison and failure diagnostics](sweep.json).

Candidate archives: `49ea2c8c3` (student-state implementation), `dc58b4ed1`
(test tier), `e089c523a` (order report and pilot entry). The target-index
repair is included with this receipt. Commits are the archives; raw native
order traces remain in `output/stage4e-student-order-20261017/`.
