# Student-state stage 4e — preparation; the free gate fails

Work is on `main` at Alec's request. Production source matches the archived
`1d1ce73b2` candidate; the only subsequent executable change removes dead
teacher-state branches from the diagnostic script. The corrected pilot passes,
but the full free-reading gate fails. **No constituent or identity learning
claim; no identity run.**

## Stage 3, then its own gate diagnostic

The original order run used seed **20261010**, LR **.003**, 64 B4 batches per
lesson: **512 training presentations, 256 paired updates, 16 evaluations**.
It took 116.61 minutes and suppressed one optional batch-end update. Both
original CE gates failed at their recorded final training batch; final free
order was 8/8. The checkpoint was reused, never retrained for this pass.

After 4e failed, one read-only diagnostic evaluated the original eight fields
from that checkpoint and rescored their actual states. It applied **zero
optimizer updates**, with zero scorer parameter change and exact native/rescore
logit equality. The CE bar remains **≤ .01 nats/read on every row**.

| Lesson | Original last-training CE | CE at saved checkpoint after both lessons | Free order |
|---|---:|---:|---:|
| Two words | .032011–.032286 | **.000224–.000227: passes** | 4/4 |
| Four words | .016490–.016762 | **.016268–.016439: fails** | 4/4 |

The original two-word report predates the four-word lesson; it is not the
checkpoint's final two-word confidence. Historical results are retained.
There is no measured plateau: over batches 60–64, mean CE still fell **22.0%**
and **8.8%**, respectively. At the saved checkpoint, four-word target
probabilities are .9769–.9794 on competitive reads, while greedy order is
correct. Its hidden/readout weight gradient norms are **.01662 / .02235**;
first-layer preactivations are **−.3962 to .3822**, with no saturation.
The remaining miss is confidence at the fixed budget, with live gradients;
no new budget, optimizer change or gate relaxation is inferred.
[Original gate](order-gate.json), [saved training trace](order-trace-summary.json),
[read-only diagnostic](order-diagnostic.json), [diagnostic source](diagnose_order.py).

## Student-state targets and corrected pilot

The recorder retains the native chooser's actual windows, masks, references,
context and pressure arguments, including waits and a paired trial's reused
prefix. Annotation labels the ordered top operands; it never installs a
teacher parse. A reversed symmetric pair is therefore labelled wait, while
the unchanged bracket grader compares sets. Native masks stay unchanged;
unreachable targets retain the finite CE floor and are reported explicitly.

The earlier pilot used a rule id as an expanded native action index. Its
fold measurement is **invalid**, preserved in [the original receipt data](pilot-validity.json).
The repair sums all native action probabilities for the annotated operator,
leaving reference choice unsupervised. STOP remains the final action.
The six post-repair focused checks passed, including the native slow contract.
[Focused evidence](focused.json).

The corrected pilot restores `output/stage4e-student-order-20261017/order.ckpt`:
**16 presentations, eight paired updates, no extra step**, 214.11 seconds.
The matched first-batch probe counts trial-states, including reused states in
both trials, rather than distinct states.

| Target | Trial-states | Before CE | After CE | Movement gate |
|---|---:|---:|---:|---|
| Wait | 145 | 5.775928 | 2.680222 | Pass |
| Fold | 4 | 7.941597 | 3.586233 | Pass |

All four fold targets are reachable; three waits remain unreachable. Kept
source order is 14/16. The full run discards pilot learning and restores the
same checkpoint and seed. [Pilot result](corrected-pilot-results.json),
[probe states](corrected-pilot-student-states.json),
[kept programs](corrected-pilot-kept-programs.json).

## Full rerun

The declared run completes **144 training presentations, 72 paired updates,
40 frozen held-out readings**, seed **20261010**, LR **.003**, in **1,905.57 s**.
There are no optional batch-end step attempts or applied extra steps.
The driver's base-commit field is `main`'s then-HEAD, `b09c6c3d4`; its working
production source still matches `1d1ce73b2`.

| Free held-out gate | Result | Gate |
|---|---:|---|
| Exact main brackets | **0/32** | Fail: ≥90% |
| One matching order-1 occurrence per singular NP | **8/48** | Fail: ≥95%; 19 unexpected occurrences |
| Lowered NP with its own order-1 occurrence | **7/48** | Fail: every NP |
| Unlowered descriptions without occurrences | **3/8** | Fail: all eight |

Duplicate writes and address collisions are both **zero**. All **40/40**
readings preserve source order, so this bracket failure persists even when
reading order is correct. The eight bare-description brackets are exact,
but five controls acquire an unwanted occurrence. All 27 written occurrences
are order 1; all 41 reported matching description stamps are 0, without an
input-stamp requirement. Two closes have no fused point, five have zero
closed meaning norm, and the remaining 33 have positive norms (maximum .666667).

Epoch mean wait CE is **4.400679 → 4.544698** and fold CE
**3.099113 → 2.019931**, on changing student states; unreachable trial-targets
are **45 → 52**. Those are observations, not matched-state improvements or
passing free gates. No further 4e change or learning run followed the failure;
stage 3's own gate was diagnosed first, above.
[Analysis](full-analysis.json), [raw graded results](full-results.json),
[kept programs](full-kept-programs.json), [lesson audits](full-lesson-audits.json).

## One final default sweep and review

The final source completes **5,803/5,803 selected cases** in **1,355.71 s**:
**5,466 passed, 335 skipped, one XFAIL, one failed**. Two workers retain the
8 GiB worker / 28 GiB aggregate limits; observed peaks are 4.72 / 6.07 GiB.
The runner emits three additional zero-duration pass reports for one flags
case; counts above count each selected case once. No case was retried.
The existing weekly-record warning remains (16 days old); this is a default
sweep, not a weekly run. [Final sweep](final-sweep.json).

| Carried certificate | Final sweep |
|---|---|
| c: committed question must answer `five` | Fail: unresolved subject reference |
| Exact NOT/OR reconstruction | Pass, unchanged |
| Paired bootstrap zero-cost certificate | Pass; historical .0517 counterexample remains open |

c is a changed outcome relative to the accepted green intersection sweep and
a recurring failure relative to this branch's earlier sweep. No assertion is
relaxed. The [earlier pre-index-repair sweep](sweep.json) remains separate.
Claude's [source review](source-review.json) found no blocking defect. Its
unused diagnostic branches were removed and its count/order clarifications
are recorded above. Claude's [final review](final-review.json) approves the preparation landing,
with no learning claim. All 787 source files match the final sweep; the
six affected documentation-link checks pass ([checks](final-checks.json)).
