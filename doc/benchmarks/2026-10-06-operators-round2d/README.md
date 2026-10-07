# Operators update, round 2d — October 6, 2026

Status: delivered for Claude's review; **the standing gate is missed**.
All thirty trainings are complete. Not accepted; no commit or push.

| Measurement | Round 2d | 6.8 landing | Reading |
| --- | ---: | ---: | --- |
| Class | **2/10** | 7/10 | Below the standing gate. |
| Reconstruction | **10/10** | 9/10 | Every run recovers all four multisets. |
| Joint class and reconstruction | **2/10** | 6/10 | Retained comparison. |
| Sum controls | **10/10 at ¼** | 10/10 | MSE range .2499999851–.25. |
| MM_xor | **10/10** | 10/10 | Raw count; RNG-only path excluded under amended §5. |

Every final class root is conjunction and every run gets all four labels
right. Runs 6 and 9 meet MSE < .05; the other eight lie in §20.5's “between”
band, at MSE .0639815362–.191986635. The full sweep is green, and the
sentence-path perception gradients, code displacement and ownership conflicts
are zero. [All thirty results](measurement-results.md),
[machine-readable summary](measurements/summary.json), and
[validation](results-validation.json) retain the failures as measured.

Start: frozen round-2c manifest
`f712768dd44310461b4723cfe5f8d794bcf2c19ac8c14cabfa731103336b00a5`, 702 files,
all matching the working tree before edits. The plan and the user's Philosophy
changes are preserved; 5,276 files in receipts 2, 2b and 2c are inventoried in
`before/intact-receipts.json` and are not changed.

The one runtime change draws the sentence departure by walk first. Among `W`
walks with eligible rounds, draw uniformly; draw uniformly among that walk's
`R_walk` eligible rounds; retain the chooser's uniform draw among `K` eligible
alternatives. Credit is `K·R_walk·W·p(a_dep)·Δ(R+E+A)`. This cancels proposal
probability `1/(W·R_walk·K)` exactly. The keep, reader, costs, owners,
registration, reduction and budgets are unchanged. Both walks record their
analytic and finite-difference checks; each sentence records its walk, W,
R_walk and K. The audit retains total eligible rounds separately.

The full delivered-source sweep precedes exactly thirty unseeded trainings:
ten 400-epoch sum controls, ten shared 400-epoch class/reconstruction runs,
then ten MM_xor runs with the unchanged 200-epoch maximum and `.20` bar.
Class requires all four labels and MSE < .05; reconstruction requires four
multisets. §20.5 bands and all budgets remain unchanged. No retry, replacement
run, seed or tuning selects gate outcomes.

The explicit seed-zero MM first-forward bisection and paired seeds 0, 1 and 2
are diagnostics separate from those thirty trainings. §20 applies §5's
RNG-only amendment only after this candidate's bisection confirms the path.
Raw MM counts and all trajectories remain reported. Frozen source, archives,
diffs, full test-port bodies, helper hashes, process logs and every run are
retained here. Work stops for review before any commit.

## Preflight and frozen source

Candidate manifest:
[`d7e2ead0a7ce294ed99fccb9646b62630da5c7167e014a960902852d851420bb`](delivered-source/source.json),
703 files; [source archive](delivered-source/source.zip),
[diff from 2c](delivered-source/changes.patch),
[five full test ports](delivered-source/test-ports.json),
[179 frozen helper hashes](delivered-source/measurement-helpers.json).
Only `SentenceCredit.py` and the two correction sites in `Models.py` change
runtime behavior. The 48 focused checks pass, including the observer contract.
No seed calls changed. The tests use exact proposal-interval midpoints to
verify equal walk probability with unequal round counts, and enumerate the
corrected estimator to check its expectation and gradient.
The [frozen-source contract check](runtime-contract-check.json) confirms
identical keep, component, expectation-gate, score-function body and reader
functions after excluding docstrings. The only changed runtime functions are
the departure draw and its two correction consumers.
Focused process logs are `development/focused-initial.log` and
`development/focused-observer.log`.

The delivered source's one [full sweep](full-sweep/result.json) passes:
**4,977 passed, 285 skipped, one xpassed**, all **5,263** collected tests
completed in **205.694 seconds**. The gate campaign starts only after that
green result and validates the source/helper hashes throughout.

The [repeated MM bisection](mm-first-forward/result.json) matches the 2c
record: construction, scope and kept actions are equal; old percept trial
costs tie; restoring the old stage restores the landing's first output;
restoring its post-greedy RNG state restores 2d's output. Removing the 33
draws shifts 12 positions in `create_ir_mask`'s Bernoulli mask. Opening scope
does not change the result. The 2d sentence-departure function is never
called on MM_xor. This establishes the RNG-only amendment independently
of the raw convergence count.

All six [paired-MM children](paired-mm/result.json) complete 200 epochs.
For seeds 0, 1 and 2, construction parameters and RNG match the landing;
each first forward and all 200 recorded epoch trajectories differ. These
replays use the exact delivered 2d manifest, without a later runtime change.
The archive of each temporary landing checkout is retained. The repeated
bisection establishes why this trajectory difference is excluded under §5;
the replay's difference alone would not establish the RNG-only path.

Round 2c's [negative-pole fixture](../2026-10-06-operators-round2c/disjunction-result.json)
and [actual landing-cost comparison](../2026-10-06-operators-round2c/landing-fixture-result.json)
are carried forward: 4/4 multisets, zero pair residual and identical owner
costs at the saved initialization. Interpretation, pole handoff, inverse
search and reader code are unchanged in 2d. The current focused tests also
retain the nonzero-meaning fixture, zero-width narrowing ties, one reader
update, image, containment and detached-owner checks.

## Operator transitions and the class misses

Runs 1, 2, 7 and 9 use conjunction throughout. Runs 3, 5 and 10 start with
disjunction on all four rows and use conjunction throughout from epochs
**22, 65 and 106**, respectively. Runs 4, 6 and 8 start with mixed greedy
operators and become all-conjunction from epochs **24, 8 and 22**.
These are the epoch after the last observed greedy disjunction, including
any intermediate reversals. All forty final roots are conjunction. The
walk-first proposal supplies **7,978 compose departures / 16,000 sentences
(49.8625%)**, against 5,350/16,000 in the saved 2c campaign. The unseeded
campaigns are not paired causal estimates.

Faster operator selection does not recover the class gate. Three failed runs
(1, 2 and 7) already use conjunction throughout; all eight misses classify
the labels correctly but do not drive their final committed-root MSE below
.05. Every failed run's compose comparisons, including those opposing
conjunction, are in [miss-costs.json](miss-costs.json). The saved evidence
locates these misses in the final answer read after conjunction is selected;
it does not establish a causal explanation for the remaining reader error.
No extra training or counterfactual is used to explain away a miss.

The table gives the **last saved compose comparison favoring greedy
conjunction** in each failed run. Explore is disjunction. Each row has
`W=2, R_walk=1, K=1`, `E=0`, and equal reconstruction totals, so the keep is
`tie:greedy` and the positive advantage comes entirely from the answer.
The full records retain every component, sign and keep, including contrary
comparisons; these examples are not the final evaluation MSE.

| Failed run | Epoch / row | R, both trials | A greedy conjunction | A explore disjunction | ΔC |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 400 / 3 | .231853560 | .0291812047 | .465261847 | +.436080664 |
| 2 | 399 / 0 | 2.40062885e-8 | .173652753 | .395865023 | +.222212270 |
| 3 | 399 / 3 | .252515644 | .203663409 | .537747264 | +.334083885 |
| 4 | 400 / 3 | .332548708 | .0902947858 | .507427275 | +.417132467 |
| 5 | 400 / 3 | 6.64127668e-8 | .199857607 | .465106219 | +.265248597 |
| 7 | 400 / 3 | .198700428 | .245070860 | .332051724 | +.0869808793 |
| 8 | 400 / 1 | 5.91635808e-6 | .0151393898 | .332713813 | +.317574441 |
| 10 | 400 / 3 | .186470911 | .215559393 | .419238091 | +.203678727 |

Owner-step R includes the reconstruction terms, not just the pair-search
residual. It is distinct from the exact-recomposition preflight above.
In the last twenty epochs, disjunction departures still receive credit
against the conjunction keep on 5 rows in run 2, 5 in run 7 and 10 in run 8;
the other seven runs favor conjunction on every such late comparison.
[The aggregate](aggregate-audit.json) retains all epochs, transitions,
component sums, late comparisons and final selecting costs.

## Departure, reader and ownership audits

Every sentence records the walk, `W`, `R_walk`, `K`, the sampling correction,
both trial components, the reconstruction keep and the advantage's sign.
The [all-run audit](audit-details.json) verifies `scale = W·R_walk·K` on
all **32,000** sum/XOR sentence records. In XOR, compose has `W=2,
R_walk=1, K=1`; narrowing has `W=2`, `R_walk=2` (3 during initial rounds),
and `K=1` or 3. In the sum control only narrowing is eligible, so `W=1`;
its `R_walk` is 2 or 3 and K is 1 or 2. All proposal counts appear in the
[measurement tables](measurement-results.md#walk-proposals).

| Walk / departure action | XOR departures | Nonzero advantages |
| --- | ---: | ---: |
| narrowing / descend | 16 | 0 |
| narrowing / and | 1,324 | 0 |
| narrowing / or | 1,314 | 0 |
| narrowing / not | 1,356 | 0 |
| narrowing / gloss | 4,012 | 0 |
| compose / conjunction | 328 | 328 |
| compose / disjunction | 7,650 | 7,650 |

All **8,022 XOR narrowing** departures and **16,000 sum narrowing** departures
tie exactly in R, E and A. The 7,978 compose departures have nonzero answer
differences; E is zero, and R differs on 967. Across XOR, reconstruction
ties on 15,033 rows, keeps explore on 16 and keeps greedy strictly on 951.
The total credits against the keep on **1,375** rows, all compose and all
answer-driven. On disjunction departures, the total favors greedy conjunction
6,434 times and explore disjunction 1,216 times; on conjunction departures,
it favors explore conjunction 175 times and greedy disjunction 153 times.
These are per-row comparisons across learning, not uniform answer preference.

Every sum and XOR run takes **400 reader updates**, with all active reader
Adam counters ending at 400: one update per four-row sentence batch/epoch.
XOR uses weights `[.5,.5]` on the 7,978 compose rows and `[1,0]` on the
8,022 narrowing rows; sum uses `[1,0]` on all 16,000 rows. No additional
batch-end reader step occurs.

Analytic and finite-difference comparisons cover all 32,000 records. The
maximum discrepancies are **2.23517418e-8** analytically and **1.76165817e-8**
by finite differences; narrowing discrepancies are zero. The focused tests
also give narrowing a nonzero synthetic advantage, so its gradient check
does not depend only on the gates' ties. Exact enumeration checks the
estimator's expectation under unequal round and alternative counts.
The [tenth-run ownership audit](measurements/audit-summary.json) records zero
conflicts and no SCG step reaching the decoder. All sum/XOR runs' perception
gradients on the sentence path and native-code displacements are zero.

Both walks' logit ranges are saved per epoch. In run 10, compose changes
from `[-.004566008,.299900025]` to `[-.379914492,.926691771]`, and narrowing
from `[0,0]` to `[0,4.59908247]` through the shared chooser. Every XOR compose
range moves; sum ranges stay fixed with their zero advantages.

## Carried mechanisms and consumer census

All four declared readers of `_attention_poles` remain unchanged and are
counted when a pair is available:

| Consumer | XOR total | Sum total | MM_xor total |
| --- | ---: | ---: | ---: |
| `_attention_sentence_payload` | 36,060 | 36,060 | 0 |
| `_pushed_word_slab:poles` | 0 | 0 | 0 |
| `commit_word_reference_slab:per_word` | 40,061 | 40,060 | 0 |
| `commit_word_reference_slab:whole_slab` | 4,010 | 4,010 | 0 |

The explicit-pole branch is exercised by the focused fixture; these grammar
gate runs use codes. Raw MM_xor makes no owner-step trial comparison and
reaches no pole consumer. MM_grammar uses the word pipeline and reference
publications, as the preserved
[forward-only census](../2026-10-06-operators-round2b/consumer-path-check.json)
and the unchanged current grammar fixtures show. No MM path is attributed
to a consumer that did not run.

XOR_grammar and MM_grammar retain fourteen content coordinates in a
twenty-two-coordinate form block, **concept width 0**, and **image maximum 0**.
Every measured grammar closing records those zeros. The focused nonzero
complement fixture still checks the six image-table cases, κ=0 and exact
restoration. There is no new loss, optimizer or owner.

The bank-wide first-order containment audit still finds containers through
positive-net part postings and records before/after comparisons. All measured
order-zero gate banks have **zero nontrivial comparable pairs** and maximum
violation **zero**; this gate-bank check is vacuous. The `a/ab/ac` focused
fixture supplies two genuine comparable pairs with zero violation and
incomparable `ab/ac`. The fold-monotonicity table remains
[catalogue §12.3](../../specs/2026-09-29-operator-catalogue.md#123-operators-update-round-2b-2026-10-06-rejected-candidate).
No centroid projection or enforcement above order zero is added.

## Frozen evidence and review boundary

- [Source manifest](delivered-source/source.json) and
  [archive](delivered-source/source.zip), archive SHA-256
  `935d24a91b04ab7e898aeff71de24cb03a2c74601d76d89814a06734678ef3b1`.
- [Diff at freeze](delivered-source/changes.patch),
  [five complete test ports](delivered-source/test-ports.json),
  [seed-call audit](delivered-source/seed-port-audit.json), and
  [complete changed runtime texts](changed-runtime-texts.json).
- [179 frozen helper hashes](delivered-source/measurement-helpers.json),
  [measurement input hashes](measurement-inputs.json), and
  [supplemental report-helper hashes](supplemental-helper-hashes.json).
  Supplemental `.py.txt` helpers only read saved evidence or package the review.
- [Full sweep result](full-sweep/result.json) and
  [driver log](full-sweep-driver.log); [campaign completion](measurements/complete.json)
  and [process log](campaign-process.log), including all eight failed class runs.
- [MM bisection log](mm-first-forward-process.log) and
  [paired replay log](paired-mm-process.log), with every child process record.
- [Final documentation diff](documentation-final.patch),
  [archive](documentation-final.zip), [hashes](documentation-final.json), and
  [review-state verification](review-state.json).

The source and frozen helpers match the sweep, all thirty trainings and the
diagnostics. The 5,276 inventoried files in rounds 2, 2b and 2c remain intact.
The final documentation link check passes **299 tests in 2.50 seconds**
(`doc-links-process.log`), and `git diff --check` is clean.
The plan and Philosophy edits are unchanged. The BasicModel and WikiOracle
HEADs remain `73cd7b71baeb64135c1e2e841b9bfc321339d8bc` and
`c9670b545ff88ee1b8176aa44f6c192ad84b597c`; both indexes are empty. No commit,
push or submodule bump is made. This receipt is the review boundary.
