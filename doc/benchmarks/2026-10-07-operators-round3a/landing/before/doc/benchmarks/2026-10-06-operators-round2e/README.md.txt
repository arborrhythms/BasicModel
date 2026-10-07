# Operators update, round 2e — October 6, 2026

Status: delivered for Claude's review; **the measured standing gate is met**.
All thirty trainings are complete. Acceptance is pending review; no commit or push.

| Measurement | Round 2e | 6.8 landing | Criterion |
| --- | ---: | ---: | --- |
| Class | **9/10** | 7/10 | Four correct labels and MSE < .05 from the presented reader. |
| Reconstruction | **10/10** | 9/10 | All four multisets recovered. |
| Joint class and reconstruction | **9/10** | 6/10 | Same ten XOR trainings. |
| Sum controls | **10/10 at ¼** | 10/10 | Affine control remains at the quarter floor. |
| MM_xor | **10/10** | 10/10 | Raw count; bisection confirms the §5 RNG-only amendment. |

The final grammar evaluation has **40/40 conjunction roots** and
**40/40 correct labels**. Class is the `XOR_grammar.xml` presented
answer read against the supplied 0/1 targets. It is separate from both the
sum-only affine control and raw `MM_xor.xml`.
[All thirty results](measurement-results.md), [summary](measurements/summary.json),
and [validation](results-validation.json) retain every outcome.

## The one change

Start: frozen 2d manifest
`d7e2ead0a7ce294ed99fccb9646b62630da5c7167e014a960902852d851420bb`,
703 source files matching the working tree before edits. The answer owner
now has two independent readers of the same form:

- The presented reader trains once per sentence on the strict reconstruction
  keep. Its rejected-root weight is exactly zero. Public answers, writing,
  and the class gate use this reader.
- The comparison reader trains once per sentence with 2d's rule: the mean of
  the two root losses on compose departures, the kept root on narrowing
  departures. Only its answer cost supplies A in the chooser's advantage.

Both read detached roots. Comparison parameters are copied before the first
update, without an additional initialization draw. Their independently reduced
losses are summed in the existing output objective; separate parameters give
each reader one update. There is no new optimizer or objective owner.
The reconstruction keep, `Δ(R+E+A)`, walk-first proposal, `K·R_walk·W`
correction, registration, reduction and budgets remain as in 2d.

The [frozen AST contract](runtime-contract-check.json) confirms every executable
function in `SentenceCredit.py` is unchanged, as are the score-function
consumer and public forward head. The core reader body is identical apart
from saving detached predictions. Runtime edits are confined to the comparison
parameter bank and reader integration in `Models.py`; the remaining
`SentenceCredit.py` change is a docstring. Complete old/new runtime texts are
[saved](changed-runtime-texts.json).

The focused tests cover zero rejected-root weight, separate gradients and
optimizer counters, exact reader reductions, detached roots, no RNG use at
clone creation, restoration of tied parameter bindings, checkpoint roundtrip,
comparison-only advantage, and isolation of ordinary presented output.
Pre-freeze development failures and their fixes remain in `development/`.
The final frozen-source sweep passes all executed checks.

## Frozen source and measurement

Candidate manifest: [`3fd07f0cd516da3102867207614e11a9153638d37057bcc82a8c7ff821689106`](delivered-source/source.json),
**705 files**. The [source archive](delivered-source/source.zip) hashes to
`822b49c8b966394af66846a60d850ada329f11175cd7c3570fedf31c8ae3e092`. The receipt includes the [diff from 2d](delivered-source/changes.patch),
[five complete test ports](delivered-source/test-ports.json),
[180 frozen helper hashes](delivered-source/measurement-helpers.json),
and [unchanged seed-call audit](delivered-source/seed-port-audit.json).

The [full sweep](full-sweep/result.json) completes all **5,269** selected
cases: **4,983 passed, 285 skipped, one xpassed**. The initial command
mistakenly set an 8 GiB aggregate memory ceiling and stopped at that guard
after 2,289 completed cases, with no failed test report. Its
[original result](full-sweep/initial-result.json) is retained. Only the 2,980
unfinished cases continued, with three workers, 8 GiB per worker and 24 GiB
aggregate. [Coverage validation](full-sweep/coverage-validation.json) confirms
no completed test repeated and no failure discarded. Combined elapsed time
is 340.443 seconds. The gate campaign starts after that
complete green coverage.

Exactly thirty unseeded gate trainings run on this frozen source: ten
400-epoch sum controls, read before starting the ten shared 400-epoch
class/reconstruction trainings, then ten unchanged MM runs with the 200-epoch
maximum. The class bar and §20.5 bands are unchanged. No retry, replacement,
extra training, tuning or selected seed changes a count. Source and helper
hashes are checked throughout. The separate seed-zero bisection and the
three predeclared paired seeds are diagnostics, not gate runs.

## Reader fit, flips and remaining misses

Both readers' raw MSE is recorded each epoch on the same reconstruction-kept
roots, before either trial update. The final class MSE is the ordinary
presented read after training. These are different measurement times.
The [reader report](reader-trajectories.md) includes all curves, active weight
norms, final gate MSE and per-run comparisons; [raw diagnostics](reader-diagnostics.json)
also retain actual committed disjunction exposure.

| Run | Class | Final MSE | Stable greedy conjunction from epoch | Training epochs from flip | Presented MSE before epoch 400 | Comparison MSE before epoch 400 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | pass | 3.72895048e-11 | 15 | 386 | 5.11457543e-11 | 0.0113968477 |
| 2 | pass | 0.00687594264 | 49 | 352 | 0.00697644614 | 0.0730375051 |
| 3 | pass | 0.0202075404 | 1 | 400 | 0.0203880705 | 0.113823459 |
| 4 | pass | 0.000153554436 | 14 | 387 | 0.000158083451 | 0.0771247074 |
| 5 | pass | 0.000318584207 | 18 | 383 | 0.00032755945 | 0.0452256538 |
| 6 | fail | 0.0801115114 | 140 | 261 | 0.0805749074 | 0.111055657 |
| 7 | pass | 0.00146726917 | 1 | 400 | 0.00149540592 | 0.0223850515 |
| 8 | pass | 0.00652075451 | 1 | 400 | 0.00660469569 | 0.0555371009 |
| 9 | pass | 0.00110106565 | 33 | 368 | 0.00112510286 | 0.0272399988 |
| 10 | pass | 3.25428486e-07 | 1 | 400 | 1.60336924e-07 | 0.0185926482 |

Flip epochs follow 2d: the epoch after the last greedy disjunction, including reversals; a conjunction policy throughout is epoch 1. Exploration can still commit disjunction when R is strictly lower. The reader audit separately counts those actual commitments.

Miss categories under the predeclared [measurement protocol](measurement-protocol.json): **1 late flip, 0 reader plateau, 0 other**. Late flip means stable greedy conjunction after epoch 100. Plateau means correct labels, early conjunction and at most 1% active weight-norm change over the final 50 updates. The report includes both endpoint change and the full tail range. These are descriptions of saved evidence, not causal counterfactuals.

Run 6 is a **late flip** miss: stable conjunction from epoch 140, leaving 261 training epochs including the flip epoch, with all 4/4 labels correct. Its presented MSE before the epoch update falls from 0.106529593 at epoch 350 to 0.0805749074 at epoch 400; its active weight norm grows 24.88% over those 50 updates. The final gate MSE is 0.0801115114.

The about-epoch-100 flip expectation is exceeded in run(s) 6. The remaining runs are stably all-conjunction by epoch 49. The measured standing count passes; the slower flip remains visible for review.

Every failed run’s compose comparison is in [miss-costs.json](miss-costs.json), including contrary comparisons. The table shows the last saved comparison rewarding greedy conjunction in each miss. A is the comparison reader’s relative answer cost, not the presented reader’s raw gate MSE.

| Failed run | Epoch / row | R greedy / explore | E greedy / explore | A greedy / explore | Advantage | Keep |
| --- | --- | --- | --- | --- | ---: | --- |
| 6 | 400 / 3 | 0.335241526 / 0.335241526 | 0 / 0 | 0.218461499 / 0.29193148 | +0.0734699965 | tie:greedy |

## Departure and owner audits

Every sentence records its drawn walk, W, R_walk, K, proposal correction, trial R/E/A components, reconstruction keep and advantage sign. All 32,000 sum/XOR sentence records satisfy `scale = W·R_walk·K`. All narrowing component differences are exactly zero in these zero-meaning-width gates. The focused gradient fixture also checks a nonzero narrowing advantage.

| XOR departure | Count | Nonzero advantage |
| --- | ---: | ---: |
| compose/conjunction | 443 | 443 |
| compose/disjunction | 7631 | 7631 |
| narrowing/and | 1286 | 0 |
| narrowing/descend | 14 | 0 |
| narrowing/gloss | 3974 | 0 |
| narrowing/not | 1341 | 0 |
| narrowing/or | 1311 | 0 |

The maximum analytic discrepancy is **2.23517418e-08**; the maximum finite-difference discrepancy is **1.26672103e-08**. Both walks are covered. [Audit details](audit-details.json) and [all-run aggregate](aggregate-audit.json) retain the per-run counts, logit ranges, transitions and late costs.

Each sum/XOR run takes **400 updates per reader**; every active reader Adam counter ends at 400. All presented rows use `[1,0]` or `[0,1]` according to the keep. The comparison reader uses `[.5,.5]` on compose departures and the keep weights otherwise. There is no additional batch-end answer update. The saved validator checks both sets of weights and both MSE series against every raw trial.

Sentence-path perception gradients, native-code displacement and ownership conflicts are **zero**. No score-function step reaches the decoder. Both readers remain in the answer owner. The [tenth-run ownership audit](measurements/audit-summary.json) and all-run validator retain the evidence.

## Carried mechanisms and MM amendment

The image, magnitude interpretation, negative-pole handoff, inverse search and form containment are unchanged. The 2c [negative-pole fixture](../2026-10-06-operators-round2c/disjunction-result.json) and [landing-cost comparison](../2026-10-06-operators-round2c/landing-fixture-result.json) remain carried: all four multisets recovered, zero pair residual and identical owner costs at the saved initialization. Current focused tests retain the nonzero-complement image and genuine `a/ab/ac` containment fixtures.

The grammar gates retain fourteen content coordinates in a twenty-two-coordinate form block, concept width zero and closing image maximum zero. Their order-zero banks have no nontrivial comparable pairs, so the measured containment check is vacuous; the focused fixture supplies the nontrivial pairs. No centroid placement or higher-order enforcement is added. The fold-monotonicity table remains in [catalogue §12.3](../../specs/2026-09-29-operator-catalogue.md#123-operators-update-round-2b-2026-10-06-rejected-candidate).

| Pole consumer | XOR total | Sum total | MM total |
| --- | ---: | ---: | ---: |
| `_attention_sentence_payload` | 36060 | 36060 | 0 |
| `_pushed_word_slab:poles` | 0 | 0 | 0 |
| `commit_word_reference_slab:per_word` | 40061 | 40060 | 0 |
| `commit_word_reference_slab:whole_slab` | 4010 | 4010 | 0 |

The explicit-pole path is exercised by the focused fixture. These grammar gates use codes; raw MM makes no sentence trial comparison and reaches no pole consumer. MM_grammar’s word-pipeline census is carried from the preserved [2b consumer check](../2026-10-06-operators-round2b/consumer-path-check.json).

The repeated [seed-zero MM bisection](mm-first-forward/result.json) confirms the same path as 2c/2d. Construction, scope and kept actions match the landing; restoring the old percept stage restores its first output, and restoring the post-greedy RNG state restores this candidate’s output. The retired 33 random operations change 12 positions in the later Bernoulli reconstruction mask. Opening scope changes nothing; sentence departure is never called. This establishes §5’s RNG-only amendment on the delivered source.

All six [paired-MM children](paired-mm/result.json) complete the declared 200 epochs. For seeds 0, 1 and 2, construction parameters and RNG match; first forwards and all epoch trajectories differ. The bisection, rather than trajectory difference alone, establishes the amendment. Raw MM outcomes, including any failures, stay visible.

## Evidence and review boundary

- [Campaign completion](measurements/complete.json), [process log](campaign-process.log), and [sum-first read](measurements/sum-read-first.json).
- [MM bisection log](mm-first-forward-process.log) and [paired replay log](paired-mm-process.log), with all child records and landing-source archives.
- [Measurement inputs](measurement-inputs.json), [supplemental helper hashes](supplemental-helper-hashes.json), [documentation diff](documentation-final.patch), [archive](documentation-final.zip), and [document hashes](documentation-final.json). Supplemental `.py.txt` reports only inspect saved evidence or package the review.
- [Final review-state verification](review-state.json): frozen source and helpers match; all **6,523** inventoried files in rounds 2, 2b, 2c, 2d and the credit-loop toy remain unchanged. Claude’s plan and the existing Philosophy edits are untouched.

The final documentation-link check passes **303 tests in 2.47 seconds** (`doc-links-process.log`), and `git diff --check` is clean.

No commit, push, acceptance or later-round work is included. Work stops here for Claude’s review.
