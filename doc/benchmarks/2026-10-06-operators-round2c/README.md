# Operators update, round 2c — October 6, 2026

Status: measured; **standing gate missed**. Held for Claude's review; not
accepted, staged, committed or pushed.

The baseline is the intact round-2b candidate manifest
`d99b205afba2e1db0dbd0197d2506fdb100f71cd9cd3f2b0161bccdcea19ddba` (701 files).
The source matched every manifest entry before edits. Prior receipts are
inventoried in `before/intact-receipts.json`; the plan is not edited.

The delivered source is frozen as
[`f712768dd44310461b4723cfe5f8d794bcf2c19ac8c14cabfa731103336b00a5`](delivered-source/source.json)
(702 files). Its one full sweep is green: **4,971 passed, 285 skipped, one
xpassed**, all 5,257 collected tests completed in 207.642 seconds. All thirty
declared trainings then completed, including failures, in 1,101.149 seconds.
No seed, retry, replacement, tuning or budget change selected these outcomes.
The explicitly requested MM bisection and three paired replays are separate
diagnostics.

| Gate | 6.8 landing | Round 2c | Result |
| --- | ---: | ---: | --- |
| Class: 4/4 correct and MSE < .05 | 7/10 | **5/10** | Miss |
| Four reconstructed multisets | 9/10 | **10/10** | Pass |
| MM_xor: best MSE < .20 within 200 epochs | 10/10 | **9/10** | Miss; live |
| Sum control | 10/10 | **10/10** | Pass; all at ¼ |
| Class and reconstruction together | 6/10 | 5/10 | Below landing |

The §20.5 class bands are five at zero and five between zero and ¼. All final
greedy roots use conjunction. Eight runs classify all four rows correctly;
only five also meet the unchanged MSE threshold. Every run recovers all four
multisets. The ten sum MSEs are `0.249999985–0.250000000`; the largest deviation
of a prediction from `.5` is `6.56e-7`, and the largest absolute checkerboard
contrast is `5.96e-8`.

The [per-run tables](measurement-results.md), [complete campaign summary](measurements/summary.json),
[validation](results-validation.json), and [process log](campaign-process.log)
retain every outcome. Sentence-path perception gradient, native-code
displacement, and ownership conflicts are all **zero**. MM is live in all
three paired comparisons, so §5's identical-trajectory exception does not apply.

## Implementation and measurement contract

- Interpretation, retained reconstruction leaves and temporary reading leaves
  share `[form × |a| | meaning × a]`; occurrence coordinates survive unchanged.
  The leaf keeps its signed activation and explicit evidence pair. Resolved
  grammar identities are also used by both reference publications, so a valid
  grammar word cannot lose its evidence to an absent serial row.
- The reader takes one optimizer step per supplied sentence. Compose departures
  average both detached-root losses; narrowing departures use the strict-R-kept
  root. Row weights are recorded, including mixed batches. The first trial's
  reconstruction/expectation step omits the reader; the second applies the
  combined reader gradient. The batch-end answer read reports but does not
  train again. Synthesized meanings detach explicitly too.
- Strictly lower `R` keeps explore; ties keep greedy. Credit is still
  `K·R·p(a_departure)·Δ(R+E+A)`, with the existing registration and reduction.
  Pair search, teachers, owners, budgets, image and containment mechanisms
  remain unchanged. Neither a random seed nor a retry selects gate outcomes.
- Ten sum controls and ten XOR runs use 400 epochs. Each XOR training feeds
  both unchanged class and four-multiset reconstruction gates. Ten MM_xor runs
  retain the 200-epoch maximum and `.20` convergence bar. The class read is the
  final greedy committed root. Bands remain §20.5: below `.05`, within `.02`
  of `.25`, otherwise below/above `.25`. Sum contrast and the quarter-floor
  band are reported separately.

## What the failed runs' costs say

[Concrete cost records for every miss](miss-costs.json) preserve all compose
comparisons in the five failed class runs, plus the last costs rewarding each
operator. The two trial roots are costed before either update. `E=0` in these
gates. The table below shows a late disjunction departure from a greedy
conjunction: `R` ties, so the keep is greedy; the listed positive `ΔA` also
credits conjunction. Rows are zero-based and inputs are saved in the JSON.

| Failed class run | MSE | All rows stay on greedy conjunction from epoch | Cost record (epoch / row) | R, both trials | A greedy → explore | ΔC |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| 1 | .218508970 | 309 | 400 / 2 | .092508137 | .293365449 → .347818106 | +.054452658 |
| 2 | .051585235 | 151 | 400 / 2 | .284397930 | .253275126 → .386114568 | +.132839441 |
| 4 | .071264958 | 5 | 400 / 2 | .168286577 | .140008524 → .968647659 | +.828639150 |
| 7 | .176236955 | 291 | 399 / 1 | .000001236 | .183780402 → .575129867 | +.391349494 |
| 10 | .159531065 | 332 | 400 / 3 | .316027135 | .038670242 → .400249183 | +.361578912 |

The answer term also supports incumbent disjunction on individual earlier
rows. The last such conjunction departures in runs 1, 2, 7 and 10 are,
respectively, epoch/row `307/3`, `144/3`, `289/3`, and `327/2`, with positive
advantages `.148947358`, `.039727032`, `.048798233`, and `.013949364`.
All four have equal `R` and zero `E`; the answer alone opposes that departure.
The full costs, probabilities and `K·R` factors are in the linked records.
The late policy switches in runs 1, 7 and 10 leave only 92, 110 and 69 epochs
with conjunction on all four greedy rows. That timing describes their limited
reader exposure; it is not a complete explanation of every miss: run 4 uses
conjunction from epoch 5 and still ends above `.05`. These are remaining
class/reader convergence misses, not a residual signed-form decomposition
failure or a final disjunction policy. No extra epochs were run to test a
counterfactual.

MM run 1 is the only MM failure: best MSE **.200930029** at epoch 35, final
MSE **.252170831** at epoch 200. It never crosses the strict `.20` bar. This
configuration makes zero owner-step trial comparisons and reaches zero pole
consumers, so no sentence `R/E/A` keep decision can be assigned to its miss.
Its actual changed path is identified below; the full 200-epoch trajectory
and failure are retained in [the run](measurements/mm-01/observations.jsonl).

## Before the thirty trainings

The captured round-2b random state is reused without optimizer updates.
Forced disjunction after `not/and/or/descend` recovers **4/4** multisets; each
selected pair recomposes with **zero relative residual**. The old signed-form
variant recovers **0/4**, with pair residuals `3.85–3.91`. The corrected variant's
owner reconstruction costs exactly equal an independent replay on the actual
6.8 landing source (`42daf96f4`) with the same saved initialization and forced
prefix (`landing-fixture-result.json`). On this untrained fixture they are
`[0.000034862, 0.000641961, 0.0660632, 0.0638630]`.
Those free-decoder likelihood costs are not the pair residual and are not
claimed to be the trained landing's approximately `1e-15` costs. The thirty
runs report the actual trained costs. The positive-pole diagnostic's raw-bank
pair residual uses unscaled prototypes, so only its normal decoder costs and
4/4 readback supply the matched baseline comparison.

The two gate-grammar mechanism tests force a narrowing `not`: leaf evidence
and closing polarity flip, roots and all three cost components are exactly
equal, the keep is greedy, and the reader's Adam counters advance once.
The nonzero-complement test checks the signed meaning separately. Composition
mechanism tests check mean loss and one step on both detached roots. Source
ports preserve the full old/new tests; the earlier stopped sweeps and fixes
are retained under `development/`.

## MM_xor's actual path

The seed-zero bisection is in `mm-first-forward/result.json`. Construction
parameters and RNG states match the landing. Percept-stage trial costs tie on
all four rows; no old explore walk wins. Kept actions and scope match. Retiring
that exploration removes **33 global RNG draws** (one round draw and 32
alternative draws). The later `create_ir_mask` Bernoulli call consequently
changes **12 mask positions**. Restoring the old percept stage restores the
landing output; restoring the RNG state after its greedy walk restores the
round-2 output. Opening scope does not change that output. No pole consumer
is implicated.

All six 200-epoch children for the three paired seeds completed before the
final grammar-only readback correction. They remain the only paired trainings;
none was retried. Their tested source is preserved in
`development/second-freeze/` (`ccb5cf239338…`). The final delta changes tensor
interpretation and the word pipeline/readback, bypassed by MM_xor's constant
`word_brackets=false` route. `paired-mm/source-coverage.json` records zero calls
to those functions at the final seed-zero forward and exact equality with the
paired candidate's first prediction. This is a source-coverage justification,
not a claim that the full paired manifest equals the delivered manifest.
The paired helper's result-reading filename was ported after its six successful
children; the original failure log is retained and the result was assembled
from those saved trajectories, without training again.

| Paired seed | Construction parameters and RNG | First forward | Equal epoch trajectories |
| --- | --- | --- | ---: |
| [0](paired-mm/round2c-0.json) | Equal | Different | 0/200 |
| [1](paired-mm/round2c-1.json) | Equal | Different | 0/200 |
| [2](paired-mm/round2c-2.json) | Equal | Different | 0/200 |

The [paired results and plan](paired-mm/result.json) link to the saved landing
revision and archive hash; all six child process records and logs are in
`paired-mm/`. Neither the MM count nor its live status is excused by the
grammar-only correction after those paired runs.

## Carried audits

All four `_attention_poles` consumers retain their declarations and call census:
`_attention_sentence_payload`, `_pushed_word_slab:poles`, and both per-word and
whole-slab `commit_word_reference_slab` paths. MM_xor bypasses them. The two
grammar fixtures have 14 perceptual content coordinates in a 22-coordinate
form block and **zero** meaning-complement width, hence an identically zero
closing image. The six-row nonzero-complement mechanism fixture, zero presence
and exact restoration checks remain in the focused suite.

| Consumer, calls with a pair available | XOR total | Sum total | MM_xor total |
| --- | ---: | ---: | ---: |
| `_attention_sentence_payload` | 36,060 | 36,060 | 0 |
| `_pushed_word_slab:poles` | 0 | 0 | 0 |
| `commit_word_reference_slab:per_word` | 40,061 | 40,060 | 0 |
| `commit_word_reference_slab:whole_slab` | 4,010 | 4,010 | 0 |

These gates use codes, so the explicit-pole branch is covered by the focused
fixture rather than the campaign. `MM_grammar`, separately from raw `MM_xor`,
uses the word pipeline and reference publications; the two current grammar
mechanism tests verify its sign/evidence and closing-polarity behavior. The
prior [forward-only call profile](../2026-10-06-operators-round2b/consumer-path-check.json)
is preserved as historical path evidence.

Across XOR there are **10,650 narrowing departures, all exact ties**, including
1,801 `not` departures. All **5,350 compose departures have nonzero advantage**:
1,483 to conjunction and 3,867 to disjunction. `R` ties on 15,876 of 16,000
sentence rows, keeps explore once, and keeps greedy strictly on the remaining
123. The answer changes credit against the keep on 1,184 rows, all compose.
For disjunction departures, the answer favors greedy conjunction on 3,374
rows and disjunction on 493; including reconstruction, the total favors
conjunction on 3,409 and disjunction on 458. For conjunction departures the
answer favors explore on 727 rows and greedy disjunction on 756. These counts
are per-row comparisons, not a claim of uniform answer preference at every
stage of learning. [All action totals and per-run records](aggregate-audit.json)
include the corresponding component sums.

All **16,000 sum departures** are narrowing departures and exact ties. Each
XOR and sum run has exactly **400 reader optimizer updates**, one per supplied
sentence batch/epoch; the three active reader Adam counters all end at 400.
XOR weights are `[.5,.5]` on 5,350 compose rows and `[1,0]` on 10,650 narrowing
rows. Sum has `[1,0]` on all 16,000 rows. There is no second batch-end reader
step.

The tenth XOR audit contains 1,600 surrogate records: 518 nonzero compose
credits and 1,082 narrowing ties. Maximum analytic-gradient discrepancy is
`1.49e-8`; maximum finite-difference discrepancy is `7.69e-9`. No SCG step
reaches the decoder. Chooser ranges are saved at every epoch for both walks;
in the tenth run compose changes from `[-.03259,.30083]` to
`[-.08262,.68429]`, and narrowing from `[0,0]` to `[0,4.50753]` through the
shared chooser. All XOR compose ranges move; sum ranges stay fixed with
their zero advantages. [Tenth-run audit](measurements/audit-summary.json).

The bank-wide order-zero containment audit uses positive-net part postings
and reports before/after pair counts and largest violations. Its a/ab/ac fixture
has two nontrivial containment pairs and incomparable ab/ac. The fold monotonicity
table remains catalogue §12.3. No centroid projection or higher-order enforcement
is introduced. The old divide/descend attribution is withdrawn; round-2b's saved
recheck found descend throughout the frozen evaluation records.

The campaign's order-zero gate banks have **zero nontrivial comparable pairs**
and maximum before/after violation **zero**: that part of the bank audit is
vacuous here. The a/ab/ac fixture supplies two genuine containment pairs with
zero violation, plus incomparability of ab/ac. Raw MM_xor has no applicable
mereological bank. The image audits report complement width **0** and image
maximum **0** in every recorded grammar closing.

## Frozen evidence and review boundary

- [Source manifest](delivered-source/source.json) and [archive](delivered-source/source.zip),
  archive SHA-256 `f08a94367ac71f9458149a8ed79a7bb8c7037ff650291ba46d5ea22a79afae39`.
- [Source diff at freeze, relative to round 2b](delivered-source/changes.patch),
  [complete before/after texts of all 12 test ports](delivered-source/test-ports.json),
  [seed-call audit](delivered-source/seed-port-audit.json), and
  [complete changed runtime texts](changed-runtime-texts.json).
- [Frozen hashes of 181 measurement helpers](delivered-source/measurement-helpers.json),
  [input hashes](measurement-inputs.json), and [supplemental report-helper hashes](supplemental-helper-hashes.json).
  Supplemental `.py.txt` helpers only read saved results or package evidence;
  the separately listed landing-fixture helper performs the requested saved-state,
  zero-update preflight. They do not alter the campaign or replace a run.
- [Full sweep report](full-sweep/report.html), [sweep result](full-sweep/result.json),
  [campaign plan](measurements/plan.json), [completion and all process records](measurements/complete.json),
  and [all per-run measurement tables](measurement-results.md).
- [Disjunction diagnostic](disjunction-result.json), [actual landing fixture comparison](landing-fixture-result.json),
  [MM first-forward bisection](mm-first-forward/result.json),
  [paired trajectory comparisons](paired-mm/result.json), and
  [final-source MM coverage](paired-mm/source-coverage.json).
- [Documentation diff from the start of this round](documentation-final.patch),
  [documentation archive](documentation-final.zip), [document hashes](documentation-final.json),
  and [final preservation/source/Git checks](review-state.json).

Three development sweeps stopped on stale expectations or a missed signed
readback path before the final source freeze. Their original sources, tests,
process records and failures remain under `development/first-*`, `second-*`
and `third-*`. The final delivered source received one complete green sweep,
followed by exactly the declared thirty gate trainings. The paired helper's
postprocessing filename failure and its six successful child logs are also
retained; only saved results were re-read. There were no replacement trainings.

GradientFlow, Architecture, catalogue §12.4, FutureWork and todo describe the
candidate and its measured limits. The plan and the user's Philosophy changes
are unchanged. The 3,556 prior-receipt files match their starting hashes.
Source, test and frozen-helper hashes still match the delivered candidate.
The final documentation-only link check passes **297/297**
([log](document-links.log)); `git diff --check` is clean. This check performs
no training and does not repeat the source sweep.
Basicmodel remains at `73cd7b71baeb64135c1e2e841b9bfc321339d8bc`; WikiOracle
remains at `c9670b545ff88ee1b8176aa44f6c192ad84b597c`. No index, commit, push,
or submodule-pointer change was made in this round. Review stops here.
