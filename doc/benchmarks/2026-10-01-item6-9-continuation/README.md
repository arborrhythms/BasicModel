# Item 6.9 continuation — October 1, 2026

Stopped for review under Alec's October 1 continuation and plan §9. The
implementation, gate measurements and single full sweep are recorded below;
the candidate retains the red results and unresolved ports. Nothing is
committed. The six reviewed repairs are the starting candidate, verified by
`starting-record.json`; no HEAD measurements are being run. The accepted item 7
baseline remains 44/49 in the XOR table, 12/15 exact round trips, and MM_grammar
median ending MSE .1066. Those historical results are not replaced by the new
single-run receipt rule.

The environment is unchanged (`environment-before.txt`). Every model worker
keeps its 8 GiB guard, with a 24 GiB aggregate dispatch budget. No initialization
is selected. Configuration files, training budgets and acceptance thresholds
are unchanged. Part 4 of the hand-off is deferred until review.

## Step 5a

Each supplied answer contributes its live error to both sentence trials. Both
costs are computed before either update; each backward uses its own perception
pullback. The separate batch-end answer update remains. Without a supplied
answer, the readout boundary retains its state cut. The native answer generation
path also accepts an explicit live state for this trial loss.

The original bodies are under `step5a/`; the prepared-boundary extension has a
separate `step5a-prepared/` snapshot. `step5a-red/` saved the failing whole-path
probe before the implementation. `step5a-prepared-red/` saved the two prepared
boundary failures before extending generation. The first real gradient probe
used a wrong parameter-name prefix; its failure and complete correction to
parameter identity are under `step5a-green/` and `step5a-probe-repair/`.

Validation completed so far: the numeric path, independent perception pullbacks
and equal-parameter comparison passed; the corrected real whole-path probe
passed; all nine prepared-boundary and supplied-answer checks passed in
`step5a-prepared-green/`.
`step5a-native-training/` contains a collection-only error from a mistaken
parameter ID; no model ran there. The corrected selectors are recorded in
`step5a-native-training-corrected/`.

The mixed supplied/automatic native training case passed in 703.3 seconds.
The native conditioner training case failed in 576.4 seconds because the new
preview attempted to turn a temporary two-role meaning into a completed field.
`step5a-native-field/` saves the pre-repair source and complete bodies. The
repair uses the same one-slot point conversion as `_commit_sentence`, retaining
the graph. It also checks for a supplied answer before starting generation.
Both focused failures are saved in `step5a-native-field-red/`; all five supplied
answer tests then passed in `step5a-native-field-green/`. The native
integration recheck passed in 721.1 seconds in `step5a-native-field-integration/`
(peak 4.93 GiB).
A one-second native process sample is saved in the corrected training run; it
shows Dynamo and autograd activity, not I/O waiting.

Ten unseeded class-gate runs completed in `step5a-class-ten/`: **8/10** met
the unchanged bar (four right, MSE < .05). The campaign took 320.526 seconds.
Every answer and error is in [the ten-run table](step5a-class-results.md).
The count did not stop steps 6–9, as Alec explicitly decided. The campaign
included the native integration supervisor in its 24 GiB aggregate accounting
(`step5a-companion.json`). No runs were selected or retried.


## Steps 6 and 7

Step 6 selects one candidate pair of minimum residual for the forward inverse,
while keeping the former soft mixture's gradient. The gate reads the actual
concluded state through the selected grammar operations before their transient
record is discarded. Its candidate bank is the known object vocabulary; the
original leaves and target text are not inverse witnesses. All four word
multisets must match, preserving multiplicity and admitting only transpositions.
The native byte-training reconstruction objective and bank are unchanged.

`step6-red/` saves the pair-blending and missing-gate failures. The two focused
checks then passed. `step6-capture-corrected/` found a higher-order-loop input
alias between the current sentence and the full bank. The independent tensor
copies are recorded in `step6-alias/`; the capture check passed in 9.5 seconds
in `step6-capture-alias-green/`. `step6/` contains complete old/new bodies.
The initial `step6-capture/` attempt had a nonexistent selector and ran no model.

Step 7 carries each slot's last unary through pushes and binary compaction,
resets history for a new binary result, and masks a declared immediate inverse
at that same slot. It covers parallel derivation, serial tensor composition and
the eager/compiled closing. Not and non declare their self-inverses; unrelated
unaries remain eligible. The public compiled result remains 21 values.

Exploration selects a used round with another legal action. When a sentence
has only one legal derivation, both sentence trials can be costed identically;
the second cannot win a tie. The low-level explicit request for a distinct
parallel derivation still raises if none exists. Its existing assertion and
the uniform-selection assertion remain intact.

`step7-red/` saved all six new failures before implementation. `step7-green/`
ran 38 cases (34 passed); it exposed two implementation issues caught by existing
exploration assertions, a trace fixture missing the new metadata, and a new
single-choice probe that mistakenly supplied two possible binary locations.
`step7-followup/` retains the complete corrections. No existing assertion was
removed or relaxed. The trace fixture now supplies an explicit false alternative
flag; the new single-choice probe uses two operands. All 13 follow-up cases
passed, including the real compiled forward/backward and gate-capture cases.
`step7/` contains the complete step bodies and patch.

## Step 8: sum-only control — failed its intended control

The one 400-epoch unseeded control completed in 64.88 seconds with an 8 GiB
worker guard (peak .67 GiB). [Its patch](sum-control/control.patch) changes only
the receipt's copied grammar to `sum`; it has no unary operation. The repository
fixture remains byte-for-byte unchanged. Keeping an input-dependent `not` would
not be an additive-only control.

Answers, in fixture order, were **[.32371652, .78582978, .78723955, .39919221]**:
**4/4 correct, MSE .08882067509**. It fails the settled MSE bar, but it **does not
stay at one half**, as the plan requires of the control. All reconstructions
were `there there there`. The worker's zero exit means the unchanged settled
class bar was not met; [the explicit control verdict](sum-control/verdict.json)
is **FAIL**. This result does not establish that the class gate isolates the
nonlinear grammar. No additional control training, configuration tuning or
threshold change was made. The cause has not been isolated by this receipt.

## Step 9: final measurements

The final ten runs of each grammar gate completed in **623.60 seconds** in
`final-grammar-ten/`: **10/10 class successes, 0/10 reconstruction successes**.
Peak aggregate memory was 2.16 GiB. [The complete table](final-grammar-results.md)
records four answers and four actual grammar reconstructions for every run.
None of the 40 sentences in the reconstruction-gate runs matched its word
multiset. All recovered texts have three word tokens: the current forward
pushes three leaves (including the separator), and the lexical inverse returns
an owned word for each recovered leaf. The gate does not drop an extra word
or forgive it as a transposition. This remains a failure for review, alongside
the sum-only control; the receipt does not claim reconstruction is learned. Trial 1 of
each gate is predeclared as its row in the full named XOR table, so the table
will not launch an eleventh model or select a favorable run. The named XOR table is **34/35**: the class row passes, the reconstruction
row fails, and every other named proof passes. The **one MM_20M_xor exact round
trip passes (1/1)**. See [every named result and slow proof](final-measurements/candidate/table.md).
The remaining campaign took 317.13 seconds, with peak aggregate memory 14.68 GiB.
The final [ten MM_grammar runs](final-mm-results.md) have median ending MSE
**1.83977055812e-10**, against the accepted item 7 baseline **.1066**. All 900
updates ran in every trial. Two runs plateaued at .125; they are retained.
This is the accepted raw-forward measurement, not a claim that native sentence
trials learned the grammar. `final-source.json`
and `final-inputs.json` freeze the source and fixture snapshot for the final
measurements and the one full sweep.

The prior red XOR_grammar receipts remain unchanged under the September 30
and October 1 directories. The [accepted red depth-three campaign](../2026-09-27-item7-5-landing/README.md)
and [September 30 recheck](../2026-09-30-item6-9/depth3-after/run/result.json)
also remain unchanged. Historical 12/15 exact
round trips are not replaced by a claim that the new single trial proves
reliability. The sum-only control is a review blocker, even if other gates pass.


## Final full sweep

The one full sweep completed all **5,193 cases in 113.5 minutes**: **4,849 passed,
21 failed, 322 skipped and 1 xfailed**, plus four passed subtests. It matches the
frozen 720-file source snapshot and supporting fixtures. There were no missing
cases, duplicate completions, resource stops or compile-cache retries. The
standing RUN_SLOW=0, CPU/eager defaults, 8 GiB per-worker guard and 24 GiB
aggregate dispatch budget were retained; aggregate peak was 17.84 GiB. Slow XOR
proofs were measured separately above.

Round 5 had 5,122 cases in 89.6 minutes: this sweep has 71 more cases and took
23.9 minutes longer. Case sets and early failures differ, so this is not a
controlled speed comparison. [The sweep summary](full-sweep/summary.md) lists
every failure and links its full report. [Final verification](final-verification.json)
confirms the source and supporting inputs still match, the environment freeze
is unchanged, `git diff --check` passes, and HEAD is still `d679df2b`.

[Source delta](source-delta.json) lists the three runtime files, four affected
existing test files, and three new test files relative to the reviewed October
1 candidate. No configuration, dependency requirement, or runner guard changed.
[Port index](port-index.json) links the twelve retained test-body records.
Nothing is committed. Part 4 remains deferred until review.


### Failures found during the frozen sweep

Twenty-one cases failed. Four stop in the shared output-policy probe at
its older assertion that no policy cost exists during sentence backward.
The no-answer test stops during its supervised warm-up, before testing the
no-answer phase. [The diagnosis](full-sweep-review/output-policy-note.md)
distinguishes that observation from a proven no-answer gradient regression;
the later assertions remain unverified. The candidate needs review of this
transient record or a corresponding probe port, without relaxing policy-gradient,
zero-weight, row-mask, or no-answer guarantees.

Sixteen cases construct the old private sentence-state tuple: fifteen through
`reading_fixtures.commit_reading` and one directly. They lack step 7's two new
fields. [The fixture note](full-sweep-review/fixture-note.md) records the unresolved
ports. One additional test still expects a blended operand from the bounded
inverse; [the inverse note](full-sweep-review/inverse-probe-note.md) records its
unchanged assertion and the hard-pair contract. Neither tests nor production
source have been changed during this single sweep. These failures remain red.
All six previously reviewed repair cases pass in this sweep.


## Objective-conflict stage 1 — held for Claude

The additional measurement request and its open one-run versus cut-training
comparison are recorded in [the hold note](objective-conflicts/STATUS.md).
Neither requested configuration has started. The draft supervisor was stopped
while waiting for the full sweep, following Alec's “Let’s hear back from Claude
first.” The existing full sweep has now finished. No new measurement will
start automatically. The observer is a draft, not evidence. Part 4 and item
6.85 remain deferred.
