# Item 6.9 continuation — October 1, evening

**Latest candidate:** the [§14 continuation receipt](review14/README.md)
records the static journal buckets, completed reading-mode migration, failure
dispositions and completed weekly triage. Closing measurements are complete:
class 8/10, reconstruction 3/10, sum control 10/10, the named table 30/34 and
MM_grammar median ending MSE 4.36e-11. The source-matched full sweep attempted
all 5,060 cases in 214.66 minutes: 4,672 passed, 314 skipped, 71 reported
failures and three stopped without a result. The receipt discloses ten
inadvertently repeated case IDs, retains first-attempt outcomes and includes
that interruption in wall time. Stopped for review; nothing committed.
The sections below preserve the earlier §12/§13 receipts and decisions as they
stood then; their old-mode hold and pending weekly status are historical.

One working tree; nothing committed. This receipt follows plan §12 in five parts:
operation-record repair, production native measurements, suite trim items 1–10,
the cost function within 6.9, then the closing gates and one full sweep. The venv
rebuild and old reading-mode migration remain held until 6.9 closes. There is no
separate item 6.85 and no intermediate full sweep or review stop.

## 1. Item 7 operation-record memory repair

The saved failing probes are under `stage1/native-memory-diagnosis.json`,
`stage1/before-journal-repair-24gib/`, and `stage1/repairs/native-journal/`.
At production batch 28, the original numerical journals have shapes
`[28,4864,3096]` and `[28,4864,2064]`. Each compiled operation functionally
rewrites a whole journal; one value-scatter result alone is 1.57 GiB. Both
original production arms exceeded the 24 GiB slow ceiling before a trial cost
or optimizer update (24.39/24.33 GiB sampled peaks). Earlier 12 GiB sizing
attempts are retained separately in `stage1/sizing-12gib/`.

The repair sizes these numerical journals to one open sentence, maps global
structural addresses to sentence-local numerical addresses, and performs the
compiled operation writes in a word-sized window before publishing that window.
At batch 28 the journals become `[28,25,3096]` and `[28,25,2064]`. All numerical
values remain live: no new gradient cut was introduced. Inactive gaps share
frames 0–2; their row gates suppress writes. Active words count only active
predecessors, so gaps neither split a sentence nor alias its active frames. Complete old/new production
bodies, patches, the original layout failure and the inactive-gap failure are
saved in `stage1/repairs/native-journal/`.

The same graph diagnostic on published HEAD d679df2b confirms that the journal
growth was present at the item 7 landing. HEAD/candidate graph-capture peaks are
10.00/3.78 GiB; these are capture-only measurements, not training peaks.
The focused suite passed 29/29 cases in 795.7 seconds, followed by 3/3 journal
layout/address/gradient cases after the inactive-gap correction. The real
compiled packed-sentence case passed unchanged.

## 2. Native production stage 1

The two fresh unseeded arms (step 5a and the trial-answer cut) run as RUN_SLOW
tests at the unchanged production training batch of 28. The slow worker and
aggregate ceilings are **24 GiB**, one worker at a time, with a 30-minute worker
deadline. The ordinary sweep retains **8 GiB per worker / 24 GiB aggregate**.
The weekly target states the native exception explicitly. No environment
rebuild, production fixture change, or training seed was applied. The native
fixture's pre-existing fixed dataset seed remains unchanged.

The first attempt after the journal repair completed training and its automatic
16-row endpoint evaluation, but then stalled in console reconstruction rendering.
An interrupt traceback identified `RadixLayer.reverse` scanning the whole concept
bank for each rendered position. That attempt peaked at **22.24 GiB**, below the
slow ceiling but above the 8 GiB target; it was interrupted after 27.4 minutes.
Its numerical data and interrupted companion arm are retained under
`stage1/before-report-finalizer-repair/`. The saved failing report probe and
complete old/new bodies are in `stage1/repairs/native-report-finalizer/`.

The measurement observer now saves the same endpoint predictions and bypasses
only the subsequent console rendering. It preserves the original endpoint
`runEpoch`, batch size, sigma transitions, optimizer behavior and all slow-test
assertions. Both arms passed on identical source and supporting-input hashes. Step 5a took
1,276.8 seconds and peaked at **22.03 GiB**; the answer cut took 1,277.3 seconds
and peaked at **21.04 GiB**. Both include the automatic endpoint evaluation.
The complete measurements are in [stage1/README.md](stage1/README.md);
[the weight interpretation](stage1/native-weight-interpretation.md) explains
each weight and its purpose. Earlier attempts are retained but are not counted
as successful completed arms. The slow ceiling is met; the 8 GiB target is not.

## 3. Suite trim

Items 1–10 are implemented. The precursor has a function/test coverage map and
live UTF-8 witness ports; parked classes and the etc examples keep their inline
unit tests. Unused APIs and retired skips are removed with use audits. Current
checkpoint migrations and the deferred old reading modes remain. Eighty-one
absence-only guards share one table; behavioral negative checks remain separate.
The 46 review-round files are now 42 behavior files (245 moved definitions,
no duplicate-test deletions in this step). Complete old/new bodies are retained
in `suite-trim/ports-and-retirements.json`.

Targeted checks: inline Legacy/SPNN/SigmaPi/SymPercept passed; 72 cases for
items 1–5 passed; the two thought-fixture ports failed first, were saved and
corrected, and passed; the consolidated guard passed. Full collection succeeds
with **5,045 cases** after removing one unused import, whose failing collection
and before/after source are saved. This is collection, not a full sweep.

The 25 previously slowest cases all pass with their subjects and assertions
retained. Their summed time falls from **19,742.7 seconds to 1,712.8 seconds**
with profiling overhead included. The real graph tests remain compiled. See
[suite-trim/performance-comparison.md](suite-trim/performance-comparison.md)
for each before/after value, including the former 30-minute interleave timeout.
Final unprofiled times will come from the part-5 sweep.

`make test_slow_weekly` selects **302 slow cases** and writes dated records to
`tmp/slow-tests/`. Selection/accounting/age checks passed (5 cases); all three
MPS cases passed. The W16/W64 checks uncovered and saved a fixture anchor
omission, a compiled closing input alias, and an inactive pipeline address in
the compact journal. Their unchanged assertions now pass, together with the
three local-journal checks (5/5). The two MPS cases took about 273 seconds each,
with process peaks of 3.39 and 4.52 GiB. Ordinary workers remain capped at 8 GiB;
the two native production arms have the explicit 24 GiB slow exception.
The Sunday 03:00 launchd schedule is proposed in the receipt, not installed.
No full weekly-run record has been manufactured from these partial checks.

Use/coverage dispositions, checkpoint string searches, and complete original,
intermediate and final bodies are under `suite-trim/`. The final source-matched
sweep remains in part 5.

## 4. Cost function within 6.9

Implemented and frozen for the closing receipt. This includes the mixing
reconstruction bank, its two XOR stage-1 arms, reconstruction on the retained
reading paths, relative errors assembled by the Error registry, reconstruction
precedence for answer gradients and trial selection, and term-by-term
documentation. The earlier prototypes and their failures are recorded below;
the final implementation follows the §10.2a scope decision.

The earlier receipt-local `reconstructInLoop=true` XOR arms both failed before
training because the mixing reading boundary had no reconstruction bank. Those
failures and the fixture patch remain under `stage1/`; they are the saved probes
for this part, not numerical gate results. The bank is now staged without publishing native word identities. Both source-matched
400-epoch XOR arms completed: step 5a in 80.4 seconds and the trial-answer cut
in 79.9 seconds, each below 0.673 GiB. Step 5a classified 4/4 with MSE
0.0643559 and reconstructed 2/4; the cut classified 4/4 with MSE 0.0214290
and reconstructed 4/4. These are independent unseeded runs, not paired
initializations. The step-5a selector kept worse reconstruction 21/1600 times.
Full reach, weights, single-state gradients, selection audit, endpoint costs and
term distributions are in [cost-function/README.md](cost-function/README.md).

The saved bank attempt first exposed an extra target look-ahead column; after
that shape repair, uncached reconstruction loop capture cost about 15 seconds
per XOR epoch. The no-compile execution backend now runs the identical
functional traversal with ordinary autograd. Its value/gradient parity and
actual eager path probes pass (3/3). Compiled execution retains the HOP. All
failed/interrupted attempts and complete old/new repair bodies are preserved.

The universal-reconstruction prototype and its one-batch configuration audit are
saved separately in `cost-function/universal-reconstruction-repair.json` and
[cost-function/configuration-timing-comparison.md](cost-function/configuration-timing-comparison.md).
Before: 23 completed, 7 configuration/data failures, 6 memory stops. Prototype:
13 completed, 17 failures, 6 memory stops. Both phase manifests match their source.
The conservative 36-entry audit includes the default model.xml, which runtime
identifies as parallel and does not count as gaining the serial inverse.

The default XOR reconstruction probe passes with the prototype. Its other
failures expose missing BPE surface metadata, missing concluded-state products
in grammar-free serial paths, and sentences with no admitted surface candidates.
The mode/shape audit is saved in `cost-function/old-reading-staging-audit.json`.
Retired-student and explicit-off fixtures also fail at the new schema boundary;
those failures are saved before any port. No test assertion was changed.
Only this unfinished prototype has been restored out, with exact file-hash
verification; the verified mixing-bank/eager-loop repairs remain in production.

Claude's §10.2a baseline decision now supersedes that restored prototype. The retained meronomy readings reconstruct through their understanding; exempt old modes and grammar-free configurations are listed in [configuration-scope.md](cost-function/configuration-scope.md). Missing admitted-candidate sentences are counted and omitted from the reconstruction mean. The Error registry owns the trained totals and normalizes each targeted term against the uninformed baseline, with penalties apart and existing priorities explicit. Supplied-answer gradients are projected only on parameters shared with reconstruction, and an explore trial needs a lower total without higher reconstruction. Both trials are costed before either update.

Implementation details, saved failures and port dispositions are in [implementation-notes.md](cost-function/implementation-notes.md) and [port dispositions](cost-function/port-dispositions.md). The final focused selection passed **308/308** in 67.36 seconds. Complete old/new bodies, changed-file archives and hashes accompany it. Projection requires two extra objective-gradient reads (reconstruction and answer) before the ordinary total backward; the spec had estimated one. No extra optimizer step is introduced for projection. The unchanged proximal L1 strength is now staged on each applicable sentence optimizer step as well as any applicable batch step, and is never differentiated again.

The isolated reconstruction timing is [recorded per configuration](cost-function/scoped-timing.md): three complete comparisons; eight double memory stops under 8 GiB and one double sentence-capacity rejection. These are first-batch numerical timings, not full epochs. No successful timing is claimed for a guarded-out configuration.

## 5. Closing receipt

The closing measurements finished in **43.81 minutes**, source matched and
without a resource stop. [Every run's answers, read-backs, contrasts and raw
process records](closing/README.md) are retained.

| Measurement | Result |
|---|---|
| Class gate, unchanged all-four / MSE < .05 bar | **9/10**; every run classified all four correctly |
| Reconstruction gate, four word multisets | **0/10** |
| Sum-only control, settled float32 one-half comparison | **0/10**; maximum deviations .00100–.00331, all XOR contrasts at most 2.98e-8 |
| Named XOR table | **33/34**, including the single exact round trip; sole red case is reconstruction |
| MM_grammar, ten 900-update runs | Median ending MSE **.0625**, versus accepted **.1066**; five near zero, three .125, two .25 |

The old table's 49 cases included fifteen exact trials and the SPNN smoke
case. The receipt rule and explicit trim give 49−14−1=34 cases. The accepted
44/49 and 12/15 results remain the baseline; one successful current exact
round trip does not replace the historical repeated-run result.

The unchanged XOR_grammar fixture inherits legacy lexicon synthesis and is
exempt from automatic tied reconstruction under §10.2a. Its reconstruction-
enabled stage-1 copy remains a separately labelled measurement. The deferred
old-mode migration is therefore material to interpreting this red gate.

The one final source-matched sweep completed **5,090/5,090 cases in 110.70
minutes**: **4,760 passed, 11 failed, 318 skipped and one expected failure**.
Previous: **5,219 cases / 122 minutes**; the differences are −129 cases and
−11.30 minutes. Each worker retained 8 GiB; two workers shared 16 GiB alongside
the independent 8 GiB weekly worker. The largest worker reached **7.24 GiB**.
There were no resource stops or compilation-cache retries. This is an observed
wall-time comparison: the prior runner allowed ten workers with batches of up
to 256 selectors / 16 files; this run used two workers and eight selectors /
one file. `closing/fixture-batching-audit.json` confirms that both consumers of
the shared `trained_ladder` fixture ran in the same worker.

The [eleven saved failures](closing/review-findings.md) remain unresolved.
Nine involve fixtures or subcases that still expect the former registry,
reconstruction-scope or empty-candidate contract. The other two are the
zero-valued compose-deadline result and an extra graph capture across sentence
lengths. No assertions were changed to turn this receipt green. Of the 25
previously profiled cases, 24 pass on final source; the legacy-reporting case
fails its old reconstruction-mode assertion. Their individual final times and
the remaining runtime hotspots are in the [closing receipt](closing/README.md).

The runner's raw counts include four additional passed subtest events from
one flags test. [Case accounting](closing/case-accounting.json) preserves those
events and the correct one-result-per-collected-case totals; no test was rerun
for this reporting correction.

The weekly run continues on its earlier frozen review-13 source. Its
[timestamped partial record](review13/weekly-progress-at-review.json) is kept
separate from validation of this cost-function candidate. Nothing is committed;
**6.9 remains open and the candidate is stopped for review**. The environment
rebuild and old-reading-mode migration remain held.
