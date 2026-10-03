# Item 6.9 — free read-back, operators and stable ownership (§§21–22)

Measurements complete; stopped for review. **This candidate has unresolved regressions and incomplete ports; item 6.9 remains open.** One working tree, no commits or HEAD run. The [§20 receipt](../2026-10-02-item6-9-reconstruction/README.md) is the unaccepted baseline. The source before this round is preserved in `before.zip` and `before.json`, with the environment in `environment-freeze.txt`. Saved prior failures are in `prior-triage.json` and `prior-failure-notes.json`. New failing probes precede each repair.

This round removes inverse witnesses; uses product conjunction and mean disjunction; compares only reconstruction; trains the answer reader only on the kept trial; and gives reconstruction gradient descent with momentum while keeping Adam for its readers. Existing bars, guards and unseeded measurement rules remain. Class passes **9/10**, reconstruction **5/10**, sum control **10/10**, and the named XOR table **33/34** with only the declared MM_xor failure. The full sweep has **8 failures** and the moved cases **6**. All first outcomes are retained below.

**Scope issue for review:** witness offsets and numerical frames in the understanding are removed, but the operation journal still retains temporary numerical frames for item-7 clause closing. Thus the journal is not entirely reduced to rules and positions as §21.6 specifies. Those frames do not feed either read-back or the answer. The retained closing semantics and this incomplete scope are described below; this receipt does not claim the journal-only requirement is fully implemented.

## Implementation and evidence

The candidate is frozen in [source-final.zip](source-final.zip), with per-file hashes in [source-final.json](source-final.json). The closing driver checks those hashes between stages. Source before the round is in [before.zip](before.zip). The original eight sweep failures and moved-case failures are retained in [prior-results.json](prior-results.json) and [prior-failure-notes.json](prior-failure-notes.json). All new diagnostic attempts, including failed intermediate repairs, remain under `probes/`; [focused-results.json](focused-results.json) indexes their process outcomes, time and peak memory.

| Change | Saved failing evidence | Verification before the closing campaign |
|---|---|---|
| Product conjunction, mean disjunction, momentum reconstruction owner | `probes/contracts-before/` | `operators-optimizer-after`; catalogue three-face tests |
| Reconstruction-only comparison, ties greedy; reader trains only retained rows | `probes/trial-before/` | `trial-after`; exact row-gradient and empty-reader-step tests |
| Free-only reconstruction; remove witness offsets | `probes/contracts-before/`, prior free-branch compiler failure | `free-after`; eight compiled sentence/query/traversal cases pass without raising the capture limit |
| Search both operands for affine and discarding operators as well | `probes/free-lift-before/`, `probes/free-discard-before/` | `reference-and-bank-after`: 53 passed |
| Carry the repeated conjunction reference to the next operation | `probes/same-reference-before/` | repeated-reference clause test and clause/journal regression cases pass |
| Mixing binding supplies native object ids to clause scope | `probes/ports-and-operators-after/` (unavailable shared row) | clause, separator and sentence integration checks pass |
| Resume adaptive readers while resetting reconstruction's old Adam moments | `probes/optimizer-resume-before/` | named-owner migration regression and existing checkpoint tests |
| Displacement/stability audit observes actual calls | `probes/audit-before/` | `audit-after`; one-batch validation artifacts in `audit-validation/` |

The final grouped port check passed 273 cases, with four existing skips. The unchanged 40-epoch serial supervised learning case passed. The unchanged three-epoch free-derivation round-trip case remained at zero exact recovery in its diagnostic; its bar remains 1.0. Its closing moved-case measurement also saves actual decoded strings and the already-computed inverse inputs/results. It is not reclassified as an expected failure.

### Operators and reconstruction

Conjunction binds two vectors as `norm(x) * norm(y) * unit(x*y)`. Equal native references return the reference, including when the result is used in the next conjunction. Numerical equality at distinct references does not trigger idempotence. Disjunction is the mean and `not` still negates. Catalogue `min`/`max` retain coordinate minimum/maximum with compose, generate and searched reverse faces; current grammars select neither. Tests whose subject is the separate lattice meet/join now use their explicit intersection/union names.

The only input-reading reconstruction term is `reconstruction.free_bytes`, divided by `log(256)` and weighted by the existing `reconstructionScale`. Both operands are found over the primed bank through the same compose kernel. Read-back and its gate share signed activation × cosine × priming scores. The source code values remain live for reconstruction; target bytes, priming weights and selected addresses are detached. The hard least-residual pair retains its existing soft gradient and bounds. There is no code normalization or distributional pressure.

The understanding record contains root, end slots/depth, packed sentence roots/depths, per-word values/rows/validity, recorded rule ids/arities/validity, operand positions, sentence id, and the primed-symbol snapshot (rows, live codes, weights, own-row mask, surface bytes and validity). It contains neither witness offsets nor numerical operation frames. Reconstruction and the answer still consume the same trial record, with the answer cut intact. The answer's fixed affine record features shrink accordingly.

The two witness-offset slabs and their recorder/consumer paths are removed. Temporary numerical operation/reference frames remain exclusively for the item-7 clause closing: it must record the operation actually performed, including non-replayed relation semantics. They are excluded from the understanding and free inverse and cleared at the closing. This distinction was raised for review rather than silently removing item-7 semantics; it is not a numerical witness supplied to decoding.

### Selection, optimizers and audit

Both trials are costed at one parameter state before either optimizer step. Explore wins only for strictly lower reconstruction. Answer and expectation are excluded from selection. Reconstruction and expectation retain their two trial updates; an answer update uses only the retained rows. A trial with no retained reader rows has no reader gradient or Adam momentum-only update.

Reconstruction uses momentum SGD at the configured learning rate, with momentum 0.9: the momentum is `0.9 * previous + gradient`, and displacement is `-lr * momentum`. The compact sparse form stores a float32 prefix and updates only observed nonzero-gradient rows or active proximal rows. The first step is proportional to the gradient, without Adam's variance division. Readers and expectation predictors retain Adam. No seed, learning rate, threshold, capacity, code norm constraint or guard changed.

Current checkpoint inventory is in [checkpoint-inventory.json](checkpoint-inventory.json). Existing Adam reader states resume by name; an Adam state for a now-reconstruction-owned parameter is deliberately dropped when moving to SGD. Shared weights survive. The new migration test verifies both behaviors and the first reconstruction step.

The audit writes each code/chooser anchor gradient and displacement coordinate to compressed arrays, with per-step norms and cosines in its events. Unlisted sparse rows have zero gradient and zero displacement under the row-local update contract. Derivation stability uses the kept record's rule sequence, arities and operand positions, reporting modal-epoch fraction and distinct derivations per sentence. Code geometry, four-root singular values, VQ cluster sizes, declared/observed owners, objective reach, costs, weights, gradient cosines, selection differences and term magnitudes remain included. These are observers of the required runs, not extra training campaigns.

## Ports

[ports.json](ports.json) lists every changed test/helper file, each changed or renamed test, and its reason. The `ports/` directory contains complete old and new files, preserving full bodies and decorators. No assertion threshold is relaxed; where a decided operator or ownership contract replaced the old subject, the new assertion is explicit. Five new regression files cover the operator catalogue, retained-row readers, momentum updates, optimizer migration and compiled inverse gradients.

## Closing measurement policy

One native production-batch stage-1 measurement (batch 28, existing 24 GiB slow-only ceiling), then ten class, ten reconstruction and ten sum-control runs; one named XOR table with its first gate runs reused; ten MM_grammar runs; conditional attribution; moved weekly cases; one source-matched ordinary sweep. The ordinary worker guard remains 8 GiB. The sweep starts with ten workers and the machine's existing 28 GiB aggregate bound. Moved slow cases start in separate workers so a failed case's retained graph cannot consume the next case's budget. They are attempted once; stopped cases are reported, never retried. No HEAD run, venv rebuild or commit.

Class and MM_grammar final answer errors are classified as **at 0** (`< .05`), **at ¼** (`abs(error-.25) <= .02`), **between**, or **above ¼**. Sum control requires absolute checkerboard contrast at most `1e-4` and no class-bar success. Attribution runs only if class is below 8/10 or reconstruction is at most 3/10, with ten fresh runs each of R, R+E, R+A and all three.

The baseline exceptions are unchanged: MM_xor convergence remains red by §17 (the old reading's cross-word chunk shortcut is gone; word-level convergence belongs to 6.8), and MM_grammar has its documented occasional .25 stop. All other failures remain visible.

### Completed learning measurements

| Measurement | First outcomes | Comparison |
|---|---|---|
| Class gate | **9/10** at the full bar; all ten classify all four inputs correctly | §14: 8/10; §20: 0/10 |
| Reconstruction gate | **5/10**, accepting word transpositions as before | §14: 3/10; §20: 2/10 |
| Sum-only control | **10/10** satisfy contrast ≤ `1e-4` and miss the class bar | Every MSE is in the quarter-error band |
| Named XOR table | **33/34** | Sole failure: unchanged, deferred MM_xor convergence; best MSE `.2416596` against `.20` |
| MM_grammar | **9 at 0, 1 at ¼**; median ending training MSE **`4.10104e-7`** | Run 8 remains exactly `.25` after 900 epochs and evaluation |

The class errors are nine **at 0**, none **at ¼**, one **between** (`.0578323`), and none **above ¼**. The MM_grammar errors are nine **at 0**, one **at ¼**, none **between**, and none **above ¼**. Thus the predicted all-or-quarter class pattern is not exact at this fixed training budget. These are unpaired samples; their count differences alone do not identify which change caused an improvement.

Both XOR_exact checks and the single MM_20M exact round trip pass. The first predeclared class/reconstruction runs supply the named-table rows; they are not repeated for the table. The [attribution decision](attribution-decision.json) is false under the required rule (class ≥ 8/10 and reconstruction > 3/10), so no attribution runs were added. [Measurements](measurements.md) records every answer, read-back, contrast, final error band, process duration and peak; [results.json](results.json) retains the underlying outcomes.

### Full sweep, moved cases and remaining work

The single source-matched sweep completed all **5,000 selected cases in 1,167.114 seconds (19.45 minutes)** on the reference **ten-worker** schedule: **4,706 passed, 8 failed, 285 skipped and 1 expected failure**. Against 5,219 cases / 122 minutes, this is 219 fewer cases and 102.55 fewer minutes. The suite and source differ, so this is an end-to-end comparison, not an isolated speedup attributable to one change. The ordinary guard remains **8 GiB per worker / 28 GiB aggregate**; observed aggregate peak is **9.42 GiB**. There are no process stops, unattempted cases or compiler-cache retries. Raw pytest reports include three extra passing `unittest.subTest` reports under their enclosing case's ID; they are not reruns. [Case summary](full-sweep/case-summary.json) and [full receipt](full-sweep/receipt.json) retain the counts, schedule and source hashes. The weekly warning remains visible: the latest weekly record contains failures.

The **165** newly dispatched moved cases completed in **1,235.574 seconds (20.59 minutes)**: **159 passed, 6 failed**, no stopped or unattempted cases. Seven additional moved selectors reuse their named-table results. All three prior stopped cases now pass: compiled expectation (502.7 seconds, 2.74 GiB), the real Inductor word loop (816.7 seconds, 3.21 GiB), and padding. All **21 weekly-tier attention cases** pass; combining ordinary coverage gives 46 passing attention cases. Both real MNIST training arms and the LFS-pointer loader check pass. [Coverage summary](coverage-summary.json) lists the exact cases.

All **eight prior full-sweep failures pass**. Across the sixteen prior failed/stopped cases tracked in [triage](triage.md), thirteen now pass; the serial learning case, free-derivation learning case and terminal-field observer remain red. Current failures are unwaived:

| Current failures | Diagnosis from the saved result and source |
|---|---|
| 2 answer-generation cases | **Production scope regression:** the new input-only free-search branch also intercepts the shared sum inverse used by answer generation. An unavailable split remains pending and emits no leaves. These are not old-operator fixture ports. |
| 2 stored-truth penalty cases | **Production kernel coupling:** `falsity_penalty` still calls the now-mean disjunction kernel for a union. Averaging reduces norm even for an agreeing or zero proposition, giving the exact unexpected penalties `.2828426` and `.7071068`. |
| Compiled byte-score parity | **Numerical regression:** a deterministic Inductor cost differs from eager by `4.6052e-5`, beyond unchanged tolerances. Gradient parity is not reached; the exact numerical transformation is not isolated. |
| 2 optimizer-state observers | They still require Adam `exp_avg` for reconstruction-owned parameters now using momentum SGD. No compatibility state was fabricated. |
| One-batch chooser update | Operand-order weights change, but no `mlp.*` weight moves bitwise. Its gradients were not saved, so an explanation from the separate XOR audit remains unproved for this case. |
| Serial supervised learning | Best accuracy `.96875` against the unchanged `1.0` bar after 40 epochs. The earlier focused pass does not replace this failed closing outcome. |
| Free-derivation round trip | Still `0/64` exact after three epochs, with actual decoded strings and inverse tensors retained; details below. |
| 4 moved observers/fixtures | The direct byte scorer assumes the retired activation-free fallback; a synthetic chunk chain supplies vectors absent from the bank; the terminal-field test still applies a physical order slice to gathered field evidence; a reverse chooser constructs the former radmax parent for disjunction. No replacement assertion is silently treated as passing. |

[Failure notes](failure-notes.json) identify every case, reached and unreached assertions, prior outcome, confirmed cause and uncertainty. Complete old/new bodies for changed ports remain in [ports.json](ports.json); unchanged tests are in the before/final source archives. The source stays frozen after these outcomes; no seed, retry, threshold or marker was used to select a pass.

### Native compilation repair before completing measurements

The first production attempt failed during AOT capture, before an optimizer step, at 10.48 GiB in 226.8 seconds. `torch.diag(d)`'s backward returned a diagonal view of stride `D+1`, while the inactive conditional branch returned a contiguous zero gradient. The conditional compiler could not merge them. This is newly exposed by searching through affine compose kernels in the free inverse, not a memory guard or convergence failure.

The complete failed source, native trace and three interrupted first class attempts are saved under [probes/native-conditional-stride](probes/native-conditional-stride/interruption.json). No class run had completed, and none was included as a success or failure of the closing gate. A four-dimensional reproducer failed with the same stride mismatch before the repair (`native-diagonal-before`). Building the identical diagonal with identity × row vector gives a dense backward reduction; the exact active/inactive-gradient regression and 48 related cases pass (`native-diagonal-after`). The corrected source is frozen separately for the closing campaign. This interrupted attempt is disclosed; it is not a silently discarded completed run or an unchanged-source retry to obtain a better learning result.

The first diagonal repair covered `compute_W` but not the separate `functional_forward(..., naive=True)` used by the native inverse. A second production capture therefore failed at the same point (227.5 seconds, 10.48 GiB), before any update; no gates started in that attempt. It is saved in `probes/native-functional-diagonal/`. The functional four-dimensional reproducer also failed before that path was repaired (`native-functional-before`). Both exact diagonal regressions and a compiled native-operator free-search test now pass (`native-free-aot-after`), including equality of eager and compiled parameter gradients. The subsequent production attempt is the first allowed to advance to learning measurements. Both failed pre-training attempts remain available for review.

## Audits on the completed required runs

The production stage-1 test passes at batch 28 in 787.2 seconds (13.12 minutes), with a guarded process-tree peak of 21.91 GiB. This fits the existing 24 GiB slow ceiling, not the 8 GiB ordinary ceiling. Its three backward calls record zero writer conflicts (41 active parameter records, 111 inactive). The first code displacement is proportional to the current gradient; the next includes momentum. The native configuration has one training epoch; its 21 distinct sentence identities therefore do not provide a multi-epoch stability experiment. Its first-sentence training batch supplies no active expectation target.

The first class run supplies the XOR audit, with no additional training. It reaches answer MSE `1.0667e-12` but recovers only two of four word multisets. All 1,200 backwards record zero ownership conflicts. Across 800 trial updates, the largest code-gradient norm is `1.6221e-10`; the largest chooser-anchor norm is `1.2091e-12`. **Every saved code/anchor displacement is exactly zero at float32 precision.** The before/after dictionary geometry and root spectra are consequently unchanged; VQ cluster sizes remain all ones. This run demonstrates learning by the answer reader, not by these code or chooser weights. The saved evidence does not establish how a different gradient scale would behave, and no scale or threshold was changed.

This is not evidence that reconstruction reached its floor: its endpoint relative error is `.9861974`. The existing byte scorer retains temperature `.1` and clamps target probabilities at `1e-6` before taking the logarithm; probabilities below that boundary have zero derivative through the clamp. The audit did not save per-byte probabilities or each trial's full priming weights, so it cannot attribute the observed gradient magnitude to that boundary alone. No additional training run or scorer change was made to test that hypothesis.

The kept derivations' modal fractions in sentence order are `.7075, 1, 1, .68`, with `3, 1, 1, 2` distinct derivations over 400 epochs. Explore is selected for 245 of 1,600 active sentence rows, with zero reconstruction-selection violations. The kept answer error is worse than the rejected trial in 240 rows; expectation is worse in 244. Those objectives do not participate in the decided comparison. Native explore is selected for 2/28 rows, also without selection violations. Neither measured configuration has an activated extra word candidate outranking an own word; the observed candidate count itself is zero, so this does not test competition from activated words.

[Audits](audits.md) gives every objective weight and purpose, first-state group norms/cosines for both trials, selection gaps, endpoint costs with and without the answer, term magnitudes, geometry, displacement summaries and per-sentence stability. [audit-summary.json](audit-summary.json) preserves the full summaries; per-coordinate arrays remain beside each run's event log. Displacement summaries recompute norms/cosines in float64 from the saved arrays to avoid small float32 dot/norm roundoff in the online event ratios.

### Read-back limit isolated from saved tensors

[recorded-root-diagnostic.json](recorded-root-diagnostic.json), produced by [inspect_recorded_xor.py](inspect_recorded_xor.py), uses the saved first-audit codes and roots only: no model construction, forward, training or new random draw. The four recorded roots match `conjunction(h,w)`, `not(conjunction(h,t))`, `conjunction(not(l),w)` and `not(conjunction(not(l),t))` to at most `2.98e-8`.

For the first two, undoing any outer `not` leaves a root exactly recomposable from the two raw bank candidates. For the last two, the inverse searches raw dictionary pairs before unwinding the inner `not`; the negated intermediate is not among those pairs. The least-residual pair is therefore `(loving, loving)`, with parent MSE `.187811` and `.217822`, instead of the two original words. This exactly accounts for the audited repeated-word read-backs. It is a remaining structural limitation of the free search, not evidence that a larger answer gradient is needed. The closing candidate, fixture and assertions are left fixed; this evidence is for review, not an unmeasured repair after seeing gate outcomes.

### Saved moved-case recovery evidence

[Saved free-derivation analysis](saved-free-derivation-analysis.json), reproduced by [analyse_saved_moved.py](analyse_saved_moved.py), uses only the final evaluation tensors from the failed MM_ladder run. All 346 active unit occurrences have their own candidate and none reports truncation. Only 42 recover the correct surface; none of 128 whitespace occurrences does, and `1` is chosen in 279 positions. For example, `1 plus 1` becomes `1 1 1 1 plus`. Two-digit inputs already represented as separate digit references account for 26 extra positions before inversion. These observations distinguish admission/representation from subsequent inverse errors.

Reconstructing target-byte/end probabilities from the saved recovered vectors, codes, priming and candidate surfaces puts **548 of 884** below the existing `1e-6` clamp. Their direct probability derivatives through that clamp are zero. This is evidence about this failed moved run; it does not recover the first XOR audit's unsaved per-byte probabilities or establish how an unclipped objective would train. No model, optimizer or new random draw is used by the analysis.

The [direct-scorer source comparison](byte-fidelity-source.patch) and [arithmetic counterexample](byte-fidelity-contract-diagnostic.json) preserve the separate fixture issue: the old no-priming path ignored activation; the new unified score retains it. A correctly directed low-activation leaf need not have near-zero cross-entropy. The actual failed fixture's geometry was not retained, so the precise `.0152916` is not reproduced or waived.

Final integrity checks confirm all 669 measured source files and all 33 saved port body pairs match, HEAD remains `d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`, and the before/after `pip freeze` output is identical. `git diff --check` is clean. The separate check after receipt/documentation updates passes all 267 documentation-link cases; it is not another full sweep. These checks are recorded in `receipt-integrity.json`. Nothing is committed.
