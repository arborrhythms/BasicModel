# Decoder, operators and 6.8 — §16.3 implementation, measurement held

2026-10-05. HEAD remains `802abb1acc95e1bddc8cb237b13230a336681c49`. One working tree; nothing committed. Delivered conceptual capacities are **6 (XOR_grammar) / 8 (MM_grammar)**. The [incoming §15 receipt](README-before-review16.md), older receipts and historical gate results are preserved.

**Status:** the score-function chooser and its test ports are implemented. The final ten-worker default sweep completed **5,187 cases: 4,900 passed, 286 skipped, one non-strict XPASS, zero failed assertions** in **773.4 seconds**. Every pytest worker exited zero. The bounded supervisor treats XPASS as failure and returned one, so **the required green sweep has not been achieved and zero §16.3 gate trainings have run**. The [full report](review16-green-sweep/report.html) and [summary](review16-green-sweep/summary.json) retain this distinction.

The sole remaining blocker is `test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`. It passed its unchanged **0.8** overlap assertion while carrying an existing `xfail(strict=False)` mark. Both its body and mark are unchanged from the incoming candidate. The bounded runner and campaign guard are also unchanged. A decision is pending on accepting that named non-strict XPASS, holding for Claude, or retiring its expected-failure mark. **No random rerun, test omission or guard exception has been used.**

## Delivered mechanism

For each sentence with a sampled compose departure, the reconstruction owner receives `p(a_dep | shared prefix) * (C_explore - C_greedy)`, with both costs detached and the existing active-batch mean reduction. A cheaper alternative raises its probability, a dearer one lowers it, and a tie registers no term or optimizer update for the chooser. The §15 strict-win term and its separate owner registration are deleted. Both complete trials are still costed before learning; only strictly lower reconstruction keeps the explore derivation, and ties keep greedy.

The departure round is drawn uniformly from eligible rounds, and its action uniformly from value-distinct eligible alternatives. The proposal is independent of chooser logits; the loss uses the chooser's original softmax probability. Duplicate values, identity-equivalent unaries, and STOP when both leaves already read back exactly are excluded. The greedy prefix is replayed with the same parameters. Greedy compose is argmax and supplies no pathwise chooser gradient. Its selected operator retains the full operand gradient. Detached scorer features keep the surrogate's gradient at chooser parameters and operation anchors.

Pair search remains the §15 hard pick, with detached candidates in the relative residual and no soft blend or temperature. Byte-scoring bank codes remain detached and the recovered leaf live. The generate policy's straight-through walk is unchanged. Perception remains the sole writer of prototypes and evidence. No model dimensions, learning rate, optimizer, budget, seed, gate bar or resource guard changed in this round.

**Estimator scale:** the implemented formula is exactly the requested `p·ΔC`, without multiplying by the number of eligible alternatives. Conditional on a uniform draw among K alternatives, its expectation is `1/K` times the corresponding summed gradient over those alternatives. Uniform round sampling adds its own averaging. The measured derivative below verifies this surrogate; it is not evidence that the unscaled estimator equals a sum over every original policy action and round.

## Mechanism checks on the frozen source

The [ordinary chooser batch](review16-ordinary-frozen.json) uses the original **seed 613** and **["a b c d e", "f g h i j"]**, one real training batch, with **no cost override**. Once ineffective departures are excluded, this batch no longer ties: both selected departures are the value-distinct `non` unary, at rounds 50 and 4. The former §15 observation of exact ties is preserved as an observation of that earlier sampler.

| Row | Departure | C_greedy | C_explore | ΔC | p before → after |
|---|---|---:|---:|---:|---|
| 0 | non | 0.484053940 | 0.177577868 | -0.306476057 | 0.1664561629 → 0.1664604694 |
| 1 | non | 0.180075958 | 0.411828935 | +0.231752977 | 0.1666011512 → 0.1666000783 |

The gradient on all chooser logits agrees with `ΔC·∇p` after the existing two-row mean reduction: maximum absolute error **1.862645149e-9**. Central finite differences on departure logit 20, epsilon **1e-4**, are **−.02126154928** and **+.01608889453**; the corresponding analytical derivatives are **−.02126155049** and **+.01608889550**. Maximum finite-difference error is **1.205640629e-9**. Observed finite logits span **[−.19550702, .09702440]**, with no NaN or positive infinity. Ownership conflicts are zero. This is a mechanism check, not a convergence run.

The authorized chooser test now has explicit equal-cost and cheaper-departure cases, retaining its original seed, batch, setup and optimizer. Those two fixtures control only the detached comparison costs passed to the surrogate; actual trials and the understanding's keep rule still execute. The natural batch above is recorded separately so the test does not claim a tie that the new sampler does not produce. Additional checks cover a dearer departure, exact ties, proportional gradient magnitude, uniform eligible sampling, duplicate-value exclusions, and full operand gradients without greedy chooser gradients. Complete old/new bodies are in the [round contracts](review16-contracts-final.json) and [frozen-source ports](review16-source/test-ports.json).

The [one ordinary XOR batch](review16-mechanism/probe-context.json), collected by a passing wiring test in the final sweep, contains **four departed sentences, zero nonzero advantages**, unchanged departure probabilities, and **zero ownership conflicts**. Its [run audit](review16-mechanism/run-audit.json) records zero sentence-path gradient at perception prototypes and evidence, as well as root/code geometry, support, net-evidence ranges, room reports and one reader epoch. Finite chooser logits span **[.0000356668, .28572097]** in that epoch. The inherited metadata's label “tenth shared gate training” is explicitly corrected by `probe-context.json`: this folder is **one batch, not the tenth 400-epoch gate run**.

The one-batch room audit retains residual violations without hiding or repairing them: stage 1 starts with **20 / .04184762** (count / maximum), then **2 / .03320757** after the first room pass; its end report has **1 / 3.72529e-8** before and **0 / 0** after. These are mechanism observations, not the requested ten-run start/end reports. No room or code-derivation change was made.

## Tests, ports and preservation

All **sixteen original output-gradient regressions pass with their assertions unchanged**. The final sweep includes every file in the [57-file focused list](review16-focused-files.txt), which retains the prior 51 and adds six relevant files. Its extracted subset is **537 passed, 33 skipped, one XPASS**; no focused file was dropped. This subset is read from the full sweep, not another execution.

The first full §16.3 sweep, saved before the final ports, completed **4,898 passed / 286 skipped / one XPASS / two failed**. Both failures encoded the superseded compose straight-through backward rule: probability-scaled operand gradients and chooser-anchor gradients directly from the chosen hard value. They are classified as **ports**, with exact failure messages in the [first-sweep summary](review16-final-sweep/summary.json). Their replacements check full selected-operator gradients and chooser credit exclusively through the score-function term. No remaining assertion failure is concealed as a port or skip.

Earlier failing focused probes and their source snapshots remain under [probes](probes/), [the wider diagnostic](review16-focused-diagnostic/), [the incoming snapshot](review16-before/) and [the pre-sweep snapshot](review16-pre-sweep/). The last eight-file contract probe passed **66 tests** before the full sweep exposed the two additional backward-rule ports. Every narrow probe's request/process command records its file list. The final full sweep covers all of them.

The [frozen source](review16-source/source.zip) preserves **715 files**, **179 complete old/new test ports against published HEAD**, and **zero changed seed calls**. The [round contracts](review16-contracts-final.json) additionally compare complete test bodies with the incoming §15 candidate, verify unchanged gate tests, runner guards and original output assertions, and record capacities 6/8. **2,013 previously archived evidence files** match their earlier manifests in the [preservation check](review16-historical-preservation.json).

Architecture, GradientFlow, Philosophy, Spaces, the accessible-mind spec and the operator catalogue were not rewritten. The 6.8 plan and FutureWork changed externally during this work; their complete observed changes are [preserved separately](review16-external-docs.json). Those external edits include §16.4. This delivery implements the user's §16.3 instruction; it does not implement an inverse/decomposition chooser or change the generate policy on the authority of those file edits.

## Measurement remains pending

The [campaign](review16_campaign.py) refuses to start without the required sweep result and matching frozen source. It is prepared to run sum ×10 first, requiring **10/10**, then ten XOR trainings each consumed by both unchanged bars, then MM_xor ×10. There have been **no gate retries, no tuning and no §16.3 gate trainings**.

The observers are prepared for all requested per-run bands, joint count, named operators and per-word read-back annotations; start/end pairwise word cosines and mean cos(L), root centered singular values and unit-root XOR interaction, code support, `d = relu(e_for-e_against)` ranges, room reports and reader weight trajectories. The actual tenth run would also supply per-sentence/per-step costs, advantage, action and probability before/after; nonzero-advantage count and per-epoch logit range; ownership, decoder margin/gradient and derivation stability. None of those **ten-run** results is claimed here. Reader norms are saved per parameter as well as in aggregate, so a flat aggregate cannot by itself establish a stationary affine head.

The requested advance reading remains a forecast: reconstruction 10/10, MM_xor and sum 10/10, and a majority of class runs near zero, with quarter-error runs interpreted through geometry and the reader trajectory. It has not been confirmed or refuted on this frozen source.

| Historical record | XOR / MM_grammar capacity | XOR class | XOR reconstruction | Joint | MM_xor | Sum |
|---|---|---|---|---|---|---|
| 6.9 closing | accepted source | MSE .1147481948 | 0/4 sentences | — | red through §17 | — |
| §12 | 6 / 8 | 0/10 | 7/10 | 0/10 | 10/10 | 10/10 |
| §13 | 262 / 264 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 |
| §14 before addendum | 262 / 264 | 1/10 | 8/10 | 1/10 | 10/10 | 10/10 |
| §15 | 6 / 8 | not run | not run | not run | not run | not run |
| §16.3 | 6 / 8 | not run | not run | not run | not run | not run |

The accepted baseline retains zero ownership conflicts; prior class **9/10** and reconstruction **5/10** remain historical. Under the two-spaces/one-index reading, this XOR table measures perception's composition of forms, its inverse, the affine read and ownership. Order-zero meanings have identical empty contexts in these fixtures; their coincidence is correct. MM_xor's field path measures connectives where meanings exist. Forms remain the join of positively evidenced parts at full presence; wholes bound them, certainty is activation, and perception owns their parameters. Complement bootstrap remains deferred.

Only this receipt, todo 6.8 and the 6.9 §20.3 status are updated from these results. Unit-sphere codes remain retired, magnitude/certainty remains returned in cube form, the antipode row remains removed, and the distributional row is unchanged. The carried operators work, REBAR/RELAX control variates and compose/generate scorer unification remain deferred. No native benchmark or fresh BasicModel scoring ran; frozen evaluation admits nothing, and the trained NanoChat gate waits for item 4's checkpoint. **Nothing committed. Measurement is held at the XPASS decision; Claude reviews before any commit.**
