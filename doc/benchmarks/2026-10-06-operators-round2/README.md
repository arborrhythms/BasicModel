# Operators update, round 2 — credit

**Status: standing gate failed; frozen, measured candidate held for Claude’s review. No round-2 commit or push.** Round 1 was accepted on 2026-10-06 and landed as `73cd7b71baeb64135c1e2e841b9bfc321339d8bc`; WikiOracle records it in `c9670b545ff88ee1b8176aa44f6c192ad84b597c`.

All thirty fresh, unseeded gate trainings completed without retry, replacement, seed selection or tuning. The delivered source and measurement helpers stayed unchanged throughout. Development checks and both earlier candidate sweeps are retained under `development/`. Historical document snapshots retain their complete bytes as `.md.txt` so their relative links are not treated as live documentation. Clyde’s completed document state is saved under `before/finished-docs/`; the plan is not edited.

The candidate compares owner-step reconstruction, gated expectation and supplied-answer errors, makes one departure across narrowing and compose, and hands the narrowed poles into the sentence. The concept-only closing image is implemented; its complement width is zero in the two grammar configurations.

The unchanged MM_xor convergence test supplies its target after a parallel raw forward and uses an external optimizer. It bypasses the sentence leaf consumer and owner step, so published narrowing poles alone do not establish live credit. The observer records actual consumer calls and owner-step comparisons. This coverage limitation is recorded separately from the standing convergence result; the requested claim that MM_xor tests joint departure credit cannot be made on that unchanged caller.

## Delivered implementation

- Each sentence trial exposes relative reconstruction `R`, role/presence/kind expectation `E` gated by detached `g·κ`, and supplied-answer `A`. Only a strictly smaller total keeps explore. Predictor training stays ungated; the reader trains only on kept rows. Comparison values and advantage are detached.
- The one departure is uniform over the eligible rounds of narrowing and compose. Its eligible alternative is uniform, and the existing `reconstruction.compose_score_function` registration receives `K·R·p(a)·detach(C_explore−C_greedy)` with its original reduction. No separate attention term remains.
- The narrowed pair reaches the word evidence and closing polarity. Field extents survive splitting, and their operations resolve in walk order over the completed native witnesses. This is necessary when a field operation precedes an unknown word's identifying descent; retaining its earlier `(0,0)` erased newly identified evidence. Eligibility, local reads and the walk budget are unchanged. Values remain local to the walk.
- The closing image subtracts only on the concept complement. Thought reads the conceived value; storage and detached targets keep the observation. Both grammar fixtures have fourteen form-content coordinates in a twenty-two-coordinate carrier and zero concept complement: their image is exactly zero. The nonzero fixture checks the six cases, zero confidence, protected form/address coordinates and restoration.

Pair search, decomposition teachers, reconstruction objectives, owner lists, gate selectors, budgets and §20.5 bands remain unchanged. The unchanged MM caller is the coverage limitation described above.

## Freeze and verification

The [source manifest](delivered-source/source.json) covers 700 files. Its SHA-256 is `63d68a9b9c38d14470293e12e7d7e85103c24bfa7572d605babafed0c7e7043a`. The [source archive](delivered-source/source.zip), [diff at freeze](delivered-source/changes.patch), [measurement-helper hashes](delivered-source/measurement-helpers.json), [complete old/new test texts](delivered-source/test-ports.json), and [seed-call audit](delivered-source/seed-port-audit.json) are frozen together. Eleven test ports include the prior and replacement audit probe. No existing seed call changed, and the campaign sets no seed.

The delivered-source [full sweep](full-sweep/result.json) completed 5,247/5,247 selected cases in 219.9 seconds: 4,961 passed, 285 skipped, one non-strict expected failure passed, no failures. Normal slow exclusions and the weekly-record warning remain in the raw result. The focused round-2 mechanism check passed 21 cases; the audit/ownership check passed 15. The affected output and receipt-link check passed 393 cases.

The first development sweep found the pre-descent pole defect, snapshot-relative links, and a reconstruction registry assertion that omitted the shared SCG term. The second found only the compose-pair assertion that incorrectly required a compose departure on every row. Both complete runs, their frozen sources and failures are retained under `development/candidate-01/` and `development/candidate-02/`. The final test port checks that exactly one walk departs, that compose's replay prefix is preserved when applicable, and that the actual selected walk changes. These development repairs preceded all thirty gate trainings.

The [documentation diff from Clyde's completed state](documentation-from-finished.patch) separates these edits from his work. Philosophy and the plan match their saved finished hashes. Source and helper hash matching is enforced before and throughout the campaign.

The [review patch](review-changes.patch) combines runtime and test changes with
only that documentation delta, excluding Clyde's preexisting edits. The
[review state](review-state.json) records the hashes and empty Git index.
The final [documentation-link check](final-doc-links.log) passed 294 cases.


## Gate and thirty trainings

| Gate | Accepted 6.8 | Round 2 | Result |
| --- | ---: | ---: | --- |
| XOR class | 7/10 | **4/10** | below landing |
| XOR reconstruction | 9/10 | **6/10** | below landing |
| XOR joint (reported) | 6/10 | 4/10 | below landing |
| Sum control | 10/10 | 10/10 | retained |
| MM_xor convergence | 10/10 | **9/10** | numerical miss; coverage gap below |
| Full sweep | green | green | retained |
| Sentence-path gradient at perception | 0 | 0 | retained |
| Code displacement | 0 | 0 | retained |
| Ownership conflicts | 0 | 0 | retained |

The [result validation](results-validation.json), [measurement summary](measurements/summary.json), and [tenth-run audit summary](measurements/audit-summary.json) read saved observations only. All 30 processes completed in 1,083.9 seconds. The [campaign plan](measurements/plan.json) preserves ten 400-epoch sum controls, ten 400-epoch XOR trainings, and ten MM trainings at at most 200 epochs. Each XOR model supplies both original bars; the class measurement reads its committed greedy root, without blending trials. No gate assertion, budget or threshold was ported. The sum group passed 10/10 before XOR or MM began.

The §20.5 bands are unchanged: at zero is MSE < .05; at quarter is |MSE−.25| ≤ .02; the remaining values below .25 are between, and those above are above quarter. XOR: **4 at zero, 2 at quarter, 2 between, 2 above quarter**. All ten sum controls are at quarter.

| XOR run | MSE | Band | Correct labels | Reconstructed sentences | Final greedy operator | Class / reconstruction |
| --- | ---: | --- | ---: | ---: | --- | --- |
| 01 | 0.4036407142 | above 1/4 | 2/4 | 0/4 | disjunction | miss / miss |
| 02 | 2.355893258e-13 | at 0 | 4/4 | 4/4 | conjunction | pass / pass |
| 03 | 2.577493774e-12 | at 0 | 4/4 | 4/4 | conjunction | pass / pass |
| 04 | 0.2582679013 | at 1/4 | 2/4 | 0/4 | disjunction | miss / miss |
| 05 | 0.244562657 | at 1/4 | 2/4 | 0/4 | disjunction | miss / miss |
| 06 | 0.0910004125 | between | 4/4 | 4/4 | conjunction | miss / pass |
| 07 | 0.2961553703 | above 1/4 | 2/4 | 0/4 | disjunction | miss / miss |
| 08 | 0.01236373059 | at 0 | 4/4 | 4/4 | conjunction | pass / pass |
| 09 | 0.1040427383 | between | 4/4 | 4/4 | conjunction | miss / pass |
| 10 | 0.0006547981262 | at 0 | 4/4 | 4/4 | conjunction | pass / pass |

The six runs ending with conjunction recovered all four sentences; four met the class bar. Runs 6 and 9 had four correct labels but missed its unchanged MSE bound. The four runs ending with disjunction recovered no complete sentences, returning one word from each pair. Their complete texts, losses and failures remain in each run directory. These observations locate the misses; they do not justify selecting another run or adjusting the training.

## Credit and ownership audit

Across the ten XOR runs, **5,426 of 16,000 sentence records had nonzero advantage**. The [aggregate credit audit](aggregate-credit-audit.json) records 682 `narrowing:not`, 392 `narrowing:descend`, 1,378 `compose:conjunction`, and 2,974 `compose:disjunction` nonzero cases. All other walk/action combinations had zero nonzero-advantage cases. Sum has 16,000 records with zero nonzero advantages, as expected for its single-operation control.

Every trial record contains `[R,E,A]`, both totals, the signed component differences, the strict keep, and the deciding components. A deciding component is one whose signed difference supports the selected total direction; ties are `tie:greedy`. Across XOR the decisions were 3,676 answer, 1,628 reconstruction plus answer, 122 reconstruction, and 10,574 ties. The expectation component was exactly zero in these gate trials; this campaign therefore exercises the added answer cost and narrowed evidence, while the gated expectation mechanism is checked by its focused tests.

The [tenth-run audit](measurements/xor-10/run-audit.json) contains 1,600 sentence records and 554 nonzero advantages: 36 narrowing `not`, 2 narrowing `descend`, 507 compose `disjunction`, and 9 compose `conjunction`. The maximum analytic gradient discrepancy is `2.384185791015625e-7`; maximum finite-difference discrepancy is `3.043895722143475e-7`, over both walks. Per-epoch logit ranges are saved for every run and both walks, and they move. No separate attention registration remains.

Each run's `run-audit.json` includes `final_committed_training_operators`, with the kept derivation, both trials' components and the cost that selected it. `final_greedy_compose` in the gate observations separately records the actual evaluated derivation and its available evaluation cost. Evaluation has no supplied-answer selection. In run 1's final training step, rows 0 and 3 kept conjunction by total cost while greedy evaluation still chose disjunction: the new comparison can see the better answer, but this training did not make the final greedy policy select it. These are recorded results, not a proposed cause or a waiver of the gate.

Zero sentence-path gradients and zero code displacement hold in all twenty XOR/sum trainings. The tenth-run ownership audit records zero conflicts. Both grammar mechanism fixtures report concept-complement width zero and image zero; every recorded XOR/sum closing agrees. No new loss, optimizer or owner was introduced.

## MM coverage limitation

| MM run | Best MSE | Epochs used | .20 bar |
| --- | ---: | ---: | --- |
| 01 | 0.1856247783 | 62 | pass |
| 02 | 0.1995426565 | 44 | pass |
| 03 | 0.1883889139 | 34 | pass |
| 04 | 0.2372350693 | 200 | miss |
| 05 | 0.1403090209 | 54 | pass |
| 06 | 0.1880542338 | 27 | pass |
| 07 | 0.171446532 | 50 | pass |
| 08 | 0.1694803536 | 25 | pass |
| 09 | 0.153709203 | 106 | pass |
| 10 | 0.1794308424 | 99 | pass |

MM run 4 exhausted 200 epochs at best MSE `0.23723506927490234`; it was not retried. All ten saved MM trajectories include every epoch's prediction and MSE.

**The hand-off's requirement that MM_xor be live for joint credit is not fulfilled by this candidate's unchanged MM configuration and caller.** The [convergence test](../../../test/test_mm_xor.py) constructs an external Adam optimizer, calls `forward(inp)`, then supplies its target to MSE. MM_xor is parallel (`word_brackets=False`), while `_sentence_run` enables sentence endings only for word brackets. It therefore does not enter the sentence owner-step comparison or consume the sentence leaf evidence. Each of the ten observers records **zero owner-step comparisons and zero leaf-evidence consumer calls**. Publishing `_attention_poles` is insufficient to establish a live mechanism.

Whether a same-initialization MM trajectory differs from the landing is **not established**. These are independent unseeded runs, not a paired replay, and no extra baseline or replacement training was added. The observed convergence miss remains in the table. Its result cannot be presented as a regression test of the new joint credit; neither is trajectory identity proved here. Resolving the parallel path and target/owner-step contract requires review before changing the specified caller, configuration or owner behavior. An API-only test port would not resolve the parallel-path bypass.

The plan is Claude's and remains unchanged. Round 1 is landed and pushed; round 2, its failed gate and this unresolved coverage requirement are held for Claude's review before any commit.
