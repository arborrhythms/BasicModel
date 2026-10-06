# Operators update, round 1 — accepted 2026-10-06

**Accepted by Alec on 2026-10-06; commit, push and the WikiOracle submodule bump authorized.** The measured source is unchanged. Claude’s plan §5 establishes trajectory identity with the 6.8 landing on MM_xor for 33/33 checked seeds; Alec accepted the round with those findings carried forward. The recorded MM_xor 9/10 is retained. Under the accepted amendment, a trajectory-identical gate is not a regression test of this round.

[Acceptance record](acceptance.json). The candidate assessment and measurements below are preserved as the pre-acceptance record.

2026-10-05. **Do not commit this candidate: MM_xor is 9/10 against the required 10/10.** Round 1 is implemented and the requested single thirty-training campaign is complete. The measured source remains uncommitted for review. The cause of the MM miss is not established by these observations; it is not waived as random variation and no replacement run was taken.

| Gate | Accepted 6.8 (`42daf96f4`) | Round 1 | Result |
| --- | ---: | ---: | --- |
| XOR class | 7/10 | 8/10 | above landing |
| XOR reconstruction | 9/10 | 10/10 | above landing |
| XOR joint (reported) | 6/10 | 8/10 | above landing |
| Sum control | 10/10 | 10/10 | retained |
| MM_xor | 10/10 | **9/10** | **fails standing gate** |
| Full sweep | green | green | retained |
| Sentence-path gradient at perception | 0 | 0 | retained |
| Code displacement | 0 | 0 | retained |
| Ownership conflicts | 0 | 0 | retained |

The [result validation](results-validation.json), [measurement summary](measurements/summary.json), and [audit summary](measurements/audit-summary.json) are computed solely from saved observations. All 30 jobs completed, without retry, in 999.6 seconds. Source and frozen measurement-helper hashes match the sweep and every training.

## Delivered changes

1. Input attention uses compose's score-function surrogate, `K·R·p(a)·detach(C_explore−C_greedy)`, with a uniform legal non-greedy departure, the greedy cost as baseline, detached features and a detached sentence handoff. Reconstruction consumes it once. There is no pathwise attention credit.
2. A pair whose squared relative residual is at most `(8·finfo(dtype).eps)²` wins at the residual argmin before context. Activation coefficients are standardized over each valid shortlist. The generate policy learns undo/unary/STOP by teacher-forced CE on detached states from the actual compose tree; free inference uses no teacher and has no straight-through transition. The [saved §22 miss](review22-miss.json) recomposes correctly even under extreme context weights in the [focused test](../../../test/test_operators_round1.py).
3. `not` exchanges poles and `non` clears the expressed pole; both preserve code identity. Explicit-pole conjunction is `(min positive, max negative)`. The binding kernel retains the accepted form composition. The landing's explicit-pole `NonLayer` already zeroed the expressed pole; that historical defect was not reproduced. Its code-zeroing face was removed. [Saved old/new bodies](pole-bodies.json) are executed by the [body regression test](../../../test/test_operator_pole_bodies.py).
4. Loaded rules declare form/meaning/pole reads and writes, checked against the implementation contract and declared subsystem effects. Missing, extra, duplicate or unknown footprint components fail at load. Aliases retain the same contract.
5. Coded intersection, including its butterfly pair, uses exact silence-preserving min. Verb/adverb chart transforms add a zero-initialized learned translation, allowing a silent coordinate to become expressed and retaining an inverse given the modifier. Gain-only checkpoints load with zero translation.

Two integration repairs accompany the required semantics. Free code decoding rejects identity unary steps that cannot make progress. Preserving `non`'s code exposed a silent leading role hidden by the initial truncated-identity answer reader. New readers initialize a folded identity over all coordinates with unit-norm columns and no RNG consumption; saved factors load unchanged. The original native output assertions pass. This reader change is additional scope made necessary by the output regression and should be reviewed with the pole changes.

[Catalogue §12.1](../../specs/2026-09-29-operator-catalogue.md#121-operators-update-round-1-2026-10-05-review-candidate) records the implementation; [GradientFlow](../../GradientFlow.md#operators-update-round-1-october-5) specifies the estimators and ownership.

## Sweep and provenance

The delivered-source [full sweep](full-sweep/result.json) completed 5,225/5,225 selected cases in 213.7 seconds: 4,939 passed, 285 skipped, one non-strict expected failure passed, zero failures. The canonical sweep retains the landing's normal slow-test exclusions. Its warning about the separate weekly slow-test record is retained in the raw result.

The [source manifest](delivered-source/source.json), [source archive](delivered-source/source.zip), [diff at freeze](delivered-source/changes.patch), [measurement-helper hashes](delivered-source/measurement-helpers.json), [test ports](delivered-source/test-ports.json), and [unchanged-seed audit](delivered-source/seed-port-audit.json) identify the candidate. The [final review patch](review-changes.patch) includes the implementation, both new tests, and completed GradientFlow/catalogue text, excluding the preexisting user edits. Thirteen existing mechanism-test files were ported for removed straight-through/sign semantics and the extended audit. Complete old/new texts are saved. The class, reconstruction, MM_xor and native output regression assertions remain unchanged. Existing user edits in `doc/FutureWork.md`, `todo.md`, and the operators plan were left alone. The [final documentation-link check](final-doc-links.log) passed 293 tests.

An initial sweep was stopped for repairs; its [result](initial-sweep/result.json) and [source](initial-source/source.json) are preserved. It found the native output failures, an attention straight-through assertion still needing its port, and links to the then-missing README. Pre-repair inference-only ablations are saved as `diagnose-output-*.json` with their [driver](diagnose_output.py). The initial identity-unary hypothesis alone did not fix the output failure; the readout repair did. The [focused repair run](repair-readout.log) passed 364 tests with 23 skips; the [closing observer/body checks](closing-observer.log) passed 16. The final sweep followed these repairs, then the thirty trainings ran on that immutable source.

## Thirty trainings and §20.5 bands

The [campaign](campaign.py) preserves selectors and budgets: ten sum controls at 400 epochs, ten XOR trainings at 400 epochs with class and reconstruction consuming one shared trained fixture, then ten MM_xor trainings at at most 200 epochs. It sets no seed and allows no retry. Sum passed 10/10 before either other group began. Guards remain 8 GiB and 1,800 seconds per worker, at most three workers, and a 24 GiB aggregate reservation. The [plan](measurements/plan.json), [completion record](measurements/complete.json), process logs and reports retain every run, including failures.

Bands are unchanged: “at 0” means MSE < .05; “at ¼” means |MSE − .25| ≤ .02; “between” is the remaining MSE below .25; “above ¼” is the remaining MSE above .25. Class also requires four correct rows. Reconstruction requires all four sentence word multisets and availability. Sum requires checkerboard contrast ≤ 1e−4 in magnitude and a failed class bar. MM_xor requires best MSE < .20.

XOR has eight runs at 0, two between, none at ¼ and none above ¼. All ten classify all four rows correctly; runs 4 and 6 fail the MSE threshold. Their disjunction results are **between**, not rounded to the affine floor. Mixed conjunction/disjunction runs 8 and 10 reach the at-0 band. This is a measured association, not a causal attribution to one change in this round.

| XOR run | Class MSE | Band | Final operators | Class bar | Reconstructed |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 0.00159491134 | at 0 | conjunction | pass | 4/4 |
| 2 | 0.000219919751 | at 0 | conjunction | pass | 4/4 |
| 3 | 5.06514381e-05 | at 0 | conjunction | pass | 4/4 |
| 4 | 0.119316555 | between | disjunction | miss | 4/4 |
| 5 | 2.30188941e-06 | at 0 | conjunction | pass | 4/4 |
| 6 | 0.165190104 | between | disjunction | miss | 4/4 |
| 7 | 0.000130611967 | at 0 | conjunction | pass | 4/4 |
| 8 | 9.78343722e-07 | at 0 | conjunction, disjunction | pass | 4/4 |
| 9 | 0.000815493404 | at 0 | conjunction | pass | 4/4 |
| 10 | 3.39974449e-09 | at 0 | conjunction, disjunction | pass | 4/4 |

Every final reconstruction contains the correct “hello there”; no “hello hello” miss recurred. All 40 sentence multisets are recovered. The saved §22 root also passes the separate exact-argmin replay, which isolates the requested precedence from new random initializations.

All ten sum controls are at ¼: MSE ranges from 0.250008374 to 0.26018098; maximum absolute checkerboard contrast is 1.1920929e-07.

MM_xor's final endpoint bands are nine between and one at ¼. All ten best MSEs are in the between band; the separate unchanged .20 bar determines pass/fail.

| MM_xor run | Epochs | Best MSE | Final MSE | Final band | Bar |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 90 | 0.126218945 | 0.126218945 | between | pass |
| 2 | 77 | 0.182555825 | 0.182555825 | between | pass |
| 3 | 183 | 0.18927671 | 0.18927671 | between | pass |
| 4 | 77 | 0.177330047 | 0.177330047 | between | pass |
| 5 | 200 | 0.220072806 | 0.250018835 | at ¼ | **miss** |
| 6 | 35 | 0.198455378 | 0.198455378 | between | pass |
| 7 | 63 | 0.196787685 | 0.196787685 | between | pass |
| 8 | 31 | 0.193646938 | 0.193646938 | between | pass |
| 9 | 79 | 0.167931557 | 0.167931557 | between | pass |
| 10 | 77 | 0.185210049 | 0.185210049 | between | pass |

[MM_xor run 5](measurements/mm-05/run.log) exhausted 200 epochs, with best MSE 0.220072806 and final MSE 0.250018835. It is the sole count below the landing and therefore blocks acceptance. These saved endpoints do not identify its cause.

## Tenth-run audit

The [raw run audit](measurements/xor-10/run-audit.json) extends the accepted observers with attention and teacher records; [ownership events](measurements/xor-10/ownership/events.jsonl) and the [ownership result](measurements/xor-10/ownership/ownership.json) retain the original observations.

* Attention: 1,600 records and departures, each with greedy/explore costs, action, K and R, probability before/after, analytic gradient and finite difference. **All advantages are zero** in this gate run, so it supplies no nonzero attention learning signal. Maximum analytic and finite-difference errors are zero. Nonzero positive/negative advantage behavior is covered by the focused analytic and finite-difference tests.
* Compose: 1,600 records and departures, also all zero advantages in this tenth run; analytic and finite-difference errors are zero. The audit does not claim score-function learning where the paired byte costs tie.
* Walk teacher: 3,200 records, maximum gradient error against CE **7.45e−9**. All 3,200 fixed-state records show a policy-logit change across their actual owner step; maximum absolute change is 0.0145463. The free decoder's 1,600 first-step records receive no pathwise gradient across the 1,200 owner steps.
* Pair chooser: all four true pairs are present in the shortlist; no targets are absent. The retained ordered-pair metric is 2/4 at both endpoints, while the final reconstruction bar is 4/4 word multisets. Exact-fit precedence does not impose source order on a commutative pair.
* All ten XOR and ten sum audits show **zero code displacement and zero sentence-path gradient at perception**. The tenth-run ownership audit has **zero conflicts**.

The [saved-results validator](validate_results.py) imports no model and performs no additional forward or training. Its own hash is recorded in [results-validation.json](results-validation.json), separately from the frozen measurement helpers.

Rounds 2–4 remain deferred. Round 1 awaits review with a failed standing gate. No files were staged and no commit was made.
