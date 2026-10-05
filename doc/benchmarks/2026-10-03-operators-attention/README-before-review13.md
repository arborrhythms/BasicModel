# Decoder, operators and 6.8 — §12 composition mechanism results

This is the one receipt for the uncommitted candidate from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`, in the existing working tree.
The controlling review and reading are [6.8 §§12–12.1](../../plans/2026-09-27-item-6-8-one-attention.md).
**Nothing is committed or pushed. Work stops for Claude's review.**

One measurement on frozen source, thirty trainings and no retries:
**XOR_grammar class 0/10, reconstruction 7/10,
joint 0/10; MM_xor 10/10; sum 10/10.**
All ten controls completed and were read as 10/10 before XOR or MM started.
Below-comparison counts: **class_pass 0/10 versus 2/10, joint 0/10 versus 2/10**. Every result is retained.

| Measure | §11 | §12 |
| --- | --- | --- |
| XOR class | 2/10 | 0/10 |
| XOR reconstruction | 6/10 | 7/10 |
| Both in the same training | 2/10 | 0/10 |
| MM_xor | 10/10 | 10/10 |
| Sum control | 10/10 | 10/10 |

The table is the **composition mechanism gate**: nonlinear composition into
one understanding, free decoding, affine answer reading, and one objective
owner per parameter. It does not test which operation a grammatical
construction means. Supervised grammatical learning is measured by
`MM_grammar_wording`'s compose lesson and bounded wording gate; the reviewed
item-9 evaluation measures the unsupervised form on item 0's future checkpoint.
This round adds no grammar-learning training or benchmark.

The accepted 6.9 baseline is preserved: **MSE .1147481948, reconstruction 0/4,
zero ownership conflicts**. §22's class **9/10** and reconstruction **5/10**
remain historical measurements. MM_xor was red through 6.9 §17, then measured
10/10 with the affine head in §11. The [§11 results](README-before-review12.md)
stand unchanged and were not retried. [§10](README-before-review11.md) and the
[initial round](README-before-review10.md) remain preserved. The original
single-run class comparison remains **not a regression finding**. No conference freeze.

## Implementation and audit

Disjunction computes `(norm(x)+norm(y)-norm(x)*norm(y))*unit(x+y-x*y)`.
The zero direction yields zero; no norm clamp, parameter, capacity or learning
setting is added. Reverse and generate inherit conjunction's free bounded pair
search through the new kernel. `complete.grammar`, XOR_grammar and MM_grammar
(through `default.grammar`) already select this name and now use its new
implementation. The three XML edits change comments only; parsed elements match.

`sum` retains arithmetic mean and its three numerical faces. The direct balanced
inverse recomposes the parent; the free decoder searches both operands in its
primed bank. It remains the additive control. Both binary choices in XOR_grammar
are now nonlinear, restoring the intended condition of 6.9 §3.11. A fixed-example
rank probe checks affine XOR readability; it does not guarantee convergence or
prevent degenerate learned codes. The unchanged bars remain the criteria.

Compose derivation records contain rule ID, operator name, surface alias, arity
and position. Decoder derivations and margin captures name their rules too.
Names come from the model's held catalogue. Every new XOR run captures final
greedy compose derivations at its existing evaluation boundary, with no extra
forward. Raw placement indices in walk stability are explicitly distinguished
from grammar IDs. Diagnostic strings stay outside model state and gradients.

Sampled departures, same-parameter comparisons, strict owner-cost selection,
ties to greedy, affine numeric reading and support-governed decoding remain.
Frozen evaluation admits no definitions or reservations. The small three-item
evaluator mechanism check passed among the focused tests; the existing
[acceptance probe](nanochat_acceptance_probe.py) remains. **Fresh BasicModel
scoring stays stopped; the trained NanoChat gate waits for item 4's checkpoint.**
Item 1 retains its measured 2.6× forward slowdown.

## §11 retrospective — saved observations only

[Extraction with artifact hashes](review12-retrospective.json),
[reader](review12_retrospective.py). No training or forward was run. Names are
resolved from the frozen §11 grammar and implementation.

| §11 run | Saved MSE | Band | Saved final compose |
| --- | --- | --- | --- |
| 1 | 0.250024897 | at 1/4 | not recorded |
| 2 | 0.000021170 | at 0 | not recorded |
| 3 | 0.232321466 | at 1/4 | not recorded |
| 4 | 0.250000087 | at 1/4 | not recorded |
| 5 | 0.199993321 | between | not recorded |
| 6 | 0.250001577 | at 1/4 | not recorded |
| 7 | 0.000189622 | at 0 | not recorded |
| 8 | 0.201953973 | between | not recorded |
| 9 | 0.125663548 | between | not recorded |
| 10 | 0.250078655 | at 1/4 | disjunction (then mean), rule 2, all four sentences |

Run 10 supports the mean hypothesis. Runs 1–9 lack saved final compose
derivations, including four of the five quarter-band runs: the claim that all
five settled on mean is **unconfirmed**. Missing operators are not inferred
from MSE. Run 3's .232321 is in the unchanged quarter band, not exactly at the
additive floor. Run 10's final greedy evaluation used mean on all four inputs;
its training audit also contains mixed derivations.

## One frozen-source measurement

[Plan](review12-measurements/plan.json), [completion](review12-measurements/complete.json),
[controls read before gates](review12-measurements/sum-read-first.json),
[per-run summaries](review12-measurements/summary.json), [source](review12-source/manifest.json).

Each XOR model trains once for 400 epochs and supplies both unchanged tests;
the observer checks the same model identity for both consumers. Class requires
four correct labels and MSE < .05. Reconstruction requires all four word
multisets and no unavailable inverse; the old selector containing `50_pct` is
unchanged. §20.5 bands: at 0, MSE < .05; at ¼, abs(MSE−.25) ≤ .02; remaining
errors between or above ¼. Bands do not replace bars.

Compose names are in input order **hello world / hello there / loving world /
loving there**; `×4` means the same whole sequence for all four. Per-run JSON
retains each rule ID, name, arity, position and word row.

| Run | MSE | Band | Labels /4 | Read-back /4 | Class | Reconstruction | Joint | Final greedy compose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.174577616 | between | 4 | 2 | fail | fail | fail | conjunction ×4 |
| 2 | 0.179248216 | between | 4 | 4 | fail | pass | fail | conjunction ×4 |
| 3 | 0.179517262 | between | 4 | 4 | fail | pass | fail | conjunction / conjunction / disjunction / disjunction |
| 4 | 0.242915508 | at 1/4 | 3 | 4 | fail | pass | fail | disjunction / disjunction / conjunction / conjunction |
| 5 | 0.218240400 | between | 4 | 4 | fail | pass | fail | disjunction / disjunction / conjunction / conjunction |
| 6 | 0.136760605 | between | 3 | 4 | fail | pass | fail | conjunction / conjunction / disjunction / disjunction |
| 7 | 0.196491267 | between | 4 | 4 | fail | pass | fail | conjunction ×4 |
| 8 | 0.170682153 | between | 4 | 2 | fail | fail | fail | conjunction ×4 |
| 9 | 0.185242474 | between | 4 | 2 | fail | fail | fail | conjunction ×4 |
| 10 | 0.228597979 | between | 4 | 4 | fail | pass | fail | conjunction ×4 |

Bands: **{"at 0": 0, "at 1/4": 1, "between": 9, "above 1/4": 0}**. Joint: **0/10**.

MM uses its unchanged best-MSE < .20 convergence test, up to 200 calls.
Sum uses the original harness, substituting only `sum.forward(S,S)` for the
three compose rules. Its unchanged control is absolute checkerboard contrast
≤ 1e-4 and failure of the class bar. All ten final sum derivations are saved too.

| Run | MM best MSE | MM calls | MM bar | Sum MSE | Sum contrast | Sum bar |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.174940974 | 25 | pass | 0.250000417 | -2.98023224e-08 | pass |
| 2 | 0.173332006 | 35 | pass | 0.250002623 | 0 | pass |
| 3 | 0.182707667 | 41 | pass | 0.250000328 | -2.98023224e-08 | pass |
| 4 | 0.188894108 | 40 | pass | 0.250013292 | 0 | pass |
| 5 | 0.170454144 | 51 | pass | 0.250001431 | 0 | pass |
| 6 | 0.185661137 | 31 | pass | 0.250000179 | -2.98023224e-08 | pass |
| 7 | 0.183500662 | 32 | pass | 0.250000000 | 0 | pass |
| 8 | 0.194120497 | 26 | pass | 0.250001490 | 0 | pass |
| 9 | 0.197054893 | 40 | pass | 0.250002682 | 5.96046448e-08 | pass |
| 10 | 0.172300309 | 114 | pass | 0.250000536 | 5.96046448e-08 | pass |

## Tenth-run ownership, margins and walk stability

[Full audit](review12-measurements/xor-10/ownership/),
[summary](review12-measurements/audit-summary.json), [plot](review12-measurements/decoder-margin.png).
Ownership conflicts: **0**; 23
active and 64 inactive parameters over
1200 backwards. There are 1600
batched first-logit captures and 1200 optimizer steps;
800 steps reach decoder graphs.
First-step row/path eligibility: **{"compound": 6400}**.

| Undo | Raw margin mean, epoch 1 → 400 | Final range | Nonzero gradient differences | Nonzero fixed-parent changes | Mean fixed-parent change |
| --- | --- | --- | --- | --- | --- |
| conjunction (rule 1) | 1.992617 → 1.955212 | 1.918829–2.017436 | 755/6400 | 6214/6400 | -6.09242916e-05 |
| disjunction (rule 2) | 1.985850 → 2.038061 | 2.015598–2.058290 | 755/6400 | 6220/6400 | 6.09261915e-05 |

Gradient difference means `dL/dSTOP − dL/dundo`. Largest absolute STOP gradient:
**0**; gradient difference: **0.00375378039**;
fixed-parent margin change: **0.000302553177**. Changes come
from the actual optimizer update with the parent held fixed. Raw epoch margins
can also change with the parent; discarded paths may have zero gradient.
Masked actions remain in the audit.

| Walk | Comparisons | Explore kept | Strict violations | Kept-path stability |
| --- | --- | --- | --- | --- |
| attention.input | 1600 | 0 | 0 | 1584/1596 = 0.992481 |
| generate.decoder | 3200 | 78 | 0 | 30/3196 = 0.009387 |
| compose | 1600 | 4 | 0 | 1592/1596 = 0.997494 |

The same tenth training supplies both bars and every audit. No attribution
training, full sweep or native run was added. Observed associations are not
assigned as causes of a failed class or decoding bar.

## Verification and complete ports

[Review-start archive](review12-before/manifest.json), [contracts](review12-before-measurement-contracts.json),
[probe/source bridge](review12-source/probe-source-bridge.json). There are
**165 complete old/new published-HEAD test ports** and **zero changed seed calls**.
Existing assertions in this round's ports are unchanged. Protected XOR and MM
tests, guards, Makefile, pytest.ini and NanoChat manifest match HEAD byte for byte.

| Saved probe | Result and disposition |
|---|---|
| [before-operator-and-audit](probes/review12-before-operator-and-audit/run.log) | 9 fail, 1 pass: old semantics, missing audit helper, and a probe that had not followed MM_grammar's grammar-file reference |
| [formula-and-old-port](probes/review12-formula-and-old-port/run.log) | 18 pass; old mean expectation fails, saved before its parameter-data port |
| [focused](probes/review12-focused/run.log) | 141 pass, 1 fail, 1 skip; synthetic margin fixture lacked new rule metadata |
| [metadata-port](probes/review12-metadata-port/run.log) | 5 pass, including unchanged margin assertions and compiled exploration fixtures |
| [observer](probes/review12-observer/run.log) | Failed before updates: the trial record is local before commit; saved before passing it explicitly |
| [observer-repaired](probes/review12-observer-repaired/run.log) | Failed diagnostic serialization after one batch: placement indices mistaken for rule IDs; saved before annotation repair |
| [observer-final](probes/review12-observer-final/run.log) | One ordinary batch and evaluation pass: four margin captures, three owned steps, zero conflicts, four named final derivations |

Final focused coverage is **142 distinct passing checks and one existing skip**,
plus the observer mechanism check. Bounded mechanism probes are separate from
gate trainings. Every failing probe and source archive precedes its repair.
Seeds, bars, existing assertions, optimizers, learning rates, budgets and guards
are unchanged. Earlier failures and complete ports remain in archived receipts.

All thirty measured processes retain output, exit, guards and final arrays.
Hashes are checked throughout; no measured model was repaired or retried.
Elapsed: **862.577 seconds**; largest worker:
**679,216,640 bytes**. Guards: **8 GiB / 1,800 seconds per worker**, at most three
one-thread workers reserving **24 GiB**, within the standing **28 GiB** ceiling
and CPU headroom. One actual working tree; no commits or pushes.

[Final delivered source and ports](review12-final/manifest.json),
[final source bridge](review12-final-bridge.json), [final contracts](review12-contracts-final.json).
