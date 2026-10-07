# Item 6.5: definedness, identity columns and change columns

**Accepted by Alec as the item 6.5 mechanism landing, October 7.** Claude
reviewed the §9 choices in [spec §10](../../specs/2026-09-26-independent-components.md#10-review-of-the-october-7-candidate-claude-2026-10-07).
The [acceptance record](acceptance.json) authorizes commit, push and the
WikiOracle bump, including the source, tests, specification, receipt,
[thought-loop plan](../../plans/2026-10-07-thought-loop.md) and current document edits.
Learning acceptance is deferred as stated below. The baseline is
`f4a68404ea8d4704bfeb6279a738c93356d289d6`, the accepted operators repair.
The [protocol](measurement-protocol.json) was
recorded before measurement. All standing trainings use the same frozen source
with no selected seed, retry or replacement.

Plan §44 came first. Words and stored sentence identities use a unit
direction scaled by `n/(n+4)`. Row re-witnesses and containing-row counts
have separate meanings; forms, evidence poles and trust stay separate.
The exact convergence error is `4/(n+4)`. The original magnitude failures
and their affected checks remain in this directory.

The implementation adds recurrence-gated native identity columns, sparse
coding over the retained LTM population, one-sparse change columns, fresh
source encodings and bind/mint variants in the global grammar softmax.
Only a selected mint can advance individual admission. Kind/extension
readings choose no member; bind can re-witness an existing column. Change
observations leave the population when their LTM occurrences leave, while
learned columns remain for item 5's pruning decision.

The [implementation proposal](../../specs/2026-09-26-independent-components.md#9-implementation-proposal-october-7)
records the three choices accepted in spec §10. The implemented objective is an amortized
Laplace sparse-coding energy, not exact marginal likelihood or the square
Infomax determinant. At the accepted sentence update boundary, the latest
source is freshly encoded through live columns; the previous compose graph
is not retained across an update. The bounded situation continues to carry
its existing detached role view alongside its native occurrence anchor.
The role-view duplication is transitional and remains a follow-up.

## Validation

The final [full receipt](retained-sweep-summary.json) is green: **5,527/5,527
selected cases completed**, with **5,241 passed, 285 skipped and one existing
non-strict XPASS**, in 187 seconds. The XPASS is the unchanged
`test_topk_recovered_words_overlap_input` marker. The 25 component mechanism
[cases](mechanism-results.json) pass. The latest [focused run](population-after.log) passes **55 checks**,
including both preserved admission/population probes, native checkpoint round
trips, optimizer ownership, compiled reference selection, grammar scopes and
relative-clause source ceilings.

The [718-file measured source](measured-source/source.json) has manifest SHA256
`d8f374317ea8054cdb9e99077ccdf35df76866e28d6bf46492903f3f4948b099`.
[Source archive](measured-source/source.zip),
[measurement helper archive](measured-source/measurement-helpers.zip).
The final CPU regression run sets `MODEL_COMPILE=none` for automatic model
compilation. Explicit full-graph and compiled-boundary tests select their own
backends and remain in the suite. [Execution declaration](measured-source/execution.json).

| Earlier receipt | Outcome and source |
| --- | --- |
| [Initial sweep](full-sweep/result.json) | Interrupted diagnostic, 5,469/5,519 completed; 40 failures retained. [Initial source](first-source/source.json). |
| [Second sweep](green-sweep/result.json) | Interrupted diagnostic, 5,479/5,522 completed; one legacy identity assertion failed. Its directory name is historical, not a passing status. [Second source](second-source/source.json). |
| [Pre-population-repair sweep](final-sweep/result.json) | Complete and green, 5,526/5,526. [That source](before-population-source/source.json). It predates the retained-innovation fix. |

Failures exposed tensor-only checkpoint metadata, omitted optimizer ownership,
old local-binding fixtures, an unresolved dictionary alias assumption, kind-only
admission and orphaned innovation history. Original test bodies are preserved
in [the port archive](ported-tests-before.json). The admission and population
bugs each have a saved before/after probe. The automatic-compilation diagnostics
are retained; neither interrupted sweep is called green.

## Standing measurements

The [thirty-run campaign](measurements/summary.json) completed on the frozen
source in 1,252 seconds. Each XOR training supplied both unchanged gates.
Every run is preserved; no trained result was retried or replaced.

| Gate | Result |
| --- | --- |
| Sum negative control | **10/10**, every final MSE exactly 0.25 |
| XOR classification | **10/10**, MSE 1.78×10⁻¹⁵ to 4.34×10⁻¹³ |
| XOR reconstruction | **10/10**, all four word multisets in every run |
| Joint XOR class/reconstruction | **10/10** from the same ten trainings |
| MM_xor convergence | **10/10**, best MSE 0.17312–0.19922 within the original 200-epoch limit |

These counts match the accepted final-b comparison. Reconstruction and
expectation costs are **identically zero across 64,160 observed grammar
trial rows**, including training and final evaluation.
[Source, gate-file and cost integrity](measurements/integrity-and-costs.json).
The tenth XOR run reports **zero ownership conflicts over 800 training
backwards**; it is reused for the audit, with no extra training.
[Ownership audit](measurements/audit-summary.json).

The older [required XOR table](xor-table-default-sweep.json) has six grounded
cases and three native curriculum cases passing in the default sweep. The
four additional slow cases also **pass 4/4**, each run once under its
[separate protocol](supplemental-slow-protocol.json), after the thirty-run
campaign. [Results and predictions](supplemental-slow/summary.json).
Both XOR_exact CLI runs report 4/4 reconstruction and output MSE 0.0 at
the CLI's reporting precision. The older MM signal case records .25003
against its .26 affine-floor bar; MM_grammar records .18614 against .20.
Those loose MSE bars do not establish all-four classification. These four
runs are separate from the thirty standing trainings above.

Ten [initial launcher failures](startup-failures/summary.json) occurred before
the training function or model construction: two audit helper dependencies
had not been copied. Their logs and original helper manifest are preserved.
The unchanged dependencies were restored from the accepted receipt, and an
[observer preflight](observer-preflight.log) passed with model construction
and training forbidden. No trained result was discarded or retried. Model,
configuration and test source did not change during this harness repair.

## Learning limits

[Checkpoint metadata](learning-readiness.json) supplies no checkpoint with
one million completed FineWeb training sentences. Held-out anaphora versus
`206a0146`, verb reuse, prediction control, shuffled order, renamed vocabulary
and the shuffled-determiner control therefore remain **skipped, not passed**.
The declared seeds are 0/1/2 and all must be retained when the prerequisite is
met. No fresh-model bulk scoring was performed. Mechanism checks and standing
XOR controls cannot establish these learning claims.

The mechanism is accepted; **held-out anaphora, verb reuse, prediction control,
shuffled order, renamed vocabulary and determiner control, across seeds 0/1/2,
remain pending the million-sentence checkpoint and are not claimed**.
The original [review delivery](delivery.json) remains a historical record;
[acceptance](acceptance.json) supersedes its review status without changing
the measured source or repeating measurements.

Next: query and ask under [thought-loop §§5–7](../../plans/2026-10-07-thought-loop.md#5-the-query-rename-claude-for-codex-2026-10-07),
then item 6.2, thinking, whose specification follows from Claude.
