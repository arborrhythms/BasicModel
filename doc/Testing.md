# Bounded development tests

Step 0 is published on both mains: BasicModel `d6eb0cf3b`
(source `bfc834d79`), WikiOracle `cb4d718be`. The
[round-2 receipt](benchmarks/2026-10-19-item4-step0-round2/README.md)
records the unchanged decomposition certificates passing in both gain arms,
the repaired .0517 invariant and the scoped initial-content neutrality bound.
The accepted first-round 16/16 standing XOR cases in each arm remain their
original measurement. One full sweep completed 5,861 cases with one failure;
the empty-menu repair then passed the affected checks, whole §14 file and
both decomposition arms. The receipt reports those results separately.
The copular-artifact prerequisite skip and the Item 0 / 5.5 learning gates
remain explicit.

Item 3.5 step 1 is published on both mains: BasicModel `204e1be55`,
WikiOracle `a89c958`. Its [landing receipt](benchmarks/2026-10-19-item3-5-bind/landing.md)
records one 5,907-case sweep, the 48 failures and scoped repairs, final
native validation of 72 passed/one copular prerequisite skip, and both
decomposition lengths passing in both gain arms with all eight questions
correct.

Step 1b remains an unpublished candidate on `codex/item3-5-first-pass`.
The [current receipt](benchmarks/2026-10-20-item3-5-stm1/status.md) records
one full sweep: **5,965 completed; 5,420 passed, 219 failed, 325 skipped,
one XPASS**. Later scoped checks or declared fixture ports close 218 of
those failures; native MM_grammar pool exhaustion remains unresolved, and
publication is held for bind-plan §10's allocation decision and repair.
Native-120 completes 316/316 (309 passed, three fixture failures, four
prerequisite skips); native-121 passes the four repaired fixture checks.
Whole §14, G1–G3, reference-close, cube/inverse, word-store and compiled
boundaries pass. Saved arrivals 32/75 pass in both orders in native-122/123;
the new sweep arrival failure passes at its exact captured state in native-125.
The random paired case and saved state 43 tie exactly at zero reconstruction
cost. Claude's unsaved state is unavailable and is not claimed as replayed.
Both decomposition lengths pass in both gain arms in attempt 124, all eight
companion questions correct at 234/512 and 557/768 work. Original snapshots,
scopes, budgets and assertions are unchanged.
All failed and incomplete attempts remain in the receipt; the three
signed-coordinate artifacts remain retired and prerequisite-skipped.
The optional cold-start six-epoch learning run timed out without a learning
verdict and is explicitly carried to Item 0, without extra training.

Item 4's source `fdeef42b8` and October 19 [repair receipt](benchmarks/2026-10-19-item4-repair/README.md)
records each §9 gate, the review of record, and close-out measurements.
Alec's B3 decision keeps the selection prime on and φ=.5 with all gains zero.
The declared parity configuration keeps prediction enabled, disables selection
priming and restores uniform alternatives. Its native paired comparison has
zero parameter updates and records the changed warm-document behavior.
The full §14 file completes 13/13: 11 passed, the carried initial-distribution
failure, and the absent copular-identity skip. All three N2 regressions pass.
The single full landing sweep completes **5,834/5,834: 5,475 passed,
322 skipped, 39 failed and one XFAIL**, in 1,340.06 s. Post-sweep source and
fixture repairs are covered separately: 510 pass, one skip and two failures
in the 513-case scoped check. One failure is the carried .0517 paired-cost
counterexample; the other stale non-winner assertion is corrected. The final **10/10**
check passes its witness contract, both compiled gain settings and both NP
checkpoint restorations. The full sweep is not repeated or relabeled green. The original failed native learning run, Item 0
findings and independent 5.5 gates are unchanged. Historical negative-image
sections below describe their original sources and do not certify the gain path.

Item 6 **part 1** lands the [reviewed repair pass](benchmarks/2026-10-10-item6/part1/README.md):
shared bounded inversion, support charges in the keep cost, captured-input
isolation, candidates of every family on either side, and residual-surprise
minting with greedy sparse encoding. The four requested repair checks pass,
as do the recurrence regression, 102 affected cases, the strengthened lexical
materialization check, two default-backend checks and two explicit compiled checks.
The one source-matched sweep completes **5,745 cases: 5,454 passed,
287 skipped, 3 failed and 1 expected failure**.
The [comparison with accepted 6.1](benchmarks/2026-10-10-item6/part1/full/baseline-comparison.json)
names the new coverage-scope failure and b's failed bound as new relative to
that accepted sweep. b was already documented as flaky; no initialization is
retried. The earlier empty cleared-cache inverse was new with the candidate;
it now passes without changing its assertion. The existing overlap marker
moves from the baseline's XPASS to XFAIL (.000 versus .800), and remains visible.

Lower at 2/3/5 is incomplete with a charge for dropped support. Lift recovers
2/3/5 with live wholes; stored dictionary-only 3/5 remain incomplete. The
zero-gradient order lesson, capacity-blocked MM grammar recovery and intermittent
compiled retention remain open alongside part 2, with all earlier receipts
preserved. The [identity-from-data amendment](plans/2026-10-10-item-6-stored-idea-generativity.md#2-identity-from-data-amendment-alec-2026-10-10)
requires corpus and measurement before retiring rules. Part 1 makes no learned
identity or stored-compound recovery claim; item 5 may not discard derivations.

Item 6.5's [October 7 accepted mechanism receipt](benchmarks/2026-10-07-item6-5/README.md)
records magnitude definedness first, then native identity/change columns and
global bind/mint selection. Its final source has a green **5,527-case** sweep:
5,241 passed, 285 skipped and one existing non-strict XPASS. The 25 component
mechanism cases pass. The thirty unseeded standing trainings on that exact
source pass: sum 10/10, joint XOR class/reconstruction 10/10 and MM_xor 10/10.
The receipt preserves earlier failures, pre-training launcher errors and every
training outcome. The four older slow-table checks also pass, each run once
and reported separately with predictions and errors.
The [acceptance](benchmarks/2026-10-07-item6-5/acceptance.json) records Alec's
authorization after Claude's spec §10 review. Held-out anaphora, verb reuse,
prediction control, shuffled order, renamed vocabulary and determiner control
across seeds 0/1/2 remain pending the million-sentence checkpoint and are not
claimed. No inspected checkpoint meets that exposure prerequisite.

Run affected files or individual cases during development:

```sh
make test TEST_ARGS='test/test_query_registry.py --timeout 120 --suite-timeout 600'
make test TEST_ARGS='test/test_query_registry.py -k declaration'
```

Run the complete default suite in the background before a commit:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py
```

`make test` uses the same runner. `make test_all` additionally enables the existing
`RUN_SLOW=1` tests. Slow selection never changes the device: cases default to CPU,
and only an explicit `device` marker or `BASICMODEL_DEVICE` requests another device.
`make testp` is deliberately a direct `pytest-xdist` iteration command (`TEST_JOBS=auto`
by default), not a bounded receipt. `make preflight` and `make preflight_full` use the
receipt runner. Select affected tests for short development cycles; the complete default
suite remains the commit gate. Compilation remains enabled for tests that exercise it.

## Fast regression and slow experiments

Long convergence, memorization and performance experiments use `@pytest.mark.slow`.
The central collection hook skips them before fixture setup unless `RUN_SLOW=1`.
`RUN_SLOW=0` stays fast. The older per-test `RUN_SLOW` gates remain supported.
A slow check keeps its assertions and any explicit expected-failure status;
skipping it in the default run is not evidence that its learning gate passed.

**Repeated-work budget (Alec, 2026-10-26).** `RUN_SLOW=1` now defaults to
`RUN_SLOW_FRACTION=0.5` for the explicitly budgeted training and stability
fixtures. Their epoch, update or per-epoch batch counts use `slow_count`;
rounding keeps at least one observation and each fixture's structural minimum.
For example, projection stability runs three epochs instead of six, the long
reconstruction experiments run 80 instead of 160, and the 256-presentation
successor/ladder replays train 16 batches per epoch instead of 32. The latter
keep their corpus, epochs and evaluation set. No test or assertion is removed,
no seed changes, and one-shot mechanism and compiler checks retain their work.
`RUN_SLOW=1 RUN_SLOW_FRACTION=1` restores the historical repetition budgets.
The pytest header reports the fraction. Shorter measurements state their
actual work; they do not replace the retained full-budget evidence. Fixed
construction and compilation costs mean wall time need not scale exactly
with the repeated work.

**The generated math suite under `RUN_SLOW` (Alec, 2026-10-20).** Beside
the convergence experiments, the slow tier carries a **generated, graded
corpus of mathematical sentences in English** that trains a grammar for
real, at scale, and whose gates are the things we claim the model can
learn: equality and definition by name, `VP → VP NP` applied twice with
order-sensitive roles ("x minus one" against "one minus x"), one-step then
two-step decomposition with the answer line as supervision, premises in
either order, `what` bound by `bind`, yes/no questions on the truth lanes,
conditionals and counterfactuals with computable consequents, ellipsis and
coordination, generics against particulars, attribution with conflicting
sources and known truth, the inverse verb (minus undoes plus) and the verb
groups (the additive verbs jointly diagonalizable in one basis, the
multiplicative in another — a translation along the successor chain of
opaque numerals is a diagonal in the chain's Fourier basis, so joint
diagonalization has an exact answer to find), and the carrier variants of
occurrence §13 (the same meaning under "equals", "is", "sums to", and the
marker moved from word to affix). Numerals stay opaque words, so the suite
tests mechanisms, never arithmetic competence. The corpus is produced in
the Viterbi order (thinking spec §13.9: one more step per item, bottom-up)
by the `MM_math_chain` generator and its successors; each run records
throughput so item 1's runtime blocker is measured where it bites. **The
same run is the compiler test at scale:** compiled and eager execution of
the identical corpus, seed and budget must agree in actions,
probabilities, costs and parameter updates, which no fixture-sized check
can show. Coverage, and what the suite cannot reach, is in
[Training, "The generated math suite"](Training.md#the-generated-math-suite-what-it-exercises-and-what-it-cannot-2026-10-20).

**Required XOR table (Alec, 2026-09-29).** Every receipt includes the six
grounded cases in `test_grounded_xor.py`, the three curriculum cases in
`test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse`,
both `TestXorExactCliReconstruction` gates in `test_explicit_dimensions.py`,
and all three slow `TestMMXorConvergence` cases in `test_mm_xor.py`
(`test_convergence`, `test_learns_xor_signal`, and
`test_mm_grammar_learns_xor_signal`). Run the slow gates explicitly with
`RUN_SLOW=1`; the default sweep's skips do not supply these measurements.
Also run and record `TestXorGrammarLearnsXor::test_xor_class_accuracy` and
`TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`.
XOR_grammar is a gate of the grammatical operators update, not of item 7.
The [round-3 receipt](benchmarks/2026-09-29-item7-review-round3/README.md)
records the baseline and each stage's complete table. Keep the predictions
and errors, including failures; a loose signal threshold alone does not
establish XOR learning. A case stopped by the unchanged 8 GiB guard runs
once more without that guard as a separate diagnostic, never as a replacement
passing gate. See [the decided work list](specs/2026-09-16-two-truths-ideas-and-relations.md#16-hand-off-to-codex-claude-2026-09-29-what-to-change-after-review-round-3).

The September 29 implementation's [final explicit selection](benchmarks/2026-09-29-item7-review-round3/explicit-after-owner/summary.json)
accounts for 199 cases: 195 passed, two XOR_grammar failures, one depth-3
skip for the missing mature checkpoint, and one graph-release memory stop.
All 170 item 7 cases and all four strict parity checks pass. The category
codebook and 64-word trace remain below 1 GiB. Reconstruction keeps all eight
declared seeds per tree, including the candidate seed-2 memory stop and its
separate diagnostic. The ten MM_grammar attempts complete 10/10 at HEAD and
8/10 in the candidate; two clause-capacity failures remain visible. The
receipt reports the incomplete predictor-context observation separately.
The [one full sweep](benchmarks/2026-09-29-item7-review-round3/full-sweep/summary.json)
matches that source and completes all 5,050 cases: **4,714 passed, 13 failed,
322 skipped and one expected failure**. Six of those failures passed in
round 2; fourteen of its sixteen regressions now pass. The sweep has no
resource stops and peaks at 7.24 GiB per worker. The candidate remains
unaccepted and uncommitted, stopped for Claude's review with every failure
and the separate explicit resource failures recorded.

```sh
make test_all TEST_ARGS='test/test_output_path_supervised.py -k memorizes --timeout 1800'
```

The initial timing audit found a 119-second isolated compiler test, a 72-second
XOR training test and 130 seconds of repeated 20M-model blind-decoder training.
Two unmarked expected-failure arithmetic experiments also trained for 40 and
30 epochs. 245 long-running functions now use the central slow gate. These include
repeated corpus-learning experiments, six-batch phrase/checkpoint training,
multi-step Adam checks, full compiled reconstruction-order integration and
production-size 20M configuration/reconstruction
integration checks. The next bounded audit reached its 30-minute worker deadline
while tracing/backpropagating a three-batch tied-reconstruction integration. That
run failed the gate; its process receipt and stack sample identify the workload.
The L1 multi-batch case, full-model backward/cache checks and the large serial
output fixture now run explicitly with `RUN_SLOW=1`. The smaller compiler,
numerical training, ordinary forward/backward, ownership,
checkpoint and query regressions stay in the default suite. Run the slow checks
explicitly when their corresponding training paths change.

Native answer fixtures use 264-wide concepts and 136-wide percepts during
development. Their ownership, checkpoint and gradient checks retain distinct
widths. Seven corresponding 1032-wide cases remain available under `RUN_SLOW=1`.
Repeated packed training, expectation toggling and multi-brick optimizer
integrations also use the slow gate. A further default profile completed only
912 of 4,458 cases in 26.2 minutes before an explicit stop: repeated 20M legacy
builds caused 4–7 GiB footprints and frequent fresh-worker restarts. The same
production-size recurrence, dual-input/tower, symbolic-iteration, decode,
export and per-word compiler fixtures now run with `RUN_SLOW=1`, as do full CLI
training/reconstruction checks. Lightweight geometry, masking, query, dictionary,
checkpoint and compiled-kernel checks remain in the default suite.
The subsequent profile completed 1,246 of 4,458 cases in 25.8 minutes before
another explicit stop. A fixture audit found additional 16,000–65,000-row legacy
configurations in global attention, reading, masked semantics, word storage and
serial reconstruction. Their full-model checks now use the same slow gate;
small layer-level lifecycle, loss, projection and runtime-inference checks stay
in the default suite. All assertions and reasoning methods are retained.
The compact numerical readout and metadata tests continue covering larger
production shapes without materializing their full training graph.

The next attempt recorded 2,907 of 4,458 cases before a 30-minute worker timeout.
That batch completed its two full reconstruction graph-capture checks and its
tied-gradient check, but could not finish its remaining selection. The full-model sentence-traversal integration checks and a measured
280-second sequence-training check now use the slow gate. The first traversal
check alone subsequently measured 102 seconds in the affected run.
Their function bodies and assertions are unchanged. The direct binary recomposition/residual checks in that file and the
smaller operator, reconstruction-objective, byte-fidelity and role/gradient
regressions remain available in the default suite. This timed-out attempt is not a passing validation result.

Memory- and time-triggered recycling permits larger case batches (256 cases / 16 files)
without retaining a worker past its resource boundary. The fast runner uses a
small concurrent pool under one aggregate reservation rather than one serial
worker; it retains process isolation and hard caps while avoiding needless
Python/PyTorch import waits. Per-case timings in the durable receipt identify
further expensive checks; a timeout is a failure that needs investigation,
never an automatic slow mark.

## Tiers for a model that has not learned (Alec, 2026-10-16)

The suite is test-heavy for a model that has not fully learned: of ~5,800
cases a few dozen are learning certificates and the rest are mechanism
contracts, which say whether the code still does what it did last week —
and the design is converging on [Orders](Orders.md), so last week's
contract is often this week's retired rule. Every test stays in the
harness; what changes is when it runs.

- **Default sweep, every landing:** every test under 5 s, plus every
  learning certificate regardless of its time — XOR composition
  (`test_grounded_xor`, `test_word_admission`'s XOR, `test_mm_xor`), the
  operators' certificates, the math-chain e/pending-premise
  certificates, the curriculum stage gates, the trained-artifact
  coverage certificate. A round's measure is which gate moved.
- **`PHASE_CERTIFICATES`, considered in every default sweep:** certificates
  that require a preceding lesson use `@pytest.mark.phase('phase_name')`.
  `test/fixtures/phase-artifacts.json` maps the phase to its artifact path,
  SHA-256 and producing receipt. An absent artifact skips with the phase and
  reason; an incorrect hash or a rejected restore fails. These certificates
  run regardless of a historical slow mark when their prerequisite exists.
  The JSON, HTML and terminal reports summarize skipped certificates by phase.
- **Weekly `RUN_SLOW=1`:** mechanism contracts at or above 5 s. From the
  October 10 sweep's timings (1,634 s of test time over 5,745 cases): 76
  tests take 73 % of it; 53 of those are contracts (834 s) and move to the
  weekly tier; the 23 certificates (358 s) stay. The remaining 5,669 tests
  take 442 s.
- **Retired contracts are deleted, not carried red**: a test of a
  mechanism retired by decision goes, and one that protects a behaviour the
  replacement must keep is rewritten against the new API. One regression
  per repaired defect; no families of mechanism checks.

Phase prerequisites are separate from learning outcomes. The October 20
migration decision retired `stage3`, `np_definite` and `np_description`:
their registry entries are null, and their signed-coordinate files are
archived beside their producing receipts and never read. Their certificates
remain prerequisite-skipped until the producing lessons are next needed in
Item 3.5, with their fixed budgets and seeds. FineWeb, copular identity (`c`)
and the full 4e kept-readings certificate also have no qualifying artifact
registered. The column-copula test remains an unconditional mechanism contract.

`restore_phase(loader)` verifies the marked prerequisite before calling the
consumer's loader with its resolved path. There is no training or fresh-model
fallback. Paths are relative to the repository root, or absolute. For an
external artifact, select a complete registry with
`BASICMODEL_PHASE_ARTIFACTS=/path/to/registry.json`. FineWeb entries also name
`config` and optionally `minimum_sentences` (default 1,000,000); the completed
training-progress check still applies. A `stage4e` entry hashes the produced
`kept-programs.json` that the certificate grades. This replaces the old
FineWeb and constituent artifact environment switches; archived receipts
retain their historical commands.

The slow set, from those timings (durations in seconds, call phase):

| s | tier | test |
|---:|---|---|
| 99 | weekly (slow contract) | `test/test_math_chain_section14.py::test_ordinary_initial_binding_distribution_includes_every_retained_candidate` |
| 83 | default (certificate) | `test/test_math_chain.py::test_native_worked_document_trains_unforced_without_numeric_oracles` |
| 70 | weekly (slow contract) | `test/test_trial_policy_ownership.py::test_trial_and_batch_answer_leave_the_decoder_to_reconstruction[True-explore]` |
| 68 | weekly (slow contract) | `test/test_trial_policy_ownership.py::test_trial_and_batch_answer_leave_the_decoder_to_reconstruction[True-exploit]` |
| 59 | weekly (slow contract) | `test/test_math_chain_repair.py::test_ordinary_driver_two_three_eight_stream_groups_after_single_write` |
| 56 | weekly (slow contract) | `test/test_output_walk.py::test_walk_compiles_fullgraph_and_matches_eager` |
| 39 | phase (copular identity, 2026-10-18) | `test/test_math_chain_section14.py::test_forced_ordinary_answer_fills_committed_question_without_its_own_episode` |
| 31 | default (certificate) | `test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[True]` |
| 31 | default (certificate) | `test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[False]` |
| 28 | weekly (slow contract) | `test/test_bounded_test_recycling.py::test_time_pressure_recycles_after_complete_cases_without_dropping_coverage` |
| 28 | default (certificate) | `test/test_math_chain_section14.py::test_forced_ordinary_bound_declaratives_open_no_episode` |
| 22 | weekly (slow contract) | `test/test_bounded_test_recycling.py::test_live_worker_receipt_keeps_completed_cases_when_a_later_case_times_out` |
| 19 | default (certificate) | `test/test_math_chain.py::test_zero_budget_loads_and_trains_the_ordinary_vp_without_attention` |
| 19 | weekly (slow contract) | `test/test_math_chain_repair2.py::test_eight_corpus_sentences_train_at_distinct_fork_rounds` |
| 17 | weekly (slow contract) | `test/test_arithmetic_isolation.py::test_supervised_update_cannot_use_exact_arithmetic_or_fallback_codes` |
| 17 | weekly (slow contract) | `test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[True]` |
| 16 | weekly (slow contract) | `test/test_output_walk.py::test_output_ignores_input_derivation_on_conceptual_slots` |
| 16 | weekly (slow contract) | `test/test_interleave_schedule.py::test_one_native_read_retains_shared_owner[True]` |
| 16 | weekly (slow contract) | `test/test_compiled_operator_diagonal.py::test_free_candidate_search_compiles_backward_through_native_operators` |
| 15 | weekly (slow contract) | `test/test_tied_operator_reconstruction.py::test_snapshot_owns_values_across_same_shape_dictionary_update` |
| 15 | weekly (slow contract) | `test/test_sentence_journal.py::test_no_numerical_frames_without_a_sentence_closing_reader` |
| 15 | weekly (slow contract) | `test/test_negative_expectation.py::test_native_nonzero_composition_is_bit_identical_at_every_stance` |
| 14 | retired (2026-10-18) | Category-VQ context smoke; replaced by the distributional certificates in `test/test_parts_of_speech.py` |
| 14 | weekly (slow contract) | `test/test_echoic_decoder.py::test_same_walk_infers_binary_and_unary_without_a_journal[True]` |
| 14 | weekly (slow contract) | `test/test_compiled_word_chunk.py::test_tensor_peer_word_loop_rejects_external_full_stm` |
| 13 | retired (2026-10-18) | Category-VQ allocation/EM smoke; occupation counts have no trained codebook |
| 12 | default (certificate) | `test/test_learning_readiness.py::test_native_progress_and_trained_artifact_evaluation_path` |
| 10 | weekly (slow contract) | `test/test_bounded_test_recycling.py::test_recycle_on_memory_pressure_keeps_every_case_once` |
| 10 | weekly (slow contract) | `test/test_output_path_supervised.py::test_legacy_parity_memory_cannot_change_an_owned_conceptual_answer[development]` |
| 10 | weekly (slow contract) | `test/test_negative_expectation.py::test_native_unlabelled_batch_trains_the_same_chooser` |
| 9 | weekly (slow contract) | `test/test_output_path_supervised.py::test_native_checkpoint_restores_active_answer_widths[False-development]` |
| 9 | weekly (slow contract) | `test/test_prepared_answer_boundary.py::test_prepared_concept_handoff_cuts_answer_state_gradient[True-False]` |
| 8 | weekly (slow contract) | `test/test_arithmetic_isolation.py::test_numeric_and_renamed_sentence_paths_keep_oracles_out[texts0]` |
| 8 | default (certificate) | `test/test_grounded_xor.py::test_native_unseeded_xor[0-8]` |
| 8 | default (certificate) | `test/test_grounded_xor.py::test_native_unseeded_xor[2-8]` |
| 8 | default (certificate) | `test/test_grounded_xor.py::test_native_unseeded_xor[1-8]` |
| 8 | default (certificate) | `test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[8-1]` |
| 8 | default (certificate) | `test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[8-0]` |
| 8 | default (certificate) | `test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[8-2]` |
| 8 | weekly (slow contract) | `test/test_one_attention.py::test_narrowing_uses_the_shared_operation_chooser_and_compiles` |
| 8 | weekly (slow contract) | `test/test_bounded_test_runner.py::test_real_pytest_coverage_fresh_workers_and_failure_receipt` |
| 8 | default (certificate) | `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1]` |
| 8 | weekly (slow contract) | `test/test_unified_thought_controller.py::test_runbatch_credits_each_controller_row_from_its_own_answer` |
| 8 | default (certificate) | `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0]` |
| 8 | default (certificate) | `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2]` |
| 8 | weekly (slow contract) | `test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser[tie]` |
| 8 | weekly (slow contract) | `test/test_input_byte_addresses.py::test_promoted_percepts_keep_their_original_byte_addresses[stream_smoke]` |
| 8 | weekly (slow contract) | `test/test_expectation_defaults.py::test_native_future_and_other_row_changes_do_not_change_first_estimate` |
| 7 | weekly (slow contract) | `test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser[distinct]` |
| 7 | default (certificate) | `test/test_math_chain_section14.py::test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage` |
| 7 | weekly (slow contract) | `test/test_query_native_ids.py::test_real_sentence_capture_owns_native_object_ids_across_later_staging` |
| 7 | weekly (slow contract) | `test/test_compiled_word_chunk.py::test_peer_leg_profiler_executes_real_ps_ws_and_conceptual_bodies` |
| 7 | weekly (slow contract) | `test/test_review14_contracts.py::test_review17_measurement_wiring_on_one_ordinary_batch` |
| 7 | weekly (slow contract) | `test/test_arithmetic_isolation.py::test_numeric_and_renamed_sentence_paths_keep_oracles_out[texts1]` |
| 6 | weekly (slow contract) | `test/test_ltm_consolidation.py::TestSerialForwardAppendsStore::test_forward_grows_store_not_deque` |
| 6 | weekly (slow contract) | `test/test_output_walk.py::test_question_conditioner_optimizer_steps_both_widths` |
| 6 | weekly (slow contract) | `test/test_output_walk.py::test_packed_recall_observes_captured_sentence_programs` |
| 6 | weekly (slow contract) | `test/test_bounded_test_runner.py::test_parallel_workers_share_the_aggregate_budget_and_overlap` |
| 6 | weekly (slow contract) | `test/test_training_device.py::test_all_dispatches_explicit_device_marks_and_keeps_slow_proofs_on_cpu` |
| 6 | weekly (slow contract) | `test/test_decoder_exploration.py::test_single_departure_replays_prefix_and_is_greedy_after_it[True]` |
| 6 | weekly (slow contract) | `test/test_joint_objectives.py::test_real_intermediate_and_final_ends_have_same_canonical_roles` |
| 6 | default (certificate) | `test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[4-0]` |
| 6 | default (certificate) | `test/test_grounded_xor.py::test_native_unseeded_xor[2-4]` |
| 6 | default (certificate) | `test/test_grounded_xor.py::test_native_unseeded_xor[0-4]` |
| 6 | default (certificate) | `test/test_grounded_xor.py::test_native_unseeded_xor[1-4]` |
| 6 | default (certificate) | `test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[4-2]` |
| 6 | default (certificate) | `test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[4-1]` |
| 6 | weekly (slow contract) | `test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[False]` |
| 6 | weekly (slow contract) | `test/test_training_device.py::test_parallel_pool_keeps_one_accelerator_training_lane` |
| 6 | weekly (slow contract) | `test/test_relation_closing.py::test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows` |
| 6 | weekly (slow contract) | `test/test_stored_idea_generation.py::test_compound_candidates_are_not_word_emissions[True]` |
| 5 | weekly (slow contract) | `test/test_review11_contracts.py::test_supported_compound_cannot_stop_even_with_a_chunk_code[True-False]` |
| 5 | weekly (slow contract) | `test/test_bounded_test_runner.py::test_test_failure_does_not_cancel_remaining_worker_coverage` |
| 5 | weekly (slow contract) | `test/test_gradient_factorization.py::test_normal_batch_logs_owned_optimizer_gradients` |
| 5 | weekly (slow contract) | `test/test_ltm_consolidation.py::TestRuntimeUserIngestion::test_user_rows_coexist_with_conversation_rows` |
| 5 | weekly (slow contract) | `test/test_surface_grammar.py::test_normal_batch_trains_supplied_grammar_lessons` |

The landing tier implementation is in `test/slow_tests.py`, keyed by exact
parameterized node IDs. Of the historical 53 slow entries, 52 still exist;
`test_eight_corpus_sentences_train_at_distinct_fork_rounds` was already retired
by the §13 decision. Of the 23 timed certificates, 22 still exist; the old
bound-declarative departure-diversity test was retired at the same time.
Their replacement is `test_exploration_coverage.py`'s trained-artifact
certificate, in the default tier with its stated probability bound. These
two historical names in the table are provenance, not tests to resurrect.
The MM XOR and explicit XOR convergence/reconstruction certificates are also
default tests; their old `RUN_SLOW` guards are removed without changing any
assertion. Artifact certificates still require the declared trained artifact.
The exact landing commit and its one full sweep are recorded in the
[WikiOracle landing receipt](https://github.com/arborrhythms/WikiOracle/blob/main/doc/benchmarks/2026-10-16-item6-landing.md).

## Limits and isolation

- One full receipt per user across worktrees, enforced by a nonblocking file
  lock. Selected-file, `-k`, and marker runs may proceed concurrently.
- Up to `cpu−4` concurrent, one-thread fresh workers by default: **10** on
  this 14-core machine, leaving four cores for SSH and interactive work. Each
  receives at most 256 selected cases from 16 files; a new process releases
  its model and compiler allocations at a boundary. `--workers` can lower the
  pool for diagnosis.
- The aggregate default is physical RAM minus **8 GiB**: **28 GiB** on this
  36 GiB machine. Each worker retains an **8 GiB** `taskpolicy -m ... -P kill`
  ceiling; the supervisor samples every active worker tree and stops the
  largest worker if their aggregate physical footprint crosses 28 GiB. This
  supports legitimate 4.8 GiB cases without allowing the pool to consume the
  8 GiB reserved for the machine. `--memory-gib` may lower the aggregate
  budget, but cannot leave less than 8 GiB for the machine.
- At 80% of a worker deadline, or at the earlier of 80% of its 8 GiB cap and
  that cap minus 64 MiB (never later than half its cap for a small cap), a
  worker finishes its current test and exits. A fresh worker runs remaining
  cases. Coverage records count one final completion per selected case; resource
  recycling does not repeat cases. The hard memory limit still applies within a
  test, and a killed worker is a failure.
- An ordinary pytest assertion, setup or teardown failure does **not** cancel
  the remaining workers. The receipt keeps its `test_failure` outcome and
  runs every selected case so a diagnostic full run reports all failures.
  Process-boundary failures (timeout, memory kill, unavailable accounting,
  collection/protocol failure, interrupted execution or invalid coverage)
  still stop the pool immediately to protect the machine and preserve an
  honest receipt.
- A typed Inductor `CppCompileError` reporting a modified input to a stale
  precompiled header gets **one** fresh-worker retry with
  `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0`. Compilation still runs;
  shared caches are not deleted. The receipt records `compile_cache_retries`
  and marks the original report `compile_cache_retry`; raw worker JSON and
  logs retain the original failure. The retry must pass normally. A second
  cache failure, an ordinary compiler error, or any simultaneous assertion,
  setup or teardown failure remains a failure. No other errors are retried.
- CPU, BLAS and compiler pools use one thread. The existing on-disk Inductor cache
  remains reusable between workers. macOS workers run at `nice -n 10`, not the
  throughput-throttling `taskpolicy -b` background class.
- Default deadline: 1,800 seconds per worker, including collection; 10,800 seconds
  for the whole suite. Configure finite limits with `--timeout` and
  `--suite-timeout`. Tests exceeding a limit fail the run; they are not skipped.
- `--batch-size 1` reduces accumulation within a batch when diagnosing a memory
  failure. Reducing batch size does not omit any selected cases.

Tests that read one trained model use a module-scoped fixture and
`@pytest.mark.shared_training("name")`. The bounded runner keeps the selected
members of that group in one worker, including the weekly run at batch size
one. Recycling waits until the group's final selected test; the memory and
time ceilings remain unchanged. Selecting one test alone does not add its
partners. `TestXorGrammarLearnsXor` and `TestXorGrammarReconstruction` use this
contract: ten unseeded trainings report both unchanged bars and joint success,
and the first training supplies both rows of the named XOR table.

## GPU training tests

Slow selection and device dispatch are independent. A test marked only `slow`
keeps the CPU calibration, including XOR proofs and exact/blind MM round trips.
`@pytest.mark.device("gpu")` explicitly requests the available accelerator;
`device("mps")`, `device("cuda")` and `device("cpu")` select one directly.
A caller's explicit `BASICMODEL_DEVICE` applies to its whole targeted run.
Only one accelerator worker runs at a time. A requested unavailable accelerator
fails rather than falling back. MPS-only and CUDA-only test bodies retain their
availability checks; the weekly run enables the MPS gates on this machine.

```sh
RUN_SLOW=1 BASICMODEL_DEVICE=mps .venv/bin/python test/test_report.py test/test_inter_contrastive_predict.py
```

Worker receipts identify the resolved device. End-to-end GPU timing must include
compilation, forward, backward and optimizer work; no speedup is implied by
device selection. The same aggregate memory, thread and deadline limits apply.

macOS accounting uses `proc_pid_rusage` physical footprint, including compressed
memory. RSS alone can become misleading during compression. Each macOS worker
has an 8 GiB kernel limit through `taskpolicy -m ... -P kill`, while the
supervisor samples all active process groups and enforces the 28 GiB aggregate
reservation. Linux accounting uses RSS plus swap. The monitor samples each
process group and known children, including compiler subprocesses; sampling can
allow brief overshoot.
Timeout, memory exhaustion, unavailable accounting or interrupted execution
terminates the worker and its children and returns failure. Linux enforcement
is sampled, without the additional macOS kernel cap. Unsupported platforms
fail before running tests.

## Evidence and reports

Each invocation creates a new `output/tests/<timestamp>-<id>/` directory, or the
new directory named by `--run-dir`. It contains:

- `source-manifest.json`: validated hashes of code, tests, data/grammar and
  configuration, plus separately recorded documentation and `todo.md` hashes.
  Documentation changes do not void a code receipt; code/data/grammar changes
  still do.
- Collection output and the complete list of selected pytest node IDs.
- Per-worker logs, process exit status, elapsed time and peak measured memory.
- Per-worker JSON progress after collection and each test transition, including
  the active case and confirmed completed cases. `result.json` lists all active
  worker progress files; the aggregate completion list updates when each worker
  exits. A later timeout retains earlier confirmed completions.
- `result.json`: durable overall status, limits and exact selected/completed cases.
- `report.html`: escaped test diagnostics, including setup and teardown failures.

A run is green only after every selected case completed exactly once, every
worker exited successfully and the source snapshot still matches. Expected
failures and explicit skips retain their pytest outcomes. Empty selection,
changed source, missing coverage, collection errors and killed workers cannot
receive a success receipt. In-progress receipts remain incomplete even if all
workers completed so far have passed. Logs and receipts survive disconnection.

Do not edit files in the tested snapshot while a suite runs. Use an isolated
checkout for independent implementation work. Direct `python -m pytest` calls
(including `make testp`) are the fast iteration path and do not create a bounded
receipt. Use the bounded entry point for full commit gates.

## Validation

Resource probes exercise real hanging processes, resistant children, aggregate
child memory, native compressed-memory accounting, monitor failures, lock
exclusion, fresh-worker coverage, teardown failures, source mutation, empty
selection and interruption with persistent evidence. The checkpoint tooling selection passed **29 passed in 138.27s; peak aggregate footprint 0.56 GiB**
(20 resource/coverage cases, six device cases and three slow-switch cases).
Earlier affected-file runs and their tested source hashes are archived separately;
those earlier runs do not validate later runner revisions.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#fast-aggregate-runner-evidence-september-19) (2026-10-30): "Fast aggregate-runner evidence (September 19)".*
## Future work: bounded fixture reuse

Evaluate compatible preallocated fixture reuse with complete reset checks and
separate construction/compilation counters. Existing fixed-capacity contracts
remain in place. [Assessment and acceptance criteria](plans/2026-09-17-bounded-fixture-reuse.md)
record the follow-up; comparative speedups remain unmeasured.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#selected-meaning-and-one-controller-september-20) (2026-10-30): "Selected meaning and one controller (September 20)".*
## October 1 suite trim: weekly slow coverage

`make test_slow_weekly` uses the existing virtual environment and does not depend
on the venv rebuild stamp. It selects central `slow` markers, historical
`RUN_SLOW` decorators/body gates, and the three `RUN_MPS_SLOW` cases. The older
switches retain their original gating behavior; the common marker now exposes
them to selection. CUDA-only cases keep their availability skips.

This machine's weekly target uses the suite's CPU fixture contract; the three
MPS-specific tests explicitly switch to MPS. Ordinary cases have an **8 GiB
per-worker ceiling**. The production objective measurement in
`test_objective_conflicts_slow.py`, at its unchanged batch of 28, has a
**24 GiB per-worker ceiling**. The October 2 ownership contract replaces the
former two-arm test with one ownership measurement; historical weekly records
retain the original arms. Both tiers run one worker at a time under a
**24 GiB aggregate reservation** and the existing **30-minute worker deadline**.
The ordinary full sweep's 8 GiB worker ceiling is unchanged.

The same weekly target also runs `python bin/Legacy.py` and the three
`bin/etc` inline unittest suites once, under the ordinary 8 GiB worker ceiling.
Their individual test counts and failures join the weekly record. The Makefile's
`SigmaPi`, `SymPercept` and `SPNN` demonstration targets pass `--demo`.

The weekly target freezes source and fixtures into its receipt's `source/`
directory before collection (a plain file copy, not another Git worktree).
Workers read that copy, so subsequent cost-function edits cannot change the
measurement. Its manifest identifies the exact uncommitted candidate tested.
Each run writes its detailed bounded receipts beneath `tmp/slow-tests/`, plus a
small `record.json` and `latest.json`: UTC date, commit, case counts, outcomes,
failures, duration, peaks, ceilings and source-match status. A full sweep's
summary warns when that record is missing, older than seven days, incomplete,
or failed. A subset run cannot stand in for the weekly record. A guard-stopped case remains a failed attempt; the weekly dispatcher continues only cases that have not started, without retrying failures or raising their ceilings. The record distinguishes attempted cases from cases with a completed pytest report. A complete attempt record can therefore still contain process failures. The frozen source shares the unchanged `.venv` runtime through a link so subprocess tests resolve the same interpreter; source and fixture files remain independent copies.

A Sunday 03:00 local launchd schedule is proposed in the October 1 receipt;
it has not been installed. The receipt preserves every removed or relocated
test's disposition, full old/new bodies and the 25-case performance profiles.
Compiler tests retain capture; ordinary ownership and gradient tests can use
`eager_reading`, which executes the same loop bodies with ordinary autograd.
Setting `MODEL_COMPILE=eager` alone still captures higher-order loop graphs.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#october-3-decoder-exploration-operators-and-one-attention--review-candidate) (2026-10-30): "October 3 decoder exploration, operators and one attention".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#item-50-forgetting-candidate-2026-10-19) (2026-10-30): "Item 5.0 forgetting candidate (2026-10-19)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#item-4-step-0-second-repair-round-2026-10-19) (2026-10-30): "Item 4 Step 0, second repair round (2026-10-19)".*
### Item 2.7 large-file corpus tier — 2026-10-29

At Alec's request, the capacity search and selected large corpus prefix are
registered as `test/test_corpus_large_slow.py::test_capacity_search_and_selected_corpus_prefix`.
Ordinary collection skips this test before construction with `RUN_SLOW` unset.
The November weekly runner discovers it alongside the existing slow cases,
then runs it alone with a 28 GiB process-tree budget and 48-hour outer guard.
Its fixed-grid search uses fresh native B28/W256 construction, optimizer state,
backward and update, with 28 GiB/one-hour per-trial guards. A selected prefix
must pass whole-document admission and a fresh brick before a full epoch
(24-hour guard). Known native admission exhaustion may select the preceding
whole document; unrelated errors cannot be classified as capacity pressure.

The test writes its commands, source snapshot, grid, trial results, admission,
selected configuration, and initial/periodic/final checkpoints under the weekly
receipt's `corpus-measurements`. It never overwrites the checked-in candidate
configuration. The development search was interrupted before choosing a final
size; its partial results remain in the item 2.7 final-round receipt. No slow
run was performed to register this test. The ordinary scoped and lesson gates
remain in the current round.


## Item 2.5 freeze: xfail what fails on an unresolved deviation (Alec, 2026-11-01)

Alec: "You can implement everything and xfail what tests you need to, let's
lock down what is working going forward." For item 2.5's stage 2 freeze
([thinking spec §32.2](plans/2026-10-27-item2-5-thinking.md), item 14): a
standing test or certificate that fails on a deviation the round does not
resolve is marked `pytest.mark.xfail(strict=True, reason=...)` with the
reason naming the deviation, listed in the return, and the round proceeds to
the freeze; the receipts record the xfails. The round stops only for a
source-hash mismatch or a numerical divergence, which stays fail-loud. `make
test` remains the criterion: a strict xfail that starts passing fails the
suite, and is removed together with the deviation it named.
