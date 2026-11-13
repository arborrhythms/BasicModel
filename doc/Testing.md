# Bounded development tests

This document describes how the suite runs (the bounded runner, its tiers and
its limits) and the rules that every test, fixture and measurement follows.

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

External data is prepared explicitly with `.venv/bin/python bin/prepare_data.py`.
The tracked manifest `data/external-files.json` records its URLs, byte sizes and
SHA-256 digests. Missing MNIST or FineWeb inputs skip with that command; pytest
never downloads them. The developmental assembly is a separate preparation:
`.venv/bin/python bin/prepare_corpus.py --recipe data/corpus-sources.json --output data/corpus/assembly-01`.
Its content identity excludes the elapsed `assembly_seconds` field.

Each bounded worker is capped at 8 GiB. The pool reserves eight GiB for the
machine and four CPU slots; this 24 GiB, eight-core machine admits two workers.
`make testp` is deliberately a direct `pytest-xdist` iteration command (`TEST_JOBS=auto`
by default), not a bounded run. `make preflight` and `make preflight_full` use the
bounded runner. Select affected tests for short development cycles; the complete default
suite remains the commit gate. Compilation remains enabled for tests that exercise it.

The item 0.5 saved-stage tests are in `test/test_stage_models.py`. Default
runs skip null registry entries; an explicitly requested trained tier requires
the actual artifact, its recorded configuration and hashes, strict byte-exact
restoration, declared criteria, and the reproducible operator column. Producer
metadata and recorded null outcomes survive registry round trips. The original
grammar-XOR producer descriptions are historical: the MM learning test now
exercises the numeric field, and the XOR class bar is numeric while its separate
grammar inverse still trains a model. Neither supplies the old combined
grammar-stage artifact criterion.

Source snapshots include tracked files and unignored new source, and exclude
Git-ignored local artifacts. Thus a prepared assembly does not change the
source identity; its tracked manifest still names its content hashes. Historical
receipts retain the identity rules under which they were measured.

## Rules for tests and measurements

**Seeds.** No random seed is chosen, pinned or changed so that an assertion
passes. Learning certificates run unseeded, and a learning claim holds across
unseeded runs or is reported with its pass count (for example, five or more of
ten trainings). A seed is fixed only to make something reproducible: a seed that
names a generated dataset is that dataset's identity and is recorded in its
manifest, and a recorded random stream may be reinstalled to replay a failure,
never to select a passing initialization. The former grammar-XOR seeded producer was superseded by the current numeric
field certificate; its old phase entry remains null and is not a producer.

**Replaying a failure.** Results do not depend on the process hash: a set whose
iteration feeds a computation is enumerated in admission or declaration order at
its producer; `test/test_hash_order_producers.py` certifies this for the
collision queue of `WordIdentity.admit` and for the registration of the
grammar's symbolic hosts. The bounded runner draws a `PYTHONHASHSEED` for
each worker when none is supplied, and `test/random_streams.py` records it with
the torch, NumPy and Python random states (and the CUDA or MPS generator states
where used) and their hashes: once for each worker before collection, and, for
a failing case, the states captured before its setup. Recording installs and
advances nothing and refuses to overwrite. A generator with no seed, such as
`MathProblemGenerator(seed=None)`, draws from the recorded Python stream, never
from operating-system entropy. A component constructed, or a code minted, while
the model reads leaves the global stream untouched: it draws inside a forked
stream, or from a generator its owner seeds once from a constant and checkpoints
(a document cover's code), so adding a component does not shift a fixture's
later draws.

**Capacities and shapes.** A model's capacities are fixed when it is
constructed, and compiled code is built for them. Changing a capacity or shape
at runtime is a resize: it invalidates compiled shapes, and an attempt to grow
a preallocated store raises. No capacity is changed to make a case pass. A
fixture that needs room for an added row preallocates the larger capacity at
construction, keeps its behavioural assertions, and writes any fractional
threshold against the new capacity. A fixture's capacities are sized for one
presentation and written beside their derivation: its concept seats are the
admitted words, plus one cover per distinct document across the splits it
presents, plus the concepts its closes can mint, plus the fixed partition's
slack; its LTM rows are one presentation's rows plus the provisioned seed.
Nothing is derived from an epoch count, since each training epoch starts from
empty memories. The thought content store is the one principled exception: it
may grow to its exact size within a declared bound (Alec, 2026-11-11; see
[Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11)).

**Writing a test.** An expected value is computed in the test from its stated
formula, never obtained by calling the code under test and never pinned to a
decimal that a run happened to produce. Fixtures supply learned state through
the production writers (`definition_fixtures.agreeing_views` and
`learned_operation`, and `gate_data` for the trust gates), never by writing
counters. A test that needs an order or a part of speech that the model has not
yet learned forces it through one helper,
`test/order_fixtures.py::force_order(cs, word, order)`, which records the order
and settles the concept, so every forced order is declared and can be listed. A
certificate that forces a choice says so; a learning certificate forces
nothing, and a mechanism probe never satisfies a learning gate. A test of a
mechanism keeps the smallest operation menu that reads its sentence, so that it
asserts nothing about an untrained chooser. No natural language is named in
code, configuration, fixtures or test names; the words of a corpus are data.

**Failures, ports and producers.** A broken calculation fails loudly: no
handler swallows it and no default hides a missing field. A case that fails in
a sweep is not rerun for a pass. One diagnostic copy outside the suite replays
it to attribute the failure, and the producer, or the failure to attribute it,
is recorded; replay is diagnosis, not a retry. A red case whose fixture lags the
design is ported with its assertions unchanged. A port that changes an
expectation keeps its before and after text and its reason, and one that lowers
an expectation is recorded as a weak port. A producer defect is repaired at its
producer, never by a default that serves a fixture, and its whole test file is
rerun together with every test file that imports the changed module. Nothing is
retired without a record of what it did.

**Proportion.** A change is measured only where it can change a result. While a
change is built, only fast checks run: its direct before-and-after cases (red on
the source before the change, green after) and the whole test files that it
touches. When the change is complete, one full default sweep runs on exactly
that source, with nothing deselected, as its acceptance gate:

```sh
env -u RUN_TRAINED -u RUN_SLOW make test
```

A fixture port or a producer repair found by the sweep reruns its own files,
not the sweep. An incomplete sweep stops the work, and so does a failure of
either of two cases: `test_mm_xor.py::TestMMXorConvergence::test_convergence`
and
`test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`.
The grammar XOR reconstruction gate and the forced pending-premise pair
(`test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders`)
run inside every sweep. `TestXorExactCliReconstruction` is an unseeded learning
certificate: it is recorded when red, never a stop.

**Measurements.** Learning measurements (lesson arms, repeated fits, the
learning metrics) are taken once, after the implementation is finished, from
fresh unseeded runs with no retry. A null result is recorded as a null with its
budget; nothing is regraded, extended or drawn again. Before a learning
measurement is declared, one development epoch for each condition, on a
throwaway start, measures the wall time per sentence kind, and the declaration
is sized from it. A throughput figure names the source commit, the
configuration overrides, the device, the compile backend, the losses actually
executed and the workload. It measures warmed full training steps (forward,
backward and optimizer) and reports compile time, recompilations and peak
memory beside sentences per second, with reconstruction fidelity and the
declared parameter gradients; it varies sentence and output length, batch size
and basis size wherever the change under test can depend on them. A
`backend="eager"` graph-capture test establishes one-graph structure, not
throughput.

**Documentation.** Documentation is part of the suite. `test/test_doc_links.py`
requires every relative Markdown link under `doc/`, in `README.md` and in
`todo.md` to resolve to an existing file; external links and in-page anchors are
not checked. A retirement is certified by a test that asserts the absence of the
retired functions, attributes and configuration names, and such a certificate
may also read `data/model.xml`, `data/model.xsd` and the parameter tables of
[Params](Params.md): `test/test_priming_diffusion.py` asserts that the retired
`geometric*`, `primingSpread` and `primingMaxBoost` knobs have no table row. A
retired name is therefore written in prose, never as a table row.

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
diagonalization has an exact answer to find), and the carrier variants (the
same meaning under "equals", "is", "sums to", and the marker moved from word
to affix). Numerals stay opaque words, so the suite
tests mechanisms, never arithmetic competence. The corpus is produced in
the Viterbi order (one more step per item, bottom-up) by the `MM_math_chain`
generator and its successors; each run records
throughput so item 1's runtime blocker is measured where it bites. **The
same run is the compiler test at scale:** compiled and eager execution of
the identical corpus, seed and budget must agree in actions,
probabilities, costs and parameter updates, which no fixture-sized check
can show. Coverage, and what the suite cannot reach, is in
[Training, "The generated math suite"](Training.md#the-generated-math-suite-what-it-exercises-and-what-it-cannot-2026-10-20).

Long full-model work runs under the slow gate: repeated corpus-learning
experiments, multi-batch and checkpoint training, multi-step optimizer checks,
full compiled reconstruction and traversal integrations, production-size
configuration checks, and full CLI training or reconstruction. Small
layer-level, numerical, ownership, gradient, checkpoint, query and
compiled-kernel regressions stay in the default suite, and wider variants of a
fixture's checks remain available under `RUN_SLOW=1`. A slow mark keeps the
test's body and assertions. Run the slow checks of a training path when that
path changes, for example:

```sh
make test_all TEST_ARGS='test/test_output_walk.py --timeout 1800'
```

Memory- and time-triggered recycling permits larger case batches (256 cases / 16 files)
without retaining a worker past its resource boundary. The fast runner uses a
small concurrent pool under one aggregate reservation rather than one serial
worker; it retains process isolation and hard caps while avoiding needless
Python/PyTorch import waits. Per-case timings in the durable receipt identify
further expensive checks; a timeout is a failure that needs investigation,
never an automatic slow mark.

### The XOR gates

Every landing sweep runs the XOR certificates of the default tier: the six
grounded cases in `test_grounded_xor.py`, the three curriculum cases in
`test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse`,
the grounded XOR with word boundaries in `test_word_admission.py`, both
`TestXorExactCliReconstruction` gates in `test_explicit_dimensions.py`,
`TestMMXorConvergence::test_convergence` and `test_learns_xor_signal` in
`test_mm_xor.py`, and
`TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`. The two
grammar-XOR fits, `TestXorGrammarLearnsXor::test_xor_class_accuracy` and
`TestMMXorConvergence::test_mm_grammar_learns_xor_signal`, are in the trained
tier, where `RUN_TRAINED=1` fails them strictly while their phase artifacts are
absent. A result keeps its predictions and errors, failures included; a loose
signal threshold alone does not establish XOR learning. A case stopped by the
unchanged 8 GiB guard runs once more without that guard as a separate
diagnostic, never as a replacement passing gate. Item 1.8 will move the XOR
proofs to numbers in and out on the field
([last architecture round](plans/2026-11-11-item1-8-last-architecture-round.md),
§3.9).

**The grammar XOR gate** (`XOR_grammar.xml`) poses XOR to the design as it
stands. A continuous output is a modality: the numeric answer is an affine read
of the kept path's concluded sentence record, with no generate grammar
(`answerSynthesis` false). A gate tests the design and never earns a mechanism
of its own: a read-out, head, rule or reader added so that a gate passes is a
workaround, removed when found, and what then fails to learn is diagnosed. The
class certificate is measured by its criterion over ten unseeded trainings of
the fixture through its producer, `ModelFactory.run`, and fewer than five
passes stop the work: the failing runs are traced from their recorded initial
random streams (each epoch's kept composition, head cost and loss) and the
cause is repaired at its producer. XOR proofs set `omissionTarget` to 1, since
with two words an omitted operand leaves the answer undefined.

The class certificate requires all four answers on the correct side of one half
and a mean squared error below 0.05. The reconstruction gate reads the grammar
inverse (Alec, 2026-11-10): `reconstruct_grammar_sentence` runs generate on the
concluded sentence's own end slots, the occurrence writer and the common
`reverse`, and all four sentences must come back as their own word multisets,
each with an available inverse. Conjunction and disjunction are symmetric, so
the understanding carries no word order and a transposition is the only error
admitted. The System 1 reading of the observed field
(`reconstruct_field_sentence`) is printed beside it as an audit count and
asserted nowhere. Both certificates read one shared training
(`shared_training('xor_grammar')`); the reconstruction test keeps its
historical name while it asserts four of four.

**What the gates certify.** The XOR table measures the composition mechanism:
two words composed nonlinearly into one understanding, decoded back, and read
by an affine head with one owner per weight. It does not measure which operator
a construction means; grammatical learning is measured by the grammar lessons
and the trained-model evaluation, never by the four-sentence fixture. Under
identity by construction the roots of both conjunction and disjunction over the
gate words are affinely separable, so the class gate certifies identity
(anagram and length separation, exact reconstruction) and the head's
convergence on whichever root the chooser keeps, not the choice of operator;
the chooser is expected to be indifferent between the two. The operators differ
in meaning, and that difference is certified on meanings rather than here: a
union answers membership of either operand, an intersection of both. The class
result counts as evidence of composition only while the sum control stays at
chance: the same fixture with its binary rules replaced by `sum`, which composes
by the mean, leaves the pairwise dependency out of reach of an affine read and
must stay at the affine floor $\tfrac14$. A nonlinear reader in a numeric head
(a learned retrieval reader, or a normalization of each root) voids the gate.

**Reading an XOR result.** On the four corners the target $(0,1,1,0)$ is
orthogonal to the affine span of the inputs: centered, the target
$(-\tfrac12,\tfrac12,\tfrac12,-\tfrac12)$ has zero inner product with both
inputs, $(-\tfrac12,-\tfrac12,\tfrac12,\tfrac12)$ and
$(-\tfrac12,\tfrac12,-\tfrac12,\tfrac12)$. The best affine predictor is
therefore the constant $\tfrac12$, with mean squared error exactly $\tfrac14$
(Minsky and Papert 1969), and an affine reader over an additive representation
can do no better; the sum control is this theorem run as an experiment. With
one nonlinear layer under the reader the global minima are at 0 (Rumelhart,
Hinton and Williams 1986), and the symmetric configuration with output
$\tfrac12$ everywhere is a saddle region on which descent can stay for a long
time (Hamey 1998; Sprinkhuizen-Kuyper and Boers 1998; Fukumizu and Amari 2000).
A run therefore ends at 0, where XOR is learned, or on the plateau at
$\tfrac14$, which is a rate to report under unseeded starts and a defect only
if every run lands there. A final error strictly between 0 and $\tfrac14$ is
not a stationary point of an affine reader over a fixed representation: either
the budget ended mid-descent, which the trajectory shows, or the
representation moved under the reader.

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

**Mechanism certificates and learning claims.** A mechanism certificate runs
on the ordinary training path: a tiny corpus of real documents presented
through the real driver, with a batch of at least two rows that depart at
different rounds, asserting on what was committed (LTM rows, MTM, the
history). Tests on hand-constructed rows remain unit tests and certify no
mechanism. A mechanism certificate never certifies learning. A claim that the
model has learned something useful needs held-out data, predeclared metrics and
thresholds, and controls at matched resources: ordered against shuffled and
context-free input for prediction, and direct-answer and equal-compute
no-subgoal baselines for thinking. Variation across runs is reported, and so
are null and negative results; counting records or activations is no
substitute. Three comparisons separate the value of an internal question: vary
the question's content with everything else fixed; keep the questions and
remove or corrupt the intermediate results; and mask only the stored questions
from later reads. Asking adds no knowledge by itself, re-reading one's own
answer is not independent corroboration, and nothing rewards asking more.

Phase prerequisites are separate from learning outcomes. No phase artifact is
registered yet: every registry entry is null, so each phase certificate skips
with its phase and reason until its producing lesson registers one. The retired
`stage3`, `np_definite` and `np_description` signed-coordinate artifacts
(October 20 migration decision) are never read. The column-copula test remains
an unconditional mechanism contract.

`restore_phase(loader)` verifies the marked prerequisite before calling the
consumer's loader with its resolved path. There is no training or fresh-model
fallback. Paths are relative to the repository root, or absolute. For an
external artifact, select a complete registry with
`BASICMODEL_PHASE_ARTIFACTS=/path/to/registry.json`. FineWeb entries also name
`config` and optionally `minimum_sentences` (default 1,000,000); the completed
training-progress check still applies. A `stage4e` entry hashes the produced
`kept-programs.json` that the certificate grades. This replaces the old
FineWeb and constituent artifact environment switches.

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
| 14 | retired (2026-10-18) | Category-VQ context smoke; covered by the native argument-concept certificates in `test/test_parts_of_speech.py` |
| 14 | weekly (slow contract) | `test/test_echoic_decoder.py::test_same_walk_infers_binary_and_unary_without_a_journal[True]` |
| 14 | weekly (slow contract) | `test/test_compiled_word_chunk.py::test_tensor_peer_word_loop_rejects_external_full_stm` |
| 13 | retired (2026-10-18) | Category-VQ allocation/EM smoke; argument concepts now use shared native concept storage (2026-11-08) |
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

The tier implementation is in `test/slow_tests.py`, keyed by exact
parameterized node IDs. Two names in the table are retired and kept as
provenance, not tests to resurrect:
`test_eight_corpus_sentences_train_at_distinct_fork_rounds`, and the
bound-declarative departure-diversity certificate
`test_forced_ordinary_bound_declaratives_open_no_episode`, which
`test_exploration_coverage.py`'s trained-artifact certificate replaces in the
default tier with its stated probability bound. The MM XOR and explicit XOR
convergence and reconstruction certificates are default tests; artifact
certificates still require the declared trained artifact.

## Limits and isolation

- One full receipt per user across worktrees, enforced by a nonblocking file
  lock. Selected-file, `-k`, and marker runs may proceed concurrently.
- One result records two memory phases. Ordinary cases use
  `max(1, min(cpu−4, aggregate_bytes // (8 GiB)))` one-thread workers:
  **2** on an eight-core, 24 GiB machine. Each reservation is
  `min(8 GiB, aggregate // workers)`: **8 GiB** on that machine. A worker receives
  at most 256 selected cases from 16 files. `--workers` may lower this pool;
  its cap follows the same formula. User `-m` selection is applied before
  partitioning, so every selected case belongs to exactly one phase.
- A case whose measured solo footprint exceeds the ordinary half-cap
  allowance is marked `large_memory`. The second phase runs **one case per
  fresh worker**, under an **8 GiB** cap, with `aggregate // cap` workers:
  **2** on the 24 GiB machine. If the aggregate itself is below 8 GiB, it is the ceiling and
  only one large worker fits. In both phases workers times cap is at most
  the aggregate. Collection runs alone within the same aggregate.
- The aggregate stays physical RAM minus **8 GiB**: **16 GiB** on that machine, with
  the sampled aggregate guard as a backstop and a `taskpolicy -m ... -P kill`
  ceiling per worker. `--memory-gib` can lower the aggregate and cannot take
  the machine's reserve. An unmarked case that exceeds its reservation
  stops with `memory` and its case name; neighbouring cases cannot consume
  its reservation. The inherited sentence word bound remains 256.
- A worker finishes its current case and recycles at **half its reservation**
  (**4 GiB** in this default ordinary phase), or 80% of its deadline.
  A fresh worker takes the remaining cases, preserving one reported
  completion per selector. Each receipt records the case's starting,
  finishing and sampled peak physical footprint for subsequent measurement.
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
`@pytest.mark.shared_training("name")`. In the ordinary phase the bounded runner
keeps the selected members of that group in one worker, including the weekly
run at batch size one. Recycling waits until the group's final selected test;
the memory and time ceilings remain unchanged. A `large_memory` case instead
gets its own fresh worker in phase two. Selecting one test alone does not add its
partners. `TestXorGrammarLearnsXor` and `TestXorGrammarReconstruction` share
one training this way, so each training reports both bars.

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
RUN_SLOW=1 BASICMODEL_DEVICE=mps .venv/bin/python test/test_report.py test/test_compiled_word_chunk.py
```

Worker receipts identify the resolved device. GPU timing follows the throughput
rule above, and device selection implies no speedup. The same aggregate memory,
thread and deadline limits apply.

macOS accounting uses `proc_pid_rusage` physical footprint, including compressed
memory. RSS alone can become misleading during compression. Each macOS worker
has its reserved worker limit (at most 8 GiB by default) applied through
`taskpolicy -m ... -P kill`, while the supervisor samples all active process
groups and enforces the aggregate reservation. Linux accounting uses RSS plus swap. The monitor samples each
process group and known children, including compiler subprocesses; sampling can
allow brief overshoot.
Timeout, memory exhaustion, unavailable accounting or interrupted execution
terminates the worker and its children and returns failure. Linux enforcement
is sampled, without the additional macOS kernel cap. Unsupported platforms
fail before running tests.

## Evidence and reports

Each bounded invocation writes its run record, the receipt, into a new
`output/tests/<timestamp>-<id>/` directory, or the new directory named by
`--run-dir`. It contains:

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

The runner's own tests (`test/test_bounded_test_runner.py` and
`test/test_bounded_test_recycling.py`) exercise real hanging processes,
resistant children, aggregate child memory, native compressed-memory
accounting, monitor failures, lock exclusion, fresh-worker coverage, teardown
failures, source mutation, empty selection and interruption with persistent
evidence; `test/test_training_device.py` and `test/test_test_modes.py` cover
device dispatch and the slow switches. A runner revision is validated only by
these tests, run on its own source.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#fast-aggregate-runner-evidence-september-19) (2026-10-30): "Fast aggregate-runner evidence (September 19)".*
## Future work: bounded fixture reuse

A reused fixture would be one compact, preallocated model per compatible
family inside a bounded worker, keyed by configuration contents, grammar,
backend, dtype and device, and declared capacity, and sized for that family
rather than for production tables. There is no unbounded global model cache:
the central memory cap, deadlines and worker recycling stay, incompatible
families are evicted, and the existing fixed-capacity contracts remain.
Construction, ownership, checkpoint-migration and reset-isolation tests keep
fresh models. A complete reset restores in place the weights, buffers,
gradients, optimizer state, random state, dictionaries and allocators, row
activation, grammar ownership, dataset and configuration globals, STM and LTM,
thought history, expectation stream, traces, hooks and retained autograd
graphs. `util.TheCompileBackend` is read once at import, so a fixture selects
and restores the backend explicitly rather than by changing the environment.
Reuse is accepted only when the same selection, seeds, orderings and backend
give identical outcomes, storage and optimizer identity are preserved, no
gradient or memory state crosses between tests, memory stays bounded, and the
time saving is reproducible, measured separately for construction, compilation
and warmed steps. No saving has been measured.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#selected-meaning-and-one-controller-september-20) (2026-10-30): "Selected meaning and one controller (September 20)".*
## Weekly slow coverage

`make test_slow_weekly` uses the existing virtual environment and does not depend
on the venv rebuild stamp. It selects central `slow` markers, historical
`RUN_SLOW` decorators/body gates, and the three `RUN_MPS_SLOW` cases. The older
switches retain their original gating behavior; the common marker now exposes
them to selection. CUDA-only cases keep their availability skips.

This machine's weekly target uses the suite's CPU fixture contract; the three
MPS-specific tests explicitly switch to MPS. Weekly ordinary cases retain an **8 GiB
per-worker ceiling**. The production objective measurement in
`test_objective_conflicts_slow.py`, at its unchanged batch of 28, has a
**24 GiB per-worker ceiling**; it makes one ownership measurement. Both tiers
run one worker at a time under a **24 GiB aggregate reservation** and the
existing **30-minute worker deadline**.
The ordinary full sweep uses the same 8 GiB worker ceiling.

The same weekly target also runs `python bin/Legacy.py` and the three
`bin/etc` inline unittest suites once, under the weekly ordinary 8 GiB worker ceiling.
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

A Sunday 03:00 local launchd schedule is proposed and not installed.
Compiler tests retain capture; ordinary ownership and gradient tests can use
`eager_reading`, which executes the same loop bodies with ordinary autograd.
Setting `MODEL_COMPILE=eager` alone still captures higher-order loop graphs.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#october-3-decoder-exploration-operators-and-one-attention--review-candidate) (2026-10-30): "October 3 decoder exploration, operators and one attention".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#item-50-forgetting-candidate-2026-10-19) (2026-10-30): "Item 5.0 forgetting candidate (2026-10-19)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#item-4-step-0-second-repair-round-2026-10-19) (2026-10-30): "Item 4 Step 0, second repair round (2026-10-19)".*
### Large-file corpus tier (item 2.7, 2026-10-29)

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
configuration.


## Strict xfail for an unresolved deviation (Alec, 2026-11-01)

Alec: "You can implement everything and xfail what tests you need to, let's
lock down what is working going forward." A standing test or certificate that
fails on a deviation that a change does not resolve is marked
`pytest.mark.xfail(strict=True, reason=...)`, with the reason naming the
deviation, and the work proceeds. Only a source-hash mismatch or a numerical
divergence stops it; both stay fail-loud. `make test` remains the criterion: a
strict xfail that starts passing fails the suite, and is removed together with
the deviation it named.

### Unexpected recompilation checks (2026-11-03)

`test_compile_guard.py` checks that each declared model execution phase admits its initial graph, reuses it for compatible inputs, and raises on a later shape or owner-attribute guard failure. The failure must reach the caller even when an outer compiler policy requests silent fallback. Initial training/evaluation and decoder trial phases are explicit; arbitrary new shapes are not warmup exceptions. `test_generation_budget.py` checks the construction-time free-generation limit and preserves complete recorded-derivation replay independently of it.

Item 2.3's concept radii, ordered effective relation evidence, definition-index threshold and lexical-admission certificates (`test_trust_round1.py`) run in the default sweep.

Item 2.3 admission uses `WordSurfaces.is_lexical_form` in corpus, aligned, grammar-word and eager paths; `test_word_admission.py` certifies Unicode classification and retained punctuation percepts without lexical identities.

Item 2.3's named-sign certificates use `test_trust_round1.close_named_sign` on the paired fixture to close a kept sign through `define_pending(null_referent=True)`, with no explicit promotion; the full `SentenceFixture` case also covers an already resolved remembered endpoint.


### The trained tier (`RUN_TRAINED`)

`RUN_TRAINED` is independent of `RUN_SLOW`. `@pytest.mark.trained('phase')` skips before fixture setup unless `RUN_TRAINED=1`; the phase and reason are counted in terminal output, `skipped_trained_by_phase` in JSON, and the HTML report. Enabled tests require the exact artifact through `phase_artifacts.PhaseRegistry`, failing on unknown phases, null or absent files, wrong hashes and absent producing receipts. `BASICMODEL_PHASE_ARTIFACTS` selects a private registry for the tier's own certificate. Existing `phase` certificates are unchanged. The registry names `words`, `interpret`, `clause`, `ask`, `verb`, `modifiers`, `determiners`, `adverb`, `prepositions` and `relations`, each with a null artifact and a producer row naming the lesson runner, `bin/corpus_lessons.py`, and its row in the lesson order.

The tier holds the ten functions below (the answer-quality function with both parameter cases), with their assertion bodies unchanged. This prerequisite replaces the two syntax-learning expected-failure decorators and the successor and free-derivation `RUN_SLOW` skip guards, and central `slow` markers no longer gate a `trained` test.

| Test | Phase |
|---|---|
| `test/test_meronomy_ladder.py::test_cold_start_learns_space_as_the_basic_boundary` | `words` |
| `test/test_math_thinking_training.py::test_stage_zero_direct_arithmetic_learns_as_syntax` | `verb` |
| `test/test_math_thinking_training.py::test_stage_one_dependency_chains_learn_as_syntax` | `relations` |
| `test/test_math_thinking_training.py::test_successor_is_learned_as_a_verb` | `verb` |
| `test/test_what_training.py::test_joint_training_keeps_both_primary_costs_in_band` | `ask` |
| `test/test_what_training.py::test_supplied_text_answer_quality_improves_with_training` | `ask` |
| `test/test_output_path_supervised.py::test_output_path_memorizes_supervised_labels_on_the_serial_topology` | `ask` |
| `test/test_reconstruction_roundtrip.py::test_mm20m_grammar_free_derivation_roundtrip` | `clause` |
| `test/test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal` | `mm_xor_grammar` (2026-11-08) |
| `test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy` | `xor_grammar` (2026-11-08) |

The two grammar-XOR fits are in this tier by Alec's decision of 2026-11-08. `mm_xor_grammar` registers a null artifact with the test's own `_fit_xor` producer: `data/MM_grammar.xml`, seed 41001, 900 epochs and the 0.20 bar. `xor_grammar` registers a null artifact with `ModelFactory.run(data/XOR_grammar.xml)` as producer and the criterion of four answers correct with MSE below 0.05. The `shared_training('xor_grammar')` group and `TestXorGrammarReconstruction` remain in the default suite.

The criterion is acquisition from presented sentences or a required free chooser decision in an untrained first batch. Numeric XOR convergence, direct tensor optimization, loss/gradient ownership, admission and checkpoint mechanics, and fixtures with explicit forced choices remain default/slow mechanisms under their existing rules; older artifact-backed `phase` tests are unchanged. The three-document token case retains its body and remains a mechanism: `test_reading_round.symbol_interpretation` uses the existing `OperationSelectionLayer.set_stage_menu` on fixture instances to exclude `sign`, fixing the second decision at the symbol without a production setting.

`definition_fixtures.learned_operation` sits beside `agreeing_views` and requires an explicit kept-use count and fixture occupants. It calls `PartsOfSpeech.observe_use`, the same application writer used by `observe_kept`; it never writes counts directly. It supplies input member views to native argument concepts; complete kept programs retain their ordinary output/head-flow member views. `test_trust_round2` certifies one and three kept uses, argument agreement and shared membership, fixed and identity exemptions, checkpoint pointers/catalogue/native argument geometry/use counts/effective pairs, epoch persistence, empty occupations and hand-counted curriculum measurements. These and the private-registry trained-tier certificate run in the default sweep. Gate mechanisms are certified separately by the closed-gate fixtures below.

The moved trained-tier tests still train a fresh model after the prerequisite is checked; restoring the registered artifact into that model belongs to the lessons item. The two strict expected failures recorded before the tier (`test_stage_zero_direct_arithmetic_learns_as_syntax` and `test_stage_one_dependency_chains_learn_as_syntax`) are now trained-tier skips without `RUN_TRAINED=1`.

`test_trust_round2` also certifies first-writer operation provenance, naming that clears an operation cap, ownerless catalogue reattachment and bind result-order eligibility. The three-document mechanism uses the existing instance menu with `STOP`, `interpret` and `union`: interpretation retains the symbol and union reduces the word and punctuation spans. The user-ingestion fixture supplies union through `learned_operation` and checks the stored source pair plus effective evidence by the measured operation formula; the checkpoint inventory includes the operation column and catalogue extra state.

Both parameters of `test_identity_threading::test_compose_bind_locations_share_the_operation_chooser_and_zero_dummy` deposit each antecedent with order 1, as a working token requires.

`test_trust_gates.py` uses `gate_data` to close gates through construction data and `gate_fixture` to attach the paired owner to the ordinary clause writer, without unused thought rows or resizing. Its certificates cover named-row caps with identity maps, opening and later falling readiness, operation and fixed-relation caps and exceptions, lift definiteness, epoch/checkpoint persistence and missing-state rejection, open default and closed lesson models, readiness reports without lesson execution, empty lift classes and invalid data. In `test_trust_round2::test_fixed_operations_and_identity_mechanics_stay_at_one` the part row carries its catalogue index and still contributes operation trust one. All existing test configurations retain open starts.

The 2026-11-08 argument-concept correction uses `argument_fixtures.argument_inventory` and `view_counts` for native seated words and explicit concept views. `test_argument_concepts` covers zero-view declarations, atomic capacity refusal, native orders, native checkpoint addresses, member centroids, shared roles, generic nested-class feedback, preserved forms/radii, and whole-value chart behavior. The nested-class certificate has no grammar catalogue. `test_trust_gates` now requires zero empty-class readiness and tests empty `except` elements. The input carrier retains repeated lexical occurrences; `test_direct_operand_occupations` observes exact writer calls and per-class view increments rather than a stored word-by-role count table. The kept-reading certificate compares the view increments against its independent journal replay and proves annotations and rejected trials cannot write them.

Representation ports replace exclusive argmax/probability assertions in `test_parts_of_speech`, denotation-order count slices in `test_orders_learned`, count migration in `test_grammar_checkpoint`, count snapshots in `test_part_closing`, and the operation/gate probability formula in `test_trust_round2` and `test_trust_gates`. `test_reference_participants` supplies actual lexical members separate from their object referents. `test_config_scoping` includes the fixed declaration seats in reference capacity while retaining ordinary input/output sizes. The two small round-1 paired fixtures and round-2 fixtures declare capacity for the formal concepts before construction. No mechanism is retired. Artifact-backed restoration requires a new native-geometry artifact; old count artifacts fail loudly. `RUN_TRAINED`, phase prerequisites, and the default sweep's XOR and pending-premise cases retain their tier rules.

Argument-concepts continuation certificates cover coverage fit, separability/evidence, boundary scope and scheduling, fit-weighted centroids, neutral unmeasurable charts, unique membership edges, object/binding isolation and named formal-class proof paths.

The argument-concepts-iii ports retain consolidation → promotion → one `ConceptHierarchy.refresh` derivation → condition, and read order-independent class fits after the boundary: aligned classes each have coverage one even with different radii.

### Item 2.0 certificates

Item 2.0's direct certificates are in `test/test_item20_round*.py`: the memory
and journal move, thought WHEN stamps, absent nonoperation journal rows,
symbol-description rejection and the shared memory snapshot base (round 1);
admission and terminal routing, recorded action witnesses, ownership and
address migration, and candidate-owned action bases with free walks that need
no action record or live STM, including compiled gradients (round 2, among them
[test_item20_round2_generation.py](../test/test_item20_round2_generation.py));
two-sided gradient and cost routing, participation re-reads, rates, clamp
velocity, output typing, and graph lifetime with an unseeded packed-sentence
scan and an optional chooser-owner certificate (rounds 3 and 4); reference
paths, state errors, shared strata, seat authority and basic percepts
(round 5); the common decoder, field gradient, output and timed memory
(round 6); and concept episodes and the one sentence and thought clock
(round 7).

### Data certificates of the training corpus (item 0.5, 2026-11-12)

Four fast certificates read the frozen corpora and never regenerate them:
`test/test_math_corpus.py` (hashes, grade order, every answer and
intermediate answer, held-out and beyond pairs), `test/test_game_corpus.py`
(hashes, `verify` on every row, splits, transfer words kept out of the
games), `test/test_lesson_stages.py` (hashes, phases in lesson order,
answers, held-out words, the lesson driver reads every stage) and
`test/test_training_corpus.py` (stage hashes, part A disjointness, the
held-out cost set, the next-word gate's documents, the recorded part B
scan). Local part B data is checked by hash only where present.
