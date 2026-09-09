# What spec: remaining deliverables and execution plan

> **Status:** audit written 2026-09-08 against `main` a8a5142; Steps 1-4, 6 (branch diagnostics), 8 (root through `reverseOutput()`) landed 2026-09-09 (uncommitted at time of writing)
> (BasicModel PR #8 + the ConceptsFromPercepts / reconstruction-priority
> landing). Governs the next iteration of
> `doc/specs/2026-07-27-teaching-modes-and-next-iteration.md` (the "What
> spec") and `doc/WhatSpacetimeDesign.md`.

## 0. Headline

The What spec's *question* machinery is built (questions, `Data.what()`,
LTM slot parity, iterative `think()`, target-free chooser context, and — as
of 8975365 — the absolute `.where` given to the model). Its *answer*
machinery is not: there is no per-call `Understanding`, no
`Model.reverseReconstruct()`, no `ConceptualSpace.synthesize()` /
`PerceptualSpace.synthesize()`, and **no `Model.reverseOutput()`**. `Model.what()`
still wraps the established forward and its direct projected head, and the
model-level `reverse()` still carries the ambiguous "generative" meaning the
spec retires.

**The reconstruction (`reverse()`) path and the answer (`reverseOutput()`) path
must be two paths.** `reverse()` / `Model.reverseReconstruct()` is the algebraic,
input-associated inverse that ends at `InputSpace`; `Model.reverseOutput()` starts
from a *resolved answer symbol* and descends through conceptual and
perceptual synthesis to `OutputSpace`. They share parameters (including tied
invertible matrices) but not carriers; either may run first without changing
the other's result (spec §5.2–5.4). Nothing of the second path exists yet.

## 1. Step-by-step status (spec §10)

| Step | Deliverable | Status |
|---|---|---|
| 0 | question representation, `Data.what()`, LTM slot/parity, chooser context, compatibility adapters | **done** (PR #8, 8975365) |
| 1 | explicit per-call `Understanding` (perceptual context, conceptual state, symbolic state, reconstruction carriers) replacing successive overwrites of `Space.subspace` | **landed 2026-09-09**: `bin/Understanding.py`, `BasicModel.understand()` / `_capture_understanding()`; the forward-local merge-diff carriers travel in `reconstruction_carriers` (found by making `reverseReconstruct()` repeatable). Remaining: a carrier-pure reverse chain (the Space-level inverses still write the live `Space.subspace`; `_synthesis_guard` isolates them for now). `test/test_understanding_reconstruct.py`. |
| 2 | collective field repair (§5.5) | **done** (a8a5142) |
| 2 | `Model.reverseReconstruct(understanding)` over the inverse path; `Space.reverse()` reserved for algebraic inversion; named `input_reconstruction` cost | **landed 2026-09-09**: the `lossRev` block moved verbatim into `reconstruct(understanding, target=None, train=False)`; `runBatch` only scores; repeatable and order-independent with `reverseOutput()`; `primary_costs()` reports `input_reconstruction` (legacy channel), `input_reconstruction_reverse` (reconstruct()'s cost), `answer_construction`. |
| 3 | `ConceptualSpace.synthesize()` / `PerceptualSpace.synthesize()`; `OutputSpace` as modality adapter over constructed percepts; `Model.reverseOutput()` | **landed 2026-09-09, parameter-gated** (`<answerSynthesis>`, default false = byte-identical): `SubSpace.carrier_like()`, `Space.synthesize()`, `ConceptualSpace.synthesize()` (symbol -> concept via the terminal WholeSpace inverse), perceptual synthesis via `_reverse_body` on a fresh carrier, `OutputSpace.from_percepts()` (lazily built percept adapter), `Model.reverseOutput()` under `_synthesis_guard` (hides reconstruction carriers, restores live state). `test/test_output_synthesis.py`. Grammatical resolution by relation (2026-09-09): present/supervised/inference = identity; `past -k` = RECALL from the discourse ARMA ring; `future +k` = PREDICTION via `predict_next()` rolled forward without committing; cold memory -> identity flagged `resolved=False`. Remaining: named perceptual bindings beyond slot selection; reasoning-kernel resolution for prompted inference questions. |
| 4 | `Model.what()` = `forward()` → rational symbolic evaluation → `reverseOutput()`; grammar chooser / LTM / answer symbol / derivation / prefix / synthesis references connected | **landed 2026-09-09 under `<answerSynthesis>`**: `what()` routes `produced` through `reverseOutput()`; `runBatch` scores `construction.actual`. The projected head remains the migration oracle when the gate is off. |
| 5 | `Data.what()` is the answer-loss authority; `input_reconstruction` / `answer_construction` names | **done for all families** (temporal text answers, 2026-09-09: `reverseOutput()` realizes a `surface` through `_reverse_perceptual` + `inputSpace.reverse`; `_embed_answer_texts` embeds the target sentence under the guard; scored band-aware by `_reverse_event_loss`, masked to answerable rows). Earlier:: `_what_answer_target()` builds the target from `_last_what_desired` with row masking, wired at the `lossOut` site; `primary_costs()`; `test/test_what_training.py` (supervised target == loader tensor, masked inference/text rows, no-scoreable-row omission, per-family training steps, desired answer never recorded as the response, strict-xfail Step 7 placeholder). Past/future targets are only scoreable once Step 3/4 lets the head emit input-shaped answers. |
| 6 | joint losses from one `Understanding`, reconstruction-priority projection, one optimizer step, gradient-norm/cosine diagnostics, tied-operator derivatives | **mostly landed**: priority projection + one-step seam (a8a5142); `runBatch` reuses `what()`'s `Understanding` for `reverseReconstruct()` (2026-09-09); sampled `branch_gradient_diagnostics()` under `<branchDiagnosticsEvery>` (a read before backward, never an update). Tied-operator autograd-vs-FD test added (`test/test_tied_operator_gradient.py`). |
| 7 | train present/past/future; ablation shows causal use of temporal context | **done 2026-09-09**: a learned, zero-initialised `question_conditioner` (target-free 29-wide context -> answer-symbol root slot) at `_resolve_answer` puts the question on the differentiable answer path; `test_temporal_question_content_is_causally_used` trains with the question as the only distinguishing signal and shows a wrong question hurts by >1.5x while the input is held fixed (measured: untrained right = wrong = 0.34446; trained right 0.00054, wrong 0.6751 -- training improves the right-question answer ~600x and makes a wrong question actively misleading); `test_swapping_question_positions_changes_the_answer`. Earlier block (for the record): on MM_xor (+`<transformChooser>mlp`, `<answerSynthesis>`, discourse) the MLP chooser scores every batch and `_what_bias` is applied, but `what_projection.weight.grad` is `None` even with `superposition_temperature=0.5` -- the chooser scores are not on the loss graph in this configuration, so the question's absolute `.where` cannot yet influence the answer by learning. Prerequisite: put the chooser (and its What bias) in the gradient path (the two-pass soft-superposition workstream), then flip the strict xfail `test_swapping_question_positions_changes_the_answer` to required. |
| 8 | LTM-driven thought: independent input/output halves, subquestion feed-forward, grammar-driven root scoring | **substrate done** (PR #8); root answer built through `reverseOutput()` when `<answerSynthesis>` is on (`test_think_constructs_the_root_answer_through_output`). Remaining: grammar-driven (non-identity) resolution. |
| 9 | retire Teacher-oriented APIs and the direct head; curriculum; B24 benchmark after cutover | **cutover + curriculum landed 2026-09-09**: canonical `data/BasicModel.xml` runs `<answerSynthesis>true` (the direct head is the migration oracle only), `<synthesisBindings>4`, `<whatCurriculum>full` (present -> +one-step past/future -> +inference, `whatCurriculumRatio` 0.25, `whatCurriculumDistance` 1) through `runEpoch` -> `Model.run()`; fixtures keep the defaults. Teacher: `Teacher.What` is a documented adapter over `Data.what()`; Teacher still owns the per-batch error registry (`teacher.add`), so 12.16 is PARTIAL by design until the registry moves to the model. B24: see section 7. |

## 2. Required-test coverage (spec §11)

| Group | Covered today | Missing |
|---|---|---|
| Interfaces & compatibility | `Data.what(present)`, byte-identical adapter (`test_what_spacetime`) | one shared `forward()` result for both branches; `reverseReconstruct()` / `reverseOutput()` reachability; order independence |
| Reconstruction & answer paths | invertibility contracts, collective recovery, readout sparsity (`test_invertibility_contract`, `test_concepts_from_percepts`, `test_concept_readout_l1`) | carrier separation; symbol → conceptual → perceptual → `OutputSpace` trajectory; no SymbolicSpace→OutputSpace shortcut; named-binding-only perceptual context |
| Coordinates & targets | zero-based/stable indices, document/split boundaries, temporal meaning changes target, coordinates in context not output (`test_what_spacetime`) | ablation harms address-sensitive tasks (xfail placeholder) |
| Learning isolation | target unreachable from `what()` call graph; LTM stores actual response; inference attaches without relabel (`test_what_spacetime`, `test_what_training`) | none pending beyond Step 4 re-verification |
| Loss & gradients | priority projection family (`test_reconstruction_priority`) ; `primary_costs()` names (`test_what_training`) | contextual-completion reconstruction gradient; tied-operator autograd-vs-FD; branch-point diagnostics; exact-identity-is-not-learning guard |
| LTM parity & thinking | all listed parity/closure behaviours (`test_what_spacetime`) | root answer scored via `reverseOutput()` |
| Grammar & performance | slot ops / temporal use in trace | output derivation/prefix/synthesis refs in trace; separate reporting of present/past/future/supervised quality; B24 within 15% |

## 3. Execution order (recommended)

1. **Step 1 — `Understanding`.** Introduce an immutable per-call value
   returned alongside the established forward tuple (adapter keeps the
   tuple). Fields per §5.1. No behaviour change; tests: forward result
   exposes it, contains no desired answer, and a second `forward()` does not
   mutate the first's `Understanding`.
2. **Step 2 — `Model.reverseReconstruct(understanding)`.** Move the current
   inverse orchestration (the model-level `reverse()` call chain) behind it;
   return `(reconstructed_input, input_reconstruction)`. Reserve
   `Space.reverse()` for algebra. Tests: uses tied inverses when
   `invertible`; consumes only `reconstruction_carriers`.
3. **Step 3 — synthesis + `Model.reverseOutput()`.** `ConceptualSpace.synthesize`
   and `PerceptualSpace.synthesize` as generated-carrier entry points (may
   reuse inverse-direction operators; may not read reconstruction carriers);
   `OutputSpace.forward(answer_percepts)`. Keep the old projection as a
   migration oracle only. Tests: trajectory, carrier separation, order
   independence, no shortcut.
4. **Step 4 — `Model.what()` through `reverseOutput()`**, then **Step 5 completes**
   (past/future targets become scoreable: the answer is input-shaped), then
   **Step 6** joins the two costs from one `Understanding`, **Step 7**'s
   ablation test turns from xfail to required, **Step 8** root scoring,
   **Step 9** cutover + B24.

## 4. Follow-ons already noted elsewhere

Finer intra-datum `.where` rung, corpus-byte rung, document-boundary bit
(`WhatSpacetimeDesign.md` §2).

## 5. WIP landing (a8a5142) tie-off, 2026-09-09

- `MultiOptimizer` gained the torch-optimizer surface callers rely on:
  merged `state` (parameter -> state view over the children),
  `add_param_group` (routed to the dense child; used by the lazily built
  leaf-distill head), and `state_dict()` now carries torch-style `state` /
  `param_groups` views alongside the per-child `optimizers` layout the
  checkpoint remapper consumes (`load_state_dict` accepts both).
- `test_dual_towers` key pin re-baselined for the live
  `concepts_from_percepts` / `concept_source_readout` modules and the
  relocated stage butterfly (+63/-15 keys on MM_20M_xor; grammar config
  683 -> 731).
- `test_word_store::test_ws_word_whole_registry_resolves_to_rows` was
  order-dependent (it rode on an earlier test's direct
  `_autobind_word_wholes` call); it now drives the real sentence-boundary
  path (`dispatch_per_row_reset`) itself.
- Latent, pre-existing (also on 8975365): on the serial word-store smoke
  config a SECOND bare `runBatch(train=True)` raises an autograd
  in-place error even with a boundary reset between steps; `runEpoch` is
  the supported driver and is unaffected. Not chased here.

## 6. Naming and dedicated synthesis weights (Alec, 2026-09-09)

- `Model.reconstruct` -> `reverseReconstruct`, `Model.output` ->
  `reverseOutput`: both are duals of `forward()` along the reverse path.
- `reverseOutput` has its OWN weights: per-Space `synthesis_layer`
  (`InvertibleLinearLayer`, identity at build) applied after the shared
  inverse chain, plus `OutputSpace.percept_adapter`. Lazily built modules
  are queued by `_collect_fresh_synthesis_modules` and handed to the live
  optimizer by `runBatch` via `add_param_group` (previously the adapter was
  built after `getOptimizer` and never stepped -- fixed).

## 7. Spec sweep, 2026-09-09 (after Step 7)

- Named perceptual bindings (5.3 / 11): `_select_perceptual_bindings` names
  the `<synthesisBindings>` most salient perceptual slots; they ride in the
  derivation (`bindings`, `synthesis_references`) and trace; synthesis adds
  ONLY those slots. Tests: bound slot changes realization, unbound cannot;
  default 0 keeps context out of the answer.
- Identity guard (11): `reverseReconstruct` flags an exact unmasked
  zero-cost identity (`_reconstruction_identity_flag`, warn-once) instead of
  counting it as learning.
- Reporting (11 grammar & performance): `what_report()` gives per-family
  mean `answer_construction` / `input_reconstruction`, thinking episodes /
  mean iterations / forced-closure rate, batches / sentences / sentences-per-
  second (seconds accumulate when the driver supplies them).
- Reasoning hook (5.3): prompted supervised/inference questions consult
  `answer_query` when reasoning is enabled; posture/confidence enter the
  derivation trace (`source="reasoning"`).
- Curriculum (Step 9): `_curriculum_questions` counts training batches
  (not `runEpoch`'s row step) and cycles past/future(/inference) trials,
  starting with past.
- Joint-training band (12.14): `test_joint_training_keeps_both_primary_costs_in_band`.
- Throughput (12.15): `bench_train_word_bucket.py --config data/BasicModel.xml
  --corpus --batch 24 --warmup 2 --steps 12` on MPS, same tree:
  answerSynthesis+curriculum ON = 12.20 sentences/s (median 1.969 s/step),
  OFF = 12.34 sentences/s (1.947 s/step), pre-change a8a5142 BASELINE =
  12.27 sentences/s (1.957 s/step; measured in a worktree with only the
  negative-row clamp applied, since its bench crashed otherwise). The
  cutover costs ~0.5% versus the baseline and ~1.1% versus gate-off --
  inside the 15% band (12.15 met).

## 8. Codex review fixes, 2026-09-09

- P1 checkpoint reload: answer-path modules with construction-known widths
  (both Space `synthesis_layer`s, the `question_conditioner`) are built
  eagerly when `<answerSynthesis>` is on (`_materialize_answer_path`), and
  any answer-path module named in a checkpoint is instantiated from the
  saved shapes before the key audit
  (`_materialize_answer_path_from_checkpoint`); optimizer registration
  de-duplicates against parameters already in the optimizer. Test:
  save -> fresh model -> `load_weights(require_match=True)` -> identical
  weights and answer.
- P2 per-row resolution: `_resolve_answer` resolves each lane's own
  relation/offset (`AnswerDerivation.row_sources`; `source="mixed"` when
  they differ; a cold lane leaves only that lane unresolved).
- P2 recall layout: the ARMA ring's fill phase writes the newest rep at the
  LOW end (`cap-count-1`) and flips to newest-at-tail once rolling, so
  indexing `_s_history[:, -k]` returned the oldest during fill. `past -k`
  now reads the model's own per-row chronological `_what_recall_history`,
  appended at every discourse observe (`_observe_discourse`), cleared with
  the row's hard reset. Tests cover fill-phase recall, mixed offsets and
  relations, and cold lanes.
- Also: `_questions_for_batch` folds driver counters into the split extent
  and treats `-1` pads as missing (the throughput bench's negative warm-up
  `batchNum` crashed every run since PR #8, including on a8a5142).

## 9. Learning tests for memory/prediction and a target-encoding fix, 2026-09-09

- Alec asked whether recall/prediction QUALITY (not just mechanism) is
  tested. Added `test_temporal_answer_quality_improves_with_training[past|
  future]`: on the discourse+synthesis XOR fixture (one document), 30
  training steps of past / future questions reduce the temporal
  `answer_construction` (constructed surface vs embedded target sentence) by
  >10%. Supervised improvement is asserted by the Step 7 test and the joint
  band test; reconstruction by the pre-existing convergence gates.
- Silent bug found while writing them: `_embed_answer_texts` called the bare
  `InputSpace.forward` (which only lexes; the PartSpace embed happens in the
  stem), so its "target" was the live input carrier -- the presented input's
  own embedding -- not the desired sentence. It now runs the real stem
  (`_lex_embed_stem`) inside the synthesis guard with
  `_online_learning_frozen` set on every Space (no word promotion, live
  carriers restored by value, model-level stem stash restored): a read-only
  target encoding. The scoring test now asserts target != presented input,
  no promotion, and an unchanged carrier.
- Step 7 numbers: untrained right = wrong = 0.34446; trained right 0.00054,
  wrong 0.6751.

## 11. Thinking (What spec sections 7-9.5), 2026-09-09

Built under the [mathematical thinking plan](2026-09-09-mathematical-thinking.md)
(Phases 0-5 landed on `claude/math-thinking-spec-formalize-055ce3`):
`WhatInteractionMemory` + episode boundary, `WhatStepChooser` resolve step
with exact primitives (`bin/exact.py`, `<Queries>` ops in `math.grammar`),
per-row `think()` over one forward, `runBatch` episodes with root scoring
after parity and `whatThinkingPolicyWeight` credit, `serve.py` thinking
payload, `bin/eval_math_thinking.py`. Mechanism tests:
`test_what_episode_memory`, `test_what_thinking_episode`,
`test_math_thinking_training`, `test_serve_thinking`,
`test_eval_math_thinking`, `test_exact_primitives`, `test_math_dataset`.
Learning gates: NOT met (pilot report in `doc/benchmarks/`); the RUN_SLOW
floor is a strict xfail until a configuration passes.

## 10. The three formerly-open items, closed 2026-09-09

- Carrier purity: `SubSpace.carrier_pure` + `carrier_like` with fresh
  per-batch bases (sharing only parameter-bearing bases); `Space`
  `_reverse_target` / `_adopt_reverse_carrier` / `_reverse_stash` route every
  reverse write (event, activation, painted event, recovered-input stash,
  concepts, basis `setW`) to the carrier for pure carriers; merge glue and
  `_reverse_body` never read reconstruction carriers for pure carriers;
  `ConceptualSpace.synthesize` restores grammar cursors. Leak found on the
  way: the fresh carrier had SHARED the live event basis object.
- Teacher decoupling: `record_loss` (18 sites), `_primary_loss`,
  `_open_batch`, `_runtime_batch`, model-owned `data`/`errors`/`loss`/
  `legacy_prediction_enabled`; `stage_batch_sources` and `bind_input`
  guarded. Detached-Teacher training test.
- Chooser context: root cause was that `_stm_bounded_reduce_step` calls the
  grammar layers without `what_ctx`, and those are the only choices
  `record_choice` credits. Fixed at both ends (layer default + installation
  on grammar layers). `test_chooser_question_bias_trains_through_the_policy_objective`
  on MM_phrase_decode (5/15 rules; MM_xor has one rule per arity, XOR_grammar
  is not a pipeline width) shows `what_projection.grad` on the credited
  (binary) chooser and a distinct `what_report()["policy"]` entry.
