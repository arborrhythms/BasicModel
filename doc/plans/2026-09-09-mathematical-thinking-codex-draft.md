# Thinking: mathematical reasoning plan and tests

Status: proposed, 2026-09-09. Execution plan for the thinking requirements in the
[What specification](../specs/2026-07-27-teaching-modes-and-next-iteration.md#7-iterative-thinking),
sections 7–9.5.

## Objective and testbed

Make the model choose useful subquestions through recurrent `what()` calls,
retain their results, and construct a better root answer. Exact mathematics
lets us measure progressive illumination of the relevant conceptual state.

| Stage | Example | Capability |
|---|---|---|
| Dependency arithmetic | `a = 3; b = a + 4; c = 2*b; What is c?` → `14` | Resolve dependencies, retain `b = 7`, and use it to answer the parent. |
| Simultaneous constraints | `x + y = 12; y = 2*x; What is x?` → `4` | Combine constraints on the same variables until the answer is determined. |

Start with bounded integers and unique solutions. Supply exact lookup,
arithmetic, and substitution as named grammar operations; the model selects
operands and subquestions. An independent verifier checks emitted steps. The
complete solver is available only to dataset generation and scoring. Learning
arithmetic primitives is a later experiment.

## Implementation plan

1. **Enable learning.** Use a small CPU/eager configuration with interaction LTM
   independent of legacy prediction, nonzero reconstruction and answer weights,
   and verified gradients. Preserve separate reconstruction and answer paths.
2. **Connect grammatical control.** Learn open/complete/close decisions using
   the existing chooser and LTM parity stack. Internal subquestions reference
   concepts and accumulated bindings. Record operations, operands, results, and
   references; bound primitive executions per `what()` call.
3. **Accumulate and answer.** Feed subanswers into subsequent conceptual
   evaluation. Expose variable bindings/candidate masks for constraint tasks.
   Computed results set the answer symbol for `reverseOutput()`. Increasing
   closure pressure and a finite budget enforce LIFO closure; forced answers
   remain concrete, possibly low-confidence, and use the same synthesis path.
4. **Train episodes.** Score the root answer after parity. Keep generated states
   differentiable within the episode; detach durable memory at its update
   boundary. Use the existing estimator or policy loss for hard choices,
   report its credit separately, and take one optimizer step. Any intermediate
   supervision scores that subquestion's answer.
5. **Increase difficulty.** Train dependency depths 1–3; test unseen structures
   at depths 1–3 and 4–6. Then add simultaneous constraints. Generate fresh
   constants and variable names, shuffle premises, and add distractors. Split
   by dependency structure; randomize dataset positions independently of answers.

## Documentation updates

Treat documentation as part of each implementation milestone. First catch up
the main `doc/*.md` files with the recent What work: several still describe the
old mean readout, pre-What memory, or the separate thinking kernel. Then document
new thinking behavior as it lands, keeping proposed capabilities explicitly
separate from implemented interfaces and demonstrated learned behavior.

| Documents / sections | Required update alongside the work |
|---|---|
| [Architecture](../Architecture.md) — overview/training loop; [Componentization](../Componentization.md) — runtime ownership | Catch up `Understanding`, `what()`, `reverseReconstruct()`, and `reverseOutput()`; distinguish live synthesis from the gated legacy head. For steps 1–4, describe episode ownership and remaining Teacher dependencies. |
| [Spaces](../Spaces.md) — ConceptualSpace, OutputSpace, live paths | Replace stale mean-readout descriptions with the current concept readouts; document generated-carrier synthesis, branch isolation, and adapters. Distinguish this machinery from still-unproven collective concept-only inversion. |
| [STM](../STM.md) — LTM and inter-sentence prediction | Document paired What slots, chronological per-row recall, resets, and capacity. For steps 1–4, specify LIFO parity, closure pressure, active-episode gradients, durable-memory detachment, and memory availability independent of legacy prediction. |
| [Language](../Language.md) — grammar/chooser; [Reasoning](../Reasoning.md) — query reasoning and thinking kernel | For steps 2–3, define subquestion/control operations, bindings, exact mathematical primitives, and how verified results set the answer symbol. Explain the relationship to the older `ThinkingKernel`/`think_about()` path; distinguish trace-only reasoning from causal answer production. |
| [Training](../Training.md) — objectives/training loop; [Params](../Params.md) — architecture/training knobs | Catch up `Data.what()` targets, both primary losses, reconstruction-priority gradients, synthesis gates, and curricula. For steps 1–4, document episode updates, chooser credit, detach boundaries, and budgets; distinguish parser defaults from canonical XML values and ensure the pilot trains both branches. |
| [What design](../WhatSpacetimeDesign.md); [specification](../specs/2026-07-27-teaching-modes-and-next-iteration.md); [deliverables tracker](2026-09-08-what-spec-deliverables.md) | Reconcile stale “current boundary,” “needs doing,” and test-coverage claims with landed interfaces and remaining behavioral gaps. Update acceptance status with test evidence at every milestone. |
| [README documentation index](../../README.md#documentation); [benchmark reports](../benchmarks/) | Refresh Teacher-era labels and link this plan. For step 5, add reproducible training/evaluation commands and a thinking report with checkpoint/commit, config, hardware, dataset splits, seeds, budgets, ablations, and matched baselines. |

Each milestone must update the affected runtime docs, parameter reference, and
spec/tracker together. Link actual tests and measured results; interface tests
or scripted traces alone must not be reported as learned thinking. Historical
benchmark results retain their original workload and configuration labels.

## Test plan

| Test | Required evidence |
|---|---|
| Runtime | Learned, unscripted subquestions feed the parent. Nested questions close in LIFO order; budget exhaustion always restores parity with a scoreable answer. |
| Answer causality | Changing a necessary intermediate result changes the emitted answer or triggers correction. Trace-only changes fail. Forced answers also traverse `reverseOutput()`. |
| Learning and isolation | Root loss reaches earlier continuous states and credits the chooser. One update per episode; targets remain inaccessible; durable memory detaches at the declared boundary. |
| Useful recurrence | Compare one checkpoint at budgets of 1, 4, 8, and 16 calls with equal primitive allowance per call. More iterations improve held-out multi-step accuracy; removing intermediate memory harms it. |
| Progressive illumination | Decode variable candidates from conceptual state after each step. Useful steps narrow candidates while retaining the true solution. The independent verifier scores validity; oracle candidates never enter model context. |
| Generalization | Unseen structures and longer chains work. Variable renaming, premise order, distractors, and dataset-position reassignment preserve answers. |
| Regression | Preserve branch separation, checkpoint reload, per-row independence/resets, finite gradients, and existing What/LTM tests. |
| Documentation | Relative links resolve; documented pilot commands run; flags/defaults match code and XML; implementation and learned-behavior claims cite the appropriate tests or evaluation report. |

Proposed pilot gates: at least 95% exact-answer accuracy on unseen problems at
trained depths, at least 80% at depths 4–6, and a gain of at least 10 percentage
points over one-call evaluation on the multi-step subset, across three seeds.
Every accepted derivation step must verify; count rejected steps and forced
closures separately. Freeze datasets and criteria before evaluating.

Report accuracy by depth/budget, derivation validity, memory ablations,
candidate reduction, iterations, forced closures, both primary losses,
primitive counts, and latency. These gates establish bounded mathematical
thinking; broader conceptual-field and general-reasoning requirements remain.
