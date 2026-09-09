# Mathematical thinking: execution plan

Status: plan, 2026-09-09. Drives the
[mathematical thinking specification](../specs/2026-09-09-mathematical-thinking.md)
(What spec §7–9.5). Supersedes the Codex draft, archived verbatim as
[2026-09-09-mathematical-thinking-codex-draft](2026-09-09-mathematical-thinking-codex-draft.md).
Branch: `claude/math-thinking-spec-formalize-055ce3` in both repositories.

Every phase: basicmodel commit(s) on the branch → push to
`arborrhythms/BasicModel` → WikiOracle commit "Bump basicmodel to <sha>
(<summary>)" on the same-named branch → push. No PR unless asked. Default
configs stay byte-identical in every phase; each gated landing states its
justification (the live-wiring directive).

## Gap audit (against basicmodel 077c18d)

| # | Requirement | Code today | Gap |
|---|---|---|---|
| G1 | learned subquestion choice | `choose_what_slot` always returns a complete slot; `think()` re-runs `forward()` per iteration | real |
| G2 | subanswers feed the parent | `_resolve_answer` never reads LTM; LTM reaches the chooser as 8 summary scalars | real |
| G3 | interaction LTM independent of prediction | `_what_memory()` = `symbolSpace.discourse`, built only under `<sentencePrediction>true` | real |
| G4 | root scored after parity, one step | `runBatch` calls `what()` once and never `think()` | real |
| G5 | episode-scoped credit | `append_what_slot` detaches at append | real |
| G6 | exact primitives as named ops | none; the Thinking Kernel is a truth-interval tool, unconnected to `think()` | real |
| G7 | closure pressure, forced LIFO closure | implemented (`think()`, `append_what_slot`); schedule fixed, parity checked on row 0 only | partial |
| G9 | runtime docs stale | STM.md has no What-slot section; Params.md lacks the What knobs; README labels the spec "Teacher"; Reasoning.md documents only the kernel; Spaces.md verified current | real |

## Phase 0 — documents and the archive (no code) — landed with this plan

`doc/specs/2026-09-09-mathematical-thinking.md` (new), this file (replaces
the draft), the archived draft, `README.md` rows (spec renamed "What
specification"; the two new files listed), `doc/Reasoning.md` "Two thinking
loops", `doc/STM.md` §13 "Interaction LTM and the What stack",
`doc/Params.md` (the landed What knobs; the planned `whatThinking*` knobs in
a clearly separate "planned" table), `doc/Training.md` "What questions and
the two primary costs", `test/test_doc_links.py` (every relative link under
`doc/**/*.md` and `README.md` resolves). Spaces.md was checked against What
spec §5.5 and needed no change.

## Phase 1 — exact primitives, dataset, verifier (no model change)

Files: `bin/exact.py` (new: `ExactLexer`, `ExactState`, primitives
`lookup/evaluate/bind/substitute/constrain`, `numeral_code`,
`MathProblemGenerator`, `ExactVerifier`, `illumination`), `bin/data.py`
(`Data.load` dispatch `"math"`, `loadMath()` reading `<data><mathRange>`,
`<mathDepths>`, `<mathDistractors>`, `<mathStage>`, `<mathSeed>`),
`data/math.grammar` (default.grammar plus the `<Queries>` exact ops),
`data/model.xsd` (the `<data>` math elements).
Tests (`test/test_exact_primitives.py`, `test/test_math_dataset.py`): the
lexer round-trips generator output; each primitive is total and exact on
random problems; the verifier accepts solver-order traces and rejects an
edited result; generator invariants (unique solution, range, distractors
off-chain, split-by-structure, position independence);
`Data.what(supervised)` returns `onehot(answer, R)` with
`has_supervised_outputs`.
Acceptance: new tests green; `Data.load("xor")` fixtures unchanged.

## Phase 2 — interaction memory and episode boundary

Files: `bin/Layers.py` (`WhatInteractionMemory` extracted from
`InterSentenceLayer`; `begin_what_episode` / `end_what_episode`; live
overlay under `episode` detach; `what_context` gains `open_question`,
`latest_output`), `bin/Language.py` (`SymbolSubSpace.what_memory` built
under `<whatThinkingMemory>`), `bin/Models.py` (`_what_memory()` fallback;
knob reads for `whatThinkingMemory`, `whatThinkingDetach`),
`data/model.xsd`, `doc/Params.md`, `doc/STM.md`.
Tests: `test/test_what_spacetime.py` parity tests run against both memory
owners (parametrized fixture); `test/test_what_episode_memory.py` (the
episode overlay keeps `requires_grad` until `end`; `slot` detaches at
append; standalone memory without `sentencePrediction`; per-row episodes).
Acceptance: `sentencePrediction` fixtures byte-identical; new tests green.

## Phase 3 — resolve step, `WhatStepChooser`, primitives in the loop

Files: `bin/Output.py` (`AnswerDerivation.step`, `StepChoice`),
`bin/Language.py` (`WhatStepChooser(nn.Module)`: 29-dim what-context +
candidate features → logits over {ANSWER, OPEN(v), EXECUTE(op, operands)};
zero-init output layer; `sample()` / `argmax()`; `log_prob`),
`bin/Models.py` (`_resolve_answer` takes `ltm_context`, `pressure`, `σ`;
`choose_what_slot` reads `derivation.step`; `think()` reuses the execution
at iteration ≥ 1 and handles per-row parity; `_best_effort_what` realizes
through `reverseOutput()`; `understand()` runs `ExactLexer` when primitives
are on; knob reads for `whatThinkingIterations`, `whatThinkingPressure`,
`whatThinkingPrimitives`; `_materialize_answer_path` builds the chooser
eagerly so checkpoints round-trip), `data/model.xsd`, `doc/Params.md`,
`doc/Language.md` (chooser section), `doc/Reasoning.md`.
Fixture: `data/MM_math.xml` (from `MM_xor.xml`: `<answerSynthesis>true`,
`<transformChooser>mlp`, `<whatThinkingMemory>true`,
`<whatThinkingIterations>8`, `<whatThinkingPrimitives>4`, dataset `math`,
grammar `math.grammar`, `OutputSpace.nOutput = R`).
Tests (`test/test_what_thinking_episode.py`): spec invariants 1, 2, 3, 5,
6, 8, 9; `think()` on `MM_math` opens / executes / closes with a scripted
chooser (monkeypatched logits) and with the untrained neutral chooser
(first option = ANSWER, today's behaviour); checkpoint save → fresh model →
`load_weights(require_match=True)` → identical answer.
Acceptance: `MM_xor` and the canonical `BasicModel.xml` byte-identical
(pinned by `test_what_training` / `test_output_synthesis`); new tests green.

## Phase 4 — training through the episode

Files: `bin/Models.py` (`runBatch` drives `think()` when iterations > 1;
root scoring after parity; `_what_thinking_policy_loss` with baseline,
weighted by `<whatThinkingPolicyWeight>`; `what_report()` gains
`policy.thinking`, `thinking.detach`, `thinking.primitives`;
`end_what_episode` after the optimizer step; optional self-cloning from
verifier-accepted traces under `<whatThinkingCloneWeight>`),
`bin/Optimizer.py` only if the chooser needs its own `add_param_group`
(reuse the `_collect_fresh_synthesis_modules` queue first),
`doc/Training.md`, `doc/Params.md`.
Tests (`test/test_math_thinking_training.py`): invariant 7 (the root
gradient reaches iteration-0 states under `episode`, not under `slot`);
exactly one optimizer step per batch (spy on `optimizer.step`); target
unreachable (the `test_what_training` isolation pattern); `what_report()`
fields; 30-step smoke (policy credit finite, answer cost non-increasing
trend on a fixed tiny set); `RUN_SLOW`-gated learning test (depth 1–2
problems reach > 50 % exact accuracy on CPU — a floor, not the pilot gate).
Acceptance: byte-identical when iterations = 1; smoke green in the default
suite; the slow test green under `RUN_SLOW=1`.

## Phase 5 — serve, reporting, evaluation script, pilot

Files: `bin/serve.py` (`thought_free` ⇒ one iteration; attach
`response["thinking"]` = iterations / forced / trace summary when an
episode ran), `bin/eval_math_thinking.py` (budgets 1 / 4 / 8 / 16, three
seeds, memory ablation, illumination probe, verifier counts → markdown
report), `doc/benchmarks/<date>-math-thinking-pilot.md` (checkpoint,
commit, config, hardware, splits, seeds, budgets, ablations, matched
baselines), `README.md` (benchmark link), the What spec status record and
the deliverables tracker row "Thinking". WikiOracle: `doc/Implementation.md`
thought-free row ("… restricted grammar path and no internal dialogue"),
submodule bump.
Tests: `test/test_serve_thinking.py` (Flask test client; a `thought_free`
payload yields no open slots); `test/test_eval_math_thinking.py` (the script
runs on a 4-problem set and emits every report column).
Acceptance: the spec §10 gates evaluated and reported; the B24 band
re-measured with thinking off (expected unchanged: the gate-off path is
untouched).

## Test evidence rule

Interface tests and scripted traces are reported as *mechanism*; only the
Phase 5 report may claim learned thinking, with the checkpoint and commit.

## Sequencing summary

| Phase | basicmodel commit | WikiOracle bump |
|---|---|---|
| 0 | "Specify mathematical thinking; catch up What docs" | "Bump basicmodel to <sha> (mathematical thinking spec)" |
| 1 | "Add exact primitives, math dataset and verifier" | bump |
| 2 | "Extract WhatInteractionMemory; episode credit boundary" | bump |
| 3 | "Resolve step with WhatStepChooser and exact primitives" | bump |
| 4 | "Train through thinking episodes (root scoring, policy credit)" | bump |
| 5 | "Serve/eval for thinking; pilot report" | bump + Implementation.md row |
