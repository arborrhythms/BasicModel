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

> Superseded by Phase 8: the EXECUTE candidates, `ExactLexer` in
> `understand()`, `math.grammar` and `<whatThinkingPrimitives>` were
> removed from the runtime. The text below is the record of what was
> built and measured before the decision.

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
grammar `math.grammar`, `OutputSpace.nOutput = R` (the one-hot width)).
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
`end_what_episode` after the optimizer step; self-cloning from
verifier-accepted traces was NOT implemented -- spec section 12 Q2),
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
Status 2026-09-09: landed; the slow floor is a strict xfail (12.5 % exact
accuracy after 150 steps on the fixture; see the Phase 5 pilot report).
Found on the way: list-label datasets prep `[N]` rows to `[B, 1, N]` while
the one-hot head emits `[B, N, 1]`, which the output shape gate rejected
(the answer loss was silently zero on the math dataset);
`_align_output_pred` now reconciles that transpose.

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
Status 2026-09-09: serve payload, evaluation script and pilot report
landed; the §10 learning gates are NOT met (report); the illumination
*probe* (a learned readout over conceptual states) is not implemented --
the report's candidate reduction is the oracle-side measure over the
accepted derivation steps; the B24 band was not re-measured in this
session (the canonical `BasicModel.xml` keeps `whatThinkingIterations`
at its default 1, so the gate-off path is byte-identical by
construction, pinned by the existing suites).

## Phase 6 — direct arithmetic first (Alec, 2026-09-09)

Diagnosis: stage-1 problems were not answered because the stack had not
learned direct arithmetic; there were not enough simple subproblems.
Stage 0 (`mathStage=0`): the input is `a + b`, the answer `c` (one-hot),
the input reconstructed as usual, many stochastic examples split by
unseen pairs, no thinking (`whatThinkingIterations` 1). Files:
`bin/exact.py` (`_stage0`, `split_by_surface`, `<mathOperators>`),
`bin/data.py`, `data/MM_add.xml`, tests in `test/test_math_dataset.py`.
Acceptance: measured train / held-out-pair accuracy and both primary
costs over epochs, reported in the pilot report; the answer-synthesis
path compared against the direct head. Only after stage 0 is learned does
the curriculum proceed to variables (stage 1) and substitution (stage 2).
Result 2026-09-09 (pilot report, "Stage 0"): the LEARNED direct answer
(gradient only, no primitives) stays at chance at 14 and 64 wide, for
two-digit and single-digit sums, and collapses to the majority answer
when the answer distribution is skewed; a linear probe of the untrained
states recovers the operands only partly and the sum not at all. Through
the EXACT route (bare expression lexed as `_ = a + b`, learned chooser
under policy credit choosing evaluate / bind / answer, one-hot numeral
code) the model reaches 100 % on training and on unseen pairs by epoch 10
(4096 problems, R = 32, 64 wide) and by epoch 25 on the small test
configuration (`test_stage_zero_direct_arithmetic_learns_through_the_exact_route`).
So the stack does direct arithmetic by SELECTING the exact operation, not
by computing it in the folds; the next rung is stage 1 with the same
route (the depth-1 chain: open the operand's premise, evaluate, bind).

## Phase 7 — stage 1 on the exact route (2026-09-09, same day)

Two changes made the stage-1 chain learnable: dense per-row credit for
each variable the scratchpad binds (`WHAT_BIND_REWARD`, spec 8.3) and
content features on the candidates (READY / DEPENDENCY) so the policy
generalizes across premise counts. Also fixed: forced-closure answers
were the whole batch tensor for a row (`_best_effort_what` now slices the
row). Results (pilot report, "Stage 1"): depth 1 trained → unseen
depth-1 structures 100 %, untrained depth 2 83 %; depths 1–2 trained with
features → unseen depth 1 100 %, depth 2 96 %, untrained depth 3 100 % at
epoch 5. The RUN_SLOW xfail is replaced by
`test_stage_one_dependency_chains_learn_through_the_exact_route`
(default suite, ~50 s). Remaining: depths 4–6, stage 2 (substitute /
constrain), the policy's sampling variance on train, and episode length.

## Phase 8 — the syntactic route (Alec, 2026-09-09)

Decision: no mathematical machinery in the runtime; math is a syntax that
tests the UG; `plus` is a transitive verb with the existing verb
definition; intermediate thoughts are LTM slots. Landed: the exact
scratchpad, the EXECUTE candidates, the binding credit, the READY /
DEPENDENCY features and the numeral / referent codes are removed from
the runtime (`bin/exact.py` is data generation and evaluation only;
`<whatThinkingPrimitives>` retired, raises at load; `math.grammar`
deleted). The resolve step chooses ANSWER or OPEN(word) over the
presented words; the root answer attends over the row's LTM outputs
(`ltm_attention`, zero-initialised); corpora render as words (`3 plus
4`, `b equals a plus 4 ; what is b`). The stage-0 / stage-1 learning
tests are strict xfails on the syntactic route. The verb grammar
configuration is `data/MM_add_verb.xml` (230M parameters); its first
30-epoch run stayed at the majority baseline (pilot report, "Stage 0 as
syntax"). Alec's correction: addition is iterated
succession (`next(one) = two`); `mathOperators=succ` is the rung below.
`test_verb_successor.py` shows the VP is the successor by construction
and exposed a dead zone (zero-init readout under a zero-derivative
threshold: no verb could learn by gradient), now fixed straight-through.
Result (pilot report, "The successor is learned as a VP"): after the
answer path was seeded from the root idea (`Understanding.answer_seed`;
the `symbols` tensor was constant across sentences on the serial path)
and the reduce gate lowered so short sentences compose, the successor is
learned exactly over the single-digit nouns; every failure is a
two-digit numeral read as one of its digit parts. Next: multi-digit
numerals as wholes over digit parts with `.where` as position (the
compound representation), then counting ("n plus m") through the
thinking loop with unseen pairs held out, with serial carries through
LTM.

Review follow-up (Codex, 2026-09-09): the ANSWER step for a pending
subquestion used the root idea, so the subanswer did not depend on which
subquestion was active (confirmed: identical COMPLETE-slot outputs for
two different words on a fresh model). Fixed: the subquestion's answer
starts from QUERY(w) (live, so the answer loss reaches it) and then
attends over the row's LTM outputs; the root's answer still starts from
the root idea. Test: `test_subanswer_is_conditioned_on_the_active_subquestion`
(controlled for the forward path's priming drift by comparing
same-position episodes on fresh models). The spec's section 5.1 / 6.3 /
8.4 text now describes the retired primitives as history.

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

## Phase 9 — wholes smaller than words (Alec, 2026-09-10)

Finding (pilot report, "The successor is learned as a VP"): every
failure of the successor run was a two-digit numeral read as one of its
digit parts. On the trained checkpoint `12` and `14` are attested store
entries, so the within-whole division leaves them as single wholes, and
the radix word vector max-fuses sub-tokens without order.

Landed: the **digit whole** — `<digitWholes>true</digitWholes>` on
WholeSpace makes every digit byte a whole by itself at the analysis
cut (`_type_run_spans(..., singleton=...)`; `12 plus 1` -> `1`, `2`,
`plus`, `1`, each digit with its own `.where`). Default off,
byte-identical. Tests: `test_type_run_spans.py` (digit whole section).
Two cutters exist: the sentence-wide `stage_analysis_spans` (stage-0
evidence and meronomy bookkeeping) and the per-word
`stage_word_property_weights`, whose view the serial word loop's concept
evidence reads (`_ar_whole_reference_presence`). The per-word cutter runs
only under the aligned per-word protocol (`propertyBasis`,
`conceptBinding` aligned, `serialObjectMeta`: BasicModel's settings),
and that protocol requires the STM depth to equal the concept location
count, which the three-stage verb topology violates (its stages halve
locations to 4, 2, 2). On the verb fixture the digit whole was therefore
byte-identical (a run reproduced the earlier losses exactly). The
experiment moved to the canonical topology: BasicModel's configuration
with the successor corpus, a one-hot head and a 65k concept inventory
(43M parameters), digit wholes on versus off.

Two regime facts found on the way to that experiment, both fixed or
recorded: (1) BasicModel's `serialWordCapacity` / `serialWordBuckets` of
256 make every batch iterate 256 word steps whatever the sentence
length (29 minutes per epoch on the three-word corpus; 40 seconds at
capacity 8), and its `whatCurriculum` replaces the supervised questions,
zeroing the answer loss on a supervised corpus; the experiment config
sets capacity 8 and no curriculum. (2) Under `reconstructionPriority`
the protected set included the answer path's own operators (the
Spaces' synthesis layers, the percept adapter, the question
conditioner); reconstruction never reaches them, so their answer
gradient was capped at zero and the answer cost could not move at all.
They are now exempt (`_reconstruction_priority_parameters` skips
`synthesis_parameters()`); test
`test_answer_path_operators_are_not_protected_and_keep_learning`.

Open defect found while scoring: on the aligned protocol a checkpoint
does not round-trip the eagerly resolved word concept identities. After
`save_weights` / `load_weights` the words of `12 plus 1` resolve to no
concept row (`_ar_word_concept_rows` all -1; the radix part ids also
differ from the fresh model's), so a loaded model answers a constant
while the live model in the same process answers per fact. Per-fact
tables are therefore taken on the live model. The aligned checkpoint
tests cover capacity growth and allocator resync, not this identity
round trip on a math corpus; a minimal reproduction is the next step.

Next (Alec): treat the field width as a top-k over a wider retrieval —
retrieve 16 parts / wholes per step, attend 8 (or fewer), and feed only
the attended ones back to PerceptualSpace to determine future context;
both numbers as model-file knobs. With the field laid out as lower /
basic / higher bands, "no smaller (larger) whole at this `.where`" is an
empty band and is the signal to provide a smaller (larger) context. The
serial word stream is unchanged by the digit whole: `12` is still one
word step whose WholeSpace view now carries two digit wholes.
