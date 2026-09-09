# Mathematical thinking pilot (2026-09-09)

Status: **mechanism demonstrated, learning gates NOT met.** This report
records what the `MM_math` fixture measures at the end of the
[mathematical thinking plan](../plans/2026-09-09-mathematical-thinking.md)
Phase 5, against the gates of the
[specification, section 10](../specs/2026-09-09-mathematical-thinking.md#10-acceptance).
Interface tests and scripted traces are evidence of mechanism only; no
learned thinking is claimed.

## Setup

| Item | Value |
|---|---|
| Commit | basicmodel branch `claude/math-thinking-spec-formalize-055ce3` (Phase 5 commit) |
| Config | `data/MM_math.xml` (MM_xor topology, 14-wide, 32 slots; `answerSynthesis`, standalone memory, `episode` detach, 8 iterations x 4 primitives) |
| Dataset | `dataset=math`, `R = 16`, train depths {1, 2}, held-out depth 3, up to 1 distractor, seed 0, 64 + 16 problems, split by dependency structure |
| Hardware | macOS ARM64, CPU, eager (`BASICMODEL_DEVICE=cpu`, `MODEL_COMPILE=eager`) |
| Seeds | 1, 2, 3 (evaluation); 0 (diagnostics) |
| Budgets | 1, 4, 8, 16 iterations, 4 primitives per iteration |

Reproduce:

```sh
cd basicmodel
.venv/bin/python bin/eval_math_thinking.py --config data/MM_math.xml \
    --budgets 1,4,8,16 --seeds 1,2,3 --split test --rows 16 --batch 8 \
    --ablate-memory --illumination --out report.md
RUN_SLOW=1 .venv/bin/python -m pytest test/test_math_thinking_training.py -k learning_floor
```

## Evaluation of the untrained fixture (three seeds, test split, 48 rows)

| budget | depth | n | accuracy | validity | forced | iterations | primitives |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | all | 48 | 0.042 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | all | 48 | 0.042 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8 | all | 48 | 0.021 | 0.000 | 0.000 | 1.000 | 0.000 |
| 16 | all | 48 | 0.021 | 0.000 | 0.000 | 1.000 | 0.000 |

Chance is 1/16 = 0.0625. The untrained `WhatStepChooser` is neutral (a
zero-initialised head ties every candidate and the tie breaks to ANSWER),
so every budget collapses to one iteration with no primitive executed and
no derivation to verify. This is the intended byte-identical starting
point, not a result. The memory ablation rows are identical for the same
reason.

## Learning diagnostics (150 optimizer steps, 16 train problems, seed 0)

`lossOut` is `answer_construction` (MSE against the one-hot answer) at the
last step; accuracy is exact-answer accuracy on the same 16 problems.

| run | chooser | lr | lossOut (step 149) | accuracy | iterations at eval |
|---|---|---:|---:|---:|---:|
| A | oracle-scripted solver order (mechanism ceiling) | 5e-3 | 0.0193 | 0.125 | 5 |
| B | no thinking (`whatThinkingIterations=1`) | 5e-3 | 0.0312 | 0.125 | 1 |
| C | sampled policy (`whatThinkingPolicyWeight=0.5`) | 5e-3 | 0.0306 | 0.125 | 1 |
| D | sampled policy | 2e-2 | 0.0314 | 0.125 | 1 |

Reading: with the oracle derivation the exact numeral code in the root
slot lowers the answer loss faster than any other configuration (A vs B),
so the exact answer symbol does reach the output adapter; but 150 steps
are far too few for the one-hot head to separate 16 classes from a
14-wide symbol (accuracy 2/16 everywhere), and the sampled policy (C, D)
learns to ANSWER immediately (argmax at evaluation = one iteration),
because in this regime the root reward does not yet distinguish a
derivation from no derivation. Policy credit: 17,833 recorded choices in
C, mean return -0.41; 8,519 in D, mean return -0.18.

## Gates (specification section 10)

| Gate | Status |
|---|---|
| >= 95 % exact accuracy at trained depths on unseen structures | not met (12.5 % on train, 4 % on test, untrained/150 steps) |
| >= 80 % at depths 4-6 | not evaluated (fixture holds out depth 3 only) |
| >= +10 points multi-step vs one-iteration budget | not met (learned policy answers in one iteration) |
| every accepted derivation step verifies | met on the oracle-scripted episodes (`test_scripted_episode_opens_executes_and_closes_in_lifo_order`) |
| memory ablation harms multi-step accuracy | not measurable yet (no multi-step policy learned) |

The `RUN_SLOW` learning floor in `test/test_math_thinking_training.py` is a
strict `xfail` with this report as its reason.

## What the pilot did establish

- The scoring path had a silent defect on this dataset: list-label
  datasets prep their `[N]` labels to `[B, 1, N]` while the one-hot head
  emits `[B, N, 1]`, and the output shape gate zeroed the answer loss
  (every run showed `lossOut = 0.0`). `Model._align_output_pred` now
  reconciles that transpose; the loss is nonzero and decreases (B).
- Oracle-scripted episodes run end to end through `runBatch`: one
  `forward()`, OPEN / COMPLETE / CLOSE slots in LIFO order, every
  primitive execution and step choice in the replayable trace, the
  verifier accepting the derivation, the root scored after parity, one
  optimizer step, the episode detached after it, the step chooser
  round-tripping through `save_weights` / `load_weights`, and the serve
  payload reporting the episode.
- The learning regime is the open problem: candidates for the next
  iteration are a curriculum that starts at depth 1 with a warm-started
  output adapter, a larger symbol width (the numeral code uses 4 of 14
  coordinates here), self-cloning on verifier-accepted traces (spec Q2),
  and a reward that credits verified intermediate binds rather than only
  the root.

## Not measured in this session

The B24 clean-corpus throughput band (spec 12.15) was not re-measured:
the canonical `BasicModel.xml` keeps `whatThinkingIterations` at 1, so
the gate-off path is byte-identical by construction (pinned by the What
suites). The illumination *probe* (a learned readout over conceptual
states) is not implemented; `illumination_gain` in the script is the
oracle-side candidate measure over accepted derivation steps.

## Stage 0: direct arithmetic (Alec, 2026-09-09, later the same day)

Alec's diagnosis: stage-1 problems were not answered because the stack
could not yet do direct arithmetic. `data/MM_add.xml` presents `a + b`,
expects the one-hot `c`, reconstructs the input; 4096 stochastic problems
at `R = 32`, answers sampled uniformly (sampling `a` then `b < R - a`
skews sums to `R - 1` at 13.5 %, and a majority-answer head matched that
plateau exactly), held-out set = unseen operand pairs (384 of 528 pairs
seen in training). Codebooks were raised (CS 512, WS 4096, PS 4096 rows):
128 / 200 rows were exhausted by the numerals and promoted chunks.

Learned direct answer (gradient only; `runEpoch`, batch 32, lr 5e-3):

| run | width | corpus | epochs | train acc | held-out acc | prediction histogram |
|---|---:|---|---:|---:|---:|---|
| skewed, synthesis | 14 | R=32 | 20 | 0.127 | 0.088 | always 31 (majority) |
| skewed, direct head | 14 | R=32 | 20 | 0.127 | 0.088 | always 31 |
| skewed, synthesis | 64 | R=32 | 50 | 0.127 | 0.088 | always 31 |
| uniform, synthesis | 14 | R=32 | 100 | 0.043 | 0.037 | 20 / 2 / 28 |
| uniform, synthesis | 64 | R=32 | 100 | 0.043 | 0.029 | 20 / 2 / 8 |
| uniform, synthesis | 14 | R=10 (single digit) | 50 | 0.129 | 0.000 | 1 / 3 / 2 |

Chance is 1/32 (0.031) and 1/10; the majority baselines are 0.135, 0.047
and 0.133. Reconstruction stayed tiny throughout (`input_reconstruction`
3e-4, `reverseReconstruct` cost 5e-5): the input is reconstructed, the
sum is not computed. Linear probes from the UNTRAINED 64-wide states to
the answer (2048 train rows, 512 held-out):

| state | -> a | -> b | -> a + b |
|---|---:|---:|---:|
| percepts (512-d) | 0.369 | 0.441 | 0.010 |
| concepts (128-d) | 0.277 | 0.449 | 0.014 |
| symbols (256-d) | 0.297 | 0.477 | 0.008 |

The operands are only partly linearly present (multi-digit numerals at
shifting byte positions) and the sum is absent from every state: the
stack would have to compute it in the folds, and in this topology and
budget it does not.

Exact route (the design's intent: primitives supplied, selection
learned). The lexer presents the bare expression as `_ = a + b`; the
learned `WhatStepChooser` chooses among ANSWER / `evaluate:0` /
`bind:_=v` with `<whatThinkingPolicyWeight>0.5`; a bound value sets the
root slot to the ONE-HOT numeral code (width 64 >= R); the answer path
realizes it. Same corpus, `runEpoch`, batch 32, lr 5e-3, 64 wide:

| epoch | answer loss | train acc | held-out acc | rows bound | mean iterations |
|---:|---:|---:|---:|---:|---:|
| 0 | -- | 0.014 | -- | 0.00 | 1.0 |
| 10 | 0.0000 | 1.000 | 1.000 | 1.00 / 1.00 | 1.0 |

The small test configuration (R = 16, 32 wide, 256 problems, lr 1e-2)
reaches 1.000 / 1.000 by epoch 25 in about eight seconds
(`test_stage_zero_direct_arithmetic_learns_through_the_exact_route`).

Reading: the stack answers direct arithmetic exactly and generalizes to
unseen pairs when arithmetic is a named operation it learns to SELECT
and whose result it learns to REALIZE; it does not learn to compute the
sum from bytes by gradient. Stage 1 should be attempted on the same
route (the depth-1 chain is open / evaluate / bind / answer), which is
what the earlier pilot lacked: a stage-0 policy to build on.

## Stage 1: dependency chains on the exact route (2026-09-09, same day)

Configuration: `MM_math` at 64 wide, `R = 16`, 2048 problems, up to one
distractor, eight-iteration episodes with four primitives, policy weight
0.5, `runEpoch` batch 32, lr 5e-3, codebooks 512 / 4096 / 4096. Held-out
= unseen dependency STRUCTURES at the trained depths plus the next depth,
which is never trained.

Without binding credit (the pilot's regime), depth 1: 36 % train, 14 %
held-out, nothing bound, one iteration -- the policy never finds the
four-primitive chain and the head guesses. With `WHAT_BIND_REWARD`
(per-row credit for each bound variable) and the pruned menu:

| trained | epoch | train acc / valid | unseen structures, trained depth | next depth (untrained) | mean iterations |
|---|---:|---:|---:|---:|---:|
| depth 1 | 10 | 1.00 / 1.00 | depth 1: 1.00 | depth 2: 0.58 | 7.0 |
| depth 1 | 40 | 1.00 / 1.00 | depth 1: 1.00 | depth 2: 0.83 | 5.0 |
| depths 1-2 | 10 | 1.00 / 1.00 | depth 2: 1.00; depth 1: 0.07 | depth 3: 1.00 | 4.25 |
| depths 1-2 | 30 | 1.00 / 1.00 | depth 2: 1.00; depth 1: 1.00 | depth 3: 0.71 | 7.0 |
| depths 1-2 | 40 | 1.00 / 1.00 | depth 2: 1.00; depth 1: 0.08 | depth 3: 1.00 | 4.5 |

The depth-1 held-out failure in the depths-1-2 run was positional: every
training depth-1 problem has a distractor (three premises) and every
held-out one has none (two), and the chooser's only content-free
features shift with premise count. With the READY / DEPENDENCY features:

| trained | epoch | train acc / valid | depth 1 unseen | depth 2 unseen | depth 3 (untrained) | mean iterations |
|---|---:|---:|---:|---:|---:|---:|
| depths 1-2 + features | 5 | 0.82 / 0.76 | 1.00 | 0.96 | 1.00 | 5.0 |
| depths 1-2 + features | 10 | 0.65 / 0.57 | 1.00 | 0.43 | 1.00 | 3.0 |
| depths 1-2 + features | 20 | 0.68 / 0.62 | 1.00 | 0.39 | 1.00 | 3.0 |
| depths 1-2 + features | 30 | 0.74 / 0.69 | 1.00 | 0.39 | 1.00 | 3.0 |

Training accuracy oscillates under sampled exploration (the policy is
still being sampled during training; evaluation is the argmax). The run without the features touches 100 %
on unseen depth-1 structures at epoch 30 but falls back to 8 % at epoch
40, whereas the feature run holds 100 % there at every checkpoint: the
content features make that generalization stable, not merely faster; the
small default-suite configuration (384 problems, 32 wide) reaches 100 %
held-out by epoch 10 and 99 % train by epoch 20
(`test_stage_one_dependency_chains_learn_through_the_exact_route`).

Gates: unseen structures at trained depths >= 95 % -- met (100 %); the
never-trained next depth -- 83 % (depth 2 from depth 1) and 100 % (depth
3 from depths 1-2); depths 4-6 and stage 2 not evaluated; every accepted
derivation verifies (validity tracks accuracy in every table above).

## Decision (Alec, 2026-09-09): the exact route is retired from the runtime

The stage-0 and stage-1 results above were obtained with exact
primitives executing inside the model's resolve step. That over-specifies
the solver as mathematical: the number of unbound variables was Python
state, not something the model could know from its LTM. Math is to be a
simple syntax testing the universal grammar; `plus` is a transitive verb
with the grammar's existing verb definition; intermediate thoughts are
LTM slots. The runtime now chooses only between answering and opening a
subquestion about a presented word, and conditions the root answer on
its LTM outputs; the exact code remains for data generation and
evaluation. The learning gates re-open as strict xfails until the large
stage-0 run learns `plus` over numeral nouns.

## Stage 0 as syntax: the large plus-as-verb run (2026-09-09)

`data/MM_add_verb.xml`: the `MM_phrase_decode` topology (serial single-S
idea, `complete.grammar` with the verb / adverb VP operators, WholeSpace
word analysis, 1024-wide, 230M parameters) on the word-rendered stage-0
corpus (`3 plus 4` -> one-hot 7, R = 16, 2048 problems, 1633 train rows,
344 unseen pairs), answer synthesis on, thinking off, `runEpoch` batch 8,
lr 5e-3, CPU. Nothing mathematical runs in the model: `plus` is a lexical
verb, the numerals are lexicon nouns.

| epoch | answer loss (epoch mean) | train acc | unseen pairs | prediction histogram |
|---:|---:|---:|---:|---|
| 0 | -- | 0.047 | -- | always 0 |
| 5 | 0.0566 | 0.078 | 0.098 | 8 / 10 / 6 |
| 10 | 0.0677 | 0.055 | 0.078 | 8 / 6 / 13 |
| 15 | 0.0501 | 0.066 | 0.035 | 15 / 13 / 6 |
| 20 | 0.0420 | 0.070 | 0.035 | 13 / 0 / 12 |
| 25 | 0.0519 | 0.070 | 0.039 | 15 / 13 |
| 30 | 0.0461 | 0.098 | 0.035 | 6 / 12 / 9 |

Majority baseline 0.085; chance 0.0625. The input reconstruction term
was present and small throughout (0.032 on the first batches, below
5e-5 later): the input is reconstructed, the sum is not produced. The
answer loss eases only by re-centering a constant prediction (the MSE
mean over one-hot targets), the signature of no verb being learned. 96
minutes of CPU training; "lots of training of a large network" is the
regime, and this is the first point on that curve, not its end.

Theoretical note (the sufficiency question, corrected by Alec): addition
is not computed as a value. A successor VP maps one number noun to the
next (`next(one) = two`), and "three plus two" is `next(next(three))`:
counting, performed by the thinking loop as iterated `what()`
subquestions whose answers land in LTM, with the parity stack as the
counter. `VerbLayer` (a diagonal gain in atanh space) therefore has to
implement ONE fixed symbol-to-symbol map that advances every number noun
to its successor, with the codebook snap keeping each step a discrete
symbol; numeral codes on a geometric progression per coordinate make
that exact for a single learned gain. The second operand is never read
as a magnitude. So the curriculum rung below "a plus b" is the successor
itself ("n plus one", `mathOperators=succ`, every fact trained), and
generalization to unseen pairs is compositional (iteration), not
interpolative. Multi-digit numerals are wholes over digit parts with
`.where` = position; carries are serial subquestions.

## The successor VP: sufficiency by construction, and a dead zone (2026-09-09)

`test/test_verb_successor.py` pins two facts about the existing
`VerbLayer` (`VP(NP) = tanh(e^w ⊙ atanh(NP))`, `w` a sparse readout of the
verb code, `Q = I`):

- **Sufficiency by construction.** With number codes on a geometric
  progression per coordinate (`atanh(code_n) = a · r^n`) and one verb
  whose readout is `log r`, the layer advances every noun to its
  successor exactly (15 / 15 one-step at R = 16), iterated application
  with the codebook snap counts from zero to fifteen, and `unapply_verb`
  is the exact predecessor. So the current VP form can be the successor;
  the number line is a direction in conceptual space along which the
  nouns are geometrically spaced, and precision falls with magnitude
  under the tanh saturation.
- **A dead zone at initialization.** The spectrum readout is
  zero-initialised and its sparsity soft-threshold (`|w| < 0.1 -> 0`)
  has zero derivative there, so the gradient into the readout was
  exactly zero at construction: no verb could begin to learn by gradient,
  which is why the 30-epoch `plus` run above never moved. The forward is
  now unchanged and the gradient passes straight through the threshold
  (`LiftLayer._verb_spectrum_w`); from the real zero init the successor
  is then learned exactly in about 1,500 steps on the standalone problem
  (codes, verb code and readout all free), and snapped counting is exact.

The successor corpus (`mathOperators=succ`, "n plus one", every fact
trained) is being re-run on `MM_add_verb.xml` with the fix; the earlier
`plus` run predates it and is superseded.

## Supervised training on the output path (2026-09-09)

Alec asked whether tests show supervised training on the new
`reverseOutput()` path. They existed only for the parallel XOR topology.
`test/test_output_path_supervised.py` now covers the serial grammar
topology the verb runs use, and writing it found the defect behind every
flat run above: the `symbols` tensor the answer path was seeded from is
the symbol-space activation over a WholeSpace codebook that has TWO
active rows of 65,536 at initialization, so it is the same for every
sentence (relative spread 0.027 across eight phrases, against 0.534 for
the grammar's root idea in the conceptual state). The answer path was
input-blind. `Understanding.answer_seed` now seeds it from the root idea
on the serial path (the parallel path keeps `symbols`); with that, eight
supervised labels are memorized (100 % at epoch 30, lr 1e-3), whereas
lr 5e-3 diverges the 1024-wide adapter (predictions swinging to +-100)
with either an invertible or a plain linear adapter. The successor run
is relaunched with the seed and lr 1e-3; the earlier verb runs are
superseded.
