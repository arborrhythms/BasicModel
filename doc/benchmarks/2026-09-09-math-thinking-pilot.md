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
