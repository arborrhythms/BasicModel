# Fold-ladder throughput and word-unit assurance (2026-09-11)

Status: **measured on the canonical ladder topology; the legacy radix
comparison could not be run on equal footing.** Alec asked for two
measures on the [meronomy fold-ladder](../plans/2026-09-10-meronomy-fold-ladder.md)
path: sentences per second, and an assurance that the model still
operates over words now that the analysis tiling is learned rather than
fixed. Both now ride the epoch report (doc/Training.md, "Epoch report").

## Setup

| Item | Value |
|---|---|
| Commit | basicmodel `main`, third slice of the fold-ladder plan (this commit) |
| Configs | `data/MM_ladder.xml` (successor corpus, W=16 for the loop), `data/MM_ladder_text.xml` (FineWeb shard, one sentence per row, W=128), `data/MM_ladder_textpacked.xml` (packed rows, W=256, `maxSentenceWords` 40) |
| Hardware | macOS ARM64, MPS (`BASICMODEL_DEVICE` default), torch 2.14.0.dev20260722 |
| Schedulers | `static`: `MODEL_COMPILE=none` (eager A/B/C scheduler); `eager HOP`: `MODEL_COMPILE=none BASICMODEL_TENSOR_PEER_WHILE=always` (the `torch.while_loop` word pipeline run eagerly); `compiled`: default `MODEL_COMPILE` (inductor, one fullgraph word loop) |
| Batches | 16 optimizer batches of B=8 rows; steady state = median per-batch wall-clock after the first three batches |

Reproduce (one cell):

```sh
cd basicmodel
MODEL_COMPILE=none BASIC_MAX_BATCHES=16 BASIC_NUM_EPOCHS=1 \
    .venv/bin/python bin/Models.py data/MM_ladder_text.xml
```

## Sentences per second

| Corpus | Scheduler | Steady sentences/s | Median batch (8 rows) |
|---|---|---|---|
| successor (math, W=16) | static | 4.39 | 1.82 s |
| successor (math, W=16) | eager HOP | 1.50 | 5.32 s |
| successor (math, W=16) | compiled | 1.50 | 5.35 s |
| FineWeb, one sentence per row (W=128) | static | 0.71–0.91 (two runs) | 8.8–11.3 s |
| FineWeb, one sentence per row (W=128) | eager HOP | 0.46–0.49 (two runs) | 16.2–17.2 s |
| FineWeb, one sentence per row (W=128) | compiled | 0.42 (measured while the test suite ran on the CPU) | 19.3 s |
| FineWeb, packed rows (W=256) | static | about 1.3 (29 sentences per brick, 21.9 s per brick; 2.2 incl. warm-up over the first two bricks) | 21.9 s |

Observations.

- The `torch.while_loop` word pipeline is about three times slower than
  the static eager scheduler on both corpora, and compiling it (inductor
  on MPS) does not recover the difference: the compiled successor run
  matches the eager HOP within noise. The per-batch time is dominated by
  work outside the loop (the eager stem, the boundary learner's host
  counters, the reduce/reverse tail), so the loop's compilation is not the
  lever for throughput on this topology.
- Text is an order of magnitude slower than the successor corpus per
  sentence: rows carry about 2.5 units per whitespace word (spaces and
  punctuation are units of their own), and the reconstruction tail scales
  with the row.
- For context, the pre-ladder production text config measured 6.8
  sentences/s on its own topology (`2026-07-27-pre-teacher-baseline.md`);
  the ladder fixture here is a 3.4M-parameter canonical topology with an
  8192-byte input, not that config.

## Where did the throughput go? (2026-09-12)

> Superseded for approval purposes by the measurement protocol section
> below (Codex review, requirement 6): the per-brick times here are single
> or few samples of the forward-dominated brick, some taken while other
> work ran on the machine. They remain as the record of the attribution.

Alec asked whether the July record of about 40 sentences/s
(`2026-07-27-teacher-reconstruction-b24.md`: `BasicModel.xml`, B24, W256,
packed FineWeb, MPS, Inductor fullgraph) was lost to the radix trie's
replacement. Measured on the same config against the last pre-ladder
commit (`fc93560`, a worktree) on the same machine, without the prefetch
thread (`numWorkers` 0; see the Metal note below), 400 documents:

| Tree | Steady brick, B=8 (forward / total) | Brick, B=24 compiled | Notes |
|---|---|---|---|
| pre-ladder `fc93560` | 7.7 s / 8.0–8.3 s | 10.0–12.3 s | one compile at brick 0 |
| current, before this pass | 8.2 s / 8.6 s | 27.7 s (one sample) | a second compile on brick 1, and one more on every new unit count |
| current, after this pass | 8.2 s / 8.7–8.9 s | not re-measured (GPU memory) | one compile at brick 0 |

So the ladder front end itself costs about 7 % per brick (the stem: 0.35
s versus 0.25 s at B=8; units per whitespace word are 1.2 on this config,
since spaces are not units here and punctuation is), and the compiled
word loop is unchanged (10.9 s for 258 trips on both trees in the
profiled B=24 brick). The July rate of about 40 sentences/s is
consistent with the pre-ladder brick times measured today (about 300
sentences per 10–12 s brick on this machine, which is under other load)
and is not what the ladder lost. What the ladder had lost, and this pass
fixed:

- three recompile triggers of the word loop, each a full 15–23 s
  compile: the staged unit parent/clause maps had a width equal to the
  batch's unit count (now padded to the word capacity at the stem), and
  the chunk proposal slab and the STM whole slab were first allocated
  inside the compiled call, on the wrong device and batch (now
  allocated eagerly on the word slab's device before any compile);
- the boundary learner's candidate tilings ran with the learning rate at
  zero (0.8 s per B=8 brick; now gated), and the LBG pull recorder did
  one device sync per unit and row (0.9 s; now one host pass);
- the sentence packer's unit counter, running on the prefetch thread,
  read the boundary parameters on the GPU while the main thread was
  encoding: Metal aborted the process. The tiler now reads a host copy,
  primed whenever the predicates are built or updated and refreshed by
  every main-thread tiling; a worker thread that finds no copy fails
  loud instead of touching the accelerator. Verified with a
  torch-function mode on the prefetch thread (no accelerator op issued
  over three bricks) and two production runs with the default two
  workers (no Metal assertion; they end on the memory ceiling below).

The production config at 2,000 documents does not fit this machine's
16.85 GiB MPS limit past three bricks on either tree (the aligned prefix
growth), so the B=24 numbers above are three-brick samples.

## Measurement protocol and results (2026-09-12, Codex review requirement 6)

Protocol (`bin/bench_training_step.py`): `data/BasicModel.xml` unchanged except the
environment overrides `BASIC_MAX_DOCS=400 BASIC_BATCH_SIZE=B
BASIC_MAX_BATCHES=n BASIC_NUM_EPOCHS=1`; default `MODEL_COMPILE` (Inductor
on MPS, the production backend, one fullgraph word loop per bucket), the
config's two prefetch workers; torch 2.14.0.dev20260722, macOS ARM64,
38.65 GB unified memory, 16.85 GiB MPS allocation limit. Each brick is one
full training step (`runBatch`: forward, loss, backward, optimizer,
resets), wall-clock with device synchronisation. Executed losses on this
config: `lossIn` = the detached idea-only reverse student (with gradient),
`lossOut` = 0 without supervision at B=8 (a 0.10 term with gradient
appears at B=16); `d3_active` true, `detached_reverse` true on both trees.
Recompiles from dynamo's frame counters; peak memory from
`torch.mps.driver_allocated_memory`. Trees: pre-ladder `fc93560` (a
worktree) and the current `main` (`874303c`).

| Tree | B | Compile brick | Steady bricks (s) | Backward-compile brick | Recompiles | Peak MPS memory |
|---|---|---|---|---|---|---|
| pre-ladder | 8 | 22.4 s | 8.11, 8.33, then 7.94, 7.95 | 28.7 s | 0 | 13.1 GB |
| current | 8 | 25.5 s | 8.94, 9.15, then 12.86, 12.92 | 37.9 s | 0 | 15.3 GB |
| pre-ladder | 16 | 23.5 s | 8.97, 11.55 | 37.2 s | 0 | 16.0 GB |
| current | 16 | 26.5 s | 9.51, 10.32 | 40.3 s | 0 | 16.0 GB |

Reading: the first two steady bricks are forward-dominated (the optimizer
brick boundary lands later at this batch size); the bricks after the
backward-compile brick include backward and optimizer work. On those the
current tree is 12.9 s against 7.9 s, with peak memory 2.2 GB higher and
close to the MPS limit; the forward-dominated bricks are 10 % slower. Both
trees show zero recompiles after the first brick. The idea-only student's
loss values track each other (4.83 to 4.66 over six bricks on both). The
compile bricks are 3 s longer on the current tree.

Re-run of the current tree at B=8 with the charts as custom autograd
Functions (saving only their input instead of the straight-through
surrogate's intermediates): compile brick 25.4 s, steady 9.15, 9.28, then
12.68, 12.69 s, backward-compile brick 35.5 s, peak 16.7 GB. The
backward-brick gap (about 4.8 s per brick against the pre-ladder tree)
therefore does not come from the charts' saved tensors, and the peak-memory
reading is not a reliable discriminator at this distance from the limit.
The gap remains open and unattributed; the next step is the same protocol
with the backward profiled at B=8 (the forward-dominated bricks are
within 10 %).

## Legacy radix comparison

The plan's Phase 4 asks for the ladder against the legacy `radix`/`word`
front end. On this topology the legacy configuration cannot be built: the
legacy InputSpace lexer allocates two dense `(nOutput * nDim)^2` matrices
(`inputSpace.layers.0.raw_L/raw_U`, 69632 x 69632 at a 512-byte input,
about 10 GB each; 1.1M x 1.1M at the fixture's 8192-byte input), and the
process is killed while moving them to the GPU. The ladder path builds no
such matrices. A fair comparison needs a legacy configuration that fits,
which is a topology choice for Alec, not a benchmark setting.

## Word-unit assurance

`word units` in the epoch line counts the whitespace-delimited words of
the epoch that were staged as exactly one unit
(`BaseModel.word_unit_fraction()`), with a letters-only figure beside it.

| Corpus | word units | letters-only words |
|---|---|---|
| FineWeb (2056 words, 16 batches) | 84.6 % | 100.0 % (1715 of 1715) |
| successor corpus (`12 plus 1`) | 2 of 3 words per problem | 100 % (`plus`) |

The 15 % of FineWeb tokens that are not one unit contain punctuation or
digits (`word,` is two units; `1998` is four digit units under
`<digitWholes>`), which is the canonical tiling by design. Every
letters-only word is one unit. Under `<boundaryTypes>none</boundaryTypes>`
the figure starts at 0 % (atomic tiling) and rises as the boundary learner
acquires the whitespace boundary (test/test_meronomy_ladder.py).

## Defects found on the way (fixed in this commit)

- Compiled forward lost the STM after the word loop: with a
  `torch.while_loop` in the graph, dynamo (torch 2.14 nightly) drops a
  later attribute *assignment* of an attribute assigned earlier in the
  same graph (the per-forward STM seed), so the sentence reduce saw an
  empty STM under compile and the published root idea was zero. On the
  ladder text config that made `lossIn` (there the per-word D3
  reconstruction objective driven from the root) a constant without
  gradient; on the successor corpus training continued through the
  answer loss. The STM's live state is now committed by in-place copy
  under the compiler (`ShortTermMemory._assign_live`), and the compiled
  sentence idea equals the eager one. Correction (Codex review, 2026-09-12):
  `reverseReconstruct` itself is not a training path on any of these
  configs; with `<detachedReverse>` (production `BasicModel.xml`) the
  training `lossIn` is the detached idea-only student
  (`_detached_reverse_construction_loss`), and the trace-driven un-fold is
  evaluation only.
- Host-side tiling tensors were created on the default device (MPS) and
  mixed with CPU indices; pinned to CPU.
- Sentence packing counted whitespace words while the ladder stages units;
  bricks are now budgeted and laid out in units (doc/Training.md).
- The legacy row map `_ws_row_to_pos` was read unguarded on the decode
  path under the property basis.

## Packed rows with whitespace units: the NaN (diagnosed and fixed 2026-09-12)

The non-finite gradient in `perceptualSpace.sigma.raw_bfly_L` on packed
FineWeb rows was traced node by node (autograd hooks on every node of the
loss graph, anomaly-mode forward tracebacks). Findings, in backward order:

- The first node that turns a moderate gradient into a huge one is the
  atanh chart of the grammar's `lift` fold (`SigmaLayer.compose`,
  `atanh(x.clamp(-1+1e-7, 1-1e-7))`): a gradient of 135 became 1.9e6 on
  operands sitting at exactly +-1. Such operands are ordinary: a unit's
  rung-0 code is a max over atom codes (many coordinates at +-1), the
  `chunk` op is an unnormalised sum (up to +-2), and a deep fold
  saturates. The chart's derivative, `1/(1-x^2)`, is 5e6 at the clamp.
- Each nested fold of a long packed row applied that chart again; the
  gradient grew about 100x per level over 15 levels (2e34 at the leaves)
  and overflowed in the pi chart's `log`. Whitespace units matter only
  because they double the row's depth; with them off the same rows
  merely grew to 1e14.
- The same chart (`2*atanh`) sits inside the pi (`lower`) fold, and every
  sigma butterfly pair op, the part-code synthesis and the balanced
  inverse use the 1e-7 clamp. Each fold's own gain is otherwise about 1
  (measured per stage: tanh 0.4, inner map 1.0, chart 2), and the reduce
  step's chooser blend adds 1.1-1.5.

Fix: `Layers.bounded_atanh` (and `PiLayer._log_mult`) keep the exact
forward value and clamp, and cap the backward slope at the tangent at
`|x| = 0.9` (5.3 for atanh, 10.5 for the log-odds chart) as a
straight-through estimator. Every atanh chart in Layers, Language and
Spaces goes through it. Forward numerics are byte-identical; only the
gradient of near-saturated coordinates is bounded. Validation: the same
packed configuration (`data/MM_ladder_textpacked.xml`, whitespace units
on, CPU) trained 12 bricks with finite gradients (643 sentences; word
units 85.7 %, letters-only 100 %), where it failed on the fourth before;
test/test_bounded_charts.py pins the charts and a 30-deep saturated fold.

## Open

- The production config at 2,000 documents exceeds this machine's
  16.85 GiB MPS limit after two or three bricks on either tree (the
  aligned prefix growth); 400 documents fit.
