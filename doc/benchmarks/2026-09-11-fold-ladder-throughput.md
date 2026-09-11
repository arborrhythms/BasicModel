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
  same graph (the per-forward STM seed), so the sentence reduce and the
  reconstruction loss saw an empty STM under compile (loss without
  gradient on text; the successor corpus trained only through the answer
  loss). The STM's live state is now committed by in-place copy under the
  compiler (`ShortTermMemory._assign_live`), and the compiled sentence
  idea equals the eager one.
- Host-side tiling tensors were created on the default device (MPS) and
  mixed with CPU indices; pinned to CPU.
- Sentence packing counted whitespace words while the ladder stages units;
  bricks are now budgeted and laid out in units (doc/Training.md).
- The legacy row map `_ws_row_to_pos` was read unguarded on the decode
  path under the property basis.

## Open

- Packed FineWeb rows with `<whitespaceUnits>true</whitespaceUnits>` hit a
  non-finite gradient in `perceptualSpace.sigma.raw_bfly_L` by the fourth
  brick (also on CPU; not the atanh clamp, not the membership-curve
  discriminant, both tried). With whitespace units off the same rows train
  for 48 bricks. The anomaly trace ends at the unary chooser's
  `action_probs * branches` blend (bin/Language.py), where an infinite
  incoming gradient meets an exactly-zero branch probability; the source
  of the infinite gradient is not yet located. One-sentence-per-row text
  and the successor corpus do not show it.
- MPS runs of the packed configuration also died twice in the Metal
  driver (`A command encoder is already encoding` abort; a segfault at
  12.5 GB footprint), independent of the NaN.
