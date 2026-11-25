# Item 1.5: Codex / Claude work split

Requested by Alec on November 25. Split the remaining implementation at the
handoff from located percepts to grammatical selection. These are two
substantial halves with one shared acceptance gate; neither half alone closes
item 1.5.

## Starting point

- Existing checkout: `/Users/arogers/github/WikiOracle/basicmodel`, branch
  `item1-5`. Codex continues here.
- Tested code checkpoint: `79240a2bed` (parent `24a01c5597`). The original
  integration base remains `ad8fafea9278`, including reviewed 1.7 `d380291131`.
- Claude should start a separate branch/worktree from this checkpoint. Do
  not run two writers in the existing checkout. Commit locally; Alec owns
  review, pushing and integration with Claude's item 1.6 work.
- Governing decisions: [device sentence](2026-11-24-item1-5-device-sentence.md),
  [previous checkpoint](2026-11-24-item1-5-handoff.md), `todo.md`, and
  [item 1.5 scope](2026-11-14-item1-5.md). Ask Alec before further architecture
  changes. This split allocates implementation ownership; it changes none
  of those decisions.

## Half A — Codex: observed bytes through located percepts

Remove the first real fullgraph failure and its upstream/downstream lexical
dependencies. The current failure is in
`PartSpace._embed_ladder_word_major`: Python consumes learned span endpoints
before spelling admission. Converting only that comparison is insufficient;
the endpoints also feed string slices, identity admission, omission and
duplicate Python word/interval records.

Deliver:

1. Device-resident learned spans, disjoint spelling parts and observed-byte
   evidence, with tensor occupancy and fixed capacity. Input admission may
   supply bytes and valid extent, not learned word boundaries or fixations.
2. Native spelling admission and collision/containment repair preserving the
   current 1.7 definitions, code identity and observing-page visibility.
   `WordIdentity.bit_stream` currently uses SHA-256 and NumPy permutation;
   preserve its numerical contract if replacing it. No replacement code for
   this primitive has been written at the checkpoint.
3. The model's direct located-percept choice from observations, including
   disjoint parts and muxed `.what`/`.where`, plus each path's own causal
   forecast. Preserve wide-open attention, one whole-field reconstruction,
   subsequent narrowed looks and optional thinking. No supplied fixations.
4. Producer-side tensors and inverse spelling that work from fresh inventory,
   through repeated/escaped bytes, omissions, packed sentences and mixed
   candidate winners. Remove superseded host metadata after consumers move.

Primary files: `WordIdentity.py`, `Meronomy.py`, `WordSurfaces.py`,
`BasicPercepts.py`, `BlockPercepts.py`, `FieldReverse.py`, `SentenceLooks.py`,
`ModelAttention.py`, `LookForecast.py`, and lexical/input/whole-field methods
in `Spaces.py`. Codex also owns the benchmark harness and final matched CPU/MPS
measurements. Coordinate `BasicPercepts`/`WordIdentity` changes with 1.6.

## Half B — Claude: grammatical selection through retained sentence close

Make the chooser's STM head the actual parse, then carry that representation
through scoring, selected learning and retention without host decisions.
Start independently with tensor fixtures that supply located percepts; also
exercise the existing live producer while Half A is being converted.

Deliver:

1. Write located STM0 occurrences and operator-prefixed ordered STM1 links
   when the chooser makes its decisions. Use the emitted handles immediately.
   Preserve exact operand order, distinct repeated occurrences, unary and
   binary operations, bind, meaningful transpositions, partial reductions
   and the raw no-grammar chain. There is at most one root per stream/tick.
2. Move grammar consumers from `ReadingIndex`, `recorded_steps` and Python
   action replay to the existing typed references. Keep loss operands,
   results and policy probabilities live for differentiation; do not add
   persistent action/kind/parent columns to the parse. Preserve exact inverse
   behavior and unambiguous operator interfaces.
3. Device-side candidate scoring and per-row winner selection, selected
   backward, optimizer update and close. A rejected candidate must leave the
   winning stream's STM, LTM, clock, optimizer and gradients intact. Reuse the
   checkpoint's tensor-masked `NativeReading.commit(paths, winners)`.
4. Preserve STM → MTM → LTM ownership and the reachable-head copy. Remove
   remaining host grammar, clause and event dependencies inside close. Do not
   implement a second parser, replay at commit, or a second memory owner.

Primary files: grammatical `_derivation_program`, `_capture_reading_programs`,
`_program_entries`, chooser and sentence-close/training methods in `Models.py`;
`ReadingIndex.py`, `SentenceStrata.py`, `Extension.py`, `Comprehension.py`,
`SentenceCredit.py`, `WholePaths.py`, `ReadingMemory.py`, `Language.py`, and
related `CandidateGrowth`, `MediumTermMemory`, clause/grounding and memory
writers as needed. This is the half closest to Claude's number, object
grounding and felt-world work; keep its changes reviewable apart from 1.6.

**Optimizer constraint:** preserve absent gradients as absent. The approved
regression distinguishes an inactive Adam parameter with existing momentum
from a participating parameter with exactly zero derivative. The proposed
participation-mask architecture was rejected. Do not silently replace
`.grad is None` with zeros or change the optimizer to obtain capture. If this
requires a new architectural decision, isolate the reproduction and ask Alec.

## Shared interface and edit boundaries

Use the existing `SentenceLooks.Looks.run(arrive)` / grammatical arrival seam.
Half A supplies each selected occurrence's existing tensor value, native
typed identity, ordered observed byte parts and their extents, location/time
and active-row mask. Half B writes and consumes STM references for that
arrival. Tensor shape/capacity may be known to Python; learned values and
selection decisions stay on the device. Keep one independent sentence per
batch row, with explore/exploit paths inside that row.

Both halves use Space-owned working storage and the Model forward/reverse/
reset lifecycle. Agree any required tensor signature change before changing
both ends. Preserve the current interface until its replacement has a
producer and a consumer; historical adapters belong at load or in tests.

`Models.py`, `Spaces.py` and `SentenceField.py` are shared integration files:
Codex owns input/opening/located-field production, Claude owns grammatical
arrival, semantic interpretation, selection and retained close. Send small
method-level patches for a change crossing that line. In `SentenceField.py`,
Codex owns field reconstruction/percept production; Claude owns action/parse
consumers and typed-head traversal. Avoid whole-file rewrites. Flag redundant
structures rather than introducing parallel representations.

## Checkpoint verification and remaining gate

`79240a2bed` removes weight-dependent graph compaction in forecast/salience
and host winner-list use in native forecast retention. It also adds native
operator timing to the separate diagnostic pass. It does **not** fix lexical
admission or whole-sentence capture.

- CPU focused suite: 83 passed, 5 skipped. Additional forecast/ownership
  suite: 32 passed. These groups overlap; do not sum them as unique tests.
- Physical MPS/CPU parametrized retention, structure and salience tests:
  6 passed. Actual MPS training inputs capture `ConceptSpread.structure` and
  `Salience._concepts_asked` with strict Inductor. These are subroutines only.
- Fresh restored reading: 122/122 sampled and 122/122 evaluation on CPU and
  MPS. Wall time to the qualifying saved-reader result: 40.730 s CPU and
  145.527 s MPS, seed 1505. CPU needed two lesson epochs, MPS one; the gate
  and settings match, not the number of updates to qualification.
- Verbal XOR both boundaries 4/4, numeric XOR convergence, six native numeric
  XOR cases and raised-class assertion: 9 passed in 213.17 s. The external
  run used seed 1859289298; existing assertions were retained.
- CPU three-exploration replay: all 628 tensors checked exactly equal,
  including 28 live gradients. MPS maximum parameter differences were
  2.98e-8 (one exploration) and 6.33e-8 (three); maximum live gradient
  differences 2.91e-11 and 5.82e-11. Gradient presence and sparse indices
  agree. MPS is not bit-reproducible; exact comparison reports remain saved.
- Four-row MPS, 12 full training updates, first three excluded: before/after
  2.326/2.425 sentences/s with one exploration; 1.317/1.393 with three.
  These are single paired observations. Full post-integration CPU/MPS matched
  one/three-exploration and time-to-accuracy benchmarks are still required.
- Receipts have prefix `output/item1-5/resume-20261125-`: `baseline-strict`,
  `salience-suite.log`, `forecast-suite2.log`, `latest-metal-tests.log`,
  `salience-strict-metal`, `structure-strict-metal`, `fresh-lesson-{cpu,mps}`,
  `learning-gates.log`, `salience-three-cpu`, and `{before,after}-{one,three}-mps`.
- The documented 1.8 omitted-word failure remains separate. Its assertion
  has not been weakened or reclassified as an item 1.5 pass.

After both halves land, require strict **whole-sentence** capture through
scoring, selected backward, optimizer and close, plus zero in-sentence host
dependencies audited on physical MPS. Do not move decisions outside the
measured boundary. Re-run fresh learning and mixed-winner state/gradient
gates, then synchronized matched benchmarks with compilation and warm-up
separate. The [MPS research note](2026-11-25-item1-5-mps-research.md) records
why host interrupts are still a hypothesis and how profiling perturbs timing.
ExecuTorch/MLX is deferred; reuse the existing `export_mlx.py` when resumed.
