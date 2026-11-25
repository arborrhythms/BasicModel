# Item 1.5 low-level continuation

Parent: `67b22d7d92`, on local branch `item1-5`. Alec stopped further
performance investigation and directed removal of observed `.item()` and
`.tolist()` calls within Codex's perception/spelling half. Claude's grammar,
memory, scoring and optimizer files are untouched. No push or integration.

## Changes

- `BasicPercepts.lookup` no longer copies incoming native IDs and extents to
  Python to expand recorded ancestry. Tensor indexing preserves nested and
  repeated leaves, original occurrence order, byte extents and empty rows.
  Scratch width is bounded by input slots and the admitted ancestry; the
  existing occurrence merge still selects the final live width. No new
  persistent inventory or cache was added.
- Native inventory references in `_boundary_codes` use their known address
  namespace directly, avoiding two device predicates and boolean compaction.
  Empty reads retain absent gradients. The duplicated native-half gather in
  `_stored_boundary_codes` now shares the same helper.
- `NativeDecode.forward` uses the existing sentence assertion helper. Its
  redundant `no_grad` wrapper was removed: a failed assertion in the captured
  wrapper left the caller's gradient mode disabled. External validation and
  compiled Metal assertion failures remain tested.

## Validation

- Perception/spelling suite: **99 passed, 11 skipped, 1 baseline failure**.
  `test_reconstruction_extends_only_with_agreeing_proper_occurrences` fails
  identically at parent `67b22d7d92`: its fixture lacks `form_root`. Its
  assertions were not modified. This is separate from the documented 1.8
  omitted-word failure.
- Focused physical CPU/MPS checks: **10 passed**. The additional explicit
  prohibition of `.item()`/`.tolist()` during ancestry expansion: **2 passed**.
  Expansion captures with `fullgraph=True`, backend eager, on both devices.
- Earlier boolean-only checks: all **628** CPU replay tensors matched exactly,
  including optimizer state and 28 live gradients. MPS maximum differences:
  parameters `3.35e-8`, optimizer `7.45e-9`, live gradients `2.91e-11`, with
  matching gradient presence and sparse indices. These replays preceded the
  ancestry edit; they do not certify that edit's full-update equivalence.
- Before the instruction to stop profiling, the real MPS decoder captured
  with strict Inductor: one graph, 1.093 s compiler time. The boolean-only
  paired four-row lesson removed seven Python boolean reads (154 → 147),
  with essentially unchanged throughput (2.186 → 2.187 sentences/s).
  There is no new throughput claim for the ancestry edit.

Receipts are under `output/item1-5/resume-20261125-`: `ancestry-suite.log`,
`ancestry-baseline-failure.log`, `ancestry-metal.log`,
`ancestry-host-read-check.log`, `bool-three-cpu/compare.json`,
`bool-after-one-mps/{compare,gradient_magnitudes}.json`, and
`bool-decode-strict-mps/`. An earlier broader reverse suite also exposed two
unchanged fixtures missing `_concept_allocator`; both reproduce at the parent
(`bool-baseline-failures.log`).

Whole-sentence capture is still incomplete. Learned-span admission remains
the first failure. Remaining dynamic perception reads and the lexical writer
still need conversion; do not interpret these function checks as the item 1.5
completion gate. The prior fresh-reader/XOR receipts remain in the parallel
handoff; they were not rerun after these edits.
