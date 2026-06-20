# Static Per-Word Loop And OOV Reserve Spec

## Summary

Stabilize `MM_5M` compilation by removing the data-dependent
`while next_word()` termination from the compiled path. The per-word
body will run a fixed `InputSpace.nOutput` loop, and NULL/padding
positions will be tensor-masked no-ops. OOV inserts will happen only
before compiled compute, into preallocated rows; when capacity fills,
training falls back and counts misses instead of resizing codebooks.

## Key Changes

- Replace the compiled per-word `while next_word() is not None` path
  with a static loop over `range(InputSpace.outputShape[0])`.
- Add a fixed-index `word_at(p)` style feed that always returns
  `[B, 1, D]`, paired with a device-side `word_active[:, p]` mask.
- Remove compiled-path dependence on `InputSpace._valid_len_host`; it
  may remain as an eager diagnostic but must not gate compiled control
  flow.
- Define NULL-word no-op semantics as masked state preservation:
  inactive rows do not update recurrent carriers, STM depth, STM slots,
  or per-word concept output.
- Preallocate the per-word concept output as `[B, N, D]`; inactive
  positions stay zero.
- Keep per-sentence working state object-stable: allocate
  `wordSpace._work` before compile, reset it in place, and keep
  `cs_for_ps` / `cs_for_ss` carriers as stable objects whose tensors
  are updated in place.
- Move side effects out of compiled per-word compute: no Python list
  growth, `None` sentinels, `int(tensor)`, `.item()`, `Error.add()`
  dict mutations, or debug counter mutation inside the compiled loop.
- Stage discourse prediction before compiled compute and cast staged
  cache tensors to the active autocast dtype so dtype guards do not
  split graphs.

## OOV/Codebook Policy

- Preallocate reserve capacity for codebooks that may receive OOV rows.
- Perform OOV staging before compiled forward/reset-step compute:
  collect unique OOV keys for the upcoming batch/sentence, assign
  inactive reserve rows, copy vectors under `torch.no_grad()`, update
  Python key maps outside compiled code, and clear optimizer state for
  newly activated rows if Adam state already exists.
- Never resize codebook parameters during training.
- When reserve capacity is full, route OOV entries to the BPE/byte/UNK
  fallback, increment an OOV fallback counter, and keep a small sample
  of missed keys for offline rebuild or pruning.
- Keep 1M lexical coverage in PerceptualSpace/MPHF/BPE. Do not make
  SymbolicSpace a 1M-row lexical table unless that is a separate,
  explicit experiment.

## Test Plan

- Compile stability: run batches with different observed word counts
  such as 5, 8, and 15 words, then assert no recompiles are caused by
  `_valid_len_host`.
- Compile hygiene: assert no recompiles from `wordSpace._work is None`,
  `Error.add()` dict mutation, `cs_for_ps` empty/non-empty object
  identity changes, or discourse cache dtype changes.
- NULL no-op correctness: an active prefix followed by padding must
  produce the same STM depth/state as running only the active prefix.
- Row masking correctness: inactive rows must not update recurrent
  carriers, STM, or per-word concept slots.
- OOV reserve correctness: inserting OOV rows must not change codebook
  parameter shape or parameter identity, and newly activated rows must
  be visible through key lookup before compiled compute.
- Full reserve behavior: exhausting reserve capacity must trigger
  fallback-and-count behavior, not growth.
- CUDA acceptance on `metalbaby`: run
  `TORCH_LOGS=recompiles ... --batches 5 --compile-mode max-autotune`
  and confirm there are no repeated whole-forward recompiles for
  changing sentence length.

## Assumptions

- Use the full `InputSpace.nOutput` loop, not bucketing.
- NULL-word no-op means preserving all row state, not merely feeding a
  zero vector.
- OOV-full behavior is fallback-and-count, not eviction or abort.
- Codebook tensor shapes are fixed for the duration of a training run.
- The primary fix target is the repeated `_valid_len_host` guard seen
  in remote CUDA `TORCH_LOGS=recompiles` output.
