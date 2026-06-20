# WordSubSpace / STM Layer Refactor

## Summary

Move standalone helper modules into the classes that own their behavior,
remove stale aliases and shims, and fix the data model around
`WordSubSpace`.

The key model change: `WordSubSpace` is the mutable data carrier.
`ShortTermMemory` is a data-free `Layer` owned by `ConceptualSpace` that
stores weights and acts on `WordSubSpace`.

## Key Changes

- Make `WordSubSpace` a real `SubSpace` subclass.
  - Move the stack data currently split across `ConceptualSpace.stm` and
    `typed_stack.py` into `WordSubSpace`: payload frames, depth, category,
    order, ref id, cursor/recur_pass, and parse state.
  - Remove `SubSpace.wordSpace` back-pointer plumbing and pass
    `wordSubSpace` explicitly.
  - Remove `WordSpace = WordSubSpace` and rename runtime/model fields from
    `wordSpace` to `wordSubSpace`.
  - Keep XML/config section name `<WordSpace>` for compatibility.

- Move `ShortTermMemory` to `Layers.py` and derive it from `Layer`.
  - It stores scorer/reducer weights only, no `_buffer`, `_depth`, or
    per-sentence data.
  - Inline `stm_driver.py` and `stm_trainer.py` behavior as methods on
    `ShortTermMemory`.
  - Use methods like `shift(word_subspace, ...)`,
    `reduce_step(word_subspace, ...)`, `reduce_step_soft(word_subspace, ...)`,
    and `train_scorer_step(word_subspace, ...)`.
  - `ConceptualSpace` owns `self.stm` and invokes it during `forward`.

- Change the conceptual forward contract.
  - Replace `ConceptualSpace.forward(PS_subspace, SS_subspace=None)` with
    `ConceptualSpace.forward(subspace, word_subspace)`.
  - `WordSubSpace` takes over the symbolic-loop data-container role formerly
    held by the separate `SS_subspace`.
  - Update `PerceptualSpace`, `SymbolicSpace`, and model loop call sites to
    pass `wordSubSpace` explicitly.

- Move GPU/token-symbol helpers into `Layers.py`.
  - Add `BPEGpuLayer(Layer)` and `MPHFGpuLayer(Layer)` with nested exceptions
    and private/static helper methods.
  - `PerceptualSpace` owns instances of these layers and their static-table
    caches.
  - Delete `bpe_gpu.py` and `mphf_gpu.py`.

- Move symbol learning into `Layers.py`.
  - Add `SymbolLearningLayer(Layer)` and inline stats, policy, config
    enablement, and flush/promote behavior.
  - `ConceptualSpace` owns and wires it.
  - Symbol promotion still happens only at explicit flush boundaries, not
    inside autograd `forward`.

## Removed Public Surfaces

- Delete modules without compatibility shims: `bpe_gpu.py`, `mphf_gpu.py`,
  `stm_driver.py`, `stm_trainer.py`, `typed_stack.py`, and
  `symbol_learning.py`.
- Remove direct imports of `WordSpace`; all code imports and uses
  `WordSubSpace`.
- Remove direct construction of `TypedStack`, `STMDriver`, `RuleScorer`,
  `SymbolLearningStats`, and `SymbolLearningPolicy` outside their owning
  classes.

## Test Plan

- Update STM tests to construct `WordSubSpace` data plus `ShortTermMemory`
  layer instead of `TypedStack`/`STMDriver`.
- Update symbol-learning tests to target `SymbolLearningLayer`.
- Update BPE/MPHF tests to use `BPEGpuLayer` and `MPHFGpuLayer`.
- Run focused suites for STM, BPE/MPHF, symbol learning, per-word pipeline,
  and `ConceptualSpace.forward` call-site changes.
- Add regression checks that no `WordSpace` alias, `SubSpace.wordSpace`
  back-pointer, or duplicate STM stack storage remains.

## Assumptions

- No compatibility shims are kept for removed Python modules.
- Existing XML `<WordSpace>` config remains unchanged in this refactor.
- `WordSubSpace` is a data carrier despite inheriting `SubSpace`; it is not a
  pipeline `Space`.
- `ShortTermMemory` owns weights and algorithms only; all runtime stack
  contents live on `WordSubSpace`.
