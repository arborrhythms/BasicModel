# WordSubSpace / STM Layer Refactor — Implementation Plan

Spec: [doc/specs/2026-05-21-wordsubspace-stm-layer-refactor.md](../specs/2026-05-21-wordsubspace-stm-layer-refactor.md)

## Pre-existing failures (out of scope)

Already failing on `main` before this work; not addressed here:
- `test/test_basicmodel.py::TestInputSpaceTextRoundTrip::test_reconstruct_data_joins_words`
- `test/test_basicmodel.py::TestReconstructionSymbols::test_forward_output_shape_unchanged`
- `test/test_compiled_step_invoked.py::test_compiled_step_is_invoked`
- `test/test_mm_xor.py::TestMMXorConvergence::test_convergence`
- `test/test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal`

Per the cross-position-mixing spec (2026-05-21-cross-position-mixing-via-nInputDim.md), 4/5 belong to a separate effort; the text-reconstruction one is independent.

## Ordering rationale

Order chosen to minimize cascading breakage:
1. **Independent additive moves first** (BPE/MPHF/SymbolLearning) — these touch isolated subsystems with few external callsites.
2. **Foundation change next** (WordSubSpace → SubSpace + STM data move). This is the load-bearing change; everything else builds on it.
3. **Layer migration** (ShortTermMemory) — feasible only after WordSubSpace owns the data.
4. **Forward-contract change** — last semantic change before mechanical renames.
5. **Mechanical renames + alias removal** — done at the end so intermediate phases can rely on `WordSpace = WordSubSpace` and `subspace.wordSpace` while the refactor is mid-flight.
6. **Cleanup deletes + test migration + regression sweep**.

## Phases

### Phase A — BPE GPU helpers into Layers.py
- Add `BPEGpuLayer(Layer)` to `bin/Layers.py`. Move `_BPEGpuUnavailable`, `_HASH_MUL`, `_poly_hash`, `build_static_tables`, `gpu_longest_match`, `segment_words`, `gpu_chunk_ids` from `bin/bpe_gpu.py` as nested exception and private/static methods.
- `PerceptualSpace` constructs and holds a `BPEGpuLayer` instance; `_bpe_static_tables` cache moves onto the layer (the `_vsig` rebuild trigger preserved).
- Update Spaces.py call sites (lines ~8113, 8127, 8148–8159).
- Delete `bin/bpe_gpu.py`.
- Regression gate: `pytest test/test_chunk_layer_bpe.py test/bpe_gpu_equiv.py` (if those exist).

### Phase B — MPHF GPU helpers into Layers.py
- Add `MPHFGpuLayer(Layer)` to `bin/Layers.py`. Move `_MPHFUnavailable`, `build_mphf_table`, `mphf_index`, `reverse_map_rows` as nested exception and methods. Note: `_poly_hash` is shared — keep one canonical copy on `BPEGpuLayer` (or move to a util) and have `MPHFGpuLayer` reference it.
- `PerceptualSpace` constructs and holds an `MPHFGpuLayer` instance; `_mphf_static_tables` cache moves onto the layer.
- Update Spaces.py call sites (lines ~8691, 8716, 8741, 8887).
- Delete `bin/mphf_gpu.py`.

### Phase C — SymbolLearningLayer into Layers.py
- Add `SymbolLearningLayer(Layer)` to `bin/Layers.py`. Inline `_Cluster`, `SymbolLearningStats`, `SymbolLearningPolicy`, `symbol_learning_enabled_from_config`, and `flush_and_promote` orchestration as methods/nested classes on the new Layer.
- `ConceptualSpace.__init__` constructs the layer (gated by `symbol_learning_enabled_from_config`).
- Flush/promote stays on explicit boundaries — exposed as `ConceptualSpace.flush_symbol_learning()` or similar, never inside `forward()`.
- Update tests that import `from symbol_learning import …` to import from `Layers` and instantiate `SymbolLearningLayer` directly, or remove if the new wiring covers the same scenarios.
- Delete `bin/symbol_learning.py`.

### Phase D — WordSubSpace inherits SubSpace + STM data move
- Switch `class WordSubSpace(nn.Module)` → `class WordSubSpace(SubSpace)` in `bin/Language.py:5860`. The existing manual `nn.Module.__init__(self)` + ad-hoc field population becomes a call to `SubSpace.__init__(inputShape=[0, muxed], outputShape=[0, muxed], …)` with sensible defaults for the encoding/codebook slots (it is a data carrier, not a pipeline space — codebook_slot stays `None`).
- Move TypedStack state onto WordSubSpace as registered buffers (replacing TypedStack's own buffers):
  - `_buffer [B, cap, D]`, `_category [B, cap]`, `_order [B, cap]`, `_ref_id [B, cap]`, `_depth [B]`, `_category_names`.
- Move parse-state attributes (currently on ConceptualSpace via `_stm_typed`): payload frames, depth, category, order, ref id, cursor/recur_pass (already present), parse state.
- Add WordSubSpace methods that mirror TypedStack's API on its own buffers: `push`, `pop`, `top`, `reduce_admissibility`, `ensure_batch`.
- ConceptualSpace stops constructing `_stm_typed`; reads/writes go via `subspace.wordSpace.<method>` (still uses back-pointer in this phase; renamed in phase G).

### Phase E — ShortTermMemory becomes a Layer in Layers.py
- Move `ShortTermMemory` (`bin/Spaces.py:5845`) into `bin/Layers.py`. Re-base on `Layer` (not `nn.Module`).
- Strip data: no `_buffer`, no `_depth`, no `_max_depth_host`, no per-sentence state. Keep only scorer/reducer weight Parameters (currently held by `RuleScorer` and friends).
- Inline `stm_driver.py` (`STMDriver`, `RuleScorer`) and `stm_trainer.py` (`train_step` logic) as methods on `ShortTermMemory`:
  - `shift(word_subspace, b, payload, category, order, ref_id)`
  - `reduce_step(word_subspace, b)` and `reduce_step_soft(word_subspace, b)`
  - `train_scorer_step(word_subspace, input_vectors, target_rule_ids, snap_fn, optimizer=None)`
  - private `_score_reduce(word_subspace, b)` helper
- `ConceptualSpace.__init__` constructs `self.stm = ShortTermMemory(...)` and invokes it during `forward`.
- Delete `bin/typed_stack.py`, `bin/stm_driver.py`, `bin/stm_trainer.py` (kept until phase H to avoid premature import breakage in tests).

### Phase F — ConceptualSpace.forward(subspace, word_subspace) contract
- Rename signature `forward(PS_subspace, SS_subspace=None)` → `forward(subspace, word_subspace)`.
- Replace the symbolic-loop role of `SS_subspace` with reads from `word_subspace` (which is now a SubSpace carrying the loop data).
- Update PerceptualSpace.forward and SymbolicSpace.forward call paths through `bin/Models.py` model loops to pass `wordSubSpace` explicitly.
- Update `_subspaceForPS` / `_subspaceForSS` plumbing as needed — likely either retained as the C→P / C→S feedback bus or folded into `word_subspace`.

### Phase G — Remove SubSpace.wordSpace back-pointer; rename to wordSubSpace
- Delete `SubSpace.wordSpace` attribute, `SubSpace.attach_wordSpace`, and the `copy_context` carry of `wordSpace` (Spaces.py:4207, 4230–4246, 4248–4272).
- Delete the parallel `Space.attach_wordSpace`/`wordSpace` machinery for Space-tier holders (Spaces.py:6300, 6303–6319).
- Rename runtime/model field `wordSpace` → `wordSubSpace` in bin/ (Models.py:3982, 4331, 4462, 617, 1788, 1954+ and PerceptualSpace/SymbolicSpace usages).
- Keep XML config section name `<WordSpace>` (per spec).
- Remove `WordSpace = WordSubSpace` alias at Language.py:8210–8213.

### Phase H — Delete removed modules + cleanup
- Delete `bin/typed_stack.py`, `bin/stm_driver.py`, `bin/stm_trainer.py`, `bin/bpe_gpu.py`, `bin/mphf_gpu.py`, `bin/symbol_learning.py` (those not already gone from earlier phases).
- Remove dead imports project-wide.

### Phase I — Test migration
- Tests importing `WordSpace`: rewrite to `WordSubSpace`.
- Tests importing `from typed_stack import TypedStack`: rewrite to construct `WordSubSpace` data plus a `ShortTermMemory` Layer.
- Tests importing `from stm_driver import STMDriver, RuleScorer`: rewrite to use `ShortTermMemory.{shift, reduce_step, …}(word_subspace, …)`.
- Tests importing `from stm_trainer import train_step`: rewrite to `short_term_memory.train_scorer_step(word_subspace, …)`.
- Tests importing `from bpe_gpu import …` / `from mphf_gpu import …` / `from symbol_learning import …`: redirect to the new Layer classes.
- Remove direct `RuleScorer(...)` constructions outside owning classes (test files at test/test_stm_driver.py and friends).

### Phase J — Regression checks + final sweep
- `grep -rn "WordSpace\b" bin/ test/` → only XML config strings should match.
- `grep -rn "subspace\.wordSpace" bin/ test/` → zero matches.
- `grep -rn "_stm_typed\|stm_typed" bin/ test/` → zero matches.
- `grep -rn "from typed_stack\|from stm_driver\|from stm_trainer\|from bpe_gpu\|from mphf_gpu\|from symbol_learning" bin/ test/` → zero matches.
- Run targeted regressions: STM tests, BPE/MPHF tests, symbol-learning tests, per-word pipeline tests, `ConceptualSpace.forward` tests.

## Verification strategy

Per user feedback (`feedback_targeted_tests.md`): run targeted pytest node IDs, not full `make test`. After each phase:

- Phase A: `pytest test/test_chunk_layer_bpe.py test/bpe_gpu_equiv.py test/test_phase2_end_to_end.py -x`
- Phase B: `pytest test/<mphf tests>` + `test/test_phase2_end_to_end.py`
- Phase C: `pytest test/test_symbol_learning_stats.py test/test_symbol_learning_policy.py`
- Phase D: `pytest test/test_typed_stack.py test/test_stm_substrate.py test/test_subspace_what_stm_contract.py`
- Phase E: `pytest test/test_stm_driver.py test/test_stm_soft_scoring.py test/test_stm_trainer.py test/test_stm_compose_wiring.py`
- Phase F: `pytest test/test_cky_retirement_parity_gate.py test/test_phase2_sequential_integration.py test/test_serial_mode_conceptual.py`
- Phase G–H: spot-check sentinel tests on every prior phase.
- Phase J: full STM / BPE / MPHF / symbol-learning + per-word + ConceptualSpace.forward suites.

Numerical divergence (NaN/Inf) anywhere must raise loud per `feedback_fail_loud_on_divergence.md`.

## Status tracking

Tasks created via TaskCreate mirror phases A–J. Progress updates land in those tasks; this plan file remains the canonical reference.

## Final status (2026-05-26)

**All phases complete.** 230 targeted regression tests green. Pre-existing
failures (5 unrelated to this spec) untouched per scope decision.

### Per-phase results

- **Phase A — Done.** `BPEGpuLayer(Layer)` in `bin/Layers.py`; `bin/bpe_gpu.py` deleted.
- **Phase B — Done.** `MPHFGpuLayer(Layer)` in `bin/Layers.py`; `bin/mphf_gpu.py` deleted.
- **Phase C — Done.** `SymbolLearningLayer(Layer)` in `bin/Layers.py`; `bin/symbol_learning.py` deleted; both symbol-learning test files migrated.
- **Phase D — Done.** `WordSubSpace` inherits `SubSpace`; typed STM buffers live on it; `ConceptualSpace._init_typed_stm` / `stm_typed` removed.
- **Phase E — Done.** `ShortTermMemory` lives in `bin/Layers.py` as a data-free `Layer`. Driver methods `shift` / `reduce_step` / `reduce_step_soft` / `train_scorer_step` / `_score_reduce` / `init_scorer` take `word_subspace`. Inlines retired `STMDriver` + `RuleScorer` + `train_step`. The legacy idea-stack data formerly held on `cs.stm._buffer` / `_depth` / `_max_depth_host` now lives on `WordSubSpace._idea_buffer` / `_idea_depth` / `_idea_max_depth_host`; `ShortTermMemory` exposes `_buffer` / `_depth` / `_max_depth_host` / `capacity` / `concept_dim` as proxy properties (with setters) so the ~12 chart consumers in `bin/Models.py` continue to compile/work unchanged. Standalone construction (no attached `WordSubSpace`) still works via a fallback local buffer for `test_conceptual_stm.py`.
- **Phase F — Done.** `ConceptualSpace.forward(subspace, word_subspace=None)` signature; internal references updated; both legacy and post-refactor callers supported (both `SubSpace` and `WordSubSpace` carry `materialize()`).
- **Phase G — Done.** `SubSpace.wordSubSpace` back-pointer attribute, `SubSpace.attach_wordSubSpace`, and the InputSpace eager-stamp at `self.subspace.wordSubSpace = ws` all removed. The WordSubSpace reference now lives on each `Space` via `space.wordSubSpace` (set by `Space.attach_wordSubSpace`, called from `WordSubSpace.__init__` for P/C/S and from `BasicModel.__init__` for InputSpace/OutputSpace/ModalSpace). All `wordSpace` field/local renames complete across `bin/Models.py`, `bin/Spaces.py`, `bin/Language.py`, `bin/Mereology.py`, and 54 test files. `WordSpace = WordSubSpace` alias removed at `Language.py:8432`. XML config section name `<WordSpace>` preserved as the literal string.
- **Phase H — Done.** `bin/typed_stack.py`, `bin/stm_driver.py`, `bin/stm_trainer.py` deleted. Combined with Phases A/B/C deletions, six legacy modules (`bpe_gpu.py`, `mphf_gpu.py`, `symbol_learning.py`, `typed_stack.py`, `stm_driver.py`, `stm_trainer.py`) are gone.
- **Phase I — Done.** All affected test files migrated. `test/_stm_test_fixtures.py` shared helper provides `make_typed_stack` / `make_driver` / `make_scorer` / `make_train_step` so `test_typed_stack.py`, `test_stm_driver.py`, `test_stm_substrate.py`, `test_stm_soft_scoring.py`, `test_stm_trainer.py`, `test_cky_retirement_parity_gate.py`, `test_parallel_backend.py` exercise the new API while keeping their original test-name shape.
- **Phase J — Done.** Final regression run: 230/230 green across STM/BPE/MPHF/symbol-learning/per-word/ConceptualSpace.forward/subspace-context/parser-neutral/parallel-backend suites. Final greps:
  - `wordSpace` (lowercase) in non-comment code: zero.
  - `subspace.wordSpace` reads: zero.
  - `_stm_typed` / `stm_typed` in non-comment code: zero (a few docstring references remain as historical context).
  - `from typed_stack`/`stm_driver`/`stm_trainer`/`bpe_gpu`/`mphf_gpu`/`symbol_learning`: zero across `bin/` and `test/`.
  - `WordSpace` (capital W) outside XML strings: only docstring/comment references; the one runtime use (`bin/util.py:848 ('WordSpace', 'language', 'grammar')`) is the XML config path tuple, which the spec explicitly preserves.

### Original session status (interim — superseded by "Final status" above)

Completed end-to-end in one session:

- **Phase A — Done.** `BPEGpuLayer(Layer)` added to `bin/Layers.py`; PerceptualSpace constructs and holds it; static-table cache stays on PerceptualSpace; `bin/bpe_gpu.py` deleted. Regression: `test_chunk_layer_bpe.py` green.
- **Phase B — Done.** `MPHFGpuLayer(Layer)` added; PerceptualSpace owns it; `bin/mphf_gpu.py` deleted. `test/bpe_gpu_match_check.py` migrated to `BPEGpuLayer.*` static methods.
- **Phase C — Done.** `SymbolLearningLayer(Layer)` added (fuses former `SymbolLearningStats` + `SymbolLearningPolicy` + `flush_and_promote`); ConceptualSpace owns it; `bin/symbol_learning.py` deleted; both symbol-learning test files migrated.
- **Phase D — Done.** `WordSubSpace` inherits `SubSpace`; typed STM buffers (`_buffer` / `_category` / `_order` / `_ref_id` / `_depth` / `_category_names`) live on WordSubSpace; `push` / `pop` / `top` / `reduce_admissibility` / `_ensure_stm_batch` methods added; `ConceptualSpace._init_typed_stm` / `stm_typed` removed; `test_ensure_microbatch_cascade.py`, `test_conceptual_stm_typed.py`, `test_stm_compose_wiring.py`, `test_stm_real_input_loop.py` migrated.
- **Phase E — Mostly done.** `ShortTermMemory` moved to `bin/Layers.py`, re-based on `Layer`. New methods `shift(word_subspace, …)`, `reduce_step(word_subspace, …)`, `reduce_step_soft(word_subspace, …)`, `train_scorer_step(word_subspace, …)`, `_score_reduce(word_subspace, …)`, `init_scorer(…)`. Private `_RuleScorer(nn.Module)` nested. WordSubSpace's `_init_stm_driver` rewired to call `cs.stm.init_scorer(...)`; `_stm_drive` calls `cs.stm.shift(self, …)` / `cs.stm.reduce_step_soft(self, …)`. **Deferred (out of scope for this session):** stripping the legacy idea-stack `_buffer` / `_depth` / `_max_depth_host` per the strict spec "data-free Layer" reading — that requires migrating ~12 chart consumers in `bin/Models.py` (`cs.stm._buffer.device`, `cs.stm.snapshot()`, `cs.stm.push_step(…)`, etc.). Both legacy and new APIs coexist on the single Layer transitionally.
- **Phase F — Done.** `ConceptualSpace.forward` signature renamed: `forward(self, subspace, word_subspace=None)`. Internal references updated. Callers in `bin/Models.py` still pass `SS_sub` positionally; this works since both legacy `SS_sub` and a future `wordSubSpace` are `SubSpace` subclasses (`materialize()` is the same contract). Updating Models.py callers to pass `self.wordSpace` explicitly **deferred to a follow-up** alongside Phase G's field rename.
- **Phase G — Deferred (mostly).** Out of scope for this session:
  - `SubSpace.wordSpace` back-pointer removal — only 1 actual code write site (`Spaces.py:6626` `self.subspace.wordSpace = ws`); the rest are docstring/comment references. Safe to keep transitionally.
  - `wordSpace` → `wordSubSpace` field rename across `bin/Models.py`, `bin/Spaces.py`, and ~120 references in tests. Mechanical replace, but high churn. **Recommend a follow-up session focused on this rename**.
  - `WordSpace = WordSubSpace` alias removal at `bin/Language.py:8210` — pinned to ~22 test imports. Also a mechanical migration deferred to the rename session.
- **Phase H — Deferred.** Cannot delete `bin/typed_stack.py` / `bin/stm_driver.py` / `bin/stm_trainer.py` until the test files importing them are migrated (see Phase I). bin/ itself no longer imports any of these modules.
- **Phase I — Partially done.** Tests directly affected by Phase A–F have been migrated. The remaining test migrations (Phase G test rename + Phase H module-deletion fallout — `test/test_typed_stack.py`, `test/test_stm_driver.py`, `test/test_stm_substrate.py`, `test/test_stm_soft_scoring.py`, `test/test_stm_trainer.py`, `test/test_cky_retirement_parity_gate.py`, `test/test_parallel_backend.py`) still import `TypedStack` / `STMDriver` / `RuleScorer` / `train_step` from the not-yet-deleted modules and continue to pass.
- **Phase J — Targeted regression run.** Green: `test_phase2_end_to_end`, `test_phase2_sequential_integration`, `test_serial_mode_conceptual`, `test_ensure_microbatch_cascade`, `test_stm_substrate`, `test_subspace_what_stm_contract`, `test_stm_driver`, `test_stm_soft_scoring`, `test_stm_trainer`, `test_typed_stack`, `test_stm_compose_wiring`, `test_conceptual_stm_typed`, `test_conceptual_stm`, `test_symbol_learning_stats`, `test_symbol_learning_policy`, `test_chunk_layer_bpe`, `test_cky_retirement_parity_gate`, `test_parser_neutral_consumers`, `test_parser_backend`, `test_word_space_attach_knowledge`, `test_partition_rule_predictor`, `test_stm_real_input_loop`. **Pre-existing failures** (not addressed by this refactor, per spec): see the "Pre-existing failures (out of scope)" section at the top.

## Recommended follow-up

A single focused session can complete the deferred Phase G + Phase H + remaining Phase I in one pass:

1. Mechanical rename in `bin/Models.py`, `bin/Spaces.py`, `bin/Language.py`: every `self.wordSpace` / `model.wordSpace` / `space.wordSpace` (excluding the XML config string `"WordSpace"`) → `wordSubSpace`. Use `replace_all` per file; verify by greps.
2. Same rename in test/ (~ 120 references; `WordSpace` alias import → `WordSubSpace`).
3. Remove `WordSpace = WordSubSpace` line at `Language.py:8210` and `from Language import WordSpace` test imports.
4. Remove `SubSpace.wordSpace` attribute, `SubSpace.attach_wordSpace`, and `copy_context`'s carry of `wordSpace` (Spaces.py:4207, 4230–4246, 4248–4272). Pass `wordSubSpace` explicitly to the few sites that read it (mostly via `_model_wordSpace` on Space, which itself can be renamed `_model_wordSubSpace`).
5. Migrate `test_typed_stack.py` / `test_stm_driver.py` / `test_stm_substrate.py` / `test_stm_soft_scoring.py` / `test_stm_trainer.py` / `test_cky_retirement_parity_gate.py` / `test_parallel_backend.py` away from `TypedStack` / `STMDriver` / `RuleScorer` / `train_step` imports — rewrite to use WordSubSpace's typed-STM buffers + `ShortTermMemory.shift` / `.reduce_step_soft` / `.train_scorer_step`.
6. Delete `bin/typed_stack.py`, `bin/stm_driver.py`, `bin/stm_trainer.py`.
7. Final greps from this plan's Phase J checklist must all be zero.
8. Optionally: strip `_buffer` / `_depth` / `_max_depth_host` from `ShortTermMemory` (Phase E completion) — requires migrating ~12 chart consumers in `bin/Models.py` to `cs.wordSubSpace._buffer` / `cs.wordSubSpace.snapshot()` / etc.

