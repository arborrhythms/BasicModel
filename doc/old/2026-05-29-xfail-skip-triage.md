# XFAIL / SKIP triage + legacy cleanup (2026-05-29)

## Context

After the recent chart-parser → routing-grammar reorg (commits 5fb08a9,
b3ea539, 41d44d9) and the parse-state / PerceptStore split (commit
2b63aaf, "delete legacy parser files"), the test suite carried ~76
xfail / skip / expectedFailure markers across ~30 files. Many were
written speculatively against architecture moves that have since
landed or that the code has drifted past. Goal: get to an accurate
marker set — every remaining xfail/skip carries a *currently true*
reason — and excise legacy code that the new architecture left
behind.

## What changed

### Tests removed (legacy contracts that no longer exist)

- `test/test_xor_grammar_gradient_flow.py` — **deleted**. The test
  exercised `ws.chart._pair_feature_dim` / `_rule_bias`; none of those
  symbols exist post-reorg, and `bin/Pipeline.py::ChartCompose` is
  gone. Old chart pipeline is retired.
- `test/test_partition_symbolicspace_state.py` — **2 functions
  deleted** plus their helper:
  - `test_forward_updates_self_subspace_from_incoming`
  - `test_multiple_forwards_accumulate_until_percepts_exhausted`
  - `_make_integrated_system` helper (only consumed by the two above)
  These tested the legacy `setW`-shadow contract that Stage 4 of
  `2026-05-21-active-payload-retirement.md` retired. File now contains
  just `test_chunk_static_modes_smoke`.
- `test/test_streaming_ar_training.py::test_arlm_runbatch_trains_without_reverse`
  — **deleted**. It monkey-patched `model.reverse` to count calls; the
  `reverse()` method no longer exists on `BasicModel`, so the test's
  premise (`reverse not called`) is structurally enforced.

### Markers updated, not removed

- `test/test_streaming_ar_training.py::test_arlm_forward_returns_predictions_list_and_no_reconstruction`
  — re-skipped with an updated reason. Original block (serial-mode
  refactor pending) is gone, but the test now fails on a fresh fixture
  staleness: `shape '[1, -1, 100]' is invalid for input of size 104`
  at `bin/Spaces.py:6459`. Same SymbolicSpace forwardEnd reshape
  mismatch hits 3 tests total (see "Fixture staleness" below).

### Markers kept as-is (still accurate)

All ~70 remaining markers were verified to still describe a real,
current failure. See "Surviving marker clusters" for the breakdown.

### Code / doc cleanup

- `bin/Language.py` lines 33-37 — **deleted duplicate import block**.
  Lines 26-28 already imported `Layer, PiLayer, SigmaLayer,
  LinearLayer, AttentionLayer, CertaintyWeightedCrossEntropy, Loss,
  ModelLoss, epsilon`. Line 37 also imported `Error, TheError` which
  grep confirms are never used in the file body — dropped them too.
- `bin/Models.py` ~line 6524 — **removed stale ParseState retirement
  comment** in `write_syntax_tree`. The comment explained why a code
  path no longer exists; per the project comment-policy memory, that
  belongs in commit history, not the file.
- `doc/Language.md` — **fixed line 5-6 module list** (dropped
  references to deleted `typed_stack.py`, `stm_driver.py`,
  `parse_state.py`); **deleted ParseState section** (lines 158-177
  documented an API that no longer exists); **trimmed Syntax And SVO
  section** to remove ParseState references and just point readers at
  the `BasicModel.write_syntax_tree` docstring.

### Code / doc deliberately kept

Per user direction:

- All of `bin/etc/` (old.py, SPNN.py, SigmaPi.py, SymPercept.py) and
  their associated test classes — kept as pedagogical references.
- `doc/Params.md` line 223 PerceptStore reference — checked, already
  accurate (says "relocated 2026-05-29 from `bin/PerceptStore.py`",
  which is a correct historical note).

## Surviving marker clusters (still failing — explanations)

These all carry currently-true reasons after triage. None should be
silently stripped; each is blocked on real work.

| Cluster | Tests | Root cause |
|---|---|---|
| **MM_5M dim mismatch** (12 → 1028 at `bin/Spaces.py:10255`) | `test_space_equiv_selfcheck.py::test_identity_candidate_passes`, `test_per_word_ss_padding_noop.py::test_stm_depth_tracks_valid_len_not_N`, 3× `test_per_word_capture_gate.py`, 2× `test_rule_gate_isolates_side_effects.py`, `test_compile_static_loop.py::test_per_word_body_callable_with_static_signature` (8) | `_stm_shift_and_push` writes a percept-dim `idea` vector into a concept-dim STM buffer. The docstring at `bin/Spaces.py:10229` explicitly still says "signal-router-based grammar dispatch (Stage 3, not wired yet)". Despite the user's recollection that the router is wired, this code path doesn't lift percept → concept dims before pushing. |
| **ProjectionBasis convergence regression (2026-05-13)** | 3× `test_explicit_dimensions.py` (XOR_recon CLI, XOR_exact CLI, XOR_grammar pinned-seed) + `TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct_with_pinned_seed` | The CLI XOR runs fail to reconstruct ≥50% of inputs even with pinned seed. Marker text still accurate; no progress since 2026-05-13. |
| **Fixture staleness (SS width 104 vs OS reshape target 100)** | `test_streaming_ar_training.py::test_arlm_forward_returns_predictions_list_and_no_reconstruction`, `test_reasoning.py::TestWriteMask::test_partition_isolation`, `test_hierarchical.py::TestBackwardCompat::test_mentalmodel_unchanged` (3) | `MentalModel.xml` / `RamsifiedModel.xml` have SS dims that mismatch the current SymbolicSpace.forwardEnd reshape. Likely fixable by an XML refresh; not attempted in this pass. |
| **Bare-PiLayer C-tier convergence pending** | `test_mm_boolean.py::TestMMBoolean::test_explicit_test_sentences`, `test_implicit_classification` (and `test_encode_decode_by_best_fit` if you count an unmarked dep) | Encode-decode round trip collapses to a single class ("hits=['A'], misses=['A','A','A']"). Awaiting "explicit wrapper" landing. |
| **Trained LM_5M artifacts absent** | 6× `test_reasoning.py` English-level tests (`test_syllogism_*`, `test_inconsistency_detected`, `test_semantic_equivalence_paraphrase`, `test_semantic_nonequivalence`, `test_extrapolate_transitive`); `test_svo_end_to_end.py::test_svo_extraction_on_real_sentence`; `test_universality.py` cluster (7+) | `data/LM_5M.ckpt` and `data/LM_5M.kv` don't exist. Markers use `run=False` so they don't even attempt; appropriate. |
| **Task 2.4 `SymbolicSpace.inside()` unimplemented** | `test_partition_resolve.py::test_inside_of_parthood_matches_part_primitive`, `test_outside_is_negation_of_inside` (2) | Skip marker accurate; feature still pending. |
| **Slow / opt-in (RUN_SLOW=1)** | `test_stream_smoke.py`, `test_universality.py::...`, others (~4) | Correct guards, intentional. |
| **Runtime conditional inline skips** | ~26 inline `pytest.skip()` calls across many files | Guard on missing GPU, `torch._dynamo` not importable, padding columns absent, etc. These are correct defensive skips, not bugs. |

## Remaining work (not done in this pass)

1. **Refresh stale XML fixtures** — `data/MentalModel.xml` and
   `data/RamsifiedModel.xml` carry SS widths (104) that don't match
   the current SymbolicSpace.forwardEnd reshape target (100).
   Refreshing the XML would unblock 3 expectedFailure tests in one
   sweep.
2. **Investigate the MM_5M signal-router gap** — the xfail reason
   text and the source comment at `bin/Spaces.py:10229` both say the
   signal router isn't wired into the per-word STM path. The user's
   recollection that it IS wired suggests either a code path other
   than `_stm_shift_and_push` was wired, or that wiring exists but
   doesn't lift percept→concept dims. Worth a 30-minute pass through
   `_per_word_body_step` + the router to confirm.
3. **Pursue ProjectionBasis convergence regression** — three CLI
   XOR tests have been failing since 2026-05-13. Either dig into the
   regression or accept it as expected for the current basis design.
4. **No-op test cleanups** — `test_streaming_ar_training.py` has two
   functions that just `return  # retired` (`test_arir_requires_reconstruct_not_none`
   line 46, `test_basicmodel_arlm_runbatch_uses_streaming_predictions`
   line 191). Harmless but dead.

## Files modified

```
M  bin/Language.py           (dropped duplicate imports L33-37)
M  bin/Models.py             (dropped stale comment in write_syntax_tree)
M  doc/Language.md           (module list fix; deleted ParseState section)
M  test/test_partition_symbolicspace_state.py  (rewrote: 2 tests deleted, helper deleted, docstring updated)
M  test/test_streaming_ar_training.py          (1 test deleted, 1 skip reason updated)
D  test/test_xor_grammar_gradient_flow.py      (entire file)
```

## Verification

- `PYTHONPATH=bin .venv/bin/python -c "import Language; import Models; import Spaces"` → OK
- `pytest test/test_partition_symbolicspace_state.py test/test_streaming_ar_training.py` → 8 passed, 1 skipped (the re-skipped streaming_ar test)
- `pytest --runxfail` on Pass A cluster (8 tests) → all 8 still FAIL with documented dim mismatch; markers retained
- `pytest --runxfail` on Pass C cluster (8 tests) → all still fail with documented reasons; markers retained
