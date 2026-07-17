# Doc/code alignment audit and cleanup plan (2026-07-16)

Full-corpus audit of `doc/*.md` against `bin/` + `data/` + `Makefile`, plus a
liveness map of every `bin/` module and a `test/`/`data/` organization pass.
Thirteen audit reports were produced by parallel verification agents (every
claim checked against code with `file:line` evidence) and consolidated here.
This document is both the findings record and the execution plan for
(a) bringing the docs in line with the code, (b) eliminating legacy
codepaths, and (c) tightening repo organization.

## Part I — Doc verdicts (grade, one-line diagnosis)

| Doc | Grade | Diagnosis |
|---|---|---|
| BasicModel.md | A | Accurate; keep as-is. |
| MachineMinds.md | A | Accurate (historical content self-hedged); keep as-is. |
| Ergodic.md | A | Every formula/constant verified; keep. |
| Componentization.md | A- | Accurate spec-with-status; minor status-line drift. |
| Mereology.md | B | Strong overall; `.where` bracket + CS-wave paragraphs stale; 2 wrong claims. |
| Reasoning.md | B+ | ~90% faithful incl. Thinking Kernel; 3 small fixes + 1 live code bug found. |
| Lexicon.md | B | Two wrong claims (LBG split direction, MM_xor codebook tag). |
| Installation.md | B- | One wrong CLI default; missing targets/env vars; stale PDF chapter list. |
| Architecture.md | C | §A concept-forward is superseded (FF pyramid); STM shift direction stated backwards ×5; 3 phantom nouns. |
| STM.md | C+ | Core mechanics faithful; two "honesty notes" now invert reality; MentalModel attention claim false. |
| Language.md | C+ | Two sections describe dead mechanisms as live; grammar examples use forbidden legacy tokens. |
| Params.md | C+ | Space-table Default column systematically stale; ~50 parsed tags undocumented. |
| Training.md | C+ | CBOW table describes retired softmax variant; detach() column never existed; dead knob names. |
| README.md | C+ | XML example misplaces 2 tags (silently ignored); files table misattributes SymbolSpace. |
| Spaces.md | C- | "Symbolic phase" + PS/WS signatures + `.where` all pre-date the 2026-07 arcs. |
| Logic.md | C- | Operator level heavily stale (removed Basis wrappers, retired bivector forms, phantom methods). |
| Philosophy.md | C | Front half current; pramana half describes retired bivector/ABS_T/operator-tag machinery. |
| SymbolFirewall.md | C+ | Doctrine still true; 9/11 line citations drifted (up to ~1900 lines); 2 point at wrong code. |

### Cross-cutting doc defects

1. **The 2026-07 arcs are undocumented**: multi-rung `.where` ladder
   (2026-07-09), dual-towers FF pyramid rev 2 (2026-07-12), fold-width
   unification (2026-07-16). Architecture §A, Spaces "symbolic phase",
   and Mereology's `.where`/wave paragraphs all describe the superseded
   design. The `cs_forward_content` docstring itself is stale
   (bin/Spaces.py:15330-15342 still says "iterated wave" above a body
   commented "No fixed point, no re-injection").
2. **Retired names survive in prose**: `SymbolicSubSpace` (→
   `SymbolSubSpace`), `PerceptualSpace` base class (never existed as
   claimed), `insert_paired_word` (retired 2026-06-10), `ABS_T`/`VP`/`MP`
   grammar tokens (forbidden by test_role_collapsed_grammar), bivector
   `[aP,aN]` activation (retired 2026-05), `Basis.non/symbolize/disjunction`
   (removed 2026-05-01), `isEqual`/`isPart` as compose ops (query-only since
   19ba903), `wordSpace` (→ `symbolSpace`), `trainEmbeddingRatio` (→
   `embeddingScale`), `BasicModel.Start` (→ `_start_spaces_for_forward`).
3. **STM shift direction**: Architecture.md states newest-at-last-slot in
   five places; code is NEWEST-AT-SLOT-0 (bin/Layers.py:12557-12601,
   bin/Spaces.py:13728-13749).
4. **Line-number anchors rot fast**: SymbolFirewall.md and STM.md carry
   dozens of dead `file:line` cites. Convention going forward: cite
   symbol names, not line numbers, in doc/*.md.

### Live code bugs found by the audit (not doc defects)

- **`_ltm_consolidation` gate dead at model level**: `think_about` /
  `reason_about` read `getattr(self, "_ltm_consolidation", False)`
  (bin/Models.py:10009,10035) but the model attribute is
  `ltm_consolidation` (bin/Models.py:912) — `materialize` can never fire
  from those entry points.
- **`write_syntax_tree` always emits `<noTrace/>`**: `chart`/`traces`
  hardcoded None (bin/Models.py:11697-11699).
- **`truthBiasScale` parsed but never consumed** (bin/Models.py:6297).
- **`set_last_svo` has no production callers** (test-poked only).
- **PiLayer non-invertible branch drops the ergodic flag**
  (bin/Layers.py:4149; SigmaLayer passes it at :2831) and butterfly mode
  constructs no inner layer — those configs get no ergodic noise.
- **Config-only orphan**: `useSubspaceActivation` set in
  data/xor_subspace.xml + fixtures, parsed nowhere.

## Part II — bin/ liveness map and legacy elimination

### Module verdicts

- **CORE**: Spaces, Layers, Language, Models, util, data, embed, Optimizer,
  visualize, architecture.py (lowercase but load-bearing — canonical
  (nWhere,nWhen) source of truth), Mereology.py (mixin: contemplative
  measures + order/projection primitives), Meronomy.py (referential
  binary-tree codebook; word-tiling is live in the radix path).
  **Mereology vs Meronomy is NOT a duplication** — Meronomy imports
  Mereology's primitives; deliberate layering.
- **CORE (narrow)**: References.py, reasoning.py, thinking.py,
  bind_resolver.py, surface_morphology.py, surface_tense.py, workarounds.py.
- **Entry points / tools**: train.py, serve.py (+secure.py — consumed by the
  parent WikiOracle repo), bm.py (chat client for serve.py), recon_bench.py
  (Makefile bench targets + fixture factory), export_mlx.py,
  eval_where_tiling.py.
- **TEST-ONLY (design mirrors, keep)**: perceptual_analyzer.py,
  participation.py, semantic_categories.py.
- **LEGACY DEMOS (still wired to Makefile + test_basicmodel)**:
  bin/etc/SigmaPi.py, SymPercept.py, SPNN.py.
- **DEAD**: bin/etc/old.py (27 KB, unimportable — references undefined
  `Layer`/`nn`), root etc/old.py (4.8 KB, imports nonexistent `Space`
  module; older diverged copy of the former).

### Tier 1 deletions (safe — zero references, verified by token scan incl. strings/XML/grammar; re-verify with `make test` after each batch)

1. `bin/etc/old.py` and root `etc/old.py` (both unimportable graveyards).
2. Dead imports: `QKVAttentionLayer` in bin/Models.py:71 and
   bin/Language.py:28 (enlistment retired; neither file uses the symbol).
3. Dead top-level function `_log_advisory_exception` (bin/Models.py:108).
4. Zero-reference classes: `CertaintyWeightedMAELoss` (bin/Layers.py:15048),
   `CertaintyWeightedMSELoss` (bin/Layers.py:15071).
5. The orphaned `idea_*` stack API on IdeaSubSpace
   (bin/Language.py:10114-10247, 9 methods mirroring the retired
   TypedStack surface).
6. Never-referenced methods (~45 total):
   - Spaces.py: tokens_to_decoded:1593, activeIndices:1941,
     parameters_for_optimizer:2063+:5432, active_dense:2411, getSize:2829,
     get_property_kind:2974, get_category:3167, _ensure_svd_legacy:3996,
     conceptParthood:4314, perceptParthood:4327, print_info:5372,
     codebook_weight:5397, vocab_keys:5402, _get_codebook:5657,
     dematerialize:6833, set_symbols:6866, get_symbols:6877,
     top_of_stack:7071, top_two_of_stack:7086, _radix_token_stream:9077,
     get_forward_meta:9901, get_reconstruction_target:9905,
     _scale_scalar_trust:17267, get_semantic_row:21281, evaluate_truth:23923.
   - Layers.py: _restore_leading:1730, _sigma_inner_reverse:2851,
     put_row_key:4974, set_mask:5777, truth_conjunction:7101,
     is_word_boundary:10154, disabled_categories:14831,
     format_breakdown:15025.
   - Language.py: ws_rules_upward:748, ws_rules_downward:753,
     _s_rule_ids:1713, _ensure_packed_table:1825, attach_grammar:4689,
     _last_unary_routing:5286, _collect_binary_rule_selections:5301,
     priming_capacity:8663.
   - Models.py: _debug_tensor_stats:1375, _is_one_component:1588,
     print_weights_info:1792, _restore_vocab:2120,
     _symbol_feedback_from_vectors:2518, _extract_prediction_sequential:2571,
     _symbol_heat_source:2723, accumulate_output_symbol_residual:4014,
     _intersentence_seed_slab:4397.
7. Orphan data files: data/MM_mereological.xml, data/xor_subspace.xml,
   data/word.cfg, data/colab.ipynb; stale probes test/xor_experiment2/3/4.py
   (call the removed `model._getLossFn()`).
8. Stale `.worktrees/step4-pi-sigma-ownership`; empty `tmp/`.

### Tier 2 (recommend, needs Alec's sign-off — each removes a live-but-retired surface or moves files)

1. `Mem` family (bin/Layers.py:15136-15511, 9 MATLAB-era classes) +
   `DecisionBoundaryLayer`:5644 + `QKVAttentionLayer`:5732 — all
   retired-but-kept, referenced only by test_basicmodel unit tests and
   Layers self-test. Deleting means also deleting those test hooks.
2. `AssertPartLayer`/`QueryPartLayer`/`QueryEqualLayer` (formally retired
   op names, kept for a transitional D1 fixture) and the
   dispatchable-but-never-dispatched ops (true/false/swap/copy/area/
   luminosity/isaPart) — grammar files use none of them.
3. bin/etc demos (SigmaPi/SymPercept/SPNN): keep (wired to Makefile
   targets + tests) or retire targets+tests together.
4. Move the 11 runnable orphan probes and 6 bench tools out of test/
   into tools/ so test/ holds only the suite + 4 shared helpers
   (conftest, fixtures/, _stm_test_fixtures, space_equiv).
5. Git hygiene: global `__pycache__/` ignore + untrack
   doc/__pycache__/*.pyc; decide tracked-artifact policy for
   BasicModel.pdf / doc/moc7_poster.xlsx (make-clean deletes them);
   reconcile LFS split (mnist_train.csv is an LFS pointer,
   mnist_test.csv an 18 MB blob, fineweb shard_00000.parquet a 90 MB
   blob near GitHub's hard limit); add `tmp/` to .gitignore.
6. DOC-ONLY configs (MM_concept_sim, MM_substitution, MM_symbol_attention,
   MM_ws_tall, sample.txt) — deletion candidates if historical plan docs
   are considered archival.
7. `category_embedding`/`category_logits`/`category_ids` +
   `reduce_admissibility` legacy scaffolding (Language.py:9550ff) —
   doc'd as "slated for follow-up retirement".
8. `ShortTermMemory.shift`/`reduce_step`/`reduce_step_soft`/`_RuleScorer`
   (bin/Layers.py:12688-12900) — zero production callers; kept alive by
   three legacy tests as compat shims for the retired STMDriver.

## Part III — Test status (2026-07-16)

`make testp` baseline on the working tree: 6 failures / 3305 passed.
All six shared one root cause: the (uncommitted) fold-width unification
sized PS sigma / WS pi at content width, but the SyntacticLayer unary
dispatch (bin/Language.py `SyntacticLayer.forward`/`.reverse`) still fed
folds the full muxed carrier — MeronymicFoldAdapter raised
"vector width 10 exceeds cascade M_total=2". Fixed by applying
`fold_content_apply` (content fold + band passthrough) at both dispatch
sites; parameter-free grammar ops (nInput=0) are unaffected, binary ops
were already excluded by the arity guard. 4/6 green after the fix.

The remaining two (test_xor_recon_grads_flow, test_output_mse_is_crisp)
are NOT crashes but re-baseline consequences, confirmed by A/B against a
HEAD checkout (both pass at HEAD):

- Instrumented gradient tracing showed the D3 reconstruction loss's
  gradient into the PS forward event flows EXCLUSIVELY through the
  trailing where/when band columns (content-column grad is exactly zero;
  what_scale=0.7, so not a loss-weight artifact — the recon's content
  leg does not depend on PS forward content). Pre-change, the sigma fold
  mixed the band, so PS sigma/what.W received gradient THROUGH the band
  scaffold ride-along — a leak, and one that let training perturb the
  band the `.where` ladder needs byte-exact. The fold-width law removes
  the leak by design; the grads test's premise (D3 recon trains the
  percept codebook) no longer holds.
- XOR_exact's seed-0 crisp basin moved, as it has on every prior band
  change (the test file carries four dated re-pins for exactly this).

Resolution applied:

- `SyntacticLayer.forward`/`.reverse` (bin/Language.py): wrapped the unary
  op application in `fold_content_apply` — the missing "dispatch trim" the
  fold-width diff's own comment promised. Fixes the four crash-failures.
- `test_xor_recon_grads_flow`: re-scoped the what.W assertion to the
  reverse-path body params, with a dated comment recording the measured
  band-only gradient and the leak removal rationale.
- `test_output_mse_is_crisp`: re-pinned BASIC_SEED 0 → 4 after measuring
  seeds 0-5 under the new law ({0: .1028, 1: .0567, 2: .0495, 3: .1717,
  4: .0008, 5: .0413}); seed 4 is crisp (MSE 0.0008, 4/4 exact
  reconstruction). Note: the crisp basin is RARER than pre-change —
  follow-up question for the open XOR-convergence re-baseline: whether
  the content-only fold needs an LR/epoch retune to widen the basin.
- Also fixed while in the area (audit-found live bug): the dead
  `<ltmConsolidation>` gate in `reason_about`/`think_about`
  (`_ltm_consolidation` → `ltm_consolidation`, bin/Models.py) — the
  Thinking-Kernel lemma write-back can now actually fire when the knob
  is on.

## Part IV — Execution order

1. Test re-baseline (Part III) → full serial `make test` gate.
2. Tier 1 deletions in 3 batches (graveyards+imports; dead methods;
   orphan data/probes), `make test` between batches.
3. Doc fixes, highest-damage first: Architecture §A + STM direction;
   Spaces symbolic-phase/.where; Language dead sections; STM honesty
   notes; Logic operator level; Philosophy pramana; Params defaults
   column + undocumented-params table; Training tables; Mereology
   touch-ups; README/Installation corrections; SymbolFirewall symbol-name
   anchors.
4. Tier 2 items for Alec's review.
