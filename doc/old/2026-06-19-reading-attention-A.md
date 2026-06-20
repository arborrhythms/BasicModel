# 2026-06-19 — Attention workstream: reading (A) + global (B) LANDED

**Everything below is UNCOMMITTED** (Alec commits; the basicmodel submodule is
still at its prior pointer with last session's changes + this session's on top).
Builds on the 2026-06-19 handoff (pending #2–#5 already landed: MM_20M sO=3
combine fix, the cleanup batch, the `<mereologyRaise>` handoff, the orders spec +
`syntacticOrder`).

## (B) stochastic element + typed addressable global attention — LANDED

Per Alec's scope ("do the stochastic piece, then stop; `.where` accesses input
window / STM / LTM / Symbolic codebook; then commit"; book paging = decide-later).

- **Stochastic element (no new flag).** `ReadingAttention.superposition_scale(t)
  = 1 − clamp(t)` scales the **preference logits before the −1e9 mask** in both
  `ReadingAttention` and `GlobalAttention` (coverage survives any temperature).
  `t=0`/None → `×1.0` → byte-identical; pass B `exploreTemperature` → flatter →
  exploration. `_set_superposition_temperature` now also stashes
  `self._superposition_temperature`; `_reading_attention_step` /
  `_global_attention_step` read it. Reuses the existing two-pass `<learning>`.
- **`Spaces.GlobalAttention`, `<globalAttention>`.** Free attention over a typed
  registry (`Models._addressable_spaces`): INPUT window (per-batch `[B,K,D]`
  span keys), STM (`stm.snapshot`), LTM (`symbolicSpace.ltm_store.slots[:count]`
  pooled, SHARED `[M,D]`), CODEBOOK (WS-then-PS `getW()`, SHARED `[V,D]`,
  `boosts = intent_priming_weights`). One temperature-scaled softmax across ALL
  candidates (no monotonic mask) → typed `.where` (`space_id`+`[start,end]`) +
  soft-read `Σ αₖ·keyₖ`. **SHARED stores matmul'd (`α@keys`, `qn@kn.t()`),
  never broadcast to `[B,M,D]`** (65 536-row codebook = `[B,V]·[V,D]`). Result
  parked on `_global_attention_obs`; soft-read **not fed back** (dark → forward
  byte-identical with the flag on). Both helpers `@torch.compiler.disable`'d.
  Gradient stops at the keys (detached); scorer + `space_bias` train.
- Files: `bin/Spaces.py` (GlobalAttention + the temperature in ReadingAttention),
  `bin/Models.py` (`_addressable_spaces`, `_global_attention_step`, the t>0
  wiring, getOptimizer loop over both modules, the temperature stash),
  `data/model.xsd` (`<globalAttention>`), `data/MM_global.xml` (new),
  `test/test_global_attention.py` (new, 14), spec §10 + flags + §7 checklist.
- **Verified:** full suite **2711 passed / 0 failed** (= 2697 + 14). Config
  sweep builds MM_global + MM_reading; compile-sensitive tests green (host
  helpers `@torch.compiler.disable`'d). **Adversarial 4-lens review: clean** —
  gradient boundary detached, byte-identical-off, memory-safe (codebook never
  `[B,V,D]`), mask-before-temperature, optimizer registration all confirmed; no
  defects.
- **DEFERRED (next):** the global-attention **consumer** (feed the soft-read
  back into the forward + train by the downstream task error) and **out-of-core
  book paging** (the coarse-index representation — codebook-as-index vs separate
  RAG index — is "decide later"). One mechanism: reading = monotonic sweep,
  find = content-addressed jump, introspect = codebook/LTM; the monotonic mask
  on = read, off = search. (Lit anchor: NTM/DNC content+location addressing,
  DRAW location+scale read head = the endpoint-sum `.where`, RAG/RETRO paging.)

---

## What landed (verified)

**Deliverable (A) of `doc/specs/reading-attention.md` — the learned `.where`
producer behind `<readingAttention>`, dark by default → byte-identical.** It is
the producer of the `_passback_scope_where` the `<mereologyRaise>` handoff
already consumes; trained by the text-mode next-word CE loss.

Files touched: `bin/Spaces.py`, `bin/Models.py`, `data/model.xsd`,
`data/MM_reading.xml` (new), `test/test_reading_attention.py` (new),
`doc/specs/reading-attention.md` (§9 + flag table).

- **`Spaces.ReadingAttention(nn.Module)`** — scores the staged analysis spans
  (`_staged_analysis_spans` `[B,K,2]` word brackets) from a query = pooled prior
  concept (`prevCS_forSS`, subsymbolic) + pooled STM symbols (`cs.stm.snapshot`,
  symbolic), via a small MLP over six DETACHED features + a non-learned
  shift-bootstrap bias. Feature[0] (subsymbolic) is the **codebook-retrieval
  prior** — the literal `intent_boosts` path:
  `maxᵥ(cos(span, rowᵥ)·boostᵥ)` over the PartSpace percept codebook
  (`intent_priming_weights` + the `(sim·boosts).amax` reduction
  `WholeSpace._topk_priming_mask` uses; `boostᵥ` = the tower's primed-intent
  state if set, else derived from the concept), with a **concept-content cosine
  fallback** when the space carries no codebook. Feature[1] = `cos(symbols,
  span)`; then `start/N`, `end/N`, `extent/N`, signed cursor distance. Zero-init
  readout head ⇒ at init the argmax IS `read_idx = t−1` (the serial for-loop);
  the CE only refines it. Monotonic/coverage mask (consumed `k<read_idx` + pad
  extent 0 → −∞). `next_where = Σ αₖ·spanₖ` normalized. Defense-in-depth detach
  in `_cos` and `_codebook_retrieval_prior` (the codebook rows `W` have
  `requires_grad=True` — they are detached before the prior touches them).
- **`BasicModel._reading_attention_step`** — runs the producer at each `t>0`
  pass in `_forward_body` (parallel path) BEFORE `_passback_scope_ps`; writes
  `wholeSpaces[0]._passback_scope_where` (TEACHER span in training, predicted
  soft at eval; `None` past the last word) and, in text mode
  (`model_type=="embedding"`, training), adds the next-word CE
  (`reading_attention`, category `symbol`) to `CS_sub.errors`. The
  copy_context-shared pipeline Error carries it into `totalLoss`.
- **Flag** `<readingAttention>` (default false): `model.xsd` element +
  `create_from_config` read (`self.reading_attention_enabled`); the module is
  built only when on (`self.reading_attention`, else `None`). Its readout params
  are added to the optimizer in `getOptimizer` (the `self.spaces` walk misses a
  model-level module).
- **Gradient boundary** — every score input detached ⇒ the loss trains the MLP
  readout ONLY, never the EMA-only VQ codebooks (C-9/C-11) nor the primed
  symbols. Verified by `test_gradient_stops_at_primed_symbols`.
- **Config** `data/MM_reading.xml` (from `MM_mereology`: adds
  `<readingAttention>true` + WholeSpace `<analysis>word`).

## Verification

- Full suite **2696 passed / 0 failed** on CPU + the known unseeded flaky-tail
  `test_output_mse_is_crisp` (XOR MSE CLI gate) which **passes on retry** (NOT a
  regression — XOR_exact is byte-identical, reading attention is `None` for it;
  do NOT re-pin a seed). = the 2676 baseline + 21 new reading-attention tests
  (incl. 4 codebook-retrieval-prior tests) − the flaky's one-run flip.
- Config-load sweep (`test_modality_configs.py`) **42 configs** build incl.
  MM_reading. All `test_*compile*` / `test_brick_no_sync` / capture-gate tests
  green (the compile guard is inert for default configs).
- The 2 `test_mlx_export` lowering tests **stay xfailed** (MM_20M byte-identical
  — reading attention is `None` for it). `symbolicOrder` untouched. XOR MSE gate
  not modified.

## Adversarial review (workflow, 4 lenses + verify)

Gradient-boundary lens: **fully clean** (every score input detached; the
codebook rows `W` — `requires_grad=True` — are detached before the prior).
Correctness lens: **sound** (width slicing, V-conformance, amax reduction,
NaN/inf guards). Byte-identical-off lens: **passed**. Two medium findings,
adjudicated:
1. *Width slice "silent truncation"* — **false positive**: the min-slice is
   content-vs-content by construction (the key's `.where`/`.when` tail is
   intentionally dropped; W is content-only). Documented with a clarifying
   comment; no assertion added (it would false-fail on the legitimate 1024-vs-
   1020 width).
2. *Compile guard* — **fixed**: added `@torch.compiler.disable` to
   `_reading_attention_step` (matching `_ss_compose_eager`/`_ss_generate_eager`),
   so a compiled config with reading attention on graph-breaks cleanly instead
   of failing inside `materialize()`. Inert for default configs (verified).

## Caveats / decisions

- **Eager-only:** the producer materializes + reads shapes host-side (like the
  gated handoff it feeds), so a compiled config with `<readingAttention>` on
  would graph-break. MM_reading runs eager. Default-off keeps compiled configs
  byte-identical.
- **Checkpoint:** `<readingAttention>` on adds `reading_attention.scorer.*`
  state_dict keys (new config only); existing checkpoints unaffected.
- **Remaining refinement (hook, not deviation):** the **symbolic** term rides
  the STM-symbol query rather than materializing the CS symbol table as separate
  relation/co-occurrence keys; the table-as-keys + relation-store bias is the
  natural bridge to (B). *(The subsymbolic term was completed to the literal
  `intent_boosts`/`selection_boost_fn` codebook-retrieval prior — Alec's "run to
  completion", 2026-06-19; PartSpace percept codebook, cosine fallback for
  codebook-less spaces.)*

## Next (per reading-attention.md)

- **(B) global attention** — needs the stochastic element first (the
  `exploreTemperature` two-pass superposition extended to the codebook/attention
  selection); reuses the §9 machinery + a temperature-softmax over the
  selection, trained by the downstream task error.
- **(C) idea decode** — the parse-tree-deleted reverse.
