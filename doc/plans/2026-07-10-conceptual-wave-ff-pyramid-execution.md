# Symmetric Dual Towers + Callosum-Fed Pyramid — Execution Plan (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans
> (INLINE — tasks share bin/Spaces.py / bin/Models.py and are strictly
> sequential). Checkbox (`- [ ]`) steps. **Alec does ALL git writes** — no
> commit steps; each GATE is where he commits. Design (APPROVED, rev 2):
> [2026-07-10-conceptual-wave-ff-pyramid-design.md](2026-07-10-conceptual-wave-ff-pyramid-design.md).
> Rev 2 supersedes rev 1 in place (rev 1's nVectors-taper Tasks 1-6 are VOID;
> the rev-1 Task-1 HEAD baselines are preserved in EXECUTION NOTES below).

**Goal:** PS/WS become symmetric duals (`forward(in_sub, CS_out)`, no
`PerceptualSpace` class, InputSpace to both); the symbolic phase feeds from
the callosum-glued 8-tile frame instead of the rectified snap readout; the
concept pyramid is edge-structured $\sigma$ composition with top-K `.index`
attention ($8\to4\to2\to1$) — mechanism-live on `MM_sparse_concept`, byte-
identical off-path.

**Architecture:** see design §§1-5. The pump-loop call sites
(bin/Models.py:6813, 6862) adopt the dual signature; routing (WS
universe-primary) engages on the parallel path only until the serial
CHECKPOINT; the P3 cutover (bin/Models.py:7067) reads the glued frame; the
pyramid selects per-order top-K by setting subspace `.index`
(bin/Spaces.py:6556) so generic `materialize()` (no `mode=`) pulls the
selected codes.

**House rules for every task:** failing test FIRST (targeted pytest,
`PYTHONPATH=test:bin .venv/bin/python -m pytest ...`); no git writes;
one-liner comments; Inf/NaN + overflow fail loud; cpu/eager seeded runs for
the fidelity diagnostics; `make test` only at the final gate on a quiet
tree; probes in scratchpad. Off-path guard after EVERY task:
`test_reconstruction_roundtrip.py` fast tier green.

---

## Task A — Symmetric towers (class removal + dual signatures + IS-to-both)

**Files:**
- Modify: `bin/Spaces.py` (`PerceptualSpace` 9593, `PartSpace` 9643 +
  forward 16698, `WholeSpace` 17106 + forward 21097), `bin/Models.py`
  (call sites 6813/6862, `_make_perceptual_space`, isinstance sites),
  any other `PerceptualSpace` references (grep first)
- Test: `test/test_dual_towers.py` (new; replaces rev-1
  `test_ff_pyramid.py`, which is DELETED — its off-path nVectors pins move
  here)
- Config: `data/MM_sparse_concept.xml` WS transpose — **DONE 2026-07-10**
  (`nOutput` 1024$\to$8, `nOutputDim` 8$\to$1024; builds clean, WS matches
  PS $[8,1024]$).

- [ ] **A.1 Scope sweep:** `grep -rn "PerceptualSpace" bin/ test/` — list
  every reference (class def, subclass headers, isinstance, imports,
  `_make_perceptual_space`, docs strings). Record the list in EXECUTION
  NOTES before editing.
- [ ] **A.2 Failing test** (`test/test_dual_towers.py`): (a) `PartSpace`
  and `WholeSpace` subclass `Space` directly and no module attribute
  `PerceptualSpace` exists in `Spaces`; (b) `PartSpace.NULL_PERCEPT_KEY`
  still resolves; (c) build `MM_sparse_concept` — WS output shape matches
  PS ($[8,1024]$ views); (d) off-path shape pins (from rev-1 baselines):
  xor CS stores `[8,8,8]`, grammar `[4,2,2]`; (e) state_dict key set of an
  off-path build is UNCHANGED vs HEAD (capture once, pin).
- [ ] **A.3 Run red**, then remove the class: fold `NULL_PERCEPT_KEY` into
  `PartSpace`; `isinstance(x, PerceptualSpace)` $\to$
  `isinstance(x, (PartSpace, WholeSpace))`; rerun green.
- [ ] **A.4 Dual signatures (behavior-preserving migration):**
  `PartSpace.forward(in_sub, cs_out=None)` — `word_subspace` param retires
  or aliases; `WholeSpace.forward(in_sub, cs_out=None)` — internal
  routing: parallel-path predicate $\to$ universe-primary (`in_sub` = the
  IS unity view, `cs_out` = feedback); else LEGACY routing (`cs_out`
  primary, exactly today's tensors) so off-path math is untouched. Update
  the two call sites (6813/6862): pass `in_sub`/`_staged_concepts_in` and
  the per-tower CS feedback explicitly every stage.
- [ ] **A.5 Liveness + byte-identity:** new test — on `MM_sparse_concept`
  the WS forward consumes the universe view at every stage (assert the
  routed source, e.g. a stamped attr); off-path guard green
  (`test_reconstruction_roundtrip.py` + `test_recon_bench.py`).
- [ ] **A.6 CHECKPOINT (STOP, present to Alec):** serial-path routing
  migration — keep legacy routing for serial/sO=0 (default), or migrate
  and honestly re-baseline the grammar pins? Do NOT proceed past this
  checkpoint without an answer recorded in EXECUTION NOTES.

**GATE A (Alec commits):** class gone, dual signatures in, IS-to-both wired
on the parallel path, WS transpose recorded, off-path green.

---

## Task B — Callosum-fed order-0 (retire the snap/rectifier path)

**Files:**
- Modify: `bin/Spaces.py` (`cs_symbolic_phase` 14892, `cs_snap_order0`
  14631 — retire on this path), `bin/Models.py` (P3 cutover 7067-7095)
- Test: `test/test_dual_towers.py`
- Read first: `bin/Layers.py:1671-1714` (`_combine` / `views` / `glue`),
  the bind carrier the pump already produces (`carriers` list,
  bin/Models.py:6834; `ConceptualCombine.glue(carrier)` $\to [B,8,D]$).

- [ ] **B.1 Failing test:** on `MM_sparse_concept` (2 epochs, seed 0, cpu)
  the symbolic phase's order-0 input is the GLUED frame: nonzero,
  grad-bearing, shape $[8,B]$-compatible — and the epoch-1 all-negative
  annihilation signature is structurally impossible (no `clamp(min=0)` in
  the path; assert no rectifier call via a probe hook or by asserting
  signed values present).
- [ ] **B.2 Run red; implement:** at the cutover, read the terminal bind
  carrier's `glue(...)` (or the staged PS/WS views stacked $\to$ callosum)
  as the order-0 activation source; delete the `cs_snap_order0` call from
  `cs_symbolic_phase` (keep the method only if another consumer greps —
  verify; if EMA identity-trace consumers exist, decide keep/retire and
  record).
- [ ] **B.3 Run green; off-path guard** (parallel-only change — serial
  never enters `cs_symbolic_phase`).

**GATE B (Alec commits):** glued order-0 live; snap/rectifier retired on
the path; off-path green.

---

## Task C — Top-K `.index` pyramid ($8\to4\to2\to1$) + observability

**Files:**
- Modify: `bin/Spaces.py` (`cs_forward_content` 14698 — becomes the
  pyramid; `set_index` 6556 consumers; store sizing off tiles + pool)
- Test: `test/test_dual_towers.py`
- Read first: design §3-4; `bin/Spaces.py:3794` (top-K codebook forward
  pattern); `_populate_concept_weights` 14822 (constituent edges).

- [ ] **C.1 CHECKPOINT (STOP, present to Alec):** the composition/selection
  split — per order, the edge-structured $\sigma$ fold composes candidates
  and `.index` top-K selects; confirm mechanics (what composes the
  candidates concretely: the constituent-edge `SparseLayer.forward`, or
  selection-only over already-materialized codes) BEFORE writing code.
- [ ] **C.2 Failing test:** per-order selection counts $8/4/2/1$ (read the
  subspace `.index` shape after a forward); generic `materialize()` (no
  `mode=` argument) returns exactly the selected codes per order.
- [ ] **C.3 Implement** per the C.1 answer; store sized tiles(8)+pool(7);
  overflow loud (`pytest.warns(RuntimeWarning)` pin).
- [ ] **C.4 Liveness pins:** every level nonzero post-train (2 epochs, seed
  0); `param.grad.abs().sum() > 0` per level's parameters; the stacked
  activations reach `_concept_activations` (grad-bearing) and
  `_cs_parallel_slab` (shapes unchanged for the SS leg / SBOW loss).
- [ ] **C.5 Observability:** `_cs_level_acts` (per-order magnitudes),
  per-level `.index` occupancy, per-fold grad norms — report-only; smoke
  assertion they populate. `_cs_wave_qe` retires.

---

## Task D — Regression + gate

- [ ] **D.1 Off-path byte-identity sweep:** xor + grammar via
  `recon_bench` reproduce pre-change `output_loss`/`recon_loss` to 6
  decimals.
- [ ] **D.2 Targeted regression:**
  `PYTHONPATH=test:bin .venv/bin/python -m pytest test/test_dual_towers.py
  test/test_reconstruction_roundtrip.py test/test_recon_bench.py
  test/test_conceptualize.py test/test_conceptual_recurrence.py
  test/test_mereology_word_binding.py test/test_iterated_symbolic_wave.py -q`
  (the wave tests will need updating/retiring with `_cs_wave_qe` — do that
  here deliberately, recording each retired pin).
- [ ] **D.3 `make test`** on a quiet tree (mlx D2/D3 xfail per baseline).

**GATE C (Alec commits; plan done):** pyramid live + observability;
`make test` green.

---

## EXECUTION NOTES (append during execution)

1. **Rev-1 HEAD nVectors baselines (2026-07-10, pre-edit):**
   `MM_sparse_concept` CS stores `[8,8,8]` (config says 32 — the
   recontextualized Task-11 finding); `MM_20M_xor` `[8,8,8]`;
   `MM_20M_grammar` `[4,2,2]`.
2. **Probe evidence (2026-07-10, cpu/eager seed 0):** order-0 snap readout
   pre-clamp epoch 0 $\sim\pm$1e-3 (50% neg, 8/16 survive), epoch 1 all
   negative (mean $-$6.4e-3, 0/16 survive) $\to$ `_cs_wave_qe` $=[0,0,0]$;
   stages 1-2 never run the symbolic phase (single stage-0 cutover is BY
   DESIGN, bin/Models.py:7073-7076).
3. **WS transpose DONE (2026-07-10):** data/MM_sparse_concept.xml WholeSpace
   `nOutput` 1024$\to$8, `nOutputDim` 8$\to$1024 (+ one-liner comment);
   build verified clean, WS matches PS $[8,1024]$.
4. **Rev-1 test file deleted:** `test/test_ff_pyramid.py` (its off-path
   nVectors pins move into `test/test_dual_towers.py` A.2d).

### Task A execution (2026-07-11, cpu/eager)

5. **A.1 sweep:** `PerceptualSpace` had THREE subclasses — PS, WS, and
   `SymbolSpace` (bin/Language.py:12525, "peer perceptual tower") — plus one
   import (Language.py:54). NO `isinstance(..., PerceptualSpace)` call sites
   anywhere; all `NULL_PERCEPT_KEY` consumers already use
   `PartSpace.NULL_PERCEPT_KEY`. SymbolSpace included in the removal (all
   three now subclass `Space` directly).
6. **A.2-A.3 green:** class deleted; `NULL_PERCEPT_KEY` folded into
   PartSpace; state_dict key fingerprints byte-identical (xor 1875 keys
   sha16 8f4dc250..., grammar 2664 keys 3e5ff795...); off-path CS stores
   unchanged (xor `[8,8,8]`, grammar `[4,2,2]`).
7. **A.4 dual signatures in:** `forward(in_sub, cs_out=None)` on both
   towers; WS routing branch — parallel predicate
   (`_symbolic_order>0 and not _serial`, both already stamped per-WS at
   bin/Models.py:883-885) $\to$ universe-primary EVERY stage
   (`_stage0_unity_forward`), stamps `_ws_routed_source="universe"`, stashes
   `cs_out` as `_cs_feedback`; else legacy mapping (cs_out primary, unity
   read only on an empty carrier — pre-rev-2 dataflow verbatim), stamps
   `"legacy"`. The old repeated-injection `NotImplementedError` retired (the
   stamp is the loud signal). Three WS call sites migrated
   (bin/Models.py 6862, 7781, 7896); PS single-positional callers unchanged.
8. **Old-contract test migrations (deliberate, recorded):**
   `test_dual_input_contract.py` — 10 mechanical arg swaps;
   `test_ws_rejects_concepts_with_nonempty_cs` REWRITTEN to
   `test_ws_legacy_routing_with_nonempty_cs` (rev-2 law: unity + live
   carrier is legal; legacy stamp replaces the raise);
   `test_model_forward_passes_unity_at_stage0` capture updated (unity now
   the first positional; pin = offered every stage + all-legacy stamps).
   `test_perceptual_loopback.py` PS/WS arity pins $\to$ dual signature;
   `test_phase2_pipeline_primitives.py` arity table PS (1,1)$\to$(1,2).
9. **A.5 green:** 53 passed across the four contract files; off-path guard
   `test_reconstruction_roundtrip.py` + `test_recon_bench.py` 24 passed /
   2 skipped (RUN_SLOW) — the exact-match seeded pins prove grammar/xor
   dataflow preserved.
10. **Deviations:** (a) grammar (serial) drives only the TERMINAL stage's
    WS (per-word path), so stages 0-1 stamp None — the off-path pin asserts
    ABSENCE of "universe", not all-"legacy". (b) The serial per-word PS
    calls (bin/Models.py:9246/9253) pass `prevCS_forSS` as `in_sub`
    positionally — behavior-preserving but semantically the C$\to$P
    feedback; fold into the A.6 serial checkpoint decision.

### Tasks B+C execution (2026-07-11, post-sync onto d469a19)

11. **A.6 checkpoint decision (Alec):** option 1 — legacy routing stays for
    serial/sO=0; serial migration deferred to its own phase with an honest
    re-baseline. Alec also authorized git sync + commit for this work.
12. **Upstream sync:** stash $\to$ ff-merge cda1fae$\to$d469a19 ("Wholes are
    types") $\to$ pop; two conflicts resolved (bin/Spaces.py — upstream's
    property-comment rewrite kept, stale PerceptualSpace-docstring ref
    fixed, `property_basis` flag stays deleted; todo.md — upstream on both
    blocks, the local QUERY-TOOLS entry was the older pre-design framing).
    Wholes-are-types research: NO interaction with this work; the feature
    is DORMANT on MM_sparse_concept (`<analysis>raw</analysis>`); flipping
    to `word` (typed runs into the glued frame) is an explicit future
    decision — it activates every span consumer + checkpoint vocab_extras.
13. **Research correction (3-agent sweep):** the cutover's
    `last_cs.materialize()` ALREADY IS the callosum-glued frame
    ($[B,8,1024]$; `bind_streams` writes `glue(full)` as the CS event;
    MM_sparse_concept is 3-stream PS+WS+SS-zero, callosum $[24,8]$) — the
    darkness was the RECTIFIED re-read of that frame against the codebook,
    not missing plumbing. Task B therefore reduces to the SIGNED snap.
14. **Task B (signed snap):** `cs_snap_order0` reads `nonneg=False`
    (bin/Spaces.py $\approx$14620); EMA identity trace kept (it is what
    makes order-0 rows discriminative; its pins survive). `_cs_last_a0`
    stashed report-only. Cutover gains fail-loud shape asserts
    (acts $[nVectors, B]$ rows-first; bin/Models.py $\approx$7090).
15. **Task C (pyramid):** `cs_forward_content` is now the feedforward
    edge-structured $\sigma$-pyramid — $a^0$ = signed order-0 tiles padded to
    the inventory; per rung $k$: ONE hop `forward_linear([a[:S]|1])` in
    STORE coordinates ($S = \sum caps$, the `_sizer` contract), gather
    block-$k$ rows, tanh, top-$caps[k]$ per batch, winners scatter
    (out-of-place `index_copy`), losers zero. No fixed point, no source
    re-injection. `_order_caps` $\to$ the per-order taper with BASE-FIT
    (halve the tile base until $\sum caps \le nVectors$: 32/8/K3 $\to$
    (8,4,2,1); 8/8/K3 $\to$ (4,2,1,1) mirroring the legacy split);
    `_csw_concept_row` $\to$ per-order blocks (namespace `("o<k>", cid)`);
    `add_concept_edge` translates the global bias convention
    (col == nVectors) to the store's own bias column ($S$). CS-only
    validator relaxation: pure-event ConceptualSpace accepts
    `nVectors >= nActive` (inventory decoupled post-P3) — this is what
    finally lets config `<nVectors>=32` reach the parallel store
    (Task 11 CLOSED; sizing gate at bin/Models.py $\approx$5989 reads the
    XML directly since symbolicOrder parses after create).
16. **Selection staging:** `cs_symbolic_phase` stages the concatenated
    per-order winners on `cs.subspace.set_index([B, n_sel, 1])`; a new
    ADDITIVE `materialize()` path (`_index_basis` = the similarity
    codebook; active+event modes only; int index only) pulls exactly the
    selected codes on the otherwise pure-event CS subspace. Staging is
    PER-BATCH state (SubSpace.End() releases it) — the pin reads inside
    the consumption window. Stale-index clear added at
    ConceptualSpace.forward start.
17. **Retired:** `_cs_wave_qe` (now None; per-rung `_cs_level_acts` /
    `_cs_level_rows` replace it) and `cs_groundedness_probe` + its four
    tests (design decision 5: a feedforward net cannot represent
    self-sustaining loops).
18. **Two real bugs caught by the wave-era suites during migration:**
    (a) taper degeneracy at nVectors == nActive (whole store became snap;
    fixed by the base-fit halving); (b) bias-column mistranslation
    (edges to col nVectors=32 on a 15-col store; fixed by the
    add_concept_edge translation).
19. **Test migrations (agent-executed, per-node record in the agent
    report):** test_iterated_symbolic_wave.py (depth-schedule $\to$
    structural completion; cycle/QE + groundedness nodes retired; helpers
    to per-order namespaces), test_cs_sparse_weights.py (taper/store/
    signed-snap/rung-algebra/empty-store-pass-through/level-stats/signed-
    winner/differentiable-via-minted-rung rewrites; K-schedule retired),
    test_sparse_concept_e2e.py (optimizer-step scenario mints rows so
    grads reach values). 38 passed across the three; test_dual_towers +
    test_sparse_layer confirmed untouched (29 passed).
20. **First `make test` (19 failed / 3061 passed) caught two more
    consumer gaps, both fixed in bin/Spaces.py:** (a) old-style positional
    `ws.forward(carrier)` callers (test fixtures) crashed on the legacy
    mapping — WS.forward now TYPE-discriminates its first arg (SubSpace
    $\to$ the pre-rev-2 legacy carrier call shape; raw $[B,1,N]$ tensor
    $\to$ the universe view), and the parallel branch never consumes a
    carrier as unity; (b) `_row_to_concept` filtered row namespaces to
    ("snap", "pool") — the per-order `("o<k>", cid)` blocks never inverted,
    so `typed_definition` heads lost their cid. After both:
    test_subspace_what_stm_contract + test_symbolic_iteration +
    test_typed_definition 65 passed / 1 xfailed; contract pins re-verified
    (62 passed).
### Serial-routing migration + attention audit (2026-07-11, post-d293b4d)

22. **Serial migration executed (Alec follow-on decision 1).** Final state
    — THE TYPED + LIVENESS ROUTING LAW in WS.forward: a raw unity tensor
    is OFFERED at every pump (no ``serialObjectMeta`` feed gate, no
    serial/parallel routing branch); it routes UNIVERSE-primary when its
    analysis is ALIVE (nonzero event) or the carrier is empty (bootstrap)
    or the path is parallel (glue contract); a DEAD unity with a live
    carrier routes the CARRIER body (the recurrent leg + grammar dispatch
    + quantize snap + stack router + truth recording). The "legacy" stamp
    is renamed "carrier" — it is the machinery body, not a compat shim.
    Truth recording was EXTRACTED (`_record_truth_activations`) and fires
    from BOTH branches, so `store_truths` ingestion works under either
    routing.
23. **Measured verdicts and the honest finding (cpu/eager seed 0,
    recon_bench 3 epochs):** NO shipped config has a LIVE universe
    analysis today — grammar's unity event materializes ALL-ZERO
    ($[B,2,8]$, max 0.0), xor's unity is a degenerate flattened grid
    ($[B,1,8323072]$ int) whose analysis is a zero grad-free constant, and
    MM_sparse_concept (`<analysis>raw`) likewise. Consequently: grammar
    routes carrier (recon 0.051825, exact **1.0**, byte-stable); xor
    byte-identical ($0.200180/0.009222$; the every-stage universe attempt
    broke its grads-flow + round-trip pins — its $t>0$ carrier legs are
    load-bearing). The intermediate "universe-every-pump" experiment
    IMPROVED grammar recon 10$\times$ ($0.0518 \to 0.0051$ at exact 1.0) —
    but the unity being all-zero means this was a WS-LEG ABLATION effect
    (silencing the serial WS carrier contribution), NOT universe
    re-anchoring. Recorded as a finding: the serial WS leg's recurrent
    contribution HURTS recon on grammar; worth a deliberate ablation
    experiment, not a silent routing side effect.
24. **THE TRADE (what serial migration buys/costs):** GAINED — one uniform
    typed+liveness law (no mode branches; ``serialObjectMeta`` no longer
    gates the unity feed); the universe path is fully plumbed end-to-end
    and LIGHTS UP AUTOMATICALLY (per space, per pump) the moment a live
    analysis front-end lands (`<analysis>word</analysis>` — follow-on
    decision 2), with truth recording already wired on that branch; the
    pre-rev-2 "WS processes symbols" leftover gone. COST/KEPT — the
    carrier body remains the $t>0$ recurrent leg wherever the universe is
    dead (today: everywhere), so no behavioral change ships for
    serial/sO=0 configs (grammar exact bar unchanged at 1.0, xor
    byte-identical); FULL carrier-path removal is bounded by giving
    inputs a live universe analysis, and is deferred to that phase.
27. **Three-bases relevance stub (Alec, 2026-07-11; Architecture sec C
    rewritten in the relevance $\to$ attention $\to$ awareness frame):**
    per-tower `relevance_weights()` contract on Space (PS part-salience /
    WS whole-relevance / SS symbolic history; all default None =
    byte-identical, each carrying its intended-live-source note), and the
    CS readout consumes `_relevance_priority` ($[N]$ or $[N,B]$) as a
    RANKING bias on the per-order top-K — winners change, activations
    never distort (pinned in test/test_relevance_bases.py, 3 tests).
    OPEN SPEC (deliberate): (a) PS salience polarity (settle-residual
    surprise vs snap-strength familiarity — literature favors surprise/
    distinctiveness for the exogenous basis) and the slot$\to$row
    projection; (b) WS whole-relevance projection (intent affinity /
    readingAttention scores $\to$ CS rows); (c) SS heat rows $\to$ CS
    inventory projection + the combination law (product vs sum — the
    priority-map literature sums). Structural note: today allocation
    never exceeds the taper caps, so priority REORDERS rather than
    EXCLUDES; competitive exclusion engages when the store outgrows the
    taper.
28. **Relevance projections IMPLEMENTED (Alec: "spec and implement",
    2026-07-11; spec = design doc follow-on decision 4).** Gated
    `<architecture><relevance>` (default false, byte-identical; XSD +
    parse). Under the gate, the cutover assembles the priority
    (`_assemble_relevance_priority`, no_grad): per-tower NOVELTY = settle
    residual (`snap_settle_qe`) on each tower's VIEW HALF of the bind
    carrier (stashed; `relevance_weights()` pull-API live on PS/WS),
    slot$\to$row by the snap argmax (scatter-amax), SUMMED with the
    symbolic-history projection (`symbol_history_priority(heat)`: heat
    over `('sym', id)` constituent records $\to$ concept rows; live-dark —
    `_symbol_heat_source()` returns None until priming is enabled). The
    pyramid consumes it with the SPREAD LAW: rung score $= p +$
    `forward_linear_abs` hop ($|W|p$; new abs-weight kernel in
    SparseLayer._matmul), rank $= |cand|(1 + score)$, admitted rows only
    carry their score upward. 6 pins in test_relevance_bases.py (spread,
    projection, gated e2e, default-off, rerank, no-distortion).
    Deferred: readingAttention's symbolic origin (next projection);
    per-order competitive EXCLUSION still awaits store growth past the
    taper (priority reorders today).
29. **Simplified relevance law implemented (supersedes the item-28
    assembly; Alec's one-quadratic reframe).** `Space.prime_seen`
    (bump + `<primingDecay>` decay toward neutral) / `prime_desire`
    (signed, floor 0) / `priming_weights`; surfaces: CS $=$ concept
    inventory (`_priming_dim` override), WS $=$ analysis-store rows.
    Writes: WS seen $=$ stage-0 selections (`_stage0_indices`, at the
    gated cutover assembler); CS seen $=$ the pyramid's ADMITTED rows
    (awareness primes, post-`cs_symbolic_phase`); desire/hate $=$
    `prime_desire` (intent wiring later). Reads: `_relevance_priority`
    $=$ CS boost $-$ 1 (the spread law unchanged); `_primed_reading_step`
    $=$ hard-coded readingAttention (hottest-primed word-whole's span,
    learned-producer contract, active when `<relevance>` on and
    `<readingAttention>` off). The item-28 settle-residual/view-half
    novelty assembly is DELETED; `relevance_weights()` now IS
    `priming_weights()`. 9 pins in test_relevance_bases.py.
26. **Trace safety:** the liveness probe is data-dependent control flow —
    `make test` caught it via the compiled-CLI XOR node and the mlx export
    (GuardOnDataDependentSymNode). Fixed with the house
    `torch.compiler.is_compiling()` guard: under trace the probe is
    skipped and the carrier path is taken (factually correct — all
    shipped universes are dead); revisit the compiled story when
    `<analysis>word</analysis>` lights the universe path up.
25. **Attention audit (agent sweep, 9 mechanisms):** the pyramid replaced
    exactly ONE mechanism — the ConceptualAttentionLayer WAVE (same store,
    same $\tanh(W[a|1])$ hop, iteration $\to$ per-order top-K). INTACT and
    composing: ReadingAttention (gate `<readingAttention>`, default off;
    MM_qa/MM_query_reasoning/MM_global/MM_reading enable), GlobalAttention
    (+`<globalAttentionConsume>`, default off; MM_qa/MM_query_reasoning/
    MM_global/MM_symbol_attention/MM_ws_tall), intent priming /
    `_topk_priming_mask` (dark until an intent is set; same top-K idiom as
    the pyramid but on the WS word-codebook axis — composes, no
    duplication), reverse-side symbolic-heat `<attention>` modes +
    `<symbolicPriming>` (default off; NO shipped config enables them).
    Already-inert pre-rewrite: `hasAttention` (deprecated alias),
    QKVAttentionLayer (retired class, test-only). Residuals: `wave_step`
    orphaned (kept, pinned by 2 substrate tests), `_cs_wave_qe=None`
    write-only retirement marker (pinned deliberately).

21. **Second `make test` (4 failed / 3076) caught the bias-edge LIFECYCLE
    gap:** bias edges were ADDED at the store's own bias column but
    REMOVED at `col == nVectors` (and both remove sites resolved rows via
    the retired ("pool", cid) namespace) — retired EVERYTHING poles would
    never have pruned. Fixed with two CS helpers (`_csw_row_of` across
    snap/pool/`o<k>` namespaces; `_bias_col` = store nOutput) used at the
    add + both remove sites, PLUS read-side back-translation in
    `concept_weights` (callers always see the GLOBAL bias convention
    col == nVectors; storage stays internal). Three freshly-migrated
    store-coordinate pins updated to the global convention. Also migrated
    the remaining Stage-1.A arity pin
    (test_ps_single_arg_refactor $\to$ dual signature). Affected set:
    157 passed / 1 xfailed across eight suites.
