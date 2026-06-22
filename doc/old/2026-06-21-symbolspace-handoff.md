# SymbolSpace refactor — handoff plan (2026-06-21)

Handoff for the next conversation. Captures everything DONE this session and every
REMAINING / DEFERRED item, with entry points + design pointers so the next session
starts cold-ready. All work this session is **UNCOMMITTED** (Alec commits himself).

Design specs this builds on:
- `doc/old/2026-06-21-higher-order-symbolic-composition.md` (§2 collapse→1:1, §2b
  meta-concepts, §3 intensional, §4b conceptualize spec, §4c storage/discretization,
  §4d Phase-2 wiring DONE)
- `doc/old/2026-06-21-terminology-percepts-concepts-symbols.md` (percepts/concepts/
  symbols; the resolved code-rename decisions)
- memory `symbolspace-refactor` (the running state, codebook architecture, decisions)

---

## DONE this session — green (off-path suite 2742/2743 baseline, byte-identical)

- **Stage 1** rename `SymbolicSpace`→`SymbolSpace`.
- **Stage 3** `SymbolSpace(PerceptualSpace)` — the 3rd tower (owns `SymbolicSubSpace`).
- **3-stream peer bind** — `ConceptualCombine(n_streams=3)` (Layers.py:1513) +
  `bind_streams` optional 3rd SS leg (Spaces.py:14677); gated `<symbolTower>`.
- **Stage 4 = the conceptualize redesign:**
  - `SymbolicSubSpace.conceptualize(order, …)` (Language.py) — orders 0=`relate`
    [part,whole], 1=`create_word_object_meta` [object isa word], 2=`synthesize_higher_order`,
    3=`conceptualize_chain` (Gallistel sequence). Tested: `test/test_conceptualize.py`.
  - Symbol codebook on `SS.subspace.what` (Language.py `SymbolicSubSpace.__init__`).
  - **CS-mediated bind leg** — `ConceptualSpace._build_symbol_leg` (Spaces.py, eager
    island) reads order-raising "meta" codes via the WS ref, syncs them onto
    `SS.subspace.what`, builds the leg; `bind_streams` calls it; per-stage CS get
    `_model_symbolSpace` (Models.py ~5859, gated). `SymbolSpace.forward_symbol` REMOVED.
- **Autobind unblocked** (the symbol data source): WS `<codebook>quantize</codebook>`
  (was `none` → the `isinstance(ws_what, Codebook)` gate no-op'd) + a caller-side
  seed-trim in `_autobind_word_wholes` (Spaces.py ~13238: muxed seed 1024 → codebook
  nDim 1020). Now `meta=32, raised=4` flow; on-path loss moved 0.1750→0.1712.
- **Terminology** — `[whole, part]` decided; doc pass over 7 live docs.
- **Compile** — fullgraph relaxed for `symbol_tower` (Models.py ~2960 `_host_islands`).
- **Config** — `data/MM_symbol_tower.xml` (symbolTower + mereologyRaise + WS quantize).
- **MPS** anchor fix (AnchorDotTransformChooser `.to(device)`).

---

## REMAINING — the refactor + sharpenings

### R1. Higher-order SEMANTIC layer (the real symbol) — biggest, next-iteration
Today the SS leg carries the order-raising code (synced to `SS.subspace.what`); the
channel is real but the symbol is not yet the *meaningful* 1:1-with-concept symbol.
Build (spec: higher-order plan §2/§3):
- **CS collapse → 1:1**: collapse all concepts at the same `.where` + `.when` into one
  (binding by co-presence) = create the higher-order `[whole, part]` superset that
  subsumes the members → 1:1 concept↔symbol, no lookup. This IS `maybe_raise_order` /
  `synthesize_higher_order` driven by co-presence; hook it as the symbol derivation.
- **1:1-by-concept-index**: `SS.subspace.what` rows indexed by concept (not the current
  sync of meta-codes to leading rows).
- **Intensional higher-order concepts** + the correlation loss to the extensional
  percepts (σ pulls the many → quantize collapses onto the higher-order codebook).
- Entry: `ConceptualSpace._build_symbol_leg` (Spaces.py), `synthesize_higher_order`
  (Spaces.py:13513), `maybe_raise_order` (Spaces.py:16844).

### R2. Attention → PS/WS/SS, home = CS
Scoped + documented, NOT built. Attention ranges over the three towers' `.what`; CS
shapes the `.where` (every PerceptualSpace has `.where`). Wire CS as the home of the
`.where`-producer over PS/WS/SS (aligns with existing `_passback_scope_*` /
`passback_action` + `GlobalAttention`). Entry: Models.py `GlobalAttention` (~772) +
the CS `.where` passback; memory `reading-attention-A-landed`.

### R3. Stage 6 — WS-tall + delete `symbol_dim=8`
The symbol migrated to SS (cutover); make WholeSpace emit TALL whole-percepts
(`nOutputDim` 8→1024, drop the `forwardEnd` compact reshape, Spaces.py:7734) and
remove the `symbol_dim=8` banding. Touches WS.outputShape → OS head + validate_config
+ checkpoint regen. Gated/dedicated config; not byte-identical on-path.

### R4. Stage 5 — collapse the C/S grammar tiers into ONE dispatch
(+ Stage 2, dead-P-tier removal, folded in.) The grammar dispatch should be one
tier, not C/S split. Entry: the SyntacticLayer per-tier dispatch (Language.py
`_attach_per_space_syntactic_layer` ~9386), `SymbolSpace.forward/reverse` (the grammar
compose/generate). Pending.

### R5. Code-identifier pass (decided, suite-gated)
- `_sym_*` → `_concept_*` + the verbs (`reify_relation`→`reify_concept`,
  `new_symbol`→`new_concept`, `symbol_is_identity`→`concept_is_identity`, …).
- `SymbolicSubSpace` → `SymbolSubSpace`.
- Word-bounded substring sweep like the SymbolicSpace→SymbolSpace rename; one churn,
  post the structural work, suite-green gate. Spec: terminology doc "Resolved decisions".

### R6. Invocation wiring (APIs built, not yet called from the live path)
- `conceptualize_chain` — wire the **sentence-level call site** (build a chain from the
  reduced word-concept sequence at sentence end). Currently the API exists, unused.
- `conceptualize` (relation API) — the grammar COMPOSE path can call it for relation
  formation (Option B); not yet wired. The autobind (Reset) remains the concept-former.
- `subspace.index` STM live-slot marker — deferred to its consumer (the index-based
  grammar dispatch); the typed STM already pushes-to-END.

---

## DEFERRED — separate, pre-existing tasks (not the refactor)

- **D1. MM_20M perf fixes** (verified, unapplied; memory `mm20m-training-fixes`):
  (a) grammar bloat 339M/114s→26M/0.46s = cap combine `n_vectors` at `nConcepts`
  (Models.py) + WS butterfly false; (b) MPS inductor `r0_0` codegen workaround
  (`pop().root.cache_clear()` monkeypatch) → compiled-MPS trains.
- **D2. Reasoning** implementation — `doc/old/2026-06-20-reasoning.md`.
- **D3. Decoder** — wire the dual grammar onto the reverse path (parse-tree-free
  decode); memory `idea-decoder-design` (live decode is NOT parse-tree-driven; build =
  wire `role_collapsed.grammar <generate>` onto the reverse path + real ±/0 inverses).
- **D4. MPS** device-move unification (broader than the anchor fix) — deferred until
  after the refactor (Alec's call); re-verify against `BASICMODEL_DEVICE=gpu`.

---

## CAVEATS for the next session

- **Don't edit source files while a suite runs** — `inspect.getsource`-based tests
  (e.g. `test_symbolic_space_init_builds_rule_codebook`) read the file by line number
  and fail spuriously if lines shift mid-run. (Caused 2 phantom failures this session.)
- **Flaky tail**: `test_explicit_dimensions.py::…::test_output_mse_is_crisp` (XOR MSE
  CLI) is unseeded → flaky; passes on retry; do NOT re-pin.
- **Gating contract**: every symbol-tower feature is gated `<symbolTower>` (default
  off → byte-identical). Verify off-path with the full suite each step.
- **Run pytest from `basicmodel/`**, CPU-pinned by conftest; GPU/MPS for training only.
- **Commit**: leave changes in the working tree; Alec commits.

---

## KEY ENTRY POINTS (quick map)

- `bin/Language.py`: `SymbolSpace` (~12571), `SymbolicSubSpace` (9122) +
  `conceptualize` / codebook on `.what`, `forward`/`reverse` (grammar).
- `bin/Spaces.py`: `ConceptualSpace._build_symbol_leg` + `bind_streams` (14677),
  `conceptualize_chain` / `synthesize_higher_order` (13513) / `relate` (13393) /
  `create_word_object_meta` (13579) / `maybe_raise_order` (16844),
  `_commit_autobind_from_stash` (12909), CS.forward + STM (14776).
- `bin/Layers.py`: `ConceptualCombine` (1513).
- `bin/Models.py`: gate `symbol_tower` (~794), combine `n_streams` (~6074), per-stage
  loop bind (~6479), per-stage CS attach (~5859), fullgraph relax (~2960).
- `data/MM_symbol_tower.xml`, `data/model.xsd` (`<symbolTower>`).
