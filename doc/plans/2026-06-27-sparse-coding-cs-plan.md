# Sparse-Coding Conceptual Space — implementation plan (build fresh)

**Date:** 2026-06-27
**Status:** Plan only. Write the code fresh from this. The exploratory changes in the working tree
(`bin/Spaces.py` edge-store/scatter/`_stacked_basis` methods + `test/test_cs_edge_store.py`) are
**reference-only and should be reverted** before starting — see "Existing code" below. They predate the
final SS/CS spec; `_stacked_basis`/`_symbol_stack_ranges` in particular encode a wrong model and must
NOT be recreated.
**Companion:** the "what was explored" record is `doc/plans/2026-06-27-sparse-coding-cs-handoff.md`
(historical; this plan supersedes its "already built / keep it" framing).

---

## Context / goal

Part B needs a real conceptual space: a concept with a *located code*, not a row index. The prior SBOW
spine was forward-disconnected (trained a side codebook nothing downstream read). The locked
resolution:

> **A concept is a sparse, weighted linear combination over a basis of symbols** — and *symbols are
> concepts from the previous ramsified order*, so the whole thing is one self-referential sparse map.

Outcome: substitutable concepts co-locate; the concept code is forward-connected by construction
(gradient reaches it through the transform); existing `symbolicOrder=0` configs stay bit-for-bit
identical. Git is the user's; this plan touches no git.

---

## Architecture (locked)

1. **One row-aligned symbol/concept inventory.** There is a single inventory of located codes. Each
   row has two *views*: the **CS (concept) view is MUXED** (`.what (+) .where (+) .when`, width
   `muxedSize`); the **SS (symbol) view is UNMUXED** (`.what` only; where/when stamped at
   `materialize()`). "Symbol $i$ == the unmuxed view of concept $i$." (PS/WS are likewise unmuxed
   perceptual spaces.)
2. **A concept = a sparse weighted combo over that inventory.** Stored as a **three-column COO table**
   `(ConceptIndex, SymbolIndex, Weight)` — space-optimal over a million-row inventory, extensible by
   appending. `Weight` is learnable, default uniform (normalized by per-concept degree so uniform ==
   mean).
3. **The transform is a differentiable scatter-add SpMM** (NOT `torch.sparse` — MLX/executorch
   safety):
   `concept[c] = sum_{e: ConceptIndex[e]=c} Weight[e] * Basis[SymbolIndex[e]]`
   realized as `index_add_(index_select(Basis, 0, sym) * w, concept)`. Differentiable in `Weight`
   **and** `Basis`.
4. **Two directions, each owned by a space, each fed through `.forward()`:**
   - **PS/WS $\to$ CS** (build a concept from constituents) runs in `ConceptualSpace.forward`.
   - **CS $\to$ SS** (a concept $\to$ its symbol / representation, which decodes to the percept) runs
     in `SymbolSpace.forward`.
   It is a sparse autoencoder: `PS/WS -(W)-> CS -(view/snap)-> SS -(W^T / edges)-> percept`.
5. **Recursion = ramsified order.** Order-$k$ concepts are built from order-$(k-1)$ **symbols** (which
   are order-$(k-1)$ concepts). Base order: constituents are PS/WS percepts.
6. **The symbolic loop runs in PARALLEL mode** (`serial=false`, `symbolicOrder>0`) — the parallel body
   `_forward_body` (`bin/Models.py` `for t, stage` loop; `cs.forward` `:6562`, `cs.bind_streams`
   `:6677`), NOT the per-word serial path.
7. **Activation is by `symbolicOrder`, not a config gate.** `symbolicOrder=0` $\Rightarrow$ empty edge
   table $\Rightarrow$ the forward falls back bit-for-bit. No new `<...>` knob.

### THE dataflow rule (the thing the old code broke)

**Cross-space interaction happens ONLY through `.forward()`.** A space may gather from (a) the codebook
it **owns**, or (b) a **subspace passed into its `forward(...)`** — because *the codebook rides the
subspace through forward* (Spaces are operators). A space may **NOT** dereference a stashed pointer to
another space (e.g. the retired `_model_symbolSpace`) or reach into another space's codebook directly.

**`.forward()`-safe basis sourcing (the key design resolution):**
- **CS $\to$ SS** (`SymbolSpace.forward(concept_sub)`): SS gathers from the **symbol codebook it
  owns** (row-aligned with concepts). No reach.
- **PS/WS $\to$ CS** (`ConceptualSpace.forward(percept_sub, whole_sub)`): the base-order constituent
  rows come from the **percept/whole subspaces passed in** (their codebooks ride them). No reach.
- **Higher orders**: the prior order's symbols are **passed forward** as the next order's
  constituents (the loop carries `cs._subspaceForWS` $\to$ next `ws.forward`, etc.).

The existing `_stacked_basis(ps_W, ws_W, ss_W)` is wrong on two counts and is dropped: it models the
basis as a `[PS|WS|SS]` concatenation of *three separate* codebooks (but the SS block IS the concept
codebook, row-aligned), and using it requires CS to gather other spaces' codebooks (a reach). Sourcing
is per the rule above instead.

---

## Open decisions for the user (resolve before/while building)

- **D1 — one inventory or two synced codebooks?** Recommended: **one** located-code store with a muxed
  CS view and an unmuxed SS view (reuse the always-built muxed `similarity_codebook`,
  `bin/Spaces.py:12584`, `nDim=muxedSize`, has `self.vq` EMA — as the muxed view; the SS unmuxed view
  is its `.what` slice). Alternative: separate CS and SS codebooks kept row-synced (more state, must
  not drift). This sets where the basis lives and how growth stays row-aligned.
- **D2 — does the scatter-add REPLACE the combine in the active path?** When `symbolicOrder>0` + edges
  exist, the concept code is the scatter-add output; the learned `combine`/`unbind` (`bind_streams`,
  `bin/Spaces.py:15276`) is bypassed for concept *production*, and **reconstruction is referential via
  the edges** (the COO table is the decomposition), not via `unbind`. Recommended: **yes, replace in
  the active path** (keep `combine` for the `symbolicOrder=0` byte-identical path). Confirm — it
  decides the reconstruction story.
- **D3 — CS$\to$SS exact computation.** With symbols row-aligned to concepts, `SymbolSpace.forward`
  most simply returns the symbol view of the concept rows (unmuxed) for the bind leg / next order. Is
  there any learned map concept$\to$symbol, or is it the row-aligned view + snap? Recommended:
  row-aligned view + EMA snap (the discrete symbol identity), no extra learned map.

---

## Build order (phases; each lands byte-identical at `symbolicOrder=0`)

### Phase 1 — COO edge store (host-side, on ConceptualSpace; no forward change)
Mirror the existing relation tables (`_concept_tables` `bin/Spaces.py:13539`, `new_concept`,
`add_part`/`add_whole`, idempotency cache `_concept_relate_idx`, lifecycle `retire_concept` `:13783`,
`resolve_identities`, `refine_over_collected`).
- `_edge_tables()` — lazy-init the three COO columns + a `(concept,symbol)->row` dedup dict +
  `_edge_dirty`.
- `add_edge(concept, symbol, *, weight=1.0)` — idempotent per `(concept,symbol)`; sets dirty.
- `concept_edges(concept)` — deterministic `(symbol, weight)` list.
- Lifecycle hooks: `retire_concept` **drops** the concept's edges (pure no-op when the table is
  unbuilt $\Rightarrow$ byte-identical); `resolve_identities` **keeps** edges; `refine_over_collected`
  has the minted `H` **inherit the union** of constituents' edges.
- `_rebuild_edge_pools()` — rebuild-on-dirty: long index columns + the learnable `Weight` `Parameter`
  (grown tail-preserving, like `grow_to` `:3032`). Never per-forward.
**Tests** (host-side, instantiate a bare `ConceptualSpace` via `test_basicmodel._populate_test_config`,
no model): dedup; weight carried; retire drops; rebuild preserves trained weights on grow; byte-
identical (empty table $\Rightarrow$ `torch.equal` on a full forward/backward).

### Phase 2 — the scatter-add kernel (pure, on ConceptualSpace)
- `scatter_concept_event(n_concepts, basis, *, normalize=True)` — the SpMM above; `index_select` +
  `index_add`, `1/deg` normalize so uniform == mean; differentiable in `Weight` and `basis`. NO
  `torch.sparse`. Returns `[n_concepts, D]` (expand to `[B, N, D]` at the call site).
**Tests:** matches a dense `einsum` reference; uniform == mean; gradient reaches `Weight` and `basis`;
no `torch.sparse` symbol used. (Pass a small hand-built `basis` directly — do NOT build a
cross-space dictionary.)

### Phase 3 — PS/WS $\to$ CS in `ConceptualSpace.forward` (`.forward()`-safe basis)
Wire the scatter-add into concept production, sourcing the basis only from the **passed-in** percept/
whole subspaces (codebooks riding them) and CS's own inventory — never a cross-space reach.
- When `symbolicOrder>0` + `not serial` + the edge table is non-empty, the concept content is the
  scatter-add output (per D2, this supersedes the `combine` content in the active path); else the
  current path, unchanged (byte-identical). Carry where/when from the live constituent events into the
  muxed concept (the mux happens here — the perceptual$\to$conceptual transition).
**Tests:** on a populated edge table, the concept moves toward its weighted constituents; gradient
reaches the forward concept code; `symbolicOrder=0` byte-identical.

### Phase 4 — CS $\to$ SS in `SymbolSpace.forward`, in the symbolic loop (retire the violation)
- Add `SymbolSpace.forward_concept_to_symbol(concept_sub)` (or extend `SymbolSpace.forward`,
  `bin/Language.py:12702`) — receives the concept **through the arg**, returns the symbol-view
  subspace from SS's **own** codebook (row-aligned). No `_model_symbolSpace`, no WS-meta sourcing.
- In the parallel body, after `cs.forward` (`bin/Models.py:6562`), pass the concept through
  `symbolSpace.forward_concept_to_symbol(CS_sub)` $\to$ `SS_sub`, then
  `cs.bind_streams(ps_t, WS_sub, CS_sub, SS_sub=SS_sub)` (`:6677`).
- **Retire** `ConceptualSpace._build_symbol_leg` (`bin/Spaces.py:15188`) and the
  `_model_symbolSpace` stamp (`bin/Models.py:6055`) — the dataflow-rule violation. Watch the other
  `_model_symbolSpace` readers (`bin/Spaces.py:8830, 8963, 9048, 13070`) — scope the change to the
  symbol-leg path only.
- Reuse the EMA `similarity_codebook` (`:12584`) as the muxed concept VQ; the SS view is its `.what`
  slice; `grow_to` row-aligned (D1).
**Tests:** `<symbolTower>` config; the symbol leg comes from the concept rows via `SymbolSpace.forward`
(assert no `_model_symbolSpace` access on that path); `symbolTower` off $\Rightarrow$ byte-identical
(2-stream path at `bin/Spaces.py:15296`).

### Phase 5 — populate edges at mint (decoupled; ungated by `<mereologyRaise>`)
Add a small ungated population path that calls `add_edge` from the mint sites:
`create_word_object_meta` (`bin/Spaces.py:13956`), `synthesize_higher_order`, `relate`/`reify_concept`,
`_populate_cs_symbols`. **Min-support invariant:** a concept is eligible for the scatter only with
`>=1 PS-constituent + >=1 WS-constituent`, or `>=2 symbol-constituents`; else fall back to the current
content. Assert each `SymbolIndex` is in range of the inventory.

### Phase 6 — driver config + end-to-end verification
`data/MM_sparse_concept.xml` (sibling of `data/MM_substitution.xml`): **`symbolicOrder>=1`**,
**`<serial>false</serial>`** (override the back-compat `serial = symbolicOrder>0` default,
`bin/Models.py:619-653`), `subsymbolicOrder=3`, `conceptualSimilarityScale=0.1`, `symbolTower=true`;
reuse `loadSubstitution` (`bin/data.py:830`). Verify: gradient reaches the forward concept code;
substitutable concepts' rows co-locate (rising cosine); full `symbolicOrder=0` suite byte-identical
(`make test`).

### (Later) Situate signal
The scatter-add is forward-connected by construction. The substitutability *situating* loss can reuse
`conceptual_sbow_loss_codes` (`bin/embed.py:1728-1801`) pointed at the grad-bearing scatter-add codes
(mutually exclusive with the legacy disconnected snap at `bin/Models.py:3310-3351`); confirm the
co-presence window source in the parallel symbolic body. Register `Weight` into the optimizer here.

---

## Byte-identical contract / traps

- **The gate is `symbolicOrder`.** Most of the suite runs `symbolicOrder=0` and MUST stay bit-for-bit
  identical: empty edge table + the `n_streams==2` off-path (`bin/Spaces.py:15296`). Verify with a
  seeded build-twice + `torch.equal` (cf. `test/test_codebook_update_law.py:107-114`).
- **Mux seam (highest risk).** PS/WS/SS views are unmuxed; the concept is muxed. The where/when must
  be carried from the **live constituent events** at scatter time (not invented from a static
  codebook). Land the dense-match kernel test before trusting the muxing.
- **Dataflow rule.** No method on one space may read another space's codebook or a stashed Space
  pointer; only owned codebooks + subspaces passed through `forward`. This is the rule the retired
  `_build_symbol_leg` broke.
- **"CS codebook rejected"** rationale (`bin/Spaces.py:14002-14010`) forbade a *learnable* CS `.what`
  codebook for relation-only symbols; reusing the existing EMA `similarity_codebook` (D1) stays within
  it — do not mint a new learnable CS codebook.
- **`bin/Spaces.py` line numbers** drift as code is added; the refs here are current as of this date —
  re-grep before editing.

---

## Existing code (revert; do not recreate as-is)

In the working tree (uncommitted), `bin/Spaces.py` `ConceptualSpace` gained `_edge_tables`/`add_edge`/
`concept_edges`/`_drop_concept_edges` (`:13591`-`:13652`), `_rebuild_edge_pools`/
`scatter_concept_event` (`:13653`-`:13705`), `_symbol_stack_ranges`/`_stacked_basis` (`:13707`-`:13737`),
a `retire_concept` edge hook (`:13793`), and `test/test_cs_edge_store.py` (10 tests).

- **Definitely discard:** `_symbol_stack_ranges` / `_stacked_basis` (cross-space `[PS|WS|SS]`
  dictionary — wrong model, violates the dataflow rule).
- **Re-derive from this plan (don't inherit the integration assumptions):** the edge store + scatter
  kernel are the right *design* (Phases 1-2) but were CS-resident with cross-space basis assumptions;
  rebuild them per the `.forward()`-safe sourcing here.
- Recommend reverting all of these so the tree is clean for the fresh build (the user does git; I can
  revert the non-git working-tree edits on request).

## Key files

- `bin/Spaces.py` — `ConceptualSpace` (`forward` for PS/WS$\to$CS, the relation tables `:13539`+,
  `similarity_codebook` `:12584`, `bind_streams` `:15276`, `_build_symbol_leg` to retire `:15188`,
  `grow_to` `:3032`).
- `bin/Models.py` — parallel body + slot-in (`:6562`, `:6677`), `_model_symbolSpace` stamp to retire
  (`:6055`), `serial`/`symbolicOrder` derivation (`:619-653`), legacy SBOW (`:3310-3351`).
- `bin/Language.py` — `SymbolSpace.forward`/`reverse` + the dataflow-rule comment (`:12695-12718`); SS
  `.what` codebook under `<symbolTower>` (`:9189-9197`).
- `bin/embed.py` — `conceptual_sbow_loss_codes` (`:1728-1801`).
- `data/model.xsd`, `data/MM_substitution.xml`, `bin/data.py:830` — config + dataset scaffolding.

## Harness

- Targeted test: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/<file> -q`
  (`test/` on the path for the `from test_basicmodel import _populate_test_config` helper).
- Full gate: `make test`; `RUN_SLOW=1 make test_all` for heavy configs.
- iCloud evicts the venv — see `memory/icloud-venv-eviction` for the force-reinstall recovery.
