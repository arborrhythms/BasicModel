# Sparse-Coding Conceptual Space — implementation handoff

**Date:** 2026-06-27
**Supersedes/continues:** the approved plan `~/.claude/plans/that-sseems-coorect-please-sequential-corbato.md`
and the original Part B handoff `doc/plans/2026-06-25-part-b-handoff.md`.
**Status:** Phase 1 (COO edge store) and the Phase 2 *kernel* (scatter-add SpMM + dictionary) are
**implemented and green** (10 tests, byte-identical: nothing reads the table in forward yet). The
remaining work — the CS$\to$SS forward-mediated transform, the forward wiring, and Phases 3-6 — is
specified below for a fresh conversation to finish.
**Provenance:** grounded by repeated read-only mapping (3 Explore digests + a Plan agent) and verified
against source; the file:line refs below are current (note: `bin/Spaces.py` lines shifted ~`+127`
from the Phase-1/2 edits — trust the refs here, not older docs).

---

## TL;DR — the locked idea

> **A Concept is a sparse, weighted linear combination over a basis of *symbols*.**

- **Storage:** a three-column COO table `(ConceptIndex, SymbolIndex, Weight)` — space-optimal (exactly
  the nonzeros, never a dense `[N x millions]` matrix), extensible by appending. `Weight` is a
  learnable `Parameter`.
- **Compute (PS/WS $\to$ CS):** a differentiable scatter-add SpMM
  `concept = index_add_(Basis[SymbolIndex] * Weight, ConceptIndex)` (`index_select`+`index_add_`, **no
  `torch.sparse`** — MLX/executorch export safety). Differentiable in *both* the weights and the
  constituent codes, so the concept code is **forward-connected by construction** (this dissolves the
  old "Stage 0 forward-disconnect" blocker at `bin/Models.py:3321-3328`).
- **Symbols = concept rows.** The SS (symbol) codebook is *row-aligned* with the CS (concept)
  codebook: symbol $i$ is the unmuxed view of concept $i$. A concept is built out of symbols, which
  are concepts **from the previous ramsified order** — the recursion is the ramsified-order hierarchy.
- **CS$\to$SS (concept $\to$ symbol / representation, the percept):** the reverse transform — a sparse
  autoencoder: `PS/WS -(W)-> CS -(snap)-> SS -(W^T)-> percept`.
- **Mux principle:** PS/WS/SS subspaces are **unmuxed** (keep `.what`/`.where`/`.when` as separate
  channels; where/when stamped at `materialize()`). The **CS subspace is muxed**
  (`.what (+) .where (+) .when`, width `muxedSize`). The scatter-add gathers unmuxed symbol rows and
  produces a **muxed** concept — muxing *is* the perceptual$\to$conceptual transition.

**Activation is by `symbolicOrder`, not a config gate.** The symbolic loop (CS$\to$SS, ramsified
concept-building) runs in **parallel mode** (`serial=false`) with `symbolicOrder>0`. At
`symbolicOrder=0` the edge table is empty and the forward falls back bit-for-bit. There is **no new
`<...>` gate** — `symbolicOrder` controls it.

All git operations are left to the user (the working tree carries uncommitted Part B scaffolding on
`main`).

---

## What is BUILT and GREEN (Phases 1 + 2-kernel)

All in `bin/Spaces.py` on `class ConceptualSpace`, tested in `test/test_cs_edge_store.py` (10 tests,
pristine). Nothing touches the forward path yet $\Rightarrow$ byte-identical everywhere.

| Piece | Location | Notes |
|---|---|---|
| `_edge_tables()` | `bin/Spaces.py:13591` | lazy-init `_edge_concept`/`_edge_symbol`/`_edge_weight` lists + `_edge_index` dedup dict + `_edge_dirty`. Mirrors `_concept_tables` (`:13539`). |
| `add_edge(concept, symbol, *, weight=1.0)` | `bin/Spaces.py:13603` | idempotent per `(concept,symbol)`; sets dirty; returns COO row. |
| `concept_edges(concept)` | `bin/Spaces.py:13621` | `(symbol, weight)` list, sorted by `repr`. |
| `_drop_concept_edges(concept)` | `bin/Spaces.py:13632` | rebuild COO + dedup excluding a concept; **pure no-op when the table is unbuilt** (byte-identical). |
| `retire_concept` hook | `bin/Spaces.py:13793` | added `self._drop_concept_edges(s)` to the existing `retire_concept` (`:13783`). |
| `_rebuild_edge_pools()` | `bin/Spaces.py:13653` | rebuild-on-dirty: long index columns + learnable `_edge_weight_pool` `Parameter` (grown **tail-preserving**, so trained weights survive new edges). Never per-forward. |
| `scatter_concept_event(n_concepts, basis, *, normalize=True)` | `bin/Spaces.py:13679` | the SpMM; `normalize=True` divides by per-concept degree so **uniform == mean**. Differentiable in weights + basis. |
| `_symbol_stack_ranges(n_ps, n_ws, n_ss=0)` | `bin/Spaces.py:13707` | `[PS|WS|SS]` offsets `(off_ps, off_ws, off_ss, total)`. |
| `_stacked_basis(ps_W, ws_W, ss_W=None, *, muxed_dim=None)` | `bin/Spaces.py:13715` | concat the codebook `.what` rows; lift each unmuxed what-row into the muxed `[what|where|when]` frame (what in leading slots, band zero-padded; where/when stamped from live events at scatter time). |

**Tests** (`test/test_cs_edge_store.py`; helper `_cs(nS)` at `:27` instantiates a bare
`ConceptualSpace` via `_populate_test_config`, no full model): `test_add_edge_dedup`,
`test_add_edge_carries_weight`, `test_retire_drops_edges`, `test_scatter_add_matches_dense`,
`test_scatter_is_differentiable_in_weights_and_basis`, `test_scatter_uniform_is_mean`,
`test_rebuild_preserves_trained_weights_on_grow`, `test_symbol_stack_ranges`,
`test_stacked_basis_concat_and_lift`, `test_percept_to_concept_transform_over_dictionary`.

So the **PS/WS $\to$ CS sparse-matrix transform is complete and verified end-to-end** over a real
constituent dictionary.

---

## THE IMMEDIATE NEXT TASK — CS$\to$SS, forward-mediated, in the symbolic loop

This is the substance the handoff picks up. Two **hard constraints from the user**:

1. **CS$\to$SS runs in the SymbolicOrder loop, in PARALLEL mode** (`serial=false`, `symbolicOrder>0`).
   That is the parallel body `_forward_body` (the `for t, stage` loop; `cs.forward` at
   `bin/Models.py:6562`, `cs.bind_streams` at `:6677`), **not** the per-word serial path
   (`_forward_body_per_word`). `self.serial` derives from `symbolicOrder>0` by default
   (`bin/Models.py:619-653`), so the driver config must set `<serial>false</serial>` explicitly
   alongside `symbolicOrder>0`.
2. **Cross-space interaction goes ONLY through `.forward()`** (the dataflow rule: Spaces are
   operators; pass data through forward; do NOT dereference stashed Space pointers or reach into
   another space's codebook).

**The current code VIOLATES rule 2 — and reverses a deliberate prior choice.** Today the symbol leg is
*CS-mediated*: `bin/Models.py:6671-6676` documents *"SymbolSpace.forward_symbol retired ... CS, which
sees all three towers, does it."* `ConceptualSpace._build_symbol_leg` (`bin/Spaces.py:15188`) reads a
stashed `_model_symbolSpace` pointer (stamped once at `bin/Models.py:6055-6058`, gated `symbolTower`)
and **mutates `SS.subspace.what` in place under `no_grad`**, sourcing rows from the **WholeSpace meta
codebook**. That is exactly the reach the user forbids. **Confirmed with the user: reverse it.**

There is already a clean precedent for the rule in the code: `bin/Language.py:12695` —
*"CS interacts with SymbolSpace ONLY through these"* — `SymbolSpace.forward(snap)` / `reverse(snap)`
(`bin/Language.py:12702`, `:12711`) dispatch grammar over a CS-provided STM snapshot. CS$\to$SS should
follow the same shape.

### Refactor to implement (TDD)

1. **`SymbolSpace.forward_concept_to_symbol(concept_subspace)` (new, `bin/Language.py` near `:12709`)**
   — receives the concept (a SubSpace / codes) **through the argument** and returns the symbol leg
   SubSpace `[B, N, D]`. The symbol is the **row-aligned representation of the concept** (symbol $i$ =
   the `.what` slice of concept $i$): build it from the **concept codes**, *not* from WS meta rows and
   *not* via `_model_symbolSpace`. SymbolSpace mutating its **own** `subspace.what` is allowed (a
   space owning its codebook); reaching into *another* space's is not. Keep it
   `@torch.compiler.disable` if host-side dict iteration remains (matching the existing islands), but
   prefer a tensor gather.
2. **Wire it in the parallel body** (`bin/Models.py`, right after `CS_sub = cs.forward(...)` at
   `:6562`, before `cs.bind_streams` at `:6677`):
   ```python
   SS_sub = (self.symbolSpace.forward_concept_to_symbol(CS_sub)
             if (self.symbol_tower and not self.serial
                 and self.symbolicOrder > 0) else None)
   full_t = cs.bind_streams(ps_t, WS_sub, CS_sub, SS_sub=SS_sub,
                            seed_payload=(seed_payload if t == 0 else None))
   ```
   (`bind_streams` at `:15276` already accepts `SS_sub`; `:15341` fits it as the 3rd peer leg.)
3. **Retire the violation:** drop `_build_symbol_leg`'s internal `_model_symbolSpace` dereference + the
   WS-meta `no_grad` copy. Either delete `_build_symbol_leg` (now that the caller passes `SS_sub`) or
   make its 3-stream branch (`bin/Spaces.py:15341` area) consume only the handed-in `SS_sub`.
4. **Byte-identical:** with `symbolTower` off (default), `n_streams==2`, no SS leg $\Rightarrow$ the
   2-stream path is untouched. Verify with a `torch.equal` test on a `symbolicOrder=0` config.

### How CS$\to$SS connects to the sparse machinery

The symbol leg = the concepts' row-aligned codes. Under sparse coding the concept code is the
scatter-add (`scatter_concept_event` over `_stacked_basis`); the **SS codebook is its row-aligned
unmuxed view** (Phase 4 EMA-snaps the muxed `similarity_codebook` to give the discrete symbol
identity, `bin/Spaces.py:12584` — already always-built, has `self.vq` EMA). So
`forward_concept_to_symbol` produces the symbol slab by gathering the (snapped) concept rows — the
reverse/decode direction of the same dictionary.

---

## Remaining staged plan (after CS$\to$SS lands)

- **Phase 3 — Situate / forward-connect.** The scatter-add is already differentiable (forward-connect
  by construction). Add the substitutability *situating* signal via `conceptual_sbow_loss_codes`
  (`bin/embed.py:1728-1801`) pointed at the grad-bearing scatter-add codes (mutually exclusive with
  the legacy forward-disconnected snap at `bin/Models.py:3310-3351`). Register `_edge_weight_pool`
  into the optimizer (`self.params`) here. Confirm the co-presence window source in the parallel
  symbolic body (the STM slab of co-present concepts), since the legacy park ran differently.
- **Phase 4 — Concept VQ = EMA-snap.** Reuse `similarity_codebook` (`bin/Spaces.py:12584`, muxed,
  `nDim=muxedSize`, `self.vq` EMA) as the concept VQ; EMA-snap the scatter-add codes under `no_grad`;
  `grow_to` (`bin/Spaces.py:3032`) lockstep with the SS codebook to keep row-alignment;
  `_concept_ss_row` (`bin/Spaces.py:15225`) is the stable concept$\to$row map.
- **Phase 5 — Populate edges at mint (decoupled).** Add a small **ungated** population path (decoupled
  from `<mereologyRaise>`) that calls `add_edge` from the mint sites: `create_word_object_meta`
  (`bin/Spaces.py:13956`), `synthesize_higher_order`, `relate`/`reify_concept`, `_populate_cs_symbols`
  — using `_symbol_stack_ranges` to map a PS/WS code to its global stacked index. **Min-support
  invariant:** a concept is eligible for the scatter only with `>=1 PS-col + >=1 WS-col`, **or**
  `>=2 SS-cols`; otherwise fall back to the current content. Assert `symbol < total` and within range.
- **Phase 6 — Driver config + verification.** `data/MM_sparse_concept.xml` (sibling of
  `data/MM_substitution.xml`) with **`symbolicOrder>=1`**, **`<serial>false</serial>`**,
  `subsymbolicOrder=3`, `conceptualSimilarityScale=0.1`, `symbolTower=true`; reuse `loadSubstitution`
  (`bin/data.py:830`). Verify: gradient reaches the forward concept code; substitutable concepts'
  rows co-locate (rising cosine); full `symbolicOrder=0` suite byte-identical.

---

## Resolved design decisions (do not re-litigate)

- **Three-column COO table** `(ConceptIndex, SymbolIndex, Weight)`; `Weight` learnable, **default
  uniform 1.0**, normalized by `1/deg` so uniform == mean.
- **No config gate** — `symbolicOrder=0` (empty table) is "off"; `symbolicOrder>0` + `serial=false`
  is "on". Driver config sets `<serial>false</serial>` explicitly.
- **Symbols = stored concept rows** (row-aligned SS$\leftrightarrow$CS); the SS basis resolves to the
  **stored** concept-codebook rows (grad-bearing params), not a recursive recompute.
- **CS$\to$SS via `SymbolSpace.forward(...)`** (reverse the documented CS-mediated symbol leg).
- **Edge population decoupled** from `<mereologyRaise>`.
- **Scatter feeds the concept content; existing combine still handles band re-mux / invertibility**
  for now. Whether the sparse code eventually *replaces* the learned combine is future scope.

---

## Traps / risks

1. **Line shift.** The Phase-1/2 edits added ~127 lines before `bin/Spaces.py:13738`; everything after
   shifted. The refs in this doc are current; older docs (incl. `2026-06-25-part-b-handoff.md`) are
   stale for `Spaces.py`.
2. **Mux seam (highest risk).** PS/WS/SS codebooks are unmuxed (what-only); the concept is muxed.
   `_stacked_basis` lifts via zero-pad; the **where/when must be carried from the live materialized
   events at scatter time**, exactly as `_build_symbol_leg` carried the band
   (`bin/Spaces.py:15255-15269`). Land the dense-match oracle (`test_scatter_add_matches_dense`) before
   trusting the lift.
3. **Byte-identical is the gate.** `symbolicOrder=0` configs (most of the suite) MUST stay bit-for-bit
   identical: `make test` green. The off-path is the `n_streams==2` branch (`bin/Spaces.py:15296`) and
   the empty edge table.
4. **`_model_symbolSpace` has other readers** (`bin/Spaces.py:8830, 8963, 9048, 13070`) — the autobind
   path uses it too. Retiring it for the SS *leg* must not break those; scope the change to the
   `_build_symbol_leg` reach only.
5. **`forward_concept_to_symbol` must not desync reconstruction.** `unbind` reads
   `CS_sub._bind_carrier = full` (`bin/Spaces.py:15360`). Keep the bind/reverse contract intact; the
   COO edges are the sparse-coding reconstruction source (read the edges), separate from `unbind`.
6. **"CS codebook rejected"** (`bin/Spaces.py:14002-14010`) forbade a *learnable* CS
   `.what` codebook for relation-only symbols; Phase 4 reuses the *existing* EMA `similarity_codebook`,
   staying within that rationale — don't mint a new learnable CS codebook.

---

## Environment + how to run

The repo + `.venv` live under iCloud Drive and get **evicted** (imports fail despite pip "satisfied",
sometimes whole packages gone). Recovery that worked 2026-06-27 (see also
`memory/icloud-venv-eviction`): `pip install --force-reinstall --no-deps torch==2.12.0` (restores the
bundled `torchgen/`), then bulk `--force-reinstall --no-deps` the non-torch `pip freeze` set, then
`pip install -r requirements.txt` for fully-gone deps (`multiprocess`, `datasets`, ...). Run pip with
the sandbox disabled (needs network).

- **Targeted test (needs both dirs on path):**
  `BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test .venv/bin/python -m pytest test/test_cs_edge_store.py -q`
- **Full gate:** `make test` (`BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python test/test_report.py`);
  `RUN_SLOW=1 make test_all` for heavy configs.
- **Byte-identical idiom:** seed (`torch.manual_seed(0)`), build twice, diverge only in the feature,
  `torch.equal` on params/state (cf. `test/test_codebook_update_law.py:107-114`).

## Key files

- `bin/Spaces.py` — `ConceptualSpace`: edge store + scatter (`:13591-13737`), `bind_streams`
  (`:15276`), `_build_symbol_leg` to retire (`:15188`), `similarity_codebook` (`:12584`).
- `bin/Models.py` — parallel body + slot-in (`:6562`, `:6677`), `_model_symbolSpace` stamp (`:6055`),
  `serial`/`symbolicOrder` (`:619-653`), legacy SBOW (`:3310-3351`).
- `bin/Language.py` — `SymbolSpace.forward`/`reverse` + the dataflow-rule comment (`:12695-12718`);
  SS `.what` codebook under `symbolTower` (`:9189-9197`).
- `bin/embed.py` — `conceptual_sbow_loss_codes` (`:1728-1801`), reused as-is.
- `test/test_cs_edge_store.py` — the 10 green tests + `_cs()` helper.
- `data/model.xsd`, `data/MM_substitution.xml`, `bin/data.py:830` — config + dataset scaffolding.
