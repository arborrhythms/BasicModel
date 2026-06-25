# Part B — Conceptual Similarity Space: implementation handoff

**Date:** 2026-06-25
**Companion design doc:** [doc/plans/2026-06-23-conceptual-similarity-space.md](2026-06-23-conceptual-similarity-space.md) (the locked design — read it first for the *why*)
**Status:** Part A (the `[0,1]` percept-cube migration) is **finished**; this is the handoff for Part B.
**Provenance:** grounded by a 5-surface read-only mapping workflow; central claims verified against source (file:line refs below are the *verified* ones — the design doc's refs are stale, trust these).

---

## TL;DR — the one fact that changes everything

**Part B is NOT "nothing built."** A prior session (in-code date **2026-06-24**) landed almost the entire SBOW *spine* — the codebook, the loss kernel, the consumer, the parallel-pass slab park, the config gate, two driver configs, and the dataset — all gated dark (`conceptualSimilarityScale=0` default) and **byte-identical when off**.

**But it is forward-disconnected and collapses.** The code says so itself, at [bin/Models.py:3321-3328](../../bin/Models.py):

> NOTE (2026-06-24): the ConceptualSpace has NO forward codebook — its `.what` is a *computed* Tensor from the percept binding, and `subspace.codebook()` is `None`. So this `similarity_codebook` is forward-disconnected: situating it cannot differentiate the (computed, collinear) conceptual representation. Real co-location needs the percept→concept binding to differentiate upstream (or a commitment loss training that binding), not a codebook to situate.

So: the SBOW gradient trains a **side-table that nothing downstream reads**. The machinery *fires* but *collapses*. **Stage 0 (forward-connect the concept code) is the real substance of Part B**; everything else is dark scaffolding until it lands.

---

## Locked design (one paragraph)

Part B builds a **Gärdenfors similarity layer** sitting between mereonomy (part/whole) and taxonomy (is-a). Zero-order **word-concept (A)** and **object-concept (B)** get **located codes** — vectors with a definite position — turning concept identity from a row *index* into a *code*. The codes live on a **ConceptualSpace `subspace.what`** codebook (Alec's dataflow rule: the codebook rides the subspace and is *passed through* forward; Spaces are operators). That subspace is **passed to SymbolSpace**, where symbols are **references** into it (1:1 snap, detached — **not a copy**). **`meta(C)` stays a set-based taxonomy** — is-a is *not* geometrized ("Gärdenfors for *how-alike*, sets for *what-kind-of*"); only A/B located codes are situated. Situating is by **SBOW (CBOW-NS)** at the **parallel pass** (`symbolicOrder=0`, all words co-present): in-group rotated toward the gaussian-weighted leave-one-out neighborhood centroid (the **+pode**), out-group **SGNS-repelled** from random negatives (*not* pulled to the −pode, which is a concept's negation). Geometry = **plain unit ball**, antipode = −pode (no torus / RP^D). The **radius (certainty) is left untouched; only the angle (meaning) rotates.** The `perceptDim`/`conceptDim`/`symbolDim` decouple (design §8) is **Part B-2 — HIGH RISK, deferred.**

---

## What already exists (built, dark, byte-identical off — all uncommitted)

| Component | Location | Notes |
|---|---|---|
| SBOW loss kernel | [embed.py:1728-1801](../../bin/embed.py) `conceptual_sbow_loss_codes` | **Exact locked geometry**: in-group attract to detached row-normalized leave-one-out `pode_dir` (tangential/rotation-only grad), out-group SGNS repel from `neg_k` random pool negatives, plain unit ball, gaussian-weight option. Defaults `gaussian=False`, `neg_k=None`. |
| Standalone codebook | [Spaces.py:12584-12594](../../bin/Spaces.py) `ConceptualSpace.similarity_codebook` | `Codebook(use_dot_product=True, monotonic=True, invertible=False)`, width = `outputShape[1]` (concept content dim). **Held as a plain attr — NOT handed to SubSpace**, so `codebook_slot` stays None and `.event` forward is byte-identical. Built **RNG-neutral** (save/restore global RNG around `create()`). |
| Consumer glue | [Models.py:3310-3351](../../bin/Models.py) `conceptual_sbow_loss` | No-grad cosine-argmax snaps the parked slab to nearest `similarity_codebook` row, gathers grad-bearing rows as the SBOW window, calls the kernel with `pool=rows, scale=1.0`. **Does NOT pass `gaussian`/`sigma`/`neg_k`.** |
| Park hook | [Models.py:6562-6572](../../bin/Models.py) | `self._cs_parallel_slab = CS_sub.materialize().detach()` taken from `cs.forward` at stage 0, gated `t==0 and not serial and conceptual_similarity_scale>0`. The "park slab after `cs.bind_streams`" parallel-pass site. |
| runBatch consumer | [Models.py:4289-4299](../../bin/Models.py) | Adds `_cscale*csbow` to `totalLoss` + `TheError.add("conceptual_sbow")`, gated `train and not serial and scale>0`. |
| Config knob | [model.xsd:721](../../data/model.xsd) `<conceptualSimilarityScale>` | nonNegativeFloat, minOccurs=0; read at [Models.py:991-992](../../bin/Models.py) → `loss.conceptual_similarity_scale`, **default 0.0 = OFF**. |
| Driver configs (untracked) | `data/MM_concept_sim.xml`, `data/MM_substitution.xml` | Both set `conceptualSimilarityScale=0.1`, `symbolicOrder=0`, `subsymbolicOrder=3`, `dataType=embedding`. The smoke vehicles. |
| Dataset | [data.py:830](../../bin/data.py) `loadSubstitution` (dispatch 547-548) | noun×verb substitutability grid. |
| SS reference plumbing | [Spaces.py:15032-15118](../../bin/Spaces.py) `_build_symbol_leg` | Copies **WholeSpace meta-codebook** rows (`W_ws` via `_relation_store`) into `SS.subspace.what` at a stable 1:1 `_concept_ss_row` (first-seen), detached/no-grad, width-clipped `cws=min(W_ws,W_ss)`. Called from `bind_streams` only when `n_streams>=3`. **⚠ Today it references WS meta rows, NOT the CS located codes.** |
| SS tower codebook | [Language.py:9189-9197](../../bin/Language.py) | `SS.subspace.what = _sym_cb.create(1, nSymbols, symbol_dim)`, built only under `<symbolTower>` (default off → empty → byte-identical). |
| CS→SS back-ref | [Models.py:6053-6058](../../bin/Models.py) | Stamps `_cs._model_symbolSpace` per-stage, gated `symbolTower`; read by `_build_symbol_leg` at [Spaces.py:15076](../../bin/Spaces.py). |
| C=meta minting (set-based, relation-only) | [Spaces.py:13800-13843](../../bin/Spaces.py) `create_word_object_meta` | Mints A=`new_concept` (word parts), B=`new_concept` (`{ATOM}`/`{UNIVERSE}` poles), C=`reify_concept(A,B)`. **No vectors, no codebook rows** — set-valued `_concept_parts`/`_concept_wholes` (dict[int→set]). |

---

## Staged plan

### Stage 0 — forward-connect the concept code  ⟵ **THE blocker. Do first.**
**Goal:** make the located concept code part of the **forward representation** so SBOW can actually differentiate it. Today `similarity_codebook` trains in isolation and nothing reads it. Two options the code itself lists ([Models.py:3325-3327](../../bin/Models.py)): **(a)** mint a real CS `.what` *content* codebook on the subspace in `_build_what_basis` and route the percept→concept binding through it (snap); **(b)** add a **commitment loss** training the percept→concept binding toward the situated codes.
**Entry points:**
- [Spaces.py:12596](../../bin/Spaces.py) `_build_what_basis` (returns None unless `codebook_mode=="project"`; the project branch is a `ProjectionBasis`, *not* a content codebook — this is where a `conceptDim`-wide content codebook would be minted so it rides the subspace).
- [Models.py:3321-3328](../../bin/Models.py) (the forward-disconnect NOTE — the precise statement of the blocker + the two fix options).
- [Spaces.py:12584-12594](../../bin/Spaces.py) (the standalone `similarity_codebook` — decide: promote ONTO `subspace.what`, or replace. **Preserve the RNG save/restore.**)
**Verify:** train `data/MM_concept_sim.xml` a few epochs; assert the SBOW gradient reaches the **forward** concept representation (the percept→concept binding output changes — e.g. cosine between two substitutable concepts' forward codes increases), not just the side codebook rows. **Byte-identical when `conceptualSimilarityScale=0` (full suite green, ~2843 baseline).**

### Stage 1 — real multi-concept sibling slab
**Goal:** ensure the parked `_cs_parallel_slab` holds **≥2 distinct co-present** concept codes (the substitutability window). Today the parallel-pass slab tends to be a single collinear bound whole → the leave-one-out centroid is degenerate and codes collapse onto one axis. The kernel returns None when N<2 ([embed.py:1766](../../bin/embed.py)), so a degenerate slab silently no-ops.
**Entry points:** [Models.py:6562-6572](../../bin/Models.py) (the park site), [Models.py:3336](../../bin/Models.py) (the `slab.shape[1]<2` silent no-op guard).
**Verify:** instrument the slab on a multi-word sentence; assert `shape[1]>=2` and rows non-collinear (pairwise cosine < 1); confirm the loss returns non-None.

### Stage 2 — thread the locked SBOW hyperparameters
**Goal:** the design specifies a **gaussian-weighted** neighborhood centroid, but the consumer calls the kernel with `gaussian=False`, no `sigma`/`neg_k` ([Models.py:3350-3351](../../bin/Models.py)) → uniform leave-one-out. Thread `gaussian=True` + `sigma` + `neg_k` through (ideally as config knobs). Resolve the design §7 window/sibling definition.
**Entry points:** [Models.py:3350-3351](../../bin/Models.py) (the call site), [embed.py:1773-1790](../../bin/embed.py) (the gaussian kernel + neg sampling — already implemented, not invoked).
**Verify:** unit-test the kernel `gaussian=True` vs `False` on a fixed window (centroid weighting differs); train with `sigma` set, confirm position-local siblings dominate.

### Stage 3 — repoint the SS reference at the CS located code
**Goal:** per design point 2, SS symbols must reference the **CS located codes**, not the WholeSpace meta rows. Today `_build_symbol_leg` sources from `W_ws`. Re-source to the forward-connected CS code (Stage 0), keeping the 1:1 `_concept_ss_row` map and **detached/no-grad** reference semantics (Alec forbids a divergent copy / correlation-loss-on-codebook). Honor the dataflow rule by **passing the CS subspace** into the leg through forward rather than dereferencing the stashed `_model_symbolSpace` Space pointer.
**Entry points:** [Spaces.py:15050-15062](../../bin/Spaces.py) (the `W_ws` source), [Spaces.py:15069-15075](../../bin/Spaces.py) (`_concept_ss_row`), [Spaces.py:15095-15112](../../bin/Spaces.py) (the no_grad copy — keep detached), [Models.py:6053-6058](../../bin/Models.py) (the `_model_symbolSpace` stamp — candidate to replace with a forward-passed subspace), [Language.py:9189-9197](../../bin/Language.py) (the SS `.what` slot).
**Verify:** run a `<symbolTower>`+`<conceptualSimilarityScale>` config; assert `SS.subspace.what` rows match the CS located codes (1:1, `requires_grad=False`), and grad flows **only** to the CS code, never through the SS copy.

### Stage 4 — located codes for A=word, B=object
**Goal:** give A and B located codes at minting (the indices→codes substance), keeping C=meta set-based. Today `create_word_object_meta` mints all three relation-only.
**Entry points:** [Spaces.py:13800-13843](../../bin/Spaces.py) (mint A/B located codes; leave C=`reify_concept` set-based), [Spaces.py:13598](../../bin/Spaces.py) (`reify_concept`, 2-arity, unchanged for C).
**Verify:** after the substitution grid, A and B have codebook rows while C stays relation-only; substitutable nouns/verbs co-locate (high cosine), non-substitutable do not.

### Stage 5 — meta = set-based taxonomy store  *(OPTIONAL, deferred — independent of SBOW geometry)*
**Goal:** meta is strictly **binary** everywhere today (`taxonomy[meta]=[ps,ws]` 2-tuple, `meta_pair_to_idx` pair-keyed, `reify` 2-arity). A set-based taxonomy (`frozenset(concept refs)→meta` + property-at-level subsumption) has no home. Either extend `insert_meta`/`reify_concept` to set-arity, or repurpose the already-set-valued `_concept_parts`/`_concept_wholes`.
**Entry points:** [Spaces.py:17017](../../bin/Spaces.py) (`insert_meta`, binary), [Spaces.py:13539-13560](../../bin/Spaces.py) (`_concept_parts`/`_concept_wholes` — candidate home), [Spaces.py:13703](../../bin/Spaces.py) (`synthesize_higher_order` — existing frozenset-keyed set-arity primitive).
**Verify:** set-keyed meta store; is-a subsumption over feature-sets queryable and **non-geometric** (no distance); part-whole/is-a transitivity fallacy impossible by construction.

### Stage 6 — decouple perceptDim / conceptDim / symbolDim  *(HIGH RISK, deferred — design §8)*
**Goal:** promote `perceptDim`/`conceptDim`/`symbolDim` to first-class knobs so a thin orthographic mereonomy pairs with a fat conceptual space.
**⚠ Terminology trap:** source `symbol_dim` ([Models.py:5520](../../bin/Models.py)) = **WholeSpace nWhat**, NOT SymbolSpace (whose `symbolDim` = `_sym_cb`, **already structurally decoupled** at [Language.py:9193](../../bin/Language.py)). The genuinely-tied pair is **perceptDim↔conceptDim**.
**⚠ `_bind_fit` is a zero-pad/truncate CLIP**, not a learned expander ([Spaces.py:14879-14897](../../bin/Spaces.py)) — a real `perceptDim<<conceptDim` needs an actual PS→CS embedding.
**Entry points:** [Models.py:5511-5521](../../bin/Models.py) (build dim chain) **+** [Models.py:10000-10038](../../bin/Models.py) (validator copy — **must move in lockstep**); [Models.py:10035-10110](../../bin/Models.py) (`effective_concept_dim==symbol_nwhat` invariant + relaxes); [Models.py:10134-10146](../../bin/Models.py) (flat-slab `ps_slab==cs_slab` invariant); [Layers.py:1513](../../bin/Layers.py) (`ConceptualCombine` single content_dim); [Spaces.py:18038](../../bin/Spaces.py) (`decode_to_concept`, identity because `symbol_dim==concept_dim`); [architecture.py:20-33](../../bin/architecture.py) (`_CANONICAL_SHAPE` band).
**Verify:** set `conceptDim != perceptDim`; assert the percept→concept step is a learned expansion (not zero-pad), the bind is invertible on the new width, no validator crashes. **Plan checkpoint migration** (state_dict keys are sized to chained dims).

---

## Risks / traps

1. **Forward-disconnect is the central, self-documented blocker.** Do NOT treat the existing `similarity_codebook` plumbing as "done." Stage 0 is the real work.
2. **Three candidate code homes risk divergence:** `similarity_codebook` (off-subspace, what SBOW reads) vs a future `CS.subspace.what` content codebook (what the plan + `_build_symbol_leg` expect) vs the WS meta codebook (what `_build_symbol_leg` actually copies today). **Pick ONE home** (Alec's rule: codebook on the subspace) and repoint all readers.
3. **S3 "CS codebook rejected" reversal is scoped:** the original rationale ([Spaces.py:13845-13858](../../bin/Spaces.py)) denied CS a `.what` codebook because symbols/concepts were relation-only. Part B reverses this **only** for the A/B located-code *geometry*; the A/B/C taxonomy **table stays set-based**. Do not over-reverse into geometrizing is-a.
4. **Dim-decouple terminology trap** (see Stage 6): an agent that decouples "symbol_dim" via `decode_to_concept`/`VerbLayer`/the WS invariant is working the **wrong axis**.
5. **RNG-neutrality:** any new always-built CS codebook **must** preserve the `torch.get_rng_state()`/`set_rng_state()` guard ([Spaces.py:12584/12591](../../bin/Spaces.py)) or it shifts every config's init and breaks byte-identical-off (tips XOR_exact).
6. **Radius drift:** the codebook is unit-norm (`use_dot_product`) with no separate radius column; a finite tangential step still drifts radius 2nd-order (grad ~1/r, worst small-r). Exact decoupling needs `d_c` (unit dir) + `r_c` (scalar) stored separately, realized = `r_c·d_c`.
7. **Zero test coverage:** grep of `test/` for `conceptual_sbow` / `similarity_codebook` / `loadSubstitution` / the driver configs returns nothing. Add a consumer test (config-driven train step: codebook rows move, radius stays stable, substitutable concepts co-locate).
8. **Serial-only guard:** park hook (6568), consumer (3334), call site (4292) all bail on `self.serial` — SBOW runs **only** at `symbolicOrder=0`. Smoking a serial config is a silent no-op.
9. **Gate families are independent:** the SS reference path needs **both** `<symbolTower>` (for the SS `.what` slot + the CS→SS stamp + 3-stream bind) **and** `<conceptualSimilarityScale>`. Stage 3 must reconcile them.
10. **`_build_symbol_leg` is a `@torch.compiler.disable` eager island** ([Spaces.py:15043](../../bin/Spaces.py)) with host-side dict iteration. Keep re-sourced located-code gathers host-side; don't move into the compiled path without checking the fullgraph relax.
11. **All uncommitted** (Alec commits himself): `bin/Models.py`, `bin/Spaces.py`, `bin/embed.py`, `bin/data.py`, `data/model.xsd` modified; the two `MM_*` configs untracked. A fresh session must not assume this surface is on `main`.

---

## Open questions for Alec (resolve before / during Stage 0)

1. **Located-code home:** promote the standalone `similarity_codebook` **onto** `CS.subspace.what` (so it's forward-connected + passed to SS per the dataflow rule), or **mint a fresh** content codebook in `_build_what_basis` and retire `similarity_codebook`? *(The plan text + `_build_symbol_leg` expect `subspace.what`; the build chose the side-attr.)*
2. **Forward-connect mechanism (Stage 0):** route the percept→concept binding through a **snapped** CS `.what` codebook, **or** add a separate **commitment loss** training the binding toward the situated codes? *(Both listed at Models.py:3325-3327 — which do you want?)*
3. **Concept→symbol snap:** keep `_concept_ss_row` strictly **1:1** (current), or relax to **many-to-one VQ** as the design hints?
4. **SBOW window:** whole parallel-pass window vs a bounded gaussian neighborhood; and for grounded objects, the spatiotemporal co-presence set.
5. **Dense pre-snap vs discrete:** situate the pre-snap **continuous** code with a low-freq/EMA codebook (smooth mineable location AND snaps to a symbol) — is the EMA codebook the right discretizer, and how low-frequency?
6. **Convexity pressure:** explicit Criterion-P / convexity regularizer, or let SBOW + EMA produce convex regions implicitly?
7. **Dim decouple scope:** wanted now (thin-form/fat-meaning for text), or strictly deferred until Stage 0–4 land?
8. **Dataset:** is the noun×verb substitution grid the right verification vehicle, or does B need a richer corpus where substitutability is the actual learning signal?

---

## First step

Start at [Models.py:3321-3328](../../bin/Models.py) (the forward-disconnect NOTE) and resolve **Stage 0**. Concretely: **decide the code home** (Q1) and the **forward-connect mechanism** (Q2), then wire the percept→concept binding to differentiate that code. **Verify** by training `data/MM_concept_sim.xml` (`conceptualSimilarityScale=0.1`, `symbolicOrder=0`) and asserting the SBOW gradient changes the **forward** conceptual representation (substitutable concepts' forward codes move closer) — while keeping byte-identical behavior when `conceptualSimilarityScale=0` (full suite green). **Everything downstream is dark scaffolding until this lands.**
