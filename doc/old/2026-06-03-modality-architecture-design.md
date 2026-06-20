# Modality Re-Architecture -- Design Spec

> **Supersedes** `doc/plans/2026-06-03-single-architecture-convergence-design.md` (the SS-promotion convergence). Consolidates the architecture convergence and the evolved modality semantics into one design.
>
> **Status:** design (approved scope: "modality re-architecture core" + the original-plan Phase 7 MorphologyLayer, Section 12). Next: `superpowers:writing-plans` -- one consolidated plan for all remaining work.
>
> **Already built:** convergence Phase 1 (unit-bracket `.when`: `WhenRangeEncoding.encode/forward/aspect_interval`, `AspectLayer` center-read, fixtures) -- **kept**. Phase 2's width-guard tests (`test/test_convergence_widths.py`) proved the SubSpace muxing is width-agnostic for where/when at width 2 with codebooks (useful evidence), but their tier shapes predate the revised table (Section 3); the new plan re-guards the actual shapes (CS `where=2/when=2` with an `.event` codebook; SS `where=0/when=0`). `bin/architecture.py` exists but its table is **revised** below.
>
> **Git:** Alec performs all git writes.

**Goal:** Converge on a single architecture in which the four subspace modalities have clean, cognitively-grounded semantics: `.where`/`.when` are properties of **occurrences/events**, not of symbols. They are architectural constants (not config options), ride the muxed event through IS $\to$ PS $\to$ CS, and are demuxed away at CS $\to$ SS so a symbol *qua* symbol carries neither. `.where` stops being a codebook row-index and becomes a manipulable spatial/relational extent. `.when` is the signed unit-bracket interval (already built), and lifting advances it.

---

## 1. Modality glossary (corrected against the code)

| modality | meaning | carrier |
|---|---|---|
| **`.what`** | content / prototype / meaning | codebook prototype rows (`Codebook.W`) |
| **`.where`** | the occurrence's **active spatial/relational extent** at its abstraction level: input span (word token), 3D extent (NP), path/control manifold (VP), spatial support (composed event). **Not** a codebook index. | muxed event slots `[nWhat : nWhat+nWhere]` |
| **`.when`** | the event/claim **interval**, incl. tense/aspect; signed, zero-centered, unit-bracket | muxed event slots `[nWhat+nWhere :]` |
| **`.activation`** | codebook **identity** -- which row/type was selected | see below (two distinct fields) |

Two fields are easy to conflate (verified in `bin/Spaces.py`):

- **`Codebook.activation`** (`bin/Spaces.py:2602`, shape `[B, codebookSize]`) -- the per-row snap score. This is the "set of activations": which codebook entries fired.
- **`SubSpace.activation`** (`bin/Spaces.py:4478`, shape `[B, N]`) -- a single **signed Degree-of-Truth scalar** $aP - aN \in [-1, +1]$. *Not* a set; the old 4-valued bivector was retired (2026-05).
- **`SubSpace._active`** (`bin/Spaces.py:4554`, shape `[B, N, M]`) -- the **selection mask/index**, overloaded by path: float presence flags $\{0, 1\}$ on the demuxed path (`_compute_active`, `:4778-4799`); int64 **codebook row-indices** on the muxed/codebook path (`set_forward_content`, `:5052-5076`).

`subspace.what[subspace._active]` $\to$ `[B, N, D]` is realized by `materialize()` as `codebook.W[_active[:, :, 0]]` ($D = $ `muxedSize`; `bin/Spaces.py:5732-5740`, `lookup` `:1790-1817`). The selector is **`_active`** (indices), not `.activation`.

**`.activation` and `._active` were never merged.** On the demuxed path they *coincide* (both keyed off `.what` being nonzero -- so "nonzero", not "non-NaN"; there are no NaN sentinels). On the codebook path they **diverge**: `_active` holds the indices while `activation` defaults to all-ones. Stale docs to fix as part of this work: the `SymbolicSpace` class docstring (`bin/Spaces.py:11809-11817`, claims the retired `nWhat=2`/`_active_payload` scheme) and the stray `_active_payload` token at `bin/Spaces.py:4558`.

**`.what` is the codebook; the event carries a selection.** `.what` is the one (unmuxed) codebook content slot -- not two forms. Prototypes are built at construction (`_build_what_basis`) and learned via the per-batch snap (`set_what` $\to$ `Codebook.forward`, `bin/Spaces.py:4971-5004`); **`materialize()` is read-only on the codebook** (`mode="event"` reconstructs `codebook.W[_active]`, `:5732-5740`, and never rewrites it; the `set_what` width-mismatch branch `:4990-4998` skips redundant per-batch writes in favor of selection-based reconstruction). A transformed / transmitted idea is carried on the `.event` as the `_active` codebook **selection** (indices) + `.where`(2) + `.when`(2) -- the `[B, N, 1+2+2]` form (the `1` is the dim-1 `.what` selection/activation). Because grammar operations **combine** selections, the result does not itself hold the prototypes needed for reconstruction; those are recovered from the **WordSubSpace** codebook (the lexicon) via the `.active` indices, and `materialize()` expands the selection back to the muxed event (a PS symbol activation `[B, 8, 1+2+2]` $\to$ CS STM `[B, 8, 1024]`). The non-muxed per-batch `.what` read (`materialize(mode="what")`) **is** used -- by the grammar ops (`bin/Language.py:4282`+) and inter-space content passing (`bin/Spaces.py:8289`+) -- so that path is kept. This reinforces Section 4: codebook *identity* lives in `.active` / `.activation`, which frees `.where` to be pure spatial extent.

---

## 2. The architecture: modalities as occurrence properties

**Cognitive grounding (the design rationale).** Concrete prototypes are muxed event-like representations -- *what* happened, *where*, and *when* are initially bound. Generalization abstracts over particular places/times into **event prototypes**; at that level there is no NP/VP split, only generalized events. NP and VP are *later* linguistic abstractions out of those event generalizations. So `.where`/`.when` belong to perceptual/conceptual occurrences, and a symbol acquires location/time only as instantiated in perception over its extension.

**Modality flow.** `.where`/`.when` mux into the muxed event at **PS $\to$ CS** and demux at **CS $\to$ SS**:

```
IS ---> PS ---> CS ---> SS ---> OS
[w/n]  [w/n]  [w/n]   [ -- ]  [ -- ]
              ^mux PS->CS  ^demux CS->SS
```

- **IS, PS, CS** carry `.where`(2) + `.when`(2) in the muxed event `[what | where | when]` (where/when are the last $nWhere+nWhen$ slots; `bin/Spaces.py:226`, `5281-5295`).
- **CS $\to$ SS demuxes** them: SS operates on a concept-dim `.what` slab only (this already happens -- `bin/Spaces.py:14724-14733`: "trim the extra columns here"). SS carries `.what` + `.activation`, no spatial/temporal location.
- **OS carries neither.**

This dissolves the prior blocker: the superseded design tried to *promote SS* to a `where=2/when=2` carrier, which desynced its `.what`-sized codebook. Here SS keeps its current symbolic shape; only CS needs to start *carrying* where/when in its event (today CS is `0/0` while PS is `2/2`, so where/when are dropped at PS $\to$ CS -- this design carries them through instead).

---

## 3. Canonical per-tier shape (REVISED)

`bin/architecture.py` `canonical_shape(section)` is **revised** from the superseded convergence values:

| section | `.where` | `.when` | note |
|---|---|---|---|
| InputSpace | 2 | 2 | unchanged |
| PerceptualSpace | 2 | 2 | unchanged |
| **ConceptualSpace** | **2** | **2** | was `(0,2)`; CS now *carries* the event where/when (mux at PS $\to$ CS) |
| **SymbolicSpace** | **0** | **0** | was `(2,2)`; symbols carry neither (demux at CS $\to$ SS) -- no promotion |
| **OutputSpace** | **0** | **0** | was `(0,2)`; OS carries neither |
| WordSpace | 0 | 0 | unchanged |

`MANDATORY_CODEBOOK_TIERS = {PerceptualSpace, SymbolicSpace}` is unchanged -- SS keeps its mandatory `.what` codebook; it just carries no where/when.

---

## 4. `.where`: extent, not a codebook index

Today `.where` plays three roles (`bin/Spaces.py`): (a) a **cross-codebook row key** via the `WhereEncoding` codebook-offset registry (`:232-289`, `allocate_codebook_slice` `:1939-1947`); (b) an **input-space offset** -- byte offset of a token into the source buffer, with the PS codebook filling `.what` (`:7061-7092`, `:7476-7479`); (c) a **positional/spatial extent** (sin/cos, `:299-326`).

**Change:** abandon role (a). Codebook identity moves to `.activation` / `_active` (the selection index) and sparse symbolic keys -- not a `.where` value. `.where` keeps roles (b) and (c): it carries the active spatial/relational extent at the current abstraction level, and (with `.when`) indexes into the **input space** alongside the PS codebook that fills `.what`. This frees `.where`/`.when` to be **manipulated by VP/PP operations** rather than being codebook addresses.

This **reconciles with** `doc/plans/2026-05-28-where-keyed-taxonomy.md` (already targets retiring `.where`-as-row-index). The plan from this spec must not conflict with or duplicate that work -- it consumes/extends it.

---

## 5. Grammar ops operate event $\to$ event; per-op modality manipulation

All symbol/composition operations operate on the **materialized `.event`** from CS (`[what | where | when]`) and produce an event, automatically demuxed back into `.what`/`.where`/`.when`. They are **event $\to$ event**: mux/demux is transparent, so the trailing `.where`/`.when` dims are **not** treated as special by sigma/pi or any other operation -- an op sees one event tensor. (Mechanism: the CS-tier `compose` operand is the muxed event, not the content-only `.what` slice -- Section 6.4; the byte-tokenization / SS-tier / demux / surface-reconstruction reads stay content-only `mode="what"`, where where/when ride as parallel slabs read as siblings.)

Operations manipulate the modalities they characteristically own; others pass them through:

- **`LIFT` / `LOWER` alter `.when`** -- they add a **span $> 1$** to `.when` (the process's temporal extent), shifting its dimensionality from a point toward an interval. (`.when` is the signed, zero-centered, 2-dim unit-bracket range from convergence Phase 1: point $t \to (t-0.5, t+0.5)$, present $(-0.5, 0.5)$, tense a phase rotation, aspect a center-anchored reshape -- `SIMPLE`$=(r-0.5, r+0.5)$, `PERFECT`$=(r-1, r)$, `PROGRESSIVE`$=(r-1, r+1)$. `LIFT` extends/advances it via `WhenRangeEncoding.rotate` + interval reshape.)
- **`PREPOSITION` / PP / spatial ops modify `.where`** -- the active spatial/relational extent.
- Most other ops pass `.where`/`.when` through unchanged.

**Composition order $\to$ abstraction.** First-order generalizations yield **contiguous** `.where`/`.when` (concrete spatial/temporal support); higher-order operations may produce **abstract** `.where`/`.when` (no longer a contiguous concrete extent) -- which is what lets symbols generalize away from particular places and times.

---

## 6. Size invariant and construction changes

**Invariant (confirmed with the author): `.where`/`.when` ADD on top of `nDim`.** Content width `nWhat = nDim` is preserved; the muxed width `muxedSize = nDim + nWhere + nWhen` grows. No `nDim` reduction (that underflows small spaces, e.g. OutputSpace `nDim=1`).

Construction changes (cited from the dim-flow investigation):

1. **Source where/when from `canonical_shape(section)`** at every config read site: `bin/Spaces.py:6304-6305`; `bin/Models.py:374-391` (`_obj_size`), `:567-568` (global, derive from IS); `bin/util.py:993-994`; `bin/Spaces.py:8025-8026`; `ModelLoss` (`bin/Layers.py:11205-11228`, `bin/Models.py:4413-4414`) -- the loss must use the **output tier's** shape `canonical_shape("OutputSpace")=(0,0)`, not InputSpace.
2. **Grammar per-stage path adds the event width to CS (only).** `bin/Models.py:4564-4570` currently sets `cs_out`/`ss_out` to bare `concept_dim`, dropping the where/when. Add `_obj_size("ConceptualSpace")=4` to the **CS output muxed width** so `SubSpace.nWhat == nDim` and the CS event carries where/when. **Leave SS bare** (`canonical SS = (0,0)`; the existing CS $\to$ SS trim is the demux). Keep the codebook sizing (`stage_space_*`, `_build_*_basis`) at bare `nDim` -- where/when ride as traces, not codebook content (`bin/Spaces.py:13732-13736`, `13764-13769`).
3. **Downstream of the grown muxed width:** flat-slab dim validator (`bin/Models.py:7452-7497`) compares on `nDim` (preserved) -- keep it `nDim`-based and keep `nOutputDim == nDim` (so the SS trim at `:14730-14733` and the validator stay correct). IS $\to$ PS divisibility (`bin/Spaces.py:8024-8041`) self-adjusts from the three widths. `held_at_zero`/`empty_state`/`dereference` read `muxedSize` and self-adjust.
4. **Grammar compose operand = the event, on the C-tier path.** Today `LanguageLayer.reduce` hands `layer.compose(left, right)` a content-only `.what` slice (`bin/Language.py:4384-4386` $\to$ `:6063/:6069`). On the C-tier grammar path (CS carries where/when), build the operand from the **muxed event** so `LIFT`/`LOWER`/`PREPOSITION` can manipulate `.where`/`.when` (Section 5). The SS stack route stays content-only (SS is `0/0`). The byte-tokenization / demux / SS / surface-reconstruction `materialize(mode="what")` reads are **unchanged**.

---

## 7. Migration

- **Code:** revise `bin/architecture.py` table (Section 3); wire the read sites (Section 6.1); the CS per-stage fix (Section 6.2); decouple `.where` from the codebook registry (Section 4, reconciled with the where-keyed-taxonomy plan); enforce mandatory PS/SS codebooks (`bin/Models.py:4336-4339` + `MANDATORY_CODEBOOK_TIERS`, reject `none`); make the CS-tier grammar `compose` operand the muxed event so `LIFT`/`LOWER` alter the `.when` span and `PREPOSITION` modifies `.where` (Sections 5, 6.4); fix the stale docs (Section 1).
- **Configs (all `data/*.xml`, none removed):** strip `<nWhere>`/`<nWhen>` (now from `canonical_shape`); PS/SS `<codebook>` $\ne$ `none` (e.g. MentalModel PS `none` $\to$ `quantize`); `<nDim>` **unchanged**; drop explicit `<nInputDim>`/`<nOutputDim>` that hardcode old muxed widths (preferred over bumping -- bumping trips the flat-slab validator).
- **Tests:** migrate shape-dependent tests (`TestSubSpaceDerivedSizes`, `TestWhenEncodingRoundTrip`, `test_use_flags`, XOR reconstruction now that PS/SS codebooks are mandatory and CS carries where/when); add a build-every-live-config guard asserting per-tier `canonical_shape` and a finite forward.
- **Compatibility:** no shim; pre-change checkpoints retrain; confirm `autoload` defaults safe.

---

## 8. Documented direction (NOT built in this effort)

The architecture above is the substrate for a linguistic layer to be built later:

- **NP** carries object-like content: present-time anchoring + 3D location/body/extent in `.where`.
- **VP** carries process-like content: present $\to$ future temporal extent in `.when` + a low-dimensional path / mapping / affordance / control trajectory in `.where`.
- **LIFT(NP, VP)** composes a **4D spacetime event** (NP's spatial extent $\times$ VP's temporal extent + path). `LIFT`/`LOWER` already altering the `.when` span (Section 5) is the first concrete step toward this.
- **`.where` width scales with abstraction level:** 1D for a word token's input span, **3D** for an NP's spatial extent (location / body / extent), a low-dim manifold for a VP's path/control. The current `.where` is 2-dim; a widened / per-level `.where` is future work (noted in `todo.md`).

These require new representational machinery (3D extents, control manifolds) and are out of scope now; this spec only ensures the modality substrate supports them.

---

## 9. Superseded / dissolved

- Supersedes the SS-promotion convergence design. SS is **not** promoted to a where/when carrier; the SS-codebook-desync blocker is dissolved.
- Convergence Phase 1 (unit-bracket `.when`) is **kept** as-is. Phase 2's guards proved width-agnostic muxing but tested superseded tier shapes; the new plan re-guards the revised shapes (CS `where=2/when=2` `.event` codebook; SS `where=0/when=0`).
- The convergence plan's Phase 3 migration (resize `outputShape` / reduce `nDim`) is **replaced** by Sections 6-7 (keep `nDim`, grow muxed, CS per-stage fix).

## 10. Risks

1. **Grammar per-stage CS fix** (Section 6.2) is the load-bearing code change; without it CS's `nWhat` desyncs (verified). Guarded by a per-config build + forward test.
2. **`.where` decoupling** intersects the where-keyed-taxonomy plan and the `WhereEncoding` codebook registry -- must be reconciled, not duplicated; risk of double-touching the registry.
3. **Per-config XML migration** across ~30 files; the build-all-configs guard is the safety net; drop (don't bump) explicit dim overrides.
4. **Order-fragility** of `test_basicmodel.py`: judge against an isolation baseline.
5. `LIFT` incrementing `.when` must not perturb `.what`/`.where`; unit-test the `.when` tail only.

## 11. Success criteria

- `canonical_shape`: IS/PS/CS report `where=2/when=2`; SS/OS/Word report `0/0`; no `<nWhere>`/`<nWhen>` tags remain in `data/*.xml`.
- CS muxed event carries where/when (mux at PS $\to$ CS); SS/OS carry neither (demux at CS $\to$ SS); every live config builds + runs a finite forward.
- `.where` no longer indexes a codebook; codebook identity is via `.activation`/`_active`; the where-keyed-taxonomy work is reconciled.
- PS/SS codebooks mandatory; `LIFT` advances `.when` (unit-tested); unit-bracket `.when` intact.
- Stale `SymbolicSpace`/`_active_payload` docs corrected.
- `test_basicmodel.py` failures $\le$ isolation baseline; `test_role_collapsed_grammar.py` + `test_d1_pos_recovery_gate.py` green.
- **MorphologyLayer** (Section 12): `surface_morphology.analyze` returns `(lemma, features)` for the focused cases incl. the fixed over-fire; `MorphologyLayer` registered + auto-wired; its grammar rule loads; the no-POS guard stays green.

---

## 12. MorphologyLayer (original-plan Phase 7, on the converged substrate)

Folds the original plan's Phase 7 into this effort, built **after** the modality re-architecture so it targets the converged `.what`/`.where`/`.when` substrate (a morphology op built first would need re-architecting once the modalities settle).

**Purpose.** Generalize the verb-only `bin/surface_tense.py` into a morphological analyzer/synthesizer that decomposes a surface word form into a **base lemma** (the `.what` content / codebook key) plus **role-neutral morphological features** that drive the modalities: tense/aspect $\to$ `.when` (via the existing `TenseLayer`/`AspectLayer`), with number/degree/etc. as the growth path. **No global POS inventory** (the hard constraint): `features` are morphological annotations, not parts of speech.

**Pieces.**
- `bin/surface_morphology.py` (pure, table-driven, no torch) -- `analyze(token) -> (lemma, features)` generalizing `surface_tense.normalize_surface`. Subsumes the verb tense/aspect tables and adds a **corrected lemmatizer** that fixes the known over-fire (`surface_tense._base_of` strips `-ed`/`-ing` too aggressively, e.g. `seed -> se`, `freed -> fre`) via a minimum-stem-length + known-irregular + suffix-gate; `features` is a small dict (e.g. `{"tense": "PAST", "aspect": ["PERFECT"]}`), role-neutral.
- `MorphologyLayer(GrammarLayer)` in `bin/Language.py` -- a C-tier op following the `PrepositionLayer`/`TenseLayer` pattern: `forward` normalizes a surface token to its base concept (lemma $\to$ `.what`) and routes tense/aspect features onto the event `.when` by **delegating to the existing `TenseLayer`/`AspectLayer`** (not duplicating them); `reverse` re-synthesizes the surface form from base + features. Registered in `GRAMMAR_LAYER_CLASSES` + `_OPERATOR_SURFACE_SCHEMAS`; a role-only rule in `data/role_collapsed.grammar`; auto-wired via the standard `_wire_signal_router_grammar_ops` path.

**Design points to confirm at implementation** (flagged, bounded by the Phase-4 ops + the modality architecture): the exact feature$\to$modality routing (which features touch `.where` vs `.when` vs `.activation`); whether `MorphologyLayer` is unary (normalizes one token) or marker-based (absorb/emit affixes); and the precise division of labour with `surface_tense`/`TenseLayer`/`AspectLayer` (the layer should *call* `surface_morphology` and *delegate* tense/aspect, so the Phase-4 ops are reused, not re-implemented).
