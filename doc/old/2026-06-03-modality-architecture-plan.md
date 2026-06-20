# Modality Re-Architecture + MorphologyLayer -- Implementation Plan (all remaining work)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans, task-by-task. Steps use checkbox (`- [ ]`) tracking.
>
> **Companion design spec (read first):** `doc/plans/2026-06-03-modality-architecture-design.md`. This plan implements that spec; section references below (e.g. "spec Section 6.2") point into it.
>
> **Git:** Per repo convention **Alec performs all git writes.** Pause at each "Commit (Alec)" checkpoint; do NOT run `git` writes.
>
> **Test runner:** `.venv/bin/python -m pytest <nodeid> -v` from the repo root. `test/test_basicmodel.py` is order-fragile; judge regressions against an **isolation** run (currently 205 passed / 2 skipped / 0 failed).

**Goal:** Implement the consolidated modality architecture (spec): `.where`/`.when` become architectural constants that ride the muxed event through IS$\to$PS$\to$CS and demux at CS$\to$SS (SS/OS carry neither); `.where` stops being a codebook row-index; grammar ops operate event$\to$event so LIFT/LOWER alter the `.when` span and PREPOSITION modifies `.where`; PS/SS codebooks become mandatory; all model configs migrate; stale docs are fixed; and the original-plan **MorphologyLayer** is built on the converged substrate.

**Architecture:** A single per-tier `canonical_shape(section)` (in `bin/architecture.py`) replaces the `<nWhere>`/`<nWhen>` config tags. Content width `nWhat = nDim` is preserved; the muxed width `nDim + nWhere + nWhen` grows. The grammar per-stage path adds the where/when width to CS (not SS). MorphologyLayer is a C-tier `GrammarLayer` over a pure `surface_morphology` table.

**Tech Stack:** Python 3, PyTorch, pytest, XML config, the `GrammarLayer`/`SubSpace`/`Space`/`Codebook`/`Encoding` substrate.

---

## Working-tree starting state (what is ALREADY built -- do NOT redo)

This plan builds on uncommitted work already present in the repo (commit it first if you prefer a clean base; the plan does not depend on whether it is committed):

- **Contextual-bind grammar ops** (`bin/Language.py`): `PrepositionLayer`, `ContextualBindLayer`, `TenseLayer`, `AspectLayer` (+ registries + `data/role_collapsed.grammar` rules); `bin/bind_resolver.py`; `bin/surface_tense.py`. Tests: `test/test_grammar_preposition.py`, `test/test_contextual_bind.py`, `test/test_tense_aspect.py`, `test/test_grammar_fixtures.py`, `test/test_when_grammar_rules.py`.
- **Unit-bracket `.when`** (`bin/Spaces.py` `WhenRangeEncoding`): `encode(t)=encode_range(t-0.5,t+0.5)`; present stamp `(-0.5,0.5)`; `aspect_interval` center-anchored (`SIMPLE=(r-0.5,r+0.5)`, `PERFECT=(r-1,r)`, `PROGRESSIVE=(r-1,r+1)`); `AspectLayer.forward` reads `r` from the interval center. Tests: `test/test_when_bracket.py`, `test/test_when_range_encoding.py`.
- **Width-guard tests** `test/test_convergence_widths.py` -- proved the SubSpace muxing is width-agnostic; their **tier shapes are stale** (Phase 1 re-guards the revised shapes).
- **`bin/architecture.py`** -- exists with an OLD table (`SS=(2,2)`, `CS=(0,2)`, `OS=(0,2)`); **Phase 1 revises it**. Test `test/test_canonical_shape.py` exists for the old values.

## Grounding (verified `file:line`; read before editing)

- Per-section where/when reads: `bin/Spaces.py:6304-6305`; `bin/Models.py:374-391` (`_obj_size`), `:567-568` (global, $\to$ `architecture.objectSize`), `:4413-4414` (ModelLoss); `bin/util.py:993-994`; `bin/Spaces.py:8025-8026`; `bin/Layers.py:11205-11228` (ModelLoss self-resolve).
- `nWhat = nDim`; `muxedSize = nDim + nWhere + nWhen` (`bin/Spaces.py:6308-6309`); IO shapes `[count, nDim + objectSize]` (`bin/Models.py:4447-4452`). `SubSpace.nWhat = outputShape[1] - nWhere - nWhen` (`:4475`) -- agrees iff `outputShape[1] = nDim + nWhere + nWhen`.
- **Grammar per-stage path** (the load-bearing fix): `bin/Models.py:4564-4570` sets `cs_out`/`ss_out` to bare `concept_dim`, dropping where/when. The plain (non-grammar) path already adds `obj` via `symbolShape` (`:4451`).
- Config inheritance: `XMLConfig.space(section,key)` falls back to `architecture[key]` (`bin/util.py:1069-1080`); every config overlays `data/model.xml` (`bin/Models.py:545-546`), whose `architecture` is `nWhere=0/nWhen=0` (`data/model.xml:87-88`).
- Flat-slab dim validator (`nDim`-based, `modelType=embedding` only): `bin/Models.py:7452-7497`. IS$\to$PS divisibility: `bin/Spaces.py:8024-8041`. SS where/when trim (the demux): `bin/Spaces.py:14724-14733`.
- Codebook gating: `<codebook>` $\to$ `Space.normalize_codebook_mode`; `bin/Models.py:4336-4339`; `codebook_slot` `bin/Spaces.py:4503-4515`; `!= "none"` `:6481`. SS `.what` codebook sized to `nDim` (`bin/Spaces.py:13764-13769`).
- Grammar compose: `LanguageLayer.reduce` operand $\to$ `syntactic_layer.execute` $\to$ `layer.compose` (`bin/Language.py:4384-4386`, `:6063`, `:6069`); rules bind by tier (`:6039-6043`). `materialize(mode="what")` reads that stay content-only: `bin/Language.py:4282,4453,4619`; `bin/Spaces.py:8289,8422,8573,8639,9232,9320,9389,10047,14568,14658`; `bin/Models.py:7000` (spec Section 5 / the classification).
- `.where` codebook registry (to decouple): `bin/Spaces.py:232-289`, `1939-1947`, `3289-3305`; reconcile with `doc/plans/2026-05-28-where-keyed-taxonomy.md`. `.where`/`.when` input-space offsets (kept): `bin/Spaces.py:7061-7092`, `7476-7479`.
- Stale docs: `SymbolicSpace` class docstring `bin/Spaces.py:11809-11817`; stray `_active_payload` token `:4558`.
- `surface_tense._base_of` over-fire (MorphologyLayer fixes): `bin/surface_tense.py` (`seed -> se`, `freed -> fre`).
- **Shared test preamble** (new test files):
  ```python
  import math, os, sys, unittest
  from pathlib import Path
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
  import torch
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
  ```
- **Model build idiom** (tests): `from util import init_config; init_config(path=cfg, defaults_path="data/model.xml"); Language.TheGrammar._configured = False; model,_ = Models.BasicModel.from_config(cfg); model.eval()`. Per-tier shape via `model.named_modules()` carriers exposing `.nWhere`/`.nWhen`/`.whenEncoding` (see `test/test_when_grammar_rules.py::TestMentalModelWhenEnabled`).

---

## Phase 1 -- Revise `canonical_shape`; re-guard the new tier shapes

### Task 1.1: Revise the canonical table (spec Section 3)
**Files:** Modify `bin/architecture.py`; Modify `test/test_canonical_shape.py`.

- [ ] **Step 1: Update the failing test** -- in `test/test_canonical_shape.py`, change the expected table to: `InputSpace=(2,2)`, `PerceptualSpace=(2,2)`, `ConceptualSpace=(2,2)`, `SymbolicSpace=(0,0)`, `OutputSpace=(0,0)`, `WordSpace=(0,0)`. Run $\to$ FAIL (old values).
- [ ] **Step 2: Implement** -- in `bin/architecture.py`, set `_CANONICAL_SHAPE` to `{"InputSpace":(2,2), "PerceptualSpace":(2,2), "ConceptualSpace":(2,2), "SymbolicSpace":(0,0), "OutputSpace":(0,0), "WordSpace":(0,0)}`. `MANDATORY_CODEBOOK_TIERS = {"PerceptualSpace","SymbolicSpace"}` unchanged.
- [ ] **Step 3: Run** $\to$ PASS.
- [ ] **Step 4: Commit (Alec)** -- `feat: revise canonical_shape (CS carries where/when; SS/OS carry neither)`

### Task 1.2: Re-guard the revised SubSpace shapes
**Files:** Modify `test/test_convergence_widths.py` (or add `test/test_modality_widths.py`).

- [ ] **Step 1: Update the guards** to the revised target shapes: a **CS-shaped** SubSpace `where=2/when=2` with an **`.event` codebook** (`codebook_slot='event'`, `muxed=True`) round-trips a `.where` span + `.when` bracket and reconstructs; an **SS-shaped** SubSpace `where=0/when=0` with a **`.what` codebook** round-trips content only. Reuse the construction idiom already in this file (`Codebook().create(...)`, `SubSpace(..., object=cb)` for event / `what=cb` for what; `WhereEncoding.reset_codebook_registry()` per test). Assert `muxedSize`, the round-trips, and `torch.isfinite(...).all()`.
- [ ] **Step 2: Run** $\to$ PASS (the muxing is width-agnostic; these should pass without production change -- if one fails, fix the hardcoded width it exposes, reading `whenEncoding.nDim`/`whereEncoding.nDim`).
- [ ] **Step 3: Commit (Alec)** -- `test: re-guard modality SubSpace shapes (CS event 2/2; SS what 0/0)`

---

## Phase 2 -- Substrate flip + config migration (the big one; red interior, final green)

Wire `canonical_shape` into construction, fix the grammar per-stage CS width, adjust downstream, make PS/SS codebooks mandatory, and migrate every config. The build-all-configs guard is the green gate. Per spec Sections 6-7.

### Task 2.1: Wire read sites to `canonical_shape`
**Files:** Modify `bin/Spaces.py:6304-6305`, `bin/Models.py:374-391` + `:567-568` + `:4413-4414`, `bin/util.py:993-994`, `bin/Spaces.py:8025-8026`, `bin/Layers.py:11205-11228`.

- [ ] **Step 1:** At each site, replace the `TheXMLConfig.space(section,"nWhere"/"nWhen")` / `architecture.*` reads with `from architecture import canonical_shape; nw, nn = canonical_shape(section)` (use the section name in scope; for the global/loss sites that have no section, use `canonical_shape("InputSpace")` for IS-width reconstruction and `canonical_shape("OutputSpace")` for the **output-tier** loss term -- see spec Section 6.1). Verify every section-name string matches a `canonical_shape` key.
- [ ] **Step 2:** Build `MentalModel.xml` (build idiom above). The Input/Perceptual subspaces should report `where=2/when=2`; **CS/SS/OS will be wrong until Tasks 2.2-2.4** -- that is the expected red interior.
- [ ] **Step 3: Commit (Alec)** -- `feat: source per-tier .where/.when from canonical_shape`

### Task 2.2: Grammar per-stage path adds the event width to CS (only)
**Files:** Modify `bin/Models.py:4564-4570` (the `useGrammar=="all"` per-stage block).

- [ ] **Step 1:** Add `_obj_size("ConceptualSpace")` (= `sum(canonical_shape("ConceptualSpace"))` = 4) to the **CS output muxed width** so `SubSpace.nWhat == nDim` and the CS event carries where/when. **Leave SS bare** (`canonical SS=(0,0)`; the existing CS$\to$SS trim at `bin/Spaces.py:14724-14733` is the demux). Keep `stage_space_*` / `_build_*_basis` codebook sizing at bare `nDim` (where/when ride as muxed traces, not codebook content). Keep `nOutputDim == nDim` (so the SS trim + flat-slab validator stay correct).
- [ ] **Step 2:** Build `MentalModel.xml`; the CS subspace should now report `nWhat == nDim` and `muxedSize == nDim + 4`, and a forward should not raise the SS `.what`/`nWhat` desync. (If `nOutputDim` widened unintentionally, pin it to `nDim`.)
- [ ] **Step 3: Commit (Alec)** -- `feat: grammar per-stage path carries where/when on the CS event`

### Task 2.3: Mandatory PS/SS codebooks
**Files:** Modify `bin/Models.py:4336-4339` (codebook-mode resolution); Test `test/test_modality_codebook.py`.

- [ ] **Step 1: Failing test** -- build `MentalModel.xml`, assert the PerceptualSpace and a SymbolicSpace subspace report `codebook_mode != "none"`; and that a temp config setting `SymbolicSpace <codebook>none</codebook>` raises at build (use the `xml.etree` temp-config idiom from `test/test_basicmodel.py::TestReconstructionSymbols._create_xor_model`).
- [ ] **Step 2: Run** $\to$ FAIL.
- [ ] **Step 3: Implement** -- for `section in architecture.MANDATORY_CODEBOOK_TIERS`, if the resolved codebook mode is `"none"`, raise `ValueError(f"{section}: codebook is mandatory; codebook=none is not allowed")`; if it would default to `none`, force `"quantize"`.
- [ ] **Step 4: Run** $\to$ PASS.
- [ ] **Step 5: Commit (Alec)** -- `feat: make PerceptualSpace/SymbolicSpace codebooks mandatory`

### Task 2.4: Migrate all model XMLs + build-all guard
**Files:** Modify all `data/*.xml`; Test `test/test_modality_configs.py`.

Per-config edit (spec Section 7): **strip** `<nWhere>`/`<nWhen>` (now from `canonical_shape`); for PS/SS set `<codebook>` $\ne$ `none` (e.g. MentalModel PS `none` $\to$ `quantize`); leave `<nDim>` **unchanged**; **drop** explicit `<nInputDim>`/`<nOutputDim>` that hardcode old muxed widths (do NOT bump them -- bumping trips the flat-slab validator).

- [ ] **Step 1: Build-all guard test** -- `_live = sorted(data/*.xml) minus _BROKEN` where `_BROKEN = {"model.xml","MM_5M.xml","MM_400M.xml","MM_shamatha.xml","MM_xor_step4.xml"}` (broken pre-existing, per `test/test_use_flags.py`); for each live config, build it and assert every whenEncoding-carrier reports `nWhen == 2`, IS/PS/CS report `where=2`, SS/OS report `where=0/when=0`, and a tiny forward is finite. Run $\to$ FAIL (configs unmigrated).
- [ ] **Step 2: Implement** -- apply the per-config edit to every `data/*.xml`, in small batches, re-running Step 1's guard. `grep -rl "nWhere\|nWhen" data/*.xml` must come back empty when done. If a live config fails for a reason clearly unrelated to where/when/codebook, move it to `_BROKEN` with a one-line note and report it.
- [ ] **Step 3: Run** $\to$ PASS.
- [ ] **Step 4: Commit (Alec)** -- `chore: migrate all model XMLs to canonical modality shape`

---

## Phase 3 -- Grammar ops operate event $\to$ event (CS compose operand)

Per spec Section 5 / 6.4. Make the C-tier `compose` operand the muxed event so LIFT/LOWER alter `.when` and PREPOSITION modifies `.where`; everything else (byte-tokenization / SS / demux reads) stays `mode="what"`.

### Task 3.1: C-tier `reduce`/`compose` operand = event
**Files:** Modify `bin/Language.py` (`LanguageLayer.reduce` operand construction `:4384-4386`); Test `test/test_event_compose.py`.

- [ ] **Step 1: Failing test** -- construct the C-tier path so a `LIFT`/`compose` receives an operand with a `.when` tail and assert the result's `.when` tail is altered (span $> 1$) while `.what` is preserved; assert an SS-tier `reduce` operand stays content-only. (Mirror the dispatch in `test/test_grammar_rewrite.py`; read `:6039-6043` for tier binding.)
- [ ] **Step 2: Run** $\to$ FAIL (operand currently content-only).
- [ ] **Step 3: Implement** -- on the C-tier (CS) grammar path, build the `left`/`right` operands from `materialize(mode="event")` (muxed `[what|where|when]`) instead of the `.what` slice; keep the SS stack route on `mode="what"`. Verify the byte-tokenization / demux / SS / surface-reconstruction `materialize(mode="what")` reads (grounding list) are untouched.
- [ ] **Step 4: Run** $\to$ PASS.
- [ ] **Step 5: Commit (Alec)** -- `feat: C-tier grammar compose operates on the muxed event`

### Task 3.2: LIFT/LOWER alter the `.when` span; PREPOSITION modifies `.where`
**Files:** Modify `bin/Language.py` (`LiftLayer`/`LowerLayer`/`PrepositionLayer` forward); Test `test/test_event_compose.py` (extend).

- [ ] **Step 1: Failing tests** -- `LIFT` on an event adds a span $> 1$ to `.when` (decode the `.when` tail: end$-$start $> 1$, and the center advances per the verb-advances-future rule); `LOWER` is its inverse; `PREPOSITION` modifies the `.where` block while leaving `.what`/`.when`; all leave `.what` unchanged (`abs_tol` per the `.when` unit-bracket tolerances, `abs_tol=0.05`).
- [ ] **Step 2: Run** $\to$ FAIL.
- [ ] **Step 3: Implement** -- in the relevant layers' `forward`, operate on the event tail: LIFT/LOWER reshape/rotate the trailing `.when` 2-dim via `WhenRangeEncoding` (extend the interval to a span $> 1$; advance the center); PREPOSITION rewrites the `.where` 2-dim. Reuse `TenseLayer`/`AspectLayer`/`WhenRangeEncoding` helpers; do not special-case the columns elsewhere. (Confirm the precise span/where semantics against spec Section 5; flagged design point.)
- [ ] **Step 4: Run** $\to$ PASS; re-run `test/test_grammar_fixtures.py` (Phase-5 fixtures) and adjust any `.when` expectations the span change shifts.
- [ ] **Step 5: Commit (Alec)** -- `feat: LIFT/LOWER alter .when span; PREPOSITION modifies .where`

---

## Phase 4 -- Decouple `.where` from codebook indexing

Per spec Section 4. **Reconcile with `doc/plans/2026-05-28-where-keyed-taxonomy.md` -- read it first; do not duplicate it.**

> **RECONCILIATION DECISION (Alec, 2026-06-04).** The where-keyed-taxonomy
> plan is *landed* (`WhereEncoding.recover`, `SymbolicSpace.allocate_position`/
> `_next_position`, position-keyed taxonomy + persistence, and the
> `WhereEncoding` codebook-slice registry all exist in code) and it makes
> `.where` the canonical positional *identity* -- the OPPOSITE of this
> Phase's premise. They genuinely conflict; the spec Section 4 claim that the
> where-keyed plan "already targets retiring `.where`-as-row-index" was a
> misread (that plan *replaces* signed-int row indices *with* `.where`-keyed
> positions, entrenching `.where` as identity).
>
> **Resolution:** revert the `.where`-as-codebook-index mechanism back to
> **regular hierarchy row-indexing**. Codebook identity is the row index (the
> `_active` selection), not a `.where` quadrature position; `.where` is freed
> to be the spatial/relational extent (the muxed-event `.where` Phase 3
> manipulates). Consequence (accepted): the CS$\to$SS `reverse()` is **no
> longer an exact inversion** unless the codebook row indices are cached --
> in general we do **approximate inverses based on priming + content match**.
> This SUPERSEDES the where-keyed-taxonomy plan's `.where`-as-identity choice.
>
> **Scope of the revert (load-bearing; do as a focused effort):** retire
> `WhereEncoding.allocate_codebook_slice` / `_codebook_registry` / `where_offset`
> (`bin/Spaces.py:256-290`, `:1947`, `:3306`) and the `.where`-position
> identity (`allocate_position`/`recover` callers at `:12646`, `:12669`,
> `:13634`); re-key the taxonomy + `meta_pair_to_idx` + LBG split + reverse
> decode on plain row indices; make reverse decode content-match/priming based;
> migrate the where-keyed taxonomy tests (`test_where_encoding_recovery.py`,
> `test_two_codebook_meta_taxonomy.py`, `test_positive_int_taxonomy.py`).
> NOTE: those taxonomy tests are *already* red after Phase 2 -- not from this
> revert but from the substrate flip's +4 muxed width tripping the IS$\to$PS
> divisibility on their hand-built models (Phase 6 fixture migration).

### Task 4.1: Reconcile + retire the `.where`-as-row-index role
**Files:** Read `doc/plans/2026-05-28-where-keyed-taxonomy.md`; Modify `bin/Spaces.py` (`WhereEncoding` codebook registry `:232-289`, `1939-1947`, `3289-3305`); Test `test/test_where_decoupling.py`.

- [ ] **Step 1:** Read the where-keyed-taxonomy plan; determine its overlap with "`.where` is not a codebook row index." Write a short reconciliation note at the top of the test file (what that plan owns vs what this task does).
- [ ] **Step 2: Failing test** -- assert codebook *identity* is recoverable via `.active`/`.activation` (the selection index), and that `.where` carries spatial extent (input-span / positional), NOT a codebook row key, for a constructed SubSpace.
- [ ] **Step 3: Run** $\to$ FAIL (where currently keys the registry).
- [ ] **Step 4: Implement** -- remove the `.where`-as-codebook-row-index path (the `allocate_codebook_slice`/registry role used as identity), keeping `.where`'s input-offset and positional-extent roles (`bin/Spaces.py:7061-7092`, `299-326`). Route codebook identity through `_active`. Coordinate with the where-keyed-taxonomy plan's mechanism rather than re-introducing a parallel one.
- [ ] **Step 5: Run** $\to$ PASS; re-run `test/test_basicmodel.py -q` (isolation) -- failures $\le$ baseline.
- [ ] **Step 6: Commit (Alec)** -- `feat: decouple .where from codebook row-indexing (identity via .active)`

> **If the where-keyed-taxonomy plan is mid-flight or conflicts**, STOP and surface to Alec rather than double-touching the registry (spec Risk 2).

---

## Phase 5 -- Doc cleanup

### Task 5.1: Correct stale modality docs (spec Section 1)
**Files:** Modify `bin/Spaces.py:11809-11817` (SymbolicSpace docstring), `:4558` (stray `_active_payload`).

- [ ] **Step 1:** Update the `SymbolicSpace` class docstring to the live behavior (inherits `nWhat==nDim`; bivector lives on `subspace.activation`; `_active_payload` retired -- see `doc/plans/2026-05-21-active-payload-retirement.md`). Remove/repair the stray `_active_payload` token at `:4558`.
- [ ] **Step 2:** `.venv/bin/python -c "import sys; sys.path.insert(0,'bin'); import Spaces"` imports clean; `grep -n "_active_payload" bin/Spaces.py` shows only intentional references (ideally none).
- [ ] **Step 3: Commit (Alec)** -- `docs: correct stale SymbolicSpace/_active_payload docstrings`

---

## Phase 6 -- Test migration + regression gate

### Task 6.1: Migrate shape-dependent tests
**Files:** Modify `test/test_basicmodel.py` (`TestSubSpaceDerivedSizes` `~:603`, `TestReconstructionSymbols`), `test/test_use_flags.py`.

- [ ] **Step 1:** Update `TestSubSpaceDerivedSizes` muxed-width expectations for the revised per-tier shapes; update the XOR reconstruction expectations (PS/SS codebooks now mandatory; CS carries where/when); update `test_use_flags.py` per-config expectations if any `useGrammar`/instantiability flips.
- [ ] **Step 2: Run** each touched file `-v` $\to$ PASS.
- [ ] **Step 3: Commit (Alec)** -- `test: migrate shape-dependent tests to the modality architecture`

### Task 6.2: Regression gate
- [ ] **Step 1:** `.venv/bin/python -m pytest test/test_basicmodel.py -q` -- failures $\le$ isolation baseline.
- [ ] **Step 2:** `.venv/bin/python -m pytest test/test_role_collapsed_grammar.py test/test_d1_pos_recovery_gate.py -v` -- green (no POS leak).
- [ ] **Step 3:** `.venv/bin/python -m pytest test/test_when_bracket.py test/test_when_range_encoding.py test/test_tense_aspect.py test/test_grammar_fixtures.py test/test_modality_widths.py test/test_modality_configs.py test/test_modality_codebook.py test/test_event_compose.py test/test_where_decoupling.py test/test_canonical_shape.py -q` -- the modality surface, all green.
- [ ] **Step 4:** Confirm `<autoload>` defaults safe in live XMLs (no shim; pre-change checkpoints retrain). `grep -rn "autoload" data/*.xml`.
- [ ] **Step 5: Commit (Alec)** -- `chore: regression gate for the modality re-architecture`

---

## Phase 7 -- MorphologyLayer (original-plan Phase 7; spec Section 12)

### Task 7.1: `bin/surface_morphology.py` (pure table; fixes the over-fire)
**Files:** Create `bin/surface_morphology.py`; Test `test/test_surface_morphology.py`.

- [ ] **Step 1: Failing tests** -- `from surface_morphology import analyze`:
```python
def test_verb_subsumes_surface_tense():
    assert analyze("ran")     == ("run",  {"tense": "PAST",    "aspect": []})
    assert analyze("running") == ("run",  {"tense": "PRESENT", "aspect": ["PROGRESSIVE"]})  # bare participle
def test_lemmatizer_overfire_fixed():
    assert analyze("seed")[0]  == "seed"     # NOT "se"
    assert analyze("freed")[0] == "freed"    # NOT "fre" (or "free" if you add it to the irregulars)
def test_plain_token_passthrough():
    assert analyze("cat") == ("cat", {})
```
- [ ] **Step 2: Run** $\to$ FAIL (ImportError).
- [ ] **Step 3: Implement** `analyze(token) -> (lemma, features)`: reuse `surface_tense`'s verb tense/aspect tables for verb forms; for lemmatization, apply `-ed`/`-ing` stripping ONLY when the residue passes a min-stem-length ($\ge 3$) + known-suffix gate + irregular table, so `seed`/`freed` are NOT mis-stripped (they fail the gate / are stoplisted). `features` is a role-neutral dict; non-verb/unknown tokens return `(token, {})`. Pure Python, no torch, **no global POS inventory**.
- [ ] **Step 4: Run** $\to$ PASS.
- [ ] **Step 5: Commit (Alec)** -- `feat: add surface_morphology (lemma + features; fixes -ed/-ing over-fire)`

### Task 7.2: `MorphologyLayer` + registration + grammar rule
**Files:** Modify `bin/Language.py` (after `AspectLayer`) + `GRAMMAR_LAYER_CLASSES` + `_OPERATOR_SURFACE_SCHEMAS`; Modify `data/role_collapsed.grammar`; Test `test/test_morphology_layer.py` + extend `test/test_when_grammar_rules.py`.

- [ ] **Step 1: Failing tests** -- `MorphologyLayer.rule_name == "morphology"`, `tier == 'C'`; construction parameter-free (`cls()`); `forward` on a surface token's event normalizes `.what` to the lemma's concept and routes tense/aspect onto `.when` by **delegating to `TenseLayer`/`AspectLayer`** (assert the `.when` tail matches what those layers produce for the analyzed features -- do not re-derive the math); `reverse` recovers base+features; the `"morphology"` rule is present in the loaded grammar (`Grammar().load_from_grammar_file("role_collapsed.grammar")`, method-name set) and resolves via `_resolve_rule_layer('C','morphology')`.
- [ ] **Step 2: Run** $\to$ FAIL.
- [ ] **Step 3: Implement** -- `MorphologyLayer(GrammarLayer)` following the `PrepositionLayer`/`TenseLayer` pattern (same `__init__` signature, `compose`/`generate` dispatch): `forward` calls `surface_morphology.analyze`, sets `.what` to the lemma and applies the tense/aspect features via the existing `TenseLayer`/`AspectLayer` (compose, do not duplicate); `reverse` synthesizes. Add `'morphology': MorphologyLayer,` to `GRAMMAR_LAYER_CLASSES` and `'morphology': T1_UNARY_AFFIX,` to `_OPERATOR_SURFACE_SCHEMAS`; add `<rule>morphology_O1 = morphology.forward(morphology_I1)</rule>` (`<compose>`) and `<rule>morphology_I1 = morphology.reverse(morphology_O1)</rule>` (`<generate>`) to `data/role_collapsed.grammar`.
- [ ] **Step 4: Run** $\to$ PASS; re-run `test/test_role_collapsed_grammar.py` (no-POS guard) $\to$ green.
- [ ] **Step 5: Commit (Alec)** -- `feat: add MorphologyLayer (lemma + feature routing over the converged substrate)`

> **Flagged design points (confirm during Task 7.2, spec Section 12):** unary vs marker-based (absorb/emit affixes); exact feature$\to$modality routing (which features touch `.where`/`.activation` beyond tense/aspect$\to$`.when`); keep the division of labour so `surface_tense`/`TenseLayer`/`AspectLayer` are reused, not re-implemented.

---

## Self-Review Against Spec

| Spec section | Plan task(s) |
|---|---|
| 1 (glossary) + stale docs | Phase 5 |
| 2-3 (flow, canonical table) | 1.1, 2.1, 2.2, 2.4 |
| 4 (`.where` decoupling) | Phase 4 |
| 5, 6.4 (event$\to$event ops; LIFT/LOWER/PREPOSITION) | Phase 3 |
| 6.1-6.3 (size invariant, read sites, CS fix, downstream) | 2.1, 2.2 |
| 6.5 / mandatory codebooks | 2.3 |
| 7 (migration) | 2.4, 6.1 |
| 11 (success criteria) | 6.2 |
| 12 (MorphologyLayer) | Phase 7 |

**Already-built prerequisites** (unit-bracket `.when`, contextual-bind ops, width guards) are listed under "Working-tree starting state" and are not re-done. **Flagged design decisions** (CS operand details, LIFT/LOWER span + PREPOSITION-where semantics, `.where`-decoupling reconciliation, MorphologyLayer routing) are called out at their tasks for confirmation during execution.

## Risks

1. **Phase 2 has a red interior** (2.1 flips reads before 2.2/2.4 fix CS + configs). Keep 2.1-2.4 in one working session; the gate is the final green (the build-all guard), not per-task green.
2. **`.where` decoupling** (Phase 4) intersects `doc/plans/2026-05-28-where-keyed-taxonomy.md`; reconcile, do not duplicate; STOP and escalate on conflict.
3. **LIFT/LOWER `.when` span + PREPOSITION `.where` semantics** (Phase 3.2) are partly a design choice -- confirm against spec Section 5; fixtures lock the chosen values.
4. **Per-config migration** (~30 files) -- the build-all guard is the safety net; drop (don't bump) explicit dim overrides; `test_basicmodel.py` is order-fragile (judge against isolation).
5. **MorphologyLayer routing** (Phase 7) -- keep it delegating to the Phase-4 tense/aspect ops; do not re-implement the `.when` math.
