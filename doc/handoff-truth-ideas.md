# Handoff — Truth / Ideas processing (next session)

> **STATUS — EXECUTED 2026-06-18.** Stages 1–5 are built behind the `<truthIdeas>` gate
> (flag-off byte-identical; full suite 2610/0). See `doc/specs/mereological-order-raising.md`
> § "Truth / Ideas processing" → "Handoff build stages" for the per-stage landing notes and the
> `ConceptualSpace` methods (`_route_learned_relation` / `stm_end_state_trust` / `reason` /
> `verify_relation`, …) + `test/test_truth_ideas_routing.py` (25 tests). Stage 6 deferred; the
> episodic SOURCE (pre-parsed Wikipedia / `.events` LTM) is the one remaining FUTURE TODO. The
> brief below is retained as the original design handoff.

You are continuing work on the `basicmodel/` submodule of WikiOracle. Read these first:
- **Spec:** `doc/specs/mereological-order-raising.md` — esp. the new section **"Truth / Ideas
  processing (DESIGN, approved 2026-06-18)"** at the end (the approved design + build stages), and the
  updated **"Critical path"** status note.
- **Memory:** the auto-memory index + `s3-symbol-table-relocation.md`,
  `meronomy-order0-vs-symbolic-subsumption.md`, `xor-paths-and-mereological-order-raising.md`.

## Where things stand (all green, uncommitted on top of the last commit)

The prior session took the **relation-level / bridge-independent** mereology work essentially to
completion. Landed (gated `<mereologyRaise>` / `<serialObjectMeta>`, byte-identical with flags off):
- **Relation-only CS symbol table** (`_sym_*`) + **word/object/meta (A/B/C)** creation
  (`ConceptualSpace.create_word_object_meta`), live-wired.
- **S3** as ownership-by-reference (`terminalConceptualSpace_ref` + CS forwarding relation read-API).
- **Serial dual view:** 2b PS hard-mask to the active word (`word_span_window`) + per-word commit gate;
  2c WS reads the unity at the §6c prelude's pump 0 (`_stage0_unity_forward`). Both PS and WS see the
  input, then subsymbolic, then symbolic.
- **§6c sentence-protocol prelude default-ON in serial** (`symbolicOrder>=1`) — supplies whole-sentence
  context as the gist/intent.
- **Over-collection lifecycle** (`refine_over_collected` + `synthesize_higher_order`, the σ AND-combine),
  live in `_autobind_cross_tower`; **`PiLayer.factorize_over_set`** (the π inverse primitive).
- **SS→WS** terminology rename (comments/locals; load-bearing ids — `wholeSpace`, `symbolicOrder`,
  the `"ws"` `_pos_kind` tags — preserved).
- Config: `data/MM_mereology.xml` (parallel) + `data/MM_mereology_serial.xml` (serial). **NEVER mutate
  `data/MM_20M.xml`.**

Full suite: **2585 passed / 0 failed**.

## Your task: build the Truth / Ideas processing

Implement the approved design in the spec's "Truth / Ideas processing" section. Key points:
- **Entities (codes + ideas, any order) are ABSOLUTE; relations between them are RELATIVE (trust-
  bearing).** Codebooks are assumed-valid; testimony writes **relations**, never entities.
- **Reducibility:** try to reduce a parsed predication to symbols; if ineffable, approximate with a
  longer-description **idea**. Effable → relation over codes; else → relation over ideas.
- **Idea identity** is graded by shared parts/wholes (collection-based).
- **Reasoning** = modus ponens over relations: given `A→B` (trust `t₁`) and `C` (trust `t₂`), map C to
  antecedent A by **parthood** (raise/lower order to compare); if C ⊆ A, apply consequent B to C → a
  new concept with trust `t₁×t₂`; this expands the **area of luminosity**.
- **LTM is persisted STM:** serial = 1 position (absolute) or 2 positions (relative; parthood over
  extension); parallel = full N≈8 ideas; over time = the STM stack. Stateless ⇒ LTM is **provisioned**
  (pre-parsed), not accumulated.

### Build stages
1. **Map** the existing `RelativeTruthStore` / `TruthLayer` / `_maybe_learn_relation` /
   `learn_relations_from_stm` APIs (PREREQUISITE — wire onto them, do not duplicate).
2. **Store routing + trust** — parse → reduce-or-describe → relation-with-trust into the relative store;
   entities stay absolute.
3. **STM→LTM persistence** — 1/2-position serial, N-stack parallel; the provisioning format.
4. **Reasoning engine** — modus ponens + parthood-matching (order raise/lower) + `t₁×t₂` + luminosity.
5. **Verification** — episodic order-0 parthood (`.where`/co-occurrence) → trust update.
6. **(deferred nice-to-haves)** too-many-wholes OR-combine; the π-split; the σ tower-codebook
   geometric realization (`SigmaLayer.synthesize_over_set`).

## Working constraints
- Run pytest from `basicmodel/` on CPU: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/ -q`. Baseline ~2585 passed / 0 failed.
- Gate new behavior; keep flag-off byte-identical. **Do NOT edit source files while a full suite runs**
  (the `inspect.getsource` contract tests false-fail on line-shift).
- **Do NOT `git commit`** — Alec makes commits; leave changes in the working tree and flag good commit
  points.
- Start with stage 1 (the mapping) before building; confirm the design with Alec where it touches
  load-bearing non-gated paths.
