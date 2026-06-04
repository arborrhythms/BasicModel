# Single-Architecture Convergence -- Design Spec

> **Status:** design (approved shape, pre-plan). Next step: `superpowers:writing-plans` turns this into a task-by-task implementation plan.
>
> **Save location:** `doc/plans/2026-06-03-single-architecture-convergence-design.md`.
>
> **Git:** Per repo convention **Alec performs all git writes** -- this file is written but not committed; commit + review before the plan is generated.

**Goal:** Converge `basicmodel` on a single architecture by making the spatial/temporal carriers (`.where` / `.when`) and the Perceptual/Symbolic codebooks **architectural constants instead of per-config options**, and by unifying the `.when` encoding on a zero-duration-free **unit-bracket** convention. Every existing model config is migrated (none removed); pre-convergence checkpoints are retrained (no shim).

**Motivation:** Config variation across `.where` / `.when` / codebook presence is a recurring source of shape-fragility and per-config special-casing. Fixing the canonical shape removes a class of bugs and lets downstream work (e.g. the MorphologyLayer) target one substrate. `.when` specifically is needed at **every** tier because parallel-mode operation builds **behavioral sequences**, which require a temporal index everywhere -- not just on the input/perceptual carriers.

---

## Target architecture (the canonical shape)

| tier | `.where` | `.when` | codebook |
|---|---|---|---|
| IS (Input) | 2 | 2 | as today |
| PS (Perceptual) | 2 | 2 | **mandatory** |
| SS (Symbolic) | 2 (was 0) | 2 (was 0) | **mandatory** |
| CS (Conceptual) | 0 (muxed on `.event`) | 2 (new) | as today |
| OS (Output) | 0 (muxed / derived) | 2 (new) | as today |

- **`.when` $=2$ in all five tiers** -- the temporal index is universal (parallel-mode behavioral sequences).
- **`.where` $=2$ only on the real carriers (IS / PS / SS)** -- CS and OS are muxed/derived representations on `.event` and carry no separate spatial index.
- **SS is promoted** from `0/0` to `2/2` and gains a mandatory codebook -- the largest single representational change here.
- The muxed event layout per tier remains `[what | where | when]`; `muxedSize = nWhat + nWhere + nWhen` (`bin/Spaces.py:4472`).

---

## Decisions locked (from brainstorming)

| Question | Decision |
|---|---|
| `.where` / `.when` configurability | **Architectural constants**, not config options. The `<nWhere>` / `<nWhen>` tags are removed from every XML; a single per-tier source of truth in code supplies the canonical shape. |
| Which tiers get `.when` | **All five** (IS/PS/SS/CS/OS), for parallel-mode behavioral sequences. |
| Which tiers get `.where` | **IS/PS/SS only**; CS/OS stay muxed on `.event` (`where=0`). |
| PS / SS codebooks | **Mandatory** -- `codebook=none` is forbidden for these two tiers. |
| `.when` encoding | **Unit-bracket** convention everywhere, incl. the event-muxing `encode` (see below). |
| Config migration | **Migrate all configs; remove none.** Absorb the added widths by adjusting the muxed budget per config. |
| Checkpoint compatibility | **No shim.** Pre-convergence checkpoints are retrained; confirm `autoload` defaults stay safe. |
| Scope breadth | **Bounded** to the items above (no broader option cull in this effort). |
| Phases | **All phases implemented**, including the original-plan Phase 7 (MorphologyLayer), which is designed separately on the converged substrate. |

---

## Mechanisms

### 1. `.where` / `.when` as architectural constants

Today both are read per-section from config: `TheXMLConfig.space(section, "nWhere"|"nWhen")` (`bin/Spaces.py:6303-6304`), the architecture-level `TheXMLConfig.get("architecture.nWhere"|"nWhen")` (`bin/Models.py:567-568`), the per-section reads at `bin/Models.py:384-388`, the upward-tier reads at `bin/Spaces.py:8024-8025`, and `bin/util.py:994`.

**Change:** introduce one canonical per-tier table (e.g. `CANONICAL_WHERE` / `CANONICAL_WHEN` keyed by tier, or a small helper `canonical_shape(tier) -> (nWhere, nWhen)`) as the single source of truth. Every read site consults it instead of the config. The `<nWhere>` / `<nWhen>` tags are stripped from the XMLs; a stale tag, if present, is ignored with a warning (the plan finalizes ignore-vs-reject).

`nWhat` is **derived** (`nWhat = outputShape[1] - nWhere - nWhen`, `bin/Spaces.py:4471`), so growing `nWhere`/`nWhen` automatically shrinks `nWhat` for a fixed `outputShape[1]`; the migration must size `outputShape[1]` (and `nVectors`) so `nWhat` stays valid (see Migration).

### 2. `.when` everywhere; `.where` on carriers

CS and OS gain `nWhen=2`. Their `.event`-muxed representation must carry the 2 `.when` slots; the mux/demux/reverse path (`SubSpace.decode`, `bin/Spaces.py:6097-6152`) and the `.when` loss (`whenScale`, `bin/Models.py:610`) must stay finite at the new widths -- the same width-guard discipline Phase 3 applied for the 2-wide `.when` on the carriers, now extended to CS/OS.

### 3. Mandatory PS / SS codebooks

Codebook presence is gated by `codebook_mode` (`Space.normalize_codebook_mode(...)`, read from the per-space `<codebook>` tag; `bin/Models.py:4336-4339`), with `"none"` meaning no codebook and the `codebook_slot` (`event`/`what`/`None`) set at `bin/Spaces.py:4499-4514`; the truthiness shim is `codebook_mode != "none"` (`bin/Spaces.py:6481`).

**Change:** PS and SS must always have a codebook -- `codebook=none` is forbidden for these two tiers (the mode itself, e.g. `quantize`, may stay configurable; only the `none` opt-out is removed). IS/CS/OS codebook config is untouched. Configs that currently disable PS/SS codebooks are migrated -- notably `data/XOR_exact.xml`, which sets `<codebook>none</codebook>` on SymbolicSpace (lines ~76-80) and drives `test_xor_perfect_reconstruction`.

### 4. Unit-bracket `.when` encoding (folds in the approved bracket change)

The `.when` key currently has two inconsistent point conventions: event-muxing `encode(t) = q(t)` (magnitude $1$) vs. tense/aspect `encode_range(0,0)` (magnitude $2$). A magnitude-$1$ point fed to the range `decode` recovers the correct center but a spurious duration. Resolution (verified suite-safe):

- **`encode(t)` becomes `encode_range(t-0.5, t+0.5)`** -- a single time becomes a unit-width bracket; magnitude $\approx 1.998$, decodes to $(t-0.5,\ t+0.5)$.
- **Present default** and **`SIMPLE` aspect** become unit brackets `(r-0.5, r+0.5)` instead of zero-width points.
- **`PERFECT` / `PROGRESSIVE`** are re-derived relative to the bracketed reference; `aspect_interval` (`bin/Spaces.py` `WhenRangeEncoding.aspect_interval`) updates accordingly.
- **Tense rotation is unchanged** -- it already maps the present unit-bracket onto past/future unit-brackets: `rotate(present, -1)` gives `(-1.5, -0.5)` and `rotate(present, +1)` gives `(0.5, 1.5)`, bracketing the past/future references at $-1$ and $+1$.
- Phase 4 (`test/test_tense_aspect.py`) and Phase 5 (`test/test_grammar_fixtures.py`) fixtures update to the new expected ranges.

Benefits: encode/decode become mutual inverses; every `.when` value carries a meaningful, recoverable duration; the decode sits off the radius-$2$ singularity that caused the arccos precision wobble at zero duration.

---

## Migration strategy

**Every config is migrated; none is removed** (`data/*.xml`, ~30 files).

- **Strip** `<nWhere>` / `<nWhen>` tags (now architectural constants).
- **Enforce** PS/SS codebook (drop/migrate `<codebook>none</codebook>` on those tiers).
- **Compensate sizes** so each config stays valid and downstream shapes hold. Because `nWhat` is derived from `outputShape[1]`, the per-config work is sizing `outputShape[1]` / `nVectors` so the added `.where`/`.when` widths are absorbed and `nWhat` stays sane. Exact per-config arithmetic is enumerated in the implementation plan.
- Configs already broken for reasons unrelated to `.where`/`.when` (e.g. `MM_5M` `reconstruct=concepts`, `MM_400M` butterfly N-halving, `MM_shamatha` / `MM_xor_step4` `nVectors != nActive`, per `test/test_use_flags.py`) are **not removed** and their unrelated breakage is **out of scope** -- they are migrated for the converged shape only.

**Tests to migrate** (shape-dependent):
- `test/test_basicmodel.py` -- `TestSubSpaceDerivedSizes` (`~:603`, `WhereEncoding`/`WhenEncoding(10000,2)` size assertions) and `TestWhenEncodingRoundTrip` (`~:557`) for any new widths.
- `test/test_use_flags.py` -- per-config instantiation expectations.
- `test/test_basicmodel.py::TestReconstructionSymbols` -- XOR reconstruction (SS codebook now mandatory; `XOR_exact.xml` migrated).
- Phase 4/5 fixtures (`test/test_tense_aspect.py`, `test/test_grammar_fixtures.py`) for the bracket `.when` values.
- Any test asserting a muxed width or `nWhat` for CS/OS now that they carry `.when`.

**Compatibility:** no shim. Pre-convergence checkpoints will not load (SS and CS/OS widths change); they are retrained. Confirm `autoload` defaults are safe so a stale checkpoint is never silently loaded against the new shape.

---

## Risks / uncertainties

1. **Every tier's muxed layout changes.** SS most (gains `where=2`/`when=2` + mandatory codebook); CS/OS gain `when=2`. Each tier's mux / demux / reverse / reconstruction / loss must be width-guarded -- the Phase 3.3 `.when` width guard generalized to SS/CS/OS. **Highest-risk area.**
2. **SS promotion touches the symbolic-conceptual interface.** `_tie_lexicon_to_codebook` (`bin/Spaces.py:3389`) and the SS/CS orth (`bin/Spaces.py:3660`) assume the current SS shape; both need review when SS gains `.where`/`.when` and a mandatory codebook.
3. **Per-config size compensation is error-prone.** Each XML needs the right `outputShape`/`nVectors` so `nWhat` stays positive and downstream layers match. The plan enumerates this per config; a width-assertion test per migrated config is the guard.
4. **Bracket `.when` ripples into aspect math.** `PERFECT`/`PROGRESSIVE` relative to a bracketed reference must be re-derived; fixtures lock the new values.
5. **Order-fragility of `test_basicmodel.py`.** Per project memory the suite is order-fragile (batch failures are not necessarily regressions); migration regressions are judged against an isolation baseline, not a single full-suite run.

---

## Out of scope

- Broader config-option cull beyond `.where` / `.when` / PS-SS codebooks.
- Fixing legacy configs broken for reasons unrelated to `.where`/`.when`.
- Phase 7 (MorphologyLayer) itself -- designed and implemented separately, after this convergence, on the converged substrate (it is still on the docket; "all phases" includes it).

---

## Success criteria

- No `<nWhere>` / `<nWhen>` tags remain in `data/*.xml`; the canonical per-tier shape is the single source of truth in code; IS/PS/SS report `where=2/when=2`, CS/OS report `where=0/when=2`.
- PS and SS always have a codebook; `codebook=none` for them is rejected.
- `.when` uses the unit-bracket convention everywhere; `encode`/`decode` are mutual inverses; Phase 4/5 fixtures pass at the new values.
- Every migrated config instantiates and runs a finite forward (incl. finite `.when` loss) at the new shape; `test/test_use_flags.py` and the reconstruction tests pass.
- `test/test_role_collapsed_grammar.py` and `test/test_d1_pos_recovery_gate.py` stay green (no POS leakage from any incidental change).
- `test_basicmodel.py` failures do not exceed the isolation baseline.
