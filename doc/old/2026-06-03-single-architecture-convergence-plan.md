# Single-Architecture Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design spec:** `doc/plans/2026-06-03-single-architecture-convergence-design.md` (approved).
>
> **Git:** Per repo convention **Alec performs all git writes** -- pause at each "Commit (Alec)" checkpoint rather than running `git`.

**Goal:** Converge `basicmodel` on one architecture: `.where`/`.when` and PS/SS codebooks become architectural constants (not per-config), `.when` lives on every tier, SS is promoted to a full `2/2` carrier with a codebook, and `.when` uses a unit-bracket encoding -- migrating all model configs and tests, no checkpoint shim.

**Architecture:** A single per-tier canonical-shape source of truth in code replaces the `<nWhere>`/`<nWhen>` config tags; the muxed event layout `[what | where | when]` is unchanged but the per-tier widths are fixed (IS/PS/SS `where=2,when=2`; CS/OS `where=0,when=2`). `WhenRangeEncoding` point times become unit brackets so `encode`/`decode` are mutual inverses. Every XML is resized to absorb the new widths (keeping `nWhat` constant); PS/SS codebooks may no longer be `none`.

**Tech Stack:** Python 3, PyTorch, pytest (`.venv/bin/python -m pytest`), XML config, the `Encoding`/`SubSpace`/`Space`/`GrammarLayer` substrate.

---

## Context

The design spec gives the rationale. Key grounded facts (verify before editing):

- `.where`/`.when` read sites: per-section `TheXMLConfig.space(section, "nWhere"|"nWhen")` (`bin/Spaces.py:6303-6304`); architecture-level `TheXMLConfig.get("architecture.nWhere"|"nWhen")` (`bin/Models.py:567-568`); per-section reads `bin/Models.py:384-388`; upward-tier `bin/Spaces.py:8024-8025`; `bin/util.py:994`.
- `nWhat` is DERIVED: `self.nWhat = outputShape[1] - self.nWhere - self.nWhen` (`bin/Spaces.py:4471`); `self.muxedSize = self.nWhat + self.nWhere + self.nWhen` (`:4472`, `:6305`). So growing `nWhere`/`nWhen` at fixed `outputShape[1]` shrinks `nWhat`.
- `.when` encoding: `WhenRangeEncoding` (`bin/Spaces.py:394`), with `encode` currently the inherited single-endpoint `q(t)` (the `# NOTE` block), `encode_range`, `decode`, `rotate`, `aspect_interval`; construction `WhenRangeEncoding(64, _nWhen)` (`bin/Spaces.py:6314`).
- Tense/aspect ops: `TenseLayer` / `AspectLayer` (`bin/Language.py:2597` / `:2622`); `AspectLayer.forward` reads `r` from the decoded END endpoint (`:2636`).
- Codebook gating: `<codebook>` per-space tag $\to$ `Space.normalize_codebook_mode(...)`; `bin/Models.py:4336-4339`; `codebook_slot` set `bin/Spaces.py:4499-4514`; truthiness `codebook_mode != "none"` (`bin/Spaces.py:6481`).
- SS/CS interface (review when SS gains shape): `_tie_lexicon_to_codebook` (`bin/Spaces.py:3389`), SS/CS orth (`bin/Spaces.py:3660`).
- Model build idiom (tests): `init_config(path, defaults_path="data/model.xml")`; `Language.TheGrammar._configured = False`; `model, _cfg = Models.BasicModel.from_config(path)` (see `test/test_use_flags.py:19-25`). Per-tier shape is introspectable via `model.named_modules()` carriers exposing `.nWhen` / `.whenEncoding` / `.nWhere` (see the Phase 6 guard in `test/test_when_grammar_rules.py`).
- **Baseline:** `test/test_basicmodel.py` is order-fragile; judge regressions against an isolation run (currently 205 passed / 2 skipped / 0 failed in isolation).
- **Shared test preamble** (every new test file):
  ```python
  import math, os, sys, unittest
  from pathlib import Path
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
  import torch
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
  ```

## Canonical per-tier shape (the target, used throughout)

| section | nWhere | nWhen |
|---|---|---|
| InputSpace | 2 | 2 |
| PerceptualSpace | 2 | 2 |
| SymbolicSpace | 2 | 2 |
| ConceptualSpace | 0 | 2 |
| OutputSpace | 0 | 2 |
| WordSpace | 0 | 0 |

> **Verify during Phase 3:** `WordSpace` is set to `0/0` (lexicon space, no event slots) as the safe default; confirm no construction path expects otherwise. If a section name differs in code, map it to this table.

## File Structure

**New files**
- `bin/architecture.py` -- the single source of truth: `canonical_shape(section) -> (nWhere, nWhen)` plus a `MANDATORY_CODEBOOK_TIERS = {"PerceptualSpace", "SymbolicSpace"}` constant. Pure, no torch.
- `test/test_canonical_shape.py`, `test/test_when_bracket.py`, `test/test_convergence_configs.py`, `test/test_mandatory_codebook.py`.

**Modified files**
- `bin/Spaces.py` -- `WhenRangeEncoding.encode` (bracket) + `aspect_interval` (re-derived); `SubSpace`/`Space` construction reads `canonical_shape` instead of config (`:6303-6314`); width-guards for SS/CS/OS muxing.
- `bin/Language.py` -- `AspectLayer.forward` reads `r` from center; `TenseLayer`/`AspectLayer` unchanged otherwise.
- `bin/Models.py` -- read sites `:384-388`, `:567-568` use `canonical_shape`; PS/SS codebook forced non-`none` (`:4336-4339`).
- `bin/util.py` -- `:994` uses `canonical_shape("InputSpace")`.
- `data/*.xml` (~30) -- strip `<nWhere>`/`<nWhen>`; resize `outputShape`/`nVectors`; PS/SS `<codebook>` not `none`.
- `test/test_basicmodel.py`, `test/test_tense_aspect.py`, `test/test_grammar_fixtures.py`, `test/test_use_flags.py` -- updated for new widths/values.

---

## Phase 1 -- Unit-bracket `.when` encoding

`.when` point times become unit brackets so `encode`/`decode` are mutual inverses and every `.when` carries a recoverable duration. **Aspect semantics (LOCKED, adjustable):** `AspectLayer` reads reference `r` from the interval **center**; `SIMPLE = (r-0.5, r+0.5)`, `PERFECT = (r-1.0, r)`, `PROGRESSIVE = (r-1.0, r+1.0)`; tense rotates the whole key (unchanged). Present default stamp = `encode_range(-0.5, 0.5)`.

### Task 1.1: `WhenRangeEncoding.encode` -> unit bracket

**Files:** Modify `bin/Spaces.py` (the `# NOTE` block where `encode` is intentionally not overridden, just before `def decode`); Test `test/test_when_bracket.py`.

- [ ] **Step 1: Write the failing test** (`test/test_when_bracket.py`, preamble + `from Spaces import WhenRangeEncoding`)
```python
def _enc(): return WhenRangeEncoding(64, 2)

def test_encode_is_unit_bracket_and_inverts_decode():
    enc = _enc()
    for t in (-1.5, -1.0, 0.0, 0.5, 2.0):
        key = enc.encode(t)
        assert torch.allclose(key, enc.encode_range(t - 0.5, t + 0.5), atol=1e-6)
        ds, de = enc.decode(key)
        assert math.isclose(float(ds), t - 0.5, abs_tol=2e-3)
        assert math.isclose(float(de), t + 0.5, abs_tol=2e-3)

def test_encode_tensor_input_round_trips():
    enc = _enc(); ts = torch.tensor([-1.0, 0.0, 1.0])
    ds, de = enc.decode(enc.encode(ts))
    assert torch.allclose((ds + de) / 2.0, ts, atol=2e-3)   # centers recover the times
```
- [ ] **Step 2: Run, verify it fails** -- `.venv/bin/python -m pytest test/test_when_bracket.py -v` $\to$ FAIL (current `encode` is single-endpoint `q(t)`, magnitude 1; decode of it does not give `(t-0.5, t+0.5)`).
- [ ] **Step 3: Implement** -- replace the `# NOTE: encode() is the inherited...` comment block in `bin/Spaces.py` (just before `def decode`) with:
```python
    def encode(self, offsets):
        """Point time(s) -> unit-width bracket key encode_range(t-0.5, t+0.5).
        A single stamped time is the unit window [t-0.5, t+0.5] (magnitude
        ~1.998), so encode is the mutual inverse of the range decode and every
        .when carries a recoverable duration. Used by the event-muxing sites
        (whenEncoding.encode) and anywhere a scalar/tensor time is stamped."""
        return self.encode_range(offsets - 0.5, offsets + 0.5)
```
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit (Alec)** -- `feat: encode .when point times as unit brackets (encode == inverse of range decode)`

### Task 1.2: `aspect_interval` re-derivation + `AspectLayer` reads center

**Files:** Modify `bin/Spaces.py` (`WhenRangeEncoding.aspect_interval`); Modify `bin/Language.py` (`AspectLayer.forward`, `:2636`); Test `test/test_when_bracket.py` (extend).

- [ ] **Step 1: Append failing tests**
```python
def test_aspect_interval_bracket_shapes():
    enc = _enc()
    assert enc.aspect_interval(0.0, "SIMPLE")      == (-0.5, 0.5)
    assert enc.aspect_interval(0.0, "PERFECT")     == (-1.0, 0.0)
    assert enc.aspect_interval(0.0, "PROGRESSIVE") == (-1.0, 1.0)
    assert enc.aspect_interval(-1.0, "SIMPLE")     == (-1.5, -0.5)

def test_aspect_layer_reads_center_not_end():
    from Language import AspectLayer
    enc = _enc()
    # event .when = a NON-symmetric range whose center is -1: (-1.5, -0.5).
    head = torch.randn(1, 1, 6)
    x = torch.cat([head, enc.encode_range(-1.5, -0.5).expand(1, 1, -1)], dim=-1)
    a = AspectLayer(); a.set_op("SIMPLE")
    ds, de = enc.decode(a.forward(x)[..., -2:])      # center -1 -> SIMPLE (-1.5, -0.5)
    assert math.isclose(float(ds.reshape(-1)[0]), -1.5, abs_tol=0.05)
    assert math.isclose(float(de.reshape(-1)[0]), -0.5, abs_tol=0.05)
```
- [ ] **Step 2: Run, verify it fails** (current `aspect_interval`: SIMPLE `(r,r)`, PERFECT `(r-1,r)`, PROGRESSIVE `(r-eps,r+eps)`; current `AspectLayer` reads END).
- [ ] **Step 3: Implement.** In `bin/Spaces.py`, replace `WhenRangeEncoding.aspect_interval` body with:
```python
    @staticmethod
    def aspect_interval(r, kind, eps=0.25):
        """(start, end) for reference r (the interval CENTER) and aspect kind.
        Unit-bracket convention: SIMPLE is a unit window at r; PERFECT a unit
        window ending at r (completed, relevant at r); PROGRESSIVE a 2-wide
        window spanning r (extended/ongoing). eps retained for API compat."""
        r = float(r)
        if kind == "SIMPLE":      return (r - 0.5, r + 0.5)
        if kind == "PERFECT":     return (r - 1.0, r)
        if kind == "PROGRESSIVE": return (r - 1.0, r + 1.0)
        raise ValueError(f"unknown aspect kind {kind!r}")
```
In `bin/Language.py` `AspectLayer.forward`, change the reference read from END to CENTER:
```python
    def forward(self, x):
        head, when = self._split_when(x); enc = self._when_encoding()
        start_t, end_t = enc.decode(when)
        flat_s = start_t.reshape(-1); flat_e = end_t.reshape(-1)
        r = (float(flat_s[0]) + float(flat_e[0])) / 2.0 if flat_e.numel() else 0.0
        s, e = enc.aspect_interval(r, self._op, eps=self._eps)
        key = enc.encode_range(s, e).to(x.device).expand(*when.shape[:-1], -1)
        return torch.cat([head, key], dim=-1)
```
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit (Alec)** -- `feat: re-derive .when aspect intervals for the unit-bracket convention (center-anchored)`

### Task 1.3: Present-default stamp -> unit bracket

**Files:** Modify `bin/Spaces.py` (`WhenRangeEncoding.forward`); Test `test/test_when_bracket.py` (extend).

- [ ] **Step 1: Append failing test**
```python
def test_forward_stamps_present_unit_bracket():
    enc = _enc(); y = enc.forward(torch.zeros(2, 3, 10))
    ds, de = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(ds), -0.5, abs_tol=2e-3) and math.isclose(float(de), 0.5, abs_tol=2e-3)
```
- [ ] **Step 2: Run, verify it fails** (current `forward` stamps `encode_range(0,0)`).
- [ ] **Step 3: Implement** -- in `WhenRangeEncoding.forward`, change the stamped key from `self.encode_range(0.0, 0.0)` to `self.encode_range(-0.5, 0.5)` (the present unit bracket). Keep the `nDim == 0` early return and the slot-index math unchanged.
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit (Alec)** -- `feat: stamp present .when as a unit bracket (-0.5, 0.5)`

### Task 1.4: Update Phase 4 tense/aspect fixtures

**Files:** Modify `test/test_tense_aspect.py`.

- [ ] **Step 1: Recompute expected values** for the unit-bracket scheme and update the `.when` assertions:
  - `_event_with_present_when` builds `.when = encode_range(-0.5, 0.5)` (present bracket).
  - PAST identity-aspect: `rotate(present, -1) -> (-1.5, -0.5)`.
  - PRESENT identity: unchanged.
  - PERFECT on present (center 0): `(-1.0, 0.0)`.
  - PROGRESSIVE on present (center 0): `(-1.0, 1.0)`.
  Set each assertion's expected `(start, end)` accordingly, `abs_tol=0.05`. The class-contract / set_op-validation / head-pass-through tests are unchanged.
- [ ] **Step 2: Run** -- `.venv/bin/python -m pytest test/test_tense_aspect.py -v` $\to$ PASS.
- [ ] **Step 3: Commit (Alec)** -- `test: update tense/aspect fixtures for unit-bracket .when`

### Task 1.5: Update Phase 5 spec fixtures

**Files:** Modify `test/test_grammar_fixtures.py`.

- [ ] **Step 1: Recompute the tense/aspect fixtures (5-8)** under the bracket scheme; the PREPOSITION/BIND fixtures (1-4) are unaffected.
  - `Alice ran` (PAST SIMPLE): present bracket `(-0.5,0.5)` -> SIMPLE (center 0) `(-0.5,0.5)` -> PAST rotate `-1` -> `(-1.5, -0.5)`.
  - `Alice is running` (PRESENT PROGRESSIVE): PROGRESSIVE (center 0) `(-1.0, 1.0)`.
  - `Alice has run` (PRESENT PERFECT): PERFECT (center 0) `(-1.0, 0.0)`.
  - `Alice had been running` (PAST PERFECT PROGRESSIVE, innermost-first): PROGRESSIVE(present)=`(-1,1)` (center 0); PERFECT(center 0)=`(-1,0)`; PAST rotate `-1` -> `(-2.0, -1.0)`.
  Update each assertion with `abs_tol=0.05`.
- [ ] **Step 2: Run** -- `.venv/bin/python -m pytest test/test_grammar_fixtures.py -v` $\to$ PASS (8 fixtures).
- [ ] **Step 3: Commit (Alec)** -- `test: update spec fixtures for unit-bracket .when`

---

## Phase 2 -- Width-guard SS/CS/OS muxing at the target widths

De-risk the substrate BEFORE flipping config-driven shapes: prove a `SubSpace` constructed directly at the new per-tier widths muxes/demuxes/reverses and reconstructs correctly. No config or canonical-shape change yet -- these tests construct `SubSpace`/codebook objects directly (mirror `test/test_when_range_encoding.py` Task 3.3 and the `SubSpace(...)` idiom there).

### Task 2.1: SS-shaped SubSpace (where=2, when=2, with codebook) round-trips

**Files:** Test `test/test_when_bracket.py` (extend) or a new `test/test_ss_shape.py`.

- [ ] **Step 1: Write the failing/guard test** -- construct a `SubSpace` with `whereEncoding=WhereEncoding(64, 2, 2)`, `whenEncoding=WhenRangeEncoding(64, 2)`, an SS-style codebook, and `inputShape`/`outputShape` whose width is `nWhat + 2 + 2`. Assert: `muxedSize == nWhat + 4`; a stamped event with a `.where` span and a `.when` bracket demuxes (`SubSpace.decode`) so both round-trip; the codebook slot coexists with the where/when tail. Use the real `SubSpace`/`Codebook` constructors (read `bin/Spaces.py:4447`+, `:6303`+ for signatures; mirror the existing Task 3.3 test).
- [ ] **Step 2: Run.** If it passes immediately, the muxing is already width-agnostic for SS -- record that and move on. If it fails, that is the width bug to fix in Step 3.
- [ ] **Step 3: Implement (only if Step 2 failed)** -- fix the hardcoded-width assumption it exposes (read `whenEncoding.nDim`/`whereEncoding.nDim` rather than a literal). Keep changes minimal and localized.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit (Alec)** -- `test: guard SS-shaped SubSpace mux/demux at where=2/when=2`

### Task 2.2: CS/OS-shaped SubSpace (where=0, when=2, muxed on .event) round-trips

**Files:** Test as above (extend).

- [ ] **Step 1: Write the guard test** -- construct a CS/OS-style `SubSpace` with `whereEncoding` width 0, `whenEncoding=WhenRangeEncoding(64, 2)`, codebook on the `.event` slot (`codebook_slot == 'event'`, `muxed == True`; `bin/Spaces.py:4503-4514`). Assert: `muxedSize == nWhat + 0 + 2`; a stamped `.when` bracket demuxes and round-trips; the `.event` muxing coexists with the 2 when-slots; the `.when` loss path stays finite.
- [ ] **Step 2: Run.** Pass $\to$ record; fail $\to$ fix in Step 3.
- [ ] **Step 3: Implement (only if needed)** -- fix any hardcoded `nWhen==0` assumption on the muxed (`.event`) path so CS/OS carry the 2 when-slots.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit (Alec)** -- `test: guard CS/OS-shaped (muxed .event) SubSpace with when=2`

---

## Phase 3 -- Canonical shape constant + flip + migrate all configs

Introduce the single source of truth, switch all read sites to it (this flips every config to the canonical shape at once), and migrate every XML to absorb the new widths. Phase 2 made the muxing robust, so the remaining risk is per-config sizing. This phase has a red interior (between the code flip and the last config migration); the Step ordering keeps it short and the final state green.

### Task 3.1: `bin/architecture.py` -- canonical shape source of truth

**Files:** Create `bin/architecture.py`; Test `test/test_canonical_shape.py`.

- [ ] **Step 1: Write the failing test**
```python
from architecture import canonical_shape, MANDATORY_CODEBOOK_TIERS
def test_canonical_shape_table():
    assert canonical_shape("InputSpace")      == (2, 2)
    assert canonical_shape("PerceptualSpace") == (2, 2)
    assert canonical_shape("SymbolicSpace")   == (2, 2)
    assert canonical_shape("ConceptualSpace") == (0, 2)
    assert canonical_shape("OutputSpace")     == (0, 2)
    assert canonical_shape("WordSpace")       == (0, 0)
def test_mandatory_codebook_tiers():
    assert MANDATORY_CODEBOOK_TIERS == {"PerceptualSpace", "SymbolicSpace"}
```
- [ ] **Step 2: Run, verify it fails** (`ImportError`).
- [ ] **Step 3: Implement** `bin/architecture.py` (pure, no torch):
```python
"""Single source of truth for the converged architecture's fixed per-tier
shape. .where/.when are no longer config options: every space's spatial/
temporal widths come from canonical_shape(section). .when is universal (every
tier; parallel-mode behavioral sequences); .where is on the real carriers
(IS/PS/SS) only -- CS/OS are muxed/derived on .event."""

_CANONICAL_SHAPE = {
    "InputSpace":      (2, 2),
    "PerceptualSpace": (2, 2),
    "SymbolicSpace":   (2, 2),
    "ConceptualSpace": (0, 2),
    "OutputSpace":     (0, 2),
    "WordSpace":       (0, 0),
}
MANDATORY_CODEBOOK_TIERS = {"PerceptualSpace", "SymbolicSpace"}

def canonical_shape(section):
    """(nWhere, nWhen) for a space section. Raises on an unknown section so a
    new tier cannot silently default to a wrong shape."""
    try:
        return _CANONICAL_SHAPE[section]
    except KeyError:
        raise ValueError(f"canonical_shape: unknown section {section!r}")
```
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit (Alec)** -- `feat: add canonical per-tier architecture shape (source of truth)`

### Task 3.2: Wire construction to `canonical_shape` (the flip)

**Files:** Modify `bin/Spaces.py:6303-6314`, `bin/Models.py:384-388` and `:567-568`, `bin/util.py:994`; Test `test/test_convergence_configs.py`.

- [ ] **Step 1: Write the failing test** (`test/test_convergence_configs.py`) -- build `MentalModel.xml` and assert the converged per-tier shape via `named_modules()` introspection:
```python
def _build(cfg):
    import os, warnings, Models, Language
    from util import init_config
    data = str(Path(__file__).resolve().parent.parent / "data")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        init_config(path=os.path.join(data, cfg), defaults_path=os.path.join(data, "model.xml"))
        Language.TheGrammar._configured = False
        m, _ = Models.BasicModel.from_config(os.path.join(data, cfg)); m.eval()
    return m

def test_mentalmodel_converged_shape():
    m = _build("MentalModel.xml")
    shapes = {n: (getattr(mod, "nWhere", None), getattr(mod, "nWhen", None))
              for n, mod in m.named_modules() if getattr(mod, "whenEncoding", None) is not None}
    # every subspace carrier reports when=2; where=2 on IS/PS/SS subspaces, 0 on CS/OS.
    assert any("inputSpace.subspace"      in n and v == (2, 2) for n, v in shapes.items()), shapes
    assert any("perceptualSpace.subspace" in n and v == (2, 2) for n, v in shapes.items()), shapes
    assert any("symbolicSpaces" in n and v == (2, 2) for n, v in shapes.items()), shapes
    assert all(v[1] == 2 for v in shapes.values()), shapes          # when=2 everywhere
```
- [ ] **Step 2: Run, verify it fails** (SS/CS/OS not yet at canonical shape; SS is `0/0`).
- [ ] **Step 3: Implement** -- at each read site, replace the config read with `canonical_shape(section)` (import from `architecture`). Concretely:
  - `bin/Spaces.py:6303-6304`: `_nWhere, _nWhen = canonical_shape(section)` (instead of the two `TheXMLConfig.space(...)` reads); keep `self.nWhere/_nWhen` assignments and the `WhereEncoding(...)/WhenRangeEncoding(64, _nWhen)` construction.
  - `bin/Models.py:384-388`: same substitution using the section name in scope.
  - `bin/Models.py:567-568`: `architecture.nWhere/nWhen` -- these are the architecture-level defaults; set from `canonical_shape("InputSpace")` (or the appropriate reference tier; verify which tier this `_objectSize` feeds).
  - `bin/util.py:994`: `nw, nn = canonical_shape("InputSpace"); return nw + nn`.
  **Verify** the section-name string available at each site matches the `canonical_shape` keys; map if needed.
- [ ] **Step 4: Run** -- the test from Step 1 should now pass IF the configs' `outputShape` is wide enough; configs whose `outputShape` is too narrow (negative `nWhat`) will raise. That is expected and fixed in Task 3.3. Confirm `MentalModel.xml` (already `where/when` heavy) builds.
- [ ] **Step 5: Commit (Alec)** -- `feat: drive per-tier .where/.when from canonical_shape (architectural constant)`

### Task 3.3: Migrate all model XMLs (resize + strip tags)

**Files:** Modify all `data/*.xml` (~30); Test `test/test_convergence_configs.py` (extend).

The transformation per config, per space section:
1. Let `(cw, cn) = canonical_shape(section)` and `(ow, on)` = the section's current `<nWhere>`/`<nWhen>` (default 0 if absent).
2. The section's content width `nWhat` must be preserved: `nWhat = outputShape[1] - ow - on` (old). Set the new muxed width `outputShape[1] += (cw - ow) + (cn - on)` (and `inputShape[1]` likewise where the section sets it) so `nWhat` is unchanged.
3. Remove the `<nWhere>`/`<nWhen>` tags (now ignored; `canonical_shape` rules).
4. If `section in {"PerceptualSpace", "SymbolicSpace"}` and `<codebook>` is `none`, change it to `quantize` (the canonical mode -- matches `MentalModel.xml`'s PerceptualSpace, `data/MentalModel.xml:61`); Phase 4 enforces this in code.

- [ ] **Step 1: Write the comprehensive failing test** -- every live config builds at the converged shape and runs a finite forward (live = all `data/*.xml` minus the known-broken-for-unrelated-reasons set per `test/test_use_flags.py`):
```python
# Broken before this work for reasons unrelated to .where/.when (test_use_flags.py):
# migrated for shape but NOT expected to instantiate.
_BROKEN = {"model.xml", "MM_5M.xml", "MM_400M.xml", "MM_shamatha.xml", "MM_xor_step4.xml"}

def _live_configs():
    data = Path(__file__).resolve().parent.parent / "data"
    return sorted(p.name for p in data.glob("*.xml") if p.name not in _BROKEN)

def test_all_live_configs_build_and_when_is_two():
    for cfg in _live_configs():
        m = _build(cfg)
        whens = [getattr(mod, "nWhen", None) for _n, mod in m.named_modules()
                 if getattr(mod, "whenEncoding", None) is not None]
        assert whens and all(w == 2 for w in whens), (cfg, whens)
```
  (If a config in the live set fails to build for a reason clearly unrelated to `.where`/`.when`/codebook, move it to `_BROKEN` with a one-line note rather than forcing it -- and say so in the report.)
- [ ] **Step 2: Run, verify it fails** (configs not yet resized; negative `nWhat` or width mismatches).
- [ ] **Step 3: Implement** -- apply the per-section transformation above to every `data/*.xml`. Do it mechanically and re-run after each handful; a `<nWhere>`/`<nWhen>` grep must come back empty when done:
  `grep -rl "nWhere\|nWhen" data/*.xml` $\to$ (empty). Configs already broken for unrelated reasons (`MM_5M`, `MM_400M`, `MM_shamatha`, `MM_xor_step4`, per `test_use_flags`) still get the tag-strip + resize but are not expected to instantiate; do not delete them.
- [ ] **Step 4: Run, verify it passes** (all live configs build; `.when == 2` everywhere).
- [ ] **Step 5: Commit (Alec)** -- `chore: migrate all model XMLs to the canonical shape (resize + strip nWhere/nWhen)`

---

## Phase 4 -- Mandatory PS/SS codebooks

### Task 4.1: Forbid `codebook=none` for PS/SS

**Files:** Modify `bin/Models.py:4336-4339` (and the codebook-mode resolution path); Test `test/test_mandatory_codebook.py`.

- [ ] **Step 1: Write the failing test** (define the `_build(cfg)` helper as in `test_convergence_configs.py`)
```python
def test_ps_ss_codebook_is_present_in_mentalmodel():
    m = _build("MentalModel.xml")
    modes = {n: getattr(mod, "codebook_mode", None) for n, mod in m.named_modules()}
    assert any("perceptualSpace" in n and v not in (None, "none") for n, v in modes.items()), modes
    assert any("symbolicSpaces"  in n and v not in (None, "none") for n, v in modes.items()), modes

def test_codebook_none_for_ss_is_rejected():
    # A config that sets SymbolicSpace <codebook>none</codebook> must raise at build.
    import os, tempfile, warnings
    import xml.etree.ElementTree as ET
    import Models, Language
    from util import init_config
    data = str(Path(__file__).resolve().parent.parent / "data")
    tree = ET.parse(os.path.join(data, "MentalModel.xml"))
    ss = tree.getroot().find("SymbolicSpace")
    cb = ss.find("codebook") or ET.SubElement(ss, "codebook")
    cb.text = "none"
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
    tree.write(tmp, xml_declaration=True); tmp.close()
    try:
        raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                init_config(path=tmp.name, defaults_path=os.path.join(data, "model.xml"))
                Language.TheGrammar._configured = False
                Models.BasicModel.from_config(tmp.name)
            except Exception:
                raised = True
        assert raised, "SymbolicSpace codebook=none must be rejected"
    finally:
        os.unlink(tmp.name)
```
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** -- in the codebook-mode resolution for PS/SS, consult `architecture.MANDATORY_CODEBOOK_TIERS`: if the section is mandatory and the resolved mode is `"none"`, raise a clear `ValueError(f"{section}: codebook is mandatory; codebook=none is not allowed")`. (If the resolved mode would otherwise default to `none` for these tiers, force `quantize`.)
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit (Alec)** -- `feat: make PerceptualSpace/SymbolicSpace codebooks mandatory`

---

## Phase 5 -- Test migration, regression gate, compatibility

### Task 5.1: Migrate shape-dependent tests

**Files:** Modify `test/test_basicmodel.py` (`TestSubSpaceDerivedSizes` `~:603`, `TestWhenEncodingRoundTrip` `~:557`), `test/test_use_flags.py`, `test/test_basicmodel.py::TestReconstructionSymbols`.

- [ ] **Step 1: Update** `TestSubSpaceDerivedSizes` expected muxed widths for the new `WhenEncoding`/`WhereEncoding` widths where it constructs spaces with explicit `nWhere`/`nWhen`; update `TestWhenEncodingRoundTrip` only if its constructions changed. Update `test_use_flags.py`'s instantiation expectations if SS-shape/codebook changes flip any `useGrammar`/instantiability. Update the XOR reconstruction expectations now that SS carries a codebook.
- [ ] **Step 2: Run** each touched test file `-v` $\to$ PASS.
- [ ] **Step 3: Commit (Alec)** -- `test: migrate shape-dependent tests to the converged architecture`

### Task 5.2: Regression gate + compatibility

- [ ] **Step 1:** `.venv/bin/python -m pytest test/test_basicmodel.py -q` -- record failures; must be `<=` the isolation baseline (`0` extra new failures attributable to this change; the suite is order-fragile).
- [ ] **Step 2:** `.venv/bin/python -m pytest test/test_role_collapsed_grammar.py test/test_d1_pos_recovery_gate.py -v` -- must stay green (no POS leakage).
- [ ] **Step 3:** `.venv/bin/python -m pytest test/test_when_bracket.py test/test_tense_aspect.py test/test_grammar_fixtures.py test/test_canonical_shape.py test/test_convergence_configs.py test/test_mandatory_codebook.py -q` -- the convergence surface, all green.
- [ ] **Step 4: Autoload safety** -- confirm `<autoload>` defaults safe (false) in the live XMLs so a stale checkpoint (pre-convergence shape) is never silently loaded against the new shape; `grep -rn "autoload" data/*.xml` and verify, since there is no shim.
- [ ] **Step 5: Commit (Alec)** -- `chore: regression gate + autoload check for the architecture convergence`

---

## Self-Review Against Spec

| Spec item | Covered by |
|---|---|
| `.where`/`.when` as architectural constants | Tasks 3.1, 3.2 (canonical_shape + wiring), 3.3 (strip tags) |
| `.when` on all five tiers; `.where` IS/PS/SS only | `canonical_shape` table; Tasks 2.2, 3.2, 3.3 |
| SS promoted to 2/2 + mandatory codebook | Tasks 2.1, 3.2/3.3 (shape), 4.1 (codebook) |
| Mandatory PS/SS codebooks (forbid none) | Task 4.1 |
| Unit-bracket `.when` (incl. encode) | Tasks 1.1-1.3; fixtures 1.4-1.5 |
| Migrate all configs (size compensation), none removed | Task 3.3 |
| No shim; retrain; autoload safe | Task 5.2 Step 4 |
| Test migration | Tasks 1.4, 1.5, 5.1 |
| Regression / POS guards | Task 5.2 |

**Intentionally deferred:** Phase 7 (MorphologyLayer) -- separate design+plan on the converged substrate. **Flagged design choice:** the unit-bracket aspect semantics (Phase 1, center-anchored SIMPLE/PERFECT/PROGRESSIVE) are a locked-but-adjustable proposal; confirm before executing Phase 1.

## Risks

1. **Phase 3 has a red interior** (code flip in 3.2 precedes the last config migration in 3.3). Keep 3.2 and 3.3 close; the gate is the final green, not per-task green, for these two.
2. **Per-config size arithmetic** (Task 3.3) is the most error-prone step -- the `test_all_live_configs_build` test is the guard; resize in small batches.
3. **SS promotion** touches `_tie_lexicon_to_codebook` / SS-CS orth (`bin/Spaces.py:3389`, `:3660`); Task 2.1 should exercise an SS-shaped codebook to surface breakage early.
4. **Order-fragility** of `test_basicmodel.py`: judge Task 5.2 against an isolation baseline, not a single full-suite run.
5. **Aspect semantics** are a design choice (see flag); if changed, only Phase 1's `aspect_interval` + fixtures move.
