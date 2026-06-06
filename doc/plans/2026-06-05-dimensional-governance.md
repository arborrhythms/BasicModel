# Dimensional Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every space-to-space handoff a pure reshape and the $\Pi$/$\Sigma$ operators square + invertible (deep CS hub, wide PS/SS) — proven first on `MM_5M` (parallel) and `MM_5M_grammar` (serial), then enforced globally.

**Architecture:** CS is the deep hub `[B,N,D]`; PS/SS are wide. $\Pi$ (at PS) and $\Sigma$ (at SS) are square LDU-invertible bridges over a constant content slab; the deep idea lives in the bridge matrix, not in any single percept/symbol. A new `<sigmaPi>last|butterfly|full</sigmaPi>` knob selects the operator span; `full` is the square bridge.

**Tech Stack:** PyTorch, custom `Spaces`/`Layers`/`Models`/`Language` in `bin/`, pytest, XSD validation via `xmllint`.

**Spec:** `doc/specs/2026-06-05-dimensional-governance.md`. Read it before starting.

---

## Working agreement (read first)

- **TDD where tractable.** Schema, configs, grammar conversions, and tests get full step-level code. The deep refactors of `bin/Spaces.py` / `bin/Layers.py` / `bin/Models.py` name the exact files, the **defining test**, and the change targets; the replacement code is **finalized at execution after re-reading the named functions** — these are large, evolving files where inventing code ahead of the read is guesswork. (Same convention as `doc/plans/2026-06-05-tier-free-bounded-stm-fold.md`.)
- **Test runner:** `KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python -m pytest <nodeid> -q`. Run **targeted** node IDs. `test/test_basicmodel.py` fails ~24 on clean HEAD and the suite is order-fragile — ignore unrelated reds.
- **Compile gate:** `MODEL_COMPILE=eager` (Dynamo trace; inductor's C++ build is broken by the space in this repo path). An inductor `CppCompileError` is not a real graph break.
- **Validation gate is live.** `data/model.xsd` validation is now a fail-loud precursor to model creation (`util.XMLConfig._validate_against_schema`). Keep `MM_5M.xml` / `MM_5M_grammar.xml` valid as you edit: `xmllint --noout --schema data/model.xsd data/<f>.xml`.
- **Git is the user's.** Every "Commit" step is a **checkpoint**: stop, summarize the diff, let the user commit. Do **not** run `git add/commit/checkout/stash`.
- **Phasing is load-bearing.** Phases 0–2 must be green before Phase 3 touches any existing config. Phase 3 is where flexibility is removed; it is test-first and the riskiest.

---

## File map

| File | Role in this work |
|---|---|
| `data/model.xsd` | add `<sigmaPi>` enum on PS/CS/SS; keep `<butterfly>` as a deprecated alias |
| `bin/Spaces.py` | PS.$\Pi$ construction (~7866–7916), SS.$\Sigma$ construction (~12082–12132), `forwardEnd` (~6500), `SymbolicSpace.forward` (~14975); the `<butterfly>`→`<sigmaPi>` read + the new `full` span |
| `bin/Layers.py` | `PiLayer` / `SigmaLayer` — add the `full` (dense flattened square LDU) span alongside `last`/`butterfly` |
| `bin/Models.py` | `ModelFactory.validate_config` flat-slab check (~7444–7480) — Phase 3 enforcement; `create_from_config` |
| `bin/architecture.py` | `canonical_shape` (the where/when band) — read for the content computation |
| `data/MM_5M.xml` | rewrite: parallel, deep CS / wide PS,SS, `<sigmaPi>full</...>`, ~5M params |
| `data/MM_5M_grammar.xml` | **new**: serial, matching MM_5M dims, `role_collapsed.grammar` |
| `data/xor.grammar`, `data/default.grammar`, `data/shamatha.grammar` | convert to role-collapsed format (keep `complete.grammar` as baseline) |
| `test/test_dimensional_governance.py` | **new**: sigmaPi span, flat-slab, invertibility, MM_5M + MM_5M_grammar build/forward/reconstruct |

---

## Phase 0 — the `<sigmaPi>` knob

### Task 1: Schema — add `<sigmaPi>`, deprecate `<butterfly>`

**Files:**
- Modify: `data/model.xsd` (add a `sigmaPiEnum` simpleType; add `<sigmaPi>` to `perceptualSpaceType`, `conceptualSpaceType`, `symbolicSpaceType`; leave `<butterfly>` in place, re-commented as deprecated)

- [ ] **Step 1: Add the enum + elements.** After the existing enum block in `data/model.xsd`, add:

```xml
  <!-- Span of the Pi/Sigma operator (replaces <butterfly>): last = per-slot
       (last dim only); butterfly = O(N log N) cross-dim cascade; full = dense
       flattened square matrix (full interconnect, the square invertible
       bridge). Default last. -->
  <xs:simpleType name="sigmaPiEnum">
    <xs:restriction base="xs:string">
      <xs:enumeration value="last"/>
      <xs:enumeration value="butterfly"/>
      <xs:enumeration value="full"/>
    </xs:restriction>
  </xs:simpleType>
```

Then add `<xs:element name="sigmaPi" type="sigmaPiEnum" minOccurs="0"/>` next to the existing `butterfly` element in `perceptualSpaceType`, `conceptualSpaceType`, and `symbolicSpaceType` (re-comment the existing `<butterfly>` element as "deprecated alias for `<sigmaPi>`: true→butterfly, false/absent→last").

- [ ] **Step 2: Validate the schema is well-formed.**

Run: `xmllint --noout data/model.xsd && echo OK`
Expected: `OK` (no double-hyphen-in-comment errors — avoid `--` inside XML comments).

- [ ] **Step 3: Confirm existing configs still validate.**

Run: `for f in data/*.xml; do xmllint --noout --schema data/model.xsd "$f" || echo "FAIL $f"; done; echo done`
Expected: no `FAIL` lines.

- [ ] **Step 4: Commit** — checkpoint ("schema: add <sigmaPi> span enum; deprecate <butterfly>").

### Task 2: Construction — read `sigmaPi`, normalize the legacy boolean

**Files:**
- Create: `test/test_dimensional_governance.py`
- Modify: `bin/Spaces.py` (the `<butterfly>` reads at ~7881 (PS) and ~12114 (SS): add a shared `sigma_pi_mode(section)` resolution)

- [ ] **Step 1: Write the failing test** in `test/test_dimensional_governance.py`:

```python
"""Dimensional-governance gates (doc/specs/2026-06-05-dimensional-governance.md)."""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Spaces

def test_sigma_pi_mode_resolves_and_aliases_butterfly():
    f = Spaces.Space.sigma_pi_mode  # staticmethod (raw value, str|bool) -> mode
    assert f("last") == "last"
    assert f("butterfly") == "butterfly"
    assert f("full") == "full"
    assert f(True) == "butterfly"      # legacy <butterfly>true</butterfly>
    assert f(False) == "last"          # legacy <butterfly>false</butterfly>
    assert f(None) == "last"           # absent
```

- [ ] **Step 2: Run it — Expected: FAIL** (`Space` has no `sigma_pi_mode`).

Run: `KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python -m pytest test/test_dimensional_governance.py::test_sigma_pi_mode_resolves_and_aliases_butterfly -q`

- [ ] **Step 3: Implement** `Space.sigma_pi_mode` (a staticmethod on `Space` in `bin/Spaces.py`, near `normalize_codebook_mode`):

```python
@staticmethod
def sigma_pi_mode(raw):
    """Resolve the <sigmaPi> span. Accepts the new enum (last|butterfly|
    full) and the legacy <butterfly> boolean (true->butterfly,
    false/None->last)."""
    if isinstance(raw, bool):
        return "butterfly" if raw else "last"
    if raw is None:
        return "last"
    v = str(raw).strip().lower()
    if v in ("last", "butterfly", "full"):
        return v
    if v in ("true", "1"):
        return "butterfly"
    if v in ("false", "0", ""):
        return "last"
    raise ValueError(f"<sigmaPi> must be last|butterfly|full (legacy "
                     f"true/false accepted); got {raw!r}")
```

- [ ] **Step 4: Run it — Expected: PASS.**

- [ ] **Step 5: Wire it into the PS.$\Pi$ and SS.$\Sigma$ construction.** At `bin/Spaces.py:~7881` (PS) and `~12114` (SS), resolve the mode via `Space.sigma_pi_mode(TheXMLConfig.space(section, "sigmaPi", default=None) or TheXMLConfig.space(section, "butterfly", default=None))` and branch: `last` → today's per-slot construction; `butterfly` → today's `butterfly=True, N=...`; `full` → Task 3's dense span. (Read both construction blocks before editing.)

- [ ] **Step 6: Confirm the existing butterfly configs still build** (XOR_exact uses `<butterfly>true</butterfly>` on PS/CS/SS):

Run: `KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python -m pytest test/test_cs_reentrancy.py test/test_pi_sigma_ownership.py -q`
Expected: PASS (butterfly alias resolves to the same construction).

- [ ] **Step 7: Commit** — checkpoint ("feat: sigma_pi_mode resolution + wiring; butterfly is a legacy alias").

---

## Phase 1 — MM_5M (parallel) + MM_5M_grammar (serial): minimal working

### Task 3: The `full` span — dense flattened square $\Pi$/$\Sigma$ (the wide$\leftrightarrow$deep bridge)

This is the core new code: the square LDU matrix over the flattened content slab `[B, N*D]` that re-dimensions wide PS/SS $\leftrightarrow$ deep CS by pure reshape + a dense invertible map (replacing the retired `sigma_percept` lift).

**Files:**
- Modify: `bin/Layers.py` — `SigmaLayer` / `PiLayer` (read both `__init__` + `forward`/`reverse` first; `SigmaLayer` ~2204)
- Modify: `bin/Spaces.py` — the `full` branch from Task 2 Step 5
- Test: `test/test_dimensional_governance.py`

- [ ] **Step 1 (read):** Read `SigmaLayer.__init__`/`forward`/`reverse` and the existing `butterfly=True, N=...` path. Confirm how the flattened element count is derived (`inputShape[0] * nOutputDim` today) and how reverse inverts (LDU). Note whether a non-per-slot dense `dim` is already supported (a `SigmaLayer(N*D, N*D)` with no butterfly is a dense square LDU over the flattened slab).

- [ ] **Step 2 (test): invertibility gate.** Add to `test/test_dimensional_governance.py`:

```python
import torch
def test_full_sigma_pi_is_invertible_over_flat_slab():
    from Layers import SigmaLayer
    B, N, D = 2, 8, 6
    op = SigmaLayer(N * D, N * D, invertible=True, nonlinear=True,
                    stable=True)  # full: dense square over the flattened slab
    x = torch.randn(B, N, D).clamp(-0.5, 0.5)
    flat = x.reshape(B, N * D)
    y = op.forward(flat)
    x_rec = op.reverse(y).reshape(B, N, D)
    assert (x - x_rec).abs().max().item() < 1e-3
```

- [ ] **Step 3: Run it.** Expected: PASS if `SigmaLayer(N*D, N*D)` already gives a dense invertible map; FAIL if the constructor assumes per-slot. Fix the constructor/forward minimally so a dense flattened square is invertible (finalize after Step 1's read).

- [ ] **Step 4: Implement the `full` branch** in `bin/Spaces.py` (Task 2 Step 5): when mode is `full`, build the operator with `dim = inputShape[0] * content_dim` (the whole flattened slab) and reshape `[B,N,D] <-> [B, N*D]` around `forward`/`reverse`. Store `self.sigma_pi_mode = "full"`. (Finalize against the actual PS/SS construction + forward you read in Task 2.)

- [ ] **Step 5: Run the invertibility gate + the butterfly regression** (Task 2 Step 6) — Expected: PASS both.

- [ ] **Step 6: Commit** — checkpoint ("feat: <sigmaPi>full</> dense flattened square Pi/Sigma bridge").

### Task 4: Write `data/MM_5M.xml` (parallel, ~5M, per spec)

**Files:**
- Modify: `data/MM_5M.xml`

- [ ] **Step 1: Compute the dims.** Choose a flat-slab `T` and the deep/wide split so `IS.nOut*content == PS.nOut*content == CS.nOut*content` (content = nDim − nWhere − nWhen; band default 4). Target ~5M params dominated by the SS symbol codebook. Concretely (band 4):
  - CS deep: `nOutput=8` (N=STM), `nDim=1028` (content 1024) → slab 8192.
  - PS wide: `nOutput=1024`, `nDim=12` (content 8) → slab 8192; `<sigmaPi>full</sigmaPi>`.
  - SS wide: `nVectors≈1_000_000` (symbol codebook; ~5M params at content 5), `nOutput=1024`, `nDim=12` (content 8) → slab 8192; `<sigmaPi>full</sigmaPi>`.
  - IS 2-D: `nOutput=1024`, `nDim=12` (content 8) → slab 8192.
  - OS 2-D: `nOutput=1`, small.
  - Record the exact param estimate in an XML comment.

- [ ] **Step 2: Write the file** with `<conceptualMode>parallel</conceptualMode>`, the dims above, `<sigmaPi>full</sigmaPi>` on PS and SS, `<codebook>` per spec (CS quantize or none; SS quantize), and `xsi:noNamespaceSchemaLocation="model.xsd"` on the root.

- [ ] **Step 3: Validate.** Run: `xmllint --noout --schema data/model.xsd data/MM_5M.xml && echo OK`. Expected: `OK`.

- [ ] **Step 4: Commit** — checkpoint ("config: MM_5M parallel, deep-CS/wide-PS,SS, sigmaPi=full").

### Task 5: Get MM_5M building + forward + reconstruct

**Files:**
- Test: `test/test_dimensional_governance.py`
- Modify (as the forward demands): `bin/Spaces.py` (`forwardEnd` ~6500 reshape; PS/SS forward; the bridge)

- [ ] **Step 1: Write the build+forward gate:**

```python
def _build(cfg_name):
    import Models, Language
    from util import init_config
    p = os.path.join(os.path.dirname(_BIN), "data", cfg_name)
    init_config(path=p, defaults_path=os.path.join(os.path.dirname(_BIN), "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m

def test_mm_5m_builds_and_forwards():
    import torch, Models
    m = _build("MM_5M.xml"); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    inp_items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(inp_items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run it.** Expected: FAIL initially (the `full` bridge / forward reshape needs to agree with the deep$\leftrightarrow$wide flat-slab). Read `Space.forwardEnd` (~6500, the `reshape(B,-1,nOutputDim)` that previously crashed on MM_5M) and the PS/CS/SS forward; make the forward route the flat-slab through the `full` $\Pi$/$\Sigma$ as a pure reshape. Finalize minimally to pass.

- [ ] **Step 3: Run it — Expected: PASS.**

- [ ] **Step 4: Reconstruction gate** (the bridge is invertible end-to-end):

```python
def test_mm_5m_reconstructs():
    import torch, Models
    m = _build("MM_5M.xml"); m.eval(); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    inp_items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(inp_items)
    _, _, _, recon = m.forward(x)
    assert recon is not None and torch.isfinite(
        recon if torch.is_tensor(recon) else torch.tensor(0.0)).all()
```

- [ ] **Step 5: Run it — Expected: PASS.** (If reconstruction is gated off by config, assert the forward path is invertible at the $\Sigma$ step instead — finalize against the SS forward you read.)

- [ ] **Step 6: Commit** — checkpoint ("feat: MM_5M forward+reconstruct via the deep<->wide full bridge").

### Task 6: Convert grammars to role-collapsed format

**Files:**
- Modify: `data/xor.grammar`, `data/default.grammar`, `data/shamatha.grammar` (keep `data/complete.grammar` as baseline)
- Reference: `data/role_collapsed.grammar` (the target format), `doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md`

- [ ] **Step 1 (read):** Read `data/role_collapsed.grammar` end-to-end and `test/test_role_collapsed_grammar.py` to learn the exact contract: PS uses `U` (analyzer root) with `<start name="everything">U</start>`; SS uses role names `op_I[n]`/`op_O1`, `<start name="...">op_O1</start>` per start operator, `query="true|false"` on relational rules.

- [ ] **Step 2: Convert `xor.grammar`.** PS: `SURF` → `U`, add `<start name="everything">U</start>`. SS: rewrite `S = not.forward(S)` etc. into role form — `not_O1 = not.forward(not_I1)`, `intersection_O1 = intersection.forward(intersection_I1, intersection_I2)`, `union_O1 = union.forward(union_I1, union_I2)` — with mirrored `<generate>` and a `<start>` for each output role. Keep `<grammar name="xor">`.

- [ ] **Step 3: Convert `default.grammar` and `shamatha.grammar`** the same way (role-name each operator's operands/outputs; declare starts; mirror generate).

- [ ] **Step 4: Gate.** For each converted grammar, load it and assert it parses with role-collapsed structure:

```python
def test_converted_grammars_load_role_collapsed():
    import Language
    for g in ("xor.grammar", "default.grammar", "shamatha.grammar"):
        Language.TheGrammar._configured = False
        gr = Language.Grammar(); gr.load_from_grammar_file(
            os.path.join(os.path.dirname(_BIN), "data", g))
        assert len(gr.rules) > 0
        assert any("_O1" in (r.lhs or "") for r in gr.rules)  # role form
```

(Finalize the loader call against `test_role_collapsed_grammar.py`'s actual API.)

- [ ] **Step 5: Run it — Expected: PASS.** Also run `test/test_role_collapsed_grammar.py -q` — Expected: PASS.

- [ ] **Step 6: Commit** — checkpoint ("grammar: convert xor/default/shamatha to role-collapsed").

### Task 7: Write `data/MM_5M_grammar.xml` (serial, matching dims)

**Files:**
- Create: `data/MM_5M_grammar.xml`

- [ ] **Step 1: Write it** matching MM_5M's IS/PS/CS/SS/OS shapes (§7 of the spec), with `<conceptualMode>serial</conceptualMode>`, the serial symbol content (4 + where 2 + when 2 = 8 — set SS content accordingly), `<WordSpace><language><grammar>role_collapsed.grammar</grammar></language></WordSpace>`, and the schema reference on the root. Param count follows the serial grammar-op budget (small ops), so the wide SS codebook may be smaller than MM_5M's.

- [ ] **Step 2: Validate.** Run: `xmllint --noout --schema data/model.xsd data/MM_5M_grammar.xml && echo OK`. Expected: `OK`.

- [ ] **Step 3: Commit** — checkpoint ("config: MM_5M_grammar serial, matching MM_5M dims").

### Task 8: Get MM_5M_grammar building + forward (serial fold)

**Files:**
- Test: `test/test_dimensional_governance.py`
- Modify (only if the serial fold disagrees with the new dims): `bin/Spaces.py` / `bin/Models.py` serial path

- [ ] **Step 1: Build+forward gate:**

```python
def test_mm_5m_grammar_builds_and_forwards():
    import torch, Models
    m = _build("MM_5M_grammar.xml"); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(inp_items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()
    cap = int(m.conceptualSpace.stm.capacity)
    assert int(m.conceptualSpace.stm._depth.max().item()) <= cap
```

- [ ] **Step 2: Run it.** Expected: FAIL if the serial fold's dims disagree with the new shapes. The tier-free fold already exists (`_stm_bounded_reduce_step`, `_stm_reduce_to_single_S`); reconcile its widths with the serial SS content (read `_forward_body_per_word` / `_per_word_body_step`). Finalize minimally to pass.

- [ ] **Step 3: Run it — Expected: PASS.**

- [ ] **Step 4: Commit** — checkpoint ("feat: MM_5M_grammar serial forward under the new dims").

---

## Phase 2 — tests (consolidate the gates)

### Task 9: Flat-slab + sigmaPi span gates (MM_5M)

**Files:**
- Modify: `test/test_dimensional_governance.py`

- [ ] **Step 1: Add the flat-slab assertion** for MM_5M (content slab equal across IS/PS/CS):

```python
def test_mm_5m_flat_slab_is_square():
    from architecture import canonical_shape as cshape
    import util
    util.init_config(path=os.path.join(os.path.dirname(_BIN), "data", "MM_5M.xml"),
                     defaults_path=os.path.join(os.path.dirname(_BIN), "data", "model.xml"))
    sp = util.TheXMLConfig.space
    def slab(name):
        return int(sp(name, "nOutput")) * (int(sp(name, "nDim")) - sum(cshape(name)))
    assert slab("InputSpace") == slab("PerceptualSpace") == slab("ConceptualSpace")
```

- [ ] **Step 2: Add a sigmaPi-span assertion** (PS/SS report `full`):

```python
def test_mm_5m_uses_full_sigma_pi():
    m = _build("MM_5M.xml")
    assert getattr(m.perceptualSpace, "sigma_pi_mode", None) == "full"
    assert getattr(m.symbolicSpace, "sigma_pi_mode", None) == "full"
```

- [ ] **Step 3: Run all MM_5M gates — Expected: PASS.** Commit — checkpoint ("test: MM_5M flat-slab + sigmaPi gates").

### Task 10: Serial fold + grammar gates (MM_5M_grammar)

- [ ] **Step 1:** Ensure the Task 8 capacity assertion + a "fold reaches root" assertion (mirror `test/test_bounded_stm_fold.py::test_sentence_end_reduces_toward_root`) run against `MM_5M_grammar.xml`.
- [ ] **Step 2: Run — Expected: PASS.** Commit — checkpoint ("test: MM_5M_grammar serial fold gates").

- [ ] **Step 3: Phase-1/2 regression sweep** (no new reds in the touched areas):

Run: `KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python -m pytest test/test_dimensional_governance.py test/test_modality_configs.py test/test_cs_reentrancy.py test/test_pi_sigma_ownership.py test/test_role_collapsed_grammar.py test/test_bounded_stm_fold.py -q`
Expected: PASS (build-all included).

---

## Phase 3 — reduce flexibility (test-first; only after Phases 0–2 are green)

### Task 11: Enforce the flat-slab invariant globally

**Files:**
- Modify: `bin/Models.py` `ModelFactory.validate_config` (~7444–7480) and/or the $\Pi$/$\Sigma$ construction in `bin/Spaces.py`
- Test: `test/test_dimensional_governance.py`

- [ ] **Step 1 (read):** Re-read `validate_config`'s flat-slab block and every place an operator currently *absorbs* a dim mismatch (the `forwardEnd` reshape, any `sigma_percept`/lift remnants).
- [ ] **Step 2 (test):** Add a test that a config whose IS/PS/CS slabs are unequal **raises** at `from_config` (fail-loud), and that the operators no longer silently re-dimension.
- [ ] **Step 3:** Run it — Expected: FAIL (today some mismatches are absorbed).
- [ ] **Step 4 (implement):** Make the mismatch a hard error at construction; remove the absorbing reshape path. Finalize after Step 1.
- [ ] **Step 5:** Run it + the full `test/test_modality_configs.py` build-all — Expected: PASS (after Task 12 migrates configs).
- [ ] **Step 6: Commit** — checkpoint ("refactor: flat-slab invariant is enforced; operators no longer absorb dim mismatch").

### Task 12: Migrate existing configs to comply

**Files:**
- Modify: the `data/*.xml` configs that Task 11's gate now rejects (run the build-all to enumerate them)

- [ ] **Step 1:** Run `for f in data/*.xml; do KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu .venv/bin/python -c "import sys; sys.path.insert(0,'bin'); import Models; Models.BasicModel.from_config('$f')" 2>&1 | grep -q "flat-slab" && echo "MIGRATE $f"; done` to list non-compliant configs.
- [ ] **Step 2:** For each, adjust IS/PS/CS/SS dims so the content slabs are equal (pure reshape), keeping the model's intent. Validate each against the schema.
- [ ] **Step 3:** Run `test/test_modality_configs.py -q` — Expected: PASS. Commit — checkpoint ("config: migrate all data configs to the flat-slab invariant").

### Task 13: Remove legacy re-dimensioning paths

**Files:**
- Modify: `bin/Spaces.py` / `bin/Language.py` — delete the retired `sigma_percept` lift remnants and any dead re-dimensioning branches surfaced by Tasks 11–12

- [ ] **Step 1 (read):** `grep -n sigma_percept bin/*.py` and the `forwardEnd` reshape; identify code now unreachable under the enforced invariant.
- [ ] **Step 2:** Delete the dead paths; keep a one-line note where a behavior moved.
- [ ] **Step 3:** Full targeted regression: `test/test_dimensional_governance.py test/test_modality_configs.py test/test_cs_reentrancy.py test/test_role_collapsed_grammar.py test/test_bounded_stm_fold.py -q` — Expected: PASS.
- [ ] **Step 4: Commit** — checkpoint ("refactor: remove retired percept->concept lift + dead re-dimensioning paths").

---

## Self-review

- **Spec coverage:** §1 roles/shapes → Tasks 4,7; §2 flat-slab → Tasks 9,11; §3 $\Pi$/$\Sigma$ + `<sigmaPi>` → Tasks 1–3; §4 PS/SS asymmetry → Tasks 4,5 (choose-1/choose-N is config + the SS codebook path; the by-reference *transport* is spec §10 non-goal); §5 directionality → Task 5 (invertible bridge both legs); §6 parallel/serial → Tasks 5,8; §7 reference configs → Tasks 4,7; §8 grammar → Task 6; §9 phasing → Phases 0–3; §10 non-goals → not implemented (by design).
- **Placeholders:** deep refactor tasks (3,5,8,11,13) intentionally finalize code at execution after a named read — per the working agreement, with exact files + defining tests given. All schema/config/grammar/test tasks have full content.
- **Type/name consistency:** `Space.sigma_pi_mode` (Task 2) is referenced consistently; `sigma_pi_mode` attribute (Task 9) is set in Task 4 Step 2 / Task 3 Step 4; `_build`/`slab` helpers defined before use.
- **Scope:** one paradigm + two reference configs + grammar conversion (Phases 0–2), with global enforcement isolated to Phase 3 so Phases 0–2 stand alone as working software.
