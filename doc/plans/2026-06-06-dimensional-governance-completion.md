# Dimensional Governance Completion + MLX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the dimensional-governance work as a single track: replace the parallel conceptual recurrence with one square augment-threaded invertible combine (richer forward, exact reconstruction), make data thread through the forward (which unblocks `fullgraph=True`), get the serial sibling green, enforce the invariants + drop dead reconstruction paths globally, and lower the tensor-core forward to an MLX `.pte` via ExecuTorch.

**Architecture:** Per-stage conceptual update is one `InvertibleLinearLayer` over $[PS_t \Vert SS_t \Vert CS_t]$ emitting $[next\_CS \Vert aug]$; `<perfectReconstruction>` gates whether the $2D$ augment is threaded to the reverse (exact) or dropped (exact-on-rank). Spaces hold only weights; per-batch data threads through the forward, so no stored buffer oscillates `requires_grad`. A single architecture-level `<sigmaPi>` (last | butterfly | full; default butterfly) governs the span of EVERY Pi/Sigma/combine construction (PS bridge, SS bridge, IntraSentence predictor, ConceptualCombine). The fullgraph-clean forward is the prerequisite for `torch.export` $\to$ MLX lowering.

**Tech Stack:** PyTorch, custom `Spaces`/`Layers`/`Models`/`Language` in `bin/`, pytest, XSD validation via `xmllint`, ExecuTorch + MLX delegate.

**Specs (read before starting):**
- `doc/plans/2026-06-06-parallel-conceptual-recurrence.md` (the design for Phase A --- combine, data-flow, reconstruction cleanup, interSentence, PS-not-IS).
- `doc/specs/2026-06-05-dimensional-governance.md` (original spec; the flat-slab + $\Pi$/$\Sigma$ intent that Phase C enforces).

**This plan SUPERSEDES** `doc/plans/2026-06-05-dimensional-governance.md` (its remaining open work --- Task 8 + Phase 3 --- is folded into Phases B and C below).

---

## Status carried in from prior work (do NOT redo)

| Item | State |
|---|---|
| Tasks 1-7 of the 2026-06-05 plan (`<sigmaPi>` knob, `full` span, grammars, MM_5M parallel build+forward, serial config file) | DONE, green |
| Uniform $(2,2)$ band across tiers; `nDim = nWhat + nWhere + nWhen` | DONE |
| `OutputSpace = (0,0)` exception (terminal answer carries no band); OS `nDim` reverts | DONE, NaN-loss fixed |
| PS.pi butterfly sizing fix (`outputShape[0]`, not `inputShape[0]`) | DONE: 406M$\to$21M params, ~56x faster, regression green |
| MM_5M parallel `python bin/Models.py data/MM_5M.xml` | runs clean, finite loss |
| `BASICMODEL_MPS_COMPILE=1` escape hatch; `_maybe_autobind_meta` `.item()` batched | DONE (kept) |

**Superseded from the 2026-06-05 plan:** Task 4/5 MM_5M dims (the user re-authored MM_5M: deep CS, butterfly bridges, reduced IS handoff); the `full` dense bridge stays implemented but MM_5M uses `butterfly` for the param budget.

---

## Working agreement (read first)

- **TDD where tractable.** Schema, config edits, and tests get full step-level code. The deep refactors of `bin/Spaces.py` / `bin/Layers.py` / `bin/Models.py` name the exact files, the **defining test**, and the change targets; replacement code is **finalized at execution after re-reading the named functions** (same convention as the 2026-06-05 plan; these are large evolving files).
- **Test runner:** `KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python -m pytest <nodeid> -q`. Run **targeted** node IDs. `test/test_basicmodel.py` fails ~24 on clean HEAD and the suite is order-fragile.
- **Compile gates:** `MODEL_COMPILE=eager` for trace correctness; for the recompile gate use `MODEL_COMPILE=aot_eager` (CPU) or `BASICMODEL_MPS_COMPILE=1` (MPS) with `TORCH_LOGS=recompiles`. An inductor `CppCompileError` (repo-path space) is not a graph break.
- **Schema gate is live.** Keep every edited config valid: `xmllint --noout --schema data/model.xsd data/<f>.xml`.
- **Git is the user's.** Every "Checkpoint" step means: stop, summarize the diff, let the user commit. Do **NOT** run `git add/commit/checkout/stash`.
- **Docs are LaTeX.** In `doc/*.md` use `$\to$`, `$\Sigma$`, `$\approx$`, `$\le$` etc., never Unicode glyphs (`make doc` / xelatex fails on missing glyphs).
- **Phasing is load-bearing.** A must be green before B; B before C (C removes flexibility); C before D (D needs the fullgraph-clean forward). Each phase is a checkpoint.

---

## Global `<sigmaPi>` switch (settles the param budget)

A single architecture-level `<sigmaPi>` (last | butterfly | full) governs ALL Pi/Sigma/combine spans --- the PS bridge, the SS bridge, the IntraSentence predictor, and the new ConceptualCombine. There is no per-construction param knob; the global switch settles it:

- `butterfly` (default): $O(N \log N)$ cross-element cascade, $\sim 0.3$M/stage --- keeps the combine in budget.
- `full`: dense LDU, $\approx 2(3D)^2$/stage ($\approx 18.9$M at $D{=}1024$) --- exact over a small slab when wanted.
- `last`: per-slot square fold (no cross-element mix).

The existing per-space `<sigmaPi>` (PS/SS, from the 2026-06-05 Task 1) is migrated to the architecture level in Task A1, so one setting drives every construction.

---

## Phase A --- parallel conceptual recurrence (the redesign)

### Task A1: Schema --- global `<sigmaPi>`, `<perfectReconstruction>` + `<prediction>`, drop `<reconstruct>`

**Files:**
- Modify: `data/model.xsd`
- Modify: configs that set `<reconstruct>` (`grep -rl "<reconstruct>" data/*.xml`) or a per-space `<sigmaPi>` (`grep -rl "<sigmaPi>" data/*.xml`)

- [ ] **Step 1: Move `<sigmaPi>` global; add the boolean + prediction enum; remove `reconstructEnum`.** In `data/model.xsd`: delete the `reconstructEnum` simpleType (lines ~262-269) and the `<reconstruct>` element (~84); REMOVE the per-space `<sigmaPi>` element from `perceptualSpaceType`/`conceptualSpaceType`/`symbolicSpaceType`; add to the architecture type (the `sigmaPiEnum` simpleType already exists from the 2026-06-05 Task 1):

```xml
  <xs:element name="sigmaPi" type="sigmaPiEnum" minOccurs="0"/>
  <xs:element name="perfectReconstruction" type="xs:boolean" minOccurs="0"/>
  <xs:element name="prediction" type="predictionEnum" minOccurs="0"/>
```

and the enum:

```xml
  <xs:simpleType name="predictionEnum">
    <xs:restriction base="xs:string">
      <xs:enumeration value="none"/>
      <xs:enumeration value="interSentence"/>
    </xs:restriction>
  </xs:simpleType>
```

- [ ] **Step 2: Migrate configs.** Drop the `<reconstruct>concepts</reconstruct>` line from the 2 configs that set it. Move every per-space `<sigmaPi>` to ONE architecture-level `<sigmaPi>` (MM_5M has `<sigmaPi>butterfly</sigmaPi>` on PS and SS $\to$ one `<sigmaPi>butterfly</sigmaPi>` under `<architecture>`); repeat for any other config with a per-space `<sigmaPi>`.
- [ ] **Step 3: Validate schema + all configs.** Run `xmllint --noout data/model.xsd && for f in data/*.xml; do xmllint --noout --schema data/model.xsd "$f" || echo "FAIL $f"; done; echo done`. Expected: no `FAIL`.
- [ ] **Step 4: Checkpoint** ("schema: global sigmaPi; perfectReconstruction + prediction; drop reconstruct enum").

### Task A2: Parse the new knobs

**Files:**
- Create test: `test/test_conceptual_recurrence.py`
- Modify: `bin/Models.py` `create_from_config` (the architecture-field parsing)

- [ ] **Step 1: Failing test** in `test/test_conceptual_recurrence.py`:

```python
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path: sys.path.insert(0, _BIN)

def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(os.path.dirname(_BIN), "data", name)
    init_config(path=p, defaults_path=os.path.join(os.path.dirname(_BIN), "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m

def test_perfect_reconstruction_flag_parsed():
    m = _build("MM_5M.xml")
    assert hasattr(m, "perfect_reconstruction")
    assert isinstance(m.perfect_reconstruction, bool)
```

- [ ] **Step 2: Run --- Expected FAIL** (`perfect_reconstruction` absent). Run `... -m pytest test/test_conceptual_recurrence.py::test_perfect_reconstruction_flag_parsed -q`.
- [ ] **Step 3: Implement** the parse in `bin/Models.py create_from_config`: read `architecture.perfectReconstruction` (default `False`) $\to$ `self.perfect_reconstruction`; read `architecture.prediction` (default `none`) $\to$ `self.prediction_mode`; read `architecture.sigmaPi` (default `butterfly`) through `Space.sigma_pi_mode` $\to$ `self.sigma_pi_mode` (the single global span every construction reads). (Read the surrounding architecture-field block first.)
- [ ] **Step 4: Run --- Expected PASS.**
- [ ] **Step 5: Checkpoint** ("feat: parse perfectReconstruction + prediction").

### Task A3: The square augment-threaded combine layer

**Files:**
- Modify: `bin/Layers.py` (reuse `InvertibleLinearLayer`, ~974; add a thin `ConceptualCombine` wrapper that owns the per-stage layer, splits $[next\_CS \Vert aug]$, and exposes `forward(ps, ss, cs) -> (next_cs, aug)` and `reverse(next_cs, aug) -> (ps, ss, cs)`)
- Test: `test/test_conceptual_recurrence.py`

- [ ] **Step 1 (read):** Re-read `InvertibleLinearLayer.__init__` (974), `forward` (1257), `_solve_ldu` (1181), and `reverse` (1362). Confirm the `naive=False` square round-trip is exact and the rectangular path zero-pads in the $L$-basis.
- [ ] **Step 2 (test): exact square round-trip.**

```python
import torch
def test_combine_square_roundtrip_exact():
    from Layers import ConceptualCombine
    B, D = 2, 6
    c = ConceptualCombine(content_dim=D, naive=False, sigma_pi_mode="full")
    ps, ss, cs = (torch.randn(B, D).clamp(-0.5, 0.5) for _ in range(3))
    nxt, aug = c.forward(ps, ss, cs)
    ps2, ss2, cs2 = c.reverse(nxt, aug)
    assert max((ps-ps2).abs().max(), (ss-ss2).abs().max(), (cs-cs2).abs().max()) < 1e-3
```

- [ ] **Step 3: Run --- Expected FAIL** (`ConceptualCombine` absent).
- [ ] **Step 4: Implement `ConceptualCombine`** wrapping `InvertibleLinearLayer(3*content_dim, 3*content_dim, naive=False, invertible=True, stable=True)`; the `sigma_pi_mode` arg (the global `<sigmaPi>`) selects the span via `Space.sigma_pi_mode` --- `butterfly` $\to$ `butterfly=True, N=3*content_dim`; `full` $\to$ dense LDU; `last` $\to$ per-slot. `forward` concatenates $[ps \Vert ss \Vert cs]$, applies the layer, slices `next_cs = out[..., :D]`, `aug = out[..., D:]`. `reverse(next_cs, aug)` concatenates and inverts (perfect); a `reverse_dropped(next_cs)` variant concatenates `aug=0` (approximate). (Finalize against the read.)
- [ ] **Step 5: Run --- Expected PASS.** Add `test_combine_dropped_aug_exact_on_rank` (reverse with zero aug recovers the surviving subspace; assert finite + bounded error).
- [ ] **Step 6: Checkpoint** ("feat: ConceptualCombine square augment-threaded invertible combine").

### Task A4: Wire the combine into the parallel body; remove the alternation

**Files:**
- Modify: `bin/Models.py` `_forward_body` (5079, parallel branch), `_symbolic_sigma_step` (6351), `_reverse_body` (6230-area)
- Modify: `bin/Spaces.py` the per-stage combine construction
- Test: `test/test_conceptual_recurrence.py`

- [ ] **Step 1 (read):** Re-read `_forward_body` (5079) parallel loop, `_symbolic_sigma_step` (6351), `_reverse_body`. Identify where $CS_{t+1}$ is produced and where the reverse walks $CS_T \to CS_0$.
- [ ] **Step 2 (test): MM_5M still forwards + finite loss** (regression guard for the rewire):

```python
def test_mm5m_forward_finite_after_combine():
    import torch
    m = _build("MM_5M.xml")
    import Models; Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()
```

- [ ] **Step 3: Implement.** Replace the alternation ($\pi$ at $t{=}0$, $\sigma$ after) with: stage 0 reads PS once; every stage calls `ConceptualCombine.forward(PS_t, SS_t, CS_t) -> (CS_{t+1}, aug_t)`; collect `augs` as a **local list** (threaded, NOT stored). `_reverse_body` consumes `augs[t]` when `perfect_reconstruction` else `reverse_dropped`. Construct one `ConceptualCombine` per stage (weights on the space), each built with the global `self.sigma_pi_mode`. While here, route the existing PS.pi and SS.sigma construction (and the IntraSentence predictor) through `self.sigma_pi_mode` too, so the per-space `<sigmaPi>` read is fully retired.
- [ ] **Step 4: Run Step-2 test + `python bin/Models.py data/MM_5M.xml` --- Expected: finite.**
- [ ] **Step 5 (test): perfect round-trip end-to-end** --- with `<perfectReconstruction>true`, the per-stage `augs` make `_reverse_body` reproduce $CS_0$ to tolerance (assert reconstruction error small). Add `test_mm5m_perfect_reconstruction`.
- [ ] **Step 6: Checkpoint** ("feat: parallel body uses ConceptualCombine; alternation removed; augment threaded").

### Task A5: Data threads through --- remove the stored STM buffer (fullgraph fix)

**Files:**
- Modify: `bin/Layers.py` `ShortTermMemory` (`_fallback_buffer` 9380; the proxy `_buffer` 9414) --- remove the stored fallback; route the slab through a threaded tensor
- Modify: `bin/Spaces.py` `_stm_predict_then_perceive_parallel` (11001), `_stm_set_all_slots` (10676) --- read/write threaded tensors, not a stored buffer
- Modify: `bin/Models.py` `_forward_per_stage` (6661) signature: add `prev_cs=None`, RETURN the terminal CS; `runBatch` holds the returned CS and passes `prev_cs = returned_cs.detach()` next call
- Test: `test/test_conceptual_recurrence.py`

- [ ] **Step 1 (read):** Re-read the four sites; confirm every per-batch write currently lands on a `ShortTermMemory` attribute (`_fallback_buffer` / `_fallback_depth`) or `_stm_predicted_slab`.
- [ ] **Step 2 (test): no per-batch recompile under compile.**

```python
def test_no_recompile_fullgraph():
    import subprocess, os
    env = {**os.environ, "BASICMODEL_DEVICE": "cpu", "MODEL_COMPILE": "aot_eager",
           "KMP_DUPLICATE_LIB_OK": "TRUE", "TORCH_LOGS": "recompiles"}
    # 3-batch run; assert no _fallback_buffer requires_grad guard failure
    cfg = "/tmp/MM_5M_3ep.xml"  # numEpochs=3 copy written by the test setup
    out = subprocess.run([os.path.join(os.path.dirname(_BIN), ".venv/bin/python"),
        os.path.join(_BIN, "Models.py"), cfg], env=env, capture_output=True, text=True)
    assert "_fallback_buffer" not in (out.stdout + out.stderr)
    assert "requires_grad mismatch" not in (out.stdout + out.stderr)
```

- [ ] **Step 3: Run --- Expected FAIL** (today the guard flips). 
- [ ] **Step 4: Implement.** The slab and `prev_cs` thread through the forward as tensors; `L_intra`'s previous is `prev_cs` (the prior forward's returned CS, detached by the caller). Delete `ShortTermMemory._fallback_buffer`; the serial path threads its accumulation too (preserve serial semantics --- read `_forward_body_per_word`). Finalize against the read.
- [ ] **Step 5: Run Step-2 test + the recompile gate --- Expected PASS** (compiles once).
- [ ] **Step 6: Checkpoint** ("feat: STM/slab data threads through forward; no stored buffer; fullgraph compiles once").

### Task A6: interSentence seed + PS-not-IS pin

**Files:**
- Modify: `bin/Models.py` `_forward_per_stage` (the $CS_{-1}$ seed); reuse `InterSentenceLayer.predict()` / `discourse.predict()`
- Test: `test/test_conceptual_recurrence.py`

- [ ] **Step 1 (test):** `test_intersentence_seed_used` --- with `<prediction>interSentence`, the stage-0 seed equals `InterSentenceLayer.predict(...)` (assert the seed is non-empty and sourced from predict); with `none`, the seed is empty.
- [ ] **Step 2: Run --- Expected FAIL.**
- [ ] **Step 3: Implement** the seed dispatch (unify with the existing `discourse.predict()` priming so there is one seed source).
- [ ] **Step 4 (test): PS-not-IS** --- `test_parallel_ps_called_once`: monkeypatch-count `perceptualSpace.forward` over one MM_5M forward; assert exactly 1 call and that its input is the reduced percepts (shape $\le$ nOutput), never the raw IS buffer.
- [ ] **Step 5: Run both --- Expected PASS.**
- [ ] **Step 6: Phase-A regression** --- `test/test_conceptual_recurrence.py test/test_modality_configs.py test/test_invertibility.py test/test_pi_sigma_ownership.py test/test_cs_reentrancy.py test/test_dimensional_governance.py -q`. Expected: PASS. Checkpoint ("feat: interSentence seed + PS-not-IS pinned").

---

## Phase B --- serial sibling (folds in 2026-06-05 Task 8)

### Task B1: MM_5M_grammar builds + forwards under the new dims + threaded STM

**Files:**
- Test: `test/test_conceptual_recurrence.py`
- Modify (only if the serial fold disagrees): `bin/Spaces.py` / `bin/Models.py` `_forward_body_per_word`, `_per_word_body_step`

- [ ] **Step 1: Build+forward gate:**

```python
def test_mm5m_grammar_builds_and_forwards():
    import torch, Models
    m = _build("MM_5M_grammar.xml"); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()
    cap = int(m.conceptualSpace.stm.capacity)
    assert int(m.conceptualSpace.stm.size(0)) <= cap
```

- [ ] **Step 2: Run --- Expected FAIL** if the serial fold's widths disagree with the new dims OR the threaded-STM change (Task A5) broke serial accumulation. Reconcile `_stm_bounded_reduce_step` / `_stm_reduce_to_single_S` widths and the threaded accumulation. Finalize against the read.
- [ ] **Step 3: Run --- Expected PASS.**
- [ ] **Step 4: Checkpoint** ("feat: MM_5M_grammar serial forward under new dims + threaded STM").

---

## Phase C --- enforce invariants + drop dead paths (folds in 2026-06-05 Tasks 11-13)

### Task C1: Enforce the handoff invariant; remove absorbing reshapes

**Files:**
- Modify: `bin/Models.py` `ModelFactory.validate_config` (the slab/handoff block, ~7440-7490)
- Test: `test/test_conceptual_recurrence.py`

- [ ] **Step 1 (read):** Re-read `validate_config`'s handoff block and every place an operator still *absorbs* a dim mismatch (the `forwardEnd` reshape, any `sigma_percept` lift remnants).
- [ ] **Step 2 (test):** a config whose IS$\to$PS$\to$CS$\to$SS handoffs are inconsistent **raises** at `from_config` (fail-loud); `space[i].nOutputDim == space[i+1].nInputDim` is checked (the handoff validator the user asked for).
- [ ] **Step 3: Run --- Expected FAIL** (some mismatches absorbed today).
- [ ] **Step 4: Implement** the hard error; remove the absorbing reshape path. Finalize after Step 1.
- [ ] **Step 5: Run + `test/test_modality_configs.py` build-all --- Expected PASS** (after C2 migrates configs).
- [ ] **Step 6: Checkpoint** ("refactor: handoff invariant enforced; operators no longer absorb mismatch").

### Task C2: Migrate non-compliant configs

**Files:**
- Modify: the `data/*.xml` configs C1's gate rejects

- [ ] **Step 1:** Run the build-all to enumerate failures: `for f in data/*.xml; do KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu .venv/bin/python -c "import sys; sys.path.insert(0,'bin'); import Models; Models.BasicModel.from_config('$f')" 2>&1 | grep -qiE "handoff|flat-slab|nOutputDim" && echo "MIGRATE $f"; done`.
- [ ] **Step 2:** For each, align the dims (pure reshape, preserve intent); validate against the schema.
- [ ] **Step 3:** Run `test/test_modality_configs.py -q` --- Expected PASS. Checkpoint ("config: migrate configs to the enforced handoff invariant").

### Task C3: Remove legacy reconstruction + re-dimensioning paths

**Files:**
- Modify: `bin/Models.py` / `bin/Spaces.py` / `bin/Language.py`

- [ ] **Step 1 (read):** `grep -nE "reconstruct == |sigma_percept|recon.symbol|reserved for reconstruction" bin/*.py` and the `forwardEnd` reshape; identify code unreachable now that reconstruction is concepts-only and the invariant is enforced.
- [ ] **Step 2:** Delete the `none|symbols|both` reconstruct branches, recon-symbols machinery, and dead re-dimensioning; keep a one-line note where behavior moved.
- [ ] **Step 3:** Targeted regression --- `test/test_conceptual_recurrence.py test/test_modality_configs.py test/test_cs_reentrancy.py test/test_role_collapsed_grammar.py test/test_bounded_stm_fold.py test/test_dimensional_governance.py -q`. Expected PASS.
- [ ] **Step 4: Checkpoint** ("refactor: remove dead reconstruction + re-dimensioning paths").

---

## Phase D --- MLX export via ExecuTorch

Prerequisite delivered by Phase A: the tensor-core forward is `fullgraph=True`-clean (no stored-data guard, no per-batch recompile). The remaining gap is that host-side Python (the `analyse` chunker / word-segmentation and `_maybe_autobind_meta` taxonomy growth) is excluded from the compiled region; `torch.export` needs a single traceable graph, so those stay as host pre/post-steps around an exported **tensor core**.

### Task D1: Carve the exportable tensor core

**Files:**
- Modify: `bin/Models.py` (a `forward_core(staged_input) -> output_tensors` entry that runs IS-embed $\to$ PS $\to$ CS $\to$ SS $\to$ OS on already-staged tensors, with chunking/autobind done by the host before/after)
- Test: `test/test_mlx_export.py` (new)

- [ ] **Step 1 (read):** Re-read `_forward_per_stage` (6661) and the staged/compiled path (`_staged_in_sub`, `_compiled_step`). Identify the exact host pre-step (lex+chunk+embed) and post-step (autobind/taxonomy) boundaries already carved for the compiled path.
- [ ] **Step 2 (test): `torch.export` traces the core.**

```python
def test_forward_core_exports():
    import torch
    m = _build("MM_5M.xml"); m.eval()
    import Models; Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    staged = m.stage_for_core(x)           # host pre-step (chunk+embed)
    ep = torch.export.export(m.forward_core, (staged,))
    assert ep is not None
```

- [ ] **Step 3: Run --- Expected FAIL** (no `forward_core`/`stage_for_core`).
- [ ] **Step 4: Implement** `stage_for_core` (host chunk+embed $\to$ the staged tensor the compiled path already produces) and `forward_core` (the captured tensor-only forward). Reuse the existing compiled-step boundary. Finalize after Step 1.
- [ ] **Step 5: Run --- Expected PASS** (`torch.export` returns an `ExportedProgram`).
- [ ] **Step 6: Checkpoint** ("feat: exportable tensor-core forward (host chunk/autobind lifted out)").

### Task D2: Lower to an MLX `.pte`

**Files:**
- Create: `bin/export_mlx.py` (a script: export $\to$ `to_edge_transform_and_lower(..., partitioner=[MLXPartitioner()])` $\to$ write `.pte`)
- Test: `test/test_mlx_export.py`

- [ ] **Step 1 (read):** Read the installed ExecuTorch + MLX-delegate API (`executorch.exir.to_edge_transform_and_lower`, the MLX `Partitioner` entry point). Finalize the exact import paths at execution against the installed versions.
- [ ] **Step 2 (test): lowering produces a non-empty `.pte`.**

```python
def test_mlx_lower_writes_pte(tmp_path):
    import os, subprocess
    pte = tmp_path / "mm5m.pte"
    r = subprocess.run([os.path.join(os.path.dirname(_BIN), ".venv/bin/python"),
        os.path.join(_BIN, "export_mlx.py"), "data/MM_5M.xml", str(pte)],
        capture_output=True, text=True)
    assert pte.exists() and pte.stat().st_size > 0, r.stderr
```

- [ ] **Step 3: Run --- Expected FAIL** (`export_mlx.py` absent).
- [ ] **Step 4: Implement** `bin/export_mlx.py`: build model, `stage_for_core`, `torch.export`, `to_edge_transform_and_lower` with the MLX partitioner, serialize `.pte`. Skip-with-clear-message if the MLX delegate package is not installed (so the test xfails cleanly off-Mac).
- [ ] **Step 5: Run --- Expected PASS** (or xfail-skip when MLX absent).
- [ ] **Step 6: Checkpoint** ("feat: export_mlx.py lowers the tensor core to an MLX .pte").

### Task D3: Runtime parity

**Files:**
- Test: `test/test_mlx_export.py`

- [ ] **Step 1 (test): `.pte` output matches eager** on one batch within tolerance --- run the `.pte` through the ExecuTorch runtime, compare to `m.forward_core(staged)`; `max abs diff < 1e-2`.
- [ ] **Step 2: Run --- Expected FAIL/skip** until the runtime path is wired.
- [ ] **Step 3: Implement** the runtime load+run (ExecuTorch `Module`), wrapped with the host chunk pre-step + autobind post-step so the end-to-end answer matches.
- [ ] **Step 4: Run --- Expected PASS** (or skip off-Mac). Checkpoint ("feat: MLX .pte runtime parity with eager").

---

## Self-review

- **Spec coverage:** `2026-06-06-parallel-conceptual-recurrence.md` sec.3 combine $\to$ A3/A4; sec.2/4 data-flow $\to$ A5; sec.5 interSentence $\to$ A6; sec.6 PS-not-IS $\to$ A6; sec.7 reconstruction cleanup $\to$ A1/C3; `<perfectReconstruction>` $\to$ A1/A2/A4. 2026-06-05 Task 8 $\to$ B1; Tasks 11-13 $\to$ C1/C2/C3. MLX (torch.export $\to$ MLXPartitioner $\to$ .pte $\to$ runtime) $\to$ D1/D2/D3.
- **Placeholders:** schema/config/test steps carry full code; deep `bin/` refactors (A3-A5, B1, C1, C3, D1) name files + defining test + change targets and finalize at execution per the working agreement; MLX API import paths (D2) finalize against the installed package.
- **Type consistency:** `ConceptualCombine` (A3) is constructed in A4; `perfect_reconstruction`/`prediction_mode` (A2) drive A4/A6; `forward_core`/`stage_for_core` (D1) are reused in D2/D3.
- **Scope:** four sequential phases, each a working-software checkpoint; the param-budget ruling gates A4; D is isolated behind the Phase-A fullgraph prerequisite and skips cleanly off-Mac.
