# Phase 1: Compiled per-batch step — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans
> (inline; this work owns a live metalbaby verify loop and is
> recon-driven, so it is not subagent-parallelizable). Steps use
> `- [ ]`. Design: `doc/plans/2026-05-16-compiled-step-boundary-design.md`.

**Goal:** Make the per-batch `forward+loss+backward` an actually-invoked
`torch.compile`d callable and drive the **non-grammar path** to
`fullgraph=True` (0 graph breaks) per shape-bucket; grammar-path breaks
deferrable with recorded reason.

**Architecture:** `runEpoch`/`runBatch` stay eager (streaming/staging);
`runBatch` invokes a compiled callable for the per-batch compute.
Break-elimination is recon-driven: enumerate breaks (`fullgraph=True`/
`torch._dynamo.explain`), tag grammar vs non-grammar, eliminate
non-grammar by *relocate-to-producer* / *rewrite-static* / *unroll*,
verify each.

**Tech Stack:** PyTorch 2.11 (`torch.compile`, dynamo, inductor),
pytest, metalbaby (CUDA, bounded runs only).

---

## File structure

- `bin/Models.py` — `ModelFactory.run` (stop the no-op `compile(m)`);
  `BasicModel` gains a compiled-step handle invoked from `runBatch`.
- `bin/util.py` — reuse `compile()` backend/mode selection for a
  callable (not just an nn.Module).
- `test/test_compiled_step_invoked.py` — NEW: asserts dynamo actually
  traces the per-batch compute (the O1 regression net).
- `test/brick_recon.py` — NEW: recon harness; emits the tagged
  graph-break backlog (`doc/plans/recon-breaks-MM_xor.md`).
- Per-break edits: `bin/Spaces.py` / `bin/Layers.py` / `bin/Models.py`
  (relocate-to-producer or rewrite-static), one break-class per task.

---

### Task 1: Make a compiled per-batch callable that is actually invoked

**Files:**
- Modify: `bin/Models.py` `ModelFactory.run` (`m = compile(m)` site)
- Modify: `bin/Models.py` `BasicModel.runBatch` (forward call site ~2640)
- Modify: `bin/Models.py` `BasicModel` (init a compiled-step handle)
- Test: `test/test_compiled_step_invoked.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_compiled_step_invoked.py
"""O1 regression: the per-batch compute is actually torch.compiled
(dynamo traces >=1 frame). Device-independent."""
import os, sys
from pathlib import Path
os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "eager"   # always-succeeds backend
_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin")); sys.path.insert(0, str(_p / "bin"))
import torch, torch._dynamo
from data import TheData
from Models import BaseModel
from util import init_config


def test_compiled_step_is_invoked():
    cfg = str(_p / "data" / "MM_xor.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("xor")
    m, _ = BaseModel.from_config(cfg, data=TheData)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = m.to(dev)
    opt = m.getOptimizer(lr=1e-4)
    m.enable_compiled_step()           # NEW api (Step 3)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    m.runEpoch(optimizer=opt, batchSize=2, split="train", max_batches=2)
    frames = dict(torch._dynamo.utils.counters.get("frames", {}))
    assert frames.get("ok", 0) + frames.get("total", 0) > 0, (
        f"compiled step never traced; counters={frames}")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd basicmodel && BASICMODEL_DEVICE=cpu .venv/bin/python -m pytest test/test_compiled_step_invoked.py -q`
Expected: FAIL — `AttributeError: 'BasicModel' object has no attribute 'enable_compiled_step'`.

- [ ] **Step 3: Implement — compiled-step handle invoked from runBatch**

In `bin/Models.py` `BasicModel.__init__` add:

```python
        self._compiled_step = None      # set by enable_compiled_step()
```

Add method on `BasicModel`:

```python
    def enable_compiled_step(self):
        """Compile the per-batch forward and route runBatch through it.

        O1: `ModelFactory.run` used to `compile(m)` the whole module and
        then call `m.run()`, which delegates to the eager `_orig_mod`,
        so the compiled callable was never invoked. Instead we compile
        the forward *callable* and invoke that from runBatch; the eager
        run/runEpoch orchestration stays Python (streaming/staging).
        Graph breaks are expected here -- this only makes compile
        actually run; the non-grammar path is driven break-free later.
        """
        from util import compile as _compile
        self._compiled_step = _compile(self.forward, verbose=True)
```

In `runBatch`, change the forward call (~Models.py:2640) from
`self.forward(inputTensor)` to:

```python
            _fwd = self._compiled_step if self._compiled_step is not None else self.forward
            forwardInput, symbols, predictions, _ = _fwd(inputTensor)
```

In `ModelFactory.run` replace `m = compile(m)` with:

```python
        # O1: compiling the *module* and then calling m.run() is a no-op
        # (m.run delegates to the eager _orig_mod). Compile the per-batch
        # forward callable and invoke it from runBatch instead.
        m.enable_compiled_step()
```

Ensure `util.compile` accepts a callable: `torch.compile` already does;
no util change needed (verify in Step 4).

- [ ] **Step 4: Run test, verify it passes**

Run: `cd basicmodel && BASICMODEL_DEVICE=cpu .venv/bin/python -m pytest test/test_compiled_step_invoked.py -q`
Expected: PASS (frames ok/total > 0).

- [ ] **Step 5: CPU regression (representative order)**

Run:
```
cd basicmodel && PYTORCH_ENABLE_MPS_FALLBACK=1 BASICMODEL_DEVICE=cpu .venv/bin/python -m pytest -p no:randomly \
 test/test_mm_xor.py test/test_mm_boolean.py test/test_universality.py \
 test/test_invertibility.py test/test_xor_spaces.py \
 "test/test_basicmodel.py::TestNormalizeFlag" test/test_lexicon_ownership.py \
 test/test_conceptual_bivector.py test/test_perceptual_bivector.py \
 test/test_phase2_pipeline_primitives.py test/test_compiled_step_invoked.py -q
```
Expected: `175 passed` (+1 new) `, 0 regressions` (compile-with-breaks
is still numerically correct).

- [ ] **Step 6: Update `_brick_preflight` to profile the compiled step**

In `bin/Models.py` `_brick_preflight`, before the profiled `runEpoch`,
ensure `m.enable_compiled_step()` has run (it has, via
`ModelFactory.run`); add a one-line `TheMessage` noting the pre-flight
now measures the compiled step (prior eager DtoH numbers are moot).
No assert change yet (recon first).

(No git commit — this session does not commit unless asked.)

---

### Task 2: Recon harness — enumerate & tag the graph-break backlog

**Files:**
- Create: `test/brick_recon.py`
- Create (output): `doc/plans/recon-breaks-MM_xor.md`

- [ ] **Step 1: Write the recon harness**

```python
# test/brick_recon.py
"""Phase-1 recon: enumerate torch.compile graph breaks in the per-batch
forward and tag each grammar / non-grammar. Not a pytest test."""
import os, sys, re
from pathlib import Path
os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "eager"
_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin")); sys.path.insert(0, str(_p / "bin"))
import torch, torch._dynamo
from data import TheData
from Models import BaseModel
from util import init_config

# Source frames that constitute the GRAMMAR path (deferrable). Anything
# else is NON-GRAMMAR (must-fix). Tag by the explanation's user stack.
_GRAMMAR = ("SyntacticLayer", "_compose", "chart", "grammar", "_apply_rule",
            "derivation", "GrammarLayer", "partForward")


def _tag(stack_text):
    return "grammar" if any(g in stack_text for g in _GRAMMAR) else "non-grammar"


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "MM_xor.xml"
    cfg = str(_p / "data" / cfg_name)
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("xor" if cfg_name != "MM_5M.xml" else "text")
    m, _ = BaseModel.from_config(cfg, data=TheData)
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    m = m.to(dev)
    opt = m.getOptimizer(lr=1e-4)
    # Build one real batch via a bounded warm-up so shapes are realistic.
    m.runEpoch(optimizer=opt, batchSize=2, split="train", max_batches=1)
    expl = torch._dynamo.explain(m.forward)
    # Capture the explanation by running it on one staged input:
    # re-drive a single bounded tick capturing the inputTensor.
    captured = {}
    _orig = m.runBatch
    def _grab(*a, **k):
        bo = k.get("batch_override")
        if bo is not None and "inp" not in captured:
            captured["inp"] = bo[0]
        return _orig(*a, **k)
    m.runBatch = _grab
    m.runEpoch(optimizer=opt, batchSize=2, split="train", max_batches=1)
    m.runBatch = _orig
    explanation = torch._dynamo.explain(m.forward)(captured["inp"])
    out = ["# Phase-1 recon — graph-break backlog (%s)\n" % cfg_name,
           "total breaks: %d\n" % explanation.graph_break_count]
    for i, br in enumerate(explanation.break_reasons):
        stack = "".join(getattr(br, "user_stack", []) and
                         [str(s) for s in br.user_stack] or [str(br)])
        out.append("- [%s] #%d %s\n  %s" % (
            _tag(stack), i, getattr(br, "reason", ""), stack.replace("\n", " ")[:300]))
    (_p / "doc" / "plans" / ("recon-breaks-%s.md" % cfg_name.replace(".xml", ""))
     ).write_text("\n".join(out))
    print("\n".join(out[:2]))
    ng = sum(1 for o in out if o.startswith("- [non-grammar]"))
    print(f"[recon] non-grammar breaks (must-fix): {ng}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run recon locally (MPS/CPU — device-independent for break enumeration)**

Run: `cd basicmodel && .venv/bin/python test/brick_recon.py MM_xor.xml`
Expected: writes `doc/plans/recon-breaks-MM_xor.md`; prints total +
non-grammar count. (If the `explain` API shape differs in torch 2.11,
adapt to `torch._dynamo.explain(fn)(*args)` returning an object with
`.graph_break_count` / `.break_reasons`; fall back to
`fullgraph=True` try/except capturing the first break if needed.)

- [ ] **Step 3: Confirm on metalbaby (tiny bounded, authoritative)**

Run: `make sync HOST=mb` (from WikiOracle parent) then
`ssh admin@metalbaby.local 'cd .../basicmodel && MODEL_DEBUG=0 .venv/bin/python test/brick_recon.py MM_xor.xml'`
Expected: same break set (compile dispatch is device-independent;
this confirms no CUDA-only breaks differ).

- [ ] **Step 4: Checkpoint — report the backlog**

Summarize: total breaks, non-grammar (must-fix) vs grammar
(deferrable) counts, and the non-grammar break classes. This sizes
Task 3 and is the natural checkpoint to report before mass edits.

---

### Task 3: Eliminate non-grammar breaks — one class per iteration

This is a **procedure** (the backlog is data-driven from Task 2), run
non-grammar-first until `fullgraph=True` passes for the non-grammar
path. Worked template for ONE break:

**Files:** the break's source module (`bin/Spaces.py` / `bin/Layers.py`
/ `bin/Models.py`), per the recon stack.

- [ ] **Step 1: Pick the top non-grammar break** from
  `doc/plans/recon-breaks-MM_xor.md`. Classify the cause:
  - data-dependent Python on host values (lex/OOV/cursor/string ops)
    → **relocate to the eager `TheData` producer**;
  - data-dependent tensor control flow (`bool()`, `.item()`,
    boolean-mask gather, `if t.any()`) → **rewrite static on-device**
    (the landed pattern: `torch.where`, `compute_masked`,
    `torch.zeros((),…)`, masked-sum/clamp);
  - bounded Python loop over a fixed count → **unroll / vectorize**.

- [ ] **Step 2: Write/extend a failing test** asserting the new
  behavior is numerically identical to the old (reuse the
  representative-order suite; add a focused unit test if the site has
  none — e.g. compare old-vs-new output on a fixed tensor).

- [ ] **Step 3: Implement the one change** (relocate / rewrite-static
  / unroll). No batching multiple breaks.

- [ ] **Step 4: Verify** — CPU representative order (Task 1 Step 5
  command), 0 regressions; then re-run recon (Task 2 Step 2): the
  targeted break is gone, **no new non-grammar break introduced**.

- [ ] **Step 5: Repeat** Steps 1–4 until recon reports
  `non-grammar breaks (must-fix): 0`.

---

### Task 4: Acceptance gate

**Files:** Modify `bin/Models.py` `enable_compiled_step` +
`_brick_preflight`.

- [ ] **Step 1:** In `enable_compiled_step`, add a non-grammar
  acceptance mode: when `MODEL_FULLGRAPH=1`, compile with
  `fullgraph=True` (raises on any residual break). Grammar-path code
  is invoked through an explicit eager escape hatch
  (`torch._dynamo.disable`) so only non-grammar must be break-free.

- [ ] **Step 2:** Decide grammar isolation: wrap the grammar entry
  (`SyntacticLayer` compose / chart dispatch) in
  `@torch._dynamo.disable` so `fullgraph=True` enforces the
  non-grammar contract while grammar stays eager (deferred per design).

- [ ] **Step 3: Verify** — `MODEL_FULLGRAPH=1` non-grammar compiles
  with 0 breaks (CPU); CPU representative order 0 regressions; tiny
  bounded metalbaby: O1 frames > 0, step-scoped
  `cudaMemcpyDtoH` re-measured (record the real number — prior
  58/430 were eager).

- [ ] **Step 4:** Update `doc/BrickHostSyncStatus.md` and the parent
  spec status: O1 fixed, non-grammar break-free, real device-event
  number, grammar breaks deferred (list them).

---

## Self-review

- **Spec coverage:** boundary (Task 1), whole-forward-break-free
  non-grammar (Tasks 2–4), bucketing (recon shapes feed Task 3/later
  phases), vocab non-issue (precondition noted, no task needed),
  fullgraph gate (Task 4), method=recon-then-eliminate (Tasks 2–3),
  grammar deferral (Task 4 Step 2). Covered.
- **Placeholders:** Task 3 is a procedure not a placeholder (backlog
  is recon-data-driven; the worked template is concrete). No TBDs.
- **Type consistency:** `enable_compiled_step()` / `_compiled_step`
  used consistently in Tasks 1 & 4; recon output path
  `doc/plans/recon-breaks-MM_xor.md` consistent in Task 2.

## Verification (every task)

CPU representative order (Task 1 Step 5 command) → tiny bounded
metalbaby (O1 frames>0; recon parity; step-scoped DtoH). No commits
unless the owner asks. metalbaby runs tiny/bounded/single (OOM
caution).
