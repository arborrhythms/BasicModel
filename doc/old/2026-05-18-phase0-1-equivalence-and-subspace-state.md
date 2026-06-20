# Phase 0 + Phase 1 Implementation Plan — Equivalence Harness & Subspace-Carried State

> **STATUS (2026-05-19): COMPLETE & VERIFIED.** Phase 0 (equivalence gate)
> and Phase 1.0–1.6 + 1A.1/1A.2 all landed, gated, behaviour-identical /
> idempotent. See the authoritative *MISSION COMPLETION STATUS* in
> `doc/plans/2026-05-18-two-loop-pipeline-architecture.md` for the
> end-to-end picture (incl. Reworks A/B and the deferred Phase-5 capture
> closeout). Code uncommitted on `main` per the project git convention.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **PROJECT GIT CONVENTION (hard rule):** agents NEVER run `git` writes
> (`add`/`commit`/`rm`/`checkout`/`stash`/`push`). Every "Commit" step below is a
> **hand-off to the human owner** with a suggested message; the owner performs the
> `git` write. Read-only `git` (`status`/`diff`/`log`) is fine.

**Goal:** Build a reusable bit-identical equivalence gate over any `Space.forward`/
`Space.reverse`, then relocate all per-round mutable Python/nn.Module state in the
traced forward onto a single per-sentence working subspace threaded through the
forward chain — eliminating the recompile/implicit-sync class structurally.

**Architecture:** Phase 0 generalizes the proven inline candidate-vs-reference
monkeypatch in `test/bpe_gpu_equiv.py` into `test/space_equiv.py` (run candidate +
reference back-to-back on the *same* inputs at the *same* codebook state, assert
bit-identical, proceed on the reference). Phase 1 introduces one carrier object
allocated in the per-sentence reset path and threaded through every
`Space.forward()`/`reverse()`; each offender (cursor/generation counter,
`_recurrent_pass_idx`, `_subspaceForPS/SS`, the `inputs/percepts/...` aliases,
`_embedded_input/_ss_cache/_staged_*/_c_prior`) is moved onto it one class at a
time, each move gated behavior-identical by the Phase 0 harness + the CPU
representative suite.

**Tech Stack:** Python, PyTorch (`torch.compile`/dynamo/inductor, CUDA-graph
trees), pytest. Repo entrypoints: `bin/Models.py`, `bin/Spaces.py`,
`bin/Language.py`. Reference config: `data/MM_5M.xml` (frozen-vocab GPU-train).

**Source spec:** `doc/plans/2026-05-18-two-loop-pipeline-architecture.md`
(Phases 0 & 1, the State-ownership contract §82-108, and the Phase 1-D DECISION
§0 premise corrections).

---

## Why these two phases, and in this order

- Phase 0 is the **safety net for every later phase** — nothing in Phase 1 (or
  2–5) may land without a bit-identical gate, because a silent numeric
  divergence corrupts training without failing any test (project memory:
  *fail loud on numerical divergence*).
- Phase 1 is **the highest-value phase** (spec §181-183) and the **hard
  prerequisite for Phase 2**: the Phase 1-D selection tensor, the NULL-derived
  seal, and the STM stack state all require the threaded per-sentence carrier
  to exist first. Phase 2's shift-reduce/selection code cannot be written
  without placeholders until this carrier's shape is concrete — which is the
  output of Phase 1.

## Premise corrections this plan assumes (verified in code; spec body is wrong)

Per the Phase 1-D DECISION §0 of the source spec:
1. `_forward_body` recurrence is **PS_t, SS_t → CS_t** consuming round-*t*
   (`Models.py:4283-4291`); only CS's own output feeds the next pass. There is
   no "round N-1 of PS/SS". `conceptualOrder` defaults to **1**.
2. The chart→STM producer is **retired** (commit `a8737da`); irrelevant to
   Phase 0/1 (no STM work here) but do not "fix" the inert `_chart_compose_at_C`.
3. STM has no reducer (irrelevant until Phase 2).

These do not change Phase 0/1 mechanics but prevent the executor from
"correcting" code that is intentionally inert.

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `test/space_equiv.py` | **Create** | Reusable inline equivalence gate over an arbitrary `Space` method (the generalized `bpe_gpu_equiv.py` pattern). |
| `test/test_space_equiv_selfcheck.py` | **Create** | pytest self-check: identity candidate passes; perturbed candidate is caught (fail-loud). |
| `bin/Spaces.py` | Modify | Add the `WorkingState` carrier class; allocate it in the reset path; thread it through `Space.forward/reverse`; absorb `_subspaceForPS/SS`. |
| `bin/Language.py` | Modify | Allocate the carrier in `WordSpace.soft_reset`/`Reset`; relocate `_compose_generation`/`_cursor_compose*` onto it. |
| `bin/Models.py` | Modify | Pass the carrier from the reset lifecycle into the forward chain; relocate `_recurrent_pass_idx`, the `inputs/percepts/...` aliases, `_embedded_input/_ss_cache/_staged_*/_c_prior`. |

Carrier design (locked here so later tasks are placeholder-free):

```python
# bin/Spaces.py  (new class, module scope, near SubSpace)
class WorkingState:
    """Per-sentence working state, allocated ONCE in the per-sentence reset
    path (never in __init__, never in the traced forward) and threaded
    through every Space.forward()/reverse() as the single carrier of all
    cross-stage/cross-round state. Fixed-shape, tensor-carrying. No
    nn.Module registration (plain object) so it never mutates _modules
    inside the trace (the dynamo guard that forces unique_graphs=2).

    Fields (all device tensors or plain refs, fixed shape per sentence):
      cursor          int64 [n_tiers]   per-tier rule cursor (was _cursor_compose*)
      gen             int64 []          compose generation (was _compose_generation)
      recur_pass      int64 []          recurrent pass index (was _recurrent_pass_idx)
      cs_for_ps       SubSpace          CS->PS carrier (was _subspaceForPS)
      cs_for_ss       SubSpace          CS->SS carrier (was _subspaceForSS)
      valid_mask      Tensor|None       carried by reference (was copy_context)
      errors          Error             carried by reference
    Extended in Phase 2 with the selection tensor + STM stack handles.
    """
    __slots__ = ("cursor", "gen", "recur_pass", "cs_for_ps", "cs_for_ss",
                 "valid_mask", "errors")
```

`__slots__` + plain `object` (NOT `nn.Module`) is mandatory: this is exactly
why the existing band-aids use `object.__setattr__` — registering carrier
fields in `_modules`/`_buffers` inside the traced forward fails the
`len(self._modules)==N` guard and forces `unique_graphs=2`.

---

# PHASE 0 — Reusable equivalence gate

### Task 0.1: Generalized inline equivalence gate `test/space_equiv.py`

**Files:**
- Create: `test/space_equiv.py`
- Reference pattern (do not modify): `test/bpe_gpu_equiv.py:51-124`

- [ ] **Step 1: Write the gate module**

Create `test/space_equiv.py` with the generalized pattern. This lifts
`_clone`/`_grab`/`_assert_eq`/`install_gate`/`run_gate` from
`test/bpe_gpu_equiv.py` but parameterizes (a) the class+method to gate,
(b) the output-snapshot function, so it wraps *any* `Space.forward`/
`Space.reverse`:

```python
"""Reusable bit-identical equivalence gate over an arbitrary Space method.

Generalizes test/bpe_gpu_equiv.py: for every gated call, run the CANDIDATE
and the REFERENCE back-to-back on the SAME inputs at the SAME codebook
state, snapshot each one's outputs, assert bit-identical, then let the
model proceed on the REFERENCE result (deterministic prod path; we are
validating the candidate against it). Capture-then-replay is unfaithful
(the codebook trains between batches). Fail loud on the first divergence
(project memory: fail loud on numerical divergence -- silent here).

Usage (as a gate tool, NOT pytest):
    from test.space_equiv import run_space_gate
    n = run_space_gate(Spaces.ConceptualSpace, "forward",
                        candidate_fn=<new impl>, snapshot=<fn(space,out)->dict>)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "none"
os.environ["BASICMODEL_DEVICE"] = "cpu"

_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin"))
sys.path.insert(0, str(_p / "bin"))

import torch
from data import TheData
from Models import BaseModel
from util import init_config, init_device
import Spaces


def _clone(t):
    return t.detach().clone() if torch.is_tensor(t) else t


def _assert_eq(name, ref, cand, idx):
    if ref is None and cand is None:
        return
    assert (ref is None) == (cand is None), (
        f"call#{idx} {name}: one side None")
    if torch.is_tensor(ref):
        assert ref.shape == cand.shape, (
            f"call#{idx} {name}: shape {tuple(ref.shape)} != "
            f"{tuple(cand.shape)}")
        if not torch.equal(ref, cand):
            d = (ref.float() - cand.float()).abs()
            raise AssertionError(
                f"call#{idx} {name}: DIVERGENCE -- "
                f"n_diff={int((d > 0).sum())}/{d.numel()} "
                f"max|Δ|={float(d.max()):.3e}")
    else:
        if ref != cand:
            raise AssertionError(
                f"call#{idx} {name}: DIVERGENCE -- {ref!r} != {cand!r}")


def default_snapshot(space, out):
    """Default output snapshot: the muxed event W (the exact float tensor
    fed downstream) -- the same criterion bpe_gpu_equiv uses.
    """
    ev = None
    if out is not None and getattr(out, "event", None) is not None:
        ev = out.event.getW()
    return {"event_W": _clone(ev)}


def install_space_gate(cls, method_name, candidate_fn, snapshot):
    """Monkeypatch ``cls.method_name`` so every call runs candidate then
    reference back-to-back, asserts bit-identical via ``snapshot``, and
    lets the model proceed on the REFERENCE result (deterministic prod
    path). Returns {"n": call_count}; raises on first divergence. Restores
    the original on the returned ``state['restore']()``.

    This is the EXACT proven pattern of ``test/bpe_gpu_equiv.py``,
    parameterized by (class, method, snapshot).

    GATE-TARGET CONTRACT: the gated method must NOT mutate a graph-saved
    leaf (an ``nn.Parameter`` consumed by a retained autograd graph) in
    the call. Whole ``Space.forward``/``reverse`` currently violate this
    -- the VQ codebook EMA-writes ``subspace.event.W`` (an nn.Parameter)
    in-place every call (``bin/Spaces.py:2048``) while the model
    accumulates one autograd graph across the 3 recurrent forwards/batch;
    a second (reference) call + the next batch's backward then trip the
    version-counter check. ``bpe_gpu_equiv`` sidesteps this by gating
    ``_embed_bpe`` -- a PRE-VQ substep that, under frozen vocab
    (``word_learning=0``), mutates nothing. **Until Phase 1A removes
    VQ-EMA, gate pre-VQ substeps (``_embed_bpe``) and the post-carrier
    Phase-1 relocations via the representative suite + fullgraph gate.
    After Phase 1A (no in-call Parameter EMA), ``forward``/``reverse``
    become directly gateable bit-identical** -- add them as targets then.
    """
    reference_fn = getattr(cls, method_name)
    state = {"n": 0}

    def _gated(self, *args, **kwargs):
        cand_out = candidate_fn(self, *args, **kwargs)
        cand = snapshot(self, cand_out)
        ref_out = reference_fn(self, *args, **kwargs)
        ref = snapshot(self, ref_out)
        i = state["n"]
        for k in ref:
            _assert_eq(k, ref[k], cand.get(k), i)
        state["n"] = i + 1
        return ref_out

    setattr(cls, method_name, _gated)
    state["restore"] = lambda: setattr(cls, method_name, reference_fn)
    return state


def run_space_gate(cls, method_name, candidate_fn, snapshot=default_snapshot,
                   max_batches=5, batch_size=8):
    """Drive a real frozen MM_5M runEpoch with the gate installed."""
    init_device("cpu")
    cfg = str(_p / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                 num_shards=1, max_docs=64)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    m.perceptualSpace.chunk_layer.word_learning = 0  # frozen-vocab contract
    state = install_space_gate(cls, method_name, candidate_fn, snapshot)
    try:
        opt = m.getOptimizer(lr=1e-4)
        m.runEpoch(optimizer=opt, batchSize=batch_size, split="train",
                   max_batches=max_batches)
    finally:
        state["restore"]()
    return state["n"]
```

- [ ] **Step 2: Write the failing self-check test**

Create `test/test_space_equiv_selfcheck.py`:

```python
"""Self-check: the gate passes for an identity candidate and FAILS LOUD
for a perturbed candidate. Targets PerceptualSpace._embed_bpe -- the
same pre-VQ, frozen-vocab-idempotent anchor test/bpe_gpu_equiv.py uses
(whole-forward is not bit-identically gateable until Phase 1A removes
VQ-EMA; see install_space_gate's GATE-TARGET CONTRACT). If the gate
cannot catch a planted divergence it is worthless as a safety net.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
import pytest
import torch
import Spaces
from test.space_equiv import run_space_gate, _clone


def _bpe_snapshot(ps, out):
    # event_W + the routing mask -- exactly bpe_gpu_equiv._grab's criterion.
    ev = out.event.getW() if (out is not None and out.event is not None) else None
    return {"event_W": _clone(ev),
            "word_active": _clone(getattr(ps, "_bpe_word_mask", None))}


def test_identity_candidate_passes():
    # Candidate == the real trie reference => every call bit-identical.
    n = run_space_gate(Spaces.PerceptualSpace, "_embed_bpe",
                        candidate_fn=Spaces.PerceptualSpace._embed_bpe_trie,
                        snapshot=_bpe_snapshot, max_batches=2)
    assert n > 0, "gate exercised nothing (config not BPE mode?)"


def test_perturbed_candidate_is_caught():
    ref = Spaces.PerceptualSpace._embed_bpe_trie

    def perturbed(self, *a, **k):
        out = ref(self, *a, **k)
        if out is not None and getattr(out, "event", None) is not None:
            w = out.event.getW()
            if torch.is_tensor(w) and w.numel():
                out.event.setW(w + 1e-3)  # plant a divergence
        return out

    with pytest.raises(AssertionError, match="DIVERGENCE"):
        run_space_gate(Spaces.PerceptualSpace, "_embed_bpe",
                        candidate_fn=perturbed,
                        snapshot=_bpe_snapshot, max_batches=2)
```

- [ ] **Step 3: Run the self-check, expect FAIL**

Run: `.venv/bin/python -m pytest test/test_space_equiv_selfcheck.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `test.space_equiv`
or `AttributeError` until Step 1's file is saved and importable.

- [ ] **Step 4: Make it importable and run again, expect PASS**

Ensure `test/__init__.py` exists (create empty if absent:
`ls test/__init__.py || : > test/__init__.py` — the owner creates the file
if missing; do not `git add`). Re-run:
`.venv/bin/python -m pytest test/test_space_equiv_selfcheck.py -v`
Expected: PASS (2 passed) — identity passes, perturbed raises
`AssertionError: ... DIVERGENCE`.

- [ ] **Step 5: Commit (hand-off)**

Tell the owner: *"Phase 0 gate ready. Suggested commit:"*
```
Phase 0: reusable bit-identical Space.forward/reverse equivalence gate

Generalizes test/bpe_gpu_equiv.py into test/space_equiv.py; self-check
proves it catches a planted divergence (fail-loud).
```
Owner runs `git add test/space_equiv.py test/test_space_equiv_selfcheck.py
test/__init__.py && git commit`. Do not run git yourself.

### Task 0.2: Wire the gate into the representative suite doc

**Files:**
- Modify: `doc/plans/2026-05-18-two-loop-pipeline-architecture.md` (Verification §1 — note the new tool)

- [ ] **Step 1:** Append one line to the spec's Verification §1 noting the
  reusable tool path: `test/space_equiv.py::run_space_gate(cls, method, ...)`
  is the Phase-0 gate every later phase calls. (Documentation only — no code.)
- [ ] **Step 2: Commit (hand-off)** — suggested message:
  `Phase 0: document the reusable equivalence gate in the source spec`.

---

# PHASE 1 — Subspace-carried-state contract

> Each task is gated **behavior-identical**: the Phase 0 gate (event_W
> bit-identical) **plus** the CPU representative suite with zero
> regressions. Per project memory *run targeted testpoints, not full
> suite*: iterate each task on the specific suite below; the full `test/`
> suite is only the final Phase-1 gate.

**Representative CPU suite (the per-task net), exact command:**

```
.venv/bin/python -m pytest -q \
  test/test_mm_xor.py test/test_mm_boolean.py test/test_universality.py \
  test/test_invertibility.py test/test_discourse_space.py \
  test/test_xor_grammar.py test/test_signal_router_xor_grammar.py \
  test/test_perceptualspace_bpe_forward.py \
  test/test_phase2_pipeline_primitives.py \
  test/test_compiled_step_invoked.py test/test_brick_no_sync.py
```
Expected baseline: **≥207 passed, 0 failed** (spec Verification §2). Record
the count before Task 1.0 as the regression baseline.

### Task 1.0: Add the `WorkingState` carrier class (inert)

**Files:**
- Modify: `bin/Spaces.py` (add `WorkingState` at module scope, immediately
  before the `SubSpace` class definition)

- [ ] **Step 1: Add the class**

Insert the `WorkingState` class exactly as specified in the *File
Structure* section above (the `__slots__` plain-object form). Add a
module-level factory used by the reset path:

```python
def new_working_state(n_tiers, device):
    ws = WorkingState()
    ws.cursor = torch.zeros(n_tiers, dtype=torch.int64, device=device)
    ws.gen = torch.zeros((), dtype=torch.int64, device=device)
    ws.recur_pass = torch.zeros((), dtype=torch.int64, device=device)
    ws.cs_for_ps = None        # set lazily by ConceptualSpace.forward
    ws.cs_for_ss = None
    ws.valid_mask = None
    ws.errors = None
    return ws
```

- [ ] **Step 2: Prove it is inert** — nothing references `WorkingState`
  yet. Run the representative suite.
  Run: the representative-suite command above.
  Expected: identical pass count to the recorded baseline (≥207, 0
  regressions) — adding an unreferenced class changes nothing.

- [ ] **Step 3: Run the Phase 0 gate (identity)** to confirm the harness
  still green on this tree:
  Run: `.venv/bin/python -m pytest test/test_space_equiv_selfcheck.py -v`
  Expected: 2 passed.

- [ ] **Step 4: Commit (hand-off)** — suggested message:
  `Phase 1.0: add inert WorkingState carrier (plain-object, __slots__)`.

### Task 1.1: Allocate + thread the carrier (still inert end-to-end)

**Files:**
- Modify: `bin/Language.py` — `WordSpace.soft_reset:5779-5835`,
  `WordSpace.Reset:5715-5777` (allocate the carrier on the per-sentence
  reset path)
- Modify: `bin/Models.py` — `dispatch_soft_reset:3176-3188` and the
  runEpoch tail `:3594-3597` (own the carrier at lifecycle scope)
- Modify: `bin/Spaces.py` — `Space.forward`/`Space.reverse` signatures to
  accept an optional `work=None` kwarg threaded through (unused this task)

- [ ] **Step 1: Allocate in the per-sentence reset path**

In `bin/Language.py`, `WordSpace.soft_reset`, add carrier (re)allocation.
Current head of the `batch is None` branch (`Language.py:5800-5807`,
verbatim):

```python
        if batch is None:
            self.arm_stm()
            self.clear_last_svo()
            self.clear_sentence_completed()
            # Reset every row's parse-side working state. clear_sentence
            # zeroes the WordSubSpace stack; the category and
            # reconstruction stacks fan out to the same row count.
            self.clear_sentence()
```

Insert immediately after `self.clear_sentence_completed()` (before the
`clear_sentence()` comment), and symmetrically in the `b = int(batch)`
branch after `self.clear_sentence_completed(b)` (`Language.py:5820`):

```python
            # Per-sentence working carrier: allocated HERE (reset path),
            # never in __init__, never in the traced forward. Single
            # carrier of all cross-stage/cross-round state (spec §84-89).
            from Spaces import new_working_state
            self._work = new_working_state(n_tiers=3, device=self._device())
```
(Use the existing device accessor; if `self._device()` does not exist,
use `self.subspace.event.getW().device` — confirm by reading
`WordSpace` for an existing device helper before writing; pick one and
use it consistently.)

- [ ] **Step 2: Initialize the slot in `__init__`** so the eager/uncompiled
  and first-forward paths never `AttributeError`. In `WordSpace.__init__`
  near the cursor/generation init (`Language.py:4632-4633`, verbatim):

```python
        self._compose_generation = 0
        self._generate_generation = 0
```
add directly below:
```python
        self._work = None  # allocated per-sentence in soft_reset/Reset
```

- [ ] **Step 3: Thread `work` through the Space forward chain (inert)**

Add an optional `work=None` parameter to `Space.forward` and
`Space.reverse` in `bin/Spaces.py` and pass it down unchanged from
`Models._forward_per_stage`/`_forward_body` call sites. This task only
*plumbs* it; no body reads it yet. Concretely, at each
`*.forward(...)` call inside `_forward_body` (`Models.py:4283`, `:4290`,
`:4291`) add `, work=self.wordSpace._work` as a trailing kwarg, and add
`work=None` to the receiving signatures (`PerceptualSpace.forward`
`Spaces.py:7696`, `SymbolicSpace.forward` `Spaces.py:10187`,
`ConceptualSpace.forward` `Spaces.py:8753`, base `Space.forward`).

- [ ] **Step 4: Behavior-identical gate**

Run: representative suite command. Expected: ≥207 passed, 0 regressions
(threading an unread kwarg changes nothing).
Run the Phase-0 gate against `ConceptualSpace.forward` identity:
```
.venv/bin/python -c "import Spaces; from test.space_equiv import run_space_gate; \
print(run_space_gate(Spaces.ConceptualSpace,'forward', \
candidate_fn=Spaces.ConceptualSpace.forward))"
```
Expected: prints a call count `> 0`, no `AssertionError` (identity ⇒
bit-identical).

- [ ] **Step 5: Commit (hand-off)** — suggested message:
  `Phase 1.1: allocate per-sentence WorkingState in reset path; thread work= (inert)`.

### Task 1.2: Relocate the rule cursor / generation counter (kills recompile #3)

**Files:**
- Modify: `bin/Language.py` — `SyntacticLayer.__init__:3989-3992`,
  `SyntacticLayer._next_rule_name:4007-4053`, `WordSpace.compose:5155`
  (the `self._compose_generation += 1` bump)

This is the structural fix for recompile cause #3 (spec §24-26, §138-141).
Verbatim current `_next_rule_name` compose head (`Language.py:4022-4029`):

```python
    ws = self._word_space
    if direction == 'compose':
        rules = ws.current_rules
        gen = ws._compose_generation
        if gen != self._cursor_compose_gen:
            self._cursor_compose = 0
            self._cursor_compose_gen = gen
        cursor = self._cursor_compose
```

- [ ] **Step 1: Replace the data-dependent Python branch with a carrier
  tensor read.** The cursor becomes a per-tier int64 slot on
  `ws._work.cursor`; the generation gate is removed (the reset path
  zeroes `cursor` per sentence, so the "new generation ⇒ reset cursor"
  branch is structurally unnecessary). New body:

```python
    ws = self._word_space
    work = ws._work
    tier_ix = self._tier_index()  # 0/1/2 for P/C/S; add this helper
    if direction == 'compose':
        rules = ws.current_rules
        if work is not None:
            cursor = int(work.cursor[tier_ix])
        else:
            cursor = self._cursor_compose  # eager/uncompiled fallback
```
Add a `_tier_index(self)` helper on `SyntacticLayer` mapping
`self.tier` ('P'/'C'/'S') → 0/1/2.

- [ ] **Step 2: Advance the cursor on the carrier, not the nn.Module int.**
  Current advance (`Language.py:~4043`, verbatim region):
```python
        if direction == 'compose':
            self._cursor_compose = cursor + 1
```
becomes:
```python
        if direction == 'compose':
            if work is not None:
                work.cursor[tier_ix] = cursor + 1
            else:
                self._cursor_compose = cursor + 1
```

- [ ] **Step 3: Replace the generation-counter reset with an explicit
  per-`compose` cursor zero (BEHAVIOR-FAITHFUL).** The OLD semantics: the
  per-tier cursor resets to 0 **on every `WordSpace.compose` call** (the
  `gen != self._cursor_compose_gen` branch in `_next_rule_name`, driven by
  `self._compose_generation += 1` at `Language.py:5155`). Relying on
  `soft_reset` to zero the carrier cursor is **only** equivalent if
  `compose()` is called ≤1× per sentence — NOT guaranteed (recurrent
  passes / `conceptualOrder` / per-forward chart dispatch can call it
  multiple times per `soft_reset` interval). The faithful, capture-safe
  translation is an **unconditional** tensor zero at the top of
  `WordSpace.compose` (carrier path) — no data-dependent Python branch
  (recompile #3 still gone) and exactly reproduces the old per-`compose`
  reset regardless of call frequency. In `WordSpace.compose`, locate
  `self._compose_generation += 1` (`Language.py:5155`) and make it:
```python
        if self._work is not None:
            self._work.cursor.zero_()   # per-compose cursor reset (was gen-counter)
        else:
            self._compose_generation += 1   # eager/uncompiled fallback
```
  Mirror the same in `WordSpace.generate` for `_generate_generation`
  (`Language.py:5194`) if Task 1.2 also relocates the generate cursor;
  otherwise leave generate on the eager counter and note it.
  Correspondingly, **Step 1 must NOT keep the `gen`-comparison branch**:
  the carrier-path `_next_rule_name` reads `work.cursor[tier_ix]` with no
  generation gate at all (the zero now happens in `compose`). Keep the
  `gen`/`_cursor_compose_gen` logic ONLY on the `work is None` eager
  fallback.

- [ ] **Step 4: Behavior-identical gate.** Run the representative suite
  (≥207, 0 regr.) **and** the Phase-0 gate over `SymbolicSpace.forward`
  identity (the S-tier dispatcher is the heaviest cursor consumer):
```
.venv/bin/python -c "import Spaces; from test.space_equiv import run_space_gate; \
print(run_space_gate(Spaces.SymbolicSpace,'forward', \
candidate_fn=Spaces.SymbolicSpace.forward))"
```
Expected: call count `>0`, no `AssertionError`.

- [ ] **Step 5: Commit (hand-off)** — suggested message:
  `Phase 1.2: carry rule cursor as WorkingState tensor; remove gen-counter branch (recompile #3)`.

### Task 1.3: Relocate `_recurrent_pass_idx` onto the carrier

**Files:**
- Modify: `bin/Models.py:4282` (set), `:4316` (reset),
  `bin/Spaces.py:7787-7790` and `:7722-7726` (read in
  `PerceptualSpace.forward`)

Verbatim current set/reset (`Models.py:4282`, `:4316`):
```python
            self.perceptualSpace._recurrent_pass_idx = t
...
        self.perceptualSpace._recurrent_pass_idx = 0
```
Verbatim current read (`Spaces.py:7787-7790`):
```python
        _oi = min(max(self._recurrent_pass_idx, 0), len(self.pi_input) - 1)
        pi_input_k   = self.pi_input[_oi]
        pi_concept_k = self.pi_concept[_oi]
```

- [ ] **Step 1:** In `_forward_body`, replace the set/reset with carrier
  writes (the carrier is owned by `wordSpace._work`; available via the
  threaded `work`):
```python
            if work is not None: work.recur_pass.fill_(t)
            else: self.perceptualSpace._recurrent_pass_idx = t
...
        if work is not None: work.recur_pass.zero_()
        else: self.perceptualSpace._recurrent_pass_idx = 0
```
- [ ] **Step 2:** In `PerceptualSpace.forward`, read from `work` when
  present (it is now a threaded kwarg from Task 1.1):
```python
        _rp = int(work.recur_pass) if work is not None else self._recurrent_pass_idx
        _oi = min(max(_rp, 0), len(self.pi_input) - 1)
        pi_input_k   = self.pi_input[_oi]
        pi_concept_k = self.pi_concept[_oi]
```
Apply the same `work`-aware read at the serial-warm-path gate
(`Spaces.py:7722-7726`).
- [ ] **Step 3: Behavior-identical gate.** Representative suite (≥207, 0
  regr.) + Phase-0 gate over `PerceptualSpace.forward` identity (the
  `_recurrent_pass_idx` consumer). Expected: no `AssertionError`.
- [ ] **Step 4: Commit (hand-off)** — suggested message:
  `Phase 1.3: carry _recurrent_pass_idx on WorkingState`.

### Task 1.4: Fold `_subspaceForPS`/`_subspaceForSS` into carrier views

**Files:**
- Modify: `bin/Spaces.py` — `ConceptualSpace.forward:8927`, `:8944`,
  `:8951-8953` (the `object.__setattr__` band-aids) and the
  `_forward_body` reads `Models.py:4303-4304`

Verbatim current sites (from exploration; `Spaces.py:8927/8944/8951`):
```python
        object.__setattr__(self, "_subspaceForSS", vspace)
...
            self._subspaceForPS.set_event(lifted)
...
            object.__setattr__(self, "_subspaceForPS", SubSpace(...))
```
Verbatim current carry (`Models.py:4303-4304`):
```python
            prevCS_forPS = cs._subspaceForPS
            prevCS_forSS = cs._subspaceForSS
```

- [ ] **Step 1:** Write the two CS output views onto the carrier instead of
  `object.__setattr__` on the nn.Module. In `ConceptualSpace.forward`:
```python
        if work is not None:
            work.cs_for_ss = vspace
        else:
            object.__setattr__(self, "_subspaceForSS", vspace)
```
and analogously for the `_subspaceForPS` set/`set_event`/fallback (keep
the persistent `_subspaceForPS` SubSpace object reused via `set_event`;
only the *reference handoff* moves to `work.cs_for_ps`).
- [ ] **Step 2:** In `_forward_body`, read the carrier when present:
```python
            prevCS_forPS = (work.cs_for_ps if work is not None
                            else cs._subspaceForPS)
            prevCS_forSS = (work.cs_for_ss if work is not None
                            else cs._subspaceForSS)
```
- [ ] **Step 3: Behavior-identical gate.** Representative suite (≥207, 0
  regr.) + Phase-0 gate `ConceptualSpace.forward` identity. Expected: no
  `AssertionError`.
- [ ] **Step 4: Commit (hand-off)** — suggested message:
  `Phase 1.4: fold _subspaceForPS/SS handoff into WorkingState views`.

### Task 1.5: Relocate the `inputs/percepts/...` aliases + scratch caches

**Files:**
- Modify: `bin/Models.py` — `_forward_per_stage` `:4553-4566` and the
  empty-input duplicate `:4506-4524` (the five `object.__setattr__`
  aliases); `_embedded_input`/`_ss_cache`/`_staged_*`/`_c_prior`
  definitions/uses (`Spaces.py:~6677,~8826`, `Models.py`)

Verbatim alias block (`Models.py:4558-4565`):
```python
        object.__setattr__(self, "inputs",   self.inputSpace.subspace)
        object.__setattr__(self, "percepts", self.perceptualSpace.subspace)
        object.__setattr__(self, "concepts", self.conceptualSpace.subspace)
        object.__setattr__(self, "symbols",  self.symbolicSpace.subspace)
        object.__setattr__(self, "outputs",  self.outputSpace.subspace)
```

- [ ] **Step 1: Classify each.** These five are *read-only convenience
  handles* to each Space's terminal subspace — they are **not**
  cross-round state and are already out of `_modules` via
  `object.__setattr__`. The spec (§253) says the band-aids "should remain
  until Phase 1 subsumes them". Subsume = move the reads to go through the
  owning Space directly (`self.perceptualSpace.subspace`), deleting the
  aliases entirely. Grep every read site:
  Run: `grep -rn "self\.\(inputs\|percepts\|concepts\|symbols\|outputs\)\b" bin/`
  and replace each `self.percepts` → `self.perceptualSpace.subspace` (etc.).
- [ ] **Step 2:** Delete the two `object.__setattr__` alias blocks
  (`:4558-4565` and the `:4506-4524` duplicate) entirely.
- [ ] **Step 3:** For genuine cross-round scratch (`_embedded_input`,
  `_ss_cache`, `_staged_*`, `_c_prior`): confirm via grep whether each is
  (a) cross-round state (→ move onto `work`) or (b) eager-only
  producer/consumer scope (→ leave; spec §106-108 says these belong to
  eager/runBatch scope). For each that is cross-round, add a `__slots__`
  field to `WorkingState` and route reads/writes through `work`, mirroring
  Task 1.4. Document the (a)/(b) classification inline in the commit
  message (it is the audit the spec asks for).
- [ ] **Step 4: Behavior-identical gate.** Representative suite (≥207, 0
  regr.) + Phase-0 gate over `PerceptualSpace.forward` AND
  `SymbolicSpace.forward` identity. Expected: no `AssertionError`.
- [ ] **Step 5: Commit (hand-off)** — suggested message:
  `Phase 1.5: subsume inputs/percepts/... aliases; relocate cross-round scratch onto WorkingState`.

---

# PHASE 1A — Eliminate VQ-EMA (learnable lookup + CS-entry symbol-snap)

> Implements the **VQ-EMA ELIMINATION DECISION** in the source spec
> (`2026-05-18-two-loop-pipeline-architecture.md`). This removes the last
> quantizer-driven in-trace persistent-buffer mutation. Gate mode for the
> perceptual→concept path is **characterize-and-approve** (semantic delta,
> NOT bit-identical); every other behavior must stay bit-identical.
> Depends on Phase 0 (gate) + Phase 1 (carrier) landed.

### Task 1A.1: Perceptual VQ snap → learnable concept-activation lookup

**Files:** `bin/Spaces.py` — `PerceptualSpace.__init__` (add the
`nn.Embedding(<nVectors>, concept_dim)` lookup, preallocated), the VQ snap
`:7807-7818` (replace with a gather from the lookup keyed by the chunk/word
index already resolved in `_embed`/`_embed_bpe`/`_embed_byte`), `_bpe_emit`
(OOV → existing segmented max-fuse over sub-token lookup rows, no VQ).

- [ ] Implement the learnable lookup; key = the resolved chunk/word index;
  OOV/not-in-live-view = sub-token rows + the existing max-fuse. No VQ
  call, no EMA, no table growth in the traced path (preallocated view).
- [ ] **Gate (characterize-and-approve):** run `test/space_equiv.py`
  `run_space_gate(Spaces.PerceptualSpace,'forward', candidate=<new>,
  reference=<old VQ path>)` — it WILL diverge (expected). Capture the
  delta magnitude/shape, write a 5-line characterization, present to the
  owner for approval. Representative CPU suite: no *crash* regressions
  (numeric deltas on the perceptual path are expected and characterized).
- [ ] Commit (hand-off): `Phase 1A.1: perceptual VQ-EMA -> learnable concept-activation lookup`.

### Task 1A.2: Symbol codebook → learnable (drop EMA)

**Files:** `bin/Spaces.py` — `SymbolicSpace` codebook path
(`:10405-10434` default VQ snap), `Codebook`/`VectorQuantize`
(`:1283`/`:1769-1781`) — replace the EMA buffer update with a
**gradient-trained** embedding (nearest-neighbor assignment with
straight-through estimator for the backward; NO `update_ema` buffer
write). SymbolicSpace stays the single writer.

- [ ] Replace EMA with straight-through learnable codebook (the param is
  updated by `optimizer.step`, eager, not in-trace). Keep assignment
  (argmin/topK) identical so symbol identity is preserved.
- [ ] **Gate:** characterize-and-approve (training dynamics change:
  EMA→gradient). Representative suite: no crash regressions.
- [ ] Commit (hand-off): `Phase 1A.2: symbol codebook EMA -> gradient-trained (learnable)`.

### Task 1A.3: CS-entry snap against the learnable symbol codebook

**Files:** `bin/Spaces.py` — `ConceptualSpace.forward` (add a read-only
nearest-neighbor snap of the incoming activation against
`self.symbolicSpace_ref.subspace.what` BEFORE the C-tier work; do not
write that codebook — single-writer preserved).

- [ ] Add the read-only snap (gather nearest learnable symbol row;
  straight-through for gradient). Confirm via grep that ConceptualSpace
  never writes `SymbolicSpace.subspace.what`.
- [ ] **Gate:** characterize-and-approve. Representative suite: no crash
  regressions; the equivalence gate over `SymbolicSpace.forward` (which
  no longer mutates EMA) should now pass **bit-identical without the
  snapshot/restore** as a sanity signal (note it, do not remove the
  snapshot/restore — still needed for other relocations).
- [ ] Commit (hand-off): `Phase 1A.3: snap incoming activations vs learnable symbol codebook at CS entry`.

### Task 1.6: Full-suite + metalbaby capture gate (Phase 1 acceptance)

**Files:** none (verification only)

- [ ] **Step 1: Full CPU suite, zero regressions.**
  Run: `.venv/bin/python -m pytest -q test/`
  (full suite needs `PYTORCH_ENABLE_MPS_FALLBACK=1` for the unrelated MPS
  `_cdist_backward` gap — prefix the command with it). Expected: ≥ the
  pre-Phase-1 baseline pass count, 0 new failures. Investigate any delta
  (project memory: *fail loud on numerical divergence* — a silent
  event_W change that the gate missed must be root-caused, never
  nan_to_num'd away).
- [ ] **Step 2: Faithful real-compile gate** (NOT the `explain` recon —
  spec Notes §249): run the real `fullgraph=True` + bounded `runEpoch`
  oracle the spec's Verification §2 names; expect FULLGRAPH-CLEAN for
  MM_xor/MM_5M/MM_grammar.
- [ ] **Step 3: metalbaby (bounded, single run — OOM history).**
  `make sync HOST=mb`; on metalbaby, frozen MM_5M, `reduce-overhead`,
  `TORCH_LOGS=recompiles,cudagraphs`, one tiny bounded run. **Acceptance
  (spec §182-183):** `unique_graphs=1` (was 2 — recompile #3 gone),
  `cudaMemcpyDtoH` count not increased vs the pre-Phase-1 12/step, no new
  recompile guards naming `_cursor_compose*`/`_recurrent_pass_idx`/
  `_subspaceForSS`. Record before/after numbers.
- [ ] **Step 4: Hand-off summary to owner.** Report: full-suite delta,
  fullgraph status per config, metalbaby `unique_graphs`/DtoH
  before→after. If `unique_graphs` is still 2, attribute the residual
  guard from the `TORCH_LOGS=recompiles` log (do not declare Phase 1 done
  on a green CPU suite alone — the spec's whole point is the capture
  step-function).
- [ ] **Step 5: Commit (hand-off)** — suggested message:
  `Phase 1: subspace-carried-state contract complete; unique_graphs 2->1, syncs attributed`.

---

## Self-Review (performed against the source spec)

- **Spec coverage (Phase 0):** §174-178 (generalize the
  `bpe_gpu_equiv.py` inline gate wrapping `Space.forward`/`reverse`) →
  Task 0.1/0.2. Fail-loud requirement (Verification §1, project memory) →
  the planted-divergence self-check (Task 0.1 Step 2).
- **Spec coverage (Phase 1):** §84-89 allocate carrier in
  `reset(hard=False)` → Task 1.1. §104-108 offenders: cursor/gen →
  Task 1.2; `_recurrent_pass_idx` → Task 1.3; `_subspaceForPS/SS` →
  Task 1.4; `inputs/percepts/...` aliases + `_embedded_input/_ss_cache/
  _staged_*/_c_prior` → Task 1.5. §96-99 invariant (no mutable
  object-owned data in forward/reverse) → enforced incrementally,
  asserted by the Phase-0 gate each task. §181-183 expected
  `unique_graphs→1`, syncs gone → Task 1.6 metalbaby gate.
- **Placeholder scan:** every code step shows verbatim before-code (read
  from the live tree) and concrete after-code; one residual judgement
  call is explicit and bounded — Task 1.5 Step 3's (a)/(b) classification
  of `_embedded_input/_ss_cache/_staged_*/_c_prior` is an *audit the spec
  itself defers* (§106-108 "session/exploration, with refs"), so the plan
  prescribes the grep + decision rule rather than a fake fixed answer.
- **Type consistency:** `WorkingState` `__slots__` fields
  (`cursor/gen/recur_pass/cs_for_ps/cs_for_ss/valid_mask/errors`),
  `new_working_state(n_tiers, device)`, and `SyntacticLayer._tier_index()`
  are used consistently across Tasks 1.0→1.5.
- **Scope:** Phase 0+1 only; Phases 2–5 get their own just-in-time plans
  once this carrier's shape is concrete (the explicit Phase-2
  prerequisite).

## Risks / notes

- The `WorkingState` MUST stay a plain `object` with `__slots__` — making
  it an `nn.Module` or registering its tensors as buffers re-creates the
  exact `len(self._modules)`/`_buffers` dynamo guard that forces
  `unique_graphs=2`. This is the whole point; do not "tidy" it into a
  Module.
- Keep every eager/uncompiled `else:` fallback (`work is None`) — the CPU
  test suite and `embed.py` CBOW pretrain run uncompiled and must stay
  behaviour-identical; the gate covers the compiled path, the fallback
  covers the eager path.
- Do not delete `_compose_generation`/`_cursor_compose*`/
  `_recurrent_pass_idx` attributes outright while a fallback path reads
  them; Phase 1 *relocates the authoritative copy* onto the carrier and
  keeps the attribute as the eager fallback (spec §253: band-aids remain
  until subsumed — Task 1.5 removes the now-dead alias band-aids
  specifically, not the fallbacks).
- metalbaby runs are bounded/single (OOM/reboot history, spec §238).
