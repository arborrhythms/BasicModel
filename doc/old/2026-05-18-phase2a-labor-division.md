# Phase 2A — Labor Division (selector-in-S + C/S executors) Implementation Plan

> **STATUS (2026-05-19): COMPLETE & VERIFIED.** Phase 2A.1–2A.7 landed,
> eager byte-identical; the selector-in-S contract + C/S executor split
> are the basis the Rework-A/B per-word IR objective builds on. See the
> authoritative *MISSION COMPLETION STATUS* in
> `doc/plans/2026-05-18-two-loop-pipeline-architecture.md`. Code
> uncommitted on `main` per the project git convention.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **PROJECT GIT CONVENTION (hard rule):** agents NEVER run `git` writes. Every
> "Commit" step is a hand-off: report the suggested message; the human commits.

**Goal:** Implement the ratified Phase 1-D labor division — `_RULE_TIER`
rewritten to the C/S split, the P-tier `SyntacticLayer` retired, soft-
superposition *selection* centralized in SymbolicSpace as a fixed-op-axis
tensor on the threaded `WorkingState`, the two `SyntacticLayer`s demoted to
thin C/S *executors*, and the (previously-deferred) read-only symbol snap
landed at the owner-fixed locus (post-`sigma_percept` `y`).

**Architecture:** Selection (which ops, what weights) is computed once in
SymbolicSpace and written to `WorkingState` as a fixed-op-axis weight tensor —
no cursor, no data-dependent Python branch. Two executors (C on ConceptualSpace,
S on SymbolicSpace) read that read-only tensor and apply *their* tier's ops to
*their own* codebook; they share no mutable state. The concept↔symbol shared
locus is the post-`sigma_percept` `[·, CS_dim]` tensor `y` (owner decision).

**Tech Stack:** Python, PyTorch, pytest. `bin/{Language,Spaces,Models}.py`.

**Source spec:** `doc/plans/2026-05-18-two-loop-pipeline-architecture.md` —
the **Phase 1-D DECISION** (§1 labor division, constraints 1–4) and the
**VQ-EMA ELIMINATION DECISION** (1A.3 deferral + the locus resolution).

**Prerequisite (landed, uncommitted):** Phase 0, Phase 1 (1.0–1.5), Phase
1A.1/1A.2 — `WorkingState` carrier exists and is threaded; perceptual & symbol
codebooks are learnable (idempotent forwards). Representative-suite baseline:
**135 passed / 12 skipped / 4 xfailed / 0 failed** (151 collected). Full-suite:
1049 passed, 5 pre-existing-unrelated failures. Phase-0 gate:
`test/space_equiv.py` (`run_space_gate`); self-check
`test/test_space_equiv_selfcheck.py` (2 passed).

---

## Scope & non-goals

- **In scope (2A):** tier table, P-tier retire, selector tensor, C/S
  executors, the read-only symbol snap at post-`sigma_percept` `y`.
- **Out of scope (Phase 2B, next plan):** rebuilding the retired chart→STM
  producer, the bounded soft shift-reduce over STM-7, the NULL-seal
  finalize, the metalbaby perf gate vs CKY+resize. The current chart path
  is dormant (`WordSpace.compose` called 0× in the CPU suite — proven in
  Phase 1.2), so 2A's selector/executor changes are gated by the
  representative + characterize gates here; the *live* exercise of these
  paths is Phase 2B's producer.
- **Gate mode:** behaviour-identical (bit-identical via Phase-0 anchor +
  representative suite 0-failed) where the path is dormant/unchanged;
  **characterize-and-approve** where a path's semantics change (selector,
  executors, snap). Never weaken a test. Fail loud on divergence.

## File Structure

| File | Responsibility in 2A |
|---|---|
| `bin/Spaces.py` | `WorkingState` gains the selection-tensor field; `ConceptualSpace.forward` C-executor + the post-`sigma_percept` symbol snap; read-only `symbolicSpace_ref`. |
| `bin/Language.py` | `Chart._RULE_TIER` rewrite; `SyntacticLayer` demoted to executor (selection-tensor-driven, no cursor on the carrier path); P-tier attach removed. |
| `bin/Models.py` | `object.__setattr__(cs,'symbolicSpace_ref',ss)` build-time attach (beside the existing `perceptualSpace_ref`). |
| `test/test_phase2a_labor_division.py` | New: unit-locks the tier table, the selection-tensor contract, executor single-writer, snap read-only-ness. |

---

### Task 2A.1: `WorkingState` gains the selection tensor (inert)

**Files:** Modify `bin/Spaces.py` (`WorkingState.__slots__` + `new_working_state`).

- [ ] **Step 1:** Add fields to `WorkingState.__slots__` (currently
  `("cursor","gen","recur_pass","cs_for_ps","cs_for_ss","valid_mask","errors")`):
  add `"op_sel"` (fixed-op-axis weight tensor, `[n_ops]`, float32) and
  `"op_operands"` (operand routing, `int64 [n_ops, 2]`). In
  `new_working_state(n_tiers, device)` add params `n_ops=0` and initialise:
```python
    ws.op_sel = (torch.zeros(n_ops, dtype=torch.float32, device=device)
                 if n_ops else None)
    ws.op_operands = (torch.zeros(n_ops, 2, dtype=torch.int64, device=device)
                      if n_ops else None)
```
  Keep the existing call sites working: `new_working_state` must remain
  callable as today (n_ops defaults 0 → fields None → inert).
- [ ] **Step 2 (gate, inert):** representative suite must stay **135
  passed/12 skipped/4 xfailed/0 failed**; `test/test_space_equiv_selfcheck.py`
  → 2 passed. (Adding unread `__slots__` fields + defaulted factory args
  cannot change behaviour.) Run:
```
.venv/bin/python -m pytest -q test/test_mm_xor.py test/test_mm_boolean.py test/test_universality.py test/test_invertibility.py test/test_discourse_space.py test/test_xor_grammar.py test/test_signal_router_xor_grammar.py test/test_perceptualspace_bpe_forward.py test/test_phase2_pipeline_primitives.py test/test_compiled_step_invoked.py test/test_brick_no_sync.py
.venv/bin/python -m pytest test/test_space_equiv_selfcheck.py -v
```
- [ ] **Step 3 (commit hand-off):** `Phase 2A.1: WorkingState gains inert fixed-op-axis selection tensor fields`.

### Task 2A.2: `_RULE_TIER` rewrite to the Phase-2 C/S split

**Files:** Modify `bin/Language.py` `Chart._RULE_TIER` (~2597-2622).

- [ ] **Step 1:** Read the current `_RULE_TIER` dict verbatim. Rewrite the
  mapping to exactly (per spec Phase-2):
  - **C** = `union, intersection, swap, copy, not, non, true, false, part,
    query, area, luminosity, equal`
  - **S** = `conjunction, disjunction, isEqual, isaPart, lift, lower`
  (`lift`/`lower` move P→**S**; **no rule maps to `P` anymore**.) Keep the
  dict's structure/format; only the tier values change. Update the adjacent
  tier-meaning docstring to the 2-tier (C/S) description.
- [ ] **Step 2:** Add a unit test `test/test_phase2a_labor_division.py`:
```python
def test_rule_tier_is_cs_split():
    from Language import Chart
    rt = Chart._RULE_TIER
    C = {"union","intersection","swap","copy","not","non","true","false",
         "part","query","area","luminosity","equal"}
    S = {"conjunction","disjunction","isEqual","isaPart","lift","lower"}
    for m in C: assert rt[m] == "C", (m, rt.get(m))
    for m in S: assert rt[m] == "S", (m, rt.get(m))
    assert "P" not in set(rt.values()), "P tier must be retired"
```
- [ ] **Step 3 (run, expect FAIL then PASS):**
  `.venv/bin/python -m pytest test/test_phase2a_labor_division.py::test_rule_tier_is_cs_split -v`
  (FAIL before Step 1's edit if written first; PASS after.)
- [ ] **Step 4 (characterize gate):** representative suite. `_tier_for_method`
  / `_tier_for_rule` consume `_RULE_TIER`; in default-only grammar configs
  the chart is dormant so this is inert → expect **0 failed** vs baseline.
  If a grammar test (`test_xor_grammar`, `test_signal_router_xor_grammar`,
  `test_phase2_pipeline_primitives`) changes, characterize it (name / mode
  / is it asserting the old P-tier placement of lift/lower → that is the
  intended owner-approved delta). Do NOT weaken.
- [ ] **Step 5 (commit hand-off):** `Phase 2A.2: rewrite _RULE_TIER to the C/S split (lift/lower -> S, P retired)`.

### Task 2A.3: Retire the P-tier `SyntacticLayer`

**Files:** Modify `bin/Language.py` — `WordSpace._attach_per_space_syntactic_layer`
P-tier attach (~4669-4680: the `if perceptualSpace is not None: ...
_attach_per_space_syntactic_layer(perceptualSpace, tier='P')` block) and the
tier-`'P'` branch of `_attach_per_space_syntactic_layer` (~5553-5570).

- [ ] **Step 1:** Stop attaching a `SyntacticLayer` to PerceptualSpace:
  remove the `_attach_per_space_syntactic_layer(perceptualSpace, tier='P')`
  call (keep `perceptualSpace.attach_wordSpace(self)` — that wiring is
  unrelated). Leave the tier-`'P'` code branch in place but unreached
  (dead-safe) OR delete it if trivially safe — read it; if other code
  references `perceptualSpace.syntacticLayer`, guard those reads
  (`getattr(...,'syntacticLayer',None)` is already the codebase idiom).
  `grep -n "syntacticLayer" bin/Spaces.py` to confirm `PerceptualSpace.forward`
  already treats a missing `syntacticLayer` as a no-op (it does:
  `if getattr(self,'syntacticLayer',None) is None:`-style guard).
- [ ] **Step 2:** Add to `test/test_phase2a_labor_division.py`:
```python
def test_perceptual_has_no_syntactic_layer():
    # Build a model from the gate config; PerceptualSpace must not own a
    # SyntacticLayer after the P-tier retire.
    import os; os.environ["BASICMODEL_DEVICE"] = "cpu"
    from test.space_equiv import _p
    from util import init_config, init_device
    from data import TheData
    from Models import BaseModel
    init_device("cpu")
    cfg = str(_p / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    assert getattr(m.perceptualSpace, "syntacticLayer", None) is None
```
- [ ] **Step 3 (run):** that test → PASS after Step 1.
- [ ] **Step 4 (gate):** representative suite 0-failed vs baseline +
  selfcheck 2 passed (the P-tier layer was a backward-compat no-op for
  default grammars — retiring it is behaviour-identical there; characterize
  any grammar-test delta).
- [ ] **Step 5 (commit hand-off):** `Phase 2A.3: retire the P-tier SyntacticLayer`.

### Task 2A.4: Wire the read-only `symbolicSpace_ref` (build-time)

**Files:** Modify `bin/Models.py` per-stage build loop (~3921-3930, where
each `cs`/`ss` pair is constructed; the existing
`object.__setattr__(ss,'perceptualSpace_ref',self.perceptualSpace)` is
~3981-3983 — read both).

- [ ] **Step 1:** In the per-stage build loop where `cs` and its paired
  `ss` are both in scope (`ss.conceptualSpace is cs`), add — mirroring the
  existing `perceptualSpace_ref` attach exactly (non-owning back-ref via
  `object.__setattr__`, NOT nn.Module registration; rationale documented at
  `Spaces.py:9004-9016`):
```python
            object.__setattr__(cs, 'symbolicSpace_ref', ss)
```
- [ ] **Step 2:** Add to `test/test_phase2a_labor_division.py` a check
  (reuse the model built in `test_perceptual_has_no_syntactic_layer` via a
  module fixture, or rebuild) asserting
  `m.conceptualSpace.symbolicSpace_ref is m.symbolicSpace` and that it is
  NOT in `m.conceptualSpace._modules` (no nn.Module registration):
```python
def test_symbolicspace_ref_is_readonly_backref(built_model):
    cs, ss = built_model.conceptualSpace, built_model.symbolicSpace
    assert cs.symbolicSpace_ref is ss
    assert 'symbolicSpace_ref' not in cs._modules
```
- [ ] **Step 3 (gate):** representative suite 0-failed (adding an unread
  back-ref is inert) + selfcheck 2 passed.
- [ ] **Step 4 (commit hand-off):** `Phase 2A.4: read-only symbolicSpace_ref back-ref on ConceptualSpace`.

### Task 2A.5: Read-only symbol snap at post-`sigma_percept` `y`

**Files:** Modify `bin/Spaces.py` `ConceptualSpace.forward` — the
post-`sigma_percept` / pre-concept-codebook point where `y` is the
`[·, CS_dim]` concept tensor (read the body: after
`y = self.sigma_percept(x)` / the `syntacticLayer` branch produces `y`,
BEFORE the concept codebook step ~8872-8895).

- [ ] **Step 1:** Insert a read-only nearest-neighbour snap of `y` against
  the learnable symbol codebook, straight-through, no write to that
  codebook (single-writer = SymbolicSpace). Use the solved sub-problems
  from the 1A.3 investigation: read `P = self.symbolicSpace_ref.subspace.what.W`
  **directly** (read-only; NOT `.getW()`, NOT `Codebook.forward`):
```python
        # Phase 2A: snap the post-sigma_percept concept tensor to the
        # learnable symbol codebook (read-only; SymbolicSpace single-writer).
        ssr = getattr(self, 'symbolicSpace_ref', None)
        what = getattr(ssr.subspace, 'what', None) if ssr is not None else None
        P = getattr(what, 'W', None) if what is not None else None
        if (P is not None and torch.is_tensor(P) and P.dim() == 2
                and y.dim() == 3 and y.shape[-1] == P.shape[-1]):
            d = torch.cdist(y.reshape(-1, y.shape[-1]).float(),
                            P.float())                       # [B*C, S]
            idx = d.argmin(dim=-1)                            # [B*C]
            snapped = P.index_select(0, idx).view_as(y).to(y.dtype)
            y = y + (snapped - y).detach()                    # STE, read-only
```
  Robust no-op when shapes/refs are unavailable (mirrors the 1A.1/1A.2
  helper guard style). The snap MUST be read-only — never call `.insert`/
  `.setW`/`.forward`(EMA) on the symbol codebook from ConceptualSpace.
- [ ] **Step 2 (idempotency — must not regress):** the snap is read-only ⇒
  `ConceptualSpace.forward` must not gain NEW non-idempotency. Run:
```
.venv/bin/python -c "
import Spaces
from test.space_equiv import run_space_gate, default_snapshot
try:
    print('CS', run_space_gate(Spaces.ConceptualSpace,'forward',candidate_fn=Spaces.ConceptualSpace.forward,snapshot=default_snapshot,max_batches=2))
except AssertionError as e:
    print('CS-DIVERGENCE', e)
"
```
  PASS criterion: divergence profile UNCHANGED from the known pre-existing
  baseline (`call#2`, `max|Δ|≈6.3e-3`, localized to `_cursor_compose`/
  `valid_mask` — owner-accepted, NOT this task's concern). Any NEW
  divergence, or any mutation of `subspace.what`, is a 2A.5 bug → fix
  (snap must be read-only). Report before/after profile.
- [ ] **Step 3 (single-writer grep):** `grep -n "symbolicSpace_ref\|subspace\.what"`
  within `ConceptualSpace.forward` — prove only `.W` reads, no
  `.insert/.setW/.forward(EMA)/.replace`.
- [ ] **Step 4 (characterize gate):** representative suite — additive
  semantic change; NO crashes; characterize every status change (name /
  mode / tied to the new snap / owner-approved delta). selfcheck 2 passed.
- [ ] **Step 5 (commit hand-off):** `Phase 2A.5: read-only symbol snap of post-sigma_percept y (1A.3, locus resolved)`.

### Task 2A.6: Tensorized selector-in-S + C/S executor demotion

**Files:** Modify `bin/Language.py` `SyntacticLayer` (`_next_rule_name`
~4007-4053, `forward`/`reverse` ~4080-4163) and `bin/Spaces.py`
`SymbolicSpace.forward` (where `current_rules['S']` drives the per-step
`syntacticLayer.forward` loop ~10240-10256) + `ConceptualSpace.forward`
(C-tier `syntacticLayer` branch ~8807-8818).

> This is the structural heart. The chart path is **dormant**
> (`WordSpace.compose` 0× in the CPU suite, proven in Phase 1.2), so the
> *selection source* (chart) is rebuilt in Phase 2B; **2A.6 establishes the
> selection-tensor *contract* and the executor split** so the gate stays
> behaviour-identical now and Phase 2B only has to populate the tensor.

- [ ] **Step 1:** Define the selection-tensor contract on `WorkingState`
  (fields from 2A.1): `op_sel[n_ops]` = soft weights over the fixed grammar
  op axis (order = `TheGrammar.rules` index order; document it),
  `op_operands[n_ops,2]` = operand routing. Add a `WordSpace` helper
  `selection_from_current_rules(tier, work)` that, **when `work.op_sel` is
  None (the dormant/eager path — today's reality)**, returns the existing
  `current_rules[tier]` per-step list UNCHANGED (byte-identical eager
  fallback), and when `op_sel` is present yields the tensor-driven
  selection. This keeps 2A behaviour-identical (op_sel is None until 2B
  populates it) while locking the interface.
- [ ] **Step 2 (C/S executor):** In `SyntacticLayer.forward`, when
  `ws._work` is not None AND `ws._work.op_sel` is not None, execute the
  selected op(s) for THIS layer's tier by reading the read-only selection
  tensor and applying each op's contribution on its own slot of a fixed op
  axis, combined by a **single weighted reduce** (no shared in-place
  accumulator — the proven `_embed_bpe` pattern):
```python
        # executor: independent per-op contributions -> one weighted reduce
        contribs = []                       # fixed op axis
        for k, op in enumerate(self._tier_ops()):     # tier's ops, fixed order
            contribs.append(op_apply(op, x))          # no shared accumulator
        stacked = torch.stack(contribs, dim=0)        # [n_tier_ops, ...]
        w = work.op_sel[self._tier_op_slice()].view(-1, *([1]*(stacked.dim()-1)))
        y = (w * stacked).sum(dim=0)                  # single weighted reduce
```
  When `op_sel` is None → the existing cursor/`current_rules` path
  (byte-identical eager fallback from Phase 1.2; unchanged). C-tier
  executor runs in ConceptualSpace on the post-`sigma_percept` `y`
  (single-writer concept codebook); S-tier in SymbolicSpace on the symbol
  codebook. Read the actual op-application call (`layer.forward`/`compose`)
  and `_by_name` to implement `op_apply`/`_tier_ops`/`_tier_op_slice`
  faithfully — keep the eager path EXACTLY as Phase 1.2 left it.
- [ ] **Step 3 (selector-in-S stub):** In `SymbolicSpace.forward`, add the
  *call site* that would compute and write `work.op_sel`/`op_operands` from
  the chart selection — but gated `if work is not None and <chart active>`;
  since the chart is dormant in 2A this is unreached (the populate logic is
  Phase 2B). Add a clear `# Phase 2B: populate work.op_sel from the chart
  soft-superposition here` marker at the exact site. This locks WHERE
  selection is centralized (S) without changing dormant behaviour.
- [ ] **Step 4 (behaviour-identical gate — op_sel is None in 2A):**
  representative suite MUST be **0 failed** vs baseline (the tensor path is
  unreached; only the byte-identical eager fallback runs). selfcheck 2
  passed. If any test changes, the eager fallback is not byte-identical —
  fix it (do not weaken).
- [ ] **Step 5 (contract unit test):** in `test/test_phase2a_labor_division.py`
  add a test constructing a `WorkingState` with a hand-set `op_sel`
  (one-hot) and asserting `SyntacticLayer.forward` selects exactly that op
  via the weighted-reduce path (proves the executor contract without the
  chart), and `op_sel=None` reproduces the cursor path.
- [ ] **Step 6 (commit hand-off):** `Phase 2A.6: selection-tensor contract + C/S executor split (eager fallback byte-identical; selector populate deferred to 2B)`.

### Task 2A.7: Phase 2A acceptance

- [ ] **Step 1:** Full representative suite + `test_phase2a_labor_division.py`
  + `test_perceptual_loopback.py` + selfcheck — all 0-failed vs the Phase-1
  baseline (characterize any owner-approved grammar-delta from 2A.2/2A.5).
- [ ] **Step 2:** Faithful fullgraph gate (`test/brick_recon.py` MM_xor/
  MM_5M/MM_grammar) — still **FULLGRAPH-CLEAN** (no new break from the
  selector/executor/snap). Report calls_captured/unique_graphs.
- [ ] **Step 3:** Idempotency unchanged: PS/SS `run_space_gate` forward
  identity still clean; CS divergence profile unchanged from the
  owner-accepted pre-existing baseline.
- [ ] **Step 4 (hand-off summary):** report what is behaviour-identical vs
  characterized-delta; confirm 2A leaves the system ready for Phase 2B
  (producer rebuild + STM-7 shift-reduce) — the selection tensor + executor
  split are in place, populate-from-chart is the single remaining wire.
- [ ] **Step 5 (commit hand-off):** `Phase 2A: labor division complete (selector contract + C/S executors + symbol snap); 2B = producer + STM-7 shift-reduce`.

---

## Self-Review (against the spec)

- **Spec coverage:** Phase 1-D §1 labor division → 2A.2 (tier), 2A.3
  (P-retire), 2A.6 (selector contract + executor split); constraint 1
  (single-writer) → 2A.5/2A.6 (C ops on concept codebook, S on symbol; snap
  read-only); constraint 2 (selection vs execution separated) → 2A.6;
  constraint 3 (independent contributions → single weighted reduce, no
  shared accumulator) → 2A.6 Step 2; constraint 4 (selection as subspace
  tensor data, no cursor/branch) → 2A.1 + 2A.6 (op_sel tensor; cursor only
  on the dormant eager fallback). VQ-EMA-decision 1A.3 + locus → 2A.4
  (ref) + 2A.5 (snap at post-`sigma_percept` `y`).
- **Deliberate decomposition:** the *selection source* (chart soft-
  superposition) and STM-7 shift-reduce producer are **Phase 2B** — 2A
  builds the contract + executors so 2B only populates `op_sel`. This is
  the scope-check decomposition (each plan ships independently-testable
  software); recorded so it is not a gap.
- **Placeholder scan:** the only "deferred" items (2A.6 Step 3 populate;
  Phase 2B) are explicitly scoped follow-ons with the marker at the exact
  code site, not vague TODOs — the spec itself stages them.
- **Type consistency:** `op_sel`/`op_operands` (2A.1) used consistently in
  2A.6; `symbolicSpace_ref` (2A.4) used in 2A.5; `selection_from_current_rules`
  / `_tier_ops` / `_tier_op_slice` / `op_apply` defined in 2A.6 Step 1-2.

## Risks

- 2A.6 eager fallback MUST stay byte-identical to Phase 1.2 (op_sel is None
  until 2B) — the representative suite 0-failed is the contract; any delta
  is a real bug, not a tolerance to relax.
- The snap (2A.5) is additive + characterized; it is read-only — the
  idempotency profile check (Step 2) is the guard against accidentally
  writing the symbol codebook.
- Phase 2B (producer rebuild + bounded shift-reduce + metalbaby perf gate
  vs CKY+resize) is authored just-in-time after 2A lands and the
  selector/executor interfaces are concrete; its design is the spec's
  Phase-1-D §3 (already ratified).
