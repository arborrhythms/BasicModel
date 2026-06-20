# Tier-Free Bounded-STM Grammar Fold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-word *full re-parse* with a capacity-driven structured fold over a bounded STM, and delete the P/C/S tier machinery — keeping per-word online word learning.

**Architecture:** Per-word ingestion (P learns each word, pushes to an 8-slot STM); on overflow, one structured reduce frees a slot; a sentence-end sweep collapses to root(s). SS soft-scores each reduce (gradient), CS commits the discrete non-interfering tiling (the parse). Single reduction tier; `lift`/`lower` become SS codebook round-trips.

**Tech Stack:** PyTorch 2.11, custom layers in `bin/`, pytest. Spec: `doc/specs/2026-06-05-parallel-whole-slab-grammar-fold.md`.

---

## Working agreement (read first)

- **TDD:** write the failing test, watch it fail, implement minimally, watch it pass.
- **Test runner:** `BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest <nodeid> -q`. Run **targeted** node IDs, not the full suite. `test/test_basicmodel.py` fails ~24 on clean HEAD and is order-fragile — ignore unrelated reds.
- **Compile gate:** use `MODEL_COMPILE=eager` (Dynamo trace path; inductor's C++ build is separately broken by the space in this repo path). `BASIC_FULLGRAPH=0` enumerates all breaks; `=1` (default) raises on the first.
- **Bounded training probe:** `BASIC_NUM_EPOCHS=N` overrides the XML epoch count for quick learning-curve checks.
- **Git is the user's.** Every "Commit" step is a **checkpoint**: stop, summarize the diff, let the user commit. Do not run `git add/commit/checkout`.
- **Later-phase detail:** Tasks 1–4 (revert + Phase 1) are fully specified. Tasks 5–10 (Phases 2–5) give exact files, deletion targets, and the test that defines "done"; the precise replacement code is finalized at execution time after re-reading the named functions — these are refactors of large, evolving files where inventing code ahead of the read would be guesswork. Expand each to step-level code immediately before executing it.

---

## File map

| File | Role in this work |
|---|---|
| `bin/Language.py` | `LanguageLayer.compose` (tier loop, reduce rounds); `BinaryStructuredReductionLayer` (tier mask, soft/hard, lift/lower); revert target for the `want_hard` experiment |
| `bin/Models.py` | `_forward_body_per_word` / `_per_word_body_step` (per-word loop, the compose fire to delete, the existing capacity back-pressure); `_stm_bounded_reduce_step`, `_stm_reduce_to_single_S` (the reduce primitives — kept) |
| `bin/Spaces.py` | `ConceptualSpace.forward` (STM push), `SymbolicSpace.forward` (SS); `_stm_shift_and_push` (per-word push — kept) |
| `test/test_ir_fullgraph_compile.py` | fullgraph compile gate (has the serial-path test) |
| `test/test_bounded_stm_fold.py` | **new** — capacity invariant + fold-correctness gates |
| `test/test_cs_stm_bookkeeping.py`, `test/test_conceptual_stm.py`, `test/test_grammar_*.py` | regression gates (STM + grammar) |

---

## Task 1: Revert the soft-`want_hard` experiment (clean base)

The spec keeps the fold **discrete**, so the in-flight pure-soft edits in `bin/Language.py` are removed before the refactor. (The STM tensorizations and the new fullgraph test from this session are **kept**.)

**Files:**
- Modify: `bin/Language.py` — `BinaryStructuredReductionLayer.forward` and `LanguageLayer.compose` / `_compose_rules_from_routings`

- [ ] **Step 1: Confirm the experimental hunks are present**

Run: `BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -c "import Language; print('want_hard' in Language.BinaryStructuredReductionLayer.forward.__doc__ or True)"`
Then `grep -n "want_hard" bin/Language.py` — expect matches in `BinaryStructuredReductionLayer.forward`.

- [ ] **Step 2: Restore the pre-experiment code**

Revert each hunk so `forward` again computes `hard = binary_tiling_viterbi(...)` unconditionally, `compact_hard` runs unconditionally, the `routing` dict always carries the hard meta, the return is `return hard_slab, soft_slab, routing`, and `compose` reads `b_routing["action_kind"]` (not `.get`). The hunks to revert are the five `want_hard`-gated regions and the two `.get("action_kind")` guards. Apply this as **targeted edits to the `want_hard` hunks only** — do **NOT** `git checkout bin/Language.py`: the working tree carries unrelated uncommitted changes to that file that a checkout would destroy. `git diff bin/Language.py` first to see exactly which hunks are the experiment.

- [ ] **Step 3: Verify grammar + STM regressions pass**

Run: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin .venv/bin/python -m pytest test/test_grammar_binary_ops.py test/test_grammar_rewrite.py test/test_cs_stm_bookkeeping.py -q`
Expected: PASS (no `want_hard` paths remain).

- [ ] **Step 4: Checkpoint** — clean base restored; user commits ("revert: drop pure-soft want_hard experiment").

---

## Phase 1 — Remove the per-word re-parse; rely on capacity back-pressure + end sweep

The per-word `wordSubSpace.compose(_snap)` fire ([Models.py:6112-6130](../../bin/Models.py)) is the $\approx 89\%$ cost. The capacity back-pressure reduce **already exists** ([Models.py:5936-5940](../../bin/Models.py): `if stm._max_depth_host >= stm.capacity: self._stm_bounded_reduce_step(); ...`) and the sentence-end sweep already exists (`_stm_reduce_to_single_S`). Phase 1 deletes the per-word fire and verifies the fold still forms from those two.

### Task 2: Capacity-invariant gate (new test)

**Files:**
- Create: `test/test_bounded_stm_fold.py`

- [ ] **Step 1: Write the failing test**

```python
"""Bounded-STM fold gates: capacity invariant + per-word ingestion."""
import os, sys
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import torch, warnings
import Models, Language
from util import init_config

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_PROJECT, "data")

def _model():
    init_config(path=os.path.join(_DATA, "MM_grammar.xml"),
                defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(os.path.join(_DATA, "MM_grammar.xml"))
    Models.TheData.load("xor")
    return m

def test_stm_never_exceeds_cap_after_forward():
    m = _model(); m.train()
    cap = int(m.conceptualSpace.stm.capacity)
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    depth = m.conceptualSpace.stm._depth
    assert int(depth.max().item()) <= cap, f"STM depth {int(depth.max())} > cap {cap}"
```

- [ ] **Step 2: Run it on current code**

Run: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin .venv/bin/python -m pytest test/test_bounded_stm_fold.py::test_stm_never_exceeds_cap_after_forward -q`
Expected: PASS today (back-pressure already bounds depth) — this is the *invariant we must not break*. If it fails, fix the harness before proceeding.

- [ ] **Step 3: Checkpoint** — user commits ("test: STM capacity invariant gate").

### Task 3: Delete the per-word compose fire

**Files:**
- Modify: `bin/Models.py:6112-6130` (the `router_wire_serial in ("per-word","both")` block that calls `self.wordSubSpace.compose(_snap)`)

- [ ] **Step 1: Capture the baseline learning curve**

Run: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none BASIC_NUM_EPOCHS=60 PYTHONPATH=bin /usr/bin/time -p .venv/bin/python bin/Models.py data/MM_grammar.xml 2>&1 | grep -E "reconstruction|^real"`
Record final reconstruction + wall (expect plateau ~0.25, ~26s).

- [ ] **Step 2: Delete the per-word fire**

Remove the entire `if (active_host and getattr(self, 'router_wire_serial', 'both') in ("per-word","both") ...): _snap = ...; self.wordSubSpace.compose(_snap)` block at [Models.py:6112-6130](../../bin/Models.py). Leave the immediately-following `CS_sub, idea_bd = self._per_word_body_step(...)` call intact.

- [ ] **Step 3: Run the capacity invariant + a short train**

Run: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin .venv/bin/python -m pytest test/test_bounded_stm_fold.py -q`
Then: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none BASIC_NUM_EPOCHS=60 PYTHONPATH=bin /usr/bin/time -p .venv/bin/python bin/Models.py data/MM_grammar.xml 2>&1 | grep -E "reconstruction|^real"`
Expected: invariant PASS; wall drops sharply (compose gone); reconstruction **at least as good** as baseline (target: descends below the ~0.25 plateau, toward ~0.11 — the no-per-word-routing curve).

- [ ] **Step 4: Run STM + grammar regressions**

Run: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin .venv/bin/python -m pytest test/test_cs_stm_bookkeeping.py test/test_conceptual_stm.py test/test_grammar_binary_ops.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — user commits ("perf: drop per-word re-parse; fold via capacity back-pressure + end sweep").

### Task 4: Confirm the sweep still reaches root(s)

**Files:**
- Modify: `test/test_bounded_stm_fold.py` (add a fold-reaches-root assertion)

- [ ] **Step 1: Add the test**

```python
def test_sentence_end_reduces_toward_root():
    m = _model(); m.train()
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m.forward(x)
    S, post_depth = m._stm_reduce_to_single_S()
    assert int(post_depth.max().item()) <= max(1, 3), "absolute rows must collapse near root"
    assert torch.isfinite(S).all(), "root state must be finite"
```

- [ ] **Step 2: Run it**

Run: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin .venv/bin/python -m pytest test/test_bounded_stm_fold.py::test_sentence_end_reduces_toward_root -q`
Expected: PASS.

- [ ] **Step 3: Checkpoint** — user commits ("test: sentence-end sweep reaches root").

---

## Phase 2 — Tier elimination

Collapse C/S to one reduction tier; delete the tier mask + propagation; re-express `lift`/`lower` as SS codebook round-trips. **Expand these to step-level code after re-reading the named functions.**

### Task 5: Delete `position_tier` / tier mask in `BinaryStructuredReductionLayer`

**Files:**
- Modify: `bin/Language.py` — `BinaryStructuredReductionLayer.__init__` ([5605-5628](../../bin/Language.py)), `forward` ([5736-5757](../../bin/Language.py) tier mask, [5853-5868](../../bin/Language.py) `next_position_tier`)

- [ ] **Step 1 (read):** Read `BinaryStructuredReductionLayer.forward` end-to-end and every caller passing `position_tier` (`grep -n position_tier bin/Language.py`).
- [ ] **Step 2 (test):** Add to `test/test_bounded_stm_fold.py`:

```python
def test_binary_reducer_is_tier_free():
    import inspect, Language
    src = inspect.getsource(Language.BinaryStructuredReductionLayer)
    assert "op_tier_idx" not in src and "position_tier" not in src, "tier machinery must be gone"
```

- [ ] **Step 3:** Run it — Expected: FAIL.
- [ ] **Step 4 (implement):** Remove `op_tier_idx`/`op_tier_delta`/`position_tier` buffers and the tier-mask + `next_position_tier` blocks; drop `position_tier` from the signature; the tier mask becomes all-legal (no masking).
- [ ] **Step 5:** Run `test_binary_reducer_is_tier_free` + `test/test_grammar_binary_ops.py` — Expected: PASS.
- [ ] **Step 6: Checkpoint** — user commits ("refactor: remove C/S tier mask + propagation").

### Task 6: Collapse the per-tier `compose` loop to a single tier

**Files:**
- Modify: `bin/Language.py` — `LanguageLayer.compose` ([4184-4262](../../bin/Language.py)); grammar tier assignment ([1337](../../bin/Language.py))

- [ ] **Step 1 (read):** Read `compose` and `_reassign_tiers_from_layer_classes`; list where `r.tier == 'S'` / `'C'` is consulted (start symbol [1464](../../bin/Language.py), relative rules).
- [ ] **Step 2 (test):** Add a test asserting `compose` produces the same root state for an absolute sentence before/after collapse (capture `_last_root_state` on a fixed input pre-change as the golden value).
- [ ] **Step 3:** Run it — Expected: FAIL (or set up golden).
- [ ] **Step 4 (implement):** Run reduction in one tier (C); keep the start-symbol and relative-rule (`isEqual`/`isPart`) semantics that lived at S (verify against §13 risk). Set `max_rounds = max(conceptualOrder, x.shape[1] - 1)` at [4226](../../bin/Language.py).
- [ ] **Step 5:** Run the golden test + `test/test_grammar_rewrite.py test/test_grammar_identity_rule.py` — Expected: PASS.
- [ ] **Step 6: Checkpoint** — user commits ("refactor: single-tier compose; loop bound max(order,N-1)").

### Task 7: Re-express `lift`/`lower` as SS codebook round-trips

**Files:**
- Modify: `bin/Language.py` (lift/lower op definitions, ~[2610-2825](../../bin/Language.py)); `bin/Spaces.py` `SymbolicSpace` codebook

- [ ] **Step 1 (read):** Read the `lift`/`lower` op classes and the SS codebook quantize/dequantize path. Identify the current C$\leftrightarrow$S effect.
- [ ] **Step 2 (test):** Add a faithfulness test: a known relative sentence (a grammar with `isEqual`/`isPart`) yields the same predicate/idea1/idea2 end-state as the pre-refactor path (golden capture).
- [ ] **Step 3:** Run it — Expected: FAIL.
- [ ] **Step 4 (implement):** Make `lift`/`lower` invoke the SS codebook round-trip as their op effect (no tier delta).
- [ ] **Step 5:** Run the faithfulness test + `test/test_grammar_preposition.py` (relative-rule coverage) — Expected: PASS.
- [ ] **Step 6: Checkpoint** — user commits ("refactor: lift/lower as SS codebook round-trip").

---

## Phase 3 — SS / CS split

### Task 8: Move soft scoring into `SS.forward`, the discrete fold into `CS.forward`

**Files:**
- Modify: `bin/Spaces.py` — `SymbolicSpace.forward`, `ConceptualSpace.forward`; `bin/Language.py` — `compose` (becomes the SS-side scorer)

- [ ] **Step 1 (read):** Read `SymbolicSpace.forward` and `ConceptualSpace.forward` and trace how `compose`'s scores/fold are consumed today.
- [ ] **Step 2 (test):** Assert `SS.forward` returns the soft-score dict (no committed fold) and `CS.forward` performs the reduce+push; check arity is unchanged (`test/test_cs_stm_bookkeeping.py::TestCSForwardArity` style).
- [ ] **Step 3:** Run it — Expected: FAIL.
- [ ] **Step 4 (implement):** Relocate scoring (soft DP) to SS; CS commits the non-interfering tiling + push. Straight-through preserved.
- [ ] **Step 5:** Run STM bookkeeping + the capacity invariant — Expected: PASS.
- [ ] **Step 6: Checkpoint** — user commits ("refactor: SS scores, CS folds").

---

## Phase 4 — STM capacity parameter

### Task 9: `cap in {8, N}`, shared path, default 8

**Files:**
- Modify: `bin/Spaces.py` STM (capacity source); `bin/Models.py` (force-to-fit conditional)

- [ ] **Step 1 (read):** Read `stm.capacity` source and `_stm_bounded_reduce_step` trigger.
- [ ] **Step 2 (test):**

```python
def test_cap_equivalence_short_sentence():
    # For N <= 8, cap=8 and cap=N give identical STM end-states.
    # (parameterize via the XML <stmCapacity>/<wMax>; assert equal buffers)
    ...
```

(Finalize once the capacity knob path is read.)

- [ ] **Step 3:** Run it — Expected: FAIL.
- [ ] **Step 4 (implement):** Capacity becomes a parameter; force-to-fit reduce is a no-op when `cap >= N`. Default 8.
- [ ] **Step 5:** Run equivalence + capacity invariant — Expected: PASS.
- [ ] **Step 6: Checkpoint** — user commits ("feat: STM capacity parameter, default 8").

---

## Phase 5 — Re-confirm fullgraph + learning

### Task 10: Compile gate + learning verification

**Files:**
- Modify: `test/test_ir_fullgraph_compile.py` (the serial-path test; flip xfail to expected-pass if the captured forward is now clean)

- [ ] **Step 1:** Run the fullgraph gate: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager BASIC_NUM_EPOCHS=1 PYTHONPATH=bin .venv/bin/python bin/Models.py data/MM_grammar.xml` — inspect for `torch._dynamo.exc` / `Graph break`.
- [ ] **Step 2:** If clean, remove the `xfail` marker on `test_serial_per_word_forward_compiles_fullgraph_eager` and run it — Expected: PASS. If a break remains, enumerate with `BASIC_FULLGRAPH=0 TORCH_LOGS=graph_breaks` and address (likely a residual host op surfaced by the refactor).
- [ ] **Step 3:** Learning curve: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none BASIC_NUM_EPOCHS=200 BASIC_RUN_TEST="" PYTHONPATH=bin .venv/bin/python bin/Models.py data/MM_grammar.xml 2>&1 | grep -E "reconstruction|Accuracy"` — Expected: reconstruction descends below the ~0.25 plateau; no regression vs Phase 1.
- [ ] **Step 4:** Full targeted regression: `BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONPATH=bin .venv/bin/python -m pytest test/test_bounded_stm_fold.py test/test_cs_stm_bookkeeping.py test/test_conceptual_stm.py test/test_grammar_binary_ops.py test/test_grammar_rewrite.py -q` — Expected: PASS.
- [ ] **Step 5: Checkpoint** — user commits ("perf+refactor: tier-free bounded-STM fold complete").

---

## Self-review (against the spec)

- **§2-§5 (per-word ingestion, capacity reduce, loop bound):** Tasks 2-4 (delete re-parse, rely on back-pressure + sweep) and Task 6 (loop bound). ✓
- **§6 (tier elimination):** Tasks 5-7. ✓
- **§7 (cap parameter):** Task 9. ✓
- **§8 (gradient/causality):** preserved by keeping the discrete fold (Task 1 revert) + straight-through (Task 8). ✓
- **§9 deletions:** per-word fire (Task 3), tier machinery (Task 5-6), want_hard (Task 1). ✓
- **§10 keeps:** per-word loop + `_stm_shift_and_push` (untouched), reduce primitives (used by Tasks 3-4). ✓
- **§11 testing:** capacity invariant (Task 2), online learning (implicit — loop kept), compile gate + learning (Task 10). ✓
- **§13 risks:** lift/lower faithfulness (Task 7 golden test), start-symbol/relative-rules (Task 6), reduce granularity (open — decide during Task 3/9 A/B).

**Note on detail:** Tasks 1-4 are step-complete. Tasks 5-10 specify files, deletion targets, and the defining test; their replacement code is finalized at execution after the read steps (large evolving files — see Working agreement).
