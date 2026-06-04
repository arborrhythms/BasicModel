# Contextual BIND, PREPOSITION, and `.when` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Save location:** `doc/plans/2026-06-03-contextual-bind-preposition-when.md` (mirrors the parent spec `../doc/plans/2026-06-03-contextual-bind-preposition-when.md`).
>
> **Git:** Commit steps mark logical checkpoints. Per repo convention **Alec performs all git writes** — the implementer should pause at each commit point rather than running `git` directly.

**Goal:** Add `PREPOSITION(P,X)`, contextual `BIND` (wired into the live parse), a zero-centered signed `.when` range encoding that mirrors `.where` in **2 dims**, table-driven tense/aspect surface normalization, and a morphological lemma↔surface bridge that keeps the SS codebook lemma-only, in `basicmodel` as `GrammarLayer`-derived operations plus a quadrature `Encoding` subclass and a PS↔SS boundary transform — without reintroducing declared POS categories.

**Architecture:** Four parameter-free `GrammarLayer` subclasses (`PrepositionLayer`, `ContextualBindLayer`, `TenseLayer`, `AspectLayer`) in `bin/Language.py`, registered in `GRAMMAR_LAYER_CLASSES` + `_OPERATOR_SURFACE_SCHEMAS`, fired automatically by role-only rules in `data/role_collapsed.grammar` through the existing `_wire_signal_router_grammar_ops` $\to$ `_resolve_rule_layer` path. Contextual BIND resolves **at parse time, wired into the live fold**: `BinaryStructuredReductionLayer.forward` stashes the current constituent slab onto the BIND op before applying it, so `ContextualBindLayer` resolves a missing NP against the constructed left-context. The `.when` work replaces `WhenEncoding` (`bin/Spaces.py`) with a **2-dim endpoint-sum** range key mirroring `EndpointSumWhere` (`bin/perceptual_analyzer.py`), made signed/zero-centered. A pure surface-rewrite table (`bin/surface_tense.py`) maps morphology onto `PRESENT/PAST/FUTURE` $\times$ `SIMPLE/PERFECT/PROGRESSIVE`. A `MorphologyLayer` (`bin/morphology.py`) factors single-word inflection (eats↔eat+3sg) at the PS↔SS edge — keeping the SS codebook lemma-only and reconstruction exact via the `.where`-keyed PS codebook + recorded marker route, with agreement carried on the learned-participation (collapsed-POS) marker space.

**Tech Stack:** Python 3, PyTorch, pytest (`.venv/bin/python -m pytest`), XML grammar/config, the project's `GrammarLayer` / `Encoding` / `SubSpace` / `BinaryStructuredReductionLayer` substrate.

---

## Context — why this change

The parent spec (`../doc/plans/2026-06-03-contextual-bind-preposition-when.md`, status *proposal*) asks for the **minimum grammar machinery** for three capabilities the role-collapsed grammar cannot yet express:

1. **Clausal participation** — embedding a clause/phrase under a marker (`I think that Alice left`, `Alice wants to run`, `in the room`). Today function words like `that` / `to` / `in` are only learnable as absorbed surface markers; there is no operation that packages a marker with a phrase so the result can fill another operator's role.
2. **Implicit argument recovery** — control constructions where an NP is missing on the surface but recoverable from an already-built participant (`Alice wants to run` $\to$ Alice runs; `Alice persuaded Bob to run` $\to$ Bob runs). No operation resolves a missing NP against constructed participants.
3. **Tense/aspect normalization** — `ran`, `is running`, `has run`, `had been running` must normalize onto a temporal subspace. A `WhenEncoding` exists but is a single monotonic *counter*, not the zero-centered range the spec needs.

The hard constraint (spec "Design Constraints"): **do not reintroduce a global POS inventory.** Markers stay learned-and-owned by operators (absorb/emit); the new ops add *operator roles*, not parts of speech. This plan honors that — every addition is a role-only grammar rule plus a `GrammarLayer`, and the role-collapse forbidden-token gate (`test/test_role_collapsed_grammar.py`) must stay green.

**Intended outcome:** the 8 focused fixtures from the spec parse and round-trip structurally, `.when` round-trips signed times and rotates under tense, and full-English coverage can be re-evaluated without declared POS categories. This is a *focused, pre-corpus* milestone (spec Phase 5), not a training run.

---

## Decisions locked (from user + spec open questions)

| Question | Decision |
|---|---|
| Scope | **One plan, all four** additions, following the spec's 6 phases. |
| BIND resolution (spec open Q2) | **Resolve at parse time, wired into the live fold.** `BinaryStructuredReductionLayer.forward` stashes the live constituent slab on the BIND op; `ContextualBindLayer` resolves the missing NP against it (nearest-left constituent, locality). The `bind_resolver` ranking (licensing: `want`=subject-control, `persuade`=object-control, + participation) is the unit-tested core and the documented refinement path. |
| PREPOSITION arg gating (spec open Q1) | **Permissive** — accept NP/VP/S now; per-marker gating is a learned *hook* (docstring), not built. |
| `will` (spec open Q3) | `FUTURE(run)` now; `MODAL(will, X)` noted as a hook, not built. |
| `.when` encoding (spec open Q4) | **Mirror `.where` in 2 dims (`nWhen=2`, single frequency):** a zero-centered, **signed** endpoint-sum key `q(start)+q(end)` (angle=center time, magnitude=duration), like `EndpointSumWhere`. **Supersedes the spec's per-endpoint constant-norm multi-frequency text** (user directive): the magnitude carries duration exactly as `.where` carries span length, and absolute time (past/future) does *not* drag the norm — only duration does. |
| Tense/aspect placement (spec open Q5) | Operate on the VP/event `.when` subspace **before** the subject LIFT (C-tier); equivalence to post-LIFT noted. |
| Deliverable | TDD task-by-task, concrete code, exact paths + pytest node IDs. |

---

## Grounding (verified against the code)

- **Auto-wiring**: `_wire_signal_router_grammar_ops` (`bin/Language.py:8921`) iterates `TheGrammar.rules_upward`, resolves a layer via `_resolve_rule_layer(tier, ...)` (`bin/Language.py:9000`, instantiates with `cls()` and returns `None` on `TypeError`), and wraps binary ops in `_BinaryGrammarOpAdapter` (`bin/Language.py:3660`, calls `gl.compose(left,right)`). **Consequence:** a new op fires automatically once it is (a) in `GRAMMAR_LAYER_CLASSES`, (b) constructible with no required args, and (c) named by a `<rule>`.
- **Binary reduction**: `BinaryStructuredReductionLayer` (`bin/Language.py:5139`) is the fold. `_stacked_reduced` (`:5221`) applies **every** op to **all** adjacent pairs at once (`[op(left, right) for op in self.ops]` on `left=x[:, :-1]`, `right=x[:, 1:]`), then Viterbi routing selects one op per site. `LanguageLayer.compose` (`:3810`) calls `binary_layer(x, position_tier=...)` once per round, folding `[B, N, D]` down to the root state `x[:, 0:1, :]`. **This is the BIND wiring point:** `forward` has the full live slab `x` in scope (`:5284`), just before `_stacked_reduced` at `:5300`.
- **Base class**: `GrammarLayer` (`bin/Layers.py:1611`) — class attrs `rule_name`, `arity`, `invertible`, `lossy`, `tier` (`'S'/'C'/'P'/'L'`), `reads_activation`, `surface_schema`; override `forward`/`reverse`; `absorb`/`bind_marker`/`bound_markers` for learned markers. Surface templates T1–T5 at `bin/Layers.py:1546`. Mirror `ConjunctionLayer.__init__(self, nInput=0, nOutput=0, butterfly=False, N=None)` (`bin/Language.py:2851`).
- **Registries**: `GRAMMAR_LAYER_CLASSES` (`bin/Language.py:3474`) and `_OPERATOR_SURFACE_SCHEMAS` (`bin/Language.py:3510`) — both get one line per new op.
- **Grammar file**: `data/role_collapsed.grammar` (current default) — `op_O1 = op.forward(op_I1, op_I2)` under `<compose>`, mirror `op_I1, op_I2 = op.reverse(op_O1)` under `<generate>`. Role tokens are learned-participation symbols, not POS.
- **`.where` span encoding (the model for `.when`)**: `EndpointSumWhere` (`bin/perceptual_analyzer.py:27`) packs a range into **2 dims**: key `= q(start)+q(end)` with `q(t)=[sin(t·dt), cos(t·dt)]`; by sum-to-product `= 2·cos((end−start)·dt/2)·[sin(center·dt), cos(center·dt)]`, so **angle=center, magnitude=length** (`encode` `:54`, `decode` `:62` via `center=atan2(w0,w1)/dt`, `length` from radius, `is_recoverable` `:81`). "Do NOT normalise the key — the magnitude carries the length." `.when` mirrors this but is **signed/zero-centered** (no `% 2π`, no integer snap).
- **`.when` plumbing**: `WhenEncoding` (`bin/Spaces.py:394`, minimal — replaced). `SubSpace` (`:4249`) holds `.when`; `nWhen = whenEncoding.nDim`; `muxedSize = nWhat + nWhere + nWhen` (`:4406`, `:6239`). Construction site: `WhenEncoding(10000, _nWhen)` at `bin/Spaces.py:6245`; demux reverse at `:6100`. With `nWhen=2`, `.when` occupies the same width as `.where` in the muxed `[what|where|when]` layout. `whenScale=0.1` loss weight read at `bin/Models.py:610`.
- **Tests**: `test/test_grammar_*.py` pattern (instantiate, `compose`, assert shape/finite/range/roundtrip); run `.venv/bin/python -m pytest <nodeid>`. `test/test_basicmodel.py` ~24 pre-existing failures (order-fragile). No sentence corpus.
- **Shared test preamble** (every new test file begins with this; not repeated per-task below):
  ```python
  import math, os, sys, unittest
  from pathlib import Path
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
  import torch
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
  ```

---

## File Structure Map

**New files**
- `bin/surface_tense.py` — pure table-driven surface $\to$ `(tense, aspect_chain, base_verb)` normalizer (no torch). One responsibility: lexical/auxiliary-pattern lookup.
- `bin/bind_resolver.py` — parse-time candidate ranking for contextual BIND (pure logic over small dataclass records; unit-testable without a live parse).
- `bin/morphology.py` — bijective single-word `analyze`/`inflect` paradigm + `agreement_category` + `MorphologyLayer` boundary transform (Phase 7; pure logic + a non-fold transform).
- `test/test_when_range_encoding.py`, `test/test_grammar_preposition.py`, `test/test_contextual_bind.py`, `test/test_tense_aspect.py`, `test/test_when_grammar_rules.py`, `test/test_grammar_fixtures.py`, `test/test_morphology.py`.

**Modified files**
- `bin/Spaces.py` — replace `WhenEncoding` (`394-453`) with `WhenRangeEncoding` (2-dim endpoint-sum, signed); keep `WhenEncoding = WhenRangeEncoding` alias for the existing import/construction; update construction `:6245` to `WhenRangeEncoding(64, _nWhen)`.
- `bin/Language.py` — add `PrepositionLayer`, `ContextualBindLayer`, `TenseLayer`, `AspectLayer` (after `LowerLayer` ~`2490`); 4 entries each in `GRAMMAR_LAYER_CLASSES` (`3474`) and `_OPERATOR_SURFACE_SCHEMAS` (`3510`); add the BIND-context stash to `BinaryStructuredReductionLayer.forward` (`~5300`).
- `data/role_collapsed.grammar` — compose+generate rule pairs for `preposition`, `bind`, `tense`, `aspect`.
- `bin/perceptual_analyzer.py` — Phase 7 morphology hooks: `MeronymicAnalyzer.push` (`~263`, analysis) and `synthesize_tree` (`~427`, synthesis).
- `bin/participation.py` — read only (`learned_collapse` `:172` supplies the collapsed-POS category that doubles as the agreement marker).
- `data/MentalModel.xml` — Phase 6 only: set `<nWhen>2</nWhen>` (same width as `<nWhere>2</nWhere>`).
- `bin/Models.py` — no new param (`.when` uses a fixed period like `.where`); just confirm the `nWhen` space-read reaches the construction at `bin/Spaces.py:6245`.

---

## Phase 1 — `PREPOSITION(P, X)`

`PREPOSITION` packages a learned marker `P` with a phrase `X` (NP/VP/S) and is **transparent to `X`'s content** (`forward(P,X)` returns `X`), so a downstream `lift`/`intersection` reads the phrase. The marker is recorded via absorb/emit, *not* folded into content — PREPOSITION does not decide the final relation (spec "Operation 1"). C-tier (composes phrases like `lift`, feeds the S-tier relations).

### Task 1.1: `PrepositionLayer` forward/reverse (parameter-free, content-preserving)

**Files:** Modify `bin/Language.py` (after `LowerLayer` ~`2490`); Test `test/test_grammar_preposition.py`.

- [ ] **Step 1: Write the failing test** (`test/test_grammar_preposition.py`, after the shared preamble + `from Language import PrepositionLayer`)
```python
class TestPrepositionLayer(unittest.TestCase):
    def test_class_contract(self):
        self.assertEqual(PrepositionLayer.rule_name, "preposition")
        self.assertEqual(PrepositionLayer.arity, 2)
        self.assertEqual(PrepositionLayer.tier, "C")
    def test_parameter_free_construction(self):
        self.assertIsInstance(PrepositionLayer(), PrepositionLayer)  # _resolve_rule_layer uses cls()
    def test_forward_is_content_transparent(self):
        layer = PrepositionLayer()
        marker, phrase = torch.randn(2, 3, 6), torch.randn(2, 3, 6)
        out = layer.forward(marker, phrase)
        self.assertEqual(out.shape, phrase.shape)
        self.assertTrue(torch.allclose(out, phrase, atol=1e-6))  # phrase survives unchanged
    def test_compose_matches_forward(self):
        layer = PrepositionLayer(); m, p = torch.randn(1,4,8), torch.randn(1,4,8)
        self.assertTrue(torch.allclose(layer.compose(m, p), layer.forward(m, p), atol=1e-6))
    def test_reverse_structural_split(self):
        layer = PrepositionLayer(); parent = torch.randn(2, 3, 6)
        left, right = layer.reverse(parent)
        self.assertTrue(torch.allclose(right, parent, atol=1e-6))  # content side recovers phrase
        self.assertEqual(left.shape, parent.shape)
    def test_permissive_arguments(self):  # accepts NP/VP/S-shaped content; gating is a learned hook
        layer = PrepositionLayer()
        for d in (2, 6, 10):
            x = torch.randn(2, 3, d); self.assertEqual(layer.forward(x, x).shape, x.shape)
```
- [ ] **Step 2: Run, verify it fails** — `.venv/bin/python -m pytest test/test_grammar_preposition.py -v` $\to$ FAIL `ImportError: cannot import name 'PrepositionLayer'`.
- [ ] **Step 3: Write minimal implementation** — add after `LowerLayer` in `bin/Language.py`:
```python
class PrepositionLayer(GrammarLayer):
    """preposition(P, X) -- marker-headed phrase packaging (binary, C-tier).

    Packages a learned surface marker P (that / to / in / because / when)
    with a phrase X (NP / VP / S). Transparent to X's content: forward(P, X)
    returns X unchanged so a downstream lift / intersection reads the
    phrase. The marker is recorded through the base-class absorb / emit
    machinery, NOT folded into content -- PREPOSITION does not decide the
    final relation; that is learned from how the marker-headed phrase
    participates downstream (spec "Operation 1: PREPOSITION").

    Starts PERMISSIVE: any [B, N, D] content is accepted as X. Per-marker
    argument gating (NP-only `in` vs S-only `that`) is a learned hook for
    later (bound_markers participation), not built here. reverse is the
    structural (X, X) split: the content side recovers the phrase exactly;
    the marker side is realized by emit from the bound marker.
    """
    rule_name = "preposition"; arity = 2
    invertible = True; lossy = False; tier = 'C'; reads_activation = False

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
    def forward(self, left, right):
        return right                       # P is the marker (absorbed), X passes through
    def reverse(self, parent):
        return parent, parent              # (marker_placeholder, phrase); emit realizes the marker
    def compose(self, left, right):
        return self.forward(left, right)
    def generate(self, parent):
        return self.reverse(parent)
```
- [ ] **Step 4: Run, verify it passes** — `.venv/bin/python -m pytest test/test_grammar_preposition.py -v` $\to$ PASS.
- [ ] **Step 5: Commit** (Alec) — `feat: add PrepositionLayer (marker-headed phrase, content-transparent)`

### Task 1.2: Register PREPOSITION (T3 directional marker)

**Files:** Modify `bin/Language.py:3474` + `:3510`; extend `test/test_grammar_preposition.py`.

- [ ] **Step 1: Failing test** (append)
```python
class TestPrepositionRegistration(unittest.TestCase):
    def test_in_grammar_layer_classes(self):
        from Language import GRAMMAR_LAYER_CLASSES
        self.assertIs(GRAMMAR_LAYER_CLASSES["preposition"], PrepositionLayer)
    def test_surface_schema_assigned(self):
        from Layers import T3_BINARY_DIRECTIONAL   # learned PRE marker, does not select the op
        self.assertEqual(PrepositionLayer.surface_schema, T3_BINARY_DIRECTIONAL)
```
- [ ] **Step 2: Run, verify it fails** — `... -k TestPrepositionRegistration` $\to$ FAIL `KeyError: 'preposition'`.
- [ ] **Step 3: Implement** — add `'preposition': PrepositionLayer,` after `'lower': LowerLayer,` in `GRAMMAR_LAYER_CLASSES`; add `'preposition': T3_BINARY_DIRECTIONAL,` after `'lower': T4_BINARY_JUXTAPOSE,` in `_OPERATOR_SURFACE_SCHEMAS` (a learned PRE marker that does *not* select the op, like the old `PP = lift(P, NP)`).
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: register preposition op (T3 directional schema)`

### Task 1.3: Add PREPOSITION grammar rules

**Files:** Modify `data/role_collapsed.grammar`; Test `test/test_when_grammar_rules.py`.

- [ ] **Step 1: Failing test** (`test/test_when_grammar_rules.py`) — load the grammar and assert the rule is present (use the project's grammar loader; mirror `test/test_role_collapsed_grammar.py` for the load idiom):
```python
def test_preposition_rule_present():
    from Language import Grammar
    g = Grammar(); g.load_grammar("data/role_collapsed.grammar"); g.configure()
    names = {r.method_name for r in g.rules_upward}
    assert "preposition" in names
```
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — in `data/role_collapsed.grammar` `<SymbolicSpace>`, add to `<compose>`: `<rule>preposition_O1 = preposition.forward(preposition_I1, preposition_I2)</rule>` and to `<generate>`: `<rule>preposition_I1, preposition_I2 = preposition.reverse(preposition_O1)</rule>`. (The layer's `tier='C'` re-tags the rule per the fixup at `bin/Language.py:1353`.)
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add preposition grammar rules`

### Task 1.4: Confirm PREPOSITION binds through the signal router

**Files:** Test `test/test_when_grammar_rules.py` (extend).

- [ ] **Step 1: Failing test** — assert `_resolve_rule_layer('C', 'preposition')` returns a `PrepositionLayer` (it is the function `_wire_signal_router_grammar_ops` calls):
```python
def test_preposition_resolves_to_layer():
    from Language import PrepositionLayer
    # build the model's LanguageLayer per the project's standard construction,
    # then: layer = lang._resolve_rule_layer('C', 'preposition')
    # assert isinstance(layer, PrepositionLayer)
```
(Use the construction idiom from `test/test_grammar_rewrite.py`; the exact builder is `grep -n "_resolve_rule_layer\|_wire_signal_router_grammar_ops" test/`.)
- [ ] **Step 2–4:** Run $\to$ confirm green (no impl change expected; this guards the auto-wiring contract).
- [ ] **Step 5: Commit** (Alec) — `test: guard preposition signal-router binding`

---

## Phase 2 — Contextual `BIND` (wired into the live fold)

Surface `LIFT(BIND, VP)` is reinterpreted at parse time as `LIFT(resolved_ref, VP)` (spec "Operation 2"). BIND is a contextual marker (no spoken NP, no plain codebook lookup). Resolution is parse-time and **wired into the binary reduction**: the fold stashes the live constituent slab on the BIND op, which resolves the missing NP against the nearest-left constructed participant. C-tier (its output feeds `lift`).

### Task 2.1: `bind_resolver` ranking (pure logic, no torch ops)

**Files:** Create `bin/bind_resolver.py`; Test `test/test_contextual_bind.py`.

- [ ] **Step 1: Failing test** (`test/test_contextual_bind.py`)
```python
from bind_resolver import Participant, rank_candidates, resolve_bind

def _p(id, role, position, participation=0.0):
    return Participant(id=id, vec=torch.full((4,), float(id)), role=role,
                       position=position, participation=participation)

def test_subject_control_picks_subject():
    # "Alice wants to run": only NP1=Alice accessible; want=subject-control.
    _ranked, chosen = rank_candidates([_p(1, "subject", 0)], licensing="subject_control")
    assert chosen == 0
def test_object_control_picks_object_not_nearer_subject():
    # "Alice persuaded Bob to run": NP1=Alice(subj), NP2=Bob(obj); persuade=object-control.
    parts = [_p(1, "subject", 0), _p(2, "object", 1)]
    _ranked, chosen = rank_candidates(parts, licensing="object_control")
    assert parts[chosen].role == "object"
def test_subject_control_does_not_grab_nearer_object():
    parts = [_p(1, "subject", 0), _p(2, "object", 1)]
    _ranked, chosen = rank_candidates(parts, licensing="subject_control")
    assert parts[chosen].role == "subject"
def test_unknown_licensing_falls_back_to_locality():
    _ranked, chosen = rank_candidates([_p(1, "other", 0), _p(2, "other", 5)], licensing=None)
    assert chosen == 1                      # most recent wins
def test_no_participants_is_unresolved():
    assert rank_candidates([], "subject_control") == ([], None)
    assert resolve_bind([], "subject_control") == (None, None)
```
- [ ] **Step 2: Run, verify it fails** — `ImportError`.
- [ ] **Step 3: Implement** `bin/bind_resolver.py`:
```python
"""Parse-time contextual-BIND ranking. Ranks accessible participants from the
current parse's left-context by: (1) constructional licensing -- want =>
subject-control (prefer subject NP), persuade => object-control (prefer
object NP); (2) locality -- more recent (higher position) wins; (3) learned
participation -- additive score hook (default 0.0). Pure ranking over small
records so it is unit-testable without a live parse. This is the resolution
*core* and the licensing refinement path; the live fold (Task 2.4) uses its
locality branch as a vectorized nearest-left pick when lemmas are unavailable."""
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Participant:
    id: int
    vec: torch.Tensor          # constructed NP content, e.g. NP1 = INTERSECT(Alice, tired)
    role: str                  # 'subject' / 'object' / 'other'
    position: int              # surface order index (higher = more recent)
    participation: float = 0.0 # learned score hook

_LICENSE_ROLE = {"subject_control": "subject", "object_control": "object"}

def _score(part: Participant, licensing: Optional[str]) -> tuple:
    pref = _LICENSE_ROLE.get(licensing or "")
    licensed = 1.0 if (pref is not None and part.role == pref) else 0.0
    return (licensed, part.participation, float(part.position))

def rank_candidates(participants, licensing=None):
    """Return (ranked_best_first, chosen_index_into_original) or ([], None)."""
    if not participants:
        return [], None
    indexed = sorted(enumerate(participants),
                     key=lambda iv: _score(iv[1], licensing), reverse=True)
    return [p for _i, p in indexed], indexed[0][0]

def resolve_bind(participants, licensing=None):
    """Return (vec, chosen_index) or (None, None) when nothing is accessible."""
    _ranked, chosen = rank_candidates(participants, licensing=licensing)
    return (None, None) if chosen is None else (participants[chosen].vec, chosen)
```
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add parse-time contextual-BIND ranking`

### Task 2.2: `ContextualBindLayer` (resolves against the live slab or a participant list)

**Files:** Modify `bin/Language.py` (after `PrepositionLayer`) + registries; extend `test/test_contextual_bind.py`.

- [ ] **Step 1: Failing test** (append)
```python
from Language import ContextualBindLayer
def test_resolves_nearest_left_from_slab():
    layer = ContextualBindLayer()
    alice = torch.tensor([1.,0,0,0]); bind_m = torch.tensor([0.,1,0,0]); run = torch.tensor([0.,0,1,0])
    slab = torch.stack([alice, bind_m, run]).unsqueeze(0)        # [1, 3, 4]
    layer.set_bind_context(slab=slab)
    left, right = slab[:, :-1, :], slab[:, 1:, :]               # pairs as the fold passes them
    out = layer.compose(left, right)                            # [1, 2, 4]
    assert torch.allclose(out[:, 1, :], alice)                  # pair (BIND, run) -> nearest-left Alice
def test_resolves_object_control_from_participants():
    layer = ContextualBindLayer()
    alice = Participant(1, torch.zeros(1,1,4), "subject", 0)
    bob   = Participant(2, torch.ones(1,1,4),  "object",  1)
    layer.set_bind_context(participants=[alice, bob], licensing="object_control")
    out = layer.compose(torch.randn(1,1,4), torch.randn(1,1,4))
    assert torch.allclose(out, bob.vec.expand_as(out), atol=1e-6)
def test_no_context_passes_marker_through():
    layer = ContextualBindLayer()
    m = torch.randn(1,1,4)
    assert torch.allclose(layer.compose(m, torch.randn(1,1,4)), m, atol=1e-6)
def test_class_contract():
    assert (ContextualBindLayer.rule_name == "bind" and ContextualBindLayer.arity == 2
            and ContextualBindLayer.tier == "C")
```
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — add after `PrepositionLayer`:
```python
class ContextualBindLayer(GrammarLayer):
    """bind(BIND, VP) -- contextual missing-NP marker (binary, C-tier).

    Surface LIFT(BIND, VP) is reinterpreted at parse time as
    LIFT(resolved_ref, VP) (spec "Operation 2"). compose(BIND_marker, VP)
    resolves an accessible participant already constructed in the current
    parse and returns its vector, so the enclosing lift sees a real NP in
    the left slot. Two context modes (stashed via set_bind_context before
    dispatch; a plain attribute, not an nn.Module child):

      * slab=[B, N, D]  -- the live constituent slab from the fold
        (Task 2.4). Resolution is vectorized NEAREST-LEFT: for each adjacent
        pair p the missing NP is constituent p-1 (the most recently built
        phrase before the BIND operand). Position 0 has no left context ->
        the marker passes through. This is the locality branch of
        bind_resolver, expressed as a tensor roll so it runs inside the
        parallel fold.
      * participants=[Participant] + licensing -- the ranked path
        (bind_resolver.resolve_bind): want=>subject-control,
        persuade=>object-control + locality + learned participation. Used by
        fixtures / unit tests and as the licensing refinement over locality.

    No context, or no candidate => the marker passes through unchanged --
    BIND never invents a binding.
    """
    rule_name = "bind"; arity = 2
    invertible = False; lossy = True; tier = 'C'; reads_activation = False

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_bind_context', None)
    def set_bind_context(self, *, slab=None, participants=None, licensing=None):
        object.__setattr__(self, '_bind_context',
                           {'slab': slab, 'participants': participants, 'licensing': licensing})
    def clear_bind_context(self):
        object.__setattr__(self, '_bind_context', None)
    def forward(self, left, right):
        ctx = self._bind_context
        if ctx and ctx.get('slab') is not None:
            slab = ctx['slab']                              # [B, N, D] live constituents
            # nearest-left participant for each pair p: constituent p-1;
            # position 0 keeps itself (no left context -> passthrough).
            prior = torch.cat([slab[:, :1, :], slab[:, :-1, :]], dim=1)  # [B, N, D]
            return prior[:, :-1, :]                          # [B, N-1, D], aligned to pairs
        if ctx and ctx.get('participants'):
            from bind_resolver import resolve_bind
            vec, _chosen = resolve_bind(ctx['participants'], licensing=ctx.get('licensing'))
            if vec is not None:
                return vec.expand_as(left) if vec.shape != left.shape else vec
        return left
    def reverse(self, parent):
        return parent, parent              # lossy: contextual binding is not recoverable
    def compose(self, left, right):
        return self.forward(left, right)
    def generate(self, parent):
        return self.reverse(parent)
```
Add `'bind': ContextualBindLayer,` to `GRAMMAR_LAYER_CLASSES`; add `'bind': T5_BINARY_ELISION,` to `_OPERATOR_SURFACE_SCHEMAS` (a contextual marker with no spoken surface form $\to$ elision; the VP survives).
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add ContextualBindLayer (slab nearest-left + ranked participant resolution)`

### Task 2.3: Add BIND grammar rules

**Files:** Modify `data/role_collapsed.grammar`; extend `test/test_when_grammar_rules.py`.

- [ ] **Step 1: Failing test** — assert `"bind"` in `{r.method_name for r in g.rules_upward}` (same loader idiom as Task 1.3).
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — add to `<compose>`: `<rule>bind_O1 = bind.forward(bind_I1, bind_I2)</rule>`; to `<generate>`: `<rule>bind_I1, bind_I2 = bind.reverse(bind_O1)</rule>`.
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add bind grammar rules`

### Task 2.4: Wire `set_bind_context` into the live fold

**Files:** Modify `bin/Language.py` (`BinaryStructuredReductionLayer.forward`, `~5300`); extend `test/test_contextual_bind.py`.

The fold applies every op to all adjacent pairs in `_stacked_reduced` (`bin/Language.py:5221`). Just before that, stash the current slab `x` on any op that exposes `set_bind_context`, so when its `compose(left, right)` fires it can resolve against the live left-context. Duck-typed (no isinstance/ordering coupling); unwraps `_BinaryGrammarOpAdapter` via `.gl`.

- [ ] **Step 1: Failing test** (append)
```python
def test_reduction_layer_stashes_live_slab_on_bind():
    from Language import (BinaryStructuredReductionLayer, ContextualBindLayer,
                          _BinaryGrammarOpAdapter)
    bind = ContextualBindLayer()
    layer = BinaryStructuredReductionLayer(d_model=4, ops=[_BinaryGrammarOpAdapter(bind)])
    alice = torch.tensor([1.,0,0,0]); bind_m = torch.tensor([0.,1,0,0]); run = torch.tensor([0.,0,1,0])
    x = torch.stack([alice, bind_m, run]).unsqueeze(0)         # [1, 3, 4]
    layer.forward(x)                                           # triggers the stash
    assert bind._bind_context is not None and bind._bind_context['slab'] is x
    # and the wired resolution gives nearest-left for the (BIND, run) pair:
    out = bind.compose(x[:, :-1, :], x[:, 1:, :])
    assert torch.allclose(out[:, 1, :], alice)
```
- [ ] **Step 2: Run, verify it fails** — the stash isn't there yet, `bind._bind_context is None`.
- [ ] **Step 3: Implement** — in `BinaryStructuredReductionLayer.forward`, immediately after `h = self.context_net(x)` and before `stacked_reduced = self._stacked_reduced(x)` (`~bin/Language.py:5300`):
```python
        # Wire contextual BIND to live parse state: stash the current
        # constituent slab on any op that resolves a missing NP against the
        # constructed left-context. Applied before _stacked_reduced so the
        # op's compose(left, right) over all pairs sees the live slab.
        for _op in self.ops:
            _gl = getattr(_op, 'gl', _op)          # unwrap _BinaryGrammarOpAdapter
            if hasattr(_gl, 'set_bind_context'):
                _gl.set_bind_context(slab=x)
```
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: wire contextual BIND to the live reduction slab`

> **Scope note (honest):** the resolution mechanism is fully wired and exercised — the fold calls `set_bind_context(slab=x)` every round and `ContextualBindLayer.compose` consumes it. The first-cut fold policy is **locality** (nearest-left constituent), which correctly handles both spec fixtures (`Alice` for "wants to run"; `Bob` for "persuaded Bob to run", since Bob is the nearest constituent before the infinitival). **Licensing-based reranking inside the fold** (want=subject-control skipping a nearer object) needs the governing verb's lemma at the reduction site — that is the documented refinement carried by `bind_resolver` + learned participation, not a separate deferral. Introducing a *covert* BIND constituent into a subjectless infinitival (empty-category positing) is the surface-analysis seam the corpus-scale grammar will own; the focused fixtures (Task 5.1) supply the BIND constituent explicitly, per the spec's "focused fixtures before corpus-scale training" staging.

---

## Phase 3 — `.when` range encoding (2-dim endpoint-sum, zero-centered, signed — mirrors `.where`)

Replace `WhenEncoding` with a **2-dim** range key that mirrors `.where`/`EndpointSumWhere`: `.when = q(start) + q(end)` with `q(t) = [sin(t·dt), sin... ]` — by sum-to-product the key is `2·cos((end−start)·dt/2)·[sin(center·dt), cos(center·dt)]`, so **angle carries the center time, magnitude carries the duration**. Difference from `.where`: `.when` is **zero-centered** ($t=0$ = present reference) and **signed** (past $<0<$ future) — decode uses `atan2`'s native $(-\pi, \pi]$ **without** `% 2\pi` and does **not** snap to the integer grid (aspect uses fractional $\varepsilon$). Tense $= q(t+\delta) = R(\delta)\,q(t)$ is a single $2\times2$ rotation of the key; aspect reshapes the interval (= the key's duration). `nWhen=2`, same width as `nWhere=2`.

> **Deviation from spec text (user directive):** the spec's "each endpoint has constant norm / multi-frequency $q(t)$" is superseded — the endpoint-sum key's magnitude carries duration, exactly like `.where`. Absolute time (past/future) does not drag the norm (rotation preserves magnitude); only duration changes it. Single frequency is sufficient for the small signed integer/fractional times used here.

### Task 3.1: `WhenRangeEncoding` — 2-dim signed endpoint-sum encode/decode + rotation

**Files:** Modify `bin/Spaces.py` (replace `WhenEncoding` `394-453`); Test `test/test_when_range_encoding.py`.

- [ ] **Step 1: Failing test** (`test/test_when_range_encoding.py`, after preamble + `from Spaces import WhenRangeEncoding`)
```python
def _enc(maxT=64):
    return WhenRangeEncoding(maxT=maxT, n_when=2)

def test_ndim_is_two_like_where():
    assert _enc().nDim == 2
def test_disabled_when_zero_width():
    assert WhenRangeEncoding(maxT=64, n_when=0).nDim == 0
def test_sin_cos_layout_present():
    q = _enc().q(0.0)                                  # [sin(0), cos(0)] = (0, 1)
    assert math.isclose(float(q[0]), 0.0, abs_tol=1e-6) and math.isclose(float(q[1]), 1.0, abs_tol=1e-6)
def test_present_default_key_is_zero_two():
    key = _enc().encode_range(0.0, 0.0)                # q(0)+q(0) = (0, 2)
    assert math.isclose(float(key[0]), 0.0, abs_tol=1e-6) and math.isclose(float(key[1]), 2.0, abs_tol=1e-6)
def test_signed_decode_negative_times():              # past must decode negative, not maxT-1
    enc = _enc()
    for (s, e) in [(-1.0, -1.0), (-1.0, 0.0), (-2.0, -1.0), (0.0, 0.0)]:
        ds, de = enc.decode(enc.encode_range(s, e))
        assert math.isclose(float(ds), s, abs_tol=1e-4) and math.isclose(float(de), e, abs_tol=1e-4)
def test_magnitude_carries_duration_not_position():
    enc = _enc()
    # same duration (0) at different times -> same magnitude; longer duration -> smaller magnitude
    assert math.isclose(float(enc.encode_range(0,0).norm()), float(enc.encode_range(-1,-1).norm()), rel_tol=1e-5)
    assert float(enc.encode_range(-1,0).norm()) < float(enc.encode_range(0,0).norm())
def test_tense_rotation_is_phase_shift():
    enc = _enc()
    assert torch.allclose(enc.rotate(enc.encode_range(0,0), -1.0), enc.encode_range(-1,-1), atol=1e-5)  # PAST
    assert torch.allclose(enc.rotate(enc.encode_range(0,0),  1.0), enc.encode_range( 1, 1), atol=1e-5)  # FUTURE
def test_rotate_range_shifts_both_endpoints():
    enc = _enc(); s, e = enc.decode(enc.rotate_range(enc.encode_range(-1.0, 0.0), -1.0))  # PAST(PERFECT)
    assert math.isclose(float(s), -2.0, abs_tol=1e-4) and math.isclose(float(e), -1.0, abs_tol=1e-4)
def test_aspect_interval_shapes():
    enc = _enc()
    assert enc.aspect_interval(0.0, "SIMPLE") == (0.0, 0.0)
    assert enc.aspect_interval(0.0, "PERFECT") == (-1.0, 0.0)
    s, e = enc.aspect_interval(0.0, "PROGRESSIVE", eps=0.25); assert math.isclose(s, -0.25) and math.isclose(e, 0.25)
def test_forward_stamps_present_default_and_reverse_zeros():
    enc = _enc(); y = enc.forward(torch.zeros(2, 4, 10))
    ds, de = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(ds), 0.0, abs_tol=1e-4) and math.isclose(float(de), 0.0, abs_tol=1e-4)
    cleaned, _ = enc.reverse(y); idx = enc.resolve(cleaned.shape[-1])
    assert torch.allclose(cleaned[..., idx], torch.zeros_like(cleaned[..., idx]), atol=1e-6)
```
- [ ] **Step 2: Run, verify it fails** — `ImportError: cannot import name 'WhenRangeEncoding'`.
- [ ] **Step 3: Implement** — replace `WhenEncoding` (`bin/Spaces.py:394-453`) with (keep top-of-file `import math`, `import numpy as np`):
```python
class WhenRangeEncoding(QuadratureEncoding):
    """Zero-centered, SIGNED, two-endpoint temporal RANGE in 2 dims --
    mirrors .where (EndpointSumWhere, bin/perceptual_analyzer.py). The .when
    key is the endpoint SUM q(start)+q(end) with q(t) = [sin(t*dt),
    cos(t*dt)]. By sum-to-product the key is
    2*cos((end-start)*dt/2) * [sin(center*dt), cos(center*dt)] with
    center=(start+end)/2: the ANGLE carries the center time, the MAGNITUDE
    carries the duration -- so a range fits in nDim=2, exactly like .where.

    Difference from .where: .when is ZERO-CENTERED (t=0 = present reference)
    and SIGNED (past<0<future), so decode uses atan2's native (-pi, pi]
    range WITHOUT the % 2*pi fold and does NOT snap to the integer grid
    (aspect uses fractional eps). Default present/simple = q(0)+q(0) = (0, 2).

    Tense is a phase rotation R(delta): shifting both endpoints by delta
    rotates the summed 2-vector by delta*dt (center+delta, duration
    unchanged). Aspect reshapes the interval (aspect_interval) -> the key's
    magnitude. Do NOT normalise the key: the magnitude carries the duration.
    nDim=2 (== nWhere); disabled (nDim=0) when n_when=0.
    """
    index = []
    t = 0

    def __init__(self, maxT=64, n_when=0):
        if n_when > 0:
            super().__init__([-2, -1], maxT)        # div_term = 2*pi/maxT; slots [-2, -1]
            self.nDim = 2
        else:
            Encoding.__init__(self, [], 1)
            self.div_term = 2 * math.pi / max(1, maxT)
            self.nDim = 0
        self.t = 0

    def q(self, t):
        """One endpoint phasor [sin(t*dt), cos(t*dt)] (QuadratureEncoding order)."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), device=TheDevice.get())
        ang = t * self.div_term
        return torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)

    def encode_range(self, start_t, end_t):
        """Endpoint-sum key q(start)+q(end) -> [..., 2]; magnitude carries duration."""
        return self.q(start_t) + self.q(end_t)

    def decode(self, encoded):
        """(start_t, end_t) signed floats. center = atan2(sin, cos)/dt (signed,
        no % 2*pi); half-duration from |key| = 2*cos(half*dt)."""
        w0, w1 = encoded[..., 0], encoded[..., 1]
        dt = self.div_term
        center = torch.atan2(w0, w1) / dt
        radius = torch.sqrt(w0 * w0 + w1 * w1).clamp(max=2.0)
        half = torch.arccos((radius / 2.0).clamp(-1.0, 1.0)) / dt
        return center - half, center + half

    def rotate(self, key, delta):
        """Tense: shifting both endpoints by delta rotates the key by
        a = delta*dt (single 2x2 rotation; magnitude/duration preserved)."""
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(float(delta), device=key.device)
        a = delta * self.div_term
        ca, sa = torch.cos(a), torch.sin(a)
        s, c = key[..., 0], key[..., 1]              # (sin-part, cos-part)
        out = key.clone()
        out[..., 0] = s * ca + c * sa                # sin(theta + a)
        out[..., 1] = c * ca - s * sa                # cos(theta + a)
        return out

    rotate_range = rotate                            # the key already aggregates both endpoints

    @staticmethod
    def aspect_interval(r, kind, eps=0.25):
        """(start_t, end_t) for reference r and aspect kind (spec)."""
        r = float(r)
        if kind == "SIMPLE":      return (r, r)
        if kind == "PERFECT":     return (r - 1.0, r)
        if kind == "PROGRESSIVE": return (r - eps, r + eps)
        raise ValueError(f"unknown aspect kind {kind!r}")

    def is_recoverable(self, start, end):
        """True iff [start, end] round-trips: |center| and duration below the
        half-period pi/dt (mirrors EndpointSumWhere, but signed center)."""
        center = (float(start) + float(end)) / 2.0
        dur = float(end) - float(start)
        hp = math.pi / self.div_term
        return abs(center) < hp and 0.0 <= dur < hp

    def forward(self, x):
        """Stamp the present default key [q(0)+q(0)] = (0, 2) into the when slots."""
        if self.nDim == 0:
            return x
        index = np.add([x.shape[-1]] * len(self.index), self.index)
        key = self.encode_range(0.0, 0.0)
        y = x.clone()
        y[:, :, index] = key.to(y.device).expand(x.shape[0], x.shape[1], -1)
        return y

    def increment(self, batch):
        """No-op: .when is zero-centered on the reference moment, not a counter."""
        return

    @staticmethod
    def test():
        """Self-test: encode a present-perfect range and decode it."""
        te = WhenRangeEncoding(64, n_when=2)
        print(f"present-perfect decoded: {te.decode(te.encode_range(-1.0, 0.0))}")


# Back-compat alias: existing import (bin/Models.py) / construction
# (bin/Spaces.py:6245) resolve to the range encoding. The single-(sin,cos)
# monotonic-counter WhenEncoding is superseded (spec ".when Encoding").
WhenEncoding = WhenRangeEncoding
```
- [ ] **Step 4: Run, verify it passes** (10 tests).
- [ ] **Step 5: Commit** (Alec) — `feat: replace WhenEncoding with 2-dim signed endpoint-sum WhenRangeEncoding`

### Task 3.2: Construction site — pass `nWhen` with a suitable period

**Files:** Modify `bin/Spaces.py:6245`; Test `test/test_when_grammar_rules.py` (extend).

- [ ] **Step 1: Failing test** — assert that building the model leaves `.when` disabled by default (`subspace.nWhen == 0`) and that, when `nWhen=2`, the constructed encoding has `nDim == 2` and `decode(encode_range(-1,0)) ≈ (-1, 0)`.
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — change `WhenEncoding(10000, _nWhen)` at `bin/Spaces.py:6245` to `WhenRangeEncoding(64, _nWhen)` (period 64 suits the small signed times; the alias keeps any `WhenEncoding` import working). No new XML param — `.when` uses a fixed period like `.where`.
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: construct WhenRangeEncoding with a zero-centered period`

### Task 3.3: Width guard — SubSpace mux/demux/materialize at `nWhen=2`

**Files:** Test `test/test_when_range_encoding.py` (extend). Proves `SubSpace` is shape-agnostic for the 2-wide `.when` before the Phase 6 flip.

- [ ] **Step 1: Test** — construct a `SubSpace` with `whenEncoding = WhenRangeEncoding(64, 2)`, stamp a present-perfect range into the `.when` block of a muxed event, `materialize(mode="when")`, `demux`, assert the range round-trips and `muxedSize == nWhat + nWhere + 2`. Also assert `EventEncoding.decode` (`bin/Spaces.py:6097-6101`, `objects, time = self.whenEncoding.reverse(objects)`) returns a usable `time` for the **enabled** encoding.
- [ ] **Step 2–4:** Run; `.when=2` matches the `.where=2` width already in the muxed layout, so the mux/demux should be green immediately. If a literal width is hardcoded anywhere, fix it to read `whenEncoding.nDim`. **`reverse`/`decode` contract:** `WhenRangeEncoding.decode` returns a `(start, end)` tuple, so base `Encoding.reverse` (`bin/Spaces.py:107`) surfaces that tuple as `time` (only when enabled — the `nDim==0` branch returns a zeros tensor while disabled, so Phases 1–5 are unaffected). If any consumer of `EventEncoding.decode`'s `time` expects a single `[B, V]` tensor, override `WhenRangeEncoding.reverse` to return `torch.stack([start, end], dim=-1)` (`[B, V, 2]`) instead of the raw tuple; keep `decode` returning `(start, end)` for the layer/test API. Confirm the `.when` loss (`whenScale`, `bin/Models.py:610`) stays finite.
- [ ] **Step 5: Commit** (Alec) — `test: guard SubSpace .when mux/demux + reverse contract at width 2`

---

## Phase 4 — Tense / aspect as `.when` operations

`ran` $\to$ `PAST(run)`, `is running` $\to$ `PRESENT(PROGRESSIVE(run))`, etc. Tense = phase rotation; aspect = interval shaping. Both are **unary C-tier** ops modifying only the `.when` tail of the materialized muxed event; `.what`/`.where` pass through. Applied to the VP/event **before** the subject LIFT.

### Task 4.1: `surface_tense` normalization table (pure, no torch)

**Files:** Create `bin/surface_tense.py`; Test `test/test_tense_aspect.py`.

- [ ] **Step 1: Failing test** (`test/test_tense_aspect.py`)
```python
from surface_tense import normalize_surface
def test_ran():            assert normalize_surface(["ran"]) == ("PAST", [], "run")
def test_is_running():     assert normalize_surface(["is", "running"]) == ("PRESENT", ["PROGRESSIVE"], "run")
def test_has_run():        assert normalize_surface(["has", "run"]) == ("PRESENT", ["PERFECT"], "run")
def test_had_been_running():
    assert normalize_surface(["had", "been", "running"]) == ("PAST", ["PERFECT", "PROGRESSIVE"], "run")
def test_did_run():        assert normalize_surface(["did", "run"]) == ("PAST", [], "run")
def test_will_run():       assert normalize_surface(["will", "run"]) == ("FUTURE", [], "run")  # MODAL hook noted
```
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** `bin/surface_tense.py` — pure tables + `normalize_surface(tokens) -> (tense, aspect_chain, base_verb)`:
  - `_FINITE_TENSE = {"ran":"PAST","had":"PAST","did":"PAST","was":"PAST","were":"PAST","will":"FUTURE", ...}` (default `PRESENT`).
  - `be + V-ing` $\to$ `PROGRESSIVE`; `have + V-en` $\to$ `PERFECT`; `do/did/does + V` $\to$ tense-support, no aspect (`SIMPLE`); `will + V` $\to$ `FUTURE` (`# MODAL(will, X) hook noted, not built`).
  - Tense from the first finite element; aspects collected in surface order (outermost first); `base_verb = _base_of(tokens[-1])` via an irregular table (`ran`$\to$`run`, `been`$\to$`be`) with `-ing`/`-ed` suffix-strip fallback.
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add table-driven surface tense/aspect normalizer`

### Task 4.2: `TenseLayer` and `AspectLayer` operate on the VP/event `.when`

**Files:** Modify `bin/Language.py` (after `ContextualBindLayer`) + registries; extend `test/test_tense_aspect.py`.

- [ ] **Step 1: Failing test** (append) — build a muxed event `[B, V, nWhat+nWhere+2]` whose `.when` tail is `encode_range(0,0)`, then:
```python
from Language import TenseLayer, AspectLayer
def test_past_rotates_when_backward():
    t = TenseLayer(); t.set_op("PAST")
    # event with .when tail = encode_range(0,0); after forward, decode tail -> (-1, -1)
def test_perfect_shapes_interval_around_end():
    a = AspectLayer(); a.set_op("PERFECT")
    # present event r=0 -> tail decodes to (-1, 0)
def test_what_where_pass_through():
    # assert the head columns (everything but the 2-wide .when tail) are unchanged
```
(Use `WhenRangeEncoding` to build/inspect the tail; assert `tier == 'C'`, `arity == 1`.)
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — add after `ContextualBindLayer`:
```python
class _WhenOpMixin:
    """Shared helpers for unary ops that rewrite the .when tail of a
    materialized muxed event [B, V, nWhat + nWhere + nWhen]. Modifies ONLY
    the trailing nWhen (=2) columns; .what / .where pass through. Builds a
    matching WhenRangeEncoding to interpret / rewrite the key. Tense/aspect
    operate on the VP/event .when BEFORE the subject LIFT (spec note:
    equivalent post-LIFT)."""
    _WHEN_WIDTH = 2
    def _when_encoding(self):
        from Spaces import WhenRangeEncoding
        return WhenRangeEncoding(64, self._WHEN_WIDTH)
    def _split_when(self, x):
        w = self._WHEN_WIDTH
        if x.shape[-1] < w:
            raise ValueError(f"{type(self).__name__}: event width {x.shape[-1]} < "
                             f".when width {w}; is nWhen enabled?")
        return x[..., :-w], x[..., -w:]

class TenseLayer(_WhenOpMixin, GrammarLayer):
    """tense(X) -- shift the event .when reference time (unary, C-tier).
    PRESENT keeps reference at 0 (identity); PAST rotates the key backward
    (delta=-1); FUTURE forward (delta=+1). Tense is a phase rotation
    q(t+delta)=R(delta)q(t). Selected per-instance via set_op before dispatch."""
    rule_name = "tense"; arity = 1
    invertible = True; lossy = False; tier = 'C'; reads_activation = False
    _DELTA = {"PRESENT": 0.0, "PAST": -1.0, "FUTURE": 1.0}
    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_op', "PRESENT")
    def set_op(self, tense):
        if tense not in self._DELTA: raise ValueError(f"unknown tense {tense!r}")
        object.__setattr__(self, '_op', tense)
    def forward(self, x):
        head, when = self._split_when(x); delta = self._DELTA[self._op]
        if delta == 0.0: return x
        return torch.cat([head, self._when_encoding().rotate(when, delta)], dim=-1)
    def reverse(self, y):
        head, when = self._split_when(y); delta = self._DELTA[self._op]
        if delta == 0.0: return y
        return torch.cat([head, self._when_encoding().rotate(when, -delta)], dim=-1)
    def compose(self, x):     return self.forward(x)
    def generate(self, parent): return self.reverse(parent)

class AspectLayer(_WhenOpMixin, GrammarLayer):
    """aspect(X) -- shape the event .when interval (unary, C-tier). Relative
    to reference r (decoded from the range's END endpoint): SIMPLE->[r,r];
    PERFECT->[r-1,r]; PROGRESSIVE->[r-e,r+e], re-encoded as the key. Perfect
    is ASPECT, not future tense. Selected per-instance via set_op before dispatch."""
    rule_name = "aspect"; arity = 1
    invertible = False; lossy = True; tier = 'C'; reads_activation = False
    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        object.__setattr__(self, '_op', "SIMPLE"); object.__setattr__(self, '_eps', 0.25)
    def set_op(self, kind, eps=0.25):
        object.__setattr__(self, '_op', kind); object.__setattr__(self, '_eps', float(eps))
    def forward(self, x):
        head, when = self._split_when(x); enc = self._when_encoding()
        _s, end_t = enc.decode(when)
        r = float(end_t.reshape(-1)[0]) if end_t.numel() else 0.0
        s, e = enc.aspect_interval(r, self._op, eps=self._eps)
        key = enc.encode_range(s, e).to(x.device).expand(*when.shape[:-1], -1)
        return torch.cat([head, key], dim=-1)
    def reverse(self, parent): return parent      # lossy
    def compose(self, x):      return self.forward(x)
    def generate(self, parent): return self.reverse(parent)
```
Add `'tense': TenseLayer,` and `'aspect': AspectLayer,` to `GRAMMAR_LAYER_CLASSES`; add `'tense': T1_UNARY_AFFIX,` and `'aspect': T1_UNARY_AFFIX,` to `_OPERATOR_SURFACE_SCHEMAS` (unary affixes carried by morphology/auxiliaries).
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add TenseLayer/AspectLayer over the event .when subspace`

### Task 4.3: Add tense/aspect grammar rules

**Files:** Modify `data/role_collapsed.grammar`; extend `test/test_when_grammar_rules.py`.

- [ ] **Step 1: Failing test** — assert `"tense"` and `"aspect"` in `{r.method_name for r in g.rules_upward}`.
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — add to `<compose>`: `<rule>tense_O1 = tense.forward(tense_I1)</rule>` and `<rule>aspect_O1 = aspect.forward(aspect_I1)</rule>`; to `<generate>`: `<rule>tense_I1 = tense.reverse(tense_O1)</rule>` and `<rule>aspect_I1 = aspect.reverse(aspect_O1)</rule>`.
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add tense/aspect grammar rules`

---

## Phase 5 — Focused fixtures

### Task 5.1: Structural parse/round-trip harness for the 8 spec sentences

**Files:** Test `test/test_grammar_fixtures.py`. Exercises the new ops' APIs directly (the deterministic guarantee lives here, since `test_basicmodel.py` is order-fragile and there is no trained corpus).

- [ ] **Step 1: Write the tests** — one per spec sentence, asserting the structural composition the spec gives:
  - `that Alice left` $=$ `PREPOSITION(that, LIFT(Alice, left))` — PREPOSITION transparent to its phrase.
  - `Alice wants to run` — `VP2 = LIFT(BIND[Alice], run)`. Build the live-slab path: a `[Alice, BIND, run]` slab, `set_bind_context(slab=...)`, assert the `(BIND, run)` pair resolves to `Alice` (nearest-left). Also assert the ranked path (`participants=[Alice]`, `licensing="subject_control"`) resolves to `Alice`.
  - `The tired Alice wants to sleep` — `NP1 = INTERSECT(Alice, tired)`; the bind target is the **constructed** `NP1` (pass `NP1`'s vector as the slab constituent / participant, assert it — not the bare `Alice` symbol).
  - `Alice persuaded Bob to run` — slab `[Alice, persuaded, Bob, BIND, run]`; nearest-left of `(BIND, run)` is `Bob`. Also the ranked path with `licensing="object_control"` over `[Alice(subject), Bob(object)]` resolves to `Bob`.
  - `Alice ran` — `surface_tense(["ran"]) == ("PAST", [], "run")`; `TenseLayer("PAST")` rotates a present event `.when` to `(-1, -1)`.
  - `Alice is running` — `("PRESENT", ["PROGRESSIVE"], "run")`; aspect $\to (-\varepsilon, +\varepsilon) \approx (-0.25, 0.25)$.
  - `Alice has run` — `("PRESENT", ["PERFECT"], "run")`; aspect $\to (-1, 0)$.
  - `Alice had been running` — `("PAST", ["PERFECT","PROGRESSIVE"], "run")`. Apply **aspects innermost-first** (PROGRESSIVE then PERFECT), then tense PAST: PROGRESSIVE makes $[-\varepsilon, \varepsilon]$; PERFECT around END ($r=\varepsilon$) $\to [\varepsilon-1, \varepsilon]$; PAST shifts $-1$ $\to$ end $\approx \varepsilon-1 = -0.75$, start $\approx \varepsilon-2 = -1.75$. Assert ordering + past shift (`abs_tol=0.05`).
- [ ] **Step 2: Run, verify they fail (then pass as deps land).**
- [ ] **Step 3:** No new production code — fixtures compose the Phase 1–4 APIs.
- [ ] **Step 4: Run, verify all 8 pass** — `.venv/bin/python -m pytest test/test_grammar_fixtures.py -v`.
- [ ] **Step 5: Commit** (Alec) — `test: add 8 focused fixtures (PREPOSITION/BIND/.when/tense/aspect)`

---

## Phase 6 — Turn on `.when` + re-evaluation

### Task 6.1: Enable `nWhen=2` in `MentalModel.xml` (isolated, bisectable)

**Files:** Modify `data/MentalModel.xml`; Test `test/test_when_grammar_rules.py` (extend).

- [ ] **Step 1: Failing test** — `test_mentalmodel_when_width_is_two`: build from `MentalModel.xml`, assert the enabled `.when` subspaces report `nWhen == 2` (matching `nWhere == 2`) and the `.when` loss is finite on a tiny forward.
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — set `<nWhen>2</nWhen>` wherever `<nWhere>2</nWhere>` appears in `MentalModel.xml` (InputSpace/PerceptualSpace/SymbolicSpace; leave Conceptual/Output at 0). **This is the single "turn it on" switch — kept in its own commit so a bisect lands precisely.**
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: enable .when (nWhen=2) in MentalModel`

### Task 6.2: Full-suite regression gate + baseline accounting

- [ ] **Step 1:** Run `.venv/bin/python -m pytest test/test_basicmodel.py -q` — record failure count; must be **≤ baseline ~24** (memory: order-fragile, ~24 pre-existing on clean HEAD).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest test/test_role_collapsed_grammar.py test/test_d1_pos_recovery_gate.py -v` — must stay green (no POS tokens leaked by the new role tokens).
- [ ] **Step 3:** Run width-sensitive suites: `.venv/bin/python -m pytest test/test_grammar_binary_ops.py test/test_grammar_order_typing.py -q`.
- [ ] **Step 4:** If any regression traces to the `nWhen` width change, revert Task 6.1's commit (isolated) and fix the hardcoded width before re-enabling.
- [ ] **Step 5: Commit** (Alec) — `chore: regression gate for .when + new grammar ops`

---

## Phase 7 — Morphological lemma↔surface bridge (`MorphologyLayer`)

Factor inflected surface forms into a deep lemma + structured residual so the **SS codebook stays lemma-only** ("eat" is one row; "eat / eats / ate / eaten" stay distinct PS atoms), while percept↔symbol reconstruction stays **exact and repeatable**. Exactness rests on two existing invertible mechanisms, **not** on the lossy lemma map:

- **PS is the authoritative `.where`-keyed surface codebook** (`bin/Spaces.py:7841`, `:8014`) — surface atoms never change identity; `.where` (position) is held fixed across the transform.
- **The marker route** records the exact PS atom (`_marker_ps_id`, `bin/Language.py:6675`); `synthesize_tree` (`bin/perceptual_analyzer.py:427`) + `emit(marker_id=...)` (`bin/Layers.py:2069`, "route-recorded id for exact replay") reconstruct it.

The transform runs **before** the SS write (analysis: surface $\to$ lemma + `.when` + agreement marker) and **after** the SS read (synthesis: lemma + `.when` + agreement $\to$ surface), paired with `SymbolizeLayer` at the C↔S boundary. The verb tense/aspect half reuses `surface_tense` (Phase 4); **morphology owns the single-word inflection** (eats↔eat+3sg, ran↔run+past, mice↔mouse+pl) — multi-word aspect (has run, is running) stays the auxiliary-chain job of Phase 4. Agreement is **not** a new declared feature: it rides the **learned-participation (collapsed-POS) marker space** (Task 7.2), honoring the "no declared POS" constraint.

**Exactness invariant (Task 7.4):** `inflect(analyze(s)) == s` for every surface atom `s` in the PS vocabulary, and `MorphologyLayer.reverse(MorphologyLayer.forward(atom)) == atom` — **fail loud** on any divergence (`feedback_fail_loud_on_divergence`). Where the paradigm is ambiguous (irregular/unseen), the recorded PS atom (exact replay) wins and the divergence is surfaced, not swallowed.

### Task 7.1: `bin/morphology.py` — bijective `analyze` / `inflect` paradigm pair (pure)

**Files:** Create `bin/morphology.py`; Test `test/test_morphology.py`.

- [ ] **Step 1: Failing test** (`test/test_morphology.py`, after preamble)
```python
from morphology import analyze, inflect
def test_eats_factors_to_eat_present_3sg():
    assert analyze("eats", "verb") == ("eat", ("PRESENT", []), "3sg")
def test_ran_factors_to_run_past():
    assert analyze("ran", "verb") == ("run", ("PAST", []), "base")
def test_mice_factors_to_mouse_pl():
    assert analyze("mice", "noun") == ("mouse", None, "pl")
def test_round_trip_is_a_bijection_over_vocab():   # the exactness invariant (single-word forms)
    for s in ["eat", "eats", "ate", "run", "runs", "ran", "walk", "walks", "walked"]:
        l, w, a = analyze(s, "verb"); assert inflect(l, w, a, "verb") == s, (s, l, w, a)
    for s in ["cat", "cats", "mouse", "mice", "goose", "geese"]:
        l, w, a = analyze(s, "noun"); assert inflect(l, w, a, "noun") == s, (s, l, w, a)
```
- [ ] **Step 2: Run, verify it fails** — `ImportError`.
- [ ] **Step 3: Implement** `bin/morphology.py` (single-word paradigm; shares the irregular tables with `surface_tense`):
```python
"""Bijective surface<->lemma morphology bridge (Phase 7). Keeps the SS
codebook lemma-only while PS stays the authoritative .where-keyed surface
codebook. Morphology owns SINGLE-WORD inflection (eats<->eat+3sg, ran<->run
+past, mice<->mouse+pl); multi-word aspect (has run, is running) is the
auxiliary chain handled by surface_tense (Phase 4). Agreement rides the
learned-participation marker space (Task 7.2). The analyze/inflect pair is a
CHECKED bijection -- inflect(analyze(s)) == s over the vocabulary, fail loud
on divergence (Task 7.4). Table-driven initially; spelling rules (drop-e,
y->ies, consonant doubling) are the documented growth path, caught loudly by
the round-trip gate when a new form does not round-trip."""

_VERB_PAST = {"run": "ran", "eat": "ate", "go": "went", "leave": "left"}     # extend as lexicon grows
_VERB_PAST_INV = {v: k for k, v in _VERB_PAST.items()}
_NOUN_PLURAL = {"mouse": "mice", "goose": "geese", "child": "children"}
_NOUN_PLURAL_INV = {v: k for k, v in _NOUN_PLURAL.items()}

def analyze(surface, kind="verb"):
    """surface -> (lemma, when, agr). verb: when=(tense, aspect_chain),
    agr in {'3sg','base'}. noun: when=None, agr in {'pl','sg'}. Deterministic;
    the round-trip gate (Task 7.4) proves the inverse over the vocabulary."""
    if kind == "noun":
        if surface in _NOUN_PLURAL_INV:
            return _NOUN_PLURAL_INV[surface], None, "pl"
        if surface.endswith("s") and not surface.endswith("ss"):
            return surface[:-1], None, "pl"
        return surface, None, "sg"
    if kind == "verb":
        if surface in _VERB_PAST_INV:                       # irregular past (ate -> eat)
            return _VERB_PAST_INV[surface], ("PAST", []), "base"
        if surface.endswith("ed") and len(surface) > 3:     # regular past (walked -> walk)
            return surface[:-2], ("PAST", []), "base"
        if surface.endswith("s") and not surface.endswith("ss"):  # 3sg present (eats -> eat)
            return surface[:-1], ("PRESENT", []), "3sg"
        return surface, ("PRESENT", []), "base"
    raise ValueError(f"unknown kind {kind!r}")

def inflect(lemma, when, agr, kind="verb"):
    """Inverse of analyze: (lemma, when, agr) -> surface."""
    if kind == "noun":
        return _NOUN_PLURAL.get(lemma, lemma + "s") if agr == "pl" else lemma
    if kind == "verb":
        tense = (when or ("PRESENT", []))[0]
        if tense == "PAST":
            return _VERB_PAST.get(lemma, lemma + "ed")
        if agr == "3sg":
            return lemma + "s"
        return lemma                                        # PRESENT base / FUTURE (will carries it)
    raise ValueError(f"unknown kind {kind!r}")
```
(DRY: if `surface_tense` also needs `_VERB_PAST`, define it once here and import it there, or vice versa — one irregular table, two consumers.)
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add bijective surface<->lemma morphology paradigm`

### Task 7.2: Agreement marker shares the collapsed-POS (participation) space

**Files:** Modify `bin/morphology.py` (agreement section); Test `test/test_morphology.py` (extend).

The agreement signal is **the subject NP's collapsed-POS category** — the same learned-participation category that distinguishes count-noun from pronoun/mass. It is *not* a separate declared feature: the category id from `participation.learned_collapse` (`bin/participation.py:172`) **is** the agreement marker id, bound on the composing `lift` operator via `absorb(..., marker_id=cat_id)` (`bin/Layers.py:2011`) and recovered via `emit` / `canonical_marker`. So "is this NP a count noun?" and "does the verb take 3sg `-s`?" draw from one space.

- [ ] **Step 1: Failing test** (append) — two NP symbols with different participation collapse to different category ids; the count-noun-singular category drives 3sg:
```python
def test_agreement_category_is_the_collapsed_pos():
    from participation import learned_collapse
    # build the role-collapsed grammar per the loader idiom (Task 1.3);
    collapse = learned_collapse(grammar)                 # symbol -> category id
    # a count-noun-like subject and a pronoun-like subject land in different categories:
    # assert collapse[count_noun_sym] != collapse[pronoun_sym]
    # and the count-singular category maps to 3sg so the verb inflects with -s:
    from morphology import agr_from_category, inflect
    agr = agr_from_category(collapse[count_noun_sym], number="sg")
    assert agr == "3sg" and inflect("eat", ("PRESENT", []), agr, "verb") == "eats"
def test_agreement_marker_round_trips_through_operator():
    from Language import LiftLayer
    op = LiftLayer()
    op.absorb(torch.randn(1,1,4), torch.randn(1,1,4), marker_id=("CAT", 7))   # category id as marker
    assert op.emit(marker_id=("CAT", 7)) == ("CAT", 7)                        # exact replay
    assert ("CAT", 7) in op.bound_markers()
```
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — add to `bin/morphology.py`:
```python
def agreement_category(subject_symbol, collapse):
    """The subject NP's collapsed-POS category id (from
    participation.learned_collapse). This id IS the agreement marker: subjects
    the collapse groups as count-noun-like carry the id the verb operator
    absorbs to license 3sg. One space for 'what category is this NP' and 'what
    agreement does the verb take' -- no separate declared agreement feature."""
    return collapse.get(subject_symbol)

# category -> surface agreement key. The count-noun-singular category licenses
# 3sg -s; everything else is base. Learnable; table-driven initially. The set
# of "count-singular" category ids is supplied by the collapse (the categories
# whose members participate like singular count nouns).
def agr_from_category(category_id, *, number, count_singular_categories=frozenset()):
    if number == "sg" and category_id in count_singular_categories:
        return "3sg"
    return "base"
```
Wire it at the `lift(subject, verb)` analysis site: `lift_op.absorb(subject, verb, marker_id=agreement_category(subject_symbol, collapse))`; on synthesis, `lift_op.emit(marker_id=route_recorded)` (exact) or `canonical_marker()` (generation) yields the category, which `agr_from_category` turns into the `agr` key `inflect` consumes.
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: agreement marker shares the collapsed-POS participation space`

### Task 7.3: `MorphologyLayer` boundary transform + analyzer/synthesizer integration

**Files:** Modify `bin/morphology.py` (add `MorphologyLayer`); Modify `bin/perceptual_analyzer.py` (`MeronymicAnalyzer.push` `~263`, `synthesize_tree` `~427`); Test `test/test_morphology.py` (extend).

`MorphologyLayer` is a **boundary transform** (like `SymbolizeLayer`, and like the `surface_tense` / `bind_resolver` helpers), invoked at the PS↔SS boundary — **not** a slab-fold `GrammarLayer` (so it is *not* in `GRAMMAR_LAYER_CLASSES`; it has no `op(left,right)` reduction role). It threads `.what`/`.when`/`.where`/agreement and the marker route.

- [ ] **Step 1: Failing test** (append)
```python
from morphology import MorphologyLayer
def test_layer_forward_factors_and_reverse_rebuilds():
    ml = MorphologyLayer()
    lemma, when, agr, route = ml.forward("eats", kind="verb", route=42)
    assert lemma == "eat" and when[0] == "PRESENT" and agr == "3sg" and route == 42
    assert ml.reverse(lemma, when, agr, kind="verb") == "eats"      # generation path
def test_reverse_prefers_exact_route_replay():
    ml = MorphologyLayer()
    # route resolves to the exact recorded surface; it is returned verbatim:
    out = ml.reverse("eat", ("PRESENT", []), "3sg", kind="verb",
                     route=42, marker_resolver=lambda r: "eats")
    assert out == "eats"
def test_reverse_fails_loud_on_divergence():
    ml = MorphologyLayer()
    with __import__("pytest").raises(AssertionError):
        ml.reverse("eat", ("PAST", []), "base", kind="verb",         # inflect -> 'ate'
                   route=99, marker_resolver=lambda r: "eaten")       # route -> 'eaten' (mismatch)
```
- [ ] **Step 2: Run, verify it fails.**
- [ ] **Step 3: Implement** — add `MorphologyLayer` to `bin/morphology.py`:
```python
class MorphologyLayer:
    """Surface<->deep morphology boundary (paired with SymbolizeLayer). Runs
    BEFORE the SS write (forward/analysis) and AFTER the SS read
    (reverse/synthesis), holding .where (position) fixed.

      forward(surface, *, kind, route) -> (lemma, when, agr, route)
        analyze surface -> lemma (the .what written to the SS codebook) +
        when (-> WhenRangeEncoding via the Tense/Aspect layers) + agreement
        category (-> learned marker absorbed on the operator, Task 7.2). The
        caller passes the recorded PS atom id as ``route`` for exact replay.

      reverse(lemma, when, agr, *, kind, route=None, marker_resolver=None)
        -> surface
        Prefer the route-recorded PS atom (exact replay via marker_resolver);
        else inflect(lemma, when, agr) (generation). When both exist they MUST
        agree -- a mismatch raises (fail loud); the exact route is the truth,
        the divergence is a paradigm-table learning signal."""
    def forward(self, surface, *, kind="verb", route=None):
        lemma, when, agr = analyze(surface, kind=kind)
        return lemma, when, agr, route
    def reverse(self, lemma, when, agr, *, kind="verb", route=None, marker_resolver=None):
        generated = inflect(lemma, when, agr, kind=kind)
        if route is not None and marker_resolver is not None:
            exact = marker_resolver(route)
            if exact is not None:
                assert exact == generated, (
                    f"morphology divergence: inflect={generated!r} route={exact!r}")
                return exact
        return generated
```
Integration (concrete hooks):
- **Analysis** — in `MeronymicAnalyzer.push` (`bin/perceptual_analyzer.py:263`), after computing the surface atom + `.where`, call `MorphologyLayer.forward(text, kind=..., route=part_id)`; write `lemma` as the deep symbol, set `.when` from `when` (via the Tense/Aspect rotation), and `absorb` the agreement category marker on the composing operator. `.where` is unchanged.
- **Synthesis** — in `synthesize_tree` (`bin/perceptual_analyzer.py:427`), where the marker is resolved (`_resolve_marker(op.emit(), marker_resolver)`), call `MorphologyLayer.reverse(lemma, when, agr, route=recorded_ps_id, marker_resolver=marker_resolver)` to realize the exact surface form (replay) or generate it (`inflect`).
- [ ] **Step 4: Run, verify it passes.**
- [ ] **Step 5: Commit** (Alec) — `feat: add MorphologyLayer boundary transform + analyzer/synthesizer hooks`

### Task 7.4: Exactness round-trip gate (the invariant)

**Files:** Test `test/test_morphology.py` (extend). The guarantee the whole phase exists for.

- [ ] **Step 1: Test** — property test over a fixture PS vocabulary: for every surface atom, `MorphologyLayer.reverse(*MorphologyLayer.forward(s, kind), kind) == s` and `inflect(analyze(s)) == s`; **fail loud** on any divergence. Include the spec/Phase-5 forms (`eats`, `ran`, `cats`, `mice`) so this gate guards the fixtures.
- [ ] **Step 2–4:** Run; green when Tasks 7.1–7.3 hold. Any new vocabulary that does not round-trip raises here (the loud signal to extend the paradigm tables).
- [ ] **Step 5: Commit** (Alec) — `test: lock morphology exact round-trip invariant`

---

## Self-Review Against Spec

| Spec section | Covered by |
|---|---|
| Op 1: PREPOSITION (marker-headed, NP/VP/S, doesn't decide relation, permissive) | Tasks 1.1–1.4 |
| Op 2: Contextual BIND (`LIFT(BIND,VP)`$\to$`LIFT(resolved_ref,VP)`, subject/object control, ranks locality+licensing+participation, binds constructed NP, no free binding, **wired into the fold**) | Tasks 2.1–2.4, 5.1 |
| `.when` (zero-centered, two-endpoint range, default present, signed; **2-dim endpoint-sum mirroring `.where`** per user directive) | Tasks 3.1–3.3 |
| Tense = phase rotation; PRESENT/PAST/FUTURE | Tasks 4.2, 5.1 |
| Aspect = SIMPLE/PERFECT/PROGRESSIVE; perfect-is-aspect | Tasks 4.2, 5.1 |
| Surface rewrite (table-driven; `ran`/`is running`/`has run`/`had been running`/`did run`/`will run`) | Task 4.1 |
| Phases 1–6 ordering (PREP$\to$BIND$\to$`.when`$\to$tense/aspect$\to$fixtures$\to$re-evaluate) | Plan phases mirror exactly; deps respected |
| Open Qs 1–5 | Decisions table (permissive PREP; parse-time BIND wired into fold; `will`$\to$FUTURE; `.when` 2-dim mirroring `.where`; before-LIFT C-tier) |
| **Morphology bridge (user-added, beyond the spec's 6 phases)** — SS codebook lemma-only; exact percept↔symbol reconstruction; agreement = learned collapsed-POS marker | Phase 7, Tasks 7.1–7.4 (bijective `analyze`/`inflect`; `agreement_category` shares `participation.learned_collapse`; `MorphologyLayer` boundary + analyzer/synthesizer hooks; round-trip exactness gate) |

**Intentionally deferred (per "note as a hook, don't build"):** per-marker PREPOSITION gating; `MODAL(will, X)`; relative-gap/wh-gap/pronoun-discourse BIND. **Not deferred:** BIND is wired into the live fold (Task 2.4); the locality policy is the first cut and licensing-in-fold is the documented participation refinement (Risk #2). The deviation from the spec's "constant-norm multi-frequency `.when`" is intentional (user directive: mirror `.where` in 2 dims).

## Verification

- **Per-phase node IDs:** `test/test_grammar_preposition.py`; `test/test_contextual_bind.py`; `test/test_when_range_encoding.py`; `test/test_tense_aspect.py`; `test/test_when_grammar_rules.py`; `test/test_grammar_fixtures.py`. All via `.venv/bin/python -m pytest <nodeid> -v`.
- **`.when` round-trip demo (read-only):**
  `.venv/bin/python -c "import sys; sys.path.insert(0,'bin'); from Spaces import WhenRangeEncoding; e=WhenRangeEncoding(64,2); print('present-perfect', e.decode(e.encode_range(-1,0))); print('PAST(present)==past', e.decode(e.rotate(e.encode_range(0,0),-1.0)))"`
  Expected: `present-perfect (tensor(-1.), tensor(0.))` and `PAST(present)==past (tensor(-1.), tensor(-1.))` (signed decode + phase rotation, 2-dim key).
- **BIND wiring demo (read-only):** after Task 2.4, the `test_reduction_layer_stashes_live_slab_on_bind` test proves `set_bind_context` is called from `BinaryStructuredReductionLayer.forward` and the `(BIND, run)` pair resolves to the nearest-left constituent.
- **Regression:** Task 6.2 — `test_basicmodel.py` failures ≤ baseline; `test_role_collapsed_grammar.py` green.

## Risks / Uncertainties

1. **Enabling `nWhen=2` widens the embedding by 2 (low impact).** `muxedSize = nWhat + nWhere + nWhen`; the `.where` slot index is `-(nWhere+nWhen)+i`. Going $0 \to 2$ adds the `.when` block to the muxed `[what|where|when]` layout — the same magnitude as the existing `.where` block, not the large ripple a wide encoding would cause. *Mitigations:* Task 3.3 guards mux/demux at width 2 before the flip; Task 6.1 isolates the flip; Task 6.2 runs width-sensitive suites. *Residual:* a checkpoint trained at `nWhen=0` is shape-incompatible — verify `<autoload>` is false in `MentalModel.xml` before training (flag to Alec).
2. **BIND fold policy is locality (first cut), not full control.** Task 2.4 wires `set_bind_context` into the live fold and resolves nearest-left — this handles both spec fixtures (`Alice` for "wants to run"; `Bob` for "persuaded Bob to run"). **Licensing-based reranking inside the fold** (want=subject-control skipping a nearer object) needs the governing verb's lemma at the reduction site; that is carried by `bind_resolver` + learned participation as the refinement, unit-tested via the participant-list path. Also, a *covert* BIND constituent must be present in the stream for the fold to resolve it — the focused fixtures supply it explicitly (spec Phase 5 staging); corpus-scale empty-category positing is the surface-analyzer's job, out of scope for this milestone.
3. **Tense/aspect tier (S vs C) resolved to C.** They modify the VP `.when` before the subject lift (also C-tier), and reduction runs C before S, so ordering holds. If a future need applies tense to a fully-composed S, re-tag `tier='S'` (one line; the mixin is tier-agnostic). Spec says before/after-LIFT are equivalent, so C is correct now.
4. **Aspect must run before tense.** `AspectLayer` decodes reference $r$ from the range's END endpoint; if tense already shifted it, PERFECT/PROGRESSIVE compound. Fixtures apply aspect innermost-first (matching `PAST(PERFECT(PROGRESSIVE(run)))`); the rule ordering must preserve this if the learned router reorders — documented in the docstrings, locked by fixtures.
5. **`surface_tense` is a table + suffix heuristic.** Irregulars outside the table mislemmatize (`swam`$\to$`swam`). Acceptable for the 8 fixtures ("table-driven initially"); growth path is extending the base-verb table.
6. **Single-frequency `.when` aliasing at the period seam.** Center decodes mod the period ($2\pi/dt$); with `maxT=64` and times in $[-2, +0.25]$ the angle is $\le 0.2$ rad — far from $\pm\pi$. Durations stay below the half-period (`is_recoverable` guards this). For larger times raise `maxT`. This mirrors `.where`'s single-frequency recoverability contract.

## Critical Files

- `bin/Spaces.py` — `WhenRangeEncoding` (replaces `WhenEncoding` `394-453`); construction `:6245`; demux reverse `:6100`; `SubSpace` `:4249`.
- `bin/perceptual_analyzer.py` — `EndpointSumWhere` `:27` (the 2-dim span model `.when` mirrors).
- `bin/Language.py` — new layers after `LowerLayer` ~`2490`; `GRAMMAR_LAYER_CLASSES` `:3474`; `_OPERATOR_SURFACE_SCHEMAS` `:3510`; auto-wiring `_wire_signal_router_grammar_ops` `:8921`, `_resolve_rule_layer` `:9000`, `_BinaryGrammarOpAdapter` `:3660`; **BIND wiring** in `BinaryStructuredReductionLayer.forward` `~5300` (`_stacked_reduced` `:5221`); fold `LanguageLayer.compose` `:3810`.
- `bin/Layers.py` — `GrammarLayer` `:1611`; SurfaceSchema T1–T5 `:1546`.
- `bin/bind_resolver.py`, `bin/surface_tense.py`, `bin/morphology.py` — new pure-logic modules (+ `MorphologyLayer` boundary transform).
- `bin/participation.py` — `learned_collapse` `:172` (collapsed-POS category = agreement marker space); `bin/Layers.py` — `absorb` `:2011` / `emit` `:2069` (marker bind + exact-replay).
- `bin/perceptual_analyzer.py` — `synthesize_tree` `:427`, `MeronymicAnalyzer.push` `:263` (morphology hooks); `EndpointSumWhere` `:27`.
- `data/role_collapsed.grammar` — new rule pairs; `data/MentalModel.xml` — `nWhen=2`.
