# Handoff: Step 4 — PiLayer / SigmaLayer ownership flip

**Date.** 2026-04-25
**Owner.** Alec
**Status.** Not started.  Step 3 (bivector activation in ConceptualSpace)
landed and was merged into `main` as commit `67bfdf3`.

---

## What this hands off

Step 4 of the lift / lower / bivector refactor:
- Plan: [2026-04-24-lift-lower-bivector-refactor.md §Step 4](2026-04-24-lift-lower-bivector-refactor.md)
  (lines 329–375).
- Spec: [2026-04-24-lift-lower-bivector-design.md](../specs/2026-04-24-lift-lower-bivector-design.md)
  §B-summary (lines 1202–1213) names which space owns which layers
  after the flip.
- Doc: [Logic.md §8 *The level-crossing axis*](../Logic.md) and
  [Spaces.md *ConceptualSpace* / *SymbolicSpace*](../Spaces.md)
  carry the architectural definition the handoff implements.  Both
  were updated 2026-04-25 to reflect the framing this handoff
  executes (C → S = AND = Pi.forward, S → C = OR = Sigma.forward).

---

## Architectural framing (resolved)

The earlier draft of this handoff treated the C ↔ S relationship as
a single layer with two directions and asked whether the new Sigma
and Pi should share weights.  The user's clarification supersedes
that:

> "S is the N-way AND of C, C is the N-way OR of S.  I don't think
> any weight sharing is necessary, but as we talked about, we may
> wish to select only two operands in those methods to implement
> methods for the grammar.  Both sigma and pi are invertible."

This resolves the central ambiguity:

1. **Direction-to-math binding.**
   - C → S forward = N-way AND aggregation = `PiLayer.forward`
     (multiplicative, log domain).
   - S → C forward = N-way OR aggregation = `SigmaLayer.forward`
     (additive, tanh domain).
2. **No weight sharing.**  Pi and Sigma are independent layers with
   independent weights; each is invertible via its own internal
   `InvertibleLinearLayer`.
3. **Each layer is invertible.**  `Pi.reverse` undoes Pi's AND
   (multiplicative inverse).  `Sigma.reverse` undoes Sigma's OR
   (additive inverse).  Round-trip exactness is *per-layer*, not
   across the Pi-↔-Sigma pair.
4. **Optional binary-operand mode for grammar.**  The same
   primitives specialize to two-operand calls:
   $\operatorname{Pi}(a, b) = a \land b$ and
   $\operatorname{Sigma}(a, b) = a \lor b$.  Grammar method bodies
   select two operands and the layer collapses to a binary AND or
   OR.  This is a follow-on convenience; the bulk N-ary projection
   semantics are the load-bearing case for Step 4.

**Acceptance assertion** (replaces the parent plan's brittle "fwd
== reverse" form):
- Each layer round-trips through *its own* inverse:
  `pi.reverse(pi.forward(c)) ≈ c`,
  `sigma.reverse(sigma.forward(s)) ≈ s`.
- Composition `sigma.forward ∘ pi.forward` is **not** identity in
  general (the two layers do different math on different weights);
  that test is dropped from the parent plan.

---

## Pre-state (verified 2026-04-25)

Step 3 has landed:
- `ConceptualSpace.activation_signed()` shim is in
  [bin/Spaces.py](../../bin/Spaces.py) (line 5866 area).
- TetralemmaPolicy XML block + per-space override markers are in
  [data/MentalModel.xml](../../data/MentalModel.xml).
- `XMLConfig.tetralemma_policy(space_name)` resolver is in
  [bin/util.py](../../bin/util.py).
- New tests in [test/test_conceptual_bivector.py](../../test/test_conceptual_bivector.py)
  — 14/14 pass.
- Acceptance sweep at Step 3: **212 passed**.  Post-merge sanity:
  68/68 on the lift/lower/quaternary subset.

Current ownership (read from
[bin/Spaces.py](../../bin/Spaces.py)):

| Space | Layer attribute | Layer type | Direction it currently encodes |
|---|---|---|---|
| InputSpace | (n/a — embed pipeline) | — | byte → percept |
| PerceptualSpace | (none) | — | percept ingestion |
| ConceptualSpace | `self.sigma` | SigmaLayer | P → C synthesis (correct) |
| SymbolicSpace | `self.layer` | **PiLayer** | C → S projection (multiplicative) |
| OutputSpace | (its own) | — | downstream |

Active-code consumers reaching the C ↔ S layer by attribute name
(`grep` results):

- [bin/Models.py:3669](../../bin/Models.py) — `pi_layer = self.symbolicSpace.layer`
- [test/test_reasoning.py:283](../../test/test_reasoning.py) —
  `D = model.symbolicSpace.layer.nOutput`

These are the only two consumers reaching in by name.  Pipeline.py
(146 lines) is glue (`ReverseModuleAdapter`, `GrammarMergeGlue`)
and does not reference layer attributes; data flow goes through
`Space.forward` / `Space.reverse`.

---

## Target ownership (matches doc/Logic.md and doc/Spaces.md)

After Step 4:

| Space          | Owns                                                         |
|----------------|--------------------------------------------------------------|
| PerceptualSpace | `PiLayer` (`self.pi`, analysis: P → sub-percept, deferred per §O3) |
| ConceptualSpace | `SigmaLayer` (`self.sigma`, P → C, OR, **existing**), `PiLayer` (`self.pi`, **new**, C → S, AND) |
| SymbolicSpace   | `SigmaLayer` (`self.sigma`, **new**, S → C, OR)               |

Movement summary:
- The PiLayer that currently lives at `SymbolicSpace.layer` **moves**
  to `ConceptualSpace.pi`.  This preserves trained C → S weights —
  no retraining is required for the C → S forward path.
- A **new** SigmaLayer is added at `SymbolicSpace.sigma` for the
  S → C forward direction.  Initial weights from the existing
  `SymbolicSpace.layer.reverse` (PiLayer's inverse) — see §Sequencing
  below for whether this is a copy or a fresh init.
- A **new** PiLayer is added at `PerceptualSpace.pi` (deferred per
  §O3 — constructor only; no Pipeline wiring yet).

---

## What Step 4 changes

1. **Add `PerceptualSpace.pi` (PiLayer).**
   - Constructor in [bin/Spaces.py](../../bin/Spaces.py) at
     `PerceptualSpace.__init__` (around line 4909).
   - `nonlinear=True`, `invertible=True`, `monotonic=True` to match
     the bivector path landed in Step 3.
   - No Pipeline wiring; the layer sits dormant until sub-perceptual
     structure exists (per spec §O3 *defer*).

2. **Move the C → S PiLayer to ConceptualSpace.pi.**
   - Construct in `ConceptualSpace.__init__` (around line 5781):
     ```python
     self.pi = PiLayer(nConceptDim, nSymbolDim,
                       invertible=True, monotonic=True,
                       nonlinear=nonlinear)
     ```
   - `ConceptualSpace.forward` keeps using `self.sigma` for the
     P → C path (unchanged).
   - The new C → S routing through `self.pi.forward` happens at the
     SymbolicSpace stage — but `SymbolicSpace.forward` calls into
     ConceptualSpace's pi layer via the `conceptualSpace` reference
     it already holds (`SymbolicSpace.__init__` already takes
     `conceptualSpace=conceptualSpace`).

3. **Add the new SigmaLayer at SymbolicSpace.sigma.**
   - Construct in `SymbolicSpace.__init__` (around line 5963):
     ```python
     self.sigma = SigmaLayer(nSymbolDim, nConceptDim,
                             invertible=True, monotonic=True,
                             nonlinear=nonlinear,
                             stable=True, naive=naive,
                             ergodic=ergodic)
     ```
   - **Initial weights**: this is a fresh layer, independent of the
     existing PiLayer's weights.  S → C decomposition behavior will
     differ from the legacy `PiLayer.reverse` numerics.  Tests
     downstream of S → C may shift; see §Risks.

4. **Pipeline routing.**
   - `SymbolicSpace.forward(subspace)` (the C → S path) should now
     call `self.conceptualSpace.pi.forward(c)` instead of
     `self.layer.forward(c)`.
   - `SymbolicSpace.reverse(subspace)` (the S → C path) should now
     call `self.sigma.forward(s)` instead of
     `self.layer.reverse(s)`.
   - The change is **in `bin/Spaces.py` only** — Pipeline.py is
     untouched.  Verify by `grep "\.layer\.\(forward\|reverse\)" bin/`
     after editing.

5. **Backward-compat shim for active-code consumers.**
   - Add a property on SymbolicSpace:
     ```python
     @property
     def layer(self):
         """Deprecated alias for the C → S Pi (now owned by
         ConceptualSpace).
         # XXX Remove in Step 7 consumer migration."""
         return self.conceptualSpace.pi
     ```
   - This keeps [bin/Models.py:3669](../../bin/Models.py) and
     [test/test_reasoning.py:283](../../test/test_reasoning.py)
     working without an immediate edit.  The variable name
     `pi_layer` at Models.py:3669 stays accurate
     (`SymbolicSpace.layer` is still a PiLayer via the shim — it
     just lives elsewhere).

6. **Migrate the legacy 2-arg deprecation-warning callers.**
   - [bin/Language.py:926](../../bin/Language.py) currently calls
     `Ops.lift(left, right)` (legacy 2-arg).  Change to the new
     mode-dispatch form:
     `Ops.lower(left, right, mode='AND')` per the deprecation
     warning's own guidance.
   - [bin/Language.py:934](../../bin/Language.py) currently calls
     `Ops.lower(left, right)` (legacy 2-arg).  Change to:
     `Ops.lift(left, right, mode='OR')`.
   - The deprecation aliases themselves stay (per spec §Q5 with the
     `# XXX` markers); only these two internal call sites migrate.
     Eliminates ~700K warnings per acceptance sweep without any
     behavior change at the call sites.

---

## File map

Files Step 4 will touch:

**Code:**
- `bin/Spaces.py:4909` (`PerceptualSpace.__init__`) — add `self.pi`.
- `bin/Spaces.py:5781` (`ConceptualSpace.__init__`) — add `self.pi`.
- `bin/Spaces.py:5963` (`SymbolicSpace.__init__`) — add `self.sigma`,
  add `layer` property shim, swap forward/reverse to route through
  `self.conceptualSpace.pi` and `self.sigma`.
- `bin/Layers.py` — confirm `PiLayer` and `SigmaLayer` constructors
  do not need new parameters.  Expected: no change.
- `bin/Language.py:926`, `:934` — migrate legacy 2-arg calls.
- `bin/Models.py:3669` — no change required (shim keeps it working).
  Optional cosmetic rename of local `pi_layer`.
- `bin/Pipeline.py` — confirm untouched.

**Docs (updated 2026-04-25, pre-implementation):**
- [doc/Logic.md §8](../Logic.md) — *The level-crossing axis*
  rewritten to the C → S = AND, S → C = OR framing.
- [doc/Spaces.md](../Spaces.md) — ConceptualSpace and SymbolicSpace
  ownership tables updated to the post-Step-4 layout.

**Docs (sync as part of Step 4 implementation):**
- [doc/Architecture.md](../Architecture.md) lines 28–30 — ownership
  table for PerceptualSpace / ConceptualSpace / SymbolicSpace.
  Currently describes pre-Step-4 (SymbolicSpace = PiLayer + Codebook).
  Update at sequencing step 7 (after the property shim lands and
  the new ownership is observable).
- [doc/Reasoning.md](../Reasoning.md) lines ~137 and ~151 —
  references "SymbolicSpace's PiLayer" projecting concepts → symbols.
  Update to point at `ConceptualSpace.pi` once the migration in
  sequencing step 4 is complete.
- [doc/Params.md](../Params.md) line 210 — describes
  `PiLayer(nConcepts, nSymbols, ...)` as a SymbolicSpace layer.
  Update to note the ownership move to ConceptualSpace.

Files Step 4 must NOT touch (until Step 7):
- Test files reading `.layer` — protected by the property shim.

---

## New tests required

`test/test_pi_sigma_ownership.py` covering:

1. **Ownership assertions** — each space exposes the documented
   layer attribute and type:
   ```python
   assert isinstance(model.perceptualSpace.pi, PiLayer)
   assert isinstance(model.conceptualSpace.sigma, SigmaLayer)
   assert isinstance(model.conceptualSpace.pi, PiLayer)
   assert isinstance(model.symbolicSpace.sigma, SigmaLayer)
   ```
2. **Per-layer round-trip recovery (no cross-layer assumptions)** —
   ```python
   c = torch.randn(1, N, concept_dim).clamp(-1+eps, 1-eps)
   c_back = model.conceptualSpace.pi.reverse(
                 model.conceptualSpace.pi.forward(c))
   torch.testing.assert_close(c, c_back, atol=1e-5, rtol=1e-4)

   s = torch.randn(1, N, symbol_dim).clamp(-1+eps, 1-eps)
   s_back = model.symbolicSpace.sigma.reverse(
                 model.symbolicSpace.sigma.forward(s))
   torch.testing.assert_close(s, s_back, atol=1e-5, rtol=1e-4)
   ```
3. **No weight sharing** —
   ```python
   assert model.symbolicSpace.sigma.layer is not model.conceptualSpace.pi.layer
   ```
4. **Backward-compat shim** —
   ```python
   assert isinstance(model.symbolicSpace.layer, PiLayer)
   _ = model.symbolicSpace.layer.nOutput   # no AttributeError
   ```
5. **Optional binary-operand sanity** (mark as
   `@unittest.skipIf(not BINARY_MODE, ...)` until the binary form
   lands) —
   ```python
   a = torch.tensor([0.7, 0.3])
   b = torch.tensor([0.5, 0.9])
   pi_ab = pi_layer.binary(a, b)        # ≈ a ∧ b (per-element min)
   sigma_ab = sigma_layer.binary(a, b)  # ≈ a ∨ b (per-element max)
   ```

---

## Open spec resolutions to honor

The spec already commits the following — no fresh decisions
needed before starting:

- **§B-summary** — ownership table after Step 4 matches
  PerceptualSpace owns Pi, ConceptualSpace owns Sigma + Pi,
  SymbolicSpace owns Sigma.
- **§O3** — PerceptualSpace SigmaLayer DECIDED *defer*.  Step 4
  adds PerceptualSpace's PiLayer only.
- **§B5** — `monotonic=True` propagation: all Pi/Sigma
  constructors in Step 4 set `monotonic=True`.
- **§Q5** — deprecation aliases stay (do **not** remove the
  legacy 2-arg `Ops.lift` / `Ops.lower` aliases); only the call
  sites at [Language.py:926, :934](../../bin/Language.py) migrate.

---

## Sequencing within Step 4

Smallest blast first:

1. **Add `PerceptualSpace.pi` (PiLayer).**  Additive; no consumers.
   No tests affected.
2. **Add `ConceptualSpace.pi` (PiLayer) — new, with fresh weights.**
   At this stage `SymbolicSpace.layer` (the existing PiLayer)
   still owns the C → S forward path.  ConceptualSpace.pi sits
   dormant.  Ownership tests pass; no behavior change.
3. **Add `SymbolicSpace.sigma` (SigmaLayer) — new, with fresh
   weights.**  At this stage SymbolicSpace.layer.reverse still owns
   S → C.  SymbolicSpace.sigma sits dormant.  Ownership tests pass;
   no behavior change.
4. **Migrate the trained PiLayer.**  Two options:
   - **(a)** Have `ConceptualSpace.pi` and `SymbolicSpace.layer`
     both reference the same `PiLayer` instance.  This preserves
     trained weights; the move is by-reference.
   - **(b)** Delete `ConceptualSpace.pi` (created fresh in step 2)
     and re-construct it as a reference to the existing
     `SymbolicSpace.layer`.  Then SymbolicSpace stops owning it;
     consumer shim (step 6) makes `SymbolicSpace.layer` an alias.
   Pick **(b)** for cleanliness; (a) is the same shape but two
   names for one instance is harder to reason about.
5. **Flip C → S routing in `SymbolicSpace.forward` to call
   `self.conceptualSpace.pi.forward`.**  No numeric change since
   the same trained PiLayer is being called via a different
   reference.
6. **Flip S → C routing in `SymbolicSpace.reverse` to call
   `self.sigma.forward`.**  This **is** a numeric change —
   `Sigma.forward` is additive, `PiLayer.reverse` was
   multiplicative-inverse.  Expect tests downstream of S → C to
   shift.  Run the full sweep here; if it stays green, proceed.
   If it regresses materially, escalate.
7. **Add the `SymbolicSpace.layer` property shim.**  Existing
   consumers continue to work via the alias.
8. **Migrate the two legacy 2-arg `Ops.lift` / `Ops.lower` callers
   in [bin/Language.py:926, :934](../../bin/Language.py).**
   Eliminates the deprecation-warning storm.
9. **Add `test/test_pi_sigma_ownership.py`.**  Asserts the new
   shape and per-layer round-trip recovery.

Steps 1–5 and 7 are reversible without touching tests.  Step 6 is
the only behavior change; isolate it as its own commit so a
revert is surgical if the sweep regresses.

---

## Verification commands

Per memory: tests via `basicmodel/.venv/bin/python -m pytest`.
Never run `make train` locally.  User manages git commits.

```bash
# unit-level (after each sequencing step)
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_pi_sigma_ownership.py -v
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_conceptual_bivector.py -v   # should still be 14/14
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_ops_lift_lower.py -v        # should still be 40/40

# acceptance sweep (after step 6 of sequencing — the behavior change)
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_grammar_derivation.py \
  basicmodel/test/test_toy_grammar.py \
  basicmodel/test/test_subspace_context.py \
  basicmodel/test/test_head_divergence.py \
  basicmodel/test/test_serial_mode_integration.py \
  basicmodel/test/test_quaternary_corners.py \
  basicmodel/test/test_partition_grammar_rewrite.py \
  basicmodel/test/test_mask_dispatch.py \
  basicmodel/test/test_mental_model.py \
  basicmodel/test/test_reasoning.py
```

Pre-Step-4 baseline of the sweep: **212 pass** (Step 3 acceptance).
Step 4 acceptance: same 212 pass (or a documented count after
the S → C math change) plus the new
`test/test_pi_sigma_ownership.py` cases.  Deprecation-warning
count should drop to 0 after the Language.py migration in
sequencing step 8.

---

## Risks / things to watch

- **S → C numeric change.**  Sequencing step 6 is the load-bearing
  behavior change.  `PiLayer.reverse` (multiplicative inverse) and
  `SigmaLayer.forward` (additive OR) compute different quantities.
  Tests that assert specific S → C numerics will need to be
  re-baselined.  If many regress, the cleanest revert is to keep
  step 5 (C → S re-routing through `ConceptualSpace.pi`) and skip
  step 6, leaving S → C on the legacy path.

- **Construction order in `ModelFactory`.**  ConceptualSpace must
  be built before SymbolicSpace so SymbolicSpace's `conceptualSpace`
  reference can reach `conceptualSpace.pi`.  Verify in
  [bin/Models.py](../../bin/Models.py) `ModelFactory.create`
  before sequencing step 4.

- **`processSymbols` interaction with new pi.forward.**
  ConceptualSpace.reverse currently calls `self.dereference(y)`
  when `processSymbols=True`.  The new `ConceptualSpace.pi.forward`
  is called from `SymbolicSpace.forward` (not from
  `ConceptualSpace.reverse`), so the two paths shouldn't conflict —
  but verify no test exercises both simultaneously.

- **Bivector + monotonic propagation.**  After Step 3, ConceptualSpace
  activations are bivector with the catuskoti layout.  All new
  Pi/Sigma constructors must set `monotonic=True` to match.
  Otherwise the round-trip recovery test will fail in bitonic mode.

- **PerceptualSpace constructor signature.**  Audit before adding
  `self.pi`: the constructor may not currently expose `nonlinear`
  or `invertible` flags directly; if the defaults differ from
  ConceptualSpace, the new PiLayer's behavior may diverge from
  expectations.

- **Models.py:3669 variable name.**  After the shim, the local
  `pi_layer` still IS a PiLayer (just owned at a different space),
  so the name remains accurate.  No rename needed.

---

## Last task: create the Step 5 handoff

After Step 4 lands and the acceptance sweep passes, write the Step
5 handoff at
`doc/plans/<date>-step5-ops-wiring-handoff.md` covering the
parent plan §Step 5 (Wire `Ops.lift` / `Ops.lower` into the layer
forward / reverse paths).  The Step 5 handoff should:

1. Reference Step 4's outcome (trained PiLayer migrated to
   ConceptualSpace.pi; new SigmaLayer at SymbolicSpace.sigma; both
   invertible).
2. Describe what changes in
   [bin/Layers.py](../../bin/Layers.py) `PiLayer.forward` /
   `SigmaLayer.forward` to call `Ops.lift` / `Ops.lower` with the
   appropriate `mode` and `kind`, instead of running the math
   directly inside the layer body.
3. Define the binary-operand convenience form (per the architectural
   framing above): `pi.binary(a, b) ≈ a ∧ b` and
   `sigma.binary(a, b) ≈ a ∨ b`, which the grammar uses for binary
   rule application.  Decide whether `binary` is a method on the
   layer or a free function; decide whether it shares weights with
   the N-ary forward.
4. Identify the consumers (grammar method bodies in
   [bin/Language.py](../../bin/Language.py)) that should switch to
   the binary form.
5. Acceptance: existing Pi/Sigma tests pass (40/40 lift_lower,
   plus Step 4's ownership tests); new binary-form tests pass;
   no new deprecation warnings introduced.
6. Conclude with the same recursion: "after Step 5 lands, write
   the Step 6 handoff at <path>".

The Step 5 handoff is a deliverable of this Step 4 work — write
it after the implementation lands and the acceptance sweep is
green, but before declaring Step 4 complete.
