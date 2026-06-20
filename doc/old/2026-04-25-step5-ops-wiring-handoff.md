# Handoff: Step 5 — Wire `Ops.lift` / `Ops.lower` into the layer
forward / reverse paths

**Date.** 2026-04-25
**Owner.** Alec
**Status.** Not started.  Step 4 (PiLayer / SigmaLayer ownership flip)
landed and was verified by the Step 4 acceptance sweep + the new
`test_pi_sigma_ownership.py` (11/11 passing).

---

## What this hands off

Step 5 of the lift / lower / bivector refactor:
- Plan: [2026-04-24-lift-lower-bivector-refactor.md §Step 5](2026-04-24-lift-lower-bivector-refactor.md)
  (lines 378–401).
- Spec: [2026-04-24-lift-lower-bivector-design.md](../specs/2026-04-24-lift-lower-bivector-design.md)
  §B (the unified `Ops.lift` / `Ops.lower` signature with mode dispatch).
- Step 4 handoff: [2026-04-25-step4-pi-sigma-ownership-handoff.md](2026-04-25-step4-pi-sigma-ownership-handoff.md)
  (the immediate predecessor; defines the post-Step-4 ownership this
  handoff builds on).

---

## Pre-state (Step 4 outcome)

Step 4 landed the ownership flip:

- `PerceptualSpace.pi`  — PiLayer (P → sub-percept; dormant per spec §O3).
- `ConceptualSpace.sigma` — SigmaLayer (P → C, OR; existing).
- `ConceptualSpace.pi`  — PiLayer (C → S, AND; **moved** from
  SymbolicSpace; the trained instance lives here now).
- `SymbolicSpace.sigma` — SigmaLayer (S → C, OR; **new**).
- `SymbolicSpace.layer` — `@property` alias for `conceptualSpace.pi`
  (deprecated; consumers migrate in Step 7 of the parent plan).

Both Pi and Sigma are independently invertible; the architectural
framing (Logic.md §8) names Pi = N-way AND, Sigma = N-way OR with no
weight sharing.  Round-trip recovery is per-layer:
`pi.reverse(pi.forward(c)) ≈ c` and
`sigma.reverse(sigma.forward(s)) ≈ s` — verified by
`test/test_pi_sigma_ownership.py::TestPerLayerRoundTrip`.

The deprecation-warning storm from `Ops.lift(left, right)` /
`Ops.lower(left, right)` (legacy 2-arg form) at
[bin/Language.py:926, :934](../../bin/Language.py) was eliminated in
Step 4 by migrating those two call sites to the new mode-dispatch form
with `kind='smooth'` to preserve bit-exact behavior (elementwise
product / arithmetic mean).

---

## What Step 5 changes

The plan summary (parent §Step 5, lines 378–401):

1. `PiLayer.forward` calls `Ops.lower(input, self.weight, mode='AND',
   kind='smooth')` — the smoothed (product / mean) form is the natural
   fit for differentiable pooling, not the strict min/max.
2. `SigmaLayer.forward` calls `Ops.lift(input, self.weight, mode='OR',
   kind='smooth')`.
3. The reverse paths use `inverse=True` against the same body.
4. The hard-lattice variant (`kind='strict'`) is available for
   inference-time / extraction-time use but not used in training
   forward passes (sparse gradients).

Practical scope, given the post-Step-4 layout:

- **`bin/Layers.py` `PiLayer.forward`** currently runs the
  log-domain multiplicative body inline (`_to_mult`, `_from_mult`,
  `self.layer.forward`).  After Step 5, the math goes through `Ops`:
  the same numbers come out, but the formula lives in one place.
- **`bin/Layers.py` `SigmaLayer.forward`** currently runs the
  additive body inline (`atanh` → `self.layer.forward` → `tanh`).
  After Step 5, the math goes through `Ops`.
- **`bin/Layers.py` `PiLayer.reverse` / `SigmaLayer.reverse`** call
  the same `Ops` body with `inverse=True`.

The *behavior* is unchanged.  The change is structural:
duplication-elimination — each of the four layer methods (forward /
reverse × Pi / Sigma) becomes a thin dispatcher into `Ops`.

---

## Binary-operand convenience form

The architectural framing (Step 4 handoff §"Architectural framing
(resolved)") committed to a binary specialization:

> "S is the N-way AND of C, C is the N-way OR of S.  ... we may wish
> to select only two operands in those methods to implement methods
> for the grammar.  Both sigma and pi are invertible."

Concretely, define on each layer (or as a free function paired with
the layer):

```python
PiLayer.binary(a, b)    # ≈ a ∧ b  (per-element AND on regions/points)
SigmaLayer.binary(a, b) # ≈ a ∨ b  (per-element OR on regions/points)
```

**Open decisions for Step 5 (decide before writing code):**

1. **Method on the layer or free function on `Ops`?**
   Recommended: **method on the layer**, since the binary form lives
   inside the same activation domain (the layer's `_to_mult` /
   tanh wrappings) and the layer's weights gate the result.  The free
   `Ops.lift(a, b, mode='OR')` / `Ops.lower(a, b, mode='AND')` already
   exists for the weight-free case; the layer method composes it with
   the activation transform.

2. **Weight-sharing with the N-ary forward?**
   Recommended: **share** — `binary(a, b)` is just the N-ary forward
   with N=2 operands packed into the same weight matrix.  This keeps
   training simple (no separate-binary-weight bookkeeping) and
   matches the grammar use case (binary rules are a strict subset of
   the N-ary projection the layer already learns).

3. **Activation transform around the binary call?**
   Recommended: **same transform as the N-ary forward** (`atanh` /
   `_to_mult` on input, `tanh` / `_from_mult` on output) so the
   binary form is a true specialization, not a parallel
   implementation.

These recommendations are tentative; revisit during Step 5
implementation if the parent plan's binary-operand semantics
suggest otherwise.

---

## Consumers to migrate

The binary form is consumed by grammar method bodies in
[bin/Language.py](../../bin/Language.py).  Pre-Step-5, those methods
hand-roll AND / OR via the post-PiLayer-roundtrip in-space body:

- [bin/Language.py:920–926 `liftForward`](../../bin/Language.py)
  — currently `Ops.lower(left, right, mode='AND', kind='smooth')`
  (post Step 4 migration).  Step 5 swaps the explicit `Ops.lower`
  call for the layer's `pi.binary(left, right)` once `pi` is the
  layer the grammar's lift target.
- [bin/Language.py:932–938 `lowerForward`](../../bin/Language.py)
  — currently `Ops.lift(left, right, mode='OR', kind='smooth')`
  (post Step 4 migration).  Step 5 swaps for `sigma.binary(left,
  right)`.
- [bin/Language.py:928–930 `liftReverse`](../../bin/Language.py),
  [bin/Language.py:936–938 `lowerReverse`](../../bin/Language.py)
  — currently call `Ops.liftReverse` / `Ops.lowerReverse` (legacy
  body inverses).  Step 5 routes these through the layer's
  `binary` invocation with `inverse=True` (or its layer-wrapped
  equivalent).

The grammar method bodies are the single biggest consumer; once
they migrate, the legacy `Ops.liftReverse` / `Ops.lowerReverse`
free functions become candidates for removal (Step 7 of the parent
plan).

---

## File map

Files Step 5 will touch:

**Code:**
- `bin/Layers.py` `PiLayer.forward`, `PiLayer.reverse`,
  `SigmaLayer.forward`, `SigmaLayer.reverse` — replace inline math
  with calls into `Ops`.  Add `PiLayer.binary` and `SigmaLayer.binary`
  (the binary-operand convenience).
- `bin/Language.py` `liftForward`, `liftReverse`, `lowerForward`,
  `lowerReverse` — switch from explicit `Ops.lift` / `Ops.lower`
  with `mode=...` to `pi.binary` / `sigma.binary`.

**Docs (sync as part of Step 5 implementation):**
- [doc/Logic.md](../Logic.md) — extend §8 with the binary
  specialization (one paragraph noting binary is N=2 N-ary).
- [doc/Spaces.md](../Spaces.md) — note `binary` methods on the
  per-space tables.

Files Step 5 must NOT touch:
- The layer constructors (no parameter change; `monotonic`,
  `invertible`, `nonlinear` already cover the configurable surface).
- The Pipeline routing (Step 4 already wired forward / reverse to
  the right per-direction layers; Step 5 changes layer internals
  only).

---

## New tests required

`test/test_ops_layer_wiring.py` (or extend the existing
`test/test_ops_lift_lower.py`) covering:

1. **Numerical equivalence pre / post Step 5.**  Snapshot the layer
   output on a fixed seed and inputs before the wiring change,
   compare after.  `torch.testing.assert_close(out_old, out_new,
   atol=1e-6, rtol=1e-5)`.

2. **Binary form sanity.**
   ```python
   a = torch.tensor([0.7, 0.3])
   b = torch.tensor([0.5, 0.9])
   pi_ab = pi_layer.binary(a, b)        # close to a * b after weights
   sigma_ab = sigma_layer.binary(a, b)  # close to (a + b) / 2 after weights
   ```
   Exact values depend on the layer weights; the test asserts the
   *form* (e.g., AND-like = monotone-decreasing in disagreement,
   OR-like = monotone-increasing in agreement) rather than a fixed
   number.

3. **Round-trip recovery still holds** after the wiring change —
   the existing `test_pi_sigma_ownership.py::TestPerLayerRoundTrip`
   suite is the regression net.

4. **No new deprecation warnings.**  The acceptance sweep's warning
   count should remain at 0 (post Step 4 cleanup).

---

## Acceptance

- 40/40 [test/test_ops_lift_lower.py](../../test/test_ops_lift_lower.py)
  passes.
- 11/11 [test/test_pi_sigma_ownership.py](../../test/test_pi_sigma_ownership.py)
  passes.
- New `test_ops_layer_wiring.py` (binary-form + equivalence) passes.
- Full acceptance sweep (the same set Step 4 ran):
  179 passed + 6 xfailed (or a documented count if the binary form
  surfaces a correctness bug worth fixing in Step 5).
- `grep -nE "Ops\.lift\(|Ops\.lower\(" bin/Layers.py` shows the new
  calls; no inline `_to_mult` / `_from_mult` in `PiLayer.forward`.

---

## Risks / things to watch

- **Numerical drift from path swap.**  The mathematical body is
  identical, but operation order can shift floating-point results
  by ~1e-7.  Use generous `atol` / `rtol` in equivalence tests.

- **Binary-form weight-sharing decision.**  If `binary(a, b)` shares
  weights with the N-ary forward, the grammar's per-rule call has
  the entire N-ary projection in scope.  This may over-mix when
  the grammar wanted a clean two-operand pick.  Acceptable for
  Step 5 if it falls out of the layer wiring; the parent plan §Step
  6 (grammar-driven dispatch) is where this gets exercised.

- **`self.weight` vs `self.layer.weight`.**  `PiLayer` and
  `SigmaLayer` wrap an `InvertibleLinearLayer` (or
  `NonNegativeInvertibleLinearLayer`) at `self.layer`.  The weight
  the grammar's binary form should multiply against is
  `self.layer.weight` (the inner linear), not a separate top-level
  weight.  Verify the existing `forward()` body to extract the
  right reference.

- **`InvertibleLinearLayer.reverse` semantics.**  `Ops.lift` /
  `Ops.lower` with `inverse=True` do the **codebook-search** inverse
  per the Ops module's definition (line 4374 / 4451-4453, raising
  `NotImplementedError` without a codebook).  The layer's reverse
  needs the **linear-algebraic** inverse (the LDU triangular solve
  in the inner layer), which is a different inverse.  Step 5 must
  decide whether to:
  (a) keep the existing `self.layer.reverse(y)` call for the
      linear inverse (Ops only handles the forward body),
  (b) extend `Ops` with a layer-aware inverse hook, or
  (c) accept that the wiring is forward-only and rely on the
      existing reverse path.
  Recommended: **(a)** for Step 5 — match the parent plan's "the
  reverse paths use `inverse=True`" without committing to a new
  inversion abstraction.  Re-evaluate in Step 6.

- **Regression in `test_reasoning.py`.**  The grammar paths under
  RamsifiedModel.xml use `liftForward` / `lowerForward` heavily.
  After binary-form migration, watch for regressions.

---

## Sequencing within Step 5

Smallest blast first:

1. **Add `PiLayer.binary(a, b)` and `SigmaLayer.binary(a, b)`** as
   new methods that compose the existing `forward` body with N=2
   inputs.  Additive; no consumers yet.  Add unit tests.
2. **Replace the inline math in `PiLayer.forward` with `Ops.lower(
   ..., mode='AND', kind='smooth')`.**  Snapshot test confirms
   numerical equivalence.
3. **Replace the inline math in `SigmaLayer.forward` with `Ops.lift(
   ..., mode='OR', kind='smooth')`.**  Snapshot test confirms.
4. **Wire `PiLayer.reverse` / `SigmaLayer.reverse` through `Ops`**
   (with the §Risks note about the linear inverse).
5. **Migrate the four `Language.py` grammar method bodies** to call
   `pi.binary` / `sigma.binary`.  This is the largest behavior
   surface; run the full acceptance sweep after.

Steps 1–4 are reversible with no consumer churn; step 5 is the
behavior-touching commit, isolate it for a clean revert.

---

## Verification commands

Per memory: tests via `basicmodel/.venv/bin/python -m pytest`.
Never run `make train` locally.  User manages git commits.

```bash
# unit-level (after each sequencing step)
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_ops_lift_lower.py -v        # 40/40 baseline
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_pi_sigma_ownership.py -v    # 11/11 baseline
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_ops_layer_wiring.py -v      # new in Step 5

# acceptance sweep (after sequencing step 5 — the grammar migration)
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
  basicmodel/test/test_reasoning.py \
  basicmodel/test/test_pi_sigma_ownership.py
```

Pre-Step-5 baseline of the sweep: **179 passed + 6 xfailed + 11 new
ownership tests = 190 / 6 xfail** (Step 4 acceptance).
Step 5 acceptance: same total (or a documented count if the binary
form surfaces a behavior shift worth keeping).  Deprecation-warning
count should remain at 0 (Step 4 already drained it).

---

## Last task: create the Step 6 handoff

After Step 5 lands and the acceptance sweep passes, write the Step
6 handoff at
`doc/plans/<date>-step6-grammar-cfg-handoff.md` covering the parent
plan §Step 6 (Grammar-driven dispatch via `data/grammar.cfg`).  The
Step 6 handoff should:

1. Reference Step 5's outcome (PiLayer / SigmaLayer wired through
   `Ops`; binary form available for grammar consumption; Language.py
   grammar method bodies migrated).
2. Describe the explicit-op grammar form per parent plan §Step 6
   (lines 405–470): each production's RHS *is* a function call
   naming the operation directly; LHS = produced state; rule table
   consolidates `todo.md`'s in-progress rules.
3. Identify the consumers: the grammar loader, the parser's
   apply-rule step, the codebook initialization in `bin/Spaces.py`
   (reserve category-vector slots).
4. Acceptance: `data/grammar.cfg` parses cleanly; the existing
   grammar derivation tests pass on the new dispatch path; no new
   deprecation warnings introduced.
5. Conclude with the same recursion: "after Step 6 lands, write the
   Step 7 handoff at <path>".

The Step 6 handoff is a deliverable of this Step 5 work — write it
after the implementation lands and the acceptance sweep is green,
but before declaring Step 5 complete.
