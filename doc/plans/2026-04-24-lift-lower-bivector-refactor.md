# Plan: Lift / Lower as the Canonical Level-Crossing Axis, Bivector Lower into ConceptualSpace

**Date.** 2026-04-24
**Owner.** Alec
**Status.** Draft.  Code unchanged; doc preconditions in place
([Logic.md](../Logic.md) §8, [OpsComparison.md](../OpsComparison.md)
§7).

---

## Goal

Refactor the level-crossing operations between PerceptualSpace,
ConceptualSpace, and SymbolicSpace around the **Cantorian lift /
lower axis**:

- **lift** (going up — many → one) = synthesis = union (∨), realized
  at the layer scale by `SigmaLayer`.
- **lower** (going down — one → many) = analysis = intersection (∧),
  realized at the layer scale by `PiLayer`.

Three substantive changes follow:

1. **Pi / Sigma ownership flips** so synthesis is owned by the
   receiving "one" layer and analysis by the producing "many" layer.
2. **`Ops.lift` / `Ops.lower` are renamed and extended** to a single
   mode-dispatched signature
   `Y = f(X1, X2=None, mode='AND'|'OR'|'NOT', inverse=False)`,
   covering point and region scales.
3. **`ConceptualSpace.subspace.activation` becomes bivector** (`[N,
   2]`) so the S → C lower preserves contradiction (`[1, 1]`) vs.
   ignorance (`[0, 0]`) instead of collapsing both to `0` under a
   signed-sum readout.

---

## Why now

- The conceptual model in Logic.md §8 is now stable: lift / lower /
  Cantorian polarity / convexity asymmetry / bivector-lower-for-
  contradiction.
- The current code arrangement (SymbolicSpace owns the `PiLayer` at
  `bin/Spaces.py:5945`) is inverted relative to the Cantorian
  ownership rule.  Building further symbolic features on top of the
  inverted arrangement compounds the cleanup.
- `Ops.lift = ⊙` and `Ops.lower = mean` ([bin/Layers.py:4308–4326](../../bin/Layers.py))
  are correct in body but inverted in label.  Every new caller risks
  reinforcing the inversion.
- `tetralemma_balance_penalty` already exists for the symbolic
  layer; lifting it to per-concept under bivector ConceptualSpace is
  a small extension once the activation shape is right.

---

## Non-goals

- No change to RadMin / RadMax / NOT / NON semantics in §3 / §7 of
  Logic.md.  Pointwise bitonic logic is unchanged.
- No removal or rewrite of `TruthLayer`, `ImpenetrableLayer`, or any
  loss class.  Their inputs may switch shape but their internal
  algorithms stay.
- No multi-region / DNF disjunction at this pass.  Sigma-as-synthesis
  with hard-pair extraction is the closest we get; full DNF stays
  deferred (Logic.md §8 *Convexity asymmetry*).
- No grammar-rule changes.  The S-tier rules `part(S, S)`,
  `equals(S, S)`, etc. continue to call `Basis.part` / `Basis.equal`
  unchanged.
- No retraining or training-loop change required to land Steps 1–3.
  Steps 4–6 affect training pressure and need a fresh local run, not
  a remote retrain (per memory: never run `make train` locally).

---

## Architecture, before and after

### Before (current)

```
PerceptualSpace        — owns nothing in Pi/Sigma family
ConceptualSpace        — owns nothing
SymbolicSpace          — constructs PiLayer for C → S forward pass
                         (bin/Spaces.py:5945, invertible=True)
ConceptualSpace.subspace.activation  — single vector per concept
SymbolicSpace.subspace.what          — bivector (nWhat = 2)

Ops.lift(left, right)   — elementwise product           (analysis-shaped)
Ops.lower(left, right)  — arithmetic mean               (synthesis-shaped)
Ops.conjunction(x, y)   — min / RadMin                  (pointwise lower-AND)
Ops.disjunction(x, y)   — max / RadMax                  (pointwise lift-OR)
```

### After

```
PerceptualSpace        — owns PiLayer (analysis down to features)
ConceptualSpace        — owns SigmaLayer (synthesis from P)
                       — owns PiLayer    (analysis to S)
SymbolicSpace          — owns SigmaLayer (synthesis from C)
ConceptualSpace.subspace.activation  — bivector [N, 2]  (mirrors symbolic)
SymbolicSpace.subspace.what          — bivector [N, 2]  (unchanged)

Ops.lift(X1, X2=None, mode='OR',  inverse=False)   — synthesis dispatcher
Ops.lower(X1, X2=None, mode='AND', inverse=False)  — analysis dispatcher
    mode in {AND, OR, NOT}
    inverse in {False, True}
    X1, X2 each point or region — point auto-promotes to degenerate region
                                  (ℓ = min(0, x), u = max(0, x))

Ops.conjunction / Ops.disjunction  — kept as deprecated thin
                                     forwarders to lower(mode=AND) /
                                     lift(mode=OR) for one release
```

---

## Steps

Steps 1–3 are independent and can be ordered freely.  Step 4 depends
on Step 3.  Step 5 depends on 1, 2, 4.  Step 6 depends on 5.

### Step 1 — Rename `Ops.lift` / `Ops.lower` and extend to the
unified signature

**Files.** `bin/Layers.py` (`Ops` class, around lines 4308–4326),
`bin/Spaces.py` (`Basis` delegations, around lines 905–960), test
files exercising `Ops.lift` / `Ops.lower`.

**Change.**

1. Add the new unified signatures:

   ```python
   @staticmethod
   def lift(X1, X2=None, mode='OR', inverse=False):
       """Cantorian synthesis: many → one (∨).  Default is union/OR."""
       ...

   @staticmethod
   def lower(X1, X2=None, mode='AND', inverse=False):
       """Cantorian analysis: one → many (∧).  Default is intersection/AND."""
       ...
   ```

2. Mode dispatch table:

   | `mode` | `lower` (forward)            | `lift` (forward)             |
   |--------|-------------------------------|-------------------------------|
   | `AND`  | min (point) / `(max ℓ, min u)` (region) / elementwise product (smoothed) | n/a — `lift` defaults to OR |
   | `OR`   | n/a — `lower` defaults to AND | max (point) / `(min ℓ, max u)` (region) / arithmetic mean (smoothed) |
   | `NOT`  | bivector pole flip / sign flip | bivector pole flip / sign flip (self-inverse) |

3. Inverse semantics:
   - `lower(..., inverse=True)` → partial inverse via codebook-search
     witness recovery (current `conjunctionReverse` body when
     `mode='AND'`).
   - `lift(..., inverse=True)` → partial inverse via codebook-search
     (current `disjunctionReverse` body when `mode='OR'`).
   - `mode='NOT'` is self-inverse: `inverse=True` returns the same
     pole-flip / sign-flip operation.

4. Region coercion helper:

   ```python
   def _as_region(x):
       """Promote a point to a degenerate region containing the origin."""
       if isinstance(x, tuple) and len(x) == 2:
           return x  # already (ℓ, u)
       return torch.minimum(x, torch.zeros_like(x)), torch.maximum(x, torch.zeros_like(x))
   ```

   Used at the top of every dispatch when the body is the region
   form; bypassed when the body is the pointwise form.

5. **Deprecation alias.**  Keep the current `Ops.lift(left, right)`
   and `Ops.lower(left, right)` names as thin forwarders that emit a
   `DeprecationWarning` and dispatch to the inverted Cantorian
   call:

   ```python
   @staticmethod
   def lift(left, right):  # deprecated old signature
       warnings.warn("Ops.lift(left, right) is the *analysis* product; "
                     "use Ops.lower(x, y, mode='AND') for the Cantorian "
                     "polarity.  Will be removed in next release.",
                     DeprecationWarning, stacklevel=2)
       return Ops.lower(left, right, mode='AND')  # ← was lift's old body
   ```

   The two-arg legacy form must not collide with the new
   keyword-argument signature.  Easiest: dispatch on `isinstance(X2,
   (int, float, torch.Tensor))` and treat positional second-arg as
   the legacy call.

6. **`Basis` delegations** in `bin/Spaces.py`: add `Basis.lift` /
   `Basis.lower` thin wrappers paralleling the existing
   `Basis.conjunction` / `Basis.disjunction` ones.  Codebook-search
   inverse uses `self.getW()` as before.

**Tests.**

- New unit tests in `test/test_ops_lift_lower.py` (or extend an
  existing Ops test file) covering:
  - `lower(x, y, mode='AND')` returns elementwise product on points
    and `(max ℓ, min u)` on regions.
  - `lift(x, y, mode='OR')` returns mean on points and
    `(min ℓ, max u)` on regions.
  - `lift(x, mode='NOT')` is self-inverse.
  - Point-as-degenerate-region promotion.
  - Inverse via `inverse=True` matches current
    `conjunctionReverse` / `disjunctionReverse` outputs.
- Deprecation warning fires on the old positional form.

**Risk.** Any caller that uses `Ops.lift(x, y)` or `Ops.lower(x, y)`
with the old (Pi-named) semantics will silently get the new
(Cantorian) body if the legacy alias dispatch is wrong.  Mitigation:
the deprecation alias forwards to the body that *matches the old
behavior*, i.e. legacy `lift(x, y)` → new `lower(x, y, mode='AND')`
(both elementwise product); legacy `lower(x, y)` → new `lift(x, y,
mode='OR')` (both arithmetic mean).  The labels swap; the bodies do
not.  Test that the legacy-form output matches the pre-refactor
output bit-for-bit.

**Acceptance.** All existing tests pass.  New tests pass.  Grep
shows no remaining caller of `Ops.lift(...)` / `Ops.lower(...)` in
non-test code other than the deprecation aliases (after Step 5).

---

### Step 2 — Conjunction / disjunction become deprecated forwarders

**Files.** `bin/Layers.py` (`Ops`), `bin/Spaces.py` (`Basis`).

**Change.**

1. `Ops.conjunction` becomes a thin forwarder to `Ops.lower(x, y,
   mode='AND')` plus the `monotonic` parameter (which selects strict-
   lattice min vs RadMin same-sign-min-magnitude).
2. `Ops.disjunction` becomes a thin forwarder to `Ops.lift(x, y,
   mode='OR')` plus `monotonic`.
3. Add a `kind` parameter to lift / lower for selecting strict-
   lattice vs smoothed-pooling pointwise:
   - `kind='strict'` (default for non-monotonic): min for AND, max
     for OR.
   - `kind='smooth'`: elementwise product for AND, arithmetic mean
     for OR.
   - `kind='radial'`: RadMin for AND, RadMax for OR.
4. Existing callers of `Basis.conjunction` / `Basis.disjunction` are
   unaffected — they keep calling those names; bodies just route
   through the new dispatcher.

**Tests.** All `Basis.conjunction` / `Basis.disjunction` tests must
pass unchanged.  Bit-exact output match.

**Acceptance.** No caller-visible behavior change; only the routing
underneath shifts.

---

### Step 3 — Bivector activation in ConceptualSpace

**Files.** `bin/Spaces.py` (`ConceptualSpace.__init__`, around line
5730+), wherever `ConceptualSpace.subspace.activation` is read or
written, the `Pipeline` if it carries the activation through stages,
loss / regularizer code that consumes it.

**Change.**

1. `ConceptualSpace.subspace.nWhat = 2` (mirror SymbolicSpace at
   `bin/Spaces.py:5933`).  Total content width grows to `2 + nWhere
   + nWhen` per concept.
2. `ConceptualSpace.subspace.activation` shape becomes
   `[batch, N, 2]` (or `[batch, N, 2 + nWhere + nWhen]` when
   positional content is included).  The two `.what` dims are
   `[aP, aN]` per concept.
3. **Producers** (anything that *writes* into the conceptual
   activation):
   - PerceptualSpace → ConceptualSpace path: previously emitted a
     single signed scalar per concept.  Now emits a bivector.  For
     the initial landing, the simplest mapping is signed →
     `[max(0, x), max(0, -x)]` so positive evidence goes to `aP` and
     negative evidence goes to `aN`, with `[0, 0]` for ignorance and
     `[1, 1]` only emerging once contradictions accumulate.
   - SymbolicSpace → ConceptualSpace lower path (currently a
     symbol-only signed-sum projection): emits a bivector by
     preserving each pole pair through the lower's pooled product.
4. **Consumers** (anything that *reads* the conceptual activation):
   - Anywhere the activation is consumed as a scalar should derive
     `signed = aP − aN` explicitly.  Where contradiction-awareness
     is wanted, also derive `contradiction = aP · aN` as a sibling.
   - `ImpenetrableLayer` already operates on bivector codebook rows
     for SymbolicSpace; the same path applies to ConceptualSpace's
     codebook under bivector activation.
   - `tetralemma_balance_penalty` ([bin/Layers.py:2464](../../bin/Layers.py))
     extends from a symbol-only input to also accept conceptual
     bivectors.  No internal change — the function is shape-
     agnostic; it just now sees a different upstream caller.
5. Backward compatibility shim during transition: a
   `ConceptualSpace.activation_signed()` accessor that returns
   `aP − aN` for old callers that haven't migrated.  Remove after
   one release.

**Tests.**

- Add `test/test_conceptual_bivector.py` covering:
  - `[1, 0]` activations route correctly (positive evidence).
  - `[0, 1]` activations route correctly (negative evidence).
  - `[0, 0]` is distinguishable from `[1, 1]` downstream:
    `signed=0` for both, `contradiction=0` vs `contradiction=1`.
  - The signed-collapse accessor returns the same numbers as the
    pre-refactor single-scalar activation for `[max(0, x), max(0,
    -x)]` inputs.
- Existing tests that consume `ConceptualSpace.subspace.activation`
  must pass via the signed-collapse shim, then be migrated to
  bivector-aware reads in a follow-up.

**Risk.** Largest blast radius of the plan.  Many callers read
conceptual activations.  Mitigation: the signed-collapse shim is
the safety valve — every existing caller continues to see the same
scalar via `.activation_signed()`, only callers that opt into the
bivector see the new shape.  Land Step 3 with the shim in place;
migrate consumers in a follow-up (Step 6).

**Acceptance.** Existing test suite green via shim.  New bivector
tests pass.  No retraining required to validate (per memory: never
run `make train` locally — local runs are inference / unit tests).

---

### Step 4 — PiLayer / SigmaLayer ownership flip

**Files.** `bin/Spaces.py` (`SymbolicSpace.__init__`,
`ConceptualSpace.__init__`, `PerceptualSpace.__init__` if exists),
`bin/Pipeline.py` if it wires layers across spaces, `bin/Layers.py`
if `PiLayer` / `SigmaLayer` constructors need adjustment for the new
direction of use.

**Change.**

1. **Move the C → S `PiLayer` off SymbolicSpace.**  SymbolicSpace
   instead constructs a `SigmaLayer` (synthesis from C up).  The
   layer's forward pass does the C → S projection; its reverse pass
   does the S → C lower for free if the layer is invertible.
2. **Add a `PiLayer` to ConceptualSpace** for the C → S analysis
   direction.  This is the lower primitive for going from a
   conceptual region back down to its concept-space components.  Its
   forward direction is the S → C decomposition.
3. **Add a `SigmaLayer` to ConceptualSpace** for the P → C synthesis
   direction (gathering percepts into concepts).  PerceptualSpace
   gets a `PiLayer` for its analysis direction.
4. The `invertible=True` flag stays where it is on the layer that
   owns the relation.  Each layer has both forward (synthesis or
   analysis) and reverse (the partial inverse) directions.
5. Update Pipeline wiring: at C → S, the data flows through
   SymbolicSpace's SigmaLayer (forward = synthesis).  At S → C, the
   data flows through ConceptualSpace's PiLayer (forward = analysis).

**Tests.**

- Existing C → S forward tests must still pass — same numbers, new
  ownership.  Easiest: assert that
  `SymbolicSpace.layer.forward(c) == ConceptualSpace.pi_layer.reverse(c)`
  to within numeric tolerance, where `pi_layer.reverse` is the
  inverse used to be on the symbolic side.
- Add ownership assertions in `__init__` checks: each space exposes
  its owned layers via attributes that other spaces import as a
  formal contract, not by reaching into another space's internals.

**Risk.** Bookkeeping-only refactor, but cross-cuts.  Mitigation:
land Step 4 *after* Step 1 so the renamed `Ops.lift` / `Ops.lower`
are available as the body of the layers' forward/reverse.

**Acceptance.** All existing layer-level tests pass.  Spaces own the
layers documented in Logic.md §8 *Ownership*.

---

### Step 5 — Wire `Ops.lift` / `Ops.lower` into the layer
forward / reverse paths

**Files.** `bin/Layers.py` (`PiLayer.forward`, `SigmaLayer.forward`,
their reverse paths).

**Change.**

1. `PiLayer.forward` calls `Ops.lower(input, self.weight, mode='AND',
   kind='smooth')` — the smoothed (product / mean) form is the
   natural fit for differentiable pooling, not the strict
   min/max.
2. `SigmaLayer.forward` calls `Ops.lift(input, self.weight, mode='OR',
   kind='smooth')`.
3. The reverse paths use `inverse=True`.
4. The hard-lattice variant (`kind='strict'`) is available for
   inference-time / extraction-time use but not used in training
   forward passes (sparse gradients).

**Tests.** Numerical equivalence to pre-Step-5 layer behavior under
matched seeds and inputs.  Layer training tests pass.

**Acceptance.** Layers and `Ops` share the same primitive names and
bodies; no duplicated formulas.

---

### Step 6 — Grammar-driven dispatch via `data/grammar.cfg`

This step can be developed in parallel with Steps 3–5 / Step 7.  It
depends only on Step 1 (the unified `Ops.lift` / `Ops.lower`
signature with mode dispatch); it does not require the bivector
ConceptualSpace or the ownership flip.

**Files.** `data/grammar.cfg` (annotation), the grammar loader
(parses `grammar.cfg` into a rule table), the parser's
apply-rule step, the codebook initialization in
`bin/Spaces.py` (reserve category-vector slots).

**Change.**

1. **Adopt the explicit-op grammar form** already partially in
   `todo.md`.  Each production's RHS *is* a function call naming
   the operation directly; no separate `(mode, dir)` annotation
   needed.  Form:

   ```
   LHS = op(arg1, arg2[, ...])     # binary or unary
   LHS = arg                       # PROJECT (terminal projection)
   ```

   The unified table below is the complete commit, consolidating
   the in-progress `todo.md` rules, translating remaining `->`
   rules into the new form, and resolving the `#`-marked
   uncertain rules with proposed annotations.

   #### Layer 1 — Syntactic productions

   These build a higher-level state from constituents.  Each row's
   RHS is the dispatched call; the LHS is the produced state and
   gets stamped with its category vector.

   | Production | Resolves to (`Ops`) | Notes |
   |---|---|---|
   | `S = NP` | `project(NP)` | terminal projection — pass NP up, type-stamp as S |
   | `S = lift(NP, VP)` | `Ops.lift(NP, VP, mode='OR')` | subject + predicate; Cantorian synthesis |
   | `S = lift(NP, VO)` | `Ops.lift(NP, VO, mode='OR')` | subject + verb-object; uses `VO` state below |
   | `S = equals(NP, NP)` | `Ops.equal(NP1, NP2)` | copula identification *X is the Y* |
   | `S = equals(NP, AP)` | `Ops.equal(NP, AP)` | predicative attribution *X is red* |
   | `S = part(NP, NP)` | `Ops.part(NP1, NP2)` | mereological *X is part of Y* |
   | `S = not(S)` | `Ops.lift(S, mode='NOT')` | propositional negation |
   | `S = conjunction(S, S)` | `Ops.lower(S1, S2, mode='AND')` | propositional ∧ |
   | `S = disjunction(S, S)` | `Ops.lift(S1, S2, mode='OR')` | propositional ∨ |
   | `S = query(NP, AP)` | `Ops.query(NP, AP)` | **new op** — interrogative *is X red?* |
   | `S = query(NP, NP)` | `Ops.query(NP1, NP2)` | **new op** — interrogative *is X a Y?* |
   | `S = intersection(MP, S)` ⚠ | `Ops.lower(MP, S, mode='AND')` | resolves `#S = MP S` (modal modifies S) |
   | `S = intersection(PP, S)` ⚠ | `Ops.lower(PP, S, mode='AND')` | resolves `#S = PP S` (fronted PP modifies S) |
   | `VO = intersection(VP, NP)` | `Ops.lower(VP, NP, mode='AND')` | introduces `VO` (verb-object) state |
   | `NP = N` | `project(N)` | |
   | `NP = intersection(AP, NP)` | `Ops.lower(AP, NP, mode='AND')` | canonical ADJ ∩ N |
   | `NP = intersection(NP, PP)` | `Ops.lower(NP, PP, mode='AND')` | PP modifies NP |
   | `NP = conjunction(NP, NP)` | `Ops.lower(NP1, NP2, mode='AND')` | ⚠ commits to AND-meet semantics — see open question O8 |
   | `NP = disjunction(NP, NP)` | `Ops.lift(NP1, NP2, mode='OR')` | OR-coordination, entity-set union |
   | `VP = V` | `project(V)` | |
   | `VP = intersection(ADV, VP)` | `Ops.lower(ADV, VP, mode='AND')` | adverbial modification |
   | `VP = intersection(V, PP)` | `Ops.lower(V, PP, mode='AND')` | V + PP |
   | `VP = not(VP)` | `Ops.lift(VP, mode='NOT')` | predicate negation |
   | `VP = intersection(MP, VP)` ⚠ | `Ops.lower(MP, VP, mode='AND')` | resolves `#VP = MP VP` |
   | `VP = intersection(ADJ, VP)` ⚠ | `Ops.lower(ADJ, VP, mode='AND')` | resolves `#VP = ADJ VP` (rare; mark unusual) |
   | `VP = intersection(V, NP)` ⚠ | `Ops.lower(V, NP, mode='AND')` | resolves `#VP = V NP`; predicate-arg meet (or use `bind`) |
   | `VP = intersection(V, S)` ⚠ | `Ops.lower(V, S, mode='AND')` | resolves `#VP = V S`; sentential complement |
   | `VP = intersection(V, MP)` ⚠ | `Ops.lower(V, MP, mode='AND')` | resolves `#VP = V MP` |
   | `VP = intersection(VP, PP)` ⚠ | `Ops.lower(VP, PP, mode='AND')` | resolves `#VP = VP PP` |
   | `VP = intersection(DEF, VP)` ⚠ | `Ops.lower(DEF, VP, mode='AND')` | resolves `#VP = DEF VP`; copula + VP (passive aux) — flag for review |
   | `AP = ADJ` | `project(ADJ)` | |
   | `AP = DET` | `project(DET)` | |
   | `AP = intersection(ADJ, AP)` | `Ops.lower(ADJ, AP, mode='AND')` | adjective stacking |
   | `AP = scale(DEG, AP)` ⚠ | **new op** `Ops.scale` (or `Ops.lower(DEG, AP, mode='AND')` as fallback) | degree intensification — see open question O9 |
   | `MP = ADV` | `project(ADV)` | |
   | `MP = intersection(ADV, MP)` | `Ops.lower(ADV, MP, mode='AND')` | adverb stacking (translated from `MP -> ADV MP`) |
   | `PP = bind(P, NP)` | `Ops.bind(P, NP)` (or `Ops.lower(P, NP, mode='AND')` as fallback) | preposition + complement; consider new `bind` mode for asymmetric head-complement |
   | `DEF = IS` | `project(IS)` | |
   | `DEF = not(IS)` | `Ops.lift(IS, mode='NOT')` | translated from `DEF -> IS NOT` |
   | `HAS = POSSESS` | `project(POSSESS)` | |
   | `HAS = not(POSSESS)` | `Ops.lift(POSSESS, mode='NOT')` | translated from `HAS -> POSSESS NOT` |

   **Legend.**  ⚠ = resolution-needed during implementation
   (proposed annotation given; confirm or revise per measured
   parsing behavior).  **bold** = new `Ops` method to add.

   #### Layer 2 — Symbol-level ops on existing states

   The `<upward>` section in `todo.md` lists operations applied to
   *already-formed* S states (or other states), not productions.
   These belong in the `Ops` dispatch table, not in the syntactic
   rule loader — they are post-hoc transformations the parser /
   grammar / runtime can call on a parsed S, much as any of the
   pointwise primitives can be called on any activation.

   | Op (XML form) | Arity | `Ops` method | Status |
   |---|---|---|---|
   | `true(S)` | unary | constant `+1` assertion (or call site for `Ops.true()`) | EXISTS (returns 1) — extend to apply across an S |
   | `false(S)` | unary | constant `-1` assertion | EXISTS (returns -1) — extend to apply |
   | `non(S)` | unary | `Ops.non(S)` | EXISTS — triangular residual / Kleene |
   | `not(S)` | unary | `Ops.lift(S, mode='NOT')` | EXISTS via negation |
   | `what(S)` | unary | `Ops.what(S, nWhat, nWhere, nWhen)` | EXISTS — axis selector |
   | `where(S)` | unary | `Ops.where(...)` | EXISTS |
   | `when(S)` | unary | `Ops.when(...)` | EXISTS |
   | `conjunction(S, S)` | binary | `Ops.lower(S1, S2, mode='AND')` | EXISTS as `Ops.conjunction` |
   | `disjunction(S, S)` | binary | `Ops.lift(S1, S2, mode='OR')` | EXISTS as `Ops.disjunction` |
   | `intersection(S, S)` | binary | `Ops.lower(S1, S2, mode='AND')` (region) | EXISTS for points; region-pair extension lands in Step 1 |
   | `union(S, S)` | binary | `Ops.lift(S1, S2, mode='OR')` (region) | same |
   | `equals(S, S)` | binary | `Ops.equal(S1, S2)` | EXISTS |
   | `part(S, S)` | binary | `Ops.part(S1, S2)` | EXISTS |
   | `lower(S, S)` | binary | `Ops.lower(S1, S2)` (default mode='AND') | EXISTS after Step 1 |
   | `lift(S, S)` | binary | `Ops.lift(S1, S2)` (default mode='OR') | EXISTS after Step 1 |
   | `query(S, S)` | binary | `Ops.query(S1, S2)` | **NEW** — interrogative speech act; produces a *question* state, not a truth assertion |
   | `swap(S, S)` | binary | `Ops.swap(S1, S2)` | **NEW** — argument-position swap (re-ordering) |

   #### Layer 2.5 — Reverse productions (functionReverse form)

   The downward (reverse) direction of every Layer-1 production has
   the same explicit-op grammar shape, with the LHS becoming a
   tuple of recovered constituents and the op carrying a `Reverse`
   suffix.  General form:

   ```
   arg1, arg2, ... = opReverse(LHS)
   ```

   Each Layer-1 production therefore comes with a paired Layer-2.5
   reverse production, derived mechanically by swapping the LHS
   and RHS and substituting `op` → `opReverse`.  For the productions
   from the `<upward>` section in `todo.md`:

   | Forward (Layer 1) | Reverse (Layer 2.5) | Inverse exactness |
   |---|---|---|
   | `S = lift(NP, VP)` | `NP, VP = liftReverse(S)` | lossy (codebook-search witness recovery) |
   | `S = lift(NP, VO)` | `NP, VO = liftReverse(S)` | lossy |
   | `VO = intersection(VP, NP)` | `VP, NP = intersectionReverse(VO)` | lossy (pseudo-inverse) |
   | `S = conjunction(S, S)` | `S, S = conjunctionReverse(S)` | lossy (`Ops.conjunctionReverse` already exists, codebook search) |
   | `S = disjunction(S, S)` | `S, S = disjunctionReverse(S)` | lossy (`Ops.disjunctionReverse` already exists) |
   | `S = not(S)` | `S = notReverse(S)` | self-inverse (exact) |
   | `S = equals(NP, NP)` | `NP, NP = equalsReverse(S)` | symmetric — order-ambiguous |
   | `S = equals(NP, AP)` | `NP, AP = equalsReverse(S)` | typed by LHS / RHS category vectors |
   | `S = part(NP, NP)` | `NP, NP = partReverse(S)` | converse — exact in form, lossy in functional inverse (residual `P_⊥`) |
   | `S = query(NP, AP)` | `NP, AP = queryReverse(S)` | NEW; depends on query-op design |
   | `S = query(NP, NP)` | `NP, NP = queryReverse(S)` | NEW |
   | `NP = N` | `N = projectReverse(NP)` | exact (identity-with-typing-strip) |
   | `NP = intersection(AP, NP)` | `AP, NP = intersectionReverse(NP)` | lossy |
   | `NP = intersection(NP, PP)` | `NP, PP = intersectionReverse(NP)` | lossy |
   | `NP = conjunction(NP, NP)` | `NP, NP = conjunctionReverse(NP)` | lossy |
   | `NP = disjunction(NP, NP)` | `NP, NP = disjunctionReverse(NP)` | lossy |
   | `VP = V` | `V = projectReverse(VP)` | exact |
   | `VP = intersection(ADV, VP)` | `ADV, VP = intersectionReverse(VP)` | lossy |
   | `VP = intersection(V, PP)` | `V, PP = intersectionReverse(VP)` | lossy |
   | `VP = not(VP)` | `VP = notReverse(VP)` | self-inverse (exact) |
   | `AP = scale(DEG, AP)` | `DEG, AP = scaleReverse(AP)` | NEW; depends on scale-op design |
   | `PP = bind(P, NP)` | `P, NP = bindReverse(PP)` | lossy or exact depending on bind design |
   | `DEF = not(IS)` | `IS = notReverse(DEF)` | self-inverse |
   | `HAS = not(POSSESS)` | `POSSESS = notReverse(HAS)` | self-inverse |

   Reverse productions are not a separate file — they are derived
   from the forward productions at grammar-load time.  Implementation:

   ```python
   for rule in forward_rules:
       lhs, op_name, args = rule
       reverse_rule = (args, op_name + 'Reverse', [lhs])
       reverse_rules.append(reverse_rule)
   ```

   The dispatcher then has both directions available as a property
   of each forward production: `production.forward(...)` runs the
   Layer-1 op; `production.reverse(...)` runs the Layer-2.5 op
   with multi-return arity matching the forward arity (unary
   forward → unary reverse; binary forward → binary reverse with
   tuple return; etc.).

   **Multi-return signature on `Ops`.**  The unified op signature
   `Y = f(X1, X2=None, mode=..., inverse=False)` already carries
   `inverse=True` to mean "run the reverse direction."  For
   compatibility, `opReverse(Y)` is a convenience alias for `op(Y,
   None, mode=..., inverse=True)`, with a calling convention that
   the result is a tuple sized to match the forward op's arity.
   Concretely:

   ```python
   @staticmethod
   def liftReverse(Y):
       """Convenience for lift(Y, inverse=True) returning a tuple."""
       return Ops.lift(Y, inverse=True)   # returns (X1, X2)

   @staticmethod
   def conjunctionReverse(Y):
       """Already exists at Layers.py:4252 with (result, y, W) signature.
       Multi-return form wraps to return (X1, X2)."""
       ...
   ```

   The existing `Ops.conjunctionReverse(result, y, W, monotonic)`
   and `Ops.disjunctionReverse(result, y, W, monotonic)` (Layers.py:4252,
   4262) take a `y` operand and a `W` codebook for the search and
   return a single recovered left operand.  Under the multi-return
   convention they need a thin wrapper that supplies `W` from the
   layer context (already done in `Basis.conjunctionReverse` at
   Spaces.py:936) and returns both operands as a tuple — the
   recovered left operand from the search, and the right operand
   that the search was conditioned on.

   **Exactness annotation on each reverse rule.**  The third
   column of the table above (lossy / exact / self-inverse) tells
   the runtime which inverses are functional (always recover the
   original) vs pseudo (recover *some* operand pair that re-lifts
   to the LHS).  Encoded as a flag on each forward production:

   ```python
   class Production:
       lhs: str
       op: str
       args: list[str]
       reverse_exact: bool   # True if functional inverse; False if pseudo
   ```

   Production rules whose forward op is `not`, `project`, or
   `negation` get `reverse_exact=True`; everything else defaults to
   `False`.

2. **Extend the grammar loader** to parse the explicit-op form:

   ```
   LHS '=' op_name '(' arg_name (',' arg_name)* ')'   # n-ary
   LHS '=' arg_name                                    # PROJECT
   ```

   Each parsed production stores `(lhs, op_name, [arg_names])`.
   The dispatcher resolves `op_name` against an `Ops` dispatch
   table at runtime, with default `bind`/`intersection` chosen for
   unknown ops (with a warning).

3. **Reserve category-vector slots in the conceptual codebook.**
   For each grammatical category in `grammar.cfg` (S, NP, VP, AP,
   MP, PP, DEF, HAS, N, V, ADJ, ADV, plus closed-class IS, POSSESS,
   NOT, AND, OR, P, DET, DEG), reserve a row in the conceptual
   codebook to hold the category vector.  Two options:
   - **Option A (shared codebook).**  Reserve K_cat rows at the
     start of the conceptual codebook for category vectors; content
     concepts populate the rest.  One regularizer
     (`ImpenetrableLayer`) covers both.
   - **Option B (separate codebook).**  Add a second codebook
     `Spaces.ConceptualSpace.category_codebook` of size K_cat.
     Cleaner separation, more parameters, two regularizers.
   Recommend **A** (Cantorian unity: ADJ-ness *is* a concept), but
   flag for review during implementation.

4. **Each grammatical state carries `category_vector` and
   `activation`.**  At parse time, when a production fires:

   ```python
   def apply_rule(rule, rhs_states):
       mode, direction = rule.annotation
       if direction == 'lift':
           op = Ops.lift
       else:  # 'lower' or 'relational'
           op = Ops.lower

       # Unary case (NOT, PROJECT)
       if len(rhs_states) == 1:
           result_act = op(rhs_states[0].activation, mode=mode)
       # Binary case (AND, OR, BIND, PART, EQUAL, HAS)
       elif len(rhs_states) == 2:
           result_act = op(rhs_states[0].activation,
                           rhs_states[1].activation,
                           mode=mode)
       # Ternary (rare — three-arg S productions)
       else:
           result_act = _fold_n_ary(op, rhs_states, mode)

       lhs_state = State(category=rule.lhs)
       lhs_state.category_vector = codebook.row(rule.lhs)
       lhs_state.activation = result_act
       # Optionally stamp the LHS category onto the activation:
       lhs_state.activation = _type_stamp(result_act,
                                          lhs_state.category_vector)
       return lhs_state
   ```

   `_type_stamp` is a small operation that biases the activation
   toward the LHS category region — e.g. add a small fraction of
   the category vector, or run a single Sigma step with the
   category as the synthesis target.  Implementation detail; the
   functional behavior is "the result has its LHS category as a
   tag."

5. **PROJECT mode** is unary identity-with-typing: pass the RHS
   activation through unchanged but stamp the LHS category onto
   it.  This is the trivial 1-of-1 pooling that types a state.

6. **BIND mode** is the generic compositional bind.  Routes to
   `lower(..., mode='AND')` by default — the bound state is the
   meet of its operands restricted to their category dimensions.
   This is a pragmatic catch-all; specific BIND patterns
   (predicate-argument, head-complement) may want their own modes
   later, but the default is sound.

7. **PART, EQUAL, HAS** route to the existing mereology suite
   (`Basis.part`, `Basis.equal`) rather than to a new mode body.
   The dispatcher recognizes these annotations and calls the
   appropriate `Basis` method, lifting the result to the LHS state.

**Tests.**

- Unit tests in `test/test_grammar_dispatch.py` covering:
  - Each annotation parses correctly from `grammar.cfg`.
  - `apply_rule` for `NP -> AP NP` produces `lower(AP_act, NP_act,
    mode='AND')` and stamps NP category.
  - `apply_rule` for `S -> S AND S` produces `lower(S1, S2,
    mode='AND')`.
  - `apply_rule` for `NP -> NP AND NP` produces `lift(NP1, NP2,
    mode='OR')`.
  - `apply_rule` for `S -> NOT S` produces `lift(S, mode='NOT')`.
  - PROJECT preserves the activation while stamping the LHS
    category.
- Codebook initialization test: the K_cat reserved rows exist and
  are accessible by category name.

**Risk.** Two areas:

- **Codebook collision.**  If category vectors share rows with
  content concepts (Option A), they need to be initialized in a
  protected region and possibly frozen during early training, or
  `ImpenetrableLayer` may push them around.  Mitigation: tag
  category rows in the codebook with a flag that
  `ImpenetrableLayer` honors — exempt from inter-pair penalties
  with content rows, but apply within-category-class pressure (ADJ
  vs N should still be `disjoint`).
- **Annotation ambiguity.**  Some rules may be defensible under
  multiple annotations (e.g. `S -> NP VP` could be `BIND` or `PART`
  with NP as the subject region containing the VP-described
  property).  The annotation table is a starting commitment; revise
  per measured parsing behavior.

**Acceptance.**  Each rule in `grammar.cfg` carries an annotation;
the loader parses them; `apply_rule` dispatches `Ops.lift` /
`Ops.lower` with the correct mode and operands; category vectors
are reserved and tagged in the codebook; existing parser tests pass
with the dispatch wiring in place; new tests exercise each
annotation path.

---

### Step 7 — Migrate conceptual-activation consumers off the
signed-collapse shim

**Files.** Wherever `ConceptualSpace.subspace.activation` is read
*after* Step 3 lands.  Probably `bin/Pipeline.py`, several places in
`bin/Layers.py` (TruthLayer, ImpenetrableLayer, loss classes), and
some `bin/Spaces.py` paths.

**Change.**

1. Replace each `activation_signed()` call with explicit bivector
   handling, choosing one of:
   - `aP − aN` (signed scalar — same number as before),
   - `(aP − aN, aP · aN)` (signed + contradiction mask),
   - direct bivector use (if the consumer can take both poles).
2. Once no caller of `activation_signed()` remains, remove the
   shim.

**Tests.** A grep gate that fails CI if `activation_signed` is
referenced outside the `ConceptualSpace` definition itself, applied
once Step 6 is complete.

**Acceptance.** Shim removed; all consumers operate on bivector
activations or explicit derivations from them.

---

## Open questions to resolve during implementation

1. **Hard-pair extraction timing.**  Sparsity + cardinality + anneal
   on `SigmaLayer.weight` and `PiLayer.weight` to produce
   `(op, c_a, c_b)` triples per symbol — is this part of training
   or a post-training extraction pass?  The conversation leans
   toward "annealed during training so the codebook is interpretable
   at any checkpoint," but the cost on training stability is
   unknown.  Run a small ablation before committing.
2. **`SigmaLayer` max readout vs averaging readout.**  Logic.md §8
   describes Sigma as "weighted sum + max readout" for the
   synthesis-as-union semantics; `Ops.lift(mode='OR', kind='smooth')`
   currently routes to arithmetic mean.  Decide whether the smooth
   form should be mean (current) or softmax-weighted average (closer
   to max-readout) — affects how bridge points show up.  Likely
   needs a `softmax_temperature` parameter or two distinct kinds.
3. **PerceptualSpace owning a SigmaLayer?**  At the bottom of the
   stack PerceptualSpace produces the P-layer many; there is no
   layer below to synthesize from.  But if percepts themselves are
   composed (BPE chunks of subword units), there may be a sub-
   PerceptualSpace layer that wants synthesis.  Defer to a separate
   plan if that subdivision arrives.
4. **`tetralemma_balance_penalty` over conceptual bivectors.**
   Allowed-corner config is currently set per-SymbolicSpace; do we
   want independent config for conceptual bivectors, or shared?
   Probably shared by default; revisit if the two layers diverge in
   contradiction-tolerance needs.

5. **Are query and swap really new ops, or compositions?**
   `query(NP, AP)` could be implemented as a flag-bearing variant of
   `equals(NP, AP)` that does not commit to a truth value, just
   produces an interrogative S that can later be answered.  `swap(S,
   S)` could be a permutation primitive on the symbol activation
   indices.  Both deserve their own decision in implementation —
   tracked here so they don't silently materialize as untyped
   methods.

6. **`bind` vs `intersection` for asymmetric head-complement.**
   The plan currently routes `PP = bind(P, NP)` through a proposed
   `Ops.bind` (or falls back to `intersection`).  The asymmetry
   (preposition is the head, NP is the complement) might warrant a
   distinct mode that preserves head-marking, especially if the
   parser later needs to recover head-of-phrase.  Decide during
   parser wiring.

7. **`scale` for degree modification.**  `AP = scale(DEG, AP)` is
   *intensification*, not strict meet.  "Very red" is more-red, not
   the meet of very-ness with red.  Strict meet would predict
   "extremely red" ⊆ "red" ⊆ "DEG" — fine on the red axis, weird on
   the DEG axis.  Options: (a) treat as `intersection(DEG, AP)` and
   accept the imprecision; (b) add an `Ops.scale` that
   multiplicatively biases `AP`'s envelope toward saturation
   weighted by `DEG`'s magnitude.  (b) is more faithful linguistically;
   (a) is simpler and works in monotonic mode where DEG-magnitude
   already saturates the AP region.  Decide during AP wiring.

8. **`NP = conjunction(NP, NP)` semantics.**  The grammar in
   `todo.md` rewrites *NP and NP* to `conjunction(NP, NP)`, which is
   AND/lower (intersection of regions).  But "apples and oranges"
   refers to a set containing both — the entity-set union.  Two
   readings:
   - **Propositional reading.**  The resulting NP carries a meet of
     the two conceptual commitments ("both NP-categories are in
     scope").  Predicates over the NP later distribute.  Matches the
     grammar.
   - **Lattice reading.**  Region union (lift OR), as in
     `disjunction(NP, NP)`.  Linguistically natural for entity
     sets, but then `conjunction(NP, NP)` and `disjunction(NP, NP)`
     produce identical regions and the surface AND/OR distinction
     is lost.

   Recommend committing to the **propositional reading** (per the
   grammar) and documenting that NP-coordination's lattice effect
   is the meet, with the surface AND/OR distinction reasserting at
   predicate consumption.  Test on real sentences during
   implementation.

9. **DEG modification, restated.**  See O7.  Same question.

10. **Layer 2.5 reverse productions are derived, not authored.**
    Each forward production `LHS = op(arg1, arg2, ...)` mechanically
    yields a reverse production
    `arg1, arg2, ... = opReverse(LHS)` at grammar-load time — no
    separate file, no separate authoring.  The exactness flag
    (`reverse_exact: bool`) is the only per-production annotation
    needed beyond the forward rule itself.  The `<upward>` XML in
    `todo.md` is therefore better read as a *checklist* of which
    forward ops have working reverses, not as a separate grammar
    section.

---

## Test plan

- **Unit-level.**  Each step has a focused test file (see per-step
  *Tests* sections).
- **Integration.**  After Step 5, run a full inference epoch
  (`runEpoch(split="runtime")` on a small fixture) and compare
  pre/post outputs.  Numerical drift should be limited to the
  bivector-vs-scalar accessor difference (Step 3) and the legacy-
  alias path (Step 1).
- **Training (remote only — never local per memory).**  After Step 6
  ships, queue a training job on ArborMini.local via
  `make sync_local` to verify no regression in convergence behavior
  on a held-out task.  Compare loss curves to a pre-refactor run.

---

## Rollback plan

Each step lands in its own commit and is independently revertible:

- Steps 1–2 are renames + forwarders; revert restores
  `Ops.conjunction` / `Ops.disjunction` paths.
- Step 3 is a shape change with an explicit shim; revert removes the
  shim and restores the single-scalar activation field.
- Step 4 is bookkeeping; revert restores ownership on
  SymbolicSpace.
- Step 5 is wiring; revert restores the previous layer forward
  bodies.
- Step 6 is consumer migration; revert restores the shim and the
  callers' use of it.

The bivector activation (Step 3) is the largest blast radius and
the only one that materially changes a tensor shape exposed across
spaces.  It is the rollback point most likely to be exercised; the
shim exists specifically to make rollback cheap.

---

## Doc cross-references

- [Logic.md](../Logic.md) §8 *Regions, Witness Sets, and the Slab
  Lattice* — conceptual ground truth for the framing.
- [OpsComparison.md](../OpsComparison.md) §7 *Recommendations* —
  links here for the implementation plan; this plan supersedes
  Recommendations 4–7 of that document.
- [Mereology.md](../Mereology.md) — to receive a one-line
  cross-reference back to Logic.md §8 once landed (independent
  doc-only follow-up).
- [BuddhistParallels.md](../BuddhistParallels.md) — tetralemma
  semantics that the bivector activation realizes; no change needed
  but worth re-reading before implementing Step 3.
