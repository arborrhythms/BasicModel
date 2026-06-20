# Handoff: Step 6 — Grammar-driven dispatch via `data/grammar.cfg`

**Date.** 2026-04-25
**Owner.** Alec
**Status.** Not started.  Step 5 (binary forward mode + STE selector)
landed; the N-ary `Ops` rewiring sub-steps were deferred (see
*Pre-state* below).

---

## What this hands off

Step 6 of the lift / lower / bivector refactor:
- Plan: [2026-04-24-lift-lower-bivector-refactor.md §Step 6](2026-04-24-lift-lower-bivector-refactor.md)
  (lines 405–610).
- Step 5 handoff: [2026-04-25-step5-ops-wiring-handoff.md](2026-04-25-step5-ops-wiring-handoff.md)
  (the immediate predecessor; defines what landed and what was
  deferred).

---

## Pre-state (Step 5 outcome)

### What landed

- **`Ops.top2_select_ste(x, dim=-1)`** ([bin/Layers.py](../../bin/Layers.py)
  in the `Ops` class).  Hard top-2-by-`|x|` selection along an axis,
  with straight-through gradient (forward = hard mask, backward =
  identity).  No-op when the axis has ≤ 2 entries.  Zero is the
  neutral element for both Pi (mult-identity 1 after `_to_mult`) and
  Sigma (additive zero after `atanh`), so unselected operands drop
  cleanly out of either fold.
- **`PiLayer.forward(x, binary=False)`** ([bin/Layers.py](../../bin/Layers.py)).
  When `binary=True`, applies `Ops.top2_select_ste(x)` before the
  log-domain entry transform.  `binary=False` (default) is
  bit-equivalent to the pre-Step-5 body.
- **`SigmaLayer.forward(x, binary=False)`** ([bin/Layers.py](../../bin/Layers.py)).
  Same pattern: `binary=True` selects top-2 input operands before the
  `atanh` entry transform; `binary=False` default unchanged.
- **`test/test_ops_layer_wiring.py`** — covers the helper, both
  layers' `binary=True` selection, the long-tail-→-identity property,
  STE gradient flow to all input dims, and round-trip recovery on the
  `binary=False` default.
- **Doc updates.**  [doc/Logic.md §8](../Logic.md) and
  [doc/Spaces.md *ConceptualSpace* / *SymbolicSpace*](../Spaces.md)
  describe the `binary=True` mode, the STE backward, and the
  zero-neutral-element justification.

### What was deferred from Step 5 (and is now Step 6's problem)

The original Step 5 plan called for three additional rewirings that
**did not land** and now belong to Step 6 (or a follow-on micro-step).
Rationale and pointers below so Step 6 can pick them up cleanly:

1. **Rewriting `PiLayer.forward` / `SigmaLayer.forward` N-ary bodies
   through `Ops.lower` / `Ops.lift`** (parent plan §Step 5 items 1–2,
   handoff sequencing items 2–3).  The literal call
   `Ops.lower(input, self.weight, mode='AND', kind='smooth')` returns
   `input * self.weight` (elementwise product, see
   [bin/Layers.py:4495-4496](../../bin/Layers.py)), which is **not**
   the log-domain matrix multiplication PiLayer's existing forward
   computes via `compute_W_current()`.  Forcing the literal Ops call
   would break numerical equivalence (Step 5 acceptance: `atol=1e-6`).
   Resolution paths for Step 6 (or later):
   - (a) Extend `Ops` with a layer-projection mode that takes a
     weight matrix and does the log-domain matmul (new public API).
   - (b) Refactor the layers to delegate to a private helper that
     captures "_to_mult → log → matmul → tanh(/2)" (or the additive
     analogue), keeping the body inline-equivalent but de-duplicated
     across Pi and Sigma — this is structural cleanup, not an Ops
     wiring.
   - (c) Accept the inline body as the canonical formulation; the
     "lives in one place" goal is satisfied by `binary=True` going
     through `Ops.top2_select_ste`, not by the N-ary body.
   The Step 5 handoff favored (c) implicitly by deferring; Step 6
   should pick one explicitly.

2. **Wiring `PiLayer.reverse` / `SigmaLayer.reverse` through `Ops`**
   (parent plan §Step 5 item 3, handoff sequencing item 4).  Same
   tension — the layer reverse uses `self.layer.reverse(y)` for the
   linear-algebraic LDU inverse, which is a different operation from
   `Ops.lower(..., inverse=True)` (the codebook-search inverse,
   raises `NotImplementedError` without a `W` per
   [bin/Layers.py:4484-4487](../../bin/Layers.py)).  Resolution
   parallel to (1).

3. **Migrating `Language.py` grammar method bodies** to the
   binary form (handoff sequencing item 5).  The original plan said
   `liftForward(left, right) → pi.binary(left, right)`, with a
   `binary` method matching the (left, right) two-operand shape.
   The Step 5 redesign moved the binary selector onto
   `forward(x, binary=True)` as auto top-2 selection over a single
   N-element input — which has **different semantics** from the
   grammar's explicit (left, right) pairing.  The four bodies at
   [Language.py:920–942](../../bin/Language.py) (`liftForward`,
   `liftReverse`, `lowerForward`, `lowerReverse`) keep their
   explicit `Ops.lower` / `Ops.lift` / `Ops.liftReverse` /
   `Ops.lowerReverse` calls — no migration occurred.

   Step 6 should decide whether the grammar's per-rule application
   wants:
   - (a) the explicit two-operand elementwise form (current — keep
     `Ops.lower` / `Ops.lift` directly in Language.py);
   - (b) auto top-2 over a packed N-element state (call
     `forward(x, binary=True)` on the appropriate space's layer);
   - (c) both, dispatched on rule type.
   The grammar-cfg-driven dispatch this Step 6 introduces is the
   natural place to make this call per-rule.

### What is unchanged

- Ownership of Pi / Sigma per Step 4 still holds:
  `ConceptualSpace.sigma`, `ConceptualSpace.pi`,
  `SymbolicSpace.sigma`, `PerceptualSpace.pi` (dormant).
- The `SymbolicSpace.layer` deprecated alias is still in place.
- Acceptance sweep numerics from Step 4 are preserved (the only
  Step 5 surface change is the new `binary=True` opt-in flag,
  unused by default).

---

## What Step 6 changes

The core deliverable is **explicit-op grammar dispatch driven by
`data/grammar.cfg`**, replacing the current implicit
rule-name-to-method dispatch ([Language.py:465-489 `_RULE_METHODS`](../../bin/Language.py))
with a config-loaded rule table where each production's RHS is
literally a function call naming the `Ops` method to invoke.

### Rule form

```
LHS = op(arg1, arg2[, ...])     # binary or unary
LHS = arg                       # PROJECT (terminal projection)
```

The full Layer-1 (syntactic productions) and Layer-2 (post-hoc S-ops)
rule tables are spelled out in parent plan §Step 6 lines 440–516.
Notable points consolidated from those tables:

- **Layer 1** consolidates `todo.md`'s in-progress rules and the
  `#`-marked uncertain ones, with proposed annotations.  31 rows;
  five new ops or fallbacks (`query`, `equal`, `part`, `bind`,
  `scale`).
- **Layer 2** lists post-hoc symbol-level ops applied to already-
  formed S states (or other states).  These are not productions —
  they belong in the `Ops` dispatch table, callable on any
  activation regardless of which production produced it.  16 rows.
- **Layer 2.5** is the mechanical reverse of Layer 1: derive
  `arg1, arg2 = opReverse(LHS)` from each forward production at
  grammar-load time.  Implementation pattern at parent plan
  lines 562–568.

### Multi-return reverse signature on `Ops`

Parent plan §Step 6 lines 577–610 commits to a multi-return form for
reverse:

```python
@staticmethod
def liftReverse(Y):
    """Convenience for lift(Y, inverse=True) returning a tuple."""
    return Ops.lift(Y, inverse=True)   # returns (X1, X2)
```

Existing reverse helpers ([Ops.conjunctionReverse](../../bin/Layers.py)
at the Layers.py:4252 area, [Ops.disjunctionReverse](../../bin/Layers.py)
at :4262) take an extra `(y, W)` codebook-search signature and
return a single recovered left operand; Step 6 needs a thin wrapper
that supplies `W` from the layer context (the pattern is already
established at [Spaces.py:936 `Basis.conjunctionReverse`](../../bin/Spaces.py))
and returns the tuple.

---

## Consumers to update

1. **The grammar loader** — currently parses XML rules out of
   [data/MentalModel.xml](../../data/MentalModel.xml) (or whichever
   config the model is built from); needs an additional path that
   reads `data/grammar.cfg` and builds the rule table from the
   explicit-op form.  Pick: keep both paths during transition, or
   gate via XML config.

2. **The parser's apply-rule step** — currently routes through
   `SyntacticLayer._RULE_METHODS` ([Language.py:465-489](../../bin/Language.py))
   which maps rule name → `(forwardName, reverseName, binary_arity)`
   and dispatches into method bodies on `SyntacticLayer`.  After
   Step 6, the rule table comes from `grammar.cfg` and the dispatch
   target is the `Ops` method named in the RHS, called with the
   constituent activations as positional args.

3. **Codebook initialization in `bin/Spaces.py`** — reserve
   category-vector slots for newly-introduced LHS categories (`VO`,
   any `query` result types, etc.).  The category codebook lives in
   `WordSpace.category_codebook` (per spec §B6); slot reservation
   happens at WordSpace construction.

4. **Step 5 deferrals** (per *Pre-state* above) — Step 6 is the
   right place to decide:
   - Whether grammar rule application calls
     `Ops.lower(left, right, mode='AND')` directly (current) or
     routes through `space.pi.forward(packed, binary=True)` (auto
     top-2).  The dispatcher built in Step 6 makes either choice
     practical to wire per-rule.

---

## File map

Files Step 6 will touch:

**Code:**
- `data/grammar.cfg` (NEW or extended) — rule table in explicit-op
  form per parent plan §Step 6 lines 440–558.
- `bin/Language.py` — grammar loader extension; `_RULE_METHODS`
  either deprecated in favor of a runtime rule table loaded from
  `grammar.cfg`, or kept as a hard-coded fallback.
- `bin/Layers.py` — multi-return reverse wrappers for `Ops.lift` /
  `Ops.lower` (the `liftReverse(Y)` form returning a tuple).
- `bin/Spaces.py` — category codebook slot reservation for new
  LHS categories.

**Docs (sync as part of Step 6 implementation):**
- [doc/Language.md](../Language.md) — grammar-cfg dispatch
  description; rule table format.
- [doc/Logic.md](../Logic.md) — note multi-return reverse on
  `Ops.lift` / `Ops.lower`.

Files Step 6 must NOT touch:
- The Pi / Sigma layer internals ([bin/Layers.py](../../bin/Layers.py)
  `PiLayer` / `SigmaLayer`) — unless picking up one of the deferred
  Step 5 items, in which case isolate that change to its own commit.
- The Pipeline routing — Step 6 is rule-table and dispatch only;
  the Pipeline already routes activations through the right layers.

---

## New tests required

`test/test_grammar_cfg_dispatch.py` (or extend the existing
`test/test_grammar_derivation.py`) covering:

1. **`grammar.cfg` parses cleanly** — every row produces a valid
   rule object with op-name, arity, and arg category list.
2. **Round-trip on the rule table** — for each forward production,
   the derived reverse production is well-formed.
3. **Dispatch correctness** — for a representative set of rules,
   the dispatcher invokes the named `Ops` method with the right args
   and produces an output equal to the explicit hand-written call.
4. **No regression on existing grammar tests** — the Step 4
   acceptance sweep still passes (179 / 6 xfail baseline).
5. **Optional**: smoke test for the binary-rule dispatch decision
   from Step 5 deferral item 3 (whether per-rule application uses
   `Ops.lower` direct or `forward(binary=True)` packed).

---

## Acceptance

- `data/grammar.cfg` parses cleanly into the rule table; the
  dispatcher produces the same output as the hand-coded
  `_RULE_METHODS` path on the test grammar.
- Existing grammar derivation tests pass on the new dispatch path:
  - `test/test_grammar_derivation.py`
  - `test/test_toy_grammar.py`
  - `test/test_partition_grammar_rewrite.py`
  - `test/test_reasoning.py`
- `test/test_pi_sigma_ownership.py` (Step 4) and
  `test/test_ops_layer_wiring.py` (Step 5) continue to pass.
- No new deprecation warnings introduced.
- Acceptance sweep (the same set Step 4 / Step 5 ran):
  179 passed + 6 xfailed (or a documented count if the rule-table
  consolidation surfaces a behavior shift worth keeping).

---

## Risks / things to watch

- **Rule-table consolidation surfaces ambiguity.**  The `#`-marked
  uncertain rules in `todo.md` get explicit annotations in the new
  table; some of those annotations may not match the parser's
  current implicit behavior.  Watch for tests that exercise
  modally-modified S or VP — the proposed `intersection(MP, S)` /
  `intersection(MP, VP)` annotations may need adjustment.

- **`grammar.cfg` parser as a new dependency surface.**  Keep the
  parser minimal — line-oriented, comment-prefixed, no third-party
  config language.  The rule body is a single function call
  expression; a small recursive-descent parser is sufficient.

- **Backward compatibility with `_RULE_METHODS`.**  Decide up front
  whether the new dispatch path replaces the old one (clean cut) or
  runs alongside it (gated).  A clean cut is simpler but blocks
  rollback if the grammar tests regress; the gated form costs a
  small amount of code complexity and one config knob.

- **Multi-return reverse changes call shapes.**  Existing callers
  of `Ops.liftReverse(result, right)` (analytic single-operand
  inverse, [bin/Layers.py:4426](../../bin/Layers.py)) and
  `Ops.lowerReverse(result, right)` ([:4503](../../bin/Layers.py))
  return a single recovered operand.  The new convention
  `liftReverse(Y) -> (X1, X2)` returns a tuple.  Either deprecate
  the analytic form (it's marked `# XXX deprecated alias body` per
  spec §Q5) and keep callers on the multi-return form, or pick a
  different name (e.g., `liftReverseAll(Y)`) for the multi-return.

- **Deferred Step 5 items.**  If Step 6 also resolves any of the
  three deferred items from Step 5 (N-ary Ops wiring, reverse Ops
  wiring, or Language.py grammar body migration), keep each in its
  own commit so revert is surgical.  Don't bundle them with the
  grammar-cfg dispatch change.

---

## Sequencing within Step 6

Smallest blast first:

1. **Define `data/grammar.cfg` format** and write a minimal parser
   in [bin/Language.py](../../bin/Language.py).  Output: a list of
   `(lhs, op_name, arg_categories)` tuples.  No dispatch wiring
   yet; just parse and assert structure.
2. **Generate the reverse rule table** mechanically per the
   parent plan's pattern (lines 562–568).  Stored alongside the
   forward table.
3. **Add the multi-return reverse wrappers** (`Ops.liftReverse(Y) →
   (X1, X2)`, `Ops.lowerReverse(Y) → (X1, X2)`).  Decide whether to
   shadow the existing analytic-inverse names or pick new names.
4. **Wire the dispatcher** to call into `Ops` methods (or
   `space.pi.forward` / `space.sigma.forward`) per the rule table.
   Run the existing grammar tests; expect green.
5. **Reserve category-vector slots** for newly-introduced LHS
   categories (`VO`, query result types, etc.) in
   [bin/Spaces.py](../../bin/Spaces.py) WordSpace construction.
6. **Decide on the Step 5 deferral 3** (binary rule application:
   `Ops.lower` direct vs `forward(binary=True)` packed) and wire
   the chosen path.  Run the full acceptance sweep.

Steps 1–3 are reversible without touching consumers.  Step 4 is the
behavior-touching commit; isolate it.

---

## Verification commands

Per memory: tests via `basicmodel/.venv/bin/python -m pytest`.
Never run `make train` locally.  User manages git commits.

```bash
# unit-level (after each sequencing step)
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_grammar_cfg_dispatch.py -v   # new in Step 6
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_grammar_derivation.py -v
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_ops_layer_wiring.py -v       # Step 5 baseline
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_pi_sigma_ownership.py -v     # Step 4 baseline

# acceptance sweep (after sequencing step 4 — the dispatcher wiring)
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
  basicmodel/test/test_pi_sigma_ownership.py \
  basicmodel/test/test_ops_layer_wiring.py
```

Pre-Step-6 baseline: Step 5's totals (Step 4 baseline + new
`test_ops_layer_wiring.py` cases — exact count pending the venv
torch reinstall blocking Step 5 verification, see *Pre-state* note
above).  Step 6 acceptance: same totals plus the new
`test_grammar_cfg_dispatch.py` cases.

---

## Last task: create the Step 7 handoff

After Step 6 lands and the acceptance sweep passes, write the Step
7 handoff at
`doc/plans/<date>-step7-consumer-migration-handoff.md` covering the
parent plan §Step 7 (consumer migration: removing the
`SymbolicSpace.layer` shim, deprecating the legacy 2-arg `Ops.lift`
/ `Ops.lower` aliases marked `# XXX`, and any other migrations the
prior steps left as `XXX` markers).  The Step 7 handoff should:

1. Reference Step 6's outcome (`grammar.cfg`-driven dispatch live;
   multi-return reverse on `Ops`; category slots reserved).
2. Identify the consumers reaching `SymbolicSpace.layer` by name
   ([Models.py:3669](../../bin/Models.py),
   [test_reasoning.py:283](../../test/test_reasoning.py) — see Step
   4 handoff §Pre-state) and migrate each to
   `model.conceptualSpace.pi`.
3. Remove the `SymbolicSpace.layer` `@property` shim once consumers
   are clean.
4. Audit for remaining `# XXX` markers from Steps 1–6 and resolve
   each (deletion, migration, or upgrade-in-place).
5. Acceptance: full sweep passes; `grep "XXX" bin/` returns only
   intentional / out-of-scope markers.
6. Conclude with the same recursion if there are further plan
   steps; otherwise close out the lift / lower / bivector refactor.

The Step 7 handoff is a deliverable of this Step 6 work — write it
after the implementation lands and the acceptance sweep is green,
but before declaring Step 6 complete.
