# Inverse-recommendation for union / intersection (conjunction / disjunction)

## Context

`union`/`intersection` (a.k.a. `disjunction`/`conjunction`) are lossy lattice
ops (`max`/`min`). Today their inverse is one of:

- `IntersectionLayer.reverse` / `UnionLayer.reverse` → `(parent, parent)` —
  trivial, no information recovery ([Layers.py:2187](../../bin/Layers.py),
  [Layers.py:2243](../../bin/Layers.py)). **Out of scope** (user chose
  "replace codebook search", not "wire into Layer.reverse"; these C-tier
  layers have no codebook handle).
- The codebook-search path `_binary_op_inverse_impl` → brute-force K² pair
  search returning **only the left operand** ([Layers.py:7352](../../bin/Layers.py)),
  surfaced via `conjunctionReverse`/`disjunctionReverse`, `Basis.*Reverse`,
  `Basis.lift/lower(inverse=True)`, and `liftReverseAll`/`lowerReverseAll`.

We replace that codebook-search core with a structured, mereology-guided
recommendation that returns **both operands** `(x1, x2)`, per the user's
spec. Goal: a principled inverse that proposes a real `(x1, x2)` whose
`op(x1, x2) ≈ y`, drawn from the **Symbolic codebook**
(`SymbolicSpace.subspace.what.W`, shape `[V_sym, 2]`).

## Algorithm (monotonic lattice; operands are `[pos, neg]` bivectors)

Candidate set `C` = augmented codebook: the learned codebook rows plus two
synthetic sentinels — **⊥ "nothing"** (`zeros(1, D)`) and **⊤ "everything"**
(`ones(1, D)`). Sentinels guarantee the hard filter is never empty (this is
the agreed relaxation: no fallback, the filter always has ⊥ and ⊤).

Per result position (vectorized over `N = prod(result.shape[:-1])`, `K`
augmented candidates):

**`union` inverse** `y = union(x1, x2)` (`union = max`):
1. `x1 = argmax_{S ∈ C, partOf(S, y)} norm(S)` — largest part `≤ y`.
2. `r = y − x1` (copart / residual; vector form `y − x1`).
3. `x2 = argmin_{S ∈ C, wholeOf(S, r)} norm(S)` — smallest part `≥ r`.
4. return `(x1, x2)`.

**`intersection` inverse** `y = intersection(x1, x2)` (`intersection = min`):
1. `x1 = argmin_{S ∈ C, wholeOf(S, y)} norm(S)` — smallest part `≥ y`.
2. `x2 = argmin_{S ∈ C, wholeOf(S, y), S ≠ x1}` of
   `norm(overlapOf(S, x1))`, tie-broken by smaller `norm(S)` — smallest
   part `≥ y` that minimizes overlap with `x1`.
3. return `(x1, x2)`.

Infeasible candidates are masked (score `±inf`) before `argmin/argmax`.
Size metric is `Ops.norm` (last-dim L2) — the codebase's only canonical
size ([Layers.py:6976](../../bin/Layers.py)). Cost: `O(N·K)` (cheaper than
today's `O(N·K²)`).

## New helper methods (on `Ops`, beside the mereology block ~[Layers.py:7778](../../bin/Layers.py))

Plain `@staticmethod`s (not `_OpHandle`-bound — no grammar surface needed):

- `Ops.partOf(S1, S2)` → elementwise `S1 ≤ S2` (per-position boolean over
  the last dim, reduced with `.all(dim=-1)`), after order-align.
- `Ops.wholeOf(S1, S2)` → elementwise `S1 ≥ S2`, after order-align.
- `Ops.overlapOf(S1, S2)` → elementwise zero-directed min — reuse
  `Ops._radmin` semantics ([Layers.py:6987](../../bin/Layers.py)).

**Order-align ("same order, upcast if necessary") — interpretation to
confirm:** this codebase has *no* formal multivector grade system; the only
"orders" present are scalar post-codebook activation `[…]` vs. bivector
`[…, 2]`. Reading: align operands to the common (higher) last-dim — upcast
a scalar to a bivector via the existing scalar→bivector lift used in
`set_activation` (else zero-pad/broadcast to the wider last dim). At the
actual reverse-search site both operands are already bivector `D=2`, so the
align is a thin guard in practice. **Flagged for review** in case "order"
means something more specific.

## Codebook augmentation (non-mutating)

In the reverse core, before selection:
```
W = basis.getW()                                  # learned [K, D]
bottom = torch.zeros(1, D, dtype=W.dtype, device=W.device)
top    = torch.ones (1, D, dtype=W.dtype, device=W.device)
C = torch.cat([bottom, W, top], dim=0)            # transient, [K+2, D]
```
Never touches the learned `Parameter` (no `Basis.insert`/`setW`).

## Files to change

**`bin/Layers.py`**
- Add `Ops.partOf` / `Ops.wholeOf` / `Ops.overlapOf` after `Ops.copart`
  (~[Layers.py:7778](../../bin/Layers.py)).
- Add new core `Ops._binary_op_recommend(result, W, op_name, monotonic=True)`
  → returns `(x1, x2)` implementing the algorithm above.
- Rewrite `Ops._binary_op_inverse_impl` ([Layers.py:7352](../../bin/Layers.py))
  to delegate to `_binary_op_recommend` and **return the pair**.
- `Ops.conjunctionReverse` / `Ops.disjunctionReverse`
  ([Layers.py:7326](../../bin/Layers.py), [Layers.py:7339](../../bin/Layers.py)):
  return the `(x1, x2)` pair; update docstrings ("returns the pair
  (x1, x2)", drop "best-matching left operand").
- `Ops.liftReverseAll` / `Ops.lowerReverseAll`
  (~[Layers.py:7525](../../bin/Layers.py), ~[Layers.py:7625](../../bin/Layers.py)):
  pass the real pair through (drop the `(recovered, Y)` placeholder; the
  `W is None` branch still returns `(Y, Y)`).

**`bin/Spaces.py`**
- `Basis.conjunctionReverse` / `Basis.disjunctionReverse`
  ([Spaces.py:986](../../bin/Spaces.py), [Spaces.py:999](../../bin/Spaces.py)):
  propagate the pair; `W is None` branch returns `(result, result)`
  (shape-correct pair, mirroring the prior single-tensor fallback).
- `Basis.lift` / `Basis.lower` inverse branches
  ([Spaces.py:1030](../../bin/Spaces.py), [Spaces.py:1036](../../bin/Spaces.py),
  [Spaces.py:1055](../../bin/Spaces.py), [Spaces.py:1061](../../bin/Spaces.py)):
  return the pair; `W is None` branch returns `(X1, X1)`.

**Generate consumer (verify + adjust)**
- Binary-rule inverse is delegated to chart generate (not the unary
  `SyntacticLayer._write_subspace`, [Language.py:4225](../../bin/Language.py)).
  Locate the binary-rule generate consumer (`host_layer.generate(parent)` /
  chart generate in `bin/Models.py` / `bin/Language.py`) and confirm it
  unpacks `(x1, x2)` as the two children; adjust if it still assumes a
  single tensor. (This is the one consumer not yet fully traced — must be
  pinned during implementation, not assumed.)

## Tests

- Update `test/test_ops_lift_lower.py:209-220`
  (`test_lower_and_inverse_via_W_matches_conjunctionReverse`): unpack
  `x1, x2 = Ops.conjunctionReverse(...)`, assert shapes of both.
- Add `test/test_inverse_recommend.py`:
  - `partOf/wholeOf/overlapOf` truth tables (incl. ⊥/⊤).
  - union inverse: `union(x1, x2) ≈ y` for codebook-drawn `y`; `x1 ≤ y`;
    `x2 ≥ y−x1`; sentinels guarantee non-empty selection.
  - intersection inverse: `x1 ≥ y`, smallest; `x2 ≥ y` and minimizes
    `overlapOf(x2, x1)`; `intersection(x1, x2) ≈ y`.
  - empty learned codebook → selection still works via ⊥/⊤.
- Confirm unchanged: `test/test_grammar_binary_ops.py` Intersection/Union
  `decompose` still returns `(parent, parent)` (those layers untouched).
- Regression: `test/test_invertibility.py`, `test/brick_recon.py`.

## Verification

1. `cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"`
2. `python -m pytest test/test_inverse_recommend.py test/test_ops_lift_lower.py test/test_grammar_binary_ops.py test/test_invertibility.py -q`
3. `python -m pytest test/brick_recon.py test/test_basicmodel.py -q` (recon regression).
4. Manual sanity: build a small SymbolicSpace basis, pick `y = union(W[a], W[b])`, assert `Ops.disjunctionReverse(y, y, W)` returns `(x1, x2)` with `union(x1, x2) ≈ y` and `x1`/`x2` satisfying the order constraints; repeat for intersection.

## Critical files

- [bin/Layers.py](../../bin/Layers.py) — `Ops` mereology + reverse core (primary).
- [bin/Spaces.py](../../bin/Spaces.py) — `Basis` reverse wrappers + codebook access.
- [bin/Language.py](../../bin/Language.py) / [bin/Models.py](../../bin/Models.py) —
  binary-rule generate consumer (verify pair unpacking).
- [test/test_ops_lift_lower.py](../../test/test_ops_lift_lower.py),
  [test/test_grammar_binary_ops.py](../../test/test_grammar_binary_ops.py),
  [test/test_invertibility.py](../../test/test_invertibility.py).
