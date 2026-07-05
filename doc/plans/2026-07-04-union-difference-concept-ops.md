# Union/Difference — the residual-bearing op pair on concepts

> **Proposal (Alec, 2026-07-04):** "implement a union/difference that
> operate on concepts, analogous to the conjunction/disjunction that
> operate on symbols." Implemented same-day on Alec's directive.
> **NAMING FINAL (Alec, 2026-07-05): "union/difference is best: fusion
> is a different operator"** — the pair holds the ``union``/``difference``
> rule names; the saturating lattice max vacated ``union`` and is now
> ``join`` (``JoinLayer``, matching ``Mereology.join_from_bottom``; a
> DIFFERENT historical FusionLayer was retired 2026-05-04, confirming
> the collision). The initial landing used ``fusion``; renamed same
> night, sweep: 11 config XMLs (``union(S,S)`` → ``join(S,S)``), the
> registry/fixity tables, the Layers re-export, and every test/doc
> reference.

## Hypothesis (measured, 2026-07-04 blind-bar characterization)

Inverses can only recover constituents if the composition carries an
exact RESIDUAL, and with ONE symbol per word there is no ensemble
redundancy to compensate:

- The symbol-layer folds are lattice ops — `join_from_bottom` $=
  \tanh(\sum_k \mathrm{atanh}(part_k))$ (Mereology.py:69), the grammar's
  `union`/`intersection` (RadMax/RadMin), `conjunction`/`disjunction`
  (DoT kernels). All saturating, `invertible=False`, `(parent, parent)`
  pseudo-inverses: conjunction/disjunction-class, residual-FREE.
- The `[0,1]` presence cube bunches margins structurally: the xor
  store's 18 rows have pairwise cosine 0.267–0.827 (median 0.34, one
  positive cone, nothing orthogonal); under reverse degradation every
  query collapses toward the cone's shared component (the measured
  0.55-bunched rankings, the NUL shadow, 'loving'→'hello').
- The one op that inverts cleanly on the live path is the additive
  linear combine (verified 6/6 calls, exact on the surviving rank) —
  the union/difference class.

## The pair

`UnionLayer` (rule `union`) and `DifferenceLayer` (rule
`difference`), arity-2 CS-space_role GrammarLayers:

- $\mathrm{union}(a, b) = a + b$ — the mereological sum on concept
  content. No tanh, no clamp, no normalize: the residual IS the point.
- $\mathrm{difference}(w, a) = w - a$ — the exact residual (recovers
  $b$ from $\mathrm{union}(a, b)$ to float rounding; bit-exact on
  integer-valued content). Order-directional (T3 fixity, like `part`).
- `UnionLayer.reverse(parent)` $= (parent, \mathbf{0})$ — the
  $\emptyset$-decomposition ($w = w \sqcup \emptyset$): mereologically
  honest, composes back EXACTLY, and is the additive identity the
  serial plan's NULL-word already assumes. NOT a partition-blind
  halving.
- `UnionLayer.reverse(parent, basis=W)` — the PEEL step (mirrors the
  lattice pair's basis-recommender signature, but exact): pick the
  best-matching store row $x_1$, return $(x_1, parent - x_1)$;
  composing back is exact by construction.
- `UnionLayer.peel(whole, basis, max_parts)` — greedy matching
  pursuit: subtract the best row repeatedly until the residual is
  $\approx 0$. Exact for near-orthogonal (signed-domain) parts; the
  positive cone is exactly where it degrades — which is the
  hypothesis, pinned by the contrast test.

## Naming

RESOLVED (see header): the pair is `union`/`difference`; the lattice
max is `join`. `intersection` keeps its name (Alec did not ask for the
`meet` sweep; the lattice pair reads join/intersection).

## Registration

`GRAMMAR_LAYER_CLASSES` (+`union`, `difference`, `join`); fixity
`union: T2_BINARY_INFIX`, `difference: T3_BINARY_DIRECTIONAL`,
`join: T2` (ex-`union`); no Ops
kernel binding (the pair is plain tensor arithmetic). Butterfly cascade
supported for structural parity (pair op $[a{+}b, a{+}b]$ resp.
$[a{-}b, a{-}b]$, $\emptyset$-decomposition reverse). Live: any grammar
may declare `C = union(C, C)` / `C = difference(C, C)`; the serial
reducer dispatches over declared arity-2 ops, so a grammar that
declares union makes the reduce peelable. No shipped .grammar file is
changed here (consumers = the serial-derivation plan: the peel-decode
loop, the reduce op set, the blind-decode content path).

## Tests

`test/test_union_difference_ops.py`: additive compose; exact residual
(bit-exact integer pin + float32 tolerance); $\emptyset$-decomposition
generate-exactness; basis-guided peel step; multiset recovery by peel
(signed rows); THE CONTRAST — the lattice `union` provably destroys the
residual (two distinct operands, same join) while fusion/difference
recovers; registry/fixity/attribute pins.
