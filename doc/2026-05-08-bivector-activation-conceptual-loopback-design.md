# Bivector Activation + Conceptual Loopback — Design Spec

## Context

Wires the codebook's `project=True` snap into the C-tier forward/reverse so
the C-S boundary IS the bivector projection. Total effect:

- ConceptualSpace's *output* is the per-prototype catuskoti bivector
  `[B, V_C, 2]`.
- ConceptualSpace's *input* is the concatenation `[Perceptual ||
  Symbolic_{t-1}]` — the "right-half loopback".
- PerceptualSpace serves as the subsymbolic substrate (no parallel
  `SubsymbolicSpace` class at runtime).
- `useButterflies=false`, `passThrough=false`, and the loopback is always-on.

Working configuration is `MM_xor.xml`. The single end-to-end test to keep
green is the round-trip reconstruction in `test/test_mm_xor.py` plus
convergence and forward-runs tests.

## Naming

The "wire `Codebook.forward(input, project=True)` into the C-tier" work is
referred to as **ConceptualSpace Bivector Projection** (CSBP).

## Target architecture

### Pipeline shapes (MM_xor)

```
Input          [B, 8, 4]      (text → InputSpace lift)
Perceptual     [B, 8, 4]      (passthrough or codebook-quantized)
Conceptual_in  [B, 8, 4 + 2]  ← P_event || S_event_{t-1}
Conceptual_out [B, 8, 2]      ← per-prototype catuskoti bivector
Symbolic_in    [B, 8, 2]
Symbolic_out   [B, 8, 2]      ← per-prototype catuskoti bivector
Output         [B, 1, 1]      (XOR prediction head)
```

`P_event` survives end-to-end across stages; `S_event_{t-1}` is the previous
iteration's symbolic output. At sentence start `S_event_0 = 0`; PiLayer's
identity-on-zero matches the no-loopback pipeline at order 1.

### Iteration without butterflies

`useButterflies=false`, `conceptualOrder=N` runs N iterations at the same
shape:

```
for t in 1..N:
    C_in_t = [P_event || S_out_{t-1}]      # right-half = previous symbol
    C_out_t = ConceptualSpace.forward(C_in_t)
    S_out_t = SymbolicSpace.forward(C_out_t)
```

P is constant context; S is the iterating channel. The grammar XML drives
the fold sequence; in MM_xor's simplest form (`<S>not</S>`,
`<S>intersection</S>`, `<S>union</S>`) the chart picks rule sequences.

### Bivector everywhere from C onward

Once C emits `[B, V_C, 2]`, every downstream tensor lives in bivector-space
until OutputSpace projection. Per-slot D collapses from `D_C = 6` to `D = 2`.
Deep content lives only in the codebook on `subspace.what`, read indirectly
via `Codebook.forward(x, project=True)` (inner product against each row,
accumulated into pos/neg) and `Codebook.reverse(bivec, project=True)`
(cached SVD pseudo-inverse).

## Implementation outline

Six gateable stages. Stages 1-2 trim before architectural changes; stages
3-6 introduce the bivector + loopback. Butterflies are *not* removed here
(Note A); bivector configs set `useButterflies=false`.

### Stage 1 — Remove `passThrough=true` codepath

With architecture locked in, every Space does work — no "skip" mode.

Touchpoints: `Space.passThrough` short-circuits, `Codebook.passThrough`
short-circuit, `passThrough=not self.codebook` derivation in
`_build_what_basis`, `<passThrough>` XML reads.

XML migrations: `idempotent.xml` (`<passThrough>true</passThrough>` on
PerceptualSpace) needs a real config or rewrite using `<codebook>false</codebook>`.

Acceptance: `passThrough` reads gone; `<passThrough>` removed from XML;
tests green.

### Stage 2 — Make subsymbolic-loopback always-on

Drop `<subsymbolicEnabled>` and `<mode>`; PerceptualSpace serves as the
subsymbolic substrate.

Touchpoints:
- `BasicModel._create_per_stage` and flat path: drop
  `if self.subsymbolicEnabled`. `self.subsymbolicSpace = None` always.
- `mode == "grammar"` / `"parallel"` branches: delete.
- `ConceptualSpace.__init__`: `subsymbolic_widen_dim` becomes derived
  (= `symbolShape[1]`), not flag-gated.
- `_build_combined_input`: iterate only `symbolicSpace_ref`.
- `SubsymbolicSpace` class: **not** removed; stays as a configurable
  variant (regular Space with `<codebook>false</codebook>`).

Validator: replace the "subsymbolicEnabled requires shared nDim" check
with `C.nInputDim == P.nOutputDim + S.nOutputDim`.

### Stage 3 — Per-stage widening

After Stage 2 the loopback is always wired, but only stage 0's
ConceptualSpace is widened. Widen every stage:

```python
stage_widen_dim = symbolShape[1]   # unconditional
```

Wire `symbolicSpace_ref` on every stage's ConceptualSpace:

```python
for t, cs in enumerate(self.conceptualSpaces):
    if t == 0:
        # Stage 0 cold-starts: no previous symbol, _build_combined_input
        # zero-fills the right half.
        ref = self.symbolicSpaces[0]
    else:
        ref = self.symbolicSpaces[t - 1]
    object.__setattr__(cs, 'symbolicSpace_ref', ref)
    object.__setattr__(cs, 'subsymbolicSpace_ref', None)
```

`object.__setattr__` bypasses `nn.Module` submodule registration —
otherwise SymbolicSpace would be in multiple parents in the module tree.

Acceptance: 2-stage MM_xor variant (`conceptualOrder=2`, no butterflies)
forwards through both stages without shape errors.

### Stage 4 — ConceptualSpace Bivector Projection (CSBP)

Wire `Codebook.forward(input, project=True)` into the C-tier forward and
`Codebook.reverse(bivec, project=True)` into the reverse.

**Forward swap** in `ConceptualSpace.forward`:

```python
if self.codebook:
    y = self.subspace.what.forward(y, project=True)        # [B, V_S, 2] bivec
```

The wide-codebook `topK` branch goes away — the bivector is per-prototype
regardless of how many prototypes exist.

**Reverse lift** in `ConceptualSpace.reverse`: prepend a lift step ahead of
the existing `pi.reverse` / syntacticLayer.reverse dispatch:

```python
y = self.reverseBegin(vspace, returnVectors=True)
if self.codebook:
    y = self.subspace.what.reverse(y, project=True)        # [B, V, D] lifted
# pi.reverse / syntacticLayer.reverse ...
```

**SVD cache lifecycle.** `Codebook.project` stores `_project_cache` on the
instance; `project_reverse` reads it. Contract: **one forward followed by
one reverse on the same codebook instance**, no intervening forward. Cache
is per-instance, so per-stage ConceptualSpaces each have their own —
no cross-stage interference.

**Codebook initialization.** `project_reverse` uses `1/S` (inverse singular
values). Random codebook + small S blows up the lift. Mitigation:
SVD-orthogonalize at construction:

```python
with torch.no_grad():
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W.copy_(U @ Vh)   # nearest orthonormal — all singular values 1
self._svd_dirty = True
```

All singular values pinned at 1; `1/S` is identity-ish; lift is
well-conditioned from the first forward call. EMA/training updates deform
from this start.

**Range check.** The bivec `[pos, neg]` has values bounded by `V_in *
max(|input| * |W|)` — not in [-1, 1]. Loosen the range check for
bivector-output spaces: add `kind="bivector"` to normalize with `lo=0,
hi=None` (non-negativity only).

### Stage 5 — Symbolic side mirror

SymbolicSpace's forward / reverse should symmetrically use `project=True`,
so its output is also `[B, V_S, 2]`. Same swaps as Stage 4 but in
`SymbolicSpace.forward` and `reverse`. Legacy VQ snap branches
(`use_vqvae_reversible`, `use_vqvae_nonreversible`, `hard_quantize`) are
vestigial under the bivector regime; kept for legacy XMLs but not exercised
by MM_xor.

### Stage 6 — Downstream consumer sweep

After C and S both emit bivectors:

- `OutputSpace` projection from `[B, 8, 2]` to `[B, 1, 1]` — verify shape.
- `_compute_symbol_terms` / `_emit_symbol_terms` — assumes deep content;
  needs to read the bivec format.
- Loss broadcasting — trace and fix the `5 vs 4` shape error.
- TruthLayer / Mereology readers — verify they consume the bivec correctly.

## Test plan

Each stage leaves the test suite green.

| Stage | Acceptance |
|-------|------------|
| 1 | `passThrough` reads gone; `<passThrough>` removed from XML; tests green |
| 2 | `<subsymbolicEnabled>` and `<mode>` reads gone; SubsymbolicSpace not auto-constructed; validator asserts dim contract |
| 3 | 2-stage MM_xor variant forwards through both stages; each C reads `[P || S_{t-1}]` |
| 4 | `ConceptualSpace.forward` returns `[B, V, 2]`; reverse returns `[B, V, D]`; SVD-orthogonal init keeps round-trip MSE <1e-3 for in-span inputs |
| 5 | `SymbolicSpace.forward` returns `[B, V_S, 2]`; C→S→C round-trip MSE ≤5e-2 from fresh init, decreasing with training |
| 6 | `test_mm_xor.py::test_forward_reverse_reconstructs_input_state` passes (threshold may relax from 1e-2); convergence passes; full `test_basicmodel.py` green |

## Risks

1. **Untrained-codebook amplification.** Resolved by Stage 4
   SVD-orthogonalization (`W ← U @ V.T`).
2. **per-prototype vs per-slot shape.** `[B, V_S, 2]` with `V_S != V_in`
   requires diagonal or outer-product disambiguation. Spec assumes
   `V_S == V_in` for now (true in MM_xor).
3. **Commitment-loss replacement.** Legacy VQ-VAE `commit_loss` pulls the
   encoder toward the codebook. Under CSBP the snap is projection onto
   span(W), not discrete substitution; commit loss may not have the right
   gradient. Decision deferred to Stage 5.
4. **Discourse / TruthLayer interactions.** `record_batch` sees `[B, V_S, 2]`
   instead of `[B, V_S, D_S]`; verify trust-score still makes sense on
   bivector poles. Surfaces in Stage 6.

## Note A — Butterflies removed (2026-05-12 follow-up)

Butterfly mode (`<useButterflies>true</useButterflies>`) has been retired
wholesale. The `_butterfly_*` machinery in Layers.py, the `useButterflies`
XML knob, and the volume-equality validation are all gone. Configs that
previously relied on butterfly halving (MM_5M, MM_400M, MM_xor's
pre-bivector flavor) now use the plain per-stage path. See the Butterfly
Removal handoff for the migration log.

## Current state at handoff

Changes from the rollback session in the working tree but not in `git log`
of the starting point:

- **`Codebook.STE=False` flag** in `Codebook.create`. When True, `forward`
  and `reverse` wrap the snap with `input + (snapped - input).detach()` for
  STE gradient identity. `SymbolicSpace._build_what_basis` passes
  `STE=True`. CSBP replaces the snap; the STE wrap will need to be retained
  or discarded — Stage 4 decision.
- **`WhereEncoding._codebook_registry`** + `allocate_codebook_slice` +
  `Codebook.where_offset` + `addVectors` size assert — sequential per-codebook
  offset allocation in the global where-space.
- **`Models._create_per_stage` `conceptOutputShape`** uses explicit
  `<nOutput>` / `<nOutputDim>` instead of the broken volume formula. Don't
  revert.
- **`XMLConfig.space()`** raises `ValueError` on duplicate scalar config tags.
- **`bin/bm.py`** is defensive about missing `numShards` / `maxDocs` /
  `shardDir` in `<data>` blocks.
- **`TheData.loadInline`** synthesizes deterministic random sentences when
  `<dataset>inline</dataset>` has no `<input>` / `<output>` children.
- **`test/test_partition_resolve.py`** assertions migrated to
  `nWhat == sym.nDim`.

### MM_xor.xml status

Reverted to the pre-bivector-experiment shape (`conceptualOrder=3`,
`useButterflies=true`, all spaces at `nDim=10` / `nOutputDim=10`, no
`<subsymbolicEnabled>`, no `<mode>`, ConceptualSpace
`<codebook>false</codebook>`, SymbolicSpace `<codebook>true</codebook>`
`<useVQVAE>true</useVQVAE>`). The grammar block has explicit per-tier folds.
Tests green at this baseline.

### idempotent.xml status

Uses `<passThrough>true</passThrough>` on PerceptualSpace, which Stage 1
removes. Migration required (rewrite as `<codebook>false</codebook>`
perceptual or delete).

## References

- Rollback plan: [doc/plans/2026-05-07-finish-symbolic-rollback.md](plans/2026-05-07-finish-symbolic-rollback.md)
- Codebook primitives: [bin/Spaces.py](../bin/Spaces.py) `Codebook.project`,
  `Codebook.project_reverse`.
- Idempotent-loop test: [test/test_idempotent_loop.py](../test/test_idempotent_loop.py).
- Combined-input wiring: `_build_combined_input` in [bin/Spaces.py](../bin/Spaces.py).
- Per-stage construction: `_create_per_stage` in [bin/Models.py](../bin/Models.py).
- Subsymbolic plan (predecessor): [doc/plans/2026-05-05-subsymbolic-knowing-handoff.md](plans/2026-05-05-subsymbolic-knowing-handoff.md).
