# Reintroduce Cross-Position Mixing via `nInputDim` Flatten

**Status:** spec — proposes restoring butterfly-style position mixing as
an XML-configurable knob using existing layer infrastructure. Zero new
classes. No bivector resurrection. Compatible with the
`_active_payload` retirement migration (Stage 4 landed
2026-05-21; see `doc/plans/2026-05-21-active-payload-retirement.md`).

## Problem

`test/test_mm_xor.py::TestMMXorConvergence::test_convergence` and
`test_learns_xor_signal` plateau at ~0.25 loss against a 0.20 / 0.15
threshold. The XOR objective requires combining information across
input positions; the current architecture's only learnable
cross-position path is the grammar layer at the SymbolicSpace tier
(`S = intersection(S, S)`, `S = union(S, S)`, `S = not(S)`),
chart-dispatched by `WordSubSpace.compose`. That path is bimodal in
learnability — per the test docstring, `~30% of random inits fall
into a zero-loss basin, the rest plateau near 1/6`. The plateau is
the failure-mode basin.

The current `ConceptualSpace.sigma_percept = SigmaLayer(D, D)` is
per-position only (operates on the feature axis of a `[B, N, D]`
tensor; never mixes across `N`). PerceptualSpace's `hasAttention=false`
in MM_xor. No other inter-`N` mixing primitive exists in the forward
pipeline.

## History

Git history confirms two earlier removals that contributed:

```
36d978d Add Conceptual Basis and Bivector Activation
4a324e1 Switch to bivectorOutput and get MM_5M working
...
8d66cdd Bivector retirement Phase 0
ff4f8d4 Bivector retirement Phase 1
a716c5a Remove bivectors in forward/reverse paths
```

```
111786f Update to ramsified algorithm butterflies.
4d290d1 get ramsified butterflys working.
2b9e38e refactoring for the run loop and ramsified butterflies.
d7f0fd5 begin cleanup of butterfly and grammar pathways
b6e291b Clean up butterfly code, remove ramsified.
```

The historical "ramsified butterflies" cascaded `PiLayer` instances
over `level_shapes` — each level a different position-axis dimension
`n_t`. The pre-removal code (search any tag before `b6e291b` for
`pi_layers`, `level_shapes`, or `_hierarchical`):

```python
if level_shapes is not None and len(level_shapes) >= 1:
    self._hierarchical = True
    self._level_shapes = level_shapes
    self.pi_layers = nn.ModuleList()
    for t, (n_t, d_t) in enumerate(level_shapes):
        self.pi_layers.append(
            PiLayer(d_t, nSymbolDim, invertible=True, monotonic=True))
```

The cascade + permutation **wiring** was removed; the per-stage
primitive (`PiLayer` / `SigmaLayer`) survived.

Test docstring `test/test_mm_xor.py::TestMMXorConvergence::test_convergence`
preserves the consequence as a comment:

> Threshold relaxed from 0.15 -> 0.20 after the SS-tier sigma layer
> was removed: the cascade now has T learned PiLayers (CS only)
> instead of 2T (CS + SS), so 200 epochs reaches a higher floor on
> this synthetic task.

The current 0.25 plateau is one notch past that relaxed 0.20 — the
removal of butterflies cost the model another notch of cross-position
expressivity.

## Discovery: the mechanism already exists

`Space.forwardBegin` / `Space.forwardEnd` (`bin/Spaces.py:6441-6500`)
already implement a flatten-aware reshape pair:

```python
# forwardBegin
x = vspace.materialize()                          # [B, N, D]
if self.nInputDim != -1:
    self._pre_reshape_input = (x.shape[1], x.shape[2])
    x = x.reshape(x.shape[0], -1, self.nInputDim) # [B, N*D/nInputDim, nInputDim]
```

And `SubSpace.getEncodedOutputSize` (`bin/Spaces.py:4749-4765`)
computes the matching output dim so `forwardEnd` reshapes back to
the SubSpace's expected `[B, oS[0], nOutputDim]`:

```python
if self._nInputDim == iS[1]:
    return self._nOutputDim
oS = self.objectEncoding.outputShape
return oS[0] * self._nOutputDim * self._nInputDim // (iS[0] * iS[1])
```

So `SigmaLayer(input, output)` in ConceptualSpace already inflates
to match the requested chunk width. The dial is `nInputDim`:

| `nInputDim` | Reshape | Effective mixing |
|---|---|---|
| `D` | `[B, N, D]` | Per-position only (current) |
| `2*D` | `[B, N/2, 2D]` | One butterfly level — adjacent pairs |
| `4*D` | `[B, N/4, 4D]` | Two-position chunks |
| `N*D` | `[B, 1, N*D]` | Full cross-position in one Sigma pass |

The historical butterfly cascade was a sequence of these reshapes
with permutations between; one such reshape gives most of the
benefit for small `N` (XOR has N=2-8 effective positions).

## Proposed change

**Single-line XML config in MM_xor.xml** to enable cross-position
mixing without touching code:

```xml
<ConceptualSpace>
    <nInput>8</nInput>
    <nInputDim>80</nInputDim>      <!-- was 10 (D), now N*D for full mix -->
    <nVectors>8</nVectors>
    <nDim>10</nDim>
    <nOutput>8</nOutput>
    <nOutputDim>80</nOutputDim>    <!-- was 10, now N*D -->
    <invertible>true</invertible>
</ConceptualSpace>
```

(Or `<nInputDim>20</nInputDim>` for pair-level mixing — fewer
parameters, less expressive.)

That's the entire change for the proof-of-concept. The existing
`ConceptualSpace.sigma_percept = SigmaLayer(input, output, ...)`
where `input = self.subspace.getEncodedInputSize()` (`Spaces.py:9731`)
will pick up the new `nInputDim` and build a 80→80 Sigma instead of
a 10→10 Sigma. `forwardBegin` flattens, `sigma_percept` mixes
end-to-end, `forwardEnd` reshapes back. Round-trip invertibility
preserved (LDU under the hood).

## Validation plan

1. Apply the XML change to `data/MM_xor.xml` (one or two-line edit).
2. Run `test/test_mm_xor.py::TestMMXorConvergence::test_convergence`
   and `::test_learns_xor_signal`. Expect convergence below 0.20
   / 0.15 in the seeded `for seed in (42, 123, 7)` loop without
   flakiness.
3. Run `test/test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols`
   to confirm shape contracts still hold.
4. Spot-check `test/test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs`
   for no shape / invertibility regressions.
5. Run `test/test_active_payload_audit.py` to confirm the
   `_active_payload` retirement audit baselines stay at zero (the
   XML change doesn't touch the migration's contract).
6. Run a broader sweep against the existing MM_xor users (anything
   that imports `data/MM_xor.xml` or its variants like
   `MM_xor_loopback.xml`).

## Scope decisions

* **Single-level butterfly (one Sigma over `N*D`)** is the right
  first move, not a multi-level cascade. The historical "ramsified"
  cascade was O(N log N) parameters with permutations; for MM_xor's
  N=8 the single-level full-mix is O(N²·D²) parameters
  (80×80 = 6400 vs the historical ~800), but parameter count isn't
  the constraint at this scale — convergence reliability is.
* **No new layer class.** `PairwiseSigmaButterfly` etc. are
  premature; the existing `nInputDim` reshape is the butterfly
  primitive.
* **No bivector resurrection.** The scalar activation contract from
  Stage 4 of the `_active_payload` migration stands. Cross-position
  mixing operates on scalar features end-to-end.
* **No `setW` band-aid.** The flatten/un-flatten happens before /
  after the SigmaLayer call; no codebook prototype slot is touched.
* **`useGrammar` and `useButterflies` are still mutually exclusive**
  per the historical comment in `Models.py` (`b6e291b`'s pre-removal
  state). MM_xor.xml has `<WordSpace><language>` blocks defining
  grammar — those callers should turn the grammar off when enabling
  full-flatten ConceptualSpace mixing, to avoid double-counting
  cross-position information through two paths. Alternatively, keep
  grammar on with a smaller `nInputDim=2*D` (pair-only mixing) so
  the grammar still handles the macroscopic composition.

## Open questions for the new conversation

1. **Should the flatten happen at ConceptualSpace, SymbolicSpace, or
   both?** Historically the cascade had `pi_layers` on SymbolicSpace
   (the snippet quoted above). The C-tier fold is the obvious place
   for percept→concept cross-position mixing; the S-tier fold did
   another round before the SS-sigma removal noted in the
   convergence-threshold comment. Each location is independently
   configurable via its own `<nInputDim>`.
2. **Should `nOutputDim` match `nInputDim` (square inflation) or
   reduce (e.g., N*D → D)?** Square keeps the per-position shape
   `[B, N, D]` after `forwardEnd` reshapes back. Non-square would
   require a different output reshape contract.
3. **Permutations between cascaded levels?** Skipped for the
   single-level proof of concept. If multi-level butterflies are
   needed later, the bit-reversal permutation between Sigmas is
   what FFT-style butterflies do. A `Permutation(N)` op with a
   fixed bit-reversal index tensor is trivial; the historical
   "ramsified" structure may have used a different permutation
   pattern worth recovering.
4. **MM_5M and MM_grammar configs:** these are larger and use
   `useGrammar=true`. They likely don't want full-flatten (would
   blow up parameter count); they might benefit from pair-level
   mixing (`nInputDim = 2*D`) as a structural prior on top of
   grammar. Out of scope for the XOR-focused first pass.
5. **Effect on training stability / lr:** the inflated Sigma has
   more parameters; the existing learning rate (0.01 for MM_xor)
   may need adjustment. Worth confirming with the seeded loop.

## Files to read before starting

1. `doc/specs/2026-05-21-subspace-slot-architecture.md` — the SubSpace
   storage contract the migration enforces; the butterfly change is
   orthogonal to it but must respect `set_event` / `set_what` /
   `materialize` boundaries.
2. `bin/Spaces.py:6441-6500` — `forwardBegin` / `forwardEnd` reshape
   pair (the existing flatten mechanism).
3. `bin/Spaces.py:4743-4765` — `getEncodedInputSize` /
   `getEncodedOutputSize` (computes the matching layer dim).
4. `bin/Spaces.py:9655-9800` — `ConceptualSpace.__init__` and
   `sigma_percept` construction.
5. `bin/Layers.py:1762-1900` — `SigmaLayer` (operates on the
   flattened feature axis, supports `invertible=True` LDU
   parameterization).
6. `data/MM_xor.xml` — the config to edit for the proof of concept.
7. Historical `pi_layers` butterfly cascade at
   `git show 4d290d1^:bin/Spaces.py | sed -n '4490,4550p'` (Symbolic
   space pre-removal).
8. `test/test_mm_xor.py` — the convergence tests this targets.

## Done when

* MM_xor.xml's ConceptualSpace (and optionally SymbolicSpace) uses
  `nInputDim`/`nOutputDim` > `nDim` to enable cross-position mixing.
* `test_mm_xor.py::TestMMXorConvergence::test_convergence` and
  `::test_learns_xor_signal` pass with their original thresholds
  (0.20 / 0.15) without flakiness across the seeded loop.
* No regression in `test/test_active_payload_audit.py` baselines.
* No regression in the wider MM_xor / MM_xor_loopback / MM_5M test
  suites (or the regressions are explicitly documented and
  mitigated).
* The `_pre_reshape_input` round-trip continues to work (forwardBegin
  records the shape, forwardEnd uses it to reshape back).

## Out of scope

* Multi-level butterfly cascade with explicit permutations
  (`PairwiseSigmaButterfly` wrapping class). Useful for larger N
  configs; defer until single-level proves out.
* Reviving the historical `<useButterflies>` config flag for explicit
  enable/disable. The `nInputDim` knob already provides per-config
  control; a high-level flag is redundant.
* Changes to grammar layer or chart parser. The grammar path stays
  as a separate (orthogonal) cross-position mechanism; configs that
  want both should be deliberate about it.
* Bivector activation re-introduction. The Stage 4 scalar contract
  stands.
* Touching the `_active_payload` migration. The butterfly change
  doesn't interact with codebook-bearing slot storage.
