# Cross-Position Mixing via Butterfly Mode on `GrammarLayer`

> **Absorption note (2026-05-27):** the architectural framing and design of butterfly mode on `GrammarLayer` have been integrated into the **integrated multi-stage substrate refactor plan** at [`doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md`](../plans/2026-05-26-two-loop-pi-sigma-substrate.md). That plan is the canonical reference for: which spaces own what; the two operating modes (SERIAL/GRAMMATICAL and PARALLEL); how the signal router's per-position copy / reduce marginals replace the prior `self.blend` weight; the `PiLayer` / `SigmaLayer` → `GrammarLayer` inheritance migration; the signal-router-as-canonical-parser commitment; and the staged implementation order. **Butterfly mode is Stage 5 of the master plan.** This document remains as the focused implementation-detail reference for the butterfly cascade itself (per-pair op contract, packed `nn.Parameter` layout, bit-reversal permutations, LDU per node, validation plan). Read both together.

**Status:** spec — proposes a butterfly cascade mode at the `GrammarLayer` base class, inherited by all subclasses (`PiLayer`, `SigmaLayer`, `IntersectionLayer`, `UnionLayer`, …). Enables pairwise operations over all of STM, which is the efficient computation pattern for soft superposition across STM contents. Part of the substrate refactor ([`doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md`](../plans/2026-05-26-two-loop-pi-sigma-substrate.md)). **Changes are limited to layer internals** — no Space-level orchestration, no XML reshape knob, no widening of the layer's effective input / output dim from the Space side.

## Context

The two-loop substrate refactor relocates `pi` from ConceptualSpace to PerceptualSpace, where it handles the CS-side feedback into PS:

```python
class PerceptualSpace:
    def forward(self, IS_subspace, CS_subspace):
        here  = self.codebook.forward( IS_subspace.materialize() )    # MPHF → SS.codebook
        there = self.pi( CS_subspace.materialize() )
        return here + there
```

`self.pi` is the natural first beneficiary of cross-position mixing. In PARALLEL mode with the full sentence flattened, the full mixing would be an `N*D × N*D` weight matrix — too large to manifest naively. Butterfly decomposition reduces this to `O(N · log N · D²)` parameters in a cascade of pairwise small matrices, all kept inside the layer.

Beyond PiLayer, **other GrammarLayer subclasses benefit from the same pairwise structure** — pairwise operations over all of STM is the efficient computation pattern for soft superposition. `IntersectionLayer`, `UnionLayer`, `ConjunctionLayer`, `DisjunctionLayer`, and similar binary grammar operators can compute cross-STM aggregations via the same cascade structure that butterfly provides. Placing the butterfly machinery on `GrammarLayer` (the shared base) lets every subclass opt in via a constructor flag.

## Code / docstring reconciliation

`GrammarLayer` ([`bin/Layers.py:1532-1546`](../../bin/Layers.py)) currently declares:

> Note (2026-04-30): ``PiLayer`` / ``SigmaLayer`` inherit directly from `GrammarLayer` so the parameterized fold layers ARE Grammar layers, not wrapped by separate `IntersectionLayer` / `UnionLayer` adapters.

…but the actual class headers say:

- `class SigmaLayer(Layer):` ([line 1762](../../bin/Layers.py))
- `class PiLayer(Layer):` ([line 3387](../../bin/Layers.py))

This spec assumes the **docstring intent is canonical** and migrates `PiLayer` and `SigmaLayer` to `GrammarLayer` inheritance as a precondition of butterfly being a GrammarLayer-level feature. The migration is small (change base class; verify any `isinstance(layer, GrammarLayer)` consumers handle the now-broader set), and it aligns the code with the existing design intent.

## Problem

`test/test_mm_xor.py::TestMMXorConvergence::test_convergence` and `test_learns_xor_signal` plateau at ~0.25 loss against a 0.20 / 0.15 threshold. The XOR objective requires combining information across input positions; the current architecture's only learnable cross-position path is the grammar layer at the SymbolicSpace tier (`S = intersection(S, S)`, `S = union(S, S)`, `S = not(S)`), chart-dispatched by `WordSubSpace.compose`. That path is bimodal in learnability — per the test docstring, `~30% of random inits fall into a zero-loss basin, the rest plateau near 1/6`. The plateau is the failure-mode basin.

The current `ConceptualSpace.sigma_percept = SigmaLayer(D, D)` is per-position only (operates on the feature axis of a `[B, N, D]` tensor; never mixes across `N`). No other inter-`N` mixing primitive exists in the forward pipeline.

After the 2026-05-26 substrate refactor, the natural site for cross-position mixing moves to PS (`PerceptualSpace.pi`), and the broader pattern (pairwise across STM) becomes a GrammarLayer-level capability.

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

The historical "ramsified butterflies" cascaded `PiLayer` instances over `level_shapes` — each level a different position-axis dimension `n_t`. The pre-removal code (search any tag before `b6e291b` for `pi_layers`, `level_shapes`, or `_hierarchical`):

```python
if level_shapes is not None and len(level_shapes) >= 1:
    self._hierarchical = True
    self._level_shapes = level_shapes
    self.pi_layers = nn.ModuleList()
    for t, (n_t, d_t) in enumerate(level_shapes):
        self.pi_layers.append(
            PiLayer(d_t, nSymbolDim, invertible=True, monotonic=True))
```

The cascade + permutation wiring was removed; the per-stage primitive (`PiLayer` / `SigmaLayer`) survived. The new design returns the cascade — this time entirely inside a single `GrammarLayer` instance, with packed `nn.Parameter` storage rather than per-node submodules.

Test docstring `test/test_mm_xor.py::TestMMXorConvergence::test_convergence` preserves the consequence:

> Threshold relaxed from 0.15 -> 0.20 after the SS-tier sigma layer was removed: the cascade now has T learned PiLayers (CS only) instead of 2T (CS + SS), so 200 epochs reaches a higher floor on this synthetic task.

The current 0.25 plateau is one notch past that relaxed 0.20 — the removal of butterflies cost the model another notch of cross-position expressivity.

## Proposed: butterfly mode on `GrammarLayer`

A `butterfly=True, N=N` constructor flag on the `GrammarLayer` base class. Subclasses (PiLayer, SigmaLayer, IntersectionLayer, …) inherit the flag and the underlying cascade machinery. Each subclass implements the per-pair operator that the cascade applies; the cascade structure (levels, permutations, identity init, packed Parameters) is shared at the base.

### API surface

```python
# All GrammarLayer subclasses accept the same flag:
PiLayer(D, D, invertible=True, butterfly=True, N=8)
SigmaLayer(D, D, invertible=True, butterfly=True, N=8)
IntersectionLayer(butterfly=True, N=8)
# …
```

Same `forward(x: [B, N, D]) -> y: [B, N, D]` interface as the per-position mode. The flag changes the **internal** weight storage and forward / reverse paths only; the layer's external shape contract is unchanged.

### Internal structure (on `GrammarLayer`)

The base class adds the butterfly machinery as a shared substrate:

- `self.butterfly: bool` — flag.
- `self.N: int` — number of positions to mix.
- `self.n_levels = log2(N)` — cascade depth.
- `self.butterfly_W: nn.Parameter` of shape `[n_levels, N // 2, 2D, 2D]` — packed weights for all butterfly nodes. Single packed Parameter, not a ModuleList. Identity init.
- `self.butterfly_perms: torch.Tensor (buffer)` — precomputed bit-reversal permutation indices, shape `[n_levels, N]`. Self-inverse.

Subclasses implement a `_butterfly_pair_op(x_pair, W_node)` method — the per-pair operator. Defaults at `GrammarLayer` to a plain matmul:

```python
def _butterfly_pair_op(self, x_pair, W_node):    # x_pair: [B, M, 2D]; W_node: [M, 2D, 2D]
    return torch.einsum('bmi,mij->bmj', x_pair, W_node)
```

`PiLayer` overrides it with atanh → matmul → tanh; `SigmaLayer` overrides with atanh → matmul → tanh (different math); `IntersectionLayer` / `UnionLayer` override with their respective tetralemma ops; etc.

### Forward / reverse on `GrammarLayer`

The base class provides the cascade orchestration:

```python
def forward(self, x):
    if not self.butterfly:
        return self._per_position_forward(x)      # subclass-defined per-position path
    return self._butterfly_forward(x)

def _butterfly_forward(self, x):                  # x: [B, N, D]
    for level in range(self.n_levels):
        x_pairs = x.reshape(-1, self.N // 2, 2 * self.D)              # pair positions
        W_level = self.butterfly_W[level]                              # [N//2, 2D, 2D]
        x_pairs = self._butterfly_pair_op(x_pairs, W_level)            # per-pair op
        x = x_pairs.reshape(-1, self.N, self.D)                        # un-pair
        x = x[:, self.butterfly_perms[level], :]                       # bit-reversal permute
    return x

def _butterfly_reverse(self, y):                  # symmetric: inverse pair op in reversed order
    for level in reversed(range(self.n_levels)):
        y = y[:, self.butterfly_perms[level], :]                       # self-inverse permute
        y_pairs = y.reshape(-1, self.N // 2, 2 * self.D)
        W_level = self.butterfly_W[level]
        y_pairs = self._butterfly_pair_op_reverse(y_pairs, W_level)
        y = y_pairs.reshape(-1, self.N, self.D)
    return y
```

(Sketch — actual indexing requires care; pair grouping and permutation conventions match the historical FFT-style layout.)

### Invertibility

Each `butterfly_W[level, m]` is a `2D × 2D` matrix. For invertibility, the base parameterizes it through LDU (reusing the existing `InvertibleLinearLayer` machinery, applied per node within the packed Parameter). The cascade composes: `reverse(forward(x)) == x` up to numerical precision. Permutations are self-inverse via bit-reversal symmetry.

### Identity initialization

Each node's `[2D, 2D]` matrix initializes to the identity (via PiLayer / SigmaLayer's `L=I, d=1, U=I` LDU default applied per node). Cascade overall is identity at init — `forward(x) == x` before training.

### Parameter count

`O(N · log N · D²)` for the cascade vs `O(N² · D²)` for a single big matrix.

| N | D | Cascade params | Single-matrix params | Ratio |
|---|---|----------------|----------------------|-------|
| 8 | 10 | 8 · 3 · 100 = 2 400 | 6 400 | 0.38× |
| 32 | 10 | 32 · 5 · 100 = 16 000 | 102 400 | 0.16× |
| 128 | 10 | 128 · 7 · 100 = 89 600 | 1 638 400 | 0.05× |

Savings grow rapidly with N.

### Soft superposition across STM

The pairwise cascade structure is the efficient computation pattern for **soft superposition** across STM contents. STM has 7 slots (Miller cap); other GrammarLayer subclasses (IntersectionLayer, UnionLayer, ConjunctionLayer, DisjunctionLayer, …) can use butterfly mode to compute weighted combinations of all STM pairs without manifesting the full pairwise interaction matrix.

For an `STM` of N items, a soft superposition would be a learned weighted combination of all `N · (N-1) / 2` pairs of items. The butterfly cascade computes this in `O(N · log N)` ops with `O(N · log N · D²)` parameters — vs `O(N²)` ops / params for the naive all-pairs computation.

Concretely: `IntersectionLayer(butterfly=True, N=7)` over STM computes a learned cross-STM intersection cascade, where the layer's parameters control how pairs at each butterfly level combine. The composition pattern matches what `WordSubSpace.compose` is doing today via chart dispatch, but at the layer level with packed Parameters and `O(N · log N)` cost.

## Code reuse: composing with existing `Layer` infrastructure

The base class machinery reuses existing primitives:

- **LDU per node**: the same `InvertibleLinearLayer` LDU code that `PiLayer.layer` / `SigmaLayer.layer` already use. Applied per butterfly node via the packed Parameter tensor.
- **Ergodic noise**: the `ergodic=True` and `set_sigma` propagation already handled by `Layer` flows through cleanly; each butterfly node sees its own noise injection if ergodic mode is on.
- **`reads_activation`, `tier`, `rule_name`**: GrammarLayer's existing class attributes work unchanged; butterfly mode doesn't affect rule registration.

## Use sites

**Immediate (this spec's primary target):**
- `PerceptualSpace.pi = PiLayer(D, D, invertible=True, butterfly=True, N=N)` per the substrate refactor.

**Followups (separate from this spec):**
- `SymbolicSpace.pi` — if the non-idempotent SS loop doesn't already provide enough cross-symbol mixing.
- `IntersectionLayer` / `UnionLayer` / `ConjunctionLayer` / `DisjunctionLayer` over STM — soft-superposition cross-STM aggregation, deferred until grammar overlay design lands.
- Any other GrammarLayer subclass that needs cross-STM pairwise operations.

## Note: the existing `nInputDim` reshape mechanism is orthogonal

`Space.forwardBegin` / `forwardEnd` ([`bin/Spaces.py:6441-6500`](../../bin/Spaces.py)) implement a flatten-aware reshape that lets a layer see a wider effective per-position dim. That mechanism remains a valid Space-level primitive — useful for configurations where the layer needs a different per-position dim than the SubSpace's natural `D`. It is **not** the butterfly impl, and the butterfly change does not touch it.

The original 2026-05-21 version of this spec proposed using `nInputDim` to widen a single `SigmaLayer` to `N*D × N*D` (one big matrix at CS, with a Space-level reshape on either side). That proposal is superseded by GrammarLayer-level butterfly — same diagnostic, different implementation site and structure. Git history preserves the original text.

## Validation plan

1. **Migrate `PiLayer` and `SigmaLayer` to inherit from `GrammarLayer`** (resolve the existing docstring / code mismatch). Verify any `isinstance(layer, GrammarLayer)` consumers still behave correctly with the broader subclass set.
2. **Unit-test butterfly mode** on the base class.
   - `forward(reverse(y)) ≈ y` and `reverse(forward(x)) ≈ x` for random inputs at N = 2, 4, 8, 16.
   - Identity init: `forward(x) == x` at init (within numerical precision).
   - Parameter count matches `O(N · log N · D²)` formula.
   - Each subclass's per-pair op composes correctly with the cascade (PiLayer's atanh/tanh, SigmaLayer's tanh, IntersectionLayer's tetralemma, etc.).
3. **Wire `PiLayer(butterfly=True)` into `PerceptualSpace.pi`** per the substrate refactor.
4. **Run XOR convergence tests**.
   - `test/test_mm_xor.py::TestMMXorConvergence::test_convergence` and `::test_learns_xor_signal` should converge below 0.20 / 0.15 in the seeded `for seed in (42, 123, 7)` loop without flakiness.
   - `test/test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols` confirms shape contracts.
   - `test/test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs` confirms no shape / invertibility regressions.
5. **No regression in `test/test_active_payload_audit.py`** baselines — butterfly is orthogonal to that migration.
6. **Broader sweep against MM_xor / MM_xor_loopback / MM_5M / MM_grammar** to confirm no unexpected interactions.

## Scope decisions

- **Butterfly mode lives on `GrammarLayer`** as a base-class capability. Subclasses opt in via a constructor flag; they implement the per-pair operator while inheriting the cascade machinery.
- **Single packed `nn.Parameter`** of shape `[n_levels, N // 2, 2D, 2D]` for all butterfly weights at a layer. Not a ModuleList of submodules — avoids the module hierarchy overhead and stays Dynamo / fullgraph friendly.
- **`PiLayer` and `SigmaLayer` migrate to `GrammarLayer` inheritance** as a precondition. Resolves the existing docstring / code mismatch and makes the butterfly capability uniformly available.
- **LDU per node**, reusing the existing `InvertibleLinearLayer` machinery applied per butterfly node within the packed Parameter.
- **Bit-reversal permutations** between levels (standard FFT butterfly layout). Other permutation patterns deferred until empirical comparison.
- **Identity init** at every level. Cascade is identity at init.
- **N must be a power of 2**. For non-power-of-2 N, pad to the next power of 2 (caller responsibility, or pad inside the layer with a configurable strategy).
- **No bivector resurrection.** The Stage 4 scalar contract from `_active_payload` retirement stands.
- **No Space-level changes.** All butterfly logic lives inside `GrammarLayer` and its subclasses; Spaces just construct layers with `butterfly=True` where needed.

## Files to read before starting

1. [`doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md`](../plans/2026-05-26-two-loop-pi-sigma-substrate.md) — the substrate refactor that puts `pi` on PS. Butterfly's primary use site is `PS.pi`.
2. [`doc/specs/2026-05-21-subspace-slot-architecture.md`](2026-05-21-subspace-slot-architecture.md) — the SubSpace storage contract; butterfly is orthogonal but must respect `materialize` boundaries.
3. `bin/Layers.py` — `GrammarLayer` ([line 1532](../../bin/Layers.py)), `SigmaLayer` ([line 1762](../../bin/Layers.py)), `PiLayer` ([line 3387](../../bin/Layers.py)), `IntersectionLayer`, `UnionLayer`, and the other GrammarLayer subclasses (lines 2035+).
4. `bin/Layers.py` — `InvertibleLinearLayer` (LDU factorization that gives each butterfly node its invertibility).
5. Historical `pi_layers` butterfly cascade at `git show 4d290d1^:bin/Spaces.py | sed -n '4490,4550p'` — reference for the permutation / cascade conventions.
6. `test/test_mm_xor.py` — the convergence tests this targets.
7. `data/MM_xor.xml` — the config that needs `PerceptualSpace.pi` set with `butterfly=True`.

## Done when

- `GrammarLayer` has butterfly mode (`butterfly=True, N=N` flag) with packed `nn.Parameter` storage, bit-reversal permutations, identity init, and LDU per node.
- `PiLayer` and `SigmaLayer` inherit from `GrammarLayer` (resolving the existing docstring / code mismatch), with subclass-specific `_butterfly_pair_op` implementations.
- `PerceptualSpace.pi` instantiates a butterfly-mode PiLayer when configured for PARALLEL mode (via XML or default).
- `test_mm_xor.py::TestMMXorConvergence::test_convergence` and `::test_learns_xor_signal` pass with their original thresholds (0.20 / 0.15) without flakiness across the seeded loop.
- Forward / reverse roundtrip unit tests pass for butterfly-mode PiLayer / SigmaLayer at multiple N.
- No regression in `test/test_active_payload_audit.py` baselines.
- No regression in the wider MM_xor / MM_xor_loopback / MM_5M / MM_grammar test suites (or regressions are explicitly documented and mitigated).
- `Space.forwardBegin` / `forwardEnd` and the `nInputDim` mechanism remain untouched.

## Out of scope

- Implementation. This spec documents the design; implementation lands after the two-loop substrate refactor.
- Butterfly on `IntersectionLayer` / `UnionLayer` / `ConjunctionLayer` / `DisjunctionLayer` for cross-STM soft-superposition computation. The machinery will be available at GrammarLayer base, but wiring these specific use sites belongs to the grammar overlay design.
- Multi-pattern butterfly (alternative permutations beyond bit-reversal). Defer until empirical comparison shows benefit.
- Reviving the historical `<useButterflies>` XML flag. The new design uses a constructor argument at the layer level, not a Space-level flag.
- Changes to grammar layer dispatch or chart parser. Grammar dispatch stays orthogonal.
- Bivector activation re-introduction. The Stage 4 scalar contract stands.
- Touching the `_active_payload` migration.
- Touching the `nInputDim` / `forwardBegin` / `forwardEnd` reshape mechanism. It remains as a separate Space-level primitive.
