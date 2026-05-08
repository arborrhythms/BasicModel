# Bivector Activation + Conceptual Loopback — Design Spec

## Context

The post-2026-05-07 rollback landed the architectural simplifications
(codebook width = `nDim`, `default_rule` removal, grammar XML as sole
source of truth, doc updates). It deferred one item from Phase 4
Step 3 — wiring the codebook's `project=True` snap into the C-tier
forward / reverse so the C-S boundary IS the bivector projection.

This spec covers a related cluster of changes whose total effect is:

- ConceptualSpace's *output* is the per-prototype catuskoti bivector
  `[B, V_C, 2]` (not the deep snapped event `[B, V_C, D_C]`).
- ConceptualSpace's *input* is the concatenation `[Perceptual ||
  Symbolic_{t-1}]` -- the "right-half loopback" already wired for
  subsymbolic mode, repurposed.
- PerceptualSpace serves as the subsymbolic substrate (no parallel
  `SubsymbolicSpace` class is constructed at runtime).
- `useButterflies=false`, `passThrough=false`, and the loopback is
  always-on. Trim the codepaths that supported the alternatives.

Working configuration is `MM_xor.xml`. The single end-to-end test we
want to keep green is the round-trip reconstruction in
`test/test_mm_xor.py` plus the convergence and forward-runs tests.

The work below is **not** in the current branch. The current branch
is the rolled-back-and-passing state; this spec describes the next
landing. Before any code in this spec lands we will revert the
in-progress XML edits in MM_xor.xml so the test suite is clean for a
release tag.

## Naming

The "wire `Codebook.forward(input, project=True)` into the C-tier
forward and `Codebook.reverse(bivec, project=True)` into the reverse"
work is referred to throughout this spec as **ConceptualSpace
Bivector Projection** (CSBP). It used to be tracked as Phase 4 Step
3 in the rollback plan.

## Target architecture

### Pipeline shapes (MM_xor as the canonical example)

```
Input          [B, 8, 4]      (text → InputSpace lift)
Perceptual     [B, 8, 4]      (passthrough or codebook-quantized)
Conceptual_in  [B, 8, 4 + 2]  ← P_event || S_event_{t-1}
Conceptual_out [B, 8, 2]      ← per-prototype catuskoti bivector
Symbolic_in    [B, 8, 2]
Symbolic_out   [B, 8, 2]      ← per-prototype catuskoti bivector
Output         [B, 1, 1]      (XOR prediction head)
```

`P_event` is the perceptual representation that survives end-to-end
across stages; `S_event_{t-1}` is the previous iteration's symbolic
output. At sentence start `S_event_0 = 0`, and the PiLayer's
identity-on-zero property makes order-1 behaviour match the
no-loopback pipeline.

### Iteration without butterflies

`useButterflies=false`, `conceptualOrder=N` runs N iterations at the
same shape:

```
for t in 1..N:
    C_in_t = [P_event || S_out_{t-1}]      # right-half = previous symbol
    C_out_t = ConceptualSpace.forward(C_in_t)
    S_out_t = SymbolicSpace.forward(C_out_t)
```

P is constant context across iterations; S is the iterating channel.
The grammar XML drives whatever fold sequence runs at each stage; in
its simplest form (`MM_xor` with `<S>not</S>`, `<S>intersection</S>`,
`<S>union</S>`) the chart picks rule sequences and the per-stage
SymbolicSpace dispatches them.

### Bivector everywhere from C onward

Once C emits `[B, V_C, 2]`, every downstream tensor lives in
bivector-space until the OutputSpace projection. The per-slot D
collapses from `D_C = 6` (concept-shape) to `D = 2` (bivector). The
codebook geometry is the only place "deep" content lives -- on
`subspace.what` -- and is read indirectly via `Codebook.forward(x,
project=True)` (inner product against each row, accumulated into
pos/neg) and `Codebook.reverse(bivec, project=True)` (cached SVD
pseudo-inverse).

## Implementation outline

The work decomposes into seven gateable stages. Stages 1–3 trim the
codebase to the cognitive subset before the architectural changes
land, so the bivector + loopback work in stages 4–7 navigates fewer
codepaths.

### Stage 1 — Remove `useButterflies=true` codepath

The bivector regime can't satisfy the butterfly volume-equality
constraint (`nPercepts × state_dim == nSymbols × symbol_width` at
`ModelFactory.validate_config`) once the C-tier output narrows to
`D=2`. Butterflies are also load-bearing for none of the cognitive
properties this rollback targets — they're a compute-layout
optimization. Remove them.

Code touchpoints:

- `_create_per_stage` butterfly branch ([Models.py:4329-4364](basicmodel/bin/Models.py:4329)) —
  drop the entire `if self.useButterflies` block.
- `_level_shapes` callers and the `_butterfly_state_*` /
  `_butterfly_symbol_factor` instance fields.
- `ButterflyLayer` class in [Layers.py](basicmodel/bin/Layers.py)
  (`_butterfly_pack` / `_butterfly_unpack` / `_butterfly_merge` /
  `_butterfly_unmerge`).
- `PiLayer` / `SigmaLayer` constructor branches that switch on
  `stage_idx` / `n_t` / `is_last` for butterfly-mode pack/unpack.
- `validate_config`'s butterfly volume-equality requirement
  registration in [Models.py](basicmodel/bin/Models.py).
- All `<useButterflies>` reads. Drop the XML element from the
  schema.

XML migrations:

- `MM_xor.xml`, `MM_5M.xml`, `MM_400M.xml`, `MM_xor_step4.xml`, and
  any other config that sets `<useButterflies>true</useButterflies>`
  needs the flag removed and the model architecture re-validated.
  `MM_xor` will need a non-butterfly substitute (e.g.,
  `conceptualOrder=1` or non-butterfly `conceptualOrder>1`).

Acceptance:

- The butterfly branch in `_create_per_stage` is dead code (and
  removed). `useButterflies=true` is rejected at validation with a
  clear "no longer supported, see spec X" error.
- `test/test_basicmodel.py` and `test/test_mm_xor.py` stay green
  (after MM_xor's butterflies are flipped off and shapes are
  reconciled).

### Stage 2 — Remove `passThrough=true` codepath

`passThrough` is a "skip the layer's transform" shortcut used by
test configs and minimal probes. With the cognitive architecture
locked in, every Space does its work — there's no "skip" mode.

Code touchpoints:

- `Space.passThrough` short-circuit at the top of every `forward` /
  `reverse` (multiple call sites in [Spaces.py](basicmodel/bin/Spaces.py)).
- `Codebook.passThrough` short-circuit at [Spaces.py:1622](basicmodel/bin/Spaces.py:1622).
- The `passThrough=not self.codebook` derivation in
  `_build_what_basis` (per-Space). When passThrough goes away,
  `<codebook>false</codebook>` simply means "don't construct a
  Codebook; use a passthrough Tensor for `.what`" — explicit.
- `<passThrough>` reads in Space `__init__`s.

XML migrations:

- `idempotent.xml` uses `<passThrough>true</passThrough>` on
  PerceptualSpace; it'll need a real PerceptualSpace config or a
  rewrite using `<codebook>false</codebook>` to mean "trivial pass
  through with no learned operation".
- Any minimal-test config that sets `passThrough=true` on
  Symbolic / Output spaces needs migration.

Acceptance:

- `passThrough` reads gone from the codebase. `<passThrough>`
  removed from XML schema.
- Test suite stays green after migration.

### Stage 3 — Make subsymbolic-loopback always-on

After stages 1–2, the only remaining gating around the conceptual
loopback is `<subsymbolicEnabled>`. Drop the flag and make the
loopback unconditional. PerceptualSpace serves as the subsymbolic
substrate; no parallel `SubsymbolicSpace` instance is constructed.

Code touchpoints:

- `MentalModel._create_per_stage` ([Models.py:4427](basicmodel/bin/Models.py:4427)) and the flat path
  ([Models.py:1852](basicmodel/bin/Models.py:1852)): drop the conditional
  `if self.subsymbolicEnabled`. SubsymbolicSpace is no longer
  constructed; `self.subsymbolicSpace = None` always. Wire the
  `symbolicSpace_ref` on ConceptualSpace unconditionally.
- `mode == "grammar"` / `mode == "parallel"` branches ([Models.py:1860-1863](basicmodel/bin/Models.py:1860), [Models.py:4433-4436](basicmodel/bin/Models.py:4433)):
  delete. With no SubsymbolicSpace and only the symbolic loopback,
  `<mode>` is meaningless. Drop the flag from XMLs and the
  `architecture.mode` read.
- `ConceptualSpace.__init__` ([Spaces.py:7113](basicmodel/bin/Spaces.py:7113)): the
  `subsymbolic_widen_dim` parameter becomes derived (=
  `symbolShape[1]`), not flag-gated.
- `_build_combined_input` ([Spaces.py:7225](basicmodel/bin/Spaces.py:7225)): iterate only
  `symbolicSpace_ref` (drop the `subsymbolicSpace_ref` half of the
  loop).
- `SubsymbolicSpace` class itself: **not** removed. It stays in the
  codebase as a configurable variant (a regular Space with
  `<codebook>false</codebook>` and a limited grammar achieves the
  same role); just not auto-constructed by MentalModel.

Validator changes:

- Drop the "subsymbolicEnabled requires shared nDim across P / C / S
  / SubS" check.
- Replace with the unconditional contract `C.nInputDim ==
  P.nOutputDim + S.nOutputDim`.

XML migrations:

- Drop `<subsymbolicEnabled>` and `<mode>` lines from all configs
  that set them. Default behaviour is "loopback always on".

Acceptance:

- `subsymbolicEnabled` and `mode` reads gone from the codebase.
  XML schema drops the flags.
- All existing configs construct without `<mode>` or
  `<subsymbolicEnabled>`. C.nInputDim auto-validates against
  P.nOutputDim + S.nOutputDim.
- Test suite green.

### Stage 4 — Per-stage widening

After Stage 3 the loopback is always wired, but only stage 0's
ConceptualSpace is widened (legacy: `stage_widen_dim = ... if t == 0
else 0`). Under "every stage sees [P || S_{t-1}]", widen every
stage:

```python
stage_widen_dim = symbolShape[1]   # unconditional after Stage 3
```

Wire `symbolicSpace_ref` on every stage's ConceptualSpace. The
reference points at the SymbolicSpace whose *previous* iteration's
event we want to read; for shared-instance per-stage configs that's
`self.symbolicSpace`. For per-stage `symbolicSpaces[t]` arrays, each
stage references the previous: `t == 0` → zeros (cold start), `t >
0` → `symbolicSpaces[t-1]`.

Acceptance:

- A 2-stage MM_xor variant (no butterflies, `conceptualOrder=2`)
  forwards through both stages without shape errors. Each stage's
  C reads `[P || S_{t-1}]`.

### Stage 5 — ConceptualSpace Bivector Projection (CSBP)

Wire `Codebook.forward(input, project=True)` into the C-tier forward
and `Codebook.reverse(bivec, project=True)` into the reverse.

#### Forward swap

`ConceptualSpace.forward` ([Spaces.py:7335-7351](basicmodel/bin/Spaces.py:7335)): replace the legacy
VQ snap with the bivector projection. Today:

```python
if self.codebook:
    if (isinstance(self.subspace.what, Codebook)
            and self.nVectors > self.outputShape[0]):
        y = self.subspace.what.forward(y, topK=self.outputShape[0])
    else:
        y = self.subspace.get_vectors().forward(y)        # legacy snap
```

After CSBP:

```python
if self.codebook:
    y = self.subspace.what.forward(y, project=True)        # [B, V_S, 2] bivec
```

The wide-codebook `topK` branch goes away in this regime — the
bivector is per-prototype regardless of how many prototypes exist.

#### Reverse lift

`ConceptualSpace.reverse` ([Spaces.py:7374-7387](basicmodel/bin/Spaces.py:7374)): prepend a lift step
ahead of the existing `pi.reverse` / syntacticLayer.reverse dispatch.
Today:

```python
y = self.reverseBegin(vspace, returnVectors=True)
if getattr(self, 'syntacticLayer', None) is None:
    y = self._pi_reverse(y)
else:
    vspace.set_event(y)
    vspace = self.syntacticLayer.reverse(vspace)
    y = vspace.materialize()
```

After CSBP:

```python
y = self.reverseBegin(vspace, returnVectors=True)
if self.codebook:
    y = self.subspace.what.reverse(y, project=True)        # [B, V, D] lifted
# pi.reverse / syntacticLayer.reverse ...
```

#### SVD cache lifecycle

`Codebook.project` stores `self._project_cache` on the instance.
`Codebook.project_reverse` reads it. The contract is **one forward
followed by one reverse on the same codebook instance**, with no
intervening forward call that would overwrite the cache. Stage 3
makes this contract load-bearing; we should add a brief assertion in
`project_reverse` that the cache is non-None and emit a clear
diagnostic if not.

The cache is per-instance, so per-stage ConceptualSpaces each have
their own `_project_cache` -- no cross-stage interference.

#### Range check semantics

The bivec `[pos, neg]` from project=True has values that are sums of
relu projections, bounded by `V_in * max(|input| * |W|)`. They're
not in [-1, 1]. Two options:

1. **Normalize the bivec** at the C output. Divide by some scale (e.g.
   `sqrt(V_in)` or `V_in`) so the range is bounded. The downside is
   it's a scale-loss operation that changes the codebook gradient.
2. **Loosen the range check** for bivector-output spaces. Replace the
   `vspace.normalize("concepts", target="what")` call with a
   non-negativity range check (bivec values are in `[0, ∞)`).

Recommendation: option 2. Add a `kind="bivector"` to the normalize
contract with `lo=0, hi=None` so the check is non-negativity only.

### Stage 6 — Symbolic side mirror

SymbolicSpace's forward / reverse should symmetrically use the
project=True path so its output is also `[B, V_S, 2]`. Same swaps as
Stage 5 but in `SymbolicSpace.forward` and `SymbolicSpace.reverse`.
The legacy VQ snap branches (use_vqvae_reversible /
use_vqvae_nonreversible / hard_quantize) are vestigial under the
bivector regime; they stay in the codebase for now (legacy XMLs that
still want VQ-VAE training) but are not exercised by MM_xor under
this design.

### Stage 7 — Downstream consumer sweep

After C and S both emit bivectors, sweep downstream consumers for
the `[B, V, 2]` shape:

- `OutputSpace` projection from `[B, 8, 2]` to `[B, 1, 1]` — verify
  shape contract.
- `_compute_symbol_terms` / `_emit_symbol_terms` — symbol residual /
  commitment loss assumes deep content; needs to read the bivec
  format.
- Loss broadcasting — the `5 vs 4` shape error we hit earlier comes
  from one of these consumers; trace and fix.
- TruthLayer / Mereology readers — they consume conceptual
  activations; verify they read the bivec correctly.

## Cleanup notes (folded into stages 1–3)

The "no butterflies, no passThrough, always-on subsymbolic" trim is
distributed across stages 1–3 above so the architectural changes in
stages 4–7 navigate fewer codepaths. Two additional cleanup
candidates surface naturally during that work and should be
addressed as the corresponding stage lands:

- **VQ-VAE branches in `SymbolicSpace.forward`** (`use_vqvae_reversible`,
  `use_vqvae_nonreversible`, `hard_quantize`): under the bivector
  regime they're never reached. Stage 6 either deletes them or gates
  them behind a legacy-only flag. Decision deferred until then.
- **`_NON_MERGING_PATHS` for grammar**: keep, still load-bearing.

## Test plan

Each stage should leave the test suite green before the next stage
lands.

### Stage 1 acceptance — Remove butterflies

- The butterfly branch in `_create_per_stage` is gone. `ButterflyLayer`
  removed.
- Configs that previously set `<useButterflies>true</useButterflies>`
  are migrated or archived. `MM_xor.xml` runs with butterflies off
  (likely `conceptualOrder=1` until Stage 4 lands the multi-stage
  non-butterfly path properly).
- `test/test_basicmodel.py` and `test/test_mm_xor.py` stay green.

### Stage 2 acceptance — Remove passThrough

- `passThrough` reads gone. `<passThrough>` removed from XML schema.
- Configs that used it are migrated.
- Test suite green.

### Stage 3 acceptance — Always-on subsymbolic loopback

- `<subsymbolicEnabled>` and `<mode>` reads gone. SubsymbolicSpace is
  never auto-constructed by MentalModel.
- `_build_combined_input` always wires the symbolic right-half.
- Validator asserts `C.nInputDim == P.nOutputDim + S.nOutputDim`.
- Test suite green.

### Stage 4 acceptance — Per-stage widening

- A 2-stage MM_xor variant (`conceptualOrder=2`, no butterflies)
  forwards through both stages without shape errors. Each stage's
  C reads `[P || S_{t-1}]`.

### Stage 5 acceptance — CSBP

- ConceptualSpace.forward returns `[B, V, 2]` on a probe call.
- ConceptualSpace.reverse with the matching SVD cache returns
  `[B, V, D]`.
- Round-trip MSE on the C-tier alone (no SymbolicSpace involvement)
  is < 1e-3 for in-span inputs (analogous to test_idempotent_loop,
  but on the live ConceptualSpace from MM_xor).

### Stage 6 acceptance — Symbolic mirror

- SymbolicSpace.forward returns `[B, V_S, 2]` bivector.
- The C → S → C round-trip MSE is bounded by the codebook span
  losses on each side; documented threshold (likely ≤ 5e-2 with
  random init, decreasing with training).

### Stage 7 acceptance — Downstream consumer sweep

- `test_mm_xor.py::test_forward_reverse_reconstructs_input_state`
  passes under the bivector regime (threshold may need to relax
  from 1e-2; document the new threshold and the reasoning).
- `test_mm_xor.py::test_convergence` passes (training still
  converges to a useful XOR signal).
- The full `test/test_basicmodel.py` suite stays green.

## Risks and open questions

1. **Untrained-codebook amplification.** `project_reverse` uses
   `1/S` (inverse singular values). With a random codebook, small
   S blow up the lift. Pre-CSBP testing needs a strategy: either
   warmstart the codebook with an SVD-orthogonalized init, clamp
   the lift output, or accept that the round-trip MSE is high
   pre-training and the convergence test relaxes its threshold.

2. **per-prototype vs per-slot shape.** `[B, V_S, 2]` with
   `V_S != V_in` requires diagonal or outer-product
   disambiguation. Spec assumes `V_S == V_in` for now (true in
   MM_xor: 8 == 8). Configs with different V_S vs V_in are out of
   scope for this spec.

3. **Commitment-loss replacement.** The legacy VQ-VAE `commit_loss`
   pulls the encoder toward the codebook. Under CSBP the snap is
   the projection onto span(W), not a discrete substitution; the
   commit loss as written may not have the right gradient
   structure. May need to rederive or remove for the bivector
   regime.

4. **Discourse / TruthLayer interactions.** TruthLayer accumulates
   per-symbol activations. Under the bivector regime its
   `record_batch` sees `[B, V_S, 2]` instead of `[B, V_S, D_S]`;
   need to verify the trust-score computation still makes sense
   on bivector poles.

5. **VQ-VAE legacy configs.** MM_5M, MM_400M, etc. use the legacy
   VQ-VAE snap with `useVQVAE=true`. They're orthogonal to MM_xor
   under this spec but the cleanup pass removes the
   `useButterflies=true` path that some of them depend on. Need
   a migration plan or archival decision for those configs.

## Files to revert before this spec begins

To get to a clean baseline before the spec work starts, revert the
in-progress edits in MM_xor.xml back to a state that passes all
tests. Specifically:

- `<conceptualOrder>3</conceptualOrder>` (was 1)
- `<useButterflies>true</useButterflies>` (was false)
- Remove `<subsymbolicEnabled>` and `<mode>` lines.
- PerceptualSpace `nDim=10`, `nOutputDim=10` (was 4).
- ConceptualSpace `nInputDim=10`, `nDim=10`, `nOutputDim=10` (was 6
  / 6 / 2). `codebook=false` (was true).
- SymbolicSpace `nInputDim=10`, `nDim=10`, `nOutputDim=10` (was 2 /
  2 / 2).
- OutputSpace `nInputDim=10` (was 2).

After revert, run the full test suite to confirm green, then tag
the release. The spec work resumes in a separate conversation.

## References

- Rollback plan: [doc/plans/2026-05-07-finish-symbolic-rollback.md](basicmodel/doc/plans/2026-05-07-finish-symbolic-rollback.md)
- Codebook primitives: [bin/Spaces.py](basicmodel/bin/Spaces.py) `Codebook.project` (line 1436),
  `Codebook.project_reverse` (line 1500).
- Idempotent-loop test: [test/test_idempotent_loop.py](basicmodel/test/test_idempotent_loop.py).
- Combined-input wiring: [bin/Spaces.py](basicmodel/bin/Spaces.py) `_build_combined_input` (line 7225).
- Per-stage construction: [bin/Models.py](basicmodel/bin/Models.py) `_create_per_stage` (line 4060+).
- Subsymbolic plan (predecessor): [doc/plans/2026-05-05-subsymbolic-knowing-handoff.md](basicmodel/doc/plans/2026-05-05-subsymbolic-knowing-handoff.md).
