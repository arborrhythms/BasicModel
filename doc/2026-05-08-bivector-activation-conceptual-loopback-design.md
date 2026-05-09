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

The work decomposes into six gateable stages. Stages 1–2 trim the
codebase to a smaller subset before the architectural changes land,
so the bivector + loopback work in stages 3–6 navigates fewer
codepaths. Butterflies are deliberately *not* removed in this spec
iteration (see Note A); the bivector configs simply set
`useButterflies=false`. The butterfly path stays available for
legacy configs (MM_5M, MM_400M, etc.) and gets retired in a
follow-up.

### Stage 1 — Remove `passThrough=true` codepath

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

### Stage 2 — Make subsymbolic-loopback always-on

After stage 1, the only remaining gating around the conceptual
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

### Stage 3 — Per-stage widening

After Stage 2 the loopback is always wired, but only stage 0's
ConceptualSpace is widened (legacy: `stage_widen_dim = ... if t == 0
else 0`). Under "every stage sees [P || S_{t-1}]", widen every
stage:

```python
stage_widen_dim = symbolShape[1]   # unconditional after Stage 2
```

Wire `symbolicSpace_ref` on every stage's ConceptualSpace. Sketch:

```python
# In MentalModel._create_per_stage, after the spaces are constructed:
for t, cs in enumerate(self.conceptualSpaces):
    if t == 0:
        # Stage 0 cold-starts: no previous symbol, _build_combined_input
        # zero-fills the right half. Set the ref to the same-stage
        # SymbolicSpace -- on the first call its event is None / pre-
        # forward, which _build_combined_input treats as zero anyway.
        ref = self.symbolicSpaces[0]
    else:
        # Subsequent stages read the previous stage's symbolic event.
        ref = self.symbolicSpaces[t - 1]
    object.__setattr__(cs, 'symbolicSpace_ref', ref)
    object.__setattr__(cs, 'subsymbolicSpace_ref', None)
```

`object.__setattr__` is required to bypass `nn.Module` submodule
registration -- otherwise the cross-reference would put SymbolicSpace
into multiple parents in the module tree (breaks `.to(device)`,
double-counts parameters).

Acceptance:

- A 2-stage MM_xor variant (no butterflies, `conceptualOrder=2`)
  forwards through both stages without shape errors. Each stage's
  C reads `[P || S_{t-1}]`.

### Stage 4 — ConceptualSpace Bivector Projection (CSBP)

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
intervening forward call that would overwrite the cache. Stage 4
makes this contract load-bearing; we should add a brief assertion in
`project_reverse` that the cache is non-None and emit a clear
diagnostic if not.

The cache is per-instance, so per-stage ConceptualSpaces each have
their own `_project_cache` -- no cross-stage interference.

#### Codebook initialization (resolves Risk 1)

`project_reverse` uses `1/S` (inverse singular values of W). With a
random codebook, small S blow up the lift. The chosen mitigation is
**SVD-orthogonalize the codebook at construction**: in
`Codebook.addVectors` (or its caller for ConceptualSpace), after the
random init, run an SVD on `W` and replace it with `U @ V.T` (the
nearest orthonormal matrix to the random init). This pegs all
singular values at 1, so `1/S` is identity-ish and the lift is well-
conditioned from the first forward call. EMA / training updates then
deform the codebook from this orthonormal start; values stay
bounded as long as training itself is stable.

Implementation:

```python
# After the random init in Codebook.addVectors / vq.codebook init:
with torch.no_grad():
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W.copy_(U @ Vh)   # nearest orthonormal -- all singular values 1
self._svd_dirty = True
```

This is gated on a flag (e.g., `<svdOrthogonalInit>true</svdOrthogonalInit>`
on ConceptualSpace) so legacy configs aren't disturbed.

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

### Stage 5 — Symbolic side mirror

SymbolicSpace's forward / reverse should symmetrically use the
project=True path so its output is also `[B, V_S, 2]`. Same swaps as
Stage 4 but in `SymbolicSpace.forward` and `SymbolicSpace.reverse`.
The legacy VQ snap branches (use_vqvae_reversible /
use_vqvae_nonreversible / hard_quantize) are vestigial under the
bivector regime; they stay in the codebase for now (legacy XMLs that
still want VQ-VAE training) but are not exercised by MM_xor under
this design.

### Stage 6 — Downstream consumer sweep

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

### Stage 1 acceptance — Remove passThrough

- `passThrough` reads gone. `<passThrough>` removed from XML schema.
- Configs that used it are migrated.
- Test suite green.

### Stage 2 acceptance — Always-on subsymbolic loopback

- `<subsymbolicEnabled>` and `<mode>` reads gone. SubsymbolicSpace is
  never auto-constructed by MentalModel.
- `_build_combined_input` always wires the symbolic right-half.
- Validator asserts `C.nInputDim == P.nOutputDim + S.nOutputDim`.
- Test suite green.

### Stage 3 acceptance — Per-stage widening

- A 2-stage MM_xor variant (`conceptualOrder=2`, no butterflies)
  forwards through both stages without shape errors. Each stage's
  C reads `[P || S_{t-1}]`.

### Stage 4 acceptance — CSBP

- ConceptualSpace.forward returns `[B, V, 2]` on a probe call.
- ConceptualSpace.reverse with the matching SVD cache returns
  `[B, V, D]`.
- SVD-orthogonal init keeps the round-trip MSE on the C-tier alone
  (no SymbolicSpace involvement) under 1e-3 for in-span inputs from
  the very first forward call (no training warm-up needed).

### Stage 5 acceptance — Symbolic mirror

- SymbolicSpace.forward returns `[B, V_S, 2]` bivector.
- The C → S → C round-trip MSE is bounded by the codebook span
  losses on each side; documented threshold (≤ 5e-2 from a fresh
  SVD-orthogonal init, decreasing with training).

### Stage 6 acceptance — Downstream consumer sweep

- `test_mm_xor.py::test_forward_reverse_reconstructs_input_state`
  passes under the bivector regime (threshold may need to relax
  from 1e-2; document the new threshold and the reasoning).
- `test_mm_xor.py::test_convergence` passes (training still
  converges to a useful XOR signal).
- The full `test/test_basicmodel.py` suite stays green.

## Risks and open questions

1. **Untrained-codebook amplification → SVD-orthogonal init.**
   `project_reverse` uses `1/S` (inverse singular values). With a
   random codebook, small S blow up the lift. **Resolved**: Stage 4
   SVD-orthogonalizes the codebook at construction
   (`W ← U @ V.T` so all singular values = 1). Gated on a flag
   (`<svdOrthogonalInit>true</svdOrthogonalInit>` on
   ConceptualSpace) so legacy configs are unaffected.

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
   regime. Decision deferred to Stage 5.

4. **Discourse / TruthLayer interactions.** TruthLayer accumulates
   per-symbol activations. Under the bivector regime its
   `record_batch` sees `[B, V_S, 2]` instead of `[B, V_S, D_S]`;
   need to verify the trust-score computation still makes sense
   on bivector poles. Surfaces in Stage 6 (downstream sweep).

## Note A — Butterflies stay (deferred)

This spec deliberately does *not* remove the `useButterflies=true`
codepath. Original draft had a Stage 1 that removed butterflies
because the bivector regime can't satisfy the volume-equality check
(`nPercepts × state_dim == nSymbols × symbol_width`). That's still
true, but rather than trim now, we leave butterflies in place for
legacy configs (MM_5M, MM_400M, MM_xor's pre-bivector flavor) and
simply set `<useButterflies>false</useButterflies>` on the bivector
configs. Butterfly removal is a follow-up spec iteration once the
bivector path is stable.

Practical implication: don't touch `_create_per_stage`'s butterfly
branch, `ButterflyLayer`, or the `validate_config` volume-equality
check. They keep working for `useButterflies=true` configs;
bivector configs avoid them by setting the flag false.

## Current state at handoff

The codebase already has the following changes from the rollback
session that are *not* in `git log` of the next conversation's
starting point but *are* in the working tree (the next
implementer needs to know they exist before treating them as
out-of-scope):

- **`Codebook.STE=False` flag** added to `Codebook.create`. When
  True, `forward` and `reverse` wrap the snap with
  `input + (snapped - input).detach()` so the encoder gets gradient
  identity through the snap. `SymbolicSpace._build_what_basis`
  passes `STE=True`. CSBP (Stage 4) replaces the snap with the
  bivector projection, so the STE wrap will need to be either
  retained for the bivector-output case or discarded; that decision
  is part of Stage 4's work.
- **`WhereEncoding._codebook_registry`** + `allocate_codebook_slice`
  + `Codebook.where_offset` + `addVectors` size assert — sequential
  per-codebook offset allocation in the global where-space. Used
  for the planned codebook-key story (`.where` as a global lookup
  index) but not yet wired into the lookup path.
- **`Models._create_per_stage` `conceptOutputShape`** uses explicit
  `<nOutput>` / `<nOutputDim>` instead of the broken volume formula.
  This is what made `is_empty()` stop short-circuiting on
  `inputShape[0]==0`. Don't revert.
- **`XMLConfig.space()`** raises `ValueError` on duplicate scalar
  config tags (e.g., two `<nVectors>` in one space).
- **`bin/bm.py`** is defensive about missing `numShards` /
  `maxDocs` / `shardDir` in `<data>` blocks. `Embedding.predict(None)`
  returns `[]`. `Models.infer` treats empty decoded as
  end-of-stream.
- **`TheData.loadInline`** synthesizes deterministic random
  sentences when `<dataset>inline</dataset>` has no `<input>` /
  `<output>` children. Used by `idempotent.xml`.
- **`test/test_partition_resolve.py`** assertions migrated to
  `nWhat == sym.nDim`.
- **`test/test_mental_model.py::test_configured_grammar_matches_xml`**
  assertion migrated to compare against the S-tier rule subset
  (`grammar.symbolic()`) since MentalModel.xml now has explicit
  P / C tier rules from Phase 4 Step 6.

### MM_xor.xml status

`MM_xor.xml` has been reverted to the pre-bivector-experiment
shape (`conceptualOrder=3`, `useButterflies=true`, all spaces at
`nDim=10` / `nOutputDim=10`, no `<subsymbolicEnabled>`, no
`<mode>`, ConceptualSpace `<codebook>false</codebook>`,
SymbolicSpace `<codebook>true</codebook>` `<useVQVAE>true</useVQVAE>`).
The grammar block has the explicit per-tier folds added in Phase 4
Step 6. Test suite is green at this baseline.

### idempotent.xml status

`idempotent.xml` is in the `passThrough=true` / `nDim=100` regime
from the discussion. It still uses `<passThrough>true</passThrough>`
on PerceptualSpace, which Stage 1 of this spec removes. Migration
required (rewrite as `<codebook>false</codebook>` perceptual or
delete the file).

## References

- Rollback plan: [doc/plans/2026-05-07-finish-symbolic-rollback.md](basicmodel/doc/plans/2026-05-07-finish-symbolic-rollback.md)
- Codebook primitives: [bin/Spaces.py](basicmodel/bin/Spaces.py) `Codebook.project` (line 1436),
  `Codebook.project_reverse` (line 1500).
- Idempotent-loop test: [test/test_idempotent_loop.py](basicmodel/test/test_idempotent_loop.py).
- Combined-input wiring: [bin/Spaces.py](basicmodel/bin/Spaces.py) `_build_combined_input` (line 7225).
- Per-stage construction: [bin/Models.py](basicmodel/bin/Models.py) `_create_per_stage` (line 4060+).
- Subsymbolic plan (predecessor): [doc/plans/2026-05-05-subsymbolic-knowing-handoff.md](basicmodel/doc/plans/2026-05-05-subsymbolic-knowing-handoff.md).
