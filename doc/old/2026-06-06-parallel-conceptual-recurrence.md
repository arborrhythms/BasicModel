# Parallel Conceptual Recurrence: Invertible 3-Stream Combine + Reconstruction Cleanup

Date: 2026-06-06
Status: Design (approved for implementation)
Supersedes the parallel-mode portion of `doc/plans/2026-06-05-dimensional-governance.md`
(the invertible $\pi$/$\sigma$ bridge substance lands here).

## 1. Motivation

The PARALLEL conceptual recurrence (`_forward_body`, the $T = \mathrm{conceptualOrder}$
stage loop) currently advances the conceptual carrier by a STRICT alternation: a
$\pi$ lift of the perceptual contribution at stage $0$, then a square $\sigma$
advance at every stage after. In the $\alpha$-notation the user proposed, that is
exactly $\alpha_{ps} = [1, 0, 0, \dots]$, $\alpha_{ss} = [0, 1, 1, \dots]$,
$\alpha_{cs} = 0$. The alternation is forced by one fact:

> An ADDITIVE combine of two streams has no closed-form inverse. In
> $CS_{t+1} = \pi(\alpha_{ps}\,PS_t) + \sigma(\alpha_{ss}\,SS_t)$ both terms are
> functions of the same $CS_{t-1}$, so the sum entangles them; one-shot reversal
> requires exactly one live term per stage.

This design lifts that restriction. A single SQUARE `InvertibleLinearLayer`
($3D \to 3D$, `naive=False`) over $[PS_t; SS_t; CS_t]$ mixes all three streams
EVERY stage (a strictly richer forward than the alternation) and stays EXACTLY
invertible by threading a $2D$ augment from the forward into the reverse. This
resolves the user's original tension --- "a richer forward transformation that
preserves reconstruction" --- in one operator.

Two cleanups ride along because they share the same code surface:

- The recompile that blocks `fullgraph=True` (priority 1) is caused by
  per-batch DATA stored as a Layer attribute (`ShortTermMemory._fallback_buffer`),
  whose `requires_grad` oscillates across forwards. The fix is the load-bearing
  principle below, not a local patch.
- `<reconstruct>` is always-from-concepts in every live config; the enum and the
  historical recon-symbols paths are dead weight.

## 2. Load-bearing principles

1. **Spaces hold WEIGHTS only** --- `nn.Parameter`s, updated at the optimize step
   AFTER the sentence completes. Weights are frozen within a forward.
2. **Per-batch DATA threads through the forward** as tensors and is NEVER persisted
   as accumulated state on a space or a Layer. No `_fallback_buffer`, no stashed
   augment, no cross-forward attribute writes.

Principle 2 alone is the `fullgraph` fix: nothing with an oscillating
`requires_grad` survives across forwards, so the Dynamo guard never flips and the
compiled step compiles once.

## 3. The core combine

### 3.1 Operator

Per stage $t$, over the per-position content streams
$PS_t, SS_t, CS_t \in \mathbb{R}^{D}$:

$$[\,next\_CS_t \;\Vert\; aug_t\,] \;=\; \mathrm{ILL}_t\big([\,PS_t \;\Vert\; SS_t \;\Vert\; CS_t\,]\big),
\qquad \mathrm{ILL}_t : \mathbb{R}^{3D} \to \mathbb{R}^{3D}$$

with $next\_CS_t \in \mathbb{R}^{D}$ the conceptual carrier passed to stage $t+1$
and $aug_t \in \mathbb{R}^{2D}$ the augment. `ILL` is the existing
`InvertibleLinearLayer` (`bin/Layers.py:974`) built `naive=False` and `square`.
The scalar $\alpha_{ps}, \alpha_{ss}, \alpha_{cs}$ are ABSORBED into the learned
LDU weights --- there are no separate $\alpha$ knobs; the layer learns the
effective per-channel mixing, which generalises any scalar setting.

### 3.2 Reverse and the two reconstruction regimes

`InvertibleLinearLayer.reverse` with `naive=False` routes through `_solve_ldu`
(`bin/Layers.py:1181`): `solve_triangular(U)` $\to$ reciprocal on the rank-$D$
diagonal $\to$ structured zero-pad of dropped dims in the $L$-basis $\to$
`solve_triangular(L)`. This is the exact structured inverse (same math as the
materialised $U^{-1} D^{+} L^{-1}$ but $O(b\,n^2)$, no $O(n^3)$ build, far less FP
drift). It is strictly better than a generic least-squares `pinv`; do NOT use the
`naive=True` path.

- **`<perfectReconstruction>true`**: thread $aug_t$ (per stage) forward into
  `_reverse_body`. The reverse inverts the SQUARE map exactly:
  $[\,PS_t \Vert SS_t \Vert CS_t\,] = \mathrm{ILL}_t^{-1}([\,next\_CS_t \Vert aug_t\,])$,
  a closed-form round-trip exact to the LDU solve tolerance.
- **`<perfectReconstruction>false`**: drop $aug_t$. The reverse reconstructs the
  preimage with the augment treated as zero (the structured zero-pad), which is
  EXACT on the rank-$D$ subspace that survived the forward. Which $D$-subspace
  survives is learned through $L$, so training keeps the reconstructable content
  alive rather than least-squares-projecting it; $L_{recon}$ absorbs the residual.

So the SAME square layer serves both regimes; the boolean only gates whether the
$2D$ augment is threaded (exact) or zeroed at reverse (approximate).

### 3.3 Stage wiring (replaces the current alternation)

```
CS_{-1}      = seed (sec. 5)
PS_0         = perceptualSpace.forward(reduced_percepts, CS_{-1})   # sec. 6
for t in 0 .. T-1:
    SS_t           = ss[t].forward(CS_{t-1})
    next, aug_t    = ILL_t([PS_t ; SS_t ; CS_t])
    CS_{t+1}       = next
    PS_{t+1}       = 0          # alpha_ps live only at t = 0 (read input once)
augments = [aug_0 .. aug_{T-1}]    # threaded local; consumed by _reverse_body
```

`PS_t` is non-zero only at $t = 0$ (input read once); `SS` and `CS` mix at every
stage through the learned ILL. The reverse walks $CS_T \to CS_0$ using
`augments[t]` (perfect) or zero (approximate), then `ILL_t.reverse`.

## 4. Data-flow / threading (the fullgraph fix)

- `ShortTermMemory._fallback_buffer` and the per-stage stashed slabs are REMOVED.
  The slab and `augments` are LOCALS threaded through the $T$ stages and into
  `_reverse_body` within the SAME forward call.
- `forward` gains a `prev_cs` parameter and RETURNS the terminal CS state. The
  caller (`runBatch`) holds the returned CS across forwards as a local and passes
  `prev_cs = returned_cs.detach()` into the next forward. The detach makes each
  forward independent (truncated BPTT across sentences --- standard for recurrent
  state).
- `L_intra`'s "previous slab" is `prev_cs` (the prior forward's returned CS,
  threaded through the caller), NOT a stored buffer. The intra-sentence predictor
  (`IntraSentenceLayer`, `bin/Layers.py:6869`) reads `prev_cs`, predicts the stage
  slab, and accumulates $L_{intra} = \mathrm{MSE}(\mathrm{pred}, \mathrm{perceived})$.

## 5. Inter-sentence seed

New knob `<prediction>interSentence</prediction>` (architecture-level). When on,
the stage-$0$ seed is $CS_{-1} = \mathrm{InterSentenceLayer.predict}(\mathit{lastSentence})$.
`InterSentenceLayer` (ARMA$(p,q)$ over sentence reps, `bin/Layers.py`) and
`discourse.predict()` already exist; this wires `predict()` output as the seed.
`lastSentence` is passed in by the caller (data, not stored). Default off $\to$
$CS_{-1}$ is the empty seed (current behaviour).

## 6. PerceptualSpace does not read raw IS (parallel)

Verified: in parallel mode `perceptualSpace.forward` is called exactly ONCE
(stage 0) and the per-stage loop never re-invokes it; its input `in_sub` is the
POST-CHUNK reduced percepts (the word vectors), not the 8192-char IS buffer. This
design preserves that and pins it with a test (sec. 8). PS reads the reduced
percepts once, with the $CS_{-1}$ seed as the C$\to$P feedback arg.

## 7. Reconstruction cleanup

- Delete `reconstructEnum` from `data/model.xsd` and the `<reconstruct>` element;
  delete the `none|symbols|both` branches in `bin/Models.py`. Reconstruction is
  unconditionally from concepts.
- Remove recon-symbols / historical-reconstruction references (the
  "reserved for reconstruction" SymbolicSpace-slot machinery and any
  `reconstruct == "symbols"/"both"` dispatch).
- Drop the `<reconstruct>concepts</reconstruct>` line from the 2 configs that set
  it (the only live setters; both already `concepts`).
- Add `<perfectReconstruction>` (`xs:boolean`, default `false`) to the schema and
  `architecture` parsing.

## 8. Testing

- **Invertibility (perfect):** with `<perfectReconstruction>true`,
  `reverse(forward(x))` $\approx x$ to LDU tolerance on MM_5M-shaped slabs; assert
  per-stage `ILL_t.reverse(ILL_t(z)) \approx z` (square round-trip).
- **Reconstruction (approximate):** with `false`, the concepts round-trip is
  exact on the surviving subspace; $L_{recon}$ is finite and decreases over a short
  train.
- **fullgraph compiles once:** under `BASICMODEL_MPS_COMPILE=1` (and CPU
  `aot_eager`), `TORCH_LOGS=recompiles` shows NO `_fallback_buffer`
  `requires_grad` guard failure across $\ge 3$ batches.
- **PS-not-IS:** the parallel forward invokes `perceptualSpace.forward` exactly
  once and never reads the raw IS buffer after stage 0.
- **Gate:** `python bin/Models.py data/MM_5M.xml` runs clean with finite
  `output` and `reconstruction` loss.
- **Regression:** `test_modality_configs`, `test_invertibility`,
  `test_pi_sigma_ownership`, `test_cs_reentrancy`, `test_dimensional_governance`,
  `test_role_collapsed_grammar` stay green.

## 9. Code surface

- `bin/Models.py`: `_forward_body` (5079, parallel branch), `_symbolic_sigma_step`
  (6351), `_forward_per_stage` (6661) signature (`prev_cs` in / CS out),
  `_reverse_body` (augment consumption), reconstruct-branch removal, `runBatch`
  caller threading.
- `bin/Spaces.py`: `_stm_predict_then_perceive_parallel` (11001),
  `_stm_set_all_slots` (10676) --- replace stored-buffer reads/writes with threaded
  tensors; the conceptual combine construction (new `ILL_t` per stage).
- `bin/Layers.py`: reuse `InvertibleLinearLayer` (974) `naive=False`; remove
  `ShortTermMemory._fallback_buffer` (9380) and the proxy fallback.
- `data/model.xsd`: drop `reconstructEnum`/`<reconstruct>`, add
  `<perfectReconstruction>` and `<prediction>`.
- 2 configs: drop `<reconstruct>concepts</reconstruct>`.

## 10. Risks / open decisions

- **Param budget (the one real fork).** A DENSE square LDU over $3D$ costs
  $\approx 2(3D)^2$ params per stage. For $D = 1024$ that is $\approx 18.9$M
  per stage, $\approx 57$M over $T = 3$ --- dwarfing the post-butterfly-fix
  $\sim 21$M model. Options: (a) BUTTERFLY-structure the $3{:}1$ combine
  ($O(N \log N)$, $\sim 0.3$M/stage, matching the existing bridge budget) while
  keeping the augment-threading invertibility; (b) run the combine on a narrower
  content width than the full event; (c) accept the cost. The user specified
  "InvertibleLinearLayer (LDU)"; this needs an explicit ruling on spec review.
- **Compiled-step boundary.** Adding `prev_cs` to the forward signature and
  threading `augments` changes the captured-forward inputs; confirm the staged
  (compiled) path still traces `fullgraph=True`.
- **interSentence interaction.** Seeding $CS_{-1}$ from `predict()` overlaps the
  existing `discourse.predict()` priming in `_forward_per_stage`; unify so there is
  one seed source.
- **Serial mode untouched.** This redesign is the PARALLEL branch only; the serial
  (`_forward_body_per_word`) STM accumulation keeps its semantics. The
  `_fallback_buffer` removal must preserve serial behaviour via threaded state too.
