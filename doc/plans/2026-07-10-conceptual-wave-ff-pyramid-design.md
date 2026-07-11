# Symmetric Dual Towers + Callosum-Fed Concept Pyramid — Design (APPROVED, rev 2)

> **STATUS: APPROVED 2026-07-10 (Alec, brainstorm session; rev 2 same day).**
> Un-parks the two items the reconstruction-fidelity design deferred — wave
> brightness / training dynamics AND Task 11 nVectors wiring. **Rev 2
> supersedes rev 1 in place** (rev 1 planned a geometric `<nVectors>` taper +
> $K$ untied folds; the brainstorm continuation corrected the frame — see
> "Correction" below). Predecessor:
> [2026-07-03-reconstruction-fidelity-design.md](2026-07-03-reconstruction-fidelity-design.md)
> (Scope "Parked", line 36). Root findings:
> [2026-07-03-iterated-symbolic-loop-execution.md](2026-07-03-iterated-symbolic-loop-execution.md)
> EXECUTION NOTES items 12-13. Execution plan:
> [2026-07-10-conceptual-wave-ff-pyramid-execution.md](2026-07-10-conceptual-wave-ff-pyramid-execution.md).

## Why (fresh probe evidence, 2026-07-10)

The conceptual wave — `cs_forward_content` (bin/Spaces.py $\approx$ 14698), an
iterated clamped-SOURCE settling recurrence $a^{i+1}=\tanh(W[a^i \mid 1]+s)$ —
runs DARK on `MM_sparse_concept` (sO=3 parallel). A current-code probe
(cpu/eager, seed 0, 2 train epochs) reproduced the parked findings:

1. **Sign-then-clamp annihilation.** The order-0 snap readout rectifier
   (`nonneg=True`, bin/Spaces.py:14612) meets a training-driven mean-negative
   readout: epoch 0 pre-clamp $\sim\pm$1e-3 (50% negative, 8/16 survive);
   epoch 1 fully negative (mean $-$6.4e-3, 0/16 survive) $\to a_0=0 \to$
   `_cs_wave_qe` $=[0,0,0]$.
2. **Scale-blindness.** Readout magnitude $\sim$1e-3: the settled field and the
   order-0 codebook are near-orthogonal (P4 snap-blindness carried forward).
3. **Capacity gap (Task 11).** `<ConceptualSpace><nVectors>`$=32$ never reaches
   the stage store (`stage_space_concept = [cs_out[0], ...]`,
   bin/Models.py:5989): every stage runs $N=8$, caps (snap 4, pool 4); the
   pool overflows from batch 1.

## Correction (rev 2): the darkness was the WRONG INPUT PATH

Rev 1 read the darkness as a settling-dynamics pathology and planned a
geometric taper of `<nVectors>`$=32$ with $K$ untied folds. The brainstorm
continuation corrected the frame:

- `<nVectors>` is the codebook PROTOTYPE INVENTORY (bin/Spaces.py:3496
  "covers nVectors prototypes"; the top-K forward at 3794 exists for
  `nVectors >> nOutput`). It is NOT the concept base.
- The concept base is the **8 tiles** (`<nOutput>`$=8$) that tile the input.
  PS and WS each present an 8-slot view; the corpus callosum
  (`ConceptualCombine`, bin/Layers.py:1455 — the learned $[2N,N]$ glue,
  production $16\times 8$) stacks PS(8) $\oplus$ WS(8) and glues them to the
  shared 8-tile frame.
- Attention is defined ONLY over that shared 8-frame — WholeSpace has no
  attention over its own space — so the concept pyramid is a top-K selection
  over the frame, not a taper of the inventory.
- The order-0 signal must come FROM THE CALLOSUM, not from the
  `cs_snap_order0` snap readout over the 32-row inventory. The rectifier
  annihilation (symptom 1) and scale-blindness (symptom 2) are symptoms of
  reading the wrong path; with the callosum feed they are moot. Symptom 3
  dissolves likewise: the store sizes off the tiles + top-K pool, and
  `<nVectors>` stays inventory.

## Theory

PS and WS are **symmetric spaces, duals of one another**: one receives the
input ATOMS (PartSpace — bottom-up $\sigma$-synthesis: parts chunk into wholes),
the other the input UNIVERSE (WholeSpace — top-down $\pi$-analysis: wholes split
into parts). Both represent THE SAME THING — the input — from dual
orientations, so they present identical $[8,1024]$ views, literally stacked at
the corpus callosum and glued to one 8-tile frame. InputSpace already produces
both views at the source (`in_sub, concepts_in = inputSpace.forward(x)`,
bin/Models.py:3736): the atom stem and the unity/universe view.

The conceptual pyramid is feedforward over that frame: edge-structured
$\sigma$ composition (the constituent edges — explicit ramsification) with a
top-K halving $8\to 4\to 2\to 1$ per ramsified order as the attention. There
is no settling recurrence, hence no trivial fixed point and no darkness
mechanism. Ramsification lives in the constituent DAG (a symbol's order $=$
$1+\max$ constituent order), which the untied feedforward composition
preserves. The cycles/groundedness reading (the source-release half of
`cs_groundedness_probe`) is sacrificed — accepted (the wave was dark anyway).

## Decisions (Alec, 2026-07-10; rev-2 continuation)

1. **Replace, do not tune** (carried from rev 1): the settling wave becomes a
   feedforward composition; no fixed-point dynamics.
2. **PS/WS are symmetric duals** with the SAME signature —
   `partSpace.forward(in_sub, CS_out_PS)` /
   `wholeSpace.forward(in_sub, CS_out_WS)` — where `in_sub` is each tower's
   view of the input (atoms / universe) and `CS_out` its own conceptual
   feedback. InputSpace reaches BOTH towers.
3. **`PerceptualSpace` (the shared derived class) is removed** — both spaces
   subclass `Space` directly. The base is thin (no params, no submodules;
   state_dict unaffected): relocate `NULL_PERCEPT_KEY`, update isinstance
   sites.
4. **WS transposes to $[8,1024]$** matching PS — the $[1024,8]$ wide-symbol
   shape was a leftover from when WS was going to process symbols. (Config
   edit DONE 2026-07-10 in data/MM_sparse_concept.xml.)
5. **Order-0 $=$ the callosum-glued frame.** `cs_snap_order0` and its
   `nonneg` rectifier retire on this path.
6. **The pyramid halving is top-K via `.index`**: set the subspace `.index`
   (renamed from `.active`) so a GENERIC `materialize()` pulls exactly the
   selected codes — no `mode=`/"property" parameter at these call sites.
   $8\to 4\to 2\to 1$ over the shared frame.
7. **`<nVectors>` stays the inventory.** Store sizing follows the 8 tiles +
   the top-K pool ($4+2+1$), not a geometric taper. (Task 11 closes by
   re-derivation, not by wiring 32 into the store.)
8. **Off-path discipline:** the recon round-trip pins (xor, grammar) are
   approved bars — no silent re-baseline. The shared pump loop serves every
   config, so WS's routing change engages on the PARALLEL path first;
   migrating serial routing is an explicit CHECKPOINT (Alec decides, with an
   honest re-baseline if taken).

## Scope

**In:** the PS/WS symmetry refactor (class removal, dual `forward(in_sub,
CS_out)` signatures, IS-to-both wiring); WS transpose (done); callosum-fed
order-0; the top-K `.index` pyramid over the frame with edge-structured
$\sigma$ composition; per-level observability; tests.

**Out:** serial-path routing migration (checkpoint only); wave-usefulness /
task-metric improvements (XOR plateau) — mechanism-correctness is this
phase's bar; `<nVectors>` inventory changes; OutputSpace / head changes.

## Design

### 1. Symmetric towers

Both spaces subclass `Space`; `PerceptualSpace` deleted. One signature:

```python
PS_sub = partSpace.forward(in_sub_atoms,   CS_out_PS)   # sigma: chunk up
WS_sub = wholeSpace.forward(is_universe,   CS_out_WS)   # pi:    split down
```

`CS_out_PS` / `CS_out_WS` are the per-tower conceptual feedbacks (today's
C$\to$P demux feedback and `prevCS_forSS` respectively). Migration is
behavior-preserving off-path: the new signature carries the SAME tensors the
legacy routing consumed unless the parallel-path predicate is active
(decision 8).

### 2. Callosum-fed order-0

At the P3 cutover the symbolic phase reads the glued frame — the
`ConceptualCombine` stack PS(8)$\oplus$WS(8) $\to [2N,N]$ glue $\to$ 8 —
instead of snapping the settled field against the order-0 codebook block.
`cs_snap_order0` + the `nonneg` rectifier retire on this path (EMA identity
trace: keep only if a consumer still needs it — verify at implementation).

### 3. The pyramid (composition + top-K attention)

Per ramsified order $k$: the edge-structured $\sigma$ fold composes order-$k$
candidates from order-$(k{-}1)$ codes along the constituent edges; the
attention sets the subspace `.index` to the top-K ($8\to 4\to 2\to 1$) and
generic `materialize()` pulls exactly those codes. Stacked per-order
activations feed the SS leg / losses through the EXISTING interfaces
(`_concept_activations` $[N,B]$ grad-bearing; `_cs_parallel_slab`
$[B,N,\mathrm{CDim}]$) — the cutover block, SS leg, and SBOW loss keep their
shapes. ($\sigma$-composition mechanics vs pure `.index` selection: confirm
the exact split at the Task-C checkpoint before implementing.)

### 4. Capacity (Task 11 re-derived)

Store $=$ 8 tile rows (order 0, no in-edges) $+$ top-K pool rows
($4+2+1=7$). `<nVectors>`$=32$ remains the prototype inventory the top-K
draws from. Per-level overflow stays loud.

### 5. Observability

Per order, report-only: selected `.index` occupancy, activation magnitude,
per-fold gradient norms. These are the mechanism-live pins (replacing
`_cs_wave_qe`).

## Follow-on decisions (Alec, 2026-07-11)

1. **Serial routing migrates to the TYPED + LIVENESS law** (outcome of
   "update serial; remove legacy if it works"): the unity is offered at
   every pump with no mode gates; a LIVE unity routes universe-primary, a
   DEAD unity with a live carrier routes the carrier body (the recurrent
   leg + machinery). No shipped config has a live universe analysis yet,
   so behavior is unchanged today — the universe path lights up
   automatically when decision 2's `<analysis>word</analysis>` lands.
   Full carrier-path removal is bounded by that. Trade + the WS-leg
   ablation finding recorded in the execution notes (items 22-24).
2. **`<analysis>word</analysis>` typed runs into the glued frame are
   sanctioned as a USEFUL INTERIM:** until serial mode can work on its own
   AND a better attention mechanism lets the mind itself determine the
   basic level, the word cut supplies that level externally — words are a
   good first test of that level-finding. When an attention mechanism can
   discover the basic level, the externally-imposed word cut retires.
3. **Groundedness probe stays retired**; the bottom-up-attention AUDIT
   (2026-07-11) confirms the pyramid replaced only the wave — every other
   attention mechanism (ReadingAttention, GlobalAttention, intent
   priming / `_topk_priming_mask`, reverse-side heat + `<attention>`
   modes) is present, orthogonal, and composes; see the execution notes.
4. **Relevance projections SPEC (Alec: "spec and implement in this
   round", 2026-07-11).** Frame: relevance $=$ ORIGIN $\times$ AXIS
   (Architecture sec C). Implementation law, gated
   `<architecture><relevance>` (default false $\to$ byte-identical):
   - **Bottom-up (salience/novelty), both towers:** signal $=$ the
     per-percept settle residual (`snap_settle_qe`; novelty polarity —
     parts/properties that RESIST settling are salient) computed on each
     tower's VIEW HALF of the bind carrier (`combine.views`): particle
     novelty from the PS view, property novelty from the WS view; stashed
     per tower (`relevance_weights()` pull-API returns it).
   - **Slot $\to$ row projection:** the snap's own argmax map (settled
     slot $\to$ order-0 codebook row), scatter-amax of slot novelty onto
     rows.
   - **Symbolic history:** heat over symbols $\to$ concept rows through
     the allocator's `('sym', id)` constituent records
     (`ConceptualSpace.symbol_history_priority(heat)`) — the projection
     is implemented and unit-tested; the live heat source stays dark
     until `<symbolicPriming>`/`<attention>` are enabled.
   - **Combination:** SUM of bases (the priority-map law; no veto).
   - **Upward spread:** priority climbs the pyramid through EDGE
     MAGNITUDES — rung score $= p[\mathrm{row}] + (|W|\,p)[\mathrm{row}]$
     (`forward_linear_abs`), rank $= |cand| \cdot (1 + \mathrm{score})$,
     and only ADMITTED (top-K) rows carry their score upward — relevance
     spreads through awareness, never through rejected content.
   - **Deferred to the next projection:** readingAttention's symbolic
     origin (template from hot symbols / pyramid winners via the
     symbol$\to$word-whole map).
5. **SIMPLIFIED LAW (Alec, 2026-07-11 — supersedes most of decision 4):**
   "bottom-up things are primed in virtue of being SEEN; top-down things
   are primed in virtue of being DESIRED or HATED. That gives us a single
   quadratic mapping over which we can implement readingAttention
   (hard-coded)." Implementation: ONE priming surface per space
   (`prime_seen` bump + `<primingDecay>` exponential decay;
   `prime_desire` signed with floor 0 — suppression, never veto;
   per-codebook rows, CS over the concept inventory, WS over the
   analysis store). The CS surface IS the pyramid's ranking score
   (boost $-$ 1; the spread law from decision 4 is KEPT); the pyramid's
   admitted rows write back via `prime_seen` (awareness primes).
   `readingAttention` hard-coded: scope $=$ span of the hottest-primed
   word-whole (`_primed_reading_step`, active when `<relevance>` on and
   the learned producer off). DELETED from decision 4: the
   settle-residual novelty assembly, per-tower view-half scatter
   projection. KEPT: `forward_linear_abs` + the spread law,
   `symbol_history_priority` (the SS$\to$CS bridge), the `<relevance>`
   gate.

## Gates (Alec commits at each)

- **Gate A:** PS/WS symmetry refactor green — class removed, dual signatures
  in, IS-to-both wired, WS transpose recorded; off-path byte-identical (recon
  round-trip suite green). Serial-routing CHECKPOINT presented.
- **Gate B:** callosum-fed order-0 live on `MM_sparse_concept` (glued frame
  reaches the symbolic phase; snap/rectifier retired on the path).
- **Gate C:** top-K `.index` pyramid live — per-order selection $8/4/2/1$,
  grads reach every level, overflow loud; `make test` green.

## Success criteria

1. **Symmetry:** one dual signature, no `PerceptualSpace`, both towers fed
   from InputSpace views of the same input; state_dict keys unchanged.
2. **Mechanism-live:** on `MM_sparse_concept`, the glued order-0 frame is
   nonzero and grad-bearing; every pyramid level selects its top-K and passes
   gradient (`param.grad.abs().sum() > 0` per level) — all pinned.
3. **Capacity:** store sized off tiles + pool; overflow loud; `<nVectors>`
   untouched as inventory.
4. **No off-path regression:** serial + sO=0 byte-identical (recon round-trip
   suite green); `make test` green. Serial routing migration only via the
   explicit checkpoint.

## References

- doc/plans/2026-07-03-reconstruction-fidelity-design.md (Parked scope)
- doc/plans/2026-07-03-iterated-symbolic-loop-execution.md items 12-13
- doc/Architecture.md "corpus callosum" ($\approx$ 577); doc/Spaces.md 36/87
- bin/Layers.py: `ConceptualCombine` 1455 (callosum 1657-1714);
  `SparseLayer` 4405; `ConceptualAttentionLayer` 4591
- bin/Spaces.py: `PerceptualSpace` 9593; `PartSpace` 9643; `WholeSpace`
  17106; `set_index` 6556; `_apply_index_selection` 6980; `materialize`
  7009; top-K codebook forward 3794; `cs_snap_order0` 14631;
  `cs_forward_content` 14698
- bin/Models.py: IS dual product 3736; stage pump 6800-6869; P3 cutover
  7067; stage sizing 5989
