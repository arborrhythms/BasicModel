# Asymmetric VQ training and the subsymbolic / symbolic (PS / SS) split

Status: **design -- basis for code + doc changes.** Supersedes the
contraction-fix line of `doc/plans/2026-06-08-mm20m-xor-collapse.md`
(LayerNorm / residual / sum-of-outputs / `reconstruct=expectation`),
which is retired by the findings below.
Scope: `Models.py`, `Spaces.py`, `Layers.py`, `embed.py`, `model.xsd`;
`MM_20M.xml` (parallel) and `MM_20M_grammar.xml` (serial).

## TL;DR

The MM_20M XOR hunt bottomed out on **two clean separations** that
should drive the symbolic layer:

1. **PS is subsymbolic, SS is symbolic.** PerceptualSpace is continuous
   (`<codebook>none</codebook>`) and carries lossless byte
   reconstruction through the percept ids. SymbolicSpace is a quantized
   codebook (`<codebook>quantize</codebook>`) -- a lossy, meaning-bearing
   bottleneck.
2. **Output trains the encoder, input trains the codebook.** The forward
   (predict output) routes its gradient through the quantizer to the
   encoder via STE; the reverse (reconstruct input) lands its gradient on
   the codebook with no STE. The two passes cover both parameters, which
   makes the standard VQ **commitment loss and EMA update unnecessary**.

This plan records the findings, the decisions, and the tasks to land
them. XOR is *not* a validator for the symbolic side (see $\S$7).

## 1. What the XOR hunt established

- The original failure was a **data-path bug, not an architecture flaw**:
  `lexer=byte` $\to$ byte-cursor $\to$ self-supervised AR $\to$ the XOR
  labels were replaced by zero-stub targets. `lexer=raw` lets the real
  labels reach the loss.
- With correct labels, XOR is solved by keeping the four percepts
  **distinct**. The radix store promotes each recurring sentence to a
  single atomic percept; the anti-collapse term that keeps them apart is
  the **SBOW** negative-sampling repulsion on the percept codebook.
- **SBOW works on a single percept**: a centroid of one still has an
  antipode, so the negative-sampling term still repels random codebook
  rows. This retired the earlier word-split hack -- the radix learns its
  own chunking and the 1-vector SBOW supplies the repulsion. Working
  recipe: `lexer=raw` + radix + 1-vector SBOW + plain `mixed` combine
  $\to$ XOR solved at $300$--$450$ epochs, reconstruction byte-perfect.
- The **carrier "contraction"** (the conceptual carrier *content* shrinks
  $\sim 15\times$ per stage; rowdist $0.50 \to 0.06$) is real for the
  content channel but a red herring for the *state*: the recurrent state
  is the whole SubSpace, and reconstruction stays byte-perfect through the
  apparent collapse. The contraction-fighting machinery is therefore
  retired -- it bought speed at the cost of invertibility (LayerNorm) or
  actively hurt (sum-of-outputs diluted the deep answer with the shallow).
- **The SS `codebook=quantize` path is a no-op in parallel mode**
  (`Codebook.quantize()` is never called; verified by probe -- $0$ calls)
  and **live in serial / grammar mode** (the `VectorQuantize` customVQ
  path fires; $4$ calls in `MM_20M_grammar.xml`). So
  *parallel $=$ subsymbolic, serial $=$ symbolic* is already the de-facto
  split in the code.

## 2. The two separations (the through-line)

**(a) PS subsymbolic / SS symbolic.**

- PS: continuous, `<codebook>none</codebook>`, lossless byte
  reconstruction via percept ids. SBOW gives it a meaningful continuous
  space (anti-collapse via the antipode).
- SS: a quantized symbol codebook, **semantic** (not byte-faithful). A
  symbol is a lossy, meaning-bearing bottleneck; do not expect the SS
  codes themselves to reconstruct bytes -- that is PS's job.

**(b) output $\to$ encoder / input $\to$ codebook.**

- Forward (predict output): STE routes the gradient through the quantizer
  to the encoder ($\Sigma$ / $\Pi$); the codebook is detached. **Output
  shapes the encoder only.**
- Reverse (reconstruct input): the gather $z_q = e[\mathrm{idx}]$ is a
  plain differentiable index, so the recon gradient lands on the codebook;
  the argmin blocks the path to the encoder. **Input shapes the codebook
  only.**

## 3. Why this retires commitment + EMA

Each objective's gradient **vanishes at the argmin for one of the two
parameters**, and the two vanishings sit on opposite sides:

- output $\to$ codebook vanishes (the code is detached under STE);
- recon $\to$ encoder vanishes (the argmin is non-differentiable).

So output trains the encoder and recon trains the codebook -- the two
passes cover both parameters with no scaffolding. Commitment
($\beta\,\lVert x - \mathrm{sg}(e)\rVert^2$, encoder $\to$ code) and EMA
($e \leftarrow \gamma e + (1-\gamma)\,\overline{x}$, code $\to$ encoder)
were single-objective crutches to train the codebook when only
reconstruction was present:

- commitment is **replaced by STE** (the encoder gets the output gradient
  directly);
- EMA is **replaced by the reconstruction gradient on the codebook**
  (exact, not a running average -- a richer input-faithfulness).

**The one crutch that remains.** STE is biased unless $z_e \approx z_q$,
and commitment also kept that gap small. So the encoder bound (the
$\pm 1$ saturated combiner) must carry that job. The codebook gradient
(reverse) is *clean* (no argmin in its path); only the encoder gradient
(forward) still rides STE. We have halved the pathology, not erased it.

## 4. The asymmetric routing (implementation shape)

- Forward / output: $z_q = z_e + \mathrm{sg}(e[\mathrm{idx}] - z_e)$
  (standard STE) $\Rightarrow$ $\partial L_{\text{out}}/\partial z_e =
  \partial L_{\text{out}}/\partial z_q$ and $\partial
  L_{\text{out}}/\partial e = 0$.
- Reverse / recon: $z_q = e[\mathrm{idx}]$ (plain gather, no STE)
  $\Rightarrow$ $\partial L_{\text{recon}}/\partial e[\mathrm{idx}] =
  \partial L_{\text{recon}}/\partial z_q$ and $\partial
  L_{\text{recon}}/\partial z_e = 0$.
- Rule of thumb: **STE $=$ "send this objective to the encoder"; plain
  gather $=$ "send it to the codebook."** Apply STE on the pass whose
  objective should shape the encoder (output), withhold it on the pass
  whose objective should shape the codebook (reconstruction).

## 5. The conceptual combiner -- a 2-stream invertible bind

Replace the 3-stream `combine(PS, SS, CS)` $\to$ `next_cs` + augment --
which read the carrier from the ps-slot only (1/3 rank, bleeding 2/3 to
the augment every stage: the $15\times$/stage contraction of $\S$1) --
with a **2-stream invertible bind** of the two input streams:

- Forward: $\mathrm{CS} = \mathrm{ILL}([\mathrm{PS} \Vert \mathrm{SS}])$
- Reverse: $[\mathrm{PS} \Vert \mathrm{SS}] = \mathrm{ILL}^{-1}(\mathrm{CS})$

**`ILL` is an `InvertibleLinearLayer` with the butterfly parameter set** --
NOT a separate "butterfly" class, just the `sigmaPi=butterfly` span of the
existing ILL (or the `SigmaLayer`, if that is what carries the butterfly
parameter). The matrix is square over $(8+8)\cdot n\mathrm{Dim} =
16\,n\mathrm{Dim}$ (parallel), so $\mathrm{CS}$ holds the *whole* mixed
$[\mathrm{PS}\Vert\mathrm{SS}]$, not a 1/3 slice.

**Why it is the right inversion.**

- **Contraction gone at the source** -- nothing is shed to an augment;
  $\mathrm{CS}$ is the full bind, so there is no rank bottleneck and no
  per-stage scale collapse.
- **Exact inversion, no augment threading** -- the butterfly ILL is a true
  bijection on its $16\,n\mathrm{Dim}$ domain, so $\mathrm{CS} \to
  (\mathrm{PS}, \mathrm{SS})$ is exact by construction; the whole
  augment-pairing reverse is deleted.
- **Right semantics** -- a concept is just the reversible bind of its
  perceptual form (PS) and symbolic form (SS).

**Implementation shape (per the design).**

1. The CS layer **materialises both input subspaces** (PS, SS) into a
   single double-wide event, runs the **butterfly ILL**, and **stores the
   result to a single CS subspace**.
2. It **returns two output arguments** -- in practice the *same* `.what`
   Basis with different `.active` index views (PS view $=$ first 8 slots,
   SS view $=$ last 8 slots of the inverse). No second Basis is allocated.
3. **Parameters are unchanged** -- the existing ILL, sized to the
   double-wide.
4. **`OutputSpace` must accept both subspaces** (PS and SS views) and
   combine them for its final output.

**Pin-downs.**

- **The double-wide is internal to CS** -- the CS subspace *stores* the
  $16\,n\mathrm{Dim}$ mix, but the input/output **interface is the
  individual subspaces** ($8\,n\mathrm{Dim}$ each, the `.active` views), so
  nothing downstream widens: the head and next stage consume PS/SS views,
  not a doubled carrier. (Butterfly is $O(M \log M)$.)
- **Recurrence re-close** -- with no prior-$\mathrm{CS}$ input, stage
  $t{+}1$'s SS comes *from* $\mathrm{CS}_t$: $\mathrm{ILL}^{-1}(\mathrm{CS}_t)
  \to \mathrm{SS}_t$, then $\mathrm{SS}_{t+1} = \sigma(\mathrm{SS}_t)$,
  staying $8\,n\mathrm{Dim}$. Confirm the loop closes at $8\,n\mathrm{Dim}$
  and does not re-widen.
- **$\pm 1$ is the operating range, not a `tanh`** -- a linear butterfly
  ILL is an exact bijection but is not $\pm 1$-bounded; a saturating squash
  inside the combiner would break exact inversion. Keep the ILL purely
  linear and let PS/SS arrive already bounded (codebook / $\sigma$),
  norm-preserved through the butterfly.
- **Reconstruction ends at the percept store** -- $\mathrm{ILL}^{-1}
  (\mathrm{CS}) \to \mathrm{PS}$ recovers the continuous percept *vector*
  exactly; the vector-to-bytes step is still the unchanged id/store decode
  (PS owns reconstruction).

## 6. Semantic arrangement of the symbol space

- Not a per-symbol word2vec (a symbol has no sentence-like context the way
  a word does). Instead a **post-sentence** step over SS heat across
  conceptual orders: pode $=$ semantic centroid (attraction), antipode
  $=$ repulsion from the rest of the codebook.
- The pode is a *real-corpus* feature (it needs co-occurrence statistics);
  the antipode is the universal anti-collapse.
- **Open decision:** single-snap vs sparse multi-symbol code. The "compact
  basis" / centroid framing implies a *sparse* code; a single VQ snap with
  a centroid training target is a soft-target / hard-snap mismatch that
  only STE papers over.

## 7. Tasks

### A. Cleanup -- revert the dead experiments (immediate, low-risk)

1. Remove the contraction-fix machinery: `reconstruct=expectation` (carrier
   residual + per-stage LayerNorm) in `Models.py` and its enum value in
   `model.xsd`; the `XOR_SUM_OUT` sum-of-outputs readout in `Models.py`.
2. Strip all diagnostic probes / gates added during the hunt:
   `XNORM_PROBE`, `XHEAD_PROBE` / `=XPI=` / `=XPCPT=`, `XSS_PROBE`,
   `XLOSS_PROBE` / `XLOSS_BREAKDOWN`, `XOR_USE_LABELS` (`Models.py`);
   `XVQ_PROBE` (`Layers.py`); `=XQZ=` / `=XQZ-fb=` (`Spaces.py`);
   `XOR_NO_PODE` / `XOR_ONLY_PODE` (`embed.py`); the
   `XOR_COMBINE_ADD` / `XOR_RES_NORM` / `XOR_NO_RESID` / `XOR_NO_NORM`
   gates (these fall out with the expectation block).
3. `make test` green after cleanup (the gate).

### B. Keep + harden the fixes that worked

4. **SBOW-on-1-vector** (`embed.py:sbow_loss_indices` accepts $N \geq 1$,
   centroid $=$ the vector itself for $N = 1$; `Models.py:
   perceptual_sbow_loss` row guard $< 1$). Keep; document as the percept
   anti-collapse.
5. **Word-split removal** (`Spaces.py:_embed_radix` no longer whitespace-
   splits; the radix learns its own chunking). Keep.
6. **`lexer=raw` label path**: decouple `use_byte_cursor` from the lexer in
   the data loader so supervised labels survive a byte lexer; repin the
   byte-pinned configs (`MM_xor.xml`, `MM_20M_grammar.xml`).

### C. New architecture -- asymmetric VQ training

7. **Drop the `<codebook>` element (the config knob) from BOTH PS and SS** --
   it is no longer optional. The codebook itself **stays** in both spaces;
   only the settable option goes: **SS** codebook is mandatory (`quantize`,
   hardwired); **PS** codebook is **integrated with `<chunking>`** (the
   chunking / radix store *is* the PS codebook).
8. **Parallel-mode quantization** -- **DECISION (2026-06-09): parallel SS
   genuinely quantizes** (the symbolic/quantized path is NOT serial-only).
   Make `Codebook.quantize()` actually fire in the parallel
   `_forward_per_stage` path (today it is a no-op there), so the SS codebook
   is live in both modes.
9. Implement the **asymmetric STE routing** ($\S$4): STE on the
   forward / output pass (gradient $\to$ encoder); no STE on the
   reverse / recon pass (gradient $\to$ codebook).
10. **2-stream invertible combiner** ($\S$5): replace the 3-stream
    `combine(PS, SS, CS)` with $\mathrm{CS} = \mathrm{ILL}([\mathrm{PS}
    \Vert \mathrm{SS}])$ (the butterfly `InvertibleLinearLayer`). The CS
    layer materialises PS $+$ SS into one double-wide event, runs the ILL,
    stores a single CS subspace, and returns the PS / SS views (same
    `.what` Basis, different `.active`). `OutputSpace` accepts both
    subspaces. No new parameters.
11. **Drop the standard commitment loss and EMA** for the SS VQ -- replaced
    by the dual gradients.
12. **Confirm the $\pm 1$ operating range** keeps $z_e$ near its code
    (carries commitment's runaway-guard / STE-bias job) before deleting
    commitment outright.
13. **Semantic arrangement** as a post-sentence step over SS heat across
    orders (pode + antipode), not a per-symbol word2vec. Decide single-snap
    vs sparse code ($\S$6).

### D. Validation

14. Validate the symbolic side on a **corpus via the serial / grammar
    path**, not XOR. Use XOR only as a "did we lose distinctness /
    reconstruction?" regression guard.

## 8. Why XOR cannot validate the symbolic side

XOR runs in the **parallel** path, where the SS VQ is a no-op -- it never
exercises a codebook. It is also a **parity** (anti-similarity) task,
where the semantic pode is neutral-to-harmful (the antipode / distinctness
is what helped XOR, not the pode). Two independent reasons the
semantic-codebook work must be measured on a real corpus on the serial
path.

## 9. The two separations, restated

Carry these as the through-line for every code and doc change:

- **PS subsymbolic / SS symbolic.**
- **output $\to$ encoder / input $\to$ codebook.**
