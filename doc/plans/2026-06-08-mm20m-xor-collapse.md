# MM_20M XOR head outputs constant 0.5 -- pipeline collapse

Status: **diagnosed (collapse location identified), not fixed**.
Scope: `MM_20M.xml`, `chunking=radix`, `lexer=byte`, all four XOR
prompts. CPU.

## TL;DR

After the radix reconstruction fix (`embed.py` neg-sampling SBOW)
landed, MM_20M's per-prompt **reconstruction is byte-perfect**
across all four XOR sentences -- but the OutputSpace head still
outputs **exactly $0.5$** on every input, label or no label. Logit
$= 0$ uniformly is what MSE training produces when the head sees the
**same** input on every batch row: the gradient pulls it to the
mean label, and the head is the regression of the (constant) symbol
slab onto the (varying) label sequence -- by row, that is the mean
$\frac{1}{4}(0 + 1 + 1 + 0) = 0.5$.

The reconstruction works because it walks the PS reverse
(percept-id $\to$ byte lookup), which does **not** require the
CS / SS forward to be informative. So reconstruction proves the
percepts are distinct; it does not prove the conceptual or symbolic
stages preserve that distinction.

## Confirmed: butterflies on $\sigma\pi$ ARE wired

The user's standing guidance is "make sure we are using butterflies
on SigmaPi if that is the route the model is taking." Verified:

- `data/MM_20M.xml` contains `<sigmaPi>butterfly</sigmaPi>` (line
  44, with a long comment block explaining why).
- `<conceptualOrder>3</conceptualOrder>` (cubic combine).
- `<conceptualMode>parallel</conceptualMode>` (the per-stage
  `_forward_per_stage` path, not the per-word grammar loop).

`ConceptualCombine` (`bin/Layers.py:1528`) uses
`GrammarLayer(butterfly=True)`; the 2x2-LDU butterfly forward /
reverse went green this round (370 grammar / invertibility / sigmapi
tests pass) including the new `_stable_pair_d` $|d|$ clamp. So the
$\sigma\pi$ machinery is structurally fine.

## Diagnostic attempt and why it returned nothing

I added a `XOR_DBG`-gated probe in the post-train
`runReconstruction` report that reads
`{perceptual,conceptual,symbolic,output}Space.subspace.materialize()`
and reports per-row $L^{2}$ norms and pairwise distances across the
4 prompts. Result: every subspace returned `<no tensor>` -- by the
time the reconstruction report runs, the subspaces have been
cleared / are not populated. The probe was removed.

To localize this properly the next investigation needs a probe **at
the end of a forward pass on a single XOR batch**, not at the
post-epoch report. The right insertion point is the final
materialization right before
`outputs = self.outputSpace.forward(symbols)` at
`bin/Models.py:2035`.

## What the constant-$0.5$ output tells us

Inputs ('hello world', 'hello there', 'loving world', 'loving there')
become exactly four distinct percept-id sequences (reconstruction
is byte-perfect). One or more of the following stages collapses
them to a single representative state seen by
`outputSpace.forward`:

1. **`ConceptualCombine`** (cubic combine + $\sigma\pi$ butterfly).
   If the codebook=`none` symbolic flow drops the $\sigma\pi$
   contribution (or zero-inits something), the cubic combine
   degenerates. **Most likely** given `<codebook>none</codebook>` on
   both PS and SS in the sweep (`SWEEP_CODEBOOK=none`).
2. **Symbolic codebook = `none`.** The harness override
   (`SWEEP_CODEBOOK=none`) drops the SS quantize codebook to a
   pass-through. If the pass-through accidentally averages /
   $\mathrm{argmax}$es to the same row across all inputs (e.g. via an
   un-trained projection whose initial output happens to be the
   same constant), the OS head sees one input. MM_20M's checked-in
   default is `<codebook>quantize</codebook>` on SS; running with
   the harness override may itself be the bug.
3. **OS-head $\mathrm{nDim} = 1$, $\mathrm{nVectors} = 1$.** `MM_20M.xml`
   defines OS as `nInput=8, nInputDim=1024, nOutput=1, nOutputDim=1,
   nVectors=1`. A 1-vector OS codebook can only quantize to a single
   row -- the head literally cannot represent two outputs. This is
   the architectural concern logged as Task #11
   ("Head/endpoint dim sizing -- ARCHITECTURAL, needs decision").
   **Even if** CS / SS preserve the distinction, the OS head must
   be re-sized for two classes (or wired as unquantized regression).

## The "unweighted XOR head" framing (Task #10)

Task #10 in the project's existing backlog was filed as
"Fix MM_20M_grammar unweighted XOR head". The earlier finding (not
re-verified this round) was that the OS head's loss weighting
dropped the XOR signal under the labeled-data loss term. With the
reconstruction now actually distinguishing the four inputs (a
prerequisite Task #10 didn't have), the right next step is to:

1. Re-run with the **checked-in** SS codebook
   (`<codebook>quantize</codebook>`) instead of the harness
   `SWEEP_CODEBOOK=none` override, with reconstruction-fix and
   d-clamp in place. See whether the predicted column moves off
   $0.5$ at all.
2. Add an in-forward XOR_DBG probe right before
   `outputSpace.forward` (per "Diagnostic attempt" above) to
   identify the exact collapse stage.
3. If CS / SS preserve the distinction and OS is still constant:
   confirm Task #11 -- re-size OS (`nVectors=2` for two-class,
   or drop the codebook entirely for sigmoid regression) and
   ensure the OS-head loss weight is non-zero (Task #10's original
   subject).
4. If CS / SS collapse: the cubic combine $\sigma\pi$ butterfly
   isn't producing distinguishable concepts despite being wired
   on. Most likely the SS `codebook=none` path doesn't carry the
   conceptual signal through. Re-test with
   `codebook=quantize`; if that fails, instrument
   `ConceptualCombine` directly.

## What landed this round (kept)

- Radix reconstruction fix: `embed.py` neg-sampling SBOW via
  `PretrainModel.sbow_loss_indices`, integrated into
  `Models.perceptual_sbow_loss`. Four XOR prompts now reconstruct
  byte-perfect on CPU.
- Butterfly $|d|$ clamp: `GrammarLayer._stable_pair_d` with
  $|d| \in [10^{-3}, 1]$, applied identically in forward /
  reverse. 370 grammar / invertibility / sigmapi tests green.
- `RadixLayer.reverse` decode fix: falls back to the PS table when
  the SS codebook is absent (`codebook=none`); bounds the
  reconstruction search to active rows.

## What did NOT land

- MPS divergence for `chunking=radix`. Documented in
  `doc/plans/2026-06-08-radix-mps-adam-step-zero.md`. Symptom
  ("non-finite reconstruction loss") was a misdiagnosis; the
  underlying corruption is a radix-only MPS in-place 0-dim
  increment being dropped (the Adam step counter and the
  `isfinite` reduction both surface it). CPU is the verification
  target for now.
- XOR head making the predicted column move off $0.5$. The
  reconstruction fix unblocks Task #10 but does not solve it.
- BPE / MPHF reconstruction. Deferred per the user's directive
  ("share codebooks across chunking methods, but thats a bigger
  refactor, so BPE/MPHF can wait").
