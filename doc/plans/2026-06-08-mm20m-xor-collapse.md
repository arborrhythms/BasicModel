# MM_20M XOR head outputs constant 0.5 -- pipeline collapse

Status: **RESOLVED (2026-06-09)** -- both root causes fixed (`lexer=raw`
label path so supervised labels are not dropped; 1-vector SBOW + radix
self-chunking for percept anti-collapse), and a latent butterfly SS-sizing
bug (exposed by the SS-sigma restoration) repaired. XOR solves at
300--450ep with byte-perfect reconstruction. The contraction-fix
exploration in the later sections (LayerNorm / residual / sum-of-outputs /
`reconstruct=expectation`) was a dead-end, **superseded by
`doc/plans/2026-06-09-asymmetric-vq-symbolic-ss.md`**.
Scope: `MM_20M.xml`, `chunking=radix`, all four XOR prompts. CPU.

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

---

# 2026-06-08 evening -- deeper investigation (resolves several open Qs above)

All runs CPU, `chunking=radix`, `lexer=byte`, `MODEL_COMPILE=none`,
four XOR prompts, via temp configs built from `MM_20M.xml`.
Diagnostics added as **env-gated probes** (off by default): see
"Probes" below. **Nothing committed.**

## TL;DR of new findings

1. **The "XOR" task is DEGENERATE**: the whole sentence is encoded as
   **one percept** (chunking promotes the whole line). So "XOR"
   reduces to a 4-atom lookup, not word-structure XOR. A whitespace
   word-split fixes it ($1 \to 2$ percepts), but the collapse below
   persists either way.
2. **The head loss is NOT unweighted** (corrects the Task #10
   framing): `runBatch:3288` applies the supervised head loss with
   `weight=1.0`; pred/target shapes $(4,1,1)$ match. The head *gets*
   XOR gradient -- it just sees a collapsed input.
3. **ConceptualCombiner derivation RESOLVED**: `next_cs` reads the
   **ps-slot only**; the `ss` and `cs` streams go into the *augment*,
   never the carrier. So the combine does NOT integrate the symbol or
   the prior carrier into the head-visible state at init.
4. **Two distinct collapses**: at order $1$ the **percepts** rotate to
   a common (null) direction; at order $\ge 2$ the **carrier drains**
   at $t{=}1$ and the **SS event collapses** ($\approx$ identical
   across all 4 inputs).
5. **SBOW is the anti-collapse force**, DISABLED by one-percept rows
   (`perceptual_sbow_loss` skips rows with $<2$ percepts). Word-split
   re-enables it ($\texttt{body\_sub}$ end-distinctness $0.17 \to
   0.41$, $2.4\times$) but is insufficient on its own.

## 1. Chunking degeneracy [NEW]

`PerceptualSpace._embed_radix` is handed `batch_tokens =
['hello world']` -- the byte lexer passes the **whole line** as one
token, and `observe_chunk` promotes it. Probe (`XCHUNK_PROBE`):

    non-null-pids/row = [1,1,1,1]   pid_2d[0] = [14, 0,0,0,0,0,0,0]
    lexer-tokens[0]   = ['hello world']

So each of the four sentences is a **single distinct atomic percept**.
"XOR" is then a 4-way lookup over 4 orthogonal vectors -- *linearly
separable* -- so the failure is NOT a missing nonlinearity; it is the
collapse (below). The docstring at `_embed_radix` (Spaces.py:8810)
*claims* "whitespace-split each row's text into chunks" but the split
never happens. An env-gated `RADIX_WORD_SPLIT` patch (Spaces.py, after
`batch_tokens` is resolved) splits on whitespace $\Rightarrow$
`non-null-pids/row = [2,2,2,2]`, `pid_2d[0] = [13,14,0,...]`
(`'hello'`=13, `'world'`=14). Real word structure, verified.

## 2. The head loss IS applied [resolves Task #10 framing]

Probe (`XLOSS_PROBE`) at `runBatch` (Models.py:3150, 3301, 3318):

    runBatch ENTER train=True split='train'
    REACHED outputTensor=(4,1,1) outputDataPred=(4,1,1)
    pred=(4,1,1) tgt=(4,1,1) match=True   -> output_weight = 1.0

So the supervised XOR head loss is **on, weight $1.0$**, shapes match.
Task #10's "unweighted head" is not the mechanism on this path; the
head receives gradient but `body_sub` (its input) is collapsed.
(Earlier I briefly mis-concluded the branch never ran -- that was a
test harness bug, I printed only `stderr`, not `stdout`. Corrected.)

## 3. ConceptualCombiner derivation [RESOLVES "is the combine the issue?"]

`ConceptualCombine.forward` (Layers.py:1651):

    x       = cat([ps, ss, cs])          # [..., 3D],  ps FIRST
    out     = layer.forward(x)           # invertible layer (identity-init)
    next_cs = out[..., :D]               # FIRST D coords
    aug     = out[..., D:]               # the rest

At identity init `layer(x) = x`, so $\texttt{next\_cs} = \texttt{ps}$
**only**. The `ss` and `cs` streams land entirely in `aug` -- they
**never reach the carrier** (hence never the head) until the combine
*learns* a non-trivial mix, which it does not (weak gradient /
collapse). Consequences:

- The carrier $=$ `PS_t`. With the current code `PS_t` is the percept
  every stage (the "percept-every-stage" edit survived,
  Models.py:5554), so the carrier carries the percept and does NOT
  drain to zero -- but it also does NOT "think" (no `ss`/`cs`).
- The restored **SS $\sigma$ fold's nonlinearity is in `ss` $\to$
  shunted to `aug`**, so it cannot help the head at init. This
  explains why restoring the SS sigma fold did not move predicted.
- **At order $1$** the combine is *fine*: `SS_t = CS_t = 0` at $t{=}0$
  (SS runs on the empty seed; $CS_{-1}$ is the seed), so
  $\texttt{next\_cs} = \texttt{percept}$. The order-1 blocker is the
  percept collapse (#4), not the combine.
- **At order $\ge 2$** `ss`/`cs` still never reach the carrier.

**Proposed fix (additive-init).** Initialize the combine so
$\texttt{next\_cs} = \texttt{ps} + \texttt{ss} + \texttt{cs}$, kept
invertible by
$W = \left[\begin{smallmatrix} I&I&I\\ 0&I&0\\ 0&0&I \end{smallmatrix}\right]$
(reverse: $\texttt{ps} = \texttt{next\_cs} - \texttt{ss} - \texttt{cs}$,
read $\texttt{ss},\texttt{cs}$ from `aug`). Then the carrier integrates
the symbol (the $\sigma$ nonlinearity) and the prior carrier from
$t{=}0$. NOTE: MM_20M uses the **butterfly** combine
(`sigmaPi=butterfly`), so the additive-init must be derived for the
2x2-LDU cascade, or the combine switched to the dense `full` bridge
for the XOR experiment.

## 4. The percept collapse [order-1 blocker]

`XHEAD_PROBE` measures pairwise row-distinctness over training.
Order $1$, 150 epochs:

    raw percepts (pre-pi):  maxpairdist 2.57 -> 0.26   norms 1.80 -> 1.21
    body_sub  (head input): maxpairdist 2.57 -> 0.17

The norms barely shrink (33%) while distinctness collapses (90%): the
four percepts **rotate to a common direction** (the null percept), not
shrink to zero. Ruled OUT as the sole driver (each disabled alone, the
collapse persists):

- reconstruction loss: `reconstructionScale=0` still collapses.
- L1 sparsity: `l1Lambda=0` still collapses.
- masked-LM: `maskRate=0` still collapses (2.59 -> 0.25).

**SBOW** (`Models.perceptual_sbow_loss` $\to$ `embed.sbow_loss_indices`)
is the codebook's antipode repulsion ("keeps the codebook from
collapsing toward a single point / the null percept", embed.py:1670).
It loops **per row** and `if len(used) < 2: continue` -- so a
one-percept sentence gets **no repulsion at all**. Word-split (2
percepts/row $\to$ SBOW fires) lifts `body_sub` end-distinctness
$0.17 \to 0.41$ ($2.4\times$) -- real, but still too tight for the
head (predicted stays $\approx 0$).

**OPEN.** With SBOW the only known anti-collapse term and it
re-enabled, `body_sub` still contracts $2.5 \to 0.41$, so a *residual*
force rotates the percepts toward null. (The user flagged "codebook
contraction is incorrect hypothesis" -- consistent with this being a
gradient pulling the percepts together rather than the codebook
self-shrinking; the driver is not yet pinned. Per-loss-term
instrumentation via `TheError.breakdown()` is the next probe.)

## 5. SS symbolic collapse [order $\ge 2$]

`XSS_PROBE` on the SS event (`SS_sub.materialize()`), order $3$:

    t=0  SS empty (runs on the seed)
    t=1  SS_event maxpairdist = 0.016     (~identical across 4 inputs)
    t=2  SS_event maxpairdist = 0.020

So by $t{=}1$ the symbolic representation no longer distinguishes the
inputs -- downstream of the carrier behaviour in #3/#4.
(`subspace._active` is `None` on this path; the SS-event distinctness
is the usable signal.)

## Changes currently in the working tree (uncommitted, default-off)

- `RADIX_WORD_SPLIT` env $\Rightarrow$ whitespace-split tokens in
  `_embed_radix` (Spaces.py). The principled version is a config knob,
  not an env gate.
- `reconstruct` enum (`perfect|mixed`, default `mixed`) replacing the
  `<perfectReconstruction>` boolean (Models.py, data/model.xsd, test).
- `PS_t` = percept every stage (Models.py:5554) -- caller no longer
  zeroes PS post-$t{=}0$.
- SS $\sigma$ fold restored in parallel mode (Spaces.py
  `SymbolicSpace.forward`); nonlinear `tanh`. NOTE per #3 this is
  currently shunted to the combine augment.
- `<conceptualOrder>1</conceptualOrder>` in `MM_20M.xml` (was 3).
- The forward stage-reorder (cs $\to$ ss $\to$ combine) was tried and
  **reverted** -- it broke reconstruction (reverse + STM + ss.forward
  input all need coordinated changes).

## Probes (env-gated, all off by default)

- `XHEAD_PROBE`: `body_sub` (Models.py:7320), raw-pre-pi vs post-pi
  (`=XPI=`, Spaces.py), percept (`=XPCPT=`).
- `XSS_PROBE`: SS active symbols + event distinctness (Models.py after
  `SS_sub`).
- `XLOSS_PROBE`: head-loss branch entry / shapes (Models.py:3150+).
- `XCHUNK_PROBE`: radix chunk count / lexer tokens (Spaces.py
  `_embed_radix`).

These should be removed (or folded into a single `XOR_DBG`) before any
commit.

## Recommended next steps

1. **Pin the residual percept-collapse driver** (#4): dump
   `TheError.breakdown()` per step to see which weighted term's
   gradient rotates the percepts to null. SBOW counters it but doesn't
   win.
2. **Combine integration** (#3): implement the additive-init so
   `next_cs = ps + ss + cs` (derive for butterfly, or switch to dense
   `full` for the experiment). This is the structural change that lets
   the symbol/carrier reach the head.
3. **Make `RADIX_WORD_SPLIT` a real config knob** so XOR is tested on
   word structure, not a single atom.
4. Head sizing (Task #11) is secondary -- the head loss is already
   weighted and the head flattens (not means), so it can represent two
   outputs once `body_sub` is distinct.

---

# 2026-06-08 LATE EVENING -- ROOT CAUSE: `lexer=byte` discards the supervised labels

**CRITICAL.** This supersedes/confounds much of $\S4$--$\S5$ above:
those were all measured with `lexer=byte`, under which **the XOR
labels never reach the loss**. Found while instrumenting the output
loss with `XLOSS_PROBE` (prints `pred` vs `tgt` values).

## The bug

`runEpoch` picks the training data path by

    use_byte_cursor = (text_input and byte_lexer)      # Models.py:4266

With `lexer=byte`, `use_byte_cursor=True` and the **autoregressive
byte-cursor** path runs (self-supervised on input bytes). It feeds a
fixed **zero stub** as the target and DROPS the per-batch labels:

    if use_byte_cursor:
        byte_stub_output = prepOutput(self._stub_outputs(B_eff))  # B zeros
    ...
    outputTensor = byte_stub_output                    # Models.py:4381

`_stub_outputs` (Models.py:4050): *"B zero scalars ... matches the
placeholder created by Data.processLM when labels are absent."*

## Proof

`XLOSS_PROBE` dumping the unique `tgt` patterns across ALL training
batches:

    lexer=byte:  tgt = [0.0, 0.0, 0.0, 0.0]   <- XOR labels GONE
    lexer=raw:   tgt = [0.0, 1.0, 1.0, 0.0]   <- real XOR labels

So with `lexer=byte` the head trains against constant zero $\to$
predicts $0$, and the model has **no reason to keep the four inputs
distinct** -- which is most of the "collapse" measured in $\S4$/$\S5$.

**I had been running `SWEEP_LEXER=byte` throughout.** `MM_20M.xml` does
NOT set `lexer=byte` (its `<lexer>` is commented out). So the byte-lexer
runs were a *self-inflicted confound* layered on top of the real
problem. This is almost certainly the long-standing **XOR_exact vs
MM_20M difference**: XOR_exact uses a word/lexicon lexer $\Rightarrow$
`use_byte_cursor=False` $\Rightarrow$ the Trial path
(`outputTensor = prepOutput(out_items)`) $\Rightarrow$ labels reach the
loss. (A patch at Models.py:4381 to forward `out_items` did NOT help --
`out_items` is *already* zeros in the byte cursor, so the labels are
dropped at the data-loader/`tick` level, not just at 4381.)

## With labels fixed (`lexer=raw`), the GENUINE XOR problem remains

`lexer=raw` + radix, 300 epochs:

    predicted = [0.477, 0.523, 0.479, 0.521]   labels [0,1,1,0]   ->  ~0.5

The tiny split is by the **second word** (`world`$\to 0.478$,
`there`$\to 0.52$) -- a **linear** feature, not XOR. The head separates
only linearly, so on balanced XOR it sits at the mean. This is exactly
the representation problem the combiner derivation ($\S3$) predicts:
`next_cs = ps`-slot, so `ss`/`cs` (the nonlinear interaction) never
reach the head. **The additive-init combiner ($\S3$ fix) is the real
remaining work**, now testable on correct labels.

## Revised next steps (supersedes the earlier list)

1. **Stop using `lexer=byte` for the supervised XOR task** -- use `raw`
   or `word`. The principled fix is to make the **data loader / cursor
   `tick`** carry the real `out_items` labels in byte-cursor mode (they
   are zeroed before Models.py:4381, so the loader is the site).
2. **Re-validate $\S4$/$\S5$ (percept/SS collapse, SBOW, masked-LM) on
   `lexer=raw`** -- the byte-lexer numbers are contaminated by the zero
   targets; the real collapse magnitude on correct labels is what
   matters.
3. **Additive-init combiner ($\S3$)** so `next_cs = ps + ss + cs` -- the
   structural fix for the residual $\approx 0.5$ (the head needs the
   nonlinear `ss`/`cs` interaction to separate XOR).
4. The chunking word-split ($\S1$) and SS $\sigma$ fold are still
   wanted, but secondary to (1)--(3).

## Addendum: additive-combine experiment (inconclusive)

Tried the $\S3$ fix as a quick residual (env `XOR_COMBINE_ADD`,
Models.py after the combine): `next_cs_content += SS_t + CS_t`.
`lexer=raw`, order $3$, 400 epochs $\Rightarrow$ predicted
$[0.477, 0.523, 0.479, 0.521]$ -- **byte-identical to without it**. So
`SS_t + CS_t \approx 0` at the combine: the symbolic / carrier streams
are themselves collapsed (or the `cs_event is not None` guard skips the
combine block). So the carrier-integration fix can't help until `ss`/
`cs` actually carry distinct signal -- i.e. the collapse is upstream of
the combine, in the SS/CS forward, not (only) in how the combine reads
its slots. This re-points at $\S5$ (SS event collapse), to be
re-measured on `lexer=raw`.

## Working-tree state at end of 2026-06-08 (NOTHING COMMITTED)

Non-gated behavioural changes (alter the model; made earlier per user
direction or exploration -- review given the byte-lexer finding):

- `PS_t` = percept every stage (Models.py ~5554).
- SS $\sigma$ fold restored in parallel `SymbolicSpace.forward`
  (Spaces.py) -- note $\S3$: currently shunted to the combine augment.
- `reconstruct` enum (`perfect|mixed`, default `mixed`) -- Models.py,
  data/model.xsd, test/test_conceptual_recurrence.py.
- `<conceptualOrder>1</conceptualOrder>` in `MM_20M.xml` (was 3).

Env-gated experiments (off by default, harmless):

- `RADIX_WORD_SPLIT` (Spaces.py `_embed_radix`) -- whitespace-split.
- `XOR_USE_LABELS` (Models.py ~4381) -- ineffective (`out_items`
  already zeroed upstream); keep only as a marker for the data-loader
  fix.
- `XOR_COMBINE_ADD` (Models.py after combine) -- the additive residual
  above.

Diagnostic probes (env-gated, off by default; remove or fold into one
`XOR_DBG` before any commit): `XHEAD_PROBE`, `XSS_PROBE`, `XLOSS_PROBE`,
`XCHUNK_PROBE`, `XLOSS_BREAKDOWN`.
