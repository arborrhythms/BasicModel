# Architecture

## Overview

BasicModel is a bidirectional neural architecture organized as a pipeline of five
**spaces** plus a grammar host (`WordSpace`), each implementing a distinct
representational transformation:

```
Forward:  InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace
Reverse:  OutputSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace

Feedback loops (cross-stage / cross-call):
    SymbolicSpace ──→ ConceptualSpace   (S→C symbolic loopback; per stage at order ≥ 1)
    ConceptualSpace ──→ PerceptualSpace (C→P subsymbolic loopback; cross-forward)
```

The forward pass transforms raw input into predictions; the reverse pass
reconstructs the original input from the symbolic representation. Both
directions are trained simultaneously with a single optimizer minimizing a
combined loss:

```
totalLoss = (1 - reconRatio) * outputLoss + reconRatio * reconstructionLoss
```

The legacy `SubsymbolicSpace` class (a parallel-mode bitonic Space alongside
SymbolicSpace) and `SyntacticSpace` (a separate derivation-tree Space) have
both been retired. The subsymbolic role is filled by `PerceptualSpace` itself
plus the new C→P feedback loop; syntax / grammar dispatch lives on
`WordSpace` and is attached to `SymbolicSpace` (the canonical grammar host).
The `MereologicalTree` sidecar that backed `part` / `equals` / `query` is
also retired — those operations are now pure-geometric clipped-cosine
projections on the SymbolicSpace bivector codebook.

### Spaces

| Space | Role | Layer Type | Notes |
|-------|------|-----------|-------|
| **InputSpace** | Lifts raw data into working dimensionality | LiftingLayer | nActive, nDim, nVectors |
| **PerceptualSpace** | Per-percept aggregation; subsymbolic substrate | `self.sigma` activated as the **Lift operator** at concept_dim → concept_dim (replaces the dormant per-percept sigma). Reads C→P feedback via `_sourced_input` (bivector-gated). | nActive, nDim, nVectors, invertible, bivectorOutput |
| **ConceptualSpace** | Multiplicative abstraction (P → C); host of `ShortTermMemory` | `self.pi` (used as the **Lower operator** too, via the gated path). Reads S→C feedback via `_sourced_input`. | nActive, nDim, nVectors, invertible, bivectorOutput, stmCapacity |
| **SymbolicSpace** | Discrete activation bottleneck (C → S) + Codebook + **grammar's calculator** | `self.sigma` + Codebook (bivector prototypes). SyntacticLayer dispatches the chart's S-tier rules here. Codebook plays the unified role of *codebook + lexicon-reference + meronymic-tree* — three jobs on one structure (geometric). | nActive, nVectors, codebook, bivectorOutput |
| **OutputSpace** | Final prediction | LinearLayer | nActive, nDim, nVectors |

Each level-crossing space owns one bidirectional layer that handles both
forward and reverse via its own self-inverse: `ConceptualSpace.pi` does P->C /
C->P; `SymbolicSpace.sigma` does C->S / S->C. Pi and Sigma have independent
weights. See [Logic.md §8](Logic.md) and [Spaces.md](Spaces.md).

Dimensions (`nDim`) are read from `TheObjectEncoding`. Codebook sizes
(`nVectors`) are likewise on `TheObjectEncoding`; the factory validates
`nVectors >= nActive`.

![MM_5M Hierarchical Progressive Bottleneck](diagrams/mm5m_architecture.svg)

Layer selection by `<reconstruct>` and `invertible`:

1. **`reconstruct=NONE`**: Non-invertible layers (`PiLayer`, `SigmaLayer`)
   forward-only. No reverse pipeline.
2. **`reconstruct=<any>` + `invertible`**: Single invertible layer
   (`PiLayer(invertible=True)`, etc.) serves both directions, sharing weights.
3. **`reconstruct=<any>` + not `invertible`**: Two layers with separate
   weights — `forward()` on one, `reverse()` on the other. Avoids the
   expressivity limitation where a non-invertible layer can't represent the
   inverse of another. Reverse uses matrix `pinv` (may be numerically
   unstable from SVD convergence). `<invertible>true</invertible>` avoids
   this via shared-weight inversion.

### Reconstruction Symbols

The symbolic bottleneck can lose information needed for reconstruction. XOR
maps 2 inputs to 1 output, but `XOR(0,0)=0` and `XOR(1,1)=0` are distinct
inputs producing the same output. A single output symbol cannot reconstruct
which input was presented.

`nSymbols` is split:

- **`nOutputSymbols`** `= OutputSpace.nActive` — fed to OutputSpace for prediction
- **`nReconSymbols`** `= nSymbols - nOutputSymbols` — carried in `end_state`
  for reconstruction

A skip connection through the symbolic bottleneck. Reconstruction symbols
receive gradient only from reconstruction loss.

### Single Optimizer with Overlapping Weight Spaces

The forward and reverse passes share a **single Adam optimizer** that
minimizes the combined loss. Forward and reverse weight spaces **partially
overlap** — neither disjoint (allowing independent optimizers) nor identical
(creating destructive interference). Some layers share weights between
directions (shared embeddings, the symbolic bottleneck); others are
direction-specific (`pi1`/`pi2`, `sigma1`/`sigma2`, `linear1`/`linear2`).

- **Shared weights** receive gradient from both losses, learning
  representations useful in both directions.
- **Direction-specific weights** specialize without interference.
- **No ping-pong**: separate optimizers on overlapping parameters would pull
  weights in alternating, conflicting directions each step.

When `invertible=true`, overlap is total: one invertible layer serves both
directions and receives the full combined gradient.

Reference: A.M. Rogers, T.T. Shannon, and G.G. Lendaris, "A comparison of DHP
based antecedent parameter tuning strategies for fuzzy control,"
*Proceedings Joint 9th IFSA World Congress and 20th NAFIPS International
Conference*, 2001, doi:
[10.1109/NAFIPS.2001.944317](https://ieeexplore.ieee.org/document/944317).

### Training Loop

Single Adam optimizer with persistent state (momentum/variance accumulate
across epochs):

1. Forward pass: input $\to$ prediction + `end_state`
2. Compute `outputLoss` from prediction vs. target
3. Reverse pass: `end_state` $\to$ reconstructed input
4. Compute `reconstructionLoss` from reconstruction vs. original input
5. Backpropagate combined `totalLoss`
6. If ergodic: run `paramUpdate()` (gradient energy sensor updates alpha)
7. Optimizer step (embedding params excluded when `trainEmbedding` is `NONE`,
   `CBOW`, or `SBOW`)
8. If `trainEmbedding` is `CBOW`, `SBOW`, or `BOTH`: run embedding update step

Alpha annealing (ergodic): starts at `1.0` (full exploration), decays to
`0.0` within the first 5% of epochs via `alpha = max(0, 1 - epoch / warmup)`
where `warmup = numEpochs // 20`. Code convention (`alpha=1` means explore)
is the inverse of the ergodic math convention; layers translate internally.

See [Params.md](Params.md) for all XML parameters. See
[Training.md](Training.md) for embedding modes.

### Modes of operation

Three orthogonal mode dimensions plus the two feedback loops.

**Butterfly mode** (`<useButterflies>true</useButterflies>`):
Pairwise sigma/pi with N-halving across `<conceptualOrder>` stages. Each
per-stage `ConceptualSpace` / `SymbolicSpace` is built with a butterfly-mode
SigmaLayer / PiLayer that packs `[B, N, D]` to `[B, N/2, 2*D]` before the
inner LDU and unpacks after. Slot count halves per stage; per-slot dim
doubles; total per-stage volume preserved. Validated by `nPercepts *
state_dim == nSymbols * symbol_width` at `ModelFactory.validate_config`.
Used by MM_xor, MM_5M, MM_400M.

**Serial mode** (`BASICMODEL_DEVICE` / runtime flag `serial_mode`):
Runtime fast path for streaming / autoregressive contexts.
`PerceptualSpace` / `ConceptualSpace` may use slide-and-recompute with the
previous step's per-cell warm cache (`subspace.serial_cache`). Independent
of butterfly mode. Cache is keyed on owner-Space id, cleared on hard
`Reset`. (Distinct from the **serial / shift-reduce parser** — a separate
deferred refactor; see [`doc/plans/`](plans/).)

**The two feedback loops**:
The architecture now exposes two concurrent recurrent paths between the
spaces. Their iteration cadences differ — they do NOT both follow
`conceptualOrder`.

| Loop | Direction | Iterations per forward call | Cadence governed by |
|---|---|---|---|
| **Symbolic loopback** | `S → C` | `T = conceptualOrder` (each stage `t ≥ 1` reads stage `t-1`'s S event) | `<conceptualOrder>T</conceptualOrder>` directly |
| **Subsymbolic loopback (new)** | `C → P` | **Exactly 1** per forward call (stem `P` reads `conceptualSpaces[-1]`.event before any C stage fires) | Cross-forward; not per-stage |

Conceptual input sourcing: each stage's `ConceptualSpace.forward`
combines two sources via `_sourced_input` — its own perceptual primary
(stem-cached) AND the previous stage's `SymbolicSpace.event`, lifted
from the bivector handoff `[B, V_S, 2]` back to concept content
`[B, V, concept_dim]` via `Codebook.project_reverse`. The combination
is additive after the bivector lift. Stage 0 cold-starts (no S
sibling event yet); the previous-stage symbolic loopback fires from
stage 1 onwards.

The new C→P loopback at PerceptualSpace works the same way (mirror of
the C-side mechanism): P's `_sourced_input` reads its primary input
plus the terminal stage's `ConceptualSpace.event` (lifted via the
C-tier codebook reverse), averaged when shape-compatible. Bivector-
gated: if the C-tier codebook is not present or the event's last
dim isn't 2, the loopback no-ops and P uses its primary input alone.

### Pipeline as a unit, two-tier reset

`runBatch` is a pure compute brick: forward $\to$ loss $\to$ backward $\to$
optimizer.step. It does **not** decide when to reset per-row state, does
**not** consume `_end_of_stream` for control flow, and (after §6
vectorization) does **not** issue any GPU$\to$host sync inside the brick.

Reset lives in `runEpoch`. The same loop drives both byte cursor (AR text
byte) and trial cursor (non-AR); `next_tick` is universal dispatch:

```
while not ds.all_done():
    inp, out, hard_eos = ds.next_tick()              # 3-tuple, host-side
    runBatch(inp, out)                                # compute brick
    flush_word_buffers()                              # materialize subspace.word
    dispatch_per_row_reset(hard_eos)                  # hard resets
    dispatch_soft_reset()                             # grammar <start> reductions
    post_tick_compact()                               # truth_layer.compact
```

For AR text byte, `inp` is a byte slab and `hard_eos[b]` flips True when
row b's cursor exhausts a doc. For non-AR / numeric data, each tick yields
one batch of trials with `hard_eos = [True] * B`.

**Hard reset.** `TheData` walks each document one slab of ≤1024 bytes at a
time. `hard_eos` flips True on cursor exhaustion. Full row-state cascade
fires for that row only; other rows continue mid-document with state
preserved.

**Soft reset.** `Chart.compose` signals when a row's parse reduces to
`<start>`. `wordSpace._sentence_completed` is drained per-tick: re-arms
`_stm_fired[b]`, clears `_last_svo[b*K..]` and parse-stack rows for `b`,
but **preserves discourse history** (discourse accumulates across sentences
within a document and clears only on hard reset).

**No truncation.** Documents longer than `slab_bytes` span multiple ticks;
concatenating per-tick slabs for any row reproduces the original document
byte-exact. `valid_mask: [B, K]` handles partial-fill tails via NULL-padding.

**Compute-brick contract.** No `.item()`, no `.tolist()`, no Python
conditional on a tensor value, no GPU$\to$host copy inside `runBatch`. Two
residual `.tolist()` calls in `Chart._chart_inside` (`best_pair`,
`best_rule_local`) plus a few data-dependent control points produce graph
breaks; deferred to GB10-side capture wiring.

### Three-File Architecture

| File | Contents | Managed by |
|------|----------|-----------|
| **XML config** (e.g. `BasicModel.xml`) | Architecture, hyperparameters | Hand-edited |
| **Embedding artifact** (e.g. `BasicModel.kv`) | Word vectors (codebook) | Phase 1: `embed.py` |
| **Weights checkpoint** (e.g. `BasicModel.ckpt`) | Model layer parameters (excludes embeddings) | Phase 2: `BasicModel.py` |

`save_weights()` filters out embedding parameters (`_emb.weight`). Enables:
retraining embeddings without touching model weights; retraining model with
frozen embeddings; swapping codebooks between models with same architecture.

---

## Language System

The chart-driven grammar runs at the S tier. SymbolicSpace maps continuous
concept activations to a discrete bivector codebook via the C → S
SigmaLayer + Codebook snap, and `WordSpace` (the grammar host) attaches a
`SyntacticLayer` to S that dispatches the chart's per-cell rules over the
S bivector activation.

`SymbolicSpace` plays the role of the architecture's **calculator**: the
chart at S reads operand cells, invokes the grammar layer's compose /
forward / reverse, and writes the result back. The dispatch is reactive
to the chart's rule queue rather than running unconditionally.

All compositional operations live on the S tier over the bivector-shaped
SymbolicSubSpace. The S-tier productions include:

- **Codebook-bivector unary / binary operators** (geometric, propositional):
  `not(S)`, `non(S)`, `intersection(S, S)`, `union(S, S)`, `conjunction`,
  `disjunction`, `swap`, `copy`, `true`, `false`.
- **Mereological operators**: `part(S, S)`, `equals(S, S)`, `query(S, S)`.
  All three are now pure-geometric — the `MereologicalTree` sidecar that
  formerly stored explicit parent / equality links has been retired in
  favour of clipped-cosine parthood on the bivector codebook. See
  [Mereology.md](Mereology.md).
- **Bivector lift / lower** (`lift(VP, NP)` / `lower(VP, NP)`): these
  factor the substrate's sigma / pi via an elementwise gate at C-tier
  whose values come from VP's symbolic codebook row. The same shared
  `L · U` basis on the substrate sigma (or pi) is reused across every
  VP; the per-call gate `VP_c * NP_c` (elementwise multiplicative) is
  what makes different VPs produce different transformations. See
  [Spaces.md](Spaces.md#liftlower-factorization) and
  [Layers.md](Layers.md#liftlayer--lowerlayer).

Parthood (`part`) is the **fundamental** mereological operation, realized
as clipped cosine projection on the bivector symbol subspace. The full
suite (`whole`, `equal`, `overlap`, `underlap`, `boundary`) composes
through `part` on `Basis`. `equals(S, S)` is propositional identity on S;
delegates to `Basis.equal`.

### Short-Term Memory on ConceptualSpace

`ConceptualSpace.stm` (an instance of `ShortTermMemory`) is a per-batch
stack of unquantized C-tier "ideas" — the continuous compositions that
will accumulate as the (future) serial / shift-reduce parser processes
words. Capacity defaults to 9 (the upper bound of the classical 7±2
linguistic working-memory limit) and is configurable via
`<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>`.
The STM is cleared on hard `Reset` (sentence boundary) and survives
soft reset. The current batched-CKY chart does not yet consume the STM;
this is the structural slot the deferred serial parser will fill. See
[Spaces.md](Spaces.md#shorttermmemory).

### Per-word operational flow (planned, for the deferred serial parser)

> See [`doc/plans/2026-05-12-serial-parser-handoff.md`](plans/2026-05-12-serial-parser-handoff.md)
> for the full handoff spec — motivation, resolved design decisions
> (Q1-Q7: reduce policy, dispatch ordering, truth attachment,
> lift/lower invocation, backward-compat strategy, per-word iteration
> count, operation-by-tier classification), implementation outline,
> and files+line-numbers to read in Phase 1.

The deferred serial parser will require each word to traverse a
per-word round trip through the full pipeline, so that a single word
ends up on the STM as a single post-quantized idea. The shape:

```
byte stream  →  P (BPE lex + per-percept features)
             →  C (project lexed word onto concept space)
             →  S (codebook snap = word identity + POS)
             →  C (S→C reverse, the unquantized "idea")
             →  ConceptualSpace.stm.push(idea)
```

This per-word cadence is **different from** today's per-stage
(symbolic) and per-forward-call (subsymbolic) cadences. The parser
will drive its own per-word forward, likely leveraging the existing
`serial_mode` warm-cache machinery on PerceptualSpace which already
slides one position at a time through the codebook lookup.

**POS rides the codebook for free.** The SymbolicSpace bivector
codebook carries two POS-bearing fields per atom: `category_ids: [V]`
(hard POS tag — one of the grammar's nonterminals, e.g. NP / VP / N /
V / ADJ) and `category_logits: [V, C]` (learnable soft POS distribution
per atom, EMA-updated by the chart's `_apply_codebook_pos_seed`
mechanism in [Language.py:2638](../bin/Language.py)). So per-word snap
returns `(word_id, POS)` simultaneously — no separate POS tagger
needed. The parser uses POS for typing reduce candidates (NP + VP → S,
etc.); POS is *learned through parsing* alongside the codebook.

See [Logic.md](Logic.md), [Mereology.md](Mereology.md), and
[Language.md](Language.md).

**Shamatha Speech target.** Planned narrow grammar for one-pointed object
speech: complete DNF over active percepts, permitting each `conjunction` /
`disjunction` only when operands' `where()` supports are connected and
`when()` supports are continuous. See
[Language.md](Language.md#shamatha-speech-mode).

---

## Sigma and Pi Layers

For weight matrix $W \in \mathbb{R}^{m \times n}$ and input $x \in
\mathbb{R}^n$:

Sigma layer:

$$y_j = W x + b = b_j + \sum_{i=1}^{n} W_{ji} x_i$$

Pi layer (log-space linear):

$$s_i = \log\!\frac{1 + x_i}{1 - x_i} = 2\,\mathrm{atanh}(x_i)$$
$$z_j = \sum_i W_{ji}\, s_i + b_j$$
$$y_j = \frac{e^{z_j} - 1}{e^{z_j} + 1} = \tanh(z_j / 2)$$

Forward maps $[-1,1] \to (0,\infty)$ via `_to_mult`, log, linear, exp,
`_from_mult`. Domain and range both $[-1,1]$. Reverse inverts each step:
`_to_mult(y)`, log, $W^{-1}(z - b)$, exp, `_from_mult`.

**Motivation.** The classical product form $y_j = b_j \prod_i (1 + W_{ji}
x_i)$ becomes, after taking logs, a sum. The code moves into a
log-multiplicative domain via atanh, performs a linear op there, returns
via tanh. The atanh transform stretches values near $\pm 1$ toward infinity,
making the layer sensitive to strong activations.

**Monotonicity of the bivector chain.** When `<bivectorOutput>true` is
configured on P/C/S, every activation lives on the non-negative paired-index
cone `[0, 1]^{2K}`. Pi/Sigma select `NonNegativeInvertibleLinearLayer` (or
`NonNegativeLinearLayer`) under `monotonic=True`, giving $W \geq 0$. Positive
matrices are the canonical monotone operators on the positive cone, so
every lift / lower preserves parthood pole-by-pole. PerceptualSpace joins via
the Q2 promotion `(aP, aN) = (max(0, x), max(0, -x))`. See
[Spaces.md](Spaces.md#monotonicity-of-the-lift--lower-chain).

---

## Dimensionality Constraints

- Input layer output dim = perceptual layer output dim (conceptual operates
  on both).
- Symbolic layer input dim = perceptual layer input dim (both operate on
  conceptual output).
- Output layer input dim = sum of symbolic layers' output dims.

---

## Invertible Linear Layer (LDU)

Factors $W = L \cdot D_{\text{embed}} \cdot U$:

- $L$: unit lower-triangular ($nIn \times nIn$, diagonal = 1).
- **D**: diagonal vector of length `rank = min(nIn, nOut)`, embedded into
  $[nIn, nOut]$ by zero-padding.
- $U$: unit upper-triangular ($nOut \times nOut$).

**Exact inverse via triangular solves:** $W^{-1} = U^{-1} \cdot D^{-1} \cdot
L^{-1}$. Each factor inverted by `torch.linalg.solve_triangular`. No SVD;
inverse exact when all D entries are nonzero. Parameter count: $nIn^2 +
\mathrm{rank} + nOut^2$. Initialized at $L = I, d = 1, U = I$ (identity).

`naive=False` (default) applies L/D/U sequentially without materialising
`W_eff` as a full matrix. `naive=True` materialises `W_eff` and its inverse.

Ergodic noise injection at the factor level, plus the `stable=True` clamp
and the noise lifecycle, are documented in [Ergodic.md](Ergodic.md).

---

## Ergodic Exploration

See [Ergodic.md](Ergodic.md).
