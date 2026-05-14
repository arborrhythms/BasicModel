# Architecture

## Overview

BasicModel is a bidirectional neural architecture organized as a pipeline of five
**spaces** plus a grammar host (`WordSpace`), each implementing a distinct
representational transformation:

```
Forward:  InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> OutputSpace
Reverse:  OutputSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace

Feedback loops (cross-stage / cross-call):
    SymbolicSpace ---> ConceptualSpace   (S->C symbolic loopback; per stage at order >= 1)
    ConceptualSpace ---> PerceptualSpace (C->P subsymbolic loopback; cross-forward)
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
plus the new $C \to P$ feedback loop; syntax / grammar dispatch lives on
`WordSpace` and is attached to `SymbolicSpace` (the canonical grammar host).
The `MereologicalTree` sidecar that backed `part` / `equals` / `query` is
also retired --- those operations are now pure-geometric clipped-cosine
projections on the SymbolicSpace bivector codebook.

### Spaces

| Space | Role | Layer Type | Notes |
|-------|------|-----------|-------|
| **InputSpace** | Lifts raw data into working dimensionality | LiftingLayer | nActive, nDim, nVectors |
| **PerceptualSpace** | Per-percept aggregation; subsymbolic substrate | Owns **`pi_input`** (`input_dim -> percept_dim`) and **`pi_concept`** (`concept_dim -> percept_dim`). Both fire unconditionally each forward; outputs summed.  $C \to P$ feedback flows through `pi_concept`. | nActive, nDim, nVectors, invertible, bivectorOutput |
| **ConceptualSpace** | Additive concept abstraction ($P \to C$); host of `ShortTermMemory` | Owns **`sigma_percept`** (`percept_dim -> concept_dim`) only.  SS feedback has **no default fold** --- only grammar ops at S. | nActive, nDim, nVectors, invertible, bivectorOutput, stmCapacity |
| **SymbolicSpace** | Discrete activation bottleneck (C $\to$ S) + Codebook + **grammar's calculator** | No default sigma or pi.  Grammar ops only (`intersection`, `union`, `lift`, `lower`, ...) dispatched by SyntacticLayer.  Codebook plays the unified role of *codebook + lexicon-reference + meronymic-tree* --- three jobs on one structure (geometric). | nActive, nVectors, codebook, bivectorOutput |
| **OutputSpace** | Final prediction | LinearLayer | nActive, nDim, nVectors |

The cross-space fold contract --- the canonical composition is:

```
C  =  sigma_percept(  pi_input(IS)  +  pi_concept(C_prev)  )
```

`pi_input` is the input boundary fold; `pi_concept` carries the
subsymbolic $C \to P$ feedback.  Both fire on every PerceptualSpace forward
and are summed (no /2 averaging --- the previous `(primary + c_event) / 2`
in `_sourced_input` is retired). `sigma_percept` then folds the
combined P-state into a C-state.  See [Spaces.md Section "Sigma / Pi
ownership"](Spaces.md#sigma--pi-ownership-2026-05-13-rebalance)
for the cognitive rationale (lift vs lower as rule-id annotations
over a shared composition primitive) and the grammar-XML migration
table.  See [Logic.md Section 8](Logic.md) for the algebraic constraints on
sigma/pi.

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
   weights --- `forward()` on one, `reverse()` on the other. Avoids the
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

- **`nOutputSymbols`** `= OutputSpace.nActive` --- fed to OutputSpace for prediction
- **`nReconSymbols`** `= nSymbols - nOutputSymbols` --- carried in `end_state`
  for reconstruction

A skip connection through the symbolic bottleneck. Reconstruction symbols
receive gradient only from reconstruction loss.

### Single Optimizer with Overlapping Weight Spaces

The forward and reverse passes share a **single Adam optimizer** that
minimizes the combined loss. Forward and reverse weight spaces **partially
overlap** --- neither disjoint (allowing independent optimizers) nor identical
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

Two orthogonal mode dimensions plus the two feedback loops.

**Serial mode** (`BASICMODEL_DEVICE` / runtime flag `serial_mode`):
Runtime fast path for streaming / autoregressive contexts.
`PerceptualSpace` / `ConceptualSpace` may use slide-and-recompute with the
previous step's per-cell warm cache (`subspace.serial_cache`). Cache is
keyed on owner-Space id, cleared on hard `Reset`. (Distinct from the
**serial / shift-reduce parser** --- a separate deferred refactor; see
[`doc/plans/`](plans/).)

**The two feedback loops**:
The architecture now exposes two concurrent recurrent paths between the
spaces. Their iteration cadences differ --- they do NOT both follow
`conceptualOrder`.

| Loop | Direction | Iterations per forward call | Cadence governed by |
|---|---|---|---|
| **Symbolic loopback** | $S \to C$ | `T = conceptualOrder` (each stage $t \ge 1$ reads stage `t-1`'s S event) | `<conceptualOrder>T</conceptualOrder>` directly |
| **Subsymbolic loopback (new)** | $C \to P$ | **Exactly 1** per forward call (stem `P` reads `conceptualSpaces[-1]`.event before any C stage fires) | Cross-forward; not per-stage |

Conceptual input sourcing: each stage's `ConceptualSpace.forward`
combines two sources via `_sourced_input` --- its own perceptual primary
(stem-cached) AND the previous stage's `SymbolicSpace.event`, lifted
from the bivector handoff `[B, V_S, 2]` back to concept content
`[B, V, concept_dim]` via `Codebook.project_reverse`. The combination
is additive after the bivector lift. Stage 0 cold-starts (no S
sibling event yet); the previous-stage symbolic loopback fires from
stage 1 onwards.

The new $C \to P$ loopback at PerceptualSpace works the same way (mirror of
the C-side mechanism): P's `_sourced_input` reads its primary input
plus the terminal stage's `ConceptualSpace.event` (lifted via the
C-tier codebook reverse), averaged when shape-compatible. Bivector-
gated: if the C-tier codebook is not present or the event's last
dim isn't 2, the loopback no-ops and P uses its primary input alone.

### Pipeline as a unit, two-tier reset

`runBatch` is a pure compute brick: forward $\to$ loss $\to$ backward $\to$
optimizer.step. It does **not** decide when to reset per-row state, does
**not** consume `_end_of_stream` for control flow, and (after Section 6
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

**Hard reset.** `TheData` walks each document one slab of $\le$1024 bytes at a
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

### Two-File Architecture

| File | Contents | Managed by |
|------|----------|-----------|
| **XML config** (e.g. `BasicModel.xml`) | Architecture, hyperparameters | Hand-edited |
| **Weights checkpoint** (e.g. `BasicModel.ckpt`) | Full integrated bundle: model parameters, register-buffer state, embedding vectors, vocabulary mappings, BPE codebook | Training (`save_weights`) |

The 2026-05-12 *integrated-weights* refactor retired the separate
`.kv` embedding artifact: embeddings, vocabulary mappings, and the
BPE codebook now ride inside the single `.ckpt` bundle alongside the
model's other parameters. The bundle layout is:

* `state_dict`: every `nn.Parameter` and `register_buffer` in the
  module tree (model weights, `wv._vectors`, `TruthLayer.truths`,
  etc.) --- serialised by the normal PyTorch path.
* `vocab_extras`: the WordVectors Python-side mappings that don't
  live in `state_dict` (`index_to_key`, `counts`, `total_count`).
* `bpe_extras`: the ChunkLayer's pure-Python state (merges list,
  vocab dict, `id_to_bytes`, growth cursors). Required because
  `ChunkLayer` stores its merge table as Python dicts/lists, not
  tensors.

`bin/embed.py` still produces standalone `.kv` artifacts for
CBOW/SBOW *pre-training* studies, but those artifacts are no longer
part of the runtime artifact set. Cold-start training initialises
the vocabulary and BPE codebook from scratch and learns them
end-to-end alongside the model weights.

---

## Language System

The chart-driven grammar runs at the S tier. SymbolicSpace maps continuous
concept activations to a discrete bivector codebook via the C $\to$ S
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
  All three are now pure-geometric --- the `MereologicalTree` sidecar that
  formerly stored explicit parent / equality links has been retired in
  favour of clipped-cosine parthood on the bivector codebook. See
  [Mereology.md](Mereology.md).
- **Bivector lift / lower** (`lift(VP, NP)` / `lower(VP, NP)`): these
  factor the substrate's sigma / pi via an elementwise gate at C-tier
  whose values come from VP's symbolic codebook row. The same shared
  $L \cdot U$ basis on the substrate sigma (or pi) is reused across every
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
stack of unquantized C-tier "ideas" --- the continuous compositions that
accumulate as the per-word stem processes words. Capacity auto-sizes to
`<WordSpace><wMax>` (the chart's sentence-length bound), so each word
fills a slot before the chart fires at C-tier in the body; an explicit
`<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>` still
overrides for subsymbolic configs that want a larger buffer. The STM
is cleared on hard `Reset` (sentence boundary) and survives soft reset.
The body's chart fires over `stm.snapshot()` at every stage. See
[Spaces.md](Spaces.md#shorttermmemory).

### Per-word operational flow

Each word traverses a per-word round trip through the full pipeline,
so that a single word ends up on the STM as a single post-quantized
idea. The shape:

```
byte stream  ->  P (BPE lex + per-percept features)
             ->  C (project lexed word onto concept space)
             ->  S (codebook snap = word identity + POS)
             ->  C (S->C reverse, the unquantized "idea")
             ->  ConceptualSpace.stm.push(idea)
```

This per-word cadence runs **inside the stem** before the body's
per-stage chart firing. The body then iterates over its `body_stages`
ModuleList, calling `_chart_compose_at_C` over `stm.snapshot()` at
every stage so the chart's per-rule selections shape SymbolicSpace
dispatch.  The reverse mirror (`_chart_generate_from_stm`) fires at
the symmetric C-tier point inside the reverse pipeline.

**POS rides the codebook for free.** The SymbolicSpace bivector
codebook carries two POS-bearing fields per atom: `category_ids: [V]`
(hard POS tag --- one of the grammar's nonterminals, e.g. NP / VP / N /
V / ADJ) and `category_logits: [V, C]` (learnable soft POS distribution
per atom, EMA-updated by the chart's `_apply_codebook_pos_seed`
mechanism in [Language.py:2638](../bin/Language.py)). So per-word snap
returns `(word_id, POS)` simultaneously --- no separate POS tagger
needed. The parser uses POS for typing reduce candidates ($NP + VP \to S$,
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

---

## Sentence-level AR (`InterSentenceLayer`)

Within-sentence training is IR-only (BERT-style masked-LM at the
P-tier; see `doc/Spaces.md` Section "Within-sentence AR retirement"). The
**autoregressive** signal in this architecture lives one scale up:
between sentences, on a per-sentence representation `s_t`. That's the
job of `InterSentenceLayer` (alias `wordSpace.discourse`).

### Sentence representation

`s_t` is the **root S-tier slot** of the body's final stage: the
single vector the start-symbol reduction wrote into. The chart's
parse trace already commits to this slot at sentence end; the layer
pools `[B, N, D] -> [B, D]` by taking row 0 (root). Width is
`sentence_dim = n_dim` (one vector per row), **not** the full
`n_symbols * n_dim` flatten that the pre-2026-05-14 contrastive layer
used --- that broader rep would have blown the predictor's Linear past
the allocator budget on MM_5M_bivector-scale configs.

### ARMA(p, q) predictor

`InterSentenceLayer` runs an autoregressive moving-average predictor:

```
s_hat_t = predictor(s_{t-1..t-p}, e_{t-1..t-q})
e_t     = s_t - s_hat_t
loss    = MSE(s_hat_t, s_t)        # accumulated per batch
```

- `p` = AR lag count (default 5) --- last p sentence reps.
- `q` = MA lag count (default 2) --- last q prediction errors.
- `predictor` = `nn.Sequential(Linear(p*D + q*D, H), Tanh,
  Linear(H, D))`, with `H = min(1024, 2*sentence_dim)`.

The MA term lets the predictor correct for systematic bias in the AR
extrapolation: if the AR model consistently under-predicts the
sentence rep, the residual `e_t` carries that signal forward.

Buffers (per row, non-persistent):

- `_s_history`: `[B, p, sentence_dim]` ring of last p sentence reps
  (most recent at index `-1`).
- `_e_history`: `[B, q, sentence_dim]` ring of last q residuals.
- `_s_count` / `_e_count`: `[B]` long, fill levels (cap at p / q).

`ensure_batch(B)` resizes these on cascade from
`WordSpace.ensure_batch`; `Reset()` clears them on hard / discourse
boundary. Default behaviour is to **not** auto-reset across document
boundaries --- the AR lags carry information through discourse
continuity unless the caller explicitly calls `Reset`.

### Wiring into the training loop

1. After the body finishes (sentence end), `_forward_per_stage`
   stashes the S-tier event on `_current_discourse_s`.
2. In `runBatch`, when training, `discourse.observe(s_tensor)`:
   - Pools `s_t = sigma_S(s_tensor[:, 0, :])`.
   - Computes `s_hat_t = predictor(_s_history, _e_history)`.
   - Returns `MSE(s_hat_t, s_t)` (None on the first call per row
     when the ring is empty).
   - Computes `e_t = s_t - s_hat_t`, pushes both into the rings.
3. `runBatch` adds the loss to `TheError` under category
   `"discourse"` with weight `armaScale` (training XSD knob, default
   0.1).

### Inference (chat-loop seeding)

`BasicModel.generate_sentence(seed_text)`:

1. Calls `discourse.predict_next()` for the ARMA-predicted
   sentence-rep prior `s_hat_{t+1}`.
2. Lifts it through `discourse.cast` to `concept_dim` and stages on
   `ConceptualSpace._c_prior`.
3. Runs the IR forward --- the body's first sigma_percept output gets
   `_c_prior` summed in as a sentence-level conditioning bias before
   the codebook lookup. Cleared after the forward consumes it.
4. The post-body perceptual event is decoded by nearest-neighbour
   against the perceptual codebook, producing the
   `(slot, original, predicted)` triples for the seed text's masked
   positions.
5. Commits the produced sentence's S-tier root to the ARMA ring via
   `discourse.observe(s_tensor)`.

The IR head plays no role at inference --- the prediction lives at the
masked P-tier positions, decoded against the (frozen) perceptual
codebook.

### Configuration

| XSD knob | Section | Default | Notes |
|---|---|---|---|
| `<armaP>` | `<WordSpace>` | 5 | AR lag count |
| `<armaQ>` | `<WordSpace>` | 2 | MA lag count |
| `<armaHiddenDim>` | `<WordSpace>` | `2*sentence_dim` (cap 1024) | predictor hidden width |
| `<armaScale>` | `<architecture><training>` | 0.1 | ARMA loss weight added to `TheError` |
| `<sentencePrediction>` | `<architecture><training>` | false | Gates `InterSentenceLayer` construction |

The retired pre-2026-05-14 knobs (`<sentenceContextWindow>`,
`<sentenceCentroidHistory>`, `<sentenceLambda>`,
`<sentencePredictionScale>`, `<sentencePredictiveScale>`,
`<sentenceContrastiveScale>`) shaped the legacy contrastive cosine
machinery (recent-centroid attraction + older-centroid repulsion).
They are not parsed; configs that still set them are tolerated
silently.
