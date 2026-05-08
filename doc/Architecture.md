# Architecture

> **Stage 2 in progress.** A wide ConceptualSpace codebook (nVectors $\gg$ nOutput), a shared `SparsityRegLayer` attached to Perceptual / Conceptual / Symbolic spaces, a three-way PerceptualSpace chunking switch (`raw | bpe | lexicon`), and a tri-state `useGrammar` config (`all | thoughtFree | none`) are landing progressively. Design: [`specs/2026-04-16-stage2-wide-conceptual-codebook-design.md`](specs/2026-04-16-stage2-wide-conceptual-codebook-design.md). Plan: [`plans/2026-04-16-stage2-wide-conceptual-codebook-plan.md`](plans/2026-04-16-stage2-wide-conceptual-codebook-plan.md). Phases 1-5.1/5.2 are in; Task 5.3 (merged mode-blind outer forward loop) is deferred -- WordSpace's actual shape is a per-tier dispatcher rather than a single `forward(ss)`, so the outer-loop refactor needs a design addendum before implementation.

## Overview

BasicModel is a bidirectional neural architecture organized as a pipeline of six
**spaces**, each implementing a distinct representational transformation:

```
Forward:  InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace -> SyntacticSpace -> OutputSpace
Reverse:  OutputSpace -> SyntacticSpace -> SymbolicSpace -> ConceptualSpace -> PerceptualSpace -> InputSpace
```

The forward pass transforms raw input into predictions. The reverse pass
reconstructs the original input from the symbolic representation. Both directions
are trained simultaneously with a single optimizer minimizing a combined loss:

```
totalLoss = (1 - reconRatio) * outputLoss + reconRatio * reconstructionLoss
```

### Spaces

| Space | Role | Layer Type | Parameters |
|-------|------|-----------|------------|
| **InputSpace** | Lifts raw data into working dimensionality | LiftingLayer | nActive, nDim, nVectors |
| **PerceptualSpace** | Per-percept aggregation | SigmaLayer (`self.sigma`, P -> sub-percept; dormant pending sub-perceptual structure) | nActive, nDim, nVectors, invertible |
| **ConceptualSpace** | Multiplicative abstraction (P -> C) | PiLayer (`self.pi`, P -> C, with `forwardPi` / `reversePi` pointer aliases hiding the one-or-two-layer split) | nActive, nDim, nVectors, invertible |
| **SymbolicSpace** | Discrete activation bottleneck (C -> S) + Codebook | SigmaLayer (`self.sigma`, C -> S, with `forwardSigma` / `reverseSigma` pointer aliases) + Codebook | nActive, nVectors, passThrough, codebook |
| **SyntacticSpace** | Binary derivation tree (CNF grammar) | Grammar + WordEncoding | nActive, nDim, nVectors |
| **OutputSpace** | Final prediction | LinearLayer | nActive, nDim, nVectors |

Each level-crossing space owns one bidirectional layer that handles
both forward and reverse via its own self-inverse:
`ConceptualSpace.pi` does P->C forward and C->P reverse;
`SymbolicSpace.sigma` does C->S forward and S->C reverse.  Pi and
Sigma have independent weights -- there is no shared layer across the
boundary.  See [doc/Logic.md §8](Logic.md) and [doc/Spaces.md](Spaces.md)
for the full framing.

Dimensions (`nDim`) are not passed to Space constructors; each subclass reads its
content dimensionality from `TheObjectEncoding` (e.g., `TheObjectEncoding.perceptDim`
for PerceptualSpace).  Codebook sizes (`nVectors`) are likewise stored on
`TheObjectEncoding` and may differ from the active count (`nActive`); the factory
validates `nVectors >= nActive` for every space.

![MM_5M Hierarchical Progressive Bottleneck](diagrams/mm5m_architecture.svg)

Layer selection depends on `<reconstruct>` and `invertible`:

1. **`reconstruct=NONE`**: Use non-invertible layers (`PiLayer`, `SigmaLayer`) for
   the forward pass only. No reverse pipeline is created.
2. **`reconstruct=<any>` + `invertible`**: A single invertible layer
   (`PiLayer(invertible=True)`, `SigmaLayer(invertible=True)`) serves both
   directions, sharing weights.
3. **`reconstruct=<any>` + not `invertible`**: Two layers with separate weights --
   call `forward()` on one and `reverse()` on the other.  This avoids the
   expressivity limitation where a non-invertible layer's forward pass cannot
   represent the inverse of another (e.g., PiLayer's product structure is not
   closed under inversion).  **Note:** The reverse path uses a matrix
   pseudoinverse (`pinv`) which may be numerically unstable due to SVD
   convergence issues.  Setting `<invertible>true</invertible>` avoids this
   by using shared-weight inversion instead.

### Reconstruction Symbols

The symbolic bottleneck can lose information needed for reconstruction. For example,
XOR maps 2 inputs to 1 output, but `XOR(0,0)=0` and `XOR(1,1)=0` are distinct inputs
producing the same output. A single output symbol cannot reconstruct which input was
presented.

To solve this, the `nSymbols` produced by SymbolicSpace are split:

- **`nOutputSymbols`** `= OutputSpace.nActive` -- fed to OutputSpace for prediction
- **`nReconSymbols`** `= nSymbols - nOutputSymbols` -- carried in `end_state` for reconstruction

This is essentially a skip connection through the symbolic bottleneck: an extra channel
that preserves information lost in the output projection. Reconstruction symbols receive
gradient only from reconstruction loss, never from output loss.

For XOR with `nSymbols=3`, `nOutput=1`: symbol 0 predicts the output; symbols 1-2 carry
enough information to distinguish all 4 input patterns, enabling perfect reconstruction.

### Single Optimizer with Overlapping Weight Spaces

A critical design choice: the forward and reverse passes share a **single Adam
optimizer** that minimizes a combined loss:

```
totalLoss = (1 - reconRatio) * outputLoss + reconRatio * reconstructionLoss
```

This works because the forward and reverse weight spaces **partially overlap**.
They are neither disjoint (which would allow independent optimizers) nor identical
(which would create destructive interference). Instead, certain layers share
weights between directions (e.g., shared embeddings, the symbolic bottleneck
itself) while others are direction-specific (e.g., `pi1`/`pi2`, `sigma1`/`sigma2`,
`linear1`/`linear2`).

The partial overlap means:
- **Shared weights** receive gradient from *both* output and reconstruction loss,
  learning representations useful in both directions simultaneously.
- **Direction-specific weights** receive gradient from only their respective loss,
  specializing freely without interference.
- **No ping-pong**: separate optimizers on overlapping parameters would pull weights
  in alternating, conflicting directions each step. A single optimizer sees the
  combined gradient and finds a consistent descent direction.

This resolves the fundamental tension between forward prediction and backward
reconstruction in bidirectional architectures -- a problem first identified in
A.M. Rogers, T.T. Shannon, and G.G. Lendaris, "A comparison of DHP based
antecedent parameter tuning strategies for fuzzy control," *Proceedings Joint
9th IFSA World Congress and 20th NAFIPS International Conference*, Vancouver,
BC, Canada, 2001, pp. 580-585, doi:
[10.1109/NAFIPS.2001.944317](https://ieeexplore.ieee.org/document/944317).
See also `doc/research/` for a local copy.

When `invertible=true`, the overlap is total: a single invertible layer serves both
directions, and its weights receive the full combined gradient. The single-optimizer
approach handles this case naturally.

### Training Loop

Training uses the single Adam optimizer with persistent state (momentum/variance
accumulate across epochs):

1. Forward pass: input $\rightarrow$ prediction + `end_state`
2. Compute `outputLoss` from prediction vs. target
3. Reverse pass: `end_state` $\rightarrow$ reconstructed input
4. Compute `reconstructionLoss` from reconstruction vs. original input
5. Backpropagate `totalLoss = (1 - reconRatio) * outputLoss + reconRatio * reconstructionLoss`
6. If ergodic: run `paramUpdate()` (gradient energy sensor updates alpha)
7. Optimizer step (embedding params excluded when `trainEmbedding` is `NONE`, `CBOW`, or `SBOW`)
8. If `trainEmbedding` is `CBOW`, `SBOW`, or `BOTH`: run embedding update step on same sentence

Alpha annealing (ergodic mode): the code-level exploration parameter starts at
`1.0` (full exploration) and decays to `0.0` (full exploitation) within the first
5% of epochs via `alpha = max(0, 1 - epoch / warmup)` where
`warmup = numEpochs // 20`.  Note: the code convention (`alpha=1` means explore)
is the inverse of the ergodic math convention (`alpha=0` means explore); layers
translate between them internally.

When `<useButterflies>true</useButterflies>` (and `conceptualOrder > 1`),
the Sigma-Pi loop switches from a flat concatenation-based cycle to a
pairwise butterfly architecture with per-level butterfly-mode Pi/Sigma layers
and a geometrically partitioned symbol dimension.  `<useGrammar>true</useGrammar>`
(legacy spelling for `useGrammar="all"` in `<WordSpace>`) selects a
grammar-directed progressive-bottleneck variant instead.  Full grammar
mode and butterflies are mutually exclusive -- butterfly permutations
fight constituency structure.  The planned `shamathaSpeech` mode is a
narrow DNF-object grammar with contiguity checks, not the full
constituency path.  See [Reasoning.md](Reasoning.md) Section
Architecture Modes for the full comparison.

See [Params.md](Params.md) for all XML configuration parameters.
See [Training.md](Training.md) for embedding pretraining, SBOW, masked prediction
modes, and the `<trainEmbedding>` control.

### Modes of operation

The architecture has three orthogonal mode dimensions plus a
subsymbolic-enable flag, set independently in the model XML.

**Butterfly mode** (`<useButterflies>true</useButterflies>`):

  Pairwise sigma/pi with N-halving across `<conceptualOrder>` stages.
  Each per-stage `ConceptualSpace` / `SymbolicSpace` instance is built
  with a butterfly-mode SigmaLayer / PiLayer that packs `[B, N, D]` to
  `[B, N/2, 2*D]` before the inner LDU and unpacks after. Slot count
  halves per stage; per-slot dim doubles; total per-stage volume
  preserved. Validated by
  `nPercepts × state_dim == nSymbols × symbol_width` at
  `ModelFactory.validate_config`. Used by MM_xor, MM_5M, MM_400M.
  Without butterflies, every stage shares the same shape and no
  halving happens (the "plain" / `useGrammar=all` paths).
  Reference: `bin/Models.py` `_build_staged_pipeline` butterfly
  branch; `bin/Layers.py` `ButterflyLayer._butterfly_pack` /
  `_butterfly_unpack` / `_butterfly_merge` / `_butterfly_unmerge`.

**Serial mode** (`BASICMODEL_DEVICE` / runtime flag `serial_mode`):

  A runtime fast path for streaming / autoregressive contexts.
  `PerceptualSpace` / `ConceptualSpace` may use the slide-and-recompute
  path where the previous step's per-cell warm cache (kept on
  `subspace.serial_cache`) is reused for cells that haven't shifted.
  Independent of butterfly / parallel mode -- gated only by the
  runtime flag. The cache is keyed on the owner-Space id, cleared on
  hard `Reset`, rebuilt cheaply on the next forward.
  Reference: `Space.serial_mode` flag (set by
  `BaseModel.create_from_config`), `SubSpace.serial_cache` dict, and
  the `test_serial_mode_*.py` tests.

**Parallel vs Grammar mode**
(`<architecture><mode>grammar|parallel</mode>`):

  Mutually exclusive Phase-1 modes:
  - **`grammar` mode**: `SymbolicSpace` is active;
    `SubsymbolicSpace.held_at_zero = True`. The symbolic re-entrant
    loop fires; the subsymbolic event tensor is held at zero and
    contributes nothing to the next conceptual order's combined
    input.
  - **`parallel` mode**: `SubsymbolicSpace` is active;
    `SymbolicSpace.held_at_zero = True`. The subsymbolic / felt-
    sense re-entrant loop fires; the symbolic event tensor is
    zeroed.

  Both modes wire a `SubsymbolicSpace` parallel to `SymbolicSpace`;
  only one is "running" per pass. The other's event tensor is summed
  elementwise into the next conceptual order's combined input (zeros
  are identity under the additive sum, so the held-at-zero side
  contributes nothing). Reference: `held_at_zero` attribute on
  `SymbolicSpace` and `SubsymbolicSpace`; `BaseModel.__init__`
  mode-dispatch around the subsymbolic enable check.

**Subsymbolic enabled** (`<subsymbolicEnabled>true</subsymbolicEnabled>`,
default false):

  Independent of the mode flag — controls whether `SubsymbolicSpace`
  is *constructed* at all. When false, only `SymbolicSpace` is built;
  no parallel re-entrant loop; the per-stage ConceptualSpace's
  PiLayer is not widened. When true, both spaces exist and the mode
  flag determines which one's event contributes to the next
  conceptual order.

### Pipeline as a unit, two-tier reset

> *Post the rolling-cursor handoff (`plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md`)
> and the brick-vectorization handoff
> (`plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md`).*

`runBatch` is a pure compute brick: forward → loss → backward →
optimizer.step. It does **not** decide when to reset per-row state, does
**not** consume `_end_of_stream` for control flow, and (after the §6
vectorization landed) does **not** issue any GPU→host sync inside the
brick body.

Reset lives in the outer doc-streaming loop in `runEpoch`. The same
loop drives both the byte cursor (AR text byte) and the trial cursor
(non-AR / numeric / non-byte) -- there is no longer a separate
DataLoader-iteration branch; `next_tick` is the universal dispatch
(§8e of the brick-vectorization handoff):

```
while not ds.all_done():
    inp, out, hard_eos = ds.next_tick()              # 3-tuple, host-side
    runBatch(inp, out)                                # the compute brick
    flush_word_buffers()                              # §6c materialize subspace.word
    dispatch_per_row_reset(hard_eos)                  # hard resets
    dispatch_soft_reset()                             # grammar <start> reductions
    post_tick_compact()                               # truth_layer.compact
```

For AR text byte, `inp` is a byte slab and `hard_eos[b]` flips True
when row b's cursor exhausts a doc. For non-AR / numeric data the
cursor aligns with the trial: each tick yields one batch of trials
with `hard_eos = [True] * B` (every row ends its trial each tick).

**Hard reset.** `TheData` walks each row's document one slab of up to 1024 bytes at
a time. A row's `hard_eos` flips True the tick its cursor exhausts the
current document. The full row-state cascade fires for that row only.
Other rows continue mid-document with state preserved.

**Soft reset.** `Chart.compose` detects when a row's parse derivation
reduces to `<start>` (a new top-level grammar element naming the start
symbol; see [Language.md](Language.md)). The signal accumulates on
`wordSpace._sentence_completed: list[bool]` and is drained per-tick.
A soft reset re-arms `_stm_fired[b]` and clears `_last_svo[b*K..]` and
the parse-stack rows for `b`, but **preserves discourse history** —
discourse accumulates across sentences within a document and clears only
on hard reset.

**No truncation.** A document longer than `slab_bytes` (= 1024 by default)
spans multiple ticks of its row. Concatenating the per-tick slabs for any
row reproduces the original document byte-exact. The `valid_mask: [B, K]`
contract handles partial-fill tails (last slab of a doc shorter than
`slab_bytes`) via the same NULL-padding semantics already in place.

**Compute-brick contract.** No `.item()`, no `.tolist()`, no Python
conditional on a tensor value, no GPU→host copy inside `runBatch`. The
brick-vectorization handoff (§6) made this true:

- §6a removed `stm_residual_microbatch`'s `.item()` early-out (always
  call `disc.predict()`; gate the bias on already-fired rows via
  multiplication).
- §6b dropped the per-cell `should_store` gate from the truth layer;
  `record_batch` stages every cell with a trust score, and post-tick
  `compact()` filters in one host sync outside the brick.
- §6c adds tensor `word_records` / `word_count` buffers to `SubSpace`
  so the chart compose can scatter entries inside the brick;
  `flush_word_buffer` materializes them onto `subspace.word` once per
  tick from the outer loop. The chart compose dispatches one vector
  `add_word` call per depth step (replacing the per-row scalar loop).

CUDA-graph capture of the brick (§7 in the same handoff) is the
remaining piece. Two residual `.tolist()` calls in
`Chart._chart_inside` (`best_pair`, `best_rule_local`) plus a few
`if compat.sum() == 0: break`-style data-dependent control flow
points produce graph breaks; the plan defers handling those to the
GB10-side capture wiring.

### Three-File Architecture

Model behaviour is partitioned across three independent artifacts:

| File | Contents | Managed by |
|------|----------|-----------|
| **XML config** (e.g. `BasicModel.xml`) | Architecture, hyperparameters, paths | Hand-edited |
| **Embedding artifact** (e.g. `BasicModel.kv`) | Word vectors (codebook) | Phase 1: `embed.py` |
| **Weights checkpoint** (e.g. `BasicModel.ckpt`) | Model layer parameters (excludes embeddings) | Phase 2: `BasicModel.py` |

The `save_weights()` method explicitly filters out embedding parameters
(`_emb.weight`) from the checkpoint. Embedding vectors live in the `.kv`
artifact and are loaded separately. This separation allows:

- Retraining embeddings without touching model weights
- Retraining model layers with frozen embeddings
- Swapping codebooks between models that share the same architecture

---

## Language System

The symbolic and syntactic spaces implement a binary deep-structure grammar in
Chomsky Normal Form. SymbolicSpace maps continuous concept activations to a
discrete one-hot encoding via an invertible layer and codebook. SyntacticSpace
generates a derivation tree from the active symbols, stored as word tuples
`(batch, vector, rule)`.

**Unified S-tier grammar (2026-04-19 rewrite).** The previous C/P/S
three-tier partition was collapsed: all compositional operations now live
on the single S tier over a bivector-shaped SymbolicSubSpace. The 17
S-tier productions include the ternary operators (`true(S)`, `false(S)`,
`non(S)`), the what/where/when column selectors, and the mereological
and logical operators (`not(S)`, `part(S, S)`, `intersection(S, S)`,
`union(S, S)`, `equals(S, S)`, `conjunction`, `disjunction`, `swap`,
`query`, `lift`, `lower`).

Parthood (`part`) is the **fundamental** mereological operation, realized
as clipped cosine projection on the bivector symbol subspace. The full
mereological suite (`whole`, `equal`, `overlap`, `underlap`, `boundary`)
composes through `part` on `Basis`. `equals(S, S)` is propositional
identity on S and delegates to `Basis.equal` (mutual parthood).

See [Logic.md](Logic.md) for the parthood formula and suite,
[Mereology.md](Mereology.md) for the five-relations reference and the
`ImpenetrableLayer` regularizer, and [Language.md](Language.md) for the
full grammar, word encoding, and open implementation questions about
differentiable tree structure and rule operations.

**Shamatha Speech target.** A separate narrow grammar mode is planned for
one-pointed object speech: form a complete DNF over the active percepts,
but permit each `conjunction` / `disjunction` only when the operands'
`where()` supports are connected and their `when()` supports are
continuous.  This is not serial cursor mode; it may compose over all
active percepts, then render the resulting object as strict DNF English.
See [Language.md](Language.md#shamatha-speech-mode) and
[plans/2026-04-28-shamatha-speech-contiguity-handoff.md](plans/2026-04-28-shamatha-speech-contiguity-handoff.md).

---

## Sigma and Pi Layers

Given a weight matrix $W \in \mathbb{R}^{m \times n}$ and input vector $x \in \mathbb{R}^n$, the output vector $y \in \mathbb{R}^m$ is computed as:

For the Sigma layer:

$$y_j = W x + b$$
$$y_j = b_j + \sum_{i=1}^{n} W_{ji} x_i$$

For the Pi layer (code implementation -- log-space linear):

$$s_i = \log\!\frac{1 + x_i}{1 - x_i} = 2\,\mathrm{atanh}(x_i)$$
$$z_j = \sum_i W_{ji}\, s_i + b_j$$
$$y_j = \frac{e^{z_j} - 1}{e^{z_j} + 1} = \tanh(z_j / 2)$$

The forward path maps $[-1,1] \to (0,\infty)$ via `_to_mult`, takes the log,
applies a linear transform (InvertibleLinearLayer), exponentiates, and maps
back via `_from_mult`.  Domain and range are both $[-1,1]$.  The reverse
path inverts each step exactly: `_to_mult(y)`, log, $W^{-1}(z - b)$, exp,
`_from_mult`.

**Conceptual motivation.**  The classical product form
$y_j = b_j \prod_i (1 + W_{ji} x_i)$ becomes, after taking logs, a sum
$\log y_j = \log b_j + \sum_i \log(1 + W_{ji} x_i)$.  The code
generalises this idea: it moves into a log-multiplicative domain via atanh,
performs a standard linear operation there, and returns via tanh.  The atanh
transform stretches values near $\pm 1$ toward infinity, making the layer
sensitive to strong activations -- the multiplicative structure emerges from
the nonlinear domain transform rather than from literal products.

---

## Dimensionality Constraints

* The output dimensionality of the input layer must be equal to the output dimensionality of the perceptual layer, since the conceptual layer operates on both.
* The input dimensionality of the symbolic layer must be equal to the input dimensionality of the perceptual layer, since they both operate on the output of the conceptual layer.
* The input dimensionality of the output layer must be equal to the sum of the output dimensionalities of the symbolic layers.

---

## Pi Layer Derivation (Historical)

The original motivation for the multiplicative layer was to factorise an
exponential sum into a product of linear terms:

$$e^{y_j} =  e^{b_j} + \sum_{i=1}^{n} e^{1+W_{ji} x_i}$$
$$y_j = b_j \cdot \prod_{i=1}^{n} (1+W_{ji} x_i)$$

The current implementation replaces this with the log-space linear form
described above, which achieves the same multiplicative semantics through
atanh/tanh domain transforms while guaranteeing exact invertibility via the
LDU-factored InvertibleLinearLayer.

---

## Invertible Linear Layer (LDU)

### InvertibleLinearLayer -- LDU Factorisation

The core invertible linear primitive factors the weight matrix W as:

```
W = L @ D_embed @ U
```

where:

- $L \in \mathbb{R}^{nIn \times nIn}$: unit lower-triangular matrix (diagonal fixed at 1). Stored as
  `raw_L`; the strict lower triangle is extracted and the diagonal is forced to 1 at
  each forward call.
- **D**: diagonal vector of length `rank = min(nIn, nOut)`, embedded into a rectangular
  `[nIn, nOut]` matrix `D_embed` by zero-padding.
- $U \in \mathbb{R}^{nOut \times nOut}$: unit upper-triangular matrix (diagonal fixed at 1). Stored as
  `raw_U`; the strict upper triangle is extracted symmetrically.

**Exact inverse via triangular solves.**

```
W^{-1} = U^{-1} @ D^{-1} @ L^{-1}
```

Each factor is inverted by a triangular solve (`torch.linalg.solve_triangular`). No SVD
is required, and the inverse is exact as long as all diagonal entries of D are nonzero.

**Parameter count.** $nIn^2 + \mathrm{rank} + nOut^2$ total parameters. Initialized at L = I, d = 1,
U = I so that the initial map is the identity.

### Ergodic Noise Injection (Factor-Level)

Rather than the matrix-level blend $W_{\text{eff}} = b \cdot W + t \cdot N$ (which destroys the LDU
structure and makes the inverse approximate), noise is injected directly into the raw
parameters of each factor before the triangular structure is extracted:

```
L_eff = I + strict_lower(raw_L + t * noise_raw_L)
U_eff = I + strict_upper(raw_U + t * noise_raw_U)
d_eff = b * d_effective + t * noise_d
W_eff = L_eff @ D_eff_embed @ U_eff
```

Because `W_eff` is still in LDU form, its exact inverse is always available by the same
triangular solves -- no approximation, regardless of the noise level.

### stable=True

When `stable=True`, `_d_effective()` clamps each diagonal entry `d_i` to magnitude
`[eps, 1]` with sign preserved before the ergodic blend:

```
d_clamped_i = sign(d_i) * clamp(|d_i|, eps, 1)
d_eff = b * d_clamped + t * noise_d
```

This keeps `d_eff` bounded away from zero, preventing `W_eff` from becoming singular.
`stable=True` is the only place the stability constraint is enforced; no additional
clamp on `d_eff` is needed.

### noise_d

Noise for the diagonal factor is sampled with $|d_i| \in [\text{eps}, 1]$ and random sign, so
the noise factor is itself always invertible. This ensures that even at full temperature
(`b=0, t=1`) the effective matrix is still well-conditioned.

### naive Flag

| `naive` | Forward path | Reverse path |
|---------|-------------|-------------|
| `False` (default) | Apply L, D, U sequentially to x; backprop through each factor separately | Triangular solves: $U^{-1}$, $D^{-1}$, $L^{-1}$ applied in sequence |
| `True` | Materialise `W_eff` as a dense matrix; apply `W_eff @ x` | Materialise the dense LDU inverse and apply it to `y` |

The `naive=False` path never materialises W_eff as a full matrix, saving memory and
allowing each factor's gradients to flow independently.

### Noise Lifecycle in Ergodic Mode

Noise buffers (`noise_raw_L`, `noise_raw_U`, `noise_d`) are:

1. **Resampled at the start of every ergodic `forward()`** -- new noise is drawn before
   constructing `W_eff`.
2. **Resampled again at the end of every ergodic `reverse()`** -- fresh noise is drawn
   after the reverse computation completes.

The window between the two calls is exactly when `reverse()` needs to reconstruct the
same `L_eff`, `U_eff`, `d_eff` that `forward()` used. Because the buffers are not
resampled during that window, `reverse()` reads the stored buffers directly -- no
explicit caching of `W_eff` is required.

### Class Renames

The LDU layer supersedes the previous SVD-based implementation:

| Old name | New name | Status |
|----------|----------|--------|
| `LULayer` | `InvertibleLinearLayer` | Renamed |
| `InvertibleLinearLayer` (SVD) | -- | Removed |
| `InvertibleSigmaLayer` | `SigmaLayer(invertible=True)` | Merged |
| `InvertiblePiLayer` | `PiLayer(invertible=True)` | Merged |

---

## Ergodic Exploration

The ergodic bias-variance control system is documented in [Ergodic.md](Ergodic.md).
