# Radix on MPS: Adam step-zero misdiagnosis

Status: **investigated, not fixed; documented as a blocked task**.
Scope: `chunking=radix` with `lexer=byte` on Apple-Silicon MPS only.
CPU path (CPU-only or `BASICMODEL_DEVICE=cpu`): green.

## TL;DR

What was shipped under the headline "non-finite reconstruction loss
$\to$ divergence" for `chunking=radix` on MPS is **not** a numerical
divergence. The reconstruction loss is fp32-finite. The crash is a
**downstream symptom** of a radix-specific MPS command-queue
corruption: the Adam optimizer's in-place step increment
(`step_t += 1`) is silently dropped on MPS for every gradient-bearing
dense parameter on batch 0, so the next line
`step_size = lr / bias_correction1` divides by
$1 - 0.9^{0} = 0$ and raises `ZeroDivisionError`. The `isfinite`
false-positive at the loss check is the **same** corruption surfacing
one call earlier.

## Two real fixes that did land this round

These are independent of MPS and are kept in the tree.

1. **Radix reconstruction collapse (CPU + MPS once unblocked).**
   `perceptual_sbow_loss` was a centroid-only cosine attractor on the
   PS codebook; with the null-percept slots dominating it pulled every
   active percept toward `\x00`. Replaced with the negative-sampling
   (pode/antipode) SBOW already present in `bin/embed.py` -- the
   centroid-attractor is balanced by random-row repulsion via
   `_neg_sampling_loss`, exactly the topology described in
   `embed.py`'s `PretrainModel`. Plumbed through a new
   `PretrainModel.sbow_loss_indices(idx_list)` that operates on the
   gathered percept ids (`_forward_input['indices']`) with the
   `\x00` (null_pid) slots excluded. Result: radix reconstructs the
   prompts byte-perfect on CPU (`'hello world'` $\to$
   `'hello world \x00 \x00 ...'`).

2. **Butterfly diagonal clamp.**
   `GrammarLayer._butterfly_pair_reverse` divided by the **raw**
   learnable diagonal `d_node` with no clamp, so as soon as a learnable
   `|d|$\to$ 0` the inverse $1/d \to \infty$. Added
   `GrammarLayer._stable_pair_d` applied identically in
   `_butterfly_pair_forward` and `_butterfly_pair_reverse`:
   $\mathrm{sign}(d) \cdot |d|.\mathrm{clamp}(\epsilon, 1)$ with
   $\epsilon = 10^{-3}$, and `sign(0) = +1`. Invertibility is
   preserved (same map applied in both directions). All
   370 grammar/invertibility/sigmapi tests stay green.

Both fixes are in `bin/Models.py` (`perceptual_sbow_loss`,
`PretrainModel.sbow_loss_indices`) and `bin/Layers.py`
(`GrammarLayer._stable_pair_d`, `RadixLayer.reverse` decode fix that
bounds the search to active rows and falls back to the PS table when
the SS codebook is absent).

## The MPS issue, characterized

`chunking=radix` with `lexer=byte` is the only chunking mode that
takes the `byte_mode` codebook path (`analyse`/`lexicon` use the
`raw`/`word` codebook). Symptoms on MPS:

- Default check (`isfinite(lossIn).all()` evaluated on MPS): raises
  the "non-finite reconstruction loss" `RuntimeError`.
- Same check moved to CPU (`isfinite(lossIn.detach().cpu()).all()`):
  passes. The loss is finite.
- Past the check, training reaches `optimizer.step()` and dies in
  `torch/optim/adam.py:_single_tensor_adam` line 533:
  `step_size = lr / bias_correction1` $\to$ `ZeroDivisionError`,
  because `bias_correction1 = 1 - $\beta_1^{\,\text{step}} = 1 - 0.9^{0} = 0$.
- Optimizer-state dump at the failure point:
  `Adam g[0] betas=(0.9, 0.999) cap=False nP=60 step0=16
   stepNone=44`. All 16 gradient-bearing dense parameters have
  step $= 0$ after `step_t += 1`. The 44 stepNone are params that
  received no gradient on this batch (no state yet), which is expected.

So the same radix-on-MPS state corruption produces **two** symptoms:
the `isfinite` reduction misreports a finite scalar, AND a 0-dim
`+= 1` on the optimizer step tensor silently no-ops. Same class.

## What this is NOT (ruled out)

Each item below was tested in isolation; all increment the Adam step
to 1.0 correctly on MPS in a single-script repro. None reproduce the
in-model failure.

- **Plain `nn.Adam` on MPS** (single param, multi-param, multi-param
  with mixed `None` grads matching the radix shape of 60 params /
  16 grad-bearing). Step $\to$ 1.0.
- **`SparseAdam` + dense `Adam` coexistence** under the
  `_MultiOptimizer` wrapper (`bin/Models.py`). Both step $\to$ 1.0.
- **Step tensor on `mps:0`** (which is what the model uses, because
  `util.py:247` calls `torch.set_default_device('mps')` $\to$ the
  default-device `torch.tensor(0.0)` Adam allocates for `step` lands
  on MPS). A manually-placed mps step still increments.
- **Heavy-queue / pre-existing graph** before the step: many large
  matmuls then a 0-dim `+= 1` $\to$ correct.
- **`torch.mps.synchronize()` before `optimizer.step()`**: no effect,
  still drops the increment. So it's not a stale-host-read race.
- **`addcmul_` / `addcdiv_` non-contiguous silent-no-op MPS bug**
  (Apple Silicon, fixed $\geq$ macOS 15): host is macOS 26, torch 2.12;
  neither op is on the relevant path anyway.
- **Uninitialized memory.** `torch.utils.deterministic
  .fill_uninitialized_memory = True` set at process start (so every
  `torch.empty` allocates NaN) on CPU: radix still reconstructs
  perfectly. No uninitialized read.
- **Autocast / BF-16 / FP-16.** `MODEL_AMP` defaults to off and
  `util.py:649` notes "MPS has no working autocast path"; MPS runs
  fp32. AMP is not engaged.
- **`maskRate=0`** (skips `create_ir_mask`): still crashes. The
  masked-LM `lossIn` derivation
  (`pred_full = PS.subspace.materialize(); compute_masked(...)`) is
  not the corruptor by itself.
- **Skip the reverse loss term entirely** (`SKIP_REV=1` gating
  `forwardInput is not None`): still crashes. Not the reverse.
- **Disable the new neg-sampling SBOW** (`NO_SBOW=1`): still
  crashes. Not the SBOW.
- **Reduce conceptual order 3 $\to$ 2 $\to$ 1**: all three still
  crash. Not the cubic combine.
- **Promotion disabled** (`chunkPromotionThreshold=100000`): still
  crashes. Not the radix promotion / codebook rebuild.
- **`mid-epoch rebuild_optimizer`**: byte_mode gates the OOV-insert
  rebuild path (`bin/Spaces.py:8743` $\to$
  `if oov_words and not getattr(codebook, 'byte_mode', False)`). The
  optimizer is **not** rebuilt mid-run for radix.
- **`pred_full = materialize()` non-contiguous slice write**
  (`new_event[..., :nWhat] = torch.where(...)` at
  `bin/Models.py:2358`): a stand-alone repro of the slice-write +
  backward + Adam step works on MPS.
- **Codebook clamp / `_wrap_unit_ball` `torch.remainder` on MPS**:
  not isolated yet, but doesn't fit the "step counter dropped"
  symptom.

## What it likely IS (best remaining hypothesis)

The single property that holds for `chunking=radix` and not for
`analyse`/`lexicon` is the **preallocated byte-codebook surface**:

- The PS codebook is `nVectors = nVectors_max` (e.g. 65536) rows from
  the start, but only the active byte cells receive gradients on any
  given batch (radix is byte_mode $\to$ a fixed 256-cell active surface,
  the rest reserve).
- `SparseAdam` is registered over the codebook; **dense `Adam`** is
  registered over every other parameter (the 60 dense params shown
  above), and it is dense Adam whose step is dropped.
- `torch.set_default_device('mps')` puts the dense Adam's step tensor
  on `mps:0` (rather than CPU, which is where `capturable=False`
  Adam normally keeps it) -- a property unique to this codebase, but
  shown above to be necessary-not-sufficient in isolation.

The strongest remaining hypothesis is that some op in the radix-path
forward or backward (the `RadixLayer` traversal, the percept-store
gather/scatter, or the `nVectors_max`-row codebook's `_wrap_unit_ball`
$\to$ `torch.remainder` path, or something in the
`SparseAdam`/`dense Adam` co-residency on a large preallocated
codebook) wedges the MPS command queue such that **the next on-device
0-dim integer increment is dropped**. The step counter is the only
0-dim integer the optimizer touches; the `isfinite` reduction is the
only 0-bit boolean the loss check touches; both surface the same
corruption.

This pattern is consistent with the user's standing intuition that the
issue is "in-place writes silently failing on MPS" -- which is real,
documented for `addcmul_`/`addcdiv_` pre-macOS-15, and a known MPS
hazard class. We just haven't isolated the producing op yet.

## Next steps for whoever picks this up

1. **Bisect on what radix does that analyse does not.** The two
   differentiators are (a) `byte_mode=True` (so the OOV-insert/rebuild
   gate flips, `null_percept_idx` is defined, `create_ir_mask` runs,
   and `compute_masked` is the loss); (b) `chunking=radix` $\to$ the
   percept-store path goes through `_embed_radix` and `RadixLayer`.
   `(a)` is independently testable on `lexicon` if you force its
   codebook to byte_mode -- if `lexicon`-with-byte still works on MPS,
   the corruptor is in `(b)`.
2. **Force-CPU just the optimizer-state tensors.** Without changing
   any of the model's compute path, place each Adam param group's
   `step` / `exp_avg` / `exp_avg_sq` on CPU. If the dense Adam then
   trains, the corruption is **only** the on-MPS 0-dim integer write,
   not the on-MPS gradient. (This is *not* the "CPU workaround" the
   user vetoed -- the forward/backward stays on MPS; just the
   optimizer book-keeping moves.)
3. **Per-op probe inside `RadixLayer.forward`** with reliable
   CPU-first reductions (`x.detach().cpu().isfinite().all()`, never
   `x.isfinite().all().cpu()` -- the latter reduces on MPS first and
   is exactly how I missed the corruption for several iterations).
4. **File the minimal repro upstream** once we have it. Apple has
   fixed two similar bugs already (addcmul/addcdiv, gumbel_softmax
   $\to$ NaN). A clean 30-line repro of "0-dim `+= 1` dropped on MPS
   after radix-shaped op X" is exactly the report shape they ship a
   patch against.

## Sanity gates that still pass with the current fixes

- `pytest test/test_basicmodel.py -q`: 199 passed, 2 skipped,
  6 xfailed.
- `pytest test/test_grammar.py test/test_invertibility.py
  test/test_sigmapi.py -q`: green (the d-clamp didn't regress).
- `BASICMODEL_DEVICE=cpu test/_sweep_chunking.py` with
  `SWEEP_CHUNKS=radix SWEEP_LEXER=byte SWEEP_CODEBOOK=none`:
  reconstructs byte-perfect on all four prompts. `predicted` stays
  at $0.5$, which is the XOR-head story below, not a reconstruction
  problem.
- `pytest test/test_embed.py -q` (covers the
  `PretrainModel.sbow_loss_indices` integration): green.
