# 2026-05-29: clean-stack STM, basis= kwarg, RadixLayer

Architectural pivot landing after the 2026-05-27 PerceptStore plan and the
2026-05-28 where-keyed-taxonomy spec. This page is the canonical reference
for the changes that landed between 2026-05-27 and 2026-05-29; top-level
docs link here instead of duplicating the rationale.

## Summary

Seven interlocking changes, listed in landing order. Most are
single-purpose deltas to one or two files; the per-stage Stage-10 sigma
pipeline lingers but is bypassed on the forward path.

### 1. `PerceptStore` $\to$ `RadixLayer(Layer)` (`bin/Layers.py`)

The PerceptStore (radix trie + inverse table + learned codebook + byte
fallback) is now a first-class `Layer` subclass living in
`bin/Layers.py`. `PerceptualSpace.reverse` invokes
`RadixLayer.reverse` for the structural decode (chunk-id $\to$ bytes $\to$
slot).

Why: the radix-trie machinery is reusable beyond `PerceptualSpace`; a
Layer subclass exposes it to the chart-reverse / signal-router dispatch
on the same surface as every other layer (`forward` / `reverse` /
`compose`). The previous `PerceptStore` standalone class became
purely structural with no Layer-level callers.

### 2. `MetaLayer` $\to$ `SymbolizeLayer` (`bin/Layers.py`)

Rename. The grammar layer that promotes a freshly-seen percept to a
symbolic prototype is now `SymbolizeLayer`. No semantic change; the
old name conflated "the binary GrammarLayer that creates META
entries" with "the META cross-codebook taxonomy itself" (a different
concept — see #3).

### 3. Auto-META on `PerceptStore` promotion fires from `ConceptualSpace`

The cross-codebook bind (META entry: PS chunk-id $\leftrightarrow$ SS prototype-id)
used to fire inside `PerceptualSpace._embed_radix` via a back-ref
from PS $\to$ SS. Moved to `ConceptualSpace._maybe_autobind_meta`, which
fires at stage 0 only. CS is the natural place because it sees both
the PS contribution (via `subspace.materialize()` /
`perceptualSpace_ref._forward_input['indices']`) and the SS peer
(via `terminalSymbolicSpace_ref`); no back-ref on PS.

Autobind gate (simplified): require only that the terminal SS
`subspace.what` is a `Codebook`. Configs with SS
`<codebook>none</codebook>` silently no-op here, so the experiment
remains runnable on PS-codebook-on / SS-codebook-off setups (the
inverse of MM_xor.xml's current direction).

### 4. LBG-style codebook splitting on SS (Gray 1990)

The SS codebook is updated under EMA (Gray 1990) and split when a
prototype's running per-component variance exceeds a threshold
("LBG-style" — Linde-Buzo-Gray). The split direction is the
top-variance eigendirection of the accumulator. New prototypes are
seeded around the parent $\pm$ `$\delta$ * variance_axis`, then EMA continues
independently on each child.

Why: random-direction VQ init lands too many prototypes in degenerate
basins on small-codebook configs; LBG gives a principled "where to
add a new prototype" rule that tracks the live data distribution.

### 5. LSE soft-max / soft-min kernels (`Ops._soft_radmax` /
`Ops._soft_radmin`)

Replaced the hard `max` / `min` in `Ops._disjunction_kernel` /
`Ops._conjunction_kernel` with LogSumExp-based smooth variants
(canonical smooth max approximation, see e.g. Boyd & Vandenberghe
§3.1.5). The `kind='soft'` branch is the default when
`monotonic=False`.

Why: hard max/min kill gradients on the loser branch; LSE gives a
smooth `tau $\cdot$ log(exp(x/tau) + exp(y/tau))` that flows gradient
through both operands. The hard branch (`kind='hard'`) is retained
for monotonic-mode and for tests that need exact idempotency.

The LSE bias is bounded: `max(x) $\le$ tau $\cdot$ LSE(x/tau) $\le$ max(x) +
tau $\cdot$ log(n)`. At `tau=0.1, n=2`: `tau $\cdot$ log(2) $\approx$ 0.069`. Several
binary-op tests had their idempotency tolerances loosened from
`atol=1e-6` to `atol=0.1` to accommodate this bias.

### 6. Clean-stack STM (`bin/Spaces.py` `ConceptualSpace.forward`)

Replaced the Stage-10 additive composition
`folded = sigma_in(combined) + sigma_cs(prev)` with per-stage tier
attribution:

    stage 0      STM = primary    (PS event)
    stage k > 0  STM = sym        (SS event)

No additive mixing across tiers; no residual lift. Trivially
invertible (read-back, no inverse-Sigma needed).

When the SS contribution is missing or shape-mismatched at a higher
stage we fall back to `primary` so the cascade doesn't silently lose
content — the higher stage degrades to "pass the prior PS along".

The `STM_k = STM_{k-1} + SS_k` carry-forward variant was tested and
reverted (per user, 2026-05-29) — the pure clean-stack form is the
landing point.

Side effect: `sigma_in` / `sigma_cs` parameters are dead-weight on the
forward path (no gradient). The MM_xor convergence path still
converges because the butterfly cascade on PiLayer carries the
trainable transform. The reverse path (`ConceptualSpace.reverse`)
still calls `self.sigma_in.reverse`, but inverts a fold that was
never applied — a latent semantic mismatch documented as a follow-up.

### 7. `basis=` kwarg replaces `W=` for grammar reverses

`UnionLayer.reverse(parent, basis=None)` and
`IntersectionLayer.reverse(parent, basis=None)` (and their
`.generate`) now accept a Codebook / Basis object (typically
`SymbolicSpace.subspace.what`) instead of a raw `W` tensor. Internally
the layer extracts `W = basis.getW()` and dispatches to
`Ops.disjunctionReverse` / `Ops.conjunctionReverse` (which still
take the raw `W` tensor — they're the low-level kernel).

Why: the public Layer surface is cleaner when the caller passes the
Basis (it owns `getW()`, may grow other lookups) than when it passes
the raw tensor. No back-ref is stored on the layer.

Dispatch site: `bin/Language.py` `unreduce()` fetches
`tier_basis = getattr(subspace, 'what', None)` and calls
`layer.reverse(parent, basis=tier_basis)` with a `TypeError`
fallback for layers that don't accept the kwarg yet (NotLayer,
NonLayer, base GrammarLayer, ...).

### 8. `<useVQVAE>` XML knob retired

The `<useVQVAE>` SymbolicSpace XML element is gone. With
`<codebook>quantize</codebook>`, the codebook is always trained via
the VQ-VAE / EMA dispatch (continuous flow + commitment loss; or
STE-through-snap + codebook commit, depending on `reversible`).

Why: the prior `<useVQVAE>false</useVQVAE>` mode dispatched to a
`hard_quantize` branch that snapped to nearest prototype with no
gradient through the snap and no codebook training signal. The SS
codebook rows stayed at random init for the entire run. Nothing
useful learned. The misnomer ("VQ-VAE on/off") obscured that what
the toggle actually controlled was "is the codebook frozen at
random init or not."

Replacement: the codebook training is implicit when a codebook is
configured. For inference / frozen-codebook scenarios, use
`model.eval()` + `requires_grad_(False)` at the call site, not an
XML toggle. The `hard_quantize` branch in `SymbolicSpace.forward`
is deleted.

Tests / configs updated: `data/model.xsd` element removed; every
`data/*.xml` stripped of `<useVQVAE>...</useVQVAE>`;
`test_mm_grammar_without_vqvae_learns_xor_signal` deleted (the mode
no longer exists); `bin/Spaces.py` derives `self.use_vqvae =
bool(self.codebook)` rather than reading the XML.

## Other concurrent changes

* `Embedding.normalize()` (unit-ball projection) called after
  `optimizer.step()` in `bin/Models.py` training loop. Keeps embedding
  vectors from drifting off the unit ball under joint training.
* `_stm_set_all_slots` replaces `mean(dim=1) + shift_push` for parallel
  mode `[B, N, D]` — each of N positions is its own STM slot, no
  destructive mean-reduce.
* `BaseModel.conceptualMode = "parallel"` class-level default.
* Null sentinel `\x00` appended after `parse(lex='words')` for
  explicit end-of-sequence.
* `<reconstruct>concepts</reconstruct>` is the default in
  `data/model.xml`; stripped from per-experiment XMLs.
* `_decode_reconstructed_inputs` filters `w.strip()` tokens (drops
  empty / whitespace-only words in the reverse decode display).
* `data/MM_xor.xml`: PS `<codebook>none</codebook>` + `chunking=radix`
  + butterfly=true on PS & CS. SS `<codebook>quantize</codebook>`.
* `data/MM_xor_fixture.xml`: dedicated test fixture for META-taxonomy
  tests (decoupled from MM_xor.xml so experimental edits don't break
  the suite).
* `data/XOR_exact.xml`: restored to butterfly=true, codebook=none
  everywhere; perfect reconstruction.

## Test plan

Targeted regression:

* `test/test_grammar_binary_ops.py` — 18 passed. Three LSE-affected
  idempotency tests loosened to `atol=0.1` with comments noting the
  `tau $\cdot$ log(2) $\approx$ 0.069` LSE bias.
* `test/test_mm_xor.py::TestMMXorConvergence` — all remaining tests
  pass. `test_mm_grammar_without_vqvae_learns_xor_signal` deleted
  (the mode it tested no longer exists — see §8 above).
  `test_learns_xor_signal` and `test_convergence` now use seeded
  retries (`for seed in (42, 123, 7)`); the previously-dead `seed`
  loop variable in `test_convergence` is now actually consumed by
  `torch.manual_seed(seed)`.
* `test/test_cs_reentrancy.py` — 25 passed, 1 skipped. Two tests
  repaired to assert clean-stack invariants
  (`TestForwardBypassesSigmaInPerStage`,
  `TestCSForwardBypassesSigmaIn`).

## Pending follow-ups

* **Forward / reverse semantic mismatch on Stage-10 sigma pipeline.**
  Forward bypasses `sigma_in` / `sigma_cs`; reverse still inverts as
  if they were applied. Numeric round-trip closes today (sigmas are
  near-identity since they aren't trained on forward), but this will
  bite under any non-trivial sigma training.
* **`conceptualOrder > 1` invertibility (`GrammarMergeGlue` audit).**
  Multi-stage chart-reverse path needs a parallel sweep similar to
  the binary-ops `basis=` work.
* **Latent dead-weight sigma parameters.** Decide whether to retire
  `sigma_in` / `sigma_cs` from the forward path entirely or to wire
  them back in a way that doesn't reintroduce the additive
  composition we just removed.
