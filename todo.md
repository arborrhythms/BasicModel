# TODO

================================== Codebook VQ Consolidation ==================================

## Remove `vector_quantize_pytorch`; own VQ in Codebook

The external `vector_quantize_pytorch` package's outputs (commit_loss,
rotation_trick gradients, dead-code revival, EMA) are currently
instantiated but not consumed by the training loop — all three callers
of `Codebook.quantize()` discard `commit_loss` with `_`. Reclaim the
machinery, drop the dependency.

### 1. Add gradient-estimator dispatch on `Codebook`

Add a helper `Codebook.apply_gradient_estimator(e, q, mode)` with three
modes:

- `"snap"` — forward `q`, zero gradient to `e` (current default when
  `customVQ=False`).
- `"ste"` — forward `q`, backward identity to `e`. One-liner:
  `e + (q - e).detach()`. Already used inline at `Spaces.py:5122` in
  `SymbolicSpace.forward`'s `use_vqvae_nonreversible` branch —
  consolidate so there is one implementation.
- `"rotation"` — forward `q`, backward applies a Householder-rotation
  of the upstream gradient from `q`'s direction back to `e`'s
  direction, scaled by `||q|| / ||e||`. Implement as a
  `torch.autograd.Function` (~20 lines).

Wire via a new config key `<gradientMode>snap|ste|rotation</gradientMode>`
under each space's codebook section (SymbolicSpace first; extend to
ConceptualSpace later if useful). Default `"snap"` keeps existing
behavior.

**Refactor site:** `Spaces.py:5122` — replace
`z_q_ste = z_e + (quantized_detached - z_e).detach()` with
`z_q_ste = self.subspace.what.apply_gradient_estimator(z_e, quantized, mode="ste")`.

**Tests:** extend `basicmodel/test/test_wide_concept_codebook.py`:
- `"snap"` → `e.grad` is zero after backward.
- `"ste"` → `e.grad == incoming_grad`.
- `"rotation"` → `e.grad.norm() ≈ incoming_grad.norm() * (||q||/||e||)`
  and direction differs from incoming when `e` and `q` aren't
  collinear.

### 2. Port VQ-VAE features into `Codebook` directly

Subsume the external package's features into the existing `customVQ`
branch of `Codebook`:

- **Commitment loss**: add `Codebook.commit_loss(e, q)` returning
  `self.commitment_weight * F.mse_loss(e, q.detach())`. Stop discarding
  it at the three consumer sites (`Spaces.py:911`, `Spaces.py:3327`,
  `Layers.py` VQLayer usage). Thread it into `_symbol_objective_terms`
  alongside the existing `symbol_commitment` bookkeeping already in
  SymbolicSpace (`Spaces.py:~5107`).
- **Dead-code revival**: periodically identify codebook entries whose
  EMA weight has fallen below `threshold_ema_dead_code` and re-seed
  them from the most-surprising recent encoder outputs. Can reuse the
  existing EMA update slot in `Codebook.updateWeights`.
- **EMA updates**: keep the simpler manual EMA in `Codebook.forward`
  (`w[idx, :] = self.eta * w[idx, :] + (1 - self.eta) * x[b, v, :]`).
  The external package's structured EMA callback doesn't add value
  here.

### 3. Per-entry codebook freezing via `ColumnUsageTracker` + σ

`Layers.ColumnUsageTracker` already tracks per-column gradient norms and
freezes low-activity columns in linear layers
(`basicmodel/bin/Layers.py:777`). Adapt that pattern for codebook
entries: track per-entry activation frequency (from
`Codebook.activation`, which already exists at `Spaces.py:908`) and
freeze entries whose σ-weighted usage crosses a stability threshold.

Design sketch:
- Each `Codebook` gets a `usage_sigma` buffer `[nVectors]` accumulated
  across batches: `usage_sigma += sigma_of(activation)` where
  `sigma_of` follows the ergodic-mode sigma logic (see
  `Layers.py` search for `sigma` / `observe_sigma`).
- A `Codebook.freeze_well_learned(threshold)` method walks
  `usage_sigma` and zeroes the gradient for entries whose σ has
  dropped below `threshold` (i.e. the entry has stabilized —
  consistent gradient norm over a window implies the entry has
  converged).
- Similarity to `ColumnUsageTracker.freeze()`: sliding window of
  gradient norms; freeze when mean norm falls below threshold.
  Difference: operates on codebook entries, not linear-layer columns,
  and uses σ (ergodic measure) rather than raw grad-norm.

**Test plan:**
- Unit test: feed a codebook with stable activation over N batches;
  verify `freeze_well_learned()` marks the stable entries frozen.
- Unit test: feed a codebook with fluctuating activation; verify those
  entries remain unfrozen.
- Integration test: full training step with freezing enabled — ensure
  frozen entries retain their values across the optimizer step.

### 4. Remove the external package

Once steps 1–3 are in place and tests are green:

- Delete the try/except fallback blocks for `VectorQuantize` and
  `ResidualVQ` in `basicmodel/bin/Spaces.py` and
  `basicmodel/bin/etc/SigmaPi.py` (they were inlined there when
  `vq_compat.py` was removed).
- Remove `vector_quantize_pytorch` from requirements
  (`basicmodel/.venv` dependency list, pyproject / requirements.txt
  wherever it lives).
- `Codebook.customVQ=True` branch collapses away; the manual path
  becomes the only path with the gradient-estimator dispatch from
  step 1 and the VQ-VAE features from step 2.
- `VQLayer` in `basicmodel/bin/etc/SigmaPi.py` either gets reworked
  to use manual residual-VQ logic or is removed outright (its
  forward path in `LogicalFunctionNet` is already commented out).

================================== Stage 2 — Next ==================================

## Stage 2 follow-up: outer-loop refactor + residual cleanup

Spec: [basicmodel/doc/specs/2026-04-16-stage2-wide-conceptual-codebook-design.md](basicmodel/doc/specs/2026-04-16-stage2-wide-conceptual-codebook-design.md)
Prior plan (mostly landed): [basicmodel/doc/plans/2026-04-16-stage2-wide-conceptual-codebook-plan.md](basicmodel/doc/plans/2026-04-16-stage2-wide-conceptual-codebook-plan.md)

### 1. Outer-loop refactor (originally Task 5.3, revised)

Drop mode-specific branching in the outer forward pipeline. WordSpace is
not invoked from the outer scope — it threads through as `wordSpace=...`
on C/S forwards as today. The loop is a plain `for j in
range(self.conceptualOrder)` — no `while`, no `done()`. Because
`useGrammar=="thoughtFree" ⇒ conceptualOrder==0` is enforced at
config-load, all modes share a single loop body without runtime mode
checks.

Target pseudocode for `BasicModel.forward` (and `MentalModel.forward` if
it diverges):

```
is_ = self.inputSpace.forward(sample_in)
ps = self.perceptSpace.forward(is_, wordSpace=self.wordSpace)
ss = self.symbolSpace.empty_state()
ws = self.wordSpace.subspace.read()
for j in range(self.conceptualOrder):
    cs = self.conceptSpace.forward([ps, ss, ws], wordSpace=self.wordSpace)
    # Top-K is already applied inside Codebook.forward when nVectors > nOutput.
    for k in range(cs.shape[-2]):
        ss = self.symbolSpace.forward(cs[..., k, :], wordSpace=self.wordSpace)
return self.outputSpace.forward([ps, ss])
```

Implementation notes:
- Current `BasicModel.forward` orchestrates per-tier calls with
  grammar-path / butterfly-path branching at `Models.py:~2164` (`elif
  self.useGrammar == "all":`). Identify the top-level branch that maps
  to this simpler loop and replace it.
- Gate behind a new `architecture.legacyLoop` config flag (default
  `false` once validated) so existing configs can opt into the pre-
  refactor path while this is being validated.
- Parallel (`thoughtFree`) and raw (`none`) modes have `conceptualOrder
  == 0`, so the j-loop body runs zero times. The seed emission must
  still happen — decide whether the post-loop emission is (a) an
  implicit j=-1 pre-seed C→S pass, or (b) done by driving the loop with
  `range(max(conceptualOrder, 1))` and then deciding inside. Prefer (a)
  since it keeps the assertion literal.
- Tests to add in `basicmodel/test/test_merged_loop.py`:
  - `useGrammar="all"` with `conceptualOrder=3` runs three j-iterations.
  - `useGrammar="thoughtFree"` with `conceptualOrder=0` runs the pre-
    seed emission once, no j-iterations.
  - Full training-loop smoke test unchanged for existing configs.

### 2. Delete unused scaffolding

Earlier Task 5.1+5.2 added scaffolding that the revised Task 5.3 does
**not** use. After Task 1 lands cleanly, delete:
- `WordSpace.done()`, `note_emit()`, `reset_emit_state()`, `_parent`,
  `_emit_count`, `_n_percepts_consumed` (in `basicmodel/bin/Spaces.py`)
- `SymbolicSpace.empty_state()` — unless the refactor ends up calling
  it to seed `ss`; check both call sites before removing.
- The corresponding tests in `basicmodel/test/test_merged_loop.py`
  (replace with the integration tests listed above).

### 3. Kill dead `stm_decay` wiring

`architecture.STM_decay` is read into `self.stm_decay` at
`basicmodel/bin/Models.py:2003` and never consumed. Remove the config
key from the XSD and the load-site. Small one-line cleanup.

### 4. Lexicon migration (InputSpace → PerceptualSpace)

The spec moves the lexicon from `InputSpace.codebook` to
`PerceptualSpace`. Stage 2 added the three-way chunking switch on
PerceptualSpace but did **not** remove the InputSpace lexicon code — the
legacy path stays live. A follow-up plan should:
- Make `InputSpace.codebook=false` the validated default.
- Verify PerceptualSpace's `chunking_mode` is the only lexical path in
  the pipeline.
- Migrate existing configs one by one after smoke-testing each.

### 5. Parallel-mode supervision signal

The spec's open item: what loss supervises `ss` in `thoughtFree` /
`none` modes, where grammar doesn't compose?
- Candidate A: multi-label BCE over the symbol codebook indices.
- Candidate B: cosine to a bag-of-symbols target superposition.
- Candidate C: reuse the existing output-reconstruction loss.
Empirical exploration needed — one training run per candidate, compare
convergence.

### 6. Checkpoint migration for the boolean → tri-state `useGrammar` change

The loader's legacy fallback (`parse_use_grammar` with `thought_free`
sidecar) makes existing checkpoints/configs load correctly, but a
one-time migration script that rewrites old XML configs in place would
reduce drift. Low priority; the legacy path is stable.

### 7. `useGrammar="none"` assertion (decided to skip)

Considered but rejected: `useGrammar=="none" ⇒ conceptualOrder==0`
would break six legitimate butterfly configs where `conceptualOrder`
drives butterfly N-halving stage count, not grammar composition. The
`thoughtFree` assertion is sufficient.

================================== MentalModel ==================================

================================== April 24 ==================================

# Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

# Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

# Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

## Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov’s guardrails on the output, which is a famous failure mode in terms of it’s loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.


## Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d² / 2(σx² + σy²))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere — a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform σ as a proxy — high gradient variance (unstable region) → larger σ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

## Implement forgetting
If someone has contributed information to an LLM and asked for the LLM
to learn from that data, even when the data is revoked, it will be remembered
in some vague way by the weights. So implement non-destructive forgetting:
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).

## Meaningful Prediction
Since we predict both words and sentences, the best prediction of future content is *meaningful*. Predictions of the next sentence should be ConceptualSpace predictions of the head (as a token), then of the NP+VP (as two tokens), iteratively refining until the full sentence has been specified as a grammatical tree — rather than strictly AR-1 over words (which is a surface-level prediction, not a meaningful prediction).

For example, instead of predicting "the fast dog jumped" token-by-token left to right, predict an XML-encoded version of `(((dog) fast) the) jumped` — the head of the sentence first, then iterative refinement of that conceptual space. This takes the same number of production steps as a current LLM, but the iterative refinement of the next-sentence production is conceptually much different, and closer to human reasoning where there is a core truth (S) with spatial NP and temporal VP that are successively refined by adjectives and adverbs that scope the conceptual space of that kernel sentence.

* Ensure words are aligned with sentences when there is a single unambiguous head.
* Head-first prediction requires that the head can be reliably identified from the composition.

## Future Work: Parsed Training Dataset
It is desirable to create a small training and testing dataset for the network consisting of statements that are already parsed. This would allow direct comparison between the grammatical derivation produced by traditional English parsers and the deep structure produced by BasicModel's ConceptualSyntacticLayer.

* See `bin/parse.py` as a starting point for producing grammatical derivations via NLTK POS tagging and CFG parsing.
* Such a dataset would also enable evaluation of head identification accuracy and composition quality.
