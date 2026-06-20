# 2026-05-14 — Retire `<maskedPrediction>`; within-sentence training is IR-only; sentence-level AR moves to `InterSentenceLayer` (ARMA(5,2))

## Motivation

After the 2026-05-13 sigma/pi rebalance + AR cursor-unfold retirement,
within-sentence training paths benchmarked on `MM_5M_bivector + BPE`
(`bs=128`, `max-autotune`, GB10) as:

| Mode | Sent/sec | Notes |
|---|---|---|
| **AR** / **ARIR** / **ARUS** | ~18 | K-loop pipeline: K body+head calls per batch |
| **IR** (post-2026-05-13 fast path) | **~61** | one body+head call per batch, mask + predict |

AR-mode within-sentence training is ~3× slower than IR. The real AR
objective in this architecture is **next-sentence** prediction (the
discourse / inter-sentence layer) — not next-token within a sentence.
Within-sentence training can therefore be limited to IR (masked-LM
with `mask_rate` percent of positions replaced by `NULL_PERCEPT` and
the model predicting the originals); sentence-level AR moves into
`InterSentenceLayer`, rewritten as an ARMA(5, 2) MLP over sentence
representations.

Collapsing AR within-sentence also lets us delete the
`<reconstruct>=output` path — the only configuration that still
routes through the full reverse pipeline (`_reverse_per_stage` →
`_pair_unmerge`) inside training. That path has a known shape bug
under `useGrammar="all"` (the C-tier bivector width `D=2` doesn't
match the unwind's wider intermediate state); retiring it removes
the bug from the training surface entirely.

This plan is the **single in-scope unit** covering four coupled
changes: (1) retire `<maskedPrediction>`, (2) retire
`<reconstruct>=output`, (3) rename `<reverseScale>` →
`<reconstructionScale>` (semantic kept; field stays alive as the
output ↔ reconstruction blend weight), (4) rewrite
`InterSentenceLayer` as an ARMA(5, 2) MLP. The four are coupled —
the IR-only forward stabilizes the sentence-rep capture site that
the ARMA layer reads from — so they ship together.

## Current state (post-2026-05-13, after the K-axis retirement)

- `<architecture><maskedPrediction>` ∈ {`NONE`, `IR`, `AR`, `ARUS`, `ARIR`}
  selects within-sentence objective. `ARIR` is the default in
  `MM_5M_bivector.xml`.
- `<architecture><reconstruct>` ∈ {`none`, `symbols`, `concepts`,
  `output`, `both`} selects the reconstruction-loss target.
- `<training><reverseScale>` ∈ `[0, 1]` is the blend weight between
  output loss and reconstruction loss (confirmed at
  `bin/Models.py:3158`: `output_weight = (1 - reverse_scale)`).
- `BasicModel.forward` (`_forward_per_stage`) dispatches:
  - AR / ARIR / ARUS (non-runtime) → `_forward_per_stage_no_unfold`
    serial K-loop with causal mask on N.
  - IR (non-runtime) → `_forward_per_stage_no_unfold` single-shot
    with random mask at P-tier (the "fast IR" path that bypasses the
    legacy reverse pipeline).
  - NONE / ARIR-runtime → legacy `_forward_per_stage` body.
- `runBatch` IR branch (line 3209) computes the loss at the P-tier
  event vs `_ir_pre_mask_input` at masked positions — no
  `self.reverse()`.
- `self.reverse()` fires from ARIR-runtime inference, non-AR
  reversible reconstruction (`reverseScale > 0`), and direct test
  calls. `_pair_unmerge` lives in `_reverse_per_stage`'s
  `useGrammar == "all"` progressive-bottleneck unwind and breaks
  under the bivector dim contract.
- `_flatten_k` / `_restore_k` and `SubSpace.k_axis` are **already
  retired** (2026-05-13).
- `InterSentenceLayer` ([bin/Layers.py:5680](../../bin/Layers.py:5680))
  is the per-sentence substrate: stores `[n_sentence × n_dim]`
  snapshots ( `[S | W]` concatenation: SymbolicSpace's final
  `materialize()` rows + WordSpace's `read()` rows) and scores them
  via dual-force cosine contrastive loss (attractive toward a
  `context_window=12` centroid, repulsive from older centroids in a
  `centroid_history=3` ring at `lambda=1.01`).
- `PartLayer` / `EqualsLayer` / `QueryLayer` are pure-geometric
  operations on the SymbolicSpace bivector codebook (S-tier rules,
  per `doc/Architecture.md` §"Monotonicity of the bivector chain").

## Target architecture

### 1. Retire `<maskedPrediction>` — within-sentence training is IR

Within-sentence training is **always IR** (masked-LM at the P-tier).
There is no within-sentence AR objective. Sentence-level AR moves to
`InterSentenceLayer` (see §4).

- `data/model.xsd`: delete `<xs:simpleType name="maskedPredictionEnum">`
  plus both `<xs:element name="maskedPrediction" ...>` declarations
  (architecture, training).
- `bin/Models.py`:
  - Remove the `self.masked_prediction` field and the constructor
    parameter that takes `masked_prediction=...`.
  - Remove the dispatch in `_forward_per_stage` that branches on
    `self.masked_prediction in ('AR', 'ARUS', 'ARIR')`. All
    non-runtime forwards now go through the IR single-shot path.
  - Delete the AR K-loop branch in `_forward_per_stage_no_unfold` —
    keep only the IR fast path. The method body simplifies to:
    1. InputSpace.forward → PerceptualSpace.forward (`[B, N, D]`).
    2. `create_ir_mask(p_sub)` (random mask at P-tier).
    3. `_forward_body(p_sub)` (T stages on B rows).
    4. `_forward_head(body_sub)` → `[B, N, predDim]`.
    5. Return predictions; runBatch computes IR loss at the P-tier.
  - Remove the `arir_mode` runtime branch from `runBatch` (within-
    sentence AR generation is gone). Chat-loop inference now routes
    through `InterSentenceLayer`'s ARMA predictor (see §4) for
    sentence-level next-token sampling.
  - Remove the `is_ar_mode` / `is_ir_mode` flag plumbing in
    `runBatch`; replace with the single IR loss path.
  - Delete `_forward_stem_per_word_serial` (the serial K-loop stem;
    unused after AR retirement).
  - Delete `is_runtime_arir` / `_runtime_mode == 'ARIR'` branches
    throughout. Audit `inputSpace.data._runtime_mode` for remaining
    non-ARIR uses; retire the attribute if dead.
- `bin/Spaces.py`:
  - `InputSpace.forward`: no longer needs the `if not is_ar` early
    return — all paths emit `[B, N, D]`. Lex+embed runs once; the
    legacy AR / ARIR unfold blocks (already empty after the K-axis
    retirement) are deleted.

### 2. Retire `<reconstruct>=output`

`<reconstruct>=output` is the only path that exercises the reverse
pipeline (`_reverse_per_stage` → `_pair_unmerge`). Drop it.

- `data/model.xsd`: remove the `output` enumeration from
  `reconstructEnum`. New set: `{none, symbols, concepts, both}`.
- `bin/Models.py`:
  - Remove the `rc == "OUTPUT"` (or equivalent) branches in
    `runBatch`'s reconstruction-loss path. Without `output`, the
    reconstruction loss never calls `self.reverse(...)`.
  - Delete `_reverse_per_stage`, `_run_pipeline_rev`,
    `_reverse_body`, `_reverse_head`, `_reverse_perceptual`,
    `_chart_generate_from_stm`, `_pair_unmerge`,
    `_should_reconstruct`, `_run_reverse_sequential`, and
    `BasicModel.reverse`.
- `bin/Language.py` / chart cleanup:
  - Audit chart-side `generate` (the outside / reverse pass) — if it
    is only called by the deleted reverse path, retire it.
- The `<reconstruct>` enum's surviving values mean:
  - `none`: no reconstruction loss. Train on IR prediction loss only.
  - `symbols`: reconstruction target is the symbolic intermediate.
    Compare the body's S-tier event at masked positions to a
    target symbolic state. Forward-only.
  - `concepts`: reconstruction target is the conceptual
    intermediate. Compare the body's C-tier event at masked
    positions to a target conceptual state. Forward-only.
  - `both`: both `symbols` and `concepts` losses fire (summed).

### 3. Rename `<reverseScale>` → `<reconstructionScale>` (keep, do not retire)

`<reverseScale>` is the **blend weight between output loss and
reconstruction loss** (`output_weight = (1 - reverse_scale)`,
`recon_weight = reverse_scale`). The legacy name was tied to the
now-retired "reverse" pipeline; rename to match the actual semantic:

```
total_loss = (1 - reconstructionScale) * output_loss
           +      reconstructionScale  * reconstruction_loss
```

- `data/model.xsd`: rename `<xs:element name="reverseScale" ...>`
  to `<xs:element name="reconstructionScale" type="unitInterval"
  minOccurs="0"/>`. Keep `unitInterval` (range `[0, 1]`).
- `bin/Models.py`: rename constructor parameter `reverse_scale` →
  `reconstruction_scale`; rename `self.loss.reverse_scale` →
  `self.loss.reconstruction_scale`; update all 11 references in
  Models.py and `bin/Layers.py:ModelLoss` to match.
- XML migration: every `<reverseScale>X</reverseScale>` →
  `<reconstructionScale>X</reconstructionScale>` across
  `data/*.xml`.
- Parser back-compat: accept the legacy `<reverseScale>` name with a
  one-line deprecation warning, route to
  `reconstruction_scale` internally. Remove the back-compat shim in
  the next release.

### 4. Rewrite `InterSentenceLayer` as ARMA(5, 2) MLP over sentence reps

The existing `InterSentenceLayer` ([bin/Layers.py:5680](../../bin/Layers.py:5680))
is the sentence-level substrate; rewrite its loss from dual-force
cosine contrastive to **ARMA(5, 2)** next-sentence prediction.

**Sentence representation `s_t`** is either:

- **one idea** — the final stage's S-tier event pooled into a single
  vector per row (the existing `_ss_cache[-1]` is the natural
  capture site), OR
- **a relation over multiple ideas** — `PART(a, b)` / `EQUALS(a, b)` /
  `QUERY(a, b)` (the existing `PartLayer` / `EqualsLayer` /
  `QueryLayer` at S-tier produce binary-fold outputs that can serve
  as the relation rep).

The choice of single-idea vs relation is made by the grammar: a
sentence parses to either a single concept rule (`pi(C)`-style fold
to one slot) or a binary relation rule firing `part` / `equals` /
`query`. The chart's parse trace already knows which fired; the
sentence rep is the S-tier slot that the root rule wrote into.

**ARMA(5, 2) means**: predict sentence `t` from
- the **5** previous sentence reps `s_{t-1..t-5}` (AR lags), and
- the **2** previous prediction errors `e_{t-1..t-2}` (MA lags).

`p = 5`, `q = 2` are the defaults; both exposed as XSD knobs.

**Layer shape** (B-shaped, per-row buffers):

- `_s_history`: ring of last `p` sentence reps,
  `[B, p, sentence_dim]`.
- `_e_history`: ring of last `q` prediction errors,
  `[B, q, sentence_dim]`.
- `predictor`: `nn.Sequential(Linear(p*D + q*D, hidden), tanh,
  Linear(hidden, D))`.

**Forward / loss flow (per sentence)**:

1. Capture `s_t` from the S-tier root slot at sentence end (after
   the body finishes its T stages).
2. If the predictor has been primed (i.e. `_s_history` has at least
   one real entry), compute `s_hat_t = predictor(concat(_s_history,
   _e_history))`. Accumulate `loss_arma = MSE(s_hat_t, s_t)` into
   `TheError` under category `"arma"` with weight
   `(1 - reconstructionScale)` (or its own knob — see "Open
   questions").
3. Compute `e_t = s_t - s_hat_t` (the new MA residual).
4. Shift `_s_history` left and append `s_t`; shift `_e_history` left
   and append `e_t`.
5. At hard sentence-boundary / discourse-boundary reset, clear both
   buffers (reuse the existing `InterSentenceLayer.Reset()` if
   present, else add it).

**Retire** the contrastive cosine machinery once the ARMA loss
trains comparably: `context_window`, `centroid_history`, `lam`,
the dual-force loss. Keep the `_e_history` ring (functionally
parallel to the old `centroid_history`).

**XSD additions** (under `WordSpace`):

```xml
<xs:element name="armaP" type="xs:positiveInteger" minOccurs="0"/>
<xs:element name="armaQ" type="xs:nonNegativeInteger" minOccurs="0"/>
<xs:element name="armaHiddenDim" type="xs:positiveInteger" minOccurs="0"/>
```

Defaults: `armaP=5`, `armaQ=2`, `armaHiddenDim=2*sentence_dim`.

**Retire**:

- `<xs:element name="contextWindow" ...>` and
  `<xs:element name="centroidHistory" ...>` (if present in the
  current XSD).

**Inference (chat-style sentence generation)**:

1. Read `_s_history` / `_e_history` for the row.
2. Predict next sentence rep `s_hat_{t+1} = predictor(...)`.
3. Seed the body's IR-style forward at the C/S boundary with the
   predicted rep (lift `s_hat_{t+1}` through `sigma_percept` to
   C-tier as the "expected concept" prior), then sample tokens via
   the IR head's mask-position predictions iteratively (one mask +
   predict step per token; advance the perceptual buffer's
   non-mask positions with previously-sampled tokens).
4. Once the sentence terminates (`\0` / EOS / max-len), pool the
   produced S-tier root into `s_{t+1}` and commit to the history.

The seeding step is the one design choice that this plan defers to
implementation — concretely, the body's `_forward_per_stage_no_unfold`
needs an optional "C-prior" argument that, when present, is summed
into the C-tier event before the codebook lookup. The cleanest
hook is to add it as an attribute on `conceptualSpace` (set by the
inference loop before each forward, cleared after).

## Implementation tasks

In order (each is independently testable):

### Phase 1 — Retire `<maskedPrediction>` (§1)

1. **XSD cleanup**: delete `maskedPredictionEnum` + its two
   `<xs:element>` uses; validate `MM_5M_bivector.xml` against the
   new XSD.
2. **Strip `<maskedPrediction>` from XMLs**: audit all `data/*.xml`,
   remove every `<maskedPrediction>...</maskedPrediction>`.
3. **`bin/Models.py`**:
   - Delete `self.masked_prediction` field + constructor parameter.
   - Delete the AR/IR dispatch branch in `_forward_per_stage`; route
     all non-runtime calls through `_forward_per_stage_no_unfold`
     (now IR-only).
   - Slim `_forward_per_stage_no_unfold` to the IR single-shot path
     (delete the K-loop, causal mask, `head_outputs.append(...)`
     accumulation, `mask_n` construction).
   - Delete `_forward_stem_per_word_serial`.
   - Slim `runBatch`: single training branch. Loss = head output +
     optional C/S reconstruction per `<reconstruct>`.
4. **`bin/Spaces.py`**:
   - Remove the `if not is_ar` early-return in `InputSpace.forward`;
     all paths emit `[B, N, D]`.
   - Delete the ARIR-runtime unfold residue.
   - Audit `is_runtime_arir`; retire if dead.
5. **`bin/data.py`**: audit `_runtime_mode == 'ARIR'` references;
   retire the attribute if no non-AR caller uses it.

### Phase 2 — Retire `<reconstruct>=output` (§2)

6. **XSD**: remove `output` from `reconstructEnum`.
7. **`bin/Models.py`**:
   - Remove `rc == "OUTPUT"` branches in `runBatch`.
   - Delete `_reverse_per_stage`, `_run_pipeline_rev`,
     `_reverse_body`, `_reverse_head`, `_reverse_perceptual`,
     `_chart_generate_from_stm`, `_pair_unmerge`,
     `_should_reconstruct`, `_run_reverse_sequential`, and
     `BasicModel.reverse`.
8. **`bin/Language.py`**: audit chart-side `generate` (outside pass);
   retire if only called by the deleted reverse path.

### Phase 3 — Rename `<reverseScale>` → `<reconstructionScale>` (§3)

9. **XSD**: rename element + update comments.
10. **`bin/Models.py`** and **`bin/Layers.py`**: rename
    `reverse_scale` → `reconstruction_scale` and
    `self.loss.reverse_scale` → `self.loss.reconstruction_scale`
    everywhere (11 sites).
11. **`data/*.xml`**: rename `<reverseScale>` →
    `<reconstructionScale>` across all configs.
12. **Parser back-compat**: in `bin/util.py:TheXMLConfig` (or wherever
    `training.reverseScale` is read), accept the legacy name as a
    fallback with a one-line deprecation warning. Remove the shim
    next release.

### Phase 4 — Rewrite `InterSentenceLayer` as ARMA(5, 2) (§4)

13. **`bin/Layers.py:InterSentenceLayer`**:
    - Replace the contrastive-cosine machinery with the ARMA
      predictor described in §4.
    - Add `_s_history`, `_e_history` buffers; size them from
      `armaP`, `armaQ` XSD knobs.
    - Add the MLP predictor module.
    - Implement `observe(s_t)` (push `s_t`, compute and accumulate
      loss, update buffers).
    - Implement `predict_next()` (return `s_hat_{t+1}` without
      committing — for inference).
    - Implement `Reset()` (clear both rings on hard / discourse
      boundary).
14. **XSD**: add `<armaP>`, `<armaQ>`, `<armaHiddenDim>` under
    `wordSpaceType`. Remove `<contextWindow>`, `<centroidHistory>`,
    `<lam>` if present.
15. **`bin/Language.py:_attach_discourse`** (or wherever
    `InterSentenceLayer` is instantiated): read the new knobs and
    pass to the constructor.
16. **`bin/Models.py:runBatch`**:
    - After the body finishes each sentence, capture `s_t` from the
      S-tier root slot and call
      `self.wordSpace.discourse.observe(s_t)`.
    - The ARMA loss is added to `TheError` by `observe` itself
      (category `"arma"`).
17. **Chat-loop inference**:
    - Build (or extend) a `generate_sentence(...)` entry point on
      `BasicModel` that:
      a. Calls `self.wordSpace.discourse.predict_next()` for the
         next sentence rep prior.
      b. Sets `self.conceptualSpace._c_prior` to the lifted prior.
      c. Runs the IR forward iteratively, sampling tokens at masked
         positions until EOS / max-len.
      d. Captures the produced sentence's S-tier root and calls
         `discourse.observe(s_t)` to commit.
    - Add an optional `_c_prior` attribute on `ConceptualSpace` that,
      when set, is added to the C-tier event before the codebook
      lookup; cleared after each forward.

### Phase 5 — Tests, docs, XML migration audit (§§1–4)

18. **Retire tests** that exercise the deleted reverse pipeline:
    - `test_mm_xor.py::TestMMXorConvergence::test_forward_reverse_reconstructs_input_state`
    - `test_mm_xor.py::TestMMXorConvergence::test_runbatch_losses_stay_finite`
    - `test_mm_xor.py::TestMMXorConvergence::test_vqvae_ste_registers_commitment_and_moves_encoder`
    - `test_phase2_sequential_integration.py::test_sequential_reconstruction_produced`
    - `test_invertibility.py::TestPerceptualSpaceReverseRangeCheck::test_roundtrip_output_in_range`
    - `test_invertibility.py::TestConceptualSpaceReverseRangeCheck::test_nonlinear_extreme_input_bounded`
19. **Update tests** that construct a model with
    `masked_prediction='AR'` (or other AR values) — remove the
    parameter or replace with the new defaults.
20. **Add tests**:
    - XSD validation rejects `<maskedPrediction>...</maskedPrediction>`.
    - XSD validation rejects `<reconstruct>output</reconstruct>`.
    - IR forward produces `[B, N, predDim]`.
    - `<reconstruct>concepts</reconstruct>` adds a C-tier
      reconstruction loss term to `TheError`.
    - `<reconstructionScale>0.5</reconstructionScale>` parses; the
      legacy `<reverseScale>` triggers a deprecation warning and
      maps to the same field.
    - `InterSentenceLayer.predict_next()` returns the right shape
      after `armaP` observations.
    - `InterSentenceLayer.observe(s_t)` accumulates a non-zero ARMA
      loss after `armaP` steps.
    - `BasicModel.generate_sentence(...)` produces a non-empty
      sentence after one warm-up sentence.
21. **Docs**:
    - Update `doc/Spaces.md` §"AR cursor unfold retirement" to note
      that within-sentence AR is also retired; the section now
      describes only the IR single-shot pipeline.
    - Update `doc/Training.md` (if it documents
      `maskedPrediction`) to remove the enum and describe the new
      IR-only contract.
    - Add a "Sentence-level AR (`InterSentenceLayer`)" section to
      `doc/Architecture.md` describing the ARMA(5, 2) design.
    - Update any doc that mentions `<reverseScale>` to use the new
      name.

## XML config migration

After Phase 1–3, `MM_5M_bivector.xml`'s training section looks like:

```xml
<training>
  <numEpochs>1</numEpochs>
  <batchSize>16</batchSize>
  <numWorkers>4</numWorkers>
  <learningRate>0.0005</learningRate>
  <maskRate>0.15</maskRate>
  <reconstructionScale>0.5</reconstructionScale>   <!-- renamed -->
  <reconstruct>concepts</reconstruct>               <!-- replaces ARIR -->
  <trainEmbedding>JOINT</trainEmbedding>
  <!-- maskedPrediction retired -->
</training>
```

After Phase 4, the WordSpace section gains ARMA knobs:

```xml
<WordSpace>
  <armaP>5</armaP>
  <armaQ>2</armaQ>
  <armaHiddenDim>128</armaHiddenDim>     <!-- 2 * sentence_dim default -->
  <!-- contextWindow / centroidHistory / lam retired -->
</WordSpace>
```

## Verification criteria

- `pytest test/` passes (modulo the explicitly retired
  reverse-pipeline tests).
- `grep -rn "maskedPrediction\|masked_prediction\|_pair_unmerge\|_reverse_per_stage\|BasicModel.reverse\|reverse_scale" bin/ data/`
  returns only docstrings / comments + the back-compat shim.
- `_pair_unmerge` is no longer in `bin/Models.py`.
- `bin/train.py --model data/MM_5M_bivector.xml --data text --num-epochs 1 --batches 10`
  runs at **≥ 60 sent/sec** at `bs=128` with `max-autotune` (matching
  the current IR throughput).
- A model trained on `<reconstruct>concepts</reconstruct>` produces
  non-zero gradient on the C-tier `sigma_percept` weights.
- `InterSentenceLayer.observe(...)` accumulates a non-zero ARMA loss
  after `armaP=5` sentences.
- `BasicModel.generate_sentence(...)` produces a syntactically
  non-degenerate sentence (passes a basic length / EOS check).

## Open questions

- **Reconstruction-loss target shape**: when
  `<reconstruct>concepts</...>` is set, the target is the *original*
  C-tier state at masked positions. But the body's C-tier output is
  the *predicted* C-tier state. The pre-mask C-tier state isn't
  naturally available unless the forward also runs un-masked to
  capture it (doubling forward cost). Two options:
  - **A**: run a second forward un-masked to get the target. Doubles
    the body cost but gives a clean target.
  - **B**: use the P-tier `_ir_pre_mask_input` as the target and lift
    it through `sigma_percept` to C-tier on the fly. One forward,
    but the target is a derived approximation (the model's own
    sigma_percept applied to the pre-mask embedding).
  - Recommend **B** for speed; can revisit if reconstruction quality
    stalls. Identical question for `<reconstruct>symbols</...>`
    (lift through `sigma_percept` + SS pass-through).
- **ARMA loss weight**: should the ARMA loss share the
  `reconstructionScale` blend, or get its own knob
  (`<armaScale>`)? Recommend its own knob — the ARMA is a
  sentence-level signal and shouldn't be entangled with the within-
  sentence reconstruction blend.
- **Sentence-rep pooling**: which S-tier slot is the sentence rep?
  The root cell of the parse (after the start-symbol reduction) is
  the natural choice; the chart already tracks this on
  `WordSpace.current_rules`. Alternatively, the final-stage
  `_ss_cache[-1]` event pooled by mean/max. Decide at
  implementation time; the test in Phase 5 checks that the
  selected pool produces a stable rep across forwards on the same
  input.
- **`InterSentenceLayer` reset boundary**: the current contrastive
  loss resets on document boundary. For ARMA, we want the AR lags
  to carry information *across* document boundary if the discourse
  is continuous. Define a `<discourseBoundary>` semantic separate
  from the per-document reset, or make `armaP=5` lags reach back
  through document boundaries by default. Recommend the latter
  (lags are content-bearing; resets erase prior context which is
  fine for cosine but harmful for AR).
- **Chat-loop seeding**: the body's IR forward needs a C-prior hook
  for sentence-level conditioning. The cleanest API is an attribute
  on `ConceptualSpace` (`_c_prior`) added in the forward path; the
  inference loop sets it before each sentence and clears it after.
  Confirm at implementation time that this doesn't leak across
  training forwards (training never sets `_c_prior`, so the default
  `None` keeps it inert).

## Estimated effort

| Phase | Hours |
|---|---|
| 1. Retire `<maskedPrediction>` | 2-3 |
| 2. Retire `<reconstruct>=output` | 1-2 |
| 3. Rename `<reverseScale>` | 0.5 |
| 4. Rewrite `InterSentenceLayer` as ARMA(5, 2) | 3-4 |
| 5. Tests, docs, XML migration audit | 2 |
| Regression cycle (full pytest + benchmark) | 1 |
| **Total** | **~10-12 hours** |

## Files touched (predicted)

| File | Changes |
|---|---|
| `data/model.xsd` | retire `maskedPredictionEnum`, drop `output` from `reconstructEnum`, rename `reverseScale` element, add ARMA knobs |
| `data/MM_5M_bivector.xml` | retire `<maskedPrediction>`, rename `<reverseScale>`, add ARMA knobs |
| `data/*.xml` | audit & migrate for `<maskedPrediction>`, `<reverseScale>` rename |
| `bin/Models.py` | large refactor: delete reverse methods, slim `_forward_per_stage_no_unfold`, slim `runBatch`, rename `reverse_scale`, wire ARMA `observe` / `predict_next` |
| `bin/Spaces.py` | remove `is_ar` branch in `InputSpace.forward`; add optional `ConceptualSpace._c_prior` for chat-loop seeding |
| `bin/Layers.py` | rename `ModelLoss.reverse_scale`; rewrite `InterSentenceLayer` to ARMA(5, 2) |
| `bin/Language.py` | audit chart.generate retirement; update `_attach_discourse` for new ARMA knobs |
| `bin/util.py` | parser back-compat shim for `<reverseScale>` |
| `bin/data.py` | audit `_runtime_mode` |
| `test/test_mm_xor.py` | retire 3 reverse-pipeline tests |
| `test/test_phase2_sequential_integration.py` | retire 1 test |
| `test/test_invertibility.py` | retire 2 tests |
| `test/test_*.py` | audit for `masked_prediction` param |
| `test/test_intersentence_arma.py` (new) | ARMA observe / predict shape tests |
| `test/test_generate_sentence.py` (new) | chat-loop integration smoke |
| `doc/Spaces.md` | rewrite IR-only section |
| `doc/Training.md` | replace `maskedPrediction` doc with IR-only contract |
| `doc/Architecture.md` | add §"Sentence-level AR (`InterSentenceLayer`)" |
