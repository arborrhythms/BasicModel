# IR Mode: Sentence-Level Masked Input Reconstruction

**Date:** 2026-04-29
**Status:** Design spec, not yet implemented.
**Target config:** `data/MM_5M_IR.xml` (new)
**Prerequisites:**
- Mode rename (MLM→IR, ARLM→AR, RARLM removed) is in main, uncommitted.
- **Torus geometry landed (2026-04-29):** lexicon and SymbolicSpace codebooks
  now use periodic-wrap geometry on `[-1, 1)^D` with wrapped-MSE distance,
  unified at the `Basis` level via the `unit_ball` property (derived from
  `not use_dot_product`). ConceptualSpace stays on the sphere
  (`use_dot_product=True`). This was the lexicon-geometry follow-up; it
  landed inline rather than as a separate spec. See "Geometry contract"
  below.

## Context

`MM_5M` today trains in **ARIR** mode (autoregressive iterative reconstruction): per-position next-BPE-token prediction (~90% of loss via `lossOut`) plus reverse-pipeline reconstruction (~10% via `lossIn`, weighted by `<reverseScale>0.1</reverseScale>`). The model produces predicted *embedding vectors* at each position; nearest-neighbor lookup against the BPE codebook recovers a token sequence which decodes to bytes.

Two things are wrong with this for the goal of conversational generation:

1. **Predictions are mushy.** MSE-on-embedding training rewards "near the right semantic neighborhood" rather than "correct token identity." Decoded text reads as on-topic semantic-mush rather than coherent prose.
2. **Output unit is BPE-token, not word.** Even with a sharp prediction, the decoded BPE tokens have to be recombined into bytes and then re-segmented as words. The composition is lossy and the model never sees whole-word semantic structure during training.

**IR (Input Reconstruction)** is the sibling-of-ARIR mode that does sentence-level bidirectional masked-position reconstruction. The model is shown a sentence with ~15% of percept positions replaced by a NULL-percept token, and is trained to reconstruct the original percepts at those positions. Reconstruction loss applies *only at the masked positions* (BERT-MLM style) — sharp gradient signal on the prediction targets, no dilution from trivial copy-the-input behavior on visible positions.

IR is the non-AR sibling of ARIR. The rename of MLM→IR (already in main) makes this naming axis explicit:

| Suffix | Loss objective |
|---|---|
| LM | Predict at masked position (target = original percept; no reverse reconstruction) |
| US | Loss suppressed |
| IR | Reconstruct at masked position (uses reverse pipeline; loss only at masked positions) |

| Prefix | Mask pattern |
|---|---|
| (empty) | Bidirectional, no causal constraint |
| AR | Causal, monotonic left-to-right |

So `AR + LM = ARLM (renamed AR)`. `AR + IR = ARIR`. **`IR` (no prefix) = bidirectional, masked-position reconstruction.** That's this spec.

`MM_5M_IR` differs from `MM_5M` in three ways: prediction mode (IR vs ARIR), tokenization (word/lexicon vs byte+BPE), and lexicon size/dim (200K@D=6 vs 4096@D=6 — the torus geometry that makes 200K@D=6 viable already landed; see Prerequisites). This spec assumes the lexicon is already constructed — it does not prescribe SBOW retraining.

## Geometry contract

The lexicon (PerceptualSpace.vocabulary, an `Embedding`) and the SymbolicSpace codebook (a `Codebook`) both live on the *torus*: vectors are stored in `[-1, 1)^D` with periodic wrap; distance is `mean(wrapped_delta^2)` where `wrapped_delta(a, b) = ((a - b + 1) mod 2) - 1`. ConceptualSpace's codebook stays on the sphere (`use_dot_product=True`); high-D + low-V means it has no Tammes-packing pressure. Selection is automatic via the `Basis.unit_ball` property — IR doesn't need to set any geometry flags.

For IR specifically:
- Reverse-pipeline reconstruction is compared to the original (pre-mask) embedded input via wrapped MSE — the same metric used by codebook lookups during reverse, so the loss surface is consistent with the snap.
- NULL-percept slot is initialized with uniform random values in `[-1, 1)` (`_random_unit_ball` from `embed.py`), not L2-normalized.
- Codebook search at masked positions (during inference / generation) uses wrapped MSE on the torus.

## Mode contract

| | ARIR (MM_5M today) | IR (MM_5M_IR new) |
|---|---|---|
| `<maskedPrediction>` | `ARIR` | `IR` |
| Mask pattern (input corruption) | Causal: positions ahead of predict-cursor zeroed | Random: ~15% of positions replaced with NULL-percept |
| Output prediction loss (`lossOut`) | Trained, weight `1 - reverseScale` | **Disabled** (weight = 0) |
| Reverse reconstruction loss (`lossIn`) | Trained over all positions, weight `reverseScale` | **Trained only at masked positions**, weight 1.0 |
| Causal info flow | Yes (positions ahead are zeroed) | No (full bidirectional) |
| Inference / generation | AR: iterate by appending predictions | Parallel infill: NULL at target positions, one forward fills them |

The brick body, codebook lookup, reverse pipeline, and discourse layer are all unchanged structurally. IR adds a mask-injection step in the prep phase and modifies which positions contribute to `lossIn`.

## Configuration

New file: `data/MM_5M_IR.xml`. Identical to `data/MM_5M.xml` except for these fields (all under `<architecture>` or its sub-elements):

```xml
<!-- IR-specific changes from MM_5M.xml -->
<reconstruct>symbols</reconstruct>      <!-- ensure reverse pipeline runs; existing setting -->
<reverseScale>1.0</reverseScale>        <!-- was 0.1; reconstruction is now primary -->
<maskedPrediction>IR</maskedPrediction> <!-- was ARIR -->
<maskRate>0.15</maskRate>               <!-- new field, see "MaskRate" below -->

<InputSpace>
  <lexer>word</lexer>                   <!-- was "byte" -->
  ...
</InputSpace>

<PerceptualSpace>
  <chunking>none</chunking>             <!-- was "bpe"; word lexer feeds whole-word tokens -->
  <nVectors>200000</nVectors>           <!-- was 4096; targeting full word lexicon -->
  ...
</PerceptualSpace>
```

`<maskRate>` is a new XML field under `<architecture><training>`. Default 0.15 (BERT-style). Range [0.0, 1.0]. At 0.0 the model trains nothing (every position is visible, no masked positions, loss is empty). At 1.0 the model trains pure unconditional generation. Useful range is 0.1–0.5.

## NULL-percept

A dedicated codebook slot, separate from any real word/byte/BPE token. Reused across tokenizations (byte mode could use it; BPE mode could use it; for MM_5M_IR with word mode, it lives at the end of the word codebook).

**Storage.** Codebook size grows by 1: `<nVectors>200000</nVectors>` describes 200000 real word entries plus a 200001th NULL-percept slot. The slot's embedding is a learnable parameter, initialized via `_random_unit_ball((1, nDim))` (uniform draw in `[-1, 1)^D` — matches every other torus-codebook init). It receives gradient through the masking path during training and gets re-wrapped post-step alongside the rest of the lexicon.

**Distinction from other "null" concepts in the codebase.** The byte `\x00` codebook entry (used today in byte mode for end-of-content padding) is unrelated to NULL-percept. NULL-percept is a separate slot at index `nVectors` (i.e., one past the last real word). Byte `\x00` keeps its byte-character meaning and remains a valid prediction target.

**Why a separate slot, not reusing `\x00`.** The model needs to distinguish "this position was masked, predict here" from "this position originally contained character `\x00`." If they share an embedding, the loss can use a side-band mask tensor to distinguish them (mask says "credit me here" vs "don't"), but the *model's input* is identical for the two cases — the model can't condition on which positions are masked. With a separate slot, the model sees different embeddings and can learn that NULL-percept means "predict me" while `\x00` is just a character.

## Mask injection

Injected after embedding lookup, before the brick body consumes the embeddings. Per tick:

1. **Sample mask positions.** Generate `mask_positions: [B, K]` boolean. Sample by Bernoulli at rate `maskRate` per position. Excludes blank/padding positions if any (positions whose original embedding is the byte-`\x00` blank).

2. **Replace embeddings at masked positions.** `embedded[mask_positions] = NULL_PERCEPT_EMBED`. The shape of `embedded` is unchanged (`[B, K, D]`); only the content at masked positions becomes the NULL-percept embedding.

3. **Pass `mask_positions` to the loss path.** The mask is needed by the loss head to gate which positions contribute. Stash it on a per-batch attribute (e.g., `self._ir_mask_positions`) or pass as an extra arg through the runBatch chain.

4. **Brick proceeds normally.** Forward sees corrupted embeddings, produces symbols. Reverse runs, produces `inputDataPred: [B, K, D]`. The brick body is unchanged.

## Loss

```python
# Pseudocode for IR's lossIn computation (replaces the existing non-AR
# reverseible branch at bin/Models.py:2491-2504 when in IR mode)

if self.masked_prediction == 'IR':
    # Reverse pipeline already produced inputDataPred [B, K, D].
    # Compare to original (pre-mask-corruption) embeddings at masked
    # positions only.
    pred_at_masked = inputDataPred[mask_positions]      # [n_masked, D]
    target_at_masked = forwardInput_pre_mask[mask_positions]  # [n_masked, D]
    lossIn = self.loss.compute(pred_at_masked, target_at_masked)
    # Equivalent: F.mse_loss(pred_at_masked, target_at_masked)

    TheError.add("reconstruction", lossIn,
                 weight=1.0,
                 space="InputSpace", category="reconstruction")

    # No lossOut term in IR mode.
    lossOut = None
    output_weight = 0.0
```

`forwardInput_pre_mask` is the original embedded tensor *before* mask injection — captured at step 2 above. The reverse pipeline's reconstruction is compared against that pre-mask version, so the model is trained to recover the original content at NULL-percept positions.

`self.loss.compute` is MSE-on-embedding (the existing `lossIn` machinery). On the torus, "MSE-on-embedding" is implicitly **wrapped MSE** — the lexicon and reverse-pipeline output both live in `[-1, 1)^D`, so the residual `(pred - target)` is already in the wrapped cell and squaring it gives the same gradient signal as `_wrapped_mse_score`. No code change is needed at the loss site to honor the torus; the contract follows from where pred and target are sourced. A future variant could use cross-entropy over wrapped-MSE codebook-similarity logits for sharper categorical prediction; that's deferred.

## Inference / generation

IR has no separate inference mode — `train=False` (the existing runBatch parameter) handles loss/backward suppression. The generation pattern is determined by *how the user fills the input buffer*, not by a mode flag:

**Parallel infill.** User specifies which positions are NULL-percept. Run forward once. Read predictions at NULL positions. One brick forward = K parallel predictions.

**Sequential AR-style generation.** User starts with a prompt followed by NULL-percepts at all "future" positions. Run forward, sample the leftmost NULL's prediction, write that prediction back into the input (replacing the NULL), run forward again, etc. K passes for K output tokens.

**Beam search / mixed strategies.** Standard combinatorial generation patterns work over the same forward primitive.

The loss path doesn't fire during inference (`train=False` suppresses the entire backward path). The mask attribute can either be empty (the model produces predictions at all positions; the caller picks out the NULL positions externally) or set by the caller as a hint.

## Tokenization changes

**InputSpace.lexer:** `byte` → `word`. The word lexer at [bin/util.py:224](../../bin/util.py:224) currently uses:

```python
re.compile(r'[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]+|\s+')
```

This groups punctuation runs and whitespace runs into single tokens. For MM_5M_IR, change it to tokenize **non-letters individually**:

```python
re.compile(r'[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]|\s')
```

Letter runs stay grouped (one token per word). Each digit, each punctuation character, each whitespace character becomes its own token. So `"Hello, world!"` becomes `["Hello", ",", " ", "world", "!"]` (5 tokens).

This change is to `parse(text, lex='words')` — affects all word-mode consumers, not just IR. Confirm the existing word-mode consumers (XOR, MNIST, tomatoes via word lexer if applicable) tolerate the change. It's likely fine but should be verified — they'll see slightly more tokens for non-letter content.

**PerceptualSpace.chunking:** `bpe` → `none`. With word-mode lexer, the perceptual codebook holds whole-word entries (and individual non-letter characters). No BPE merge step.

**Vocabulary.** The word codebook needs to hold the union of: the top-N most-frequent words in the training data (covering ≥99% of text) plus all individual non-letter characters (digits, punctuation, whitespace). For fineweb-edu at V≈200K, this is ~199.5K word entries plus ~50 character entries. The pretrained `.kv` artifact for MM_5M_IR is a different file from `MM_5M.kv` (which is BPE) — needs to be built via Phase 1 SBOW training with the word-mode lexer.

## Implementation plan

The implementation is layered. Land each step end-to-end (verify it runs, verify it doesn't break existing tests) before moving to the next.

### Step 1: NULL-percept codebook slot

**File:** [bin/Spaces.py](../../bin/Spaces.py), the Embedding class around line 1488 (`def create`).

After the codebook is constructed at `nVectors` real entries, add one trailing slot for NULL-percept. The slot's embedding is a learnable parameter (created via the same machinery as other entries; init via `_random_unit_ball((1, nDim))` so it lands in `[-1, 1)^D` like every other lexicon entry, gradient-tracking).

Expose the NULL-percept index via `Embedding.null_percept_idx` (Python attribute = `self.nVectors`, the index of the trailing slot).

**Verify:** existing tests pass; the codebook tensor's shape is now `[nVectors+1, D]` instead of `[nVectors, D]`; downstream code that reads codebook size by `getW().shape[0]` correctly sees the +1.

### Step 2: Word-mode regex change (individual non-letters)

**File:** [bin/util.py:224](../../bin/util.py:224).

Change the regex to:
```python
_PARSE_WORD_RE = re.compile(r'[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]|\s')
```

**Verify:** add unit tests for the new tokenization. Confirm `"Hello, world!"` tokenizes to 5 tokens. Confirm existing word-lexer callers don't break.

### Step 3: `<maskRate>` XML field

**Files:** [data/model.xsd](../../data/model.xsd) (add `<maskRate>` to `trainingType`), [bin/Models.py](../../bin/Models.py) ModelFactory parsing (where other training XML fields are read).

Default 0.15. Range validation: `0.0 <= maskRate <= 1.0`.

**Verify:** existing configs without `<maskRate>` parse and run with default 0.15. New configs with `<maskRate>` use the specified value.

### Step 4: IR-mode dispatch in runBatch

**Files:** [bin/Models.py](../../bin/Models.py), specifically the AR-mode dispatch tuple around line 2380 (currently `('AR', 'ARUS', 'ARIR')` after rename) and the loss-construction branches.

Add `'IR'` to the set of modes that exercise the reverse pipeline. **Critical:** IR is **not** an AR mode — it should NOT go through the AR-window unfold step (the `tensor.unfold(1, N, 1)` that builds K progressive-prefix windows). IR uses the full sentence as one bidirectional input.

Specifically, route IR through the non-AR path: single forward, single backward, with mask injection added to the prep phase.

The dispatch logic looks roughly like:
```python
is_ar_mode = self.masked_prediction in ('AR', 'ARUS', 'ARIR')
is_ir_mode = self.masked_prediction == 'IR'
if is_ar_mode:
    # Existing AR path with unfold and per-position prediction
    ...
elif is_ir_mode:
    # IR path: full sentence, mask injection, reverse-only loss
    ...
else:
    # NONE or other: existing non-AR path
    ...
```

### Step 5: Mask injection step

**File:** [bin/Models.py](../../bin/Models.py), in `runEpoch`'s prep section (around line 3358 in the live code) when in IR mode.

After the input is lexed and embedded but before the brick body consumes it:

1. Capture `forwardInput_pre_mask` = the embedded tensor before any modification.
2. Sample `mask_positions: [B, K]` via `torch.bernoulli(torch.full([B, K], maskRate))` (cast to bool). Exclude blank-padding positions (where the original token was byte `\x00` or equivalent end-of-content marker — these aren't real prediction targets).
3. Replace embeddings at masked positions: `embedded[mask_positions] = null_percept_embedding`.
4. Stash `mask_positions` and `forwardInput_pre_mask` on a model attribute for the loss path to read (e.g., `self._ir_mask_positions`, `self._ir_forwardInput_pre_mask`). Clear them at end of runBatch.

### Step 6: IR loss path

**File:** [bin/Models.py](../../bin/Models.py), the loss-construction branches around lines 2491-2504.

Replace the current non-AR reversible branch with an IR-aware version: when `is_ir_mode`, compute `lossIn` only at masked positions (using stashed `mask_positions` and `forwardInput_pre_mask`), set `lossOut = None` and `output_weight = 0.0`. Existing TheError accumulator handles the rest.

Pseudocode at "Loss" section above. The reverse pipeline (`self.reverse(symbols, outputDataPred)`) is unchanged — it produces `inputDataPred: [B, K, D]` as today. The change is which positions of that prediction contribute to the loss.

### Step 7: Disable lossOut under IR

In IR mode, the AR/output prediction head shouldn't fire — it has no role under IR. Two options:

- **Skip the head computation entirely.** Add an `if not is_ir_mode:` guard around the `outputDataPred` computation and the `lossOut` calculation. Cleaner; saves FLOPs.
- **Compute but zero-weight.** Let the head run, set `output_weight = 0.0`. Slower but minimal code change.

Recommend the first option for clarity. The brick's output projection layer (the OutputSpace machinery) becomes vestigial in IR mode but stays wired for code-path symmetry; it just doesn't contribute to the loss.

### Step 8: Inference / generation primitive

The user-facing inference path needs a function that:

1. Takes user input bytes/text.
2. Lex → embed via the word-lexer pipeline.
3. Allocates a `[B=1, K, D]` buffer with the user's input occupying the leading positions and NULL-percept embeddings at all "to be predicted" positions.
4. Runs `runBatch(train=False)` on that buffer.
5. Returns predictions at the NULL positions, decoded back to words via nearest-neighbor lookup against the word codebook.

This is a new public method on `BasicModel` — call it `BasicModel.infer_ir()` or similar. Lives next to the existing `BasicModel.run_ar_inference()` (at [bin/Models.py:1951](../../bin/Models.py:1951)).

For the prototype, support only "parallel infill" — one forward, predictions at all NULL positions read out. AR-style sequential generation is a follow-up that iterates this primitive.

### Step 9: Build the MM_5M_IR.kv lexicon

The pretrained word-vector artifact for MM_5M_IR doesn't exist yet. It needs to be built via Phase 1 SBOW training with the word-mode lexer at the new vocabulary size and dimension.

**Note:** the lexicon's geometry (sphere vs ball, normalization choice) is the topic of the **lexicon-geometry follow-up spec**. For *this* IR spec, build the lexicon however SBOW currently does (sphere-normalized) and accept the geometry constraint. Once IR is functional, a separate experiment can A/B against ball-interior lexicons.

```bash
# Build the word lexicon (placeholder; actual command depends on Phase 1 entry point)
make train MODEL=data/MM_5M_IR.xml
```

This produces `data/MM_5M_IR.kv` with whatever V × D dimensions are configured in the XML.

### Step 10: End-to-end smoke

```bash
# Verify training runs
make train_micro MODEL=data/MM_5M_IR.xml

# Verify a small batch produces non-NaN losses
BASIC_MAX_BATCHES=10 make train_micro MODEL=data/MM_5M_IR.xml
```

Compare loss curves with `MM_5M.xml` (ARIR baseline) over a fixed compute budget. Document throughput (sentences/sec) and final loss in the handoff for whoever runs the comparison.

## Verification

### Functional tests

Add these tests to `test/test_ir_mode.py`:

1. **NULL-percept slot exists.** After codebook construction at `nVectors=N`, the codebook tensor has `N+1` rows; `embedding.null_percept_idx == N`.
2. **Mask injection produces correct masking.** Construct a small input, mask at 50% rate, verify ~50% of positions hold the NULL-percept embedding.
3. **Loss at masked positions only.** Compute IR loss with a known mask pattern, verify the loss equals MSE over only the masked positions.
4. **lossOut is suppressed.** Verify `result.lossOut is None` (or zero) when running in IR mode.
5. **AR(1) reduction.** With a causal mask (positions [k+1, K-1] all masked), verify the loss is dominated by predictions at the rightmost masked position — i.e., IR with causal masking trains a similar gradient surface to ARLM.

### Throughput / quality benchmarks (on metalbaby)

After step 10 lands:
- **Throughput:** `MODEL_PROF=1` runs of MM_5M_IR vs MM_5M. Measure `forward`, `backward`, `prep`, `dispatch` mean ms/batch over 32+ batch windows. Expect IR to be roughly comparable to ARIR per tick (same brick architecture, similar FLOPs).
- **Quality:** train both for the same number of optimizer steps, compare final reconstruction loss on a held-out validation set. Generate sample text from each via inference. Subjective + held-out cross-entropy.

## Open decisions

1. **Loss formulation: wrapped MSE vs cross-entropy on codebook similarity.** This spec uses wrapped MSE on the torus for v0. Cross-entropy over codebook-similarity logits (using `_wrapped_mse_score` as the logit source) is a natural follow-up if MSE-trained predictions are too soft.
2. **Mask sampling pattern: random per-position vs structured (T5-style spans).** This spec uses random per-position. T5-style span masking (mask contiguous runs) is a follow-up.
3. **Whether to keep the OutputSpace head in IR mode.** Spec recommends yes (kept but inert) for code-path symmetry; could remove for cleanliness later.
4. **Whether to expose the choice between "loss only at masked" and "loss everywhere weighted" as a config knob.** Spec hardcodes "only at masked" for v0; could parameterize via `<maskOnlyLoss>true</maskOnlyLoss>` if comparison becomes interesting.

## Pointers

- **Mode dispatch and loss construction:** [bin/Models.py:2380, 2491-2504](../../bin/Models.py).
- **Reverse pipeline:** `BasicModel.reverse(...)` (search "def reverse" in [bin/Models.py](../../bin/Models.py)).
- **Embedding / codebook:** [bin/Spaces.py:1437-1443](../../bin/Spaces.py) for `Embedding.getW()`; [bin/Spaces.py:1488](../../bin/Spaces.py) for `Embedding.create()`.
- **Word lexer regex:** [bin/util.py:224](../../bin/util.py:224).
- **Existing inference primitive (AR-style):** `BasicModel.run_ar_inference()` near [bin/Models.py:1951](../../bin/Models.py:1951) — follow this pattern for `infer_ir()`.
- **MM_5M baseline config:** [data/MM_5M.xml](../../data/MM_5M.xml).
- **Mode rename context:** the rename of MLM→IR, ARLM→AR, RARLM removal landed in main on 2026-04-29 (uncommitted at time of writing). The doc tables at [doc/Training.md:139](../../doc/Training.md:139) and [doc/Params.md:19](../../doc/Params.md:19) are already updated.
- **Historical context for prior MLM removal:** [doc/plans/2026-04-21-next-percept-decoupling.md](../plans/2026-04-21-next-percept-decoupling.md). The 2026-04-21 refactor removed MLM-specific dispatch (the `expand_masked` methods on InputSpace/OutputSpace). IR is a fresh implementation, not a restoration of those methods.

## Lexicon-geometry follow-up — DONE (2026-04-29, inline)

The geometry transition that this section originally deferred has landed. Summary of what shipped:

- **Lexicon (`PerceptualSpace.vocabulary` / `Embedding`)**: vectors are stored in `[-1, 1)^D` via periodic wrap; nearest-neighbor lookups use wrapped MSE (`_wrapped_mse_score`); save/load round-trip wraps idempotently. Legacy sphere `.kv` artifacts (norm ≈ 1, components in `[-1, 1]`) are accepted unchanged — wrap is a no-op on them.
- **SymbolicSpace codebook (`Codebook` with `use_dot_product=False`)**: same torus geometry. Codebook search via `Basis._snap_content` and `Basis.codebookDistance` dispatches on `unit_ball`; logical-op reverses (`conjunctionReverse`/`disjunctionReverse`/`lift`/`lower`) thread `unit_ball` through `Ops._binary_op_inverse_impl` so the codebook-pair search uses wrapped Euclidean distance on the torus.
- **ConceptualSpace codebook**: stays on the sphere (`use_dot_product=True`) — high-D, low-V, no Tammes pressure; cosine geometry preserves belief-magnitude semantics.
- **Wrap helpers** (`_wrap_unit_ball`, `_wrapped_mse_score`, `_random_unit_ball`) live in `bin/embed.py` at module scope and are imported by `Spaces.py`.

Files touched: `bin/embed.py`, `bin/Spaces.py`, `bin/Layers.py` (`Ops._binary_op_inverse_impl`, `Ops.conjunctionReverse`, `Ops.disjunctionReverse`), `data/model.xsd` (dropped `<unitBall>` field), `test/test_lexicon_ownership.py`. The `<unitBall>` XML knob was removed — geometry is automatic per-codebook via `use_dot_product`.

## Reproduction recipe

Once implementation is complete:

```bash
cd ~/WikiOracle/basicmodel

# Phase 1: build the word lexicon (rebuilds MM_5M_IR.kv from fineweb)
make train MODEL=data/MM_5M_IR.xml

# Phase 2: train the IR model
make train_micro MODEL=data/MM_5M_IR.xml

# Inference / chat (once infer_ir() is wired)
.venv/bin/python -c "
from bin.Models import BasicModel
m = BasicModel.from_config('data/MM_5M_IR.xml')
print(m.infer_ir('The quick brown fox '))
"
```

Throughput target: comparable to MM_5M baseline at K=1024, B=256. Quality target: visibly improved generation coherence over MM_5M's BPE-decode output, evaluated subjectively on chat prompts plus a held-out reconstruction-loss benchmark.
