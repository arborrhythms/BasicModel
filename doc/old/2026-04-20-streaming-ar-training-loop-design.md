# Streaming AR Training Loop

**Date:** 2026-04-20
**Status:** design (approved, ready for implementation plan)

## Goal

Speed up AR-mode training (`ARLM`, `ARUS`, `ARIR`) by replacing the per-position
mask-and-rerun loop with a single left-to-right pass inside
`MentalModel.forward()`. ARLM becomes forward-only; ARIR does one terminal
reverse for reconstruction.

## Current Behavior

`runEpoch()` (Models.py:2217-2272) drives an AR-specific loop:

```
for pos in range(N):
    masked = expand_masked_batched(embedded, sentences, mode, pos)
    runBatch(batch_override=(step_input, targets))   # full fwd + reverse + backward + step
```

Per DataLoader yield with B sentences × N positions:
- N full `MentalModel.forward()` + `reverse()` calls
- N `backward()` + `optimizer.step()` calls

The inner `MentalModel.forward()` (Models.py:2815) already has a
`for t in range(conceptualOrder)` loop; it runs in full each time runBatch
is called.

## Target Behavior

Per DataLoader yield:
- One `model.forward()` call containing two nested loops: outer
  `for pos in range(N)` (words, new), inner `for t in range(conceptualOrder)`
  (grammar production, existing).
- Per-pos prediction emitted inside the words loop via `Finish()`.
- One terminal `reverse()` outside both loops when
  `masked_prediction == 'ARIR'` (i.e. the mode that trains reconstruction).
  ARLM/ARUS skip reverse regardless of `<reconstruct>`.
- One `backward()` + `optimizer.step()` per yield (in `runBatch`).

Grammar/symbol state carries across the outer pos loop within a single
forward call: `self.percepts`, `self.concepts`, `self.symbols`, `symbolic_state`,
`sym_feedback`, and each `subspace.event` persist from pos=t to pos=t+1.
State is reset at the top of each `forward()` via a new `Start()` cascade.

## Mode Contract

| Mode   | Per-pos output prediction | Terminal reverse | Reconstruction loss |
|--------|---------------------------|-------------------------------|------------------|
| `ARLM` | Yes (next-word MSE)       | No (ignores `<reconstruct>`)  | Not trained      |
| `ARUS` | No (no output term)       | No (ignores `<reconstruct>`)  | Not trained      |
| `ARIR` | Yes (next-word MSE)       | Yes                           | Over [B, N, D]   |

Changes from today:
- ARLM never triggers reverse and never adds a reconstruction term to
  the loss, regardless of `<reconstruct>` config. `<reconstruct>` is
  unconstrained under ARLM — no error if it is `symbols`, `output`, or
  `both`; the setting is simply inert.
- ARIR becomes a valid training `maskedPrediction` value (today it is a
  runtime-only inference mode). Training ARIR = ARLM forward pass + one
  terminal reverse. ARIR requires `<reconstruct> != NONE` at validation
  (otherwise the mode is meaningless).
- `reverse_scale` is a weight on the reconstruction term; it only
  contributes under ARIR. Under ARLM/ARUS it is inert (no validation).

ARIR's pre-existing runtime/inference state machine coexists with the
new AR training path. When `data._runtime_mode == 'ARIR'` is set (by
`infer(mode='ARIR')` / `TheData.runtime_batch(..., mode='ARIR')`), the
outer pos loop is **skipped** and the single-pass non-AR path runs
instead. The state machine's cache-injection mechanism
(`InputSpace._cached_embedding`) remains functional: it bypasses re-lex
in `InputSpace.forward()` and stages the mask-at-cursor embedding
directly, as it did before this refactor. This avoids double-sliding
the same tokens into the persistent AR buffer during inference.

## Architecture Changes

### `MentalModel.forward(inputData)` — new structure

```python
def forward(self, inputData):
    self.Start(inputData)                     # cascade reset; embed once
    embedded = self.inputs.materialize()      # [B, N_max, D]

    predictions = []
    for pos in range(N):
        expand_masked_batched(embedded, inp_items, self.masked_prediction, pos)
        # AR(1) masking: reveals embedded[:, pos, :], zeros future (>pos).
        # Returns [B, nVec, embSize] -- does NOT expand batch.

        for t in range(self.conceptualOrder):
            # existing inner body (Models.py:2815-2876): sigma, pi, feedback.
            # Reads self.percepts / self.concepts / self.symbols / ss
            # carried from previous pos.
            ...

        pred = self.Finish(self.symbols[:, :self.nOutputSymbols, :])
        predictions.append(pred)

    if self.masked_prediction == 'ARIR':
        reconstruction = self.reverse(self.symbols, ...)   # one-shot
    else:
        reconstruction = None

    return forwardInput, self.symbols, predictions, reconstruction
```

Return signature gains a `predictions` list (stackable to `[B, N, outDim]`)
and a `reconstruction` tensor (or `None`). runBatch consumes both.

### `Start()` cascade (new)

```
MentalModel.Start(inputData)
    for space in self.spaces:
        space.Start()
            for layer in self.layers:
                layer.Start()
            self.subspace.reset_event()
    self.symbol_states = []
    self._nonrams_sym_feedbacks = []
    self.symbolic_state = self.symbolicSpace.empty_state(batch=B)   # renamed from ``ss``
    self.inputs = self.inputSpace.forward(inputData)     # embed once
```

`MentalModel` and `BasicModel` both extend `BaseModel` (not each other).
`MentalModel.Start()` is new — defined directly on `MentalModel`. Its
semantics are reset + embed only. Downstream pipeline execution moves
into the outer pos loop.

`BasicModel.Start()` keeps its current semantics (reset + full-pipeline
pass) and does not use the new cascade.

`Space` and `Layer` base classes gain `Start()` methods (empty default is
acceptable). For layers that currently have `reset()` (Layers.py:3336,
4253, 4507, 4609), `Start()` calls the existing `reset()` body. The new
cascade is invoked only from `MentalModel.Start()`; `BasicModel.Start()`
continues to call `ws.clear_sentence()` and
`symbolicSpace.reset_symbol_objective()` directly as today.

### Outer pos loop mechanics

**Sliding-window buffer, cross-sentence persistent.** `InputSpace` owns a
persistent buffer of shape `[B, nActive, embSize]`, initialized to zeros.
Tokens enter the buffer at the **rightmost slot** and older tokens fall
off the left as the buffer slides.

- At pos=t:
  1. Run the pipeline on the current buffer state (pre-slide).
  2. `Finish()` emits the per-pos prediction against target
     `embedded[:, t, :]` (the token we're about to slide in — the model
     predicts what it is about to see, given prior context).
  3. Slide the buffer left by one slot; append `embedded[:, t, :]` at
     the rightmost slot. Ready for pos=t+1.
- Buffer does **not** reset at sentence boundaries. The last tokens of
  sentence k remain visible during sentence k+1, providing cross-sentence
  context. A new explicit `InputSpace.reset_buffer()` method resets it at
  epoch boundaries (or when the user wants a clean slate).
- When `nActive > sentence length`, earlier sentences contribute context.
  When `nActive < sentence length`, early tokens of the current sentence
  fall off the left — the model sees only the trailing `nActive` tokens.
- `expand_masked_batched` is no longer called by the training path; the
  method remains for ad-hoc use but the per-pos masking logic is not the
  training-time mechanism.
- Inner conceptualOrder loop body is unchanged.
- `self.percepts / self.concepts / self.symbols / symbolic_state /
  sym_feedback` persist across pos iterations — their values at the start
  of pos=t+1 are whatever the inner loop left them at the end of pos=t.
- `Finish()` emits the per-pos prediction, appended to `predictions`.

### Terminal `reverse()` (ARIR only)

Called once after the pos loop completes, operating on the final
`self.symbols`. Produces a `[B, N, D]` reconstruction. Signature and
internal logic unchanged — only the call site moves.

### `runEpoch()` — branch deleted

Lines 2217-2272 (the `if masked_pred and text_batch:` AR branch) are
removed. runEpoch becomes:

```python
for inp_items, out_items in loader:
    inputTensor = self.inputSpace.prepInput(inp_items)
    outputTensor = self.outputSpace.prepOutput(out_items)
    result, _ = self.runBatch(
        train=training, batchNum=step, batchSize=B, split=split,
        optimizer=optimizer,
        batch_override=(inputTensor, outputTensor),
    )
    if result is not None:
        record(result)
    step += 1
```

AR and non-AR yields now flow through the same path.

### `runBatch()` — AR handling

Shape unchanged: one forward + loss + backward + step per yield.
Loss computation uses `TheError.compute()` where shapes fit:

```python
TheError.reset()
TheError.attach(self.loss)

forwardInput, symbols, predictions, reconstruction = self.forward(inputTensor)

output_weight = (1 - self.loss.reverse_scale) if self.masked_prediction == 'ARIR' else 1.0

if self.masked_prediction != 'ARUS':
    pred_stack = torch.stack(predictions, dim=1)         # [B, N, outDim]
    target_stack = _targets_per_pos(outputTensor, inp_items)
    TheError.compute("output", pred_stack, target_stack,
                     method="output", weight=output_weight,
                     space="OutputSpace", category="prediction")

if reconstruction is not None:        # only under ARIR
    TheError.compute("reconstruction", reconstruction, embedded_target,
                     method="compute", weight=self.loss.reverse_scale,
                     space="InputSpace", category="reconstruction")

# Other terms (symbol_objective, sbow, discourse, truth) stay on TheError.add().

TheError.snapshot()
totalLoss = TheError.total()
totalLoss.backward()
optimizer.step()
```

### `expand_masked_batched()` — in-place call site move

Method itself is unchanged. It:
- Returns `[B, nVec, embSize]` (no batch expansion).
- Masks position `pos` and zeros future positions for ARLM/ARUS.

The call site moves from `runEpoch` (today) into the outer pos loop of
`MentalModel.forward()`.

## What Gets Deleted

- The AR branch in `runEpoch` (Models.py:2217-2272).
- The per-position cached-embedding bypass machinery on `InputSpace`
  (`_cached_embedding`, `_unmasked_embedding`, `_mask_positions`, and the
  `batch_override` rewiring that reads them).
- Any reverse triggered under ARLM when `reversible=true` (ARLM is now
  forward-only; reverse path is ARIR-only).

## What Stays

- `BasicModel` (non-AR model) — semantics unchanged. `BasicModel.Start()`
  keeps its current reset + pipeline-pass body. `BasicModel.forward()` is
  minimally updated to return a 4-tuple
  `(input, symbols, outputData, None)` so the shared `runBatch` can
  unpack uniformly with `MentalModel.forward`; behaviour is unchanged
  and the fourth slot is always `None` for BasicModel.
- Butterflies path — unchanged. Still mutually exclusive with useGrammar.
  Butterflies do not use masked_prediction modes in the same way; this
  refactor applies to the grammar/flat paths.
- `MentalModel.reverse()` — signature and logic unchanged; only the call
  site moves.
- `Finish()`, `StartReverse()`, `FinishReverse()` — unchanged.
- `runTrial`, `run` — unchanged.
- Other loss terms (symbol objective, sbow, discourse, truth-modulated)
  stay on `TheError.add()` since their shapes don't fit `TheError.compute()`'s
  pred/target interface.

## Config Validation

New rule added to the existing `TheXMLConfig.require(...)` checks:

- `maskedPrediction == 'ARIR'` requires `reconstruct != 'NONE'`.

No rule constrains `<reconstruct>` under ARLM or ARUS; the setting is
inert in those modes.

## Testing

- Unit test: `MentalModel.forward()` called with a fixed 3-token batch
  returns `len(predictions) == N` and shapes match expected
  `[B, outDim]` per entry.
- Unit test: after forward, `self.percepts.materialize()` at pos=N-1
  differs from its value at pos=0 (confirming state carryover).
- Regression: `test_mm_xor.py` still passes (butterflies untouched).
- Regression: `test_use_flags.py` still passes (config flags unchanged).
- New: ARIR + `reconstruct=NONE` config is rejected at model build.
- New: ARIR training run produces non-None `reconstruction` output; ARLM
  training run produces `reconstruction=None` regardless of `<reconstruct>`
  config value.
- Perf: training time per epoch on FineWeb-EDU drops measurably
  (target: ~Nx reduction in reverse/backward count).

## Non-Goals

- No architectural change to butterflies or flat (non-grammar) paths
  beyond what the new Start() cascade requires.
- No change to inference (`infer()`) paths.
- No change to the `TruthLayer`, `DiscourseSpace`, or embedding training.
- RARLM is out of scope for this refactor.
