# Next-Percept Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sever three tangled responsibilities currently bound up in `InputSpace`: dataset batching, MLM-style input masking, and the lexicon-access shim. After this refactor, `InputSpace` is a pure tokenize+stage primitive; `TheData` owns batching; `PerceptualSpace` owns the lexicon; and training runs a unified "predict next percept" objective with no MLM fallback path.

**Architecture:** The refactor is a top-down deletion sweep. Each task removes one layer of accidental coupling and relocates exactly one external caller. MLM is dropped entirely (it was only exercised through `InputSpace.getBatch`'s masked-mode branch and direct test callers). The AR modes (ARLM, ARUS, ARIR) already flow through the sliding-window `_ar_buffer` path and are unaffected by MLM removal. ARIR's stateful inference loop is relocated from a hidden branch of `getBatch` into an explicit public method so callers stop pretending it's a generic batch fetcher. The four InputSpace lexicon delegators (`predict`, `embed_token`, `get_space_embedding`, `get_mask_embedding`) and their `_peer_embedding`/`_peer_perceptual` back-pointers come out together; their one external caller is rewired to reach `perceptualSpace.vocabulary` directly.

**Tech Stack:** Python 3.12, PyTorch, pytest. Run commands using `basicmodel/.venv/bin/python` (never system python3.12). Working directory: `/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel`. All file paths below are relative to that working directory.

---

## Preconditions

- Clean working tree on `main` (or a feature branch cut from main).
- `.venv/bin/python -m pytest test/ -q` currently passes 842 tests.
- User manages git commits manually. **Do not `git commit` without explicit user permission.** Stage changes with `git add` at the end of each task and hand back to the user.

## File Structure (what each task touches)

| File | What changes |
|------|--------------|
| `bin/Spaces.py` | Delete `InputSpace.getBatch`, `expand_masked`, `expand_masked_batched`, `_unmasked_embedding`, `_mask_positions`, `_peer_embedding`, `_lexicon()`, `predict`, `embed_token`, `get_space_embedding`, `get_mask_embedding`. Also delete `OutputSpace.expand_masked`. Rename `_getBatch_arir` → `arir_step` (public). Keep `_cached_embedding` and its `forward` bypass branch — still used by `arir_step`. Keep `_peer_perceptual` — back-reference used by `arir_step` internals. |
| `bin/Models.py` | Remove `_peer_embedding`/`_peer_perceptual` setattr calls in both model constructors. Rewire `BasicModel.run_ar_inference` to call `perceptualSpace.vocabulary.predict(...)` instead of `self.inputSpace.predict(...)`. Update `runBatch` to call `inputSpace.arir_step` in the ARIR inference branch. Simplify `output_weight` computation (MLM branch removed). Remove `masked_pred` local gate on MLM — keep only the AR-mode branch. |
| `bin/data.py` | Delete the `self.masked_prediction = 'MLM'` auto-assignment (dead after MLM removal). |
| `bin/visualize.py` | Remove any residual `masked_prediction == 'MLM'` references (research report flagged one). |
| `test/test_basicmodel.py` | Delete `TestMLM` class and any tests that assert on MLM-specific behavior. Update direct `inputSpace.getBatch` callers (lines 2750, 2391, 2400, 2409) to pull from `TheData.data_loader` + `prepInput`. |
| `test/test_mm_xor.py` | Replace 7 `inputSpace.getBatch` call sites (lines 94, 109, 130, 244, 291, 444) with `TheData.data_loader` + `prepInput`. |
| `test/test_stream_dataset.py` | Delete `expand_masked_batched` tests (lines 167–270 per research). |
| `test/test_streaming_ar_training.py` | No code changes required (AR modes unaffected) — verify tests still pass. |
| `test/test_head_divergence.py` | Replace single `inputSpace.getBatch(0, 1, "runtime")` call (line 79) with `inputSpace.arir_step(0)`. |
| `test/test_xor_spaces.py` | Replace 2 `inputSpace.getBatch` calls (lines 198, 209). |
| `test/test_testpoint.py` | Replace 1 `inputSpace.getBatch` call (line 295). |
| `test/diag_where.py` | Replace 1 `inputSpace.getBatch` call (line 30). |

## Testing Commands (used throughout)

- Full suite: `.venv/bin/python -m pytest test/ -q` — expected 842 tests before refactor, ~820–840 after (MLM tests are deleted).
- Single test: `.venv/bin/python -m pytest test/test_basicmodel.py::TestMLM -v` (pre-refactor sanity) / `.venv/bin/python -m pytest test/test_mm_xor.py -v` (per-task verification).

---

## Task 1: Delete MLM path from `InputSpace.getBatch` and kill `expand_masked`

**Context:** MLM is the masked-language-model training mode. Its only invocation site is `InputSpace.getBatch`'s masked-mode branch at `bin/Spaces.py:4328-4363`, which was only reachable through direct test calls. `runEpoch` always passes `batch_override` (it pulls from `TheData.data_loader` directly), so production training never exercised this path. Deleting it is safe behavior-wise for training. We also delete the two `expand_masked` methods (on `InputSpace` and `OutputSpace`) and `expand_masked_batched` since they have no other callers after this task.

**Files:**
- Modify: `bin/Spaces.py` (delete ~85 lines in `InputSpace.getBatch` masked branch; delete `expand_masked` at ~4080-4145; delete `expand_masked_batched` at ~4147-4264; delete `OutputSpace.expand_masked` at ~6470-6500; delete `_unmasked_embedding` and `_mask_positions` attribute init in `InputSpace.__init__`).
- Modify: `bin/Models.py` (simplify `runBatch` `output_weight` computation; remove `masked_pred` local).
- Modify: `bin/data.py` line 594 (`self.masked_prediction = 'MLM'` auto-assignment — delete).
- Modify: `bin/visualize.py` — find and delete residual MLM reference.
- Delete tests: `test/test_basicmodel.py` `TestMLM` class; `test/test_stream_dataset.py` `expand_masked_batched` tests.

- [ ] **Step 1: Run the full suite to establish baseline**

Run: `.venv/bin/python -m pytest test/ -q`
Expected: `842 passed, 14 skipped, 11 xfailed, 1 xpassed, 11 subtests passed`

If any test is already failing, stop and escalate.

- [ ] **Step 2: Write a failing test that MLM is gone**

Add to `test/test_lexicon_ownership.py` (file already exists):

```python
def test_inputspace_expand_masked_is_gone():
    """MLM removal: InputSpace no longer exposes expand_masked."""
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'expand_masked'), \
        "InputSpace.expand_masked should be deleted (MLM removed)"

def test_outputspace_expand_masked_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.OutputSpace, 'expand_masked'), \
        "OutputSpace.expand_masked should be deleted (MLM removed)"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_expand_masked_is_gone test/test_lexicon_ownership.py::test_outputspace_expand_masked_is_gone -v`
Expected: FAIL — `expand_masked` still exists.

- [ ] **Step 4: Delete `InputSpace.expand_masked`**

Open `bin/Spaces.py`. Locate `def expand_masked(self, embedded, sentence_text, maskedPrediction='MLM', n_words=None):` around line 4080. Delete that method entirely (through the final `return masked, list(range(N))`). Locate `def expand_masked_batched(...)` at ~4147 and delete that method entirely (~120 lines). Both methods live on `InputSpace`.

- [ ] **Step 5: Delete `OutputSpace.expand_masked`**

In the same file, locate `def expand_masked(self, embedded, sentence_text, maskedPrediction='MLM'):` around line 6470 (this is in class `OutputSpace`). Delete that method entirely.

- [ ] **Step 6: Delete the masked branch of `InputSpace.getBatch`**

In `InputSpace.getBatch` starting around line 4268, find the `else: # Masked prediction...` branch at line 4328. Replace the entire `if not use_masked: ... else: ...` structure with just the unmasked body:

```python
        # Standard mode: fixed-size batch slicing
        i = batchNum * batchSize
        if i >= len(inputData):
            return None, batchNum
        inputBatch = inputData[i:i + batchSize]
        inputTensor = self.prepInput(inputBatch)
        if outputData is not None:
            outputBatch = outputData[i:i + batchSize]
            outputTensor = self.outputSpace.prepOutput(outputBatch)
        else:
            outputTensor = None
        return (inputTensor, outputTensor), batchNum + 1
```

Also delete the `use_masked = ...` computation and the `sentences = ...` preamble (lines ~4302-4312) — they feed the masked branch and become dead.

- [ ] **Step 7: Delete attributes `_unmasked_embedding` and `_mask_positions`**

In `InputSpace.__init__` (around `bin/Spaces.py:3931`), find and delete the two lines that initialize:

```python
        self._unmasked_embedding = None
        self._mask_positions = None
```

Also search the file for any remaining read of these names (e.g., `self._unmasked_embedding` appears at ~4325, 4356, 4372) and delete those lines too. The `get_reconstruction_target()` method at ~4372 that reads `_unmasked_embedding` should be simplified: if it returns `(_unmasked_embedding, _mask_positions)` today, change it to return `(None, None)` (callers already handle `None`; check `runBatch` at `Models.py:2112, 2134` to confirm).

- [ ] **Step 8: Simplify `runBatch` output_weight and delete masked_pred gate**

In `bin/Models.py` around line 2001, find:

```python
        masked_pred = hasattr(self, 'masked_prediction') and self.masked_prediction != 'NONE'
```

Delete this line. `masked_pred` is not used anywhere else post-refactor (verify with grep; if any reads remain, remove those too).

Also at line 2080-2092, the AR output_weight dispatch:

```python
            if self.masked_prediction == 'ARUS':
                lossOut = torch.tensor(0.0, device=TheDevice.get())
                output_weight = 0.0
            elif pred_stack.numel() == 0:
                ...
            else:
                lossOut = self.loss.output(pred_stack, target_stack)
                output_weight = ((1 - self.loss.reverse_scale)
                                 if self.masked_prediction == 'ARIR' else 1.0)
```

Leave this alone — these are AR-mode dispatches, not MLM. Only delete the MLM-specific line (`masked_pred`).

- [ ] **Step 9: Delete `self.masked_prediction = 'MLM'` in data.py**

In `bin/data.py` around line 594:

```python
            self.masked_prediction = 'MLM'
```

Delete this line. Search the rest of the file for any reference to the default-to-MLM behavior and remove it.

- [ ] **Step 10: Delete MLM reference in visualize.py**

Run: `.venv/bin/python -c "import re; f=open('bin/visualize.py').read(); [print(i,l) for i,l in enumerate(f.split('\n')) if 'MLM' in l or 'masked_prediction' in l]"`
Expected: one or zero hits. Delete any hit that mentions MLM specifically (leave AR-mode references alone).

- [ ] **Step 11: Delete `TestMLM` class and MLM-specific tests**

In `test/test_basicmodel.py`, locate the `class TestMLM` block around line 2297. Delete the whole class and any fixtures it depends on that are not shared with other tests. Also delete any remaining test methods that use `expand_masked` directly.

In `test/test_stream_dataset.py`, delete tests at lines 167–270 that exercise `expand_masked_batched`.

- [ ] **Step 12: Run the new gone-tests — they should pass**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_expand_masked_is_gone test/test_lexicon_ownership.py::test_outputspace_expand_masked_is_gone -v`
Expected: PASS.

- [ ] **Step 13: Run full suite, confirm no new regressions**

Run: `.venv/bin/python -m pytest test/ -q`
Expected: ~820–835 passed (down by the 15-30 MLM tests just deleted), 0 failed.

If any non-MLM test now fails, stop and diagnose. Likely suspects:
- A test reads `inputSpace._unmasked_embedding` — delete or rewrite
- A test reads `masked_prediction='MLM'` — delete
- `get_reconstruction_target()` returns the wrong shape — check its remaining callers

- [ ] **Step 14: Stage changes and hand back to user**

```bash
git add bin/Spaces.py bin/Models.py bin/data.py bin/visualize.py test/test_basicmodel.py test/test_stream_dataset.py test/test_lexicon_ownership.py
git status
```

Do **not** run `git commit`. Report to user: "Task 1 complete. Staged changes remove MLM path and `expand_masked`. Ready for your review and commit."

---

## Task 2: Expose `InputSpace.arir_step` as a public method; delete `_getBatch_arir` alias

**Context:** ARIR (autoregressive iterative refinement) is a legitimate inference-time state machine. Today it hides inside `InputSpace._getBatch_arir` and is reached only through `getBatch`'s `if split == "runtime" and ... == 'ARIR'` dispatch at `Spaces.py:4299`. We expose it as an explicit method so the next task (`getBatch` deletion) is safe. No behavior change.

**Files:**
- Modify: `bin/Spaces.py` (rename `_getBatch_arir` → `arir_step`, make it public, keep its body identical).
- Modify: `bin/Models.py` (`runBatch` ARIR branch calls `arir_step` instead of hitting `getBatch`).
- Modify: `test/test_head_divergence.py` (one call site).

- [ ] **Step 1: Write a failing test that arir_step exists**

Add to `test/test_lexicon_ownership.py`:

```python
def test_inputspace_arir_step_is_public():
    from bin import Spaces
    assert hasattr(Spaces.InputSpace, 'arir_step'), \
        "InputSpace.arir_step should be a public method (promoted from _getBatch_arir)"
```

- [ ] **Step 2: Run the test — it should fail**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_arir_step_is_public -v`
Expected: FAIL.

- [ ] **Step 3: Rename `_getBatch_arir` to `arir_step`**

In `bin/Spaces.py`, locate `def _getBatch_arir(self, inputData, batchNum):` around line 4418. Rename to `def arir_step(self, inputData, batchNum):`. Update the docstring's first line if it references `getBatch()` by name.

Locate the call site inside `InputSpace.getBatch` at line 4300:

```python
        if split == "runtime" and getattr(self.data, '_runtime_mode', None) == 'ARIR':
            return self._getBatch_arir(inputData, batchNum)
```

Change to call `self.arir_step(inputData, batchNum)` (we'll delete this whole branch in Task 3, but this keeps things consistent for now).

- [ ] **Step 4: Run the test — it should pass**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_arir_step_is_public -v`
Expected: PASS.

- [ ] **Step 5: Update `test/test_head_divergence.py` to call arir_step directly**

Open `test/test_head_divergence.py` line 79:

```python
batch, _ = self.model.inputSpace.getBatch(0, 1, "runtime")
```

If the test is specifically exercising ARIR (check the surrounding context for `_runtime_mode == 'ARIR'` setup), replace with:

```python
inputData = self.model.inputSpace.data.train_input
batch, _ = self.model.inputSpace.arir_step(inputData, 0)
```

If it's not ARIR-specific, leave the `getBatch` call and let Task 3 handle it.

- [ ] **Step 6: Run test_head_divergence to confirm it still passes**

Run: `.venv/bin/python -m pytest test/test_head_divergence.py -v`
Expected: PASS (same number of tests as before).

- [ ] **Step 7: Run full suite**

Run: `.venv/bin/python -m pytest test/ -q`
Expected: same pass count as Task 1 result.

- [ ] **Step 8: Stage and hand back**

```bash
git add bin/Spaces.py test/test_head_divergence.py test/test_lexicon_ownership.py
git status
```

Report: "Task 2 complete. `arir_step` is now the public entry point. Ready for commit."

---

## Task 3: Delete `InputSpace.getBatch`

**Context:** With MLM gone (Task 1) and ARIR promoted to `arir_step` (Task 2), `getBatch`'s remaining responsibilities are (a) slice `train_input` / `outputData` by batchSize and (b) call `prepInput` / `prepOutput`. That's exactly what `TheData.data_loader` + `prepInput` do. We delete `getBatch`, rewire its callers, and route the inference-ARIR path through `arir_step` directly. **`_cached_embedding` is NOT deleted** — it remains the bypass mechanism that `arir_step` uses to hand a pre-built embedded tensor into `inputSpace.forward`. Its writers are now confined to `arir_step` only.

**Files:**
- Modify: `bin/Spaces.py` — delete `def getBatch` entirely. Leave `_cached_embedding` alone (it's still needed by `arir_step`'s bypass into `forward`).
- Modify: `bin/Models.py` — `runBatch` can no longer call `inputSpace.getBatch`. Route the inference path explicitly: if `split == "runtime"` and `_runtime_mode == "ARIR"`, call `inputSpace.arir_step`; otherwise require `batch_override`. If `batch_override` is None and it's not ARIR, raise `RuntimeError`.
- Modify: test files — replace `inputSpace.getBatch(...)` with `data_loader` + `prepInput` construction.

- [ ] **Step 1: Write a failing test that `getBatch` is gone**

Add to `test/test_lexicon_ownership.py`:

```python
def test_inputspace_getbatch_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'getBatch'), \
        "InputSpace.getBatch should be deleted"
```

- [ ] **Step 2: Run test — it should fail**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_getbatch_is_gone -v`
Expected: FAIL.

- [ ] **Step 3: Update test callers of `inputSpace.getBatch` — batch over DataLoader**

For each call site below, replace `batch, _ = m.inputSpace.getBatch(0, batchSize=N)` with the following pattern. Use `m.inputSpace` where `m` is the test's model variable (may be `self.model` or `model`):

```python
loader = m.inputSpace.data.data_loader(split="train", num_streams=N)
inp_items, out_items = next(iter(loader))
inputTensor = m.inputSpace.prepInput(inp_items)
outputTensor = (m.outputSpace.prepOutput(out_items)
                if out_items is not None else None)
batch = (inputTensor, outputTensor)
```

Files + line numbers:
- `test/test_mm_xor.py:94, 109, 130, 244, 291, 444` (6 call sites; keep `split="train"` unless the test uses a specific split)
- `test/test_basicmodel.py:2391, 2400, 2409, 2750` (4 call sites; line 2750 uses `split="runtime"` — adjust the `split=` argument in the loader call)
- `test/test_xor_spaces.py:198, 209`
- `test/test_testpoint.py:295`
- `test/diag_where.py:30`

If a caller is using `split="runtime"` and `m.inputSpace.data._runtime_mode == "ARIR"`, it needs `arir_step` instead (per Task 2) — check the test's setup code.

- [ ] **Step 4: Run each updated test file and confirm it passes against the OLD Spaces.py**

Run: `.venv/bin/python -m pytest test/test_mm_xor.py test/test_basicmodel.py test/test_xor_spaces.py test/test_testpoint.py -q`
Expected: same pass count as after Task 2.

If any test fails because the `data_loader` iteration produces a different shape than `getBatch`, the tests were relying on `getBatch`'s batchSize-slicing — adjust `num_streams` to match. The loader yields `[num_streams, seq_len]` tensors just like `getBatch`'s first element.

- [ ] **Step 5: Delete `InputSpace.getBatch`**

In `bin/Spaces.py`, locate `def getBatch(self, batchNum, batchSize=10, split="train"):` around line 4268. Delete the entire method including its docstring and all fall-through logic. The method ended around line 4368 before this task's edits.

Also delete the `arir_step` dispatch block inside `getBatch` — it's gone with the parent method.

- [ ] **Step 6: Update `runBatch` ARIR dispatch**

In `runBatch` around line 1996, the old code was:

```python
        if batch_override is not None:
            batch = batch_override
        else:
            batch, batchNum = self.inputSpace.getBatch(batchNum, batchSize, split)
            if batch is None:
                return None, batchNum
```

Change to:

```python
        if batch_override is not None:
            batch = batch_override
        elif (split == "runtime"
              and getattr(self.inputSpace.data, '_runtime_mode', None) == 'ARIR'):
            inputData = self.inputSpace.data.train_input
            result, batchNum = self.inputSpace.arir_step(inputData, batchNum)
            if result is None:
                return None, batchNum
            batch = result
        else:
            raise RuntimeError(
                "runBatch: no batch_override and not in ARIR runtime mode. "
                "Callers must pass batch_override or set _runtime_mode='ARIR'."
            )
```

The subsequent `inputTensor, outputTensor = batch` unpacking is unchanged. `arir_step` returns `((dummy_input, None), batchNum+1)` which unpacks cleanly and hands `dummy_input` to `self.forward`; `inputSpace.forward` sees `_cached_embedding` is set (from `arir_step`'s write) and takes the bypass path, ignoring `dummy_input`. This preserves the existing ARIR mechanic.

- [ ] **Step 7: Run the gone-test — it should now pass**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py::test_inputspace_getbatch_is_gone -v`
Expected: PASS.

- [ ] **Step 8: Run full suite**

Run: `.venv/bin/python -m pytest test/ -q`
Expected: same pass count as after Task 2 (no regressions).

Likely failure modes:
- A test caller was missed in Step 3 → grep `inputSpace.getBatch` in test/ and fix.
- `arir_step`'s caller signature changed and a test wasn't updated → fix the caller.
- `runBatch` now raises for a non-obvious caller → find and pass `batch_override`.

- [ ] **Step 9: Stage and hand back**

```bash
git add bin/Spaces.py bin/Models.py test/
git status
```

Report: "Task 3 complete. `getBatch` and `_cached_embedding` deleted. Ready for commit."

---

## Task 4: Delete InputSpace lexicon shim (`_peer_embedding`, `_peer_perceptual`, delegators)

**Context:** After Task 3, `InputSpace.getBatch`'s internal uses of `_peer_perceptual._lex_and_embed` are gone. The remaining users of the shim are the four InputSpace delegators (`predict`, `embed_token`, `get_space_embedding`, `get_mask_embedding`), which exist solely to keep back-compat with one caller in `bin/Models.py:1709`. We rewire that caller to `perceptualSpace.vocabulary.predict(...)` and delete the shim outright.

**Files:**
- Modify: `bin/Spaces.py` — delete `_peer_embedding`, `_peer_perceptual` attribute init; delete `_lexicon()`, `predict`, `embed_token`, `get_space_embedding`, `get_mask_embedding` methods on `InputSpace`. Update `InputSpace.arir_step` internal calls (lines 4433, 4451) to call `self._peer_perceptual.vocabulary.embedding_dim` / `.embed_token("\x00")` — wait, but `_peer_perceptual` is deleted. **Fix:** keep a single back-pointer for `arir_step`'s internal use, or pass the `perceptualSpace` reference through explicitly. Simpler: give `InputSpace` a `self._peer_perceptual` only and delete `_peer_embedding` (which is a shortcut to `_peer_perceptual.vocabulary`). Then `arir_step` reads `self._peer_perceptual.vocabulary.embedding_dim` and `.embed_token("\x00")` directly. Also delete the four public delegators.
- Modify: `bin/Models.py` — delete `_peer_embedding` setattr at lines 1334 and 2632. Keep `_peer_perceptual` setattr (renaming to clarify — it's no longer a shim, it's a legitimate back-reference for `arir_step`). Rewire the caller at line 1709.

**Correction to file structure:** Keep `_peer_perceptual`. Delete `_peer_embedding`. Delete the four delegators + `_lexicon()`. This is enough to kill the shim and force external callers onto `perceptualSpace.vocabulary` directly.

- [ ] **Step 1: Write failing tests**

Add to `test/test_lexicon_ownership.py`:

```python
def test_inputspace_peer_embedding_is_gone():
    from bin import Spaces
    import inspect
    src = inspect.getsource(Spaces.InputSpace.__init__)
    assert '_peer_embedding' not in src, \
        "_peer_embedding shortcut is gone; use _peer_perceptual.vocabulary"

def test_inputspace_predict_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'predict'), \
        "InputSpace.predict delegator deleted; callers use perceptualSpace.vocabulary.predict"

def test_inputspace_embed_token_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'embed_token')

def test_inputspace_get_space_embedding_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'get_space_embedding')

def test_inputspace_get_mask_embedding_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, 'get_mask_embedding')

def test_inputspace_lexicon_helper_is_gone():
    from bin import Spaces
    assert not hasattr(Spaces.InputSpace, '_lexicon')
```

- [ ] **Step 2: Run tests — expect 6 failures**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py -k "is_gone" -v`
Expected: the 6 new tests FAIL.

- [ ] **Step 3: Rewire the external caller in Models.py**

In `bin/Models.py` line 1709:

```python
                    decoded = self.inputSpace.predict(result.outputPred)
```

Change to:

```python
                    decoded = self.perceptualSpace.vocabulary.predict(result.outputPred)
```

- [ ] **Step 4: Delete `_peer_embedding` setattr in Models.py**

In `bin/Models.py`, locate the block at line 1330-1338:

```python
        if isinstance(self.perceptualSpace.vocabulary, Embedding):
            object.__setattr__(self.inputSpace, '_peer_embedding',
                               self.perceptualSpace.vocabulary)
            object.__setattr__(self.inputSpace, '_peer_perceptual',
                               self.perceptualSpace)
```

Change to:

```python
        if isinstance(self.perceptualSpace.vocabulary, Embedding):
            object.__setattr__(self.inputSpace, '_peer_perceptual',
                               self.perceptualSpace)
```

Apply the same edit at line 2631-2634 (the `MentalModel` constructor).

- [ ] **Step 5: Delete `_peer_embedding` init in InputSpace**

In `bin/Spaces.py` around line 3931:

```python
        self._peer_embedding = None
```

Delete this line. Keep `self._peer_perceptual = None` immediately below it.

- [ ] **Step 6: Delete `_lexicon()` and the four delegators**

In `bin/Spaces.py`, locate (around line 4387):

```python
    def _lexicon(self):
        ...
    def predict(self, vector):
        ...
    def embed_token(self, word):
        ...
    def get_space_embedding(self):
        ...
    def get_mask_embedding(self):
        ...
```

Delete these five methods (keep everything above and below). The methods span roughly lines 4387-4415.

- [ ] **Step 7: Update `arir_step` internal lexicon calls**

In `arir_step` (around lines 4433 and 4451 per research):

Old:
```python
        nWhat = self._lexicon().embedding_dim
```
New:
```python
        nWhat = self._peer_perceptual.vocabulary.embedding_dim
```

Old:
```python
            null_emb = self._lexicon().embed_token("\x00")
```
New:
```python
            null_emb = self._peer_perceptual.vocabulary.embed_token("\x00")
```

- [ ] **Step 8: Run the gone-tests — they should pass**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py -k "is_gone" -v`
Expected: all 6 new tests PASS.

- [ ] **Step 9: Run full suite**

Run: `.venv/bin/python -m pytest test/ -q`
Expected: same pass count as after Task 3.

Likely failure modes:
- The AR decode test (exercises `run_ar_inference`) now calls `perceptualSpace.vocabulary.predict` — if that method doesn't exist on the Embedding class, it means the attribute name was wrong in Step 3. Verify `Embedding.predict` exists in `bin/Spaces.py` via `grep -n "def predict" bin/Spaces.py`.
- An internal `arir_step` call still references `_lexicon()` or `_peer_embedding` — grep and fix.

- [ ] **Step 10: Stage and hand back**

```bash
git add bin/Spaces.py bin/Models.py test/test_lexicon_ownership.py
git status
```

Report: "Task 4 complete. InputSpace lexicon shim deleted. Ready for commit."

---

## Task 5: Dead-code sweep and final regression

**Context:** After Tasks 1-4, several names are likely orphaned:
- `_forward_input` may be used only inside `arir_step`
- `get_recovered_word` may be used only inside `arir_step`
- `_arir_stamp_where` is still needed by `arir_step`, no action
- `get_reconstruction_target()` may return `(None, None)` always — if so, delete it and its callers
- Stale `masked_prediction` / `MLM` strings in comments or docstrings

This task sweeps the dead references and runs the full suite end-to-end.

**Files:**
- Modify: `bin/Spaces.py`, `bin/Models.py`, `bin/data.py`, `bin/visualize.py` — dead-code removal.
- Run: `.venv/bin/python -m pytest test/ -q` — full regression.

- [ ] **Step 1: Grep for `MLM` in bin/**

Run: `.venv/bin/python -c "import subprocess; r=subprocess.run(['grep','-rn','MLM','bin/'], capture_output=True, text=True); print(r.stdout)"`

For each hit:
- If it's in a comment or docstring describing old behavior, delete the comment/paragraph.
- If it's code (e.g., `if ... == 'MLM'`), delete the branch.

- [ ] **Step 2: Grep for dead shim references**

Run: `.venv/bin/python -c "import subprocess; [print(p, subprocess.run(['grep','-rn',p,'bin/'],capture_output=True,text=True).stdout) for p in ['_peer_embedding', '_cached_embedding', '_unmasked_embedding', '_mask_positions', 'expand_masked']]"`

Expected: zero hits. If any appear, delete them.

- [ ] **Step 3: Check `get_reconstruction_target()` callers**

Run: `.venv/bin/python -c "import subprocess; r=subprocess.run(['grep','-rn','get_reconstruction_target','.'],capture_output=True,text=True); print(r.stdout)"`

If the method now always returns `(None, None)`, replace its body with just `return None, None` and simplify callers in `bin/Models.py:2112, 2134` to drop the branch that uses a non-None result.

If the method has other usage pathways (e.g., set by forward in some non-MLM mode), leave it alone.

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/python -m pytest test/ -q`
Expected: all non-deleted tests pass (target ~820-835 passing).

- [ ] **Step 5: Run once more with `-v` on the lexicon_ownership file to verify all gone-tests pass**

Run: `.venv/bin/python -m pytest test/test_lexicon_ownership.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Stage and hand back**

```bash
git add -A bin/ test/
git status
```

Report: "Task 5 complete. Dead-code sweep done; full suite passes. Ready for final commit."

---

## Post-refactor state (informational)

After all five tasks:

**Gone:**
- `InputSpace.getBatch`
- `InputSpace.expand_masked`, `InputSpace.expand_masked_batched`
- `OutputSpace.expand_masked`
- `InputSpace._cached_embedding`, `_unmasked_embedding`, `_mask_positions`
- `InputSpace._peer_embedding`, `InputSpace._lexicon()`
- `InputSpace.predict`, `embed_token`, `get_space_embedding`, `get_mask_embedding`
- `MLM` as a valid `masked_prediction` value
- All tests exercising MLM or `expand_masked`

**Kept / relocated:**
- `InputSpace._peer_perceptual` (back-reference used by `arir_step`)
- `InputSpace.arir_step` (formerly `_getBatch_arir`; now public)
- `InputSpace._arir_staged_embedding` (formerly the ARIR uses of `_cached_embedding`)
- AR modes (`ARLM`, `ARUS`, `ARIR`, `RARLM`) unchanged
- All training paths unchanged — `runEpoch` already pulled from `data_loader`

**Loss surface:**
- Output prediction loss (unchanged)
- Reconstruction loss (unchanged)
- SBOW / embedding loss (unchanged)
- MLM loss (gone)

The user's "next-percept prediction" framing is now coherent: every active loss term is already a form of "predict the next percept" (AR modes predict the next token-as-percept; non-AR modes predict the output-as-percept). The JEPA-shaped loss rewrite (if desired) becomes a separate plan with a clean slate to start from.
