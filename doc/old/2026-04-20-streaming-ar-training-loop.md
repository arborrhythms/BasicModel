# Streaming AR Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-position mask-and-rerun training loop with a single left-to-right pass inside `MentalModel.forward()`, making ARLM/ARUS forward-only and ARIR forward + one terminal reverse.

**Architecture:** Two nested loops in `MentalModel.forward()` — outer `for pos in range(N)` (token consumption, new), inner `for t in range(conceptualOrder)` (grammar production, existing). Percept/concept/symbol/`symbolic_state` tensors persist across the outer loop. One `backward()` + `optimizer.step()` per DataLoader yield. ARLM never reverses; ARIR does a single terminal reverse.

**Tech Stack:** Python 3.12, PyTorch (ROCm on user's Strix Halo hardware), pytest.

**Spec:** [basicmodel/doc/specs/2026-04-20-streaming-ar-training-loop-design.md](../specs/2026-04-20-streaming-ar-training-loop-design.md)

**Commit policy:** The user manages git commits. Each task ends with "Pause — user will commit." Do not run `git commit` yourself.

**Test runner:** `basicmodel/.venv/bin/python -m pytest basicmodel/test/<testfile>.py -v` (per user memory).

---

## File Structure

**Modified:**
- `basicmodel/bin/Layers.py` — add `Layer.Start()` base method
- `basicmodel/bin/Spaces.py` — add `Space.Start()` base method; add `SubSpace.reset_event()` method
- `basicmodel/bin/Models.py` — rename `ss` → `symbolic_state`, add `MentalModel.Start()`, rewrite `MentalModel.forward()`, update `runBatch` loss flow, delete runEpoch AR branch, remove `_cached_embedding` bypass, add ARIR config validation
- `basicmodel/bin/Spaces.py` (InputSpace class) — remove `_cached_embedding` / `_unmasked_embedding` / `_mask_positions` attrs
- `basicmodel/test/test_mental_model.py` — update for new `forward()` return signature
- `basicmodel/test/test_merged_loop.py` — update for new `forward()` return signature
- `basicmodel/doc/Training.md` — describe new training loop

**Created:**
- `basicmodel/test/test_streaming_ar_training.py` — unit tests for new streaming AR path

---

## Task 1: Rename `ss` → `symbolic_state` in MentalModel

**Files:**
- Modify: `basicmodel/bin/Models.py` (MentalModel.forward at line 2742, MentalModel.reverse at line 2984)

Pure rename — no behavior change. Reduces risk before the structural rewrite.

- [ ] **Step 1: Verify existing tests pass before the rename**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_merged_loop.py basicmodel/test/test_mental_model.py -v
```

Expected: all pass.

- [ ] **Step 2: Rename all uses of local `ss` variable in MentalModel.forward**

Open `basicmodel/bin/Models.py`. In `MentalModel.forward()` (line 2742 onward), find every occurrence of the local variable `ss` and rename to `self.symbolic_state`. Specifically:

Line 2802: change
```python
            ss = self.symbolicSpace.empty_state(batch=B).to(percepts.device)
```
to:
```python
            self.symbolic_state = self.symbolicSpace.empty_state(batch=B).to(percepts.device)
```

Line 2829-2830: change
```python
                ss_feedback = self._symbol_feedback_from_vectors(
                    ss, self._symbol_shape[0], percepts.shape[-1])
```
to:
```python
                ss_feedback = self._symbol_feedback_from_vectors(
                    self.symbolic_state, self._symbol_shape[0], percepts.shape[-1])
```

Line 2872: change
```python
                ss = sym_vectors
```
to:
```python
                self.symbolic_state = sym_vectors
```

Line 2884: change
```python
            self._nonrams_sym_feedbacks.append(ss)
```
to:
```python
            self._nonrams_sym_feedbacks.append(self.symbolic_state)
```

- [ ] **Step 3: Run existing tests to verify rename is safe**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_merged_loop.py basicmodel/test/test_mental_model.py -v
```

Expected: all pass (behavior unchanged).

- [ ] **Step 4: Pause — user will commit**

Suggested commit message: `refactor: rename ss to self.symbolic_state in MentalModel`

---

## Task 2: Add `Layer.Start()` base method

**Files:**
- Modify: `basicmodel/bin/Layers.py` (Layer class at line 33)
- Test: `basicmodel/test/test_streaming_ar_training.py` (new file)

Infrastructure only — introduces the method but no caller yet. Existing `reset()` methods stay.

- [ ] **Step 1: Write the failing test**

Create `basicmodel/test/test_streaming_ar_training.py` with:

```python
"""Tests for the streaming AR training loop refactor.

Plan reference: basicmodel/doc/plans/2026-04-20-streaming-ar-training-loop.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

from Layers import Layer


def test_layer_has_start_method():
    """Layer base class exposes a Start() method that is callable."""
    layer = Layer(nInput=4, nOutput=4)
    assert callable(getattr(layer, "Start", None))
    layer.Start()   # no-op, must not raise


def test_layer_start_cascades_to_children():
    """Layer.Start() walks self.layers and calls Start() on each."""
    parent = Layer(nInput=4, nOutput=4)
    child_a = Layer(nInput=4, nOutput=4)
    child_b = Layer(nInput=4, nOutput=4)
    called = []
    child_a.Start = lambda: called.append('a')
    child_b.Start = lambda: called.append('b')
    parent.layers = [child_a, child_b]
    parent.Start()
    assert called == ['a', 'b']
```

- [ ] **Step 2: Run test to verify it fails**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_layer_has_start_method -v
```

Expected: FAIL with `AttributeError: 'Layer' object has no attribute 'Start'` (or similar).

- [ ] **Step 3: Implement `Layer.Start()` in `basicmodel/bin/Layers.py`**

After the existing `set_sigma` / `observe_sigma` / `sigma_to_ergodic` methods in the `Layer` class (around line 60-67), add:

```python
    def Start(self):
        """Per-sentence state reset. Cascades to child layers.

        Layers with per-call state (e.g. cached ButterflyStage diffs) override
        this to clear that state. The default walks self.layers.
        """
        for layer in self.layers:
            if hasattr(layer, 'Start'):
                layer.Start()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: For each existing layer with `reset()`, add a `Start()` that calls `reset()`**

Layers with existing `reset()` (from spec): Layers.py:3336, 4253, 4507, 4609.

For each of those four classes, add immediately after the `reset()` definition:

```python
    def Start(self):
        super().Start()
        self.reset()
```

This keeps today's reset behavior while participating in the new cascade.

- [ ] **Step 6: Run the full Layers test suite**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py basicmodel/test/test_smoothing_regularizer.py -v
```

Expected: all pass.

- [ ] **Step 7: Pause — user will commit**

Suggested message: `feat: add Layer.Start() cascade base method`

---

## Task 3: Add `Space.Start()` base method and `SubSpace.reset_event()`

**Files:**
- Modify: `basicmodel/bin/Spaces.py` (Space class at line 3452; SubSpace class at line 2087)
- Test: `basicmodel/test/test_streaming_ar_training.py`

- [ ] **Step 1: Write the failing tests**

Append to `basicmodel/test/test_streaming_ar_training.py`:

```python
from Spaces import SubSpace


def test_subspace_has_reset_event():
    """SubSpace.reset_event() clears the cached event tensor."""
    import torch
    ss = SubSpace.__new__(SubSpace)
    ss.event = torch.randn(2, 3, 4)
    assert ss.event.abs().sum().item() > 0
    ss.reset_event()
    # After reset, event is either None or a zero tensor.
    assert ss.event is None or ss.event.abs().sum().item() == 0


def test_space_has_start_method():
    """Space base class exposes a Start() method."""
    from Spaces import Space
    assert callable(getattr(Space, "Start", None))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_subspace_has_reset_event basicmodel/test/test_streaming_ar_training.py::test_space_has_start_method -v
```

Expected: FAIL (`reset_event` and `Start` missing).

- [ ] **Step 3: Implement `SubSpace.reset_event()` in Spaces.py**

In the `SubSpace` class (line 2087), add:

```python
    def reset_event(self):
        """Clear the cached event tensor.

        Called from Space.Start() at the top of each MentalModel.forward()
        so that state carried across the outer pos loop does not leak into
        the next DataLoader yield.
        """
        self.event = None
```

- [ ] **Step 4: Implement `Space.Start()` in Spaces.py**

In the `Space` class (line 3452), add near the `paramUpdate` method (around line 3778):

```python
    def Start(self):
        """Per-sentence state reset. Cascades to child layers and subspace.

        Called from MentalModel.Start() once per forward() invocation. Subclasses
        with additional per-call state override this, calling super().Start()
        first.
        """
        for layer in self.layers:
            if hasattr(layer, 'Start'):
                layer.Start()
        sub = getattr(self, 'subspace', None)
        if sub is not None and hasattr(sub, 'reset_event'):
            sub.reset_event()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py -v
```

Expected: all pass.

- [ ] **Step 6: Pause — user will commit**

Suggested: `feat: add Space.Start() and SubSpace.reset_event() for per-sentence reset cascade`

---

## Task 4: Add `MentalModel.Start(inputData)` method

**Files:**
- Modify: `basicmodel/bin/Models.py` (MentalModel class)
- Test: `basicmodel/test/test_streaming_ar_training.py`

Reset cascade + embed only. Does NOT run the downstream pipeline. Called at the top of `forward()` (in a later task). Must not collide with the inherited `BasicModel.Start()` — MentalModel extends BaseModel directly, so there is no inherited `Start` to worry about (see spec).

- [ ] **Step 1: Write the failing test**

Append to `basicmodel/test/test_streaming_ar_training.py`:

```python
def test_mentalmodel_start_resets_and_embeds():
    """MentalModel.Start(inputData) resets spaces, initializes symbolic_state,
    and embeds the input via inputSpace.forward()."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    import torch

    import Models
    import Language

    # Build a minimal MentalModel from MM_xor.xml (small + fast).
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MM_xor.xml")
    Language.TheGrammar._configured = False
    model, _ = Models.MentalModel.from_config(src)

    # Fabricate a valid input tensor via getTrainData.
    Models.TheData.load("xor")
    train_input, _ = model.inputSpace.getTrainData()
    x = model.inputSpace.prepInput(train_input[:2])

    # After Start(), self.inputs is non-None and self.symbolic_state is
    # initialized with the right shape.
    model.eval()
    with torch.no_grad():
        model.Start(x)

    assert getattr(model, 'inputs', None) is not None
    assert getattr(model, 'symbolic_state', None) is not None
    # symbolic_state shape: [B, nOutput, nDim] from SymbolicSpace.outputShape
    sshape = tuple(model.symbolic_state.shape)
    assert len(sshape) == 3
    assert sshape[0] == x.shape[0]   # batch dim matches
```

- [ ] **Step 2: Run test to verify it fails**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_mentalmodel_start_resets_and_embeds -v
```

Expected: FAIL (`Start` missing or wrong behavior).

- [ ] **Step 3: Implement `MentalModel.Start(inputData)`**

In `basicmodel/bin/Models.py`, inside the `MentalModel` class (which starts at line 2314), add — before `def forward(self, inputData):` at line 2742 — a new method:

```python
    def Start(self, inputData):
        """Per-sentence state reset + input embedding.

        Replaces today's inline setup inside forward(): cascades reset to
        every Space (and every Layer within), clears the cached event on
        each subspace, initializes ``self.symbolic_state`` for the flat
        recurrent path, and embeds the input once.

        Does NOT run the downstream pipeline — the outer pos loop in
        forward() does that. See spec:
        basicmodel/doc/specs/2026-04-20-streaming-ar-training-loop-design.md
        """
        if isinstance(inputData, torch.Tensor):
            inputData = inputData.to(TheDevice.get())

        # 1. Cascade reset.
        for space in self.spaces:
            if hasattr(space, 'Start'):
                space.Start()
        if self.wordSpace is not None:
            self.wordSpace.clear_sentence()

        # 2. Scratch fields that accumulate across the outer pos loop are
        # reset; carried across inner conceptualOrder iterations only.
        self.symbol_states = []
        self._nonrams_sym_feedbacks = []
        if hasattr(self, 'symbolicSpace'):
            self.symbolicSpace.reset_symbol_objective()

        # 3. Embed once. Later per-pos calls read from the cached embedding
        # via InputSpace.forward returning the same subspace.
        self.inputs = self.inputSpace.forward(inputData)
        input_state = self.inputs.materialize()
        B = input_state.shape[0]

        # 4. Seed flat-path symbolic feedback state.
        self.symbolic_state = self.symbolicSpace.empty_state(batch=B).to(input_state.device)

        return input_state
```

- [ ] **Step 4: Run test to verify it passes**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py -v
```

Expected: all pass.

- [ ] **Step 5: Pause — user will commit**

Suggested: `feat: add MentalModel.Start() reset + embed cascade`

---

## Task 5: Add config validation — ARIR requires `reconstruct != 'NONE'`

**Files:**
- Modify: `basicmodel/bin/Models.py` (MentalModel.create around line 2326, where other `TheXMLConfig.require` calls live)
- Test: `basicmodel/test/test_streaming_ar_training.py`

- [ ] **Step 1: Write the failing test**

Append to `basicmodel/test/test_streaming_ar_training.py`:

```python
def test_arir_requires_reconstruct_not_none():
    """A config with maskedPrediction=ARIR and reconstruct=NONE is rejected."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    import Models
    import Language

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    arch = root.find("architecture")
    arch.find("maskedPrediction").text = "ARIR" \
        if arch.find("maskedPrediction") is not None \
        else None
    # Ensure both maskedPrediction=ARIR and reconstruct=NONE.
    mp = arch.find("maskedPrediction")
    if mp is None:
        mp = ET.SubElement(arch, "maskedPrediction")
    mp.text = "ARIR"
    rc = arch.find("reconstruct")
    if rc is None:
        rc = ET.SubElement(arch, "reconstruct")
    rc.text = "NONE"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    tree.write(tmp.name)
    tmp.close()

    try:
        Language.TheGrammar._configured = False
        import pytest
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            Models.MentalModel.from_config(tmp.name)
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_arir_requires_reconstruct_not_none -v
```

Expected: FAIL (no validation rule raises).

- [ ] **Step 3: Add the validation rule in `MentalModel.create`**

In `basicmodel/bin/Models.py`, find the block of `TheXMLConfig.require(...)` calls in `MentalModel.create` (starts near line 2371). Append:

```python
        # ARIR mode exists to train input reconstruction. Without a reverse
        # pass (reconstruct=NONE), there is nothing to train — the mode is
        # meaningless. ARLM/ARUS do not reverse, so <reconstruct> is inert
        # in those modes and no constraint is placed on them.
        TheXMLConfig.require(
            lambda cfg, _mp=masked_prediction,
                   _rc=str(TheXMLConfig.get("architecture.reconstruct")).upper():
                _mp != 'ARIR' or _rc != 'NONE',
            f"maskedPrediction=ARIR requires reconstruct != 'NONE' "
            f"(got maskedPrediction={masked_prediction!r}, "
            f"reconstruct={str(TheXMLConfig.get('architecture.reconstruct'))!r})"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_arir_requires_reconstruct_not_none -v
```

Expected: PASS.

- [ ] **Step 5: Pause — user will commit**

Suggested: `feat: add ARIR config validation (requires reconstruct != NONE)`

---

## Task 6: Rewrite `MentalModel.forward()` with outer pos loop for AR modes

**Files:**
- Modify: `basicmodel/bin/Models.py` (MentalModel.forward at line 2742)
- Test: `basicmodel/test/test_streaming_ar_training.py`

The core structural change. Non-AR path (masked_prediction='NONE') preserves today's single-pass behavior. AR modes (ARLM/ARUS/ARIR) get the new outer pos loop.

Return signature changes from 3-tuple `(input_state, symbols, outputData)` to 4-tuple `(input_state, symbols, predictions, reconstruction)`. For non-AR, `predictions` is the single output tensor (not a list) and `reconstruction` is None. For AR, `predictions` is a list of length N.

- [ ] **Step 1: Write the failing tests**

Append to `basicmodel/test/test_streaming_ar_training.py`:

```python
def test_arlm_forward_returns_predictions_list_and_no_reconstruction():
    """ARLM: forward() returns (input_state, symbols, predictions_list, None)."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    import torch

    import Models
    import Language

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    Language.TheGrammar._configured = False
    model, _ = Models.MentalModel.from_config(src)

    # Force ARLM mode on this model post-load (config sets ARLM by default
    # in MentalModel.xml, but make it explicit here).
    model.masked_prediction = 'ARLM'

    sentences = ['the cat sat on the mat']
    outputs = [torch.tensor([0.0])]
    with Models.TheData.runtime_batch(sentences, outputs):
        train_input, _ = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:1])

        model.eval()
        model.set_sigma(0)
        import warnings
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            out = model.forward(x)

    assert len(out) == 4, f"expected 4-tuple return, got {len(out)}"
    input_state, symbols, predictions, reconstruction = out
    assert isinstance(predictions, list), "ARLM must return a list of predictions"
    assert len(predictions) > 0, "ARLM must emit at least one per-pos prediction"
    assert reconstruction is None, "ARLM must not produce a reconstruction"


def test_arir_forward_returns_reconstruction():
    """ARIR: forward() returns (input_state, symbols, predictions_list, reconstruction)."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    import torch

    import Models
    import Language

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    arch = root.find("architecture")
    mp = arch.find("maskedPrediction")
    if mp is None:
        mp = ET.SubElement(arch, "maskedPrediction")
    mp.text = "ARIR"
    rc = arch.find("reconstruct")
    if rc is None:
        rc = ET.SubElement(arch, "reconstruct")
    rc.text = "symbols"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    tree.write(tmp.name)
    tmp.close()

    try:
        Language.TheGrammar._configured = False
        model, _ = Models.MentalModel.from_config(tmp.name)

        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]
        with Models.TheData.runtime_batch(sentences, outputs):
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])

            model.eval()
            model.set_sigma(0)
            import warnings
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                out = model.forward(x)

        _, _, predictions, reconstruction = out
        assert isinstance(predictions, list)
        assert reconstruction is not None, "ARIR must produce a reconstruction"
    finally:
        os.unlink(tmp.name)


def test_state_carries_across_pos_loop():
    """After forward() in ARLM mode, self.percepts state at pos=N-1 differs
    from initial zero state — confirming state accumulates."""
    import os
    import torch

    import Models
    import Language

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    Language.TheGrammar._configured = False
    model, _ = Models.MentalModel.from_config(src)
    model.masked_prediction = 'ARLM'

    sentences = ['the cat sat on the mat']
    outputs = [torch.tensor([0.0])]
    with Models.TheData.runtime_batch(sentences, outputs):
        train_input, _ = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:1])

        model.eval()
        model.set_sigma(0)
        import warnings
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.forward(x)

    final_symbols = model.symbols.materialize()
    # Non-zero final symbol state demonstrates the inner conceptualOrder
    # loop ran and state accumulated.
    assert final_symbols.abs().sum().item() > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_arlm_forward_returns_predictions_list_and_no_reconstruction basicmodel/test/test_streaming_ar_training.py::test_arir_forward_returns_reconstruction basicmodel/test/test_streaming_ar_training.py::test_state_carries_across_pos_loop -v
```

Expected: FAIL (today's forward returns 3-tuple, no pos loop, no predictions list).

- [ ] **Step 3: Extract inner conceptualOrder loop body into a helper**

In `basicmodel/bin/Models.py`, in the `MentalModel` class, add a new method `_run_conceptual_order(percepts, B, discourse_for_prime, ws)` that contains today's inner loop body (lines 2815-2876). It reads and writes `self.percepts / self.concepts / self.symbols / self.symbolic_state / self._sym_feedbacks / self._nonrams_sym_feedbacks`. Returns `sym_vectors` (last iteration's symbols tensor).

Add this method immediately before `def forward(self, inputData):` (so after `Finish`):

```python
    def _run_conceptual_order(self, percepts, B, discourse_for_prime, ws):
        """Run one pass of the inner conceptualOrder loop.

        Reads and writes instance state (self.percepts, self.concepts,
        self.symbols, self.symbolic_state, self._sym_feedbacks,
        self._nonrams_sym_feedbacks, self.symbol_states). Returns the final
        iteration's sym_vectors tensor.

        Called from the outer pos loop in forward(). State carries across
        calls within a single forward() -- the instance attributes hold the
        accumulated state.
        """
        sym_feedback = getattr(self, '_sym_feedback', None)
        x = percepts
        sym_vectors = None

        if self.useButterflies:
            x = percepts
            sym_feedback = None
        elif self.useGrammar == "all":
            x = percepts
            sym_feedback = None

        for t in range(self.conceptualOrder):
            # (body moved verbatim from lines 2816-2876 of the old forward)
            if self.useButterflies:
                concept_input = x
            elif self.useGrammar == "all":
                x = self._butterfly_merge(x)
                if sym_feedback is not None:
                    sym_feedback = (sym_feedback[:, 0::2, :] + sym_feedback[:, 1::2, :]) / 2
                    self._sym_feedbacks.append(sym_feedback)
                    x = x + sym_feedback
                else:
                    self._sym_feedbacks.append(None)
                concept_input = x
            else:
                ss_feedback = self._symbol_feedback_from_vectors(
                    self.symbolic_state, self._symbol_shape[0], percepts.shape[-1])
                self._nonrams_sym_feedbacks.append(ss_feedback)
                concept_input = self._bound_concept_input(percepts + ss_feedback)

            if t == 0 and discourse_for_prime is not None:
                bias = discourse_for_prime.prime(
                    self._predicted_snapshot,
                    self._predicted_confidence,
                    self.sentence_priming_scale)
                if bias is not None:
                    concept_input = concept_input + bias.view(1, 1, -1)

            self.percepts.set_event(concept_input)
            c_target = (None if (self.useButterflies or self.useGrammar == "all")
                        else self.nOutputSymbols)
            self.concepts = self.conceptualSpace[t].forward(
                self.percepts, wordSpace=ws, target_count=c_target)
            concept_vectors = self.concepts.materialize()
            if self.useButterflies or self.useGrammar == "all":
                x = concept_vectors

            is_last_step = (t == self.conceptualOrder - 1)
            quantize_sym = bool(self.useButterflies)
            self.symbols = self.symbolicSpace[t].forward(
                self.concepts, wordSpace=ws,
                quantize=quantize_sym, is_last=is_last_step)
            sym_vectors = self.symbols.materialize()

            if self.useGrammar == "all" and t < self.conceptualOrder - 1:
                _N_t, D_t = self._level_shapes_list[t]
                sym_norms = sym_vectors.norm(dim=-1, keepdim=True)
                sym_feedback = sym_norms.expand(-1, -1, D_t)
            elif not (self.useButterflies or self.useGrammar == "all"):
                self.symbolic_state = sym_vectors

            self.symbol_states.append(sym_vectors.clone())
            self._unified_j_iterations += 1

        # conceptualOrder==0 pre-seed path (spec's implicit j=-1)
        if (self.conceptualOrder == 0
                and not self.useButterflies
                and self.useGrammar != "all"):
            self._nonrams_sym_feedbacks.append(self.symbolic_state)
            self.percepts.set_event(self._bound_concept_input(percepts))
            self.concepts = self.conceptualSpace[0].forward(
                self.percepts, wordSpace=ws, target_count=self.nOutputSymbols)
            self.symbols = self.symbolicSpace[0].forward(
                self.concepts, wordSpace=ws, quantize=False, is_last=True)
            sym_vectors = self.symbols.materialize()

        # Butterfly path post-loop commit to subspace events.
        if self.useButterflies:
            self.conceptualSpace.subspace.set_event(x)
            self.concepts = self.conceptualSpace.subspace
            self.symbolicSpace.subspace.set_event(sym_vectors)
            self.symbols = self.symbolicSpace.subspace

        self._sym_feedback = sym_feedback
        return sym_vectors
```

- [ ] **Step 4: Rewrite `MentalModel.forward` to use the new helper and add outer pos loop**

Replace the existing `MentalModel.forward` method (lines 2742-2982) with:

```python
    def forward(self, inputData):
        """Forward pass with optional outer token-consumption loop for AR modes.

        Non-AR (masked_prediction='NONE'): single pass through the inner
        conceptualOrder loop, one output prediction. Returns
        (input_state, symbols, predictions, None) where `predictions` is
        the single output tensor.

        AR (masked_prediction in {ARLM, ARUS, ARIR}): outer pos loop reveals
        one token at a time via expand_masked_batched, runs the inner loop,
        emits a per-pos prediction via Finish(). State (self.percepts /
        self.concepts / self.symbols / self.symbolic_state) carries across
        the outer loop. Returns (input_state, symbols, predictions_list,
        reconstruction_or_None) where `predictions_list` is a list of length
        N and `reconstruction` is non-None only for ARIR.

        See spec:
        basicmodel/doc/specs/2026-04-20-streaming-ar-training-loop-design.md
        """
        ws = self.wordSpace

        # Shared setup (replaces old lines 2742-2806 state init).
        input_state = self.Start(inputData)                         # cascade + embed
        self.percepts = self.perceptualSpace.forward(
            self.inputs, wordSpace=ws,
            quantize=not (self.reversible and
                          getattr(self.perceptualSpace, "invertible", False)))
        percepts = self.percepts.materialize()
        B = percepts.shape[0]

        # Per-sentence discourse snapshot prediction (today's lines 2768-2782).
        self._predicted_snapshot = None
        self._predicted_confidence = None
        discourse_for_prime = (
            self.wordSpace.discourse
            if self.wordSpace is not None else None)
        if discourse_for_prime is not None:
            pred, conf = discourse_for_prime.predict()
            self._predicted_snapshot = pred
            self._predicted_confidence = conf

        # State buffers that the helper reads/writes.
        if self.useGrammar == "all":
            self._sym_feedbacks = []
        self._unified_j_iterations = 0

        # ---------------- outer dispatch ----------------
        is_ar_mode = self.masked_prediction in ('ARLM', 'ARUS', 'ARIR')

        if not is_ar_mode:
            # Non-AR path: single pass (today's flat/butterfly/grammar
            # behavior). Emit one output prediction.
            sym_vectors = self._run_conceptual_order(
                percepts, B, discourse_for_prime, ws)

            # Universality + downward-head emission (unchanged).
            self._universality_score = None
            truth_layer = (self.wordSpace.truth_layer
                           if self.wordSpace is not None else None)
            syntactic_layer = (self.wordSpace.syntacticLayer
                               if self.wordSpace is not None else None)
            svo = syntactic_layer.last_svo if syntactic_layer is not None else None
            lifting_layer = syntactic_layer.lifting_layer if syntactic_layer is not None else None
            if (truth_layer is not None and len(truth_layer) > 0
                    and svo is not None and lifting_layer is not None):
                s, v, o = svo
                self._universality_score = truth_layer.universality(
                    s, v, o, lifting_layer, self.symbolicSpace)

            # Discourse snapshot (today's lines 2899-2929).
            self._current_discourse_s = None
            self._current_discourse_w = None
            discourse = getattr(self.wordSpace, 'discourse', None) if self.wordSpace is not None else None
            if discourse is not None:
                try:
                    s_state = self.symbolicSpace.subspace.materialize()
                except Exception:
                    s_state = sym_vectors
                try:
                    w_state = self.wordSpace.read()
                except Exception:
                    w_state = None
                if w_state is None:
                    w_state = torch.zeros(
                        discourse.max_depth,
                        discourse.n_dim,
                        device=s_state.device, dtype=s_state.dtype)
                self._current_discourse_s = s_state.detach()
                self._current_discourse_w = w_state.detach()

            if self.useButterflies or self.useGrammar == "all":
                self.symbolicSpace.subspace.set_event(sym_vectors)
            output_syms = sym_vectors[:, :self.nOutputSymbols, :].clone()
            outputData = self.Finish(output_syms)
            symbols = sym_vectors.norm(dim=-1).unsqueeze(-1).expand(
                -1, -1, percepts.shape[-1])
            return input_state, symbols, outputData, None

        # ---------------- AR path: outer pos loop ----------------
        embedded = self.inputs.materialize()       # [B, nVec, embSize]
        inp_items = getattr(self.inputSpace, '_last_sentences', None)
        # (Sentences are available on InputSpace after prepInput; if the
        # caller did not stash them, fall back to N = embedded.shape[1].)
        if inp_items is not None:
            N = min(max(len(s.split()) for s in inp_items), embedded.shape[1])
        else:
            N = embedded.shape[1]

        predictions = []
        for pos in range(N):
            if inp_items is not None:
                masked, targets, _ = self.inputSpace.expand_masked_batched(
                    embedded, inp_items, self.masked_prediction, pos)
                self.inputs.set_event(masked)
            # Re-compute percepts for this pos's masked input.
            self.percepts = self.perceptualSpace.forward(
                self.inputs, wordSpace=ws,
                quantize=not (self.reversible and
                              getattr(self.perceptualSpace, "invertible", False)))
            percepts = self.percepts.materialize()
            sym_vectors = self._run_conceptual_order(
                percepts, B, discourse_for_prime, ws)

            # Per-pos output prediction.
            if self.useButterflies or self.useGrammar == "all":
                self.symbolicSpace.subspace.set_event(sym_vectors)
            output_syms = sym_vectors[:, :self.nOutputSymbols, :].clone()
            pred = self.Finish(output_syms)
            predictions.append(pred)

        symbols = sym_vectors.norm(dim=-1).unsqueeze(-1).expand(
            -1, -1, percepts.shape[-1])

        # Terminal reverse for ARIR only.
        reconstruction = None
        if self.masked_prediction == 'ARIR':
            # self.symbols already holds the final accumulated symbol state.
            # reverse() expects a sym-shaped tensor and an outputData
            # operand; the latter is irrelevant when reconstruct='symbols'.
            try:
                # use_cached path reads self.symbolicSpace.subspace.materialize()
                recon_input, _ = self.reverse(symbols, None)
                reconstruction = recon_input
            except Exception:
                reconstruction = None

        return input_state, symbols, predictions, reconstruction
```

- [ ] **Step 5: Add a sentence-stash hook on InputSpace**

The AR path reads `inp_items` (the raw sentence strings) to compute N. Today's
path passes it explicitly through the call stack. To keep the new forward
self-contained, have `InputSpace.prepInput` stash the last sentences as
`self._last_sentences` for the duration of the next forward call.

In `basicmodel/bin/Spaces.py`, locate `InputSpace.prepInput` (search for
`def prepInput`). Add at the top of the method body:

```python
        # Stash raw sentence strings so MentalModel.forward() can compute N
        # for the outer pos loop without plumbing them through the call stack.
        if isinstance(inp_items, list) and inp_items and isinstance(inp_items[0], str):
            self._last_sentences = list(inp_items)
        else:
            self._last_sentences = None
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py -v
```

Expected: all streaming-AR tests pass.

- [ ] **Step 7: Pause — user will commit**

Suggested: `feat: rewrite MentalModel.forward() with outer pos loop for AR modes`

---

## Task 7: Update `runBatch` loss flow for new forward signature

**Files:**
- Modify: `basicmodel/bin/Models.py` (BasicModel.runBatch at line 1842, which MentalModel uses via class alias)

runBatch unpacks the 4-tuple and uses `TheError.compute()` for output and reconstruction losses.

- [ ] **Step 1: Write the test**

Append to `basicmodel/test/test_streaming_ar_training.py`:

```python
def test_arlm_runbatch_trains_without_reverse():
    """Full runBatch under ARLM trains the model (loss decreases after step)
    and never calls reverse()."""
    import os
    import torch

    import Models
    import Language

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "MentalModel.xml")
    Language.TheGrammar._configured = False
    model, _ = Models.MentalModel.from_config(src)
    model.masked_prediction = 'ARLM'
    model.reconstruct = 'symbols'    # inert under ARLM -- no error expected

    opt = model.getOptimizer(lr=0.01)

    # Stub reverse so we can detect if it is called.
    call_count = {'n': 0}
    original_reverse = model.reverse
    def tracking_reverse(*args, **kwargs):
        call_count['n'] += 1
        return original_reverse(*args, **kwargs)
    model.reverse = tracking_reverse

    sentences = ['the cat sat on the mat']
    outputs = [torch.tensor([0.0])]
    with Models.TheData.runtime_batch(sentences, outputs):
        train_input, output_target = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:1])
        y = model.outputSpace.prepOutput(output_target[:1])

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result, _ = model.runBatch(
                train=True, batchNum=0, batchSize=1, split="train",
                optimizer=opt, batch_override=(x, y))

    assert result is not None
    assert call_count['n'] == 0, "ARLM must not call reverse()"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_arlm_runbatch_trains_without_reverse -v
```

Expected: FAIL (current runBatch calls reverse when `reversible=True`).

- [ ] **Step 3: Rewrite the forward + loss block inside `runBatch`**

In `basicmodel/bin/Models.py`, locate `BasicModel.runBatch` (line 1842). Find the block starting at line 1887:

```python
        # Forward pass (masking, if any, is applied inside InputSpace.forward())
        forwardInput, symbols, outputDataPred = self.forward(inputTensor)
```

...and continuing through line 1975 (the end of the output_loss / lossIn / use_recon block).

Replace that block with:

```python
        # Forward pass. AR modes return a per-pos predictions list and,
        # for ARIR, a terminal reconstruction tensor. Non-AR modes return
        # a single prediction tensor (as a 3rd element) and None for
        # reconstruction -- same shape as the old path.
        forwardInput, symbols, predictions, reconstruction = self.forward(inputTensor)

        if arir_mode:
            inputPred = reconstruction
            return self.BatchResult(
                outputPred=predictions if not isinstance(predictions, list)
                           else torch.stack(predictions, dim=1),
                symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=inputPred, forwardInput=forwardInput,
            ), batchNum

        if inference_only:
            return self.BatchResult(
                outputPred=predictions if not isinstance(predictions, list)
                           else torch.stack(predictions, dim=1),
                symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=None, forwardInput=forwardInput,
            ), batchNum

        if outputTensor is None:
            raise RuntimeError(
                f"runBatch: missing output targets for split='{split}'. "
                "For inference use split='runtime', or stage runtime_batch(..., outputs=...) "
                "if targets are required."
            )

        # Output loss. For AR modes, stack per-pos predictions + targets.
        if isinstance(predictions, list):
            # AR path. Target at pos=t is the t-th word's embedding, which
            # is already in forwardInput (the embedded input). Stack
            # predictions to [B, N, outDim].
            pred_stack = torch.stack(predictions, dim=1)
            # Build per-pos targets. OutputSpace target for ARLM/ARIR is
            # the embedding at pos t; outputTensor holds it as [B, N, D].
            target_stack = outputTensor
            if target_stack.dim() == 2:
                target_stack = target_stack.unsqueeze(1).expand(
                    -1, pred_stack.shape[1], -1)

            if self.masked_prediction == 'ARUS':
                lossOut = torch.tensor(0.0, device=TheDevice.get())
                output_weight = 0.0
            else:
                lossOut = self.loss.output(pred_stack.squeeze(), target_stack.squeeze())
                output_weight = ((1 - self.loss.reverse_scale)
                                 if self.masked_prediction == 'ARIR' else 1.0)
        else:
            # Non-AR path (masked_prediction='NONE'): today's behavior.
            outputPred = predictions.squeeze()
            output     = outputTensor.squeeze()
            lossOut    = self.loss.output(outputPred, output)
            self.accumulate_output_symbol_residual(outputTensor, predictions)
            output_weight = 1.0 - self.loss.reverse_scale

        TheError.add(
            "output", lossOut,
            weight=output_weight,
            space="OutputSpace", category="prediction",
        )

        # Reconstruction loss -- only present for ARIR (or non-AR reversible).
        use_recon = (reconstruction is not None
                     or (not isinstance(predictions, list)
                         and self.reversible and self.loss.reverse_scale > 0))
        if use_recon and reconstruction is not None:
            # ARIR terminal reverse already ran inside forward().
            pred_sq = reconstruction.squeeze() if hasattr(reconstruction, 'squeeze') else reconstruction
            recon_target, _ = self.inputSpace.get_reconstruction_target()
            target_sq = (recon_target.squeeze() if recon_target is not None
                         else forwardInput.squeeze())
            lossIn = self.loss.compute(pred_sq, target_sq)
            TheError.add(
                "reconstruction", lossIn,
                weight=self.loss.reverse_scale,
                space="InputSpace", category="reconstruction",
            )
            inputDataPred = reconstruction
        elif use_recon and not isinstance(predictions, list):
            # Non-AR reversible path: today's reverse branch.
            inputDataPred, inputPred = self.reverse(symbols, predictions)
            pred_sq = inputDataPred
            recon_target, recon_mask = self.inputSpace.get_reconstruction_target()
            target_sq = (recon_target.squeeze() if recon_target is not None
                         else forwardInput.squeeze())
            if recon_mask is not None and pred_sq.dim() >= 2:
                mask = recon_mask
                if pred_sq.dim() == 3:
                    mask = mask.unsqueeze(-1).expand_as(pred_sq)
                lossIn = self.loss.compute(pred_sq[mask], target_sq[mask])
            elif self.loss.nWhere > 0:
                lossIn = self.loss.compute_piecewise(pred_sq, target_sq)
            else:
                lossIn = self.loss.compute(pred_sq, target_sq)
            TheError.add(
                "reconstruction", lossIn,
                weight=self.loss.reverse_scale,
                space="InputSpace", category="reconstruction",
            )
        else:
            inputDataPred = None
            lossIn = None
```

- [ ] **Step 4: Update the end-of-method BatchResult and totalLoss paths**

The block after the section above (today's lines 1977 onward for JOINT embeddings, discourse, truth, and the final `totalLoss = ...` computation) continues to work unchanged — it reads `lossOut` / `lossIn` which we set above.

One correction: at the end of runBatch (line 2118), the old `BatchResult` is built from `outputDataPred`. Change that to use `pred_stack` if AR mode, else `predictions`. Specifically, replace the `result = self.BatchResult(...)` block (around line 2118) with:

```python
        result = self.BatchResult(
            outputPred=(pred_stack if isinstance(predictions, list)
                        else predictions),
            symbols=symbols,
            lossOut=lossOut,
            lossIn=lossIn,
            inputPred=inputDataPred,
            forwardInput=forwardInput,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py::test_arlm_runbatch_trains_without_reverse -v
```

Expected: PASS.

- [ ] **Step 6: Run the full AR test file**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py -v
```

Expected: all pass.

- [ ] **Step 7: Pause — user will commit**

Suggested: `feat: wire runBatch to new forward signature; ARLM skips reverse`

---

## Task 8: Delete the AR branch in `runEpoch`

**Files:**
- Modify: `basicmodel/bin/Models.py` (runEpoch at line 2128; AR branch at lines 2217-2272)

AR and non-AR now both flow through the same `runBatch` path. The per-pos loop that wrapped runBatch N times is obsolete.

- [ ] **Step 1: Run existing tests to establish baseline**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_mental_model.py basicmodel/test/test_mm_xor.py basicmodel/test/test_merged_loop.py -v
```

Expected: either all pass, or known failures are tagged (record them).

- [ ] **Step 2: Delete the AR branch**

In `basicmodel/bin/Models.py`, `runEpoch` method (line 2128). The AR branch spans lines 2213-2287 (the `if masked_pred and text_batch:` block through just before the `else:` at line 2288). Replace the entire block:

```python
                text_batch = (isinstance(inp_items, list)
                              and inp_items
                              and isinstance(inp_items[0], str))

                if masked_pred and text_batch:
                    # ... AR branch -- delete lines 2217-2287 ...
                else:
                    # Single forward/backward per B-wide batch.
                    result, _ = self.runBatch(
                        train=training, batchNum=step,
                        batchSize=B, split=split,
                        optimizer=optimizer,
                        batch_override=(inputTensor, outputTensor),
                    )
                    if result is not None:
                        record(result)
                    step += 1
```

...with the simpler unified path:

```python
                result, _ = self.runBatch(
                    train=training, batchNum=step,
                    batchSize=B, split=split,
                    optimizer=optimizer,
                    batch_override=(inputTensor, outputTensor),
                )
                if result is not None:
                    record(result)
                step += 1
```

- [ ] **Step 3: Delete the now-unused `masked_pred` local**

Also in `runEpoch`, lines 2181-2182 define:

```python
        masked_pred = (hasattr(self, 'masked_prediction')
                       and self.masked_prediction != 'NONE')
```

Delete both lines (no remaining reader).

- [ ] **Step 4: Run the full test suite**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py basicmodel/test/test_mental_model.py basicmodel/test/test_mm_xor.py basicmodel/test/test_merged_loop.py -v
```

Expected: existing tests that passed before still pass; new tests also pass.

- [ ] **Step 5: Pause — user will commit**

Suggested: `refactor: delete per-pos AR branch in runEpoch`

---

## Task 9: Delete the `_cached_embedding` bypass machinery

**Files:**
- Modify: `basicmodel/bin/Spaces.py` (InputSpace class)
- Modify: `basicmodel/bin/Models.py` (runBatch — remove batch_override rewiring that reads the bypass attrs)

- [ ] **Step 1: Find the bypass attr uses**

```bash
basicmodel/.venv/bin/python -c "
import sys; sys.path.insert(0, 'basicmodel/bin')
" 
```

Grep to confirm what references them:

```bash
# From repo root:
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_mental_model.py -v  # sanity
```

Then grep:

```bash
grep -n "_cached_embedding\|_unmasked_embedding\|_mask_positions" basicmodel/bin/Spaces.py basicmodel/bin/Models.py
```

Expected: three attrs referenced in Spaces.py (InputSpace) and Models.py (runEpoch's deleted AR branch — already gone, and InputSpace.forward which reads them).

- [ ] **Step 2: Remove the attrs and the branch that reads them from `InputSpace.forward`**

In `basicmodel/bin/Spaces.py`, locate `InputSpace.forward` and find the block that checks `_cached_embedding` (branches on the cached path). Remove that branch: the unconditional path is the only one that remains after this refactor. Also remove the class-level default assignments `self._cached_embedding = None`, `self._unmasked_embedding = None`, `self._mask_positions = None` from `InputSpace.__init__` if present.

The exact grep from Step 1 pinpoints the lines; delete each.

- [ ] **Step 3: Remove any leftover initializations in `MentalModel.create`**

Grep again to confirm no remaining references in Models.py:

```bash
grep -n "_cached_embedding\|_unmasked_embedding\|_mask_positions" basicmodel/bin/Models.py
```

Expected: no matches remain.

- [ ] **Step 4: Run the test suite**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py basicmodel/test/test_mental_model.py basicmodel/test/test_mm_xor.py -v
```

Expected: all pass.

- [ ] **Step 5: Pause — user will commit**

Suggested: `refactor: remove _cached_embedding bypass machinery`

---

## Task 10: Update existing tests for new `forward()` return signature

**Files:**
- Modify: `basicmodel/test/test_mental_model.py` (line 60: 3-tuple unpack → 4-tuple)
- Modify: `basicmodel/test/test_merged_loop.py` (if it unpacks forward's return)

- [ ] **Step 1: Update `test_mental_model.py`**

Line 60 today reads:

```python
                input_state, concepts, symbols = model.forward(x)
```

Replace with:

```python
                input_state, concepts, symbols, _ = model.forward(x)
```

- [ ] **Step 2: Inspect and update `test_merged_loop.py`**

```bash
grep -n "= model.forward\|= self.forward" basicmodel/test/test_merged_loop.py
```

If any 3-tuple unpacks remain, extend each to 4 values (the fourth is discarded).

- [ ] **Step 3: Run the suite**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_mental_model.py basicmodel/test/test_merged_loop.py basicmodel/test/test_streaming_ar_training.py -v
```

Expected: all pass.

- [ ] **Step 4: Pause — user will commit**

Suggested: `test: update existing tests for new forward() 4-tuple return`

---

## Task 11: Run full regression suite

**Files:** (none modified)

- [ ] **Step 1: Run the full test suite**

```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/ -v --timeout=120
```

Expected: all tests that passed on main before this refactor still pass. Flag any regressions.

- [ ] **Step 2: Pause — user will either commit any fixes or proceed**

If regressions are found, stop and fix them per root cause. Do not proceed until the suite is green.

---

## Task 12: Update `basicmodel/doc/Training.md`

**Files:**
- Modify: `basicmodel/doc/Training.md` (current section on masked prediction training, lines around 252-278 per earlier grep)

- [ ] **Step 1: Rewrite the "Masked Prediction" training section**

Open `basicmodel/doc/Training.md` and replace the pseudocode under the training-loop section (today describes the per-pos rerun) with:

```markdown
### Masked Prediction Training (ARLM / ARUS / ARIR)

For each DataLoader yield of B sentences:

1. `MentalModel.Start(inputData)` cascades reset through every Space and
   Layer, then embeds the input once: `embedded = InputSpace.forward(inputData)`.
2. The outer token-consumption loop runs inside `MentalModel.forward()`:

   ```
   for pos in range(N):
       expand_masked_batched(embedded, sentences, mode, pos)
       for t in range(conceptualOrder):
           # existing sigma / pi / feedback body -- state carries across pos
       predictions.append(Finish(symbols))
   ```

   Percept / concept / symbol / symbolic_state tensors persist across the
   outer pos loop within a single `forward()` call.

3. For ARIR only, a single terminal `reverse(symbols)` runs after the pos
   loop, producing a `[B, N, D]` reconstruction.

4. `runBatch` computes the loss via `TheError.compute()`:
   - ARLM: output prediction only.
   - ARUS: no output term (suppressed); no reconstruction.
   - ARIR: output prediction + reconstruction, weighted by `reverse_scale`.

5. One `backward()` + `optimizer.step()` per DataLoader yield.

Compared to the previous N-passes-per-sentence loop, this saves N-1 reverse
calls per AR sentence and collapses N optimizer steps into one.
```

- [ ] **Step 2: Pause — user will commit**

Suggested: `docs: describe streaming AR training loop in Training.md`

---

## Self-Review

Spec coverage check:

- **ARLM/ARUS/ARIR mode contract** — Tasks 5 (validation), 6 (forward branching), 7 (runBatch loss flow). Covered.
- **State carryover** — Task 6 (`_run_conceptual_order` reads/writes instance state; outer loop does not reset between pos). Covered.
- **Start() cascade** — Tasks 2, 3, 4. Covered.
- **`ss` → `symbolic_state`** — Task 1. Covered.
- **Terminal reverse for ARIR only** — Task 6 (last block of forward). Covered.
- **Delete runEpoch AR branch** — Task 8. Covered.
- **Delete `_cached_embedding` bypass** — Task 9. Covered.
- **ARIR requires reconstruct != NONE validation** — Task 5. Covered.
- **No constraint on reconstruct for ARLM** — implicit (no rule added). Covered.
- **Tests: state carryover, predictions shape, mode contracts, ARIR validation** — Tasks 2-7 include tests, Task 10 updates legacy tests, Task 11 runs regression. Covered.
- **Docs update** — Task 12. Covered.

Placeholder scan: no "TBD" / "implement later" / "similar to Task N" / unresolved stubs. The code snippets in each task are executable inline (minus indentation into context, which the engineer will preserve from the surrounding method).

Type / signature consistency:
- `MentalModel.forward()` consistently returns `(input_state, symbols, predictions, reconstruction)` across tasks 6-7.
- `MentalModel.Start(inputData)` returns `input_state` (the materialized embedding tensor) in Task 4; Task 6 consumes that return.
- `_run_conceptual_order` signature `(percepts, B, discourse_for_prime, ws)` is stable between Tasks 6-7.
- `SubSpace.reset_event()` / `Space.Start()` / `Layer.Start()` are used symmetrically across Tasks 2-4.
- `self.symbolic_state` (not `ss`) used consistently from Task 1 onward.
