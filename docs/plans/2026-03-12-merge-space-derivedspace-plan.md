# Merge Space and DerivedSpace — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge the parallel Space/DerivedSpace class hierarchies into a single unified Space hierarchy, rename DerivedModel to SimpleModel, and factor shared infrastructure into BaseModel.

**Architecture:** DerivedSpace disappears. Space gains an `ergodic` flag, universal `getParameters()`/`paramUpdate()`, and `createVectorSet(quantized)`. SigmaLayer gains a `deterministic` flag. ConceptualSpace absorbs DerivedConceptualSpace's ergodic/non-ergodic paths. DerivedModel becomes SimpleModel using unified spaces.

**Tech Stack:** Python 3, PyTorch, unittest

**Test runner:** `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel -v`

---

## Task 0: Regression Test Harness

Lock down current behavior before any refactoring.

**Files:**
- Modify: `test/test_basicmodel.py`

**Step 1: Write regression tests for Space forward/reverse shapes**

Add these test classes to `test/test_basicmodel.py`:

```python
# ---------------------------------------------------------------------------
# Regression: Space and DerivedSpace shape contracts
# ---------------------------------------------------------------------------
class TestCanonicalSpaceShapes(unittest.TestCase):
    """Lock down tensor shapes for canonical Space subclasses."""

    def setUp(self):
        from BasicModel import (TheObjectEncoding, InputSpace,
                                ConceptualSpace, OutputSpace, TheData)
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, outputDim=4)
        self.B = 2  # batch

    def test_conceptual_space_forward_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace([nIn, TheObjectEncoding.inputDim],
                             [nOut, TheObjectEncoding.conceptDim],
                             nOut, TheObjectEncoding.conceptDim)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        x = torch.randn(self.B, nIn, inEmb)
        y = cs(x)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace([nIn, TheObjectEncoding.inputDim],
                             [nOut, TheObjectEncoding.conceptDim],
                             nOut, TheObjectEncoding.conceptDim,
                             reversePass=True)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        y = torch.randn(self.B, nOut, outEmb)
        x = cs.reverse(y)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        from BasicModel import OutputSpace, TheObjectEncoding
        nIn, nOut = 4, 4
        os_ = OutputSpace([nIn, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        x = torch.randn(self.B, nIn, inEmb)
        y = os_(x)
        self.assertEqual(list(y.shape), [self.B, nOut, TheObjectEncoding.outputDim])


class TestDerivedSpaceShapes(unittest.TestCase):
    """Lock down tensor shapes for DerivedSpace subclasses."""

    def setUp(self):
        self.B = 2

    def test_derived_input_forward_shape(self):
        from BasicModel import DerivedInputSpace
        nIn, nDim = 8, 1
        dis = DerivedInputSpace([nIn, nDim], nIn, nDim, nOutput=nIn)
        x = torch.randn(self.B, nIn, nDim)
        y = dis(x)
        self.assertEqual(list(y.shape), [self.B, nIn, nDim])

    def test_derived_conceptual_ergodic_forward_shape(self):
        from BasicModel import DerivedConceptualSpace
        nIn, nDim, nConcepts, cDim = 8, 1, 4, 1
        dcs = DerivedConceptualSpace([nIn, nDim], nConcepts, cDim,
                                     nOutput=nConcepts, ergodic=True)
        x = torch.randn(self.B, nIn, nDim)
        y = dcs(x)
        self.assertEqual(list(y.shape), [self.B, nConcepts, cDim])

    def test_derived_conceptual_traditional_forward_shape(self):
        from BasicModel import DerivedConceptualSpace
        nIn, nDim, nConcepts, cDim = 8, 1, 4, 1
        dcs = DerivedConceptualSpace([nIn, nDim], nConcepts, cDim,
                                     nOutput=nConcepts, ergodic=False)
        x = torch.randn(self.B, nIn, nDim)
        y = dcs(x)
        self.assertEqual(list(y.shape), [self.B, nConcepts, cDim])

    def test_derived_conceptual_reverse_shape(self):
        from BasicModel import DerivedConceptualSpace
        nIn, nDim, nConcepts, cDim = 8, 1, 4, 1
        dcs = DerivedConceptualSpace([nIn, nDim], nConcepts, cDim,
                                     nOutput=nConcepts, ergodic=True,
                                     reversePass=True)
        y = torch.randn(self.B, nConcepts, cDim)
        x = dcs.reverse(y)
        self.assertEqual(list(x.shape), [self.B, nIn, nDim])

    def test_derived_output_forward_shape(self):
        from BasicModel import DerivedOutputSpace
        nIn, nDim, nOut, oDim = 4, 1, 3, 1
        dos = DerivedOutputSpace([nIn, nDim], nOut, oDim, nOutput=nOut)
        x = torch.randn(self.B, nIn, nDim)
        y = dos(x)
        self.assertEqual(list(y.shape), [self.B, nOut, oDim])

    def test_derived_output_reverse_shape(self):
        from BasicModel import DerivedOutputSpace
        nIn, nDim, nOut, oDim = 4, 1, 3, 1
        dos = DerivedOutputSpace([nIn, nDim], nOut, oDim, nOutput=nOut,
                                 reversePass=True)
        y = torch.randn(self.B, nOut, oDim)
        x = dos.reverse(y)
        self.assertEqual(list(x.shape), [self.B, nIn, nDim])


class TestVectorSetVariants(unittest.TestCase):
    """Lock down quantized vs unquantized VectorSet behavior."""

    def test_unquantized_passthrough(self):
        from BasicModel import UnquantizedVSet
        vs = UnquantizedVSet()
        vs.create(4, 4, 1)
        x = torch.randn(2, 4, 1)
        y = vs.forward(x)
        self.assertTrue(torch.equal(x, y))

    def test_quantized_shape(self):
        from BasicModel import VectorSet
        vs = VectorSet()
        vs.create(4, 4, 3, customVQ=False)
        vs.addVectors(nVec=4)
        x = torch.randn(2, 4, 3)
        y = vs.forward(x)
        self.assertEqual(list(y.shape), [2, 4, 3])


class TestModelEndToEnd(unittest.TestCase):
    """Lock down full model forward shapes and loss compatibility."""

    def test_derived_model_ergodic_shapes(self):
        from BasicModel import DerivedModel
        model = DerivedModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(concepts.shape[0], 2)
        self.assertEqual(concepts.shape[1], 8)

    def test_derived_model_traditional_shapes(self):
        from BasicModel import DerivedModel
        model = DerivedModel()
        model.ergodic = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(concepts.shape[0], 2)
        self.assertEqual(concepts.shape[1], 8)

    def test_derived_model_reverse_shapes(self):
        from BasicModel import DerivedModel
        model = DerivedModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.reversePass = True
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        data, percepts = model.reverse(concepts)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_derived_model_loss_runs(self):
        """Verify forward + loss + backward doesn't crash."""
        from BasicModel import DerivedModel, CertaintyWeightedCrossEntropy
        model = DerivedModel()
        model.ergodic = True
        model.certainty = True
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        target = torch.randn(2, 4)
        out, concepts = model.forward(x)
        loss_fn = CertaintyWeightedCrossEntropy()
        loss = loss_fn(out.squeeze(), target)
        loss.backward()
        # No crash = pass
```

**Step 2: Run tests to verify they pass**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel -v`

Expected: All new tests pass (they test existing behavior).

**Step 3: Commit**

```bash
git add test/test_basicmodel.py
git commit -m "test: add regression tests for Space/DerivedSpace shape contracts"
```

---

## Task 1: Universal Training Contract on Space

**Files:**
- Modify: `bin/BasicModel.py:949-1052` (Space class)

**Step 1: Write the test**

Add to `test/test_basicmodel.py`:

```python
class TestUniversalTrainingContract(unittest.TestCase):
    """All spaces expose getParameters() and paramUpdate()."""

    def test_space_has_training_contract(self):
        from BasicModel import Space
        s = Space([4, 8], [4, 8], 4, 8)
        self.assertEqual(s.getParameters(), [])
        s.paramUpdate()  # should be a no-op, not crash

    def test_conceptual_space_has_training_contract(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, outputDim=4)
        cs = ConceptualSpace([4, TheObjectEncoding.inputDim],
                             [4, TheObjectEncoding.conceptDim],
                             4, TheObjectEncoding.conceptDim)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        cs.paramUpdate()  # no crash
```

**Step 2: Run test to verify it fails**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestUniversalTrainingContract -v`

Expected: FAIL with `AttributeError: 'Space' object has no attribute 'getParameters'`

**Step 3: Implement universal training contract on Space**

In `bin/BasicModel.py`, modify `Space.__init__` (line 955-968):

Add after line 968 (`self.processSymbols = processSymbols`):
```python
        self.params = []
        self.layers = []
```

Add after `reshape()` method (after line 1052):
```python
    def getParameters(self):
        return self.params
    def paramUpdate(self):
        for l in self.layers:
            l.paramUpdate()
```

**Step 4: Run test to verify it passes**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestUniversalTrainingContract -v`

Expected: PASS

**Step 5: Run full regression suite**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel -v`

Expected: All tests pass — no behavioral change.

**Step 6: Commit**

```bash
git add bin/BasicModel.py test/test_basicmodel.py
git commit -m "feat: add universal getParameters()/paramUpdate() contract to Space"
```

---

## Task 2: SigmaLayer Deterministic Mode

**Files:**
- Modify: `bin/Model.py:731-752` (SigmaLayer class)

**Step 1: Write the test**

Add to `test/test_basicmodel.py`:

```python
class TestSigmaLayerDeterministic(unittest.TestCase):
    """SigmaLayer(deterministic=True) behaves like LinearLayer + Tanh."""

    def test_deterministic_matches_linear_tanh(self):
        from Model import SigmaLayer, LinearLayer
        torch.manual_seed(42)
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, deterministic=True)
        sigma.train()

        # Build a matching LinearLayer + Tanh with same weights
        linear = LinearLayer(nIn, nOut, hasBias=True)
        with torch.no_grad():
            linear.W.copy_(sigma.layer.W)
            linear.B.copy_(sigma.layer.B)
        tanh = torch.nn.Tanh()

        x = torch.randn(2, nIn)
        y_sigma = sigma(x)
        y_manual = tanh(linear(x))
        self.assertTrue(torch.allclose(y_sigma, y_manual, atol=1e-6),
                        f"Deterministic SigmaLayer should match LinearLayer+Tanh")

    def test_deterministic_same_train_eval(self):
        from Model import SigmaLayer
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, deterministic=True)
        x = torch.randn(2, nIn)

        sigma.train()
        y_train = sigma(x).detach().clone()
        sigma.eval()
        y_eval = sigma(x).detach().clone()
        self.assertTrue(torch.allclose(y_train, y_eval, atol=1e-6),
                        "Deterministic mode should produce same output in train and eval")

    def test_non_deterministic_default(self):
        from Model import SigmaLayer
        sigma = SigmaLayer(nInput=8, nOutput=4)
        self.assertFalse(hasattr(sigma, 'deterministic') and sigma.deterministic)
```

**Step 2: Run test to verify it fails**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestSigmaLayerDeterministic -v`

Expected: FAIL — `SigmaLayer.__init__() got an unexpected keyword argument 'deterministic'`

**Step 3: Add deterministic flag to SigmaLayer**

In `bin/Model.py`, modify `SigmaLayer.__init__` (line 732):

Change:
```python
    def __init__(self, nInput, nOutput, permuteInput=False):
```
To:
```python
    def __init__(self, nInput, nOutput, permuteInput=False, deterministic=False):
```

Add after line 736 (`self.activation = torch.zeros(1,nOutput,1)`):
```python
        self.deterministic = deterministic
```

Modify `SigmaLayer.forward` (lines 741-752). Change:
```python
    def forward(self, x):
        bias, temp = self.layer_tradeoff()
        if not self.training:
            bias, temp = 1.0, 0.0      # pure learned weights, no noise
```
To:
```python
    def forward(self, x):
        if self.deterministic:
            bias, temp = 1.0, 0.0
        elif not self.training:
            bias, temp = 1.0, 0.0      # pure learned weights, no noise
        else:
            bias, temp = self.layer_tradeoff()
```

Note: The original had `layer_tradeoff()` called first, then overridden in eval. The new version only calls `layer_tradeoff()` when actually needed (training + non-deterministic).

**Step 4: Run test to verify it passes**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestSigmaLayerDeterministic -v`

Expected: PASS

**Step 5: Run full regression suite**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel -v`

Expected: All tests pass — default `deterministic=False` preserves existing behavior.

**Step 6: Commit**

```bash
git add bin/Model.py test/test_basicmodel.py
git commit -m "feat: add deterministic mode to SigmaLayer"
```

---

## Task 3: Widen createVectorSet

**Files:**
- Modify: `bin/BasicModel.py:1001-1004` (Space.createVectorSet)

**Step 1: Write the test**

Add to `test/test_basicmodel.py`:

```python
class TestCreateVectorSetQuantized(unittest.TestCase):
    """Space.createVectorSet supports both quantized and unquantized paths."""

    def test_quantized_creates_vectorset(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4, 3)
        s.createVectorSet(quantized=True)
        self.assertIsInstance(s.vectors(), VectorSet)

    def test_unquantized_creates_unquantized_vset(self):
        from BasicModel import Space, UnquantizedVSet
        s = Space([4, 3], [4, 3], 4, 3)
        s.createVectorSet(quantized=False)
        self.assertIsInstance(s.vectors(), UnquantizedVSet)

    def test_default_is_quantized(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4, 3)
        s.createVectorSet()
        self.assertIsInstance(s.vectors(), VectorSet)
```

**Step 2: Run test to verify it fails**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestCreateVectorSetQuantized -v`

Expected: FAIL — `createVectorSet() got an unexpected keyword argument 'quantized'`

**Step 3: Widen createVectorSet**

In `bin/BasicModel.py`, change `Space.createVectorSet` (lines 1001-1004):

Change:
```python
    def createVectorSet(self):
        self.vectorSet.append(VectorSet())
        self.vectors().create(self.inputShape[0], self.nVectors, self.nDim, self.customVQ) # can be bigger than nVectors, cannot be smaller
        self.vectors().addVectors(nVec=self.nPrototypes)
```
To:
```python
    def createVectorSet(self, quantized=True):
        if quantized:
            self.vectorSet.append(VectorSet())
            self.vectors().create(self.inputShape[0], self.nVectors, self.nDim, self.customVQ)
            self.vectors().addVectors(nVec=self.nPrototypes)
        else:
            vs = UnquantizedVSet()
            vs.create(self.inputShape[0], self.nVectors, self.nDim)
            self.vectorSet.append(vs)
```

Also change `self.vectorSet = []` to `self.vectorSet = nn.ModuleList()` in `Space.__init__` (line 963) to match DerivedSpace's pattern and ensure proper parameter registration:

Change:
```python
        self.vectorSet    = []
```
To:
```python
        self.vectorSet    = nn.ModuleList()
```

And update the class-level attribute (line 951) from `vectorSet = []` to remove it (the instance attribute in `__init__` takes precedence, but the class-level mutable list is a latent bug).

**Step 4: Run test to verify it passes**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestCreateVectorSetQuantized -v`

Expected: PASS

**Step 5: Run full regression suite**

Expected: All tests pass.

**Step 6: Commit**

```bash
git add bin/BasicModel.py test/test_basicmodel.py
git commit -m "feat: Space.createVectorSet supports quantized and unquantized paths"
```

---

## Task 4: Migrate ConceptualSpace

Add `ergodic` flag to ConceptualSpace so it can handle both the canonical sigma path and the derived traditional/ergodic paths.

**Files:**
- Modify: `bin/BasicModel.py:1378-1449` (ConceptualSpace)

**Step 1: Write the test**

Add to `test/test_basicmodel.py`:

```python
class TestConceptualSpaceErgodic(unittest.TestCase):
    """ConceptualSpace with ergodic flag matches DerivedConceptualSpace behavior."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        # Save original objectSize so we can restore
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def _set_zero_object_encoding(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0

    def test_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nIn, nDim, nOut, cDim = 8, 1, 4, 1
        cs = ConceptualSpace([nIn, nDim], [nOut, cDim], nOut, cDim,
                             ergodic=True, useVQ=False)
        x = torch.randn(2, nIn, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nOut, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nIn, nDim, nOut, cDim = 8, 1, 4, 1
        cs = ConceptualSpace([nIn, nDim], [nOut, cDim], nOut, cDim,
                             ergodic=False, useVQ=False)
        x = torch.randn(2, nIn, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nOut, cDim])

    def test_ergodic_with_norm(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nIn, nDim, nOut, cDim = 8, 1, 4, 1
        cs = ConceptualSpace([nIn, nDim], [nOut, cDim], nOut, cDim,
                             ergodic=True, hasNorm=True, useVQ=False)
        x = torch.randn(2, nIn, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nOut, cDim])

    def test_ergodic_reverse_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nIn, nDim, nOut, cDim = 8, 1, 4, 1
        cs = ConceptualSpace([nIn, nDim], [nOut, cDim], nOut, cDim,
                             ergodic=True, reversePass=True, useVQ=False)
        y = torch.randn(2, nOut, cDim)
        x = cs.reverse(y)
        self.assertEqual(list(x.shape), [2, nIn, nDim])

    def test_ergodic_exposes_params(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        cs = ConceptualSpace([8, 1], [4, 1], 4, 1,
                             ergodic=True, useVQ=False)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `ConceptualSpace.__init__() got an unexpected keyword argument 'ergodic'`

**Step 3: Add ergodic flag to ConceptualSpace**

In `bin/BasicModel.py`, modify `ConceptualSpace.__init__` (line 1382):

Change:
```python
    def __init__(self, inputShape, outputShape, nVectors, nDim, useVQ=True, reversePass=False, nPrototypes=0, processSymbols=False, invertible=False, hasNorm=False):
        super(ConceptualSpace, self).__init__(inputShape, outputShape, nVectors, nDim, useVQ=useVQ, nPrototypes=nPrototypes, reversePass=reversePass, processSymbols=processSymbols)
        input, output = self.getEmbeddedIO()
```
To:
```python
    def __init__(self, inputShape, outputShape, nVectors, nDim, useVQ=True, reversePass=False, nPrototypes=0, processSymbols=False, invertible=False, hasNorm=False, ergodic=False):
        super(ConceptualSpace, self).__init__(inputShape, outputShape, nVectors, nDim, useVQ=useVQ, nPrototypes=nPrototypes, reversePass=reversePass, processSymbols=processSymbols)
        self.ergodic = ergodic
        input, output = self.getEmbeddedIO()
```

Then modify the sigma layer creation section (lines 1386-1398). Currently:
```python
        self.hasNorm = hasNorm
        self.attention = AttentionLayer(output, output)
        if hasNorm:
            self.norm = NormLayer(input, input)
        if invertible:
            self.sigma = ReversibleSigmaLayer(input, self.nDim, permuteInput=False)
            self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
        elif reversePass:
            self.sigma1 = SigmaLayer(input, self.nDim, permuteInput=False)
            self.sigma2 = SigmaLayer(self.nDim, input, permuteInput=False)
            self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.forward
        else:
            self.sigma = SigmaLayer(input, self.nDim, permuteInput=False)
            self.forwardSigma = self.sigma.forward
```

Change to:
```python
        self.hasNorm = hasNorm
        self.attention = AttentionLayer(output, output)
        # Norm adds +2 features when enabled
        if hasNorm:
            self.norm = NormLayer(input, input + 2)
            input += 2
        if invertible:
            self.sigma = ReversibleSigmaLayer(input, self.nDim, permuteInput=False)
            self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
            self.params = self.sigma.getParameters()
            self.layers = [self.sigma]
        elif reversePass:
            self.sigma1 = SigmaLayer(input, self.nDim, permuteInput=False, deterministic=not ergodic)
            self.sigma2 = SigmaLayer(self.nDim, input, permuteInput=False, deterministic=not ergodic)
            self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.forward
            self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
            self.layers = [self.sigma1, self.sigma2]
        else:
            self.sigma = SigmaLayer(input, self.nDim, permuteInput=False, deterministic=not ergodic)
            self.forwardSigma = self.sigma.forward
            self.params = self.sigma.getParameters()
            self.layers = [self.sigma]
```

Key changes:
- `deterministic=not ergodic` on SigmaLayer — when `ergodic=False`, SigmaLayer acts as plain LinearLayer+Tanh
- NormLayer now uses `input + 2` and `input += 2` to match DerivedConceptualSpace's behavior
- `self.params` and `self.layers` populated for the universal training contract

**Step 4: Run test to verify it passes**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestConceptualSpaceErgodic -v`

Expected: PASS

**Step 5: Run full regression suite**

Expected: All tests pass. Existing ConceptualSpace callers pass `ergodic=False` by default — no behavioral change.

**Step 6: Commit**

```bash
git add bin/BasicModel.py test/test_basicmodel.py
git commit -m "feat: ConceptualSpace gains ergodic flag with deterministic SigmaLayer"
```

---

## Task 5: Migrate InputSpace and OutputSpace

**Files:**
- Modify: `bin/BasicModel.py:1273-1312` (InputSpace)
- Modify: `bin/BasicModel.py:1549-1583` (OutputSpace)

**Step 1: Write the test**

Add to `test/test_basicmodel.py`:

```python
class TestInputSpaceUnquantized(unittest.TestCase):
    """InputSpace works with unquantized codebook (objectSize=0)."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_unquantized_forward_shape(self):
        from BasicModel import TheObjectEncoding, InputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        nIn, nDim = 8, 1
        inp = InputSpace([nIn, nDim], [nIn, nDim], nIn, nDim=nDim, useVQ=False)
        x = torch.randn(2, nIn, nDim)
        y = inp(x)
        self.assertEqual(list(y.shape), [2, nIn, nDim])


class TestOutputSpaceUnquantized(unittest.TestCase):
    """OutputSpace works with objectSize=0."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_forward_shape_zero_object_size(self):
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], nOut, 1)
        x = torch.randn(2, nIn, 1)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])

    def test_reverse_shape_zero_object_size(self):
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], nOut, 1, reversePass=True)
        y = torch.randn(2, nOut, 1)
        x = os_.reverse(y)
        self.assertEqual(list(x.shape), [2, nIn, 1])
```

**Step 2: Run test to verify it fails**

Expected: May pass or fail depending on how InputSpace currently handles `useVQ=False`. Run to find out.

**Step 3: Widen InputSpace**

In `bin/BasicModel.py`, modify `InputSpace.__init__` (lines 1275-1293):

When no LM is provided and `useVQ=False`, create an unquantized codebook:

Change:
```python
        else:
            self.createVectorSet()
```
To:
```python
        else:
            self.createVectorSet(quantized=self.useVQ)
```

This makes InputSpace create an `UnquantizedVSet` when `useVQ=False`.

**OutputSpace** should already work at `objectSize=0` since `getEmbeddedIO()` returns raw dims. Verify — if tests pass without changes, no modification needed.

**Step 4: Run tests**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestInputSpaceUnquantized test.test_basicmodel.TestOutputSpaceUnquantized -v`

Expected: PASS

**Step 5: Run full regression suite**

Expected: All tests pass.

**Step 6: Commit**

```bash
git add bin/BasicModel.py test/test_basicmodel.py
git commit -m "feat: InputSpace/OutputSpace support unquantized codebook and objectSize=0"
```

---

## Task 6: Delete DerivedSpace Hierarchy and Rename DerivedModel

**Files:**
- Modify: `bin/BasicModel.py` — delete `DerivedSpace`, `DerivedInputSpace`, `DerivedConceptualSpace`, `DerivedOutputSpace` (lines 1591-1779)
- Modify: `bin/BasicModel.py` — rename `DerivedModel` to `SimpleModel`, update to use unified spaces
- Modify: `test/test_basicmodel.py` — update references

**Step 1: Write the migration test**

Add to `test/test_basicmodel.py`:

```python
class TestSimpleModel(unittest.TestCase):
    """SimpleModel (renamed DerivedModel) uses unified Space hierarchy."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_simple_model_ergodic_shapes(self):
        from BasicModel import SimpleModel, TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        model = SimpleModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(concepts.shape[0], 2)
        self.assertEqual(concepts.shape[1], 8)

    def test_simple_model_traditional_shapes(self):
        from BasicModel import SimpleModel, TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        model = SimpleModel()
        model.ergodic = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        from BasicModel import SimpleModel, TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        model = SimpleModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.reversePass = True
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        data, percepts = model.reverse(concepts)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `ImportError: cannot import name 'SimpleModel'`

**Step 3: Implement SimpleModel using unified spaces**

In `bin/BasicModel.py`:

1. Delete the entire `DerivedSpace` class and its subclasses (lines 1585-1779) — all of `DerivedSpace`, `DerivedInputSpace`, `DerivedConceptualSpace`, `DerivedOutputSpace`.

2. Rename `DerivedModel` to `SimpleModel`. Update its `create()` method to use the unified space classes:

```python
class SimpleModel(BaseModel):
    """Minimal pipeline: InputSpace → ConceptualSpace → OutputSpace.

    Three independent levers:
      ergodic   – SigmaLayer (with temperature/alpha adaptation) vs deterministic
      certainty – CertaintyWeightedCrossEntropy vs nn.CrossEntropyLoss
      quantized – VectorSet codebook snapping vs pass-through
    """
    name       = "SimpleModel"
    vSet       = None
    invertible = False
    hasNorm    = False
    ergodic    = True
    certainty  = True
    quantized  = False

    def create(self, nInput=32, nConcepts=256, nOutput=32):
        self.nInput    = nInput
        self.nConcepts = nConcepts
        self.nOutput   = nOutput

        inputDim   = 1
        conceptDim = 1
        outputDim  = 1

        self.inputSpace = InputSpace([self.nInput, inputDim],
                                     [self.nInput, inputDim],
                                     self.nInput, nDim=inputDim,
                                     useVQ=self.quantized)
        self.conceptualSpace = ConceptualSpace([self.nInput, inputDim],
                                               [self.nConcepts, conceptDim],
                                               self.nConcepts, conceptDim,
                                               reversePass=self.reversePass,
                                               invertible=self.invertible,
                                               hasNorm=self.hasNorm,
                                               ergodic=self.ergodic,
                                               useVQ=self.quantized)
        self.outputSpace = OutputSpace([self.nConcepts, conceptDim],
                                       [self.nOutput, outputDim],
                                       self.nOutput, outputDim,
                                       reversePass=False)

        self.spaces = [self.inputSpace, self.conceptualSpace, self.outputSpace]
```

The rest of SimpleModel (`forward`, `reverse`, `getOptimizer`, `runEpoch`, `run`, etc.) stays the same as the current DerivedModel code, just with the class name changed.

3. Add `DerivedModel = SimpleModel` alias for backward compatibility if needed by external code.

**Step 4: Run test to verify it passes**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel.TestSimpleModel -v`

Expected: PASS

**Step 5: Update existing DerivedModel tests**

In `test/test_basicmodel.py`, update `TestDerivedModel` to import `SimpleModel`:

```python
class TestDerivedModel(unittest.TestCase):
    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_derived_model_creation(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        self.assertIsNotNone(model)

    def test_derived_model_traditional(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic   = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=28*28, nConcepts=20, nOutput=10)
        x = torch.randn(2, 28*28, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)

    def test_derived_model_ergodic(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic   = True
        model.certainty = True
        model.quantized = False
        model.create(nInput=28*28, nConcepts=20, nOutput=10)
        x = torch.randn(2, 28*28, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
```

**Step 6: Run full regression suite**

Expected: All tests pass.

**Step 7: Commit**

```bash
git add bin/BasicModel.py test/test_basicmodel.py
git commit -m "refactor: delete DerivedSpace hierarchy, rename DerivedModel to SimpleModel"
```

---

## Task 7: Factor Shared Code Into BaseModel

**Files:**
- Modify: `bin/BasicModel.py` — BaseModel, SimpleModel, BasicModel

**Step 1: Write the test**

Add to `test/test_basicmodel.py`:

```python
class TestBaseModelFactory(unittest.TestCase):
    """BaseModel.from_config factory creates the correct model type."""

    def test_factory_creates_simple_model(self):
        import tempfile, os
        from BasicModel import BaseModel, SimpleModel
        xml = """<model>
  <architecture>
    <type>simple</type>
    <nInput>16</nInput>
    <nConcepts>8</nConcepts>
    <nOutput>4</nOutput>
  </architecture>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = BaseModel.from_config(path)
            self.assertIsInstance(model, SimpleModel)
        finally:
            os.unlink(path)

    def test_factory_creates_basic_model(self):
        import tempfile, os
        from BasicModel import BaseModel, BasicModel as BM
        xml = """<model>
  <architecture>
    <type>basic</type>
    <nInput>8</nInput>
    <nPercepts>16</nPercepts>
    <nConcepts>32</nConcepts>
    <nSymbols>2</nSymbols>
    <nOutput>4</nOutput>
  </architecture>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = BaseModel.from_config(path)
            self.assertIsInstance(model, BM)
        finally:
            os.unlink(path)
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `AttributeError: type object 'BaseModel' has no attribute 'from_config'`

**Step 3: Factor shared code into BaseModel**

In `bin/BasicModel.py`, add to `BaseModel` (after `load_config`):

```python
    @staticmethod
    def from_config(config_path=None, LM=None):
        """Factory: create the right model type from XML config."""
        cfg = BaseModel.load_config(config_path)
        arch = cfg.get("architecture", {})
        model_type = arch.get("type", "basic")
        if model_type == "simple":
            model = SimpleModel()
        else:
            model = BasicModel()
        model.create_from_config(config_path, LM)
        return model, cfg
```

Add `create_from_config` to SimpleModel if it doesn't have one:

```python
    def create_from_config(self, config_path=None, LM=None):
        cfg = self.load_config(config_path)
        arch = cfg.get("architecture", {})
        self.ergodic = arch.get("ergodic", self.ergodic)
        self.certainty = arch.get("certainty", self.certainty)
        self.quantized = arch.get("quantized", self.quantized)
        self.create(
            nInput=arch.get("nInput", 32),
            nConcepts=arch.get("nConcepts", 256),
            nOutput=arch.get("nOutput", 32),
        )
        return cfg
```

Move shared training methods from SimpleModel/BasicModel into BaseModel:
- `getOptimizer(lr)` — uses universal `getParameters()` contract
- `paramUpdate()` — already exists on BaseModel (line 1106)

**Step 4: Run tests**

Expected: PASS

**Step 5: Run full regression suite**

Expected: All tests pass.

**Step 6: Commit**

```bash
git add bin/BasicModel.py test/test_basicmodel.py
git commit -m "refactor: factor shared infrastructure into BaseModel, add from_config factory"
```

---

## Task 8: Final Acceptance

**Files:**
- No new files

**Step 1: Run full regression suite**

Run: `cd basicmodel && KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest test.test_basicmodel -v`

Expected: ALL tests pass.

**Step 2: Verify no DerivedSpace references remain**

Run: `grep -rn "DerivedSpace\|DerivedInputSpace\|DerivedConceptualSpace\|DerivedOutputSpace" bin/BasicModel.py`

Expected: No output (or only the backward-compat alias `DerivedModel = SimpleModel`).

**Step 3: Verify SimpleModel uses unified Space**

Run: `grep -n "class SimpleModel" bin/BasicModel.py`

Expected: Shows `SimpleModel(BaseModel)` using `InputSpace`, `ConceptualSpace`, `OutputSpace`.

**Step 4: Commit final cleanup if any**

```bash
git commit -m "chore: final cleanup after Space/DerivedSpace merge"
```
