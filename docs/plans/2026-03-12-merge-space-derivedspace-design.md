# Merge Space and DerivedSpace

**Date:** 2026-03-12
**Status:** Approved

## Problem

Space and DerivedSpace are parallel class hierarchies that duplicate tensor
representation, vector-set behavior, layer orchestration, model construction,
and training hooks. They differ only in whether `TheObjectEncoding` adds
positional/temporal overhead (`objectSize > 0`) or is a no-op (`objectSize = 0`).

## Key Insight

When `TheObjectEncoding` is configured with `nWhere=0, nWhen=0`, its
`objectSize` is 0 and `getEmbeddingSize(D) == D`. The canonical tensor format
`[B, N, D+S]` degenerates to `[B, N, D]` — identical to DerivedSpace's raw
format. This means `TheObjectEncoding` can be used everywhere; it is simply a
no-op for derived/simple model configurations.

## Design Decisions

1. **No codec abstraction.** `TheObjectEncoding` is called everywhere. When
   `objectSize=0` it is a no-op.
2. **Universal training contract.** All spaces expose `getParameters()` and
   `paramUpdate()`. Non-ergodic spaces return empty lists / no-op.
3. **Two model classes.** `DerivedModel` is renamed to `SimpleModel`.
   `BasicModel` and `SimpleModel` both extend `BaseModel` with shared
   infrastructure. They are not folded into one class.
4. **SigmaLayer deterministic mode.** Instead of branching between
   `SigmaLayer` and `LinearLayer+Tanh` in ConceptualSpace, push the
   difference into `SigmaLayer` via a `deterministic` flag that forces
   `bias=1.0, temp=0.0` always (both train and eval).

## Architecture After Merge

### Unified Space Base Class

```
Space(nn.Module)
  __init__(inputShape, outputShape, nVectors, nDim,
           useVQ=False, customVQ=True, nPrototypes=0,
           reversePass=False, processSymbols=False,
           ergodic=False)

  embeddingSize = TheObjectEncoding.getEmbeddingSize(nDim)
    # When objectSize=0: embeddingSize == nDim

  # Universal training contract
  params: list          # trainable parameters (for ergodic layers)
  layers: list          # layers needing paramUpdate()
  getParameters() -> list
  paramUpdate()

  # Tensor plumbing (unchanged — works for both objectSize>0 and =0)
  getEmbeddedIO() -> (input_embedded, output_embedded)
  forwardBegin(x, t=0.0, reshape=False)
  forwardEnd(x, t=0.0, reshape=False)
  reverseBegin(y, t=0.0, reshape=False)
  reverseEnd(y, t=0.0, reshape=False)
  flatten(x, forward=True)
  reshape(y, forward=True)

  # Vector set (supports both quantized and unquantized)
  createVectorSet(quantized=True)
    # quantized=True  -> VectorSet (codebook snapping)
    # quantized=False -> UnquantizedVSet (pass-through)
  vectors()
```

### Subclass Mapping

| Before (canonical)     | Before (derived)           | After                                  |
|------------------------|----------------------------|----------------------------------------|
| `Space`                | `DerivedSpace`             | `Space` (unified)                      |
| `InputSpace`           | `DerivedInputSpace`        | `InputSpace` (widened)                 |
| `ConceptualSpace`      | `DerivedConceptualSpace`   | `ConceptualSpace` (+ ergodic flag)     |
| `OutputSpace`          | `DerivedOutputSpace`       | `OutputSpace` (widened)                |
| `PerceptualSpace`      | —                          | `PerceptualSpace` (unchanged)          |
| `SymbolicSpace`        | —                          | `SymbolicSpace` (unchanged)            |
| `SyntacticSpace`       | —                          | `SyntacticSpace` (unchanged)           |

### ConceptualSpace With Ergodic Flag

```
ConceptualSpace(Space)
  __init__(..., ergodic=False, invertible=False, hasNorm=False)

  When ergodic=True:
    SigmaLayer (with temperature/alpha adaptation)
  When ergodic=False:
    SigmaLayer(deterministic=True)  # bias=1.0, temp=0.0 always
  When invertible=True:
    ReversibleSigmaLayer
```

### SigmaLayer Deterministic Mode

```
SigmaLayer(ErgodicLayer)
  __init__(nInput, nOutput, permuteInput=False, deterministic=False)

  forward(x):
    if deterministic:
      bias, temp = 1.0, 0.0        # pure LinearLayer + Tanh
    elif not self.training:
      bias, temp = 1.0, 0.0
    else:
      bias, temp = self.layer_tradeoff()
    ...
```

### Model Hierarchy

```
BaseModel(nn.Module)
  # Shared infrastructure:
  load_config() / from_config() factory
  getOptimizer(lr)          # uses universal getParameters() contract
  paramUpdate()             # iterates self.spaces
  run() / runEpoch()        # epoch loop, logging, annealing
  mnistReport()             # evaluation reporting
  plotLoss()                # visualization
  save_weights() / load_weights()

BasicModel(BaseModel)
  # Full pipeline: Input → [Percept → Concept → Symbol]* → Output
  # Uses TheObjectEncoding with objectSize > 0
  # nSubThoughts, nThoughts control pipeline depth

SimpleModel(BaseModel)           # renamed from DerivedModel
  # Minimal pipeline: Input → Concept → Output
  # Uses TheObjectEncoding with objectSize = 0
  # Flags: ergodic, certainty, quantized
```

### Shared XML Config

```xml
<model>
  <architecture>
    <type>simple</type>              <!-- "simple" or "basic" -->
    <ergodic>true</ergodic>
    <certainty>true</certainty>
    <quantized>false</quantized>
    <nInput>32</nInput>
    <nConcepts>256</nConcepts>
    <nOutput>32</nOutput>
    <!-- basic-only -->
    <nPercepts>64</nPercepts>
    <nSymbols>2</nSymbols>
    <nWords>16</nWords>
    <reversePass>true</reversePass>
  </architecture>
</model>
```

Factory method:
```python
@staticmethod
def from_config(config_path=None, LM=None):
    cfg = BaseModel.load_config(config_path)
    model_type = cfg["architecture"].get("type", "basic")
    if model_type == "simple":
        model = SimpleModel()
    else:
        model = BasicModel()
    model.create_from_config(config_path, LM)
    return model, cfg
```

## Migration Steps

Each step preserves identical tensor shapes and compatible losses.

### Step 0: Regression Test Harness

Add tests before any refactoring:

- Canonical forward shapes: `InputSpace`, `ConceptualSpace`, `OutputSpace`
- Derived forward shapes: `DerivedInputSpace`, `DerivedConceptualSpace`,
  `DerivedOutputSpace`
- Reverse-path coverage (`reversePass=True`)
- Ergodic vs non-ergodic conceptual behavior
- Quantized vs unquantized vector behavior
- Symbol-processing cases depending on `TheObjectEncoding.objectSize`
- **Acceptance gate:** `BasicModel` and `DerivedModel` forward passes produce
  same tensor shapes and compatible losses as before

### Step 1: Universal Training Contract on Space

Add to `Space` base class:
- `self.params = []`
- `self.layers = []`
- `getParameters()` returning `self.params`
- `paramUpdate()` iterating `self.layers`

No behavioral change. Existing canonical spaces return empty lists.

### Step 2: SigmaLayer Deterministic Mode

Add `deterministic` flag to `SigmaLayer.__init__`. When `True`, force
`bias=1.0, temp=0.0` in both train and eval.

**Testpoint:** `SigmaLayer(deterministic=True)` produces same output as
`LinearLayer + Tanh` for identical weights/inputs.

### Step 3: Widen createVectorSet

Add `quantized` parameter to `Space.createVectorSet()`:
- `quantized=True` → `VectorSet` (existing behavior)
- `quantized=False` → `UnquantizedVSet`

### Step 4: Migrate ConceptualSpace

Add `ergodic` flag to `ConceptualSpace`:
- `ergodic=False` → `SigmaLayer(deterministic=True)`
- `ergodic=True` → `SigmaLayer` with temperature
- Add `hasNorm` support (NormLayer with +2 input dim)

**Testpoint:** `ConceptualSpace(ergodic=True, objectSize=0)` produces
identical outputs to `DerivedConceptualSpace(ergodic=True)`.

### Step 5: Migrate InputSpace and OutputSpace

Widen to handle both codebook paths:
- `InputSpace`: support unquantized codebook
- `OutputSpace`: support `createVectorSet(quantized=False)`

**Testpoint:** shape compatibility with derived variants.

### Step 6: Delete DerivedSpace Hierarchy

Remove `DerivedSpace`, `DerivedInputSpace`, `DerivedConceptualSpace`,
`DerivedOutputSpace`. Rename `DerivedModel` → `SimpleModel`. Update
`SimpleModel` to use unified `InputSpace`, `ConceptualSpace`, `OutputSpace`.

### Step 7: Factor Into BaseModel

Pull shared code into `BaseModel`:
- `getOptimizer()` (uses universal `getParameters()`)
- `runEpoch()` / `run()` (epoch loop, loss selection, annealing)
- `paramUpdate()` (iterate `self.spaces`)
- `mnistReport()` and plotting
- `from_config()` factory with shared XML schema

### Step 8: Final Acceptance

Both `BasicModel` and `SimpleModel` produce same tensor shapes and compatible
losses as the original `BasicModel` and `DerivedModel`.
