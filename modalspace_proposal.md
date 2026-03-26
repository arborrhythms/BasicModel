# ModalSpace Proposal

## Goal

Introduce two new spaces:

- `DemuxedInputSpace`
- `ModalSpace`

The purpose is to make the `what/where/when` factorization explicit inside `SubSpace`, while preserving the current external tensor contract for downstream spaces.

## Scope

Restricted to:

- adding `DemuxedInputSpace`
- adding `ModalSpace`
- minimal plumbing in `BasicModel.py` to instantiate them
- at least one equivalence testpoint proving that demuxed input materializes to exactly the same muxed tensor that the current `InputSpace` emits

## Why This Design

The current `PerceptualSpace` is parameterized by the last dimension, so processing separate `what`, `where`, and `when` as allowed by the new subspace model cannot share one instance. They need separate weights and therefore separate `PerceptualSpace` instances.

It is useful to keep the constructor shape-compatible with the rest of the architecture, so assume that `dEvent` has been added to the constructor size, although the SubSpace that is passed to the forward method will have a demuxed `dWhat`, `dWhere` and `dWhen` as provided by the DemuxedInputSpace.

## 1. `DemuxedInputSpace`

`DemuxedInputSpace` should subclass `InputSpace` and keep the same constructor signature:

```python
DemuxedInputSpace(inputShape, spaceShape, outputShape, model_type="simple")
```

### Behavior

- It should build the lexical/codebook basis for the `what` branch, not for the muxed `event`.
- It should emit a `SubSpace` with these populated slots:
  - `what`: lexical/content vectors only
  - `where`: spatial encoding only
  - `when`: temporal encoding only
  - If materialize is called on this subspace, it should emit a subspace with `event`, a mux of `[what, where, when]` in the exact same order as the current dense representation (i.e. this would be exactly identical to the existing InputSpace's returned subspace)
- It should not stamp spatial/temporal encodings into every codebook vector coming out of the lexical basis.
- Instead, it should create `where` and `when` as standalone tensors (which should be considerably easier, and they can be reused from invocation to invocation).

### Advantage

- the codebook maps directly to the `what` space
- positional and temporal metadata stop contaminating the lexical/codebook branch
- downstream code that still calls `materialize()` continues to see the exact same tensor as before

### Muxing Rule

- Preserve current layout: `event = concat([what, where, when], dim=-1)`
- This matches the current convention where `where` and `when` occupy reserved tail dimensions

### Compatibility Note

- Call sites in `BasicModel.py` that currently reach through `self.inputSpace.subspace.get_vectors()` should be switched to `self.inputSpace.get_vectors()`

## 2. `ModalSpace`

`ModalSpace` should be a composite space that keeps the same external constructor shape as other spaces:

```python
ModalSpace(inputShape, spaceShape, outputShape)
```

Internally, in its constructor, it should derive and create:

```python
self.whatSpace  = PerceptualSpace(...)
self.whereSpace = PerceptualSpace(...)
self.whenSpace  = PerceptualSpace(...)
```

### Shape Derivation

- `what` branch:
  - input dim = `inputShape[1] - input_objectSize`
  - output dim = `outputShape[1] - output_objectSize`
  - internal `spaceShape` uses the ordinary perceptual content width
- `where` branch:
  - input dim = input `nWhere`
  - output dim = output `nWhere`
  - internal `spaceShape` derived from `nWhere`
- `when` branch:
  - input dim = input `nWhen`
  - output dim = output `nWhen`
  - internal `spaceShape` derived from `nWhen`

Assumption:

- `nWhere` and `nWhen` are already consistent enough across the architecture to derive these branch shapes without new constructor args

### Forward Contract

- Input is one demuxed `SubSpace`
- `ModalSpace.forward(subspace)` does:
  - send `subspace.what` through `whatSpace`
  - send `subspace.where` through `whereSpace`
  - send `subspace.when` through `whenSpace`
  - collect branch outputs into a new `SubSpace`
  - return that `SubSpace` such that materializing it will set `event = concat([what_out, where_out, when_out], dim=-1)`

### Reverse Contract

- Split incoming `event` into `what/where/when` slices
- Call each branch's `reverse`
- Rebuild a demuxed `SubSpace`
- Re-mux into `event`

## 3. Integration in `BasicModel`

In `BasicModel.py`:

- replace `InputSpace(...)` with `DemuxedInputSpace(...)`
- replace `PerceptualSpace(...)` with `ModalSpace(...)`

So the pipeline stays structurally the same:

```python
self.inputs = self.inputSpace.forward(inputData)
self.percepts = self.perceptualSpace.forward(self.inputs)
self.concepts = self.conceptualSpace.forward(self.percepts)
```

That is the main benefit of eagerly populating `event`: `ConceptualSpace` can remain untouched.

For models with a ModalSpace that have nWhere/nWhen = 0, this will be identical to a since PerceptualSpace 

## 4. Testpoint / Acceptance Contract

Add one explicit equivalence testpoint:

- For the same input batch and same learned lexical/codebook basis:
  - legacy `InputSpace.forward(input).materialize()`
  - new `DemuxedInputSpace.forward(input).materialize()`
- must be exactly equal

This is the core contract:

- demuxing changes internal representation
- muxed output must remain identical

### Recommended Tests in `test_basicmodel.py`

- `DemuxedInputSpace` populates `what`, `where`, `when`
- `ModalSpace.forward()` accepts a demuxed `SubSpace` and returns a demuxed `SubSpace`
- `ModalSpace.materialize()` shape matches the current `PerceptualSpace` output shape

## Implementation Notes

- Prefer minimal support changes to `SubSpace`
- Keep `ConceptualSpace` oblivious to demuxing
- change XML/config by introducing a `ModalSpace`
