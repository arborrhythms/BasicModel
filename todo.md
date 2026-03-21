## Per-Space Encoding, `ActiveEncoding`, and `SubSpace` Refactor


### Summary
The runtime now has:

- per-space `WhatEncoding`, `WhereEncoding`, `WhenEncoding`, and `ActiveEncoding`
- a single runtime `SubSpace` object carrying `what`, `where`, `when`, and `activation`
- `SubSpace.materialize()` as the compatibility bridge into the current dense tensor math
- `Basis`, `Tensor`, `Codebook`, and `Embedding` as the payload interface layer

`ActiveEncoding` is not only metadata for fuzzy sparsity. Because it is aligned
per active slot, it can also serve as a selector over the other subspace
factors: `activation` can be interpreted as the mask or weighting field that
determines which `what` / `where` / `when` entries belong to the current sparse
vector view. v1 still materializes densely, but this alignment is the bridge to
future sparse execution.

The remaining work is to make the internal space-to-space pipeline exchange `SubSpace`
objects consistently instead of dropping back to raw tensors between stages.

### Part 3: Pass `SubSpace` objects between spaces
Migrate the internal space-to-space pipeline in [basicmodel/bin/BasicModel.py](/Users/arogers/Library/Mobile%20Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/BasicModel.py) so spaces exchange `SubSpace` instead of raw tensors.

- Change `InputSpace`, `PerceptualSpace`, `ConceptualSpace`, `SymbolicSpace`, `SyntacticSpace`, and `OutputSpace` so their internal interfaces accept and return `SubSpace`.
- Keep backward compatibility at boundaries:
  - if input is already `SubSpace`, use it
  - if input is a tensor, wrap it into `SubSpace`
- Each computation boundary should follow one pattern:
  - normalize input to `SubSpace`
  - call `.materialize()` to get the dense tensor
  - run the existing dense layer, attention, VQ, or linear math unchanged
  - wrap the result into a new `SubSpace`
- Preserve current semantics:
  - `quantize=True` means transform, then quantize
  - `quantize=False` means transform and return an unquantized `SubSpace`
- For `reshape=True`, carry the full payload in `what` and leave `activation`, `where`, and `when` unset unless a space has a specific reason to preserve them separately.
- For `reshape=False`, populate `where` and `when` only where the current code already has meaningful data for them.

### Key Migration Rules
- Reverse and reconstruction paths must use the encodings attached to the current `SubSpace`, not global helpers.
- Loss partitioning must stop depending on global `nWhere` / `nWhen` and instead derive widths from local encodings or the realized `SubSpace` layout.
- Symbol concatenation and slicing can remain tensor-based in v1 by materializing, operating, and wrapping again.
- `SubSpace` becomes the stored runtime type for the current `inputs`, `percepts`, `concepts`, `symbols`, and `outputs` state fields.
- `ActiveEncoding` should remain aligned with the slot structure of `what`, `where`, and `when` so it can act as a selector or weighting field for sparse-vector interpretations later.

### Performance Expectations

This refactor is primarily an architectural cleanup, not an immediate optimization pass.

Expected near-term impact:
- v1 is likely performance-neutral if `SubSpace` mostly holds references and `materialize()` returns cached tensors or cheap views
- v1 may be slightly slower if `materialize()` repeatedly allocates, copies, or reassembles tensors
- removing `ObjectEncoding` alone is not expected to produce a meaningful speedup

Current hot paths are elsewhere:
- VQ currently round-trips through CPU on MPS
- VQ forward and reverse still use Python loops over batch/vector entries
- `where` / `when` stamping and re-encoding still perform extra tensor work

Optimization headroom unlocked by this refactor:
- quantize only `what` instead of full `[what | where | when]` vectors
- carry activation and VQ/codebook state forward explicitly instead of recomputing or re-looking up it later
- avoid some decode/re-encode glue in reverse paths
- use `activation` as the selector for sparse views over `what` / `where` / `when` without re-deriving active slots from dense content
- enable future sparse or partially sparse execution once spaces can consume structured `SubSpace` inputs directly

Implementation guardrails for performance:
- `SubSpace` should hold references, not duplicate tensors
- `materialize()` should be lazy and cached when possible
- `materialize()` should return the current dense stand-in tensor without changing semantics in v1
- do not split and recombine `what` / `where` / `when` unless a caller actually needs that structure
- do not inject `activation` into the dense tensor by default in v1

### Tests
- Add regression tests that current activation-producing paths still produce the same scalar activation values after being routed through `SubSpace.activation`.
- Add regression tests showing two different spaces can use different `WhatEncoding` dimensions without shared global coupling.
- Update shape-oriented tests to assert on `.materialize().shape` where spaces now return `SubSpace`.
- Add a regression test that spaces still accept raw tensor input and wrap it internally.
- Keep XML/default cleanup out of scope for this refactor.

### Assumptions
- `Model.py` remains dense in v1.
- `SubSpace` is the only runtime state type.
- `activation` is first-class and carried explicitly as fuzzy sparsity, but not globally injected into dense materialization semantics.
- `where` and `when` remain nullable fields on `SubSpace`.
- `reshape=True` means the full representation is carried as `what`.
- `Codebook` refactoring in v1 is limited to removing global encoding dependencies, not redesigning codebooks by factor.
