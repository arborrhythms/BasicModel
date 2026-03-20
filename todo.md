## Per-Space Encoding, `ActiveEncoding`, and `SubSpace` Refactor

### Summary
Refactor the model away from the global `ObjectEncoding` / `TheObjectEncoding` design and toward per-space encoding ownership plus a single runtime `SubSpace` object.

The first step is to add `ActiveEncoding` and make `activation` a first-class field on `SubSpace`. This is the representation for fuzzy sparsity: a per-slot degree of presence/relevance that travels with `what`, `where`, and `when`. `WhatEncoding` remains the owner of reshape/flatten/unflatten behavior. There is no `ConcreteSubSpace`; `where` and `when` live directly on `SubSpace` and default to `null`.

For v1, keep `Model.py` dense. `SubSpace.materialize()` is the compatibility bridge into the current tensor math.

### Part 1: Add `ActiveEncoding` and unify runtime state under `SubSpace`
Implement this design in [basicmodel/bin/BasicModel.py](/Users/arogers/Library/Mobile%20Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/BasicModel.py), but do not add a second runtime state type.

- Add a single `SubSpace` runtime object with nullable fields:
  - `activation`
  - `what`
  - `where`
  - `when`
  - `activeEncoding`
  - `whatEncoding`
  - `whereEncoding`
  - `whenEncoding`
  - `reshape`
- Add `ActiveEncoding` as a per-space encoding object.
  - Its job is to encode/decode the `activation` factor only.
  - It represents fuzzy sparsity as a per-slot activation field aligned with the active slots of the subspace.
  - In v1, the canonical payload is dense scalar activation per active slot, not a new sparse-index structure.
- Keep `activation` as the payload name and `ActiveEncoding` as the codec name.
- Keep encode/decode logic on the `*Encoding` objects only. `SubSpace` is a state container and materialization adapter, not a codec.
- Make dimensions and sizes derived from attached tensors and encoding objects, not from a global singleton.
- `SubSpace.materialize()` should return the dense tensor expected by the current code.
  - For `reshape=True`, the whole representation is treated as `what`.
  - For `reshape=False`, materialize the current dense stand-in tensor without inventing new implicit activation math.
  - Do not automatically multiply `what` by `activation` inside `materialize()` in v1; keep activation explicit so current behavior stays stable.

### Part 1a: Define how `ActiveEncoding` relates to current activation logic
Current spaces already derive scalar activations in different ways. Preserve that asymmetry.

- `ActiveEncoding` should hold and validate the activation representation, but it should not impose one global formula for deriving activations from content.
- The producing `Space` remains responsible for how activation is computed from dense content, using the same formulas it already uses today.
  - Perceptual symbol-processing path: keep the current norm-based reduction.
  - Conceptual symbol-processing path: keep the current sum-based reduction.
  - Symbolic and syntactic paths: keep their current norm-based activation behavior.
- After that reduction, the result should be stored as `SubSpace.activation` and described by `SubSpace.activeEncoding`.
- This makes `ActiveEncoding` the carrier of fuzzy sparsity without forcing all spaces to share one activation semantics.

### Part 2: Remove global encoding ownership and make encodings per-space
Delete `ObjectEncoding` and `TheObjectEncoding` from the runtime design.

- Each `Space` should own the encoding instances needed to interpret the states it emits and consumes.
- `WhatEncoding` must be per-space and parameterized by that space’s content width.
  - It owns reshape, flatten, unflatten, and shape validation.
  - It must not depend on global `inputDim`, `perceptDim`, `conceptDim`, `objectSize`, `nWhere`, or `nWhen`.
- `ActiveEncoding`, `WhereEncoding`, and `WhenEncoding` must also be owned per space when that space uses them.
- `create_from_config()` must stop writing dimensions into a global encoding singleton and instead pass those values into each space constructor.
- `VectorSet`, `Embedding`, and related helpers must stop depending on global object-encoding state and instead receive the sizes and encoding references they need from the owning space.

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
- `WhatEncoding` remains the owner of reshape logic.
- `ActiveEncoding` is introduced first, but activation should stay explicit rather than being folded into `what` globally.
- Reverse and reconstruction paths must use the encodings attached to the current `SubSpace`, not global helpers.
- Loss partitioning must stop depending on global `nWhere` / `nWhen` and instead derive widths from local encodings or the realized `SubSpace` layout.
- Symbol concatenation and slicing can remain tensor-based in v1 by materializing, operating, and wrapping again.
- `SubSpace` becomes the stored runtime type for the current `inputs`, `percepts`, `concepts`, `symbols`, and `outputs` state fields.

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
- enable future sparse or partially sparse execution once spaces can consume structured `SubSpace` inputs directly

Implementation guardrails for performance:
- `SubSpace` should hold references, not duplicate tensors
- `materialize()` should be lazy and cached when possible
- `materialize()` should return the current dense stand-in tensor without changing semantics in v1
- do not split and recombine `what` / `where` / `when` unless a caller actually needs that structure
- do not inject `activation` into the dense tensor by default in v1

### Tests
- Add unit tests for `SubSpace` derived size properties.
- Add unit tests for `SubSpace.materialize()` in reshaped and non-reshaped cases.
- Add unit tests for `ActiveEncoding` carrying scalar fuzzy-sparsity payloads.
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
- `VectorSet` refactoring in v1 is limited to removing global encoding dependencies, not redesigning codebooks by factor.
