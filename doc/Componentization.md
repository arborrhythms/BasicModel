# Runtime Architecture and Componentization

> **Status: architectural assessment, 2026-07-13.** This document describes
> the live software boundaries in `bin/`, not the cognitive decomposition of
> perceptual, conceptual, whole, and symbolic space. It is a refactoring guide,
> not a claim that the proposed components already exist.

The focused ownership and pipeline-parallel contract for sparse, per-execution
`SubSpace` carriers and Space-owned codebooks is specified in
[`plans/2026-07-16-sparse-subspace-carrier-design.md`](plans/2026-07-16-sparse-subspace-carrier-design.md).

The conceptual architecture is substantially more modular than its runtime.
Layers, bases, spaces, grammar operations, and typed `SubSpace` carriers give
the model a useful domain vocabulary. The non-modularity is concentrated in
the mechanisms that *run* those objects: configuration hydration, corpus
streaming, per-step state, STM mutation, objective collection, checkpointing,
and reconstruction.

That distinction matters. Moving methods out of a large file would reduce file
size without fixing ownership. The high-value work is to give each lifecycle
one owner, explicit inputs and outputs, and serializable state.

## What is already a useful boundary

The following structures should be preserved and made easier to compose:

- `Layer` subclasses own tensor transformations and, in most cases, expose a
  forward/reverse contract.
- `Space` subclasses name the cognitive roles and own their codebooks.
- `SubSpace` is the typed carrier for event, activation, basis, and word data.
- `SentenceStreamDataset` already encapsulates the rolling multi-stream cursor
  better than a conventional shuffled `DataLoader` would.
- `Error` is a promising loss registry: it names, weights, categorizes, and
  reports objective terms.
- `ModelFactory.validate_config` and the preflight tests establish the right
  fail-before-training principle.

The recommended refactor wraps these mechanisms; it does not replace them with
generic framework abstractions.

## The current ownership problem

`bin/Models.py` is nearly 12,000 lines, but line count is only the symptom.
`BasicModel` and `BaseModel` jointly own construction, the training loop,
compiled execution, forward and reverse routing, checkpoint schema, vocabulary
migration, corpus progress, reconstruction reports, generation, truth storage,
and reasoning. The same object also serves as the mutable message bus between
phases through attributes such as `_staged_in_sub`, `_staged_concepts_in`,
`_prev_cs_for_ps`, `_prev_cs_for_ss`, `_ir_mask_positions`, `_stm_single_S`,
and `_current_discourse_s`.

The result is a system with named domain modules but implicit runtime APIs:

- **Launch configuration.** State and policy span `bin/train.py`,
  `ModelFactory`, `BaseModel.create_from_config`, `util.TheXMLConfig`, and
  `BASIC_*` environment variables. Effective configuration is reconstructed in
  stages and can differ between validation, data loading, construction, and
  training.
- **Corpus execution.** `Data`, `SentenceStreamDataset`,
  `BasicModel.runEpoch`, and `BasicModel.runBatch` share responsibility.
  Dataset mode, cursor semantics, reset policy, and device conversion leak into
  the model.
- **Forward execution.** `BasicModel._begin_step`, `_forward_body`,
  `_forward_body_per_word`, `_forward_per_stage`, and mutable attributes on
  several spaces collectively define the path. A stage's true input includes
  hidden state left by earlier calls.
- **STM and grammar.** `ShortTermMemory`, `SymbolSubSpace`,
  `ConceptualSpace.forward`, and the `_stm_*` methods on `BasicModel` divide
  storage, typed metadata, prediction, push/reduce policy, and
  sentence-boundary reset among different owners.
- **Objectives.** `ModelLoss`, module-level `TheError`, per-`SubSpace`
  registries, spaces that add terms, and `runBatch` all participate. It is
  difficult to prove which objectives are active for one launch or which phase
  owns a term.
- **Checkpoints.** `BaseModel.save_weights/load_weights`, vocabulary and BPE
  helpers, optimizer setup, RNG globals, and stream progress in the training
  loop contribute to one artifact. Adding a persistent mechanism requires
  editing the central serializer and knowing construction order.
- **Reconstruction.** Forward caches on spaces, `_reconstruction_seed`,
  `_reverse_body`, several `_reverse_*` helpers, and surface rendering in
  `InputSpace`/`PartSpace` form an implicit trace of the preceding forward
  pass. Cleared-cache and shape xfails expose this coupling.
- **Reasoning and serving.** Truth/LTM methods, reasoning adapters,
  realization, query parsing, and generation methods on `BasicModel` make
  training-time model state and application behavior share one facade and
  lifecycle.

There is a second form of coupling in the global runtime singletons:
`TheXMLConfig`, `TheData`, `TheDevice`, `TheMessage`, `TheGrammar`, and
`TheError`. These are convenient service locators, but they make isolation,
multiple models in one process, exact launch manifests, and construction tests
harder than necessary.

## Consolidate before componentizing

A mechanism should first be consolidated when it has one lifecycle but several
owners. It should become a separate component only after that lifecycle has a
stable boundary.

For example, STM should not initially be split into more classes. Its buffer,
typed metadata, push/reduce policy, predictor, and reset rules should first be
made one coherent runtime API. Only then is it useful to distinguish a storage
object from a grammar policy. Conversely, launch resolution already has a
natural immutable output and can become a component immediately.

The practical rule is:

1. Identify the authoritative state.
2. Move every mutation behind one owner.
3. Replace parked attributes with an explicit request/result object.
4. Characterize the boundary with tests.
5. Extract the owner without changing tensor math.

## Proposed component boundaries

### 1. `LaunchSpec` and `RuntimeContext` - priority P0

`train.py` should be the CLI/remote transport adapter, not a second
configuration system. XML defaults, XML overlay values, CLI flags, and
environment overrides should resolve once into an immutable `LaunchSpec`.
That spec should contain the dataset limits, training schedule, device,
precision, compilation policy, seed, artifact paths, and the model's resolved
architecture settings.

`RuntimeContext` should hold the explicitly process-scoped services needed to
execute that spec: device, logger, RNG handles, grammar catalog, and project
paths. Model and data constructors should receive these values rather than read
global singletons after construction.

The first extraction can be conservative: keep `TheXMLConfig` internally, but
have a resolver produce a frozen snapshot and require `ModelFactory`, `Data`,
and checkpoint metadata to consume the same snapshot. Environment variables
may remain an input to the resolver; they should not remain a late-bound input
to the model.

**Acceptance seam:** one test renders the fully resolved launch manifest and
asserts that validation, data loading, model construction, compilation, and the
checkpoint record all see the same values.

### 2. `TrainingSession` and `TensorStep` - priority P0

The host-eager training lifecycle and the compiled tensor computation need a
hard boundary.

`TrainingSession` should own:

- epoch/trial iteration and batch budgets;
- stream acquisition and sentence/document resets;
- optimizer construction and stepping;
- checkpoint cadence and progress reporting;
- profiling, preflight, and failure handling.

`TensorStep` should accept a `StepInput` plus explicit recurrent state and
return a `StepResult` containing predictions, next state, objective terms, and
diagnostics. It should contain only the shape-stable tensor work intended for
`torch.compile`.

Today `runBatch` is both of these things. That makes the compile boundary a
list of exceptions rather than an interface: host-side vocabulary growth,
reporting, stream state, error aggregation, reset dispatch, and tensor forward
all meet in one method. Separating the two gives the FineWeb launch a testable
unit of progress and makes a compiled step callable without manufacturing a
whole training run.

**Acceptance seam:** eager and compiled `TensorStep` return the same named
outputs and objective terms for a fixed `StepInput`; `TrainingSession` can be
tested with a deterministic fake step.

### 3. `CorpusSource`, `StreamCursor`, and `BatchAdapter` - priority P0

`Data` currently combines dataset selection, downloading, parsing, synthetic
fixtures, normalization, split storage, tokenization, target preparation, and
loader construction. Preserve `SentenceStreamDataset`, but put a protocol in
front of it:

- `CorpusSource` yields stable document/sentence identities and a source
  manifest.
- `StreamCursor` owns resumable multi-row position and reset state.
- `BatchAdapter` converts a cursor tick into the model's typed `StepInput`.

The distinction between a corpus byte stream and a labeled trial dataset then
belongs in adapters rather than in `runEpoch`. A checkpoint stores
`StreamCursor.state_dict()` and the `CorpusSource` manifest, not a collection of
model-private counters that are later interpreted as cursor movement.

This boundary is essential for a long FineWeb.edu run: exact resumption is a
data property, while batch counting is only training telemetry.

**Acceptance seam:** interrupt/resume produces the same next document IDs,
bytes, row reset mask, and targets as an uninterrupted run, across a shard
boundary.

### 4. `ForwardFrame` and `ModelRuntimeState` - priority P1

The forward path needs explicit dataflow objects. A useful split is:

- `ForwardFrame`: immutable or single-use values for one step, including the
  input event, masks, per-stage PS/WS/CS carriers, active-row gates, and the
  reconstruction trace.
- `ModelRuntimeState`: recurrent values that intentionally survive steps,
  including discourse state, model time, inter-sentence state, and STM state.
- `StepResult`: terminal output, next runtime state, loss terms, attention
  observations, and optional reconstruction artifacts.

This replaces the model-as-message-bus pattern. In particular, values like
`_prev_cs_for_ps`, `_prev_cs_for_ss`, `_staged_in_sub`, and `_serial_row_gate`
should be passed, not discovered on sibling objects. Persistent state should be
listed and checkpointable; everything else should die with the frame.

Do not attempt this as one rewrite. Start with `_begin_step` / `_end_step`, then
thread the existing carrier values through the per-word loop while leaving the
underlying `Space.forward` methods unchanged.

**Acceptance seam:** running two model instances interleaved in one process
produces the same results as running each alone, with no state crossing between
frames.

### 5. `STMCoordinator` - priority P1

The current STM implementation has at least four legitimate concerns, but no
single authoritative facade:

- `ShortTermMemory` provides the live idea buffer and rule scorer.
- `SymbolSubSpace` owns typed stack metadata and some backing capacity.
- `ConceptualSpace` predicts, perceives, and pushes events.
- `BasicModel` decides bounded reduction, sentence-relative protection,
  compaction, row reset, and reverse snapshot use.

First create an `STMCoordinator` API over the existing objects:

```text
begin_step(batch, reset_mask)
predict(prior, routing) -> prediction
perceive(event, row_gate)
reduce(policy, protection) -> ReductionTrace
snapshot() -> STMSnapshot
end_sentence(row_mask)
```

`STMSnapshot` should contain payload, depth, typed references/categories, and
ordering convention. Both reconstruction and discourse prediction should
consume that object. Once every mutation is behind this API, storage and
reduction policy can be separated if doing so still buys clarity.

**Acceptance seam:** the existing serial/parallel, padding no-op, bounded-fold,
typed-STM, and cleared-cache reconstruction tests run through the coordinator;
no caller reads `_buffer`, `_depth`, or a sibling's transient STM attributes.

### 6. `ObjectiveSet` - priority P1

`Error` already has most of the reporting behavior needed for an
`ObjectiveSet`, but it should be instantiated per step/session rather than
imported as `TheError`. Spaces may return named `LossTerm` values; they should
not write into a process-global registry.

A `LossTerm` should record name, scalar tensor, configured weight, category,
producer, normalization denominator, and active-row count. The objective
coordinator should be the only code that applies the curriculum and combines
terms for backward. This makes an unlabeled FineWeb batch auditable: the launch
can fail if the resolved objective set is empty, non-finite, or contains only a
term whose active count is zero.

Keep the existing `ModelLoss` math. The componentization target is ownership
and observability, not a new loss formula.

**Acceptance seam:** a preflight test asserts the exact active objective names,
weights, finite values, and nonzero gradient-bearing total for the production
config.

### 7. `CheckpointManager` and component state providers - priority P1

Checkpoint persistence is currently a central method that knows how to reach
inside the lexicon, WholeSpace taxonomy, PartSpace ramsification, BPE chunker,
optimizer, RNGs, counters, and corpus manifest. This will grow every time a new
store, such as the word store, becomes persistent.

Introduce a versioned `CheckpointBundle` with top-level sections:

```text
model_state
optimizer_state
runtime_state
data_cursor_state
rng_state
component_state
launch_manifest
schema_version
```

Each stateful component should implement a small provider contract such as
`checkpoint_state()` and `restore_checkpoint_state(blob, version)`. The manager
owns atomic write, version dispatch, manifest validation, and diagnostics.
Components own their schemas and migrations.

This is consolidation as much as extraction: vocabulary and BPE migrations
remain close to those components, while the manager stops knowing their
private attributes.

**Acceptance seam:** a checkpoint saved after several streamed steps restores
parameters, optimizer moments, all RNG streams, cursor position, vocabulary,
word-store/taxonomy state, and the next-step loss exactly enough for the
project's determinism contract.

### 8. `ReconstructionPipeline` and `ForwardTrace` - priority P2

Reconstruction is the clearest place where a nominally modular forward/reverse
API is undermined by implicit state. The reverse path can require caches,
generated rules, STM snapshots, active masks, span metadata, and codebook
identity left on objects by the most recent forward.

Create an explicit `ForwardTrace` containing only the information that cannot
be recovered from the terminal idea and persistent model state. A
`ReconstructionPipeline` should then own three phases:

1. plan/unfold the symbolic or grammatical derivation;
2. reverse the space transformations using the trace;
3. render percept/byte/word identities into the requested surface form.

The trace makes two architectural questions answerable in tests: whether a
reverse is a true inverse of a transformation, and whether generation is able
to proceed from an idea without replaying a particular forward pass.

The remaining reverse/shape xfails are useful acceptance tests for this
boundary. They should remain strict, but be grouped separately from xfails
whose missing prerequisite is semantic learning rather than runtime ownership.

### 9. `ReasoningRuntime` - priority P2

Truth storage, LTM provisioning, query detection, search/reasoning, and surface
realization are application services over a trained model. They should share
the model's spaces and stores through explicit read/write interfaces, but they
do not need to live on the training model class.

Extracting them later keeps the initial work focused on large-scale training.
The useful early change is simply to stop adding new serving methods to
`BasicModel`; new reasoning behavior should enter through a dedicated facade.

## Target dependency direction

The desired dependency direction is deliberately one-way:

```text
CLI / remote launcher
        |
        v
LaunchSpec -----> RuntimeContext
        |                |
        v                v
TrainingSession ----> CheckpointManager
        |
        +----> CorpusSource / StreamCursor / BatchAdapter
        |
        v
TensorStep(StepInput, ModelRuntimeState)
        |
        +----> ForwardFrame ----> Spaces / Layers
        |              |
        |              +----> STMCoordinator
        |
        +----> ObjectiveSet
        |
        v
StepResult(next state, losses, trace, diagnostics)
                         |
                         +----> ReconstructionPipeline
                         +----> ReasoningRuntime
```

Lower layers must not import the session, launcher, or global objective
registry. The launcher should not know the internals of a vocabulary or STM
buffer. The checkpoint manager coordinates component state without interpreting
it.

## Refactoring sequence for large-corpus readiness

The order below minimizes semantic change and puts the highest-risk FineWeb
mechanisms first.

0. **Characterization tests and this ownership map.** Refactoring an implicit
   API without pinning it only moves the bugs. Gate: the existing full suite,
   strict-xfail sweep, and production preflight remain green.
1. **Frozen `LaunchSpec` plus launch manifest.** Every later component needs
   one authoritative configuration. Gate: XML/CLI/environment precedence and
   manifest parity.
2. **`CorpusSource` plus resumable `StreamCursor`.** A 20M-scale run must be
   restartable before other cleanup is valuable. Gate: interrupted versus
   uninterrupted shard-boundary equivalence.
3. **Versioned `CheckpointManager`.** Cursor and launch state need a durable
   home. Gate: exact next-step resume, including optimizer and RNG state.
4. **`TrainingSession` / `TensorStep` boundary.** This stabilizes the compiled
   hot path and makes preflight representative. Gate: eager/compiled parity,
   the zero-host-sync CUDA gate, and bounded smoke training.
5. **`ForwardFrame` and explicit runtime state.** This removes hidden per-step
   dependencies before STM or reverse behavior changes. Gate: interleaved-model
   isolation and no-stale-frame tests.
6. **Consolidated `STMCoordinator` and `ObjectiveSet`.** These affect the
   meaning and gradient of every production step. Gate: typed STM, gated-row
   no-op, reduction, and active-objective tests.
7. **`ReconstructionPipeline` with explicit trace.** This addresses the main
   cluster of architectural xfails without destabilizing launch/resume work.
   Gate: cleared-cache, dimensional, grammar-operation, and idea-only
   reconstruction tests.
8. **`ReasoningRuntime`.** This is important, but not a prerequisite for safe
   corpus-scale substrate training. Gate: truth/LTM/reasoning suites through
   the new facade.

## Tests as component contracts

The test suite should increasingly describe public boundaries rather than
private method choreography. Existing tests already provide good raw material:

- `test_fineweb_preflight.py`, `test_stream_dataset.py`,
  `test_cursor_universal.py`, and `test_data_no_byte_loss.py` define the corpus
  and cursor contract.
- `test_compiled_step_invoked.py`, `test_ir_fullgraph_compile.py`,
  `test_brick_no_sync.py`, and `test_cuda_graph_capture.py` define the tensor
  step's compile boundary.
- the `test_*stm*` files, `test_per_row_hard_reset.py`, and
  `test_padded_rows_no_op.py` define STM ownership and row-gating behavior.
- `test_reconstruction_roundtrip.py`, `test_explicit_dimensions.py`, and
  `test_stm_recon_from_cleared_cache.py` define the reverse trace contract.
- `test_ltm_consolidation.py`, `test_truth_grounded_reasoning.py`, and
  `test_reasoning.py` define the later reasoning facade.

During extraction, replace tests of private helper choreography with boundary
behavior. Keep a strict xfail only when it names intended, specific behavior;
an obsolete call sequence is not a compatibility contract.

## Definition of a successful component

A new boundary is doing real architectural work only if it satisfies all of
the following:

- It has one authoritative owner for every state value it mutates.
- Its inputs, outputs, and persistent state can be named without referring to a
  sibling object's private attributes.
- It can be constructed in a test without initializing unrelated global
  services.
- Its checkpoint state is explicit and versioned when it survives a step.
- Host-eager and compiled responsibilities do not cross accidentally.
- Its tests assert observable behavior, including negative and resume cases.
- Removing the old forwarding shim would not require callers to know the new
  component's internals.

The goal is not a larger class count. It is a runtime whose state transitions
are as explicit and typed as the cognitive spaces already are.
