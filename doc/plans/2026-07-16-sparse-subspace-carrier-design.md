# Sparse SubSpace carriers, Space-owned codebooks, and pipeline parallelism

> **Status:** implementation in progress, updated 2026-07-17. The immutable
> carrier, sparse reader, Space ownership, boundary, and bounded-executor
> foundations are implemented. Native carrier-pure `BasicModel` stages and
> removal of the legacy `SubSpace(nn.Module)` surface remain migration work.
>
> **Directive:** make `SubSpace` a clean, per-execution value passed through
> `forward()` and `reverse()`. A `Space` owns its layers, encoders, codebooks,
> parameters, and structural mutations. A `SubSpace` may hold a read-only
> reference to a Space-owned codebook, but it never owns or mutates that
> codebook. Preserve sparsity: represent a selected codebook value by its
> indices and gates until dense materialization is required. The contract must
> permit multiple microbatches to occupy different Spaces concurrently without
> changing serial semantics.

## 0. Implementation status

The implementation is deliberately split at a guarded compatibility seam:

| Area | Status |
|---|---|
| frozen/slotted carrier, typed payload/control/effects/trace | implemented in `bin/pipeline.py` |
| pure sparse materialization and restricted codebook readers | implemented and gradient-tested |
| single Space ownership of Basis, encoder, VQ Parameter, and percept store | implemented, including strict legacy checkpoint remapping |
| bounded queue executor, overlap, ordering, cancellation, telemetry, and version barrier | implemented for native carrier stages |
| same-device sparse boundaries, owner-side training materialization, inference replicas | implemented in-process; custom CUDA stream/event orchestration remains |
| legacy Space adapter | implemented in `bin/space_carrier.py`; snapshots a fresh carrier after each call |
| mid-execution exclusive mutation barriers and native recurrent-state partitioning | not yet implemented; legacy stateful stages remain serialized/guarded |
| native carrier-pure `Space.forward`/`reverse` across `BasicModel` | not yet complete |
| replacement of legacy reverse backchannels with typed trace entries | type/executor support implemented; individual legacy stages still need migration |
| removal of persistent `Space.subspace` and legacy `SubSpace(nn.Module)` | not yet complete (Phase 6) |

Legacy adapters snapshot each result before accepting another microbatch, so
their output payloads cannot alias. That alone cannot stabilize a live reader
if the old forward mutates durable state; wrappers therefore default to
`pipeline_safe=False` and are rejected until that stage passes a durability
audit. Pipeline training additionally requires `training_safe=True` after its
EMA, recurrent, vocabulary, and other durable writes move to declared
barriers. Legacy reverse wrapping is opt-in for the same reason: the caller
must first prove that the stage uses only its carrier's trace.

## 1. Why this refactor exists

The cognitive pipeline already has the right external shape:

```text
SubSpace -> Space.forward -> SubSpace
SubSpace -> Space.reverse -> SubSpace
```

The current ownership beneath that shape is mixed. `SubSpace` is an
`nn.Module`; it contains encoding objects, `Basis` objects, codebook Parameters,
registered scratch buffers, batch payloads, routing flags, errors, serial
caches, and dynamically attached reverse carriers. Each `Space` also owns a
persistent `self.subspace` and mutates it on every call. `Start()` and `End()`
must therefore distinguish model state from stale batch state and clear the
latter correctly.

This design separates two concepts:

* A **Space** is a durable transformation and the owner of learned structure.
* A **SubSpace** is one value at one point in one pipeline execution.

The refactor is successful when ownership is obvious from the type, a selected
value stays sparse until use, and a second forward cannot overwrite the value
or reverse trace produced by the first. The pipeline-parallel acceptance bar is
stronger: an interleaved schedule across Spaces must produce the same outputs,
losses, state transitions, and gradients as the characterized serial schedule.

## 2. Governing principles

### 2.1 One owner for durable state

Every Parameter, persistent buffer, codebook, encoder, and structural side
table has exactly one owning `Space` or stateful layer. It appears under that
owner in `state_dict()` and optimizer discovery.

`SubSpace` owns no Parameter, persistent buffer, encoder, codebook, vocabulary,
or recurrent store.

### 2.2 Sparse by representation

When a value is a selection from a codebook, its canonical representation is:

```text
read-only codebook capability + indices + activation + non-codebook bands
```

It is not a stored `[B, N, D]` duplicate of the selected rows. Dense gathering
is an explicit, local operation performed only by a consumer that needs the
dense tensor. Materialization does not cache its result on the carrier or the
codebook.

This specification does **not** require PyTorch sparse tensor layouts.
Codebook indices and typed row restrictions are the relevant sparsity. A
`torch.sparse_*` representation should be introduced only where a benchmark
shows a benefit.

### 2.3 One canonical payload form

A carrier contains exactly one of:

1. a dense event;
2. factored dense `what`/`where`/`when` values; or
3. a sparse codebook selection plus any bands not stored in that codebook.

It never carries a sparse selection and its dense materialization at the same
time. The payload type, not flags such as `muxed`, `codebook_slot`, or
`_demuxed`, identifies the representation.

### 2.4 No hidden communication

Pipeline communication travels on typed carrier fields or explicit function
arguments. It does not travel through dynamically attached attributes on the
model, a `Space`, or a `SubSpace`.

There is no generic `metadata: dict`. New metadata requires a named field, a
documented lifetime, and at least one test that identifies its producer and
consumer.

### 2.5 Shallow immutability, functional stage boundaries

Carrier objects are frozen and slotted: fields cannot be added or rebound.
PyTorch tensors are not recursively immutable, so stages must treat an incoming
carrier as read-only and return a new carrier. In-place tensor operations are
permitted only on state owned by the receiving `Space` or a declared stateful
component, never on an incoming carrier payload.

### 2.6 Stage isolation and version coherence

Every in-flight carrier has an explicit execution/microbatch identity. A Space
may process carriers independently unless its contract declares ordered
recurrent state. Shared Parameters and codebooks are read at one pinned version
for the duration of a pipeline training step. Optimizer updates and structural
codebook mutations occur only at a drain barrier, or begin a new explicitly
versioned execution after all readers of the old version have completed.

## 3. Ownership model

| Object | Owns | May mutate |
|---|---|---|
| `Space` | schema, layers, encoders, codebooks, structural side tables | its own structure through named methods |
| `SubSpace` | one batch payload, typed pipeline control, emitted effects, reverse trace | nothing after construction |
| codebook reader | no storage; read capability over one Space-owned codebook | nothing |
| stateful layer | STM, discourse state, clocks, recurrent caches | its declared recurrent state |
| optimizer | parameter update state | Parameters discovered under owning modules |
| training session | epochs, batches, checkpoints, reporting | orchestration state |
| pipeline executor | stage placement, bounded queues, schedule, barriers, cancellation | execution state, never model state |
| boundary adapter | representation conversion and device transfer | fresh carrier copies only |

The optimizer is an intentional second writer to trainable codebook Parameters.
The Space remains the structural owner: insertion, growth, replacement, EMA
updates, category changes, and other no-grad mutations occur only through the
Space or a child module to which it explicitly delegates ownership.

## 4. Proposed carrier contract

The names below specify responsibilities. Final implementation names may vary
only if the same separation remains visible.

```python
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class SubSpaceSchema:
    role: str
    n_what: int
    n_where: int
    n_when: int
    geometry: str


@dataclass(frozen=True, slots=True)
class DenseEvent:
    event: Tensor
    activation: Tensor | None = None


@dataclass(frozen=True, slots=True)
class FactoredEvent:
    what: Tensor | None = None
    where: Tensor | None = None
    when: Tensor | None = None
    activation: Tensor | None = None


@dataclass(frozen=True, slots=True)
class CodebookIdentity:
    owner_path: str
    structure_version: int
    parameter_version: int


@dataclass(frozen=True, slots=True)
class SelectedEvent:
    reader: "CodebookReader"
    indices: Tensor
    slot: SelectionSlot = SelectionSlot.EVENT
    activation: Tensor | None = None
    where: Tensor | None = None
    when: Tensor | None = None


Payload: TypeAlias = DenseEvent | FactoredEvent | SelectedEvent


@dataclass(frozen=True, slots=True)
class PipelineAddress:
    execution_id: int
    microbatch_id: int
    parameter_version: int
    stream_ids: Tensor | None = None
    sequence_step: int = 0


@dataclass(frozen=True, slots=True)
class PipelineControl:
    address: PipelineAddress
    valid_mask: Tensor | None = None
    reset_mask: Tensor | None = None
    row_gate: Tensor | None = None
    recurrent_pass: int = 0


@dataclass(frozen=True, slots=True)
class LossTerm:
    name: str
    value: Tensor
    weight: float = 1.0
    category: str = "model"


@dataclass(frozen=True, slots=True)
class PipelineEffects:
    losses: tuple[LossTerm, ...] = ()
    diagnostics: tuple["Diagnostic", ...] = ()
    deferred_mutations: tuple["DeferredMutation", ...] = ()


@dataclass(frozen=True, slots=True)
class ReverseTrace:
    entries: tuple["TraceEntry", ...] = ()


@dataclass(frozen=True, slots=True)
class SubSpace:
    schema: SubSpaceSchema
    payload: Payload
    control: PipelineControl
    effects: PipelineEffects = field(default_factory=PipelineEffects)
    trace: ReverseTrace = field(default_factory=ReverseTrace)
```

The top-level carrier stays small. Closely related optional values live in
typed subrecords rather than accumulating as top-level fields. Empty tuples and
`None` are canonical; empty placeholder tensors are not allocated merely to
fill the contract.

### 4.1 Schema ownership

The owning Space constructs one immutable `SubSpaceSchema` and may share that
schema object with every carrier it emits. Encoders remain on the Space. The
schema reports dimensions and geometry but performs no transformation and
contains no mutable lookup tables.

### 4.2 Payload laws

* `DenseEvent` is used when the event is intrinsically dense and not a
  codebook selection.
* `FactoredEvent` is used when the bands must remain separately addressable.
* `SelectedEvent` is used when at least the `what` or full event content is
  represented by codebook rows.
* A selected event's `indices` are integral. They may retain a rectangular
  `[B, N]` shape for compilation, but padding and validity are expressed by
  activation/control masks and do not require allocated dense zero vectors.
* `where` and `when` remain independent tensors when they are not part of the
  selected codebook row. They are not repeatedly concatenated into cached event
  slabs.

### 4.3 Materialization laws

`SubSpace.materialize(mode)` may remain as a convenience API, but it is a pure
read:

```text
materialize(selected) = reader.lookup(indices), then apply named bands/gates
```

It does not write to the carrier, the reader, or the owning Space. Repeated
materialization may recompute a gather; a caller that needs the tensor multiple
times keeps it in a local variable for the duration of that operation.

`mode` must become a closed enum or a set of named methods. Unsupported modes
raise rather than silently falling through to another representation.

## 5. Read-only codebook capability

Freezing a reference does not make its target immutable. A carrier must not
receive the `Codebook` module or its Parameter directly. It receives a
capability that exposes operations, not storage:

```python
class CodebookReader(Protocol):
    @property
    def identity(self) -> "CodebookIdentity": ...

    @property
    def size(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def lookup(self, indices: Tensor) -> Tensor: ...

    def nearest(
        self,
        values: Tensor,
        *,
        allowed_rows: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...


class ParameterVersionOwner(Protocol):
    @property
    def codebook_parameter_version(self) -> int: ...

    def set_codebook_parameter_version(self, version: int) -> int: ...
```

Specialized operations use narrower capability protocols, for example
`PropertyCodebookReader.materialize_property(...)`. A general reader does not
grow every domain-specific method.

The reader does not expose `W`, `.data`, `getW`, `setW`, `replace_W`, a raw
prototype tensor, the owning Space, or a generic escape hatch. `lookup()` must
remain differentiable with respect to a trainable Space-owned prototype.

`nearest(..., allowed_rows=...)` preserves typed sparsity. Callers pass the
small candidate row set rather than gathering the entire vocabulary and
masking it afterward.

### 5.1 View lifetime and versioning

A reader is an in-process view, not checkpoint data. It may remain valid across
updates to its owner, but its identity changes: insertion/growth/replacement
increments `structure_version`, while optimizer/EMA content updates increment
`parameter_version`. Derived caches key on every version that can change their
answer. Carriers are not serialized, so checkpoint loading never reconstructs
a stale carrier-side view.

`CodebookIdentity` names the authoritative owner and version without exposing
it. A boundary adapter uses that identity to decide whether a selected payload
can be rebound to an exact read-only replica on the target device. A replica is
derived execution state: it is not a Parameter, optimizer member, checkpoint
owner, or mutation target.

## 6. Space contract

```python
class Space(nn.Module):
    def forward(self, incoming: SubSpace) -> SubSpace: ...
    def reverse(self, outgoing: SubSpace) -> SubSpace: ...
```

The laws are:

1. `forward` and `reverse` do not mutate their carrier argument.
2. Each returns a fresh carrier; no result is parked on `self.subspace`.
3. Dense temporaries are locals and die at the stage boundary unless the
   reverse trace explicitly requires them.
4. Space-owned learned or recurrent state may change only at a named update
   seam. Such side effects are documented by that Space.
5. Empty input returns a valid empty carrier of the correct schema, not the
   incoming carrier of a different role.
6. Device placement follows the owning Space and incoming tensors. A carrier's
   `to(device)`—if provided—returns a new carrier and does not move the reader's
   owner.

Removing persistent result carriers makes ordinary `SubSpace.Start()` and
`SubSpace.End()` unnecessary. `Reset()` remains only on objects that own real
cross-step state, such as STM or discourse components.

During migration, `SpaceCarrierMixin.as_pipeline_stage(...)` is the sole
legacy wrapper. It constructs an unowned legacy input adapter, calls the old
method, and immediately snapshots the result into a fresh immutable carrier.
Its defaults `pipeline_safe=False` and `training_safe=False` are hard guards,
not documentation. Every executor rejects an unaudited legacy stage; a
training executor also rejects a stage until its durable writes have explicit
commit semantics. `reverse_call` is absent by default and is enabled only for
a stage whose reverse inputs are fully represented by `ReverseTrace`.

## 7. Pipeline-parallel execution between Spaces

The ownership refactor must support a bounded pipeline in which different
microbatches occupy different Spaces concurrently:

```text
time ->

InputSpace       A  B  C
PartSpace           A  B  C
ConceptualSpace        A  B  C
WholeSpace                A  B  C
OutputSpace                  A  B  C
```

This is a throughput schedule across independent microbatches. It does not
remove the mathematical dependencies within one microbatch and does not imply
that two stages on one saturated device can execute simultaneously.

### 7.1 Stage and executor ownership

Each Space is one logical stage with one authoritative device placement. The
same device may host several adjacent stages; a later partitioning pass may
place stages on different devices without changing their `forward`/`reverse`
contracts.

The `PipelineExecutor`, not a Space, owns:

* the ordered stage list and device placement;
* one bounded input queue per stage;
* microbatch splitting and result reassembly;
* the fill/drain or 1F1B schedule;
* device streams, transfer events, and backpressure;
* parameter/codebook version barriers;
* cancellation and failure propagation;
* pipeline timing and memory telemetry.

A Space never reads another stage's queue and never retains a carrier after its
call returns. Queue capacity is explicit and bounded; an upstream stage waits
instead of allowing unbounded in-flight activations.

### 7.2 Addressing and ordering

Every carrier has one `PipelineAddress`:

* `execution_id` identifies the top-level inference request or training step;
* `microbatch_id` identifies its position in that execution;
* `parameter_version` pins the Parameter snapshot used by the execution;
* `stream_ids` identify recurrent document/dialogue streams by batch row;
* `sequence_step` orders values within each recurrent stream.

Stages may interleave or complete independent stream IDs in any order. Values
for the same stream are committed in increasing `sequence_step` order. The
executor reassembles externally visible results in `microbatch_id` order unless
the caller explicitly requests unordered streaming output.

An address is control identity, not learned content. It is never embedded into
the event tensor or used as a model feature.

### 7.3 Reference schedules

The first training implementation uses a deterministic fill/drain schedule:

1. split one logical batch into microbatches;
2. run the forward wave through all Spaces;
3. when reconstruction is enabled, run the cognitive `Space.reverse()` wave in
   reverse Space order and compute its losses;
4. run autograd backward through the microbatches;
5. drain every stage and boundary;
6. commit deferred state changes in address order;
7. call `optimizer.step()` once;
8. increment `parameter_version`.

All microbatches in the step therefore observe the same Parameters and
codebook structure. A 1F1B schedule may replace fill/drain only after tests
prove identical results and gradients for the supported model/configuration.

Terminology is strict:

* `Space.reverse()` is the model's cognitive/reconstruction direction;
* autograd `backward()` computes training gradients;
* reverse pipeline scheduling orders `Space.reverse()` stages;
* backward scheduling orders gradient computation.

These operations may travel in the same physical direction but are not the
same contract and must not share an implicit cache.

### 7.4 Boundary adaptation and device transfer

A stage emits a carrier local to its device. The boundary adapter creates the
carrier accepted by the next stage. It chooses exactly one representation:

1. **Same-device sparse pass-through.** Retain `SelectedEvent`; no gather or
   tensor copy is required.
2. **Owner materialization.** Gather once on the source/codebook device, retain
   only the bands required by the consumer, and transfer a `DenseEvent` or
   `FactoredEvent`. This is the default cross-device training path because the
   gather remains connected to the authoritative Parameter for autograd.
3. **Replica rebind.** Transfer indices/gates/bands and construct a new
   `SelectedEvent` bound to an exact, read-only target-device replica with the
   same `CodebookIdentity`. This is permitted for inference. Training use
   requires a separately specified gradient-reduction contract.

The resulting carrier never contains both the source sparse selection and the
dense transferred value. Device transfer is non-blocking when supported; the
consumer waits on an explicit event rather than forcing a process-wide device
synchronization.

Reader objects themselves are local capabilities and do not cross a device or
process boundary. A boundary either materializes their value or rebinds the
selection to a verified target-local reader. A version mismatch is an error;
it never falls back silently to a different codebook.

The first executor may be single-process/multi-device. Its transport packet
must nevertheless contain only typed immutable scalars and tensors. On the
replica path the source reader is removed before transfer; the packet carries
`CodebookIdentity` plus indices/gates/bands, and the target adapter attaches its
local reader only after validating the identity. A future process boundary
therefore does not require a new model contract.

### 7.5 Read-only codebook replicas

A replica is a performance cache controlled by the authoritative owner and the
executor:

* it has the same identity/version/content as the owner snapshot;
* it is not registered as a Parameter or persistent buffer;
* it is absent from optimizer groups and checkpoints;
* it has no structural or content mutation API;
* it is refreshed only while the pipeline is drained;
* it is discarded when its version becomes stale.

Inference may use replicas to keep queued carriers sparse across devices.
Training defaults to source materialization. An autograd-aware training replica
is out of scope until its gradient reduction, ordering, and optimizer ownership
are separately specified and tested.

### 7.6 Mutation and commit barriers

No Parameter, codebook structure, vocabulary, truth store, or persistent
recurrent state may change invisibly while older carriers are in flight.

Stage work that can be deferred emits a typed `DeferredMutation` in
`PipelineEffects`. The executor commits those mutations after drain in
`(execution_id, sequence_step, microbatch_id)` order. Sequence order precedes
microbatch identity so a later-numbered microbatch cannot commit a newer
recurrent step before an earlier step. Examples include
structural codebook promotion, vocabulary growth, reporting records, and truth
store writes.

If a mutation is mathematically required to compute the current output, the
stage declares an exclusive mutation barrier. The executor drains readers of
the old version, applies the mutation at its owner, refreshes/discards replicas,
increments the version, and resumes. It does not race a write against in-flight
lookups.

EMA/VQ updates must be classified explicitly as either optimizer-like updates
at the training-step barrier or ordered structural mutations. Their current
in-forward placement is not inherited implicitly.

### 7.7 Recurrent state

Stateful components retain ownership of STM, discourse, clocks, and other
cross-step state, but state is partitioned by `stream_ids`. Independent streams
may overlap. Two values for the same stream may not commit out of order.

The stage applies `reset_mask` before reading recurrent state for the addressed
rows. A recurrent transition either:

* produces a typed deferred state update committed in sequence order; or
* executes under a stream-local lock/order gate when the next output depends on
  the newly written state.

There is no batch-global fallback for a row-addressed recurrent mechanism.
Configurations that intentionally define one global recency window declare
that stage globally ordered; the executor exposes the resulting serialization
instead of pretending that stage is parallel.

### 7.8 Reverse and activation lifetime

Each microbatch retains only the payload and sparse trace required for its
cognitive reverse and autograd backward. The executor may release a carrier as
soon as both consumers are complete. Activation checkpointing/recomputation may
be added later, but it must reproduce the same codebook/parameter versions.

Semantic reverse runs through Spaces in the opposite logical order using the
trace carried by that microbatch. It never reads trace or activations belonging
to another in-flight microbatch.

### 7.9 Failure, cancellation, and observability

A failure is attributed to one `PipelineAddress` and stage. The executor stops
admitting work for that execution, cancels or drains its queued carriers,
releases retained graphs, and does not partially commit its deferred mutations.
Failures from another execution are isolated when no shared version barrier was
entered.

The executor records, per stage and boundary:

* compute time;
* queue wait and idle/bubble time;
* transfer time and bytes;
* sparse materialization time and bytes;
* peak queue depth and activation memory;
* version-barrier and recurrent-order stalls.

This telemetry supplies the measured upper bound for stage balancing:

```text
ideal steady-state speedup = sum(stage times) / max(stage time)
```

Pipeline parallelism is not considered successful merely because multiple
queues exist; the measurements must show useful overlap without semantic drift.

## 8. Sparse reverse trace

Exact reversal sometimes needs information that cannot be recovered from the
output alone. That information travels in `ReverseTrace`, subject to these
rules:

* Identity and analytically invertible stages append no trace entry.
* A nontrivial stage appends one typed entry bearing its stable stage identity.
* Entries prefer indices, masks, choices, spans, and seeds over dense copies.
* A trace never stores a codebook prototype matrix or mutable module reference.
* A dense intermediate is permitted only when no smaller sufficient statistic
  exists; its necessity and shape are documented on its entry type.
* `reverse` consumes the entry addressed to its stage and returns the remaining
  trace. It does not consult a cache from "the most recent forward."

The target space bound is proportional to the active batch/slot structure,
not the vocabulary size:

```text
trace storage = O(active selections + non-invertible local choices)
```

not `O(number of codebook rows * codebook width)`.

Likely migrations include:

| Current backchannel | Target |
|---|---|
| `_bind_carrier` | typed bind trace entry |
| `_stm_single_S` | explicit root/derivation trace entry |
| `_ir_mask_positions` | `PipelineControl` or masked-prediction trace entry |
| `_serial_row_gate` | `PipelineControl.row_gate` |
| `_prev_cs_for_ps`, `_prev_cs_for_ss` | explicit feedback payload/control field |
| `_staged_analysis_spans`, `_staged_analysis_kinds` | typed analysis trace/control |

The exact classification of feedback as payload versus control must be fixed
before its migration. It must not become a general context dictionary.

## 9. Effects and loss sparsity

Losses emitted by a stage travel as immutable `LossTerm` records. A stage emits
only terms it actually computed; it does not preallocate zero terms for the
global inventory. The training boundary reduces the tuple into the existing
reporting taxonomy.

This makes objective ownership visible and prevents a process-global registry
from serving as a hidden communication bus. During migration, an adapter may
fold carrier terms into `TheError`, but the carrier is authoritative for newly
migrated stages.

Diagnostics obey the same rule: emit a small typed record when enabled. Debug
mode must not change the mathematical payload or force dense materialization
of every selected event.

`DeferredMutation` records are likewise sparse and typed: only a stage that
requests a durable change emits one. The training/pipeline boundary, not an
arbitrary downstream Space, is responsible for ordered commit.
`DeferredMutation` denotes a closed union of owner-specific records, not a
generic `(kind, payload)` bag and not a Python closure. Each variant names its
owner type, tensor/scalar fields, validation law, and commit method in the
owner/executor adapter.

## 10. Serialization, optimizer, and compilation

### 10.1 Serialization

Only owners appear in `state_dict()`:

* codebook Parameters and persistent metadata under their Space;
* recurrent state under its stateful component when intentionally persistent;
* no carrier payload, trace, reader, batch mask, or diagnostic.

Checkpoint loading provides an explicit key-remapping layer for legacy
SubSpace-owned codebook paths. Saving uses only the new owner paths after the
corresponding Space migrates.

### 10.2 Optimizer discovery

Every trainable codebook Parameter appears exactly once in
`model.named_parameters()`. Moving ownership must neither duplicate a Parameter
nor silently drop it from existing optimizer groups. Parameter identity is
preserved where optimizer-state continuity requires it.

Pipeline training pins all microbatches in one optimizer step to the same
`parameter_version`. `optimizer.step()` is illegal while carriers or backward
work for that version remain in flight.

### 10.3 Compilation

The tensor payload and selection shapes are the compiled data boundary.
Capability calls used inside a compiled region must lower to ordinary tensor
operations without Python-side structural mutation. Structural codebook growth
and trace records containing host objects remain outside compiled regions.

Compilation is per stage/device. Boundary copies and queue operations stay
outside compiled stage functions. A compiled stage returns the same typed
tensor fields as eager execution so scheduling mode does not change the carrier
contract.

## 11. Cleanliness constraints

The following are forbidden in the final contract:

* `SubSpace(nn.Module)`;
* codebook/Basis ownership on `SubSpace`;
* persistent `Space.subspace` result objects;
* `SubSpace.__dict__` and dynamically attached fields;
* generic `metadata`, `extras`, or `cache` dictionaries;
* both indices and a cached dense gather for the same value;
* raw codebook `W`/prototype access through the carrier;
* `getattr(..., default)` as the contract for pipeline state;
* broad exception handling that changes representation or disables a path;
* cleanup methods required only because the previous batch lived on a durable
  module;
* unbounded stage queues or unaddressed in-flight carriers;
* optimizer/codebook mutation while an older version remains in flight;
* passing a live source-device reader into a target-device stage;
* batch-global recurrent writes for a state contract declared per stream.

Compatibility adapters may temporarily violate a named item, but every such
adapter has a removal phase and may not be used by new code.

## 12. Acceptance tests

The refactor is complete only when the following are pinned:

### Ownership

1. A `SubSpace` has no Parameters, buffers, child modules, or `state_dict()`.
2. Every codebook Parameter appears exactly once under its owning Space.
3. A reader has no mutation method and exposes no raw prototype storage.
4. Structural codebook mutation is reachable only through the owner seam.

### Sparsity

5. A selected carrier stores indices/gates/bands but no `[B, N, D]` selected
   row copy.
6. `materialize()` gathers the correct values and leaves no dense cache.
7. `nearest(allowed_rows=...)` examines only the permitted candidate set.
8. Identity forward/reverse adds no trace entry.
9. Debugging and loss reporting do not densify an otherwise sparse carrier.

### Correctness

10. Gradients through `reader.lookup(indices)` reach the owner Parameter.
11. Forward/reverse round trips match the characterized legacy output for each
    migrated Space.
12. Running two forwards before reversing either leaves both outputs valid and
    independently reversible.
13. A second batch cannot observe payload, masks, losses, or trace from the
    first.
14. Empty and partially active batches preserve schema and mask semantics.

### Lifecycle and persistence

15. Carrier creation and collection require no `Start()`/`End()` cleanup.
16. Old checkpoints load through the key remapper; a new save/reload preserves
    outputs and optimizer membership.
17. Device movement of a Space moves its codebook once; carrier movement does
    not duplicate or move the owner.
18. Eager and compiled execution agree for selected and dense payloads.

### Contract cleanliness

19. Frozen/slotted carriers reject unknown attributes.
20. A static audit finds no new `self._staged_*` or dynamic pipeline fields on
    model/Space objects.
21. Each carrier field has a named producer, consumer, and lifetime in its type
    docstring.

### Pipeline parallelism

22. A deliberately interleaved two-microbatch schedule matches serial outputs,
    losses, persistent state, and gradients.
23. Instrumented stages prove that microbatch B may enter an upstream Space
    while microbatch A occupies a downstream Space.
24. Every microbatch in one training step observes the same
    `parameter_version`; `optimizer.step()` is rejected before drain.
25. Cross-device training materializes a selected value on its owner device,
    transfers only the required dense/factored payload, and preserves gradient
    flow to the authoritative codebook Parameter.
26. Inference replica rebinding preserves values while transferring only
    indices/gates/bands; a stale or mismatched replica raises.
27. Stage queues enforce their configured bound and final results reassemble in
    `microbatch_id` order.
28. Independent recurrent stream IDs overlap without state leakage, while two
    steps for one stream commit in `sequence_step` order.
29. Structural codebook mutation is deferred until drain, increments the
    version, and invalidates old replicas before the next execution starts.
30. Cognitive reverse for several in-flight microbatches consumes only each
    carrier's own trace and matches serial reverse results.
31. Cancellation releases queued carriers/graphs and commits none of the
    cancelled execution's deferred mutations.
32. Stage telemetry reports compute, queue wait, transfer, materialization,
    barrier wait, queue depth, and activation bytes without densifying queued
    sparse carriers.

## 13. Migration plan

This is not a big-bang rewrite. Each phase lands with green characterization
tests and leaves the next phase mechanically smaller.

### Phase 0 — characterize and inventory

* Pin materialization, gradient, state-dict, optimizer, reset, and reverse
  behavior for each Space archetype.
* Inventory all dynamically attached pipeline fields and classify each as
  payload, control, effect, reverse trace, recurrent state, or dead state.
* Record legacy checkpoint keys for every codebook-bearing slot.

**Exit:** every current backchannel and durable object has one proposed owner.

### Phase 1 — introduce values and readers behind adapters

* Add frozen/slotted schema, payload, control, effect, and trace types.
* Add restricted codebook readers backed by the existing codebooks.
* Implement pure materialization for the new payload variants.
* Keep current `SubSpace` and `Space.subspace` as compatibility adapters.

**Exit:** new unit tests use no raw `getW()`/prototype access and prove lookup
gradient flow.

### Phase 2 — move ownership by archetype

Move one storage archetype at a time:

1. tensor-only spaces;
2. fixed-size `Codebook` owners;
3. `Embedding`/lexicon owners;
4. dynamic stores and their structural side tables.

For each owner, preserve Parameter identity or provide an explicit optimizer
state migration, remap old checkpoint keys, and delete the SubSpace-owned copy.

**Exit:** no codebook or encoder is owned by a `SubSpace`.

### Phase 3 — fresh forward carriers

* Convert Space methods to construct and return fresh carriers.
* Replace `self.subspace` reads with the incoming/outgoing carrier.
* Move masks, row gates, feedback, and effects into typed fields.
* Remove per-batch `Start()`/`End()` behavior as each Space becomes carrier-pure.

**Exit:** two-forward/two-reverse isolation passes for the full pipeline.

### Phase 4 — sparse reverse trace

* Replace reconstruction and grammar backchannels with typed trace entries.
* Delete dense caches when an index/mask/choice is sufficient.
* Make reverse fail clearly when a required trace entry is absent; do not fall
  back silently to the most recent Space cache.

**Exit:** cleared-cache and interleaved reverse tests pass without model-level
staging attributes.

### Phase 5 — pipeline executor and boundary adapters

* Put a `PipelineAddress` on every carrier and pin one parameter version per
  execution.
* Implement the deterministic fill/drain executor with bounded stage queues.
* Land same-device interleaving first to prove semantic isolation independent
  of hardware concurrency.
* Add source-materializing cross-device boundaries, explicit transfer events,
  and stage telemetry.
* Add inference-only replica rebinding and version invalidation.
* Move durable in-forward side effects to deferred commits or declared
  exclusive barriers; enforce recurrent stream ordering.

**Exit:** pipeline acceptance tests 22–32 pass; unsupported recurrent/global
stages report their serialization rather than racing.

### Phase 6 — remove compatibility surface

* Make the value carrier the only `SubSpace`.
* Remove its `nn.Module` inheritance and the legacy Basis-slot API.
* Remove `Space.subspace`, generic materialization modes, and compatibility
  checkpoint writes.
* Retain legacy checkpoint reads for the documented support window.

**Exit:** all cleanliness constraints and the static audit pass.

## 14. Non-goals

This refactor does not:

* change the cognitive order, geometry, grammar, or meaning of any Space;
* introduce a second pipeline envelope beside `SubSpace`;
* require dense payloads to become PyTorch sparse tensors;
* redesign STM or discourse algorithms, beyond giving their state an explicit
  owner;
* change codebook learning rules, row identity, or candidate restrictions;
* combine this ownership migration with performance-driven tensor rewrites;
* guarantee a throughput improvement before stage timing demonstrates useful
  overlap;
* require multi-process distributed autograd in the first executor;
* use read-only codebook replicas for training before an explicit gradient
  reduction design exists;
* reorder dependent steps within one recurrent stream.

Behavioral or mathematical changes discovered during migration are recorded as
separate work. The acceptance baseline for this design is representation and
ownership cleanup with byte-identical behavior wherever the existing path is
deterministic.

## 15. Initial implementation slice (landed)

The first code change after approval was deliberately small:

1. define `CodebookReader` and an internal read-only view over the existing
   `Codebook`;
2. define `CodebookIdentity` and `SelectedEvent` with pure, uncached
   materialization;
3. test lookup values, gradient flow, allowed-row nearest search, absence of a
   dense cache, and absence of mutation APIs;
4. adapt one fixed-size codebook read path without moving ownership yet.

That slice validated the sparse representation and capability boundary before
the ownership remapper, legacy adapters, boundaries, and executor were added.
The remaining work is tracked in the implementation-status table and Phases
3, 4, and 6 above; the existence of the executor does not make an unsafe
legacy training stage pipeline-ready.
