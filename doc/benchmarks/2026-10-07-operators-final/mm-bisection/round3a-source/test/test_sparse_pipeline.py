"""Acceptance tests for the sparse carrier and pipeline runtime."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from threading import Event, Thread

import pytest
import torch
from torch import nn

from pipeline import (
    BoundaryAdapter,
    CancellationToken,
    CodebookIdentity,
    CodebookReader,
    DeferredMutation,
    DenseEvent,
    FactoredEvent,
    LossTerm,
    MaterializeMode,
    PipelineAddress,
    PipelineControl,
    PipelineEffects,
    PipelineExecutionCancelled,
    PipelineExecutionError,
    PipelineExecutor,
    PipelineStage,
    ReplicaRegistry,
    ReverseTrace,
    SelectedEvent,
    SelectionSlot,
    SubSpace,
    SubSpaceSchema,
    TraceEntry,
    make_codebook_reader,
)


SCHEMA = SubSpaceSchema(role="test", n_what=2, n_where=1, n_when=1)


class _Basis(nn.Module):
    def __init__(self, rows: torch.Tensor, *, use_dot_product: bool = False):
        super().__init__()
        self.W = nn.Parameter(rows.clone())
        self.use_dot_product = use_dot_product
        self.lookup_calls = 0

    def prototype(self) -> torch.Tensor:
        return self.W

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        self.lookup_calls += 1
        return self.W[indices]


def _reader(
    basis: _Basis,
    *,
    owner: str = "model.test.codebook",
    version: int = 0,
):
    return make_codebook_reader(
        basis,
        owner_path=owner,
        identity=lambda: CodebookIdentity(owner, parameter_version=version),
    )


def _carrier(
    microbatch_id: int,
    value: float = 0.0,
    *,
    version: int = 0,
    stream_ids: torch.Tensor | None = None,
    sequence_step: int = 0,
) -> SubSpace:
    address = PipelineAddress(
        execution_id=7,
        microbatch_id=microbatch_id,
        parameter_version=version,
        stream_ids=stream_ids,
        sequence_step=sequence_step,
    )
    event = torch.full((1, 1, SCHEMA.event_width), value)
    return SubSpace(SCHEMA, DenseEvent(event), PipelineControl(address))


def _add(carrier: SubSpace, amount: float) -> SubSpace:
    event = carrier.materialize(MaterializeMode.EVENT)
    assert event is not None
    return carrier.with_payload(DenseEvent(event + amount))


def test_carrier_is_frozen_slotted_and_not_a_module():
    carrier = _carrier(0)

    assert not isinstance(carrier, nn.Module)
    assert not hasattr(carrier, "state_dict")
    assert not hasattr(carrier, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        carrier.untyped_pipeline_cache = torch.ones(1)  # type: ignore[attr-defined]


def test_selected_materialization_is_pure_sparse_and_differentiable():
    basis = _Basis(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )
    )
    payload = SelectedEvent(
        _reader(basis),
        torch.tensor([[2, 0]]),
        activation=torch.tensor([[0.5, 1.0]]),
    )
    carrier = SubSpace(SCHEMA, payload, _carrier(0).control)

    first = carrier.materialize(MaterializeMode.EVENT)
    second = carrier.materialize(MaterializeMode.EVENT)
    assert torch.equal(first, basis.W[torch.tensor([[2, 0]])])
    assert torch.equal(second, first)
    assert basis.lookup_calls == 2
    assert isinstance(carrier.payload, SelectedEvent)
    assert not hasattr(carrier.payload, "event")
    assert not hasattr(carrier, "_materialized")

    active = carrier.materialize(MaterializeMode.ACTIVE)
    assert torch.equal(active, first * torch.tensor([[[0.5], [1.0]]]))
    assert active is not None
    active.sum().backward()
    assert basis.W.grad is not None
    assert torch.equal(basis.W.grad[1], torch.zeros(4))
    assert torch.count_nonzero(basis.W.grad[0]) == 4
    assert torch.count_nonzero(basis.W.grad[2]) == 4


def test_reader_lookup_lowers_through_torch_compile():
    class CompileBasis(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.arange(12.0).reshape(3, 4))

        def prototype(self):
            return self.W

        def lookup(self, indices):
            return self.W[indices]

    basis = CompileBasis()
    reader = make_codebook_reader(basis, owner_path="compiled.codebook")

    def lookup(indices):
        return reader.lookup(indices)

    compiled = torch.compile(lookup, backend="eager", fullgraph=True)
    indices = torch.tensor([[2, 0]])
    torch.testing.assert_close(compiled(indices), lookup(indices))


def test_selected_what_retains_factored_where_and_when():
    basis = _Basis(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    payload = SelectedEvent(
        _reader(basis),
        torch.tensor([[1]]),
        slot=SelectionSlot.WHAT,
        where=torch.tensor([[[5.0]]]),
        when=torch.tensor([[[6.0]]]),
    )
    carrier = SubSpace(SCHEMA, payload, _carrier(0).control)

    assert torch.equal(
        carrier.materialize(MaterializeMode.EVENT),
        torch.tensor([[[3.0, 4.0, 5.0, 6.0]]]),
    )
    assert torch.equal(carrier.materialize(MaterializeMode.WHAT), torch.tensor([[[3.0, 4.0]]]))
    assert torch.equal(carrier.materialize(MaterializeMode.WHERE), payload.where)
    assert torch.equal(carrier.materialize(MaterializeMode.WHEN), payload.when)


def test_factored_materialization_has_no_cache_and_validates_leading_shape():
    payload = FactoredEvent(
        what=torch.ones(2, 3, 2),
        where=torch.ones(2, 3, 1) * 2,
        when=torch.ones(2, 3, 1) * 3,
    )
    carrier = SubSpace(SCHEMA, payload, _carrier(0).control)
    assert carrier.materialize(MaterializeMode.EVENT).shape == (2, 3, 4)
    assert isinstance(carrier.payload, FactoredEvent)

    bad = replace(payload, where=torch.ones(2, 4, 1))
    with pytest.raises(ValueError, match="leading shapes"):
        replace(carrier, payload=bad).materialize(MaterializeMode.EVENT)


def test_sparse_empty_check_does_not_lookup_rows():
    basis = _Basis(torch.ones(2, 4))
    carrier = SubSpace(
        SCHEMA,
        SelectedEvent(_reader(basis), torch.empty(1, 0, dtype=torch.long)),
        _carrier(0).control,
    )
    assert carrier.is_empty()
    assert basis.lookup_calls == 0


def test_reader_is_a_capability_without_mutation_or_storage_surface():
    basis = _Basis(torch.arange(12.0).reshape(3, 4))
    reader = _reader(basis)

    assert isinstance(reader, CodebookReader)
    assert reader.size == 3
    assert reader.width == 4
    for forbidden in (
        "W",
        "data",
        "prototype",
        "getW",
        "setW",
        "replace_W",
        "owner",
    ):
        assert not hasattr(reader, forbidden)


def test_nearest_honors_allowed_rows():
    basis = _Basis(torch.tensor([[0.0], [10.0], [20.0]]))
    reader = _reader(basis)

    unrestricted, _ = reader.nearest(torch.tensor([[9.0]]))
    restricted, selected = reader.nearest(
        torch.tensor([[9.0]]), allowed_rows=torch.tensor([0, 2])
    )
    assert unrestricted.item() == 1
    assert restricted.item() == 0
    assert selected.item() == 0.0


def test_reader_identity_is_live_but_replica_resolution_is_version_exact():
    basis = _Basis(torch.ones(2, 4))
    version = {"parameter": 0}
    reader = make_codebook_reader(
        basis,
        owner_path="model.test.codebook",
        identity=lambda: CodebookIdentity(
            "model.test.codebook", parameter_version=version["parameter"]
        ),
    )
    assert reader.identity.parameter_version == 0
    version["parameter"] = 1
    assert reader.identity.parameter_version == 1

    replicas = ReplicaRegistry()
    replicas.install(reader.identity, basis.W, device="cpu")
    assert replicas.resolve(reader.identity, "cpu").identity.parameter_version == 1
    with pytest.raises(LookupError, match="no exact replica"):
        replicas.resolve(CodebookIdentity("model.test.codebook", parameter_version=0), "cpu")

    replicas.invalidate_owner("model.test")
    with pytest.raises(LookupError, match="no exact replica"):
        replicas.resolve(reader.identity, "cpu")


class _RemoteReader:
    def __init__(self, identity: CodebookIdentity):
        self._identity = identity

    @property
    def identity(self) -> CodebookIdentity:
        return self._identity

    @property
    def size(self) -> int:
        return 2

    @property
    def width(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device("meta")

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        raise AssertionError("inference replica path must not read the remote owner")

    def nearest(self, values, *, allowed_rows=None):
        raise AssertionError("not used")


def test_inference_boundary_rebinds_exact_replica_without_densifying():
    identity = CodebookIdentity("remote.codebook", structure_version=2, parameter_version=3)
    rows = torch.arange(8.0).reshape(2, 4)
    replicas = ReplicaRegistry()
    replica = replicas.install(identity, rows, device="cpu")
    payload = SelectedEvent(_RemoteReader(identity), torch.tensor([[1]]))
    carrier = SubSpace(
        SCHEMA,
        payload,
        PipelineControl(PipelineAddress(1, 0, parameter_version=3)),
    )

    moved, stats = BoundaryAdapter(
        "cpu", training=False, replicas=replicas
    ).adapt(carrier)
    assert isinstance(moved.payload, SelectedEvent)
    assert moved.payload.reader is replica
    assert stats.representation == "replica-sparse"
    assert stats.materialized_bytes == 0
    assert torch.equal(moved.materialize(MaterializeMode.EVENT), rows[torch.tensor([[1]])])

    stale = replace(
        carrier,
        payload=replace(
            payload,
            reader=_RemoteReader(
                CodebookIdentity("remote.codebook", structure_version=2, parameter_version=4)
            ),
        ),
    )
    with pytest.raises(LookupError, match="no exact replica"):
        BoundaryAdapter("cpu", training=False, replicas=replicas).adapt(stale)


def test_training_boundary_materializes_on_owner_and_preserves_gradient():
    basis = _Basis(torch.arange(12.0).reshape(3, 4))
    payload = SelectedEvent(_reader(basis), torch.tensor([[2, 0]]))
    carrier = SubSpace(SCHEMA, payload, _carrier(0).control)

    # cpu:0 is a distinct torch.device identity while remaining available on
    # CPU-only CI, so it exercises the owner-materialization branch.
    moved, stats = BoundaryAdapter("cpu:0", training=True).adapt(carrier)
    assert isinstance(moved.payload, DenseEvent)
    assert stats.representation == "owner-materialized"
    assert stats.materialized_bytes == 2 * 4 * basis.W.element_size()
    assert stats.transferred_bytes == stats.materialized_bytes
    moved.payload.event.sum().backward()
    assert basis.W.grad is not None
    assert torch.count_nonzero(basis.W.grad[1]) == 0


def test_executor_interleaves_stages_and_reassembles_microbatch_order():
    downstream_a = Event()
    upstream_b = Event()

    def upstream(carrier: SubSpace) -> SubSpace:
        if carrier.control.address.microbatch_id == 1:
            assert downstream_a.wait(2.0)
            upstream_b.set()
        return _add(carrier, 1.0)

    def downstream(carrier: SubSpace) -> SubSpace:
        if carrier.control.address.microbatch_id == 0:
            downstream_a.set()
            assert upstream_b.wait(2.0)
        return _add(carrier, 10.0)

    executor = PipelineExecutor(
        [PipelineStage("upstream", upstream), PipelineStage("downstream", downstream)],
        queue_capacity=1,
    )
    run = executor.run([_carrier(0, 0.0), _carrier(1, 1.0)])

    assert [c.control.address.microbatch_id for c in run.outputs] == [0, 1]
    assert torch.all(run.outputs[0].payload.event == 11.0)
    assert torch.all(run.outputs[1].payload.event == 12.0)
    assert upstream_b.is_set()
    assert all(item.calls == 2 for item in run.telemetry)
    assert all(item.peak_input_depth <= 1 for item in run.telemetry)
    assert all(item.activation_bytes > 0 for item in run.telemetry)


def test_executor_rejects_a_stage_that_returns_its_input_carrier():
    executor = PipelineExecutor(
        [PipelineStage("alias", lambda carrier: carrier)]
    )
    with pytest.raises(PipelineExecutionError, match="fresh immutable value"):
        executor.run([_carrier(0)])


def test_executor_sorts_unsorted_inputs_and_accepts_equal_cloned_stream_ids():
    def clone_address(carrier: SubSpace) -> SubSpace:
        address = carrier.control.address
        cloned = replace(address, stream_ids=address.stream_ids.clone())
        return replace(carrier, control=replace(carrier.control, address=cloned))

    executor = PipelineExecutor([PipelineStage("clone", clone_address)])
    stream = torch.tensor([4, 8])
    run = executor.run(
        [
            _carrier(2, 2.0, stream_ids=stream, sequence_step=0),
            _carrier(0, 0.0, stream_ids=stream, sequence_step=0),
            _carrier(1, 1.0, stream_ids=stream, sequence_step=0),
        ]
    )
    assert [c.control.address.microbatch_id for c in run.outputs] == [0, 1, 2]


def test_executor_enforces_parameter_and_stream_versions():
    executor = PipelineExecutor(
        [PipelineStage("identity", lambda carrier: replace(carrier))],
        parameter_version=2,
    )
    with pytest.raises(RuntimeError, match="carrier parameter_version"):
        executor.run([_carrier(0, version=1)])

    with pytest.raises(ValueError, match="sequence regressed"):
        PipelineExecutor(
            [PipelineStage("identity", lambda carrier: replace(carrier))]
        ).run(
            [
                _carrier(0, stream_ids=torch.tensor([9]), sequence_step=2),
                _carrier(1, stream_ids=torch.tensor([9]), sequence_step=1),
            ]
        )


def test_parameter_version_cannot_advance_before_pipeline_drain():
    entered = Event()
    release = Event()
    result: list[object] = []

    def blocked(carrier: SubSpace) -> SubSpace:
        entered.set()
        assert release.wait(2.0)
        return replace(carrier)

    executor = PipelineExecutor([PipelineStage("blocked", blocked)])

    def run_pipeline() -> None:
        result.append(executor.run([_carrier(0)]))

    thread = Thread(target=run_pipeline)
    thread.start()
    assert entered.wait(2.0)
    with pytest.raises(RuntimeError, match="in flight"):
        executor.advance_parameter_version()
    release.set()
    thread.join(2.0)
    assert not thread.is_alive()
    assert len(result) == 1
    assert executor.advance_parameter_version() == 1


@dataclass(frozen=True, slots=True)
class _AddTrace(TraceEntry):
    stage: str
    amount: float

    @property
    def stage_name(self) -> str:
        return self.stage


def test_reverse_uses_each_carriers_own_lifo_trace():
    def make_stage(name: str, amount: float) -> PipelineStage:
        def forward(carrier: SubSpace) -> SubSpace:
            traced = carrier.trace.pushed(
                _AddTrace(stage=name, amount=amount)
            )
            return _add(carrier.with_trace(traced), amount)

        def reverse(carrier: SubSpace) -> SubSpace:
            entry, remaining = carrier.trace.pop_for(name)
            assert isinstance(entry, _AddTrace)
            return _add(carrier.with_trace(remaining), -entry.amount)

        return PipelineStage(name, forward, reverse=reverse)

    executor = PipelineExecutor([make_stage("one", 1.0), make_stage("two", 2.0)])
    forward = executor.run([_carrier(0, 4.0), _carrier(1, 7.0)])
    backward = executor.run_reverse(forward.outputs)
    assert all(carrier.trace == ReverseTrace() for carrier in backward.outputs)
    assert torch.all(backward.outputs[0].payload.event == 4.0)
    assert torch.all(backward.outputs[1].payload.event == 7.0)


@dataclass(frozen=True, slots=True)
class _AppendMutation(DeferredMutation):
    path: str
    value: int

    @property
    def owner_path(self) -> str:
        return self.path

    def commit(self, owner) -> None:
        owner.append(self.value)


@dataclass(frozen=True, slots=True)
class _RejectMutation(DeferredMutation):
    path: str
    value: int

    @property
    def owner_path(self) -> str:
        return self.path

    def validate(self, owner) -> None:
        raise ValueError("invalid mutation")

    def commit(self, owner) -> None:
        owner.append(self.value)


def _with_mutation(carrier: SubSpace, path: str, value: int) -> SubSpace:
    return carrier.with_effects(
        PipelineEffects(deferred_mutations=(_AppendMutation(path, value),))
    )


def test_deferred_mutations_commit_in_address_order_after_drain():
    executor = PipelineExecutor(
        [PipelineStage("identity", lambda carrier: replace(carrier))]
    )
    outputs = (
        _with_mutation(_carrier(2), "owner", 2),
        _with_mutation(_carrier(0), "owner", 0),
        _with_mutation(_carrier(1), "owner", 1),
    )
    owner: list[int] = []
    executor.commit_mutations(outputs, {"owner": owner})
    assert owner == [0, 1, 2]


def test_missing_mutation_owner_is_detected_before_any_commit():
    executor = PipelineExecutor(
        [PipelineStage("identity", lambda carrier: replace(carrier))]
    )
    outputs = (
        _with_mutation(_carrier(0), "present", 1),
        _with_mutation(_carrier(1), "missing", 2),
    )
    owner: list[int] = []
    with pytest.raises(KeyError, match="missing mutation owner"):
        executor.commit_mutations(outputs, {"present": owner})
    assert owner == []


def test_mutations_validate_as_a_batch_before_any_commit():
    executor = PipelineExecutor(
        [PipelineStage("identity", lambda carrier: replace(carrier))]
    )
    first = _with_mutation(_carrier(0), "owner", 1)
    rejected = _carrier(1).with_effects(
        PipelineEffects(
            deferred_mutations=(_RejectMutation("owner", 2),)
        )
    )
    owner: list[int] = []
    with pytest.raises(ValueError, match="invalid mutation"):
        executor.commit_mutations((first, rejected), {"owner": owner})
    assert owner == []


def test_failure_cancels_execution_and_does_not_commit_mutations():
    owner: list[int] = []

    def produce(carrier: SubSpace) -> SubSpace:
        return _with_mutation(carrier, "owner", carrier.control.address.microbatch_id)

    def fail_second(carrier: SubSpace) -> SubSpace:
        if carrier.control.address.microbatch_id == 1:
            raise ValueError("boom")
        return replace(carrier)

    executor = PipelineExecutor(
        [PipelineStage("produce", produce), PipelineStage("fail", fail_second)]
    )
    with pytest.raises(PipelineExecutionError, match="boom"):
        executor.run([_carrier(0), _carrier(1)])
    assert owner == []
    assert not executor.active


def test_cooperative_cancellation_waits_for_stage_release_and_returns_no_run():
    token = CancellationToken()
    entered = Event()
    caught: list[BaseException] = []

    def cancellable(carrier: SubSpace) -> SubSpace:
        entered.set()
        assert token.wait(2.0)  # stage-specific cooperative check
        return replace(carrier)

    executor = PipelineExecutor([PipelineStage("cancellable", cancellable)])

    def run_pipeline() -> None:
        try:
            executor.run([_carrier(0), _carrier(1)], cancel=token)
        except BaseException as exc:  # captured for assertion in the test thread
            caught.append(exc)

    thread = Thread(target=run_pipeline)
    thread.start()
    assert entered.wait(2.0)
    token.cancel()
    thread.join(2.0)
    assert not thread.is_alive()
    assert len(caught) == 1
    assert isinstance(caught[0], PipelineExecutionCancelled)
    assert not executor.active


def test_training_step_aggregates_losses_then_advances_one_version():
    weight = nn.Parameter(torch.tensor(1.0))

    def trainable(carrier: SubSpace) -> SubSpace:
        event = carrier.materialize(MaterializeMode.EVENT)
        assert event is not None
        prediction = event * weight
        loss = prediction.square().mean()
        return carrier.with_payload(DenseEvent(prediction)).with_effects(
            PipelineEffects(losses=(LossTerm("square", loss),))
        )

    executor = PipelineExecutor(
        [PipelineStage("trainable", trainable)], training=True
    )
    optimizer = torch.optim.SGD([weight], lr=0.1)
    run, loss = executor.train_step(
        [_carrier(0, 1.0), _carrier(1, 2.0)], optimizer
    )

    assert len(run.outputs) == 2
    assert loss.item() == pytest.approx(2.5)
    assert weight.item() == pytest.approx(0.5)
    assert executor.parameter_version == 1
    with pytest.raises(RuntimeError, match="carrier parameter_version"):
        executor.run([_carrier(0, version=0)])
