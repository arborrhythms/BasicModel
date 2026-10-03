"""Sparse, immutable carriers and a bounded Space pipeline executor.

This module is the executable contract for
``doc/plans/2026-07-16-sparse-subspace-carrier-design.md``.  It deliberately
does not import :mod:`Spaces`: the value layer must stay below the legacy
Space/SubSpace implementation so Spaces can migrate one at a time.

The important split is:

* model/Space objects own Parameters, codebooks, encoders and recurrent state;
* :class:`SubSpace` is a shallow-immutable value for one pipeline execution;
* :class:`CodebookReader` exposes lookup capabilities, never raw storage;
* :class:`PipelineExecutor` owns queues, overlap, ordering and commit barriers.

``torch.Tensor`` is mutable even inside a frozen dataclass.  Callers must treat
carrier tensors as read-only.  Stage functions return a new carrier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Callable, Iterable, Protocol, Sequence, TypeAlias, runtime_checkable
import weakref

import torch
from torch import Tensor


class SelectionSlot(str, Enum):
    """Which event band is supplied by a selected codebook row."""

    EVENT = "event"
    WHAT = "what"


class MaterializeMode(str, Enum):
    EVENT = "event"
    WHAT = "what"
    WHERE = "where"
    WHEN = "when"
    ACTIVATION = "activation"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class CodebookIdentity:
    """Stable owner and versions for a read-only codebook snapshot."""

    owner_path: str
    structure_version: int = 0
    parameter_version: int = 0

    def __post_init__(self) -> None:
        if not self.owner_path:
            raise ValueError("CodebookIdentity.owner_path must not be empty")
        if self.structure_version < 0 or self.parameter_version < 0:
            raise ValueError("codebook versions must be non-negative")


@runtime_checkable
class CodebookReader(Protocol):
    """Capability-only read surface over an authoritative codebook.

    ``nearest`` returns ``(row_indices, selected_rows)``.  The selected rows
    are gathered through the authoritative lookup path, so trainable readers
    preserve autograd connectivity.
    """

    @property
    def identity(self) -> CodebookIdentity: ...

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


@runtime_checkable
class ParameterVersionOwner(Protocol):
    """Space-side endpoint for the executor's global write epoch."""

    @property
    def codebook_parameter_version(self) -> int: ...

    def set_codebook_parameter_version(self, version: int) -> int: ...


class CodebookReaderView:
    """A narrow reader backed by private callables.

    The view intentionally has no ``W``, ``prototype``, ``getW``, ``setW`` or
    owner property.  Name-mangled closures are an API boundary rather than a
    security boundary; the contract prevents accidental mutation by ordinary
    pipeline code.
    """

    __slots__ = (
        "__identity_fn",
        "__lookup_fn",
        "__nearest_fn",
        "__size_fn",
        "__width_fn",
        "__device_fn",
    )

    def __init__(
        self,
        *,
        identity: Callable[[], CodebookIdentity],
        lookup: Callable[[Tensor], Tensor],
        nearest: Callable[[Tensor, Tensor | None], tuple[Tensor, Tensor]],
        size: Callable[[], int],
        width: Callable[[], int],
        device: Callable[[], torch.device],
    ) -> None:
        object.__setattr__(self, "_CodebookReaderView__identity_fn", identity)
        object.__setattr__(self, "_CodebookReaderView__lookup_fn", lookup)
        object.__setattr__(self, "_CodebookReaderView__nearest_fn", nearest)
        object.__setattr__(self, "_CodebookReaderView__size_fn", size)
        object.__setattr__(self, "_CodebookReaderView__width_fn", width)
        object.__setattr__(self, "_CodebookReaderView__device_fn", device)

    @property
    def identity(self) -> CodebookIdentity:
        return self.__identity_fn()

    @property
    def size(self) -> int:
        return int(self.__size_fn())

    @property
    def width(self) -> int:
        return int(self.__width_fn())

    @property
    def device(self) -> torch.device:
        return torch.device(self.__device_fn())

    def lookup(self, indices: Tensor) -> Tensor:
        if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"codebook indices must be integral, got {indices.dtype}")
        return self.__lookup_fn(indices.long())

    def nearest(
        self,
        values: Tensor,
        *,
        allowed_rows: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if allowed_rows is not None:
            if allowed_rows.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            ):
                raise TypeError("allowed_rows must be an integral tensor")
            allowed_rows = allowed_rows.long()
        return self.__nearest_fn(values, allowed_rows)


def _basis_rows(basis: Any) -> Tensor:
    rows = basis.prototype() if hasattr(basis, "prototype") else None
    if rows is None and hasattr(basis, "getW"):
        rows = basis.getW()
    if not torch.is_tensor(rows) or rows.ndim != 2:
        raise RuntimeError("codebook reader requires a live [V, D] prototype")
    return rows


def make_codebook_reader(
    basis: Any,
    *,
    owner_path: str,
    identity: Callable[[], CodebookIdentity] | None = None,
    use_dot_product: bool | None = None,
) -> CodebookReaderView:
    """Create a read-only capability over an existing Basis/Codebook.

    The returned capability holds only a weak reference.  It neither reparents
    the Basis nor registers it as a child module.  ``basis.lookup`` is used for
    selected rows so gradients continue to the owner's Parameter.
    """

    basis_ref = weakref.ref(basis)

    def live() -> Any:
        obj = basis_ref()
        if obj is None:
            raise RuntimeError(f"codebook owner {owner_path!r} no longer exists")
        return obj

    if identity is None:
        identity = lambda: CodebookIdentity(owner_path=owner_path)

    def lookup(indices: Tensor) -> Tensor:
        obj = live()
        if hasattr(obj, "lookup"):
            return obj.lookup(indices)
        return _basis_rows(obj)[indices]

    def nearest(values: Tensor, allowed_rows: Tensor | None) -> tuple[Tensor, Tensor]:
        obj = live()
        rows = _basis_rows(obj)
        if values.shape[-1] != rows.shape[-1]:
            raise ValueError(
                f"nearest width mismatch: values={values.shape[-1]}, rows={rows.shape[-1]}"
            )
        if allowed_rows is None:
            candidate_ids = torch.arange(rows.shape[0], device=rows.device)
            candidates = rows
        else:
            candidate_ids = allowed_rows.to(device=rows.device, dtype=torch.long)
            if candidate_ids.numel() == 0:
                raise ValueError("allowed_rows must contain at least one row")
            if bool((candidate_ids < 0).any()) or bool((candidate_ids >= rows.shape[0]).any()):
                raise IndexError("allowed_rows contains an out-of-range codebook row")
            # index_select returns a new tensor rather than exposing prototype storage.
            candidates = rows.index_select(0, candidate_ids)

        query = values.to(candidates.device)
        flat = query.reshape(-1, query.shape[-1])
        dot_metric = (
            bool(getattr(obj, "use_dot_product", False))
            if use_dot_product is None
            else bool(use_dot_product)
        )
        if dot_metric:
            scores = flat @ candidates.transpose(0, 1)
            local = scores.argmax(dim=-1)
        else:
            # Negative squared distance avoids materializing a [*, V, D]
            # difference tensor while preserving exact argmin semantics.
            q2 = (flat * flat).sum(dim=-1, keepdim=True)
            c2 = (candidates * candidates).sum(dim=-1).unsqueeze(0)
            scores = -(q2 + c2 - 2.0 * flat @ candidates.transpose(0, 1))
            local = scores.argmax(dim=-1)
        selected_ids = candidate_ids.index_select(0, local).reshape(query.shape[:-1])
        return selected_ids, lookup(selected_ids)

    return CodebookReaderView(
        identity=identity,
        lookup=lookup,
        nearest=nearest,
        size=lambda: int(_basis_rows(live()).shape[0]),
        width=lambda: int(_basis_rows(live()).shape[1]),
        device=lambda: _basis_rows(live()).device,
    )


class ReplicaCodebookReader:
    """Read-only, non-Parameter target-device replica used for inference."""

    __slots__ = ("__identity", "__rows", "__dot")

    def __init__(
        self,
        identity: CodebookIdentity,
        rows: Tensor,
        *,
        use_dot_product: bool = False,
    ) -> None:
        if rows.ndim != 2:
            raise ValueError("replica rows must be [V, D]")
        object.__setattr__(self, "_ReplicaCodebookReader__identity", identity)
        object.__setattr__(self, "_ReplicaCodebookReader__rows", rows.detach().clone())
        object.__setattr__(self, "_ReplicaCodebookReader__dot", bool(use_dot_product))

    @property
    def identity(self) -> CodebookIdentity:
        return self.__identity

    @property
    def size(self) -> int:
        return int(self.__rows.shape[0])

    @property
    def width(self) -> int:
        return int(self.__rows.shape[1])

    @property
    def device(self) -> torch.device:
        return self.__rows.device

    def lookup(self, indices: Tensor) -> Tensor:
        return self.__rows[indices.to(device=self.device, dtype=torch.long)]

    def nearest(
        self,
        values: Tensor,
        *,
        allowed_rows: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if allowed_rows is None:
            ids = torch.arange(self.size, device=self.device)
        else:
            if allowed_rows.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("allowed_rows must be an integral tensor")
            ids = allowed_rows.to(device=self.device, dtype=torch.long)
            if ids.numel() == 0:
                raise ValueError("allowed_rows must contain at least one row")
            if bool((ids < 0).any()) or bool((ids >= self.size).any()):
                raise IndexError("allowed_rows contains an out-of-range codebook row")
        candidates = self.__rows.index_select(0, ids)
        flat = values.to(self.device).reshape(-1, self.width)
        if self.__dot:
            local = (flat @ candidates.T).argmax(dim=-1)
        else:
            q2 = (flat * flat).sum(-1, keepdim=True)
            c2 = (candidates * candidates).sum(-1).unsqueeze(0)
            local = (-(q2 + c2 - 2.0 * flat @ candidates.T)).argmax(dim=-1)
        selected_ids = ids.index_select(0, local).reshape(values.shape[:-1])
        return selected_ids, self.lookup(selected_ids)


class ReplicaRegistry:
    """Version-exact inference replicas keyed by identity and device."""

    def __init__(self) -> None:
        self._readers: dict[tuple[CodebookIdentity, str], ReplicaCodebookReader] = {}
        self._lock = Lock()

    @staticmethod
    def _device_key(device: torch.device | str) -> str:
        return str(torch.device(device))

    def install(
        self,
        identity: CodebookIdentity,
        rows: Tensor,
        *,
        device: torch.device | str,
        use_dot_product: bool = False,
    ) -> ReplicaCodebookReader:
        target = torch.device(device)
        reader = ReplicaCodebookReader(
            identity,
            rows.to(target),
            use_dot_product=use_dot_product,
        )
        with self._lock:
            self._readers[(identity, self._device_key(target))] = reader
        return reader

    def resolve(
        self,
        identity: CodebookIdentity,
        device: torch.device | str,
    ) -> ReplicaCodebookReader:
        key = (identity, self._device_key(device))
        with self._lock:
            reader = self._readers.get(key)
        if reader is None:
            raise LookupError(
                f"no exact replica for {identity.owner_path!r} "
                f"structure={identity.structure_version} "
                f"parameters={identity.parameter_version} on {key[1]}"
            )
        return reader

    def invalidate_owner(self, owner_path: str) -> None:
        with self._lock:
            prefix = owner_path.rstrip('.') + '.'
            stale = [
                key
                for key in self._readers
                if key[0].owner_path == owner_path
                or key[0].owner_path.startswith(prefix)
            ]
            for key in stale:
                del self._readers[key]

    def invalidate_all(self) -> None:
        """Drop every derived replica after a global Parameter update."""
        with self._lock:
            self._readers.clear()


@dataclass(frozen=True, slots=True)
class SubSpaceSchema:
    role: str
    n_what: int
    n_where: int
    n_when: int
    geometry: str = "euclidean"

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("SubSpaceSchema.role must not be empty")
        if min(self.n_what, self.n_where, self.n_when) < 0:
            raise ValueError("SubSpace dimensions must be non-negative")

    @property
    def event_width(self) -> int:
        return self.n_what + self.n_where + self.n_when


@dataclass(frozen=True, slots=True)
class DenseEvent:
    """Intrinsic dense stage payload plus an optional per-position gate."""

    event: Tensor
    activation: Tensor | None = None


@dataclass(frozen=True, slots=True)
class FactoredEvent:
    """Separately addressable dense bands; absent bands use ``None``."""

    what: Tensor | None = None
    where: Tensor | None = None
    when: Tensor | None = None
    activation: Tensor | None = None


@dataclass(frozen=True, slots=True)
class SelectedEvent:
    """Sparse codebook selection and only the bands outside that codebook."""

    reader: CodebookReader
    indices: Tensor
    slot: SelectionSlot = SelectionSlot.EVENT
    activation: Tensor | None = None
    where: Tensor | None = None
    when: Tensor | None = None

    def __post_init__(self) -> None:
        if self.indices.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise TypeError("SelectedEvent.indices must be integral")


Payload: TypeAlias = DenseEvent | FactoredEvent | SelectedEvent


@dataclass(frozen=True, slots=True)
class PipelineAddress:
    """Executor-produced identity preserved unchanged by every stage."""

    execution_id: int
    microbatch_id: int
    parameter_version: int
    stream_ids: Tensor | None = None
    sequence_step: int = 0

    def __post_init__(self) -> None:
        if self.microbatch_id < 0:
            raise ValueError("microbatch_id must be non-negative")
        if self.parameter_version < 0:
            raise ValueError("parameter_version must be non-negative")
        if self.sequence_step < 0:
            raise ValueError("sequence_step must be non-negative")
        if self.stream_ids is not None and self.stream_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("stream_ids must be an integral tensor")


@dataclass(frozen=True, slots=True)
class PipelineControl:
    """Address plus executor/input-produced masks consumed during this run."""

    address: PipelineAddress
    valid_mask: Tensor | None = None
    reset_mask: Tensor | None = None
    row_gate: Tensor | None = None
    recurrent_pass: int = 0

    def __post_init__(self) -> None:
        if self.recurrent_pass < 0:
            raise ValueError("recurrent_pass must be non-negative")
        if self.valid_mask is not None and self.valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be boolean")
        if self.reset_mask is not None and self.reset_mask.dtype is not torch.bool:
            raise TypeError("reset_mask must be boolean")


@dataclass(frozen=True, slots=True)
class LossTerm:
    name: str
    value: Tensor
    weight: float = 1.0
    category: str = "model"


class Diagnostic(ABC):
    """Typed diagnostic record base; subclasses declare named fields."""

    @property
    @abstractmethod
    def diagnostic_name(self) -> str: ...

    def tensor_values(self) -> tuple[Tensor, ...]:
        return ()

    def to_device(
        self,
        device: torch.device,
        *,
        non_blocking: bool,
    ) -> "Diagnostic":
        return self


class DeferredMutation(ABC):
    """Closed-union base for owner-specific, drain-barrier mutations.

    Concrete variants must be frozen/slotted dataclasses. They carry typed
    scalar/tensor fields, validate without mutation in ``validate``, and
    perform their declared owner write in ``commit``.
    """

    @property
    @abstractmethod
    def owner_path(self) -> str: ...

    def validate(self, owner: Any) -> None:
        """Validate this mutation before the batch commit begins.

        Concrete variants override this when their commit has preconditions.
        The executor validates the whole batch first so an ordinary validation
        error cannot leave an earlier mutation committed.
        """

    @abstractmethod
    def commit(self, owner: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class PipelineEffects:
    """Sparse stage emissions accumulated until the training/run boundary."""

    losses: tuple[LossTerm, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    deferred_mutations: tuple[DeferredMutation, ...] = ()

    def merged(self, other: "PipelineEffects") -> "PipelineEffects":
        return PipelineEffects(
            losses=self.losses + other.losses,
            diagnostics=self.diagnostics + other.diagnostics,
            deferred_mutations=self.deferred_mutations + other.deferred_mutations,
        )


class TraceEntry(ABC):
    """Typed reverse record base; subclasses declare sufficient statistics."""

    @property
    @abstractmethod
    def stage_name(self) -> str: ...

    def tensor_values(self) -> tuple[Tensor, ...]:
        return ()

    def to_device(
        self,
        device: torch.device,
        *,
        non_blocking: bool,
    ) -> "TraceEntry":
        return self


@dataclass(frozen=True, slots=True)
class ReverseTrace:
    """Per-microbatch LIFO sufficient statistics consumed by reverse stages."""

    entries: tuple[TraceEntry, ...] = ()

    def pushed(self, entry: TraceEntry) -> "ReverseTrace":
        return ReverseTrace(self.entries + (entry,))

    def pop_for(self, stage: str) -> tuple[TraceEntry, "ReverseTrace"]:
        if not self.entries:
            raise LookupError(f"reverse trace is empty; expected stage {stage!r}")
        entry = self.entries[-1]
        if entry.stage_name != stage:
            raise LookupError(
                f"reverse trace top belongs to {entry.stage_name!r}; "
                f"expected {stage!r}"
            )
        return entry, ReverseTrace(self.entries[:-1])


def _parts_for_factored(payload: FactoredEvent) -> list[Tensor]:
    return [p for p in (payload.what, payload.where, payload.when) if p is not None]


def _apply_activation(value: Tensor | None, activation: Tensor | None) -> Tensor | None:
    if value is None or activation is None:
        return value
    gate = activation
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    return value * gate


@dataclass(frozen=True, slots=True)
class SubSpace:
    """One immutable value at one point in one pipeline execution.

    The owning Space shares ``schema``; the current stage produces ``payload``;
    the executor/input owns ``control``; stages append sparse ``effects`` and
    ``trace`` records. The carrier dies after forward, cognitive reverse, and
    autograd consumers release that microbatch.
    """

    schema: SubSpaceSchema
    payload: Payload
    control: PipelineControl
    effects: PipelineEffects = field(default_factory=PipelineEffects)
    trace: ReverseTrace = field(default_factory=ReverseTrace)

    def with_payload(self, payload: Payload, *, schema: SubSpaceSchema | None = None) -> "SubSpace":
        return replace(self, payload=payload, schema=self.schema if schema is None else schema)

    def with_effects(self, effects: PipelineEffects) -> "SubSpace":
        return replace(self, effects=self.effects.merged(effects))

    def with_trace(self, trace: ReverseTrace) -> "SubSpace":
        return replace(self, trace=trace)

    def is_empty(self) -> bool:
        """Report structural emptiness without gathering selected rows."""

        payload = self.payload
        if isinstance(payload, SelectedEvent):
            return payload.indices.numel() == 0
        if isinstance(payload, DenseEvent):
            return payload.event.numel() == 0
        parts = _parts_for_factored(payload)
        return not parts or all(part.numel() == 0 for part in parts)

    def _selected_parts(self, payload: SelectedEvent) -> tuple[Tensor, Tensor | None, Tensor | None]:
        selected = payload.reader.lookup(payload.indices)
        if payload.slot is SelectionSlot.EVENT:
            return selected, None, None
        return selected, payload.where, payload.when

    def _event_and_activation(self) -> tuple[Tensor | None, Tensor | None]:
        payload = self.payload
        if isinstance(payload, DenseEvent):
            return payload.event, payload.activation
        if isinstance(payload, FactoredEvent):
            parts = _parts_for_factored(payload)
            if not parts:
                return None, payload.activation
            lead = parts[0].shape[:-1]
            if any(part.shape[:-1] != lead for part in parts[1:]):
                raise ValueError("factored event bands have incompatible leading shapes")
            return torch.cat(parts, dim=-1), payload.activation
        selected, where, when = self._selected_parts(payload)
        if payload.slot is SelectionSlot.EVENT:
            return selected, payload.activation
        parts = [selected] + [p for p in (where, when) if p is not None]
        return torch.cat(parts, dim=-1), payload.activation

    def materialize(self, mode: MaterializeMode | str = MaterializeMode.ACTIVE) -> Tensor | None:
        """Pure, uncached materialization of one named representation."""

        try:
            mode = MaterializeMode(mode)
        except ValueError as exc:
            raise ValueError(f"unsupported materialization mode {mode!r}") from exc

        payload = self.payload
        event, activation = self._event_and_activation()
        if mode is MaterializeMode.ACTIVATION:
            return activation
        if mode is MaterializeMode.EVENT:
            return event
        if mode is MaterializeMode.ACTIVE:
            return _apply_activation(event, activation)

        if isinstance(payload, FactoredEvent):
            return {
                MaterializeMode.WHAT: payload.what,
                MaterializeMode.WHERE: payload.where,
                MaterializeMode.WHEN: payload.when,
            }[mode]
        if isinstance(payload, SelectedEvent) and payload.slot is SelectionSlot.WHAT:
            if mode is MaterializeMode.WHAT:
                return payload.reader.lookup(payload.indices)
            if mode is MaterializeMode.WHERE:
                return payload.where
            if mode is MaterializeMode.WHEN:
                return payload.when
        if event is None:
            return None
        w0 = self.schema.n_what
        w1 = w0 + self.schema.n_where
        if mode is MaterializeMode.WHAT:
            return event[..., :w0]
        if mode is MaterializeMode.WHERE:
            return event[..., w0:w1]
        if mode is MaterializeMode.WHEN:
            return event[..., w1:w1 + self.schema.n_when]
        raise AssertionError(f"unhandled materialization mode {mode}")


def _move_tensor(value: Tensor | None, device: torch.device, non_blocking: bool) -> Tensor | None:
    return None if value is None else value.to(device=device, non_blocking=non_blocking)


def _move_address(address: PipelineAddress, device: torch.device, non_blocking: bool) -> PipelineAddress:
    return replace(
        address,
        stream_ids=_move_tensor(address.stream_ids, device, non_blocking),
    )


def _move_control(control: PipelineControl, device: torch.device, non_blocking: bool) -> PipelineControl:
    return replace(
        control,
        address=_move_address(control.address, device, non_blocking),
        valid_mask=_move_tensor(control.valid_mask, device, non_blocking),
        reset_mask=_move_tensor(control.reset_mask, device, non_blocking),
        row_gate=_move_tensor(control.row_gate, device, non_blocking),
    )


def _move_trace(trace: ReverseTrace, device: torch.device, non_blocking: bool) -> ReverseTrace:
    entries = tuple(
        entry.to_device(device, non_blocking=non_blocking)
        for entry in trace.entries
    )
    return ReverseTrace(entries)


def _move_effects(effects: PipelineEffects, device: torch.device, non_blocking: bool) -> PipelineEffects:
    losses = tuple(
        replace(term, value=term.value.to(device=device, non_blocking=non_blocking))
        for term in effects.losses
    )
    diagnostics = tuple(
        diagnostic.to_device(device, non_blocking=non_blocking)
        for diagnostic in effects.diagnostics
    )
    return replace(effects, losses=losses, diagnostics=diagnostics)


@dataclass(frozen=True, slots=True)
class BoundaryStats:
    elapsed_s: float = 0.0
    transferred_bytes: int = 0
    materialized_bytes: int = 0
    representation: str = "same-device"


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def _payload_bytes(payload: Payload) -> int:
    if isinstance(payload, DenseEvent):
        return _tensor_bytes(payload.event) + _tensor_bytes(payload.activation)
    if isinstance(payload, FactoredEvent):
        return sum(_tensor_bytes(v) for v in (payload.what, payload.where, payload.when, payload.activation))
    return sum(_tensor_bytes(v) for v in (payload.indices, payload.activation, payload.where, payload.when))


def _transfer_bytes(value: Tensor | None, target: torch.device) -> int:
    if value is None or value.device == target:
        return 0
    return _tensor_bytes(value)


def _payload_transfer_bytes(payload: Payload, target: torch.device) -> int:
    if isinstance(payload, DenseEvent):
        return _transfer_bytes(payload.event, target) + _transfer_bytes(
            payload.activation, target
        )
    if isinstance(payload, FactoredEvent):
        return sum(
            _transfer_bytes(value, target)
            for value in (payload.what, payload.where, payload.when, payload.activation)
        )
    return sum(
        _transfer_bytes(value, target)
        for value in (payload.indices, payload.activation, payload.where, payload.when)
    )


def _auxiliary_transfer_bytes(carrier: SubSpace, target: torch.device) -> int:
    """Count non-payload tensors the boundary adapter moves to a stage."""

    control = carrier.control
    address = control.address
    total = sum(
        _transfer_bytes(value, target)
        for value in (
            address.stream_ids,
            control.valid_mask,
            control.reset_mask,
            control.row_gate,
        )
    )
    total += sum(
        _transfer_bytes(term.value, target) for term in carrier.effects.losses
    )
    total += sum(
        _transfer_bytes(tensor, target)
        for diagnostic in carrier.effects.diagnostics
        for tensor in diagnostic.tensor_values()
    )
    total += sum(
        _transfer_bytes(tensor, target)
        for entry in carrier.trace.entries
        for tensor in entry.tensor_values()
    )
    return total


class BoundaryAdapter:
    """Create the one canonical carrier representation for a stage boundary."""

    def __init__(
        self,
        target_device: torch.device | str,
        *,
        training: bool,
        replicas: ReplicaRegistry | None = None,
        non_blocking: bool = True,
    ) -> None:
        self.target_device = torch.device(target_device)
        self.training = bool(training)
        self.replicas = replicas
        self.non_blocking = bool(non_blocking)

    def adapt(self, carrier: SubSpace) -> tuple[SubSpace, BoundaryStats]:
        started = perf_counter()
        payload = carrier.payload
        materialized_bytes = 0
        transferred_bytes = _auxiliary_transfer_bytes(
            carrier, self.target_device
        )
        representation = "dense-transfer"

        if isinstance(payload, SelectedEvent):
            same_device = payload.reader.device == self.target_device
            if same_device:
                transferred_bytes += _payload_transfer_bytes(
                    payload, self.target_device
                )
                moved_payload: Payload = replace(
                    payload,
                    indices=_move_tensor(
                        payload.indices, self.target_device, self.non_blocking
                    ),
                    activation=_move_tensor(
                        payload.activation, self.target_device, self.non_blocking
                    ),
                    where=_move_tensor(
                        payload.where, self.target_device, self.non_blocking
                    ),
                    when=_move_tensor(
                        payload.when, self.target_device, self.non_blocking
                    ),
                )
                representation = "same-device-sparse"
            elif not self.training and self.replicas is not None:
                transferred_bytes += _payload_transfer_bytes(
                    payload, self.target_device
                )
                reader = self.replicas.resolve(payload.reader.identity, self.target_device)
                moved_payload = replace(
                    payload,
                    reader=reader,
                    indices=_move_tensor(payload.indices, self.target_device, self.non_blocking),
                    activation=_move_tensor(payload.activation, self.target_device, self.non_blocking),
                    where=_move_tensor(payload.where, self.target_device, self.non_blocking),
                    when=_move_tensor(payload.when, self.target_device, self.non_blocking),
                )
                representation = "replica-sparse"
            else:
                transferred_bytes += _transfer_bytes(
                    payload.indices, payload.reader.device
                )
                transferred_bytes += _transfer_bytes(
                    payload.activation, self.target_device
                )
                if payload.slot is SelectionSlot.WHAT:
                    transferred_bytes += _transfer_bytes(
                        payload.where, self.target_device
                    )
                    transferred_bytes += _transfer_bytes(
                        payload.when, self.target_device
                    )
                owner_indices = payload.indices.to(
                    device=payload.reader.device,
                    dtype=torch.long,
                    non_blocking=self.non_blocking,
                )
                selected = payload.reader.lookup(owner_indices)
                materialized_bytes = _tensor_bytes(selected)
                transferred_bytes += _transfer_bytes(selected, self.target_device)
                selected = selected.to(self.target_device, non_blocking=self.non_blocking)
                activation = _move_tensor(payload.activation, self.target_device, self.non_blocking)
                if payload.slot is SelectionSlot.EVENT:
                    moved_payload = DenseEvent(selected, activation)
                else:
                    moved_payload = FactoredEvent(
                        what=selected,
                        where=_move_tensor(payload.where, self.target_device, self.non_blocking),
                        when=_move_tensor(payload.when, self.target_device, self.non_blocking),
                        activation=activation,
                    )
                representation = "owner-materialized"
        elif isinstance(payload, DenseEvent):
            transferred_bytes += _payload_transfer_bytes(
                payload, self.target_device
            )
            moved_payload = DenseEvent(
                payload.event.to(self.target_device, non_blocking=self.non_blocking),
                _move_tensor(payload.activation, self.target_device, self.non_blocking),
            )
        else:
            transferred_bytes += _payload_transfer_bytes(
                payload, self.target_device
            )
            moved_payload = FactoredEvent(
                what=_move_tensor(payload.what, self.target_device, self.non_blocking),
                where=_move_tensor(payload.where, self.target_device, self.non_blocking),
                when=_move_tensor(payload.when, self.target_device, self.non_blocking),
                activation=_move_tensor(payload.activation, self.target_device, self.non_blocking),
            )

        moved = replace(
            carrier,
            payload=moved_payload,
            control=_move_control(carrier.control, self.target_device, self.non_blocking),
            effects=_move_effects(carrier.effects, self.target_device, self.non_blocking),
            trace=_move_trace(carrier.trace, self.target_device, self.non_blocking),
        )
        return moved, BoundaryStats(
            elapsed_s=perf_counter() - started,
            transferred_bytes=transferred_bytes,
            materialized_bytes=materialized_bytes,
            representation=representation,
        )


@dataclass(frozen=True, slots=True)
class PipelineStage:
    name: str
    forward: Callable[[SubSpace], SubSpace]
    device: torch.device | str = "cpu"
    reverse: Callable[[SubSpace], SubSpace] | None = None
    globally_ordered: bool = False
    parameter_owner: ParameterVersionOwner | None = None
    pipeline_safe: bool = True
    training_safe: bool = True
    serialization_reason: str | None = None

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass(frozen=True, slots=True)
class StageTelemetry:
    name: str
    calls: int
    compute_s: float
    queue_wait_s: float
    barrier_wait_s: float
    boundary_s: float
    transferred_bytes: int
    materialized_bytes: int
    activation_bytes: int
    peak_input_depth: int


class _MutableTelemetry:
    __slots__ = (
        "name", "calls", "compute_s", "queue_wait_s", "barrier_wait_s",
        "boundary_s", "transferred_bytes", "materialized_bytes",
        "activation_bytes", "peak_input_depth", "lock",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.compute_s = 0.0
        self.queue_wait_s = 0.0
        self.barrier_wait_s = 0.0
        self.boundary_s = 0.0
        self.transferred_bytes = 0
        self.materialized_bytes = 0
        self.activation_bytes = 0
        self.peak_input_depth = 0
        self.lock = Lock()

    def snapshot(self) -> StageTelemetry:
        with self.lock:
            return StageTelemetry(
                name=self.name,
                calls=self.calls,
                compute_s=self.compute_s,
                queue_wait_s=self.queue_wait_s,
                barrier_wait_s=self.barrier_wait_s,
                boundary_s=self.boundary_s,
                transferred_bytes=self.transferred_bytes,
                materialized_bytes=self.materialized_bytes,
                activation_bytes=self.activation_bytes,
                peak_input_depth=self.peak_input_depth,
            )


@dataclass(frozen=True, slots=True)
class PipelineRun:
    outputs: tuple[SubSpace, ...]
    telemetry: tuple[StageTelemetry, ...]


class PipelineExecutionError(RuntimeError):
    def __init__(self, stage: str, address: PipelineAddress | None, cause: BaseException):
        self.stage = stage
        self.address = address
        self.cause = cause
        where = "unknown address" if address is None else (
            f"execution={address.execution_id}, microbatch={address.microbatch_id}"
        )
        super().__init__(f"pipeline stage {stage!r} failed at {where}: {cause}")


class PipelineExecutionCancelled(RuntimeError):
    """Raised after a cooperative cancellation drains worker ownership."""


class CancellationToken:
    """Thread-safe cooperative cancellation handle for one pipeline run."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation; useful to cooperative stage functions."""
        return self._event.wait(timeout)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class PipelineExecutor:
    """Bounded, deterministic threaded executor with one worker per Space.

    Threads provide genuine overlap when adjacent stages use independent
    resources.  Each stage still processes one carrier at a time, preserving
    stage-local mutation safety during the migration.
    """

    _SENTINEL = object()

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        queue_capacity: int = 2,
        parameter_version: int | None = None,
        training: bool = False,
        replicas: ReplicaRegistry | None = None,
    ) -> None:
        if not stages:
            raise ValueError("PipelineExecutor requires at least one stage")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("pipeline stage names must be unique")
        unsafe_pipeline = [
            (
                stage.name,
                stage.serialization_reason
                or "stage has undeclared durable forward mutations",
            )
            for stage in stages
            if not stage.pipeline_safe
        ]
        if unsafe_pipeline:
            details = "; ".join(
                f"{name}: {reason}" for name, reason in unsafe_pipeline
            )
            raise RuntimeError(
                f"pipeline execution requires isolated stages; {details}"
            )
        if training:
            unsafe = [
                (
                    stage.name,
                    stage.serialization_reason
                    or "stage has undeferred durable forward mutations",
                )
                for stage in stages
                if not stage.training_safe
            ]
            if unsafe:
                details = "; ".join(
                    f"{name}: {reason}" for name, reason in unsafe
                )
                raise RuntimeError(
                    f"pipeline training requires mutation-safe stages; {details}"
                )
        self.stages = tuple(stages)
        self.queue_capacity = int(queue_capacity)
        owners: list[ParameterVersionOwner] = []
        seen_owners: set[int] = set()
        for stage in stages:
            owner = stage.parameter_owner
            if owner is not None and id(owner) not in seen_owners:
                owners.append(owner)
                seen_owners.add(id(owner))
        owner_versions = {
            int(owner.codebook_parameter_version) for owner in owners
        }
        if parameter_version is None:
            if len(owner_versions) > 1:
                raise ValueError(
                    f"pipeline parameter owners disagree: {sorted(owner_versions)}"
                )
            parameter_version = next(iter(owner_versions), 0)
        elif any(version != int(parameter_version) for version in owner_versions):
            raise ValueError(
                f"executor parameter_version={parameter_version} does not match "
                f"owners={sorted(owner_versions)}"
            )
        self.parameter_version = int(parameter_version)
        self._parameter_owners = tuple(owners)
        self.training = bool(training)
        self.replicas = replicas
        self._state_lock = Lock()
        self._active = False

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._active

    def _set_active(self, value: bool) -> None:
        with self._state_lock:
            if value and self._active:
                raise RuntimeError("pipeline execution is already active")
            self._active = value

    def advance_parameter_version(self) -> int:
        with self._state_lock:
            if self._active:
                raise RuntimeError("cannot change Parameters while carriers are in flight")
            next_version = self.parameter_version + 1
            for owner in self._parameter_owners:
                owner.set_codebook_parameter_version(next_version)
            self.parameter_version = next_version
            if self.replicas is not None:
                self.replicas.invalidate_all()
            return next_version

    def _validate_inputs(self, carriers: Sequence[SubSpace]) -> None:
        if not carriers:
            raise ValueError("pipeline execution requires at least one carrier")
        execution_ids = {c.control.address.execution_id for c in carriers}
        if len(execution_ids) != 1:
            raise ValueError("all microbatches in one run must share execution_id")
        ids = [c.control.address.microbatch_id for c in carriers]
        if len(set(ids)) != len(ids):
            raise ValueError("microbatch_id values must be unique within an execution")
        for carrier in carriers:
            got = carrier.control.address.parameter_version
            if got != self.parameter_version:
                raise RuntimeError(
                    f"carrier parameter_version={got}, executor={self.parameter_version}"
                )
            if isinstance(carrier.payload, SelectedEvent):
                identity = carrier.payload.reader.identity
                if identity.parameter_version != got:
                    raise RuntimeError(
                        f"selected reader {identity.owner_path!r} has "
                        f"parameter_version={identity.parameter_version}, carrier={got}"
                    )

        # Input queue order is the commit order for each recurrent stream.
        last: dict[int, int] = {}
        for carrier in carriers:
            address = carrier.control.address
            if address.stream_ids is None:
                continue
            for stream in address.stream_ids.detach().cpu().reshape(-1).tolist():
                stream = int(stream)
                previous = last.get(stream)
                if previous is not None and address.sequence_step < previous:
                    raise ValueError(
                        f"stream {stream} sequence regressed from {previous} "
                        f"to {address.sequence_step}"
                    )
                last[stream] = address.sequence_step

    @staticmethod
    def _validate_stage_output(stage: PipelineStage, incoming: SubSpace, outgoing: SubSpace) -> None:
        if not isinstance(outgoing, SubSpace):
            raise TypeError(
                f"stage {stage.name!r} returned {type(outgoing).__name__}, expected SubSpace"
            )
        if outgoing is incoming:
            raise RuntimeError(
                f"stage {stage.name!r} returned its input carrier; stages must "
                f"return a fresh immutable value"
            )
        # Dataclass equality is invalid for tensor-valued ``stream_ids``:
        # Tensor.__eq__ returns a tensor whose truth value is ambiguous.
        a = incoming.control.address
        b = outgoing.control.address
        scalar_same = (
            a.execution_id == b.execution_id
            and a.microbatch_id == b.microbatch_id
            and a.parameter_version == b.parameter_version
            and a.sequence_step == b.sequence_step
        )
        streams_same = (
            a.stream_ids is b.stream_ids
            or (
                a.stream_ids is not None
                and b.stream_ids is not None
                and torch.equal(a.stream_ids, b.stream_ids)
            )
        )
        if not scalar_same or not streams_same:
            raise RuntimeError(f"stage {stage.name!r} changed PipelineAddress")

    def _ordered_stages(self, reverse: bool) -> tuple[PipelineStage, ...]:
        if not reverse:
            return self.stages
        missing = [stage.name for stage in self.stages if stage.reverse is None]
        if missing:
            raise RuntimeError(f"reverse functions missing for stages: {', '.join(missing)}")
        return tuple(reversed(self.stages))

    def _run(
        self,
        carriers: Sequence[SubSpace],
        *,
        reverse: bool,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        carriers = tuple(carriers)
        self._validate_inputs(carriers)
        ordered = self._ordered_stages(reverse)
        self._set_active(True)
        try:
            queues: list[Queue[Any]] = [
                Queue(maxsize=self.queue_capacity) for _ in ordered
            ]
            final_queue: Queue[Any] = Queue()  # collector must never backpressure workers
            error_queue: Queue[PipelineExecutionError] = Queue()
            stop = Event()
            telemetry = [_MutableTelemetry(stage.name) for stage in ordered]

            def put_bounded(
                queue: Queue[Any],
                item: Any,
                stats: _MutableTelemetry | None = None,
            ) -> bool:
                started = perf_counter()
                while not stop.is_set() and not (cancel is not None and cancel.cancelled):
                    try:
                        queue.put(item, timeout=0.05)
                        if stats is not None:
                            with stats.lock:
                                stats.barrier_wait_s += perf_counter() - started
                        return True
                    except Full:
                        continue
                if stats is not None:
                    with stats.lock:
                        stats.barrier_wait_s += perf_counter() - started
                return False

            def worker(index: int) -> None:
                stage = ordered[index]
                incoming_queue = queues[index]
                outgoing_queue = final_queue if index == len(ordered) - 1 else queues[index + 1]
                stats = telemetry[index]
                call = stage.reverse if reverse else stage.forward
                assert call is not None
                while not stop.is_set():
                    if cancel is not None and cancel.cancelled:
                        stop.set()
                        final_queue.put(self._SENTINEL)
                        return
                    wait_started = perf_counter()
                    try:
                        item = incoming_queue.get(timeout=0.05)
                    except Empty:
                        continue
                    with stats.lock:
                        stats.queue_wait_s += perf_counter() - wait_started
                        stats.peak_input_depth = max(stats.peak_input_depth, incoming_queue.qsize() + 1)
                    if item is self._SENTINEL:
                        if outgoing_queue is final_queue:
                            final_queue.put(self._SENTINEL)
                        else:
                            put_bounded(outgoing_queue, self._SENTINEL)
                        return
                    carrier = item
                    address = carrier.control.address if isinstance(carrier, SubSpace) else None
                    try:
                        compute_started = perf_counter()
                        outgoing = call(carrier)
                        compute_elapsed = perf_counter() - compute_started
                        self._validate_stage_output(stage, carrier, outgoing)
                        with stats.lock:
                            stats.calls += 1
                            stats.compute_s += compute_elapsed
                            stats.activation_bytes += _payload_bytes(outgoing.payload)

                        if index < len(ordered) - 1:
                            target = ordered[index + 1].torch_device
                            adapter = BoundaryAdapter(
                                target,
                                training=self.training,
                                replicas=self.replicas,
                            )
                            outgoing, boundary = adapter.adapt(outgoing)
                            with stats.lock:
                                stats.boundary_s += boundary.elapsed_s
                                stats.transferred_bytes += boundary.transferred_bytes
                                stats.materialized_bytes += boundary.materialized_bytes
                        if outgoing_queue is final_queue:
                            final_queue.put(outgoing)
                        elif not put_bounded(outgoing_queue, outgoing, stats):
                            return
                    except BaseException as exc:
                        error_queue.put(PipelineExecutionError(stage.name, address, exc))
                        stop.set()
                        final_queue.put(self._SENTINEL)
                        return

            workers = [
                Thread(target=worker, args=(i,), name=f"pipeline:{stage.name}", daemon=True)
                for i, stage in enumerate(ordered)
            ]
            for thread in workers:
                thread.start()

            try:
                for carrier in carriers:
                    if cancel is not None and cancel.cancelled:
                        break
                    if not put_bounded(queues[0], carrier):
                        break
                put_bounded(queues[0], self._SENTINEL)

                outputs: list[SubSpace] = []
                while len(outputs) < len(carriers):
                    if cancel is not None and cancel.cancelled:
                        stop.set()
                        raise PipelineExecutionCancelled("pipeline execution was cancelled")
                    if not error_queue.empty():
                        raise error_queue.get_nowait()
                    try:
                        item = final_queue.get(timeout=0.05)
                    except Empty:
                        if all(not thread.is_alive() for thread in workers):
                            break
                        continue
                    if item is self._SENTINEL:
                        break
                    outputs.append(item)

                if not error_queue.empty():
                    raise error_queue.get_nowait()
                if len(outputs) != len(carriers):
                    raise RuntimeError(
                        f"pipeline produced {len(outputs)} of {len(carriers)} outputs"
                    )
            finally:
                stop.set()
                for queue in queues:
                    try:
                        queue.put_nowait(self._SENTINEL)
                    except Full:
                        pass
                for thread in workers:
                    # Cancellation is cooperative: an in-progress stage call
                    # finishes, then every worker relinquishes its carrier
                    # before the execution barrier opens again.  Returning
                    # while a daemon still held a graph would make a later
                    # optimizer step unsafe.
                    thread.join()

            outputs.sort(key=lambda c: c.control.address.microbatch_id)
            return PipelineRun(
                outputs=tuple(outputs),
                telemetry=tuple(item.snapshot() for item in telemetry),
            )
        finally:
            self._set_active(False)

    def run(
        self,
        carriers: Sequence[SubSpace],
        *,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        return self._run(carriers, reverse=False, cancel=cancel)

    def run_reverse(
        self,
        carriers: Sequence[SubSpace],
        *,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        return self._run(carriers, reverse=True, cancel=cancel)

    @staticmethod
    def _ordered_mutations(outputs: Sequence[SubSpace]) -> list[DeferredMutation]:
        ordered = sorted(
            outputs,
            key=lambda c: (
                c.control.address.execution_id,
                c.control.address.sequence_step,
                c.control.address.microbatch_id,
            ),
        )
        return [mutation for carrier in ordered for mutation in carrier.effects.deferred_mutations]

    def commit_mutations(
        self,
        outputs: Sequence[SubSpace],
        owners: dict[str, Any],
    ) -> None:
        if self.active:
            raise RuntimeError("cannot commit mutations while carriers are in flight")
        mutations = self._ordered_mutations(outputs)
        resolved: list[tuple[DeferredMutation, Any]] = []
        for mutation in mutations:
            try:
                owner = owners[mutation.owner_path]
            except KeyError as exc:
                raise KeyError(f"missing mutation owner {mutation.owner_path!r}") from exc
            resolved.append((mutation, owner))
        for mutation, owner in resolved:
            mutation.validate(owner)
        for mutation, owner in resolved:
            mutation.commit(owner)
            if self.replicas is not None:
                self.replicas.invalidate_owner(mutation.owner_path)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> int:
        if self.active:
            raise RuntimeError("optimizer.step() is illegal while carriers are in flight")
        optimizer.step()
        return self.advance_parameter_version()

    def train_step(
        self,
        carriers: Sequence[SubSpace],
        optimizer: torch.optim.Optimizer,
        *,
        mutation_owners: dict[str, Any] | None = None,
        loss_reduction: str = "mean",
    ) -> tuple[PipelineRun, Tensor]:
        if not self.training:
            raise RuntimeError("train_step requires PipelineExecutor(training=True)")
        optimizer.zero_grad()
        run = self.run(carriers)
        weighted = [
            term.value * term.weight
            for carrier in run.outputs
            for term in carrier.effects.losses
        ]
        if not weighted:
            raise RuntimeError("pipeline training step emitted no loss terms")
        # Autograd supports explicit device copies, but torch.stack requires
        # all inputs to start on one device.  Aggregate on the final loss
        # term's device so multi-device stage losses retain gradient paths.
        loss_device = weighted[-1].device
        loss = torch.stack(
            [value.reshape(()).to(device=loss_device) for value in weighted]
        ).sum()
        if loss_reduction == "mean":
            loss = loss / len(weighted)
        elif loss_reduction != "sum":
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        loss.backward()
        if mutation_owners is not None:
            self.commit_mutations(run.outputs, mutation_owners)
        self.optimizer_step(optimizer)
        return run, loss.detach()


__all__ = [
    "BoundaryAdapter",
    "BoundaryStats",
    "CancellationToken",
    "CodebookIdentity",
    "CodebookReader",
    "CodebookReaderView",
    "DeferredMutation",
    "DenseEvent",
    "Diagnostic",
    "FactoredEvent",
    "LossTerm",
    "MaterializeMode",
    "ParameterVersionOwner",
    "PipelineAddress",
    "PipelineControl",
    "PipelineEffects",
    "PipelineExecutionCancelled",
    "PipelineExecutionError",
    "PipelineExecutor",
    "PipelineRun",
    "PipelineStage",
    "ReplicaCodebookReader",
    "ReplicaRegistry",
    "ReverseTrace",
    "SelectedEvent",
    "SelectionSlot",
    "StageTelemetry",
    "SubSpace",
    "SubSpaceSchema",
    "TraceEntry",
    "make_codebook_reader",
]
