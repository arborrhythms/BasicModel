"""Teacher-owned reconstruction lessons, source queries, and loss bookkeeping.

The Teacher is deliberately outside ``nn.Module``. It owns request-scoped
training state, clean targets, and the objective corpus address of each
presentation. Objective source coordinates are *not* the model's subjective
``.where`` / ``.when`` event bands: Teacher never reads, writes, or replaces
those bands.

Version 1 is the complete-present-input lesson. Degradation can be added at
this boundary later: the Teacher will retain a clean target while presenting a
masked Perception to the student. The initial seam is numerically conservative
and delegates the established reconstruction math to ``ModelLoss``.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

import torch

from Layers import ModelLoss, Error, TheError
from What import What


NOTHING_CONTEXT_ID = 0
EVERYTHING_CONTEXT_ID = 1


def _jsonable(value):
    """Return a deterministic JSON-compatible view of manifest metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _stable_context_id(payload: Mapping[str, Any]) -> int:
    """Map context metadata to a stable positive signed-int64 identity."""
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity = int.from_bytes(
        hashlib.sha256(encoded).digest()[:8], byteorder="big", signed=False
    )
    identity &= (1 << 63) - 1
    if identity <= EVERYTHING_CONTEXT_ID:
        identity += 2
    return identity


def _intrinsic_manifest(manifest: Mapping[str, Any] | None):
    """Select source identity, excluding loader/view parameters.

    ``max_docs`` and batching policy change a training view, not the corpus
    snapshot. Shard content hashes (or an inline content hash) are the version
    coordinate that must remain stable across such views.
    """
    manifest = dict(manifest or {})
    keys = (
        "dataset",
        "corpus",
        "source",
        "shards",
        "snapshot",
        "revision",
        "content_sha256",
    )
    return {
        key: manifest[key]
        for key in keys
        if key in manifest
    }


@dataclass(frozen=True)
class ContextWhole:
    """An explicitly named discourse whole beneath ``EVERYTHING``.

    This is an addressing identity, not a physical container. A book can move
    while remaining the same context whole. Event-local containment is
    represented elsewhere by the world mereology.
    """

    identity: int
    name: str
    kind: str
    parent_identity: int = EVERYTHING_CONTEXT_ID
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data, *, name: str | None = None) -> "ContextWhole":
        manifest = dict(getattr(data, "source_manifest", None) or {})
        identity_manifest = _intrinsic_manifest(manifest)
        dataset = str(manifest.get("dataset", "unknown"))
        if name is None or not str(name).strip():
            shards = manifest.get("shards") or ()
            shard_label = ",".join(str(value) for value in shards)
            name = f"{dataset}:{shard_label}" if shard_label else dataset
        payload = {
            "name": str(name),
            "kind": "text" if dataset in ("text", "inline") else dataset,
            "manifest": identity_manifest,
            "parent_identity": EVERYTHING_CONTEXT_ID,
        }
        return cls(
            identity=_stable_context_id(payload),
            name=str(name),
            kind=str(payload["kind"]),
            metadata=_jsonable(manifest),
        )


@dataclass(frozen=True)
class ObjectiveWhere:
    """Lossless spatial address inside a named corpus snapshot.

    ``row`` is the direct table lookup. ``document`` / ``sentence`` / character
    span are retained as independently checkable provenance rather than asking
    the student to infer sentence identity from a scalar row number.
    """

    context_identity: int
    split: str
    row: int
    document: int
    sentence: int
    char_start: int
    char_end: int
    external_id: str | None = None


@dataclass(frozen=True)
class ObjectiveWhen:
    """Version coordinate at which an objective ``where`` is valid.

    A source publication date is optional metadata. The snapshot identity is
    mandatory because publication time alone neither identifies a document nor
    distinguishes corpus revisions.
    """

    snapshot_identity: int
    source_time: str | None = None


@dataclass
class ReconstructionLesson:
    """Request-scoped information shared by Teacher and student."""

    split: str
    batch_size: int
    training: bool
    context: ContextWhole
    reading_mode: bool = True
    clean_input: Any = None
    word_mask: torch.Tensor | None = None
    objective_where: tuple[tuple[ObjectiveWhere, ...], ...] = ()
    objective_when: tuple[tuple[ObjectiveWhen, ...], ...] = ()
    objective_where_code: torch.Tensor | None = None
    objective_when_code: torch.Tensor | None = None
    objective_mask: torch.Tensor | None = None
    clean_what: tuple[tuple[Any, ...], ...] = ()


class Teacher:
    """Own data access, reconstruction loss, and the batch error registry."""

    def __init__(
        self,
        data,
        *,
        enabled: bool = False,
        context_name: str | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
        errors: Error | None = None,
    ):
        if data is None:
            raise ValueError("Teacher requires a Data instance")
        self.data = data
        self.enabled = bool(enabled)
        self.loss = ModelLoss(**dict(loss_kwargs or {}))
        # Keep the established process-wide sink during the migration: spaces
        # and diagnostic callers already refer to TheError. Teacher is now its
        # lifecycle owner, so the object can become instance-local later
        # without changing lesson semantics.
        self.errors = errors if errors is not None else TheError
        self.errors.attach(self.loss)
        self.context = ContextWhole.from_data(data, name=context_name)
        self.snapshot_identity = _stable_context_id({
            "context": self.context.identity,
            "manifest": _intrinsic_manifest(
                getattr(data, "source_manifest", None)),
        })
        self.lesson: ReconstructionLesson | None = None
        self._staged_split: str | None = None
        self._staged_source_rows = None
        self._install_context(self.context)

    def __deepcopy__(self, memo):
        """Copy model-owned Teacher state without copying a live loss graph.

        ``Error`` may contain non-leaf tensors from the last batch, and a
        lesson may retain detached request tensors. Neither is model state.
        Rebuild the request-local pieces while preserving shared identities
        through ``memo`` so ``copied_model.loss is copied_model.teacher.loss``.
        """
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        copied.data = copy.deepcopy(self.data, memo)
        copied.enabled = self.enabled
        copied.loss = copy.deepcopy(self.loss, memo)
        copied.errors = Error(copied.loss)
        copied.errors._disabled = set(self.errors._disabled)
        copied.context = self.context
        copied.snapshot_identity = self.snapshot_identity
        copied.lesson = None
        copied._staged_split = None
        copied._staged_source_rows = None
        copied._install_context(copied.context)
        return copied

    @property
    def legacy_prediction_enabled(self) -> bool:
        """Whether independent next-state predictor losses may train."""
        return not self.enabled

    def _install_context(self, context: ContextWhole) -> None:
        setter = getattr(self.data, "set_context_whole", None)
        if callable(setter):
            setter(context)
        else:
            setattr(self.data, "active_context", context)

    def set_context(
        self,
        name: str,
        *,
        kind: str = "interactive",
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextWhole:
        """Select a new corpus/book/conversation whole for later lessons."""
        payload = {
            "name": str(name),
            "kind": str(kind),
            "metadata": dict(metadata or {}),
            "parent_identity": EVERYTHING_CONTEXT_ID,
        }
        self.context = ContextWhole(
            identity=_stable_context_id(payload),
            name=str(name),
            kind=str(kind),
            metadata=_jsonable(dict(metadata or {})),
        )
        self.snapshot_identity = _stable_context_id({
            "context": self.context.identity,
            "metadata": dict(metadata or {}),
        })
        self._install_context(self.context)
        return self.context

    def where(self, split: str, row: int) -> ObjectiveWhere:
        """Return the objective source address for one dataset row."""
        split = str(split)
        row = int(row)
        getter = getattr(self.data, "source_address", None)
        try:
            address = getter(split, row) if callable(getter) else None
        except (KeyError, IndexError):
            address = None
        if address is None:
            address = {
                "split": split,
                "row": row,
                "document": row,
                "sentence": 0,
                "char_start": 0,
                "char_end": 0,
            }
        return ObjectiveWhere(
            context_identity=int(self.context.identity),
            split=str(address.get("split", split)),
            row=int(address.get("row", row)),
            document=int(address.get("document", row)),
            sentence=int(address.get("sentence", 0)),
            char_start=int(address.get("char_start", 0)),
            char_end=int(address.get("char_end", 0)),
            external_id=(
                str(address["external_id"])
                if address.get("external_id") is not None else None
            ),
        )

    def when(self, where: ObjectiveWhere) -> ObjectiveWhen:
        """Return the immutable snapshot/source-time coordinate for ``where``."""
        getter = getattr(self.data, "source_address", None)
        try:
            address = (
                getter(where.split, where.row) if callable(getter) else {}
            )
        except (KeyError, IndexError):
            address = {}
        source_time = address.get("source_time")
        return ObjectiveWhen(
            snapshot_identity=int(self.snapshot_identity),
            source_time=(str(source_time) if source_time is not None else None),
        )

    def What(self, where: ObjectiveWhere, when: ObjectiveWhen):
        """Compatibility adapter for the canonical ``Data.what()`` lookup."""
        if int(where.context_identity) != int(self.context.identity):
            raise KeyError(
                "objective where belongs to a different context whole"
            )
        if int(when.snapshot_identity) != int(self.snapshot_identity):
            raise KeyError(
                "objective when belongs to a different corpus snapshot"
            )
        canonical_where = self.where(where.split, where.row)
        if canonical_where != where:
            raise KeyError(
                "objective where does not match the canonical source address"
            )
        canonical_when = self.when(where)
        if canonical_when != when:
            raise KeyError(
                "objective when does not match the canonical source version"
            )
        answer = self.data.what(What.present(
            int(where.row), split=str(where.split)))
        if not answer.available:
            raise KeyError(answer.reason)
        return answer.what

    def stage_batch_sources(self, split: str, source_rows) -> None:
        """Stage row indices emitted by the data cursor for the next lesson."""
        self._staged_split = str(split)
        self._staged_source_rows = source_rows

    def _resolve_staged_sources(self, split: str, batch_size: int):
        rows = self._staged_source_rows
        staged_split = self._staged_split
        self._staged_source_rows = None
        self._staged_split = None
        if rows is None or staged_split != str(split):
            return (), ()
        # Packed input is ``list[list[row]]``; ordinary input is ``list[row]``.
        nested = (
            rows if rows and isinstance(rows[0], (list, tuple))
            else [[row] for row in rows]
        )
        nested = list(nested)[:int(batch_size)]
        while len(nested) < int(batch_size):
            nested.append([])
        where_rows = []
        when_rows = []
        for source_row in nested:
            addressed = tuple(self.where(split, int(row)) for row in source_row)
            where_rows.append(addressed)
            when_rows.append(tuple(self.when(where) for where in addressed))
        return tuple(where_rows), tuple(when_rows)

    def begin_batch(
        self,
        *,
        split: str,
        batch_size: int,
        training: bool,
        clean_input=None,
    ) -> ReconstructionLesson:
        """Open one complete-input reconstruction lesson."""
        self.errors.reset()
        self.errors.attach(self.loss)
        self._install_context(self.context)
        objective_where, objective_when = self._resolve_staged_sources(
            split, batch_size
        )
        clean_what = tuple(
            tuple(
                self.What(where, when)
                for where, when in zip(where_row, when_row)
            )
            for where_row, when_row
            in zip(objective_where, objective_when)
        )
        self.lesson = ReconstructionLesson(
            split=str(split),
            batch_size=int(batch_size),
            training=bool(training),
            context=self.context,
            clean_input=(
                clean_input.detach()
                if torch.is_tensor(clean_input)
                else clean_input
            ),
            objective_where=objective_where,
            objective_when=objective_when,
            clean_what=clean_what,
        )
        return self.lesson

    @staticmethod
    def _split_code(split: str) -> int:
        return {
            "train": 0,
            "validation": 1,
            "test": 2,
            "runtime": 3,
        }.get(str(split), 4)

    @staticmethod
    def _optional_identity(value: str | None) -> int:
        if value is None:
            return 0
        return _stable_context_id({"value": str(value)})

    def _encode_objective_query(self):
        """Build exact padded objective-address tensors for the student seam.

        These tensors are categorical identifiers and integer positions; their
        numeric magnitude is not semantic. A future student-side address
        encoder may embed them, but they never occupy the subjective event
        ``.where`` / ``.when`` bands.
        """
        lesson = self.lesson
        if lesson is None:
            return None, None, None
        batch = int(lesson.batch_size)
        slots = max(
            (len(row) for row in lesson.objective_where),
            default=0,
        )
        where_code = torch.zeros(
            batch, slots, 8, dtype=torch.long
        )
        when_code = torch.zeros(
            batch, slots, 2, dtype=torch.long
        )
        mask = torch.zeros(
            batch, slots, dtype=torch.bool
        )
        for b, (where_row, when_row) in enumerate(zip(
                lesson.objective_where, lesson.objective_when)):
            for s, (where, when) in enumerate(zip(where_row, when_row)):
                where_code[b, s] = torch.tensor(
                    (
                        int(where.context_identity),
                        self._split_code(where.split),
                        int(where.row),
                        int(where.document),
                        int(where.sentence),
                        int(where.char_start),
                        int(where.char_end),
                        self._optional_identity(where.external_id),
                    ),
                    dtype=torch.long,
                )
                when_code[b, s] = torch.tensor(
                    (
                        int(when.snapshot_identity),
                        self._optional_identity(when.source_time),
                    ),
                    dtype=torch.long,
                )
                mask[b, s] = True
        return where_code, when_code, mask

    def bind_input(self, subspace, *, word_mask=None):
        """Expose the objective query on the Teacher/student boundary.

        The returned lesson is available as ``model.teacher.lesson``. This
        method deliberately performs no assignment to ``subspace`` and does
        not inspect its subjective coordinate bands.
        """
        if subspace is None:
            return None
        inferred_mask = (
            word_mask if torch.is_tensor(word_mask)
            else getattr(subspace, "valid_mask", None)
        )
        batch = (
            int(inferred_mask.shape[0])
            if torch.is_tensor(inferred_mask) and inferred_mask.dim() > 0
            else (
                int(self.lesson.batch_size)
                if self.lesson is not None else 0
            )
        )
        if batch < 1:
            return None
        if self.lesson is None or self.lesson.batch_size != batch:
            self.begin_batch(
                split="runtime",
                batch_size=batch,
                training=False,
            )
        if word_mask is None:
            word_mask = inferred_mask
        if torch.is_tensor(word_mask):
            word_mask = word_mask.detach().to(dtype=torch.bool)

        where_code, when_code, address_mask = self._encode_objective_query()

        self.lesson.context = self.context
        self.lesson.word_mask = word_mask
        self.lesson.objective_where_code = where_code
        self.lesson.objective_when_code = when_code
        self.lesson.objective_mask = address_mask
        return self.lesson

    def primary_loss(self, loss_out, loss_in=None, sbow=None):
        """Compose the established task and reconstruction objective."""
        return self.loss.total(loss_out, loss_in, sbow)

    def add(self, name, value, *, weight=1.0, space=None, category="other"):
        """Register one Teacher-scored loss component."""
        self.errors.add(
            name, value, weight=weight, space=space, category=category
        )

    @contextmanager
    def runtime_batch(self, inputs, outputs=None, *, context=None):
        """Stage interactive data under an explicit temporary context."""
        previous = self.context
        if context is not None:
            if isinstance(context, ContextWhole):
                self.context = context
                self._install_context(context)
            else:
                self.set_context(str(context), kind="interactive")
        try:
            count = (
                int(inputs.shape[0])
                if torch.is_tensor(inputs)
                else len(inputs)
            )
            self.lesson = None
            self.stage_batch_sources("runtime", list(range(count)))
            with self.data.runtime_batch(
                inputs, outputs, context=self.context
            ):
                yield
        finally:
            self.context = previous
            self._install_context(previous)
