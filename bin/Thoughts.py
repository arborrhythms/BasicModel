"""Levelled execution records and replay over the existing What memory owner.

No memory is allocated here. The mixin uses WhatInteractionMemory._what_slots;
context views are recovered from that one chronological history, never stored
as a second planner stack. ConceptualMeaning remains the semantic value.
"""

from dataclasses import dataclass, replace
import collections
from collections.abc import Mapping
import math

from Meaning import ConceptualMeaning, _freeze_metadata
from Queries import ThoughtResult
from What import LTMSlot, WhatSlotOperation


@dataclass(frozen=True)
class ThoughtRecord:
    id: int
    episode: int
    kind: str
    level: int
    meaning: ConceptualMeaning | None = None
    operation: str = ""
    cost: int = 0
    budget: int = 0
    pressure: float = 0.0
    forced: bool = False
    support_true: float = 0.0
    support_false: float = 0.0
    evidence_kind: str = "unverified"
    sources: tuple = ()
    reason: str = ""
    semantic_delta: float = 0.0
    result: ThoughtResult | None = None

    def __post_init__(self):
        for name in ("id", "episode", "level", "cost", "budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"thought {name} must be a non-negative integer")
        if self.kind not in (
            "begin",
            "thought",
            "descend",
            "return",
            "cutoff",
            "finish",
        ):
            raise ValueError("unknown thought transition")
        if self.meaning is not None and not isinstance(self.meaning, ConceptualMeaning):
            raise TypeError("thought content requires a complete ConceptualMeaning")
        if self.kind in ("begin", "thought", "descend") and self.meaning is None:
            raise ValueError("thought content cannot be empty")
        if self.kind in ("begin", "descend") and self.meaning.mode != "interrogative":
            raise ValueError(
                "a thinking context requires an interrogative initiating meaning"
            )
        for name in ("support_true", "support_false"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    "thought evidence degrees must be finite and between zero and one"
                )
        if self.meaning is None and (self.support_true or self.support_false):
            raise ValueError(
                "thought support requires its complete grammatical proposition"
            )
        if not math.isfinite(float(self.pressure)) or self.pressure < 0:
            raise ValueError("thought pressure must be finite and non-negative")
        if not math.isfinite(float(self.semantic_delta)) or self.semantic_delta < 0:
            raise ValueError("thought semantic delta must be finite and non-negative")
        if self.evidence_kind not in (
            "unverified",
            "inference",
            "fact",
            "question",
            "observation",
            "estimate",
            "taxonomy",
            "meronymy",
            "retrieval",
            "concept-codebook",
            "conceptual-identity",
            "subgoal",
        ):
            raise ValueError("unknown thought evidence kind")
        if self.result is not None:
            if not isinstance(self.result, ThoughtResult):
                raise TypeError("thought result requires a checked ThoughtResult")
            if self.kind not in ("thought", "return", "finish"):
                raise ValueError("only executed/returned/finished thoughts carry a result")
            if self.meaning is None:
                raise ValueError("thought result requires its complete grammatical meaning")
            if self.result.request.roles.shape != self.meaning.roles.shape:
                raise ValueError("thought result changed the episode conceptual width")
        object.__setattr__(self, "sources", _freeze_metadata(self.sources))

    def snapshot(self, *, detach):
        meaning = self.meaning
        if meaning is not None:
            meaning = meaning.detached() if detach else replace(meaning)
        result = self.result
        if result is not None and detach:
            result = result.detached()
        return replace(self, meaning=meaning, result=result)


@dataclass(frozen=True)
class ThoughtContextView:
    """A derived view, with no parent pointer or independent mutable state."""

    initial: ConceptualMeaning
    meaning: ConceptualMeaning
    results: tuple = ()


@dataclass(frozen=True)
class ThoughtState:
    episode: int
    contexts: tuple
    work_budget: int
    work_spent: int
    pressure: float
    finished: bool = False
    cutoff_depth: int | None = None
    drain_count: int = 0

    @property
    def level(self):
        return len(self.contexts) - 1

    @property
    def work_remaining(self):
        return self.work_budget - self.work_spent

    @property
    def forced(self):
        return self.cutoff_depth is not None


def replay_thoughts(records):
    """Validate chronological execution and derive the most recent episode."""
    state = None
    previous = -1
    for record in records:
        if not isinstance(record, ThoughtRecord):
            continue  # explicitly retained legacy prefix, not execution evidence
        if record.id <= previous:
            raise ValueError("thought occurrences must be strictly chronological")
        previous = record.id
        if record.kind == "begin":
            if state is not None and not state.finished:
                raise ValueError("new thought episode overlaps an active context")
            if (
                record.level != 0
                or record.episode != record.id
                or record.cost
                or record.forced
            ):
                raise ValueError("invalid root thought episode")
            state = ThoughtState(
                record.episode,
                (ThoughtContextView(record.meaning, record.meaning),),
                record.budget,
                0,
                0.0,
            )
        else:
            if state is None or state.finished or record.episode != state.episode:
                raise ValueError("thought event has no matching active episode")
            if record.budget:
                raise ValueError("only the root begin event may declare a work budget")
            if (
                record.meaning is not None
                and record.meaning.roles.shape != state.contexts[0].initial.roles.shape
            ):
                raise ValueError("thought meaning changed the episode conceptual width")
            if record.kind == "cutoff":
                if (
                    state.forced
                    or record.level != state.level
                    or record.cost
                    or not record.forced
                ):
                    raise ValueError("invalid or repeated thought cutoff")
                state = replace(state, cutoff_depth=state.level)
            else:
                if state.forced:
                    if (
                        record.kind not in ("return", "finish")
                        or record.cost
                        or not record.forced
                    ):
                        raise ValueError(
                            "budget cutoff permits only one bounded return/finish drain"
                        )
                    state = replace(state, drain_count=state.drain_count + 1)
                    if state.drain_count > state.cutoff_depth + 1:
                        raise ValueError("thought termination drain exceeded its bound")
                else:
                    if (
                        record.cost < 1
                        or record.forced
                        or record.cost > state.work_remaining
                    ):
                        raise ValueError(
                            "ordinary thought work exceeded the shared budget"
                        )
                    spent = state.work_spent + record.cost
                    state = replace(
                        state,
                        work_spent=spent,
                        pressure=spent / max(1, state.work_budget),
                    )
                contexts = list(state.contexts)
                if record.kind == "thought":
                    if record.level != state.level:
                        raise ValueError(
                            "ordinary thought cannot change execution level"
                        )
                    contexts[-1] = replace(contexts[-1], meaning=record.meaning)
                elif record.kind == "descend":
                    if record.level != state.level + 1:
                        raise ValueError("thought descent skipped a level")
                    contexts.append(ThoughtContextView(record.meaning, record.meaning))
                elif record.kind == "return":
                    if state.level == 0 or record.level != state.level - 1:
                        raise ValueError("thought return underflow or non-LIFO level")
                    contexts.pop()
                    contexts[-1] = replace(
                        contexts[-1], results=contexts[-1].results + (record,)
                    )
                elif record.kind == "finish":
                    if state.level != 0 or record.level != 0:
                        raise ValueError("only the root context can finish an episode")
                    if record.meaning is not None:
                        contexts[0] = replace(contexts[0], meaning=record.meaning)
                    state = replace(state, finished=True)
                state = replace(state, contexts=tuple(contexts))
        if abs(float(record.pressure) - state.pressure) > 1e-9:
            raise ValueError("thought pressure differs from the shared work history")
    if (
        state is not None
        and not state.finished
        and not state.work_remaining
        and not state.forced
    ):
        raise ValueError("exhausted thought history is missing its cutoff event")
    return state


def _thought_references(value, namespace):
    """Find semantic occurrence references; arbitrary integers are not IDs."""
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        if (
            len(value) == 3
            and value[0] == "thought"
            and value[1] == namespace
            and type(value[2]) is int
        ):
            return {value[2]}
        out = set()
        for child in value:
            out.update(_thought_references(child, namespace))
        return out
    return set()


def _legacy_prefix_pressure(records):
    """Validate the retained compatibility prefix without inferring ordinary levels."""
    pending = 0
    pressure = 0.0
    ordinary = False
    for record in records:
        if isinstance(record, ThoughtRecord):
            if pending:
                raise ValueError("ordinary history overlaps an open legacy interaction")
            ordinary = True
            continue
        if not isinstance(record, LTMSlot) or ordinary:
            raise ValueError("legacy records must precede ordinary thought history")
        if not math.isfinite(float(record.closure_pressure)):
            raise ValueError("legacy closure pressure must be finite")
        if pending and record.closure_pressure < pressure:
            raise ValueError(
                "legacy closure pressure decreased while an input remained open"
            )
        if record.operation is WhatSlotOperation.OPEN:
            pending += 1
        elif record.operation is WhatSlotOperation.CLOSE:
            if not pending:
                raise ValueError("legacy return underflow: no open input")
            pending -= 1
        pressure = float(record.closure_pressure) if pending else 0.0
    return pressure


class LevelledThoughtHistory:
    """Methods on the existing interaction owner; `_what_slots` owns all records."""

    def _thought_row(self, b):
        if type(b) is not int or not 0 <= b < self.batch:
            raise IndexError("thought row is outside the current batch")
        return b

    def thought_history(self, b=0):
        return [
            record
            for record in self._what_slots[self._thought_row(b)]
            if isinstance(record, ThoughtRecord)
        ]

    def thought_window(self, b=0, limit=4):
        """At most ``limit`` recent owned slots, without a full-history scan."""
        if type(limit) is not int or limit < 0:
            raise ValueError("thought window requires a non-negative bound")
        slots = self._what_slots[self._thought_row(b)]
        from itertools import islice
        window = tuple(islice(reversed(slots), limit))
        return tuple(record for record in reversed(window)
                     if isinstance(record, ThoughtRecord))

    def retrieved_frames(self, b=0, limit=32):
        """Only what effects enter this STM view; never read the store here."""
        frames, seen = [], set()
        for record in reversed(self.thought_window(b=b, limit=limit)):
            result = record.result
            if result is None or result.semantic_id != 'what':
                continue
            for frame in result.evidence.get('frames', ()):
                reference = frame['occurrence']
                if reference not in seen:
                    frames.append(frame)
                    seen.add(reference)
        return tuple(reversed(frames))

    def thought_state(self, b=0):
        return replay_thoughts(self.thought_history(b=b))

    def retained_ltm_occurrences(self, namespace):
        """Derive LTM roots from retained ordinary thought records.

        This walks the existing chronological owner at the moment retention is
        requested; it does not allocate a reference-count table or inspect
        content tensors.  Roles, bindings, scopes and recorded sources may all
        name durable LTM occurrences.  Finished records remain roots while
        their history remains retained.
        """
        if not isinstance(namespace, str):
            raise TypeError("LTM occurrence namespace must be a string")
        found, pending = set(), []
        for row in range(self.batch):
            for record in self.thought_history(b=row):
                pending.append(record.sources)
                pending.append(record.result)
                if record.meaning is not None:
                    pending.append(record.meaning.metadata())
        while pending:
            value = pending.pop()
            if isinstance(value, ThoughtResult):
                pending.append(value.evidence)
                continue
            if isinstance(value, ConceptualMeaning):
                pending.append(value.metadata())
                continue
            if isinstance(value, Mapping):
                pending.extend(value.values())
                continue
            if not isinstance(value, (tuple, list)):
                continue
            if (len(value) == 3 and value[0] == "ltm"
                    and value[1] == namespace
                    and type(value[2]) is int and value[2] >= 0):
                found.add(tuple(value))
                continue
            pending.extend(value)
        return tuple(sorted(found))

    def thought_reference(self, record, b=0):
        if not isinstance(record, ThoughtRecord) or not any(
            record is item for item in self.thought_history(b=b)
        ):
            raise ValueError(
                "thought reference requires an occurrence owned by this row"
            )
        return ("thought", self._thought_namespace, record.id)

    def resolve_thought(self, reference, *, b=0, max_records=1024, work=None):
        if (
            not isinstance(reference, tuple)
            or len(reference) != 3
            or reference[:2] != ("thought", self._thought_namespace)
            or type(reference[2]) is not int
            or reference[2] < 0
        ):
            raise ValueError("thought occurrence reference is unavailable")
        if type(max_records) is not int or max_records < 0:
            raise ValueError("thought read limit must be a non-negative integer")
        records = self._what_slots[self._thought_row(b)]
        iterator = iter(records)
        for index in range(min(len(records), max_records)):
            if work is not None:
                work.require("record")
            record = next(iterator)
            if isinstance(record, ThoughtRecord) and record.id == reference[2]:
                if record.meaning is None:
                    raise ValueError("transition occurrence has no grammatical content")
                return record.meaning, index + 1
        raise ValueError(
            "thought occurrence is unavailable within this row and read limit"
        )

    def _retain_thought_history(self, records, reserve=0):
        """Evict only whole closed prefixes with no retained semantic dependents."""
        records = list(records)
        while len(records) + reserve > self.capacity:
            cut = None
            if isinstance(records[0], LTMSlot):
                level = 0
                for index, record in enumerate(records):
                    if not isinstance(record, LTMSlot):
                        break
                    level += int(record.operation is WhatSlotOperation.OPEN)
                    level -= int(record.operation is WhatSlotOperation.CLOSE)
                    if level == 0:
                        cut = index + 1
                        break
            else:
                for index, record in enumerate(records):
                    if isinstance(record, ThoughtRecord) and record.kind == "finish":
                        cut = index + 1
                        break
            if cut is None:
                raise OverflowError(
                    "thought capacity cannot evict active contexts or their drain reserve"
                )
            removed = {
                record.id
                for record in records[:cut]
                if isinstance(record, ThoughtRecord)
            }
            needed = set()
            for record in records[cut:]:
                if isinstance(record, ThoughtRecord):
                    needed.update(
                        _thought_references(record.sources, self._thought_namespace)
                    )
                    if record.meaning is not None:
                        needed.update(
                            _thought_references(
                                record.meaning.metadata(), self._thought_namespace
                            )
                        )
            if removed & needed:
                raise OverflowError(
                    "thought capacity cannot evict referenced semantic occurrences"
                )
            del records[:cut]
        return records

    def _store_thought_events(self, events, *, b, reserve=0):
        current = list(self._what_slots[b])
        live = self.detach_mode == "episode" and b in self._episode_live
        stored = [record.snapshot(detach=not live) for record in events]
        candidate = self._retain_thought_history(current + stored, reserve)
        replay_thoughts(candidate)  # validation precedes every owner mutation
        self._what_slots[b] = collections.deque(candidate)
        self._thought_next_id += len(stored)
        if live:
            self._episode_live[b].extend(stored)
        return stored[0]

    def begin_thought_episode(self, meaning, *, b=0, work_budget=32):
        b = self._thought_row(b)
        state = self.thought_state(b=b)
        if state is not None and not state.finished:
            raise RuntimeError("a thought episode is already active")
        if b in self._episode_live:
            raise RuntimeError(
                "end the previous episode credit before starting another episode"
            )
        if self.open_what_slots(b=b):
            raise RuntimeError(
                "legacy unanswered interactions must close before ordinary thought execution"
            )
        if type(work_budget) is not int or work_budget < 0:
            raise ValueError("thought work budget must be a non-negative integer")
        identifier = self._thought_next_id
        record = ThoughtRecord(
            identifier, identifier, "begin", 0, meaning, budget=work_budget
        )
        events = [record]
        if work_budget == 0:
            events.append(
                ThoughtRecord(
                    identifier + 1,
                    identifier,
                    "cutoff",
                    0,
                    forced=True,
                    reason="work_budget",
                )
            )
        # Reserve a cutoff and the one root finish, or just the pending finish.
        reserve = 1 if work_budget == 0 else 2
        self._retain_thought_history(list(self._what_slots[b]) + events, reserve)
        self._episode_live[b] = []
        try:
            return self._store_thought_events(events, b=b, reserve=reserve)
        except Exception:
            self._episode_live.pop(b, None)
            raise

    def _thought_event(
        self,
        kind,
        meaning=None,
        *,
        b=0,
        operation="",
        work=1,
        support_true=0.0,
        support_false=0.0,
        evidence_kind="unverified",
        sources=(),
        reason="",
        result=None,
    ):
        b = self._thought_row(b)
        state = self.thought_state(b=b)
        if state is None or state.finished:
            raise RuntimeError("thought episode is absent or finished")
        if (
            meaning is not None
            and isinstance(meaning, ConceptualMeaning)
            and meaning.roles.shape != state.contexts[0].initial.roles.shape
        ):
            raise ValueError("thought meaning changed the episode conceptual width")
        if kind == "return" and state.level == 0:
            raise ValueError("thought return underflow at root level")
        if kind == "finish" and state.level != 0:
            raise ValueError("only root level can finish a thought episode")
        if state.forced and kind not in ("return", "finish"):
            raise RuntimeError("budget cutoff permits only return and finish")
        if type(work) is not int or work < 1:
            raise ValueError("ordinary thought work must be a positive integer")
        cost = 0 if state.forced or kind == "cutoff" else work
        if cost > state.work_remaining:
            raise RuntimeError(
                "selected work exceeds the remaining shared thought budget"
            )
        level = state.level + (kind == "descend") - (kind == "return")
        spent = state.work_spent + cost
        pressure = spent / max(1, state.work_budget)
        delta = 0.0
        if (
            meaning is not None
            and state.contexts[-1].meaning.roles.shape == meaning.roles.shape
        ):
            delta = float(
                (meaning.roles.detach() - state.contexts[-1].meaning.roles.detach())
                .square()
                .mean()
                .sqrt()
            )
        # A checked result is hard-boundary evidence even while the ordinary
        # episode keeps its selected grammatical meanings live for one
        # optimizer step.  Preserve an owned snapshot here rather than letting
        # a history record alias the controller's transient return object.
        if result is not None:
            if not isinstance(result, ThoughtResult):
                raise TypeError("thought result requires a checked ThoughtResult")
            result = result.detached()
        record = ThoughtRecord(
            self._thought_next_id,
            state.episode,
            kind,
            level,
            meaning,
            operation=operation,
            cost=cost,
            pressure=pressure,
            forced=state.forced or kind == "cutoff",
            support_true=support_true,
            support_false=support_false,
            evidence_kind=evidence_kind,
            sources=sources,
            reason=reason,
            semantic_delta=delta,
            result=result,
        )
        events = [record]
        forced = state.forced or kind == "cutoff"
        if spent == state.work_budget and not forced and kind != "finish":
            events.append(
                ThoughtRecord(
                    record.id + 1,
                    state.episode,
                    "cutoff",
                    level,
                    pressure=pressure,
                    forced=True,
                    reason="work_budget",
                )
            )
            forced = True
        # Room for one cutoff if needed, all LIFO returns and one root finish.
        reserve = 0 if kind == "finish" else level + 1 + (not forced)
        return self._store_thought_events(events, b=b, reserve=reserve)

    def commit_thought(self, meaning, *, operation="refine", **kwargs):
        return self._thought_event("thought", meaning, operation=operation, **kwargs)

    def descend_thought(self, meaning, **kwargs):
        return self._thought_event("descend", meaning, operation="descend", **kwargs)

    def return_thought(self, meaning=None, **kwargs):
        return self._thought_event("return", meaning, operation="return", **kwargs)

    def finish_thought(self, meaning=None, **kwargs):
        return self._thought_event("finish", meaning, operation="finish", **kwargs)

    def cutoff_thought(self, *, reason="work_budget", **kwargs):
        return self._thought_event(
            "cutoff", operation="cutoff", reason=reason, **kwargs
        )

    def thought_extras(self):
        """Detached versioned snapshot; the existing owner remains authoritative."""
        rows = []
        for row in self._what_slots:
            _legacy_prefix_pressure(row)
            values = []
            for record in row:
                if isinstance(record, LTMSlot):
                    values.append(
                        {
                            "legacy": replace(
                                record,
                                input=self._detach_what_value(record.input),
                                output=self._detach_what_value(record.output),
                                question=self._detach_what_value(record.question),
                                grammar_trace=self._detach_what_value(
                                    record.grammar_trace
                                ),
                            )
                        }
                    )
                    continue
                value = dict(record.__dict__)
                meaning = value.pop("meaning")
                result = value.pop("result")
                value["meaning"] = (
                    None
                    if meaning is None
                    else {
                        "roles": meaning.roles.detach().cpu().clone(),
                        "role_mask": meaning.role_mask.detach().cpu().clone(),
                        **meaning.metadata(),
                    }
                )
                value["result"] = (
                    None if result is None else result.checkpoint()
                )
                values.append(value)
            rows.append(values)
        return {
            "version": 3,
            "namespace": self._thought_namespace,
            "next_id": self._thought_next_id,
            "batch": self.batch,
            "rows": rows,
        }

    def load_thought_extras(self, extras):
        """Validate every row before restoring history; rebuild context by replay."""
        if not isinstance(extras, dict) or extras.get("version") not in (1, 2, 3):
            raise ValueError("unsupported thought history checkpoint")
        version = extras["version"]
        rows = extras.get("rows")
        batch = extras.get("batch")
        if (
            type(batch) is not int
            or batch < 1
            or not isinstance(rows, list)
            or len(rows) != batch
        ):
            raise ValueError("invalid thought history rows")
        namespace = extras.get("namespace")
        if not isinstance(namespace, str) or len(namespace) != 32:
            raise ValueError("invalid thought occurrence namespace")
        try:
            if bytes.fromhex(namespace).hex() != namespace:
                raise ValueError("invalid thought occurrence namespace")
        except ValueError as error:
            raise ValueError("invalid thought occurrence namespace") from error
        counter = extras.get("next_id")
        if type(counter) is not int or counter < 0:
            raise ValueError("invalid thought occurrence counter")
        candidate = []
        legacy_pressure = []
        identifiers = set()
        for row in rows:
            records = []
            for raw in row:
                value = dict(raw)
                if set(value) == {"legacy"}:
                    if not isinstance(value["legacy"], LTMSlot):
                        raise ValueError("invalid legacy thought checkpoint prefix")
                    legacy = value["legacy"]
                    records.append(
                        replace(
                            legacy,
                            input=self._detach_what_value(legacy.input),
                            output=self._detach_what_value(legacy.output),
                            question=self._detach_what_value(legacy.question),
                            grammar_trace=self._detach_what_value(legacy.grammar_trace),
                        )
                    )
                    continue
                meaning = value.pop("meaning")
                if meaning is not None:
                    meaning = ConceptualMeaning(**meaning).detached()
                raw_result = value.pop("result", None)
                if raw_result is not None:
                    if version < 2:
                        raise ValueError("legacy thought checkpoint has a typed result")
                    result = ThoughtResult.from_checkpoint(raw_result)
                else:
                    result = None
                record = ThoughtRecord(meaning=meaning, result=result, **value)
                if record.id in identifiers or record.id >= counter:
                    raise ValueError("invalid or reused thought occurrence identity")
                identifiers.add(record.id)
                records.append(record)
            legacy_pressure.append(_legacy_prefix_pressure(records))
            state = replay_thoughts(records)
            reserve = (
                0
                if state is None or state.finished
                else state.level + 1 + (not state.forced)
            )
            if len(records) + reserve > self.capacity:
                raise OverflowError(
                    "thought checkpoint exceeds capacity including active drain reserve"
                )
            candidate.append(collections.deque(records))
        self.batch = batch
        self._what_slots = candidate
        self._what_closure_pressure = legacy_pressure
        self._episode_live = {
            b: []
            for b, row in enumerate(candidate)
            if (state := replay_thoughts(row)) is not None and not state.finished
        }
        self._thought_namespace = namespace
        self._thought_next_id = counter
