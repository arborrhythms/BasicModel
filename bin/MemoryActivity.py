"""The common bounded activity of an idea or a relation."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Activity:
    default: float = .5
    step: float = .1
    decay: float = .9
    promotion: float = .9

    def __post_init__(self):
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for value in (self.default, self.step, self.decay, self.promotion)):
            raise ValueError('activity parameters must be finite and in [0, 1]')

    @classmethod
    def from_xml(cls):
        from util import TheXMLConfig
        return cls(*(float(TheXMLConfig.get('architecture.activity' + name, default=value))
                     for name, value in (('Default', .5), ('Step', .1),
                                         ('Decay', .9), ('Promotion', .9))))


@dataclass(frozen=True)
class ReadEffect:
    count: int = 0
    written: bool = False


@torch.no_grad()
def written(store, row, surprise=-1.):
    if not math.isfinite(float(surprise)) or not (surprise == -1. or 0 <= surprise <= 1.):
        raise ValueError('write surprise must be unknown or in [0, 1]')
    store.activity[row] = store.activity_config.default if surprise < 0 else surprise


@torch.no_grad()
def accessed(store, rows):
    rows = torch.as_tensor(rows, device=store.activity.device, dtype=torch.long).reshape(-1).unique()
    journal = _reads.get()
    if journal is not None:
        reads = journal.setdefault(_stream.get(), {})
        for row in rows.tolist():
            previous = reads.get((store,row),ReadEffect())
            reads[store,row] = ReadEffect(previous.count+1, previous.written)
    else:
        apply({None: {(store,int(row)): ReadEffect(1) for row in rows.tolist()}})


@torch.no_grad()
def decay(store, rows):
    rows = torch.as_tensor(rows, device=store.activity.device, dtype=torch.long).reshape(-1)
    values = store.activity[rows]
    store.activity[rows] = torch.where(values == 1, values, values * store.activity_config.decay)


from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps

_reads = ContextVar('memory_activity_reads', default=None)
_stream = ContextVar('memory_activity_stream', default=None)


def deferred():
    return _reads.get() is not None


def snapshot():
    journal = _reads.get()
    return None if journal is None else {stream: dict(reads) for stream, reads in journal.items()}


def restore(saved):
    if saved is None:
        return
    journal = _reads.get()
    if journal is None:
        apply(saved)
    else:
        journal.clear()
        journal.update({stream: dict(reads) for stream, reads in saved.items()})


def select(states, winners):
    if all(state is None for state in states):
        return None
    selected = {}
    for stream, choice in enumerate(winners.tolist()):
        state = states[choice] or {}
        if stream in state:
            selected[stream] = dict(state[stream])
    # A batch-wide read is performed once if any selected path used it.
    common = {}
    for choice in set(winners.tolist()):
        for key, effect in (states[choice] or {}).get(None, {}).items():
            previous = common.get(key,ReadEffect())
            common[key] = ReadEffect(max(previous.count,effect.count), previous.written or effect.written)
    if common:
        selected[None] = common
    return selected


@torch.no_grad()
def apply(journal):
    counts = {}
    for reads in journal.values():
        for key, effect in reads.items():
            previous = counts.get(key,ReadEffect())
            counts[key] = ReadEffect(previous.count+effect.count, previous.written or effect.written)
    promote = {}
    for (store, row), effect in counts.items():
        value = float(store.activity[row])
        if value == 1. and not effect.written:
            continue
        store.activity[row] = min(1., value + effect.count * store.activity_config.step)
        promote.setdefault((store,effect.written), []).append(row)
    for (store,written_now), rows in promote.items():
        _promote(store, rows, written_now=written_now)


def _promote(store, rows, *, written_now=False):
    from Memory import LTM0
    if isinstance(store, LTM0) and store.owner() is not None:
        from MemoryConsolidation import consolidate
        consolidate(store.owner(), store, rows=rows, written_now=written_now)


def written_complete(store, rows):
    journal = _reads.get()
    if journal is None:
        _promote(store, rows, written_now=True)
    else:
        for row in rows:
            reads = journal.setdefault(_stream.get(), {})
            key = store,int(row)
            previous = reads.get(key,ReadEffect())
            reads[key] = ReadEffect(previous.count, True)


def trial_activity(*, commit=False):
    """Forkable read effects, committed only with the selected whole path."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            if deferred():
                return function(*args, **kwargs)
            token = _reads.set({})
            try:
                result = function(*args, **kwargs)
                selected = snapshot()
            finally:
                _reads.reset(token)
            if commit:
                apply(selected)
            return result
        return run
    return decorate


def read_stream(name='row', *, attribute=None):
    """Attribute a read to the stream whose selected path consumes it."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            stream = getattr(args[0], attribute) if attribute else kwargs.get(name, _stream.get())
            token = _stream.set(stream)
            try:
                return function(*args, **kwargs)
            finally:
                _stream.reset(token)
        return run
    return decorate
