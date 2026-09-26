"""Alternate complete passes over the same sentences in one model inventory."""
from contextlib import contextmanager
from functools import wraps
import inspect
import re


def scheduled_epoch(method):
    """Keep a training continuation independent of an evaluation cursor."""
    @wraps(method)
    def run(model, *args, **kwargs):
        schedule = getattr(model, 'mode_schedule', None)
        if schedule is None or not schedule.every:
            return method(model, *args, **kwargs)
        call = inspect.signature(method).bind(model, *args, **kwargs)
        call.apply_defaults()
        evaluation = call.arguments['optimizer'] is None
        resuming = (call.arguments.get('split', 'train') == 'train'
                    and bool(getattr(model, '_resume_batches_to_skip', 0)))
        pending = schedule.pending
        if evaluation or not resuming:
            # A fresh epoch starts its own cursor. An actual mid-epoch
            # training resume keeps the unread prefix restored with it.
            schedule.pending = []
        try:
            return method(model, *args, **kwargs)
        finally:
            if evaluation:
                schedule.pending = pending
    return run


class ModeSchedule:
    def __init__(self, value):
        value = str(value).strip()
        match = re.fullmatch(r'interleave:([1-9][0-9]*)', value)
        if value not in ('serial', 'parallel') and match is None:
            raise ValueError('modeSchedule must be serial, parallel, or interleave:N (N > 0)')
        self.value = value
        self.every = int(match[1]) if match else 0
        self.pending = []
        self.completed_parallel = 0

    @property
    def serial(self):
        return self.value != 'parallel'

    def context_for(self, sentences, context=None):
        """Validate a serial prefix against its already-read native group."""
        sentences = tuple(sentences)
        if not sentences:
            raise ValueError('interleave requires complete nonempty sentence groups')
        if self.pending:
            if context is not None:
                raise ValueError('interleave context was already read for the pending group')
            group = tuple(self.pending)
        else:
            group = tuple(sentences if context is None else context)
            if not 1 <= len(group) <= self.every:
                raise ValueError('interleave context must contain at most N complete sentences')
            if context is None and len(group) != self.every:
                raise ValueError('interleave needs look-ahead: supply schedule_context for the coming group')
        if sentences != group[:len(sentences)]:
            raise ValueError('serial sentences do not match the pending interleave context')
        return None if self.pending else group

    def state_dict(self):
        return dict(version=2, value=self.value, pending=list(self.pending),
                    completed_parallel=self.completed_parallel)

    def load_state_dict(self, state):
        pending = list(state.get('pending', ()))
        if state.get('version') != 2 and pending:
            raise ValueError('cannot resume serial-first interleave state; finish that group on its original runtime')
        if pending and state.get('value') != self.value:
            raise ValueError('cannot change modeSchedule with an unfinished interleave group')
        if len(pending) > self.every or not all(isinstance(s, str) for s in pending):
            raise ValueError('invalid pending interleave sentences')
        self.pending = pending
        self.completed_parallel = int(state.get('completed_parallel', 0))

    def validate_resume(self, state):
        """A changed grouping would make saved cursor ticks mean other rows."""
        saved = (state or {}).get('value', '')
        if self.every or saved.startswith('interleave:'):
            if not state or state.get('version') != 2 or saved != self.value:
                raise ValueError('mid-epoch interleave resume requires the same parallel-first modeSchedule')

    @contextmanager
    def parallel_pass(self, model):
        """Change execution mode only; all learned owners remain in place."""
        saved = []
        def set_attr(owner, name, value):
            if owner is not None and hasattr(owner, name):
                saved.append((owner, name, getattr(owner, name)))
                object.__setattr__(owner, name, value)
        set_attr(model, 'serial', False)
        for name in ('_compiled_step', '_active_compiled_step'):
            set_attr(model, name, None)
        for name in ('reconstruct_in_loop', 'reconstruct_from_idea', 'answer_synthesis', 'output_in_loop'):
            set_attr(model, name, False)
        for owner in (*model.conceptualSpaces, *model.wholeSpaces):
            set_attr(owner, '_serial', False)
            set_attr(owner, '_serial_object_meta', False)
        set_attr(model.perceptualSpace, '_serial_object_meta', False)
        set_attr(model.inputSpace, '_serial_object_meta', False)
        try:
            yield
        finally:
            for owner, name, value in reversed(saved):
                object.__setattr__(owner, name, value)


class InterleaveCursor:
    """Stage N whole sentences, then yield serial batches without crossing groups.

    Staging reads host corpus records, including their targets and source ids.
    It does no model work. A short final group is emitted in full. Resume can
    replay this deterministic cursor while skipping completed serial batches.
    """
    def __init__(self, source, *, every, batch_size):
        self.inputs, self.outputs = source.inputs, source.outputs
        if (not isinstance(self.inputs, (list, tuple)) or not self.inputs
                or not all(isinstance(s, str) for s in self.inputs)):
            raise ValueError('interleave requires a corpus of complete text sentences')
        self.document_ids = getattr(source, 'document_ids', None)
        self.every = int(every)
        self.num_streams = min(int(batch_size), self.every, len(self.inputs))
        if min(self.every, self.num_streams) < 1:
            raise ValueError('interleave requires positive group and batch sizes')
        self.position = self.group_end = 0
        self.last_source_indices = None
        self.context_sentences = None

    def all_done(self):
        return self.position >= len(self.inputs)

    def progress(self):
        return self.position / len(self.inputs)

    def next_tick(self):
        if self.all_done():
            raise StopIteration
        self.context_sentences = None
        if self.position == self.group_end:
            self.group_end = min(self.position + self.every, len(self.inputs))
            self.context_sentences = tuple(self.inputs[self.position:self.group_end])
        end = min(self.position + self.num_streams, self.group_end)
        indices = list(range(self.position, end))
        inputs = [self.inputs[i] for i in indices]
        outputs = (None if self.outputs is None else
                   [self.outputs[i] for i in indices])
        self.position = end
        self.last_source_indices = indices
        remaining = (self.group_end - end if end < self.group_end else
                     min(self.every, len(self.inputs) - end))
        next_size = min(self.num_streams, remaining)
        hard = [self.document_ids is None or next_size != len(indices)
                or self.document_ids[i] != self.document_ids[end + b]
                for b, i in enumerate(indices)]
        return inputs, outputs, hard
