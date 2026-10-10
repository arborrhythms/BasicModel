"""Named input parts teach reconstruction and, explicitly, reading order."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PartLesson:
    need: torch.Tensor
    targets: tuple

    def cost(self, record, decoder):
        leaves, count = decoder[:2]
        B, T, D = leaves.shape
        if len(self.targets) != B:
            raise ValueError('part targets must match the current batch')
        bank = record.primed
        targets = torch.zeros_like(leaves)
        length = torch.zeros_like(count)
        for b, words in enumerate(self.targets):
            if len(words) > T:
                raise ValueError('part target exceeds the native decoder allowance')
            for w, raw in enumerate(words):
                value = torch.tensor(list(raw), dtype=bank.bytes.dtype, device=leaves.device)
                width = bank.bytes.shape[-1]
                if len(value) > width:
                    raise ValueError('part target is absent from the sentence dictionary')
                wanted = F.pad(value, (0, width - len(value)))
                match = bank.valid[b] & ((bank.bytes[b] == wanted) | ~bank.byte_valid[b]).all(-1)
                # The native bank masks surface bytes; its padded NUL is not
                # part of that mask. Length distinguishes a prefix word.
                match &= bank.byte_valid[b].sum(-1).eq(len(value))
                if int(match.sum()) != 1:
                    raise ValueError('part target must identify exactly one native word')
                targets[b, w] = bank.codes[b, match.nonzero()[0, 0]].detach()
            length[b] = len(words)
        active = torch.arange(T, device=leaves.device)[None] < torch.maximum(length, count)[:, None]
        error = torch.where(active[..., None], (leaves - targets).abs(), 0.).sum((1, 2))
        baseline = targets.abs().sum((1, 2))
        return error, baseline


@dataclass(frozen=True)
class ReadingLesson:
    """The corpus names successive word parts, never scorer features.

    Resolve a named word against the unread native support only at the
    lesson boundary. Repeated spellings must be named with an occurrence
    (word, ordinal) in the lesson, rather than guessed by the reader.
    """
    parts: tuple

    def target(self, field):
        forms = field.model._attention_forms[0]
        target = torch.zeros_like(field.remaining)
        if len(self.parts) != len(target):
            raise ValueError('reading lesson must match the current batch')
        for b in range(len(target)):
            if not bool(field.active[b]):
                continue
            turn = int(field.iterations[b])
            if turn >= len(self.parts[b]):
                raise ValueError('reading lesson ended before the field was read')
            part = self.parts[b][turn]
            word, occurrence = part if isinstance(part, tuple) else (part, None)
            columns = [i for i, position in enumerate(field.positions.tolist())
                       if (forms[b][position].encode('utf8') if isinstance(forms[b][position], str)
                           else forms[b][position]) == word and bool(field.valid[b, i])]
            if occurrence is not None:
                columns = columns[occurrence:occurrence+1]
            columns = [i for i in columns if bool(field.remaining[b, i])]
            if len(columns) != 1:
                available = [forms[b][position] for i, position in enumerate(field.positions.tolist())
                             if bool(field.remaining[b, i])]
                raise ValueError(f'lesson part {part!r} must identify one unread native support; available={available!r}')
            target[b, columns[0]] = True
        return target


@contextmanager
def reading_lesson(model, parts):
    """Teach the supplied part sequence during this training invocation only."""
    def encode(part):
        if isinstance(part, tuple):
            word, occurrence = part
            if type(occurrence) is not int or occurrence < 0:
                raise ValueError('a named occurrence must be a nonnegative integer')
            return (word.encode('utf8') if isinstance(word, str) else word, occurrence)
        return part.encode('utf8') if isinstance(part, str) else part
    lesson = ReadingLesson(tuple(tuple(encode(part) for part in row) for row in parts))
    old = getattr(model, '_attention_reading_lesson', None)
    model._attention_reading_lesson = lesson
    try:
        yield lesson
    finally:
        model._attention_reading_lesson = old


def _prompt_need(model, prompts):
    prompts = tuple(prompts)
    if not prompts or any(not prompt.split() for prompt in prompts):
        raise ValueError('a prompt need requires nonempty presented questions')
    identity = model.perceptualSpace.percept_store.identity
    like = model.conceptualSpace.stm._buffer
    width = int(model.conceptualSpace.stm.concept_dim)
    # Use the existing fixed native form keys. This encodes the question's
    # words, without parsing an ordinal or looking up its requested answer.
    forms = [torch.stack([identity.form(word.encode('utf8')) for word in prompt.split()]).mean(0)
             for prompt in prompts]
    return F.pad(torch.stack(forms).to(like), (0, width - identity.width))


@contextmanager
def prompt_need(model, prompts):
    """Supply the existing prompt encoding without a part target or loss."""
    need = _prompt_need(model, prompts)
    old = getattr(model, '_attention_prompt_need', None)
    model._attention_prompt_need = need
    try:
        yield need
    finally:
        model._attention_prompt_need = old


@contextmanager
def part_lesson(model, prompts, targets):
    """Supply natural prompts and desired word strings for one invocation."""
    prompts, targets = tuple(prompts), tuple(targets)
    if len(prompts) != len(targets) or not prompts:
        raise ValueError('part prompts and targets must have the same nonempty batch')
    need = _prompt_need(model, prompts)
    lesson = PartLesson(need, tuple(tuple(word.encode('utf8') for word in target.split())
                                   for target in targets))
    old = getattr(model, '_attention_part_lesson', None)
    model._attention_part_lesson = lesson
    try:
        yield lesson
    finally:
        model._attention_part_lesson = old
