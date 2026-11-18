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

    def positions(self, forms, spans):
        """Resolve the lesson's named occurrences in the staged presentation."""
        if len(self.parts) != len(forms):
            raise ValueError('reading lesson must match the current batch')
        result = []
        for words, parts, extents in zip(forms, self.parts, spans.tolist(), strict=True):
            words = [word.encode('utf8') if isinstance(word, str) else word for word in words]
            live = {i for i, (lo, hi) in enumerate(extents) if hi > lo}
            ordered = []
            for part in parts:
                word, ordinal = part if isinstance(part, tuple) else (part, None)
                matches = [i for i, candidate in enumerate(words) if candidate == word]
                if ordinal is not None:
                    matches = matches[ordinal:ordinal+1]
                if not matches:
                    raise ValueError(f'lesson part {part!r} is absent from the presentation')
                here = [i for i in matches if i in live]
                if not here:
                    continue  # Named in another sentence of this packed stream.
                if len(here) != 1 or here[0] in ordered:
                    raise ValueError(f'lesson part {part!r} must identify one unread native support')
                ordered.append(here[0])
            if set(ordered) != live:
                raise ValueError('reading lesson must name every word of its sentence')
            result.append(ordered)
        return result


class ReadingTargets:
    """Supervision at the two decisions actually made during a look."""
    def __init__(self, lesson, forms, state):
        self.order = lesson.positions(forms, state.spans)
        self.skipped = [[] for _ in self.order]
        self.loss = state.keys.new_zeros(len(forms))
        self.count = state.spans.new_zeros(len(forms))

    def next(self, state):
        pairs = state.poles
        both = (torch.zeros_like(state.known) if pairs is None else (pairs > 0).all(-1))
        pending = ~state.accepted & ~both
        target = []
        for b, order in enumerate(self.order):
            self.skipped[b] = [i for i in order if bool(both[b,i])]
            target.append(next((i for i in order if bool(pending[b,i])), -1))
        return state.spans.new_tensor(target)

    def focus(self, state, view, targets):
        """Shortest legal sequence to the named occurrence. Only the lesson plans."""
        from functools import lru_cache
        from Attention import focus_cuts
        table, covered, _, evidence = view[:4]
        legal = view[8]
        actions = []
        for b, target in enumerate(targets.tolist()):
            candidates = (covered[b,:,target] & legal[b,:,:2].any(-1)).nonzero() if target >= 0 else ()
            if not len(candidates):
                actions.append(-1)
                continue
            slot = int(candidates[0,0])
            ordinals = tuple(covered[b,slot].nonzero().flatten().tolist())
            @lru_cache(None)
            def shortest(words, room, allowed):
                if len(words) == 1:
                    return (1, 1) if allowed[1] else (float('inf'), -1)
                if room == 0:
                    return float('inf'), -1
                indices = state.spans.new_tensor(words)
                spans = state.spans[b,indices][None]
                lanes = evidence[b,indices][None]
                cuts = focus_cuts(torch.ones_like(spans[...,0],dtype=torch.bool), spans, lanes)
                best = (float('inf'), -1)
                for op, cut in enumerate(cuts):
                    if not allowed[op]:
                        continue
                    on_left = spans[0,:,1] <= cut
                    left = bool(on_left[words.index(target)])
                    child = tuple(i for i, keep in zip(words, on_left.tolist()) if keep == left)
                    if left and len(child) == 1 and bool((evidence[b,target]>0).sum()==1):
                        rest = 0   # grammar sees the left child on this very look
                    else:
                        both = bool((evidence[b,list(child)]>0).any(0).all())
                        rest, _ = shortest(child, room-1, (both and 0 in state.chooser.attention_operations,
                                                           1 in state.chooser.attention_operations))
                    best = min(best, (1+rest, op))
                return best
            distance, op = shortest(ordinals, int((~table.valid[b]).sum()), tuple(legal[b,slot,:2].tolist()))
            actions.append(slot*legal.shape[-1]+op if distance < float('inf') else -1)
        return state.spans.new_tensor(actions)

    def book(self, log_probability, active):
        self.loss = self.loss - torch.where(active, log_probability, 0.)
        self.count = self.count + active.long()

    def cost(self):
        return self.loss / self.count.clamp_min(1)


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
