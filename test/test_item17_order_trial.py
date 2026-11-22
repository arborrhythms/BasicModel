"""Round 6d of item 1.7, ninth pass, steps 2 and 3 (sigma-pi §19.124): a
record of the request and its gradient at the first two looks after the
reading lesson, and a trial of Alec's cue for the next look. Records, not
repairs: nothing here changes a source under bin/, and nothing is tuned."""
import os

import torch

from test_item17_places import (HELD_OUT, LESSON, minimal, model_for, nominate, observe_runs, places_of,
                                staged_forms, teach)


def _round(values, digits=4):
    return [round(float(v), digits) for v in values]


def test_the_request_and_its_gradient_at_the_first_two_looks_after_the_lesson(tmp_path, monkeypatch):
    """Step 2, a record (Codex, §19.124, 1): after the reading lesson, at the
    first two looks of "This is Felix." read with its reading named (rate
    nothing, so the gradient is read and nothing moves): the query's
    greatest coordinate against the codes', how many of the query's
    coordinates on two lanes stand above a code's (where the match is flat),
    each place concept's request, and the gradient at the request and at
    the query. Asserted: two looks were recorded."""
    import SentenceLooks
    from AttentionLesson import reading_lesson
    from Layers import signed_lanes
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        forms = {text: staged_forms(model, text) for text in one + two + lesson}
        teach(model, forms)
        _, concept = places_of(model)
        places = sorted(concept.values())
        asked = []
        original = SentenceLooks.requested
        def requested(model, query):
            if query.requires_grad:
                query.retain_grad()
            request = original(model, query)
            if request.requires_grad:
                request.retain_grad()
            asked.append((query, request))
            return request
        monkeypatch.setattr(SentenceLooks, 'requested', requested)
        optimizer = model.getOptimizer(lr=0.)
        text = 'This is Felix.'
        with reading_lesson(model, [forms[text]]):
            read(model, text, training=True, optimizer=optimizer)
        model.End()
        from Spaces import _concept_alloc_of
        owner = model._concept_owner()
        held = sorted(set(_concept_alloc_of(owner).layer(0)._tensor_rows.values()))
        codes = owner.similarity_codebook.W.detach()[held]
        code_lanes = signed_lanes(codes)                                           # [V', D, 2]
        looks = []
        for query, request in asked[:2]:
            q = query.detach()[0]
            width = min(len(q), codes.shape[-1])
            q_lanes = signed_lanes(torch.nn.functional.pad(q[:width], (0, codes.shape[-1] - width)))
            # Where a query lane stands above a code's (and the code holds
            # something there), the match's minimum is the code's: flat.
            above = (q_lanes[None] > code_lanes) & (code_lanes > 0)
            looks.append(dict(
                query_greatest=round(float(q.abs().max()), 4),
                codes_greatest=round(float(codes.abs().max()), 4),
                query_lanes_above_a_code=f'{int(above.sum())} of {int((code_lanes > 0).sum())}',
                place_request_for=_round(request.detach()[places, 0, 0]),
                place_request_against=_round(request.detach()[places, 0, 1]),
                request_gradient=None if request.grad is None else round(float(request.grad.abs().sum()), 6),
                place_request_gradient=None if request.grad is None else round(
                    float(request.grad[places].abs().sum()), 6),
                query_gradient=None if query.grad is None else round(float(query.grad.abs().sum()), 6),
                query_requires_grad=bool(query.requires_grad)))
        moved = None
        if len(asked) >= 2:
            (q0, r0), (q1, r1) = asked[:2]
            moved = dict(query=round(float((q1.detach() - q0.detach()).abs().max()), 4),
                         place_request=round(float((r1.detach()[places] - r0.detach()[places]).abs().max()), 6))
        print(dict(start=start, places=len(places), looks=len(asked), first_two=looks,
                   change_from_look_one_to_two=moved))
        assert len(asked) >= 2
    finally:
        model.End()


#: The lesson's sentences with their words in the opposite order, the
#: closing mark kept last: the same words in orders the lesson never shows.
REORDERED = tuple(dict.fromkeys(' '.join(reversed(t[:-1].split())) + t[-1] for t in LESSON))
#: The cue's sharpness among a sentence's unread words, as the reading
#: lesson's (``ReadingTargets.SHARPNESS``).
SHARPNESS = 10.


def where_codes(bank, width):
    """Each word slot's ``.where``, ``[B, width, 4]``: the where wires of its
    reading's start, as the bank's registry gives them."""
    return bank.registry.band(bank.where[..., 0])[:, :width]


def nearness(codes, cue):
    """How near each word's ``.where`` ``[B, W, 4]`` is to the cue's ``[B,
    4]``, by shared size on two lanes, as ``place_factor`` reads a row's
    entries (``field_factor``), ``[B, W]``."""
    from BasicPercepts import field_factor
    lanes = torch.stack((codes, 1 - codes), -1)
    entries = torch.stack((cue, 1 - cue), -1).clamp(0., 1.)
    return torch.stack([field_factor(lanes[b:b + 1], entries[b:b + 1])[0, 0] for b in range(len(cue))])


class Cue:
    """Alec's cue, in its simplest form (§19.124), held by the trial and not
    by the model: the request is "words" (the request's learned start; no
    context map), and the cue carries the ``.where`` of the word just read
    plus a learned ``offset`` on the ``.where``; each unread word the look
    found is scored by how near its ``.where`` is to the cue's. The first
    look has no cue: it takes what "words" lights, a draw among equals."""
    def __init__(self):
        self.offset = torch.zeros(4, requires_grad=True)
        self.path = None

    def install(self, monkeypatch):
        import SentenceLooks
        from SentenceLooks import Looks
        cue = self
        original_request, original_activity, run = (SentenceLooks.requested, SentenceLooks.word_activity,
                                                    Looks.run)
        def requested(model, query):
            return original_request(model, torch.zeros_like(query))
        def word_activity(field, concepts, known, extent, own, bank):
            words = original_activity(field, concepts, known, extent, own, bank)
            last = cue.last_read(len(words))
            if last is None:
                return words
            codes = where_codes(bank, words.shape[1]).to(words)
            at = codes[torch.arange(len(words)), last.clamp_min(0)] + cue.offset.detach().to(words)
            near = nearness(codes, at)
            return torch.where(torch.isfinite(words) & (last[:, None] >= 0), near, words)
        def observed(path, arrive, **kwargs):
            cue.path = path
            try:
                return run(path, arrive, **kwargs)
            finally:
                cue.path = None
        monkeypatch.setattr(SentenceLooks, 'requested', requested)
        monkeypatch.setattr(SentenceLooks, 'word_activity', word_activity)
        monkeypatch.setattr(Looks, 'run', observed)

    def last_read(self, B):
        """The word each row read last, ``[B]`` (-1 where none), or None
        before any word is read."""
        if self.path is None or not self.path.decisions:
            return None
        last = torch.full((B,), -1, dtype=torch.long)
        for turn in self.path.decisions:
            last = torch.where(turn['interpreted'].cpu() & (turn['index'].cpu() >= 0), turn['index'].cpu(), last)
        return None if bool((last < 0).all()) else last

    def cost(self, sentences):
        """The cost of the next word over trained data: for each sentence's
        ``.where`` codes ``[slots, 4]`` read in order, at each word but the
        last, the next slot against the slots still unread."""
        loss, count = self.offset.new_zeros(()), 0
        for codes in sentences:
            for i in range(len(codes) - 1):
                near = nearness(codes[None, i + 1:], (codes[i] + self.offset)[None])[0]
                loss = loss - (SHARPNESS * near).log_softmax(-1)[0]
                count += 1
        return loss / max(count, 1)


def test_the_cue_for_the_next_look_a_trial(tmp_path, monkeypatch):
    """Step 3, a trial recorded and not asserted (§19.124, Alec's cue): the
    sentences read in order (step 1's measure) with the request "words" and
    the cue, before and after the cue's offset is learned on the two-word
    and the lesson's sentences read in order, on the two-word, lesson,
    held-out and reordered sentences; beside each count, the readings whose
    first look took the first word, and the looks after the first that took
    the next word to the right of the last one read. Asserted only that
    every sentence was read."""
    from AttentionLesson import live_slots, looks_in_order
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        sets = dict(two=two, lesson=lesson, held=HELD_OUT, reordered=REORDERED)
        nominate(model, one + two + lesson + HELD_OUT + REORDERED)
        seen = observe_runs(monkeypatch)
        cue = Cue()
        cue.install(monkeypatch)
        import SentenceLooks
        banks = []
        original_activity = SentenceLooks.word_activity
        def captured(field, concepts, known, extent, own, bank):
            banks.append((bank, concepts.shape[1]))
            return original_activity(field, concepts, known, extent, own, bank)
        monkeypatch.setattr(SentenceLooks, 'word_activity', captured)

        def reading(text):
            seen.clear()
            banks.clear()
            with torch.no_grad():
                read(model, text)
            model.End()
            path = seen[0]
            slots = live_slots(path.walk.spans)[0]
            orders = path.source_orders()[0]
            return path, slots, orders

        def count(texts):
            ordered = first = nexts = after_first = 0
            for text in texts:
                path, slots, orders = reading(text)
                ordered += all(looks_in_order(path))
                first += bool(orders) and orders[0] == slots[0]
                for a, b in zip(orders, orders[1:]):
                    right = [s for s in slots if s > a and s not in orders[:orders.index(b)]]
                    nexts += bool(right) and b == right[0]
                    after_first += bool(right)
            return dict(in_order=ordered, first_word_first=first, next_to_the_right=f'{nexts} of {after_first}')

        before = {name: count(texts) for name, texts in sets.items()}
        # The trained data: the two-word and the lesson's sentences, each
        # word's .where in the order of its slots.
        trained = []
        for text in two + lesson:
            _, slots, _ = reading(text)
            bank, width = banks[0]
            codes = where_codes(bank, width)[0].detach().float()
            trained.append(codes[slots])
        optimizer = torch.optim.Adam([cue.offset], lr=.01)
        first = cue.cost(trained)
        first.backward()
        cost_before, gradient_before = float(first), float(cue.offset.grad.abs().sum())
        for _ in range(200):
            optimizer.zero_grad()
            loss = cue.cost(trained)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            cost_after = float(cue.cost(trained))
        after = {name: count(texts) for name, texts in sets.items()}
        codes = torch.cat(trained)
        print(dict(start=start, of={name: len(texts) for name, texts in sets.items()},
                   before=before, after=after, offset=_round(cue.offset.detach(), 5),
                   cost=dict(before=round(cost_before, 5), after=round(cost_after, 5)),
                   offset_gradient_at_the_start=gradient_before,
                   where_code_range=dict(least=_round(codes.amin(0)), greatest=_round(codes.amax(0)))))
        assert all(sum(1 for _ in texts) for texts in sets.values())
    finally:
        model.End()
