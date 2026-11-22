"""The model's own reading (round 6d of item 1.7, eleventh pass; sigma-pi
§19.129, §19.130, rule 80): reading and counting, two output tasks over the
same input sentences, on one model with the same weights, taught by the
output's cost alone. Nothing names the task in the input, and no concept's
activity is set by the test. The order of teaching (§19.130): each word
alone, said back, until every word is said back; then the sentences over
those words, reading and counting together, until both are right. A stage
is left when its measure passes, or at its bound, which is reported. At
evaluation each task is given by the first word of its output supplied, or
by nothing. Records, not repairs: starts drawn afresh and printed (and
written to ``OWN_READING_OUT`` where it is set); no assertion on the outcome
of a draw or of learning."""
import json
import os

import pytest
import torch

from test_item17_places import HELD_OUT, LESSON, REORDERED, minimal, model_for, nominate, staged, words_of

#: The counting words, opaque.
COUNTING = tuple(str(k) for k in range(1, 10))
#: A sentence longer than any taught (five words and its mark; the
#: lesson's longest is three and its mark). A sentence of seven words and
#: its mark overflowed the thought store of this small model after a lesson
#: (``OverflowError`` in ``STM1.add``, a first run of this pass).
LONGER = ('Who is there is Felix.',)
#: The bounds of the two stages, in rounds over the stage's set.
WORD_BOUND, SENTENCE_BOUND = 16, 12


def fresh_start():
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    return start


def staged_of(model, text):
    from test_item18_expect import read
    with torch.no_grad():
        read(model, text)
    model.End()
    return staged(model)


def counted(model, text):
    """The counting task's output for ``text``: one counting word for each
    word the sentence stages (its closing mark counted as a word: a look
    takes it as it takes any word), in order."""
    return list(COUNTING[:len([w for w in staged_of(model, text) if w])])


class Paths:
    """The first path's looks of each reading."""
    def __init__(self, monkeypatch):
        from SentenceLooks import Looks
        run, self.seen = Looks.run, []
        def observed(path, arrive, **kwargs):
            result = run(path, arrive, **kwargs)
            self.seen.append(path)
            return result
        monkeypatch.setattr(Looks, 'run', observed)


def taught(model, optimizer, text, output=None):
    """One presentation: ``text`` read and its output said (the sentence
    itself where ``output`` is not given)."""
    from contextlib import nullcontext
    from AttentionLesson import output_lesson
    from test_item18_expect import read
    with nullcontext() if output is None else output_lesson(model, [output]):
        read(model, text, training=True, optimizer=optimizer)
    model.End()


def reading_of(model, paths, text, *, output=None, supplied=0):
    """One reading of ``text`` with nothing learned (no gradient): the
    order taken, the words said against the output's, and the stop read
    before each look. A word said is scored only where the output holds a
    word concept (the closing mark has none)."""
    from AttentionLesson import live_slots, looks_in_order, output_lesson
    from contextlib import nullcontext
    from test_item18_expect import read
    paths.seen.clear()
    given = nullcontext() if output is None else output_lesson(model, [output], supplied=[supplied])
    with torch.no_grad(), given:
        read(model, text)
    model.End()
    path = paths.seen[0]
    slots = live_slots(path.walk.spans)[0]
    orders = path.source_orders()[0]
    forms = staged(model)
    targets = path.output.targets[0] if path.output is not None else []
    said = [int(d['said'][0]) for d in path.decisions if int(d['index'][0]) >= 0]
    stops = [float(s[0]) for s in path.stops]
    n = len(slots)
    scored = [(said[t] if t < len(said) else None, w) for t, w in enumerate(targets[:n]) if w >= 0 and t >= supplied]
    finite = []
    for t, d in enumerate(path.decisions):
        unread = [w for w in slots if w not in orders[:t]]
        if unread:
            finite.append(bool(torch.isfinite(d['scores'][0, unread[0]])))
    return dict(in_order=all(looks_in_order(path)), first=bool(orders) and orders[0] == slots[0],
                said_right=all(a == b for a, b in scored), said=said, targets=targets, looks=len(said), words=n,
                stopped_early=any(s > 0 for s in stops[:n]),
                stopped_at_the_end=len(stops) > n and stops[n] > 0 and all(s <= 0 for s in stops[:n]),
                finite=finite, is_read=[w in orders for w in slots if forms[w] == 'is'],
                once=sorted(orders) == slots)


def tally(results):
    finite = [f for r in results for f in r['finite']]
    is_read = [f for r in results for f in r['is_read']]
    return dict(of=len(results), in_order=sum(r['in_order'] for r in results),
                first_word_first=sum(r['first'] for r in results),
                said_right=sum(r['said_right'] for r in results),
                stopped_at_the_end=sum(r['stopped_at_the_end'] for r in results),
                stopped_early=sum(r['stopped_early'] for r in results),
                each_word_once=sum(r['once'] for r in results),
                intended_finite=f'{sum(finite)} of {len(finite)}', is_read=f'{sum(is_read)} of {len(is_read)}')


def request_at_the_first_two_looks(model, monkeypatch, text):
    """The request (its greatest and least for lane) and its gradient at
    the first two looks of ``text`` read for the reading task at rate
    nothing (the gradient is read; nothing moves)."""
    import SentenceLooks
    from test_item18_expect import read
    asked, original = [], SentenceLooks.attend
    def requested(model, units):
        request = original(model, units)
        request.retain_grad()
        asked.append(request)
        return request
    monkeypatch.setattr(SentenceLooks, 'attend', requested)
    try:
        read(model, text, training=True, optimizer=model.getOptimizer(lr=0.))
        model.End()
    finally:
        monkeypatch.setattr(SentenceLooks, 'attend', original)
    return [dict(greatest=round(float(r.detach()[..., 0].max()), 4), least=round(float(r.detach()[..., 0].min()), 4),
                 gradient=None if r.grad is None else round(float(r.grad.abs().sum()), 6)) for r in asked[:2]]


def words_stage(model, paths, optimizer, words):
    """Each word alone (``"word."``) said back, a round at a time, until
    every word is said back at its look, or the bound. The measure after
    each round: the words said back."""
    progress = []
    for _ in range(WORD_BOUND):
        for word in words:
            taught(model, optimizer, word + '.')
        right = sum(reading_of(model, paths, word + '.')['said_right'] for word in words)
        progress.append(right)
        if right == len(words):
            break
    return dict(of=len(words), said_back_by_round=progress, passed=progress[-1] == len(words),
                rounds=len(progress), bound_hit=progress[-1] != len(words))


def sentences_stage(model, paths, optimizer, texts, counts):
    """The sentences over those words, each read for reading (the sentence
    said) and for counting (its counting words said), in turn, a round at a
    time, until every sentence is said right in both tasks with its
    output's first word supplied (with nothing supplied the two tasks have
    the same input), or the bound."""
    progress = []
    for _ in range(SENTENCE_BOUND):
        for text in texts:
            taught(model, optimizer, text)
            taught(model, optimizer, text, counts[text])
        reading = sum(reading_of(model, paths, text, output=[w for w in staged_of(model, text) if w],
                                 supplied=1)['said_right'] for text in texts)
        counting = sum(reading_of(model, paths, text, output=counts[text], supplied=1)['said_right'] for text in texts)
        progress.append(dict(reading=reading, counting=counting))
        if reading == counting == len(texts):
            break
    done = progress[-1]['reading'] == progress[-1]['counting'] == len(texts)
    return dict(of=len(texts), right_by_round=progress, passed=done, rounds=len(progress), bound_hit=not done)


@pytest.mark.parametrize('run', range(int(os.environ.get('OWN_READING_RUNS', '3'))))
def test_reading_and_counting_over_the_same_sentences_words_first(tmp_path, monkeypatch, run):
    """Recorded, not asserted, for a start drawn afresh: the two stages'
    measures round by round; then each task with nothing supplied and with
    its output's first word supplied; reading on the two-word, the lesson's,
    the held-out and the reordered sentences; counting right by number of
    words; what is said for a sentence longer than any taught; the request
    and its gradient at the first two looks, before and after. Asserted
    only that every count was made."""
    one, two, lesson_texts = minimal()
    distinct = tuple(dict.fromkeys(lesson_texts))
    sets = dict(two=two, lesson=distinct, held=HELD_OUT, reordered=REORDERED)
    start = fresh_start()
    model = model_for(tmp_path, monkeypatch)
    try:
        # The counting words are words like any other: read until MTM
        # nominates them (no concept is made or lit by the test).
        nominate(model, one + two + lesson_texts + HELD_OUT + REORDERED + LONGER
                 + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        paths = Paths(monkeypatch)
        texts = two + distinct + HELD_OUT + REORDERED + LONGER
        counts = {text: counted(model, text) for text in texts}
        record = dict(start=start, request_before=request_at_the_first_two_looks(model, monkeypatch, 'This is Felix.'))
        optimizer = model.getOptimizer(lr=.01)
        record['words'] = words_stage(model, paths, optimizer, words_of(LESSON) + COUNTING[:6])
        record['sentences'] = sentences_stage(model, paths, optimizer, two + distinct, counts)
        record['request_after'] = request_at_the_first_two_looks(model, monkeypatch, 'This is Felix.')
        for supplied in (0, 1):
            reading = {name: tally([reading_of(model, paths, text, supplied=supplied, output=None if not supplied
                                               else [w for w in staged_of(model, text) if w]) for text in group])
                       for name, group in sets.items()}
            counting = {}
            for text in two + distinct + HELD_OUT:
                r = reading_of(model, paths, text, output=counts[text], supplied=supplied)
                by = counting.setdefault(r['words'], dict(of=0, right=0, once=0))
                by['of'] += 1
                by['right'] += r['said_right']
                by['once'] += r['once']
            try:
                longer = reading_of(model, paths, LONGER[0], output=counts[LONGER[0]], supplied=supplied)
                longer = dict(said=longer['said'], counting_words=longer['targets'], once=longer['once'])
            except OverflowError as error:
                model.End()
                longer = dict(said=str(error))
            record[f'supplied_{supplied}'] = dict(reading=reading, counting=counting, longer=longer)
        print(record)
        out = os.environ.get('OWN_READING_OUT')
        if out:
            with open(out, 'a') as handle:
                handle.write(json.dumps(record) + '\n')
    finally:
        model.End()
    assert 'supplied_1' in record
