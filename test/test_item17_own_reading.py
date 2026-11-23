"""Measure reading and counting through the returned answer, with the task in the input."""
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


def taught(model, optimizer, text):
    """One ordinary unsupervised sentence presentation for the root trace."""
    from test_item18_expect import read
    try:
        read(model, text, training=True, optimizer=optimizer)
    finally:
        model.End()


@pytest.mark.parametrize('run', range(int(os.environ.get('OWN_READING_RUNS', '3'))))
def test_reading_and_counting_over_the_same_sentences_words_first(tmp_path, monkeypatch, run):
    from test_item17_echo import read_once
    one, two, lesson = minimal()
    texts = tuple(dict.fromkeys(two + lesson))
    start = fresh_start()
    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, one + two + lesson + HELD_OUT + REORDERED + LONGER + COUNTING + ('say', 'count'))
        groups = dict(lesson=texts, held=HELD_OUT, reordered=REORDERED, longer=LONGER)
        words = words_of(LESSON) + COUNTING[:6]
        single = [('say ' + w, w) for w in words]

        def tasks(texts):
            return [(cue + ' ' + text, text if cue == 'say' else ' '.join(counted(model, text)))
                    for text in texts for cue in ('say', 'count')]

        paired = {name: tasks(group) for name, group in groups.items()}
        samples = dict(single + [pair for group in paired.values() for pair in group])
        inputs = {text: i for i, text in enumerate(samples)}
        data = model.inputSpace.data
        monkeypatch.setattr(data, 'has_supervised_outputs', True)
        monkeypatch.setattr(data, 'text_answers', {'train': list(samples.values()), 'validation': []})
        optimizer = model.getOptimizer()

        def present(pairs, train=False):
            return [read_once(model, inputs[text], text, answered=True, train=train, optimizer=optimizer)
                    for text, target in pairs]

        record = dict(start=start)
        for stage, pairs, bound in [('words', single, WORD_BOUND), ('sentences', paired['lesson'], SENTENCE_BOUND)]:
            curve = []
            for _ in range(bound):
                present(pairs, train=True)
                right = sum(actual == target for actual, (_, target) in zip(present(pairs), pairs, strict=True))
                curve.append(right)
                if right == len(pairs):
                    break
            record[stage] = dict(of=len(pairs), right_by_round=curve)
        for name, pairs in paired.items():
            record[name] = list(zip(pairs, present(pairs), strict=True))
        print(record)
        out = os.environ.get('OWN_READING_OUT')
        if out:
            with open(out, 'a') as handle:
                handle.write(json.dumps(record) + '\n')
        assert all(len(record[name]) == len(rows) for name, rows in paired.items())
    finally:
        model.End()
