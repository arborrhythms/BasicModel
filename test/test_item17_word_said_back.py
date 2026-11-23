"""Conditional speech through the public answer of the lesson model with its full grammar."""

import torch

from test_item17_echo import answer_text
from What import What
from test_item17_places import model_for, nominate

PAIRS = (('say Felix', 'Felix'), ('Ciao', ''), ('Felix', ''), ('say Ciao', 'Ciao'))


def test_space_is_a_native_percept(tmp_path, monkeypatch, eager_reading):
    from SentenceLooks import Looks
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    spaces = []
    run = Looks.run

    def observed(path, arrive):
        field = model.perceptualSpace._forward_input
        ids, spans = field['native_indices'][0], field['native_part_spans'][0]
        store = field['percept_store']
        spaces.append([(lo, hi) for pid, (lo, hi) in zip(ids.tolist(), spans.tolist(), strict=True)
                       if pid >= 0 and store.bytes_for(pid) == b' '])
        return run(path, arrive)

    monkeypatch.setattr(Looks, 'run', observed)
    try:
        read(model, 'say  Felix')
        assert spaces and all(row == [(3, 4), (4, 5)] for row in spaces), spaces
    finally:
        model.End()


def command_trial(model, pairs, rounds):
    """Teach paired texts and evaluate with one common, target-free inference question."""
    data = model.inputSpace.data
    data.has_supervised_outputs = True
    data.text_answers = {'train': [], 'validation': []}
    optimizer = model.getOptimizer()
    source = model.inputSpace
    question = What.inference(0)
    teacher = What.supervised(0)

    def present(text, question, train):
        with torch.set_grad_enabled(train):
            batch, _ = model.runBatch(train=train, optimizer=optimizer, batchSize=1, split='train',
                questions=(question,), batch_override=(source.prepInput([text]), torch.empty(1, 0)))
        return batch

    answers = []
    for epoch in range(rounds + 1):
        if epoch:
            for text, target in pairs:
                data.text_answers['train'] = [target]
                present(text, teacher, True)
                model.End()
        if epoch in (0, rounds):
            answers = []
            data.text_answers['train'] = []
            for text, _ in pairs:
                batch = present(text, question, False)
                answer = answer_text(batch.outputPred)
                derivation = model._last_answer_construction.derivation
                print(epoch, repr(text), repr(answer), derivation.row_sources,
                      [len(row) for row in derivation.answer_meanings], flush=True)
                answers.append(answer)
                model.End()
    return answers


def test_the_answer_depends_on_the_spoken_request(tmp_path, monkeypatch, eager_reading):
    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('say', 'Felix', 'Ciao') + tuple(text for text, _ in PAIRS))
        assert command_trial(model, PAIRS, 6) == [target for _, target in PAIRS]
    finally:
        model.End()
