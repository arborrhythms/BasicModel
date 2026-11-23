"""Compare conditional speech with the perception-only echo on ordinary text inputs."""
import pytest

from test_item17_echo import WORDS, echo_model
from test_item17_places import model_for, nominate
from test_item17_word_said_back import command_trial


@pytest.mark.parametrize('factory,words', [(model_for, WORDS), (echo_model, ('Felix', 'Ciao'))],
                         ids=['full-grammar', 'perception-echo'])
def test_inventory(tmp_path, monkeypatch, eager_reading, factory, words):
    model = factory(tmp_path, monkeypatch)
    try:
        nominate(model, ('say',) + words)
        pairs = [(text, target) for word in words
                 for text, target in [('say ' + word, word), (word, '')]]
        answers = command_trial(model, pairs, 3)
        requested = sum(word == answer for word, answer in zip(words, answers[::2], strict=True))
        silent = answers[1::2].count('')
        print('requested words', requested, 'bare words silent', silent, 'of', len(words))
        assert len(answers) == 2 * len(words)
    finally:
        model.End()



def test_transfer_to_a_word_heard_but_never_requested(tmp_path, monkeypatch, eager_reading):
    model = model_for(tmp_path, monkeypatch)
    words = ('Felix', 'Ciao', 'This', 'That', 'Here', 'There')
    try:
        nominate(model, ('say', *words))
        pairs = [(text, expected) for word in words[:-1]
                 for text, expected in [('say ' + word, word), (word, '')]]
        probes = [*pairs, ('say ' + words[-1], words[-1]), (words[-1], '')]
        returned = command_trial(model, pairs, 6, probes=probes)
        print('taught pairs', sum(answer == target for answer, (_, target)
              in zip(returned[:-2], pairs, strict=True)), '/', len(pairs),
              'untaught pair', returned[-2:], flush=True)
        assert returned == [target for _, target in probes]
    finally:
        model.End()
