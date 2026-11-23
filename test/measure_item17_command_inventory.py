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
