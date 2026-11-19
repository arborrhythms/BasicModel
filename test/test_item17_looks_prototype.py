"""Item 1.7 prototype of the looks (sigma-pi §19.47): its claims as tests.

Tests only; nothing of bin/ is used but ``Spaces.WhereEncoding``. The
tables of §20 come from ``looks_prototype_report.py``.
"""
import pytest
import torch

from looks_prototype import (LESSON, Bank, Request, fallback, interpret, lesson_layer, look,
                             place_profile, projection, pullback_by_backward,
                             pullback_by_transposes)


@pytest.fixture(scope='module')
def lesson():
    return lesson_layer(words=('dog',), texts=('Fe li x is here.', 'xileF is here.'))


def one_hot(layer, word):
    return torch.tensor([float(w == word) for w in layer.vocabulary], dtype=torch.float64)


def place_at(layer, bank, token, concentration=200.):
    layer.places = []
    layer.add_place(bank.start_code[token], concentration)
    return torch.ones(1, dtype=torch.float64)


@pytest.mark.parametrize('sigma', ['slope', 'weights'])
def test_the_pullback_by_transposes_is_the_framework_backward(lesson, sigma):
    layer, codebook = lesson
    generator = torch.Generator().manual_seed(0)
    for text in LESSON:
        bank = Bank.read(text, codebook)
        live = (bank.token >= 0).to(torch.float64)
        for activity in (live, torch.rand(len(text), generator=generator, dtype=torch.float64) * live):
            for request in (Request(meta=1.), Request(whole=1.),
                            Request(word=one_hot(layer, bank.words[-2]), whole=.5, meta=.3)):
                torch.testing.assert_close(
                    pullback_by_transposes(layer, bank, activity, request, sigma=sigma),
                    pullback_by_backward(layer, bank, activity, request, sigma=sigma),
                    rtol=0, atol=1e-12)


def test_a_word_comes_back_at_its_token_and_an_absent_word_not_at_all(lesson):
    layer, codebook = lesson
    bank = Bank.read('This is Felix.', codebook)
    out = look(layer, bank, Request(word=one_hot(layer, 'Felix')))
    assert interpret(layer, bank, out)[:2] == ('Felix', 2)
    assert float(out['word'][:2].max()) < .5
    other = Bank.read('Who is here?', codebook)
    absent = Request(word=one_hot(layer, 'dog'))
    assert interpret(layer, other, look(layer, other, absent))[0] is None
    # The weights alone let a word through on one shared letter ('o' of Who).
    assert bool(projection(layer, other, absent)[:3].gt(0).any())


def test_the_binding_by_token_stops_scattered_letters_and_not_an_anagram(lesson):
    layer, codebook = lesson
    request = Request(word=one_hot(layer, 'Felix'))
    scattered = Bank.read('Fe li x is here.', codebook)
    assert interpret(layer, scattered, look(layer, scattered, request))[0] is None
    anagram = Bank.read('xileF is here.', codebook)
    assert interpret(layer, anagram, look(layer, anagram, request))[:2] == ('Felix', 0)


def test_the_meta_concept_with_a_place_takes_the_word_at_the_place(lesson):
    layer, codebook = lesson
    for text in LESSON:
        bank = Bank.read(text, codebook)
        for token in range(len(bank.spans) - 1):
            request = Request(meta=1., place=place_at(layer, bank, token))
            out = look(layer, bank, request, at='token')
            word, at, _, _ = interpret(layer, bank, out, ties='largest')
            assert (word, at) == (bank.words[token], token), (text, token)


def test_a_saturated_sigma_gives_its_strongest_token_almost_no_share(lesson):
    layer, codebook = lesson
    bank = Bank.read('This is Felix.', codebook)
    live = (bank.token >= 0).to(torch.float64)
    share = pullback_by_transposes(layer, bank, live, Request(meta=1.), sigma='slope')
    by_token = [float(share[bank.token == k].sum()) for k in range(3)]
    # Token 0 holds two whole words (This, and is inside it), so the meta
    # concept is most saturated there and its slope gives it least.
    assert by_token[0] < 1e-2 * max(by_token)  # about 1 to 300
    weights = pullback_by_transposes(layer, bank, live, Request(meta=1.), sigma='weights')
    assert all(float(weights[bank.token == k].min()) >= 1 for k in range(3))


def test_the_short_rung_alone_lets_in_a_place_one_period_away():
    near, far = place_profile(200, (4, 256), capacity=4096, rungs='short', concentration=200.).tolist()
    assert near < .2 and far > .99
    both = place_profile(200, (4, 256), capacity=4096, rungs='both', concentration=200.).tolist()
    assert both[1] < .01


def test_the_fallback_takes_the_largest_word_in_a_given_extent(lesson):
    layer, codebook = lesson
    bank = Bank.read('This is Felix.', codebook)
    assert fallback(layer, bank, 0, 7) == ('word', 'This', 0)
    assert fallback(layer, bank, 13, 14) == (None, None, None)


def test_a_taught_place_reads_the_lesson_in_order(lesson):
    from looks_prototype import order_trial
    layer, codebook = lesson
    result = order_trial(layer, codebook, steps=60)
    assert result['in_order_before'] == 0 and result['in_order_after'] == 16
    assert .5 < result['offset_after'] < 1.5
