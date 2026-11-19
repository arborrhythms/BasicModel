"""The trial of the looks (sigma-pi §22.2): the claims of steps 1, 3, 5 and 8.

Tests only; nothing of bin/ is used but ``Spaces.WhereEncoding``. The tables
and the learning results are printed by ``looks_round2_report.py``.
"""
import pytest
import torch

from looks_prototype import LESSON
from looks_round2 import (CAPACITY, ICE, SENTENCES, WORDS, Bank, Reader, Request, Trial, WhereConcept,
                          concept_mask, interpret, lexicon, look, present, read_in_order, returned,
                          salience, salience_by_transposes, where_code)

GARBAGE = 'qzv Felix xwpt.'


def make(where_of='combined'):
    trial = Trial(lexicon(texts=SENTENCES + (ICE, GARBAGE)))
    trial.where_of = where_of
    return trial


@pytest.fixture(scope='module')
def trial():
    return make()


def test_a_forms_code_is_its_characters_and_the_parts_that_tell_forms_apart(trial):
    lex = trial.lex
    codes = list(lex.codes.values())
    assert len(set(codes)) == len(codes)
    assert lex.added == ('ac',)                       # `cat` and `act` have the same letters
    assert lex.codes['cat'] == {'c', 'a', 't'} and 'ac' in lex.codes['act']
    assert ' ' in lex.codes['ice cream']
    assert lex.codes['ice'] <= lex.codes['ice cream'] and lex.codes['cream'] <= lex.codes['ice cream']
    # No end marks: `is` has one code, alone and inside `This`.
    assert lex.codes['is'] <= set('This')
    bank = Bank.read(ICE, lex)
    assert bank.words == ['This', 'is', 'ice', 'ice cream', 'cream', '.']
    assert bank.spans[3][0] == bank.spans[2][0]       # a form of two tokens is an event at its start


@pytest.mark.parametrize('where_of', ['combined', 'head', 'start'])
@pytest.mark.parametrize('mode', ['slope', 'weights'])
def test_the_salience_map_by_the_transposes_is_the_framework_backward(where_of, mode):
    trial = make(where_of)
    lex = trial.lex
    generator = torch.Generator().manual_seed(0)
    for text in SENTENCES + (ICE,):
        bank = Bank.read(text, lex)
        for activity in (bank.live(), torch.rand(len(bank.row), generator=generator, dtype=torch.float64)):
            for request in (Request(words=1.), Request(concepts=concept_mask(lex, 'is')),
                            Request(words=1., where=(WhereConcept('sigma', trial.code[5], 100.),)),
                            Request(words=.5, where=(WhereConcept('pi', trial.code[5], 200.),))):
                what, where, _ = salience(trial, bank, activity, request, mode=mode)
                what_t, where_t, _ = salience_by_transposes(trial, bank, activity, request, mode=mode)
                torch.testing.assert_close(what_t, what, rtol=0, atol=1e-6)
                torch.testing.assert_close(where_t, where, rtol=0, atol=1e-6)


@pytest.mark.parametrize('mode,expected', [('slope', 229), ('weights', 304), ('entries', 304)])
def test_the_304_requests_return_exactly_the_asked_words_present(trial, mode, expected):
    lex, exact = trial.lex, 0
    for text in SENTENCES:
        bank = Bank.read(text, lex)
        for asked in WORDS + ('words',):
            request = Request(words=1.) if asked == 'words' else Request(concepts=concept_mask(lex, asked))
            got = returned(trial, bank, look(trial, bank, request, mode=mode))
            exact += got == present(trial, bank, lex.forms if asked == 'words' else (asked,))
    assert exact == expected


def test_a_word_comes_back_at_its_token_and_an_absent_word_not_at_all(trial):
    lex = trial.lex
    bank = Bank.read('This is Felix.', lex)
    got = interpret(trial, bank, look(trial, bank, Request(concepts=concept_mask(lex, 'Felix'))))
    assert (got.form, got.token) == ('Felix', 2) and got.identified
    other = Bank.read('Who is here?', lex)
    assert interpret(trial, other, look(trial, other, Request(concepts=concept_mask(lex, 'Felix')))).form is None
    for text in ('Fe li x is here.', 'xileF is here.'):
        scattered = Bank.read(text, lex)
        assert not any(f == 'Felix' for _, f in returned(trial, scattered, look(trial, scattered, Request(words=1.))))


def test_the_where_parameter_is_the_code_takes_a_gradient_and_is_in_no_optimizer():
    trial = make('head')
    code = where_code(torch.arange(CAPACITY), CAPACITY).to(torch.float64)
    assert torch.equal(trial.theta.detach(), code)
    reader = Reader(trial)
    Uw = torch.zeros_like(trial.Uw).requires_grad_()
    with torch.no_grad():
        Uw[trial.lex.word_rows, 3] = -1.
    optimizer = torch.optim.Adam([reader.A, Uw], lr=.05)
    bank = Bank.read(LESSON[0], trial.lex)
    loss, _ = read_in_order(trial, reader, bank, train=True, Uw=Uw)
    loss.backward()
    optimizer.step()
    assert torch.equal(trial.theta.detach(), code)
    assert trial.theta.grad is not None and bool(trial.theta.grad.ne(0).any())
    assert all(p is not trial.theta for group in optimizer.param_groups for p in group['params'])


def test_a_gradient_from_an_asked_concept_that_reads_the_where_reaches_its_band():
    trial = make('head')
    lex = trial.lex
    bank = Bank.read('This is Felix.', lex)
    request = Request(concepts=concept_mask(lex, 'is'))
    Uw = torch.zeros_like(trial.Uw)
    _, where, _ = salience(trial, bank, bank.live(), request, Uw=Uw)
    assert float(where.abs().max()) == 0.
    Uw[lex.index['is']] = torch.tensor([0., 0., 1., -1.], dtype=torch.float64)
    _, where, _ = salience(trial, bank, bank.live(), request, Uw=Uw)
    assert float(where[1, lex.index['is']].abs()) > 0


@pytest.mark.parametrize('where_of,way,form,k', [('head', 'sigma', 'a', 10.), ('start', 'sigma', 'a', 1000.),
                                                 ('combined', 'pi', 'c', 200.)])
def test_words_with_a_where_concept_returns_the_word_at_the_place(where_of, way, form, k):
    trial = make(where_of)
    for text in LESSON:
        bank = Bank.read(text, trial.lex)
        for t in bank.word_tokens():
            spec = WhereConcept(way, trial.code[bank.spans[t][0]], k)
            got = interpret(trial, bank, look(trial, bank, Request(words=1., where=(spec,)), form=form))
            assert got.identified and got.token == t, (text, t)


def test_the_where_share_of_a_peaked_pi_row_is_nothing_at_its_own_place():
    trial = make('head')
    bank = Bank.read('This is Felix.', trial.lex)
    spec = WhereConcept('pi', trial.code[8], 200.)
    _, where, _ = salience(trial, bank, bank.live(), Request(where=(spec,)))
    assert abs(float(where[2, bank.head_rows[2]])) < 1e-9


@pytest.mark.parametrize('form', ['a', 'c'])
def test_words_alone_returns_every_word_and_lowers_none(trial, form):
    for text in LESSON:
        bank = Bank.read(text, trial.lex)
        out = look(trial, bank, Request(words=1.), form=form)
        opened = trial.up(bank, bank.live())['cw'][:, trial.lex.word_rows]
        torch.testing.assert_close(out['word'].max(-1).values, opened.max(-1).values)
        assert returned(trial, bank, out) == present(trial, bank, trial.lex.forms)


def test_words_presses_against_the_space(trial):
    lex = trial.lex
    bank = Bank.read(ICE, lex)
    out = look(trial, bank, Request(words=1.))
    value = out['word'].max(-1).values
    assert float(value[3]) < 1e-3 < float(min(value[2], value[4]))      # `ice cream` below `ice`, `cream`
    assert float(out['what'][bank.row == lex.space].max()) == 0.
    at8 = (WhereConcept('pi', trial.code[8], 200.),)
    got = interpret(trial, bank, look(trial, bank, Request(where=at8), form='c'))
    assert (got.form, got.identified) == ('ice cream', True)            # the larger form fits
    got = interpret(trial, bank, look(trial, bank, Request(words=1., where=at8), form='c'))
    assert (got.form, got.identified) == ('ice', True)
    for text in LESSON:
        b = Bank.read(text, lex)
        assert not bool(((look(trial, b, Request(words=1.))['what'] > 0) & (b.row == lex.space)).any())


def test_a_word_among_garbage_letters_is_returned_alone(trial):
    bank = Bank.read(GARBAGE, trial.lex)
    out = look(trial, bank, Request(words=1.))
    assert returned(trial, bank, out) == {(1, 'Felix')}
    got = interpret(trial, bank, out)
    assert (got.form, got.token, got.identified) == ('Felix', 1, True)
