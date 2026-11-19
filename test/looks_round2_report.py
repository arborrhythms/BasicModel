"""Tables of the trial of the looks (sigma-pi §22.2, returned as §23.2).

Run: PYTHONPATH=bin:test .venv/bin/python test/looks_round2_report.py
Everything but step 6 is deterministic. The agreement check draws its
random activities from a generator of its own, seeded 0. Step 6 draws each
start's seed from the system and prints it, so a start can be run again.
"""
import os
import time

import torch

from looks_prototype import LESSON
from looks_round2 import (EXTRA_FORMS, ICE, SENTENCES, WORDS, Bank, Request, Trial, WhereConcept,
                          concept_mask, interpret, lexicon, look, present, returned, salience,
                          salience_by_transposes, teach_order, count_in_order)

GARBAGE = ('qzv Felix xwpt.', 'qzFelixvb.')
TWICE = 'Felix is Felix.'
# Sentences of known forms in orders the lesson never shows, for step 6.
HELD_OUT = ('Felix is here.', 'Ciao is that cat.', 'Who is Felix?', 'The cat is Felix.',
            'Here is The cat.', 'is This Felix?')


def make(where_of='combined'):
    lex = lexicon(texts=SENTENCES + (ICE, TWICE) + GARBAGE + HELD_OUT)
    trial = Trial(lex)
    trial.where_of = where_of
    return trial


def step1(trial):
    lex = trial.lex
    print('### Step 1: the bank\n')
    print(f'Rows: {lex.R} ({len(lex.part_names)} parts, {len(lex.forms)} forms). '
          f'Parts added to tell forms apart: {", ".join(repr(p) for p in lex.added) or "none"}.\n')
    print('| Sentence | Tokens (form or unknown) | Events |\n| --- | --- | ---: |')
    for text in SENTENCES + (ICE,):
        bank = Bank.read(text, lex)
        toks = ' '.join(f'`{w}`' + ('' if f is not None else '?') for w, f in zip(bank.words, bank.form))
        print(f'| "{text}" | {toks} | {len(bank.row)} |')
    print()


def agreement():
    print('### Certificate 31 (a): the transposes against the framework backward\n')
    print('| Where of a reading | Sigma passed by | Comparisons | Largest difference |')
    print('| --- | --- | ---: | ---: |')
    for where_of in ('combined', 'head', 'start'):
        trial = make(where_of)
        lex = trial.lex
        generator = torch.Generator().manual_seed(0)
        for mode in ('slope', 'weights'):
            worst, count = 0., 0
            for text in SENTENCES + (ICE,):
                bank = Bank.read(text, lex)
                for activity in (bank.live(), torch.rand(len(bank.row), generator=generator,
                                                         dtype=torch.float64)):
                    for request in (Request(words=1.), Request(concepts=concept_mask(lex, 'is')),
                                    Request(words=1., where=(WhereConcept('sigma', trial.code[5], 100.),)),
                                    Request(words=.5, where=(WhereConcept('pi', trial.code[5], 200.),))):
                        a1, b1, _ = salience(trial, bank, activity, request, mode=mode)
                        a2, b2, _ = salience_by_transposes(trial, bank, activity, request, mode=mode)
                        worst = max(worst, float((a1 - a2).detach().abs().max()), float((b1 - b2).detach().abs().max()))
                        count += 1
            print(f'| {where_of} | {mode} | {count} | {worst:.1e} |')
    print()


def step3(trial):
    lex = trial.lex
    print('### Step 3: the 304 requests\n')
    print('| The request carried down by | Returns exactly the asked words present | Misses a present word | Returns one not asked |')
    print('| --- | ---: | ---: | ---: |')
    names = {'slope': 'the plain gradient (each Sigma by its slope)',
             'weights': 'the gradient, each Sigma before its squash',
             'entries': 'the rows\' entries alone'}
    for mode in ('slope', 'weights', 'entries'):
        exact = miss = extra = n = 0
        for text in SENTENCES:
            bank = Bank.read(text, lex)
            for asked in WORDS + ('words',):
                request = (Request(words=1.) if asked == 'words' else
                           Request(concepts=concept_mask(lex, asked)))
                got = returned(trial, bank, look(trial, bank, request, mode=mode))
                want = present(trial, bank, lex.forms if asked == 'words' else (asked,))
                n += 1
                exact += got == want
                miss += bool(want - got)
                extra += bool(got - want)
        print(f'| {names[mode]} | {exact} of {n} | {miss} | {extra} |')
    print()


def graded(lex, name):
    if name == 'the same word twice, at 1 and at .4':
        bank = Bank.read(TWICE, lex)
        a = bank.live()
        a[bank.token == 2] = .4
        return bank, a
    bank = Bank.read('Here is Felix.', lex)
    a = bank.live()
    events = (bank.token == 2).nonzero().flatten().tolist()
    chars = [e for e in events if e != int(bank.head[2])]
    if name == 'the whole word at .3':
        a[events] = .3
    elif name == 'three of its five parts at .5':
        a[chars[:3]] = .5
    else:
        a[chars] = torch.linspace(1., .2, len(chars), dtype=torch.float64)
    return bank, a


def step4(trial):
    lex = trial.lex
    print('### Step 4: parts present by degree\n')
    print('Values are the most active word concept at each token; "open" is the open run. '
          '"times" is rule 34 (own activity times the share capped at one); "min" is the share '
          'capped at the event\'s own activity.\n')
    print('| Case | Request | Way | Rule | Open run | Returned | Taken |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for name in ('the whole word at .3', 'three of its five parts at .5',
                 'its parts falling from 1 to .2 along it', 'the same word twice, at 1 and at .4'):
        bank, a = graded(lex, name)
        opened = trial.up(bank, a)['cw'][:, lex.word_rows].max(-1).values
        open_text = ' '.join(f'{v:.3f}' for v in opened.tolist())
        for req_name, request in (('Felix', Request(concepts=concept_mask(lex, 'Felix'))),
                                  ('words', Request(words=1.))):
            for mode in ('slope', 'weights', 'entries'):
                for rule in ('times', 'min'):
                    if mode == 'slope' and rule == 'min':
                        continue
                    out = look(trial, bank, request, activity=a, mode=mode, what_rule=rule)
                    vals = ' '.join(f'{v:.3f}' for v in out['word'].max(-1).values.tolist())
                    got = interpret(trial, bank, out)
                    taken = (f'{got.form}@{got.token}' + (' (tie)' if got.tied else '')) if got.form else '-'
                    print(f'| {name} | {req_name} | {mode} | {rule} | {open_text} | {vals} | {taken} |')
    print()


KS = {'sigma': (1., 10., 100., 1e3, 1e4), 'pi': (12.5, 50., 200., 800., 3200.)}


def step5():
    print('### Step 5: the where match\n')
    print('"words" with a where concept at each word\'s start, on the 16 sentences '
          '(48 places). Each cell: sentences wholly right, and places right.\n')
    print('| Where of a reading | Way | Form | ' + ' | '.join(f'k{i + 1}' for i in range(5)) + ' |')
    print('| --- | --- | --- |' + ' ---: |' * 5)
    for where_of in ('combined', 'head', 'start'):
        trial = make(where_of)
        banks = [Bank.read(t, trial.lex) for t in LESSON]
        for way in ('sigma', 'pi'):
            for form in ('a', 'b', 'c'):
                cells = []
                for k in KS[way]:
                    sentences = places = 0
                    for bank in banks:
                        whole = True
                        for t in bank.word_tokens():
                            spec = WhereConcept(way, trial.code[bank.spans[t][0]], k)
                            got = interpret(trial, bank, look(trial, bank, Request(words=1., where=(spec,)), form=form))
                            ok = got.identified and got.token == t
                            places += ok
                            whole &= ok
                        sentences += whole
                    cells.append(f'k={k:g}: {sentences}, {places}/48')
                print(f'| {where_of} | {way} | {form} | ' + ' | '.join(cells) + ' |')
    print()
    trial = make('combined')
    banks = [Bank.read(t, trial.lex) for t in LESSON]
    print('"words" alone, nothing of where asked: sentences on which every word is returned '
          'at the same value as in the open run, none lowered.\n')
    print('| Form | Sentences |\n| --- | ---: |')
    for form in ('a', 'b', 'c'):
        n = 0
        for bank in banks:
            out = look(trial, bank, Request(words=1.), form=form)
            values = out['word'].max(-1).values[bank.word_tokens()]
            opened = trial.up(bank, bank.live())['cw'][:, trial.lex.word_rows].max(-1).values[bank.word_tokens()]
            n += bool(torch.allclose(values, opened))
        print(f'| ({form}) | {n} of 16 |')
    print()
    text = 'This is Felix.'
    print(f'Growth: "words" with a where concept at `Felix` (place 8) on "{text}"; the most active '
          'word concept at each token. "once": the match on a token\'s reading; "inside": on each '
          'event at its own place.\n')
    print('| Where of a reading | Way | Form | Applied | k | This | is | Felix | Taken |')
    print('| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |')
    for where_of, way, form in (('head', 'sigma', 'a'), ('combined', 'sigma', 'a'),
                                ('head', 'sigma', 'c'), ('head', 'pi', 'a'), ('head', 'pi', 'c')):
        trial = make(where_of)
        bank = Bank.read(text, trial.lex)
        for apply in ('once', 'inside'):
            for k in KS[way]:
                spec = WhereConcept(way, trial.code[8], k)
                out = look(trial, bank, Request(words=1., where=(spec,)), form=form, apply=apply)
                v = out['word'].max(-1).values.tolist()
                got = interpret(trial, bank, out)
                taken = (f'{got.form}' + (' (tie)' if got.tied else '')) if got.form else '-'
                print(f'| {where_of} | {way} | {form} | {apply} | {k:g} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {taken} |')
    print()


def step6(starts=5, stages=(10, 10, 60)):
    print('### Step 6: reading in order, with no location concept made\n')
    print(f'Stages: one-word sentences, two-word sentences, the lesson; {stages} steps; Adam, '
          'rate 0.05, full batch; the request map starts at "words" alone with noise of 0.01 on '
          'its other entries; the where weights start at zero. In order: of the lesson\'s 16, '
          'every word taken in order, each identified (no tie).\n')
    print('| Where of a reading | Learned | Start (seed) | Curve: (stage, step, in order) | Seconds | Held out in order | With the where match off | Where weights: largest norms |')
    print('| --- | --- | --- | --- | ---: | ---: | ---: | --- |')
    for where_of in ('combined', 'head', 'start'):
        for learned in (True, False):
            for _ in range(starts):
                seed = int.from_bytes(os.urandom(4), 'little')
                trial = make(where_of)
                generator = torch.Generator().manual_seed(seed)
                t0 = time.time()
                curve, reader, Uw = teach_order(trial, stages=stages, generator=generator,
                                                where_learned=learned, every=5)
                seconds = time.time() - t0
                held = [Bank.read(t, trial.lex) for t in HELD_OUT]
                lesson = [Bank.read(t, trial.lex) for t in LESSON]
                held_n = count_in_order(trial, reader, held, Uw=Uw)
                off = count_in_order(trial, reader, lesson, Uw=Uw, form='none')
                norms = Uw.norm(dim=-1)
                top = norms.argsort(descending=True)[:4]
                words_mean = float(norms[trial.lex.word_rows].mean())
                tops = ', '.join(f'`{trial.lex.names[i]}` {float(norms[i]):.2f}' for i in top)
                short = ' '.join(f'{s}.{k}:{n}' for s, k, n in curve if s == 'start' or k % 10 == 0 or k <= 5)
                what = 'request and where' if learned else 'request alone'
                print(f'| {where_of} | {what} | {seed} | {short} | {seconds:.1f} | {held_n} of {len(held)} | '
                      f'{off} of 16 | {tops}; word concepts {words_mean:.2f} |')
    print()


def step8():
    print('### Step 8: certificates 34 and 35 on this bank\n')
    for where_of in ('combined', 'head', 'start'):
        trial = make(where_of)
        lex = trial.lex
        bank = Bank.read(ICE, lex)
        print(f'Where of a reading: {where_of}. "{ICE}", tokens {bank.words}.\n')
        print('| Request | This | is | ice | ice cream | cream | Taken |')
        print('| --- | ---: | ---: | ---: | ---: | ---: | --- |')
        at8 = (WhereConcept('pi', trial.code[8], 200.),)
        for name, request, form in (('"words"', Request(words=1.), 'a'),
                                    ('none', Request(), 'a'),
                                    ('a place at 8 (Pi row, form c)', Request(where=at8), 'c'),
                                    ('"words" and a place at 8 (Pi row, form c)', Request(words=1., where=at8), 'c')):
            out = look(trial, bank, request, form=form)
            v = out['word'].max(-1).values.tolist()
            got = interpret(trial, bank, out)
            taken = (f'{got.form}@{got.token}' + (f' tied with {got.tied}' if got.tied else '')) if got.form else '-'
            print(f'| {name} | ' + ' | '.join(f'{x:.3f}' for x in v[:5]) + f' | {taken} |')
        print()
        held = 0
        for text in LESSON:
            b = Bank.read(text, lex)
            out = look(trial, b, Request(words=1.))
            held += bool(((out['what'] > 0) & (b.row == lex.space)).any())
        print(f'On the lesson\'s 16, sentences with a part holding a space selected by "words": {held}.\n')
        print('| Garbage | Tokens | Returned by "words" | Taken |\n| --- | --- | --- | --- |')
        for text in GARBAGE:
            b = Bank.read(text, lex)
            out = look(trial, b, Request(words=1.))
            got = interpret(trial, b, out)
            print(f'| "{text}" | {b.words} | {sorted(returned(trial, b, out))} | {got.form or "-"} |')
        print()


def main():
    trial = make('combined')
    step1(trial)
    agreement()
    step3(trial)
    step4(trial)
    step5()
    step6()
    step8()


if __name__ == '__main__':
    main()
