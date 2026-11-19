"""Tables of the item 1.7 looks prototype (sigma-pi §19.47, returned as §20).

Run: PYTHONPATH=bin:test .venv/bin/python test/looks_prototype_report.py
Deterministic: the random activities of the agreement check use a fixed
generator of their own; no model is built and no seed of the model is set.
"""
import torch

from looks_prototype import (LESSON, GAIN, Bank, Request, fallback, interpret, lesson_layer, look,
                             operations, order_trial, place_profile, projection, pullback_by_backward,
                             pullback_by_transposes, where_code)

CAPACITY = 4096
CONCENTRATION = 200.


def one_hot(layer, word):
    return torch.tensor([float(w == word) for w in layer.vocabulary], dtype=torch.float64)


def returned(layer, bank, out, floor=.5):
    """The word concepts at or above ``floor``, token by token."""
    words = []
    for k in range(len(bank.spans)):
        at = [layer.vocabulary[w] for w in range(len(layer.vocabulary)) if float(out['word'][k, w]) >= floor]
        if at:
            words.append(f"{k}:{'/'.join(at)}")
    return ' '.join(words) or '(none)'


def projected(bank, values):
    lit = sorted({int(bank.token[p]) for p in range(len(bank.text)) if float(values[p]) > 0})
    return ' '.join(f'{k}:{bank.words[k]}' for k in lit) or '(none)'


def place_request(layer, bank, token, concentration=CONCENTRATION):
    layer.places = []
    index = layer.add_place(bank.start_code[token], concentration)
    place = torch.zeros(index + 1, dtype=torch.float64)
    place[index] = 1.
    return place


def agreement(layer, codebook):
    generator = torch.Generator().manual_seed(0)
    print('### Pullback: transposes against the framework backward\n')
    print('| Sigma passed by | comparisons | largest difference |\n| --- | ---: | ---: |')
    for sigma in ('slope', 'weights'):
        worst, count = 0., 0
        for text in LESSON:
            bank = Bank.read(text, codebook, capacity=CAPACITY)
            live = (bank.token >= 0).to(torch.float64)
            draws = [live] + [torch.rand(len(text), generator=generator, dtype=torch.float64) * live
                              for _ in range(3)]
            requests = [Request(meta=1.), Request(whole=1.), Request(word=one_hot(layer, bank.words[-2])),
                        Request(word=torch.rand(len(layer.vocabulary), generator=generator,
                                                dtype=torch.float64), whole=.5, meta=.3)]
            for activity in draws:
                for request in requests:
                    a = pullback_by_transposes(layer, bank, activity, request, sigma=sigma)
                    b = pullback_by_backward(layer, bank, activity, request, sigma=sigma)
                    worst = max(worst, float((a - b).abs().max()))
                    count += 1
        print(f'| its {sigma} | {count} | {worst:.1e} |')
    print()


VARIANTS = (
    ('meta', 'slope', 'percept'), ('meta', 'weights', 'percept'),
    ('word', 'weights', 'percept'), ('whole', 'weights', 'percept'),
    ('absent', 'weights', 'percept'),
    ('place', 'weights', 'percept'), ('place', 'weights', 'token'),
    ('meta+place', 'weights', 'percept'), ('meta+place', 'weights', 'token'))


def requests_table(layer, codebook):
    print('### Five requests on the sixteen sentences, and the meta concept with a place\n')
    print(f'Place rows: capacity {CAPACITY}, concentration {CONCENTRATION:g}, applied inside the products.\n')
    print('| Request | Sigma passed by | Place read at | Word meant | Taken as meant, first of equals | Taken as meant, largest of equals | Taken with a tie | Returned on "This is Felix."; taken | Projection on "This is Felix." |')
    print('| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |')
    for kind, sigma, at in VARIANTS:
        meant_ok, largest_ok, tied, example = 0, 0, 0, None
        for text in LESSON:
            bank = Bank.read(text, codebook, capacity=CAPACITY)
            layer.places = []
            if kind == 'meta':
                request, meant = Request(meta=1.), None
            elif kind == 'word':
                meant = bank.words[-2]
                request = Request(word=one_hot(layer, meant))
            elif kind == 'whole':
                request, meant = Request(whole=1.), None
            elif kind == 'absent':
                request, meant = Request(word=one_hot(layer, 'dog')), '(nothing)'
            elif kind == 'place':
                request, meant = Request(place=place_request(layer, bank, 1)), bank.words[1]
            else:
                request = Request(meta=1., place=place_request(layer, bank, 2))
                meant = bank.words[2]
            out = look(layer, bank, request, sigma=sigma, at=at)
            word, token, value, ties = interpret(layer, bank, out)
            large, large_token, _, _ = interpret(layer, bank, out, ties='largest')
            if meant == '(nothing)':
                meant_ok += word is None
                largest_ok += large is None
            elif meant is not None:
                meant_ok += word == meant and bank.words[token] == meant
                largest_ok += large == meant and bank.words[large_token] == meant
            tied += bool(ties)
            if text == 'This is Felix.':
                proj = projection(layer, bank, request) if request.asks_what else out['where']
                example = (returned(layer, bank, out), projected(bank, proj),
                           f'{word}@{token}' if word else '(none)', ties)
        meant_text = {'meta': 'any word', 'word': 'the last word', 'whole': 'any word',
                      'absent': '`dog`: nothing', 'place': 'the word at token 1',
                      'meta+place': 'the word at token 2'}[kind]
        ok = f'{meant_ok} of 16' if kind not in ('meta', 'whole') else '—'
        large = f'{largest_ok} of 16' if kind not in ('meta', 'whole') else '—'
        print(f'| {kind} | its {sigma} | {at} | {meant_text} | {ok} | {large} | {tied} of 16 | {example[0]}; {example[2]} | {example[1]} |')
    print()


def binding_table(layer, codebook):
    print('### What the binding by token prevents\n')
    print('| Sentence | Request | Returned | Taken |\n| --- | --- | --- | --- |')
    for text, word in (('Fe li x is here.', 'Felix'), ('xileF is here.', 'Felix'),
                       ('This is Felix.', 'is'), ('Felix is here.', 'Felix')):
        bank = Bank.read(text, codebook, capacity=CAPACITY)
        layer.places = []
        out = look(layer, bank, Request(word=one_hot(layer, word)))
        w, k, value, ties = interpret(layer, bank, out)
        taken = f'{w}@{k}' + (f' (tied with {ties})' if ties else '') if w else '(none)'
        print(f'| "{text}" | {word} | {returned(layer, bank, out)} | {taken} |')
    print()


def cost_table(layer, codebook):
    ups, backs, words = [], [], []
    for text in LESSON:
        bank = Bank.read(text, codebook, capacity=CAPACITY)
        up, back = operations(layer, bank)
        ups.append(up)
        backs.append(back)
    mean = lambda v: sum(v) / len(v)
    print('### Cost, multiply-adds per sentence (mean over the sixteen)\n')
    print('| One way up | The pullback | One look (pullback and the way up again) |')
    print('| ---: | ---: | ---: |')
    print(f'| {mean(ups):.0f} | {mean(backs):.0f} | {mean(backs) + mean(ups):.0f} |\n')


def dimming_table(layer, codebook):
    text = 'This is Felix.'
    print(f'### Dimming by the asking row: the meta concept with a place at token 2, on "{text}"\n')
    print('| Concentration | Read at | Applied | This@0 | is@1 | Felix@2 | Taken |')
    print('| ---: | --- | --- | ---: | ---: | ---: | --- |')
    bank = Bank.read(text, codebook, capacity=CAPACITY)
    for concentration in (12.5, 50., 200., 800., 3200.):
        for at in ('percept', 'token'):
            for dim in ('inside', 'once'):
                request = Request(meta=1., place=place_request(layer, bank, 2, concentration))
                out = look(layer, bank, request, dim=dim, at=at)
                best = out['word'].max(-1).values.tolist()
                w, k, value, ties = interpret(layer, bank, out)
                taken = f'{w}@{k}' if w else '(none)'
                print(f'| {concentration:g} | {at} | {dim} | {best[0]:.3f} | {best[1]:.3f} | {best[2]:.3f} | {taken} |')
    print()


def sinusoid_table():
    offsets = (1, 2, 4, 8, 16, 64, 128, 255, 256, 257, 512, 1024, 4096)
    print('### The where sinusoid as a coordinate: a place row\'s reading at a distance, over its reading at the place\n')
    for capacity in (4096, 64):
        print(f'Capacity {capacity}; concentration {CONCENTRATION:g}.\n')
        print('| Rung | ' + ' | '.join(f'+{d}' for d in offsets) + ' |')
        print('| --- |' + ' ---: |' * len(offsets))
        for rungs in ('long', 'short', 'both'):
            values = place_profile(200, offsets, capacity=capacity, rungs=rungs,
                                   concentration=CONCENTRATION)
            print(f'| {rungs} | ' + ' | '.join(f'{v:.3f}' for v in values.tolist()) + ' |')
        print()
    from Spaces import WhereEncoding
    for capacity in (4096, 64):
        w = WhereEncoding(nWhere=4).set_capacity(capacity)
        print(f'At capacity {capacity} the long rung\'s period is {w.maxVal} bytes and the short rung\'s {w.period_hf:g}.')
    print()


def fallback_table(layer, codebook):
    print('### The fallback: a .where given outright, then the largest word in it\n')
    print('| Sentence | Extent | Taken |\n| --- | --- | --- |')
    for text, lo, hi in (('This is Felix.', 0, 4), ('This is Felix.', 0, 7), ('This is Felix.', 8, 14),
                         ('This is Felix.', 13, 14), ('This is Felix.', 0, 14), ('Who is here?', 4, 12),
                         ('Fe li x is here.', 0, 6)):
        bank = Bank.read(text, codebook, capacity=CAPACITY)
        kind, value, k = fallback(layer, bank, lo, hi)
        taken = f'{kind} `{value}` at token {k}' if kind else '(nothing)'
        print(f'| "{text}" | [{lo}, {hi}) "{text[lo:hi]}" | {taken} |')
    print()


def main():
    layer, codebook = lesson_layer(words=('dog',), texts=('Fe li x is here.', 'xileF'))
    agreement(layer, codebook)
    requests_table(layer, codebook)
    binding_table(layer, codebook)
    cost_table(layer, codebook)
    dimming_table(layer, codebook)
    sinusoid_table()
    fallback_table(layer, codebook)
    print('### Reading in order: the place of the next look, taught on the request alone\n')
    print('| Steps | Offset before | Offset after | Sentences read wholly in order, before | after |')
    print('| ---: | ---: | ---: | ---: | ---: |')
    for steps in (0, 20, 150):
        result = order_trial(layer, codebook, steps=steps)
        print(f"| {steps} | {result['offset_before']:.2f} | {result['offset_after']:.3f} | "
              f"{result['in_order_before']} of 16 | {result['in_order_after']} of 16 |")
    print()


if __name__ == '__main__':
    main()
