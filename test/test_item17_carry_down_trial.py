"""Item 1.7, round 6d: the carry-down of the request (rule 79, §19.121).

The seventh pass tried it in tests, (i) by the transposes as built and (ii)
by the ways back. Ported in the eighth pass, which built (ii) in the source
(:func:`Salience.salience`, ``BasicPercepts.attended``): the transposes and
``where_match`` are gone, and the request attend gives is itself bounded
(``OperationSelectionLayer.attend``). (a) keeps both ways of the scratch
toy; (b) and (c) measure the looks as built. The tables are printed; the
assertions are only that each ran.
"""
import os
import re

import pytest
import torch


def _draw():
    seed = int.from_bytes(os.urandom(4), 'little')
    print('start', seed)
    return torch.Generator().manual_seed(seed)


# --- (a) the 304 requests of way_back_check.py (§19.52, §19.60) -------------

LESSON = ('This is Felix.', 'What is this?', 'That is Ciao.', 'What is that?', 'Here is Felix.', 'Who is here?',
          'There is Ciao.', 'Who is there?', 'That is Felix.', 'What is that?', 'This is Ciao.', 'What is this?',
          'Here is Ciao.', 'Who is here?', 'There is Felix.', 'Who is there?')
EXTRA = ('Fe li x is here.', 'xileF is here.', 'The cat is there.')


def test_a_the_304_requests():
    """(a) The scratch check of §19.52 (rule 2's rows: a word's row reads its
    own form and its letters; the word Sigma ``tanh(G z)``; the meta concept
    ``tanh(G sum c)``): the request carried down by the transposes as built
    (each Sigma by its weights, the Pi by its derivative) and by the ways
    back (the Sigmas' inverses, the meta concept's by the pseudoinverse,
    then each row's read of each occurrence, its own entries), each with the
    mask as asked and bounded (a mask of ones and nothings is its own
    bound). Asked words present and returned exactly; a word present and
    missed."""
    G, EPS = 3., 1e-6
    toks = lambda s: re.findall(r"[A-Za-z]+", s)
    vocab = sorted({w for s in LESSON for w in toks(s)} | {'the', 'cat'})
    letters = sorted({ch for s in LESSON + EXTRA for ch in s if ch.isalpha()})
    W = torch.zeros(len(vocab), len(letters) + len(vocab), dtype=torch.float64)
    for r, w in enumerate(vocab):
        for ch in w:
            W[r, letters.index(ch)] += 1.
        W[r, len(letters) + r] = 1.

    def entries(sentence):
        words = toks(sentence)
        m = torch.zeros(len(words), W.shape[1], dtype=torch.float64)
        for k, w in enumerate(words):
            for ch in w:
                m[k, letters.index(ch)] = 1.
            if w in vocab:
                m[k, len(letters) + vocab.index(w)] = 1.
        return words, m

    def up(a):
        z = (a.clamp_min(EPS).log() @ W.T).exp()
        return torch.tanh(G * z)

    def transposes(m, mask, meta):
        a = m.clone().requires_grad_()
        z = (a.clamp_min(EPS).log() @ W.T).exp()
        c = G * z                                       # the Sigma by its weights
        asked = (c * mask).sum() + meta * (G * c.sum(-1)).sum()
        g, = torch.autograd.grad(asked, a)
        return (m * g).clamp(0, 1)

    def ways_back(m, mask, meta):
        # The word Sigma is diagonal (weight G): its inverse asks each row
        # mask / G; the meta concept's Sigma (one row of G) asks every word
        # concept alike, by its pseudoinverse, 1 / (V G), and so every row.
        asked = mask / G + meta / (len(vocab) * G * G)
        salience = m * ((asked[None] @ (W > 0).double()) > 0)          # each row's read of its entries
        return m * (salience > 0)                                       # the most salient are returned

    def returned(activity, words):
        c = up(activity)
        return {(k, vocab[r]) for k in range(len(words)) for r in range(len(vocab)) if c[k, r] >= .01}

    counts = {}
    for s in LESSON + EXTRA:
        words, m = entries(s)
        for name, mask, meta in [(w, torch.eye(len(vocab), dtype=torch.float64)[vocab.index(w)], 0.) for w in vocab] + \
                [('meta', torch.zeros(len(vocab), dtype=torch.float64), 1.)]:
            want = {(k, w) for k, w in enumerate(words) if w in vocab and (name == 'meta' or w == name)}
            for how, way in (('transposes', transposes), ('ways back', ways_back)):
                for bound in ('as asked', 'bounded'):
                    # A mask of ones and nothings: bounded by shared size, each
                    # asked concept's code is wholly held; the request is unchanged.
                    got = returned(way(m, mask, meta), words)
                    key = (how, bound)
                    exact, missed = counts.get(key, (0, 0))
                    counts[key] = (exact + (got == want), missed + bool(want - got))
    print({f'{how}, {bound}': dict(returned_exactly=e, missed_a_present_word=x)
           for (how, bound), (e, x) in counts.items()})
    assert all(sum(v) for v in counts.values()) and len(counts) == 4


# --- (b) and (c): the lesson's model -----------------------------------------

@pytest.fixture(scope='module')
def lesson():
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(16)
    yield model
    model.End()
    patch.undo()


def _spans_of(text, word):
    return [(m.start(), m.end()) for m in re.finditer(r'\b%s\b' % word, text)]


def test_b_a_sentence_of_more_than_256_occurrences(lesson, monkeypatch):
    """(b) One sentence of more occurrences than a look's table holds, the
    request on the concept of the short word ``is``: how many of its
    occurrences are in the look's table, the most salient (a draw among
    equals). Ported (eighth pass): the seventh pass compared the table by
    extent first (3 of 5) with the ways back in tests (5 of 5)."""
    import SentenceLooks
    from BasicPercepts import attended
    from Salience import salience
    from test_item17_looks import held_banks, word_rows
    from test_item18_expect import read
    model = lesson
    # A sentence holds 32 readings at most (the lesson's serial word bucket),
    # so it cannot hold 256 occurrences; the table is made a quarter of the
    # sentence's occurrences instead (as certificate 63 halves it).
    text = ' '.join(['There is Felix', 'Here is Ciao', 'That is Felix', 'There is Ciao', 'Here is Felix']) + '.'
    banks = held_banks(monkeypatch)
    with torch.no_grad():
        read(model, text)
    model.End()
    owner, bank = model._concept_owner(), banks[-1]
    O = bank.token.shape[1]
    capacity = O // 4
    import BasicPercepts
    monkeypatch.setattr(BasicPercepts, 'attended_capacity', lambda: capacity)
    model._sentence_bank = bank
    concept, = word_rows(model, 'is')
    request = owner.similarity_codebook.W.new_zeros(int(owner.nVectors), 1, 2)
    request[concept, 0, 0] = 1.
    spans = bank.span[0].tolist()
    asked = [o for o, sp in enumerate(spans) if tuple(sp) in set(_spans_of(text, 'is'))]
    with torch.no_grad():
        _field, activity = SentenceLooks.look(model, bank, request, bank.activity)
        table = attended(bank, share=activity.sum(-1))
        built = set(table.index[0][table.filled[0]].tolist())
        share = salience(owner, bank, request).events.sum(-1)
    report = dict(occurrences=O, table=capacity, asked_occurrences=len(asked),
                  in_the_table=len(built & set(asked)),
                  salient_occurrences=int((share[0] > 0).sum()),
                  greatest_on_an_asked=float(share[0, asked].max()),
                  greatest_elsewhere=float(share[0][[o for o in range(O) if o not in asked]].max()))
    print(report)
    assert O > capacity and asked


def test_c_the_first_two_looks_of_this_is_felix(lesson, monkeypatch):
    """(c) The first two looks of "This is Felix." after the lesson: each
    place's request (for lane, greatest and least), each word's salience
    (the greatest over its occurrences), and the word taken, as built.
    Ported (eighth pass): the seventh pass compared the transposes with and
    without ``where_match``, the ways back, and the request bounded."""
    import SentenceLooks
    from test_item17_places import places_of
    from test_item18_expect import read
    g = _draw()
    model = lesson
    _basic, places = places_of(model)
    place_rows = torch.tensor(sorted(places.values()), dtype=torch.long)
    records = []
    original_look = SentenceLooks.look

    def look(model, bank, request, own):
        result = original_look(model, bank, request, own)
        if len(records) < 2:
            records.append(_measure(model, bank, request.detach(), own.detach(), result, place_rows, g))
        return result

    monkeypatch.setattr(SentenceLooks, 'look', look)
    with torch.no_grad():
        read(model, 'This is Felix.')
    model.End()
    for k, record in enumerate(records):
        print(f'look {k + 1}:', record)
    assert len(records) == 2


def _measure(model, bank, request, own, result, place_rows, g):
    from SentenceLooks import extent_of, word_activity
    from Salience import salience
    owner = model._concept_owner()
    concepts = model._attention_word_concepts
    W = concepts.shape[1]
    known = concepts >= 0
    places = request[place_rows.to(request.device), 0, 0]
    field, activity = result
    share = salience(owner, bank, request).events.sum(-1)
    value = word_activity(field, concepts, known, extent_of(bank, activity, W), own, bank)
    return dict(request_greatest=round(float(request.max()), 4),
                request_concepts_asked=int((request[..., 0] > 0).sum()),
                place_request=[round(float(places.max()), 4), round(float(places.min()), 4)] if len(places) else None,
                word_salience=_by_word(share, bank.token, W), word_values=_rounded(value[0]),
                taken=_taken(value[0], g))


def _by_word(share, token, W):
    return [round(float(torch.where(token[0] == w, share[0], 0.).max()), 4) for w in range(W)]


def _rounded(value):
    return [round(float(v), 4) if torch.isfinite(v) else None for v in value]


def _taken(value, g):
    """The most active word, a draw among equals; ``None`` where none."""
    finite = torch.isfinite(value)
    if not finite.any():
        return None
    best = value[finite].max()
    equal = (finite & (value == best)).nonzero().flatten()
    return int(equal[torch.randint(len(equal), (1,), generator=g)])
