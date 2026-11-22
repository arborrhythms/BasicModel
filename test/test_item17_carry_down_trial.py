"""Item 1.7, round 6d, seventh pass, step 4: the trial of the carry-down
(rule 79, §19.121), in tests only. Nothing in the source is changed by it.

The request is carried down (i) by the transposes as built
(:func:`Salience.salience`) and (ii) by the ways back: the order-zero
Sigma's inverse (:meth:`Layers.SparseLDULayer.back`) gives what each row is
asked, and a row's salience at an occurrence is that times the row's read of
the occurrence (the first Pi's own read, :func:`BasicPercepts.held_at` and
:meth:`BasicPercepts.BasicPercepts.place_factor`), over every occurrence of
the sentence's field. Above order zero no layer has a way back yet (the
symbols' Pi and the kinds' Sigma), so a request there reaches the order-zero
concepts by the transposes of the structure above them, as built. Each way
with the request as attend gives it and bounded (a concept asked for to the
degree its code is held by the query, by shared size, on each lane:
:func:`ConceptualMatch.match`). The tables are printed; the assertions are
only that each way ran.
"""
import os
import re
from dataclasses import replace

import pytest
import torch

from Layers import signed_lanes


def _draw():
    seed = int.from_bytes(os.urandom(4), 'little')
    print('start', seed)
    return torch.Generator().manual_seed(seed)


# --- (ii) the ways back ------------------------------------------------------

def concept_request(owner, bank, request, perception):
    """Each concept's request with what reaches it from above order zero,
    ``[V, B, 2]``: the walk of :func:`Salience.salience` on the structure
    without the order-zero Sigma's and the first Pi's edges."""
    import ConceptSpread
    from Salience import _down
    readings = perception.values.shape[2]
    device = perception.values.device
    graph = ConceptSpread.structure(owner, device)
    above = graph.child < graph.percept_start
    graph = replace(graph, parent=graph.parent[above], child=graph.child[above],
                    weight=graph.weight[above], negated=graph.negated[above])
    nodes = torch.unique(torch.cat((graph.parent, graph.child, torch.arange(int(owner.nVectors), device=device))))
    values = request.new_zeros(len(nodes), request.shape[1], readings, 2)
    held = (graph.parent >= graph.concepts) & (graph.parent < graph.percept_start)
    parent, child = torch.searchsorted(nodes, graph.parent[held]), torch.searchsorted(nodes, graph.child[held])
    logs = torch.log(values.index_select(0, child).clamp_min(1e-30))
    product = torch.exp(torch.zeros_like(values).index_add(0, parent, logs))
    is_symbol = (nodes >= graph.concepts) & (nodes < graph.percept_start)
    values = torch.where(is_symbol[:, None, None, None], product, values)
    grad = _down(graph, request.to(values), values, readings, 2 * (len(owner._order_caps()) + 1), nodes=nodes)
    V = int(owner.nVectors)
    return grad.index_select(0, torch.searchsorted(nodes, torch.arange(V, device=device))).amax(2)


def rows_asked(owner, bank, concepts):
    """What each of the bank's rows is asked, ``[R, B, 2]``, by the
    order-zero Sigma's way back of each lane of the concepts' request
    ``[V, B, 2]``: a lane's result below nothing asks the row's other lane."""
    sigma = owner.order_zero
    S = sigma.slots
    R, B = len(bank.rows), concepts.shape[1]
    if not S or not R:
        return concepts.new_zeros(R, B, 2)
    n = min(len(concepts), sigma.nOutput)
    at = sigma.output_slot[:n].to(concepts.device)
    x = concepts.new_zeros(S + 1, B, 2).index_add(0, torch.where(at >= 0, at, S), concepts[:n])
    back = sigma.back(x[:S])
    back = torch.cat((back, back.new_zeros(1, B, 2)))
    rows = torch.tensor(bank.rows, dtype=torch.long, device=concepts.device)
    slot = sigma.input_slot.to(concepts.device).index_select(0, rows.clamp_max(sigma.nInput - 1))
    slot = torch.where((rows < sigma.nInput) & (slot >= 0), slot, S)
    asked = back.index_select(0, slot)                                    # [R, B, 2] signed
    f, a = signed_lanes(asked[..., 0]), signed_lanes(asked[..., 1])
    return torch.stack((f[..., 0] + a[..., 1], f[..., 1] + a[..., 0]), -1)


def rows_read(owner, bank):
    """Each row's read of every occurrence of the bank, on its two lanes,
    ``[R, B, O, 2]``: the first Pi's own read (``held_at`` on the ``.what``,
    ``place_factor`` on the ``.where`` and the ``.when``)."""
    from BasicPercepts import held_at
    basic = owner.basic_percepts
    if not bank.rows:
        return bank.table.new_zeros(0, *bank.token.shape, 2)
    part, whole, added = held_at(bank, bank.table)
    place = basic.place_factor(bank.table, bank.rows)
    conjunction = torch.tensor([r in basic.conjunctions for r in bank.rows], device=place.device)[:, None, None]
    keep = (~conjunction).to(place)
    return torch.stack((part * whole * place * keep, part * torch.maximum(1 - whole, added) * place * keep), -1)


def inverse_salience(owner, bank, request, perception):
    """Each occurrence's salience by the ways back, ``[B, O]``: the sum over
    the rows of what a row is asked times its read there, on each lane."""
    asked = rows_asked(owner, bank, concept_request(owner, bank, request, perception))
    read = rows_read(owner, bank).to(asked)
    return (asked[:, :, None, :] * read).sum((0, -1))


def most_salient(salience, capacity, g):
    """The ``capacity`` most salient occurrences of each stream, a draw
    among equals; ``[B, O]`` True where taken (nothing below or at zero)."""
    B, O = salience.shape
    order = torch.randperm(O, generator=g)
    ranked = torch.sort(salience[:, order], dim=-1, descending=True, stable=True).indices
    taken = torch.zeros(B, O, dtype=torch.bool)
    taken.scatter_(1, order[ranked[:, :capacity]], True)
    return taken & (salience > 0)


def bounded(model, query, request):
    """The request bounded: each concept asked for to the degree its code,
    on its two lanes, is held by the query's (the query's lanes exchanged
    for asked against), by shared size; the learned start kept."""
    from ConceptualMatch import match
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    held = sorted(set(_concept_alloc_of(owner).layer(0)._tensor_rows.values()))
    V = int(owner.nVectors)
    rows = torch.tensor([r for r in held if 0 <= r < V], dtype=torch.long)
    codes = owner.similarity_codebook.W.detach().index_select(0, rows.to(owner.similarity_codebook.W.device)).to(query)
    width = min(codes.shape[-1], query.shape[-1])
    c = signed_lanes(codes[:, :width]).flatten(1)                       # [rows, 2D]
    q = signed_lanes(query[:, :width].detach())                         # [B, D, 2]
    asked_for = match(q.flatten(1)[:, None], c[None])[0]                 # [B, rows]
    asked_against = match(q.flip(-1).flatten(1)[:, None], c[None])[0]
    out = request.new_zeros(V, query.shape[0], 2)
    out[rows.to(out.device)] = torch.stack((asked_for, asked_against), -1).transpose(0, 1).to(out)
    out[..., 0] += model.request_start.value.detach().to(out)[:, None]
    return out


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
    occurrences are in the table, as built (the table by extent first, the
    share within a level) and by the ways back (the most salient, a draw
    among equals)."""
    import SentenceLooks
    from BasicPercepts import attended
    from test_item17_looks import held_banks, word_rows
    from test_item18_expect import read
    g = _draw()
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
        perception = owner.basic_percepts.read_bank(bank)
        _field, activity, _match = SentenceLooks.look(model, bank, request, bank.activity, perception)
        table = attended(bank, share=activity.sum(-1))
        built = set(table.index[0][table.filled[0]].tolist())
        salience = inverse_salience(owner, bank, request, perception)
        taken = most_salient(salience, capacity, g)[0]
    report = dict(occurrences=O, table=capacity, asked_occurrences=len(asked),
                  as_built_in_the_table=len(built & set(asked)),
                  ways_back_in_the_most_salient=int(taken[asked].sum()),
                  ways_back_salient_occurrences=int((salience[0] > 0).sum()),
                  ways_back_greatest_on_an_asked=float(salience[0, asked].max()),
                  ways_back_greatest_elsewhere=float(salience[0][[o for o in range(O) if o not in asked]].max()))
    print(report, '| bounded: the request is a mask of one, its own bound')
    assert O > capacity and asked


def test_c_the_first_two_looks_of_this_is_felix(lesson, monkeypatch):
    """(c) The first two looks of "This is Felix." after the lesson: each
    place's request (for lane, greatest and least), each word's salience
    (the greatest over its occurrences), and the word taken, by the
    transposes with and without ``where_match`` and by the ways back, each
    with the request as attend gives it and bounded."""
    import SentenceLooks
    from test_item17_places import places_of
    from test_item18_expect import read
    g = _draw()
    model = lesson
    _basic, places = places_of(model)
    place_rows = torch.tensor(sorted(places.values()), dtype=torch.long)
    queries, records = [], []
    original_requested, original_look = SentenceLooks.requested, SentenceLooks.look

    def requested(model, query):
        queries.append(query.detach())
        return original_requested(model, query)

    def look(model, bank, request, own, perception):
        if len(records) < 2 and queries:
            records.append(_measure(model, bank, request.detach(), own.detach(), perception, queries[-1],
                                    place_rows, g, original_look))
        return original_look(model, bank, request, own, perception)

    monkeypatch.setattr(SentenceLooks, 'requested', requested)
    monkeypatch.setattr(SentenceLooks, 'look', look)
    with torch.no_grad():
        read(model, 'This is Felix.')
    model.End()
    for k, record in enumerate(records):
        print(f'look {k + 1}:', record)
    assert len(records) == 2


def _measure(model, bank, request, own, perception, query, place_rows, g, look):
    import SentenceLooks
    from SentenceLooks import extent_of, word_activity
    owner = model._concept_owner()
    B = own.shape[0]
    concepts = model._attention_word_concepts
    W = concepts.shape[1]
    known = concepts >= 0
    token = bank.token
    out = {}
    for bound, asked in (('as attend gives it', request), ('bounded', bounded(model, query, request))):
        places = asked[place_rows.to(asked.device), 0, 0]
        row = dict(request_greatest=round(float(asked.max()), 4), request_concepts_asked=int((asked[..., 0] > 0).sum()),
                   place_request=[round(float(places.max()), 4), round(float(places.min()), 4)] if len(places) else None)
        with torch.no_grad():
            for where in (True, False):
                original = SentenceLooks.where_match
                if not where:
                    SentenceLooks.where_match = lambda bank, band, activity, readings: torch.ones(
                        activity.shape[0], readings, device=activity.device)
                try:
                    field, activity, _match = look(model, bank, asked, own, perception)
                finally:
                    SentenceLooks.where_match = original
                from Salience import salience
                share = salience(owner, bank, asked, perception=perception).events.sum(-1)
                extent = extent_of(bank, activity, W)
                value = word_activity(field, concepts, known, extent, own, bank)
                key = 'transposes' + ('' if where else ', no where_match')
                row[key] = dict(word_salience=_by_word(share, token, W), word_values=_rounded(value[0]),
                                taken=_taken(value[0], g))
            salience = inverse_salience(owner, bank, asked, perception)
            activity = own * (salience / salience.amax(-1, keepdim=True).clamp_min(1e-30))[..., None]
            field = SentenceLooks.perceive(model, activity)
            value = word_activity(field, concepts, known, extent_of(bank, activity, W), own, bank)
            row['ways back'] = dict(word_salience=_by_word(salience, token, W), word_values=_rounded(value[0]),
                                    taken=_taken(value[0], g))
        out[bound] = row
    return out


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
