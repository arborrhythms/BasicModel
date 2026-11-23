"""Places as rows of the first Pi (sigma-pi §19.115, step 20 of round 6,
§23.6.4): nominated by the cone over the occurrences met at a place, read
on the way up at the reading's start, passed down by their entries onto the
band, asked for by the request as any concept is."""
import os

import torch

#: The reading lesson's sentences (the trial's), minimal sentences first.
LESSON = ('This is Felix.', 'What is this?', 'That is Ciao.', 'What is that?',
          'Here is Felix.', 'Who is here?', 'There is Ciao.', 'Who is there?',
          'That is Felix.', 'What is that?', 'This is Ciao.', 'What is this?',
          'Here is Ciao.', 'Who is here?', 'There is Felix.', 'Who is there?')
#: The same words in orders the lesson never shows.
HELD_OUT = ('is This Felix?', 'Felix is here.', 'Ciao is that.', 'is that Ciao?', 'here is Felix.',
            'Felix is there.')
#: The lesson's sentences with their words in the opposite order, the
#: closing mark kept last: the same words in orders the lesson never shows.
REORDERED = tuple(dict.fromkeys(' '.join(reversed(t[:-1].split())) + t[-1] for t in LESSON))


def words_of(texts):
    return tuple(dict.fromkeys(w.strip('.?') for t in texts for w in t.split()))


def minimal():
    one = tuple(f'{w}.' for w in words_of(LESSON))
    two = tuple(dict.fromkeys(' '.join(t.split()[:2]).rstrip('.?') + '.' for t in LESSON))
    return one, two, LESSON


def model_for(tmp_path, monkeypatch, rules=None):
    """The small model of the looks' tests, with room for the lesson's words
    and its places. ``rules``: the grammar's rules available at the start
    (the key ``rulesAvailable``: names separated by spaces, empty for none);
    by default the key's own default, every rule."""
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=64, training_overrides={'reconstructionPlacement': 'eager'},
        language_overrides=None if rules is None else {'rulesAvailable': rules})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def nominate(model, texts):
    from test_item18_expect import read
    from nomination_fixtures import readings_before_nomination
    for text in texts:
        readings_before_nomination(model, lambda: read(model, text))
        read(model, text)
        model.End()


def observe_runs(monkeypatch):
    """The paths' looks (each a ``Looks``), as each reading runs them."""
    from SentenceLooks import Looks
    run, seen = Looks.run, []
    def observed(self, arrive, **kwargs):
        result = run(self, arrive, **kwargs)
        seen.append(self)
        return result
    monkeypatch.setattr(Looks, 'run', observed)
    return seen


def staged(model):
    return [w.decode() if isinstance(w, bytes) else w for w in model._attention_forms[0][0]]


def in_order(model, seen, text):
    """Whether an unforced reading of ``text`` took its words in order: the
    first path's interpreted arrivals against the sentence's live source
    slots (sigma-pi §19.124, 3). No reading lesson is installed, so no
    named target is replayed (§19.124, 2)."""
    from AttentionLesson import looks_in_order
    from test_item18_expect import read
    assert getattr(model, '_attention_output_lesson', None) is None
    assert getattr(model, '_attention_supplied_reading', None) is None
    seen.clear()
    with torch.no_grad():
        read(model, text)
    model.End()
    return all(looks_in_order(seen[0]))


def places_of(model):
    owner = model._concept_owner()
    basic = owner.basic_percepts
    from Spaces import _concept_alloc_of
    members = _concept_alloc_of(owner).order_zero_members
    concept = {percept: row for row, percept, _ in members if percept in basic.place_rows}
    return basic, concept


def test_a_place_is_nominated_by_the_cone_and_brings_a_concept(tmp_path, monkeypatch):
    """The occurrences met at a place are counted under one key; on the count
    that nominates a form, the cone over them is a row: entries on the where
    wires, and a what code (the least parts and the greatest wholes over the
    codes met) only where the occurrences there share a part. Each
    place's row has a concept of order zero; no content row and no concept
    has an entry on a where wire."""
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        basic, concept = places_of(model)
        rows = sorted(basic.place_rows)
        pure = [row for row in rows if basic.place_rows[row][2] is None]
        kept = {row: basic.place_rows[row][2] for row in rows if basic.place_rows[row][2] is not None}
        print(dict(places=len(rows), with_no_what=len(pure),
                   kept={row: what.nonzero().flatten().tolist() for row, what in kept.items()}))
        # A kept what holds a part.
        assert all(bool(what[:basic.identity_width].any()) for what in kept.values())
        assert pure and set(rows) == set(concept)
        for row in range(len(basic.index)):
            if row in basic.place_rows:
                continue
            assert not any(key[0] == 'where' for key in basic.pi.row_members(row))
        sigma = model._concept_owner().order_zero
        inputs = sigma.entry_input[:sigma.nnz].tolist()
        # The order-zero Sigma reads rows: a place's concept reads its row,
        # not a wire.
        assert set(concept) <= set(inputs)
    finally:
        model.End()


def test_a_place_concept_asked_lights_its_own_readings_band(tmp_path, monkeypatch):
    """A request for a place's concept is passed down by the row's entries:
    the share on the where wires is greatest at an occurrence of the reading
    at that place."""
    import Salience
    from test_item17_looks import held_banks
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('This is Felix.', 'That is Ciao.', 'Here is Felix.', 'Who is there?'))
        banks = held_banks(monkeypatch)
        read(model, 'This is Felix.')
        model.End()
        owner, bank = model._concept_owner(), banks[-1]
        basic, concept = places_of(model)
        perception = basic.read_bank(bank)
        start = bank.where[0, :, 0]
        checked = 0
        print(dict(start=start.tolist(), values={row: [round(v, 4) for v in perception.values[
            bank.rows.index(row), 0, :, 0].tolist()] for row in concept if row in bank.rows}))
        for row, inventory in concept.items():
            if row not in bank.rows:
                continue
            value = perception.values[bank.rows.index(row), 0, :, 0]
            k = int(value.argmax())
            if float(value[k]) <= 0.:
                continue
            request = torch.zeros(int(owner.nVectors), 1, 2)
            request[inventory, 0, 0] = 1.
            # Ported (eighth pass): the share on the where wires (``band``) is
            # gone (rule 79); the salience, by the ways back, is read instead.
            band = Salience.salience(owner, bank, request).events[0].sum(-1)
            at = int(band.argmax())
            print(dict(place=row, reading=k, value=round(float(value[k]), 4), band_at=at,
                       band_span=bank.span[0, at].tolist(), start=start[k].item(), share=round(float(band[at]), 4)))
            # The share lands on an occurrence of that reading, where the
            # row reads most in it (printed: whether that is its start).
            assert int(bank.token[0, at]) == k
            checked += 1
        assert checked
    finally:
        model.End()


def test_the_request_for_a_place_at_an_untaught_start(tmp_path, monkeypatch):
    """What ``attend`` asks of each place's concept at an untaught start
    (the same on every one), and the gradient the output's cost gives it
    after one presentation (printed; not nothing at some look). Ported
    (eleventh pass): the request started at nothing on the places and was
    taught by the reading lesson; it is now the read of rule 80, taught by
    the sentence said. At the first presentation the word said is read at
    nothing, so no gradient reaches the look; one presentation is taught
    first."""
    import SentenceLooks
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('This is Felix.', 'That is Ciao.', 'Here is Felix.', 'Who is there?'))
        basic, concept = places_of(model)
        asked = []
        read(model, 'This is Felix.', training=True, optimizer=model.getOptimizer(lr=.01))
        model.End()
        original = SentenceLooks.attend
        def requested(model, units):
            request = original(model, units)
            if request.requires_grad:
                request.retain_grad()
            asked.append(request)
            return request
        monkeypatch.setattr(SentenceLooks, 'attend', requested)
        read(model, 'This is Felix.', training=True, optimizer=model.getOptimizer(lr=0.))
        model.End()
        rows = torch.tensor(sorted(concept.values()))
        values = [request[rows, 0, 0].detach() for request in asked]
        grads = [request.grad[rows, 0] if request.grad is not None else None for request in asked]
        codes = model._concept_owner().similarity_codebook.W.detach()[rows]
        print(dict(looks=len(asked), asked=[v.tolist() for v in values],
                   gradient=[None if g is None else g.abs().sum().item() for g in grads],
                   requires_grad=[request.requires_grad for request in asked],
                   whole_gradient=[None if r.grad is None else r.grad.abs().sum().item() for r in asked],
                   codes_norm=codes.norm(dim=-1).tolist()))
        assert asked and float(values[0].max() - values[0].min()) < 1e-3
        # The where match passes a gradient onto a place's concept in the
        # request at some look, so that the request can learn to ask for it.
        assert any(g is not None and float(g.abs().sum()) > 0. for g in grads)
    finally:
        model.End()


def staged_forms(model, text):
    """The forms a reading of ``text`` stages, in order."""
    from test_item18_expect import read
    with torch.no_grad():
        read(model, text)
    model.End()
    return staged(model)


def teach(model, forms):
    """The reading taught by the sentence said (rule 80; the reading
    lesson's targets are gone, eleventh pass), minimal sentences first:
    each one-word sentence once, each two-word sentence and each of the
    lesson's twice, at a rate of .01. ``forms`` is kept for its callers."""
    from test_item18_expect import read
    one, two, lesson = minimal()
    optimizer = model.getOptimizer(lr=.01)
    for grade, presentations in ((one, len(one)), (two, 2 * len(two)), (lesson, 2 * len(lesson))):
        for p in range(presentations):
            text = grade[p % len(grade)]
            read(model, text, training=True, optimizer=optimizer)
            model.End()


def looked_at(model, monkeypatch, text):
    """At each look of a reading of ``text``: the request on the place
    concepts (its for lane, greatest and least) and the where match."""
    import SentenceLooks
    from test_item18_expect import read
    _, concept = places_of(model)
    rows = torch.tensor(sorted(concept.values()))
    asked, original_request, original_match = [], SentenceLooks.attend, SentenceLooks.where_match
    def requested(model, units):
        request = original_request(model, units)
        asked.append(dict(places=[round(float(request[rows, 0, 0].max()), 4),
                                  round(float(request[rows, 0, 0].min()), 4)],
                          each=[round(float(v), 4) for v in request[rows, 0, 0].tolist()]))
        return request
    def where_match(*args):
        match = original_match(*args)
        asked[-1]['match'] = [round(v, 4) for v in match[0].tolist()]
        return match
    monkeypatch.setattr(SentenceLooks, 'attend', requested)
    monkeypatch.setattr(SentenceLooks, 'where_match', where_match)
    with torch.no_grad():
        read(model, text)
    model.End()
    monkeypatch.setattr(SentenceLooks, 'attend', original_request)
    monkeypatch.setattr(SentenceLooks, 'where_match', original_match)
    return asked


def test_reading_in_order_before_and_after_the_lesson(tmp_path, monkeypatch):
    """A learning certificate, recorded and not asserted: over starts drawn
    afresh, the sentences read in order with no reading supplied, before and
    after the reading lesson (minimal sentences first), on the two-word
    sentences, the lesson's and orders it never shows."""
    runs = int(os.environ.get('PLACES_RUNS', '3'))
    records = []
    for run in range(runs):
        start = int.from_bytes(os.urandom(6), 'little')
        torch.manual_seed(start)
        (tmp_path / str(run)).mkdir()
        model = model_for(tmp_path / str(run), monkeypatch)
        seen = observe_runs(monkeypatch)
        try:
            one, two, lesson = minimal()
            nominate(model, one + two + lesson + HELD_OUT)
            basic, _ = places_of(model)
            forms = {text: staged_forms(model, text) for text in one + two + lesson + HELD_OUT}
            def count(texts):
                return sum(in_order(model, seen, text) for text in texts)
            before = dict(two=count(two), lesson=count(lesson), held=count(HELD_OUT))
            teach(model, forms)
            after = dict(two=count(two), lesson=count(lesson), held=count(HELD_OUT))
            asked = looked_at(model, monkeypatch, 'This is Felix.')
            _, concept = places_of(model)
            # The places' concepts' codes: their norms and the greatest
            # difference between two.
            codes = model._concept_owner().similarity_codebook.W.detach()[sorted(concept.values())]
            spread = (codes[:, None] - codes[None]).abs().amax(-1)
            band = codes[:, basic.width:basic.width + 8]
            place_codes = dict(norms=sorted(set(round(float(v), 4) for v in codes.norm(dim=-1))),
                               greatest_difference=round(float(spread.max()), 4) if len(codes) else None,
                               greatest_difference_of_where_and_when=round(float(
                                   (band[:, None] - band[None]).abs().amax(-1).max()), 4) if len(codes) else None)
            records.append(dict(start=start, places=len(basic.place_rows), place_codes=place_codes,
                                first_sentence=asked,
                                pure=sum(w is None for _, _, w in basic.place_rows.values()),
                                before=before, after=after))
            print(records[-1])
        finally:
            model.End()
            monkeypatch.undo()
    # The start seeds only weights that this measure does not read, so its
    # runs give one sample, not several.
    print(dict(of=dict(two=len(minimal()[1]), lesson=len(LESSON), held=len(HELD_OUT)),
               runs=records, samples='one: the starts seed nothing the measure reads'))
