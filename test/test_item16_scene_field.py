"""Item 1.6, step C: the image read with its sentence, put back and charged.

The certificates C1 to C5 of §32.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured, on the text model of the echo test with the scene's box
(``test_item16_scene``'s), fed by ``felt_world_corpus.rows``.

At a sentence's opening a row's image has a bank of its own; its rows,
read by numerical parthood, and the sentence's, read by the first Pi over
forms, stand in one table of lanes that goes to the one Not
(``ImagePercepts``). The image's readings are the places its pixels hold
and the bracket over the box. The image is put back from the opened field
alone and charged in the opening's errors as
``reconstruction_order_zero.image``.
"""
import copy
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch


def scene_model(tmp_path, monkeypatch):
    """The echo test's text model with the scene's box declared."""
    from test_item17_echo import echo_model
    from test_item16_scene import _declared_shape
    return echo_model(tmp_path, monkeypatch, architecture_overrides={"sceneShape": _declared_shape()})


@pytest.fixture
def model(tmp_path, monkeypatch):
    from reading_fixtures import use_eager_reading
    m = scene_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    yield m
    m.End()


class Opening:
    """Reads, never writes: at each opening of the first sentence, the
    opened field, the opening's errors (their values by name), what the
    image's charge put back, and the text the sentence's reconstruction
    put back. ``during``: called inside the opening with this record."""
    def __init__(self, monkeypatch, during=None):
        import ImagePercepts
        import SentenceField
        from test_item17_echo import text
        self.seen = []
        charge, one = ImagePercepts.charge, SentenceField.reconstruction

        def reconstructed(model, field, **kw):
            result = one(model, field, **kw)
            self.text = text(result.materialize()[0])
            return result

        def charged(model, errors, sid, field, active):
            back = charge(model, errors, sid, field, active)
            if int(sid) == 0:
                record = dict(field=field, back=None if back is None else back.detach(),
                              errors={name: errors._value(term).detach() for name, term in errors._terms.items()},
                              terms=errors._terms, text=self.text)
                if during is not None:
                    during(model, errors, field, record)
                self.seen.append(record)
            return back
        monkeypatch.setattr(SentenceField, 'reconstruction', reconstructed)
        monkeypatch.setattr(ImagePercepts, 'charge', charged)

    @property
    def last(self):
        return self.seen[-1]


def present(model, rows, *, train=True):
    """One reading of ``rows``, a batch of one stream for each."""
    optimizer = model.getOptimizer()
    try:
        with torch.set_grad_enabled(train):
            source = model.inputSpace.prepInput(list(rows))
            model.runBatch(train=train, optimizer=optimizer, batchSize=len(rows),
                           batch_override=(source, torch.empty(len(rows), 0)))
    finally:
        model.End()


def known(model, rows):
    """Each row presented as often as nominates a form, and once more."""
    for _ in range(model._concept_owner().concept_mint_count + 1):
        present(model, rows)


def scene_range(model):
    return model.where_registry.slices['scene']


def value_and_place_concepts(model):
    """The concept ids of order zero that read a value's row, and those
    that read a place's row of the box."""
    import ImagePercepts
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    basic = owner.basic_percepts
    image = set(ImagePercepts.image_rows(model))
    values, places = set(), set()
    for concept, percept, negated in _concept_alloc_of(owner).order_zero_members:
        cid = owner.concept_id_at_row(concept)
        if negated or cid is None or percept not in image:
            continue
        (values if basic.is_value(percept) else places if percept in basic.place_rows else set()).add(int(cid))
    return values, places


def category(number, split='train'):
    import felt_world_corpus as felt
    return [felt.rows(document) for document in felt.load(number, split)]


def the_felt(model, scene):
    """A scene over the box, ``[places]``: each pixel's value at its place."""
    import ImagePercepts
    saved = model.inputSpace._scenes
    model.inputSpace._scenes = [model.inputSpace.scene_pixels(scene)]
    try:
        image, _ = ImagePercepts.felt(model, 'cpu', torch.float32)
    finally:
        model.inputSpace._scenes = saved
    return image[0]


# -- C1: text as it was -------------------------------------------------------------

def test_c1_text_as_it_was(tmp_path, monkeypatch):
    """Two models built alike and given the same draws, each step taken by
    one and then by the other from the same state of the generator."""
    from test_item17_places import nominate
    from reading_fixtures import use_eager_reading
    sentence = 'this is eight.'
    for name in ('one', 'two'):
        (tmp_path / name).mkdir()
    state = torch.get_rng_state()
    one = scene_model(tmp_path / 'one', monkeypatch)
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    two = scene_model(tmp_path / 'two', monkeypatch)
    use_eager_reading(monkeypatch)
    observed = Opening(monkeypatch)
    try:
        assert torch.equal(after, torch.get_rng_state())
        state = torch.get_rng_state()
        nominate(one, (sentence,))
        torch.set_rng_state(state)
        nominate(two, (sentence,))
        state = torch.get_rng_state()
        present(one, [sentence], train=False)
        plain = observed.last
        torch.set_rng_state(state)
        present(two, [dict(text=sentence, scene=None)], train=False)
        nulled = observed.last
    finally:
        one.End()
        two.End()
    a, b = plain['field'], nulled['field']
    same_field = all(torch.equal(getattr(a, name), getattr(b, name))
                     for name in ('concept_ids', 'evidence', 'where', 'when'))
    same_wires = (a.wires is None and b.wires is None) or torch.equal(a.wires, b.wires)
    same_errors = (plain['errors'].keys() == nulled['errors'].keys()
                   and all(torch.equal(plain['errors'][k], nulled['errors'][k]) for k in plain['errors']))
    print(f"C1 {sentence!r} as a string and as a row with scene null: field {tuple(a.evidence.shape)} "
          f"same {same_field}, wires same {same_wires}; put back {plain['text']!r} and {nulled['text']!r}; "
          f"errors {sorted(plain['errors'])} same {same_errors}")
    assert same_field and same_wires
    assert plain['text'] == nulled['text']
    assert same_errors
    assert 'reconstruction_order_zero.image' not in plain['errors']
    assert plain['back'] is None and nulled['back'] is None


# -- C2: one Not ---------------------------------------------------------------------

def test_c2_one_not(model, monkeypatch):
    from test_item17_places import nominate
    import ImagePercepts
    rows, _, _ = category(2)[0]
    row = rows[0]
    nominate(model, (row['text'],))
    known(model, [row])
    owner = model._concept_owner()
    basic = owner.basic_percepts
    calls, opening = [], []
    order_zero_field, read = owner.order_zero_field, ImagePercepts.read

    def counted(lanes, inputs, rows_):
        if opening:
            calls.append((tuple(int(i) for i in inputs), tuple(lanes.shape)))
        return order_zero_field(lanes, inputs, rows_)

    def reading(model_, sid, bank):
        opening.append(sid)
        try:
            return read(model_, sid, bank)
        finally:
            opening.pop()
    monkeypatch.setattr(owner, 'order_zero_field', counted)
    monkeypatch.setattr(ImagePercepts, 'read', reading)
    present(model, [row], train=False)
    assert len(calls) == 1, calls
    inputs, shape = calls[0]
    # With the unit delay (architecture.perceptDelay) the inputs after the
    # rows are the rows one step back, ``capacity + row``: they are not
    # basic percept rows, so only the rows are counted.
    inputs = tuple(p for p in inputs if p < int(basic.capacity))
    forms = [p for p in inputs if basic.forms[p] is not None and not basic.is_value(p)]
    values = [p for p in inputs if basic.is_value(p)]
    image = set(ImagePercepts.image_rows(model))
    places = [p for p in inputs if p in image and p in basic.place_rows]
    print(f"C2 {row!r}: order_zero_field called {len(calls)} time(s) at the opening, lanes {shape}; "
          f"its rows hold {len(forms)} forms' rows, {len(values)} values' rows "
          f"(values {[round(float(basic.felt([p])[0]), 4) for p in values]}), {len(places)} places' rows of the box")
    assert forms and values and places


# -- C3: in the field ----------------------------------------------------------------

def test_c3_in_the_field(model, monkeypatch):
    """The model reads one stream at a time (its STM is built for the
    training batch, one), so each row is a batch of its own; the readings a
    batch of three rows would join are read from the image's bank."""
    import ImagePercepts
    rows, _, _ = category(2)[0]
    row = dict(text='', scene=rows[0]['scene'])
    known(model, [row])
    values, places = value_and_place_concepts(model)
    lo, hi = scene_range(model)
    observed = Opening(monkeypatch)
    found = {}
    for name, given in (('image', row), ('null', dict(text=rows[0]['text'], scene=None)), ('empty', dict(text='', scene=[]))):
        present(model, [given], train=False)
        field = observed.last['field']
        ids = (field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, 0]).tolist()
        where = field.where[0]
        inside = (where[:, 0] >= lo) & (where[:, 1] <= hi) & (where[:, 1] > where[:, 0])
        read_for = field.evidence[:, 0, :, 0].detach() > 0                                   # [C, E]
        at = lambda wanted: [tuple(where[e].tolist()) for e in range(len(where))
                             if any(bool(read_for[c, e]) for c, cid in enumerate(ids) if cid in wanted)]
        found[name] = dict(readings=[tuple(w) for w, i in zip(where.tolist(), inside.tolist()) if i],
                           value=at(values), place=at(places), reads=bool((read_for & inside[None]).any()))
    # One batch of the three, as the image's bank has it.
    saved = model.inputSpace._scenes
    model.inputSpace._scenes = [model.inputSpace.scene_pixels(s) for s in (row['scene'], None, [])]
    try:
        bank, _ = ImagePercepts.image_bank(model)
    finally:
        model.inputSpace._scenes = saved
    batch = [[tuple(w) for w in readings if w[1] > w[0]] for readings in bank.where.tolist()]
    print(f"C3 scene range {lo}..{hi}; {len(values)} values' and {len(places)} places' concepts; "
          f"each row alone: {found}; one batch of the three, its readings by row: {batch}")
    pixel = [o + p for o, p in zip(model.where_registry.boxes['scene'][0], rows[0]['scene'][0][:3])]
    address = int(model.where_registry.address(torch.tensor(pixel)))
    image = found['image']
    assert (address, address + 1) in image['value'] and (address, address + 1) in image['place']
    assert all(lo <= w[0] and w[1] <= hi for w in image['value'] + image['place'])
    assert found['null']['readings'] == [] and not found['null']['reads']
    assert found['empty']['readings'] == [(lo, hi)] and not found['empty']['reads']
    assert batch == [[(address, address + 1), (lo, hi)], [], [(lo, hi)]]


# -- C4: put back --------------------------------------------------------------------

def test_c4_put_back(model, monkeypatch):
    images = [rows[0] for rows, _, _ in category(1) if rows[0]['scene'] is not None]
    # As many of category 1's images, in order, as the model's order zero
    # has room for each one's value and place.
    owner = model._concept_owner()
    room = int(owner._order_caps()[0]) // 2
    chosen, values, places = [], set(), set()
    for row in images:
        (x, y, z, steps), = row['scene']
        if len(values | {steps}) + len(places | {(x, y, z)}) > room:
            continue
        chosen.append(row)
        values.add(steps)
        places.add((x, y, z))
        if len(chosen) == 8:
            break
    for row in chosen:
        known(model, [row])
    observed = Opening(monkeypatch)
    lo, hi = scene_range(model)
    results = []
    for row in chosen:
        present(model, [row], train=False)
        back = observed.last['back'][0]
        given = the_felt(model, row['scene'])
        results.append((row['scene'], [(int(i), round(float(back[i]), 4)) for i in back.nonzero().flatten()],
                        torch.equal(back, given)))
    print(f"C4 category 1, {len(chosen)} images ({len(values)} steps, {len(places)} places), put back: {results}")
    assert len(values) > 1
    assert all(exact for *_, exact in results)
    # A sentence with its image (category 2): both are put back.
    from test_item17_places import nominate
    rows, _, _ = category(2)[0]
    row = rows[0]
    nominate(model, (row['text'],))
    known(model, [row])
    present(model, [row['text']], train=False)
    alone = observed.last['text']
    present(model, [row], train=False)
    both = observed.last
    exact = torch.equal(both['back'][0], the_felt(model, row['scene']))
    print(f"C4 category 2, {row!r}: the sentence put back {both['text']!r} (alone {alone!r}); "
          f"the image put back exactly {exact}")
    assert both['text'] == alone
    assert exact


# -- C5: charged ---------------------------------------------------------------------

def test_c5_charged(model, monkeypatch):
    import ImagePercepts
    from Layers import Error
    rows, _, _ = category(2)[0]
    row = dict(text='', scene=rows[0]['scene'])
    owner = model._concept_owner()
    reached = []
    charge = ImagePercepts.charge

    def gradient(model_, errors, field, record):
        """Inside a training opening: the image's term with the image
        taken as empty, and its gradient at the Sigma of order zero."""
        if record['back'] is None or not torch.is_grad_enabled():
            return
        saved = model_.inputSpace._scenes
        model_.inputSpace._scenes = [() for _ in saved]
        try:
            empty = Error(row_mask=torch.ones(len(saved), dtype=torch.bool))
            charge(model_, empty, 0, field, torch.ones(len(saved), dtype=torch.bool))
        finally:
            model_.inputSpace._scenes = saved
        term = empty._value(empty._terms['reconstruction_order_zero.image'])
        grad = (torch.autograd.grad(term, owner.order_zero.value, retain_graph=True, allow_unused=True)[0]
                if term.requires_grad else None)
        reached.append((float(term.detach()), None if grad is None else float(grad.abs().sum())))

    observed = Opening(monkeypatch, during=gradient)
    present(model, [row])
    first = observed.last
    known(model, [row])
    after = observed.last
    present(model, [dict(text=rows[0]['text'], scene=None)], train=False)
    none = observed.last
    term = 'reconstruction_order_zero.image'
    entries = after['terms'][term]['entries'][-1].detach().tolist()
    # A batch of a row with an image and a row with none, charged on the
    # field the image's row opened: the second row is charged nothing.
    from Layers import Error
    saved = model.inputSpace._scenes
    model.inputSpace._scenes = [model.inputSpace.scene_pixels(row['scene']), None]
    field = after['field']
    from dataclasses import replace
    pair = replace(field, concept_ids=(field.concept_ids if field.concept_ids.ndim == 1
                                       else field.concept_ids.expand(-1, 2)), evidence=field.evidence.expand(-1, 2, -1, -1), where=field.where.expand(2, -1, -1),
                   when=field.when.expand(2, -1, -1), wires=None)
    try:
        paired = Error(row_mask=torch.ones(2, dtype=torch.bool))
        charge(model, paired, 0, pair, torch.ones(2, dtype=torch.bool))
    finally:
        model.inputSpace._scenes = saved
    numerators = paired._terms[term]['numerators'][0].detach().tolist()
    entries = paired._terms[term]['entries'][0].detach().tolist()
    print(f"C5 {row!r}: the image's term at first sight {float(first['errors'][term]):.4f}, "
          f"once known {float(after['errors'][term]):.4f}; with a row with no image, by row: "
          f"numerators {numerators}, entries {entries}; a batch with no image has the term "
          f"{term in none['errors']}; the image taken as empty, term and |gradient| at the Sigma "
          f"of order zero, last {reached[-1] if reached else None}")
    assert float(first['errors'][term]) > 0
    assert float(after['errors'][term]) == 0
    assert entries[0] > 0 and entries[1] == 0 and numerators[1] == 0
    assert term not in none['errors']
    assert reached and reached[-1][0] > 0 and reached[-1][1] is not None and reached[-1][1] > 0
