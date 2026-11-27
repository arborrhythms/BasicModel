"""Things wider than a place.

The certificates E1 to E8 of §40.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured. As a word is a run of letters between spaces, a thing in an image
is a run of touching places that are not background: an occurrence beside
its pixels' occurrences, its ``.where`` the box of its places. Its percept is
made of its pixels' as a form's is of its characters: its parts its pixels'
basic percepts in their order within its box, each at its place from the
box's origin, its whole the least upper cut all its pixels lie under. It is
read as a form is, for and against, by the Pi of the value space, nominated
on recurrence, and put back from its concept to its parts' centroids.
"""
import json
import os
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch


def _image_model():
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_image.xml')
    return model


def _image(pixels):
    """A 28 by 28 image of the numeric stem, ``{(x, y): value}``."""
    image = torch.zeros(1, 28, 28)
    for (x, y), value in pixels.items():
        image[0, y, x] = value
    return image


def _run(at, values):
    """A run of places along x from ``at``, ``{(x, y): value}``."""
    return {(at[0] + i, at[1]): v for i, v in enumerate(values)}


def _bank(model, image):
    """The numeric stem's bank of ``image`` with its things, and each
    thing's row ``[1, things]``, as ``NumericPercepts.read`` makes them."""
    basic = model._concept_owner().basic_percepts
    values = image.reshape(1, -1)
    A = values.shape[1]
    places = model.where_registry.intervals('numeric', torch.arange(A)[None])
    percepts = basic.perceive_intensities(values)
    bank = basic.value_bank(values, places, registry=model.where_registry,
                            when_encoding=model.when_encoding, time=model.present())
    return basic.with_things(bank, percepts)


def _address(model, x, y):
    return model.where_registry.slices['numeric'][0] + 28 * y + x


def _read(model, bank, rows):
    basic = model._concept_owner().basic_percepts
    return basic.read_values(replace(bank, rows=tuple(rows), codes=basic.bank_codes(list(rows))))


# -- E1: a run is a thing ------------------------------------------------------------

def test_e1_a_run_is_a_thing():
    """Two groups of touching places are two things, each with the box of
    its places; a place alone is a thing of one place and gains no second
    row."""
    model = _image_model()
    basic = model._concept_owner().basic_percepts
    first, second, alone = _run((3, 2), [.2, .5, .2]), {(10, 10): .9, (10, 11): .9}, {(20, 20): .5}
    bank, things = _bank(model, _image({**first, **second, **alone}))
    A = 28 * 28
    boxes = [tuple(w) for w in bank.where[0, A + 1:].tolist() if w[1] > w[0]]
    expected = [(_address(model, 3, 2), _address(model, 5, 2) + 1),
                (_address(model, 10, 10), _address(model, 10, 11) + 1)]
    pixel = basic.perceive_intensity(torch.tensor(.5))
    print(dict(things=things.tolist(), boxes=boxes, expected=expected,
               arrangements={row: basic.things[row] for row in basic.things},
               rows=len(basic.index), rows_of_the_lone_place=[r for r in range(len(basic.index))
                                                               if basic._keys[r] == basic._keys[pixel]]))
    assert sorted(boxes) == sorted(expected)
    assert len(basic.things) == 2 and sorted(things[0].tolist()) == sorted(basic.things)
    assert all(len(arrangement) > 1 for arrangement in basic.things.values())
    # Three pixels' rows (.2, .5, .9) and the two things'; the lone .5 is a
    # pixel of a row the first thing's middle already made.
    assert len(basic.index) == 3 + 2


# -- E2: the same thing elsewhere ----------------------------------------------------

def test_e2_the_same_thing_elsewhere():
    model = _image_model()
    basic = model._concept_owner().basic_percepts
    here, there = (3, 2), (15, 20)
    a, rows_a = _bank(model, _image(_run(here, [.2, .5, .2])))
    b, rows_b = _bank(model, _image(_run(there, [.2, .5, .2])))
    A = 28 * 28
    print(dict(rows=(rows_a.tolist(), rows_b.tolist()), where=(a.where[0, A + 1].tolist(), b.where[0, A + 1].tolist()),
               things=len(basic.things)))
    assert rows_a.tolist() == rows_b.tolist() and len(basic.things) == 1
    assert a.where[0, A + 1].tolist() == [_address(model, *here), _address(model, here[0] + 2, here[1]) + 1]
    assert b.where[0, A + 1].tolist() == [_address(model, *there), _address(model, there[0] + 2, there[1]) + 1]


# -- E3: parts and whole -------------------------------------------------------------

def test_e3_parts_and_whole():
    """A thing's row holds its pixels' percepts in order as parts, each at
    its place in its box, and the least upper cut over them as its whole, in
    the Pi of the value space; its code is the greatest of its parts' codes
    and the least of its wholes'."""
    model = _image_model()
    basic = model._concept_owner().basic_percepts
    _, things = _bank(model, _image({(3, 2): .2, (4, 2): .5, (4, 3): .2}))
    row = int(things[0, 0])
    low, middle = (basic.perceive_intensity(torch.tensor(v)) for v in (.2, .5))
    arrangement = basic.things[row]
    parts, wholes = basic.occurrences[row]
    code = basic.code(row)
    print(dict(row=row, arrangement=arrangement, parts=parts, wholes=wholes,
               entries_first_pi=basic.pi.row_members(row), entries_value_pi=basic.value_pi.row_members(row),
               code=code.tolist(), key=basic._keys[row]))
    assert arrangement == (((0, 0, 0), low), ((1, 0, 0), middle), ((1, 1, 0), low))
    assert parts == tuple(basic.occurrences[p][0][0] for p in (low, middle, low))
    assert wholes == basic.occurrences[middle][1]
    assert basic.pi.row_members(row) == []
    assert basic.value_pi.row_members(row) == [('ps', arrangement), ('ws', wholes[0])]
    # Its whole is the OR of its pixels' own wholes: it holds each of them.
    cuts = sorted({basic.occurrences[p][1][0] for p in (low, middle)})
    torch.testing.assert_close(code, torch.cat((basic.part_codes(list(parts)).amax(0),
                                                basic.whole_codes(cuts).amax(0))))
    iw = basic.identity_width
    torch.testing.assert_close(code[:iw], basic.code(middle)[:iw])
    for pixel in (low, middle):
        assert float(torch.maximum(1 - basic.code(pixel)[iw:], code[iw:]).amin()) == 1.
    assert not basic.is_value(row)


# -- E4: for and against -------------------------------------------------------------

def test_e4_for_and_against():
    """The thing's row reads for at its own occurrence and against at an
    occurrence that holds its parts and one pixel above its whole, nothing
    where a part is not held; a run within a thing reads at it where it
    lies within it place by place; a pixel's row reads at the pixel's
    reading and not at the thing's."""
    model = _image_model()
    basic = model._concept_owner().basic_percepts
    A = 28 * 28
    _, thing = _bank(model, _image(_run((3, 2), [.2, .5, .2])))
    _, small = _bank(model, _image(_run((3, 2), [.2, .5])))
    row, within = int(thing[0, 0]), int(small[0, 0])
    middle = basic.perceive_intensity(torch.tensor(.5))
    read = {}
    for name, values in (('own', [.2, .5, .2]), ('one above', [.2, .9, .2]), ('his [1,3,3,1]', [.2, .9, .9, .2]),
                         ('a part not held', [.2, .2, .2])):
        bank, _ = _bank(model, _image(_run((3, 2), values)))
        read[name] = tuple(_read(model, bank, [row])[0, 0, A + 1].tolist())
    bank, _ = _bank(model, _image(_run((3, 2), [.2, .5, .2])))
    lanes = _read(model, bank, [within, middle])
    at_pixel = _address(model, 4, 2) - model.where_registry.slices['numeric'][0]
    read['the run [.2,.5] within the thing'] = tuple(lanes[0, 0, A + 1].tolist())
    read['the pixel .5 at the thing'] = tuple(lanes[1, 0, A + 1].tolist())
    read['the pixel .5 at its place'] = tuple(lanes[1, 0, at_pixel].tolist())
    read['the run [.2,.5] at the pixel'] = tuple(lanes[0, 0, at_pixel].tolist())
    print(dict(read=read))
    assert read['own'] == (1., 0.)
    assert read['one above'] == (0., 1.) and read['his [1,3,3,1]'] == (0., 1.)
    assert read['a part not held'] == (0., 0.)
    assert read['the run [.2,.5] within the thing'] == (1., 0.)
    assert read['the pixel .5 at the thing'] == (0., 0.) and read['the pixel .5 at its place'] == (1., 0.)
    assert read['the run [.2,.5] at the pixel'] == (0., 0.)


# -- E5: recurrence ------------------------------------------------------------------

def test_e5_recurrence():
    model = _image_model()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    model.train()
    image = _image(_run((3, 2), [.2, .5, .2]))
    seen = []
    for time in range(owner.concept_mint_count):
        model.forward(image)
        seen.append({row: (bool(basic.promoted[row]), owner.concept_of_basic_percept(row)) for row in basic.things})
    print(dict(mint_count=owner.concept_mint_count, seen=seen))
    (row, (promoted, concept)), = seen[0].items()
    assert not promoted and concept is None
    assert all(s[row] == (False, None) for s in seen[:-1])
    assert seen[-1][row][0] and seen[-1][row][1] is not None


# -- E6: the limit -------------------------------------------------------------------

def test_e6_the_limit():
    """With intensityPartSigma one no thing's row is made; the XOR gate,
    the XOR certificate and test_concept_output.py are certified by the run
    that names them."""
    from test_item16_values import _xor, _recur
    model = _xor()
    basic = model._concept_owner().basic_percepts
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1)
    _recur(model, values)
    model.forward(values)
    field = model.numeric_percepts.read(values)
    A = model.numeric_percepts.dimensions
    places = model.where_registry.intervals('numeric', torch.arange(A).expand(1, -1))
    flat = torch.tensor([[1., 1.]])
    percepts = basic.perceive_intensities(flat)
    bank = basic.value_bank(flat, places, registry=model.where_registry,
                            when_encoding=model.when_encoding, time=model.present())
    _, at_one = basic.with_things(bank, percepts)
    print(dict(part_sigma=basic.part_sigma, things=basic.things,
               readings=field.where.shape[1], places=A, at_one=at_one.tolist()))
    assert basic.part_sigma == 1
    assert basic.things == {} and field.where.shape[1] == A + 1 and at_one.shape[1] == 0


# -- E7: put back --------------------------------------------------------------------

def _thing_concepts(owner):
    from Spaces import _concept_alloc_of
    basic = owner.basic_percepts
    return {int(owner.concept_id_at_row(concept)) for concept, percept, negated
            in _concept_alloc_of(owner).order_zero_members
            if not negated and percept in basic.things and owner.concept_id_at_row(concept) is not None}


def _only(field, concepts):
    ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, 0]
    keep = torch.tensor([int(c) in concepts for c in ids.tolist()])
    return replace(field, evidence=field.evidence * keep[:, None, None, None].to(field.evidence))


def test_e7_put_back_on_the_numeric_stem():
    """After a thing of several pixels has recurred, its image comes back at
    every place from the thing's concept alone, to its parts' centroids;
    with the pixels' concepts too, they agree."""
    from test_item16_values import _recur
    model = _image_model()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    thing = _run((3, 2), [.2, .5, .2])
    image = _image({**thing, (20, 20): .9})
    _recur(model, image)
    field = model.numeric_percepts.read(image)
    concepts = _thing_concepts(owner)
    alone = model.numeric_percepts.reverse(replace(_only(field, concepts), percepts=None))
    both = model.numeric_percepts.reverse(replace(field, percepts=None))
    places = image[0] > 0
    centroid = {v: float(basic.felt([basic.perceive_intensity(torch.tensor(v))])[0]) for v in (.2, .5)}
    print(dict(thing_concepts=sorted(concepts), from_the_thing=alone[0][places].tolist(),
               with_the_pixels=both[0][places].tolist(), given=image[0][places].tolist(), centroids=centroid))
    assert concepts
    expected = torch.zeros_like(image)
    for (x, y), v in thing.items():
        expected[0, y, x] = centroid[v]
    assert torch.equal(alone, expected)
    # The levels of intensity are linear and fixed (item 1.4, §14.6): a value
    # is put back at the level it falls to, within half a step.
    at_levels = torch.where(image > 0, basic.levels[basic.fall(image)].to(image), torch.zeros_like(image))
    assert torch.equal(both, at_levels)
    assert float((both - image).abs().max()) <= .5 / (len(basic.levels) - 1) + 1e-7


def test_e7_put_back_with_a_sentence(tmp_path, monkeypatch):
    """The same with a sentence's image, and the charge: where a thing's
    and a pixel's concepts both give a place back and differ, the lesser is
    put back and the difference charged."""
    import ImagePercepts
    from reading_fixtures import use_eager_reading
    from test_item16_scene_field import scene_model, Opening, present, known, the_felt
    model = scene_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        owner = model._concept_owner()
        basic = owner.basic_percepts
        thing = [[1, 1, 0, 51], [2, 1, 0, 128], [3, 1, 0, 51]]
        row = dict(text='', scene=thing)
        known(model, [row])
        observed = Opening(monkeypatch)
        present(model, [row], train=False)
        field = observed.last['field']
        concepts = _thing_concepts(owner)
        alone = ImagePercepts.put_back(model, _only(field, concepts))[0]
        given = the_felt(model, thing)
        term = observed.last['errors'].get('reconstruction_order_zero.things')
        # Three places at the middle's value: the pixels give it back at
        # each, the known thing reads for there and gives back its lesser ends.
        present(model, [dict(text='', scene=[[x, y, z, 128] for x, y, z, _ in thing])], train=False)
        flat = observed.last
        lo = model.where_registry.slices['scene'][0]
        origin = model.where_registry.boxes['scene'][0]
        at = [int(model.where_registry.address(torch.tensor([o + q for o, q in zip(origin, p[:3])]))) - lo
              for p in thing]
        print(dict(thing_concepts=sorted(concepts), from_the_thing=[round(float(alone[i]), 4) for i in at],
                   given=[round(float(given[i]), 4) for i in at], things_term_when_they_agree=term,
                   flat_put_back=[round(float(flat['back'][0][i]), 4) for i in at],
                   flat_things_term=float(flat['errors']['reconstruction_order_zero.things'])))
        assert concepts
        assert torch.equal(alone, given)
        assert term is None or float(term) == 0
        assert [round(float(flat['back'][0][i]), 4) for i in at] == [round(float(given[i]), 4) for i in at]
        assert float(flat['errors']['reconstruction_order_zero.things']) > 0
    finally:
        model.End()


# -- E8: the corpus (a measure) ------------------------------------------------------

def test_e8_the_corpus(tmp_path, monkeypatch):
    """Over the categories whose images hold things wider than a place, in
    their train split: the number of things, and of their rows, and of the
    rows met in as many images as nominates. Reported, on the scene model's
    box and widths, in a table of basic percepts of its own (the model's has
    room for a lesson, not a corpus)."""
    import felt_world_corpus as felt
    from BasicPercepts import BasicPercepts
    from util import TheXMLConfig
    from test_item16_scene_field import scene_model
    model = scene_model(tmp_path, monkeypatch)
    try:
        owner = model._concept_owner()
        registry, basic = model.where_registry, owner.basic_percepts
        mint = owner.concept_mint_count
        width = int(TheXMLConfig.get('architecture.intensityPartSigma'))
        lo = registry.slices['scene'][0]
        origin = registry.boxes['scene'][0]
        table = BasicPercepts(None, basic.identity_width, basic.property_width, 1 << 15)
        table.bind(None)
        manifest = json.loads((felt.FELT / 'manifest.json').read_text())
        report, met = {}, {}
        for entry in manifest['files']:
            scenes = [s for d in felt.load(entry['category'], 'train') for s in d.scenes if s]
            if not scenes:
                continue
            A = max(map(len, scenes))
            values, addresses = torch.zeros(len(scenes), A), torch.full((len(scenes), A), lo, dtype=torch.long)
            for b, scene in enumerate(scenes):
                at = registry.address(torch.tensor([[o + p for o, p in zip(origin, pixel[:3])] for pixel in scene]))
                order = at.argsort()
                addresses[b, :len(scene)] = at[order]
                values[b, :len(scene)] = (torch.tensor([pixel[3] for pixel in scene]) / 255)[order]
            places = torch.stack((addresses, torch.where(values > 0, addresses + 1, addresses)), -1)
            percepts = table.perceive_intensities(values)
            bank = table.value_bank(values, places, registry=registry, when_encoding=model.when_encoding, time=0)
            _, things = table.with_things(bank, percepts)
            rows = {}
            for b in range(len(scenes)):
                for row in set(things[b].tolist()) - {-1}:
                    rows[row] = rows.get(row, 0) + 1
                    met[row] = met.get(row, 0) + 1
            if rows:
                report[entry['category']] = dict(images=len(scenes), things=int((things >= 0).sum()),
                                                 rows=len(rows), nominated=sum(n >= mint for n in rows.values()))
        total = dict(things=sum(r['things'] for r in report.values()), rows=len(met),
                     nominated=sum(n >= mint for n in met.values()))
        print(dict(intensityPartSigma=width, mint_count=mint, categories=sorted(report), by_category=report,
                   all=total))
        assert report and total['things'] >= total['rows'] >= total['nominated']
    finally:
        model.End()
