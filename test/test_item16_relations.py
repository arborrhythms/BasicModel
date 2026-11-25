"""Relations of place.

The certificates G1 to G3 of §42.2 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured. An axis of place is a sorted line. For the things of an image's
bank, each thing of several places and each pixel a thing of one place,
``ImagePercepts.relations`` reads every ordered pair: on each axis of the box
whether the first lies wholly before the second (its greatest coordinate less
than the other's least), and across each axis whether the two touch by a
side; each on two lanes, for and against. The corpus puts the lesser y above
(``felt_world_corpus.p_row_before``: the first's greatest y less than the
second's least) and the lesser x to the left (``p_column_before``).
"""
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

X, Y, Z = 0, 1, 2
BEFORE, TOUCHING = 0, 1


@pytest.fixture
def reader(tmp_path, monkeypatch):
    """The scene model's box and widths, with a table of basic percepts of
    its own (the model's has room for a lesson, not a corpus)."""
    from BasicPercepts import BasicPercepts
    from test_item16_scene_field import scene_model
    model = scene_model(tmp_path, monkeypatch)
    basic = model._concept_owner().basic_percepts
    table = BasicPercepts(None, basic.identity_width, basic.property_width, 1 << 15)
    table.bind(None)
    yield model, table
    model.End()


def bank_of(reader, scene):
    """The image's bank of ``scene`` with its things, as the image's reader
    makes it (``ImagePercepts.image_bank``, ``BasicPercepts.with_things``),
    and the intensity of each pixel's occurrence ``[1, A]``."""
    model, table = reader
    registry = model.where_registry
    lo = registry.slices['scene'][0]
    origin = registry.boxes['scene'][0]
    at = registry.address(torch.tensor([[o + p for o, p in zip(origin, pixel[:3])] for pixel in scene]))
    order = at.argsort()
    addresses = at[order][None]
    values = (torch.tensor([pixel[3] for pixel in scene], dtype=table.value_codes.dtype) / 255)[order][None]
    places = torch.stack((addresses, addresses + 1), -1)
    percepts = table.perceive_intensities(values)
    bank = table.value_bank(values, places, registry=registry, when_encoding=model.when_encoding,
                            time=model.present())
    bank, _ = table.with_things(bank, percepts)
    assert lo <= int(addresses.min())
    return bank, values


def held(bank, registry, o):
    """The places of the box that the thing at occurrence ``o`` holds."""
    origin = registry.boxes['scene'][0]
    members = dict(bank.things[0] or ()).get(o) if bank.things else None
    occurrences = [p for _, p in members] if members else [o]
    places = registry.place(bank.span[0, occurrences, 0]).tolist()
    return frozenset(tuple(a - b for a, b in zip(p, origin)) for p in places)


def read(reader, scene):
    """The relations of ``scene``'s things, ``{(first, second): lanes [2, axes, 2]}``,
    each thing named by the places it holds."""
    import ImagePercepts
    model, _ = reader
    bank, _ = bank_of(reader, scene)
    occurrence, lanes = ImagePercepts.relations(bank)
    things = [held(bank, model.where_registry, o) for o in occurrence[0].tolist() if o >= 0]
    return {(a, b): lanes[0, i, j] for i, a in enumerate(things) for j, b in enumerate(things) if i != j}


def places(*points):
    return frozenset(tuple(p) for p in points)


# -- G1: order on an axis ------------------------------------------------------------

def test_g1_order_on_an_axis(reader):
    """Two things apart on every axis, and two whose extents overlap on x."""
    run = [[1, 1, 0, 51], [2, 1, 0, 128]]
    found = read(reader, run + [[4, 3, 2, 200]])
    a, b = places((1, 1, 0), (2, 1, 0)), places((4, 3, 2))
    forward = found[(a, b)][BEFORE].tolist()
    backward = found[(b, a)][BEFORE].tolist()
    over = read(reader, [[1, 1, 0, 51], [2, 1, 0, 128], [3, 1, 0, 51], [2, 4, 0, 200]])
    c, d = places((1, 1, 0), (2, 1, 0), (3, 1, 0)), places((2, 4, 0))
    across = over[(c, d)][BEFORE].tolist(), over[(d, c)][BEFORE].tolist()
    print(f"G1 {sorted(a)} and {sorted(b)}: before on x, y, z (for, against) {forward}, "
          f"the other way round {backward}; {sorted(c)} and {sorted(d)}, overlapping on x: "
          f"{across[0]}, the other way round {across[1]}")
    assert forward == [[1., 0.]] * 3 and backward == [[0., 1.]] * 3
    assert across[0][X] == [0., 1.] and across[1][X] == [0., 1.]
    assert across[0][Y] == [1., 0.] and across[1][Y] == [0., 1.]
    assert across[0][Z] == [0., 1.] and across[1][Z] == [0., 1.]


# -- G2: touching --------------------------------------------------------------------

def test_g2_touching(reader):
    """Two places that share a side touch across the axis they differ on;
    two a place apart, or meeting at a corner, do not. A thing and a pixel of
    its own share a place and do not touch. Touching places are one thing of
    the bank, so the things that share a side are its pixels."""
    scene = [[1, 1, 0, 51], [2, 1, 0, 128], [2, 2, 0, 51], [2, 2, 1, 51], [4, 1, 0, 200], [5, 2, 0, 200]]
    found = read(reader, scene)
    one = lambda x, y, z: places((x, y, z))
    cases = {'side on x': (one(1, 1, 0), one(2, 1, 0), X), 'side on y': (one(2, 1, 0), one(2, 2, 0), Y),
             'side on z': (one(2, 2, 0), one(2, 2, 1), Z), 'a place apart': (one(2, 1, 0), one(4, 1, 0), None),
             'at a corner': (one(4, 1, 0), one(5, 2, 0), None),
             'thing and its pixel': (places((1, 1, 0), (2, 1, 0), (2, 2, 0), (2, 2, 1)), one(1, 1, 0), None)}
    touching = {name: (found[(a, b)][TOUCHING].tolist(), found[(b, a)][TOUCHING].tolist())
                for name, (a, b, _) in cases.items()}
    print(f"G2 {scene}: touching across x, y, z (for, against), each way round: {touching}")
    for name, (a, b, axis) in cases.items():
        for lanes in touching[name]:
            assert lanes == [[1., 0.] if k == axis else [0., 1.] for k in (X, Y, Z)], name


# -- G3: the corpus ------------------------------------------------------------------

PREDICATES = ('row_before', 'column_before', 'on_top', 'next_to')


def meaning_read(lanes, first, second, predicate):
    """A meaning's relation of two things read from the image's relations:
    ``first`` and ``second`` the indices of the things of the image that hold
    each of its two things. Wholly before is before for every pair; sharing
    a line of the other axis is, for some pair, neither before the other on
    it; touching is touching for some pair."""
    pairs = lanes[first][:, second]                                           # [F, S, 2, axes, 2]
    back = lanes[second][:, first].transpose(0, 1)
    before = lambda axis: bool((pairs[:, :, BEFORE, axis, 0] > 0).all())
    share = lambda axis: bool(((pairs[:, :, BEFORE, axis, 1] > 0) & (back[:, :, BEFORE, axis, 1] > 0)).any())
    touch = lambda axes: bool((pairs[:, :, TOUCHING, axes, 0] > 0).any())
    if predicate == 'row_before':
        return before(Y) and share(X)
    if predicate == 'column_before':
        return before(X) and share(Y)
    if predicate == 'on_top':
        return before(Y) and share(X) and touch([X, Y, Z])
    return touch([X])


def test_g3_the_corpus(reader):
    """Every image of the frozen corpus whose row records a meaning of above,
    below, to the left of, to the right of, on top of, under or next to
    between two things (the predicates ``row_before``, ``column_before``,
    ``on_top`` and ``next_to``, whose frames say "below", "to the right of"
    and "under" with the arguments in the same order). A meaning's thing is
    the corpus's places of one value; it is mapped to the things of the
    image's bank whose pixels' steps all lie in that value's band."""
    import ImagePercepts
    import felt_world_corpus as felt
    manifest = json.loads((felt.FELT / 'manifest.json').read_text())
    images, checked, wrong, by, merged = set(), 0, [], {}, 0
    for entry in manifest['files']:
        for line in (felt.FELT / entry['path']).read_text().splitlines():
            row = json.loads(line)
            for meaning in row['meaning']:
                scene = row['entries'][meaning['entry']]['scene']
                if meaning['predicate'] not in PREDICATES or not scene:
                    continue
                bank, values = bank_of(reader, scene)
                occurrence, lanes = ImagePercepts.relations(bank)
                composite = dict(bank.things[0] or ()) if bank.things else {}
                value = lambda o: felt.value_of(round(float(values[0, o]) * 255))
                kinds = [{value(p) for _, p in composite[o]} if o in composite else {value(o)}
                         for o in occurrence[0].tolist() if o >= 0]
                a, b = meaning['slots']['a'], meaning['slots']['b']
                first = [i for i, k in enumerate(kinds) if k == {a}]
                second = [i for i, k in enumerate(kinds) if k == {b}]
                found = meaning_read(lanes[0], first, second, meaning['predicate']) if first and second else None
                merged += any({a, b} <= k for k in kinds)
                images.add((row['key'], meaning['entry']))
                checked += 1
                count = by.setdefault(meaning['predicate'], [0, 0])
                count[0 if meaning['truth'] else 1] += 1
                if found != meaning['truth']:
                    wrong.append((row['key'], meaning['entry'], meaning['predicate'], a, b, meaning['truth'], found))
    print(f"G3 the frozen corpus: {len(images)} images, {checked} relations checked "
          f"(by predicate, true and false: {by}); {merged} of them with the two in one thing of the bank, read by its pixels; disagreeing {len(wrong)} {wrong[:10]}")
    assert checked and not wrong
