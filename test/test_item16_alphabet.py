"""An alphabet of intensity, combined as letters are.

The certificates A1 to A8 of §46.2 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured. Text's atoms are its character set, given; intensity's are an
alphabet of levels of one pixel on the sorted line of values, learned by
vector quantization, ``intensityPartCardinality`` of them. An atom's basic
percept is a row over a part and a whole taken from the alphabet: its level,
and the set of values allowed to it; its centroid is its level. Adjacent
atoms within the aperture are a part, its parts in order, as letters are;
the aperture is a Gaussian about a part's centre, ``intensityPartSigma``
places wide, the limit on whatever reaches across places, the Pi that
combines and the readout's pooling of a concept over its readings. A
Sigma above order zero ranges over the rows beneath it: a thing of one
concept, and a basic percept's row through the Not of order zero.
"""
import ast
import os

import pytest
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_ROOT, "bin")


def _table(monkeypatch=None, **keys):
    """A table of basic percepts of its own, three coordinates wide a half,
    with the keys of the architecture ``keys`` set while it is made."""
    from BasicPercepts import BasicPercepts
    from util import TheXMLConfig
    for name, value in keys.items():
        monkeypatch.setitem(TheXMLConfig._data['architecture'], name, value)
    table = BasicPercepts(None, 3, 3, 1024)
    table.bind(None)
    return table


def _image_bank(model, table, image):
    """``image`` ``[1, 28, 28]`` as the numeric stem's bank on ``table``, with
    its things, and each place's row."""
    values = image.reshape(1, -1)
    places = model.where_registry.intervals('numeric', torch.arange(values.shape[1])[None])
    percepts = table.perceive_intensities(values)
    bank = table.value_bank(values, places, registry=model.where_registry,
                            when_encoding=model.when_encoding, time=model.present())
    return table.with_things(bank, percepts)


def _run_image(values, at=(3, 2)):
    image = torch.zeros(1, 28, 28)
    for i, v in enumerate(values):
        image[0, at[1], at[0] + i] = v
    return image


# -- A1: the alphabet -----------------------------------------------------------------

def test_a1_the_alphabet(monkeypatch):
    """By default 256 levels in order on 0..1, each eight-bit step falling to
    a level of its own; the key sets another number; no number of levels is
    in the code."""
    table = _table()
    levels = table.levels
    steps = torch.arange(256) / 255
    fallen = table.fall(steps)
    print(dict(levels=len(levels), first=float(levels[0]), last=float(levels[-1]),
               in_order=bool((levels[1:] > levels[:-1]).all()), distinct_levels_of_the_steps=len(fallen.unique()),
               step_falls_to_its_own=bool(torch.equal(fallen, torch.arange(256)))))
    assert len(levels) == 256 and float(levels[0]) == 0 and float(levels[-1]) == 1
    assert bool((levels[1:] > levels[:-1]).all())
    assert torch.equal(fallen, torch.arange(256))
    other = _table(monkeypatch, intensityPartCardinality=16)
    print(dict(set_by_the_key=len(other.levels), levels=[round(float(v), 4) for v in other.levels]))
    assert len(other.levels) == 16 and bool((other.levels[1:] > other.levels[:-1]).all())
    # The number is read from the key and written nowhere in the code that
    # makes, finds or learns a level.
    with open(os.path.join(_BIN, 'BasicPercepts.py')) as handle:
        tree = ast.parse(handle.read())
    numbers = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                '__init__', 'fall', 'learn', 'level_seat', 'perceive_intensity', 'perceive_intensities', 'value_bank'):
            numbers[node.name] = sorted({n.value for n in ast.walk(node) if isinstance(n, ast.Constant)
                                         and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
                                         and n.value > 2})
    print(dict(numbers_in_the_code=numbers))
    assert not any(256 in found or 255 in found for found in numbers.values())
    with open(os.path.join(_ROOT, 'data', 'model.xml')) as handle:
        assert '<intensityPartCardinality>256</intensityPartCardinality>' in handle.read()


# -- A2: learned ----------------------------------------------------------------------

def test_a2_learned():
    """Intensities presented about a value move the level they fall to toward
    their mean, very slowly (item 1.4, §14.3: the alphabet is a linear
    self-organizing map), the levels beside it less, and the levels stay in
    order; what fell to the level goes on falling to it."""
    table = _table()
    table.levels_fixed = False        # the learned alphabet (``intensityLevelsFixed`` false)
    level = int(table.fall(torch.tensor(.3)))
    before = float(table.levels[level])
    step = float(table.levels[1] - table.levels[0])
    about = torch.tensor([[before + d * step for d in (-.4, -.1, .05, .3, .35)]])
    mean = float(about.double().mean())
    table.perceive_intensities(about)
    after = float(table.levels[level])
    print(dict(level=level, before=before, presented=about.tolist(), their_mean=mean, after=after,
               neighbours=[float(table.levels[level - 1]), float(table.levels[level + 1])]))
    assert table.fall(about).unique().tolist() == [level]
    rate = 1 - (1 - table.LEVEL_RATE) ** about.shape[1]
    assert after == pytest.approx(before + rate * (mean - before), abs=1e-7)
    assert 0 < abs(after - before) < step / 100 and abs(after - mean) < abs(before - mean)
    # Every intensity of a fine grid of the line, presented: the levels move
    # and stay in order.
    grid = torch.arange(1, 1000)[None] / 999
    old = table.levels.clone()
    table.perceive_intensities(grid)
    moved = int((table.levels != old).sum())
    print(dict(grid=grid.shape[1], levels_moved=moved, in_order=bool((table.levels[1:] > table.levels[:-1]).all()),
               greatest_move=float((table.levels - old).abs().max())))
    assert moved > 0 and bool((table.levels[1:] > table.levels[:-1]).all())


# -- A3: an atom's row ----------------------------------------------------------------

def test_a3_an_atoms_row():
    """Its part is its level and its whole the set of values allowed to it,
    the level's seat as a lower cut and as an upper cut, read for by what falls
    to the level and by nothing above it (item 1.4: nothing is read against); its centroid is its level; a
    value met once has no row promoted, and met as often as nominates it has
    one."""
    from test_item16_values import _xor, _bank
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    model.train()
    value = .5
    level = int(basic.fall(torch.tensor(value)))
    seen = []
    for _ in range(owner.concept_mint_count):
        model.forward(torch.tensor([[[value], [0.]]]))
        rows = [r for r in range(len(basic.index)) if basic.is_value(r)]
        seen.append([(r, bool(basic.promoted[r]), owner.concept_of_basic_percept(r)) for r in rows])
    (row, promoted, concept), = seen[-1]
    seat = int(basic.level_seats[level])
    below, middle, above = basic.levels[level - 1:level + 2].tolist()
    # Two values within the level's cell, a quarter of the way to each
    # neighbour, and the two neighbouring levels.
    values = torch.tensor([[middle - (middle - below) / 4, middle + (above - middle) / 4], [above, below]])
    bank, _ = _bank(model, values)
    bank = replace(bank, rows=(row,), codes=basic.bank_codes([row]))
    lanes = basic.read_values(bank)[0, :, :2]
    cell = ((below + middle) / 2, (middle + above) / 2)
    print(dict(level=level, seat=seat, occurrence=basic.occurrences[row], key=basic._keys[row],
               centroid=float(basic.felt([row])[0]), level_value=middle, cell_allowed=cell, seen=seen,
               read={f'{float(v):.5f}': tuple(lanes[b, a].tolist()) for b in range(2) for a, v in enumerate(values[b])}))
    assert basic.occurrences[row] == ((seat,), (seat,))
    assert float(basic.felt([row])[0]) == float(basic.levels[level]) == float(basic.felt_of(seat))
    assert seen[0] == [(row, False, None)] and all(s == [(row, False, None)] for s in seen[:-1])
    assert promoted and concept is not None
    assert lanes[0].tolist() == [[1., 0.], [1., 0.]]
    assert lanes[1].tolist() == [[0., 0.], [0., 0.]]


# -- A4: the aperture -----------------------------------------------------------------

def test_a4_the_aperture(monkeypatch):
    """At one, no part is wider than a place and no concept is pooled across
    places at the output; at three, a part's places lie nearer its centre
    than three, each weighed by the Gaussian; at nothing, a part is a whole
    run."""
    from Spaces import _concept_alloc_of, pooled_over_readings
    from recon_bench import _build_model
    from test_item16_values import _xor, _recur
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1)
    _recur(model, values)
    model.forward(values)
    model.eval()
    field = model.numeric_percepts.read(values)
    pooled = pooled_over_readings(field, owner)
    A = model.numeric_percepts.dimensions
    kinds = {}
    for concept, percept, negated in _concept_alloc_of(owner).order_zero_members:
        cid = owner.concept_id_at_row(concept)
        if not negated and cid is not None:
            kind = ('value' if basic.is_value(percept) else 'place' if percept in basic.place_rows
                    else 'conjunction' if percept in basic.conjunctions else 'other')
            kinds[int(cid)] = kind
    out = {}
    for c, cid in enumerate(field.concept_ids.tolist()):
        if cid in kinds:
            out[f'{kinds[cid]} {cid}'] = dict(at_places=field.evidence[c, :, :A, 0].tolist(),
                                              at_bracket=field.evidence[c, :, A, 0].tolist(),
                                              as_the_readout_reads_it=pooled[c, :, 0].tolist())
    print(dict(part_sigma=basic.part_sigma, things=basic.things, readout=out))
    assert basic.part_sigma == 1 and basic.things == {}
    value = [c for c, cid in enumerate(field.concept_ids.tolist()) if kinds.get(cid) == 'value']
    place = [c for c, cid in enumerate(field.concept_ids.tolist()) if kinds.get(cid) == 'place']
    assert value and place
    for c in value:
        assert bool(field.evidence[c, :, :A, 0].any()) and not bool(pooled[c].any())
    for c in place:
        torch.testing.assert_close(pooled[c, :, 0], field.evidence[c, :, :, 0].amax(-1))
    # A run of nine places, under an aperture of three and of nothing.
    image_model, *_ = _build_model('data/MM_image.xml')
    run = [.2, .3, .4, .5, .6, .5, .4, .3, .2]
    parts = {}
    for sigma in (3, 0):
        table = _table(monkeypatch, intensityPartSigma=sigma)
        bank, things = _image_bank(image_model, table, _run_image(run))
        (thing,), = things.tolist()
        offsets = [offset for offset, _ in table.things[thing]]
        centre = [m / 2 for m in map(max, zip(*offsets))]
        parts[sigma] = dict(places=len(offsets), offsets=offsets, centre=centre,
                            felt=[round(float(table.felt([p])[0]), 4) for _, p in table.things[thing]],
                            weights=[round(float(w), 4) for w in table.weights(offsets)])
        if sigma:
            assert all(abs(a - c) < sigma for offset in offsets for a, c in zip(offset, centre))
            assert len(offsets) == 2 * sigma - 1
        else:
            assert len(offsets) == len(run) and all(w == 1 for w in parts[sigma]['weights'])
    print(dict(parts=parts))


# -- A5: XOR --------------------------------------------------------------------------

def test_a5_xor():
    """test/numeric_xor.py's certificate as it is, with ``pooled`` gone: the
    readout keeps the two inputs apart by the aperture. The stopping gate and
    test_concept_output.py whole are certified by the run that names them."""
    from numeric_xor import certificate
    import NumericPercepts
    model, prediction, error = certificate()
    basic = model._concept_owner().basic_percepts
    print(dict(pooled_in_the_numeric_stem=hasattr(NumericPercepts.NumericPercepts, 'pooled'),
               answers=prediction.detach().tolist(), mse=error, conjunctions=basic.conjunctions))
    assert not hasattr(NumericPercepts.NumericPercepts, 'pooled')


# -- A6: a Sigma over the rows beneath it ------------------------------------------

def test_a6_a_sigma_over_the_rows_beneath_it(tmp_path, monkeypatch):
    """No concept is promoted alone to a symbol: after a word and a pixel
    have each recurred, neither concept has a symbol made for it alone. A
    concept of order one given an entry on a word's concept, on an
    intensity's concept (each through a thing of that one concept, read as
    the product of what it holds, through the Not) and on a basic percept's
    row (from the Not of order zero) reads each: the value where the entry is
    positive, the not-value where it is negative. The entry on the basic
    percept's row makes no symbol's row."""
    from FieldProjection import close
    from Layers import NotLayer3
    from reading_fixtures import use_eager_reading
    from test_item16_scene_field import known
    from test_item16_naming import naming_model, word_of, value_of, symbols_of, WORD, PIXEL
    model = naming_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        known(model, [dict(text=WORD, scene=None)])
        known(model, [dict(text='', scene=[PIXEL])])
        owner = model._concept_owner()
        basic = owner.basic_percepts
        store = owner._concept_allocator.layer(0)
        table = owner._concept_allocator.symbol_table()
        word, value = word_of(model, WORD), value_of(model, PIXEL[3])
        percept, = [p for p in owner.concept_percepts(owner._concept_row_of(value)) if basic.is_value(p)]
        handles = {name: [table.handles[s] for s in symbols_of(model, cid)] for name, cid in
                   (('word', word), ('value', value))}
        assert not any(h[0] == 'alone' for found in handles.values() for h in found)
        assert not handles['value']
        # A thing of one concept for each: the word's kind's symbol where one
        # holds it alone, else one written here (none is made by a rule).
        things = {}
        for name, cid in (('word', word), ('value', value)):
            alone = [s for s in symbols_of(model, cid) if table.members(s) == [cid]]
            things[name] = alone[0] if alone else owner.nominate_symbol(('certificate', cid), cid)
        symbols = len(table)
        order_one = owner.new_concept(order=1)
        row = owner._csw_concept_row(1, order_one)
        rows = [owner._concept_row_of(word), owner._concept_row_of(value), row]
        lanes = {'for': (.76, 0.), 'against': (0., .76)}
        read = {}
        for target in ('word', 'value', 'percept'):
            for sign in (1., -1.):
                for side, (f, a) in lanes.items():
                    # Only this entry: every other written at nothing.
                    for (r, column), position in store._index.items():
                        if r == row:
                            store.values.data[position] = 0.
                    store.percept_entries.clear()
                    if target == 'percept':
                        store.percept_entries[(row, percept)] = sign
                    else:
                        column = store.column_of_thing(('sym', things[target]))
                        position = store.add_edge(row, column, sign)
                        store.values.data[position] = sign
                    evidence = torch.zeros(3, 1, 1, 2)
                    given = torch.tensor([f, a])
                    if target != 'percept':
                        evidence[0 if target == 'word' else 1, 0, 0] = given
                    percepts = (given.reshape(1, 1, 1, 2), [percept])
                    out = close(owner, rows, evidence, percepts)[2, 0, 0]
                    read[(target, sign, side)] = tuple(round(float(x), 4) for x in out)
        print(dict(word=word, value=value, percept=percept, handles=handles, things=things,
                   order_one=(order_one, row), read={f'{t} entry {s:+.0f} at {d}': x for (t, s, d), x in read.items()},
                   symbols_with_the_things=symbols, symbols_after_the_entries=len(table), percept_entries=dict(store.percept_entries)))
        for (target, sign, side), (f, a) in read.items():
            # The value where the entry is positive, the not-value where negative.
            expected_for = (sign > 0) == (side == 'for')
            assert (f > 0 and a == 0) if expected_for else (a > 0 and f == 0), (target, sign, side, f, a)
        assert len(table) == symbols
    finally:
        model.End()


# -- A7: what stood -------------------------------------------------------------------

#: The tests of this item ported where a point of §46.2 made an assertion
#: false, by the point: old and new are in the return.
PORTED = {
    'test_item16_values.py::test_v3_the_pi_over_values': 1,
    'test_item16_values.py::test_v9_a_region_is_nominated_on_recurrence': 1,
    'test_item16_values.py::test_v10_atomic_over_digits': 4,
    'test_item16_values.py::test_v11_reconstituted_to_the_centroids': 1,
    'test_item16_values.py::test_v12_the_corpus_values_fall_in_rows_of_their_own': 1,
    'test_item16_things.py::test_e6_the_limit': 4,
    'test_item16_things.py (_bank, _read, E8)': 4,
    'test_item16_relations.py::bank_of': 4,
    'test_item16_naming.py::test_n2_each_known_alone': 5,
    'test_item17_symbols.py::test_71_no_symbol_is_made_for_a_concept_and_every_symbol_is_mtms': 5,
}


def test_a7_what_stood():
    """V, S, C, E, F, G and N of this item are certified by the run that names
    them; the ports are listed. No older form is kept: the cuts given
    beforehand, the old key and the numeric stem's pooling are named nowhere in
    the source or the tests."""
    gone = ('intensityWhereWidth', 'where_width', 'self.pooled(', 'numeric_percepts.pooled', '.cut(', 'centroids[')
    found = []
    for folder in ('bin', 'test', 'data'):
        for name in sorted(os.listdir(os.path.join(_ROOT, folder))):
            if name.endswith(('.py', '.xml', '.xsd')) and name != os.path.basename(__file__):
                with open(os.path.join(_ROOT, folder, name), errors='replace') as handle:
                    text = handle.read()
                found += [(folder, name, word) for word in gone if word in text]
    print(dict(ported=PORTED, older_forms_named=found))
    assert not found


# -- A8: the corpus (a measure) -------------------------------------------------------

def test_a8_the_corpus(tmp_path, monkeypatch):
    """Categories 1 and 2 of the felt world, their train split, on the scene
    model's widths: for each value, the number of atoms' rows its steps make.
    Reported."""
    import felt_world_corpus as felt
    from BasicPercepts import BasicPercepts
    from test_item16_scene_field import scene_model
    model = scene_model(tmp_path, monkeypatch)
    try:
        basic = model._concept_owner().basic_percepts
        table = BasicPercepts(None, basic.identity_width, basic.property_width, 1 << 15)
        table.bind(None)
        report, both = {}, {}
        for category in (1, 2):
            steps = [pixel[3] for d in felt.load(category, 'train') for s in d.scenes if s for pixel in s]
            values = torch.tensor(steps, dtype=table.levels.dtype)[None] / 255
            rows = table.perceive_intensities(values)[0].tolist()
            by_value = {}
            for step, row in zip(steps, rows):
                by_value.setdefault(felt.value_of(step), {}).setdefault(step, set()).add(row)
                both.setdefault(felt.value_of(step), {}).setdefault(step, set()).add(row)
            report[category] = {d: dict(steps=len(found), rows=len(set().union(*found.values())))
                                for d, found in sorted(by_value.items())}
        report['1 and 2'] = {d: dict(steps=len(found), rows=len(set().union(*found.values())))
                             for d, found in sorted(both.items())}
        print(dict(levels=len(table.levels), widths=(table.identity_width, table.property_width), by_category=report))
        assert report[1] and report[2]
        for found in report.values():
            assert all(entry['rows'] == entry['steps'] for entry in found.values())
    finally:
        model.End()
