"""Values in an image's bank.

The certificates V1 to V7 of §20.8, V10 of §30.1, and V3, V9, V11 and V12
of §36.2 of doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing
what it measured. A basic percept of intensity is a row over a part and a
whole taken from the alphabet of intensity, a level of it: it identifies a
region and has a centroid, its level, whatever its place; a reader of intensities takes its
input in 0..1 as it is; the numeric stem's input is an image's bank, read by
the Pi over values as a form is read; a place is told by its wires, by a
place's row that keeps the value's ``.what``; one Not reads the rows of a
pass. V7's tests of the image configurations and V8 are certified by the run
that names them; V7's measure of the image's read is here.
"""
import ast
import inspect
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
# Values on a grid of the line, none chosen: every k / 37 for k = 1..37.
_GRID = [k / 37 for k in range(1, 38)]


def _xor():
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_xor.xml')
    return model


def _source(name):
    with open(os.path.join(_BIN, name)) as handle:
        return ast.parse(handle.read())


def _names_in(tree, function):
    """The attributes and names a function (by its name) uses."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return ({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                    | {n.id for n in ast.walk(node) if isinstance(n, ast.Name)})
    raise AssertionError(f'{function} not found')


def test_v1_one_table():
    """The same value at two places is one row; its code is the bins at or
    below it, in order, in both halves; partOf on two values' codes is
    less-than-or-equal; nothing takes a place and no key holds one."""
    from BasicPercepts import BasicPercepts
    from Layers import Ops
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    field = model.numeric_percepts.read(torch.tensor([[[.5], [.5]], [[.5], [0.]], [[0.], [.5]]]))
    rows = field.percepts.tolist()
    one = {row for line in rows for row in line if row >= 0}
    print(dict(percepts=rows, rows_of_the_value=sorted(one)))
    assert len(one) == 1
    row, = one
    felt = torch.tensor(.5)
    code = basic.code(row)
    print(dict(code=code.tolist()))
    torch.testing.assert_close(code, torch.cat((basic.value_code(felt, basic.identity_width),
                                                basic.value_whole(felt))))
    half = code[:basic.identity_width]
    ones = int(half.sum())
    assert half[:ones].eq(1).all() and half[ones:].eq(0).all()
    # The wholes half: its level's group alone.
    assert int(code[basic.identity_width:basic.width].sum()) == 1
    # partOf is less-than-or-equal in the parts half, over every ordered
    # pair of the grid.
    table = BasicPercepts(None, 64, 64, 64)
    table.bind(None)
    grid = [table.perceive_value(torch.tensor(v)) for v in _GRID]
    codes = table.lookup_rows(grid)[:, :table.identity_width]
    bins = table.value_bin(torch.tensor(_GRID), 64)
    agree = 0
    for a in range(len(grid)):
        for b in range(len(grid)):
            assert bool(Ops.partOf(codes[a], codes[b])) == bool(bins[a] <= bins[b]), (a, b)
            agree += 1
    print(dict(pairs=agree, values=len(grid), bins=sorted(set(bins.tolist()))))
    # Nothing draws a value's coordinates by its place, and no key holds one.
    assert not hasattr(BasicPercepts, 'value_coordinates') and not hasattr(basic, 'places')
    for name in ('value_code', 'perceive_value', 'reserve', 'bin_of', '_boundary_codes'):
        assert 'place' not in inspect.signature(getattr(BasicPercepts, name)).parameters, name
    keys = {key for key in basic.index if key[0] not in ('place', 'conjunction')}
    assert keys and all(len(key) == 2 for key in keys), keys
    print(dict(keys=sorted(map(str, keys))))


def test_v2_input_in_0_to_1():
    """A reader of intensities takes a value in 0..1 as it is and gives
    back the centroid of its basic percept through the field once a concept
    reads it there (nominated after the image has recurred as often as
    nominates a form); on the first presentations, when no concept reads a value's row
    or its place's row, and wherever the field reads nothing, it gives
    nothing back. It refuses a value below 0, above 1 or not a number; no
    reader applies the logarithmic chart."""
    from dataclasses import replace
    from recon_bench import _build_model
    from NumericPercepts import intensities
    numbers = torch.tensor([0., *_GRID])
    assert torch.equal(intensities(numbers), numbers)
    refused = {}
    for bad in (-1e-6, 1 + 1e-6, float('nan')):
        with pytest.raises(ValueError, match='takes numbers from 0 to 1') as raised:
            intensities(torch.tensor([.5, bad]))
        refused[bad] = str(raised.value)
    print(dict(refused=refused))
    model = _xor()
    with pytest.raises(ValueError, match='takes numbers from 0 to 1'):
        model.numeric_percepts.read(torch.tensor([[[2.], [0.]]]))
    # An image of every value of the grid and nothing between them. On its
    # first presentation to a fresh model no concept reads it, and nothing
    # is given back; after it has recurred, each place is given back the
    # centroid of the basic percept its value falls in.
    image, *_ = _build_model('data/MM_image.xml')
    given = torch.tensor(([0.] + _GRID) * 22)[:784].reshape(1, 28, 28)
    first = image.numeric_percepts.reverse(image.numeric_percepts.read(given))
    print(dict(first_presentation_greatest=float(first.abs().max())))
    assert not bool(first.any())
    _recur(image, given)
    field = image.numeric_percepts.read(given)
    back = image.numeric_percepts.reverse(field)
    basic = image._concept_owner().basic_percepts
    centroid = torch.where(field.percepts >= 0, basic.felt(field.percepts.clamp_min(0)), 0.).reshape(given.shape)
    print(dict(given=given[0, 0, :6].tolist(), back=back[0, 0, :6].tolist(),
               centroids=centroid[0, 0, :6].tolist(), places=len(basic.place_rows)))
    assert torch.equal(back.to(given), centroid.to(given))
    # The way back goes through the field: where it reads nothing, nothing
    # is given back.
    silent = image.numeric_percepts.reverse(replace(field, evidence=torch.zeros_like(field.evidence)))
    print(dict(silent_greatest=float(silent.abs().max())))
    assert not bool(silent.any())
    # The chart is a step before a reader: neither the numeric stem nor the
    # basic percepts name it.
    named = {(file, f): 'feeling' in _names_in(_source(file), f)
             for file, f in (('NumericPercepts.py', 'read'), ('NumericPercepts.py', 'reverse'),
                             ('BasicPercepts.py', 'read_values'), ('BasicPercepts.py', 'value_bank'),
                             ('BasicPercepts.py', 'perceive_value'))}
    print(dict(chart_named=named))
    assert not any(named.values())


def _bank(model, values):
    """The numeric stem's bank of ``values`` ``[B, 2]`` and the value rows."""
    from dataclasses import replace
    owner = model._concept_owner()
    basic = owner.basic_percepts
    B, A = values.shape
    places = model.where_registry.intervals('numeric', torch.arange(A).expand(B, -1))
    rows = {float(v): basic.perceive_intensity(v) for v in values.unique() if v > 0}
    bank = basic.value_bank(values, places, registry=model.where_registry,
                            when_encoding=model.when_encoding, time=model.present())
    order = sorted(rows.values())
    return replace(bank, rows=tuple(order), codes=basic.bank_codes(order)), rows


def test_v3_the_pi_over_values():
    """On an image's bank, a basic percept of intensity reads for at an
    occurrence of its level, and nothing above it, below it or where nothing
    is (item 1.4, §10.2: no evidence against at order zero): the first Pi's test of
    part and whole on two lanes, the level and the occurrence compared by
    their values. Its row is keyed by its part and its whole taken from the
    alphabet, each value of the image falls to a level of its own, and its
    entries are in the Pi of the value space, none in the first Pi."""
    model = _xor()
    basic = model._concept_owner().basic_percepts
    values = torch.tensor([[.1, .2], [.5, 0.], [.9, 1.], [.5, .2]])
    bank, rows = _bank(model, values)
    lanes = basic.read_values(bank)
    width = basic.identity_width
    bins = {v: int(basic.fall(torch.tensor(v))) for v in rows}
    print(dict(width=width, levels=bins, rows=rows))
    assert len(set(rows.values())) == 5
    B, A = values.shape
    read = {}
    for value, row in rows.items():
        r, k = bank.rows.index(row), bins[value]
        for b in range(B):
            for a in range(A):
                there = float(values[b, a])
                j = int(basic.fall(torch.tensor(there)))
                expected = (1., 0.) if there > 0 and j == k else (0., 0.)
                got = tuple(lanes[r, b, a].tolist())
                read[(k, there)] = got
                assert got == expected, (value, there, got)
    print(dict(read={f'level {k} at {t}': x for (k, t), x in sorted(read.items())}))
    keys = {row: basic._keys[row] for row in set(rows.values())}
    entries = {row: (basic.pi.row_members(row), basic.value_pi.row_members(row)) for row in keys}
    print(dict(keys=keys, entries_first_pi_and_value_pi=entries,
               cuts={row: basic.value_codes[[-ref - width - 1 for ref in (*key[0], *key[1])]].tolist()
                     for row, key in keys.items()}))
    for row, (parts, wholes) in keys.items():
        assert len(parts) == 1 and len(wholes) == 1
        assert entries[row] == ([], [('ps', parts), ('ws', wholes[0])])
        torch.testing.assert_close(basic.code(row), torch.cat((basic.part_codes(parts)[0], basic.whole_codes(wholes)[0])))
    # Read as a form is read, and the numeric stem reads its bank by it.
    assert 'held_at' in _names_in(_source('BasicPercepts.py'), 'value_events')
    assert 'value_events' in _names_in(_source('BasicPercepts.py'), 'read_values')
    used = _names_in(_source('NumericPercepts.py'), 'read')
    assert 'read_values' in used and 'read_bank' not in used
    print(dict(numeric_read_uses=sorted(used & {'read_values', 'read_bank', 'value_bank'})))


def _recur(model, values):
    """The image ``values`` read as often as nominates a form, in training."""
    owner = model._concept_owner()
    model.train()
    for _ in range(owner.concept_mint_count):
        model.forward(values)
    return owner, owner.basic_percepts


def test_v4_a_place_by_its_wires():
    """After an image has recurred as often as nominates a form, each place
    that held a value has a place's row that keeps that value's .what; the
    same value at two places is one value's row and two places' rows."""
    from dataclasses import replace
    from recon_bench import _build_model
    model = _xor()
    owner, basic = _recur(model, torch.tensor([[[1.], [1.]]]))
    registry = model.where_registry
    base = registry.slices['numeric'][0]
    places = {}
    for row, (plus, minus, what) in basic.place_rows.items():
        address = int(registry.decode(plus[None].to(registry.grid))[0])
        places[address - base] = row
        assert what is not None
    field = model.numeric_percepts.read(torch.tensor([[[1.], [1.]]]))
    value = int(field.percepts[0, 0])
    print(dict(places={a: r for a, r in sorted(places.items())}, value_row=value,
               percepts=field.percepts.tolist()))
    assert sorted(places) == [0, 1]
    assert set(field.percepts[0].tolist()) == {value}
    for address, row in places.items():
        torch.testing.assert_close(basic.place_rows[row][2], basic.bank_codes([value])[0, 0].cpu())
        assert owner.concept_of_basic_percept(row) is not None
    # Each place's row holds its own place and not the other.
    bank, _ = _bank(model, torch.tensor([[1., 1.]]))
    bank = replace(bank, rows=(places[0], places[1]), codes=basic.bank_codes([places[0], places[1]]))
    held = basic.parthood(bank)[:, 0]
    print(dict(place_rows_at_the_two_places=held.tolist()))
    assert held.tolist() == [[1., 0.], [0., 1.]]
    # An image: each lit place has a place's row keeping its value's .what.
    from image_corpus import load
    images, _ = load('shapes', 'test')
    image = images[:1]
    model, *_ = _build_model('data/MM_image.xml')
    owner, basic = _recur(model, image)
    registry = model.where_registry
    base = registry.slices['numeric'][0]
    lit = set((image.reshape(-1) > 0).nonzero().flatten().tolist())
    kept = {}
    for row, (plus, minus, what) in basic.place_rows.items():
        kept[int(registry.decode(plus[None].to(registry.grid))[0]) - base] = what
    value = basic.perceive_intensity(torch.tensor(1.))
    code = basic.bank_codes([value])[0, 0].cpu()
    print(dict(lit=len(lit), places=len(kept), values=sorted(set(image.reshape(-1).tolist()))))
    assert set(kept) == lit
    assert all(torch.equal(what, code) for what in kept.values())


def test_v5_xor_from_input_to_output():
    """test/numeric_xor.py's certificate: its error, its answers and its
    exact reconstruction as they are; its conjunction is over two places'
    rows (asserted there)."""
    from numeric_xor import certificate
    model, prediction, error = certificate()
    basic = model._concept_owner().basic_percepts
    (conjunction, members), = basic.conjunctions.items()
    print(dict(conjunction=conjunction, members=members,
               places=all(m in basic.place_rows for m in members), mse=error))
    assert all(m in basic.place_rows for m in members)


def test_v6_one_not():
    """A pass of reading sends its rows through order_zero_field once, and
    the rows it sends are the value's, the places' and their conjunction's."""
    model = _xor()
    owner = model._concept_owner()
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1)
    _recur(model, values)
    model.forward(values)
    calls = []
    original = owner.order_zero_field

    basic = owner.basic_percepts

    def counted(lanes, inputs, rows):
        # With the unit delay (architecture.perceptDelay) the inputs after
        # the rows are the rows one step back, ``capacity + row``: they are
        # not basic percept rows, so only the rows are counted.
        calls.append([r for r in inputs if r < int(basic.capacity)])
        return original(lanes, inputs, rows)
    owner.order_zero_field = counted
    try:
        model.forward(values)
    finally:
        del owner.order_zero_field
    kinds = [dict(values=sum(basic.is_value(r) for r in inputs),
                  places=sum(r in basic.place_rows for r in inputs),
                  conjunctions=sum(r in basic.conjunctions for r in inputs)) for inputs in calls]
    print(dict(calls=len(calls), rows=kinds))
    assert len(calls) == 1
    assert kinds[0]['values'] and kinds[0]['places'] == 2 and kinds[0]['conjunctions'] == 1


def test_v7_the_image_read_and_put_back():
    """The image configuration reads a shapes image, once it has recurred
    as often as nominates a form, and puts it back exactly; the time and the
    greatest memory of each are printed (the tests of the image
    configurations are certified by the run)."""
    from recon_bench import _build_model
    from image_corpus import load
    from corpus_audit import peak_bytes
    images, _ = load('shapes', 'test')
    sample = images[:1]
    model, *_ = _build_model('data/MM_image.xml')
    _recur(model, sample)
    started = time.perf_counter()
    model.forward(sample)
    forward_seconds = time.perf_counter() - started
    forward_peak = peak_bytes()
    understood = model.understand(sample)
    started = time.perf_counter()
    rebuilt, _ = model.reverseReconstruct(understood)
    reverse_seconds = time.perf_counter() - started
    print(dict(forward_seconds=round(forward_seconds, 4), reverse_seconds=round(reverse_seconds, 4),
               forward_peak_bytes=forward_peak, reverse_peak_bytes=peak_bytes(),
               lit=int((sample > 0).sum())))
    torch.testing.assert_close(rebuilt, sample, rtol=0, atol=0)


def test_v9_a_region_is_nominated_on_recurrence():
    """A level met once has no row of its own (its row is reserved, not
    promoted, and has no concept); met as often as nominates a form, by a
    different value each time, it has one row, promoted, with its concept,
    whatever exact values fell to it, and its centroid is their mean; no
    row is kept for an exact value."""
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    model.train()
    n = owner.concept_mint_count
    width = basic.identity_width
    # n values about the middle level, within half a step of it, none chosen.
    level = int(basic.fall(torch.tensor(.5)))
    step = float(basic.levels[1] - basic.levels[0])
    met = [float(basic.levels[level]) + ((i + 1) / (n + 1) - .5) * step / 2 for i in range(n)]
    seen = []
    for time, value in enumerate(met):
        model.forward(torch.tensor([[[value], [0.]]]))
        values = [r for r in range(len(basic.index)) if basic.is_value(r)]
        seen.append(dict(time=time + 1, value=round(value, 4), value_rows=values,
                         promoted=[bool(basic.promoted[r]) for r in values],
                         concept=[owner.concept_of_basic_percept(r) for r in values]))
    row, = seen[-1]['value_rows']
    seat = int(basic.level_seats[level])
    cuts = ((seat,), (seat,))
    print(dict(mint_count=n, seen=seen, key=basic._keys[row], seats_of_the_level=cuts,
               centroid=float(basic.felt([row])[0]), mean=sum(float(torch.tensor(v)) for v in met) / n))
    assert seen[0]['value_rows'] == [row] and seen[0]['promoted'] == [False] and seen[0]['concept'] == [None]
    assert all(s['value_rows'] == [row] and s['promoted'] == [False] for s in seen[:-1])
    assert seen[-1]['promoted'] == [True] and seen[-1]['concept'][0] is not None
    torch.testing.assert_close(basic.felt([row])[0], torch.tensor(met).mean().to(basic.levels))
    # Its key is the level's part and whole, and no value's own.
    assert basic._keys[row] == cuts


def test_v10_atomic_over_digits():
    """On an XOR configuration no reading wider than one place reads a
    value's row or its concept, the readout does not pool a value's concept
    over the places, and no row but the conjunction spans two places."""
    import xml.etree.ElementTree as ElementTree
    root = os.path.dirname(_BIN)
    widths = {name: ElementTree.parse(os.path.join(root, 'data', name)).getroot().find(
                  'architecture/intensityPartSigma').text
              for name in ('model.xml', 'MM_xor.xml', 'MM_xor_fixture.xml', 'XOR_exact.xml')}
    print(dict(intensityPartSigma=widths))
    assert widths == {'model.xml': '0', 'MM_xor.xml': '1', 'MM_xor_fixture.xml': '1', 'XOR_exact.xml': '1'}
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    assert basic.part_sigma == 1
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    _recur(model, values.unsqueeze(-1))
    # One presentation more nominates the conjunction (as in V6).
    model.forward(values.unsqueeze(-1))
    model.eval()
    field = model.numeric_percepts.read(values.unsqueeze(-1))
    B, A = values.shape
    wide = (field.where[..., 1] - field.where[..., 0]) > 1                     # [B, K]
    assert bool(wide[:, A].all()) and not bool(wide[:, :A].any())
    # The value's row, read by the Pi over values.
    value = basic.perceive_intensity(torch.tensor(1.))
    rows = [r for r in range(len(basic.index)) if bool(basic.promoted[r])]
    assert value in rows
    bank, _ = _bank(model, values)
    from dataclasses import replace
    bank = replace(bank, rows=tuple(rows), codes=basic.bank_codes(rows))
    lanes = basic.read_values(bank)
    at_places = lanes[rows.index(value), :, :A, 0]
    print(dict(value_row_at_places=at_places.tolist(), value_row_at_bracket=lanes[rows.index(value), :, A].tolist()))
    assert torch.equal(at_places, values)
    assert not bool(lanes[rows.index(value)][wide].any())
    # Its concept, in the field and as the readout reads it.
    concept = owner.concept_id_at_row(owner.concept_of_basic_percept(value))
    c = field.concept_ids.tolist().index(int(concept))
    evidence = field.evidence[c, ..., 0].detach()
    from Spaces import pooled_over_readings
    pooled = pooled_over_readings(field, owner)[c, :, 0].detach()
    print(dict(value_concept_at_places=evidence[:, :A].tolist(), at_bracket=evidence[:, A].tolist(),
               as_the_readout_reads_it=pooled.tolist()))
    assert torch.equal(evidence[:, :A] > 0, values > 0)
    assert not bool(evidence[wide].any()) and not bool(pooled.any())
    # Every row is a value's, a place's or a conjunction; a place's row
    # holds one place of the box, and only the conjunction spans two.
    wires = model.where_registry.band(field.where[0, :A, 0]).detach().cpu()
    spans = {}
    for row in range(len(basic.index)):
        if row in basic.place_rows:
            plus, minus, _what = basic.place_rows[row]
            spans[row] = ('place', int(((wires >= plus - 1e-6) & (1 - wires >= minus - 1e-6)).all(-1).sum()))
        elif row in basic.conjunctions:
            spans[row] = ('conjunction', len(basic.conjunctions[row]))
        else:
            assert basic.is_value(row), row
            spans[row] = ('value', 1)
    print(dict(spans=spans))
    assert all(n == 1 for kind, n in spans.values() if kind != 'conjunction')
    assert [n for kind, n in spans.values() if kind == 'conjunction'] == [2]


def test_v11_reconstituted_to_the_centroids():
    """After its values and places are known an image comes back to its
    percepts' centroids, from the basic percepts its concepts read for:
    each value at the level it falls to, within half a step, and two values
    of one level as one (item 1.4, §14.3); with the field's evidence at
    nothing, nothing comes back; the way back reads no value percepts kept
    beside the field."""
    from dataclasses import replace
    from recon_bench import _build_model
    image, *_ = _build_model('data/MM_image.xml')
    basic = image._concept_owner().basic_percepts
    width = basic.identity_width
    # One value alone at a level near each end of the line, two at the
    # middle level, each a quarter of a step from it.
    step = float(basic.levels[1] - basic.levels[0])
    middle = float(basic.levels[int(basic.fall(torch.tensor(.5)))])
    alone, shared = [1 / 6 / 2 + 0 / 6, 1 / 6 / 2 + 5 / 6], [middle - step / 4, middle + step / 4]
    regions = {v: int(basic.fall(torch.tensor(v))) for v in alone + shared}
    assert len({regions[v] for v in alone + shared}) == 3 and regions[shared[0]] == regions[shared[1]]
    given = torch.tensor(([0.] + [alone[0], shared[0], shared[1], alone[1]]) * 157)[:784].reshape(1, 28, 28)
    _recur(image, given)
    field = image.numeric_percepts.read(given)
    back = image.numeric_percepts.reverse(replace(field, percepts=None)).to(given)
    mean = given[(given == shared[0]) | (given == shared[1])].double().mean()
    by_value = {round(v, 4): back[given == v].unique().tolist() for v in alone + shared}
    print(dict(width=width, regions={round(v, 4): r for v, r in regions.items()}, back_by_value=by_value,
               mean_of_the_shared=float(mean)))
    # Item 1.4, §14.3: a value is put back at the level it falls to, which
    # moves toward what falls to it very slowly; so it comes back within
    # half a step, and two values of one level come back as one.
    for v in alone:
        assert len(back[given == v].unique()) == 1
        assert float((back[given == v] - given[given == v]).abs().max()) <= step / 2
    middle = back[(given == shared[0]) | (given == shared[1])]
    assert len(middle.unique()) == 1
    assert abs(float(middle[0]) - float(mean)) <= step / 2
    assert float(middle[0]) == pytest.approx(float(basic.levels[regions[shared[0]]]), abs=1e-6)
    assert not bool(back[given == 0].any())
    silent = image.numeric_percepts.reverse(replace(field, evidence=torch.zeros_like(field.evidence), percepts=None))
    print(dict(silent_greatest=float(silent.abs().max())))
    assert not bool(silent.any())
    names = _names_in(_source('NumericPercepts.py'), 'reverse')
    print(dict(reverse_names_percepts='percepts' in names))
    assert 'percepts' not in names


def test_v12_the_corpus_values_fall_in_rows_of_their_own(tmp_path, monkeypatch):
    """On the felt world's model, the steps of each of the corpus's nine
    values fall in rows of their own, which no other value's step falls in,
    and a value's reserved step falls in a row of its own, a level of the
    alphabet no other step fell to."""
    from felt_world_corpus import VALUES, FREE_STEPS, reserved_step
    from test_item16_scene_field import scene_model
    model = scene_model(tmp_path, monkeypatch)
    try:
        basic = model._concept_owner().basic_percepts
        felt = lambda steps: torch.tensor(steps / 255)
        rows = {d: {basic.perceive_intensity(felt(s)) for s in FREE_STEPS[d]} for d in VALUES}
        before = len(basic.index)
        reserved = {d: basic.perceive_intensity(felt(reserved_step(d))) for d in VALUES}
        print(dict(width=basic.identity_width, rows={d: sorted(r) for d, r in rows.items()},
                   reserved={d: (reserved_step(d), r) for d, r in reserved.items()},
                   rows_made_by_reserved=len(basic.index) - before))
        made = {r for found in rows.values() for r in found}
        for d in VALUES:
            assert not any(rows[d] & rows[e] for e in VALUES if e != d), d
            assert reserved[d] not in made, d
        assert len(basic.index) == before + len(VALUES)
    finally:
        model.End()
