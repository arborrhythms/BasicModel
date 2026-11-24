"""Values in an image's bank.

The certificates V1 to V7 of §20.8 and V3 (replaced) and V9 to V11 of §30.1
of doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured. A value is one row of one table of values, whatever its place; a
reader of intensities takes its input in 0..1 as it is; the numeric stem's
input is an image's bank, read by the Pi over values; a place is told by its
wires, by a place's row that keeps the value's ``.what``; one Not reads the
rows of a pass. V7's tests of the image configurations and V8 are certified
by the run that names them; V7's measure of the image's read is here.
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
                                                basic.value_code(felt, basic.property_width))))
    for half in (code[:basic.identity_width], code[basic.identity_width:basic.width]):
        ones = int(half.sum())
        assert half[:ones].eq(1).all() and half[ones:].eq(0).all()
    # partOf is less-than-or-equal, over every ordered pair of the grid.
    table = BasicPercepts(None, 64, 64, 64)
    table.bind(None)
    grid = [table.perceive_value(torch.tensor(v)) for v in _GRID]
    codes = table.lookup_rows(grid)
    bins = BasicPercepts.value_bin(torch.tensor(_GRID), 64)
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
    """A reader of intensities takes a value in 0..1 as it is and gives it
    back exactly through the field once a concept reads it there (a place's
    row, nominated after the image has recurred as often as nominates a
    form); on the first presentations, when no concept reads a value's row
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
    # is given back; after it has recurred, each value is given back exactly.
    image, *_ = _build_model('data/MM_image.xml')
    given = torch.tensor(([0.] + _GRID) * 22)[:784].reshape(1, 28, 28)
    first = image.numeric_percepts.reverse(image.numeric_percepts.read(given))
    print(dict(first_presentation_greatest=float(first.abs().max())))
    assert not bool(first.any())
    _recur(image, given)
    field = image.numeric_percepts.read(given)
    back = image.numeric_percepts.reverse(field)
    print(dict(given=given[0, 0, :6].tolist(), back=back[0, 0, :6].tolist(),
               places=len(image._concept_owner().basic_percepts.place_rows)))
    assert torch.equal(back.to(given), given)
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
    rows = {float(v): basic.perceive_value(v) for v in values.unique() if v > 0}
    bank = basic.value_bank(values, places, registry=model.where_registry,
                            when_encoding=model.when_encoding, time=model.present())
    order = sorted(rows.values())
    return replace(bank, rows=tuple(order), codes=basic.bank_codes(order)), rows


def test_v3_the_pi_over_values():
    """On an image's bank, a value's row reads for at an occurrence of an
    equal or greater value and nothing at a lesser one or where nothing is,
    a lesser value in the same bin among them; its against lane is nothing
    everywhere: numerical parthood. It is a function of the value space and
    names nothing of the first Pi over forms."""
    from BasicPercepts import BasicPercepts
    model = _xor()
    basic = model._concept_owner().basic_percepts
    values = torch.tensor([[.5, .75], [.25, 0.], [.5, .5], [1., .25], [.3, .25]])
    bank, rows = _bank(model, values)
    lanes = basic.read_values(bank, values, where_width=model.numeric_percepts.where_width)
    B, A = values.shape
    read = {}
    for value, row in rows.items():
        r = bank.rows.index(row)
        for b in range(B):
            for a in range(A):
                there = float(values[b, a])
                expected = (1., 0.) if there > 0 and there >= value else (0., 0.)
                got = tuple(lanes[r, b, a].tolist())
                read[(value, there)] = got
                assert got == expected, (value, there, got)
    bins = BasicPercepts.value_bin(torch.tensor([.25, .3]), basic.identity_width).tolist()
    print(dict(width=basic.identity_width, bins_of_25_and_30=bins,
               read={f'{v} at {t}': x for (v, t), x in sorted(read.items())}))
    low, high = float(torch.tensor(.25)), float(torch.tensor(.3))
    assert bins[0] == bins[1] and read[(high, low)] == (0., 0.) and read[(low, high)] == (1., 0.)
    assert not bool(lanes[..., 1].any())
    # The Pi over values names nothing of the first Pi over forms, and the
    # numeric stem reads its bank by it.
    for function in ('read_values', 'parthood'):
        assert 'held_at' not in _names_in(_source('BasicPercepts.py'), function), function
    used = _names_in(_source('NumericPercepts.py'), 'read')
    assert 'read_values' in used and 'read_bank' not in used
    print(dict(numeric_read_uses=sorted(used & {'read_values', 'read_bank', 'value_bank'}),
               read_values_names_held_at=False))


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
    held = basic.parthood(bank, torch.tensor([[1., 1.]]))[:, 0]
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
    value = basic.perceive_value(torch.tensor(1.))
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

    def counted(lanes, inputs, rows):
        calls.append(list(inputs))
        return original(lanes, inputs, rows)
    owner.order_zero_field = counted
    try:
        model.forward(values)
    finally:
        del owner.order_zero_field
    basic = owner.basic_percepts
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


def test_v9_a_value_is_nominated_on_recurrence():
    """A value met once has no concept and its row is not promoted; met as
    often as nominates a form, its row is promoted and has its concept."""
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    model.train()
    image = torch.tensor([[[.5], [0.]]])
    seen = []
    for time in range(owner.concept_mint_count):
        model.forward(image)
        row = basic.perceive_value(torch.tensor(.5))
        seen.append(dict(time=time + 1, promoted=bool(basic.promoted[row]),
                         concept=owner.concept_of_basic_percept(row)))
    print(dict(mint_count=owner.concept_mint_count, seen=seen))
    assert not seen[0]['promoted'] and seen[0]['concept'] is None
    assert all(not s['promoted'] and s['concept'] is None for s in seen[:-1])
    assert seen[-1]['promoted'] and seen[-1]['concept'] is not None


def test_v10_atomic_over_digits():
    """On an XOR configuration no reading wider than one place reads a
    value's row or its concept, the readout does not pool a value's concept
    over the places, and no row but the conjunction spans two places."""
    import xml.etree.ElementTree as ElementTree
    root = os.path.dirname(_BIN)
    widths = {name: ElementTree.parse(os.path.join(root, 'data', name)).getroot().find(
                  'architecture/intensityWhereWidth').text
              for name in ('model.xml', 'MM_xor.xml', 'MM_xor_fixture.xml', 'XOR_exact.xml')}
    print(dict(intensityWhereWidth=widths))
    assert widths == {'model.xml': '0', 'MM_xor.xml': '1', 'MM_xor_fixture.xml': '1', 'XOR_exact.xml': '1'}
    model = _xor()
    assert model.numeric_percepts.where_width == 1
    owner = model._concept_owner()
    basic = owner.basic_percepts
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
    value = basic.perceive_value(torch.tensor(1.))
    rows = [r for r in range(len(basic.index)) if bool(basic.promoted[r])]
    assert value in rows
    bank, _ = _bank(model, values)
    from dataclasses import replace
    bank = replace(bank, rows=tuple(rows), codes=basic.bank_codes(rows))
    lanes = basic.read_values(bank, values, where_width=1)
    at_places = lanes[rows.index(value), :, :A, 0]
    print(dict(value_row_at_places=at_places.tolist(), value_row_at_bracket=lanes[rows.index(value), :, A].tolist()))
    assert torch.equal(at_places, values)
    assert not bool(lanes[rows.index(value)][wide].any())
    # Its concept, in the field and as the readout reads it.
    concept = owner.concept_id_at_row(owner.concept_of_basic_percept(value))
    c = field.concept_ids.tolist().index(int(concept))
    evidence = field.evidence[c, ..., 0].detach()
    pooled = model.numeric_percepts.pooled(field).evidence[c].detach().amax(1)[..., 0]
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


def test_v11_reconstituted_from_what_is_read_for():
    """After its values and places are known an image comes back exactly,
    at graded values, from the values' rows its concepts read for, placed
    by their overlap; with the field's evidence at nothing, nothing comes
    back; the way back reads no value percepts kept beside the field."""
    from dataclasses import replace
    from recon_bench import _build_model
    image, *_ = _build_model('data/MM_image.xml')
    given = torch.tensor(([0.] + _GRID) * 22)[:784].reshape(1, 28, 28)
    _recur(image, given)
    field = image.numeric_percepts.read(given)
    back = image.numeric_percepts.reverse(replace(field, percepts=None))
    basic = image._concept_owner().basic_percepts
    bins = sorted(set(basic.value_bin(torch.tensor(_GRID), basic.identity_width).tolist()))
    print(dict(width=basic.identity_width, bins_of_the_grid=bins, given=given[0, 1, :6].tolist(),
               back=back[0, 1, :6].tolist(), exact=bool(torch.equal(back.to(given), given))))
    assert torch.equal(back.to(given), given)
    silent = image.numeric_percepts.reverse(replace(field, evidence=torch.zeros_like(field.evidence), percepts=None))
    print(dict(silent_greatest=float(silent.abs().max())))
    assert not bool(silent.any())
    names = _names_in(_source('NumericPercepts.py'), 'reverse')
    print(dict(reverse_names_percepts='percepts' in names))
    assert 'percepts' not in names
