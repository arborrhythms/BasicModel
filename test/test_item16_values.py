"""Item 1.6, round one, step B: values in an image's bank.

The certificates V1 to V7 of §20.8 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
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
    """On an image's bank, a value's row reads for at an occurrence of its
    own bin, against at a greater value, and nothing at a lesser value or
    where nothing is: :func:`held_at` on a line. It is a function of the
    value space, apart from the first Pi over forms."""
    from BasicPercepts import BasicPercepts, held_at
    model = _xor()
    basic = model._concept_owner().basic_percepts
    values = torch.tensor([[.5, .75], [.25, 0.], [.5, .5], [1., .25]])
    bank, rows = _bank(model, values)
    lanes = basic.read_values(bank)                                   # [R, B, readings, 2]
    bins = BasicPercepts.value_bin(values, basic.identity_width)
    part, whole, _ = held_at(bank, bank.table)
    read = {}
    for value, row in rows.items():
        r = bank.rows.index(row)
        own = int(BasicPercepts.value_bin(torch.tensor(value), basic.identity_width))
        for b in range(values.shape[0]):
            for a in range(values.shape[1]):
                there = float(values[b, a])
                expected = ((1., 0.) if there > 0 and int(bins[b, a]) == own else
                            (0., 1.) if there > 0 and int(bins[b, a]) > own else (0., 0.))
                got = tuple(lanes[r, b, a].tolist())
                read[(value, there)] = got
                assert got == expected, (value, there, got)
                # held_at on a line: part is the occurrence holding the
                # value, whole the value's bound holding the occurrence.
                if there > 0:
                    assert float(part[r, b, a]) == float(int(bins[b, a]) >= own)
                    assert float(whole[r, b, a]) == float(int(bins[b, a]) <= own)
    print(dict(width=basic.identity_width, read={f'{v} at {t}': x for (v, t), x in sorted(read.items())}))
    # The numeric stem reads its bank by the Pi over values, never by the
    # first Pi over forms.
    used = _names_in(_source('NumericPercepts.py'), 'read')
    assert 'read_values' in used and 'read_bank' not in used
    assert BasicPercepts.read_values is not BasicPercepts.read_bank
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
    held = basic.place_bounds(bank.table, [places[0], places[1]])[:, 0]
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
