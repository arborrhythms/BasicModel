"""Wholes over values are ORed in the space of properties.

A number is the segment from zero to itself, so parthood over values is
less-than: 1 is a part of 2. In the whole space a range of values is the OR
of the levels inside it, as a class of characters is the OR of its
characters; two ranges OR into the whole that holds the values of both and
no others; and the top holds every value. The read is the first Pi's:
a whole ``r`` holds a percept ``p`` when ``min_k max(1 - p_k, r_k)`` is one.

These state the requirement of doc/plans/2026-11-18-item1-6-felt-math-world.md
("Wholes over values"). They fail while a value's wholes half is the bins at
or below it: the OR of such codes keeps only the greatest level, so a range
loses its lower end and holds every lesser value.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

WIDTH = 64


def _table():
    from BasicPercepts import BasicPercepts
    table = BasicPercepts(None, WIDTH, WIDTH, 1024)
    table.bind(None)
    return table


def _part(table, value):
    """A value's code in the parts half."""
    return table.value_code(torch.tensor(float(value)), WIDTH)


def _whole(table, value):
    """A single value's own whole in the property half."""
    reference = table._seat(torch.tensor(float(value)))
    return table._boundary_codes([reference], whole=True)[0]


def _range(table, low, high):
    """The whole over the levels of the alphabet from ``low`` to ``high``: the OR of theirs."""
    inside = [float(v) for v in table.levels if low <= float(v) <= high]
    assert inside, 'no level of the alphabet lies in the range'
    return torch.stack([_whole(table, v) for v in inside]).amax(0)


def _holds(whole, table, value):
    percept = _whole(table, value)
    return float(torch.maximum(1 - percept, whole).amin())


def test_a_lesser_value_is_a_part_of_a_greater():
    table = _table()
    for low, high in ((.1, .2), (.3, .6), (.5, 1.)):
        assert bool((_part(table, low) <= _part(table, high)).all())
        assert not bool((_part(table, high) <= _part(table, low)).all())


def test_a_range_holds_its_own_values_and_no_others():
    table = _table()
    whole = _range(table, .3, .4)
    assert _holds(whole, table, .35) == 1.
    assert _holds(whole, table, .1) == 0., 'a range holds a value below its lower end'
    assert _holds(whole, table, .9) == 0.


def test_two_ranges_or_into_the_whole_of_both():
    table = _table()
    whole = torch.maximum(_range(table, .3, .4), _range(table, .6, .7))
    assert _holds(whole, table, .35) == 1. and _holds(whole, table, .65) == 1.
    assert _holds(whole, table, .5) == 0., 'the OR of two ranges holds a value between them'
    assert _holds(whole, table, .1) == 0. and _holds(whole, table, .9) == 0.


def test_the_top_holds_every_value():
    table = _table()
    top = _range(table, 0., 1.)
    for value in (.05, .35, .5, .65, .95):
        assert _holds(top, table, value) == 1.


def test_ranges_hold_after_the_alphabet_has_learned():
    table = _table()
    for value in torch.rand(200, generator=torch.Generator().manual_seed(0)).tolist():
        if value > 0:
            table.perceive_intensity(value)
    assert bool((table.levels[1:] > table.levels[:-1]).all()), 'the levels left their order'
    whole = torch.maximum(_range(table, .3, .4), _range(table, .6, .7))
    assert _holds(whole, table, .35) == 1. and _holds(whole, table, .65) == 1.
    assert _holds(whole, table, .5) == 0.
