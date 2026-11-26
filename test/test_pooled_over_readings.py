"""The aperture limits pooling over the readings of the spaces of intensity alone."""
from types import SimpleNamespace

import torch


def _owner():
    registry = SimpleNamespace(slices={'text': (0, 10), 'numeric': (10, 14)})
    return SimpleNamespace(basic_percepts=SimpleNamespace(place_rows={}),
                           _concept_allocator=SimpleNamespace(order_zero_members=[]),
                           _concept_row_of=lambda cid: cid, _model=SimpleNamespace(where_registry=registry))


def _pooled(monkeypatch, where, evidence):
    import util
    from Spaces import pooled_over_readings
    monkeypatch.setattr(util.TheXMLConfig, 'get', lambda *a, **k: 1)
    lanes = torch.zeros(1, 1, len(where), 2)
    lanes[0, 0, :, 0] = torch.tensor(evidence)
    field = SimpleNamespace(evidence=lanes, concept_ids=torch.tensor([5]), where=torch.tensor([where]))
    return round(float(pooled_over_readings(field, _owner())[0, 0, 0]), 6)


def test_an_empty_reading_does_not_move_the_bracket(monkeypatch):
    # Two places and their bracket: a concept of no place is read at the bracket.
    assert _pooled(monkeypatch, [[10, 11], [11, 12], [10, 12]], [.9, .1, .7]) == .7
    for empty in ([-1, -1], [10, 10]):
        assert _pooled(monkeypatch, [[10, 11], [11, 12], [10, 12], empty], [.9, .1, .7, 0.]) == .7


def test_the_aperture_does_not_narrow_the_readings_of_text(monkeypatch):
    # A sentence wider than the aperture: a word's concept is pooled over its readings.
    assert _pooled(monkeypatch, [[0, 1], [1, 2], [2, 3], [0, 3]], [.1, .9, .1, .2]) == .9
    # Beside an image, the image's readings alone are narrowed to their bracket.
    where = [[0, 1], [1, 2], [0, 2], [10, 11], [11, 12], [10, 12]]
    assert _pooled(monkeypatch, where, [.1, .6, .2, .9, .1, .3]) == .6
