"""Aligned concept formation across ordered PS and WS fold ladders."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import Codebook, ConceptualSpace, SubSpace  # noqa: E402
from test_serial_object_meta import _serial_model_and_batch  # noqa: E402
from test_abstraction_order_canonical import _whole_space  # noqa: E402


class _Sub:
    """Small SubSpace-compatible carrier for binder-only tests."""

    def __init__(self, event):
        self.event = event

    def is_empty(self):
        return self.event is None

    def materialize(self):
        return self.event

    def set_event(self, event):
        self.event = event

    def copy_context(self, _other):
        return None


def test_basicmodel_selects_three_folds_per_tower():
    root = ET.parse(os.path.join(_PROJECT, "data", "BasicModel.xml")).getroot()
    arch = root.find("architecture")
    assert arch is not None
    assert arch.findtext("conceptBinding") == "aligned"
    assert int(arch.findtext("subsymbolicOrder")) - 1 == 3


def test_aligned_binder_uses_all_six_sources_without_location_mixing():
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpace
    # Actual serial geometry: the current word's PS carrier has one location,
    # while WS exposes the eight-location field. Aligned binding pads PS on
    # the location axis; it never reinterprets feature coordinates as slots.
    part_values = [1.0, 2.0, 3.0]
    whole_values = [4.0, 5.0, 6.0]
    part_tensors = [
        torch.full((1, 1, 4), value, requires_grad=True)
        for value in part_values
    ]
    whole_tensors = [
        torch.full((1, 8, 4), value, requires_grad=True)
        for value in whole_values
    ]
    out = SubSpace((1, 1), (1, 1), 1, 1)
    out.set_event(torch.zeros(1, 1, 4))
    carrier = cs.bind_fold_streams(
        [_Sub(x) for x in part_tensors],
        [_Sub(x) for x in whole_tensors],
        out,
        part_passes=(0, 1, 2),
        whole_passes=(0, 1, 2),
    )

    assert carrier.shape == (1, 6, 8, 4)
    assert torch.equal(out._concept_orders, torch.full((1, 8), 3))
    # Location 0 receives all six sources; later locations receive WS and
    # explicit zero PS padding, never coordinate-regrouped PS fragments.
    assert torch.allclose(out.materialize()[:, 0], torch.full((1, 4), 3.5))
    assert torch.allclose(out.materialize()[:, 1], torch.full((1, 4), 2.5))
    out.materialize().sum().backward()
    for source in part_tensors + whole_tensors:
        assert source.grad is not None
        assert bool((source.grad != 0).all())

    support = out._fold_support
    assert support["source_count"] == 6
    assert support["part_folds"][-1]["path"] == [
        [0, Codebook.FOLD_SIGMA],
        [1, Codebook.FOLD_SIGMA],
        [2, Codebook.FOLD_SIGMA],
    ]
    assert support["whole_folds"][-1]["path"] == [
        [0, Codebook.FOLD_PI],
        [1, Codebook.FOLD_PI],
        [2, Codebook.FOLD_PI],
    ]


def test_meta_fold_support_roundtrips_with_vocab_extras():
    support = ConceptualSpace._ordered_fold_support(
        (0, 1, 2), (0, 1, 2))
    ws = _whole_space()
    ps_pos = ws.ensure_ps_position(7)
    ws_pos = ws.insert_whole(init_vec=torch.randn(8))
    meta = ws.insert_meta(
        ps_pos, ws_pos, fused_vec=torch.randn(8),
        fold_support=support)
    assert ws.meta_fold_support[meta] == support

    blob = ws.vocab_extras()
    assert blob["meta_fold_support"][meta] == support
    ws2 = _whole_space()
    ws2.load_vocab_extras(blob)
    assert ws2.meta_fold_support[meta] == support
