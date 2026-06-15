"""ObjectSubSpace -- durable PartSpace meronymic-analysis carrier.

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md ("Carrier
State" + "Absorb/Emit/Swap codification"): the PS analogue of WordSubSpace.
It holds spans, part ids, parent/child links, route ids/scores, depth, and
the marker-route replay metadata (_marker_ps_id / _marker_span / _order_bit
/ _marker_position). All parallel buffers stay in sync under push / pop /
update; the live count is _depth.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


def _oss(cap=4, batch=1, dim=4):
    from Language import ObjectSubSpace
    return ObjectSubSpace(percept_dim=dim, capacity=cap, batch=batch)


def test_push_keeps_parallel_buffers_in_sync():
    oss = _oss()
    slot = oss.push(
        0, torch.arange(4, dtype=torch.float32),
        part_id=7, span_start=0, span_end=3, span_where=[1.0, 2.0],
        parent_id=-1, left_id=-1, right_id=-1,
        route_id=2, route_score=0.5)
    assert slot == 0
    assert oss.depth(0) == 1
    got = oss.get(0, 0)
    assert got["part_id"] == 7
    assert got["span_start"] == 0 and got["span_end"] == 3
    assert torch.allclose(got["span_where"], torch.tensor([1.0, 2.0]))
    assert got["route_id"] == 2
    assert abs(got["route_score"] - 0.5) < 1e-6
    assert torch.allclose(got["vec"], torch.arange(4, dtype=torch.float32))


def test_marker_route_metadata_round_trips():
    """The absorb/emit replay metadata is stored on the carrier."""
    oss = _oss()
    oss.push(0, torch.zeros(4), part_id=1,
             marker_ps_id=42, marker_span=[3.0, 4.0],
             order_bit=1, marker_position=1)  # INFIX
    got = oss.get(0, 0)
    assert got["marker_ps_id"] == 42
    assert torch.allclose(got["marker_span"], torch.tensor([3.0, 4.0]))
    assert got["order_bit"] == 1
    assert got["marker_position"] == 1


def test_update_one_field_does_not_desync_others():
    oss = _oss()
    oss.push(0, torch.ones(4), part_id=5, span_start=0, span_end=2, route_id=-1)
    oss.push(0, torch.full((4,), 2.0), part_id=6, span_start=2, span_end=4)
    assert oss.depth(0) == 2
    # The analyzer writes the chosen route + child links back to slot 0.
    oss.update(0, 0, route_id=3, left_id=0, right_id=1, route_score=0.9)
    a = oss.get(0, 0)
    b = oss.get(0, 1)
    assert a["route_id"] == 3 and a["left_id"] == 0 and a["right_id"] == 1
    assert abs(a["route_score"] - 0.9) < 1e-6
    # untouched fields on slot 0 and all of slot 1 stay intact -> no desync
    assert a["part_id"] == 5 and a["span_start"] == 0 and a["span_end"] == 2
    assert b["part_id"] == 6 and b["span_start"] == 2 and b["span_end"] == 4
    assert oss.depth(0) == 2  # update never changes depth


def test_pop_clears_slot_and_decrements_depth():
    oss = _oss()
    oss.push(0, torch.ones(4), part_id=5)
    oss.push(0, torch.full((4,), 2.0), part_id=6)
    top = oss.pop(0)
    assert top["part_id"] == 6
    assert oss.depth(0) == 1
    # popped slot is cleared back to defaults
    assert int(oss._part_id[0, 1].item()) == -1
    assert torch.allclose(oss._buffer[0, 1], torch.zeros(4))


def test_overflow_raises():
    oss = _oss(cap=2)
    oss.push(0, torch.zeros(4))
    oss.push(0, torch.zeros(4))
    try:
        oss.push(0, torch.zeros(4))
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected overflow past capacity to raise"


def test_live_mask_reflects_depth():
    oss = _oss(cap=4)
    oss.push(0, torch.zeros(4))
    oss.push(0, torch.zeros(4))
    mask = oss.live_mask()
    assert mask.shape == (1, 4)
    assert mask[0].tolist() == [True, True, False, False]


def test_ensure_batch_preserves_state():
    oss = _oss(cap=4, batch=1)
    oss.push(0, torch.arange(4, dtype=torch.float32), part_id=9)
    oss.ensure_batch(3)
    assert oss.batch == 3
    assert oss.depth(0) == 1
    assert oss.get(0, 0)["part_id"] == 9
    # fresh rows start empty
    assert oss.depth(1) == 0 and oss.depth(2) == 0
