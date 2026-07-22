"""Peer-pipelined runtime ownership and ordering pins."""
import os
import sys
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

import pytest
import torch

from Spaces import SubSpace, guard_peer_views
from Models import StaticPeerPipeline


def test_subspace_view_has_no_setters_and_owner_commits():
    sub = SubSpace(inputShape=(1, 4), outputShape=(1, 4))
    owner = object()
    object.__setattr__(sub, "_owner_space", owner)
    view = sub.view()
    assert not hasattr(view, "set_event")
    assert not hasattr(view, "set_what")
    with pytest.raises(PermissionError):
        sub.commit_event(object(), torch.ones(1, 1, 4))
    sub.commit_event(owner, torch.ones(1, 1, 4))
    assert torch.equal(view.materialize(mode="event"), torch.ones(1, 1, 4))


def test_peer_guard_detects_direct_or_setter_mutation():
    sub = SubSpace(inputShape=(1, 2), outputShape=(1, 2))
    sub.set_event(torch.zeros(1, 1, 2))
    with pytest.raises(RuntimeError, match="peer mutation guard"):
        with guard_peer_views(sub.view()):
            sub.set_event(torch.ones(1, 1, 2))


def test_static_pipeline_warmup_drain_and_two_symbolic_index_latch():
    width = 16
    pipeline = StaticPeerPipeline(width)
    b_seen, c_seen = [], []

    def stage_a(word, index):
        return (word, index)

    def stage_b(a_result, index, feedback):
        b_seen.append((index, feedback))
        return (a_result, f"grammar-{index}")

    def stage_c(b_result, index):
        c_seen.append((index, b_result[0]))

    trace = pipeline.run(list(range(width)), stage_a, stage_b, stage_c)
    assert trace[:3] == [("A", 0), ("A", 1), ("B", 0)]
    assert trace[-1] == ("C", width - 1)
    assert b_seen[0] == (0, None)
    assert b_seen[1] == (1, None)
    assert b_seen[2] == (2, "grammar-0")
    assert c_seen == [(i, i) for i in range(width)]


@pytest.mark.parametrize("width", (1, 15, 17, 64))
def test_pipeline_requires_canonical_static_buckets(width):
    if width == 64:
        assert StaticPeerPipeline(width).width == width
    else:
        with pytest.raises(ValueError):
            StaticPeerPipeline(width)
