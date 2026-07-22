"""Fixed word-loop buckets and explicit STM order provenance."""

import functools
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import torch
import torch.nn as nn
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from Spaces import InputSpace
from recon_bench import _build_model, _resolve_config


def _surface(n):
    return " ".join(f"w{i}" for i in range(int(n)))


def test_smallest_fixed_word_bucket_is_selected():
    inp = SimpleNamespace()
    ps = SimpleNamespace()
    widths = (16, 32, 64, 128, 256)
    for n, expected in ((1, 16), (16, 16), (17, 32), (33, 64),
                        (65, 128), (129, 256), (256, 256)):
        sub = SimpleNamespace(_host_tokens=[[_surface(n)]])
        got = InputSpace.select_word_loop_bucket(
            inp, sub, widths, perceptual_space=ps)
        assert got == expected
        assert inp._serial_word_capacity == expected
        assert ps._serial_word_capacity == expected
        assert inp._serial_word_count_host == n


def test_overlong_sentence_is_rejected_not_clipped():
    inp = SimpleNamespace()
    sub = SimpleNamespace(_host_tokens=[[_surface(257)]])
    with pytest.raises(ValueError, match="rather than clipping"):
        InputSpace.select_word_loop_bucket(inp, sub, (16, 32, 64, 128, 256))


def test_basicmodel_declares_four_buckets_and_independent_inventories():
    root = ET.parse(ROOT / "data" / "BasicModel.xml").getroot()
    assert root.findtext("./architecture/serialWordBuckets") == (
        "16,32,64,128,256")
    assert int(root.findtext("./architecture/serialWordCapacity")) == 256
    ps = int(root.findtext("./PartSpace/nVectors"))
    ps_max = int(root.findtext("./PartSpace/maxVectors"))
    cs = int(root.findtext("./ConceptualSpace/nVectors"))
    ws = int(root.findtext("./WholeSpace/nVectors"))
    # All three dictionaries are separate namespaces. Alignment binds only
    # the two eight-location live fields; it does not equate row capacities.
    assert ps == 32768
    assert ps_max == 1048576
    assert cs == 1048576
    assert ws == 8
    assert root.findtext("./WholeSpace/propertyBasis") == "true"
    assert int(root.findtext("./ConceptualSpace/activeVectors")) == 32768
    assert root.find("./WholeSpace/activeVectors") is None
    # PS/WS recurse in native 128-WHAT events. Their sparse codebook
    # activations arrive at CS already decoded to 1024 WHAT; CS performs no
    # feature-width conversion. XML dimensions include the shared 8-D band.
    assert int(root.findtext("./PartSpace/nDim")) == 136
    assert int(root.findtext("./PartSpace/nOutputDim")) == 136
    assert int(root.findtext("./WholeSpace/nDim")) == 136
    assert int(root.findtext("./WholeSpace/nOutputDim")) == 136
    assert int(root.findtext("./ConceptualSpace/nInputDim")) == 1032
    assert int(root.findtext("./ConceptualSpace/nDim")) == 1032
    assert int(root.findtext("./ConceptualSpace/nOutputDim")) == 1032
    assert int(root.findtext("./ConceptualSpace/nOutput")) == 8
    assert int(root.findtext("./WholeSpace/nOutput")) == 8
    assert root.findtext("./architecture/weightsPath") == "BasicModel.ckpt"


@functools.lru_cache(maxsize=1)
def _grammar_model():
    torch.manual_seed(19)
    return _build_model(_resolve_config("data/MM_grammar.xml"))[0]


def test_binary_reduce_preserves_concept_order_and_raises_grammar_depth():
    model = _grammar_model()
    stm = model.conceptualSpace.stm
    stm.begin_forward(1, device=torch.device("cpu"))
    D = int(stm.concept_dim)
    gate = torch.ones(1, 1, dtype=torch.bool)
    stm.push_step_masked(
        torch.randn(1, D), gate,
        orders=torch.tensor([1]), grammar_orders=torch.tensor([0]))
    stm.push_step_masked(
        torch.randn(1, D), gate,
        orders=torch.tensor([3]), grammar_orders=torch.tensor([0]))

    reduced = model._stm_bounded_reduce_step(gate_tau=0.0)
    assert bool(reduced.item())
    assert int(stm._orders[0, 0]) == 3
    assert int(stm._grammar_orders[0, 0]) == 1


class _AlwaysUnary(nn.Module):
    def forward(self, x):
        candidate = x + 0.125
        routing = {
            "apply_mask": torch.ones(
                x.shape[0], x.shape[1], 1,
                dtype=x.dtype, device=x.device),
        }
        return candidate, candidate, routing


def test_unary_step_preserves_concept_order_and_raises_grammar_depth():
    model = _grammar_model()
    stm = model.conceptualSpace.stm
    stm.begin_forward(1, device=torch.device("cpu"))
    D = int(stm.concept_dim)
    stm.push_step_masked(
        torch.randn(1, D), torch.ones(1, 1, dtype=torch.bool),
        orders=torch.tensor([2]), grammar_orders=torch.tensor([0]))
    old = getattr(model, "_stm_unary_rewriter_cached", None)
    object.__setattr__(model, "_stm_unary_rewriter_cached", _AlwaysUnary())
    try:
        applied = model._stm_bounded_unary_step(
            row_gate=torch.ones(1, dtype=torch.bool))
    finally:
        if old is None:
            delattr(model, "_stm_unary_rewriter_cached")
        else:
            object.__setattr__(model, "_stm_unary_rewriter_cached", old)
    assert bool(applied.item())
    assert int(stm._orders[0, 0]) == 2
    assert int(stm._grammar_orders[0, 0]) == 1
