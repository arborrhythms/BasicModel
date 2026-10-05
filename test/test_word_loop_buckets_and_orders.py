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
    return " ".join("word" for i in range(int(n)))


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


def test_single_dynamic_capacity_accepts_short_and_long_complete_sentences():
    inp = SimpleNamespace()
    ps = SimpleNamespace()
    for n in (1, 64, 65, 128, 256):
        sub = SimpleNamespace(_host_tokens=[[_surface(n)]])
        assert InputSpace.select_word_loop_bucket(
            inp, sub, (256,), perceptual_space=ps) == 256
        assert inp._serial_word_count_host == n
    with pytest.raises(ValueError, match="rather than clipping"):
        InputSpace.select_word_loop_bucket(
            inp, SimpleNamespace(_host_tokens=[[_surface(257)]]), (256,))


def test_basicmodel_declares_one_dynamic_capacity_and_independent_inventories():
    root = ET.parse(ROOT / "data" / "BasicModel.xml").getroot()
    assert root.findtext("./architecture/serialWordBuckets") == "256"
    assert int(root.findtext("./architecture/serialWordCapacity")) == 256
    ps = int(root.findtext("./PartSpace/nVectors"))
    assert root.find("./PartSpace/maxVectors") is None
    cs = int(root.findtext("./ConceptualSpace/nVectors"))
    ws = int(root.findtext("./WholeSpace/nVectors"))
    # All three dictionaries are separate namespaces. Alignment binds only
    # the two eight-location live fields; it does not equate row capacities.
    assert ps == 32768
    assert cs == 65536
    assert ws == 9
    assert root.find("./WholeSpace/propertyBasis") is None
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


def test_one_operation_layer_owns_both_arities():
    model = _grammar_model()
    shared = model.symbolSpace.languageLayer.operation_layer
    assert model._stm_reducer() is shared
    assert shared.r_reduce > 0 and shared.r_apply == 0
    assert 5 in shared.attention_operations
    assert not hasattr(model.symbolSpace.languageLayer, '_unary_layers')
