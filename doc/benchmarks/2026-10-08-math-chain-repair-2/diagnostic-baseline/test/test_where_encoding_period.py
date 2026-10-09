"""The model derives one address ladder from its complete registry allocation."""

import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch

_DATA_DIR = os.path.join(_PROJECT, "data")
_FIXTURE = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")

_WHERE_PERIOD_DEFAULT = 8192


def _write_config(tmp_path, where_period=None):
    """The xor fixture with an optional <wherePeriod> injected."""
    with open(_FIXTURE) as f:
        xml = f.read()
    if where_period is not None:
        xml = xml.replace(
            "<architecture>",
            f"<architecture>\n    <wherePeriod>{where_period}</wherePeriod>",
            1)
    p = tmp_path / f"where_period_{where_period}.xml"
    p.write_text(xml)
    return str(p)


def _build(config_path):
    """Fixture-model build mirroring bin/recon_bench (data loaded first)."""
    import Language
    import Models
    from data import TheData
    from util import init_config
    init_config(path=config_path, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    TheData.load("xor")
    model, _ = Models.BaseModel.from_config(config_path, data=TheData)
    return model


def _where_enc(model):
    return model.perceptualSpace.subspace.whereEncoding


def test_where_period_is_derived_from_the_registry():
    model = _build(_FIXTURE)
    enc = _where_enc(model)
    assert enc is model.where_encoding
    assert enc.maxVal > model.where_registry.capacity
    assert enc.period_hf <= 256
    for name, (start, end) in model.where_registry.slices.items():
        if end > start:
            addresses = torch.tensor([start, end - 1])
            torch.testing.assert_close(enc.decode_index(enc.encode(addresses)), addresses)


def test_input_and_inventory_share_the_same_encoding():
    model = _build(_FIXTURE)
    spaces = (model.inputSpace, model.perceptualSpace, *model.wholeSpaces, model.symbolSpace)
    assert all(space.subspace.whereEncoding is model.where_encoding for space in spaces)
    assert all(space.subspace.whenEncoding is model.when_encoding for space in spaces)


def test_retired_period_knob_is_rejected(tmp_path):
    # A separately configured period could silently alias the registry. The
    # schema rejects that old configuration rather than accepting a no-op.
    from util import XMLConfig
    with pytest.raises(ValueError, match='wherePeriod'):
        XMLConfig._validate_against_schema(_write_config(tmp_path, where_period=4))


def test_forward_overflow_assert_holds():
    """(d) the WhereEncoding.forward monotonic-counter overflow assert
    is untouched by the period decoupling."""
    from Spaces import WhereEncoding
    enc = WhereEncoding(4, nWhere=2, nWhen=0)
    x = torch.zeros([5, 1, 12])
    with pytest.raises(AssertionError, match="Overflow"):
        enc.forward(x)


def test_reconstruct_buffer_period_assert_holds():
    """(d) the reconstruct_to_buffer periodicity assert still fires when
    the render buffer exceeds the period, and its remedy names
    <wherePeriod> (not the retired nObjects coupling)."""
    model = _build(_FIXTURE)
    model.set_sigma(0)
    model.train(False)
    with torch.no_grad():
        model.runEpoch(batchSize=4, split="test")
    psp = model.perceptualSpace
    over = _where_enc(model).maxVal + 1
    with pytest.raises(AssertionError, match="construction-time input capacity"):
        psp.reconstruct_to_buffer(buf_size=over)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
