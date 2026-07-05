"""Task 1 of the `.where`/`.when` encoding pass (2026-07-04 plan):
`.where` period decoupled from ``architecture.nObjects``.

The period becomes config-derived: ``<architecture><wherePeriod>`` with
default 8192 (the input/sentence byte cap), NOT $\\Sigma$ nVectors. The
build seam keeps raise-to-fit semantics, but any raise past the
configured period WARNS ONCE (config name, length, period, "increase
<wherePeriod>") -- never silent aliasing. The runtime guards (the
``forward`` counter-overflow assert and the ``reconstruct_to_buffer``
periodicity assert) still hold.
"""

import os
import re
import sys
import warnings

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


def test_where_period_tag_sets_maxval(tmp_path):
    """(a) <wherePeriod>8192</wherePeriod> -> maxVal == 8192 regardless of
    nObjects (the fixture's nObjects == 33 would have been the old value)."""
    model = _build(_write_config(tmp_path, where_period=8192))
    enc = _where_enc(model)
    assert enc.maxVal == 8192, enc.maxVal
    from util import TheXMLConfig
    n_objects = int(TheXMLConfig.get("architecture.nObjects"))
    assert enc.maxVal != n_objects, "period must not track nObjects"


def test_where_period_default_8192():
    """(b) absent tag -> default 8192 (the input/sentence cap), not
    nObjects; xor's 11-byte inputs never trigger the raise."""
    model = _build(_FIXTURE)
    assert _where_enc(model).maxVal == _WHERE_PERIOD_DEFAULT


def test_where_period_overflow_warns_once_and_raises_to_fit(tmp_path):
    """(c) input longer than the configured period -> exactly ONE
    RuntimeWarning naming config, length, period, and the remedy -- and the
    period is raised to fit (never silent aliasing)."""
    cfg = _write_config(tmp_path, where_period=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _build(cfg)
    hits = [w for w in caught
            if "wherePeriod" in str(w.message)
            and issubclass(w.category, RuntimeWarning)]
    assert len(hits) == 1, [str(w.message) for w in caught]
    msg = str(hits[0].message)
    assert os.path.basename(cfg) in msg or cfg in msg, msg
    # xor's longest train input measures 12 bytes (dataset truth).
    assert re.search(r"\b12\b", msg), f"input byte length missing: {msg}"
    assert re.search(r"\b4\b", msg), f"configured period missing: {msg}"
    assert "increase <wherePeriod>" in msg, msg
    # Raise-to-fit: the seam's existing 2x headroom for string inputs.
    assert _where_enc(model).maxVal == 24, _where_enc(model).maxVal


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
    with pytest.raises(AssertionError, match="wherePeriod"):
        psp.reconstruct_to_buffer(buf_size=over)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
