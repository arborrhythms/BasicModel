"""The model enables refinement on the conceptual field.

Attribution focus and its nonzero floor are covered by test_item11c_review;
retained occurrence routing is covered by test_refine_raise.
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import pytest
import torch

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def _batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


@pytest.mark.slow
def test_mereology_raise_enables_conceptual_refinement():
    # <mereologyRaise> stamps the stage-0 WholeSpace (so it can compute the
    # read-only run-structure obs) and the default forward stays deterministic
    # + finite -- the pass-back is "noop" without attention.
    m = _build("MM_mereology.xml")
    assert m.mereology_raise
    assert m.conceptualSpaces[0]._mereology_raise
    x = _batch(m)
    with torch.no_grad():
        out1 = m.forward(x)[2]
        out2 = m.forward(x)[2]
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2), "the dark pass-back must be deterministic"
