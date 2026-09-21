"""Production composition is independent of the staged expectation (spec test 3).

Replaces four tests that required the now-deleted additive prior.
"""
import os
import sys
import warnings

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from Spaces import SubSpaceView
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, 'data')
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_plain_model():
    """Build a working model from MM_xor_loopback.xml + xor data -- the
    same cheap-boot pattern used by ``test_cs_stm_bookkeeping.py`` /
    ``test_ps_single_arg_refactor.py`` / ``test_pi_sigma_ownership.py``."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model



@pytest.mark.parametrize("gain", [0., 1.])
@pytest.mark.parametrize("staged", [False, True])
def test_staged_expectation_never_enters_conceptual_forward(gain, staged):
    torch.manual_seed(414)
    model = _make_plain_model()
    cs = model.conceptualSpace
    model.expectation_gain = gain
    ps = model.perceptualSpace
    loader = model.inputSpace.data.data_loader(split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    x = model.inputSpace.prepInput(inp_items)
    with torch.no_grad():
        incoming, _ = model.inputSpace.forward(x)
        ps_sub = ps.forward(incoming)
        event = ps_sub.materialize()
        base = torch.zeros_like(event)
        if staged:
            discourse = model.symbolSpace.discourse
            roles = torch.ones(3, discourse.concept_dim)
            discourse.predict_and_observe_stm_end_state([3], [roles], layout="infix")
            assert discourse.expect_next_meaning() is not None
        assert not hasattr(cs, "_c_prior")
        view = SubSpaceView.snapshot(base, owner=ps, context=ps_sub.view().context())
        cs.forward(view)
        torch.testing.assert_close(cs.CSsub.materialize(mode="event"), base, rtol=0, atol=0)
        assert not hasattr(cs, "_c_prior")
