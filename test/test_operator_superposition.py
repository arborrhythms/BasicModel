"""Standalone numerical operator mixtures and their one-hot limit.

The current sentence chooser's hard paths are tested separately. WholeSpace's
retired operator-codebook lookup is dispositioned in the item-7 review receipt.
"""

import os
import sys

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


# -- standalone soft compose (no model) -------------------------------

def test_soft_operator_compose_one_hot_equals_hard():
    """A one-hot operator distribution reduces to that operator's hard
    compose -- the typed grammar is preserved."""
    from perceptual_analyzer import soft_operator_compose
    from Language import GRAMMAR_LAYER_CLASSES
    left, right = torch.tensor([0.2, 0.0]), torch.tensor([0.8, 0.0])
    hard = GRAMMAR_LAYER_CLASSES["intersection"]().compose(left, right)
    soft = soft_operator_compose({"intersection": 1.0}, left, right)
    assert torch.allclose(soft, hard)


def test_soft_operator_compose_superposes_operators():
    """A spread distribution superposes operators into a genuine blend
    distinct from either hard operator (A-and-B vs A-or-B discrimination)."""
    from perceptual_analyzer import soft_operator_compose
    from Language import GRAMMAR_LAYER_CLASSES
    left, right = torch.tensor([0.2, 0.0]), torch.tensor([0.8, 0.0])
    inter = GRAMMAR_LAYER_CLASSES["intersection"]().compose(left, right)
    union = GRAMMAR_LAYER_CLASSES["union"]().compose(left, right)
    assert not torch.allclose(inter, union)   # min vs max really differ
    soft = soft_operator_compose(
        {"intersection": 0.5, "union": 0.5}, left, right)
    assert torch.allclose(soft, 0.5 * inter + 0.5 * union)
    assert not torch.allclose(soft, inter) and not torch.allclose(soft, union)
