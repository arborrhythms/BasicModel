"""Operator-superposition composition (Phase 9).

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md ("Operator-
Superposition Composition" + integration map): STM applies a SOFT operator
distribution over the operator-prefixed tree (resolved against the SS
operation codebook) instead of a hard part-of-speech grammar. A one-hot
distribution reduces to the typed grammar (compatibility); a spread
distribution superposes operators -- load-bearing for discriminating
A AND B from A OR B.
"""

import copy
import os
import sys
import warnings

import pytest
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


# -- codebook-resolved superposition (model) --------------------------

@pytest.fixture(autouse=True)
def _restore_global_singletons():
    import Language
    from util import TheXMLConfig
    snap = {
        "data": copy.deepcopy(TheXMLConfig._data),
        "sources": list(TheXMLConfig._sources),
        "requirements": list(TheXMLConfig._requirements),
        "grammar": copy.deepcopy(Language.TheGrammar.__dict__),
    }
    try:
        yield
    finally:
        TheXMLConfig._data = copy.deepcopy(snap["data"])
        TheXMLConfig._sources = list(snap["sources"])
        TheXMLConfig._requirements = list(snap["requirements"])
        Language.TheGrammar.__dict__.clear()
        Language.TheGrammar.__dict__.update(copy.deepcopy(snap["grammar"]))
        Language.TheGrammar._configured = False


def _make_model():
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    m.eval()
    return m


def test_operator_superposition_resolves_against_codebook():
    """Querying an operation's own codebook vector yields a distribution
    that peaks on that operation."""
    import Language
    ws = _make_model().wholeSpace
    g = Language.TheGrammar
    ops = sorted({r.method_name for r in g.rules if r.method_name})
    target = ops[0]
    qv = ws.operation_vector(target)
    dist = ws.operator_superposition(qv)
    assert set(dist) == set(ops), (set(dist), set(ops))
    assert max(dist, key=dist.get) == target, (target, dist)
    assert abs(sum(dist.values()) - 1.0) < 1e-5   # a normalized distribution
