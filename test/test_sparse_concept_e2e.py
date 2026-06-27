"""Phases 3 + 6: the sparse-coding concept transform, end to end.

The scatter-add feeds the concept content in ConceptualSpace.forward (D2) over
the CS-owned similarity_codebook inventory (D1), gated on the symbolic order. A
driver config (MM_sparse_concept.xml) builds and runs the parallel symbolic
loop; the concept code is forward-connected, so the gradient reaches the
learnable edge Weight. With the transform inactive the forward is byte-identical.
"""
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
import Language
from test_basicmodel import _populate_test_config

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")
_D = 8


def _cs_active(nS=64):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", 1)
    object.__setattr__(cs, "_serial", False)
    return cs


def _build(name):
    import Models
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


# -- Phase 3: the forward injection in isolation ------------------------------

def test_maybe_sparse_is_noop_when_inactive():
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D, nInput=nP, nPercepts=nP, nConcepts=64,
        nSymbols=64, nWords=64, nOutput=64, nWhere=0, nWhen=0)
    cs = Spaces.ConceptualSpace([nP, _D], [64, _D], [64, _D])   # NOT active
    folded = torch.randn(2, 8, _D)
    out = cs._maybe_sparse_concept_content(folded)
    assert out is folded                          # untouched -> byte-identical


def test_maybe_sparse_is_noop_with_empty_edges():
    cs = _cs_active()
    folded = torch.randn(2, 8, _D)
    out = cs._maybe_sparse_concept_content(folded)
    assert torch.equal(out, folded)               # no edges -> unchanged


def test_maybe_sparse_injects_and_is_differentiable():
    cs = _cs_active()
    # populate a couple of concepts' edges
    cs.create_word_object_meta([1, 2], 7, key="cat")
    cs.create_word_object_meta([3, 4], 7, key="dog")
    folded = torch.randn(2, 8, _D)
    out = cs._maybe_sparse_concept_content(folded)
    assert out.shape == folded.shape
    assert not torch.equal(out, folded)           # the edged slots changed
    # forward-connected: the gradient reaches the learnable edge Weight.
    _, _, w = cs._rebuild_edge_pools()
    out.sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


# -- Phase 6: the driver config builds + runs, grad reaches the edge Weight ---

def test_sparse_concept_config_builds_parallel():
    m = _build("MM_sparse_concept.xml")
    assert m.serial is False and m.symbolicOrder == 1
    assert m.symbol_tower is True


def test_sparse_concept_forward_smoke():
    """The parallel symbolic loop runs end to end without error (the byte-
    identical-off path stays valid; the gated scatter is a no-op until edges
    exist)."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_sparse_concept.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.train()
    out = m.forward(x)
    assert out is not None


def test_sparse_concept_config_activates_transform_on_cs():
    """The config's stamp marks the ConceptualSpaces sparse-active, so a mint on
    the model's own CS populates edges and the scatter injects a grad-bearing
    concept content (the forward-connection the plan delivers)."""
    m = _build("MM_sparse_concept.xml")
    css = [cs for cs in m.conceptualSpaces if cs._sparse_active()]
    assert css, "the config stamp should mark a ConceptualSpace sparse-active"
    cs = css[-1]                                 # the terminal CS
    cs.create_word_object_meta([1, 2], 7, key="cat")
    cs.create_word_object_meta([3, 4], 7, key="dog")
    D = int(cs.outputShape[1])
    folded = torch.randn(2, int(cs.outputShape[0]), D)
    out = cs._maybe_sparse_concept_content(folded)
    assert not torch.equal(out, folded)          # the edged slots took the code
    _, _, w = cs._rebuild_edge_pools()
    out.sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0
