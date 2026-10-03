"""Sigma composition and native span geometry after WholeSpace taxonomy retirement."""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Spaces
from Layers import SigmaLayer
from test_basicmodel import _populate_test_config

_D = 8


def _whole_space(nS=128):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    return Spaces.WholeSpace([nP, _D], [nS, _D], [nS, _D])


# -- SigmaLayer.synthesize_over_set ---------------------------------------

def test_synthesize_over_set_shape_and_binary_equivalence():
    torch.manual_seed(0)
    sig = SigmaLayer(nInput=_D, nOutput=_D, invertible=True, nonlinear=True)
    codes = (torch.randn(3, 4, _D).tanh() * 0.8)          # [B=3, M=4, D]
    out = sig.synthesize_over_set(codes)
    assert tuple(out.shape) == (3, _D)
    assert torch.all(out.abs() <= 1.0)                     # tanh-bounded
    # M==2 reduces exactly to compose(left, right).
    pair = torch.randn(2, _D).tanh() * 0.8
    viaset = sig.synthesize_over_set(pair.unsqueeze(0))     # [1, 2, D] -> [1, D]
    viacompose = sig.compose(pair[0:1], pair[1:2])
    assert torch.allclose(viaset, viacompose, atol=1e-5)


# Geometry is owned by the conceptual field. Native WS spans are inputs.

def _word_ws():
    ws = _whole_space()
    ws.analysis_mode = "meronomy"
    return ws


def test_conceptualspace_owns_run_structure_layer():
    from Layers import RunStructureLayer
    from test_cs_sparse_weights import _cs
    cs = _cs()
    assert isinstance(cs.run_structure, RunStructureLayer)
    assert cs.run_structure in list(cs.layers)        # on the cascade


def test_run_structure_over_word_spans_counts_words():
    ws = _word_ws()
    # "ab cd ef": whitespace (byte 32) at positions 2 and 5 -> word spans
    # (0,2),(3,5),(6,8); the Layer reports 3 runs (== word count), 2 gaps.
    concepts = torch.tensor([[97, 98, 32, 99, 100, 32, 101, 102]], dtype=torch.long)
    spans = ws.stage_analysis_spans(concepts)              # [B, K, 2]
    from Layers import RunStructureLayer
    out = RunStructureLayer()(spans.to(torch.float32))        # the forward data path
    assert int(out["n_runs"][0]) == 3
    assert int(out["n_gaps"][0]) == 2


def test_run_structure_single_blob_is_singleton():
    ws = _word_ws()
    # no whitespace -> one span over the whole -> 1 run (a singleton; the whole
    # should be further analysed to find boundaries).
    concepts = torch.tensor([[97, 98, 99, 100]], dtype=torch.long)
    spans = ws.stage_analysis_spans(concepts)
    from Layers import RunStructureLayer
    out = RunStructureLayer()(spans.to(torch.float32))
    assert int(out["n_runs"][0]) == 1


def test_stage_analysis_spans_is_structural_only():
    # the eager-stem helper no longer stashes the ratio (that moved into the
    # compiled forward); it just returns the spans tensor.
    ws = _word_ws()
    ws._mereology_raise = True
    concepts = torch.tensor([[97, 98, 32, 99, 100]], dtype=torch.long)
    spans = ws.stage_analysis_spans(concepts)
    assert spans is not None
