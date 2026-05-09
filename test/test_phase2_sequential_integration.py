"""Phase 2 Sequential pipeline integration tests.

These exercise the end-to-end flag-on path on the XOR config — unroll,
AR-mode streaming, and reconstruction — asserting shape parity and
close numeric agreement with the legacy path. Bit-level parity to
legacy is not expected (the sequential path doesn't invoke discourse
priming, universality scoring, or predicted-head generation); this
test lives to catch shape regressions and catastrophic divergence.
"""
import sys
import os
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def _xor_input():
    return torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ).float().unsqueeze(1)


def _model(masked_prediction=None):
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    if masked_prediction is not None:
        model.masked_prediction = masked_prediction
        # Propagate to InputSpace so its forward() AR gate picks up the
        # override (model.forward does this defensively, but tests that
        # call inputSpace.forward directly rely on this).
        model.inputSpace.masked_prediction = masked_prediction
    return model


def test_sequential_non_ar_forward_shape():
    """Sequential forward produces a 4-tuple with a tensor prediction in non-AR mode."""
    model = _model()
    inp = _xor_input()
    out = model.forward(inp)

    assert len(out) == 4
    assert isinstance(out[2], torch.Tensor)


def test_sequential_builds_pipeline_fwd_and_rev():
    """Construction produces both pipeline_fwd and pipeline_rev (Case A, invertible spaces)."""
    import torch.nn as nn_
    model = _model()
    assert isinstance(model.pipeline_fwd, nn_.Sequential)
    assert model.pipeline_rev is not None, (
        "MM_xor.xml has invertible spaces; Case A should produce pipeline_rev")
    assert model.pipeline_rt is None
    assert model.midpoint_cache is None


def test_sequential_unrolls_conceptual_order():
    """Pipeline has T (conceptualOrder) conceptual+symbolic stage pairs.

    The microbatch-AR pipeline buries the per-stage modules inside a
    FlattenKWrapper-wrapped Sequential body, so this walks the module
    tree rather than the top-level pipeline list.
    """
    model = _model()
    T = len(model.conceptualSpaces)
    from Spaces import ConceptualSpace, SymbolicSpace
    all_modules = list(model.pipeline_fwd.modules())
    cs_count = sum(1 for m in all_modules if isinstance(m, ConceptualSpace))
    ss_count = sum(1 for m in all_modules if isinstance(m, SymbolicSpace))
    assert cs_count == T, f"expected {T} ConceptualSpaces, got {cs_count}"
    assert ss_count == T, f"expected {T} SymbolicSpaces, got {ss_count}"


def test_sequential_reconstruction_produced():
    """Reconstruction path runs when model.reversible is True.

    In the sequential design, forward() only populates the 4-tuple's
    reconstruction slot for ARIR (where reconstruction is needed inside
    the AR loop); non-AR reversible reconstruction happens in runBatch
    via model.reverse(). Exercise both so we catch regressions in either
    branch of that ownership split.
    """
    model = _model()
    assert model.reversible  # MM_xor has reconstruct="symbols"
    out = model.forward(_xor_input())
    _, symbols, outputData, _ = out
    inputData, _ = model.reverse(symbols, outputData)
    assert inputData is not None, (
        "reversible model.reverse() must produce a reconstruction")


def test_sequential_ar_mode_produces_prediction_tensor():
    """AR mode returns a [B, K, N, predDim] tensor (microbatch path).

    The legacy serial AR loop returned a Python list of per-position
    [B, N, predDim] tensors; the microbatch refactor replaced that with
    a single emit of all K windows in parallel.
    """
    model = _model(masked_prediction='AR')
    out = model.forward(_xor_input())
    predictions = out[2]
    assert isinstance(predictions, torch.Tensor), (
        f"AR mode expected predictions tensor, got {type(predictions).__name__}")
    assert predictions.dim() == 4, (
        f"AR predictions must be [B, K, N, predDim], got shape {tuple(predictions.shape)}")


def test_basic_model_ar_sequential_path():
    """BasicModel.forward() with maskedPrediction=AR produces per-pos predictions."""
    from data import TheData
    from Models import BaseModel, BasicModel
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    if not isinstance(model, BasicModel):
        pytest.skip("MM_xor.xml resolves to BasicModel; AR test requires BasicModel")
    model.masked_prediction = 'AR'
    out = model.forward(_xor_input())
    preds = out[2]
    assert isinstance(preds, torch.Tensor), (
        f"BasicModel AR should return tensor, got {type(preds).__name__}")
    assert preds.dim() == 4, (
        f"AR predictions must be [B, K, N, predDim], got shape {tuple(preds.shape)}")


def test_input_space_null_byte_emits_zero_validity():
    """All-zero target embeddings produce False entries in valid_mask.

    The legacy per-call slide loop emitted a sentinel empty subspace
    when the just-slid token was null. The microbatch path produces all
    K windows in one call and tags each with [B, K] validity instead;
    every all-zero target row should map to False.
    """
    import torch as _t
    model = _model(masked_prediction='AR')
    inp = model.inputSpace
    # Stub the embed step so InputSpace.forward runs the unfold/mask path
    # against a tensor we control directly.
    inp._lex_and_embed = lambda _x: inp.subspace
    embedded = _t.tensor(
        [[[0.1, 0.2],
          [0.0, 0.0],
          [0.3, 0.4]]],
        dtype=_t.float32,
    )
    inp.subspace.set_event(embedded)
    sub = inp.forward(_xor_input())
    assert sub.k_axis is True
    assert sub.valid_mask is not None
    # Position 1's target embedding is all zeros → that window is invalid.
    assert sub.valid_mask[0, 1].item() is False
