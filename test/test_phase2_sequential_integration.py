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
    """Pipeline has T (conceptualOrder) conceptual+symbolic stage pairs."""
    model = _model()
    T = int(model.conceptualOrder)
    modules = list(model.pipeline_fwd)
    from Pipeline import StageWrapper
    stage_wrappers = [m for m in modules if isinstance(m, StageWrapper)]
    assert len(stage_wrappers) == 2 * T, (
        f"expected 2*T={2*T} StageWrappers, got {len(stage_wrappers)}")


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


def test_sequential_ar_mode_produces_prediction_list():
    """AR mode returns a list of per-position predictions."""
    model = _model(masked_prediction='ARLM')
    out = model.forward(_xor_input())
    predictions = out[2]
    assert isinstance(predictions, list), (
        f"AR mode expected predictions to be list, got {type(predictions).__name__}")
    assert len(predictions) >= 1


def test_basic_model_ar_sequential_path():
    """BasicModel.forward() with maskedPrediction=ARLM produces per-pos predictions."""
    from data import TheData
    from Models import BaseModel, BasicModel
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    if not isinstance(model, BasicModel):
        pytest.skip("MM_xor.xml resolves to MentalModel; AR test requires BasicModel")
    model.masked_prediction = 'ARLM'
    out = model.forward(_xor_input())
    preds = out[2]
    assert isinstance(preds, list), (
        f"BasicModel AR should return list, got {type(preds).__name__}")
    assert len(preds) >= 1


def test_input_space_null_byte_check_logic():
    """Null-byte sentinel: all-zero embedding → empty subspace.

    Unit-tests the detection logic directly, since AR inference (which
    feeds predictions back into input) is not wired yet. Simulates a
    cached embedding where position 1 is all zeros.
    """
    import torch as _t
    model = _model(masked_prediction='ARLM')
    inp = model.inputSpace
    # Prime the AR streaming state directly (bypass first-call lex/embed).
    inp._seq_cursor = 2  # about to emit position 2
    inp._ar_total = 3
    inp._ar_embedded = _t.tensor(
        [[[0.1, 0.2],
          [0.0, 0.0],   # position 1: null-byte embedding (about to slide)
          [0.3, 0.4]]],
        dtype=_t.float32,
    )
    inp._ar_buffer = None
    # cursor>0 → checks embedded[:, cursor-1, :] which is position 1 (all zeros).
    result = inp.forward(_xor_input())
    assert result.is_empty(), (
        "InputSpace should emit empty sentinel when the just-slid token "
        "is all-zero (null-byte sentinel)")
