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


def _model():
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")

    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return model


def test_sequential_non_ar_forward_shape():
    """Sequential forward produces a 4-tuple with a tensor prediction in non-AR mode."""
    model = _model()
    inp = _xor_input()
    out = model.forward(inp)

    assert len(out) == 4
    assert isinstance(out[2], torch.Tensor)


def test_sequential_builds_body_stages_and_invertible_path():
    """Construction produces ``body_stages`` (an nn.ModuleList of
    ModuleDicts driven by ``_forward_body``).  The reverse pipeline
    ``reverse()`` reconstructs input from the terminal ConceptualSpace
    state; ``midpoint_cache`` survives as a no-op attribute.
    """
    import torch.nn as nn_
    model = _model()
    assert isinstance(model.body_stages, nn_.ModuleList)
    assert model.any_invertible is True, (
        "MM_xor.xml has invertible spaces; expected any_invertible=True")
    assert model.midpoint_cache is None
    assert callable(getattr(model, '_forward_body', None))
    assert callable(getattr(model, 'reverse', None)), (
        "reverse() is the reconstruction reverse pipeline "
        "(reconstructs input from the terminal ConceptualSpace state)")


def test_sequential_unrolls_subsymbolic_order():
    """body_stages has T (subsymbolicOrder) per-stage ModuleDicts.

    Replaces the prior pipeline_fwd.modules() walk: the per-stage
    structure is now first-class in ``body_stages`` (no more buried
    Sequential).
    """
    model = _model()
    T = len(model.conceptualSpaces)
    from Spaces import ConceptualSpace, WholeSpace
    assert len(model.body_stages) == T
    cs_count = sum(
        1 for stage in model.body_stages
        if isinstance(stage["cs"], ConceptualSpace))
    ws_count = sum(
        1 for stage in model.body_stages
        if isinstance(stage["ws"], WholeSpace))
    assert cs_count == T, f"expected {T} ConceptualSpaces, got {cs_count}"
    assert ws_count == T, f"expected {T} WholeSpaces, got {ws_count}"


# test_sequential_reconstruction_produced retired 2026-05-14: the
# reverse pipeline it exercised was retired alongside
# <reconstruct>output</...>.
# test_sequential_ar_mode_produces_prediction_tensor +
# test_basic_model_ar_sequential_path retired in the same change:
# within-sentence training is now IR-only and emits [B, N, predDim],
# not the legacy [B, K, N, predDim] AR microbatch shape.


# ``test_input_space_null_byte_emits_zero_validity`` retired: it asserted
# the AR per-window ``[B, K]`` InputSpace ``valid_mask`` (zero row at a
# given window → False). The IR-only refactor retired AR windowing;
# InputSpace ``valid_mask`` is now row-level ``[B, 1]`` with no
# per-position equivalent. Per-cell NULL gating under IR is exercised by
# ``test_padded_rows_no_op`` (valid_mask propagation via copy_context).
