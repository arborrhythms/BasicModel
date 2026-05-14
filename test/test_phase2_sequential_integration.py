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
    was retired 2026-05-14 alongside ``<reconstruct>output</...>`` so
    ``_run_pipeline_rev`` is gone; ``midpoint_cache`` survives as a
    no-op attribute.
    """
    import torch.nn as nn_
    model = _model()
    assert isinstance(model.body_stages, nn_.ModuleList)
    assert model.any_invertible is True, (
        "MM_xor.xml has invertible spaces; expected any_invertible=True")
    assert model.midpoint_cache is None
    assert callable(getattr(model, '_forward_body', None))
    assert not hasattr(model, '_run_pipeline_rev'), (
        "_run_pipeline_rev was retired with the reverse pipeline")


def test_sequential_unrolls_conceptual_order():
    """body_stages has T (conceptualOrder) per-stage ModuleDicts.

    Replaces the prior pipeline_fwd.modules() walk: the per-stage
    structure is now first-class in ``body_stages`` (no more buried
    Sequential).
    """
    model = _model()
    T = len(model.conceptualSpaces)
    from Spaces import ConceptualSpace, SymbolicSpace
    assert len(model.body_stages) == T
    cs_count = sum(
        1 for stage in model.body_stages
        if isinstance(stage["cs"], ConceptualSpace))
    ss_count = sum(
        1 for stage in model.body_stages
        if isinstance(stage["ss"], SymbolicSpace))
    assert cs_count == T, f"expected {T} ConceptualSpaces, got {cs_count}"
    assert ss_count == T, f"expected {T} SymbolicSpaces, got {ss_count}"


# test_sequential_reconstruction_produced retired 2026-05-14: the
# reverse pipeline it exercised was retired alongside
# <reconstruct>output</...>.
# test_sequential_ar_mode_produces_prediction_tensor +
# test_basic_model_ar_sequential_path retired in the same change:
# within-sentence training is now IR-only and emits [B, N, predDim],
# not the legacy [B, K, N, predDim] AR microbatch shape.


def test_input_space_null_byte_emits_zero_validity():
    """All-zero target embeddings produce False entries in valid_mask.

    The legacy per-call slide loop emitted a sentinel empty subspace
    when the just-slid token was null. The microbatch path produces all
    K windows in one call and tags each with [B, K] validity instead;
    every all-zero target row should map to False.
    """
    import torch as _t
    model = _model()
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
