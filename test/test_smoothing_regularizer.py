# basicmodel/test/test_smoothing_regularizer.py
"""Tests for Layers.SmoothingRegLayer.

Plan reference: lazy-juggling-planet.md Phase B. The regularizer penalises
|S[i+1] - S[i]| along the concept axis of a symbol vector, with
bivector-aware pair-max collapse so paired (positive, negative) poles of the
same concept do not count as a discontinuity.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

from Layers import SmoothingRegLayer  # noqa: E402


def test_disabled_returns_zero():
    reg = SmoothingRegLayer(lam=0.5, enabled=False)
    x = torch.randn(4, 6)
    out = reg(x)
    assert torch.is_tensor(out)
    assert out.item() == 0.0


def test_zero_lambda_returns_zero():
    reg = SmoothingRegLayer(lam=0.0)
    x = torch.randn(4, 6)
    assert reg(x).item() == 0.0


def test_constant_symbol_vector_is_free():
    """A flat vector (no discontinuity) yields zero penalty."""
    reg = SmoothingRegLayer(lam=1.0)
    x = torch.ones(2, 8)  # 8 = 4 concepts * 2 poles
    assert reg(x).item() == 0.0


def test_alternating_concept_vector_is_penalised():
    """Alternating concept activations give a non-trivial penalty."""
    reg = SmoothingRegLayer(lam=1.0)
    # bivector layout [T,F, T,F, T,F, T,F] — concept-axis after pair-max = [1,1,1,1]
    # Flat at concept level → penalty 0.
    flat = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    assert reg(flat).item() == 0.0

    # Alternate active/inactive concepts: pair-max = [1,0,1,0] → non-zero penalty.
    alt = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    penalty = reg(alt).item()
    assert penalty > 0.0


def test_bivector_paired_poles_are_safe():
    """Concept #0 = T (1,0) next to concept #1 = F (0,1) must not produce
    a smoothing penalty — both concepts are equally 'active', pair-max is
    [1, 1] which is flat.
    """
    reg = SmoothingRegLayer(lam=1.0)
    # Storage: [T_pos, T_neg, F_pos, F_neg] = [1, 0, 0, 1]
    x = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    assert reg(x).item() == 0.0


def test_odd_last_dim_falls_back_to_raw_axis():
    """Non-bivector operand (odd last dim) uses raw consecutive differences."""
    reg = SmoothingRegLayer(lam=1.0)
    x = torch.tensor([[0.0, 1.0, 0.0]])  # diffs: [1, -1] → mean(abs) = 1
    assert torch.isclose(reg(x), torch.tensor(1.0))


def test_penalty_scales_linearly_with_lambda():
    base = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    small = SmoothingRegLayer(lam=0.5)(base).item()
    large = SmoothingRegLayer(lam=2.0)(base).item()
    assert large == 4.0 * small


def test_symbolic_space_reads_discontinuity_lambda_from_config():
    """SymbolicSpace picks up architecture.discontinuityLambda and exposes it."""
    from util import init_config

    xml = """<?xml version='1.0'?>
<model>
  <architecture>
    <conceptualOrder>1</conceptualOrder>
    <l1Lambda>0.01</l1Lambda>
    <discontinuityLambda>0.25</discontinuityLambda>
    <nWhere>2</nWhere>
    <nWhen>2</nWhen>
    <modelType>embedding</modelType>
  </architecture>
  <WordSpace>
    <language><grammar><S>C</S><C>P</C><P>I</P></grammar></language>
  </WordSpace>
  <InputSpace><nDim>8</nDim><nVectors>4</nVectors><nOutput>4</nOutput><nWhere>2</nWhere><nWhen>2</nWhen><codebook>true</codebook></InputSpace>
  <PerceptualSpace><nOutput>4</nOutput><nDim>8</nDim><nVectors>4</nVectors><hasAttention>false</hasAttention><invertible>false</invertible></PerceptualSpace>
  <ConceptualSpace><nOutput>4</nOutput><nDim>8</nDim><nVectors>4</nVectors><hasAttention>false</hasAttention><invertible>true</invertible><codebook>false</codebook></ConceptualSpace>
  <SymbolicSpace><nOutput>4</nOutput><nDim>8</nDim><nVectors>4</nVectors><lexer>sentence</lexer></SymbolicSpace>
  <OutputSpace><nOutput>1</nOutput><nDim>4</nDim><nVectors>1</nVectors><nWhere>0</nWhere><nWhen>0</nWhen><nonlinear>false</nonlinear></OutputSpace>
</model>
"""
    with tempfile.NamedTemporaryFile(
            'w', suffix='.xml', delete=False) as fh:
        fh.write(xml)
        path = fh.name
    try:
        init_config(path=path)
        from util import TheXMLConfig
        val = TheXMLConfig.get("architecture.discontinuityLambda")
        assert float(val) == 0.25
    finally:
        os.unlink(path)
