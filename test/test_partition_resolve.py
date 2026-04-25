"""Tests for SymbolicSpace.resolve() -- Task 2.1 (failing harness).

resolve(subspace) will set subspace.activation to the scalar sum
``pos + neg`` per symbol, where ``subspace.what[..., 0]`` is the
positive pole and ``subspace.what[..., 1]`` is the negative pole of the
4-valued bivector stored per symbol slot.

These tests are intentionally FAILING until Task 2.2 implements resolve().
The expected failure mode is:
    AttributeError: 'SymbolicSpace' object has no attribute 'resolve'
"""
import os
import sys
import math
import pytest
import torch

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as test_bivector_basis.py
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

import Models
import Spaces
from test_basicmodel import _populate_test_config


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _make_symbolic_space(nSymbols=3, symbolDim=4, conceptDim=4):
    """Construct a minimal SymbolicSpace via TheXMLConfig + direct constructor.

    Uses the same _populate_test_config / TheXMLConfig approach used by the
    rest of basicmodel's unit tests.  No nWhere / nWhen so the muxed layout
    is [what(=2), ] only.

    After construction, subspace.nWhat == 2 (the bivector override that
    SymbolicSpace.__init__ applies) and subspace.what is a Tensor basis
    whose setW/getW accept any [B, N, 2] tensor.
    """
    _populate_test_config(
        inputDim=conceptDim,
        perceptDim=conceptDim,
        conceptDim=conceptDim,
        symbolDim=symbolDim,
        wordDim=symbolDim,
        outputDim=symbolDim,
        nInput=nSymbols,
        nPercepts=nSymbols,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        nWords=nSymbols,
        nOutput=nSymbols,
        nWhere=0,
        nWhen=0,
    )

    # Shapes mirror Models.BasicModel._create_spaces for the non-flatten path.
    # obj = nWhere + nWhen = 0, so outputShape = [nSymbols, symbolDim].
    inputShape  = [nSymbols, conceptDim]   # ConceptualSpace output
    spaceShape  = [nSymbols, symbolDim]    # codebook internal shape
    outputShape = [nSymbols, symbolDim]    # SymbolicSpace output

    sym = Spaces.SymbolicSpace(inputShape, spaceShape, outputShape)
    # Invariant: SymbolicSpace.__init__ sets nWhat=2 (bivector)
    assert sym.subspace.nWhat == 2, (
        f"Expected subspace.nWhat==2 after SymbolicSpace init, got {sym.subspace.nWhat}"
    )
    return sym


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resolve_sums_pos_and_neg():
    """resolve(subspace) sets subspace.activation = pos + neg per symbol.

    what[..., 0] is the positive pole, what[..., 1] is the negative pole.
    Activation scalar = pos + neg (total signal strength, unsigned).
    """
    sym = _make_symbolic_space()

    # Shape [B=1, N=3, nWhat=2]: (pos_pole, neg_pole) per symbol
    what = torch.tensor([[[0.5, 0.0],   # symbol 0: pos=0.5, neg=0.0 → 0.5
                          [0.2, 0.3],   # symbol 1: pos=0.2, neg=0.3 → 0.5
                          [0.0, 0.7]]]) # symbol 2: pos=0.0, neg=0.7 → 0.7
    sym.subspace.what.setW(what)

    # This line fails until Task 2.2 implements resolve():
    sym.resolve(sym.subspace)  # expected: AttributeError

    activation = sym.subspace.activation.getW()
    expected = torch.tensor([[0.5, 0.5, 0.7]])   # [B=1, N=3]
    assert torch.allclose(activation, expected), (
        f"Expected {expected}, got {activation}"
    )


def test_resolve_serial_lossless():
    """Under serial processing (one pole non-zero at a time), resolve is lossless.

    When only one pole fires per slot, pos + neg == the firing pole's value,
    so no information is lost in the scalar reduction.
    """
    sym = _make_symbolic_space()

    what = torch.tensor([[[0.8, 0.0],   # pos-only: 0.8
                          [0.0, 0.6],   # neg-only: 0.6
                          [0.9, 0.0]]]) # pos-only: 0.9
    sym.subspace.what.setW(what)

    # This line fails until Task 2.2 implements resolve():
    sym.resolve(sym.subspace)  # expected: AttributeError

    activation = sym.subspace.activation.getW()
    expected = torch.tensor([[0.8, 0.6, 0.9]])
    assert torch.allclose(activation, expected), (
        f"Expected {expected}, got {activation}"
    )


# ---------------------------------------------------------------------------
# Task 2.3 — failing tests for SymbolicSpace.inside() / outside()
# ---------------------------------------------------------------------------

def test_inside_of_parthood_matches_part_primitive():
    """inside(point, symbol_idx) delegates to Basis.part() on resolved activation.

    Shape notes
    -----------
    We use nSymbols=1 so the codebook has a single slot: what shape [B=1, N=1, 2].
    After resolve(), subspace.activation has shape [B=1, N=1].
    Indexing symbol_idx=0 yields a slice of shape [B=1] = [1].
    We therefore make point shape [1] so the comparison broadcasts cleanly.

    Semantic intuition
    ------------------
    what = [[[0.9, 0.1]]] → pos=0.9, neg=0.1 → activation = 0.9 + 0.1 = 1.0.
    point_inside = [0.2]: magnitude 0.2 < 1.0 → inside the symbol's extent.
    point_outside = [1.5]: magnitude 1.5 > 1.0 → outside.
    The exact comparison rule is left to Task 2.4; these values cleanly separate.

    This test FAILS until Task 2.4 implements inside():
        AttributeError: 'SymbolicSpace' object has no attribute 'inside'
    """
    sym = _make_symbolic_space(nSymbols=1)

    # [B=1, N=1, nWhat=2]: one symbol, pos=0.9, neg=0.1 → activation 1.0
    what = torch.tensor([[[0.9, 0.1]]])
    sym.subspace.what.setW(what)
    sym.resolve(sym.subspace)

    # shape [1] — matches activation[..., 0] which has shape [B=1]
    point_inside = torch.tensor([0.2])   # 0.2 < 1.0: inside
    point_outside = torch.tensor([1.5])  # 1.5 > 1.0: outside

    # Both lines below fail until Task 2.4 adds inside():
    assert sym.inside(point_inside, symbol_idx=0)
    assert not sym.inside(point_outside, symbol_idx=0)


def test_outside_is_negation_of_inside():
    """outside(point, symbol_idx) is the logical complement of inside(...).

    Same shape setup as test_inside_of_parthood_matches_part_primitive.
    We check the complement property for a single representative point.

    This test FAILS until Task 2.4 implements inside() / outside():
        AttributeError: 'SymbolicSpace' object has no attribute 'inside'
    """
    sym = _make_symbolic_space(nSymbols=1)

    what = torch.tensor([[[0.9, 0.1]]])
    sym.subspace.what.setW(what)
    sym.resolve(sym.subspace)

    point_inside = torch.tensor([0.2])

    # Both calls fail until Task 2.4 is done:
    assert sym.inside(point_inside, symbol_idx=0) == (
        not sym.outside(point_inside, symbol_idx=0)
    )


# ---------------------------------------------------------------------------
# Task 2.5 — codebook quantizes .activation (1-D), not .what (bivector)
# ---------------------------------------------------------------------------

def test_symbol_codebook_quantizes_activation_not_what():
    """SymbolicSpace.forward() calls resolve() so output.activation is 1-D.

    Two key assertions:

    1. ``output.activation`` is shape ``[B, N]`` (not ``[B, N, 2]`` bivector)
       after forward().  This confirms resolve() was called inside forward().

    2. Both symbols produce the same output activation when their concept inputs
       are identical (zeros → PiLayer(zeros) → same act → same resolved scalar
       → same codebook snap).

    Failure mode before Task 2.5:
      forward() does NOT call resolve(), so activation is the bivector
      ``[aP, aN]`` set by ``set_event()`` → shape ``[B, N, 2]``.
      ``activations[..., 0]`` indexes the first bivector component for ALL
      symbols (== 1.0) and ``activations[..., 1]`` indexes the second (== 0.0);
      these are NOT allclose, so the second assert fails.  The ndim check
      also fails (3 != 2).

    After Task 2.5:
      forward() calls resolve() before quantization, writing
      ``subspace.activation = pos + neg`` ([B, N]).
      Both zero-input symbols produce the same resolved scalar → same codebook
      entry → ``activations[..., 0] == activations[..., 1]``.
    """
    sym = _make_symbolic_space(nSymbols=2)

    # Feed identical concept vectors (zeros) for both symbol slots so the
    # PiLayer produces identical symbol-dim outputs.  The event is
    # concept_dim=4 wide so the PiLayer matmul succeeds.
    concept_input = torch.zeros(1, 2, 4)   # [B=1, N=2, concept_dim=4]
    sym.subspace.set_event(concept_input)

    sym.quantize = True
    output = sym.forward(sym.subspace)
    activations = output.activation.getW()

    # Assertion 1: activation is 1-D [B, N] after resolve(), not [B, N, 2].
    assert activations.ndim == 2, (
        f"forward() must leave activation as [B, N] scalar (resolve called); "
        f"got shape {list(activations.shape)}"
    )

    # Assertion 2: identical concept inputs → identical resolved activations.
    assert torch.allclose(activations[..., 0], activations[..., 1]), (
        f"Symbols with identical concept inputs should produce equal activations "
        f"after quantization; got {activations[..., 0]} vs {activations[..., 1]}"
    )
