"""Tests for WholeSpace.resolve() -- Task 2.1.

resolve(subspace) sets subspace.activation to the scalar **balance of
evidence** ``pos - neg`` per symbol, where ``subspace.what[..., 0]`` is
the positive pole (evidence FOR) and ``subspace.what[..., 1]`` is the
negative pole (evidence AGAINST) of the 4-valued bivector stored per
symbol slot.  Range is roughly [-1, +1] (signed Degree of Truth):
positive = affirmation; negative = negation; zero = balanced/unknown.

inside() / outside() take the absolute value of this signed DoT when
treating activation as a magnitude-based extent, so a strongly-negated
symbol has the same extent as a strongly-affirmed one.
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

def _make_symbolic_space(nSymbols=3, symbolDim=8, conceptDim=8):
    """Construct a minimal WholeSpace via TheXMLConfig + direct constructor.

    Uses the same _populate_test_config / TheXMLConfig approach used by the
    rest of basicmodel's unit tests.  No nWhere / nWhen so the muxed layout
    is [what(=nDim), ] only.

    After construction (post-2026-05-07 rollback), subspace.nWhat ==
    sym.nDim (no leading [pos, neg] bivector pinned in the codebook).
    subspace.what is a Tensor basis whose setW/getW accept any
    [B, N, *] tensor; these tests still feed [B, N, 2] bivectors
    directly because resolve() reads only the leading 2 columns.
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
    outputShape = [nSymbols, symbolDim]    # WholeSpace output

    sym = Spaces.WholeSpace(inputShape, spaceShape, outputShape)
    # Width contract under the uniform (2,2) band (2026-06): content nWhat ==
    # nDim minus the where/when band (the SS=(0,0) special case where
    # nWhat == nDim exactly was retired).
    from architecture import canonical_shape
    _band = sum(canonical_shape("WholeSpace"))
    assert sym.subspace.nWhat == sym.nDim - _band, (
        f"Expected subspace.nWhat==sym.nDim-band ({sym.nDim - _band}) after "
        f"WholeSpace init, got {sym.subspace.nWhat}"
    )
    return sym


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# resolve(): the bivector [B,N,2] pos/neg pole-collapse tests
# (test_resolve_balances_pos_and_neg / test_resolve_serial_lossless)
# were retired with the bivector substrate (2026-05). resolve() now
# reads the signed Degree-of-Truth scalar directly; its scalar contract
# is covered by test_parthood_orders.py and the project-config suite.


# ---------------------------------------------------------------------------
# Task 2.3 — failing tests for WholeSpace.inside() / outside()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task 2.5 — codebook quantizes .activation (1-D), not .what (bivector)
# ---------------------------------------------------------------------------

def test_symbol_codebook_quantizes_activation_not_what():
    """WholeSpace.forward() calls resolve() so output.activation is 1-D.

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
    # PiLayer produces identical symbol-dim outputs.  The event is the full
    # muxed EVENT width (conceptDim=6 = content 2 + (2,2) band) so the
    # PiLayer matmul / event reshape succeeds under the uniform-(2,2) space_role.
    from architecture import canonical_shape as _cs
    _cdim = 2 + sum(_cs("ConceptualSpace"))   # content(2) + band(4) = 6
    concept_input = torch.zeros(1, 2, _cdim)   # [B=1, N=2, concept_event]
    sym.subspace.set_event(concept_input)

    sym.quantize = True
    output = sym.forward(sym.subspace)
    activations = output.materialize(mode="activation")

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
