"""Tests for the PoS codebook and PoS stack on WordSpace.

Tasks 3.2 and 3.4 (TDD harness) + Tasks 3.3 and 3.5 (implementation).

WordSpace cannot be constructed in isolation -- it requires real
PerceptualSpace, ConceptualSpace, and SymbolicSpace objects (it calls
attach_wordSpace() on each home space and grammar.symbolic() etc.).
We build the minimal chain using _populate_test_config + direct Space
constructors, the same pattern used by test_partition_resolve.py for
SymbolicSpace alone.
"""
import os
import sys
import pytest
import torch

# ---------------------------------------------------------------------------
# Path bootstrap
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
import Language
from test_basicmodel import _populate_test_config


# ---------------------------------------------------------------------------
# Minimal space chain builder
# ---------------------------------------------------------------------------

def _make_word_space(nSymbols=3, symbolDim=4, conceptDim=4, nPercepts=3):
    """Construct the minimal WordSpace via TheXMLConfig + direct constructors.

    Builds the full PerceptualSpace -> ConceptualSpace -> SymbolicSpace chain
    required by WordSpace.__init__, then constructs WordSpace from it.

    WordSpace is NOT an isolated object: it back-wires all three home spaces
    via attach_wordSpace(), configures TheGrammar, and reads TheXMLConfig for
    TruthLayer capacity and discourse gating.  The minimal chain is therefore
    required -- there is no lighter-weight path.
    """
    _populate_test_config(
        inputDim=conceptDim,
        perceptDim=conceptDim,
        conceptDim=conceptDim,
        symbolDim=symbolDim,
        wordDim=symbolDim,
        outputDim=symbolDim,
        nInput=nPercepts,
        nPercepts=nPercepts,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        nWords=nSymbols,
        nOutput=nSymbols,
        nWhere=0,
        nWhen=0,
    )

    # Shapes mirror Models.BasicModel._create_spaces for the non-flatten path.
    inputShape   = [nPercepts,  conceptDim]
    spaceShape   = [nSymbols,   symbolDim]
    outputShape  = [nSymbols,   symbolDim]

    percept_space   = Spaces.PerceptualSpace(inputShape, spaceShape, outputShape)
    concept_space   = Spaces.ConceptualSpace(inputShape, spaceShape, outputShape)
    symbolic_space  = Spaces.SymbolicSpace(inputShape, spaceShape, outputShape)

    # Reset grammar so WordSpace.__init__ can (re)configure it cleanly.
    Language.TheGrammar._configured = False

    ws = Language.WordSpace(
        perceptualSpace=percept_space,
        conceptualSpace=concept_space,
        symbolicSpace=symbolic_space,
        nPercepts=nPercepts,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        concept_dim=conceptDim,
        symbol_dim=symbolDim,
    )
    return ws


# ---------------------------------------------------------------------------
# Task 3.2 -- PoS codebook tests
# ---------------------------------------------------------------------------

def test_pos_codebook_shape():
    """PoS category embedding is 64 x 4.

    Post-2026-05-20: ``category_codebook`` (Codebook) was retired in
    favor of ``category_embedding`` (nn.Embedding). Shape is preserved.
    """
    ws = _make_word_space()
    assert ws.category_embedding.weight.shape == (64, 4)


def test_pos_codebook_lookup_deterministic():
    """Same symbolic-activation pattern always maps to the same PoS row."""
    ws = _make_word_space()
    active = torch.tensor([0.9, 0.0, 0.3])  # 3 symbols (matches nSymbols=3)
    v1 = ws.pos_lookup(active)
    v2 = ws.pos_lookup(active)
    assert torch.allclose(v1, v2)
    assert v1.shape == (4,)


# ---------------------------------------------------------------------------
# Task 3.4 -- PoS stack tests
# ---------------------------------------------------------------------------

def test_category_stack_push_pop_roundtrip():
    ws = _make_word_space()
    v = torch.randn(4)
    ws.category_stack.push(0, v)
    popped = ws.category_stack.pop(0)
    assert torch.equal(popped, v)


def test_category_stack_depth_matches_push_count():
    ws = _make_word_space()
    assert ws.category_stack.depth(0) == 0
    ws.category_stack.push(0, torch.zeros(4))
    ws.category_stack.push(0, torch.zeros(4))
    ws.category_stack.push(0, torch.zeros(4))
    assert ws.category_stack.depth(0) == 3


def test_category_stack_flatten_shape():
    """flatten(b) returns [depth * nPoSDim] for rule predictor input."""
    ws = _make_word_space()
    for _ in range(5):
        ws.category_stack.push(0, torch.randn(4))
    flat = ws.category_stack.flatten(0)
    assert flat.shape == (5 * 4,)
