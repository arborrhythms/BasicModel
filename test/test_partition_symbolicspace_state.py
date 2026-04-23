"""End-to-end test of SymbolicSpace.forward over a toy grammar.

Task 6.1 (TDD harness) -- failing tests written BEFORE Task 6.2 implementation.

Task 6.2 will:
- Rewrite ``SymbolicSpace.forward`` to take an ``incoming_subspace`` as first
  arg (not the current ``vspace``).
- Add ``SymbolicSpace._build_incoming_subspace(pos_vector=...)`` helper.
- Add ``_op_for_rule``, ``_superposed_op`` helpers + per-rule ``_op_<name>``
  methods.

WordSpace cannot be constructed in isolation -- it requires real
PerceptualSpace, ConceptualSpace, and SymbolicSpace objects.  We build the
minimal chain using _populate_test_config + direct Space constructors, the
same pattern used by test_partition_pos_codebook.py and
test_partition_rule_predictor.py.  Unlike those builders we return BOTH the
SymbolicSpace and the WordSpace so the tests can exercise
``sym.forward(incoming, wordSpace=ws)`` directly.
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
from Spaces import SymbolicSpace
from Language import WordSpace
from test_basicmodel import _populate_test_config


# ---------------------------------------------------------------------------
# Minimal space chain builder -- returns BOTH SymbolicSpace and WordSpace
# ---------------------------------------------------------------------------

def _make_integrated_system(nSymbols=3, symbolDim=4, conceptDim=4, nPercepts=3):
    """Builds a linked (SymbolicSpace, WordSpace) pair on a toy grammar.

    Identical to the chain builders in test_partition_pos_codebook.py and
    test_partition_rule_predictor.py, but returns both the SymbolicSpace and
    the WordSpace so callers can invoke ``sym.forward(incoming, wordSpace=ws)``
    directly.
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
    inputShape   = [nPercepts, conceptDim]
    spaceShape   = [nSymbols,  symbolDim]
    outputShape  = [nSymbols,  symbolDim]

    percept_space  = Spaces.PerceptualSpace(inputShape, spaceShape, outputShape)
    concept_space  = Spaces.ConceptualSpace(inputShape, spaceShape, outputShape)
    symbolic_space = Spaces.SymbolicSpace(inputShape, spaceShape, outputShape)

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
    return symbolic_space, ws


# ---------------------------------------------------------------------------
# Task 6.1 -- SymbolicSpace.forward contract tests (expected to FAIL before 6.2)
# ---------------------------------------------------------------------------

def test_forward_updates_self_subspace_from_incoming():
    sym, ws = _make_integrated_system()
    sym.wordSpace = ws
    sym.subspace.what.setW(torch.zeros_like(sym.subspace.what.getW()))
    incoming = sym._build_incoming_subspace(pos_vector=torch.tensor([0.5, 0.0, 0.3]))
    sym.forward(incoming)
    out = sym.subspace.what.getW()
    assert not torch.equal(out, torch.zeros_like(out)), (
        "forward() did not update self.subspace.what"
    )


def test_multiple_forwards_accumulate_until_percepts_exhausted():
    sym, ws = _make_integrated_system()
    sym.wordSpace = ws
    percepts = [torch.tensor([0.5, 0.0, 0.0]),
                torch.tensor([0.0, 0.4, 0.0]),
                torch.tensor([0.2, 0.1, 0.3])]
    for p in percepts:
        incoming = sym._build_incoming_subspace(pos_vector=p)
        sym.forward(incoming)
    assert ws.pos_stack.depth(0) == len(percepts)


# ---------------------------------------------------------------------------
# Task 7.1 -- characterization: chunking behavior must be unchanged by the
# grammar-dispatch refactor.  PerceptualSpace.chunk_static is a pure static
# method (no space state), so we can call it without constructing any Space
# chain.  Covers the raw and lexicon modes that survive the refactor.
# ---------------------------------------------------------------------------

def test_chunk_static_modes_smoke():
    """Smoke-test the three chunking modes.  Task 7.2 removed the
    grammar-client call from PerceptualSpace.forward; ``chunk_static``
    itself was never in scope, so this is a belt-and-braces sentinel
    against accidental drift of the three-way switch.
    """
    lex = Spaces.PerceptualSpace.chunk_static(b"the cat sat", mode="lexicon")
    assert lex == [b"the", b"cat", b"sat"]
    raw = Spaces.PerceptualSpace.chunk_static(b"abc", mode="raw")
    assert raw == [b"a", b"b", b"c"]
    bpe = Spaces.PerceptualSpace.chunk_static(b"abc", mode="bpe")
    assert bpe == [b"a", b"b", b"c"]
