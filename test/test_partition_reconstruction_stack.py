"""Tests for the reconstruction stack (rule_ids + word_ids) on SymbolSpace.

Task 5.1 (TDD harness) -- failing tests written BEFORE Task 5.2 implementation.

SymbolSpace cannot be constructed in isolation -- it requires real
PartSpace, ConceptualSpace, and WholeSpace objects.  We build the
minimal chain using _populate_test_config + direct Space constructors, the
same pattern used by test_partition_pos_codebook.py and
test_partition_rule_predictor.py.
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
    """Construct the minimal SymbolSpace via TheXMLConfig + direct constructors.

    Builds the full PartSpace -> ConceptualSpace -> WholeSpace chain
    required by SymbolSpace.__init__, then constructs SymbolSpace from it.

    SymbolSpace is NOT an isolated object: it back-wires all three home spaces
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

    percept_space   = Spaces.PartSpace(inputShape, spaceShape, outputShape)
    concept_space   = Spaces.ConceptualSpace(inputShape, spaceShape, outputShape)
    symbolic_space  = Spaces.WholeSpace(inputShape, spaceShape, outputShape)

    # Reset grammar so SymbolSubSpace.__init__ can (re)configure it cleanly.
    Language.TheGrammar._configured = False

    ss = Language.SymbolSubSpace(
        perceptualSpace=percept_space,
        conceptualSpace=concept_space,
        wholeSpace=symbolic_space,
        nPercepts=nPercepts,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        concept_dim=conceptDim,
        symbol_dim=symbolDim,
    )
    return ss


# ---------------------------------------------------------------------------
# Task 5.1 -- reconstruction stack contract tests (expected to FAIL before 5.2)
# ---------------------------------------------------------------------------

def test_reconstruction_stack_records_rule_and_word():
    ss = _make_word_space()
    ss.record_derivation(rule_id=3, word_id=7)
    ss.record_derivation(rule_id=1, word_id=9)
    assert ss.reconstruction_stack.depth(0) == 2
    top = ss.reconstruction_stack.peek(0)
    assert top == (1, 9)


def test_reconstruction_stack_roundtrip():
    ss = _make_word_space()
    entries = [(2, 5), (0, 8), (4, 1)]
    for rid, wid in entries:
        ss.record_derivation(rule_id=rid, word_id=wid)
    for rid, wid in reversed(entries):
        assert ss.reconstruction_stack.pop(0) == (rid, wid)
    assert ss.reconstruction_stack.depth(0) == 0


# ---------------------------------------------------------------------------
# Task 5.3 -- discourse predictor input width narrowed to S only
# ---------------------------------------------------------------------------

def test_sentence_prediction_uses_root_slot_only():
    """Discourse predictor consumes the root S-space_role slot, not the full
    [S | W] snapshot.

    Pre-2026-05-14 the contrastive layer flattened ``n_symbols * n_dim``
    into the QKVAttentionLayer predictor's input; the bin would blow past
    the allocator on MM_5M_bivector-scale configs (V_S * D > 100k).
    The ARMA(p, q) layer pools sentence rep to the **root cell** (the
    start-symbol reduction's slot) -- a single ``[n_dim]`` vector --
    and the MLP predictor's input is ``(p + q) * n_dim``, bounded
    regardless of ``n_symbols``.
    """
    import Layers
    n_symbols = 4
    max_depth = 6
    n_dim = 8
    concept_dim = 12
    p, q = 5, 2

    discourse = Layers.InterSentenceLayer(
        n_symbols=n_symbols,
        max_depth=max_depth,
        n_dim=n_dim,
        p=p, q=q,
        concept_dim=concept_dim,
    )
    assert discourse.predictor is not None
    assert discourse.sentence_dim == n_dim, (
        "Sentence rep is the root S-space_role slot of width n_dim, not the "
        "full n_symbols * n_dim flatten.")
    in_dim = discourse.predictor[0].in_features
    assert in_dim == (p + q) * n_dim, (
        f"predictor input width should be (p+q)*n_dim = {(p+q)*n_dim}, "
        f"got {in_dim}")
