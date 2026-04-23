"""Tests for the reconstruction stack (rule_ids + word_ids) on WordSpace.

Task 5.1 (TDD harness) -- failing tests written BEFORE Task 5.2 implementation.

WordSpace cannot be constructed in isolation -- it requires real
PerceptualSpace, ConceptualSpace, and SymbolicSpace objects.  We build the
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
# Task 5.1 -- reconstruction stack contract tests (expected to FAIL before 5.2)
# ---------------------------------------------------------------------------

def test_reconstruction_stack_records_rule_and_word():
    ws = _make_word_space()
    ws.record_derivation(rule_id=3, word_id=7)
    ws.record_derivation(rule_id=1, word_id=9)
    assert ws.reconstruction_stack.depth(0) == 2
    top = ws.reconstruction_stack.peek(0)
    assert top == (1, 9)


def test_reconstruction_stack_roundtrip():
    ws = _make_word_space()
    entries = [(2, 5), (0, 8), (4, 1)]
    for rid, wid in entries:
        ws.record_derivation(rule_id=rid, word_id=wid)
    for rid, wid in reversed(entries):
        assert ws.reconstruction_stack.pop(0) == (rid, wid)
    assert ws.reconstruction_stack.depth(0) == 0


# ---------------------------------------------------------------------------
# Task 5.3 -- discourse predictor input width narrowed to S only
# ---------------------------------------------------------------------------

def test_sentence_prediction_consumes_s_only():
    """Discourse predictor input width equals ``n_symbols * n_dim`` (S only),
    not the full ``[S | W]`` ``snapshot_dim``.

    Task 5.3 deleted the ``[S | W]`` augmentation of the prediction head:
    W is the per-sentence WordSpace buffer's own concern and was redundant
    input to the sentence-level AR predictor.  The discourse substrate's
    snapshot history, contrastive loss, and ``_assemble`` all still carry
    the full ``[S | W]`` rows -- only the predictor narrowed.

    WordSpace gates discourse on ``TheXMLConfig.training("sentencePrediction")``
    and that flag is off in the minimal test config.  Rather than also
    rewiring the test config, instantiate InterSentenceLayer directly --
    it's the unit of interest and has no hidden coupling to the surrounding
    space stack.
    """
    import Layers
    n_symbols = 4
    max_depth = 6
    n_dim = 8
    concept_dim = 12

    discourse = Layers.InterSentenceLayer(
        n_symbols=n_symbols,
        max_depth=max_depth,
        n_dim=n_dim,
        context_window=3,
        centroid_history=2,
        lam=1.01,
        concept_dim=concept_dim,
    )
    assert discourse.predictor is not None, (
        "InterSentenceLayer should build the predictor when concept_dim "
        "is provided.")

    expected = n_symbols * n_dim
    # s_dim is the attribute Task 5.3 exposes for the S-only flattened
    # width; keep it in the invariant for readability.
    assert discourse.s_dim == expected

    # AttentionLayer inherits nInput/nOutput from Layer; the public
    # dimension attribute exposed by the codebase is ``nInput``.  Fall
    # back to ``in_features`` for compatibility with the plan's naming.
    actual = getattr(discourse.predictor, 'nInput',
                     getattr(discourse.predictor, 'in_features', None))
    assert actual is not None, (
        "AttentionLayer did not expose nInput (or in_features) for "
        "width introspection")
    assert actual == expected, (
        f"predictor input width should be n_symbols * n_dim = {expected} "
        f"(S-only), got {actual}")
    assert actual < discourse.snapshot_dim, (
        f"predictor width {actual} should be strictly less than the full "
        f"[S | W] snapshot_dim {discourse.snapshot_dim}; W block was not "
        f"removed from the prediction head.")
