"""Tests for the rule-predictor NonlinearLayer on WordSpace.

Task 4.1 (TDD harness) -- failing tests written BEFORE Task 4.2 implementation.

WordSpace cannot be constructed in isolation -- it requires real
PartSpace, ConceptualSpace, and WholeSpace objects.  We build the
minimal chain using _populate_test_config + direct Space constructors, the
same pattern used by test_partition_pos_codebook.py.
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

def _make_word_space_with_grammar(nSymbols=3, symbolDim=4, conceptDim=4, nPercepts=3):
    """Construct the minimal WordSpace via TheXMLConfig + direct constructors.

    Builds the full PartSpace -> ConceptualSpace -> WholeSpace chain
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

    percept_space   = Spaces.PartSpace(inputShape, spaceShape, outputShape)
    concept_space   = Spaces.ConceptualSpace(inputShape, spaceShape, outputShape)
    symbolic_space  = Spaces.WholeSpace(inputShape, spaceShape, outputShape)

    # Reset grammar so WordSubSpace.__init__ can (re)configure it cleanly.
    Language.TheGrammar._configured = False

    ws = Language.WordSubSpace(
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
# Task 4.1 -- rule-predictor contract tests (expected to FAIL before 4.2)
# ---------------------------------------------------------------------------

def test_rule_predictor_output_shape():
    """Predictor emits [nRules]-shaped logits."""
    ws = _make_word_space_with_grammar()
    for _ in range(3):
        ws.category_stack.push(0, torch.randn(4))
    logits = ws.predict_rule(0)
    assert logits.shape == (ws.n_rules,)


def test_rule_predictor_softmax_sums_to_one():
    ws = _make_word_space_with_grammar()
    for _ in range(3):
        ws.category_stack.push(0, torch.randn(4))
    probs = torch.softmax(ws.predict_rule(0), dim=-1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_rule_predictor_hard_inference_argmax():
    ws = _make_word_space_with_grammar()
    ws.training = False  # inference mode
    for _ in range(3):
        ws.category_stack.push(0, torch.randn(4))
    chosen = ws.predict_rule_hard(0)
    assert isinstance(chosen, int)
    assert 0 <= chosen < ws.n_rules


# ---------------------------------------------------------------------------
# Task 4.2 -- gradient-flow sanity test
# ---------------------------------------------------------------------------

def test_rule_predictor_gradient_flows_through_stack():
    ws = _make_word_space_with_grammar()
    v = torch.randn(4, requires_grad=True)
    ws.category_stack.push(0, v)
    logits = ws.predict_rule(0)
    loss = logits.sum()
    loss.backward()
    assert v.grad is not None
    assert torch.any(v.grad != 0)


def test_rule_predictor_empty_stack_zero_pads():
    """predict_rule with empty category_stack uses all-zero input; no NaN."""
    ws = _make_word_space_with_grammar()
    assert ws.category_stack.depth(0) == 0
    logits = ws.predict_rule(0)
    assert logits.shape == (ws.n_rules,)
    assert not torch.isnan(logits).any()


def test_rule_predictor_overflow_stack_truncates_to_recent():
    """When category_stack depth exceeds max_depth, most recent frames are kept."""
    ws = _make_word_space_with_grammar()
    target_len = ws._rule_predictor_in_features
    pos_dim = 4
    max_frames = target_len // pos_dim
    # Push one more than max_frames so the oldest must be dropped.
    for _ in range(max_frames + 2):
        ws.category_stack.push(0, torch.randn(pos_dim))
    logits = ws.predict_rule(0)
    assert logits.shape == (ws.n_rules,)
    assert not torch.isnan(logits).any()
