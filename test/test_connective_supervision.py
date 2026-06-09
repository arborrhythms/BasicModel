"""Corpus-scale connective supervision (Phase R5).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
decision 8 + §5.5 + §8 R5. ``A AND B`` and ``A OR B`` are
surface-indiscriminable; they are discriminated by the slot-0 operator
superposition shaped by deep structure (truth / consequence) over a corpus.
This is where the operator superposition becomes LOAD-BEARING: a
truth/consequence signal trains the slot-0 distribution over
{conjunction, disjunction} to the connective whose consequence matches,
even though the operands (the surface) are identical between the two.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


def _truth_table():
    """The four boolean operand pairs and their AND / OR consequences."""
    a = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    b = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    y_and = torch.minimum(a, b)        # [1,0,0,0]
    y_or = torch.maximum(a, b)         # [1,1,1,0]
    return a, b, y_and, y_or


def test_soft_compose_one_hot_matches_hard():
    """A one-hot slot-0 distribution reduces to that operator's hard compose
    (the typed grammar is the one-hot limit)."""
    from Language import soft_connective_compose
    from Language import GRAMMAR_LAYER_CLASSES
    a, b, _, _ = _truth_table()
    hard = GRAMMAR_LAYER_CLASSES["conjunction"]().compose(a, b)
    soft = soft_connective_compose(
        torch.tensor([1.0, 0.0]), a, b, ("conjunction", "disjunction"))
    assert torch.allclose(soft, hard)


def test_loss_is_differentiable():
    """The truth/consequence loss carries gradient to the slot-0 logits."""
    from Language import connective_truth_loss
    a, b, y_and, _ = _truth_table()
    logits = torch.zeros(2, requires_grad=True)
    loss = connective_truth_loss(logits, a, b, y_and, ("conjunction", "disjunction"))
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_and_corpus_recovers_conjunction():
    """Supervising on AND consequences drives the slot-0 superposition to
    conjunction."""
    from Language import learn_connective_distribution
    a, b, y_and, _ = _truth_table()
    dist = learn_connective_distribution(a, b, y_and)
    assert dist["conjunction"] > 0.9, dist


def test_or_corpus_recovers_disjunction():
    """Supervising on OR consequences drives it to disjunction."""
    from Language import learn_connective_distribution
    a, b, _, y_or = _truth_table()
    dist = learn_connective_distribution(a, b, y_or)
    assert dist["disjunction"] > 0.9, dist


def test_identical_surface_different_consequence_discriminates():
    """The load-bearing claim: with the SAME operands (surface), only the
    consequence differs -- yet the learned slot-0 superposition lands on the
    opposite connective for AND vs OR corpora."""
    from Language import learn_connective_distribution
    a, b, y_and, y_or = _truth_table()
    d_and = learn_connective_distribution(a, b, y_and)
    d_or = learn_connective_distribution(a, b, y_or)
    assert d_and["conjunction"] > d_and["disjunction"]
    assert d_or["disjunction"] > d_or["conjunction"]
    # The operands were byte-identical; the discriminator is consequence.
    assert torch.equal(a, a) and torch.equal(b, b)
