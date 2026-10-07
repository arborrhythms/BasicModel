"""Constituent-stack ownership and the serial/parallel sigma contract.

WholeSpace's retired word-dictionary SHIFT fixtures are dispositioned in the
item-7 review receipt; these tests exercise the live SymbolSubSpace methods.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch
import torch.nn as nn

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

from Layers import SigmaLayer2

D = 4
CAP = 8




def make_ws(batch=1, dim=D, cap=CAP):
    """Bare SymbolSubSpace with idea-stack buffers (the established
    object.__new__ fixture idiom from _stm_test_fixtures)."""
    from Language import SymbolSubSpace
    ss = object.__new__(SymbolSubSpace)
    nn.Module.__init__(ss)
    ss._stm_payload_dim = int(dim)
    ss._idea_capacity = int(cap)
    ss._idea_max_depth_host = 0
    ss._idea_buffer = torch.zeros(batch, cap, dim)
    ss._idea_depth = torch.zeros(batch, dtype=torch.long)
    return ss








# ---------------------------------------------------------------------------
# The SS-side constituent stack.
# ---------------------------------------------------------------------------

def test_constituent_stack_mechanics():
    ss = make_ws()
    assert ss.constituent_depth_of(0) == 0, "dark until first use"
    c1, c2 = torch.rand(D), torch.rand(D)
    ss.constituent_push(0, c1)
    ss.constituent_push(0, c2)
    assert ss.constituent_depth_of(0) == 2
    assert torch.equal(ss.constituent_peek(0, 0), c2), "newest at slot 0"
    assert torch.equal(ss.constituent_peek(0, 1), c1)
    top = ss.constituent_pop(0)
    assert torch.equal(top, c2)
    assert ss.constituent_depth_of(0) == 1
    ss.constituent_clear()
    assert ss.constituent_depth_of(0) == 0


def test_constituent_capacity_is_the_miller_cap():
    ss = make_ws()
    for i in range(CAP):
        ss.constituent_push(0, torch.rand(D))
    with pytest.raises(RuntimeError):
        ss.constituent_push(0, torch.rand(D))


def test_split_replaces_whole_with_parts():
    ss = make_ws()
    whole, left, right = torch.rand(D), torch.rand(D), torch.rand(D)
    ss.constituent_push(0, whole)
    ss.constituent_split(0, left, right)
    assert ss.constituent_depth_of(0) == 2
    assert torch.equal(ss.constituent_peek(0, 0), left), (
        "left-to-right analysis: left is newest")
    assert torch.equal(ss.constituent_peek(0, 1), right)


def test_serial_reduce_chain_matches_parallel_sigma_extent():
    # §10.11: for associative content the serial reduce chain's extent
    # matches the parallel σ extent to tolerance. At near-identity init
    # the σ2 kernel is the probabilistic-sum family, whose parallel
    # n-ary extent is 1 − Π(1 − m_i).
    torch.manual_seed(4)
    sig = SigmaLayer2(2 * D, D, blocks=2)
    A = torch.rand(1, D) * 0.3 + 0.2
    B = torch.rand(1, D) * 0.3 + 0.2
    C = torch.rand(1, D) * 0.3 + 0.2
    chain = sig.compose(sig.compose(A, B), C)          # serial reduces
    parallel = 1.0 - (1 - A) * (1 - B) * (1 - C)        # parallel extent
    assert torch.allclose(chain, parallel, atol=0.08), (
        f"reduce chain {chain.tolist()} vs parallel {parallel.tolist()}")
    # And the chain is associative to tolerance (the content is).
    chain2 = sig.compose(sig.compose(A, C), B)
    assert torch.allclose(chain, chain2, atol=0.08)


def test_parallel_mode_leaves_serial_stacks_untouched():
    ss = make_ws(batch=2)
    # The parallel-mode push (idea_push_step) is the whole-slab path;
    # the SS-side analysis stack must stay dark through it.
    ss.idea_push_step(torch.rand(2, D))
    assert getattr(ss, '_constituent_buffer', None) is None
    assert ss.constituent_depth_of(0) == 0
    assert ss.constituent_depth_of(1) == 0
