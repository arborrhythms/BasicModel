"""Tests for the STM substrate properties (after the 2026-05-21
SymbolSubSpace/STM Layer refactor).

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — STM parity work, step 1 ("Fix the substrate
before adding features"):

  * The STM driver is an ``nn.Module`` (now via ``ShortTermMemory``,
    a ``Layer``) so its rule scorer's parameters register through the
    standard ``parameters()`` walk.
  * The typed STM stack is part of an ``nn.Module`` (now SymbolSubSpace
    itself, inherited from SubSpace) with the parallel tensors
    registered as buffers so ``.to(device)`` moves them together.
  * ``category_embedding`` parameters land in ``SymbolSpace.params`` (the
    manual optimizer-feed list, not just ``parameters()``).
"""
import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

from _stm_test_fixtures import make_typed_stack

# The shift/reduce rule-scorer (STMDriver / _RuleScorer) these two tests
# exercised was deleted in the 2026-07-17 cleanup (Tier-2 item 8; zero
# production callers). The live typed-stack substrate tests below remain.


def test_typed_stack_is_nn_module():
    """The typed STM stack lives on SymbolSubSpace, an ``nn.Module``, so
    ``.to(device)`` moves all parallel tensors together."""
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    assert isinstance(ts, nn.Module)


def test_typed_stack_tensors_are_registered_buffers():
    """``_buffer`` / ``_category`` / ``_order`` / ``_ref_id`` /
    ``_depth`` are registered as buffers on the SymbolSubSpace so
    ``.to(device)`` and ``state_dict`` see them as a single module."""
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    buffer_names = {n for n, _ in ts.named_buffers()}
    expected = {'_buffer', '_category', '_order', '_ref_id', '_depth'}
    assert expected.issubset(buffer_names), (
        f"missing buffers: {expected - buffer_names}")


def test_typed_stack_to_moves_all_tensors():
    """``ts.to(device)`` propagates to every parallel tensor."""
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    ts.push(0, torch.tensor([1.0, 2.0, 3.0, 4.0]),
            category_id_str='X', order=0, ref_id=0)
    cpu_ts = ts.to('cpu')
    for name, buf in cpu_ts.named_buffers():
        assert buf.device.type == 'cpu', (
            f"buffer {name} on {buf.device}")


def test_word_space_params_include_category_embedding():
    """``SymbolSpace.params`` contains the ``category_embedding`` weight
    so the optimizer-feed list (used by ``getParameters``) reflects
    every gradient-flowing tensor."""
    from test_partition_pos_codebook import _make_word_space
    ss = _make_word_space()
    embed_param = ss.category_embedding.weight
    param_ids = {id(p) for p in ss.params}
    assert id(embed_param) in param_ids
