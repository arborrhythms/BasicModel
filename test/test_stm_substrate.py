"""Tests for the STM substrate properties (after the 2026-05-21
WordSubSpace/STM Layer refactor).

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — STM parity work, step 1 ("Fix the substrate
before adding features"):

  * The STM driver is an ``nn.Module`` (now via ``ShortTermMemory``,
    a ``Layer``) so its rule scorer's parameters register through the
    standard ``parameters()`` walk.
  * The typed STM stack is part of an ``nn.Module`` (now WordSubSpace
    itself, inherited from SubSpace) with the parallel tensors
    registered as buffers so ``.to(device)`` moves them together.
  * ``category_embedding`` parameters land in ``WordSpace.params`` (the
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

from _stm_test_fixtures import make_typed_stack, make_driver


def test_stm_driver_is_nn_module():
    """``ShortTermMemory`` (the STM driver Layer) is an ``nn.Module`` so
    its registered scorer's parameters are reachable via the standard
    ``parameters()`` walk."""
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    driver = make_driver(ts, rule_signatures=[
        {'lhs_category': 'S', 'lhs_order': 3,
         'rhs_categories': ['S'], 'rhs_orders': [3],
         'op_name': 'not', 'order_delta': 0}], payload_dim=4)
    assert isinstance(driver._stm, nn.Module)


def test_stm_driver_parameters_include_scorer_params():
    """``ShortTermMemory.parameters()`` walks into the registered scorer."""
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    driver = make_driver(ts, rule_signatures=[
        {'lhs_category': 'S', 'lhs_order': 3,
         'rhs_categories': ['S'], 'rhs_orders': [3],
         'op_name': 'not', 'order_delta': 0}], payload_dim=4)
    driver_param_ids = {id(p) for p in driver._stm.parameters()}
    scorer_param_ids = {id(p) for p in driver.scorer.parameters()}
    assert scorer_param_ids
    assert scorer_param_ids.issubset(driver_param_ids)


def test_typed_stack_is_nn_module():
    """The typed STM stack lives on WordSubSpace, an ``nn.Module``, so
    ``.to(device)`` moves all parallel tensors together."""
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    assert isinstance(ts, nn.Module)


def test_typed_stack_tensors_are_registered_buffers():
    """``_buffer`` / ``_category`` / ``_order`` / ``_ref_id`` /
    ``_depth`` are registered as buffers on the WordSubSpace so
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
    """``WordSpace.params`` contains the ``category_embedding`` weight
    so the optimizer-feed list (used by ``getParameters``) reflects
    every gradient-flowing tensor."""
    from test_partition_pos_codebook import _make_word_space
    ws = _make_word_space()
    embed_param = ws.category_embedding.weight
    param_ids = {id(p) for p in ws.params}
    assert id(embed_param) in param_ids


def test_word_space_params_include_stm_driver_scorer_params():
    """Stage 3 (2026-05-27): the STM shift-reduce driver retired
    alongside the chart. The signal router's per-tier scorer
    parameters are still registered via the LanguageLayer's
    nn.Module walk (covered by test_signal_router_layer.py); this
    test stub stays as a placeholder so the suite size is stable."""
    import pytest
    pytest.skip("STM shift-reduce driver retired in Stage 3")
