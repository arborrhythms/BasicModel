"""Tests for step 1 of the CKY-retirement path: substrate fixes.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — STM parity work, step 1 ("Fix the substrate
before adding features"):

  * ``STMDriver`` is an ``nn.Module`` so ``RuleScorer`` parameters
    register through the standard ``parameters()`` walk.
  * ``TypedStack`` is an ``nn.Module`` with ``register_buffer`` so
    ``.to(device)`` moves all parallel tensors together.
  * ``category_embedding`` parameters land in ``WordSpace.params``
    (the manual optimizer-feed list, not just ``parameters()``).

Without these, the STM scorer trains on a stale optimizer copy, the
typed stack stays on CPU when ``.to(device)`` fires elsewhere, and
the category embedding doesn't appear in optimizer state.
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


def test_stm_driver_is_nn_module():
    """``STMDriver`` inherits ``nn.Module`` so its registered scorer's
    parameters are reachable via the standard ``parameters()`` walk."""
    from stm_driver import STMDriver, RuleScorer
    from typed_stack import TypedStack
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=2)
    driver = STMDriver(typed_stack=ts, rule_signatures=[], scorer=scorer)
    assert isinstance(driver, nn.Module)


def test_stm_driver_parameters_include_scorer_params():
    """``STMDriver.parameters()`` walks into the registered scorer."""
    from stm_driver import STMDriver, RuleScorer
    from typed_stack import TypedStack
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=2)
    driver = STMDriver(typed_stack=ts, rule_signatures=[], scorer=scorer)
    driver_param_ids = {id(p) for p in driver.parameters()}
    scorer_param_ids = {id(p) for p in scorer.parameters()}
    assert scorer_param_ids
    assert scorer_param_ids.issubset(driver_param_ids)


def test_typed_stack_is_nn_module():
    """``TypedStack`` is an ``nn.Module`` so ``.to(device)`` moves all
    parallel tensors together."""
    from typed_stack import TypedStack
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    assert isinstance(ts, nn.Module)


def test_typed_stack_tensors_are_registered_buffers():
    """``_buffer`` / ``_category`` / ``_order`` / ``_ref_id`` /
    ``_depth`` are registered as buffers so ``.to(device)`` and
    ``state_dict`` see them as a single module."""
    from typed_stack import TypedStack
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    buffer_names = {n for n, _ in ts.named_buffers()}
    expected = {'_buffer', '_category', '_order', '_ref_id', '_depth'}
    assert expected.issubset(buffer_names), (
        f"missing buffers: {expected - buffer_names}")


def test_typed_stack_to_moves_all_tensors():
    """``ts.to(device)`` propagates to every parallel tensor. Smoke
    test that all five buffers travel together — we re-route to CPU
    (the only universally-available target) so the test is portable
    across MPS / CUDA / CPU."""
    from typed_stack import TypedStack
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    ts.push(0, torch.tensor([1.0, 2.0, 3.0, 4.0]),
            category_id_str='X', order=0, ref_id=0)
    cpu_ts = ts.to('cpu')
    # Every registered buffer lands on the requested device.
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
    """Once the STM driver is constructed, its scorer's parameters
    land in ``WordSpace.params`` too."""
    from test_partition_pos_codebook import _make_word_space
    from Language import Grammar
    from embed import build_knowledge_section, KnowledgeView
    ws = _make_word_space()
    # Build minimal view + attach + parser_backend so _init_stm_driver runs
    g = Grammar()
    g.rules = [
        g._parse_rule("NP", "conjunction(DET, N)", tier='S'),
        g._parse_rule("S", "disjunction(NP, VP)", tier='S'),
    ]
    g._configured = True
    import Language
    Language.TheGrammar = g
    view = KnowledgeView(build_knowledge_section(g))
    ws.attach_knowledge(view)
    ws.parser_backend = 'stm'
    # Force driver construction
    ws._init_stm_driver()
    driver_params = list(ws.stm_driver.parameters())
    assert driver_params
    param_ids = {id(p) for p in ws.params}
    for p in driver_params:
        assert id(p) in param_ids, (
            f"scorer param {p.shape} missing from ws.params")
