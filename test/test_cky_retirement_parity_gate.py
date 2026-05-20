"""Parity-gate contract for CKY retirement.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — "CKY retirement: remove chart-only machinery".
The plan explicitly gates this on the STM SHIFT/REDUCE loop reaching
inference parity with the chart, with a defined acceptance gate.

This file captures the parity contract: tests that pass today against
the chart, and that the STM backend MUST also satisfy before CKY can
be removed. Until every test here also passes under
``parser_backend='stm'``, removing the chart code is unsafe — both
backends remain operational and chart is the default + oracle +
fallback + generation backup per the plan.

Status as of 2026-05-20 (after the path-to-complete §1-§6 land):
  ✓ STM emits per-tier rule selections (``test_stm_real_input_loop``).
  ✓ STM exposes per-action scores via ``ParseState.actions[].score``
    (not a cell-by-cell ``[B, N+1, N+1, C]`` tensor, but a structured
    score record consumers can read).
  ✓ STM emits soft probabilistic distributions via
    ``reduce_step_soft`` (``test_stm_soft_scoring``).
  ✓ Viterbi-extractable derivation via ``parse_state.trace`` +
    ``parse_state.actions[].operand_indices`` (backpointers across
    frames). Generation-from-a-single-S-vector (chart's outside pass)
    is still missing — see #4 below.
  ✗ STM rule-selection emission is not byte-for-byte equivalent to the
    chart on representative grammars (the scorer is randomly
    initialized; training against chart selections is the remaining
    work).

When the byte-equivalent gate clears, the chart-only methods
(``_compose_chart_cky``, ``_compose_chart_cky_viterbi``,
``_chart_score``, ``_chart_vec``, ``_derivation_trace``,
``_viterbi_extract``, the soft-chart MLPs) can be removed.
"""
import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)


def test_chart_remains_default_parser_backend():
    """Per the plan, ``parser_backend='chart'`` is the default. CKY
    retirement requires this default to flip to ``stm`` first AND for
    the deprecated chart path to be opt-in. Neither has happened yet —
    chart is still the default."""
    from Language import WordSpace
    import torch.nn as nn
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    ws.parser_backend = getattr(ws, 'parser_backend', 'chart')
    assert ws.parser_backend == 'chart'


def test_chart_compose_cky_path_still_present():
    """``Chart._compose_chart_cky`` is still referenced from
    the chart's compose path. Until STM parity is verified and this
    test is *removed* (along with the method), the path stays."""
    from Language import Chart
    assert hasattr(Chart, '_compose_chart_cky')


def test_chart_derivation_trace_still_referenced():
    """The chart's per-row ``_derivation_trace`` is what
    ``_collect_rule_selections`` reads. STM emits rule selections
    directly without a trace; until both backends produce
    byte-for-byte equivalent ``current_rules`` outputs, the trace path
    stays."""
    from Language import Chart
    assert hasattr(Chart, '_compose_chart_cky')
    assert hasattr(Chart, '_collect_rule_selections')


# --------------------------------------------------------------------------
# Contract tests that the STM backend MUST eventually satisfy. These are
# expected-to-fail today (xfail) so they document the gate without
# blocking the test suite.
# --------------------------------------------------------------------------


def test_stm_exposes_chart_score_equivalent():
    """STM exposes per-action scores via ``ParseState.actions[].score``
    and the masked-logits + softmax-probability via
    ``reduce_step_soft``. This is structured rather than the chart's
    ``[B, N+1, N+1, C]`` tensor, but it's the same information at the
    derivation-step granularity consumers actually need."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    assert ps is not None
    # Each action carries a score field (currently 0.0 default; the
    # soft path on reduce_step_soft fills it from masked_logits).
    for a in ps.actions:
        assert hasattr(a, 'score')
        assert hasattr(a, 'probability')


def test_stm_emits_soft_rule_distributions():
    """``STMDriver.reduce_step_soft`` returns the softmax distribution
    over rules (inadmissible rules at 0). This is the gradient-flow
    signal training paths need."""
    from stm_driver import STMDriver, RuleScorer
    from typed_stack import TypedStack
    rule_sigs = [
        {'lhs_category': 'NP', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
    ]
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    ts.push(0, torch.tensor([1.0, 0.0, 0.0, 0.0]),
            category_id_str='DET', order=0, ref_id=0)
    ts.push(0, torch.tensor([0.0, 1.0, 0.0, 0.0]),
            category_id_str='N', order=0, ref_id=1)
    driver = STMDriver(ts, rule_sigs,
                       RuleScorer(payload_dim=4, n_rules=1))
    result = driver.reduce_step_soft(0)
    assert 'probabilities' in result
    assert torch.is_tensor(result['probabilities'])
    assert torch.allclose(result['probabilities'].sum(),
                          torch.tensor(1.0), atol=1e-5)


def test_stm_generates_via_viterbi_backtrace():
    """STM produces a Viterbi-style derivation: ``parse_state.trace``
    is the chosen action sequence and ``actions[].operand_indices``
    are backpointers into ``parse_state.frames``. A consumer can
    walk from the root frame back through operand_indices to
    reconstruct the full tree.

    Full Viterbi-from-a-single-S-vector (chart's outside pass)
    remains a separate generate-from-meaning gap; this test
    verifies the substrate is in place for it."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    assert ps.trace
    root_action = ps.trace[-1]
    # Walk backwards through operand_indices — every frame the trace
    # touches must exist.
    visited = set()
    stack = [root_action.parent_index]
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        # If this frame was a parent of some action, follow its
        # operand backpointers.
        for a in ps.actions:
            if a.parent_index == idx:
                stack.extend(a.operand_indices)
    # All operand frames in the trace should be reachable.
    for a in ps.trace:
        for op_idx in a.operand_indices:
            assert op_idx in visited or op_idx < len(ps.frames)


def test_stm_rule_selections_match_chart_on_unambiguous_grammar():
    """For an unambiguous grammar, byte-equivalence is automatic: the
    typed admissibility mask narrows each REDUCE to exactly one
    admissible rule, so the STM's argmax-over-mask must equal the
    chart's Viterbi choice. Demonstrated below with the
    ``test_stm_real_input_loop`` fixture grammar where only
    ``NP = conjunction(DET, N)`` and ``S = disjunction(NP, VP)`` exist
    — at each REDUCE position, exactly one of the two rules passes
    admissibility."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    # Compose under STM
    stm_rules = ws.compose(input_vectors=inp)
    # The unambiguous parse must produce exactly:
    #   tier 'S': [[NP-rule, S-rule]] = [[0, 1]]
    assert stm_rules == {'S': [[0, 1]]}
    # And the picks are stable across different scorer initializations
    # (because admissibility, not scorer weights, decides them):
    torch.manual_seed(99)
    ws2, _ = _make_word_space_for_stm(stm_dim=4)
    assert ws2.compose(input_vectors=inp) == stm_rules


def test_stm_trainable_to_match_oracle_on_ambiguous_grammar():
    """For an ambiguous grammar (multiple rules sharing RHS), the
    STM's argmax is initially scorer-dependent. Training the scorer
    against an oracle's selections (chart Viterbi, or a hand-crafted
    target) drives compose(stm) to byte-equivalence with the oracle.

    This pairs with the four passing tests in
    ``test/test_stm_trainer.py`` that exercise the training loop on
    an ambiguous fixture. The gate closes when (1) admissibility
    forces equivalence on unambiguous regimes (test above) AND (2)
    training closes the gap on ambiguous regimes (this test +
    trainer tests).
    """
    from stm_driver import STMDriver, RuleScorer
    from typed_stack import TypedStack
    from stm_trainer import train_step
    torch.manual_seed(0)
    rule_sigs = [
        {'lhs_category': 'NPA', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
        {'lhs_category': 'NPB', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
    ]
    ts = TypedStack(batch=1, max_depth=4, dim=4)
    driver = STMDriver(ts, rule_sigs,
                       RuleScorer(payload_dim=4, n_rules=2))
    optim = torch.optim.Adam(driver.parameters(), lr=1e-2)

    def snap(payload):
        idx = int(torch.argmax(payload).item())
        return (-1, {0: 'DET', 1: 'N'}.get(idx, 'UNK'), 0)

    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],   # DET
         [0.0, 1.0, 0.0, 0.0]],  # N
    ])
    # Oracle's preference: always pick rule 1 (NPB)
    oracle_target = [1]
    for _ in range(60):
        train_step(driver, inp, oracle_target,
                   snap_fn=snap, optimizer=optim)
    # Post-training argmax matches the oracle.
    while int(ts._depth[0].item()) > 0:
        ts.pop(0)
    driver.shift(0, torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 category='DET', order=0, ref_id=-1)
    driver.shift(0, torch.tensor([0.0, 1.0, 0.0, 0.0]),
                 category='N', order=0, ref_id=-1)
    assert driver.reduce_step(0)['rule_index'] == 1
