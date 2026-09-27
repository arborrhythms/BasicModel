"""Hard-path provenance and the exploration runtime transaction."""
from types import SimpleNamespace

import pytest
import torch

from Models import BasicModel
from Language import LanguageSpace


def _program_fixture():
    ids = torch.full((1, 14), -1, dtype=torch.long)
    arities = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.zeros_like(ids)
    # Push a, push b, then rewrite the older a and fold (rewrite(a), b).
    ids[0, 3:5] = torch.tensor([11, 10])
    arities[0, 3:5] = torch.tensor([1, 2])
    mask[0, 3:5] = True
    positions[0, 3] = 1
    trace = SimpleNamespace(choices=lambda: (ids, arities, mask),
                            _choice_positions=positions)
    language = SimpleNamespace(_cs_binary_rule_ids=torch.tensor([10]),
                               _cs_unary_rule_ids=torch.tensor([11]),
                               local_op_from_rule_ids=LanguageSpace.local_op_from_rule_ids)
    model = SimpleNamespace(_reconstruction_stack=lambda: trace, languageSpace=language,
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1, 2, dtype=torch.bool)),
        conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=2)), _walk_budget=lambda: 16)
    return model, trace, ids


def test_older_slot_unary_is_preserved_in_the_committed_postfix_program():
    model, _, _ = _program_fixture()
    positions, actions, targets = BasicModel._derivation_program(model)
    assert positions.tolist() == [[0, 1]]
    assert actions.tolist() == [[[0, -1, 0], [2, 0, -1], [0, -1, 1], [1, 0, -1]]]
    # Unary doubling and ordered binary subtraction expose an operand swap.
    stack = []
    for kind, _, word in actions[0].tolist():
        if kind == 0:
            stack.append([3., 5.][word])
        elif kind == 2:
            stack[-1] *= 2
        else:
            right, left = stack.pop(), stack.pop()
            stack.append(left - right)
    assert stack == [1.]
    assert targets[0, :4].tolist() == [0, 2, 1, 2]


def test_invalid_unary_position_is_not_silently_dropped():
    model, trace, _ = _program_fixture()
    trace._choice_positions[0, 3] = 2
    with pytest.raises(ValueError, match='unary position'):
        BasicModel._derivation_program(model)


def test_unknown_recorded_operation_is_not_silently_dropped():
    model, _, ids = _program_fixture()
    ids[0, 3] = 99
    with pytest.raises(ValueError, match='unknown recorded operation'):
        BasicModel._derivation_program(model)








def test_incomplete_forest_keeps_its_slots_for_reconstruction():
    buffer = torch.tensor([[[2., 3.], [5., 7.], [0., 0.]]])
    model = SimpleNamespace(conceptualSpace=SimpleNamespace(
        stm=SimpleNamespace(_buffer=buffer)))
    slots, depth = BasicModel._final_end_state(model, buffer[:, 1], torch.tensor([-2]))
    torch.testing.assert_close(slots, buffer)
    assert depth.tolist() == [-1]  # still ineligible for memory


def test_parallel_completion_guards_memory_without_serial_depths():
    model = SimpleNamespace(serial=False, symbolSpace=SimpleNamespace(languageLayer=SimpleNamespace(
        _last_derivation={'complete': torch.tensor([True, False])})))
    assert BasicModel._compose_completed_rows(model).tolist() == [True, False]
    # A previous serial sentence has a different batch width. Parallel
    # context must use its own completion mask, not the stale serial depth.
    model._stm_post_depth = torch.tensor([-2])
    assert BasicModel._compose_completed_rows(model).tolist() == [True, False]
    model.serial = True
    model._stm_post_depth = torch.tensor([-2, 1])
    assert BasicModel._compose_completed_rows(model).tolist() == [False, True]


def test_exploration_forces_one_used_round_for_each_packed_sentence():
    B, W, cap = 2, 4, 2
    actions = torch.full((B, 3 * W + W * 2 * cap), -1, dtype=torch.long)
    # Interleaved rows have distinct sets of used rounds, including both seals.
    actions[0, [0, 3, 20, 6, 9, 12]] = 2
    actions[1, [1, 21, 7, 13]] = 1
    attempted = actions >= 0
    model = SimpleNamespace(
        _reconstruction_stack=lambda: SimpleNamespace(_choice_actions=actions, _choice_attempted=attempted),
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(B, W, dtype=torch.bool),
            _packed_sentence_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])),
        conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=cap)))
    model._compose_round_owners = lambda actions: BasicModel._compose_round_owners(model, actions)
    copied, forced, owners = BasicModel._exploration_constraints(model)
    assert torch.equal(copied, actions)
    assert not bool((forced & ~attempted).any())
    for sid in (0, 1):
        assert ((owners == sid) & forced).sum(1).tolist() == [1, 1]














def test_one_word_packed_sentence_has_its_own_seal_and_program():
    model, trace, ids = _program_fixture()
    model.inputSpace._sentence_pack_enabled = True
    model.inputSpace._packed_sentence_ids = torch.tensor([[0, 1]])
    ids.fill_(-1)
    _ids, arities, mask = trace.choices()
    arities.zero_(); mask.zero_(); trace._choice_positions.zero_()
    # W=2, cap=2: final group at 6, first-word intermediate group at 10.
    ids[0, 10] = 11; arities[0, 10] = 1; mask[0, 10] = True
    owners, _, _ = BasicModel._compose_round_owners(model, ids)
    assert owners[0, 6:10].tolist() == [1] * 4
    assert owners[0, 10:14].tolist() == [0] * 4
    _, first, _ = BasicModel._derivation_program(model, t=0)
    final_positions, final, _ = BasicModel._derivation_program(model, t=1)
    assert first[0, :2].tolist() == [[0, -1, 0], [2, 0, -1]]
    assert final_positions.tolist() == [[1]]
    assert final[0, :1].tolist() == [[0, -1, 0]]


@pytest.mark.parametrize('seal_budget', (0, 1))
def test_tensor_pipeline_records_both_one_word_packed_seals(tmp_path, monkeypatch, seal_budget):
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_reverse_traversal import _stage_packed
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
                                  concept_rows=256, part_rows=128)
    model._tensor_peer_while_eager = True
    model.syntacticOrder = seal_budget
    # Unary-only rounds keep occupancy fixed and exhaust the declared budget.
    # That exposes a packed/final budget mismatch without a random STOP.
    chooser = model.languageSpace.language_layer.operation_layer.chooser
    def binary(x, candidates, *_args, **_kwargs):
        return (x.new_full((*x.shape[:2], 1), -1e6),
                x.new_full(candidates.shape[:-1], -1e6))
    def unary(x, candidates, *_args, **_kwargs):
        scores = x.new_full(candidates.shape[:-1], -1e6)
        scores[..., 0] = 0.
        return x.new_full((*x.shape[:2], 1), -1e6), scores
    chooser.score_binary, chooser.score_unary = binary, unary
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    model._install_unit_span_fn()
    try:
        _stage_packed(model, [['a', 'b']])
        with torch.no_grad():
            result = model._forward_with_compiled_sentence_state(None)
        model._publish_compiled_sentence_state(result)
        trace = model._reconstruction_stack()
        W = model.inputSpace._word_active_mask.shape[1]
        width = 2 * model.conceptualSpace.stm.capacity
        assert bool(trace._choice_attempted[0, 3 * W])
        assert bool(trace._choice_attempted[0, 3 * W + width])
        expected = min(seal_budget, width) if seal_budget > 0 else width
        assert int(trace._choice_attempted[0, 3 * W:3 * W + width].sum()) == expected
        assert int(trace._choice_attempted[0, 3 * W + width:3 * W + 2 * width].sum()) == expected
    finally:
        model.End()
        model.symbolSpace.soft_reset()
