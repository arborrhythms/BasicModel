"""Reviewer probes for §10.4 measurement isolation and accounting."""
import torch
import pytest

from bench_sentence_expectation import meaning_windows, summarize_steps


def test_windows_exclude_targets_future_sentences_and_other_documents():
    first = torch.arange(4 * 3 * 2.).reshape(4, 3, 2).requires_grad_()
    second = torch.full((3, 3, 2), 100.).requires_grad_()
    values, masks, targets, addresses, target_masks = meaning_windows([first, second], 3)
    assert addresses == [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    assert len(targets) == (len(first)-1) + (len(second)-1)
    torch.testing.assert_close(values[0, -1], first[0])
    torch.testing.assert_close(values[3, -1], second[0])
    assert torch.count_nonzero(values[3, :-1]) == 0
    assert masks[3].tolist() == [[False]*3, [False]*3, [True]*3]
    assert target_masks.all()
    assert not targets.requires_grad
    changed = first.detach().clone()
    changed[2:] += 99
    later = second.detach() * -3
    altered = meaning_windows([changed, later], 3)
    torch.testing.assert_close(altered[0][:2], values[:2])


def test_masked_target_roles_and_history_are_reported_separately():
    values = torch.arange(18.).reshape(3, 3, 2)
    occupied = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    x, masks, target, addresses, target_mask = meaning_windows([values], 2, [occupied])
    assert masks[1].tolist() == [[True, False, False], [True, True, False]]
    assert target_mask[1].tolist() == [True, True, True]
    assert addresses[1] == (0, 2)


def test_throughput_excludes_warmup_and_counts_actual_sentences_pairs_and_labels():
    steps = [dict(seconds=100, input_sentences=99, observations=99, predicted_targets=98,
                  cold_starts=1, reconstruction=10., answer=10., supplied_answer_rows=99),
             dict(seconds=2, input_sentences=4, observations=4, predicted_targets=3,
                  cold_starts=1, reconstruction=2., answer=1., supplied_answer_rows=2),
             dict(seconds=3, input_sentences=2, observations=2, predicted_targets=2,
                  cold_starts=0, reconstruction=4., answer=3., supplied_answer_rows=1)]
    measured = summarize_steps(steps, warmup=1)
    assert measured['input_sentences'] == 6
    assert measured['predicted_targets'] == 5
    assert measured['supplied_answer_rows'] == 3
    assert measured['input_sentences_per_second'] == 1.2
    assert measured['predicted_targets_per_second'] == 1.
    assert measured['supplied_answer_rows_per_second'] == .6
    assert measured['document_boundary_fraction'] == pytest.approx(1/6)
    assert measured['reconstruction_mean'] == 3
    assert measured['supplied_answer_mean'] == 2
    assert measured['warmup_seconds'] == 100
    with pytest.raises(ValueError, match='warmup'):
        summarize_steps(steps, warmup=3)


def test_sentence_rate_is_independent_of_predictor_observation_eligibility():
    steps = [dict(seconds=2, input_sentences=3, observations=0, predicted_targets=0,
                  cold_starts=0, reconstruction=1., answer=1., supplied_answer_rows=3)]
    measured = summarize_steps(steps, warmup=0)
    assert measured["input_sentences"] == 3
    assert measured["input_sentences_per_second"] == 1.5
    assert measured["observations"] == 0
    assert measured["predicted_targets"] == 0


def test_counts_one_joint_step_for_native_dense_and_sparse_optimizer():
    from bench_sentence_expectation import observe_optimizer_steps
    from test_multi_optimizer_surface import _two_family, _step

    dense, sparse, optimizer = _two_family()
    original_step = optimizer.step
    calls = []
    handle = observe_optimizer_steps(optimizer, lambda *_: calls.append(1))
    _step(dense, sparse, optimizer)
    assert calls == [1]
    for parameter in list(dense.parameters()) + list(sparse.parameters()):
        assert parameter in optimizer.state
    handle.remove()
    assert optimizer.step == original_step
    _step(dense, sparse, optimizer)
    assert calls == [1]


def test_fixed_meaning_controls_keep_the_input_device_under_an_ambient_device():
    from bench_sentence_expectation import _controlled_inputs

    rows = torch.arange(5 * 3 * 2., device="cpu").reshape(5, 3, 2)
    cpu = meaning_windows([rows], 3)
    expected = _controlled_inputs(cpu[0], cpu[1], "shuffled", 27)
    ambient = "mps" if torch.backends.mps.is_available() else "meta"
    with torch.device(ambient):
        views = meaning_windows([rows], 3)
        assert all(views[i].device.type == "cpu" for i in (0, 1, 2, 4))
        shuffled = _controlled_inputs(views[0], views[1], "shuffled", 27)
    torch.testing.assert_close(shuffled[0], expected[0])
    torch.testing.assert_close(shuffled[1], expected[1])
