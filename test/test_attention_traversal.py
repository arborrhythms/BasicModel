"""One supported candidate per read, with an independent work allowance."""
import torch
from AttentionTraversal import FieldTraversal
from SentenceCredit import score_function


def _field(allowance=8):
    spans = torch.tensor([[[0., 1.], [1., 2.], [2., 3.]]])
    return FieldTraversal(torch.ones(1, 3, 2, 2), spans, spans,
                          torch.ones(1, 3, dtype=torch.bool), torch.tensor([allowance]))


def test_disabled_reads_source_order_once_and_enabled_uses_candidate_scores():
    field = _field()
    assert field.select(None).tolist() == [[True, False, False]]
    assert field.select(torch.tensor([[99., 0., 2.]])).tolist() == [[False, False, True]]
    assert field.select(None).tolist() == [[False, True, False]]
    assert field.iterations.tolist() == [3]
    assert not field.active.any()


def test_no_evidence_is_ineligible_but_against_only_is_supported():
    field = _field()
    field = FieldTraversal(torch.tensor([[[[0., 0.]], [[0., 1.]], [[1., 0.]]]]),
        field.where, field.when, field.valid, field.allowance)
    assert field.select(None).tolist() == [[False, True, False]]
    assert field.valid.tolist() == [[False, True, True]]


def test_uniform_candidate_departure_reaches_the_only_alternative_and_is_credited():
    field = _field()
    field.remaining[0, 2] = False
    scores = torch.tensor([[2., 1., -1.]], requires_grad=True)
    admitted = field.select(scores, alternative=torch.tensor([True]), greedy=torch.tensor([0]),
                            scale=torch.tensor([6]))
    assert admitted.tolist() == [[False, True, False]]
    backup = field.backup
    assert backup['scale'].item() == 6
    loss, _ = score_function(backup['probability'][:, None], backup['scale'][:, None],
        backup['rows'][:, None], torch.tensor([[2., 1.]]))
    loss.sum().backward()
    assert scores.grad[0, 1] < 0 < scores.grad[0, 0]
    assert scores.grad[0, 2] == 0


def test_budget_counts_reads_and_does_not_change_the_lanes():
    field = _field(1)
    before = field.support.clone()
    field.select(None)
    assert field.iterations.tolist() == [1]
    assert not field.active.any()
    torch.testing.assert_close(field.support, before)
