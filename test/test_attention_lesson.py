"""Labels name decoded words, without selecting admission or stopping."""
from types import SimpleNamespace

import torch

from AttentionLesson import PartLesson


def test_part_supervision_scores_missing_and_extra_leaves_after_decoding():
    bank = SimpleNamespace(bytes=torch.tensor([[[97, 0], [98, 0]]]),
        byte_valid=torch.tensor([[[True, False], [True, False]]]), valid=torch.ones(1, 2, dtype=torch.bool),
        codes=torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True))
    record = SimpleNamespace(primed=bank)
    lesson = PartLesson(torch.zeros(1, 2), ((b'b',),))
    actual = torch.tensor([[[0., 1.], [1., 0.]]], requires_grad=True)
    error, baseline = lesson.cost(record, (actual, torch.tensor([2])))
    assert error.item() == baseline.item() == 1.
    error.sum().backward()
    assert bank.codes.grad is None
    torch.testing.assert_close(actual.grad[0, 1], torch.tensor([1., 0.]))
    missing, _ = lesson.cost(record, (torch.zeros_like(actual), torch.tensor([0])))
    assert missing.item() == 1.


def test_targets_cannot_change_the_prompt_need():
    need = torch.tensor([[.2, .3]])
    first = PartLesson(need, ((b'a',),))
    second = PartLesson(need, ((b'b',),))
    torch.testing.assert_close(first.need, second.need, rtol=0, atol=0)
