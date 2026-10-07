"""The shared operation choice preserves STM metadata exactly."""
import pytest
import torch
from Spaces import ConceptualSpace, LanguageOperationChoice


def state():
    return (torch.tensor([[[2.], [1.], [3.], [0.]]]), torch.tensor([3]),
            torch.tensor([[2, 1, 4, -1]]), torch.tensor([[0, 1, 2, -1]]),
            torch.tensor([[22, 11, 33, -1]]), torch.tensor([[.2, .1, .3, 0.]]))


def choice(kind, position=0):
    return LanguageOperationChoice(torch.tensor([[7.]]), torch.tensor([kind]),
        torch.tensor([position]), torch.tensor([0]), torch.tensor([kind != 0]),
        torch.tensor([.5]), torch.tensor([0]), torch.tensor([True]))


@pytest.mark.parametrize('position', [0, 1])
def test_unary_invalidates_only_its_operand_reference(position):
    before = state()
    after = ConceptualSpace.apply_language_choice(before, choice(2, position))
    assert after[1].tolist() == [3]
    assert after[2].tolist() == before[2].tolist()
    assert after[3][0, position] == before[3][0, position] + 1
    assert after[4][0, position] == -1
    assert after[5][0, position] == 0
    other = 1 - position
    assert after[4][0, other] == before[4][0, other]
    assert before[4][0, position] != -1


def test_stop_preserves_all_slots_and_references():
    before = state()
    before[0][0, 0] = .7
    after = ConceptualSpace.apply_language_choice(before, choice(0))
    for a, b in zip(after, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_binary_shifts_surviving_reference_and_updates_orders():
    before = state()
    after = ConceptualSpace.apply_language_choice(before, choice(1))
    assert after[1].tolist() == [2]
    assert after[2].tolist() == [[2, 4, -1, -1]]
    assert after[3].tolist() == [[2, 2, -1, -1]]
    assert after[4].tolist() == [[-1, 33, -1, -1]]
    torch.testing.assert_close(after[5], torch.tensor([[0., .3, 0., 0.]]))
