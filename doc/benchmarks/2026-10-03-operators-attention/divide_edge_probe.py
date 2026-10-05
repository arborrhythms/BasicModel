"""A contradictory first word cannot turn divide into a repeated no-op."""
import torch


def test_divide_isolates_impure_first_word_and_reaches_pure_neighbors():
    from Attention import narrow_words
    from Language import OperationSelectionLayer
    result = narrow_words(OperationSelectionLayer(d_model=3), torch.eye(3)[None],
        torch.tensor([[[0, 2], [3, 5], [6, 8]]]), torch.ones(1, 3, dtype=torch.bool),
        poles=torch.tensor([[[1., 1.], [1., 0.], [1., 0.]]]), budget=16)
    assert result.accepted.tolist() == [[False, True, True]]
    assert not result.descended.any()
    assert result.table.spent.item() <= 16
