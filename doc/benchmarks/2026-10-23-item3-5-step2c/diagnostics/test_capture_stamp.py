"""A capture preserves the actual provisional identity reading stamp."""
from types import SimpleNamespace
import torch
from Models import BasicModel


def test_program_keeps_the_identity_leaf_stamp_before_reference_resolution():
    # One staged frame, two words, and a three-column operation journal.
    origins = torch.zeros(1, 6, 3, dtype=torch.long)
    origins[0, 1] = torch.tensor([4, 0, 1])
    trace = SimpleNamespace(_choice_values=torch.zeros(1, 3, 12),
        _identity_state=(torch.zeros(1, 6, 4), torch.zeros(1, 6, dtype=torch.bool),
            torch.zeros(1, 2, 6, dtype=torch.long), origins,
            torch.zeros(1, 3, 6, dtype=torch.long)))
    observed = []
    def resolve(owner, rows, ids, actions, *, leaf_orders, admit):
        observed.append(leaf_orders.clone())
        return ids.clone(), torch.full_like(ids, -1)
    model = SimpleNamespace(_reconstruction_stack=lambda: trace,
        inputSpace=SimpleNamespace(),
        symbolSpace=SimpleNamespace(_word_reference_orders=torch.tensor([[0, 2]])),
        languageSpace=SimpleNamespace(resolve_lexical_references=resolve),
        _concept_owner=lambda: object())
    program = (torch.tensor([[0, 1]]),
        torch.tensor([[[0, -1, 0], [0, -1, 1], [1, 0, -1]]]),
        torch.tensor([[0, 0, 0]]), torch.tensor([[0, 1, 2]]))
    entry, = BasicModel._program_entries(model, program, torch.zeros(1, 2, 4),
        torch.tensor([[10, 11]]), torch.tensor([[10, 11]]), torch.ones(1, 2),
        torch.zeros(1, 3, 4), concept_ids=torch.tensor([[101, 102]]))
    assert entry.leaf_orders.tolist() == [1, 2]
    assert observed[0].tolist() == [1, 2]
    assert entry.identity_leaves.tolist() == [2, 3]
    assert entry.concept_ids.tolist() == [101, 102]
    assert model.symbolSpace._word_reference_orders.tolist() == [[0, 2]]
