"""A legal unary preference must still fit the row by its closing."""
import torch

from test_subspace_what_stm_contract import _xor_model


def test_shared_stack_contract_allows_unary_until_the_deadline(_xor_model, monkeypatch):
    from test_subspace_what_stm_contract import test_shared_operation_writes_the_caller_owned_stack

    chooser = _xor_model.languageSpace._tree_layer(2).chooser
    score = chooser.score_unary

    def prefer_unary(*args, **kwargs):
        stop, unary = score(*args, **kwargs)
        return stop, torch.full_like(unary, 1e6)

    monkeypatch.setattr(chooser, 'score_unary', prefer_unary)
    test_shared_operation_writes_the_caller_owned_stack(_xor_model)
