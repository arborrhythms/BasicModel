"""An unpaired configuration never uses the meaning-block phase inverse."""
from types import MethodType, SimpleNamespace
import pytest
import torch

def test_unpaired_binary_selection_uses_the_existing_case_search():
    from Language import LanguageSpace, CompoundLayer
    from DerivationReconstruction import inverse_step
    from test_compound_cases import _bank
    bank = _bank()._replace(eligible=torch.tensor([[True, False, False, True, False]]))
    owner = SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(ops=[CompoundLayer(2, 2)]))
    owner._reverse_of_binary_op = MethodType(LanguageSpace._reverse_of_binary_op, owner)
    owner.reverse_binary_step = MethodType(LanguageSpace.reverse_binary_step, owner)
    frame = torch.stack((bank.codes[3], bank.codes[0], torch.tensor([1., 0.])))
    children = inverse_step(owner, 1, 0, frame[2:3], frame, case_bank=bank)
    for (value, defined), expected in zip(children, (bank.codes[3:4], bank.codes[:1]), strict=True):
        assert defined.all()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
