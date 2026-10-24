"""An unpaired configuration never uses the meaning-block phase inverse."""
from types import MethodType, SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('compiled', (False, True))
def test_unpaired_absent_selection_reconstructs_its_identity(compiled):
    from Language import LanguageSpace, CompoundLayer, CompoundUnaryLayer
    from DerivationReconstruction import inverse_step
    op = CompoundUnaryLayer(CompoundLayer(2, 2))
    assert op.meaning_layout is None
    layer = SimpleNamespace(unary_ops=[op])
    owner = SimpleNamespace(_tree_layer=lambda arity: layer, _generate_unary_ops=[op])
    owner.reverse_unary_step = MethodType(LanguageSpace.reverse_unary_step, owner)
    owner.generate_unary_step = MethodType(LanguageSpace.generate_unary_step, owner)
    value = torch.tensor([[.2, .7]], requires_grad=True)
    def run(point):
        parent = op.compose(point)
        ((recovered, defined),) = inverse_step(owner, 2, 0, parent, point[None])
        generated, unavailable = owner.generate_unary_step(parent, torch.tensor([0]), torch.tensor([True]), return_status=True)
        return recovered, defined, generated, unavailable
    actual, defined, generated, unavailable = (torch.compile(run, backend='aot_eager', fullgraph=True) if compiled else run)(value)
    assert defined.all()
    torch.testing.assert_close(actual, value, rtol=0, atol=0)
    torch.testing.assert_close(op.reverse(actual), value, rtol=0, atol=0)
    torch.testing.assert_close(generated, value, rtol=0, atol=0)
    assert not unavailable.any()
    actual.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)


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
