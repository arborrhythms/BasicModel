"""The catalogue's product, mean, min and max each have three live faces."""
import pytest
import torch
from Language import ConjunctionLayer, DisjunctionLayer, MinLayer, MaxLayer


@pytest.mark.parametrize('layer,kernel', [
    (ConjunctionLayer, lambda a,b: a.norm(dim=-1,keepdim=True)*b.norm(dim=-1,keepdim=True)*torch.nn.functional.normalize(a*b, dim=-1)),
    (DisjunctionLayer, lambda a,b: (a+b)/2),
    (MinLayer, torch.minimum), (MaxLayer, torch.maximum)])
def test_three_faces_bind_and_search_the_same_pair(layer, kernel):
    codes = torch.tensor([[.2, -.7, .4], [.8, .3, -.5], [-.6, .2, .9]], requires_grad=True)
    op = layer()
    left, right = codes[0:1], codes[1:2]
    parent = kernel(left, right)
    torch.testing.assert_close(op(left, right), parent)
    torch.testing.assert_close(op.compose(left, right), parent)
    for face in (op.reverse, op.generate):
        a,b = face(parent, basis=codes, left_rows=torch.tensor([0]), right_rows=torch.tensor([1]))
        torch.testing.assert_close(a,left)
        torch.testing.assert_close(b,right)
        torch.testing.assert_close(op.compose(a,b), parent)
    op.compose(left,right).sum().backward()
    assert codes.grad[0].norm() > 0 and codes.grad[1].norm() > 0


def test_conjunction_same_reference_is_identity_but_equal_codes_are_distinct():
    x = torch.tensor([[.2, -.7, .4]])
    op = ConjunctionLayer()
    torch.testing.assert_close(op.compose(x,x), x)
    different_reference = x.clone()
    expected = x.norm(dim=-1,keepdim=True).square()*torch.nn.functional.normalize(x.square(), dim=-1)
    torch.testing.assert_close(op.compose(x,different_reference), expected)
    assert not torch.allclose(expected,x)


def test_conjunction_zero_operand_has_zero_result_and_finite_gradient():
    x = torch.tensor([[.2,-.7,.4]],requires_grad=True)
    y = torch.zeros_like(x,requires_grad=True)
    result=ConjunctionLayer().compose(x,y)
    assert result.count_nonzero() == 0
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_free_lift_inverse_searches_both_operands_from_the_bank(tmp_path):
    from test_reverse_traversal import _traversal_model
    model=_traversal_model(tmp_path)
    try:
        language=model.languageSpace
        binary=language._tree_layer(2)
        index=list(binary.op_names).index('lift')
        op=getattr(binary.ops[index],'gl',binary.ops[index])
        dimension=model.conceptualSpace.stm.concept_dim
        codes=torch.arange(1,1+3*dimension,dtype=torch.float32).reshape(1,3,dimension)/100
        parent=op.compose(codes[:,0],codes[:,1])
        left,right,bad=language.reverse_binary_step(parent,torch.tensor([index]),torch.tensor([True]),
            basis=codes,basis_valid=torch.ones(1,3,dtype=torch.bool),basis_priming=torch.ones(1,3),return_status=True)
        assert not bad.any()
        assert any(torch.allclose(left,codes[:,i],atol=1e-6,rtol=0) for i in range(3))
        assert any(torch.allclose(right,codes[:,i],atol=1e-6,rtol=0) for i in range(3))
        torch.testing.assert_close(op.compose(left,right),parent)
    finally:
        model.End()


def test_free_discarding_operator_still_searches_both_bank_operands():
    from Language import LanguageSpace
    class Part:
        rule_name='part'
        def compose(self,left,right):return right
    basis=torch.tensor([[[.2,.5],[.9,.3]]])
    parent=torch.tensor([[.8,.4]])
    a,b,ready=LanguageSpace._bounded_binary_reconstruction(Part(),parent,torch.zeros_like(parent),
        torch.tensor([False]),torch.tensor([False]),basis,torch.ones(1,2,dtype=torch.bool),2)
    assert ready.all()
    for value in (a,b):
        assert any(torch.equal(value,basis[:,row]) for row in range(2))


def test_conjunction_keeps_the_repeated_reference_for_the_next_operation():
    from types import SimpleNamespace
    from ClauseScope import ClauseScope
    scope=ClauseScope([SimpleNamespace(method_name='conjunction')],[])
    state=torch.tensor([[[0,7],[0,7],[0,7]]])
    choice=SimpleNamespace(kind=torch.tensor([1]),local_op=torch.tensor([0]),position=torch.tensor([0]),applied=torch.tensor([True]))
    once,closing=scope.apply(state,choice,torch.tensor(0))
    assert once[0,0,1] == 7 and closing.item() == 0
    twice,_=scope.apply(once,choice,torch.tensor(1))
    assert twice[0,0,1] == 7
