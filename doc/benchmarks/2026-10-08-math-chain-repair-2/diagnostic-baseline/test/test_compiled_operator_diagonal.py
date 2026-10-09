"""Conditional free-inverse backward retains exact LDU diagonal gradients."""
import torch
from Layers import InvertibleLinearLayer

def test_ldu_diagonal_cond_backward_is_dense_and_exact():
    layer=InvertibleLinearLayer(4,4)
    def forward(selected):
        return torch.cond(selected,lambda:layer._D_embed(),lambda:torch.zeros((4,4)),())
    compiled=torch.compile(forward,backend='aot_eager',fullgraph=True)
    weights=torch.arange(16,dtype=torch.float32).reshape(4,4)
    (compiled(torch.tensor(True))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,weights.diagonal())
    layer.d.grad=None
    (compiled(torch.tensor(False))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,torch.zeros(4))


def test_functional_ldu_cond_backward_is_dense_and_exact():
    layer=InvertibleLinearLayer(4,4,naive=True)
    x=torch.arange(1,13,dtype=torch.float32).reshape(3,4)/17
    def forward(selected):
        return torch.cond(selected,lambda:layer.functional_forward(x),lambda:torch.zeros_like(x),())
    compiled=torch.compile(forward,backend='aot_eager',fullgraph=True)
    weights=torch.arange(12,dtype=torch.float32).reshape(3,4)
    (compiled(torch.tensor(True))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,(x*weights).sum(0))
    layer.d.grad=None
    (compiled(torch.tensor(False))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,torch.zeros(4))


def test_free_candidate_search_compiles_backward_through_native_operators(tmp_path):
    from test_reverse_traversal import _traversal_model
    model=_traversal_model(tmp_path)
    try:
        language=model.languageSpace
        binary=language._tree_layer(2)
        index=list(binary.op_names).index('lift')
        op=getattr(binary.ops[index],'gl',binary.ops[index])
        width=model.conceptualSpace.stm.concept_dim
        codes=(torch.arange(1,1+6*width,dtype=torch.float32).reshape(2,3,width)/(7*width)).requires_grad_()
        parent=op.compose(codes[:,0],codes[:,1]).detach().requires_grad_()
        selected=torch.full((2,),index,dtype=torch.long)
        valid=torch.ones(2,dtype=torch.bool)
        bank_valid=torch.ones(2,3,dtype=torch.bool)
        priming=torch.ones(2,3)
        def forward(root,bank):
            left,right=language.reverse_binary_step(root,selected,valid,basis=bank,
                basis_valid=bank_valid,basis_priming=priming)
            return left+right
        parameters=(parent,codes,*tuple(language.parameters()))
        expected=forward(parent,codes)
        expected_grad=torch.autograd.grad(expected.square().sum(),parameters,allow_unused=True)
        compiled=torch.compile(forward,backend='aot_eager',fullgraph=True)
        actual=compiled(parent,codes)
        actual_grad=torch.autograd.grad(actual.square().sum(),parameters,allow_unused=True)
        torch.testing.assert_close(actual,expected)
        for parameter,a,b in zip(parameters,actual_grad,expected_grad):
            if a is None and b is None:continue
            if a is None:a=torch.zeros_like(parameter)
            if b is None:b=torch.zeros_like(parameter)
            torch.testing.assert_close(a,b)
    finally:
        torch._dynamo.reset()
        model.End()
