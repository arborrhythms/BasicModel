"""Position-indexed phase views must survive lexical alternatives."""
import pytest
import torch

@pytest.mark.parametrize('arity', (1,2))
@pytest.mark.parametrize('compiled', (False,True))
def test_phase_candidates_keep_positions_separate_from_reading_alternatives(monkeypatch, arity, compiled):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer, CompoundUnaryLayer, OperationSelectionLayer, _BinaryGrammarOpAdapter, _UnaryGrammarOpAdapter
    from Queries import StructuralGrammarContext, ConceptualSpaceCapability
    from CaseSelection import CaseView
    base=CompoundLayer().double()
    unary=CompoundUnaryLayer(base).double()
    binary_op,unary_op=_BinaryGrammarOpAdapter(base),_UnaryGrammarOpAdapter(unary)
    layer=OperationSelectionLayer(d_model=12,ops=[binary_op],unary_ops=[unary_op]).double()
    with torch.no_grad(): base.meaning_kernel.bias.fill_(.03)
    batch,positions,alternatives=2,3,3
    values=torch.linspace(.1,.8,batch*positions*alternatives*12,dtype=torch.float64).reshape(batch,positions,alternatives,12).requires_grad_()
    codes=torch.linspace(.15,.75,batch*4*12,dtype=torch.float64).reshape(batch,4,12).requires_grad_()
    count=positions if arity==1 else positions-1
    weights=torch.zeros(batch,count,4,2,dtype=torch.float64)
    for b in range(batch):
        for p in range(count): weights[b,p,(b+p)%4]=1.
    reference=values[:, :count,0]
    view=CaseView(codes,weights,torch.ones_like(reference),torch.zeros(batch,count,dtype=torch.bool),reference)
    context=StructuralGrammarContext((),ConceptualSpaceCapability(12),None,'compose',case_view=view if arity==2 else None,case_unary_view=view if arity==1 else None)
    slab=values[:,:,0]
    def compose(v):
        if arity==1:
            return layer._stacked_applied(v[:,:,0],context,dict(unary_ops=(0,0,0),unary=v))
        return layer._stacked_reduced(v[:,:,0],context,dict(binary_ops=(0,0,0),left=v[:,:-1],right=v[:,1:]))
    if arity==1:
        expected=torch.stack([unary_op.forward_with_grammar_context(values[:,:,i],context=context) for i in range(alternatives)],2)
    else:
        expected=torch.stack([binary_op.forward_with_grammar_context(values[:,:-1,i],values[:,1:,i],slab,context=context) for i in range(alternatives)],2)
    run=torch.compile(compose,backend='aot_eager',fullgraph=True) if compiled else compose
    actual=run(values)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    inputs=(values,codes,base.meaning_kernel.bias)
    expected_grad=torch.autograd.grad(expected.square().sum(),inputs,retain_graph=True)
    actual_grad=torch.autograd.grad(actual.square().sum(),inputs)
    for observed,wanted in zip(actual_grad,expected_grad):
        torch.testing.assert_close(observed,wanted,rtol=1e-12,atol=1e-12)
    assert actual_grad[1].abs().sum()>0
