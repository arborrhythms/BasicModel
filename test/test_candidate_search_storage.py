"""Native hard-search pullbacks against the preserved pre-repair function."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from test_meaning_kernels import paired


def previous():
    path=Path(__file__).parent/'fixtures/item27-pre-search-storage.py'
    spec=importlib.util.spec_from_file_location('item27_pre_search_storage',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module._bounded_binary_reconstruction


@pytest.mark.parametrize('operator',['ConjunctionLayer','DisjunctionLayer','SumLayer','ProductLayer','AdverbLayer'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_chosen_pair_preserves_values_eligibility_gradients_and_optimizer_step(paired,operator,dtype):
    import Language
    from DecompositionChooser import DecompositionChooser
    cls=getattr(Language,operator)
    op=cls() if operator in ('ConjunctionLayer','DisjunctionLayer') else cls(12,12)
    op=op.to(dtype=dtype)
    op.meaning_layout=(4,12)
    a=torch.tensor([.2,.4,.2,.4,.2,.3,.4,.2,.2,.3,.4,.2],dtype=dtype)
    b=torch.tensor([.4,.2,.4,.2,.6,.6,.6,.6,.6,.6,.6,.6],dtype=dtype)
    bank=torch.stack((a,b,a*.8,b*.7))[None].expand(3,-1,-1).clone()
    with torch.no_grad():
        parent=op.compose_blocks(op.compose,a,b,counts=(2,1))[None].expand(3,-1).clone()
    reference=torch.stack((a,b,a))
    known=torch.tensor([False,True,True]);right=torch.tensor([False,True,False])
    outcomes=[]
    for search in (previous(),Language.LanguageSpace._bounded_binary_reconstruction):
        operator_copy=copy.deepcopy(op);chooser=DecompositionChooser().to(dtype=dtype)
        root=parent.clone().requires_grad_()
        parameters=(root,*operator_copy.parameters(),*chooser.parameters())
        saved=[]
        def pack(tensor):saved.append(tensor.numel()*tensor.element_size());return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            result=search(operator_copy,root,reference,right,known,bank,
                torch.ones(3,4,dtype=torch.bool),4,chooser=chooser,return_details=True)
        left,right_value,valid,details=result
        loss=(left.square()+right_value*.3).sum()+details['logits'].square().sum()
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        trainable=list(operator_copy.parameters())+list(chooser.parameters())
        optimizer=torch.optim.SGD(trainable,lr=.01,momentum=.9)
        for parameter,gradient in zip(trainable,gradients[1:]):
            optimizer.state[parameter]['momentum_buffer']=torch.full_like(parameter,.1)
            parameter.grad=gradient
        optimizer.step()
        outcomes.append((result,gradients,[p.detach().clone() for p in trainable],sum(saved)))
    before,after=outcomes
    for left,right in zip(before[0][:3],after[0][:3]):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    for key in before[0][3]:
        torch.testing.assert_close(before[0][3][key],after[0][3][key],rtol=0,atol=0)
    tolerance=3e-6 if dtype==torch.float32 else 1e-12
    for expected,actual in zip(before[1],after[1]):
        assert (expected is None)==(actual is None)
        if actual is not None:torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    for expected,actual in zip(before[2],after[2]):
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    assert after[3]<before[3]


def test_native_selected_search_compiles_with_the_same_pullback(paired):
    from Language import LanguageSpace,ConjunctionLayer
    op=ConjunctionLayer();op.meaning_layout=(4,12)
    bank=torch.tensor([[[.2,.4,.2,.4,.2,.3,.4,.2,.2,.3,.4,.2],
                       [.4,.2,.4,.2,.6,.6,.6,.6,.6,.6,.6,.6]]],dtype=torch.float64)
    parent=op.compose_blocks(op.compose,bank[:,0],bank[:,1],counts=(2,1)).detach().requires_grad_()
    def search(root):
        flag=torch.zeros(1,dtype=torch.bool)
        return LanguageSpace._bounded_binary_reconstruction(op,root,torch.zeros_like(root),
            flag,flag,bank,torch.ones(1,2,dtype=torch.bool),2)
    eager=search(parent)
    compiled=torch.compile(search,backend='aot_eager',fullgraph=True)(parent)
    for a,b in zip(eager,compiled):torch.testing.assert_close(a,b,rtol=0,atol=0)
    first=torch.autograd.grad((eager[0]+eager[1]).sum(),parent)[0]
    second=torch.autograd.grad((compiled[0]+compiled[1]).sum(),parent)[0]
    torch.testing.assert_close(first,second,rtol=1e-12,atol=1e-12)
