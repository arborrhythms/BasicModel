"""Completed native-width forests, counted folds, absorb and pair transpose."""
from types import SimpleNamespace
import pytest
import torch


def test_completed_native_forest_has_exact_leaves_and_live_source_pullback(monkeypatch):
    import util
    from data import TheData
    from Models import BaseModel
    from Language import SurfaceLayer, SurfaceUnaryLayer, TransposeLayer
    from DerivationReconstruction import reconstruct, certified_map
    monkeypatch.setattr(util,'TheCompileBackend','none')
    TheData.processLM({s:dict(text=['.'] if s=='train' else [],label=[])
                      for s in ('train','validation','test')})
    model=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0]
    try:
        D=model._generate_walk_width()
        for dtype in (torch.float32,torch.float64):
            surface=SurfaceLayer(D,D).to(dtype)
            absorb=SurfaceUnaryLayer(surface).to(dtype)
            transpose=TransposeLayer(D,D)
            assert certified_map(surface) and certified_map(absorb)
            width=surface.meaning_layout[0]
            assert width>=model._readback_percept_width and D>width
            percept_width=model._readback_percept_width
            with torch.no_grad():
                surface.meaning_kernel.log_scale.fill_(.03)
                surface.meaning_kernel.bias.fill_(.01)
                surface.meaning_kernel.context.diagonal().fill_(.02)
            for postposed in (False,True):
                leaves=torch.linspace(.15,.65,3*D,dtype=dtype).reshape(3,D).requires_grad_()
                actions=[[0,-1,0],[0,-1,1],[1,0,-1],[0,-1,2]]
                pair=surface.compose_blocks(surface.compose,leaves[:1],leaves[1:2],counts=(1,1))
                a,b=pair,leaves[2:3]
                counts=(2,1)
                if postposed:
                    a,b=transpose.compose(a,b);counts=tuple(reversed(counts));actions.append([3,-1,-1])
                parent=surface.compose_blocks(surface.compose,a,b,counts=counts)
                fold=len(actions);actions.append([1,0,-1])
                unary=len(actions);actions.append([2,0,-1])
                root=absorb.compose(parent)
                frames=[leaves.new_zeros(3,D) for _ in actions]
                frames[2]=torch.cat((leaves[:1],leaves[1:2],pair))
                frames[fold]=torch.cat((a,b,parent))
                frames[unary]=torch.cat((parent,parent,root))
                numbers=leaves.new_ones(len(actions),2);numbers[fold]=numbers.new_tensor(counts)
                program=SimpleNamespace(actions=torch.tensor(actions),operation_values=torch.stack(frames),
                    operand_counts=numbers,form_wholes=None)
                layer=SimpleNamespace(ops=(surface,),unary_ops=(absorb,))
                language=SimpleNamespace(_tree_layer=lambda _:layer,
                    _compose_binary_rules=(SimpleNamespace(method_name='surface',relation_kind=None),),
                    _compose_unary_rules=(SimpleNamespace(method_name='surface',relation_kind=None),),
                    reverse_unary_step=lambda value,*args,**kwargs:(absorb.reverse(value),torch.zeros(len(value),dtype=torch.bool)))
                record=SimpleNamespace(word_values=leaves[None],word_valid=torch.ones(1,3,dtype=torch.bool),
                    end_depth=torch.ones(1,dtype=torch.long),end_slots=root[:,None],primed=None)
                recovered,unavailable,defined=reconstruct(language,record,(program,),lambda _:list(range(3)))
                atol=2e-6 if dtype==torch.float32 else 1e-12
                torch.testing.assert_close(recovered[0],leaves,atol=atol,rtol=0)
                assert defined.all() and not unavailable.any() and language._last_derivation_certified.all()
                direction=torch.linspace(-.7,.9,3*D,dtype=dtype).reshape(3,D)
                gradient,=torch.autograd.grad((recovered[0]*direction).sum(),leaves,retain_graph=True)
                torch.testing.assert_close(gradient,direction,atol=atol*4,rtol=0)
                # §42's residual chart retains signed corrections and the
                # same numerical loss, with the teacher and prior detached.
                from Expectation import percept_reconstruction
                prior=torch.full_like(leaves[None,:,:width],.8,requires_grad=True)
                correction,target,rebuilt,error=percept_reconstruction(recovered[...,:width],leaves[None,:,:width],prior)
                assert (correction<0).any()
                torch.testing.assert_close(error,(rebuilt-leaves[None,:,:width].detach()).square(),atol=1e-12,rtol=1e-5)
                assert torch.autograd.grad(error.sum(),prior,allow_unused=True)[0] is None
    finally:
        model.End();model.symbolSpace.soft_reset()


def test_unsupported_inverse_keeps_its_learning_gradient():
    from DerivationReconstruction import certified_map
    from Language import ProductLayer
    assert not certified_map(ProductLayer(8,8))
    from AttentionObjective import field_cost
    observed=torch.tensor([[[.7,.4]]],requires_grad=True)
    recovered=torch.tensor([[[.3,.2]]],requires_grad=True)
    expected=torch.tensor([[[.4,.5]]],requires_grad=True)
    cost=field_cost(observed,recovered,torch.ones(1,1),torch.ones(1,1,dtype=torch.bool),floor=.1,expected=expected)
    cost['inside'].sum().backward()
    assert recovered.grad.abs().sum()>0 and observed.grad is None and expected.grad is None
