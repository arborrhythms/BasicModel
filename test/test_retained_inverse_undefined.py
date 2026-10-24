"""An unavailable parent coordinate is not passed to the next inverse map."""
from types import MethodType, SimpleNamespace
import pytest
import torch
from test_meaning_kernels import paired

@pytest.mark.parametrize('arity', (1,2))
@pytest.mark.parametrize('compiled', (False,True))
def test_retained_inverse_carries_undefined_content_without_evaluating_it(paired,arity,compiled):
    from Language import LanguageSpace, VerbLayer, SurfaceLayer, SurfaceUnaryLayer
    from DerivationReconstruction import inverse_step
    op=VerbLayer(12,12) if arity==2 else SurfaceUnaryLayer(SurfaceLayer(12,12))
    layer=SimpleNamespace(ops=[op],unary_ops=[op])
    owner=SimpleNamespace(_tree_layer=lambda _:layer)
    owner.reverse_unary_step=MethodType(LanguageSpace.reverse_unary_step,owner)
    parent=torch.tensor([[.2,.3,.4,.5,1.9,-.7,float('nan'),1.5,1.2,-.3,2.,1.8]])
    coordinates=torch.tensor([[True]*4+[False]*8])
    frame=torch.full((3,12),.25);frame[:,:4]=parent[0,:4]
    def run(value,defined):
        return inverse_step(owner,1 if arity==2 else 2,0,value,frame,coordinates=defined)
    run=torch.compile(run,backend='aot_eager',fullgraph=True) if compiled else run
    result=run(parent,coordinates)
    for value,defined in result:
        assert torch.isfinite(value).all()
        assert defined.tolist()==coordinates.tolist()
        torch.testing.assert_close(value[:,:4],parent[:,:4],rtol=0,atol=0)

@pytest.mark.parametrize('arity',(1,2))
def test_retained_inverse_still_rejects_out_of_cube_defined_content(paired,arity):
    from Language import LanguageSpace, VerbLayer, SurfaceLayer, SurfaceUnaryLayer
    from DerivationReconstruction import inverse_step
    op=VerbLayer(12,12) if arity==2 else SurfaceUnaryLayer(SurfaceLayer(12,12))
    layer=SimpleNamespace(ops=[op],unary_ops=[op])
    owner=SimpleNamespace(_tree_layer=lambda _:layer)
    owner.reverse_unary_step=MethodType(LanguageSpace.reverse_unary_step,owner)
    parent=torch.full((1,12),.25);parent[0,4]=1.9
    with pytest.raises(RuntimeError,match='finite coordinates in'):
        inverse_step(owner,1 if arity==2 else 2,0,parent,torch.full((3,12),.25))
