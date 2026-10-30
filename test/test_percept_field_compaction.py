"""Native event positions survive removal of all-stream padding columns."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from PerceptField import read_percepts
from Spaces import PartSpace
from PerceptProperties import PrimitiveProperties


def test_compact_native_events_preserve_values_locations_and_property_gradients():
    primitive=PrimitiveProperties(1)
    with torch.no_grad():primitive.members[0,65]=.3;primitive.members[0,66]=.7
    raw=torch.tensor([[65,66],[65,66]])
    ids=torch.tensor([[7,8],[7,8]])
    spans=torch.tensor([[[0,1],[1,2]]]).expand(2,-1,-1)
    whole=torch.tensor([[[0,2]]]).expand(2,-1,-1)
    brackets=torch.tensor([[[0,1],[0,2],[0,0]]]).expand(2,-1,-1)
    keys=(('ps',(7,)),('ws',0))
    def read(padded):
        parts=torch.cat((spans,torch.zeros(2,13,2,dtype=torch.long)),1) if padded else spans
        rows=torch.cat((ids,torch.full((2,13),-1)),1) if padded else ids
        wholes=torch.cat((whole,torch.zeros(2,7,2,dtype=torch.long)),1) if padded else whole
        return read_percepts((rows,parts,primitive,raw,wholes),brackets,keys,
                             part_reader=PartSpace.part_memberships,dtype=torch.float32)
    short=read(False);compact=read(True)
    for name in ('values','events','spans','where','when'):
        torch.testing.assert_close(getattr(short,name),getattr(compact,name),rtol=0,atol=0)
    assert compact.events.shape==(2,2,3,3,2)
    torch.testing.assert_close(compact.spans,torch.cat((spans,whole),1),rtol=0,atol=0)
    assert compact.values[0,:,0,0].eq(1).all() and not compact.values[:,:,2].any()
    torch.testing.assert_close(compact.events[1,:,1,-1],torch.tensor([[.3,.3],[.3,.3]]))
    a,=torch.autograd.grad(short.values[1].sum(),primitive.members,retain_graph=True)
    b,=torch.autograd.grad(compact.values[1].sum(),primitive.members)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert a.abs().sum()>0
    selected=compact.select(1,1)
    assert selected.events.shape==(2,1,1,3,2)
    torch.testing.assert_close(selected.event_evidence(keys),compact.events[:,1:2,1:2])


def test_captured_b28_w256_field_does_not_cross_features_with_padded_event_capacity(tmp_path):
    """The captured allocation was [150,28,256,2058,2], or 17.7 GB."""
    B,W,P,K=28,256,2058,150
    class Bounded(TorchDispatchMode):
        def __init__(self):super().__init__();self.shapes=set()
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if args and isinstance(args[0],(list,tuple)) and all(type(x) is int for x in args[0]):
                assert tuple(args[0])!=(K,B,W,P,2),'dense padded event allocation'
            result=func(*args,**(kwargs or {}))
            from torch.utils._pytree import tree_leaves
            self.shapes.update(tuple(x.shape) for x in tree_leaves(result) if torch.is_tensor(x))
            return result
    spans=torch.zeros(B,P-1,2,dtype=torch.long)
    spans[:,0]=torch.tensor([0,1]);spans[:,1]=torch.tensor([1,2])
    ids=torch.full((B,P-1),-1,dtype=torch.long);ids[:,0]=7;ids[:,1]=8
    raw=torch.full((B,2),65,dtype=torch.long)
    brackets=torch.zeros(B,W,2,dtype=torch.long);brackets[:,:2]=torch.tensor([[0,1],[1,2]])
    whole=torch.tensor([[[0,2]]]).expand(B,-1,-1)
    observer=Bounded()
    with observer:
        field=read_percepts((ids,spans,None,raw,whole),brackets,
            [('ps',(i,)) for i in range(K)],part_reader=PartSpace.part_memberships,dtype=torch.float32)
    assert field.events.shape==(K,B,W,3,2) and field.values.shape==(K,B,W,2)
    assert field.values[7,:,0,0].eq(1).all() and field.values[8,:,1,0].eq(1).all()
    assert not any(len(shape)==5 and shape[:3]==(K,B,W) and shape[3]>3 for shape in observer.shapes)
    import json,os
    from pathlib import Path
    target=Path(os.environ.get('ITEM27_RECEIPT',str(tmp_path)));target.mkdir(parents=True,exist_ok=True)
    (target/'percept-field-allocation.json').write_text(json.dumps(dict(
        captured_shape=[K,B,W,P,2],actual_shape=list(field.events.shape),
        no_padded_event_product=True,shapes=sorted([list(x) for x in observer.shapes],key=lambda x:(len(x),x))),indent=2)+'\n')


def test_interior_gap_does_not_become_an_ordered_literal_match():
    ids=torch.tensor([[7,-1,8,-1]])
    spans=torch.tensor([[[0,1],[0,0],[1,2],[0,0]]])
    raw=torch.tensor([[65,66]])
    brackets=torch.tensor([[[0,2]]])
    field=read_percepts((ids,spans,None,raw,spans[:,:0]),brackets,[('ps',(7,8))],
                        part_reader=PartSpace.part_memberships,dtype=torch.float32)
    assert field.events.shape[-2]==3 and not field.values.any()
    torch.testing.assert_close(field.spans,spans[:,:3])
