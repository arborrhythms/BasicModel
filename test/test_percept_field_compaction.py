"""Members read at a sentence's occurrences: padding is no occurrence, and no
read crosses features with an axis of events beside one of readings."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from PerceptField import read_occurrences
from Spaces import PartSpace
from PerceptProperties import PrimitiveProperties


def test_padding_reads_nothing_and_teaches_no_property():
    primitive=PrimitiveProperties(1)
    with torch.no_grad():primitive.members[0,65]=.3;primitive.members[0,66]=.7
    raw=torch.tensor([[65,66],[65,66]])
    ids=torch.tensor([[7,8],[7,8]])
    spans=torch.tensor([[[0,1],[1,2]]]).expand(2,-1,-1)
    whole=torch.tensor([[[0,2]]]).expand(2,-1,-1)
    occurrences=torch.cat((spans,whole,torch.zeros(2,1,2,dtype=torch.long)),1)
    keys=(('ps',(7,)),('ws',0))
    def read(padded):
        parts=torch.cat((spans,torch.zeros(2,13,2,dtype=torch.long)),1) if padded else spans
        rows=torch.cat((ids,torch.full((2,13),-1)),1) if padded else ids
        wholes=torch.cat((whole,torch.zeros(2,7,2,dtype=torch.long)),1) if padded else whole
        return read_occurrences((rows,parts,primitive,raw,wholes),occurrences,keys,
                                part_reader=PartSpace.part_memberships,dtype=torch.float32)
    short=read(False);compact=read(True)
    torch.testing.assert_close(short,compact,rtol=0,atol=0)
    assert compact.shape==(2,2,4,2)
    # The run (7,) is held by its own event and by the whole; nothing by
    # the padded occurrence.
    assert compact[0,:,0,0].eq(1).all() and compact[0,:,2,0].eq(1).all() and not compact[:,:,3].any()
    torch.testing.assert_close(compact[1,:,0],torch.tensor([[.3,.7],[.3,.7]]))
    # The order-zero read reads the memberships and takes their gradient:
    # declared classes are not fixed.
    assert compact.requires_grad


def test_captured_b28_w256_read_has_no_axis_of_events_beside_readings(tmp_path):
    """The captured allocation was [150,28,256,2058,2], or 17.7 GB."""
    B,W,P,K=28,256,2058,150
    class Bounded(TorchDispatchMode):
        def __init__(self):super().__init__();self.shapes=set()
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            from torch.utils._pytree import tree_leaves
            self.shapes.update(tuple(x.shape) for x in tree_leaves(result) if torch.is_tensor(x))
            return result
    spans=torch.zeros(B,P-1,2,dtype=torch.long)
    spans[:,0]=torch.tensor([0,1]);spans[:,1]=torch.tensor([1,2])
    ids=torch.full((B,P-1),-1,dtype=torch.long);ids[:,0]=7;ids[:,1]=8
    raw=torch.full((B,2),65,dtype=torch.long)
    occurrences=torch.zeros(B,W,2,dtype=torch.long);occurrences[:,:3]=torch.tensor([[0,1],[1,2],[0,2]])
    observer=Bounded()
    with observer:
        members=read_occurrences((ids,spans,None,raw,spans[:,:0]),occurrences,
            [('ps',(i,)) for i in range(K)],part_reader=PartSpace.part_memberships,dtype=torch.float32)
    assert members.shape==(K,B,W,2)
    assert members[7,:,0,0].eq(1).all() and members[8,:,1,0].eq(1).all() and members[7,:,2,0].eq(1).all()
    assert not any(len(shape)==5 and shape[:2]==(K,B) for shape in observer.shapes)
    assert not any(len(shape)>=4 and shape[-2:]==(W,W) for shape in observer.shapes)


def test_interior_gap_does_not_become_an_ordered_literal_match():
    ids=torch.tensor([[7,-1,8,-1]])
    spans=torch.tensor([[[0,1],[0,0],[1,2],[0,0]]])
    raw=torch.tensor([[65,66]])
    occurrences=torch.tensor([[[0,1],[1,2],[0,2]]])
    members=read_occurrences((ids,spans,None,raw,spans[:,:0]),occurrences,[('ps',(7,8))],
                             part_reader=PartSpace.part_memberships,dtype=torch.float32)
    assert not members.any()
