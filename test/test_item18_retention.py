"""A recomputation oracle for the write-maintained forgetting columns."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from Memory import LTM0
from Forgetting import retention_values,ForgettingConfig


def recomputed(store,config,alive=None):
    n=len(store);alive=set(range(n)) if alive is None else alive
    points=F.normalize(store.slots[:n],dim=-1)
    p,q=store.truth_plus[:n],store.truth_minus[:n]
    trust=torch.maximum(p,q)
    signs=(p.gt(0)&q.eq(0)).long()-(q.gt(0)&p.eq(0)).long()
    scopes=[repr((store._semantic_rows[int(key)].get('bindings',()),store._semantic_rows[int(key)].get('scope',()))) for key in store.address_keys[:n]]
    values=torch.ones(n)
    for i in range(n):
        values[i]=float(not any(scopes[i]==scopes[j] and store.order_of_row(i)==store.order_of_row(j)
            and signs[i]*signs[j]==-1 and trust[j]>trust[i]
            and points[i]@points[j]>config.similarity for j in alive))
    if config.coverage and alive:
        ids=sorted(alive);codes=store.slots[ids].abs().clamp(0,1)
        positive,negative=codes*p[ids,None],codes*q[ids,None]
        def cover(a,b):
            if not len(a):return torch.tensor(0.)
            a,b=a.amax(0),b.amax(0)
            return (a-b-torch.minimum(a,b)).mean()
        full=cover(positive,negative)
        for i,row in enumerate(ids):
            keep=torch.arange(len(ids))!=i
            values[row]=(full-cover(positive[keep],negative[keep])).clamp(0,1)
    return values


def test_maintained_retention_matches_recompute_after_witness_scope_and_delete():
    store=LTM0(4,capacity=24)
    for i in range(16):
        point=torch.rand(4)
        store.append_idea(point,evidence=(.4 if i%2 else 0.,.8 if not i%2 else 0.),
            document_key=('row',i),time=False)
    def verify():
        for coverage in (False,True):
            config=ForgettingConfig(coverage=coverage,similarity=.4)
            for alive in (None,set(range(len(store)-2))):
                expected=recomputed(store,config,alive)
                from torch.utils._python_dispatch import TorchDispatchMode
                class NoProducts(TorchDispatchMode):
                    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                        assert str(func) not in ('aten.mm.default','aten.bmm.default','aten.mv.default')
                        return func(*args,**(kwargs or {}))
                with NoProducts():
                    actual=retention_values(store,config,alive=alive)['luminosity']
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                assert actual.argsort(stable=True).tolist()==expected.argsort(stable=True).tolist()
    verify()
    store.set_evidence(0,0.,.99);store.set_trust(1,.91)
    store.set_scope(2,(('document','new'),));store.witness_stream(3,0)
    verify()
    store.compact_rows([i for i in range(16) if i not in (0,3,7)])
    verify()
