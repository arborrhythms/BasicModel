import torch
from configuration_fixtures import parallel_concepts
from recon_bench import _build_model
from Spaces import ConceptualSpace, _concept_alloc_of
original=ConceptualSpace._populate_concept_weights
def traced(self,cid,**kwargs):
    print('POPULATE',cid,kwargs,'symbolic',self._symbolic_order,'sourceorder',self._concept_source_order(cid),'caps',self._order_caps(),flush=True)
    result=original(self,cid,**kwargs)
    st=_concept_alloc_of(self).layer()
    print('AFTER',st.features.nnz,st._tensor_rows, 'singletons',_concept_alloc_of(self).singletons,flush=True)
    return result
ConceptualSpace._populate_concept_weights=traced
with parallel_concepts(category=True) as path:
    model,*_=_build_model(path)
cs=model.conceptualSpaces[0]
model.runEpoch(optimizer=model.getOptimizer(lr=.01),batchSize=4,split='train',max_batches=1)
print('FEATURES',_concept_alloc_of(cs).layer().features._index)
assert _concept_alloc_of(cs).layer().features.nnz>0
