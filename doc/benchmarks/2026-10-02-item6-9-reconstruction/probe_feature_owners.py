import json, torch
from configuration_fixtures import parallel_concepts
from recon_bench import _build_model
from Spaces import _concept_alloc_of
with parallel_concepts(category=True) as path:
    model,*_=_build_model(path)
model.runEpoch(optimizer=model.getOptimizer(lr=.01),batchSize=4,split='train',max_batches=1)
for i,cs in enumerate(model.conceptualSpaces):
    print('SPACE',i,'owner',cs is model._concept_owner(),'features',_concept_alloc_of(cs).layer().features.nnz,
          'a0',getattr(cs,'_cs_last_a0',None),'roles',getattr(cs,'_category_n_roles',None),'priming',cs.priming_weights())
