import torch
from configuration_fixtures import parallel_concepts
from recon_bench import _build_model
from Spaces import _concept_alloc_of
with parallel_concepts(category=True) as path:
    model,*_=_build_model(path)
model.runEpoch(optimizer=None,batchSize=4,split='train',max_batches=1)
owner=model._concept_owner()
book=owner.similarity_codebook
before=book.W.detach().clone()
owner._refresh_feature_codes()
change=float((before-book.W.detach()).abs().max())
print('feature_refresh_without_optimizer',change,flush=True)
print('reading_owner',list(model.conceptualSpaces).index(owner),'features',_concept_alloc_of(owner).layer().features.nnz,
      'field_populated',getattr(owner,'_cs_last_a0',None) is not None,flush=True)
assert change==0, 'reconstruction alone may write codes'
assert getattr(owner,'_cs_last_a0',None) is not None, 'the field must read its definitions owner'
