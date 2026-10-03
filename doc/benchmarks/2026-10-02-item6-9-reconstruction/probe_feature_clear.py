import sys, traceback
from configuration_fixtures import parallel_concepts
from recon_bench import _build_model
from Spaces import _concept_alloc_of
with parallel_concepts(category=True) as path:
    model,*_=_build_model(path)
cs=model.conceptualSpaces[0]
store=_concept_alloc_of(cs).layer()
last=0
prev=None
def trace(frame,event,arg):
    global last,prev
    if event=='line' and '/basicmodel/bin/' in frame.f_code.co_filename:
        n=store.features.nnz
        if last>0 and n==0:
            print('CLEARED after',prev,flush=True)
            traceback.print_stack(frame)
        last=n
        prev=(frame.f_code.co_filename,frame.f_code.co_name,frame.f_lineno,n)
    return trace
sys.settrace(trace)
model.runEpoch(optimizer=model.getOptimizer(lr=.01),batchSize=4,split='train',max_batches=1)
sys.settrace(None)
assert store.features.nnz>0
