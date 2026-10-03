import sys, importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/'bin'),str(root/'test')]
sys.argv=['objective_conflicts_probe.py','--config','XOR_grammar','--output',str(Path(__file__).parent/'repairs/observer-one-batch-audit-ranking'),'--validate-only']
spec=importlib.util.spec_from_file_location('observer',root/'test/objective_conflicts_probe.py')
observer=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(observer)
except SystemExit:pass
from test_mm_xor import _fresh_model
model,_,data=_fresh_model(str(root/'data/XOR_grammar.xml'))
optimizer=model.getOptimizer(lr=.01)
raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',
    batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
assert observer.P.training_batches == 1
assert {g['trial'] for g in observer.P.gradients if g['scope']=='trial'} == {'exploit','explore'}
assert all(g['rng_unchanged'] for g in observer.P.gradients)
