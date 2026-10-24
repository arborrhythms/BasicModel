"""One first-batch timing reproduction, not a curriculum or learning gate."""
import json,sys,time,warnings,random
from pathlib import Path
sys.path[:0]=['bin','test']
import torch,numpy as np,pytest
from MeaningKernels import MeaningAction
from identity_experiment import make_model,restore_warmup
from constituent_budget import paired_updates_only
from attention_curriculum_gates import present,STAGES
from Models import BaseModel
restore=BaseModel.load_weights
out=Path('output/item3-5-verb-modal-20261023/development/38-stage3-profile');out.mkdir()
torch.set_num_threads(1);random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
with pytest.MonkeyPatch.context() as patch,warnings.catch_warnings():
 warnings.simplefilter('ignore')
 model=make_model(out,patch)
 restore_warmup(model,Path('output/item3-5-verb-modal-20261023/swap-verb/final/source-03/lessons/warmup/warmup.ckpt'),restore)
 optimizer=model.getOptimizer(lr=.003)
 stage=next(s for s in STAGES if s['stage']==3)
 with paired_updates_only(model,optimizer) as updates:
  start=time.perf_counter();initial=present(model,stage);evaluated=time.perf_counter()
  import cProfile,pstats
  profiler=cProfile.Profile();profiler.enable()
  row=present(model,stage,train=True,optimizer=optimizer)
  profiler.disable()
  profiler.dump_stats(str(out/'profile.dat'))
  with (out/'profile.txt').open('w') as stream:pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(60)
  print(json.dumps(dict(eval_seconds=evaluated-start,train_seconds=time.perf_counter()-evaluated,ce=row['lesson_cross_entropy'],updates=updates,diagnostic_only=True)),flush=True)
 model.End()
