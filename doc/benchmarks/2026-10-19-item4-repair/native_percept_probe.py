#!/usr/bin/env python3
"""Native identify-before-attenuate read across the full percept-gain interval."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch,pytest,util
from test_compiled_word_chunk import _tiny_canonical_model,_stage_fullgraph_tensor_peer

def run():
 torch.set_num_threads(1)
 rows=[]
 with tempfile.TemporaryDirectory() as td,pytest.MonkeyPatch.context() as patch:
  patch.setattr(util,'TheCompileBackend','none')
  model=_tiny_canonical_model(Path(td),patch,input_width=32,word_buckets='8',stm_capacity=3,chooser_depth=1)
  model.eval();model.set_sentence_expectation(False)
  with torch.no_grad():
   _stage_fullgraph_tensor_peer(model,['cat cat','dog dog'])
   fi=model.perceptualSpace._forward_input
   moments=model.percept_expectation
   moments.observe(fi['seed_event'][...,:len(moments.mean)],fi['indices']>=0)
   baseline=None
   for g in (0.,.01,.1,.25,.5,.75,.9,.99,1.):
    model.percept_expectation_gain=g
    _stage_fullgraph_tensor_peer(model,['cat cat','dog dog'])
    values=model.inputSpace._ar_word_object_atoms.detach().clone()
    ids=model.inputSpace._ar_word_concept_ids.detach().clone()
    if baseline is None:baseline=values,ids
    same=bool(torch.equal(values,baseline[0]) and torch.equal(ids,baseline[1]))
    assert same
    rows.append(dict(gain=g,identity_unchanged=same,region_gain=model._percept_word_gain.tolist(),ids=ids.tolist(),moment_count=int(moments.count)))
  model.End()
 return rows
if __name__=='__main__':
 result=run();(Path(__file__).parent/'native_percept_probe.json').write_text(json.dumps(result,indent=2)+'\n')
 print([(r['gain'],r['identity_unchanged']) for r in result])
