#!/usr/bin/env python3
"""Compare native zero-expectation consumer views to Item 5's landing."""
import argparse,sys,tempfile,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--restore',action='store_true');args=parser.parse_args()
sys.path[:0]=[str(args.root/'bin'),str(args.root/'test')]
import torch,pytest,util
from test_compiled_word_chunk import _tiny_canonical_model,_stage_fullgraph_tensor_peer

def plain(t):return t.detach().cpu().tolist() if torch.is_tensor(t) else t
with tempfile.TemporaryDirectory() as td,pytest.MonkeyPatch.context() as patch:
 torch.set_num_threads(1);torch.manual_seed(23)
 patch.setattr(util,'TheCompileBackend','none')
 model=_tiny_canonical_model(Path(td),patch,input_width=16,word_buckets='8',stm_capacity=3,chooser_depth=1,
  training_overrides={'intraLossWeight':0.})
 model.eval();model._tensor_peer_while_eager=True
 model.set_sentence_expectation(False)
 for name in ('percept_expectation_gain','word_expectation_gain','expectation_gain','document_expectation_gain'):setattr(model,name,0.)
 state={n:v.detach().cpu().clone() for n,v in model.named_parameters()}
 matched=[];unmatched=[]
 if args.restore:
  previous=torch.load(args.state,weights_only=True)
  with torch.no_grad():
   for name,value in model.named_parameters():
    if name in previous and previous[name].shape==value.shape:
     value.copy_(previous[name]);matched.append(name)
    else:unmatched.append(name)
 else:torch.save(state,args.state)
 views=[]
 with torch.no_grad():
  for samples in (['cat','dog'],['cat dog','dog cat']):
   raw=_stage_fullgraph_tensor_peer(model,samples)
   staged=dict(ids=plain(model.inputSpace._ar_word_concept_ids),keys=plain(model.inputSpace._ar_word_object_atoms),words=plain(model.inputSpace._ar_embedded_N))
   with model._sentence_run():result=model._forward_with_compiled_sentence_state(raw)
   public=model._publish_compiled_sentence_state(result);model._end_step()
   staged.update(end_slots=plain(model._tensor_final_end_slots),depth=plain(model._tensor_final_end_depth),
     field_roles=[None if f is None else plain(f.meaning.roles) for f in model._sentence_fields[0]])
   views.append(staged);model.End();model.symbolSpace.soft_reset()
 args.output.write_text(json.dumps(dict(root=str(args.root),matched=matched,unmatched=unmatched,views=views),indent=2)+'\n')
