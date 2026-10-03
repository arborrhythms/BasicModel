import sys,os,tempfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false')
import torch,util
util.TheCompileBackend='none'
from test_reverse_traversal import _traversal_model,_run
with tempfile.TemporaryDirectory() as directory:
 m=_traversal_model(Path(directory)); seen=[]
 original=m._reconstruct_trial
 def capture(record):
  result=original(record)
  seen.append((record,result))
  return result
 m._reconstruct_trial=capture
 with m._sentence_run():_run(m,['12 plus 1','3 plus 4'])
 summary={'public_cost':m._recon_cost.tolist(),'transaction_costs':[v[2].tolist() for v in m._sentence_reconstructions], 'records':[]}
 for original_record,result in seen:
  record=m._sentence_understandings[int(original_record.sentence)]
  diffs={name:float((getattr(original_record,name).float()-getattr(record,name).float()).abs().max()) for name in record.__dataclass_fields__ if torch.is_tensor(getattr(record,name)) and getattr(record,name).numel()}
  with torch.no_grad():
   raw=m._reconstruct_sentences(original_record.root,original_record.word_values,original_record.roots,original_record.depths,original_record.end_slots,original_record.end_depth,original_record.sentence,understanding=original_record)
   retained=m._reconstruct_sentences(record.root,record.word_values,record.roots,record.depths,record.end_slots,record.end_depth,record.sentence,understanding=record)
  summary['records'].append(dict(original=result[2].tolist(),replay_original=raw[2].tolist(),replay_retained=retained[2].tolist(),field_changes=diffs))
 print(json.dumps(summary))
 Path(__file__).with_suffix('.json').write_text(json.dumps(summary,indent=2))
 m.End();m.symbolSpace.soft_reset();torch._dynamo.reset()
assert summary['public_cost']==summary['transaction_costs'][0], 'explicit result replayed a discarded journal instead of publishing the scored sentence'
