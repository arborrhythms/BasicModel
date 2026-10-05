"""Summarize saved observations; no model construction or additional training."""
from collections import Counter
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
for folder in ('decoder-xor','checkpoint-xor'):
 p=HERE/folder;own=p/'ownership'
 observations=[json.loads(x) for x in (p/'observations.jsonl').read_text().splitlines()]
 gate=next(x for x in observations if x['kind']=='grammar')
 y=gate['predictions'];target=gate['targets'];mse=sum((a-b)**2 for a,b in zip(y,target))/len(y)
 events=[json.loads(x) for x in (own/'events.jsonl').read_text().splitlines()]
 comparisons=[x for x in events if x['kind']=='decoder_comparison']
 counters=[Counter() for _ in y];greedy=[Counter() for _ in y];counts=wins=violations=0
 for r in comparisons:
  for i,(c,w,d,g,e) in enumerate(zip(r['costs'],r['wins'],r['departure'],r['greedy'],r['explore'])):
   counts+=1;wins+=bool(w);violations+=int(bool(w)!=(c[1]<c[0] and d>=0))
   counters[i][tuple(a for a in (e if w else g) if a>=0)]+=1
   greedy[i][tuple(a for a in g if a>=0)]+=1
 stability=lambda groups:[{'modal_fraction':c.most_common(1)[0][1]/sum(c.values()),'distinct':len(c),'paths':[{'actions':a,'count':n} for a,n in c.most_common()]} for c in groups]
 geometries={}
 for time in ('start','end'):
  geometry=json.loads((own/f'geometry-{time}.json').read_text())
  geometries[time]=geometry
 result={'source':str((p/'source.json').relative_to(HERE)), 'trainings':1,'epochs':400,'seed':None,
  'class_mse':mse,'correct_classes':sum((a>.5)==(b>.5) for a,b in zip(y,target)),
  'class_bar':mse<.05 and all((a>.5)==(b>.5) for a,b in zip(y,target)),
  'reconstructed':sum(Counter(a.split())==Counter(b.split()) for a,b in zip(gate['inputs'],gate['gate_reconstructions'])),
  'gate':gate,'same_model_for_both_gates':len({r['model_identity'] for r in observations if r['kind']=='shared_gate_consumer'})==1,
  'decoder_comparisons':counts,'decoder_explore_wins':wins,'decoder_win_fraction':wins/counts,
  'decoder_selection_violations':violations,'kept_decoder_stability':stability(counters),'greedy_decoder_stability':stability(greedy),
  'compose_stability':json.loads((own/'derivation-stability.json').read_text()),
  'ownership':json.loads((own/'ownership.json').read_text()),'geometry':geometries,
  'process':json.loads((p/'process.json').read_text()),'source_check':json.loads((p/'complete.json').read_text())}
 (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(folder,{k:result[k] for k in ('class_mse','correct_classes','class_bar','reconstructed','decoder_comparisons','decoder_explore_wins','decoder_win_fraction','decoder_selection_violations','same_model_for_both_gates')})
