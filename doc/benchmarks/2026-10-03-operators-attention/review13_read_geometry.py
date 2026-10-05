"""Read the audited run's saved tensors after a count falls below comparison.

No model construction, training, RNG draw, extra inference or changed gate.
"""
import itertools
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE/'review13-measurements'
AUDIT=OUT/'xor-10/ownership'
summary=json.loads((OUT/'summary.json').read_text())
assert summary['below_comparison'], 'additional reading requires a below-comparison count'
result=dict(scope='Saved observations of XOR run 10 only; no new model or training', phases={})
for phase in ('start','end'):
    geometry=json.loads((AUDIT/f'geometry-{phase}.json').read_text())
    words={}
    for book in geometry['dictionary']:
        if not book['word_perceptual_support']:
            continue
        data=torch.load(AUDIT/f"geometry-{phase}-stage-{book['stage']}.pt",map_location='cpu',weights_only=True)
        index={row:i for i,row in enumerate(data['rows'])}
        for word in book['word_perceptual_support']:
            words[word['word']]=data['codes'][index[word['row']],:word['dimension']].double()
    pairs=[]
    for a,b in itertools.combinations(sorted(words),2):
        x,y=words[a],words[b]
        residual=y-x*((x@y)/(x@x))
        pairs.append(dict(words=[a,b],cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)),
            distance=float((x-y).norm()),
            proportional_residual=float(residual.norm()),
            relative_proportional_residual=float(residual.norm()/y.norm())))
    roots=torch.tensor(geometry['roots']['values'],dtype=torch.float64)
    result['phases'][phase]=dict(word_perceptual_coordinates={w:c.tolist() for w,c in words.items()},
        word_pairs=pairs,root_norms=roots.norm(dim=-1).tolist(),
        centered_root_singular_values=geometry['roots']['centered_singular_values'])
events=[json.loads(x) for x in (AUDIT/'events.jsonl').read_text().splitlines()]
first=[r for r in events if r['kind']=='decoder_first_logits']
legal_counts={}
for row in first:
    for live,mask in zip(row['live'],row['legal']):
        if live:
            n=sum(mask);legal_counts[n]=legal_counts.get(n,0)+1
result['first_step_legal_action_counts']=legal_counts
result['stop_eligible']=sum(bool(mask[row['stop_index']]) for row in first for live,mask in zip(row['live'],row['legal']) if live)
result['decoder_parameter_updates']={}
for event in events:
    if event['kind']!='optimizer_displacement':continue
    for row in event['parameters']:
        if 'generate_policy' not in row['parameter']:continue
        item=result['decoder_parameter_updates'].setdefault(row['parameter'],dict(observations=0,nonzero=0,max_norm=0.))
        value=row.get('displacement_norm',row.get('displacement',0.))
        item['observations']+=1;item['nonzero']+=int(value!=0);item['max_norm']=max(item['max_norm'],value)
with (OUT/'saved-geometry-reading.json').open('x') as handle:
    json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps(result))
