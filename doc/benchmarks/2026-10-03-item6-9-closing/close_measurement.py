"""Summarize the sole shared training from saved arrays, without another model."""
from collections import Counter
import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
P=R/'xor-ownership'
observations=[json.loads(x) for x in (R/'xor/observations.jsonl').read_text().splitlines()]
gate=next(x for x in observations if x['kind']=='grammar')
y=torch.tensor(gate['predictions']);target=torch.tensor(gate['targets'])
error=float((y-target).square().mean())
exact=sum(Counter(a.split()) == Counter((b or '').split()) for a,b in zip(gate['inputs'],gate['gate_reconstructions']))
events=[json.loads(x) for x in (P/'events.jsonl').read_text().splitlines()]
decoder=[x for x in events if x['kind']=='decoder']
training=[x for x in decoder if x['train']]
summary=dict(trainings=1,epochs=400,class_correct=int(((y>.5)==(target>.5)).sum()),
    error=error,error_band='at 0' if error<.05 else 'at 1/4' if abs(error-.25)<=.02 else 'between' if error<.23 else 'above 1/4',
    class_bar=bool(((y>.5)==(target>.5)).all() and error<.05),reconstruction_bar=exact==4,
    joint_bar=bool(((y>.5)==(target>.5)).all() and error<.05 and exact==4),
    exact_readbacks=exact,contrast=float(y[0]+y[3]-y[1]-y[2]),
    answers=gate['predictions'],targets=gate['targets'],inputs=gate['inputs'],
    readbacks=gate['gate_reconstructions'],available=[not x for x in gate['grammar_reconstruction_unavailable']],
    perceptual_reporting_view=gate['reconstructions'],
    shared_model=len({x['model_identity'] for x in observations if x['kind']=='shared_gate_consumer'})==1,
    decoder_calls=len(training),decoder_row_visits=sum(len(x['rows']) for x in training),
    actions=Counter(op for x in training for ops in x['chosen_operations'] for op in ops),
    emitted_counts=Counter(len(leaves) for x in training for leaves in x['leaves']),pairs={})
for name in ('world/there','hello/loving'):
    values=[x['pair_cosines'][name] for x in decoder if name in x['pair_cosines']]
    summary['pairs'][name]=dict(start=values[0],end=values[-1],minimum=min(values),maximum=max(values))
a=torch.load(P/'geometry-start-stage-1.pt',weights_only=True)['codes']
b=torch.load(P/'geometry-end-stage-1.pt',weights_only=True)['codes']
summary['dictionary_frobenius_displacement']=float((b-a).norm())
(R/'measurement.json').write_text(json.dumps(summary,indent=2)+'\n')
trajectory=[{k:x[k] for k in ('epoch','batch','trial','train','sid','rows','surfaces','priming','pair_cosines','truncated','chosen_operations','compose_rules','compose_arities','leaves','leaf_code_cosines')} for x in decoder]
(R/'decoder-trajectory.json').write_text(json.dumps(trajectory,indent=2)+'\n')
print(json.dumps(summary,indent=2))
