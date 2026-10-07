"""Postprocess saved campaigns only: no model, RNG, or optimizer invocation."""
from collections import Counter,defaultdict
import json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
O=H/'measurements'
def read(p):return json.loads(p.read_text())
def stats(xs):return dict(count=len(xs),minimum=min(xs,default=0),maximum=max(xs,default=0),mean=sum(xs)/max(1,len(xs)))
def logit_epoch(a, epoch):
    return {(k if ':' in k else 'compose:'+k):v for k,v in a['chooser_logit_ranges'].items()
            if k == str(epoch) or k.endswith(':'+str(epoch))}
results={}
for kind in ('sum','xor'):
    combined=Counter();runs=[];buckets=defaultdict(Counter)
    for run in range(1,11):
        a=read(O/f'{kind}-{run:02}'/'run-audit.json')
        steps={(x['epoch'],x['batch_row']):x for x in a['compose_score_function_steps']}
        action_names={}
        for step in steps.values():
            if step.get('walk')=='compose':
                assert step['action_name']==action_names.setdefault(step['action'],step['action_name'])
        if kind=='xor':
            last_trial=[t for t in a['sentence_trials'] if t['training']][-1]
            for row in a['final_committed_training_operators']:
                b=row['batch_row'];path=last_trial['actions'][int(last_trial['wins'][b])][b]
                # With two words and no compose unary, the first singleton's
                # only action is STOP. Match the remaining placement action
                # to the held final derivation, including policies whose
                # greedy operator was never sampled as a departure.
                applied=[v for v in path if v>=0 and v!=path[0]]
                assert len(applied)==len(row['sequence'])==1
                name=row['sequence'][0]['rule_name']
                assert name==action_names.setdefault(applied[0],name)
        policy_epochs=[]
        counter=Counter();examples=[];narrow_nonzero=[];costs=[]
        run_buckets=defaultdict(Counter);late_buckets=defaultdict(Counter)
        largest=[];late_examples=[];proposal_counts=Counter()
        for t in a['sentence_trials']:
            if not t['training']:continue
            if kind=='xor':
                # Learn flattened placement-action names from this run's observed
                # departures. These gates have one binary placement; do not
                # mistake placement actions for global grammar rule IDs.
                greedy=[[action_names[v] for v in path if v in action_names] for path in t['actions'][0]]
                explore=[[action_names[v] for v in path if v in action_names] for path in t['actions'][1]]
                assert all(len(x)==1 for x in greedy+explore)
                policy_epochs.append(dict(epoch=t['epoch'],greedy=greedy,
                    committed=[explore[i] if win else greedy[i] for i,win in enumerate(t['wins'])]))
            for row,advantage in enumerate(t['advantage']):
                s=steps[t['epoch'],row];walk=s['walk'];name=s['action_name']
                delta=t['delta'][row];key=str(walk)+'/'+str(name);bucket=buckets[key]
                assert s['sampling_scale']==s['W']*s['R_walk']*s['K']
                proposal_counts[(walk,s['W'],s['R_walk'],s['K'])]+=1
                counters=[counter,bucket,run_buckets[key]]
                if t['epoch']>380:counters.append(late_buckets[key])
                for c in counters:
                    c['departures']+=int(s['departed']);c['nonzero']+=int(advantage!=0)
                    c['reward_explore']+=int(advantage<0);c['reward_greedy']+=int(advantage>0)
                    c['keep_explore']+=int(t['wins'][row]);c['keep_tie']+=int(delta[0]==0)
                    c['answer_against_keep']+=int(t['answer_against_keep'][row]);c['policy_against_keep']+=int(t['policy_against_keep'][row])
                    c['answer_rewards_explore']+=int(delta[2]<0);c['answer_rewards_greedy']+=int(delta[2]>0)
                    c['R_nonzero']+=int(delta[0]!=0);c['E_nonzero']+=int(delta[1]!=0);c['A_nonzero']+=int(delta[2]!=0)
                    for i,label in enumerate(('R','E','A')):c['delta_'+label+'_sum']+=delta[i]
                    c['advantage_sum']+=advantage
                event=dict(epoch=t['epoch'],row=row,walk=walk,action=name,components=t['components'][row],
                    delta=delta,advantage=advantage,keep=t['keep_decision'][row],deciding=t['deciding'][row],
                    W=s['W'],R_walk=s['R_walk'],K=s['K'],sampling_scale=s['sampling_scale'])
                if advantage!=0 and len(examples)<12:examples.append(event)
                if advantage!=0 and walk=='narrowing':narrow_nonzero.append(event)
                if advantage!=0:
                    largest.append(event);largest.sort(key=lambda e:abs(e['advantage']),reverse=True);del largest[12:]
                    if t['epoch']>380:late_examples.append(event)
                costs.append(advantage)
        combined.update(counter)
        reader=a['reader_training_rows'];row_weights=Counter(tuple(w) for r in reader for w in r['weights'])
        evals=[t for t in a['sentence_trials'] if not t['training']]
        last_disjunction=max((p['epoch'] for p in policy_epochs if any('disjunction' in row for row in p['greedy'])),default=0)
        policy_transitions=[p for i,p in enumerate(policy_epochs) if i==0 or p['greedy']!=policy_epochs[i-1]['greedy']]
        runs.append(dict(run=run,counts=counter,advantages=stats(costs),reader_steps=len(reader),reader_row_weights={str(k):v for k,v in row_weights.items()},
            reader_optimizer_steps=a['reader_weights'][-1]['optimizer_steps'],reader_updates_by_epoch=dict(Counter(r['reader_updates'] for r in a['reader_weights'])),
            final_operators=a['final_committed_training_operators'],evaluation_components=evals[-1]['components'] if evals else None,
            by_walk_action=run_buckets,last_twenty_epochs_by_walk_action=late_buckets,
            proposal_counts=[dict(walk=k[0],W=k[1],R_walk=k[2],K=k[3],count=v) for k,v in proposal_counts.items()],
            examples=examples,largest_advantages=largest,last_twenty_epochs_nonzero=late_examples,narrowing_nonzero=narrow_nonzero,
            observed_compose_action_names=action_names,policy_epochs=policy_epochs,policy_transitions=policy_transitions,
            last_greedy_disjunction_epoch=last_disjunction,
            nonzero_by_walk_action=a['nonzero_advantage_by_walk_action'],pole_consumers=a['pole_consumers'],
            containment=a['containment'],logit_first=logit_epoch(a,1),
            logit_last=logit_epoch(a,400),closing_image=a['closing_image']))
    results[kind]=dict(counts=combined,by_walk_action=buckets,runs=runs)
results['postprocessor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(H/'aggregate-audit.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:v['counts'] for k,v in results.items() if k in ('sum','xor')}))
