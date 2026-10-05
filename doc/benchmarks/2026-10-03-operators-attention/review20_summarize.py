"""Read saved §20 measurements and audits; no model, RNG or extra training."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
OUT = HERE/'review20-measurements'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def band(value):
    if value < .05:
        return 'at 0'
    if abs(value-.25) <= .02:
        return 'at 1/4'
    return 'between' if value < .25 else 'above 1/4'


def describe(values):
    return dict(n=len(values), min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), max=max(values)) if values else dict(n=0)


def summarize():
    complete = read(OUT/'complete.json')
    assert complete['completed'] and complete['source_matched'] and len(complete['jobs'])==30
    xor, mm, sums = [], [], []
    for run in range(1, 11):
        folder = OUT/f'xor-{run:02}'
        events = lines(folder/'observations.jsonl')
        grammar, = [event for event in events if event['kind']=='grammar']
        consumers = [event for event in events if event['kind']=='shared_gate_consumer']
        assert len(consumers)==2 and len({event['model_identity'] for event in consumers})==1
        calls = [row for row in lines(folder/'reports.jsonl') if row['phase']=='call']
        assert len(calls)==2
        y, target = grammar['predictions'], grammar['targets']
        mse = sum((a-b)**2 for a,b in zip(y,target))/len(y)
        correct = sum((a>.5)==(b>.5) for a,b in zip(y,target))
        recovered = sum(b is not None and Counter(a.split())==Counter(b.replace(chr(0),' ').split())
                        for a,b in zip(grammar['inputs'],grammar['gate_reconstructions']))
        class_pass = next(row['outcome']=='passed' for row in calls if 'test_xor_class_accuracy' in row['nodeid'])
        reconstruction_pass = next(row['outcome']=='passed' for row in calls if 'test_piecewise_overall' in row['nodeid'])
        assert class_pass == (correct==4 and mse<.05)
        assert reconstruction_pass == (recovered==4 and not any(grammar['grammar_reconstruction_unavailable']))
        derivations = grammar['final_greedy_compose']
        assert len(derivations) == len(y) == 4
        assert all(row['sequence'] for row in derivations)
        assert all(step['rule_name'] for row in derivations for step in row['sequence'])
        xor.append(dict(run=run,mse=mse,band=band(mse),correct=correct,recovered=recovered,
            class_pass=class_pass,reconstruction_pass=reconstruction_pass,joint=class_pass and reconstruction_pass,
            predictions=y,targets=target,inputs=grammar['inputs'],readbacks=grammar['gate_reconstructions'],
            readback_unavailable=grammar['grammar_reconstruction_unavailable'],
            final_greedy_compose=derivations,
            readback_decisions=grammar['readback_decisions'],
            run_audit=read(folder/'run-audit.json'),
            readback_counts=dict(Counter(row['decided_by'] for row in grammar['readback_decisions'])),
            inventories=grammar['inventories'],
            operator_names=[[step['rule_name'] for step in row['sequence']] for row in derivations],
            same_training=True,process=read(folder/'process.json')))
        folder=OUT/f'mm-{run:02}'
        observation, = [row for row in lines(folder/'observations.jsonl') if row['kind']=='mm']
        call, = [row for row in lines(folder/'reports.jsonl') if row['phase']=='call']
        assert (call['outcome']=='passed') == (observation['best']<.20)
        mm.append(dict(run=run,passed=call['outcome']=='passed',**observation,process=read(folder/'process.json')))
        folder=OUT/f'sum-{run:02}'
        observation=read(folder/'measurement.json')
        sums.append(dict(run=run,band=band(observation['mse']),**observation,process=read(folder/'process.json')))
    counts=dict(class_pass=sum(row['class_pass'] for row in xor),
        reconstruction_pass=sum(row['reconstruction_pass'] for row in xor),joint=sum(row['joint'] for row in xor),
        mm_pass=sum(row['passed'] for row in mm),sum_pass=sum(row['sum_bar'] for row in sums),
        xor_bands={name:sum(row['band']==name for row in xor)
                   for name in ('at 0','at 1/4','between','above 1/4')},
        sum_bands={name:sum(row['band']==name for row in sums)
                   for name in ('at 0','at 1/4','between','above 1/4')})
    below=[]
    for name,old in [('class_pass',0),('reconstruction_pass',10),('joint',0),('sum_pass',10),('mm_pass',10)]:
        if counts[name]<old:below.append(dict(count=name,prior=old,current=counts[name]))
    result=dict(counts=counts,comparison=dict(class_pass=0,reconstruction_pass=10,joint=0,sum_pass=10,
        mm_pass=10),below_comparison=below,
        xor=xor,mm=mm,sum=sums,complete=complete)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    audit()
    print(json.dumps(dict(counts=counts,below_comparison=below)))


def audit():
    folder=OUT/'xor-10/ownership'
    events=lines(folder/'events.jsonl')
    first={event['id']:event for event in events if event['kind']=='decoder_first_logits'}
    example = next(iter(first.values()))
    binary_catalogue = {index:dict(rule_id=rule_id, rule_name=name)
        for index,rule_id,name in zip(example['binary_indices'], example['binary_rule_ids'], example['binary_rule_names'], strict=True)}
    decoder_catalogue = {step['action_id']:dict(rule_id=step['rule_id'], rule_name=step['rule_name'])
        for event in events if event['kind']=='decoder' for row in event['decoder_derivations'] for step in row}
    steps=[event for event in events if event['kind']=='decoder_margin_step']
    comparisons=[event for event in events if event['kind']=='decoder_comparison']
    ownership=read(folder/'ownership.json')
    assert len(first)==1600 and len(steps)==1200
    assert len([step for step in steps if step['walks']])==800
    assert {walk['id'] for step in steps for walk in step['walks']} == set(first)
    by_epoch=defaultdict(lambda:defaultdict(list))
    by_binary=defaultdict(lambda:defaultdict(list))
    paths=[Counter() for _ in range(4)]
    greedy_paths=[Counter() for _ in range(4)]
    for row in comparisons:
        for i,(cost,win,departure,g,e) in enumerate(zip(row['costs'],row['wins'],row['departure'],row['greedy'],row['explore'])):
            assert win == (departure>=0 and cost[1]<cost[0])
            paths[i][tuple(a for a in (e if win else g) if a>=0)]+=1
            greedy_paths[i][tuple(a for a in g if a>=0)]+=1
    for step in steps:
        for walk in step['walks']:
            initial=first[walk['id']]
            for row,live in enumerate(initial['live']):
                if not live:continue
                for column,binary in enumerate(initial['binary_indices']):
                    values=dict(margin=initial['margin'][row][column],
                        stop_gradient=walk['gradient'][row][initial['stop_index']],
                        undo_gradient=walk['gradient'][row][binary],
                        gradient_difference=walk['stop_minus_undo_gradient'][row][column],
                        fixed_parent_change=walk['fixed_parent_margin_change'][row][column])
                    for key,value in values.items():
                        by_binary[binary][key].append(value)
                        by_epoch[(initial['epoch'],binary)][key].append(value)
    def stats(values):
        result={key:describe(value) for key,value in values.items()}
        gradient=values['gradient_difference'];delta=values['fixed_parent_change']
        result.update(nonzero_gradient=sum(value!=0 for value in gradient),
            nonzero_change=sum(value!=0 for value in delta),
            gradient_favours_undo=sum(value>0 for value in gradient),
            margin_reduced=sum(value<0 for value in delta),
            margin_increased=sum(value>0 for value in delta))
        return result
    def stability(groups):
        return [dict(modal_fraction=group.most_common(1)[0][1]/sum(group.values()),distinct=len(group),
            paths=[dict(actions=actions,count=count,
                derivation=[dict(action_id=action,**decoder_catalogue[action]) for action in actions])
                for actions,count in group.most_common()]) for group in groups]
    masks=Counter(('compound' if not entry['legal'][row][entry['stop_index']] and any(entry['legal'][row][:len(entry['binary_indices'])]) else 'singular' if entry['legal'][row][entry['stop_index']] else 'pending') for entry in first.values() for row,live in enumerate(entry['live']) if live)
    chooser=read(OUT/'xor-10/run-audit.json')
    score_rows=chooser['compose_score_function_steps']
    assert len(score_rows)==1600
    assert len(chooser['chooser_logit_ranges'])==400
    result=dict(decomposition_chooser=chooser['decomposition_chooser'],chooser=dict(nonzero_advantage_sentences=chooser['nonzero_advantage_sentences'],
        sentences=len(score_rows),departures=sum(r['departed'] for r in score_rows),
        per_epoch_ranges=chooser['chooser_logit_ranges'],
        maximum_gradient_error=max(r.get('gradient_max_error',0.) for r in score_rows),
        maximum_finite_difference_error=max(r.get('finite_difference',{}).get('error',0.) for r in score_rows)),
        first_step_eligibility=dict(masks),first_step_records=len(first),optimizer_steps=len(steps),
        steps_reaching_decoder=sum(bool(step['walks']) for step in steps),
        no_decoder_steps=sum(not step['walks'] for step in steps),
        ownership=ownership,walks=read(folder/'walk-audit.json')['counts'],
        binary={binary:dict(**binary_catalogue[binary], **stats(values)) for binary,values in by_binary.items()},
        epochs=[dict(epoch=epoch,binary=binary,**binary_catalogue[binary],**stats(values)) for (epoch,binary),values in sorted(by_epoch.items())],
        kept_paths=stability(paths),greedy_paths=stability(greedy_paths),
        root_geometry={phase:read(folder/f'geometry-{phase}.json')['roots'] for phase in ('start','end')},
        word_perceptual_support={phase:[word for book in read(folder/f'geometry-{phase}.json')['dictionary']
            for word in book['word_perceptual_support']] for phase in ('start','end')},
        bootstrap_learning='Deferred to the operators update by the user; context complements can remain zero without a learned whole seed.',
        activated_competitors=sum(row['with_activated_candidate'] for row in events if row['kind']=='activated_word_ranking'),
        activated_outranks_own=sum(row['activated_outranks_own'] for row in events if row['kind']=='activated_word_ranking'),
        readback_counts={scope:dict(Counter(decision['decided_by'] for row in events
            if row['kind']=='readback_decisions' and row['train']==train for decision in row['decisions']))
            for scope,train in [('training',True),('evaluation',False)]},
        note='STOP-minus-undo gradient is dL/dSTOP minus dL/dundo. Fixed-parent deltas observe the actual momentum/policy update, not a predicted independent-logit update. Decoder path gradients may be zero when the other path is selected.')
    (OUT/'audit-summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    summarize()
