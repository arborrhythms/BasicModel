"""Read saved item 6.2 measurements and audits; no model, RNG or extra training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE/'measurements'


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
            storage=grammar['storage'],storage_after_training=grammar['storage_after_training'],
            class_pass=class_pass,reconstruction_pass=reconstruction_pass,joint=class_pass and reconstruction_pass,
            predictions=y,targets=target,inputs=grammar['inputs'],readbacks=grammar['gate_reconstructions'],
            readback_unavailable=grammar['grammar_reconstruction_unavailable'],
            final_greedy_compose=derivations,
            readback_decisions=grammar['readback_decisions'],
            run_audit=str(folder.relative_to(HERE)/'run-audit.json'),
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
        sum_floor_pass=sum(row['floor_bar'] for row in sums),
        xor_bands={name:sum(row['band']==name for row in xor)
                   for name in ('at 0','at 1/4','between','above 1/4')},
        sum_bands={name:sum(row['band']==name for row in sums)
                   for name in ('at 0','at 1/4','between','above 1/4')})
    below=[]
    for name,old in [('class_pass',10),('reconstruction_pass',10),('joint',10),('sum_pass',10),('mm_pass',10)]:
        if counts[name]<old:below.append(dict(count=name,prior=old,current=counts[name]))
    result=dict(counts=counts,comparison=dict(class_pass=10,reconstruction_pass=10,joint=10,sum_pass=10,
        mm_pass=10),below_comparison=below,
        xor=xor,mm=mm,sum=sums,complete=complete,
        pre_training_launcher_failures=[],thinking=[dict(run=folder.name,**read(folder/'thinking.json')) for folder in sorted(OUT.iterdir()) if folder.is_dir() and (folder/'thinking.json').exists()])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    audit()
    integrity_and_costs()
    print(json.dumps(dict(counts=counts,below_comparison=below)))


def integrity_and_costs():
    root = HERE.parents[2]
    sys.path.insert(0, str(root/'test'))
    import bounded_tests as bounded
    from verification import validate
    source = bounded.source_snapshot(root)
    sweep = validate(source)
    helpers = read(HERE/'measured-source/measurement-helpers.json')
    helper_matches = all(hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
                         for name, sha in helpers.items())
    protected = ('test/test_explicit_dimensions.py', 'test/test_mm_xor.py',
                 'data/XOR_grammar.xml', 'data/MM_xor.xml')
    protected_matches = {name: (root/name).read_bytes() == subprocess.check_output(
        ['git', 'show', '631d9e8e44b7c8034263e22b36dc74fa5df4eb75:'+name], cwd=root)
        for name in protected}
    rows = []
    for kind in ('xor', 'sum'):
        for run in range(1, 11):
            folder = OUT/f'{kind}-{run:02}'
            audit = read(folder/'run-audit.json')
            costs = [trial for event in audit['sentence_trials']
                     for pair in event['components'] for trial in pair]
            rows.append(dict(kind=kind, run=run, observed_trials=len(costs),
                maximum_abs_reconstruction=max(abs(row[0]) for row in costs),
                maximum_abs_expectation=max(abs(row[1]) for row in costs),
                nonzero_reconstruction=sum(row[0] != 0 for row in costs),
                nonzero_expectation=sum(row[1] != 0 for row in costs),
                unseeded_entry_saved=(folder/'unseeded-entry.pt').is_file()))
    value = dict(source_matched=True, helpers_matched=helper_matches,
        protected_gate_files_unchanged=protected_matches,
        complete_sweep=dict(selected=len(sweep['selected']), completed=len(sweep['completed']),
                            exit_code=sweep['exit_code'], reason=sweep['reason']),
        sentence_costs=rows,
        mm_scope='The unchanged raw-forward MM gate has no paired sentence owner-step cost; its trajectories are in summary.json.',
        seed=None, retries=0, replacements=0)
    (OUT/'integrity-and-costs.json').write_text(json.dumps(value,indent=2)+'\n')
    assert helper_matches and all(protected_matches.values())


def audit():
    folder=OUT/'xor-10/ownership'
    saved=read(OUT/'xor-10/run-audit.json')
    rows=saved['compose_score_function_steps']
    events=lines(folder/'events.jsonl')
    result=dict(ownership=read(folder/'ownership.json'),
        chooser=dict(records=len(rows),nonzero_advantage_sentences=saved['nonzero_advantage_sentences'],
            by_walk_action=saved['nonzero_advantage_by_walk_action'],
            per_epoch_ranges=saved['chooser_logit_ranges'],
            maximum_gradient_error=max(r.get('gradient_max_error',0.) for r in rows),
            maximum_finite_difference_error=max(r.get('finite_difference',{}).get('error',0.) for r in rows)),
        steps_reaching_decoder=sum(bool(row['walks']) for row in events if row['kind']=='decoder_margin_step'),
        closing_image=saved['closing_image'],
        sentence_trials=len(saved['sentence_trials']),
        deciding_components=dict(Counter(name for trial in saved['sentence_trials'] if trial['training']
                                         for name in trial['deciding'])))
    (OUT/'audit-summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    summarize()
