"""Verify saved measurement provenance and reporting completeness; never train."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'review13-measurements'
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import source_snapshot

def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]

source=read(HERE/'review13-source/source.json')
assert source==source_snapshot(ROOT)==read(OUT/'source.json')
reporting=read(HERE/'review13-source/reporting-source.json')
for name in ('review13_campaign.py','review13_gate_observer.py'):
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==reporting[name]
assert hashlib.sha256((ROOT/'doc/benchmarks/2026-10-01-item6-9-review/separator_campaign.py').read_bytes()).hexdigest()==reporting['separator_campaign.py']
complete=read(OUT/'complete.json')
assert complete['completed'] and complete['source_matched'] and complete['retries']==0
assert len(complete['jobs'])==30
assert {(r['kind'],r['run']) for r in complete['jobs']}=={(kind,run) for kind in ('sum','xor','mm') for run in range(1,11)}
assert all(row['process']['reason']=='exit' and row['process']['exit_code'] in (0,1) for row in complete['jobs'])
assert all(row['process']['peak_memory_bytes']<=8*1024**3 for row in complete['jobs'])
assert all(row['process']['elapsed_seconds']<1800 for row in complete['jobs'])
controls=read(OUT/'sum-read-first.json')
assert controls['passed']==controls['total']==10 and controls['before_xor_or_mm']
assert len(controls['completed_processes'])==10
assert {r['kind'] for r in controls['completed_processes']}=={'sum'}
for run in range(1,11):
    folder=OUT/f'xor-{run:02}'
    observations=lines(folder/'observations.jsonl')
    grammar,=[r for r in observations if r['kind']=='grammar']
    consumers=[r for r in observations if r['kind']=='shared_gate_consumer']
    assert len(consumers)==2 and len({r['model_identity'] for r in consumers})==1
    final=grammar['final_greedy_compose']
    assert [r['batch_row'] for r in final]==[0,1,2,3]
    assert all(r['sequence'] for r in final)
    decisions=grammar['readback_decisions']
    assert isinstance(decisions,list)
    assert all(row['decided_by'] in ('code','priming','tie') and 0<=row['batch_row']<4
               and row['position']>=0 for row in decisions)
    assert all(book['percept_width']==6 and book['percept_event_width']==14
               and book['code_width']==14 and book['context_width']==0
               for book in grammar['inventories'])
    for row in final:
        for step in row['sequence']:
            assert step['rule_name']=={0:'not',1:'conjunction',2:'disjunction'}[step['rule_id']]
    reports=lines(folder/'reports.jsonl')
    assert len([r for r in reports if r['phase']=='call'])==2
    assert all(r['outcome']=='passed' for r in reports if r['phase']!='call')
    control=read(OUT/f'sum-{run:02}/measurement.json')
    assert control['sum_bar'] and len(control['final_greedy_compose'])==4
    assert all(step['rule_name']=='sum' for row in control['final_greedy_compose'] for step in row['sequence'])
    mm=lines(OUT/f'mm-{run:02}/observations.jsonl')
    assert len([r for r in mm if r['kind']=='mm'])==1

folder=OUT/'xor-10/ownership'
assert read(folder/'complete.json')['source_matched']
events=lines(folder/'events.jsonl')
named=0
for event in events:
    if event['kind']=='decoder':
        assert event['compose_derivations'] and event['decoder_derivations']
        for row in event['compose_derivations']:
            for step in row['sequence']:
                assert step['rule_name'] and isinstance(step['rule_id'],int)
                named+=1
        for row in event['decoder_derivations']:
            for step in row:
                assert step['rule_name'] and ('rule_id' in step)
    elif event['kind']=='decoder_comparison':
        assert event['greedy_derivations'] and event['explore_derivations']
    elif event['kind']=='decoder_first_logits':
        assert event['binary_rule_ids']==[1,2]
        assert event['binary_rule_names']==['conjunction','disjunction']
ownership=read(folder/'ownership.json')
assert ownership['conflicts']==0
assert all(not row['writers'] or row['writers']==[row['owner']] for row in ownership['parameters'])
assert any(row['parameter'].startswith('perceptualSpace.') and row['writers']==['reconstruction']
           for row in ownership['parameters'])
supports={}
for phase in ('start','end'):
    geometry=read(folder/f'geometry-{phase}.json')
    roots=geometry['roots']
    assert roots['shape']==[4,14] and len(roots['pairwise_cosines'])==4
    assert len(roots['centered_singular_values'])==4
    words=[row for book in geometry['dictionary'] for row in book['word_perceptual_support']]
    assert {row['word'] for row in words}=={'hello','world','loving','there'}
    assert all(row['dimension']==6 and row['percept_event_width']==14
               and 0<=row['nonzero_fraction']<=1 and row['minimum_absolute_value']>=0
               for row in words)
    supports[phase]=words
for row in read(folder/'derivation-stability.json'):
    for step in row['modal_derivation']:assert step['rule_name'] and isinstance(step['rule_id'],int)

value=dict(source_matched=True,measurement_harness_matched=True,trainings=30,retries=0,
    shared_xor_trainings=10,named_final_xor_sentences=40,named_final_sum_sentences=40,
    named_compose_steps_in_audit=named,ownership_conflicts=ownership['conflicts'],
    guards_unchanged=True,sum_read_first=True,word_perceptual_support=supports,
    bootstrap_learning='deferred to operators update',antipode_training=False)
with (HERE/(sys.argv[1] if len(sys.argv)>1 else 'review13-verification.json')).open('x') as handle:
    json.dump(value,handle,indent=2);handle.write('\n')
print(json.dumps(value))
