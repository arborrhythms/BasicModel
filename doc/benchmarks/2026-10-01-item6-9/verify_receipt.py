"""Audit coverage, probe mechanics, environment and the uncommitted source."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded
from audit import bodies


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


start = read(HERE / 'starting-record.json')
source = read(HERE / 'repaired-source.json')
assert bounded.source_snapshot(ROOT) == source
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == start['head']
assert sha(HERE/'deriv69.py') == start['claude_probe']['sha256']
assert sha(HERE/'measure_mm_grammar.py') == sha(HERE.parent/'2026-09-30-item7-review-round5/measure_mm_grammar.py')
assert read(HERE/'saved-six-failures/source.json') == start['validated_source']
assert len(read(HERE/'saved-six-failures/failures.json')) == 6
assert read(HERE/'repaired-six/source-manifest.json')['validated_source'] == source
assert read(HERE/'repaired-xor/candidate/source-manifest.json')['validated_source'] == source
manifest = read(HERE/'measurements/source-manifest.json')
assert manifest['validated_source'] == source
assert all(sha(HERE/name) == digest for name,digest in manifest['harness'].items())
assert [n for n in sorted(source.keys() | start['validated_source'].keys())
        if source.get(n) != start['validated_source'].get(n)] == ['bin/Models.py']
historical = read(HERE.parent/'2026-09-30-item6-9/starting-record.json')['preserved']
assert {name:sha(ROOT/name) for name in historical} == historical

ledgers=[]
for label,after in (('compiled-head',HERE/'grammar-staging/bin/Models.py'),
                    ('grammar-staging',ROOT/'bin/Models.py')):
    old=bodies(HERE/label/'bin/Models.py')
    new=bodies(after)
    ledger=read(HERE/label/'bodies.json')
    changed={name for name in old.keys() | new.keys() if old.get(name)!=new.get(name)}
    assert {row['symbol'] for row in ledger}==changed
    assert all(row['old_body']==old.get(row['symbol']) and row['new_body']==new.get(row['symbol']) for row in ledger)
    ledgers.append(dict(repair=label,complete_old_new_bodies=len(ledger)))
assert read(HERE/'compiled-head/after-source.json')==read(HERE/'grammar-staging/source.json')
assert read(HERE/'grammar-staging/after-source.json')==source
affected=read(HERE/'repaired-six/run/result.json')
assert affected['exit_code']==0 and len(affected['selected'])==len(affected['completed'])==12
xor=read(HERE/'repaired-xor/candidate/summary.json')
assert xor['named_pytest_outcomes']==dict(passed=33,failed=2)
assert xor['roundtrip_attempts']==1 and xor['guarded_roundtrip_outcomes']==dict(passed=1)

rows = []
for variant in ('a','b','c'):
    for trial in range(1,11):
        name = f'{variant}-{trial:02}'
        path = HERE/'measurements'/name
        value = read(path/'measurement.json')
        process = read(path/'process.json')
        assert process['exit_code'] == 0 and process['reason'] == 'exit'
        assert process['peak_memory_bytes'] < 8 * bounded.GIB
        answers,targets=value['answers'],value['targets']
        assert len(answers)==len(targets)==4 and all(math.isfinite(x) for x in answers)
        error=sum((a-b)**2 for a,b in zip(answers,targets))/4
        correct=sum((a>.5)==(b>.5) for a,b in zip(answers,targets))
        assert abs(error-value['mse']) < 1e-12
        assert value['settled_bar'] == (correct==4 and error<.05)
        pairs=[json.loads(line) for line in (path/'pairs.jsonl').read_text().splitlines()]
        train=[r for r in pairs if r['training']]
        expected=5 if variant=='b' else 2
        assert len(train)==400
        assert all(r['equal_parameter_versions'] and r['all_costed_before_training']
                   and r['optimizer_steps']==expected and all(len(c)==expected for c in r['costs']) for r in train)
        events=[json.loads(line) for line in (path/'trials.jsonl').read_text().splitlines()]
        counts=Counter(r['epoch'] for r in events if r['training'])
        assert counts==Counter({i:expected for i in range(400)})
        assert all(r['rebuild_deviation']<2e-5 and r.get('root_deviation',0)<2e-5 for r in events)
        assert value['best_first_epoch']['epoch']==0 and value['best_last_epoch']['epoch']==399
        assert value['patch_sha256']==sha(HERE/'probe-patches'/f'{variant}.patch')
        norms=[sample['norms'] for sample in value['answer_gradient_samples']]
        rows.append(dict(name=name,gate=value['settled_bar'],training_trials=sum(counts.values()),
            code_answer_gradient_observed=any(any(n.startswith('conceptualSpaces.') and n.endswith('.W') for n in sample) for sample in norms),
            chooser_answer_gradient_observed=any(any('operation_layer.' in n for n in sample) for sample in norms)))

mm=[]
for trial in range(1,11):
    path=HERE/'measurements'/f'mm-{trial:02}'
    result,process=read(path/'measurement.json'),read(path/'process.json')
    assert result['completed_epochs']==900 and process['exit_code']==0 and process['reason']=='exit'
    assert process['peak_memory_bytes'] < 8 * bounded.GIB
    assert read(path/'dispatch.json') == dict(kind='mm', model_compile='eager',
                                            device='cpu', seed=None, autoload='false')
    mm.append(result['ending_training_mse'])

progress=read(HERE/'measurements/progress.json')
assert progress['complete'] and len(progress['completed']) == 40
assert not progress['active'] and progress['pending'] == 0
assert progress['peak_memory_bytes'] < 24 * bounded.GIB
mm_attempts=read(HERE/'mm-environment-failure/attempts.json')
assert len(mm_attempts)==3
assert all(row['completed_epochs']<900 and row['process']['exit_code']!=0 for row in mm_attempts)

freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],cwd=ROOT,text=True)
(HERE/'environment-after.txt').write_text(freeze)
assert freeze==(HERE/'environment-before.txt').read_text()
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert not staged
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
docs=read(HERE/'final-doc-links/run/result.json')
assert docs['exit_code']==0 and len(docs['selected'])==len(docs['completed'])==6
assert read(HERE/'final-doc-links/source-manifest.json')['validated_source']==source
result=dict(head=start['head'],source_files=len(source),source_matches_all_receipts=True,
    production_files_changed_in_followup=['bin/Models.py'],test_bodies_unchanged=True,
    configurations_unchanged=True,environment_unchanged=True,protected_historical_records=len(historical),
    nothing_staged=True,completed_variant_runs=30,final_mm_runs=10,variant_checks=rows,
    completed_measurement_guard_stops=0,peak_aggregate_gib=progress['peak_memory_bytes']/bounded.GIB,
    observer_attempts=dict(failed=1,interrupted=2,completed=0),
    mm_environment_attempts=dict(interrupted=3,completed=0,partial_epochs=[row['completed_epochs'] for row in mm_attempts]),
    mm_compile_backend='eager',mm_helper_matches_accepted_item7=True,
    repair_body_ledgers=ledgers,affected_checks_passed=12,
    named_xor_outcomes=xor['named_pytest_outcomes'],exact_roundtrip_attempts=1,
    reporting_correction='XOR grammar diagnostics recomputed from stored answers; no training rerun.',
    documentation_checks=dict(passed=6,elapsed_seconds=docs['elapsed_seconds']),
    part4_started=False,decision='Stop for review; nothing committed.')
bounded.write_json(HERE/'verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='variant_checks'},indent=2))
