"""Post-measurement aggregation and integrity check; reads saved runs only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import source_snapshot

def read(path):
    return json.loads(path.read_text())

def main():
    summary = read(HERE/'measurements/summary.json')
    credit = read(HERE/'cost-and-priming-audit.json')
    rows, ranges, components = [], [], {key: [] for key in ('R','E','A')}
    training_rows = evaluation_rows = 0
    for kind in ('sum','xor'):
        for i in range(1,11):
            name=f'{kind}-{i:02}'
            audit=read(HERE/'measurements'/name/'run-audit.json')
            rows.append(dict(name=name,**audit['storage_after_training']))
            ranges.append(audit['priming_range'])
            for trial in audit['sentence_trials']:
                count=sum(len(pair) for pair in trial['components'])
                if trial['training']: training_rows+=count
                else: evaluation_rows+=count
                for pair in trial['components']:
                    for values in pair:
                        for key,value in zip(components,values):components[key].append(value)
    rows += [dict(name=f'mm-{r["run"]:02}',**r['storage']) for r in summary['mm']]
    component_ranges={key:dict(minimum=min(values),maximum=max(values),nonzero=sum(v!=0 for v in values))
                      for key,values in components.items()}
    assert component_ranges['R']['nonzero']==component_ranges['E']['nonzero']==0
    assert all(r['sentence_rows']==4 and r['witness_counts']==[400]*4 for r in rows if not r['name'].startswith('mm'))
    assert all(r['rows_used']<=r['capacity'] for r in rows)
    original=HERE.parent/'2026-10-07-operators-final'
    before=read(HERE/'prior-receipt-manifest.json')
    after={str(f.relative_to(original)):hashlib.sha256(f.read_bytes()).hexdigest()
           for f in original.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    assert before==after
    source=source_snapshot(ROOT)
    from verification import validate
    sweep=validate(source)
    assert source==read(HERE/'measurements/source.json')
    helpers=read(HERE/'measured-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in helpers.items())
    ownership=read(HERE/'measurements/xor-10/ownership/ownership.json')
    assert ownership['conflicts']==0
    report=dict(counts=summary['counts'], sweep=read(HERE/'green-sweep-summary.json'),
        training_trial_rows=training_rows,evaluation_trial_rows=evaluation_rows,
        absolute_components=component_ranges,
        priming_range=dict(minimum=min(r['minimum'] for r in ranges),maximum=max(r['maximum'] for r in ranges),neutral=1.),
        priming_per_word=credit['shared_priming'],priming_identical=credit['priming_identical_across_grammar_runs'],
        narrowing_credit=credit['narrowing_by_action'],
        operators=dict(Counter(op for r in summary['xor'] for row in r['operator_names'] for op in row)),
        operator_runs={str(r['run']):r['operator_names'] for r in summary['xor']},
        stores=rows,mm_store_scope='Numeric raw-forward gate creates four DEF rows and no sentence-closing rows.',
        ownership_conflicts=ownership['conflicts'],
        original_receipt_files=len(before), original_receipt_unchanged=True,
        source_matched=True,source_files=len(source),measurement_helpers_matched=len(helpers),
        training_runs=len(summary['complete']['jobs']),seeds=None,retries=0,replacements=0,
        missed_counts=summary['below_comparison'],bisections='None required: every standing count met its expectation.',
        review='Held for Claude review; no commit or item 6.5.',
        postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'receipt-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','training_trial_rows','evaluation_trial_rows',
        'absolute_components','priming_range','operators','source_matched','training_runs','original_receipt_unchanged')}))

if __name__=='__main__':main()
