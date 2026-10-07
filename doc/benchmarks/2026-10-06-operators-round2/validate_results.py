"""Postprocess saved observations only; imports no model and draws no RNG."""
from collections import Counter
import hashlib, json, math, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'measurements'
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests

def read(path): return json.loads(path.read_text())
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')
def main():
    source=read(HERE/'delivered-source/source.json')
    assert source==bounded_tests.source_snapshot(ROOT)==read(OUT/'source.json')
    assert source==read(HERE/'full-sweep/source-manifest.json')['validated_source']
    helpers=read(HERE/'delivered-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha for n,sha in helpers.items())
    summary=read(OUT/'summary.json');audit=read(OUT/'audit-summary.json')
    assert summary['complete']['completed'] and summary['complete']['retries']==0
    jobs=summary['complete']['jobs']
    assert len(jobs)==len({(r['kind'],r['run']) for r in jobs})==30
    reports={}
    for file in sorted((HERE/'full-sweep').glob('worker-*.json')):
        for r in read(file).get('reports',[]):
            if r['phase']=='call' or r['outcome'] in ('skipped','failed'):reports[r['nodeid']]=r
    sweep=read(HERE/'full-sweep/result.json')
    assert sweep['exit_code']==0 and set(reports)==set(sweep['selected'])
    runs=[]
    for kind in ('sum','xor'):
        for run in range(1,11):
            a=read(OUT/f'{kind}-{run:02}/run-audit.json')
            books=[]
            for start,end in zip(a['start']['codes'],a['end']['codes'],strict=True):
                assert start['rows']==end['rows']
                delta=max((abs(v-w) for row,other in zip(start['values'],end['values'],strict=True)
                          for v,w in zip(row,other,strict=True)),default=0.)
                books.append(dict(rows=len(start['rows']),code_maximum_change=delta))
            gradients=a['sentence_gradients']
            assert gradients
            runs.append(dict(kind=kind,run=run,codebooks=books,
                zero_sentence_path_gradient=all(g['maximum']==0 and g['nonzero']==0 for g in gradients.values()),
                gradients=gradients,reader_epochs=len(a['reader_weights'])))
    tenth=read(OUT/'xor-10/run-audit.json')
    assert len(tenth['compose_score_function_steps'])==1600
    assert len([r for r in tenth['sentence_trials'] if r['training']])==400
    assert all(r.get('gradient_max_error',0)<=2e-6 and
               r.get('finite_difference',{}).get('error',0)<=2e-6
               for r in tenth['compose_score_function_steps'])
    assert all(v==0 for v in tenth['closing_image']['concept_widths'].values())
    assert all(v==0 for v in tenth['closing_image']['image_max'].values())
    assert audit['steps_reaching_decoder']==0
    zero_codes=all(b['code_maximum_change']==0 for r in runs for b in r['codebooks'])
    zero_gradients=all(r['zero_sentence_path_gradient'] for r in runs)
    zero_conflicts=audit['ownership']['conflicts']==0
    per_run=[]
    for x in summary['xor']:
        names=sorted(set(name for row in x['operator_names'] for name in row))
        per_run.append(dict(run=x['run'],mse=x['mse'],band=x['band'],correct=x['correct'],recovered=x['recovered'],
            class_pass=x['class_pass'],reconstruction_pass=x['reconstruction_pass'],operators=names))
    result=dict(source_matched=True,measurement_helpers_matched=True,gate_trainings=30,retries=0,
        sweep=dict(selected=len(sweep['selected']),completed=len(sweep['completed']),
            outcomes=dict(Counter(r['outcome'] for r in reports.values())),seconds=sweep['elapsed_seconds'],
            warnings=sweep['warnings']),
        counts=summary['counts'],below_comparison=summary['below_comparison'],
        xor_runs=per_run,runs=runs,zero_code_displacement=zero_codes,
        zero_sentence_path_gradient=zero_gradients,zero_ownership_conflicts=zero_conflicts,
        standing_gate_passed=not summary['below_comparison'] and zero_codes and zero_gradients and zero_conflicts,
        postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(HERE/'results-validation.json',result)
    print(json.dumps({k:result[k] for k in ('counts','below_comparison','standing_gate_passed','sweep')}))
if __name__=='__main__':main()
