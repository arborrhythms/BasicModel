"""Summarize the one final receipt and verify the unchanged candidate snapshot."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'test'),str(HERE.parent/'2026-09-28-item7-review')]
from bounded_tests import source_snapshot
from review_source import supporting_inputs


def write(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2)+'\n')


def freeze():
    write('final-source.json',source_snapshot(ROOT))
    write('final-inputs.json',supporting_inputs(ROOT))


def finish():
    result_path=HERE/'full-sweep/combined-result.json'
    if not result_path.exists():
        result_path=HERE/'full-sweep/run/result.json'
    result=json.loads(result_path.read_text())
    assert result['reason']!='running'
    reports=[dict(r,log=w['log']) for w in result['workers'] for r in w['reports']]
    priority={'passed':0,'skipped':1,'xfailed':2,'xpassed':3,'failed':4}
    outcomes={}
    for report in reports:
        node=report['nodeid'];value=report['outcome']
        if node not in outcomes or priority.get(value,5)>priority.get(outcomes[node],5):
            outcomes[node]=value
    for node in result['completed']:
        outcomes.setdefault(node,'no-report')
    for node,stop in result.get('stopped_cases',{}).items():
        assert node not in outcomes
        outcomes[node]=stop['reason']
    for node in result['selected']:
        outcomes.setdefault(node,'unexecuted')
    failures=[r for r in reports if r['outcome'] in ('failed','xpassed')]
    write('full-sweep/failures.json',failures)
    prior=json.loads((HERE.parent/'2026-10-01-item6-9-continuation/full-sweep/summary.json').read_text())
    ports=json.loads((HERE/'port-selectors.json').read_text())
    historical=json.loads((HERE/'policy/unchanged-historical-tests.json').read_text())
    known=list(prior['accepted_repairs'])+ports+['test/test_output_walk.py::'+name for name in historical['cases']]
    coverage_path=HERE/'full-sweep/combined-coverage.json'
    if not coverage_path.exists():
        coverage_path=HERE/'full-sweep/coverage.json'
    segment_paths=result.get('segments',[str(result_path)])
    runner_seconds=sum(json.loads(Path(p).read_text())['elapsed_seconds'] for p in segment_paths)
    summary=dict(selected=len(result['selected']),completed=len(result['completed']),
        result_path=str(result_path),reason=result['reason'],
        case_outcomes=dict(Counter(outcomes.values())),
        extra_pass_reports=sum(r['outcome']=='passed' for r in reports)-sum(v=='passed' for v in outcomes.values()),
        wall_seconds=result['elapsed_seconds'],wall_minutes=result['elapsed_seconds']/60,
        runner_seconds=runner_seconds,
        inter_segment_seconds=max(0.,result['elapsed_seconds']-runner_seconds),
        baseline_round5_cases=5122,baseline_round5_minutes=89.6,
        case_delta=len(result['selected'])-5122,wall_delta_minutes=result['elapsed_seconds']/60-89.6,
        peak_aggregate_gib=result['peak_aggregate_memory_bytes']/2**30,
        compile_cache_retries=result['compile_cache_retries'],
        coverage=json.loads(coverage_path.read_text()),
        segments=result.get('segments',[str(result_path)]),
        interrupted_attempts=result.get('interrupted_attempts',[]),
        guard_stopped_cases=result.get('stopped_cases',{}),
        required_repair_outcomes={node:outcomes.get(node,'not selected') for node in known})
    write('full-sweep/summary.json',summary)
    text=['# Final source-matched sweep','',
        f"{summary['completed']}/{summary['selected']} cases completed in {summary['wall_minutes']:.3f} minutes.",
        f"Case outcomes: {summary['case_outcomes']}. Extra passed subtest reports: {summary['extra_pass_reports']}.",
        f"Round 5: 5,122 cases, 89.6 minutes. Change: {summary['case_delta']:+d} cases, {summary['wall_delta_minutes']:+.3f} minutes.",
        f"Runner time: {runner_seconds/60:.3f} minutes; continuation preparation between runner segments: {summary['inter_segment_seconds']/60:.3f} minutes (included in wall time).",
        f"Peak aggregate memory: {summary['peak_aggregate_gib']:.3f} GiB. Worker guard: 8 GiB; aggregate guard: 24 GiB.",
        f"Result: {summary['reason']}. Coverage details: `{coverage_path.name}`.",
        f"Guard-stopped cases: {len(summary['guard_stopped_cases'])}; interrupted sibling attempts: {len(summary['interrupted_attempts'])}.",
        'The original collected list was continued after a worker guard stopped dispatch. Completed cases and the case stopped by its own guard were not rerun; the original three-hour suite deadline and memory limits remained.', '',
        '| Guard-stopped case | Reason | Seconds |','|---|---|---:|',
        *['| '+node+' | '+stop['reason']+' | '+f"{stop['elapsed_seconds']:.3f}"+' |' for node,stop in summary['guard_stopped_cases'].items()], '',
        '| Required regression/port | Outcome |','|---|---|']
    text+=['| '+node+' | '+value+' |' for node,value in summary['required_repair_outcomes'].items()]
    (HERE/'full-sweep/summary.md').write_text('\n'.join(text)+'\n')
    mm=HERE/'final-measurements/candidate/mm'
    rows=[dict(trial=i+1,**json.loads((mm/f'run-{i:02}.json').read_text())) for i in range(10)]
    write('final-mm-summary.json',dict(rows=rows,
        median_ending_training_mse=statistics.median(r['ending_training_mse'] for r in rows),
        median_after_900_updates_mse=statistics.median(r['after_900_updates_mse'] for r in rows),
        accepted_item7_baseline_median=.1066,
        meaning='Same raw-forward 900-update measurement as the accepted receipt; descriptive, not a sentence-trial grammar-learning proof.'))
    environment=subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],cwd=ROOT)
    (HERE/'environment-after.txt').write_bytes(environment)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    frozen=json.loads((HERE/'final-source.json').read_text())
    verification=dict(head=head,source_files=len(frozen),source_matches_frozen=frozen==source_snapshot(ROOT),
        supporting_inputs_match_frozen=json.loads((HERE/'final-inputs.json').read_text())==supporting_inputs(ROOT),
        environment_unchanged=environment==(HERE/'environment-before.txt').read_bytes(),
        freeze_sha256=hashlib.sha256(environment).hexdigest(),git_diff_check='passed' if diff.returncode==0 else diff.stdout+diff.stderr,
        committed=False)
    write('final-verification.json',verification)
    print(json.dumps(dict(sweep=summary['case_outcomes'],minutes=summary['wall_minutes'],verification=verification),indent=2))
    assert all(verification[k] for k in ('source_matches_frozen','supporting_inputs_match_frozen','environment_unchanged'))
    assert head=='d679df2b5a2665d72a99ca4b6dfd47c1ba048e99' and diff.returncode==0


if __name__=='__main__':
    globals()[sys.argv[1]]()
