"""Collect completed raw artifacts without rewriting a failed or partial run."""
from collections import Counter
import gzip, hashlib, json, shutil, sys, tarfile
from pathlib import Path
sys.path[:0]=['test']
from bounded_tests import source_snapshot
root=Path.cwd();out=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def save(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')

full=root/'output/item6-identity-full-meaning'
if (full/'result.json').is_file():
    result=read(full/'result.json');reports=[r for w in result['workers'] for r in w['reports']]
    by_node={}
    for row in reports:
        node=row['nodeid'];prior=by_node.get(node)
        if prior is None or row['outcome']=='failed' or (prior['outcome']!='failed' and row['phase']=='call'):
            by_node[node]=row
    counts=Counter(r['outcome'] for r in by_node.values());failures=[r for r in by_node.values() if r['outcome']=='failed']
    baseline=read(root/'doc/benchmarks/2026-10-09-item6-1/landing/sweep-summary.json')
    basefails={r['nodeid'] for r in baseline['failures']};now={r['nodeid'] for r in failures}
    summary=dict(exit_code=result['exit_code'],reason=result['reason'],selected=len(result['selected']),
        completed=len(result['completed']),outcomes=dict(counts),unique_reports=len(by_node),raw_reports=len(reports),
        elapsed_seconds=result['elapsed_seconds'],limits=result['limits'],peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
        source_matched=source_snapshot(root)==read(out/'source.json'),failure_nodes=sorted(now))
    previous=read(out/'development/taxonomy-underflow-attempt/full/baseline-comparison.json')
    before_restore=read(out/'development/strict-checkpoint-attempt/full/baseline-comparison.json')
    compare=dict(baseline_commit='7c9aeac67840165fc1ac116eca8a528ca716c30e',baseline_counts=baseline['counts'],
        candidate_counts=dict(counts),new_failures_vs_accepted_baseline=sorted(now-basefails),persisting_failures=sorted(now&basefails),
        baseline_failures_with_current_outcome={n:by_node.get(n,{}).get('outcome') for n in sorted(basefails)},
        adverse_xpass_to_xfail=['test/test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input'],
        retained_first_attempt_new_failures=previous['new_failures_vs_accepted_baseline'],
        retained_pre_restoration_new_failures=before_restore['new_failures_vs_accepted_baseline'],
        note='New means absent from the accepted 6.1 failing-node list, not proof of causal attribution. Passing historical failures remain open.')
    save(out/'full/summary.json',summary);save(out/'full/baseline-comparison.json',compare);save(out/'full/failures.json',failures)
    save(out/'full/result.json',result)
    for name in ('report.html','source.json'):
        if (full/name).exists():shutil.copyfile(full/name,out/'full'/name)
    with tarfile.open(out/'full/logs.tar.gz','w:gz') as archive:
        for p in sorted(full.iterdir()):
            if p.is_file() and (p.suffix in ('.log','.json','.txt') or p.name=='report.html'):
                archive.add(p,arcname=p.name)
    print('full',summary)
run=root/'output/item6-identity-meaning'
if (run/'source-check.json').is_file():
    if not read(run/'source-check.json')['source_matched']:raise RuntimeError('experiment source changed')
    # Earlier complete/partial copies are retained in development before a
    # corrected run is collected. Do not mix their markers or model XMLs.
    if (out/'experiment').exists():shutil.rmtree(out/'experiment')
    reports={};status=read(run/'branch-status.json') if (run/'branch-status.json').exists() else {}
    for name in read(run/'plan.json')['branches']:
        path=run/name
        if (path/'results.json').is_file():
            r=read(path/'results.json')
            reports[name]={k:v for k,v in r.items() if k not in ('results','rows')}
            if (path/'failure.json').is_file():
                reports[name]['post_result_failure']=read(path/'failure.json')
        elif (path/'failure.json').is_file():reports[name]=read(path/'failure.json')
        else:reports[name]=dict(completed=False,not_run=not (path/'start.json').exists())
        reports[name]['process_exit_code']=status.get(name)
    warm=run/'warmup'
    warm_report=read(warm/'results.json') if (warm/'results.json').exists() else []
    stages=[]
    for row in warm_report:
        if row['stage']<3:continue
        ce=[v for values in row.get('lesson_cross_entropy',[]) if values is not None
            for v in (values if isinstance(values,list) else [values])]
        final=row.get('final',{})
        last_ce=row.get('lesson_cross_entropy',[None])[-1]
        stages.append(dict(name=row['name'],presentations=row['presentations'],
            exact_readbacks=final.get('exact_readbacks'),free_exact=final.get('free_exact'),
            answer_exact=final.get('answer_exact'),native_reported_passed=final.get('passed'),
            order_ce_gate=(all(v<=.01 for v in last_ce) if row['stage']==3 and last_ce is not None else None),
            ce_range=[min(ce),max(ce)] if ce else None,
            final_ce=last_ce))
    needed=['recovery-factorial-on','recovery-factorial-off','recovery-confounded-off']
    invalid=read(run/'invalid-measurement.json') if (run/'invalid-measurement.json').exists() else None
    gate=dict(outcome='not_measured',retire=False)
    if all(reports[n].get('completed') and status.get(n)==0 for n in needed):
        on,off,conf=[reports[n] for n in needed]
        tests=dict(factorial_overall=off['identification']>=.9,
            each_kind=all(off['by_atom'][k]['identification']>=.9 for k in ('cat','dog','bird','horse')),
            beats_on=off['identification']>on['identification'],
            confounded_loses=conf['identification']<off['identification'],
            confounded_merges=conf['merged_pairs']>0)
        gate=dict(outcome='passed' if all(tests.values()) else 'failed',retire=False,tests=tests,
            note='A passing numerical result would still require lesson review; this collector never retires a rule.')
    if invalid:
        gate.update(numerical_outcome=gate['outcome'],outcome='invalid_measurement',retire=False,
            note=invalid['reason'])
    save(out/'experiment/summary.json',dict(branches=reports,exit_codes=status,
        warmup=dict(completed=(warm/'warmup.ckpt').exists(),stages=stages,
                    failure=read(warm/'failure.json') if (warm/'failure.json').exists() else None),
        selected_mint_lesson=gate,measurement_valid=invalid is None,invalid_measurement=invalid,
        recovery_interpretability=(read(out/'validation/recovery-summary.json')['interpretability']
            if (out/'validation/recovery-summary.json').exists() else None),
        source_check=read(run/'source-check.json')))
    shutil.copyfile(root/'output/item6-identity-meaning.log',out/'experiment/run.log')
    for p in sorted(run.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.xml','.log','.pt'):
            q=out/'experiment'/p.relative_to(run);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        elif p.is_file() and p.suffix=='.jsonl':
            q=out/'experiment'/p.relative_to(run).with_suffix('.jsonl.gz');q.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(q,'wb') as f:f.write(p.read_bytes())
    for p in run.rglob('*.ckpt'):
        q=out/'experiment'/p.relative_to(run).with_suffix('.ckpt.tar.gz');q.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(q,'w:gz') as a:a.add(p,arcname=p.name)
    print('experiment branches',list(reports))
