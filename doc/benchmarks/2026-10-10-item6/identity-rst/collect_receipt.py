"""Collect this one frozen-source rerun, retaining raw failures and byte hashes."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import source_snapshot
from identity_measurement import valid_reference, summarize


def read(path):return json.loads(path.read_text())
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def archive_run(source,destination):
    manifest={}
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(source)
        if path.suffix=='.ckpt':
            target=destination/relative.with_suffix('.ckpt.tar.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            with tarfile.open(target,'w:gz') as archive:archive.add(path,arcname=path.name)
            with tarfile.open(target) as archive:
                with archive.extractfile(path.name) as handle:
                    actual=hashlib.file_digest(handle,'sha256').hexdigest()
        elif path.suffix=='.jsonl' or path.stat().st_size>10_000_000:
            target=destination/Path(str(relative)+'.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as inp,gzip.open(target,'wb') as out:shutil.copyfileobj(inp,out)
            with gzip.open(target,'rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        else:
            target=destination/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);actual=sha(target)
        expected=sha(path)
        assert actual==expected,path
        manifest[str(relative)]=dict(sha256=expected,archive=str(target.relative_to(OUT)),bytes=path.stat().st_size)
    return manifest


def archive_log(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    expected=sha(source)
    assert sha(target)==expected
    return dict(sha256=expected,archive=str(target.relative_to(OUT)),bytes=source.stat().st_size)


def collect_full():
    folder=ROOT/'output/item6-rst-full'
    if not (folder/'result.json').exists():return
    result=read(folder/'result.json')
    if len(result['completed'])!=len(result['selected']):
        print('full sweep still incomplete; not collecting a final receipt',flush=True)
        return
    rows={}
    for worker in result['workers']:
        for row in worker['reports']:
            previous=rows.get(row['nodeid'])
            if previous is None or row['outcome']=='failed' or (previous['outcome']!='failed' and row['phase']=='call'):
                rows[row['nodeid']]=row
    counts=dict(Counter(row['outcome'] for row in rows.values()))
    failures=[row for row in rows.values() if row['outcome']=='failed']
    baseline=read(ROOT/'doc/benchmarks/2026-10-09-item6-1/landing/sweep-summary.json')
    before={r['nodeid'] for r in baseline['failures']}
    prep=read(OUT.parent/'identity-first/full/summary.json')
    now={r['nodeid'] for r in failures}
    save(OUT/'full/summary.json',dict(selected=len(result['selected']),completed=len(result['completed']),
        outcomes=counts,exit_code=result['exit_code'],elapsed_seconds=result['elapsed_seconds'],
        limits=result['limits'],peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
        source_matched=read(OUT/'source.json')==source_snapshot(ROOT),failure_nodes=sorted(now)))
    save(OUT/'full/failures.json',failures)
    save(OUT/'full/baseline-comparison.json',dict(baseline_commit='7c9aeac67840165fc1ac116eca8a528ca716c30e',
        baseline_counts=baseline['counts'],current_counts=counts,new_vs_accepted_6_1=sorted(now-before),
        persisting_vs_accepted_6_1=sorted(now&before),new_vs_preparation=sorted(now-set(prep['failure_nodes'])),
        baseline_failure_outcomes={node:rows.get(node,{}).get('outcome') for node in sorted(before)},
        adverse_overlap_outcome=rows.get('test/test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input',{}).get('outcome'),
        note='New means absent from the baseline failing-node list; it does not itself establish causality. Historical open failures are not retired by a passing run.'))
    manifest=archive_run(folder,OUT/'full/raw')
    manifest['../item6-rst-full.log']=archive_log(ROOT/'output/item6-rst-full.log',OUT/'full/run.log')
    save(OUT/'full/archive-manifest.json',manifest)
    print('full',counts,flush=True)


def collect_experiment(*,archive=True):
    run=ROOT/'output/item6-identity-rst'
    if not (run/'source-check.json').exists():return
    check=read(run/'source-check.json')
    assert check['source_matched'] and check['corpus_matched'],check
    status=read(run/'branch-status.json') if (run/'branch-status.json').exists() else {}
    plan=read(run/'plan.json');reports={};hashes={}
    for name in plan['branches']:
        folder=run/name
        if not (folder/'results.json').exists():
            reports[name]=dict(completed=False,process_exit_code=status.get(name),
                failure=read(folder/'failure.json') if (folder/'failure.json').exists() else None)
            continue
        native=read(folder/'results.json')
        report={k:v for k,v in native.items() if k not in ('rows','results')}
        report['process_exit_code']=status.get(name)
        if (folder/'failure.json').exists():report['post_result_failure']=read(folder/'failure.json')
        report['start']=read(folder/'start.json')
        if (folder/'final.ckpt').exists():hashes[name]=sha(folder/'final.ckpt')
        if name.startswith('identity'):
            assert native['summary']==summarize(native['results'])
            observations=read(folder/'observations.json')
            tokens=[v for row in observations for v in row['provenance'].values()]
            refs=[v for row in observations for v in row['references'].values()]
            assert all(valid_reference(ref) or ref is None for ref in refs)
            assert all(valid_reference(t.get('reference')) or t.get('missing_reason') for t in tokens)
            report['mention_audit']=dict(count=len(refs),valid=sum(map(valid_reference,refs)),
                missing=sum(r is None for r in refs),sentinels=sum(r in (-1,0) for r in refs),
                missing_reasons=dict(Counter(t['missing_reason'] for t in tokens if not valid_reference(t.get('reference')))),
                sources=dict(Counter(t.get('source','missing') for t in tokens)))
            bindings=[row['binding'] for row in native['results'] if row['binding']]
            report['binding_reference_audit']=dict(probes=len(bindings),
                valid_selected=sum(valid_reference(b['selected_reference']) for b in bindings),
                valid_candidates=sum(sum(map(valid_reference,b['candidate_references'])) for b in bindings),
                candidate_mentions=sum(len(b['candidate_references']) for b in bindings),
                complete_candidate_sets=sum(all(map(valid_reference,b['candidate_references'])) for b in bindings),
                nonempty_native_banks=sum(bool(row.get('native_bank',{}).get('ids'))
                                         for row in native['results'] if row['binding']))
        reports[name]=report
    stages=[]
    for row in read(run/'warmup/results.json'):
        if row['stage']<3:continue
        final=row['final'];initial=row.get('initial',{})
        ce=row.get('lesson_cross_entropy',[None])[-1]
        stages.append(dict(name=row['name'],presentations=row['presentations'],
            trace_initial=initial.get('exact_readbacks'),trace_final=final.get('exact_readbacks'),
            free_final=final.get('free_exact'),answer_final=final.get('answer_exact'),
            final_untaught_read_ce=final.get('lesson_cross_entropy'),
            initial_ce=initial.get('lesson_cross_entropy'),
            last_training_ce=row.get('lesson_cross_entropy',[None])[-1],
            order_ce_passed=None if ce is None else all(v<=.01 for v in ce)))
    factorial=reports['recovery-factorial-off']
    on=reports['recovery-factorial-on']
    confounded=reports['recovery-confounded-off']
    lesson=None
    if all(r.get('completed') for r in (factorial,on,confounded)):
        lesson=dict(factorial_overall=factorial['identification']>=.9,
            each_kind=all(factorial['by_atom'][k]['identification']>=.9 for k in ('cat','dog','bird','horse')),
            beats_gate_on=factorial['identification']>on['identification'],
            confounded_loses_identification=confounded['identification']<factorial['identification'],
            confounded_has_merge=confounded['merged_pairs']>=1)
        lesson['passed']=all(lesson.values())
    save(OUT/'experiment/summary.json',dict(branches=reports,status=status,stages=stages,
        source_check=check,checkpoints=hashes,on_off_equal={c:bool(hashes.get(c+'-on')) and hashes.get(c+'-on')==hashes.get(c+'-off')
            for c in ('recovery-factorial','recovery-confounded','identity-balanced','identity-biased')},
        gate_lesson=lesson,rule_retired=False))
    if archive:
        manifest=archive_run(run,OUT/'experiment/raw')
        manifest['../item6-identity-rst.log']=archive_log(ROOT/'output/item6-identity-rst.log',OUT/'experiment/run.log')
        save(OUT/'experiment/archive-manifest.json',manifest)
    print('experiment',status,flush=True)


def collect_attention():
    folder=ROOT/'output/item6-rst-attention'
    if not (folder/'result.json').exists():return
    result=read(folder/'result.json')
    assert len(result['steps'])==64
    source=read(OUT/'source.json')
    assert all(source[path]==digest for path,digest in result['source_sha256'].items())
    def stage(value,reads):
        return dict(ce=value['lesson_cross_entropy'],trace_exact=value['exact_readbacks'],
            free_exact=value['free_exact'],orders=value['orders'],
            saturated_units=sum(r['saturated'] for r in reads),
            preactivation_min=min(r['min'] for r in reads),
            preactivation_max=max(r['max'] for r in reads),
            two_candidate_logit_gaps=[abs(row[0]-row[1]) for r in reads
                                     for row in r['logits'] if len(row)==2])
    initial=stage(result['initial'],result['initial_reads'])
    final=stage(result['final'],result['final_reads'])
    final_ce=result['steps'][-1]['ce']
    save(OUT/'validation/attention-summary.json',dict(checkpoint_sha256=result['checkpoint_sha256'],
        source_sha256=result['source_sha256'],source_matched=True,presentations=64,
        initial=initial,final=final,training_saturated_units=sum(r['saturation'] for r in result['steps']),
        first_training_ce=result['steps'][0]['ce'],last_training_ce=final_ce,
        every_last_training_ce_below_log2_over2=all(c<math.log(2)/2 for c in final_ce),
        order_ce_threshold=.01,order_ce_passed=all(c<=.01 for c in final_ce),
        diagnostic_state_discarded=True))
    manifest=archive_run(folder,OUT/'validation/attention')
    manifest['../item6-rst-attention.log']=archive_log(ROOT/'output/item6-rst-attention.log',OUT/'validation/attention/run.log')
    save(OUT/'validation/attention-archive-manifest.json',manifest)
    print('attention',final_ce,flush=True)


def collect_warmup_comparison():
    current=ROOT/'output/item6-identity-rst/warmup'
    previous=ROOT/'output/item6-identity-meaning/warmup'
    if not (current/'checkpoint.json').exists():return
    def stages(folder):
        return {r['name']:dict(trace_exact=r['final'].get('exact_readbacks'),
            free_exact=r['final'].get('free_exact'),answer_exact=r['final'].get('answer_exact'),
            last_training_ce=r.get('lesson_cross_entropy',[None])[-1],
            trace_readbacks=r['final'].get('readbacks'),
            free_readbacks=r['final'].get('free_readbacks'),orders=r['final'].get('orders'))
            for r in read(folder/'results.json') if r['stage']>=3}
    before,after=stages(previous),stages(current)
    assert before.keys()==after.keys()
    adverse=[]
    for name in before:
        for metric in ('trace_exact','free_exact','answer_exact'):
            old,new=before[name][metric],after[name][metric]
            if old is not None and new is not None and new<old:
                adverse.append(dict(stage=name,metric=metric,previous=old,current=new))
    save(OUT/'experiment/warmup-comparison.json',dict(previous=before,current=after,
        new_adverse_results=adverse,previous_checkpoint=read(previous/'checkpoint.json'),
        current_checkpoint=read(current/'checkpoint.json')))
    print('warmup adverse',adverse,flush=True)


def collect_recovery_replay():
    folder=ROOT/'output/item6-rst-recovery-new'
    if not (folder/'result.json').exists():return
    result=read(folder/'result.json')
    rows=result['captures']
    assert len(rows)==48 and not result['retained_attention']
    previous=read(OUT/'validation/recovery-retained/result.json')
    assert [r['id'] for r in rows[:4]]==[r['id'] for r in previous['captures']]
    def row_summary(row):
        methods=[r['method'] for r in row['derivation'] if r['kind']]
        return dict(id=row['id'],text=row['text'],meaning_norm=row['fused_point_meaning_norm'],
            lower_non_non=any(methods[i:i+3]==['lower','non','non'] for i in range(len(methods)-2)),
            methods=methods)
    details=[row_summary(row) for row in rows]
    content=[json.loads(line)['reconstruction.content']
             for line in (folder/'costs.jsonl').read_text().splitlines()]
    charges=[value for term in content for value in term['row_values']]
    assert len(charges)==len(rows)
    save(OUT/'validation/recovery-summary.json',dict(checkpoint_sha256=result['checkpoint_sha256'],
        rows=len(rows),nonzero_meaning_rows=sum((r['meaning_norm'] or 0)>0 for r in details),
        lower_non_non_rows=sum(r['lower_non_non'] for r in details),
        content_charge_min=min(charges),content_charge_max=max(charges),
        content_active_entries=sum(term['active_entries'] for term in content),
        content_trained=all(term['trained'] for term in content),
        retained_first_four=[row_summary(r) for r in previous['captures']],rows_after=details,
        no_optimizer=True,conceptual_admission_frozen=True))
    manifest=archive_run(folder,OUT/'validation/recovery-new')
    manifest['../item6-rst-recovery-new.log']=archive_log(ROOT/'output/item6-rst-recovery-new.log',OUT/'validation/recovery-new/run.log')
    save(OUT/'validation/recovery-archive-manifest.json',manifest)
    print('recovery replay',len(rows),flush=True)


if __name__=='__main__':
    collect_full()
    collect_experiment()
    collect_warmup_comparison()
    collect_attention()
    collect_recovery_replay()
