"""B/C/D off individually on every missed standing run, never replacements."""
import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(HERE)]
import bounded_tests as bounded
import campaign

def read(path):return json.loads(path.read_text())

def main():
    output=HERE/'bisections';output.mkdir(exist_ok=False)
    snapshot=bounded.source_snapshot(ROOT)
    assert snapshot==read(HERE/'delivered-source/source.json')
    summary=read(HERE/'measurements/summary.json')
    jobs=[]
    for kind, group in [('sum',summary['sum']),('xor',summary['xor']),('mm',summary['mm'])]:
        for row in group:
            passed=row.get('floor_bar',row.get('floor_pass')) if kind=='sum' else row.get('joint') if kind=='xor' else row.get('passed',row.get('pass'))
            if passed is None:raise ValueError((kind,list(row)))
            if not passed:
                jobs += [dict(kind=kind,run=row['run'],part=part,name=f'{kind}-{row["run"]:02}-{part}-off') for part in 'BCD']
    bounded.write_json(output/'plan.json',dict(jobs=jobs,seed=None,retries=0,replacements=0,
        source='recorded unseeded entry state of each failed standing run',count=len(jobs)))
    active=[];done=[]
    while jobs or active:
        assert snapshot==bounded.source_snapshot(ROOT)
        for job in active[:]:
            result=job['process'].poll()
            if result is not None:
                bounded.write_json(job['folder']/'process.json',result)
                done.append({k:v for k,v in job.items() if k not in ('folder','process')}|dict(process=result))
                active.remove(job);print(json.dumps(done[-1]),flush=True)
        while jobs and len(active)<3:
            job=jobs.pop(0);folder=output/job['name'];folder.mkdir()
            original=HERE/'measurements'/f'{job["kind"]}-{job["run"]:02}'
            env=campaign.environment();env.update(OPERATORS_REPLAY_RNG=str(original/'unseeded-entry.pt'),OPERATORS_DISABLE=job['part'])
            if job['kind']=='sum':command=[sys.executable,str(HERE/'campaign.py'),'sum',str(folder)]
            else:
                env.update(PYTEST_PLUGINS='operators_gate_observer',ITEM7_XOR_GATE='5' if job['kind']=='xor' else '2',
                    ITEM7_XOR_MEASUREMENTS=str(folder/'observations.jsonl'),REVIEW17_REPORTS=str(folder/'reports.jsonl'))
                command=[sys.executable,'-m','pytest','-q',*(campaign.XOR if job['kind']=='xor' else campaign.MM)]
            process=bounded.GuardedProcess(command,cwd=ROOT,env=env,log_path=folder/'run.log',memory_bytes=8*bounded.GIB,timeout=1800).start()
            active.append(job|dict(folder=folder,process=process))
        bounded.write_json(output/'progress.json',dict(done=done,pending=len(jobs),active=[j['name'] for j in active]))
        time.sleep(.5)
    bounded.write_json(output/'complete.json',dict(jobs=done,source_matched=snapshot==bounded.source_snapshot(ROOT)))

if __name__=='__main__':main()
