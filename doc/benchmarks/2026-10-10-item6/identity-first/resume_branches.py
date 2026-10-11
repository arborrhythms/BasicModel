"""Continue unstarted lessons from the retained warm-up, without new warm-up.

The learner and corpus must match that warm-up. Only the restoration adapter
and its regression may differ. Failed original starts remain in their run.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'bin')]
import torch
from bounded_tests import source_snapshot
from identity_experiment import branch, digest, save, DEST


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('prior',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--branch')
    args=parser.parse_args()
    args.prior=args.prior.resolve();args.out=args.out.resolve()
    torch.set_num_threads(1)
    if args.branch:
        branch(args.out/args.branch,args.out/'warmup',args.branch)
        return 0
    source=source_snapshot(ROOT)
    prior=read(args.prior/'plan.json')
    proof=read(args.prior/'source-check.json')
    changed=sorted(k for k in set(source)|set(prior['source']) if source.get(k)!=prior['source'].get(k))
    if set(changed)-{'test/identity_experiment.py','test/test_identity_experiment.py'}:
        raise RuntimeError(f'learner source changed since warm-up: {changed}')
    if not (proof['source_matched'] and proof['corpus_matched']):
        raise RuntimeError('the retained warm-up is not source/corpus matched')
    corpus={p.name:digest(p) for p in DEST.iterdir() if p.is_file()}
    if corpus!=prior['corpus_sha256']:
        raise RuntimeError('corpus changed since warm-up')
    warm=args.prior/'warmup'
    checkpoint=digest(warm/'warmup.ckpt')
    if checkpoint!=read(warm/'checkpoint.json')['sha256']:
        raise RuntimeError('retained warm-up checkpoint changed')
    if any((args.prior/name/'progress.json').exists() for name in prior['branches']):
        raise RuntimeError('the original branches already presented examples')
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(warm,args.out/'warmup')
    shutil.copytree(args.prior/'corpus',args.out/'corpus')
    save(args.out/'warmup-source.json',prior['source'])
    adapter=digest(Path(__file__))
    plan=dict(prior,source=source,continuation=dict(prior_run=str(args.prior),
        checkpoint_sha256=checkpoint,warmup_source_check=proof,
        changed_source_files=changed,adapter_sha256=adapter,
        restoration_attempt=2,warmup_exposures_added=0,
        reason='Materialize the saved comparison reader before strict parent loading. No original branch reached training.'))
    save(args.out/'plan.json',plan)
    pending=list(plan['branches']);running=[];exit_codes={};started=time.monotonic()
    try:
        while pending or running:
            while pending and len(running)<plan['workers']:
                name=pending.pop(0);log=(args.out/(name+'.log')).open('w')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                    str(args.prior),str(args.out),'--branch',name],stdout=log,stderr=subprocess.STDOUT)
                running.append((name,process,log))
            for name,process,log in tuple(running):
                code=process.poll()
                if code is not None:
                    log.close();running.remove((name,process,log));exit_codes[name]=code
                    save(args.out/'branch-status.json',exit_codes)
            time.sleep(1)
    finally:
        save(args.out/'source-check.json',dict(source_matched=source==source_snapshot(ROOT),
            corpus_matched=corpus=={p.name:digest(p) for p in DEST.iterdir() if p.is_file()},
            continuation_harness_matched=adapter==digest(Path(__file__)),seconds=time.monotonic()-started))
    return int(any(code!=0 for code in exit_codes.values()))


if __name__=='__main__':raise SystemExit(main())
