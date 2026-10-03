"""One unseeded production-sized first training batch per newly tied configuration."""
import argparse, json, os, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]

def worker(config, output):
    import recon_bench, util
    result=dict(config=config, seed=None, measurement='one first training batch, including cold capture; no evaluation', backend=util.TheCompileBackend)
    started=time.monotonic();model=None
    try:
        model,device,lr,batch=recon_bench._build_model(str(ROOT/config))
        result.update(build_seconds=time.monotonic()-started, batch=batch, serial=model.serial, tied=model.reconstruct_in_loop)
        t=time.monotonic()
        optimizer=model.getOptimizer(lr=lr)
        out,rec,_,_=model.runEpoch(optimizer=optimizer,batchSize=batch,split='train',max_batches=1)
        result.update(training_seconds=time.monotonic()-t, output_loss=float(out), reconstruction_loss=float(rec), completed=True)
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),completed=False)
        raise
    finally:
        result['total_seconds']=time.monotonic()-started
        output.write_text(json.dumps(result,indent=2)+'\n')
        if model is not None:model.End()

def driver(phase, configs):
    import bounded_tests as bounded
    env=bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED',None)
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='none', BASIC_AUTOLOAD='false')
    folder=HERE/('configuration-timing-'+phase);folder.mkdir(exist_ok=False)
    source=bounded.source_snapshot(ROOT)
    bounded.write_json(folder/'manifest.json',dict(source=source,seed=None,worker_gib=8,backend='none',method='one first training batch at configured batch; identical backend before and after',configs=configs))
    results=[]
    for config in configs:
        assert source==bounded.source_snapshot(ROOT)
        name=Path(config).stem
        process=bounded.GuardedProcess([sys.executable,__file__,'--worker',config,'--output',str(folder/(name+'.json'))],cwd=ROOT,env=env,log_path=folder/(name+'.log'),memory_bytes=8*bounded.GIB,timeout=1800).start()
        try:
            while (result:=process.poll()) is None:time.sleep(.25)
        finally:
            if not process.finished:process.stop(exit_code=130,reason='measurement_stopped')
        bounded.write_json(folder/(name+'-process.json'),result)
        results.append(dict(config=config,reason=result['reason'],exit_code=result['exit_code']))
        print(json.dumps(results[-1]),flush=True)
    assert source==bounded.source_snapshot(ROOT)
    bounded.write_json(folder/'complete.json',dict(results=results,source_matched=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--output',type=Path);p.add_argument('--phase');p.add_argument('--configs',nargs='*');a=p.parse_args()
    if a.worker:worker(a.worker,a.output)
    else:
        configs=a.configs or [r['config'] for r in json.loads((HERE/'configuration-audit.json').read_text()) if r.get('serial') and not r.get('old_tied')]
        driver(a.phase,configs)
