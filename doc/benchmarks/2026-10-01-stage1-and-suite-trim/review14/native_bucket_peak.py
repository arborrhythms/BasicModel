"""One unseeded production first batch after the fixed journal repair."""
from pathlib import Path
import sys,os,json,time,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import bounded_tests as bounded
if '--child' in sys.argv:
    import recon_bench
    result=dict(seed=None,config='data/BasicModel_answers_tied_benchmark.xml',backend='eager',
        measurement='first training batch at production batch 28, including cold capture; no evaluation',journals=[])
    model=None;start=time.monotonic()
    try:
        model,device,lr,batch=recon_bench._build_model(str(ROOT/result['config']))
        result.update(build_seconds=time.monotonic()-start,batch=batch)
        assert batch==28
        original=model._run_sentence_word_bricks
        def observe(*args,**kwargs):
            result['journals'].append(list(args[6][22].shape))
            bounded.write_json(HERE/'native-bucket-peak/observation.json',result)
            return original(*args,**kwargs)
        model._run_sentence_word_bricks=observe
        begin=time.monotonic()
        out,rec,_,_=model.runEpoch(optimizer=model.getOptimizer(lr=lr),batchSize=batch,split='train',max_batches=1)
        result.update(training_seconds=time.monotonic()-begin,output_loss=float(out),reconstruction_loss=float(rec),completed=True)
    except BaseException as exc:
        result.update(completed=False,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['seconds']=time.monotonic()-start
        bounded.write_json(HERE/'native-bucket-peak/observation.json',result)
        if model is not None:model.End()
else:
    out=HERE/'native-bucket-peak';out.mkdir(exist_ok=False)
    frozen=bounded.source_snapshot(ROOT)
    env=bounded.worker_environment(ROOT);env.pop('BASIC_SEED',None)
    env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',RUN_SLOW='1')
    bounded.write_json(out/'manifest.json',dict(source=frozen,worker_gib=24,ordinary_sweep_gib=8,seed=None))
    p=bounded.GuardedProcess([sys.executable,__file__,'--child'],cwd=ROOT,env=env,log_path=out/'run.log',memory_bytes=24*bounded.GIB,timeout=1800).start()
    try:
        while (r:=p.poll()) is None:
            bounded.write_json(out/'progress.json',dict(pid=p.proc.pid,seconds=time.monotonic()-p.started,memory_bytes=p.current_memory_bytes))
            time.sleep(.25)
    finally:
        if not p.finished:p.stop(exit_code=130,reason='probe_stopped')
    bounded.write_json(out/'process.json',r)
    bounded.write_json(out/'complete.json',dict(source_matched=frozen==bounded.source_snapshot(ROOT),process=r))
    print(json.dumps(r),flush=True)
