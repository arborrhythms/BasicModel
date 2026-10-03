"""Receipt-only paired first-batch price of automatic tied reconstruction.

Both arms use the same candidate source and unchanged configuration. The
control overrides only automatic scope, restoring the XML's false setting;
the treatment uses the retained-path policy. Ambient independent starts.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[3]
sys.path.insert(0,str(MAIN/'test'))
import bounded_tests as bounded

def child(root, config, arm, output):
    sys.path[:0]=[str(root/'bin'), str(root/'test')]
    import recon_bench, Models
    begin=time.monotonic(); model=None
    result=dict(config=config, arm=arm, seed=None, configuration_changed=False,
        measurement='one first training batch at configured batch; numerical execution without graph capture',
        backend='none')
    try:
        scope=Models.BasicModel._understanding_reconstruction_scope
        def control(self):
            value=scope(self)
            return 'receipt control: XML reconstruction setting' if arm=='before' and value=='understanding' else value
        with patch.object(Models.BasicModel,'_understanding_reconstruction_scope',control):
            model,device,lr,batch=recon_bench._build_model(str(root/config))
        result.update(build_seconds=time.monotonic()-begin,batch=batch,
            reconstruct_in_loop=model.reconstruct_in_loop,scope=model.reconstruction_scope)
        optimizer=model.getOptimizer(lr=lr)
        start=time.monotonic()
        output_loss,recon_loss,_,_=model.runEpoch(optimizer=optimizer,batchSize=batch,split='train',max_batches=1)
        result.update(training_seconds=time.monotonic()-start,output_loss=float(output_loss),
            reconstruction_loss=float(recon_loss),completed=True)
    except BaseException as exc:
        result.update(completed=False,error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        result['total_seconds']=time.monotonic()-begin
        bounded.write_json(output,result)
        if model is not None:model.End()

def run(source, out, configurations=None):
    out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((HERE/'scoped-configuration-audit.json').read_text())
    configs=configurations or [r['config'] for r in rows if r['gains_reconstruction']]
    frozen=bounded.source_snapshot(source)
    bounded.write_json(out/'manifest.json',dict(source=frozen,configs=configs,worker_gib=8,
        max_workers=1,seed=None,backend='none',reason='isolate the automatic reconstruction switch; unchanged numerical parameters'))
    env=bounded.worker_environment(source)
    env.pop('BASIC_SEED',None)
    env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',RUN_SLOW='1')
    done=[]
    for config in configs:
        for arm in ('before','after'):
            stem=Path(config).stem+'-'+arm
            assert frozen==bounded.source_snapshot(source)
            process=bounded.GuardedProcess([sys.executable,str(Path(__file__).resolve()),'--child',
                '--source',str(source),'--config',config,'--arm',arm,'--output',str(out/(stem+'.json'))],
                cwd=source,env=env,log_path=out/(stem+'.log'),memory_bytes=8*bounded.GIB,timeout=1800).start()
            try:
                while (result:=process.poll()) is None:
                    bounded.write_json(out/'progress.json',dict(completed=done,active=stem,pid=process.proc.pid,
                        memory_bytes=process.current_memory_bytes))
                    time.sleep(.25)
            finally:
                if not process.finished:process.stop(exit_code=130,reason='measurement_stopped')
            bounded.write_json(out/(stem+'-process.json'),result)
            done.append(dict(config=config,arm=arm,exit_code=result['exit_code'],reason=result['reason']))
            print(json.dumps(done[-1]),flush=True)
    assert frozen==bounded.source_snapshot(source)
    bounded.write_json(out/'complete.json',dict(results=done,source_matched=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config');p.add_argument('--arm');p.add_argument('--configs',nargs='+');a=p.parse_args()
    if a.child:child(a.source.resolve(),a.config,a.arm,a.output.resolve())
    else:run(a.source.resolve(),a.output.resolve(),a.configs)
