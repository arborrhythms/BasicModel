"""Seed-zero first-forward bisection; no training or gate replacement."""
import ast, hashlib, io, json, os, random, subprocess, sys, tarfile, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'mm-first-forward'

def child(variant):
    import numpy as np
    import torch
    sys.path[:0]=[str(Path.cwd()/'bin'),str(Path.cwd()/'test')]
    from test_mm_xor import _fresh_model
    from util import init_device
    from unittest.mock import patch
    from contextlib import ExitStack
    init_device('cpu');random.seed(0);np.random.seed(0);torch.manual_seed(0)
    def digest(model):
        h=hashlib.sha256()
        for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return dict(parameters=h.hexdigest(),torch_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    model,_,data=_fresh_model(str(Path.cwd()/'data/MM_xor.xml'))
    result=dict(variant=variant,construction=digest(model),seed=0,training_steps=0)
    import ModelAttention, WalkTrials
    from Models import BasicModel
    def record(value):
        if torch.is_tensor(value):return value.detach().tolist()
        if isinstance(value,dict):return {k:record(v) for k,v in value.items()}
        return value
    with ExitStack() as stack:
        if variant!='landing':
            import SentenceCredit
            original_departure=SentenceCredit.departure
            result['sentence_departure_calls']=0
            def observed_departure(*args,**kwargs):
                result['sentence_departure_calls']+=1
                return original_departure(*args,**kwargs)
            stack.enter_context(patch.object(SentenceCredit,'departure',observed_departure))
        if variant.startswith('old_stage'):
            source=(OUT/'landing-source/bin/ModelAttention.py').read_text()
            namespace={};exec(compile(source,'landing-ModelAttention','exec'),namespace)
            tree=ast.parse((OUT/'landing-source/bin/WalkTrials.py').read_text())
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='narrowing_pair')
            env=dict(vars(WalkTrials));exec(compile(ast.Module(body=[node],type_ignores=[]),'landing-narrowing-pair','exec'),env)
            pair=env['narrowing_pair']
            def observed_pair(read,score):
                traces=[]; greedy_rng=[]
                def observed_read(**kwargs):
                    reading=read(**kwargs);traces.append(reading)
                    if len(traces)==1:greedy_rng.append(torch.get_rng_state())
                    return reading
                reading,audit=pair(observed_read,score)
                result['percept_comparison']=record(audit)
                result['greedy_actions']=record(traces[0].actions)
                result['selected_actions']=record(reading.actions)
                if variant=='old_stage_greedy':reading=traces[0]
                if variant=='old_stage_restore_rng':torch.set_rng_state(greedy_rng[0])
                return reading,audit
            stack.enter_context(patch.object(WalkTrials,'narrowing_pair',observed_pair,create=True))
            def padded_old_stage(m):
                namespace['stage_input'](m)
                r=m._attention_words
                width=r.accepted.shape[1]-r.poles.shape[1]
                m._attention_words=r._replace(poles=torch.nn.functional.pad(r.poles,(0,0,0,width)),
                    pole_changes=torch.nn.functional.pad(r.pole_changes,(0,width)))
            stack.enter_context(patch.object(ModelAttention,'stage_input',padded_old_stage))
        if variant=='open_scope':
            handoff=ModelAttention.handoff
            def open_scope(m,r):
                handoff(m,r)
                m.conceptualSpace._passback_scope_where=torch.tensor([[0.,1.]]).expand(len(r.accepted),-1)
            stack.enter_context(patch.object(ModelAttention,'handoff',open_scope))
        raw,answer=next(iter(data.data_loader(split='train',num_streams=4)))
        inp=model.inputSpace.prepInput(raw)
        from torch.utils._python_dispatch import TorchDispatchMode
        import traceback
        stochastic=[]
        class RandomTrace(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result=func(*args,**(kwargs or {}))
                if torch.Tag.nondeterministic_seeded in func.tags:
                    stochastic.append(dict(operation=str(func),
                        callers=[f'{Path(f.filename).name}:{f.lineno}:{f.name}' for f in traceback.extract_stack() if '/bin/' in f.filename][-5:]))
                return result
        with RandomTrace(): _,_,value,_=model.forward(inp)
        result['random_operations']=stochastic
        result.update(predictions=record(value),after_forward=digest(model),
            ir_mask=record(getattr(model,'_ir_mask_positions',None)),
            scope=record(getattr(model.conceptualSpace,'_passback_scope_where',None)),
            actions=record(model._attention_words.actions),
            landing_attention_comparison=record(getattr(model,'_last_attention_comparison',None)))
    (OUT/(variant+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));model.End();model.symbolSpace.soft_reset()

def main():
    sys.path.insert(0,str(ROOT/'test'))
    import bounded_tests
    manifest=json.loads((HERE/'delivered-source/source.json').read_text())
    assert manifest==bounded_tests.source_snapshot(ROOT)
    OUT.mkdir(exist_ok=False)
    tar_bytes=subprocess.check_output(['git','archive','42daf96f4','bin','test','data'],cwd=ROOT)
    (OUT/'landing-source.tar').write_bytes(tar_bytes)
    (OUT/'landing-source').mkdir()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:tar.extractall(OUT/'landing-source',filter='data')
    with zipfile.ZipFile(HERE/'before/source.zip') as saved:saved.extractall(OUT/'round2e-source')
    processes=[]
    for variant,source in [('landing',OUT/'landing-source'),('round2e',OUT/'round2e-source'),('round3a',ROOT),('old_stage',ROOT),('old_stage_greedy',ROOT),('old_stage_restore_rng',ROOT),('open_scope',ROOT)]:
        env=dict(os.environ,BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false')
        env.pop('BASIC_SEED',None)
        with (OUT/(variant+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'child',variant],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        processes.append(dict(variant=variant,exit_code=p.returncode));print(processes[-1],flush=True)
        (OUT/'processes.json').write_text(json.dumps(processes,indent=2)+'\n')
    assert all(p['exit_code']==0 for p in processes)
    read=lambda name:json.loads((OUT/(name+'.json')).read_text())
    landing,current,old,restore,opened=map(read,('landing','round3a','old_stage','old_stage_restore_rng','open_scope'))
    prior=json.loads((HERE.parent/'2026-10-06-operators-round2c/mm-first-forward/round2.json').read_text())
    def flat(value):
        return [item for child in value for item in flat(child)] if isinstance(value,list) else [value]
    result=dict(seed=0,training_steps=0,
        source_manifest_sha256=hashlib.sha256((HERE/'delivered-source/source.json').read_bytes()).hexdigest(),
        construction_equal=landing['construction']==current['construction'],
        scope_equal=landing['scope']==current['scope'],kept_actions_equal=landing['actions']==current['actions'],
        landing_percept_explore_wins=old['percept_comparison']['wins'],
        old_percept_stage_matches_landing=old['predictions']==landing['predictions'],
        rng_after_greedy_restoration_matches_candidate=restore['predictions']==current['predictions'],
        open_scope_matches_candidate=opened['predictions']==current['predictions'],
        candidate_matches_round2c_bisection=current['predictions']==prior['predictions'],
        round2e_construction_equal=current['construction']==read('round2e')['construction'],
        round2e_first_forward_equal=current['predictions']==read('round2e')['predictions'],
        round2e_after_forward_equal=current['after_forward']==read('round2e')['after_forward'],
        first_forward_differs_from_landing=current['predictions']!=landing['predictions'],
        sentence_departure_calls=current['sentence_departure_calls'],
        mask_changed_positions=sum(a!=b for a,b in zip(flat(landing['ir_mask']),flat(current['ir_mask']),strict=True)),
        random_operations=dict(landing=len(landing['random_operations']),round3a=len(current['random_operations'])),
        current_source_matched=manifest==bounded_tests.source_snapshot(ROOT),
        path=['ModelAttention.stage_input','retired WalkTrials.narrowing_pair draws','Torch global RNG state',
              'BasicModel.create_ir_mask: Bernoulli','masked perceptual forward','answer head'])
    result['rng_only_path_confirmed']=(all(result[k] for k in ('construction_equal','scope_equal','kept_actions_equal',
        'old_percept_stage_matches_landing','rng_after_greedy_restoration_matches_candidate','open_scope_matches_candidate',
        'candidate_matches_round2c_bisection','round2e_construction_equal','round2e_first_forward_equal','round2e_after_forward_equal','first_forward_differs_from_landing','current_source_matched'))
        and not any(result['landing_percept_explore_wins']) and result['sentence_departure_calls']==0)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['rng_only_path_confirmed']
if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[2])
    else:main()
