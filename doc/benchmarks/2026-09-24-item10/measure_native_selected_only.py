"""Preserved serial workload; measure retained-program operands after timing."""
import importlib.util
import json
from pathlib import Path
import sys
from collections import deque

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
from Layers import SigmaLayer,PiLayer
from Models import BaseModel


def main():
    result_path=Path(sys.argv[sys.argv.index('--out')+1])
    retained=deque(maxlen=4)
    models=[]
    original_build=BaseModel.from_config
    def build(*args,**kwargs):
        model,config=original_build(*args,**kwargs)
        models.append(model)
        original=model.runBatch
        def capture(*a,**kw):
            result=original(*a,**kw)
            understanding=getattr(model,'_last_understanding',None)
            retained.append(tuple(getattr(understanding,'answer_program',()) or ()))
            return result
        model.runBatch=capture
        return model,config
    BaseModel.from_config=staticmethod(build)
    path=ROOT/'doc/benchmarks/2026-09-21-item10/probe.py'
    spec=importlib.util.spec_from_file_location('serial_probe',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.main()
    finally:
        BaseModel.from_config=original_build
    model=models[-1]
    report=dict(scope='last four validation batches, after seven optimizer updates',
                source='owned AnswerProgram leaves and actions, replayed through current grammar after timing',
                folds={},replay_root_errors=[],leaf_norm_max=0.)
    originals={}
    for kind in (SigmaLayer,PiLayer):
        original=kind.compose
        originals[kind]=original
        def record(self,left,right,*args,_kind=kind,_original=original,**kwargs):
            key=f'{_kind.__name__}/{self.nInput}'
            row=report['folds'].setdefault(key,dict(calls=0,vectors=0,unit_energy_exceeded=0,
                max_abs=0.,max_norm=0.,sample=deque(maxlen=128)))
            row['calls']+=1
            for value in (left,right):
                flat=value.detach().reshape(-1,value.shape[-1]).cpu()
                norms=flat.norm(dim=-1)
                row['vectors']+=len(flat)
                row['unit_energy_exceeded']+=int((norms>1).sum())
                row['max_abs']=max(row['max_abs'],float(flat.abs().max()))
                row['max_norm']=max(row['max_norm'],float(norms.max()))
                row['sample'].extend(flat.tolist())
            return _original(self,left,right,*args,**kwargs)
        kind.compose=record
    try:
        with torch.no_grad():
            for programs in retained:
                for program in programs:
                    if program is None:continue
                    leaves=program.leaves[None]
                    report['leaf_norm_max']=max(report['leaf_norm_max'],float(leaves.norm(dim=-1).max()))
                    replay,_=model._replay_program(leaves,program.actions[None])
                    report['replay_root_errors'].append(float((replay[0]-program.end_state).abs().max()))
    finally:
        for kind,method in originals.items():kind.compose=method
    for row in report['folds'].values():
        values=list(row['sample']);x=torch.tensor(values,dtype=torch.float64)
        u=(x.clamp(-1,1)+1)/2
        arithmetic=u.mean(-1)
        geometric=u.clamp_min(1e-6).log().mean(-1).exp()
        row['sample']=dict(vectors=values,mean_norm=float(x.norm(dim=-1).mean()),
                          am_gm_gap_mean=float((arithmetic-geometric).mean()),
                          am_gm_gap_max=float((arithmetic-geometric).max()))
    result_path.with_name(result_path.stem+'-ranges.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
