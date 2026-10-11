"""Saved two-word checkpoint: bound preactivation, then teach 64 native presentations.

This diagnostic state is discarded. The eight branches use their own fresh,
identical-budget warm-up, not this continuation.
"""
import argparse
import json
from pathlib import Path
import warnings
import pytest
import torch
import Models
from attention_curriculum_gates import present, STAGES
from identity_experiment import make_model, restore_warmup, digest, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    restore=Models.BaseModel.load_weights
    reads=[]
    with pytest.MonkeyPatch.context() as patch,warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=make_model(args.out,patch)
        try:
            drift=restore_warmup(model,args.checkpoint,restore)
            optimizer=model.getOptimizer(lr=.003)
            def linear(module,inputs,output):
                reads.append(dict(min=float(output.detach().min()),max=float(output.detach().max())))
            def tanh(module,inputs,output):
                reads[-1]['saturated']=int(output.detach().abs().eq(1).sum())
            def logits(module,inputs,output):
                reads[-1]['logits']=output.detach().tolist()
            layer=model.candidate_attention
            handles=[layer.hidden[0].register_forward_hook(linear),
                     layer.hidden[1].register_forward_hook(tanh),layer.register_forward_hook(logits)]
            ce=[]
            initial=present(model,STAGES[0])
            initial_reads=list(reads)
            for step in range(64):
                reads.clear()
                result=present(model,STAGES[0],train=True,optimizer=optimizer)
                row=dict(step=step+1,ce=result['lesson_cross_entropy'],
                    saturation=sum(x['saturated'] for x in reads),
                    preactivation_min=min(x['min'] for x in reads),
                    preactivation_max=max(x['max'] for x in reads))
                ce.append(row);save(args.out/'progress.json',row)
                with (args.out/'steps.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            reads.clear()
            final=present(model,STAGES[0])
            save(args.out/'result.json',dict(protocol=__doc__,checkpoint_sha256=digest(args.checkpoint),
                source_sha256={p:digest(Path(p)) for p in ('bin/CandidateAttention.py','bin/Models.py')},
                native_loader_drift=drift,initial=initial,initial_reads=initial_reads,
                final=final,final_reads=reads,steps=ce))
            for handle in handles:handle.remove()
        finally:model.End()


if __name__=='__main__':main()
