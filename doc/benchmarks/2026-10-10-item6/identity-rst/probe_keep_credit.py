"""One native paired walk on four retained clauses, with zero learning rate.

This is a discarded diagnostic copy, not an additional experiment branch.
Conceptual admission is frozen and trainable parameters must remain identical.
The training path is used only because evaluation visits the greedy reading.
"""
import argparse
import json
from pathlib import Path

import pytest
import torch

import Models
from identity_experiment import digest, make_model, present_documents, recovery_data, restore_warmup, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    native_restore=Models.BaseModel.load_weights
    streams,_=recovery_data()
    traces=[]
    with pytest.MonkeyPatch.context() as patch:
        model=make_model(args.out,patch)
        try:
            drift=restore_warmup(model,args.checkpoint,native_restore)
            owner=model._concept_owner()
            owner._online_learning_frozen=True
            optimizer=model.getOptimizer(lr=0.)
            assert all(group['lr']==0 for group in optimizer.param_groups)
            parameters={name:value.detach().clone() for name,value in model.named_parameters()}
            score=model._sentence_path_cost
            def scoring(state,sid,active):
                result=score(state,sid,active)
                registry=model._sentence_cost_registry
                chart=owner.components._ready()
                roots=result[2]['record'].root.detach()
                traces.append(dict(trial=model._sentence_trial,sentence=int(sid),
                    reconstruction=registry.total(objective='reconstruction').detach().tolist(),
                    content=registry.breakdown()['reconstruction.content'],
                    closed_meaning_norm=roots[...,chart.meaning_start:chart.content_width].norm(dim=-1).tolist()))
                return result
            patch.setattr(model,'_sentence_path_cost',scoring)
            present_documents(model,streams['heldout'][:4],optimizer=optimizer)
            changed=[name for name,value in model.named_parameters()
                     if name not in parameters or not torch.equal(value.detach(),parameters[name])]
            audit=model._last_sentence_credit
            result=dict(protocol=__doc__,checkpoint_sha256=digest(args.checkpoint),
                script_sha256=digest(Path(__file__)),native_loader_drift=drift,
                parameter_changes=changed,trials=traces,
                keep_costs=audit['keep_costs'].tolist(),keep_decision=audit['keep_decision'],
                advantage=audit['advantage'].tolist(),
                departure={k:v.tolist() for k,v in audit['departure'].items()})
            save(args.out/'result.json',result)
            assert not changed,changed
            print(json.dumps(result),flush=True)
        finally:
            model.End()


if __name__=='__main__':main()
