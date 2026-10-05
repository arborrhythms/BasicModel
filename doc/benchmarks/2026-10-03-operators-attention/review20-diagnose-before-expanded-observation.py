"""Observe the three §19 failures on unchanged source and unchanged fixtures."""
import json, inspect, os, traceback
from pathlib import Path
import pytest
import torch
H=Path(__file__).resolve().parent

def serial(x):
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    return repr(x)

@pytest.fixture(autouse=True)
def review20_diagnostic(request,monkeypatch):
    name=request.node.nodeid
    if not any(x in name for x in ('test_normal_text_reconstruction_updates_the_grammar_chooser','test_small_inventory_pairs_words_without_allocating_letter_rows')):
        yield;return
    from Models import BaseModel
    from Language import LanguageSpace
    from Layers import Error
    import Layers
    RowLayer=next(v for v in vars(Layers).values() if isinstance(v,type) and "assign_row" in vars(v))
    from ObjectiveOwnership import registry_costs
    report=dict(test=name,terms=[],decoder=[],allocations=[])
    original=BaseModel._backward_training_loss
    def backward(model,total_loss,*args,**kwargs):
        sentence=bool(getattr(model,'_sentence_backward',False))
        registry=model._sentence_cost_registry if sentence else model.errors
        chooser=model.languageSpace._tree_layer(2).chooser
        targets=[('compose.'+n,p) for n,p in chooser.named_parameters() if p.requires_grad]
        targets += [('generate.'+n,p) for n,p in model.languageSpace.generate_policy.named_parameters() if p.requires_grad]
        for term,rec in registry._terms.items():
            one=Error(row_mask=registry.row_mask);one._terms={term:rec};one._disabled=registry._disabled
            costs=registry_costs(one,reader_rows=getattr(model,'_sentence_reader_rows',None) if sentence else None)
            for owner,value in costs.items():
                grads=(torch.autograd.grad(value,[p for _,p in targets],retain_graph=True,allow_unused=True) if value.requires_grad else [None]*len(targets))
                report['terms'].append(dict(sentence=sentence,trial=getattr(model,'_sentence_trial',None),term=term,owner=owner,value=serial(value),gradients={n:None if g is None else float(g.to_dense().norm()) for (n,_),g in zip(targets,grads)}))
        return original(model,total_loss,*args,**kwargs)
    monkeypatch.setattr(BaseModel,'_backward_training_loss',backward)
    eligible=LanguageSpace.decoder_eligibility
    def eligibility(parent,lefts,rights,available,*args,**kwargs):
        value=eligible(parent,lefts,rights,available,*args,**kwargs)
        report['decoder'].append(dict(legal=value,counts=value.sum(-1),parent_norms=parent.norm(dim=-1)))
        return value
    monkeypatch.setattr(LanguageSpace,'decoder_eligibility',staticmethod(eligibility))
    assign=RowLayer.assign_row
    def allocation(layer,key,*args,**kwargs):
        old=dict(layer._tensor_row_keys)
        result=assign(layer,key,*args,**kwargs)
        if old!=layer._tensor_row_keys:
            frames=[]
            for f in inspect.stack()[1:12]:
                scalars={k:serial(v) for k,v in f.frame.f_locals.items() if (isinstance(v,(str,int,float,bytes)) and len(str(v))<160) or (torch.is_tensor(v) and v.numel()<=8)}
                frames.append(dict(function=f.function,file=f.filename,line=f.lineno,scalars=scalars))
            report['allocations'].append(dict(key=key,row=result,rows=layer._tensor_row_keys.copy(),frames=frames))
        return result
    monkeypatch.setattr(RowLayer,'assign_row',allocation)
    try:yield
    finally:
        p=H/'review20-diagnosis';p.mkdir(exist_ok=True)
        (p/(name.split('::')[-1]+'.json')).write_text(json.dumps(report,indent=2,default=serial)+'\n')
