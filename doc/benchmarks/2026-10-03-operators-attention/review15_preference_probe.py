"""Fixed-candidate compose logits around the actual preference-owned step.

Observation only: score the saved candidate features with the chooser, without
recomposing, resampling, or writing parameter/gradient buffers.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch


def frozen(value):
    if isinstance(value,torch.nn.Parameter): return value
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, dict): return {k:frozen(v) for k,v in value.items()}
    if isinstance(value, tuple): return tuple(frozen(v) for v in value)
    if isinstance(value, list): return [frozen(v) for v in value]
    return value


@contextmanager
def observe_preferences(result):
    from Language import OperationSelectionLayer, MLPTransformChooser, AnchorDotTransformChooser
    from Models import BasicModel
    methods={(cls,name):getattr(cls,name) for cls in (MLPTransformChooser,AnchorDotTransformChooser)
             for name in ('score_binary','score_unary')}
    forward,preference,train=(OperationSelectionLayer.forward,
        BasicModel._compose_preference_loss,BasicModel._sentence_train_step)
    current=[]; pending=[]; selected=[]
    records=result.setdefault('compose_preference_steps',[])
    def scorer(cls,name):
        def observed(chooser,*args,**kwargs):
            value=methods[cls,name](chooser,*args,**kwargs)
            if current and current[-1]['module'].chooser is chooser:
                current[-1][name]=(frozen(args),frozen(kwargs))
                current[-1][name+'_initial']=frozen(value)
            return value
        return observed
    def operation(module,x,**kwargs):
        masked=kwargs.get('masked_action')
        capture=torch.is_tensor(masked) and bool((masked>=0).any())
        if not capture:return forward(module,x,**kwargs)
        entry=dict(module=module,forced=masked.detach().ge(0),
                   depth=frozen(kwargs.get('depth')),width=x.shape[1])
        current.append(entry)
        try:value=forward(module,x,**kwargs)
        finally:current.pop()
        route=value[2]
        entry.update(logits=route['logits'],actions=route['action'].detach(),
                     original=route['logits'].detach().clone())
        pending.append(entry)
        return value
    @torch.no_grad()
    def scores(entry):
        chooser=entry['module'].chooser
        bargs,bkwargs=entry['score_binary'];uargs,ukwargs=entry['score_unary']
        stop,binary=methods[type(chooser),'score_binary'](chooser,*bargs,**bkwargs)
        _,unary=methods[type(chooser),'score_unary'](chooser,*uargs,**ukwargs)
        B=stop.shape[0];depth=entry['depth'];N=entry['width']
        if depth is None:depth=torch.full((B,),N,device=stop.device)
        live=torch.arange(N,device=stop.device)[None]<depth[:,None]
        def joined(stop,binary,unary):
            stop=(stop.squeeze(-1)*live).sum(-1)/depth.clamp_min(1)
            return torch.cat((binary.reshape(B,-1),unary.reshape(B,-1),stop[:,None]),-1)
        old_stop,old_binary=entry['score_binary_initial']
        _,old_unary=entry['score_unary_initial']
        delta=joined(stop,binary,unary)-joined(old_stop,old_binary,old_unary)
        return entry['original']+delta
    def compared(model,path,wins):
        value=preference(model,path,wins)
        selected.clear()
        for entry in pending:
            entry['wins']=wins.detach().clone()
            entry['selected']=entry['forced'] & wins
            entry['gradient']=(torch.autograd.grad(value,entry['logits'],retain_graph=True,
                allow_unused=True)[0] if value.requires_grad else None)
            selected.append(entry)
        pending.clear()
        return value
    def stepped(model,loss):
        if model._sentence_trial!='explore':return train(model,loss)
        rng=torch.random.get_rng_state()
        for entry in selected:entry['before']=scores(entry)
        assert torch.equal(rng,torch.random.get_rng_state())
        value=train(model,loss)
        rng=torch.random.get_rng_state()
        for entry in selected:
            after=scores(entry);before=entry['before'];delta=after-before
            legal=torch.isfinite(entry['original'])
            records.append(dict(step=len(records),selected=entry['selected'],wins=entry['wins'],
                actions=entry['actions'],legal=legal,preference_logit_gradient=entry['gradient'],
                action_names=((entry['module'].op_names or [])*(entry['width']-1)
                              +(entry['module'].unary_names or [])*entry['width']+['STOP']),
                fixed_candidate_logits_before=torch.where(legal,before,0.),
                fixed_candidate_logits_after=torch.where(legal,after,0.),
                fixed_candidate_logit_change=torch.where(legal,delta,0.),
                interpretation='Chooser scores on the same saved operand/candidate features; priors and eligibility fixed.'))
        assert torch.equal(rng,torch.random.get_rng_state())
        selected.clear()
        return value
    with ExitStack() as stack:
        for cls,name in methods:stack.enter_context(patch.object(cls,name,scorer(cls,name)))
        stack.enter_context(patch.object(OperationSelectionLayer,'forward',operation))
        stack.enter_context(patch.object(BasicModel,'_compose_preference_loss',compared))
        stack.enter_context(patch.object(BasicModel,'_sentence_train_step',stepped))
        yield
