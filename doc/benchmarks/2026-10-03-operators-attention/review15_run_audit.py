"""Observe the existing training; fixed-feature chooser reads draw no RNG."""
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
import torch
import os


def json_value(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def native_parameters(model):
    candidates = [model.perceptualSpace.subspace.what.W]
    candidates += [ws.subspace.what.W for ws in model.wholeSpaces]
    candidates += [cs._concept_allocator.layer().features.values for cs in model.conceptualSpaces]
    ids = {id(p) for p in candidates if torch.is_tensor(p) and p.requires_grad}
    return [(name,p) for name,p in model.named_parameters() if id(p) in ids]


@torch.no_grad()
def geometry(model, record):
    rows = sorted(set(record.primed.rows[record.primed.valid].detach().cpu().tolist()))
    root = record.root.detach().cpu()
    def metrics(values):
        unit = torch.nn.functional.normalize(values,dim=-1)
        cosines = unit@unit.T
        upper = torch.triu_indices(len(values), len(values), offset=1)
        mean = float(cosines[upper[0], upper[1]].mean()) if upper.shape[1] else None
        return dict(values=values,pairwise_cosines=cosines,mean_pairwise_cosine=mean,
                    centered_singular_values=torch.linalg.svdvals(values-values.mean(0,keepdim=True)))
    books=[]
    for cs in model.conceptualSpaces:
        cb=cs.similarity_codebook
        words=[r for r in rows if cs.word_surface_for_row(r) is not None]
        values=cb.lookup_rows(torch.tensor(words,device=cb.W.device)).detach().cpu()
        definitions=cb.mereology._definitions(words)
        evidence={label:[d for edges in definitions.values() for t,p,d in edges if t==tower]
                  for tower,label in ((0,'parts'),(1,'wholes'))}
        ranges={label:dict(count=len(v), minimum=min(v) if v else None,
                          maximum=max(v) if v else None, positive=sum(d>0 for d in v))
                for label,v in evidence.items()}
        forms=values[:,:cb.mereology.percept_width]
        books.append(dict(rows=words,**metrics(values),forms=metrics(forms),
                          d_definition='relu(e_for - e_against), summed net 11b evidence per native address; d > 0 selects a full-presence part',
                          net_evidence=ranges, support=cb.mereology.support_audit(words),
                          room=cb.mereology.room_report(model._review15_margin)))
    owner=model._concept_owner()
    inputs=[' '.join((owner.word_surface_for_row(int(row)) or b'').decode('utf8')
                     for row,valid in zip(rows,mask) if valid)
            for rows,mask in zip(record.word_rows.detach().cpu().tolist(),
                                 record.word_valid.detach().cpu().tolist())]
    order=['hello world','hello there','loving world','loving there']
    unit=torch.nn.functional.normalize(root,dim=-1)
    interaction=None
    if len(inputs)==4 and set(inputs)==set(order):
        hw,ht,lw,lt=[unit[inputs.index(text)] for text in order]
        interaction=float((hw-ht-lw+lt).norm())
    return dict(roots=metrics(root),root_inputs=inputs,codes=books,
                unit_root_xor_interaction=interaction,interaction_order=order)


@contextmanager
def observe_run():
    from Models import BasicModel, BaseModel
    from util import TheXMLConfig
    result=dict(reader_weights=[],sentence_gradients={},room={})
    trial,epoch,project,backward=BasicModel._reconstruct_trial,BasicModel.runEpoch,BasicModel._project_sentence_parameters,BaseModel._backward_training_loss
    def observed_trial(model, record):
        model._review15_margin=float(TheXMLConfig.space('ConceptualSpace','latticeMargin',0.))
        train=bool(getattr(model,'_sentence_training',False))
        if 'start' not in result: result['start']=geometry(model,record)
        if not train: result['end']=geometry(model,record)
        value=trial(model,record)
        if train and 'before_learning_readback' not in result:
            trace=model._last_decoder_trace
            result['before_learning_readback']=dict(
                texts=model._generated_word_text(trace[0],trace[1],bank=record.primed),
                truncated=trace[2])
        return value
    def observed_epoch(model,*args,**kwargs):
        value=epoch(model,*args,**kwargs)
        if kwargs.get('optimizer') is not None:
            named={id(p):name for name,p in model.named_parameters()}
            params=model.objective_parameter_groups(kwargs['optimizer'])['output']
            weights={named[id(p)]:float(p.detach().norm()) for p in params if p.ndim>=2}
            result['reader_weights'].append(dict(epoch=len(result['reader_weights'])+1,
                norm=sum(n*n for n in weights.values())**.5,parameters=weights))
        return value
    def observed_projection(model,*args,**kwargs):
        value=project(model,*args,**kwargs)
        reports=[getattr(cs.similarity_codebook.mereology,'last_room_projection',None) for cs in model.conceptualSpaces]
        if any(r is not None for r in reports):
            result['room'].setdefault('start', reports)
            result['room']['end']=reports
        return value
    def observed_backward(model,total,*args,**kwargs):
        if getattr(model,'_sentence_backward',False):
            named=native_parameters(model)
            pullback=getattr(model,'_sentence_pullback',None)
            parameters=[p for _,p in named]
            grads=(pullback.gradients(total,parameters) if pullback is not None else
                   torch.autograd.grad(total,parameters,retain_graph=True,allow_unused=True))
            for (name,p),g in zip(named,grads):
                row=result['sentence_gradients'].setdefault(name,dict(backwards=0,absent=0,zero=0,nonzero=0,maximum=0.))
                maximum=0. if g is None else float(g.detach().abs().max())
                row['absent']+=int(g is None); row['zero']+=int(g is not None and maximum==0)
                row['backwards']+=1; row['nonzero']+=int(maximum>0); row['maximum']=max(row['maximum'],maximum)
        return backward(model,total,*args,**kwargs)
    from review15_preference_probe import observe_preferences
    preferences=observe_preferences(result) if os.environ.get('OWNERSHIP_OBSERVER_OUTPUT') else nullcontext()
    with preferences, patch.object(BasicModel,'_reconstruct_trial',observed_trial), patch.object(BasicModel,'runEpoch',observed_epoch), patch.object(BasicModel,'_project_sentence_parameters',observed_projection), patch.object(BaseModel,'_backward_training_loss',observed_backward):
        yield result
