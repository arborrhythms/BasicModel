"""Observe the existing training; fixed-feature chooser reads draw no RNG."""
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
import torch
import os
import hashlib


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
def storage_snapshot(model):
    store=model.symbolSpace.ltm_store
    n=len(store)
    sentences=(store.rel_type[:n] != store.REL_DEF).nonzero().flatten()
    return dict(rows_used=n, capacity=store.capacity, sentence_rows=len(sentences),
        definition_rows=int((store.rel_type[:n] == store.REL_DEF).sum()),
        addresses=store.address_keys[sentences].detach().cpu().tolist(),
        sentence_indices=store.sentence_index[sentences].detach().cpu().tolist(),
        content_keys=[store.content_key(int(row)).hex() for row in sentences],
        witness_counts=store.witness_count[sentences].detach().cpu().tolist(),
        timestamps=store.timestamp[sentences].detach().cpu().tolist(),
        all_witness_counts=store.witness_count[:n].detach().cpu().tolist())


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
                          containment=cb.mereology.containment_audit(), room=cb.mereology.room_report(model._review17_margin)))
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
    fit = None
    if len(inputs)==4 and set(inputs)==set(order):
        target = root.new_tensor([dict(zip(order,(0.,1.,1.,0.)))[text] for text in inputs])
        fit = {}
        for name, values in (('roots', root), ('unit_roots', unit),
                             ('reader_features', record.reader_features().detach().cpu())):
            design = torch.cat((values.double(), torch.ones(len(values),1,dtype=torch.float64)), -1)
            prediction = design @ (torch.linalg.pinv(design) @ target.double())
            linear_weights = torch.linalg.pinv(values.double()) @ target.double()
            fit[name] = dict(affine_rank=int(torch.linalg.matrix_rank(design)),
                             optimal_affine_mse=float((prediction-target).square().mean()),
                             linear_rank=int(torch.linalg.matrix_rank(values.double())),
                             optimal_linear_mse=float((values.double()@linear_weights-target).square().mean()),
                             minimum_linear_weight_norm=float(linear_weights.norm()))
    identity = getattr(getattr(model.perceptualSpace, 'percept_store', None), 'identity', None)
    identity_audit = None if identity is None else dict(**identity.audit(),
        projection_sha256=hashlib.sha256(identity.projection.cpu().numpy().tobytes()).hexdigest(),
        rung0=getattr(model, '_last_rung0_identity_audit', None))
    return dict(storage=storage_snapshot(model), roots=metrics(root),root_inputs=inputs,codes=books,
                unit_root_xor_interaction=interaction,interaction_order=order,
                affine_fit=fit,identity=identity_audit)


@contextmanager
def observe_run():
    from Models import BasicModel, BaseModel
    from util import TheXMLConfig
    result=dict(reader_weights=[],comparison_reader_weights=[],reader_mse=[],
                reader_training_rows=[],sentence_gradients={},room={})
    commit = BasicModel._commit_sentence
    result['sentence_trials'] = []
    def observed_commit(model,state,sid,active,*args):
        audit=getattr(model,'_last_sentence_credit',None)
        if audit is not None:
            result['sentence_trials'].append(dict(epoch=result.get('_active_epoch',0),
                training=model._sentence_training, **audit))
            if model._sentence_training:
                reads=audit['readers']
                if len(reads)==2 and all(r is not None for r in reads):
                    row=dict(epoch=result.get('_active_epoch',0),sentence=sid,
                             timing='before either trial update, selected by reconstruction keep')
                    for reader in ('presented','comparison'):
                        first,second=reads[0][reader],reads[1][reader]
                        selected=torch.where(audit['wins'],second['mse'],first['mse'])
                        gate=audit['wins'].reshape((-1,)+(1,)*(first['predictions'].ndim-1))
                        prediction=torch.where(gate,second['predictions'],first['predictions'])
                        row[reader]=dict(mse=float(selected[active].mean()),rows=selected.detach().tolist(),
                                         predictions=prediction.detach().tolist())
                    result['reader_mse'].append(row)
                from derivation_probe import compose_derivations
                final=[]
                for row in compose_derivations(model,state,sid,active):
                    b=row['batch_row']; kept=int(audit['wins'][b])
                    row.update(epoch=result.get('_active_epoch',0),
                        trial='explore' if kept else 'greedy',
                        cost_components=dict(zip(('reconstruction','expectation','answer'),
                            audit['components'][b,kept].tolist())),
                        trial_components=audit['components'][b].tolist(),
                        selected_cost=float(audit['keep_costs'][b,kept]), credit_total=float(audit['costs'][b,kept]), decision=audit['keep_decision'][b], advantage_sign=float(audit['advantage_sign'][b]), answer_against_keep=bool(audit['answer_against_keep'][b]))
                    final.append(row)
                result['final_committed_training_operators']=final
        value=commit(model,state,sid,active,*args)
        result['closing_image']=getattr(model,'_last_closing_image_audit',{})
        result['storage']=storage_snapshot(model)
        return value
    trial,epoch,project,backward=BasicModel._reconstruct_trial,BasicModel.runEpoch,BasicModel._project_sentence_parameters,BaseModel._backward_training_loss
    def observed_trial(model, record):
        model._review17_margin=float(TheXMLConfig.space('ConceptualSpace','latticeMargin',0.))
        train=bool(getattr(model,'_sentence_training',False))
        if 'start' not in result: result['start']=geometry(model,record)
        if not train: result['end']=geometry(model,record)
        value=trial(model,record)
        audit = getattr(model, '_last_rung0_identity_audit', None)
        if audit is not None:
            totals = result.setdefault('rung0_audit', dict(trials=0, identified_reads=0, errors=0))
            totals['trials'] += 1
            totals['identified_reads'] += audit['identified_words']
            totals['errors'] += audit['errors']
        if train and 'before_learning_readback' not in result:
            trace=model._last_decoder_trace
            result['before_learning_readback']=dict(
                texts=model._generated_word_text(trace[0],trace[1],bank=record.primed),
                truncated=trace[2])
        return value
    def observed_epoch(model,*args,**kwargs):
        result['_active_epoch']=len(result['reader_weights'])+1
        value=epoch(model,*args,**kwargs)
        key='storage_after_training' if kwargs.get('optimizer') is not None else 'storage_after_evaluation'
        result[key]=storage_snapshot(model)
        if kwargs.get('optimizer') is not None:
            named={id(p):name for name,p in model.named_parameters()}
            params=model.objective_parameter_groups(kwargs['optimizer'])['output']
            comparison={id(p) for p in model.comparison_reader.parameters()} if model.comparison_reader is not None else set()
            for judge,key in ((False,'reader_weights'),(True,'comparison_reader_weights')):
                selected=[p for p in params if (id(p) in comparison)==judge]
                weights={named[id(p)]:float(p.detach().norm()) for p in selected if p.ndim>=2}
                stepped={named[id(p)] for p in selected if p in kwargs['optimizer'].state
                         and 'step' in kwargs['optimizer'].state[p]}
                result[key].append(dict(epoch=len(result[key])+1,
                    norm=sum(n*n for n in weights.values())**.5,parameters=weights,
                    active_norm=sum(n*n for name,n in weights.items() if name in stepped)**.5,
                    reader_updates=(model._sentence_comparison_reader_updates if judge else model._sentence_reader_updates),
                    optimizer_steps={named[id(p)]:float(kwargs['optimizer'].state[p]['step']) for p in selected
                        if p in kwargs['optimizer'].state and 'step' in kwargs['optimizer'].state[p]}))
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
            if model._sentence_reader_costs:
                result['reader_training_rows'].append(dict(epoch=result.get('_active_epoch',0), trial=model._sentence_trial,
                    rows=model._sentence_reader_rows.detach().tolist(), weights=model._sentence_reader_weights.detach().tolist(),
                    comparison_weights=model._sentence_comparison_reader_weights.detach().tolist()))
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
    from round2_score_probe import observe_score_function
    from review17_decomposition_probe import observe_decomposition
    preferences=observe_score_function(result)
    from handoff_probe import observe_handoff
    with observe_handoff(result), preferences, patch.object(BasicModel,'_commit_sentence',observed_commit), observe_decomposition(result), patch.object(BasicModel,'_reconstruct_trial',observed_trial), patch.object(BasicModel,'runEpoch',observed_epoch), patch.object(BasicModel,'_project_sentence_parameters',observed_projection), patch.object(BaseModel,'_backward_training_loss',observed_backward):
        try:
            yield result
        finally:
            result.pop('_active_epoch',None)
