"""One declared, matched native identity experiment; never retries a failed run.

The text and grading labels are separated before presentation. The only native
ablation is IndependentComponents.commit(require_selected_mint=False). All
branches restore the same staged warm-up model and RNG checkpoint, then use
fresh optimizers at the same learning rate.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from identity_corpus import ROOT, DEST, KINDS, PROPERTIES, VERBS
from identity_measurement import load_split, native_observer, grade_capture, score_document, summarize
from bounded_tests import source_snapshot


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identification(dictionary, value, atoms, *, source_limit):
    """Count present atoms explained by ACTIVE columns, not nearest columns."""
    matrix = dictionary.matrix().detach().to(value)
    codes = dictionary.encode(value, max_sources=source_limit).codes.detach()
    active = codes.ne(0)
    similarities = (F.normalize(atoms, dim=-1) @ matrix.T).abs()
    match = (similarities >= .9) & atoms.norm(dim=-1).gt(0)[:, None]
    hits = (match & active[None]).any(-1)
    return dict(hits=hits.tolist(), active_columns=[cid for cid, live in zip(dictionary.ids, active) if bool(live)],
                total=len(atoms), correct=int(hits.sum()))


def noun_frame(roles):
    """Historical order-two recovery measurement, not individual identification."""
    if roles.shape[-2] != 3:
        raise ValueError('a recovery frame requires three role slots')
    return roles[..., 0, :] + roles[..., 2, :]


def recovery_value(chart, meaning, clause):
    """Score the completed row used by component cost/admission.

    A fused idea contains its whole sentence, including a predicate that is
    absent from the pre-fusion NP pair. Relative clauses keep that NP pair.
    """
    value = (chart.project(clause.point.detach())
             if clause is not None and clause.point is not None
             else noun_frame(chart.project(meaning.roles.detach())))
    return chart.expand(value)


def recovery_data():
    """Support 1/2/3; hold out a balanced quarter of the factorial combinations.

    Confounded training has identical documents/batches/support/verb budgets,
    but each kind always co-occurs with its fixed property. Labels stay here.
    """
    streams = {name: [] for name in ('factorial', 'confounded', 'heldout')}
    labels = {}
    for count, start, prop, verb_i in product((1, 2, 3), range(4), range(4), range(4)):
        held = (start + prop + verb_i) % 4 == 0
        kinds = [KINDS[(start+i) % 4] for i in range(count)]
        properties = [PROPERTIES[(prop+i) % 4] for i in range(count)]
        for name in (('heldout',) if held else ('factorial', 'confounded')):
            props = ([PROPERTIES[KINDS.index(k)] for k in kinds] if name == 'confounded' else properties)
            det = 'the' if held else 'a'
            text = ' and '.join(f'{det} {p} {k}' for p,k in zip(props, kinds)) + f' {VERBS[verb_i]} .'
            key = hashlib.sha256(f'identity-experiment:{name}:{count}:{start}:{prop}:{verb_i}'.encode()).hexdigest()[:20]
            streams[name].append(dict(id=key, sentences=[text]))
            atoms = kinds + props + [VERBS[verb_i]]
            labels[key] = dict(atoms=atoms, kinds=kinds, properties=props, support=count,
                               positions=[text.split().index(a) for a in atoms])
    return streams, labels


def reset(model, batch_size=4):
    model.flush_word_buffers()
    model.dispatch_packed_soft_reset([True]*batch_size)
    model.dispatch_per_row_reset([True]*batch_size)
    model.dispatch_soft_reset()
    model.post_tick_compact()


def make_model(folder, patch):
    import Language, Models, util
    patch.setattr(util,'TheCompileBackend','none')
    # Preserve the production distinction between the 104-coordinate form
    # event and its meaning complement. The word-chunk parity fixture makes
    # them equal and therefore cannot measure independent components.
    tree=ET.parse(ROOT/'data/BasicModel.xml');root=tree.getroot()
    settings={
        'InputSpace/nOutput':128,'InputSpace/nDim':16,
        'PartSpace/nInput':128,'PartSpace/nInputDim':16,'PartSpace/nVectors':256,
        'ConceptualSpace/nInputDim':232,'ConceptualSpace/nVectors':4096,
        'ConceptualSpace/activeVectors':2048,'ConceptualSpace/nDim':232,
        'ConceptualSpace/nOutputDim':232,'ConceptualSpace/meaningWidth':128,
        'WholeSpace/nInputDim':232,'OutputSpace/nInputDim':232,
        'SymbolSpace/ltmCapacity':8192,
        'architecture/serialWordCapacity':64,'architecture/serialWordBuckets':64,
        'architecture/weightsPath':folder/'unused.ckpt',
        'architecture/training/batchSize':4,'architecture/training/numWorkers':0,
        'architecture/training/branchDiagnosticsEvery':0,
        'architecture/training/seed':20261010,
        'architecture/training/autoload':False,'architecture/training/autosave':False,
        'architecture/training/checkpointEveryBatches':0,
        'architecture/training/intraLossWeight':0.,
        'architecture/training/reconstructionPlacement':'eager'}
    for name,value in settings.items():
        node=root.find('./'+name)
        if node is None:raise RuntimeError(f'missing native configuration field: {name}')
        node.text=str(value).lower() if isinstance(value,bool) else str(value)
    config=folder/'identity_model.xml';tree.write(config,encoding='unicode')
    util.init_device('cpu')
    util.init_config(path=str(config),defaults_path=str(ROOT/'data/model.xml'))
    Language.TheGrammar._configured=False
    patch.setattr(Models.BaseModel,'load_weights',lambda self,*a,**k:False)
    model,_=Models.BasicModel.from_config(str(config),data=Models.TheData)
    model.train()
    derived=model._concept_owner().similarity_codebook.mereology
    if derived.meaning_pairs!=64:
        raise RuntimeError(f'identity measurement needs 64 meaning pairs, got {derived.meaning_pairs}')
    model._tensor_peer_while_eager = True
    model.checkpoint_every_batches = 0
    model.inputSpace.data.has_supervised_outputs = False
    # A read-only audit retains the first unresolved sym cue encountered by
    # native retrieval. It supplies no addresses or scores to the model.
    from MemoryIndex import LeafCodeIndex
    from Taxonomy import concept_reference
    cued_rows = LeafCodeIndex.cued_rows
    captured = False
    def audit_cue(store, cue, **kwargs):
        nonlocal captured
        if not captured:
            for reference in cue.role_refs:
                if not (isinstance(reference, tuple) and reference and reference[0] == 'sym'):
                    continue
                try:
                    concept_reference(reference)
                except TypeError:
                    captured = True
                    save(folder/'unresolved-cue.json', dict(reference=reference,
                        meaning=cue.metadata(), stack=traceback.format_stack()))
                    torch.save(dict(roles=cue.roles.detach().cpu(),
                        role_mask=cue.role_mask.cpu(), metadata=cue.metadata()), folder/'unresolved-cue.pt')
                    break
        return cued_rows(store, cue, **kwargs)
    patch.setattr(LeafCodeIndex, 'cued_rows', audit_cue)
    return model


def component_signal(model):
    """Observe whether ordinary text reaches the native meaning coordinates."""
    component=model._concept_owner().components
    component.begin_forward();chart=component._ready()
    population=component._population
    return dict(meaning_pairs=chart.meaning_pairs,chart_rows=len(chart.rows),
        nonzero_basis_rows=int(chart.basis.norm(dim=-1).gt(0).sum()),
        observation_rows=len(population),
        nonzero_observation_rows=int(population.norm(dim=-1).gt(0).sum()))


def model_hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k,v in model.state_dict().items() if torch.is_tensor(v)}


def restore_warmup(model, path, restore):
    """Use the native metadata loader, then restore the exact frozen tensors.

    The existing loader resynchronizes the shared concept codebook. Record
    that round-trip drift; it must not alter the matched experiment's start.
    This measurement adapter does not repair or certify checkpoint retention.
    """
    bundle=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    saved=bundle['state_dict']
    catalog=bundle.get('grammar_catalog')
    if catalog is None and any(key.endswith('._next_ts') for key in saved):
        # The two retained 4e checkpoints use this archived complete grammar.
        # This is a one-way storage migration, never a runnable old path.
        catalog=json.loads((ROOT/'test/fixtures/grammar-catalog-63ece2e1.json').read_text())
    # BaseModel's preflight runs before this lazy child's own load hook.
    # Materialize its saved bank through that hook, retaining strict key
    # checking and every learned weight. This presents no training example.
    prefix='comparison_reader.'
    comparison={key[len(prefix):]:value for key,value in saved.items() if key.startswith(prefix)}
    if comparison:
        model.comparison_reader.load_state_dict(comparison,strict=True)
    if not restore(model,str(path),require_match=True,grammar_catalog=catalog):
        raise RuntimeError('warm-up checkpoint restore failed')
    from GrammarCheckpoint import align
    saved,drift,_=align(model,saved,catalog)
    for name,module in model.named_modules(remove_duplicate=False):
        migrate=getattr(type(module),'migrate_generate_checkpoint',None)
        if migrate is not None:migrate(module,saved,name+'.')
    current=model.state_dict()
    with torch.no_grad():
        for key,value in saved.items():
            if key.endswith('parts_of_speech._extra_state'):
                # Equal strings can have different pickle sharing after a
                # fresh load, changing the byte envelope's size. Validate
                # the native restored metadata, never its serialized size.
                from IndependentComponents import _unpack_state
                if key not in current or _unpack_state(current[key]) != _unpack_state(value):
                    raise RuntimeError(f'occupation state differs after native restore: {key}')
                continue
            if key not in current or current[key].shape!=value.shape:
                raise RuntimeError(f'checkpoint shape differs: {key}')
            if not torch.equal(current[key],value):
                drift[key]=dict(elements=int(current[key].ne(value).sum()),
                    max_difference=float((current[key].double()-value.double()).abs().max()))
                if key.endswith('._extra_state'):
                    model.get_submodule(key[:-len('._extra_state')]).set_extra_state(value)
                else:
                    current[key].copy_(value)
    final=model.state_dict()
    for key,value in saved.items():
        if key.endswith('._extra_state'):
            # torch.save's byte envelope is not a canonical metadata encoding.
            from IndependentComponents import _unpack_state
            actual,expected=_unpack_state(final[key]),_unpack_state(value)
            if key.endswith('parts_of_speech._extra_state'):
                if actual != expected:
                    raise RuntimeError(f'occupation state differs after restore: {key}')
                continue
            if key.endswith('nouns._extra_state') and expected.get('pending') == [] and 'pending' not in actual:
                expected=dict(expected);expected.pop('pending')
                drift.setdefault(key,{})['retired_empty_pending']=True
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            continue
        if not torch.equal(final[key],value):
            raise RuntimeError(f'checkpoint retention differs after restore: {key}')
    return drift


def warmup(folder, order_steps, sentence_steps):
    from attention_curriculum_gates import present, identities, STAGES, PERCEPTS, INTERRUPTION, NOVEL
    folder.mkdir(parents=True, exist_ok=True)
    reports=[]
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=make_model(folder,patch)
        optimizer=model.getOptimizer(lr=.003)
        try:
            for label, fields in [('first',PERCEPTS),('repeat',PERCEPTS),('interruption',INTERRUPTION),
                                  ('after_gap',PERCEPTS),('known',PERCEPTS),('novel_first',NOVEL),('novel_second',NOVEL)]:
                report=present(model,dict(fields=fields))
                reports.append(dict(stage=1 if label in ('first','repeat','interruption','after_gap') else 2,
                    name=label, identities=identities(model,fields), **report))
                save(folder/'results.json',reports)
            # Lexical recognition is shared before the branch split; these
            # single-word presentations do not force any noun column or parse.
            atoms=list((*KINDS,*PROPERTIES,*VERBS))
            for repeat in range(4):
                for start in range(0,len(atoms),4):
                    report=present(model,dict(fields=atoms[start:start+4]))
                    reports.append(dict(stage=2,name=f'lexical:{repeat}:{start}',**report))
            save(folder/'results.json', reports)
            signal=component_signal(model)
            save(folder/'coordinate-signal.json',signal)
            if not signal['nonzero_basis_rows'] or not signal['nonzero_observation_rows']:
                raise RuntimeError('the lexical warm-up supplied no nonzero component meaning signal')
            model.save_weights(str(folder/'last-completed.ckpt'))
            for stage in STAGES:
                count=order_steps if stage['stage']==3 else sentence_steps
                initial=present(model,stage); ce=[]
                with (folder/(stage['name']+'.jsonl')).open('w') as log:
                    for step in range(count):
                        report=present(model,stage,train=True,optimizer=optimizer)
                        ce.append(report.get('lesson_cross_entropy'))
                        log.write(json.dumps(dict(step=step+1,**report))+'\n');log.flush()
                        save(folder/'progress.json',dict(stage=stage['name'],step=step+1,budget=count))
                final=present(model,stage)
                reports.append(dict(stage=stage['stage'],name=stage['name'],initial=initial,final=final,
                                    presentations=count,lesson_cross_entropy=ce))
                save(folder/'results.json',reports)
                model.save_weights(str(folder/'last-completed.ckpt'))
                print(json.dumps(dict(stage=stage['name'],done=count,readback=final['exact_readbacks'],ce=ce[-1])),flush=True)
                if stage['stage']==4:
                    # Supplied answers name ordinary output words, never an
                    # operator or an attention action. Failed prerequisites
                    # are retained, not used to claim this later gate passed.
                    taught=dict(fields=stage['fields'],prompts=stage['questions'],targets=stage['answers'])
                    for step in range(sentence_steps):
                        present(model,taught,train=True,optimizer=optimizer)
                    final=present(model,taught)
                    reports.append(dict(stage=4,name=stage['name']+'_answer',presentations=sentence_steps,final=final))
                    save(folder/'results.json',reports)
                    model.save_weights(str(folder/'last-completed.ckpt'))
            reset(model)
            model.save_weights(str(folder/'warmup.ckpt'))
            save(folder/'state-hashes.json',model_hashes(model))
            save(folder/'checkpoint.json',dict(sha256=digest(folder/'warmup.ckpt'),
                note='Fixed-budget warm-up, not a certificate that stages 3/4 passed; no fresh disabled training control.'))
        except Exception:
            save(folder/'failure.json',dict(error=traceback.format_exc()))
            model.save_weights(str(folder/'failed-state.ckpt'))
            raise
        finally:model.End()


@contextmanager
def admission_ablation(model, off):
    component=model._concept_owner().components
    commit=component.commit
    counts=Counter()
    def observed(row,meaning,prediction=None,*,individual_references=()):
        counts['committed_rows']+=1
        counts['selected_individual']+=bool(individual_references)
        counts['selected_mint']+=(-1 in individual_references)
        return commit(row,meaning,prediction,individual_references=individual_references,
                      require_selected_mint=not off)
    component.commit=observed
    try:yield counts
    finally:component.commit=commit


@contextmanager
def observed_rows(model):
    """Read chosen meanings and actual bank IDs; no grading data enters calls."""
    saved={}; banks={}; observation=model._sentence_observation; bank=model._sentence_reference_bank
    def observe(state,sid,active,**kw):
        result=observation(state,sid,active,**kw)
        if kw.get('admit'):
            component=model._concept_owner().components
            chart=component._ready()
            words=model._tensor_pushed_ideas.detach()
            saved[int(sid)]=[dict(value=None if meaning is None else recovery_value(chart,meaning,result['clauses'][b]).clone(),
                words=chart.expand(chart.project(words[b])).clone(),
                source_limit=component.source_limit(result['clauses'][b]))
                for b,meaning in enumerate(result['meanings'])]
        return result
    def read_bank(active,like):
        value=bank(active,like)
        banks[int(model._open_sentence_slot)]=[
            value.ids[b][value.valid[b]].detach().tolist() for b in range(len(active))]
        return value
    model._sentence_observation=observe; model._sentence_reference_bank=read_bank
    try:yield saved,banks
    finally:model._sentence_observation=observation;model._sentence_reference_bank=bank


def present_documents(model, docs, *, optimizer=None, labels=None, recovery_labels=None, log=None):
    from What import What
    training=optimizer is not None; results=[]; evidence=[]; atom_examples={}
    for offset in range(0,len(docs),4):
        batch=docs[offset:offset+4]; B=len(batch); addresses=[]; indices=[]
        split='train' if training else 'validation'
        for doc in batch:
            indices.append([])
            for sid,_ in enumerate(doc['sentences']):
                indices[-1].append(len(addresses))
                addresses.append(dict(document=doc['id'],sentence=sid,document_key=doc['id'],sentence_index=sid,split=split))
        data=model.inputSpace.data;data.has_supervised_outputs=False;data.source_addresses={split:addresses}
        with native_observer(model) as capture, observed_rows(model) as (rows,banks), torch.set_grad_enabled(training):
            inputs=model.inputSpace.prepPackedInput([d['sentences'] for d in batch])
            model.runBatch(train=training,optimizer=optimizer,batchNum=offset//4,batchSize=B,split=split,
                batch_override=(inputs,torch.empty(B,0)),source_rows=indices,
                questions=tuple(What.present(b,split=split) for b in range(B)))
        if labels:
            for b,doc in enumerate(batch):
                label=labels[doc['id']];obs=grade_capture(label,capture,b)
                result=score_document(label,obs);probe=label['probe']
                if probe:
                    mention=next(m for m in label['mentions'] if m['id']==probe['mention'])
                    bank_ids=banks.get(mention['sentence'],[[]]*B)[b]
                    result['native_bank']=dict(ids=bank_ids,
                        candidates_in_bank=[obs['references'].get(mid) in bank_ids for mid in probe['candidates']])
                results.append(result);evidence.append(dict(id=doc['id'],**obs))
        if recovery_labels:
            for b,doc in enumerate(batch):
                label=recovery_labels[doc['id']]; observed=rows.get(0,[None]*B)[b]
                component=model._concept_owner().components
                if observed is None or observed['value'] is None:
                    metric=dict(hits=[False]*len(label['atoms']),active_columns=[],total=len(label['atoms']),correct=0,unavailable=True)
                else:
                    atoms=observed['words'][label['positions']]
                    metric=identification(component.nouns,observed['value'],atoms,source_limit=observed['source_limit'])
                    metric['unavailable']=False
                    for name,atom in zip(label['atoms'],atoms):atom_examples.setdefault(name,atom.detach())
                results.append(dict(id=doc['id'],**label,**metric))
        reset(model,B)
        if log:
            save(log/'progress.json',dict(training=training,completed=offset+B,total=len(docs)))
            save(log/'partial-results.json',results)
        print(json.dumps(dict(training=training,completed=offset+B,total=len(docs))),flush=True)
    return results,evidence,atom_examples


def recovery_summary(rows, component, atoms):
    total=sum(r['total'] for r in rows); correct=sum(r['correct'] for r in rows)
    by_atom={}
    for name in (*KINDS,*PROPERTIES,*VERBS):
        hits=[h for r in rows for a,h in zip(r['atoms'],r['hits']) if a==name]
        by_atom[name]=dict(correct=sum(hits),total=len(hits),identification=sum(hits)/max(1,len(hits)))
    merges=[]; matrix=component.nouns.matrix().detach()
    names=[a for a in (*KINDS,*PROPERTIES,*VERBS) if a in atoms]
    atom_matrix=F.normalize(torch.stack([atoms[a] for a in names]),dim=-1) if names else matrix[:0]
    clean=(atom_matrix@matrix.T).abs().ge(.9).any(0) if names else torch.zeros(len(matrix),dtype=torch.bool)
    for kind,prop in zip(KINDS,PROPERTIES):
        if kind not in atoms or prop not in atoms:continue
        merged=F.normalize(F.normalize(atoms[kind],dim=-1)+F.normalize(atoms[prop],dim=-1),dim=-1)
        matched=(matrix@merged).abs().ge(.9)&~clean
        merges.append(dict(kind=kind,property=prop,columns=[cid for cid,hit in zip(component.nouns.ids,matched) if bool(hit)]))
    return dict(correct=correct,total=total,identification=correct/max(1,total),by_atom=by_atom,
        unavailable_rows=sum(r['unavailable'] for r in rows),columns=len(matrix),clean_columns=int(clean.sum()),merges=merges,
        merged_pairs=sum(bool(m['columns']) for m in merges),pending=len(component.nouns._pending))


def branch(folder, warm, name):
    import Models
    restore=Models.BaseModel.load_weights
    folder.mkdir(parents=True,exist_ok=True)
    off=name.endswith('-off'); category,corpus,_=name.split('-')
    with pytest.MonkeyPatch.context() as patch,warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=make_model(folder,patch)
        try:
            drift=restore_warmup(model,warm/'warmup.ckpt',restore)
            expected=json.loads((warm/'state-hashes.json').read_text());actual=model_hashes(model)
            mismatch=[k for k in expected if expected[k]!=actual.get(k)]
            save(folder/'start.json',dict(checkpoint_sha256=digest(warm/'warmup.ckpt'),gate='off' if off else 'on',
                native_loader_drift=drift,exact_start_tensor_restoration=True,
                differing_checkpoint_tensors=mismatch,extra_tensors=sorted(set(actual)-set(expected))))
            if mismatch:raise RuntimeError(f'checkpoint tensors differ: {mismatch[:8]}')
            signal=component_signal(model)
            save(folder/'coordinate-signal.json',signal)
            if not signal['nonzero_basis_rows'] or not signal['nonzero_observation_rows']:
                raise RuntimeError('the restored warm-up supplied no nonzero component meaning signal')
            optimizer=model.getOptimizer(lr=.003)
            component=model._concept_owner().components; before=set(component.nouns.ids)
            with admission_ablation(model,off) as admission:
                if category=='recovery':
                    streams,labels=recovery_data()
                    present_documents(model,streams[corpus],optimizer=optimizer,log=folder)
                    minted=set(component.nouns.ids)-before
                    model._concept_owner()._online_learning_frozen=True
                    rows,_,atoms=present_documents(model,streams['heldout'],recovery_labels=labels,log=folder)
                    report=recovery_summary(rows,component,atoms)
                    report.update(rows=rows,training_documents=len(streams[corpus]),heldout_documents=len(streams['heldout']))
                else:
                    train,train_labels=load_split('train'); biased,_=load_split('biased_train')
                    shared=[d for d in train if train_labels[d['id']]['family']=='prediction_vocabulary']
                    pronouns=[d for d in train if train_labels[d['id']]['family']=='pronoun'] if corpus=='balanced' else biased*2
                    determiner=[d for d in train if train_labels[d['id']]['family'] in ('determiner','same_kind')]
                    # Both branches receive 16 vocabulary, 32 pronoun and 88
                    # determiner/same-kind presentations, in this order.
                    lessons=shared+pronouns+determiner
                    present_documents(model,lessons,optimizer=optimizer,log=folder)
                    minted=set(component.nouns.ids)-before
                    model._concept_owner()._online_learning_frozen=True
                    docs,labels=load_split('eval');rev,revlabels=load_split('reversed_eval');labels.update(revlabels)
                    rows,evidence,_=present_documents(model,docs+rev,labels=labels,log=folder)
                    report=dict(summary=summarize(rows),results=rows,training_documents=len(lessons),evaluation_documents=len(docs+rev))
                    save(folder/'observations.json',evidence)
                report.update(branch=name,gate='off' if off else 'on',admission=dict(admission),columns_minted=len(minted),
                    columns=len(component.nouns.ids),admission_drops=component.admission_drops,completed=True)
                save(folder/'results.json',report)
            model.save_weights(str(folder/'final.ckpt'))
        except Exception:
            save(folder/'failure.json',dict(error=traceback.format_exc(),completed=False))
            model.save_weights(str(folder/'failed-state.ckpt'))
            raise
        finally:model.End()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    parser.add_argument('--order-steps',type=int,default=64);parser.add_argument('--sentence-steps',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--branch');args=parser.parse_args()
    if min(args.order_steps, args.sentence_steps, args.workers) < 1:
        parser.error('presentation counts and worker count must be positive')
    args.out=args.out.resolve();torch.set_num_threads(1)
    if args.branch:
        branch(args.out/args.branch,args.out/'warmup',args.branch);return
    args.out.mkdir(parents=True,exist_ok=False)
    source=source_snapshot(ROOT);corpus={p.name:digest(p) for p in DEST.iterdir() if p.is_file()}
    branches=[f'recovery-{c}-{g}' for c in ('factorial','confounded') for g in ('on','off')]+[
        f'identity-{c}-{g}' for c in ('balanced','biased') for g in ('on','off')]
    streams,labels=recovery_data()
    for name,docs in streams.items():save(args.out/'corpus'/(name+'.text.json'),docs)
    save(args.out/'corpus'/'labels.json',labels)
    save(args.out/'plan.json',dict(seed=20261010,retries=0,order_presentations=args.order_steps,
        sentence_presentations=args.sentence_steps,learning_rate=.003,branches=branches,workers=args.workers,
        identification='Share of present word atoms with abs(cos)>=.9 to a NONZERO column in that held-out row encoding; missing meanings count as misses.',
        merges='A column with abs(cos)>=.9 to a normalized kind+property sum and <.9 to every single atom; no production columns are merged or removed.',
        retirement_gate='Off branch factorial identification >=.9 for all atoms and each kind, better than on; confounded off loses identification with at least one merged pair. Otherwise retain selected-mint rule.',
        identity_budgets=dict(prediction_vocabulary=16,pronoun=32,determiner=80,same_kind=8),
        model=dict(percept_event_width=104,concept_event_width=232,meaning_pairs=64,
                   concept_rows=4096,part_rows=256,ltm_capacity=8192),
        warmup='Stages 1/2 recognition; 64 presentations each of stages 3 two/four words; 8 each stage 4a-d reconstruction and 8 supplied-answer lessons. Continue fixed diagnostic budget after a failed stage, without claiming a passed stage or retiring a rule.',
        source=source,corpus_sha256=corpus))
    random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    started=time.monotonic()
    try:
        warmup(args.out/'warmup',args.order_steps,args.sentence_steps)
        # Independent one-thread processes, all from the same checkpoint.
        # Independent branches never exchange learned state.
        pending=list(branches);running=[];exit_codes={}
        while pending or running:
            while pending and len(running)<args.workers:
                name=pending.pop(0);log=(args.out/(name+'.log')).open('w')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(args.out),'--branch',name],stdout=log,stderr=subprocess.STDOUT)
                running.append((name,process,log))
            for name,process,log in tuple(running):
                code=process.poll()
                if code is not None:
                    log.close();running.remove((name,process,log));exit_codes[name]=code
                    save(args.out/'branch-status.json',exit_codes)
            time.sleep(1)
    finally:
        save(args.out/'source-check.json',dict(source_matched=source==source_snapshot(ROOT),
            corpus_matched=corpus=={p.name:digest(p) for p in DEST.iterdir() if p.is_file()},seconds=time.monotonic()-started))


if __name__=='__main__':main()
