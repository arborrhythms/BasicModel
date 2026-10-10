"""Identity lesson measurement; labels are consumed only after native closing.

The native observer records selected semantic references and written individual
rows. It never supplies a reading, reference, candidate or target to the model.
Missing references and candidate collisions stay in the unconditional denominator.
This cold baseline is not a learning or rule-retirement certificate.
"""
from collections import defaultdict
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import time

from identity_corpus import DEST, ROOT


def load_split(split, directory=DEST):
    texts = [json.loads(s) for s in (directory/f'{split}.text.jsonl').read_text().splitlines()]
    labels = {r['id']:r for s in (directory/f'{split}.labels.jsonl').read_text().splitlines()
              for r in [json.loads(s)]}
    if len(labels)!=len(texts) or set(labels)!={r['id'] for r in texts}:
        raise ValueError('text/label IDs must match exactly')
    return texts, labels


def valid_reference(value):
    # Negative native occurrence addresses are valid; -1 and 0 are sentinels.
    return type(value) is int and value not in (-1,0)


def score_document(label, observation):
    refs = observation.get('references', {})
    result = dict(id=label['id'], family=label['family'], error=observation.get('error'),
                  binding=None, rows=None)
    probe=label['probe']
    if probe:
        candidates=[refs.get(mid) for mid in probe['candidates']]
        selected=refs.get(probe['mention'])
        complete=all(valid_reference(r) for r in candidates)
        distinct=complete and len(set(candidates))==len(candidates)
        selected_index=candidates.index(selected) if distinct and selected in candidates else None
        target_index=probe['candidates'].index(probe['target'])
        # An ID shared by two candidates cannot identify either one.
        result['binding']=dict(candidate_count=probe['candidate_count'],
            recency_rank=probe['recency_rank'], role=probe['role'],
            first_mention_rank=probe['first_mention_rank'], informative=probe['informative'],
            candidate_coverage=sum(valid_reference(r) for r in candidates)/len(candidates),
            collision=complete and not distinct, resolved=selected_index is not None,
            correct=selected_index==target_index if selected_index is not None else False,
            chance=1/len(candidates), selected_index=selected_index, target_index=target_index,
            selected_reference=selected, candidate_references=candidates)
        mentions={m['id']:m for m in label['mentions']}
        ranked=sorted(range(len(candidates)),key=lambda i:(mentions[probe['candidates'][i]]['sentence'],
                        mentions[probe['candidates'][i]]['word']),reverse=True)
        result['binding']['selected_recency_rank']=(None if selected_index is None else ranked.index(selected_index))
    pair=label['row_pair']
    if pair:
        values=[refs.get(mid) for mid in pair['mentions']]
        resolved=all(valid_reference(v) for v in values)
        actual=len(set(values)) if resolved else None
        result['rows']=dict(expected=pair['expected'], actual=actual, resolved=resolved,
                            correct=actual==pair['expected'], references=values)
    return result


def summarize(results):
    groups=defaultdict(list)
    for row in results:
        b=row.get('binding')
        if b:
            for name in ('all',f"family={row['family']}",f"candidates={b['candidate_count']}",
                         f"recency={b['recency_rank']}",f"role={b['role']}",
                         f"first_mention={b['first_mention_rank']}",
                         f"cell={b['candidate_count']}/{b['recency_rank']}/{b['role']}"):
                groups[name].append(b)
        if row.get('rows'):
            groups[f"rows={row['family']}/{row['rows']['expected']}"].append(row['rows'])
    summary={}
    for name, values in sorted(groups.items()):
        n=len(values); resolved=sum(v['resolved'] for v in values)
        summary[name]=dict(n=n, correct=sum(v['correct'] for v in values),
            accuracy=sum(v['correct'] for v in values)/n, resolved=resolved,
            coverage=resolved/n,
            conditional_accuracy=(sum(v['correct'] for v in values)/resolved if resolved else None))
        if 'candidate_count' in values[0]:
            summary[name].update(chance=sum(v['chance'] for v in values)/n,
                candidate_coverage=sum(v['candidate_coverage'] for v in values)/n,
                collisions=sum(v['collision'] for v in values),
                selects_recent=sum(v['selected_recency_rank']==0 for v in values)/n,
                informative=all(v['informative'] for v in values))
    return dict(documents=len(results), errors=sum(r.get('error') is not None for r in results), groups=summary)


def control_observation(label, policy):
    """Evaluator sanity controls. These are never native-model observations."""
    entities=list(dict.fromkeys(m['entity'] for m in label['mentions']))
    refs={m['id']:100+entities.index(m['entity']) for m in label['mentions']}
    probe=label['probe']
    if probe and policy=='recent':
        mentions={m['id']:m for m in label['mentions']}
        recent=max(probe['candidates'],key=lambda mid:(mentions[mid]['sentence'],mentions[mid]['word']))
        refs[probe['mention']]=refs[recent]
    return dict(references=refs)


@contextmanager
def native_observer(model):
    """Read-only taps on the chosen journal and the existing row writer.

    First/second mention offsets are *grading* provenance. Mode names never
    enter the metrics; the current journal adapter reads requested order-one
    references, and the writer supplies actual addresses for minted rows.
    A future adapter may expose column explanations at this same seam.
    """
    import torch
    from ReferenceContext import reference_requests
    captured={}; current_sid=[None]
    observe=model._sentence_observation
    store=model.symbolSpace.ltm_store
    write=store.write_clause
    def observation(state,sid,active,**kwargs):
        value=observe(state,sid,active,**kwargs)
        if not kwargs.get('admit'):
            return value
        current_sid[0]=int(sid)
        positions=model._derivation_program(sid)[0]
        isp=model.inputSpace
        records=[]
        for b,program in enumerate(value['entries']):
            tokens={}
            if program is not None:
                source=(isp._word_active_mask[b] & (isp._packed_sentence_ids[b]==sid)).nonzero().flatten().tolist()
                requests=reference_requests(model.languageSpace,program.actions,operation_refs=program.operation_refs)
                for leaf,position in enumerate(positions[b].tolist()):
                    if position<0 or leaf>=len(program.word_rows) or position not in source:
                        continue
                    word=source.index(position)
                    request=requests.get(leaf)
                    reference=(request[2] if request is not None and request[0]==1 else None)
                    tokens[word]=dict(reference=reference, word_id=int(program.word_ids[leaf]),
                        requested=request is not None and request[0]==1,
                        source='selected' if valid_reference(reference) else 'unresolved')
            records.append(dict(tokens=tokens))
        captured[int(sid)]=records
        return value
    def writing(clause,*args,**kwargs):
        row=write(clause,*args,**kwargs)
        sid=current_sid[0]; b=kwargs.get('stream',-1)
        if sid in captured and 0<=b<len(captured[sid]):
            written=kwargs.get('written_rows') or {}
            by_word=defaultdict(set)
            def visit(node):
                index=written.get(id(node))
                # A sentence and its owned NP can share an order and head.
                # Follow the writer's actual head reference instead of
                # counting the enclosing sentence as another individual.
                head=node.refs[0]
                child=(node.children[head[1]] if isinstance(head,tuple) else None)
                owns_head=(child is not None and child.order==1 and
                           child.subject_word_id==node.subject_word_id)
                if index is not None and node.order==1 and not owns_head:
                    by_word[node.subject_word_id].add(int(store.row_ids[index]))
                for child in (*node.children,*node.companions): visit(child)
            visit(clause)
            for token in captured[sid][b]['tokens'].values():
                matches=by_word.get(token['word_id'],set())
                if token['reference']==-1 and token['requested'] and len(matches)==1:
                    token.update(reference=next(iter(matches)),source='written individual')
        return row
    model._sentence_observation=observation; store.write_clause=writing
    try: yield captured
    finally:
        model._sentence_observation=observe; store.write_clause=write


def grade_capture(label, captured, batch_row):
    refs={}; provenance={}
    for mention in label['mentions']:
        entries=captured.get(mention['sentence'],())
        token=(entries[batch_row]['tokens'].get(mention['word'],{}) if batch_row<len(entries) else {})
        refs[mention['id']]=token.get('reference')
        provenance[mention['id']]=token
    return dict(references=refs,provenance=provenance)


def run(args):
    import tempfile
    import traceback
    import warnings
    import pytest
    import torch
    import util
    from What import What
    from bounded_tests import source_snapshot
    from test_compiled_word_chunk import _tiny_canonical_model
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)  # Measurement reproducibility; never an assertion.
    util.TheCompileBackend='none'
    source=source_snapshot(ROOT)
    corpus_before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob('*') if p.is_file()}
    texts,labels=load_split(args.split)
    train_texts,_=load_split(args.train_split)
    if args.limit is not None: texts=texts[:args.limit]
    if args.train_limit is not None: train_texts=train_texts[:args.train_limit]
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]; observed=[]; start=time.monotonic(); trained=0; failure=None
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as patch:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model=_tiny_canonical_model(Path(tmp),patch,input_width=128,word_buckets='64',
                concept_rows=4096,training_overrides={'intraLossWeight':0.,'reconstructionPlacement':'eager'})
        (args.out/'model.xml').write_text((Path(tmp)/'tiny_chunk_model.xml').read_text())
        model._tensor_peer_while_eager=True; model.checkpoint_every_batches=0
        model.inputSpace.data.has_supervised_outputs=False
        optimizer=model.getOptimizer(lr=.003) if args.epochs else None
        data=model.inputSpace.data
        plans=[('train',train_texts)]*args.epochs+[('evaluation',texts)]
        try:
            for epoch,(phase,selected) in enumerate(plans):
                training=phase=='train'
                for offset in range(0,len(selected),args.batch_size):
                    batch=selected[offset:offset+args.batch_size]; B=len(batch)
                    # The model sees only the public text and opaque document
                    # addresses. Gold candidate lists/roles remain in the grader.
                    addresses=[]; indices=[]
                    split='train' if training else 'validation'
                    for doc in batch:
                        indices.append([])
                        for sid,_ in enumerate(doc['sentences']):
                            indices[-1].append(len(addresses))
                            addresses.append(dict(document=doc['id'],sentence=sid,document_key=doc['id'],
                                                  sentence_index=sid,split=split))
                    data.source_addresses={split:addresses}
                    captured={}; error=None
                    try:
                        with native_observer(model) as captured,torch.set_grad_enabled(training),warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            inputs=model.inputSpace.prepPackedInput([doc['sentences'] for doc in batch])
                            model.runBatch(train=training,optimizer=optimizer,batchNum=offset//args.batch_size,
                                batchSize=B,split=split,batch_override=(inputs,torch.empty(B,0)),source_rows=indices,
                                questions=tuple(What.present(b,split=split) for b in range(B)))
                    except Exception:
                        error=traceback.format_exc()
                    if training:
                        if error is None: trained+=B
                    else:
                        for b,doc in enumerate(batch):
                            label=labels[doc['id']]
                            obs=grade_capture(label,captured,b) if error is None else dict(references={},error=error)
                            observed.append(dict(id=doc['id'],**obs)); results.append(score_document(label,obs))
                    progress=dict(phase=phase,epoch=epoch,completed=offset+B,selected=len(selected),
                                  trained=trained,error=error)
                    (args.out/'progress.json').write_text(json.dumps(progress)+'\n')
                    print(json.dumps(progress),flush=True)
                    # Failed native batches are not rerolled or bypassed. All
                    # remaining evaluation cases stay in the denominator.
                    if error is not None:
                        failure=error; break
                    model.flush_word_buffers(); model.dispatch_packed_soft_reset([True]*B)
                    model.dispatch_per_row_reset([True]*B); model.dispatch_soft_reset(); model.post_tick_compact()
                if failure is not None:break
            if failure is not None:
                done={r['id'] for r in results}
                for doc in texts:
                    if doc['id'] in done:continue
                    obs=dict(references={},error='not run after native batch failure')
                    observed.append(dict(id=doc['id'],**obs)); results.append(score_document(labels[doc['id']],obs))
            component_columns=len(model._concept_owner().components.nouns.ids)
        finally:model.End()
    controls={name:summarize([score_document(labels[d['id']],control_observation(labels[d['id']],name))
                            for d in texts]) for name in ('oracle','recent')}
    report=dict(protocol='native text; no forced reading, binding label supervision or rule retirement',
        seed=args.seed, split=args.split, selected=len(texts), train_split=args.train_split,
        epochs=args.epochs, training_documents=trained, requested_training_documents=len(train_texts)*args.epochs,
        failure=failure, corpus_sha256=corpus_before, seconds=time.monotonic()-start,
        source_manifest=source, component_columns=component_columns,
        summary=summarize(results), controls=controls, results=results)
    (args.out/'observations.json').write_text(json.dumps(observed,indent=2)+'\n')
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    assert source_snapshot(ROOT)==source,'source changed during measurement'
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob('*') if p.is_file()}==corpus_before
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('eval','reversed_eval'),default='eval')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--epochs',type=int,default=0)
    parser.add_argument('--train-split',choices=('train','biased_train'),default='train')
    parser.add_argument('--train-limit',type=int)
    run(parser.parse_args())
