"""Observe the selected journal and actual writer tree on retained identity text."""
import argparse
import json
from pathlib import Path
import pytest
import torch
import Models
from identity_experiment import make_model, restore_warmup, present_documents
from identity_measurement import load_split
from ReferenceContext import reference_requests
from ThoughtReferences import bindings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--retained-attention', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    restore = Models.BaseModel.load_weights
    texts, labels = load_split('eval')
    if args.all:
        reverse, reverse_labels = load_split('reversed_eval')
        texts += reverse
        labels.update(reverse_labels)
    else:
        texts = texts[:4]
    logs = []
    with pytest.MonkeyPatch.context() as patch:
        model = make_model(args.out, patch)
        try:
            restore_warmup(model, args.checkpoint, restore)
            if args.retained_attention:
                # Replay the old checkpoint's original scorer for the
                # observer regression only, keeping S out of this audit.
                layer = model.candidate_attention
                def original_forward(context, candidates):
                    inputs = torch.cat((context[:,None].expand(-1,candidates.shape[1],-1),candidates),-1)
                    return layer.readout(layer.hidden(inputs.detach())).squeeze(-1)
                patch.setattr(layer,'forward',original_forward)
            model._concept_owner()._online_learning_frozen = True
            store = model.symbolSpace.ltm_store
            write, observe = store.write_clause, model._sentence_observation
            def observation(state, sid, active, **kw):
                result = observe(state, sid, active, **kw)
                if kw.get('admit'):
                    logs.append(dict(sid=int(sid), programs=[None if p is None else dict(
                        word_ids=p.word_ids.tolist(), word_rows=p.word_rows.tolist(),
                        actions=p.actions.tolist(), operation_refs=p.operation_refs.tolist(),
                        reference_orders=p.reference_orders.tolist(),
                        requests=reference_requests(model.languageSpace,p.actions,operation_refs=p.operation_refs))
                        for p in result['entries']]))
                return result
            def writing(clause, *a, **kw):
                row = write(clause, *a, **kw)
                written = kw.get('written_rows', {})
                def node(value):
                    index = written.get(id(value))
                    return dict(word=value.subject_word_id, order=value.order, refs=str(value.refs),
                        row=index, address=None if index is None else int(store.row_ids[index]),
                        formations=bindings(value.meaning).get('_formation_records'),
                        children=[node(v) for v in value.children], companions=[node(v) for v in value.companions])
                logs.append(dict(stream=kw.get('stream'), tree=node(clause)))
                return row
            model._sentence_observation = observation
            store.write_clause = writing
            try:
                results, evidence, _ = present_documents(model,texts,labels=labels,log=args.out)
            finally:
                model._sentence_observation = observe
                store.write_clause = write
            (args.out/'result.json').write_text(json.dumps(dict(texts=texts, logs=logs,
                retained_attention=args.retained_attention, results=results, evidence=evidence),indent=2)+'\n')
        finally:
            model.End()


if __name__ == '__main__':
    main()
