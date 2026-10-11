"""One frozen-checkpoint recovery replay: locate the absent meaning signal.

No optimizer, answer target or reading guidance. Conceptual online admission
is frozen. This probe's state is discarded and never enters the experiment.
"""
import argparse
import json
from pathlib import Path

import pytest
import torch

import Models
from identity_experiment import digest, make_model, present_documents, recovery_data, restore_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    native_restore = Models.BaseModel.load_weights
    streams, _ = recovery_data()
    docs = streams['heldout'][:4]
    captures = []
    with pytest.MonkeyPatch.context() as patch:
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.checkpoint, native_restore)
            owner = model._concept_owner()
            owner._online_learning_frozen = True
            observe = model._sentence_observation

            def observed(state, sid, active, **kwargs):
                result = observe(state, sid, active, **kwargs)
                if kwargs.get('admit'):
                    chart = owner.components._ready()
                    isp = model.inputSpace
                    words = model._tensor_pushed_ideas.detach()
                    grammar = getattr(isp, '_ar_grammar_object_atoms', None)
                    rows = getattr(isp, '_ar_grammar_object_rows', None)
                    for b, doc in enumerate(docs):
                        text = doc['sentences'][int(sid)]
                        positions = (isp._word_active_mask[b] &
                                     (isp._packed_sentence_ids[b] == sid)).nonzero().flatten()
                        point = result['clauses'][b].point if result['clauses'][b] is not None else None
                        tokens = []
                        for token, position in zip(text.split(), positions.tolist()):
                            w = words[b, position]
                            g = grammar[b, position].detach() if torch.is_tensor(grammar) else None
                            tokens.append(dict(token=token, position=position,
                                grammar_row=int(rows[b, position]) if torch.is_tensor(rows) else None,
                                grammar_meaning_norm=None if g is None else float(g[chart.meaning_start:chart.content_width].norm()),
                                pushed_full_norm=float(w.norm()),
                                pushed_meaning_norm=float(w[chart.meaning_start:chart.content_width].norm()),
                                projected_norm=float(chart.project(w).norm())))
                        captures.append(dict(id=doc['id'], text=text, tokens=tokens,
                            fused_point_available=point is not None,
                            fused_point_meaning_norm=None if point is None else float(point[chart.meaning_start:chart.content_width].norm()),
                            role_meaning_norm=(None if result['meanings'][b] is None else
                                float(result['meanings'][b].roles[..., chart.meaning_start:chart.content_width].norm()))))
                    torch.save(dict(pushed=words.cpu(), grammar=None if grammar is None else grammar.detach().cpu(),
                                    grammar_rows=None if rows is None else rows.detach().cpu()), args.out/'word-tensors.pt')
                return result

            model._sentence_observation = observed
            try:
                present_documents(model, docs)
            finally:
                model._sentence_observation = observe
            result = dict(protocol=__doc__, checkpoint_sha256=digest(args.checkpoint),
                script_sha256=digest(Path(__file__)), native_loader_drift=drift, captures=captures)
            (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
        finally:
            model.End()


if __name__ == '__main__':
    main()
