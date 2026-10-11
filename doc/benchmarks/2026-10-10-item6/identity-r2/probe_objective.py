"""R2 diagnostic before changing the objective; discarded, frozen copies.

Compare the native reading with a supplied all-sum/STOP derivation and a
diagnostic which substitutes STOP only when the native action is `non`.
Forcing changes selection only; no weights, objective, or production rule
are changed. The all-sum derivation is an explicit mechanism witness, not
a learned parse or an English rule.
"""
import argparse
import copy
import json
from pathlib import Path

import pytest
import torch

import Models
from identity_experiment import make_model, present_documents, recovery_data, restore_warmup, save


def run(checkpoint, folder, mode):
    folder.mkdir(parents=True, exist_ok=False)
    original_loader = Models.BaseModel.load_weights
    streams, _ = recovery_data()
    reports, first, raw_unary = [], [], []
    with pytest.MonkeyPatch.context() as patch:
        model = make_model(folder, patch)
        try:
            layer = model.languageSpace._tree_layer(2)
            initial = copy.deepcopy(layer.chooser)
            drift = restore_warmup(model, checkpoint, original_loader)
            owner = model._concept_owner()
            owner._online_learning_frozen = True
            parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
            forward, choose, unary = layer.forward, layer.select_logits, layer.chooser.score_unary
            context = None

            def forwarding(x, **kwargs):
                nonlocal context
                data = kwargs.get('reference_data') or {}
                previous = context
                context = dict(width=x.shape[1],
                    binary=data.get('binary_ops', tuple(range(layer.r_reduce))),
                    unary=data.get('unary_ops', tuple(range(layer.r_apply))))
                try:
                    return forward(x, **kwargs)
                finally:
                    context = previous

            def scoring_unary(*args, **kwargs):
                result = unary(*args, **kwargs)
                if context is not None and not raw_unary:
                    with torch.no_grad():
                        original = initial.score_unary(*args, **kwargs)
                    raw_unary.append(dict(warm_scores=result[1].detach().tolist(),
                        initial_scores_same_input=original[1].detach().tolist(),
                        warm_stop=result[0].detach().tolist(),
                        initial_stop_same_input=original[0].detach().tolist(),
                        input=args[0].detach().tolist(),
                        op_indices=None if kwargs.get('op_indices') is None else kwargs['op_indices'].tolist()))
                return result

            def selecting(logits, **kwargs):
                native = choose(logits, **kwargs)
                if context is None:
                    return native
                nb = (context['width'] - 1) * len(context['binary'])
                labels = ([model.languageSpace._compose_binary_rules[i].method_name
                           for i in context['binary']] * (context['width'] - 1)
                          + [model.languageSpace._compose_unary_rules[i].method_name
                             for i in context['unary']] * context['width'] + ['STOP'])
                if not first:
                    first.append(dict(logits=logits.detach().tolist(), labels=labels,
                        structural=list(kwargs['structural']), selected=native[0].tolist(),
                        selected_names=[labels[i] for i in native[0].tolist()],
                        exact_max_ties=logits.eq(logits.amax(-1, keepdim=True)).sum(-1).tolist(),
                        raw_unary=raw_unary))
                if mode == 'greedy':
                    return native
                forced = native[0].clone()
                for b in range(len(logits)):
                    if mode in ('retaining', 'retaining_sum', 'retaining_union'):
                        method = dict(retaining='lift', retaining_sum='sum', retaining_union='union')[mode]
                        choices = [i for i in range(nb) if labels[i] == method and torch.isfinite(logits[b, i])]
                        forced[b] = choices[0] if choices else len(labels)-1
                    elif labels[int(forced[b])] == 'non':
                        forced[b] = len(labels)-1
                assert torch.isfinite(logits.gather(1, forced[:, None])).all(), 'supplied reading must be legal'
                return choose(logits, **dict(kwargs, replay_action=forced))

            patch.setattr(layer, 'forward', forwarding)
            patch.setattr(layer, 'select_logits', selecting)
            patch.setattr(layer.chooser, 'score_unary', scoring_unary)
            score = model._sentence_path_cost
            def cost(state, sid, active):
                result = score(state, sid, active)
                registry = model._sentence_cost_registry
                chart = owner.components._ready()
                programs = []
                for program in result[2]['entries']:
                    if program is None:
                        programs.append(None)
                        continue
                    actions = []
                    for kind, operation, word in program.actions.tolist():
                        rules = (model.languageSpace._compose_binary_rules if kind == 1 else
                                 model.languageSpace._compose_unary_rules if kind == 2 else ())
                        actions.append(dict(kind=kind, operation=operation, word=word,
                            method=rules[operation].method_name if rules else None))
                    programs.append(actions)
                reports.append(dict(trial=model._sentence_trial,
                    keep_cost=registry.total(objective='reconstruction').detach().tolist(),
                    terms={name: dict(value=registry._value(rec).detach().tolist(),
                        weight=rec['weight'], trained=rec['trained'], objective=rec['objective'])
                        for name, rec in registry._terms.items()}, breakdown=registry.breakdown(),
                    closed_meaning_norm=result[2]['record'].root[..., chart.meaning_start:chart.content_width].norm(dim=-1).tolist(),
                    clause_point_meaning_norm=[None if clause is None or clause.point is None else
                        float(clause.point[chart.meaning_start:chart.content_width].norm())
                        for clause in result[2]['clauses']],
                    role_meaning_norm=[None if meaning is None else
                        float(meaning.roles[..., chart.meaning_start:chart.content_width].norm())
                        for meaning in result[2]['meanings']],
                    operation_meaning_norms=[None if program is None else
                        program.operation_values[..., chart.meaning_start:chart.content_width].norm(dim=-1).tolist()
                        for program in result[2]['entries']],
                    programs=programs))
                return result
            patch.setattr(model, '_sentence_path_cost', cost)
            present_documents(model, streams['heldout'][:4])
            changed = [k for k, v in model.named_parameters() if k not in parameters or not torch.equal(v, parameters[k])]
            result = dict(mode=mode, checkpoint=str(checkpoint),
                native_loader_drift=drift, parameter_changes=changed,
                first_word=first, readings=reports,
                documents=streams['heldout'][:4])
            save(folder/'result.json', result)
            # The warm-up checkpoint has not seen all held-out forms; its
            # native read may provision percept rows. No optimizer runs and
            # the diagnostic state is discarded. Record that exception.
            assert not set(changed) - {'perceptualSpace._owned_bases.what.W'}, changed
            print(json.dumps(dict(mode=mode, costs=[r['keep_cost'] for r in reports],
                norms=[r['closed_meaning_norm'] for r in reports])), flush=True)
            return result
        finally:
            model.End()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--modes', nargs='+', default=['greedy', 'retaining_sum', 'skip_non'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {mode: run(args.checkpoint, args.out/mode, mode) for mode in args.modes}
    save(args.out/'result.json', dict(protocol=__doc__, results=results))


if __name__ == '__main__':
    main()
