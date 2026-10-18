"""Read the saved stage-3 scorer once; no lessons, optimizer, or weight writes."""
import argparse
import json
from pathlib import Path
import random
import traceback
import warnings

import numpy as np
import pytest
import torch

from attention_curriculum_gates import STAGES, present
from identity_experiment import make_model, restore_warmup, save


def run(args):
    from Models import BaseModel
    from SentenceAttention import SentenceField
    restore, native_place = BaseModel.load_weights, SentenceField.place
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(20261010)
    np.random.seed(20261010)
    torch.manual_seed(20261010)
    report = dict(checkpoint=str(args.checkpoint), seed=20261010,
        evaluation_presentations=8, training_presentations=0, optimizer_steps=0,
        completed_evaluation_presentations=0,
        protocol='One free B4 reading per original stage-3 field set; rescore the visited states for CE and gradients only.',
        stages=[])
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            report['loader_drift'] = restore_warmup(model, args.checkpoint, restore)
            model._concept_owner()._online_learning_frozen = True
            layer = model.candidate_attention
            before = {name: p.detach().clone() for name, p in layer.named_parameters()}
            states = []

            def place(field, stm):
                observed = {}
                def capture(_layer, inputs, output):
                    observed.update(context=inputs[0].detach().clone(),
                        candidates=inputs[1].detach().clone(), scores=output.detach().clone())
                hook = layer.register_forward_hook(capture)
                try:
                    result = native_place(field, stm)
                finally:
                    hook.remove()
                eligible = field.remaining & field.active[:, None]
                target = torch.where(eligible, field.positions[None],
                                     torch.iinfo(torch.long).max).argmin(-1)
                states.append(dict(observed, eligible=eligible.clone(),
                    active=field.active.clone(), target=target.clone()))
                return result

            patch.setattr(SentenceField, 'place', place)
            for stage in (row for row in STAGES if row['stage'] == 3):
                states.clear()
                with torch.no_grad():
                    free = present(model, stage)
                report['completed_evaluation_presentations'] += 4
                costs, diagnostics, preactivations = [], [], []
                for state in states:
                    scores = layer(state['context'], state['candidates'])
                    logits = scores.masked_fill(~state['eligible'], -torch.inf)
                    logits = torch.where(state['active'][:, None], logits, 0.)
                    cost = -logits.log_softmax(-1).gather(1, state['target'][:, None]).squeeze(1)
                    cost = torch.where(state['active'], cost, 0.)
                    costs.append(cost)
                    target_score = scores.gather(1, state['target'][:, None]).squeeze(1)
                    other = state['eligible'].scatter(1, state['target'][:, None], False)
                    gap = target_score - scores.masked_fill(~other, -torch.inf).amax(-1)
                    diagnostics.append(dict(eligible=state['eligible'].sum(-1).tolist(),
                        ce=cost.detach().tolist(), target_probability=(-cost.detach()).exp().tolist(),
                        target_gap=[None if not torch.isfinite(v) else float(v) for v in gap.detach()],
                        rescore_max_difference=float((scores.detach()-state['scores']).abs().max())))
                    with torch.no_grad():
                        context = state['context'] / state['context'].norm(dim=-1, keepdim=True).clamp_min(1.)
                        inputs = torch.cat((context[:, None].expand(-1, state['candidates'].shape[1], -1),
                                            state['candidates']), -1)
                        preactivations.append(layer.hidden[0](inputs).flatten())
                counts = torch.stack([s['active'] for s in states]).sum(0).clamp_min(1)
                ce = torch.stack(costs).sum(0) / counts
                named = tuple(layer.named_parameters())
                gradients = torch.autograd.grad(ce.mean(), [p for _, p in named], allow_unused=True)
                z = torch.cat(preactivations)
                report['stages'].append(dict(name=stage['name'], free_orders=free['orders'],
                    free_order_passed=free['read_gate_passed'], row_ce=ce.detach().tolist(),
                    ce_at_saved_checkpoint_passed=bool((ce.detach() <= .01).all()),
                    reads=diagnostics,
                    preactivation=dict(min=float(z.min()), max=float(z.max()),
                        fraction_abs_above_5=float((z.abs() > 5).float().mean()),
                        mean_tanh_derivative=float((1-z.tanh().square()).mean())),
                    gradient_norms={name: None if g is None else float(g.norm())
                                    for (name, _), g in zip(named, gradients)}))
                save(args.out/'results.json', report)
            report['scorer_parameter_max_change'] = max(float((p.detach()-before[name]).abs().max())
                                                       for name, p in layer.named_parameters())
            report['completed'] = True
            save(args.out/'results.json', report)
            print(json.dumps(report, indent=2))
        except BaseException:
            report.update(completed=False, error=traceback.format_exc())
            save(args.out/'failure.json', report)
            raise
        finally:
            model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    run(parser.parse_args())
