"""Read-only attention audit on a retained warm-up checkpoint.

One ordinary evaluation, no optimizer or lesson guidance. Hooks observe the
native scorer's linear input, tanh output and final logits without replacing
any value. This state is never reused by the matched experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

import Models
from attention_curriculum_gates import present, STAGES
from identity_experiment import make_model, restore_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    native_restore = Models.BaseModel.load_weights
    reads = []
    first = {}
    handles = []
    with pytest.MonkeyPatch.context() as patch:
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.checkpoint, native_restore)

            def linear_hook(module, inputs, output):
                value = output.detach()
                reads.append(dict(candidates=value.shape[1],
                    preactivation_min=float(value.min()),
                    preactivation_max=float(value.max()),
                    preactivation_abs_ge_10=float(value.abs().ge(10).float().mean()),
                    preactivation_candidate_spread=(value.amax(1)-value.amin(1)).amax(-1).tolist()))
                if not first:
                    first.update(input=inputs[0].detach().cpu().clone(),
                                 preactivation=value.cpu().clone())

            def tanh_hook(module, inputs, output):
                value = output.detach()
                reads[-1]['hidden_candidate_spread'] = (value.amax(1)-value.amin(1)).amax(-1).tolist()
                reads[-1]['hidden_abs_eq_one'] = float(value.abs().eq(1).float().mean())
                if 'hidden' not in first:
                    first['hidden'] = value.cpu().clone()

            def score_hook(module, inputs, output):
                value = output.detach()
                reads[-1]['logits'] = value.tolist()
                reads[-1]['logit_candidate_spread'] = (value.amax(1)-value.amin(1)).tolist()
                if 'logits' not in first:
                    first['logits'] = value.cpu().clone()

            layer = model.candidate_attention
            handles = [layer.hidden[0].register_forward_hook(linear_hook),
                       layer.hidden[1].register_forward_hook(tanh_hook),
                       layer.register_forward_hook(score_hook)]
            report = present(model, STAGES[0])
            result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol=__doc__, native_loader_drift=drift,
                fields=STAGES[0]['fields'], reads=reads,
                readbacks=report['readbacks'], free_readbacks=report['free_readbacks'],
                orders=report['orders'])
            (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            torch.save(first, args.out/'first-read.pt')
            print(json.dumps(result), flush=True)
        finally:
            for handle in handles:
                handle.remove()
            model.End()


if __name__ == '__main__':
    main()
