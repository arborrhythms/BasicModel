"""Compare paired full-update receipts, including live optimizer-boundary gradients."""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = [json.loads((p / 'timing.json').read_text()) for p in (args.before, args.after)]
    for key in ('seed', 'torch', 'threads', 'batch_size', 'device', 'config_sha256',
                'lesson_sha256', 'texts', 'input_shape', 'compiled_step'):
        if records[0].get(key) != records[1].get(key):
            raise ValueError(f'incomparable measurement setting: {key}')
    counts, failures = Counter(), []

    def compare(left, right, path, group):
        if torch.is_tensor(left):
            counts[group] += 1
            if not torch.is_tensor(right) or left.shape != right.shape or left.layout != right.layout:
                failures.append(path + ': tensor structure')
                return
            if left.is_sparse:
                left, right = left.coalesce(), right.coalesce()
                same = torch.equal(left.indices(), right.indices()) and torch.equal(left.values(), right.values())
            else:
                same = torch.equal(left, right)
            if not same:
                failures.append(path)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                failures.append(path + ': keys')
                return
            for key in left:
                compare(left[key], right[key], f'{path}.{key}', group)
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                failures.append(path + ': sequence')
                return
            for i, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(a, b, f'{path}.{i}', group)
        elif left != right:
            failures.append(path)

    snapshots = [torch.load(p / 'updated.pt', weights_only=False) for p in (args.before, args.after)]
    for snapshot in snapshots:
        # These nonpersistent capacity buffers use torch.empty. Only their
        # initialized prefix is read by the model; compare every other buffer
        # in full, and compare the recorded lengths as well.
        for name, length in snapshot['initialized_buffer_lengths'].items():
            snapshot['buffers'][name] = snapshot['buffers'][name][:length]
    for group in snapshots[0]:
        compare(snapshots[0][group], snapshots[1][group], group, group)
    gradients = [torch.load(p / 'gradients.pt', weights_only=False) for p in (args.before, args.after)]
    if not gradients[0] or not all(any(g is not None for g in step.values()) for step in gradients[0]):
        raise ValueError('the audit did not capture live gradients')
    compare(*gradients, 'gradients', 'live_gradients')
    for record in records:
        if not all(step['updates'] == 1 for step in record['steps']):
            raise ValueError('each timed presentation must include one optimizer update')
    if len(records[0]['steps']) != len(records[1]['steps']):
        raise ValueError('timed update counts differ')
    report = dict(equal=not failures, tensors_compared=dict(counts), differences=failures,
                  gradient_steps=len(gradients[0]),
                  median_seconds=[r['steady_median_seconds'] for r in records],
                  mean_seconds=[r['steady_mean_seconds'] for r in records])
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['equal'] else 1)


if __name__ == '__main__':
    main()
