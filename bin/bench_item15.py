"""Full reading updates, with timing and a separately profiled update.

Run from the checkout with BASICMODEL_DEVICE=cpu MODEL_COMPILE=none
PYTHONHASHSEED=0. A fixed seed is for paired measurements, never a test gate.
"""
import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1505)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--lesson', action='store_true')
    parser.add_argument('--compiled-step', action='store_true',
                        help='enable the native compiled entry (step mode only)')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-revision', help='revision of an archived --source without Git metadata')
    args = parser.parse_args()
    if args.lesson and args.compiled_step:
        parser.error('--lesson uses the native lesson runner; --compiled-step requires step mode')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.source.resolve() / 'bin'))
    from pretrain_reading import ROOT, LESSON, load_lesson, run
    from Models import BaseModel
    from data import Data
    from What import What
    spec = load_lesson(LESSON)
    config = ROOT / 'data/XOR_grammar.xml'
    record = dict(source=args.source_revision or subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                  seed=args.seed, torch=torch.__version__, platform=platform.platform(),
                  threads=torch.get_num_threads(), batch_size=spec['batch_size'],
                  compiled_step=args.compiled_step,
                  config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  lesson_sha256=hashlib.sha256(LESSON.read_bytes()).hexdigest(),
                  environment={k: v for k, v in os.environ.items() if k.startswith(('MODEL_', 'BASICMODEL_', 'PYTHONHASH', 'TORCHINDUCTOR'))},
                  steps=[])
    record['source_files'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / 'bin').glob('*.py'))}

    def save():
        (args.output / 'timing.json').write_text(json.dumps(record, indent=2) + '\n')

    # runBatch belongs to BasicModel; observe the actual entry, not a kernel.
    from Models import BasicModel
    original = BasicModel.runBatch

    def measured(model, *a, **kw):
        device = next(model.parameters()).device
        record['device'] = str(device)
        def sync():
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize(device)
        from torch._dynamo.utils import counters
        before = dict(counters['stats'])
        updates = getattr(model, '_training_step_count', 0)
        sync()
        start = time.perf_counter()
        result = original(model, *a, **kw)
        sync()
        row = dict(seconds=time.perf_counter() - start,
                   updates=getattr(model, '_training_step_count', 0) - updates,
                   graphs=counters['stats'].get('unique_graphs', 0) - before.get('unique_graphs', 0),
                   train=kw.get('train', False))
        record['steps'].append(row)
        save()
        print('step', len(record['steps']), row, flush=True)
        return result

    BasicModel.runBatch = measured
    start = time.perf_counter()
    if args.lesson:
        report = run(config, LESSON, args.output / 'reader')
        record['lesson_seconds'] = time.perf_counter() - start
        record['passed'] = report['passed']
    else:
        model, _ = BaseModel.from_config(str(config), data=Data())
        if args.compiled_step:
            model.enable_compiled_step()
        record['construction_seconds'] = time.perf_counter() - start
        record['device'] = str(next(model.parameters()).device)
        optimizer = model.getOptimizer()
        model._optimizer = optimizer
        model.train()
        texts = ['hello world hello', 'world hello world', 'red blue red', 'blue red blue']
        record['texts'] = texts
        data = model.inputSpace.data

        def step():
            supervised = data.has_supervised_outputs
            try:
                with data.runtime_batch(texts):
                    data.has_supervised_outputs = False
                    source = model.inputSpace.prepInput(texts)
                    record['input_shape'] = list(source.shape)
                    return model.runBatch(train=True, optimizer=optimizer, batchSize=len(texts),
                        split='runtime', questions=tuple(What.present(i, split='runtime') for i in range(len(texts))),
                        batch_override=(source, torch.empty(len(texts), 0, device=source.device)))
            finally:
                data.has_supervised_outputs = supervised
                model.End()
                model.dispatch_per_row_reset([True] * len(texts))

        try:
            for _ in range(args.steps):
                step()
            torch.save(dict(parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters()},
                            buffers={n: p.detach().cpu().clone() for n, p in model.named_buffers()},
                            initialized_buffer_lengths={f'{n}.{suffix}': sum(module._priming_bridge_counts)
                                for n, module in model.named_modules() if hasattr(module, '_priming_bridge_counts')
                                for suffix in ('_priming_bridge_src', '_priming_bridge_dst', '_priming_bridge_bits')},
                            optimizer=optimizer.state_dict(),
                            rng=torch.get_rng_state()), args.output / 'updated.pt')
            # Instrumentation is outside the timing sample, on the next update.
            BasicModel.runBatch = original
            profiler = cProfile.Profile()
            profiler.runcall(step)
            profiler.dump_stats(str(args.output / 'update.prof'))
            stats = pstats.Stats(profiler)
            rows = []
            for (file, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
                rows.append(dict(file=file, line=line, name=name, calls=calls,
                                 own_seconds=own, cumulative_seconds=cumulative))
            record['profile'] = sorted(rows, key=lambda r: -r['cumulative_seconds'])
            record['host_reads'] = [r for r in rows if any(token in r['name'] for token in ("'tolist'", "'item'", "'cpu'", "'__bool__'", "'__int__'", "'__float__'"))]
            # A second, untimed update counts implicit Python scalar reads
            # that cProfile's C-call events do not report. Counter bookkeeping
            # never reads a tensor to identify a repeated lookup.
            import SentenceLooks
            from Spaces import Codebook
            host, at_look, calls = Counter(), Counter(), Counter()
            in_look = [False]
            methods = {name: getattr(torch.Tensor, name) for name in
                       ('tolist', 'item', 'cpu', '__bool__', '__int__', '__float__')}
            def counted(name, function):
                def invoke(*a, **kw):
                    host[name] += 1
                    if in_look[0]:
                        at_look[name] += 1
                    return function(*a, **kw)
                return invoke
            lookup, looks = Codebook.lookup_rows, SentenceLooks.Looks.run
            def rows_read(book, indices, **kw):
                caller = sys._getframe(1)
                calls[(Path(caller.f_code.co_filename).name, caller.f_code.co_name)] += 1
                return lookup(book, indices, **kw)
            def looking(*a, **kw):
                in_look[0] = True
                try:
                    return looks(*a, **kw)
                finally:
                    in_look[0] = False
            try:
                for name, function in methods.items():
                    setattr(torch.Tensor, name, counted(name, function))
                Codebook.lookup_rows, SentenceLooks.Looks.run = rows_read, looking
                step()
            finally:
                for name, function in methods.items():
                    setattr(torch.Tensor, name, function)
                Codebook.lookup_rows, SentenceLooks.Looks.run = lookup, looks
            record['host_census'] = dict(update=dict(host), looks=dict(at_look),
                                         lookups_by_caller={':'.join(k): v for k, v in calls.items()})
            # Capture live gradients at every optimizer boundary in a separate
            # complete update. runBatch clears .grad before returning, so a
            # post-update .grad snapshot would certify nothing.
            optimizer_step, gradients = optimizer.step, []
            def audited_step(*a, **kw):
                gradients.append({n: None if p.grad is None else p.grad.detach().cpu().clone()
                                  for n, p in model.named_parameters()})
                return optimizer_step(*a, **kw)
            optimizer.step = audited_step
            try:
                step()
            finally:
                optimizer.step = optimizer_step
            record['gradient_audit'] = dict(steps=len(gradients),
                live_tensors=[sum(g is not None for g in snapshot.values()) for snapshot in gradients])
            torch.save(gradients, args.output / 'gradients.pt')
        finally:
            model.End()
    record['total_seconds'] = time.perf_counter() - start
    record['warmup_steps'] = record['steps'][:3]
    steady = [r['seconds'] for r in record['steps'][3:] if not r['graphs'] and r['train']]
    record['steady_median_seconds'] = statistics.median(steady) if steady else None
    record['steady_mean_seconds'] = statistics.mean(steady) if steady else None
    save()
    print('finished', {k: v for k, v in record.items() if k.endswith('seconds')}, flush=True)


if __name__ == '__main__':
    main()
