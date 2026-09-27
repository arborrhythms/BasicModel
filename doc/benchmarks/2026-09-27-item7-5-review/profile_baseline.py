"""Observational timing around the unchanged item-8/item-10 serial protocol."""
import argparse
from functools import wraps
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test'),
                str(ROOT / 'doc/benchmarks/2026-09-21-item10')]


def main(output):
    from Models import BasicModel
    spec = importlib.util.spec_from_file_location('reviewed_measure',
        ROOT / 'doc/benchmarks/2026-09-26-item8/measure.py')
    reviewed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reviewed)
    rows, current = [], None
    phase, train_index = 'exploit', 0
    saved = {}
    components = ('exploit_forward', 'explore_forward', 'exploit_backward',
                  'explore_backward', 'snapshot', 'restore')

    def install(name, value):
        saved[name] = BasicModel.__dict__.get(name)
        setattr(BasicModel, name, value)

    def instrument(name, component, *, static=False):
        original = getattr(BasicModel, name)
        @wraps(original)
        def timed(*args, **kwargs):
            label = component() if callable(component) else component
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if current is not None:
                    current[label] += time.perf_counter() - start
        install(name, staticmethod(timed) if static else timed)

    once = BasicModel._run_batch_once
    @wraps(once)
    def trial(self, *args, **kwargs):
        nonlocal phase
        prior, phase = phase, ('explore' if kwargs.get('exploration_trial') else 'exploit')
        try:
            return once(self, *args, **kwargs)
        finally:
            phase = prior
    install('_run_batch_once', trial)
    instrument('_what_or_think', lambda: phase + '_forward')
    instrument('_backward_training_loss', lambda: phase + '_backward')
    instrument('_compose_state_snapshot', 'snapshot')
    instrument('_restore_compose_state', 'restore', static=True)
    run_batch = BasicModel.runBatch
    @wraps(run_batch)
    def batch(self, *args, **kwargs):
        nonlocal current, train_index
        train = kwargs.get('train', True)
        current = dict.fromkeys(components, 0.)
        current.update(index=len(rows), train=train, split=kwargs.get('split'),
                       warmup=bool(train and train_index < 2))
        start = time.perf_counter()
        try:
            return run_batch(self, *args, **kwargs)
        finally:
            current['total'] = time.perf_counter() - start
            current['other'] = current['total'] - sum(current[k] for k in components)
            current['winner'] = getattr(self, '_compose_winner', None) if train else None
            current['losses'] = ({k: float(v) for k, v in self._compose_trial_losses.items()}
                                 if train and hasattr(self, '_compose_trial_losses') else None)
            rows.append(current)
            train_index += int(train)
            current = None
    install('runBatch', batch)
    try:
        code = reviewed.baseline(output)
    finally:
        for name, value in saved.items():
            if value is None:
                delattr(BasicModel, name)
            else:
                setattr(BasicModel, name, value)
        summaries = {}
        for label, selected in (
                ('warmup_training', [r for r in rows if r['train'] and r['warmup']]),
                ('warmed_training', [r for r in rows if r['train'] and not r['warmup']]),
                ('evaluation', [r for r in rows if not r['train']])):
            if selected:
                means = {k: statistics.mean(r[k] for r in selected)
                         for k in (*components, 'other', 'total')}
                summaries[label] = dict(batches=len(selected), mean_seconds=means,
                    fraction={k: means[k] / means['total'] for k in (*components, 'other')})
        Path(output).with_name('batch-timing.json').write_text(json.dumps(dict(
            clock='time.perf_counter; CPU synchronous wall time',
            forward_scope='BasicModel._what_or_think: forward, understanding and answer construction',
            backward_scope='BasicModel._backward_training_loss; optimizer steps are in other',
            snapshot_scope='both pre-exploit and exploit-end runtime snapshots',
            restore_scope='pre-explore restore plus exploit-end restore when exploit wins',
            other_scope='loss assembly, optimizer steps, staging, housekeeping and remaining batch work',
            epoch_tail='excluded here; unchanged baseline throughput includes epoch tails',
            rows=rows, summaries=summaries), indent=2) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    raise SystemExit(main(parser.parse_args().out))
