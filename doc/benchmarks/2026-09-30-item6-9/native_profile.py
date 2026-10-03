"""Unseeded timing of the existing serial baseline protocol."""
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
    import SentenceCompose
    spec = importlib.util.spec_from_file_location('reviewed_measure',
        Path(__file__).with_name('native_probe.py'))
    reviewed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reviewed)
    rows, current = [], None
    train_index = 0
    saved = {}
    components = ('exploit_compose', 'explore_compose', 'exploit_backward',
                  'explore_backward', 'batch_backward', 'sentence_scoring',
                  'snapshot', 'restore')

    def install(name, value):
        saved[name] = BasicModel.__dict__.get(name)
        setattr(BasicModel, name, value)

    def instrument(name, component, *, static=False):
        original = getattr(BasicModel, name)
        @wraps(original)
        def timed(*args, **kwargs):
            label = component(args[0]) if callable(component) else component
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if current is not None:
                    current[label] += time.perf_counter() - start
        install(name, staticmethod(timed) if static else timed)

    instrument('_backward_training_loss', lambda model:
        (getattr(model, '_sentence_trial', None) or 'batch') + '_backward')
    instrument('_snapshot_sentence_state', 'snapshot', static=True)
    instrument('_restore_sentence_state', 'restore')
    instrument('_sentence_path_cost', 'sentence_scoring')
    pair = SentenceCompose.sentence_pair
    @wraps(pair)
    def timed_pair(cache, compose, *args, **kwargs):
        def timed_compose(cache, prior):
            label = 'exploit_compose' if prior is None else 'explore_compose'
            nested = sum(current[k] for k in ('snapshot', 'restore')) if current else 0.
            start = time.perf_counter()
            try:
                return compose(cache, prior)
            finally:
                if current is not None:
                    elapsed = time.perf_counter() - start
                    nested = sum(current[k] for k in ('snapshot', 'restore')) - nested
                    current[label] += elapsed - nested
        return pair(cache, timed_compose, *args, **kwargs)
    SentenceCompose.sentence_pair = timed_pair
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
            current['sentence_winners'] = [v.cpu().tolist() for v in self._sentence_winners]
            current['sentence_costs'] = [v.cpu().tolist() for v in self._sentence_trial_costs]
            rows.append(current)
            train_index += int(train)
            current = None
    install('runBatch', batch)
    try:
        previous_argv = sys.argv
        sys.argv = [str(Path(__file__).with_name('native_probe.py')), '--out', output]
        try:
            reviewed.main()
            code = 0
        finally:
            sys.argv = previous_argv
    finally:
        SentenceCompose.sentence_pair = pair
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
            compose_scope='sentence_pair compose callbacks, excluding separately timed restore; cached perception',
            backward_scope='BasicModel._backward_training_loss; optimizer steps are in other',
            scoring_scope='sentence reconstruction and row-local prediction preview',
            snapshot_scope='pre-sentence scratch tensor bindings; no deep model copy',
            restore_scope='scratch STM/trace publication before each compose trial',
            other_scope='perception, optimizer steps, winner commits, staging, answers and batch housekeeping',
            epoch_tail='excluded here; unchanged baseline throughput includes epoch tails',
            rows=rows, summaries=summaries), indent=2) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    raise SystemExit(main(parser.parse_args().out))

