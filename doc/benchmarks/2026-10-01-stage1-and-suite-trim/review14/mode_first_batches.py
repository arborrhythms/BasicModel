"""One candidate-only production first batch for every migrated configuration.

No model seed; no configuration/threshold changes. CPU numerical execution
excludes graph capture, matching the preceding configuration-price receipt.
Each worker retains the sweep's 8 GiB ceiling and 30-minute deadline.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded


def child(config, output):
    sys.path.insert(0, str(ROOT/'bin'))
    import recon_bench
    started = time.monotonic()
    model = None
    result = dict(config=config, seed=None, backend='none',
        measurement='one first training batch at configured batch; no graph capture or evaluation',
        guard_gib=8)
    try:
        model, device, lr, batch = recon_bench._build_model(str(ROOT/config))
        result.update(build_seconds=time.monotonic()-started, batch=batch,
            scope=getattr(model, 'reconstruction_scope', 'no grammar reading'),
            reconstruct_in_loop=getattr(model, 'reconstruct_in_loop', False))
        optimizer = model.getOptimizer(lr=lr)
        before = time.monotonic()
        output_loss, recon_loss, _, _ = model.runEpoch(
            optimizer=optimizer, batchSize=batch, split='train', max_batches=1)
        result.update(training_seconds=time.monotonic()-before,
            output_loss=float(output_loss), reconstruction_loss=float(recon_loss), completed=True)
        missing = getattr(model.inputSpace, '_reconstruction_missing_sentence_count', None)
        result['missing_surface_sentences_at_last_boundary'] = int(missing) if missing is not None else None
    except BaseException as exc:
        result.update(completed=False, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['total_seconds'] = time.monotonic()-started
        bounded.write_json(output, result)
        if model is not None:
            model.End()


def run(output):
    output.mkdir(exist_ok=False)
    prior = json.loads((HERE.parent/'cost-function/scoped-configuration-audit.json').read_text())
    retired = {'data/MM_bpe.xml', 'data/MM_20M_legacy.xml'}
    configs = [r['config'] for r in prior
        if r['scope'] in ('exempt: old reading mode', 'perception: numeric configuration')
        and r['config'] not in retired | {'data/model.xml'}]
    # Start the newly scoped XOR fixture first; every configuration is still run once.
    configs.sort(key=lambda c: (c != 'data/XOR_grammar.xml', c))
    frozen = bounded.source_snapshot(ROOT)
    bounded.write_json(output/'manifest.json', dict(source=frozen, configs=configs,
        guard_gib=8, timeout_seconds=1800, workers=1, seed=None, backend='none',
        retired=sorted(retired), template='data/model.xml is inherited, not an executable configuration'))
    env = bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED', None)
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='none', BASIC_AUTOLOAD='false', RUN_SLOW='1')
    results = []
    for config in configs:
        assert bounded.source_snapshot(ROOT) == frozen
        stem = Path(config).stem
        guarded = bounded.GuardedProcess([sys.executable, str(Path(__file__).resolve()),
            '--child', '--config', config, '--output', str(output/(stem+'.json'))],
            cwd=ROOT, env=env, log_path=output/(stem+'.log'),
            memory_bytes=8*bounded.GIB, timeout=1800).start()
        try:
            while (result := guarded.poll()) is None:
                bounded.write_json(output/'progress.json', dict(completed=results, active=config,
                    pid=guarded.proc.pid, memory_bytes=guarded.current_memory_bytes))
                time.sleep(.25)
        finally:
            if not guarded.finished:
                guarded.stop(exit_code=130, reason='measurement_stopped')
        bounded.write_json(output/(stem+'-process.json'), result)
        results.append(dict(config=config, **result))
        print(json.dumps({k: results[-1][k] for k in
            ('config','exit_code','reason','peak_memory_bytes','elapsed_seconds')}), flush=True)
    assert bounded.source_snapshot(ROOT) == frozen
    bounded.write_json(output/'complete.json', dict(results=results, source_matched=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--output', type=Path, default=HERE/'mode-first-batches')
    args = parser.parse_args()
    if args.child:
        child(args.config, args.output)
    else:
        run(args.output)
