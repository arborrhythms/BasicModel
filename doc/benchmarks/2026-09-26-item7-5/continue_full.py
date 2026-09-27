"""Continue only unfinished cases after the full pool's recorded memory stop."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import LOCK, run_suite, source_snapshot, suite_lock


def main():
    original_path = HERE / 'full/result.json'
    original = json.loads(original_path.read_text())
    assert original['reason'] == 'memory' and original['exit_code'] == 137
    source = json.loads((HERE / 'full/source-manifest.json').read_text())['validated_source']
    assert source_snapshot(ROOT) == source
    assert not original.get('active_workers')
    completed = set(original['completed'])
    # Aborted peers can have durable completed reports that the supervisor
    # has not accounted. Do not silently discard or repeat any such result.
    for path in (HERE / 'full').glob('worker-*.json'):
        if '.request.' in path.name or '.recycle.' in path.name:
            continue
        progress = json.loads(path.read_text())
        assert set(progress.get('completed', [])) <= completed, path
    unfinished = [node for node in original['selected'] if node not in completed]
    stopped_file = 'test/test_stm_relative_sentence_end_state.py'
    isolated = [node for node in unfinished if node.split('::', 1)[0] == stopped_file]
    remaining = [node for node in unfinished if node not in isolated]
    assert len(isolated) == 3 and len(unfinished) == 4077
    plan = dict(original='full', completed_without_repetition=len(completed),
        reason='Keep the memory stop visible; isolate its unfinished cases at the same cap, then continue.',
        isolated=isolated, remaining=remaining,
        workers=3, aggregate_memory_bytes=24 * 2**30,
        per_worker_memory_bytes=8 * 2**30, worker_seconds=1800)
    (HERE / 'full-continuation-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
    # Preserve the original total suite deadline as well as worker limits.
    deadline = (time.monotonic() + original['limits']['suite_seconds']
                - original['elapsed_seconds'] - (time.time() - original_path.stat().st_mtime))
    results = [original]
    names = ['full']
    with suite_lock(LOCK):
        for name, selected, batch_size in (
                ('full-memory-remainder', isolated, 1),
                ('full-continuation', remaining, 32)):
            assert source_snapshot(ROOT) == source
            result = run_suite(root=ROOT, selectors=selected, run_dir=HERE / name,
                memory_bytes=24 * 2**30, worker_memory_bytes=8 * 2**30,
                workers=3, timeout=1800, suite_timeout=max(1, deadline-time.monotonic()),
                batch_size=batch_size, max_files=1)
            results.append(result)
            names.append(name)
            assert Counter(result['selected']) == Counter(selected)
            assert Counter(result['completed']) == Counter(selected), name
    assert source_snapshot(ROOT) == source
    all_completed = [node for result in results for node in result['completed']]
    assert Counter(all_completed) == Counter(original['selected'])
    coverage = dict(segments=names, selected=len(original['selected']),
        completed=len(all_completed), unique_completed=len(set(all_completed)),
        source_unchanged=True, resource_failure_retained=dict(segment='full',
            reason=original['reason'], exit_code=original['exit_code']),
        segment_exit_codes={name: result['exit_code'] for name, result in zip(names, results)},
        segment_seconds={name: result['elapsed_seconds'] for name, result in zip(names, results)})
    (HERE / 'full-coverage.json').write_text(json.dumps(coverage, indent=2)+'\n')
    print(json.dumps(coverage), flush=True)


if __name__ == '__main__':
    main()
