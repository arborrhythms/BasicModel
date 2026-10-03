"""Run the complete slow selection once, in fresh bounded workers.

Ordinary cases retain 8 GiB. Only the two native production objective arms
receive 24 GiB, one worker at a time. Both tiers reserve 24 GiB in aggregate.
This command never rebuilds the environment or installs a schedule.
"""
from collections import Counter
from datetime import datetime, timezone
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

import bounded_tests as bounded

NATIVE_FILE = 'test/test_objective_conflicts_slow.py'
NATIVE_WORKER_GIB = 24
ORDINARY_WORKER_GIB = 8
AGGREGATE_GIB = 24
INLINE_SUITES = ('bin/Legacy.py', 'bin/etc/SPNN.py', 'bin/etc/SigmaPi.py', 'bin/etc/SymPercept.py')


def snapshot_source(root, destination):
    """Freeze sources and fixtures for measurement; this is not a Git checkout.

    The weekly process must not observe subsequent working-tree edits. On
    macOS, copy-on-write clones avoid duplicating the dataset's disk blocks.
    No source is hardlinked or symlinked back to the editable tree.
    """
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    paths = [root / name for name in ('bin', 'test', 'data', 'doc', 'scripts')]
    paths += [p for p in root.iterdir() if p.is_file() and not p.name.startswith('.')]
    for path in paths:
        if not path.exists():
            continue
        target = destination / path.name
        if sys.platform == 'darwin':
            subprocess.run(['/bin/cp', '-cRL', str(path), str(target)], check=True)
        elif path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    # Tests spawning PROJECT/.venv/bin/python must use this same environment.
    # Runtime dependencies are shared; measured source/fixtures remain copies.
    runtime = root / '.venv'
    if runtime.is_dir():
        (destination / '.venv').symlink_to(runtime.resolve(), target_is_directory=True)


def run_tier(root, nodes, directory, ceiling):
    """Attempt every selected case once, continuing after a bounded stop.

    A killed worker remains a failed attempt. Never retry it; only dispatch
    the selectors that the previous bounded run had not started.
    """
    remaining = list(nodes)
    results, attempted = [], set()
    while remaining:
        folder = directory if not results else directory.with_name(directory.name + f'-continued-{len(results):02}')
        result = bounded.run_suite(root=root, selectors=remaining, run_dir=folder,
            memory_bytes=AGGREGATE_GIB*bounded.GIB,
            worker_memory_bytes=ceiling*bounded.GIB, workers=1,
            timeout=1800, suite_timeout=86400, batch_size=1, max_files=1)
        results.append((folder, result))
        started = {node for worker in result['workers'] for node in worker['selected']}
        if not started:
            break  # collection/infrastructure failed; no spin or false completion
        attempted.update(started)
        remaining = [node for node in remaining if node not in attempted]
    return results, attempted


def run_inline(root, directory, environment):
    """Run each parked suite once and retain its real unittest case count."""
    results = []
    for name in INLINE_SUITES:
        expected = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                       for node in ast.walk(ast.parse((root / name).read_text())))
        log = directory / (Path(name).stem + '.log')
        process = bounded.run_guarded([sys.executable, str(root / name), '-v'],
            cwd=root, env=environment, log_path=log,
            memory_bytes=ORDINARY_WORKER_GIB * bounded.GIB, timeout=1800)
        output = log.read_text()
        match = re.search(r'^Ran (\d+) tests? in ', output, re.MULTILINE)
        count = int(match.group(1)) if match else 0
        failures = re.findall(r'^(?:FAIL|ERROR): (.+)$', output, re.MULTILINE)
        skipped = len(re.findall(r"\.\.\. skipped ", output))
        results.append(dict(suite=name, expected=expected, completed=count,
            counts=dict(passed=max(0, count-len(failures)-skipped), failed=len(failures), skipped=skipped),
            failures=failures, process=process, log=str(log)))
    return results


def run(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    started = time.monotonic()
    date = datetime.now(timezone.utc)
    directory = root / 'tmp/slow-tests' / (date.strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    record = dict(date=date.isoformat(), commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        complete=False, selected=0, completed=0, attempted=0, counts={}, failures=[], runs=[],
        duration_seconds=0, source_matched=False,
        ceilings_gib=dict(ordinary_worker=ORDINARY_WORKER_GIB,
                          native_worker=NATIVE_WORKER_GIB, aggregate=AGGREGATE_GIB),
        record_directory=str(directory), working_tree=str(root), inline_runs=[])
    frozen = bounded.source_snapshot(root)
    source = directory / 'source'
    snapshot_source(root, source)
    assert bounded.source_snapshot(source) == frozen, 'weekly source copy is incomplete'
    record['source_directory'] = str(source)
    bounded.write_json(directory / 'source.json', frozen)
    root = source
    previous = {key:os.environ.get(key) for key in
                ('RUN_SLOW', 'RUN_MPS_SLOW', 'OBJECTIVE_CONFLICTS_OUTPUT', 'BASICMODEL_DEVICE')}
    # The suite's ordinary fixtures declare CPU. MPS-only cases explicitly
    # switch to MPS; CUDA-only cases keep their availability skips.
    os.environ.update(RUN_SLOW='1', RUN_MPS_SLOW='1',
                      OBJECTIVE_CONFLICTS_OUTPUT=str(directory/'native-measurements'))
    # Only explicit device marks choose accelerators; slow proofs retain CPU.
    os.environ.pop('BASICMODEL_DEVICE', None)
    exit_code = 125
    try:
        with bounded.suite_lock():
            env = bounded.worker_environment(root)
            env['BASICMODEL_DEVICE'] = 'cpu'  # collection imports never allocate on GPU
            request, response = directory/'collect.request.json', directory/'collect.json'
            bounded.write_json(request, dict(selectors=['test'], collect=True))
            collected = bounded.run_guarded(
                [sys.executable, str(root/'test/pytest_worker.py'), str(request), str(response)],
                cwd=root, env=env, log_path=directory/'collect.log',
                memory_bytes=ORDINARY_WORKER_GIB*bounded.GIB, timeout=1800)
            record['collection'] = collected
            if collected['exit_code']:
                record['failures'].append(dict(phase='collection', reason=collected['reason']))
                exit_code = collected['exit_code']
                return exit_code, directory
            slow = json.loads(response.read_text())['slow_selected']
            assert slow and len(slow) == len(set(slow))
            tiers = [('ordinary', [node for node in slow if not node.startswith(NATIVE_FILE+'::')],
                      ORDINARY_WORKER_GIB),
                     ('native', [node for node in slow if node.startswith(NATIVE_FILE+'::')],
                      NATIVE_WORKER_GIB)]
            record['selected'] = len(slow)
            bounded.write_json(directory/'selected.json', slow)
            counts = Counter()
            exit_code = 0
            inline = run_inline(root, directory, env)
            record['inline_runs'] = inline
            for result in inline:
                record['selected'] += result['expected']
                record['completed'] += result['completed']
                record['attempted'] += result['expected']
                counts.update(result['counts'])
                if result['process']['exit_code'] or result['completed'] != result['expected']:
                    exit_code = result['process']['exit_code'] or 125
                    record['failures'].append(result)
            record['counts'] = dict(counts)
            bounded.write_json(directory/'record.json', record)
            for name, nodes, ceiling in tiers:
                if not nodes:
                    continue
                assert bounded.source_snapshot(root) == frozen, 'source changed between slow tiers'
                print(f'Slow {name}: {len(nodes)} cases, {ceiling} GiB worker / 24 GiB aggregate', flush=True)
                segments, attempted = run_tier(root, nodes, directory/name, ceiling)
                record['attempted'] += len(attempted)
                for folder, result in segments:
                    record['runs'].append(dict(tier=name, result=str(folder/'result.json'),
                        reason=result['reason'], exit_code=result['exit_code'],
                        peak_memory_bytes=result.get('peak_aggregate_memory_bytes',0)))
                    record['completed'] += len(result['completed'])
                    for worker in result['workers']:
                        for report in worker['reports']:
                            counts[report['outcome']] += 1
                            if report['outcome'] in ('failed','xpassed'):
                                record['failures'].append(report)
                        for node in set(worker['selected']) - set(worker['completed']):
                            counts['process_failed'] += 1
                            record['failures'].append(dict(nodeid=node, phase='process',
                                outcome='process_failed', reason=worker['reason']))
                    if result['exit_code']:
                        exit_code = result['exit_code']
                        if result['reason'] != 'test_failure':
                            record['failures'].append(dict(tier=name, reason=result['reason']))
                record['counts'] = dict(counts)
                record['duration_seconds'] = time.monotonic() - started
                bounded.write_json(directory/'record.json', record)
            record['complete'] = record['attempted'] == record['selected']
            return exit_code, directory
    finally:
        record['duration_seconds'] = time.monotonic() - started
        record['source_matched'] = bounded.source_snapshot(root) == frozen
        record['exit_code'] = exit_code
        bounded.write_json(directory/'record.json',record)
        bounded.write_json(directory.parent/'latest.json',record)
        for key,value in previous.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value


if __name__ == '__main__':
    code,path = run()
    print(path/'record.json',flush=True)
    raise SystemExit(code)
