"""Freeze and check the §14.11 closing source, leaving campaign receipts intact."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent/'2026-10-08-math-chain-repair'
REVISION = '' if len(sys.argv) < 3 else '-'+sys.argv[2]
sys.path[:0] = [str(ROOT/'test'), str(PRIOR)]
from bounded_tests import source_snapshot, main as bounded


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrity():
    retained = json.loads((HERE/'stopped-by-decision/retained-files-sha256.json').read_text())
    assert all(digest(HERE/name) == value for name, value in retained.items())
    contracts = json.loads((HERE/'frozen-contracts.json').read_text())
    assert all(name == 'bin/BindingAnswers.py' or digest(ROOT/name) == value for name, value in contracts.items())
    return dict(retained_files=len(retained), retained_unchanged=True, frozen_contracts_unchanged=True)


def freeze():
    verified = integrity()
    output = HERE/('closing-source'+REVISION)
    output.mkdir(exist_ok=False)
    source = source_snapshot(ROOT)
    original = json.loads((HERE/'measured-source/source.json').read_text())
    changed = [name for name, value in source.items() if original.get(name) != value]
    (output/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    helpers = [HERE/name for name in ('closing_checks.py', 'decomposition_certificate.py',
        'forced_decomposition_grammar.py', 'closing_episode_state.py')]
    helpers.extend((PRIOR/'thinking-gate-plan.json', PRIOR/'thinking_gate_observer.py'))
    helpers.extend(HERE.parent/'2026-10-07-math-chain'/name for name in ('math_observer.py', 'protocol.json'))
    helpers = {str(path.relative_to(ROOT)): digest(path) for path in helpers}
    (output/'helpers.json').write_text(json.dumps(helpers, indent=2)+'\n')
    with zipfile.ZipFile(output/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in (*source, *helpers):
            archive.write(ROOT/name, name)
    patches = []
    with zipfile.ZipFile(HERE/'measured-source/source.zip') as before:
        for name in changed:
            old = before.read(name).decode().splitlines(True) if name in original else []
            patches.extend(difflib.unified_diff(old, (ROOT/name).read_text().splitlines(True),
                fromfile='campaign/'+name, tofile='closing/'+name))
    (output/'source-delta.patch').write_text(''.join(patches))
    report = dict(verified, source_files=len(source), changed=changed, seed=None, retries=0,
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        authority='User authorized minimal general binding/return repair; §14.11 forced demonstration only.',
        learning_claim=False, standing_thirty='Retained as already measured, not rerun.')
    (output/'freeze.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def main(phase):
    if phase == 'freeze':
        return freeze()
    source = json.loads((HERE/('closing-source'+REVISION)/'source.json').read_text())
    assert source_snapshot(ROOT) == source
    for name, value in json.loads((HERE/('closing-source'+REVISION)/'helpers.json').read_text()).items():
        assert digest(ROOT/name) == value, name
    integrity()
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu', OMP_NUM_THREADS='1',
        BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join((str(PRIOR), str(ROOT/'bin'), str(ROOT/'test'))))
    if phase == 'thinking':
        os.environ.update(RUN_SLOW='1', PYTEST_PLUGINS='thinking_gate_observer',
            THINKING_GATE_OUTPUT=str(HERE/('closing-thinking-observations'+REVISION)))
        selectors = json.loads((PRIOR/'thinking-gate-plan.json').read_text())['selectors']
        code, _ = bounded([*selectors, '--workers', '1', '--memory-gib', '8', '--batch-size', '128',
                           '--run-dir', str(HERE/('closing-thinking'+REVISION))])
    elif phase == 'certificates':
        os.environ['RUN_SLOW'] = '1'
        code = subprocess.call([sys.executable, '-m', 'pytest', 'test/test_math_chain_decomposition.py',
            '-q', '--basetemp='+str(HERE/('closing-certificates'+REVISION)),
            '--junitxml='+str(HERE/('closing-certificates'+REVISION+'.xml'))], cwd=ROOT)
    elif phase == 'sweep':
        os.environ.pop('RUN_SLOW', None)
        code, _ = bounded(['--workers', '2', '--memory-gib', '16', '--batch-size', '128',
                           '--run-dir', str(HERE/('closing-sweep'+REVISION))])
    else:
        raise ValueError(phase)
    assert source_snapshot(ROOT) == source
    integrity()
    (HERE/f'closing-{phase}{REVISION}-exit.json').write_text(json.dumps(dict(exit_code=code,
        source_verified=True, retained_verified=True), indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main(sys.argv[1])
