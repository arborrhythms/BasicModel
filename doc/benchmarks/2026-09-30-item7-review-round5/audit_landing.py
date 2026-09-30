"""Verify section 25's test-only delta and the source stored in its commit."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LANDING = HERE / 'landing'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    source = source_snapshot(ROOT)
    assert source == read(LANDING / 'source-manifest.json')
    swept = read(HERE / 'final-source.json')
    changed = sorted(p for p in source.keys() | swept.keys() if source.get(p) != swept.get(p))
    ports = read(LANDING / 'test-ports.json')
    assert changed == sorted(p['file'] for p in ports)
    for port in ports:
        text = (ROOT / port['file']).read_text()
        node = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == port['name'])
        assert ast.get_source_segment(text, node) == port['after']
        assert sha((ROOT / port['file']).read_bytes()) == port['after_file_sha256']
        assert sha((LANDING / 'before-source' / port['file']).read_bytes()) == port['before_file_sha256']
        old = [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(port['before']))
               if isinstance(n, ast.Assert)]
        new = [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(port['after']))
               if isinstance(n, ast.Assert)]
        if port['file'].endswith('test_arithmetic_isolation.py'):
            assert old[1:] == new[2:]
        else:
            assert old == new
    inputs = read(HERE / 'final-inputs.json')
    for path, expected in inputs.items():
        assert sha((ROOT / path).read_bytes()) == expected
    for path, expected in read(HERE / 'scheduling-harness-source.json')['files'].items():
        assert sha((HERE / path).read_bytes()) == expected, path
    results = {}
    for label in ('cases', 'files', 'item7', 'docs'):
        result = read(LANDING / label / 'run/result.json')
        assert result['reason'] == 'passed' and result['exit_code'] == 0
        assert Counter(result['selected']) == Counter(result['completed'])
        assert all(n == 1 for n in Counter(result['completed']).values())
        assert result['limits']['per_worker_memory_bytes'] == 8 * 1024**3
        assert result['limits']['aggregate_memory_bytes'] <= 24 * 1024**3
        assert read(LANDING / label / 'diagnostics.json') == []
        reports = [r for w in result['workers'] for r in w.get('reports', [])]
        assert all(r['outcome'] == 'passed' for r in reports)
        results[label] = dict(passed=len(result['completed']),
                              seconds=result['elapsed_seconds'],
                              receipt=f'{label}/run/result.json')
    assert results['cases']['passed'] == 6
    assert results['files']['passed'] == 59
    assert results['item7']['passed'] == 202
    report = dict(source_files=len(source),
                  source_digest=sha(json.dumps(source, sort_keys=True).encode()),
                  changed_since_sweep=changed, runtime_and_configuration_unchanged=True,
                  all_other_assertions_unchanged=True, results=results,
                  original_full_sweep='../full-sweep/summary.json',
                  no_new_full_sweep=True)
    if len(sys.argv) > 1:
        commit = subprocess.check_output(['git', 'rev-parse', sys.argv[1]], cwd=ROOT, text=True).strip()
        paths = source | inputs
        proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        for path, expected in paths.items():
            proc.stdin.write(f'{commit}:{path}\n'.encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            assert len(header) == 3 and header[1] == 'blob', (path, header)
            content = proc.stdout.read(int(header[2]))
            assert proc.stdout.read(1) == b'\n'
            assert sha(content) == expected, path
        proc.stdin.close()
        assert proc.wait() == 0
        report.update(commit=commit, verified_committed_source_files=len(source),
                      verified_committed_supporting_inputs=len(inputs))
        target = 'committed-source-verification.json'
    else:
        target = 'source-audit.json'
    (LANDING / target).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
