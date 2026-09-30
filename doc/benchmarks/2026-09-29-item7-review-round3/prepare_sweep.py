"""Generate round-3 receipt drivers from the unchanged round-2 protocol."""
from pathlib import Path
p=Path(__file__).resolve().parent
prior=p.parent/'2026-09-28-item7-review-round2'
s=(prior/'run_full_sweep.py').read_text()
a=s.index('    for label in (')
b=s.index('    output.mkdir',a)
s=s[:a]+'''    for label in ('head', 'candidate'):
        processes = json.loads((HERE / (label + '-reconstruction') / 'processes.json').read_text())
        assert set(processes) == set(map(str, range(8)))
        trials = json.loads((HERE / (label + '-mm-grammar') / 'processes.json').read_text())
        assert set(trials) == set(map(str, range(10)))
    assert json.loads((HERE / 'candidate-reconstruction/source-manifest.json').read_text()) == source
    assert json.loads((HERE / 'candidate-mm-grammar/source-manifest.json').read_text())['validated_source'] == source
    assert json.loads((HERE / 'final-xor/source-manifest.json').read_text())['validated_source'] == source
    assert json.loads((HERE / 'graph-release-head/result.json').read_text())['reason'] != 'running'
'''+s[b:]
s=s.replace("validated_source=source, supporting_inputs=inputs,\n        measurement_bridge_sha256=hashlib.sha256((HERE / 'measurement-source-bridge.json').read_bytes()).hexdigest())", "validated_source=source, supporting_inputs=inputs)")
a=s.index('    for path in (PRIOR')
b=s.index('        raw = path.read_bytes()',a)
s=s[:a]+'''    for path in (HERE.parent / '2026-09-28-item7-review-round2/full-sweep/combined-result.json',):
'''+s[b:]
s=s.replace("    os.environ.update(BASICMODEL_DEVICE='cpu'", "    os.environ.pop('BASIC_SEED', None)\n    os.environ.update(BASICMODEL_DEVICE='cpu'")
(p/'run_full_sweep.py').write_text(s)
s=(prior/'continue_sweep.py').read_text().replace('Resource-limited cases stay red and are not run again.', 'Resource-limited cases stay red; one unguarded diagnostic is recorded apart.')
s=s.replace("    os.environ.update(BASICMODEL_DEVICE='cpu'", "    os.environ.pop('BASIC_SEED', None)\n    os.environ.update(BASICMODEL_DEVICE='cpu'")
s=s.replace('import time\n', 'import time\nimport subprocess\n')
anchor='\ndef main():\n'
helper='''
def diagnose(resource_cases):
    path = OUT / 'unguarded-diagnostics.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    for node, failure in resource_cases.items():
        if failure['reason'] not in ('memory', 'aggregate_memory') or node in records:
            continue
        directory = OUT / f'unguarded-{len(records):02}'
        directory.mkdir()
        started = time.monotonic()
        with (directory/'pytest.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', '--tb=short',
                '-p', 'no:cacheprovider', node], cwd=ROOT, env=os.environ.copy(),
                stdout=log, stderr=log, start_new_session=True)
            tree, peak, reason = bounded.ProcessTree(proc.pid), 0, 'completed'
            while proc.poll() is None:
                peak = max(peak, tree.sample())
                if time.monotonic()-started > 1800:
                    tree.terminate(proc,.5)
                    reason = 'timeout'
                    break
                time.sleep(.1)
        records[node] = dict(diagnostic_only=True, memory_guard=None,
            reason=reason, exit_code=proc.returncode, peak_memory_bytes=peak,
            elapsed_seconds=time.monotonic()-started, log=str(directory/'pytest.log'))
        path.write_text(json.dumps(records, indent=2)+'\\n')
        print(json.dumps(dict(diagnostic=node, **records[node])), flush=True)

'''
s=s.replace(anchor,'\n'+helper+anchor)
s=s.replace("    verify()\n    parts, workers, resource_cases, interrupted = read_parts()\n", "    verify()\n    parts, workers, resource_cases, interrupted = read_parts()\n    diagnose(resource_cases)\n    verify()\n")
# Keep the resource diagnostics outside the gate's cumulative dispatch budget.
(p/'continue_sweep.py').write_text(s)
