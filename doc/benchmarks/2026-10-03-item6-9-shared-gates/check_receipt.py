"""Check the saved campaign and source; never construct or train a model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded


def read(name):
    return json.loads((HERE / name).read_text())


source = read('source-final.json')
assert source == bounded.source_snapshot(ROOT)
start = read('start.json')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == start['head']
frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
(HERE / 'environment-final-freeze.txt').write_bytes(frozen)
assert frozen == (HERE / 'environment-freeze.txt').read_bytes()
for port in read('ports.json'):
    for label in ('old', 'new'):
        assert hashlib.sha256((HERE / port[label]).read_bytes()).hexdigest() == port[label + '_sha256']
    assert source[port['file']] == port['new_sha256']
results = read('results.json')
xor = [row for row in results['gates'] if row['gate'] == 'xor']
assert len(xor) == 10 and all(row.get('same_model') for row in xor)
assert all(row['both_bar'] == (row['class_bar'] and row['reconstruction_bar']) for row in xor)
jobs = read('measurements/complete.json')['jobs']
assert len(jobs) == 42
assert sum(job['kind'] == 'xor' for job in jobs) == 10
assert sum(job['kind'] == 'sum' for job in jobs) == 10
assert sum(job['kind'] == 'mm' for job in jobs) == 10
table = {row['gate']: row for row in results['xor_table']}
for gate in (5, 6):
    assert len(table[gate]['cases']) == 1
    assert table[gate]['process'] == xor[0]['process']
counts = results['counts']
required = counts['class']['passed'] < 8 or counts['reconstruction']['passed'] <= 3
assert results['attribution_decision']['required'] == required
assert len(results['attribution']) == (40 if required else 0)
assert results['complete']['source_matched']
assert results['native_process']['exit_code'] == 0
assert results['extras']['source_matched'] and not results['extras']['pending']
sweep = read('full-sweep/case-summary.json')
assert sweep['source_matched'] and sweep['no_unattempted_cases']
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
record = dict(source_files=len(source), source_matched=True, environment_unchanged=True,
              head_unchanged=True, complete_port_pairs=len(read('ports.json')),
              xor_trainings=10, both_bars_on_same_model=True, named_jobs=len(jobs),
              first_training_supplies_both_table_rows=True, attribution_required=required,
              attribution_trainings=len(results['attribution']), diff_check=True)
(HERE / 'receipt-integrity.json').write_text(json.dumps(record, indent=2) + '\n')
(HERE / 'documentation-final.json').write_text(json.dumps(bounded.documentation_snapshot(ROOT), indent=2))
print(json.dumps(record))
