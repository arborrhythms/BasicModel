"""Verify the capacity addendum and saved evidence without running a model."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def seeds(value):
    text = value.decode()
    return sorted(ast.get_source_segment(text, node) for node in ast.walk(ast.parse(text))
                  if isinstance(node, ast.Call) and 'seed' in ast.unparse(node.func).lower())


head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == '802abb1acc95e1bddc8cb237b13230a336681c49'
before = read(HERE / 'review14-addendum-before/source.json')
current = source_snapshot(ROOT)
changed = sorted(p for p in before.keys() | current.keys() if before.get(p) != current.get(p))
expected = ['bin/MereologicalCodes.py', 'bin/Spaces.py', 'data/MM_grammar.xml',
            'data/XOR_grammar.xml', 'test/objective_conflicts_probe.py',
            'test/test_derived_concept_codes.py', 'test/test_review13_subspaces.py',
            'test/test_review14_addendum.py']
assert changed == sorted(expected), changed
archive = zipfile.ZipFile(HERE / 'review14-addendum-before/source.zip')
seed_changes = [p for p in changed if p.endswith('.py') and p in before
                and seeds(archive.read(p)) != seeds((ROOT / p).read_bytes())]
assert not seed_changes
assert not seeds((ROOT / 'test/test_review14_addendum.py').read_bytes())
protected = ['test/test_explicit_dimensions.py', 'test/test_mm_xor.py',
             'test/test_reconstruction_roundtrip.py', 'test/bounded_tests.py',
             'test/pytest_worker.py', 'Makefile', 'pytest.ini', 'data/eval/nanochat_grammar_gate.json']
assert all((ROOT / p).read_bytes() == archive.read(p) for p in protected)

capacities = {}
for name, capacity in [('XOR_grammar', 6), ('MM_grammar', 8)]:
    path = f'data/{name}.xml'
    old = ET.fromstring(archive.read(path))
    new = ET.fromstring((ROOT / path).read_bytes())
    old_n, new_n = old.find('ConceptualSpace/nVectors'), new.find('ConceptualSpace/nVectors')
    capacities[name] = dict(before=int(old_n.text), after=int(new_n.text),
                           perceptual=int(new.find('PartSpace/nVectors').text))
    assert int(new_n.text) == capacity and int(old_n.text) == capacity + 256
    new_n.text = old_n.text
    assert ET.tostring(old) == ET.tostring(new), path

plan = 'doc/plans/2026-09-27-item-6-8-one-attention.md'
assert (ROOT / plan).read_bytes() == archive.read(plan)
catalogue = 'doc/plans/2026-09-29-item-6-9-xor-grammar.md'
def distributional_row(text):
    return [line for line in text.splitlines() if line.startswith('| **Distributional pressure on the codes**')]
assert distributional_row((ROOT / catalogue).read_text()) == distributional_row(archive.read(catalogue).decode())

probes = {}
selectors = (HERE / 'review14-addendum-focused-files.txt').read_text().splitlines()
assert selectors[:-1] == (HERE / 'review14-focused-files.txt').read_text().splitlines()
for label in ('before', 'after'):
    folder = HERE / f'probes/review14-addendum-{label}'
    process = read(folder / 'process.json')
    assert process['command'][4:] == selectors
    assert process['reason'] == 'exit' and process['peak_memory_bytes'] <= 8 * 1024**3
    assert process['elapsed_seconds'] < 1800
    probes[label] = process
assert probes['before']['exit_code'] == 1
assert probes['after']['exit_code'] == 0
assert current == read(HERE / 'probes/review14-addendum-after/source.json')

saved = read(HERE / 'review14-delivery/supplement-files.json')
prefix = str((HERE / 'review14-measurements').relative_to(ROOT)) + '/'
measurements = {p: value for p, value in saved.items() if p.startswith(prefix)}
assert measurements and all(digest((ROOT / p).read_bytes()) == value for p, value in measurements.items())
ports = [dict(path=p, old=archive.read(p).decode() if p in before else None,
              new=(ROOT / p).read_text()) for p in changed if p.startswith('test/')]
report = dict(head=head, source_matches_final_probe=True, source_changes=changed,
              capacities=capacities, seed_changes=seed_changes, protected=protected,
              xml_change_only_concept_capacity=True, incoming_review_preserved=True,
              original_measurement_files_unchanged=len(measurements),
              gate_trainings_in_addendum=0, full_sweeps_in_addendum=0,
              focused_files=selectors, probes=probes, complete_old_new_test_ports=ports)
with (HERE / 'review14-addendum-verification.json').open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('complete_old_new_test_ports', 'probes')}))
