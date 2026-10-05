"""Archive the held §16.3 delivery without running tests or training."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'review16-delivery'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    source = read(HERE / 'review16-source/source.json')
    assert source == source_snapshot(ROOT) == read(OUT / 'source.json')
    assert source == read(HERE / 'review16-green-sweep/source-manifest.json')['validated_source']
    assert not (HERE / 'review16-measurements').exists()
    assert not read(OUT / 'seed-port-audit.json')['changed_seed_calls']
    complete = read(OUT / 'complete-source.json')
    assert all(digest(ROOT / path) == sha for path, sha in complete.items())
    helpers = read(HERE / 'review16-source/measurement-helpers.json')
    assert all(digest(ROOT / path) == sha for path, sha in helpers.items())
    contracts = read(HERE / 'review16-contracts-final.json')
    assert all(contracts['protected'].values()) and not contracts['seed_changes']
    assert all(contracts['output_assertions_unchanged'].values())
    preservation = read(HERE / 'review16-historical-preservation.json')
    assert not preservation['changed']
    summary = read(HERE / 'review16-green-sweep/summary.json')
    assert summary['counts'] == dict(passed=4900, skipped=286, xpassed=1)
    assert summary['exit_code'] == 1 and not summary['full_sweep_green']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == '802abb1acc95e1bddc8cb237b13230a336681c49'

    links = re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text())
    missing = [link for link in links if not link.startswith(('https:', 'http:', '#'))
               and not (HERE / link.split('#')[0]).exists()]
    assert not missing, missing
    files = {ROOT / 'todo.md', HERE / 'README.md', HERE / 'README-before-review16.md',
             ROOT / 'doc/plans/2026-09-29-item-6-9-xor-grammar.md'}
    for path in HERE.iterdir():
        if path.name.startswith('review16') and path != OUT:
            files.update([path] if path.is_file() else (p for p in path.rglob('*') if p.is_file()))
    for folder in (HERE / 'probes').glob('review16*'):
        files.update(p for p in folder.rglob('*') if p.is_file())
    files = sorted(p for p in files if '__pycache__' not in p.parts and p.suffix != '.pyc')
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in files}
    write_new(OUT / 'supplement-files.json', manifest)
    with zipfile.ZipFile(OUT / 'review-supplement.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    bridge = dict(
        head=head, status='implemented; gate campaign held at non-strict XPASS decision',
        full_sweep_counts=summary['counts'], full_sweep_green=False,
        all_pytest_workers_exit_zero=True, bounded_supervisor_exit=1,
        compile_cache_retries=0, gate_trainings=0,
        source_matches_full_sweep_and_frozen_mechanism=True,
        whole_old_new_test_ports=len(read(OUT / 'test-ports.json')),
        round_existing_test_ports=len(contracts['complete_changed_test_ports']),
        changed_seed_calls=0, gate_bars_and_runner_guards_unchanged=True,
        original_output_assertions_unchanged=True, original_output_regressions_passed=16,
        historical_evidence_files_unchanged=preservation['checked'],
        source_files=len(source), complete_files=len(complete),
        receipt_links_verified=len(links), supplement_files=len(files),
        source_zip_sha256=digest(OUT / 'source.zip'),
        supplement_zip_sha256=digest(OUT / 'review-supplement.zip'),
        supplement_manifest_sha256=digest(OUT / 'supplement-files.json'),
        pending='Resolve the named non-strict XPASS under the unchanged green-sweep guard; '
                'then once-only sum, XOR and MM campaigns, reports, and Claude review before commit.')
    write_new(OUT / 'bridge.json', bridge)
    print(json.dumps(bridge))


if __name__ == '__main__':
    main()
