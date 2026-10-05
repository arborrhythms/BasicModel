"""Archive §17's measured delivery; never train, retry, commit or push."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'review17-delivery'
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


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def main():
    source = read(HERE / 'review17-final-source/source.json')
    assert source == source_snapshot(ROOT) == read(OUT / 'source.json')
    assert source == read(HERE / 'review17-final-sweep/source-manifest.json')['validated_source']
    assert not read(OUT / 'seed-port-audit.json')['changed_seed_calls']
    complete = read(OUT / 'complete-source.json')
    assert all(digest(ROOT / path) == sha for path, sha in complete.items())
    helpers = read(HERE / 'review17-final-source/measurement-helpers.json')
    assert all(digest(ROOT / path) == sha for path, sha in helpers.items())
    contracts = read(HERE / 'review17-contracts-final.json')
    assert all(contracts['protected'].values()) and not contracts['changed_seed_calls']
    assert all(contracts['unchanged_methods'].values())
    assert all(contracts['original_output_assertions_unchanged'].values())
    assert contracts['guard_change_only_non_strict_xpass']
    summary = read(HERE / 'review17-final-sweep/summary.json')
    assert summary['exit_code'] == 0 and not summary['failures']
    assert summary['selected'] == summary['completed']
    assert all(value == 'passed' for value in summary['original_output_regressions'].values())
    measurement = read(HERE / 'review17-measurements/summary.json')
    assert measurement['complete']['completed'] and measurement['complete']['source_matched']
    head = git('rev-parse', 'HEAD')
    assert head == 'eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d'
    assert git('branch', '--show-current') == 'main'
    assert git('rev-parse', '6.8-s16.3-candidate^{}') == head
    parent = git('ls-files', '--stage', 'basicmodel', cwd=ROOT.parent)
    assert parent == '160000 802abb1acc95e1bddc8cb237b13230a336681c49 0\tbasicmodel'

    mutable = {'todo.md', 'doc/GradientFlow.md', str((HERE / 'README.md').relative_to(ROOT)),
               'doc/plans/2026-09-29-item-6-9-xor-grammar.md'}
    history = read(HERE / 'review16-historical-preservation.json')['files']
    history.update(read(HERE / 'review16-delivery/supplement-files.json'))
    history = {name: sha for name, sha in history.items() if name not in mutable}
    changed = [name for name, sha in history.items() if digest(ROOT / name) != sha]
    write_new(HERE / 'review17-historical-preservation.json',
              dict(checked=len(history), changed=changed, excluded_mutable_files=sorted(mutable), files=history))
    assert not changed, changed
    links = re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text())
    missing = [link for link in links if not link.startswith(('https:', 'http:', '#'))
               and not (HERE / link.split('#')[0]).exists()]
    assert not missing, missing

    files = {ROOT / name for name in mutable}
    files.add(HERE / 'README-before-review17.md')
    for path in HERE.iterdir():
        if path.name.startswith('review17') and path != OUT:
            files.update([path] if path.is_file() else (p for p in path.rglob('*') if p.is_file()))
    for folder in (HERE / 'probes').glob('review17*'):
        files.update(p for p in folder.rglob('*') if p.is_file())
    files = sorted(p for p in files if '__pycache__' not in p.parts and p.suffix != '.pyc')
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in files}
    write_new(OUT / 'supplement-files.json', manifest)
    with zipfile.ZipFile(OUT / 'review-supplement.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    bridge = dict(candidate_commit=head, candidate_tag='6.8-s16.3-candidate',
        status='§17 measured; new work uncommitted, awaiting review; nothing pushed',
        parent_index=parent, full_sweep_counts=summary['counts'], full_sweep_green=True,
        gate_counts=measurement['counts'], gate_trainings=30, gate_retries=0,
        source_matches_full_sweep_and_measurement=True,
        whole_old_new_test_ports=len(read(OUT / 'test-ports.json')), changed_seed_calls=0,
        gate_bars_unchanged=True, resource_guards_unchanged=True,
        guard_change='Non-strict XPASS counts as pass; strict XPASS still fails.',
        original_output_regressions_passed=len(summary['original_output_regressions']),
        historical_evidence_files_unchanged=len(history), source_files=len(source),
        complete_files=len(complete), receipt_links_verified=len(links), supplement_files=len(files),
        source_zip_sha256=digest(OUT / 'source.zip'),
        supplement_zip_sha256=digest(OUT / 'review-supplement.zip'),
        supplement_manifest_sha256=digest(OUT / 'supplement-files.json'))
    write_new(OUT / 'bridge.json', bridge)
    print(json.dumps(bridge))


if __name__ == '__main__':
    main()
