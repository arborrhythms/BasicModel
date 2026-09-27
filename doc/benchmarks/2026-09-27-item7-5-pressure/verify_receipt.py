"""Verify the source, measurements, complete coverage and final review artifacts."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import documentation_snapshot, source_snapshot


def read(path):
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes()))


def main():
    source = source_snapshot(ROOT)
    assert source == read(HERE / 'source-manifest.json')
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    summary = read(HERE / 'validation-summary.json')
    assert summary['source_sha256'] == digest
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == '206a01468d424338a0966b5ec082278b3175241b'
    for name in ('feedback-regressions', 'explicit-verified', 'full', 'documentation'):
        run = read(HERE / name / 'result.json')
        assert not run.get('active_workers'), name
        manifest = read(HERE / name / 'source-manifest.json')
        assert manifest['validated_source'] == source, name
        assert Counter(run['selected']) == Counter(run['completed']), name
        assert len(run['completed']) == len(set(run['completed'])), name
        assert not summary['runs'][name]['source_delta'], name
    full = read(HERE / 'full/result.json')
    assert sum(summary['runs']['full']['counts'].values()) == len(full['selected'])
    schedule = read(HERE / 'full-schedule.json')
    assert schedule['expected_cases'] == len(full['selected'])
    assert Counter(node for batch in schedule['batches'] for node in batch) == Counter(full['selected'])
    for path, expected in schedule['historical_inputs'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    documentation = read(HERE / 'documentation/source-manifest.json')
    assert documentation['recorded_documentation'] == documentation_snapshot(ROOT)
    assert summary['runs']['documentation']['exit_code'] == 0
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == row['sha256']
               for path, row in read(HERE / 'unchanged-contracts.json').items())
    with tarfile.open(HERE / 'review-source.tar.gz') as archive:
        assert set(archive.getnames()) == set(source)
        for path, expected in source.items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected, path
    for manifest in ('receipt-drivers.json', 'measurements/drivers.json'):
        for path, expected in read(HERE / manifest).items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert source == read(HERE / 'measurements/source-manifest.json')
    processes = read(HERE / 'measurements/processes.json')
    assert all(p['exit_code'] == 0 and p['reason'] == 'exit' for p in processes.values())
    comparison = read(HERE / 'reconstruction-comparison.json')
    assert not comparison['source_delta']
    spec = ROOT / 'doc/specs/2026-09-26-one-operation-per-round.md'
    assert (HERE / 'reviewed-spec.md').read_bytes() == spec.read_bytes()
    integrity = dict(head=head, uncommitted=True, source_files=len(source),
        source_sha256=digest, final_runs_share_source=True, source_archive_verified=True,
        documentation_snapshot_matches=True,
        documentation_passed=summary['runs']['documentation']['counts']['passed'],
        full_unique_coverage=len(full['selected']), full_counts=summary['runs']['full']['counts'],
        full_seconds=full['elapsed_seconds'], protected_contracts_unchanged=True,
        measurement_processes_passed=True,
        packed_single_records_equal=comparison['parity']['sentences'],
        unapplied_fixture_followups=summary['unapplied_fixture_followups'],
        receipt_driver_hashes_verified=True, reviewed_spec_exact=True)
    (HERE / 'review-integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')
    print(json.dumps(integrity, indent=2))


if __name__ == '__main__':
    main()
