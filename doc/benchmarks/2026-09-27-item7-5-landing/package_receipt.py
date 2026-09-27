"""Bind the accepted fixture patches, complete sweep and prior measurements.

The production/configuration source is unchanged from round 3. Only the two
approved test fixtures differ; retain that exact delta when carrying forward
the measurements and explicit XOR/MM outcomes. Raw receipts remain archived.
"""
import argparse
from collections import Counter
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / '2026-09-27-item7-5-pressure'
BASE_PACKAGER = HERE.parent / '2026-09-26-item7-5/package_receipt.py'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import documentation_snapshot, source_snapshot

spec = importlib.util.spec_from_file_location('original_packager', BASE_PACKAGER)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
read, write, delta = original.read, original.write, original.delta

FIXTURE_PORT = {
    'test/test_compiled_word_chunk.py': {
        'measured': '77613d6c87d336501125d8d4af2c70d5e07bffb468090e971998134e3213711b',
        'current': 'c02fa54cbf07515218425bc76b87c4a0d1045324d59588fd4cdd4fbee453458a',
    },
    'test/test_ltm_consolidation.py': {
        'measured': '5550a6b841a2955c238468656ce5010f53757bf42c41bcf2ee859d6874b962ad',
        'current': '1238c7a4c221999b9096735d792daeeefffae0b1a01f47b276532f59a2745ffe',
    },
}
DEPTH3 = ('test/test_thinking_kernel.py::TestDepth3RelativeEndState::'
          'test_first_trained_read_reaches_depth3_end_state')


def main(before_documentation=False):
    source = source_snapshot(ROOT)
    baseline = read(PREVIOUS / 'source-manifest.json')
    assert delta(baseline, source) == FIXTURE_PORT
    runs = {}
    names = ('fixtures', 'full') if before_documentation else ('fixtures', 'full', 'documentation')
    for name in names:
        result = read(HERE / name / 'result.json')
        assert not result.get('active_workers'), name
        assert Counter(result['selected']) == Counter(result['completed']), name
        assert len(result['completed']) == len(set(result['completed'])), name
        runs[name] = original.package_run(HERE / name, source)
        assert not runs[name]['source_delta'], name
        assert sum(runs[name]['counts'].values()) == len(result['selected']), name
    assert runs['fixtures']['exit_code'] == 0
    if not before_documentation:
        assert runs['documentation']['exit_code'] == 0
    assert runs['full']['exit_code'] == 1
    assert [failure['nodeid'] for failure in runs['full']['failures']] == [DEPTH3]
    assert not runs['full']['compile_cache_retries']
    full = read(HERE / 'full/result.json')
    schedule = read(HERE / 'full-schedule.json')
    assert Counter(node for batch in schedule['batches'] for node in batch) == Counter(full['selected'])
    for path, expected in schedule['historical_inputs'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    prior_full = read(PREVIOUS / 'full/result.json')
    old, new = Counter(prior_full['selected']), Counter(full['selected'])
    selection_delta = read(HERE / 'selection-delta.json')
    assert selection_delta == dict(previous_cases=len(prior_full['selected']),
        current_cases=len(full['selected']), removed=list((old - new).elements()),
        added=list((new - old).elements()))
    if not before_documentation:
        documentation = read(HERE / 'documentation/source-manifest.json')
        assert documentation['recorded_documentation'] == documentation_snapshot(ROOT)

    contracts = read(PREVIOUS / 'unchanged-contracts.json')
    assert all(source[path] == expected['sha256'] for path, expected in contracts.items())
    write(HERE / 'unchanged-contracts.json', contracts)
    measurement_source = read(PREVIOUS / 'measurements/source-manifest.json')
    explicit_source = read(PREVIOUS / 'explicit-verified/source-manifest.json')['validated_source']
    assert measurement_source == explicit_source == baseline
    comparison = read(PREVIOUS / 'reconstruction-comparison.json')
    assert not comparison['source_delta']
    explicit = read(PREVIOUS / 'validation-summary.json')['runs']['explicit-verified']
    carried = dict(receipt=str(PREVIOUS.relative_to(ROOT)),
        source_delta=FIXTURE_PORT, production_configuration_unchanged=True,
        measurements_rerun=False, reconstruction=comparison,
        explicit_xor_mm=explicit,
        historical_mm=dict(cost=.21757, threshold=.20, epochs=900, outcome='failed',
            receipt='doc/benchmarks/2026-09-21-item10/README.md#validation-and-limits'))
    write(HERE / 'carried-measurements.json', carried)

    changes = []
    with tarfile.open(PREVIOUS / 'review-source.tar.gz') as archive:
        assert set(archive.getnames()) == set(baseline)
        for path in sorted(baseline):
            before = archive.extractfile(path).read()
            assert hashlib.sha256(before).hexdigest() == baseline[path], path
            after = (ROOT / path).read_bytes()
            if before != after:
                changes.extend(difflib.unified_diff(before.decode().splitlines(True),
                    after.decode().splitlines(True), fromfile='a/' + path, tofile='b/' + path))
    patch = ''.join(changes)
    approved = ''.join((PREVIOUS / name).read_text() for name in
        ('overflow-fixture-followup.patch', 'user-truth-fixture-followup.patch'))
    assert patch == approved
    (HERE / 'changes-since-pressure-review.patch').write_text(patch)
    with tarfile.open(HERE / 'landing-source.tar.gz', 'w:gz') as archive:
        for path in sorted(source):
            info = archive.gettarinfo(ROOT / path, arcname=path)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / path).open('rb') as stream:
                archive.addfile(info, stream)
    with tarfile.open(HERE / 'landing-source.tar.gz') as archive:
        assert set(archive.getnames()) == set(source)
        for path, expected in source.items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected, path

    specification = ROOT / 'doc/specs/2026-09-26-one-operation-per-round.md'
    assert (HERE / 'accepted-spec.txt').read_bytes() == specification.read_bytes()
    drivers = sorted(HERE.glob('*.py')) + [BASE_PACKAGER, specification,
        HERE / 'accepted-spec.txt', HERE / 'full-schedule.json', HERE / 'selection-delta.json',
        PREVIOUS / 'overflow-fixture-followup.patch', PREVIOUS / 'user-truth-fixture-followup.patch']
    write(HERE / 'receipt-drivers.json', {str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in drivers})
    write(HERE / 'source-manifest.json', source)
    summary = dict(runs=runs, source_files=len(source),
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        approved_fixture_delta=FIXTURE_PORT, approved_patches_applied_exactly=True,
        complete_unique_full_coverage=len(full['selected']),
        protected_contracts_unchanged=True, source_archive_verified=True,
        documentation_snapshot_matches=not before_documentation,
        depth3_campaign='red; unchanged assertion retained',
        learned_utility='unproven',
        deferred_decisions=['nonzero training temperature', 'sentence parsimony/work term'],
        publication=('accepted by Alec after Claude round 3; documentation check pending'
                     if before_documentation else
                     'accepted by Alec after Claude round 3; validated for the reviewed landing'))
    verification = HERE / 'committed-source-verification.json'
    if verification.exists():
        committed = read(verification)
        assert committed['source_sha256'] == summary['source_sha256']
        assert committed['all_committed_source_blobs_match']
        summary['implementation_commit'] = committed['commit']
    write(HERE / 'validation-summary.json', summary)
    print(json.dumps({name: {key: row[key] for key in
        ('exit_code', 'counts', 'completed', 'seconds')} for name, row in runs.items()}, indent=2))
    print(summary['source_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-documentation', action='store_true')
    main(parser.parse_args().before_documentation)
