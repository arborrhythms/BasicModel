"""Preserve every attempt and bind review evidence to its exact source."""
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
PREVIOUS = HERE.parent / '2026-09-26-item7-5'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

spec = importlib.util.spec_from_file_location('previous_packager', PREVIOUS / 'package_receipt.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
read, write, delta = previous.read, previous.write, previous.delta


def main(partial):
    source = source_snapshot(ROOT)
    runs = {p.name: previous.package_run(p, source) for p in sorted(HERE.iterdir())
            if p.is_dir() and ((p / 'result.json').exists() or (p / 'result.json.gz').exists())}
    required = ('affected-verified', 'explicit-verified') if partial else ('affected-verified', 'explicit-verified', 'full')
    for name in required:
        assert name in runs and not runs[name]['source_delta'], name
        assert runs[name]['selected'] == runs[name]['completed'], name
    # Publish actual outcomes, including any additional failures. A red
    # assertion is evidence, never a passing or expected result by packaging.
    assert runs['affected-verified']['exit_code'] in (0, 1)
    if not partial:
        full = read(HERE / 'full/result.json')
        assert Counter(full['completed']) == Counter(full['selected'])
        assert len(full['completed']) == len(set(full['completed']))
        assert sum(runs['full']['counts'].values()) == len(full['selected'])
    measurements = HERE / 'measurements'
    assert not delta(read(measurements / 'source-manifest.json'), source)
    processes = read(measurements / 'processes.json')
    assert all(p['exit_code'] == 0 for p in processes.values()), processes
    prior = read(PREVIOUS / 'measurements/serial-baseline.json')
    current = read(measurements / 'serial-baseline.json')
    phases = lambda report: {p['name']: p['reconstruction_mean'] for p in report['phases']}
    comparison = dict(source_delta={}, processes=processes,
        serial=dict(reviewed=phases(prior), current=phases(current)),
        training_sentences_per_second=current['phases'][1]['sentences_per_second'],
        batch_timing=read(measurements / 'batch-timing.json'))
    packed, single = [read(measurements / f'{name}.json')['parity'] for name in ('packed', 'single')]
    comparison['parity'] = {k: packed[k] == single[k] for k in
        ('initial_parameters_sha256', 'initial_dictionary_sha256', 'sentences', 'mean_sentence_byte_cost')}
    comparison['byte_cost'] = dict(
        reviewed={k: read(PREVIOUS / f'measurements/{k}.json')['parity']['mean_sentence_byte_cost']
                  for k in ('packed', 'single')},
        current=dict(packed=packed['mean_sentence_byte_cost'], single=single['mean_sentence_byte_cost']))
    write(HERE / 'reconstruction-comparison.json', comparison)
    contracts = read(PREVIOUS / 'unchanged-contracts.json')
    for path, expected in contracts.items():
        assert source[path] == expected['sha256'], path
    write(HERE / 'unchanged-contracts.json', contracts)
    summary = dict(runs=runs, measurements=comparison, source_files=len(source),
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        learned_utility='unproven',
        publication=('per-sentence development evidence; final sweep pending'
                     if partial else 'awaiting Claude review; uncommitted'))
    write(HERE / 'source-manifest.json', source)
    write(HERE / 'validation-summary.json', summary)
    drivers = sorted(HERE.glob('*.py')) + [PREVIOUS / 'package_receipt.py']
    if (HERE / 'full-schedule.json').exists():
        drivers.append(HERE / 'full-schedule.json')
    write(HERE / 'receipt-drivers.json', {str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in drivers})
    baseline = read(HERE / 'reviewed-source-manifest.json')
    changes = []
    with tarfile.open(PREVIOUS / 'review-source.tar.gz') as archive:
        for path in sorted(set(source) | set(baseline)):
            before = archive.extractfile(path).read() if path in baseline else b''
            assert path not in baseline or hashlib.sha256(before).hexdigest() == baseline[path]
            after = (ROOT / path).read_bytes() if path in source else b''
            if before != after:
                changes.extend(difflib.unified_diff(before.decode().splitlines(True),
                    after.decode().splitlines(True), fromfile='a/' + path, tofile='b/' + path))
    (HERE / 'changes-since-review.patch').write_text(''.join(changes))
    with tarfile.open(HERE / 'review-source.tar.gz', 'w:gz') as archive:
        for path in sorted(source):
            info = archive.gettarinfo(ROOT / path, arcname=path)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / path).open('rb') as stream:
                archive.addfile(info, stream)
    print(json.dumps({k: {f: r[f] for f in ('exit_code', 'counts', 'completed')}
                      for k, r in runs.items()}, indent=2))
    print(summary['source_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
