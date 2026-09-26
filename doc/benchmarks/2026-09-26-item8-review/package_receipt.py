"""Reissue the review receipt without replacing the original diagnostic record."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '2026-09-26-item8'
spec = importlib.util.spec_from_file_location('item8_receipt_helpers', OLD / 'package_receipt.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
helpers.HERE = HERE


def main():
    source = helpers.source_snapshot(ROOT)
    summary = {}
    for label, directory in (
        ('red', 'item8-review-red'),
        ('selector-error', 'item8-review-affected'),
        ('affected', 'item8-review-affected-final'),
        ('xor-gates', 'item8-review-xor-gates'),
        ('full', 'item8-review-full')):
        summary[label] = helpers.package_run(label, directory, source, exact=label != 'red')
    assert summary['affected']['exit_code'] == summary['full']['exit_code'] == 0
    assert summary['full']['unique_completed'] == summary['full']['selected']
    if (ROOT / 'output/item8-review-doc-links/result.json').exists():
        summary['documentation'] = helpers.package_run(
            'documentation', 'item8-review-doc-links', source, exact=True)
        assert summary['documentation']['exit_code'] == 0

    run = ROOT / 'output/item8-review-measurements'
    assert json.loads((run / 'source-manifest.json').read_text()) == source
    initial_processes = json.loads((run / 'processes.json').read_text())
    assert initial_processes['serial-baseline']['exit_code'] == 0
    parity_run = ROOT / 'output/item8-review-parity'
    assert json.loads((parity_run / 'source-manifest.json').read_text()) == source
    processes = dict(serial_baseline=initial_processes['serial-baseline'],
                     **json.loads((parity_run / 'processes.json').read_text()))
    assert all(p['exit_code'] == 0 for p in processes.values())
    for path in run.iterdir():
        if path.is_file():
            helpers.compress(path, HERE / 'diagnostics/measurements-import-error' / (path.name + '.gz'))
            if path.name.startswith('serial-baseline.') or path.name == 'routing.json':
                helpers.compress(path, HERE / 'measurements' / (path.name + '.gz'))
    for path in parity_run.iterdir():
        if path.is_file():
            helpers.compress(path, HERE / 'measurements' / (path.name + '.gz'))
    baseline = json.loads((run / 'serial-baseline.json').read_text())
    reviewed = json.loads(gzip.decompress((HERE.parent /
        '2026-09-26-item9b-occurrence-fix/measurements/serial-baseline.json.gz').read_bytes()))
    phases = lambda r: {p['name']:p['reconstruction_mean'] for p in r['phases']}
    assert phases(baseline) == phases(reviewed)
    packed, single = [json.loads((parity_run / f'{layout}.json').read_text())['parity']
                      for layout in ('packed', 'single')]
    fields = ('initial_parameters_sha256', 'initial_dictionary_sha256',
              'sentences', 'mean_sentence_byte_cost')
    parity = {key:packed[key] == single[key] for key in fields}
    assert all(parity.values())
    initial = json.loads(gzip.decompress((OLD / 'measurements/packed.json.gz').read_bytes()))['parity']
    assert all(packed[key] == initial[key] for key in fields)
    helpers.write(HERE / 'reconstruction-comparison.json', dict(
        baseline=phases(baseline), reviewed_9b_exact=True, parity=parity,
        original_candidate_exact=True, byte_cost=packed['mean_sentence_byte_cost'],
        training_sentences_per_second=baseline['phases'][1]['sentences_per_second']))
    summary['measurements'] = dict(source_delta={}, processes=processes,
        initial_import_failures=initial_processes)
    original_source = json.loads(gzip.decompress((OLD / 'measurements/source-manifest.json.gz').read_bytes()))
    summary['retained_seed_diagnostics'] = dict(
        receipt='../2026-09-26-item8/xor-summary.json', rerun=False,
        source_delta={p:dict(measured=original_source.get(p), current=source.get(p))
            for p in sorted(set(source) | set(original_source))
            if source.get(p) != original_source.get(p)},
        seed_1='incomplete: original 8 GiB memory stop, not replaced')
    summary.update(source_files=len(source),
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        learned_utility='unproven', opaque_routing_decline='unproven')
    helpers.write(HERE / 'source-manifest.json', source)
    helpers.write(HERE / 'validation-summary.json', summary)
    helpers.write(HERE / 'receipt-drivers.json', {
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), HERE / 'run_measurements.py', OLD / 'package_receipt.py',
                  OLD / 'PROTOCOL.md', OLD / 'measure.py')})
    with tarfile.open(HERE / 'review-source.tar.gz', 'w:gz') as archive:
        for path in sorted(source):
            info = archive.gettarinfo(ROOT / path, arcname=path)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / path).open('rb') as stream:
                archive.addfile(info, stream)
    patch = subprocess.check_output(['git', 'diff', '--', 'bin', 'test', 'data'], cwd=ROOT)
    for path in ('bin/GrammarPreference.py', 'bin/GrammarEvidence.py',
                 'test/test_structural_preference.py', 'test/test_arithmetic_isolation.py'):
        patch += subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', path],
                                cwd=ROOT, capture_output=True).stdout
    (HERE / 'changes.patch').write_bytes(patch)
    print(json.dumps({key:{k:summary[key][k] for k in ('exit_code', 'counts', 'completed')}
                      for key in ('affected', 'xor-gates', 'full')}))
    print(summary['source_sha256'])


if __name__ == '__main__':
    main()
