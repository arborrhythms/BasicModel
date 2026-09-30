"""Bind every result to its actual source; preserve failures and raw receipts."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes()))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    source = source_snapshot(ROOT)
    spec = importlib.util.spec_from_file_location('prior_packager',
        HERE.parent / '2026-09-26-item7-5/package_receipt.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runs = {}
    for name in ('affected', 'explicit', 'explicit-verified', 'full', 'documentation'):
        if name == 'documentation' and not (HERE / name).exists():
            continue
        runs[name] = helper.package_run(HERE / name, source)
        assert not runs[name]['source_delta'], name
        result = read(HERE / name / 'result.json')
        assert Counter(result['selected']) == Counter(result['completed']), name
        assert len(result['completed']) == len(set(result['completed'])), name
    assert runs['affected']['exit_code'] == 0
    if 'documentation' in runs:
        assert runs['documentation']['exit_code'] == 0
    assert sum(runs['full']['counts'].values()) == runs['full']['selected']
    if (HERE / 'documentation-precommit').exists():
        runs['documentation-precommit'] = helper.package_run(HERE / 'documentation-precommit', source)
        assert runs['documentation-precommit']['exit_code'] == 0
    development = HERE / 'development/affected-r6'
    if development.exists():
        runs['development-affected-r6'] = helper.package_run(development, source)
    for name in ('affected', 'fixture-followup'):
        development = HERE / 'development/before-presented-form' / name
        if development.exists():
            runs['development-before-form-' + name] = helper.package_run(development, source)
    measured = read(HERE / 'measurements/source-manifest.json')
    assert measured == source
    identity = read(HERE / 'identity.json')
    assert identity['source_manifest'] == source
    assert identity['driver_sha256'] == hashlib.sha256(
        (HERE / 'measure_identity.py').read_bytes()).hexdigest()
    serial = read(HERE / 'measurements/serial-baseline.json')
    preceding = read(HERE.parent / '2026-09-27-item7-5-pressure/measurements/serial-baseline.json')
    reviewed = read(HERE.parent / '2026-09-26-item8-review/measurements/serial-baseline.json')
    phases = lambda report: {p['name']: p['reconstruction_mean'] for p in report['phases']}
    packed, single = [read(HERE / f'measurements/{name}.json')['parity'] for name in ('packed', 'single')]
    comparison = dict(reviewed_baseline=phases(reviewed), preceding_7_5=phases(preceding),
        current=phases(serial), rebaselined=False,
        training_sentences_per_second=serial['phases'][1]['sentences_per_second'],
        parity={key: packed[key] == single[key] for key in (
            'initial_parameters_sha256', 'initial_dictionary_sha256', 'sentences', 'mean_sentence_byte_cost')},
        byte_cost=dict(reviewed=.6839025616645813, preceding_7_5=.18002260848879814,
                       packed=packed['mean_sentence_byte_cost'], single=single['mean_sentence_byte_cost']))
    comparison['sentence_counts'] = dict(packed=len(packed['sentences']),
                                          single=len(single['sentences']))
    comparison['truncation'] = dict(
        packed=[step['truncated'] for step in packed['steps']],
        single=[step['truncated'] for step in single['steps']])
    comparison['sentence_differences'] = [dict(index=i,
        differing_fields=sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k)),
        packed_byte_cost=left['byte_cost'], single_byte_cost=right['byte_cost'],
        byte_cost_delta=right['byte_cost'] - left['byte_cost'])
        for i, (left, right) in enumerate(zip(packed['sentences'], single['sentences']))
        if left != right]
    write('reconstruction-comparison.json', comparison)
    processes = read(HERE / 'measurements/processes.json')
    assert all(result['exit_code'] == 0 for result in processes.values())
    baseline = read(HERE.parent / '2026-09-27-item7-5-landing/full/result.json')
    current = read(HERE / 'full/result.json')
    grouped, failed_nodes = {}, {}
    for failure in runs['full']['failures']:
        message = str(failure['message'])
        lines = [line for line in message.splitlines() if line.startswith('E ')]
        failed_nodes.setdefault(failure['nodeid'], lines[0] if lines else message[-200:])
    for node, error in failed_nodes.items():
        grouped.setdefault(error, []).append(node)
    assert len(failed_nodes) == runs['full']['counts'].get('failed', 0)
    write('full-failure-groups.json', dict(
        grouping='First reported exception/assertion line; not a root-cause classification',
        cases=len(failed_nodes), groups=[dict(error=error, count=len(nodes), nodes=sorted(nodes))
            for error, nodes in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))]))
    write('selection-delta.json', dict(
        added=sorted(set(current['selected']) - set(baseline['selected'])),
        removed=sorted(set(baseline['selected']) - set(current['selected']))))
    protected = ('data/XOR_grammar.xml', 'data/MM_grammar.xml', 'test/test_mm_xor.py',
                 'test/test_explicit_dimensions.py', 'test/test_thinking_kernel.py',
                 'test/test_stm_relative_sentence_end_state.py', 'pytest.ini')
    contracts = {}
    for name in protected:
        raw = (ROOT / name).read_bytes()
        assert raw == subprocess.check_output(['git', 'show', f'1678ee1f:{name}'], cwd=ROOT), name
        contracts[name] = hashlib.sha256(raw).hexdigest()
    write('unchanged-contracts.json', contracts)
    write('source-manifest.json', source)
    with tarfile.open(HERE / 'review-source.tar.gz', 'w:gz') as archive:
        for name in sorted(source):
            archive.add(ROOT / name, arcname=name)
    with tarfile.open(HERE / 'review-source.tar.gz') as archive:
        assert set(archive.getnames()) == set(source)
        for name, digest in source.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
    drivers = [p for p in HERE.glob('*.py')]
    write('receipt-drivers.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(drivers)})
    for name, digest in read(HERE / 'measurements/drivers.json').items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    summary = dict(runs=runs, source_files=len(source),
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        measurements=comparison, measurement_processes=processes,
        identity={k: v for k, v in identity.items() if k != 'source_manifest'},
        publication='Uncommitted, incomplete item 7 candidate; stop for Claude review',
        documentation_pending='documentation' not in runs,
        learned_utility='Not established by forced mechanism tests or the small predictor measurement')
    write('validation-summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k in ('source_files', 'source_sha256', 'publication')}))


if __name__ == '__main__':
    main()
