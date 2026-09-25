"""Package the reviewed-source checks without substituting earlier runs."""
from collections import Counter
import ast
import gzip
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'output/item10-forward-reconstruction'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def receipt(name, source):
    run = RUNS / name
    result = json.loads((run / 'result.json').read_text())
    assert result['exit_code'] == 0, (name, result['reason'])
    assert len(set(result['selected'])) == len(result['selected'])
    assert set(result['selected']) == set(result['completed'])
    manifest = json.loads((run / 'source-manifest.json').read_text())
    assert manifest['validated_source'] == source, name
    reports = []
    for path in sorted(run.glob('worker-*.json')):
        reports.extend(json.loads(path.read_text()).get('reports', []))
    counts = Counter()
    for node in result['selected']:
        matches = [r for r in reports if r['nodeid'] == node]
        calls = [r for r in matches if r['phase'] == 'call']
        exceptions = [r for r in matches if r['outcome'] in ('failed', 'skipped', 'xfailed')]
        final = (calls or exceptions)[-1]
        assert final['outcome'] not in ('failed', 'xpassed'), final
        counts[final['outcome']] += 1
    summary = {key: result[key] for key in ('reason', 'exit_code', 'elapsed_seconds',
               'peak_aggregate_memory_bytes', 'compile_cache_retries', 'limits')}
    summary.update(selected=len(result['selected']), completed=len(result['completed']),
                   outcomes=dict(counts), source_files=len(source),
                   source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest())
    write_json(OUT / f'{name}-summary.json', summary)
    (OUT / f'{name}-result.json.gz').write_bytes(gzip.compress((run / 'result.json').read_bytes(), mtime=0))
    shutil.copyfile(run / 'source-manifest.json', OUT / f'{name}-source-manifest.json')
    shutil.copyfile(RUNS / f'{name}-driver.log', OUT / f'{name}-driver.log')
    logs = b''.join((f'\n{p.name}\n'.encode() + p.read_bytes()) for p in sorted(run.glob('worker-*.log')))
    (OUT / f'{name}-workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
    return summary, reports


def main():
    source = source_snapshot(ROOT)
    summaries, reports = {}, {}
    for name in ('affected-final', 'xor-final', 'explicit-final', 'full'):
        summaries[name], reports[name] = receipt(name, source)
    assert summaries['xor-final']['outcomes'] == {'passed': 2}
    write_json(OUT / 'validation-summary.json', summaries)
    native = []
    for report in reports['full']:
        if report.get('phase') != 'call' or not report['nodeid'].startswith((
                'test/test_grounded_xor.py::test_',
                'test/test_concept_output.py::test_native_cli_')):
            continue
        for line in report.get('stdout', '').splitlines():
            if line.startswith('{'):
                native.append(dict(nodeid=report['nodeid'], result=ast.literal_eval(line)))
    assert len(native) == 9, len(native)
    write_json(OUT / 'native-learning-runs.json', native)
    cli = [dict(nodeid=r['nodeid'], outcome=r['outcome'], stdout=r.get('stdout', ''))
           for r in reports['xor-final'] if r['phase'] == 'call']
    write_json(OUT / 'cli-gates.json', cli)
    # Diagnostics are explicitly separate from the final passing receipt.
    diagnostics = OUT / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for folder in ('before', 'probe', 'native', 'attribution-before', 'native-fixed',
                   'attribution-fixed', 'affected', 'full-before-category', 'category-fixed',
                   'affected-concurrent-abort'):
        for path in (RUNS / folder).glob('*'):
            if path.suffix in ('.json', '.log'):
                (diagnostics / f'{folder}-{path.name}.gz').write_bytes(
                    gzip.compress(path.read_bytes(), mtime=0))
    for name in ('diagnose.py', 'diagnose.log'):
        (diagnostics / f'{name}.gz').write_bytes(gzip.compress((RUNS / name).read_bytes(), mtime=0))
    shutil.copyfile(RUNS / 'category-serial-probe.log', diagnostics / 'category-serial-probe.log')
    serial = RUNS / 'serial'
    manifest = json.loads((serial / 'manifest.json').read_text())
    assert manifest['source_unchanged'] and manifest['source'] == source
    assert all(r['exit_code'] == 0 for r in manifest['completed'])
    destination = OUT / 'serial'
    destination.mkdir(exist_ok=True)
    for path in serial.iterdir():
        if path.suffix == '.log':
            (destination / f'{path.name}.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        elif path.is_file():
            shutil.copyfile(path, destination / path.name)
    write_json(OUT / 'review-source.json', source)
    prior = json.loads((OUT.parent / '2026-09-25-item10-resolution' /
                        'full-source-manifest.json').read_text())['validated_source']
    write_json(OUT / 'source-delta.json', {
        name: dict(before=prior.get(name), after=source.get(name))
        for name in sorted(set(prior) | set(source)) if prior.get(name) != source.get(name)})
    with tarfile.open(OUT / 'review-source.tar.gz', 'w:gz') as archive:
        for name in sorted(source):
            info = archive.gettarinfo(ROOT / name, arcname=name)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / name).open('rb') as handle:
                archive.addfile(info, handle)
    patch = subprocess.check_output(['git', 'diff', '--', 'bin', 'data', 'test'], cwd=ROOT)
    (OUT / 'tracked-source.patch').write_bytes(patch)
    write_json(OUT / 'heads.json', {
        'basicmodel': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'wikioracle': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip()})
    full = summaries['full']
    count = next(re.search(r'reconstruction count: (\d+)/(\d+)', item['stdout'])
                 for item in cli if 'reconstruction count:' in item['stdout'])
    matches, total = map(int, count.groups())
    assert 2 * matches >= total
    validation = (
        f"The single final [full receipt](full-result.json.gz) completes **{full['selected']:,} "
        f"unique cases** with exit zero and no red outcomes. Every check below and "
        f"the serial measurements match the same **{len(source)} source files**, "
        f"SHA-256 of the sorted source map:\n\n`{full['source_sha256']}`.\n\n"
        "| Check | Passed | Skipped | Existing expected failure |\n"
        "| --- | ---: | ---: | ---: |\n")
    for name, summary in summaries.items():
        counts = summary['outcomes']
        validation += (f"| [{name}]({name}-summary.json) | {counts.get('passed', 0):,} | "
                       f"{counts.get('skipped', 0):,} | {counts.get('xfailed', 0)} |\n")
    validation += (
        f"\nThe unseeded [CLI reconstruction gate](cli-gates.json) reaches **{matches}/{total}** "
        "against the fifty-percent bar. Crisp output passes its unchanged MSE < .05 assertion. "
        "The [nine native learning runs](native-learning-runs.json) retain every declared run. "
        "The new serial exclusion check is marked slow and passes in the explicit selection. "
        "The full suite's expected failure is the existing cleared-cache word-overlap probe; "
        "no failing gate is reclassified.\n\n"
        "[Serial reconstruction](serial/comparison.json) after training is **.0891618710**, "
        "against **.0923267286** at `d4dc385`, without a tolerance gate. Pi off/on and frozen "
        "priors agree. [Prior and segmentation measurements](serial/priors.json) preserve "
        "all eight rows over 256 bytes and all 160 runs across 32 sentences.\n\n"
        "The [source archive](review-source.tar.gz), [hash map](review-source.json), "
        "[correction delta](source-delta.json) and [tracked diff](tracked-source.patch) "
        "make the proposed source reviewable. Earlier failing and interrupted attempts "
        "are diagnostics only. A concurrent affected-file attempt encountered a temporary "
        "test-config deletion; final checks ran sequentially after the full suite.\n\n"
        "BasicModel remains at `606683a8e32aac66569669a5e607f64eeec3ae32` and WikiOracle at "
        "`8e7123a1c55ce10ce924d1f27b1e2037dc8890c7`. No commit, push, parent bump or item 9 work.\n")
    readme = OUT / 'README.md'
    placeholder = ("Final counts and source hashes are filled from the completed receipt by\n"
                   "[package_receipt.py](package_receipt.py). The diagnostic attempts remain\n"
                   "separate from the single final full receipt.")
    readme.write_text(readme.read_text().replace(placeholder, validation))
    testing = ROOT / 'doc/Testing.md'
    text = testing.read_text()
    if full['source_sha256'] not in text:
        text += (f"\nThe final source-matched full receipt completes **{full['selected']:,} cases**: "
                 f"**{full['outcomes'].get('passed', 0):,} passed**, "
                 f"**{full['outcomes'].get('skipped', 0):,} skipped** and "
                 f"**{full['outcomes'].get('xfailed', 0)} existing expected failure**, "
                 f"with no red outcomes. All {len(source)} source files match "
                 f"`{full['source_sha256']}` across the full, affected, CLI and explicit "
                 "slow receipts and the serial measurements. The new slow serial exclusion "
                 "check is included in the explicit passing selection. Work stops for review "
                 "before commit, push or parent bump.\n")
        testing.write_text(text)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
