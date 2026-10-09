"""Freeze the new learning source and all measurement dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded


def main():
    source = bounded.source_snapshot(ROOT)
    result = json.loads((HERE / 'final-sweep/result.json').read_text())
    manifest = json.loads((HERE / 'final-sweep/source-manifest.json').read_text())
    assert result['exit_code'] == 0 and result['reason'] == 'passed'
    assert sorted(result['selected']) == sorted(result['completed'])
    assert source == manifest['validated_source']
    out = HERE / 'measured-source'
    out.mkdir(exist_ok=False)
    (out / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    with zipfile.ZipFile(out / 'source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(ROOT / name, name)
    dependencies = set(HERE.glob('*.py')) | {HERE / 'protocol.json', HERE / 'test-retirements.json',
                                           HERE / 'spec-at-start.txt',
                                           ROOT / 'test/fixtures/when-readers-round4a0.json'}
    dependencies.update((HERE.parent / '2026-10-03-operators-attention').glob('*.py'))
    dependencies.update(HERE.parent / '2026-10-01-item6-9-review' / name
                        for name in ('separator_campaign.py', 'measure.py'))
    dependencies.add(HERE.parent / '2026-09-24-item11c/explicit-result.json.gz')
    helpers = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(dependencies)}
    (out / 'measurement-helpers.json').write_text(json.dumps(helpers, indent=2) + '\n')
    with zipfile.ZipFile(out / 'measurement-helpers.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in helpers:
            archive.write(ROOT / name, name)
    (out / 'tracked-changes.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    protected = ('test/test_explicit_dimensions.py', 'test/test_mm_xor.py',
                 'data/XOR_grammar.xml', 'data/MM_xor.xml')
    equal = {name: (ROOT / name).read_bytes() == subprocess.check_output(
        ['git', 'show', 'e43638a:' + name], cwd=ROOT) for name in protected}
    assert all(equal.values())
    summary = dict(source_files=len(source), protected_matches=equal,
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        helpers=len(helpers), base_commit='e43638a747e373b3b10343c642f76cd1d0757ec1')
    (out / 'freeze.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
