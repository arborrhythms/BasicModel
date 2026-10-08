"""Verify frozen evidence and record later document changes without rerunning it."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-10-07-item6-2'
ARCHIVES = HERE / 'measured-source'
SPEC = 'doc/specs/2026-10-07-thinking.md'


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def verify_archive(manifest, archive):
    with zipfile.ZipFile(archive) as saved:
        assert set(saved.namelist()) == set(manifest)
        for name, expected in manifest.items():
            assert digest(saved.read(name)) == expected, name


def main():
    sys.path[:0] = [str(HERE), str(ROOT / 'test')]
    import bounded_tests
    from verification import validate

    source = bounded_tests.source_snapshot(ROOT)
    validate(source)
    verify_archive(source, ARCHIVES / 'source.zip')
    helpers = read(ARCHIVES / 'measurement-helpers.json')
    verify_archive(helpers, ARCHIVES / 'measurement-helpers.zip')
    assert all(digest((ROOT / name).read_bytes()) == value
               for name, value in helpers.items())
    first = read(HERE / 'first-receipt-hashes.json')
    assert first == {str(p.relative_to(PRIOR)): digest(p.read_bytes())
                     for p in PRIOR.rglob('*') if p.is_file()}
    assert read(HERE / 'thinking-gate/source-manifest.json')['validated_source'] == source
    assert read(HERE / 'measurements/source.json') == source
    assert read(HERE / 'mm-query-configured/plan.json')['source'] == source
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert base == '631d9e8e44b7c8034263e22b36dc74fa5df4eb75'
    freeze = read(ARCHIVES / 'freeze.json')
    for name in freeze['protected_matches']:
        assert (ROOT / name).read_bytes() == subprocess.check_output(
            ['git', 'show', f'{base}:{name}'], cwd=ROOT)
    original = 'test/test_item6_2_thinking.py'
    with zipfile.ZipFile(HERE / 'baseline-source.zip') as archive:
        assert archive.read(original) == (ROOT / original).read_bytes()

    old_manifest_path = ARCHIVES / 'review-documents-before-final-verification.json'
    old_archive_path = ARCHIVES / 'review-documents-before-final-verification.zip'
    assert not old_manifest_path.exists(), 'do not replace the prior review snapshot'
    old_manifest = read(ARCHIVES / 'review-documents.json')
    verify_archive(old_manifest, ARCHIVES / 'review-documents.zip')
    mismatches = [name for name, value in old_manifest.items()
                  if digest((ROOT / name).read_bytes()) != value]
    assert mismatches == [SPEC], mismatches
    shutil.copyfile(ARCHIVES / 'review-documents.json', old_manifest_path)
    shutil.copyfile(ARCHIVES / 'review-documents.zip', old_archive_path)
    with zipfile.ZipFile(old_archive_path) as archive:
        old_spec = archive.read(SPEC)
    observed_spec = (ROOT / SPEC).read_bytes()
    (HERE / 'spec-observed-at-final-verification.txt').write_bytes(observed_spec)
    (HERE / 'spec-review-to-final-verification.patch').write_text(''.join(
        difflib.unified_diff(old_spec.decode().splitlines(True),
                             observed_spec.decode().splitlines(True),
                             fromfile='archived-review/' + SPEC,
                             tofile='observed-at-final-verification/' + SPEC)))
    style = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, text=True,
                           capture_output=True, check=False)
    assert style.returncode == 2
    assert style.stdout.strip() == 'test/test_frozen_concepts.py:120: new blank line at EOF.'
    findings = {
        'source_files_verified': len(source),
        'source_sha256': freeze['source_sha256'],
        'measurement_helpers_verified': len(helpers),
        'first_receipt_files_verified_unchanged': len(first),
        'all_measurements_match_source': True,
        'source_archive_verified': True,
        'helper_archive_verified': True,
        'standing_protected_files_match_6_5': True,
        'original_25_certificate_source_unchanged': True,
        'HEAD_unchanged': base,
        'git_diff_check': {'exit_code': style.returncode, 'output': style.stdout,
                           'disposition': 'Style-only trailing blank line retained in the frozen measured source; deferred to a subsequent authorized source revision.'},
        'spec_revision': {
            'path': SPEC,
            'archived_review_sha256': digest(old_spec),
            'observed_sha256': digest(observed_spec),
            'archived_review_already_has_section_10': b'## 10.' in old_spec,
            'snapshot': 'spec-observed-at-final-verification.txt',
            'diff': 'spec-review-to-final-verification.patch',
            'disposition': 'The spec was revised concurrently during finalization. This receipt measures the recorded repair protocol only. No MM_math_chain gate or amended equality-rewrite contract is claimed.'},
        'training_retries': 0,
        'commit': False,
    }
    write_json(HERE / 'final-verification.json', findings)

    path = HERE / 'README.md'
    text = path.read_text()
    for before, after in {
        "| Original 25 certificates | {'passed': 25} |": '| Original 25 certificates | 25/25 passed |',
        "| Twelve ownership certificates | {'passed': 12} |": '| Twelve ownership certificates | 12/12 passed |',
        "| Ported 11c nodes | {'passed': 17} |": '| Ported 11c nodes | 17/17 passed |',
        "| MM configuration / optimizer smoke | {'passed': 2} |": '| MM configuration / optimizer smoke | 2/2 passed |',
        "| Per-row answer integration | {'failed': 1} |": '| Per-row answer integration | Seed guard failed before model construction |',
        'The configured run has 1 recorded credit\nobservations, 0 with a nonzero raw\ngradient into the shared operation scorer, and 1\nexact cost ties.': 'The configured run has one recorded credit observation: an exact cost tie,\nwith no nonzero raw gradient into the shared operation scorer.',
    }.items():
        assert before in text, before
        text = text.replace(before, after)
    text += '''
## Final verification and concurrent spec revision

The [final verification](final-verification.json) matches the measured source,
all 191 frozen measurement helpers and all 1,632 first-receipt files. The
original 25-certificate source and four standing files remain unchanged.
`git diff --check` reports one trailing blank line in the retired-test file;
it is left in the frozen measured source and recorded for a later revision.

The [earlier review-document archive](measured-source/review-documents-before-final-verification.zip)
already contains an earlier §10. Concurrent changes revised that math-chain
gate and the equality contract; the [observed revision](spec-observed-at-final-verification.txt)
and its [difference](spec-review-to-final-verification.patch) are saved.
This receipt covers the recorded repair protocol. **MM_math_chain and the
amended equality-rewrite contract have not been implemented or measured in
this repair.** The live spec may continue to change independently of these
archived observations.
'''
    path.write_text(text)
    summary = read(HERE / 'review-summary.json')
    summary['final_verification'] = 'final-verification.json'
    summary['spec_revision'] = findings['spec_revision']
    summary['style_findings'] = [findings['git_diff_check']]
    write_json(HERE / 'review-summary.json', summary)

    paths = set(old_manifest)
    for name in ('seal_review.py', 'final-verification.json',
                 'spec-observed-at-final-verification.txt',
                 'spec-review-to-final-verification.patch'):
        paths.add(str((HERE / name).relative_to(ROOT)))
    # Archive one coherent byte snapshot even if the externally edited spec changes again.
    contents = {name: (observed_spec if name == SPEC else (ROOT / name).read_bytes())
                for name in sorted(paths)}
    manifest = {name: digest(data) for name, data in contents.items()}
    write_json(ARCHIVES / 'review-documents.json', manifest)
    with zipfile.ZipFile(ARCHIVES / 'review-documents.zip', 'w',
                         compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    verify_archive(manifest, ARCHIVES / 'review-documents.zip')
    print(json.dumps({'source_files': len(source), 'helpers': len(helpers),
                      'first_receipt_files': len(first),
                      'status': summary['status'],
                      'spec_observed_sha256': digest(observed_spec)}))


if __name__ == '__main__':
    main()
