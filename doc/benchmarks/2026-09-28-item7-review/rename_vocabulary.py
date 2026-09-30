"""Step 1 only: reversible spelling substitution, without design edits.

Historical benchmark paths and benchmark contents retain their original names.
The full token map and before/after files make the mechanical change reviewable
independently of the subsequent implementation.
"""
import ast
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / 'rename'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

SPECIAL = {
    'ClauseSeal': 'ClauseRow', 'ClauseSeals': 'ClauseRows',
    'seal_clause': 'write_clause', '_sentence_seals': '_sentence_ends',
    '_choice_seals': '_choice_ends',
    '_last_operator_gradient_seals': '_last_operator_gradient_sentences',
    '_run_sealed_word_bricks': '_run_sentence_word_bricks',
    '_sealed_clause_state': '_clause_end_state',
    'already_sealed': 'already_ended', 'record_seal': 'record_clause_end',
    'choose_sentence_seal_binary': 'choose_sentence_closing_binary',
    'sealed': 'ended', 'seal': 'closing', 'seals': 'endings',
    'sealing': 'ending', 'unseal': 'unfold',
    'Seal': 'Closing', 'Sealed': 'End', 'Seals': 'Endings', 'SEAL': 'ENDING',
}


def rename_token(token):
    if token in SPECIAL:
        return SPECIAL[token]
    parts = token.split('_')
    if any(p in ('seal', 'sealed', 'seals', 'sealing', 'unseal') for p in parts):
        parts = [{'seal': 'closing', 'sealed': 'end', 'seals': 'ends',
                  'sealing': 'ending', 'unseal': 'unfold'}.get(p, p) for p in parts]
        return '_'.join(parts)
    return token


def substitute(text, mapping):
    # A receipt's location is immutable even when its surrounding prose changes.
    historical_path = r'(?:[.\w/-]*benchmarks/)[^\s)\]"<>`]+'
    protected = {}
    def protect(match):
        key = f'VOCABULARYHISTORICALPATH{len(protected)}'
        protected[key] = match.group()
        return key
    text = re.sub(historical_path, protect, text)
    text = re.sub(r'[A-Za-z_][A-Za-z_0-9]*', lambda m: mapping.get(m.group(), m.group()), text)
    return re.sub(r'VOCABULARYHISTORICALPATH\d+',
                  lambda match: protected[match.group()], text)


def main():
    before = source_snapshot(ROOT)
    baseline_path = ROOT / 'doc/benchmarks/2026-09-27-item7/full/source-manifest.json.gz'
    baseline = json.loads(gzip.decompress(baseline_path.read_bytes()))['validated_source']
    assert before == baseline, 'The existing full sweep must match the pre-rename source exactly.'
    paths = [ROOT / 'todo.md', ROOT / 'README.md']
    for directory in ('bin', 'test', 'data', 'doc'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.md', '.xml', '.xsd', '.grammar', '.json', '.svg')
                     and 'benchmarks' not in p.relative_to(ROOT).parts)
    original = {str(p.relative_to(ROOT)): p.read_text() for p in sorted(set(paths))}
    tokens = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', '\n'.join(original.values())))
    mapping = {t: rename_token(t) for t in sorted(tokens) if rename_token(t) != t}
    filenames = {'bin/ClauseSeal.py': 'bin/ClauseRow.py',
                 'test/test_seal_gradient_diagnostics.py': 'test/test_closing_gradient_diagnostics.py'}
    changed = {name: substitute(text, mapping) for name, text in original.items()
               if substitute(text, mapping) != text}
    with tarfile.open(HERE / 'before-files.tar.gz', 'w:gz') as archive:
        for name in sorted(changed):
            archive.add(ROOT / name, arcname=name)
    patch = []
    for name, text in changed.items():
        destination = filenames.get(name, name)
        patch.extend(difflib.unified_diff(original[name].splitlines(True), text.splitlines(True),
                                         fromfile='a/' + name, tofile='b/' + destination))
        (ROOT / destination).write_text(text)
        if destination != name:
            (ROOT / name).unlink()
        if destination.endswith('.py'):
            ast.parse(text, filename=destination)
    (HERE / 'mechanical.patch').write_text(''.join(patch))
    report = dict(baseline='doc/benchmarks/2026-09-27-item7/full/result.json.gz',
                  baseline_source_exactly_matches=True, token_map=mapping, file_map=filenames,
                  changed_files=sorted(changed), before_source=before, after_source=source_snapshot(ROOT),
                  policy='No seed overrides; unchanged full-sweep selection and resource protocol.')
    (HERE / 'mapping.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(files=len(changed), tokens=len(mapping), baseline_source_exactly_matches=True)))


if __name__ == '__main__':
    main()
