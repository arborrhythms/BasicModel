"""Standing-thirty preflight for the exact reviewed §14.13 closing source."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVIEW = HERE.parent/'2026-10-08-math-chain-repair-2'
EXPECTED = '21320d471189ff2e5668cbd95394c8a50da4224338befde3ef8829f590924373'


def validate(source):
    read = lambda path: json.loads(path.read_text())
    assert len(source) == 744
    assert hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() == EXPECTED
    assert source == read(REVIEW/'closing-source-4/source.json')
    assert source == read(HERE/'measured-source/source.json')
    helpers = read(HERE/'measured-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value
               for name, value in helpers.items())
    for directory, cases in (('closing-sweep-3', 5646), ('closing-thinking-4', 57)):
        result = read(REVIEW/directory/'result.json')
        assert result['exit_code'] == 0 and result['reason'] == 'passed'
        assert len(result['completed']) == len(result['selected']) == cases
        assert sorted(result['completed']) == sorted(result['selected'])
        assert source == read(REVIEW/directory/'source-manifest.json')['validated_source']
    retained = read(REVIEW/'stopped-by-decision/retained-files-sha256.json')
    assert all(hashlib.sha256((REVIEW/name).read_bytes()).hexdigest() == value
               for name, value in retained.items())
    return dict(source_sha256=EXPECTED, source_files=744, sweep=True, thinking=57,
                stopped_files_unchanged=len(retained))
