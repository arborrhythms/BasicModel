"""Require complete green coverage on the exact measured source."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent

def validate(source):
    read = lambda path: json.loads(path.read_text())
    result = read(HERE/'retained-sweep/result.json')
    assert result['exit_code'] == 0 and result['reason'] == 'passed', result['reason']
    assert sorted(result['completed']) == sorted(result['selected'])
    assert source == read(HERE/'retained-sweep/source-manifest.json')['validated_source']
    assert source == read(HERE/'measured-source/source.json')
    return result
