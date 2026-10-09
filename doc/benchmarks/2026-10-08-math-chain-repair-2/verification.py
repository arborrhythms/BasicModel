"""Refuse measurement unless its declaration and exact-source checks hold."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def validate(source, *, gates=False):
    read = lambda path: json.loads(path.read_text())
    protocol = read(HERE/'protocol.json')
    development = read(HERE/protocol['development_certificate_folder']/'result.json')
    acceptance = read(HERE/protocol['development_acceptance'])
    assert development['status']=='completed' and acceptance['passed']
    assert acceptance['result_sha256'] == hashlib.sha256(
        (HERE/protocol['development_certificate_folder']/'result.json').read_bytes()).hexdigest()
    assert source == read(HERE/'measured-source/source.json')
    helpers = read(HERE/'measured-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
               for name,digest in helpers.items())
    result = read(HERE/'final-sweep/result.json')
    assert result['exit_code']==0 and result['reason']=='passed'
    assert sorted(result['completed'])==sorted(result['selected'])
    assert source==read(HERE/'final-sweep/source-manifest.json')['validated_source']
    if gates:
        thinking = read(HERE/'thinking-gate/result.json')
        assert thinking['exit_code']==0 and thinking['reason']=='passed'
        assert len(thinking['selected'])==len(thinking['completed'])==57
        assert source==read(HERE/'thinking-gate/source-manifest.json')['validated_source']
        standing = read(HERE/'standing-thirty-verification.json')
        assert standing['attempts']==30 and standing['verified']
        assert standing['source']==source
    return result
