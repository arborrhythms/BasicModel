"""Require the green full sweep on the exact source used by the campaign."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def validate(source):
    read=lambda p:json.loads(p.read_text())
    result=read(HERE/'green-sweep/result.json')
    assert result['exit_code']==0, result['reason']
    assert len(result['completed'])==len(result['selected'])
    assert source==read(HERE/'green-sweep/source-manifest.json')['validated_source']
    assert source==read(HERE/'measured-source/source.json')
    certificate=read(HERE/'repair-certificates.json')
    assert all(r['multisets']==4 and r['residual']==[0.]*4 and r['R']==[0.]*4 for r in certificate['reads'])
    assert not read(HERE/'tetralemma-path-audit.json')['violations']
    return result
