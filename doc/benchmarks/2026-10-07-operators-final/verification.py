"""Preserve the one original sweep and validate its explicit focused repairs."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def validate(source):
    read=lambda p:json.loads(p.read_text())
    sweep=read(HERE/'full-sweep/result.json')
    assert len(sweep['completed'])==len(sweep['selected'])
    assert read(HERE/'sweep-source/source.json')==read(HERE/'full-sweep/source-manifest.json')['validated_source']
    final=read(HERE/'repair-verification.json')
    assert source==final['source'] and final['failures_addressed']
    for path in final['receipts']:
        result=read(HERE/path/'result.json')
        assert result['exit_code']==0 and len(result['completed'])==len(result['selected'])
        assert source==read(HERE/path/'source-manifest.json')['validated_source']
    return dict(original_sweep=sweep['reason'],focused_repairs=final)
