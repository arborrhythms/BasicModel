"""Retain one isolated rerun of the unchanged observer assertion after its failure.

This is a diagnostic ordinary XOR batch, separate from the eight-branch
experiment. No seed is pinned and no failing assertion is relaxed.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

import Language
import operators_run_audit as audit


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    original_observe = audit.observe_run
    original_forward = Language.OperationSelectionLayer.forward
    active, calls = {}, []

    def forward(module, *args, **kwargs):
        value = original_forward(module, *args, **kwargs)
        logits = value[2]['logits']
        calls.append(dict(epoch=active.get('result', {}).get('_active_epoch', 0),
            grad_enabled=torch.is_grad_enabled(), requires_grad=logits.requires_grad,
            finite=int(torch.isfinite(logits).sum()), shape=list(logits.shape)))
        return value

    @contextmanager
    def observe():
        result = None
        try:
            with original_observe() as result:
                active['result'] = result
                yield result
        finally:
            (out/'calls.json').write_text(json.dumps(calls, indent=2)+'\n')
            if result is not None:
                (out/'audit.json').write_text(json.dumps(result,
                    default=audit.json_value, indent=2)+'\n')
            active.clear()

    source = Path(audit.__file__).with_name('observer_probe.py')
    with patch.object(audit, 'observe_run', observe), patch.object(
            Language.OperationSelectionLayer, 'forward', forward):
        code = pytest.main(['-q', str(source)])
    (out/'result.json').write_text(json.dumps(dict(exit_code=int(code),
        original_assertion=str(source), seed_pinned=False,
        interpretation='A diagnostic rerun never replaces the failed full-sweep result.'), indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
