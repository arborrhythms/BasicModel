"""Read-only description of an unsuccessful atomic Part admission."""
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def inspect_part_capacity(monkeypatch, request):
    from PartClosing import PartClosing
    original = PartClosing.preflight

    def preflight(closing):
        try:
            return original(closing)
        except RuntimeError:
            folder = Path(os.environ['PART_DIAGNOSTIC'])/request.node.name
            folder.mkdir(parents=True, exist_ok=True)
            rows = [dict(symbol=r['symbol'], seat=r['seat'], pair=r['pair'],
                order=r['value'].order, reference=r['value'].reference,
                operation=r['value'].operation, types=r['types'])
                for r in closing._plan]
            (folder/'plan.json').write_text(json.dumps(dict(
                seed=torch.initial_seed(), caps=closing.owner._order_caps(),
                nVectors=closing.owner.nVectors, plan=rows), indent=2)+'\n')
            raise
    monkeypatch.setattr(PartClosing, 'preflight', preflight)
