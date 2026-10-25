"""Observed chooser calls need not own the sampled policy gradient."""
from pathlib import Path

import torch


def test_observer_counts_detached_compose_calls(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'test/fixtures/2026-10-03-operators-attention'))
    monkeypatch.syspath_prepend(str(root / 'test/fixtures/2026-10-07-operators-final'))
    from round2_score_probe import observe_score_function
    from Language import OperationSelectionLayer
    layer = OperationSelectionLayer(d_model=4, chooser='mlp')
    result = {'_active_epoch': 1}
    # This is the native compose setting when the sampled departure belongs
    # to narrowing. The chooser runs, but its logits own no policy gradient.
    with observe_score_function(result):
        _, _, route = layer(torch.ones(2, 1, 4), policy_grad=False)
    assert not route['logits'].requires_grad
    measured = result['chooser_logit_ranges']['1']
    assert measured['calls'] == 1 and measured['gradient_calls'] == 0
    assert measured['minimum'] <= measured['maximum']
