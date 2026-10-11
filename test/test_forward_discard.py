"""Forward meaning loss pays locally; inverse undefinedness is unchanged."""
from types import SimpleNamespace

import torch

from SentenceCredit import forward_discard


def test_retained_four_clauses_keep_the_trial_skipping_one_meaning_removing_non():
    from pathlib import Path
    from SentenceCredit import comparison
    saved = torch.load(Path(__file__).resolve().parents[1] /
        'doc/benchmarks/2026-10-10-item6/identity-r2/validation/retained-trials.pt',
        map_location='cpu', weights_only=True)
    parts = []
    for trial in saved['trials']:
        programs = [SimpleNamespace(**p) for p in trial['programs']]
        charge = forward_discard(programs, saved['meaning_start'], trial['other_cost'])
        torch.testing.assert_close(charge, trial['discard'])
        torch.testing.assert_close(trial['other_cost'], torch.full((4,), 1.85))
        cost = trial['other_cost'] + charge
        parts.append(torch.stack((cost, torch.zeros_like(cost), torch.zeros_like(cost)), -1))
    result = comparison(torch.stack(parts, 1), torch.ones(4, dtype=torch.bool))
    assert result['keep_decision'] == ['explore'] * 4
    assert result['advantage'].lt(0).all()
