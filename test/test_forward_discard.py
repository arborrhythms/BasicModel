"""Forward meaning loss pays locally; inverse undefinedness is unchanged."""
from types import SimpleNamespace

import torch

from SentenceCredit import forward_discard


def test_retained_four_clauses_keep_the_trial_skipping_one_meaning_removing_non(tmp_path, monkeypatch):
    from pathlib import Path
    from identity_experiment import make_model
    from DerivationReconstruction import inverse_step
    from SentenceCredit import comparison
    saved = torch.load(Path(__file__).resolve().parents[1] /
        'doc/benchmarks/2026-10-10-item6/identity-r2/validation/retained-trials.pt',
        map_location='cpu', weights_only=True)
    model = make_model(tmp_path, monkeypatch)
    parts = []
    try:
        for trial, expected in zip(saved['trials'], (3., 2.), strict=True):
            programs = [SimpleNamespace(**p) for p in trial['programs']]
            charge = forward_discard(model.languageSpace, programs, saved['meaning_start'], trial['other_cost'])
            torch.testing.assert_close(charge, torch.full((4,), expected))
            torch.testing.assert_close(trial['other_cost'], torch.full((4,), 1.85))
            cost = trial['other_cost'] + charge
            parts.append(torch.stack((cost, torch.zeros_like(cost), torch.zeros_like(cost)), -1))
        # A lower step remains locally defined even below an undefined non.
        program = programs[0]
        index = int((program.actions[:, 0] == 1).nonzero()[0])
        kind, op, _ = program.actions[index].tolist()
        frame = program.operation_values[index]
        local = inverse_step(model.languageSpace, kind, op, frame[2:3], frame)
        inherited = inverse_step(model.languageSpace, kind, op, frame[2:3], frame,
            coordinates=torch.zeros_like(frame[2:3], dtype=torch.bool))
        assert all(domain.all() for _, domain in local)
        assert all(not domain.any() for _, domain in inherited)
    finally:
        model.End()
    result = comparison(torch.stack(parts, 1), torch.ones(4, dtype=torch.bool))
    assert result['keep_decision'] == ['explore'] * 4
    assert result['advantage'].lt(0).all()


def test_exact_or_reading_pays_zero_forward_discard(monkeypatch):
    from Models import BasicModel
    from test_operators_final_b import forced_prefix_certificate
    score = BasicModel._sentence_path_cost
    charges = []
    def observed(model, *args, **kwargs):
        result = score(model, *args, **kwargs)
        registry = model._sentence_cost_registry
        charges.append(registry._value(registry._terms['reconstruction.forward_discard']).detach())
        return result
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', observed)
    report = forced_prefix_certificate()
    assert report['reads'][1]['meaning_norm'] > 0
    assert all(read['multisets'] == 4 and read['R'] == [0.]*4 for read in report['reads'])
    assert charges and all(not value.any() for value in charges)
