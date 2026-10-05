"""Observe teacher coverage and hard picks without changing inference or RNG."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


@contextmanager
def observe_decomposition(result):
    from Models import BasicModel
    teacher = BasicModel._decomposition_teacher_loss
    observation = BasicModel._sentence_observation
    report = result.setdefault('decomposition_chooser', {})

    def snapshot(model):
        chooser = model.languageSpace.decomposition_chooser
        rows = list(model._last_decomposition_teacher)
        count = len(rows)
        present = sum(row['present'] for row in rows)
        correct = sum(row['picked_true'] for row in rows)
        return dict(feature_names=list(chooser.feature_names),
            weights=chooser.weight.detach().cpu().tolist(), undos=count,
            true_pair_in_shortlist=present, absent_targets=count-present,
            pick_equals_true=correct,
            true_pair_in_shortlist_rate=present/count if count else None,
            pick_equals_true_rate=correct/count if count else None,
            pick_equals_true_given_present=correct/present if present else None,
            rows=rows)

    def observed_teacher(model, entries, record):
        state = torch.random.get_rng_state()
        value = teacher(model, entries, record)
        assert torch.equal(state, torch.random.get_rng_state())
        report.setdefault('start', snapshot(model))
        report['training_calls'] = report.get('training_calls', 0) + 1
        return value

    def observed_observation(model, state, sid, active, **kwargs):
        value = observation(model, state, sid, active, **kwargs)
        if not getattr(model, '_sentence_training', False):
            rng = torch.random.get_rng_state()
            with torch.no_grad():
                teacher(model, value, model._trial_understanding(state, sid, active))
                report['end'] = snapshot(model)
            assert torch.equal(rng, torch.random.get_rng_state())
        return value

    with patch.object(BasicModel, '_decomposition_teacher_loss', observed_teacher), \
         patch.object(BasicModel, '_sentence_observation', observed_observation):
        yield
