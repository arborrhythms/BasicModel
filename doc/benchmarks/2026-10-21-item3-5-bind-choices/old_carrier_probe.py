"""Diagnostic only: isolate the pre-67c5b203f captured identity lane."""
import pytest
import torch


@pytest.fixture(autouse=True)
def old_carrier(request, monkeypatch):
    if request.node.name != 'test_both_training_trials_share_bootstrap_snapshot':
        return
    from Models import BasicModel
    original = BasicModel._program_entries

    def entries(model, *args, **kwargs):
        kwargs['concept_ids'] = torch.full_like(kwargs['concept_ids'], -1)
        return original(model, *args, **kwargs)
    monkeypatch.setattr(BasicModel, '_program_entries', entries)
