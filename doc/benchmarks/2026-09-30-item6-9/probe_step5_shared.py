"""Locate the shared autograd owner in two pre-update sentence trials."""
import torch
from test_sentence_compose import test_real_packed_ends_train_before_the_next_sentence as _packed


def test_packed_trials_keep_shared_graph_until_both_backwards(tmp_path, monkeypatch):
    with torch.autograd.detect_anomaly():
        _packed(tmp_path, monkeypatch, False)
