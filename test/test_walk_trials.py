"""The owner costs complete alternative walks before either can train."""
import torch
import pytest














def test_walk_audit_records_strict_wins_and_sentence_stability():
    from types import SimpleNamespace
    from WalkTrials import observe_comparison
    model=SimpleNamespace(_attention_forms=([['a','b']],None,None,None))
    audit=dict(costs=torch.tensor([[2.,1.]]),wins=torch.tensor([True]),
        departure=torch.tensor([0]),greedy=torch.tensor([[0,1]]),explore=torch.tensor([[2,1]]))
    for kind in ('compose','generate.decoder','generate.output','think'):
        observe_comparison(model,kind,audit,sentence=0)
        observe_comparison(model,kind,audit,sentence=0)
        row=model._walk_audit[kind]
        assert row['walks']==2 and row['explore_wins']==2 and row['strict_violations']==0
        assert row['stability_pairs']==1 and row['stable_pairs']==1
        assert row['explore_fraction']==1.
