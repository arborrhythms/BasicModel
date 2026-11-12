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


def test_expectation_is_global_and_does_not_open_an_anticipatory_episode():
    from test_negative_expectation import _anticipating_model
    model,owner,meaning=_anticipating_model()
    model.train()
    before=tuple(model._what_memory().thought_history())
    model._stage_attention_meters(training=True)
    assert tuple(model._what_memory().thought_history())==before
    assert not hasattr(model, '_expectation_policy_loss')
    assert owner.expect_next_meaning(0,record=False) is not None




def test_forecast_previews_publish_only_the_committed_observation():
    from test_negative_expectation import _anticipating_model
    model, owner, meaning = _anticipating_model()
    model.train()
    model._stage_attention_meters(training=True)
    pending = owner._inter_last_meaning[0]
    mask = torch.ones(1, dtype=torch.bool)
    for _ in range(2):
        _, held = owner.sentence_prediction_cost([3], [meaning.roles], mask,
            layout='infix', role_masks=[meaning.role_mask])
        assert held[0].prediction is not None
        assert owner._inter_last_meaning[0] is pending
        assert not hasattr(owner, 'consume_expectation_walk_outcomes')
    owner._inter_last_meaning = held
    owner.observe_stm_end_state([3], [meaning.roles], mask=mask, layout='infix',
        role_masks=[meaning.role_mask], train_prediction=False)
    assert owner.last_expectation_comparison(0) is not None
    assert not hasattr(owner, 'consume_expectation_walk_outcomes')
