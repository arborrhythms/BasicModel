from test_negative_expectation import _anticipating_model, observe

def test_anticipation_holds_both_walks_until_future_return():
    model, owner, meaning = _anticipating_model()
    model.train()
    before = tuple(model._what_memory().thought_history())
    model._stage_expectation_queries(training=True)
    pending = owner._inter_last_meaning[0]
    assert pending.walk is not None, 'prior-only thought has no paired future-return comparison'
    assert tuple(model._what_memory().thought_history()) == before
    assert pending.walk.other.versions == pending.versions
    observe(owner, meaning.roles)
    model._expectation_policy_loss()
    assert model._walk_audit['think.anticipation']['walks'] == 1
    assert model._walk_audit['think.anticipation']['strict_violations'] == 0
