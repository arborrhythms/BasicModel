def test_prior_thought_reserves_the_input_allowance():
 from test_negative_expectation import _anticipating_model
 model,owner,meaning=_anticipating_model()
 model.train();model.attention_budget=4
 model._stage_expectation_queries(training=True)
 pending=owner._inter_last_meaning[0]
 assert pending.work<=model.attention_budget, (pending.work,model.attention_budget)
 assert pending.walk.other.work<=model.attention_budget
 assert model._pending_attention_meters[0].spent==max(pending.work,pending.walk.other.work)
