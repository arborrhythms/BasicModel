"""Two-truths §7.15: grammatical kind is a supervised prediction target."""
import copy

import torch

from Layers import InterSentenceLayer


def layer():
    return InterSentenceLayer(n_symbols=4, max_depth=8, n_dim=4,
        concept_dim=4, expectation_scope='structured')


def observe(model, roles, kind):
    model.predict_and_observe_stm_end_state([3], [roles], layout='infix',
        documents=['document'], sentence_kinds=[kind])


def test_kind_is_supervised_independently_of_role_vectors():
    idea = layer()
    relation = copy.deepcopy(idea)
    source = torch.arange(12.).reshape(3, 4) / 12
    target = torch.nn.Parameter(source + .2)
    for model in (idea, relation):
        observe(model, source, 'idea')
        assert model.expect_next_meaning().kind_logit.shape == ()
    observe(idea, target, 'idea')
    observe(relation, target, 'relation')
    idea.consume_inter_loss().backward()
    relation.consume_inter_loss().backward()
    assert target.grad is None
    assert idea._inter_predictor.kind_bias.grad > 0
    assert relation._inter_predictor.kind_bias.grad < 0
    assert idea.expectation_metrics()['kind_targets'] == 1
    assert relation.expectation_metrics()['kind_targets'] == 1


def test_kind_preview_and_committed_comparison_agree():
    model = layer()
    source = torch.eye(4)[:3]
    observe(model, source, 'idea')
    cost, _, pending = model.sentence_prediction_cost([3], [source], [True],
        documents=['document'], layout='infix', sentence_kinds=['relation'])
    model._inter_last_meaning, model._inter_last_pred_root = pending
    model.observe_stm_end_state([3], [source], layout='infix',
        documents=['document'], sentence_kinds=['relation'])
    torch.testing.assert_close(model.consume_inter_loss(), cost[0])
    comparison = model.last_expectation_comparison()
    assert comparison.sentence_kind == 'relation'
    assert comparison.kind_residual > 0


def test_pre_kind_checkpoint_keeps_role_head_and_initializes_only_kind():
    model = layer()
    before = copy.deepcopy(model.state_dict())
    old = {k: v for k, v in before.items() if 'kind_weight' not in k and 'kind_bias' not in k}
    model.load_state_dict(old, strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
