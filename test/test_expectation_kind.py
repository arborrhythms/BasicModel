"""Item 4 retires the slot/kind head; grammatical kind remains observed metadata."""
import copy
import torch
from Layers import SymbolExpectation


def test_grammatical_kind_does_not_train_a_second_expectation_head():
    idea = SymbolExpectation(4, 8, 4, concept_dim=4)
    idea.set_symbol_bank(torch.eye(4))
    relation = copy.deepcopy(idea)
    for model in (idea, relation):
        model.predict_and_observe_stm_end_state([1], [torch.eye(4)[:1]], layout='infix')
        assert not hasattr(model.expect_next_meaning(), 'kind_logit')
    for model, kind in ((idea, 'idea'), (relation, 'relation')):
        model.predict_and_observe_stm_end_state([1], [torch.eye(4)[1:2]], layout='infix', sentence_kinds=[kind])
    torch.testing.assert_close(idea.consume_inter_loss(), relation.consume_inter_loss())
    assert relation.last_expectation_comparison().sentence_kind == 'relation'
    assert not any('kind' in name for name, _ in idea.named_parameters())


def test_activation_preview_and_committed_comparison_agree():
    model = SymbolExpectation(4, 8, 4, concept_dim=4)
    model.set_symbol_bank(torch.eye(4))
    source = torch.eye(4)[:3]
    model.predict_and_observe_stm_end_state([3], [source], layout='infix')
    cost, pending = model.sentence_prediction_cost([3], [source], [True], layout='infix')
    model._inter_last_meaning = pending
    model.observe_stm_end_state([3], [source], layout='infix', sentence_kinds=['relation'])
    torch.testing.assert_close(model.consume_inter_loss(), cost[0])
    assert not hasattr(model.last_expectation_comparison(), 'kind_residual')
