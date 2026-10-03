"""Trial answer generation leaves policy credit at the batch end."""
import pytest
import torch


@pytest.mark.parametrize('trial', ['exploit', 'explore'])
def test_trial_generation_leaves_policy_credit_at_batch_end(monkeypatch, trial):
    import util
    from Meaning import ConceptualMeaning
    from Output import AnswerDerivation
    from Understanding import Understanding, SentenceEndState
    from What import What
    from test_output_walk import _model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    model = _model()
    model.train()
    width = model.conceptualSpace.stm.concept_dim
    point = torch.randn(1, width, requires_grad=True)
    meanings = tuple(ConceptualMeaning.from_description(row) for row in point)
    fields = tuple(SentenceEndState(meaning) for meaning in meanings)
    idea = torch.stack([meaning.roles for meaning in meanings])
    questions = (What.supervised(0),)
    held = AnswerDerivation(answer_symbol=None, sentence_states=fields,
        conceptual_answer=idea, questions=questions,
        answer_meanings=tuple((meaning,) for meaning in meanings))
    understanding = Understanding(conceptual_state=idea, sentence_states=fields)
    model._sentence_training = True
    model._sentence_trial = trial
    model._output_policy_cost = None
    try:
        result = model.reverseOutput(understanding, held, detach_understanding=False)
        assert model._output_policy_cost is None
        params = tuple(model.languageSpace.generate_policy.parameters())
        gradients = torch.autograd.grad(result.actual.square().mean(), (point, *params), allow_unused=True)
        assert gradients[0] is not None and bool(gradients[0].abs().sum() > 0)
        assert all(g is None or not bool(g.abs().any()) for g in gradients[1:])
        model._sentence_trial = None
        model.reverseOutput(understanding, held)
        assert torch.is_tensor(model._output_policy_cost)
        assert model._output_policy_cost.requires_grad
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
