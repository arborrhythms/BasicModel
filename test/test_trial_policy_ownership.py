"""Answer generation cannot write the reconstruction-owned decoder."""
import pytest
import torch


@pytest.mark.parametrize('trial', ['exploit', 'explore'])
@pytest.mark.parametrize('compiled', [False, True])
def test_trial_and_batch_answer_leave_the_decoder_to_reconstruction(monkeypatch, trial, compiled):
    import util
    from Meaning import ConceptualMeaning
    from Output import AnswerDerivation
    from Understanding import Understanding, SentenceEndState
    from What import What
    from test_output_walk import _model
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager' if compiled else 'none')
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    if not compiled:
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
        result = model.reverseOutput(understanding, held)
        assert model._output_policy_cost is None
        params = tuple(model.languageSpace.generate_policy.parameters())
        gradients = torch.autograd.grad(result.actual.square().mean(), (point,),
                                        allow_unused=True, retain_graph=True)
        assert gradients[0] is None
        from ObjectiveOwnership import backward_owned
        optimizer = model.getOptimizer(lr=.01)
        owners = model.objective_parameter_groups(optimizer)
        assert {id(p) for p in params} <= {id(p) for p in owners['reconstruction']}
        backward_owned({'output': result.actual.square().mean()}, owners)
        assert all(p.grad is None for p in params)
        assert any(p.grad is not None and p.grad.norm() > 0 for p in owners['output'])
        model._sentence_trial = None
        optimizer.zero_grad(set_to_none=True)
        result = model.reverseOutput(understanding, held)
        assert model._output_policy_cost is None
        backward_owned({'output': result.actual.square().mean()}, owners)
        assert all(p.grad is None for p in params)
        assert any(p.grad is not None and p.grad.norm() > 0 for p in owners['output'])
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
