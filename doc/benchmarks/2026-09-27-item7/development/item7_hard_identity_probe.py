import pytest
import torch
import ReferenceContext
from test_item7_references import language_and_program, frame, prediction


def test_hard_reference_value_is_bit_exact_at_every_softmax_score(monkeypatch):
    language, program = language_and_program()
    prior = frame()
    for logit in torch.linspace(-2., 2., 37):
        monkeypatch.setattr(ReferenceContext.F, 'cosine_similarity',
            lambda *a, **k: torch.stack((logit * 0, logit)))
        got = ReferenceContext.resolve_occurrences(language, program,
            frames=(prior,), prediction=prediction(prior.point), forced={0: prior.row_id})
        torch.testing.assert_close(got.reference_values[0], prior.point, rtol=0, atol=0)
