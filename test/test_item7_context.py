"""Situation rotation is bounded and reads only the pre-sentence snapshot."""
from types import SimpleNamespace

import torch

from Models import BaseModel
from test_contextual_concept_codebook import _codebook, _context_model


def model_and_book():
    book = _codebook('cpu', rows=8, dim=8)
    book.W.copy_(torch.eye(8))
    model = _context_model(book, 'cpu')
    model.contextual_concept_negatives = 0
    model.inputSpace._ar_word_concept_rows = torch.tensor([[1, 2]])
    model.inputSpace._word_active_mask = torch.tensor([[True, True]])
    return model, book


def test_situation_weight_adds_only_bounded_prior_anchors():
    model, book = model_and_book()
    model.contextual_situation_weight = 1.
    model.contextual_situation_anchors = 1
    roles = torch.stack((book.W[3], book.W[4], book.W[5]))
    frames = [(3, roles, torch.ones(3, dtype=torch.bool)),
              (1, torch.stack((book.W[6], book.W[0] * 0, book.W[0] * 0)),
               torch.tensor([True, False, False]))]
    disc = SimpleNamespace(_inter_context=[frames], _inter_last_meaning=[None])
    model.symbolSpace = SimpleNamespace(discourse=disc)
    BaseModel._capture_contextual_situation(model, 0, torch.tensor([True]))
    frames.clear()  # the ended update must keep its prior, not read future state
    model._update_contextual_concept_codebooks()
    assert book.W[1, 6] > 0
    assert book.W[1, 3] == 0


def test_expectation_weight_one_removes_arriving_context_from_rotation():
    first, a = model_and_book()
    second, b = model_and_book()
    for model in (first, second):
        model.contextual_expectation_weight = 1.
        model._contextual_sentence_priors = {0: ((torch.zeros(8), a.W[7].clone()),)}
    second.inputSpace._ar_word_concept_rows[0, 1] = 3
    first._update_contextual_concept_codebooks()
    second._update_contextual_concept_codebooks()
    torch.testing.assert_close(a.W[1], b.W[1], rtol=0, atol=0)
    assert a.W[1, 7] > 0 and a.W[1, 2] == 0


def test_cold_expectation_at_weight_one_does_not_learn_from_arriving_words():
    model, book = model_and_book()
    model.contextual_expectation_weight = 1.
    before = book.W.clone()
    model._update_contextual_concept_codebooks()
    torch.testing.assert_close(book.W, before, rtol=0, atol=0)
