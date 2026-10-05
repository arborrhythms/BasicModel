"""Forced identity mechanisms; no claim about learned anaphora accuracy."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from reading_fixtures import finish_reading
from Layers import MeaningExpectation
from ReferenceContext import SituationFrame
from reading_fixtures import resolve_reading_references as resolve_occurrences
from Understanding import AnswerProgram
from test_clause_storage import clause_store, idea_clause
from test_sentence_expectation import layer, observe


def language_and_program(mode='particular'):
    rule = SimpleNamespace(method_name='sum', lhs='S',
        reference_orders=(('I1', 1),), reference_kinds=(('I1', mode),))
    language = SimpleNamespace(_compose_binary_rules=(rule,), _compose_unary_rules=(),
        forward_binary_step=lambda a, b, *_: (a + b) * .5)
    leaves = torch.eye(4)[[0, 2]]
    program = AnswerProgram(rows=torch.tensor([0, 2]), word_rows=torch.tensor([0, 2]),
        activations=torch.ones(2), leaves=leaves,
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1]]),
        targets=torch.tensor([0, 1, 1]),
        end_state=torch.stack((leaves.mean(0), torch.zeros(4), torch.zeros(4))),
        concept_ids=torch.tensor([1, 3]), reference_orders=torch.tensor([1, -1]))
    return language, program


def frame(cid=5, point=None):
    point = torch.tensor([.2, .3, .4, .5]) if point is None else point
    return SituationFrame(1, torch.stack((point, point * 0, point * 0)),
                          torch.tensor([True, False, False]), cid, point)


def prediction(point):
    return MeaningExpectation(torch.stack((point, point * 0, point * 0)), torch.ones(3))


def test_same_individual_uses_prior_sentence_point_and_keeps_own_words():
    language, original = language_and_program()
    prior = frame()
    resolved = resolve_occurrences(language, original, frames=(prior,),
        prediction=prediction(prior.point), forced={0: prior.row_id})
    clause = finish_reading(language, resolved)
    assert clause.refs[0] == prior.row_id
    torch.testing.assert_close(clause.meaning.roles[0], prior.point)
    torch.testing.assert_close(clause.point, (prior.point + original.leaves[1]) * .5)
    torch.testing.assert_close(resolved.leaves, original.leaves, rtol=0, atol=0)
    torch.testing.assert_close(resolved.word_rows, original.word_rows)


def test_pronoun_requires_a_particular_and_never_uses_its_type():
    language, program = language_and_program('pronoun')
    with pytest.raises(ValueError, match='pronoun has no particular'):
        resolve_occurrences(language, program)
    prior = frame()
    kind = replace(prior, order=2)
    with pytest.raises(ValueError, match='pronoun has no particular'):
        resolve_occurrences(language, program, frames=(kind,), prediction=prediction(kind.point))
    got = resolve_occurrences(language, program, frames=(prior,), prediction=prediction(prior.point))
    assert got.reference_ids[0] == prior.row_id


def test_pronoun_can_bind_an_earlier_live_noun_without_a_reference_operator():
    language, program = language_and_program('pronoun')
    rule = language._compose_binary_rules[0]
    rule.reference_orders, rule.reference_kinds = (('I2', 1),), (('I2', 'pronoun'),)
    program = replace(program, reference_orders=torch.tensor([1, 1]))
    got = resolve_occurrences(language, program, forced={1: 1})
    assert got.reference_ids.tolist() == [1, 1]
    torch.testing.assert_close(got.reference_values[1], program.leaves[0])


def test_two_individuals_and_cold_situation_keep_distinct_choices():
    language, program = language_and_program()
    first, second = frame(), frame(6, torch.eye(4)[1])
    for cid in (1, 5, 6):
        got = resolve_occurrences(language, program, frames=(first, second),
            prediction=prediction(second.point), forced={0: cid})
        assert got.reference_ids[0] == cid
    assert resolve_occurrences(language, program).reference_ids[0] == 1
    with pytest.raises(ValueError, match='outside the bounded'):
        resolve_occurrences(language, program, forced={0: 12345})


def test_predictor_context_carries_owned_reference_without_a_memory_read():
    store, _ = clause_store()
    row = store.write_clause(idea_clause())
    discourse = layer()
    discourse._ltm_store = store
    observe(discourse, store.meaning_of(row).roles)
    discourse.bind_observation_occurrence(0, store.occurrence_of(row))
    discourse.detach_prediction_context()
    discourse._ltm_store = None
    anchors = discourse.situation_references(0)
    assert len(anchors) == 1 and anchors[0].row_id == int(store.row_ids[row])
    torch.testing.assert_close(anchors[0].point, store.slots[row, 0])
    discourse.begin_document(0, 'new')
    assert discourse.situation_references(0) == ()


def test_hard_identity_choice_keeps_predictor_gradient_and_pure_predicate():
    language, program = language_and_program()
    prior = frame()
    query = torch.nn.Parameter(prior.point.clone())
    got = resolve_occurrences(language, program, frames=(prior,),
        prediction=prediction(query), forced={0: prior.row_id})
    got.reference_values[0].sum().backward()
    assert query.grad is not None and bool(query.grad.any())
    torch.testing.assert_close(got.reference_values[1], program.leaves[1])


def test_reference_request_follows_the_head_and_outer_particular_scope():
    from ReferenceContext import reference_requests
    generic = SimpleNamespace(reference_orders=(('I1', 2),),
        reference_kinds=(('I1', 'generic'),), head_role=1)
    lower = SimpleNamespace(reference_orders=(('I2', 1),),
        reference_kinds=(('I2', 'particular'),), head_role=2)
    language = SimpleNamespace(_compose_binary_rules=(lower,), _compose_unary_rules=(generic,))
    actions = torch.tensor([[0, -1, 0], [0, -1, 1], [2, 0, -1], [1, 0, -1]])
    assert reference_requests(language, actions) == {1: (1, 'particular')}


def test_hard_reference_value_is_bit_exact_at_every_softmax_score(monkeypatch):
    import ReferenceContext
    language, program = language_and_program()
    prior = frame()
    for logit in torch.linspace(-2., 2., 37):
        def scores(values, *_args, **_kwargs):
            result = values.new_zeros(values.shape[:-1])
            result[..., 1] = logit
            return result
        monkeypatch.setattr(ReferenceContext.F, 'cosine_similarity', scores)
        got = resolve_occurrences(language, program,
            frames=(prior,), prediction=prediction(prior.point), forced={0: prior.row_id})
        torch.testing.assert_close(got.reference_values[0], prior.point, rtol=0, atol=0)
