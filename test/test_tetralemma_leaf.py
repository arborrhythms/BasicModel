"""Plan §42: two positive symbols, one identity, four independent corners."""
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

CORNERS = ((1., 0.), (0., 1.), (1., 1.), (0., 0.))


@pytest.mark.parametrize('corner', CORNERS + ((.35, .8),))
def test_walk_to_leaf_reference_and_stored_row(corner, monkeypatch):
    from Models import BasicModel
    from Interpret import InterpretLayer
    from ModelAttention import reference_evidence
    from Language import SymbolSpace
    from reading_fixtures import finish_reading
    from test_clause_acceptance import SentenceFixture

    # Non-unit presence proves that the conceptual pair is not encoded in it.
    presence = torch.tensor([[.4]])
    pair = torch.tensor([[corner]])
    atoms = torch.tensor([[.6, .8, 1., .5, .25, 0., 0., 0.]])
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(
        mereology=SimpleNamespace(percept_event_width=2)))
    owner.interpret = InterpretLayer(conceptualSpace=owner)
    model = SimpleNamespace(_attention_words=SimpleNamespace(
        accepted=torch.ones(1, 1, dtype=torch.bool)), _attention_poles=pair,
        _concept_owner=lambda: owner, _word_symbol_rows=lambda: torch.tensor([[0]]),
        inputSpace=SimpleNamespace(_ar_word_concept_atoms=atoms[:, None]))
    event = torch.zeros(1, 1, 8)
    payload = (event, torch.zeros(1, 1), torch.tensor([0]), presence,
        torch.tensor([0]), torch.tensor([0]), atoms,
        torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool),
        None, None, torch.zeros(1, 2))
    handed = BasicModel._attention_sentence_payload(model, payload, torch.tensor(0))
    leaf = owner.interpret(event[:, 0], object_atoms=atoms,
        presence=handed[3].reshape(1), evidence=handed[-1])
    expected = torch.cat((atoms[:, :2] * .4,
        atoms[:, 2:5] * corner[0], atoms[:, 2:5] * corner[1]), -1)
    torch.testing.assert_close(leaf, expected, rtol=0, atol=0)
    pushed = BasicModel._pushed_word_slab(model, 1, 1, 8, event, presence[:, None])
    torch.testing.assert_close(pushed[:, 0], expected, rtol=0, atol=0)
    symbol = SimpleNamespace()
    SymbolSpace.commit_word_reference_slab(symbol, torch.tensor([[0]]), presence[:, None],
        torch.ones(1, 1, dtype=torch.bool), evidence=reference_evidence(model,
            presence[:, None], 'commit_word_reference_slab:whole_slab'))
    torch.testing.assert_close(symbol._word_reference_evidence, pair, rtol=0, atol=0)
    fixture = SentenceFixture(monkeypatch)
    program = replace(fixture.program('A'), leaves=leaf,
        leaf_evidence=symbol._word_reference_evidence[0])
    clause = finish_reading(fixture.language, program, registry=fixture.registry)
    row = fixture.store.write_clause(clause)
    assert fixture.store.row(row)['evidence'] == pytest.approx(corner)
    assert clause.meaning.polarity  # no Boolean is inferred from the pair


@pytest.mark.parametrize('left_pair,right_pair', tuple(product(CORNERS, repeat=2)))
@pytest.mark.parametrize('name', ['conjunction', 'disjunction', 'sum'])
def test_corner_composition_lane_rules_and_form_unchanged(name, left_pair, right_pair):
    from Interpret import activate_code
    from Language import GRAMMAR_LAYER_CLASSES
    code = torch.tensor([[1., .5, .25]])
    a = torch.cat((torch.tensor([[.8, .6]]), code, code * 0), -1)
    b = torch.cat((torch.tensor([[.3, .9]]), code, code * 0), -1)
    left = activate_code(a, torch.ones(1), 2, evidence=torch.tensor([left_pair]))
    right = activate_code(b, torch.ones(1), 2, evidence=torch.tensor([right_pair]))
    op = GRAMMAR_LAYER_CLASSES[name](); op.meaning_layout = (2, 8)
    result = op.compose(left, right)
    pair = ((min(left_pair[0], right_pair[0]), max(left_pair[1], right_pair[1]))
            if name == 'conjunction' else
            (max(left_pair[0], right_pair[0]), min(left_pair[1], right_pair[1]))
            if name == 'disjunction' else
            ((left_pair[0] + right_pair[0]) / 2, (left_pair[1] + right_pair[1]) / 2))
    torch.testing.assert_close(result[:, :2], op.compose(a[:, :2], b[:, :2]), rtol=0, atol=0)
    torch.testing.assert_close(result[:, 2:], torch.cat((code * pair[0], code * pair[1]), -1), rtol=0, atol=0)


@pytest.mark.parametrize('pair', CORNERS)
def test_not_exchange_and_non_withdrawal(pair):
    from Language import NotLayer, NonLayer
    value = torch.tensor([[.8, .6, pair[0], pair[0], pair[1], pair[1]]])
    negation, withdrawal = NotLayer(), NonLayer()
    assert 'meaning' in withdrawal.footprint_reads and 'meaning' in withdrawal.footprint_writes
    negation.meaning_layout = withdrawal.meaning_layout = (2, 6)
    torch.testing.assert_close(negation(value), value[:, [0, 1, 4, 5, 2, 3]], rtol=0, atol=0)
    expected = value.clone(); expected[:, 2:4] = 0
    torch.testing.assert_close(withdrawal(value), expected, rtol=0, atol=0)


def test_required_evidence_read_of_true_and_false_is_both(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import finish_reading
    fixture = SentenceFixture(monkeypatch)
    program = replace(fixture.program(('sum', 'A', 'B')),
        leaf_evidence=torch.tensor([[1., 0.], [0., 1.]]))
    clause = finish_reading(fixture.language, program, registry=fixture.registry)
    row = fixture.store.write_clause(clause)
    assert clause.evidence == fixture.store.row(row)['evidence'] == (1., 1.)


@pytest.mark.parametrize('left_pair,right_pair', tuple(product(CORNERS + ((.2, .7),), repeat=2)))
@pytest.mark.parametrize('name', ['conjunction', 'disjunction', 'sum'])
def test_inverse_recovers_each_lane_after_signless_identity(name, left_pair, right_pair):
    from MeaningCodes import compose, recover_lanes
    from Interpret import activate_code
    # Overlapping codes and a shared coordinate exercise min/max inverse
    # ambiguities and the two-column mean solve, without an input trace.
    a = torch.tensor([[.8, .6, 1., .5, 0., 0., 0., 0.]])
    b = torch.tensor([[.3, .9, 1., 0., .25, 0., 0., 0.]])
    left = activate_code(a, torch.ones(1), 2, evidence=torch.tensor([left_pair]))
    right = activate_code(b, torch.ones(1), 2, evidence=torch.tensor([right_pair]))
    parent = torch.cat((a[:, :2], compose(name, left[:, 2:], right[:, 2:])), -1)
    recovered = recover_lanes(name, parent, a, b, 2)
    torch.testing.assert_close(compose(name, recovered[0][:, 2:], recovered[1][:, 2:]), parent[:, 2:])
    assert all((v[:, 2:] >= 0).all() for v in recovered)


def test_path_ast_has_no_pole_collapse():
    from tetralemma_audit import audit
    report = audit()
    assert report['sites'] and not report['violations']


@pytest.mark.parametrize('expression', [
    'pair[..., 0] - pair[..., 1]', 'pair[:, 0].sign()',
    'pair.sum(-1)', 'pair.mean(dim=-1)', 'pair.argmax(-1)',
    'pair.sum(1)', 'pair.sum(dim=2)',
    'pair[0] >= pair[1]', 'torch.linalg.vector_norm(pair, dim=-1)',
])
def test_path_ast_rejects_injected_collapses(expression):
    import ast
    from tetralemma_audit import inspect_site
    node = ast.parse('def bad(pair):\n    result = ' + expression).body[0]
    assert inspect_site(node, {'pair'})
