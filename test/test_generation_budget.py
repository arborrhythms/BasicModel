"""A construction budget limits free generation, never the recorded input."""
from types import SimpleNamespace

import pytest
import torch

from Models import BasicModel


@pytest.mark.parametrize('text', ['0', '-1', '1.5', 'true', 'nan', 'inf', ''])
def test_generate_budget_rejects_invalid_xml(tmp_path, text):
    from util import XMLConfig
    source = tmp_path / 'budget.xml'
    source.write_text('<model><architecture><generateBudget>' + text
                      + '</generateBudget></architecture></model>')
    with pytest.raises(ValueError, match='generateBudget'):
        XMLConfig._parse_xml(str(source))


@pytest.mark.parametrize('value', [0, -1, 1.5, True, float('nan'), '32'])
def test_generate_budget_rejects_invalid_construction_values(value):
    with pytest.raises(ValueError, match='generateBudget.*positive integer'):
        BasicModel()._configure_generate_budget(value)


def test_generate_budget_defaults_and_construction_overlay(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    from util import XMLConfig
    import util
    defaults = XMLConfig(defaults_path='data/model.xml')
    assert defaults.get('architecture.generateBudget') == 32
    assert BasicModel()._walk_budget() == 32
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'generation_budget', [
        ('<architecture>', '<architecture><generateBudget>5</generateBudget>')])
    try:
        assert model.generate_budget == model._walk_budget() == 5
        assert model._derivation_trace_capacity() > 5
        with pytest.raises(AttributeError):
            model.generate_budget = 9
        with pytest.raises(ValueError, match='fixed at model construction'):
            model._configure_generate_budget(9)
        assert model._walk_budget() == 5
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def _chain(monkeypatch, budget):
    """An explicit menu oracle isolates the shared native walk's trip limit."""
    import Generative
    import util
    from generation_fixtures import exact_menu
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    language = SimpleNamespace(_generate_binary_ops=(),
        _generate_unary_ops=(SimpleNamespace(meaning_layout=(1, 1, 1)),) * 3,
        _generate_policy_width=3)
    language.generate_policy_logits = lambda top, features=None: top.new_zeros(len(top), 4)
    language.choose_generate = lambda logits: logits.argmax(-1)
    calls = []

    def menu(_language, parent, live, **options):
        calls.append(parent.detach().clone())
        left = parent[:, None].expand(-1, 4, -1).clone()
        left[:, :3] += 1
        legal = torch.stack(tuple(live & (parent[:, 0] == step + 1)
                                  for step in range(4)), -1)
        result = exact_menu(left, left.clone(), legal)
        form = left.new_full(legal.shape, .1)
        meaning = left.new_tensor([.2, .3]).expand(*legal.shape, 2).clone()
        form[:, -1] = 0
        meaning[:, -1] = 0
        return result._replace(form_residual=form, meaning_residual=meaning)

    monkeypatch.setattr(Generative, 'inverse_menu', menu)
    model = BasicModel()
    model._configure_generate_budget(budget)
    model.languageSpace = language
    model.conceptualSpace = SimpleNamespace(stm=SimpleNamespace(capacity=3))
    return model, calls


@pytest.mark.parametrize('numeric', [False, True])
def test_free_walk_cap_retains_each_lane_and_fixed_arrays(monkeypatch, numeric):
    model, calls = _chain(monkeypatch, 2)
    event = torch.tensor([[[1., 2., 3.], [0., 0., 0.], [0., 0., 0.]]])
    with torch.no_grad():
        result = model._output_generate_walk(event, model._walk_budget(),
            require_symbols=not numeric, return_candidates=True)
    assert len(calls) == 2
    assert result[1].tolist() == [0] and result[2].tolist() == [True]
    assert result[4].tolist() == [[0, 1]]
    assert result[0].shape == (1, 2, 3)
    assert result[-4].shape == (1, 2, 3)  # chosen residuals
    assert result[-2].shape == (1, 2, 4, 9)  # detached menu features
    assert result[-1].shape == (1, 2, 3)  # detached parent tops
    frontier = torch.tensor([[3., 4., 5.]])
    if numeric:
        frontier = frontier.asinh()
    torch.testing.assert_close(result[3], frontier.square() + torch.tensor([[.2, .4, .6]]))
    assert model._walk_budget() == 2


def test_explicit_walk_budget_and_sentence_decoder_share_the_cap(monkeypatch):
    model, calls = _chain(monkeypatch, 2)
    event = torch.tensor([[[1., 2., 3.], [0., 0., 0.], [0., 0., 0.]]])
    with torch.no_grad():
        completed = model._output_generate_walk(event, 4, return_candidates=True)
        assert len(calls) == 4 and completed[4].tolist() == [[0, 1, 2, 3]]
        assert completed[1].tolist() == [1] and completed[2].tolist() == [False]
        torch.testing.assert_close(completed[3], torch.tensor([[.3, .6, .9]]))
        calls.clear()
        decoded = model._decode_conceptual_sentence(event[:, 0], event,
            torch.tensor([1]), event[:, :1], torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1), return_candidates=True)
    assert len(calls) == 2 and decoded[4].tolist() == [[0, 1]]
    assert decoded[2].tolist() == [True]
    assert decoded[0].shape == (1, 2, 3)
    assert model._walk_budget() == 2  # explicit low-level calls cannot resize the model


def _recorded_model():
    """Forty recorded input pushes and unary folds exceed a free budget of two."""
    model = BasicModel()
    model._configure_generate_budget(2)
    model.serial_word_capacity = 40
    model.conceptualSpace = SimpleNamespace(stm=SimpleNamespace(capacity=8))
    model.inputSpace = SimpleNamespace(_word_active_mask=torch.ones(1, 40, dtype=torch.bool))
    ids = torch.full((1, 136), -1, dtype=torch.long)
    ids[:, :120:3] = 7
    trace = SimpleNamespace(choices=lambda: (ids, torch.ones_like(ids), ids >= 0))
    model._reconstruction_stack = lambda: trace

    def local(rule_ids, mapping):
        return torch.zeros_like(rule_ids), ((rule_ids == 7) if len(mapping)
                                            else torch.zeros_like(rule_ids, dtype=torch.bool))
    model.languageSpace = SimpleNamespace(_cs_binary_rule_ids=torch.empty(0, dtype=torch.long),
        _cs_unary_rule_ids=torch.tensor([7]), local_op_from_rule_ids=local,
        language_layer=SimpleNamespace(operation_layer=SimpleNamespace(pair_rewrites=())))
    return model, ids


def test_recorded_derivation_keeps_full_capacity_independent_of_free_budget():
    model, ids = _recorded_model()
    before = ids.clone()
    positions, actions, targets, columns = model._derivation_program()
    assert model._walk_budget() == 2
    assert targets.shape == (1, 176)
    assert positions.shape == (1, 40) and actions.shape == (1, 80, 3)
    assert targets[0, :80].tolist() == [0, 1] * 40
    assert (targets[0, 80:] == -1).all()
    torch.testing.assert_close(model._derivation_targets(), targets, rtol=0, atol=0)
    torch.testing.assert_close(ids, before, rtol=0, atol=0)
    # Existing explicit teacher-target prefixes remain available to low-level callers.
    assert model._derivation_targets(budget=2).tolist() == [[0, 1]]


def test_recorded_derivation_fails_if_the_declared_geometry_would_truncate():
    model, _ = _recorded_model()
    model.serial_word_capacity = 1
    with pytest.raises(ValueError, match='recorded derivation.*structural trace capacity'):
        model._derivation_program()
