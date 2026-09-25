"""The subsymbolic selector retargets attention within the processing bound."""
from pathlib import Path

import pytest

from Models import BasicModel
from util import XMLConfig


@pytest.mark.parametrize('value, bound, expected', [
    ('all', 1, set()),
    ('all', 4, {1, 2, 3}),
    ('off', 4, set()),
    ('1, 3', 4, {1, 3}),
])
def test_attention_pass_selection(value, bound, expected):
    assert BasicModel._subsymbolic_loop_passes(value, bound) == expected


@pytest.mark.parametrize('value', ['0', '4', '-1', '1,no'])
def test_selector_rejects_initial_or_unavailable_passes(value):
    with pytest.raises(ValueError, match='subsymbolicLoop'):
        BasicModel._subsymbolic_loop_passes(value, 4)


def test_default_and_overlay_use_subsymbolic_loop(tmp_path):
    config = XMLConfig()
    config.load(str(Path(__file__).resolve().parents[1] / 'data/model.xml'))
    assert config.get('architecture.subsymbolicLoop') == 'all'
    path = tmp_path / 'attention.xml'
    path.write_text('<model><architecture><subsymbolicLoop>off</subsymbolicLoop>'
                    '</architecture></model>')
    config.overlay(str(path))
    assert config.get('architecture.subsymbolicLoop') == 'off'


def test_overlay_rejects_retired_spelling(tmp_path):
    config = XMLConfig()
    path = tmp_path / 'retired.xml'
    path.write_text('<model><architecture><subsymbolicNoop>1</subsymbolicNoop>'
                    '</architecture></model>')
    with pytest.raises(ValueError, match='subsymbolicNoop'):
        config.overlay(str(path))


def test_selected_pass_changes_the_attentive_extent(tmp_path, monkeypatch):
    import torch
    from test_grounded_xor import grounded_model

    model, inputs = grounded_model(tmp_path)
    cs, ws = model.conceptualSpaces[0], model.wholeSpaces[0]
    prior = ws.subspace.what.primitive_properties
    prior.teach(8, [48, 49], [0., 1.])
    prior.teach(9, [48, 49], [1., 0.])
    cs._csw_concept_row(0, 10001)
    cs.add_concept_feature(0, 'ws', 8, 1.)
    cs.add_concept_feature(0, 'ws', 9, -1.)
    calls = []

    def request_right_position():
        calls.append(True)
        object.__setattr__(model.conceptualSpace, '_passback_scope_where',
                           torch.tensor([1., 2.]) / inputs.shape[-1])

    monkeypatch.setattr(model, '_primed_reading_step', request_right_position)
    model.relevance_on = True
    model.reading_attention = None
    model.subsymbolic_loop = frozenset()
    model.forward(inputs[2:3])  # 10: both over the initial two-position extent.
    torch.testing.assert_close(cs._cs_last_a0[0, 0, 0], torch.tensor([1., 1.]))
    assert not calls
    model.End()
    model.subsymbolic_loop = frozenset({1})
    model.forward(inputs[2:3])
    torch.testing.assert_close(cs._cs_last_a0[0, 0, 0], torch.tensor([0., 1.]))
    assert len(calls) == 1
