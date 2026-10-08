"""The unused binary symbolize carrier is retired (catalogue §3.4).

The complete previous class/router/numerical tests are preserved in the
operators receipt. Mean composition belongs to SumLayer; concept admission
belongs to interpret and quantize remains a thought-only capability.
"""
import pytest
import torch


def test_binary_symbolize_has_no_class_export_or_dispatch_registration():
    import Language
    import Layers
    assert not hasattr(Language, 'SymbolizeLayer')
    assert not hasattr(Layers, 'SymbolizeLayer')
    assert 'symbolize' not in Language.GRAMMAR_LAYER_CLASSES


@pytest.mark.parametrize('face', ['compose', 'generate', 'thought'])
def test_binary_symbolize_is_rejected_in_every_face(face):
    from Language import Grammar
    direction = {'compose': 'forward', 'generate': 'reverse', 'thought': 'thought'}[face]
    with pytest.raises(ValueError, match='retired'):
        Grammar()._fill_rule_list([], {'rule':
            f'symbolize_O1 = symbolize.{direction}(symbolize_I1, symbolize_I2)'},
            face='thought' if face == 'thought' else None)


@pytest.mark.parametrize('name', ['symbolize', 'what', 'true', 'lookup'])
def test_legacy_syntax_does_not_restore_retired_operators(name):
    from Language import Grammar
    with pytest.raises(ValueError, match='retired'):
        Grammar()._fill_rule_list([], {'S': f'{name}(S)'})


@pytest.mark.parametrize('name', ['quantize', 'arma'])
def test_legacy_syntax_does_not_restore_thought_only_compose(name):
    from Language import Grammar
    with pytest.raises(ValueError, match='retired'):
        Grammar()._fill_rule_list([], {'S': f'{name}(S)'})


def test_implementation_alias_cannot_restore_retired_carrier():
    from Language import Grammar
    with pytest.raises(ValueError, match='retired'):
        Grammar()._fill_rule_list([], {'rule': {
            '_': 'r_O1 = r.forward(r_I1, r_I2)', 'implementation': 'symbolize'}})


def test_quantization_is_removed_from_declared_thought_capabilities():
    from Language import Grammar,GRAMMAR_LAYER_CLASSES
    from Queries import THOUGHT_EXECUTORS
    with pytest.raises(ValueError,match='retired'):
        Grammar()._fill_rule_list([],{'rule':'quantize_O1 = quantize.thought(quantize_I1)'},face='thought')
    assert 'quantize' not in THOUGHT_EXECUTORS and 'quantize' not in GRAMMAR_LAYER_CLASSES


def test_mean_composition_is_owned_by_sum_and_keeps_operand_gradients():
    from Language import SumLayer
    layer=SumLayer()
    left=torch.tensor([[.2, -.8]], requires_grad=True)
    right=torch.tensor([[.6, .4]], requires_grad=True)
    parent=layer.compose(left, right)
    torch.testing.assert_close(parent, torch.tensor([[.4, -.2]]))
    parent.sum().backward()
    torch.testing.assert_close(left.grad, torch.full_like(left, .5))
    torch.testing.assert_close(right.grad, torch.full_like(right, .5))
    a,b=layer.generate(parent)
    torch.testing.assert_close(layer.compose(a,b), parent)
    assert not list(layer.parameters())

@pytest.mark.parametrize('name, class_name', [('queryPart','QueryPartLayer'),('queryEqual','QueryEqualLayer')])
def test_scalar_query_experiments_are_retired(name, class_name):
    import Language
    import Layers
    assert not hasattr(Language, class_name)
    assert not hasattr(Layers, class_name)
    assert name not in Language.GRAMMAR_LAYER_CLASSES
    with pytest.raises(ValueError, match='retired'):
        Language.Grammar()._fill_rule_list([], {'S': f'{name}(S)'})
