"""§12 composition semantics and observation-only derivation names."""
from types import SimpleNamespace

import pytest
import torch


def test_probabilistic_disjunction_combines_certainty_and_identity():
    from Language import DisjunctionLayer
    from Layers import Ops
    x = torch.tensor([[.3, .4]], requires_grad=True)
    y = torch.tensor([[0., .6]], requires_grad=True)
    expected = .8 * torch.tensor([[.3, .76]]) / (.3**2 + .76**2)**.5
    op = DisjunctionLayer()
    for value in (op(x, y), op.compose(x, y), Ops.disjunction(x, y)):
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(op.compose(y, x), expected)
    op.compose(x, y).square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert torch.isfinite(y.grad).all() and y.grad.norm() > 0


def test_disjunction_zero_identity_and_zero_direction_have_finite_gradients():
    from Language import DisjunctionLayer
    op = DisjunctionLayer()
    # Last pair has x+y-x*y == 0 at nonzero magnitudes; unit(0) is 0.
    x = torch.tensor([[.3, -.4], [0., 0.], [2., 0.]], requires_grad=True)
    y = torch.tensor([[0., 0.], [0., 0.], [2., 0.]], requires_grad=True)
    value = op.compose(x, y)
    torch.testing.assert_close(value, torch.tensor([[.3, -.4], [0., 0.], [0., 0.]]))
    value.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


@pytest.mark.parametrize('face', ['reverse', 'generate'])
def test_disjunction_free_inverse_recovers_a_pair_without_operand_witnesses(face):
    from Language import DisjunctionLayer
    codes = torch.tensor([[.3, .4], [0., .6], [-.4, -.3]])
    parent = .8 * torch.tensor([[.3, .76]]) / (.3**2 + .76**2)**.5
    op = DisjunctionLayer()
    a, b = getattr(op, face)(parent, basis=codes)
    recovered = {tuple(a[0].tolist()), tuple(b[0].tolist())}
    assert recovered == {tuple(codes[0].tolist()), tuple(codes[1].tolist())}
    torch.testing.assert_close(op.compose(a, b), parent)


@pytest.mark.parametrize('name', ['conjunction', 'disjunction'])
def test_nonlinear_composition_has_affine_xor_rank_while_sum_is_the_control(name):
    from Language import GRAMMAR_LAYER_CLASSES, SumLayer
    left = torch.tensor([[.2, -.3, .4], [.2, -.3, .4], [-.4, .1, .3], [-.4, .1, .3]], dtype=torch.float64)
    right = torch.tensor([[.3, .2, -.1], [-.1, .5, .2], [.3, .2, -.1], [-.1, .5, .2]], dtype=torch.float64)
    def design(op):
        return torch.cat((op.compose(left, right), torch.ones(4, 1, dtype=left.dtype)), dim=-1)
    assert torch.linalg.matrix_rank(design(GRAMMAR_LAYER_CLASSES[name]())) == 4
    assert torch.linalg.matrix_rank(design(SumLayer())) == 3


def test_compose_audit_uses_held_rule_ids_and_names_with_positions():
    from derivation_probe import named_compose_sequence
    language = SimpleNamespace(
        _cs_binary_rule_ids=torch.tensor([7, 12]),
        _cs_unary_rule_ids=torch.tensor([3]),
        _compose_binary_rules=(SimpleNamespace(method_name='conjunction', surface_name='alias_and'),
                               SimpleNamespace(method_name='disjunction', surface_name='alias_or')),
        _compose_unary_rules=(SimpleNamespace(method_name='not', surface_name='not'),))
    expected = [dict(arity=2, rule_id=12, rule_name='disjunction', surface_name='alias_or', position=1),
                dict(arity=1, rule_id=3, rule_name='not', surface_name='not', position=0)]
    assert named_compose_sequence(language, [12, 3], [2, 1], [1, 0]) == expected
    # Unknown IDs must fail rather than guess a name from the current global grammar.
    with pytest.raises(ValueError, match='99'):
        named_compose_sequence(language, [99], [2], [0])


@pytest.mark.parametrize('filename', ['complete.grammar', 'XOR_grammar.xml', 'MM_grammar.xml'])
def test_fixture_rules_select_probabilistic_disjunction(filename):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from Language import Grammar, GRAMMAR_LAYER_CLASSES
    root = ET.parse(Path(__file__).resolve().parents[1] / 'data' / filename).getroot()
    reference = root.find('./SymbolSpace/language/grammar')
    if reference is not None and not len(reference) and (reference.text or '').strip():
        root = ET.parse(Path(__file__).resolve().parents[1] / 'data' / reference.text.strip()).getroot()
    declarations = [r.text for r in root.findall('.//compose//rule') if 'disjunction.forward' in (r.text or '')]
    assert len(declarations) == 1
    grammar = Grammar()
    rules = []
    grammar._fill_rule_list(rules, {'rule': declarations[0]})
    assert rules[0].method_name == 'disjunction'
    x, y = torch.tensor([[.3, .4]]), torch.tensor([[0., .6]])
    value = GRAMMAR_LAYER_CLASSES[rules[0].method_name]().compose(x, y)
    torch.testing.assert_close(value.norm(dim=-1), torch.tensor([.8]))
