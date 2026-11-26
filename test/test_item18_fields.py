"""Direct certificates for the last architecture round's common field."""
import pytest
import torch


@pytest.mark.parametrize('operation', ('intersection', 'union'))
def test_definition_bracket_and_cue_share_weighted_arithmetic(operation):
    from FieldLogic import reduce
    from Layers import SparseLayer
    evidence = torch.tensor([[.2,.6],[.8,.3],[0.,.1]])
    from Layers import NotLayer3, sigma_read
    matrix = SparseLayer(3, 1)
    for column in range(3):
        matrix.add_edge(0,column,weight=1.)
    conjunction = operation == 'intersection'
    expected = torch.tensor([0.,.6] if conjunction else [.8,.1])
    # A definition is the store's Sigma, a sum over the Not's options, and
    # not the bracket's reduce: the three values -.4, .5 and -.1 sum to 0.
    actual = sigma_read(matrix, NotLayer3.options(evidence[:, None]))[0, 0]
    torch.testing.assert_close(actual, torch.tanh(evidence[:, 0] - evidence[:, 1]).new_tensor(
        float(torch.tanh((evidence[:, 0] - evidence[:, 1]).sum()))))
    torch.testing.assert_close(reduce(evidence,operation=operation),expected)


def test_field_negation_silence_and_mask():
    from FieldLogic import reduce
    value = torch.tensor([[.4,.2],[0.,0.]])
    torch.testing.assert_close(reduce(value,operation='not'),value.flip(-1))
    assert not reduce(value,operation='non').any()
    torch.testing.assert_close(reduce(value,operation='intersection',mask=torch.tensor([True,False])),value[0])
    assert not reduce(value,operation='union',mask=torch.tensor([False,False])).any()


@pytest.mark.parametrize('old,new', [('and','intersection'),('or','union')])
def test_retired_field_names_fail_at_parser(old,new):
    from Language import Grammar
    with pytest.raises(ValueError,match=new):
        Grammar().configure({'compose':{'rule':{'_':f'F = {old}.forward(F)','operands':'field'}}})


def test_numeric_scale_order_and_level_code():
    from NumericPercepts import feeling,number
    from BasicPercepts import BasicPercepts
    from Layers import Ops
    values = torch.tensor([0.,1.,2.,3.,10.,float('inf')])
    felt = feeling(values)
    assert felt[0] == 0 and felt[-1] == 1
    assert torch.all(felt[1:]>felt[:-1])
    torch.testing.assert_close(number(felt[:-1]).float(),values[:-1],atol=1e-5,rtol=1e-5)
    basic = BasicPercepts(None,4,4,8)
    basic.bind(None)
    rows = [basic.perceive_value(value) for value in felt[1:]]
    codes = basic.lookup_rows(rows)
    assert len(basic.index) == len(rows)
    # Rule 55: the parts half holds the bins at or below the value, in
    # order; the wholes half its level's group. A lesser value is a part of
    # a greater in the parts half.
    for row, value, code in zip(rows, felt[1:], codes):
        torch.testing.assert_close(code, torch.cat((basic.value_code(value, 4), basic.value_whole(value))))
    assert bool(Ops.partOf(codes[0][:4],codes[1][:4]))
    assert not bool(Ops.partOf(codes[1][:4],codes[0][:4]))






def test_grammar_declares_three_independent_kinds(monkeypatch):
    from Language import Grammar,TheGrammar
    from GrammarPhases import GrammarPhases
    from Spaces import ConceptualSpace
    from types import SimpleNamespace
    def reads_at_order_zero(phases):
        owner=SimpleNamespace(_model=SimpleNamespace(grammar_phases=phases),
            _symbolic_order=0,_serial=phases.per_word)
        return ConceptualSpace._sparse_active(owner)
    for source,expected in [('field.grammar',(False,False)),
                            ('complete.grammar',(True,True))]:
        grammar=Grammar();grammar.load_from_grammar_file(source)
        monkeypatch.setattr(TheGrammar,'_ensure_configured',lambda:None)
        monkeypatch.setattr(TheGrammar,'rules',grammar.rules)
        phases=GrammarPhases.configured()
        assert (phases.vector,phases.interpret)==expected
        assert reads_at_order_zero(phases) == (source=='field.grammar')

    from util import init_config,TheXMLConfig
    init_config(path='data/model.xml',defaults_path='data/model.xml')
    grammar=Grammar();grammar.configure(TheXMLConfig.get('SymbolSpace.language.grammar'))
    monkeypatch.setattr(TheGrammar,'rules',grammar.rules)
    assert not GrammarPhases.configured().per_word
    # §6 keeps the default dense numeric substrate; its declared field
    # names do not enable a new conceptual read at order zero.
    assert not reads_at_order_zero(GrammarPhases.configured())


