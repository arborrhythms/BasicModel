"""Direct certificates for the last architecture round's common field."""
import pytest
import torch


@pytest.mark.parametrize('operation', ('intersection', 'union'))
def test_definition_bracket_and_cue_share_weighted_arithmetic(operation):
    from FieldLogic import reduce
    from Layers import SparseLayer
    evidence = torch.tensor([[.2,.6],[.8,.3],[0.,.1]])
    matrix = SparseLayer(6, 1)
    for column in range(3):
        matrix.add_edge(0,column,weight=1.)
    source = evidence.T.reshape(6,1)
    conjunction = operation == 'intersection'
    actual = torch.stack([matrix.fold_presence(source, conjunctive=conjunction, dual=dual)[0,0]
                          for dual in (False,True)])
    expected = torch.tensor([0.,.6] if conjunction else [.8,.1])
    torch.testing.assert_close(actual,expected)
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


def test_numeric_scale_order_and_fixed_code():
    from NumericPercepts import feeling,number
    from BasicPercepts import BasicPercepts
    from Layers import Ops
    values = torch.tensor([0.,1.,2.,3.,10.,float('inf')])
    felt = feeling(values)
    assert felt[0] == 0 and felt[-1] == 1
    assert torch.all(felt[1:]>felt[:-1])
    torch.testing.assert_close(number(felt[:-1]).float(),values[:-1],atol=1e-5,rtol=1e-5)
    basic = BasicPercepts(None,4,0,4)
    row = basic.numeric_dimension(2)
    codes = felt[:,None]*basic.code(row)
    assert len(basic.index) == 1
    assert codes[:,:2].count_nonzero() == 0 and codes[:,3:].count_nonzero() == 0
    assert bool(Ops.partOf(codes[1],codes[2]))
    assert not bool(Ops.partOf(codes[2],codes[1]))


def test_numeric_native_field_promotes_and_reader_is_stepped():
    from recon_bench import _build_model
    model,_,_,_ = _build_model('data/MM_xor.xml')
    values = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).unsqueeze(-1)
    owner = model._concept_owner()
    reader = model.outputSpace.field_readout
    parameter = reader.weight
    optimizer = torch.optim.SGD(model.parameters(),lr=.1)
    for _ in range(owner.concept_mint_count+1):
        output = model.forward(values)[2]
    matrix = owner._concept_allocator.layer(0).conjunctive
    assert matrix.nnz == 2
    row = matrix._rows[0]
    assert set(matrix._cols) == set(model.numeric_percepts.rows)
    field = model._numeric_field
    location = (field.concept_ids == owner.concept_id_at_row(row)).nonzero().item()
    torch.testing.assert_close(field.evidence[location,:,0,0],torch.tensor([0.,0.,0.,.4093839]))
    torch.testing.assert_close(model.numeric_percepts.reverse(field),values,atol=1e-6,rtol=1e-6)
    before = parameter.detach().clone()
    output.square().sum().add((output-1).square().sum()).backward()
    optimizer.step()
    assert reader.weight is parameter and not torch.equal(before,parameter)
    model.outputSpace.retire_field_row(row)
    assert not reader.weight[:,2*row:2*row+2].any()


def test_stimulate_uses_coverage_then_closes_definitions():
    from test_cs_sparse_weights import _cs,_mint_row
    from FieldProjection import stimulate
    owner=_cs()
    rows=[_mint_row(owner,0,100+i) for i in range(4)]
    with torch.no_grad():
        owner.similarity_codebook.W[rows]=torch.tensor([
            [1.,0.,0.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.,0.,0.],
            [1.,1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.,0.,0.]])
    owner.add_concept_edge(rows[2],rows[0],1.,conjunctive=True)
    owner.add_concept_edge(rows[2],rows[1],1.,conjunctive=True)
    owner.add_concept_edge(rows[3],rows[0],1.)
    field=stimulate(owner,torch.tensor([[1.,1.,0.,0.,0.,0.,0.,0.],
                                      [1.,0.,0.,0.,0.,0.,0.,0.],
                                      [0.,0.,0.,1.,0.,0.,0.,0.]]))
    torch.testing.assert_close(field.evidence[:,0,0,0],torch.ones(4))
    torch.testing.assert_close(field.evidence[:,1,0,0],torch.tensor([1.,0.,.5,1.]))
    assert not field.evidence[:,2].any()


def test_grammar_declares_three_independent_kinds(monkeypatch):
    from Language import Grammar,TheGrammar
    from GrammarPhases import GrammarPhases
    from Spaces import ConceptualSpace
    from types import SimpleNamespace
    def reads_at_order_zero(phases):
        owner=SimpleNamespace(_model=SimpleNamespace(grammar_phases=phases),
            _symbolic_order=0,_serial=phases.per_word)
        return ConceptualSpace._sparse_active(owner)
    for source,expected in [('field.grammar',(False,False,False)),
                            ('examination.grammar',(True,False,False)),
                            ('complete.grammar',(True,True,True))]:
        grammar=Grammar();grammar.load_from_grammar_file(source)
        monkeypatch.setattr(TheGrammar,'_ensure_configured',lambda:None)
        monkeypatch.setattr(TheGrammar,'rules',grammar.rules)
        phases=GrammarPhases.configured()
        assert (phases.examination,phases.vector,phases.interpret)==expected
        assert reads_at_order_zero(phases) == (source=='field.grammar')

    from util import init_config,TheXMLConfig
    init_config(path='data/model.xml',defaults_path='data/model.xml')
    grammar=Grammar();grammar.configure(TheXMLConfig.get('SymbolSpace.language.grammar'))
    monkeypatch.setattr(TheGrammar,'rules',grammar.rules)
    assert not GrammarPhases.configured().per_word
    # §6 keeps the default dense numeric substrate; its declared field
    # names do not enable a new conceptual read at order zero.
    assert not reads_at_order_zero(GrammarPhases.configured())


def test_intersection_writer_preserves_against_lane_and_examination_proposal_rule():
    from recon_bench import _build_model
    from FamiliarTier import observe_intersection,promote_candidates
    from GrammarPhases import GrammarPhases
    from FieldProjection import close,inventory
    model,*_=_build_model('data/MM_xor.xml')
    owner=model._concept_owner();a,b=model.numeric_percepts.rows
    model.grammar_phases=GrammarPhases(True,False,False,frozenset(('intersection','divide')))
    lanes=((a,0),(b,1))
    assert observe_intersection(owner,lanes) is None
    for _ in range(owner.concept_mint_count):
        key=observe_intersection(owner,lanes,examined=True)
        owner.mtm.close(None,stream=0,references=(key,))
    promoted=promote_candidates(owner)
    assert len(promoted)==1
    row=owner._csw_row_of(promoted[0]);matrix=owner._concept_allocator.layer(0).conjunctive
    assert {(r,c) for r,c in matrix._index}=={(row,a),(row,b+matrix.nOutput+1)}
    rows=inventory(owner);evidence=torch.zeros(len(rows),1,1,2)
    evidence[rows.index(a),0,0]=torch.tensor([.8,.1])
    evidence[rows.index(b),0,0]=torch.tensor([.3,.6])
    folded=close(owner,rows,evidence)
    torch.testing.assert_close(folded[rows.index(row),0,0],torch.tensor([.6,.3]))
