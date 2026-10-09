"""Compound selection is a native sigma case read, never a learned new map."""
import pytest
import torch


def _bank():
    from CaseSelection import CaseBank
    # Head 10 has cases 20 and 30. Modifier 40 selects 20; 50 selects 30.
    # Native addresses are beside the numerical codes, not inferred from them.
    ids=torch.tensor([10,20,30,40,50])
    codes=torch.tensor([[.5,.5],[1.,0.],[0.,1.],[.2,.8],[.8,.2]])
    cases=torch.zeros(5,5,2)
    cases[0,1:3,0]=1
    observed=torch.zeros_like(cases)
    observed[torch.arange(5),torch.arange(5),0]=1
    observed[3,1,0]=1
    observed[4,2,0]=1
    return CaseBank(ids,codes,cases,observed)


def test_modifier_selects_the_heads_cases_and_refolds_only_those_cases():
    from CaseSelection import select_cases, fold_cases
    bank=_bank()
    selection=select_cases(bank,torch.tensor([40,50]),torch.tensor([10,10]))
    value=fold_cases(selection)
    torch.testing.assert_close(value,torch.eye(2))
    assert selection.available.tolist() == [True,True]
    assert not list(value.shape) == []
    # Reversing the compound cannot silently treat the modifier as its head.
    assert not select_cases(bank,torch.tensor([10]),torch.tensor([40])).available.any()


def test_case_selection_preserves_negative_evidence_and_silent_absence():
    from CaseSelection import select_cases, fold_cases
    bank=_bank()
    bank.cases[0,1]=torch.tensor([0.,.8])
    bank.observed[3,1]=torch.tensor([0.,.6])
    selected=select_cases(bank,torch.tensor([40,-1]),torch.tensor([10,10]))
    torch.testing.assert_close(fold_cases(selected),torch.tensor([[-.6,0.],[0.,0.]]))
    assert selected.available.tolist() == [True,False]


def test_case_fold_keeps_existing_code_and_membership_gradients():
    from CaseSelection import select_cases, fold_cases
    bank=_bank()
    bank.codes.requires_grad_();bank.observed.requires_grad_()
    value=fold_cases(select_cases(bank,torch.tensor([40]),torch.tensor([10])))
    value.sum().backward()
    assert bank.codes.grad[1].abs().sum() > 0
    assert bank.observed.grad[3,1].abs().sum() > 0
    assert bank.codes.grad[[0,2,3,4]].count_nonzero() == 0


def test_case_selector_and_fold_compile_as_one_tensor_body():
    from CaseSelection import select_cases, fold_cases
    bank=_bank()
    def run(modifier,head):
        selected=select_cases(bank,modifier,head)
        return fold_cases(selected), selected.available
    compiled=torch.compile(run,backend='aot_eager',fullgraph=True)
    actual=compiled(torch.tensor([40,50]),torch.tensor([10,10]))
    torch.testing.assert_close(actual[0],torch.eye(2))
    assert actual[1].all()


def test_native_case_staging_uses_existing_sigma_and_allocates_nothing():
    from CaseSelection import stage_case_bank, select_cases, fold_cases
    from test_cs_sparse_weights import _cs, _mint_row
    cs=_cs()
    # Native test rows are allocated explicitly; sigma already holds the edges.
    first=_mint_row(cs,0,101)
    second=_mint_row(cs,0,102)
    head=_mint_row(cs,1,103)
    modifier=_mint_row(cs,1,104)
    cs.add_concept_edge(head,first,1.)
    cs.add_concept_edge(head,second,1.)
    cs.add_concept_edge(modifier,second,1.)
    before=dict(cs._concept_allocator.placement)
    bank=stage_case_bank(cs,torch.tensor([103,104]),limit=8)
    assert cs._concept_allocator.placement == before
    selected=select_cases(bank,torch.tensor([104]),torch.tensor([103]))
    assert selected.available.all()
    expected=cs.similarity_codebook.lookup_rows(torch.tensor([second]))
    torch.testing.assert_close(fold_cases(selected),expected)


def test_compound_structural_face_reads_only_the_selected_cases():
    from Language import CompoundLayer, invoke_structural_face
    from Queries import StructuralGrammarContext, ConceptualSpaceCapability
    from CaseSelection import select_cases
    bank=_bank()
    selected=select_cases(bank,torch.tensor([40]),torch.tensor([10]))
    context=StructuralGrammarContext((),ConceptualSpaceCapability(2),None,'compose',selected_cases=selected)
    op=CompoundLayer()
    value=invoke_structural_face(op,(bank.codes[3:4],bank.codes[:1]),context=context)
    torch.testing.assert_close(value,torch.tensor([[1.,0.]]))
    assert not list(op.parameters())
    assert op.head_role == 2 and op.case_head_role == 2


def test_compound_candidate_is_masked_when_no_native_case_is_available():
    from ReferenceContext import ReferenceBank, prepare_operands
    from CaseSelection import select_cases
    from Language import Grammar
    bank=_bank()
    language=Grammar();rules=[]
    language._fill_rule_list(rules,{'rule':'compound_O1 = compound.forward(compound_I1, compound_I2)'})
    refs=ReferenceBank(torch.full((2,1),-1),torch.zeros(2,1,2),torch.zeros(2,1,dtype=torch.bool),
        torch.zeros(2,1,dtype=torch.bool),torch.zeros(2,2),torch.zeros(2,dtype=torch.bool),cases=bank)
    window=torch.zeros(2,2,2)
    ids=torch.tensor([[40,10],[-1,10]])
    flags=torch.zeros_like(ids)
    positions=torch.tensor([[0,1],[0,1]])
    prepared=prepare_operands(window,ids,flags,positions,rules=rules,unary_rules=(),bank=refs,
        live=(window,ids,flags,flags.bool(),positions),active=torch.ones_like(ids,dtype=torch.bool))
    assert prepared['binary_valid'][:,0,0].tolist() == [True,False]
    torch.testing.assert_close(prepared['case_weights'][0,0,0],select_cases(bank,ids[0,0],ids[0,1]).weights)


def test_compound_inverse_search_is_bounded_by_the_supplied_primed_symbols():
    from CaseSelection import search_cases
    bank=_bank()
    allowed=torch.tensor([[10,40],[10,50]])
    left,right,available=search_cases(bank,torch.eye(2),allowed_ids=allowed)
    torch.testing.assert_close(left,bank.codes[[3,4]])
    torch.testing.assert_close(right,bank.codes[[0,0]])
    assert available.all()
    _,_,absent=search_cases(bank,torch.eye(2),allowed_ids=torch.tensor([[20],[30]]))
    assert not absent.any()


def test_free_decoder_uses_case_search_without_a_compose_witness():
    from Language import LanguageSpace,CompoundLayer
    from types import SimpleNamespace,MethodType
    from dataclasses import replace
    bank=_bank()._replace(eligible=torch.tensor([[True,False,False,True,False]]))
    owner=SimpleNamespace()
    owner._reverse_of_binary_op=MethodType(LanguageSpace._reverse_of_binary_op,owner)
    owner.reverse_binary_step=MethodType(LanguageSpace.reverse_binary_step,owner)
    left,right,unavailable=owner.reverse_binary_step(torch.tensor([[1.,0.]]),torch.tensor([0]),torch.tensor([True]),
        ops=[CompoundLayer()],return_status=True,free=True,case_bank=bank)
    assert not unavailable.any()
    torch.testing.assert_close(left,bank.codes[3:4])
    torch.testing.assert_close(right,bank.codes[:1])


def test_shipped_grammar_supplies_subtyping_without_a_predefined_surface():
    from Language import Grammar
    grammar=Grammar();grammar.load_from_grammar_file('complete.grammar')
    forward=[r for r in grammar.rules_upward if r.case_head_role]
    reverse=[r for r in grammar.rules_downward if r.case_head_role]
    assert len(forward) == len(reverse) == 1
    assert forward[0].head_role == 2 and forward[0].order_delta == 0
    assert forward[0].method_name not in grammar.surface_anchors.values()


def test_compound_choice_runs_in_the_shared_compiled_chooser():
    from ReferenceContext import ReferenceBank, prepare_operands
    from Language import Grammar,CompoundLayer,OperationSelectionLayer,_BinaryGrammarOpAdapter
    from Queries import StructuralGrammarContext,ConceptualSpaceCapability
    bank=_bank()
    grammar=Grammar();rules=[]
    grammar._fill_rule_list(rules,{'rule':'compound_O1 = compound.forward(compound_I1, compound_I2)'})
    layer=OperationSelectionLayer(d_model=2,ops=[_BinaryGrammarOpAdapter(CompoundLayer())])
    context=StructuralGrammarContext((),ConceptualSpaceCapability(2),None,'compose')
    refs=ReferenceBank(torch.full((1,1),-1),torch.zeros(1,1,2),torch.zeros(1,1,dtype=torch.bool),
        torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,2),torch.zeros(1,dtype=torch.bool),cases=bank)
    def run(window):
        ids=torch.tensor([[40,10]],device=window.device)
        flags=torch.zeros_like(ids)
        positions=torch.tensor([[0,1]],device=window.device)
        data=prepare_operands(window,ids,flags,positions,rules=rules,unary_rules=(),bank=refs,
            live=(window,ids,flags,flags.bool(),positions),active=torch.ones_like(ids,dtype=torch.bool))
        value,_,route=layer(window,depth=torch.tensor([2],device=window.device),slots=1,
            sample=False,grammar_context=context,reference_data=data)
        return value,route['kind']
    compiled=torch.compile(run,backend='aot_eager',fullgraph=True)
    value,kind=compiled(torch.stack((bank.codes[3],bank.codes[0]))[None])
    assert kind.tolist() == [1]
    torch.testing.assert_close(value[:,0],torch.tensor([[1.,0.]]))


def test_answer_case_bank_detaches_reader_keys_without_detaching_parent():
    from CaseSelection import search_cases
    bank=_bank()
    bank.codes.requires_grad_(); bank.cases.requires_grad_(); bank.observed.requires_grad_()
    frozen=bank.detached()
    assert all(not value.requires_grad for value in frozen if torch.is_tensor(value))
    parent=torch.tensor([[1.,0.]],requires_grad=True)
    left,right,valid=search_cases(frozen,parent)
    assert valid.all()
    (left+right).sum().backward()
    assert parent.grad is not None
    assert bank.codes.grad is None and bank.observed.grad is None
