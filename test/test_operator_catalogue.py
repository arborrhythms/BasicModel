"""Decided operator semantics; fixed examples are not training seeds."""
import torch
import pytest


def test_intersection_is_idempotent_symmetric_and_silence_preserving():
    from Language import IntersectionLayer
    op = IntersectionLayer()
    noun = torch.tensor([[.7, -.6, .4, 0., -.5]], requires_grad=True)
    adjective = torch.tensor([[.5, -.8, 0., -.3, .4]], requires_grad=True)
    expected = torch.tensor([[.5, -.8, .4, -.3, -.5]])
    value = op.compose(noun, adjective)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(op.compose(adjective, noun), value, rtol=0, atol=0)
    torch.testing.assert_close(op.compose(noun, noun), noun, rtol=0, atol=0)
    torch.testing.assert_close(op.compose(value, value), value, rtol=0, atol=0)
    torch.testing.assert_close(op.compose(value, adjective), value, rtol=0, atol=0)
    torch.testing.assert_close(op.compose(noun, torch.ones_like(noun)),
                               torch.where(noun == 0, 1., noun), rtol=0, atol=0)
    value.sum().backward()
    assert noun.grad.isfinite().all() and adjective.grad.isfinite().all()
    assert noun.grad.norm() > 0 and adjective.grad.norm() > 0


def test_non_excludes_without_affirming_or_claiming_an_inverse():
    from Language import NonLayer
    poles = torch.tensor([[.7, .2, 19.], [0., .8, 3.]])
    op = NonLayer(representation='poles')
    torch.testing.assert_close(op(poles), torch.tensor([[0., .2, 19.], [0., .8, 3.]]))
    assert not op.invertible
    with pytest.raises((RuntimeError, NotImplementedError)):
        op.reverse(poles)
    code = NonLayer()
    torch.testing.assert_close(code(torch.tensor([[.5, -.4]])), torch.tensor([[.5, -.4]]))


def test_sum_is_a_mean_and_witness_inverse_is_exact():
    from Language import SumLayer, LanguageSpace
    from types import SimpleNamespace, MethodType
    op = SumLayer()
    left, right = torch.tensor([[.2, -.8]]), torch.tensor([[.6, .4]])
    parent = op.compose(left, right)
    torch.testing.assert_close(parent, (left+right)/2)
    owner = SimpleNamespace()
    for name in ('_reverse_of_binary_op', '_finish_binary_inverse', 'reverse_binary_step'):
        setattr(owner, name, MethodType(getattr(LanguageSpace, name), owner))
    a,b,bad = owner.reverse_binary_step(parent, torch.tensor([0]), torch.tensor([True]),
        right, ops=[op], inverses=[None], return_status=True)
    assert not bad.any()
    torch.testing.assert_close(a,left)
    torch.testing.assert_close(b,right)


def test_adverb_is_a_repeatable_gain_with_an_exact_witness_inverse():
    from Language import AdverbLayer
    op = AdverbLayer(nInput=4, nOutput=4)
    with torch.no_grad():
        op._adv_edit.weight.copy_(torch.eye(4)*.5)
        op._adv_edit.bias.zero_()
    verb = torch.tensor([[.2, -.3, 0., .4]], requires_grad=True)
    modifier = torch.ones(1,4)
    once = op.compose(verb,modifier)
    twice = op.compose(once,modifier)
    # Gains compose multiplicatively in the verb's existing chart.
    gain = torch.atanh(once)/torch.where(verb==0, 1., torch.atanh(verb))
    expected = torch.tanh(torch.atanh(verb)*gain.square())
    torch.testing.assert_close(twice, expected)
    assert (twice.abs()[verb!=0] > once.abs()[verb!=0]).all()
    recovered, witness = op.reverse(once, adverb_what=modifier)
    torch.testing.assert_close(recovered, verb)
    torch.testing.assert_close(witness, modifier)
    once.sum().backward()
    assert op._adv_edit.weight.grad.norm() > 0


def test_declared_implementation_and_predicate_survive_rule_rename():
    from Language import Grammar
    from ClauseScope import ClauseScope
    grammar = Grammar()
    rules = []
    grammar._fill_rule_list(rules, {'rule': {'_': 'r_O1 = r.forward(r_I1, r_I2)',
        'implementation': 'verb', 'clause': 'VP', 'head': 'I1',
        'predicate': 'activity'}})
    rule = rules[0]
    assert rule.method_name == 'verb'
    assert rule.surface_name == 'r'
    assert rule.predicate_identity == 'activity'
    assert grammar._thought_contract_from_rule(rule,'forward') == ('verb', ('I1','I2'), 'O1')
    table = ClauseScope(rules, []).binary
    assert table[0, :4].tolist() == [0, 0, 1, 1]


def test_explicit_effects_and_operand_kinds_are_checked_at_load():
    from Language import Grammar
    grammar = Grammar()
    base = dict(_='x_O1 = x.forward(x_I1, x_I2)', implementation='verb')
    with pytest.raises(ValueError, match='field'):
        grammar._fill_rule_list([], {'rule': base | {'operands': 'field,field'}})
    with pytest.raises(ValueError, match='writes'):
        grammar._fill_rule_list([], {'rule': base | {'writes': 'ltm'}})


def test_inverse_dispatch_uses_declared_computation_not_operator_name():
    from Language import SumLayer, LanguageSpace, NotLayer, NonLayer
    from types import SimpleNamespace, MethodType
    owner = SimpleNamespace()
    for method in ('_reverse_of_binary_op', '_finish_binary_inverse',
                   'reverse_binary_step', 'reverse_unary_step'):
        setattr(owner, method, MethodType(getattr(LanguageSpace, method), owner))
    left, right = torch.tensor([[.2, -.8]]), torch.tensor([[.6, .4]])
    mean = SumLayer()
    mean.rule_name = 'renamed_mean'
    a, b, unavailable = owner.reverse_binary_step(
        mean.compose(left, right), torch.tensor([0]), torch.tensor([True]),
        right, ops=[mean], inverses=[None], return_status=True)
    assert not unavailable.any()
    torch.testing.assert_close(a, left)
    torch.testing.assert_close(b, right)
    negate = NotLayer()
    negate.rule_name = 'renamed_negation'
    value, unavailable = owner.reverse_unary_step(
        negate(left), torch.tensor([0]), torch.tensor([True]),
        ops=[negate], return_status=True)
    assert not unavailable.any()
    torch.testing.assert_close(value, left)
    _, unavailable = owner.reverse_unary_step(
        left, torch.tensor([0]), torch.tensor([True]),
        ops=[NonLayer()], return_status=True)
    assert unavailable.all()


def test_clause_scope_uses_declared_roles_after_names_are_removed():
    from Language import Grammar
    from ClauseScope import ClauseScope
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    binary = [r for r in grammar.rules_upward if r.arity == 2 and r.space_role == 'CS']
    unary = [r for r in grammar.rules_upward if r.arity == 1]
    expected = ClauseScope(binary, unary)
    renamed = ClauseScope([r._replace(method_name='x') for r in binary],
                          [r._replace(method_name='x') for r in unary])
    torch.testing.assert_close(renamed.binary, expected.binary)
    torch.testing.assert_close(renamed.unary, expected.unary)
    torch.testing.assert_close(renamed.binary_same_reference, expected.binary_same_reference)


@pytest.mark.parametrize('filename', ['complete.grammar', 'ladder.grammar',
                                      'tied_reconstruction_output_benchmark.grammar'])
def test_retired_operators_are_absent_from_shipped_grammars(filename):
    from Language import Grammar, GRAMMAR_LAYER_CLASSES
    from Queries import THOUGHT_EXECUTORS
    grammar = Grammar()
    grammar.load_from_grammar_file(filename)
    retired = {'exist', 'true', 'lookup'}
    assert retired.isdisjoint(THOUGHT_EXECUTORS)
    assert retired.isdisjoint(GRAMMAR_LAYER_CLASSES)
    structural = {r.method_name for r in grammar.rules_upward + grammar.rules_downward}
    assert (retired | {'quantize', 'arma'}).isdisjoint(structural)
    assert all('exist' not in start for start in grammar.ws_absolute_starts)


@pytest.mark.parametrize('name', ['exist', 'true', 'lookup'])
def test_retired_operator_declarations_fail_loudly(name):
    from Language import Grammar
    with pytest.raises(ValueError, match='retired'):
        Grammar().configure({'thought': {'rule': f'{name}_O1 = {name}.thought({name}_I1)'}})


def test_equality_returns_conceptual_content_with_its_evidence():
    from test_query_vp_boundaries import _context, _signature
    from test_cs_symbol_table import _cs
    idea = torch.arange(8, dtype=torch.float32) / 10
    result = _signature('equal', 'I1', 'I2').invoke(_context(_cs()), idea, idea)
    assert result['result_kind'] == 'concept'
    torch.testing.assert_close(result['value'], idea)
    assert result['support_true'] == pytest.approx(1.)


def test_output_operand_preserves_live_slots_after_exclusion():
    from types import SimpleNamespace
    from Models import BasicModel
    owner=SimpleNamespace(conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=4)))
    idea=torch.tensor([[[0.,0.],[1.,0.],[0.,2.]], [[3.,0.],[0.,0.],[0.,4.]]])
    stacked=BasicModel._walk_operand(owner,idea)
    torch.testing.assert_close(stacked[:,:2], torch.tensor([[[0.,2.],[1.,0.]],[[0.,4.],[3.,0.]]]))
    assert torch.equal(stacked[:,2:],torch.zeros_like(stacked[:,2:]))
    infix=BasicModel._walk_operand(owner,idea,infix_rows=(True,True))
    torch.testing.assert_close(infix[:,:2], torch.tensor([[[1.,0.],[0.,2.]],[[3.,0.],[0.,4.]]]))


def test_relative_detection_reads_rule_declarations_after_renaming():
    from Language import Grammar
    grammar=Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    grammar.ws_relative_starts=frozenset()
    grammar.rules=[r._replace(method_name='opaque') for r in grammar.rules]
    grammar._relative_rule_ids_cache=None
    expected={i for i,r in enumerate(grammar.rules) if r.relation_kind in ('equal','part','whole','implies')}
    assert expected
    assert grammar._relative_rule_id_set()==expected


def test_verb_declares_first_operand_as_head():
    from Language import Grammar
    grammar=Grammar();grammar.load_from_grammar_file('complete.grammar')
    verb=next(rule for rule in grammar.rules_upward if rule.method_name == 'verb')
    assert verb.head_role == 1
    assert verb.clause_form == 'VP'


def test_verb_head_does_not_erase_its_object_in_the_clause_journal(monkeypatch):
    from Language import Grammar
    import Language
    from ClauseJournal import finish_clause
    from Understanding import AnswerProgram
    from types import SimpleNamespace
    grammar=Grammar();grammar.load_from_grammar_file('complete.grammar')
    monkeypatch.setattr(Language, 'TheGrammar', grammar)
    by_name={rule.method_name:rule for rule in grammar.rules_upward}
    # Exercise the required head even against the old unheaded declaration.
    verb=by_name['verb']._replace(head_role=1)
    owner=SimpleNamespace(_compose_binary_rules=(verb, by_name['lift']), _compose_unary_rules=())
    leaves=torch.eye(4)[:3]
    vp=leaves[1] + .25*leaves[2]
    point=.5*(leaves[0]+vp)
    frames=torch.zeros(5,3,4)
    frames[3]=torch.stack((leaves[1],leaves[2],vp))
    frames[4]=torch.stack((leaves[0],vp,point))
    entry=AnswerProgram(rows=torch.arange(3),word_rows=torch.arange(3),
        activations=torch.ones(3),leaves=leaves,
        actions=torch.tensor([[0,-1,0],[0,-1,1],[0,-1,2],[1,0,-1],[1,1,-1]]),
        targets=torch.tensor([1,-1]),end_state=point[None],
        concept_ids=torch.tensor([101,102,103]), operation_values=frames)
    result=finish_clause(owner,entry)
    assert result.factored_refs == (101,102,103)
    torch.testing.assert_close(result.meaning.roles,leaves)
    torch.testing.assert_close(result.point,point)


def test_selected_exclusion_cannot_be_recovered_as_negation(monkeypatch):
    from test_selected_relation_meaning import _program_owner
    from dataclasses import replace
    _cs,_grammar,registry,owner,_leaves,program,_a,_b=_program_owner(monkeypatch)
    local=next(i for i,rule in enumerate(owner._compose_unary_rules) if rule.polarity_effect == 'exclude')
    entry=program()
    entry=replace(entry,actions=torch.cat((entry.actions,torch.tensor([[2,local,-1]]))))
    # Exclusion lives on the clause's evidence. This request adapter has no
    # evidence carrier and must leave it to closing, never invent opposite truth.
    assert owner.program_meaning(entry,registry) is None


def test_equality_question_uses_a_declared_formula_identity_without_an_inventory_row():
    from test_grammatical_query_vps import _world
    from ClauseRow import predicate_identity, predicate_point
    cs,registry,a,b,_context=_world()
    question=registry.form('equal',a,b)
    descriptor=registry.descriptors['equal']
    assert descriptor.predicate_name == 'operation:equal'
    assert descriptor.relation_directions == ((0,2),(2,0))
    assert question.role_refs[1] == ('sym',predicate_identity(descriptor.predicate_name))
    assert cs._csw_row_of(question.role_refs[1][1]) is None
    torch.testing.assert_close(question.roles[1],predicate_point(descriptor.predicate_name,question.roles))
    assert registry.clause_reference('equal') == registry.clause_reference('part')
    assert registry.signature_for(question).operation.semantic_id == 'equal'
    assert not any('conceptual-identity:equal' in name for name in cs._frozen_named)
