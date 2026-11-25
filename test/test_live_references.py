from order_fixtures import force_order
import pytest
import torch
from ReferenceContext import ReferenceBank, resolve_operand
from nomination_fixtures import recurrent_word


def test_transparent_operation_preserves_a_vp_with_a_relational_operand():
    """A verb over a completed relation closes as its composed idea."""
    from dataclasses import replace
    from types import SimpleNamespace
    import weakref
    from ClauseJournal import finish_clause
    from ClauseRow import predicate_identity
    from test_clause_storage import clause_store, idea_clause, part_clause
    from test_sentence_references import language_and_program
    store, _ = clause_store()
    prior = store.write_clause(part_clause())
    identity = prior[1]
    head_identity = 3
    language, entry = language_and_program()
    language._compose_binary_rules = (SimpleNamespace(method_name='verb',
        clause_form='VP', head_role=1, predicate_identity='verb'),)
    language._compose_unary_rules = (SimpleNamespace(method_name='surface_unary',
        scope_transparent=True, head_role=1),)
    point, other = torch.eye(4)[2], torch.eye(4)[1]
    combined = (point + other) * .5
    # The numerical journal says the verb's right operand is the earlier
    # relation. Its head's point-valued identity survives the unary suffix.
    entry = replace(entry, reference_ids=torch.tensor([head_identity, identity]), word_ids=torch.tensor([-1,-1]),
        actions=torch.tensor([[0,-1,0], [0,-1,1], [1,0,-1], [2,0,-1]]),
        operation_refs=torch.tensor([[-1,-1],[-1,-1],[head_identity,identity],[head_identity,-1]]),
        operation_relations=torch.tensor([[False,False],[False,False],
                                           [False,True],[True,False]]),
        operation_values=torch.stack((torch.zeros(3,4),torch.zeros(3,4),
            torch.stack((point,other,combined)),
            torch.stack((combined,point*0,combined)))),
        end_state=torch.stack((combined,point*0,point*0)))
    registry = SimpleNamespace(space=SimpleNamespace(_clause_store_ref=weakref.ref(store),
        interpret=SimpleNamespace(no_object=lambda reference: False)))
    field = finish_clause(language, entry, registry=registry)
    assert field.relation is None
    torch.testing.assert_close(field.point, combined)
    assert field.slots.shape[0] == 1
    assert field.refs[0] == head_identity
    row = store.write_clause(field)
    assert isinstance(row, int)
    torch.testing.assert_close(store.row(row)['point'], combined)
    assert store.location(identity) == prior
    assert store.parts.contains(identity)


def test_grammar_order_resolution_survives_the_numerical_journal():
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from reading_fixtures import record_reading
    from test_sentence_references import language_and_program
    language, entry = language_and_program()
    language._compose_binary_rules[0].reference_roles=('argin1',)
    language._compose_binary_rules[0].reference_modes = (('argin1', 'generic'),)
    entry = record_reading(language, entry)
    entry = replace(entry, reference_ids=torch.tensor([7, 3]),
                    reference_orders=torch.tensor([2, -1]))
    field = finish_clause(language, entry)
    assert field.refs[0] == 7


@pytest.mark.parametrize('chunk_state', [None, ()], ids=['ordinary', 'identity'])
def test_existing_kind_point_is_resolved_before_the_operator(chunk_state):
    from types import SimpleNamespace
    from ReferenceContext import prepare_operands
    value = torch.tensor([[[1., 0., 0., 0.]]])
    kind = torch.tensor([.2, .3, .4, .5])
    types = SimpleNamespace(sources=torch.tensor([[1]]),
                            orders=torch.tensor([2]), identities=torch.tensor([[[7]]]),
                            values=kind.reshape(1, 1, 1, 4))
    bank = ReferenceBank(torch.empty(1, 0, dtype=torch.long), value[:, :0],
        torch.empty(1, 0, dtype=torch.bool), torch.empty(1, 0, dtype=torch.bool),
        value[:, 0], torch.tensor([False]), types=types, column_ids=torch.empty(0, dtype=torch.long))
    rule = SimpleNamespace(reference_roles=('argin1',),
                           reference_modes=(('argin1', 'generic'),))
    result = prepare_operands(value, torch.tensor([[1]]), torch.tensor([[0]]),
                              torch.tensor([[0]]), orders=torch.tensor([[2]]),
                              rules=(), unary_rules=(rule,), bank=bank, live=None,
                              active=torch.tensor([[True]]), chunk_state=chunk_state)
    torch.testing.assert_close(result['unary'][0, 0, 0], kind, rtol=0, atol=0)
    assert result['unary_refs'][0, 0, 0, 0] == 7
    assert result['unary_ops'] == (0,)


@pytest.mark.parametrize('order, literal, mode, expected', [
    (0, False, None, 7), (2, False, None, 7),
    (1, False, None, 1), (1, False, 'bind', 1),
    (0, True, None, 1), (2, True, None, 1),
    (2, True, 'generic', 7),
])
def test_identity_operand_keeps_order_one_and_implicit_literals(order, literal, mode, expected):
    from types import SimpleNamespace
    from ReferenceContext import prepare_operands, ReferenceTypes
    value = torch.tensor([[[1., 0., 0., 0.]]])
    typed = torch.tensor([.2, .3, .4, .5])
    types = ReferenceTypes(torch.tensor([[1]]), torch.tensor([order]),
                           torch.tensor([[[7]]]), typed.reshape(1, 1, 1, 4))
    bank = ReferenceBank(torch.empty(1, 0, dtype=torch.long), value[:, :0],
        torch.empty(1, 0, dtype=torch.bool), torch.empty(1, 0, dtype=torch.bool),
        value[:, 0], torch.tensor([False]), types=types, interpreter=object(),
        close_references=True, literal_ids=torch.tensor([1]) if literal else None)
    rule = SimpleNamespace(clause_form='S', reference_roles=() if mode is None else ('argin1',),
                           reference_modes=() if mode is None else (('argin1', mode),))
    result = prepare_operands(value, torch.tensor([[1]]), torch.tensor([[0]]),
        torch.tensor([[0]]), orders=torch.tensor([[order]]), rules=(), unary_rules=(rule,),
        bank=bank, live=None, active=torch.tensor([[True]]), chunk_state=())
    torch.testing.assert_close(result['unary'][0, 0, 0],
                               typed if expected == 7 else value[0, 0], rtol=0, atol=0)
    assert result['unary_refs'][0, 0, 0, 0] == expected
    assert result['unary_ops'] == (0,)
    assert result['unary_choices'].eq(-1).all()


@pytest.mark.parametrize('order', [0, 2])
@pytest.mark.parametrize('operation', ['change', 'part', 'whole', 'equal'])
def test_identity_journal_keeps_the_implicit_typed_reference_and_point(order, operation):
    from dataclasses import replace
    from types import SimpleNamespace
    from ClauseJournal import finish_clause
    from ReferenceContext import prepare_operands, ReferenceTypes
    from test_sentence_references import language_and_program
    language, entry = language_and_program()
    rule = SimpleNamespace(method_name=operation, reference_roles=(), reference_modes=(),
        clause_form='S' if operation == 'change' else None,
        relation_kind=None if operation == 'change' else operation, head_role=0)
    language._compose_binary_rules = (rule,)
    typed = torch.tensor([.2, .3, .4, .5])
    types = ReferenceTypes(torch.tensor([[1]]), torch.tensor([order]),
                           torch.tensor([[[7]]]), typed.reshape(1, 1, 1, 4))
    bank = ReferenceBank(torch.empty(1, 0, dtype=torch.long), entry.leaves[None, :0],
        torch.empty(1, 0, dtype=torch.bool), torch.empty(1, 0, dtype=torch.bool),
        entry.leaves[None, 0], torch.tensor([False]), types=types,
        interpreter=object(), close_references=True)
    orders = torch.tensor([[order, 0]])
    options = prepare_operands(entry.leaves[None], entry.concept_ids[None],
        torch.zeros(1, 2, dtype=torch.long), torch.arange(2)[None], orders=orders,
        rules=(rule,), unary_rules=(), bank=bank, live=None,
        active=torch.ones(1, 2, dtype=torch.bool), chunk_state=())
    left, right = options['left'][0, 0, 0], options['right'][0, 0, 0]
    result = (left + right) * .5
    frames = entry.leaves.new_zeros(3, 3, 4)
    frames[-1] = torch.stack((left, right, result))
    references = torch.full((3, 2), -1, dtype=torch.long)
    references[-1] = options['binary_refs'][0, 0, 0]
    entry = replace(entry, leaf_orders=orders[0], reference_orders=torch.full((2,), -1),
        operation_values=frames, operation_refs=references,
        end_state=torch.stack((result, result * 0, result * 0)),
        identity_origins=torch.empty(0, 3, dtype=torch.long),
        identity_actions=torch.zeros(3, 6, dtype=torch.long),
        identity_values=torch.empty(0, 4), identity_leaves=torch.zeros(2, dtype=torch.long))
    clause = finish_clause(language, entry)
    slot = 2 if operation == 'whole' else 0
    assert clause.refs[slot] == 7
    assert clause.factored_refs[slot] == 7
    assert clause.composition.children[0].reference == 7
    torch.testing.assert_close(clause.meaning.roles[slot], typed, rtol=0, atol=0)


def test_unknown_composite_cannot_borrow_an_unrelated_lexical_kind():
    from ReferenceContext import ReferenceTypes, resolve_order
    value = torch.tensor([[[1., 0., 0., 0.]]])
    types = ReferenceTypes(torch.tensor([[1]]), torch.tensor([1, 2]),
                           torch.tensor([[[-1, 7]]]), torch.zeros(1, 1, 2, 4))
    point, identity = resolve_order(value, torch.tensor([[-1]]), 2, types)
    torch.testing.assert_close(point, value, rtol=0, atol=0)
    assert identity.item() == -1


def test_occurrence_is_operand_and_predictor_gradient_is_live():
    value = torch.tensor([[[1., 0., 0., 0.]]])
    earlier = torch.tensor([[[.2, .3, .4, .5]]])
    query = torch.nn.Parameter(earlier[:, 0].clone())
    bank = ReferenceBank(torch.tensor([[5]]), earlier, torch.tensor([[True]]),
                         torch.tensor([[False]]), query, torch.tensor([True]))
    live = (value, torch.tensor([[1]]), torch.tensor([[1]]),
            torch.tensor([[False]]), torch.tensor([[0]]))
    resolved, ref, relation, available = resolve_operand(value, torch.tensor([[1]]), torch.tensor([[0]]),
                                                         mode='particular', bank=bank, live=live, active=torch.tensor([[True]]))
    torch.testing.assert_close(resolved, earlier, rtol=0, atol=0)
    assert ref.item() == 5 and available.item() and not relation.item()
    resolved.sum().backward()
    assert query.grad is not None and query.grad.any()


def test_cold_reference_is_exact_and_pronoun_proposal_is_maskable():
    value = torch.tensor([[[1., 0., 0., 0.]]])
    bank = ReferenceBank(torch.tensor([[-1]]), torch.zeros_like(value), torch.tensor([[False]]),
                         torch.tensor([[False]]), value[:, 0], torch.tensor([False]))
    live = (value, torch.tensor([[1]]), torch.tensor([[1]]),
            torch.tensor([[False]]), torch.tensor([[0]]))
    for mode in ('particular', 'pronoun'):
        result = resolve_operand(value, torch.tensor([[1]]), torch.tensor([[0]]), mode=mode,
                                 bank=bank, live=live, active=torch.tensor([[True]]))
        assert result[-1].item() == (mode == 'particular')
        torch.testing.assert_close(result[0], value, rtol=0, atol=0)


def test_operation_runs_on_the_selected_occurrence_eager_and_compiled():
    from types import SimpleNamespace
    from Language import OperationSelectionLayer, LanguageSpace

    class Sum(torch.nn.Module):
        def forward(self, left, right): return left+right
    class Difference(torch.nn.Module):
        def forward(self,left,right): return left-right
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(), Difference()))
    first = torch.tensor([1., 0., 0., 0.])
    second = torch.tensor([0., 0., 1., 0.])
    prior = torch.tensor([.2, .3, .4, .5])
    query = torch.nn.Parameter(prior.clone())
    from ChunkJournal import ChunkJournal
    rules = (SimpleNamespace(method_name='verb', reference_roles=('argin1',),
                             reference_modes=(('argin1', 'particular'),)),)*2
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
                            _tree_layer=lambda arity: layer,
                            _compose_binary_rules=rules, _compose_unary_rules=(),
                            chunk_journal=ChunkJournal(rules, ()), _structural_context=lambda **kw: None,
                            _reference_bank=ReferenceBank(torch.tensor([[5]]), prior[None, None], torch.tensor([[True]]),
                                                          torch.tensor([[False]]), query[None], torch.tensor([True])))
    owner._symbol_space = SimpleNamespace(conceptualSpace=SimpleNamespace(
        _model=SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=SimpleNamespace(when_start=None)))))
    from Memory import STMChunks as chunks
    state=chunks.empty(1,8,4)
    state,_=chunks.deposit(state,prior[None],torch.tensor([True]),identity=5)
    state,_=chunks.deposit(state,first[None],torch.tensor([True]),identity=5,order=1,stack=True)
    state,_=chunks.deposit(state,second[None],torch.tensor([True]),order=1,stack=True)
    scope=torch.zeros(1,8,2,dtype=torch.long)
    scope[:,:,1]=-1
    scope[0,:2,1]=torch.tensor([3,5])

    def choose(state, scope):
        return LanguageSpace.choose_operation(owner, state, torch.tensor([True]), slots=1, reference_scope=scope)
    from operation_fixtures import selected_action
    with selected_action(layer, torch.tensor([0])):
        eager = choose(state, scope)
        compiled = torch.compile(choose, backend='eager', fullgraph=True)(state, scope)
    for result in (eager, compiled):
        choice, refs, relations, operands = result
        torch.testing.assert_close(choice.candidate[0], prior+second, rtol=0, atol=0)
        assert refs.tolist() == [[5, 3]]
        torch.testing.assert_close(operands[0, 0], prior, rtol=0, atol=0)
        torch.testing.assert_close(operands[0, 1], second, rtol=0, atol=0)
        assert not relations.any()
    (-eager[0].log_probability.sum()).backward()
    assert query.grad is None
    assert any(p.grad is not None and p.grad.any() for p in layer.chooser.parameters())

