from order_fixtures import force_order
import pytest
import torch
from ReferenceContext import ReferenceBank, resolve_operand


def test_transparent_operation_preserves_a_vp_with_a_relational_operand():
    """A verb with a unary surface attachment retains three native roles."""
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
    registry = SimpleNamespace(space=SimpleNamespace(_clause_store_ref=weakref.ref(store)))
    field = finish_clause(language, entry, registry=registry)
    assert field.relation == 'operator' and field.point is None
    assert field.slots.shape[0] == 3
    assert field.refs[0] == head_identity and field.refs[2] == identity
    assert field.refs[1].identity == predicate_identity('verb')
    row = store.write_clause(field)
    assert set(store.row(row)['refs'].tolist()) == {head_identity, identity}
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
@pytest.mark.parametrize('operation', ['lift', 'part', 'whole', 'equal'])
def test_identity_journal_keeps_the_implicit_typed_reference_and_point(order, operation):
    from dataclasses import replace
    from types import SimpleNamespace
    from ClauseJournal import finish_clause
    from ReferenceContext import prepare_operands, ReferenceTypes
    from test_sentence_references import language_and_program
    language, entry = language_and_program()
    rule = SimpleNamespace(method_name=operation, reference_roles=(), reference_modes=(),
        clause_form='S' if operation == 'lift' else None,
        relation_kind=None if operation == 'lift' else operation, head_role=0)
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
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(), Difference()), chooser='mlp')
    first = torch.tensor([1., 0., 0., 0.])
    second = torch.tensor([0., 0., 1., 0.])
    prior = torch.tensor([.2, .3, .4, .5])
    query = torch.nn.Parameter(prior.clone())
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
                            _tree_layer=lambda arity: layer,
                            _compose_binary_rules=(SimpleNamespace(reference_roles=('argin1',), reference_modes=(('argin1', 'particular'),)),)*2,
                            _compose_unary_rules=(), _structural_context=lambda **kw: None,
                            _reference_bank=ReferenceBank(torch.tensor([[5]]), prior[None, None], torch.tensor([[True]]),
                                                          torch.tensor([[False]]), query[None], torch.tensor([True])))
    import STMChunks as chunks
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


def test_public_reading_fuses_the_reference_before_capture_and_write(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    from ReferenceContext import ReferenceBank
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'live_identity', [
        ('<architecture>', '<architecture><ltmConsolidation>true</ltmConsolidation>')])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    model.reconstruct_in_loop = True
    model.loss.reconstruction_scale = 1.
    model.reconstruction_placement = 'eager'

    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    language = model.languageSpace
    rules = list(language._compose_binary_rules)
    op = next(i for i, r in enumerate(rules) if r.method_name == 'lift')
    rules[op] = rules[op]._replace(clause_form='S', head_role=1)
    language._compose_binary_rules = tuple(rules)
    from ChunkJournal import ChunkJournal
    language.chunk_journal = ChunkJournal(rules,language._compose_unary_rules)
    bind = next(i for i, r in enumerate(language._compose_binary_rules) if r.method_name == 'bind')
    layer = language._tree_layer(2)
    original = layer.forward
    routes = []
    bind_done = False

    def forced(x, **kwargs):
        nonlocal bind_done
        reference = kwargs.get('reference_data')
        binary = tuple(range(layer.r_reduce)) if reference is None else reference['binary_ops']
        unary = tuple(range(layer.r_apply)) if reference is None else reference['unary_ops']
        stop = (x.shape[1]-1)*len(binary)+x.shape[1]*len(unary)
        # This fixed reading binds the first participant once, then stops
        # until its predicate arrives. Offering a current identity is legal.
        from operation_fixtures import selected_action
        columns=[i for i,local in enumerate(binary) if local==bind]
        eligible=reference['binary_valid'][:,0,columns] & (reference['binary_refs'][:,0,columns,1]>0)
        at=eligible.long().argmax(-1)
        target=torch.tensor(columns)[at]
        selected=torch.where(kwargs['depth']>=2,binary.index(op),
            torch.where(eligible.any(-1) & (not bind_done),target,stop))
        bind_done |= bool(((selected == target) & eligible.any(-1)).any())
        with selected_action(layer, selected):
            result = original(x, **kwargs)
        routes.append(result[2])
        return result
    monkeypatch.setattr(layer, 'forward', forced)
    width = model.conceptualSpace.stm.concept_dim
    prior = torch.nn.functional.normalize(torch.arange(1, width+1, dtype=torch.float32), dim=0)
    query = torch.nn.Parameter(prior.clone())
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    store = model.symbolSpace.ltm_store
    from ClauseRow import attach_clause_index
    attach_clause_index(model, store, model._concept_owner())
    index = store.write_clause(Clause(ConceptualMeaning.from_description(prior), point=prior, order=1))
    cid = int(store.row_ids[index])
    from Spaces import _concept_alloc_of
    from ReferenceContext import SituationAddress
    owner = model._concept_owner()
    owner.interpret.lookup_word(tuple(map(ord, 'alpha')), (), form='alpha')
    for word in owner.word_concepts('alpha'):
        force_order(owner, word, 1)
    occurrence = store.occurrence_of(index)
    owner.mtm.open(occurrence, kind=None, when=0, value=prior, pending=False)
    frame = SituationAddress(occurrence).read(store)
    # The live pool is prepended by the production boundary; no frame
    # overlay staging override supplies the identity anymore.

    def held(active, like):
        B = active.shape[0]
        return ReferenceBank(torch.full((B, 1), cid, dtype=torch.long), prior[None, None].expand(B, 1, width),
                             torch.ones(B, 1, dtype=torch.bool), torch.zeros(
                                 B, 1, dtype=torch.bool),
                             query[None].expand(B, width), torch.ones(B, dtype=torch.bool))
    monkeypatch.setattr(model, '_sentence_reference_bank', held)
    observed = []
    observation = model._sentence_observation

    def capture(*a, **kw):
        view = observation(*a, **kw)
        if not kw.get('admit', False):
            observed.append(view)
        return view
    monkeypatch.setattr(model, '_sentence_observation', capture)
    try:
        model(model.inputSpace.prepInput(['alpha beta']))
        view = observed[-1]
        entry = view['entries'][0]
        clause = view['clauses'][0]
        actions = (entry.actions[:, 0] == 1) & (entry.actions[:,1] == op)
        assert int(actions.sum()) == len(entry.leaves)-1
        frame = entry.operation_values[actions][-1]
        content_width=store.when_start-4
        torch.testing.assert_close(frame[0,:content_width],
            torch.maximum(entry.leaves[0],prior)[:content_width],rtol=0,atol=0)
        # Prepended occurrences now carry their own current percept bands.
        assert torch.isfinite(frame[0,content_width:]).all()
        # A fixed projection evaluated as a single word or a bank GEMM can
        # differ by float32 accumulation order; it retains the same direction.
        torch.testing.assert_close(frame[1,:content_width], entry.leaves[-1,:content_width], rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(frame[1,store.when_start:], entry.symbol_when[-1],rtol=0,atol=0)
        assert entry.identity_actions[actions][-1, 1].item() == 1
        assert clause.children[0].identity['handle'] == 1
        assert tuple(entry.identity_origins[0, :2].tolist()) == (2, occurrence[-1])
        torch.testing.assert_close(clause.composition.point, frame[2], rtol=0, atol=0)
        # Thought may complete a different working field after the close.
        # The shared recent chunk retains the written sentence's own value.
        import STMChunks as chunks
        state = model.conceptualSpace.stm.state
        written = model._sentence_fields[0][0].row_id
        recent = state[chunks.REFS][0,:,0,1] == written
        assert recent.sum() == 1 and state[chunks.STACK][0,recent].eq(-1).all()
        torch.testing.assert_close(state[0][0,recent][0,:content_width],clause.point[:content_width],rtol=0,atol=0)
        torch.testing.assert_close(state[chunks.WHEN][0,recent][0],store.entry_when(store.location(written)),rtol=0,atol=0)
        assert not torch.equal(entry.leaves[0], prior)
        active = model.inputSpace._word_active_mask
        # This held referent has no lexical realization in the primed bank.
        # Completing a shorter approximate word pair would lose its content;
        # an honest free inverse must leave it pending. Source leaves cannot
        # supply the displaced word as an inverse witness.
        assert model._recon_truncated.tolist() == [True]
        from dataclasses import replace
        record = view['record']
        expected = model._reconstruct_trial(record)[0]
        changed = model._reconstruct_trial(replace(record, word_values=record.word_values+19))[0]
        torch.testing.assert_close(model._recon_ideas[active], expected[active], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(changed[active], expected[active], rtol=0, atol=0)
        clause.composition.point.sum().backward()
        assert query.grad is None
        route = next(route for route in routes if bool((route['kind'] == 1).any()))
        credit = route['logits'].log_softmax(-1).gather(1, route['action'][:, None]).sum()
        (-credit).backward()
        assert any(p.grad is not None and p.grad.any() for p in layer.chooser.parameters())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
