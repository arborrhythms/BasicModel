import torch
from ReferenceContext import ReferenceBank, resolve_operand


def test_transparent_operation_preserves_a_vp_with_a_relational_operand():
    """The failed development verb/tense suffix retains three native roles."""
    from dataclasses import replace
    from types import SimpleNamespace
    import weakref
    from ClauseJournal import finish_clause
    from ClauseRow import predicate_identity
    from test_clause_storage import clause_store, idea_clause, part_clause
    from test_sentence_references import language_and_program
    store, _ = clause_store()
    prior = store.write_clause(part_clause())
    identity = int(store.row_ids[prior])
    head_row = store.write_clause(idea_clause(point=torch.eye(4)[0]))
    head_identity = int(store.row_ids[head_row])
    language, entry = language_and_program()
    language._compose_binary_rules = (SimpleNamespace(method_name='verb',
        clause_form='VP', head_role=1, predicate_identity='verb'),)
    language._compose_unary_rules = (SimpleNamespace(method_name='tense',
        scope_transparent=True, head_role=0),)
    point, other = entry.leaves
    # The numerical journal says the verb's right operand is the earlier
    # relation. Its head's point-valued identity survives the unary suffix.
    entry = replace(entry, reference_ids=torch.tensor([head_identity, identity]), word_ids=torch.tensor([1,3]),
        actions=torch.tensor([[0,-1,0], [0,-1,1], [1,0,-1], [2,0,-1]]),
        operation_refs=torch.tensor([[-1,-1],[-1,-1],[head_identity,identity],[head_identity,-1]]),
        operation_relations=torch.tensor([[False,False],[False,False],
                                           [False,True],[True,False]]),
        operation_values=torch.stack((torch.zeros(3,4),torch.zeros(3,4),
            torch.stack((point,other,point+other)),
            torch.stack((point+other,point*0,point+other)))),
        end_state=torch.stack((point+other,point*0,point*0)))
    registry = SimpleNamespace(space=SimpleNamespace(_clause_store_ref=weakref.ref(store)))
    field = finish_clause(language, entry, registry=registry)
    assert field.relation == 'operator' and field.point is None
    assert field.slots.shape[0] == 3
    assert field.refs[0] == head_identity and field.refs[2] == identity
    assert field.refs[1].identity == predicate_identity('verb')
    row = store.write_clause(field)
    assert int(store.rel_type[row]) == store.REL_OPERATOR
    child = store.index_of_row(int(store.refs[row,2]))
    assert child == prior and int(store.rel_type[child]) == store.REL_PARTOF


def test_grammar_order_resolution_survives_the_numerical_journal():
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from reading_fixtures import record_reading
    from test_sentence_references import language_and_program
    language, entry = language_and_program()
    language._compose_binary_rules[0].reference_orders = (('I1', 2),)
    language._compose_binary_rules[0].reference_kinds = (('I1', 'generic'),)
    entry = record_reading(language, entry)
    entry = replace(entry, reference_ids=torch.tensor([7, 3]),
                    reference_orders=torch.tensor([2, -1]))
    field = finish_clause(language, entry)
    assert field.refs[0] == 7


def test_existing_kind_point_is_resolved_before_the_operator():
    from types import SimpleNamespace
    from ReferenceContext import prepare_operands
    value = torch.tensor([[[1., 0., 0., 0.]]])
    kind = torch.tensor([.2, .3, .4, .5])
    types = SimpleNamespace(sources=torch.tensor([[1]]),
                            orders=torch.tensor([2]), identities=torch.tensor([[[7]]]),
                            values=kind.reshape(1, 1, 1, 4))
    bank = SimpleNamespace(types=types)
    rule = SimpleNamespace(reference_orders=(('I1', 2),),
                           reference_kinds=(('I1', 'generic'),))
    result = prepare_operands(value, torch.tensor([[1]]), torch.tensor([[0]]),
                              torch.tensor([[0]]), rules=(), unary_rules=(rule,), bank=bank, live=None,
                              active=torch.tensor([[True]]))
    torch.testing.assert_close(result['unary'][0, 0, 0], kind, rtol=0, atol=0)
    assert result['unary_refs'][0, 0, 0, 0] == 7


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
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(),), chooser='mlp')
    first = torch.tensor([1., 0., 0., 0.])
    second = torch.tensor([0., 0., 1., 0.])
    prior = torch.tensor([.2, .3, .4, .5])
    query = torch.nn.Parameter(prior.clone())
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
                            _compose_binary_rules=(SimpleNamespace(reference_orders=(
                                ('I1', 1),), reference_kinds=(('I1', 'particular'),)),),
                            _compose_unary_rules=(), _structural_context=lambda **kw: None,
                            _reference_bank=ReferenceBank(torch.tensor([[5]]), prior[None, None], torch.tensor([[True]]),
                                                          torch.tensor([[False]]), query[None], torch.tensor([True])))
    buffer = torch.stack((second, first, torch.zeros_like(first)))[None]
    state = (buffer, torch.tensor([2]), torch.tensor([[1, 1, -1]]), torch.zeros(1, 3, dtype=torch.long),
             torch.tensor([[2, 0, -1]]), torch.ones(1, 3))
    scope = torch.tensor([[[0, 3], [0, 1], [0, -1]]])

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
    rules[op] = rules[op]._replace(reference_orders=(
        ('I1', 1),), reference_kinds=(('I1', 'particular'),))
    language._compose_binary_rules = tuple(rules)
    layer = language._tree_layer(2)
    original = layer.forward
    routes = []

    def forced(x, **kwargs):
        reference = kwargs.get('reference_data')
        binary = tuple(range(layer.r_reduce)) if reference is None else reference['binary_ops']
        unary = tuple(range(layer.r_apply)) if reference is None else reference['unary_ops']
        stop = (x.shape[1]-1)*len(binary)+x.shape[1]*len(unary)
        # Force the grammar operation with the first (held-column) binding,
        # whose global action is no longer its local operation number.
        from operation_fixtures import selected_action
        with selected_action(layer, torch.where(kwargs['depth'] >= 2, binary.index(op), stop)):
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
    index = store.write_clause(Clause(ConceptualMeaning.from_description(prior), point=prior))
    cid = int(store.row_ids[index])

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
        actions = entry.actions[:, 0] == 1
        assert int(actions.sum()) == len(entry.leaves)-1
        frame = entry.operation_values[actions][-1]
        torch.testing.assert_close(frame[0], prior, rtol=0, atol=0)
        # A fixed projection evaluated as a single word or a bank GEMM can
        # differ by float32 accumulation order; it retains the same direction.
        torch.testing.assert_close(frame[1], entry.leaves[-1], rtol=1e-6, atol=2e-7)
        assert entry.operation_refs[actions][-1, 0].item() == cid
        assert clause.refs[0] == cid
        torch.testing.assert_close(clause.point, frame[2], rtol=0, atol=0)
        torch.testing.assert_close(
            model._sentence_fields[0][0].end_state[0], frame[2], rtol=0, atol=0)
        assert not torch.equal(entry.leaves[0], prior)
        active = model.inputSpace._word_active_mask
        assert not model._recon_truncated.any()
        # A free inverse searches from the exact fused root and rule sequence;
        # it is not required to replay the original word displaced by a live
        # reference. The source leaves must not act as inverse witnesses.
        from dataclasses import replace
        record = view['record']
        expected = model._reconstruct_trial(record)[0]
        changed = model._reconstruct_trial(replace(record, word_values=record.word_values+19))[0]
        torch.testing.assert_close(model._recon_ideas[active], expected[active], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(changed[active], expected[active], rtol=0, atol=0)
        clause.point.sum().backward()
        assert query.grad is None
        route = next(route for route in routes if bool((route['kind'] == 1).any()))
        credit = route['logits'].log_softmax(-1).gather(1, route['action'][:, None]).sum()
        (-credit).backward()
        assert any(p.grad is not None and p.grad.any() for p in layer.chooser.parameters())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
