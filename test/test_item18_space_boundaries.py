"""Round C's invariance certificate at the protected mixed-order prime."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch.nn import functional as F


def word_loop_prime(model, reducer):
    """Execute the actual protected producer block, without a grammar teacher."""
    import Models
    tree = ast.parse(Path(Models.__file__).read_text())
    method = next(node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == '_run_sentence_word_bricks')
    block = next(node for node in ast.walk(method) if isinstance(node, ast.If)
        and any(isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == '_expectation_prime'
                for target in statement.targets) for statement in node.body))
    wrapper = ast.parse('def read(self, before, B, reducer):\n    pass\n')
    wrapper.body[0].body = [block, ast.parse('return reducer._expectation_prime').body[0]]
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.fix_missing_locations(wrapper), Models.__file__, 'exec'), namespace)
    return namespace['read'](model, ((torch.zeros(1, 1, 3),),), 1, reducer)


def test_word_loop_priming_score_is_invariant_to_other_order_rotation():
    from Language import OperationSelectionLayer
    reducer = OperationSelectionLayer(d_model=3)
    keys = torch.eye(3)[:2].clone()
    prediction = NS(symbol_keys=keys, symbol_orders=torch.tensor([0, 1]),
        expected_activation=torch.tensor([[1., 0.], [1., 0.]]))
    model = NS(expectation_prime_enabled=True,
        symbolSpace=NS(expectation=NS(_inter_last_meaning=[NS(prediction=prediction)])))
    # The scored candidate is an order-zero point and is held fixed. Only
    # the order-one bank row rotates, preserving that order's geometry.
    candidate = torch.tensor([[[0., 1., 0.]]])
    orders = torch.zeros(1, 1, dtype=torch.long)
    first = reducer.prime_bias(candidate, word_loop_prime(model, reducer), orders)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1e-6, rtol=0)
    prediction.symbol_keys = keys.clone()
    prediction.symbol_keys[1] = keys[1] @ rotation
    second = reducer.prime_bias(candidate, word_loop_prime(model, reducer), orders)
    print('protected prime score before/after other-order rotation:', first.tolist(), second.tolist())
    torch.testing.assert_close(second, first, atol=1e-6, rtol=0)


def test_prime_uses_its_own_order_for_value_and_normalization():
    from Expectation import OrderPrime
    from Language import OperationSelectionLayer
    reducer = OperationSelectionLayer(d_model=3)
    prediction = NS(symbol_keys=torch.eye(3), symbol_orders=torch.tensor([0, 0, 1]),
        expected_activation=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
    prime = OrderPrime.from_predictions([prediction], torch.zeros(1, 3), 3)
    candidates = torch.eye(3)[None]
    result = reducer.prime_bias(candidates, prime, torch.tensor([[0, 0, 1]]))
    torch.testing.assert_close(result, torch.tensor([[.5, .5, 1.]]))
    prediction.expected_activation[2] = 0
    changed = OrderPrime.from_predictions([prediction], torch.zeros(1, 3), 3)
    torch.testing.assert_close(reducer.prime_bias(candidates, changed, torch.tensor([[0, 0, 1]])),
                               torch.tensor([[.5, .5, 0.]]))
    torch.testing.assert_close(reducer.prime_bias(candidates, prime, torch.full((1, 3), -1)),
                               torch.zeros(1, 3))


def test_compose_prime_orders_follow_heads_lifting_and_timeline():
    from ReferenceContext import composition_orders, OrderReference
    orders = torch.tensor([[0, 1], [2, 2]])
    ids = tuple(OrderReference(torch.tensor([10+i, 20+i]), torch.tensor([False, True])) for i in range(2))
    binary = (NS(method_name='modifier', head_role=1), NS(method_name='change'))
    unary = (NS(method_name='lift', order_delta=1), NS(method_name='sign'))
    result = composition_orders(orders, ids, binary, unary)
    torch.testing.assert_close(result, torch.tensor([[0, 0, 1, 0, 2, 0, -1],
                                                     [2, 2, 3, 0, 3, 0, -1]]))


def mixed_field():
    from argument_fixtures import argument_inventory
    from Understanding import ConceptualField
    from placement_fixtures import place_at
    owner, speech, words = argument_inventory(('modifier_argin1',))
    cid = speech.concepts['modifier_argin1']
    speech.add(words[0], 'modifier_argin1')
    place_at(owner, cid, [0.,1.,0.,0.])
    ids = torch.tensor([words[0], cid])
    field = ConceptualField(ids, torch.tensor([[[[1.,0.]]],[[[.8,.2]]]]),
                            torch.tensor([[[0,1]]]), torch.tensor([[[0,1]]]), None)
    return owner, field


def test_relation_vector_read_ignores_endpoints_of_other_orders():
    from part_fixtures import native_store
    store, _ = native_store(4,order_limit=1)
    owner = store._concept_owner()
    a, b = (owner.new_concept(order=1) for _ in range(2))
    with torch.no_grad():
        for i,cid in enumerate((a,b)):
            owner.similarity_codebook.W[owner._csw_concept_row(1,cid)] = torch.eye(4)[i]
    store.append_relation(a,b,truth_plus=1.)
    before = store.evaluate(torch.eye(4)[0],torch.eye(4)[1],orders=(0,0))
    assert before.tolist() == [0.,0.]
    assert store.consequents(torch.eye(4)[0], order=0) == []
    rotation, _ = torch.linalg.qr(torch.randn(4,4))
    with torch.no_grad():
        for cid in (a,b):
            row = owner._concept_row_of(cid)
            owner.similarity_codebook.W[row] = owner.similarity_codebook.W[row] @ rotation
    torch.testing.assert_close(store.evaluate(torch.eye(4)[0],torch.eye(4)[1],orders=(0,0)),before)
    assert store.consequents(torch.eye(4)[0], order=0) == []


def test_parameter_codebook_loss_and_gradient_do_not_read_another_order():
    from Models import BasicModel
    rows = torch.nn.Parameter(torch.tensor([[1.,.2,0.],[.2,1.,0.],[0.,.2,1.],[1.,0.,.2]]))
    owner = NS(similarity_codebook=NS(prototype=lambda: rows),order_slice=lambda k:(2*k,2*k+2))
    model = NS(conceptualSpace=owner,word_brackets=False)
    values = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
    state = torch.get_rng_state()
    def read():
        model._cs_parallel_slab = ((0,values),)
        model._cs_parallel_slab_cs = owner
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state)
            loss = BasicModel.conceptual_sbow_loss(model)
        assert loss is not None
        return loss.detach(), torch.autograd.grad(loss,rows)[0]
    loss, gradient = read()
    rotation, _ = torch.linalg.qr(torch.randn(3,3))
    with torch.no_grad():
        rows[2:] = rows[2:] @ rotation
    after, after_gradient = read()
    torch.testing.assert_close(after,loss,atol=1e-6,rtol=0)
    torch.testing.assert_close(after_gradient[:2],gradient[:2],atol=1e-6,rtol=0)
    assert gradient[2:].eq(0).all() and after_gradient[2:].eq(0).all()


def test_numeric_field_key_keeps_its_order():
    from FieldProjection import key, orders
    from placement_fixtures import place_at
    owner, field = mixed_field()
    assert orders(owner, field) == (0,1)
    before = key(owner, field, order=0)
    place_at(owner, int(field.concept_ids[1]), [1.,0.,0.,0.])
    torch.testing.assert_close(key(owner,field,order=0), before, atol=1e-6, rtol=0)


def test_stimulation_reads_only_the_requested_order_before_definition_closure(monkeypatch):
    from FieldProjection import stimulate
    codes = torch.tensor([[1.,0.],[1.,0.]])
    owner = NS(similarity_codebook=NS(lookup_rows=lambda rows: codes[rows]),
        _row_order=lambda row: row, concept_id_at_row=lambda row: row)
    monkeypatch.setattr('FieldProjection.inventory', lambda owner: [0,1])
    monkeypatch.setattr('FieldProjection.close', lambda owner,rows,evidence: evidence)
    before = stimulate(owner, torch.tensor([[1.,0.]]))
    assert before.evidence[0,0,0,0] == 1
    assert not before.evidence[1].any()
    codes[1] = torch.tensor([0.,1.])
    after = stimulate(owner, torch.tensor([[1.,0.]]))
    torch.testing.assert_close(after.evidence, before.evidence, atol=1e-6, rtol=0)
    own = stimulate(owner, torch.tensor([[0.,1.]]), order=1)
    assert own.evidence[1,0,0,0] == 1 and not own.evidence[0].any()


def test_answer_cost_compares_only_target_concepts_of_the_answer_order():
    from BindingAnswers import cost
    from Meaning import ConceptualMeaning
    from dataclasses import replace
    codes = torch.eye(2)
    owner = NS(_concept_row_of=lambda identity: identity, _row_order=lambda row: row,
        _concept_source_order=lambda ref: ref[1], word_concepts=lambda word: (1,),
        similarity_codebook=NS(lookup_rows=lambda rows,**kw: codes[rows]))
    store = NS(parts=NS(contains=lambda ref: False))
    model = NS(_concept_owner=lambda:owner, symbolSpace=NS(ltm_store=store))
    value = replace(ConceptualMeaning.from_description(codes[0]),
        role_refs=(('sym',0),None,None),bindings={'_bound_roles':(('referent',0),)})
    before = cost(model,value,'target')
    codes[1] = codes[0]
    torch.testing.assert_close(cost(model,value,'target'),before,atol=1e-6,rtol=0)
    assert before == 1


def test_candidate_centroid_and_matching_do_not_mix_support_orders():
    from FamiliarTier import centroid
    from Memory import MediumTermMemory
    codes = torch.tensor([[1.,0.],[1.,0.]])
    owner = NS(_row_order=lambda row:row, similarity_codebook=NS(W=codes,lookup_rows=lambda rows:codes[rows]))
    support = {0:1.,1:1.}
    before = centroid(owner,support,order=0)
    pool = MediumTermMemory(4,n_vectors=8,field_width=2)
    pool._owner = lambda:owner
    other = pool.open_candidate(order=1,when=0,value=codes[1],support={1:1.})
    own = pool.open_candidate(order=1,when=0,value=codes[0],support={0:1.})
    assert pool.match_candidate(1,codes[0],value_order=0,support={0:1.}) == own
    codes[1] = torch.tensor([0.,1.])
    torch.testing.assert_close(centroid(owner,support,order=0),before,atol=1e-6,rtol=0)
    assert pool.match_candidate(1,codes[0],value_order=0,support={0:1.}) == own
    assert pool.match_candidate(1,codes[1],value_order=1,support={1:1.}) == other


def test_shared_activation_prediction_never_pools_another_orders_positions():
    from Layers import SymbolExpectation
    from SymbolField import SymbolField
    head = SymbolExpectation(2,8,4,concept_dim=4)
    keys = torch.eye(4)[:2].clone()
    orders = torch.tensor([0,1])
    field = SymbolField(head,strength=.5,epsilon=1e-6,saturation='clip')
    def read():
        head.set_symbol_bank(keys,orders=orders)
        field.anticipate((keys[None],torch.tensor([2]),orders[None],orders[None],
                          torch.tensor([[0,1]]),torch.ones(1,2)))
        return field.pending[1][0,0].clone()
    before = read()
    keys[1] = torch.tensor([0.,0.,1.,0.])
    torch.testing.assert_close(read(),before,atol=1e-6,rtol=0)


def test_unplaced_candidate_recurrence_uses_its_own_reference_support():
    from Memory import MediumTermMemory
    codes = torch.zeros(3,4)
    owner = NS(_row_order=lambda row:1,similarity_codebook=NS(W=codes,lookup_rows=lambda rows:codes[rows]))
    pool = MediumTermMemory(4,n_vectors=8,field_width=2)
    pool._owner = lambda:owner
    own = pool.open_candidate(order=2,when=0,value=codes[0],support={0:2.,1:1.})
    assert pool.match_candidate(2,codes[0],value_order=1,support={0:4.,1:2.}) == own
    assert pool.match_candidate(2,codes[0],value_order=1,support={1:2.,2:1.}) is None


def test_counted_view_preserves_extension_and_position():
    from ConceptTrust import observe
    from argument_fixtures import argument_inventory
    owner, speech, words = argument_inventory(('modifier_argin1',))
    speech.add(words[0],'modifier_argin1')
    cid = speech.concepts['modifier_argin1']
    row = owner._concept_row_of(cid)
    bank = owner.similarity_codebook
    content = bank.trust.content[row].clone()
    place = bank.mereology.placement[row].clone()
    views = int(bank.trust.views[row])
    observe(owner,cid)
    assert int(bank.trust.views[row]) == views+1
    torch.testing.assert_close(bank.trust.content[row],content,atol=0,rtol=0)
    torch.testing.assert_close(bank.mereology.placement[row],place,atol=0,rtol=0)


def test_unplaced_class_keeps_member_fold_as_extension_not_position():
    from FamiliarTier import cover_members
    from test_attention_promotion import _fixture
    owner, rows = _fixture()
    cid = owner.new_concept(order=1)
    destination = owner._csw_concept_row(1,cid)
    members = {rows[0][1]:2.,rows[1][1]:1.}
    bank = owner.similarity_codebook
    before = bank.W[destination].clone()
    cover_members(owner,cid,members)
    assert bank.trust.content[destination].any()
    torch.testing.assert_close(bank.W[destination], before)


def test_activation_matching_does_not_identify_a_row_of_another_order():
    from Expectation import match_symbols
    keys = torch.eye(2)[None]
    values = keys[:,1:2].clone()
    orders = torch.tensor([0,1])
    _, matched = match_symbols(keys, values, key_orders=orders, value_orders=torch.tensor([[0]]))
    assert not matched.any()
    keys[:,1] = torch.tensor([1.,0.])
    _, after = match_symbols(keys, values, key_orders=orders, value_orders=torch.tensor([[0]]))
    assert torch.equal(after,matched)
    _, own = match_symbols(keys[:,:1],keys[:,:1],key_orders=orders[:1],value_orders=torch.tensor([[0]]))
    assert own.all()


def test_exclusion_occurrences_do_not_join_coordinate_counts_across_orders():
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from part_fixtures import native_store, symbol
    store,_ = native_store()
    owner = store.owner()
    points = torch.eye(4)
    own = symbol(store,points[0])
    other = owner.new_concept(order=1)
    for reference,point in ((own,points[0]),(other,points[1])):
        for _ in range(8):
            store.append_meaning(replace(ConceptualMeaning.from_description(point),
                role_refs=(('sym',reference),None,None)))
    assert store.coordinate_exclusions() == ()
    for _ in range(8):
        store.append_meaning(replace(ConceptualMeaning.from_description(points[1]),
            role_refs=(('sym',own),None,None)))
    assert store.coordinate_exclusions(order=0) == ((0,1),)
    assert store.coordinate_exclusions(order=1) == ()


def test_bind_chunk_menu_keeps_the_requests_order():
    from Memory import STMChunks as chunks
    state = chunks.empty(1,4,2)
    state,_ = chunks.deposit(state,torch.tensor([[1.,0.]]),torch.tensor([True]),identity=1,order=0)
    state,_ = chunks.deposit(state,torch.tensor([[1.,0.]]),torch.tensor([True]),identity=2,order=1)
    state[chunks.STACK].fill_(-1)
    _,ids,_,available = chunks.bind_menu(state,torch.tensor([[[1.,0.]]]),orders=torch.tensor([[0]]))
    assert available[ids == 1].all()
    assert not available[ids == 2].any()


def test_innovation_reads_each_role_only_on_axes_of_its_own_order():
    from test_independent_components import component_fixture
    from test_reference_close import known_concept
    components = component_fixture()
    owner = components.owner
    own = known_concept(owner,torch.eye(4)[0],order=0)
    points = torch.eye(4)[0].expand(3,-1).clone()
    orders = torch.zeros(3,dtype=torch.long)
    before = components.innovation(points,torch.zeros_like(points),orders=orders,prediction_orders=orders)
    for cid in components.individual_ids:
        if owner._concept_source_order(cid) == 1:
            with torch.no_grad():
                owner.similarity_codebook.W[owner._concept_row_of(cid)] = torch.eye(4)[0]
    after = components.innovation(points,torch.zeros_like(points),orders=orders,prediction_orders=orders)
    torch.testing.assert_close(after,before,atol=1e-6,rtol=0)
    assert after.reshape(3,-1)[:,owner._concept_row_of(own)].gt(0).all()
