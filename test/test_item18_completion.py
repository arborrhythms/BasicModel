"""Certificates for the reviewed item 1.8 defects, without selected seeds."""
import copy
import pytest
import torch

def test_empty_intuition_does_not_close_an_open_question():
    from test_thought_infer import setup, invoke
    from ThoughtReferences import with_slots, open_slots
    from Meaning import ConceptualMeaning
    from ThoughtStream import integrate
    model, registry, store, refs = setup()
    goal = with_slots(ConceptualMeaning.from_description(registry._payload(refs[0])),
                      (('referent', 0),))
    result = invoke(model, registry, registry.form('intuit', goal))
    concluded = integrate(registry, goal, result, ())
    assert open_slots(concluded) == open_slots(goal)
    assert concluded.role_refs == goal.role_refs
    torch.testing.assert_close(concluded.roles, goal.roles)


def test_perceived_class_keeps_its_own_content_beside_the_reading():
    from types import SimpleNamespace
    from ReferenceClose import ReferenceClose
    from Understanding import ConceptualField
    concept = torch.tensor([.2, .7])
    reading = torch.tensor([.5, 0.])
    occurrence = ('ltm', 'test', 17)
    model = SimpleNamespace(_sentence_open_fields={0: ConceptualField(
        torch.tensor([1]), torch.ones(1, 1, 1, 2), None, None, None)},
        _scalar_present={0: dict(point=reading, occurrence=occurrence, candidate=('candidate', 3))})
    owner = SimpleNamespace(_model=model,
        _concept_allocator=SimpleNamespace(individuals={1}, retired=set()),
        similarity_codebook=SimpleNamespace(W=concept[None]), _csw_row_of=lambda cid: 0,
        _closed_clause_store=lambda: SimpleNamespace(location=lambda ref: 0))
    records = ReferenceClose(SimpleNamespace(owner=owner)).perceived_records(0)
    torch.testing.assert_close(records[1][1]['mention_value'], concept)
    torch.testing.assert_close(records[occurrence][1]['mention_value'], reading)

@pytest.fixture
def meaning_blocks(monkeypatch):
    from util import TheXMLConfig
    config=copy.deepcopy(TheXMLConfig.data)
    config.setdefault('architecture',{}).setdefault('data',{})['dataType']='embedding'
    config.setdefault('PartSpace',{})['nDim']=4
    config.setdefault('ConceptualSpace',{})['nDim']=12
    monkeypatch.setattr(TheXMLConfig,'_data',config)

@pytest.mark.parametrize('lanes',[(.7,0.),(0.,.8),(.7,.8)])
@pytest.mark.parametrize('operation',['not','non'])
def test_negation_agrees_across_meaning_field_and_layer_forms(meaning_blocks,lanes,operation):
    from MeaningCodes import compose
    from FieldLogic import reduce
    from Language import NotLayer, NonLayer
    cls=NotLayer if operation=='not' else NonLayer
    pair=torch.tensor(lanes)
    expected=pair.flip(-1) if operation=='not' else torch.zeros_like(pair)
    torch.testing.assert_close(compose(operation,pair),expected)
    torch.testing.assert_close(reduce(pair,operation=operation),expected)
    torch.testing.assert_close(cls(representation='poles')(pair),expected)
    form=torch.tensor([.1,.2,.3,.4])
    code=torch.cat((form,pair[:1].expand(4),pair[1:].expand(4)))
    result=cls(representation='code')(code)
    torch.testing.assert_close(result[:4],form)
    torch.testing.assert_close(result[4:],torch.cat((expected[:1].expand(4),expected[1:].expand(4))))

def test_surface_projects_and_recovers_right_meaning(meaning_blocks):
    from Language import SurfaceLayer
    marker=torch.linspace(.1,.4,12)
    content=torch.linspace(.5,.8,12)
    op=SurfaceLayer(12,12)
    parent=op.compose(marker,content)
    torch.testing.assert_close(parent[4:],content[4:],rtol=0,atol=0)
    restored_marker,restored_content=op.reconstruct(parent,marker,'left')
    torch.testing.assert_close(restored_marker,marker)
    torch.testing.assert_close(restored_content,content)


def test_free_inverse_cannot_erase_both_named_operands_to_fit_an_empty_meaning(meaning_blocks):
    from Language import ConjunctionLayer, DisjunctionLayer, LanguageSpace
    conjunction, disjunction = ConjunctionLayer(12, 12), DisjunctionLayer(12, 12)
    left = torch.tensor([[1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
    right = torch.tensor([[0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
    parent = conjunction.compose(left, right)
    bank = torch.stack((left, right), 1)
    valid = torch.ones(1, 2, dtype=torch.bool)
    absent = torch.zeros(1, dtype=torch.bool)
    for operation, expected in ((conjunction, True), (disjunction, False)):
        a, b, available = LanguageSpace._bounded_binary_reconstruction(
            operation, parent, torch.zeros_like(parent), absent, absent,
            bank, valid, 2, require_progress=True)
        assert available.tolist() == [expected]
        if expected:
            torch.testing.assert_close(operation.compose(a, b), parent)
            assert a[:, 4:].any() and b[:, 4:].any()

@pytest.mark.parametrize('walk', ('reconstruction', 'output'))
def test_separate_walk_credit_updates_only_the_policy_that_drew_it(walk):
    from types import SimpleNamespace
    from torch import nn
    from Models import BaseModel
    from Layers import Error
    from Generative import WALK_FEATURE_WIDTH
    from WholePaths import credit
    from ObjectiveOwnership import registry_costs, backward_owned
    model, operation = policy_fixture()
    language = model.languageSpace
    active = torch.ones(1, dtype=torch.bool)
    features = torch.ones(1, 1, 2, WALK_FEATURE_WIDTH)
    tops = model.reconstruction_parameter.reshape(1, 1, 2)
    first = (torch.zeros(1, 1, dtype=torch.long), active[:, None], features, tops)
    other = (torch.ones(1, 1, dtype=torch.long), active[:, None], features, tops)
    draw = {name + '_round': torch.tensor([0 if name == walk else -1])
            for name in ('reconstruction', 'output', 'thought')}
    draw.update(walk_count=torch.ones(1), walk_rounds=torch.ones(1))
    exploit = SimpleNamespace(parts=torch.tensor([[3., 0., 1.]]),
        observation={'generation': SimpleNamespace(trace=first)}, output_trace=first)
    path = SimpleNamespace(parts=torch.tensor([[1., 0., 1.]]), draw=draw,
        observation={'generation': SimpleNamespace(trace=other)}, output_trace=other,
        registry=Error(row_mask=active))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    old = {name: p.detach().clone() for name, p in model.named_parameters()}
    credit(model, exploit, path, active)
    owners = model.objective_parameter_groups(optimizer)
    backward_owned(registry_costs(path.registry), owners)
    optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, old[name])}
    assert changed == {'languageSpace.generate_policy.weight', 'languageSpace.generate_policy.bias'}
    assert all(p.grad is None for p in owners['reconstruction'])


def test_stimulation_reads_positive_and_negative_meaning_lanes_independently():
    from types import SimpleNamespace
    from FieldProjection import stimulate
    from test_cs_sparse_weights import _cs, _mint_row
    owner = _cs()
    rows = [_mint_row(owner, 0, 700 + i) for i in range(2)]
    with torch.no_grad():
        owner.similarity_codebook.W[rows] = torch.tensor([
            [1., 1., 1., 1., 1., 0., 0., 0.],
            [1., 1., 1., 1., 0., 1., 0., 0.]])
    owner.similarity_codebook.mereology = SimpleNamespace(percept_event_width=4,
        derive=lambda indices, b=None: owner.similarity_codebook.W[indices])
    point = torch.tensor([[1., 1., 1., 1., .3, 0., .8, 0.]])
    field = stimulate(owner, point)
    torch.testing.assert_close(field.evidence[:, 0, 0], torch.tensor([[.3, .8], [0., 0.]]))


def policy_fixture():
    from types import SimpleNamespace
    from torch import nn
    from Models import BaseModel
    model = BaseModel()
    model.separate_chooser = True
    model.conceptualSpaces = []
    model.outputSpace = nn.Linear(2, 1)
    model.reconstruction_parameter = nn.Parameter(torch.ones(2))
    model.languageSpace = nn.Module()
    language = model.languageSpace
    language.generate_policy = nn.Linear(2, 2)
    operation = nn.Module()
    operation.chooser = nn.Linear(2, 2)
    operation.reduce_pressure = nn.Parameter(torch.ones(()))
    operation.exploration_fraction = 1.
    operation.exploration_temperature = 1.
    model.selection = operation
    model.symbolSpace = SimpleNamespace(languageLayer=SimpleNamespace(operation_layer=operation))
    language._tree_layer = lambda _: operation
    language.generate_policy_logits = lambda top, features=None: language.generate_policy(top)
    return model, operation


@pytest.mark.parametrize('departure', ('compose', 'thought'))
def test_separate_selection_credit_cannot_train_its_input_or_decoder(departure):
    from types import SimpleNamespace
    from Models import BasicModel
    from Layers import Error
    from WholePaths import credit
    from ObjectiveOwnership import registry_costs, backward_owned
    model, operation = policy_fixture()
    active = torch.ones(1, dtype=torch.bool)
    probability = operation.chooser(model.reconstruction_parameter).softmax(0)[0]
    costs = torch.tensor([[4., 2.]])
    registry = Error(row_mask=active)
    if departure == 'compose':
        model._compose_forced_slots = active[:, None]
        model._compose_sampling_scale = torch.ones(1, 1)
        model._sentence_cost_registry = registry
        state = [None] * 25
        state[15], state[24] = torch.zeros(1, 1, dtype=torch.long), probability.log().reshape(1, 1)
        BasicModel._compose_score_function_loss(model, ((), state, ()), costs)
    else:
        draw = dict(reconstruction_round=torch.tensor([-1]), output_round=torch.tensor([-1]),
            thought_round=torch.tensor([0]), walk_count=torch.ones(1), walk_rounds=torch.ones(1))
        first = SimpleNamespace(parts=torch.tensor([[3., 0., 1.]]), observation={}, output_trace=None)
        other = SimpleNamespace(parts=torch.tensor([[1., 0., 1.]]), observation={}, output_trace=None,
            draw=draw, registry=registry, thinking=SimpleNamespace(traces={0:dict(choices=[
                dict(probability=probability, alternatives=1)])}))
        credit(model, first, other, active)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    old = {name: p.detach().clone() for name, p in model.named_parameters()}
    backward_owned(registry_costs(registry), model.objective_parameter_groups(optimizer))
    optimizer.step()
    assert {name for name, p in model.named_parameters() if not torch.equal(p, old[name])} == {
        'selection.chooser.weight', 'selection.chooser.bias'}
    assert model.reconstruction_parameter.grad is None


def test_thought_faces_have_exact_ltm_method_grants():
    from Queries import THOUGHT_EXECUTORS
    expected = dict(bind=('bind_reference',), isPart=('relation_evidence',),
        isEqual=('relation_evidence',), infer=('relation_evidence',),
        intuit=('intuition_rules',), isTrue=('end_evidence','truth_evidence','resolve_description'),
        query=('query','resolve_description'), ask=('best_match','resolve_description'))
    for name, descriptor in THOUGHT_EXECUTORS.items():
        assert dict(descriptor.method_grants).get('ltm', ()) == expected.get(name, ())


def test_native_relation_does_not_alias_a_thought_store_local_relation():
    from types import SimpleNamespace as NS
    from MemberRead import read
    from Memory import SYMBOL
    def shadowed(_):
        raise AssertionError('a local thought relation shadowed the native relation')
    local = NS(location=lambda address: ('sym', 17), point_of_row=shadowed)
    parts = NS(references=lambda relation: ((SYMBOL, 1), (SYMBOL, 2)))
    store = NS(location=lambda address: ('sym', 17), parts=parts)
    codes = torch.eye(2)
    owner = NS(_csw_row_of=lambda cid: {1:0, 2:1}.get(cid),
        similarity_codebook=NS(lookup_rows=lambda rows: codes[rows]),
        stm=NS(_thought_stores=((local,),)), _closed_clause_store=lambda: store)
    torch.testing.assert_close(read(owner, ('sym', 17)), torch.ones(2))


def test_percept_definition_uses_the_same_conjunction_lanes_as_the_field():
    from test_cs_sparse_weights import _cs
    from basic_fixtures import feature
    cs = _cs()
    feature(cs, 0, 'ps', 1, 1.)
    feature(cs, 0, 'ps', 2, 1., negated=True)
    ids = torch.tensor([[1, 99, 2]])
    starts = torch.arange(3)
    spans = torch.stack((starts, starts+1), -1)[None]
    extents = torch.tensor([[[0, 3]]])
    field = cs.cs_read_memberships((ids, spans, None, ids, spans), extents)
    torch.testing.assert_close(field[0, 0, 0], torch.tensor([0., 1.]))




def test_mixed_thought_trials_keep_the_selected_history_frontier():
    from collections import deque
    from types import SimpleNamespace as NS
    from WalkTrials import ThoughtTrialState
    memory = NS(_what_slots=[deque([NS(id=8)]), deque([NS(id=9)])], _thought_next_id=10)
    model = NS(_what_memory=lambda: memory, when_time=torch.tensor(7),
        when_encoding=NS(t=7), present=lambda: 7)
    first = ThoughtTrialState(model)
    memory._what_slots = [deque([NS(id=18)]), deque([NS(id=19)])]
    memory._thought_next_id = 20
    second = ThoughtTrialState(model)
    ThoughtTrialState.keep((first, second), torch.tensor([0, 1]))
    assert [row[-1].id for row in memory._what_slots] == [8, 19]
    assert memory._thought_next_id == 20


def test_mixed_thought_trials_select_active_rows_of_reserved_stm():
    from collections import deque
    from types import SimpleNamespace as NS
    from Memory import ShortTermMemory, STMChunks
    from WalkTrials import ThoughtTrialState
    stm = ShortTermMemory(batch=4, capacity=2, concept_dim=3)
    stm.ensure_batch(2)
    memory = NS(_what_slots=[deque([1]), deque([2]), deque(), deque()], _thought_next_id=3)
    owner = NS(stm=stm)
    model = NS(_what_memory=lambda: memory, conceptualSpace=owner, when_time=torch.tensor(7),
        when_encoding=NS(t=7), present=lambda: 7)
    state = list(stm.state)
    state[STMChunks.WHEN] = torch.ones_like(state[STMChunks.WHEN])
    stm.set_state(tuple(state))
    first = ThoughtTrialState(model)
    state = list(stm.state)
    state[STMChunks.WHEN] = torch.full_like(state[STMChunks.WHEN], 2.)
    stm.set_state(tuple(state))
    memory._what_slots = [deque([11]), deque([12]), deque(), deque()]
    second = ThoughtTrialState(model)
    ThoughtTrialState.keep((first, second), torch.tensor([0, 1]))
    assert stm.state[STMChunks.WHEN][0].eq(1.).all()
    assert stm.state[STMChunks.WHEN][1].eq(2.).all()
    assert list(memory._what_slots[0]) == [1]
    assert list(memory._what_slots[1]) == [12]


def test_a_completed_reference_answer_realizes_its_binding():
    from dataclasses import replace
    from types import SimpleNamespace
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    from Output import thought_answer_meanings
    meaning = ConceptualMeaning(torch.eye(3), torch.ones(3, dtype=torch.bool),
        role_refs=(('sym', 7), ('sym', 8), ('sym', 9)), sentence_kind='relation')
    meaning = with_slots(replace(meaning, bindings={
        '_bound_roles': (('referent', 2),), '_evidence_kind': 'perception'}), (), pair=(.8, .1))
    answer, = thought_answer_meanings(SimpleNamespace(meaning=meaning, result=None))
    assert answer.role_mask.tolist() == [True, False, False]
    assert answer.role_refs == (('sym', 9), None, None)
    torch.testing.assert_close(answer.roles[0], meaning.roles[2])
