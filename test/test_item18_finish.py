"""Certificates for the final item 1.8 decisions, with no chosen seed."""
from dataclasses import replace
import pytest
import torch


@pytest.fixture
def named_scalar_model(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from ScalarPercepts import code, observe
    from test_output_walk import _stop
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    objects = {}
    for form, number in (('alpha', 1.), ('beta', 1.), ('gamma', 2.)):
        word, *_ = owner.interpret_word(model.perceptualSpace.percept_store.spell_out(form.encode()),
            model.wholeSpace.property_rows_for_bytes(form), key=form)
        for view in range(6):
            row = store.append_idea(code(owner, number), trust=1.)
            objects[form] = observe(owner, word, number, store.occurrence_of(row), stream=0)
    model.eval()
    _stop(model)
    try:
        yield model, objects
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def scalar_answer(model, point, reference):
    from Meaning import ConceptualMeaning
    from What import What
    with torch.no_grad():
        understood = model.understand(model.inputSpace.prepInput(['alpha']))
        held = model.resolveAnswer(understood, What.inference(0, prompt='alpha'))
        meaning = replace(ConceptualMeaning.from_description(point),
                          role_refs=(reference, None, None))
        held = replace(held, conceptual_answer=meaning.roles[None], answer_meanings=((meaning,),))
        return model.reverseOutput(understood, held)


def test_equal_codes_are_said_by_their_given_object_references(named_scalar_model):
    model, objects = named_scalar_model
    owner = model._concept_owner()
    values = [owner.similarity_codebook.W[owner._csw_row_of(objects[name])].detach()
              for name in ('alpha', 'beta')]
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    assert objects['alpha'] != objects['beta']
    for name, value in zip(('alpha', 'beta'), values):
        answer = scalar_answer(model, value, ('sym', objects[name]))
        assert answer.texts == (name,)


def test_reference_names_the_terminal_even_without_semantic_support(named_scalar_model):
    model, objects = named_scalar_model
    owner = model._concept_owner()
    point = torch.zeros_like(owner.similarity_codebook.W[owner._csw_row_of(objects['alpha'])])
    answer = scalar_answer(model, point, ('sym', objects['alpha']))
    assert answer.texts == ('alpha',)
    assert not answer.concepts.any()


@pytest.mark.parametrize('value,expected', [(2., 'gamma'), (9., '')])
def test_bare_scalar_terminal_uses_its_box_or_is_silent(named_scalar_model, value, expected):
    from ScalarPercepts import code
    model, _objects = named_scalar_model
    answer = scalar_answer(model, code(model._concept_owner(), value), None)
    assert answer.texts == (expected,)


def memory_brick():
    import weakref
    from types import SimpleNamespace
    from Memory import LTM0, MediumTermMemory
    store = LTM0(12, capacity=8, percept_width=12, episodic_concepts=4, read_capacity=8)
    owner = SimpleNamespace(_model=None, _csw_row_of=lambda cid: cid,
                            _row_order=lambda cid: {10: 0, 20: 1, 30: 2}[cid],
                            components=SimpleNamespace(reference_close=None), mtm=MediumTermMemory())
    store._concept_owner = lambda: owner
    for i in range(4):
        store.append_idea(torch.ones(12), trust=1.)
    return store


def brick_question(ids, values):
    from Understanding import ConceptualField
    return ConceptualField(torch.tensor(ids), torch.tensor(values, dtype=torch.float32)[:, None, None],
                           torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), None)


def test_recall_uses_each_concepts_own_order_without_carrying():
    from EpisodicMemory import query
    store = memory_brick()
    store.episodic_concepts[:2, 0] = 10
    store.episodic_values[:2, 0] = 1.
    store.episodic_concepts[0, 1] = 20
    store.episodic_values[0, 1] = 1.
    # A reference carries its native order, wherever it is in the flat entry.
    store.episodic_concepts[2, 0] = 20
    store.episodic_values[2, 0] = 1.
    for question, expected in ((brick_question([20], [[1., 0.]]), [(2, 1.), (0, 1.)]),
                     (brick_question([10, 20], [[1., 0.], [1., 0.]]), [(0, 1.)])):
        result = query(store, question, max_records=4)
        assert [(x['index'], x['match']) for x in result['value']] == expected


def test_recall_product_keeps_lanes_and_reads_only_selected_records(monkeypatch):
    import ast, inspect, textwrap
    from EpisodicMemory import query
    store = memory_brick()
    store.episodic_concepts[:4, :2] = 10
    store.episodic_values[:4, :2] = torch.tensor([[1., -1.], [1., 0.], [0., -1.], [.5, -.5]])
    reads = []
    original = store.row
    monkeypatch.setattr(store, 'row', lambda index: (reads.append(index), original(index))[1])
    result = query(store, brick_question([10], [[1., 1.]]), max_records=2)
    assert [(x['index'], x['match']) for x in result['value']] == [(0, 1.), (3, .5)]
    assert reads == [0, 3] and result['records_scanned'] == 2
    # Only selected record materialization may iterate; the numeric window
    # computation is a separately callable tensor product without row loops.
    from EpisodicMemory import matches
    tree = ast.parse(textwrap.dedent(inspect.getsource(matches)))
    assert not any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
                   for n in ast.walk(tree))


def test_full_recall_window_materializes_only_the_selected_rows(monkeypatch):
    from EpisodicMemory import query
    from Meaning import ConceptualMeaning
    store = memory_brick()
    # Fill the already constructed window; no capacity is changed.
    for i in range(len(store), store.ltm_window):
        store.append_idea(torch.ones(12), trust=1.)
    store.episodic_concepts[:, 0] = 10
    store.episodic_values[:, 0] = 1.
    reads = []
    original = store.row
    def read(index):
        reads.append(index)
        return original(index)
    monkeypatch.setattr(store, 'row', read)
    frames = query(store, brick_question([10], [[1., 0.]]), max_records=3)['value']
    assert len(store) == store.ltm_window and reads == [7, 6, 5]
    found = store.cued_rows(ConceptualMeaning.from_description(torch.ones(12)),
                           episodic_frames=frames, max_records=3)
    assert {frame['index'] for frame in found['value']} == {5, 6, 7}
    assert found['records_scanned'] == 0 and reads == [7, 6, 5]


def test_empty_question_has_no_order_match():
    from EpisodicMemory import query
    field = brick_question([10], [[1., 0.]])
    field = replace(field, evidence=field.evidence[:, :, :0], where=field.where[:, :0], when=field.when[:, :0])
    assert query(memory_brick(), field, max_records=3)['value'] == ()


def test_flat_episode_selection_keeps_the_strongest_lanes_across_orders():
    from EpisodicMemory import select
    field = brick_question([10, 20], [[.8, .6], [.9, 0.]])
    ids, values = select(field, 4, torch.tensor([True]))
    assert ids.shape == (1, 4)
    assert ids[0].tolist() == [20, 10, 10, -1]
    assert values[0].tolist() == pytest.approx([.9, .8, -.6, 0.])


def test_old_ordered_entry_loads_its_strongest_readings_with_warning():
    store = memory_brick()
    state = store.state_dict()
    state['truth_schema'] = torch.tensor(16)
    state['truth_plus'] = state.pop('trust')
    state['truth_minus'] = torch.zeros_like(state['truth_plus'])
    state['address_keys'] = torch.arange(store.capacity)
    state.pop('referent'); state.pop('activity')
    state['episodic_concepts'] = torch.full((8, 3, 4), -1, dtype=torch.long)
    state['episodic_values'] = torch.zeros(8, 3, 4)
    state['episodic_concepts'][0, 1, 0] = 20
    state['episodic_values'][0, 1, 0] = .7
    with pytest.warns(UserWarning):
        store.load_state_dict(state)
    assert store.episodic_concepts[0, 0] == 20
    assert store.episodic_values[0, 0] == pytest.approx(.7)
    assert (store.episodic_concepts[:, 1:] == -1).all()
    assert not store.episodic_values[:, 1:].any()


def test_relation_write_updates_reference_indexes_without_rebuild(monkeypatch):
    from Memory import LTM1, SYMBOL
    bank = LTM1(16)
    first = bank.add(20, 30, truth_plus=.6)
    assert bank.find(20, 30) == first
    rebuilds = []
    original = bank._index
    def index():
        version = (id(bank.refs), bank.refs._version, bank.ref_kinds._version,
                   len(bank))
        if bank._index_version != version:
            rebuilds.append(version)
        return original()
    monkeypatch.setattr(bank, '_index', index)
    second = bank.add(30, 40, truth_plus=.8)
    assert bank.find(30, 40) == second
    assert bank.parts(40) == (30,)
    assert bank.wholes(30) == (40,)
    assert bank.rows_referencing(SYMBOL, 30) == (first, second)
    # §17.17 removes operator references from relations and their index.
    assert bank.rows_referencing(SYMBOL, 50) == ()
    assert not rebuilds


def test_declared_caches_rebuild_and_are_not_checkpointed():
    from Memory import LTM0, LTM1, RELATION
    import copy
    store = LTM0(12, capacity=8, percept_width=12)
    row = store.append_idea(torch.ones(12), trust=.7)
    bank = LTM1(16)
    a = bank.add(20, 30, truth_plus=.8)
    b = bank.add((RELATION, a), 40, truth_plus=.6)
    before = (store.location(store.occurrence_of(row)), bank.references(b), bank.find(20, 30),
              bank.parts(30), bank.wholes(20), bank.rows_referencing(RELATION, a),
              tuple(float(x) for x in bank.evidence(b)))
    for memory in (store, bank):
        state = copy.deepcopy(memory.state_dict())
        assert not set(memory.caches).intersection(state)
        memory.clear_caches()
        memory.rebuild_caches()
        # A previous checkpoint is admitted, but its derived copy is ignored.
        legacy = copy.deepcopy(state)
        cache = '_cue_roles' if memory is store else 'relation_order'
        legacy[cache] = torch.ones_like(memory.slots) if memory is store else torch.ones_like(memory.relation_order)
        with pytest.warns(UserWarning, match='cache|retired'):
            memory.load_state_dict(legacy)
    after = (store.location(store.occurrence_of(row)), bank.references(b), bank.find(20, 30),
             bank.parts(30), bank.wholes(20), bank.rows_referencing(RELATION, a),
             tuple(float(x) for x in bank.evidence(b)))
    assert after == before


def test_cache_rebuild_preserves_mtm_forward_slots_and_entry_reads():
    from EpisodicMemory import query
    from Meaning import ConceptualMeaning
    from Occurrence import slot_key
    store = memory_brick()
    owner = store._concept_owner()
    owner._concept_source_order = lambda cid, seen=None: owner._row_order(cid)
    meaning = replace(ConceptualMeaning.from_description(torch.ones(12)), bindings={
        '_open_references': (('referent', 0),), '_forward_references': ((0, 10),)})
    pending = store.append_meaning(meaning)
    slot = slot_key(store.row_name(pending), 0)
    store.episodic_concepts[:4, 0] = 10
    store.episodic_values[:4, 0] = 1.
    def reads():
        frames = query(store, brick_question([10], [[1., 0.]]), max_records=2)['value']
        cued = store.cued_rows(ConceptualMeaning.from_description(torch.ones(12)), episodic_frames=frames)
        return (store.slot_of_reference(slot), store.semantic_reference(slot),
            store._forward_addresses((10,)), store.ideas().tolist(),
            tuple((frame['index'], frame['match']) for frame in cued['value']))
    before = reads()
    store.clear_caches()
    store.rebuild_caches()
    assert reads() == before


def test_cued_read_only_reaches_explicit_or_recalled_rows():
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    store = LTM0(12, capacity=8, percept_width=12)
    meaning = ConceptualMeaning.from_description(torch.ones(12))
    row = store.append_meaning(meaning, evidence=(1., 0.))
    assert not store.cued_rows(meaning)['value']
    assert [x['index'] for x in store.cued_rows(meaning, references=(store.occurrence_of(row),))['value']] == [row]
    assert [x['index'] for x in store.cued_rows(meaning, episodic_frames=(dict(store.row(row), index=row),))['value']] == [row]


def test_entry_time_has_one_unambiguous_name():
    from Memory import LTM0
    import RowTime
    store = LTM0(12, capacity=4, percept_width=12)
    now = RowTime.encoder().encode(17)
    store._model_present = lambda: 17
    row = store.append_idea(torch.ones(12))
    torch.testing.assert_close(store.entry_time(row), now)
    torch.testing.assert_close(store.time[row], now)
    torch.testing.assert_close(store.row(row)['time'], now)
    from ReferenceContext import SituationFrame
    frame = SituationFrame.from_row(store.row(row)).detached()
    torch.testing.assert_close(frame.time, now)
    assert not hasattr(store, 'when')
    for retired in ('stamp', 'read_stamp', 'write_stamp'):
        assert not hasattr(RowTime, retired)
    assert store.when_start == 8


def test_relation_evidence_with_both_ends_reads_one_row(monkeypatch):
    from part_fixtures import native_store
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    store, symbols = native_store(points=torch.eye(4))
    edges = [store.parts.add(symbols[i], symbols[(i+1)%4], truth_plus=1.) for i in range(4)]
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: 0., tau_id=.9)
    reads = []
    original = store.part_row
    def read(index, **kwargs):
        reads.append(index)
        return original(index, **kwargs)
    monkeypatch.setattr(store, 'part_row', read)
    found = reader.relation_evidence('part', ('sym', symbols[2]), ('sym', symbols[3]),
                                     max_records=16, work=QueryWorkBudget(16))
    assert found['witnesses'] == (('sym', edges[2]),)
    assert reads == [edges[2]] and found['records_scanned'] == 1


@pytest.mark.parametrize('operation', ['query', 'ask'])
def test_an_empty_query_concludes_its_unanswered_operand_not_the_request(monkeypatch, operation):
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from MemoryIndex import configure_model_index
    from QueryWork import QueryWorkBudget
    from ThoughtReferences import evidence_pair, open_slots
    from test_normal_thought_controller import _catalog_world
    model, registry, memory, part, _whole = _catalog_world()
    store = model.symbolSpace.ltm_store = __import__('Memory').STM0(8, percept_width=0, capacity=32, relation_capacity=8192)
    configure_model_index(model, store)
    cue = replace(ConceptualMeaning.from_description(torch.eye(8)[0]), role_refs=(part, None, None))
    row = store.append_meaning(cue)
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(cue, row=0, work=QueryWorkBudget(256), continuation=None)
        question = registry.form(operation, store.occurrence_of(row), context=context)
    def choose(root, active, actions, **kw):
        if kw.get('evidence') is not None:
            return None
        return next(action for action in actions if action and action.semantic_id == operation)
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(question, work_budget=256)
    assert not memory.retrieved_frames()
    assert result.meaning.role_refs == cue.role_refs
    torch.testing.assert_close(result.meaning.role_mask, cue.role_mask)
    assert evidence_pair(result.meaning) == (0., 0.)
    assert open_slots(result.meaning) == open_slots(cue)


@pytest.mark.parametrize('operation', ['query', 'ask'])
def test_an_empty_read_inside_a_conclusion_writes_its_operand_recursively(operation):
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from MemoryIndex import configure_model_index
    from QueryWork import QueryWorkBudget
    from ThoughtReferences import evidence_pair, with_slots
    from ThoughtStream import integrate, write
    from test_normal_thought_controller import _catalog_world
    model, registry, _memory, part, _whole = _catalog_world()
    store = model.symbolSpace.ltm_store = __import__('Memory').STM0(8, percept_width=0, capacity=32, relation_capacity=8192)
    configure_model_index(model, store)
    cue = replace(ConceptualMeaning.from_description(torch.eye(8)[0]), role_refs=(part, None, None))
    row = store.append_meaning(cue)
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(cue, row=0, work=QueryWorkBudget(256), continuation=None)
        request = registry.form(operation, store.occurrence_of(row), context=context)
        result = registry.execute(request, context)
        child = integrate(registry, request, result, ())
        parent = with_slots(replace(ConceptualMeaning.from_description(torch.eye(8)[1]),
            role_refs=(('constituent', 0), None, None), constituents=(child,)), (), pair=(0., 0.))
        index = write(model, parent, row=0)
    retained = store.meaning_of(store.location(store.meaning_of(index).role_refs[0]))
    assert retained.role_refs == cue.role_refs
    torch.testing.assert_close(retained.role_mask, cue.role_mask)
    torch.testing.assert_close(retained.roles, cue.roles)
    assert evidence_pair(retained) == (0., 0.)


@pytest.mark.parametrize('trial', [False, True])
def test_prepared_reference_keeps_its_order_when_thought_reads_perception(monkeypatch, trial):
    from test_reference_arrivals import world, described
    from test_reference_close import known_concept
    from chunk_fixtures import context
    from ThoughtReferences import open_slots
    resolver, store, atoms, words, _model = world(pending_atom=0)
    owner = resolver.components.owner
    clause = described(atoms[0], words['x'])
    plan = resolver.preview(clause, context(resolver, ()), stream=0)
    prepared = resolver.prepare(clause, plan)
    meaning = prepared.meaning
    if trial:
        from types import SimpleNamespace
        from Meaning import ConceptualMeaning
        from ThoughtClosing import closing_question
        point = ConceptualMeaning.from_description(clause.point)
        meaning = closing_question(point, prepared, point,
            SimpleNamespace(lexical_forms=('x',)), store=store)
    assert ('referent', 0) in open_slots(meaning)
    order = plan['items'][0]['member_order']
    same = known_concept(owner, atoms[0], order=order)
    other = known_concept(owner, atoms[0], order=order+1)
    monkeypatch.setattr(resolver, 'perceived_records', lambda stream: {
        cid: (('sym', cid), dict(identity=cid, role=0, position=(0, i),
                mention_value=atoms[0], perception=True)) for i, cid in enumerate((same, other))})
    result = resolver.bind_site(meaning, 0, (), stream=0)
    assert same in result['binding_candidates']
    assert other not in result['binding_candidates']


def test_scalar_views_do_not_rewrite_a_words_existing_member_collection(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from ScalarPercepts import observe, code
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    word, *_ = owner.interpret_word(model.perceptualSpace.percept_store.spell_out(b'alpha'),
        model.wholeSpace.property_rows_for_bytes('alpha'), key='alpha')
    text_row = store.append_idea(torch.ones(store.nDim))
    origin = store.occurrence_of(text_row)
    token = owner.mtm.open(origin, kind=None, when=0, value=torch.ones(store.nDim), reading=word)
    owner.mtm.record_link(token, origin, 0, kind='reference', order=1)
    before = owner.mtm.entries[token]
    try:
        for i in range(6):
            row = store.append_idea(code(owner, 2.), trust=1.)
            identity = observe(owner, word, 2., store.occurrence_of(row), stream=0)
        assert owner.mtm.entries[token].refinement['member_order'] == 1
        assert owner.mtm.entries[token].occurrences == before.occurrences
        torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(identity)],
                                   code(owner, 2.), rtol=0, atol=0)
        assert owner.interpret.lexical_form(identity) == 'alpha'
    finally:
        model.End()
