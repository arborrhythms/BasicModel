"""The reduced memory's time, trust, entry and activity contracts."""
import copy
import pytest
import torch
from architecture import WHERE_WIRES, WHEN_WIRES

# The percept band (.where and .when); in fixture(percept=True) the first
# wire past it (8 -> 16) is the first of its four content wires.
C = WHERE_WIRES + WHEN_WIRES


def test_appending_a_referent_keeps_the_row_names_aligned():
    from test_reference_close import fixture

    resolver, store, atoms = fixture(percept=True)
    first = store.append_idea(atoms[C])
    reference = store.occurrence_of(first)
    # The exclusion cache reads a new row's order through its referent.
    store.coordinate_exclusions(order=0)
    second = store.append_idea(atoms[C+1], referent=reference)
    third = store.append_idea(atoms[C+2])
    references = [store.occurrence_of(row) for row in (first, second, third)]
    assert len(set(references)) == 3
    for row, address in zip((first, second, third), references):
        assert store.location(address) == row
        torch.testing.assert_close(store.point_of_row(address), store.slots[row])
    assert [store.row_name(row) for row in (first, second, third)] == [1, 2, 3]


def test_pending_question_keeps_its_retained_context_point():
    from dataclasses import replace
    from ReferenceContext import SituationFrame
    from Meaning import ConceptualMeaning
    point = torch.tensor([.3, .6, .1, .4])
    meaning = replace(ConceptualMeaning.from_description(point), sentence_kind=None,
                      bindings={'_open_references': (('referent', 0),)})
    frame = SituationFrame.from_row(dict(meaning=meaning, row_id=1, point=point))
    assert frame.depth == 1 and frame.occupied.tolist() == [True, False, False]
    torch.testing.assert_close(frame.roles[0], point)


def test_recent_relation_head_cannot_alias_a_later_retained_time():
    from MemberRead import read
    from SentenceStrata import SentenceStrata
    from test_reference_close import fixture

    resolver, store, atoms = fixture(percept=True)
    owner = resolver.components.owner
    strata = owner.stm.strata = SentenceStrata(1, 3, 3, store.nDim, percept_width=C)
    first = store.append_idea(atoms[C])
    strata.keep_close(0, store.slots[first], store.entry_time(first),
                     address=store.row_name(first), source=store.parts)
    # A relation closes without another LTM0 row, but its recent head is
    # still an addressed working idea. It must not take a future row's name.
    head = strata.keep_close(0, atoms[C+1], False, source=store.parts)
    rows = [store.append_idea(atoms[i]) for i in (C+2, C+3)]
    assert head not in [store.occurrence_of(row) for row in rows]
    torch.testing.assert_close(read(owner, head), atoms[C+1])
    for row in rows:
        torch.testing.assert_close(read(owner, store.occurrence_of(row)), store.slots[row])


def test_23_9_all_retained_occurrences_supply_exclusion_after_mtm_expires():
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from RowTime import encoder
    from chunk_fixtures import context
    from test_reference_close import fixture

    histories = []
    for mentioned in (False, True):
        resolver, store, atoms = fixture(percept=True)
        for i in range(8):
            for coordinate in (C, C+1):
                point = atoms[coordinate]
                tick = 100 * (2*i + coordinate-C)
                store._model_present = lambda: tick
                request = dict(mode='bind', compose_bind=0, leaves=(0,),
                    cue=point, predicate=point*0, role=0)
                clause = Clause(ConceptualMeaning.from_description(point),
                    point=point, order=1, when=encoder().encode(tick),
                    reference_request=request if mentioned else None,
                    composition=Composition(point, order=0))
                plan = resolver.preview(clause, context(resolver, ()))
                kept = resolver.prepare(clause, plan)
                written = {}
                row = store.write_clause(kept, written_rows=written)
                if mentioned:
                    resolver.commit(plan, row, written, document='sample',
                                    position=i, training=False)
        resolver.components.owner.mtm.restore(None)
        assert len(store) == 16
        assert all(store.referent_of(i) is None for i in range(len(store)))
        assert resolver.exclusion_pairs(order=0) == ((C, C+1),)
        histories.append({name: getattr(store, name)[:len(store)].clone()
                          for name in store.columns})
        # One joint occurrence removes the empirical exclusion. Repeated
        # access alone cannot substitute for another retained occurrence.
        store.append_idea(atoms[C] + atoms[C+1])
        assert resolver.exclusion_pairs(order=0) == ()
    for name in histories[0]:
        torch.testing.assert_close(histories[0][name], histories[1][name], rtol=0, atol=0)


def test_43_writes_at_one_present_share_its_stamp_and_take_distinct_names():
    from Memory import LTM0
    from RowTime import encoder
    store = LTM0(16, capacity=4)
    store._model_present_code = lambda: encoder().encode(9)
    rows = [store.append_idea(torch.eye(16)[i]) for i in range(2)]
    assert [int(encoder().decode_index(store.entry_time(row))) for row in rows] == [9, 9]
    assert [store.row_name(row) for row in rows] == [1, 2]


@pytest.mark.parametrize('mentioned', (False, True))
def test_23_13_promotion_without_actual_events_keeps_the_current_mtm_value(mentioned):
    from ClauseRow import Clause, Composition
    from FamiliarTier import promote_identity
    from Meaning import ConceptualMeaning
    from QueryWork import QueryWorkBudget
    from RowTime import encoder
    from chunk_fixtures import context
    from test_reference_close import fixture

    resolver, store, atoms = fixture(percept=True)
    owner = resolver.components.owner
    for i in range(2):
        point, tick = atoms[i], 10*i
        store._model_present = lambda: tick
        request = dict(mode='bind', compose_bind=0, leaves=(0,),
                       cue=point, predicate=point*0, role=0)
        clause = Clause(ConceptualMeaning.from_description(point), point=point,
            order=1, when=encoder().encode(tick), evidence=(1., 0.),
            reference_request=request if mentioned else None,
            composition=Composition(point, order=0))
        plan = resolver.preview(clause, context(resolver, ()))
        kept, written = resolver.prepare(clause, plan), {}
        row = store.write_clause(kept, written_rows=written, trust=1.)
        if mentioned:
            resolver.commit(plan, row, written, document='sample',
                            position=i, training=False)
    owner.mtm.restore(None)
    store.recall_identities([store.row(i) for i in range(len(store))],
                           work=QueryWorkBudget(128))
    keys = [owner.mtm.token_of(store.occurrence_of(i)) for i in range(len(store))]
    key = owner.mtm.merge(keys[1], keys[0], when=2)
    assert not owner.mtm.entries[key].events
    current = owner.mtm.value_of(key).clone()
    torch.testing.assert_close(current[:2], current.new_tensor([1., 1.]))
    symbol = promote_identity(owner, key)
    torch.testing.assert_close(owner.similarity_codebook.trust.content[owner._concept_row_of(symbol)],
                               current, rtol=0, atol=0)
    from ConceptPlacement import seed
    torch.testing.assert_close(owner.similarity_codebook.W[owner._concept_row_of(symbol)],
                               seed(symbol, current), rtol=0, atol=0)


def test_43_a_new_thought_advances_past_retained_occurrence_times():
    from RowTime import encoder
    from test_item6_2_thinking import world
    model, registry, store, (reference, _, _) = world()
    for _ in range(4):
        store.append_idea(registry._payload(reference))
    saved = store.slots[:len(store)].clone()
    now = model._advance_when_time()
    assert now > max(int(encoder().decode_index(store.entry_time(i))) for i in range(len(store)))
    torch.testing.assert_close(store.slots[:len(store)], saved, rtol=0, atol=0)


def test_46_checkpoint_keeps_the_current_mtm_link_before_its_close():
    from test_modal_timelines import same
    from test_reference_close import fixture
    resolver, store, points = fixture()
    pool = resolver.components.owner.mtm
    row = store.append_idea(points[0])
    occurrence = store.occurrence_of(row)
    key = pool.open(occurrence, kind=None, when=0, value=points[0], pending=False)
    pool.record_link(key, occurrence, 0, kind='reference', order=0)
    before = pool.state()
    assert before['closes'] == ()
    assert pool.accesses[key] == {0: 1}
    pool.restore(before)
    same(pool.state(), before)
    invalid = copy.deepcopy(before)
    invalid['accesses'] = ((key, ((0, 2),)),)
    with pytest.raises(ValueError, match='expiry metadata'):
        pool.restore(invalid)
    same(pool.state(), before)


def test_43_63_time_and_concept_with_the_same_integer_keep_their_native_orders():
    from ConceptIndex import index_part_row
    from Memory import OCCURRENCE, SYMBOL
    from test_reference_close import fixture, known_concept
    resolver, store, points = fixture(percept=True)
    owner = resolver.components.owner
    other = known_concept(owner, points[C], order=2)
    shared = known_concept(owner, points[C+1], order=0)
    store.names_issued.fill_(other - 1)
    row = store.append_idea(points[C+2])
    assert store.row_name(row) == other
    assert owner._concept_source_order((OCCURRENCE, other)) == 0
    assert owner._concept_source_order((SYMBOL, other)) == 2
    parent = index_part_row(owner, (OCCURRENCE, other), (SYMBOL, shared))
    assert owner._concept_allocator.order_of(parent) == 1


@pytest.mark.parametrize('clock', (False, True))
def test_47_a_checkpoint_of_before_the_one_clock_is_refused_without_loading(clock):
    from Memory import LTM0
    source = LTM0(16, capacity=4)
    for i in range(2):
        source.append_idea(torch.eye(16)[i])
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint['truth_schema'] = torch.tensor(18)
    if clock:
        checkpoint['model_clock'] = checkpoint.pop('names_issued')
    before = {key: value.clone() for key, value in checkpoint.items() if torch.is_tensor(value)}
    restored = LTM0(16, capacity=4)
    with pytest.raises(ValueError, match='before the one clock is refused'):
        restored.load_state_dict(checkpoint)
    assert len(restored) == 0
    for key, value in before.items():
        assert torch.equal(checkpoint[key], value)


def test_41_relations_keep_only_pair_trust_and_activity():
    from Memory import LTM1
    store = LTM1(8)
    assert set(store.columns) == {'refs', 'ref_kinds', 'truth_plus', 'truth_minus', 'activity'}
    row = store.add(2, 3, truth_plus=.7, truth_minus=.2)
    assert float(store.activity[row]) == pytest.approx(.5)
    assert store.add(2, 3, truth_plus=.6) == row
    assert float(store.activity[row]) == pytest.approx(.6)
    assert tuple(float(value) for value in store.evidence(row)) == pytest.approx((.7, .2))
    assert float(store.activity[row]) == pytest.approx(.7)
    assert not any(name in store.state_dict() for name in store.caches)


def test_41_44_52_ideas_keep_flat_entries_and_immutable_signed_trust():
    from Memory import LTM0
    store = LTM0(16, capacity=4, episodic_concepts=2)
    assert set(store.columns) == {'slots', 'episodic_concepts', 'episodic_values', 'trust', 'referent', 'activity', 'row_names'}
    assert store.episodic_concepts.shape == (4, 2)
    first = store.append_idea(torch.ones(16), trust=-.6)
    second = store.append_idea(torch.ones(16), trust=-.6)
    assert (first, second, len(store)) == (0, 1, 2)
    torch.testing.assert_close(store.trust[:2], torch.full((2,), -.6))
    assert float(store.activity[first]) == pytest.approx(.5)
    assert store.occurrence_of(first) != store.occurrence_of(second)
    assert not hasattr(store, 'set_trust') and not hasattr(store, 'set_evidence')


def legacy_idea_state(store):
    """The preceding row layout, without inventing times or changing capacity."""
    state = copy.deepcopy(store.state_dict())
    trust = state.pop('trust')
    state['truth_plus'], state['truth_minus'] = trust.clamp_min(0), (-trust).clamp_min(0)
    state['truth_schema'] = torch.tensor(16)
    # The preceding checkpoint stored centered phase bands.
    layout = int(state['percept_layout'])
    if layout >= C:
        state['slots'] = state['slots'].clone()
        bands = state['slots'][..., layout-C:layout].reshape(store.capacity, -1, 4)
        bands.copy_(torch.where(bands.ne(0).any(-1, keepdim=True), 2*bands-1, 0.))
    state['addressing.address_keys'] = torch.arange(store.capacity) + 100
    state.pop('referent'); state.pop('activity')
    for key in ('episodic_concepts', 'episodic_values'):
        state[key] = state[key][:, None]
    return state


@pytest.mark.parametrize('model_load', (False, True))
def test_47_an_old_layout_is_refused_before_any_destination_changes(model_load):
    from Memory import LTM0
    from MemoryCheckpoint import prepare
    source = LTM0(16, capacity=4, episodic_concepts=2)
    source.append_idea(torch.eye(16)[0], trust=.7)
    checkpoint = legacy_idea_state(source)
    checkpoint['truth_minus'][0] = .3
    target = LTM0(16, capacity=4, episodic_concepts=2)
    target.append_idea(torch.eye(16)[1], trust=-.6)
    destination = target
    if model_load:
        destination = torch.nn.Module()
        destination.register_parameter('unrelated', torch.nn.Parameter(torch.ones(1)))
        destination.first = copy.deepcopy(target)
        destination.last = target
        first = legacy_idea_state(source)
        checkpoint = {'unrelated': torch.zeros(1),
            **{'first.'+key: value for key, value in first.items()},
            **{'last.'+key: value for key, value in checkpoint.items()}}
    before = copy.deepcopy(destination.state_dict())
    supplied = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError, match='before the one clock is refused'):
        if model_load:
            prepare(destination, checkpoint)
        destination.load_state_dict(checkpoint)
    assert checkpoint.keys() == supplied.keys()
    for key, value in supplied.items():
        torch.testing.assert_close(checkpoint[key], value, rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(destination.state_dict()[key], value, rtol=0, atol=0)
    assert not hasattr(target, '_legacy_times')
    if model_load:
        assert not hasattr(destination.first, '_legacy_times')


def test_42_activity_promotes_a_row_once_and_keeps_its_native_order():
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from MemoryActivity import accessed
    from MemoryConsolidation import consolidate
    from test_reference_close import fixture, known_concept
    resolver, store, points = fixture(percept=True)
    owner = resolver.components.owner
    concept = known_concept(owner, points[C], order=1)
    value = replace(ConceptualMeaning.from_description(points[C]),
        role_refs=(('sym', concept), None, None))
    row = store.append_meaning(value, trust=.6, surprise=.85)
    assert store.referent_of(row) == ('sym', concept)
    accessed(store, [row])
    promoted = store.referent_of(row)
    assert promoted != ('sym', concept) and promoted[0] == 'sym'
    assert store.order_of_row(row) == 1 and float(store.activity[row]) == 1.
    inventory = int(owner._concept_allocator.next_id)
    accessed(store, [row])
    assert consolidate(owner, store, rows=[row]) == ()
    assert int(owner._concept_allocator.next_id) == inventory
    assert store.referent_of(row) == promoted


def test_42_discarded_trial_reads_leave_activity_unchanged_and_selected_reads_apply_once():
    from Memory import LTM0
    from MemoryActivity import accessed, trial_activity, snapshot, restore, select, read_stream
    store = LTM0(16, capacity=4)
    rows = [store.append_idea(torch.eye(16)[i]) for i in range(2)]
    before = store.activity.clone()
    @read_stream('stream')
    def read(*, stream, row):
        accessed(store, [row])
    @trial_activity()
    def discarded():
        read(stream=0, row=0)
    discarded()
    torch.testing.assert_close(store.activity, before, rtol=0, atol=0)
    @trial_activity(commit=True)
    def kept():
        empty = snapshot()
        read(stream=0, row=0)
        first = snapshot()
        restore(empty)
        read(stream=0, row=1)
        second = snapshot()
        restore(select([first, second], torch.tensor([1])))
    kept()
    assert float(store.activity[0]) == .5
    assert float(store.activity[1]) == pytest.approx(.6)


from test_item18_finish import named_scalar_model


def test_37_41_a_kept_unrecorded_thought_writes_its_generated_extension(named_scalar_model):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    from ThoughtStream import write
    from ExtendedRecall import forest
    from Extension import joined, ConceptSet
    model, objects = named_scalar_model
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    obj = objects['alpha']
    point = owner.similarity_codebook.lookup_rows(owner._concept_row_of(obj)).detach()
    value = replace(ConceptualMeaning.from_description(point), role_refs=(('sym',obj),None,None))
    value = with_slots(value, (), pair=(.6,0.))
    with torch.no_grad():
        roots = forest(owner,value)
    empty = ConceptSet(torch.empty(0,dtype=torch.long),point.new_empty(0,2))
    expected = joined(roots,empty)
    assert bool(expected.values.any()) and not value.extension
    row = write(model, value, row=0)
    present = store.episodic_concepts[row] >= 0
    assert present.any()
    ids, values = store.episodic_concepts[row,present],store.episodic_values[row,present]
    for cid, strength in zip(ids, values):
        lane = int(strength < 0)
        assert abs(float(strength)) == pytest.approx(float(expected.aligned(cid.reshape(1))[0,lane]))


def test_43_kept_sentence_field_retains_its_occurrence_reference(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_definition_rows import admit
    from reading_fixtures import force_absolute_reading
    from Occurrence import operand_time
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1,
        word_buckets='8', input_width=32, concept_rows=32)
    try:
        admit(model, 'alpha')
        force_absolute_reading(model)
        inputs = model.inputSpace.prepPackedInput([['alpha']])
        model.runBatch(train=False, batchSize=1, split='runtime',
                       batch_override=(inputs, torch.zeros(1, 1, 0)))
        store = model.symbolSpace.ltm_store
        field = model._sentence_fields[0][0]
        assert field is not None and field.meaning.sentence_kind == 'idea'
        row = store.location(field.row_id)
        assert isinstance(row, int)
        assert operand_time(field.row_id) == store.row_name(row)
        assert field.evidence == store.row(row)['evidence']
        assert field.trust == store.row(row)['trust']
        torch.testing.assert_close(field.meaning.roles, store.meaning_of(row).roles, rtol=0, atol=0)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_42_activity_drives_both_store_passes_and_permanent_rows_survive():
    from Memory import LTM0
    from MemoryActivity import accessed
    from Forgetting import ForgettingConfig, forget
    store = LTM0(16, capacity=4, relation_capacity=4)
    for i, surprise in enumerate((.2, .4, 1., .8)):
        store.append_idea(torch.eye(16)[i], surprise=surprise)
        if i < 3:
            store.parts.add(10+i, 20+i, surprise=surprise)
        else:
            # Handle zero is reserved; this unchanged four-seat table has
            # three usable relation seats.
            with pytest.raises(OverflowError):
                store.parts.add(10+i, 20+i, surprise=surprise)
    before = store.slots.clone()
    accessed(store, [0]); store.parts.evidence(1)
    assert store.activity[:4].tolist() == pytest.approx([.3, .4, 1., .8])
    assert store.parts.activity[store.parts.indices()].tolist() == pytest.approx([.3, .4, 1.])
    report = forget(store, ForgettingConfig(high_water=.75, low_water=.5, min_interval=0),
                    document_boundary=True)
    assert (report['before'], report['after'], report['target']) == (4, 2, 2)
    torch.testing.assert_close(store.slots[:2], before[2:4])
    assert store.activity[:2].tolist() == pytest.approx([1., .72])
    assert store.parts.indices().tolist() == [2, 3]
    assert store.parts.activity[store.parts.indices()].tolist() == pytest.approx([.36, 1.])


def test_43_time_named_relation_survives_other_occurrences_forgetting():
    from Memory import LTM0, OCCURRENCE, SYMBOL
    from Forgetting import ForgettingConfig, forget
    store = LTM0(16, capacity=4)
    for i in range(4):
        store.append_idea(torch.eye(16)[i], surprise=.1 if i < 2 else .8)
    reference = store.occurrence_of(2)
    relation = store.parts.add((OCCURRENCE, reference[-1]), (SYMBOL, 10), truth_plus=.6)
    assert store.parts.find((OCCURRENCE, reference[-1]), (SYMBOL, 10)) == relation
    before = store.slots[2].clone()
    forget(store, ForgettingConfig(high_water=.75, low_water=.5, min_interval=0), document_boundary=True)
    assert store.location(reference) == 0
    torch.testing.assert_close(store.slots[0], before)
    assert store.parts.references(relation)[0] == (OCCURRENCE, reference[-1])
    assert store.parts.find((OCCURRENCE, reference[-1]), (SYMBOL, 10)) == relation


def test_63_equal_member_extensions_keep_distinct_referents_and_native_orders():
    from dataclasses import replace
    from types import SimpleNamespace
    from test_cs_sparse_weights import _cs
    from test_reference_close import known_concept
    from definition_fixtures import class_over
    from Memory import LTM0
    from ClauseRow import Clause, Composition, attach_clause_index
    from Meaning import ConceptualMeaning
    from Extension import ConceptSet, Context, Node
    from Language import LowerLayer
    from EpisodicMemory import for_meaning
    owner = _cs(nS=512, order=3, dim=C + 4)
    store = LTM0(C + 4, capacity=4, percept_width=C, episodic_concepts=2)
    attach_clause_index(SimpleNamespace(), store, owner)
    member = known_concept(owner, torch.eye(C + 4)[C], order=0)
    kind = known_concept(owner, torch.eye(C + 4)[C+1], order=1)
    owner.add_part(kind, ('sym', member))
    op = LowerLayer()
    entries = []
    for cid in (member, kind):
        point = owner.similarity_codebook.lookup_rows(owner._concept_row_of(cid)).detach()
        field = ConceptSet(torch.tensor([cid]), torch.tensor([[1., 0.]]))
        context = Context(owner=owner, name='universal')
        extension = Node(op.compose_extend(field, context=context))
        meaning = replace(ConceptualMeaning.from_description(point), extension=(extension,))
        clause = Clause(meaning, point=point, composition=Composition(point, reference=cid))
        row = store.write_clause(clause, episode=for_meaning(store, meaning))
        assert store.referent_of(row) == ('sym', cid)
        entries.append((store.episodic_concepts[row].clone(), store.episodic_values[row].clone()))
    assert entries[0][0].tolist() == [member, -1]
    assert entries[0][1].tolist() == [1., 0.]
    for a, b in zip(entries[0], entries[1]):
        torch.testing.assert_close(a, b)
    assert [store.order_of_row(i) for i in range(2)] == [0, 1]


from test_item18_completion import meaning_blocks


def test_64_preposition_projects_and_recovers_its_object(meaning_blocks):
    from Language import PrepositionLayer
    marker = torch.linspace(.1, .4, 12)
    content = torch.linspace(.5, .8, 12)
    operator = PrepositionLayer(12, 12)
    parent = operator.compose(marker, content)
    torch.testing.assert_close(parent[4:], content[4:], rtol=0, atol=0)
    restored_marker, restored_content = operator.generate(parent, marker=marker)
    torch.testing.assert_close(restored_marker, marker)
    torch.testing.assert_close(restored_content, content)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_selected_activity_rounds_once_after_combining_stream_counts(device):
    from Memory import LTM0
    from MemoryActivity import Activity, ReadEffect, apply
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    store = LTM0(16, capacity=128).to(device)
    start = torch.rand(128, generator=torch.Generator().manual_seed(1505))
    start[:4] = torch.tensor([0., .5, 1., .125])
    for step in (.1, 2.**-25, .009999999999999998, 1e-310):
        store.activity_config = Activity(step=step)
        store.activity.copy_(start)
        first = {(store, row): ReadEffect(row % 13) for row in range(128)}
        second = {(store, row): ReadEffect(row % 7, True) for row in range(0, 128, 2)}
        expected = [min(1., value + (row % 13 + (row % 7 if row % 2 == 0 else 0)) * step)
                    for row, value in enumerate(start.tolist())]
        apply({0: first, 1: second})
        assert torch.equal(store.activity.cpu(), torch.tensor(expected))
