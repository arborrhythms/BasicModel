"""Stage 2 repair: bounded full-memory reads and actual R/A gradient receipts."""
from types import SimpleNamespace
import torch
import pytest
from Layers import LTM0
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget
from test_one_backward import receipt


def meaning(values, *, scope=(), bindings=()):
    roles = torch.zeros(3, 8)
    roles[0, :len(values)] = torch.tensor(values)
    return ConceptualMeaning(roles, torch.tensor([True, False, False]), scope=scope, bindings=bindings)


def test_small_cued_window_preserves_match_order_filters_and_explicit_references(tmp_path, monkeypatch):
    store = LTM0(8, capacity=32, percept_width=0, stream_capacity=1)
    values = ((1., 0.), (1., 0.), (1., 0.), (1., 0.), (1., 1.), (1., 1.),
              (1., 2.), (0., 1.), (1., 0.), (1., 0.), (1., 3.), (1., 0.))
    for index, value in enumerate(values):
        kind = 'estimate' if index == 1 else 'question' if index in (2, 3) else 'fact'
        store.append_meaning(meaning(value, scope={'region': index != 9}, bindings={'bound': index != 8}),
            kind=kind, evidence=(.4 if index == 3 else 0., 0.), sentence_index=index)
    cue = meaning((1., 0.), scope={'region': True}, bindings={'bound': True})
    calls = []
    original = torch.bmm
    def product(a, b):
        calls.append((list(a.shape), list(b.shape)))
        return original(a, b)
    monkeypatch.setattr(torch, 'bmm', product)
    work = QueryWorkBudget(32)
    result = store.cued_rows(cue, references=(store.occurrence_of(5),), max_records=32, work=work)
    # The old scalar cosine reader, with every candidate fitting its cap,
    # returns these ties by ascending physical address after all filters.
    assert [row['index'] for row in result['value']] == [0, 3, 11, 4, 5, 6, 10, 7]
    assert [row['match'] for row in result['value']] == pytest.approx([1., 1., 1., 2**-.5, 2**-.5, 5**-.5, 10**-.5, 0.])
    assert result['records_scanned'] == 12 and not result['incomplete']
    assert work.spent == 8 and work.counts == {'record': 8} and len(calls) == 1
    receipt(tmp_path, 'small-window-compatibility', dict(indices=[row['index'] for row in result['value']],
        records_scanned=12, returned=8, meter=work.spent, batched_products=calls))


def test_two_thousand_rows_retrieve_the_newest_window_and_charge_only_returns(tmp_path):
    store = LTM0(8, capacity=2000, percept_width=0, stream_capacity=1)
    for index in range(2000):
        store.append_meaning(meaning((1., 0.) if index in (20, 1777) else (0., 1.)), sentence_index=index)
    before = {name: (buffer.data_ptr(), tuple(buffer.shape)) for name, buffer in store.named_buffers()}
    states = {name: value.clone() for name, value in store.state_dict().items()}
    work = QueryWorkBudget(1)
    result = store.cued_rows(meaning((1., 0.)), max_records=3, work=work)
    assert [row['index'] for row in result['value']] == [1777]
    assert result['records_scanned'] == 1024
    assert set(result['incomplete']) == {'ltm_window', 'work_budget'}
    assert work.spent == 1 and work.counts == {'record': 1}
    assert before == {name: (buffer.data_ptr(), tuple(buffer.shape)) for name, buffer in store.named_buffers()}
    assert all(torch.equal(value, store.state_dict()[name]) for name, value in states.items())
    # A returned-record limit, unlike meter exhaustion, is not work exhaustion.
    exact = store.cued_rows(meaning((1., 0.)), max_records=1, work=QueryWorkBudget(1))
    assert exact['incomplete'] == ('ltm_window',)
    # Explicit references still enter before the recency window.
    explicit = store.cued_rows(meaning((1., 0.)), references=(store.occurrence_of(20),), max_records=2, work=QueryWorkBudget(2))
    assert [row['index'] for row in explicit['value']] == [20, 1777]
    receipt(tmp_path, 'full-memory-window-and-allocation', dict(capacity=2000, window=1024,
        matched=1777, excluded=20, explicit_override=[20, 1777], records_scanned=1024,
        returned=1, meter=work.spent, incomplete=list(result['incomplete']),
        buffer_storage_unchanged=True, state_dict_equal=True,
        allocations={name: {'shape': list(shape), 'bytes': store.get_buffer(name).numel()*store.get_buffer(name).element_size()}
                     for name, (_, shape) in before.items() if name.startswith('_cue_')}))


@pytest.mark.parametrize('separate', [True, False])
def test_epoch_report_reads_actual_understanding_gradients_without_changing_the_update(separate, tmp_path):
    from SentenceTraining import begin_sentence, measure, finish_sentence, gradient_receipt
    from Spaces import Codebook
    model = torch.nn.Module()
    model.codes = Codebook()
    model.codes.W = torch.nn.Parameter(torch.tensor([[.5]]))
    model.output = torch.nn.Linear(1, 1, bias=False)
    model.conceptualSpaces = ()
    model.separate_chooser = separate
    model._measure_sentence_gradients = True
    parameter = model.codes.W
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.objective_parameter_groups = lambda optimizer: dict(reconstruction=(parameter,),
        output=tuple(model.output.parameters()), expectation=(), attention=())
    parameter.grad = torch.full_like(parameter, .25)
    for coefficient in (4., -2.):
        begin_sentence(model)
        measure(model, dict(reconstruction=parameter.square().sum(), output=-parameter.sum(),
            chooser=coefficient*parameter.sum()), optimizer)
        finish_sentence(model)
    report = gradient_receipt(model, clear=True)
    sent = report['understanding']
    assert sent['parameters'] == sent['elements'] == 1
    assert sent['reconstruction_norm'] == 2.
    assert sent['output_norm'] == (0. if separate else 2.)
    # Diagnostic hypothetical cross-owner directions remain available.
    assert report['dissonance']['understanding']['reconstruction_output_cosine'] == pytest.approx(-1.)
    assert parameter.item() == .5 and parameter.grad.item() == .25
    assert not model._epoch_objective_gradients
    receipt(tmp_path, 'understanding-gradients-' + str(separate).lower(), sent)


def test_reflexive_part_fact_is_not_an_intuition_definition_cycle(tmp_path):
    from dataclasses import replace
    from ConceptIndex import index_part_row
    from Intuition import extract, read
    from test_thought_infer import setup
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    alloc = owner._concept_allocator
    cid = refs[0][1]
    alloc.reference_orders[cid] = 1
    alloc.individuals.add(cid)
    working = torch.full((owner.outputShape[-1],), .3)
    value = replace(ConceptualMeaning.from_description(working),
                    role_refs=(('sym', cid), None, None))
    before = read(owner, extract(owner, value, work=QueryWorkBudget(32)))
    structural = alloc.records(cid)
    relation = store.append_relation(cid, cid, truth_plus=1., truth_minus=.25)
    assert index_part_row(owner, cid, cid) == cid
    states = {name: tensor.clone() for name, tensor in alloc.parts.state_dict().items()}
    pointers = {name: (tensor.data_ptr(), tuple(tensor.shape))
                for name, tensor in alloc.parts.named_buffers()}
    # The native relation remains a witnessed proposition, while its
    # structural projection agrees with the taxonomy writer's identity case.
    after = read(owner, extract(owner, value, work=QueryWorkBudget(32)))
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert alloc.records(cid) == structural
    assert alloc.parts.find(cid, cid) == relation[1]
    assert float(alloc.parts.truth_plus[relation[1]]) == 1.
    assert float(alloc.parts.truth_minus[relation[1]]) == .25
    assert all(torch.equal(tensor, alloc.parts.state_dict()[name]) for name, tensor in states.items())
    assert pointers == {name: (tensor.data_ptr(), tuple(tensor.shape))
                        for name, tensor in alloc.parts.named_buffers()}
    # Reciprocal testimony between two recorded order-one concepts remains
    # readable in both directions without making either a class member.
    other = refs[1][1]
    alloc.reference_orders[other] = 1
    for member, whole in ((cid, other), (other, cid)):
        store.append_relation(member, whole, truth_plus=1.)
        assert alloc.parts.find(member, whole) is not None
        assert ('sym', member) not in alloc.refs(whole, 'part')
    torch.testing.assert_close(read(owner, extract(owner, value, work=QueryWorkBudget(32))), before, rtol=0, atol=0)
    # Exercise the cycle guard on an actual malformed native definition.
    for member, whole in ((cid, other), (other, cid)):
        alloc.store_of(whole).add_constituent(whole, 'part', ('sym', member))
    with pytest.raises(ValueError, match='cyclic intuition definition'):
        extract(owner, value, work=QueryWorkBudget(32))
    receipt(tmp_path, 'reflexive-part-structure', dict(reflexive_fact_preserved=True,
        evidence=[1., .25], structure_unchanged=True, intuition_value_equal=True,
        relation_state_dict_equal=True, storage_unchanged=True, genuine_cycle_rejected=True))


def test_recalled_chunk_addresses_depend_only_on_their_own_stream(tmp_path):
    from dataclasses import replace
    from types import MappingProxyType
    from Layers import ShortTermMemory
    from Queries import ThoughtResult
    from ThoughtChunks import address, put, stage
    from ThoughtReferences import with_slots
    import STMChunks as chunks
    def recall(batch):
        stm = ShortTermMemory(batch=batch, concept_dim=8)
        value = with_slots(replace(meaning((1., 0.)), role_refs=(('sym', 1), None, None)), (), pair=(.8, .1))
        focus = address(1)
        put(stm, value, focus, 0, pin=True)
        if batch > 1:
            put(stm, value, address(1), 1, pin=True)
            stm.set_state(chunks.read(stm.state, torch.tensor([-1, 0]), torch.tensor([False, True]), tick=1000))
        before = tuple(t[1:].clone() for t in stm.state)
        shapes = tuple(tuple(t.shape) for t in stm.state)
        truth_pointer = stm._live_chunk_truth.data_ptr()
        memory = SimpleNamespace(_thought_stm=stm, thought_state=lambda b: SimpleNamespace(focus_where=focus))
        retrieved = replace(value, role_refs=(('sym', 2), None, None))
        result = ThoughtResult('query', 'conceptual-subgoal', 'set', 'retrieval', value,
            MappingProxyType(dict(meaning=retrieved, frames=({'occurrence': ('ltm', 'native', 7)},))))
        for _ in range(2):
            stage(memory, result, 0)
        recalled = sorted(tuple(where) for item in stm._thought_contents[0] if item is not None
                          for where, _ in (item,) if where[-1] == 3.)
        assert len(recalled) == len(set(recalled)) == 2
        assert shapes == tuple(tuple(t.shape) for t in stm.state)
        assert stm._live_chunk_truth.data_ptr() == truth_pointer
        assert all(torch.equal(old, current[1:]) for old, current in zip(before, stm.state))
        return recalled
    single, batch = recall(1), recall(2)
    assert single == batch
    receipt(tmp_path, 'stream-local-recall-addresses', dict(single=single, batch=batch,
        other_stream_read_tick=1000, other_stream_unchanged=True, shapes_unchanged=True, truth_storage_unchanged=True))


@pytest.mark.parametrize('room', [1, 2])
def test_determined_mention_keeps_an_occurrence_after_its_relation_was_written(room, tmp_path, monkeypatch):
    from ClauseRow import Clause, ClauseCapacityError, Composition
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    left, right = (f.noun(name) for name in ('first', 'second'))
    terms = tuple(Composition(f.registry._payload(('sym', ref)), reference=ref)
                  for ref in (left, right))
    point = torch.maximum(*(term.point for term in terms))
    pair = Composition(point, terms, operation='equal')
    determined = Composition(point, (pair,), operation='definite_unary', head_role=1)
    mention = Clause(ConceptualMeaning.from_description(point), point=point,
        composition=determined, reference_request=dict(mode='bind', determined=True, introduction=True))
    projected = Composition(point, (determined,), operation='preposition_unary', head_role=1)
    earlier = Clause(mention.meaning, point=point, composition=projected)
    root = Clause(mention.meaning, point=point, composition=Composition(point),
                  companions=(earlier, mention))
    # The earlier branch writes the shared term as a relation before its
    # determined introduction is closed. Admission needs two occurrences.
    f.store.count.fill_(f.store.capacity-room)
    before = len(f.store), len(f.store.parts), f.cs._concept_allocator.next_id
    pointers = {name: (value.data_ptr(), tuple(value.shape)) for name, value in f.store.named_buffers()}
    written = {}
    if room == 1:
        with pytest.raises(ClauseCapacityError, match='LTM0 construction capacity exhausted before close'):
            f.store.write_clause(root, written_rows=written)
        assert (len(f.store), len(f.store.parts), f.cs._concept_allocator.next_id) == before
        assert not written
    else:
        row = f.store.write_clause(root, evidence=(.7, .2), written_rows=written)
        retained = written[id(mention)]
        assert isinstance(retained, int) and retained != row
        assert written['occurrence', id(mention)] == f.store.occurrence_of(retained)
        assert written['occurrence', id(mention)][0] == 'ltm'
        assert written[id(earlier)][0] == 'sym'
        assert f.store.parts.find(left, right) == written[id(earlier)][1]
        torch.testing.assert_close(f.store.slots[retained], point)
        assert len(f.store) == before[0]+2
    assert pointers == {name: (value.data_ptr(), tuple(value.shape)) for name, value in f.store.named_buffers()}
    receipt(tmp_path, f'determined-occurrence-{room}', dict(available=room,
        occurrence_retained=room == 2, capacity_rejected_before_any_write=room == 1,
        storage_unchanged=True, relation_preserved=room == 2))


def test_epoch_understanding_includes_compose_only_operators():
    from types import SimpleNamespace
    from SentenceTraining import gradient_receipt, measure

    model = torch.nn.Module()
    model.compose_only = torch.nn.Linear(1, 1, bias=False)
    model.languageSpace = SimpleNamespace(_tree_layer=lambda arity:
        SimpleNamespace(ops=(model.compose_only,), unary_ops=()))
    model.separate_chooser = False
    model._measure_sentence_gradients = True
    parameter = model.compose_only.weight
    optimizer = torch.optim.SGD([parameter], lr=.1)
    model.objective_parameter_groups = lambda optimizer: dict(
        reconstruction=(parameter,), output=(), expectation=(), attention=())
    measure(model, {'output': model.compose_only(torch.ones(1, 1)).sum()}, optimizer)
    report = gradient_receipt(model)
    assert report['understanding']['output_norm'] == 1.
    assert report['dissonance']['compose_binary_ops.0.Linear']['output_norm'] == 1.
    assert parameter.grad is None
