"""Pressure-driven LTM turnover, stable references and gradual detail loss."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Forgetting import ForgettingConfig, forget, model_boundary
from Memory import LTM0, STM0
from Meaning import ConceptualMeaning


def config(**kwargs):
    return replace(ForgettingConfig(high_water=.8, low_water=.5, min_interval=0), **kwargs)


def append(store, index, *, trust=.5, surprise=-1., evidence=None):
    point = torch.zeros(store.nDim); point[-4] = 1.
    return store.append_idea(point, trust=trust, surprise=surprise, evidence=evidence)


def full_store():
    store = LTM0(12, capacity=10, percept_width=8)
    for i in range(8):
        append(store, i)
    return store




def test_trigger_interval_and_age_select_oldest_without_moving_addresses():
    store = full_store()
    ids = torch.tensor([store.row_name(row) for row in range(len(store))])
    assert forget(store, config()) is None
    assert forget(store, config(), document_boundary=True, open_sentence=True) is None
    assert forget(store, config(), document_boundary=True, stateless=True) is None
    report = forget(store, config(), document_boundary=True)
    assert report['before'] == 8 and report['after'] == 5
    torch.testing.assert_close(torch.tensor([store.row_name(row) for row in range(len(store))]), ids[3:])
    for i in range(8, 11):
        append(store, i)
    assert forget(store, config(min_interval=10), document_boundary=True) is None
    store.forget_sentences.add_(10)
    assert forget(store, config(min_interval=10), document_boundary=True)['after'] == 5






def test_ideas_forget_lower_surprise_at_equal_trust_and_age():
    store = STM0(4, capacity=2, relation_capacity=8192)
    meaning = ConceptualMeaning.from_description(torch.tensor([1., 0., 0., 0.]))
    for surprise in (.9, .05):
        store.append_meaning(meaning, trust=.5, surprise=surprise)
    assert store.slots.shape == (2, 4)
    torch.testing.assert_close(store.trust, torch.full((2,), .5))
    torch.testing.assert_close(store.activity, torch.tensor([.9, .05]))
    high, low = [store.row_name(i) for i in range(2)]
    report = forget(store, config(), document_boundary=True)
    assert report['before'] == 2 and report['after'] == 1
    assert store.index_of_row(low) is None and store.index_of_row(high) == 0




def test_protected_and_live_roots_survive_and_block_unsafe_target():
    store = full_store()
    store.activity[:3] = 1.
    roots = tuple(store.occurrence_of(i) for i in (3, 4, 5))
    before = copy.deepcopy(store.state_dict())
    with pytest.warns(RuntimeWarning, match='low water'):
        report = forget(store, config(), document_boundary=True, retained_occurrences=roots)
    assert not report['target_reached'] and len(store) == 6
    for key in ('slots', 'trust', 'referent'):
        torch.testing.assert_close(before[key][:6], store.state_dict()[key][:6])
    assert all(store.location(reference) is not None for reference in roots)


def linked_store():
    store = STM0(4, capacity=4, percept_width=0, relation_capacity=8192)
    child = append(store, 0, trust=0.)
    ref = store.occurrence_of(child)
    parent = store.append_meaning(replace(ConceptualMeaning.from_description(torch.ones(4)),
        role_refs=(ref, None, None)), trust=.9)
    append(store, 2, trust=.7)
    append(store, 3, trust=.8)
    return store, ref, store.occurrence_of(parent)


def test_idea_deletion_releases_its_identity_next_pass():
    store, child, parent = linked_store()
    cfg = config()
    # The identity is protected by the dependency snapshot. Its parent has
    # its own value; deleting that parent releases the identity next pass.
    store.activity[1] = 0.
    with pytest.warns(RuntimeWarning, match='low water'):
        report = forget(store, cfg, document_boundary=True,
                        retained_occurrences=tuple(store.occurrence_of(i) for i in (2, 3)))
    assert store.index_of_row(child[2]) is not None
    assert store.index_of_row(parent[2]) is None
    assert bool(store.forget_pending)
    report = forget(store, cfg, document_boundary=True)
    assert store.index_of_row(child[2]) is None
    assert report['target_reached']


def test_live_parent_pins_its_child():
    store, child, parent = linked_store()
    forget(store, config(), document_boundary=True, retained_occurrences=(parent,))
    assert store.index_of_row(child[2]) is not None
    assert store.meaning_of(store.index_of_row(parent[2])).role_refs[0] == child


def test_wording_is_one_stage_and_checkpoint_preserves_the_result():
    store = full_store()
    old = store.row_name(0)
    assert forget(store, config(), document_boundary=True)['after'] == 5
    restored = LTM0(12, capacity=10, percept_width=8)
    restored.load_state_dict(store.state_dict())
    assert int(restored.forget_passes) == 1
    append(restored, 20)
    assert restored.index_of_row(old) is None




def test_transitive_row_has_its_own_utility():
    from Memory import LTM1
    from Forgetting import forget_parts
    store = LTM1(capacity=5)
    rows = [store.add(left, right, truth_plus=1, surprise=utility)
            for left, right, utility in ((10, 20, .1), (20, 30, .2), (10, 30, .9))]
    low = store.add(40, 50, truth_plus=.5, surprise=0.)
    cfg = config()
    forget_parts(store, cfg)
    assert store.contains(rows[2]) and store.contains(rows[1])
    assert not store.contains(rows[0]) and not store.contains(low)


def test_epoch_trigger_and_disabled_setting():
    store = full_store()
    assert forget(store, config(enabled=False), document_boundary=True) is None
    cfg = config(high_water=1., at_epoch=True)
    assert forget(store, cfg, document_boundary=True) is None
    assert forget(store, cfg, epoch_boundary=True)['after'] == 8


def test_partial_checkpoint_cannot_infer_missing_trust_or_forgetting_state():
    store = full_store()
    state = copy.deepcopy(store.state_dict())
    del state['trust']
    for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence'):
        del state[name]
    restored = LTM0(12, capacity=10, percept_width=8)
    with pytest.raises(ValueError, match='checkpoint column'):
        restored.load_state_dict(state)
    assert len(restored) == 0


def test_boundary_rebuilds_truth_and_preserves_predictor_tensors(monkeypatch):
    store = full_store()
    tensor = torch.randn(2, 3, 4)
    before = tensor.clone()
    roots = [store.occurrence_of(7)]
    calls = []
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store,
        what_memory=None, expectation=SimpleNamespace(_inter_context_occurrences=[roots], context=tensor),
        truth_layer=SimpleNamespace(sync_from_ltm=lambda: calls.append(True))), _open_sentence_slot=None)
    monkeypatch.setattr(ForgettingConfig, 'from_xml', classmethod(lambda cls: config()))
    assert model_boundary(model, document_boundary=True)['after'] == 5
    assert calls == [True] and store.index_of_row(roots[0][2]) is not None
    torch.testing.assert_close(tensor, before)


def test_taxonomy_and_definitions_share_part_retention_not_ltm1_row_ownership():
    from test_word_interpretation import _operator
    from ClauseRow import Clause, attach_clause_index
    cs, interpret = _operator()
    left = interpret.forward(interpret.lookup_word([7], [1], form='cat'))
    right = interpret.forward(interpret.lookup_word([8], [1], form='animal'))
    store = LTM0(8, capacity=4)
    attach_clause_index(SimpleNamespace(), store, cs)
    with torch.no_grad():
        cs.similarity_codebook.W[cs._csw_row_of(left)].fill_(.25)
        cs.similarity_codebook.W[cs._csw_row_of(right)].fill_(.5)
    store._predicate_kind = lambda ref: 'part'
    clause = Clause(ConceptualMeaning(torch.ones(3, 8), torch.ones(3, dtype=torch.bool)),
                    relation='part', refs=(left, left, right))
    row = store.write_clause(clause, trust=0.)
    assert store.write_clause(clause, trust=0.) == row  # dedup keeps its support
    assert store.parts is cs._concept_allocator.parts
    edge = store.parts.find(left, right)
    assert row == ('sym', edge)
    for i in range(1, 4):
        store.append_idea(torch.ones(8), trust=1.)
    forget(store, config(),
           document_boundary=True)
    assert store.parts.contains(edge)
    store.parts.retain([i for i in store.parts.indices().tolist() if i != edge])
    assert store.parts.find(left, right) is None


@pytest.mark.parametrize('training,stateless,deleted', [(True, True, True), (False, False, True), (False, True, False)])
def test_automatic_hook_runs_after_backward_and_honors_serving_mode(monkeypatch, training, stateless, deleted):
    from Models import BasicModel
    store = full_store()
    leaf = torch.tensor(2., requires_grad=True)
    loss = (store.slots[:8, 8:].clone() * leaf).sum()
    calls = []
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store, what_memory=None,
        expectation=None, truth_layer=SimpleNamespace(sync_from_ltm=lambda: calls.append(len(store)), compact=True)),
        _forget_document_boundary=True, _forget_all_streams_closed=True, _forget_training=training,
        _open_sentence_slot=None, stateless=stateless, conceptualSpaces=[],
        _detach_persistent_state=lambda: None)
    monkeypatch.setattr(ForgettingConfig, 'from_xml', classmethod(lambda cls: config()))
    loss.backward()
    BasicModel.post_tick_compact(model)
    assert len(store) == (5 if deleted else 8)
    assert leaf.grad == 8 and '_forget_document_boundary' not in model.__dict__


def test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads(
        tmp_path, monkeypatch, eager_reading, capsys):
    from Models import BaseModel, BasicModel
    from What import What
    from test_compiled_word_chunk import _tiny_canonical_model
    restore = BaseModel.load_weights
    # Eight closings can write 24 rows; their cover needs row 25. Alec chose
    # 32 at construction (seven spare); capacity never changes at runtime.
    # Keep the original absolute
    # forgetting thresholds: start at eight, retain six.
    settings = dict(forgetHighWater=8/32, forgetLowWater=6/32,
                    forgetMinInterval=256, forgetting=True)
    # Isolate document turnover from conceived absences. Their separate
    # certificates exercise the native omission writer and capacity guards.
    turnover_training = dict(expectationOmissionThreshold=1.)
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1,
        word_buckets='16', ltm_capacity=32, architecture_overrides=settings,
        training_overrides=turnover_training)
    assert type(model) is BasicModel
    model._tensor_peer_while_eager = True
    model.stateless = False
    # Single-word documents exercise memory turnover independently of
    # random initial grammar choices. This is a mechanism certificate.
    # These are known words in this memory-turnover fixture. Pending name
    # acquisition has its own certificates and also writes mention records.
    from test_definition_rows import admit
    for word in ('cat', 'dog', 'bird', 'horse'):
        admit(model, word)
    data = model.inputSpace.data
    monkeypatch.setattr(data, 'has_supervised_outputs', False)
    store = model.symbolSpace.ltm_store
    assert store.capacity == 32
    counters = ('forget_passes', 'forget_sentences', 'forget_last_sentence')
    def document(name, sentences):
        data.source_addresses = {'validation': [dict(document=name, document_key=name,
            sentence=i, split='validation') for i in range(len(sentences))]}
        inputs = model.inputSpace.prepPackedInput([sentences])
        model._thought_document_ends = (True,)
        with torch.no_grad():
            model.runBatch(train=False, batchSize=1, split='validation',
                batch_override=(inputs, torch.empty(1, 0)), source_rows=[list(range(len(sentences)))],
                questions=(What.present(0, split='validation'),))
        model._thought_document_ends = ()
        model.flush_word_buffers()
        model.dispatch_packed_soft_reset([True])
        model.dispatch_per_row_reset([True])
        model.dispatch_soft_reset()
        model.post_tick_compact()
    try:
        document('first', ['cat', 'dog', 'bird', 'horse'] * 2)
        report = store.last_forgetting_report
        assert int(store.forget_passes) == 1
        assert report['before'] >= 8 and report['after'] == len(store)
        assert 6 <= len(store) < report['before']
        assert report['target_reached'] == (len(store) == 6)
        assert bool(store.forget_pending) == (len(store) > 6)
        messages = capsys.readouterr().out
        assert f'[LTM forgetting] {report}' in messages
        print(f'[LTM forgetting] {report}')
        snapshot = {name: getattr(store, name).clone() for name in counters}
        occupancy = len(store)
        addresses = torch.tensor([store.row_name(row) for row in range(occupancy)])
        checkpoint = tmp_path/'after-forgetting.ckpt'
        model.save_weights(str(checkpoint))
        document('second', ['dog', 'cat'])
        assert int(store.forget_sentences) == int(snapshot['forget_sentences']) + 2
        assert int(store.forget_passes) == 1  # The configured interval still applies.
        assert len(store) > occupancy
        assert any(store.row_name(row) not in addresses.tolist() for row in range(len(store)))
        assert model._last_sentence_field is not None
    finally:
        model.End()
    fresh = tmp_path/'fresh'
    fresh.mkdir()
    restored = _tiny_canonical_model(fresh, monkeypatch, batch_size=1,
        word_buckets='16', ltm_capacity=32, architecture_overrides=settings,
        training_overrides=turnover_training)
    try:
        assert restore(restored, str(checkpoint), require_match=True)
        loaded = restored.symbolSpace.ltm_store
        assert len(loaded) == occupancy
        torch.testing.assert_close(torch.tensor([loaded.row_name(row) for row in range(occupancy)]), addresses, rtol=0, atol=0)
        for name, value in snapshot.items():
            torch.testing.assert_close(getattr(loaded, name), value, rtol=0, atol=0)
        assert all(loaded.meaning_of(i) is not None for i in range(occupancy))
    finally:
        restored.End()






@pytest.mark.parametrize('kwargs', [{'low_water': .9}])
def test_invalid_configuration_fails_before_a_pass(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_packed_and_single_presentations_keep_the_same_stream_history():
    snapshots = []
    for packing in (1, 4):
        store = LTM0(12, capacity=10, percept_width=8)
        for first in range(0, 8, packing):
            for sentence in range(first, min(8, first + packing)):
                append(store, sentence, surprise=sentence / 8)
                store.forget_sentences.add_(1)
            assert forget(store, config()) is None
        forget(store, config(), document_boundary=True)
        snapshots.append(store.state_dict())
    for name in snapshots[0]:
        torch.testing.assert_close(snapshots[0][name], snapshots[1][name])


def test_checkpoint_cannot_resize_the_constructed_inventory():
    store = full_store()
    restored = LTM0(12, capacity=20, percept_width=8)
    with pytest.raises(ValueError, match='capacity/width'):
        restored.load_state_dict(store.state_dict())
    assert len(restored) == 0 and restored.capacity == 20


def test_conflicting_proof_does_not_change_another_rows_utility():
    from Forgetting import forget_parts
    from Memory import LTM1
    store = LTM1(capacity=5)
    proof = store.add(10, 20, truth_plus=1)
    store.add(20, 30, truth_plus=1)
    conclusion = store.add(10, 30, truth_plus=1)
    low = store.add(40, 50, surprise=0.)
    store.add(10, 20, truth_minus=1)
    cfg = config(low_water=.6)
    forget_parts(store, cfg)
    assert store.contains(proof) and store.contains(conclusion)
    assert not store.contains(low)


def test_stateful_infer_runs_the_host_boundary_after_the_request(monkeypatch):
    from Models import BasicModel
    events = []
    model = SimpleNamespace(stateless=False, _infer_ir=lambda *args, **kwargs: events.append('read') or ['result'])
    def compact():
        assert model._forget_document_boundary and model._forget_all_streams_closed
        events.append('compact')
    model.post_tick_compact = compact
    assert BasicModel.infer(model, 'text') == ['result']
    assert events == ['read', 'compact']


def test_compaction_preserves_protected_row_columns_and_optional_metadata():
    store = full_store()
    store.activity[7] = 1.
    key = store.row_name(7)
    before = {name: getattr(store, name)[7].clone() for name in store.columns}
    forget(store, config(), document_boundary=True)
    row = store.index_of_row(key)
    assert len(store) < 8 and row is not None
    for name, value in before.items():
        torch.testing.assert_close(getattr(store, name)[row], value, msg=name)
    restored = LTM0(12, capacity=10, percept_width=8)
    restored.load_state_dict(store.state_dict())
    assert all(restored.meaning_of(i) is not None for i in range(len(restored)))
