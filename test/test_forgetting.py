"""Pressure-driven LTM turnover, stable references and gradual detail loss."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Forgetting import ForgettingConfig, forget, history_age, retention_values, model_boundary
from Layers import LTM0
from Meaning import ConceptualMeaning


def config(**kwargs):
    return replace(ForgettingConfig(high_water=.8, low_water=.5, min_interval=0), **kwargs)


def append(store, index, *, trust=.5, surprise=1., stream=0, origin=0, evidence=None):
    row = store.append_idea(torch.tensor([1., 0., 0., 0.]), trust=trust,
                           sentence_index=index, stream=stream, surprise=surprise, evidence=evidence)
    store.set_origin(row, origin)
    return row


def full_store():
    store = LTM0(4, capacity=10)
    for i in range(8):
        append(store, i)
    return store


def test_age_is_per_stream_and_rewitnessing_does_not_refresh():
    store = LTM0(4, capacity=10)
    append(store, 0, stream=0)
    append(store, 1, stream=1)
    append(store, 2, stream=0)
    append(store, 3, stream=0)
    append(store, 4, stream=1)
    torch.testing.assert_close(history_age(store), torch.tensor([2., 1., 1., 0., 0.]))
    append(store, 0, stream=0)
    assert len(store) == 5
    assert history_age(store)[0] == 2
    store.witness_stream(0, 2)
    assert history_age(store)[0] == 0  # strongest position across its witnesses
    append(store, 5, stream=-1)
    assert history_age(store)[5] == 0  # legacy/unknown age is not a global clock


def test_trigger_interval_and_age_select_oldest_without_moving_addresses():
    store = full_store()
    ids = store.address_keys[:len(store)].clone()
    assert forget(store, config()) is None
    assert forget(store, config(), document_boundary=True, open_sentence=True) is None
    assert forget(store, config(), document_boundary=True, stateless=True) is None
    report = forget(store, config(), document_boundary=True)
    assert report['before'] == 8 and report['after'] == 5
    torch.testing.assert_close(store.address_keys[:len(store)], ids[3:])
    assert store.stream_rows(0).tolist() == list(range(5))
    for i in range(8, 11):
        append(store, i)
    assert forget(store, config(min_interval=10), document_boundary=True) is None
    store.forget_sentences.add_(10)
    assert forget(store, config(min_interval=10), document_boundary=True)['after'] == 5


@pytest.mark.parametrize('term', ['trust', 'utility'])
def test_value_terms_outweigh_age(term):
    store = full_store()
    if term == 'trust':
        store.source_trust[:8] = torch.tensor([1., .9, .8, .7, .6, .1, .2, .3])
        cfg = config(trust_weight=1, utility_weight=0, luminosity_weight=0, age_weight=0)
    else:
        store.surprise[:8] = torch.tensor([-1., .9, .8, .7, .6, .1, .2, .3])
        cfg = config(trust_weight=0, utility_weight=1, luminosity_weight=0, age_weight=0)
    ids = store.address_keys[:5].clone()
    forget(store, cfg, document_boundary=True)
    torch.testing.assert_close(store.address_keys[:len(store)], ids)


def test_source_trust_is_separate_from_both_evidence():
    store = LTM0(4, capacity=4)
    row = append(store, 0, trust=.9, evidence=(1., 1.))
    assert store.trust[row] == 0
    assert retention_values(store, config())['trust'][row] == pytest.approx(.9)
    store.set_evidence(row, .2, .7)
    assert store.source_trust[row] == pytest.approx(.9)
    store.set_trust(row, -.4)
    assert store.source_trust[row] == pytest.approx(-.4)


def test_three_role_sentences_forget_lower_surprise_at_equal_trust_and_age():
    store = LTM0(4, capacity=2)
    meaning = ConceptualMeaning(torch.eye(4)[:3], torch.ones(3, dtype=torch.bool))
    for stream, surprise in enumerate((.9, .05)):
        store.append_meaning(meaning, trust=.5, surprise=surprise, stream=stream,
                             document_key=f'document-{stream}')
    assert store.role_mask[:2].all()
    terms = retention_values(store, config())
    torch.testing.assert_close(terms['age'], torch.zeros(2))
    torch.testing.assert_close(terms['trust'], torch.full((2,), .5))
    torch.testing.assert_close(terms['utility'], torch.tensor([.9, .05]))
    high, low = store.address_keys[:2].tolist()
    report = forget(store, config(), document_boundary=True)
    assert report['before'] == 2 and report['after'] == 1
    assert store.index_of_row(low) is None and store.index_of_row(high) == 0


def test_dissonance_drops_weaker_opposition_not_intrinsic_both():
    store = LTM0(4, capacity=4)
    append(store, 0, trust=.9, evidence=(.9, 0))
    append(store, 1, trust=.2, evidence=(0, .8))
    append(store, 2, trust=.1, evidence=(1, 1))
    lum = retention_values(store, config())['luminosity']
    torch.testing.assert_close(lum, torch.tensor([1., 0., 1.]))


def test_protected_and_live_roots_survive_and_block_unsafe_target():
    store = full_store()
    for i in range(3):
        store.set_origin(i, store.ORIGIN_USER)
    roots = tuple(store.occurrence_of(i) for i in (3, 4, 5))
    before = copy.deepcopy(store.state_dict())
    with pytest.warns(RuntimeWarning, match='low water'):
        report = forget(store, config(), document_boundary=True, retained_occurrences=roots)
    assert not report['target_reached'] and len(store) == 8
    for key in ('slots', 'refs', 'address_keys', 'c_plus', 'c_minus', 'source_trust'):
        torch.testing.assert_close(before[key], store.state_dict()[key])


def linked_store(*, child_is_relation=False):
    store = LTM0(4, capacity=4)
    child = (store.append_meaning(ConceptualMeaning(torch.eye(4)[:3],
        torch.ones(3, dtype=torch.bool)), trust=0., sentence_index=0)
        if child_is_relation else append(store, 0, trust=0.))
    ref = store.occurrence_of(child)
    roles = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]])
    parent = store.append_meaning(ConceptualMeaning(roles, torch.ones(3, dtype=torch.bool),
        role_refs=(ref, ('sym', 27), None)), trust=.9, stream=0, sentence_index=1)
    store.refs[parent, 0] = int(store.address_keys[child])
    append(store, 2, trust=.7)
    append(store, 3, trust=.8)
    return store, ref, store.occurrence_of(parent)


def test_idea_coarsens_parent_and_relation_cascades():
    for relation in (False, True):
        store, child, parent = linked_store(child_is_relation=relation)
        root = store.meaning_of(1)
        cfg = config(trust_weight=1, utility_weight=0, luminosity_weight=0, age_weight=0)
        report = forget(store, cfg, document_boundary=True)
        assert store.index_of_row(child[2]) is None
        if relation:
            assert store.index_of_row(parent[2]) is None and report['cascade'] == 1
        else:
            index = store.index_of_row(parent[2])
            assert index is not None and store.refs[index, 0] == -1
            assert store.meaning_of(index).role_refs[0] is None
            torch.testing.assert_close(store.meaning_of(index).roles, root.roles)
            assert report['coarsened'] == 1


def test_live_parent_pins_its_child():
    store, child, parent = linked_store()
    forget(store, config(), document_boundary=True, retained_occurrences=(parent,))
    assert store.index_of_row(child[2]) is not None
    assert store.meaning_of(store.index_of_row(parent[2])).role_refs[0] == child


def test_wording_is_one_stage_and_checkpoint_preserves_the_result():
    store = full_store()
    for row in range(len(store)):
        store.set_origin(row, 0, text=f'source {row}')
    before = store.slots.clone()
    with pytest.warns(RuntimeWarning, match='low water'):
        result = forget(store, config(), document_boundary=True)
    assert result['wording'] == 8 and result['after'] == 8
    torch.testing.assert_close(store.slots, before)
    assert all(store.text_of(row) is None for row in range(8))
    restored = LTM0(4, capacity=10)
    restored.load_state_dict(store.state_dict())
    restored.load_semantic_extras(store.semantic_extras())
    assert int(restored.forget_passes) == 1
    assert forget(restored, config(), document_boundary=True)['after'] == 5
    # Deleted stable addresses cannot alias a later physical occupant.
    old = int(store.address_keys[0])
    append(restored, 20)
    assert restored.index_of_row(old) is None


def test_pair_unlinking_preserves_observation_and_restores_cleanly():
    store = LTM0(4, capacity=4)
    value = ConceptualMeaning.from_description(torch.tensor([1., 0., 0., 0.]))
    from meaning_fixtures import symbol_prediction, estimate_meaning
    estimate = store.append_estimate(estimate_meaning(symbol_prediction(torch.eye(4),
        torch.full((4,2), .5))), presence_logits=torch.zeros(3))
    observation = store.append_meaning(value, kind='observation', trust=1., sentence_index=1, stream=0)
    store.link_estimate_observation(estimate, observation)
    observed = store.occurrence_of(observation)
    append(store, 2, trust=.8)
    append(store, 3, trust=.9)
    cfg = config(trust_weight=1, utility_weight=0, luminosity_weight=0, age_weight=0)
    forget(store, cfg, document_boundary=True)
    assert store.index_of_row(observed[2]) is not None
    assert store._expectation_rows == {}
    restored = LTM0(4, capacity=4)
    restored.load_state_dict(store.state_dict())
    restored.load_semantic_extras(store.semantic_extras())


def test_transitive_row_is_expendable_but_its_proof_survives():
    from LTM1 import LTM1
    from Forgetting import part_supports, forget_parts
    store = LTM1(capacity=5)
    rows = [store.add(left, right, truth_plus=1)
            for left, right in ((10, 20), (20, 30), (10, 30))]
    store.add(40, 50, truth_plus=.5)
    cfg = config(trust_weight=0, utility_weight=1, luminosity_weight=0, age_weight=0)
    assert part_supports(store) == {rows[2]: [frozenset(rows[:2])]}
    forget_parts(store, cfg)
    assert not store.contains(rows[2])
    assert all(store.contains(ref) for ref in rows[:2])


def test_epoch_trigger_and_disabled_setting():
    store = full_store()
    assert forget(store, config(enabled=False), document_boundary=True) is None
    cfg = config(high_water=1., at_epoch=True)
    assert forget(store, cfg, document_boundary=True) is None
    assert forget(store, cfg, epoch_boundary=True)['after'] == 5


def test_legacy_checkpoint_does_not_infer_trust_from_evidence():
    store = full_store()
    state = copy.deepcopy(store.state_dict())
    del state['source_trust']
    for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence', 'forget_cutoff'):
        del state[name]
    restored = LTM0(4, capacity=10)
    restored.load_state_dict(state)
    assert not restored.source_trust.any()
    torch.testing.assert_close(restored.c_plus, store.c_plus)


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
        store.append_idea(torch.ones(8), trust=1., sentence_index=i)
    forget(store, config(trust_weight=1, utility_weight=0, luminosity_weight=0, age_weight=0),
           document_boundary=True)
    assert store.parts.contains(edge)
    store.parts.retain([i for i in store.parts.indices().tolist() if i != edge])
    assert store.parts.find(left, right) is None


@pytest.mark.parametrize('training,stateless,deleted', [(True, True, True), (False, False, True), (False, True, False)])
def test_automatic_hook_runs_after_backward_and_honors_serving_mode(monkeypatch, training, stateless, deleted):
    from Models import BasicModel
    store = full_store()
    leaf = torch.tensor(2., requires_grad=True)
    loss = (store.slots[:8].clone() * leaf).sum()
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
    # Eight closings now retain prediction/observation pairs plus conceived
    # absences. Allow three rows per closing while keeping the original
    # absolute forgetting thresholds: start at eight, retain six.
    settings = dict(forgetHighWater=8/24, forgetLowWater=6/24,
                    forgetMinInterval=256, forgetting=True)
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1,
        word_buckets='16', ltm_capacity=24, architecture_overrides=settings)
    assert type(model) is BasicModel
    model._tensor_peer_while_eager = True
    model.stateless = False
    # Single-word documents exercise memory turnover independently of
    # random initial grammar choices. This is a mechanism certificate.
    model.languageSpace.language_layer.operation_layer.set_stage_menu(('STOP',))
    data = model.inputSpace.data
    monkeypatch.setattr(data, 'has_supervised_outputs', False)
    store = model.symbolSpace.ltm_store
    assert store.capacity == 24
    counters = ('forget_passes', 'forget_sentences', 'forget_last_sentence', 'forget_cutoff')
    def document(name, sentences):
        data.source_addresses = {'validation': [dict(document=name, document_key=name,
            sentence=i, sentence_index=i, split='validation') for i in range(len(sentences))]}
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
        assert report['before'] >= 8 and report['after'] == len(store) == 6
        assert report['target_reached']
        messages = capsys.readouterr().out
        assert f'[LTM forgetting] {report}' in messages
        print(f'[LTM forgetting] {report}')
        snapshot = {name: getattr(store, name).clone() for name in counters}
        occupancy = len(store)
        addresses = store.address_keys[:occupancy].clone()
        checkpoint = tmp_path/'after-forgetting.ckpt'
        model.save_weights(str(checkpoint))
        document('second', ['dog', 'cat'])
        assert int(store.forget_sentences) == int(snapshot['forget_sentences']) + 2
        assert int(store.forget_passes) == 1  # The configured interval still applies.
        assert len(store) > occupancy
        assert any(key not in addresses.tolist() for key in store.address_keys[:len(store)].tolist())
        assert model._last_sentence_field is not None
    finally:
        model.End()
    fresh = tmp_path/'fresh'
    fresh.mkdir()
    restored = _tiny_canonical_model(fresh, monkeypatch, batch_size=1,
        word_buckets='16', ltm_capacity=24, architecture_overrides=settings)
    try:
        assert restore(restored, str(checkpoint))
        loaded = restored.symbolSpace.ltm_store
        assert len(loaded) == occupancy
        torch.testing.assert_close(loaded.address_keys[:occupancy], addresses, rtol=0, atol=0)
        for name, value in snapshot.items():
            torch.testing.assert_close(getattr(loaded, name), value, rtol=0, atol=0)
        assert all(loaded.meaning_of(i) is not None for i in range(occupancy))
    finally:
        restored.End()


def test_scope_prevents_false_deduction_or_dissonance():
    store = full_store()
    from Forgetting import part_supports
    # A quoted situation has its own symbol; its edge does not give the
    # unquoted middle argument a second outgoing hop.
    store.parts.add(10, 20, truth_plus=1)
    store.parts.add(200, 30, truth_plus=1)
    conclusion = store.parts.add(10, 30, truth_plus=1)
    assert conclusion not in part_supports(store.parts)
    store.set_evidence(3, 1., 0.)
    store.set_evidence(4, 0., 1.)
    store.source_trust[3:5] = torch.tensor([1., .5])
    store._semantic_rows[int(store.address_keys[4])]['scope'] = ('quoted',)
    assert retention_values(store, config())['luminosity'][4] == 1


def test_optional_coverage_separates_both_from_neither():
    store = LTM0(4, capacity=4)
    append(store, 0, evidence=(1, 0))
    append(store, 1, evidence=(0, 0))
    terms = retention_values(store, config(coverage=True))
    assert terms['luminosity'][0] > 0 and terms['luminosity'][1] == 0
    store.set_evidence(1, 1, 1)
    assert retention_values(store, config(coverage=True))['luminosity'][1] == 0


@pytest.mark.parametrize('kwargs', [{'low_water': .9}, {'half_life': 0}, {'age_weight': -1},
                                   {'similarity': float('nan')}, {'protect_origins': (3,)}])
def test_invalid_configuration_fails_before_a_pass(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_packed_and_single_presentations_keep_the_same_stream_history():
    snapshots = []
    for packing in (1, 4):
        store = LTM0(4, capacity=10)
        for first in range(0, 8, packing):
            for sentence in range(first, min(8, first + packing)):
                append(store, sentence, surprise=sentence / 8)
                store.forget_sentences.add_(1)
            assert forget(store, config()) is None
        forget(store, config(), document_boundary=True)
        snapshots.append((store.state_dict(), store.semantic_extras()))
    for name in snapshots[0][0]:
        torch.testing.assert_close(snapshots[0][0][name], snapshots[1][0][name])
    assert snapshots[0][1] == snapshots[1][1]


def test_legacy_source_trust_can_load_into_a_larger_inventory():
    store = full_store()
    state = copy.deepcopy(store.state_dict())
    state['trust'] = state.pop('source_trust')
    restored = LTM0(4, capacity=20)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.source_trust[:8], store.source_trust[:8])
    assert not restored.source_trust[8:].any()


def test_losing_a_proof_raises_pending_conclusions_utility():
    from Forgetting import part_supports, forget_parts
    from LTM1 import LTM1
    store = LTM1(capacity=5)
    proof = store.add(10, 20, truth_plus=1)
    store.add(20, 30, truth_plus=1)
    conclusion = store.add(10, 30, truth_plus=1)
    low = store.add(40, 50, surprise=0.)
    assert conclusion in part_supports(store)
    # Conflicting evidence joins the same edge, rather than a second row.
    store.add(10, 20, truth_minus=1, witness=False)
    assert conclusion not in part_supports(store)
    cfg = config(low_water=.6, trust_weight=0, utility_weight=1, luminosity_weight=0, age_weight=0)
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
    store.set_origin(7, store.ORIGIN_PROVISIONED)
    key = int(store.address_keys[7])
    before = {name: value[7].clone() for name, value in store.named_buffers()
              if value.ndim and value.shape[0] == store.capacity and not name.startswith('posting_')}
    forget(store, config(), document_boundary=True)
    row = store.index_of_row(key)
    for name, value in before.items():
        torch.testing.assert_close(getattr(store, name)[row], value, msg=name)
    # Point-only rows remain recoverable from tensor state without a sidecar.
    assert not store.metadata_required[:len(store)].any()
    restored = LTM0(4, capacity=10)
    restored.load_state_dict(store.state_dict())
    assert all(restored.meaning_of(i) is not None for i in range(len(restored)))
