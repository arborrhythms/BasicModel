"""Reviewer probes for integrated spec §10.2: whole-idea prediction.

The old root-only tests remain a labelled compatibility/benchmark baseline.
These tests exercise the production structured predictor and boundary owner.
"""
from Memory import STM0
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Layers import SymbolExpectation
from Memory import LTM0


def layer(batch=1, consolidated=False):
    torch.manual_seed(165)
    result = SymbolExpectation(
        n_symbols=4, max_depth=8, n_dim=4, concept_dim=4,
        batch=batch)
    result.set_symbol_bank(torch.eye(result.concept_dim))
    result.set_symbol_bank(torch.eye(4))
    if consolidated:
        result._ltm_store = STM0(4, capacity=32, relation_capacity=8192)
    return result


def admit(model, value, *, document="a", b=0, symbols=(), activations=()):
    """Supply a retained idea to the native recent-situation writer."""
    if model._ltm_store is None:
        model._ltm_store = STM0(model.concept_dim, capacity=model.ltm_capacity)
    model.begin_document(b, document)
    point = value.reshape(-1, model.concept_dim)[0]
    index = model._ltm_store.append_idea(point, trust=1.)
    occurrence = model._ltm_store.occurrence_of(index)
    model.include_occurrence(b, occurrence, symbols=symbols, activations=activations)
    return occurrence











def test_streaming_and_packed_boundary_use_same_full_roles(monkeypatch):
    from reading_fixtures import commit_reading, finish_reading
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entries = [f.program(('change', 'cat', ('verb', verb, obj)))
               for verb, obj in (('chases', 'mouse'), ('catches', 'bird'))]
    meanings = [finish_reading(f.language, entry, registry=f.registry).meaning for entry in entries]
    torch.testing.assert_close(meanings[0].roles[0], meanings[1].roles[0])
    assert not torch.equal(meanings[0].roles[1:], meanings[1].roles[1:])
    packed = SymbolExpectation(n_symbols=8, max_depth=8, n_dim=8, concept_dim=8)
    packed.set_symbol_bank(torch.eye(8))
    streaming = copy.deepcopy(packed)
    packed._ltm_store = streaming._ltm_store = f.store
    for sid, entry in enumerate(entries):
        commit_reading(f.language, f.registry, entry, f.store, discourse=packed, sid=sid, document='a')
    for frame in packed.situation_references():
        streaming.include_occurrence(0, frame.occurrence, symbols=frame.symbol_ids, activations=frame.activations)
    assert len(f.store) == 2
    for a, b in zip(packed.situation_references(), streaming.situation_references(), strict=True):
        assert a.occurrence == b.occurrence
        assert a.symbol_ids == b.symbol_ids and a.activations == b.activations
        torch.testing.assert_close(a.roles, b.roles)
    torch.testing.assert_close(f.store.meaning_of(0).roles[0], entries[0].end_state[0])
    assert f.store.meaning_of(0).role_mask.tolist() == [True, False, False]



def test_restore_keeps_weights_and_durable_memory_but_starts_context_cold():
    model = layer()
    admit(model, torch.ones(3, 4))
    state = copy.deepcopy(model.state_dict())
    admit(model, torch.full((3, 4), 99.))
    model.load_state_dict(state)
    assert model.situation_references() == ()


def test_source_document_addresses_follow_each_packed_sentence():
    from Models import BasicModel as Model

    data = SimpleNamespace(source_addresses={"train": [
        {"document": 11}, {"document": 11}, {"document": 12}]})
    host = SimpleNamespace(inputSpace=SimpleNamespace(data=data))
    Model._stage_expectation_documents(host, "train", [[0, 1, 2], [2]], 2)
    assert host._expectation_documents[0][0] == host._expectation_documents[0][1]
    assert host._expectation_documents[0][1] != host._expectation_documents[0][2]
    assert host._expectation_documents[0][2] == host._expectation_documents[1][0]




def test_retired_heads_have_explicit_fresh_activation_migration():
    model = layer()
    before = copy.deepcopy(model._inter_predictor.state_dict())
    old = {'_inter_predictor.network.0.weight': torch.randn(32, 120),
           'predictor.0.weight': torch.randn(16, 28),
           'word_predictor.0.weight': torch.randn(16,28)}
    with pytest.warns(UserWarning, match='fresh shared activation'):
        model.load_state_dict(old, strict=True)
    for name, value in model._inter_predictor.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert model.situation_references() == ()




def test_provisioned_ltm_does_not_make_a_cold_prediction_a_seed():
    from Models import BasicModel

    disc = layer(consolidated=True)
    disc._ltm_store.append_idea(torch.ones(4), trust=1.)
    host = SimpleNamespace(symbolSpace=SimpleNamespace(expectation=disc))
    assert disc.situation_references() == ()


def test_new_document_is_reset_before_forward():
    from Models import BasicModel

    disc = layer()
    admit(disc, torch.ones(3, 4), document=("train", 11))
    host = SimpleNamespace(
        inputSpace=SimpleNamespace(data=SimpleNamespace(
            source_addresses={"train": [{"document": 12}]})),
        symbolSpace=SimpleNamespace(expectation=disc))
    BasicModel._stage_expectation_documents(host, "train", [[0]], 1)
    assert disc.situation_references() == ()


def test_single_sentence_cursor_keeps_true_document_continuations():
    from data import SentenceStreamDataset

    cursor = SentenceStreamDataset(
        ["a", "b", "c", "d"], 1, document_ids=[10, 10, 11, 11])
    assert [cursor.next_tick()[2][0] for _ in range(4)] == [False, True, False, True]


def test_current_schema_validates_and_retired_scope_is_rejected():
    from lxml import etree
    root = Path(__file__).resolve().parents[1]
    schema = etree.XMLSchema(etree.parse(str(root / 'data/model.xsd')))
    for name in ('model.xml', 'BasicModel.xml'):
        tree = etree.parse(str(root / 'data' / name))
        assert schema.validate(tree), str(schema.error_log)
        scope = etree.SubElement(tree.find('architecture/training'), 'sentenceExpectationScope')
        for value in ('symbols', 'structured', 'root'):
            scope.text = value
            assert not schema.validate(tree)


def test_retired_scope_is_rejected_at_construction():
    with pytest.raises(TypeError, match='expectation_scope'):
        SymbolExpectation(4, 8, 4, expectation_scope='symbols')


def test_real_provisioning_is_not_an_external_prediction_stream(monkeypatch, eager_reading):
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG

    model = _make_model(_SERIAL_CONFIG)
    discourse = model.symbolSpace.expectation
    bound = []

    def forbid_external_binding(*args, **kwargs):
        bound.append((args, kwargs))
        raise AssertionError("provisioning is not an external observation")

    monkeypatch.setattr(discourse, "include_occurrence",
                        forbid_external_binding)
    model.provision_ltm()
    assert not bound
    store = model.symbolSpace.ltm_store
    assert len(store) == 3
    assert not any(discourse._inter_context)
    assert model.symbolSpace.expectation.situation_references() == ()


@pytest.mark.usefixtures('eager_reading')
def test_truth_ingestion_preserves_all_external_row_contexts_and_losses():
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG)
    disc = model.symbolSpace.expectation
    disc.train(); disc.ensure_batch(2)
    for value in (0.2, 0.4):
        for b, sign in enumerate((1, -1)):
            admit(disc, torch.full((1, disc.concept_dim), sign*value), b=b)
    before = [[(f.occurrence, f.roles.clone(), f.symbol_ids, f.activations)
               for f in disc.situation_references(b)] for b in range(2)]
    model.provision_ltm()
    assert disc._batch == 2
    assert [len(c) for c in disc._inter_context] == [2, 2]
    for b in range(2):
        after = disc.situation_references(b)
        for frame, (occurrence, roles, symbols, activations) in zip(after, before[b], strict=True):
            assert frame.occurrence == occurrence
            assert frame.symbol_ids == symbols and frame.activations == activations
            torch.testing.assert_close(frame.roles, roles, rtol=0, atol=0)
