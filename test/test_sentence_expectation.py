"""Reviewer probes for integrated spec §10.2: whole-idea prediction.

The old root-only tests remain a labelled compatibility/benchmark baseline.
These tests exercise the production structured predictor and boundary owner.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Layers import InterSentenceLayer, TernaryTruthStore


def layer(batch=1, consolidated=False):
    torch.manual_seed(165)
    result = InterSentenceLayer(
        n_symbols=4, max_depth=8, n_dim=4, concept_dim=4,
        batch=batch, expectation_scope="structured")
    if consolidated:
        result._ltm_store = TernaryTruthStore(4, capacity=32)
    return result


def observe(model, value, *, document="a", layout="infix", mask=None):
    model.predict_and_observe_stm_end_state(
        [len(value)], [value], documents=[document], layout=layout,
        role_masks=None if mask is None else [mask])


@pytest.mark.parametrize("consolidated", [False, True])
@pytest.mark.parametrize("role", [1, 2])
def test_np1_fixed_other_roles_change_prediction(consolidated, role):
    a = layer(consolidated=consolidated)
    b = copy.deepcopy(a)
    source = torch.arange(12.).reshape(3, 4) / 12
    changed = source.clone()
    changed[role] += 1
    observe(a, source)
    observe(b, changed)
    pa, pb = a.expect_next_meaning(), b.expect_next_meaning()
    assert pa is not None and pb is not None
    assert not torch.allclose(pa.roles, pb.roles)
    assert not torch.allclose(pa.roles[0], pa.roles[1])


def test_padding_and_legacy_operand_orientation_are_explicit():
    a, b, c = layer(), layer(), layer()
    full = torch.arange(12.).reshape(3, 4) / 12
    observe(a, full)
    observe(b, full[[2, 0, 1]], layout="stm")
    torch.testing.assert_close(a.expect_next_meaning().roles,
                               b.expect_next_meaning().roles)
    a.begin_document(0, "new")
    short = full[:1]
    padded = torch.cat((short, torch.full((2, 4), 999.)))
    observe(a, short, document="new")
    observe(c, padded, mask=torch.tensor([True, False, False]))
    torch.testing.assert_close(a.expect_next_meaning().roles,
                               c.expect_next_meaning().roles)


def test_document_change_is_cold_preserves_scored_loss_and_other_row():
    model = layer(batch=2)
    payloads = [torch.ones(3, 4), torch.full((3, 4), 2.)]
    for _ in range(2):
        model.predict_and_observe_stm_end_state(
            [3, 3], payloads, documents=["a", "b"], layout="infix")
    assert model._inter_loss_count == 2
    before = model.expect_next_meaning(1).roles.detach().clone()
    model.begin_document(0, "different")
    assert model.expect_next_meaning(0) is None
    torch.testing.assert_close(model.expect_next_meaning(1).roles, before)
    assert model._inter_loss_count == 2
    model.predict_and_observe_stm_end_state(
        [3, 0], [payloads[0] * 90, None], mask=[True, False],
        documents=["different", "b"], layout="infix")
    assert model._inter_loss_count == 2
    assert model.consume_inter_loss() is not None
    assert model.consume_inter_loss() is None


def test_full_role_loss_trains_source_and_predictor_but_never_target():
    model = layer()
    encoder = torch.nn.Linear(4, 4, bias=False)
    source = encoder(torch.arange(12.).reshape(3, 4) / 12)
    target = torch.nn.Parameter(torch.ones(3, 4))
    observe(model, source)
    observe(model, target)
    loss = model.consume_inter_loss()
    assert loss is not None
    loss.backward()
    assert target.grad is None
    assert encoder.weight.grad is not None and encoder.weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in model._inter_predictor.parameters())
    assert all(not p.requires_grad for _, p, _ in model.get_stm_chain())


@pytest.mark.parametrize("consolidated", [False, True])
def test_streaming_and_packed_boundary_use_same_full_roles(consolidated):
    from Models import BasicModel as Model

    packed = layer(consolidated=consolidated)
    streaming = layer(consolidated=consolidated)
    meanings = torch.arange(24.).reshape(1, 2, 3, 4) / 24
    # The stored snapshots are STM-order. Both have identical NP1 but
    # different VP/NP2; an old root-only drain cannot pass this contract.
    meanings[:, 1, 1] = meanings[:, 0, 1]
    isp = SimpleNamespace(
        _packed_sentence_slot_end_positions=torch.tensor([[0, 1]]),
        _packed_sentence_slot_mask=torch.tensor([[True, True]]),
        _packed_sentence_counts_host=(2,))
    host = SimpleNamespace(
        inputSpace=isp,
        conceptualSpace=SimpleNamespace(_ltm_consolidation=consolidated),
        symbolSpace=SimpleNamespace(discourse=packed, ltm_store=packed._ltm_store),
        _packed_sentence_roots=meanings[:, :, 1],
        _tensor_sentence_roots_live=meanings.reshape(1, 2, 12),
        _tensor_sentence_roots_depth=torch.tensor([[3, 3]]),
        _tensor_final_end_slots=meanings[:, -1],
        _tensor_final_end_depth=torch.tensor([3]),
        _expectation_documents=(("a", "a"),))
    host._expectation_documents_for_slot = lambda t, batch: ["a"]
    Model._drain_packed_stm_end_states(host)
    for t in range(2):
        observe(streaming, meanings[0, t], layout="stm")
    assert packed._inter_loss_count == streaming._inter_loss_count == 1
    torch.testing.assert_close(packed.consume_inter_loss(),
                               streaming.consume_inter_loss())
    torch.testing.assert_close(packed.expect_next_meaning().roles,
                               streaming.expect_next_meaning().roles)
    if consolidated:
        # The second external input now keeps its antecedent estimate beside
        # (not inside) the observed sequence. The predictor remains aligned
        # with the streaming path because its context contains observations
        # only; the durable owner retains the auditable pair separately.
        assert len(packed._ltm_store) == 3
        assert [packed._ltm_store.row(i)["kind"] for i in range(3)] == [
            "observation", "estimate", "observation"]
        assert (packed._ltm_store.expectation_pair(1)["source_occurrences"]
                == (packed._ltm_store.row(0)["occurrence"],))
        torch.testing.assert_close(packed._ltm_store.row(0)["vp"], meanings[0, 0, 2])


def test_restore_keeps_weights_and_durable_memory_but_starts_context_cold():
    model = layer()
    observe(model, torch.ones(3, 4))
    state = copy.deepcopy(model.state_dict())
    observe(model, torch.full((3, 4), 99.))
    model.load_state_dict(state)
    assert model.expect_next_meaning() is None
    assert model.consume_inter_loss() is None


def test_source_document_addresses_follow_each_packed_sentence():
    from Models import BasicModel as Model

    data = SimpleNamespace(source_addresses={"train": [
        {"document": 11}, {"document": 11}, {"document": 12}]})
    host = SimpleNamespace(inputSpace=SimpleNamespace(data=data))
    Model._stage_expectation_documents(host, "train", [[0, 1, 2], [2]], 2)
    assert host._expectation_documents[0][0] == host._expectation_documents[0][1]
    assert host._expectation_documents[0][1] != host._expectation_documents[0][2]
    assert host._expectation_documents[0][2] == host._expectation_documents[1][0]


def test_nonfinite_scored_role_error_is_rejected():
    model = layer()
    observe(model, torch.ones(3, 4))
    with pytest.raises(FloatingPointError, match="prediction loss"):
        observe(model, torch.full((3, 4), 1e30))


def test_legacy_root_checkpoint_has_declared_fresh_structured_head_migration():
    old = InterSentenceLayer(
        n_symbols=4, max_depth=8, n_dim=4, concept_dim=4)
    new = layer()
    before = copy.deepcopy(new._inter_predictor.state_dict())
    with pytest.warns(UserWarning, match="root.*structured"):
        new.load_state_dict(old.state_dict(), strict=True)
    for name, value in new._inter_predictor.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for name, value in new.predictor.state_dict().items():
        torch.testing.assert_close(value, old.predictor.state_dict()[name])
    assert new.expect_next_meaning() is None


def test_provisioned_ltm_does_not_make_a_cold_prediction_a_seed():
    from Models import BasicModel

    disc = layer(consolidated=True)
    disc._ltm_store.append_idea(torch.ones(4), trust=1.)
    host = SimpleNamespace(symbolSpace=SimpleNamespace(discourse=disc))
    assert disc.expect_next_meaning() is None


def test_new_document_is_reset_before_forward():
    from Models import BasicModel

    disc = layer()
    observe(disc, torch.ones(3, 4), document=("train", 11))
    host = SimpleNamespace(
        inputSpace=SimpleNamespace(data=SimpleNamespace(
            source_addresses={"train": [{"document": 12}]})),
        symbolSpace=SimpleNamespace(discourse=disc))
    BasicModel._stage_expectation_documents(host, "train", [[0]], 1)
    assert disc.expect_next_meaning() is None


def test_single_sentence_cursor_keeps_true_document_continuations():
    from data import SentenceStreamDataset

    cursor = SentenceStreamDataset(
        ["a", "b", "c", "d"], 1, document_ids=[10, 10, 11, 11])
    assert [cursor.next_tick()[2][0] for _ in range(4)] == [False, True, False, True]


def test_expectation_scope_is_a_checked_configuration_choice():
    etree = pytest.importorskip("lxml.etree")

    root = Path(__file__).resolve().parents[1]
    schema = etree.XMLSchema(etree.parse(str(root / "data/model.xsd")))
    tree = etree.parse(str(root / "data/model.xml"))
    training = tree.find("architecture/training")
    scope = training.find("sentenceExpectationScope")
    if scope is None:
        scope = etree.SubElement(training, "sentenceExpectationScope")
    for value in ("structured", "root"):
        scope.text = value
        assert schema.validate(tree), str(schema.error_log)
    scope.text = "flattened"
    assert not schema.validate(tree)


def test_unknown_expectation_scope_is_rejected_at_construction():
    with pytest.raises(ValueError, match="expectation_scope"):
        InterSentenceLayer(n_symbols=4, max_depth=8, n_dim=4,
                           concept_dim=4, expectation_scope="flattened")


def test_real_provisioning_is_not_an_external_prediction_stream(monkeypatch):
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG

    model = _make_model(_SERIAL_CONFIG)
    discourse = model.symbolSpace.discourse
    bound = []

    def forbid_external_binding(*args, **kwargs):
        bound.append((args, kwargs))
        raise AssertionError("provisioning is not an external observation")

    monkeypatch.setattr(discourse, "bind_observation_occurrence",
                        forbid_external_binding)
    model.provision_ltm()
    assert not bound
    assert len(model.symbolSpace.ltm_store) == 3
    assert not any(discourse._inter_context)
    assert model.symbolSpace.discourse.expect_next_meaning() is None


def test_truth_ingestion_preserves_all_external_row_contexts_and_losses():
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG

    model = _make_model(_SERIAL_CONFIG)
    disc = model.symbolSpace.discourse
    disc.train()
    disc.ensure_batch(2)
    d = disc.concept_dim
    for value in (0.2, 0.4):
        disc.predict_and_observe_stm_end_state(
            [1, 1], [torch.full((1, d), value), torch.full((1, d), -value)])
    before = [disc.expect_next_meaning(b).roles.detach().clone() for b in range(2)]
    loss_count = disc._inter_loss_count
    model.provision_ltm()
    assert disc._batch == 2
    assert disc._inter_loss_count == loss_count
    assert [len(c) for c in disc._inter_context] == [2, 2]
    for b in range(2):
        torch.testing.assert_close(disc.expect_next_meaning(b).roles, before[b])
