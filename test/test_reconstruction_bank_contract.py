"""A requested reconstruction must have a populated, sentence-owned bank."""
import gc
from types import SimpleNamespace

import pytest
import torch

from test_packed_reconstruction_parity import build_model
from test_reverse_traversal import _stage_packed


def _bank_fixture(device):
    from Models import BasicModel
    rows = torch.tensor([[1, 2, -1]], device=device)
    isp = SimpleNamespace(
        _word_active_mask=torch.tensor([[True, True, False]], device=device),
        _packed_sentence_ids=torch.tensor([[0, 1, -1]], device=device),
        _ar_concept_lookup_rows=rows,
        _ar_concept_lookup_atoms=torch.ones(1, 3, 4, device=device),
        _ar_concept_lookup_sentence_ids=torch.tensor([[0, 1, -1]], device=device),
        _ar_bank_bytes=torch.ones(1, 3, 2, dtype=torch.long, device=device),
        _ar_bank_valid=torch.tensor([[[True, True], [True, True], [False, False]]], device=device),
    )
    model = SimpleNamespace(inputSpace=isp)
    return lambda: BasicModel._validate_reconstruction_bank(model), isp


def test_reconstruction_bank_validation_does_not_read_device_scalars():
    from torch._subclasses.fake_tensor import FakeTensorMode
    # Fake tensors reject data-dependent scalar reads on every host.
    # Shapes and dtypes remain available without consulting device contents.
    with FakeTensorMode():
        validate, _ = _bank_fixture("cpu")
        validate()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_reconstruction_bank_validation_has_no_mps_host_sync():
    validate, _ = _bank_fixture("mps")
    torch.mps.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        validate()
    # Synchronize only outside the measured boundary to surface device errors.
    torch.mps.synchronize()
    events = {event.key for event in profile.key_averages()}
    assert "aten::_local_scalar_dense" not in events
    assert "aten::item" not in events


@pytest.mark.parametrize("field", ["_packed_sentence_ids", "_ar_concept_lookup_sentence_ids"])
def test_reconstruction_bank_rejects_out_of_range_ownership(field):
    validate, isp = _bank_fixture("cpu")
    getattr(isp, field)[0, 0] = 3
    with pytest.raises(RuntimeError, match="invalid.*sentence ownership"):
        validate()


@pytest.fixture(autouse=True)
def _release_independent_models():
    # Each fault constructs a separate model with cyclic space backrefs.
    # Release them between cases, not only at the module boundary: eight
    # retained models can exceed the bounded worker's memory cap.
    gc.collect()
    yield
    torch._dynamo.reset()
    gc.collect()


def _stage(model):
    model._install_unit_span_fn()
    _stage_packed(model, [["quorp flarn", "wug blim"], ["quorp flarn"]])


def test_first_sight_admits_concepts_before_reconstruction(tmp_path):
    model = build_model(tmp_path, active_vectors=1)
    try:
        owner = model._concept_owner()
        before = owner._concept_allocator.next_id
        active_before = model._active_inventory_rows
        assert not owner._concept_allocator.word_obj_meta
        _stage(model)
        isp = model.inputSpace
        active = isp._word_active_mask
        assert owner._concept_allocator.next_id > before
        assert model._active_inventory_rows > active_before
        assert (isp._ar_word_concept_rows[active] >= 0).all()
        assert (isp._ar_word_object_rows[active] >= 0).all()
        for row in range(active.shape[0]):
            for sentence in isp._packed_sentence_ids[row, active[row]].unique():
                candidates = isp._ar_concept_lookup_sentence_ids[row] == sentence
                assert isp._ar_bank_valid[row, candidates].any()
        admitted = owner._concept_allocator.next_id
        model._stage_serial_concept_rows()
        assert owner._concept_allocator.next_id == admitted, "known concepts must be reused"
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("fault", [
    "rows", "atoms", "ownership", "surfaces", "surface_mask", "second_sentence_empty",
])
def test_staging_rejects_missing_or_unusable_reconstruction_bank(tmp_path, monkeypatch, fault):
    model = build_model(tmp_path)
    stage = model._stage_snapshot_bytes

    def broken_stage():
        stage()
        isp = model.inputSpace
        fields = dict(rows="_ar_concept_lookup_rows", atoms="_ar_concept_lookup_atoms",
                      ownership="_ar_concept_lookup_sentence_ids", surfaces="_ar_bank_bytes",
                      surface_mask="_ar_bank_valid")
        if fault in fields:
            setattr(isp, fields[fault], None)
        else:
            # The first sentence still has candidates. Coverage must be
            # checked per sentence, not just once for the whole batch.
            foreign = isp._ar_concept_lookup_sentence_ids == 1
            isp._ar_bank_valid[foreign] = False

    monkeypatch.setattr(model, "_stage_snapshot_bytes", broken_stage)
    try:
        with pytest.raises(RuntimeError, match="reconstruction.*(bank|candidate|ownership|surface)"):
            _stage(model)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_failed_first_sight_admission_cannot_become_null_reconstruction(tmp_path, monkeypatch):
    model = build_model(tmp_path)
    monkeypatch.setattr(model._concept_owner(), "_automatic_word_object_meta", lambda *a, **k: None)
    try:
        with pytest.raises(RuntimeError, match="reconstruction.*(admission|candidate)"):
            _stage(model)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
