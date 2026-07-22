"""Compatibility tests at the legacy Space/new carrier seam."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from pipeline import (
    DenseEvent,
    LossTerm,
    MaterializeMode,
    PipelineAddress,
    PipelineControl,
    PipelineEffects,
    PipelineExecutor,
    PipelineStage,
    ReplicaRegistry,
    SelectedEvent,
    SubSpace as PipelineSubSpace,
    SubSpaceSchema,
)
from Spaces import Space


class _Basis(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.W = nn.Parameter(rows.clone())
        self.use_dot_product = False

    def prototype(self):
        return self.W

    def lookup(self, indices):
        return self.W[indices]


class _Errors:
    def __init__(self, terms=()):
        self._terms = list(terms)

    def terms(self):
        return list(self._terms)


class _LegacySubspace:
    def __init__(self, basis, *, index=None, event=None, activation=None):
        self.codebook_slot = "event" if basis is not None else None
        self.event = basis
        self.what = None
        self.where = None
        self.when = None
        self.activation = None
        self._index = index
        self.nWhat = 2
        self.nWhere = 1
        self.nWhen = 1
        self.valid_mask = torch.tensor([[True, False]])
        self.errors = _Errors(
            [("aux", torch.tensor(2.0, requires_grad=True), 0.25, "test", "other")]
        )
        self._dense_event = event
        self._activation = activation

    def effective_activation(self):
        return self._activation

    def materialize(self, mode="event"):
        if mode == "event":
            return self._dense_event
        raise ValueError(mode)


def _space(legacy, *, owner="tower.0"):
    space = Space.__new__(Space)
    nn.Module.__init__(space)
    space._codebook_parameter_version = 0
    space._codebook_structure_versions = {}
    space._codebook_owner_path = owner
    space.config_section = "TestSpace"
    space.nWhat = 2
    space.nWhere = 1
    space.nWhen = 1
    space.subspace = legacy
    return space


def _control(version=0):
    return PipelineControl(PipelineAddress(11, 0, parameter_version=version))


def test_space_exports_sparse_selection_with_live_read_only_identity():
    basis = _Basis(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    )
    legacy = _LegacySubspace(
        basis,
        index=torch.tensor([[[1], [0]]]),
        activation=torch.tensor([[1.0, 0.5]]),
    )
    space = _space(legacy)

    carrier = space.to_pipeline_carrier(_control())
    assert isinstance(carrier.payload, SelectedEvent)
    assert carrier.payload.indices.tolist() == [[1, 0]]
    assert carrier.control.valid_mask.tolist() == [[True, False]]
    assert carrier.payload.reader.identity.owner_path == "tower.0.event"
    assert not hasattr(carrier.payload.reader, "W")
    assert torch.equal(
        carrier.materialize(MaterializeMode.EVENT),
        basis.W[torch.tensor([[1, 0]])],
    )
    assert [loss.name for loss in carrier.effects.losses] == ["aux"]

    legacy._index.fill_(0)
    legacy._activation.zero_()
    assert carrier.payload.indices.tolist() == [[1, 0]]
    assert carrier.payload.activation.tolist() == [[1.0, 0.5]]

    space.mark_codebook_parameters_changed()
    assert carrier.payload.reader.identity.parameter_version == 1
    identity = space.mark_codebook_structure_changed("event")
    assert identity.structure_version == 1
    assert identity.parameter_version == 2


def test_executor_advances_bound_space_version_and_invalidates_replicas():
    basis = _Basis(torch.ones(2, 4))
    legacy = _LegacySubspace(
        basis,
        index=torch.tensor([[[1], [0]]]),
        activation=torch.ones(1, 2),
    )
    space = _space(legacy)
    reader = space.codebook_reader("event")
    old_identity = reader.identity
    replicas = ReplicaRegistry()
    replicas.install(old_identity, basis.W, device="cpu")
    executor = PipelineExecutor(
        [
            PipelineStage(
                "owned",
                lambda carrier: replace(carrier),
                parameter_owner=space,
            )
        ],
        replicas=replicas,
    )

    assert executor.advance_parameter_version() == 1
    assert space.codebook_parameter_version == 1
    assert reader.identity.parameter_version == 1
    try:
        replicas.resolve(old_identity, "cpu")
    except LookupError:
        pass
    else:
        raise AssertionError("global Parameter update must invalidate old replicas")


def test_legacy_spaces_pipeline_via_fresh_snapshot_adapters():
    first = _space(_LegacySubspace(None, event=torch.zeros(1, 2, 4)))
    second = _space(
        _LegacySubspace(None, event=torch.zeros(1, 2, 4)),
        owner="tower.1",
    )

    def add(amount):
        def call(legacy):
            event = legacy.materialize(mode="event")
            legacy.set_event(event + amount)
            return legacy

        return call

    executor = PipelineExecutor(
        [
            first.as_pipeline_stage(
                name="first",
                forward_call=add(1.0),
                device="cpu",
                pipeline_safe=True,
            ),
            second.as_pipeline_stage(
                name="second",
                forward_call=add(2.0),
                device="cpu",
                pipeline_safe=True,
            ),
        ],
        queue_capacity=1,
    )
    inputs = [
        PipelineSubSpace(
            SubSpaceSchema("input", 2, 1, 1),
            DenseEvent(torch.full((1, 2, 4), float(microbatch))),
            PipelineControl(PipelineAddress(4, microbatch, 0)),
        )
        for microbatch in (0, 1)
    ]
    run = executor.run(inputs)

    assert torch.all(run.outputs[0].payload.event == 3.0)
    assert torch.all(run.outputs[1].payload.event == 4.0)
    assert run.outputs[0] is not run.outputs[1]
    assert run.outputs[0].payload.event.data_ptr() != run.outputs[1].payload.event.data_ptr()


def test_legacy_stage_requires_explicit_training_safety_migration():
    space = _space(_LegacySubspace(None, event=torch.zeros(1, 2, 4)))
    stage = space.as_pipeline_stage(
        forward_call=lambda legacy: legacy,
        device="cpu",
        pipeline_safe=True,
    )
    with pytest.raises(RuntimeError, match="undeferred durable|legacy Space"):
        PipelineExecutor([stage], training=True)


def test_legacy_stage_requires_an_explicit_pipeline_safety_audit():
    space = _space(_LegacySubspace(None, event=torch.zeros(1, 2, 4)))
    stage = space.as_pipeline_stage(
        forward_call=lambda legacy: legacy,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="isolated stages|legacy Space"):
        PipelineExecutor([stage])


def test_dense_carrier_round_trips_through_unowned_legacy_input_adapter():
    dense = torch.arange(8.0).reshape(1, 2, 4)
    legacy = _LegacySubspace(None, event=dense, activation=torch.tensor([[1.0, 0.5]]))
    space = _space(legacy)
    carrier = space.to_pipeline_carrier(_control())
    assert isinstance(carrier.payload, DenseEvent)

    rebuilt = space.from_pipeline_carrier(carrier)
    assert torch.equal(rebuilt.materialize(mode="event"), dense)
    assert rebuilt.valid_mask.tolist() == [[True, False]]
    terms = rebuilt.errors.terms()
    assert len(terms) == 1
    assert terms[0][0] == "aux"

    dense.add_(100.0)
    assert torch.equal(
        carrier.payload.event,
        torch.arange(8.0).reshape(1, 2, 4),
    )


def test_prior_trace_and_typed_mutations_survive_legacy_snapshot():
    dense = torch.ones(1, 2, 4)
    legacy = _LegacySubspace(None, event=dense)
    space = _space(legacy)
    prior = PipelineSubSpace(
        SubSpaceSchema("upstream", 2, 1, 1),
        DenseEvent(dense),
        _control(),
        PipelineEffects(losses=(LossTerm("upstream", torch.tensor(1.0)),)),
    )

    carrier = space.to_pipeline_carrier(_control(), prior=prior)
    # The shared legacy Error snapshot is canonical and replaces the prior
    # loss tuple, avoiding duplicate upstream accumulation.
    assert [loss.name for loss in carrier.effects.losses] == ["aux"]
    assert carrier.trace is prior.trace


def test_legacy_duplicate_checkpoint_keys_migrate_to_single_owners(tmp_path):
    import recon_bench

    config = recon_bench._resolve_config("data/XOR_exact.xml")
    original, *_ = recon_bench._build_model(config)
    checkpoint = tmp_path / "legacy-ownership.ckpt"
    original.save_weights(checkpoint)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current = bundle["state_dict"]
    legacy = {}

    basis_roles = ("event", "what", "where", "when", "activation")
    encoder_roles = (
        "activeEncoding",
        "objectEncoding",
        "whatEncoding",
        "whereEncoding",
        "whenEncoding",
        "wordEncoding",
    )
    for key, value in current.items():
        old = key
        for role in basis_roles:
            old = old.replace(
                f"._owned_bases.{role}.", f".subspace.{role}."
            )
        for role in encoder_roles:
            old = old.replace(
                f"._owned_encoders.{role}.", f".subspace.{role}."
            )
        old = old.replace("._percept_store.", ".subspace.percept_store.")
        legacy[old] = value.detach().clone()

    # Recreate representative aliases from the pre-cleanup module graph.
    last_cs = len(original.conceptualSpaces) - 1
    last_ws = len(original.wholeSpaces) - 1
    for key, value in list(legacy.items()):
        if key.startswith(f"conceptualSpaces.{last_cs}."):
            legacy["conceptualSpace." + key.split(".", 2)[2]] = value.clone()
        if key.startswith(f"wholeSpaces.{last_ws}."):
            legacy["wholeSpace." + key.split(".", 2)[2]] = value.clone()
        if key.startswith("symbolSpace."):
            legacy[
                "inputSpace._model_symbolSpace." + key[len("symbolSpace.") :]
            ] = value.clone()

    # Recreate the VQ/parent duplicate for VQ-backed Basis owners.
    for key, value in list(legacy.items()):
        if not key.endswith(".W"):
            continue
        prefix = key[:-1]
        if any(candidate.startswith(prefix + "vq.") for candidate in legacy):
            legacy[prefix + "vq._codebook"] = value.clone()

    bundle["state_dict"] = legacy
    torch.save(bundle, checkpoint)

    restored, *_ = recon_bench._build_model(config)
    assert restored.load_weights(checkpoint, strict=True)
    restored_state = restored.state_dict()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored_state[key], value)

    # Even the legacy construction setter must rebind VQ to the new sole
    # owner Parameter; otherwise a post-load replacement would quantize with
    # an orphaned tensor.
    owned = next(
        module
        for module in restored.modules()
        if getattr(module, "vq", None) is not None
        and isinstance(getattr(module, "W", None), nn.Parameter)
    )
    replacement = nn.Parameter(owned.W.detach().clone())
    owned.setW(replacement)
    assert owned.W is replacement
    assert owned.vq.codebook is replacement
    assert "_codebook" not in owned.vq._parameters
