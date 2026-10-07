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
