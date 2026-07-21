"""Boundary-safe growth of the canonical PartSpace radix dictionary."""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch


_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _parameter_backed_store(*, initial=2, maximum=4, threshold=1):
    from Layers import RadixLayer

    store = RadixLayer(
        dim=3,
        initial_cap=initial,
        max_cap=maximum,
        promotion_threshold=threshold,
        promotion_min_length=2,
    )
    basis = store._basis
    current = basis.getW()
    if not isinstance(current, torch.nn.Parameter):
        basis.setW(torch.nn.Parameter(current.detach().clone()))
    return store


class _Owner:
    def __init__(self, parameter):
        self.params = [parameter]
        self.nVectors = int(parameter.shape[0])

    def _replace_radix_codebook_parameter(self, old, new, capacity):
        self.params = [new if p is old else p for p in self.params]
        self.nVectors = int(capacity)


def test_promotion_queues_without_mid_forward_parameter_replacement():
    store = _parameter_backed_store()
    store.insert(b"a")
    store.insert(b"b")
    old = store._basis._parameters["W"]
    before = old.detach().clone()

    # Promotion is due, but physical capacity is full.  The current request
    # still spells the word out losslessly as its two byte rows.
    assert store.observe_chunk(b"ab") is None
    assert store.spell_out(b"ab") == [0, 1]
    assert store._basis._parameters["W"] is old
    assert torch.equal(old.detach(), before)
    assert b"ab" not in store
    assert store.pending_promotions == 1
    pending_init = store._pending_promotions[b"ab"]
    assert pending_init.device.type == "cpu"
    assert not pending_init.requires_grad


def test_boundary_growth_preserves_rows_and_pads_adam_moments():
    store = _parameter_backed_store()
    store.insert(b"a")
    store.insert(b"b")
    old = store._basis._parameters["W"]
    owner = _Owner(old)
    optimizer = torch.optim.Adam([old], lr=1e-2)

    # Materialize Adam state, then snapshot the post-step learned prefix.
    (old.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    learned_prefix = old.detach().clone()
    exp_avg = optimizer.state[old]["exp_avg"].detach().clone()
    exp_avg_sq = optimizer.state[old]["exp_avg_sq"].detach().clone()

    assert store.observe_chunk(b"ab") is None
    result = store.flush_pending_promotions(
        optimizer=optimizer, owner_space=owner)

    new = store._basis._parameters["W"]
    assert result == {
        "inserted": 1,
        "grew": True,
        "old_capacity": 2,
        "new_capacity": 4,
        "optimizer_groups": 1,
    }
    assert new is not old
    assert owner.params == [new]
    assert optimizer.param_groups[0]["params"] == [new]
    assert old not in optimizer.state
    assert torch.equal(new[:2].detach(), learned_prefix)
    assert store.get_id(b"ab") == 2
    assert store.spell_out(b"ab") == [2]

    state = optimizer.state[new]
    assert tuple(state["exp_avg"].shape) == (4, 3)
    assert tuple(state["exp_avg_sq"].shape) == (4, 3)
    assert torch.equal(state["exp_avg"][:2], exp_avg)
    assert torch.equal(state["exp_avg_sq"][:2], exp_avg_sq)
    assert torch.count_nonzero(state["exp_avg"][2:]).item() == 0
    assert torch.count_nonzero(state["exp_avg_sq"][2:]).item() == 0


def test_eager_atomic_byte_growth_migrates_owner_and_optimizer():
    """A byte needed by this batch grows before spell-out/W gathering."""
    store = _parameter_backed_store()
    store.insert(b"a")
    store.insert(b"b")
    old = store._basis._parameters["W"]
    owner = _Owner(old)
    optimizer = torch.optim.Adam([old], lr=1e-2)

    (old.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    learned_prefix = old.detach().clone()
    old_avg = optimizer.state[old]["exp_avg"].detach().clone()

    result = store.ensure_atomic_bytes(
        [b"a", b"c", b"c"], optimizer=optimizer, owner_space=owner)

    new = store._basis._parameters["W"]
    assert result == {
        "inserted": 1,
        "grew": True,
        "old_capacity": 2,
        "new_capacity": 4,
        "optimizer_groups": 1,
    }
    assert new is not old
    assert owner.params == [new]
    assert optimizer.param_groups[0]["params"] == [new]
    assert old not in optimizer.state
    assert torch.equal(new[:2].detach(), learned_prefix)
    assert torch.equal(optimizer.state[new]["exp_avg"][:2], old_avg)
    assert torch.count_nonzero(
        optimizer.state[new]["exp_avg"][2:]).item() == 0
    assert store.spell_out(b"c") == [store.get_id(b"c")]


def test_eager_atomic_byte_exhaustion_changes_nothing():
    store = _parameter_backed_store(initial=2, maximum=2)
    store.insert(b"a")
    store.insert(b"b")
    old = store._basis._parameters["W"]
    before = old.detach().clone()
    before_hash = dict(store.hash_map)
    before_inverse = list(store.inverse_table)

    with pytest.raises(
            RuntimeError, match=r"capacity exhausted.*No percept/trie row"):
        store.ensure_atomic_bytes([b"c"], owner_space=_Owner(old))

    assert store._basis._parameters["W"] is old
    assert torch.equal(old.detach(), before)
    assert store.hash_map == before_hash
    assert store.inverse_table == before_inverse


def test_percept_master_prefix_is_byte_exact_not_clamped_view(monkeypatch):
    import Spaces

    store = _parameter_backed_store()
    store.insert(b"a")
    store.insert(b"b")
    basis = store._basis
    basis.is_percept_store = True
    monkeypatch.setattr(Spaces, "meronomy_enabled", lambda: True)
    old = basis._parameters["W"]
    with torch.no_grad():
        old.copy_(torch.tensor([
            [-2.0, 0.25, 3.0],
            [4.0, -5.0, 0.75],
        ]))
    raw_prefix = old.detach().clone()
    assert not torch.equal(basis.getW().detach(), raw_prefix)
    owner = _Owner(old)
    store._queue_promotion(b"ab", torch.zeros(3))

    store.flush_pending_promotions(owner_space=owner)
    new = basis._parameters["W"]
    assert torch.equal(new[:2].detach(), raw_prefix)


def test_boundary_growth_rejects_missing_optimizer_owner_atomically():
    store = _parameter_backed_store()
    store.insert(b"a")
    store.insert(b"b")
    old = store._basis._parameters["W"]
    owner = _Owner(old)
    unrelated = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([unrelated], lr=1e-2)
    store._queue_promotion(b"ab", torch.zeros(3))
    before = old.detach().clone()

    with pytest.raises(RuntimeError, match=r"exactly one optimizer.*found 0"):
        store.flush_pending_promotions(
            optimizer=optimizer, owner_space=owner)
    assert store._basis._parameters["W"] is old
    assert owner.params == [old]
    assert torch.equal(old.detach(), before)
    assert b"ab" not in store
    assert store.pending_promotions == 1


def test_max_capacity_exhaustion_is_atomic_and_actionable():
    store = _parameter_backed_store(initial=2, maximum=2)
    store.insert(b"a")
    store.insert(b"b")
    old = store._basis._parameters["W"]
    before = old.detach().clone()
    before_hash = dict(store.hash_map)
    before_inverse = list(store.inverse_table)
    before_trie = store.radix_trie.serialize()

    with pytest.raises(
            RuntimeError, match=r"capacity exhausted.*No percept/trie row"):
        store.observe_chunk(b"ab")

    assert store._basis._parameters["W"] is old
    assert torch.equal(old.detach(), before)
    assert store.hash_map == before_hash
    assert store.inverse_table == before_inverse
    assert store.radix_trie.serialize() == before_trie
    assert store.pending_promotions == 0
    assert b"ab" not in store


def test_partspace_maxvectors_validation_and_omitted_default(tmp_path):
    import xml.etree.ElementTree as ET

    import Language
    import Models
    from Spaces import PartSpace
    from util import TheXMLConfig, init_config

    source = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
    tree = ET.parse(source)
    part = tree.getroot().find("PartSpace")
    assert part is not None
    maximum = part.find("maxVectors")
    assert maximum is not None
    part.remove(maximum)
    config = tmp_path / "no_max.xml"
    tree.write(config, encoding="utf-8", xml_declaration=True)
    defaults = os.path.join(_PROJECT, "data", "model.xml")
    init_config(path=str(config), defaults_path=defaults)
    part_cfg = TheXMLConfig._data["PartSpace"]
    assert "maxVectors" not in part_cfg

    # Omitted maxVectors resolves to the configured physical nVectors.
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = Models.BasicModel.from_config(str(config))
    assert model.perceptualSpace.maxVectors == 8
    assert model.perceptualSpace.percept_store.max_capacity == 8

    # A smaller logical cap fails before any Parameter is allocated.
    init_config(path=str(config), defaults_path=defaults)
    TheXMLConfig._data["PartSpace"]["maxVectors"] = 7
    with pytest.raises(
            ValueError, match=r"maxVectors must be greater than or equal"):
        PartSpace([8, 16], [8, 16], [8, 16], model_type="numeric")
    init_config(path=str(config), defaults_path=defaults)


def test_partspace_byte_fallback_has_one_optimizer_owner():
    """The registered radix fallback codebook participates in training."""
    import Language
    import Models
    from util import init_config

    config = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
    defaults = os.path.join(_PROJECT, "data", "model.xml")
    init_config(path=config, defaults_path=defaults)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = Models.BasicModel.from_config(config)

    fallback = model.perceptualSpace.percept_store.byte_fallback.byte_codebook
    assert fallback.requires_grad
    optimizer = model.getOptimizer(lr=1e-3)
    occurrences = sum(
        candidate is fallback
        for group in optimizer.param_groups
        for candidate in group["params"]
    )
    assert occurrences == 1


def test_smaller_checkpoint_table_prefix_loads_into_configured_initial_rows():
    import Language
    import Models
    from util import init_config

    config = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
    defaults = os.path.join(_PROJECT, "data", "model.xml")
    init_config(path=config, defaults_path=defaults)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = Models.BasicModel.from_config(config)

    key = "perceptualSpace._owned_bases.what.W"
    live_state = dict(model.state_dict())
    live = live_state[key].detach().clone()
    saved_prefix = torch.arange(
        4 * live.shape[1], dtype=live.dtype).reshape(4, live.shape[1])
    state = {key: saved_prefix.clone()}

    assert model._expand_partspace_codebook_checkpoint_state(
        state, live_state) == 1
    assert tuple(state[key].shape) == tuple(live.shape)
    assert state[key].data_ptr() == live_state[key].data_ptr()
    assert torch.equal(state[key][:4], saved_prefix)
    assert torch.equal(state[key][4:], live[4:])


def test_grown_checkpoint_restores_owner_and_optimizer_by_name(
        tmp_path, monkeypatch):
    """A saved physical table may exceed the XML's initial allocation."""
    import xml.etree.ElementTree as ET

    import Language
    import Models
    from checkpoint_migrations import OPTIMIZER_PARAM_NAMES_KEY

    monkeypatch.setenv("BASIC_AUTOLOAD", "0")
    source = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
    tree = ET.parse(source)
    part = tree.getroot().find("PartSpace")
    assert part is not None
    maximum = part.find("maxVectors")
    if maximum is None:
        maximum = ET.SubElement(part, "maxVectors")
    maximum.text = "16"
    config = tmp_path / "grow.xml"
    tree.write(config, encoding="utf-8", xml_declaration=True)

    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source_model, _ = Models.BasicModel.from_config(str(config))
    store = source_model.perceptualSpace.percept_store
    for byte in b"abcdefgh":
        store.insert(bytes([byte]))
    old_w = source_model.perceptualSpace.subspace.what._parameters["W"]
    optimizer = source_model.getOptimizer(lr=1e-2)
    (old_w.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    store.promotion_threshold = 1
    assert store.observe_chunk(b"ab") is None
    source_model._flush_partspace_promotions(optimizer)
    grown_w = source_model.perceptualSpace.subspace.what._parameters["W"]
    assert tuple(grown_w.shape) == (16, grown_w.shape[1])
    source_model._optimizer = optimizer

    checkpoint = tmp_path / "grown.ckpt"
    source_model.save_weights(str(checkpoint))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = saved[OPTIMIZER_PARAM_NAMES_KEY]
    entries = [
        entry
        for leaf in manifest["leaves"]
        for group in leaf["param_groups"]
        for entry in group
        if entry["name"] == "perceptualSpace._owned_bases.what.W"
    ]
    assert len(entries) == 1
    assert entries[0]["shape"] == list(grown_w.shape)

    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored, _ = Models.BasicModel.from_config(str(config))
    initial_w = restored.perceptualSpace.subspace.what._parameters["W"]
    assert initial_w.shape[0] == 8
    assert restored.load_weights(str(checkpoint), require_match=True)

    part_space = restored.perceptualSpace
    restored_w = part_space.subspace.what._parameters["W"]
    assert restored_w is not initial_w
    assert restored_w.shape[0] == 16
    assert part_space.nVectors == 16
    assert part_space.subspace.event.nVectors == 16
    assert int(part_space.spaceShape[0]) == 16
    assert sum(parameter is restored_w for parameter in part_space.params) == 1
    assert int(part_space.subspace.what.ramsification[8, 0]) \
        == int(part_space.subspace.what.FOLD_SIGMA)

    restored_optimizer = restored.getOptimizer(lr=1e-2)
    assert sum(
        parameter is restored_w
        for group in restored_optimizer.param_groups
        for parameter in group["params"]
    ) == 1
    assert restored_w in restored_optimizer.state
    assert tuple(restored_optimizer.state[restored_w]["exp_avg"].shape) \
        == tuple(restored_w.shape)
