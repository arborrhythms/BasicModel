"""Boundary-safe logical admission into fixed PartSpace storage (9b §4e).

The former resizing/optimizer-migration assertions are retired. These probes
keep atomicity, existing values, optimizer ownership and checkpoint coverage.
"""
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


def _parameter_backed_store(capacity=4):
    from Layers import RadixLayer
    from Spaces import Codebook
    basis = Codebook()
    basis.create(1, capacity, 3, customVQ=False)
    basis.setW(torch.nn.Parameter(basis.W.detach().clone()))
    return RadixLayer(3, initial_cap=capacity, basis=basis,
                      promotion_threshold=1, promotion_min_length=2)


class _Owner:
    def __init__(self, parameter):
        self.params = [parameter]
        self.nVectors = len(parameter)


def test_deferred_promotion_does_not_mutate_the_live_parameter():
    store = _parameter_backed_store()
    store.ensure_atomic_bytes([b'a', b'b'])
    parameter = store._basis.W
    before = parameter.detach().clone()
    store._queue_promotion(b'ab', torch.zeros(3, requires_grad=True))
    assert store.spell_out(b'ab') == [0, 1]
    assert store._basis.W is parameter
    torch.testing.assert_close(parameter, before)
    assert b'ab' not in store and store.pending_promotions == 1
    pending = store._pending_promotions[b'ab']
    assert pending.device.type == 'cpu' and not pending.requires_grad


@pytest.mark.parametrize('deferred', [False, True])
def test_admission_preserves_existing_values_and_adam_state(deferred):
    store = _parameter_backed_store()
    store.ensure_atomic_bytes([b'a', b'b'])
    parameter = store._basis.W
    owner = _Owner(parameter)
    optimizer = torch.optim.Adam([parameter], lr=.01)
    parameter[:2].square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    prefix = parameter[:2].detach().clone()
    moments = {key: value.clone() for key, value in optimizer.state[parameter].items()}
    if deferred:
        store._queue_promotion(b'ab', torch.zeros(3))
        result = store.flush_pending_promotions(optimizer=optimizer, owner_space=owner)
    else:
        result = store.ensure_atomic_bytes([b'a', b'c', b'c'], optimizer=optimizer, owner_space=owner)
    assert result == dict(inserted=1, grew=False, old_capacity=4,
                          new_capacity=4, optimizer_groups=0)
    assert store._basis.W is parameter and owner.params == [parameter]
    assert optimizer.param_groups[0]['params'] == [parameter]
    assert len(optimizer.state) == 1
    assert torch.equal(parameter[:2].detach(), prefix)
    for key, value in moments.items():
        assert torch.equal(optimizer.state[parameter][key], value)


@pytest.mark.parametrize('deferred', [False, True])
def test_capacity_exhaustion_is_atomic_and_actionable(deferred):
    store = _parameter_backed_store(2)
    store.ensure_atomic_bytes([b'a', b'b'])
    parameter = store._basis.W
    before = parameter.detach().clone()
    before_hash = dict(store.hash_map)
    before_inverse = list(store.inverse_table)
    before_trie = store.radix_trie.serialize()
    with pytest.raises(RuntimeError, match='capacity exhausted.*nVectors'):
        if deferred:
            store._queue_promotion(b'ab', torch.zeros(3))
        else:
            store.ensure_atomic_bytes([b'c'])
    assert store._basis.W is parameter and torch.equal(parameter, before)
    assert store.hash_map == before_hash and store.inverse_table == before_inverse
    assert store.radix_trie.serialize() == before_trie
    assert store.pending_promotions == 0


def test_percept_master_prefix_is_byte_exact_not_clamped_view(monkeypatch):
    import Spaces
    store = _parameter_backed_store()
    store.ensure_atomic_bytes([b'a', b'b'])
    basis = store._basis
    basis.is_percept_store = True
    monkeypatch.setattr(Spaces, 'meronomy_enabled', lambda: True)
    parameter = basis._parameters['W']
    with torch.no_grad():
        parameter[:2].copy_(torch.tensor([[-2., .25, 3.], [4., -5., .75]]))
    raw = parameter[:2].detach().clone()
    assert not torch.equal(basis.getW()[:2].detach(), raw)
    store._queue_promotion(b'ab', torch.zeros(3))
    store.flush_pending_promotions()
    assert basis._parameters['W'] is parameter
    assert torch.equal(parameter[:2].detach(), raw)


def test_admission_cannot_adopt_an_unrelated_optimizer():
    store = _parameter_backed_store()
    store.ensure_atomic_bytes([b'a', b'b'])
    parameter = store._basis.W
    unrelated = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([unrelated], lr=.01)
    store._queue_promotion(b'ab', torch.zeros(3))
    store.flush_pending_promotions(optimizer=optimizer)
    assert store._basis.W is parameter
    assert optimizer.param_groups[0]['params'] == [unrelated]
    assert len(optimizer.state) == 0
    assert store.get_id(b'ab') == 2


def test_partspace_nvectors_is_the_only_physical_capacity_knob():
    import xml.etree.ElementTree as ET
    import Language
    import Models
    config = os.path.join(_PROJECT, 'data', 'MM_xor_fixture.xml')
    root = ET.parse(config).getroot()
    assert root.find('PartSpace/maxVectors') is None
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model, _ = Models.BasicModel.from_config(config)
    capacity = int(root.findtext('PartSpace/nVectors'))
    assert model.perceptualSpace.percept_store.capacity == capacity
    assert model.perceptualSpace.subspace.what.nVectors == capacity
    assert not hasattr(model.perceptualSpace.percept_store, '_grow_to')


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


def test_checkpoint_restores_fixed_owner_and_optimizer_by_name(
        tmp_path, monkeypatch):
    """A saved logical inventory reloads inside the declared physical allocation."""
    import xml.etree.ElementTree as ET

    import Language
    import Models
    from checkpoint_migrations import OPTIMIZER_PARAM_NAMES_KEY

    monkeypatch.setenv("BASIC_AUTOLOAD", "0")
    source = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
    tree = ET.parse(source)
    part = tree.getroot().find("PartSpace")
    assert part is not None
    part.find("nVectors").text = "16"
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
    store._queue_promotion(b"ab", torch.zeros(store.dim))
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
    assert initial_w.shape[0] == 16
    assert restored.load_weights(str(checkpoint), require_match=True)

    part_space = restored.perceptualSpace
    restored_w = part_space.subspace.what._parameters["W"]
    assert restored_w is initial_w
    assert restored_w.shape[0] == 16
    assert part_space.nVectors == 16
    assert part_space.subspace.event.nVectors == 16
    assert int(part_space.spaceShape[0]) == 16
    assert sum(parameter is restored_w for parameter in part_space.params) == 1
    assert store.get_id(b"ab") == part_space.percept_store.get_id(b"ab")

    restored_optimizer = restored.getOptimizer(lr=1e-2)
    assert sum(
        parameter is restored_w
        for group in restored_optimizer.param_groups
        for parameter in group["params"]
    ) == 1
    assert restored_w in restored_optimizer.state
    assert tuple(restored_optimizer.state[restored_w]["exp_avg"].shape) \
        == tuple(restored_w.shape)
