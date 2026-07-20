"""Safe capacity migration for the aligned ConceptualSpace inventory."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import pytest
import torch
import torch.nn as nn

import Models
from Layers import ConceptAllocator
from Optimizer import Adam
from Spaces import Codebook, ConceptualSpace
from util import init_config


_CONFIG = _ROOT / "data" / "MM_xor_fixture.xml"
_DEFAULTS = _ROOT / "data" / "model.xml"


class _ConceptHost(nn.Module):
    def __init__(self, codebook=None, allocator=None):
        super().__init__()
        if codebook is not None:
            # Match the real ownership shape: the layer list is canonical and
            # similarity_codebook is a registered compatibility alias.
            self.layers = nn.ModuleList([codebook])
            self.similarity_codebook = codebook
        if allocator is not None:
            object.__setattr__(self, "_concept_allocator", allocator)


class _WholeHost(nn.Module):
    def __init__(self, codebook):
        super().__init__()
        self.what = codebook
        self.analysis_store = codebook


def _codebook(rows, dim=3):
    cb = Codebook()
    cb.create(1, rows, dim, customVQ=True, monotonic=True)
    return cb


def _aligned_model(rows, *, cs=None, css=None, ws=None):
    model = Models.BaseModel()
    model.name = "AlignedCapacityTest"
    model.concept_binding = "aligned"
    model.serial = True
    model.nConceptCodes = int(rows)
    model.nSymbols = int(rows)
    if cs is not None and css is not None:
        raise ValueError("pass cs or css, not both")
    concept_hosts = list(css) if css is not None else (
        [cs] if cs is not None else [])
    if concept_hosts:
        model.conceptualSpaces = nn.ModuleList(concept_hosts)
    if ws is not None:
        model.wholeSpaces = nn.ModuleList([ws])
    return model


def test_smaller_codebook_checkpoint_prefix_expands_all_aliases():
    init_config(path=str(_CONFIG), defaults_path=str(_DEFAULTS))
    old_n, new_n = 4, 8
    cs_cb = _codebook(new_n)
    ws_cb = _codebook(new_n)
    model = _aligned_model(
        new_n, cs=_ConceptHost(cs_cb), ws=_WholeHost(ws_cb))

    model_state = dict(model.state_dict())
    saved = {}
    expected_prefix = {}
    excluded_ws_prefix = {}
    for key, live in model_state.items():
        value = live.detach().clone()
        if (key.startswith("conceptualSpaces.")
                and value.ndim > 0 and int(value.shape[0]) == new_n):
            value = value[:old_n].clone()
            expected_prefix[key] = value.clone()
        elif (key.startswith("wholeSpaces.")
              and value.ndim > 0 and int(value.shape[0]) == new_n):
            # A same-sized legacy/property WS table is deliberately outside
            # concept checkpoint growth, even when its shape happens to match.
            value = value[:old_n].clone()
            excluded_ws_prefix[key] = value.clone()
        saved[key] = value

    count = model._expand_aligned_codebook_checkpoint_state(
        saved, model_state)
    assert count == len(expected_prefix)
    assert expected_prefix
    assert excluded_ws_prefix
    for key, prefix in expected_prefix.items():
        assert tuple(saved[key].shape) == tuple(model_state[key].shape)
        torch.testing.assert_close(saved[key][:old_n], prefix)
        # The newly configured rows keep the construction-time state: random
        # W/embed_avg, ones cluster counts, and derived norm cache.
        torch.testing.assert_close(
            saved[key][old_n:], model_state[key][old_n:])
    for key, prefix in excluded_ws_prefix.items():
        assert tuple(saved[key].shape) == tuple(prefix.shape)
        torch.testing.assert_close(saved[key], prefix)

    # Registered aliases of the same physical Codebook reuse one expanded
    # tensor instead of multiplying peak checkpoint memory.
    alias_groups = {}
    for key, live in model_state.items():
        if key in expected_prefix:
            alias_groups.setdefault(live.data_ptr(), []).append(key)
    for keys in alias_groups.values():
        if len(keys) > 1:
            assert len({saved[key].data_ptr() for key in keys}) == 1


def test_divergent_old_stage_tables_choose_stage_zero_for_shared_dictionary():
    init_config(path=str(_CONFIG), defaults_path=str(_DEFAULTS))
    old_n, new_n = 4, 8
    shared = _codebook(new_n)
    model = _aligned_model(new_n, css=[
        _ConceptHost(shared), _ConceptHost(shared), _ConceptHost(shared),
    ])
    model.wholePropertyBasis = True
    model_state = dict(model.state_dict())
    saved = {}
    canonical_prefix = None
    for key, live in model_state.items():
        value = live.detach().clone()
        if value.ndim > 0 and int(value.shape[0]) == new_n:
            value = value[:old_n].clone()
            if (value.is_floating_point()
                    and key.startswith("conceptualSpaces.1.")):
                value.add_(100)
            elif (value.is_floating_point()
                  and key.startswith("conceptualSpaces.2.")):
                value.add_(200)
        saved[key] = value
        if key == "conceptualSpaces.0.similarity_codebook.W":
            canonical_prefix = value.clone()
    assert canonical_prefix is not None

    with pytest.warns(UserWarning, match="stage-0 similarity dictionary"):
        assert model._canonicalize_shared_concept_checkpoint_state(
            saved, model_state) > 0
    assert model._expand_aligned_codebook_checkpoint_state(
        saved, model_state) > 0

    aliases = [
        key for key in saved
        if key.startswith("conceptualSpaces.") and key.endswith(".W")
        and tuple(model_state[key].shape) == (new_n, 3)
    ]
    assert aliases
    assert len({saved[key].data_ptr() for key in aliases}) == 1
    for key in aliases:
        assert saved[key].data_ptr() == model_state[key].data_ptr()
        torch.testing.assert_close(saved[key][:old_n], canonical_prefix)
    model.load_state_dict(saved, strict=True)
    torch.testing.assert_close(shared.W[:old_n], canonical_prefix)


def test_optimizer_remap_keeps_stage_zero_and_drops_old_stage_aliases():
    from checkpoint_migrations import remap_optimizer_state_by_name

    stage0 = "conceptualSpaces.0.layers.0.W"
    stage1 = "conceptualSpaces.1.layers.0.W"
    shape = [8, 3]
    saved_manifest = {"version": 1, "leaves": [{"param_groups": [[
        {"name": stage0, "shape": shape},
        {"name": stage1, "shape": shape},
    ]]}]}
    live_manifest = {"version": 1, "leaves": [{"param_groups": [[
        {"name": stage0, "shape": shape},
    ]]}]}
    stage0_state = {"exp_avg": torch.ones(shape),
                    "exp_avg_sq": torch.ones(shape)}
    stage1_state = {"exp_avg": torch.full(shape, 2.0),
                    "exp_avg_sq": torch.full(shape, 2.0)}
    result = remap_optimizer_state_by_name(
        {"state": {0: stage0_state, 1: stage1_state},
         "param_groups": [{"params": [0, 1], "lr": 1e-3}]},
        saved_manifest,
        {"state": {}, "param_groups": [{"params": [10], "lr": 1e-3}]},
        live_manifest,
    )
    assert result.state["state"] == {10: stage0_state}
    assert result.diagnostics.restored_parameter_states == 1
    assert result.diagnostics.dropped_saved_states == (stage1,)


def test_manifestless_182_param_layout_restores_shared_stage_zero_only():
    """Reconstruct the pre-manifest 15-minute checkpoint topology.

    That artifact has 182 optimizer parameters but only 47 materialized Adam
    states.  Its radix PS table was registered in ``state_dict`` but not yet in
    the optimizer, while all four CS folds still owned independent dictionary
    Parameters.  The live topology makes the opposite changes: PS is enlisted
    and only stage 0 physically owns the shared CS dictionary.
    """
    from checkpoint_migrations import (
        infer_legacy_optimizer_param_manifest,
        remap_optimizer_state_by_name,
    )

    def manifest(names, shapes):
        return {
            "version": 1,
            "leaves": [{"param_groups": [[
                {"name": name, "shape": list(shapes[name])}
                for name in names
            ]]}],
        }

    ps_old = [f"perceptualSpace.sigma.legacy_{i}" for i in range(12)]
    radix = "perceptualSpace._owned_bases.what.W"

    cs_core = {
        stage: [
            f"conceptualSpaces.{stage}.layers.0.legacy_{i}"
            for i in range(10)
        ]
        for stage in range(4)
    }
    cs_w = {
        stage: f"conceptualSpaces.{stage}.layers.2.W"
        for stage in range(4)
    }
    cs_alias = {
        stage: f"conceptualSpaces.{stage}.similarity_codebook.W"
        for stage in range(4)
    }

    ws_main = {
        stage: f"wholeSpaces.{stage}._owned_bases.what.W"
        for stage in range(4)
    }
    ws_fold = {
        stage: [f"wholeSpaces.{stage}.layers.0.legacy_{i}"
                for i in range(4)]
        for stage in range(4)
    }
    ws_analysis = {
        stage: f"wholeSpaces.{stage}.analysis_store.W"
        for stage in range(4)
    }
    ws_pi = {
        stage: [f"wholeSpaces.{stage}._pi_stack_modules.0.legacy_{i}"
                for i in range(12)]
        for stage in range(4)
    }
    tail = [f"symbolSpace.legacy_{i}" for i in range(54)]

    old_names = list(ps_old)
    live_names = [*ps_old, radix]
    for stage in range(4):
        old_names.extend([*cs_core[stage], cs_w[stage]])
        live_names.extend(cs_core[stage])
        if stage == 0:
            live_names.append(cs_w[stage])
    for stage in range(4):
        old_names.extend([
            ws_main[stage], *ws_fold[stage], ws_analysis[stage],
            *ws_pi[stage],
        ])
        live_names.extend([
            ws_main[stage], *ws_fold[stage], *ws_pi[stage],
        ])
    old_names.extend(tail)
    live_names.extend(tail)
    assert len(old_names) == 182
    assert len(live_names) == 176

    legacy_shapes = {name: (2,) for name in old_names}
    # The newly-enlisted radix tensor is present in the old state_dict even
    # though it was absent from that optimizer parameter group.
    legacy_shapes[radix] = (4, 3)
    for stage in range(4):
        legacy_shapes[cs_w[stage]] = (4, 3)
        legacy_shapes[cs_alias[stage]] = (4, 3)
        legacy_shapes[ws_main[stage]] = (4, 3)
        legacy_shapes[ws_analysis[stage]] = (4, 3)

    live_shapes = {name: legacy_shapes[name] for name in live_names}
    live_shapes[cs_w[0]] = (8, 3)  # configured first-axis capacity growth
    live_manifest = manifest(live_names, live_shapes)

    # Materialize exactly 47 states, including all four historical CS tables,
    # so the remap proves that stage 0 survives and stages 1..3 are discarded.
    selected = {old_names.index(cs_w[stage]) for stage in range(4)}
    for position in range(len(old_names)):
        if len(selected) == 47:
            break
        selected.add(position)
    saved_states = {}
    for position in sorted(selected):
        shape = legacy_shapes[old_names[position]]
        saved_states[position] = {
            "step": torch.tensor(15.0),
            "exp_avg": torch.ones(shape),
            "exp_avg_sq": torch.full(shape, 2.0),
        }
    assert len(saved_states) == 47
    saved_optimizer = {
        "state": saved_states,
        "param_groups": [{"lr": 5e-4, "params": list(range(182))}],
    }

    inferred = infer_legacy_optimizer_param_manifest(
        live_manifest, legacy_shapes, saved_optimizer)
    inferred_names = [
        entry["name"]
        for entry in inferred["leaves"][0]["param_groups"][0]
    ]
    assert inferred_names == old_names

    live_ids = list(range(1000, 1000 + len(live_names)))
    live_optimizer = {
        "state": {},
        "param_groups": [{"lr": 5e-4, "params": live_ids}],
    }
    remapped = remap_optimizer_state_by_name(
        saved_optimizer,
        inferred,
        live_optimizer,
        live_manifest,
        reset_wholespace=True,
    )

    stage0_live_id = live_ids[live_names.index(cs_w[0])]
    assert remapped.state["state"][stage0_live_id] is saved_states[
        old_names.index(cs_w[0])]
    assert live_ids[live_names.index(radix)] not in remapped.state["state"]
    assert remapped.diagnostics.restored_parameter_states == 44
    assert remapped.diagnostics.dropped_saved_states == tuple(
        cs_w[stage] for stage in range(1, 4)
    )


def test_loaded_adam_moments_prefix_copy_and_zero_new_rows():
    old_n, new_n, dim = 3, 7, 2
    parameter = nn.Parameter(torch.arange(new_n * dim).reshape(new_n, dim).float())
    optimizer = Adam([parameter], lr=1e-2)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    state = optimizer.state[parameter]
    exp_avg = state["exp_avg"][:old_n].clone()
    exp_avg_sq = state["exp_avg_sq"][:old_n].clone()
    state["exp_avg"] = exp_avg.clone()
    state["exp_avg_sq"] = exp_avg_sq.clone()

    model = Models.BaseModel()
    model.name = "AdamCapacityTest"
    assert model._normalize_optimizer_state_shapes(optimizer) == 2
    assert any(parameter is p for g in optimizer.param_groups for p in g["params"])
    torch.testing.assert_close(state["exp_avg"][:old_n], exp_avg)
    torch.testing.assert_close(state["exp_avg_sq"][:old_n], exp_avg_sq)
    assert torch.count_nonzero(state["exp_avg"][old_n:]).item() == 0
    assert torch.count_nonzero(state["exp_avg_sq"][old_n:]).item() == 0

    # The first step after resume must consume the padded moments normally.
    parameter[-1].square().sum().backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()


def test_logical_growth_reveals_one_power_of_two_prefix_without_reallocation():
    init_config(path=str(_CONFIG), defaults_path=str(_DEFAULTS))
    n = 8
    cs_cb = _codebook(n)
    ws_cb = _codebook(n)
    model = _aligned_model(
        n, cs=_ConceptHost(cs_cb), ws=_WholeHost(ws_cb))
    codebooks = model._aligned_capacity_codebooks()
    assert codebooks == [cs_cb]
    for cb in codebooks:
        cb.vq.set_active_rows(2)
    ws_cb.vq.set_active_rows(1)
    model._active_inventory_rows = 2
    identities = [
        (id(cb.W), cb.W.data_ptr(), cb.vq.active_mask.data_ptr())
        for cb in codebooks
    ]

    assert model._ensure_aligned_active_rows(3) == 4
    assert [model._active_prefix_rows(cb.vq) for cb in codebooks] == [4]
    assert model._active_prefix_rows(ws_cb.vq) == 1
    assert identities == [
        (id(cb.W), cb.W.data_ptr(), cb.vq.active_mask.data_ptr())
        for cb in codebooks
    ]


def test_logical_growth_detects_prefix_drift_before_mutating_any_mask():
    init_config(path=str(_CONFIG), defaults_path=str(_DEFAULTS))
    n = 8
    cs_cb_0 = _codebook(n)
    cs_cb_1 = _codebook(n)
    model = _aligned_model(n, css=[
        _ConceptHost(cs_cb_0), _ConceptHost(cs_cb_1)])
    codebooks = model._aligned_capacity_codebooks()
    codebooks[0].vq.set_active_rows(2)
    codebooks[1].vq.set_active_rows(4)
    model._active_inventory_rows = 2

    before = [cb.vq.active_mask.clone() for cb in codebooks]
    with pytest.raises(RuntimeError, match="mismatch before growth"):
        model._ensure_aligned_active_rows(3)
    assert all(torch.equal(cb.vq.active_mask, old)
               for cb, old in zip(codebooks, before))


def test_resync_covers_restored_concept_allocator_next_id():
    init_config(path=str(_CONFIG), defaults_path=str(_DEFAULTS))
    n = 8
    allocator = ConceptAllocator(
        layer_sizer=lambda _order: (n + 1, n, None, None),
        order_cap=lambda: 0,
    )
    allocator.next_id = 5
    cs_cb = _codebook(n)
    cs = _ConceptHost(cs_cb, allocator=allocator)
    model = _aligned_model(n, cs=cs)
    for cb in model._aligned_capacity_codebooks():
        cb.vq.set_active_rows(2)
    model._active_inventory_rows = 2

    assert model._resync_aligned_active_codebooks() == 8
    assert all(
        model._active_prefix_rows(cb.vq) == 8
        for cb in model._aligned_capacity_codebooks())


def test_concept_id_capacity_failure_is_atomic_and_activates_before_mint():
    n = 4
    allocator = ConceptAllocator()
    calls = []
    owner = types.SimpleNamespace(
        nVectors=n,
        _concept_allocator=allocator,
        _model=types.SimpleNamespace(
            _ensure_aligned_active_rows=lambda required: calls.append(required)),
    )
    assert [ConceptualSpace.new_concept(owner) for _ in range(3)] == [1, 2, 3]
    assert calls == [2, 3, 4]
    before = allocator.next_id
    with pytest.raises(RuntimeError, match="inventory exhausted.*No concept"):
        ConceptualSpace.new_concept(owner)
    assert allocator.next_id == before


def test_legacy_mixing_relation_ids_are_not_capped_by_dense_cs_rows():
    """A legacy mixing relation table is sparse, not a VQ address space."""
    n = 4
    allocator = ConceptAllocator()
    owner = types.SimpleNamespace(
        nVectors=n,
        _concept_binding="mixing",
        _concept_allocator=allocator,
    )

    assert [ConceptualSpace.new_concept(owner) for _ in range(n + 2)] == [
        1, 2, 3, 4, 5, 6]
    assert allocator.next_id == n + 3


def _allocator_blob(old_n, *, row_next=None):
    return {
        "next_id": 8,
        "placement": {7: 0},
        "raised": set(),
        "singletons": set(),
        "retired": set(),
        "identity": {},
        "relate_idx": {},
        "chain_idx": {},
        "word_obj_meta": {},
        "joint": {},
        "layers": {
            0: {
                "nInput": old_n + 1,
                "nOutput": old_n,
                "constituents": {7: [("whole", "everything")]},
                "tensor_rows": {("shared", 7): 2},
                "row_next": {0: 3} if row_next is None else row_next,
                "rows": [2],
                # old_n is the old trailing EVERYTHING bias column.
                "cols": [old_n],
                "init_vals": [0.75],
                "values": torch.tensor([0.625]),
                "values_requires_grad": True,
            }
        },
    }


def test_allocator_square_growth_moves_only_the_everything_bias():
    old_n, new_n = 4, 8
    allocator = ConceptAllocator(
        layer_sizer=lambda _order: (new_n + 1, new_n, None, None),
        order_cap=lambda: 0,
    )
    cs = _ConceptHost(allocator=allocator)
    model = _aligned_model(new_n, cs=cs)
    model._restore_allocator_extras(cs, _allocator_blob(old_n))

    layer = allocator.layer()
    assert (layer.nOutput, layer.nInput) == (new_n, new_n + 1)
    assert layer._rows == [2]
    assert layer._cols == [new_n]
    assert layer._index == {(2, new_n): 0}
    assert layer._tensor_rows[("shared", 7)] == 2
    torch.testing.assert_close(layer.values.detach(), torch.tensor([0.625]))


def test_allocator_partitioned_growth_fails_instead_of_rekeying_rows():
    old_n, new_n = 4, 8
    allocator = ConceptAllocator(
        layer_sizer=lambda _order: (new_n + 1, new_n, None, None),
        order_cap=lambda: 0,
    )
    cs = _ConceptHost(allocator=allocator)
    model = _aligned_model(new_n, cs=cs)
    with pytest.raises(ValueError, match="partitioned row map"):
        model._restore_allocator_extras(
            cs, _allocator_blob(old_n, row_next={old_n // 2: 1}))
