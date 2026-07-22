"""Aligned concept formation across ordered PS and WS fold ladders."""

from __future__ import annotations

import copy
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import (  # noqa: E402
    Codebook,
    ConceptualSpace,
    SubSpace,
)
from test_serial_object_meta import _serial_model_and_batch  # noqa: E402
from test_abstraction_order_canonical import _whole_space  # noqa: E402


class _Sub:
    """Small SubSpace-compatible carrier for binder-only tests."""

    def __init__(self, event):
        self.event = event

    def is_empty(self):
        return self.event is None

    def materialize(self):
        return self.event

    def set_event(self, event):
        self.event = event

    def copy_context(self, _other):
        return None


def test_basicmodel_selects_three_folds_per_tower():
    root = ET.parse(os.path.join(_PROJECT, "data", "BasicModel.xml")).getroot()
    arch = root.find("architecture")
    assert arch is not None
    assert arch.findtext("conceptBinding") == "aligned"
    assert int(arch.findtext("subsymbolicOrder")) - 1 == 3
    assert arch.findtext("sentenceProtocol") == "false"


def test_aligned_binder_rejects_native_width_before_codebook_activation():
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpace
    part = torch.rand(1, 1, 2)
    whole = torch.rand(1, 8, 4)
    out = SubSpace((1, 1), (1, 1), 1, 1)
    out.set_event(torch.zeros(1, 1, 4))

    with pytest.raises(RuntimeError, match="conceptual-width part stream"):
        cs.bind_aligned_streams(_Sub(part), _Sub(whole), out)


def _event_subspace(event):
    """Build a real context-carrying SubSpace around a test event."""
    width = int(event.shape[-1])
    sub = SubSpace((int(event.shape[1]), width),
                   (int(event.shape[1]), width), width, width)
    sub.set_event(event)
    return sub


def test_aligned_serial_forward_rejects_native_width_before_legacy_adapter():
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpace
    object.__setattr__(cs, "_concept_binding", "aligned")
    expected = int(cs.concept_dim)
    native = _event_subspace(torch.zeros(1, 1, expected - 1))

    with pytest.raises(RuntimeError, match="predecoded concept-width primary"):
        cs.forward(native)


def test_aligned_serial_forward_rejects_wrong_symbolic_width():
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpace
    object.__setattr__(cs, "_concept_binding", "aligned")
    expected = int(cs.concept_dim)
    primary = _event_subspace(torch.zeros(1, 1, expected))
    symbolic = _event_subspace(torch.zeros(1, 1, expected - 1))

    with pytest.raises(RuntimeError, match="concept-width symbolic"):
        cs.forward(primary, symbolic)


def test_aligned_binder_uses_all_six_sources_without_location_mixing():
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpace
    # Actual serial geometry: the current word's PS carrier has one location,
    # while WS exposes the eight-location field. Aligned binding pads PS on
    # the location axis; it never reinterprets feature coordinates as slots.
    part_values = [1.0, 2.0, 3.0]
    whole_values = [4.0, 5.0, 6.0]
    part_tensors = [
        torch.full((1, 1, 4), value, requires_grad=True)
        for value in part_values
    ]
    whole_tensors = [
        torch.full((1, 8, 4), value, requires_grad=True)
        for value in whole_values
    ]
    out = SubSpace((1, 1), (1, 1), 1, 1)
    out.set_event(torch.zeros(1, 1, 4))
    carrier = cs.bind_fold_streams(
        [_Sub(x) for x in part_tensors],
        [_Sub(x) for x in whole_tensors],
        out,
        part_passes=(0, 1, 2),
        whole_passes=(0, 1, 2),
    )

    assert carrier.shape == (1, 6, 8, 4)
    assert torch.equal(out._concept_orders, torch.full((1, 8), 3))
    # Location 0 receives all six sources; later locations receive WS and
    # explicit zero PS padding, never coordinate-regrouped PS fragments.
    assert torch.allclose(out.materialize()[:, 0], torch.full((1, 4), 3.5))
    assert torch.allclose(out.materialize()[:, 1], torch.full((1, 4), 2.5))
    out.materialize().sum().backward()
    for source in part_tensors + whole_tensors:
        assert source.grad is not None
        assert bool((source.grad != 0).all())

    support = out._fold_support
    assert support["source_count"] == 6
    assert support["part_folds"][-1]["path"] == [
        [0, Codebook.FOLD_SIGMA],
        [1, Codebook.FOLD_SIGMA],
        [2, Codebook.FOLD_SIGMA],
    ]
    assert support["whole_folds"][-1]["path"] == [
        [0, Codebook.FOLD_PI],
        [1, Codebook.FOLD_PI],
        [2, Codebook.FOLD_PI],
    ]


def test_aligned_provenance_preserves_prior_bearing_forward_event():
    """Post-forward fold attachment must not erase CS's consumed prior."""
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpace
    B, N, D = 1, int(cs.outputShape[0]), int(cs.outputShape[1])
    source_mean = torch.full((B, N, D), 2.0)
    prior_bearing = torch.full((B, N, D), 11.0)

    # Mirror ConceptualSpace.forward: the returned carrier is CS.subspace,
    # _subspaceForWS aliases it, and the persistent PS feedback carrier has
    # already received the same prior-bearing event.
    out = cs.subspace
    out.set_event(prior_bearing.clone())
    cs._subspaceForPS.set_event(prior_bearing.clone())
    object.__setattr__(cs, "_subspaceForWS", out)
    carrier = cs.bind_fold_streams(
        [_Sub(source_mean)],
        [_Sub(source_mean)],
        out,
        part_passes=(0,),
        whole_passes=(0,),
        preserve_target_event=True,
    )

    assert torch.equal(carrier.mean(dim=1), source_mean)
    assert torch.equal(out.materialize(), prior_bearing)
    assert torch.equal(cs.subspace.materialize(), prior_bearing)
    assert torch.equal(cs._subspaceForPS.materialize(), prior_bearing)
    assert cs._subspaceForWS is out
    assert out._fold_support["source_count"] == 2


def test_mini_basicmodel_ps128_ws128_cs1024_runs_forward_backward(
        tmp_path, monkeypatch):
    """Exercise the real construction/stem with production boundary widths."""
    import Language
    import Models
    from util import init_config, init_device

    tree = ET.parse(os.path.join(_PROJECT, "data", "BasicModel.xml"))
    root = tree.getroot()

    def _set(path, value):
        node = root.find(path)
        assert node is not None, path
        node.text = str(value).lower() if isinstance(value, bool) else str(value)

    # Keep the production 128-WHAT native peers and 1024-WHAT concept geometry,
    # but replace the large physical dictionaries and raw byte ceiling with
    # smoke-test capacities.
    _set("./InputSpace/nOutput", 128)
    _set("./PartSpace/nInput", 128)
    _set("./PartSpace/nVectors", 256)
    _set("./PartSpace/maxVectors", 512)
    _set("./ConceptualSpace/nVectors", 128)
    _set("./ConceptualSpace/activeVectors", 32)
    _set("./architecture/training/batchSize", 1)
    _set("./architecture/training/numWorkers", 0)
    _set("./architecture/training/autoload", False)
    _set("./architecture/training/autosave", False)
    _set("./architecture/weightsPath", tmp_path / "unused.ckpt")
    config = tmp_path / "mini_basic_ps128_ws128_cs1024.xml"
    tree.write(config, encoding="unicode")

    init_device("cpu")
    init_config(
        path=str(config),
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
    Language.TheGrammar._configured = False
    monkeypatch.setattr(
        Models.BaseModel, "load_weights", lambda self, *a, **k: False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, _ = Models.BasicModel.from_config(str(config), data=Models.TheData)
    model.train()

    assert model.perceptualSpace.outputShape == [8, 136]
    assert model.wholeSpace.outputShape == [8, 136]
    assert model.conceptualSpace.inputShape == [8, 1032]
    assert model.conceptualSpace.outputShape == [8, 1032]
    assert not hasattr(model.conceptualSpace, "perceptual_lift")
    assert model.conceptualSpace.similarity_codebook.sparse_lookup_grad

    # Stage 0 resolves concepts, while production checkpoint collection and
    # restore use the terminal CS. Their provenance registry must therefore be
    # one shared object, not merely equivalent snapshots.
    registries = [cs._concept_fold_support for cs in model.conceptualSpaces]
    assert all(registry is registries[0] for registry in registries[1:])
    support = ConceptualSpace._ordered_fold_support(
        (0, 1, 2), (0, 1, 2))
    expected = model.conceptualSpaces[0].record_concept_fold_support(
        17, support, 3)
    assert model.conceptualSpace.concept_fold_support(17) == expected
    checkpoint_extras = model._collect_vocab_extras()
    assert checkpoint_extras is not None
    conceptual_blob = copy.deepcopy(
        checkpoint_extras["conceptual_structure"])
    assert conceptual_blob["concept_fold_support"][17] == expected
    registries[0].clear()
    model.conceptualSpace.load_vocab_extras(conceptual_blob)
    assert model.conceptualSpaces[0].concept_fold_support(17) == expected

    result = model.forward(model.inputSpace.prepInput(["alpha beta"]))
    differentiable = [
        value for value in result
        if torch.is_tensor(value) and value.requires_grad]
    assert differentiable, "mini BasicModel forward exposed no gradient path"

    # The last serial word cached a one-location native descriptor. Reversing
    # the complete sentence root must ignore that stale shape and preserve the
    # eight-location PS field recovered from the conceptual seed. Keep this in
    # the live graph: runBatch's D3 objective backpropagates through the same
    # reverse path.
    ps = model.perceptualSpace
    assert ps._pre_reshape_input == (1, 136)
    recon = model._reverse_from_S(model._stm_single_S)
    assert recon.shape == (1, 8, 136)
    differentiable.append(recon)

    loss = sum(value.square().mean() for value in differentiable)
    loss.backward()
    shared_W = model.conceptualSpace.similarity_codebook.W
    assert shared_W.grad is not None
    assert shared_W.grad.layout == torch.sparse_coo
    assert bool(torch.isfinite(shared_W.grad.coalesce().values()).all())

    def _finite_nonzero_grad(param):
        grad = param.grad
        if grad is None:
            return False
        values = grad.coalesce().values() if grad.is_sparse else grad
        return (bool(torch.isfinite(values).all())
                and bool(values.ne(0).any()))

    assert any(
        _finite_nonzero_grad(p)
        for p in model.parameters() if p.requires_grad)


def test_meta_fold_support_roundtrips_with_vocab_extras():
    support = ConceptualSpace._ordered_fold_support(
        (0, 1, 2), (0, 1, 2))
    ws = _whole_space()
    ps_pos = ws.ensure_ps_position(7)
    ws_pos = ws.insert_whole(init_vec=torch.randn(8))
    meta = ws.insert_meta(
        ps_pos, ws_pos, fused_vec=torch.randn(8),
        fold_support=support)
    assert ws.meta_fold_support[meta] == support

    blob = ws.vocab_extras()
    assert blob["meta_fold_support"][meta] == support
    ws2 = _whole_space()
    ws2.load_vocab_extras(blob)
    assert ws2.meta_fold_support[meta] == support


def test_concept_fold_support_roundtrips_with_conceptual_extras():
    support = ConceptualSpace._ordered_fold_support(
        (0, 1, 2), (0, 1, 2))
    model, _ = _serial_model_and_batch()
    cs = model.conceptualSpaces[0]
    expected = cs.record_concept_fold_support(17, support, 3)
    blob = cs.vocab_extras()

    model2, _ = _serial_model_and_batch()
    cs2 = model2.conceptualSpaces[0]
    cs2.load_vocab_extras(blob)
    assert cs2.concept_fold_support(17) == expected
