"""Focused parity contracts for the reusable aligned K=2 word cell."""

from __future__ import annotations

import os
import copy
import sys
import types
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch import nn

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Layers import IntraSentenceLayer, ShortTermMemory  # noqa: E402
import Language  # noqa: E402
import Models  # noqa: E402
from Models import BasicModel  # noqa: E402
from Spaces import ConceptualSpace  # noqa: E402
from util import init_config, init_device  # noqa: E402


_STM_ATTRS = (
    "_buffer",
    "_depth",
    "_orders",
    "_grammar_orders",
    "_concept_rows",
    "_concept_activations",
)


def _concept_space(batch, capacity, dim):
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.stm = ShortTermMemory(
        batch=batch, capacity=capacity, concept_dim=dim)
    cs.intraSentenceLayer = IntraSentenceLayer(
        concept_dim=dim, stm_capacity=capacity, routing_dim=2,
        working_dim=dim, naive=True)
    cs.intra_loss_weight = 1.0
    cs._intra_loss_accum = None
    cs._intra_loss_weight_accum = None
    cs._intra_loss_count = 0
    cs._stm_predicted_idea = None
    cs.symbolSpace = None
    cs.subspace = types.SimpleNamespace(name="test-cs")
    return cs


def _fake_word_body(self, word, p, gate_b_1, out_slot, active_host=True):
    """Small state-complete stand-in for the canonical word body.

    It deliberately reads every p-indexed surface through InputSpace, so the
    test also checks that the eager adapter's two-column swaps preserve global
    word order while the compiled cell sees only local positions 0 and 1.
    """
    del active_host
    cs = self.conceptualSpace
    stm = cs.stm
    idea = word[:, 0, :]
    if self._compiled_word_chunk_replaying:
        cs._stm_predict_then_perceive_serial_fixed(
            idea, row_gate=gate_b_1)
    else:
        cs._stm_predict_then_perceive_serial(
            idea, row_gate=gate_b_1)

    isp = self.inputSpace
    order = isp._ar_word_concept_orders[:, p]
    concept_row = isp._ar_word_concept_rows[:, p]
    activation = gate_b_1.reshape(-1).to(idea.dtype) * 0.25
    stm.push_step_masked(
        idea, gate_b_1,
        orders=order,
        grammar_orders=order + 10,
        concept_row=concept_row,
        concept_activation=activation)
    if not self._compiled_word_chunk_replaying:
        stm._max_depth_host = int(stm._depth.max().item())

    contribution = torch.where(
        gate_b_1, idea, torch.zeros_like(idea))
    out_slot[p] = contribution
    self._per_word_percept_contributions[p] = contribution * 2.0
    return cs.subspace, idea


def _harness(words, gates, *, chunk, predictor_state=None):
    batch, width, dim = words.shape
    model = BasicModel.__new__(BasicModel)
    nn.Module.__init__(model)
    cs = _concept_space(batch, capacity=3, dim=dim)
    if predictor_state is not None:
        cs.intraSentenceLayer.load_state_dict(predictor_state)
    model.conceptualSpace = cs

    part_ids = torch.arange(width).view(1, width, 1).expand(
        batch, -1, -1).clone()
    model.inputSpace = types.SimpleNamespace(
        _ar_embedded_N=words,
        _word_active_mask=gates,
        _ar_word_part_ids=part_ids,
        _ar_word_part_mask=torch.ones_like(part_ids, dtype=torch.bool),
        _ar_word_part_offsets=part_ids.clone(),
        _ar_word_concept_rows=(
            torch.arange(width).view(1, width).expand(batch, -1) + 20),
        _ar_word_concept_orders=(
            torch.arange(width).view(1, width).expand(batch, -1) + 2),
        _word_last_slot_mask=gates.clone(),
        # These sentence-wide surfaces must not be sliced by the adapter.
        _ar_concept_lookup_rows=torch.arange(8).view(1, 8).expand(batch, -1),
        _ar_concept_lookup_atoms=torch.randn(batch, 8, dim),
    )
    model._compiled_word_chunk_active = bool(chunk)
    model._compiled_word_chunk_replaying = False
    model._compiled_word_chunk_step = (
        model._aligned_word_chunk2 if chunk else None)
    model._per_word_body_step = types.MethodType(_fake_word_body, model)
    model._per_word_contributions = [None] * width
    model._per_word_percept_contributions = [None] * width
    return model


def test_k2_adapter_matches_legacy_loop_stm_loss_and_gradients():
    """K=2 replay is parity-exact for all six STM tensors and L_intra."""
    torch.manual_seed(71)
    words = torch.randn(2, 4, 5)
    gates = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])

    legacy = _harness(words, gates, chunk=False)
    predictor_state = legacy.conceptualSpace.intraSentenceLayer.state_dict()
    chunked = _harness(
        words, gates, chunk=True, predictor_state=predictor_state)

    for p in range(words.shape[1]):
        legacy._per_word_body_step(
            words[:, p:p + 1], p, gates[:, p:p + 1],
            legacy._per_word_contributions)

    bank_rows = chunked.inputSpace._ar_concept_lookup_rows
    bank_atoms = chunked.inputSpace._ar_concept_lookup_atoms
    result = chunked._run_aligned_word_chunk_loop(
        chunked._per_word_contributions, words.shape[1])
    assert result is chunked.conceptualSpace.subspace
    assert chunked.inputSpace._ar_concept_lookup_rows is bank_rows
    assert chunked.inputSpace._ar_concept_lookup_atoms is bank_atoms

    for name in _STM_ATTRS:
        torch.testing.assert_close(
            getattr(chunked.conceptualSpace.stm, name),
            getattr(legacy.conceptualSpace.stm, name), rtol=0, atol=0)
    assert (chunked.conceptualSpace.stm._max_depth_host
            == legacy.conceptualSpace.stm._max_depth_host)
    assert chunked.conceptualSpace.stm._max_depth_host > 0
    for got, expected in zip(
            chunked._per_word_contributions,
            legacy._per_word_contributions):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    for got, expected in zip(
            chunked._per_word_percept_contributions,
            legacy._per_word_percept_contributions):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    legacy_loss = legacy.conceptualSpace.consume_intra_loss()
    chunk_loss = chunked.conceptualSpace.consume_intra_loss()
    torch.testing.assert_close(chunk_loss, legacy_loss, rtol=1e-6, atol=1e-7)
    legacy_loss.backward()
    chunk_loss.backward()
    for legacy_param, chunk_param in zip(
            legacy.conceptualSpace.intraSentenceLayer.parameters(),
            chunked.conceptualSpace.intraSentenceLayer.parameters()):
        torch.testing.assert_close(
            chunk_param.grad, legacy_param.grad, rtol=2e-5, atol=2e-6)


def test_chunk_views_keep_one_graph_across_part_and_bucket_widths():
    """K=2 layouts avoid specialization by residual P or outer W."""
    torch.manual_seed(151)
    words = torch.randn(2, 2, 5)
    gates = torch.ones(2, 2, dtype=torch.bool)
    model = _harness(words, gates, chunk=True)
    original_body = model._per_word_body_step

    def part_reading_body(self, word, p, gate, out_slot,
                          active_host=True):
        result = original_body(word, p, gate, out_slot, active_host)
        isp = self.inputSpace
        residual = (
            isp._ar_word_part_ids[:, p, :].to(word.dtype)
            * isp._ar_word_part_mask[:, p, :].to(word.dtype)
            + isp._ar_word_part_offsets[:, p, :].to(word.dtype)
        ).sum(dim=-1, keepdim=True)
        out_slot[p] = out_slot[p] + residual * 1e-6
        self._per_word_percept_contributions[p] = (
            self._per_word_percept_contributions[p] + residual * 1e-6)
        return result

    model._per_word_body_step = types.MethodType(part_reading_body, model)
    graphs = []

    def record_graph(gm, _example_inputs):
        graphs.append(gm)
        return gm.forward

    def install_layout(n_words, part_width):
        batch, _, dim = words.shape
        isp = model.inputSpace
        isp._ar_embedded_N = torch.randn(batch, n_words, dim)
        isp._word_active_mask = torch.ones(
            batch, n_words, dtype=torch.bool)
        parts = torch.arange(part_width).view(1, 1, part_width).expand(
            batch, n_words, part_width).clone()
        isp._ar_word_part_ids = parts
        isp._ar_word_part_mask = torch.ones_like(
            parts, dtype=torch.bool)
        isp._ar_word_part_offsets = parts.clone()
        isp._ar_word_concept_rows = (
            torch.arange(n_words).view(1, n_words).expand(batch, -1) + 20)
        isp._ar_word_concept_orders = (
            torch.arange(n_words).view(1, n_words).expand(batch, -1) + 2)
        isp._word_last_slot_mask = isp._word_active_mask.clone()
        model._per_word_contributions = [None] * n_words
        model._per_word_percept_contributions = [None] * n_words

    try:
        model._compiled_word_chunk_step = torch.compile(
            model._aligned_word_chunk2,
            backend=record_graph,
            fullgraph=True)
        install_layout(2, 3)
        model._run_aligned_word_chunk_loop(
            model._per_word_contributions, 2)
        assert len(graphs) == 1

        install_layout(2, 5)
        model._run_aligned_word_chunk_loop(
            model._per_word_contributions, 2)
        assert len(graphs) == 1

        install_layout(4, 5)
        model._run_aligned_word_chunk_loop(
            model._per_word_contributions, 4)
        assert len(graphs) == 1
    finally:
        torch._dynamo.reset()


def _tiny_canonical_model(tmp_path, monkeypatch):
    """Build the real aligned serial model with 16-coordinate events."""
    tree = ET.parse(_ROOT / "data" / "BasicModel.xml")
    root = tree.getroot()

    def _set(path, value):
        node = root.find(path)
        assert node is not None, path
        node.text = str(value).lower() if isinstance(value, bool) else str(value)

    _set("./InputSpace/nOutput", 128)
    _set("./InputSpace/nDim", 16)
    _set("./PartSpace/nInput", 128)
    _set("./PartSpace/nInputDim", 16)
    _set("./PartSpace/nVectors", 64)
    _set("./PartSpace/maxVectors", 256)
    _set("./PartSpace/nDim", 16)
    _set("./PartSpace/nOutputDim", 16)
    _set("./ConceptualSpace/nInputDim", 16)
    _set("./ConceptualSpace/nVectors", 64)
    _set("./ConceptualSpace/activeVectors", 32)
    _set("./ConceptualSpace/nDim", 16)
    _set("./ConceptualSpace/nOutputDim", 16)
    _set("./WholeSpace/nInputDim", 16)
    _set("./WholeSpace/nDim", 16)
    _set("./WholeSpace/nOutputDim", 16)
    _set("./OutputSpace/nInputDim", 16)
    _set("./architecture/training/batchSize", 2)
    _set("./architecture/training/numWorkers", 0)
    _set("./architecture/training/autoload", False)
    _set("./architecture/training/autosave", False)
    _set("./architecture/weightsPath", tmp_path / "unused.ckpt")
    config = tmp_path / "tiny_chunk_model.xml"
    tree.write(config, encoding="unicode")

    init_device("cpu")
    init_config(
        path=str(config), defaults_path=str(_ROOT / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    monkeypatch.setattr(
        Models.BaseModel, "load_weights", lambda self, *a, **k: False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, _ = Models.BasicModel.from_config(
            str(config), data=Models.TheData)
    model.train()
    return model


def test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks(
        tmp_path, monkeypatch):
    """The real six-fold word cell agrees across a K=2 boundary.

    The prior monolithic compiled loop cannot execute the host chart callback
    embedded behind ``not torch.compiler.is_compiling()``.  Suppress that same
    callback in the eager reference so this test compares the semantics being
    replaced, including a mixed-length batch whose third word crosses the
    first K=2 boundary.
    """
    torch.manual_seed(83)
    legacy = _tiny_canonical_model(tmp_path, monkeypatch)
    chunked = copy.deepcopy(legacy)
    legacy._chart_compose_per_word = lambda: None
    chunked._chart_compose_per_word = lambda: None

    samples = ["alpha beta gamma", "delta"]
    legacy_input = legacy.inputSpace.prepInput(samples)
    chunk_input = chunked.inputSpace.prepInput(samples)

    torch.manual_seed(991)
    legacy_result = legacy.forward(legacy_input)
    chunked._compiled_word_chunk_active = True
    chunked._compiled_word_chunk_step = chunked._aligned_word_chunk2
    torch.manual_seed(991)
    chunk_result = chunked.forward(chunk_input)

    for got, expected in zip(chunk_result, legacy_result):
        if torch.is_tensor(expected):
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)
        else:
            assert got is expected is None
    for name in _STM_ATTRS:
        torch.testing.assert_close(
            getattr(chunked.conceptualSpace.stm, name),
            getattr(legacy.conceptualSpace.stm, name), rtol=0, atol=0)
    assert (chunked.conceptualSpace.stm._max_depth_host
            == legacy.conceptualSpace.stm._max_depth_host)

    legacy_loss = legacy.conceptualSpace.consume_intra_loss()
    chunk_loss = chunked.conceptualSpace.consume_intra_loss()
    torch.testing.assert_close(chunk_loss, legacy_loss, rtol=1e-6, atol=1e-7)
    assert chunked.inputSpace._ar_concept_lookup_rows.shape[1] == 128


def test_no_grad_fallback_retains_eager_stm_depth_semantics(
        tmp_path, monkeypatch):
    """A configured chunk must not alter an intentionally eager eval pass."""
    torch.manual_seed(109)
    legacy = _tiny_canonical_model(tmp_path, monkeypatch)
    chunked = copy.deepcopy(legacy)
    legacy._chart_compose_per_word = lambda: None
    chunked._chart_compose_per_word = lambda: None

    samples = ["alpha beta gamma", "delta"]
    legacy_input = legacy.inputSpace.prepInput(samples)
    chunk_input = chunked.inputSpace.prepInput(samples)
    chunked._compiled_word_chunk_active = True
    chunked._compiled_word_chunk_step = chunked._aligned_word_chunk2

    with torch.no_grad():
        torch.manual_seed(733)
        legacy_result = legacy.forward(legacy_input)
        torch.manual_seed(733)
        chunk_result = chunked.forward(chunk_input)

    for got, expected in zip(chunk_result, legacy_result):
        if torch.is_tensor(expected):
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)
        else:
            assert got is expected is None
    for name in _STM_ATTRS:
        torch.testing.assert_close(
            getattr(chunked.conceptualSpace.stm, name),
            getattr(legacy.conceptualSpace.stm, name), rtol=0, atol=0)
    assert (chunked.conceptualSpace.stm._max_depth_host
            == legacy.conceptualSpace.stm._max_depth_host)
    assert chunked.conceptualSpace.stm._max_depth_host > 0
