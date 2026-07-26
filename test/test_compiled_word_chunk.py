"""Focused parity contracts for the reusable aligned K=2 word cell."""

from __future__ import annotations

import os
import copy
import sys
import types
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
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
import util  # noqa: E402
from bench_train_word_bucket import _profile_peer_legs  # noqa: E402
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


def test_k2_adapter_maps_local_choice_trace_into_sentence_slots():
    """A reused local p=0/1 graph must not overwrite earlier word choices."""
    torch.manual_seed(73)
    words = torch.randn(1, 4, 5)
    gates = torch.ones(1, 4, dtype=torch.bool)
    model = _harness(words, gates, chunk=True)
    trace = Language.ReconstructionStack(batch=1, max_depth=16)
    trace.prepare_choices(
        1, 3 * int(words.shape[1]) + 2, device=words.device,
        unary_rule_ids=(15, 16), binary_rule_ids=(5, 6))
    model.symbolSpace = types.SimpleNamespace(reconstruction_stack=trace)
    model.trace_scale = nn.Parameter(torch.tensor(0.25))
    original_body = model._per_word_body_step

    def traced_body(self, word, p, gate, out_slot, active_host=True):
        result = original_body(word, p, gate, out_slot, active_host)
        active = gate.reshape(-1).bool()
        trace.record_choice(
            3 * p, torch.full_like(active, 5 + p, dtype=torch.long),
            arity=2, mask=active,
            local_structural_loss=(self.trace_scale * float(p + 1)).expand(
                active.shape[0]))
        trace.record_choice(
            3 * p + 2,
            torch.full_like(active, 15 + p, dtype=torch.long),
            arity=1, mask=active,
            local_structural_loss=(self.trace_scale * float(p + 1)).expand(
                active.shape[0]))
        return result

    model._per_word_body_step = types.MethodType(traced_body, model)
    model._compiled_word_chunk_step = torch.compile(
        model._aligned_word_chunk2, backend="eager", fullgraph=True)
    result = model._run_aligned_word_chunk_loop(
        model._per_word_contributions, int(words.shape[1]))
    assert result is model.conceptualSpace.subspace

    ids, arities, mask = trace.choices()
    expected_ids = [5, -1, 15, 6, -1, 16] * 2
    expected_arities = [2, 0, 1, 2, 0, 1] * 2
    assert ids[0, :12].tolist() == expected_ids
    assert arities[0, :12].tolist() == expected_arities
    assert mask[0, :12].tolist() == [value >= 0 for value in expected_ids]
    local = trace.forward_loss()
    assert local is not None
    local.backward()
    torch.testing.assert_close(model.trace_scale.grad, torch.tensor(1.5))


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


def _tiny_canonical_model(
        tmp_path, monkeypatch, *, input_width=128, batch_size=2,
        word_buckets="16,32,64,128,256", forward_grammar_weight=0.0):
    """Build the real aligned serial model with 16-coordinate events."""
    tree = ET.parse(_ROOT / "data" / "BasicModel.xml")
    root = tree.getroot()

    def _set(path, value):
        node = root.find(path)
        assert node is not None, path
        node.text = str(value).lower() if isinstance(value, bool) else str(value)

    _set("./InputSpace/nOutput", input_width)
    _set("./InputSpace/nDim", 16)
    _set("./PartSpace/nInput", input_width)
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
    _set("./architecture/training/batchSize", batch_size)
    _set("./architecture/training/numWorkers", 0)
    _set("./architecture/training/autoload", False)
    _set("./architecture/training/autosave", False)
    _set(
        "./architecture/training/forwardGrammarWeight",
        forward_grammar_weight)
    _set(
        "./architecture/serialWordCapacity",
        max(int(value) for value in str(word_buckets).split(",")))
    _set("./architecture/serialWordBuckets", word_buckets)
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


def _stage_fullgraph_tensor_peer(model, samples):
    """Mirror runBatch's eager sentence staging for a focused graph probe."""
    model._start_spaces_for_forward()
    raw = model.inputSpace.prepInput(samples)
    model._staged_in_sub = model._lex_embed_stem(raw)
    symbol = model.symbolSpace
    if not getattr(symbol, "_per_sentence_initialized", False):
        symbol.soft_reset()
        symbol._per_sentence_initialized = True
    model._stage_reconstruction_teacher()
    slab = model.inputSpace._ar_embedded_N
    model._prepare_reconstruction_choices(
        int(slab.shape[0]), int(slab.shape[1]), slab.device)
    model.conceptualSpace.stm.begin_forward(
        int(slab.shape[0]), device=slab.device, dtype=slab.dtype)
    model._stage_fixed_residual_part_capacity()
    model._stage_intersentence_seed()
    return raw


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
    lookup_rows = chunked.inputSpace._ar_concept_lookup_rows
    capacity = int(chunked.serial_word_capacity)
    assert int(lookup_rows.shape[1]) == 2 * capacity
    words = int(chunked.inputSpace._ar_word_concept_rows.shape[1])
    torch.testing.assert_close(
        lookup_rows[:, :words],
        chunked.inputSpace._ar_word_concept_rows)
    torch.testing.assert_close(
        lookup_rows[:, capacity:capacity + words],
        chunked.inputSpace._ar_word_object_rows)
    assert bool((lookup_rows[:, words:capacity] == -1).all())
    assert bool((lookup_rows[:, capacity + words:] == -1).all())


def test_tensor_peer_while_runs_symbolic_reference_transaction_and_releases_owner_state(
        tmp_path, monkeypatch):
    """The production HOP promotes words, resolves objects, and is reusable."""
    torch.manual_seed(211)
    tensor_loop = _tiny_canonical_model(tmp_path, monkeypatch)
    tensor_loop._chart_compose_per_word = lambda: None
    tensor_loop._tensor_peer_while_eager = True

    samples = ["alpha beta gamma delta", "epsilon zeta"]
    tensor_input = tensor_loop.inputSpace.prepInput(samples)
    torch.manual_seed(991)
    actual = tensor_loop.forward(tensor_input)

    assert tensor_loop.conceptualSpace.CSLang is (
        tensor_loop.conceptualSpace.stm)
    assert not any(
        "CSLang" in key for key in tensor_loop.state_dict())
    assert int(tensor_loop._tensor_peer_trip_count) == 4
    assert tensor_loop._tensor_symbolic_iterations == tensor_loop.symbolicOrder
    assert all(
        value is None or not torch.is_tensor(value)
        or bool(torch.isfinite(value).all())
        for value in actual)

    # CSLang keeps continuous concepts only in CS's fixed eight-slot STM.
    # Its word-aligned handoff to SymbolSpace is the quantized sparse
    # reference: row identity plus one signed activation, never a duplicate
    # [B,W,D_c] concept history.
    stm_buffer = tensor_loop.conceptualSpace.stm._buffer
    assert tuple(stm_buffer.shape[1:]) == (
        tensor_loop.conceptualSpace.stm.capacity,
        tensor_loop.conceptualSpace.stm.concept_dim)
    symbol_activations = (
        tensor_loop.symbolSpace._word_reference_activations)
    symbol_rows = tensor_loop.symbolSpace._word_reference_rows
    active = tensor_loop.inputSpace._word_active_mask
    assert tuple(symbol_activations.shape) == (*active.shape, 1)
    assert tuple(symbol_rows.shape) == tuple(active.shape)
    assert bool((symbol_rows[~active] == -1).all())
    assert bool((symbol_activations[~active] == 0).all())
    assert bool(torch.isfinite(symbol_activations).all())

    actual_intra = tensor_loop.conceptualSpace.consume_intra_loss()
    assert torch.isfinite(actual_intra)
    (actual[0].square().mean()
     + actual[2].square().mean()
     + actual_intra).backward()
    for parameter in tensor_loop.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        values = grad.coalesce().values() if grad.is_sparse else grad
        assert bool(torch.isfinite(values).all())

    # The HOP returns a multi-output view family. Owner-boundary clones must
    # make the normal next-sentence in-place reset legal after backward.
    tensor_loop.symbolSpace.soft_reset()


def test_tensor_peer_ws_stages_distinct_word_local_property_views(
        tmp_path, monkeypatch):
    """WS classifies each IS word once instead of rescanning the sentence."""
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    _stage_fullgraph_tensor_peer(
        model, ["Alpha 7 !", "lower case"])
    weights = model.wholeSpace._staged_word_property_weights
    active = model.inputSpace._word_active_mask

    assert tuple(weights.shape[:2]) == tuple(active.shape)
    assert tuple(weights.shape[2:]) == (
        int(model.wholeSpace.inputShape[0]),
        int(model.wholeSpace.subspace.what.getW().shape[0]))
    # Canonical property rows: letter=0, digit=1, capital=4.
    assert bool(weights[0, 0, :, 0].any())
    assert bool(weights[0, 0, :, 4].any())
    assert not bool(weights[1, 0, :, 4].any())
    assert bool(weights[0, 1, :, 1].any())
    model._tensor_peer_while_eager = True
    assert model._tensor_peer_while_ready(int(active.shape[1]))


def test_peer_leg_profiler_executes_real_ps_ws_and_conceptual_bodies(
        tmp_path, monkeypatch):
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    previous_backend = util.TheCompileBackend
    try:
        util.TheCompileBackend = "eager"
        result = _profile_peer_legs(
            torch, model, torch.device("cpu"),
            ["alpha beta", "gamma"], repeats=2)
    finally:
        torch._dynamo.reset()
        util.TheCompileBackend = previous_backend
    assert result["longest_perceptual_leg"] in ("PS", "WS")
    assert result["ps"]["median_s"] > 0
    assert result["ws"]["median_s"] > 0
    assert result["cs_sub_reduce"]["median_s"] > 0
    assert result["cs_sub"]["median_s"] > 0
    assert result["cs_sym"]["median_s"] > 0
    assert result["cs_lang"]["median_s"] > 0
    assert result["slowest_pipeline_stage"] in (
        "CSSub", "CSSym", "CSLang")
    assert result["serial_to_ideal_pipeline_ratio"] >= 1.0


def test_compiled_sentence_state_is_an_explicit_result_not_a_side_effect():
    """Dead sentence products survive even if compiler side effects do not."""

    class _SentenceHarness(nn.Module):
        _forward_with_compiled_sentence_state = (
            BasicModel._forward_with_compiled_sentence_state)
        _publish_compiled_sentence_state = (
            BasicModel._publish_compiled_sentence_state)

        def __init__(self):
            super().__init__()
            self._spaces_started_for_forward = False
            self._stm_single_S = None
            self._stm_post_depth = None
            self._tensor_peer_trip_count = None

        def _start_spaces_for_forward(self):
            self._spaces_started_for_forward = True

        def _forward_per_stage(
                self, value, in_sub_override=None, *,
                return_sentence_state=False):
            del in_sub_override
            idea = value * 2
            post_depth = torch.ones(
                value.shape[0], dtype=torch.long, device=value.device)
            trip_count = torch.tensor(
                3, dtype=torch.long, device=value.device)
            public = (value, value + 1, value + 2, None)
            return ((*public, idea, post_depth, trip_count)
                    if return_sentence_state else public)

    harness = _SentenceHarness()
    compiled = torch.compile(
        harness._forward_with_compiled_sentence_state,
        backend="eager", fullgraph=True)
    explicit = compiled(torch.randn(2, 4))
    assert len(explicit) == 7

    # Simulate a compiler/runtime that does not replay Python attribute
    # assignments made inside the captured callable.
    harness._stm_single_S = None
    harness._stm_post_depth = None
    harness._tensor_peer_trip_count = None
    public = harness._publish_compiled_sentence_state(explicit)

    assert len(public) == 4
    torch.testing.assert_close(harness._stm_single_S, explicit[4])
    torch.testing.assert_close(harness._stm_post_depth, explicit[5])
    torch.testing.assert_close(harness._tensor_peer_trip_count, explicit[6])


def test_functional_soft_reset_clears_only_selected_owner_rows():
    """Packed sentence boundaries reset CS-owned STM without peer mutation."""
    batch, capacity, dim = 3, 4, 5
    state = (
        torch.arange(batch * capacity * dim, dtype=torch.float32).reshape(
            batch, capacity, dim),
        torch.tensor([4, 3, 2], dtype=torch.long),
        torch.arange(batch * capacity, dtype=torch.long).reshape(
            batch, capacity),
        torch.arange(batch * capacity, dtype=torch.long).reshape(
            batch, capacity) + 20,
        torch.arange(batch * capacity, dtype=torch.long).reshape(
            batch, capacity) + 40,
        torch.arange(batch * capacity, dtype=torch.float32).reshape(
            batch, capacity) / 10.0,
    )
    before = tuple(value.clone() for value in state)
    reset = Models.FunctionalPeerSTM.soft_reset_rows(
        state, torch.tensor([False, True, False]))

    # Functional owner update: the viewed source tensors are untouched.
    for got, expected in zip(state, before):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # Only row 1 is reset; row 0 and row 2 remain bitwise identical.
    for got, expected in zip(reset, before):
        torch.testing.assert_close(
            got[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
    assert torch.count_nonzero(reset[0][1]) == 0
    assert reset[1][1].item() == 0
    for metadata in reset[2:5]:
        assert metadata[1].eq(-1).all()
    assert torch.count_nonzero(reset[5][1]) == 0


@pytest.mark.skipif(
    os.getenv("RUN_SLOW") != "1",
    reason="strict real-model fullgraph trace is ~15s; set RUN_SLOW=1",
)
def test_tensor_peer_complete_forward_is_one_graph_across_runtime_lengths(
        tmp_path, monkeypatch):
    """The real forward/backward reuses one graph as the trip count changes."""
    torch.manual_seed(313)
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model._prewarm_checkpoint_shapes()
    model._compiled_word_loop_fullgraph = True
    raw = _stage_fullgraph_tensor_peer(
        model, ["alpha beta gamma delta", "epsilon zeta"])
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        lambda _unused: model._forward_with_compiled_sentence_state(None),
        backend="eager", fullgraph=True)
    try:
        first = model._publish_compiled_sentence_state(compiled(raw))
        assert int(model._tensor_peer_trip_count) == 4
        (first[0].square().mean() + first[2].square().mean()).backward()
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1

        model.zero_grad(set_to_none=True)
        model.End()
        model.symbolSpace.soft_reset()
        raw = _stage_fullgraph_tensor_peer(
            model,
            ["one two three four five six seven", "one two"])
        second = model._publish_compiled_sentence_state(compiled(raw))
        assert int(model._tensor_peer_trip_count) == 7
        (second[0].square().mean() + second[2].square().mean()).backward()
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert all(
            parameter.grad is None
            or bool(torch.isfinite(
                parameter.grad.coalesce().values()
                if parameter.grad.is_sparse else parameter.grad).all())
            for parameter in model.parameters())
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(
    os.getenv("RUN_MPS_SLOW") != "1"
    or not torch.backends.mps.is_available(),
    reason="requires explicit slow MPS Inductor run",
)
def test_mps_while_loop_backward_accepts_padded_captured_view():
    """Inductor conforms reverse-body grads to Metal's saved-view stride."""
    previous_backend = util.TheCompileBackend
    previous_mode = util.TheCompileMode
    try:
        init_device("mps")
        util.TheCompileBackend = "inductor"
        util.TheCompileMode = "default"
        backing = torch.randn(1056, device="mps", requires_grad=True)
        captured = torch.as_strided(
            backing, (1, 1032), (1056, 1))
        carry = torch.zeros(
            1, 1032, device="mps", requires_grad=True)
        assert captured.stride() == (1056, 1)

        def source(initial, peer):
            index = torch.zeros(
                (), dtype=torch.int64, device=initial.device)

            def cond(tick, value):
                del value
                return tick < 3

            def body(tick, value):
                return tick + 1, value + peer

            _tick, result = torch.while_loop(
                cond, body, (index, initial))
            return result

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = util.compile(source, verbose=False, fullgraph=True)
        result = compiled(carry, captured)
        result.square().mean().backward()
        torch.mps.synchronize()

        assert backing.grad is not None
        assert carry.grad is not None
        assert bool(torch.isfinite(backing.grad).all())
        assert bool(torch.isfinite(carry.grad).all())
        assert int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
    finally:
        torch._dynamo.reset()
        util.TheCompileBackend = previous_backend
        util.TheCompileMode = previous_mode
        init_device("cpu")


@pytest.mark.skipif(
    os.getenv("RUN_MPS_SLOW") != "1"
    or not torch.backends.mps.is_available(),
    reason="requires explicit slow MPS Inductor run",
)
@pytest.mark.parametrize(
    ("first_words", "second_words", "expected_capacity", "input_width"),
    ((10, 14, 16, 128), (40, 50, 64, 512)),
)
def test_tensor_peer_mps_inductor_w16_w64_fullgraph_smoke(
        tmp_path, monkeypatch, first_words, second_words,
        expected_capacity, input_width):
    """MPS lowers real forward/backward once per static capacity."""
    previous_backend = util.TheCompileBackend
    previous_mode = util.TheCompileMode
    model = None
    try:
        torch.manual_seed(617 + expected_capacity)
        model = _tiny_canonical_model(
            tmp_path, monkeypatch, input_width=input_width).to("mps")
        init_device("mps")
        model._prewarm_checkpoint_shapes()
        model._compiled_word_loop_fullgraph = True
        util.TheCompileBackend = "inductor"
        util.TheCompileMode = "default"

        def _sentences(words):
            return [
                " ".join(["a"] * int(words)),
                " ".join(["b"] * max(1, int(words) - 3)),
            ]

        raw = _stage_fullgraph_tensor_peer(
            model, _sentences(first_words))
        assert int(model.inputSpace._ar_embedded_N.shape[1]) == (
            expected_capacity)
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        source = (
            lambda _unused:
            model._forward_with_compiled_sentence_state(None))
        compiled = util.compile(source, verbose=False, fullgraph=True)
        assert compiled is not source

        for expected_trip, words in (
                (first_words, first_words),
                (second_words, second_words)):
            if words != first_words:
                model.zero_grad(set_to_none=True)
                model.End()
                model.symbolSpace.soft_reset()
                raw = _stage_fullgraph_tensor_peer(
                    model, _sentences(words))
            result = model._publish_compiled_sentence_state(compiled(raw))
            (result[0].square().mean()
             + result[2].square().mean()).backward()
            torch.mps.synchronize()
            assert int(model._tensor_peer_trip_count) == expected_trip
            assert all(
                parameter.grad is None
                or bool(torch.isfinite(
                    parameter.grad.coalesce().values()
                    if parameter.grad.is_sparse else parameter.grad).all())
                for parameter in model.parameters())
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
    finally:
        torch._dynamo.reset()
        util.TheCompileBackend = previous_backend
        util.TheCompileMode = previous_mode
        init_device("cpu")


def test_tiny_canonical_detached_reverse_stops_at_root(tmp_path, monkeypatch):
    """The integrated canonical reverse student must not differentiate S."""
    torch.manual_seed(109)
    model = _tiny_canonical_model(
        tmp_path, monkeypatch, forward_grammar_weight=0.25)
    assert model.detached_reverse
    chooser = model.symbolSpace.reverse_chooser
    assert chooser is not None

    input_tensor = model.inputSpace.prepInput(
        ["alpha beta gamma", "delta epsilon"])
    model.forward(input_tensor)
    root = model._stm_single_S
    assert root is not None and root.grad_fn is not None
    root.retain_grad()
    loss, _metric = model._detached_reverse_construction_loss()
    assert loss is not None and torch.isfinite(loss)
    local = model.symbolSpace.reconstruction_stack.forward_loss()
    assert local is not None and torch.isfinite(local)
    assert 0.0 <= float(local.detach()) <= 1.0
    (loss + model.forward_grammar_weight * local).backward()

    assert root.grad is None
    grads = [p.grad for p in chooser.parameters() if p.grad is not None]
    assert grads and any(bool(g.abs().sum() > 0) for g in grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def test_tiny_canonical_detached_reverse_train_step_is_finite(
        tmp_path, monkeypatch):
    """One real optimizer step uses the split objectives without bad grads."""
    torch.manual_seed(127)
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    chooser = model.symbolSpace.reverse_chooser
    before = [p.detach().clone() for p in chooser.parameters()]
    optimizer = model.getOptimizer(lr=1e-3)
    input_tensor = model.inputSpace.prepInput(
        ["alpha beta gamma", "delta epsilon"])
    result, _ = model.runBatch(
        train=True, batchNum=0, batchSize=2, split="train",
        optimizer=optimizer,
        batch_override=(input_tensor, torch.empty(2, 0)))
    assert result is not None
    def _finite_grad(parameter):
        grad = parameter.grad
        if grad is None:
            return True
        checked = grad.coalesce().values() if grad.is_sparse else grad
        return bool(torch.isfinite(checked).all())

    assert all(_finite_grad(p) for p in model.parameters())
    assert any(not torch.equal(old, new.detach())
               for old, new in zip(before, chooser.parameters()))


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
