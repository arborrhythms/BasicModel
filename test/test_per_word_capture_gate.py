"""Phase-5 / D8 capture-gate: the per-word iteration body must be
fullgraph-clean.

D8 (the three-graph capture strategy in
``doc/plans/2026-05-18-two-loop-pipeline-architecture.md``) defines
the per-word body (SHIFT + REDUCE) as the "PS/SS → CS graph",
replayed N times per forward. The strict gate is

    cudaMemcpyDtoH == 0   inside each captured graph
    unique_graphs   ≤ 3   total
    boundary breaks ==  2 (IS→loop, loop→OS)
    only DtoH per word == the host ``next_word() is None`` byte

This test is the **CPU-runnable proxy** for the DtoH==0 part: if any
``.item()`` / ``bool(t)`` / ``int(tensor)`` is reached from the per-word
body's call graph, Dynamo's ``fullgraph=True`` raises ``Unsupported``
because those force a graph break. On CUDA each such break is also a
``cudaMemcpyDtoH``. CPU fullgraph passing is necessary (not sufficient)
for the CUDA DtoH==0 gate; the metalbaby unit test (separate, CUDA-only)
covers the sufficient leg.

The current scheduler receives three pure native word transactions.
The fixture captures those callables at that boundary and checks one
complete word cell, including a second call with the returned state.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")  # capture-gate only
os.environ.setdefault("MODEL_DEBUG", "0")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root.parent / "bin"))
sys.path.insert(0, str(_root / "bin"))

import pytest
import torch


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Self-isolation for the fullgraph pins: a PRIOR test in the same
    process may have blown dynamo's recompile limit on the very code
    object compiled here (test_compile_static_loop's churn test does),
    which marks it skip-forever — fullgraph then reports "no compiled
    frames. Compilation was not attempted" without ever tracing. Reset
    dynamo's global caches before each test in this file (found by
    order-bisect, 2026-07-13; deterministic in the 9-file sweep, absent
    in isolation)."""
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
    except Exception:
        pass
    yield


def _build_gate_model():
    """Build the grammar-enabled MM_20M model -- the canonical per-word
    target. Mirrors test_input_word_cursor._build_gate_model."""
    from data import TheData
    from Models import BaseModel
    from util import init_config, init_device

    init_device("cpu")
    cfg = str(_root / "data" / "MM_ladder.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    m.perceptualSpace.chunk_layer.word_learning = 0
    return m


def _stage_for_per_word(m, monkeypatch, *, samples=None):
    """Extract the live pure word transactions after their normal host staging.

    The old compatibility body predates native cube codes.  The active
    pipeline hands these three callbacks to TensorPeerWhilePipeline; this
    fixture captures that boundary without running or replacing a callback.
    """
    import Models
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer

    _stage_fullgraph_tensor_peer(m, ['alpha beta', 'gamma delta'] if samples is None else samples)
    isp = m.inputSpace
    width = int(isp._ar_embedded_N.shape[1])
    assert m._tensor_peer_while_ready(width)
    staged = {}

    class Captured(Exception):
        pass

    def capture(_pipeline, words, active, **kwargs):
        staged.update(words=words, active=active, **kwargs)
        raise Captured

    # Host closes are outside this capture gate; the same numerical callbacks
    # also run under the banked tensor scheduler with explicit state carries.
    m._sentence_ends = False
    with monkeypatch.context() as context:
        context.setattr(Models.TensorPeerWhilePipeline, 'run_cs_lanes_banked', capture)
        with pytest.raises(Captured):
            m._run_tensor_peer_word_pipeline(width)

    def step(word, gate, index, carry):
        stm, sub, sym, lang, feedback = carry
        live = torch.ones((), dtype=torch.bool, device=word.device)
        payload, sub = staged['stage_cs_sub'](word, gate, index, live, sub)
        payload, sym = staged['stage_cs_sym'](payload, feedback, index, live, sym)
        lang, stm, feedback = staged['stage_cs_lang'](payload, index, live, lang, stm)
        return stm, sub, sym, lang, feedback

    carry = tuple(staged[key] for key in ('stm_state', 'cs_sub_state',
        'cs_sym_state', 'cs_lang_state', 'empty_feedback'))
    # A loop carry has one autograd signature. Its initial zero tensors do
    # not yet have the gradient metadata of a returned native state. Derive
    # that metadata from one pure word step without consuming the walk RNG,
    # then give the original zeros the same leaf signature for isolated calls.
    with torch.random.fork_rng(devices=[]):
        template = step(staged['words'][:, :1], staged['active'][:, :1],
            staged['words'].new_zeros((), dtype=torch.long), carry)
    def typed(initial, returned):
        if torch.is_tensor(initial):
            return initial.detach().clone().requires_grad_(returned.requires_grad)
        return tuple(typed(a, b) for a, b in zip(initial, returned))
    carry = typed(carry, template)
    return step, carry, staged


def _word(staged, index):
    words, active = staged['words'], staged['active']
    assert bool(active[:, index].all())
    return words[:, index:index+1], active[:, index:index+1], words.new_tensor(index, dtype=torch.long)


@pytest.mark.slow
def test_per_word_step_runs_eagerly_end_to_end(monkeypatch):
    """Both native words traverse PS/WS, CS and grammar eagerly."""
    m = _build_gate_model()
    step, carry, staged = _stage_for_per_word(m, monkeypatch)
    first = step(*_word(staged, 0), carry)
    assert first[1][0] is not None
    second = step(*_word(staged, 1), first)
    assert second[1][0] is not None
    assert bool(torch.isfinite(second[1][0]).all())


@pytest.mark.slow
def test_per_word_step_is_extractable_as_a_standalone_callable(monkeypatch):
    """The active scheduler receives independently callable native stages."""
    m = _build_gate_model()
    step, carry, staged = _stage_for_per_word(m, monkeypatch)
    assert callable(step)
    for key in ('stage_cs_sub', 'stage_cs_sym', 'stage_cs_lang'):
        assert callable(staged[key])


@pytest.mark.slow
def test_per_word_step_compiles_fullgraph_clean(monkeypatch):
    """One native word cell compiles with graph breaks forbidden."""
    if not hasattr(torch, 'compile'):
        pytest.skip('torch.compile unavailable')
    m = _build_gate_model()
    step, carry, staged = _stage_for_per_word(m, monkeypatch)
    compiled = torch.compile(step, backend='eager', fullgraph=True)
    result = compiled(*_word(staged, 0), carry)
    choice_ids, choice_arities, choice_mask = result[3][1:4]
    assert torch.is_tensor(choice_ids)
    assert torch.is_tensor(choice_arities)
    assert torch.is_tensor(choice_mask)


@pytest.mark.slow
def test_per_word_loop_completes_two_steps_under_fullgraph(monkeypatch):
    """The next native word reuses the same fullgraph with its returned state."""
    if not hasattr(torch, 'compile'):
        pytest.skip('torch.compile unavailable')
    m = _build_gate_model()
    step, carry, staged = _stage_for_per_word(m, monkeypatch)
    graphs = []
    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(step, backend=backend, fullgraph=True)
    first = compiled(*_word(staged, 0), carry)
    second = compiled(*_word(staged, 1), first)
    assert len(graphs) == 1
    assert bool(torch.isfinite(second[0][0]).all())
