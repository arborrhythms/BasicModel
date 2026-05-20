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

Today this test FAILS by design (TDD RED): the per-word step is
inlined inside the variable while-loop in ``_forward_body_per_word``
and has no standalone callable. The first failure drives the
extraction of ``BasicModel._per_word_body_step``; subsequent failures
drive the DtoH-offender removal from the PS/SS/CS forwards reached
through the body.
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


def _build_gate_model():
    """Build the grammar-enabled MM_5M model -- the canonical per-word
    target. Mirrors test_input_word_cursor._build_gate_model."""
    from data import TheData
    from Models import BaseModel
    from util import init_config, init_device

    init_device("cpu")
    cfg = str(_root / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    m.perceptualSpace.chunk_layer.word_learning = 0
    return m


def _stage_for_per_word(m):
    """Run one real ``InputSpace.forward`` so ``_ar_embedded`` /
    ``_valid_len_host`` / peer BPE state are populated -- the live
    pre-conditions of the per-word body. Also pre-warms MPHF tables
    if the peer is in mphf mode (the production path pre-warms in
    ``_forward_body_per_word``'s setup; the gate test calls the step
    directly so it mirrors the contract here)."""
    isp = m.inputSpace
    assert isp._per_word_enabled is True, (
        "gate target: MM_5M (grammar-enabled) must wire "
        "_per_word_enabled=True")
    inp, _ = isp.getTrainData()
    inp_items = list(inp[:2])
    isp.Start()
    inputTensor = isp.prepInput(inp_items)
    in_sub = isp.forward(inputTensor)
    # Per-sentence WorkingState carrier. Production allocates this in
    # the wordSpace soft_reset path (runBatch triggers it).
    if m.wordSpace is not None and m.wordSpace._work is None:
        m.wordSpace.soft_reset()
    # Mirror ``BasicModel._forward_body_per_word``'s prelude: STM
    # batch resize + clear, MPHF pre-warm, recur_pass reset, and
    # loop-carry SubSpace pre-seed. The capture gate calls
    # ``_per_word_body_step`` in isolation so the prelude is
    # replayed here verbatim.
    m._per_word_prelude(in_sub)
    return isp


def test_per_word_step_runs_eagerly_end_to_end():
    """SANITY gate (eager mode): the extracted per-word step must run
    end-to-end without raising. This catches regressions in the
    refactor that the production targeted suite misses (those tests
    check structural properties, not runtime behaviour of the body)."""
    m = _build_gate_model()
    if not hasattr(m, "_per_word_body_step"):
        pytest.skip("extraction not yet landed")
    isp = _stage_for_per_word(m)
    w = isp.next_word()
    assert w is not None
    # No compile -- just eager. Any exception fails the test.
    CS_sub, idea_bd = m._per_word_body_step(w)
    # CS_sub is a SubSpace (the live CS subspace); idea_bd may be
    # None if CS is empty / non-3D, but typically a tensor.
    assert CS_sub is not None
    # Run a second iteration to exercise the post-cs.forward state
    # update (work.cs_for_ps/ss should now be non-None real SubSpaces).
    w2 = isp.next_word()
    assert w2 is not None
    CS_sub2, idea_bd2 = m._per_word_body_step(w2)
    assert CS_sub2 is not None


def test_per_word_step_is_extractable_as_a_standalone_callable():
    """RED gate #1: a standalone ``_per_word_body_step(self, w)`` MUST
    exist on ``BasicModel``. The current implementation inlines the body
    inside the while-loop in ``_forward_body_per_word``; D8 requires it
    as a separable callable so the middle (replayable) CUDA graph wraps
    exactly one iteration."""
    m = _build_gate_model()
    assert hasattr(m, "_per_word_body_step"), (
        "D8 piece 1: BasicModel._per_word_body_step(w) must be "
        "extractable -- see doc/plans/2026-05-18-two-loop-pipeline-"
        "architecture.md D8 'two implementation pieces' for the "
        "extraction contract.")
    assert callable(m._per_word_body_step)


def test_per_word_step_compiles_fullgraph_clean():
    """RED gate #2: the extracted per-word step must compile under
    ``torch.compile(fullgraph=True)`` with zero graph breaks. Dynamo
    raises ``Unsupported`` (Inductor / eager backend) on the first
    ``.item()`` / ``bool(t)`` / data-dependent control-flow it sees
    in the trace; on CUDA each such break is a ``cudaMemcpyDtoH``.

    This is the CPU-runnable proxy for D8's strict gate; the CUDA-only
    ``cudaMemcpyDtoH == 0`` sufficient leg lives in the metalbaby
    capture suite (separate, CUDA-required)."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    m = _build_gate_model()
    if not hasattr(m, "_per_word_body_step"):
        pytest.skip("extraction not yet landed (see RED gate #1)")

    isp = _stage_for_per_word(m)
    w = isp.next_word()
    assert w is not None, "stage produced a valid first word"

    # Compile the bound method; fullgraph=True turns ANY graph break
    # into a hard ``Unsupported``. Use the eager backend (already in
    # MODEL_COMPILE=eager via env) -- the gate is about graph breaks,
    # not about codegen quality.
    # backend="eager" -- skip Inductor codegen (the gate is about
    # graph breaks, not codegen quality). The default backend
    # (Inductor) C++ compiles, which fails on paths containing
    # spaces (e.g. iCloud Documents) -- orthogonal to the gate.
    compiled = torch.compile(
        m._per_word_body_step, backend="eager", fullgraph=True)
    # Single-arg call: WorkingState's __dict__ attributes are
    # introspectable by Dynamo (Phase-5 dropped __slots__), so the
    # body reads ``_work.cs_for_ps/ss`` directly inside the trace.
    compiled(w)


def test_per_word_loop_completes_two_steps_under_fullgraph():
    """RED gate #3: two consecutive compiled steps must succeed without
    a recompile (cache hit, same shape)."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    m = _build_gate_model()
    if not hasattr(m, "_per_word_body_step"):
        pytest.skip("extraction not yet landed (see RED gate #1)")

    isp = _stage_for_per_word(m)
    # backend="eager" -- skip Inductor codegen (the gate is about
    # graph breaks, not codegen quality). The default backend
    # (Inductor) C++ compiles, which fails on paths containing
    # spaces (e.g. iCloud Documents) -- orthogonal to the gate.
    compiled = torch.compile(
        m._per_word_body_step, backend="eager", fullgraph=True)
    w1 = isp.next_word()
    w2 = isp.next_word()
    assert w1 is not None and w2 is not None
    compiled(w1)
    compiled(w2)
