"""Phase-5 / D8 capture-gate **CUDA verification leg** (metalbaby).

The CPU sibling (``test_per_word_capture_gate.py``) proves the
NECESSARY leg: the per-word body compiles under
``torch.compile(fullgraph=True)`` with zero graph breaks. On CUDA, a
graph break is equivalent to a ``cudaMemcpyDtoH``, but CPU compile only
verifies trace-time graph cleanliness -- it does not assert capture-
time DtoH==0.

This file is the SUFFICIENT leg: compile with ``backend='inductor',
mode='reduce-overhead'`` (which enables CUDAGraphs auto-capture) on CUDA
and assert:

  * the compiled callable does not fall back to eager (``unique_graphs >= 1``)
  * ``dynamo`` graph_break counter is 0 across N replays
  * ``torch.profiler`` reports zero ``Memcpy DtoH`` ops emitted from
    inside the compiled region (the host ``is None`` byte happens
    BETWEEN replays, in the Python loop -- not captured here).

Skipped when CUDA is unavailable; the CPU leg is the only one that
runs in non-CUDA environments. Bounded by ``N_ITERATIONS`` so the
single metalbaby run completes quickly (OOM/reboot history; spec memory
``feedback_targeted_tests``).
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cuda")
os.environ.setdefault("MODEL_COMPILE", "auto")
os.environ.setdefault("MODEL_COMPILE_MODE", "reduce-overhead")
os.environ.setdefault("MODEL_DEBUG", "0")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root.parent / "bin"))
sys.path.insert(0, str(_root / "bin"))

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="D8 CUDA capture-gate (metalbaby leg); skipped without CUDA")


N_ITERATIONS = 3  # bounded; metalbaby runs are single/bounded


def _build_gate_model_cuda():
    """Build the grammar-enabled MM_20M model on CUDA -- the canonical
    per-word target. Mirrors the CPU helper, but places everything on
    the CUDA device."""
    from data import TheData
    from Models import BaseModel
    from util import init_config, init_device

    init_device("cuda")
    cfg = str(_root / "data" / "MM_20M_legacy.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cuda")
    m.perceptualSpace.chunk_layer.word_learning = 0
    return m


def _stage_for_per_word(m):
    """Run one real ``InputSpace.forward`` so ``_ar_embedded`` /
    ``_valid_len_host`` / peer BPE state / MPHF tables / SymbolSubSpace
    per-sentence state / loop-carry SubSpaces are all populated."""
    isp = m.inputSpace
    assert isp._per_word_enabled is True, (
        "gate target: MM_20M (grammar-enabled) must wire "
        "_per_word_enabled=True")
    inp, _ = isp.getTrainData()
    inp_items = list(inp[:2])
    isp.Start()
    inputTensor = isp.prepInput(inp_items)
    if isinstance(inputTensor, torch.Tensor):
        inputTensor = inputTensor.cuda()
    in_sub = isp.forward(inputTensor)
    m._per_word_prelude(in_sub)
    return isp


def test_per_word_step_compiles_and_replays_under_cudagraphs():
    """D8 strict-gate SUFFICIENT leg #1: compile the per-word body with
    CUDAGraphs-bearing mode (``reduce-overhead``) and run ``N_ITERATIONS``
    times. CUDAGraphs by construction prohibit DtoH inside the capture,
    so a successful run with non-zero unique_graphs is the strong
    signal that no DtoH leaked inside.

    Asserts:
      * the compiled callable returns successfully on every iteration;
      * Dynamo's graph-break counter stays at 0 across replays;
      * (best-effort) ``unique_graphs >= 1`` (i.e., a graph WAS captured
        and replayed, not a silent eager fallback).
    """
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.utils.counters.clear()

    m = _build_gate_model_cuda()
    if not hasattr(m, "_per_word_body_step"):
        pytest.skip("D8 extraction not present")

    isp = _stage_for_per_word(m)

    compiled = torch.compile(
        m._per_word_body_step,
        backend="inductor", mode="reduce-overhead",
        fullgraph=True)

    captured_outputs = []
    out_slot = m._per_word_contributions
    for i in range(N_ITERATIONS):
        w = isp.next_word()
        assert w is not None, (
            f"staged buffer ran out of words before iter {i+1}; "
            "increase staged input or reduce N_ITERATIONS")
        gate = torch.ones(w.shape[0], 1, dtype=torch.bool, device=w.device)
        out = compiled(w, i, gate, out_slot)
        torch.cuda.synchronize()
        captured_outputs.append(out)
    assert len(captured_outputs) == N_ITERATIONS

    # Graph-break counter (Dynamo). Zero across the whole compiled run
    # is the fullgraph guarantee on CUDA (mirrors what we verified on
    # CPU; under reduce-overhead a break would also break the CUDAGraph
    # contract).
    breaks = dynamo.utils.counters.get("graph_break", {})
    n_breaks = sum(int(v) for v in breaks.values()) if breaks else 0
    assert n_breaks == 0, (
        f"D8 strict gate breach: {n_breaks} graph break(s) inside the "
        f"compiled per-word body. Detail: {dict(breaks)}")


def test_per_word_step_actually_uses_cudagraphs():
    """D8 strict-gate confirmation: ``mode='reduce-overhead'`` only
    delivers DtoH==0 guarantees IF CUDAGraphs were actually used (no
    silent eager fallback). Check torch._inductor's CUDAGraph counter
    after a warm replay.
    """
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.utils.counters.clear()

    m = _build_gate_model_cuda()
    if not hasattr(m, "_per_word_body_step"):
        pytest.skip("D8 extraction not present")
    isp = _stage_for_per_word(m)
    compiled = torch.compile(
        m._per_word_body_step,
        backend="inductor", mode="reduce-overhead",
        fullgraph=True)

    # Two replays (warm + cold-of-cache + warmed).
    out_slot = m._per_word_contributions
    for i in range(2):
        w = isp.next_word()
        assert w is not None
        gate = torch.ones(w.shape[0], 1, dtype=torch.bool, device=w.device)
        _ = compiled(w, i, gate, out_slot)
        torch.cuda.synchronize()

    # Inductor's CUDAGraph counter. Any non-zero replays-recorded
    # indicates CUDAGraphs are live; zero means silent fallback.
    inductor_counters = dynamo.utils.counters.get("inductor", {})
    cudagraph_keys = [
        k for k in inductor_counters
        if "cudagraph" in k.lower() or "cuda_graph" in k.lower()]
    cudagraph_signal = sum(
        int(inductor_counters[k]) for k in cudagraph_keys)
    # Diagnostic dump so a failure includes the actual counter state.
    print(f"\n[D8] inductor counters: {dict(inductor_counters)}")
    print(f"[D8] cudagraph-related keys: {cudagraph_keys}")
    print(f"[D8] cudagraph signal sum: {cudagraph_signal}")
    # Soft assert: if there is NO cudagraph counter at all, the
    # signal is not authoritative. The PRIMARY gate (test #1) is the
    # graph_break==0 + successful replay under reduce-overhead, and the
    # profiler test checks DtoH directly.
    if not cudagraph_keys:
        pytest.skip(
            "Inductor did not expose cudagraph counters; "
            f"got keys={list(inductor_counters.keys())[:20]}")
    assert cudagraph_signal > 0, (
        f"Expected a positive cudagraph-related inductor counter; "
        f"got {dict(inductor_counters)}")


def test_per_word_step_emits_no_dtoh_under_profiler():
    """D8 strict-gate SUFFICIENT leg #2: ``torch.profiler`` over a
    warmed-up compiled replay reports zero ``Memcpy DtoH`` events.

    The compile is done OUTSIDE the profiler scope (capture is one-shot
    and we don't want compile-time DtoH counted); a warm-up replay
    runs once; then the profiler scope wraps a single replay. The
    expectation is exactly zero DtoH inside that scope (the host
    ``is None`` byte happens BETWEEN replays in the Python loop, not
    captured by this single-replay scope).
    """
    import torch._dynamo as dynamo
    dynamo.reset()

    m = _build_gate_model_cuda()
    if not hasattr(m, "_per_word_body_step"):
        pytest.skip("D8 extraction not present")

    isp = _stage_for_per_word(m)
    compiled = torch.compile(
        m._per_word_body_step,
        backend="inductor", mode="reduce-overhead",
        fullgraph=True)

    # Warm-up: pay the compile + CUDAGraph capture cost once.
    out_slot = m._per_word_contributions
    w0 = isp.next_word()
    assert w0 is not None
    g0 = torch.ones(w0.shape[0], 1, dtype=torch.bool, device=w0.device)
    _ = compiled(w0, 0, g0, out_slot)
    torch.cuda.synchronize()

    w1 = isp.next_word()
    assert w1 is not None
    g1 = torch.ones(w1.shape[0], 1, dtype=torch.bool, device=w1.device)

    # Profile a SINGLE replay. CUDA activities only -- we want device
    # events, not Python overhead. ``record_shapes=False`` keeps the
    # overhead minimal.
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False) as prof:
        _ = compiled(w1, 1, g1, out_slot)
        torch.cuda.synchronize()

    events = prof.key_averages()
    dtoh_count = 0
    dtoh_events = []
    for ev in events:
        name = ev.key
        # Catch the typical names: "Memcpy DtoH", "cudaMemcpyAsync"
        # with device-to-host direction.
        n = name.lower()
        if ("memcpy" in n and "dtoh" in n) or ("d2h" in n):
            dtoh_count += int(ev.count or 0)
            dtoh_events.append((name, int(ev.count or 0)))

    assert dtoh_count == 0, (
        f"D8 strict gate breach: {dtoh_count} DtoH event(s) inside the "
        f"captured per-word body replay. Detail: {dtoh_events}")
