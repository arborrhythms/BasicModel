"""Reusable bit-identical equivalence gate over an arbitrary Space method.

Generalizes test/bpe_gpu_equiv.py (parameterized by class/method/snapshot):
per gated call run candidate then reference back-to-back, assert
bit-identical via the snapshot, let the model proceed on the REFERENCE
result. Fail loud on the first divergence.

GATE-TARGET CONTRACT: the gated method must NOT mutate a graph-saved leaf
(an nn.Parameter consumed by a retained autograd graph) in the call.
Whole Space.forward/reverse violate this until Phase 1A removes VQ-EMA
(the codebook EMA-writes subspace.event.W in-place every call). Until
then gate pre-VQ substeps (_embed_bpe), exactly as bpe_gpu_equiv does.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "none"
os.environ["BASICMODEL_DEVICE"] = "cpu"

_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin"))
sys.path.insert(0, str(_p / "bin"))

import torch
from data import TheData
from Models import BaseModel
from util import init_config, init_device
import Spaces


def _clone(t):
    return t.detach().clone() if torch.is_tensor(t) else t


def _assert_eq(name, ref, cand, idx):
    if ref is None and cand is None:
        return
    assert (ref is None) == (cand is None), (
        f"call#{idx} {name}: one side None")
    if torch.is_tensor(ref):
        assert ref.shape == cand.shape, (
            f"call#{idx} {name}: shape {tuple(ref.shape)} != "
            f"{tuple(cand.shape)}")
        if not torch.equal(ref, cand):
            d = (ref.float() - cand.float()).abs()
            raise AssertionError(
                f"call#{idx} {name}: DIVERGENCE -- "
                f"n_diff={int((d > 0).sum())}/{d.numel()} "
                f"max|Δ|={float(d.max()):.3e}")
    else:
        if ref != cand:
            raise AssertionError(
                f"call#{idx} {name}: DIVERGENCE -- {ref!r} != {cand!r}")


def default_snapshot(space, out):
    ev = None
    if out is not None and getattr(out, "event", None) is not None:
        ev = out.event.getW()
    return {"event_W": _clone(ev)}


def install_space_gate(cls, method_name, candidate_fn, snapshot):
    """Monkeypatch cls.method_name: per call run candidate then reference,
    assert bit-identical via snapshot, proceed on the reference result.
    Returns {"n": count}; raises on first divergence. state['restore']()
    restores the original. Exact bpe_gpu_equiv pattern, parameterized.
    """
    reference_fn = getattr(cls, method_name)
    state = {"n": 0}

    def _gated(self, *args, **kwargs):
        cand_out = candidate_fn(self, *args, **kwargs)
        cand = snapshot(self, cand_out)
        ref_out = reference_fn(self, *args, **kwargs)
        ref = snapshot(self, ref_out)
        i = state["n"]
        for k in ref:
            _assert_eq(k, ref[k], cand.get(k), i)
        state["n"] = i + 1
        return ref_out

    setattr(cls, method_name, _gated)
    state["restore"] = lambda: setattr(cls, method_name, reference_fn)
    return state


def run_space_gate(cls, method_name, candidate_fn, snapshot=default_snapshot,
                   max_batches=5, batch_size=8):
    """Drive a real frozen MM_20M runEpoch with the gate installed."""
    init_device("cpu")
    cfg = str(_p / "data" / "MM_20M.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                 num_shards=1, max_docs=64)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    m.perceptualSpace.chunk_layer.word_learning = 0  # frozen-vocab contract
    state = install_space_gate(cls, method_name, candidate_fn, snapshot)
    try:
        opt = m.getOptimizer(lr=1e-4)
        m.runEpoch(optimizer=opt, batchSize=batch_size, split="train",
                   max_batches=max_batches)
    finally:
        state["restore"]()
    return state["n"]
