"""Bit-identical equivalence gate for the BPE tokenizer rewrite.

The GPU tokenizer must produce *exactly* the same model inputs as the
current frozen-vocab trie walk -- a one-token divergence silently
corrupts all training (see memory: fail loud on numerical divergence;
worse here, because it is silent). The switch is gated here, not on
"tests pass".

Design: **inline comparison**. Capture-then-replay is unfaithful --
the codebook trains between batches, so a replayed call gathers from a
drifted codebook. Instead, for every ``_embed_bpe`` invocation we run
the candidate AND the reference back-to-back on the *same*
``upstream_vspace`` at the *same* codebook state, snapshot each one's
outputs, assert bit-identical, then let the model proceed on the
reference result (deterministic prod path; we are validating the
candidate against it).

The criterion is the **muxed event** ``out.event.getW()`` -- the exact
float tensor fed downstream -- plus the ``_bpe_word_mask``
(``word_active``). Same call/state ⇒ identical tokenization+routing ⇒
*exactly* equal event (segmented max over the same gathered codebook
rows); any difference is a real divergence.

Usage: .venv/bin/python test/bpe_gpu_equiv.py
NOT a pytest test (no test_ prefix) -- an explicit gate tool.
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

# The reference: the verified trie-walk path (explicit method, never
# the dispatcher -- so the gate compares GPU-vs-trie, not GPU-vs-GPU).
_REFERENCE_EMBED_BPE = Spaces.PerceptualSpace._embed_bpe_trie
_CANDIDATE_EMBED_BPE = Spaces.PerceptualSpace._embed_bpe_gpu


def _clone(t):
    return t.detach().clone() if torch.is_tensor(t) else t


def _grab(ps, out):
    ev = out.event.getW() if (out is not None and out.event is not None) \
        else None
    return {
        "event_W": _clone(ev),
        "word_active": _clone(getattr(ps, "_bpe_word_mask", None)),
    }


def _assert_eq(name, ref, cand, idx):
    if ref is None and cand is None:
        return
    assert (ref is None) == (cand is None), (
        f"call#{idx} {name}: one side None")
    assert ref.shape == cand.shape, (
        f"call#{idx} {name}: shape {tuple(ref.shape)} != "
        f"{tuple(cand.shape)}")
    if not torch.equal(ref, cand):
        d = (ref.float() - cand.float()).abs()
        raise AssertionError(
            f"call#{idx} {name}: DIVERGENCE -- "
            f"n_diff={int((d > 0).sum())}/{d.numel()} "
            f"max|Δ|={float(d.max()):.3e}")


def install_gate(model, candidate_fn, max_calls=None):
    """Monkeypatch ``_embed_bpe`` so every call runs candidate+reference
    inline on the same upstream/codebook state and asserts bit-identical
    (event_W + word_active). Returns a dict updated with the call count;
    raises AssertionError on the first divergence.
    """
    state = {"n": 0}

    def _gated(ps, upstream_vspace):
        # Candidate first (it may write ps.subspace / _bpe_word_mask)...
        cand_out = candidate_fn(ps, upstream_vspace)
        cand = _grab(ps, cand_out)
        # ...then reference, overwriting ps state; model proceeds on it.
        ref_out = _REFERENCE_EMBED_BPE(ps, upstream_vspace)
        ref = _grab(ps, ref_out)
        i = state["n"]
        _assert_eq("event_W", ref["event_W"], cand["event_W"], i)
        _assert_eq("word_active", ref["word_active"],
                   cand["word_active"], i)
        state["n"] = i + 1
        return ref_out

    Spaces.PerceptualSpace._embed_bpe = _gated
    return state


def run_gate(candidate_fn, max_batches=3):
    init_device("cpu")
    cfg = str(_p / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                 num_shards=1, max_docs=64)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    # Frozen-vocab GPU-train contract: freeze so the vocab cannot grow
    # (train_step no-op) and the GPU path's static tables stay valid.
    m.perceptualSpace.chunk_layer.word_learning = 0
    state = install_gate(m, candidate_fn)
    try:
        opt = m.getOptimizer(lr=1e-4)
        m.runEpoch(optimizer=opt, batchSize=8, split="train",
                   max_batches=max_batches)
    finally:
        Spaces.PerceptualSpace._embed_bpe = _REFERENCE_EMBED_BPE
    return state["n"]


def main():
    # Real gate: GPU candidate vs trie reference, inline, frozen vocab.
    n = run_gate(_CANDIDATE_EMBED_BPE, max_batches=5)
    if n == 0:
        print("[bpe-equiv] WARNING: 0 _embed_bpe calls (config not "
              "BPE mode?) -- gate exercised nothing")
        sys.exit(2)
    print(f"[bpe-equiv] PASS: {n} _embed_bpe call(s) GPU==trie "
          f"bit-identical (event_W + word_active exactly equal).")


if __name__ == "__main__":
    main()
