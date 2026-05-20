"""Self-check: identity candidate passes; perturbed candidate FAILS LOUD.
Targets PerceptualSpace._embed_bpe -- the same pre-VQ, frozen-vocab-
idempotent anchor test/bpe_gpu_equiv.py uses.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
import pytest
import torch
import Spaces
from test.space_equiv import run_space_gate, _clone


def _bpe_snapshot(ps, out):
    ev = out.event.getW() if (out is not None and out.event is not None) else None
    return {"event_W": _clone(ev),
            "word_active": _clone(getattr(ps, "_bpe_word_mask", None))}


def test_identity_candidate_passes():
    n = run_space_gate(Spaces.PerceptualSpace, "_embed_bpe",
                        candidate_fn=Spaces.PerceptualSpace._embed_bpe_trie,
                        snapshot=_bpe_snapshot, max_batches=2)
    assert n > 0, "gate exercised nothing (config not BPE mode?)"


def test_perturbed_candidate_is_caught():
    ref = Spaces.PerceptualSpace._embed_bpe_trie

    def perturbed(self, *a, **k):
        out = ref(self, *a, **k)
        if out is not None and getattr(out, "event", None) is not None:
            w = out.event.getW()
            if torch.is_tensor(w) and w.numel():
                out.event.setW(w + 1e-3)
        return out

    with pytest.raises(AssertionError, match="DIVERGENCE"):
        run_space_gate(Spaces.PerceptualSpace, "_embed_bpe",
                        candidate_fn=perturbed,
                        snapshot=_bpe_snapshot, max_batches=2)
