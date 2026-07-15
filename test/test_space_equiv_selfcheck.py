"""Self-check: identity candidate passes; perturbed candidate FAILS LOUD.
Targets PartSpace._embed_bpe -- the same pre-VQ, frozen-vocab-
idempotent anchor test/bpe_gpu_equiv.py uses.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
import pytest
import torch
import Spaces
from space_equiv import run_space_gate, _clone


def _bpe_snapshot(ps, out):
    # Spec doc/specs/2026-05-21-subspace-slot-architecture.md: read the
    # per-batch event via ``materialize`` so the snapshot reflects
    # codebook[selection] (the reconstructed content), not the static
    # prototype Parameter that ``event.getW`` now returns.
    ev = (out.materialize(mode="event")
          if (out is not None and hasattr(out, "materialize")) else None)
    return {"event_W": _clone(ev),
            "word_active": _clone(getattr(ps, "_bpe_word_mask", None))}


def test_identity_candidate_passes():
    n = run_space_gate(Spaces.PartSpace, "_embed_bpe",
                        candidate_fn=Spaces.PartSpace._embed_bpe_trie,
                        snapshot=_bpe_snapshot, max_batches=2)
    assert n > 0, "gate exercised nothing (config not BPE mode?)"


def test_perturbed_candidate_is_caught():
    ref = Spaces.PartSpace._embed_bpe_trie

    def perturbed(self, *a, **k):
        out = ref(self, *a, **k)
        # Perturb an observable that the reference call deterministically
        # rebuilds. Selection-index perturbation became a no-op when this
        # fixture's cold-start codebook had only one row (modulo V == 1).
        active = getattr(self, "_bpe_word_mask", None)
        if torch.is_tensor(active) and active.numel():
            self._bpe_word_mask = torch.ones_like(active) - active
        return out

    with pytest.raises(AssertionError, match="DIVERGENCE"):
        run_space_gate(Spaces.PartSpace, "_embed_bpe",
                        candidate_fn=perturbed,
                        snapshot=_bpe_snapshot, max_batches=2)
