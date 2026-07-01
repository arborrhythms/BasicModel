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


@pytest.mark.xfail(
    reason="MM_20M_legacy.xml: percept_dim+nWhere+nWhen=12 != concept_dim+nWhere+"
           "nWhen=1028 since Stage 1.C retired sigma_percept (the percept-"
           "to-concept lift); signal router replacement (Stage 3) not "
           "yet wired.",
    strict=False,
)
def test_identity_candidate_passes():
    n = run_space_gate(Spaces.PartSpace, "_embed_bpe",
                        candidate_fn=Spaces.PartSpace._embed_bpe_trie,
                        snapshot=_bpe_snapshot, max_batches=2)
    assert n > 0, "gate exercised nothing (config not BPE mode?)"


@pytest.mark.xfail(
    reason="MM_20M_legacy.xml: percept_dim+nWhere+nWhen=12 != concept_dim+nWhere+"
           "nWhen=1028 since Stage 1.C retired sigma_percept (the percept-"
           "to-concept lift); signal router replacement (Stage 3) not "
           "yet wired. Same pre-existing shape mismatch as "
           "test_identity_candidate_passes -- run_space_gate raises the "
           "shape RuntimeError before reaching the DIVERGENCE check.",
    strict=False,
)
def test_perturbed_candidate_is_caught():
    ref = Spaces.PartSpace._embed_bpe_trie

    def perturbed(self, *a, **k):
        out = ref(self, *a, **k)
        # Spec doc/specs/2026-05-21-subspace-slot-architecture.md: the
        # per-batch event is reconstructed by ``materialize`` from
        # prototype + selection — there is no separate per-batch event
        # tensor to perturb. To inject a detectable perturbation under
        # the new contract, perturb the selection on ``_index``.
        if out is not None and getattr(out, "_index", None) is not None:
            active = out._index
            if torch.is_tensor(active) and active.numel():
                # Shift each per-position selection by 1 (mod V) so the
                # codebook lookup picks a neighbouring row — detectable
                # downstream via materialize / event_W snapshot.
                proto = out.prototype() if hasattr(out, "prototype") else None
                V = proto.shape[0] if proto is not None else 1
                out._index = (active + 1) % max(V, 1)
        return out

    with pytest.raises(AssertionError, match="DIVERGENCE"):
        run_space_gate(Spaces.PartSpace, "_embed_bpe",
                        candidate_fn=perturbed,
                        snapshot=_bpe_snapshot, max_batches=2)
