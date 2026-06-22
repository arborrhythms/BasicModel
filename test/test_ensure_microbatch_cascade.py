"""Task 9 verification: ensure_microbatch(B, K) cascade at InputSpace.forward.

Body-side state (subspace, stacks, last_svo) sizes to B*K so each
microbatch window has its own row inside the body's flattened view.
_stm_fired stays at B because STM firing is a per-source-row gate
shared across all K windows. Discourse buffers also stay at B because
discourse history accumulates across sentences within one source stream
(BasicModel.forward collapses K to mirror legacy last-cursor semantics
before handing the snapshot to the discourse layer).
"""
import os
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT.parent / "bin"))
sys.path.insert(0, str(_PROJECT / "bin"))

import torch


def test_inputspace_forward_triggers_ensure_microbatch():
    """InputSpace.forward in AR mode calls ensure_microbatch(B, K) on symbolSpace."""
    from data import TheData
    from Models import BaseModel
    config = str(_PROJECT / "data" / "MM_xor.xml")
    TheData.load("xor")

    model, _ = BaseModel.from_config(config, data=TheData)
    if model.symbolSpace is None:
        import pytest
        pytest.skip("Model has no SymbolSpace; cascade is a no-op")

    inp = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ).float().unsqueeze(1)  # B=3
    model.eval()
    with torch.no_grad():
        model.forward(inp)
    # XOR text input lexes to outputShape[0] tokens; AR training sets K=T
    # so symbolSpace.batch must be a multiple of B=3.
    B = 3
    assert model.symbolSpace.batch % B == 0, (
        f"symbolSpace.batch={model.symbolSpace.batch} not a multiple of B={B}")
    assert model.symbolSpace.batch >= B, (
        f"symbolSpace.batch={model.symbolSpace.batch} < B={B}")
    # Body-side state sized to B*K
    assert model.symbolSpace._last_svo.shape[0] == model.symbolSpace.batch
    assert model.symbolSpace._svo_valid.shape[0] == model.symbolSpace.batch
    # Post-Phase-D (2026-05-21 SymbolSubSpace/STM Layer refactor):
    # ``SymbolSubSpace`` IS a ``SubSpace`` subclass carrying the typed-
    # STM stack buffers. There is no longer a separate ``self.subspace``
    # peer attribute on SymbolSubSpace.
    assert model.symbolSpace.category_stack._batch == model.symbolSpace.batch
    assert model.symbolSpace.reconstruction_stack._batch == model.symbolSpace.batch
    # _stm_fired stays at B
    assert model.symbolSpace._stm_fired.shape == (B,), (
        f"_stm_fired must stay at B={B}, got {model.symbolSpace._stm_fired.shape}")


def test_ensure_microbatch_cascades_to_discourse():
    """When discourse exists, its buffer stays at B (per-source-stream),
    not B*K -- one discourse history per stream, shared by all K windows.

    Post-2026-05-14 the layer is ARMA(p, q) with ``_s_history`` /
    ``_e_history`` rings instead of the legacy ``_recent`` /
    ``_prev_centroids`` contrastive buffers; the per-batch leading
    dim contract is unchanged.
    """
    from Layers import InterSentenceLayer
    layer = InterSentenceLayer(n_symbols=2, max_depth=2, n_dim=8, batch=1)
    assert layer._batch == 1
    layer.ensure_batch(6)
    assert layer._batch == 6
    assert layer._s_history.shape[0] == 6
    assert layer._s_count.shape == (6,)
    assert layer._e_history.shape[0] == 6
    assert layer._e_count.shape == (6,)


def test_ensure_microbatch_method_explicit_BK():
    """SymbolSpace.ensure_microbatch(B, K) sizes body to B*K, _stm_fired to B,
    discourse to B."""
    from data import TheData
    from Models import BaseModel
    config = str(_PROJECT / "data" / "MM_xor.xml")
    TheData.load("xor")

    model, _ = BaseModel.from_config(config, data=TheData)
    if model.symbolSpace is None:
        import pytest
        pytest.skip("Model has no SymbolSpace")

    model.symbolSpace.ensure_microbatch(B=2, K=5)
    assert model.symbolSpace.batch == 10
    assert model.symbolSpace._stm_fired.shape == (2,)
    assert model.symbolSpace._last_svo.shape[0] == 10
    if model.symbolSpace.discourse is not None:
        assert model.symbolSpace.discourse._batch == 2, (
            f"discourse must stay at B=2, got {model.symbolSpace.discourse._batch}")


def test_stm_fired_survives_K_change():
    """``_stm_fired`` is B-indexed sentence-lifecycle state.  When the
    AR microbatch K changes between batches (PartSpace.forward
    re-quantises K to a power-of-two from the current batch's
    ``actual_max`` BPE word count), the cumulative B*K body batch
    changes -- but the per-source-row fire flag must NOT reset.
    Regression for the bug where ``ensure_batch(BK)`` reallocated
    ``_stm_fired`` to BK-zeros, then ``ensure_microbatch`` reshaped it
    back to B-zeros, wiping the firing history mid-sentence.
    """
    from data import TheData
    from Models import BaseModel
    config = str(_PROJECT / "data" / "MM_xor.xml")
    TheData.load("xor")

    model, _ = BaseModel.from_config(config, data=TheData)
    if model.symbolSpace is None:
        import pytest
        pytest.skip("Model has no SymbolSpace")
    ss = model.symbolSpace
    B = 3

    # Initial sizing at (B=3, K=4)  ->  BK=12.
    ss.ensure_microbatch(B=B, K=4)
    assert ss._stm_fired.shape == (B,)

    # Simulate a source row firing its STM residual.
    ss.mark_stm_fired(0)
    ss.mark_stm_fired(2)
    assert bool(ss._stm_fired[0].item()) is True
    assert bool(ss._stm_fired[1].item()) is False
    assert bool(ss._stm_fired[2].item()) is True

    # K changes (e.g. next batch's actual_max crosses a pow2 boundary).
    # BK goes 12 -> 24; body-side state reallocates.
    ss.ensure_microbatch(B=B, K=8)
    assert ss._stm_fired.shape == (B,)
    assert bool(ss._stm_fired[0].item()) is True, (
        "_stm_fired[0] was wiped by the K-change; sentence-lifecycle "
        "state must survive body-batch reshape")
    assert bool(ss._stm_fired[1].item()) is False
    assert bool(ss._stm_fired[2].item()) is True, (
        "_stm_fired[2] was wiped by the K-change")

    # K changes back to 4 (shrink): same invariant.
    ss.ensure_microbatch(B=B, K=4)
    assert bool(ss._stm_fired[0].item()) is True
    assert bool(ss._stm_fired[2].item()) is True

    # B changes (real sentence-stream boundary): fresh zeros is correct.
    ss.ensure_microbatch(B=B + 1, K=4)
    assert ss._stm_fired.shape == (B + 1,)
    assert not ss._stm_fired.any().item(), (
        "When B changes, _stm_fired should reset -- the rows refer to "
        "different source streams now")
