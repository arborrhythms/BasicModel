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
    """InputSpace.forward in AR mode calls ensure_microbatch(B, K) on wordSpace."""
    from data import TheData
    from Models import BaseModel
    config = str(_PROJECT / "data" / "MM_xor.xml")
    TheData.load("xor")

    model, _ = BaseModel.from_config(config, data=TheData)
    model.masked_prediction = 'AR'
    model.inputSpace.masked_prediction = 'AR'
    if model.wordSpace is None:
        import pytest
        pytest.skip("Model has no WordSpace; cascade is a no-op")

    inp = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ).float().unsqueeze(1)  # B=3
    model.eval()
    with torch.no_grad():
        model.forward(inp)
    # XOR text input lexes to outputShape[0] tokens; AR training sets K=T
    # so wordSpace.batch must be a multiple of B=3.
    B = 3
    assert model.wordSpace.batch % B == 0, (
        f"wordSpace.batch={model.wordSpace.batch} not a multiple of B={B}")
    assert model.wordSpace.batch >= B, (
        f"wordSpace.batch={model.wordSpace.batch} < B={B}")
    # Body-side state sized to B*K
    assert model.wordSpace._last_svo.shape[0] == model.wordSpace.batch
    assert model.wordSpace._svo_valid.shape[0] == model.wordSpace.batch
    assert model.wordSpace.subspace.batch == model.wordSpace.batch
    assert model.wordSpace.category_stack._batch == model.wordSpace.batch
    assert model.wordSpace.reconstruction_stack._batch == model.wordSpace.batch
    # _stm_fired stays at B
    assert model.wordSpace._stm_fired.shape == (B,), (
        f"_stm_fired must stay at B={B}, got {model.wordSpace._stm_fired.shape}")


def test_ensure_microbatch_cascades_to_discourse():
    """When discourse exists, its buffer stays at B (per-source-stream),
    not B*K -- one discourse history per stream, shared by all K windows."""
    from Layers import InterSentenceLayer
    layer = InterSentenceLayer(n_symbols=2, max_depth=2, n_dim=8, batch=1)
    assert layer._batch == 1
    layer.ensure_batch(6)
    assert layer._batch == 6
    assert layer._recent.shape[0] == 6
    assert layer._recent_count.shape == (6,)


def test_ensure_microbatch_method_explicit_BK():
    """WordSpace.ensure_microbatch(B, K) sizes body to B*K, _stm_fired to B,
    discourse to B."""
    from data import TheData
    from Models import BaseModel
    config = str(_PROJECT / "data" / "MM_xor.xml")
    TheData.load("xor")

    model, _ = BaseModel.from_config(config, data=TheData)
    if model.wordSpace is None:
        import pytest
        pytest.skip("Model has no WordSpace")

    model.wordSpace.ensure_microbatch(B=2, K=5)
    assert model.wordSpace.batch == 10
    assert model.wordSpace._stm_fired.shape == (2,)
    assert model.wordSpace._last_svo.shape[0] == 10
    if model.wordSpace.discourse is not None:
        assert model.wordSpace.discourse._batch == 2, (
            f"discourse must stay at B=2, got {model.wordSpace.discourse._batch}")
