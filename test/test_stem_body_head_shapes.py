"""Shape contracts for stem/body/head pipeline (Task 5 of microbatch plan).

The stem (InputSpace + PerceptualSpace) must emit a subspace with
event shape [B, K, N, D] and k_axis=True. K = T for AR training (one
window per token), K = T-N+1 for inference (typically K=1 with
T==N).
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import torch

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


@pytest.fixture
def model():
    """A MentalModel built from MM_xor.xml so InputSpace and PerceptualSpace
    are wired with peer references and an actual subspace."""
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return m


def test_inputspace_emits_b_k_n_d_in_ar_mode(model):
    """AR forward emits [B, K, N, D] with k_axis True and a valid mask."""
    inp = model.inputSpace
    inp.masked_prediction = "ARLM"
    inp.Start()
    # Drive a real batch so the lex/embed runs end to end.
    loader = model.inputSpace.data.data_loader(split="train", num_streams=2)
    inp_items, _ = next(iter(loader))
    inp_tensor = inp.prepInput(inp_items)

    sub = inp.forward(inp_tensor)

    assert sub.k_axis is True, "AR stem must set k_axis=True"
    event = sub.materialize()
    assert event.dim() == 4, f"AR stem must emit 4D event, got {tuple(event.shape)}"
    B, K, N, D = event.shape
    assert N == int(inp.outputShape[0]), \
        f"window N={N} must match nActive={inp.outputShape[0]}"
    assert sub.valid_mask_bk is not None
    assert sub.valid_mask_bk.shape == (B, K)


def test_inputspace_inference_k1_degenerate(model):
    """ARIR runtime path: T==N -> K=1 single-window emission."""
    inp = model.inputSpace
    inp.masked_prediction = "ARIR"
    inp.Start()
    # Mark data as runtime ARIR so the inference branch fires.
    inp.data._runtime_mode = "ARIR"
    try:
        loader = inp.data.data_loader(split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        inp_tensor = inp.prepInput(inp_items)

        sub = inp.forward(inp_tensor)

        assert sub.k_axis is True
        event = sub.materialize()
        assert event.dim() == 4
        B, K, N, D = event.shape
        assert K == 1, f"runtime ARIR must produce K=1, got K={K}"
    finally:
        inp.data._runtime_mode = None


def test_inputspace_non_ar_keeps_k_axis_false(model):
    """Non-AR mode bypasses K-windowing — k_axis stays False."""
    inp = model.inputSpace
    inp.masked_prediction = "NONE"
    inp.Start()
    loader = inp.data.data_loader(split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    inp_tensor = inp.prepInput(inp_items)

    sub = inp.forward(inp_tensor)

    assert sub.k_axis is False, "non-AR stem must leave k_axis=False"


def test_subspace_default_k_axis_false():
    """Fresh SubSpace has k_axis=False and no valid_mask."""
    from Spaces import SubSpace
    ss = SubSpace([4, 8], [4, 8], nInputDim=8, nOutputDim=8)
    assert ss.k_axis is False
    assert ss.valid_mask_bk is None


def test_inputspace_progressive_prefix_windows(model):
    """AR window k must equal a left-padded slice of the embedded input.

    Window k holds N positions: the rightmost min(k, N) slots are the
    last embedded tokens [emb[max(0,k-N)], ..., emb[k-1]] and earlier
    slots are zero-padded. Concretely window 0 is all zeros (no token
    seen yet), and window k>=N is the last N tokens.
    """
    inp = model.inputSpace
    inp.masked_prediction = "ARLM"
    inp.Start()
    loader = inp.data.data_loader(split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    inp_tensor = inp.prepInput(inp_items)

    sub = inp.forward(inp_tensor)
    event = sub.materialize()
    B, K, N, D = event.shape
    embedded = inp._ar_embedded  # [B, T, D] with T == K
    assert embedded.shape[1] == K

    # Window 0: must be all zeros (cursor 0 has no tokens yet).
    assert torch.allclose(event[:, 0], torch.zeros_like(event[:, 0]))

    # Window 1: rightmost slot equals emb[0]; earlier slots zero.
    assert torch.allclose(event[:, 1, -1, :], embedded[:, 0, :])
    assert torch.allclose(event[:, 1, :-1, :], torch.zeros_like(event[:, 1, :-1, :]))

    # Window N-1 (or last): rightmost N-1 slots equal emb[0..N-2], slot 0 is zero.
    if K >= N:
        last_full_idx = N - 1
        # event[:, last_full_idx, 1:, :] should equal embedded[:, :N-1, :]
        rhs = event[:, last_full_idx, 1:, :]
        assert torch.allclose(rhs, embedded[:, :N - 1, :])
        assert torch.allclose(
            event[:, last_full_idx, 0, :],
            torch.zeros_like(event[:, last_full_idx, 0, :]),
        )
