"""Tests for the streaming sentence-batch DataLoader path."""
import pytest
import torch

from data import SentenceStreamDataset


def test_contiguous_streams_basic():
    """B contiguous slabs: at step t, batch[b] == sentences[b*L + t]."""
    sentences = [f"s{i}" for i in range(20)]  # 20 sentences
    B = 4                                      # 4 streams of length 5
    ds = SentenceStreamDataset(sentences, num_streams=B)

    batches = list(iter(ds))
    assert len(batches) == 5, "each stream has 20 // 4 == 5 steps"

    for t, batch in enumerate(batches):
        assert len(batch) == B
        for b in range(B):
            expected = sentences[b * 5 + t]
            assert batch[b] == expected, (
                f"step {t} row {b}: got {batch[b]!r}, "
                f"expected {expected!r} (stream {b} position {t})"
            )


def test_truncates_tail_when_not_divisible():
    """If len(sentences) % B != 0, trailing remainder is dropped."""
    sentences = [f"s{i}" for i in range(23)]  # 23 sentences
    B = 4                                      # stream_len = 5, drop last 3
    ds = SentenceStreamDataset(sentences, num_streams=B)
    batches = list(iter(ds))
    assert len(batches) == 5
    assert batches[-1][-1] == "s19"


def test_empty_sentences_raises():
    with pytest.raises(ValueError):
        SentenceStreamDataset([], num_streams=4)


def test_num_streams_larger_than_sentences_raises():
    with pytest.raises(ValueError):
        SentenceStreamDataset(["a", "b", "c"], num_streams=4)


def test_dataloader_prefetch_preserves_stream_order():
    """Wrapping in DataLoader with num_workers=0 and batch_size=None yields
    the same per-stream order as direct iteration."""
    from torch.utils.data import DataLoader

    sentences = [f"doc_{i:04d}" for i in range(40)]
    B = 8  # stream_length = 5
    ds = SentenceStreamDataset(sentences, num_streams=B)
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    batches = list(loader)
    assert len(batches) == 5
    for t, batch in enumerate(batches):
        for b in range(B):
            assert batch[b] == f"doc_{b * 5 + t:04d}"


def test_loadshards_preserves_order(tmp_path, monkeypatch):
    """loadShards keeps documents in canonical shard order so the streaming
    DataLoader can produce contiguous per-row streams."""
    import data as data_mod

    docs = [f"d{i}" for i in range(100)]
    monkeypatch.setattr(
        data_mod, "iter_documents",
        lambda paths, max_docs=None: iter(docs),
    )
    monkeypatch.setattr(
        data_mod, "get_shard_paths",
        lambda d, num_shards=1: ["fake_shard.parquet"],
    )

    td = data_mod.Data()
    td.loadShards(num_shards=1, max_docs=100, shard_dir=str(tmp_path))

    # 80/10/10 split; the first 80 docs should be the train set, in order.
    assert td.train_input[:10] == [f"d{i}" for i in range(10)]
    assert td.train_input[-1] == "d79"


def test_data_loader_uses_train_input_in_order():
    """data_loader(split='train') produces a DataLoader whose batches
    reflect train_input (no re-shuffling) and respect num_streams."""
    import data as data_mod

    td = data_mod.Data()
    td.train_input = [f"s{i}" for i in range(40)]
    td.train_output = [torch.zeros(1) for _ in td.train_input]

    loader = td.data_loader(split="train", num_streams=8, num_workers=0,
                            prefetch_factor=None)
    batches = list(loader)
    assert len(batches) == 5
    # Each batch is (inputs, outputs); inputs is a list of 8 strings.
    inp, _ = batches[0]
    assert list(inp) == [f"s{b * 5}" for b in range(8)]


def test_data_loader_caps_num_streams_to_split_length():
    """batchSize>len(split): num_streams is capped so small eval sets
    still yield one rectangular batch."""
    import data as data_mod

    td = data_mod.Data()
    td.test_input = [f"t{i}" for i in range(4)]
    td.test_output = [torch.zeros(1) for _ in td.test_input]

    loader = td.data_loader(split="test", num_streams=10, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1
    inp, _ = batches[0]
    assert len(list(inp)) == 4


def test_dataset_yields_tensor_slices_for_tensor_input():
    """Tensor-shaped train_input (mnist-style) is fancy-indexed per step
    instead of list-wrapped."""
    import data as data_mod

    td = data_mod.Data()
    td.train_input = torch.arange(40).reshape(10, 4).float()
    td.train_output = torch.arange(10).unsqueeze(1).float()

    loader = td.data_loader(split="train", num_streams=2, num_workers=0)
    batches = list(loader)
    assert len(batches) == 5  # stream_length = 10 // 2
    inp0, out0 = batches[0]
    assert isinstance(inp0, torch.Tensor)
    assert inp0.shape == (2, 4)
    # Stream 0 starts at index 0, stream 1 starts at index 5 (L=5).
    assert torch.equal(inp0[0], td.train_input[0])
    assert torch.equal(inp0[1], td.train_input[5])
    assert out0.shape == (2, 1)


def test_prep_sentence_batch_delegates_to_prepinput():
    """prep_sentence_batch forwards to prepInput via the same instance path
    used in forward(). Uses a proxy object to verify delegation without
    constructing a full InputSpace."""
    import Spaces
    from types import SimpleNamespace

    seen = {}
    def fake_prep(batch):
        seen["batch"] = batch
        return "sentinel"

    proxy = SimpleNamespace(prepInput=fake_prep)
    result = Spaces.InputSpace.prep_sentence_batch(
        proxy, ("s0", "s1", "s2"))
    assert result == "sentinel"
    assert seen["batch"] == ["s0", "s1", "s2"]


# ---------------------------------------------------------------------------
# Codebook.setW activations must not overwrite the codebook Parameter
# ---------------------------------------------------------------------------

def test_codebook_setw_preserves_parameter_against_activation():
    """Regression for the checkpoint-corruption bug: once a Codebook holds
    its VQ codebook as an ``nn.Parameter``, subsequent ``setW(activation)``
    calls must stash the activation in a transient slot -- never replace
    the Parameter.

    Symptom that used to bite: save_weights serialises
    ``subspace.event.W`` with a batch-shaped tensor (``[B, N, D]``)
    instead of the codebook (``[V, D]``), and a fresh model can no
    longer load the checkpoint.
    """
    import torch.nn as nn
    import Spaces

    cb = Spaces.Codebook()
    # Install a fake codebook Parameter shaped [V=4, D=3] the way
    # ``addVectors`` would.
    codebook = nn.Parameter(torch.eye(4, 3))
    cb.setW(codebook)
    assert "W" in cb._parameters, "codebook must register as a Parameter"
    saved_shape = list(cb.state_dict()["W"].shape)
    assert saved_shape == [4, 3]

    # Now push an activation through setW -- shape [B=2, N=4, D=3], the
    # exact pattern that happens during the forward pass.
    activation = torch.randn(2, 4, 3)
    cb.setW(activation)

    # The codebook Parameter must still be the registered W.
    assert "W" in cb._parameters, (
        "setW(activation) must not strip the codebook Parameter"
    )
    state = cb.state_dict()
    assert "W" in state
    assert list(state["W"].shape) == [4, 3], (
        f"state_dict W must still be the codebook (shape [4, 3]); "
        f"got {list(state['W'].shape)} -- activations leaked into "
        "state_dict again"
    )

    # getW() must return the transient activation while cached, so the
    # forward pipeline still sees what it stored.
    got = cb.getW()
    assert got.shape == (2, 4, 3)
    assert torch.equal(got, activation)

    # Clearing the activation via setW(None) must preserve the codebook.
    cb.setW(None)
    assert "W" in cb._parameters
    restored = cb.getW()
    assert torch.equal(restored, codebook), (
        "after setW(None) getW() should fall back to the codebook Parameter"
    )


# ---------------------------------------------------------------------------
# toDevice must keep list-of-tensor splits on CPU so DataLoader workers
# can pickle them across process boundaries.
# ---------------------------------------------------------------------------

def test_todevice_keeps_list_of_tensor_outputs_on_cpu():
    """List-of-tensor splits (masked-LM placeholder outputs, per-sentence
    targets) must stay on CPU after ``toDevice()``. Live CUDA tensors
    can't be shipped from a DataLoader worker process back to the main
    process; the workers crash with 'Attempted to send CUDA tensor
    received from another process' when we try.

    This test uses whatever device ``TheDevice`` resolves to locally
    (CPU in CI), but the invariant it asserts -- ``toDevice()`` leaves
    list tensors alone -- is what prevents the crash on CUDA hosts.
    """
    import data as data_mod

    td = data_mod.Data()
    td.train_input = ["a b c", "d e f", "g h i", "j k l"]
    # Per-sentence placeholder tensors, as ``processLM`` creates for
    # masked-prediction mode.
    placeholder = [torch.zeros(1) for _ in td.train_input]
    td.train_output = placeholder
    td.validation_input = []
    td.validation_output = []
    td.test_input = []
    td.test_output = []

    td.toDevice()

    # The list identity / tensor identity should be preserved -- toDevice
    # must not have cloned each element onto some other device.
    assert td.train_output is placeholder, (
        "toDevice unexpectedly rebuilt the list-of-tensor split"
    )
    for t in td.train_output:
        assert t.device.type == "cpu", (
            f"list-of-tensor output element moved off CPU: {t.device}"
        )


def test_data_loader_list_outputs_survive_worker_pickling():
    """End-to-end: a streaming DataLoader with ``num_workers=1`` over a
    list-of-tensor output split must not raise. The worker must be able
    to pickle each slice back to the main process.

    Skipped on platforms where ``fork`` isn't available and spawning is
    too slow to be worth running in the fast suite.
    """
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) == "spawn":
        pytest.skip("spawn start-method is too slow for a fast unit test")

    import data as data_mod

    td = data_mod.Data()
    td.train_input = [f"sent {i}" for i in range(8)]
    td.train_output = [torch.zeros(1) for _ in td.train_input]
    td.toDevice()  # must leave the list on CPU

    loader = td.data_loader(
        split="train", num_streams=4,
        num_workers=1, prefetch_factor=2,
    )
    batches = list(loader)
    assert len(batches) == 2   # stream_length = 8 // 4
    inp0, out0 = batches[0]
    assert list(inp0) == ["sent 0", "sent 2", "sent 4", "sent 6"]
    assert all(isinstance(t, torch.Tensor) for t in out0)
    assert all(t.device.type == "cpu" for t in out0)
