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


# ---------------------------------------------------------------------------
# InputSpace.expand_masked_batched -- per-position masking over B rows
# ---------------------------------------------------------------------------

def _make_fake_inputspace(where_dim=2, when_dim=2):
    """Build a stand-in object with just the attributes
    ``expand_masked_batched`` reads from ``self`` -- avoids bringing up the
    full XMLConfig / TheData / Embedding stack just to exercise masking
    math.
    """
    from types import SimpleNamespace
    where_enc = SimpleNamespace(
        nDim=where_dim,
        index=[-(where_dim + when_dim) + i for i in range(where_dim)]
              if where_dim > 0 else [],
    )
    when_enc = SimpleNamespace(
        nDim=when_dim,
        index=[-2, -1] if when_dim > 0 else [],
    )
    subspace = SimpleNamespace(whereEncoding=where_enc, whenEncoding=when_enc)
    return SimpleNamespace(subspace=subspace)


def test_expand_masked_batched_mlm_zeros_content_preserves_positional():
    """MLM mode: content dims at (row b, pos) go to zero; where/when dims
    are preserved; targets[b] == embedded[b, pos] for every row with enough
    words."""
    import Spaces

    fake = _make_fake_inputspace(where_dim=2, when_dim=2)
    B, nVec, embSize = 3, 6, 8  # last 4 dims = [where0, where1, when0, when1]
    embedded = torch.arange(B * nVec * embSize,
                            dtype=torch.float).reshape(B, nVec, embSize)
    sentences = ["a b c", "d e f g", "h i j k l"]
    pos = 1

    masked, targets, mask_positions = Spaces.InputSpace.expand_masked_batched(
        fake, embedded, sentences, maskedPrediction="MLM", pos=pos,
    )

    assert masked.shape == (B, nVec, embSize)
    assert targets.shape == (B, embSize)
    assert mask_positions == [pos] * B  # all three rows have >=2 words

    # Content dims are [0..3] (first 4); positional dims are [4..7] (last 4).
    for b in range(B):
        assert torch.allclose(masked[b, pos, :4],
                              torch.zeros(4)), f"row {b} content not zero"
        assert torch.allclose(masked[b, pos, 4:],
                              embedded[b, pos, 4:]), (
            f"row {b} positional dims were overwritten"
        )
        assert torch.allclose(targets[b], embedded[b, pos]), (
            f"row {b} target != embedded[b, pos]"
        )
        # Other positions untouched
        for other in range(nVec):
            if other == pos:
                continue
            assert torch.allclose(masked[b, other], embedded[b, other]), (
                f"row {b} position {other} was modified by MLM mask"
            )


def test_expand_masked_batched_arlm_truncates_future():
    """ARLM mode: positions after the mask are zeroed for each row."""
    import Spaces

    fake = _make_fake_inputspace(where_dim=0, when_dim=0)
    B, nVec, embSize = 2, 5, 4
    embedded = torch.ones(B, nVec, embSize)
    sentences = ["a b c d e", "a b c"]
    pos = 1

    masked, _, mask_positions = Spaces.InputSpace.expand_masked_batched(
        fake, embedded, sentences, maskedPrediction="ARLM", pos=pos,
    )
    assert mask_positions == [pos, pos]
    # Row 0: positions 2,3,4 are zeroed; position 1 content zeroed; 0 kept.
    assert torch.allclose(masked[0, 0], torch.ones(embSize))
    assert torch.allclose(masked[0, 1], torch.zeros(embSize))
    assert torch.allclose(masked[0, 2:], torch.zeros(nVec - 2, embSize))
    # Row 1: same structure but only 3 words -- pos=1 is still valid.
    assert torch.allclose(masked[1, 0], torch.ones(embSize))
    assert torch.allclose(masked[1, 1], torch.zeros(embSize))
    assert torch.allclose(masked[1, 2:], torch.zeros(nVec - 2, embSize))


def test_expand_masked_batched_skips_rows_with_too_few_words():
    """Rows whose word count is <= pos are passed through unchanged and
    report mask_position == -1."""
    import Spaces

    fake = _make_fake_inputspace(where_dim=0, when_dim=0)
    B, nVec, embSize = 3, 4, 3
    embedded = torch.arange(B * nVec * embSize,
                            dtype=torch.float).reshape(B, nVec, embSize)
    sentences = ["only two words", "a", ""]  # row 1 has 1 word, row 2 empty
    pos = 2  # valid only for row 0 (3 words)

    masked, targets, mask_positions = Spaces.InputSpace.expand_masked_batched(
        fake, embedded, sentences, maskedPrediction="MLM", pos=pos,
    )
    assert mask_positions == [pos, -1, -1]
    # Row 0 was masked at pos=2
    assert torch.allclose(masked[0, pos], torch.zeros(embSize))
    assert torch.allclose(targets[0], embedded[0, pos])
    # Rows 1, 2 untouched, targets stay zero
    assert torch.allclose(masked[1], embedded[1])
    assert torch.allclose(masked[2], embedded[2])
    assert torch.allclose(targets[1], torch.zeros(embSize))
    assert torch.allclose(targets[2], torch.zeros(embSize))


def test_expand_masked_batched_rarlm_mirrors_position():
    """RARLM mode: position p = N-1-pos (counting from the end), and
    positions before p are zeroed."""
    import Spaces

    fake = _make_fake_inputspace(where_dim=0, when_dim=0)
    B, nVec, embSize = 1, 5, 2
    embedded = torch.arange(nVec * embSize, dtype=torch.float).reshape(
        1, nVec, embSize)
    sentences = ["a b c d"]  # N = 4
    pos = 0  # -> p = N-1-pos = 3

    masked, _, mask_positions = Spaces.InputSpace.expand_masked_batched(
        fake, embedded, sentences, maskedPrediction="RARLM", pos=pos,
    )
    assert mask_positions == [3]
    assert torch.allclose(masked[0, 3], torch.zeros(embSize))
    # Positions 0..2 are zeroed (previous tokens for reverse AR)
    assert torch.allclose(masked[0, :3], torch.zeros(3, embSize))
    # Position 4 (past N=4, unchanged) keeps its original value
    assert torch.allclose(masked[0, 4], embedded[0, 4])


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
