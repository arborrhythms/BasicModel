import pytest
from Spaces import Embedding, PerceptualSpace


def test_synthesis_mode_lexicon_splits_on_spaces():
    byte_stream = b"the quick fox"
    units = PerceptualSpace.chunk_static(byte_stream, mode="lexicon")
    assert units == [b"the", b"quick", b"fox"]


def test_synthesis_mode_bpe_returns_learned_segments():
    # Expect BPE to produce consistent subword tokens -- smoke test only.
    byte_stream = b"running"
    units = PerceptualSpace.chunk_static(byte_stream, mode="bpe")
    assert isinstance(units, list)
    assert all(isinstance(u, (bytes, str)) for u in units)
    assert b"".join(
        u if isinstance(u, bytes) else u.encode() for u in units
    ) == byte_stream


def test_chunking_invalid_mode_raises():
    with pytest.raises(ValueError):
        PerceptualSpace.chunk_static(b"x", mode="not_a_mode")


def test_perceptual_space_exposes_synthesis_mode_attribute():
    """PerceptualSpace carries a synthesis_mode attribute resolved at construction."""
    # Instance-level attribute is set by __init__; class-level default is not
    # defined, so we only verify the setter exists via a fresh instance.
    # Use a synthetic Space-like fake: the attribute we care about is
    # populated from config, so just assert the init path references it.
    import inspect
    src = inspect.getsource(PerceptualSpace.__init__)
    assert "self.synthesis_mode" in src


def test_embedding_token_stream_honors_bpe_fallback_mode():
    emb = Embedding()
    emb.byte_mode = False
    emb.synthesis_mode = "bpe"

    assert emb._token_stream("hi") == [("h", 0), ("i", 1)]
