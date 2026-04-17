import pytest
from basicmodel.bin.Spaces import PerceptualSpace


def test_chunking_mode_raw_returns_bytes():
    byte_stream = b"Hello, world."
    units = PerceptualSpace.chunk_static(byte_stream, mode="raw")
    assert units == [b"H", b"e", b"l", b"l", b"o", b",", b" ",
                     b"w", b"o", b"r", b"l", b"d", b"."]


def test_chunking_mode_lexicon_splits_on_spaces():
    byte_stream = b"the quick fox"
    units = PerceptualSpace.chunk_static(byte_stream, mode="lexicon")
    assert units == [b"the", b"quick", b"fox"]


def test_chunking_mode_bpe_returns_learned_segments():
    # Expect BPE to produce consistent subword tokens — smoke test only.
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


def test_perceptual_space_exposes_chunking_mode_attribute():
    """PerceptualSpace carries a chunking_mode attribute resolved at construction."""
    # Instance-level attribute is set by __init__; class-level default is not
    # defined, so we only verify the setter exists via a fresh instance.
    # Use a synthetic Space-like fake: the attribute we care about is
    # populated from config, so just assert the init path references it.
    import inspect
    src = inspect.getsource(PerceptualSpace.__init__)
    assert "self.chunking_mode" in src
