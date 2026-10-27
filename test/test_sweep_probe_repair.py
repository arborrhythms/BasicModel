"""The diagnostic keeps every cosine without a vocabulary-sized broadcast."""
import torch
from decoder_margin_probe import decoder_cosines


def test_decoder_cosines_match_broadcast_for_zero_and_small_rows():
    leaves = torch.arange(2*3*8, dtype=torch.float32).reshape(2, 3, 8) - 11.
    codes = torch.arange(2*5*8, dtype=torch.float32).reshape(2, 5, 8) - 23.
    leaves[0, 0].zero_(); codes[1, 1].zero_()
    leaves[0, 2].mul_(1e-11); codes[1, 4].mul_(1e-11)
    before = leaves.clone(), codes.clone()
    expected = torch.nn.functional.cosine_similarity(leaves[:, :, None], codes[:, None], dim=-1)
    actual = decoder_cosines(leaves, codes)
    assert actual.shape == (2, 3, 5)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(leaves, before[0], rtol=0, atol=0)
    torch.testing.assert_close(codes, before[1], rtol=0, atol=0)
