import torch
import torch.nn.functional as F

from workarounds import Workarounds


def test_adaptive_avg_pool1d_divisible_matches_torch():
    u = torch.arange(24, dtype=torch.float32).reshape(2, 1, 12)

    actual = Workarounds.adaptive_avg_pool1d(u, 3)
    expected = F.adaptive_avg_pool1d(u, 3)

    torch.testing.assert_close(actual, expected)


def test_adaptive_avg_pool1d_tuple_output_matches_torch():
    u = torch.arange(24, dtype=torch.float32).reshape(2, 1, 12)

    actual = Workarounds.adaptive_avg_pool1d(u, (4,))
    expected = F.adaptive_avg_pool1d(u, (4,))

    torch.testing.assert_close(actual, expected)


def test_adaptive_avg_pool1d_nondivisible_matches_torch():
    u = torch.arange(20, dtype=torch.float32).reshape(2, 1, 10)

    actual = Workarounds.adaptive_avg_pool1d(u, 4)
    expected = F.adaptive_avg_pool1d(u, 4)

    torch.testing.assert_close(actual, expected)
