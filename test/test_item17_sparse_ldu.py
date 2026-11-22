"""Item 1.7, round 6d, seventh pass, step 1: the sparse LDU (rule 78,
§19.122). Each start is drawn afresh and printed."""
import os
import time

import pytest
import torch

from Layers import InvertibleLinearLayer
from sparse_ldu import unit_apply, unit_solve, longest_chain, ldu_up, ldu_back


def _start():
    seed = int.from_bytes(os.urandom(4), 'little')
    print('start', seed)
    return torch.Generator().manual_seed(seed)


def _triangle(n, count, lower, g):
    """``count`` drawn entries of a strict triangle of ``n``, distinct."""
    i = torch.randint(1, n, (count,), generator=g)
    j = (torch.rand(count, generator=g) * i).long()
    pair = torch.unique(torch.stack((i, j) if lower else (j, i)), dim=1)
    value = (torch.rand(pair.shape[1], generator=g, dtype=torch.float64) - .5) * .6
    return pair[0], pair[1], value


def _dense(n, entries):
    m = torch.eye(n, dtype=torch.float64)
    m[entries[0], entries[1]] += entries[2]
    return m


@pytest.mark.parametrize('n_input,n_output', [(12, 12), (9, 14), (14, 9)])
def test_a_the_way_up_and_back_equal_the_full_layers(n_input, n_output):
    """(a) On the same entries, the way up and the way back equal
    InvertibleLinearLayer's, square and not square (D padded)."""
    g = _start()
    rank = min(n_input, n_output)
    lower, upper = _triangle(n_input, 2 * n_input, True, g), _triangle(n_output, 2 * n_output, False, g)
    d = torch.rand(rank, generator=g, dtype=torch.float64) + .5
    full = InvertibleLinearLayer(n_input, n_output, hasBias=False).double()
    with torch.no_grad():
        full.raw_L.copy_(_dense(n_input, lower) - torch.eye(n_input, dtype=torch.float64))
        full.raw_U.copy_(_dense(n_output, upper) - torch.eye(n_output, dtype=torch.float64))
        full.d.copy_(d)
    x = torch.randn(5, n_input, generator=g, dtype=torch.float64)
    up = ldu_up(x.t(), d, lower, upper, n_output).t()
    torch.testing.assert_close(up, full.forward(x), atol=1e-12, rtol=0)
    y = torch.randn(5, n_output, generator=g, dtype=torch.float64)
    back = ldu_back(y.t(), d, lower, upper, n_input,
                    longest_chain(*lower[:2]), longest_chain(*upper[:2])).t()
    torch.testing.assert_close(back, full.reverse(y), atol=1e-12, rtol=0)
    # Where W has as many outputs as inputs or more, the way back returns the input.
    if n_output >= n_input:
        torch.testing.assert_close(ldu_back(up.t(), d, lower, upper, n_input, longest_chain(*lower[:2]),
                                            longest_chain(*upper[:2])).t(), x, atol=1e-12, rtol=0)


def test_b_the_way_back_is_exact_after_as_many_steps_as_the_longest_chain():
    """(b) ``x <- y - x @ N``, as many times as the longest chain, returns
    the input exactly; one step fewer does not, where there is a chain."""
    g = _start()
    n = 64
    source, target, value = _triangle(n, 3 * n, False, g)
    # Values of size one, so that the longest chain's term is not lost in rounding.
    entries = (source, target, torch.where(value < 0, -1., 1.).to(value))
    chain = longest_chain(source, target)
    x = torch.randn(n, 4, generator=g, dtype=torch.float64)
    y = unit_apply(x, entries)
    back = unit_solve(y, entries, chain)
    error = float((back - x).abs().max())
    short = float((unit_solve(y, entries, chain - 1) - x).abs().max())
    print(dict(entries=len(entries[0]), chain=chain, error=error, one_step_short=short))
    assert error < 1e-12 and chain >= 2 and short > 1e-9


def test_c_an_entry_added_or_removed_leaves_it_invertible_with_nothing_factored_again():
    """(c) An entry appended to, or dropped from, either triangle's list
    leaves the way back exact; nothing is factored again (the lists are the
    factors)."""
    g = _start()
    n = 32
    lower, upper = _triangle(n, 2 * n, True, g), _triangle(n, 2 * n, False, g)
    d = torch.rand(n, generator=g, dtype=torch.float64) + .5
    x = torch.randn(n, 3, generator=g, dtype=torch.float64)

    def round_trip(lower, upper):
        y = ldu_up(x, d, lower, upper, n)
        back = ldu_back(y, d, lower, upper, n, longest_chain(*lower[:2]), longest_chain(*upper[:2]))
        return float((back - x).abs().max())

    errors = {'as drawn': round_trip(lower, upper)}
    added = tuple(torch.cat((t, extra)) for t, extra in zip(upper, (torch.tensor([0]), torch.tensor([n - 1]),
                                                                   torch.tensor([.4], dtype=torch.float64))))
    if not bool(((upper[0] == 0) & (upper[1] == n - 1)).any()):
        errors['an entry added to U'] = round_trip(lower, added)
    keep = torch.arange(len(lower[0])) != int(torch.randint(len(lower[0]), (1,), generator=g))
    errors['an entry removed from L'] = round_trip(tuple(t[keep] for t in lower), upper)
    print(errors)
    assert max(errors.values()) < 1e-12
