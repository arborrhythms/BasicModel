"""Item 1.7, round 6d, seventh pass, step 1: the sparse LDU (rule 78,
§19.122). Each start is drawn afresh and printed."""
import os
import time

import pytest
import torch

from Layers import InvertibleLinearLayer
from sparse_ldu import unit_apply, unit_solve, longest_chain, ldu_up, ldu_back, dense_w


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
    # Values of size about one, so that the longest chain's term is not lost
    # in rounding, and not all of one size, so that no two chains cancel.
    size = .75 + .5 * torch.rand(len(value), generator=g, dtype=value.dtype)
    entries = (source, target, torch.where(value < 0, -size, size))
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


# The order-zero Sigma on the sparse LDU (step 2).

def _dense_read(sigma, lanes, inputs):
    """What the Sigma of order zero reads by its definition, over every
    concept: ``tanh`` of each concept's column of W times its rows' values
    (for less against), W the dense ``L D U`` of the layer's factors."""
    dense = dense_w(sigma).detach()
    signed = torch.zeros(sigma.nInput, *lanes.shape[1:-1])
    signed[inputs] = lanes[..., 0] - lanes[..., 1]
    return torch.tanh(torch.einsum('i...,ic->c...', signed, dense))


@pytest.fixture(scope='module')
def lesson():
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(16)
    yield model._concept_owner()
    model.End()
    patch.undo()


def test_d_the_lessons_sigma_reads_its_entries_and_returns_its_rows(lesson):
    """(d) After the lesson's sixteen sentences the order-zero Sigma is the
    sparse LDU. On the way up it gives the Sigma's values by its definition
    (each concept's entries on its rows, and nothing else); on the way back
    it returns every row that brought a concept, exactly, all at once, where
    a least-squares way back through each row's own entries is exact only
    when every concept reading a row is asked with it."""
    from Layers import SparseSigmaLDU
    sigma, not_ = lesson.order_zero, lesson.order_zero_not
    assert isinstance(sigma, SparseSigmaLDU)
    g = _start()
    n = sigma.nnz
    rows = sorted(set(sigma.entry_input[:n].tolist()))
    lanes = torch.rand(len(rows), 3, 2, generator=g) * .5
    concepts = list(range(sigma.nOutput))
    up = sigma(not_(lanes), rows, concepts).detach()
    torch.testing.assert_close(up, _dense_read(sigma, lanes, rows), atol=1e-6, rtol=0)
    start = time.perf_counter()
    back = sigma.reverse(up, rows).detach()
    seconds = time.perf_counter() - start
    error = float((back - (lanes[..., 0] - lanes[..., 1])).abs().max())
    print(dict(rows=sigma.nInput, concepts=sigma.nOutput, entries=n, slots=sigma.slots,
               diagonal=sum(e >= 0 for e in sigma._diagonal), longest_chain=sigma.longest_chain,
               way_back_error=error, way_back_seconds=round(seconds, 6)))
    assert error < 1e-5


def test_d_a_cycle_goes_in_l_and_every_row_is_returned():
    """(d) A concept that reads, through another concept's row, the row of
    its own slot would close a cycle of U: the entry goes in L, and no
    concept leaves its slot (rule 78, decided at §23.6.12's review). The
    way up is the dense ``L D U``, which has a product of entries; the way
    back returns every row exactly.

    Ported (eighth pass): the seventh pass's test asserted that the concept
    left its slot (5 slots, a chain of 2), that rows 0 and 1 were refused
    and rows 2 and 3 returned."""
    from Layers import NotLayer3, SparseSigmaLDU
    g = _start()
    sigma, not_ = SparseSigmaLDU(4, 4, 16), NotLayer3(4)
    for r in range(4):
        sigma.add(r, r, 1.)
    sigma.add(1, 0, .5)                        # concept 1 reads row 0
    sigma.add(0, 1, .5, negated=True)          # concept 0 reads row 1: a cycle of U
    assert sigma.slots == 4 and sigma.lower_entries == 1
    assert sigma.longest_chain == 1 and sigma.longest_chain_lower == 1
    dense = dense_w(sigma).detach()
    # W at the two entries is as written; the product L d U is W's entry of
    # concept 1 on row 1 (row 1 to slot 0 by L, then slot 0 to concept 1 by U).
    assert sigma.w()[:6].detach().tolist() == [1., 1., 1., 1., .5, -.5]
    product = float(dense[1, 1]) - 1.
    lanes = torch.rand(4, 2, 2, generator=g) * .5
    up = sigma(not_(lanes), range(4), range(4)).detach()
    torch.testing.assert_close(up, _dense_read(sigma, lanes, list(range(4))), atol=1e-6, rtol=0)
    assert sorted(sigma.returned()) == [0, 1, 2, 3]
    truth = lanes[..., 0] - lanes[..., 1]
    back = sigma.reverse(up, range(4)).detach()
    print(dict(product=product, error=float((back - truth).abs().max())))
    assert abs(product + .25) < 1e-6
    torch.testing.assert_close(back, truth, atol=1e-5, rtol=0)


def test_d_an_entry_added_or_removed_needs_nothing_factored_again():
    """(d) Entries added and removed in any order: the way up reads exactly
    the entries left, and the way back returns every row with a concept of
    its own, with no factoring."""
    from Layers import NotLayer3, SparseSigmaLDU
    g = _start()
    P, C = 24, 30
    sigma, not_ = SparseSigmaLDU(P, C, 200), NotLayer3(P)
    for r in range(P):
        sigma.add(r, r, .5 + float(torch.rand(1, generator=g)))
    for c in range(P, C):
        for r in torch.randint(P, (3,), generator=g).tolist():
            if sigma.find(c, r) is None:
                sigma.add(c, r, float(torch.rand(1, generator=g)), negated=bool(torch.rand(1, generator=g) < .3))
    for c in range(P):
        r = int(torch.randint(P, (1,), generator=g))
        if r < c and sigma.find(c, r) is None:
            sigma.add(c, r, .3)                 # an earlier row: by level, no cycle
    lanes = torch.rand(P, 2, 2, generator=g) * .5
    truth = lanes[..., 0] - lanes[..., 1]
    for step in range(4):
        up = sigma(not_(lanes), range(P), range(C)).detach()
        torch.testing.assert_close(up, _dense_read(sigma, lanes, list(range(P))), atol=1e-6, rtol=0)
        kept = sorted(sigma.returned())
        back = sigma.reverse(up, kept).detach()
        torch.testing.assert_close(back, truth[kept], atol=1e-5, rtol=0)
        e = int(torch.randint(sigma.nnz, (1,), generator=g))
        sigma.remove(int(sigma.entry_row[e]), int(sigma.entry_input[e]))
        print(dict(step=step, entries=sigma.nnz, slots=sigma.slots, chain=sigma.longest_chain, returned=len(kept)))


def test_e_the_longest_chain_and_the_way_back_on_the_dictionary():
    """(e) The dictionary's forms (the first 20,000 words, 19,821, in the
    order certificate 46 admits them), each a row that brings its concept:
    the longest chain and the seconds of a way back; and, as a trial, with
    each concept also reading the rows of the stored forms it holds as runs
    (a concept after every row it reads, by level)."""
    import code_trial as ct
    from Layers import NotLayer3, SparseSigmaLDU
    try:
        words = ct.dictionary()
    except OSError:
        pytest.skip('no /usr/share/dict/words on this host')
    g = _start()
    row = {w: i for i, w in enumerate(words)}
    runs = [(row[w], row[w[a:b]]) for w in words for a in range(len(w)) for b in range(a + 2, len(w) + 1)
            if (a, b) != (0, len(w)) and w[a:b] in row]
    P = len(words)
    lanes = torch.rand(P, 8, 2, generator=g) * .5
    truth = lanes[..., 0] - lanes[..., 1]
    report = {}
    for name, extra in (('one concept a row', ()), ('and the runs it holds', runs)):
        start = time.perf_counter()
        sigma, not_ = SparseSigmaLDU(P, P, P + len(extra)), NotLayer3(P)
        for r in range(P):
            sigma.add(r, r, 1.)
        for concept, held in extra:
            sigma.add(concept, held, .1)
        added = time.perf_counter() - start
        up = sigma(not_(lanes), range(P), range(P)).detach()
        start = time.perf_counter()
        back = sigma.reverse(up, range(P)).detach()
        seconds = time.perf_counter() - start
        report[name] = dict(entries=sigma.nnz, chain=sigma.longest_chain, seconds_to_add=round(added, 2),
                            way_back_seconds=round(seconds, 4), error=float((back - truth).abs().max()))
        assert report[name]['error'] < 1e-4
    print(report)


def test_the_first_pis_way_back_is_the_percept_each_row_keeps(lesson):
    """Step 3 (rule 79; §19.121, 1): a row's way back is the percept it
    keeps, its ``.what`` code on two lanes and its ``.where`` and ``.when``
    entries. Read as an occurrence of its own, each nominated row holds
    itself wholly: its read there (``held_at`` and ``place_factor``) is one,
    to within the grading of its code."""
    from types import SimpleNamespace
    from BasicPercepts import held_at, field_factor, WHERE_WIRES, WHEN_WIRES
    basic = lesson.basic_percepts
    rows = sorted(set(lesson.order_zero.entry_input[:lesson.order_zero.nnz].tolist()))
    what, where, when = basic.kept_percept(rows)
    R, width = len(rows), basic.width
    assert what.shape == (R, 2, width) and where.shape == (R, WHERE_WIRES, 2) and when.shape == (R, WHEN_WIRES, 2)
    torch.testing.assert_close(what, basic.bank_codes(rows).to(what))
    # Each kept percept as an occurrence of a table: the what on its lanes,
    # then the where and the when wires.
    table = torch.cat((what.transpose(1, 2), where, when), 1)[None]            # [1, R, columns, 2]
    bank = SimpleNamespace(codes=what, identity_width=basic.identity_width, width=width)
    part, whole, _added = held_at(bank, table)
    place = field_factor(table[..., width:width + WHERE_WIRES, :], where) * \
        field_factor(table[..., width + WHERE_WIRES:, :], when)
    own = (part * whole * place)[torch.arange(R), 0, torch.arange(R)]
    print(dict(rows=R, own_read=own.tolist()))
    # A code is graded, so its own read is one to within its grading.
    torch.testing.assert_close(own, torch.ones(R).to(own), atol=1e-3, rtol=0)
