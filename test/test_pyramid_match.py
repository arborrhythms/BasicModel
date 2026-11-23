"""The reconstruction's error as a spatial pyramid match (`Layers.Error.pyramid`,
round 6d of item 1.7, fifteenth pass; Lazebnik, Schmid and Ponce, 2006):
against a plain reference written with loops over levels and cells, on
drawn cases (the start drawn afresh and printed); nothing for the input
against itself; rising with a misplacement of 1, 4 and 32 places; a right
word at a wrong place below a wrong word; a gradient at level 0 when every
word is misplaced; and the time of one call."""
import os
import time

import torch

from Layers import Error


def reference(put, wanted):
    """``put``, ``wanted`` ``[places, channels]``, one axis: the kernel by
    loops, each level's cells built one by one."""
    n = max(len(put), len(wanted))
    L = max(0, (n - 1).bit_length())
    size = 1 << L
    pad = lambda t: torch.cat((t, t.new_zeros(size - len(t), t.shape[1])))
    put, wanted = pad(put.clamp_min(0)), pad(wanted.clamp_min(0))
    K = 0.
    for level in range(L + 1):
        I = 0.
        for cell in range(1 << level):
            members = [a for a in range(size) if a >> (L - level) == cell]
            I += torch.minimum(put[members].sum(0), wanted[members].sum(0)).sum()
        K += I * (1 / 2 ** L if level == 0 else 1 / 2 ** (L - level + 1))
    return K, wanted.sum()


def error(put, wanted):
    K, total = Error.pyramid_match(put[None], wanted[None])
    return float(1 - K / total)


def words(text, at=0, width=64, channels=64):
    """Each byte's content: a fixed sparse code of its own, 8 of 64 wires
    on, as a part code holds few parts."""
    generator = torch.Generator().manual_seed(7)
    codes = (torch.rand(256, channels, generator=generator).argsort(-1) < 8).float()
    out = torch.zeros(width, channels)
    for i, byte in enumerate(text.encode()):
        if 0 <= at + i < width:
            out[at + i] = codes[byte]
    return out


def test_against_the_reference_on_drawn_cases():
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    for _ in range(20):
        n, m, c = (int(v) for v in torch.randint(1, 40, (3,)))
        put, wanted = torch.rand(n, c) * (torch.rand(n, c) < .5), torch.rand(m, c)
        K, total = Error.pyramid_match(put[None], wanted[None])
        k, t = reference(put, wanted)
        assert abs(float(K[0]) - float(k)) <= 1e-6 * max(1., float(k))
        assert abs(float(total[0]) - float(t)) <= 1e-6 * max(1., float(t))


def test_a_single_cell_receives_the_full_weight():
    wanted = torch.tensor([[[1., 2.]]])
    for put in (wanted, wanted / 2):
        match, total = Error.pyramid_match(put, wanted)
        torch.testing.assert_close(match, put.sum().reshape(1))
        torch.testing.assert_close(total, wanted.sum().reshape(1))


def test_the_error_on_a_sentence():
    own = words('This is Felix.')
    assert error(own, own) == 0.
    moved = [error(words('This is Felix.', k), own) for k in (1, 4, 32)]
    print('the sentence moved by 1, 4, 32:', [round(v, 4) for v in moved])
    assert moved[0] < moved[1] < moved[2]
    one = [error(words('This is ') + words('Felix', 8 + k) + words('.', 13), own) for k in (1, 4, 32)]
    print('"Felix" moved by 1, 4, 32:', [round(v, 4) for v in one])
    assert one[0] < one[1] < one[2]
    wrong = error(words('Here is Felix.'), own)
    moved_this = {k: round(error(words('This', k) + words(' is Felix.', 4), own), 4) for k in (1, 2, 4, 8, 16, 32)}
    print('"This" as "Here" at its place', round(wrong, 4), '| "This" moved by k (k: error):', moved_this)
    # A right word near its place costs less than a wrong word there; far
    # enough it costs as much as content lost (the codes share wires).
    assert moved_this[1] < wrong
    # Every word moved: level 0 alone still has a gradient toward the content.
    put = words('Here is Felix.', 32).requires_grad_(True)
    level0 = torch.minimum(put.sum(0), own.sum(0)).sum()
    gradient, = torch.autograd.grad(level0, put)
    print('every word moved by 32, level 0 gradient norm', round(float(gradient.norm()), 4))
    assert float(gradient.norm()) > 0
    edges = [round(error(words('e', p + 1), words('e', p)), 4) for p in range(16)]
    print('one byte moved by one place, at places 0 to 15:', edges)


def test_the_time_of_one_call():
    for name, places in (('lesson', 64), ('production', 8192)):
        put, wanted = torch.rand(1, places, 128), torch.rand(1, places, 128)
        Error.pyramid_match(put, wanted)
        begin = time.perf_counter()
        for _ in range(10):
            Error.pyramid_match(put, wanted)
        print(f'{name}: {places} places x 128 channels, one call {1000 * (time.perf_counter() - begin) / 10:.2f} ms')
