"""The table of forms agrees with a host dictionary of spellings and runs on the device without host reads."""
import copy
import math
import os
import random
import time
from collections import Counter

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from FormLinks import FormLinks, HALF, LEAVES, UNUSED_KEY

DEVICES = ['cpu', 'mps']
LONGEST = 8


def _device(name):
    if name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    return torch.device(name)


def _vocabulary(rng):
    words = [bytes(rng.choices(range(97, 123), k=rng.randint(1, 12))) for _ in range(60)]
    words += [b'banana', b'aaaa', b'mississippi', b'\\n', b'\\x00', b'\x00\x01\x00',
              'naïve'.encode(), '日本語'.encode(), '\U0001f600!'.encode(), b'\xff\xfe']
    return list(dict.fromkeys(words))


def _pack(units, device):
    width = max([1] + [len(u) for u in units])
    data = torch.tensor([list(u) + [0] * (width - len(u)) for u in units], dtype=torch.uint8)
    return data.to(device), torch.tensor([len(u) for u in units], device=device)


def _host_spellings(table):
    """Spelling of every row in use, by the host lists of links."""
    left, right = table.left.tolist(), table.right.tolist()
    spelled = [bytes([b]) for b in range(LEAVES)]
    for r in range(LEAVES, int(table.used)):
        spelled.append(spelled[left[r]] + spelled[right[r]][-1:])
    return spelled


def _table(device, rng, capacity=8192, threshold=0):
    table = FormLinks(capacity, LONGEST, threshold).to(device)
    words = _vocabulary(rng)
    half = len(words) // 2
    for batch in (words[:half], words[half:]):
        table.insert(*_pack(batch, device))
    return table, words


def _sentences(rng, words, count=12):
    out = [b'', b'banana bananas', 'naïve 日本 \\n'.encode()]
    for _ in range(count):
        units = [rng.choice(words) if rng.random() > .2
                 else bytes(rng.choices(range(256), k=rng.randint(1, 6))) for _ in range(rng.randint(1, 8))]
        out.append(b' '.join(units))
    return out


@pytest.mark.parametrize('device', DEVICES)
def test_rows_are_every_substring_once_with_parts_below_wholes(device):
    table, words = _table(_device(device), random.Random(0))
    spelled = _host_spellings(table)
    wanted = {w[i:j] for w in words for i in range(len(w)) for j in range(i + 2, len(w) + 1)}
    assert len(spelled) - LEAVES == len(wanted) == len(set(spelled[LEAVES:]))
    assert set(spelled[LEAVES:]) == wanted
    rows = torch.arange(LEAVES, int(table.used))
    left, right, length = (t.cpu()[rows].long() for t in (table.left, table.right, table.length))
    assert bool(((left >= 0) & (left < rows) & (right >= 0) & (right < rows)).all())
    assert bool((length == table.length.cpu()[left] + 1).all())
    assert length.tolist() == [len(spelled[r]) for r in rows.tolist()]


@pytest.mark.parametrize('device', DEVICES)
def test_recognise_equals_the_host_dictionary(device):
    rng = random.Random(1)
    table, words = _table(_device(device), rng)
    row_of = {s: r for r, s in enumerate(_host_spellings(table))}
    sentences = _sentences(rng, words)
    data, lengths = _pack(sentences, device)
    rows, best = table.recognise(data, lengths)
    assert rows.shape == (len(sentences), LONGEST, data.shape[1])
    rows, best = rows.cpu().tolist(), best.cpu().tolist()
    for b, raw in enumerate(sentences):
        for i in range(data.shape[1]):
            longest = -1
            for n in range(1, LONGEST + 1):
                expected = row_of.get(raw[i:i + n], -1) if i + n <= len(raw) else -1
                assert rows[b][n - 1][i] == expected, (raw, i, n)
                longest = expected if expected >= 0 else longest
            assert best[b][i] == longest


@pytest.mark.parametrize('device', DEVICES)
def test_spell_equals_the_host_spellings(device):
    rng = random.Random(2)
    table, _ = _table(_device(device), rng)
    spelled = _host_spellings(table)
    picked = rng.sample(range(len(spelled)), 200) + [-1, 0, 255]
    data, lengths = table.spell(torch.tensor(picked, device=device), 12)
    for r, row, n in zip(picked, data.cpu().tolist(), lengths.cpu().tolist()):
        want = b'' if r < 0 else spelled[r]
        assert n == len(want)
        assert row == list(want) + [-1] * (12 - len(want))


@pytest.mark.parametrize('device', DEVICES)
def test_tally_counts_known_units_and_admission_is_idempotent(device):
    dev = _device(device)
    rng = random.Random(3)
    table, words = _table(dev, rng)
    row_of = {s: r for r, s in enumerate(_host_spellings(table))}
    used = int(table.used)
    units = [b'banana', b'banana', b'zzzzqqqq', b'', b'a', '日本語'.encode()]
    rows = table.tally(*_pack(units, dev)).cpu().tolist()
    assert rows == [row_of.get(u, -1) if u else -1 for u in units]
    count = table.count.cpu()
    assert count[row_of[b'banana']] == 2 and count[row_of[b'a']] == 1 and int(count.sum()) == 4
    again = table.insert(*_pack(words, dev)).cpu().tolist()
    assert int(table.used) == used and again == [row_of[w] for w in words]


def test_a_full_table_raises():
    table = FormLinks(LEAVES + 3, LONGEST, 0)
    table.insert(*_pack([b'ab', b'bc'], 'cpu'))
    with pytest.raises(RuntimeError, match='FormLinks full'):
        table.insert(*_pack([b'abcd'], 'cpu'))


def test_state_dict_restores_the_table():
    rng = random.Random(4)
    table, words = _table('cpu', rng)
    table.close(*_pack(words[:5], 'cpu'))
    state = table.state_dict()
    assert set(state) == {'left', 'right', 'length', 'count', 'used', 'pair_keys', 'pair_rows', 'code',
                          'part_sum', 'part_count', 'border', 'whole_sum', 'whole_count', 'mean', 'log_var'}
    restored = FormLinks(table.capacity, LONGEST, 0)
    restored.load_state_dict(state)
    data, lengths = _pack(_sentences(rng, words), 'cpu')
    for got, want in zip(restored.recognise(data, lengths), table.recognise(data, lengths)):
        assert torch.equal(got, want)
    assert torch.equal(restored.count, table.count) and int(restored.used) == int(table.used)
    names = ('code', 'part_sum', 'part_count', 'border', 'whole_sum', 'whole_count', 'pair_keys', 'pair_rows',
             'mean', 'log_var')
    for name in names:
        assert torch.equal(getattr(restored, name), getattr(table, name)), name
    noise = table.order_noise(64, torch.Generator().manual_seed(0))
    assert torch.equal(restored.order_loss(64, noise=noise), table.order_loss(64, noise=noise))
    table.close(*_pack(words[5:9] + [b'qqqzz'], 'cpu'))
    restored.close(*_pack(words[5:9] + [b'qqqzz'], 'cpu'))
    for name in names:
        assert torch.equal(getattr(restored, name), getattr(table, name)), name


class _NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, 'host read'
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('device', DEVICES)
def test_recognise_and_spell_capture_whole_and_read_nothing_on_the_host(device, monkeypatch):
    dev = _device(device)
    rng = random.Random(5)
    table, words = _table(dev, rng, threshold=2)
    table.tally(*_pack(words + words[::3], dev))
    data, lengths = _pack(_sentences(rng, words), dev)
    rows = torch.tensor(rng.sample(range(int(table.used)), 64) + [-1], device=dev)
    expected = table.recognise(data, lengths), table.spell(rows)

    def forbidden(*args, **kwargs):
        raise AssertionError('read a tensor on the host')

    with monkeypatch.context() as patch, _NoHostScalar():
        for method in ('item', 'tolist', '__bool__'):
            patch.setattr(torch.Tensor, method, forbidden)
        eager = table.recognise(data, lengths), table.spell(rows)
    torch._dynamo.reset()
    recognise = torch.compile(table.recognise, fullgraph=True, backend='eager')
    spell = torch.compile(table.spell, fullgraph=True, backend='eager')
    captured = recognise(data, lengths), spell(rows)
    for got in (eager, captured):
        for g, w in zip(got[0] + got[1], expected[0] + expected[1]):
            assert torch.equal(g, w)


def _host_admission(batches):
    """Links and lengths of every row by the host: each length's missing substrings in order of their links."""
    row_of = {bytes([b]): b for b in range(LEAVES)}
    left, right, length = [], [], []
    for batch in batches:
        for n in range(2, max(len(u) for u in batch) + 1):
            pairs = sorted({(row_of[s[:-1]], row_of[s[1:]]) for u in batch
                            for s in (u[i:i + n] for i in range(len(u) - n + 1)) if s not in row_of})
            for lo, hi in pairs:
                spelled = [k for k, r in row_of.items() if r == lo][0] + [k for k, r in row_of.items() if r == hi][0][-1:]
                row_of[spelled] = LEAVES + len(left)
                left.append(lo), right.append(hi), length.append(n)
    return left, right, length


def test_insert_equals_the_host_admission():
    rng = random.Random(12)
    words = _vocabulary(rng) + [b'abab', b'aaaa', b'abcabcabc', b'aabaa']
    batches = [words[:30], words[30:], words[::7]]
    table = FormLinks(8192, LONGEST)
    for batch in batches:
        table.insert(*_pack(batch, 'cpu'))
    left, right, length = _host_admission(batches)
    used = int(table.used)
    assert used == LEAVES + len(left)
    assert table.left[LEAVES:used].tolist() == left and table.right[LEAVES:used].tolist() == right
    assert table.length[LEAVES:used].tolist() == length
    assert bool((table.left[used:] == -1).all() and (table.length[used:] == 0).all())
    keys = table.pair_keys[:used - LEAVES]
    assert bool((keys[1:] > keys[:-1]).all()) and bool((table.pair_keys[used - LEAVES:] == UNUSED_KEY).all())



def test_a_static_bound_on_the_units_changes_nothing():
    """Reading with the levels bounded at the longest unit makes the same table; a unit above the bound raises."""
    rng = random.Random(15)
    words = _vocabulary(rng) + [b'abab', b'aaaa', b'abcabcabc', b'aabaa']
    full, bounded = FormLinks(8192, 32), FormLinks(8192, 32)
    for batch in (words[:30], words[30:], words[::7]):
        data, lengths = _pack(batch, 'cpu')
        data = torch.nn.functional.pad(data, (0, 32 - data.shape[1]))
        top = 1 << (int(lengths.max()) - 1).bit_length()
        assert torch.equal(full.read(data, lengths), bounded.read(data, lengths, longest=top))
        rows = torch.arange(LEAVES, int(full.used))
        full.recentre_rows(rows[:40], data, lengths)
        bounded.recentre_rows(rows[:40], data, lengths, top)
    for name in ('code', 'part_sum', 'whole_sum', 'whole_count', 'count', 'pair_keys', 'left', 'right', 'border', 'used'):
        assert torch.equal(getattr(full, name), getattr(bounded, name)), name
    with pytest.raises(RuntimeError, match='longer than its bound'):
        bounded.read(*_pack([b'abcdefgh'], 'cpu'), longest=4)

HARD = [b'abab', b'aaaa', b'banana', b'mississippi', b'abcabcabc', b'aabaa', b'aaa', b'abaaba', b'aabaab']


def _brute_part_sums(table, whole, part):
    q = table._quantise(table.code[:, :HALF])
    total = torch.zeros_like(table.part_sum).index_add_(0, whole, q[part])
    count = torch.zeros_like(table.part_count).index_add_(0, whole, torch.ones_like(whole, dtype=torch.int32))
    return total, count


def test_recursive_part_sums_equal_the_distinct_parts():
    words = _dictionary(600) + HARD
    table = FormLinks(65536, LONGEST)
    for i in range(0, len(words), 150):
        table.insert(*_pack(words[i:i + 150], 'cpu'))
    spelled, row_of, whole, part = _part_pairs(table)
    total, count = _brute_part_sums(table, whole, part)
    assert torch.equal(table.part_count, count) and torch.equal(table.part_sum, total)
    bordered = [s for s in HARD if int(table.border[row_of[s]]) >= 0]
    assert spelled[int(table.border[row_of[b'abab']])] == b'ab'
    assert spelled[int(table.border[row_of[b'aaaa']])] == b'aaa'
    assert int(table.border[row_of[b'banana']]) == -1 and int(table.border[row_of[b'aabaa']]) == row_of[b'aa']
    share = float((table.border[LEAVES:int(table.used)] >= 0).float().mean())
    print(f'\nforms with a border outside the middle: {share:.4f}; hard cases with one: {bordered}')


def test_a_form_needing_two_borders_raises(monkeypatch):
    table = FormLinks(1024, LONGEST)
    data, lengths = _pack([b'aaa'], 'cpu')
    monkeypatch.setattr(FormLinks, '_next_same', lambda self, rows, k: torch.full_like(rows, rows.shape[1]))
    with pytest.raises(RuntimeError, match='two borders'):
        table.insert(data, lengths)


def test_the_kept_part_centroid_against_the_distinct_parts_after_a_stream():
    words = _dictionary(800)
    rng = random.Random(13)
    table = FormLinks(65536, LONGEST)
    for _ in range(12):
        last = rng.sample(words, 200)
        table.close(*_pack(last, 'cpu'))
    spelled, row_of, whole, part = _part_pairs(table)
    total, count = _brute_part_sums(table, whole, part)
    assert torch.equal(table.part_count, count)
    rows = torch.arange(LEAVES, int(table.used))
    gap = ((table.part_sum[rows] - total[rows]).double().abs() / (count[rows, None].double() * 2.0 ** 32)).amax(1)
    read = torch.tensor(sorted({row_of[u[i:j]] for u in last for i in range(len(u)) for j in range(i + 2, len(u) + 1)}))
    print(f'\nkept part centroid against the distinct parts, {len(rows)} forms: max {float(gap.max()):.2e}, '
          f'mean {float(gap.mean()):.2e}, stale {int((gap > 0).sum())}; forms of the last reading: max '
          f'{float(gap[read - LEAVES].max()):.1e}')
    assert torch.equal(table.part_sum[read], total[read])


@pytest.mark.parametrize('device', DEVICES)
def test_insert_captures_whole_and_reads_nothing_on_the_host(device, monkeypatch):
    dev = _device(device)
    rng = random.Random(14)
    table, words = _table(dev, rng)
    batch = _pack(words[::3] + HARD + [b'zzyzx', b'qqq'], dev)
    eager, captured = copy.deepcopy(table), copy.deepcopy(table)

    def forbidden(*args, **kwargs):
        raise AssertionError('read a tensor on the host')

    with monkeypatch.context() as patch, _NoHostScalar():
        for method in ('item', 'tolist', '__bool__'):
            patch.setattr(torch.Tensor, method, forbidden)
        want = eager.insert(*batch)
    torch._dynamo.reset()
    got = torch.compile(captured.insert, fullgraph=True, backend='eager')(*batch)
    assert torch.equal(got, want)
    for name, buffer in list(eager.named_buffers()) + list(eager.named_parameters()):
        assert torch.equal(getattr(captured, name), buffer), name


def _host_cover(raw, spans, promoted, parts):
    tiles = []
    for start, end in spans:
        out, i = [], start
        while i < end:
            n = max([1] + [m for m in range(2, min(LONGEST, end - i) + 1) if raw[i:i + m] in promoted])
            out.append((i, i + n))
            i += n
        tiles.append(out)
    return tiles


@pytest.mark.parametrize('device', DEVICES)
def test_cover_equals_the_host_tiling_and_captures(device):
    dev = _device(device)
    rng = random.Random(15)
    table, words = _table(dev, rng, threshold=2)
    table.tally(*_pack(words + words[::2] + [b'aa', b'aa', b'ana', b'ana'], dev))
    spelled = _host_spellings(table)
    count = table.count.cpu().tolist()
    promoted = {spelled[r] for r in range(LEAVES, int(table.used)) if count[r] >= 2}
    lines, spans = [], []
    for _ in range(10):
        units = [rng.choice(words) if rng.random() > .3 else bytes(rng.choices(b'abnaz\xff', k=rng.randint(1, 14)))
                 for _ in range(rng.randint(1, 6))]
        raw, at, line_spans = b'', 0, []
        for u in units:
            line_spans.append((len(raw), len(raw) + len(u)))
            raw += u + b' '
        lines.append(raw)
        spans.append(line_spans)
    width = max(len(s) for s in spans)
    data, _ = _pack(lines, dev)
    unit_spans = torch.tensor([s + [(0, 0)] * (width - len(s)) for s in spans], device=dev)
    part, extent, counts = table.cover(data, unit_spans)
    torch._dynamo.reset()
    captured = torch.compile(table.cover, fullgraph=True, backend='eager')(data, unit_spans)
    for g, w in zip(captured, (part, extent, counts)):
        assert torch.equal(g, w)
    row_of = {s: r for r, s in enumerate(spelled)}
    part, extent, counts = part.cpu().tolist(), extent.cpu().tolist(), counts.cpu().tolist()
    for b, raw in enumerate(lines):
        for w, tiles in enumerate(_host_cover(raw, spans[b], promoted, LONGEST) + [[]] * (width - len(spans[b]))):
            assert counts[b][w] == len(tiles)
            shown = tiles[:LONGEST]
            assert extent[b][w] == [list(t) for t in shown] + [[-1, -1]] * (LONGEST - len(shown))
            assert part[b][w] == [row_of[raw[i:j]] for i, j in shown] + [-1] * (LONGEST - len(shown))


# -- counting and codes -------------------------------------------------------

def test_a_row_is_a_form_only_once_counted_to_the_threshold():
    table = FormLinks(1024, LONGEST, 2)
    data, lengths = _pack([b'the cat'], 'cpu')
    unit = _pack([b'cat'], 'cpu')
    table.close(*unit)
    row = int(table._unit_rows(*unit)[0])
    assert row >= LEAVES and int(table.count[row]) == 1
    assert table.rows_of(*unit).tolist() == [-1]
    rows, best = table.recognise(data, lengths)
    assert bool((best[0, :7] < LEAVES).all()), 'below the threshold the parts are used'
    assert bool((best[0, :7] == data[0, :7].long()).all()), 'leaves always'
    table.close(*unit)
    assert table.rows_of(*unit).tolist() == [row] and int(table.count[row]) == 2
    rows, best = table.recognise(data, lengths)
    assert int(best[0, 4]) == row and int(rows[0, 2, 4]) == row
    at = table._unit_rows(*_pack([b'at'], 'cpu'))
    assert int(table.count[at]) == 0 and int(rows[0, 1, 5]) == -1, 'a part read only inside a unit is not counted'


READINGS = 8  # times each dictionary word is read; the codes settle by readings, not sweeps


def _dictionary(count):
    path = '/usr/share/dict/words'
    if not os.path.exists(path):
        pytest.skip('requires /usr/share/dict/words')
    with open(path, 'rb') as handle:
        words = sorted({w.strip() for w in handle if 0 < len(w.strip()) <= 24})
    return random.Random(7).sample(words, count)


def _read(table, stream, device, batch=500):
    for i in range(0, len(stream), batch):
        table.close(*_pack(stream[i:i + batch], device))


def _dictionary_table(device, words, readings=READINGS, capacity=131072):
    table = FormLinks(capacity, LONGEST, 2).to(device)
    rng = random.Random(8)
    for _ in range(readings):
        order = list(words)
        rng.shuffle(order)
        _read(table, order, device)
    return table


def _part_pairs(table):
    """Spellings, a row by spelling, and every (form, distinct proper substring) pair by the host."""
    spelled = _host_spellings(table)
    row_of = {s: r for r, s in enumerate(spelled)}
    whole, part = [], []
    for r in range(LEAVES, len(spelled)):
        s = spelled[r]
        for p in {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)} - {s}:
            whole.append(r)
            part.append(row_of[p])
    return spelled, row_of, torch.tensor(whole), torch.tensor(part)


@pytest.fixture(scope='module')
def dictionary():
    words = _dictionary(3000)
    table = _dictionary_table('cpu', words)
    return words, table, _part_pairs(table)


def test_every_code_is_distinct_and_wholes_lie_above_their_parts(dictionary):
    _, table, (spelled, _, whole, part) = dictionary
    used = int(table.used)
    code = table.code[:, :HALF]
    assert torch.unique(code[:used], dim=0).shape[0] == used
    assert bool((table.code[:, HALF:] == 0).all()), 'the wholes half is not written here'
    above = (code[whole] >= code[part]).all(1)
    share = float(above.float().mean())
    kinds = Counter((len(spelled[w]), len(spelled[p])) for w, p in zip(whole[~above].tolist(), part[~above].tolist()))
    print(f'\nwhole >= part everywhere: {share:.5f} of {len(whole)} pairs; '
          f'failing (whole, part) lengths: {kinds.most_common(8)}')
    assert share >= .99


def _agreement(code, items, row_of):
    """Mean cosine of mean-removed codes by shared distinct substrings, buckets 0, 2, 4, 6+."""
    centred = code - code.mean(1, keepdim=True)
    centred = centred / centred.norm(dim=1, keepdim=True).clamp_min(1e-12)
    subs = [{a[p:q] for p in range(len(a)) for q in range(p + 1, len(a) + 1)} for a in items]
    rows = torch.tensor([row_of[a] for a in items])
    cosine = (centred[rows] @ centred[rows].T).tolist()
    buckets = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            shared = len(subs[i] & subs[j])
            buckets.setdefault(min(shared // 2 * 2, 6), []).append(cosine[i][j])
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def test_direction_agrees_more_as_more_substrings_are_shared(dictionary):
    words, table, (spelled, row_of, _, _) = dictionary
    rng = random.Random(9)
    code = table.code[:, :HALF]
    for label, items in (('whole words', rng.sample(words, 400)),
                         ('forms of 3 to 6', rng.sample([s for s in spelled if 3 <= len(s) <= 6], 400))):
        means = _agreement(code, items, row_of)
        print(f'\n{label}: ' + ', '.join(f'{k}{"+" if k == 6 else ""}: {v:.3f}' for k, v in means.items()))
        values = [means[k] for k in (0, 2, 4, 6)]
        assert values == sorted(values) and values[0] < values[-1], (label, means)


def _brute_whole_sums(table, whole, part):
    q = table._quantise(table.code[:, :HALF])
    total = torch.zeros_like(table.whole_sum).index_add_(0, part, q[whole])
    count = torch.zeros_like(table.whole_count).index_add_(0, part, torch.ones_like(part, dtype=torch.int32))
    return total, count


def test_running_whole_sums_equal_the_gathered_ones():
    rng = random.Random(10)
    table, words = _table('cpu', rng)
    for _ in range(4):
        table.close(*_pack(rng.sample(words, 20), 'cpu'))
        table.insert(*_pack([bytes(rng.choices(range(97, 123), k=rng.randint(2, 9))) for _ in range(6)], 'cpu'))
    _, _, whole, part = _part_pairs(table)
    total, count = _brute_whole_sums(table, whole, part)
    assert torch.equal(table.whole_count, count) and torch.equal(table.whole_sum, total)
    held = count > 0
    gathered = total[held].double() / (count[held, None].double() * 2.0 ** 32)
    assert torch.allclose(table._whole_mean(torch.nonzero(held)[:, 0]).double(), gathered, atol=1e-6)


def test_running_whole_sums_equal_the_gathered_ones_on_the_dictionary(dictionary):
    _, table, (_, _, whole, part) = dictionary
    total, count = _brute_whole_sums(table, whole, part)
    assert torch.equal(table.whole_count, count) and torch.equal(table.whole_sum, total)


def test_room_above_a_word_by_its_readings():
    words = _dictionary(1000)
    rng = random.Random(11)
    stream = [w for rank, w in enumerate(words, 1) for _ in range(-(-300 // rank))]
    rng.shuffle(stream)
    table = FormLinks(65536, LONGEST, 2)
    _read(table, stream, 'cpu', batch=250)
    rows = table._unit_rows(*_pack(words, 'cpu'))
    room = 1 - table.code[rows, :HALF].mean(1)
    read = table.count[rows]
    table_of = {}
    for lo, hi in ((1, 1), (2, 5), (6, 30), (31, 10 ** 6)):
        m = (read >= lo) & (read <= hi)
        table_of[f'{lo}-{hi}'] = (float(room[m].mean()), int(m.sum()))
    print(f'\n{len(stream)} readings; room above a word by its readings: '
          + ', '.join(f'{k}: {v:.3f} ({n})' for k, (v, n) in table_of.items()))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_cpu_and_metal_make_equal_codes(dictionary):
    words = dictionary[0][:1000]
    metal = _dictionary_table('mps', words, readings=3)
    cpu = _dictionary_table('cpu', words, readings=3)
    assert int(metal.used) == int(cpu.used) and torch.equal(metal.left.cpu(), cpu.left)
    for name in ('whole_count', 'part_count', 'border', 'pair_keys', 'pair_rows'):
        assert torch.equal(getattr(metal, name).cpu(), getattr(cpu, name)), name
    assert (metal.code.cpu() - cpu.code).abs().max() <= 1e-5
    for name in ('part_sum', 'whole_sum'):
        gap = (getattr(metal, name).cpu() - getattr(cpu, name)).abs().max() / 2.0 ** 32
        assert gap <= 1e-5 * max(1, int(cpu.whole_count.max())), (name, float(gap))


@pytest.mark.parametrize('device', DEVICES)
def test_recentre_rows_moves_only_the_rows_given_and_keeps_the_whole_sums(device):
    dev = _device(device)
    rng = random.Random(16)
    table, words = _table(dev, rng)
    table.close(*_pack(words[:20], dev))
    spelled = _host_spellings(table)
    row_of = {s: r for r, s in enumerate(spelled)}
    given = [row_of[w] for w in words[:6] if len(w) > 1] + [row_of[b'an'], row_of[b'a'], -1]
    rows = torch.tensor(given + given[:3], device=dev)
    before = table.code.clone()
    captured = copy.deepcopy(table)
    data = _pack(words[:6] + [b'an'], dev)
    table.recentre_rows(rows, *data)
    moved = (table.code != before).any(1).nonzero().flatten().tolist()
    assert set(moved) <= {r for r in given if r >= LEAVES} and moved, 'only the rows given move; bytes never'
    _, _, whole, part = _part_pairs(table)
    total, count = _brute_whole_sums(table, whole.to(dev), part.to(dev))
    assert torch.equal(table.whole_count, count) and torch.equal(table.whole_sum, total)
    torch._dynamo.reset()
    torch.compile(captured.recentre_rows, fullgraph=True, backend='eager')(rows, *data)
    for name, buffer in table.named_buffers():
        assert torch.equal(getattr(captured, name), buffer), name


@pytest.mark.parametrize('device', DEVICES)
def test_sweep_leaves_no_stale_sum_and_keeps_links_and_counts(device):
    dev = _device(device)
    rng = random.Random(17)
    table, words = _table(dev, rng)
    for _ in range(4):
        table.close(*_pack(rng.sample(words, 12), dev))
        read = _pack(rng.sample(words, 8), dev)
        table.recentre_rows(table._unit_rows(*read), *read)
    _, _, whole, part = _part_pairs(table)
    whole, part = whole.to(dev), part.to(dev)
    stale = not torch.equal(table.part_sum, _brute_part_sums(table, whole, part)[0])
    kept = {name: getattr(table, name).clone() for name in ('left', 'right', 'length', 'count', 'used', 'border',
                                                           'pair_keys', 'pair_rows', 'code')}
    for _ in range(2):
        table.sweep(chunk=16)
        total, count = _brute_part_sums(table, whole, part)
        assert torch.equal(table.part_count, count) and torch.equal(table.part_sum, total)
        total, count = _brute_whole_sums(table, whole, part)
        assert torch.equal(table.whole_count, count) and torch.equal(table.whole_sum, total)
        for name, value in kept.items():
            if name != 'code':
                assert torch.equal(getattr(table, name), value), name
    assert stale and torch.equal(table.code[:LEAVES], kept['code'][:LEAVES]), 'stale before; bytes never move'


def test_kl_closed_form_equals_torch_distributions():
    g = torch.Generator().manual_seed(0)
    mf, mg = torch.randn(50, 64, generator=g), torch.randn(50, 64, generator=g)
    lf, lg = torch.randn(50, 64, generator=g) - 3, torch.randn(50, 64, generator=g) - 2
    D = torch.distributions
    want = D.kl_divergence(D.Independent(D.Normal(mf.double(), (lf.double() / 2).exp()), 1),
                           D.Independent(D.Normal(mg.double(), (lg.double() / 2).exp()), 1))
    assert torch.allclose(FormLinks.kl(mf.double(), lf.double(), mg.double(), lg.double()), want, rtol=1e-10)


def test_a_new_form_takes_the_mixture_moments_of_its_links_and_holds_them():
    table = FormLinks(1024, LONGEST, 0, margin=2000., gamma=500., start_var=5e-5, inside_at=1000.)
    row = int(table.insert(*_pack([b'ab'], 'cpu'))[0])
    a, b = table.mean.detach()[97], table.mean.detach()[98]
    var = table.log_var.detach()[97].exp()
    assert torch.allclose(table.mean.detach()[row], (a + b) / 2)
    assert torch.allclose(table.log_var.detach()[row].exp(), var + ((a - b) / 2).square())
    part, whole = torch.tensor([97, 98]), torch.tensor([row, row])
    d, inside = table.inside(part, whole)
    back, back_inside = table.inside(whole, part)
    assert bool((d == 0).all() & inside.all()), d
    assert bool((back > table.margin).all() & ~back_inside.any()), back
    untouched = table.mean.detach()[row + 1:].abs().sum() + (table.log_var.detach()[row + 1:] - math.log(5e-5)).abs().sum()
    assert float(untouched) == 0


def _density_table(device, mix=(1, 1, 1)):
    table = FormLinks(8192, LONGEST, 0, order_mix=mix)
    table.insert(*_pack(_vocabulary(random.Random(20)) + HARD, 'cpu'))
    return table.to(device)


@pytest.mark.parametrize('device', DEVICES)
def test_order_loss_captures_whole_and_reads_nothing_on_the_host(device, monkeypatch):
    dev = _device(device)
    table = _density_table(dev)
    noise = table.order_noise(512, torch.Generator(device=dev).manual_seed(1))
    want = table.order_loss(512, noise=noise)

    def forbidden(*args, **kwargs):
        raise AssertionError('read a tensor on the host')

    with monkeypatch.context() as patch, _NoHostScalar():
        for method in ('item', 'tolist', '__bool__'):
            patch.setattr(torch.Tensor, method, forbidden)
        eager = table.order_loss(512, noise=noise)
    torch._dynamo.reset()
    got = torch.compile(table.order_loss, fullgraph=True, backend='eager')(512, noise=noise)
    assert torch.equal(eager, want) and torch.allclose(got, want) and float(want.detach()) > 0
    got.backward()
    assert table.mean.grad is not None and table.log_var.grad is not None


def test_gradients_reach_only_the_rows_sampled():
    table = _density_table('cpu', mix=(0, 1, 1))
    spelled = _host_spellings(table)
    row_of = {s: r for r, s in enumerate(spelled)}
    word = b'mississippi'
    allowed = {row_of[word[i:j]] for i in range(len(word)) for j in range(i + 1, len(word) + 1)}
    assert table.mean.grad is None and table.log_var.grad is None
    table.order_loss(256, generator=torch.Generator().manual_seed(2), wholes=torch.tensor([row_of[word]])).backward()
    touched = set(((table.mean.grad != 0).any(1) | (table.log_var.grad != 0).any(1)).nonzero().flatten().tolist())
    assert touched and touched <= allowed, touched - allowed


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_cpu_and_metal_agree_on_the_order_loss():
    cpu = _density_table('cpu')
    metal = copy.deepcopy(cpu).to('mps')
    for step in range(3):
        noise = cpu.order_noise(1024, torch.Generator().manual_seed(step))
        a, b = cpu.order_loss(1024, noise=noise), metal.order_loss(1024, noise=noise.to('mps'))
        assert torch.allclose(b.cpu(), a, rtol=1e-4), (float(a), float(b))
        with torch.no_grad():
            for m in (cpu, metal):
                m.mean.add_(torch.randn(m.mean.shape, generator=torch.Generator().manual_seed(9)).to(m.mean.device) * .05)


# -- density order embedding --------------------------------------------------

def _eval_pairs(spelled, row_of, wholes, rng, count_rows):
    """Balanced pairs over ``wholes``: true (part, whole), and as many false, a third each reversed, sibling, random."""
    true, rev, sib, ran = [], [], [], []
    for w in wholes:
        s = spelled[w]
        subs = sorted({s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)} - {s})
        p = rng.choice(subs)
        true.append((row_of[p], w))
        rev.append((w, row_of[p]))
        for _ in range(50):
            a, b = rng.choice(subs), rng.choice(subs)
            if a not in b and b not in a:
                sib.append((row_of[a], row_of[b]))
                break
        while True:
            r = rng.randrange(count_rows)
            if spelled[r] not in s:
                ran.append((r, w))
                break
    third = len(true) // 3
    return {'true': true, 'reversed': rev[:third], 'sibling': sib[:third], 'random': ran[:third]}


def _distances(table, pairs, device):
    with torch.no_grad():
        f = torch.tensor([p for p, _ in pairs], device=device)
        g = torch.tensor([w for _, w in pairs], device=device)
        return table.divergence(f, g).cpu()


def _tune(table, val, device):
    """Threshold on d with the best balanced accuracy on ``val`` (true against the three false kinds together)."""
    pos = _distances(table, val['true'], device)
    neg = torch.cat([_distances(table, val[k], device) for k in ('reversed', 'sibling', 'random')])
    candidates = torch.cat((pos, neg, torch.tensor([0., float('inf')]))).unique()
    best, at = -1., 0.
    for t in candidates[:: max(1, len(candidates) // 2000)].tolist():
        acc = .5 * float((pos <= t).float().mean()) + .5 * float((neg > t).float().mean())
        if acc > best:
            best, at = acc, t
    return at


def _accuracy(table, pairs, t, device):
    out = {k: float(((_distances(table, v, device) <= t) == (k == 'true')).float().mean()) for k, v in pairs.items()}
    out['balanced'] = .5 * out['true'] + .5 * sum(out[k] for k in ('reversed', 'sibling', 'random')) / 3
    return out


def _shape_report(table, spelled, row_of, words, rng, t, device):
    """Closeness by shared substrings, words told apart, spread of coordinates, log-det by length, distinctness."""
    mean, log_var = table.mean.detach().cpu().double(), table.log_var.detach().cpu().double()
    items = rng.sample(words, 400)
    rows = torch.tensor([row_of[w] for w in items])
    subs = [{a[p:q] for p in range(len(a)) for q in range(p + 2, len(a) + 1)} for a in items]
    m, lv = mean[rows], log_var[rows]
    kl = FormLinks.kl(m[:, None], lv[:, None], m[None], lv[None])
    sym = (kl + kl.T) / 2
    unit = m / m.norm(dim=1, keepdim=True)
    cosine = unit @ unit.T
    buckets = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            shared = len(subs[i] & subs[j])
            key = '0' if shared == 0 else '1-2' if shared <= 2 else '3-5' if shared <= 5 else '6+'
            buckets.setdefault(key, []).append((float(sym[i, j]), float(cosine[i, j])))
    close = {k: (len(v), sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
             for k, v in sorted(buckets.items(), key=lambda kv: ['0', '1-2', '3-5', '6+'].index(kv[0]))}
    others = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(3))]
    d_other = _distances(table, list(zip(rows.tolist(), others.tolist())), device)
    d_other = d_other[rows != others]
    three = [r for r, s in enumerate(spelled) if len(s) == 3]
    own, lack = [], []
    for w in items:
        mine = [w[i:i + 3] for i in range(len(w) - 2)]
        if not mine:
            continue
        own.append((row_of[rng.choice(mine)], row_of[w]))
        while True:
            r = rng.choice(three)
            if spelled[r] not in w:
                lack.append((r, row_of[w]))
                break
    d_own, d_lack = _distances(table, own, device), _distances(table, lack, device)
    all_words = torch.tensor([row_of[w] for w in words])
    std = mean[all_words].std(0)
    eig = torch.linalg.eigvalsh(torch.cov(mean[all_words].T)).clamp_min(0)
    used = int(table.used)
    length = table.length.cpu()[:used]
    logdet = log_var[:used].sum(1)
    by_length = {n: float(logdet[length == n].mean()) for n in (1, 2, 3, 4, 6, 8, 12, 16) if bool((length == n).any())}
    distinct = torch.unique(torch.cat((mean[:used], log_var[:used]), 1), dim=0).shape[0]
    return {'close': close,
            'word_vs_other': (float(d_other.median()), float((d_other <= t).float().mean())),
            'own3_vs_lack3': (float(d_own.median()), float(d_lack.median()),
                              float((d_own <= t).float().mean()), float((d_lack > t).float().mean())),
            'coord_std': (float(std.min()), float(std.median()), float(std.max())),
            'participation': float(eig.sum() ** 2 / eig.square().sum()),
            'logdet_by_length': by_length, 'distinct': (distinct, used)}


def density_experiment(count=3000, margin=2000., gamma=500., start_var=5e-5, mix=(1, 1, 1), lr=1e-2, steps=3000,
                       sample=2048, report_at=(0, 300, 1000, 3000), new_steps=200, device='cpu', seed=0, quiet=False):
    """Fit ``order_loss`` alone on dictionary forms, then insert 10% more words untrained; print what is measured."""
    say = (lambda *a: None) if quiet else print
    dev = torch.device(device)
    rng = random.Random(seed)
    words = _dictionary(count)
    rng.shuffle(words)
    base, late = words[:count * 9 // 10], words[count * 9 // 10:]
    table = FormLinks(1 << 17, 24, 0, margin=margin, gamma=gamma, start_var=start_var, order_mix=mix).to(dev)
    for i in range(0, len(base), 500):
        table.insert(*_pack(base[i:i + 500], dev))
    spelled = _host_spellings(table)
    row_of = {s: r for r, s in enumerate(spelled)}
    forms = list(range(LEAVES, len(spelled)))
    rng.shuffle(forms)
    held = forms[:len(forms) // 20]
    train = forms[len(forms) // 20:]
    train_wholes = torch.tensor(sorted(train), device=dev)
    val = _eval_pairs(spelled, row_of, train[:1500], rng, len(spelled))
    seen = _eval_pairs(spelled, row_of, train[1500:3000], rng, len(spelled))
    unseen = _eval_pairs(spelled, row_of, held, rng, len(spelled))
    optimizer = torch.optim.Adam([table.mean, table.log_var], lr=lr)
    gen = torch.Generator(device=dev).manual_seed(seed)
    report = {'setting': dict(margin=margin, gamma=gamma, start_var=start_var, mix=mix, lr=lr), 'curve': []}
    say(f"\n== m {margin:g}, gamma {gamma:g}, var {start_var:g}, mix S1/S2/S4 {mix}, lr {lr:g}; "
        f"{len(base)} words, {len(spelled) - LEAVES} forms, {len(held)} wholes never trained as wholes, device {device}")
    start, losses = time.perf_counter(), []
    for step in range(steps + 1):
        if step in report_at:
            t = _tune(table, val, dev)
            acc = _accuracy(table, seen, t, dev), _accuracy(table, unseen, t, dev)
            report['curve'].append((step, t) + acc)
            say(f"step {step:5d} ({time.perf_counter() - start:5.1f} s, loss {sum(losses[-50:]) / max(1, len(losses[-50:])):8.1f}): "
                f"threshold {t:9.2f}")
            for label, a in zip(('trained wholes', 'held-out wholes'), acc):
                say(f"    {label}: " + ', '.join(f'{k} {v:.3f}' for k, v in a.items()))
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = table.order_loss(sample, noise=table.order_noise(sample, gen), wholes=train_wholes)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            losses.append(float(loss.detach()))
    t = report['curve'][-1][1]
    shape = _shape_report(table, spelled, row_of, base, rng, t, dev)
    report['shape'] = shape
    say('closeness of whole words by shared substrings of length >= 2 (pairs, mean symmetric KL, cosine of means): '
        + '; '.join(f'{k}: {n}, {a:.1f}, {c:.3f}' for k, (n, a, c) in shape['close'].items()))
    say(f"word against another word: median d {shape['word_vs_other'][0]:.1f}, inside at the threshold "
        f"{shape['word_vs_other'][1]:.3f} (a word against itself: d 0)")
    o = shape['own3_vs_lack3']
    say(f"own three-letter part / one it lacks: median d {o[0]:.1f} / {o[1]:.1f}; right at the threshold {o[2]:.3f} / {o[3]:.3f}")
    say(f"coordinate std of word means min/median/max {shape['coord_std'][0]:.4f} / {shape['coord_std'][1]:.4f} / "
        f"{shape['coord_std'][2]:.4f}; participation ratio {shape['participation']:.1f} of 64")
    say('log-det of variance by length: ' + ', '.join(f'{n}: {v:.0f}' for n, v in shape['logdet_by_length'].items()))
    say(f"distinct (mean, log_var): {shape['distinct'][0]} of {shape['distinct'][1]}")
    # Incremental: 10% more words, no training, then a few steps around the new forms only.
    before = int(table.used)
    old_mean, old_var = table.mean.detach()[:before].clone(), table.log_var.detach()[:before].clone()
    for i in range(0, len(late), 500):
        table.insert(*_pack(late[i:i + 500], dev))
    spelled = _host_spellings(table)
    row_of = {s: r for r, s in enumerate(spelled)}
    new_rows = list(range(before, len(spelled)))
    pairs = _eval_pairs(spelled, row_of, new_rows, rng, len(spelled))
    inc = [('inserted, untrained', _accuracy(table, pairs, t, dev))]
    new_wholes = torch.tensor(new_rows, device=dev)
    optimizer = torch.optim.Adam([table.mean, table.log_var], lr=lr)
    for _ in range(new_steps):
        optimizer.zero_grad(set_to_none=True)
        table.order_loss(sample, noise=table.order_noise(sample, gen), wholes=new_wholes).backward()
        optimizer.step()
    inc.append((f'after {new_steps} steps around them', _accuracy(table, pairs, t, dev)))
    moved_mean = (table.mean.detach()[:before] - old_mean).abs().amax(1)
    moved_var = (table.log_var.detach()[:before] - old_var).abs().amax(1)
    report['incremental'] = inc
    report['old_moved'] = (float((moved_mean > 0).float().mean()), float(moved_mean.median()), float(moved_mean.max()),
                           float(moved_var.max()))
    say(f"incremental: {len(late)} words, {len(new_rows)} new forms; threshold kept at {t:.2f}")
    for label, acc in inc:
        say(f'  {label}: ' + ', '.join(f'{k} {v:.3f}' for k, v in acc.items()))
    say(f"  old forms moved by the steps around the new: {report['old_moved'][0]:.4f} of rows; |d mean| median "
        f"{report['old_moved'][1]:.2e}, max {report['old_moved'][2]:.3f}; |d log_var| max {report['old_moved'][3]:.3f}")
    return report


SWEEP = [dict(margin=2000., gamma=500., start_var=5e-5), dict(margin=100., gamma=3., start_var=2e-4)]
MIXES = [(1, 1, 1), (3, 0, 0), (1, 1, 2)]


@pytest.mark.slow
def test_density_order_embedding_fit_on_the_dictionary():
    """Report only: the paper's settings and negative mixes, fitted by ``order_loss`` alone."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    rows = []
    for setting in SWEEP:
        for mix in MIXES:
            r = density_experiment(mix=mix, device=device, **setting)
            rows.append(r)
    print('\nsetting | trained wholes: balanced / true / reversed / sibling / random | held-out wholes balanced | incremental untrained / trained')
    for r in rows:
        seen, unseen = r['curve'][-1][2:]
        print(r['setting'], ' / '.join(f"{seen[k]:.3f}" for k in ('balanced', 'true', 'reversed', 'sibling', 'random')),
              '| held-out', f"{unseen['balanced']:.3f}",
              '|', ' / '.join(f"{a['balanced']:.3f}" for _, a in r['incremental']))
