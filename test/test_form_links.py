"""The table of forms agrees with a host dictionary of spellings and runs on the device without host reads."""
import copy
import os
import random
import time

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from FormLinks import FormLinks, HALF, LEAVES, PASSES, SCALE, UNUSED_KEY

DEVICES = ['cpu', 'mps']
LONGEST = 8
HARD = [b'abab', b'aaaa', b'banana', b'mississippi', b'abcabcabc', b'aabaa', b'aaa', b'abaaba', b'aabaab']


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


def _table(device, rng, capacity=8192, threshold=0, passes=PASSES):
    table = FormLinks(capacity, LONGEST, threshold, passes=passes).to(device)
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
    table.recentre_rows(table.read(*_pack(words[:5], 'cpu')))
    state = table.state_dict()
    assert set(state) == {'left', 'right', 'length', 'count', 'used', 'pair_keys', 'pair_rows', 'code',
                          'holder_sum', 'holder_count'}
    restored = FormLinks(table.capacity, LONGEST, 0)
    restored.load_state_dict(state)
    data, lengths = _pack(_sentences(rng, words), 'cpu')
    for got, want in zip(restored.recognise(data, lengths), table.recognise(data, lengths)):
        assert torch.equal(got, want)
    for t in (table, restored):
        t.recentre_rows(t.read(*_pack(words[5:9] + [b'qqqzz'], 'cpu')))
        t.sweep(3)
    for name in state:
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
        full.recentre_rows(rows[:40])
        bounded.recentre_rows(rows[:40])
    for name in ('code', 'holder_sum', 'holder_count', 'count', 'pair_keys', 'left', 'right', 'used'):
        assert torch.equal(getattr(full, name), getattr(bounded, name)), name
    with pytest.raises(RuntimeError, match='longer than its bound'):
        bounded.read(*_pack([b'abcdefgh'], 'cpu'), longest=4)

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


# -- counting -------------------------------------------------------------------

def test_a_row_is_a_form_only_once_counted_to_the_threshold():
    table = FormLinks(1024, LONGEST, 2)
    data, lengths = _pack([b'the cat'], 'cpu')
    unit = _pack([b'cat'], 'cpu')
    table.read(*unit)
    row = int(table._unit_rows(*unit)[0])
    assert row >= LEAVES and int(table.count[row]) == 1
    assert table.rows_of(*unit).tolist() == [-1]
    rows, best = table.recognise(data, lengths)
    assert bool((best[0, :7] < LEAVES).all()), 'below the threshold the parts are used'
    assert bool((best[0, :7] == data[0, :7].long()).all()), 'leaves always'
    table.read(*unit)
    assert table.rows_of(*unit).tolist() == [row] and int(table.count[row]) == 2
    rows, best = table.recognise(data, lengths)
    assert int(best[0, 4]) == row and int(rows[0, 2, 4]) == row
    at = table._unit_rows(*_pack([b'at'], 'cpu'))
    assert int(table.count[at]) == 0 and int(rows[0, 1, 5]) == -1, 'a part read only inside a unit is not counted'


# -- the pull -------------------------------------------------------------------

def test_points_are_uniform_reproducible_and_equal_on_metal():
    rows = torch.arange(1 << 18)
    p = FormLinks._points(rows)
    assert p.shape == (1 << 18, HALF) and float(p.min()) >= 0 and float(p.max()) < 1
    assert abs(float(p.mean()) - .5) < 1e-3 and abs(float(p.std()) - 12 ** -.5) < 1e-3
    assert torch.unique(p, dim=0).shape[0] == rows.shape[0]
    assert torch.equal(p, FormLinks._points(rows))
    if torch.backends.mps.is_available():
        assert torch.equal(FormLinks._points(rows.to('mps')).cpu(), p)


def _brute_holders(table):
    """Holder sums and counts gathered from the links: each form added once to each distinct link."""
    used = int(table.used)
    f = torch.arange(LEAVES, used, device=table.code.device)
    L, R = table.left[f].long(), table.right[f].long()
    q = table._quantise(table.code[f, :HALF])
    total = torch.zeros_like(table.holder_sum).index_add_(0, L, q)
    count = torch.zeros_like(table.holder_count).index_add_(0, L, torch.ones_like(L, dtype=torch.int32))
    other = R != L
    total.index_add_(0, R[other], q[other])
    count.index_add_(0, R[other], torch.ones_like(R[other], dtype=torch.int32))
    return total, count


@pytest.mark.parametrize('device', DEVICES)
def test_running_holder_sums_equal_the_gathered_ones(device):
    dev = _device(device)
    rng = random.Random(10)
    table, words = _table(dev, rng)
    for _ in range(4):
        read = _pack(rng.sample(words, 20), dev)
        table.recentre_rows(table.read(*read))
        table.insert(*_pack([bytes(rng.choices(range(97, 123), k=rng.randint(2, 9))) for _ in range(6)] + HARD, dev))
    total, count = _brute_holders(table)
    assert torch.equal(table.holder_count, count) and torch.equal(table.holder_sum, total)
    table.sweep(5)
    total, count = _brute_holders(table)
    assert torch.equal(table.holder_count, count) and torch.equal(table.holder_sum, total)
    assert int(count.sum()) > 0


@pytest.mark.parametrize('device', DEVICES)
def test_recentre_rows_moves_the_rows_given_and_their_links_only(device):
    dev = _device(device)
    rng = random.Random(16)
    table, words = _table(dev, rng, passes=0)  # no insertion steps: every form is off its target
    spelled = _host_spellings(table)
    row_of = {s: r for r, s in enumerate(spelled)}
    given = [row_of[w] for w in words[:6] if len(w) > 1] + [row_of[b'an'], row_of[b'a'], -1]
    rows = torch.tensor(given + given[:3], device=dev)
    allowed = {r for g in given if g >= LEAVES for r in (g, int(table.left[g]), int(table.right[g])) if r >= LEAVES}
    before = table.code.clone()
    captured = copy.deepcopy(table)
    table.recentre_rows(rows)
    moved = set((table.code != before).any(1).nonzero().flatten().tolist())
    assert moved == allowed, 'each form given and its two links take a step; bytes never'
    torch._dynamo.reset()
    torch.compile(captured.recentre_rows, fullgraph=True, backend='eager')(rows)
    for name, buffer in table.named_buffers():
        assert torch.equal(getattr(captured, name), buffer), name


@pytest.mark.parametrize('device', DEVICES)
def test_read_and_recentre_capture_whole_and_read_nothing_on_the_host(device, monkeypatch):
    dev = _device(device)
    rng = random.Random(18)
    table, words = _table(dev, rng, threshold=2)
    batch = _pack(words[::3] + HARD + [b'zzyzx', b'qqq'], dev)
    eager, captured = copy.deepcopy(table), copy.deepcopy(table)

    def forbidden(*args, **kwargs):
        raise AssertionError('read a tensor on the host')

    def step(t):
        rows = t.read(*batch, longest=16)
        t.recentre_rows(rows)
        return rows

    with monkeypatch.context() as patch, _NoHostScalar():
        for method in ('item', 'tolist', '__bool__'):
            patch.setattr(torch.Tensor, method, forbidden)
        want = step(eager)
    torch._dynamo.reset()
    got = torch.compile(step, fullgraph=True, backend='eager')(captured)
    assert torch.equal(got, want)
    for name, buffer in eager.named_buffers():
        assert torch.equal(getattr(captured, name), buffer), name


def test_sweep_keeps_links_counts_and_bytes():
    rng = random.Random(17)
    table, words = _table('cpu', rng)
    kept = {name: getattr(table, name).clone() for name in ('left', 'right', 'length', 'count', 'used',
                                                           'pair_keys', 'pair_rows', 'holder_count')}
    table.sweep(20)
    for name, value in kept.items():
        assert torch.equal(getattr(table, name), value), name
    assert torch.equal(table.code[:LEAVES, :HALF], table.step * FormLinks._points(torch.arange(LEAVES)))
    assert bool((table.code[:, HALF:] == 0).all())


# -- the pull on the dictionary ---------------------------------------------------

def _dictionary(count, seed=0):
    path = '/usr/share/dict/words'
    if not os.path.exists(path):
        pytest.skip('requires /usr/share/dict/words')
    with open(path, 'rb') as handle:
        words = list(dict.fromkeys(w.strip().lower() for w in handle if w.strip()))
    return [w for w in random.Random(seed).sample(words, count) if 3 <= len(w) <= 14]


def _fill(table, words, device, batch=500):
    for i in range(0, len(words), batch):
        table.insert(*_pack(words[i:i + batch], device))


def _part_pairs(spelled, rows):
    """Every (form, distinct proper substring) pair of ``rows`` by the host."""
    row_of = {s: r for r, s in enumerate(spelled)}
    whole, part = [], []
    for r in rows:
        s = spelled[r]
        for p in {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)} - {s}:
            whole.append(r)
            part.append(row_of[p])
    return torch.tensor(whole, dtype=torch.long), torch.tensor(part, dtype=torch.long)


def _order(table, whole, part):
    code = table.code[:, :HALF].cpu()
    return float((code[whole] >= code[part]).all(1).float().mean())


def _closeness(table, spelled, items, against=None):
    """Mean cosine of codes less the mean word, by shared distinct substrings 0-2, 3-5, 6-8, 9+."""
    row_of = {s: r for r, s in enumerate(spelled)}
    code = table.code[:, :HALF].cpu()
    pair = against is not None
    against = against if pair else items
    a, b = code[[row_of[w] for w in items]], code[[row_of[w] for w in against]]
    centre = torch.cat((a, b) if pair else (a,)).mean(0, keepdim=True)
    cosine = F.cosine_similarity((a - centre)[:, None], (b - centre)[None], dim=-1).tolist()
    subs = lambda w: {w[p:q] for p in range(len(w)) for q in range(p + 1, len(w) + 1)}
    sa, sb = [subs(w) for w in items], [subs(w) for w in against]
    buckets = {}
    for i in range(len(items)):
        for j in range(len(against)):
            if (not pair and j <= i) or items[i] == against[j]:
                continue
            buckets.setdefault(min(len(sa[i] & sb[j]) // 3, 3), []).append(cosine[i][j])
    return [sum(buckets[k]) / len(buckets[k]) for k in range(4)]


@pytest.fixture(scope='module')
def dictionary():
    """About 2,900 words, 64,000 rows, the default sweep from the random start."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    words = _dictionary(3000)
    table = FormLinks(1 << 17, 16, 0).to(device)
    _fill(table, words, device)
    start = time.perf_counter()
    table.sweep()
    table.code.sum().item()
    seconds = time.perf_counter() - start
    spelled = _host_spellings(table)
    print(f'\n{len(words)} words, {int(table.used)} rows; default sweep ({table.rounds} rounds) on {device}: {seconds:.1f} s')
    return words, table, spelled


def test_after_the_sweep_wholes_lie_above_parts_and_codes_are_distinct(dictionary):
    words, table, spelled = dictionary
    used = int(table.used)
    whole, part = _part_pairs(spelled, range(LEAVES, used))
    share = _order(table, whole, part)
    code = table.code[:, :HALF].cpu()
    lengths = torch.tensor([len(s) for s in spelled])
    by_length = {n: float(code[:used][lengths == n].mean()) for n in range(1, 15)}
    print(f'\nwhole >= part in every coordinate: {share:.5f} of {len(whole)} pairs; mean coordinate by length: '
          + ', '.join(f'{n}: {v:.2f}' for n, v in by_length.items()))
    assert share >= .99
    assert torch.unique(code[:used], dim=0).shape[0] == used
    values = list(by_length.values())
    assert values == sorted(values)
    assert torch.equal(code[:LEAVES], table.step * FormLinks._points(torch.arange(LEAVES))), 'bytes never move'
    assert bool((table.code[:, HALF:] == 0).all())


def test_whole_words_grow_closer_with_shared_substrings(dictionary):
    words, table, spelled = dictionary
    means = _closeness(table, spelled, random.Random(1).sample(words, 300))
    print('\nwhole words, closeness by shared 0-2, 3-5, 6-8, 9+: ' + ', '.join(f'{v:.3f}' for v in means)
          + ' (toy at 3,000 rounds, step 1/8: 0.45, 0.65, 0.74, 0.79)')
    assert means == sorted(means) and means[0] < means[-1]


@pytest.mark.parametrize('count', [30, 300])
def test_new_words_take_their_order_at_insertion(dictionary, count):
    """Insertion alone places a new unit's forms above their parts; closeness is reported."""
    words, settled, _ = dictionary
    device = settled.code.device
    table = copy.deepcopy(settled)
    before = int(table.used)
    late = [w for w in _dictionary(3600, seed=5) if w not in set(words)][:count]
    table.insert(*_pack(late, device))
    spelled = _host_spellings(table)
    whole, part = _part_pairs(spelled, range(before, int(table.used)))
    old = random.Random(2).sample(words, 300)
    rng = random.Random(3)
    lines, done, shares = [], 0, []
    for k in (0, 100):
        while done < k:
            batch = [rng.choice(late) if rng.random() < .5 else rng.choice(words) for _ in range(12)]
            table.recentre_rows(table.read(*_pack(batch, device), longest=16))
            done += 1
        shares.append(_order(table, whole, part))
        lines.append(f'{k} readings: order {shares[-1]:.3f}, closeness new x old '
                     + ', '.join(f'{v:.2f}' for v in _closeness(table, spelled, late[:100], old)))
    used = int(table.used)
    every = _order(table, *_part_pairs(spelled, range(LEAVES, used)))
    print(f'\n{len(late)} new words, {used - before} new forms, {len(whole)} pairs with a new whole, '
          f'{table.passes} passes at insertion:\n  ' + '\n  '.join(lines) + f'\n  order over the whole table: {every:.4f}')
    assert shares[0] >= .9 and every >= .99
    assert torch.unique(table.code[:used, :HALF].cpu(), dim=0).shape[0] == used


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_cpu_and_metal_make_equal_codes():
    words = _dictionary(1000)
    rng = random.Random(6)
    tables = {}
    for device in ('cpu', 'mps'):
        table = FormLinks(1 << 16, 16, 0).to(device)
        _fill(table, words[:600], device)
        table.sweep(100)
        _fill(table, words[600:], device)
        stream = random.Random(7)
        for _ in range(50):
            table.recentre_rows(table.read(*_pack(stream.sample(words, 12), device), longest=16))
        table.sweep(100)
        tables[device] = table
    cpu, metal = tables['cpu'], tables['mps']
    for name in ('left', 'right', 'used', 'pair_keys', 'pair_rows', 'holder_count', 'count'):
        assert torch.equal(getattr(metal, name).cpu(), getattr(cpu, name)), name
    gap = float((metal.code.cpu() - cpu.code).abs().max())
    held = (metal.holder_sum.cpu() - cpu.holder_sum).abs().max() / SCALE
    print(f'\nlargest code gap CPU against Metal: {gap:.1e}; holder sum gap {float(held):.1e}')
    assert gap <= 1e-5 and float(held) <= 1e-5 * max(1, int(cpu.holder_count.max()))
