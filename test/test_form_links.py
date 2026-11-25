"""The table of forms agrees with a host dictionary of spellings and runs on the device without host reads."""
import os
import random
from collections import Counter

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from FormLinks import FormLinks, HALF, LEAVES

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
        table.admit_forms(*_pack(batch, device))
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
    again = table.admit_forms(*_pack(words, dev)).cpu().tolist()
    assert int(table.used) == used and again == [row_of[w] for w in words]


def test_a_full_table_raises():
    table = FormLinks(LEAVES + 3, LONGEST, 0)
    table.admit_forms(*_pack([b'ab', b'bc'], 'cpu'))
    with pytest.raises(RuntimeError, match='FormLinks full'):
        table.admit_forms(*_pack([b'abcd'], 'cpu'))


def test_state_dict_restores_the_table():
    rng = random.Random(4)
    table, words = _table('cpu', rng)
    table.close(*_pack(words[:5], 'cpu'))
    state = table.state_dict()
    assert set(state) == {'left', 'right', 'length', 'count', 'used', 'pair_keys', 'pair_rows', 'code',
                          'whole_sum', 'whole_count'}
    restored = FormLinks(table.capacity, LONGEST, 0)
    restored.load_state_dict(state)
    data, lengths = _pack(_sentences(rng, words), 'cpu')
    for got, want in zip(restored.recognise(data, lengths), table.recognise(data, lengths)):
        assert torch.equal(got, want)
    assert torch.equal(restored.count, table.count) and int(restored.used) == int(table.used)
    for name in ('code', 'whole_sum', 'whole_count'):
        assert torch.equal(getattr(restored, name), getattr(table, name)), name
    table.close(*_pack(words[5:9], 'cpu'))
    restored.close(*_pack(words[5:9], 'cpu'))
    assert torch.equal(restored.code, table.code) and torch.equal(restored.whole_sum, table.whole_sum)


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
        table.admit_forms(*_pack([bytes(rng.choices(range(97, 123), k=rng.randint(2, 9))) for _ in range(6)], 'cpu'))
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
    assert torch.equal(metal.whole_count.cpu(), cpu.whole_count)
    assert (metal.code.cpu() - cpu.code).abs().max() <= 1e-5
