"""The table of forms agrees with a host dictionary of spellings and runs on the device without host reads."""
import os
import random

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
    table.tally(*_pack(words[:5], 'cpu'))
    state = table.state_dict()
    assert set(state) == {'left', 'right', 'length', 'count', 'used', 'pair_keys', 'pair_rows', 'code'}
    restored = FormLinks(table.capacity, LONGEST, 0)
    restored.load_state_dict(state)
    data, lengths = _pack(_sentences(rng, words), 'cpu')
    for got, want in zip(restored.recognise(data, lengths), table.recognise(data, lengths)):
        assert torch.equal(got, want)
    assert torch.equal(restored.count, table.count) and int(restored.used) == int(table.used)
    assert torch.equal(restored.code, table.code)


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


def _dictionary(count):
    path = '/usr/share/dict/words'
    if not os.path.exists(path):
        pytest.skip('requires /usr/share/dict/words')
    with open(path, 'rb') as handle:
        words = sorted({w.strip() for w in handle if 0 < len(w.strip()) <= 24})
    return random.Random(7).sample(words, count)


def _dictionary_table(device, words, capacity=131072):
    table = FormLinks(capacity, LONGEST, 2).to(device)
    for i in range(0, len(words), 500):
        table.close(*_pack(words[i:i + 500], device))
    return table


@pytest.fixture(scope='module')
def dictionary():
    words = _dictionary(3000)
    return words, _dictionary_table('cpu', words)


def test_codes_order_parts_below_wholes_in_both_towers_and_are_distinct(dictionary):
    _, table = dictionary
    used = int(table.used)
    rows = torch.arange(LEAVES, used)
    parts, wholes = table.code[:, :HALF], table.code[:, HALF:]
    for link in (table.left, table.right):
        part = link[rows].long()
        assert bool((parts[rows] >= parts[part]).all() and (parts[rows] > parts[part]).any(1).all())
        assert bool((wholes[part] >= wholes[rows]).all() and (wholes[part] > wholes[rows]).any(1).all())
    assert torch.unique(table.code[:used], dim=0).shape[0] == used


def test_distance_falls_as_shared_substrings_rise_among_words_of_one_length(dictionary):
    words, table = dictionary
    for n in (7, 9):
        same = [w for w in words if len(w) == n]
        code = table.code[table._unit_rows(*_pack(same, 'cpu'))]
        dist = torch.cdist(code, code)
        subs = [{w[i:j] for i in range(n) for j in range(i + 1, n + 1)} for w in same]
        few, many = [], []
        for i in range(len(same)):
            for j in range(i + 1, len(same)):
                shared = len(subs[i] & subs[j])
                (few if shared <= 2 else many if shared >= 6 else []).append(float(dist[i, j]))
        assert len(many) > 50 and sum(many) / len(many) < sum(few) / len(few), n


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_cpu_and_metal_make_equal_codes(dictionary):
    words, table = dictionary
    metal = _dictionary_table('mps', words[:1000])
    cpu = _dictionary_table('cpu', words[:1000])
    assert int(metal.used) == int(cpu.used)
    assert torch.equal(metal.code.cpu(), cpu.code) and torch.equal(metal.left.cpu(), cpu.left)
