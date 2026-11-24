"""Performance rewrites must retain the forward, gradient and capture contracts."""
import pytest
import torch
from torch.nn import functional as F

from PerceptProperties import counts_in_spans


def old_counts(byte_ids, spans, observed=None):
    batch, length = byte_ids.shape
    spans = spans.to(byte_ids.device).clamp(0, length)
    counts = F.one_hot(byte_ids.long().clamp(0, 255), 256).to(torch.get_default_dtype())
    if observed is not None:
        counts = counts * observed[..., None].to(counts)
    prefix = torch.cat((counts.new_zeros(batch, 1, 256), counts.cumsum(1)), 1)
    return (prefix.gather(1, spans[..., 1:].expand(-1, -1, 256))
            - prefix.gather(1, spans[..., :1].expand(-1, -1, 256)))


@pytest.mark.parametrize('length', [0, 17, 2048])
def test_byte_counts_keep_values_and_observation_gradients(length):
    ids = torch.arange(2 * length).reshape(2, length) % 258 - 1
    spans = torch.tensor([[[0, length], [1, 4], [4, 4], [-2, length + 7]]] * 2)
    observed = torch.rand(2, length, requires_grad=True)
    expected = old_counts(ids, spans, observed)
    actual = counts_in_spans(ids, spans, observed)
    assert torch.equal(expected, actual)
    weight = torch.randn_like(expected)
    assert torch.equal(torch.autograd.grad((expected * weight).sum(), observed)[0],
                       torch.autograd.grad((actual * weight).sum(), observed)[0])


def test_byte_counts_fullgraph():
    ids = torch.tensor([[1, 255, 0, 1, 2], [2, 1, 1, 0, 255]])
    spans = torch.tensor([[[0, 4], [2, 5]]] * 2)
    compiled = torch.compile(counts_in_spans, backend='eager', fullgraph=True)
    assert torch.equal(compiled(ids, spans, ids != 0), old_counts(ids, spans, ids != 0))


def test_whole_part_reads_fullgraph_keep_ties_and_gradients():
    from BasicPercepts import held_at
    from test_item17_looks import hand_bank
    bank = hand_bank([[1., .5, 0., 1.], [.5, 1., 1., 0.]],
                     [[1., .5, 0., 1.], [.5, .5, .5, .5], [0., 0., 1., 1.]],
                     [0, 0, 1], 2)
    compiled = torch.compile(held_at, backend='aot_eager', fullgraph=True)
    table = bank.table.clone().requires_grad_()
    expected, actual = held_at(bank, table), compiled(bank, table)
    for left, right in zip(expected, actual, strict=True):
        assert torch.equal(left, right)
    left = torch.autograd.grad(sum(v.square().sum() for v in expected), table)[0]
    right = torch.autograd.grad(sum(v.square().sum() for v in actual), table)[0]
    assert torch.equal(left, right)


def test_prepared_lookup_fullgraph_keeps_live_activity_gradients():
    from test_item17_looks import hand_bank, bare_percepts
    bank = hand_bank([[1., .5, 0., 1.], [.5, 1., 1., 0.]],
                     [[1., .5, 0., 1.], [.5, .5, .5, .5], [0., 0., 1., 1.]],
                     [0, 0, 1], 2)
    basic = bare_percepts(bank)
    bank = basic.prepare(bank)
    compiled = torch.compile(basic.read_bank, backend='aot_eager', fullgraph=True)
    activity = bank.activity.clone().requires_grad_()
    expected, actual = basic.read_bank(bank, activity), compiled(bank, activity)
    assert torch.equal(expected.values, actual.values)
    assert torch.equal(expected.boundaries, actual.boundaries)
    assert torch.equal(torch.autograd.grad(expected.values.square().sum(), activity)[0],
                       torch.autograd.grad(actual.values.square().sum(), activity)[0])


@pytest.mark.parametrize('capacity', [2, 32])
@pytest.mark.parametrize('located', [False, True])
def test_shared_opening_keeps_selected_values_and_gradients(monkeypatch, capacity, located):
    from dataclasses import replace
    from test_item17_looks import hand_bank, bare_percepts
    import BasicPercepts
    import Salience
    bank = hand_bank([[1., .2, .5], [.4, 1., .8]],
                     [[1., .3, .2], [.4, .8, .6], [.3, .5, .9], [.9, .2, .6]],
                     [0, 0, 1, 1], 2)
    if located:
        from WhereRegistry import WhereRegistry
        registry = WhereRegistry([('text', 16)])
        wires = registry.encode('text', bank.span[..., 0]).detach()
        table = torch.cat((bank.table[..., :bank.width, :], torch.stack((wires, 1-wires), -1),
                           bank.table[..., bank.width+4:, :]), -2)
        bank = replace(bank, registry=registry, table=table)
    bank = replace(bank, table=bank.table.clone().requires_grad_())
    basic = bare_percepts(bank)
    monkeypatch.setattr(BasicPercepts, 'attended_capacity', lambda: capacity)
    with torch.no_grad():
        basic.where_entries[:2].uniform_(.1, .7)
        basic.when_entries[:2].uniform_(.1, .7)
    activity = torch.rand_like(bank.activity, requires_grad=True)
    rng = torch.get_rng_state()
    expected = basic.read_bank(bank, activity)
    expected_salience = Salience.rows_read(basic, bank)
    prepared = basic.prepare(replace(bank))
    torch.set_rng_state(rng)
    actual = basic.read_bank(prepared, activity)
    actual_salience = Salience.rows_read(basic, prepared)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.boundaries, expected.boundaries)
    assert torch.equal(actual_salience, expected_salience)
    args = (bank.table, activity, basic.where_entries, basic.when_entries)
    if located:
        args += (registry.code,)
    left = torch.autograd.grad(expected.values.square().sum() + expected_salience.square().sum(),
                               args, retain_graph=True)
    right = torch.autograd.grad(actual.values.square().sum() + actual_salience.square().sum(), args)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b)
    if located:
        assert left[-1].abs().sum() > 0, 'the attended read must retain the live location wires'
    assert replace(prepared, table=bank.table * .5).read_basis is None


def test_sparse_host_index_reads_once_and_tracks_pending_additions(monkeypatch):
    from Layers import _EntryIndex
    keys = torch.arange(193) % 7
    index = _EntryIndex(7, torch.device('cpu'))
    for entry in range(129):
        index.add(int(keys[entry]), entry, keys)
    reads = []
    original = torch.Tensor.tolist
    def read(tensor):
        reads.append(tensor.shape)
        return original(tensor)
    monkeypatch.setattr(torch.Tensor, 'tolist', read)
    for _ in range(3):
        for key in range(7):
            assert index.of(key) == list(range(key, 129, 7))
    assert len(reads) == 2
    result = index.of(0)
    result.clear()
    assert index.of(0)
    for entry in range(129, len(keys)):
        index.add(int(keys[entry]), entry, keys)
    for key in range(7):
        assert index.of(key) == list(range(key, len(keys), 7))


def test_stm_builds_only_its_own_relation_table(monkeypatch):
    import Memory
    constructed = []
    original = Memory.LTM1.__init__
    def observed(self, capacity):
        constructed.append(type(self))
        original(self, capacity)
    monkeypatch.setattr(Memory.LTM1, '__init__', observed)
    store = Memory.STM0(4, 8, relation_capacity=16)
    assert constructed == [Memory.STM1]
    assert store.parts._occurrence_store() is store


def test_percept_lookup_gathers_repeated_rows_with_the_same_gradient(monkeypatch):
    from test_item17_looks import hand_bank, bare_percepts
    bank = hand_bank([[1., .5], [.5, 1.]], [[1., .5], [.5, 1.]], [0, 1], 1)
    basic = bare_percepts(bank)
    basic.index = {('fixture', row): row for row in range(3)}
    codes = torch.tensor([[.2, .8], [.7, .3], [.1, .9]], requires_grad=True)
    monkeypatch.setattr(basic, 'code', lambda row: codes[row])
    rows = torch.tensor([[2, 1, -1], [0, 2, 2]])
    valid = rows >= 0
    expected = torch.zeros(2, 3, 2)
    for row in torch.unique(rows[valid]).tolist():
        expected = expected + (valid & (rows == row))[..., None] * codes[row]
    actual = basic.lookup_rows(rows, valid=valid)
    assert torch.equal(actual, expected)
    weight = torch.arange(12.).reshape_as(actual)
    assert torch.equal(torch.autograd.grad((actual * weight).sum(), codes, retain_graph=True)[0],
                       torch.autograd.grad((expected * weight).sum(), codes)[0])
    assert torch.equal(basic.lookup_rows(rows, valid=torch.zeros_like(valid)), torch.zeros_like(actual))
    with pytest.raises(ValueError, match='unavailable basic percept row'):
        basic.lookup_rows(rows)


def test_menu_percepts_are_live_shared_and_expire_even_on_error():
    from types import SimpleNamespace
    from MemberRead import read, share_percept_reads
    weight = torch.tensor([.25, .75], requires_grad=True)
    calls = []
    def code(row):
        calls.append(row)
        return weight.square()
    owner = SimpleNamespace(basic_percepts=SimpleNamespace(code=code),
                            similarity_codebook=SimpleNamespace(W=torch.zeros(4, 3)))
    @share_percept_reads
    def menu(owner, fail=False):
        values = [read(owner, ('bp', 1)) for _ in range(4)]
        if fail:
            raise ValueError('discarded menu')
        # Readers own the padded result; modifying one cannot change a peer.
        values[0] = values[0] * 2
        return torch.stack(values).sum()
    result = menu(owner)
    assert calls == [1]
    assert torch.equal(torch.autograd.grad(result, weight)[0], 10 * weight)
    with pytest.raises(ValueError, match='discarded menu'):
        menu(owner, fail=True)
    with torch.no_grad():
        weight.add_(1)
    assert torch.equal(read(owner, ('bp', 1))[:2], weight.square())
    assert calls == [1, 1, 1]


def test_sparse_topology_tracks_swap_removal_and_checkpoint():
    from Layers import SparseSigmaLDU
    sigma = SparseSigmaLDU(5, 4, 8)
    for row, source, weight in [(1, 2, .3), (2, 3, .5), (1, 1, .6), (0, 0, .2)]:
        sigma.add(row, source, weight, negated=source == 1)
    sigma.remove(2, 3)
    saved = SparseSigmaLDU(5, 4, 8)
    saved.load_state_dict(sigma.state_dict())
    for layer in (saved, sigma):
        weights = layer.w(occupied_only=True).tolist()
        edges = list(zip(layer.entry_row[:layer.nnz].tolist(), layer.entry_input[:layer.nnz].tolist(),
                         weights, strict=True))
        expected = {r: [(c, abs(w), w < 0) for row, c, w in edges if row == r] for r in range(4)}
        assert layer.entries_by_row([1, 1, 2, 0, 3]) == expected
        assert [layer.inputs_read(r) for r in range(4)] == [[e[0] for e in expected[r]] for r in range(4)]
        assert layer.rows_reading(3) == []
        layer.clear()
        assert layer.entries_by_row(range(4)) == {r: [] for r in range(4)}


def test_inverse_admission_respects_eviction_and_reappearance_in_one_stream():
    from Generative import LiveConstituents
    from Memory import STMChunks
    words = torch.tensor([[[3., 0.]], [[3., 0.]]])
    bank = LiveConstituents(words, torch.ones(2, 1, dtype=torch.bool), 2)
    bank.codes[:, 0] = torch.tensor([5., 0.])
    bank.valid[:, 0] = True
    state = STMChunks.empty(2, 8, 2)
    values = words.new_tensor([3., 1., 2., 1., 5., float('nan')])
    state[STMChunks.BUFFER][:, :6, 0] = values
    state[STMChunks.DEPTH].fill_(6)
    state[STMChunks.STACK][:, :6] = torch.arange(6)
    bank.observe(state, torch.tensor([True, False]), None)
    assert bank.codes.tolist() == [[[2., 0.], [5., 0.]], [[5., 0.], [0., 0.]]]
    assert bank.valid.tolist() == [[True, True], [True, False]]


def test_forecast_keeps_the_same_outside_evidence_with_bounded_history():
    from LookForecast import LookForecast
    forecast = LookForecast.__new__(LookForecast)
    forecast.basic_count = forecast.order_zero_width = forecast.width = 2
    forecast.address = torch.arange(2)
    forecast._located = lambda evidence: evidence
    initial = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 24
    forecast.observations = initial
    forecast.latest = torch.zeros(2, 2)
    observed, positions = [initial], [torch.arange(3)]
    for width in (6, 2, 4, 7):
        extent = (torch.arange(width)[None] + torch.arange(2)[:, None]) % 3 == 0
        evidence = torch.rand(2, width, 2, 2)
        forecast.observe(evidence, extent, torch.zeros(2, 2, 2))
        observed.append(torch.where(extent[..., None, None], evidence, 0.))
        positions.append(torch.arange(width))
        longest = max(len(slots) for slots in positions)
        assert forecast.observations.shape[1] == longest
        for held in (0, 1, 2):
            extent = torch.arange(longest)[None].expand(2, -1) % 3 == held
            live, actual = forecast._outside(extent)
            outside = ~extent[:, torch.cat(positions)]
            expected = torch.where(outside[..., None, None], torch.cat(observed, 1), 0.).amax(1)
            assert torch.equal(actual, expected[live])


def test_sparse_entry_backward_retains_values_from_before_online_admission():
    from Layers import SparseSigmaLDU
    sigma = SparseSigmaLDU(4, 4, 16)
    sigma.add(0, 0, .8)
    sigma.add(1, 0, .3)
    rows = torch.tensor([0, 1, 3])
    inputs = torch.tensor([0, 0, 1])
    cost = sigma.entry(rows, inputs).square().sum()
    expected, = torch.autograd.grad(cost, sigma.value, retain_graph=True)
    sigma.add(2, 1, .7)
    actual, = torch.autograd.grad(cost, sigma.value)
    assert expected.count_nonzero() == 2
    assert torch.equal(actual, expected)
