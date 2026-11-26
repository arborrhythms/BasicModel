"""Performance rewrites must retain the forward, gradient and capture contracts."""
import pytest
import torch
from torch.nn import functional as F

from PerceptProperties import counts_in_spans


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_native_ancestry_expands_repeated_parts_without_host_reads(device, monkeypatch):
    from BasicPercepts import BasicPercepts
    from Layers import RadixLayer
    from Meaning import share_meaning_metadata
    from torch.utils._python_dispatch import TorchDispatchMode
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    native = RadixLayer(4, initial_cap=8).to(device)
    for raw in (b'a', b'bc', b'abc', b'aabc'):
        native.insert(raw)
    native.part_groups = {2: (0, 1), 3: (0, 2)}
    basic = BasicPercepts(native, 2, 2, 8).to(device)
    ids = torch.tensor([[3, 0, -1], [2, 1, 0], [-1, -1, -1]], device=device)
    spans = torch.tensor([[[4, 8], [9, 10], [0, 0]], [[0, 3], [3, 5], [6, 7]],
                          [[0, 0], [0, 0], [0, 0]]], device=device)
    expected_ids = ids.new_tensor([[0, 0, 1, 0], [0, 1, 1, 0], [-1] * 4])
    expected_spans = spans.new_tensor([[[4, 5], [5, 6], [6, 8], [9, 10]],
                                       [[0, 1], [1, 3], [3, 5], [6, 7]], [[0, 0]] * 4])

    class NoHostScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten._local_scalar_dense.default
            return func(*args, **(kwargs or {}))

    def forbidden(*args, **kwargs):
        raise AssertionError('native ancestry read a tensor on the host')

    with monkeypatch.context() as patch, share_meaning_metadata(), NoHostScalar():
        for method in ('item', 'tolist'):
            patch.setattr(torch.Tensor, method, forbidden)
        rows, extents = basic._expand_native_parts(ids, spans)
    torch.testing.assert_close(rows[:, :4], expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(extents[:, :4], expected_spans, rtol=0, atol=0)
    assert rows[:, 4:].eq(-1).all() and extents[:, 4:].eq(0).all()
    captured = torch.compile(basic._expand_native_parts, backend='eager', fullgraph=True)
    for actual, expected in zip(captured(ids, spans), (rows, extents), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual in basic._expand_native_parts(ids[:, :0], spans[:, :0]):
        assert actual.shape[1] == 0
    invalid = spans.clone()
    invalid[0, 0, 1] += 1
    with pytest.raises(ValueError, match='native part ancestry disagrees with its extent'):
        basic._expand_native_parts(ids, invalid)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('sparse', [False, True])
def test_inventory_boundary_reads_capture_without_predicates_and_keep_gradients(device, sparse):
    from types import SimpleNamespace
    from BasicPercepts import BasicPercepts
    from Spaces import Codebook
    from torch.utils._python_dispatch import TorchDispatchMode
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    banks = [Codebook(), Codebook()]
    for bank in banks:
        bank.W = torch.nn.Parameter(torch.arange(32, device=device).reshape(8, 4).float() / 32)
        bank.sparse_lookup_grad = sparse
    basic = BasicPercepts(SimpleNamespace(_basis=banks[0]), 2, 2, 8).to(device)
    object.__setattr__(basic, 'whole', banks[1])
    refs = [6 if sparse else -2, 0, 2, 0]
    tensor_refs = torch.tensor(refs, device=device)

    class NoHostPredicate(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func not in (torch.ops.aten._local_scalar_dense.default,
                                torch.ops.aten.nonzero.default)
            return func(*args, **(kwargs or {}))

    for whole in (False, True):
        read = basic.whole_codes if whole else basic.part_codes
        # The tensor-reference route remains the general mixed-value reference.
        expected = read(tensor_refs)
        with NoHostPredicate():
            eager = read(refs)
        # The existing sparse embedding adapter is deliberately outside capture.
        actual = read(refs) if sparse else torch.compile(read, backend='eager', fullgraph=True)(refs)
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weight = expected.new_tensor([[1., 2.], [3., -1.], [4., 0.], [-2., 5.]])
        for value in (eager, actual):
            wanted = torch.autograd.grad((expected * weight).sum(), tuple(b.W for b in banks),
                                         allow_unused=True, retain_graph=True)
            got = torch.autograd.grad((value * weight).sum(), tuple(b.W for b in banks), allow_unused=True)
            for left, right in zip(wanted, got, strict=True):
                assert (left is None) == (right is None)
                if left is not None:
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        with NoHostPredicate():
            empty = read([])
        assert empty.shape == (0, 2) and not empty.requires_grad


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_native_decode_checks_strength_on_stream_and_captures(device):
    from FieldReverse import NativeDecode
    from Meaning import share_meaning_metadata
    from torch.utils._python_dispatch import TorchDispatchMode
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    decoder = NativeDecode(4, 7, 3).to(device)
    decoder.sync([b'a', b'\x00b', b'xyz', b''])
    rows = torch.tensor([[2, 0, 0, 1, 0, 0, 3], [1, 0, 2, 0, 0, 0, 0]], device=device)
    strength = torch.tensor([[1., 1., 1., 1., 1., 0., 1.], [1., 1., 1., 1., 1., 1., 1.]], device=device)
    expected = torch.tensor([[120, 121, 122, 0, 98, 0, 0], [0, 98, 120, 121, 122, 97, 97]], device=device)

    class NoHostScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten._local_scalar_dense.default
            return func(*args, **(kwargs or {}))

    with share_meaning_metadata(), NoHostScalar():
        actual = decoder(rows, strength)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    captured = torch.compile(decoder, backend='eager', fullgraph=True)
    torch.testing.assert_close(captured(rows, strength), expected, rtol=0, atol=0)
    # Eager external input validation still raises before dispatching the walk.
    for invalid in (float('nan'), float('inf')):
        with pytest.raises(RuntimeError, match='radix activity contains NaN/Inf'):
            decoder(rows, torch.full_like(strength, invalid))
        if device == 'cpu':
            with pytest.raises(RuntimeError, match='radix activity contains NaN/Inf'):
                captured(rows, torch.full_like(strength, invalid))
        assert torch.is_grad_enabled()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize('reader', ['meaning', 'native_decode'])
def test_compiled_sentence_rejects_an_invalid_operand_on_metal(tmp_path, reader):
    """Test the project's compiler hooks in isolation from the suite's device state."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = """
import torch
import util
import sys
if sys.argv[1] == 'meaning':
    from Meaning import ConceptualMeaning
    def read(roles, mask):
        return ConceptualMeaning(roles, mask).roles.sum()
    values = torch.ones(3, 4, device='mps')
    mask = torch.tensor([True, False, True], device='mps')
    message = 'non-finite occupied meaning role'
else:
    from FieldReverse import NativeDecode
    decoder = NativeDecode(1, 4, 1).to('mps')
    decoder.sync([bytes([2])])
    def read(strength, rows):
        return decoder(rows, strength).sum()
    values = torch.ones(1, 4, device='mps')
    mask = torch.zeros(1, 4, dtype=torch.long, device='mps')
    message = 'radix activity contains NaN/Inf'

compiled = util.compile(read, fullgraph=True)
assert compiled is not read
assert compiled(values, mask).item() == 8
values[0, 0] = float('nan')
try:
    compiled(values, mask)
    torch.mps.synchronize()
except torch.AcceleratorError as exc:
    assert message in str(exc)
else:
    raise AssertionError('compiled reader accepted non-finite occupied content')
"""
    subprocess.run([sys.executable, '-c', code, reader], check=True, cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root / 'bin'), 'MODEL_COMPILE': 'inductor',
             'BASICMODEL_DEVICE': 'mps', 'TORCHINDUCTOR_CACHE_DIR': str(tmp_path)})


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


@pytest.mark.parametrize('whole', [True, False])
def test_stored_fold_matches_repeated_addresses_and_tied_gradients(whole):
    from BasicPercepts import fold_boundaries
    table = torch.tensor([[.2, .8], [.2, .4], [.9, .4]], requires_grad=True)
    addresses = torch.tensor([0, 2, 0, 0, 1])
    lengths = torch.tensor([2, 3, 0])
    rows = torch.tensor([0, 0, 1, 1, 1])
    reduce = torch.amin if whole else torch.amax
    expected = torch.stack((reduce(table[[0, 2]], 0), reduce(table[[0, 0, 1]], 0),
                            table.new_full((2,), float(whole))))
    captured = torch.compile(fold_boundaries, backend='aot_eager', fullgraph=True)
    actual = captured(table, addresses, rows, lengths, whole=whole)
    assert torch.equal(actual, expected)
    weights = torch.tensor([[1., 2.], [3., -1.], [1., 1.]])
    torch.testing.assert_close(torch.autograd.grad((actual * weights).sum(), table)[0],
                               torch.autograd.grad((expected * weights).sum(), table)[0])


def test_concept_lookup_tracks_incremental_admission_and_checkpoint_replacement():
    from Layers import ConceptualAttentionLayer
    store = ConceptualAttentionLayer.square(8)
    ids = torch.tensor([[-1, 42, 9, 8], [12, 9, 8, 42]])
    assert torch.equal(store.concept_rows(ids), torch.full_like(ids, -1))
    store.assign_row(('concept', 42), base=2)
    store.assign_row(('concept', 9), base=5)
    store.assign_row(('other', 8))
    assert store.concept_rows(ids).tolist() == [[-1, 2, 5, -1], [-1, 5, -1, 2]]
    store._tensor_rows = {('concept', 42): 1, ('concept', 8): 6, ('other', 9): 3}
    assert store.concept_rows(ids).tolist() == [[-1, 1, -1, 6], [-1, -1, 6, 1]]


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
        args += (registry.grid,)
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


@pytest.mark.parametrize("compiled", [False, True])
def test_inverse_admission_respects_eviction_and_reappearance_in_one_stream(compiled):
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
    observe = torch.compile(bank.observe, fullgraph=True, backend='eager') if compiled else bank.observe
    observe(state, torch.tensor([True, False]), None)
    assert bank.codes.tolist() == [[[2., 0.], [5., 0.]], [[5., 0.], [0., 0.]]]
    assert bank.valid.tolist() == [[True, True], [True, False]]


def test_forecast_places_the_last_addressed_slot_in_each_stream():
    from LookForecast import locate_evidence
    rows = torch.tensor([[1, -1, 1], [-1, 0, 2]])
    values = torch.arange(2 * 4 * 3 * 2).reshape(2, 4, 3, 2).float()
    expected = torch.zeros_like(values)
    expected[0, :, 1] = values[0, :, 2]
    expected[1, :, 0] = values[1, :, 1]
    expected[1, :, 2] = values[1, :, 2]
    compiled = torch.compile(locate_evidence, fullgraph=True, backend='eager')
    assert torch.equal(compiled(values, rows, 3), expected)


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


def test_meaning_shares_only_validated_metadata_and_keeps_its_roles():
    from dataclasses import replace
    from Meaning import ConceptualMeaning, expire_meaning_metadata, frozen_metadata, share_meaning_metadata
    from SentenceBoundary import detached
    source = torch.tensor([[.4], [.5], [.6]], requires_grad=True)
    mask = torch.tensor([True, False, True])
    metadata = {'roles': [['sym', 7]]}
    with share_meaning_metadata():
        meaning = ConceptualMeaning(source, mask, bindings=metadata)
        other = replace(meaning, sentence_kind='idea')
        assert other.bindings is meaning.bindings
        assert detached(other.bindings) is meaning.bindings
        metadata['roles'][0][1] = 8
        new = replace(meaning, bindings=metadata)
        assert new.bindings != other.bindings
        with torch.no_grad():
            source.fill_(1.)
            mask.fill_(True)
        assert torch.equal(other.roles.flatten(), torch.tensor([.4, 0., .6]))
        assert other.role_mask.tolist() == [True, False, True]
        assert torch.autograd.grad(other.roles.sum(), source)[0].flatten().tolist() == [1., 0., 1.]
        assert frozen_metadata(meaning.bindings)
        expire_meaning_metadata()
        assert not frozen_metadata(meaning.bindings)
    assert not frozen_metadata(other.bindings)


def test_occupied_row_names_follow_compaction_checkpoint_and_external_writes():
    from copy import deepcopy
    from Memory import LTM0
    store = LTM0(16, capacity=4)
    for _ in range(3):
        store.append_idea(torch.ones(16))
    names = [store.row_name(i) for i in range(3)]
    saved = deepcopy(store.state_dict())
    store.compact_rows([2, 0])
    assert (len(store), store.row_name(1), store.index_of_row(names[2])) == (2, names[2], 1)
    assert store.index_of_row(names[1]) is None
    clone = deepcopy(store)
    clone.row_names[0] = names[2]
    assert clone.index_of_row(names[2]) == 0
    assert store.index_of_row(names[2]) == 1
    store.load_state_dict(saved)
    assert [store.row_name(i) for i in range(len(store))] == names
    store.count.fill_(1)
    assert len(store) == 1 and store.index_of_row(names[2]) is None
    store.count = store.count.new_tensor(3)
    store.row_names = store.row_names.flip(0)
    assert len(store) == 3 and store.index_of_row(names[2]) == 1


@pytest.mark.parametrize('wireless', [False, True])
def test_kept_field_aligns_last_duplicate_seat_and_its_gradient(wireless):
    from SentenceField import select_fields
    from Understanding import ConceptualField
    sources, fields = [], []
    for ids, width in [(torch.tensor([-1, 6, 6]), 2),
                       (torch.tensor([[5, 9], [7, 6]]), 3)]:
        evidence = torch.randn(len(ids), 2, width, 2, requires_grad=True)
        locations = torch.randn(2, width, 2)
        wires = None if wireless and width == 2 else locations
        sources.append(evidence)
        fields.append(ConceptualField(ids, evidence, locations, locations, None, wires))
    kept = select_fields(fields, [0, 1])
    assert kept.concept_ids.tolist() == [-1, 6, 9]
    assert torch.equal(kept.evidence[1, 0, :2], sources[0][2, 0])
    assert torch.equal(kept.evidence[2, 1], sources[1][0, 1])
    assert not kept.evidence[2, 0].any() and not kept.evidence[0, 1].any()
    assert not kept.evidence[:, 0, 2].any() and (kept.wires is None) == wireless
    first, second = torch.autograd.grad(kept.evidence.sum(), sources)
    assert first[:, 0, 0, 0].tolist() == [1., 0., 1.]
    assert second[:, 1].eq(1).all()
    assert not first[:, 1].any() and not second[:, 0].any()


@pytest.mark.parametrize('include_gaps', [False, True])
def test_learned_unit_layout_fullgraph(include_gaps):
    from Meronomy import delivered_bytes, percept_spans, sentence_unit_spans

    raw = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [97, 98, 0, 99, 32, 100, 0],
                        [160, 133, 65, 49, 33, 0, 0], [32, 32, 0, 0, 0, 0, 0],
                        [32, 65, 66, 67, 32, 32, 46]])
    learned = torch.tensor([[(0, 0), (0, 0), (0, 0)], [(0, 2), (2, 3), (4, 7)],
                            [(0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0)],
                            [(1, 2), (2, 4), (6, 7)]])
    def layout(slab, cuts):
        return sentence_unit_spans(delivered_bytes(slab), cuts, include_gaps=include_gaps)
    compiled = torch.compile(layout, backend='eager', fullgraph=True)
    for slab in (raw, raw.flip(0)):
        got = compiled(slab, learned).tolist()
        for row, cuts, staged in zip(slab.tolist(), learned.tolist(), got, strict=True):
            surface = bytes(row).replace(b'\x00', b'')
            expected = [(a, z) for a, z in cuts if a < z <= len(surface)] or percept_spans(surface)
            if include_gaps:
                full, cursor = [], 0
                for a, z in expected:
                    if cursor < a:
                        full.append((cursor, a))
                    full.append((a, z))
                    cursor = z
                if cursor < len(surface):
                    full.append((cursor, len(surface)))
                expected = full
            assert [(a, z) for a, z in staged if z > a] == expected


def test_byte_rows_track_restore_and_leave_previous_gather_intact():
    from Layers import RadixLayer
    from FieldReverse import byte_rows
    store = RadixLayer(8)
    other = RadixLayer(8)
    store.insert(b'a')
    other.insert(b'b')
    before = byte_rows(store)
    store.load_vocab_extras(other.vocab_extras())
    restored = byte_rows(store)
    store.insert(b'c')
    assert before[[97, 98, 99]].tolist() == [0, -1, -1]
    assert restored[[97, 98, 99]].tolist() == [-1, 0, -1]
    assert byte_rows(store)[[97, 98, 99]].tolist() == [-1, 0, 1]


def test_reading_forms_use_ordered_bytes_and_only_this_page():
    from WordSurfaces import reading_forms

    rows = [b' ab ba ab aab  \t', b' ba ab ba aba  \r']
    raw = torch.tensor([list(row) for row in rows])
    cuts = torch.tensor([[(0, 3), (3, 6), (6, 9), (9, 13), (13, 15), (15, 16)]]).expand(2, -1, -1)
    active = torch.ones_like(cuts[..., 0], dtype=torch.bool)
    active[0, 0] = False
    compiled = torch.compile(reading_forms, fullgraph=True, backend='eager')
    for surface in (raw, raw.flip(0)):
        bounds, keys = compiled(surface, cuts, active)
        for b, row in enumerate(surface.tolist()):
            prior = {}
            for w, (lo, hi) in enumerate(cuts[b].tolist()):
                run = bytes(row[lo:hi])
                spelling = run.strip() if active[b, w] else b''
                expected = (lo + len(run) - len(run.lstrip()), lo + len(run.rstrip())) if spelling else (0, 0)
                assert bounds[b, w].tolist() == list(expected)
                assert int(keys[b, w]) == (prior.setdefault(spelling, w) if spelling else -1)


@pytest.mark.parametrize('seats', [12, 320])
def test_novel_spelling_sources_compare_every_byte_and_keep_first_visible_seat(seats):
    from WordSurfaces import spelling_sources
    words = (b'ab\x00', b'ab', b'\xffabcabcd\x00', b'\xffabcabcd\xff', b'\x80', b'ab\x00')
    rows = [b''.join(words), b''.join(reversed(words))]
    surfaces = torch.tensor([list(row) for row in rows])
    offsets = []
    for row in (words, tuple(reversed(words))):
        starts, at = [], 0
        for word in row:
            starts.append((at, at + len(word)))
            at += len(word)
        offsets.append([starts[i % len(row)] for i in range(seats)])
    spans = torch.tensor(offsets)
    visible = torch.ones_like(spans[..., 0], dtype=torch.bool)
    visible[0, :4] = False
    for shared in (True, False):
        result = spelling_sources(surfaces, spans, visible, across_pages=shared).tolist()
        seen = {}
        for page, (row, bounds) in enumerate(zip(rows, offsets)):
            if not shared:
                seen.clear()
            for i, (start, end) in enumerate(bounds):
                seat = page * seats + i if shared else i
                expected = seen.setdefault(row[start:end], seat) if visible[page, i] else -1
                assert result[page][i] == expected


def test_word_property_runs_fullgraph_matches_observed_bytes():
    from Meronomy import word_property_runs
    lut = torch.arange(256) % 7
    surface = torch.tensor([[11, 11, 9, 9, 8, 7, 15, 15, 16, 16, 16, 0],
                            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]])
    spans = torch.tensor([[[0, 8], [8, 11], [0, 0]], [[1, 3], [5, 8], [8, 12]]])
    active = torch.tensor([[True, True, False], [True, False, True]])
    offsets = torch.tensor([[0, 8, -1], [7, 19, 22]])
    capture = torch.compile(word_property_runs, backend='eager', fullgraph=True)
    for slots, digit in ((1, 0), (4, 2)):
        counts, edges = capture(surface, spans, active, offsets, lut, 4, digit, slots, dtype=torch.float32)
        expected = torch.zeros_like(counts)
        expected_edges = torch.zeros_like(edges)
        for b, row in enumerate(surface.tolist()):
            for w, (lo, hi) in enumerate(spans[b].tolist()):
                i, slot = lo, 0
                while active[b,w] and i < hi and slot < slots:
                    signature = int(lut[row[i]])
                    j = i + 1
                    while j < hi and int(lut[row[j]]) == signature and not signature & digit:
                        j += 1
                    if signature and not signature & 4:
                        for byte in row[i:j]:
                            expected[b,w,slot,byte] += 1
                        expected_edges[b,w,slot] = offsets[b,w] + torch.tensor([i-lo, j-lo])
                        slot += 1
                    i = j
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
        assert torch.equal(edges, expected_edges)


def test_all_attested_parts_are_found_on_their_own_contiguous_bytes(monkeypatch):
    from types import SimpleNamespace
    import FieldReverse as field

    # Force collisions so the exact comparison is exercised for every candidate.
    monkeypatch.setattr(field, 'HASH_MODULUS', 2)
    store = SimpleNamespace(inverse_table=[b'a', b'aa', b'ab', b'ba', b'aba', b'aaa', b'zz'])
    values = torch.tensor([list(b'aba aa'), list(b'zz aaa')])
    visible = torch.ones_like(values, dtype=torch.bool)
    visible[1, 1] = False
    compiled = torch.compile(field.form_presence, backend='eager', fullgraph=True)
    for table in (store.inverse_table, [b'a', b'ba', b'aa', b'zz', b'aaa', b'ab', b'aba']):
        store.inverse_table = table
        rows, spellings, groups = field.store_forms(store, values.device)
        for valid in (visible, visible.flip(0)):
            present = compiled(values, valid, spellings, groups)
            expected = []
            for index in rows.tolist():
                run = table[index]
                expected.append(any(bytes(row[i:i+len(run)]) == run and all(mask[i:i+len(run)])
                    for row, mask in zip(values.tolist(), valid.tolist(), strict=True)
                    for i in range(len(row)-len(run)+1)))
            assert present.tolist() == expected
            selected, spelled, kept, cover = field.sentence_forms(store, values, valid, with_cover=True)
            expected_cover = torch.zeros_like(cover)
            for seat, row_id in enumerate(selected.tolist()):
                run = table[row_id]
                for b, (row, mask) in enumerate(zip(values.tolist(), valid.tolist(), strict=True)):
                    for start in range(len(row) - len(run) + 1):
                        stop = start + len(run)
                        if bytes(row[start:stop]) == run and all(mask[start:stop]):
                            expected_cover[b, start:stop, seat] += 1
            assert torch.equal(cover, expected_cover)
            codes = torch.rand(len(selected), 3) * kept[:, None]
            observed = torch.nn.functional.one_hot(values, 256).float() * valid[..., None]
            dense = field.content(observed, torch.zeros(256, 3), spelled, codes)
            torch.testing.assert_close(cover.float() @ codes, dense, rtol=0, atol=0)


def test_reading_property_intersections_follow_the_native_text_chart():
    from functools import reduce
    from operator import and_
    from WordSurfaces import reading_property_signatures, trimmed_spans
    raw = b"  A\xe9 B\t\xff  "
    surface = torch.tensor([list(raw)], dtype=torch.uint8)
    spans = torch.tensor([[(a, b) for a in range(len(raw) + 1) for b in range(a, len(raw) + 1)]])
    lookup = torch.randint(0, 1 << 48, (256,), generator=torch.Generator().manual_seed(15))
    lookup |= 1 << 62
    discard = 5
    wanted = []
    for a, b in spans[0].tolist():
        spelling = raw[a:b].strip().decode('latin1').encode('utf8')
        wanted.append(reduce(and_, (int(lookup[c]) for c in spelling), -1) & ~discard if spelling else 0)
    bounds = trimmed_spans(surface, spans)
    actual = torch.compile(reading_property_signatures, backend='eager', fullgraph=True)(
        surface, bounds, lookup, discard)
    assert actual.tolist() == [wanted]


@pytest.mark.parametrize('symbol_cycle', [False, True])
def test_compiled_spread_preserves_local_stage_gradient(symbol_cycle):
    import ConceptSpread as spread

    # One concept reads one percept, which reads one part. In the second
    # graph a symbol and the concept can also send evidence around a cycle.
    symbols = int(symbol_cycle)
    percept, part = 1 + symbols, 2 + symbols
    parent = torch.tensor([0, percept] + ([0, 1] if symbol_cycle else []))
    child = torch.tensor([percept, part] + ([1, 0] if symbol_cycle else []))
    weight = torch.linspace(.3, .85, len(parent), requires_grad=True)
    graph = spread.Structure(parent, child, weight, child == 1, 1, symbols, 1, 1, ('a',))
    observed = torch.tensor([[[.3, .8], [.6, .1]]])
    parts = observed.flip(-1)

    def local_stage(graph, observed, parts):
        flow = spread._flow(graph, observed, parts, 12, native=True)
        return spread._read_stage(graph, flow, 'concepts'), *flow[:3]

    actual, source, up, down = torch.compile(local_stage, fullgraph=True, backend='eager')(
        graph, observed, parts)
    fan = torch.zeros(graph.size).index_add(0, child, weight.detach()).clamp_min(torch.finfo(weight.dtype).tiny)
    mass = torch.zeros(graph.size).index_add(0, parent, weight.detach()).clamp_min(torch.finfo(weight.dtype).tiny)
    expected_up = torch.zeros_like(source)
    expected_down = torch.zeros_like(source)
    for _ in range(12):
        expected_up = spread._carry(spread.union(source, expected_up), child, parent,
                                    weight.detach() / fan[child], graph.negated, graph.size)
    for _ in range(12):
        expected_down = spread._carry(spread.union(source, expected_up, expected_down), parent, child,
                                      weight.detach() / mass[parent], graph.negated, graph.size)
    own = parent == 0
    local = spread._carry(spread.union(source, expected_up), child[own], parent[own],
                          weight[own] / fan[child[own]], graph.negated[own], graph.size)
    expected = spread.union(local[:1], expected_down[:1])
    for left, right in ((up, expected_up), (down, expected_down), (actual, expected)):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), weight)[0],
                               torch.autograd.grad(expected.sum(), weight)[0])


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_forecast_structure_keeps_zero_and_muted_seats_without_dynamic_outputs(device):
    from types import SimpleNamespace
    import ConceptSpread as spread
    from torch._dynamo.testing import CompileCounterWithBackend
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    sigma_value = torch.tensor([.4, 0., -.6, -0.], device=device, requires_grad=True)
    pi_value = torch.tensor([[.7, .3, .8, .4, .5]], device=device, requires_grad=True)
    rows = torch.tensor([0, 0, 1, 2], device=device)
    inputs = torch.tensor([0, 1, 1, 0], device=device)
    col = torch.tensor([0, -1, 1, 1, 0], device=device)
    against = torch.tensor([False, False, True, False, False], device=device)
    pi_rows = torch.tensor([0, 0, 0, 1, 1], device=device)
    owner = SimpleNamespace(nVectors=3, _concept_allocator=None,
        _expected_knowing=torch.zeros(1, 3, 2, device=device),
        order_zero=SimpleNamespace(nnz=4, nInput=2, entry_row=rows, entry_input=inputs, w=lambda: sigma_value),
        basic_percepts=SimpleNamespace(pi=SimpleNamespace(n_rows=2, nnz=5, keys=('a', 'b'),
            crow=torch.tensor([0, 3, 5], device=device), col=col, value=pi_value, against=against)))
    observed = torch.tensor([[[.8, .1], [.1, .4]], [[.2, .4], [.9, .1]], [[.7, .3], [.4, .6]]], device=device)
    percepts = torch.tensor([[[.4, .3], [.6, .2]], [[.9, .1], [.2, .5]]], device=device)

    def reading():
        graph = spread.structure(owner, observed.device)
        flow = spread._flow(graph, observed, percepts, 8, native=True)
        return torch.cat((spread._read_stage(graph, flow, 'concepts'),
                          spread._read_stage(graph, flow, 'percepts')))

    counter = CompileCounterWithBackend('eager')
    compiled = torch.compile(reading, backend=counter, fullgraph=True)
    for weights in ([.4, 0., -.6, -0.], [0., -.2, .5, .1], [0., 0., 0., 0.]):
        with torch.no_grad():
            sigma_value.copy_(sigma_value.new_tensor(weights))
        live = sigma_value != 0
        keep = ~against & (col >= 0)
        graph = spread.Structure(torch.cat((rows[live], 3 + pi_rows[keep])),
            torch.cat((3 + inputs[live], 5 + col[keep])),
            torch.cat((sigma_value[live].abs(), pi_value[0, keep])),
            torch.cat((sigma_value[live] < 0, torch.zeros_like(col[keep], dtype=torch.bool))),
            3, 0, 2, 2, ('a', 'b'), 0)
        flow = spread._flow(graph, observed, percepts, 8, native=True)
        reference = torch.cat((spread._read_stage(graph, flow, 'concepts'),
                               spread._read_stage(graph, flow, 'percepts')))
        actual = compiled()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        expected_grads = torch.autograd.grad(reference.square().sum(), (sigma_value, pi_value))
        actual_grads = torch.autograd.grad(actual.square().sum(), (sigma_value, pi_value))
        for expected, found in zip(expected_grads, actual_grads, strict=True):
            torch.testing.assert_close(found, expected, rtol=0, atol=0)
    assert counter.frame_count == 1


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_salience_transpose_fullgraph_preserves_request_gradient(device):
    import Salience
    from ConceptSpread import Structure
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    parent = torch.tensor([2, 2, 0, 1, 3, 3], device=device)
    child = torch.tensor([0, 1, 2, 3, 4, 0], device=device)
    weight = torch.tensor([1., 1., .6, .3, .7, 0.], device=device)
    negated = torch.tensor([False, False, True, False, False, False], device=device)
    graph = Structure(parent, child, weight, negated, 2, 1, 1, 1, ('a',))
    request = torch.tensor([[[.3, -.1], [.2, .4]], [[.1, .2], [0., .6]]], device=device, requires_grad=True)
    values = torch.linspace(.1, .9, 40, device=device).reshape(5, 2, 2, 2)
    # The previous compact graph, unrolled with ordinary autograd rather
    # than the custom transpose. Include a Not, a silent seat and a part.
    keep = child < graph.part_start
    a, z = parent[keep], child[keep]
    symbols = ((a >= graph.concepts) & (a < graph.percept_start))[:, None, None, None]
    factor = torch.where(symbols, values[a] / values[z], 1.) * weight[keep, None, None, None]
    source = F.pad(torch.where(request >= 0, request, 0.)[:, :, None].expand(-1, -1, 2, -1), (0, 0, 0, 0, 0, 0, 0, 3))
    expected = source
    for _ in range(4):
        lanes = expected[a]
        lanes = torch.where(negated[keep, None, None, None], lanes.flip(-1), lanes)
        expected = source.index_add(0, z, lanes * factor)
    compiled = torch.compile(Salience._down, backend='eager', fullgraph=True)
    actual = compiled(graph, request, values, 2, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), request)[0],
                               torch.autograd.grad(expected.square().sum(), request)[0], rtol=1e-6, atol=1e-6)


def test_native_forecast_forks_at_the_shared_opening_and_scores_its_own_history():
    from Expectation import PerceptExpectation
    predictor = PerceptExpectation(2, streams=1, capacity=2)
    predictor.known.fill_(True)
    keys = torch.eye(2)[None]
    bank = lambda: [(torch.arange(2), keys)]
    active = torch.ones(1, dtype=torch.bool)
    base = predictor.history(1)
    before = base.fork()
    first, second = base.fork(), base.fork()
    opening = predictor.forecast(bank, 1, history=first)
    predictor.preview(first, opening, keys[:, 0], active)
    greedy_second = predictor.forecast(bank, 1, history=first)
    candidate_first = predictor.forecast(bank, 1, history=second)
    assert torch.equal(candidate_first[0], opening[0])
    assert not torch.equal(greedy_second[0], candidate_first[0])
    _, candidate_loss = predictor.preview(second, candidate_first, keys[:, 1], active)
    reference = .5 * (opening[0] - keys[:, 1]).square().mean(-1)
    assert torch.equal(candidate_loss, reference)
    for old, current in zip(before, base, strict=True):
        assert torch.equal(old, current)


def test_mixed_native_winners_replay_only_their_own_observations_and_gradients():
    from types import SimpleNamespace
    from Expectation import PerceptExpectation
    from ModelAttention import NativeReading, observe_percept, stage_percept_expectation

    owner = PerceptExpectation(2, streams=2, capacity=6)
    model = SimpleNamespace(percept_expectation=owner, symbolSpace=SimpleNamespace(), _sentence_training=True)
    masks = torch.eye(2, dtype=torch.bool)
    both = masks.any(0)
    stage_percept_expectation(model, both[:, None].expand(2, 2))
    opening = owner.history(2)
    paths = [NativeReading(owner, opening, model._percept_expected, model._percept_arrival_loss)
             for _ in range(2)]
    raw = torch.tensor([[[.25, .4], [.8, .1]], [[.9, .7], [.2, .3]]])
    guesses = []
    for path, columns in zip(paths, ((0, 1), (1, 0)), strict=True):
        path.publish(model)
        for turn, col in enumerate(columns):
            expected = torch.full((2, 2), .25 + .1 * turn, requires_grad=True)
            guesses.append(expected)
            observed = raw[:, col]
            payload = (observed, None, None, None, None, None, observed)
            observe_percept(model, (expected, torch.tensor([col])), payload, (None, raw), col,
                            both, path=path, turn=turn)
    assert owner.updates.item() == 0
    NativeReading.select(model, paths, torch.arange(2))
    NativeReading.commit(paths, torch.arange(2))
    assert owner.moments.count.item() == 0
    selected_contexts = [path.history.recent[row] for row, path in enumerate(paths)]
    torch.testing.assert_close(owner.context_recent, torch.stack(selected_contexts))
    control = PerceptExpectation(2, streams=2, capacity=6)
    for turn in range(2):
        for col in range(2):
            control.arrive((guesses[turn], torch.tensor([col])), raw[:, col], masks[col ^ turn], training=True)
    for actual, expected in zip(owner.history(2), control.history(2), strict=True):
        torch.testing.assert_close(actual, expected)
    model._percept_arrival_loss.sum().backward()
    for index, guess in enumerate(guesses):
        winner = index // 2
        col = (index % 2) ^ winner
        torch.testing.assert_close(guess.grad[winner], (guess[winner] - raw[winner, col]) / 2)
        assert not guess.grad[1-winner].any()


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_native_retention_captures_device_winners_and_ignores_rejected_suffix(device):
    """Mixed winners, shared readings and a losing extra look keep causal state."""
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    from Expectation import PerceptExpectation
    from ModelAttention import NativeReading
    from torch.utils._python_dispatch import TorchDispatchMode

    owner = PerceptExpectation(2, streams=3, capacity=8).to(device)
    control = PerceptExpectation(2, streams=3, capacity=8).to(device)
    paths = [NativeReading(owner, owner.history(3), torch.zeros(3, 2, device=device),
                           torch.zeros(3, device=device)) for _ in range(2)]
    active = torch.ones(3, dtype=torch.bool, device=device)
    for path, columns in zip(paths, ((0, 1), (1, 0)), strict=True):
        for turn, column in enumerate(columns):
            observed = torch.full((3, 2), .2 + .1 * column, device=device)
            expected = torch.full_like(observed, .1 * turn)
            path.arrivals[turn, column] = (observed, (expected, torch.tensor([column], device=device)), active)
    # Only path 1 sees the suffix. It must not publish keys, a forecast,
    # context, update count or last-arrival diagnostics if path 0 wins.
    suffix = torch.full((3, 2), .9, device=device)
    paths[1].arrivals[2, 0] = (suffix, (suffix * .5, torch.tensor([7], device=device)), active)
    choices = [paths[0], paths[1], paths[0]]  # an output departure shares path 0
    compiled = torch.compile(NativeReading.commit, backend='eager', fullgraph=True)

    class NoHostReads(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            assert function not in (torch.ops.aten._local_scalar_dense.default,
                                    torch.ops.aten.equal.default, torch.ops.aten.nonzero.default)
            if function == torch.ops.aten._to_copy.default:
                assert not (args[0].device.type == 'mps' and str((kwargs or {}).get('device')) == 'cpu')
            return function(*args, **(kwargs or {}))

    for selected in ((0, 1, 2), (2, 0, 0), (1, 1, 1)):
        # Reference: replay precisely the retained paths at each causal turn.
        winners = torch.tensor(selected, device=device)
        selected_paths = [choices[index] for index in selected]
        keys = sorted(set().union(*(path.arrivals for path in selected_paths)))
        for key in keys:
            observed = torch.zeros(3, 2, device=device)
            expected = torch.zeros_like(observed)
            mask = torch.zeros_like(active)
            for row, path in enumerate(selected_paths):
                if key in path.arrivals:
                    value, forecast, seen = path.arrivals[key]
                    observed[row], expected[row], mask[row] = value[row], forecast[0][row], seen[row]
                    control.known[forecast[1]] = True
            control.arrive((expected, winners[:0]), observed, mask, sentence=True)
        compiled(choices, winners)
        for actual, expected in zip(owner.history(3), control.history(3), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(owner.updates, control.updates, rtol=0, atol=0)
        torch.testing.assert_close(owner.last_arrival['local_loss'], control.last_arrival['local_loss'], rtol=0, atol=0)
        # Audit the actual eager operators too; a captured CPU helper alone
        # says nothing about synchronization on the physical Metal device.
        with NoHostReads():
            NativeReading.commit(choices, winners)
        control.load_state_dict(owner.state_dict())
        control.last_arrival = {key: value.clone() for key, value in owner.last_arrival.items()}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires a Metal device')
def test_metal_trust_radius_matches_binary64_rounding_under_fullgraph():
    import math
    from TrustRadius import gain_bits, metal_radius
    from torch._dynamo.testing import CompileCounterWithBackend
    counts = torch.tensor([0, 1, 2, 3, 7, 32, 113, 120, 2**24 + 1, 2**53 + 3, 2**63 - 1],
                          device='cpu')
    bits = torch.tensor([0, 1, 3, 2**23 - 1, 2**23, 0x3f800000, 0x7f7fffff], dtype=torch.int32,
                        device='cpu')
    agreements = torch.cat((bits.view(torch.float32), torch.tensor([1.974835753440857], device='cpu')))
    views = counts.repeat_interleave(len(agreements))
    inputs = agreements.repeat(len(counts))
    counter = CompileCounterWithBackend('inductor')
    compiled = torch.compile(metal_radius, backend=counter, fullgraph=True)
    for horizon in (3., 10.7):
        bits = gain_bits(horizon)
        gains = torch.tensor((*bits, *([bits[-1]] * (512 - len(bits)))), device='mps')
        expected = torch.tensor([a / n * -math.expm1(-n / horizon) if n else 0.
                                 for a, n in zip(inputs.tolist(), views.tolist())], device='cpu')
        actual = compiled(views.to('mps'), inputs.to('mps'), gains).cpu()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert counter.frame_count == 1


def test_group_centroids_keep_raw_occurrences_and_distinct_placed_members():
    from MereologicalCodes import member_centroids
    codes = torch.tensor([[3., 4.], [0., 0.], [2., -1.], [-4., 3.]])
    members = torch.tensor([[2, 0, 2], [2, 3, 3], [1, 0, 1]])
    ordered = torch.tensor([[0, 2], [2, 3], [0, 1]])
    placed = torch.tensor([True, True, True, False])
    expected = torch.stack((
        torch.stack([torch.nn.functional.normalize(codes[row], dim=0) for row in (0, 2)]).sum(0) / 2,
        codes[[2, 3, 3]].mean(0),
        torch.stack([torch.nn.functional.normalize(codes[row], dim=0) for row in (0, 1)]).sum(0) / 2))
    torch.testing.assert_close(member_centroids(codes, placed, members, ordered), expected, atol=0, rtol=0)
