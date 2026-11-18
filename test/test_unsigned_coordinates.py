"""Location/time phase bands use Alec's (sinusoid + 1) / 2 storage law."""
import pytest
import torch

from Spaces import WhereEncoding, WhenStartDurationEncoding, DocumentWhenEncoding, WhenRangeEncoding


@pytest.mark.parametrize('kind', ['where', 'onset', 'document'])
def test_unsigned_ladder_preserves_integer_addresses_and_phase(kind):
    encoder = (WhereEncoding(nWhere=4, nWhen=4) if kind == 'where' else
               WhenStartDurationEncoding(n_when=4) if kind == 'onset' else
               DocumentWhenEncoding(n_when=4)).set_capacity((1 << 32)-1)
    points = torch.tensor([0, 1, 63, 128, 255, 8191, 8192, 2**24+1, 2**31, 2**32-1])
    bands = encoder.encode(points)
    assert torch.isfinite(bands).all() and (bands >= 0).all() and (bands <= 1).all()
    torch.testing.assert_close(encoder.decode_index(bands), points, rtol=0, atol=0)
    phase = 2*bands-1
    radius = torch.linalg.vector_norm(phase.reshape(-1, 2), dim=-1)
    assert (radius > .999).all() and (radius <= 1.000001).all()
    if kind != 'where':
        shifted = encoder.shift_time(bands, 17)
        assert (shifted >= 0).all() and (shifted <= 1).all()
        torch.testing.assert_close(encoder.decode_index(shifted), points+17, rtol=0, atol=0)


def test_unsigned_range_preserves_extent_and_rotation():
    encoder = WhenRangeEncoding(256, n_when=2)
    band = encoder.encode(137, 149)
    assert (band >= 0).all() and (band <= 1).all()
    start, end = encoder.decode_span(band)
    torch.testing.assert_close(torch.stack((start,end)), torch.tensor([-119.,-107.]), atol=1e-4, rtol=0)
    shifted = encoder.shift_time(band, -128)
    assert (shifted >= 0).all() and (shifted <= 1).all()
    start, end = encoder.decode_span(shifted)
    torch.testing.assert_close(torch.stack((start,end)), torch.tensor([9.,21.]), atol=1e-4, rtol=0)


def test_centered_checkpoint_keeps_addresses_empty_bands_and_content():
    from copy import deepcopy
    from Memory import LTM0
    from RowTime import encoder
    store = LTM0(12, capacity=4, percept_width=8)
    value = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,.2,.4,.6,.8])
    store.append_idea(value, trust=.7)
    state = deepcopy(store.state_dict())
    original = state['slots'].clone()
    state['truth_schema'].fill_(17)
    phase = state['slots'][..., :8].reshape(4,2,4)
    phase.copy_(torch.where(phase.ne(0).any(-1,keepdim=True), 2*phase-1, 0.))
    before = state['slots'].clone()
    restored = LTM0(12, capacity=4, percept_width=8)
    with pytest.warns(UserWarning, match='sinusoids'):
        restored.load_state_dict(state)
    torch.testing.assert_close(restored.slots, original, rtol=0, atol=0)
    torch.testing.assert_close(state['slots'], before, rtol=0, atol=0)
    assert restored.occurrence_of(0) == store.occurrence_of(0)
    assert int(encoder().decode_index(restored.time[0])) == store.row_name(0)
    again = LTM0(12, capacity=4, percept_width=8)
    again.load_state_dict(deepcopy(restored.state_dict()))
    torch.testing.assert_close(again.slots, original, rtol=0, atol=0)


def test_checkpoint_sidecar_conversion_keeps_empty_bands_and_is_once_only():
    from CoordinateCheckpoint import migrate
    old = torch.tensor([0.,1.,0.,1.])
    source = dict(state_dict={}, structural_extras=dict(
        where=old, when=torch.zeros(4), point=torch.cat((old, old, torch.tensor([.3,.7])))),
        training_state=dict(corpus_runtime=dict(stm=dict(owner=dict(fields=dict(
            _live_chunk_where=old[None,None], _live_chunk_when=old[None,None]))))))
    with pytest.warns(UserWarning, match='observation bands'):
        result = migrate(source, 8)
    torch.testing.assert_close(result['structural_extras']['where'], (old+1)/2)
    assert result['structural_extras']['when'].count_nonzero() == 0
    for value in result['training_state']['corpus_runtime']['stm']['owner']['fields'].values():
        torch.testing.assert_close(value, ((old+1)/2)[None,None])
    torch.testing.assert_close(result['structural_extras']['point'][8:], torch.tensor([.3,.7]))
    assert migrate(result, 8) is result
    torch.testing.assert_close(source['structural_extras']['where'], old)


def test_checkpoint_learned_sparse_weights_are_not_located_observations():
    from CoordinateCheckpoint import migrate
    weights = torch.linspace(-1, 1, 24)
    point = torch.tensor([0.,1.,0.,1.,0.,1.,0.,1.,.3,.7])
    source = dict(state_dict={'layer.truths': point[None]}, structural_extras=dict(
        allocator=dict(layers={1: dict(nInput=24, values=weights, values_requires_grad=True)},
                       singleton_members={9: (4, 7, point)})))
    with pytest.warns(UserWarning, match='observation bands'):
        result = migrate(source, 8)
    torch.testing.assert_close(result['structural_extras']['allocator']['layers'][1]['values'], weights)
    expected = torch.cat(((point[:8]+1)/2, point[8:]))
    torch.testing.assert_close(result['structural_extras']['allocator']['singleton_members'][9][2], expected)
    torch.testing.assert_close(result['state_dict']['layer.truths'][0], expected)


def test_absent_time_band_retains_zero_tick_read():
    from RowTime import encoder
    clock = encoder()
    assert clock.decode_index(torch.zeros(4)).item() == 0
    assert clock.encode(0).ne(0).all() and clock.decode_index(clock.encode(0)).item() == 0
    for codec in (clock, WhenStartDurationEncoding(n_when=4), WhenRangeEncoding(n_when=2)):
        empty = torch.zeros(codec.nDim)
        torch.testing.assert_close(codec.shift_time(empty, 11), empty, rtol=0, atol=0)


@pytest.mark.parametrize('paired', [False, True])
def test_bind_compares_content_and_preserves_occurrence_coordinates(paired):
    from Memory import STMChunks as C
    from RowTime import encoder
    location = WhereEncoding(nWhere=4, nWhen=4).set_capacity(100)
    clock = encoder()
    content = torch.tensor([.2, .2])
    whole = torch.tensor([.8, .8])
    query = torch.cat((content, location.encode(20), clock.encode(1)))
    candidate = torch.cat((whole, location.encode(10), clock.encode(0)))
    if paired:
        query = torch.cat((query, content))
        candidate = torch.cat((candidate, whole))
    width = len(query)
    state = C.empty(1, 8, width, device=query.device, dtype=query.dtype)
    state, seat = C.deposit(state, candidate[None], torch.tensor([True]),
        identity=7, order=0)
    before = tuple(value.clone() for value in state)
    points, ids, indices, valid = C.bind_menu(state, query[None, None],
        orders=torch.tensor([[0]]), meaning_start=10 if paired else 0,
        percept_band=(2, 10, width))
    at = int(seat) + 1
    assert valid[0, 0, at] and ids[0, 0, at] == 7
    assert indices[0, 0, at] == seat
    torch.testing.assert_close(points[0, 0, at], candidate, rtol=0, atol=0)
    for actual, prior in zip(state, before):
        torch.testing.assert_close(actual, prior, rtol=0, atol=0)
    # Incompatible content is still rejected, even at the same occurrence.
    incompatible = query.clone()
    incompatible[10 if paired else 0] = .9
    _, _, _, valid = C.bind_menu(state, incompatible[None, None],
        orders=torch.tensor([[0]]), meaning_start=10 if paired else 0,
        percept_band=(2, 10, width))
    assert not valid[0, 0, at]


def test_bind_bank_with_form_only_interpreter_offers_only_dummy():
    from types import SimpleNamespace
    from ReferenceContext import prepare_operands
    from Memory import STMChunks as C

    width = 10
    value = torch.full((1, 1, width), .5)
    state = C.empty(1, 8, width, device=value.device, dtype=value.dtype)
    state, _ = C.deposit(state, value[:, 0], torch.tensor([True]), identity=7, order=0)
    # Identical forms would pass containment if the reader starts at zero.
    mereology = SimpleNamespace(percept_event_width=width, meaning_pairs=0)
    bank = SimpleNamespace(interpreter=SimpleNamespace(owner=SimpleNamespace(
        similarity_codebook=SimpleNamespace(mereology=mereology))))
    rule = SimpleNamespace(method_name='bind')
    integers = torch.zeros(1, 2, dtype=torch.long)
    menu = prepare_operands(value.expand(1, 2, width), integers, integers, integers,
        orders=integers, rules=(rule,), unary_rules=(), bank=bank, live=None,
        active=torch.tensor([[True, False]]), chunk_state=state,
        percept_band=(2, 10, width))
    assert menu['binary_valid'].sum() == 1
    assert menu['binary_valid'][..., torch.tensor(menu['binary_bind_dummy'])].all()
