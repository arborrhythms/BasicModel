"""Location/time phase bands use Alec's (sinusoid + 1) / 2 storage law."""
import pytest
import torch

from Spaces import WhereEncoding, WhenStartDurationEncoding, DocumentWhenEncoding, WhenRangeEncoding
from architecture import WHERE_WIRES, WHEN_WIRES

# A percept's band, its .where and its .when (8 -> 16).
BAND = WHERE_WIRES + WHEN_WIRES


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


def test_centered_checkpoint_is_refused_and_the_current_one_keeps_addresses_and_content():
    from copy import deepcopy
    from Memory import LTM0
    store = LTM0(BAND + 4, capacity=4, percept_width=BAND)
    value = torch.cat((torch.zeros(BAND), torch.tensor([.2,.4,.6,.8])))
    store.append_idea(value, trust=.7)
    state = deepcopy(store.state_dict())
    original = state['slots'].clone()
    state['truth_schema'].fill_(17)
    restored = LTM0(BAND + 4, capacity=4, percept_width=BAND)
    with pytest.raises(ValueError, match='before the one clock is refused'):
        restored.load_state_dict(state)
    again = LTM0(BAND + 4, capacity=4, percept_width=BAND)
    again.load_state_dict(deepcopy(store.state_dict()))
    torch.testing.assert_close(again.slots, original, rtol=0, atol=0)
    assert again.occurrence_of(0) == store.occurrence_of(0)


def test_checkpoint_sidecar_conversion_keeps_empty_bands_and_is_once_only():
    from CoordinateCheckpoint import migrate
    old = torch.tensor([0.,1.,0.,1.])
    # A saved place has the four wires of each axis; a time has four.
    where = old.repeat(WHERE_WIRES // 4)
    source = dict(state_dict={}, structural_extras=dict(
        where=where, when=torch.zeros(WHEN_WIRES), point=torch.cat((where, old, torch.tensor([.3,.7])))),
        training_state=dict(corpus_runtime=dict(stm=dict(owner=dict(fields=dict(
            _live_chunk_where=where[None,None], _live_chunk_when=old[None,None]))))))
    with pytest.warns(UserWarning, match='observation bands'):
        result = migrate(source, BAND)
    torch.testing.assert_close(result['structural_extras']['where'], (where+1)/2)
    assert result['structural_extras']['when'].count_nonzero() == 0
    fields = result['training_state']['corpus_runtime']['stm']['owner']['fields']
    for name, saved in (('_live_chunk_where', where), ('_live_chunk_when', old)):
        torch.testing.assert_close(fields[name], ((saved+1)/2)[None,None])
    torch.testing.assert_close(result['structural_extras']['point'][BAND:], torch.tensor([.3,.7]))
    assert migrate(result, BAND) is result
    torch.testing.assert_close(source['structural_extras']['where'], where)


def test_checkpoint_learned_sparse_weights_are_not_located_observations():
    from CoordinateCheckpoint import migrate
    weights = torch.linspace(-1, 1, 24)
    point = torch.cat((torch.tensor([0.,1.]).repeat(BAND // 2), torch.tensor([.3,.7])))
    source = dict(state_dict={'layer.truths': point[None]}, structural_extras=dict(
        allocator=dict(layers={1: dict(nInput=24, values=weights, values_requires_grad=True)},
                       singleton_members={9: (4, 7, point)})))
    with pytest.warns(UserWarning, match='observation bands'):
        result = migrate(source, BAND)
    torch.testing.assert_close(result['structural_extras']['allocator']['layers'][1]['values'], weights)
    expected = torch.cat(((point[:BAND]+1)/2, point[BAND:]))
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
def test_single_word_bind_is_independent_of_occurrence_coordinates(paired):
    from Memory import STMChunks as C
    from RowTime import encoder
    from test_conceptual_match import word_case, word_menu
    state, physical, seat, word, candidate, inputs, interpreter = word_case(paired)
    location = WhereEncoding(nWhere=4, nWhen=4).set_capacity(100)
    clock = encoder()
    before = tuple(value.clone() for value in state)
    menus = []
    for place, time in ((20, 1), (10, 0)):
        mention = word.clone()
        mention[2:6], mention[6:10] = location.encode(place), clock.encode(time)
        menus.append(word_menu(state, mention, inputs, interpreter))
        at = int(seat)+1
        assert menus[-1]['binary_valid'][0, 0, at]
        assert menus[-1]['binary_refs'][0, 0, at, 1] == 7
        assert menus[-1]['binary_choices'][0, 0, at, 1] == seat
        torch.testing.assert_close(menus[-1]['right'][0, 0, at], candidate, rtol=0, atol=0)
    torch.testing.assert_close(menus[0]['binary_valid'], menus[1]['binary_valid'], rtol=0, atol=0)
    for actual, prior in zip(state, before):
        torch.testing.assert_close(actual, prior, rtol=0, atol=0)
    # Incompatible object content is still refused at the same word place.
    inputs._ar_word_object_atoms[0, 0, 10 if paired else 0] = .9
    if paired:
        altered = list(state)
        altered[C.TRUTH] = state[C.TRUTH].clone()
        altered[C.TRUTH][0, physical, 1] = .2
        state = tuple(altered)
    assert not word_menu(state, mention, inputs, interpreter)['binary_valid'][0, 0, at]


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
        active=torch.tensor([[True, False]]), chunk_state=state)
    assert menu['binary_valid'].sum() == 1
    assert menu['binary_valid'][..., torch.tensor(menu['binary_bind_dummy'])].all()
