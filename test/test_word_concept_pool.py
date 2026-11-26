"""MTM chooses words in reading order without spelling or host decisions."""
import copy
import random

import pytest
import torch

from WordConceptPool import WordConceptPool


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    return torch.device(request.param)


def test_nomination_is_visible_in_its_reading_and_not_earlier_streams(device):
    pool = WordConceptPool(32, torch.tensor([71, 75], device=device), threshold=3)
    forms = torch.tensor([[4, 4, -1], [4, 8, 8], [4, 8, -1]], device=device)
    words, arrivals = pool.read(forms)
    assert words.tolist() == [[-1, -1, -1], [-1, -1, -1], [71, -1, -1]]
    assert arrivals[arrivals[..., 0] >= 0].tolist() == [[4, 71]]
    assert pool.counts[[4, 8]].tolist() == [3, 2]
    words, arrivals = pool.read(forms)
    assert words.tolist() == [[71, 71, -1], [71, 75, 75], [71, 75, -1]]
    assert arrivals[arrivals[..., 0] >= 0].tolist() == [[8, 75]]


def test_pool_matches_sequential_readings_including_capacity(device):
    rng = random.Random(1505)
    pool = WordConceptPool(20, torch.tensor([51, 52, 53, 54], device=device), threshold=3)
    counts, words, used = {}, {}, 0
    for _ in range(15):
        rows = [[rng.randrange(-1, 20) for _ in range(9)] for _ in range(4)]
        expected, new = [], []
        for row in rows:
            seen, result = set(), []
            for form in row:
                if form >= 0 and form not in words:
                    if form not in seen:
                        counts[form] = counts.get(form, 0) + 1
                    if counts[form] >= 3 and used < 4:
                        words[form] = 51 + used
                        used += 1
                        new.append([form, words[form]])
                seen.add(form)
                result.append(words.get(form, -1))
            expected.append(result)
        result, arrivals = pool.read(torch.tensor(rows, device=device))
        assert result.tolist() == expected
        assert arrivals[arrivals[..., 0] >= 0].tolist() == new
        assert int(pool.used) == used


def test_frozen_read_and_checkpoint_keep_the_pending_count(device):
    pool = WordConceptPool(20, torch.tensor([51], device=device), threshold=2)
    forms = torch.tensor([[3, 3, -1]], device=device)
    pool.read(forms)
    saved = copy.deepcopy(pool.state_dict())
    result, arrivals = pool.read(forms, learn=False)
    assert result.tolist() == [[-1, -1, -1]]
    assert (arrivals == -1).all()
    for name, value in saved.items():
        assert torch.equal(value, pool.state_dict()[name])
    restored = WordConceptPool(20, torch.tensor([51], device=device), threshold=2)
    restored.load_state_dict(saved)
    a, b = restored.read(forms), pool.read(forms)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert a[0].tolist() == [[51, 51, -1]]


def test_fullgraph_pool_updates_match_eager(device):
    pool = WordConceptPool(20, torch.tensor([51, 52, 53], device=device), threshold=2)
    other = copy.deepcopy(pool)
    captured = torch.compile(other.read, backend='eager', fullgraph=True)
    for rows in ([[3, 3, 2], [2, 9, -1]], [[9, 2, 3], [4, 4, 4]]):
        forms = torch.tensor(rows, device=device)
        eager, compiled = pool.read(forms), captured(forms)
        assert all(torch.equal(a, b) for a, b in zip(eager, compiled))
        assert all(torch.equal(value, other.state_dict()[name])
                   for name, value in pool.state_dict().items())


def test_empty_or_exhausted_pool_keeps_forms_pending(device):
    pool = WordConceptPool(20, torch.empty(0, dtype=torch.long, device=device), threshold=1)
    result, arrivals = pool.read(torch.tensor([[3, -1]], device=device))
    assert result.tolist() == [[-1, -1]]
    assert (arrivals == -1).all()
    result, arrivals = pool.read(torch.empty((2, 0), dtype=torch.long, device=device))
    assert result.shape == (2, 0) and arrivals.shape == (2, 0, 2)


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('text', ['love', 'ab' * 20])
def test_sentence_uses_mtm_name_across_a_change_of_visible_cover(monkeypatch, text):
    from pathlib import Path
    from test_mm_xor import _fresh_model, _PROJECT
    model = _fresh_model(str(Path(_PROJECT) / 'data/XOR_grammar.xml'))[0]
    owner = model._concept_owner()
    original = owner.interpret.install_nomination
    nominations = []

    def admitted(*args, **kwargs):
        result = original(*args, **kwargs)
        nominations.append(result)
        return result

    with torch.no_grad():
        for reading in range(2):
            with monkeypatch.context() as patch:
                patch.setattr(owner.interpret, 'lookup_word', lambda *a, **kw: pytest.fail('sentence used spelling lookup'))
                patch.setattr(owner.interpret, 'install_nomination', admitted)
                model.understand(model.inputSpace.prepInput([text] * 4))
            fi = model.perceptualSpace._forward_input
            if not reading:
                words = fi['pool_word_ids'][:, 0].tolist()
                assert words[:2] == [-1, -1]
                word = words[2]
                assert word > 0 and words[3] == word
                assert int(fi['word_part_mask'][0, 0].sum()) == min(len(text), 32)
            else:
                assert fi['pool_word_ids'][:, 0].tolist() == [word] * 4
                assert int(fi['word_part_mask'][0, 0].sum()) == 1
            model.End()
            model.dispatch_per_row_reset([True] * 4)
            model.perceptualSpace.admit_counted_forms()
    assert nominations == [word]
    assert owner.interpret.word_for_form(text) == word


def test_mtm_checkpoint_restores_word_names_and_pending_readings(device):
    from Memory import MediumTermMemory
    memory = MediumTermMemory()
    memory.word_pool = WordConceptPool(20, torch.tensor([51, 52], device=device), threshold=2)
    forms = torch.tensor([[3, 3, 4]], device=device)
    memory.word_pool.read(forms)
    state = memory.state()
    restored = MediumTermMemory()
    restored.restore(state, device=device)
    expected = memory.word_pool.read(forms)
    actual = restored.word_pool.read(forms)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.usefixtures('eager_reading')
def test_legacy_pending_readings_keep_the_same_nomination_sentence():
    from pathlib import Path
    from test_mm_xor import _fresh_model, _PROJECT
    from nomination_fixtures import lexical_forms, short_of_nomination
    model = _fresh_model(str(Path(_PROJECT) / 'data/XOR_grammar.xml'))[0]
    owner = model._concept_owner()
    raw, parts = lexical_forms(model, ['love'])[0]
    short_of_nomination(owner.interpret, parts, model.wholeSpace.property_rows_for_bytes(raw), form=raw)
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput(['love', '', '', '']))
    word = model.perceptualSpace._forward_input['pool_word_ids'][0, 0]
    assert int(word) > 0
    assert owner.interpret.word_for_form('love') == int(word)
    assert ('form', 'love') not in owner.mtm.sightings


def test_no_free_concept_row_leaves_a_nomination_pending(device):
    pool = WordConceptPool(20, torch.tensor([51, 52], device=device), threshold=1)
    forms = torch.tensor([[4, 3]], device=device)
    words, _ = pool.read(forms, available=forms.new_zeros(()))
    assert words.tolist() == [[-1, -1]] and int(pool.used) == 0
    words, arrivals = pool.read(forms, available=forms.new_ones(()))
    assert words.tolist() == [[51, -1]]
    assert arrivals.tolist() == [[[4, 51], [-1, -1]]]


def test_word_poles_read_pool_names_and_all_denotations_without_host_lookup(device):
    from ModelAttention import word_object_poles
    objects = {31: (42, 45), 32: (43,), 34: ()}
    names = torch.tensor([31, 32, 34], device=device)
    values = torch.tensor([[42, 45], [43, -1], [-1, -1], [-1, -1]], device=device)
    operations = []
    def backend(graph, inputs):
        operations.append(sum(node.op == 'call_function' for node in graph.graph.nodes))
        return graph.forward
    for batch in (4, 16):
        native = torch.tensor([31, 42, 45, 43, -1, 34], device=device)[:, None].expand(-1, batch).clone()
        native[1, ::2] = -1
        words = torch.tensor([31, 32, 34, 88, -1], device=device)[None].expand(batch, -1)
        field = torch.rand(6, batch, 5, 2, device=device)
        captured = torch.compile(word_object_poles, backend=backend, fullgraph=True, dynamic=False)
        actual = captured(native, field, words, names, values)
        expected = torch.zeros_like(actual)
        for b in range(batch):
            for w, word in enumerate(words[b].tolist()):
                if word < 0:
                    continue
                wanted = {word, *objects.get(word, ())}
                seats = [i for i, name in enumerate(native[:, b].tolist()) if name in wanted]
                if seats:
                    expected[b, w] = field[seats, b, w].amax(0)
        assert torch.equal(actual, expected)
    assert operations[0] == operations[1]
    # An empty denotation index still reads the word's own concept.
    assert torch.equal(word_object_poles(native, field, words, names[:0], values[-1:])[:, 2],
                       field[5, :, 2])


def test_seated_reads_share_physical_capacity_across_captured_calls(device):
    names = torch.tensor([31, 32, 33], device=device)
    pool = WordConceptPool(12, names, threshold=1)
    pool.free_rows = names.new_tensor([5, 9])
    read = torch.compile(pool.read_seated, backend='eager', fullgraph=True)
    for forms, learn, expected in [
            ([3, 3], True, [5, 5]), ([4, 3], False, [-1, 5]),
            ([4, 5], True, [9, -1]), ([5, 3], True, [-1, 5])]:
        _, seats, _ = read(names.new_tensor([forms]), learn=learn)
        assert seats.tolist() == [expected]
    assert pool.used.item() == 2


def test_native_field_reads_only_complete_visible_pool_occurrences(device):
    from types import SimpleNamespace
    from ReadingMemory import covered_units, named_extents
    units = torch.tensor([[[0, 3], [3, 6], [6, 8], [8, 9], [9, 10]]], device=device)
    spans = torch.tensor([[[0, 1], [0, 0], [1, 3], [3, 4], [5, 6],
                           [6, 8], [8, 9], [9, 10]]], device=device)
    ids = torch.tensor([[1, -1, 2, 3, 4, 5, 6, 7]], device=device)
    covered = torch.compile(covered_units, backend='eager', fullgraph=True)
    assert covered(ids, spans, units).tolist() == [[True, False, True, True, True]]
    fi = dict(unit_spans=units, pool_visible=ids.new_tensor([[1, 1, 0, 1, 1]]).bool(),
              pool_word_ids=ids.new_tensor([[71, 72, -1, -1, 73]]))
    model = SimpleNamespace(perceptualSpace=SimpleNamespace(_forward_input=fi), _concept_owner=lambda: None)
    extents, names = named_extents(model, (ids, spans))
    assert extents.tolist() == [[[0, 3], [8, 9], [9, 10]]]
    assert names.tolist() == [[71, -1, 73]]


def test_reserved_places_match_the_scalar_allocator_and_leave_rng_untouched(device):
    from ConceptPlacement import reserved_seeds, seed
    for width in (7, 128, 375):
        like = torch.empty(width, device=device)
        symbols = [17, 2003, 2027]
        state = torch.get_rng_state()
        assert torch.equal(reserved_seeds(symbols, like), torch.stack([seed(i, like) for i in symbols]))
        assert torch.equal(torch.get_rng_state(), state)


def test_sentence_denotations_use_each_streams_expected_order_and_score(device):
    from Interpret import preferred_ids
    ids = torch.tensor([[11, 20, 30], [9, 3, -1]], device=device)
    orders = ids.new_tensor([[0, 2, 2], [1, 1, -1]])
    rows = ids.new_tensor([[0, 1, 2], [3, -1, -1]])
    scores = torch.tensor([[.9, 0., 0., 0.], [0., .5, .6, .3],
                           [0., float('nan'), 0., -1.]], device=device)
    read = torch.compile(preferred_ids, backend='eager', fullgraph=True)
    assert read(ids, orders, rows, scores).tolist() == [[11, 3], [30, 9], [20, 3]]
    assert read(ids, orders, rows, scores[:, :0]).tolist() == [[20, 3]] * 3


def test_word_seat_spells_its_stored_form_in_the_nominating_read(device):
    from FormLinks import FormLinks
    from WordConceptPool import candidate_bytes
    table = FormLinks(512, longest=3, rounds=0).to(device)
    raw = "café".encode()
    forms = table.insert(torch.tensor([list(raw), list(b"hello")], device=device),
                         torch.tensor([len(raw), 5], device=device))
    pool = WordConceptPool(512, forms.new_tensor([91, 92]), threshold=1)
    pool.free_rows = forms.new_tensor([2, 6])
    pool.seat_forms = forms.new_full((9,), -1)
    _, seats, _ = pool.read_seated(forms[None], learn=True)
    assert seats.tolist() == [[2, 6]]
    read = torch.compile(candidate_bytes, backend='eager', fullgraph=True)
    wanted = forms.new_tensor([[6, 2, 6], [8, -1, 75]])
    active = forms.new_tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    data, valid = read(pool, table, wanted, active, capacity=5, longest=8)
    assert data.shape == valid.shape == (2, 3, 6)
    assert [bytes(value[mask].tolist()) for value, mask in zip(data[0], valid[0])] == [b'hello', raw, b'']
    assert not valid[1].any() and not valid[..., -1].any()


def test_word_property_meet_keeps_tied_gradients_and_empty_rows(device):
    from types import SimpleNamespace
    from BasicPercepts import BasicPercepts
    groups = [(0, 2), (), (1,), (-1, 0)]
    basic = SimpleNamespace(index={i: i for i in range(4)},
        occurrences=[((), group) for group in groups],
        whole=SimpleNamespace(primitive_properties=SimpleNamespace(on_counts=lambda counts: counts)))
    primitive = torch.ones(2, 3, 4, 3, device=device, requires_grad=True)
    rows = torch.tensor([[[0, 0], [1, 2], [2, -1]], [[3, 0], [7, 2], [0, 1]]], device=device)
    result = BasicPercepts.property_presence(basic, primitive, rows, uses_wholes=True)
    reference = torch.zeros_like(result)
    for row, group in enumerate(groups):
        presence = primitive[..., list(group)].amin(-1) if group else torch.ones_like(primitive[..., 0])
        reference = reference + presence[..., None] * (rows == row).unsqueeze(-2)
    assert torch.equal(result, reference)
    weights = torch.arange(result.numel(), device=device, dtype=result.dtype).reshape_as(result)
    got, = torch.autograd.grad((weights * result).sum(), primitive)
    expected, = torch.autograd.grad((weights * reference).sum(), primitive)
    assert torch.equal(got, expected)
    empty = BasicPercepts.property_presence(basic, primitive, -torch.ones_like(rows))
    assert not empty.requires_grad and not empty.any()


def test_nominated_characters_follow_form_rows_in_first_occurrence_order(device):
    from FormLinks import FormLinks
    from WordConceptPool import registration_witnesses
    from WordIdentity import base_atoms, bytes_of
    words = ['é🍋é!'.encode(), b'\x80\xe2\x82\x80\x80', b'\xf0\x9f', b'']
    lengths = torch.tensor(list(map(len, words)), device=device)
    data = lengths.new_tensor([list(word.ljust(max(map(len, words)), b'\0')) for word in words])
    table = FormLinks(512, longest=data.shape[1], rounds=0).to(device)
    forms = table.insert(data, lengths)
    read = torch.compile(registration_witnesses, backend='eager', fullgraph=True)
    characters, contained = read(table, forms, data, lengths)
    for word, rows, addresses in zip(words, characters.tolist(), contained.tolist(), strict=True):
        expected = [bytes_of(char) for _, char in base_atoms(word)]
        result = [(word[pos:pos + size], row) for pos, (row, size) in enumerate(rows) if size]
        assert [text for text, _ in result] == expected
        for text, row in result:
            assert table.read(data.new_tensor([list(text)]), lengths.new_tensor([len(text)]),
                              admit=False).item() == row
        plain = word.decode('utf8', 'surrogateescape')
        all_bytes = {word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + 1)}
        aligned = {bytes_of(plain[i:j]) for i in range(len(plain)) for j in range(i + 1, len(plain) + 1)}
        for octets in all_bytes:
            row = table.read(data.new_tensor([list(octets)]), lengths.new_tensor([len(octets)]),
                             admit=False).item()
            assert any(address // 2 == row for address in addresses)
            assert (2 * row + 1 in addresses) == (octets in aligned)
    assert read(table, forms, data[:, :0], lengths * 0)[0].shape == (len(words), 0, 2)


def test_contained_rows_preserve_admission_order_and_taught_runs():
    from FormLinks import FormLinks
    from Layers import RadixLayer
    from WordConceptPool import registration_holders, registration_witnesses
    from WordIdentity import WordIdentity
    words = [b'cab', b'ab', b'abcab', b'bc', 'éab'.encode(), b'\xa9a', b'\xa9ab']
    table = FormLinks(1024, longest=5, rounds=0)
    lengths = torch.tensor(list(map(len, words)))
    data = lengths.new_tensor([list(word.ljust(5, b'\0')) for word in words])
    forms = table.insert(data, lengths)
    characters, parts = registration_witnesses(table, forms, data, lengths)
    catalogs = []
    for _ in range(2):
        store = RadixLayer(64, initial_cap=512)
        identity = store.identity = WordIdentity(store)
        identity.bind_forms(table)
        catalogs.append(identity)
    reference, native = catalogs
    for i, raw in enumerate(words):
        if i == 2:
            for identity in catalogs:
                identity.admit_part('ca')
        reference.admit(raw)
        if i == 0:
            native.admit(raw)
        else:
            elements = [(raw[pos:pos + n], row) for pos, (row, n) in
                        enumerate(characters[i].tolist()) if n]
            held = registration_holders(native, table, forms[i:], data.shape[1])
            native.admit(raw, form_row=forms[i].item(), witnesses=(elements, parts[i].tolist()),
                         holder_rows=held[0].tolist())
            assert native.holders == reference.holders and native.inside == reference.inside
        assert native.vocab_extras() == reference.vocab_extras()
        seats = sorted(set(native.atoms.values()) | set(native.word_rows.values()))
        for column in ('W', 'fixed_row_codes', 'fixed_form_rows', 'fixed_row_mask'):
            assert torch.equal(getattr(native.store._basis, column)[seats],
                               getattr(reference.store._basis, column)[seats]), column


def test_wide_containment_uses_one_address_per_form():
    from FormLinks import FormLinks
    from WordConceptPool import registration_witnesses
    table = FormLinks(512, longest=48, rounds=0)
    raw = ('é' * 19).encode() + b'\xa9'
    data = torch.tensor([list(raw)])
    length = torch.tensor([len(raw)])
    roots = table.insert(data, length)
    chars, addresses = registration_witnesses(table, roots, data, length)
    assert addresses.shape == (1, table.capacity)
    # A lone continuation byte is both an interior byte and the final
    # escaped character. The character-aligned occurrence takes precedence.
    assert addresses[0, 0xa9].item() == 2 * 0xa9 + 1
    assert chars[0, -1].tolist() == [0xa9, 1]


def test_native_word_support_retiles_formed_parts_before_membership(device):
    from ModelAttention import canonical_word_parts
    from WordConceptPool import supported_parts
    leaves = {7: (1, 2), 8: (2, 3), 9: (1, 2, 1)}
    words = torch.tensor([[[7, 8, -1], [1, -1, 8]], [[7, 1, 2], [-1, -1, -1]]], device=device)
    def read(parts, seats, entries):
        current = canonical_word_parts(parts, leaves)
        return current, supported_parts(current, seats, entries, 16)
    seats = words.new_tensor([[0, 0], [1, -1]])
    allowed = words.new_tensor([3, 7, 8, 16 + 9])
    canonical, known = torch.compile(read, backend='eager', fullgraph=True)(words, seats, allowed)
    assert [[r[r >= 0].tolist() for r in page] for page in canonical] == [[[7, 8], [7, 3]], [[9, 2], []]]
    assert known.tolist() == [[True, True], [False, True]]


def test_mixing_columns_require_one_active_ending_within_the_word(device):
    from WordSurfaces import mixing_columns
    units = torch.tensor([[[0, 3], [4, 7], [8, 10], [0, 0]]], device=device)
    spans = units.new_tensor([[[0, 3], [1, 3], [3, 7], [4, 7], [8, 10]]])
    active = units.new_tensor([[1, 1, 1, 1, 0]]).bool()
    seats = units.new_tensor([[0, 0, -1, 1, 2]])
    assert mixing_columns(units, seats, spans, active, False).tolist() == [[-1, 3, -1, -1]]
    assert mixing_columns(units, seats, None, active, False).tolist() == [[1, 3, 4, -1]]
