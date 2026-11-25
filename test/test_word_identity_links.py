"""A form's links, found from the lists of its parts and characters, equal
the links found by looking through every stored form (item 1.75)."""
from test_item17_code import WORDS, coded


EXTRA = (b'a', b'ab', b'ba', b'bab', b'abc', b'cab', b'ar', b'arch', b'h', b'he', b'there is', b'cat cats')


def test_device_form_seats_read_repaired_codes_and_reject_ambiguous_parts():
    import torch
    from WordIdentity import WordIdentity
    from Layers import RadixLayer
    identity = WordIdentity(RadixLayer(83, initial_cap=280), pair_dim=65, property_width=18)
    for raw in (b'aab', b'baa', b'bab', b'baab'):
        identity.admit(raw)
    codes = torch.stack([identity.code(raw) for raw in identity.words])
    for width in (7, 64, 65, 83):
        queries = torch.cat((codes[:, :width], torch.zeros_like(codes[:1, :width])))
        assert identity.read_forms(queries) == [identity.read(q) for q in queries]
    # Two rows can disagree in the property lanes of the same packed limb.
    bank = identity._packed
    bank[1, 0] = bank[0, 0]
    bank[1, 1] = (bank[1, 1] & -2) | (bank[0, 1] & 1)
    assert identity.form_seats(codes[:1, :65]).item() == -1


def through_every_form(identity):
    words = list(identity.words)
    holders = {t: {w for w in words if w != t and t in w} for t in words}
    inside = {t: {w for w in words if w != t and w in t} for t in words}
    return holders, inside


def test_links_equal_the_look_through_every_stored_form():
    identity, *_ = coded(WORDS + EXTRA)
    holders, inside = through_every_form(identity)
    assert identity.holders == holders and identity.inside == inside
    assert any(identity.holders[w] for w in identity.words)


def test_links_after_a_load_equal_the_look_through_every_stored_form():
    identity, *_ = coded(WORDS)
    saved = identity.vocab_extras()
    other, *_ = coded(EXTRA[:3])
    other.load_vocab_extras(saved)
    for word in (b'arc', b'is this', b'c'):
        other.admit(word)
    holders, inside = through_every_form(other)
    assert other.holders == holders and other.inside == inside


def test_extent_read_joins_attested_parts_and_preserves_both_property_lanes(monkeypatch):
    import torch
    identity, basic, book, _ = coded((b'calaba', b'cabala', b'aba', b'ab', 'café'.encode()))
    data = list('calaba café cabala'.encode())
    raw = torch.tensor([data, list(reversed(data))])
    spans = torch.tensor([[(0, 6), (0, 0), (7, 12), (13, 19), (0, 3), (4, 6)],
                          [(0, 6), (1, 4), (0, 1), (4, 7), (2, 5), (0, 0)]])
    percepts = (None, None, book.primitive_properties, raw, None)
    def read():
        what, away = basic._occurrence_what(percepts, spans, (0, 1, 3))
        grad, = torch.autograd.grad(what.sum() + .3 * away.sum(), book.primitive_properties.members)
        return what.detach(), away.detach(), grad
    actual = read()
    with monkeypatch.context() as m:
        m.setattr(identity, 'occurrence_bank', lambda device: None)
        reference = read()
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0][0, 0], actual[0][0, 3])


def test_device_part_join_preserves_tie_gradients_for_repeated_spelling():
    import torch
    from BasicPercepts import occurrence_parts
    raw = torch.tensor([[97, 98, 97, 98]])
    spans = torch.tensor([[(0, 4), (1, 3), (2, 4), (0, 0)]])
    spellings = (torch.tensor([[97], [98]]), torch.tensor([[97, 98]]))
    codes = torch.tensor([[1., 0.], [1., 1.], [1., 0.]], requires_grad=True)
    catalog = torch.eye(3)
    compiled = torch.compile(occurrence_parts, fullgraph=True, backend='eager')
    parts, tight = compiled(raw, spans, spellings, codes, catalog)
    torch.testing.assert_close(tight[0], torch.tensor([[1., 1., 1.], [1., 1., 0.],
                                                       [1., 1., 1.], [1., 1., 1.]]))
    reference = torch.stack([codes.amax(0), codes[:2].amax(0), codes.amax(0), codes[0] * 0])
    torch.testing.assert_close(parts[0], reference)
    torch.testing.assert_close(torch.autograd.grad(parts.sum(), codes)[0],
                               torch.autograd.grad(reference.sum(), codes)[0])


def test_packed_parts_keep_high_bits_and_pad_partial_limbs():
    import torch
    from types import SimpleNamespace
    from WordIdentity import WordIdentity
    layout = SimpleNamespace(total=129, limbs=3)
    values = torch.zeros(3, layout.total)
    values[0, [0, 63, 64, 127, 128]] = 1.
    values[1].fill_(1.)
    values[2, ::2] = .5
    expected = torch.tensor([[-(1 << 63) + 1, -(1 << 63) + 1, 1], [-1, -1, 1], [0, 0, 0]])
    packed = WordIdentity._pack(layout, values)
    assert torch.equal(packed, expected)
    assert torch.equal(WordIdentity._unpack(layout, packed), values.eq(1).long())
    assert WordIdentity._pack(layout, values[:0]).shape == (0, 3)


def test_batched_spelling_matches_scalar_inverse_with_both_code_widths():
    import torch
    identity, *_ = coded((b'calaba', b'cabala', b'ab', b'ba'))
    forms = torch.stack([identity.code(word) for word in identity.words])
    # Graded entries below one do not select the corresponding coordinates.
    forms = torch.cat((forms, forms[:1] * .75, torch.zeros_like(forms[:1])))
    for width in (identity.width, identity.total):
        inputs = forms[:, :width]
        assert identity.read_forms(inputs) == [identity.read(form) for form in inputs]
        assert identity.read_forms(inputs[:0]) == []


def test_inverse_follows_ordered_native_children_after_restore(monkeypatch):
    from FieldReverse import spelling

    words = (b'a', b'b', b'ab', b'abab', b'aabb')
    identity, basic, _, rows = coded(words)
    saved = identity.store.vocab_extras()
    other, restored, *_ = coded(words)
    other.store.load_vocab_extras(saved)
    for current, reader in ((identity, basic), (other, restored)):
        def forbidden_name(row):
            raise AssertionError('inverse requested the whole spelling')
        monkeypatch.setattr(reader, 'form_of', forbidden_name)
        assert spelling(reader, rows[b'abab']) == b'abab'
        assert spelling(reader, rows[b'aabb']) == b'aabb'
        # A deliberately reversed native definition changes the inverse even
        # though the named form and its orthographic lookup are unchanged.
        root = current.word_rows[b'aabb']
        current.store.part_groups[root] = tuple(reversed(current.store.part_groups[root]))
        assert spelling(reader, rows[b'aabb']) != b'aabb'


def test_compiled_concatenation_keeps_null_bytes_and_empty_forms():
    import torch
    from FieldReverse import concatenate_parts

    parts = torch.tensor([[[97, 0], [98, -1], [98, -1]], [[-1, -1]] * 3])
    expected = torch.tensor([[97, 0, 98, 98, -1], [-1] * 5])
    result = torch.compile(concatenate_parts, fullgraph=True, backend='eager')(parts, 5)
    torch.testing.assert_close(result, expected)


def test_escaped_part_starts_find_the_parts_of_every_utf8_cut():
    import torch
    from BasicPercepts import escaped_part_starts
    from WordIdentity import text_of

    raw = b'x' + 'é中😀'.encode() + b'\xff\xc0\xed\xa0\x80\xf4\x90\x80\x80'
    identity, *_ = coded((b'x', b'x\xc3', 'é'.encode(), '中'.encode(), '😀'.encode()))
    for byte in raw:
        identity._atom(('part', text_of(bytes([byte]))))
    # The learned part x + escaped C3 must stop matching where C3 begins é.
    intervals = [(a, z) for a in range(len(raw)) for z in range(a + 1, len(raw) + 1)]
    surface = torch.tensor(list(raw)).expand(len(intervals), -1)
    spans = torch.tensor(intervals)[:, None]
    keys, rows, _, _ = identity.occurrence_bank(surface.device, escaped_elements=True)
    first = torch.compile(escaped_part_starts, backend='eager', fullgraph=True)(surface, spans, keys)
    for b, (lo, hi) in enumerate(intervals):
        available = [identity.atoms[atom] for atom in identity.atoms_of(raw[lo:hi]) if atom in identity.atoms]
        assert set(rows[first[b, 0] < len(raw)].tolist()) == set(available)


def test_containment_accidents_preserve_direction_and_linked_parts_in_both_limbs():
    import torch
    from WordIdentity import containment_accidents

    # One coordinate is bit 63, another is bit 64. Rows 1 and 2
    # have identical codes but no attested link to one another.
    bit = -(1 << 63)
    codes = torch.tensor([[bit, 0], [bit, 1], [bit, 1], [0, 1]])
    rows = torch.tensor([2, 1])
    # Only form 1 really contains form 0.
    linked = torch.tensor([4])
    matches = containment_accidents(codes, rows, linked)
    assert matches.tolist() == [[[True, True, False, True], [False, False, True, True]],
                                [[False, True, False, False], [False, False, True, False]]]


def test_tensor_repair_selects_private_coordinates_and_reports_exhausted_holders():
    from WordIdentity import distinguishing_coordinates
    import torch

    # Both pages must distinguish the first two outers. Their local usage
    # differs, and the final outer on page one has no available coordinate.
    lacking = torch.tensor([[[1, 1, 0], [0, 0, 1], [0, 0, 0]]] * 2, dtype=torch.bool)
    kept = torch.ones(2, 3, dtype=torch.bool)
    opposed = torch.zeros_like(kept)
    usage = torch.tensor([[0, 1, 0], [1, 0, 0]])
    pending = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    given, remaining = distinguishing_coordinates(lacking, kept, opposed, torch.zeros_like(usage), usage, pending)
    assert given.tolist() == [[True, False, True], [False, True, True]]
    assert remaining.tolist() == [[False, False, False], [False, False, True]]


def test_form_repairs_and_restore_reuse_the_reserved_native_rows():
    import torch
    from Layers import RadixLayer
    from WordIdentity import WordIdentity
    from test_item17_code import table
    words = (b'aab', b'baa', b'bab', b'baab', b'baba', b'abab', b'aabb', b'calaba', b'cabala')
    results = []
    for device in ('cpu', 'mps') if torch.backends.mps.is_available() else ('cpu',):
        store = RadixLayer(128, initial_cap=256).to(device)
        identity = WordIdentity(store, property_width=64, basic_capacity=32).to(device)
        store.identity = identity
        properties = table().to(device)
        identity.basic.bind(properties)
        buffers = (identity._packed.data_ptr(), identity.usage.data_ptr())
        for word in words:
            identity.admit(word)
        saved = identity.vocab_extras()
        before = identity._packed[:len(words)].clone()
        assert identity._packed.device.type == device
        assert len(identity._packed) == store._capacity
        identity.load_vocab_extras(saved)
        assert (identity._packed.data_ptr(), identity.usage.data_ptr()) == buffers
        assert torch.equal(identity._packed[:len(words)], before)
        assert identity.accidents() == ([], [])
        results.append((before.cpu(), saved['usage'], saved['given'], identity.word_keys))
    for packed, usage, given, keys in results[1:]:
        assert torch.equal(packed, results[0][0])
        assert (usage, given, keys) == results[0][1:]
