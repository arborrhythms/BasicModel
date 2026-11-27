"""A form's links, found from the lists of its parts and characters, equal
the links found by looking through every stored form (item 1.75)."""
from test_item17_code import WORDS, coded


EXTRA = (b'a', b'ab', b'ba', b'bab', b'abc', b'cab', b'ar', b'arch', b'h', b'he', b'there is', b'cat cats')


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
        what = basic._occurrence_what(percepts, spans, (0, 1, 3))
        grad, = torch.autograd.grad(what.sum(), book.primitive_properties.members)
        return what.detach(), grad
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


def test_a_word_admitted_twice_is_one_form_and_two_spellings_two():
    """A stored form is its row: admitting a spelling again changes nothing;
    two spellings of one set of characters are two rows, whatever their
    codes."""
    identity, *_ = coded((b'calaba', b'cabala'))
    rows = dict(identity.word_rows)
    size = len(identity.store)
    assert identity.admit(b'calaba') == identity.parts(b'calaba')
    assert identity.word_rows == rows and len(identity.store) == size
    assert rows[b'calaba'] != rows[b'cabala']
    for word in rows:
        assert identity.spelling(identity.parts(word)) == word
