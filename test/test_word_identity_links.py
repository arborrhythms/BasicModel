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


def test_ordinary_spelling_observations_preserve_the_order_and_counts_of_learning():
    serial, *_ = coded(())
    batched, *_ = coded(())
    forms = (b'cabala', b'aba', b'cabala', b'!', b'calaba', b'aba', b' ', b'caf\xc3\xa9', b'\xff')
    for raw in forms:
        serial.observe_nested(raw)
    batched.observe_forms(iter(forms))
    for name in ('words', 'word_rows', 'word_witnesses', 'nested', 'holders', 'inside'):
        assert getattr(serial, name) == getattr(batched, name), name
    for raw in forms:
        assert serial.form(raw).equal(batched.form(raw))
