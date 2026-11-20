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
