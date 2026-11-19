"""The percept's code: two halves of 64, each on two lanes (sigma-pi §21,
rules 50 to 52; certificate 46).

A form's code is its parts half and its wholes half on a for lane and an
against lane, every value within 0 to 1. The parts half's for lane is the
form's parts; its against lane is what the form's nearest holders add. The
wholes half's for lane is the AND of the codes of the wholes it lies in: the
class of all characters (all ones), the classes of characters it lies in,
and the form itself, its tightest whole. Its against lane is the codes of
the wholes directly inside those, which it does not lie in.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn

from Layers import (RadixLayer, _CHAR_CLASS_RANGES, LETTER, DIGIT, WHITESPACE, PUNCT,
                    CAPITAL, CONTROL, HIGH_BYTE, PAD)
from PerceptProperties import PrimitiveProperties, TOP
from WordIdentity import WordIdentity

CLASSES = (('letter', LETTER), ('digit', DIGIT), ('whitespace', WHITESPACE),
           ('punctuation', PUNCT), ('capital', CAPITAL), ('control', CONTROL),
           ('high byte', HIGH_BYTE), ('pad', PAD), ('word', LETTER))
WORDS = (b'archer', b'archery', b'is', b'this', b'This', b'What', b'that', b'Who',
         b'Felix', b'Ciao', b'there', b'cat', b'cats', b'Cat', b'CAT', b'c4t', b'ice cream')


def table(spare=2):
    """The table of wholes: the declared classes of characters, with
    ``spare`` free rows, in a codebook of 64 + 64."""
    from Spaces import Codebook
    book = Codebook()
    book.W = nn.Parameter(torch.zeros(len(CLASSES) + spare, 128))
    book.primitive_properties = PrimitiveProperties(len(CLASSES) + spare, property_width=64)
    with torch.no_grad():
        for row, (_name, cls) in enumerate(CLASSES):
            for lo, hi in _CHAR_CLASS_RANGES[cls]:
                book.primitive_properties.members[row, lo:hi + 1] = 1
    book.enable_property_codes(identity_width=64, property_width=64)
    return book


def coded(words=WORDS):
    """Each form admitted, reserved as a basic percept with the classes it
    lies in, and coded; returns the identity, the percepts and their rows."""
    store = RadixLayer(128, initial_cap=4 * len(words) + 256)
    identity = WordIdentity(store, property_width=64, basic_capacity=len(words) + 16)
    store.identity = identity
    for word in words:
        identity.admit(word)
    book = table()
    basic = identity.basic
    basic.bind(book)
    members = book.primitive_properties.members.detach() > 0
    rows = {}
    for word in words:
        wholes = tuple(r for r in range(len(CLASSES)) if all(bool(members[r, b]) for b in word))
        rows[word] = basic.reserve(identity.parts(word), wholes, word=identity.word_rows[word])
        basic.promote(rows[word])
    return identity, basic, book, rows


def test_46_the_code_is_two_halves_of_64_on_two_lanes():
    identity, basic, book, rows = coded()
    properties = book.primitive_properties
    # The class of all characters: the OR of the whole catalog, all ones.
    assert properties.top().eq(1).all()
    assert basic.width == 128 and identity.width == 64 and identity.property_width == 64
    lanes = {word: basic.lanes(row) for word, row in rows.items()}
    for word, row in rows.items():
        two = lanes[word]
        assert two.shape == (2, 128)
        assert bool(((two >= 0) & (two <= 1)).all())
        # No coordinate is one on both lanes.
        assert not bool((two[0].eq(1) & two[1].eq(1)).any())
        # Stored and saved with the code.
        torch.testing.assert_close(basic.codes[row], two[0])
        torch.testing.assert_close(basic.against_codes[row], two[1])
        # The parts half's for lane is at or above each of its parts'.
        for part in identity.parts(word):
            assert bool((basic.part_codes([part])[0] <= two[0, :64]).all())
        # A code is at or below each of its wholes', and below the top.
        for whole in basic.occurrences[row][1]:
            assert bool((two[0, 64:] <= basic.whole_codes([whole])[0]).all())
        # One with no holder has nothing on its parts half's against lane.
        if not identity.holders[word]:
            assert not two[1, :64].any()
    # The for lane of a form lies at or below that of a form holding it.
    for word in rows:
        for holder in identity.holders[word]:
            assert bool((lanes[word][0] <= lanes[holder][0]).all()), (word, holder)
    # archery holds archer: its y is evidence against archer.
    y = identity.atom_code(('part', 'y'))
    assert bool((lanes[b'archer'][1, :64] >= y * (1 - lanes[b'archer'][0, :64])).all())
    assert lanes[b'archer'][1, :64].any() and not lanes[b'archery'][1, :64].any()
    # The wholes half's against lane: a lower-case word lies in letter and
    # word, not in capital (inside letter) nor in digit and the rest
    # (inside the top).
    inside = properties.directly_inside()
    names = {row: name for row, (name, _cls) in enumerate(CLASSES)}
    assert {names[r] for r in inside[TOP]} == {'letter', 'word', 'digit', 'whitespace',
                                                'punctuation', 'control', 'high byte', 'pad'}
    assert {names[r] for r in inside[0]} == {'capital'}
    witnessed = (basic.whole_codes([4, 1, 2, 3, 5, 6, 7]).amax(0) * (1 - lanes[b'cat'][0, 64:]))
    torch.testing.assert_close(lanes[b'cat'][1, 64:], witnessed)
    # Cat and cat are told apart by their parts; CAT lies in capital too.
    assert not torch.equal(lanes[b'Cat'][0], lanes[b'cat'][0])
    assert 4 in basic.occurrences[rows[b'CAT']][1]
    print({word.decode(): (int(lanes[word][0, :64].sum()), int(lanes[word][1, :64].sum()),
                           int(lanes[word][0, 64:].sum()), int(lanes[word][1, 64:].sum()))
           for word in rows})


def test_46_a_percept_under_the_class_of_all_characters_alone():
    """A percept whose parts are no form of the catalog and that lies in no
    class has the top for its wholes half: all ones on the for lane."""
    from Spaces import Codebook
    from BasicPercepts import BasicPercepts
    native = Codebook()
    native.W = nn.Parameter(torch.eye(64)[:3])
    basic = BasicPercepts(NS(_basis=native), 64, 64, 4)
    basic.bind(table())
    row = basic.reserve((0, 2), ())
    two = basic.lanes(row)
    assert two[0, 64:].eq(1).all() and not two[1, 64:].any()
    assert not two[1, :64].any()


def test_46_a_whole_given_by_other_wholes_lies_inside_each():
    """A row of the table given by other rows (Alec, §19.80: "capital
    letter" from "letter" and "capital") has the AND of their codes, at or
    below each; a row given by catalog parts has the OR of theirs. Here the
    lower-case vowels are given by the vowels and the lower-case letters."""
    book = table(spare=3)
    properties = book.primitive_properties
    vowels, lower, given = len(CLASSES), len(CLASSES) + 1, len(CLASSES) + 2
    with torch.no_grad():
        properties.members[vowels, list(b'aeiouAEIOU')] = 1
        properties.members[lower, 97:123] = 1
    properties.define_by_rows(given, (vowels, lower))
    book.begin_forward()
    codes = properties.codes()
    torch.testing.assert_close(codes[given], torch.minimum(codes[vowels], codes[lower]))
    for row in (vowels, lower):
        assert bool((codes[given] <= codes[row]).all()) and not torch.equal(codes[given], codes[row])
    # A row given by catalog parts: the OR of its member bytes' codes.
    torch.testing.assert_close(codes[vowels], properties.byte_bits[list(b'aeiouAEIOU')].amax(0))
    torch.testing.assert_close(book.W[given, 64:], codes[given])
    # It lies directly inside both in the table, and a byte belongs to it as
    # far as it belongs to both.
    inside = properties.directly_inside()
    assert given in inside[vowels] and given in inside[lower]
    coefficients = properties.coefficients()
    assert coefficients[given, ord('a')] == 1 and coefficients[given, ord('A')] == 0
    assert coefficients[given, ord('b')] == 0
    restored = PrimitiveProperties(len(properties.members), property_width=64)
    restored.load_state_dict(properties.state_dict())
    assert restored.given_by == {given: (vowels, lower)}
    torch.testing.assert_close(restored.codes()[given], codes[given])


def test_46_printed_on_the_dictionary():
    """The first 20,000 words of /usr/share/dict/words as the trial takes
    them: the words that share a code on the for lanes and on both, and the
    codes at or below another's by accident on the for lane (a sample of
    every 40th against all)."""
    import code_trial as ct
    try:
        words = [w.encode() for w in ct.dictionary()]
    except OSError:
        pytest.skip('no /usr/share/dict/words on this host')
    identity, basic, _book, rows = coded(words)
    F = torch.stack([basic.codes[rows[w]] for w in words]).cpu().numpy() > .5
    A = torch.stack([basic.against_codes[rows[w]] for w in words]).cpu().numpy() > .5
    def sharing(M):
        keys = {}
        for i, r in enumerate(M):
            keys.setdefault(np.packbits(r).tobytes(), []).append(i)
        return sum(len(v) for v in keys.values() if len(v) > 1)
    parts = [set(identity.words[w]) for w in words]
    true = accident = 0
    for i in range(0, len(words), 40):
        le = (~F[i] | F).all(1)
        le[i] = False
        for j in np.flatnonzero(le):
            true, accident = (true + 1, accident) if parts[i] <= parts[j] else (true, accident + 1)
    print(dict(words=len(words), sharing_for=sharing(F), sharing_both=sharing(np.concatenate((F, A), 1)),
               sharing_parts_half=sharing(F[:, :64]), inside_true=true, inside_by_accident=accident))
    assert not (F & A).any()


def test_a_checkpoint_of_the_old_layout_is_refused_with_the_reason():
    """A basic percept table saved at 96 + 256 on one lane does not load
    into 64 + 64 on two; nothing converts it."""
    from Spaces import Codebook
    from BasicPercepts import BasicPercepts
    native = Codebook()
    native.W = nn.Parameter(torch.zeros(3, 352))
    old = BasicPercepts(NS(_basis=native), 96, 256, 4).state_dict()
    old.pop('against_codes')
    new = BasicPercepts(NS(_basis=native), 64, 64, 4)
    with pytest.raises(ValueError, match='64 parts and 64 wholes wide on each of two lanes'):
        new.load_state_dict(old)
