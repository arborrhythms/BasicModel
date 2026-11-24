"""The percept's code: two halves of 64, each on two lanes (sigma-pi §21,
rules 50 to 55 and 61; certificates 46 to 49 and 53).

A form's code is its parts half and its wholes half on a for lane and an
against lane, every value within 0 to 1. The parts half's for lane is the
form's parts; its against lane is what the form's nearest holders add. The
wholes half's for lane is the AND of the codes of the wholes it lies in: the
class of all characters (all ones), the classes of characters it lies in,
and the form itself, its tightest whole. Its against lane is what that
upper bound cuts away, one less its for lane. The first Pi reads a row's
against entries by union; a concept reads the rows, never the code.
"""
import functools
import time
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


def coded(words=WORDS, measure=None):
    """Each form admitted, reserved as a basic percept with the classes it
    lies in, and coded; returns the identity, the percepts and their rows.
    The table of wholes is bound before the forms are admitted. With ``measure``, each admission's seconds, the earlier codes it changed
    and the earlier forms it gave a coordinate are recorded there."""
    store = RadixLayer(128, initial_cap=4 * len(words) + 256)
    identity = WordIdentity(store, property_width=64, basic_capacity=len(words) + 16)
    store.identity = identity
    book = table()
    basic = identity.basic
    basic.bind(book)
    for word in words:
        start = time.perf_counter()
        identity.admit(word)
        if measure is not None:
            measure.setdefault('seconds', []).append(time.perf_counter() - start)
            measure.setdefault('changed', []).append(len(identity.last_changed))
            measure.setdefault('touched', []).append(len(identity.last_touched))
    members = book.primitive_properties.members.detach() > 0
    rows = {}
    for word in words:
        wholes = tuple(r for r in range(len(CLASSES)) if all(bool(members[r, b]) for b in word))
        rows[word] = basic.reserve(identity.parts(word), wholes, word=identity.word_rows[word])
        basic.promote(rows[word])
    return identity, basic, book, rows


@functools.lru_cache(maxsize=None)
def dictionary_coded():
    """The first 20,000 words of /usr/share/dict/words as the trial takes
    them (19,821), admitted in order, coded and measured; shared by the
    tests of this file that read them."""
    import code_trial as ct
    try:
        words = [w.encode() for w in ct.dictionary()]
    except OSError:
        pytest.skip('no /usr/share/dict/words on this host')
    measure = {}
    return (*coded(words, measure), measure)


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
        # Read at the lookup as the lanes, from the parts and the wholes.
        torch.testing.assert_close(basic.bank_codes((row,))[0], two)
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
    # The wholes half's against lane is what the upper bound cuts away, one
    # less its for lane (rule 52).
    for word in rows:
        torch.testing.assert_close(lanes[word][1, 64:], 1 - lanes[word][0, 64:])
    # The table: a lower-case word lies in letter and word, not in capital
    # (inside letter) nor in digit and the rest (inside the top).
    inside = properties.directly_inside()
    names = {row: name for row, (name, _cls) in enumerate(CLASSES)}
    assert {names[r] for r in inside[TOP]} == {'letter', 'word', 'digit', 'whitespace',
                                                'punctuation', 'control', 'high byte', 'pad'}
    assert {names[r] for r in inside[0]} == {'capital'}
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
    identity, basic, _book, rows, _measure = dictionary_coded()
    words = list(rows)
    F = torch.stack([basic.code(rows[w]) for w in words]).detach().cpu().numpy() > .5
    A = torch.stack([basic.against(rows[w]) for w in words]).detach().cpu().numpy() > .5
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
               sharing_parts_half=sharing(F[:, :64]), inside_true=true, inside_by_accident=accident,
               added_parts=len(identity.added)))
    assert not (F & A).any()
    # Rule 56: two forms left with one code get the next part; none is left.
    assert sharing(F) == 0 and sharing(np.concatenate((F, A), 1)) == 0


def test_a_checkpoint_of_the_old_layout_is_refused_with_the_reason():
    """A basic percept table saved at 96 + 256 on one lane does not load
    into 64 + 64 on two; nothing converts it."""
    from Spaces import Codebook
    from BasicPercepts import BasicPercepts
    native = Codebook()
    native.W = nn.Parameter(torch.zeros(3, 352))
    new = BasicPercepts(NS(_basis=native), 64, 64, 4)
    old = dict(new.state_dict(), codes=torch.zeros(4, 352))
    with pytest.raises(ValueError, match='64 parts and 64 wholes wide on each of two lanes'):
        new.load_state_dict(old)


# Certificate 47: for and against at the first Pi (rule 53).

def _read_form(basic, book, word, rows):
    """Each of ``rows``' two lanes on a reading of ``word``, read through the
    first Pi from the word's characters (byte rows of the store)."""
    store = basic.native
    ids = torch.tensor([[store.hash_map[bytes([b])] if bytes([b]) in store.hash_map
                         else store.insert(bytes([b])) for b in word]])
    spans = torch.tensor([[[i, i + 1] for i in range(len(word))]])
    extent = torch.tensor([[[0, len(word)]]])
    field = basic.read((ids, spans, book.primitive_properties, torch.tensor([list(word)]), extent),
                       extent, tuple(rows))
    return torch.stack([field.values[field.keys.index(('bp', row)), 0, 0] for row in rows])


def test_47_for_and_against_at_the_first_pi():
    """The row of ``archer`` reads for 1 and against 0 on ``archer``, for 1
    and against 1 on ``archery``; the row of ``is`` reads against on
    ``this``; a row with no holder never reads against from a part. The
    against entries are the parts the nearest holders add, joined by the
    union, read by the set of parts found."""
    words = (b'archer', b'archery', b'is', b'this', b'his', b'cat', b'dog')
    identity, basic, book, rows = coded(words)
    pi = basic.pi
    atoms = {row: atom for atom, row in identity.atoms.items()}
    entries = {w.decode(): sorted(atoms[key[1][0]][1] for key in pi.row_against(rows[w])) for w in words}
    print(dict(against_entries=entries))
    assert entries['archer'] == ['y'] and entries['is'] == ['h']  # his is nearer than this
    assert not entries['archery'] and not entries['cat']
    read = lambda form, row: _read_form(basic, book, form, [rows[row]])[0].tolist()
    assert read(b'archer', b'archer') == [1., 0.]
    assert read(b'archery', b'archer') == [1., 1.]
    assert read(b'this', b'is') == [1., 1.] and read(b'his', b'is') == [1., 1.]
    assert read(b'is', b'is') == [1., 0.]
    for word in (b'archery', b'cat', b'dog', b'this'):
        if not identity.holders[word]:
            for other in words:
                assert _read_form(basic, book, other, [rows[word]])[0, 1] == 0.
    # Any one entry is enough: a row with two (is, held by his and by this
    # at one remove) reads against where either is found.
    late = coded(words + (b'isa',))
    entries = sorted(late[0].atoms and {a for a in late[0].against_atoms(b'is')})
    assert entries == [('part', 'a'), ('part', 'h')]
    for form in (b'his', b'isa'):
        assert _read_form(late[1], late[2], form, [late[3][b'is']])[0, 1] == 1.


def test_47_the_first_pi_is_written_again_when_a_nearer_holder_comes():
    identity, basic, book, rows = coded((b'is', b'this'))
    key = lambda ch: ('ps', (identity.atoms[('part', ch)],))
    assert sorted(basic.pi.row_against(rows[b'is'])) == sorted([key('h'), key('t')])
    identity.admit(b'his')
    basic._refresh_pi()
    assert basic.pi.row_against(rows[b'is']) == [key('h')]
    assert basic.pi.row_members(rows[b'is']) == basic.member_keys(rows[b'is'])


# Certificate 48, on rows (rule 62): a concept reads its own row, never a
# code. Its first test, a concept reading by difference over codes, is
# retired: certificate 54 (c) in test_item17_symbols.py states it on rows
# (archer reads its own row at tanh(1), archery and reach at zero).

def test_48_on_the_dictionary():
    """Printed: the first Pi's entries a row, on the dictionary; and, for a
    sample of every 40th word, the forms whose code holds its code, by kind:
    as a run, by its parts without the run, or by accident. No form's code
    holds another's parts without the run (rule 61)."""
    identity, basic, _book, rows, _measure = dictionary_coded()
    words = list(rows)
    index = [rows[w] for w in words]
    F = torch.stack([basic.code(row) for row in index]).detach().cpu().numpy() > .5
    parts = [set(identity.words[w]) for w in words]
    kinds = {'run': 0, 'parts': 0, 'accident': 0}
    for i in range(0, len(words), 40):
        le = (~F[i, :64] | F[:, :64]).all(1)
        le[i] = False
        for j in np.flatnonzero(le):
            kinds['run' if words[i] in words[j] else 'parts' if parts[i] <= parts[j] else 'accident'] += 1
    entries = [len(basic.pi.row_against(rows[w])) + len(basic.pi.row_members(rows[w])) for w in words]
    against = [len(basic.pi.row_against(rows[w])) for w in words]
    print(dict(holders_by_kind=kinds, against_entries=(min(against), round(float(np.mean(against)), 2), max(against)),
               entries_a_row=(min(entries), round(float(np.mean(entries)), 2), max(entries)),
               rows_over_16=sum(e > 16 for e in entries), pi_entries=basic.pi.nnz,
               allocated=basic.capacity * basic.ENTRIES_PER_ROW))
    assert kinds['run'] and not kinds['parts']


def test_48_the_lessons_concepts_told_apart():
    """On the lesson, each nominated form read through the first Pi with
    every nominated row, then the Not and the Sigma of order zero: how many
    concepts read their own form above every other concept (printed), and
    every concept reads its own form at tanh(1)."""
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(16)
    try:
        owner = model._concept_owner()
        basic = owner.basic_percepts
        book = model.wholeSpace.subspace.what
        nominated = [r for r in range(len(basic.index)) if bool(basic.promoted[r])
                     and owner.concept_of_basic_percept(r) is not None and r not in basic.conjunctions
                     and r not in basic.place_rows]
        word_of = {row: word for word, row in basic.words.items()}
        forms = [owner.interpret.lexical_form(word_of[r]) for r in nominated]
        concepts = [owner.concept_of_basic_percept(r) for r in nominated]
        told = both = 0
        own = []
        for index, form in enumerate(forms):
            raw = form if isinstance(form, bytes) else str(form).encode()
            lanes = _read_form(basic, book, raw, nominated)
            both += int(((lanes[:, 0] > 0) & (lanes[:, 1] > 0)).sum())
            y = owner.order_zero_field(lanes[:, None, None], nominated, concepts)[:, 0, 0].detach()
            y = y[:, 0] - y[:, 1]
            others = torch.cat((y[:index], y[index + 1:]))
            told += int(len(others) == 0 or bool(y[index] > others.max()))
            own.append(float(y[index]))
        print(dict(nominated=[f.decode() if isinstance(f, bytes) else f for f in forms],
                   told_apart=(told, len(forms)), rows_read_both=both, own_least=min(own)))
        assert told == len(forms)
    finally:
        model.End()
        patch.undo()


# Certificate 49: the scalars (rule 55).

def test_49_collections_of_values_are_told_apart():
    """Collections at 1, 1 and 2 (alpha, beta, gamma) are told apart at the
    inputs 1, 2 and 4, each by its boundaries, 0 or 1; a value that arrives
    between two known ones lies above the one and below the other, and
    inside a collection over the two. A value's code is the bins at or
    below it, in order, in both halves; the way back reads the table."""
    from BasicPercepts import BasicPercepts
    from NumericPercepts import feeling, number
    basic = BasicPercepts(None, 64, 64, 32)
    basic.bind(None)
    felt = lambda v: feeling(torch.tensor(float(v))).to(basic.value_codes)
    percept = {v: basic.perceive_value(felt(v)) for v in (1., 2., 4., 1.5)}
    collections = dict(alpha=[percept[1.]], beta=[percept[1.]], gamma=[percept[2.]],
                       between=[percept[1.], percept[2.]])
    def point(v):
        return torch.cat((basic.value_code(felt(v), 64), basic.value_code(felt(v), 64)))[None, None]
    def read(v):
        return {name: basic.read_collection(basic.read_points(point(v), rows),
                                            torch.ones(len(rows)))[0, 0].tolist()
                for name, rows in collections.items()}
    readings = {v: read(v) for v in (1., 2., 4., 1.5)}
    print('collections read', readings)
    held = lambda v: {name for name, (f, a) in readings[v].items() if f > a}
    assert held(1.) == {'alpha', 'beta', 'between'}
    assert held(2.) == {'gamma', 'between'}
    assert held(4.) == set()
    assert held(1.5) == {'between'}
    # Each value's code: a cut of the bins, the same in both halves; one
    # value's code lies inside a greater one's.
    codes = {v: basic.code(r) for v, r in percept.items()}
    for v, code in codes.items():
        assert basic.bin_of(code) == int(BasicPercepts.value_bin(felt(v), 64))
        torch.testing.assert_close(code[:64], code[64:])
    assert bool((codes[1.] <= codes[1.5]).all() and (codes[1.5] <= codes[2.]).all())
    # The way back: the table of values holds each exactly.
    for v, r in percept.items():
        assert float(number(basic.felt([r])[0])) == pytest.approx(v, rel=1e-5)
    # What a concept reads of the values on rows (rule 62): each collection
    # has an entry on each member's row, and a value is read by its own row
    # alone, so a collection reads its members and nothing else; alpha's
    # reading of 1.5 lies between its readings of 1 and of 2.
    from Layers import NotLayer3, SparseSigmaLDU
    not_, sigma = NotLayer3(basic.capacity), SparseSigmaLDU(basic.capacity, len(collections), 16)
    names = list(collections)
    for c, name in enumerate(names):
        for r in collections[name]:
            sigma.add(c, r)
    lanes = torch.tensor([[[1., 0.]]])
    read = {v: {name: round(float(sigma(not_(lanes), [r], [c])[0, 0].detach()), 3)
                for c, name in enumerate(names)} for v, r in percept.items()}
    print('on rows', read)
    assert read[2.]['alpha'] <= read[1.5]['alpha'] <= read[1.]['alpha']
    assert read[1.]['gamma'] <= read[1.5]['gamma'] <= read[2.]['gamma']
    assert read[1.5] == {name: 0. for name in names}


def test_49_the_numeric_certificate_keeps_its_bar():
    """The numeric XOR's two places read apart, each told by its wires (its
    place's row), and the certificate's bar holds (``numeric_xor.certificate``)."""
    from numeric_xor import certificate
    model, prediction, error = certificate()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1)
    field = model.numeric_percepts.read(values)
    places = [member for members in basic.conjunctions.values() for member in members]
    axes = [owner.concept_id_at_row(owner.concept_of_basic_percept(row)) for row in places]
    features = [field.evidence[field.concept_ids == cid][0].amax(1).tolist() for cid in axes]
    print(dict(fx=features[0], fy=features[1], mse=error))
    assert features[0] != features[1]


# Certificate 53: no code inside another by accident (rule 61).

def test_53_no_accident_on_the_dictionary():
    """(a) After the 19,821 words are admitted, no code lies inside
    another's, both halves, unless the one form is linked as a part of the
    other, over all ordered pairs; no two forms share a code; every linked
    part lies inside its whole; and the packed codes are the codes.
    Printed: the forms given a coordinate, the earlier codes an admission
    changed (and the earlier forms it touched), and the milliseconds of an
    admission near 1,000, 5,000 and 19,000 stored forms. The admission is
    shared with certificates 46 and 48 (about 90 s at 64 and 64 on a quiet
    machine)."""
    from WordIdentity import bytes_of
    identity, basic, _book, rows, measure = dictionary_coded()
    words = list(rows)
    found, outside = identity.accidents()
    keys = {identity.key(identity.code(w)) for w in words}
    seconds, changed, touched = (np.array(measure[k]) for k in ('seconds', 'changed', 'touched'))
    def near(count, window=100):
        part = seconds[max(0, count - window):count] * 1000
        return round(float(part.mean()), 2), round(float(part.max()), 1)
    print(dict(width=(identity.width, identity.property_width), words=len(words),
               forms_given_a_coordinate=sum(bytes_of(atom[1]) in identity.words for atom in identity.given),
               parts_given_a_coordinate=len(identity.given), added_parts=len(identity.added),
               changed_mean=round(float(changed.mean()), 2), changed_greatest=int(changed.max()),
               touched_mean=round(float(touched.mean()), 2), touched_greatest=int(touched.max()),
               admissions_changing_any=int((changed > 0).sum()),
               ms_near={count: near(count) for count in (1000, 5000, 19000)},
               seconds_in_all=round(float(seconds.sum()), 1), accidents=len(found),
               parts_outside=len(outside), failures=len(identity.failures)))
    assert not found and not outside and not identity.failures
    assert len(keys) == len(words)
    assert all(torch.equal(identity.packed_code(w), identity.code(w)) for w in words[::7])


def test_53_a_forms_links_are_its_runs():
    """(b) archer is a part of archery and not of reach, in whatever order
    the three are admitted: the links are the runs, archer's code lies
    inside archery's and not inside reach's, nor reach's inside archer's.
    The links and the coordinates given are saved with the forms."""
    from itertools import permutations
    for words in permutations((b'archer', b'archery', b'reach')):
        identity, _basic, _book, _rows = coded(words)
        assert identity.inside[b'archery'] == {b'archer'} and identity.holders[b'archer'] == {b'archery'}
        assert not identity.inside[b'reach'] and not identity.holders[b'reach']
        code = identity.code
        assert bool((code(b'archer') <= code(b'archery')).all())
        assert not bool((code(b'archer') <= code(b'reach')).all())
        assert not bool((code(b'reach') <= code(b'archer')).all())
        assert identity.accidents() == ([], [])
        restored = WordIdentity(RadixLayer(128, initial_cap=256), property_width=64)
        restored.load_vocab_extras(identity.vocab_extras())
        assert (restored.inside, restored.holders) == (identity.inside, identity.holders)
        assert all(torch.equal(restored.code(w), identity.code(w)) for w in words)


def test_53_a_later_form_moves_an_earlier_forms_row_and_lane_and_not_its_concept():
    """(c) In the lesson's model, after its first four sentences (``is`` is
    nominated: a basic percept row with a concept), a form admitted later
    that changes the code of ``is`` changes the store row of ``is``, its
    basic percept's stored code and against lane, and its row of the first
    Pi; the concept's entries, which read the row and not its code, do not
    move (rule 62). The change is the one rule 56 makes when no run tells
    two forms apart: a part of ``is`` (its own) is given more bits, and
    every form that holds it is settled again."""
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(4)
    try:
        owner = model._concept_owner()
        basic = owner.basic_percepts
        identity = model.perceptualSpace.percept_store.identity
        row = next(r for r, raw in basic.forms.items() if raw == b'is' and bool(basic.promoted[r])
                   and owner.concept_of_basic_percept(r) is not None)
        concept = owner.concept_of_basic_percept(row)
        sigma = owner.install_order_zero()
        store_row = identity.word_rows[b'is']
        fixed = identity.store._basis.fixed_row_codes[store_row].clone()
        code, against = basic.code(row).detach().clone(), basic.against(row).detach().clone()
        members = basic.pi.row_members(row)
        entries = sigma.entries(concept)
        # Two forms of one set of characters (``si``) are told apart by their
        # own parts (rule 43), so no later form collides with ``is``.
        own = ('part', 'is')
        identity._atom(own, identity.bits.get(own, identity.ones) + 1)
        identity._keep(identity._settle(identity._holders([own])))
        assert b'is' in identity.last_changed
        now = identity.store._basis.fixed_row_codes[store_row]
        assert not torch.equal(now, fixed) and torch.equal(now[:identity.width], identity.form(b'is'))
        # The row's code is read again from its form's new code.
        assert not torch.equal(basic.code(row).detach(), code)
        torch.testing.assert_close(basic.bank_codes((row,))[0], basic.lanes(row).detach().clamp(0, 1))
        assert basic.pi.row_members(row) == basic.member_keys(row)
        assert sorted(basic.pi.row_against(row)) == sorted(basic.against_keys(row))
        # The concept reads the row, not its code: its entries do not move.
        assert sigma.entries(concept) == entries and (row, ) == tuple(i for i, _, _ in entries if i == row)
        print(dict(changed=sorted(w.decode() for w in identity.last_changed),
                   parts_gained=[k for k in basic.member_keys(row) if k not in members],
                   entries=entries))
    finally:
        model.End()
        patch.undo()
