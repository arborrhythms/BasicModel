"""The percept's code: two halves of 64, each on two lanes (sigma-pi §21,
rules 50 to 55; certificates 46 to 49).

A form's code is its parts half and its wholes half on a for lane and an
against lane, every value within 0 to 1. The parts half's for lane is the
form's parts; its against lane is what the form's nearest holders add. The
wholes half's for lane is the AND of the codes of the wholes it lies in: the
class of all characters (all ones), the classes of characters it lies in,
and the form itself, its tightest whole. Its against lane is what that
upper bound cuts away, one less its for lane. The first Pi reads a row's
against entries by union; a concept reads for less against, the wholes half
on its complement.
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
    old = BasicPercepts(NS(_basis=native), 96, 256, 4).state_dict()
    old.pop('against_codes')
    new = BasicPercepts(NS(_basis=native), 64, 64, 4)
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


# Certificate 48: scored by difference (rule 54; Alec, §19.82).

def _difference(positive, negative):
    """The column as it was before §19.82: for less against, over the
    squared length of that difference."""
    vector = positive - negative
    return vector / vector.square().sum()


def test_48_a_concept_reads_by_difference():
    """A column starts as what tells for its member over the squared length
    of that, less what tells against it over the squared length of that
    (``P / |P|^2 - N / |N|^2``): all that tells for reads one and all that
    tells against takes the reading to zero, however few its coordinates. So
    `archer` reads `archery`, and `is` reads `this`, at or near zero, while
    `reach`, which holds archer's letters and not the run, has nothing that
    tells against archer. Its own row reads one, and one more from its own
    against lane where it has a holder."""
    from OrderZero import combine
    words = (b'archer', b'archery', b'reach', b'is', b'this', b'cat')
    identity, basic, book, rows = coded(words)
    column = {w: basic.column(rows[w]) for w in words}
    before = {w: _difference(*basic.read_pair(basic.lanes(rows[w]))) for w in words}
    def reads(concept, form, columns=column):
        lanes = _read_form(basic, book, form, [rows[w] for w in words])
        x = combine(lanes[:, None], basic.read_codes([rows[w] for w in words]))[0]
        return float(((x[0] - x[1]) @ columns[concept]))
    pairs = ((b'archer', b'archer'), (b'archer', b'archery'), (b'archer', b'reach'),
             (b'is', b'is'), (b'is', b'this'), (b'cat', b'cat'))
    table = {f'{c.decode()} on {f.decode()}': (round(reads(c, f, before), 3), round(reads(c, f), 3))
             for c, f in pairs}
    print('before the squash (before, built)', table)
    # The own row: what tells for it reads one; where it has a holder its
    # own against lane reads one more.
    for word in (b'archer', b'is', b'cat'):
        holders = bool(basic.lanes(rows[word])[1, :64].any())
        assert reads(word, word) == pytest.approx(1. + holders, abs=1e-4)
    # A holder's added part takes the reading to zero or below.
    assert reads(b'archer', b'archery') <= 1e-4 and reads(b'is', b'this') <= 1e-4
    # reach holds every letter of archer, not the run, and nothing that
    # tells against archer: it reads as the share of archer's code it holds.
    assert 0 < reads(b'archer', b'reach') <= reads(b'archer', b'archer')


def test_48_on_the_dictionary_and_the_lesson():
    """Printed, before §19.82 and as built: what a part's concept reads
    (before the squash) of a form that holds it as a run, of one whose parts
    it holds without the run, and of one that holds it by accident (a
    sample of every 40th word against all, read alone); how often an
    unrelated form (one that holds neither the part's code nor its parts)
    is denied because what tells against the part is covered by accident;
    and the first Pi's entries a row."""
    import code_trial as ct
    try:
        words = [w.encode() for w in ct.dictionary()]
    except OSError:
        pytest.skip('no /usr/share/dict/words on this host')
    identity, basic, _book, rows = coded(words)
    index = [rows[w] for w in words]
    pairs = basic.read_codes(index).double()
    X = (pairs[:, 0] - pairs[:, 1])
    F = basic.codes[index].cpu().numpy() > .5
    parts = [set(identity.words[w]) for w in words]
    sums = {form: {'run': [], 'parts': [], 'accident': []} for form in ('before', 'built')}
    unrelated = {'before': [0, 0], 'built': [0, 0]}
    from OrderZero import OrderZeroSigma
    for i in range(0, len(words), 40):
        P, N = pairs[i]
        columns = dict(before=_difference(P, N), built=OrderZeroSigma.member_column(P, N))
        le = (~F[i, :64] | F[:, :64]).all(1)
        le[i] = False
        kinds = {}
        for j in np.flatnonzero(le):
            kinds[j] = 'run' if words[i] in words[j] else 'parts' if parts[i] <= parts[j] else 'accident'
        others = np.ones(len(words), dtype=bool)
        others[list(kinds)] = False
        others[i] = False
        has_against = bool(N.any())
        for form, u in columns.items():
            y = (X @ u).numpy()
            for j, kind in kinds.items():
                sums[form][kind].append(float(y[j]))
            if has_against:
                # Unrelated forms the for part alone reads above zero, and
                # of them, those that what tells against takes to zero.
                alone = (X @ (P / P.square().sum())).numpy()
                live = others & (alone > 0)
                unrelated[form][0] += int(live.sum())
                unrelated[form][1] += int((live & (y <= 0)).sum())
    table = {form: {k: (len(v), round(float(np.mean(v)), 3) if v else None) for k, v in kinds_.items()}
             for form, kinds_ in sums.items()}
    print('a part read by its concept (count, mean before the squash)', table)
    print('unrelated forms read above zero by the for part / denied',
          {form: (n, d, round(d / max(n, 1), 4)) for form, (n, d) in unrelated.items()})
    entries = [len(basic.pi.row_against(rows[w])) + len(basic.pi.row_members(rows[w])) for w in words]
    against = [len(basic.pi.row_against(rows[w])) for w in words]
    print(dict(against_entries=(min(against), round(float(np.mean(against)), 2), max(against)),
               entries_a_row=(min(entries), round(float(np.mean(entries)), 2), max(entries)),
               rows_over_16=sum(e > 16 for e in entries), pi_entries=basic.pi.nnz,
               allocated=basic.capacity * basic.ENTRIES_PER_ROW))
    # As built, a form that holds the part as a run reads below one that
    # holds its parts without the run: the run's holder adds what tells
    # against it.
    built = sums['built']
    assert built['run'] and np.mean(built['run']) < np.mean(built['parts'])


def test_48_the_lessons_concepts_told_apart():
    from OrderZero import EPS_LOG, OrderZeroSigma
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(16)
    try:
        owner = model._concept_owner()
        basic = owner.basic_percepts
        book = model.wholeSpace.subspace.what
        nominated = [r for r in range(len(basic.index)) if bool(basic.promoted[r])
                     and owner.concept_of_basic_percept(r) is not None and r not in basic.conjunctions]
        word_of = {row: word for word, row in basic.words.items()}
        forms = [owner.interpret.lexical_form(word_of[r]) for r in nominated]
        concepts = [owner.concept_of_basic_percept(r) for r in nominated]
        pairs = basic.read_codes(nominated)
        U = owner.order_zero.weight[:basic.width, concepts].detach()
        before = torch.stack([_difference(*pair) for pair in pairs], 1).to(U)
        built = torch.stack([OrderZeroSigma.member_column(*pair) for pair in pairs], 1).to(U)
        def trial(lanes):
            gaps = torch.log((1 - pairs).clamp_min(EPS_LOG))
            f, a = lanes[:, 0, None], lanes[:, 1, None]
            toward = 1 - torch.exp((f * gaps[:, 0] + a * gaps[:, 1]).sum(0))
            away = 1 - torch.exp((f * gaps[:, 1] + a * gaps[:, 0]).sum(0))
            return toward - away
        from OrderZero import combine
        told = {'net, model': 0, 'trial, model': 0, 'net, before': 0, 'net, built': 0}
        both = 0
        for index, form in enumerate(forms):
            raw = form if isinstance(form, bytes) else str(form).encode()
            lanes = _read_form(basic, book, raw, nominated)
            both += int(((lanes[:, 0] > 0) & (lanes[:, 1] > 0)).sum())
            net = (lambda c: c[0] - c[1])(combine(lanes[:, None], pairs)[0])
            for name, x, columns in (('net, model', net, U), ('trial, model', trial(lanes), U),
                                     ('net, before', net, before), ('net, built', net, built)):
                y = torch.tanh(x @ columns)
                others = torch.cat((y[:index], y[index + 1:]))
                told[name] += int(len(others) == 0 or bool(y[index] > others.max()))
        print(dict(nominated=[f.decode() if isinstance(f, bytes) else f for f in forms],
                   told_apart=told, rows_read_both=both))
        assert told['net, model'] >= told['trial, model']
    finally:
        model.End()
        patch.undo()


# Certificate 49: the scalars (rule 55).

def test_49_collections_of_values_are_told_apart():
    """Collections at 1, 1 and 2 (alpha, beta, gamma) are told apart at the
    inputs 1, 2 and 4, each by its boundaries, 0 or 1; a value that arrives
    between two known ones lies above the one and below the other, and
    inside a collection over the two. A value's code is the bins of its
    place at or below it, in both halves; the way back reads the table."""
    from BasicPercepts import BasicPercepts
    from NumericPercepts import feeling, number
    basic = BasicPercepts(None, 64, 64, 32)
    basic.bind(None)
    place = 7
    felt = lambda v: feeling(torch.tensor(float(v))).to(basic.value_codes)
    percept = {v: basic.perceive_value(felt(v), place=place) for v in (1., 2., 4., 1.5)}
    collections = dict(alpha=[percept[1.]], beta=[percept[1.]], gamma=[percept[2.]],
                       between=[percept[1.], percept[2.]])
    def point(v):
        return torch.cat((basic.value_code(felt(v), 64, place), basic.value_code(felt(v), 64, place)))[None, None]
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
    # Each value's code: a cut of its place's bins, the same in both halves;
    # one value's code lies inside a greater one's.
    codes = {v: basic.code(r) for v, r in percept.items()}
    for v, code in codes.items():
        assert basic.bin_of(code, place) == int(BasicPercepts.value_bin(felt(v), 64))
        torch.testing.assert_close(code[:64], code[64:])
    assert bool((codes[1.] <= codes[1.5]).all() and (codes[1.5] <= codes[2.]).all())
    # The way back: the table of values holds each exactly.
    for v, r in percept.items():
        assert float(number(basic.felt([r])[0])) == pytest.approx(v, rel=1e-5)
    # What a concept reads of the values by its column (rule 8b): alpha's
    # reading of 1.5 lies between its readings of 1 and of 2.
    from OrderZero import OrderZeroSigma
    pairs = {v: basic.read_pair(basic.lanes(r)) for v, r in percept.items()}
    column = {name: sum(basic.column(r) for r in rows) for name, rows in collections.items()}
    sigma = {v: {name: round(float(torch.tanh((pairs[v][0] - pairs[v][1]) @ u)), 3)
                 for name, u in column.items()} for v in pairs}
    print('by the column', sigma)
    assert sigma[2.]['alpha'] <= sigma[1.5]['alpha'] <= sigma[1.]['alpha']
    assert sigma[1.]['gamma'] <= sigma[1.5]['gamma'] <= sigma[2.]['gamma']


def test_49_the_numeric_certificate_keeps_its_bar():
    """The numeric XOR's two axes read apart, each place being a table of its
    own, and the certificate's bar holds (``numeric_xor.certificate``)."""
    from numeric_xor import certificate
    model, prediction, error = certificate()
    owner = model._concept_owner()
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1)
    field = model.numeric_percepts.read(values)
    axes = [owner.concept_id_at_row(row) for row in model.numeric_percepts.rows]
    features = [field.evidence[field.concept_ids == cid][0].amax(1).tolist() for cid in axes]
    print(dict(fx=features[0], fy=features[1], mse=error))
    assert features[0] != features[1]
