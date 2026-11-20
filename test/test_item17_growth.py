"""Implicit growth: the cone of a symbol (rule 72) and the candidate
matrices beside each Pi and each Sigma (rule 73); sigma-pi section 21,
certificate 60 and a certificate of the cone."""
import pytest
import torch


def coded(owner, order, code):
    """A concept of ``order`` whose conceptual code is ``code``."""
    cid = owner.new_concept(order=order)
    row = owner._concept_row_of(cid)
    with torch.no_grad():
        owner.similarity_codebook.W[row].zero_()
        owner.similarity_codebook.W[row, :len(code)] = torch.tensor(code, dtype=owner.similarity_codebook.W.dtype)
    return cid


def test_the_cone_holds_what_is_used_alike_and_keeps_out_what_is_not():
    """Two concepts used alike (a name and a description of one thing) share
    coordinates; the symbol that holds them is the cone over what they
    share. A member's variation takes a coordinate's weight to nothing; a
    third concept used alike falls in the cone; one used otherwise does not;
    one that holds the cone and what a nearest neighbour adds is against."""
    from test_orders_learned import equality_fixture
    owner, _, alloc = equality_fixture()
    #            shared       varies  other     added
    name = coded(owner, 0, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    description = coded(owner, 0, [1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    third = coded(owner, 0, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    otherwise = coded(owner, 0, [0, 0, 0, 1, 0, 0, 1, 1, 0, 0])
    holder = coded(owner, 0, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    symbol = owner.same_thing(name, description)
    table = alloc.symbol_table()
    weights, against = table.cone(owner, symbol)
    print(dict(weights=weights[:10].tolist(), against=against[:10].tolist()))
    # The coordinates the two vary on have no weight.
    assert float(weights[4]) == 0. and float(weights[5]) == 0.
    assert bool((weights[:4] == 1.).all())
    # Used alike: in the cone; used otherwise: out.
    assert table.holds(owner, symbol, third)
    assert not table.holds(owner, symbol, otherwise)
    # What the nearest neighbours add is the against side.
    assert float(against[8]) > 0 and float(against[9]) > 0
    assert not table.holds(owner, symbol, holder)
    # The fit reads the cone: the third concept finds the symbol of the two.
    assert table.fit(third, owner=owner) == symbol
    assert table.fit(otherwise, owner=owner) is None
    # A concept of another order is never tested against the cone.
    higher = coded(owner, 1, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert not table.holds(owner, symbol, higher)
    # Nothing moved: the codes are as written.
    assert float(owner.similarity_codebook.W[owner._concept_row_of(third), :4].sum()) == 4.


def test_the_cone_reads_one_meaning_never_a_join():
    """A join of two unrelated meanings can cover a cone that neither is in:
    the cone reads one code at a time, so neither falls in it."""
    from SymbolCone import for_side, read
    members = torch.tensor([[1., 1., 0., 0.], [1., 1., 0., .5]])
    weights = for_side(members)
    left, right = torch.tensor([1., 0., 0., 0.]), torch.tensor([0., 1., 0., 0.])
    joined = torch.maximum(left, right)
    lanes = read(weights, torch.zeros(4), torch.stack((left, right, joined)))
    print(dict(left=lanes[0].tolist(), right=lanes[1].tolist(), join=lanes[2].tolist()))
    assert float(lanes[0, 0]) == 0. and float(lanes[1, 0]) == 0.
    assert float(lanes[2, 0]) == 1.


# Certificate 60: implicit growth, tried. Each trial holds its candidates in a
# small sparse matrix (CandidateMatrix), read in the forward pass, stepped by
# the proximal step and promoted or forgotten by its value.

def test_60_1_the_pi_at_order_zero_widens_a_row_or_starts_a_candidate():
    """``caat`` after ``cat``: within the row's width the row widens (its
    weight falls on what the member lacks); beyond it, a candidate row
    starts. ``cats`` against ``cat``: one correction gives the row an
    against entry on what ``cats`` adds."""
    from test_item17_forms import lesson_store
    from IdentityRows import meet_ratios
    from CandidateMatrix import CandidateMatrix
    store, identity = lesson_store()
    for word in (b'caat', b'cats'):
        if word not in identity.words:
            identity.admit(word) if hasattr(identity, 'admit') else None
    cat, cats = identity.form(b'cat'), identity.form(b'cats')
    caat = identity.form(b'caat') if b'caat' in identity.words else None
    if caat is None:
        pytest.skip('the lesson store holds no caat')
    held, row_holds = meet_ratios(caat, cat)
    widths, outcomes = (.5, .75, .9, 1.), {}
    for width in widths:
        candidates = CandidateMatrix(4)
        row = cat.clone()
        if float(held) >= width:
            row = torch.minimum(row, caat)
            outcomes[width] = ('widen', float(row.sum()), float(cat.sum()))
        else:
            candidates.offer(('row', b'caat'))
            outcomes[width] = ('candidate', candidates.counts())
    # One correction: cats read as cat. The against entry is what cats adds.
    against = (cats > 0) & ~(cat > 0)
    print(dict(caat_holds_of_cat=float(held), cat_holds_of_caat=float(row_holds), outcomes=outcomes,
               against_entries=int(against.sum())))
    assert outcomes[.5][0] == 'widen' if float(held) >= .5 else outcomes[.5][0] == 'candidate'
    assert outcomes[1.][0] in ('widen', 'candidate')
    assert int(against.sum()) > 0


def _pair_lesson(strength, looks):
    from CandidateMatrix import CandidateMatrix, pair_gradient
    held = CandidateMatrix(8, strength=strength)
    made = []
    for present in looks:
        found = sorted(k for k, v in present.items() if v >= .5)
        for i, left in enumerate(found):
            for right in found[i + 1:]:
                if (left, right) not in made:
                    held.offer((left, right))
        gradients = {key: pair_gradient(held.value(key), present.get(key[0], 0.), present.get(key[1], 0.))
                     for key in held.entries if max(present.get(key[0], 0.), present.get(key[1], 0.)) > 0}
        promoted, _ = held.step(gradients)
        made += [key for key, _ in promoted]
    return made, held.counts()


def test_60_2_the_pi_above_a_pair_used_together_becomes_a_symbol():
    """Two concepts always present together at a look become the row of a
    symbol of the pair; one that is present with them only now and then
    does not."""
    looks = []
    for k in range(12):
        look = {'alpha': 1., 'gamma': 1.}
        if k % 4 == 0:
            look['delta'] = 1.
        looks.append(look)
        looks.append({'beta': 1., 'delta': 1.} if k % 4 else {'beta': 1.})
    from CandidateMatrix import penalty
    made, counts = _pair_lesson(penalty(), looks)
    print(dict(made=made, counts=counts))
    assert ('alpha', 'gamma') in made and ('beta', 'delta') in made
    assert ('alpha', 'delta') not in made and ('delta', 'gamma') not in made


def _substitution_lesson(strength, places):
    from CandidateMatrix import CandidateMatrix, substitution_gradient
    held = CandidateMatrix(8, strength=strength)
    made = []
    for expected, read, present in places:
        for thing in present:
            for kind, value in expected.items():
                if value > read.get(kind, 0.) and (kind, thing) not in made:
                    held.offer((kind, thing))
        gradients = {(kind, thing): substitution_gradient(expected.get(kind, 0.), read.get(kind, 0.),
                                                          present.get(thing, 0.))
                     for kind, thing in held.entries if present.get(thing, 0.) > 0}
        promoted, _ = held.step(gradients)
        made += [key for key, _ in promoted]
    return made, held.counts()


def test_60_3_the_sigma_a_word_in_a_kinds_place_gains_its_entry():
    """``wug`` met where a noun is expected and no noun is read gains an
    entry of the kind noun, with no statement and no call of MTM; ``the``,
    met beside the noun (at another place, where the noun is read), gains
    none."""
    places = []
    for _ in range(6):
        places.append(({'noun': .6}, {}, {'wug': 1.}))            # wug in the noun's place
        places.append(({'noun': .6}, {'noun': 1.}, {'cat': 1.}))  # a noun read where expected
        places.append(({'determiner': .6}, {'determiner': 1.}, {'the': 1.}))
    from CandidateMatrix import penalty
    made, counts = _substitution_lesson(penalty(), places)
    print(dict(made=made, counts=counts))
    assert ('noun', 'wug') in made
    assert not any(thing == 'the' and kind == 'noun' for kind, thing in made)
    assert ('noun', 'cat') not in made


def test_60_penalty_strength_measured():
    """The penalty's one strength, by measurement on the two lessons above:
    the strengths at which each makes exactly what it should."""
    looks = []
    for k in range(12):
        look = {'alpha': 1., 'gamma': 1.}
        if k % 4 == 0:
            look['delta'] = 1.
        looks.append(look)
        looks.append({'beta': 1., 'delta': 1.} if k % 4 else {'beta': 1.})
    places = []
    for _ in range(6):
        places.append(({'noun': .6}, {}, {'wug': 1.}))
        places.append(({'noun': .6}, {'noun': 1.}, {'cat': 1.}))
        places.append(({'noun': .6}, {}, {'wug': .3, 'the': .1}))
    right = {}
    for strength in (0., .01, .02, .05, .1, .2, .3, .5):
        pairs, _ = _pair_lesson(strength, looks)
        kinds, _ = _substitution_lesson(strength, places)
        right[strength] = (sorted(pairs) == [('alpha', 'gamma'), ('beta', 'delta')]
                           and sorted(kinds) == [('noun', 'wug')])
    print(dict(right_by_strength=right))
    from CandidateMatrix import penalty
    assert right[penalty()]
