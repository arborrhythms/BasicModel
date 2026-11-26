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


# Certificate 60: implicit growth, tried. Each trial holds its candidates in
# MTM's candidate rows (Memory.MTM1, Memory.MTMPairRows), read in the forward
# pass, stepped by the proximal step and promoted or forgotten by its value.

def test_60_1_the_pi_at_order_zero_widens_a_row_or_starts_a_candidate():
    """``caat`` after ``cat``: within the row's width the row widens (its
    weight falls on what the member lacks); beyond it, a candidate row
    starts. ``cats`` against ``cat``: one correction gives the row an
    against entry on what ``cats`` adds."""
    from test_item17_forms import lesson_store
    from IdentityRows import meet_ratios
    from Memory import MediumTermMemory, MTM1
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
        candidates = MTM1(4, strength=MediumTermMemory.penalty()).double()
        row = cat.clone()
        if float(held) >= width:
            row = torch.minimum(row, caat)
            outcomes[width] = ('widen', float(row.sum()), float(cat.sum()))
        else:
            # The candidate row of caat enters at nothing, read by nothing.
            offer = candidates.refs.new_tensor([[1, 2]])
            none = candidates.refs.new_empty(0)
            candidates.observe(offer, none, candidates.activity.new_empty(0), none, candidates.activity.new_empty(0))
            outcomes[width] = ('candidate', candidates.counts())
    # One correction: cats read as cat. The against entry is what cats adds.
    against = (cats > 0) & ~(cat > 0)
    print(dict(caat_holds_of_cat=float(held), cat_holds_of_caat=float(row_holds), outcomes=outcomes,
               against_entries=int(against.sum())))
    assert outcomes[.5][0] == 'widen' if float(held) >= .5 else outcomes[.5][0] == 'candidate'
    assert outcomes[1.][0] in ('widen', 'candidate')
    assert int(against.sum()) > 0


def _pair_lesson(strength, looks):
    from Memory import MTMPairRows
    held = MTMPairRows(8, strength=strength).double()
    names = sorted({name for look in looks for name in look})
    symbols = torch.arange(len(names))
    made = []
    for present in looks:
        found = sorted(k for k, v in present.items() if v >= .5)
        offers = [(names.index(left), names.index(right))
                  for i, left in enumerate(found) for right in found[i + 1:] if (left, right) not in made]
        promoted = held.observe_pairs(held.refs.new_tensor(offers).reshape(-1, 2), symbols,
                                      held.activity.new_tensor([present.get(name, 0.) for name in names]))
        made += [(names[left], names[right]) for left, right in held.publish(promoted)]
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
    from Memory import MediumTermMemory
    made, counts = _pair_lesson(MediumTermMemory.penalty(), looks)
    print(dict(made=made, counts=counts))
    assert ('alpha', 'gamma') in made and ('beta', 'delta') in made
    assert ('alpha', 'delta') not in made and ('delta', 'gamma') not in made
    # The other direction: a kind present at every look is present whenever
    # each word is, but is present far more often without it, so no pair of
    # a word and the kind is made.
    with_kind = [dict(look, kind=1.) for look in looks]
    made, counts = _pair_lesson(MediumTermMemory.penalty(), with_kind)
    print(dict(made_with_kind=made, counts=counts))
    assert ('alpha', 'gamma') in made and ('beta', 'delta') in made
    assert not any('kind' in key for key in made)


def _substitution_lesson(strength, places):
    from Memory import MTM1
    held = MTM1(8, strength=strength).double()
    kinds = sorted({kind for expected, read, _ in places for kind in (*expected, *read)})
    things = sorted({thing for _, _, present in places for thing in present})
    made = []
    for expected, read, present in places:
        offers = [(kinds.index(kind), things.index(thing)) for thing in present
                  for kind, value in expected.items() if value > read.get(kind, 0.) and (kind, thing) not in made]
        promoted = held.observe(held.refs.new_tensor(offers).reshape(-1, 2), torch.arange(len(kinds)),
            held.activity.new_tensor([expected.get(kind, 0.) for kind in kinds]),
            held.refs.new_tensor([things.index(thing) for thing in present]),
            held.activity.new_tensor(list(present.values())),
            observed=held.activity.new_tensor([read.get(kind, 0.) for kind in kinds]))
        made += [(kinds[kind], things[thing]) for kind, thing in held.publish(promoted)]
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
    from Memory import MediumTermMemory
    made, counts = _substitution_lesson(MediumTermMemory.penalty(), places)
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
    from Memory import MediumTermMemory
    assert right[MediumTermMemory.penalty()]


def test_60_4_on_the_lesson_beside_what_mtm_makes(tmp_path, monkeypatch):
    """The rows and entries the candidates make on the small lesson, beside
    the symbols MTM makes: printed, which are the same, which are new. A
    certificate of learning: recorded."""
    from test_item18_expect import lesson, read
    from nomination_fixtures import readings_before_nomination
    from CandidateGrowth import matrices, pair_rows
    model = lesson(tmp_path, monkeypatch)
    try:
        examples = ('alpha gamma', 'beta delta')
        for text in examples:
            readings_before_nomination(model, lambda: read(model, text))
            read(model, text)
        owner = model._concept_owner()
        table = owner._concept_allocator.symbol_table()
        before = {tuple(sorted(table.members(s))) for s in range(len(table))}
        optimizer = model.getOptimizer(lr=.01)
        for presentation in range(40):
            read(model, examples[presentation % 2], training=True, optimizer=optimizer)
        after = {tuple(sorted(table.members(s))) for s in range(len(table))}
        pairs = {tuple(sorted(pair)) for pair in pair_rows(owner)}
        print(dict(candidates={name: held.counts() for name, held in matrices(owner).items()},
                   mtm_symbols=sorted(after), made_by_mtm_in_lesson=sorted(after - before),
                   implicit_pairs=sorted(pairs), same=sorted(pairs & after), new=sorted(pairs - after)))
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_60_2_a_pair_is_offered_from_what_the_sentence_held():
    """A word the omission withheld inside a look's extent is present at the
    look for the Pi's candidates, at one: a pair is offered from what the
    sentence held, not from what was left once a word was taken out."""
    from types import SimpleNamespace
    from CandidateGrowth import withheld_words
    from Omission import Draw
    words = {'alpha': 7, 'gamma': 9}
    owner = SimpleNamespace(interpret=SimpleNamespace(word_for_form=words.get))
    forms = torch.tensor([[[False, False], [True, True]]])
    targets = dict(draw=Draw(forms.any(-1), forms, torch.zeros(1, dtype=torch.long)),
                   word_bytes=((b'alpha', b'gamma'),))
    model = SimpleNamespace(_omission_targets=targets, _concept_owner=lambda: owner)
    assert withheld_words(model, torch.tensor([[True, True]])) == [{9: 1.}]
    assert withheld_words(model, torch.tensor([[True, False]])) == [{}]
    from WordConceptPool import WordConceptPool
    pool = WordConceptPool(16, torch.tensor([23]), threshold=1)
    pool.read(torch.tensor([[5]]))
    owner.mtm = SimpleNamespace(word_pool=pool)
    targets['withheld_form_rows'] = torch.tensor([[-1, 5]])
    assert withheld_words(model, torch.tensor([[False, True]])) == [{23: 1.}]
    targets['withheld_form_rows'] = torch.tensor([[5, 6]])
    assert withheld_words(model, torch.tensor([[True, True]])) == [{}]
    assert pool.counts[6] == 0 and pool.used == 1
    model._omission_targets = None
    assert withheld_words(model, torch.tensor([[True, True]])) == [{}]


def test_omitted_field_restores_only_pi_and_keeps_observation_order():
    from types import SimpleNamespace
    from CandidateGrowth import omitted_field_readings
    ids = torch.tensor([[3, 5, 8], [4, 3, 8], [3, 5, 7]])
    evidence = torch.zeros(3, 3, 1, 2)
    evidence[:, :, 0, 0] = torch.tensor([[.8, .4, .9], [.3, .9, .9], [.2, .7, .8]])
    field = SimpleNamespace(concept_ids=ids, evidence=evidence)
    hidden = torch.tensor([[3, 6, 6], [4, -1, -1], [7, 8, -1]])
    active = torch.tensor([True, True, False])
    assert omitted_field_readings(field, hidden, active, 3) is None
    first, second = omitted_field_readings(field, hidden, active, 4)
    assert (first[0], second[0]) == (0, 1)
    names, values, ranks, count = first[1]
    assert names.tolist() == [3, 4, 5, 6]
    assert values.tolist() == pytest.approx([.2, .3, 0, 0])
    assert ranks.tolist() == [0, 1, 3, 3]
    assert count == 0
    assert first[2][1].tolist() == pytest.approx([1, .3, 0, 1])
    assert first[2][3] == 2
    assert second[1][1].tolist() == pytest.approx([.9, 0, .7, 0])
    assert second[2][1].tolist() == pytest.approx([.9, 1, .7, 0])
    assert second[1][3] == 2 and second[2][3] == 3
