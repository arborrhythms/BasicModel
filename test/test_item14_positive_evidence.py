"""Item 1.4, §10 to §12: evidence at order zero is positive.

The first Pi reads for alone. What tells against a concept is another row
found at the place, by a weight the concept's Sigma learns on a link MTM
writes at zero when the row's holder is nominated. These certificates hold
the measure of it (``PositiveEvidence``), both stopping gates, the link and
its learning, and the reading lesson; the plan's §11 has what was measured
before the removal.
"""
import pytest
import torch

import PositiveEvidence
from PositiveEvidence import Readings, concept_error, record, record_events, transfer


def _readings(positive, negative):
    lanes = torch.stack((torch.tensor(positive, dtype=torch.float64),
                         torch.tensor(negative, dtype=torch.float64)), -1)
    return list(range(lanes.shape[1])), lanes, torch.ones(len(lanes), dtype=torch.long)


def test_an_against_lane_that_is_another_rows_presence_is_a_weight_on_that_row():
    # Rows 0 and 1 are held by row 2, their conjunction: each is told
    # against where the conjunction is found.
    table = _readings([[1, 0, 0], [0, 1, 0], [1, 1, 1]], [[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    rows, H, report = transfer(table)
    assert [c.row for c in report] == [0, 1]
    assert all(c.carriers == (2,) and c.union == 0 and c.fit < 1e-12 and c.uncarried == 0 for c in report)
    torch.testing.assert_close(H, torch.tensor([[0., 0, 1], [0, 0, 1], [0, 0, 0]], dtype=torch.float64), atol=1e-12, rtol=0)


def test_an_against_lane_with_no_row_read_for_is_not_carried():
    # Row 0 is told against at a reading where nothing is read for: the
    # thing that tells against it has no row.
    rows, H, report = transfer(_readings([[1, 0], [0, 0], [0, 1]], [[0, 0], [1, 0], [0, 0]]))
    found, = report
    assert found.row == 0 and found.carriers == () and found.uncarried == 1
    assert found.union == 1 and found.fit > .5


def test_a_reading_is_kept_once_and_an_empty_one_not_at_all():
    readings = Readings()
    lanes = torch.tensor([[[[1., 0.], [1., 0.]]], [[[0., 0.], [0., 0.]]]])           # [P=2, B=1, E=2, 2]
    readings.add(lanes, [7, 9])
    readings.add(lanes[:0], [])
    rows, table, count = readings.table()
    assert rows == [7, 9] and table.shape == (1, 2, 2) and readings.calls == 2
    assert table[0].tolist() == [[1., 0.], [0., 0.]]


def _recorded(build):
    """Run ``build`` with every order-zero reading and every look recorded."""
    import BasicPercepts
    import Spaces
    readings, events = Readings(), PositiveEvidence.Events()
    field, bank_read = Spaces.ConceptualSpace.order_zero_field, BasicPercepts.BasicPercepts.read_bank

    def recorded_field(self, lanes, inputs, rows):
        readings.add(lanes, inputs)
        return field(self, lanes, inputs, rows)

    def recorded_bank(self, bank, *args, **kwargs):
        percepts = bank_read(self, bank, *args, **kwargs)
        with torch.no_grad():
            events.add(self, bank, percepts)
        return percepts
    patch = pytest.MonkeyPatch()
    patch.setattr(Spaces.ConceptualSpace, 'order_zero_field', recorded_field)
    patch.setattr(BasicPercepts.BasicPercepts, 'read_bank', recorded_bank)
    try:
        return build(), readings, events
    finally:
        patch.undo()


def test_on_the_xor_gate_nothing_is_read_against_and_mtm_links_each_place_to_the_conjunction():
    """The stopping gate's model (``MM_xor.xml``), §12.1. Before the removal
    each place's row was told against where the conjunction was found (§11.2).
    Now no row is read against, and MTM has written the link in its place:
    each place's concept has an entry on the conjunction's row, at zero
    until it is learned. The readout still solves XOR from the three rows."""
    import numeric_xor
    (model, _prediction, error), readings, _events = _recorded(numeric_xor.fit)
    owner = model._concept_owner()
    basic, sigma = owner.basic_percepts, owner.order_zero
    conjunction, = basic.conjunctions
    members = sorted(basic.conjunctions[conjunction])
    rows, lanes, _count = readings.table()
    _rows, _H, report = transfer((rows, lanes, _count))
    print(dict(error=error, calls=readings.calls, readings=len(readings.seen)))
    assert error < 1e-6
    assert report == [] and float(lanes[..., 1].abs().max()) == 0.
    assert basic.holder_rows(members[0]) == [conjunction] and basic.held_rows(conjunction) == members
    for member in members:
        entry = sigma.find(owner.concept_of_basic_percept(member), conjunction)
        assert entry is not None and float(sigma.w()[entry]) == 0.
    # The conjunction's own concept is linked to nothing it holds.
    assert all(sigma.find(owner.concept_of_basic_percept(conjunction), member) is None for member in members)
    assert owner.link_holders(conjunction) == []


def test_a_link_at_zero_is_reached_by_the_gradient_and_learned_below_zero():
    """What tells against a concept is learned on the link (§12.1): taught
    that a place's concept is absent where the conjunction is found, the
    entry on the conjunction's row goes below zero and no other is made."""
    from Layers import NotLayer3, SparseSigmaLDU
    sigma, not_ = SparseSigmaLDU(3, 2, 6), NotLayer3(3)
    sigma.add(0, 0)                       # the place's concept on its own row
    link = sigma.add(0, 2, 0.)            # MTM's link to the conjunction's row
    lanes = torch.tensor([[[1., 0.], [1., 0.]], [[0., 0.], [1., 0.]], [[0., 0.], [1., 0.]]])   # [rows, cases, 2]
    target = torch.tensor([1., 0.])       # the place alone; the place with the conjunction
    optimizer = torch.optim.SGD([sigma.value], lr=.05)
    before = sigma.nnz
    for _ in range(400):
        optimizer.zero_grad()
        read = sigma(not_(lanes), [0, 1, 2], [0])[0]
        (torch.atanh(read.clamp(-.999, .999)) - target).square().sum().backward()
        optimizer.step()
    weights = sigma.w().detach()
    print(dict(own=float(weights[0]), link=float(weights[link])))
    assert sigma.nnz == before and float(weights[link]) < -.9 and float(weights[0]) > .9


def test_the_levels_of_intensity_are_linked_as_a_line():
    """The alphabet of intensities is a linear map over 0 to 1 (Alec,
    2026-11-26: "Value levels need to be linked as linear SOM, 0..1"): a
    level's row is linked to the rows of the levels beside it, both ways,
    at zero, and to no level farther off. A level found at a place can so
    come to tell against its neighbours there."""
    from test_item16_values import _xor
    model = _xor()
    owner = model._concept_owner()
    basic = owner.basic_percepts
    model.train()
    k = len(basic.levels) // 2
    low, middle, high, far = (float(basic.levels[i]) for i in (k - 1, k, k + 1, k + 3))
    for _ in range(owner.concept_mint_count):
        model.forward(torch.tensor([[[low], [middle]], [[high], [far]]]))
    row = {i: basic.atom(i) for i in (k - 1, k, k + 1, k + 3)}
    assert all(basic.promoted[r] for r in row.values())
    assert sorted(basic.level_neighbours(row[k])) == sorted((row[k - 1], row[k + 1]))
    assert basic.level_neighbours(row[k + 3]) == [] and basic.level_neighbours(row[k - 1]) == [row[k]]
    sigma = owner.order_zero
    concept = {i: owner.concept_of_basic_percept(r) for i, r in row.items()}
    linked = {(i, j): sigma.find(concept[i], row[j]) for i in row for j in row if i != j}
    print({pair: None if entry is None else float(sigma.w()[entry]) for pair, entry in linked.items()})
    for (i, j), entry in linked.items():
        if abs(i - j) == 1:
            assert entry is not None and float(sigma.w()[entry]) == 0.
        else:
            assert entry is None
    model.End()


def test_the_alphabet_learns_as_a_linear_self_organizing_map_with_fixed_ends():
    """Alec, 2026-11-26: "each hit both moves its own vector and, to a lesser
    degree, those of its neighbors. And linked at fixed 0 and 1 endpoints";
    "they should change very slowly". Where the levels learn
    (``intensityLevelsFixed`` false), a hit moves its level a small share of
    the way to it, each neighbour half of that, the ends never; the order
    holds; and what fell to a level goes on falling to it."""
    from BasicPercepts import BasicPercepts
    table = BasicPercepts(None, 3, 3, 64)
    table.bind(None)
    table.levels_fixed = False        # the learned alphabet (``intensityLevelsFixed`` false)
    rate, share = table.LEVEL_RATE, table.NEIGHBOUR_SHARE
    assert 0 < rate <= 1e-3 and 0 < share < 1
    last = len(table.levels) - 1
    k = last // 2
    below, own, above = table.levels[k - 1:k + 2].tolist()
    step = above - own
    value = own + step / 4
    table.learn(k, value)
    assert float(table.levels[k]) == pytest.approx(own + rate * (value - own))
    assert float(table.levels[k - 1]) == pytest.approx(below + share * rate * (value - below))
    assert float(table.levels[k + 1]) == pytest.approx(above + share * rate * (value - above))
    # Several hits at once are as many hits one after another.
    twice = BasicPercepts(None, 3, 3, 64)
    twice.bind(None)
    twice.levels_fixed = False
    twice.learn(k, value, 2)
    table.learn(k, value)
    assert float(twice.levels[k]) == pytest.approx(float(table.levels[k]), abs=1e-6)
    # Slowly: after a hundred presentations of values about a level, each
    # still falls to it.
    about = torch.tensor([[own + d * step for d in (-.4, -.1, .05, .3, .35)]])
    steady = BasicPercepts(None, 3, 3, 64)
    steady.bind(None)
    steady.levels_fixed = False
    for _ in range(100):
        assert steady.fall(about).unique().tolist() == [k]
        steady.perceive_intensities(about)
    # The ends stay, hit or beside a hit, and the order holds.
    table.learn(0, .001, 5)
    table.learn(last, .999, 5)
    table.learn(1, float(table.levels[1]) / 2, 3)
    table.learn(last - 1, (1 + float(table.levels[last - 1])) / 2, 3)
    for _ in range(200):
        level = int(torch.randint(0, last + 1, ()))
        table.learn(level, float(torch.rand(())), int(torch.randint(1, 4, ())))
    assert bool((table.levels[1:] >= table.levels[:-1]).all())
    assert float(table.levels[0]) == 0. and float(table.levels[last]) == 1.


def test_the_levels_can_be_kept_linear_and_fixed(monkeypatch):
    """Alec, 2026-11-26: "add a value to the model parameters to keep the
    intensities linear and fixed, so that they map to the range 0..1 in BF16
    when compiled". With ``intensityLevelsFixed`` the levels are k / (n - 1)
    and nothing moves them; a hit is still counted; in bfloat16 the 256
    levels stay in order, within 0..1, with the ends exact."""
    from BasicPercepts import BasicPercepts
    from util import TheXMLConfig
    # Fixed is the default ("otherwise the memories of the values will
    # distort over 1 million sentences").
    with open('data/model.xml') as handle:
        assert '<intensityLevelsFixed>true</intensityLevelsFixed>' in handle.read()
    monkeypatch.setitem(TheXMLConfig._data['architecture'], 'intensityLevelsFixed', True)
    table = BasicPercepts(None, 3, 3, 1024)
    table.bind(None)
    n = len(table.levels)
    linear = torch.arange(n) / (n - 1)
    torch.testing.assert_close(table.levels, linear.to(table.levels))
    made = table.levels.clone()
    k = n // 2
    table.learn(k, float(table.levels[k]) + .001, 7)
    table.perceive_intensities(torch.rand(1, 500).clamp_min(.01))
    assert torch.equal(table.levels, made)
    assert int(table.level_meets.sum()) == 507
    half = table.levels.to(torch.bfloat16)
    assert float(half[0]) == 0. and float(half[-1]) == 1.
    assert bool((half[1:] >= half[:-1]).all()) and bool(((half >= 0) & (half <= 1)).all())
    print(dict(levels=n, distinct_in_bfloat16=int(half.unique().numel())))


def test_both_is_read_one_order_up_across_the_concepts_a_concept_reads():
    """Alec, 2026-11-26: both "is a conceptual thing, it is not a perceptual
    thing, so you can move the test up an order". A concept above order
    zero is both where one concept it reads is read for and another
    against; not where both are for, nor where one is unread; and an entry
    below zero exchanges the lanes of what it reads."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=64, order=3)
    spoon, fork, knife = (_mint_row(cs, 0, 500 + i) for i in range(3))
    tableware, cutlery = _mint_row(cs, 1, 600), _mint_row(cs, 1, 601)
    cs.add_concept_edge(tableware, spoon, 1.)
    cs.add_concept_edge(tableware, fork, 1.)
    cs.add_concept_edge(cutlery, knife, 1.)
    cs.add_concept_edge(cutlery, fork, 1., negated=True)
    lanes = torch.zeros(sum(cs._order_caps()), 1, 4, 2)
    # Reading 0: a spoon, and the fork told against. 1: a spoon and a fork.
    # 2: a spoon alone. 3: a knife and a fork.
    lanes[spoon, 0, :3, 0] = 1.
    lanes[fork, 0, 0, 1] = 1.
    lanes[fork, 0, 1, 0] = 1.
    lanes[knife, 0, 3, 0] = 1.
    lanes[fork, 0, 3, 0] = 1.
    both = cs.both_above(lanes)
    print(dict(tableware=both[tableware, 0].tolist(), cutlery=both[cutlery, 0].tolist()))
    assert both[tableware, 0].tolist() == [1., 0., 0., 0.]
    # Cutlery reads the fork by an entry below zero: a fork found for is
    # against it, beside a knife found for.
    assert both[cutlery, 0].tolist() == [0., 0., 0., 1.]
    assert not both[[spoon, fork, knife]].any()


def test_the_other_gate_reads_no_basic_percept():
    """``test_item25_completion``'s gate is over symbols alone: nothing of
    order zero is read, so item 1.4's removal cannot reach it."""
    from test_item25_completion import test_a_previously_named_shared_concept_denotes_its_raised_class as gate
    import BasicPercepts
    import Spaces
    seen = []
    patch = pytest.MonkeyPatch()
    for owner, name in ((Spaces.ConceptualSpace, 'order_zero_field'), (BasicPercepts.BasicPercepts, 'read_bank')):
        read = getattr(owner, name)
        patch.setattr(owner, name, lambda *args, _read=read, _name=name, **kwargs: (seen.append(_name), _read(*args, **kwargs))[1])
    try:
        gate()
    finally:
        patch.undo()
    assert seen == []


def test_on_text_nothing_is_read_against():
    """The reading lesson's first sixteen sentences. Before the removal every
    row was told against somewhere, mostly where an occurrence held its
    parts and lay outside its whole, and no weighting of the for lanes gave
    it (§11.3). Now no reading and no look reads a row against."""
    from test_item17_order_zero import _read_lesson
    (model, lesson_patch), readings, events = _recorded(lambda: _read_lesson(16))
    try:
        owner = model._concept_owner()
        result, report = PositiveEvidence.summary(readings, owner.order_zero)
        sigma = owner.order_zero
        links = int((sigma.w(occupied_only=True) == 0).sum())
        print(result, events.table(), dict(looks=events.looks, links_at_zero=links))
        assert readings.calls > 0 and events.looks > 0
        assert report == [] and result['against_rows'] == 0
        assert events.counts == [0, 0, 0]
    finally:
        model.End()
        lesson_patch.undo()


def test_the_recorders_leave_the_model_as_it_was():
    class Owner:
        def order_zero_field(self, lanes, inputs, rows):
            return lanes

    owner = Owner()
    with record(owner) as readings:
        owner.order_zero_field(torch.ones(1, 1, 1, 2), [0], None)
    assert readings.calls == 1 and 'order_zero_field' not in vars(owner)

    class Basic:
        conjunctions = {}

        def read_bank(self, bank):
            return bank

    basic = Basic()
    with record_events(basic) as events:
        pass
    assert events.looks == 0 and 'read_bank' not in vars(basic)


def test_a_recorder_puts_back_what_the_instance_already_had():
    """Two recorders nested, over a method the instance already overrides:
    the inner leaves the outer recording, and the outer leaves the
    instance's own override in place."""
    class Owner:
        def order_zero_field(self, lanes, inputs, rows):
            return lanes

    def mine(lanes, inputs, rows):
        return lanes

    owner, lanes = Owner(), torch.ones(1, 1, 1, 2)
    object.__setattr__(owner, 'order_zero_field', mine)
    with record(owner) as outer:
        with record(owner) as inner:
            owner.order_zero_field(lanes, [0], None)
        assert outer.calls == inner.calls == 1
        owner.order_zero_field(lanes, [0], None)
        assert outer.calls == 2 and inner.calls == 1
    assert vars(owner)['order_zero_field'] is mine

    class Basic:
        conjunctions = {}

        def read_bank(self, bank):
            return bank

    basic, kept = Basic(), (lambda bank: bank)
    object.__setattr__(basic, 'read_bank', kept)
    with record_events(basic):
        with record_events(basic):
            pass
        assert 'read_bank' in vars(basic)
    assert vars(basic)['read_bank'] is kept


def test_both_above_reads_each_page_by_its_own_bindings(monkeypatch):
    """Where the field's rows are bound page by page, a concept is read
    both on a page only across the concepts bound there: with the fork
    unbound on the second page, nothing on it is both."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=64, order=3)
    spoon, fork, knife = (_mint_row(cs, 0, 500 + i) for i in range(3))
    tableware, cutlery = _mint_row(cs, 1, 600), _mint_row(cs, 1, 601)
    cs.add_concept_edge(tableware, spoon, 1.)
    cs.add_concept_edge(tableware, fork, 1.)
    cs.add_concept_edge(cutlery, knife, 1.)
    cs.add_concept_edge(cutlery, fork, 1., negated=True)
    lanes = torch.zeros(sum(cs._order_caps()), 1, 4, 2)
    lanes[spoon, 0, :3, 0] = 1.
    lanes[fork, 0, 0, 1] = 1.
    lanes[fork, 0, 1, 0] = 1.
    lanes[knife, 0, 3, 0] = 1.
    lanes[fork, 0, 3, 0] = 1.
    one = cs.both_above(lanes)
    addresses = cs._field_inventory_rows(lanes.device)
    assert addresses.ndim == 1
    paged = torch.stack((addresses, addresses), 1).clone()
    slot = int((addresses == fork).nonzero()[0])
    paged[slot, 1] = -1
    monkeypatch.setattr(cs, '_field_inventory_rows', lambda device=None: paged)
    two = cs.both_above(lanes.repeat(1, 2, 1, 1))
    assert torch.equal(two[:, :1], one)
    assert one.any() and not two[:, 1].any()
