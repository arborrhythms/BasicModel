"""Item 1.7, order zero: basic percepts as rows of a Pi, nominated by MTM,
each with one concept: the Pi, the Not (each row's two values as one signed
value, and the sign of each entry) and a sparse Sigma over the signed values,
its weights at zero or above.

Certificates 5, 11, 12 and 23 (a) to (c) of the sigma-pi specification, §21.
A learning result draws its starts from the system and prints them.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from Layers import EPS_LOG, PiLayer3, NotLayer3, SparseSigmaLayer3
from OrderZero import SparsePiLayer3, read_word


def _drawn(count):
    return [int.from_bytes(os.urandom(4), 'little') for _ in range(count)]


# The first Pi.

def test_the_first_pi_is_pi_layer3_on_its_rows():
    """A row over two members gives the product of their memberships on its
    own lane, as a PiLayer3 row with weight one on the two members."""
    pi = SparsePiLayer3(4, 16)
    pi.append([('ps', (1,)), ('ws', 7)])
    pi.append([('ps', (2,))])
    m = torch.tensor([[.9, .2], [.8, .5], [.3, 0.]])  # three keys, two lanes
    keys = [('ps', (1,)), ('ws', 7), ('ps', (2,))]
    got = pi(m, (0, 1), keys)
    dense = PiLayer3(3, 3)
    with torch.no_grad():
        for layer in dense.lane_layers:
            for p in layer.parameters():
                p.zero_()
    lanes = m.t()  # [lane, key]
    logs = torch.log(lanes.clamp(EPS_LOG, 1.))
    weights = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])  # members of row 0 and row 1
    expected = torch.exp(logs @ weights)[:, :2].t()
    expected[1, 1] = 0.  # a member below the floor annihilates the row
    torch.testing.assert_close(got, expected)
    assert pi.rows_reading(('ws', 7)) == [0] and pi.rows_reading(('ps', (2,))) == [1]
    assert pi.row_members(0) == [('ps', (1,)), ('ws', 7)]


def test_the_first_pi_is_stored_as_compressed_rows_in_capacity_allocated_beforehand():
    pi = SparsePiLayer3(2, 3)
    before = (pi.crow.data_ptr(), pi.col.data_ptr(), pi.value.data_ptr())
    pi.append([('ps', (1,)), ('ws', 3)])
    assert (pi.crow.data_ptr(), pi.col.data_ptr(), pi.value.data_ptr()) == before
    assert pi.crow[:2].tolist() == [0, 2] and pi.col[:2].tolist() == [0, 1]
    with pytest.raises(OverflowError):
        pi.append([('ps', (2,)), ('ws', 4)])


def test_a_row_with_no_member_present_or_none_at_all_reads_nothing():
    pi = SparsePiLayer3(4, 8)
    pi.append([('ps', (1,)), ('ws', 2)])
    pi.append([])
    got = pi(torch.tensor([[1., 0.]]), (0, 1), [('ps', (1,))])
    assert torch.equal(got, torch.zeros(2, 2))


# Certificate 5: order zero, on rows (rule 62).

class Order0:
    """The Not and the Sigma over ``rows`` rows and ``concepts`` concepts,
    with ``entries`` ``[(concept, row, sign)]`` written: sign 1 a for entry,
    sign -1 a Not entry made by the Not."""

    def __init__(self, rows, concepts, entries=(), room=None):
        self.not_ = NotLayer3(rows)
        self.sigma = SparseSigmaLayer3(rows, concepts, room or 2 * (rows + concepts))
        for concept, row, sign in entries:
            if sign > 0:
                self.sigma.add(concept, row)
            else:
                self.not_.make(self.sigma, concept, row)

    def __call__(self, lanes, inputs, rows):
        return self.sigma(self.not_(lanes), inputs, rows)


def test_the_sigma_reads_each_rows_signed_value_and_an_absent_row_changes_nothing():
    """The Not offers each row's value (for less against) and not-value; a
    concept reads each row it has an entry on by its weight, through a Not
    entry the not-value,
    ``tanh(sum w s - sum w_not s)``; a row at zero, a row not among the
    inputs and a row it has no entry on change nothing; a concept with no
    entry reads zero."""
    order = Order0(6, 4, [(0, 0, 1), (0, 2, 1), (1, 1, 1), (1, 2, -1), (2, 5, 1)])
    sigma = order.sigma
    with torch.no_grad():
        sigma.value.mul_(1 + torch.rand_like(sigma.value))
    lanes = torch.rand(4, 3, 2)
    inputs = [2, 0, 1, 4]
    y = order(lanes, inputs, [0, 1, 2, 3, -1])
    dense = torch.zeros(6, 4)
    for e in range(sigma.nnz):
        dense[int(sigma.entry_input[e]), int(sigma.entry_row[e])] = sigma.value[e].detach()
    signed = torch.zeros(6, 3)
    signed[inputs] = lanes[..., 0] - lanes[..., 1]
    expected = torch.tanh(signed.t() @ dense).t()
    # Each weight is the entry's size, at zero or above; the sign is the Not's.
    assert bool((sigma.weight >= 0).all()) and NotLayer3.entries(sigma).tolist() == [False, False, False, True, False]
    torch.testing.assert_close(y[:4], expected, atol=1e-6, rtol=0)
    assert torch.equal(y[3], torch.zeros(3)) and torch.equal(y[4], torch.zeros(3))
    # An input row at zero, or one left out, changes nothing.
    torch.testing.assert_close(order(torch.cat((lanes, torch.zeros(1, 3, 2))), inputs + [5], [0, 1, 3]),
                               y[[0, 1, 3]])
    torch.testing.assert_close(order(lanes[:3], inputs[:3], [0, 1]), y[:2])
    # The order of the inputs is not read.
    torch.testing.assert_close(order(lanes.flip(0), inputs[::-1], [0, 1, 2]), y[:3])


def test_a_percept_never_given_to_a_concept_reads_zero():
    """Given up with rule 62 (§18.24, certificate 15): a concept answers
    only to the rows it has entries on. The concept of ``cat`` reads its
    own row at tanh(1) and the row of ``cats``, whose code holds all of
    cat's, at zero."""
    order = Order0(3, 1, [(0, 0, 1)])
    read = {name: float(order(torch.tensor([[[1., 0.]]]), [row], [0]).detach())
            for name, row in (('cat', 0), ('cats', 1), ('dog', 2))}
    print(read)
    assert read['cat'] == pytest.approx(torch.tanh(torch.tensor(1.)).item(), abs=1e-6)
    assert read['cats'] == 0. and read['dog'] == 0.


def test_the_mereological_order_of_the_lesson_percepts(lesson):
    """§18.24, certificate 18: every basic percept's code is at or above each
    part's and at or below each whole's; a conjunction's code is above each
    member's; no code is its bytes' sum."""
    basic = lesson['basic']
    checked = 0
    for row in range(len(basic.index)):
        code = basic.code(row)
        if row in basic.conjunctions:
            for member in basic.conjunctions[row]:
                assert bool((code >= basic.code(member)).all())
            checked += 1
            continue
        parts, wholes = basic.occurrences[row]
        identity = code[:basic.identity_width]
        part_codes = basic.part_codes(torch.tensor(parts))
        assert bool((identity[None] >= part_codes).all())
        # The greatest over its parts, not the sum of its bytes' codes.
        assert torch.equal(identity, part_codes.amax(0))
        if wholes and basic.property_width:
            whole_codes = basic.whole_codes(torch.tensor(wholes))
            assert bool((code[basic.identity_width:][None] <= whole_codes).all())
        checked += 1
    print(dict(rows=len(basic.index), checked=checked))
    assert checked == len(basic.index) > 0


def test_how_many_concepts_are_told_apart(lesson):
    """§18.24, certificate 17, reported, on rows: each nominated row read
    alone, at for 1 and against 0; the share of its concepts that read it
    above every other concept, and the largest reading a concept takes from
    a row that is not its own."""
    owner, basic = lesson['owner'], lesson['basic']
    pairs = [(row, owner.concept_of_basic_percept(row)) for row in range(len(basic.index))
             if bool(basic.promoted[row])]
    pairs = [(row, concept) for row, concept in pairs if concept is not None]
    if not pairs:
        print(dict(nominated=0))
        return
    rows = [row for row, _ in pairs]
    concepts = [concept for _, concept in pairs]
    presence = torch.eye(len(rows))[..., None].expand(-1, -1, 2).clone()
    presence[..., 1] = 0.
    y = owner.order_zero_field(presence[:, :, None], rows, concepts)[:, :, 0].detach()
    y = y[..., 0] - y[..., 1]  # [concept, reading]
    own = y.diagonal()
    other = y - torch.diag(own) - torch.eye(len(rows)) * 2
    apart = float(((own[:, None] > other.max(1, keepdim=True).values) | (len(rows) == 1)).float().mean())
    print(dict(nominated=len(rows), told_apart=apart, own_least=float(own.min()),
               largest_other=float(other.max()) if len(rows) > 1 else None))


def test_the_way_back_as_far_as_it_goes():
    """§18.24, certificate 19, on rows: with a concept for each row (and
    entries besides), the Sigma returns each row's signed value, for less
    against; with fewer concepts than rows it refuses."""
    torch.manual_seed(int.from_bytes(os.urandom(4), 'little'))
    rows = 6
    entries = [(r, r, 1) for r in range(rows)] + [(0, 3, 1), (5, 2, -1)]
    order = Order0(rows, rows, entries)
    with torch.no_grad():
        order.sigma.value.mul_(1 + torch.rand(order.sigma.capacity))
    lanes = torch.rand(rows, 2, 2) * .5
    y = order(lanes, list(range(rows)), list(range(rows)))
    back = order.sigma.reverse(y, list(range(rows)))
    error = float((back - (lanes[..., 0] - lanes[..., 1])).abs().max())
    print(dict(rows=rows, error=error))
    assert error < 1e-4
    narrow = Order0(rows, rows - 1, [(r, r, 1) for r in range(rows - 1)] + [(0, rows - 1, 1)])
    with pytest.raises(ValueError):
        narrow.sigma.reverse(torch.zeros(rows - 1, 1), list(range(rows)))


def test_the_order_zero_sigma_is_sparse_and_not_penalized(lesson):
    """After the first Pi, the Not and the Sigma of order zero: the Sigma is
    the rows by the order-zero concepts, stored as
    compressed rows in capacity allocated beforehand, a parameter of the
    model, saved with it, with no penalty. It has no input but the rows' signed
    values: no code and no place (rule 62)."""
    owner, basic = lesson['owner'], lesson['basic']
    sigma, not_ = owner.order_zero, owner.order_zero_not
    assert isinstance(sigma, SparseSigmaLayer3) and isinstance(not_, NotLayer3)
    assert (not_.rows, sigma.nInput, sigma.nOutput) == (basic.capacity, basic.capacity, owner._order_caps()[0])
    assert any(p is sigma.value for p in owner.getParameters())
    keys = {key[key.index('order_zero.'):] for key in owner.state_dict() if 'order_zero.' in key}
    assert {'order_zero.value', 'order_zero.entry_row', 'order_zero.entry_input', 'order_zero.count'} <= keys
    assert not hasattr(sigma, 'l1_lambda')
    n = sigma.nnz
    assert n > 0 and sum(len(sigma.entries(c)) for c in set(sigma.entry_row[:n].tolist())) == n
    assert bool((sigma.weight >= 0).all())
    for concept in sorted(set(sigma.entry_row[:n].tolist()))[:20]:
        assert all(concept in sigma.rows_reading(i) for i, _, _ in sigma.entries(concept))
    print(dict(entries=n, capacity=sigma.capacity, concepts=sigma.nOutput, rows=not_.rows,
               not_entries=int(NotLayer3.entries(sigma).sum())))


def test_the_sparse_sigma_is_stored_in_capacity_allocated_beforehand():
    sigma, not_ = SparseSigmaLayer3(4, 3, 3), NotLayer3(4)
    before = (sigma.value.data_ptr(), sigma.entry_row.data_ptr(), sigma.entry_input.data_ptr())
    sigma.add(0, 2)
    not_.make(sigma, 2, 1, .5)
    sigma.add(0, 2)  # an entry already written is added to
    assert (sigma.value.data_ptr(), sigma.entry_row.data_ptr(), sigma.entry_input.data_ptr()) == before
    assert sigma.entries(0) == [(2, 2., False)] and sigma.entries(2) == [(1, .5, True)]
    assert [len(sigma.entries(c)) for c in range(3)] == [1, 0, 1] and sigma.rows_reading(2) == [0]
    with pytest.raises(ValueError):
        sigma.add(1, 3, -1.)
    sigma.add(1, 3)
    with pytest.raises(OverflowError):
        sigma.add(1, 0)
    restored = SparseSigmaLayer3(4, 3, 3)
    restored.load_state_dict(sigma.state_dict())
    assert restored.entries(2) == sigma.entries(2) and restored.rows_reading(1) == [2]


def test_a_word_reads_its_rows_entry():
    """The word step reads its basic percept by the Pi of its two members,
    the Not, and its concept by its entry on the row: the same value the
    field read gives, for an entry and for a Not entry."""
    evidence = torch.tensor([[[.9, .8], [.5, -1.]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    # The run of parts is the presence on both lanes, the whole on its own:
    # [.9 * .8, .9 * 0] and [.5 * 0, .5 * 1].
    presence = torch.tensor([[.72, 0.], [0., .5]])
    for sign in (1, -1):
        order = Order0(1, 1, [(0, 0, sign)])
        with torch.no_grad():
            order.sigma.value[0] *= .7
        y = read_word(evidence, mask, order.sigma.entry([0], [0])[None, :, None])
        for location in range(2):
            expected = order(presence[location][None, None], [0], [0])[0, 0]
            torch.testing.assert_close(y[0, location], expected)
    # The against lane: a present run whose whole is witnessed not to hold
    # reads the percept against; an absent run reads nothing.
    start = torch.tensor([[[1.]]])
    assert float(read_word(torch.tensor([[[.5, -1.]]]), torch.ones(1, 1, 2, dtype=torch.bool), start)) < 0
    assert float(read_word(torch.tensor([[[-.5, -1.]]]), torch.ones(1, 1, 2, dtype=torch.bool), start)) == 0


def test_a_gradient_reaches_only_the_entries_read():
    order = Order0(5, 6, [(1, 0, 1), (4, 2, 1), (2, 3, 1), (1, 4, 1)])
    order(torch.rand(3, 2, 2), [0, 2, 4], [1, 4, -1]).sum().backward()
    sigma = order.sigma
    touched = sorted(int(sigma.entry_row[e]) * 10 + int(sigma.entry_input[e])
                     for e in sigma.value.grad.abs().nonzero().flatten().tolist())
    assert touched == [10, 14, 42]


# Certificates 11 and 12: rows by nomination, one concept for each row.

def test_no_row_on_a_first_reading(lesson):
    """Certificate 11: a new word read once has no basic percept and no
    concept, and interpret does not find it; at concept_mint_count readings
    MTM nominates its basic percept, its concept comes with it, and
    interpret finds it."""
    model, owner, basic = lesson['model'], lesson['owner'], lesson['basic']
    identity = model.perceptualSpace.percept_store.identity
    form = 'zorbix'
    parts = identity.admit(form.encode())
    wholes = model.wholeSpace.property_rows_for_bytes(form)
    rows_before, words_before = len(basic.index), dict(basic.words)
    found = []
    for reading in range(owner.concept_mint_count):
        result = owner.interpret_word(parts, wholes, key=form, occurrence=(10_000 + reading, 0, 0))
        found.append(None if result is None else int(result[0]))
        if reading < owner.concept_mint_count - 1:
            assert result is None and form not in {f for v in owner.interpret._pending.values()
                                                   for f in v['forms']}
            assert len(basic.index) == rows_before and basic.words == words_before
    word = found[-1]
    print(dict(mint_count=owner.concept_mint_count, found=found))
    assert word is not None
    assert owner.interpret.lookup_word(parts, wholes, form=form, reading=('again', 0)) == word
    row = basic.words[word]
    assert bool(basic.promoted[row]) and row == rows_before
    c_row = owner._concept_row_of(word)
    assert owner._row_order(c_row) == 0 and owner.concept_percepts(c_row) == [row]
    lesson['nominated'] = (word, row, c_row)


def test_square_a_new_concept_reads_only_its_own_row(lesson):
    """Certificate 12: at order zero the concepts equal the nominated rows in
    number; a new concept has one entry, on its own row, at one, so that
    its own row read for reads one before the squash and read both reads
    zero; it has no weight on anything else (rule 62)."""
    owner, basic = lesson['owner'], lesson['basic']
    if 'nominated' not in lesson:
        test_no_row_on_a_first_reading(lesson)
    word, row, c_row = lesson['nominated']
    assert owner.order_zero.entries(c_row) == [(row, 1., False)]
    bound = 1 - 1e-7
    read = lambda lanes: float(torch.atanh(owner.order_zero_field(torch.tensor([[[lanes]]]), [row], [c_row])
                                           .detach()[0, 0, 0, 0].clamp(-bound, bound)))
    assert read([1., 0.]) == pytest.approx(1., abs=1e-5) and read([1., 1.]) == 0.
    nominated = [r for r in range(len(basic.index)) if bool(basic.promoted[r])]
    concepts = {r: owner.concept_of_basic_percept(r) for r in nominated}
    frozen = set(owner._frozen_rows())
    from FieldProjection import inventory
    order_zero = [r for r in inventory(owner) if owner._row_order(r) == 0 and r not in frozen]
    readers = {}
    for c in order_zero:
        for percept in owner.concept_percepts(c):
            readers.setdefault(percept, []).append(c)
    print(dict(nominated_rows=len(nominated), order_zero_concepts=len(order_zero),
               rows_without_concept=sum(concept is None for concept in concepts.values()),
               rows_read_by_several=sum(len(c) > 1 for c in readers.values())))
    assert all(concept is not None for concept in concepts.values())


# Certificate 23 (a) to (c): Pi, Not, Sigma at the basic level.

def _taught(cases, *, rows, steps=2000, project=False, seed):
    """One concept taught on readings of nominated rows. ``cases`` maps the
    present base percepts to the target; a conjunction row is present when
    all its members are. The concept has an entry on every row, each at
    zero, so what is learned does not depend on a draw. Every weight is at
    zero or above; the gradient may carry an entry through zero, which makes
    it a Not entry (it reads the row's not-value): that is the only teaching
    of the Not here. With ``project`` every entry is held on the value (its
    learned number at zero or above), so there is no Not entry."""
    torch.manual_seed(seed)
    pi = SparsePiLayer3(len(rows), 4 * len(rows))
    keys = [('bp', r) for r in range(len(rows))]
    for members in rows:
        pi.append([('bp', m) for m in members] if members else [])
    base = [r for r, members in enumerate(rows) if not members]

    def presence(present):
        lanes = torch.zeros(len(rows), 2)
        for r in present:
            lanes[r, 0] = 1.
        values = list(lanes)
        for r, members in enumerate(rows):
            if members:
                values[r] = pi(torch.stack(values), (r,), keys)[0]
        return torch.stack(values)

    order = Order0(len(rows), 1, [(0, r, 1) for r in range(len(rows))])
    sigma, not_ = order.sigma, order.not_
    with torch.no_grad():
        sigma.value.zero_()
    inputs = torch.stack([presence(present) for present in cases], 1)  # [rows, case, 2]
    target = torch.tensor(list(cases.values()))
    optimizer = torch.optim.Adam(sigma.parameters(), lr=.05)
    every = list(range(len(rows)))
    for step in range(steps):
        optimizer.zero_grad()
        y = order(inputs, every, [0])[0]
        ((y - target) ** 2).mean().backward()
        optimizer.step()
        if project:
            with torch.no_grad():
                sigma.value.clamp_(min=0)

    def read(present):
        return float(order(presence(present)[:, None], every, [0]).detach())
    return order, read, base


def _not_entries(order, floor=0.):
    """Entries that are Not entries, of weight above ``floor``."""
    sigma = order.sigma
    return int((NotLayer3.entries(sigma) & (sigma.weight.detach()[:sigma.nnz] > floor)).sum())


def _with_rows(supports, conjunctions):
    """Base rows, one for each support, and conjunction rows over them."""
    return [()] * len(supports) + [tuple(c) for c in conjunctions]


def test_with_no_omission_there_is_no_not_entry_and_nothing_is_against():
    """Certificate 23 (a): nothing expected fails to come, so nothing teaches
    a Not entry, and the gradient makes none."""
    rows = _with_rows(((0, 1, 2), (3, 4, 5), (6, 7, 8)), ((0, 1),))
    seed = _drawn(1)[0]
    # What came is one; nothing expected failed to come.
    order, read, _ = _taught({(0,): 1., (1,): 1., (0, 1, 3): 1.}, rows=rows, seed=seed)
    assert _not_entries(order) == 0 and bool((order.sigma.weight >= 0).all())
    readings = {present: read(present) for present in [(), (0,), (1,), (2,), (0, 1, 3), (0, 2), (1, 2)]}
    print(dict(seed=seed, readings=readings))
    assert all(value >= 0 for value in readings.values())
    assert readings[(0, 2)] >= readings[(0,)] and readings[(1, 2)] >= readings[(1,)]


def test_the_basic_level_learns_negative_patterning_and_conditioned_inhibition():
    """Certificate 23 (b): "A or B, not both" and "A alone, not A with X",
    from what was expected and what came; after it X alone is expected
    absent. Rows for the compounds are nominated (AB, AX). The Sigma starts
    at zero, so what is learned does not depend on a draw: one run."""
    seeds = _drawn(1)
    rows = _with_rows(((0, 1, 2), (3, 4, 5), (6, 7, 8)), ((0, 1), (0, 2)))
    A, B, X, AB, AX = range(5)
    patterning = inhibition = 0
    report = []
    for seed in seeds:
        _, read, _ = _taught({(A,): 1., (B,): 1., (A, B, AB): 0.}, rows=rows, seed=seed)
        one = dict(A=read((A,)), B=read((B,)), AB=read((A, B, AB)), nothing=read(()))
        patterning += one['A'] > .5 and one['B'] > .5 and one['AB'] < .5
        order, read, _ = _taught({(A,): 1., (A, X, AX): 0., (B,): 1.}, rows=rows, seed=seed)
        two = dict(A=read((A,)), AX=read((A, X, AX)), X=read((X,)), not_entries=_not_entries(order, .01))
        inhibition += two['A'] > .5 and two['AX'] < .5 and two['X'] < 0
        report.append(dict(seed=seed, patterning=one, inhibition=two))
    print(report)
    assert patterning == len(seeds) and inhibition == len(seeds)


def test_the_basic_level_learns_xor_of_two_perceived_words_with_no_not_entry():
    """Certificate 23 (c): true and false each perceived at two places, a
    row for each pair; every entry held on the value, so no Not entry. The Sigma starts at
    zero, so what is learned does not depend on a draw: one run."""
    seeds = _drawn(1)
    words = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))  # true1, false1, true2, false2
    pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
    rows = _with_rows(words, pairs)
    cases = {(0, 2, 4): 0., (0, 3, 5): 1., (1, 2, 6): 1., (1, 3, 7): 0.}
    solved, report = 0, []
    for seed in seeds:
        order, read, _ = _taught(cases, rows=rows, project=True, seed=seed)
        readings = {present: read(present) for present in cases}
        ok = all((value > .5) == (target > .5) for value, target in
                 zip(readings.values(), cases.values()))
        solved += ok
        report.append(dict(seed=seed, readings=list(readings.values()),
                           not_entries=_not_entries(order)))
    print(report)
    assert solved == len(seeds) and all(r['not_entries'] == 0 for r in report)


# The lesson's sixteen sentences, read once for the tests that read a model.

def _read_lesson(sentences):
    """The lesson's model after its first ``sentences`` sentences (a count,
    or a function of the model), one stream; returns the model and the
    patch to undo."""
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    patch = MonkeyPatch()
    use_eager_reading(patch)
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=.001)
    curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
    texts, targets = stage_documents(TheData, curriculum['stages'][0])
    admit_training_forms(model, texts, targets)
    sentences = sentences(model) if callable(sentences) else int(sentences)
    count = 0

    class Complete(Exception):
        pass

    def batch(*args, **kwargs):
        nonlocal count
        count += 1
        if count == sentences:
            raise Complete
    try:
        present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
    except Complete:
        pass
    return model, patch


@pytest.fixture(scope='module')
def lesson():
    model, patch = _read_lesson(16)
    owner = model._concept_owner()
    yield dict(model=model, owner=owner, basic=owner.basic_percepts)
    model.End()
    patch.undo()


def test_sentence_trees_are_released_without_the_collector():
    """With automatic collection off, the lesson reads more sentences than
    a stream has trials; every trial's tree is released when its sentence
    closes, so no capture finds its trials all retained and the collector
    finds no sentence tree."""
    import gc
    from SentenceStrata import Tree
    read = []

    def sentences(model):
        trials = model.conceptualSpace.stm.strata.trials
        read.append(trials + 2)
        return trials + 2
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        model, patch = _read_lesson(sentences)
        try:
            strata = model.conceptualSpace.stm.strata
            print(f'{read[0]} sentences read, {strata.trials} trials a stream')
            assert all(lease is None or lease() is None for lease in strata._leases)
            gc.set_debug(gc.DEBUG_SAVEALL)
            try:
                gc.collect()
                # This model's trees (another model of the module may
                # release its own here).
                stores = {id(store) for store in strata.idea_stores}
                left = [item for item in gc.garbage
                        if isinstance(item, Tree) and id(item.ideas) in stores]
            finally:
                gc.set_debug(0)
                gc.garbage.clear()
            assert not left
        finally:
            model.End()
            patch.undo()
    finally:
        if enabled:
            gc.enable()
