"""Item 1.7, order zero: basic percepts as rows of a Pi, nominated by MTM,
each with one concept read by a dense Sigma over the combined vector.

Certificates 5, 11, 12 and 23 (a) to (c) of the sigma-pi specification, §21.
A learning result draws its starts from the system and prints them.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from Layers import EPS_LOG, PiLayer3
from OrderZero import PerceptPi, OrderZeroSigma, combine, read_word


def _drawn(count):
    return [int.from_bytes(os.urandom(4), 'little') for _ in range(count)]


def _codes(*supports, width=32):
    codes = torch.zeros(len(supports), width)
    for row, support in enumerate(supports):
        codes[row, list(support)] = 1.
    return codes


def _pairs(codes, against=None):
    """Read pairs, ``[P, 2, W]``: what tells for each percept (``codes``)
    and what tells against it (none, unless given)."""
    return torch.stack((codes, torch.zeros_like(codes) if against is None else against), -2)


# The first Pi.

def test_the_first_pi_is_pi_layer3_on_its_rows():
    """A row over two members gives the product of their memberships on its
    own lane, as a PiLayer3 row with weight one on the two members."""
    pi = PerceptPi(4, 16)
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
    pi = PerceptPi(2, 3)
    before = (pi.crow.data_ptr(), pi.col.data_ptr(), pi.value.data_ptr())
    pi.append([('ps', (1,)), ('ws', 3)])
    assert (pi.crow.data_ptr(), pi.col.data_ptr(), pi.value.data_ptr()) == before
    assert pi.crow[:2].tolist() == [0, 2] and pi.col[:2].tolist() == [0, 1]
    with pytest.raises(OverflowError):
        pi.append([('ps', (2,)), ('ws', 4)])


def test_a_row_with_no_member_present_or_none_at_all_reads_nothing():
    pi = PerceptPi(4, 8)
    pi.append([('ps', (1,)), ('ws', 2)])
    pi.append([])
    got = pi(torch.tensor([[1., 0.]]), (0, 1), [('ps', (1,))])
    assert torch.equal(got, torch.zeros(2, 2))


# Certificate 5: order zero.

def test_the_combined_vector_is_one_less_the_product_of_the_gaps():
    """Each percept enters once at its net, for less against: what tells
    for it joins the for lane and what tells against it the against lane at
    a net above zero, the reverse below; one read equally both adds
    nothing."""
    pairs = torch.rand(5, 2, 12)
    presence = torch.rand(5, 3, 2)
    presence[2] = 0.
    x = combine(presence, pairs)
    net = presence[..., 0] - presence[..., 1]
    up, down = net.clamp_min(0)[..., None], (-net).clamp_min(0)[..., None]
    gaps = (1 - pairs)[:, None]
    toward = 1 - torch.prod(gaps[:, :, 0] ** up * gaps[:, :, 1] ** down, 0)
    away = 1 - torch.prod(gaps[:, :, 1] ** up * gaps[:, :, 0] ** down, 0)
    torch.testing.assert_close(x, torch.stack((toward, away), -2), atol=1e-5, rtol=1e-5)
    assert bool((x >= 0).all() and (x <= 1).all())
    # A percept at zero changes nothing, and nor does one read both equally.
    torch.testing.assert_close(combine(presence[[0, 1, 3, 4]], pairs[[0, 1, 3, 4]]), x)
    both = presence.clone()
    both[2] = .7
    torch.testing.assert_close(combine(both, pairs), x)


def test_a_readings_own_place_follows_its_presence_and_is_not_combined():
    """Rule 7: a reading's own place is read at its presence; its time is
    not read, so the combined vector ends with the place."""
    codes = _codes((0, 1), (2, 3), width=8)
    presence = torch.tensor([[[.5, 0.]], [[1., 0.]]])  # two percepts, one reading
    where = torch.tensor([[.25, .75]])
    x = combine(presence, _pairs(codes), where=where)
    assert x.shape[-1] == 8 + 2
    torch.testing.assert_close(x[0, 0, 8:], torch.tensor([.25, .75]))
    torch.testing.assert_close(x[0, 1, 8:], torch.zeros(2))


def test_a_new_percept_needs_no_new_weight():
    """§18.24, certificate 15: a percept never presented with a concept reads
    it from its vector alone; a percept with an unlike vector does not."""
    sigma = OrderZeroSigma(36, 1, content_width=32)
    cat, cats, dog = _codes((0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), (10, 11, 12, 13, 14))
    sigma.start_column(0, cat)
    place = torch.zeros(1, 4)
    read = {name: float(sigma.concepts(combine(torch.tensor([[[1., 0.]]]), _pairs(code[None]), where=place), [0]))
            for name, code in (('cat', cat), ('cats', cats), ('dog', dog))}
    print(read)
    # A concept reads its own row at tanh(1); a form holding all its parts
    # reads as much; an unlike vector reads nothing.
    assert read['cat'] == pytest.approx(torch.tanh(torch.tensor(1.)).item(), abs=1e-5)
    assert read['cats'] == pytest.approx(read['cat'], abs=1e-5) and read['dog'] == 0.


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
    """§18.24, certificate 17, reported: each nominated row read alone; the
    share of its concepts that read it above every other concept, and the
    largest reading a concept takes from a row that is not its own."""
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
    codes = basic.read_codes(rows)
    x = combine(presence, codes)
    zeros = x.new_zeros(*x.shape[:-1], owner.order_zero.nInput - x.shape[-1])
    y = owner.order_zero.concepts(torch.cat((x, zeros), -1), concepts).detach()
    own = y.diagonal()
    other = y - torch.diag(own) - torch.eye(len(rows)) * 2
    apart = float(((own[:, None] > other.max(1, keepdim=True).values) | (len(rows) == 1)).float().mean())
    print(dict(nominated=len(rows), told_apart=apart, own_least=float(own.min()),
               largest_other=float(other.max()) if len(rows) > 1 else None))


def test_the_way_back_as_far_as_it_goes():
    """§18.24, certificate 19: with as many concepts as the width the Sigma
    returns the combined vector; with fewer it refuses."""
    torch.manual_seed(int.from_bytes(os.urandom(4), 'little'))
    width = 12
    sigma = OrderZeroSigma(width, width, content_width=width - 4)
    with torch.no_grad():
        sigma.weight.copy_(torch.eye(width) + .1 * torch.rand(width, width))
    codes = torch.rand(3, width - 4)
    presence = torch.rand(3, 2, 2) * .5
    x = combine(presence, _pairs(codes), where=torch.rand(2, 4))
    signed = x[..., 0, :] - x[..., 1, :]
    y = torch.tanh(signed @ sigma.weight)
    back = sigma.reverse(y)
    error = float((back[..., 0, :] - back[..., 1, :] - signed).abs().max())
    print(dict(width=width, error=error))
    assert error < 1e-4
    narrow = OrderZeroSigma(width, width - 1, content_width=width - 4)
    with pytest.raises(ValueError):
        narrow.reverse(torch.zeros(1, width - 1))


def test_place_weights_after_teaching():
    """§18.24, certificate 20, reported: a concept taught at several places
    keeps what it learns on the place coordinates (rule 7: no time is read)."""
    seed = int.from_bytes(os.urandom(4), 'little')
    torch.manual_seed(seed)
    sigma = OrderZeroSigma(12, 1, content_width=8)
    code = _codes((0, 1, 2), width=8)
    sigma.start_column(0, code)
    places = torch.rand(6, 4)
    x = combine(torch.ones(1, 6, 2) * torch.tensor([1., 0.]), _pairs(code), where=places)
    optimizer = torch.optim.Adam(sigma.parameters(), lr=.05)
    for _ in range(300):
        optimizer.zero_grad()
        ((sigma.concepts(x, [0]) - .9) ** 2).mean().backward()
        optimizer.step()
    print(dict(seed=seed, content=sigma.weight[:8, 0].tolist(), place=sigma.weight[8:, 0].tolist()))


def test_the_order_zero_sigma_is_dense_and_not_penalized(lesson):
    """The Sigma of order zero is the perceptual width by the order-zero
    rows, a parameter of the model, saved with it, with no penalty."""
    owner = lesson['owner']
    sigma = owner.order_zero
    model = lesson['model']
    origin = torch.zeros(1, dtype=torch.long)
    assert sigma.where_width == model.where_registry.encode('text', origin).shape[-1]
    # Rule 7: the reading's time is not read; the width holds no when.
    assert not hasattr(sigma, 'when_width')
    width = owner.basic_percepts.width + sigma.where_width
    assert tuple(sigma.weight.shape) == (width, owner._order_caps()[0])
    assert any(p is sigma.weight for p in owner.getParameters())
    assert any(key.endswith('order_zero.weight') for key in owner.state_dict())
    assert not hasattr(sigma, 'l1_lambda')


def test_a_word_reads_its_column_of_the_order_zero_sigma():
    """The word step reads its basic percept by the Pi of its two members
    and its concept by the column: the same value the field read gives."""
    evidence = torch.tensor([[[.9, .8], [.5, -1.]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    code = torch.tensor([.6, .3, 0., 1.])
    column = torch.tensor([.3, .2, .1, .5, .25, -.25])
    place = torch.tensor([[[.1, .9], [.4, .6]]])
    # The two columns: the for weights, and the against weights begun as
    # the for weights negated (rule 59).
    columns = torch.cat((column, -column))
    y = read_word(evidence, mask, torch.cat((columns, code, torch.zeros(4)))[None, None], place, content_width=4)
    # The run of parts is the presence on both lanes, the whole on its own:
    # [.9 * .8, .9 * 0] and [.5 * 0, .5 * 1].
    presence = torch.tensor([[[.72, 0.]], [[0., .5]]])  # [P=1 per location, lanes]
    for location in range(2):
        x = combine(presence[location][:, None], _pairs(code[None]), where=place[:, location])
        expected = torch.tanh((x[0, 0] - x[0, 1]) @ column)
        torch.testing.assert_close(y[0, location], expected)
    # The against lane: a present run whose whole is witnessed not to hold
    # reads the percept against; an absent run reads nothing.
    against = read_word(torch.tensor([[[.5, -1.]]]), torch.ones(1, 1, 2, dtype=torch.bool),
                        torch.cat((columns, code, torch.zeros(4)))[None, None], place[:, :1], content_width=4)
    assert float(against) < 0
    absent = read_word(torch.tensor([[[-.5, -1.]]]), torch.ones(1, 1, 2, dtype=torch.bool),
                       torch.cat((columns, code, torch.zeros(4)))[None, None], place[:, :1], content_width=4)
    assert float(absent) == 0


def test_a_gradient_reaches_only_the_columns_read():
    sigma = OrderZeroSigma(10, 6, content_width=8)
    with torch.no_grad():
        sigma.weight.uniform_(0, .1)
    x = combine(torch.rand(2, 3, 2), torch.rand(2, 2, 8), where=torch.rand(3, 2))
    sigma.concepts(x, [1, 4, -1]).sum().backward()
    touched = sigma.weight.grad.abs().sum(0).nonzero().flatten().tolist()
    assert touched == [1, 4]


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
    c_row = owner._csw_row_of(word)
    assert owner._row_order(c_row) == 0 and owner.concept_percepts(c_row) == [row]
    lesson['nominated'] = (word, row, c_row)


def test_square_a_new_concept_reads_only_its_own_row(lesson):
    """Certificate 12: at order zero the concepts equal the nominated rows in
    number; a new concept's column is its row's vector, normalized, with no
    weight on place and time: the row's vector over its squared length, so
    that its own row reads one before the squash (rule 8b)."""
    owner, basic = lesson['owner'], lesson['basic']
    if 'nominated' not in lesson:
        test_no_row_on_a_first_reading(lesson)
    word, row, c_row = lesson['nominated']
    column = owner.order_zero.weight[:, c_row].detach()
    # Rule 54: the member's share, from what tells for and against it.
    torch.testing.assert_close(column[:basic.width], basic.column(row))
    # Its own row reads one before the squash.
    x = combine(torch.tensor([[[1., 0.]]]), basic.read_codes([row]))
    torch.testing.assert_close(float(x[0, 0] @ column[:basic.width]), 1., rtol=0, atol=1e-5)
    assert torch.count_nonzero(column[basic.width:]) == 0
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

def _taught(cases, *, rows, codes, steps=2000, project=False, seed):
    """One concept taught on readings of nominated rows. ``cases`` maps the
    present base percepts to the target; a conjunction row is present when
    all its members are."""
    torch.manual_seed(seed)
    pi = PerceptPi(len(rows), 4 * len(rows))
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

    sigma = OrderZeroSigma(codes.shape[1], 1, content_width=codes.shape[1])
    inputs = torch.stack([combine(presence(present)[:, None], _pairs(codes))[0] for present in cases])
    target = torch.tensor(list(cases.values()))
    optimizer = torch.optim.Adam(sigma.parameters(), lr=.05)
    for _ in range(steps):
        optimizer.zero_grad()
        ((sigma.concepts(inputs, [0])[:, 0] - target) ** 2).mean().backward()
        optimizer.step()
        if project:
            with torch.no_grad():
                sigma.weight.clamp_(min=0)

    def read(present):
        return float(sigma.concepts(combine(presence(present)[:, None], _pairs(codes)), [0]))
    return sigma, read, base


def _with_rows(supports, conjunctions, width=24):
    from BasicPercepts import BasicPercepts
    codes = _codes(*supports, width=width)
    rows = [()] * len(supports) + [tuple(c) for c in conjunctions]
    joined = [codes[list(c)].amax(0) + BasicPercepts.INCREMENT * torch.rand(
        width, generator=torch.Generator().manual_seed(len(supports) + i)) for i, c in enumerate(conjunctions)]
    return rows, torch.cat((codes, torch.stack(joined).clamp(max=1.))) if joined else codes


def test_with_no_omission_there_is_no_not_entry_and_nothing_is_against():
    """Certificate 23 (a)."""
    rows, codes = _with_rows(((0, 1, 2), (3, 4, 5), (6, 7, 8)), ((0, 1),))
    seed = _drawn(1)[0]
    # What came is one; nothing expected failed to come.
    sigma, read, _ = _taught({(0,): 1., (1,): 1., (0, 1, 3): 1.}, rows=rows, codes=codes, seed=seed)
    assert int((sigma.weight < 0).sum()) == 0
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
    rows, codes = _with_rows(((0, 1, 2), (3, 4, 5), (6, 7, 8)), ((0, 1), (0, 2)))
    A, B, X, AB, AX = range(5)
    patterning = inhibition = 0
    report = []
    for seed in seeds:
        _, read, _ = _taught({(A,): 1., (B,): 1., (A, B, AB): 0.}, rows=rows, codes=codes, seed=seed)
        one = dict(A=read((A,)), B=read((B,)), AB=read((A, B, AB)), nothing=read(()))
        patterning += one['A'] > .5 and one['B'] > .5 and one['AB'] < .5
        sigma, read, _ = _taught({(A,): 1., (A, X, AX): 0., (B,): 1.}, rows=rows, codes=codes, seed=seed)
        two = dict(A=read((A,)), AX=read((A, X, AX)), X=read((X,)), not_entries=int((sigma.weight < -.01).sum()))
        inhibition += two['A'] > .5 and two['AX'] < .5 and two['X'] < 0
        report.append(dict(seed=seed, patterning=one, inhibition=two))
    print(report)
    assert patterning == len(seeds) and inhibition == len(seeds)


def test_the_basic_level_learns_xor_of_two_perceived_words_with_no_not_entry():
    """Certificate 23 (c): true and false each perceived at two places, a
    row for each pair; the weights held at zero or above. The Sigma starts at
    zero, so what is learned does not depend on a draw: one run."""
    seeds = _drawn(1)
    words = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))  # true1, false1, true2, false2
    pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
    rows, codes = _with_rows(words, pairs)
    cases = {(0, 2, 4): 0., (0, 3, 5): 1., (1, 2, 6): 1., (1, 3, 7): 0.}
    solved, report = 0, []
    for seed in seeds:
        sigma, read, _ = _taught(cases, rows=rows, codes=codes, project=True, seed=seed)
        readings = {present: read(present) for present in cases}
        ok = all((value > .5) == (target > .5) for value, target in
                 zip(readings.values(), cases.values()))
        solved += ok
        report.append(dict(seed=seed, readings=list(readings.values()),
                           not_entries=int((sigma.weight < 0).sum())))
    print(report)
    assert solved == len(seeds)


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
