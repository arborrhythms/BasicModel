"""Item 1.4, the unit delay on percepts (Alec, 2026-11-27; plan §5.8, §9).

"We can just have a unit delay on percepts (i.e. basic percepts and
symbols) where we store their activations, so that the delayed versions
are available for conceptual input at subsequent orders." Each basic
percept's activation one step back is an input of the order-zero Sigma
beside the row itself. Nothing is subtracted: a change is learned on the
two inputs by the ordinary Sigma.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import math

import pytest
import torch

PIXEL, OTHER = [3, 2, 0, 50], [5, 5, 0, 200]


def test_gained_lost_and_kept_are_each_learned_on_a_row_and_its_delayed_input():
    """One row now (input 0) and one step back (input 1). Three concepts
    are each given the two entries, the delayed one at zero as MTM writes
    it, and taught one of: gained (now and not before), lost (before and
    not now), kept (both). Each learns its own sign pattern, and reads its
    own case above the other three."""
    from Layers import NotLayer3, SparseSigmaLDU
    cases = torch.tensor([[1., 0., 1., 0.],      # now:    gained, lost, kept, neither
                          [0., 1., 1., 0.]])     # before
    lanes = torch.stack((cases, torch.zeros_like(cases)), -1)          # [inputs, cases, 2]
    targets = dict(gained=[1., -1., 0., 0.], lost=[-1., 1., 0., 0.], kept=[.5, .5, 1., 0.])
    learned = {}
    for name, target in targets.items():
        sigma, not_ = SparseSigmaLDU(2, 1, 4), NotLayer3(2)
        own, delayed = sigma.add(0, 0), sigma.add(0, 1, 0.)
        optimizer = torch.optim.SGD([sigma.value], lr=.05)
        for _ in range(600):
            optimizer.zero_grad()
            read = sigma(not_(lanes), [0, 1], [0])[0]
            (torch.atanh(read.clamp(-.999, .999)) - torch.tensor(target)).square().sum().backward()
            optimizer.step()
        weights = sigma.w().detach()
        read = sigma(not_(lanes), [0, 1], [0])[0].detach()
        learned[name] = (float(weights[own]), float(weights[delayed]), read.tolist())
    print(learned)
    assert learned['gained'][0] > .5 and learned['gained'][1] < -.5
    assert learned['lost'][0] < -.5 and learned['lost'][1] > .5
    assert learned['kept'][0] > .3 and learned['kept'][1] > .3
    for index, name in enumerate(targets):
        read = learned[name][2]
        assert read[index] == max(read) and read[3] == 0.


@pytest.fixture(scope="module")
def delayed(tmp_path_factory):
    """The naming model of item 1.6 with ``perceptDelay`` true: a pixel
    known alone, then three steps: the pixel, another pixel, nothing of
    either."""
    from test_item16_naming import naming_model, value_of
    from test_item16_scene_field import known, present
    from reading_fixtures import use_eager_reading
    import test_item17_echo as echo
    record = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        built = echo.echo_model

        def configured(tmp_path, patch, *, architecture_overrides=None):
            return built(tmp_path, patch, architecture_overrides=dict(architecture_overrides or {}, perceptDelay='true'))
        monkeypatch.setattr(echo, 'echo_model', configured)
        model = naming_model(tmp_path_factory.mktemp("delay"), monkeypatch)
        use_eager_reading(monkeypatch)
        owner = model._concept_owner()
        basic = owner.basic_percepts
        try:
            for pixel in (PIXEL, OTHER):
                known(model, [dict(text='', scene=[pixel])])
            sigma = owner.order_zero
            concepts = {name: value_of(model, pixel[3]) for name, pixel in (('pixel', PIXEL), ('other', OTHER))}
            rows = {}
            for name, concept in concepts.items():
                row = owner._concept_row_of(concept)
                percept, = [p for p in owner.concept_percepts(row) if basic.is_value(p)]
                rows[name] = (row, percept)
            record.update(on=owner.percept_delay, inputs=sigma.nInput, capacity=int(basic.capacity), rows=rows,
                          links={name: (None if sigma.find(row, owner.delayed_input(percept)) is None else
                                        float(sigma.w()[sigma.find(row, owner.delayed_input(percept))]))
                                 for name, (row, percept) in rows.items()})

            def before(name):
                return float(owner._percept_delay_previous[rows[name][1], 0, 0])
            steps = []
            for scene in ([PIXEL], [OTHER], [OTHER], [PIXEL]):
                present(model, [dict(text='', scene=scene)], train=False)
                steps.append(dict(pixel=before('pixel'), other=before('other')))
            record['steps'] = steps
        finally:
            model.End()
    return record


def test_the_order_zero_sigma_has_an_input_for_each_row_one_step_back(delayed):
    print({k: delayed[k] for k in ('on', 'inputs', 'capacity', 'rows', 'links')})
    assert delayed['on'] and delayed['inputs'] == 2 * delayed['capacity']
    # MTM's link from each concept to its own row one step back, at zero.
    assert delayed['links'] == dict(pixel=0., other=0.)


def test_what_was_found_one_step_back_is_given_at_the_next_step(delayed):
    """Four steps: the pixel, the other, the other, the pixel. At each,
    what is one step back is what the step before held, and nothing older."""
    steps = delayed['steps']
    print(steps)
    assert steps[1]['pixel'] > 0 and steps[1]['other'] == 0.
    assert steps[2]['pixel'] == 0. and steps[2]['other'] > 0
    assert steps[3]['pixel'] == 0. and steps[3]['other'] > 0


def test_with_the_key_off_the_sigma_is_as_it_was(tmp_path, monkeypatch):
    """The key is on by default; set false, the order-zero Sigma has an
    input for each basic percept and no more, as it had before the delay."""
    from test_item16_naming import naming_model
    import test_item17_echo as echo
    built = echo.echo_model

    def configured(tmp_path, patch, *, architecture_overrides=None):
        return built(tmp_path, patch, architecture_overrides=dict(architecture_overrides or {}, perceptDelay='false'))
    monkeypatch.setattr(echo, 'echo_model', configured)
    model = naming_model(tmp_path, monkeypatch)
    try:
        owner = model._concept_owner()
        assert not owner.percept_delay
        assert owner.install_order_zero().nInput == int(owner.basic_percepts.capacity)
    finally:
        model.End()


def _keyed(monkeypatch, on=True):
    """``architecture.perceptDelay`` true (false where not ``on``) for the
    spaces made in a test."""
    import Spaces
    monkeypatch.setattr(Spaces.ConceptualSpace, 'percept_delay', property(lambda self: on))


def test_the_numeric_stem_gives_the_order_zero_sigma_each_row_one_step_back(monkeypatch):
    """The numeric stem reads its rows through the order-zero Sigma itself.
    With the key on it gives it each row one step back as an input as well:
    at each of three steps, the inputs after the rows are the rows' delayed
    inputs, and what they carry is what the step before held, and nothing
    older."""
    _keyed(monkeypatch)
    from test_item16_values import _xor, _recur
    model = _xor()
    owner, basic = _recur(model, torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1))
    model.eval()
    calls, original = [], owner.order_zero_field

    def counted(lanes, inputs, rows):
        calls.append((lanes.detach().clone(), list(inputs)))
        return original(lanes, inputs, rows)
    owner.order_zero_field = counted
    try:
        for image in ([[1.], [0.]], [[0.], [1.]], [[0.], [1.]]):
            model.numeric_percepts.read(torch.tensor([image]))
            owner.step_percept_delay()
    finally:
        del owner.order_zero_field
    capacity = int(basic.capacity)
    measured = []
    for lanes, inputs in calls:
        P = len(inputs) // 2
        measured.append(dict(rows=inputs[:P], delayed=inputs[P:],
                             now=lanes[:P].amax((1, 2))[:, 0].tolist(),
                             before=lanes[P:].amax((1, 2))[:, 0].tolist()))
    print(dict(capacity=capacity, steps=measured))
    assert len(measured) == 3 and all(step['rows'] for step in measured)
    for step in measured:
        assert step['delayed'] == [capacity + row for row in step['rows']]
    assert measured[0]['before'] == [0.] * len(measured[0]['rows'])
    for previous, step in zip(measured, measured[1:]):
        held = dict(zip(previous['rows'], previous['now']))
        assert step['before'] == pytest.approx([held.get(row, 0.) for row in step['rows']])
    assert measured[1]['before'] != measured[2]['before']


def test_the_concept_store_has_a_column_for_each_concept_one_step_back(monkeypatch):
    """With the key on, the store's columns are its symbols, the bias and
    one column for each concept row one step back, keyed apart from the
    symbols; with it off they are the symbols and the bias, as they were.
    The key is on by default, so it is set false for the space made off."""
    import Spaces
    from test_cs_sparse_weights import _cs
    _keyed(monkeypatch, on=False)
    off = Spaces._concept_alloc_of(_cs(nS=64, order=2)).layer(0)
    _keyed(monkeypatch)
    on = Spaces._concept_alloc_of(_cs(nS=64, order=2)).layer(0)
    shapes = dict(off=(off.nInput, off.nOutput, off._room), on=(on.nInput, on.nOutput, on._room),
                  column_of_row_5=on.delayed_column(5), thing=on.thing_of_column(on.delayed_column(5)))
    print(shapes)
    assert off.nInput == off._room + 1 and not off.__dict__.get('_delayed_columns')
    assert on.nInput == on._room + 1 + on.nOutput
    assert shapes['thing'] == ('delayed', 5) and on.thing_rows(shapes['thing']) == ()
    assert on.thing_of_column(on.bias_column()) is None and on.thing_of_column(3) == ('sym', 3)


def _two_orders(monkeypatch):
    """A space of two orders with the key on: concepts ``x`` and ``y`` of
    order zero, and the kinds ``gained`` (``x`` now, not before) and
    ``before`` (``x`` one step back) of order one."""
    import Spaces
    from symbol_fixtures import readable
    from test_cs_sparse_weights import _cs, _mint_row
    _keyed(monkeypatch)
    cs = _cs(nS=64, order=2)
    x, y = _mint_row(cs, 0, 9001), _mint_row(cs, 0, 9002)
    gained, before = _mint_row(cs, 1, 9101), _mint_row(cs, 1, 9102)
    readable(cs, x, y)
    return cs, Spaces._concept_alloc_of(cs).layer(0), dict(x=x, y=y, gained=gained, before=before)


class _Clock:
    """A model's clock, as :meth:`delayed_percepts` reads it."""
    def __init__(self):
        self.tick = 0

    def present(self):
        return self.tick


def test_a_kind_reads_a_concept_one_step_back_and_the_step_is_the_percepts(monkeypatch):
    """``gained`` reads ``x`` now and ``x`` one step back below zero (a Not
    entry); ``before`` reads ``x`` one step back. Four steps: ``x``, ``x``,
    nothing, ``x``. Each step is a tick of the clock at which percepts are
    read, so the concepts' store steps where the percepts' does. Nothing is
    subtracted: "now and not before" is a sum of a row's option and its
    delayed input's not-value in the ordinary Sigma."""
    from FieldProjection import close
    cs, store, rows = _two_orders(monkeypatch)
    x, gained, before = rows['x'], rows['gained'], rows['before']
    cs.add_concept_edge(gained, x, weight=1.)
    cs.link_delayed_concept(gained, x, weight=1., negated=True)
    cs.link_delayed_concept(before, x, weight=1.)
    clock = _Clock()
    object.__setattr__(cs, '_model', clock)
    n0, read, closed = cs._order_caps()[0], [], []
    no_percepts = torch.zeros(0, 1, 1, 2)
    for present in (True, True, False, True):
        clock.tick += 1
        cs.delayed_percepts(no_percepts, [])          # the percepts of the step are read
        a0 = torch.zeros(n0, 1, 1, 2)
        a0[x, 0, 0, 0] = .9 if present else 0.
        field = cs.cs_forward_content(a0, publish=True)
        read.append(dict(gained=round(float(field[gained, 0, 0, 0]), 4),
                         before=round(float(field[before, 0, 0, 0]), 4)))
        # The numeric stem's reading of the orders above zero, by close.
        given = torch.zeros(3, 1, 1, 2)
        given[0] = a0[x]
        evidence = close(cs, [x, gained, before], given)
        closed.append(round(float(evidence[2, 0, 0, 0]), 4))
    print(dict(read=read, close_before=closed, delayed_entries=[t.tolist() for t in store.delayed_entries('cpu')]))
    now = round(math.tanh(.9), 4)
    assert read == [dict(gained=now, before=0.), dict(gained=0., before=now),
                    dict(gained=0., before=now), dict(gained=now, before=0.)]
    assert closed == [0., now, now, 0.]


def test_an_entry_on_a_concept_one_step_back_is_learned_by_the_sigma(monkeypatch):
    """A kind given an entry on ``x`` and one on ``x`` one step back, both at
    zero as MTM writes a link, is taught "x now and not before": the
    gradient reaches the delayed entry through the ordinary Sigma, and it
    learns an entry above zero on the row and one below zero on its delayed
    input."""
    cs, store, rows = _two_orders(monkeypatch)
    x, gained = rows['x'], rows['gained']
    own = cs.add_concept_edge(gained, x, weight=0.)
    delayed = cs.link_delayed_concept(gained, x, weight=0.)
    n0 = cs._order_caps()[0]

    def field(now, earlier):
        for value, publish in ((earlier, True), (now, False)):
            a0 = torch.zeros(n0, 1, 1, 2)
            a0[x, 0, 0, 0] = value
            result = cs.cs_forward_content(a0, publish=publish)
            if publish:
                cs.step_percept_delay()
        return result[gained, 0, 0, 0]
    cases = dict(gained=(.9, 0., 1.), lost=(0., .9, 0.), kept=(.9, .9, 0.), neither=(0., 0., 0.))
    optimizer = torch.optim.SGD([store.values], lr=.2)
    first = None
    for _ in range(150):
        optimizer.zero_grad()
        loss = sum((field(now, earlier) - target) ** 2 for now, earlier, target in cases.values())
        loss.backward()
        if first is None:
            first = float(store.values.grad[delayed])
        optimizer.step()
    weights = store.values.detach()
    learned = dict(own=float(weights[own]), delayed=float(weights[delayed]), first_delayed_grad=first,
                   reads={name: round(float(field(now, earlier)), 3) for name, (now, earlier, _) in cases.items()})
    print(learned)
    assert learned['own'] > .5 and learned['delayed'] < -.5
    assert learned['reads']['gained'] == max(learned['reads'].values())


def test_a_hard_reset_clears_what_is_one_step_back(tmp_path, monkeypatch):
    """Alec, 2026-11-27: "make sure that the delayed conceptual state is
    cleared on a reset()". After a step that found a pixel, a hard reset of
    the conceptual space leaves nothing kept, of the basic percepts or of
    the concepts, now or one step back; a soft reset leaves it."""
    from test_item16_naming import naming_model
    from test_item16_scene_field import known, present
    from reading_fixtures import use_eager_reading
    model = naming_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    owner = model._concept_owner()
    try:
        assert owner.percept_delay
        known(model, [dict(text='', scene=[PIXEL])])
        present(model, [dict(text='', scene=[PIXEL])], train=False)

        def kept():
            return [float(held.abs().sum()) for kind in owner._DELAY_BUFFERS
                    for held in (owner._delay_state(kind) or ())]
        before = kept()
        owner.Reset(hard=False)
        soft = kept()
        owner.Reset(hard=True)
        hard = kept()
    finally:
        model.End()
    print(dict(before=before, soft=soft, hard=hard))
    assert any(value > 0 for value in before) and soft == before
    assert hard and all(value == 0. for value in hard)
