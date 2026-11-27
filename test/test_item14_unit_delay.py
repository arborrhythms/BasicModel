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
    read, and each read of the concepts is an opening, so the concepts'
    store steps where the percepts' does: openings are compared with
    openings (plan §9.5). Nothing is
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
    the concepts, now or one step back, nor the concepts' field at the last
    opening or the trace; a soft reset leaves it."""
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
            # With the concepts' field at the last opening and the trace
            # of the looks (plan §9.5).
            return [*(float(held.abs().sum()) for kind in owner._DELAY_BUFFERS
                      for held in (owner._delay_state(kind) or ())),
                    *(float(getattr(owner, name).abs().sum()) for name in getattr(owner, '_DELAY_KEPT', ())
                      if getattr(owner, name, None) is not None)]
        before = kept()
        owner.Reset(hard=False)
        soft = kept()
        owner.Reset(hard=True)
        hard = kept()
    finally:
        model.End()
    print(dict(before=before, soft=soft, hard=hard))
    assert any(value > 0 for value in before) and soft == before
    assert len(hard) == 6 and all(value == 0. for value in hard)


# The unit delay steps once a look (Alec, 2026-11-27; plan §9.5): "the
# opening is look zero", openings are compared with openings, and the
# request of ``attend`` reads the field the looks have primed (the trace)
# in place of its ``memory`` input.

def _looks_recorded(monkeypatch, model):
    """Each path's looks, as they run: the field kept at the sentence's
    opening, and at each look what the unit delay has one step back as the
    look reads, what the request is given in place of ``memory``, and the
    field observed at the reading the look took."""
    import SentenceLooks
    from RequestRead import RequestRead, at_rows
    from SentenceField import inventory
    owner = model._concept_owner()
    paths = []
    run, look, given = SentenceLooks.Looks.run, SentenceLooks.look, RequestRead.given
    interpret = SentenceLooks.Looks._look_interpret_and_spread

    def kept(name):
        held = getattr(owner, name, None)
        return None if held is None else held.transpose(0, 1).clone()

    def recorded_run(self, arrive, **kwargs):
        paths.append(dict(path=self, opening=kept('_concept_delay_opening'), carried=kept('_concept_trace'),
                          previous=[], given=[], observed=[], made=[], asked=[]))
        return run(self, arrive, **kwargs)

    def recorded_look(model_, bank, request, own):
        if paths:
            paths[-1]['previous'].append(kept('_concept_delay_previous'))
        return look(model_, bank, request, own)

    def recorded_given(self, opened, memory, found):
        if paths:
            # The trace one look back, made here from what was recorded,
            # on the structure as it stands at this look.
            from RequestRead import prime
            path = paths[-1]
            asked = None
            if not path['given']:
                observed, before = path['opening'].clamp(0., 1.), path['carried']
            else:
                k = len(path['given']) - 1
                observed, before, asked = path['observed'][k], path['given'][-1], path['asked'][k]
            path['made'].append(prime(model, owner, observed, before, asked=asked))
            path['given'].append(memory.detach().clone())
        return given(self, opened, memory, found)

    def recorded_interpret(self, model_, owner_, bank, state, observation, look_field, extent,
                           words, rows, index, *args, **kwargs):
        from Salience import concepts_asked
        paths[-1]['observed'].append(at_rows(inventory(owner_, look_field), look_field,
                                             int(owner_.nVectors), len(rows), index))
        request = args[2]
        paths[-1]['asked'].append(concepts_asked(owner_, request[:model_.request_read.rows].detach()).transpose(0, 1))
        return interpret(self, model_, owner_, bank, state, observation, look_field, extent,
                         words, rows, index, *args, **kwargs)
    monkeypatch.setattr(SentenceLooks.Looks, 'run', recorded_run)
    monkeypatch.setattr(SentenceLooks, 'look', recorded_look)
    monkeypatch.setattr(RequestRead, 'given', recorded_given)
    monkeypatch.setattr(SentenceLooks.Looks, '_look_interpret_and_spread', recorded_interpret)
    return paths


def _lesson(tmp_path, monkeypatch):
    from test_item17_looks import nominated_lesson
    model = nominated_lesson(tmp_path, monkeypatch)
    model._compiled_sentence_regions = None
    return model


def _same(a, b):
    return a is not None and b is not None and a.shape == b.shape and torch.allclose(a, b, atol=1e-6)


def test_at_each_look_one_step_back_is_the_look_before_and_the_request_reads_the_trace(tmp_path, monkeypatch):
    """(a) Over the looks of a sentence of four words: what the unit delay
    has one step back as look k reads is the field observed at look k-1,
    and at the first look the field kept at the opening (look zero). What
    the request is given in place of ``memory`` is the trace one look back:
    at the first look what the opening's cycle left from the trace the
    sentence before left, and at look k what the cycle of look k-1 left:
    the trace before it decayed, the field observed at look k-1 risen
    through the structure and met by expectation, and descended masked by
    what ``attend`` asked at look k-1 (:func:`RequestRead.prime`); so it
    holds what look k-1 observed. (The request reads once more after the
    last look, to stop.)"""
    from test_item18_expect import read
    model = _lesson(tmp_path, monkeypatch)
    owner = model._concept_owner()
    try:
        assert owner.percept_delay
        paths = _looks_recorded(monkeypatch, model)
        read(model, 'delta gamma beta alpha')
        path = paths[0]
        looks = len(path['observed'])
        measured = dict(looks=looks, strength=model.word_expectation_gain,
                        opening=None if path['opening'] is None else float(path['opening'].sum()),
                        observed=[float(o.sum()) for o in path['observed']],
                        previous=[None if p is None else float(p.sum()) for p in path['previous']],
                        given=[float(g.sum()) for g in path['given']])
        print(measured)
        assert looks >= 2 and path['opening'] is not None and float(path['opening'].sum()) > 0
        opening = path['opening'].clamp(0., 1.)
        assert _same(path['previous'][0], opening)
        for k in range(1, looks):
            assert _same(path['previous'][k], path['observed'][k - 1]), k
        # The trace the sentence before left carries on into this opening.
        for k in range(looks):
            assert _same(path['given'][k], path['made'][k]), k
        for k in range(1, looks):
            assert bool((path['given'][k] >= path['observed'][k - 1] * (1 - model.word_expectation_gain) - 1e-6).all())
    finally:
        model.End()


def test_a_concept_two_steps_away_is_in_the_trace_after_two_looks():
    """(a) A chain of definitions two steps long: ``A`` holds ``x`` and
    ``y``, ``B`` holds ``y`` and ``z``. ``x`` is observed at every look,
    attention fully open. After one look's cycle the trace holds what the
    traversal from ``x`` reaches (``A`` up, ``y`` down) and nothing of ``B``
    or ``z``; after two it holds ``B`` and ``z`` above nothing, since the
    delayed set rises with the next look's field. With the decay at full
    loss (``primingDecay`` zero) nothing is delayed and ``B`` is never
    reached. At the strength of the field reads as configured and at one."""
    from types import SimpleNamespace
    from symbol_fixtures import readable
    from test_cs_sparse_weights import _cs, _mint_row
    from RequestRead import prime
    from util import TheXMLConfig
    cs = _cs(nS=64, order=2)
    x, y, z = (_mint_row(cs, 0, 9001 + i) for i in range(3))
    A, B = _mint_row(cs, 1, 9101), _mint_row(cs, 1, 9102)
    readable(cs, x, y, z)
    for kind, parts in ((A, (x, y)), (B, (y, z))):
        for part in parts:
            cs.add_concept_edge(kind, part, weight=1.)
    observed = torch.zeros(1, int(cs.nVectors), 2)
    observed[0, x, 0] = 1.
    configured = float(TheXMLConfig.training('symbolExpectationGain', 0.))
    measured = {}
    for decay in (.9, 0.):
        object.__setattr__(cs, '_priming_decay', decay)
        for strength in sorted({configured, 1.}):
            model = SimpleNamespace(word_expectation_gain=strength)
            trace, traces = None, []
            for _ in range(3):
                trace = prime(model, cs, observed, trace)
                traces.append(trace)
            measured[decay, strength] = [{name: round(float(t[0, row, 0]), 4) for name, row in
                                          dict(x=x, y=y, z=z, A=A, B=B).items()} for t in traces]
    print(measured)
    for (decay, _), (one, two, three) in measured.items():
        assert one['x'] == 1. and one['A'] > 0 and one['y'] > 0 and one['B'] == 0. and one['z'] == 0.
        if decay:
            assert two['B'] > 0 and two['z'] > 0
        else:
            assert two == one and three == one


def test_the_cycle_with_nothing_delayed_and_attention_open_is_the_field_spread_once():
    """The limiting case: with nothing delayed, the strength of the field
    reads at zero and attention fully open, the cycle of a look leaves the
    field spread once, what the request read of the last look before the
    unit delay (:func:`RequestRead.spread_once`)."""
    from types import SimpleNamespace
    from symbol_fixtures import readable
    from test_cs_sparse_weights import _cs, _mint_row
    from RequestRead import prime, spread_once
    cs = _cs(nS=64, order=2)
    x, y, z = (_mint_row(cs, 0, 9001 + i) for i in range(3))
    A, B = _mint_row(cs, 1, 9101), _mint_row(cs, 1, 9102)
    readable(cs, x, y, z)
    for kind, parts in ((A, (x, y)), (B, (y, z))):
        for part in parts:
            cs.add_concept_edge(kind, part, weight=1.)
    observed = torch.zeros(2, int(cs.nVectors), 2)
    observed[0, x, 0], observed[1, z, 0], observed[1, y, 1] = 1., .7, .4
    cycle = prime(SimpleNamespace(word_expectation_gain=0.), cs, observed)
    once = spread_once(cs, observed)
    print(dict(largest_difference=float((cycle - once).abs().max()), total=float(once.sum())))
    assert float(once.sum()) > float(observed.sum()) and torch.equal(cycle, once)


def test_what_attend_predicted_masks_the_way_down():
    """(a) ``attend``'s prediction is a mask on the way down (Alec,
    2026-11-27: "modified by attend() on the way down (unless attention is
    fully open, since attention is a mask)"). ``A`` holds ``x`` and ``y``;
    ``x`` is observed. Attention fully open, ``y`` is reached on the way
    down; asked for at nothing, it is not; asked for at one half, it is
    reached at half or less of what open attention gives. What rose (``x``
    and ``A``) is not masked."""
    from types import SimpleNamespace
    from symbol_fixtures import readable
    from test_cs_sparse_weights import _cs, _mint_row
    from RequestRead import prime
    cs = _cs(nS=64, order=2)
    x, y = _mint_row(cs, 0, 9001), _mint_row(cs, 0, 9002)
    A = _mint_row(cs, 1, 9101)
    readable(cs, x, y)
    for part in (x, y):
        cs.add_concept_edge(A, part, weight=1.)
    V = int(cs.nVectors)
    model = SimpleNamespace(word_expectation_gain=0.)
    observed = torch.zeros(1, V, 2)
    observed[0, x, 0] = 1.
    measured = {}
    for name, value in (('open', None), ('nothing', 0.), ('half', .5)):
        asked = None if value is None else torch.full((1, V, 2), value)
        trace = prime(model, cs, observed, None, asked=asked)
        measured[name] = {key: round(float(trace[0, row, 0]), 4) for key, row in dict(x=x, y=y, A=A).items()}
    print(measured)
    assert measured['open']['y'] > 0 and measured['nothing']['y'] == 0.
    assert 0 < measured['half']['y'] <= measured['open']['y'] / 2 + 1e-6
    for name in measured:
        assert measured[name]['x'] == 1. and measured[name]['A'] == measured['open']['A'] > 0


def test_openings_are_compared_with_openings(tmp_path, monkeypatch):
    """(b) Two sentences, each of two words. At the second sentence's
    opening what is one step back is the first sentence's opening field,
    not its last look's; at the second sentence's first look it is the
    second opening's field."""
    from test_item18_expect import read
    model = _lesson(tmp_path, monkeypatch)
    owner = model._concept_owner()
    at_openings, forward = [], owner.cs_forward_content

    def recorded(a_0, *, publish=False):
        result = forward(a_0, publish=publish)
        if publish:
            at_openings.append(dict(previous=owner._concept_delay_previous.transpose(0, 1).clone(),
                                    current=owner._concept_delay_current.transpose(0, 1).clone()))
        return result
    object.__setattr__(owner, 'cs_forward_content', recorded)
    try:
        paths = _looks_recorded(monkeypatch, model)
        read(model, 'alpha gamma')
        last_look = owner._concept_delay_current.transpose(0, 1).clone()
        read(model, 'beta delta')
        first, second = at_openings[0]['current'], at_openings[-1]['current']
        measured = dict(openings=len(at_openings), first=float(first.sum()), second=float(second.sum()),
                        last_look=float(last_look.sum()),
                        previous_at_second=float(at_openings[-1]['previous'].sum()),
                        first_look_previous=[None if p is None else float(p.sum()) for p in paths[-1]['previous'][:1]])
        print(measured)
        assert len(at_openings) == 2 and float(first.sum()) > 0
        assert not _same(first, last_look)
        assert _same(at_openings[-1]['previous'], first)
        assert _same(paths[-1]['previous'][0], second.clamp(0., 1.))
    finally:
        vars(owner).pop('cs_forward_content', None)
        model.End()


def test_a_discarded_path_leaves_nothing_in_what_the_kept_path_has_one_step_back(tmp_path, monkeypatch):
    """(c) A sentence read in training, by the greedy path and an
    exploration. Each path's first look has the opening one step back and
    reads the opening's trace, whatever the path before it wrote; at the
    sentence's end the unit delay holds the winning path's last look for
    each stream. The same choice made by hand over two paths with
    different kept fields writes the chosen one's and nothing of the
    other's."""
    from types import SimpleNamespace
    from test_item18_expect import read
    import SentenceLooks
    model = _lesson(tmp_path, monkeypatch)
    owner = model._concept_owner()
    kept, keep = [], SentenceLooks.Looks.keep

    def recorded_keep(paths, winners):
        kept.append(winners.clone())
        return keep(paths, winners)
    monkeypatch.setattr(SentenceLooks.Looks, 'keep', staticmethod(recorded_keep))
    try:
        paths = _looks_recorded(monkeypatch, model)
        model.train()
        read(model, 'delta gamma beta alpha', training=True)
        measured = dict(paths=len(paths), winners=[w.tolist() for w in kept],
                        first_previous=[float(p['previous'][0].sum()) if p['previous'] and p['previous'][0] is not None
                                        else None for p in paths],
                        opening=[None if p['opening'] is None else float(p['opening'].sum()) for p in paths])
        print(measured)
        assert len(paths) >= 2 and kept
        for path in paths:
            if path['previous']:
                assert _same(path['previous'][0], path['opening'].clamp(0., 1.))
                assert _same(path['given'][0], paths[0]['given'][0])
        winner = paths[int(kept[-1][0])]['path']
        held = owner._concept_trace.transpose(0, 1)
        assert winner.final is not None and _same(held, winner.final[2])
        assert _same(owner._concept_delay_current.transpose(0, 1), winner.final[1])
    finally:
        model.End()
    # By hand: two paths whose kept fields differ, stream 0 keeps path 1
    # and stream 1 keeps path 0.
    V, B = 5, 2
    owner = SimpleNamespace(**{name: torch.zeros(V, B, 2) for name in (
        '_concept_delay_previous', '_concept_delay_current', '_concept_trace')})
    finals = [tuple(torch.full((B, V, 2), float(10 * p + part)) for part in range(3)) for p in range(2)]
    paths = [SimpleNamespace(final=final, model=SimpleNamespace(_concept_owner=lambda: owner)) for final in finals]
    keep(paths, torch.tensor([1, 0]))
    written = [owner._concept_delay_previous[0, :, 0].tolist(), owner._concept_delay_current[0, :, 0].tolist(),
               owner._concept_trace[0, :, 0].tolist()]
    print(written)
    assert written == [[10., 0.], [11., 1.], [12., 2.]]


def test_a_hard_reset_clears_the_opening_and_the_trace(tmp_path, monkeypatch):
    """(d) After a sentence read by looks, the concepts' field kept at the
    opening and the trace are held; a soft reset leaves them, a hard reset
    clears them with the rest of the unit delay."""
    from test_item18_expect import read
    model = _lesson(tmp_path, monkeypatch)
    owner = model._concept_owner()
    try:
        read(model, 'delta gamma beta alpha')
        names = [*(name for kind in owner._DELAY_BUFFERS for name in owner._DELAY_BUFFERS[kind]),
                 *getattr(owner, '_DELAY_KEPT', ())]

        def kept():
            return {name: float(getattr(owner, name).abs().sum()) for name in names if getattr(owner, name, None) is not None}
        before = kept()
        owner.Reset(hard=False)
        soft = kept()
        owner.Reset(hard=True)
        hard = kept()
    finally:
        model.End()
    print(dict(before=before, soft=soft, hard=hard))
    assert before.get('_concept_delay_opening', 0) > 0 and before.get('_concept_trace', 0) > 0
    assert soft == before
    assert len(hard) == 6 and all(value == 0. for value in hard.values())


def _delay_held(owner):
    """Each store of the unit delay, by name, as it is kept now (``None``
    where it is not made), and the tick of the percepts' step."""
    names = [*(name for kind in owner._DELAY_BUFFERS for name in owner._DELAY_BUFFERS[kind]),
             *owner._DELAY_KEPT, '_percept_delay_tick']
    return {name: None if getattr(owner, name, None) is None else getattr(owner, name).detach().clone()
            for name in names}


def test_the_delay_is_saved_and_restored_with_a_checkpoint(tmp_path, monkeypatch):
    """Alec, 2026-11-27: "Delayed conceptual state should be preserved ...
    because it is still conceptual state." After a step that found a pixel
    the model's state is saved and loaded into a fresh model at the same
    number of streams: what is one step back, of the basic percepts and of
    the concepts, the present step's, the field at the last opening, the
    trace and the tick are each equal. A store later made for another
    number of streams starts empty. A checkpoint saved without the delay
    starts it empty."""
    from test_item16_naming import naming_model
    from test_item16_scene_field import known, present
    from reading_fixtures import use_eager_reading
    import Models
    load = Models.BaseModel.load_weights  # the construction stubs the autoload
    model = naming_model(_made(tmp_path / 'source'), monkeypatch)
    use_eager_reading(monkeypatch)
    owner = model._concept_owner()
    path = tmp_path / 'delay.ckpt'
    try:
        assert owner.percept_delay
        known(model, [dict(text='', scene=[PIXEL])])
        present(model, [dict(text='', scene=[PIXEL])], train=False)
        saved = _delay_held(owner)
        model.save_weights(path)
    finally:
        model.End()
    fresh = naming_model(_made(tmp_path / 'target'), monkeypatch)
    target = fresh._concept_owner()
    try:
        empty = _delay_held(target)
        assert load(fresh, path, require_match=True)
        restored = _delay_held(target)
        assert all(value is not None for value in restored.values()), {k: v is None for k, v in restored.items()}
        old = torch.load(path, map_location='cpu', weights_only=False)
        for key in [key for key in old['state_dict'] if key.rsplit('.', 1)[-1].startswith('unit_delay_')]:
            del old['state_dict'][key]
        old_path = tmp_path / 'before.ckpt'
        torch.save(old, old_path)
        # Over the restored delay: a checkpoint without it leaves it empty.
        assert load(fresh, old_path)
        before = _delay_held(target)
        assert load(fresh, path, require_match=True)
        B = int(restored['_concept_delay_current'].shape[1])
        device = restored['_concept_delay_current'].device
        target.prepare_delay(B + 1, device)
        other = _delay_held(target)
    finally:
        fresh.End()

    def sums(held):
        return {name: None if value is None else (tuple(value.shape), float(value.abs().sum()))
                for name, value in held.items()}
    print(dict(saved=sums(saved), empty=sums(empty), restored=sums(restored), other=sums(other),
               before=sums(before)))
    assert all(value is not None for value in saved.values())
    assert sum(float(value.abs().sum()) for name, value in saved.items() if name != '_percept_delay_tick') > 0
    for name, value in saved.items():
        assert restored[name] is not None and restored[name].shape == value.shape, name
        assert torch.equal(restored[name].cpu(), value.cpu()), name
    for name, value in other.items():
        if name != '_percept_delay_tick':
            assert value.shape[1] == B + 1 and float(value.abs().sum()) == 0., name
    for name, value in before.items():
        if value is not None and name != '_percept_delay_tick':
            assert float(value.abs().sum()) == 0., name


def _made(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
