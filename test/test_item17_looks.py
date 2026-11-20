"""The looks (sigma-pi §21, rules 32 to 41 and 47): certificate 56, the
analysis from above wired over both towers."""
import ast
from pathlib import Path

import pytest
import torch

import Salience
from BasicPercepts import PerceptBank
from ConceptSpread import Structure
from PerceptField import PerceptField


def hand_bank(keys, groups, events, readings_of_events, opposed=None):
    """A bank of one stream: ``events`` ``{key: [(event, for, against)]}``,
    each event in the reading ``readings_of_events[event]``."""
    P, K = len(readings_of_events), max(readings_of_events) + 1
    lanes = torch.zeros(len(keys), 1, K, P, 2)
    for i, key in enumerate(keys):
        for event, for_, against in events.get(key, ()):
            lanes[i, 0, readings_of_events[event], event] = torch.tensor([for_, against])
    spans = torch.tensor([[[p, p + 1] for p in range(P)]])
    where = torch.tensor([[[min(p for p in range(P) if readings_of_events[p] == k),
                            max(p for p in range(P) if readings_of_events[p] == k) + 1] for k in range(K)]])
    native = PerceptField(tuple(keys), lanes.amax(-2), lanes, spans, where, torch.zeros_like(where))
    opposed = tuple(() for _ in groups) if opposed is None else opposed
    return PerceptBank(native, tuple(range(len(groups))), tuple(map(tuple, groups)), opposed, tuple(keys))


def row_values(bank):
    """The rows' two lanes as the way up reads them: the run's presence
    times the product of the wholes; against, the presence times the union
    of the wholes' against lanes (each where the run lies)."""
    order = {key: i for i, key in enumerate(bank.keys)}
    values, events = bank.native.values, bank.native.events
    rows = []
    for group in bank.groups:
        run = events[order[group[0]], ..., 0]
        presence = values[order[group[0]], ..., 0]
        wholes = [(run * events[order[key], ..., 0]).amax(-1) for key in group[1:]]
        against = [(run * events[order[key], ..., 1]).amax(-1) for key in group[1:]]
        for_lane = presence * torch.stack(wholes).prod(0) if wholes else presence
        kept = torch.stack([1 - a for a in against]).prod(0) if against else torch.ones_like(presence)
        rows.append(torch.stack((for_lane, presence * (1 - kept)), -1))
    return type('Read', (), dict(values=torch.stack(rows)))()


def test_56a_a_request_for_a_words_concept_lights_its_parts_and_its_properties(tmp_path, monkeypatch):
    """(a) In a model: the request for a word's concept lands on the events of
    its row's run of parts and on the events that carry its row's properties,
    by the first Pi's entries; the other word's events take nothing."""
    from test_item18_expect import lesson, read
    from nomination_fixtures import readings_before_nomination
    from PrependedSymbols import word_view
    model = lesson(tmp_path, monkeypatch)
    for text in ('alpha gamma', 'beta delta'):
        readings_before_nomination(model, lambda: read(model, text))
        read(model, text)
    read(model, 'alpha gamma')
    owner, bank = model._concept_owner(), model._sentence_bank
    concept, other = word_view(model.inputSpace, 'concept_rows')[0, :2].tolist()
    request = torch.zeros(int(owner.nVectors), 1, 2)
    request[concept, 0, 0] = 1.
    reads = {'host': 0}
    host = {name: getattr(torch.Tensor, name) for name in ('tolist', 'item', '__bool__', '__int__', '__float__')}
    for name, original in host.items():
        def counted(self, *a, _original=original, **k):
            reads['host'] += 1
            return _original(self, *a, **k)
        monkeypatch.setattr(torch.Tensor, name, counted)
    perception = owner.basic_percepts.read_bank(bank)
    reads['host'] = 0
    found = Salience.first_pi(bank, perception, torch.ones(len(bank.rows), *perception.values.shape[1:]))
    first_pi_reads = reads['host']
    reads['host'] = 0
    salient = Salience.salience(owner, bank, request, perception=perception)
    # The structure's edges are read by ConceptSpread.structure, which the
    # spread reads too; printed.
    map_reads = reads['host']
    for name, original in host.items():
        monkeypatch.setattr(torch.Tensor, name, original)
    del found
    spans, token = bank.native.spans[0], bank.token[0]
    word = (token == 0) & (spans[:, 1] - spans[:, 0] == 1)
    beside = (token == 1) & (spans[:, 1] - spans[:, 0] == 1)
    keys = list(dict.fromkeys(bank.keys))
    group = bank.groups[bank.rows.index(next(p for r, p, n in owner._concept_allocator.order_zero_members
                                             if r == concept and not n))]
    members = {key: float(salient.members[keys.index(key), 0, 0, 0]) for key in group}
    print(dict(members=members, word_events=salient.events[0, word, 0].tolist(),
               beside_events=salient.events[0, beside, 0].tolist(), first_pi_host_reads=first_pi_reads,
               map_host_reads=map_reads))
    assert any(key[0] == 'ws' for key in group) and any(key[0] == 'ps' for key in group)
    # Each member of the row, its run of parts and each property, takes a share.
    assert all(value > 0 for value in members.values())
    # The events of the word take it; the other word's events none.
    assert bool((salient.events[0, word, 0] > 0).all())
    assert float(salient.events[0, beside].abs().sum()) == 0.
    # Nothing is below zero, and the bands take nothing (no weight of the
    # way up reads a where wire).
    assert float(salient.events.min()) >= 0. and float(salient.band.abs().sum()) == 0.
    # (c) The first Pi's way down reads nothing on the host.
    assert first_pi_reads == 0


def test_56b_a_concept_of_a_property_alone_lights_the_events_that_carry_it():
    """(b) A row whose run is a property alone: the request lights the events
    that carry the property, at the greatest of each reading, and no others."""
    keys = [('ws', 'red'), ('ps', 'b')]
    bank = hand_bank(keys, [[('ws', 'red')], [('ps', 'b')]],
                     {('ws', 'red'): [(0, 1., 0.), (2, 1., 0.)], ('ps', 'b'): [(1, 1., 0.), (2, 1., 0.)]},
                     [0, 0, 1])
    row_grad = torch.zeros(2, 1, 2, 2)
    row_grad[0, ..., 0] = 1.
    _, events = Salience.first_pi(bank, row_values(bank), row_grad)
    print(dict(events=events[0, :, 0].tolist()))
    assert events[0, :, 0].tolist() == [1., 0., 1.]


def test_56d_a_not_entry_gives_its_share_to_the_rows_against_lane():
    """(d) A requested concept with a Not entry on a row gives its share to
    that row's against lane; with a for entry, to its for lane."""
    for negated, lane in ((True, 1), (False, 0)):
        graph = Structure(torch.tensor([0]), torch.tensor([1]), torch.tensor([.7]), torch.tensor([negated]),
                          1, 0, 1, 0, ())
        request = torch.tensor([[[1., 0.]]])
        grad = Salience._down(graph, request, torch.zeros(2, 1, 1, 2), 1)
        print(dict(negated=negated, row=grad[1, 0, 0].tolist()))
        assert grad[1, 0, 0, lane] == pytest.approx(.7) and grad[1, 0, 0, 1 - lane] == 0.


def test_56e_the_pi_analyses_by_wholes_as_it_synthesizes_by_parts():
    """(e) A row with two properties reads their product, the meeting of the
    two classes, and the map gives the run and each property its share."""
    keys = [('ps', 'a'), ('ws', 'red'), ('ws', 'round')]
    bank = hand_bank(keys, [keys],
                     {('ps', 'a'): [(0, 1., 0.)], ('ws', 'red'): [(0, .8, 0.), (1, 1., 0.)],
                      ('ws', 'round'): [(0, .5, 0.), (2, 1., 0.)]},
                     [0, 0, 0])
    read = row_values(bank)
    assert float(read.values[0, 0, 0, 0]) == pytest.approx(.4)
    row_grad = torch.zeros(1, 1, 1, 2)
    row_grad[..., 0] = 1.
    members, events = Salience.first_pi(bank, read, row_grad)
    print(dict(row=float(read.values[0, 0, 0, 0]), members=members[:, 0, 0, 0].tolist(),
               events=events[0, :, 0].tolist()))
    assert members[:, 0, 0, 0].tolist() == pytest.approx([.4, .4, .4])
    # Read where the run lies: the properties' other events take nothing.
    assert events[0, :, 0].tolist() == pytest.approx([1.2, 0., 0.])


def test_56c_the_map_reads_no_parameter_of_its_own():
    """(c) By the source's names: the map makes no module, parameter or
    optimizer, and reads no tensor on the host (no tolist, item, bool,
    int, float or cpu of a tensor)."""
    source = Path(Salience.__file__).read_text()
    tree = ast.parse(source)
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {'Parameter', 'Module', 'Linear', 'optim', 'tolist', 'item', 'cpu', 'numpy'}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)}
    assert not calls & {'bool', 'int', 'float'}


# --- The look, the request and the reading (rules 34 to 40) ---------------

def nominated_lesson(tmp_path, monkeypatch, texts=('alpha gamma', 'beta delta')):
    from test_item18_expect import lesson, read
    from nomination_fixtures import readings_before_nomination
    model = lesson(tmp_path, monkeypatch)
    for text in texts:
        readings_before_nomination(model, lambda: read(model, text))
        read(model, text)
    return model


def word_rows(model, *words):
    owner = model._concept_owner()
    return [owner._concept_row_of(owner.interpret.word_for_form(word)) for word in words]


def test_37_interpret_takes_the_most_active_and_draws_among_equals():
    """The most active word, exactly; equals are a draw, marked as one, with
    a departure taking another of them; nothing found takes nothing."""
    from Attention import draw_among
    inf = torch.inf
    index, drawn, probability, others = draw_among(torch.tensor([[.2, .7, -inf]]))
    assert index.tolist() == [1] and not drawn.any() and probability.tolist() == [1.]
    tied = torch.tensor([[.5, -inf, .5]])
    taken = {int(draw_among(tied)[0]) for _ in range(64)}
    assert taken == {0, 2}
    index, drawn, probability, others = draw_among(tied)
    assert drawn.all() and probability.tolist() == [.5] and others.tolist() == [1]
    assert draw_among(tied, masked=torch.tensor([0]))[0].tolist() == [2]
    assert draw_among(tied, replay=torch.tensor([0]))[0].tolist() == [0]
    assert draw_among(torch.full((1, 3), -inf))[0].tolist() == [-1]


def test_34_the_where_match_multiplies_a_reading_once_at_its_start():
    """``exp(s - max s)`` of the share on the band of each reading's start,
    weighted by that event's presence; one where nothing of where is asked."""
    from types import SimpleNamespace
    from SentenceLooks import where_match, lower
    bank = SimpleNamespace(token=torch.tensor([[0, 0, 1, 1]]), place=torch.tensor([[0, 1, 3, 4]]))
    activity = torch.ones(1, 4, 2)
    match = where_match(bank, torch.tensor([[2., 9., 0., 9.]]), activity, 2)
    torch.testing.assert_close(match, torch.tensor([[1., torch.exp(torch.tensor(-2.))]]))
    torch.testing.assert_close(where_match(bank, torch.zeros(1, 4), activity, 2), torch.ones(1, 2))
    # The done mark lowers the bank's own activity over the form's extent.
    lowered = lower(activity, bank, torch.tensor([1]), torch.tensor([True]))
    assert lowered[0, :, 0].tolist() == [1., 1., 0., 0.]


def test_31_33d_a_request_returns_what_it_asks_and_words_returns_every_word(tmp_path, monkeypatch):
    """A request for one word's concept returns that word alone; "words"
    returns both, with no reading lowered by the where (certificate 33 (d));
    the request starts at "words" alone (rule 35)."""
    import SentenceLooks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        read(model, 'alpha gamma')
        owner, bank = model._concept_owner(), model._sentence_bank
        perception = owner.basic_percepts.read_bank(bank)
        alpha, gamma = word_rows(model, 'alpha', 'gamma')
        concepts = model._attention_word_concepts[:, :2]
        query = torch.zeros(1, int(model._stm_reducer().d_model))
        words = SentenceLooks.requested(model, query)
        asked = (words[..., 0] > 0).nonzero()[:, 0].tolist()
        assert len(asked) == 1 and not words[..., 1].any()
        found = {}
        for name, request in (('words', words), ('alpha', SentenceLooks.word_request(model, concepts, torch.tensor([0])))):
            field, activity, match = SentenceLooks.look(model, bank, request, bank.activity, perception)
            extent = SentenceLooks.extent_of(bank, activity, 2)
            value = SentenceLooks.word_activity(field, concepts, torch.ones(1, 2, dtype=torch.bool),
                                                extent, bank.activity, bank)
            found[name] = (extent[0].tolist(), torch.isfinite(value[0]).tolist(), match[0, :2].tolist())
        print(found)
        assert found['words'][:2] == ([True, True], [True, True]) and found['words'][2] == [1., 1.]
        assert found['alpha'][:2] == ([True, False], [True, False])
    finally:
        model.End()


def test_38a_every_word_is_read_once(tmp_path, monkeypatch):
    """With interpretBias at one each word of a sentence is read once: none
    twice, none left; the order is the draw's."""
    from SentenceLooks import Looks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    run, seen = Looks.run, []
    def observed(self, arrive, **kwargs):
        result = run(self, arrive, **kwargs)
        seen.append(result)
        return result
    monkeypatch.setattr(Looks, 'run', observed)
    try:
        for text in ('alpha gamma', 'beta delta', 'alpha delta beta', 'delta gamma beta alpha'):
            seen.clear()
            read(model, text)
            reading = seen[0]
            count = len(text.split())
            taken = [a for a in reading.actions[0].tolist() if a >= 0]
            assert sorted(taken) == list(range(count)), (text, taken)
            assert reading.accepted[0, :count].all() and int(reading.looks[0]) == count
            print(text, taken)
    finally:
        model.End()


def test_39_one_gain_field_gives_the_surprise_attend_reads():
    """The surprise is the excess and the omission of one gain field, the
    look's activity its attention and its forecast its expected; the old
    difference and the floor key are named nowhere a look reads."""
    from pathlib import Path
    from LookForecast import LookForecast
    from Expectation import gain_field
    root = Path(__file__).resolve().parents[1] / 'bin'
    for name in ('LookForecast.py', 'SentenceLooks.py', 'AttentionContext.py', 'Salience.py'):
        source = (root / name).read_text()
        assert 'def difference' not in source and 'attentionFloor' not in source
    actual = torch.tensor([[[.9, 0.], [0., 0.], [.2, .1]]])
    forecast = torch.tensor([[[.5, 0.], [.4, 0.], [.2, .1]]])
    pair = LookForecast.surprise(actual, forecast, torch.tensor([[True, False]]))
    field = gain_field(actual, forecast, attention=torch.ones(1, 1))
    torch.testing.assert_close(pair, torch.stack((field.excess.sum((-1, -2)), field.omission.sum((-1, -2))), -1))
    torch.testing.assert_close(pair, torch.tensor([[.4, .4]]))
    assert not LookForecast.surprise(actual, forecast, torch.tensor([[False, False]])).any()


def test_15_what_the_request_learns_of_the_what_and_a_word_present_by_degree(tmp_path, monkeypatch):
    """Observed, not repaired (step 15): the gradient the look passes to the
    request through the cap when "words" is asked at one, and what a word
    whose events are present at .3 comes back as."""
    import SentenceLooks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        read(model, 'alpha gamma')
        owner, bank = model._concept_owner(), model._sentence_bank
        perception = owner.basic_percepts.read_bank(bank)
        concepts = model._attention_word_concepts[:, :2]
        query = torch.zeros(1, int(model._stm_reducer().d_model))
        request = SentenceLooks.requested(model, query).requires_grad_()
        field, activity, _ = SentenceLooks.look(model, bank, request, bank.activity, perception)
        extent = SentenceLooks.extent_of(bank, activity, 2)
        value = SentenceLooks.word_activity(field, concepts, torch.ones(1, 2, dtype=torch.bool), extent, bank.activity, bank)
        value[torch.isfinite(value)].sum().backward()
        through_cap = float(request.grad.abs().sum())
        weak = bank.activity.clone()
        weak[bank.token == 1] *= .3
        field, activity, _ = SentenceLooks.look(model, bank, request.detach(), weak, perception)
        extent = SentenceLooks.extent_of(bank, activity, 2)
        degree = SentenceLooks.word_activity(field, concepts, torch.ones(1, 2, dtype=torch.bool), extent, weak, bank)
        print(dict(gradient_on_the_request=through_cap, words_at_full=value.detach()[0].tolist(),
                   second_word_at_point_three=degree[0].tolist()))
        assert through_cap == 0.
    finally:
        model.End()
