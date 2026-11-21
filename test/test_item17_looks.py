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
    """A bank of one stream: ``events`` ``{key: [(occurrence, for,
    against)]}``, each occurrence the span ``[p, p + 1]`` in the reading
    ``readings_of_events[p]``."""
    P, K = len(readings_of_events), max(readings_of_events) + 1
    members = torch.zeros(len(keys), 1, P, 2)
    for i, key in enumerate(keys):
        for event, for_, against in events.get(key, ()):
            members[i, 0, event] = torch.tensor([for_, against])
    spans = torch.tensor([[[p, p + 1] for p in range(P)]])
    token = torch.tensor([readings_of_events])
    where = torch.tensor([[[min(p for p in range(P) if readings_of_events[p] == k),
                            max(p for p in range(P) if readings_of_events[p] == k) + 1] for k in range(K)]])
    opposed = tuple(() for _ in groups) if opposed is None else opposed
    return PerceptBank(tuple(range(len(groups))), tuple(map(tuple, groups)), opposed, tuple(keys), members,
                       spans, token, torch.zeros_like(token), torch.zeros(1, P, 0, 2), torch.ones(1, P, 2),
                       where, torch.zeros_like(where))


def row_values(bank):
    """The rows' two lanes as the way up reads them at each occurrence: the
    run's presence times the product of the wholes; against, the presence
    times the union of the wholes' against lanes; a reading's, the greatest
    over its occurrences."""
    order = {key: i for i, key in enumerate(bank.keys)}
    m = bank.members
    at = []
    for group in bank.groups:
        presence = m[order[group[0]], ..., 0]
        wholes = [m[order[key], ..., 0] for key in group[1:]]
        against = [m[order[key], ..., 1] for key in group[1:]]
        for_lane = presence * torch.stack(wholes).prod(0) if wholes else presence
        kept = torch.stack([1 - a for a in against]).prod(0) if against else torch.ones_like(presence)
        at.append(torch.stack((for_lane, presence * (1 - kept)), -1))
    at = torch.stack(at)                                                  # [R, B, P, 2]
    K = bank.where.shape[1]
    held = bank.token[None, :, None, :] == torch.arange(K)[None, None, :, None]   # [1, B, K, P]
    values = torch.where(held[..., None], at[:, :, None], 0.).amax(-2)
    index = torch.arange(bank.token.shape[1])[None].expand(bank.token.shape[0], -1)
    return PerceptField(tuple(('bp', r) for r in bank.rows), values, at, bank.span, bank.where, bank.when,
                        reading=bank.token, index=index)


def held_banks(monkeypatch):
    """The bank of each sentence opened, kept for the test: the model lets
    a sentence's bank go when the sentence closes."""
    import FamiliarTier
    banks, opened = [], FamiliarTier.open_sentence
    def recorded(model, sid, present):
        result = opened(model, sid, present)
        banks.append(model._sentence_bank)
        return result
    monkeypatch.setattr(FamiliarTier, 'open_sentence', recorded)
    return banks


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
    banks = held_banks(monkeypatch)
    read(model, 'alpha gamma')
    owner, bank = model._concept_owner(), banks[-1]
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
    spans, token = bank.span[0], bank.token[0]
    word = token == 0
    beside = token == 1
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
    # The occurrences of the word that hold its row take it; the other
    # word's occurrences none.
    assert bool((salient.events[0, word, 0] > 0).any())
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
    _, events, _ = Salience.first_pi(bank, row_values(bank), row_grad)
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
    members, events, _ = Salience.first_pi(bank, read, row_grad)
    print(dict(row=float(read.values[0, 0, 0, 0]), members=members[:, 0, 0, 0].tolist(),
               events=events[0, :, 0].tolist()))
    assert members[:, 0, 0, 0].tolist() == pytest.approx([.4, .4, .4])
    # Read at each occurrence: the properties' other occurrences take nothing.
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
        banks = held_banks(monkeypatch)
        read(model, 'alpha gamma')
        owner, bank = model._concept_owner(), banks[-1]
        model._sentence_bank = bank   # the look reads the bank it is given
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
        banks = held_banks(monkeypatch)
        read(model, 'alpha gamma')
        owner, bank = model._concept_owner(), banks[-1]
        model._sentence_bank = bank   # the look reads the bank it is given
        perception = owner.basic_percepts.read_bank(bank)
        concepts = model._attention_word_concepts[:, :2]
        query = torch.zeros(1, int(model._stm_reducer().d_model))
        # The request's start is a parameter: the request is taken as a leaf.
        request = SentenceLooks.requested(model, query).detach().requires_grad_()
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


def test_38b_the_reading_lesson_by_grades(tmp_path, monkeypatch):
    """Certificate 38 (b), a learning certificate: the reading lesson teaches
    the request, minimal sentences first; printed, before and after, are the
    sentences read in order with no reading supplied. The grades of rule 70:
    the sentence (a look of "words" returns every word), its words in order,
    a word's parts (certificate 56 (a)). Recorded, not repaired."""
    from SentenceLooks import Looks
    from AttentionLesson import reading_lesson, minimal_first, in_order
    from test_item18_expect import read
    texts = ('alpha gamma', 'beta delta', 'alpha delta', 'beta gamma')
    model = nominated_lesson(tmp_path, monkeypatch, texts)
    run, seen = Looks.run, []
    def observed(self, arrive, **kwargs):
        result = run(self, arrive, **kwargs)
        seen.append(result)
        return result
    monkeypatch.setattr(Looks, 'run', observed)
    def count():
        found = 0
        for text in texts:
            seen.clear()
            read(model, text)
            found += in_order(seen[0], len(text.split()))
        return found
    try:
        before = count()
        optimizer = model.getOptimizer(lr=.01)
        grades = minimal_first(texts + ('alpha delta beta', 'beta gamma delta'))
        costs = []
        for grade in grades:
            for presentation in range(8):
                text = grade[presentation % len(grade)]
                with reading_lesson(model, [text.split()]):
                    seen.clear()
                    read(model, text, training=True, optimizer=optimizer)
                    costs.append(float(seen[0].looks.sum()))
        after = count()
        print(dict(grades=grades, in_order_before=before, in_order_after=after, of=len(texts),
                   looks_taught=sum(costs)))
        assert [len(grade) for grade in grades] == [4, 2] and grades[0] == list(texts)
    finally:
        model.End()


def _language():
    """A grammar of one unary rule that takes a word as its operand
    (``op``), beside interpret and sign."""
    from types import SimpleNamespace
    rule = lambda name: SimpleNamespace(method_name=name)
    return SimpleNamespace(_compose_unary_rules=(rule('interpret'), rule('sign'), rule('op')),
                           _compose_binary_rules=(rule('join'),))


def test_40_a_literal_enters_only_under_a_rule_that_takes_it():
    """Certificate 40 (rule 42): a word left uninterpreted enters the tree
    as a literal where a rule takes it as an operand; where none does, the
    sentence is not well formed and ends in silence."""
    from Interpret import leaf_readings, well_formed
    language = _language()
    shift = lambda word: [0, -1, word]
    unary = lambda local, at=0: [2, local, at]
    binary = lambda local: [1, local, -1]
    # Two words, the second uninterpreted and taken by the binary rule.
    taken = torch.tensor([shift(0), unary(0), shift(1), binary(0)])
    assert leaf_readings(language, taken, 2, forced=False) == ('symbol', 'no_op')
    assert well_formed(language, taken, 2, forced=False)
    # One word taken by the unary rule ``op`` as a literal.
    assert well_formed(language, torch.tensor([shift(0), unary(2)]), 1, forced=False)
    # One word that nothing takes: no formula.
    assert not well_formed(language, torch.tensor([shift(0)]), 1, forced=False)
    # Interpreted, it stands as a symbol.
    assert well_formed(language, torch.tensor([shift(0), unary(0)]), 1, forced=False)


def test_41_silence_is_registered_and_in_no_grammar_file():
    from Language import GRAMMAR_LAYER_CLASSES
    from Interpret import SilenceLayer
    assert GRAMMAR_LAYER_CLASSES['silence'] is SilenceLayer
    assert SilenceLayer.rule_name == 'silence' and SilenceLayer.arity == 0
    root = Path(__file__).resolve().parents[1] / 'data'
    for path in (*root.glob('*.grammar'), *root.glob('*.xml')):
        assert 'silence' not in path.read_text(), path


def test_41_a_grammar_file_may_not_name_silence():
    from Language import TheGrammar
    with pytest.raises(ValueError, match='silence'):
        TheGrammar._parse_rule('S', 'silence()')


def test_41_a_sentence_whose_trees_do_not_reduce_is_written_as_silence():
    """Certificate 41 (rule 48): with a grammar that joins no two words, a
    two-word sentence is written as S -> NULL: its row holds no word and no
    concept, MTM takes nothing, the put back is NULL, its cost is booked;
    a one-word sentence of the same grammar reduces and is unchanged."""
    import Language, Models
    from Interpret import SilenceLayer
    from util import init_config
    from data import TheData
    config = Path(__file__).resolve().parents[1] / 'data' / 'MM_ltm_consolidation_fixture.xml'
    init_config(path=str(config), defaults_path=str(config.parent / 'model.xml'))
    Language.TheGrammar._configured = False
    model, _ = Models.BasicModel.from_config(str(config))
    TheData.load('xor')
    model.eval()
    try:
        store = model.symbolSpace.ltm_store
        booked = []
        add = Models.Error.add
        def recorded(registry, name, value, *args, **kwargs):
            if name == 'compose.silence':
                booked.append(value.detach().clone())
            return add(registry, name, value, *args, **kwargs)
        Models.Error.add = recorded
        import SentenceField
        generated = SentenceField.reconstruct_generation
        put = []
        def kept(*args, **kwargs):
            # The grammar's inverse puts back something at every place, so
            # that what the silence leaves of it shows.
            result = generated(*args, **kwargs)
            result[1]._reconstruction_percepts = torch.ones_like(result[1]._reconstruction_percepts)
            put.append(result[1])
            return result
        SentenceField.reconstruct_generation = kept
        try:
            for texts, silent in ((['alpha gamma'], True), (['alpha'], False)):
                before, entries = len(store), dict(model._concept_owner().mtm.entries)
                put.clear()
                with torch.no_grad():
                    model.forward(model.inputSpace.prepInput(texts))
                assert bool(model._sentence_silence.any()) is silent
                # What the model's reconstruction puts back for a silent
                # sentence is NULL at every place; it is scored so.
                assert put and bool(put[-1]._reconstruction_percepts.any()) is not silent
                if silent:
                    rows = range(before, len(store))
                    assert len(rows) == 1
                    row = rows[0]
                    assert not store.slots[row, :store.when_start].any()
                    assert store.referent_of(row) is None and float(store.trust[row]) == 0.
                    assert model._concept_owner().mtm.entries == entries
                    assert booked and float(booked[-1].sum()) == 1.
        finally:
            Models.Error.add = add
            SentenceField.reconstruct_generation = generated
    finally:
        model.End()
    like = torch.rand(2, 5)
    assert not SilenceLayer().reverse(like).any() and not SilenceLayer()(like).any()


def test_30_33a_35_36_42_the_bank_the_where_wires_garbage_the_same_look_and_an_unknown_word(tmp_path, monkeypatch):
    """Certificate 30 (rule 33): no ``materialize()`` on a look's path,
    counted over a sentence's looks. 33 (a) (rule 47): the where wires equal
    the encoding's code at every place before and after a training step,
    and are in no optimizer's groups. 35: a known word among runs of letters
    that are no form is returned alone by "words". 36 (1.8 §74): a look with
    the same request gives the same result whatever was looked at before
    it. 42: a word not yet known, asked for in a supplied reading, is read
    whole and taken."""
    import SentenceLooks
    import Spaces
    from test_item18_expect import read
    from reading_fixtures import SourceOrder
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        # 33 (a).
        registry = model.where_registry
        code = registry.encoding.encode(torch.arange(registry.capacity)).float().to(registry.code)
        torch.testing.assert_close(registry.code.detach(), code)
        optimizer = model.getOptimizer(lr=.003)
        held = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert id(registry.code) not in held
        read(model, 'alpha gamma', training=True, optimizer=optimizer)
        torch.testing.assert_close(registry.code.detach(), code)
        # 30: count the materialize() calls made inside a look.
        inside, calls = [False], [0]
        materialize, look = Spaces.SubSpace.materialize, SentenceLooks.look
        def counted(self, *args, **kwargs):
            calls[0] += inside[0]
            return materialize(self, *args, **kwargs)
        def watched(*args, **kwargs):
            inside[0] = True
            try:
                return look(*args, **kwargs)
            finally:
                inside[0] = False
        monkeypatch.setattr(Spaces.SubSpace, 'materialize', counted)
        monkeypatch.setattr(SentenceLooks, 'look', watched)
        looks = []
        run = SentenceLooks.Looks.run
        def observed(self, arrive, **kwargs):
            result = run(self, arrive, **kwargs)
            looks.append(int(result.looks[0]))
            return result
        monkeypatch.setattr(SentenceLooks.Looks, 'run', observed)
        read(model, 'alpha gamma')
        assert looks and looks[0] >= 2
        assert calls[0] == 0
        monkeypatch.setattr(SentenceLooks, 'look', look)
        # 35 and 36: "words" over a sentence with runs of letters that are no form.
        banks = held_banks(monkeypatch)
        read(model, 'qzv alpha xwpt')
        owner, bank = model._concept_owner(), banks[-1]
        model._sentence_bank = bank
        perception = owner.basic_percepts.read_bank(bank)
        concepts = model._attention_word_concepts[:, :3]
        known = concepts >= 0
        words = SentenceLooks.requested(model, torch.zeros(1, int(model._stm_reducer().d_model)))
        def found(request, own):
            field, activity, _ = SentenceLooks.look(model, bank, request, own, perception)
            extent = SentenceLooks.extent_of(bank, activity, 3)
            value = SentenceLooks.word_activity(field, concepts, known, extent, own, bank)
            return field.evidence.detach().clone(), value[0].detach().clone()
        first_field, first = found(words, bank.activity)
        print(dict(known=known[0].tolist(), words=first.tolist()))
        assert known[0].tolist() == [False, True, False]
        assert torch.isfinite(first).tolist() == [False, True, False]
        # 36: a look for another word, then the first request again.
        found(SentenceLooks.word_request(model, concepts, torch.tensor([1])), bank.activity)
        again_field, again = found(words, bank.activity)
        torch.testing.assert_close(again_field, first_field, rtol=0, atol=0)
        torch.testing.assert_close(again, first, rtol=0, atol=0)
        # 42: a supplied reading takes the word not yet known.
        monkeypatch.setattr(model, '_attention_supplied_reading', SourceOrder(), raising=False)
        seen = []
        def kept(self, arrive, **kwargs):
            result = run(self, arrive, **kwargs)
            seen.append(result)
            return result
        monkeypatch.setattr(SentenceLooks.Looks, 'run', kept)
        read(model, 'alpha omega')
        reading = seen[0]
        print(dict(actions=reading.actions[0].tolist(), accepted=reading.accepted[0, :2].tolist(),
                   descended=reading.descended[0, :2].tolist()))
        assert [a for a in reading.actions[0].tolist() if a >= 0] == [0, 1]
        assert reading.accepted[0, :2].all() and bool(reading.descended[0, 1])
    finally:
        model.End()


def test_37_dimming_lowers_both_lanes_and_is_never_against():
    """Certificate 37 (rule 39): a reading away from where the request looks
    is lowered on both lanes by its where match, which lies in (0, 1], so it
    is never read as against; the key ``attentionFloor`` is read by no look."""
    from types import SimpleNamespace
    from SentenceLooks import where_match
    bank = SimpleNamespace(token=torch.tensor([[0, 1, 2]]), place=torch.tensor([[0, 4, 9]]))
    match = where_match(bank, torch.tensor([[3., 1., 0.]]), torch.ones(1, 3, 2), 3)
    assert match[0, 0] == 1 and bool((match > 0).all()) and bool((match <= 1).all())
    evidence = torch.tensor([[[[.8, .1], [.6, .2], [.9, 0.]]]])            # [S, B, readings, 2]
    dimmed = evidence * match[None, :, :, None]
    assert bool((dimmed <= evidence).all()) and bool((dimmed >= 0).all())
    value = dimmed[..., 0] - dimmed[..., 1]
    assert bool((torch.sign(value) == torch.sign(evidence[..., 0] - evidence[..., 1])).all())
    root = Path(__file__).resolve().parents[1] / 'bin'
    assert 'attentionFloor' not in (root / 'SentenceLooks.py').read_text()


def test_43_a_word_read_at_for_one_and_against_a_trace_is_interpreted():
    """Certificate 43 (1.8 §64.7): a word concept read at for 1 and against
    0.0002 is one signed number just under one, the most active, and taken."""
    from types import SimpleNamespace
    from SentenceLooks import word_activity
    from Attention import draw_among
    evidence = torch.tensor([[[[1., .0002], [.4, 0.]]]])                    # [S, B, W, 2]
    field = SimpleNamespace(concept_ids=torch.tensor([7]), evidence=evidence)
    concepts = torch.tensor([[7, 7]])
    bank = SimpleNamespace(token=torch.tensor([[0, 1]]))
    own = torch.ones(1, 2, 2)
    value = word_activity(field, concepts, torch.ones(1, 2, dtype=torch.bool),
                          torch.ones(1, 2, dtype=torch.bool), own, bank)
    assert float(value[0, 0]) == pytest.approx(.9998)
    assert draw_among(value)[0].tolist() == [0]


# --- The salience map against the framework's backward (certificate 31) -----

def _framework_down(graph, request, values, readings, depth):
    """The way down of the structure by the framework's backward: the way up
    unrolled ``depth`` times with each Sigma linear in its children (before
    its squash) and each symbol's Pi linearized at the open run's values."""
    keep = graph.child < graph.part_start
    parent, child = graph.parent[keep], graph.child[keep]
    weight, negated = graph.weight[keep].detach().to(request), graph.negated[keep]
    symbol = ((parent >= graph.concepts) & (parent < graph.percept_start))[:, None, None, None]
    tiny = torch.finfo(request.dtype).tiny
    coeff = torch.where(symbol, values.index_select(0, parent) / values.index_select(0, child).clamp_min(tiny),
                        1.) * weight[:, None, None, None]
    leaves = request.new_zeros(graph.size, request.shape[1], readings, 2).requires_grad_()
    x = leaves
    for _ in range(depth):
        lanes = x.index_select(0, child)
        lanes = torch.where(negated[:, None, None, None], lanes.flip(-1), lanes)
        x = leaves + torch.zeros_like(leaves).index_add(0, parent, coeff * lanes)
    asked = request[:graph.concepts, :, None, :].clamp_min(0.)
    loss = (asked * x[:graph.concepts]).sum()
    return torch.autograd.grad(loss, leaves)[0]


def test_31a_the_transposes_give_the_frameworks_backward_in_the_model(tmp_path, monkeypatch):
    """(a) In a model, on "alpha gamma": the structure passed down by its
    weights and the symbols' derivatives (``Salience._down``), and the first
    Pi passed down to its members (``Salience.first_pi``), each against the
    framework's backward on the same local derivatives (atol 1e-6)."""
    import ConceptSpread
    import SentenceLooks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        banks = held_banks(monkeypatch)
        read(model, 'alpha gamma')
        owner, bank = model._concept_owner(), banks[-1]
        basic = owner.basic_percepts
        perception = basic.read_bank(bank)
        readings = perception.values.shape[2]
        graph = ConceptSpread.structure(owner, perception.values.device)
        # The open run's values: the order-zero concepts as read, the
        # symbols the product of what they hold.
        with torch.no_grad():
            field = owner.cs_read_bank(bank, publish=False)
        values = field.new_zeros(graph.size, field.shape[1], readings, 2)
        width = min(len(field), graph.concepts)
        values[:width] = field[:width]
        held = (graph.parent >= graph.concepts) & (graph.parent < graph.percept_start)
        logs = torch.log(values.index_select(0, graph.child[held]).clamp_min(1e-30))
        product = torch.exp(values.new_zeros(values.shape).index_add(0, graph.parent[held], logs))
        is_symbol = torch.zeros(graph.size, dtype=torch.bool)
        is_symbol[graph.concepts:graph.percept_start] = True
        values = torch.where(is_symbol[:, None, None, None], product, values)
        alpha, gamma = word_rows(model, 'alpha', 'gamma')
        concepts = model._attention_word_concepts[:, :2]
        words = SentenceLooks.requested(model, torch.zeros(1, int(model._stm_reducer().d_model)))
        depth = 2 * (len(owner._order_caps()) + 1)
        compared = {}
        for name, request in (('words', words),
                              ('alpha', SentenceLooks.word_request(model, concepts, torch.tensor([0])))):
            request = request.to(values)
            mine = Salience._down(graph, request, values, readings, depth)
            theirs = _framework_down(graph, request, values, readings, depth)
            upto = graph.part_start
            compared[name] = float((mine[:upto] - theirs[:upto]).abs().max())
            assert float(mine[graph.percept_start:upto].abs().sum()) > 0
            torch.testing.assert_close(mine[:upto], theirs[:upto], atol=1e-6, rtol=0)
        # The first Pi: the members' shares against the backward of the way
        # up's rows by the members' values at each attended percept, a
        # reading the greatest over its attended percepts.
        index = perception.index
        keys = len(bank.keys)
        members = bank.members.gather(2, index[None, ..., None].expand(keys, -1, -1, 2))
        members = (members * (perception.reading >= 0)[None, ..., None]).clone().requires_grad_()
        order = {key: i for i, key in enumerate(bank.keys)}
        at = []
        for group, opposed in zip(bank.groups, bank.opposed):
            if not group:
                at.append(members.new_zeros(members.shape[1:]))
                continue
            presence = members[order[group[0]], ..., 0]
            wholes = [members[order[key], ..., 0] for key in group[1:]]
            sources = ([members[order[key], ..., 1] for key in group[1:]]
                       + [members[order[key], ..., 0] for key in opposed if key in order])
            for_lane = presence * torch.stack(wholes).prod(0) if wholes else presence
            kept = torch.stack([1 - v for v in sources]).prod(0) if sources else torch.ones_like(presence)
            at.append(torch.stack((for_lane, presence * (1 - kept)), -1))
        at = torch.stack(at)                                                    # [R, B, A, 2]
        held = perception.reading[None, :, None, :] == torch.arange(readings)[None, None, :, None]
        rows = torch.where(held[..., None], at[:, :, None], 0.).amax(-2)        # [R, B, K, 2]
        row_grad = torch.rand(rows.shape, generator=torch.Generator().manual_seed(0)) + .1
        d_members, = torch.autograd.grad((row_grad * rows).sum(), (members,))
        shares = d_members * members                                            # [keys, B, A, 2]
        theirs = torch.where(held[..., None], shares[:, :, None], 0.).sum(-2)
        read_rows = PerceptField(perception.keys, rows.detach(), at.detach(), perception.spans,
                                 perception.where, perception.when, reading=perception.reading, index=index)
        mine, _, _ = Salience.first_pi(bank, read_rows, row_grad)
        compared['first Pi'] = float((mine - theirs.detach()).abs().max())
        print('largest difference from the framework:', compared)
        assert float(mine.abs().sum()) > 0
        torch.testing.assert_close(mine, theirs.detach(), atol=1e-6, rtol=0)
    finally:
        model.End()


def test_31c_printed_parts_present_by_degree_three_ways():
    """(c) Printed: a row of a run and two properties, its parts present by
    degree, passed down three ways: by the transposes (the share, the
    gradient times the value), by the plain gradient, and by the rows'
    entries alone (each member the row's share, whatever its value)."""
    keys = [('ps', 'a'), ('ws', 'red'), ('ws', 'round')]
    for degrees in ((1., 1., 1.), (.9, .6, .3), (.5, .5, .5)):
        run, red, round_ = degrees
        bank = hand_bank(keys, [keys],
                         {('ps', 'a'): [(0, run, 0.)], ('ws', 'red'): [(0, red, 0.)],
                          ('ws', 'round'): [(0, round_, 0.)]}, [0])
        read = row_values(bank)
        row_grad = torch.zeros(1, 1, 1, 2)
        row_grad[..., 0] = 1.
        members, _, _ = Salience.first_pi(bank, read, row_grad)
        # The way up at the one occurrence: the run's presence times each
        # whole read there.
        values = bank.members[:, 0, 0, 0].clone().requires_grad_()
        row = values[0] * values[1] * values[2]
        plain, = torch.autograd.grad(row, values)
        entries = torch.ones_like(values)
        print(dict(degrees=degrees, row=round(float(row), 4),
                   transposes=[round(v, 4) for v in members[:, 0, 0, 0].tolist()],
                   plain_gradient=[round(v, 4) for v in plain.tolist()],
                   entries_alone=entries.tolist()))
        torch.testing.assert_close(float(row), float(read.values[0, 0, 0, 0]))
        torch.testing.assert_close(members[:, 0, 0, 0], (plain * values.detach()).float())
        assert float(members.min()) >= 0.


def test_32_a_known_word_is_not_read_inside_a_longer_known_word(tmp_path, monkeypatch):
    """A word's concept reads nothing at the token of a longer known word
    that holds its letters, ``is`` at ``This``, ``here`` at ``There``, ``the``
    at ``there``, and reads at its own token; the look that asks for it
    returns it there and not the longer word: the request reaches a row's
    reading through the Not, for less against, so the inner word's row at
    the longer token (read for and against alike) takes nothing."""
    import SentenceLooks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch, texts=('This is here', 'There is the', 'the is there'))
    try:
        banks = held_banks(monkeypatch)
        found = {}
        for text, word, at, longer in (('This is here', 'is', 1, 0), ('There is here', 'here', 2, 0),
                                       ('there is the', 'the', 2, 0)):
            read(model, text)
            owner, bank = model._concept_owner(), banks[-1]
            model._sentence_bank = bank
            perception = owner.basic_percepts.read_bank(bank)
            count = len(text.split())
            concepts = model._attention_word_concepts[:, :count]
            request = SentenceLooks.word_request(model, concepts, torch.tensor([at]))
            field, activity, _ = SentenceLooks.look(model, bank, request, bank.activity, perception)
            ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, 0]
            own = (ids == concepts[0, at]).nonzero().flatten()
            signed = field.evidence[own[0], 0, :count]
            inner = (signed[:, 0] - signed[:, 1]).tolist()
            extent = SentenceLooks.extent_of(bank, activity, count)
            value = SentenceLooks.word_activity(field, concepts, torch.ones(1, count, dtype=torch.bool),
                                                extent, bank.activity, bank)
            from Attention import draw_among
            index, drawn, _, _ = draw_among(value)
            found[(text, word)] = dict(inner_concept_by_reading=[round(v, 4) for v in inner],
                                       returned=[round(float(v), 4) for v in value[0]],
                                       taken=int(index[0]), drawn=bool(drawn[0]))
            assert abs(inner[longer]) < 1e-6 and inner[at] > 0, found[(text, word)]
            assert torch.isfinite(value[0, at])
            assert not torch.isfinite(value[0, longer]) or float(value[0, longer]) < float(value[0, at]), \
                found[(text, word)]
            assert int(index[0]) == at and not bool(drawn[0]), found[(text, word)]
        print(found)
    finally:
        model.End()
