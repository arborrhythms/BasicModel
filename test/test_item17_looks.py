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
