"""Retained rows and policy credit have one tensor operation graph per batch."""
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    return torch.device(request.param)


@pytest.mark.parametrize('batch_dim', [0, 1])
def test_stream_selection_preserves_values_and_gradients_with_flat_graph(device, batch_dim):
    from SentenceCompose import select_streams
    operations = []
    def backend(graph, inputs):
        operations.append(sum(node.op == 'call_function' for node in graph.graph.nodes))
        return graph.forward
    for batch in (4, 16):
        values = [torch.randn(batch, 5, 3, device=device).movedim(0, batch_dim).requires_grad_()
                  for _ in range(3)]
        winners = torch.arange(batch, device=device) % 3
        function = torch.compile(lambda values, winners: select_streams(values, winners, batch_dim=batch_dim),
                                 backend=backend, fullgraph=True, dynamic=False)
        selected = function(values, winners)
        expected = torch.stack([values[int(winners[row])].movedim(batch_dim, 0)[row]
                                for row in range(batch)]).movedim(0, batch_dim)
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(selected.square().sum(), values, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), values)
        for actual, wanted in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert len(operations) == 2 and operations[0] == operations[1]


@pytest.mark.parametrize('batch', [4, 16])
@pytest.mark.parametrize('tie', [False, True])
def test_thought_credit_matches_serial_values_and_gradients_in_one_call(device, batch, tie, monkeypatch):
    from Layers import Error
    import SentenceCredit
    from WholePaths import credit
    probabilities = torch.linspace(.1, .8, batch, device=device, requires_grad=True)
    active = torch.arange(batch, device=device) % 4 != 3
    rounds = torch.where(torch.arange(batch, device=device) % 3 == 0, -1, 0)
    draws = dict(thought_round=rounds, walk_count=torch.full_like(rounds, 2), walk_rounds=torch.full_like(rounds, 3))
    parts = torch.ones(batch, 3, device=device)
    alternative = parts.clone()
    if not tie:
        alternative[:, 0] += torch.arange(batch, device=device) % 3 - 1
    choices = {row: dict(choices=[dict(probability=probabilities[row], alternatives=row + 2)])
               for row in range(batch)}
    first = SimpleNamespace(parts=parts, output_trace=None)
    path = SimpleNamespace(parts=alternative, draw=draws, attention_words=None, output_trace=None,
                           thinking=SimpleNamespace(traces=choices), registry=Error(row_mask=active))
    costs = torch.stack((parts[:, 0] + parts[:, 2], alternative[:, 0] + alternative[:, 2]), -1)
    original = SentenceCredit.score_function
    expected = probabilities.new_zeros(batch)
    for row in range(batch):
        if int(rounds[row]) < 0:
            continue
        value, _ = original(probabilities[row:row+1, None], probabilities.new_tensor([[6 * (row + 2)]]),
                            active[row:row+1, None], costs[row:row+1])
        expected = expected.index_add(0, rounds.new_tensor([row]), value)
    calls = []
    def observe(*args):
        calls.append(args[0].shape[0])
        return original(*args)
    monkeypatch.setattr(SentenceCredit, 'score_function', observe)
    credit(None, first, path, active)
    actual = path.registry._value(path.registry._terms['output.thought_score_function'])
    assert calls == [batch]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.requires_grad == expected.requires_grad
    if actual.requires_grad:
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), probabilities, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), probabilities)[0], rtol=0, atol=0)


def test_thought_retention_aligns_percept_keys_extents_and_gradients(device):
    from Memory import ShortTermMemory
    from PerceptField import PerceptField
    from WalkTrials import ThoughtTrialState
    from clock_fixtures import for_stm
    model = for_stm(ShortTermMemory(batch=4, capacity=3, concept_dim=2))
    fields, states = [], []
    for keys, readings, events in [((('ps', 1), ('ps', 3)), 2, 3),
                                   ((('ps', 2), ('ps', 3)), 3, 2),
                                   ((('ps', 1),), 1, 4)]:
        values = torch.randn(len(keys), 4, readings, 2, device=device, requires_grad=True)
        locations = torch.randn(4, readings, 2, device=device)
        field = PerceptField(keys, values, torch.randn(len(keys), 4, events, 2, device=device),
            torch.randn(4, events, 2, device=device), locations, locations + 1,
            when_band=torch.randn(4, readings, 4, device=device),
            reading=torch.zeros(4, events, dtype=torch.long, device=device),
            index=torch.arange(events, device=device)[None].expand(4, -1))
        model.conceptualSpace._percept_field = field
        fields.append(field)
        states.append(ThoughtTrialState(model))
    winners = torch.tensor([2, 0, 1, 2], device=device)
    ThoughtTrialState.keep(states, winners)
    kept = model.conceptualSpace._percept_field
    assert kept.keys == (('ps', 1), ('ps', 2), ('ps', 3))
    expected = kept.values.new_zeros(3, 4, 3, 2)
    for row, selected in enumerate(winners.tolist()):
        field = fields[selected]
        for index, key in enumerate(field.keys):
            expected[kept.keys.index(key), row, :field.values.shape[2]] = field.values[index, row]
        torch.testing.assert_close(kept.where[row, :field.where.shape[1]], field.where[row], rtol=0, atol=0)
        assert (kept.reading[row, field.reading.shape[1]:] == -1).all()
    torch.testing.assert_close(kept.values, expected, rtol=0, atol=0)
    inputs = [field.values for field in fields]
    actual = torch.autograd.grad(kept.values.square().sum(), inputs, retain_graph=True)
    wanted = torch.autograd.grad(expected.square().sum(), inputs)
    for left, right in zip(actual, wanted, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_nested_carry_and_padded_reconstruction_keep_three_paths(device):
    from SentenceCompose import select_paths, select_rows
    from SentenceField import select_reconstructions, selected
    from Understanding import InputReconstruction
    values = [torch.randn(4, 3, device=device, requires_grad=True) for _ in range(3)]
    paths = [(value, {'nested': (value.square(), value > 0)}) for value in values]
    winners = torch.tensor([2, 0, 1, 2], device=device)
    compiled = torch.compile(select_paths, backend='eager', fullgraph=True)
    actual = compiled(paths, winners)
    expected = paths[0]
    for index, path in enumerate(paths[1:], 1):
        expected = select_rows(expected, path, winners == index)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1]['nested'][0], expected[1]['nested'][0])
    assert torch.equal(actual[1]['nested'][1], expected[1]['nested'][1])
    readings = [InputReconstruction(value[:, 0], value[:, :width], value[:, 1], value[:, 2],
        value[:, 0] > 0, value[:, :width], None) for value, width in zip(values, (1, 3, 2))]
    actual = torch.compile(select_reconstructions, backend='eager', fullgraph=True)(readings, winners)
    expected = readings[0]
    for index, reading in enumerate(readings[1:], 1):
        expected = selected(expected, reading, winners == index)
    for name in ('ideas', 'event', 'idea_cost', 'byte_cost', 'truncated', 'sentence_costs'):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    a = torch.autograd.grad(actual.event.square().sum(), values, retain_graph=True)
    b = torch.autograd.grad(expected.event.square().sum(), values)
    assert all(torch.equal(left, right) for left, right in zip(a, b))


def test_binding_close_returns_live_padded_decisions_for_device_credit(device):
    from ReferenceClose import decision_tensors, departed_probability
    operations = []
    def backend(graph, inputs):
        operations.append(sum(node.op == 'call_function' for node in graph.graph.nodes))
        return graph.forward
    for batch in (4, 16):
        logits = torch.randn(batch, 2, 3, device=device, requires_grad=True)
        probabilities = logits.softmax(-1)
        plans = [dict(decisions=[dict(candidates=(None, 7, 9), probabilities=probabilities[b, i],
                    selected=(b+i) % 3, departed=i == b % 2 and b % 4 != 0)
                    for i in range(0 if b % 4 == 0 else 2)]) for b in range(batch)]
        eligible, values, departed = decision_tensors(plans, logits[:, 0])
        assert eligible.shape == values.shape == departed.shape == (batch, 2)
        function = torch.compile(departed_probability, backend=backend, fullgraph=True, dynamic=False)
        actual = function(values, departed)
        expected = torch.stack([probabilities[b, b % 2, (b+b % 2) % 3]
                                if b % 4 else logits.new_zeros(()) for b in range(batch)])
        assert torch.equal(actual, expected)
        a = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
        b = torch.autograd.grad(expected.sum(), logits)[0]
        assert torch.equal(a, b)
    assert operations[0] == operations[1]
    empty = decision_tensors([], torch.zeros(4, 2, device=device))
    assert departed_probability(empty[1], empty[2]).tolist() == [0.] * 4


def test_temporal_head_publication_uses_each_stream_tick_and_restores(device):
    from Memory import ModelState
    state = ModelState().to(device)
    state.reserve_heads(4, 20)
    def publication(ticks, heads):
        state.publish_heads(ticks, heads)
        return state.root_at(torch.arange(len(heads), device=heads.device), ticks)
    captured = torch.compile(publication, backend='eager', fullgraph=True)
    oracle = {}
    for ticks, heads in [([5, 6, 7, 8], [11, -9, 13, -17]),
                         ([3, 5, 5, 8], [-21, 22, -1, 0]),
                         ([5, 6, 5, 6], [-25, 31, 42, -18]),
                         ([8, 8, 8, 8], [0, -1, 0, -1])]:
        ticks, heads = torch.tensor(ticks, device=device), torch.tensor(heads, device=device)
        for stream, (tick, head) in enumerate(zip(ticks.tolist(), heads.tolist())):
            if head not in (-1, 0):
                oracle[stream, tick] = head
        actual = captured(ticks, heads)
        assert actual.tolist() == [oracle.get((stream, tick), -1) for stream, tick in enumerate(ticks.tolist())]
        probes = [(stream, tick) for stream in range(-1, 6) for tick in range(-1, 10)]
        stream, tick = torch.tensor(probes, device=device).T
        assert state.root_at(stream, tick).tolist() == [oracle.get(pair, -1) for pair in probes]
    restored = ModelState().to(device)
    restored.load_state_dict(state.state_dict(), strict=True)
    assert restored.root_at(stream, tick).tolist() == state.root_at(stream, tick).tolist()
    restored.forget_heads(torch.tensor([31, 42], device=device), relations=True)
    restored.forget_heads(torch.tensor([23, 16], device=device), relations=False)
    kept = {key: value for key, value in oracle.items() if value in (31, 42, -25, -18)}
    assert restored.root_at(stream, tick).tolist() == [kept.get(pair, -1) for pair in probes]
    restored.publish_heads(4, [52, -28])
    assert restored.root_at(torch.tensor([0, 1]), 4).tolist() == [52, -28]


def test_temporal_publication_has_a_flat_batch_graph(device):
    from TemporalHeads import publish
    operations = []
    def backend(graph, inputs):
        operations.append(sum(node.op == 'call_function' for node in graph.graph.nodes))
        return graph.forward
    for batch in (4, 16):
        rows = torch.full((32, batch + 1), -1, device=device, dtype=torch.long)
        ticks = torch.arange(batch, device=device) % 3 + 6
        heads = -torch.arange(batch, device=device) - 3
        function = torch.compile(publish, backend=backend, fullgraph=True, dynamic=False)
        result = function(rows, ticks, heads)
        for b in range(batch):
            assert result[ticks[b] - 6, b + 1] == heads[b]
        assert result[3:].eq(-1).all()
    assert operations[0] == operations[1]


@pytest.mark.parametrize('sequential', [True, False])
def test_batched_relation_translation_preserves_stream_allocation_order(device, sequential):
    from Memory import OCCURRENCE, SYMBOL, RELATION
    from ReadingMemory import translate_batch
    operations = []
    def backend(graph, inputs):
        operations.append(sum(sum(node.op == 'call_function' for node in module.graph.nodes)
                              for module in graph.modules() if isinstance(module, torch.fx.GraphModule)))
        return graph.forward
    for batch in (4, 16):
        handles = torch.tensor([7, 9, 14], device=device).expand(batch, -1)
        active = torch.ones(batch, 3, device=device, dtype=torch.bool)
        active[2::3, 2] = False
        active[3::4] = False
        ends, roots = [], []
        for b in range(batch):
            ends.append([[(SYMBOL, 100+b % 2), (SYMBOL, 200)],
                         [(RELATION, 7), (OCCURRENCE, 1000+b)],
                         [(SYMBOL, 300), (RELATION, 9)]])
            roots.append((OCCURRENCE, 1000+b) if b % 4 == 3 else (RELATION, 9 if b % 3 == 2 else 14))
        endpoints, heads = torch.tensor(ends, device=device), torch.tensor(roots, device=device)
        refs = torch.zeros(96, 2, device=device, dtype=torch.long)
        tags = torch.zeros(96, 2, device=device, dtype=torch.uint8)
        refs[1] = torch.tensor([100, 200], device=device)
        tags[1] = SYMBOL
        blocked = tags[:, 0] != 0
        blocked[0] = True
        known = {((SYMBOL, 100), (SYMBOL, 200)): 1}
        next_id = 2
        expected_names = torch.zeros_like(handles)
        expected_pairs = torch.zeros_like(endpoints)
        expected_fresh = torch.zeros_like(active)
        expected_roots = []
        for b in range(batch):
            local = {}
            for c in range(3):
                if not bool(active[b, c]):
                    continue
                pair = tuple((kind, local.get(value, value) if kind == RELATION else value) for kind, value in ends[b][c])
                fresh = pair not in known
                if fresh:
                    known[pair] = next_id
                    next_id += 1
                name = known[pair]
                local[(7, 9, 14)[c]] = name
                expected_names[b, c] = name
                expected_pairs[b, c] = torch.tensor(pair, device=device)
                expected_fresh[b, c] = fresh
            kind, root = roots[b]
            expected_roots.append((kind, local.get(root, root) if kind == RELATION else root))
        function = torch.compile(lambda *args: translate_batch(*args, sequential=sequential),
                                 backend=backend, fullgraph=True, dynamic=False)
        actual = function(handles, endpoints, heads, active, refs, tags, blocked, refs.new_tensor(2))
        assert torch.equal(actual[0], expected_names)
        assert torch.equal(actual[1], expected_pairs)
        assert torch.equal(actual[2], expected_fresh)
        assert actual[3].tolist() == [list(root) for root in expected_roots]
        assert int(actual[4]) == next_id and bool(actual[5])
        for pair, name in known.items():
            assert tuple(zip(tags[name].tolist(), refs[name].tolist())) == pair
        # The same device loop and scalar allocation order for a wider batch.
    assert operations[0] == operations[1]


def test_sequential_pair_publication_matches_ordinary_writer(device):
    from Memory import LTM1, SYMBOL, RELATION
    from ReadingMemory import _publish_sequential_pairs
    result, expected = [LTM1(32).to(device) for _ in range(2)]
    for store in (result, expected):
        store.add((SYMBOL, 20), (SYMBOL, 21), symbol=2, truth_plus=.3, surprise=.45)
    refs, tags = result.refs.clone(), result.ref_kinds.clone()
    samples = [(3, (RELATION, 2), (SYMBOL, 22), True, .5, .1, .25),
               (2, (SYMBOL, 20), (SYMBOL, 21), False, .7, .2, .15),
               (3, (RELATION, 2), (SYMBOL, 22), False, .2, .6, .15),
               (4, (SYMBOL, 23), (RELATION, 3), True, .9, 0., .65),
               (2, (SYMBOL, 20), (SYMBOL, 21), False, .1, .8, .75)]
    names, fresh, observed = [], [], []
    for handle, left, right, new, plus, minus, activity in samples:
        expected.add(left, right, symbol=handle, truth_plus=plus, truth_minus=minus, surprise=activity)
        names.append(handle)
        fresh.append(new)
        observed.append([plus, minus, activity])
        tags[handle] = tags.new_tensor([left[0], right[0]])
        refs[handle] = refs.new_tensor([left[1], right[1]])
    class Allocator:
        next_id = 3

        def new_concept(self):
            self.next_id += 1

    allocator = Allocator()
    _publish_sequential_pairs(result, allocator, refs, tags, refs.new_tensor(5), refs.new_tensor(names),
                              torch.tensor(fresh, device=device), result.activity.new_tensor(observed))
    assert allocator.next_id == 5 and len(result) == len(expected)
    for column in (*result.columns, 'relation_order'):
        assert torch.equal(getattr(result, column), getattr(expected, column))
    assert result.find((SYMBOL, 23), (RELATION, 3)) == 4


def test_kept_output_uses_each_streams_path_and_retains_prior_packed_output(device):
    from Output import AnswerConstruction
    from WholePaths import keep_output
    def path(name, width):
        values = torch.arange(3 * width, device=device).reshape(3, width) + 10 * name
        answer = AnswerConstruction(values, None, values, values, values, (), tuple(str(name) for _ in values))
        return SimpleNamespace(construction=answer, output_trace=(values,))
    paths = [path(i, width) for i, width in enumerate([2, 4, 7])]
    previous = path(5, 3)
    selected = torch.tensor([1, 0, 0], device=device)
    slots = torch.tensor([[True], [False], [True]], device=device)
    model = SimpleNamespace(inputSpace=SimpleNamespace(_packed_sentence_slot_mask=slots),
                            _kept_path_output=previous.construction,
                            _kept_path_output_actions=previous.output_trace[0])
    keep_output(model, paths, selected, 0, active_winners=[1, -1, 0])
    assert model._kept_path_output.texts == ('1', '5', '0')
    assert model._kept_path_output.surface.tolist() == [[10, 11, 12, 13], [53, 54, 55, 0], [4, 5, 0, 0]]
    assert model._kept_path_output_actions.tolist() == [[10, 11, 12, 13], [53, 54, 55, -1], [4, 5, -1, -1]]


def test_retained_rows_keep_stream_names_local_and_translate_referent_chains(device):
    from Memory import OCCURRENCE, RELATION, SYMBOL
    from ReadingMemory import retained_rows
    operations = []

    def backend(graph, _inputs):
        operations.append(sum(node.op == 'call_function' for node in graph.graph.nodes))
        return graph.forward

    for batch in (4, 16):
        # The same local names designate different payload on every stream.
        names = torch.tensor([500, 3, 46], device=device).repeat(batch, 1)
        handles = names.new_tensor([9, 12]).repeat(batch, 1)
        endpoints = names.new_tensor([[(OCCURRENCE, 46), (SYMBOL, 53)],
                                      [(RELATION, 9), (OCCURRENCE, 46)]]).repeat(batch, 1, 1, 1)
        referents = names.new_tensor([(SYMBOL, 4), (SYMBOL, 5), (OCCURRENCE, 3)]).repeat(batch, 1, 1)
        heads = names.new_tensor([(RELATION, 12), (OCCURRENCE, 46),
                                  (OCCURRENCE, 500), (OCCURRENCE, 76)]).repeat(batch // 4, 1)
        shared = torch.arange(batch, device=device) % 4 == 3
        read = torch.compile(retained_rows, backend=backend, fullgraph=True, dynamic=False)
        args = (heads, handles, handles >= 0, endpoints, names, names >= 0, referents,
                names.new_tensor(1000), shared, names.new_tensor([76]), names.new_zeros(80))
        kept, relations, moved, refs, pairs, roots, valid = read(*args)
        assert valid
        first = 1001
        for b in range(batch):
            ideas = [[1, 2], [1, 2], [0], []][b % 4]
            assert kept[b].nonzero().flatten().tolist() == ideas
            assert relations[b].tolist() == [b % 4 == 0] * 2
            assert moved[b, ideas].tolist() == list(range(first, first + len(ideas)))
            if b % 4 < 2:
                assert refs[b, 2].tolist() == [OCCURRENCE, first]
            if b % 4 == 0:
                assert pairs[b, 1, 1].tolist() == [OCCURRENCE, first + 1]
            expected = [(RELATION, 12), (OCCURRENCE, first + 1), (OCCURRENCE, first), (OCCURRENCE, 76)]
            assert roots[b].tolist() == list(expected[b % 4])
            first += len(ideas)
    assert operations[0] == operations[1]
