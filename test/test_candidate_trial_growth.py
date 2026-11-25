"""Speculative learning belongs to one trial until the sentence chooses its rows."""
from types import SimpleNamespace

import pytest
import torch

import CandidateGrowth as growth


def test_provisional_sigma_serves_later_reads_without_registering_a_parameter():
    from test_item17_symbols import _small_space
    owner, store, ids, rows = _small_space()
    root, first, second, kind, _ = ids
    owner.nominate_symbol(('source', first), first)
    owner.nominate_symbol(('source', second), second)
    owner.kind_entry(kind, first)
    weight = store.values
    optimizer = torch.optim.Adam([weight], lr=.01)
    edges = store._index.copy()
    sibling, discarded = growth.GrowthTrial(owner), growth.GrowthTrial(owner)
    with discarded.active():
        assert discarded.promote(kind, second) == rows[3]
        assert owner.reads_concept(rows[3], rows[2])
        projection = growth.visible_store(store)
        assert projection.nnz == store.nnz + 1
        local_cost = projection.values.square().sum()
        assert list(store.parameters())[0] is weight
        assert not optimizer.state
    with sibling.active():
        assert not owner.reads_concept(rows[3], rows[2])
        assert growth.visible_store(store) is store
    assert store._index == edges and store.values is weight
    assert torch.equal(torch.autograd.grad(local_cost, weight)[0], 2 * weight)
    assert weight.grad is None
    assert optimizer.param_groups[0]['params'] == [weight]


def test_mixed_winners_replay_observations_in_look_then_stream_order(monkeypatch):
    from test_item17_symbols import _small_space
    owner, _, _, _ = _small_space()
    left, right = growth.GrowthTrial(owner), growth.GrowthTrial(owner)
    def observation(value):
        return ({value: .5}, {value: 1.}, [(value, .25)])
    left.looks = [{0: observation(1), 1: observation(2)}, {0: observation(3)}]
    right.looks = [{0: observation(4), 1: observation(5)}, {0: observation(6), 1: observation(7)}]
    actual = []
    monkeypatch.setattr(growth, 'pi_step', lambda owner, sentence: actual.append(('pi', sentence)))
    monkeypatch.setattr(growth, 'sigma_step', lambda owner, present, expected:
                        actual.append(('sigma', present, expected)))
    growth.GrowthTrial.commit([left, right], torch.tensor([1, 0]))
    assert actual == [event for cid in (4, 2, 6) for event in
                      [('pi', {cid: 1.}), ('sigma', {cid: .5}, [(cid, .25)])]]


def test_trial_restores_on_error_and_retained_forecast_only_has_winner_gradients():
    from test_item17_symbols import _small_space
    owner, _, _, _ = _small_space()
    owner._expected_knowing = torch.zeros(2, 16, 2)
    model = SimpleNamespace(_expectation_looks=[], _expectation_percept_looks=[])
    opening = owner._expected_knowing.clone()
    left, right = growth.GrowthTrial(owner), growth.GrowthTrial(owner)
    costs = [torch.tensor([2., 3.], requires_grad=True), torch.tensor([5., 7.], requires_grad=True)]
    for index, trial in enumerate((left, right)):
        with trial.active(model, 2):
            assert torch.equal(owner._expected_knowing, opening)
            owner._expected_knowing.fill_(index + 1)
            model._expectation_looks.append((costs[index], torch.tensor([True, True])))
        assert torch.equal(owner._expected_knowing, opening)
        assert model._expectation_looks == []
    with pytest.raises(ValueError):
        with growth.GrowthTrial(owner).active(model, 2):
            owner._expected_knowing.fill_(99)
            raise ValueError
    assert torch.equal(owner._expected_knowing, opening)
    growth.GrowthTrial.keep_forecasts(model, (left, right), torch.tensor([1, 0]))
    assert torch.equal(owner._expected_knowing[:, 0, 0], torch.tensor([2., 1.]))
    loss = sum(cost.sum() for cost, mask in model._expectation_looks)
    left_grad, right_grad = torch.autograd.grad(loss, costs)
    assert left_grad.tolist() == [0., 1.]
    assert right_grad.tolist() == [1., 0.]


def test_mixed_streams_read_their_own_promotions_and_compact_live_rows():
    from test_item17_symbols import _small_space
    from FieldProjection import close
    owner, store, ids, rows = _small_space()
    _, first, second, kind, _ = ids
    owner.nominate_symbol(('source', first), first)
    owner.nominate_symbol(('source', second), second)
    owner.kind_entry(kind, first)
    trial = growth.GrowthTrial(owner)
    trial.streams[1] = private = growth.GrowthTrial(owner)
    with private.active():
        private.promote(kind, second)
    evidence = torch.zeros(5, 3, 1, 2)
    evidence[2, :, 0, 0] = .7
    evidence.requires_grad_()
    with trial.active():
        visible = close(owner, rows, evidence)[3, :, 0, 0]
        live = torch.tensor([False, True, True])
        compact = growth.read_streams(owner, lambda: torch.full((2,), growth.visible_store(store).nnz),
                                      axis=0, live=live)
    with private.active():
        reference = close(owner, rows, evidence[:, 1:2])[3, 0, 0, 0]
    assert reference > 0
    assert visible.tolist() == [0., reference.item(), 0.]
    assert compact.tolist() == [store.nnz + 1, store.nnz]
    gradient, = torch.autograd.grad(visible.sum(), evidence)
    reference_gradient, = torch.autograd.grad(reference, evidence)
    assert reference_gradient[2, 1, 0, 0] > 0
    assert torch.equal(gradient[2], reference_gradient[2])


def test_rejected_trial_preserves_an_existing_adam_moment():
    from ConceptSpread import loss
    from test_item17_symbols import _small_space
    owner, _, _, _ = _small_space()
    owner._expected_knowing = torch.zeros(1, 16, 2)
    model = SimpleNamespace(_expectation_looks=[], _expectation_percept_looks=[])
    kept, rejected = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.Adam([kept, rejected], lr=.01)
    (kept + rejected).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in (kept, rejected)]
    state = {key: value.clone() for key, value in optimizer.state[rejected].items()}
    assert state['exp_avg'].abs().sum() > 0

    trials = [growth.GrowthTrial(owner), growth.GrowthTrial(owner)]
    for trial, cost in zip(trials, (kept * 0., rejected.square()), strict=True):
        with trial.active(model, 1):
            model._expectation_looks.append((cost, torch.ones_like(cost, dtype=torch.bool)))
    growth.GrowthTrial.keep_forecasts(model, trials, torch.tensor([0]))
    loss(model).sum().backward()
    assert torch.equal(kept.grad, torch.zeros_like(kept)) and rejected.grad is None
    optimizer.step()
    assert not torch.equal(kept, before[0])
    assert torch.equal(rejected, before[1])
    for key, value in state.items():
        assert torch.equal(optimizer.state[rejected][key], value)


def test_each_fixation_path_grades_its_own_requests_against_the_shared_page():
    from Layers import Error
    from LookForecast import OccurrenceCredit
    greedy = OccurrenceCredit.__new__(OccurrenceCredit)
    greedy.observed = torch.ones(2, 3)
    greedy.terms = []
    explore = greedy.fork()
    assert explore.observed is greedy.observed
    costs = [torch.tensor([.1, .2], requires_grad=True), torch.tensor([.3, .4], requires_grad=True)]
    winners = torch.tensor([1, 0])
    registry = Error(row_mask=torch.ones(2, dtype=torch.bool))
    for i, path in enumerate((greedy, explore)):
        path.terms.append((costs[i], torch.ones(2), torch.ones(2, dtype=torch.bool)))
        path.for_rows(winners == i).register(registry)
    loss = Error._value(registry._terms['attention.next_observation']).sum()
    left, right = torch.autograd.grad(loss, costs)
    assert left[0] == right[1] == 0
    assert left[1] == right[0] > 0
def test_mtm_relation_slots_match_ordered_admission_eviction_and_promotion():
    from CandidateMatrix import CandidateMatrix, substitution_gradient
    from Memory import MTM1
    native = MTM1(3, strength=.2, hold=3).double()
    reference = CandidateMatrix(3, strength=.2, hold=3)
    for offers, expected, things in (
            ([(0, 4), (1, 4), (2, 5), (3, 6)], {0: .8, 1: .7, 3: .75}, {4: 1., 5: .5, 6: 1.}),
            ([(1, 4), (3, 6)], {1: .7, 3: .75}, {4: 1., 6: 1.}),
            ([(4, 5), (5, 6)], {1: .7, 3: .75, 4: .8}, {4: 1., 5: .75, 6: 1.}),
            ([], {}, {5: 1., 6: 1.}),
            ([], {}, {}),
            ([(2, 4)], {2: .4}, {4: 1.}),
            ([], {}, {4: 1.})):
        for key in offers:
            reference.offer(key)
        gradients = {key: substitution_gradient(expected.get(key[0], 0.), 0., things[key[1]])
                     for key in reference.entries if key[1] in things}
        promoted, _ = reference.step(gradients)
        result = native.observe(native.refs.new_tensor(offers).reshape(-1, 2),
            native.refs.new_tensor(list(expected)), native.activity.new_tensor(list(expected.values())),
            native.refs.new_tensor(list(things)), native.activity.new_tensor(list(things.values())))
        assert [tuple(pair) for pair in result.tolist() if pair[0] >= 0] == [key for key, _ in promoted]
        assert native.counts() == reference.counts()
        assert native.diagnostic_entries() == {str(k): vars(v) for k, v in reference.entries.items()}
    private = native.fork()
    private.activity.fill_(.8)
    assert not torch.equal(native.activity, private.activity)


@pytest.mark.parametrize('strength', [.2, .125])
def test_float32_relation_pool_preserves_the_promotion_boundary(strength):
    from Memory import MTM1
    pool = MTM1(3, strength=strength, hold=1).float()
    midpoint = pool.activity.new_tensor(2 * strength)
    levels = torch.stack((midpoint.nextafter(torch.zeros_like(midpoint)), midpoint,
                          midpoint.nextafter(torch.ones_like(midpoint))))
    keys = pool.refs.new_tensor([[1, 4], [2, 5], [3, 6]])
    result = pool.observe(keys, keys[:, 0], levels, keys[:, 1], torch.full_like(levels, .5))
    expected = [tuple(key) for key, level in zip(keys.tolist(), levels.tolist(), strict=True)
                if level * .5 > strength]
    assert [tuple(key) for key in result.tolist() if key[0] >= 0] == expected


def test_distinct_observed_nominations_match_serial_eviction():
    import random
    from Memory import MTM1
    rng = random.Random(108)
    reference = MTM1(4, strength=.2).double()
    batched = reference.fork()
    kinds = torch.arange(3)
    symbols = torch.arange(5)
    for _ in range(100):
        nominations = rng.sample([(k, s) for k in range(3) for s in range(5)], rng.randrange(1, 15))
        offers = torch.tensor([pair if rng.random() < .8 else (-1, -1) for pair in nominations])
        forecast = torch.tensor(rng.choices([0., .25, .75], k=3), dtype=torch.float64)
        values = torch.tensor(rng.choices([.5, .75, 1.], k=5), dtype=torch.float64)
        args = offers, kinds, forecast, symbols, values
        expected = reference.observe(*args)
        assert torch.equal(batched.observe(*args, read_pairs=True), expected)
        for name in ('refs', 'ref_kinds', 'entered_at', 'activity', 'held', 'read_count', 'totals', 'count'):
            assert torch.equal(getattr(batched, name), getattr(reference, name)), name


def test_departures_inherit_one_opening_in_independent_mtm_slots(monkeypatch):
    from test_item17_symbols import _small_space
    owner, _, _, _ = _small_space()
    owner.mtm.reserve_relation_trials(1, 4, owner.similarity_codebook.W.device)
    model = SimpleNamespace(explore_paths=2)
    observed = []
    def look(*_):
        trial = growth._current_trial.get()
        observed.append(trial.path)
        trial.looks.append({0: ('opening',)})
        stream = growth.GrowthTrial(owner, path=trial.path, stream=0)
        trial.streams[0] = stream
        with stream.active():
            pool = growth.matrices(owner)['sigma']
            pair = pool.refs.new_tensor([[2, 4]])
            pool.observe(pair, pair[:, 0], pool.activity.new_ones(1), pair[:, 1], pool.activity.new_ones(1))
    monkeypatch.setattr(growth, 'at_look', look)
    greedy = growth.GrowthTrial(owner, path=0)
    with greedy.active():
        opening = growth.at_opening(model, None, None, None)
    slots = [opening.streams[0].candidates['sigma'], greedy.streams[0].candidates['sigma']]
    slots[1].held.add_(5)
    for path in (1, 2):
        child = growth.GrowthTrial(owner, path=path)
        with child.active():
            assert growth.at_opening(model, None, None, None, prior=opening) is opening
        assert child.looks == [{0: ('opening',)}]
        held = child.streams[0].candidates['sigma']
        assert torch.equal(held.held, slots[0].held) and int(held.held[1]) == 1
        held.held.add_(path)
        slots.append(held)
    assert observed == [0]
    assert len({pool.refs.data_ptr() for pool in slots}) == 4
