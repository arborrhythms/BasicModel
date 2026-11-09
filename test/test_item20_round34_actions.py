"""Participation re-reads, rates and clamp velocity certificates."""
from types import SimpleNamespace
import torch
from MeaningKernels import MeaningAction
from Memory import MediumTermMemory
from test_item20_round34_projection import record


def seat(*, recorded=True, alternative=False):
    before = torch.tensor([.2, .3, .4, .5, .6, .3, .2, .4], requires_grad=True)
    phrase = torch.full((8,), .3, requires_grad=True)
    value = torch.linspace(.15, .85, 8, requires_grad=True)
    action, modifier = MeaningAction(8, capacity=2), MeaningAction(8, modifier=True)
    pool = MediumTermMemory()
    key = ('ltm', 'reread', 1)
    pool.open(key, kind=11, when=1, value=before)
    timeline = pool.alternative(key, when=2) if alternative else 0
    pool.event(key, ('ltm', 'reread', 2), verb=12, role=0, when=2, timeline=timeline,
        before=before, basis=-1, verb_meaning=phrase if recorded else None,
        modifiers=(phrase,) if alternative else ())
    pool.update(key, value, when=3, timeline=timeline)
    language = SimpleNamespace(verb_action=lambda:action, modifier_action=lambda:modifier)
    owner = SimpleNamespace(mtm=pool, similarity_codebook=SimpleNamespace(mereology=None),
        _model=SimpleNamespace(languageSpace=language))
    store = SimpleNamespace(location=lambda occurrence:None)
    return owner, store, key, timeline, action, modifier, (before, phrase, value)


def test_later_answer_reads_exact_value_and_only_action_parameters():
    from PrependedSymbols import symbolize
    results = []
    for alternative in (False, True):
        owner, store, key, timeline, action, modifier, sources = seat(alternative=alternative)
        value = symbolize(owner, store, key, owner.mtm.entries[key], timeline)
        saved = owner.mtm.value_of(key, timeline=timeline)
        assert torch.equal(value, saved)
        (value - .1).square().sum().backward()
        assert action.readout.grad.abs().sum() > 0
        if alternative:
            assert modifier.readout.grad.abs().sum() > 0
        assert all(source.grad is None for source in sources)
        assert not saved.requires_grad
        results.append(dict(alternative=alternative, exact=True, stored=saved.tolist(),
            lift_gradient=float(action.readout.grad.norm()),
            adverb_gradient=None if modifier.readout.grad is None else float(modifier.readout.grad.norm()),
            input_gradients=False))
    record('reread', results)


def test_missing_recorded_input_stays_detached_and_legacy_event_loads():
    from PrependedSymbols import symbolize
    owner, store, key, timeline, *_ = seat(recorded=False)
    value = symbolize(owner, store, key, owner.mtm.entries[key], timeline)
    assert torch.equal(value, owner.mtm.value_of(key)) and not value.requires_grad
    saved = owner.mtm.state()
    saved['version'] = 10
    for row in saved['entries']:
        row['timelines'] = tuple((tuple(e[:-1] for e in events), value) for events, value in row['timelines'])
    copy = MediumTermMemory()
    copy.restore(saved)
    assert copy.entries[key].events[-1].verb_meaning is None
    record('reread-no-input', dict(exact=True, requires_grad=value.requires_grad, legacy_version=10))


def test_third_failed_fit_uses_phrase_and_assignment_is_training_only():
    from VerbLearning import observe
    action = MeaningAction(8, capacity=2, fit=.0001)
    with torch.no_grad():
        action.readout.fill_(.3)
    pool = MediumTermMemory()
    key = ('ltm', 'assign', 1)
    before, after, phrase, word = (torch.full((8,), v) for v in (.2, .7, .6, .05))
    pool.open(key, kind=11, when=1, value=before)
    reports = []
    for index in range(3):
        pool.event(key, ('ltm', 'assign', index+2), verb=12, role=0, when=index+2,
            before=before, basis=-1, verb_meaning=phrase)
        previous = dict(pool.entries)
        pool.update(key, after, when=index+3)
        report = observe(action, pool, previous, {key})[0]
        reports.append(report)
        assert (12 in action.groups.assignments) == (index == 2)
    with torch.no_grad():
        word_error = float(action.teacher(before, after, word, group=torch.tensor(-1)))
        phrase_error = float(action.teacher(before, after, phrase, group=torch.tensor(-1)))
    assert reports[0]['error'] == phrase_error and word_error != phrase_error
    before_groups = action.groups.get_extra_state().clone()
    pool.event(key, ('ltm', 'assign', 6), verb=13, role=0, when=6,
        before=before, basis=-1, verb_meaning=phrase)
    observe(action, pool, dict(pool.entries), {key}, training=False)
    assert torch.equal(action.groups.get_extra_state(), before_groups)
    record('phrase-assignment', dict(word_residual=word_error, phrase_residual=phrase_error,
        groups=[r['fit']['group'] for r in reports], no_evaluation_assignment=True))


def test_action_rates_and_realized_relative_steps():
    from ActionLearning import rate_groups
    reports = []
    for ratio in (1., .1):
        code = torch.nn.Parameter(torch.ones(1))
        lift, adverb = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            lift.weight.fill_(1); adverb.weight.fill_(1)
        language = SimpleNamespace(verb_action=lambda:lift, modifier_action=lambda:adverb)
        parameters = [code, lift.weight, adverb.weight]
        optimizer = torch.optim.SGD(rate_groups(parameters, language, ratio, .1), lr=.1, momentum=.9)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        steps = [float((p.detach()-1).abs().sum()) for p in parameters]
        torch.testing.assert_close(torch.tensor(steps), torch.tensor([.1, .1*ratio, .1*ratio**2]))
        reports.append(dict(kappa=ratio, relative_steps=steps))
    record('action-rates', reports)


def test_clamped_outward_velocity_does_not_block_first_inward_gradient():
    from PerceptProperties import PrimitiveProperties
    from Optimizer import RowLocalSGD
    reports = []
    for compact in (False, True):
        for repaired in (False, True):
            p = torch.nn.Parameter(torch.tensor([[1., 0., .5]]))
            optimizer = RowLocalSGD([p], lr=.1) if compact else torch.optim.SGD([p], lr=.1, momentum=.9)
            optimizer.state[p]['momentum_buffer'] = torch.tensor([[-2., 2., -.5]])
            owner = SimpleNamespace(members=p, fixed_rows=torch.tensor([False]), fixed_members=p.detach().clone())
            PrimitiveProperties.project(owner, optimizer=optimizer if repaired else None)
            velocity = optimizer.state[p]['momentum_buffer'].clone()
            gradient = torch.tensor([[1., -1., 0.]])
            p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), gradient, p.shape) if compact else gradient
            optimizer.step()
            PrimitiveProperties.project(owner, optimizer=optimizer if repaired else None)
            assert float(p[0,0]) < 1 if repaired else float(p[0,0]) == 1
            assert float(p[0,1]) > 0 if repaired else float(p[0,1]) == 0
            assert velocity[0,2] == -.5
            reports.append(dict(compact=compact, repaired=repaired, velocity=velocity.tolist(), after=p.detach().tolist()))
    record('clamp-velocity', reports)
