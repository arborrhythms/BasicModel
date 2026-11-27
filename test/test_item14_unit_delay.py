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
                state = owner.__dict__['_percept_delay_state']
                return float(state['previous'][rows[name][1], 0, 0])
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
    from test_item16_naming import naming_model
    model = naming_model(tmp_path, monkeypatch)
    try:
        owner = model._concept_owner()
        assert not owner.percept_delay
        assert owner.install_order_zero().nInput == int(owner.basic_percepts.capacity)
    finally:
        model.End()
