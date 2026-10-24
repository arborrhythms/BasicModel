"""Paired-block grammar certificates; fixed operations, no learning claim."""
import copy

import pytest
import torch

from MeaningKernels import LaneKernel, MeaningAction, VerbGroups, centroid, split_centroid


@pytest.fixture
def paired(monkeypatch):
    from util import TheXMLConfig
    config = copy.deepcopy(TheXMLConfig.data)
    config.setdefault('architecture', {}).setdefault('data', {})['dataType'] = 'embedding'
    config.setdefault('PartSpace', {})['nDim'] = 4
    config.setdefault('ConceptualSpace', {})['nDim'] = 12
    monkeypatch.setattr(TheXMLConfig, '_data', config)


def test_centroid_is_leaf_weighted_and_splits_exactly():
    leaves = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 12
    pair = centroid((leaves[0], leaves[1]))
    whole = centroid((pair, leaves[2]), (2, 1))
    torch.testing.assert_close(whole, leaves.mean(0), atol=1e-15, rtol=0)
    torch.testing.assert_close(split_centroid(whole, pair, (1, 2)), leaves[2], atol=1e-15, rtol=0)


def test_an_explicit_numerical_width_does_not_inherit_a_paired_layout(paired):
    from Language import LiftLayer
    op = LiftLayer(3, 3)
    assert op.meaning_layout is None and not hasattr(op, 'meaning_action')
    a, b = torch.full((1, 3), .2), torch.full((1, 3), .3)
    folded = op.compose(a, b)
    left, right = op.generate(folded)
    torch.testing.assert_close(op.compose(left, right), folded, atol=1e-6, rtol=0)


def test_grammatical_heads_start_exact_and_lanes_are_independent(paired):
    from Language import ProductLayer, LowerLayer, SurfaceLayer, VerbLayer, PrepositionLayer
    from Layers import EqualLayer
    a = torch.linspace(.1, .9, 12)
    b = a.flip(0)
    for cls in (ProductLayer, LowerLayer, SurfaceLayer, VerbLayer, PrepositionLayer, EqualLayer):
        op = cls() if cls is EqualLayer else cls(nInput=12, nOutput=12)
        result = op.compose_blocks(op.compose, a, b, counts=(3, 1))
        torch.testing.assert_close(result[:4], (a[:4]*3+b[:4])/4)
        assert torch.equal(result[4:], (a, b)[op.meaning_head_role][4:])
        with torch.no_grad():
            op.meaning_kernel.context.fill_(.02)
        changed = b.clone()
        changed[4:8] = .3
        x = op.compose_blocks(op.compose, a, b)
        y = op.compose_blocks(op.compose, a, changed)
        assert torch.equal(x[8:], y[8:])


def test_meaning_inverse_and_gradient_at_identity(paired):
    from Language import SurfaceLayer
    op = SurfaceLayer(12, 12).double()
    a = torch.linspace(.15, .8, 12, dtype=torch.float64)
    b = a.flip(0)
    parent = op.compose_blocks(op.compose, a, b, counts=(2, 1))
    left, right = op.reverse_blocks(parent, b, counts=(2, 1))
    torch.testing.assert_close(left, a, atol=1e-12, rtol=0)
    assert torch.equal(right, b)
    parent[4:].sum().backward()
    assert op.meaning_kernel.context.grad.abs().sum() > 0
    with torch.no_grad():
        op.meaning_kernel.context.fill_(.01)
        op.meaning_kernel.log_scale.fill_(.05)
    parent = op.compose_blocks(op.compose, a, b, counts=(2, 1))
    left, _ = op.reverse_blocks(parent, b, counts=(2, 1))
    torch.testing.assert_close(left, a, atol=1e-12, rtol=0)


def test_other_catalogue_faces_use_the_same_centroid_contract(paired):
    from Language import (SynthesizeLayer, CompoundLayer, MinLayer, MaxLayer,
        IsEqualLayer, IsPartLayer, PartLayer, WholeLayer, AssertPartLayer, NullLayer)
    a, b = torch.linspace(.1, .8, 12), torch.linspace(.8, .2, 12)
    for cls in (SynthesizeLayer, CompoundLayer, MinLayer, MaxLayer, IsEqualLayer,
                IsPartLayer, PartLayer, WholeLayer, AssertPartLayer):
        op = cls()
        result = op.compose(a, b)
        assert torch.equal(result[4:], a[4:])
        torch.testing.assert_close(result[:4], (a[:4]+b[:4])*.5)
        assert tuple(op.named_parameters()) and all('meaning_kernel' in name for name, _ in op.named_parameters())
    op = NullLayer()
    assert torch.equal(op(a), a)
    with torch.no_grad():
        op.meaning_kernel.bias.fill_(.01)
    torch.testing.assert_close(op.reverse(op(a)), a, atol=1e-6, rtol=0)


def test_journal_leaf_counts_drive_the_structural_face(paired):
    from types import SimpleNamespace
    from Language import ProductLayer
    from MTMOverlay import MTMOverlay
    from test_identity_threading import choice
    overlay = MTMOverlay((SimpleNamespace(method_name='product', head_role=0),), ())
    op = ProductLayer(12, 12)
    values = torch.linspace(.1, .8, 36).reshape(3, 12)
    state = overlay.empty(values[None], words=3, journal=2)
    for index in (0, 1):
        state = overlay.push(state, torch.tensor([True]), values[index:index+1],
            word=torch.tensor([index+1]), order=torch.tensor([0]), index=torch.tensor(index),
            words=3, resolved=torch.tensor([True]))
    pair = op.compose(values[0:1], values[1:2])
    state = overlay.apply(state, choice(pair, kind=1), values[:2][None], column=torch.tensor(0), words=3)
    state = overlay.push(state, torch.tensor([True]), values[2:3], word=torch.tensor([3]),
        order=torch.tensor([0]), index=torch.tensor(2), words=3, resolved=torch.tensor([True]))
    counts = (state[2][:, 1, 6], state[2][:, 0, 6])
    assert (int(counts[0]), int(counts[1])) == (2, 1)
    result = op.compose_from_grammar_context((pair, values[2:3]),
        context=SimpleNamespace(leaf_counts=counts, action_context=None))
    torch.testing.assert_close(result[0, :4], values[:, :4].mean(0))


@pytest.mark.parametrize('compiled', (False, True))
def test_action_keeps_absence_and_inverts_on_each_lane(compiled):
    action = MeaningAction(8, capacity=2).double()
    subject = torch.tensor([.2, 0., .4, .7, 0., .3, .6, 0.], dtype=torch.float64)
    verb = torch.full((8,), .2, dtype=torch.float64)
    apply = torch.compile(action, backend='eager', fullgraph=True) if compiled else action
    assert torch.equal(apply(subject, verb), subject)
    apply(subject, verb).sum().backward()
    assert action.readout.grad.abs().sum() > 0
    with torch.no_grad():
        action.readout.fill_(.15)
        action.object_vectors[0, :, :] += .05
    obj = torch.tensor([.3, 0., .2, 0., .3, 0., .2, 0.], dtype=torch.float64)
    result = apply(subject, verb, object_value=obj)
    assert torch.equal(result[subject == 0], subject[subject == 0])
    assert torch.equal(result[[1, 3, 5, 7]], subject[[1, 3, 5, 7]])
    restored = apply(result, verb, object_value=obj, inverse=True)
    torch.testing.assert_close(restored, subject, atol=1e-12, rtol=0)


def test_disjoint_objects_confine_different_actions():
    action = MeaningAction(8, capacity=2).double()
    subject = torch.full((8,), .3, dtype=torch.float64)
    verb = torch.full((8,), .2, dtype=torch.float64)
    with torch.no_grad():
        action.readout.fill_(.15)
    one = torch.tensor([1., 1., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)
    two = one.roll(2)
    a = action(subject, verb, object_value=one)
    b = action(subject, verb, object_value=two)
    assert not torch.equal(a, b)
    assert torch.equal(a[[2, 3, 6, 7]], subject[[2, 3, 6, 7]])
    assert torch.equal(b[[0, 1, 4, 5]], subject[[0, 1, 4, 5]])


def test_group_fit_recurrence_join_capacity_and_checkpoint():
    groups = VerbGroups(4, capacity=1)
    for row in range(2):
        assert groups.observe(11, ('ltm', 'unit', row), [1.], 1.)['group'] == -1
    # Repeating a witness does not mint a group.
    assert groups.observe(11, ('ltm', 'unit', 1), [1.], 1.)['group'] == -1
    assert groups.observe(11, ('ltm', 'unit', 2), [1.], 1.)['group'] == 0
    for row in range(3):
        report = groups.observe(12, ('ltm', 'unit', row+3), [.01], 1.)
    assert report['group'] == 0 and not report['capacity']
    for row in range(3):
        report = groups.observe(13, ('ltm', 'unit', row+6), [1.], 1.)
    assert report['capacity'] and report['group'] == 0
    restored = VerbGroups(4, capacity=1)
    restored.load_state_dict(groups.state_dict())
    assert restored.assignments == groups.assignments
    torch.testing.assert_close(restored(torch.tensor(0)), torch.eye(4), atol=0, rtol=0)


def test_free_search_scores_meaning_and_centroid_and_trains_inverse(paired):
    from Language import LanguageSpace, ProductLayer
    op = ProductLayer(12, 12)
    a = torch.tensor([.2, .4, .2, .4, .2, .3, .4, .2, .2, .3, .4, .2])
    b = torch.tensor([.4, .2, .4, .2, .6, .6, .6, .6, .6, .6, .6, .6])
    wrong = a.clone()
    wrong[4:] = .8
    parent = op.compose_blocks(op.compose, a, b, counts=(2, 1))[None].detach()
    bank = torch.stack((wrong, a, b))[None]
    left, right, valid, details = LanguageSpace._bounded_binary_reconstruction(op, parent,
        torch.zeros_like(parent), torch.tensor([True]), torch.tensor([False]),
        bank, torch.ones(1, 3, dtype=torch.bool), 3, return_details=True)
    assert valid.all()
    torch.testing.assert_close(left, a[None])
    torch.testing.assert_close(right, b[None])
    assert details['relative_residual'][0, 0, 2] > details['relative_residual'][0, 1, 2]
    (left[..., 4:] - .6).square().sum().backward()
    assert op.meaning_kernel.context.grad.abs().sum() > 0


def test_versioned_swap_drops_form_parameters_and_keeps_new_checkpoints(paired):
    from Language import LiftLayer, SurfaceLayer
    from GrammarCheckpoint import align, catalog
    model = torch.nn.ModuleDict(dict(lift=LiftLayer(12, 12), surface=SurfaceLayer(12, 12)))
    current = model.state_dict()
    assert not any('_sigma' in key or 'marker_map' in key for key in current)
    old = dict(current)
    old['lift._sigma.layer.weight'] = torch.eye(12)
    old['surface.marker_map.weight'] = torch.eye(12)
    previous_catalog = dict(catalog(model))
    previous_catalog.pop('meaning_kernel_schema')
    state, drift, transforms = align(model, old, previous_catalog)
    assert 'lift._sigma.layer.weight' not in state and 'surface.marker_map.weight' not in state
    assert drift['lift._sigma.layer.weight']['action'] == 'retired form kernel'
    assert transforms['surface.marker_map.weight'] is None
    model.load_state_dict(state)
    restored, drift, _ = align(model, state, catalog(model))
    assert not drift
    model.load_state_dict(restored)


def test_direct_paired_reverses_reject_a_retired_split(paired):
    from Language import SynthesizeLayer, CompoundLayer
    from Layers import EqualLayer
    a, b = torch.linspace(.15, .75, 12), torch.linspace(.8, .2, 12)
    for cls in (SynthesizeLayer, CompoundLayer, EqualLayer):
        op = cls()
        result = op.compose(a, b)
        with pytest.raises(NotImplementedError, match='candidate search'):
            op.reverse(result)
        left, right = op.reverse_blocks(result, b)
        torch.testing.assert_close(left, a)
        torch.testing.assert_close(right, b)


def test_unknown_basis_eager_and_fullgraph_have_the_same_action_pullback():
    action = MeaningAction(8, capacity=2).double()
    with torch.no_grad():
        action.readout.fill_(.1)
    before = torch.linspace(.1, .8, 16, dtype=torch.float64).reshape(2, 8).requires_grad_()
    verb = before.flip(-1).detach().requires_grad_()
    obj = before.detach().clone().requires_grad_()
    def run(value, operator, context):
        return action(value, operator, object_value=context,
            object_present=torch.zeros(2, dtype=torch.bool), group=torch.full((2,), -1))
    expected = torch.compile(run, backend='eager', fullgraph=True)(before, verb, obj)
    actual = run(before, verb, obj)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (before, verb, obj, *action.parameters())
    a = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    b = torch.autograd.grad(expected.sum(), inputs, allow_unused=True)
    for tensor, x, y in zip(inputs, a, b):
        torch.testing.assert_close(torch.zeros_like(tensor) if x is None else x,
                                   torch.zeros_like(tensor) if y is None else y, rtol=1e-9, atol=1e-9)
