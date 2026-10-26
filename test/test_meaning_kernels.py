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


def test_scalar_group_lookup_inside_nested_decoder_control_flow():
    groups = VerbGroups(64)
    value = torch.ones(64)

    def run(value):
        def cond(index, current):
            return index < 2

        def body(index, current):
            def chosen(parent):
                group = torch.zeros((), dtype=torch.long, device=parent.device)
                return groups(group) @ parent
            update = torch.cond(index == 0, chosen, lambda parent: parent.clone(), (current,))
            return index + 1, update
        return torch.while_loop(cond, body, (torch.tensor(0), value))[1]

    torch.testing.assert_close(run(value), value, rtol=0, atol=0)


def test_inverse_action_compiles_through_nested_control_flow_with_its_gradient():
    from CompilerEffects import compile_inductor
    action = MeaningAction(4).double()
    # The loop's reverse pass carries a basis gradient with this layout.
    # Give the fixture's parameter the same strides, as while_loop requires.
    action.object_vectors = torch.nn.Parameter(
        action.object_vectors.permute(2, 0, 1).contiguous().permute(1, 2, 0))
    with torch.no_grad():
        action.readout.fill_(.3)
        action.object_vectors[0, 1, 0] = .2
    value = torch.tensor([[.2, .4, .3, .6]], dtype=torch.float64, requires_grad=True)
    verb = torch.tensor([[.3, .6, .2, .4]], dtype=torch.float64, requires_grad=True)

    def chosen(current, context):
        return action(current, context, inverse=True, object_value=context,
                      object_present=torch.ones(1, dtype=torch.bool))

    def run(value, verb):
        def body(index, current):
            # Both branches use the parameter with the same gradient layout;
            # the unchosen branch must not introduce unrelated zero strides.
            result = torch.cond(index == 0, chosen, lambda x, y: chosen(x, y) * .5,
                                (current, verb))
            return index + 1, result
        return torch.while_loop(lambda i, x: i < 1, body, (torch.tensor(0), value))[1]

    expected = chosen(value, verb)
    leaves = (value, verb, action.readout, action.object_vectors)
    expected_grad = torch.autograd.grad(expected.sum(), leaves)
    actual = torch.compile(run, backend=compile_inductor, fullgraph=True)(value, verb)
    actual_grad = torch.autograd.grad(actual.sum(), leaves)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for found, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(found, wanted, atol=1e-9, rtol=1e-9)


def test_inverse_action_reports_an_unsuccessful_solve(monkeypatch):
    action = MeaningAction(4)
    solve = torch.linalg.solve_ex
    def unsuccessful(*args, **kwargs):
        value, info = solve(*args, **kwargs)
        return value, torch.ones_like(info)
    monkeypatch.setattr(torch.linalg, 'solve_ex', unsuccessful)
    with pytest.raises(RuntimeError, match='inverse is singular'):
        action(torch.full((1, 4), .3), torch.full((1, 4), .4), inverse=True,
               object_value=torch.full((1, 4), .2), object_present=torch.ones(1, dtype=torch.bool))


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


def test_given_parent_trains_the_meaning_inverse_without_a_form_kernel(paired):
    from Language import SurfaceLayer
    op = SurfaceLayer(12, 12).double()
    a = torch.linspace(.15, .8, 12, dtype=torch.float64)
    b = a.flip(0)
    parent = op.compose_blocks(op.compose, a, b).detach()
    left, _ = op.reverse_blocks(parent, b)
    parameters = tuple(op.meaning_kernel.parameters())
    gradients = torch.autograd.grad(left[4:].sum(), parameters, allow_unused=True)
    assert any(value is not None and bool(value.abs().any()) for value in gradients)
    torch.testing.assert_close(left[:4], a[:4], rtol=0, atol=1e-15)


def test_other_catalogue_faces_use_the_same_centroid_contract(paired):
    from Language import (SynthesizeLayer, CompoundLayer, MinLayer, MaxLayer,
        IsEqualLayer, IsPartLayer, PartLayer, WholeLayer, AssertPartLayer, NullLayer)
    a, b = torch.linspace(.1, .8, 12), torch.linspace(.8, .2, 12)
    for cls in (SynthesizeLayer, CompoundLayer, MinLayer, MaxLayer, IsEqualLayer,
                IsPartLayer, PartLayer, WholeLayer, AssertPartLayer):
        op = cls()
        result = op.compose(a, b)
        assert torch.equal(result[4:], b[4:] if cls is CompoundLayer else a[4:])
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
    from chunk_fixtures import Walk,choice
    from ChunkJournal import LEAVES
    walk=Walk((SimpleNamespace(method_name='product',head_role=0),),())
    op=ProductLayer(12,12)
    values=torch.linspace(.1,.8,36).reshape(3,12)
    state=walk.empty(values[:1],2)
    for index in (0,1):
        state=walk.shift(state,values[index:index+1],word=index+1,order=0,index=index)
    pair=op.compose(values[0:1],values[1:2])
    state=walk.apply(state,choice(pair,head=0),values[:2][None],0)
    state=walk.shift(state,values[2:3],word=3,order=0,index=2)
    counts=(state[1][:,1,LEAVES],state[1][:,0,LEAVES])
    assert (int(counts[0]),int(counts[1])) == (2,1)
    result=op.compose_from_grammar_context((pair,values[2:3]),
        context=SimpleNamespace(leaf_counts=counts,action_context=None))
    torch.testing.assert_close(result[0,:4],values[:,:4].mean(0))



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
        if cls is CompoundLayer:
            with pytest.raises(NotImplementedError, match='candidate search'):
                op.reverse_blocks(result, b)
            continue
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


@pytest.mark.parametrize('as_argument', [False, True])
def test_paired_lower_closes_its_composed_value_without_aliasing_the_description(monkeypatch, tmp_path, as_argument):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from Language import LowerLayer
    from identity_experiment import make_model
    model = make_model(tmp_path, monkeypatch)
    try:
        owner = model._concept_owner()
        store = owner._closed_clause_store()
        width = owner.similarity_codebook.W.shape[-1]
        op = LowerLayer(width, width)
        with torch.no_grad():
            op.meaning_kernel.context.fill_(.015)
        points = (torch.linspace(.1, .6, width), torch.linspace(.7, .2, width))
        terms = []
        for value in points:
            symbol = owner.new_concept()
            seat = owner._csw_concept_row(0, symbol)
            with torch.no_grad():
                owner.similarity_codebook.W[seat].copy_(value)
            terms.append(Composition(value, reference=symbol))
        point = op.compose(*points).detach()
        lower = Composition(point, tuple(terms), operation='lower', formal_name='lower',
                            head_role=2, order=1)
        root = (Composition(point, (lower, terms[0]), operation='verb', formal_name='verb',
                            head_role=1) if as_argument else lower)
        clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                        composition=root, order=root.order)
        written = {}
        row = store.write_clause(clause, written_rows=written)
        # The landed layout compresses resolved composition into one idea.
        # The composed value remains its own point, not a kind/Part alias.
        assert isinstance(row,int) and len(store.parts) == 0
        torch.testing.assert_close(store.slots[row,:store.when_start],root.point[:store.when_start],rtol=0,atol=0)
        assert written[id(clause)] == row
        assert not torch.equal(point, points[1])
    finally:
        model.End()


def test_selected_inverse_matches_full_menu_values_and_pullbacks(paired):
    from types import MethodType, SimpleNamespace
    from DecompositionChooser import DecompositionChooser
    from Generative import inverse_menu, selected_inverse
    from Language import LanguageSpace, SurfaceLayer, LowerLayer, NullLayer
    binary = (SurfaceLayer(12, 12).double(), LowerLayer(12, 12).double())
    unary = (NullLayer().double(),)
    for index, op in enumerate((*binary, *unary), 1):
        with torch.no_grad():
            op.meaning_kernel.log_scale.fill_(.01 * index)
            op.meaning_kernel.bias.fill_(.005 * index)
    language = SimpleNamespace(_generate_binary_ops=binary, _generate_unary_ops=unary,
                              decomposition_chooser=DecompositionChooser())
    for name in ('reverse_binary_step', '_reverse_of_binary_op', '_finish_binary_inverse',
                 'generate_unary_step'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    words = torch.tensor([[.1,.2,.3,.4,.1,.2,.3,.4,.5,.6,.7,.8],
                          [.7,.6,.5,.4,.2,.3,.4,.5,.6,.7,.8,.9],
                          [.3,.1,.7,.2,.4,.1,.6,.2,.3,.6,.5,.8]], dtype=torch.float64)
    parent = torch.stack((binary[0].compose(words[0], words[1]),
                          binary[1].compose(words[0], words[1]),
                          unary[0].compose(words[2]), words[0])).detach().requires_grad_()
    basis = words[None].expand(4, -1, -1).clone().requires_grad_()
    valid = torch.ones(4, dtype=torch.bool)
    choices = torch.arange(4)
    options = dict(basis=basis, basis_valid=torch.ones(4, 3, dtype=torch.bool))
    lefts, rights, legal = inverse_menu(language, parent, valid, **options)
    expected = (lefts[torch.arange(4), choices], rights[torch.arange(4), choices])
    actual = selected_inverse(language, parent, choices, valid, **options)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    parameters = (parent, basis, *(p for op in (*binary, *unary) for p in op.parameters()))
    weights = torch.linspace(.1, .9, 48, dtype=torch.float64).reshape(4, 12)
    gradients = [torch.autograd.grad((a*weights+b*weights.flip(-1)).sum(), parameters,
                                    allow_unused=True) for a, b in (expected, actual)]
    for parameter, a, b in zip(parameters, *gradients):
        a = torch.zeros_like(parameter) if a is None else a
        b = torch.zeros_like(parameter) if b is None else b
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert gradients[1][0].abs().sum() > 0
    assert any(g is not None and bool(g.abs().sum()) for g in gradients[1][2:])


@pytest.mark.parametrize('compound', [False, True])
def test_selected_walk_preserves_unused_gradients_and_optimizer_steps(paired, monkeypatch, compound):
    from types import MethodType, SimpleNamespace
    from DecompositionChooser import DecompositionChooser
    from Language import LanguageSpace, SurfaceLayer, LowerLayer, NullLayer
    from Models import BasicModel
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    binary = (SurfaceLayer(12, 12).double(), LowerLayer(12, 12).double())
    unary = (NullLayer().double(),)
    language = SimpleNamespace(_generate_binary_ops=binary, _generate_unary_ops=unary,
                              decomposition_chooser=DecompositionChooser())
    for name in ('reverse_binary_step', '_reverse_of_binary_op', '_finish_binary_inverse',
                 'generate_unary_step', 'choose_generate'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    language.generate_policy_logits = lambda p: p.new_tensor([2., 1., 0., 3.]).expand(len(p), -1)
    words = torch.tensor([[[.1,.2,.3,.4,.1,.2,.3,.4,.5,.6,.7,.8],
                           [.7,.6,.5,.4,.2,.3,.4,.5,.6,.7,.8,.9]]], dtype=torch.float64)
    parent = binary[0].compose(words[:, 0], words[:, 1]) if compound else words[:, 0]
    event = torch.cat((parent.detach()[:, None], torch.zeros(1, 3, 12, dtype=torch.float64)), 1).requires_grad_()
    from Models import _ensure_grad_anchors
    _ensure_grad_anchors(event.device, (event.dtype,))
    owner = SimpleNamespace(languageSpace=language)
    def walk(value):
        return BasicModel._output_generate_walk(owner, value, 8, basis=words,
            basis_valid=torch.ones(1, 2, dtype=torch.bool), return_candidates=True)
    parameters = tuple(p for op in (*binary, *unary) for p in op.parameters())
    before = [p.detach().clone() for p in parameters]
    gradients, steps, results = [], [], []
    import Generative
    selected_inverse = Generative.selected_inverse
    def full_menu(language, parent, choice, live, **kwargs):
        left, right, _ = Generative.inverse_menu(language, parent, live, **kwargs)
        selected = torch.nn.functional.one_hot(choice, left.shape[1]).to(left)
        return (left*selected[..., None]).sum(1), (right*selected[..., None]).sum(1)
    for inverse in (full_menu, selected_inverse):
        monkeypatch.setattr(Generative, 'selected_inverse', inverse)
        with torch.no_grad():
            for p, saved in zip(parameters, before):p.copy_(saved)
        optimizer = torch.optim.SGD(parameters, lr=.01, momentum=.9)
        for p in parameters:optimizer.state[p]['momentum_buffer'] = torch.full_like(p, .1)
        optimizer.zero_grad(set_to_none=True);event.grad = None
        result = walk(event)
        result[0].sum().backward()
        gradients.append([None if p.grad is None else p.grad.clone() for p in (event, *parameters)])
        results.append(result)
        optimizer.step();steps.append([p.detach().clone() for p in parameters])
    for a, b in zip(*results):torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(*gradients):
        assert (a is None) == (b is None)
        if a is not None:torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    for a, b in zip(*steps):torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
