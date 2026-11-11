"""Direct gradient ownership certificates for the common order-zero loop."""
from dataclasses import replace
from types import SimpleNamespace as NS
import torch


def test_input_reconstruction_reaches_perception_and_not_the_grammar(tmp_path, monkeypatch):
    from test_native_attention_field import _model
    model = _model(tmp_path, monkeypatch)
    captured = []
    owner = model._concept_owner()
    opened = []
    read = owner.cs_read_memberships
    def memberships(*args, **kwargs):
        field = read(*args, **kwargs)
        opened.append(owner._concept_allocator.layer(0).features.values)
        return field
    monkeypatch.setattr(owner, 'cs_read_memberships', memberships)
    original = model._sentence_path_cost
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        errors = model._sentence_cost_registry
        owner = model._concept_owner()
        parameters = (opened[-1],
            model.wholeSpace.subspace.what.primitive_properties.members,
            model.perceptualSpace.subspace.what.W,
            *tuple(model.languageSpace.generate_policy.parameters()))
        terms = [errors._value(term).sum() for name, term in errors._terms.items()
                 if name.startswith('reconstruction_reverse.')]
        gradients = torch.autograd.grad(sum(terms), parameters, retain_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gradients[:2]), [
            None if g is None else (tuple(g.shape), bool(torch.isfinite(g).all())) for g in gradients[:3]]
        assert gradients[2] is not None
        assert all(g is None for g in gradients[3:])
        captured.append(True)
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', capture)
    try:
        source = model.inputSpace.prepInput(['alpha beta'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        assert captured
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_knowing_deposit_keeps_the_selected_intuition_weights_live():
    from test_thought_infer import setup, idea
    from Intuition import Idea, Term, seed, pack_term, write_knowing
    from QueryWork import QueryWorkBudget
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    row = owner._csw_row_of(refs[1][1])
    with torch.no_grad():
        owner.similarity_codebook.W[row].fill_(1.)
    known = Idea(observed=torch.full((owner.outputShape[-1],), .3))
    antecedent = Idea(observed=torch.full((owner.outputShape[-1],), .8))
    goal = Idea(observed=torch.full((owner.outputShape[-1],), .5))
    term = seed(owner, Term(('sym', 123), antecedent, goal, (.9, .2),
        idea(registry, refs[1]), (known,), (2.,) * 4), QueryWorkBudget(256))
    meaning = replace(term.meaning, bindings={'_intuition_terms': (pack_term(term),)})
    carrier = owner.subspace
    carrier._concept_activations = torch.zeros(1, 1, 2, 2)
    carrier._concept_inventory_rows = torch.tensor([row])
    carrier._concept_where = torch.tensor([[[0, 1], [0, 0]]])
    write_knowing(model, {'meaning': meaning, 'knowing': ((row, (.1, .1)),)}, 0)
    from FieldReverse import NativeDecode, reverse
    from PerceptProperties import PrimitiveProperties
    from Understanding import ConceptualField
    from basic_fixtures import feature
    radix = owner.basic_percepts.native
    native = radix.insert(b'a')
    radix.activity_decoder = NativeDecode(radix.capacity, 8, 8)
    feature(owner, row, 'ps', native, 1.)
    decoder_model = NS(_concept_owner=lambda: owner,
        perceptualSpace=NS(percept_store=radix, subspace=NS(nWhat=radix.dim, nWhere=0, nWhen=0), _radix_where_indices=[]),
        wholeSpace=NS(subspace=NS(what=NS(primitive_properties=PrimitiveProperties(0)))))
    field = ConceptualField(torch.tensor([refs[1][1]]), carrier._concept_activations,
        carrier._concept_where, torch.tensor([[[1, 2], [0, 0]]]), None)
    loss = reverse(decoder_model, NS(carrier_like=NS), field)._radix_activity.percepts.square().sum()
    gradient, = torch.autograd.grad(loss, owner.intuition.sigma)
    assert gradient.coalesce().values().abs().sum() > 0
    assert carrier._concept_activations.shape == (1, 1, 2, 2)
    assert not carrier._concept_activations[:, :, 1].any()
    assert not hasattr(carrier, '_thought_occurrence')


def test_thought_ideas_use_when_and_keep_where_empty():
    from test_thought_infer import setup, idea
    from ThoughtChunks import put, read
    from ThoughtReferences import bindings
    from Memory import STMChunks
    model, registry, store, refs = setup()
    stm = model.conceptualSpace.stm
    times = []
    for ref in refs:
        deposited = put(stm, idea(registry, ref), row=0)
        when = bindings(deposited)['_chunk_when']
        times.append(when)
        index = int(STMChunks.locate(stm.state, when, band=STMChunks.WHEN)[0])
        assert index >= 0
        assert stm.state[STMChunks.WHERE][0, index].eq(-1).all()
        assert read(stm, when, 0).role_refs == deposited.role_refs
    assert len(times) == len(set(times))


def test_order_zero_decode_reaches_the_grammar_inverse_and_selected_definition(monkeypatch):
    from contextlib import nullcontext
    from types import MethodType
    from test_cs_sparse_weights import _cs, _mint_row
    from basic_fixtures import feature
    from FieldReverse import NativeDecode, reverse
    from PerceptProperties import PrimitiveProperties
    from SentenceUnderstanding import PrimedSymbols
    from Understanding import ConceptualField
    from OutputOccurrences import generated, realize
    from Language import LanguageSpace, SurfaceLayer, GeneratePolicy
    from DecompositionChooser import DecompositionChooser
    from Layers import Error
    from Models import BasicModel
    from Generative import selected_residual
    from WalkTrials import stage_generate_credit
    from util import TheXMLConfig
    cs = _cs(dim=12)
    rows = [_mint_row(cs, 0, cid) for cid in (101, 102)]
    radix = cs.basic_percepts.native
    for row, raw in zip(rows, (b'a', b'b')):
        feature(cs, row, 'ps', radix.insert(raw), 1.)
    feature(cs, rows[0], 'ps', radix.get_id(b'b'), .5)
    radix.activity_decoder = NativeDecode(radix.capacity, 8, 8)
    monkeypatch.setattr(cs, 'word_surface_for_row', lambda row: (b'a', b'b')[rows.index(row)])
    TheXMLConfig.data['architecture']['data']['dataType'] = 'embedding'
    TheXMLConfig.data['PartSpace']['nDim'] = 4
    TheXMLConfig.data['ConceptualSpace']['nDim'] = 12
    op = SurfaceLayer(12, 12)
    codes = torch.tensor([[[.8,.1,.1,.1, .6,.5,.4,.3, .2,.3,.4,.5],
                           [.1,.8,.1,.1, .4,.3,.2,.1, .3,.4,.5,.6]]])
    parent = op.compose_blocks(op.compose, codes[:, 0], codes[:, 1]).detach()
    with torch.no_grad():
        op.meaning_kernel.bias.fill_(.03)
    policy = GeneratePolicy(12, 2)
    language = NS(_generate_binary_ops=(op,), _generate_unary_ops=(),
        decomposition_chooser=DecompositionChooser(), generate_policy=policy, _generate_policy_width=12,
        _tree_layer=lambda arity: NS(ops=(op,), exploration_fraction=.5, exploration_temperature=1.),
        generation_scope=nullcontext)
    for name in ('reverse_binary_step', '_reverse_of_binary_op', 'generate_policy_logits'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    bank = PrimedSymbols(torch.tensor([rows]), codes, torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[[97], [98]]]), torch.ones(1, 2, 1, dtype=torch.bool), meaning_start=4)
    model = NS(_concept_owner=lambda: cs, conceptualSpace=cs, languageSpace=language, _synthesis_guard=nullcontext,
        perceptualSpace=NS(percept_store=radix, subspace=NS(nWhat=12, nWhere=0, nWhen=0), _radix_where_indices=[]),
        wholeSpace=NS(subspace=NS(what=NS(primitive_properties=PrimitiveProperties(0)))))
    model.reverse = lambda carrier, *, field: reverse(model, NS(carrier_like=NS), field)
    left, right = language.reverse_binary_step(parent, torch.tensor([0]), torch.tensor([True]),
        ops=(op,), basis=codes, basis_valid=bank.valid)
    proposal = generated(model, torch.stack((left, right), 1), torch.tensor([2]),
        torch.tensor([False]), bank, (), when=torch.tensor([[[1, 2]]]))
    decoded = realize(model, proposal)._radix_activity.percepts
    field = ConceptualField(torch.tensor([101, 102]),
        torch.stack((torch.eye(2), torch.zeros(2, 2)), -1)[:, None],
        torch.tensor([[[0, 1], [2, 3]]]), torch.tensor([[[1, 2], [1, 2]]]), None)
    observed = model.reverse(None, field=field)._radix_activity.percepts
    # Both terms use the native per-byte target. The observed-field route
    # cannot reach the inverse, while its generated field carries that pullback.
    from Layers import ModelLoss
    model.loss = ModelLoss(what_scale=1., where_scale=0., when_scale=0.)
    model.inputSpace = NS(subspace=NS(nWhere=0, nWhen=0))
    errors = Error()
    BasicModel._reverse_event_loss(model, decoded, observed.detach(), registry=errors, name='reconstruction_order_zero')
    BasicModel._reverse_event_loss(model, observed, torch.zeros_like(observed), registry=errors, name='reconstruction_reverse')
    inverse = tuple(op.meaning_kernel.parameters())
    def gradient(prefix, parameters):
        terms = [errors._value(term).sum() for name, term in errors._terms.items() if name.startswith(prefix)]
        return torch.autograd.grad(sum(terms), parameters, retain_graph=True, allow_unused=True)
    derivatives = gradient('reconstruction_order_zero.', (*inverse, cs._concept_allocator.layer(0).features.values))
    assert any(g is not None and torch.isfinite(g).all() and bool(g.abs().any()) for g in derivatives[:-1])
    assert derivatives[-1] is not None and bool(derivatives[-1].abs().any())
    assert all(g is None for g in gradient('reconstruction_reverse.', (*inverse, *policy.parameters())))
    # A nonexact proposal supplies the residual lane, using the same inverse's
    # native recomposition parameters rather than a search-gradient surrogate.
    residual = selected_residual(language, parent, codes[:, 0], codes[:, 1], torch.tensor([0]),
        torch.tensor([True]), form_width=4)
    for i, band in enumerate(('form', 'for', 'against')):
        errors.add('reconstruction.walk_residual.' + band, residual[:, i], category='reconstruction', objective='reconstruction', side='R')
    assert any(g is not None and bool(g.abs().any()) for g in gradient('reconstruction.walk_residual.', inverse))
    model._sentence_decoder_walk_credits = []
    audit = dict(departure=torch.tensor([0]), greedy=torch.tensor([[0]]), explore=torch.tensor([[1]]),
        eligible=torch.tensor([[True]]), costs=torch.tensor([[.5, 1.5]]))
    stage_generate_credit(model, audit, parent[:, None], torch.ones(1, 1, 2, 9), owner='decoder')
    errors.add('reconstruction.walk_credit', model._sentence_decoder_walk_credits[0], category='reconstruction', objective='reconstruction', side='R')
    assert any(g is not None and bool(g.abs().any()) for g in gradient('reconstruction.walk_credit', tuple(policy.parameters())))
    assert proposal.field.where[0].tolist() == [[0, 1], [2, 3]]


def test_concept_activation_preserves_both_poles_including_neither():
    from ConceptActivation import poles
    from Interpret import activate_code
    code = torch.tensor([[.7, .2, .3, .8, 0., 0.]]).expand(5, -1)
    expected = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.], [.3, .6]], requires_grad=True)
    value = activate_code(code, torch.ones(5), 2, evidence=expected)
    actual = poles(value, code, 2)
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual.sum(), expected)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[expected > 0], torch.ones_like(gradient[expected > 0]))
