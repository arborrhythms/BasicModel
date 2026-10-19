"""Item 4 gain, local prediction and raw-evidence contracts."""
import copy
import pytest
import torch
from Expectation import ActivationPredictor, PerceptMoments, gain_field, alternative_probabilities
from Layers import SymbolExpectation
from WordExpectation import word_distribution


def test_gain_preserves_identity_and_has_two_residual_signs():
    o = torch.tensor([[1., .2], [0., 0.], [.2, .5]])
    e = torch.tensor([[1., .2], [.8, .1], [.1, .1]])
    field = gain_field(o, e, strength=.75)
    torch.testing.assert_close(field.content[0], .25*o[0])
    assert torch.equal(field.content[1], o[1])
    torch.testing.assert_close(field.omission[1], e[1])
    assert bool(((field.content >= 0) & (field.content <= o)).all())
    assert torch.equal(field.content[2], o[2])
    assert torch.equal(gain_field(o, e, attention=0.).surprise, torch.zeros(3))
    assert torch.equal(gain_field(o, e, confidence=0.).content, o)
    withheld = gain_field(o, e, exempt=1., withheld=1.)
    assert torch.equal(withheld.content, o)
    assert not withheld.expected.any()


def test_entire_gain_is_detached_but_local_error_trains_predictor():
    o = torch.tensor([[.7, .1]], requires_grad=True)
    e = torch.tensor([[.8, .2]], requires_grad=True)
    field = gain_field(o, e)
    field.content.sum().backward()
    assert e.grad is None and o.grad is not None
    field.surprise.mean().backward()
    assert e.grad is not None


def test_percept_prior_is_causal_normalized_and_checkpointed():
    prior = PerceptMoments(2, rate=.5)
    values = torch.tensor([[[.3, .9], [.7, .1], [99., 99.]]])
    valid = torch.tensor([[True, True, False]])
    assert torch.equal(prior.residual(values), torch.ones_like(values))
    prior.observe(values, valid)
    torch.testing.assert_close(prior.mean, torch.tensor([.5, .5]))
    torch.testing.assert_close(prior.variance, torch.tensor([.04, .16]))
    assert int(prior.count) == 2
    clone = PerceptMoments(2)
    clone.load_state_dict(prior.state_dict())
    torch.testing.assert_close(clone.residual(values), prior.residual(values))
    torch.testing.assert_close(prior.region_gain(prior.mean[None, None], torch.ones(1,1,dtype=torch.bool), strength=.6), torch.tensor([.4]))


def owner():
    head = SymbolExpectation(4, 8, 4, concept_dim=4)
    head.set_symbol_bank(torch.eye(4), orders=torch.tensor([0, 0, 1, 2]))
    return head


def test_shared_activation_owner_is_causal_and_detaches_both_sides():
    head = owner()
    source = torch.eye(4)[:2].requires_grad_()
    head.predict_and_observe_stm_end_state([2], [source], layout='infix', documents=['a'])
    pending = head.expect_next_meaning()
    assert pending.symbol_logits.shape == (4,2)
    assert head.word_predictor is head._inter_predictor
    assert 'byte' not in head.levels
    target = torch.eye(4)[2:3].requires_grad_()
    head.observe_stm_end_state([1], [target], layout='infix', documents=['a'])
    loss = head.consume_inter_loss()
    loss.backward()
    assert source.grad is None and target.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() for p in head._inter_predictor.parameters())
    comparison = head.last_expectation_comparison()
    assert comparison.residual.shape == (4,2)
    assert set(head.expectation_metrics()['per_order_mse']) == {0,1,2}
    assert head.expect_next_meaning() is not None
    head.begin_document(0,'b')
    assert head.expect_next_meaning() is None


def test_word_forecast_never_feedbacks_prediction_and_only_attenuates():
    head = owner()
    words = torch.eye(4)[None]
    bank = words.clone()
    valid = torch.ones(1,4,dtype=torch.bool)
    targets = torch.arange(4)[None]
    before = word_distribution(head, words, bank, valid, targets)
    after = word_distribution(head, words, bank, valid, targets)
    torch.testing.assert_close(before.activation, after.activation)
    assert bool((before.content >= 0).all())
    assert bool((before.content <= words).all())
    changed = words.clone(); changed[:,2:] = .25
    other = word_distribution(head, changed, bank, valid, targets)
    torch.testing.assert_close(before.activation[:,:3], other.activation[:,:3])


def test_alternative_mixture_has_positive_floor_and_validates_settings():
    logits = torch.tensor([[100., -100., -torch.inf], [-torch.inf,-torch.inf,-torch.inf]])
    probability = alternative_probabilities(logits, fraction=.2, temperature=.7)
    torch.testing.assert_close(probability[0], torch.tensor([.9,.1,0.]))
    assert torch.equal(probability[1], torch.zeros(3))
    for kwargs in (dict(fraction=0,temperature=1),dict(fraction=.5,temperature=0)):
        with pytest.raises(ValueError): alternative_probabilities(logits, **kwargs)


def test_unlabelled_admitted_word_has_no_prior_and_passes_whole():
    head = owner()
    words = torch.tensor([[[.2,.3,.4,.5], [.6,.7,.8,.9]]])
    result = word_distribution(head, words, torch.eye(4)[None],
        torch.ones(1,4,dtype=torch.bool), torch.full((1,2), -1),
         gain=1., active=torch.tensor([[True, False]]))
    torch.testing.assert_close(result.content[:,0], words[:,0])
    assert not result.content[:,1].any()
    assert not result.loss.any()


def test_trial_keep_and_credit_use_same_r_plus_a_judge():
    from SentenceCredit import comparison
    parts = torch.tensor([[[1., 0., 3.], [2., 999., 0.]]])
    result = comparison(parts, torch.tensor([True]))
    assert result['wins'].item()
    assert not result['policy_against_keep'].item()
    torch.testing.assert_close(result['costs'], torch.tensor([[4.,2.]]))


def test_expected_identity_survives_zero_content_and_retained_pair_restore():
    from Layers import MeaningExpectation, TernaryTruthStore
    from Meaning import ConceptualMeaning, ClosingImage
    keys = torch.eye(4)
    logits = torch.full((4,2), -60.); logits[0,0] = 60.
    roles = torch.cat((keys[:1], torch.zeros(2,4)))
    prediction = MeaningExpectation(roles, torch.tensor([60.,-60.,-60.]),
        symbol_keys=keys, symbol_logits=logits, symbol_ids=torch.arange(4), symbol_orders=torch.arange(4))
    occupied = torch.tensor([True,False,False])
    view = ClosingImage.form(roles, prediction.roles, prediction.confidence,
        prediction=prediction, occupied=occupied, gain=1.)
    assert not view.conceived.any()
    torch.testing.assert_close(view.restore(), roles)
    assert view.awareness()['expectation'][0,0] == 1
    assert view.awareness()['identities'][0] == 0
    observed = ConceptualMeaning(roles, occupied)
    estimate = ConceptualMeaning(roles, occupied, mode='unspecified', bindings=prediction.retained_bindings())
    store = TernaryTruthStore(4, capacity=8)
    source = store.append_meaning(observed, kind='observation')
    e, o = store.append_expectation_pair(estimate, observed, presence_logits=prediction.presence_logits,
        source_occurrences=(store.occurrence_of(source),), document='test', stream='test')
    pair = store.expectation_pair(o, gain=1.)
    assert pair['estimate_occurrence'] == store.occurrence_of(e)
    assert pair['expected_activation'].shape == (4,2)
    assert not pair['conceived'].any()
    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    other = restored.expectation_pair(o)
    torch.testing.assert_close(other['expected_activation'], pair['expected_activation'])
    torch.testing.assert_close(other['observation'].roles, roles)


def test_non_mark_blocks_prime_and_gain_without_erasing_retained_modifier():
    from dataclasses import replace
    from Layers import MeaningExpectation
    from Meaning import ConceptualMeaning, ClosingImage, withheld_symbols
    keys = torch.eye(4)
    roles = torch.cat((keys[:1], torch.zeros(2,4)))
    mark = {'_withheld_symbols': ((0, tuple(keys[0].tolist())),)}
    prediction = MeaningExpectation(roles, torch.zeros(3), bindings=mark,
        symbol_keys=keys, symbol_logits=torch.full((4,2), 10.),
        symbol_ids=torch.arange(4), symbol_orders=torch.zeros(4,dtype=torch.long))
    assert not prediction.expected_activation[0].any()
    actual = ConceptualMeaning(roles, torch.tensor([True,False,False]), bindings=mark)
    view = ClosingImage.form(roles, roles, prediction.confidence, prediction=prediction,
        occupied=actual.role_mask, withheld=withheld_symbols(actual,prediction), gain=1.)
    assert view.gain[0] == 1
    assert not view.expected[0].any()
    assert dict(actual.detached().bindings)['_withheld_symbols'] == actual.bindings[0][1]


def test_fused_retained_row_replays_original_two_lane_target():
    """A fused row is not a new observation of its nearest bank identity."""
    from Layers import MeaningExpectation, TernaryTruthStore
    from Meaning import ConceptualMeaning
    keys = torch.eye(4)
    expected = torch.tensor([[.8, .1], [.1, .7], [.2, .2], [.1, .1]])
    target = torch.tensor([[.6, .0], [.0, .9], [.0, .0], [.0, .0]])
    ids = torch.tensor([17, 42, 103, 999])
    roles = torch.cat((keys[:1], torch.zeros(2,4)))
    occupied = torch.tensor([True, False, False])
    prediction = MeaningExpectation(roles, torch.zeros(3), symbol_keys=keys,
        symbol_logits=expected.logit(), symbol_ids=ids, symbol_orders=torch.arange(4))
    # This durable description has different coordinates from either original
    # observed symbol. Store the exact local target beside the full prior.
    fused = torch.cat(((keys[0]+keys[1])[None]/2, torch.zeros(2,4)))
    observed = ConceptualMeaning(fused, occupied, bindings={
        '_observed_symbol_activations': tuple((int(i), tuple(pair))
            for i, pair in zip(ids, target.tolist()))})
    estimate = ConceptualMeaning(roles, occupied, mode='unspecified',
        bindings=prediction.retained_bindings())
    store = TernaryTruthStore(4, capacity=8)
    _, row = store.append_expectation_pair(estimate, observed,
        presence_logits=prediction.presence_logits, document='fused', stream='fused')
    residual = target-expected
    from Expectation import retention_surprise
    surprise = retention_surprise(residual, expected)
    for replay in (store, TernaryTruthStore(4, capacity=8)):
        if replay is not store:
            replay.load_state_dict(copy.deepcopy(store.state_dict()))
            replay.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
        pair = replay.expectation_pair(row)
        torch.testing.assert_close(pair['residual'], residual)
        torch.testing.assert_close(pair['excess'], residual.relu())
        torch.testing.assert_close(pair['omission'], (-residual).relu())
        torch.testing.assert_close(pair['observation'].roles, fused)
        assert pair['surprise'] == pytest.approx(float(surprise))


def test_symbol_gain_keeps_native_identity_and_inactive_row_context():
    from SymbolField import SymbolField
    head = owner()
    # Force a perfectly predicted positive lane for every symbol.
    with torch.no_grad():
        for parameter in head._inter_predictor.parameters():
            parameter.zero_()
        head._inter_predictor.local.bias.copy_(torch.tensor([60., -60.]))
    keys = torch.eye(4)
    values = torch.stack((torch.stack((keys[0], keys[1])),
                          torch.stack((keys[2], torch.zeros(4)))))
    depth = torch.tensor([1, 1])
    order = torch.zeros(2,2,dtype=torch.long)
    ids = torch.tensor([[0, 1], [2, -1]])
    activation = torch.tensor([[.7, .4], [.8, 0.]])
    state = values, depth, order, order.clone(), ids, activation
    field = SymbolField(head, strength=1., epsilon=1e-6, saturation='clip')
    field.anticipate(state)
    arrived = (*state[:1], torch.tensor([2,1]), *state[2:])
    gained = field.arrive(arrived, torch.tensor([True,False]))
    torch.testing.assert_close(gained[0], values)
    torch.testing.assert_close(gained[4], ids)
    assert gained[5][0,0] == activation[0,0]  # no repeated attenuation
    assert gained[5][0,1] == activation[0,1]  # raw coefficient is immutable
    assert field.read().gains(values)[0,1] == 0  # content consumer receives S once
    assert gained[5][1,0] == activation[1,0]  # inactive row is untouched
    field.anticipate(gained)
    torch.testing.assert_close(field.pending[3][1,2], torch.tensor([1.,0.]))


def test_negative_evidence_is_an_activation_target_not_an_omission_sign():
    head = owner()
    roles = torch.eye(4)[:1]
    head.predict_and_observe_stm_end_state([1], [roles], layout='infix')
    head.predict_and_observe_stm_end_state([1], [roles], layout='infix', evidence=[(0., .8)])
    target = head.last_expectation_comparison().activation
    torch.testing.assert_close(target[0], torch.tensor([0., .8]))


def test_departure_policy_uses_expectation_free_logits_and_exact_proposal_credit():
    from Language import OperationSelectionLayer
    chooser = OperationSelectionLayer(d_model=4)
    chooser.exploration_fraction = .2
    logits = torch.tensor([[100., 1000., -100.]]).expand(1024,-1)
    free = torch.zeros_like(logits)
    selected, probabilities, valid, correction = chooser.select_logits(logits,
        structural=(False,)*3, masked_action=torch.ones(1024,dtype=torch.long),
        alternative_logits=free, return_proposal=True)
    assert valid.all() and not selected.eq(1).any()
    assert .4 < float(selected.eq(0).float().mean()) < .6
    torch.testing.assert_close(correction, torch.full((1024,),2.))
    assert probabilities[:,1].eq(1).all()
