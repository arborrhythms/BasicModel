"""Witness-free reconstruction competes over live, unnormalized concept codes."""
import torch


def test_readback_uses_signed_activation_cosine_and_priming():
    from SentenceUnderstanding import readback_scores
    leaf = torch.tensor([[2., 0.]])
    codes = torch.tensor([[[2., 0.], [0., 4.], [-1., 1.]]])
    weights = torch.tensor([[3., 9., 2.]])
    # The least-squares signed activation is 1, 0, -1 respectively.
    expected = torch.tensor([[3., 0., 2.**.5]])
    torch.testing.assert_close(readback_scores(leaf, codes, weights), expected)
    torch.testing.assert_close(readback_scores(-leaf, codes, weights), expected)


def test_readback_error_reaches_true_and_competing_codes_without_unit_constraint():
    from SentenceUnderstanding import readback_scores
    codes = torch.nn.Parameter(torch.tensor([[[2., .3], [.4, 3.]]]))
    leaf = torch.tensor([[1., .2]], requires_grad=True)
    scores = readback_scores(leaf, codes, torch.ones(1, 2))
    torch.nn.functional.cross_entropy(scores, torch.tensor([0])).backward()
    assert codes.grad[0, 0].norm() > 0
    assert codes.grad[0, 1].norm() > 0
    assert leaf.grad.norm() > 0
    torch.testing.assert_close(codes.detach(), torch.tensor([[[2., .3], [.4, 3.]]]))


def test_free_trial_uses_no_reference_or_offsets_and_keeps_identity_audit(monkeypatch):
    from pathlib import Path
    import Models, util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1]/'data/XOR_grammar.xml'))
    original = model.reverse
    calls=[]
    def observe(carrier, *, field):
        assert field is not None and field.percepts is None
        # No saved leaves, journal or input spans enter the decoder. The
        # two faces are a pure function of this explicitly supplied field.
        returned = original(carrier, field=field)
        altered = original(carrier.carrier_like(), field=field)
        torch.testing.assert_close(returned.materialize(), altered.materialize())
        torch.testing.assert_close(returned._reconstruction_percepts, altered._reconstruction_percepts)
        calls.append(field)
        return returned
    monkeypatch.setattr(model, 'reverse', observe)
    registries = []
    path_cost = model._sentence_path_cost
    def observe_registry(*args, **kwargs):
        result = path_cost(*args, **kwargs)
        registries.append(model._sentence_cost_registry)
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', observe_registry)
    optimizer=model.getOptimizer(lr=.01)
    try:
        raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
        model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',
            batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
        assert calls
        terms = {name:term for registry in registries for name,term in registry._terms.items()}
        reconstruction = {name for name in terms if name.startswith('reconstruction.')}
        assert reconstruction == {'reconstruction.free_bytes', 'reconstruction.coverage',
            'reconstruction.free_percept', 'reconstruction.definition_fold',
            'reconstruction.walk_residual.form', 'reconstruction.walk_residual.for',
            'reconstruction.walk_residual.against'}
        assert model.concept_binding == 'mixing'
        assert model._last_sentence_field is None
        assert not terms['reconstruction.free_bytes']['trained']
        assert not terms['reconstruction.coverage']['trained']
        assert terms['reconstruction.free_percept']['trained']
        for prefix in ('reconstruction_reverse', 'reconstruction_order_zero'):
            assert terms[prefix + '.what']['trained']
        if model._last_compose_score_function['advantage'].ne(0).any():
            assert 'output.compose_score_function' in terms
        audit = model.ownership_gradient_diagnostics(optimizer)
        assert audit['conflicts'] == 0 and audit['mode'] == 'projection'
        from SentenceTraining import gradient_receipt
        assert gradient_receipt(model)['projection']
    finally:
        model.End()


def test_terminal_copies_share_one_explanation_but_different_ties_do_not():
    from Language import LanguageSpace
    parent = torch.tensor([[.4, .2]])
    available = torch.ones(1, 1, dtype=torch.bool)
    valid = torch.ones(1, 2, dtype=torch.bool)
    def eligible(bank):
        return LanguageSpace.decoder_eligibility(parent, (), (), available, (), bank, valid)
    copies = parent[:, None].expand(1, 2, 2).clone()
    assert bool(eligible(copies)[0, 0])
    distinct = torch.stack((parent, parent * 2), 1)
    assert not bool(eligible(distinct)[0, 0])
