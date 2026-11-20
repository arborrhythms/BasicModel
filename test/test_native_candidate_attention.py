"""Native candidate reader ownership, checkpoints and batch isolation."""
import torch
import pytest

def test_attention_priming_read_never_resizes_or_republishes_its_source(tmp_path):
    from test_grounded_xor import grounded_model
    from AttentionContext import _heat
    model,_=grounded_model(tmp_path)
    space=model.perceptualSpace;owner=space._priming_target();width=owner._priming_dim()
    surface=owner._expected_knowing;pointer=surface.data_ptr();surface.zero_()
    result=_heat(space,2,width,surface[...,0])
    assert result.shape==(2,width) and result.eq(1).all()
    assert owner._expected_knowing is surface
    surface[0,0,1]=1.
    result=_heat(space,2,width,surface[...,0])
    assert result[0,0]==2 and result[1,0]==1
    with pytest.raises(RuntimeError,match='resizing is forbidden'):
        _heat(space,len(surface)+1,width,surface[...,0])
    assert surface.data_ptr()==pointer and surface[0,0,1]==1.
    model.End();model.symbolSpace.soft_reset()


def test_answer_boundary_credits_only_admission_parameters():
    from AttentionCredit import fixed_input
    from ObjectiveOwnership import backward_owned
    attention = torch.nn.Linear(1, 1, bias=False)
    code = torch.nn.Parameter(torch.tensor([[2.]]))
    reader = torch.nn.Parameter(torch.tensor(3.))
    value = attention(code)
    read = fixed_input(value, attention)
    torch.testing.assert_close(read, value, rtol=0, atol=0)
    cost = reader * read.sum()
    assert torch.autograd.grad(cost, value, retain_graph=True, allow_unused=True)[0] is None
    backward_owned({'output': cost, 'attention': cost},
        {'output': (reader,), 'attention': tuple(attention.parameters()), 'reconstruction': (code,)})
    assert code.grad is None
    torch.testing.assert_close(attention.weight.grad, torch.tensor([[6.]]))
    torch.testing.assert_close(reader.grad, value.detach().sum())


def test_admission_reader_boundary_compiles_without_opening_the_state_cut():
    from AttentionCredit import fixed_input
    attention = torch.nn.Linear(1, 1)
    code = torch.tensor([[2.]], requires_grad=True)
    read = torch.compile(lambda x: fixed_input(attention(x), attention),
                         backend='aot_eager', fullgraph=True)
    read(code).sum().backward()
    assert code.grad is None
    torch.testing.assert_close(attention.weight.grad, torch.tensor([[2.]]))
    torch.testing.assert_close(attention.bias.grad, torch.tensor([1.]))


def test_native_attention_checkpoint_and_missing_legacy_parameters(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    optimizer = model.getOptimizer()
    model._optimizer = optimizer
    with torch.no_grad():
        model.attention_context.weight.add_(.02)
    optimizer.zero_grad()
    model.attention_context.weight.square().sum().backward()
    optimizer.step()
    path = tmp_path / 'attention.ckpt'
    model.save_weights(path)
    restored, _ = grounded_model(tmp_path)
    assert restored.load_weights(path, require_match=True)
    resumed = restored.getOptimizer()
    for name, value in restored.attention_context.state_dict().items():
        torch.testing.assert_close(value, model.attention_context.state_dict()[name], rtol=0, atol=0)
    for current, owner in ((model, optimizer), (restored, resumed)):
        owner.zero_grad()
        current.attention_context.weight.square().sum().backward()
        owner.step()
    torch.testing.assert_close(restored.attention_context.weight,
                               model.attention_context.weight, rtol=0, atol=0)
    model.End();model.symbolSpace.soft_reset()
    restored.End();restored.symbolSpace.soft_reset()
    from Layers import GrammarLayer
    GrammarLayer.set_chart_authority(None)
    del model, restored, optimizer, resumed, current, owner
    import gc
    gc.collect()
    legacy = torch.load(path, map_location='cpu', weights_only=False)
    legacy['state_dict'] = {key: value for key, value in legacy['state_dict'].items()
                            if not key.startswith('attention_context.')}
    legacy['state_dict']['candidate_attention.readout.bias'] = torch.tensor([0.125])
    # A checkpoint of the linear chooser held the brackets' anchors beside
    # the space prior; the network reads none, and they are dropped.
    anchor_keys = [key[:-len('space_prior')] + 'bracket_anchor'
                   for key in legacy['state_dict'] if key.endswith('.space_prior')]
    assert anchor_keys
    for key in anchor_keys:
        legacy['state_dict'][key] = torch.full((6, 4), .375)
    torch.save(legacy, path)
    del legacy
    gc.collect()
    fresh, _ = grounded_model(tmp_path)
    initialized = {key: value.clone() for key, value in fresh.attention_context.state_dict().items()}
    with pytest.warns(UserWarning, match=r'retired candidate chooser checkpoint key candidate_attention.readout.bias'):
        assert fresh.load_weights(path, require_match=True)
    for name, value in fresh.attention_context.state_dict().items():
        torch.testing.assert_close(value, initialized[name], rtol=0, atol=0)
    assert not any(key in fresh.state_dict() for key in anchor_keys)
    fresh.End();fresh.symbolSpace.soft_reset()



def test_saved_prefix_projection_and_adam_moments_gain_two_zero_columns(tmp_path):
    from checkpoint_migrations import _optimizer_state_leaves
    from test_grounded_xor import grounded_model

    key = 'attention_context.weight'
    path = tmp_path / 'before-surprise.ckpt'
    model, _ = grounded_model(tmp_path)
    try:
        optimizer = model.getOptimizer()
        model._optimizer = optimizer
        parameter = model.attention_context.weight
        parameter.sum().backward()
        optimizer.step()
        model.save_weights(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        prefix = saved['state_dict'][key][:, :-2].clone()
        saved['state_dict'][key] = prefix
        moments = {}
        manifest = saved['optimizer_param_names']['leaves']
        for leaf, meta in zip(_optimizer_state_leaves(saved['optimizer_state']), manifest, strict=True):
            for group, names in zip(leaf['param_groups'], meta['param_groups'], strict=True):
                for pid, entry in zip(group['params'], names, strict=True):
                    if entry['name'] == key:
                        entry['shape'] = list(prefix.shape)
                        for name, value in leaf['state'][pid].items():
                            if value.shape == parameter.shape:
                                moments[name] = value[:, :-2].clone()
                                leaf['state'][pid][name] = moments[name]
        assert {'exp_avg', 'exp_avg_sq'} <= moments.keys()
        torch.save(saved, path)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
    model, _ = grounded_model(tmp_path)
    try:
        with pytest.warns(UserWarning, match='two zero surprise columns'):
            assert model.load_weights(path, require_match=True)
        parameter = model.attention_context.weight
        torch.testing.assert_close(parameter[:, :-2], prefix, rtol=0, atol=0)
        assert not parameter[:, -2:].any()
        optimizer = model.getOptimizer()
        live = next(leaf for leaf in _optimizer_state_leaves(optimizer.state_dict())
                    if any('exp_avg' in state and state['exp_avg'].shape == parameter.shape
                           for state in leaf['state'].values()))
        state, = (v for v in live['state'].values()
                  if 'exp_avg' in v and v['exp_avg'].shape == parameter.shape)
        for name, previous in moments.items():
            torch.testing.assert_close(state[name][:, :-2], previous, rtol=0, atol=0)
            assert not state[name][:, -2:].any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
