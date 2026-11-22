"""Native candidate reader ownership, checkpoints and batch isolation."""
import torch
import pytest

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
        model.request_read.first.weight.add_(.02)
    optimizer.zero_grad()
    model.request_read.first.weight.square().sum().backward()
    optimizer.step()
    path = tmp_path / 'attention.ckpt'
    model.save_weights(path)
    restored, _ = grounded_model(tmp_path)
    assert restored.load_weights(path, require_match=True)
    resumed = restored.getOptimizer()
    for name, value in restored.request_read.state_dict().items():
        torch.testing.assert_close(value, model.request_read.state_dict()[name], rtol=0, atol=0)
    for current, owner in ((model, optimizer), (restored, resumed)):
        owner.zero_grad()
        current.request_read.first.weight.square().sum().backward()
        owner.step()
    torch.testing.assert_close(restored.request_read.first.weight,
                               model.request_read.first.weight, rtol=0, atol=0)
    model.End();model.symbolSpace.soft_reset()
    restored.End();restored.symbolSpace.soft_reset()
    from Layers import GrammarLayer
    GrammarLayer.set_chart_authority(None)
    del model, restored, optimizer, resumed, current, owner
    import gc
    gc.collect()
    legacy = torch.load(path, map_location='cpu', weights_only=False)
    legacy['state_dict'] = {key: value for key, value in legacy['state_dict'].items()
                            if not key.startswith('request_read.')}
    legacy['state_dict']['candidate_attention.readout.bias'] = torch.tensor([0.125])
    # The context's map and the request's start (retired in the eleventh
    # pass, rule 80) are dropped as the linear chooser's keys are.
    legacy['state_dict']['attention_context.weight'] = torch.zeros(2, 5)
    legacy['state_dict']['request_start.value'] = torch.zeros(3)
    # A checkpoint of the linear chooser held the brackets' anchors and
    # space prior; the network reads none, and they are dropped.
    prefixes = {key[:key.index('operation_layer.') + len('operation_layer.')]
                for key in legacy['state_dict'] if 'operation_layer.' in key}
    assert prefixes
    anchor_keys = [prefix + name for prefix in prefixes for name in ('bracket_anchor', 'space_prior')]
    for key in anchor_keys:
        legacy['state_dict'][key] = torch.full((6, 4) if key.endswith('anchor') else (6,), .375)
    torch.save(legacy, path)
    del legacy
    gc.collect()
    fresh, _ = grounded_model(tmp_path)
    initialized = {key: value.clone() for key, value in fresh.request_read.state_dict().items()}
    with pytest.warns(UserWarning, match=r'retired candidate chooser checkpoint key candidate_attention.readout.bias'):
        assert fresh.load_weights(path, require_match=True)
    for name, value in fresh.request_read.state_dict().items():
        torch.testing.assert_close(value, initialized[name], rtol=0, atol=0)
    assert not any(key in fresh.state_dict() for key in anchor_keys)
    fresh.End();fresh.symbolSpace.soft_reset()


