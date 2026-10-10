"""Native candidate reader ownership, checkpoints and batch isolation."""
import torch
import pytest

def test_attention_priming_read_never_resizes_or_republishes_its_source(tmp_path):
    from test_grounded_xor import grounded_model
    from ModelCandidateAttention import _heat
    model, _ = grounded_model(tmp_path)
    space = model.perceptualSpace
    owner = space._priming_target()
    width = owner._priming_dim()
    surface = torch.ones(1, width)
    object.__setattr__(owner, '_priming_boosts', surface)
    result = _heat(space, 2, width, surface)
    assert result.shape == (2, width) and result.eq(1).all()
    assert owner._priming_boosts is surface
    surface[0, 0] = 2.
    with pytest.raises(ValueError, match='current batch'):
        _heat(space, 2, width, surface)
    assert owner._priming_boosts is surface and surface[0, 0] == 2.


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
        model.candidate_attention.readout.bias.add_(.02)
    optimizer.zero_grad()
    model.candidate_attention.readout.bias.square().sum().backward()
    optimizer.step()
    path = tmp_path / 'attention.ckpt'
    model.save_weights(path)
    restored, _ = grounded_model(tmp_path)
    assert restored.load_weights(path, require_match=True)
    resumed = restored.getOptimizer()
    for name, value in restored.candidate_attention.state_dict().items():
        torch.testing.assert_close(value, model.candidate_attention.state_dict()[name], rtol=0, atol=0)
    for current, owner in ((model, optimizer), (restored, resumed)):
        owner.zero_grad()
        current.candidate_attention.readout.bias.square().sum().backward()
        owner.step()
    torch.testing.assert_close(restored.candidate_attention.readout.bias,
                               model.candidate_attention.readout.bias, rtol=0, atol=0)
    legacy = torch.load(path, map_location='cpu', weights_only=False)
    legacy['state_dict'] = {key: value for key, value in legacy['state_dict'].items()
                            if not key.startswith('candidate_attention.')}
    legacy['optimizer_state'] = None
    torch.save(legacy, path)
    fresh, _ = grounded_model(tmp_path)
    initialized = {key: value.clone() for key, value in fresh.candidate_attention.state_dict().items()}
    assert fresh.load_weights(path, require_match=True)
    for name, value in fresh.candidate_attention.state_dict().items():
        torch.testing.assert_close(value, initialized[name], rtol=0, atol=0)


