"""Current saved Adam checkpoints resume reader state and reset R moments."""
import torch
from Optimizer import Adam, SGD, RowLocalSGD, MultiOptimizer
from checkpoint_migrations import build_optimizer_param_manifest, remap_optimizer_state_by_name


def test_reconstruction_change_keeps_reader_adam_and_starts_sgd_momentum_empty():
    code=torch.nn.Parameter(torch.tensor([1.,2.])); reader=torch.nn.Parameter(torch.tensor([3.]))
    named=[('codes',code),('reader',reader)]
    old=Adam([code,reader],lr=.02)
    (code.square().sum()+reader.square().sum()).backward();old.step();old.zero_grad()
    live=MultiOptimizer([Adam([reader],lr=.02),SGD([code],lr=.02,momentum=.9)])
    result=remap_optimizer_state_by_name(old.state_dict(),build_optimizer_param_manifest(old,named),
        live.state_dict(),build_optimizer_param_manifest(live,named))
    live.load_state_dict(result.state)
    torch.testing.assert_close(live.state[reader]['exp_avg'],old.state[reader]['exp_avg'])
    assert not live.state.get(code)
    assert live.optimizers[1].param_groups[0]['momentum'] == .9
    assert 'betas' not in live.optimizers[1].param_groups[0]
    (code.square().sum()+reader.square().sum()).backward();live.step()
    assert 'momentum_buffer' in live.state[code]
    assert result.diagnostics.dropped_saved_states == ('codes',)


def test_retired_chooser_saved_optimizer_state_is_dropped_by_name():
    reader = torch.nn.Parameter(torch.tensor([3.]))
    chooser = torch.nn.Parameter(torch.tensor([1., .2, .3, .4, .5]))
    retired = 'languageSpace.decomposition_chooser.weight'
    old = Adam([reader, chooser], lr=.02)
    (reader.square().sum() + chooser.square().sum()).backward()
    old.step(); old.zero_grad()
    assert old.state[chooser]['exp_avg'].count_nonzero() == 5
    saved = old.state_dict()
    live = Adam([reader], lr=.02)
    result = remap_optimizer_state_by_name(saved,
        build_optimizer_param_manifest(old, [('reader', reader), (retired, chooser)]),
        live.state_dict(), build_optimizer_param_manifest(live, [('reader', reader)]))
    live.load_state_dict(result.state)
    assert result.diagnostics.dropped_saved_states == (retired,)
    assert all(p is not chooser for group in live.param_groups for p in group['params'])
    assert chooser not in live.state
    torch.testing.assert_close(live.state[reader]['exp_avg'], old.state[reader]['exp_avg'], rtol=0, atol=0)
    torch.testing.assert_close(live.state[reader]['exp_avg_sq'], old.state[reader]['exp_avg_sq'], rtol=0, atol=0)
