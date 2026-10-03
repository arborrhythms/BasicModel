"""Reconstruction steps scale with gradients and retain bounded row state."""
import torch
from Optimizer import RowLocalSGD, SGD


def test_sparse_momentum_matches_dense_sgd_on_selected_rows_and_roundtrips():
    initial=torch.arange(128*3,dtype=torch.float32).reshape(128,3)/100
    p=torch.nn.Parameter(initial.clone()); q=torch.nn.Parameter(initial.clone())
    opt=RowLocalSGD([p],lr=.1,momentum=.9)
    reference=torch.optim.SGD([q],lr=.1,momentum=.9)
    rows=torch.tensor([2,9])
    for magnitude in (1., .001, .000001):
        g=torch.full((2,3),magnitude)
        p.grad=torch.sparse_coo_tensor(rows[None],g,p.shape)
        q.grad=torch.zeros_like(q);q.grad[rows]=g
        opt.step();reference.step()
        torch.testing.assert_close(p,q)
    assert opt.state[p]['momentum_buffer'].shape == (16,3)
    saved=opt.state_dict()
    restored=RowLocalSGD([p],lr=.1,momentum=.9);restored.load_state_dict(saved)
    torch.testing.assert_close(restored.state[p]['momentum_buffer'],opt.state[p]['momentum_buffer'])
    torch.testing.assert_close(p[0],initial[0],rtol=0,atol=0)


def test_first_step_is_proportional_to_gradient_without_adaptive_division():
    p=torch.nn.Parameter(torch.ones(3))
    p.grad=torch.tensor([1.,1e-4,1e-6])
    before=p.detach().clone();optimizer=SGD([p],lr=.1,momentum=.9);optimizer.step()
    torch.testing.assert_close(p,before-.1*p.grad,rtol=0,atol=0)
