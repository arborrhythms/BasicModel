"""Unseeded convex readout certificates on the model's numeric field."""
import torch
from recon_bench import _build_model


def fit(config='data/MM_xor.xml', *, promotion=True, iterations=32, steps=1, backward=None):
    model, *_ = _build_model(str(config))
    owner = model._concept_owner()
    owner.conceptual_pi = promotion
    values = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).unsqueeze(-1)
    target = torch.tensor([0.,1.,1.,0.])
    model.train()
    for _ in range(owner.concept_mint_count):
        model.forward(values)
    optimizer = torch.optim.LBFGS(model.outputSpace.field_readout.parameters(),
        max_iter=iterations,tolerance_grad=1e-10,tolerance_change=1e-12)
    def loss():
        optimizer.zero_grad()
        prediction = model.forward(values)[2].reshape(-1)
        error = (prediction-target).square().mean()
        error.backward() if backward is None else backward(model,optimizer,error)
        return error
    for _ in range(steps):
        optimizer.step(loss)
    model.eval()
    understood = model.understand(values)
    prediction = understood.execution[2].reshape(-1)
    reconstructed, _ = model.reverseReconstruct(understood)
    torch.testing.assert_close(reconstructed,values,rtol=0,atol=1e-6)
    return model, prediction, float((prediction-target).square().mean().detach())


def certificate(config='data/MM_xor.xml'):
    model, prediction, error = fit(config)
    assert error < 1e-6
    torch.testing.assert_close(prediction,torch.tensor([0.,1.,1.,0.]),rtol=0,atol=1e-6)
    owner = model._concept_owner()
    matrix = owner._concept_allocator.layer(0).conjunctive
    assert matrix.nnz == 2
    row = matrix._rows[0]
    rows = [*model.numeric_percepts.rows,row]
    weights = model.outputSpace.field_readout.weight[0,torch.tensor(rows)*2]
    assert (weights[:2]>0).all() and weights[2]<0
    print(dict(answers=prediction.detach().tolist(),weights=weights.detach().tolist(),
        promoted_row=int(row),mse=error))
    return model,prediction,error
