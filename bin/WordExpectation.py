"""Word-level ARMA distribution for the shared bracket expectation owner."""
from typing import NamedTuple
import torch
from torch.nn import functional as F

class ExpectedWords(NamedTuple):
    probabilities: torch.Tensor
    prediction: torch.Tensor
    negative_image: torch.Tensor
    surprise: torch.Tensor
    loss: torch.Tensor
    top1: torch.Tensor
    reciprocal_rank: torch.Tensor


def word_distribution(owner,words,bank,valid,targets,*,teacher_forcing,gain=1.,observed_prefix=None,active=None):
    words,bank=words.detach(),bank.detach()
    B,W,D=words.shape;C=bank.shape[1]
    if bank.shape != (B,C,D) or D != owner.word_dim:
        raise ValueError('word expectation needs one native-width candidate bank per row')
    if not C:raise ValueError('word expectation needs at least one candidate slot')
    active=torch.ones(B,W,device=words.device,dtype=torch.bool) if active is None else active
    prefix=torch.zeros_like(active) if observed_prefix is None else observed_prefix
    if prefix.shape!=(B,W) or active.shape!=(B,W):raise ValueError('word expectation masks must match word brackets')
    if W==0:
        return ExpectedWords(words.new_empty(B,0,C),words,words,words,
                             words.new_empty(B,0),active,words.new_empty(B,0))
    histories=words.new_zeros(B,owner.p,D); errors=words.new_zeros(B,owner.q,D)
    probabilities=[];predictions=[];losses=[];top=[];rank=[]
    safe=valid|(torch.arange(C,device=words.device)[None]==0)&~valid.any(-1,keepdim=True)
    keys=F.normalize(bank,dim=-1)
    steps = W
    if not torch.is_grad_enabled() and not torch.compiler.is_compiling():
        last = torch.where(active, torch.arange(W, device=words.device)[None]+1, 0).amax()
        # After the last active column, history and residuals stop changing.
        # Evaluate its next forecast once and repeat that exact inactive tail.
        steps = min(W, int(last)+1)
    for index in range(steps):
        context=torch.cat((histories.reshape(B,-1),errors.reshape(B,-1)),-1)
        point=owner.word_predictor(context)
        logits=torch.einsum('bd,bcd->bc',point,keys).masked_fill(~safe,-torch.inf)
        probability=logits.softmax(-1)
        prediction=torch.einsum('bc,bcd->bd',probability,bank)
        target=targets[:,index].clamp(0,C-1)
        observed=(targets[:,index]>=0)&valid.gather(1,target[:,None])[:,0]&active[:,index]
        loss=-logits.log_softmax(-1).gather(1,target[:,None])[:,0]
        losses.append(torch.where(observed,loss,0.))
        top.append((probability.argmax(-1)==target)&observed)
        actual=probability.gather(1,target[:,None])
        rank.append(torch.where(observed,1/(1+(probability>actual).sum(-1)).to(words),0.))
        probabilities.append(probability);predictions.append(prediction)
        forecast=bank.gather(1,probability.argmax(-1)[:,None,None].expand(B,1,D))[:,0]
        arrived=torch.where((prefix[:,index]|bool(teacher_forcing))[:,None],words[:,index],forecast)
        residual=arrived-prediction
        if owner.p:histories=torch.where(active[:,index,None,None],torch.cat((histories[:,1:],arrived[:,None]),1),histories)
        if owner.q:errors=torch.where(active[:,index,None,None],torch.cat((errors[:,1:],residual.detach()[:,None]),1),errors)
    if steps < W:
        for values in (probabilities, predictions, losses, top, rank):
            values.extend([values[-1]] * (W-steps))
    distribution=torch.stack(probabilities,1);prediction=torch.stack(predictions,1)
    negative=-float(gain)*prediction
    return ExpectedWords(distribution,prediction,negative,words+negative,
                         torch.stack(losses,1),torch.stack(top,1),torch.stack(rank,1))
