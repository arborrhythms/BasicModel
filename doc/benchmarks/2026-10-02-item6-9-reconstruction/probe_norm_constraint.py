import torch
from types import SimpleNamespace
from Models import BaseModel
w=torch.nn.Parameter(torch.tensor([[2.,3.]]))
w.grad=torch.ones_like(w)
before=w.detach().clone()
model=SimpleNamespace(conceptualSpaces=[SimpleNamespace(similarity_codebook=SimpleNamespace(W=w))])
BaseModel._normalize_conceptual_codebooks(model)
print(dict(before=before.tolist(),after=w.tolist()))
assert torch.equal(w,before), 'post-step norm projection is another code writer'
