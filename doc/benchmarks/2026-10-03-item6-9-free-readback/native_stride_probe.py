import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'bin'))
import torch
from Layers import InvertibleLinearLayer

def test_ldu_diagonal_cond_backward_is_dense_and_exact():
    layer=InvertibleLinearLayer(4,4)
    def forward(selected):
        return torch.cond(selected,lambda:layer._D_embed(),lambda:torch.zeros((4,4)),())
    compiled=torch.compile(forward,backend='aot_eager',fullgraph=True)
    weights=torch.arange(16,dtype=torch.float32).reshape(4,4)
    (compiled(torch.tensor(True))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,weights.diagonal())
    layer.d.grad=None
    (compiled(torch.tensor(False))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,torch.zeros(4))


def test_functional_ldu_cond_backward_is_dense_and_exact():
    layer=InvertibleLinearLayer(4,4,naive=True)
    x=torch.arange(1,13,dtype=torch.float32).reshape(3,4)/17
    def forward(selected):
        return torch.cond(selected,lambda:layer.functional_forward(x),lambda:torch.zeros_like(x),())
    compiled=torch.compile(forward,backend='aot_eager',fullgraph=True)
    weights=torch.arange(12,dtype=torch.float32).reshape(3,4)
    (compiled(torch.tensor(True))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,(x*weights).sum(0))
    layer.d.grad=None
    (compiled(torch.tensor(False))*weights).sum().backward()
    torch.testing.assert_close(layer.d.grad,torch.zeros(4))
