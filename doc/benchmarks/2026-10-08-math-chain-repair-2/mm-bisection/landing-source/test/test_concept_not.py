"""Negation exchanges poles without changing the concept code."""
import torch


def test_not_preserves_a_full_concept_code():
    from Language import NotLayer
    code = torch.tensor([[.1, -.2, .3, .4, -.5, .6, -.7, .8, -.9, 1.]], requires_grad=True)
    layer = NotLayer()
    reflected = layer(code)
    torch.testing.assert_close(reflected, code, rtol=0, atol=0)
    torch.testing.assert_close(layer.reverse(reflected), code, rtol=0, atol=0)
    reflected.sum().backward()
    torch.testing.assert_close(code.grad, torch.ones_like(code), rtol=0, atol=0)


def test_not_keeps_the_explicit_pole_exchange():
    from Language import NotLayer
    layer = NotLayer(representation='poles')
    poles = torch.tensor([[.8, .3], [.6, .6], [0., 0.]])
    torch.testing.assert_close(layer(poles), poles.flip(-1), rtol=0, atol=0)
    torch.testing.assert_close(layer.reverse(layer(poles)), poles, rtol=0, atol=0)
