"""A fixed reader input with a cotangent only to its admission parameters.

Answer readers must not optimize the codes or grammar they read. Attention
still needs the loss derivative of that read: it controls which existing
values are admitted. This boundary exposes precisely those parameters as
autograd inputs, while cutting the ordinary representation edge.
"""
import torch


@torch.compiler.allow_in_graph
class _AdmissionRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, *parameters):
        ctx.source = value
        ctx.parameters = parameters
        return value.detach()

    @staticmethod
    def backward(ctx, cotangent):
        with torch.enable_grad():
            gradients = torch.autograd.grad(ctx.source, ctx.parameters,
                grad_outputs=cotangent, allow_unused=True, retain_graph=True)
        return (None, *gradients)


def fixed_input(value, attention=None):
    """Keep the reader's exact forward value and its existing state cut."""
    if attention is None or not torch.is_grad_enabled() or not value.requires_grad:
        return value.detach()
    parameters = tuple(p for p in attention.parameters() if p.requires_grad)
    return _AdmissionRead.apply(value, *parameters) if parameters else value.detach()
