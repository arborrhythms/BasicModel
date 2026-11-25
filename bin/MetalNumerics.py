"""Native Metal kernels for the radius and clock computations that use binary64 on CPU."""
from functools import cache
import math
from pathlib import Path
import struct

import torch


@cache
def shader(name, steps=()):
    directory = Path(__file__).parent
    source = (directory / (name + '.metal')).read_text().replace(
        '#include "significand.metal"', (directory / 'significand.metal').read_text())
    if steps:
        def number(value):
            bits, = struct.unpack('=Q', struct.pack('=d', value))
            mantissa = (1 << 52) | (bits & ((1 << 52) - 1)) if value else 0
            return '{%dul, %d, %s}' % (mantissa, ((bits >> 52) & 2047) - 1023,
                                        'true' if value < 0 else 'false')
        for target, trig in (('TURN_SIN', math.sin), ('TURN_COS', math.cos)):
            source = source.replace(target, ', '.join(number(trig(step)) for step in steps))
    return torch.mps.compile_shader(source)


def _validate(present, stamp):
    if (stamp.ndim < 1 or stamp.shape[-1] != 4 or present.shape != stamp.shape
            or stamp.dtype != torch.float32 or present.dtype != stamp.dtype
            or present.device != stamp.device):
        raise ValueError('a clock rotation requires aligned four-wire float32 codes')


@torch.library.custom_op('basicmodel::metal_clock', mutates_args=(), device_types='mps')
def clock_code(present: torch.Tensor, stamp: torch.Tensor, low: float, high: float) -> torch.Tensor:
    _validate(present, stamp)
    out = torch.empty_like(stamp, memory_format=torch.contiguous_format)
    if out.numel():
        shader('metal_clock', (low, high)).clock_code(
            out, present.contiguous(), stamp.contiguous(), threads=[out.numel() // 2])
    return out


@clock_code.register_fake
def _(present, stamp, low, high):
    return torch.empty_like(stamp, memory_format=torch.contiguous_format)


@torch.library.custom_op('basicmodel::metal_clock_stamp_grad', mutates_args=(), device_types='mps')
def stamp_gradient(present: torch.Tensor, stamp: torch.Tensor, grad: torch.Tensor,
                   low: float, high: float) -> torch.Tensor:
    _validate(present, stamp)
    if grad.shape != stamp.shape or grad.dtype != stamp.dtype or grad.device != stamp.device:
        raise ValueError('the clock gradient must align with its four wires')
    out = torch.empty_like(stamp, memory_format=torch.contiguous_format)
    if out.numel():
        shader('metal_clock', (low, high)).clock_stamp_grad(
            out, present.contiguous(), stamp.contiguous(), grad.contiguous(), threads=[out.numel() // 2])
    return out


@stamp_gradient.register_fake
def _(present, stamp, grad, low, high):
    return torch.empty_like(stamp, memory_format=torch.contiguous_format)


def _setup(ctx, inputs, output):
    present, stamp, ctx.low, ctx.high = inputs
    ctx.save_for_backward(present, stamp)


def _backward(ctx, grad):
    present, stamp = ctx.saved_tensors
    if torch.is_grad_enabled():
        # Higher derivatives use the differentiable reference operations.
        from RowTime import _rotate
        with torch.enable_grad():
            point = stamp.to('cpu', torch.float64)
            value = _rotate(present.to('cpu', torch.float64), point, ctx.low, ctx.high)
            value = torch.where(point.ne(0).any(-1, keepdim=True), value, 0.)
            derivative, = torch.autograd.grad(value, point, grad.to(value), create_graph=True)
        return None, derivative.to(stamp), None, None
    return None, stamp_gradient(present, stamp, grad, ctx.low, ctx.high), None, None


clock_code.register_autograd(_backward, setup_context=_setup)
