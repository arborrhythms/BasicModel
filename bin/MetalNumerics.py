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
def clock_code(present: torch.Tensor, stamp: torch.Tensor, low: float, high: float,
               save_wires: bool) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(present, stamp)
    out = torch.empty_like(stamp, memory_format=torch.contiguous_format)
    saved = stamp.new_empty((2, *stamp.shape) if save_wires else (0,))
    if out.numel():
        shader('metal_clock', (low, high)).clock_code(
            out, present.contiguous(), stamp.contiguous(), saved, stamp.numel() if save_wires else 0,
            threads=[out.numel() // 2])
    return out, saved


@clock_code.register_fake
def _(present, stamp, low, high, save_wires):
    return (torch.empty_like(stamp, memory_format=torch.contiguous_format),
            stamp.new_empty((2, *stamp.shape) if save_wires else (0,)))


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
    _, _, ctx.low, ctx.high, _ = inputs
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(output[1])


def _backward(ctx, grad, saved_grad):
    saved, = ctx.saved_tensors
    # A higher derivative through the saved wires reaches the original stamp.
    extra = None if saved_grad is None else saved_grad[1]
    if grad is None:
        return None, extra, None, None, None
    present, stamp = saved.unbind(0)
    if torch.is_grad_enabled():
        # Higher derivatives use the differentiable reference operations.
        from RowTime import _rotate
        with torch.enable_grad():
            point = stamp.cpu().double()
            value = _rotate(present.cpu().double(), point, ctx.low, ctx.high)
            value = torch.where(point.ne(0).any(-1, keepdim=True), value, 0.)
            derivative, = torch.autograd.grad(value, point, grad.cpu().double(), create_graph=True)
        result = derivative.to(stamp.dtype).to(stamp.device)
    else:
        result = stamp_gradient(present, stamp, grad, ctx.low, ctx.high)
    return None, result if extra is None else result + extra, None, None, None


clock_code.register_autograd(_backward, setup_context=_setup)


@torch.library.custom_op('basicmodel::metal_pair_statistics', mutates_args=(), device_types='mps')
def pair_statistics(old: torch.Tensor, observation: torch.Tensor, value: torch.Tensor,
                    gathered: torch.Tensor, steps: torch.Tensor, penalty: torch.Tensor) -> torch.Tensor:
    result = old.new_empty((old.shape[0], 5))
    if result.numel():
        shader('pair_statistics').pair_statistics(result, old.contiguous(), observation.contiguous(),
            value.contiguous(), gathered.contiguous(), steps.contiguous(), penalty, threads=[len(old)])
    return result


@pair_statistics.register_fake
def _(old, observation, value, gathered, steps, penalty):
    return old.new_empty((old.shape[0], 5))


@torch.library.custom_op('basicmodel::metal_activity_update', mutates_args=(), device_types='mps')
def activity_update(current: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(current, memory_format=torch.contiguous_format)
    if result.numel():
        shader('activity_update').activity_update(result, current.contiguous(), increments.contiguous(),
                                                   threads=[result.numel()])
    return result


@activity_update.register_fake
def _(current, increments):
    return torch.empty_like(current, memory_format=torch.contiguous_format)
