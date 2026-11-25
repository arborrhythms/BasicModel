"""The native binary64 trust law, with one final rounding into the float32 chart."""
import math
import struct

import torch


def gain_bits(horizon):
    # At forty time constants, 1 - exp(-n/horizon) rounds to exactly one
    # in binary64. Unusually large horizons keep the existing scalar reader.
    if horizon > 65536 / 40:
        return None
    last = max(1, math.ceil(40 * horizon))
    gains = [-math.expm1(-count / horizon) for count in range(last + 1)]
    return struct.unpack(f'={len(gains)}q', struct.pack(f'={len(gains)}d', *gains))


@torch.library.custom_op('basicmodel::metal_trust_radius', mutates_args=(), device_types='mps')
def metal_radius(views: torch.Tensor, agreement: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    if (views.shape != agreement.shape or views.dtype != torch.long or agreement.dtype != torch.float32
            or gains.dtype != torch.long or gains.ndim != 1 or not len(gains)
            or views.device != agreement.device or gains.device != agreement.device):
        raise ValueError('trust requires aligned int64 counts, float32 agreements and a binary64 gain table')
    result = torch.empty_like(agreement, memory_format=torch.contiguous_format)
    if result.numel():
        from MetalNumerics import shader
        shader('trust_radius').trust_radius(result, views.contiguous(), agreement.contiguous(), gains.contiguous(),
                                     len(gains) - 1, threads=[result.numel()])
    return result


@metal_radius.register_fake
def _(views, agreement, gains):
    return torch.empty_like(agreement, memory_format=torch.contiguous_format)
