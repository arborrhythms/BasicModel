"""Backend compatibility helpers for PyTorch operators."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class Workarounds:
    """Small exact rewrites for backend-specific PyTorch limitations."""

    @staticmethod
    def adaptive_avg_pool1d(u: torch.Tensor, output_size):
        """Exact ``adaptive_avg_pool1d`` wrapper that avoids MPS limitations.

        PyTorch's MPS backend only supports adaptive average pooling when the
        input length is divisible by the output length. In that divisible
        special case, adaptive average pooling is equivalent to ordinary
        average pooling with ``kernel_size == stride``. For non-divisible
        shapes, compute PyTorch's adaptive pooling windows from prefix sums
        instead of calling the unsupported MPS kernel.
        """
        if isinstance(output_size, Sequence):
            if len(output_size) != 1:
                raise ValueError(
                    "adaptive_avg_pool1d output_size must have one element")
            out_len = int(output_size[0])
        else:
            out_len = int(output_size)

        in_len = int(u.size(-1))
        if out_len > 0 and in_len % out_len == 0:
            kernel_size = in_len // out_len
            if u.dim() == 3:
                return F.avg_pool1d(
                    u,
                    kernel_size=kernel_size,
                    stride=kernel_size,
                )

        if out_len <= 0:
            return F.adaptive_avg_pool1d(u, output_size)

        flat = u.reshape(-1, in_len)
        prefix = F.pad(flat.cumsum(dim=-1), (1, 0))
        # MLX delegate gather kernels are happier with int32 indices than
        # exported int64 constants; PyTorch accepts int32 for index_select.
        bins = torch.arange(out_len, device=u.device, dtype=torch.int32)
        starts = torch.div(bins * in_len, out_len, rounding_mode="floor")
        ends = torch.div(
            (bins + 1) * in_len + out_len - 1,
            out_len,
            rounding_mode="floor",
        )
        starts = starts.clamp(min=0, max=in_len)
        ends = ends.clamp(min=0, max=in_len)
        sums = prefix.index_select(1, ends) - prefix.index_select(1, starts)
        lengths = (ends - starts).to(dtype=sums.dtype).clamp(min=1)
        pooled = sums / lengths.unsqueeze(0)
        return pooled.reshape(*u.shape[:-1], out_len)
