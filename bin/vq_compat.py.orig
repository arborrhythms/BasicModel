"""Compatibility wrapper for vector quantization backends.

Uses ``vector_quantize_pytorch`` when available. If that import fails,
falls back to a minimal in-repo implementation that supports the subset
of the API used by BasicModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from vector_quantize_pytorch import ResidualVQ, VectorQuantize
except Exception:
    class VectorQuantize(nn.Module):
        def __init__(
            self,
            dim,
            codebook_size,
            commitment_weight=1.0,
            use_cosine_sim=False,
            **kwargs,
        ):
            super().__init__()
            self.dim = dim
            self.codebook_size = codebook_size
            self.commitment_weight = commitment_weight
            self.use_cosine_sim = use_cosine_sim
            self.codebook = torch.randn(codebook_size, dim)

        @property
        def codebook(self):
            return self._parameters["_codebook"]

        @codebook.setter
        def codebook(self, value):
            if isinstance(value, nn.Parameter):
                param = value
            else:
                param = nn.Parameter(value.detach().clone())
            if "_codebook" in self._parameters:
                self._parameters["_codebook"] = param
            else:
                self.register_parameter("_codebook", param)
            self.codebook_size = param.shape[0]

        def forward(self, x, return_all_codes=False, **kwargs):
            original_shape = x.shape
            flat = x.reshape(-1, original_shape[-1])
            codebook = self.codebook
            if self.use_cosine_sim:
                flat_cmp = F.normalize(flat, dim=-1)
                codebook_cmp = F.normalize(codebook, dim=-1)
                indices = (flat_cmp @ codebook_cmp.T).argmax(dim=-1)
            else:
                distances = torch.cdist(flat, codebook)
                indices = distances.argmin(dim=-1)
            quantized_raw = codebook[indices].reshape(original_shape)
            commit_loss = self.commitment_weight * F.mse_loss(
                x, quantized_raw.detach()
            )
            quantized = x + (quantized_raw - x).detach()
            indices = indices.reshape(original_shape[:-1])
            if return_all_codes:
                return quantized, indices, commit_loss, quantized.unsqueeze(0)
            return quantized, indices, commit_loss

    class ResidualVQ(nn.Module):
        def __init__(self, dim, codebook_size, num_quantizers=1, **kwargs):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    VectorQuantize(
                        dim=dim,
                        codebook_size=codebook_size,
                        commitment_weight=kwargs.get("commitment_weight", 1.0),
                        use_cosine_sim=kwargs.get("use_cosine_sim", False),
                    )
                    for _ in range(num_quantizers)
                ]
            )

        def forward(self, x, return_all_codes=False, **kwargs):
            residual = x
            quantized_total = torch.zeros_like(x)
            all_indices = []
            all_codes = []
            total_loss = x.new_tensor(0.0)
            for layer in self.layers:
                quantized, indices, commit_loss = layer(residual, **kwargs)
                quantized_total = quantized_total + quantized
                residual = residual - quantized.detach()
                total_loss = total_loss + commit_loss
                all_indices.append(indices)
                all_codes.append(quantized)
            stacked_indices = torch.stack(all_indices, dim=0)
            stacked_codes = torch.stack(all_codes, dim=0)
            if return_all_codes:
                return quantized_total, stacked_indices, total_loss, stacked_codes
            return quantized_total, stacked_indices, total_loss
