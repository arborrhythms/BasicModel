"""Shared conservative defaults for PyTorch's Metal allocator."""

import os


MPS_HIGH_WATERMARK_RATIO = "0.60"
MPS_LOW_WATERMARK_RATIO = "0.50"


def apply_mps_allocator_env(env=None):
    """Set bounded MPS allocator defaults without replacing caller overrides."""
    target = os.environ if env is None else env
    target.setdefault(
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        MPS_HIGH_WATERMARK_RATIO,
    )
    target.setdefault(
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        MPS_LOW_WATERMARK_RATIO,
    )
    return target
