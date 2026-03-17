"""Shared utilities for the basicmodel project."""

import os
import torch


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def resolve_device(name=""):
    """Resolve a device name string to a torch.device.

    Maps 'gpu' to the best available GPU backend (cuda > mps > cpu).
    Empty string or None falls back to cpu.
    """
    name = (name or "").strip().lower()
    if name == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name) if name else torch.device("cpu")


def auto_device():
    """Select the best device: BASICMODEL_DEVICE env var > cuda > mps > cpu."""
    override = os.environ.get("BASICMODEL_DEVICE", "").strip().lower()
    if override:
        return resolve_device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# The canonical device for this process.
TheDevice = auto_device()


def init_device(device=None):
    """Override the process-wide device.  None re-runs auto-detection."""
    global TheDevice
    TheDevice = auto_device() if device is None else device


def buffer(*size, **kwargs):
    """Allocate a zero tensor on TheDevice.  Accepts the same args as torch.zeros."""
    return torch.zeros(*size, **kwargs, device=TheDevice)


class ProjectPaths:
    """Centralized path resolution for the basicmodel project."""
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)  # basicmodel/ root
    DATA_DIR    = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")

    @classmethod
    def ensure_output_dir(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR

    @classmethod
    def output_path(cls, filename):
        return os.path.join(cls.ensure_output_dir(), filename)

    @classmethod
    def output_stem(cls, stem):
        return os.path.join(cls.ensure_output_dir(), stem)

    @classmethod
    def resolve_xml(cls, path):
        """Resolve an XML path relative to PROJECT_DIR if not absolute."""
        if not os.path.isabs(path):
            return os.path.join(cls.PROJECT_DIR, path)
        return path
