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


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

def _patch_inductor_paths():
    """Monkey-patch torch inductor to handle paths containing spaces.

    The inductor's CppBuilder assembles -L/-I flags as unquoted strings,
    then shlex.split() breaks paths with spaces (e.g. iCloud paths).
    This patch replaces the space-containing iCloud path with its /bits
    symlink equivalent in the compile command before shlex.split runs.
    """
    try:
        from torch._inductor import cpp_builder
        _orig = cpp_builder._run_compile_cmd
        _bits = "/bits"
        if os.path.islink(_bits):
            _target = os.readlink(_bits)
            def _patched_run_compile_cmd(cmd_line, cwd):
                cmd_line = cmd_line.replace(_target, _bits)
                return _orig(cmd_line, cwd)
            cpp_builder._run_compile_cmd = _patched_run_compile_cmd
    except Exception:
        pass


def compile(model, verbose=True):
    """Try to torch.compile the model; return the (possibly compiled) model.

    On GPU devices, patches inductor paths first.
    Tries default compile, then explicit inductor backend as fallback.
    """
    def _msg(text):
        if verbose:
            print(text)

    if TheDevice.type == "gpu" or TheDevice.type == "cuda":
        _patch_inductor_paths()

    try:
        model = torch.compile(model)
        # Extract the actual backend name from the compiled model
        backend_name = "unknown"
        try:
            backend_name = (model.dynamo_ctx.callback
                           ._torchdynamo_orig_backend._inner_convert
                           ._torchdynamo_orig_backend._compiler_name)
        except Exception:
            pass
        _msg(f"Model compiled ({backend_name})")
    except Exception as e:
        _msg(f"Model compile failed: {e}")
        try:
            model = torch.compile(model, backend='eager')
            _msg("Model compiled (eager)")
        except Exception:
            _msg("Model compilation failed, running eager")

    return model


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
