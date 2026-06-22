"""Shared utilities for the basicmodel project."""

import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from contextlib import nullcontext
from functools import lru_cache
import torch


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def init_runtime_env():
    """Configure writable runtime/cache paths before optional GUI imports.

    This keeps CLI runs from stalling while matplotlib/fontconfig try to build
    caches under unwritable home directories (common in sandboxed or iCloud
    workspaces). Environment overrides still win because ``setdefault`` is used.
    """
    runtime_root = os.path.join(tempfile.gettempdir(), "basicmodel-runtime")
    mpl_dir = os.path.join(runtime_root, "mplconfig")
    cache_dir = os.path.join(runtime_root, "cache")
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    return runtime_root


# Initialize runtime/cache paths on import so callers do not need to do it manually.
RUNTIME_ROOT = init_runtime_env()

class DeviceHandle(str):
    """str subclass representing a torch device that adds an optimized() query.

    Inherits from str so it can be passed directly wherever PyTorch expects a
    device string (device=TheDevice, tensor.to(TheDevice), etc.).
    """

    @property
    def type(self):
        """Device type string, e.g. 'cpu', 'cuda', 'mps'.

        Strips any trailing ``:N`` index suffix.
        """
        return self.split(":")[0]

    @property
    def index(self):
        """Device index, or None for devices without an index.

        Parses the ``:N`` suffix when present; bare device strings
        (e.g. ``cpu``, ``mps``) return None.
        """
        parts = self.split(":")
        return int(parts[1]) if len(parts) > 1 else None

    def optimized(self):
        """True when running on a real accelerator (cuda or mps), False on cpu.

        Cheap check used to gate accelerator-only fast paths so a CPU
        run does not pay autocast / cuDNN-prep overhead.
        """
        return self.type != "cpu"




def resolve_device(name=""):
    """Resolve a device name string to a torch.device.

    Special cases:
      - 'gpu'      -> best available accelerator (cuda > mps > cpu)
      - 'cuda[:N]' -> require CUDA to be available
      - 'mps'      -> require MPS to be available
      - 'cpu'      -> CPU

    Empty string or None falls back to CPU. Any other non-empty string is
    delegated to ``torch.device()`` so standard PyTorch device syntaxes such
    as ``xpu`` remain usable.
    """
    name = (name or "").strip().lower()
    if not name:
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            _warn_if_cuda_unavailable_but_nvidia_visible()
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            _warn_if_cuda_unavailable_but_nvidia_visible()
            raise ValueError(f"Requested device '{name}' but CUDA is not available.")
        return torch.device(name)
    if name.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps' but MPS is not available.")
        return torch.device(name)
    try:
        return torch.device(name)
    except (TypeError, RuntimeError, ValueError) as e:
        raise ValueError(f"Unknown device override '{name}'.") from e

@lru_cache(maxsize=1)
def _visible_nvidia_gpu_present():
    """Best-effort check for a process-visible NVIDIA GPU on non-macOS hosts.

    Inspects ``CUDA_VISIBLE_DEVICES`` (returns False for ``-1`` /
    ``none``), checks for ``/dev/nvidia*`` device nodes, and falls
    back to ``nvidia-smi -L``. Cached so the probe runs at most once.
    """
    if platform.system() == "Darwin":
        return False

    hidden = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
    if hidden in {"-1", "none"}:
        return False

    for path in ("/dev/nvidiactl", "/dev/nvidia0", "/proc/driver/nvidia/version"):
        if os.path.exists(path):
            return True

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


@lru_cache(maxsize=1)
def _warn_if_cuda_unavailable_but_nvidia_visible():
    """Warn once when an NVIDIA GPU appears visible but PyTorch CUDA is unavailable.

    Diagnoses the common "PyTorch built without CUDA, falling back to
    CPU silently" trap. ``@lru_cache`` ensures only one warning fires
    per process even if multiple device resolves attempt it.
    """
    if platform.system() == "Darwin" or torch.cuda.is_available():
        return
    if _visible_nvidia_gpu_present():
        warnings.warn(
            "NVIDIA GPU detected, but torch.cuda.is_available() is False. "
            "Falling back to CPU. Check that this environment has CUDA-enabled "
            "PyTorch and compatible NVIDIA drivers.",
            RuntimeWarning,
            stacklevel=2,
        )


def auto_device():
    """Select the runtime device using the same policy as resolve_device().

    Honors ``BASICMODEL_DEVICE`` when set; otherwise falls back to the
    'gpu' alias (cuda > mps > cpu). Returns a DeviceHandle so callers
    can ``.optimized()`` on the result.
    """
    override = os.environ.get("BASICMODEL_DEVICE", "").strip().lower()
    target = override if override else "gpu"
    return DeviceHandle(str(resolve_device(target)))

class _DeviceHolder:
    """Mutable container for the process-wide device.

    ``from util import TheDevice`` imports this holder once.  Call
    ``TheDevice.get()`` to obtain the current ``DeviceHandle``, or pass
    ``TheDevice`` directly where PyTorch expects a device (``__str__``
    delegates to the live value).

    ``init_device()`` updates the held value; all importers see the change
    immediately through ``.get()`` and ``str(TheDevice)``.
    """
    __slots__ = ('_device',)

    def __init__(self, device):
        """Hold the initial device; later changes use ``set``."""
        self._device = device

    def get(self):
        """Return the current DeviceHandle.

        Callers should use this rather than reading ``_device`` so the
        module-level singleton remains the single source of truth.
        """
        return self._device

    def set(self, device):
        """Update the held device.

        Any importer that re-reads ``TheDevice.get()`` will see the
        change immediately; existing references to the prior handle
        are not updated.
        """
        self._device = device

    # Convenience: str(TheDevice) returns the device string so existing
    # code like ``f"Device: {TheDevice}"`` still works.
    def __str__(self):
        """Delegate to the held device's string form."""
        return str(self._device)

    def __repr__(self):
        """Repr including the wrapping holder name."""
        return f"TheDevice({self._device!r})"

    # Delegate DeviceHandle properties for direct access
    @property
    def type(self):
        """Delegate to the held DeviceHandle's ``type``."""
        return self._device.type

    @property
    def index(self):
        """Delegate to the held DeviceHandle's ``index``."""
        return self._device.index

    def optimized(self):
        """Delegate to the held DeviceHandle's ``optimized()``."""
        return self._device.optimized()


# The canonical device for this process.
TheDevice = _DeviceHolder(auto_device())
torch.set_default_device(str(TheDevice.get()))


def init_device(device=None):
    """Override the process-wide device.  None re-runs auto-detection.

    Updates both the ``TheDevice`` holder and PyTorch's default device
    so subsequent tensor constructors land on the right backend.
    """
    TheDevice.set(auto_device() if device is None else DeviceHandle(str(device)))
    torch.set_default_device(str(TheDevice.get()))


def buffer(*size, **kwargs):
    """Allocate a zero tensor on TheDevice.  Accepts the same args as torch.zeros.

    Saves boilerplate at every call site that wants a scratch tensor
    on the active device without repeating ``device=TheDevice.get()``.
    """
    return torch.zeros(*size, **kwargs, device=TheDevice.get())


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_PARSE_WORD_RE     = re.compile(r'[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]|\s')
_PARSE_SENTENCE_RE = re.compile(r'[^.!?]*[.!?]+|[^.!?]+$')


def parse(data, lex="words"):
    """Tokenize *data* at the granularity given by *lex*.

    Modes:
        "bytes"     -- one token per input byte (``chr(byte)``, offset).
                       Operates on raw bytes / byte tensors directly so
                       there is no UTF-8 decode/encode round-trip --
                       tokenizing N input bytes always returns exactly
                       N tokens. (The legacy decode-then-encode path
                       could expand the byte count when the input
                       contained invalid UTF-8 sequences -- a cursor
                       slab cut mid-multi-byte produces a U+FFFD
                       replacement that re-encodes to 3 bytes, so
                       1023 raw bytes could become 1028 tokens.)
        "words"     -- regex words / punctuation / whitespace runs.
        "sentences" -- split at [.!?]; trailing non-terminated text
                       becomes a final token.

    Returns a list of ``(token_text, byte_start)`` tuples in every mode.
    Byte offsets cover the original string contiguously (words mode) or
    point to the sentence's first non-whitespace character (sentences mode).
    """
    if lex in ("bytes", "byte"):
        # Direct byte path: avoid str round-trip so the token count
        # equals the input byte count exactly.
        if torch.is_tensor(data):
            raw = bytes(data.to(torch.uint8).flatten().tolist())
        elif isinstance(data, (bytes, bytearray, memoryview)):
            raw = bytes(data)
        elif isinstance(data, str):
            raw = data.encode('utf-8')
        else:
            raw = bytes(data)
        return [(chr(b), i) for i, b in enumerate(raw)]

    if isinstance(data, bytes):
        text = data.decode('utf-8', errors='replace')
    elif isinstance(data, str):
        text = data
    else:
        text = str(data)

    if lex in ("sentences", "sentence"):
        spans = []
        for m in _PARSE_SENTENCE_RE.finditer(text):
            chunk = m.group()
            stripped = chunk.lstrip()
            if not stripped:
                continue
            leading = len(chunk) - len(stripped)
            char_start = m.start() + leading
            byte_start = len(text[:char_start].encode('utf-8'))
            spans.append((stripped, byte_start))
        return spans

    if lex in ("words", "word"):
        spans = []
        byte_pos = 0
        for m in _PARSE_WORD_RE.finditer(text):
            tok = m.group()
            spans.append((tok, byte_pos))
            byte_pos += len(tok.encode('utf-8'))
        return spans

    raise ValueError(
        f"parse(): unknown lex mode {lex!r}; "
        f"expected one of 'bytes', 'words', 'sentences'."
    )


# ---------------------------------------------------------------------------
# Compilation backend management
# ---------------------------------------------------------------------------

_COMPILE_BACKENDS = ("inductor", "eager", "aot_eager")
_COMPILE_OFF      = frozenset({"none", "off", "false", "0", "no"})

def auto_compile_backend():
    """Select the compilation backend from MODEL_COMPILE env var.

    Recognised values (case-insensitive):
      none / off / false / 0 / no  -> skip compilation entirely (return "none")
      inductor                     -> force torch.compile(backend='inductor')
      eager                        -> force torch.compile(backend='eager')
      aot_eager                    -> force torch.compile(backend='aot_eager')
      auto                         -> try inductor -> eager -> aot_eager
      (empty / unset)              -> "auto"
    """
    override = os.environ.get("MODEL_COMPILE", "").strip().lower()
    if not override:
        # Default backend by device. The inductor -> Metal (MPS) backend has an
        # upstream codegen bug: a multi-dim reduction emits an undeclared
        # ``r0_0`` index, so MM_20M / MM_xor hard-fail to compile on MPS
        # (program_source: "use of undeclared identifier 'r0_0'"). Use the GPU
        # EAGERLY there (torch.compile backend='eager' -- dynamo, no inductor)
        # until PyTorch fixes inductor-MPS. CUDA / CPU keep the inductor "auto"
        # path. Override explicitly with MODEL_COMPILE=inductor to force it.
        try:
            _dev = str(resolve_device(
                os.environ.get("BASICMODEL_DEVICE", "").strip().lower() or "gpu"))
        except Exception:
            _dev = "cpu"
        if _dev.startswith("mps"):
            return "eager"
        return "auto"
    if override in _COMPILE_OFF:
        return "none"
    if override == "auto":
        return "auto"
    if override in _COMPILE_BACKENDS:
        return override
    raise ValueError(
        f"Unknown MODEL_COMPILE value '{override}'. "
        f"Valid values: none, auto, {', '.join(_COMPILE_BACKENDS)}"
    )

# The canonical compilation backend for this process.
TheCompileBackend = auto_compile_backend()


def init_compile_backend(backend=None):
    """Override the process-wide compilation backend.  None re-runs auto-detection.

    Mutates the module-level ``TheCompileBackend`` global. Subsequent
    calls to ``compile()`` use the new value.
    """
    global TheCompileBackend
    TheCompileBackend = auto_compile_backend() if backend is None else backend


_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

def auto_compile_mode():
    """Select torch.compile mode from ``MODEL_COMPILE_MODE`` env var.

    Recognised values (case-insensitive):
      default          -> Inductor kernel fusion only.
      reduce-overhead  -> Inductor + CUDAGraphs.
      max-autotune     -> Inductor + CUDAGraphs + autotune.
      max-autotune-no-cudagraphs
                       -> autotune without CUDAGraphs.
      (empty / unset)  -> "max-autotune".

    Default is ``max-autotune`` -- empirical winner on the GB10
    training bench (basicmodel/bin/bench_compile.py, MM_20M.xml,
    bf16 + trie BPE, training pass, --batches 8 each):

        eager                       mean=39.5s   min=22.5s  max=71.0s
        default                     mean=99.6s   min=24.3s  max=165.0s
        max-autotune-no-cudagraphs  mean=94.3s   min=24.0s  max=204.7s
        reduce-overhead             mean=32.4s   min=23.3s  max=59.6s
        max-autotune                mean=30.9s   min=24.9s  max=42.0s   <-- winner

    Earlier results from the test-pass bench (forward only, fp32, no
    trie) had ``max-autotune-no-cudagraphs`` as the winner because
    CUDAGraph capture of the per-shape diversity was a wall-clock
    sink. Under training (forward + backward + step) with trie BPE
    and bf16, the picture inverted: ``max-autotune`` and
    ``reduce-overhead`` (the CUDAGraph-bearing modes) now beat both
    the no-cudagraphs autotune and eager, with much smaller tails
    (max-autotune max=42s vs no-cudagraphs max=204s). Re-bench when
    architecture changes meaningfully (new spaces, subsymbolicOrder,
    N-halving width) since shape diversity drives the trade-off.
    """
    override = os.environ.get("MODEL_COMPILE_MODE", "").strip().lower()
    if not override:
        return "max-autotune"
    if override in _COMPILE_MODES:
        return override
    raise ValueError(
        f"Unknown MODEL_COMPILE_MODE value '{override}'. "
        f"Valid values: {', '.join(_COMPILE_MODES)}"
    )

# The canonical torch.compile mode for this process.
TheCompileMode = auto_compile_mode()


def init_compile_mode(mode=None):
    """Override the process-wide torch.compile mode. None re-runs env detection.

    Mutates the module-level ``TheCompileMode`` global so subsequent
    ``compile()`` calls pick up the new mode.
    """
    global TheCompileMode
    TheCompileMode = auto_compile_mode() if mode is None else mode


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------
#
# MODEL_DEBUG gates expensive in-training checks (finite-param guards, tensor
# stat dumps).  Two layers of control:
#   1. Python's ``-O`` flag strips ``assert`` statements at bytecode compile
#      time (``__debug__`` becomes False), so asserts used as guards cost
#      nothing in an optimized run.
#   2. The MODEL_DEBUG env var toggles the runtime branches that wrap the
#      asserts (and any side-effecting stat prints, which cannot be expressed
#      as asserts).  MODEL_DEBUG=0 short-circuits before the tensor ops.
# Callers should gate with ``if not MODEL_DEBUG: return`` and then use
# ``assert`` for the actual check so ``-O`` strips the inner work too.

_DEBUG_ON = {"1", "true", "yes", "on"}


def _read_model_debug():
    return os.environ.get("MODEL_DEBUG", "").strip().lower() in _DEBUG_ON


MODEL_DEBUG = _read_model_debug()


def init_model_debug(enabled=None):
    """Override the process-wide debug flag.  None re-reads MODEL_DEBUG.

    Mutates the module-level ``MODEL_DEBUG`` global. Truthy value
    enables expensive in-training stat / finite checks.
    """
    global MODEL_DEBUG
    MODEL_DEBUG = _read_model_debug() if enabled is None else bool(enabled)


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _inductor_space_prefixes():
    """Collect path prefixes that need quoting in inductor shell commands.

    Returns absolute, normalized paths containing spaces, sorted longest-
    first so the longer match in ``_rewrite_inductor_cmd_line`` wins
    over a shorter prefix. Cached because the set is process-stable.
    """
    candidates = {
        os.getcwd(),
        sys.prefix,
        sys.exec_prefix,
        sys.base_prefix,
        os.path.dirname(torch.__file__),
    }
    candidates.update(sys.path)
    candidates.update(sysconfig.get_paths().values())

    prefixes = []
    for path in candidates:
        if not path:
            continue
        path = os.path.normpath(os.path.abspath(path))
        if " " not in path:
            continue
        prefixes.append(path)
    return tuple(sorted(set(prefixes), key=len, reverse=True))


def _rewrite_inductor_cmd_line(cmd_line, prefixes=None):
    """Quote known path prefixes so shlex.split keeps them intact.

    Wraps each unquoted occurrence of a known space-bearing prefix in
    double quotes. Defaults ``prefixes`` to ``_inductor_space_prefixes()``.
    """
    if prefixes is None:
        prefixes = _inductor_space_prefixes()
    for prefix in prefixes:
        cmd_line = re.sub(
            rf'(?<!["\']){re.escape(prefix)}(?!["\'])',
            f'"{prefix}"',
            cmd_line,
        )
    return cmd_line


def _patch_inductor_paths():
    """Monkey-patch torch inductor to quote path prefixes containing spaces.

    Wraps ``cpp_builder._run_compile_cmd`` to pre-rewrite the command
    line. Idempotent (sets ``_basicmodel_path_patch``). Silently
    suppresses any patch failure so the import side effect cannot
    break callers that don't compile.
    """
    try:
        from torch._inductor import cpp_builder
        if getattr(cpp_builder, "_basicmodel_path_patch", False):
            return
        _orig = cpp_builder._run_compile_cmd
        prefixes = _inductor_space_prefixes()
        if not prefixes:
            return

        def _patched_run_compile_cmd(cmd_line, cwd):
            return _orig(_rewrite_inductor_cmd_line(cmd_line, prefixes), cwd)

        cpp_builder._run_compile_cmd = _patched_run_compile_cmd
        cpp_builder._basicmodel_path_patch = True
    except Exception:
        pass


def compile(model, verbose=True, fullgraph=False):
    """Try to torch.compile the model; return the (possibly compiled) model.

    ``fullgraph=True`` is the strict no-graph-break gate: any graph
    break raises instead of silently falling back to eager around it.
    The caller passes it only for the non-grammar forward path (the
    recon-then-eliminate program drove that to 0 breaks); grammar-
    bearing configs pass ``fullgraph=False`` so the
    ``@torch.compiler.disable``'d grammar entry stays a tolerated
    (deferrable) break instead of a hard error.

    Respects ``TheCompileBackend`` (set by ``MODEL_COMPILE`` env var):
      "none" -> skip compilation entirely.
      "auto" -> try inductor -> eager -> aot_eager, return first success.
      <name> -> try only that backend; fall back to uncompiled on failure.

    Respects ``TheCompileMode`` (set by ``MODEL_COMPILE_MODE`` env var):
      "default"         -> Inductor kernel fusion only (no CUDAGraphs).
      "reduce-overhead" -> Inductor + CUDAGraphs; faster runtime, slow compile.
      "max-autotune"    -> Inductor + CUDAGraphs + autotune; slowest compile.

    Default mode is ``default`` (Inductor fusion, no CUDAGraph capture).
    Set ``MODEL_COMPILE_MODE=reduce-overhead`` (or ``max-autotune``) to
    opt into CUDAGraph capture. The CUDAGraph-bearing modes recompile
    per distinct static shape; for architectures with many shapes
    (N-halving / subsymbolicOrder), wall-clock can be dominated by
    per-shape capture. ``default`` skips that entirely.

    Patches inductor compile commands first when paths contain spaces.
    """
    def _msg(text):
        if verbose:
            print(text)

    if TheCompileBackend == "none":
        _msg("Model compilation skipped (MODEL_COMPILE=none)")
        _msg(f"Debug checks: {'ON (MODEL_DEBUG)' if MODEL_DEBUG else 'OFF'}"
             f"{'' if __debug__ else ' [asserts stripped by -O]'}")
        return model

    _msg(f"Debug checks: {'ON (MODEL_DEBUG)' if MODEL_DEBUG else 'OFF'}"
         f"{'' if __debug__ else ' [asserts stripped by -O]'}")
    _patch_inductor_paths()

    # dynamo specializes (guards on the value of) integer attributes of
    # nn.Modules. The grammar rule-cursor / generation counters
    # (``SyntacticLayer._cursor_compose_gen`` et al.) advance every
    # batch, so each change fails the guard and forces a recompile --
    # the measured ``unique_graphs > 1`` that prevents stable CUDAGraph
    # capture. Treat nn.Module int attributes as dynamic (PyTorch's own
    # recommended remedy for exactly this recompile symptom); the
    # cursor logic is plain Python control flow, not graph structure,
    # so unspecializing it changes no numerics.
    import torch._dynamo as _dyn
    _dyn.config.allow_unspec_int_on_nn_module = True

    # MPS: Inductor injects a device-side error buffer
    # (``c10::metal::ErrorMessages* error_buf``) into any kernel whose
    # codegen adds the "error" header. A large fusion (the MM-scale 2N×N
    # glue + slot-cascade kernels run past 10k generated Metal lines) can
    # then exceed Metal's hard 31-buffer binding limit and kernel
    # compilation fails with "no 'buffer' resource location available for
    # 'error_buf'" — observed on ``MM_20M.xml`` (2026-06-11; reproduced
    # 2026-06-19). There are TWO emitters of that header in
    # ``torch/_inductor/codegen/mps.py``:
    #   1. ``check_bounds`` (indirect-LOAD bounds) -- gated by
    #      ``config.assert_indirect_indexing`` via ``generate_assert``.
    #   2. ``device_assert_async`` (``torch._check`` / ``aten._assert_async``,
    #      e.g. the embedding-scatter index validation) -- NOT gated by that
    #      config, so disabling (1) alone still leaves ``error_buf`` (the
    #      pre-2026-06-19 mitigation missed this and MM_20M still failed).
    # Both are runtime SAFETY asserts; dropping them on MPS makes
    # out-of-range / failed-check indices UNCHECKED (the pre-error_buf
    # behavior) but removes the extra buffer argument so the kernel stays
    # within the 31-buffer limit. CUDA / CPU keep their checks (no such
    # binding limit). The class-method no-ops cover the case where the
    # config change does not reach codegen; the config set covers any
    # out-of-process compile worker (config serializes, monkeypatches do
    # not).
    try:
        if str(TheDevice.get()).startswith("mps"):
            import torch._inductor.config as _ind
            _ind.assert_indirect_indexing = False
            from torch._inductor.codegen import mps as _mps_cg
            _mps_cg.MetalKernel.device_assert_async = (
                lambda self, cond, msg: None)
            _mps_cg.MetalKernel.check_bounds = (
                lambda self, expr, size, lower, upper: None)
            _msg("MPS: inductor device asserts disabled (check_bounds + "
                 "device_assert_async) -- Metal 31-buffer limit vs error_buf")
    except Exception:
        pass

    backends = _COMPILE_BACKENDS if TheCompileBackend == "auto" else (TheCompileBackend,)
    for backend in backends:
        try:
            compiled = torch.compile(
                model, mode=TheCompileMode, backend=backend,
                fullgraph=fullgraph)
            _msg(f"Model compiled ({backend}, mode={TheCompileMode}, "
                 f"fullgraph={fullgraph})")
            return compiled
        except Exception as e:
            _msg(f"Model compile failed ({backend}, mode={TheCompileMode}): {e}")

    _msg("Model compilation failed, running eager")
    return model


# ---------------------------------------------------------------------------
# Automatic mixed precision
# ---------------------------------------------------------------------------
#
# MODEL_AMP gates torch.autocast around forward/backward regions.  Default is
# off (bit-identical to pre-change behavior).  Valid values:
#   off / none / false / 0 / no  -> fp32 (no autocast)
#   bf16                         -> torch.autocast(dtype=bfloat16), no scaler
#   fp16                         -> torch.autocast(dtype=float16); on CUDA
#                                   a singleton GradScaler is returned
# MPS has no working autocast path; fp16 on CPU is also unsupported by
# torch -- both fall back to fp32 with a one-shot warning.
# Env var wins; XML <amp> is a checked-in default applied only when the
# env var is unset (see ModelFactory.run for the hydration).

_AMP_MODES = ("off", "fp16", "bf16")
_AMP_OFF   = frozenset({"", "none", "off", "false", "0", "no"})
_AMP_WARNED = False
_AMP_FIRST_LOGGED = False


def _read_model_amp():
    """Parse ``MODEL_AMP`` env var into a canonical ``'off'``/``'fp16'``/``'bf16'``.

    Treats empty / common falsy spellings as ``'off'``. Raises
    ``ValueError`` on an unrecognized value so misconfigs surface
    loudly rather than silently disabling autocast.
    """
    raw = os.environ.get("MODEL_AMP", "").strip().lower()
    if raw in _AMP_OFF:
        return "off"
    if raw in _AMP_MODES:
        return raw
    raise ValueError(
        f"Unknown MODEL_AMP value {raw!r}. "
        f"Valid values: off, {', '.join(m for m in _AMP_MODES if m != 'off')}"
    )


MODEL_AMP = _read_model_amp()
TheAmpScaler = None  # lazily constructed on first fp16+cuda call


def init_model_amp(mode=None):
    """Override the process-wide AMP mode.  None re-reads MODEL_AMP.

    Resets the GradScaler singleton and the one-shot warning / log
    flags so the next ``amp_context()`` call reports the new mode.
    """
    global MODEL_AMP, TheAmpScaler, _AMP_WARNED, _AMP_FIRST_LOGGED
    MODEL_AMP = _read_model_amp() if mode is None else mode
    TheAmpScaler = None
    _AMP_WARNED = False
    _AMP_FIRST_LOGGED = False


def autocast_compute_dtype(device, fallback=None):
    """Return the dtype that torch.autocast would coerce ops to for
    ``device``, or ``fallback`` (default: float32) when autocast is off.

    Use this to allocate intermediate buffers (e.g. CKY chart score /
    vector tensors) so subsequent in-place writes from autocast-promoted
    ops don't hit a destination/source dtype mismatch.

    PyTorch's autocast only promotes the *output* of ops it tracks; pre-
    allocated buffers keep their original dtype, so ``empty_score =
    torch.full(..., dtype=data.dtype)`` followed by ``empty_score[...] =
    nn.Linear(data)`` raises when autocast is on (linear emits bf16, the
    buffer is fp32). This helper picks the right buffer dtype up front.
    """
    if fallback is None:
        fallback = torch.float32
    try:
        dev_type = device.type if hasattr(device, 'type') else str(device)
    except Exception:
        dev_type = "cpu"
    # PyTorch 2.4+ unified API. Handles all device types ('cpu',
    # 'cuda', 'mps', etc.) — the older per-device APIs
    # (``is_autocast_cpu_enabled`` / ``get_autocast_cpu_dtype``) are
    # deprecated in current PyTorch and have been dropped here.
    try:
        if torch.is_autocast_enabled(dev_type):
            return torch.get_autocast_dtype(dev_type)
    except (TypeError, AttributeError):
        pass
    return fallback


def amp_context():
    """Return ``(autocast_cm, scaler)`` for the current MODEL_AMP setting.

    The context manager is a fresh instance each call (autocast CMs are
    single-use).  The scaler is a process-wide singleton so optimizer
    step/update bookkeeping stays consistent across batches.

    Prints a one-shot status line on the first call so the operator can
    confirm at a glance which path engaged (bf16 active / off / MPS
    fallback). Re-armed by ``init_model_amp``.
    """
    global TheAmpScaler, _AMP_WARNED, _AMP_FIRST_LOGGED
    mode = MODEL_AMP
    if mode == "off":
        if not _AMP_FIRST_LOGGED:
            print("[AMP] off (fp32, MODEL_AMP unset or =off)")
            _AMP_FIRST_LOGGED = True
        return nullcontext(), None
    dev = TheDevice.type  # "cpu" | "cuda" | "mps" (ROCm exposes "cuda")
    if dev == "mps":
        if not _AMP_WARNED:
            print(f"MODEL_AMP={mode} unsupported on MPS; running fp32.")
            _AMP_WARNED = True
        if not _AMP_FIRST_LOGGED:
            print(f"[AMP] {mode} requested but disabled on MPS -> fp32")
            _AMP_FIRST_LOGGED = True
        return nullcontext(), None
    if dev == "cpu" and mode == "fp16":
        if not _AMP_WARNED:
            print("MODEL_AMP=fp16 unsupported on CPU; running fp32.")
            _AMP_WARNED = True
        if not _AMP_FIRST_LOGGED:
            print("[AMP] fp16 requested but disabled on CPU -> fp32")
            _AMP_FIRST_LOGGED = True
        return nullcontext(), None
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    cm = torch.autocast(device_type=dev, dtype=dtype)
    if not _AMP_FIRST_LOGGED:
        print(f"[AMP] {mode} active on {dev} (autocast dtype={dtype})")
        _AMP_FIRST_LOGGED = True
    if mode == "fp16" and dev == "cuda":
        if TheAmpScaler is None:
            TheAmpScaler = torch.amp.GradScaler("cuda")
        return cm, TheAmpScaler
    return cm, None


# ---------------------------------------------------------------------------
# Message sink
# ---------------------------------------------------------------------------

class Message():
    """Tiny callable wrapper so legacy code can swap out message sinks later.

    Always emits ``\\r\\n`` so output renders correctly both locally and when
    streamed over SSH with a pseudo-terminal (``ssh -t``).  ``flush=True``
    so progress is visible under shell redirect / nohup / log files (which
    block-buffer stdout by default and would otherwise hide progress until
    a 4-8 KB block fills).
    """
    def __call__(self, txt, newline="\r\n"):
        """Emit ``txt`` with a CRLF terminator, flushed for live logs."""
        print(txt, end=newline, flush=True)

TheMessage = Message()

# ---------------------------------------------------------------------------
# XML configuration
# ---------------------------------------------------------------------------

_MISSING = object()   # sentinel for required config lookups


class XMLConfig:
    """Centralized XML configuration store.

    Holds a merged dict parsed from one or more XML files.
    Supports dot-path access, section-scoped lookups, and
    reload/overlay for multi-file config inheritance.

    ``get()`` raises ``KeyError`` when the path is absent and no
    *default* is supplied -- so missing configuration is surfaced
    immediately rather than propagating ``None`` through the system.
    """

    def __init__(self, path=None, defaults_path=None):
        """Build an empty config; optionally seed from defaults and overlay path.

        ``defaults_path`` populates the base; ``path`` is overlay-merged
        on top. ``_requirements`` collects the keys callers declare as
        required so misconfigs raise rather than return None.
        """
        self._data = {}
        self._sources = []
        self._requirements = []
        if defaults_path:
            self.load(defaults_path)
        if path:
            self.overlay(path)

    # --- Loading ---

    def load(self, path):
        """Parse an XML file, replacing current data.

        Resets ``_sources`` to ``[path]``. Use ``overlay`` to merge
        instead of replacing.
        """
        self._data = self._parse_xml(path)
        self._apply_legacy_renames(self._data, path)
        self._sources = [path]

    # Sub-trees that an overlay file *replaces wholesale* rather than
    # deep-merging. Presence of any of these keys in the overlay
    # signals that the overlay XML takes full ownership of that
    # sub-tree -- the defaults' contents are dropped, not merged in.
    #
    # Currently a single entry: ``SymbolSpace.language.grammar``. A model
    # XML that defines its own <grammar> block fully owns the rule set;
    # the model.xml defaults' space_role-scoped grammar does not leak in.
    _NON_MERGING_PATHS = (
        ('SymbolSpace', 'language', 'grammar'),
    )

    def overlay(self, path):
        """Deep-merge another XML file on top of current data.

        Sub-trees listed in ``_NON_MERGING_PATHS`` are replaced
        wholesale when present in the overlay; other keys merge.
        """
        over = self._parse_xml(path)
        self._apply_legacy_renames(over, path)
        for section in over:
            if section not in self._data:
                self._data[section] = over[section]
            else:
                self._data[section] = self._deep_merge(
                    self._data[section], over[section],
                    path=(section,))
        self._sources.append(path)

    @staticmethod
    def _apply_legacy_renames(data, source_path):
        """Rewrite retired element names in a parsed config dict.

        Currently handles ``<reverseScale>`` -> ``<reconstructionScale>``
        (rename done 2026-05-14 alongside the maskedPrediction retirement;
        see basicmodel/doc/Training.md).  Emits a single deprecation
        warning per affected file so test churn stays bounded.
        """
        training = (data.get("architecture", {}) or {}).get("training", {})
        if not isinstance(training, dict):
            return
        if "reverseScale" in training and "reconstructionScale" not in training:
            import warnings
            warnings.warn(
                f"{source_path}: <reverseScale> is deprecated; "
                "rename to <reconstructionScale>.",
                DeprecationWarning, stacklevel=3)
            training["reconstructionScale"] = training.pop("reverseScale")
        elif "reverseScale" in training:
            training.pop("reverseScale")

    def reload(self):
        """Re-parse all previously loaded sources in order.

        Discards the in-memory ``_data`` and replays ``load`` /
        ``overlay`` for each recorded source. Useful when a source
        file is edited at runtime.
        """
        paths = list(self._sources)
        if not paths:
            return
        self.load(paths[0])
        for p in paths[1:]:
            self.overlay(p)

    # --- Access ---

    def get(self, dotted_path, default=_MISSING):
        """Dot-path lookup: ``cfg.get('architecture.training.numEpochs')``.

        Raises ``KeyError`` when the path is absent and no *default*
        is supplied.  Pass an explicit *default* (including ``None``)
        for genuinely optional / nullable config keys.
        """
        keys = dotted_path.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                if default is _MISSING:
                    raise KeyError(
                        f"Config key not found: {dotted_path!r} "
                        f"(missing at {k!r})"
                    )
                return default
        return node

    def set(self, dotted_path, value):
        """Dot-path setter for runtime overrides.

        Creates intermediate dicts along the path when missing so the
        caller doesn't have to pre-build the nesting structure.
        """
        keys = dotted_path.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def section(self, name):
        """Return a top-level section dict.

        Raises ``KeyError`` when the section is absent.
        """
        if name not in self._data:
            raise KeyError(f"Config section not found: {name!r}")
        return self._data[name]

    @property
    def data(self):
        """Raw dict access for backward compatibility.

        Returns the live ``_data`` dict (not a copy); callers must
        treat it as read-only.
        """
        return self._data

    # --- Requirements ---

    def require(self, check, description):
        """Register a config constraint.

        *check* is a callable ``check(cfg) -> bool``.  Returning
        ``False`` or raising an exception means the requirement failed.
        """
        self._requirements.append((check, description))

    def validate(self):
        """Run all registered requirements.  Raises ``ValueError`` on
        the first failure, then clears the list so a subsequent
        ``create()`` starts fresh.
        """
        try:
            for check, desc in self._requirements:
                try:
                    if check(self) is False:
                        raise ValueError(
                            f"Config requirement failed: {desc}")
                except KeyError as e:
                    raise ValueError(
                        f"Config requirement failed: {desc}") from e
        finally:
            self._requirements.clear()

    # --- Derived properties ---

    @property
    def objectSize(self):
        """nWhere + nWhen encoding overhead per vector.

        From canonical_shape("InputSpace") -- where/when are architectural
        constants (modality re-architecture), not per-config scalars.
        """
        from architecture import canonical_shape
        nw, nn = canonical_shape("InputSpace")
        return nw + nn

    @property
    def nObjects(self):
        """Total codebook vectors across all spaces.

        Sums ``nVectors`` across the five canonical spaces; absent
        sections are skipped (no error).
        """
        total = 0
        for s in ("InputSpace", "PartSpace", "ConceptualSpace",
                  "WholeSpace", "OutputSpace"):
            try:
                total += self.space(s, "nVectors")
            except KeyError:
                pass
        return total

    def encodingSize(self, nDim):
        """Full vector width: nDim + objectSize.

        Used by spaces to allocate their per-row buffers including
        the nWhere / nWhen encoding tail.
        """
        return nDim + self.objectSize

    # --- BasicModel convenience ---

    def training(self, key, default=_MISSING):
        """Shorthand for architecture.training.<key>.

        Raises ``KeyError`` when the key is absent and no *default*
        is supplied.
        """
        return self.get(f"architecture.training.{key}", default)

    def data_param(self, key, default=_MISSING):
        """Shorthand for architecture.data.<key>.

        Raises ``KeyError`` when the key is absent and no *default*
        is supplied.
        """
        return self.get(f"architecture.data.{key}", default)

    def data_type(self):
        """Resolve the data space_role: ``"embedding"`` (text / LM) or ``"numeric"``
        (dense slab). Canonical home is ``<data><dataType>`` (was the retired
        architecture-level ``<modelType>``); an architecture-level ``dataType``
        is accepted as a fallback for dict-based test overrides (which deep-
        merge at the architecture level and never round-trip through the XSD).
        Default ``"numeric"``."""
        dt = self.get("architecture.data.dataType", None)
        if dt is None:
            dt = self.get("architecture.dataType", "numeric")
        return dt

    _MISSING = object()

    def space(self, space_name, key, default=_MISSING):
        """Lookup key in <SpaceName>, fall back to <architecture>.

        Space sections are top-level siblings of <architecture> in the XML.
        Equivalent to BasicModelFactory.get_space_param.

        Raises ``KeyError`` when the key is absent from both the space
        section and the architecture section -- unless ``default`` is
        provided, in which case it is returned. Raises ``ValueError``
        when the same scalar key is listed more than once inside a
        space section (the XML parser folds repeated sibling tags into
        a list; for scalar config keys that's a dup-by-mistake whose
        downstream symptom -- ``int + list`` from arithmetic on the
        list -- is opaque, so we surface a pointed error here instead).
        """
        # Try space-specific section first (top-level)
        if space_name in self._data:
            space = self._data[space_name]
            if isinstance(space, dict) and key in space:
                v = space[key]
                if isinstance(v, list):
                    raise ValueError(
                        f"Config key {key!r} in section {space_name!r} "
                        f"is duplicated (parsed as {v!r}). Each scalar "
                        f"config key may appear at most once per space "
                        f"section. Remove the duplicate <{key}> tag(s) "
                        f"in the model XML."
                    )
                return v
        # Fall back to architecture-level default. The architecture section
        # itself may be absent in a partial/stale global config; that must
        # NOT shadow a supplied ``default`` (the docstring contract is "raises
        # only when the key is absent from both sections AND no default is
        # given"). Guard the section lookup so a missing <architecture> degrades
        # to the default rather than raising KeyError up into callers that
        # passed one (e.g. the heat-retrieval knob reads in
        # LanguageLayer.unreduce, where a raise would silently disable the
        # heat path).
        try:
            arch = self.section("architecture")
        except KeyError:
            arch = {}
        if key in arch:
            v = arch[key]
            if isinstance(v, list):
                raise ValueError(
                    f"Config key {key!r} in section 'architecture' "
                    f"is duplicated (parsed as {v!r}). Each scalar "
                    f"config key may appear at most once. Remove the "
                    f"duplicate <{key}> tag(s) in the model XML."
                )
            return v
        if default is not XMLConfig._MISSING:
            return default
        raise KeyError(
            f"Config key {key!r} not found in section {space_name!r} "
            f"or 'architecture'"
        )

    def nOutput(self, space_name):
        """Return raw nOutput for a space (0 = sentinel meaning 'same as nInput').

        Thin wrapper around ``self.space(space_name, 'nOutput')``;
        keeps sentinel handling at the call site.
        """
        return self.space(space_name, "nOutput")

    def tetralemma_policy(self, space_name):
        """Return (allow_excluded_middle, allow_contradiction, neither_threshold)
        for ``space_name``, applying inheritance from the shared
        ``<architecture><TetralemmaPolicy>`` block when
        ``<tetralemmaOverride enabled="false">``.

        Per spec O4 of the lift/lower/bivector design, the conceptual
        and symbolic layers share a default tetralemma policy unless a
        per-space override is enabled. ``space_name`` is the XML section
        tag ("ConceptualSpace", "WholeSpace", ...).

        Returns a 3-tuple. Defaults: permit NEITHER, forbid BOTH
        (Kleene). The shared block lives under ``<architecture>``;
        the legacy top-level ``<TetralemmaPolicy>`` is still read as a
        fallback for any config not yet migrated.
        """
        override_node = self.get(
            f"{space_name}.tetralemmaOverride", default=None)
        override_enabled = False
        if isinstance(override_node, dict):
            override_enabled = (str(override_node.get("enabled", "false"))
                                .strip().lower() == "true")
        if override_enabled:
            section = f"{space_name}.TetralemmaPolicy"
        else:
            section = "architecture.TetralemmaPolicy"
            if self.get(section, default=None) is None:
                section = "TetralemmaPolicy"
        return (
            int(self.get(f"{section}.allowExcludedMiddle", default=1)),
            int(self.get(f"{section}.allowContradiction", default=0)),
            float(self.get(f"{section}.neitherThreshold", default=0.1)),
        )

    def nInput(self, space_name):
        """Return raw nInput for a space (0 = sentinel meaning 'derive from previous space').

        Thin wrapper around ``self.space(space_name, 'nInput')``.
        """
        return self.space(space_name, "nInput")

    # --- Serialization ---

    def to_xml(self):
        """Serialize back to XML string.

        Round-trips ``_data`` into an ``<model>`` root. Lists become
        repeated sibling tags; booleans render lowercase ``true``/
        ``false``; ``"_"`` keys (text payloads) are dropped.
        """
        import xml.etree.ElementTree as ET

        def _dict_to_elem(tag, value):
            """Build an ET element (or list of elements) from a nested config value."""
            elem = ET.Element(tag)
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "_":
                        continue
                    elem.append(_dict_to_elem(k, v))
            elif isinstance(value, list):
                # Return multiple elements for lists
                elems = []
                for item in value:
                    elems.append(_dict_to_elem(tag, item))
                return elems
            elif isinstance(value, bool):
                elem.text = str(value).lower()
            else:
                elem.text = str(value)
            return elem

        root = ET.Element("model")
        for section, content in self._data.items():
            if isinstance(content, dict):
                root.append(_dict_to_elem(section, content))
            else:
                root.append(_dict_to_elem(section, content))
        return ET.tostring(root, encoding="unicode")

    # --- Parsing ---

    @staticmethod
    def _parse_xml(path):
        """Parse an XML file into a nested dict with auto-coercion.

        Leaf nodes are cast to bool/int/float/str.  Elements with children
        become nested dicts.  Repeated sibling tags aggregate into lists.
        Elements with XML attributes produce dicts with '_' for the text value.
        """
        import xml.etree.ElementTree as ET

        def _parse_element(elem):
            """Recursively coerce one element into a Python value or dict.

            Leaves return scalar (bool/int/float/str), or ``{"_": val, ...attrs}``
            when XML attributes are present. Element trees with children
            return a dict whose duplicate tags aggregate into lists.
            """
            children = list(elem)
            if not children:
                text = elem.text.strip() if elem.text else ""
                if text.lower() in ("true", "false"):
                    val = text.lower() == "true"
                else:
                    try:
                        val = int(text)
                    except ValueError:
                        try:
                            val = float(text)
                        except ValueError:
                            # Simple arithmetic: evaluate expressions
                            # containing only digits, *, +, -, /, parens,
                            # and whitespace (e.g. "8192*6", "4+2").
                            import re
                            if re.fullmatch(r'[\d\s\+\-\*\/\(\)\.]+', text):
                                try:
                                    result = eval(text)  # safe: only digits+ops
                                    val = int(result) if result == int(result) else float(result)
                                except Exception:
                                    val = text
                            else:
                                val = text
                if elem.attrib:
                    return {"_": val, **elem.attrib}
                return val
            d = {}
            for child in children:
                parsed = _parse_element(child)
                if child.tag in d:
                    existing = d[child.tag]
                    if isinstance(existing, list):
                        existing.append(parsed)
                    else:
                        d[child.tag] = [existing, parsed]
                else:
                    d[child.tag] = parsed
            return d

        if not os.path.exists(path):
            return {}
        XMLConfig._validate_against_schema(path)
        tree = ET.parse(path)
        root = tree.getroot()
        cfg = {}
        for section in root:
            cfg[section.tag] = _parse_element(section)
        return cfg

    # Cache of abspath -> mtime for configs that already validated cleanly,
    # so the defaults file and repeated loads don't re-run the validator.
    # A file re-validates automatically when its mtime changes.
    _SCHEMA_VALIDATION_CACHE = {}
    # One-time guard so the "no validator backend" notice prints once.
    _SCHEMA_VALIDATOR_WARNED = False

    @staticmethod
    def _validate_against_schema(xml_path):
        """Validate ``xml_path`` against the ``model.xsd`` schema.

        This is the precursor gate for model creation:
        ``create_from_config`` -> ``init_config`` -> ``load``/``overlay``
        parse every config through here, so an invalid config raises
        BEFORE any Space is constructed.

        Schema resolution prefers a ``model.xsd`` sibling of the XML, then
        falls back to the canonical ``<project>/data/model.xsd`` -- so temp
        / variant configs written outside ``data/`` still validate against
        the real schema.

        Hard-fails (raises ``ValueError``) on a schema violation. The check
        is skipped only when no validator backend is available (neither the
        ``lxml`` package nor the ``xmllint`` CLI); that case emits a
        one-time warning rather than silently passing, and does not block
        loading on missing tooling.
        """
        if not xml_path or not os.path.exists(xml_path):
            return
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(xml_path)), "model.xsd")
        canonical = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "model.xsd")
        xsd_path = sibling if os.path.exists(sibling) else (
            canonical if os.path.exists(canonical) else None)
        if xsd_path is None:
            return
        abspath = os.path.abspath(xml_path)
        try:
            mtime = os.path.getmtime(abspath)
        except OSError:
            mtime = None
        if (mtime is not None
                and XMLConfig._SCHEMA_VALIDATION_CACHE.get(abspath) == mtime):
            return
        errors = XMLConfig._run_schema_validation(xsd_path, abspath)
        if errors is None:
            return  # no validator backend; already warned once
        if errors:
            raise ValueError(
                f"XML schema validation failed for {xml_path}\n"
                f"  (schema: {xsd_path})\n{errors}\n"
                f"Fix the config, or update data/model.xsd if this is a "
                f"new config element.")
        if mtime is not None:
            XMLConfig._SCHEMA_VALIDATION_CACHE[abspath] = mtime

    @staticmethod
    def _run_schema_validation(xsd_path, xml_path):
        """Run the schema validator. Returns ``""`` when valid, an
        error-detail string when invalid, or ``None`` when no validator
        backend (the ``lxml`` package or the ``xmllint`` CLI) is available.
        """
        # Prefer lxml (in-process, fast) when installed.
        try:
            from lxml import etree as _lxml_etree
        except ImportError:
            _lxml_etree = None
        if _lxml_etree is not None:
            try:
                schema = _lxml_etree.XMLSchema(_lxml_etree.parse(xsd_path))
                doc = _lxml_etree.parse(xml_path)
            except Exception as exc:
                return f"could not parse schema/document: {exc}"
            if schema.validate(doc):
                return ""
            return "\n".join(
                f"  line {e.line}: {e.message}" for e in schema.error_log)
        # Fall back to the xmllint CLI (ships with libxml2; no Python dep).
        import shutil
        import subprocess
        xmllint = shutil.which("xmllint")
        if xmllint is None:
            if not XMLConfig._SCHEMA_VALIDATOR_WARNED:
                import warnings
                warnings.warn(
                    "XML schema validation skipped: neither the 'lxml' "
                    "package nor the 'xmllint' CLI is available. Install "
                    "either to enable config validation against "
                    "data/model.xsd.",
                    RuntimeWarning, stacklevel=4)
                XMLConfig._SCHEMA_VALIDATOR_WARNED = True
            return None
        try:
            proc = subprocess.run(
                [xmllint, "--noout", "--schema", xsd_path, xml_path],
                capture_output=True, text=True)
        except Exception:
            return None  # tooling failure -> skip, don't block on infra
        if proc.returncode == 0:
            return ""
        return (proc.stderr or "").strip()

    @staticmethod
    def _deep_merge(base, overlay, path=()):
        """Recursively merge overlay into base (overlay wins on conflicts).

        ``path`` tracks the dotted location of the current sub-tree
        relative to the document root, so the merge can short-circuit
        to wholesale-replacement at sub-trees registered in
        ``_NON_MERGING_PATHS``.
        """
        merged = dict(base)
        for k, v in overlay.items():
            sub_path = path + (k,)
            if sub_path in XMLConfig._NON_MERGING_PATHS:
                # Wholesale replacement: drop the defaults' sub-tree.
                merged[k] = v
                continue
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = XMLConfig._deep_merge(merged[k], v, path=sub_path)
            else:
                merged[k] = v
        return merged


# The canonical config for this process.
# Always load model.xml defaults so that Space constructors can read
# config keys even when create_from_config() hasn't been called yet.
_defaults_xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model.xml")
TheXMLConfig = XMLConfig(defaults_path=_defaults_xml if os.path.exists(_defaults_xml) else None)


def init_config(path=None, defaults_path=None):
    """Load (or reload) TheXMLConfig from file(s).

    Clears any stale requirements then re-runs ``load`` on the
    defaults and ``overlay`` on the main path. Mutates the
    process-wide ``TheXMLConfig`` singleton.
    """
    global TheXMLConfig
    TheXMLConfig._requirements.clear()  # clear stale requirements from prior create()/tests
    if defaults_path:
        TheXMLConfig.load(defaults_path)
    if path:
        TheXMLConfig.overlay(path)


# ---------------------------------------------------------------------------
# Path management
# ---------------------------------------------------------------------------

class ProjectPaths:
    """Centralized path resolution for the basicmodel project.

    Class-level constants for the project root and standard
    subdirectories (data, output). Helper classmethods resolve
    relative paths and ensure output dirs exist before write.
    """
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)  # basicmodel/ root
    DATA_DIR    = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")

    @classmethod
    def ensure_output_dir(cls):
        """Create ``OUTPUT_DIR`` if missing; return the absolute path."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR

    @classmethod
    def output_path(cls, filename):
        """Absolute path to ``OUTPUT_DIR/<filename>`` (creates the dir)."""
        return os.path.join(cls.ensure_output_dir(), filename)

    @classmethod
    def output_stem(cls, stem):
        """Absolute path to ``OUTPUT_DIR/<stem>`` (no extension joined)."""
        return os.path.join(cls.ensure_output_dir(), stem)

    @classmethod
    def resolve_xml(cls, path):
        """Resolve an XML path relative to PROJECT_DIR if not absolute.

        Lets call sites accept either short ``data/foo.xml`` or full
        absolute paths without branching at every entry point.
        """
        if not os.path.isabs(path):
            return os.path.join(cls.PROJECT_DIR, path)
        return path


def atomic_torch_save(obj, path):
    """torch.save to ``{path}.tmp`` then atomically rename onto ``path``.

    Leaves the original ``path`` untouched if the save fails. Any stale
    ``{path}.tmp`` orphan at the target (from a prior crashed save) is
    removed first, and a partial tmp from this save is cleaned up on
    failure so orphans cannot accumulate.
    """
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        torch.save(obj, tmp_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    os.replace(tmp_path, path)


_VALID_USE_GRAMMAR = ("all", "thoughtFree", "none")


def parse_use_grammar(value) -> str:
    """Normalize useGrammar config into the tri-state {"all", "thoughtFree", "none"}.

    Raises ``ValueError`` for any non-string or unknown value so XML
    typos surface at config-load time rather than silently disabling
    grammar dispatch downstream.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"useGrammar must be a string, got {type(value).__name__}"
        )
    if value not in _VALID_USE_GRAMMAR:
        raise ValueError(
            f"useGrammar must be one of {_VALID_USE_GRAMMAR}, got {value!r}"
        )
    return value
