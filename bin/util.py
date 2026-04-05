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
        """Device type string, e.g. 'cpu', 'cuda', 'mps'."""
        return self.split(":")[0]

    @property
    def index(self):
        """Device index, or None for devices without an index."""
        parts = self.split(":")
        return int(parts[1]) if len(parts) > 1 else None

    def optimized(self):
        """Return True when running on a real accelerator (cuda or mps), False on cpu."""
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
    """Best-effort check for a process-visible NVIDIA GPU on non-macOS hosts."""
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
    """Warn once when an NVIDIA GPU appears visible but PyTorch CUDA is unavailable."""
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
    """Select the runtime device using the same policy as resolve_device()."""
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
        self._device = device

    def get(self):
        """Return the current DeviceHandle."""
        return self._device

    def set(self, device):
        """Update the held device."""
        self._device = device

    # Convenience: str(TheDevice) returns the device string so existing
    # code like ``f"Device: {TheDevice}"`` still works.
    def __str__(self):
        return str(self._device)

    def __repr__(self):
        return f"TheDevice({self._device!r})"

    # Delegate DeviceHandle properties for direct access
    @property
    def type(self):
        return self._device.type

    @property
    def index(self):
        return self._device.index

    def optimized(self):
        return self._device.optimized()


# The canonical device for this process.
TheDevice = _DeviceHolder(auto_device())
torch.set_default_device(str(TheDevice.get()))


def init_device(device=None):
    """Override the process-wide device.  None re-runs auto-detection."""
    TheDevice.set(auto_device() if device is None else DeviceHandle(str(device)))
    torch.set_default_device(str(TheDevice.get()))


def buffer(*size, **kwargs):
    """Allocate a zero tensor on TheDevice.  Accepts the same args as torch.zeros."""
    return torch.zeros(*size, **kwargs, device=TheDevice.get())


# ---------------------------------------------------------------------------
# Compilation backend management
# ---------------------------------------------------------------------------

_COMPILE_BACKENDS = ("inductor", "eager", "aot_eager")
_COMPILE_OFF      = frozenset({"none", "off", "false", "0", "no"})

def auto_compile_backend():
    """Select the compilation backend from BASICMODEL_COMPILE env var.

    Recognised values (case-insensitive):
      none / off / false / 0 / no  → skip compilation entirely (return "none")
      inductor                     → force torch.compile(backend='inductor')
      eager                        → force torch.compile(backend='eager')
      aot_eager                    → force torch.compile(backend='aot_eager')
      (empty / unset)              → auto: try inductor → eager → aot_eager
    """
    override = os.environ.get("BASICMODEL_COMPILE", "").strip().lower()
    if not override:
        return "auto"
    if override in _COMPILE_OFF:
        return "none"
    if override in _COMPILE_BACKENDS:
        return override
    raise ValueError(
        f"Unknown BASICMODEL_COMPILE value '{override}'. "
        f"Valid values: none, {', '.join(_COMPILE_BACKENDS)}"
    )

# The canonical compilation backend for this process.
TheCompileBackend = auto_compile_backend()


def init_compile_backend(backend=None):
    """Override the process-wide compilation backend.  None re-runs auto-detection."""
    global TheCompileBackend
    TheCompileBackend = auto_compile_backend() if backend is None else backend


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _inductor_space_prefixes():
    """Collect path prefixes that need quoting in inductor shell commands."""
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
    """Quote known path prefixes so shlex.split keeps them intact."""
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
    """Monkey-patch torch inductor to quote path prefixes containing spaces."""
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


def compile(model, verbose=True):
    """Try to torch.compile the model; return the (possibly compiled) model.

    Respects TheCompileBackend (set by BASICMODEL_COMPILE env var):
      "none" → skip compilation entirely.
      "auto" → try inductor → eager → aot_eager, return first success.
      <name> → try only that backend; fall back to uncompiled on failure.

    Patches inductor compile commands first when paths contain spaces.
    """
    def _msg(text):
        if verbose:
            print(text)

    if TheCompileBackend == "none":
        _msg("Model compilation skipped (BASICMODEL_COMPILE=none)")
        return model

    _patch_inductor_paths()

    backends = _COMPILE_BACKENDS if TheCompileBackend == "auto" else (TheCompileBackend,)
    for backend in backends:
        try:
            compiled = torch.compile(model, mode="max-autotune", backend=backend)
            _msg(f"Model compiled ({backend})")
            return compiled
        except Exception as e:
            _msg(f"Model compile failed ({backend}): {e}")

    _msg("Model compilation failed, running eager")
    return model


# ---------------------------------------------------------------------------
# Message sink
# ---------------------------------------------------------------------------

class Message():
    """Tiny callable wrapper so legacy code can swap out message sinks later.

    Always emits ``\\r\\n`` so output renders correctly both locally and when
    streamed over SSH with a pseudo-terminal (``ssh -t``).
    """
    def __call__(self, txt, newline="\r\n"):
        print(txt, end=newline)

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
    *default* is supplied — so missing configuration is surfaced
    immediately rather than propagating ``None`` through the system.
    """

    def __init__(self, path=None, defaults_path=None):
        self._data = {}
        self._sources = []
        self._requirements = []
        if defaults_path:
            self.load(defaults_path)
        if path:
            self.overlay(path)

    # --- Loading ---

    def load(self, path):
        """Parse an XML file, replacing current data."""
        self._data = self._parse_xml(path)
        self._sources = [path]

    def overlay(self, path):
        """Deep-merge another XML file on top of current data."""
        over = self._parse_xml(path)
        for section in over:
            if section not in self._data:
                self._data[section] = over[section]
            else:
                self._data[section] = self._deep_merge(
                    self._data[section], over[section])
        self._sources.append(path)

    def reload(self):
        """Re-parse all previously loaded sources in order."""
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
        """Dot-path setter for runtime overrides."""
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
        """Raw dict access for backward compatibility."""
        return self._data

    @property
    def model_kind(self):
        """Infer the configured model family from the current config data."""
        return self.infer_model_kind(self._data)

    # --- Requirements ---

    def require(self, check, description):
        """Register a config constraint.

        *check* is a callable ``check(cfg) → bool``.  Returning
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
        """nWhere + nWhen encoding overhead per vector."""
        return self.space("InputSpace", "nWhere") + self.space("InputSpace", "nWhen")

    @property
    def nObjects(self):
        """Total codebook vectors across all spaces."""
        total = 0
        for s in ("InputSpace", "PerceptualSpace", "ConceptualSpace",
                  "SymbolicSpace", "OutputSpace"):
            try:
                total += self.space(s, "nVectors")
            except KeyError:
                pass
        return total

    def encodingSize(self, nDim):
        """Full vector width: nDim + objectSize."""
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

    def space(self, space_name, key):
        """Lookup key in <SpaceName>, fall back to <architecture>.

        Space sections are top-level siblings of <architecture> in the XML.
        Equivalent to BasicModelFactory.get_space_param.

        Raises ``KeyError`` when the key is absent from both the
        space section and the architecture section.
        """
        # Try space-specific section first (top-level)
        if space_name in self._data:
            space = self._data[space_name]
            if isinstance(space, dict) and key in space:
                return space[key]
        # Fall back to architecture-level default
        arch = self.section("architecture")
        if key in arch:
            return arch[key]
        raise KeyError(
            f"Config key {key!r} not found in section {space_name!r} "
            f"or 'architecture'"
        )

    def nOutput(self, space_name):
        """Return raw nOutput for a space (0 = sentinel meaning 'same as nInput')."""
        return self.space(space_name, "nOutput")

    def nInput(self, space_name):
        """Return raw nInput for a space (0 = sentinel meaning 'derive from previous space')."""
        return self.space(space_name, "nInput")

    # --- Serialization ---

    def to_xml(self):
        """Serialize back to XML string."""
        import xml.etree.ElementTree as ET

        def _dict_to_elem(tag, value):
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
        tree = ET.parse(path)
        root = tree.getroot()
        cfg = {}
        for section in root:
            cfg[section.tag] = _parse_element(section)
        return cfg

    @staticmethod
    def infer_model_kind(cfg):
        """Infer model kind from a parsed config dict.

        Checks for top-level <mentalModel> section. Falls back to the
        legacy <type> tag in <architecture> for backward compatibility.
        """
        has_mental = "mentalModel" in cfg and isinstance(cfg["mentalModel"], dict)
        if has_mental:
            return "mental"
        arch = cfg.get("architecture", {})
        return str(arch.get("type", "basic") or "basic").strip().lower()

    @staticmethod
    def _deep_merge(base, overlay):
        """Recursively merge overlay into base (overlay wins on conflicts)."""
        merged = dict(base)
        for k, v in overlay.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = XMLConfig._deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged


# The canonical config for this process.
# Always load model.xml defaults so that Space constructors can read
# config keys even when create_from_config() hasn't been called yet.
_defaults_xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "model.xml")
TheXMLConfig = XMLConfig(defaults_path=_defaults_xml if os.path.exists(_defaults_xml) else None)


def init_config(path=None, defaults_path=None):
    """Load (or reload) TheXMLConfig from file(s)."""
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
