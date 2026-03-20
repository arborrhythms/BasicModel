"""Shared utilities for the basicmodel project."""

import os
import tempfile
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


# ---------------------------------------------------------------------------
# Message sink
# ---------------------------------------------------------------------------

class Message():
    """Tiny callable wrapper so legacy code can swap out message sinks later."""
    def __call__(self, txt, newline="\n"):
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
TheXMLConfig = XMLConfig()


def init_config(path=None, defaults_path=None):
    """Load (or reload) TheXMLConfig from file(s)."""
    global TheXMLConfig
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
