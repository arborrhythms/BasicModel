"""Serial words use the shared operation layer and one sentence boundary.

The forward never invokes a separate ``symbolSpace.compose`` pass. Words
and closing rounds use the same chooser; the reverse boundary dispatch
still follows ``routerWireSerial``. The online-round fixture explicitly
forces an absolute reading, so it tests routing rather than learned parsing.
"""

import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_project = Path(__file__).resolve().parent.parent            # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import torch

from util import init_config, init_device
import Models
import Language

_DATA_DIR = str(_project / "data")
_GRAMMAR_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _write_config_with_overrides(base_config_path, symbolic_order=1,
                                 router_wire_serial=None, word_capacity=None):
    """Materialize a temp XML overlaying ``<symbolicOrder>`` and
    (optionally) ``<routerWireSerial>`` inside ``<architecture>``.

    ``BasicModel.from_config`` re-reads ``TheXMLConfig`` from disk, so the
    knobs must be written to a file (an in-memory ``set()`` is clobbered).
    Mirrors ``test_two_mode_dispatch._write_config_with_order_override``.
    """
    with open(base_config_path, "r") as f:
        text = f.read()
    text = re.sub(
        r"\s*<symbolicOrder>[^<]*</symbolicOrder>\s*\n", "\n", text)
    text = re.sub(
        r"\s*<routerWireSerial>[^<]*</routerWireSerial>\s*\n", "\n", text)
    inject = f"<symbolicOrder>{symbolic_order}</symbolicOrder>"
    if router_wire_serial is not None:
        inject += (
            f"\n    <routerWireSerial>{router_wire_serial}</routerWireSerial>")
    if word_capacity is not None:
        # Use word-major perception so the traversal capacity is independent
        # of this fixture's eight-slot conceptual workspace.
        inject += (f"\n    <serialObjectMeta>true</serialObjectMeta>"
                   f"\n    <serialWordCapacity>{word_capacity}</serialWordCapacity>")
        text = text.replace("<PartSpace>", "<PartSpace><synthesis>meronomy</synthesis>", 1)
        for section, element in (("InputSpace", "nOutput"), ("PartSpace", "nInput")):
            text = re.sub(
                rf"(<{section}>.*?<{element}>)\d+(</{element}>)",
                lambda match: match[1] + str(word_capacity * 8) + match[2],
                text, count=1, flags=re.S)
    if "<architecture>" in text:
        text = text.replace(
            "<architecture>", f"<architecture>\n    {inject}", 1)
    else:
        text = re.sub(
            r"<model[^>]*>",
            lambda m: m.group(0)
            + f"\n  <architecture>{inject}</architecture>",
            text, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


def _make_serial_model(router_wire_serial=None, word_capacity=None):
    """Build a serial-mode grammar model, optionally overriding
    ``<routerWireSerial>``."""
    init_device("cpu")
    cfg = _write_config_with_overrides(
        _GRAMMAR_CONFIG, symbolic_order=1,
        router_wire_serial=router_wire_serial, word_capacity=word_capacity)
    try:
        init_config(path=cfg, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = Models.BasicModel.from_config(cfg)
        Models.TheData.load("xor")
        model.eval()
        return model
    finally:
        try:
            os.unlink(cfg)
        except OSError:
            pass


def _one_input(model):
    loader = model.inputSpace.data.data_loader(
        split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    return model.inputSpace.prepInput(inp_items)


def _count_compose_calls(model):
    """Spy on ``symbolSpace.compose`` and run one forward; return the
    call count and the post-forward ``current_rules``."""
    ss = model.symbolSpace
    assert ss is not None, "serial grammar config must have a symbolSpace"
    orig = ss.compose
    state = {"n": 0}

    def _spy(*a, **k):
        state["n"] += 1
        return orig(*a, **k)

    ss.compose = _spy
    try:
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model.forward(x)
    finally:
        ss.compose = orig
    return state["n"], ss.current_rules


def _run_forward_spying_fold(model):
    """Count online/closing chooser calls and the single row-writing boundary."""
    from reading_fixtures import force_absolute_reading
    from unittest.mock import patch
    force_absolute_reading(model)
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    ss = model.symbolSpace
    stm = model.conceptualSpace.stm
    counts = {"compose": 0, "sweep": 0, "reduce": 0}
    orig_compose = ss.compose
    orig_sweep = model._commit_sentence
    orig_reduce = model.languageSpace.choose_operation

    def _compose_spy(*a, **k):
        counts["compose"] += 1
        return orig_compose(*a, **k)

    def _sweep_spy(*a, **k):
        counts["sweep"] += 1
        return orig_sweep(*a, **k)

    def _reduce_spy(*a, **k):
        counts["reduce"] += 1
        return orig_reduce(*a, **k)

    ss.compose = _compose_spy
    model._commit_sentence = _sweep_spy
    model.languageSpace.choose_operation = _reduce_spy
    try:
        # Sixteen words traverse an eight-slot workspace. This tests online
        # operation rounds as well as the fixed closing budget.
        x = model.inputSpace.prepInput([" ".join(["hello", "world"] * 8)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad(), patch("torch.while_loop", eager_while):
                model.forward(x)
    finally:
        ss.compose = orig_compose
        model._commit_sentence = orig_sweep
        model.languageSpace.choose_operation = orig_reduce

    post_depth = stm._depth
    single_S = getattr(model, "_stm_single_S", None)
    post_sweep_depth = getattr(model, "_stm_post_depth", None)
    return {
        "compose": counts["compose"],
        "sweep": counts["sweep"],
        "reduce": counts["reduce"],
        "capacity": int(stm.capacity),
        "post_depth_max": int(post_depth.max().item()),
        "single_S": single_S,
        "post_sweep_depth_max": (
            int(post_sweep_depth.max().item())
            if post_sweep_depth is not None else None),
    }


def test_default_router_wire_serial_is_both():
    """Default ``<routerWireSerial>`` resolves to ``both`` on the model."""
    model = _make_serial_model()
    assert getattr(model, "router_wire_serial", None) == "both", (
        "Task 4: BaseModel.__init__ must read <routerWireSerial> "
        "(default 'both') into self.router_wire_serial.")


@pytest.mark.parametrize('mode', ['both', 'per-word'])
def test_serial_words_use_the_shared_operation_layer(mode):
    model = _make_serial_model(router_wire_serial=mode, word_capacity=16)
    probe = _run_forward_spying_fold(model)
    assert probe['compose'] == 0  # no second independently selected parse
    assert probe['sweep'] == 1
    assert probe['reduce'] > 2 * probe['capacity']  # online rounds precede the closing
    assert probe['post_depth_max'] <= probe['capacity']
    depth = probe['post_sweep_depth_max']
    assert depth == 1 or depth < 0  # incomplete is explicit, never a claimed root
    assert torch.isfinite(probe['single_S']).all()


def test_router_wire_serial_boundary_no_serial_forward_compose():
    """Boundary routing does not add a second serial forward compose pass."""
    model = _make_serial_model(router_wire_serial="boundary")
    n_calls, _ = _count_compose_calls(model)
    assert n_calls == 0, (
        f"the forward boundary compose is not on the serial forward path "
        f"(it is a parallel/reverse fire) and the per-word compose fire is "
        f"deleted, so expect 0 serial-forward compose calls; got "
        f"{n_calls}.")


def test_router_wire_serial_off_no_serial_forward_compose():
    """``routerWireSerial='off'`` also fires ``compose`` ZERO times per
    serial forward — the same count every mode now yields on the serial
    forward, because no ``compose`` fire remains on that path (the per-word
    leg is deleted; the boundary leg is parallel/reverse-only)."""
    model = _make_serial_model(router_wire_serial="off")
    n_calls, _ = _count_compose_calls(model)
    assert n_calls == 0, (
        f"routerWireSerial='off' must yield 0 serial-forward compose "
        f"calls; got {n_calls}.")


def test_boundary_generate_gated_by_router_wire_serial():
    """The reverse-path boundary fire (``_chart_generate_from_stm`` ->
    ``symbolSpace.reverse``) is gated by ``<routerWireSerial>``:

      * ``both`` / ``boundary`` -> the boundary generate fires,
      * ``off`` / ``per-word``  -> the boundary generate is suppressed.

    Driven directly on the method (the reverse path's only generate site)
    so the gating is pinned independently of which forward path runs.
    """
    def _generate_fires(mode):
        model = _make_serial_model(router_wire_serial=mode)
        ss = model.symbolSpace
        # Populate STM so snapshot() is non-None (run one forward).
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model.forward(x)
        state = {"n": 0}
        orig = ss.reverse

        def _spy(*a, **k):
            state["n"] += 1
            return orig(*a, **k)

        ss.reverse = _spy
        try:
            model._chart_generate_from_stm()
        finally:
            ss.reverse = orig
        return state["n"]

    assert _generate_fires("both") >= 1, (
        "boundary generate must fire under routerWireSerial='both'")
    assert _generate_fires("boundary") >= 1, (
        "boundary generate must fire under routerWireSerial='boundary'")
    assert _generate_fires("off") == 0, (
        "boundary generate must be suppressed under routerWireSerial='off'")
    assert _generate_fires("per-word") == 0, (
        "boundary generate must be suppressed under "
        "routerWireSerial='per-word' (per-word leg only)")


def test_invalid_router_wire_serial_raises_loud():
    """An invalid ``<routerWireSerial>`` value raises loudly at config
    load (per the project's fail-loud rule)."""
    with pytest.raises(ValueError, match="routerWireSerial"):
        _make_serial_model(router_wire_serial="not_a_mode")
