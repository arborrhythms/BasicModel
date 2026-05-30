"""Task 4 (doc/plans/2026-05-29-stm-serial-parallel-modes.md §4):
per-word router firing.

The serial per-word loop (``BasicModel._forward_body_per_word``) must fire
``wordSubSpace.compose`` over the current STM snapshot ONCE PER WORD,
*before* the per-word ``cs.forward`` runs — so the in-STM predictor's
conditioning context (``wordSubSpace.current_rules``, the SS-dispatch rule
distribution) is repopulated mid-sentence rather than only at the sentence
boundary.

This file is the targeted gate for that wiring and doubles as the Task 6
verification ``test_router_fires_per_word.py``:

  * per-word fire fires > 1 time per forward in serial mode (boundary
    alone would be exactly 1; > 1 proves mid-sentence repopulation),
  * the ``<routerWireSerial>`` knob gates the per-word leg on the serial
    forward:
      - ``both`` (default): per-word leg fires      -> many compose calls,
      - ``per-word``: per-word leg fires            -> many compose calls,
      - ``boundary``: per-word leg OFF              -> 0 serial-forward
        compose calls (the forward boundary compose is a parallel/reverse
        fire, not on the serial forward path — see
        ``test_router_wire_serial_boundary_no_per_word_fire``),
      - ``off``: neither                            -> 0 compose calls,
  * the reverse-path boundary fire (``_chart_generate_from_stm`` ->
    ``wordSubSpace.generate``) is gated for ``{off, per-word}`` and fires
    for ``{both, boundary}``,
  * ``current_rules`` is non-trivially repopulated across the forward.

The capture-gate safety of the per-word fire is enforced separately by
``test/test_per_word_capture_gate.py`` (the fire lives in the host-side
loop, outside the captured per-iteration graph).

Harness mirrors ``test/test_two_mode_dispatch.py`` (the
``MM_xor_loopback.xml`` serial grammar config; cheap PS/CS/SS boot).
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


def _write_config_with_overrides(base_config_path, conceptual_mode="serial",
                                 router_wire_serial=None):
    """Materialize a temp XML overlaying ``<conceptualMode>`` and
    (optionally) ``<routerWireSerial>`` inside ``<architecture>``.

    ``BasicModel.from_config`` re-reads ``TheXMLConfig`` from disk, so the
    knobs must be written to a file (an in-memory ``set()`` is clobbered).
    Mirrors ``test_two_mode_dispatch._write_config_with_mode_override``.
    """
    with open(base_config_path, "r") as f:
        text = f.read()
    text = re.sub(
        r"\s*<conceptualMode>[^<]*</conceptualMode>\s*\n", "\n", text)
    text = re.sub(
        r"\s*<routerWireSerial>[^<]*</routerWireSerial>\s*\n", "\n", text)
    inject = f"<conceptualMode>{conceptual_mode}</conceptualMode>"
    if router_wire_serial is not None:
        inject += (
            f"\n    <routerWireSerial>{router_wire_serial}</routerWireSerial>")
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
        mode="w", suffix=".xml", delete=False,
        dir=os.path.dirname(base_config_path))
    tmp.write(text)
    tmp.close()
    return tmp.name


def _make_serial_model(router_wire_serial=None):
    """Build a serial-mode grammar model, optionally overriding
    ``<routerWireSerial>``."""
    init_device("cpu")
    cfg = _write_config_with_overrides(
        _GRAMMAR_CONFIG, conceptual_mode="serial",
        router_wire_serial=router_wire_serial)
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
    """Spy on ``wordSubSpace.compose`` and run one forward; return the
    call count and the post-forward ``current_rules``."""
    ws = model.wordSubSpace
    assert ws is not None, "serial grammar config must have a wordSubSpace"
    orig = ws.compose
    state = {"n": 0}

    def _spy(*a, **k):
        state["n"] += 1
        return orig(*a, **k)

    ws.compose = _spy
    try:
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model.forward(x)
    finally:
        ws.compose = orig
    return state["n"], ws.current_rules


def test_default_router_wire_serial_is_both():
    """Default ``<routerWireSerial>`` resolves to ``both`` on the model."""
    model = _make_serial_model()
    assert getattr(model, "router_wire_serial", None) == "both", (
        "Task 4: BaseModel.__init__ must read <routerWireSerial> "
        "(default 'both') into self.router_wire_serial.")


def test_router_fires_per_word_in_serial_mode():
    """The per-word router fire fires MORE THAN ONCE per forward in serial
    mode — the boundary fire alone is exactly one, so a count > 1 proves
    the router is re-fired mid-sentence (per word)."""
    model = _make_serial_model(router_wire_serial="both")
    n_calls, current_rules = _count_compose_calls(model)
    assert n_calls > 1, (
        f"per-word router fire must fire >1 time per forward in serial "
        f"mode (boundary alone = 1); got {n_calls}. The per-word "
        f"wordSubSpace.compose in _forward_body_per_word's loop is the "
        f"in-STM predictor's per-word conditioning context (Task 4 §4).")
    # current_rules must be a populated host-side rule dict (the SS
    # dispatch context this task delivers).
    assert isinstance(current_rules, dict) and len(current_rules) > 0, (
        f"per-word fire must populate wordSubSpace.current_rules; "
        f"got {current_rules!r}")


def test_router_wire_serial_per_word_fires_per_word():
    """``routerWireSerial='per-word'`` fires the per-word leg (> 1 call:
    one per active word; boundary leg disabled)."""
    model = _make_serial_model(router_wire_serial="per-word")
    n_calls, _ = _count_compose_calls(model)
    assert n_calls > 1, (
        f"routerWireSerial='per-word' must fire the per-word leg "
        f"(>1 call); got {n_calls}.")


def test_router_wire_serial_boundary_no_per_word_fire():
    """``routerWireSerial='boundary'`` disables the PER-WORD leg.

    NOTE on the serial forward path: the forward sentence-boundary
    ``compose`` (``_chart_compose_at_C``) is NOT on the serial per-word
    forward path — its only forward caller is ``_forward_per_stage``
    (the *parallel* per-stage body, Models.py:5098); the serial forward
    boundary instead runs ``_stm_reduce_to_single_S``. The boundary
    ``compose`` / ``generate`` fires live on the parallel forward and the
    reverse path (``_chart_generate_from_stm`` from ``_run_pipeline_rev``
    etc.). So under ``boundary`` mode a serial forward fires the router
    ZERO times via ``compose`` — the per-word leg (the only ``compose``
    fire on the serial forward) is gated off, and the boundary leg has no
    serial-forward call site. This documents that ``routerWireSerial``
    gates the per-word leg on the serial forward and the boundary leg on
    the parallel / reverse paths.
    """
    model = _make_serial_model(router_wire_serial="boundary")
    n_calls, _ = _count_compose_calls(model)
    assert n_calls == 0, (
        f"routerWireSerial='boundary' gates the per-word leg off; the "
        f"forward boundary compose is not on the serial forward path "
        f"(it is a parallel/reverse fire), so expect 0 serial-forward "
        f"compose calls; got {n_calls}.")


def test_router_wire_serial_off_disables_all_fires():
    """``routerWireSerial='off'`` disables BOTH the per-word and the
    boundary fire — zero ``compose`` calls per forward."""
    model = _make_serial_model(router_wire_serial="off")
    n_calls, _ = _count_compose_calls(model)
    assert n_calls == 0, (
        f"routerWireSerial='off' must disable all compose fires; "
        f"got {n_calls}.")


def test_boundary_generate_gated_by_router_wire_serial():
    """The reverse-path boundary fire (``_chart_generate_from_stm`` ->
    ``wordSubSpace.generate``) is gated by ``<routerWireSerial>``:

      * ``both`` / ``boundary`` -> the boundary generate fires,
      * ``off`` / ``per-word``  -> the boundary generate is suppressed.

    Driven directly on the method (the reverse path's only generate site)
    so the gating is pinned independently of which forward path runs.
    """
    def _generate_fires(mode):
        model = _make_serial_model(router_wire_serial=mode)
        ws = model.wordSubSpace
        # Populate STM so snapshot() is non-None (run one forward).
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model.forward(x)
        state = {"n": 0}
        orig = ws.generate

        def _spy(*a, **k):
            state["n"] += 1
            return orig(*a, **k)

        ws.generate = _spy
        try:
            model._chart_generate_from_stm()
        finally:
            ws.generate = orig
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
