"""Per-word router firing on the serial forward path.

ORIGINAL TASK (doc/plans/2026-05-29-stm-serial-parallel-modes.md §4): the
serial per-word loop (``BasicModel._forward_body_per_word``) fired
``symbolicSpace.compose`` over the STM snapshot ONCE PER WORD, so the in-STM
predictor's conditioning context was repopulated mid-sentence.

SUPERSEDED by the tier-free bounded-STM grammar fold
(doc/plans/2026-06-05-tier-free-bounded-stm-fold.md, Phase 1 / Task 3):

  * **Task 3 DELETED the per-word ``symbolicSpace.compose(_snap)`` fire** in
    ``_forward_body_per_word`` (the $\\approx 89\\%$-cost full re-parse).
    The grammar fold no longer happens per word. The per-word loop now only
    ingests each word (PS->CS->SS + ``stm.push_step_masked``) and folds via
    two surviving primitives:
      - capacity back-pressure: ``_stm_bounded_reduce_step`` fires inside
        ``_per_word_body_step`` whenever the STM hits ``capacity``,
      - the sentence-end sweep: ``_stm_reduce_to_single_S`` collapses the
        accumulated STM to a single root idea S at the NULL seal.
  * **There is no grammar tier anymore.** The old S/C/P-tier compose loop
    is gone. ``<routerWireSerial>`` is consequently a **no-op for any
    per-word ``compose`` fire on the serial forward** — ``compose`` fires
    ZERO times per serial forward in EVERY mode (``both``, ``per-word``,
    ``off``, ``boundary``). ``compose`` now has only the *boundary*
    ``_chart_compose_at_C`` call site, which is NOT on the serial forward
    path (its forward caller is the parallel ``_forward_per_stage``); the
    serial boundary runs ``_stm_reduce_to_single_S`` instead.

This file remains the targeted gate. The two per-word tests below assert the
NEW contract: ``compose`` is not fired per word in serial mode, and the
bounded-STM fold (back-pressure + end sweep) still forms the root within
capacity. The remaining tests pin ``<routerWireSerial>`` gating where it is
still live — the *reverse-path* boundary fire (``_chart_generate_from_stm``
-> ``symbolicSpace.generate``), which still gates on ``{both, boundary}`` vs
``{off, per-word}`` (``test_boundary_generate_gated_by_router_wire_serial``).

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


def _write_config_with_overrides(base_config_path, symbolic_order=1,
                                 router_wire_serial=None):
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
        _GRAMMAR_CONFIG, symbolic_order=1,
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
    """Spy on ``symbolicSpace.compose`` and run one forward; return the
    call count and the post-forward ``current_rules``."""
    ss = model.symbolicSpace
    assert ss is not None, "serial grammar config must have a symbolicSpace"
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
    """Run one serial forward while spying on the bounded-STM fold
    primitives. Returns a dict with:

      * ``compose``       — ``symbolicSpace.compose`` call count,
      * ``sweep``         — ``_stm_reduce_to_single_S`` (sentence-end)
                            call count,
      * ``reduce``        — ``_stm_bounded_reduce_step`` (back-pressure +
                            sweep micro-step) call count,
      * ``capacity``      — the STM capacity (int),
      * ``post_depth_max``— max STM depth after the forward (host int),
      * ``single_S``      — the collapsed root idea tensor (or ``None``),
      * ``post_sweep_depth_max`` — max ``_stm_post_depth`` (or ``None``).
    """
    ss = model.symbolicSpace
    stm = model.conceptualSpace.stm
    counts = {"compose": 0, "sweep": 0, "reduce": 0}
    orig_compose = ss.compose
    orig_sweep = model._stm_reduce_to_single_S
    orig_reduce = model._stm_bounded_reduce_step

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
    model._stm_reduce_to_single_S = _sweep_spy
    model._stm_bounded_reduce_step = _reduce_spy
    try:
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model.forward(x)
    finally:
        ss.compose = orig_compose
        model._stm_reduce_to_single_S = orig_sweep
        model._stm_bounded_reduce_step = orig_reduce

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


def test_per_word_compose_is_not_fired_in_serial_mode():
    """NEW contract (tier-free bounded-STM fold, Task 3): the per-word
    ``symbolicSpace.compose`` fire is DELETED — there is no grammar tier and
    no per-word re-parse anymore.

    A serial forward in the default (``both``) mode therefore fires
    ``symbolicSpace.compose`` ZERO times: the only surviving ``compose`` call
    site (the boundary ``_chart_compose_at_C``) is not on the serial forward
    path. Instead the fold is carried by the bounded-STM primitives, which
    we assert below are exercised and collapse the STM to a single root S
    within capacity.
    """
    model = _make_serial_model(router_wire_serial="both")
    probe = _run_forward_spying_fold(model)

    # 1) The per-word re-parse is gone: compose is NOT fired per word
    #    (nor at all, on the serial forward).
    assert probe["compose"] == 0, (
        f"Task 3 deleted the per-word symbolicSpace.compose fire; a serial "
        f"forward must fire compose 0 times (the boundary _chart_compose_"
        f"at_C is not on the serial forward path). Got {probe['compose']}.")

    # 2) The bounded-STM fold still happens: the sentence-end sweep runs
    #    and the back-pressure / sweep reduce micro-step is exercised.
    assert probe["sweep"] >= 1, (
        f"the sentence-end sweep (_stm_reduce_to_single_S) must fire once "
        f"per serial forward to collapse the STM to root; got "
        f"{probe['sweep']}.")
    assert probe["reduce"] >= 1, (
        f"the bounded reduce micro-step (_stm_bounded_reduce_step) must be "
        f"exercised by the fold (back-pressure + sweep); got "
        f"{probe['reduce']}.")

    # 3) The STM stays within capacity, and the sweep collapses to a
    #    finite single root S (depth -> 1 for the absolute XOR sentence).
    assert probe["post_depth_max"] <= probe["capacity"], (
        f"bounded STM must stay within capacity {probe['capacity']}; got "
        f"depth {probe['post_depth_max']}.")
    assert probe["post_sweep_depth_max"] is not None and (
        probe["post_sweep_depth_max"] <= 1), (
        f"the sentence-end sweep must reduce the absolute sentence to a "
        f"single root (depth 1); got post-sweep depth "
        f"{probe['post_sweep_depth_max']}.")
    S = probe["single_S"]
    assert S is not None and torch.isfinite(S).all(), (
        f"the collapsed root idea S must be produced and finite; got {S!r}")


def test_router_wire_serial_per_word_is_noop_for_compose():
    """NEW contract: ``routerWireSerial='per-word'`` is now a NO-OP for the
    per-word ``compose`` fire — that fire was deleted in Task 3, so the
    ``per-word`` setting no longer has a per-word ``compose`` leg to gate.

    A serial forward under ``per-word`` therefore fires ``compose`` ZERO
    times (identical to ``both`` / ``off`` / ``boundary`` on the serial
    forward), while the bounded-STM fold still collapses the STM to root.
    This documents that ``<routerWireSerial>`` no longer affects the serial
    per-word path; its only remaining live effect is on the reverse-path
    boundary ``generate`` fire (see
    ``test_boundary_generate_gated_by_router_wire_serial``).
    """
    model = _make_serial_model(router_wire_serial="per-word")
    probe = _run_forward_spying_fold(model)
    assert probe["compose"] == 0, (
        f"routerWireSerial='per-word' has no per-word compose leg to fire "
        f"anymore (Task 3 deleted it); expected 0 compose calls, got "
        f"{probe['compose']}.")
    # The fold is unaffected by the knob: it still sweeps to root in cap.
    assert probe["sweep"] >= 1 and probe["reduce"] >= 1, (
        f"the bounded-STM fold must run regardless of routerWireSerial; "
        f"got sweep={probe['sweep']} reduce={probe['reduce']}.")
    assert probe["post_depth_max"] <= probe["capacity"], (
        f"STM must stay within capacity {probe['capacity']}; got "
        f"{probe['post_depth_max']}.")


def test_router_wire_serial_boundary_no_serial_forward_compose():
    """``routerWireSerial='boundary'`` fires ``compose`` ZERO times on a
    serial forward.

    NOTE on the serial forward path: the forward sentence-boundary
    ``compose`` (``_chart_compose_at_C``) is NOT on the serial per-word
    forward path — its only forward caller is ``_forward_per_stage``
    (the *parallel* per-stage body); the serial forward boundary instead
    runs ``_stm_reduce_to_single_S``. The per-word ``compose`` fire that
    used to live on the serial forward was DELETED in Task 3 (no grammar
    tier / no per-word re-parse anymore). So under ``boundary`` a serial
    forward fires ``compose`` ZERO times: there is no per-word leg, and the
    boundary leg has no serial-forward call site. (The boundary ``compose``
    / ``generate`` fires live on the parallel forward and the reverse path,
    e.g. ``_chart_generate_from_stm`` from ``reverse``.)
    """
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
    ``symbolicSpace.generate``) is gated by ``<routerWireSerial>``:

      * ``both`` / ``boundary`` -> the boundary generate fires,
      * ``off`` / ``per-word``  -> the boundary generate is suppressed.

    Driven directly on the method (the reverse path's only generate site)
    so the gating is pinned independently of which forward path runs.
    """
    def _generate_fires(mode):
        model = _make_serial_model(router_wire_serial=mode)
        ss = model.symbolicSpace
        # Populate STM so snapshot() is non-None (run one forward).
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model.forward(x)
        state = {"n": 0}
        orig = ss.generate

        def _spy(*a, **k):
            state["n"] += 1
            return orig(*a, **k)

        ss.generate = _spy
        try:
            model._chart_generate_from_stm()
        finally:
            ss.generate = orig
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
