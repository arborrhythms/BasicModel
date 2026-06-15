"""Stage 1.E substrate refactor: explicit forward-dispatch depth.

Post-Stage-1.E contract (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md),
updated 2026-06-13 (conceptualMode enum -> symbolicOrder integer):

  * ``BasicModel`` reads ``<architecture><symbolicOrder>`` from XML at
    construction time and stores it on ``self.symbolicOrder`` (an int).

  * ``0`` = PARALLEL (the per-stage body via :meth:`_forward_per_stage`);
    ``>= 1`` = SERIAL / GRAMMATICAL (the per-word body via
    :meth:`_forward_body_per_word`). The substrate-level SERIAL /
    GRAMMATICAL collapse is the spec's design decision — at the substrate
    level serial is one mode; grammar dispatch is a chart / rule-catalog
    config, not a substrate mode. Values > 1 are plumbed but behave as 1.

  * Negative / non-integer values raise loudly at config load time (per
    the project's fail-loud rule).

  * ``BasicModel._forward_body`` dispatches based on ``self.symbolicOrder``
    directly — no longer indirectly via ``InputSpace._per_word_enabled``.
    The ``_per_word_enabled`` boolean is preserved for back-compat (set
    from the new knob during construction so existing readers — e.g. the
    per-word AR cursor in ``InputSpace.next_word`` — see the same value).

This file is the targeted TDD gate for Stage 1.E. It uses the same
``MM_xor_loopback.xml`` config as the sibling Stage 1.A / 1.C / 1.D /
1.F test files (cheap PS/CS/SS boot, isolates the dispatch behaviour).
"""

import os
import re
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from util import init_config, TheXMLConfig

_DATA_DIR = os.path.join(_PROJECT, 'data')
_GRAMMAR_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_NONGRAMMAR_CONFIG = os.path.join(_DATA_DIR, "MM_xor.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _write_config_with_order_override(base_config_path, override_order):
    """Materialize a temporary XML file that sets
    ``<symbolicOrder>override_order</symbolicOrder>`` on
    ``base_config_path``.

    Because ``BasicModel.from_config`` re-loads ``TheXMLConfig`` from
    disk (clobbering any in-memory ``set()`` we'd do), exercising the
    XML read path requires actually writing the override to a file.
    Strip any existing ``<symbolicOrder>`` element first so the overlay
    does not produce a list value, then inject inside ``<architecture>``
    (creating the block when missing).
    """
    with open(base_config_path, "r") as f:
        text = f.read()
    # Strip any pre-existing <symbolicOrder>...</symbolicOrder>
    text = re.sub(
        r"\s*<symbolicOrder>[^<]*</symbolicOrder>\s*\n",
        "\n", text)
    inject = f"<symbolicOrder>{override_order}</symbolicOrder>"
    if "<architecture>" in text:
        text = text.replace(
            "<architecture>", f"<architecture>\n    {inject}", 1)
    else:
        # Insert architecture block just after <model>
        text = re.sub(
            r"<model[^>]*>",
            lambda m: m.group(0) + f"\n  <architecture>{inject}</architecture>",
            text, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False,
        dir=os.path.dirname(base_config_path))
    tmp.write(text)
    tmp.close()
    return tmp.name


def _make_model(config_path, override_order=None):
    """Build a fresh BasicModel from ``config_path``. If
    ``override_order`` is set, write a sibling XML file with
    ``<architecture><symbolicOrder>override_order</symbolicOrder>``
    overlaid so the constructor reads it from disk.
    """
    if override_order is not None:
        config_path = _write_config_with_order_override(
            config_path, override_order)
    try:
        init_config(path=config_path, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            model, _ = Models.BasicModel.from_config(config_path)
        Models.TheData.load("xor")
        model.eval()
        return model
    finally:
        if override_order is not None:
            try:
                os.unlink(config_path)
            except OSError:
                pass


def _one_input(model):
    """Pull one batch off the input loader and prepInput it."""
    loader = model.inputSpace.data.data_loader(
        split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    return model.inputSpace.prepInput(inp_items)


class TestSymbolicOrderAttribute(unittest.TestCase):
    """``BasicModel`` exposes ``self.symbolicOrder`` as a non-negative
    integer after construction."""

    def test_symbolicOrder_attribute_exists(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertTrue(
            hasattr(model, "symbolicOrder"),
            "BasicModel must expose ``self.symbolicOrder`` after "
            "construction (Stage 1.E).")

    def test_symbolicOrder_is_an_int(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertIsInstance(
            model.symbolicOrder, int,
            "self.symbolicOrder must be an int "
            f"(got {type(model.symbolicOrder).__name__}).")

    def test_symbolicOrder_value_is_non_negative(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertGreaterEqual(
            model.symbolicOrder, 0,
            "self.symbolicOrder must be >= 0 "
            f"(got {model.symbolicOrder!r}).")


class TestSymbolicOrderXMLRead(unittest.TestCase):
    """Explicit XML override of ``<symbolicOrder>`` flows through to
    ``self.symbolicOrder``."""

    def test_serial_override_sticks(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=1)
        self.assertEqual(
            model.symbolicOrder, 1,
            "Explicit <symbolicOrder>1</symbolicOrder> must be read into "
            "self.symbolicOrder at construction.")

    def test_parallel_override_sticks(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=0)
        self.assertEqual(
            model.symbolicOrder, 0,
            "Explicit <symbolicOrder>0</symbolicOrder> must be read into "
            "self.symbolicOrder at construction.")


class TestSymbolicOrderInvalidRaisesLoud(unittest.TestCase):
    """Invalid ``<symbolicOrder>`` values raise loudly at config load
    (per the project memory's fail-loud rule)."""

    def test_negative_value_raises(self):
        with self.assertRaises(Exception) as cm:
            _make_model(_GRAMMAR_CONFIG, override_order=-1)
        msg = str(cm.exception)
        self.assertTrue(
            "symbolicOrder" in msg or "-1" in msg,
            "Invalid <symbolicOrder> exception must mention the knob "
            f"name or the bad value (got: {msg!r}).")

    def test_non_integer_raises(self):
        with self.assertRaises(Exception):
            _make_model(_GRAMMAR_CONFIG, override_order="not_an_int")


class TestForwardBodyDispatchesOnSymbolicOrder(unittest.TestCase):
    """The ``_forward_body`` dispatch is governed by
    ``self.symbolicOrder``, NOT by ``InputSpace._per_word_enabled``.
    Setting ``symbolicOrder`` flips which body fires."""

    def test_serial_mode_dispatches_to_per_word_body(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=1)
        x = _one_input(model)
        with mock.patch.object(
                model, "_forward_body_per_word",
                wraps=model._forward_body_per_word) as per_word_spy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with torch.no_grad():
                    model.forward(x)
        self.assertTrue(
            per_word_spy.called,
            "symbolicOrder>=1 must dispatch to _forward_body_per_word.")

    def test_parallel_mode_skips_per_word_body(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=0)
        x = _one_input(model)
        with mock.patch.object(
                model, "_forward_body_per_word",
                wraps=model._forward_body_per_word) as per_word_spy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with torch.no_grad():
                    model.forward(x)
        self.assertFalse(
            per_word_spy.called,
            "symbolicOrder=0 must NOT dispatch to _forward_body_per_word; "
            "the per-stage body owns the loop.")


class TestPerWordEnabledBackrefFromOrder(unittest.TestCase):
    """The legacy ``InputSpace._per_word_enabled`` boolean is preserved
    for back-compat (existing readers — the AR cursor in
    ``InputSpace.next_word`` and a handful of late-stage loops — must
    keep seeing it). After Stage 1.E it MUST mirror the new
    ``symbolicOrder`` knob (``>= 1`` => True; ``0`` => False)."""

    def test_serial_mode_implies_per_word_enabled_true(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=1)
        isp = model.inputSpace
        self.assertTrue(
            getattr(isp, "_per_word_enabled", False),
            "symbolicOrder>=1 must imply "
            "InputSpace._per_word_enabled=True for back-compat.")

    def test_parallel_mode_implies_per_word_enabled_false(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=0)
        isp = model.inputSpace
        self.assertFalse(
            getattr(isp, "_per_word_enabled", True),
            "symbolicOrder=0 must imply "
            "InputSpace._per_word_enabled=False for back-compat.")


class TestConfigsHaveExplicitOrder(unittest.TestCase):
    """The repo configs in ``data/*.xml`` set ``<symbolicOrder>``
    explicitly post-Stage-1.E."""

    def test_grammar_config_default_is_serial(self):
        """Grammar configs (the existing ``_per_word_enabled=True``
        configs) default to serial (``symbolicOrder >= 1``)."""
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertGreaterEqual(
            model.symbolicOrder, 1,
            f"{os.path.basename(_GRAMMAR_CONFIG)} (grammar config) must "
            f"have symbolicOrder>=1 (serial) "
            f"(got {model.symbolicOrder!r}).")


if __name__ == "__main__":
    unittest.main()
