"""Stage 1.E substrate refactor: explicit forward-dispatch mode.

Post-Stage-1.E contract (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md),
updated 2026-06-22 (serial mode split from symbolicOrder):

  * ``BasicModel`` reads ``<architecture><serial>`` and
    ``<architecture><symbolicOrder>`` from XML at construction time.

  * ``serial`` selects traversal: ``False`` = PARALLEL (the per-stage body
    via :meth:`_forward_per_stage`); ``True`` = SERIAL / GRAMMATICAL (the
    per-word body via :meth:`_forward_body_per_word`).

  * ``symbolicOrder`` is a non-negative symbolic / relational loop budget.
    Values above 1 are accepted and must not by themselves force serial
    traversal when ``serial`` is explicit.

  * Negative / non-integer ``symbolicOrder`` values raise loudly at config
    load time (per the project's fail-loud rule).

  * Back-compat: if ``<serial>`` is omitted, serial mode derives from
    ``symbolicOrder > 0`` so existing configs keep their old behavior.
    The ``_per_word_enabled`` boolean is preserved for existing readers and
    mirrors ``self.serial``.

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


def _write_config_with_arch_overrides(
        base_config_path, override_order=None, serial=None):
    """Materialize a temporary XML file that overlays architecture knobs.

    Because ``BasicModel.from_config`` re-loads ``TheXMLConfig`` from
    disk (clobbering any in-memory ``set()`` we'd do), exercising the
    XML read path requires actually writing the override to a file. Strip
    any existing ``<serial>`` / ``<symbolicOrder>`` elements first so the
    overlay does not produce list values, then inject inside
    ``<architecture>`` (creating the block when missing).
    """
    with open(base_config_path, "r") as f:
        text = f.read()
    for tag in ("serial", "symbolicOrder"):
        text = re.sub(
            rf"\s*<{tag}>[^<]*</{tag}>\s*\n",
            "\n", text)
    entries = []
    if serial is not None:
        serial_text = "true" if serial else "false"
        entries.append(f"<serial>{serial_text}</serial>")
    if override_order is not None:
        entries.append(f"<symbolicOrder>{override_order}</symbolicOrder>")
    inject = "\n    ".join(entries)
    if "<architecture>" in text:
        text = text.replace(
            "<architecture>", f"<architecture>\n    {inject}", 1)
    else:
        # Insert architecture block just after <model>
        text = re.sub(
            r"<model[^>]*>",
            lambda m: (
                m.group(0) + f"\n  <architecture>{inject}</architecture>"),
            text, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False,
        dir=os.path.dirname(base_config_path))
    tmp.write(text)
    tmp.close()
    return tmp.name


def _make_model(config_path, override_order=None, serial=None):
    """Build a fresh BasicModel from ``config_path``. If
    ``override_order`` or ``serial`` is set, write a sibling XML file with
    those architecture overrides so the constructor reads them from disk.
    """
    wrote_tmp = override_order is not None or serial is not None
    if wrote_tmp:
        config_path = _write_config_with_arch_overrides(
            config_path, override_order=override_order, serial=serial)
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
        if wrote_tmp:
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

    def test_order_above_one_sticks(self):
        model = _make_model(
            _GRAMMAR_CONFIG, override_order=2, serial=False)
        self.assertEqual(
            model.symbolicOrder, 2,
            "Explicit <symbolicOrder>2</symbolicOrder> must be accepted as "
            "a symbolic loop budget.")


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


class TestSerialAttribute(unittest.TestCase):
    """``BasicModel.serial`` is the traversal mode switch."""

    def test_serial_attribute_exists(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertTrue(
            hasattr(model, "serial"),
            "BasicModel must expose ``self.serial`` after construction.")

    def test_serial_attribute_is_bool(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertIsInstance(
            model.serial, bool,
            f"self.serial must be a bool "
            f"(got {type(model.serial).__name__}).")


class TestForwardBodyDispatchesOnSerial(unittest.TestCase):
    """The ``_forward_body`` dispatch is governed by ``self.serial``."""

    def test_explicit_serial_true_dispatches_to_per_word_body(self):
        model = _make_model(
            _GRAMMAR_CONFIG, override_order=0, serial=True)
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
            "<serial>true</serial> must dispatch to "
            "_forward_body_per_word.")

    def test_explicit_serial_false_skips_per_word_body(self):
        model = _make_model(
            _GRAMMAR_CONFIG, override_order=2, serial=False)
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
            "<serial>false</serial> must NOT dispatch to "
            "_forward_body_per_word, even when symbolicOrder > 1.")


class TestPerWordEnabledBackrefFromSerial(unittest.TestCase):
    """The legacy ``InputSpace._per_word_enabled`` boolean is preserved
    for back-compat (existing readers — the AR cursor in
    ``InputSpace.next_word`` and a handful of late-stage loops — must
    keep seeing it). It mirrors ``self.serial``."""

    def test_serial_mode_implies_per_word_enabled_true(self):
        model = _make_model(
            _GRAMMAR_CONFIG, override_order=0, serial=True)
        isp = model.inputSpace
        self.assertTrue(
            getattr(isp, "_per_word_enabled", False),
            "serial=True must imply "
            "InputSpace._per_word_enabled=True for back-compat.")

    def test_parallel_mode_implies_per_word_enabled_false(self):
        model = _make_model(
            _GRAMMAR_CONFIG, override_order=2, serial=False)
        isp = model.inputSpace
        self.assertFalse(
            getattr(isp, "_per_word_enabled", True),
            "serial=False must imply "
            "InputSpace._per_word_enabled=False for back-compat.")


class TestLegacySerialDerivation(unittest.TestCase):
    """When ``<serial>`` is omitted, legacy ``symbolicOrder`` mode
    derivation is preserved."""

    def test_order_one_defaults_serial(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=1)
        self.assertTrue(
            model.serial,
            "With <serial> omitted, symbolicOrder=1 must preserve the "
            "legacy serial mode.")
        self.assertEqual(
            model.symbolicOrder, 1,
            "The symbolic order budget should still be stored as 1.")

    def test_order_zero_defaults_parallel(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=0)
        self.assertFalse(
            model.serial,
            "With <serial> omitted, symbolicOrder=0 must preserve the "
            "legacy parallel mode.")

    def test_order_two_defaults_serial_for_legacy_configs(self):
        model = _make_model(_GRAMMAR_CONFIG, override_order=2)
        self.assertTrue(
            model.serial,
            "With <serial> omitted, symbolicOrder > 0 must preserve the "
            "legacy serial derivation.")


class TestConfigsHaveExplicitOrder(unittest.TestCase):
    """The repo configs in ``data/*.xml`` still set ``<symbolicOrder>``
    explicitly post-Stage-1.E."""

    def test_grammar_config_default_derives_serial(self):
        """Grammar configs preserve serial behavior when ``<serial>`` is
        omitted."""
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertTrue(
            model.serial,
            f"{os.path.basename(_GRAMMAR_CONFIG)} (grammar config) must "
            f"derive serial=True from symbolicOrder "
            f"(got {model.serial!r}).")


if __name__ == "__main__":
    unittest.main()
