"""Stage 1.E substrate refactor: explicit two-mode forward dispatch.

Post-Stage-1.E contract (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):

  * ``BasicModel`` reads ``<architecture><conceptualMode>`` from XML at
    construction time and stores it on ``self.conceptualMode``.

  * Valid values are ``"serial"`` (= GRAMMATICAL, the per-word body via
    :meth:`_forward_body_per_word`) and ``"parallel"`` (the per-stage
    body via :meth:`_forward_per_stage`'s body). The substrate-level
    SERIAL / GRAMMATICAL collapse is the spec's design decision — at the
    substrate level there is one ``"serial"`` mode; grammar dispatch is
    a chart / rule-catalog config, not a substrate mode.

  * Invalid values raise loudly at config load time (per the project's
    fail-loud rule).

  * ``BasicModel._forward_body`` dispatches based on
    ``self.conceptualMode`` directly — no longer indirectly via
    ``InputSpace._per_word_enabled``. The ``_per_word_enabled`` boolean
    is preserved for back-compat (it is set from the new knob during
    construction so existing readers — e.g. the per-word AR cursor in
    ``InputSpace.next_word`` — see the same value they used to).

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


def _write_config_with_mode_override(base_config_path, override_mode):
    """Materialize a temporary XML file that sets
    ``<conceptualMode>override_mode</conceptualMode>`` on
    ``base_config_path``.

    Because ``BasicModel.from_config`` re-loads ``TheXMLConfig`` from
    disk (clobbering any in-memory ``set()`` we'd do), exercising the
    XML read path requires actually writing the override to a file.
    Strip any existing ``<conceptualMode>`` element first so the
    overlay does not produce a list value, then inject inside
    ``<architecture>`` (creating the block when missing).
    """
    with open(base_config_path, "r") as f:
        text = f.read()
    # Strip any pre-existing <conceptualMode>...</conceptualMode>
    text = re.sub(
        r"\s*<conceptualMode>[^<]*</conceptualMode>\s*\n",
        "\n", text)
    inject = f"<conceptualMode>{override_mode}</conceptualMode>"
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


def _make_model(config_path, override_mode=None):
    """Build a fresh BasicModel from ``config_path``. If
    ``override_mode`` is set, write a sibling XML file with
    ``<architecture><conceptualMode>override_mode</conceptualMode>``
    overlaid so the constructor reads it from disk.
    """
    if override_mode is not None:
        config_path = _write_config_with_mode_override(
            config_path, override_mode)
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
        if override_mode is not None:
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


class TestConceptualModeAttribute(unittest.TestCase):
    """``BasicModel`` exposes ``self.conceptualMode`` as a string after
    construction, with valid values ``"serial"`` / ``"parallel"``."""

    def test_conceptualMode_attribute_exists(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertTrue(
            hasattr(model, "conceptualMode"),
            "BasicModel must expose ``self.conceptualMode`` after "
            "construction (Stage 1.E).")

    def test_conceptualMode_is_a_string(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertIsInstance(
            model.conceptualMode, str,
            "self.conceptualMode must be a string "
            f"(got {type(model.conceptualMode).__name__}).")

    def test_conceptualMode_value_is_valid(self):
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertIn(
            model.conceptualMode, ("serial", "parallel"),
            "self.conceptualMode must be one of 'serial' / 'parallel' "
            f"(got {model.conceptualMode!r}).")


class TestConceptualModeXMLRead(unittest.TestCase):
    """Explicit XML override of ``<conceptualMode>`` flows through to
    ``self.conceptualMode``."""

    def test_serial_override_sticks(self):
        model = _make_model(_GRAMMAR_CONFIG, override_mode="serial")
        self.assertEqual(
            model.conceptualMode, "serial",
            "Explicit <conceptualMode>serial</conceptualMode> must be "
            "read into self.conceptualMode at construction.")

    def test_parallel_override_sticks(self):
        model = _make_model(_GRAMMAR_CONFIG, override_mode="parallel")
        self.assertEqual(
            model.conceptualMode, "parallel",
            "Explicit <conceptualMode>parallel</conceptualMode> must be "
            "read into self.conceptualMode at construction.")


class TestConceptualModeInvalidRaisesLoud(unittest.TestCase):
    """Invalid ``<conceptualMode>`` values raise loudly at config load
    (per the project memory's fail-loud rule)."""

    def test_invalid_value_raises(self):
        with self.assertRaises(Exception) as cm:
            _make_model(_GRAMMAR_CONFIG, override_mode="invalid_mode")
        msg = str(cm.exception)
        self.assertTrue(
            "conceptualMode" in msg or "invalid_mode" in msg,
            "Invalid <conceptualMode> exception must mention the knob "
            f"name or the bad value (got: {msg!r}).")

    def test_empty_string_raises(self):
        with self.assertRaises(Exception):
            _make_model(_GRAMMAR_CONFIG, override_mode="")


class TestForwardBodyDispatchesOnConceptualMode(unittest.TestCase):
    """The ``_forward_body`` dispatch is governed by
    ``self.conceptualMode``, NOT by ``InputSpace._per_word_enabled``.
    Setting ``conceptualMode`` flips which body fires."""

    def test_serial_mode_dispatches_to_per_word_body(self):
        model = _make_model(_GRAMMAR_CONFIG, override_mode="serial")
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
            "conceptualMode='serial' must dispatch to "
            "_forward_body_per_word.")

    def test_parallel_mode_skips_per_word_body(self):
        model = _make_model(_GRAMMAR_CONFIG, override_mode="parallel")
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
            "conceptualMode='parallel' must NOT dispatch to "
            "_forward_body_per_word; the per-stage body owns the loop.")


class TestPerWordEnabledBackrefFromMode(unittest.TestCase):
    """The legacy ``InputSpace._per_word_enabled`` boolean is preserved
    for back-compat (existing readers — the AR cursor in
    ``InputSpace.next_word`` and a handful of late-stage loops — must
    keep seeing it). After Stage 1.E it MUST mirror the new
    ``conceptualMode`` knob (``"serial"`` => True; ``"parallel"`` =>
    False)."""

    def test_serial_mode_implies_per_word_enabled_true(self):
        model = _make_model(_GRAMMAR_CONFIG, override_mode="serial")
        isp = model.inputSpace
        self.assertTrue(
            getattr(isp, "_per_word_enabled", False),
            "conceptualMode='serial' must imply "
            "InputSpace._per_word_enabled=True for back-compat.")

    def test_parallel_mode_implies_per_word_enabled_false(self):
        model = _make_model(_GRAMMAR_CONFIG, override_mode="parallel")
        isp = model.inputSpace
        self.assertFalse(
            getattr(isp, "_per_word_enabled", True),
            "conceptualMode='parallel' must imply "
            "InputSpace._per_word_enabled=False for back-compat.")


class TestConfigsHaveExplicitMode(unittest.TestCase):
    """The repo configs in ``data/*.xml`` set ``<conceptualMode>``
    explicitly post-Stage-1.E."""

    def test_grammar_config_default_is_serial(self):
        """Grammar configs (the existing ``_per_word_enabled=True``
        configs) default to ``"serial"``."""
        model = _make_model(_GRAMMAR_CONFIG)
        self.assertEqual(
            model.conceptualMode, "serial",
            f"{os.path.basename(_GRAMMAR_CONFIG)} (grammar config) must "
            f"have conceptualMode='serial' "
            f"(got {model.conceptualMode!r}).")


if __name__ == "__main__":
    unittest.main()
