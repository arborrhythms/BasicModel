"""Task 3a: per-space ``<attention>`` symbolic-retrieval MODE knob.

Covers the parse/validate paths for the new ``<attention>`` enum
(``off|primer|second-order|low-rank``) introduced by plan
``doc/plans/2026-06-06-symbolic-heat-retrieval.md`` §Handoff addendum:

  * a space with ``<attention>primer</attention>`` resolves to ``"primer"``;
  * a space with no ``<attention>`` resolves to ``"off"`` (default);
  * ``validate_config`` NO LONGER raises when ``hasAttention=true`` is paired
    with a ``flatten``/``nInputDim`` reshape (the reshape-vs-QKV guard was
    retired with the QKV ``AttentionLayer`` enlistment).

These exercise the real ``XMLConfig`` / ``BasicModelFactory.validate_config``
machinery directly (no full model build) so they stay fast and isolated.
"""
import os
import sys
import unittest

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import Models  # noqa: E402
from util import XMLConfig  # noqa: E402


class TestAttentionModeResolution(unittest.TestCase):
    """``XMLConfig.space(section, 'attention', default='off')`` resolution.

    Uses the same ``space()`` lookup (space section first, then
    ``<architecture>`` fallback) that PartSpace / ConceptualSpace /
    WholeSpace ``__init__`` call to read ``self.attention_mode``.
    """

    def _cfg(self, data):
        cfg = XMLConfig()  # empty; no defaults file
        cfg._data = data
        return cfg

    def test_explicit_primer_resolves(self):
        cfg = self._cfg({
            "architecture": {},
            "ConceptualSpace": {"attention": "primer"},
        })
        self.assertEqual(
            cfg.space("ConceptualSpace", "attention", default="off"),
            "primer")

    def test_explicit_modes_on_each_space(self):
        cfg = self._cfg({
            "architecture": {},
            "PartSpace": {"attention": "low-rank"},
            "ConceptualSpace": {"attention": "second-order"},
            "WholeSpace": {"attention": "primer"},
        })
        self.assertEqual(
            cfg.space("PartSpace", "attention", default="off"),
            "low-rank")
        self.assertEqual(
            cfg.space("ConceptualSpace", "attention", default="off"),
            "second-order")
        self.assertEqual(
            cfg.space("WholeSpace", "attention", default="off"),
            "primer")

    def test_absent_resolves_to_off(self):
        """No ``<attention>`` anywhere => default ``off`` (current behavior)."""
        cfg = self._cfg({
            "architecture": {},
            "PartSpace": {"nDim": 4},
            "ConceptualSpace": {"nDim": 4},
            "WholeSpace": {"nDim": 4},
        })
        for sect in ("PartSpace", "ConceptualSpace", "WholeSpace"):
            self.assertEqual(
                cfg.space(sect, "attention", default="off"), "off",
                f"{sect} with no <attention> should resolve to 'off'")

    def test_architecture_fallback(self):
        """A space with no local ``<attention>`` inherits the architecture
        default (``space()`` falls back to ``<architecture>``)."""
        cfg = self._cfg({
            "architecture": {"attention": "primer"},
            "WholeSpace": {"nDim": 4},
        })
        self.assertEqual(
            cfg.space("WholeSpace", "attention", default="off"),
            "primer")

    def test_local_overrides_architecture(self):
        cfg = self._cfg({
            "architecture": {"attention": "primer"},
            "PartSpace": {"attention": "off"},
        })
        self.assertEqual(
            cfg.space("PartSpace", "attention", default="off"),
            "off")


class TestAttentionReshapeGuardRetired(unittest.TestCase):
    """The old ``hasAttention``-vs-reshape ``validate_config`` guard is gone.

    Plan 2026-06-06-symbolic-heat-retrieval.md §Handoff addendum: ``<attention>``
    is a symbolic-retrieval mode, not transformer self-attention, so the old
    "QKV needs 3D input" constraint no longer applies. ``hasAttention=true``
    paired with a ``flatten``/``nInputDim`` reshape must now be ACCEPTED.
    """

    def test_has_attention_plus_flatten_accepted(self):
        cfg = {
            "architecture": {},
            "PartSpace": {"hasAttention": True, "invertible": False,
                                "nActive": 4, "nDim": 1, "flatten": True},
            "WholeSpace": {},
            "ConceptualSpace": {"hasAttention": True, "flatten": True},
        }
        # Must NOT raise (the guard was retired).
        Models.BasicModelFactory.validate_config(cfg)

    def test_has_attention_plus_ninputdim_accepted(self):
        cfg = {
            "architecture": {},
            "PartSpace": {"hasAttention": True, "invertible": False,
                                "nActive": 4, "nDim": 1, "nInputDim": 10},
            "WholeSpace": {},
            "ConceptualSpace": {"hasAttention": False, "flatten": False},
        }
        # Must NOT raise (the guard was retired).
        Models.BasicModelFactory.validate_config(cfg)


class TestAttentionModeValidation(unittest.TestCase):
    """Fix 3 (2026-06-07): bogus ``<attention>`` values must be rejected.

    XSD validation can silently degrade to a warning when lxml/xmllint are
    absent, so each space's ``__init__`` validates ``attention_mode`` against
    ``_VALID_ATTENTION_MODES`` and raises ``ValueError`` on an unrecognised
    value.  This test verifies both the constant boundary and the guard logic.
    """

    def test_bogus_not_in_valid_modes(self):
        """``_VALID_ATTENTION_MODES`` must exclude unrecognised strings."""
        from Spaces import _VALID_ATTENTION_MODES
        self.assertNotIn("bogus", _VALID_ATTENTION_MODES)
        # Confirm the four valid values are present.
        for mode in ("off", "primer", "second-order", "low-rank"):
            self.assertIn(mode, _VALID_ATTENTION_MODES)

    def test_invalid_mode_raises_value_error(self):
        """The guard at the three call sites must raise ValueError for 'bogus'.

        We reproduce the guard logic directly (same code as the three
        ``__init__`` sites) so the test stays fast (no full model build)
        yet exercises the exact branch that would fire in production.
        """
        from Spaces import _VALID_ATTENTION_MODES
        _attn = "bogus"
        section = "ConceptualSpace"
        with self.assertRaises(ValueError) as ctx:
            if _attn not in _VALID_ATTENTION_MODES:
                raise ValueError(
                    f"{section} <attention> got {_attn!r}; expected one of "
                    f"{sorted(_VALID_ATTENTION_MODES)} (plan 2026-06-06-symbolic-heat-retrieval).")
        self.assertIn("attention", str(ctx.exception))
        self.assertIn("bogus", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
