"""Marker replay resolves PS ids to canonical surface (spec §9).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§9 "Carry-forward concerns": markers are surface text, but once SS learns
them they have PS identities (codebook ids). ``emit()`` returns a marker PS
codebook id, so reverse synthesis must resolve that id back to canonical
surface text -- never interpolate the opaque id as literal output.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


class _T2Schema:
    template_id = "T2"
    has_marker = True


class _OpEmits:
    """A stand-in operator whose emit() returns a fixed marker (PS id or
    surface string), for exercising synthesize_tree's marker placement."""
    surface_schema = _T2Schema()

    def __init__(self, marker):
        self._marker = marker

    def emit(self, **_kw):
        return self._marker


def test_marker_id_resolved_to_canonical_surface():
    """An int PS marker id is resolved to its canonical surface via the
    resolver, then placed at the infix position."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    tree = ("op", _OpEmits(5), ("leaf", "cat"), ("leaf", "dog"))
    out = an.synthesize_tree(tree, marker_resolver=lambda mid: {5: "and"}[mid])
    assert out == "cat and dog"


def test_opaque_marker_id_never_interpolated():
    """Without a resolver, an opaque PS marker id is NOT leaked into the
    surface; replay degrades to bare juxtaposition instead."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    tree = ("op", _OpEmits(5), ("leaf", "cat"), ("leaf", "dog"))
    out = an.synthesize_tree(tree)
    assert "5" not in out
    assert out == "cat dog"


def test_string_marker_path_unchanged():
    """A surface-string marker (the learned-co-occurrence path) still places
    directly -- back-compat with the analyze->emit E2E."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    tree = ("op", _OpEmits("and"), ("leaf", "cat"), ("leaf", "dog"))
    assert an.synthesize_tree(tree) == "cat and dog"
