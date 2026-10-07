"""SurfaceSchema templates retained by the live grammar operators."""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


# -- SurfaceSchema templates (Task #3) --------------------------------

def test_default_schema_is_bare_juxtapose():
    """The default base schema is T4 BINARY_JUXTAPOSE (bare concatenation,
    no marker) so any operator without a specific schema round-trips."""
    from Layers import GrammarLayer, T4_BINARY_JUXTAPOSE
    assert GrammarLayer.surface_schema is T4_BINARY_JUXTAPOSE
    assert GrammarLayer.surface_schema.template_id == "T4"
    assert GrammarLayer.surface_schema.name == "BINARY_JUXTAPOSE"
    assert GrammarLayer.surface_schema.has_marker is False


def test_conj_disj_isequal_share_one_template():
    """conjunction / disjunction / isEqual all use the single BINARY_INFIX
    (T2) template -- they are surface-indiscriminable, discriminated by the
    slot-0 operator vector, not by distinct surface schemas."""
    from Language import GRAMMAR_LAYER_CLASSES
    conj = GRAMMAR_LAYER_CLASSES["conjunction"]
    disj = GRAMMAR_LAYER_CLASSES["disjunction"]
    iseq = GRAMMAR_LAYER_CLASSES["isEqual"]
    assert conj.surface_schema is disj.surface_schema
    assert disj.surface_schema is iseq.surface_schema
    assert conj.surface_schema.template_id == "T2"
    assert conj.surface_schema.has_marker is True


def test_unary_ops_use_unary_affix_template():
    """not / non / exist are unary-affix (T1) operators."""
    from Language import GRAMMAR_LAYER_CLASSES
    for name in ("not", "non"):
        sch = GRAMMAR_LAYER_CLASSES[name].surface_schema
        assert sch.template_id == "T1", (name, sch.template_id)
        assert sch.arity == 1
