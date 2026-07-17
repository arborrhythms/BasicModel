"""SurfaceSchema (T1-T5 universal templates) + absorb/emit marker
codification on GrammarLayer.

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md
("Absorb / Emit / Swap codification"): the surface-realization behaviour
of each operator is declared by one of five universal templates; surface
markers are learned, owned by the operator, bound from co-occurrence on
analysis (absorb) and replayed on synthesis (emit). Emit MUST use recorded
route metadata, never the lossy generate()=(parent,parent) inverse.
"""

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
    for name in ("not", "non", "exist"):
        sch = GRAMMAR_LAYER_CLASSES[name].surface_schema
        assert sch.template_id == "T1", (name, sch.template_id)
        assert sch.arity == 1


def test_copy_swap_use_elision_template():
    """copy / swap are the T5 BINARY_ELISION surface policies. They were
    parked in bin/Legacy.py (2026-07-17) — retired from the live symbolic
    grammar, kept as the absorb/emit elision primitives — so the schema is
    now asserted on the Legacy classes, not the live registry."""
    from Legacy import CopyLayer, SwapLayer
    assert CopyLayer.surface_schema.template_id == "T5"
    assert SwapLayer.surface_schema.template_id == "T5"


# -- absorb / emit marker codification (Task #4) ----------------------

def _conjunction_layer():
    from Language import GRAMMAR_LAYER_CLASSES
    return GRAMMAR_LAYER_CLASSES["conjunction"]()


def test_marker_binds_from_cooccurrence():
    """absorb binds a co-occurring surface marker to the operator. Binding
    is many-to-one (several surface markers -> one operator); the most
    co-occurring marker becomes the operator's canonical default."""
    layer = _conjunction_layer()
    # "and" co-occurs with conjunction often; "&" rarely. marker_id is the
    # absorbed sub-span's PS codebook identity.
    layer.absorb(left="X", right="and", marker_id=10, weight=3.0)
    layer.absorb(left="X", right="&", marker_id=20, weight=1.0)
    bound = layer.bound_markers()
    assert set(bound) == {10, 20}, bound
    # Many-to-one: both markers resolve to this operator; the heaviest is
    # the canonical operator -> default-marker used by emit.
    assert layer.canonical_marker() == 10


def test_emit_replays_marker():
    """emit (synthesis) replays the operator's bound marker at the schema
    position; absorb (analysis) is its inverse."""
    layer = _conjunction_layer()
    content = layer.absorb(left="X", right="and", marker_id=10)
    assert content == "X"           # content survives, marker consumed
    assert layer.emit() == 10       # the bound marker is replayed


def test_emit_uses_route_meta_not_lossy_generate():
    """emit MUST realize the marker from recorded route metadata, never
    from the lossy generate()=(parent, parent) inverse."""
    layer = _conjunction_layer()
    layer.absorb(left="X", right="and", marker_id=10)

    # generate() is the lossy pseudo-inverse; if emit routed through it
    # this would raise.
    def _boom(*a, **k):
        raise AssertionError("emit must not call the lossy generate()")
    layer.generate = _boom

    # Exact replay from route metadata (the route-recorded marker id).
    assert layer.emit(marker_id=77) == 77
    # Canonical default also comes from the learned binding, not generate.
    assert layer.emit() == 10
